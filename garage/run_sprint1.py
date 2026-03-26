"""
run_sprint1.py — GARAGE Sprint 1: FinanceBench end-to-end run.

Runs three stages in sequence:
  1. Baseline  — evaluate the default config on FinanceBench (establishes floor)
  2. BO        — Bayesian optimisation for N trials (default 50)
  3. Report    — print best config, save results/best_config.json, plot

Usage:
    .venv/bin/python run_sprint1.py
    .venv/bin/python run_sprint1.py --n-trials 100
    .venv/bin/python run_sprint1.py --skip-baseline   # if baseline already exists
    .venv/bin/python run_sprint1.py --study-name my_study

Output:
    results/best_config.json
    results/pareto_front.png
    results/scores_vs_run.png   (run plot_runs.py separately)
    Console + .log file via RunLogger
"""

from __future__ import annotations

import os
import torch  # noqa: F401 — must load before faiss on macOS

import argparse
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------

_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            k, v = _line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

if not os.environ.get("OPENAI_API_KEY"):
    sys.exit("ERROR: OPENAI_API_KEY not set. Copy .env.example → .env and fill it in.")

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import logging
logging.getLogger("httpx").setLevel(logging.WARNING)   # silence OpenAI SDK HTTP noise

import optuna
from optuna.samplers import NSGAIISampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

from pipeline import Document, parse_documents, chunk_documents, build_index, run, PipelineOutput, RAGIndex
from config import sample as sample_config, default_config
from eval import evaluate
from logger import RunLogger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

GARAGE_DIR      = Path(__file__).parent
DATA_DIR        = GARAGE_DIR / "data"
CORPUS_DIR      = DATA_DIR / "corpus"
GTS_PATH        = DATA_DIR / "gts.jsonl"
INDEX_CACHE_DIR = DATA_DIR / "index_cache"
INDEX_CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR     = GARAGE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

STORAGE = f"sqlite:///{DATA_DIR / 'results.db'}"

# ---------------------------------------------------------------------------
# Shared parse + index caches (same logic as bo_agent.py)
# ---------------------------------------------------------------------------

_parse_cache: dict[tuple, list] = {}
_index_cache: dict[str, RAGIndex] = {}


def get_or_build_index(config: dict) -> RAGIndex:
    txt_files = list(CORPUS_DIR.glob("*.txt"))
    pdf_files = sorted(CORPUS_DIR.glob("*.pdf"))

    if not txt_files and not pdf_files:
        sys.exit(f"ERROR: No documents in {CORPUS_DIR}. Run setup_financebench.py first.")

    docs: list[Document] = []
    for p in txt_files:
        docs.append(Document(text=p.read_text(errors="replace"), source=str(p), title=p.stem))

    parse_key = (config.get("parser", "pymupdf"),
                 config.get("table_extraction_strategy", "none"),
                 config.get("ocr_enabled", False))
    if parse_key not in _parse_cache:
        print(f"  [parse] {len(pdf_files)} PDFs  parser={parse_key[0]}  tables={parse_key[1]}")
        _parse_cache[parse_key] = parse_documents(pdf_files, config)
        print(f"  [parse] done → {len(_parse_cache[parse_key])} docs")
    docs.extend(_parse_cache[parse_key])

    chunks = chunk_documents(docs, config)
    index  = build_index(chunks, config, cache_dir=INDEX_CACHE_DIR)

    if index.config_hash not in _index_cache:
        _index_cache[index.config_hash] = index
        print(f"  [index] built  hash={index.config_hash}  chunks={len(chunks)}")
    else:
        print(f"  [index] reused hash={index.config_hash}")

    return _index_cache[index.config_hash]


# ---------------------------------------------------------------------------
# Stage 1: Baseline
# ---------------------------------------------------------------------------

def run_baseline(study_name: str) -> float:
    print("\n" + "=" * 60)
    print("  STAGE 1 — Baseline (default config)")
    print("=" * 60)

    cfg = default_config()
    log = RunLogger(tag=f"{study_name}_baseline")
    log.start(cfg, baseline_config=None)

    index = get_or_build_index(cfg)

    def pipeline_fn(query: str, c: dict) -> PipelineOutput:
        return run(query, c, index)

    result = evaluate(pipeline_fn, cfg, gts_path=GTS_PATH, max_workers=4, verbose=False)
    log.finish(result, config=cfg, baseline_config=None, notes="sprint1 baseline")

    print(f"\n  Baseline composite_score : {result.composite_score:.4f}")
    print(f"  Baseline latency_ms      : {result.latency_ms:.0f}")
    return result.composite_score


# ---------------------------------------------------------------------------
# Stage 2: Bayesian Optimisation
# ---------------------------------------------------------------------------

_baseline_cfg: dict | None = None


def run_bo(study_name: str, n_trials: int) -> optuna.Study:
    global _baseline_cfg
    _baseline_cfg = default_config()

    print("\n" + "=" * 60)
    print(f"  STAGE 2 — BO ({n_trials} trials, study='{study_name}')")
    print("=" * 60)

    study = optuna.create_study(
        study_name=study_name,
        storage=STORAGE,
        load_if_exists=True,
        directions=["maximize", "minimize"],
        sampler=NSGAIISampler(seed=42),
    )

    existing = len([t for t in study.trials if t.state.is_finished()])
    remaining = n_trials - existing
    if remaining <= 0:
        print(f"  Study already has {existing} finished trials — nothing to do.")
        return study

    print(f"  Existing finished: {existing}  |  Running: {remaining} more\n")

    from tqdm import tqdm

    pbar = tqdm(total=remaining, desc="  BO trials", unit="trial")

    def objective(trial: optuna.Trial):
        cfg = sample_config(trial)
        tag = f"{study_name}_trial_{trial.number:04d}"
        log = RunLogger(tag=tag)
        log.start(cfg, baseline_config=_baseline_cfg)

        tqdm.write(f"\n── Trial {trial.number:03d} │ "
                   f"parser={cfg['parser']}  chunk={cfg['chunk_strategy']}({cfg['chunk_size']})  "
                   f"embed={cfg['embedding_model'].split('-')[-1]}  "
                   f"retrieval={cfg['retrieval_mode']}  "
                   f"reranker={cfg['reranker']}  "
                   f"query={cfg['query_strategy']}")

        try:
            index = get_or_build_index(cfg)
        except Exception as e:
            tqdm.write(f"  [bo] trial {trial.number} index error: {e}")
            pbar.update(1)
            raise optuna.exceptions.TrialPruned()

        def pipeline_fn(query: str, c: dict) -> PipelineOutput:
            return run(query, c, index)

        try:
            result = evaluate(pipeline_fn, cfg, gts_path=GTS_PATH,
                              max_workers=4, verbose=False)
        except Exception as e:
            tqdm.write(f"  [bo] trial {trial.number} eval error: {e}")
            pbar.update(1)
            raise optuna.exceptions.TrialPruned()

        log.finish(result, config=cfg, baseline_config=_baseline_cfg,
                   notes=f"sprint1 bo trial {trial.number}")

        finished = [t for t in study.trials if t.state.is_finished() and t.values]
        best = max((t.values[0] for t in finished), default=0.0)
        tqdm.write(f"  → score={result.composite_score:.4f}  lat={result.latency_ms:.0f}ms  "
                   f"best={best:.4f}")
        pbar.set_postfix({"score": f"{result.composite_score:.4f}", "best": f"{best:.4f}"})
        pbar.update(1)

        return result.composite_score, result.latency_ms

    def _cb(_study, _trial):
        pass  # updates handled inside objective

    study.optimize(objective, n_trials=remaining, callbacks=[_cb])
    pbar.close()

    return study


# ---------------------------------------------------------------------------
# Stage 3: Report
# ---------------------------------------------------------------------------

def run_report(study: optuna.Study, baseline_score: float):
    print("\n" + "=" * 60)
    print("  STAGE 3 — Results")
    print("=" * 60)

    finished = [t for t in study.trials if t.state.is_finished() and t.values]
    if not finished:
        print("  No finished trials to report.")
        return

    best = max(finished, key=lambda t: t.values[0])
    best_score = best.values[0]
    best_latency = best.values[1]

    print(f"\n  Baseline score : {baseline_score:.4f}")
    print(f"  Best BO score  : {best_score:.4f}  (+{best_score - baseline_score:+.4f})")
    print(f"  Best latency   : {best_latency:.0f} ms")
    print(f"  Pareto front   : {len(study.best_trials)} trials")

    # Reconstruct full config (Optuna only stores sampled params)
    defaults = default_config()
    full_cfg: dict = dict(defaults)
    full_cfg.update(best.params)
    # Add hardcoded fields that aren't in Optuna params
    full_cfg["ocr_enabled"] = False
    full_cfg["query_prefix"] = "none"
    full_cfg["passage_prefix"] = "none"

    best_path = RESULTS_DIR / "best_config.json"
    with open(best_path, "w") as f:
        json.dump(full_cfg, f, indent=2)
    print(f"\n  Best config saved → {best_path}")

    # Pareto front plot
    pareto_path = RESULTS_DIR / "pareto_front.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        scores   = [t.values[0] for t in finished]
        latencies = [t.values[1] for t in finished]
        pareto_s = [t.values[0] for t in study.best_trials]
        pareto_l = [t.values[1] for t in study.best_trials]

        fig, ax = plt.subplots(figsize=(9, 6))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#0f1117")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("bottom", "left"):
            ax.spines[sp].set_color("#444")
        ax.tick_params(colors="#ccc")

        ax.scatter(latencies, scores, c="#4a90d9", s=40, alpha=0.5,
                   edgecolors="#222", linewidths=0.4, label="All trials")
        ax.scatter(pareto_l, pareto_s, c="#00e676", s=80, zorder=5,
                   edgecolors="#222", linewidths=0.5, label="Pareto front")
        ax.axhline(baseline_score, color="#f0a500", linewidth=1,
                   linestyle="--", alpha=0.8, label=f"Baseline ({baseline_score:.3f})")

        ax.set_xlabel("Latency (ms)", color="#ccc", fontsize=11)
        ax.set_ylabel("Composite Score", color="#ccc", fontsize=11)
        ax.set_title("GARAGE Sprint 1 — Pareto Front (FinanceBench)", color="#eee",
                     fontsize=13, pad=10)
        ax.legend(fontsize=9, facecolor="#1a1d27", edgecolor="#555", labelcolor="#ccc")
        fig.tight_layout()
        fig.savefig(str(pareto_path), dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Pareto plot    → {pareto_path}")
    except Exception as e:
        print(f"  [plot] skipped: {e}")

    print("\n" + "=" * 60)
    print("  Sprint 1 complete.")
    print("  Next: .venv/bin/python plot_runs.py   (per-run score chart)")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="GARAGE Sprint 1 — FinanceBench end-to-end run")
    p.add_argument("--n-trials",      type=int, default=50,
                   help="BO trials to run (default: 50)")
    p.add_argument("--study-name",    default="garage_financebench",
                   help="Optuna study name (default: garage_financebench)")
    p.add_argument("--skip-baseline", action="store_true",
                   help="Skip baseline eval (use 0.0 as floor for report)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    t0 = time.time()
    print(f"\nGARAGE Sprint 1  |  study={args.study_name}  |  n_trials={args.n_trials}")
    print(f"Corpus : {len(list(CORPUS_DIR.glob('*.pdf')))} PDFs in {CORPUS_DIR}")
    print(f"GTS    : {sum(1 for _ in open(GTS_PATH))} questions in {GTS_PATH}")

    baseline_score = 0.0
    if not args.skip_baseline:
        baseline_score = run_baseline(args.study_name)

    study = run_bo(args.study_name, args.n_trials)

    run_report(study, baseline_score)

    elapsed = time.time() - t0
    print(f"\n  Total wall time: {elapsed / 60:.1f} min")
