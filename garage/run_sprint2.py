"""
run_sprint2.py — GARAGE Sprint 2: Narrowed search space on FinanceBench.

Changes from Sprint 1:
  - Narrowed config space (~25 params, down from ~45):
      fixed: metric=ip, system_prompt_variant=variant_3
      dropped: bm25_only, hybrid_cc, unstructured, cross_encoder_minilm,
               html table extraction, numbered context format, IVF index
  - Default config is now Sprint 1 best (Trial 080, score=0.7629)
  - Separate study name: garage_financebench_s2

Usage:
    .venv/bin/python run_sprint2.py
    .venv/bin/python run_sprint2.py --n-trials 100
    .venv/bin/python run_sprint2.py --skip-baseline   # if already have s1 best as floor
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
logging.getLogger("httpx").setLevel(logging.WARNING)

import optuna
from optuna.samplers import NSGAIISampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

from pipeline import Document, parse_documents, chunk_documents, build_index, run, PipelineOutput, RAGIndex, index_cached, _config_hash as _pipeline_config_hash
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
PARSE_CACHE_DIR = DATA_DIR / "parse_cache"
PARSE_CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR     = GARAGE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

STORAGE = f"sqlite:///{DATA_DIR / 'results.db'}"
STUDY_NAME = "garage_financebench_s2"

# Sprint 1 best score — used as reference floor in report
S1_BEST = 0.7629

# ---------------------------------------------------------------------------
# Shared parse + index caches
# ---------------------------------------------------------------------------

_parse_cache: dict[tuple, list] = {}
_index_cache: dict[str, RAGIndex] = {}


def _parse_cache_path(parse_key: tuple) -> Path:
    name = f"{parse_key[0]}_tables-{parse_key[1]}_ocr-{parse_key[2]}.pkl"
    return PARSE_CACHE_DIR / name


def _load_parse_cache(parse_key: tuple) -> list | None:
    import pickle
    path = _parse_cache_path(parse_key)
    if path.exists():
        return pickle.loads(path.read_bytes())
    return None


def _save_parse_cache(parse_key: tuple, docs: list) -> None:
    import pickle
    _parse_cache_path(parse_key).write_bytes(pickle.dumps(docs))


def get_or_build_index(config: dict) -> RAGIndex:
    # Fast path 1: in-memory cache (check before any filesystem work)
    cfg_hash = _pipeline_config_hash(config)
    if cfg_hash in _index_cache:
        print(f"  [index] reused hash={cfg_hash}")
        return _index_cache[cfg_hash]

    # Fast path 2: disk cache present — skip parse + chunk entirely
    if index_cached(config, INDEX_CACHE_DIR):
        index = build_index([], config, cache_dir=INDEX_CACHE_DIR)
        _index_cache[cfg_hash] = index
        print(f"  [index] disk hit  hash={cfg_hash}  chunks={len(index.chunks)}")
        return _index_cache[cfg_hash]

    # Cache miss — parse, chunk, embed, build
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
        cached_docs = _load_parse_cache(parse_key)
        if cached_docs is not None:
            print(f"  [parse] disk hit  parser={parse_key[0]}  tables={parse_key[1]}")
            _parse_cache[parse_key] = cached_docs
        else:
            print(f"  [parse] {len(pdf_files)} PDFs  parser={parse_key[0]}  tables={parse_key[1]}")
            _parse_cache[parse_key] = parse_documents(pdf_files, config)
            _save_parse_cache(parse_key, _parse_cache[parse_key])
            print(f"  [parse] done → {len(_parse_cache[parse_key])} docs")
    docs.extend(_parse_cache[parse_key])

    chunks = chunk_documents(docs, config)
    index  = build_index(chunks, config, cache_dir=INDEX_CACHE_DIR)
    _index_cache[cfg_hash] = index
    print(f"  [index] built  hash={cfg_hash}  chunks={len(chunks)}")
    return _index_cache[cfg_hash]


# ---------------------------------------------------------------------------
# Stage 1: Baseline (sprint 1 best config)
# ---------------------------------------------------------------------------

def run_baseline(study_name: str) -> float:
    print("\n" + "=" * 60)
    print("  STAGE 1 — Baseline (sprint 1 best config)")
    print(f"  Reference: sprint 1 best = {S1_BEST:.4f}")
    print("=" * 60)

    cfg = default_config()
    log = RunLogger(tag=f"{study_name}_baseline")
    log.start(cfg, baseline_config=None)

    index = get_or_build_index(cfg)

    def pipeline_fn(query: str, c: dict) -> PipelineOutput:
        return run(query, c, index)

    result = evaluate(pipeline_fn, cfg, gts_path=GTS_PATH, max_workers=4, verbose=False)
    log.finish(result, config=cfg, baseline_config=None, notes="sprint2 baseline")

    print(f"\n  Baseline composite_score : {result.composite_score:.4f}")
    print(f"  Baseline latency_ms      : {result.latency_ms:.0f}")
    print(f"  vs Sprint 1 best         : {result.composite_score - S1_BEST:+.4f}")
    return result.composite_score


# ---------------------------------------------------------------------------
# Stage 2: Bayesian Optimisation
# ---------------------------------------------------------------------------

_baseline_cfg: dict | None = None
_best_score: float = 0.0


def run_bo(study_name: str, n_trials: int) -> optuna.Study:
    global _baseline_cfg, _best_score
    _baseline_cfg = default_config()
    _best_score = 0.0

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
        global _best_score
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
                   notes=f"sprint2 bo trial {trial.number}")

        _best_score = max(_best_score, result.composite_score)
        tqdm.write(f"  → score={result.composite_score:.4f}  lat={result.latency_ms:.0f}ms  "
                   f"best={_best_score:.4f}  s1_best={S1_BEST:.4f}")
        pbar.set_postfix({"score": f"{result.composite_score:.4f}", "best": f"{_best_score:.4f}"})
        pbar.update(1)

        return result.composite_score, result.latency_ms

    study.optimize(objective, n_trials=remaining)
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

    print(f"\n  Sprint 1 best  : {S1_BEST:.4f}")
    print(f"  Sprint 2 best  : {best_score:.4f}  ({best_score - S1_BEST:+.4f} vs s1)")
    print(f"  Best latency   : {best_latency:.0f} ms")
    print(f"  Pareto front   : {len(study.best_trials)} trials")

    defaults = default_config()
    full_cfg: dict = dict(defaults)
    full_cfg.update(best.params)
    full_cfg["ocr_enabled"] = False
    full_cfg["query_prefix"] = "none"
    full_cfg["passage_prefix"] = "none"
    full_cfg["metric"] = "ip"
    full_cfg["system_prompt_variant"] = "variant_3"

    best_path = RESULTS_DIR / "best_config_s2.json"
    with open(best_path, "w") as f:
        json.dump(full_cfg, f, indent=2)
    print(f"\n  Best config saved → {best_path}")

    # Pareto front plot
    pareto_path = RESULTS_DIR / "pareto_front_s2.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        scores    = [t.values[0] for t in finished]
        latencies = [t.values[1] for t in finished]
        pareto_s  = [t.values[0] for t in study.best_trials]
        pareto_l  = [t.values[1] for t in study.best_trials]

        fig, ax = plt.subplots(figsize=(9, 6))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#0f1117")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("bottom", "left"):
            ax.spines[sp].set_color("#444")
        ax.tick_params(colors="#ccc")

        ax.scatter(latencies, scores, c="#4a90d9", s=40, alpha=0.5,
                   edgecolors="#222", linewidths=0.4, label="Sprint 2 trials")
        ax.scatter(pareto_l, pareto_s, c="#00e676", s=80, zorder=5,
                   edgecolors="#222", linewidths=0.5, label="Pareto front")
        ax.axhline(S1_BEST, color="#f0a500", linewidth=1.5,
                   linestyle="--", alpha=0.9, label=f"Sprint 1 best ({S1_BEST:.4f})")
        ax.axhline(baseline_score, color="#ff7043", linewidth=1,
                   linestyle=":", alpha=0.8, label=f"S2 baseline ({baseline_score:.4f})")

        ax.set_xlabel("Latency (ms)", color="#ccc", fontsize=11)
        ax.set_ylabel("Composite Score", color="#ccc", fontsize=11)
        ax.set_title("GARAGE Sprint 2 — Pareto Front (FinanceBench)", color="#eee",
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
    print("  Sprint 2 complete.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="GARAGE Sprint 2")
    p.add_argument("--n-trials",      type=int, default=100,
                   help="BO trials to run (default: 100)")
    p.add_argument("--study-name",    default=STUDY_NAME,
                   help=f"Optuna study name (default: {STUDY_NAME})")
    p.add_argument("--skip-baseline", action="store_true",
                   help="Skip baseline eval (use sprint 1 best as floor)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    t0 = time.time()
    print(f"\nGARAGE Sprint 2  |  study={args.study_name}  |  n_trials={args.n_trials}")
    print(f"Corpus : {len(list(CORPUS_DIR.glob('*.pdf')))} PDFs in {CORPUS_DIR}")
    print(f"GTS    : {sum(1 for _ in open(GTS_PATH))} questions in {GTS_PATH}")
    print(f"Config : narrowed search space (~25 params, fixed metric=ip + variant_3)")

    baseline_score = S1_BEST  # default floor if skipping baseline
    if not args.skip_baseline:
        baseline_score = run_baseline(args.study_name)

    study = run_bo(args.study_name, args.n_trials)

    run_report(study, baseline_score)

    elapsed = time.time() - t0
    print(f"\n  Total wall time: {elapsed / 60:.1f} min")
