"""
bo_agent.py — GARAGE Bayesian Optimisation Agent (FROZEN — never modified by agents)

Multi-objective Optuna optimisation over the RAG pipeline parameter space.
Objectives:
  - Maximise  composite_score
  - Minimise  latency_ms

Algorithm: NSGAIISampler (Pareto-optimal multi-objective BO)
Falls back to TPESampler on single-objective mode (--single-obj).

Usage:
    python bo_agent.py --n-trials 80
    python bo_agent.py --n-trials 40 --single-obj   # maximise composite only
    python bo_agent.py --n-trials 80 --study-name my_study --storage sqlite:///data/results.db

What it does each trial:
  1. Sample config via config.sample(trial)
  2. Build / reuse RAGIndex (cached by config_hash — avoids re-embedding)
  3. Run eval.evaluate() → EvalResult
  4. Log via RunLogger (console + .log file + results.db)
  5. Return (composite_score, latency_ms) to Optuna

After all trials:
  - Save best config (highest composite_score on Pareto front) → results/best_config.json
  - Plot Pareto front → results/pareto_front.png
"""

from __future__ import annotations

import os
# torch MUST load before faiss (libomp conflict on macOS)
import torch  # noqa: F401

import argparse
import json
import sys
import time
from pathlib import Path
from dataclasses import asdict
from typing import Optional

# Load .env if present
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            k, v = _line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

if not os.environ.get("OPENAI_API_KEY"):
    sys.exit("ERROR: OPENAI_API_KEY not set. Copy .env.example → .env and fill it in.")

import logging
logging.getLogger("httpx").setLevel(logging.WARNING)   # silence OpenAI SDK HTTP noise

import optuna
from optuna.samplers import NSGAIISampler, TPESampler

from pipeline import Document, chunk_documents, build_index, run, PipelineOutput, RAGIndex
from config import sample as sample_config, default_config
from eval import evaluate, GTS_PATH
from logger import RunLogger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR       = Path(__file__).parent / "data"
CORPUS_DIR     = DATA_DIR / "corpus"
INDEX_CACHE_DIR = DATA_DIR / "index_cache"
INDEX_CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

BEST_CONFIG_PATH  = RESULTS_DIR / "best_config.json"
PARETO_FRONT_PATH = RESULTS_DIR / "pareto_front.png"


# ---------------------------------------------------------------------------
# Index cache (avoids re-embedding on every trial)
# ---------------------------------------------------------------------------

_index_cache: dict[str, RAGIndex] = {}

# PDF parse cache: (parser, table_strategy) → list[Document]
# PDFs are parsed once per unique parser config and reused across trials.
_parse_cache: dict[tuple, list] = {}


def get_or_build_index(config: dict) -> RAGIndex:
    """Return a cached RAGIndex if config_hash matches, else build and cache.

    PDF parsing is cached by (parser, table_extraction_strategy) so we don't
    re-parse all 30 PDFs on every trial — only when the parser config changes.
    """
    from pipeline import parse_documents

    txt_files = list(CORPUS_DIR.glob("*.txt"))
    pdf_files = sorted(CORPUS_DIR.glob("*.pdf"))
    all_files  = txt_files + pdf_files

    if not all_files:
        sys.exit(f"ERROR: No documents found in {CORPUS_DIR}")

    # --- txt docs (cheap, always re-read) ---
    docs: list[Document] = []
    for path in txt_files:
        text = path.read_text(errors="replace")
        docs.append(Document(text=text, source=str(path), title=path.stem))

    # --- PDF docs (cached by parser config) ---
    parse_key = (config.get("parser", "pymupdf"),
                 config.get("table_extraction_strategy", "none"),
                 config.get("ocr_enabled", False))
    if parse_key not in _parse_cache:
        print(f"[bo] parsing {len(pdf_files)} PDFs  parser={parse_key[0]}  tables={parse_key[1]}")
        pdf_docs = parse_documents(pdf_files, config)
        _parse_cache[parse_key] = pdf_docs
        print(f"[bo] parsed → {len(pdf_docs)} docs cached under key={parse_key}")
    else:
        print(f"[bo] reuse parsed docs  key={parse_key}")
    docs.extend(_parse_cache[parse_key])

    chunks = chunk_documents(docs, config)
    index  = build_index(chunks, config, cache_dir=INDEX_CACHE_DIR)

    key = index.config_hash
    if key not in _index_cache:
        _index_cache[key] = index
        print(f"[bo] new index  hash={key}  chunks={len(chunks)}")
    else:
        print(f"[bo] reuse index hash={key}")

    return _index_cache[key]


# ---------------------------------------------------------------------------
# Baseline config (used for diff logging on first trial)
# ---------------------------------------------------------------------------

_baseline_config: Optional[dict] = None


def _get_baseline() -> dict:
    global _baseline_config
    if _baseline_config is None:
        _baseline_config = default_config()
    return _baseline_config


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def make_objective(gts_path: Path, single_obj: bool = False):
    """Factory: returns the Optuna objective function."""

    def objective(trial: optuna.Trial):
        cfg = sample_config(trial)

        tag = f"trial_{trial.number:04d}"
        log = RunLogger(tag=tag)
        log.start(cfg, baseline_config=_get_baseline())

        # Build / reuse index
        try:
            index = get_or_build_index(cfg)
        except Exception as e:
            print(f"[bo] trial {trial.number} index error: {e}")
            # Prune trial — signals Optuna to skip
            raise optuna.exceptions.TrialPruned()

        def pipeline_fn(query: str, c: dict) -> PipelineOutput:
            return run(query, c, index)

        # Evaluate
        try:
            result = evaluate(
                pipeline_fn, cfg, gts_path=gts_path,
                max_workers=4, verbose=False,
            )
        except Exception as e:
            print(f"[bo] trial {trial.number} eval error: {e}")
            raise optuna.exceptions.TrialPruned()

        log.finish(result, config=cfg, baseline_config=_get_baseline(),
                   notes=f"bo trial {trial.number}")

        if single_obj:
            return result.composite_score
        else:
            return result.composite_score, result.latency_ms

    return objective


# ---------------------------------------------------------------------------
# Post-study analysis
# ---------------------------------------------------------------------------

def save_best_config(study: optuna.Study, single_obj: bool) -> dict:
    """Pick best trial (highest composite_score) and save to JSON."""
    if single_obj:
        best = max(study.best_trials, key=lambda t: t.values[0])
    else:
        # Pareto front: pick trial with highest composite_score
        best = max(
            study.best_trials,
            key=lambda t: t.values[0],   # composite_score is index 0
        )

    cfg = best.params
    # Reconstruct conditional None fields from default (Optuna only stores
    # the params that were actually sampled)
    defaults = default_config()
    full_cfg: dict = {}
    for k, v in defaults.items():
        full_cfg[k] = cfg.get(k, v if v is None else v)
    # Override with sampled values
    full_cfg.update(cfg)

    BEST_CONFIG_PATH.write_text(json.dumps(full_cfg, indent=2, default=str))
    print(f"\n[bo] best config saved → {BEST_CONFIG_PATH}")
    print(f"     composite_score = {best.values[0]:.4f}")
    if not single_obj:
        print(f"     latency_ms      = {best.values[1]:.1f}")
    return full_cfg


def plot_pareto_front(study: optuna.Study) -> None:
    """Plot composite_score vs latency_ms for all + Pareto-front trials."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[bo] matplotlib not available — skipping Pareto plot")
        return

    trials = [t for t in study.trials
              if t.state == optuna.trial.TrialState.COMPLETE]
    if not trials:
        return

    scores   = [t.values[0] for t in trials]
    latencies = [t.values[1] for t in trials]

    # Pareto front trials
    pareto_trials = study.best_trials
    p_scores   = [t.values[0] for t in pareto_trials]
    p_latencies = [t.values[1] for t in pareto_trials]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(latencies, scores, alpha=0.4, s=20, label="All trials", color="steelblue")
    ax.scatter(p_latencies, p_scores, alpha=0.9, s=60, label="Pareto front",
               color="tomato", zorder=5)

    # Sort Pareto front for line
    paired = sorted(zip(p_latencies, p_scores))
    ax.plot([x for x, _ in paired], [y for _, y in paired],
            color="tomato", linewidth=1.5, linestyle="--")

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Composite Score")
    ax.set_title(f"GARAGE Pareto Front  ({len(trials)} trials)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(PARETO_FRONT_PATH), dpi=150)
    plt.close(fig)
    print(f"[bo] Pareto front plot saved → {PARETO_FRONT_PATH}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="GARAGE BO Agent")
    ap.add_argument("--n-trials",    type=int, default=50,
                    help="Number of Optuna trials (default: 50)")
    ap.add_argument("--study-name",  type=str, default="garage_mo",
                    help="Optuna study name (default: garage_mo)")
    ap.add_argument("--storage",     type=str,
                    default=f"sqlite:///{DATA_DIR / 'results.db'}",
                    help="Optuna storage URL (default: sqlite:///data/results.db)")
    ap.add_argument("--gts",         type=str, default=str(GTS_PATH),
                    help="Path to gts.jsonl")
    ap.add_argument("--single-obj",  action="store_true",
                    help="Optimise composite_score only (TPE, not NSGA-II)")
    ap.add_argument("--seed",        type=int, default=42,
                    help="Random seed (default: 42)")
    ap.add_argument("--no-plot",     action="store_true",
                    help="Skip Pareto front plot after study")
    args = ap.parse_args()

    gts_path = Path(args.gts)
    if not gts_path.exists():
        sys.exit(f"ERROR: GTS not found: {gts_path}")

    # ------------------------------------------------------------------
    # Create / resume Optuna study
    # ------------------------------------------------------------------
    if args.single_obj:
        sampler = TPESampler(seed=args.seed)
        directions = ["maximize"]
        print(f"[bo] single-objective mode (TPE), maximising composite_score")
    else:
        sampler = NSGAIISampler(seed=args.seed)
        directions = ["maximize", "minimize"]
        print(f"[bo] multi-objective mode (NSGA-II): ↑ composite_score, ↓ latency_ms")

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        directions=directions,
        sampler=sampler,
        load_if_exists=True,
    )

    n_existing = len(study.trials)
    print(f"[bo] study '{args.study_name}'  existing trials={n_existing}  "
          f"running {args.n_trials} more")

    objective = make_objective(gts_path, single_obj=args.single_obj)

    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            catch=(Exception,),
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n[bo] interrupted — saving partial results")

    # ------------------------------------------------------------------
    # Post-study analysis
    # ------------------------------------------------------------------
    complete = [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"\n[bo] complete trials: {len(complete)} / {len(study.trials)}")

    if complete:
        save_best_config(study, single_obj=args.single_obj)
        if not args.single_obj and not args.no_plot:
            plot_pareto_front(study)

    print("[bo] done.")


if __name__ == "__main__":
    main()
