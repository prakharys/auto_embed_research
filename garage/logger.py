"""
logger.py — GARAGE Run Logger

Logs every run to:
  - Console (structured, human-readable)
  - results/runs/<timestamp>_<tag>.log  (full text log)
  - data/results.db  (SQLite, queryable)

Usage:
    from logger import RunLogger
    log = RunLogger(tag="bo_trial_42")
    log.start(config)
    # ... run pipeline + eval ...
    log.finish(eval_result, baseline_config=prev_config)
"""

from __future__ import annotations

import json
import os
import sqlite3
import textwrap
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).parent / "results"
RUNS_DIR    = RESULTS_DIR / "runs"
DB_PATH     = Path(__file__).parent / "data" / "results.db"

RUNS_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT UNIQUE NOT NULL,
    tag             TEXT,
    started_at      TEXT,
    finished_at     TEXT,
    duration_s      REAL,

    -- Eval metrics
    composite_score     REAL,
    retrieval_relevance REAL,
    answer_relevance    REAL,
    groundedness        REAL,
    bertscore_f1        REAL,
    latency_ms          REAL,
    n_items             INTEGER,

    -- Full config as JSON
    config_json     TEXT,
    -- Diff vs baseline as JSON (null for first run)
    config_diff_json TEXT,

    notes           TEXT
);

CREATE TABLE IF NOT EXISTS run_per_item (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    item_idx    INTEGER,
    query       TEXT,
    answer      TEXT,
    retrieval_relevance REAL,
    answer_relevance    REAL,
    groundedness        REAL,
    bertscore_f1        REAL,
    composite_score     REAL,
    latency_ms          REAL
);
"""

def _get_db(path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Diff helpers
# ---------------------------------------------------------------------------

def diff_configs(new: dict, baseline: dict) -> dict:
    """Return only the keys where new != baseline."""
    changed = {}
    all_keys = set(new) | set(baseline)
    for k in sorted(all_keys):
        v_new = new.get(k)
        v_base = baseline.get(k)
        if v_new != v_base:
            changed[k] = {"from": v_base, "to": v_new}
    return changed


# ---------------------------------------------------------------------------
# RunLogger
# ---------------------------------------------------------------------------

class RunLogger:
    def __init__(self, tag: str = "run"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id    = f"{ts}_{tag}"
        self.tag       = tag
        self.log_path  = RUNS_DIR / f"{self.run_id}.log"
        self._start_ts: float = 0.0
        self._lines: list[str] = []

    # ------------------------------------------------------------------
    # Internal write
    # ------------------------------------------------------------------

    def _write(self, line: str = "") -> None:
        print(line)
        self._lines.append(line)

    def _section(self, title: str) -> None:
        self._write()
        self._write("=" * 60)
        self._write(f"  {title}")
        self._write("=" * 60)

    def _flush(self) -> None:
        self.log_path.write_text("\n".join(self._lines) + "\n")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, config: dict, baseline_config: dict | None = None) -> None:
        """Call at the start of a run. Logs config and diff."""
        self._start_ts = time.perf_counter()

        self._section(f"GARAGE RUN  [{self.run_id}]")
        self._write(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._write(f"  Log file: {self.log_path}")

        # --- Full config ---
        self._section("CONFIG  (all parameters)")
        _print_config(config, self._write)

        # --- Diff vs baseline ---
        if baseline_config is not None:
            diff = diff_configs(config, baseline_config)
            self._section(f"CHANGES VS BASELINE  ({len(diff)} parameter(s) changed)")
            if diff:
                for k, v in diff.items():
                    self._write(f"  {k:<35}  {str(v['from']):<20}  →  {v['to']}")
            else:
                self._write("  (no changes — this is the baseline)")
        else:
            self._section("BASELINE RUN  (no previous config to diff against)")

        self._flush()

    def finish(
        self,
        eval_result,                          # EvalResult dataclass
        baseline_config: dict | None = None,
        config: dict | None = None,
        notes: str = "",
    ) -> None:
        """Call after eval completes. Logs metrics, saves to DB."""
        duration = time.perf_counter() - self._start_ts

        self._section("EVAL RESULTS")
        d = asdict(eval_result)
        self._write(f"  {'composite_score':<28} {d['composite_score']:.4f}")
        self._write(f"  {'retrieval_relevance':<28} {d['retrieval_relevance']:.4f}")
        self._write(f"  {'answer_relevance':<28} {d['answer_relevance']:.4f}")
        self._write(f"  {'groundedness':<28} {d['groundedness']:.4f}")
        self._write(f"  {'bertscore_f1':<28} {d['bertscore_f1']:.4f}")
        self._write(f"  {'latency_ms (avg)':<28} {d['latency_ms']:.1f}")
        self._write(f"  {'n_items':<28} {d['n_items']}")
        self._write(f"  {'total_duration_s':<28} {duration:.1f}")
        if notes:
            self._write(f"  {'notes':<28} {notes}")

        self._section("PER-QUERY BREAKDOWN")
        for i, item in enumerate(d["per_item"]):
            q = item["query"][:70]
            self._write(f"  [{i+1}] {q}")
            self._write(
                f"       composite={item['composite_score']:.3f}  "
                f"ret={item['retrieval_relevance']:.2f}  "
                f"ans={item['answer_relevance']:.2f}  "
                f"grnd={item['groundedness']:.2f}  "
                f"bert={item['bertscore_f1']:.2f}  "
                f"lat={item['latency_ms']:.0f}ms"
            )
            # Show the generated answer (truncated)
            ans = textwrap.shorten(item["answer"], width=120, placeholder="...")
            self._write(f"       answer: {ans}")

        self._write()
        self._write(f"  Log saved → {self.log_path}")
        self._flush()

        # --- Persist to SQLite ---
        if config is not None:
            diff = diff_configs(config, baseline_config) if baseline_config else None
            self._save_to_db(eval_result, config, diff, duration, notes)

    def _save_to_db(
        self,
        eval_result,
        config: dict,
        diff: dict | None,
        duration_s: float,
        notes: str,
    ) -> None:
        d = asdict(eval_result)
        conn = _get_db()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                  (run_id, tag, started_at, finished_at, duration_s,
                   composite_score, retrieval_relevance, answer_relevance,
                   groundedness, bertscore_f1, latency_ms, n_items,
                   config_json, config_diff_json, notes)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    self.run_id,
                    self.tag,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    duration_s,
                    d["composite_score"],
                    d["retrieval_relevance"],
                    d["answer_relevance"],
                    d["groundedness"],
                    d["bertscore_f1"],
                    d["latency_ms"],
                    d["n_items"],
                    json.dumps(config, default=str),
                    json.dumps(diff, default=str) if diff else None,
                    notes,
                ),
            )
            # Per-item rows
            for i, item in enumerate(d["per_item"]):
                conn.execute(
                    """
                    INSERT INTO run_per_item
                      (run_id, item_idx, query, answer,
                       retrieval_relevance, answer_relevance, groundedness,
                       bertscore_f1, composite_score, latency_ms)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        self.run_id, i,
                        item["query"], item["answer"],
                        item["retrieval_relevance"], item["answer_relevance"],
                        item["groundedness"], item["bertscore_f1"],
                        item["composite_score"], item["latency_ms"],
                    ),
                )
            conn.commit()
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Formatting helper
# ---------------------------------------------------------------------------

_COMPONENT_HEADERS = {
    "parser":               "Component 1 — Document Parsing",
    "chunk_strategy":       "Component 2 — Chunking",
    "query_prefix":         "Component 3 — Embedding",
    "index_type":           "Component 4 — Indexing (FAISS)",
    "retrieval_mode":       "Component 5 — Retrieval / Hybrid",
    "query_strategy":       "Component 6 — Query Processing",
    "reranker":             "Component 7 — Reranking",
    "context_ordering":     "Component 8 — Context Assembly",
    "system_prompt_variant":"Component 9 — Answer Generation",
}

_PARAM_ORDER = [
    # C1
    "parser", "table_extraction_strategy", "ocr_enabled",
    # C2
    "chunk_strategy", "chunk_size", "chunk_overlap", "chunk_overlap_pct",
    "metadata_injection", "contextual_compression",
    # C3
    "query_prefix", "passage_prefix", "embedding_batch_size",
    # C4
    "index_type", "metric", "hnsw_m", "ivf_nlist", "ivf_nprobe",
    # C5
    "retrieval_mode", "dense_top_k", "bm25_top_k", "final_top_k",
    "bm25_tokenizer", "hybrid_alpha", "rrf_k",
    # C6
    "query_strategy", "hyde_model", "multi_query_n",
    # C7
    "reranker", "rerank_top_k_input", "rerank_top_k_output", "rerank_score_threshold",
    # C8
    "context_ordering", "deduplication", "dedup_threshold",
    "max_context_tokens", "context_format",
    # C9
    "system_prompt_variant", "temperature", "max_tokens",
    "context_in_system", "cot_enabled", "answer_format",
]


def _print_config(config: dict, write_fn) -> None:
    printed_header = set()
    for key in _PARAM_ORDER:
        # Print component header before the first param of each component
        if key in _COMPONENT_HEADERS and key not in printed_header:
            write_fn(f"\n  ── {_COMPONENT_HEADERS[key]}")
            printed_header.add(key)
        val = config.get(key, "—")
        active = val is not None
        marker = "  " if active else "  (inactive) "
        write_fn(f"  {marker}{key:<35} {val}")

    # Any extra keys not in the ordered list
    extra = sorted(set(config) - set(_PARAM_ORDER))
    if extra:
        write_fn("\n  ── Other")
        for key in extra:
            write_fn(f"    {key:<35} {config[key]}")
