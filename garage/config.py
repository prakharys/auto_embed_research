"""
config.py — GARAGE Search Space Definition (Sprint 2)

~21 active tunable parameters. Fixed/dropped based on Sprint 1 FAnova + per-value analysis.

FIXED (not tunable):
  metric=ip                    — outperforms cosine by 0.075 (counterintuitive but real)
  system_prompt_variant=v3     — 13% variance driver, clear winner
  embedding_model=large        — 0.2% avg variance but 7/10 top trials; chase the tail
  embedding_batch_size=256     — pure performance param, zero quality effect
  index_type=Flat              — 30-PDF corpus; exact search > approximate for this scale
  bm25_tokenizer=whitespace    — not in top findings; canonical default
  bm25_b=0.75                  — 6% importance is k1, not b; 0.75 is BM25 canonical default
  cot_enabled=False            — explicitly "noise" in sprint 1 (0.567 vs 0.569)
  deduplication=False          — slightly hurts (0.550 vs 0.579); removes legit table rows
  metadata_injection=False     — <3% importance, sprint 1 best used False
  context_in_system=False      — not a key finding, sprint 1 best used False
  hyde_model=gpt-4o-mini       — simplify conditional; cheaper model fine for HyDE
  max_tokens=512               — not a key finding; sprint 1 best used 512
  rerank_score_threshold=0.0   — second-order; reranker already orders by score

DROPPED (removed from search):
  bm25_only, hybrid_cc         — weak retrievers
  unstructured parser          — 3x slower, worst mean score
  cross_encoder_minilm         — ceiling 0.572, mediocre
  html table extraction        — worst format, wastes context tokens
  numbered context format      — 0.088 below plain
  IVF index type               — underperformed Flat/HNSW in sprint 1
  cosine/l2 metric             — ip dominates
  contextual_compression       — 3x slower, no consistent gain
"""

from __future__ import annotations

import optuna


def sample(trial: optuna.Trial) -> dict:
    """
    Sample a pipeline configuration from the sprint 2 search space.
    Returns a flat dict of all parameter values; inactive conditionals are None.
    """
    cfg: dict = {}

    # ------------------------------------------------------------------
    # Component 1: Document Parsing
    # ------------------------------------------------------------------
    cfg["parser"] = trial.suggest_categorical(
        "parser", ["pymupdf", "pdfplumber"]
    )
    # "html" dropped (wastes context window); "unstructured" dropped (slow + weak)
    cfg["table_extraction_strategy"] = trial.suggest_categorical(
        "table_extraction_strategy", ["none", "text", "markdown"]
    )
    cfg["ocr_enabled"] = False  # hardcoded — digital PDFs only

    # ------------------------------------------------------------------
    # Component 2: Chunking
    # ------------------------------------------------------------------
    cfg["chunk_strategy"] = trial.suggest_categorical(
        "chunk_strategy", ["fixed", "recursive", "semantic", "sentence", "paragraph"]
    )
    cfg["chunk_size"] = trial.suggest_int("chunk_size", 512, 1024, step=64)
    cfg["chunk_overlap"] = trial.suggest_int("chunk_overlap", 0, 192, step=16)
    cfg["metadata_injection"] = False          # fixed — <3% importance, sprint 1 best used False
    cfg["contextual_compression"] = False      # fixed — 3× slower, no consistent gain
    cfg["chunk_overlap_pct"] = (
        cfg["chunk_overlap"] / cfg["chunk_size"] if cfg["chunk_size"] > 0 else 0.0
    )

    # ------------------------------------------------------------------
    # Component 3: Embedding
    # ------------------------------------------------------------------
    cfg["query_prefix"]       = "none"                        # hardcoded — OpenAI models ignore prefixes
    cfg["passage_prefix"]     = "none"                        # hardcoded — OpenAI models ignore prefixes
    cfg["embedding_model"]    = "text-embedding-3-large"      # fixed — 7/10 top trials; chase the tail
    cfg["embedding_batch_size"] = 256                         # fixed — pure performance param

    # ------------------------------------------------------------------
    # Component 4: Indexing (FAISS)
    # ------------------------------------------------------------------
    cfg["index_type"] = "Flat"   # fixed — 30-PDF corpus; exact search > approximate at this scale
    cfg["metric"]     = "ip"     # fixed — outperforms cosine by 0.075 despite L2-normalised embeds
    cfg["hnsw_m"]     = None
    cfg["ivf_nlist"]  = None
    cfg["ivf_nprobe"] = None

    # ------------------------------------------------------------------
    # Component 5: Retrieval / Hybrid Search
    # ------------------------------------------------------------------
    cfg["retrieval_mode"] = trial.suggest_categorical(
        "retrieval_mode", ["dense_only", "hybrid_rrf"]
    )
    cfg["dense_top_k"] = trial.suggest_int("dense_top_k", 5, 50)
    cfg["bm25_top_k"]  = trial.suggest_int("bm25_top_k", 5, 50)
    cfg["final_top_k"] = trial.suggest_int("final_top_k", 3, 20)

    cfg["bm25_tokenizer"] = "whitespace"   # fixed — not in top findings; canonical default
    cfg["bm25_k1"] = trial.suggest_float("bm25_k1", 0.5, 2.5)  # 6% FAnova importance — keep
    cfg["bm25_b"]  = 0.75                 # fixed — importance is in k1, not b; 0.75 is canonical

    cfg["hybrid_alpha"] = None  # hybrid_cc dropped

    if cfg["retrieval_mode"] == "hybrid_rrf":
        cfg["rrf_k"] = trial.suggest_int("rrf_k", 1, 100)
    else:
        cfg["rrf_k"] = None

    # ------------------------------------------------------------------
    # Component 6: Query Processing
    # ------------------------------------------------------------------
    cfg["query_strategy"] = trial.suggest_categorical(
        "query_strategy", ["verbatim", "hyde", "step_back", "decompose", "multi_query", "keyword"]
    )

    cfg["hyde_model"] = "gpt-4o-mini" if cfg["query_strategy"] == "hyde" else None  # fixed choice

    if cfg["query_strategy"] == "multi_query":
        cfg["multi_query_n"] = trial.suggest_int("multi_query_n", 2, 5)
    else:
        cfg["multi_query_n"] = None

    # ------------------------------------------------------------------
    # Component 7: Reranking
    # cross_encoder_minilm dropped (ceiling 0.572)
    # ------------------------------------------------------------------
    cfg["reranker"] = trial.suggest_categorical(
        "reranker", ["none", "cross_encoder_bge", "rankgpt"]
    )

    if cfg["reranker"] != "none":
        cfg["rerank_top_k_input"]     = trial.suggest_int("rerank_top_k_input", 20, 100, step=5)
        cfg["rerank_top_k_output"]    = trial.suggest_int("rerank_top_k_output", 3, 15)
        cfg["rerank_score_threshold"] = 0.0   # fixed — threshold filtering is second-order noise
    else:
        cfg["rerank_top_k_input"]     = None
        cfg["rerank_top_k_output"]    = None
        cfg["rerank_score_threshold"] = None

    # ------------------------------------------------------------------
    # Component 8: Context Assembly
    # "numbered" dropped (0.088 below plain)
    # ------------------------------------------------------------------
    cfg["context_ordering"] = trial.suggest_categorical(
        "context_ordering", ["score_desc", "score_asc", "reverse_middle", "chronological"]
    )
    cfg["deduplication"]  = False    # fixed — slightly hurts (0.550 vs 0.579)
    cfg["dedup_threshold"] = None

    cfg["max_context_tokens"] = trial.suggest_int("max_context_tokens", 512, 8192, step=256)
    cfg["context_format"] = trial.suggest_categorical(
        "context_format", ["plain", "cited", "xml_tagged"]
    )

    # ------------------------------------------------------------------
    # Component 9: Answer Generation
    # ------------------------------------------------------------------
    cfg["system_prompt_variant"] = "variant_3"   # fixed — 13% variance driver, clear winner
    cfg["temperature"]    = trial.suggest_float("temperature", 0.0, 0.7)
    cfg["max_tokens"]     = 512                  # fixed — not a key finding; sprint 1 best used 512
    cfg["context_in_system"] = False             # fixed — sprint 1 best used False
    cfg["cot_enabled"]    = False                # fixed — zero difference (0.567 vs 0.569)
    cfg["answer_format"]  = trial.suggest_categorical(
        "answer_format", ["freeform", "bullet", "structured"]
    )

    return cfg


def default_config() -> dict:
    """
    Sprint 1 Trial 080 best config (score=0.7629, latency=11.0s).
    Used as baseline for sprint 2 runs.
    """
    return {
        # Parsing
        "parser":                    "pymupdf",
        "table_extraction_strategy": "text",
        "ocr_enabled":               False,
        # Chunking
        "chunk_strategy":            "recursive",
        "chunk_size":                512,
        "chunk_overlap":             64,
        "chunk_overlap_pct":         0.125,
        "metadata_injection":        False,
        "contextual_compression":    False,
        # Embedding
        "query_prefix":              "none",
        "passage_prefix":            "none",
        "embedding_model":           "text-embedding-3-large",
        "embedding_batch_size":      256,
        # Indexing
        "index_type":                "Flat",
        "metric":                    "ip",
        "hnsw_m":                    None,
        "ivf_nlist":                 None,
        "ivf_nprobe":                None,
        # Retrieval
        "retrieval_mode":            "hybrid_rrf",
        "dense_top_k":               23,
        "bm25_top_k":                9,
        "final_top_k":               11,
        "bm25_tokenizer":            "whitespace",
        "bm25_k1":                   1.5,
        "bm25_b":                    0.75,
        "hybrid_alpha":              None,
        "rrf_k":                     60,
        # Query
        "query_strategy":            "decompose",
        "hyde_model":                None,
        "multi_query_n":             None,
        # Reranking
        "reranker":                  "none",
        "rerank_top_k_input":        None,
        "rerank_top_k_output":       None,
        "rerank_score_threshold":    None,
        # Context assembly
        "context_ordering":          "reverse_middle",
        "deduplication":             False,
        "dedup_threshold":           None,
        "max_context_tokens":        7680,
        "context_format":            "plain",
        # Answer generation
        "system_prompt_variant":     "variant_3",
        "temperature":               0.67,
        "max_tokens":                512,
        "context_in_system":         False,
        "cot_enabled":               False,
        "answer_format":             "structured",
    }


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    study = optuna.create_study(directions=["maximize", "minimize"])
    trial = study.ask()
    cfg = sample(trial)

    print(f"Sampled config ({len(cfg)} total fields):")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    print(f"\nOptuna params sampled this trial: {len(trial.params)}")
    print("Default config (sprint 1 best):")
    d = default_config()
    print(f"  {len(d)} fields")
