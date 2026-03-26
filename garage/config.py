"""
config.py — GARAGE Search Space Definition (Sprint 2)

Defines ~25 parameters and their Optuna search space.
Sprint 2 narrows the space based on Sprint 1 FAnova findings:
  - Fixed: metric=ip, system_prompt_variant=variant_3
  - Dropped: bm25_only, hybrid_cc, unstructured, cross_encoder_minilm,
             html table extraction, numbered context format, IVF index, cosine/l2 metric

Called by bo_agent.py to sample a config for each trial.

Usage:
    import optuna
    import config

    def objective(trial):
        cfg = config.sample(trial)
        # ... run pipeline with cfg ...

    study = optuna.create_study(...)
    study.optimize(objective, n_trials=100)
"""

from __future__ import annotations

import optuna


def sample(trial: optuna.Trial) -> dict:
    """
    Sample a full pipeline configuration from the search space.
    Returns a flat dict of all parameter values (including conditional ones).
    Conditional parameters are set to None when inactive.
    """
    cfg: dict = {}

    # ------------------------------------------------------------------
    # Component 1: Document Parsing
    # Sprint 2: dropped "unstructured" (3× slower, 0.505 mean vs 0.599 pymupdf)
    # ------------------------------------------------------------------
    cfg["parser"] = trial.suggest_categorical(
        "parser", ["pymupdf", "pdfplumber"]
    )
    # Sprint 2: dropped "html" (wastes context window, worst performer 0.526)
    cfg["table_extraction_strategy"] = trial.suggest_categorical(
        "table_extraction_strategy", ["none", "text", "markdown"]
    )
    cfg["ocr_enabled"] = False  # hardcoded — digital PDFs only

    # ------------------------------------------------------------------
    # Component 2: Chunking
    # Sprint 2: chunk_size floor raised to 256 (small chunks hurt dense retrieval)
    # ------------------------------------------------------------------
    cfg["chunk_strategy"] = trial.suggest_categorical(
        "chunk_strategy", ["fixed", "recursive", "semantic", "sentence", "paragraph"]
    )
    cfg["chunk_size"] = trial.suggest_int("chunk_size", 256, 1024, step=64)
    cfg["chunk_overlap"] = trial.suggest_int("chunk_overlap", 0, 192, step=16)
    cfg["metadata_injection"] = trial.suggest_categorical(
        "metadata_injection", [True, False]
    )
    cfg["contextual_compression"] = False  # disabled — 3× slower with no consistent gain
    # Derived: chunk_overlap_pct (informational, not used directly)
    cfg["chunk_overlap_pct"] = (
        cfg["chunk_overlap"] / cfg["chunk_size"] if cfg["chunk_size"] > 0 else 0.0
    )

    # ------------------------------------------------------------------
    # Component 3: Embedding
    # ------------------------------------------------------------------
    cfg["query_prefix"]   = "none"   # hardcoded — not a tunable for OpenAI embeddings
    cfg["passage_prefix"] = "none"   # hardcoded — not a tunable for OpenAI embeddings
    cfg["embedding_model"] = trial.suggest_categorical(
        "embedding_model", ["text-embedding-3-small", "text-embedding-3-large"]
    )
    cfg["embedding_batch_size"] = trial.suggest_int(
        "embedding_batch_size", 16, 512, step=16
    )

    # ------------------------------------------------------------------
    # Component 4: Indexing (FAISS)
    # Sprint 2: metric fixed to "ip" (outperforms cosine by 0.075 mean)
    #           IVF dropped (underperformed Flat/HNSW in sprint 1)
    # ------------------------------------------------------------------
    cfg["index_type"] = trial.suggest_categorical(
        "index_type", ["Flat", "HNSW"]
    )
    cfg["metric"] = "ip"  # fixed — IP dominates (0.075 over cosine) despite L2-normalised embeds

    # Conditional: HNSW-only
    if cfg["index_type"] == "HNSW":
        cfg["hnsw_m"] = trial.suggest_int("hnsw_m", 16, 64, step=4)
    else:
        cfg["hnsw_m"] = None

    # IVF dropped — always None
    cfg["ivf_nlist"]  = None
    cfg["ivf_nprobe"] = None

    # ------------------------------------------------------------------
    # Component 5: Retrieval / Hybrid Search
    # Sprint 2: bm25_only dropped (best 0.564, clearly inferior)
    #           hybrid_cc dropped (worse than hybrid_rrf consistently)
    # ------------------------------------------------------------------
    cfg["retrieval_mode"] = trial.suggest_categorical(
        "retrieval_mode", ["dense_only", "hybrid_rrf"]
    )
    cfg["dense_top_k"]  = trial.suggest_int("dense_top_k", 5, 50)
    cfg["bm25_top_k"]   = trial.suggest_int("bm25_top_k", 5, 50)
    cfg["final_top_k"]  = trial.suggest_int("final_top_k", 3, 20)
    cfg["bm25_tokenizer"] = trial.suggest_categorical(
        "bm25_tokenizer", ["whitespace", "stemming", "bpe"]
    )
    cfg["bm25_k1"] = trial.suggest_float("bm25_k1", 0.5, 2.5)
    cfg["bm25_b"]  = trial.suggest_float("bm25_b",  0.0, 1.0)

    # hybrid_cc dropped — always None
    cfg["hybrid_alpha"] = None

    # Conditional: hybrid_rrf only
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

    # Conditional: HyDE only
    if cfg["query_strategy"] == "hyde":
        cfg["hyde_model"] = trial.suggest_categorical(
            "hyde_model", ["gpt-4o-mini", "gpt-4o"]
        )
    else:
        cfg["hyde_model"] = None

    # Conditional: multi_query only
    if cfg["query_strategy"] == "multi_query":
        cfg["multi_query_n"] = trial.suggest_int("multi_query_n", 2, 5)
    else:
        cfg["multi_query_n"] = None

    # ------------------------------------------------------------------
    # Component 7: Reranking
    # Sprint 2: cross_encoder_minilm dropped (ceiling 0.572, mediocre)
    # ------------------------------------------------------------------
    cfg["reranker"] = trial.suggest_categorical(
        "reranker", ["none", "cross_encoder_bge", "rankgpt"]
    )

    # Conditional: only if reranker != none
    if cfg["reranker"] != "none":
        cfg["rerank_top_k_input"]  = trial.suggest_int("rerank_top_k_input", 20, 100, step=5)
        cfg["rerank_top_k_output"] = trial.suggest_int("rerank_top_k_output", 3, 15)
        cfg["rerank_score_threshold"] = trial.suggest_float(
            "rerank_score_threshold", 0.0, 1.0
        )
    else:
        cfg["rerank_top_k_input"]     = None
        cfg["rerank_top_k_output"]    = None
        cfg["rerank_score_threshold"] = None

    # ------------------------------------------------------------------
    # Component 8: Context Assembly
    # Sprint 2: "numbered" format dropped (worst performer, 0.088 below plain)
    # ------------------------------------------------------------------
    cfg["context_ordering"] = trial.suggest_categorical(
        "context_ordering",
        ["score_desc", "score_asc", "reverse_middle", "chronological"]
    )
    cfg["deduplication"] = trial.suggest_categorical(
        "deduplication", [True, False]
    )

    # Conditional: only if dedup enabled
    if cfg["deduplication"]:
        cfg["dedup_threshold"] = trial.suggest_float("dedup_threshold", 0.7, 0.99)
    else:
        cfg["dedup_threshold"] = None

    cfg["max_context_tokens"] = trial.suggest_int(
        "max_context_tokens", 512, 8192, step=256
    )
    # Sprint 2: "numbered" dropped (adds noise tokens, hurts attention)
    cfg["context_format"] = trial.suggest_categorical(
        "context_format", ["plain", "cited", "xml_tagged"]
    )

    # ------------------------------------------------------------------
    # Component 9: Answer Generation
    # Sprint 2: system_prompt_variant fixed to variant_3 (13% variance, best mean)
    # ------------------------------------------------------------------
    cfg["system_prompt_variant"] = "variant_3"  # fixed — 13% variance driver, clear winner
    cfg["temperature"]      = trial.suggest_float("temperature", 0.0, 0.7)
    cfg["max_tokens"]       = trial.suggest_int("max_tokens", 256, 2048, step=128)
    cfg["context_in_system"] = trial.suggest_categorical(
        "context_in_system", [True, False]
    )
    cfg["cot_enabled"] = trial.suggest_categorical(
        "cot_enabled", [True, False]
    )
    cfg["answer_format"] = trial.suggest_categorical(
        "answer_format", ["freeform", "bullet", "structured"]
    )

    return cfg


def default_config() -> dict:
    """
    Return the best-known config from sprint 1 as baseline.
    Based on Trial 080 (score=0.7629, latency=11.0s).
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
        "embedding_batch_size":      128,
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
        "cot_enabled":               True,
        "answer_format":             "structured",
    }


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    study = optuna.create_study(directions=["maximize", "minimize"])
    trial = study.ask()
    cfg = sample(trial)
    print(f"Sampled {len(cfg)} parameters:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print("\nDefault config (sprint 1 best):")
    d = default_config()
    print(f"  {len(d)} parameters")
