# GARAGE — RAG Pipeline Bayesian Optimisation

Automated search over the full RAG pipeline parameter space using multi-objective
Bayesian optimisation (Optuna NSGA-II). Evaluated on **FinanceBench** (51 financial
Q&A questions across 30 annual report PDFs).

## What it does

Each trial samples a config across ~45 parameters spanning every stage of the RAG
pipeline — parsing, chunking, embedding, indexing, retrieval, query processing,
reranking, context assembly, and answer generation — then evaluates it end-to-end.

**Objectives:**
- Maximise `composite_score = 0.25·retrieval_relevance + 0.30·answer_relevance + 0.25·groundedness + 0.20·bertscore_f1`
- Minimise `latency_ms`

---

## Project structure

```
garage/
├── pipeline.py          # Full RAG pipeline implementation
├── config.py            # Optuna search space definition (~45 params)
├── bo_agent.py          # Bayesian optimisation runner (NSGA-II)
├── eval.py              # Evaluation harness (BERTScore + LLM judge)
├── eval_subprocess.py   # Subprocess wrapper for code agent eval
├── logger.py            # Run logger (console + .log + results.db)
├── run_sprint1.py       # Sprint 1 orchestration script
├── code_agent.py        # Code agent (pipeline.py improvement loop)
├── plot_live.py         # Live score dashboard
├── plot_runs.py         # Per-run score chart
├── setup_financebench.py
├── data/
│   ├── gts.jsonl        # Ground truth (51 Q&A pairs)
│   ├── corpus/          # 30 PDFs (gitignored)
│   ├── index_cache/     # Cached FAISS indexes (gitignored)
│   └── results.db       # Optuna SQLite store (gitignored)
└── results/
    ├── analysis_sprint1.py   # Sprint 1 analysis + plots
    ├── best_config.json      # Best config from last study
    └── plot_*.png            # Generated analysis plots
```

---

## Quickstart

```bash
cp .env.example .env        # add OPENAI_API_KEY
pip install -r requirements.txt

# Run BO (resumes from existing study if results.db exists)
python bo_agent.py --n-trials 100 --study-name my_study

# Or run the full sprint 1 pipeline (baseline → BO)
python run_sprint1.py --n-trials 200
```

---

## Sprint 1 Results

**155 trials completed** on FinanceBench. Baseline → best: **0.4996 → 0.7629** (+52.7% relative).

### Score distribution
| Bucket | Count |
|--------|-------|
| < 0.50 (below baseline) | 34 |
| 0.50–0.60 | 49 |
| 0.60–0.65 | 28 |
| 0.65–0.70 | 19 |
| 0.70–0.75 | 22 |
| > 0.75 (top tier) | 3 |

### Best config (Trial 080, score=0.7629, latency=11.0s)
| Component | Value |
|-----------|-------|
| Parser | pdfplumber, tables=none |
| Chunking | paragraph(320), overlap=16 |
| Embedding | text-embedding-3-large |
| Index | Flat, metric=ip |
| Retrieval | hybrid_rrf, dense_k=23, bm25_k=9, final_k=11 |
| Query | decompose |
| Reranker | none |
| Context | plain format, reverse_middle order, 7680 tokens |
| Generation | variant_3, temp=0.67, cot=True, structured |

### Best Pareto config (Trial 196, score=0.7524, latency=4.2s)
Highest score-per-second: `pdfplumber` + `recursive(576)` + `large` + `Flat/ip` +
`hybrid_rrf` + `verbatim` + no reranker + `variant_3`.

---

## Key Findings from Sprint 1

### 1. Retrieval mode dominates everything (42% of variance)
`dense_only` and `hybrid_rrf` are clearly best; `bm25_only` is never competitive
(best score 0.564). Surprisingly, `dense_only` ties `hybrid_rrf` on average —
BM25 adds noise on semantic financial queries rather than signal.

### 2. System prompt is the second biggest lever (13% of variance)
`variant_3` beats `variant_1/2` consistently. More importantly, `variant_1` has
std=0.130 (wildly inconsistent) vs `variant_3` std=0.094 — the prompt wording
affects robustness, not just mean score.

### 3. Embedding model doesn't matter on average (0.2% of variance)
`text-embedding-3-large` and `small` have identical mean scores (0.568). However,
`large` appears in 7 of the top 10 trials — it enables peak performance without
lifting the average. The difference shows in the tail, not the bulk.

### 4. Reranking is marginal
`cross_encoder_bge` and no reranker are statistically tied (+0.005 difference).
`cross_encoder_minilm` is consistently mediocre (std=0.030, ceiling at 0.572).
`rankgpt` is boom-or-bust (std=0.118) — great when retrieval is strong, worse
than nothing otherwise.

### 5. Cosine similarity underperforms inner product (counterintuitive)
`metric=ip` outperforms `cosine` by 0.075 mean score. Since OpenAI embeddings are
L2-normalised, IP and cosine are mathematically equivalent — the difference is
likely an artefact of how FAISS handles the metrics internally with HNSW/IVF.

### 6. Context format matters more than expected
`plain` text (mean=0.601) beats `numbered` (mean=0.513) by 0.088. Numbered
prefixes add noise tokens and shift LLM attention away from content.

### 7. CoT and deduplication are noise
`cot_enabled`: True vs False → 0.567 vs 0.569. Zero practical difference.
`deduplication=True` slightly *hurts* (0.550 vs 0.579) — likely removes
legitimate duplicate financial table rows.

### 8. pymupdf is fastest and best
`pymupdf` (mean=0.599) > `pdfplumber` (0.562) > `unstructured` (0.505).
`unstructured` is also 3× slower. Clear winner.

### 9. HTML table extraction hurts, text helps
Table extraction matters: `text` (0.616) > `markdown` (0.588) > `none` (0.556) > `html` (0.526).
HTML tags waste context window tokens.

### 10. No continuous parameter predicts score well
All continuous params have |r| < 0.15 with score. The optimal values are
regime-dependent, not universal — e.g., large chunks help with dense retrieval,
hurt with BM25 tokenisation.

---

## What to remove for Sprint 2 (dead weight from Sprint 1)

Based on parameter importance and per-value score distributions:

| Parameter | Remove |
|-----------|--------|
| `retrieval_mode` | Drop `bm25_only`, `hybrid_cc` |
| `parser` | Drop `unstructured` |
| `reranker` | Drop `cross_encoder_minilm` |
| `table_extraction_strategy` | Drop `html` |
| `context_format` | Drop `numbered` |
| `metric` | Fix to `ip` |
| `system_prompt_variant` | Fix to `variant_3` or replace with DSPy-optimised prompt |

---

## Parameter importance (FAnova, 155 trials)

**For composite_score:**
`retrieval_mode` (42%) → `system_prompt_variant` (13%) → `reranker` (6%) →
`bm25_k1` (6%) → `chunk_overlap` (4%) → everything else (<3% each)

**For latency_ms:**
`chunk_overlap` (51%) → `query_strategy` (14%) → `chunk_size` (14%) →
everything else (<4% each)

The two objectives are driven by almost completely different parameters —
retrieval config drives quality, chunking config drives speed.
