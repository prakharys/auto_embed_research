"""
eval.py — GARAGE GTS Scorer (FROZEN — never modified by agents)

Evaluates a RAG pipeline against the Golden Test Set (GTS).
Scoring:
  - BERTScore          (offline, reference-based)
  - LLM Judge x3      (retrieval_relevance, answer_relevance, groundedness)
Returns: EvalResult with composite_score, per-dimension scores, latency_ms
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI


# ---------------------------------------------------------------------------
# Config — read from env or .env file
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
JUDGE_MODEL    = os.environ.get("GARAGE_JUDGE_MODEL", "gpt-4o-mini")

GTS_PATH = Path(__file__).parent / "data" / "gts.jsonl"

# Composite weights — must sum to 1.0
W_RETRIEVAL   = 0.25
W_ANSWER_REL  = 0.30
W_GROUNDEDNESS = 0.25
W_BERTSCORE   = 0.20


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GTSItem:
    query: str
    expected_answer: str
    metadata: dict = None  # optional extra fields


@dataclass
class PipelineOutput:
    answer: str
    retrieved_chunks: list[str]   # raw text of retrieved passages
    latency_ms: float
    component_latency_ms: dict = None   # optional per-component breakdown


@dataclass
class EvalResult:
    composite_score: float        # 0–1, higher is better
    retrieval_relevance: float    # 0–1
    answer_relevance: float       # 0–1
    groundedness: float           # 0–1
    bertscore_f1: float           # 0–1
    latency_ms: float             # avg across GTS items
    component_latency_ms: dict = None  # avg per-component breakdown
    n_items: int = 0
    per_item: list[dict] = None   # per-query breakdown

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# GTS loader
# ---------------------------------------------------------------------------

def load_gts(path: Path = GTS_PATH) -> list[GTSItem]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            items.append(GTSItem(
                query=d["query"],
                expected_answer=d["expected_answer"],
                metadata={k: v for k, v in d.items()
                          if k not in ("query", "expected_answer")},
            ))
    return items


# ---------------------------------------------------------------------------
# LLM Judge (Azure OpenAI)
# ---------------------------------------------------------------------------

_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY, timeout=30.0)
    return _client


_JUDGE_PROMPTS = {
    "retrieval_relevance": """\
You are evaluating a retrieval system.

Query: {query}

Retrieved passages:
{passages}

Score how relevant the retrieved passages are to the query.
Return a JSON object: {{"score": <float 0-1>, "reasoning": "<one sentence>"}}
0 = completely irrelevant, 1 = perfectly relevant.
Return ONLY the JSON object, nothing else.""",

    "answer_relevance": """\
You are evaluating a QA system.

Query: {query}
Expected answer: {expected_answer}
Generated answer: {generated_answer}

Score how well the generated answer addresses the query compared to the expected answer.
Return a JSON object: {{"score": <float 0-1>, "reasoning": "<one sentence>"}}
0 = completely wrong/irrelevant, 1 = perfect match.
Return ONLY the JSON object, nothing else.""",

    "groundedness": """\
You are evaluating whether a generated answer is grounded in the provided context.

Context passages:
{passages}

Generated answer: {generated_answer}

Score how well the answer is supported by the context (no hallucinations).
Return a JSON object: {{"score": <float 0-1>, "reasoning": "<one sentence>"}}
0 = completely hallucinated, 1 = fully grounded in context.
Return ONLY the JSON object, nothing else.""",
}


def _call_judge(dimension: str, prompt: str) -> float:
    """Call the LLM judge for one dimension. Returns score 0–1."""
    client = _get_client()
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content.strip()
        data = json.loads(text)
        score = float(data.get("score", 0.0))
        return max(0.0, min(1.0, score))
    except Exception as e:
        print(f"[eval] judge error ({dimension}): {e}")
        return 0.0


def _judge_item(item: GTSItem, output: PipelineOutput) -> dict[str, float]:
    """Run 3 LLM judges in parallel for one GTS item."""
    passages_text = "\n\n".join(
        f"[{i+1}] {chunk}" for i, chunk in enumerate(output.retrieved_chunks)
    )

    prompts = {
        "retrieval_relevance": _JUDGE_PROMPTS["retrieval_relevance"].format(
            query=item.query,
            passages=passages_text,
        ),
        "answer_relevance": _JUDGE_PROMPTS["answer_relevance"].format(
            query=item.query,
            expected_answer=item.expected_answer,
            generated_answer=output.answer,
        ),
        "groundedness": _JUDGE_PROMPTS["groundedness"].format(
            passages=passages_text,
            generated_answer=output.answer,
        ),
    }

    scores: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            dim: executor.submit(_call_judge, dim, prompt)
            for dim, prompt in prompts.items()
        }
        for dim, fut in futures.items():
            scores[dim] = fut.result()

    return scores


# ---------------------------------------------------------------------------
# BERTScore (batch, offline)
# ---------------------------------------------------------------------------

_bs_tokenizer: AutoTokenizer | None = None
_bs_model: AutoModel | None = None
_BS_MODEL_NAME = "distilbert-base-uncased"

def _get_bs_model():
    global _bs_tokenizer, _bs_model
    if _bs_model is None:
        _bs_tokenizer = AutoTokenizer.from_pretrained(_BS_MODEL_NAME)
        _bs_model = AutoModel.from_pretrained(_BS_MODEL_NAME)
        _bs_model.eval()
    return _bs_tokenizer, _bs_model


def _compute_bertscore(
    hypotheses: list[str],
    references: list[str],
) -> list[float]:
    """
    BERTScore F1 computed directly via transformers on CPU.
    Bypasses the bert_score library which segfaults on macOS/MPS
    (see github.com/Tiiiger/bert_score/issues/187).

    Computes token-level cosine similarity between hypothesis and reference
    contextual embeddings, then takes the F1 of greedy max-matching.
    """
    if not hypotheses:
        return []

    tok, model = _get_bs_model()

    def _embed(texts: list[str]) -> list[torch.Tensor]:
        """Return list of (seq_len, hidden) tensors, one per text, CPU, no padding."""
        result = []
        for text in texts:
            enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                out = model(**enc)
            # mean over sub-word tokens, shape (seq_len, hidden)
            emb = out.last_hidden_state.squeeze(0)  # (L, H)
            emb = F.normalize(emb, dim=-1)
            result.append(emb)
        return result

    hyp_embs = _embed(hypotheses)
    ref_embs = _embed(references)

    f1_scores = []
    for h_emb, r_emb in zip(hyp_embs, ref_embs):
        # similarity matrix (Lh, Lr)
        sim = torch.mm(h_emb, r_emb.T)

        # Precision: for each hyp token, max similarity to any ref token
        P = sim.max(dim=1).values.mean().item()
        # Recall: for each ref token, max similarity to any hyp token
        R = sim.max(dim=0).values.mean().item()
        # F1
        if P + R > 0:
            f1 = 2 * P * R / (P + R)
        else:
            f1 = 0.0
        f1_scores.append(f1)

    return f1_scores


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def evaluate(
    pipeline_fn,                     # callable: (query: str, config: dict) -> PipelineOutput
    config: dict[str, Any],
    gts_path: Path = GTS_PATH,
    max_workers: int = 4,            # parallel GTS items for pipeline calls
    verbose: bool = False,
) -> EvalResult:
    """
    Evaluate a pipeline against the GTS.

    Args:
        pipeline_fn: Function that accepts (query, config) and returns PipelineOutput.
        config:      Pipeline configuration dict (from config.py / Optuna trial).
        gts_path:    Path to gts.jsonl.
        max_workers: Number of parallel pipeline calls.
        verbose:     Print per-item progress.

    Returns:
        EvalResult
    """
    items = load_gts(gts_path)
    if not items:
        raise ValueError(f"GTS is empty: {gts_path}")

    from tqdm import tqdm

    # ------------------------------------------------------------------
    # 1. Run pipeline on all GTS items (parallelised)
    # ------------------------------------------------------------------
    outputs: list[PipelineOutput | None] = [None] * len(items)

    def _run_one(idx: int, item: GTSItem) -> None:
        try:
            outputs[idx] = pipeline_fn(item.query, config)
        except Exception as e:
            print(f"[eval] pipeline error on item {idx}: {e}")
            outputs[idx] = PipelineOutput(
                answer="", retrieved_chunks=[], latency_ms=0.0
            )

    _log(f"eval   pipeline  n={len(items)} questions  workers={max_workers}")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_run_one, i, item) for i, item in enumerate(items)]
        with tqdm(total=len(futures), desc="pipeline", unit="q",
                  file=sys.stderr, leave=False) as pbar:
            for fut in futures:
                fut.result()
                pbar.update(1)
    _log(f"eval   pipeline  done  {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # 2. BERTScore (batch, offline)
    # ------------------------------------------------------------------
    hypotheses = [o.answer for o in outputs]
    references = [item.expected_answer for item in items]
    _log("eval   bertscore ...")
    t1 = time.time()
    with tqdm(total=1, desc="bertscore", unit="batch",
              file=sys.stderr, leave=False) as pbar:
        bertscore_f1s = _compute_bertscore(hypotheses, references)
        pbar.update(1)
    _log(f"eval   bertscore done  {time.time()-t1:.1f}s")

    # ------------------------------------------------------------------
    # 3. LLM Judges (per item, 3 dimensions in parallel within each item)
    # ------------------------------------------------------------------
    _log(f"eval   judge     n={len(items)} questions ...")
    t2 = time.time()
    judge_scores: list[dict[str, float]] = []
    for item, output in tqdm(zip(items, outputs), total=len(items),
                             desc="judge", unit="q",
                             file=sys.stderr, leave=False):
        judge_scores.append(_judge_item(item, output))
    _log(f"eval   judge     done  {time.time()-t2:.1f}s")

    # ------------------------------------------------------------------
    # 4. Aggregate
    # ------------------------------------------------------------------
    per_item = []
    for i, (item, output) in enumerate(zip(items, outputs)):
        js = judge_scores[i]
        bs = bertscore_f1s[i] if bertscore_f1s else 0.0
        composite = (
            W_RETRIEVAL   * js["retrieval_relevance"]
            + W_ANSWER_REL  * js["answer_relevance"]
            + W_GROUNDEDNESS * js["groundedness"]
            + W_BERTSCORE   * bs
        )
        per_item.append({
            "query":                  item.query,
            "answer":                 output.answer,
            "retrieval_relevance":    js["retrieval_relevance"],
            "answer_relevance":       js["answer_relevance"],
            "groundedness":           js["groundedness"],
            "bertscore_f1":           bs,
            "composite_score":        composite,
            "latency_ms":             output.latency_ms,
            "component_latency_ms":   output.component_latency_ms or {},
        })

    n = len(per_item)
    avg = lambda key: sum(d[key] for d in per_item) / n  # noqa: E731

    # Average component latencies across all items
    comp_keys = {"query_processing", "query_embedding", "retrieval",
                 "reranking", "context_assembly", "answer_generation"}
    avg_comp = {
        k: sum(d["component_latency_ms"].get(k, 0.0) for d in per_item) / n
        for k in comp_keys
        if any(k in d["component_latency_ms"] for d in per_item)
    }

    return EvalResult(
        composite_score=avg("composite_score"),
        retrieval_relevance=avg("retrieval_relevance"),
        answer_relevance=avg("answer_relevance"),
        groundedness=avg("groundedness"),
        bertscore_f1=avg("bertscore_f1"),
        latency_ms=avg("latency_ms"),
        component_latency_ms=avg_comp,
        n_items=n,
        per_item=per_item,
    )


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test eval.py")
    parser.add_argument("--gts", default=str(GTS_PATH))
    parser.add_argument("--dummy", action="store_true",
                        help="Run with a dummy pipeline (no real RAG calls)")
    args = parser.parse_args()

    if args.dummy:
        def dummy_pipeline(query: str, config: dict) -> PipelineOutput:
            return PipelineOutput(
                answer=f"Dummy answer for: {query}",
                retrieved_chunks=["Passage A about the topic.", "Passage B with more context."],
                latency_ms=42.0,
            )

        # Create a tiny dummy GTS if it doesn't exist
        gts_path = Path(args.gts)
        if not gts_path.exists():
            gts_path.parent.mkdir(parents=True, exist_ok=True)
            with open(gts_path, "w") as f:
                f.write(json.dumps({"query": "What is RAG?",
                                    "expected_answer": "RAG stands for Retrieval-Augmented Generation."}) + "\n")
                f.write(json.dumps({"query": "What is BM25?",
                                    "expected_answer": "BM25 is a ranking function used in information retrieval."}) + "\n")
            print(f"[eval] created dummy GTS at {gts_path}")

        result = evaluate(dummy_pipeline, config={}, gts_path=gts_path, verbose=True)
        print("\n--- EvalResult ---")
        for k, v in asdict(result).items():
            if k != "per_item":
                print(f"  {k}: {v}")
    else:
        print("Pass --dummy for a smoke-test without a real pipeline.")
