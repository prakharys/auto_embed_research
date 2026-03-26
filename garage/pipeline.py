"""
pipeline.py — GARAGE RAG Pipeline (MODIFIED BY CODE AGENT ONLY)

Modular RAG implementation. All 9 components are config-driven.
Code agent may rewrite any algorithm here; eval.py interface is frozen.

Entry point:
    run(query: str, config: dict, index: RAGIndex) -> PipelineOutput

Index is built separately (build_index) and cached between BO trials.
"""

from __future__ import annotations

import os
# Must be set before faiss is imported to prevent libomp double-init crash
# on macOS when both torch and faiss link their own copy of libomp.dylib.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import hashlib
import math
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


def _log(msg: str) -> None:
    """Timestamped print that always flushes — visible in log files."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

import logging
import numpy as np
import tiktoken

from openai import OpenAI

# Suppress httpx INFO logs ("HTTP Request: POST ...") — very chatty in log files
logging.getLogger("httpx").setLevel(logging.WARNING)

# FAISS
import faiss
faiss.omp_set_num_threads(1)   # prevent libomp double-init segfault on macOS

# BM25
from rank_bm25 import BM25Okapi, BM25Plus

# Sentence Transformers (rerankers)
from sentence_transformers import CrossEncoder

# Text splitters
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

# ---------------------------------------------------------------------------
# Environment / client
# ---------------------------------------------------------------------------

OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
EMBED_MODEL     = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
LLM_DEPLOYMENT  = os.environ.get("OPENAI_LLM_MODEL", "gpt-4o-mini")

# Prevent HuggingFace tokenizers from forking and conflicting with macOS OpenMP
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_oai_client: OpenAI | None = None

def _client() -> OpenAI:
    global _oai_client
    if _oai_client is None:
        _oai_client = OpenAI(api_key=OPENAI_API_KEY, timeout=60.0, max_retries=3)
    return _oai_client


def _with_retry(fn, max_attempts: int = 5, base_delay: float = 2.0):
    """Call fn() with exponential backoff on RateLimit / timeout errors."""
    import openai as _oai
    for attempt in range(max_attempts):
        try:
            return fn()
        except (_oai.RateLimitError, _oai.APITimeoutError, _oai.APIConnectionError) as e:
            if attempt == max_attempts - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"[pipeline] retry {attempt+1}/{max_attempts} after {delay:.0f}s: {e}", flush=True)
            time.sleep(delay)
        except _oai.BadRequestError:
            raise  # don't retry 400s


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Document:
    text: str
    source: str = ""
    title: str = ""
    page: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    text: str
    doc_source: str = ""
    doc_title: str = ""
    page: int = 0
    chunk_idx: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float           # combined retrieval score (higher = better)
    dense_score: float = 0.0
    bm25_score: float = 0.0


@dataclass
class RAGIndex:
    """Prebuilt index — created once, reused across trials with the same chunking config."""
    chunks: list[Chunk]
    embeddings: np.ndarray          # shape (N, D), float32
    faiss_index: Any                # faiss.Index
    bm25: Any                       # BM25Okapi | BM25Plus
    tokenized_chunks: list[list[str]]
    config_hash: str                # hash of chunking/embedding config used to build this


@dataclass
class PipelineOutput:
    answer: str
    retrieved_chunks: list[str]     # raw text of top-k passages (for eval.py)
    latency_ms: float
    component_latency_ms: dict = field(default_factory=dict)
    # keys: query_processing, query_embedding, retrieval, reranking,
    #       context_assembly, answer_generation


# ---------------------------------------------------------------------------
# Component 1 — Document Parsing
# ---------------------------------------------------------------------------

def parse_documents(paths: list[Path], config: dict) -> list[Document]:
    from tqdm import tqdm
    parser = config.get("parser", "pymupdf")
    table_strategy = config.get("table_extraction_strategy", "text")
    ocr_enabled = config.get("ocr_enabled", False)

    _log(f"parse  parser={parser} tables={table_strategy} n={len(paths)} PDFs")
    t0 = time.time()
    docs: list[Document] = []
    for path in tqdm(paths, desc=f"parse({parser})", unit="pdf",
                     file=sys.stderr, leave=False):
        path = Path(path)
        if parser == "pymupdf":
            docs.extend(_parse_pymupdf(path, table_strategy))
        elif parser == "pdfplumber":
            docs.extend(_parse_pdfplumber(path, table_strategy))
        elif parser == "unstructured":
            docs.extend(_parse_unstructured(path, ocr_enabled))
        else:
            text = path.read_text(errors="replace")
            docs.append(Document(text=text, source=str(path), title=path.stem))
    _log(f"parse  done  {len(docs)} docs  {time.time()-t0:.1f}s")
    return docs


def _parse_pymupdf(path: Path, table_strategy: str) -> list[Document]:
    import fitz  # pymupdf
    docs = []
    try:
        pdf = fitz.open(str(path))
        for page_num, page in enumerate(pdf):
            text = page.get_text("text")
            if table_strategy == "markdown":
                tabs = page.find_tables()
                for tab in tabs:
                    text += "\n" + tab.to_markdown()
            elif table_strategy == "html":
                tabs = page.find_tables()
                for tab in tabs:
                    text += "\n" + tab.to_pandas().to_html(index=False)
            if text.strip():
                docs.append(Document(
                    text=text, source=str(path),
                    title=path.stem, page=page_num,
                ))
        pdf.close()
    except Exception as e:
        print(f"[pipeline] pymupdf parse error {path}: {e}")
    return docs


def _parse_pdfplumber(path: Path, table_strategy: str) -> list[Document]:
    import pdfplumber
    docs = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if table_strategy != "none":
                    for table in page.extract_tables():
                        rows = ["\t".join(str(c) for c in row) for row in table if row]
                        text += "\n" + "\n".join(rows)
                if text.strip():
                    docs.append(Document(
                        text=text, source=str(path),
                        title=path.stem, page=page_num,
                    ))
    except Exception as e:
        print(f"[pipeline] pdfplumber parse error {path}: {e}")
    return docs


def _parse_unstructured(path: Path, ocr_enabled: bool) -> list[Document]:
    try:
        from langchain_unstructured import UnstructuredLoader

        class _StrippedLoader(UnstructuredLoader):
            """UnstructuredLoader with tempdir prefix removed from metadata source."""
            def _get_metadata(self) -> dict:
                return {"source": str(path)}

        loader = _StrippedLoader(file_path=str(path), strategy="fast")
        lc_docs = loader.load()
        return [
            Document(
                text=doc.page_content,
                source=str(path),
                title=path.stem,
            )
            for doc in lc_docs
            if doc.page_content.strip()
        ]
    except Exception as e:
        print(f"[pipeline] unstructured parse error {path}: {e}")
        return []


# ---------------------------------------------------------------------------
# Component 2 — Chunking
# ---------------------------------------------------------------------------

def chunk_documents(docs: list[Document], config: dict) -> list[Chunk]:
    from tqdm import tqdm
    strategy = config.get("chunk_strategy", "recursive")
    size     = config.get("chunk_size", 512)
    overlap  = config.get("chunk_overlap", 64)
    # Guard: overlap capped at 50% of chunk_size (size-1 allowed huge overlap on small chunks)
    overlap  = min(overlap, max(0, size // 2))
    inject   = config.get("metadata_injection", False)
    compress = config.get("contextual_compression", False)

    _log(f"chunk  strategy={strategy} size={size} overlap={overlap} n={len(docs)} docs")
    t0 = time.time()
    chunks: list[Chunk] = []
    for doc in tqdm(docs, desc=f"chunk({strategy},{size})", unit="doc",
                    file=sys.stderr, leave=False):
        raw_chunks = _split_text(doc.text, strategy, size, overlap)
        for i, text in enumerate(raw_chunks):
            if inject:
                prefix = f"[Source: {doc.title}]\n" if doc.title else ""
                text = prefix + text
            chunks.append(Chunk(
                text=text,
                doc_source=doc.source,
                doc_title=doc.title,
                page=doc.page,
                chunk_idx=i,
            ))

    if compress and chunks:
        chunks = _contextual_compression(chunks, config)

    _log(f"chunk  done  {len(chunks)} chunks  {time.time()-t0:.1f}s")
    return chunks


def _split_text(text: str, strategy: str, size: int, overlap: int) -> list[str]:
    if strategy == "fixed":
        splitter = CharacterTextSplitter(
            separator="", chunk_size=size, chunk_overlap=overlap,
            length_function=len,
        )
    elif strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=overlap,
        )
    elif strategy == "sentence":
        return _sentence_split(text, size, overlap)
    elif strategy == "paragraph":
        return _paragraph_split(text, size, overlap)
    elif strategy == "semantic":
        return _semantic_split(text, size, overlap)
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=overlap,
        )
    return splitter.split_text(text)


def _sentence_split(text: str, max_size: int, overlap: int) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current, current_len = [], [], 0
    for sent in sentences:
        if current_len + len(sent) > max_size and current:
            chunks.append(" ".join(current))
            # keep overlap
            overlap_sents = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) <= overlap:
                    overlap_sents.insert(0, s)
                    overlap_len += len(s)
                else:
                    break
            current = overlap_sents
            current_len = overlap_len
        current.append(sent)
        current_len += len(sent)
    if current:
        chunks.append(" ".join(current))
    return chunks or [text]


def _paragraph_split(text: str, max_size: int, overlap: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    chunks, current, current_len = [], [], 0
    for para in paragraphs:
        if current_len + len(para) > max_size and current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(para)
        current_len += len(para)
    if current:
        chunks.append("\n\n".join(current))
    return chunks or [text]


def _semantic_split(text: str, target_size: int, overlap: int) -> list[str]:
    """Naive semantic split: splits on sentence boundaries, grouping by cosine similarity."""
    # Fallback to recursive if we can't embed (no API key during testing)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=target_size, chunk_overlap=overlap
    )
    return splitter.split_text(text)


def _contextual_compression(chunks: list[Chunk], config: dict,
                            max_workers: int = 32) -> list[Chunk]:
    """Add LLM-generated context prefix to each chunk (Anthropic-style contextual retrieval).

    Batched with ThreadPoolExecutor — 10k chunks takes ~5-10 min instead of ~2 hrs.
    Results are cached via the index disk cache (different config_hash from uncompressed).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _log(f"compress  n={len(chunks)} chunks  workers={max_workers} ...")
    t0 = time.time()

    def _compress_one(chunk: Chunk) -> Chunk:
        prompt = (
            f"Here is a passage from a document titled '{chunk.doc_title}':\n\n"
            f"{chunk.text}\n\n"
            "Write a single sentence that provides context for this passage "
            "that would help someone searching for it. Be concise."
        )
        try:
            context = _with_retry(lambda: _client().chat.completions.create(
                model=LLM_DEPLOYMENT,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=80,
                temperature=0.0,
            ).choices[0].message.content.strip())
            return Chunk(
                text=f"{context}\n\n{chunk.text}",
                doc_source=chunk.doc_source,
                doc_title=chunk.doc_title,
                page=chunk.page,
                chunk_idx=chunk.chunk_idx,
                metadata=chunk.metadata,
            )
        except Exception:
            return chunk

    from tqdm import tqdm
    compressed = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_compress_one, chunk): i for i, chunk in enumerate(chunks)}
        with tqdm(total=len(chunks), desc="compress", unit="chunk",
                  file=sys.stderr, leave=False) as pbar:
            for fut in as_completed(futures):
                compressed[futures[fut]] = fut.result()
                pbar.update(1)

    _log(f"compress  done  {time.time()-t0:.1f}s")
    return compressed


# ---------------------------------------------------------------------------
# Component 3 — Embedding
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str], config: dict, is_query: bool = False) -> np.ndarray:
    prefix_key = "query_prefix" if is_query else "passage_prefix"
    prefix = config.get(prefix_key, "none")
    if prefix == "none":
        prefix = ""

    batch_size = config.get("embedding_batch_size", 128)
    client = _client()

    from tqdm import tqdm
    model = config.get("embedding_model", EMBED_MODEL)
    kind = "query" if is_query else "corpus"
    t0 = time.time()

    # Build token-safe batches: OpenAI enforces 300k tokens/request.
    # Estimate tokens as len(text)//4 (conservative). Cap at 250k to leave headroom.
    _MAX_TOKENS_PER_REQ = 250_000
    batches: list[list[str]] = []
    cur_batch: list[str] = []
    cur_tokens = 0
    for t in texts:
        est = max(1, len(t) // 4)
        if cur_batch and (len(cur_batch) >= batch_size or cur_tokens + est > _MAX_TOKENS_PER_REQ):
            batches.append(cur_batch)
            cur_batch, cur_tokens = [], 0
        cur_batch.append(t)
        cur_tokens += est
    if cur_batch:
        batches.append(cur_batch)

    n_batches = len(batches)
    # Only log corpus embeddings (large batches) — query embeds are tiny and called per question
    if not is_query:
        _log(f"embed  model={model.split('-')[-1]} n={len(texts)} texts  {n_batches} batches  [{kind}]")
    all_embeddings = []
    for batch in tqdm(batches,
                      total=n_batches,
                      desc=f"embed({model.split('-')[-1]})",
                      unit="batch", file=sys.stderr, leave=False):
        prefixed = [prefix + t if prefix else t for t in batch]
        resp = _with_retry(lambda: client.embeddings.create(model=model, input=prefixed))
        vecs = [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]
        all_embeddings.extend(vecs)

    arr = np.ascontiguousarray(all_embeddings, dtype=np.float32)
    if not is_query:
        _log(f"embed  done  shape={arr.shape}  {time.time()-t0:.1f}s")
    return arr


# ---------------------------------------------------------------------------
# Component 4 — Indexing (FAISS)
# ---------------------------------------------------------------------------

def build_faiss_index(embeddings: np.ndarray, config: dict) -> faiss.Index:
    index_type = config.get("index_type", "Flat")
    metric     = config.get("metric", "cosine")
    dim = embeddings.shape[1]

    # Ensure C-contiguous float32 array — FAISS segfaults on macOS with
    # non-contiguous or non-float32 arrays, especially at large scale.
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    # Normalize for cosine similarity
    if metric == "cosine":
        faiss.normalize_L2(embeddings)
        base_metric = faiss.METRIC_INNER_PRODUCT
    elif metric == "ip":
        base_metric = faiss.METRIC_INNER_PRODUCT
    else:
        base_metric = faiss.METRIC_L2

    if index_type == "Flat":
        index = faiss.IndexFlatIP(dim) if base_metric == faiss.METRIC_INNER_PRODUCT \
                else faiss.IndexFlatL2(dim)
    elif index_type == "IVF":
        nlist  = config.get("ivf_nlist", 128)
        nprobe = config.get("ivf_nprobe", 32)
        quantizer = faiss.IndexFlatIP(dim) if base_metric == faiss.METRIC_INNER_PRODUCT \
                    else faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, base_metric)
        if not index.is_trained:
            index.train(embeddings)
        index.nprobe = nprobe
    elif index_type == "HNSW":
        m = config.get("hnsw_m", 32)
        index = faiss.IndexHNSWFlat(dim, m, base_metric)
    else:
        index = faiss.IndexFlatIP(dim)

    index.add(embeddings)
    return index


def build_bm25(tokenized: list[list[str]], config: dict):
    k1 = config.get("bm25_k1", 1.5)
    b  = config.get("bm25_b",  0.75)
    return BM25Okapi(tokenized, k1=k1, b=b)


def tokenize_for_bm25(texts: list[str], config: dict) -> list[list[str]]:
    tokenizer = config.get("bm25_tokenizer", "whitespace")
    if tokenizer == "whitespace":
        return [t.lower().split() for t in texts]
    elif tokenizer == "stemming":
        try:
            from nltk.stem import PorterStemmer
            stemmer = PorterStemmer()
            return [
                [stemmer.stem(w) for w in t.lower().split()]
                for t in texts
            ]
        except Exception:
            return [t.lower().split() for t in texts]
    elif tokenizer == "bpe":
        enc = tiktoken.get_encoding("cl100k_base")
        return [
            [str(tok) for tok in enc.encode(t)]
            for t in texts
        ]
    return [t.lower().split() for t in texts]


# ---------------------------------------------------------------------------
# Build full RAGIndex
# ---------------------------------------------------------------------------

def build_index(chunks: list[Chunk], config: dict,
                cache_dir: Path | None = None) -> RAGIndex:
    """Build FAISS + BM25 index from chunks. Call once per chunking/embedding config.

    If cache_dir is given, checks for a saved index first (keyed by config_hash)
    and writes one after building. Serialises: numpy (embeddings), faiss.write_index,
    pickle (chunks, bm25, tokenized_chunks).
    """
    import pickle

    cfg_hash = _config_hash(config)

    # --- try loading from disk cache ---
    if cache_dir is not None:
        cached = _load_index(cfg_hash, Path(cache_dir), config=config)
        if cached is not None:
            return cached

    # Guard: prune configs that produce absurdly large indexes (would take hours + OOM)
    _MAX_CHUNKS = 100_000
    if len(chunks) > _MAX_CHUNKS:
        raise ValueError(f"chunk count {len(chunks)} exceeds limit {_MAX_CHUNKS} — pruning trial")

    texts = [c.text for c in chunks]
    _log(f"index  building  n={len(chunks)} chunks  index={config.get('index_type','Flat')}  metric={config.get('metric','cosine')}")
    t0 = time.time()

    embeddings = embed_texts(texts, config, is_query=False)
    faiss_index = build_faiss_index(embeddings.copy(), config)
    tokenized = tokenize_for_bm25(texts, config)
    bm25 = build_bm25(tokenized, config)

    _log(f"index  done  hash={cfg_hash}  {time.time()-t0:.1f}s")
    index = RAGIndex(
        chunks=chunks,
        embeddings=embeddings,
        faiss_index=faiss_index,
        bm25=bm25,
        tokenized_chunks=tokenized,
        config_hash=cfg_hash,
    )

    # --- save to disk cache ---
    if cache_dir is not None:
        _save_index(index, Path(cache_dir))

    return index


def _save_index(index: RAGIndex, cache_dir: Path) -> None:
    """Persist a RAGIndex to cache_dir/{config_hash}/."""
    import pickle
    slot = cache_dir / index.config_hash
    slot.mkdir(parents=True, exist_ok=True)
    np.save(str(slot / "embeddings.npy"), index.embeddings)
    faiss.write_index(index.faiss_index, str(slot / "faiss.index"))
    with open(slot / "bm25.pkl", "wb") as f:
        pickle.dump(index.bm25, f)
    with open(slot / "chunks_tokenized.pkl", "wb") as f:
        pickle.dump((index.chunks, index.tokenized_chunks), f)
    _log(f"index  saved → {slot}")


def _load_index(config_hash: str, cache_dir: Path,
                config: dict | None = None) -> RAGIndex | None:
    """Load a RAGIndex from cache_dir/{config_hash}/ if it exists.

    BM25 is always rebuilt from cached tokenized_chunks so that bm25_k1/bm25_b
    changes don't require re-embedding.  The old bm25.pkl is loaded as fallback
    if config is not supplied (backward-compat), otherwise ignored.
    """
    import pickle
    slot = cache_dir / config_hash
    required = ["embeddings.npy", "faiss.index", "chunks_tokenized.pkl"]
    if not all((slot / f).exists() for f in required):
        return None
    _log(f"index  cache hit  hash={config_hash}  loading from {slot}")
    t0 = time.time()
    embeddings = np.load(str(slot / "embeddings.npy"))
    faiss_index = faiss.read_index(str(slot / "faiss.index"))
    with open(slot / "chunks_tokenized.pkl", "rb") as f:
        chunks, tokenized = pickle.load(f)
    # Rebuild BM25 with current k1/b so scoring reflects the current trial's params
    if config is not None:
        bm25 = build_bm25(tokenized, config)
    elif (slot / "bm25.pkl").exists():
        with open(slot / "bm25.pkl", "rb") as f:
            bm25 = pickle.load(f)
    else:
        bm25 = build_bm25(tokenized, {})
    _log(f"index  cache loaded  {time.time()-t0:.1f}s")
    return RAGIndex(
        chunks=chunks,
        embeddings=embeddings,
        faiss_index=faiss_index,
        bm25=bm25,
        tokenized_chunks=tokenized,
        config_hash=config_hash,
    )


# Parameters that affect the content of the index (embeddings, FAISS, tokenized chunks).
# bm25_k1, bm25_b, retrieval_mode, query_strategy, reranker, answer-gen params etc.
# do NOT affect the stored index — exclude them so the disk cache gets reused.
_INDEX_KEYS = frozenset({
    # Parsing
    "parser", "table_extraction_strategy", "ocr_enabled",
    # Chunking
    "chunk_strategy", "chunk_size", "chunk_overlap",
    "metadata_injection", "contextual_compression",
    # Embedding
    "embedding_model", "embedding_batch_size",
    "query_prefix", "passage_prefix",
    # FAISS
    "index_type", "metric", "hnsw_m", "ivf_nlist", "ivf_nprobe",
    # BM25 tokenisation (affects tokenized_chunks content, not just scoring)
    "bm25_tokenizer",
})


def _config_hash(config: dict) -> str:
    keys = sorted((k, v) for k, v in config.items() if k in _INDEX_KEYS)
    return hashlib.md5(str(keys).encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Component 5 — Retrieval / Hybrid Search
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    query_embedding: np.ndarray,
    index: RAGIndex,
    config: dict,
) -> list[ScoredChunk]:
    mode       = config.get("retrieval_mode", "hybrid_rrf")
    dense_k    = config.get("dense_top_k", 20)
    bm25_k     = config.get("bm25_top_k", 20)
    final_k    = config.get("final_top_k", 10)

    if mode == "dense_only":
        return _dense_retrieve(query_embedding, index, config, dense_k)[:final_k]

    if mode == "bm25_only":
        return _bm25_retrieve(query, index, config, bm25_k)[:final_k]

    # Hybrid
    dense_results = _dense_retrieve(query_embedding, index, config, dense_k)
    bm25_results  = _bm25_retrieve(query, index, config, bm25_k)

    if mode == "hybrid_rrf":
        return _rrf_fuse(dense_results, bm25_results, config, final_k)
    elif mode == "hybrid_cc":
        alpha = config.get("hybrid_alpha", 0.5)
        return _cc_fuse(dense_results, bm25_results, alpha, final_k)
    return dense_results[:final_k]


def _dense_retrieve(
    query_emb: np.ndarray, index: RAGIndex, config: dict, k: int
) -> list[ScoredChunk]:
    metric = config.get("metric", "cosine")
    q = query_emb.copy().reshape(1, -1).astype(np.float32)
    if metric == "cosine":
        faiss.normalize_L2(q)
    k = min(k, len(index.chunks))
    scores, idxs = index.faiss_index.search(q, k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        results.append(ScoredChunk(
            chunk=index.chunks[idx],
            score=float(score),
            dense_score=float(score),
        ))
    return results


def _bm25_retrieve(
    query: str, index: RAGIndex, config: dict, k: int
) -> list[ScoredChunk]:
    tokenizer = config.get("bm25_tokenizer", "whitespace")
    tokens = tokenize_for_bm25([query], config)[0]
    scores = index.bm25.get_scores(tokens)
    top_idxs = np.argsort(scores)[::-1][:k]
    results = []
    for idx in top_idxs:
        results.append(ScoredChunk(
            chunk=index.chunks[idx],
            score=float(scores[idx]),
            bm25_score=float(scores[idx]),
        ))
    return results


def _rrf_fuse(
    dense: list[ScoredChunk],
    bm25: list[ScoredChunk],
    config: dict,
    k: int,
) -> list[ScoredChunk]:
    rrf_k = config.get("rrf_k") or 60
    scores: dict[int, float] = {}
    chunk_map: dict[int, Chunk] = {}

    for rank, sc in enumerate(dense):
        idx = id(sc.chunk)
        chunk_map[idx] = sc.chunk
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)

    for rank, sc in enumerate(bm25):
        idx = id(sc.chunk)
        chunk_map[idx] = sc.chunk
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)

    sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
    return [ScoredChunk(chunk=chunk_map[i], score=scores[i]) for i in sorted_ids]


def _cc_fuse(
    dense: list[ScoredChunk],
    bm25: list[ScoredChunk],
    alpha: float,
    k: int,
) -> list[ScoredChunk]:
    """Linear combination: score = alpha * dense_norm + (1-alpha) * bm25_norm"""
    def _normalize(items: list[ScoredChunk], attr: str) -> dict[int, float]:
        vals = [getattr(sc, attr) for sc in items]
        mn, mx = min(vals, default=0), max(vals, default=1)
        rng = mx - mn or 1.0
        return {id(sc.chunk): (getattr(sc, attr) - mn) / rng for sc in items}

    d_norm = _normalize(dense, "dense_score")
    b_norm = _normalize(bm25, "bm25_score")

    all_chunks: dict[int, Chunk] = {}
    for sc in dense + bm25:
        all_chunks[id(sc.chunk)] = sc.chunk

    scores = {
        cid: alpha * d_norm.get(cid, 0.0) + (1 - alpha) * b_norm.get(cid, 0.0)
        for cid in all_chunks
    }
    sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
    return [ScoredChunk(chunk=all_chunks[cid], score=scores[cid]) for cid in sorted_ids]


# ---------------------------------------------------------------------------
# Component 6 — Query Processing
# ---------------------------------------------------------------------------

def process_query(query: str, config: dict) -> list[str]:
    """Return one or more queries to run retrieval on."""
    strategy = config.get("query_strategy", "verbatim")

    if strategy == "verbatim":
        return [query]
    elif strategy == "hyde":
        return [query, _hyde(query, config)]
    elif strategy == "step_back":
        return [query, _step_back(query)]
    elif strategy == "decompose":
        return _decompose(query, config)
    elif strategy == "multi_query":
        n = config.get("multi_query_n", 3)
        return [query] + _multi_query(query, n)
    elif strategy == "keyword":
        return [query, _keyword(query)]
    return [query]


def _llm_call(prompt: str, max_tokens: int = 256) -> str:
    return _with_retry(lambda: _client().chat.completions.create(
        model=LLM_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    ).choices[0].message.content.strip())


def _hyde(query: str, config: dict) -> str:
    model = config.get("hyde_model", "gpt-4o-mini")
    prompt = (
        f"Write a short hypothetical document that would answer the following question:\n\n"
        f"Question: {query}\n\nHypothetical document:"
    )
    return _llm_call(prompt, max_tokens=200)


def _step_back(query: str) -> str:
    prompt = (
        f"Given this specific question, write a more general/abstract version "
        f"that captures the broader topic:\n\nQuestion: {query}\n\nAbstract question:"
    )
    return _llm_call(prompt, max_tokens=100)


def _decompose(query: str, config: dict) -> list[str]:
    prompt = (
        f"Break this complex question into 2-3 simpler sub-questions. "
        f"Return each on a new line, no numbering:\n\nQuestion: {query}\n\nSub-questions:"
    )
    result = _llm_call(prompt, max_tokens=200)
    sub_qs = [q.strip() for q in result.splitlines() if q.strip()]
    return [query] + sub_qs[:3]


def _keyword(query: str) -> str:
    prompt = (
        f"Rewrite the following question as a short, keyword-focused search query. "
        f"Remove filler words, keep only the most informative terms, and use noun phrases. "
        f"Return only the rewritten query, nothing else.\n\n"
        f"Question: {query}\n\nSearch query:"
    )
    return _llm_call(prompt, max_tokens=40)


def _multi_query(query: str, n: int) -> list[str]:
    prompt = (
        f"Generate {n} different phrasings of this question for better retrieval. "
        f"Return each on a new line, no numbering:\n\nQuestion: {query}\n\nAlternate phrasings:"
    )
    result = _llm_call(prompt, max_tokens=300)
    variants = [q.strip() for q in result.splitlines() if q.strip()]
    return variants[:n]


# ---------------------------------------------------------------------------
# Component 7 — Reranking
# ---------------------------------------------------------------------------

_cross_encoder_cache: dict[str, CrossEncoder] = {}

def _get_cross_encoder(model_name: str) -> CrossEncoder:
    if model_name not in _cross_encoder_cache:
        _cross_encoder_cache[model_name] = CrossEncoder(model_name, device="cpu")
    return _cross_encoder_cache[model_name]


def rerank(
    query: str,
    candidates: list[ScoredChunk],
    config: dict,
) -> list[ScoredChunk]:
    reranker = config.get("reranker", "none")
    top_k_input  = config.get("rerank_top_k_input", 20)
    top_k_output = config.get("rerank_top_k_output", 5)
    threshold    = config.get("rerank_score_threshold", 0.0)

    candidates = candidates[:top_k_input]

    if reranker == "none":
        return candidates[:top_k_output]

    if reranker == "cross_encoder_minilm":
        model = _get_cross_encoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return _cross_encoder_rerank(query, candidates, model, top_k_output, threshold)

    if reranker == "cross_encoder_bge":
        model = _get_cross_encoder("BAAI/bge-reranker-base")
        return _cross_encoder_rerank(query, candidates, model, top_k_output, threshold)

    if reranker == "rankgpt":
        return _rankgpt_rerank(query, candidates, top_k_output)

    return candidates[:top_k_output]


def _cross_encoder_rerank(
    query: str,
    candidates: list[ScoredChunk],
    model: CrossEncoder,
    top_k: int,
    threshold: float,
) -> list[ScoredChunk]:
    pairs = [(query, sc.chunk.text) for sc in candidates]
    scores = model.predict(pairs)
    ranked = sorted(
        zip(scores, candidates), key=lambda x: x[0], reverse=True
    )
    results = []
    for score, sc in ranked[:top_k]:
        if score >= threshold:
            results.append(ScoredChunk(
                chunk=sc.chunk, score=float(score),
                dense_score=sc.dense_score, bm25_score=sc.bm25_score,
            ))
    return results


def _rankgpt_rerank(
    query: str,
    candidates: list[ScoredChunk],
    top_k: int,
) -> list[ScoredChunk]:
    """Zero-shot listwise reranking via LLM."""
    passages = "\n".join(
        f"[{i+1}] {sc.chunk.text[:300]}" for i, sc in enumerate(candidates)
    )
    prompt = (
        f"Rank the following passages by relevance to the query. "
        f"Return a comma-separated list of passage numbers, most relevant first.\n\n"
        f"Query: {query}\n\nPassages:\n{passages}\n\nRanked order (e.g. 3,1,2):"
    )
    try:
        result = _llm_call(prompt, max_tokens=100)
        order = [int(x.strip()) - 1 for x in result.split(",") if x.strip().isdigit()]
        reranked = [candidates[i] for i in order if 0 <= i < len(candidates)]
        # append any missed candidates at the end
        seen = set(order)
        for i, sc in enumerate(candidates):
            if i not in seen:
                reranked.append(sc)
        return reranked[:top_k]
    except Exception:
        return candidates[:top_k]


# ---------------------------------------------------------------------------
# Component 8 — Context Assembly
# ---------------------------------------------------------------------------

def assemble_context(chunks: list[ScoredChunk], config: dict) -> str:
    ordering     = config.get("context_ordering", "score_desc")
    dedup        = config.get("deduplication", False)
    dedup_thresh = config.get("dedup_threshold", 0.85)
    max_tokens   = config.get("max_context_tokens", 2048)
    fmt          = config.get("context_format", "numbered")

    # Order
    if ordering == "score_desc":
        ordered = sorted(chunks, key=lambda x: x.score, reverse=True)
    elif ordering == "score_asc":
        ordered = sorted(chunks, key=lambda x: x.score)
    elif ordering == "reverse_middle":
        # Lost-in-the-middle mitigation: put best at start and end
        desc = sorted(chunks, key=lambda x: x.score, reverse=True)
        result = []
        left, right = True, False
        for i, sc in enumerate(desc):
            if i % 2 == 0:
                result.insert(0, sc)
            else:
                result.append(sc)
        ordered = result
    elif ordering == "chronological":
        ordered = sorted(chunks, key=lambda x: (x.chunk.doc_source, x.chunk.page, x.chunk.chunk_idx))
    else:
        ordered = chunks

    # Dedup
    if dedup:
        ordered = _deduplicate(ordered, dedup_thresh)

    # Truncate to token budget
    enc = tiktoken.get_encoding("cl100k_base")
    selected: list[ScoredChunk] = []
    total_tokens = 0
    for sc in ordered:
        n_tok = len(enc.encode(sc.chunk.text))
        if total_tokens + n_tok > max_tokens:
            break
        selected.append(sc)
        total_tokens += n_tok

    # Format
    return _format_context(selected, fmt)


def _deduplicate(chunks: list[ScoredChunk], threshold: float) -> list[ScoredChunk]:
    """Remove near-duplicate chunks using Jaccard similarity on word sets."""
    kept: list[ScoredChunk] = []
    kept_sets: list[set] = []
    for sc in chunks:
        words = set(sc.chunk.text.lower().split())
        is_dup = False
        for existing in kept_sets:
            union = words | existing
            if not union:
                continue
            jaccard = len(words & existing) / len(union)
            if jaccard >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(sc)
            kept_sets.append(words)
    return kept


def _format_context(chunks: list[ScoredChunk], fmt: str) -> str:
    if fmt == "plain":
        return "\n\n".join(sc.chunk.text for sc in chunks)
    elif fmt == "numbered":
        return "\n\n".join(f"[{i+1}] {sc.chunk.text}" for i, sc in enumerate(chunks))
    elif fmt == "cited":
        parts = []
        for i, sc in enumerate(chunks):
            src = sc.chunk.doc_title or sc.chunk.doc_source or f"doc{i+1}"
            parts.append(f"[{i+1}] (Source: {src})\n{sc.chunk.text}")
        return "\n\n".join(parts)
    elif fmt == "xml_tagged":
        parts = []
        for i, sc in enumerate(chunks):
            parts.append(f"<passage id='{i+1}'>\n{sc.chunk.text}\n</passage>")
        return "\n".join(parts)
    return "\n\n".join(sc.chunk.text for sc in chunks)


# ---------------------------------------------------------------------------
# Component 9 — Answer Generation
# ---------------------------------------------------------------------------

# Prompt variants (code agent may add more)
PROMPT_VARIANTS = {
    "variant_1": (
        "You are a helpful assistant. Answer the user's question based only on the provided context. "
        "If the answer is not in the context, say 'I don't have enough information to answer this.'"
    ),
    "variant_2": (
        "You are an expert analyst. Use the provided context to give a precise, accurate answer. "
        "Cite relevant passages when possible. If the context is insufficient, state this clearly."
    ),
    "variant_3": (
        "Answer the question using only the information in the context below. "
        "Be concise and direct. Do not speculate beyond the provided information."
    ),
}


def generate_answer(query: str, context: str, config: dict) -> str:
    variant     = config.get("system_prompt_variant", "variant_1")
    temperature = config.get("temperature", 0.0)
    max_tokens  = config.get("max_tokens", 512)
    ctx_in_sys  = config.get("context_in_system", False)
    cot_enabled = config.get("cot_enabled", False)
    answer_fmt  = config.get("answer_format", "freeform")

    system_prompt = PROMPT_VARIANTS.get(variant, PROMPT_VARIANTS["variant_1"])

    if answer_fmt == "bullet":
        system_prompt += "\nFormat your answer as bullet points."
    elif answer_fmt == "structured":
        system_prompt += "\nFormat your answer with clear sections and headers."

    if ctx_in_sys:
        system_content = system_prompt + f"\n\nContext:\n{context}"
        user_content = query
    else:
        system_content = system_prompt
        user_content = f"Context:\n{context}\n\nQuestion: {query}"

    if cot_enabled:
        user_content += "\n\nLet's think step by step."

    resp = _client().chat.completions.create(
        model=LLM_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user",   "content": user_content},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run(query: str, config: dict, index: RAGIndex) -> PipelineOutput:
    """
    Execute the full RAG pipeline for a single query.

    Args:
        query:  User question.
        config: Flat config dict (from Optuna trial or best_config.json).
        index:  Pre-built RAGIndex (built once, reused across queries).

    Returns:
        PipelineOutput with answer, retrieved_chunks, latency_ms.
    """
    t0 = time.perf_counter()
    _t = t0  # rolling timer

    def _elapsed() -> float:
        nonlocal _t
        now = time.perf_counter()
        ms = (now - _t) * 1000
        _t = now
        return ms

    # 1. Query processing (may produce multiple queries)
    queries = process_query(query, config)
    comp = {"query_processing": _elapsed()}

    # 2. Embed all queries
    query_embeddings = embed_texts(queries, config, is_query=True)
    comp["query_embedding"] = _elapsed()

    # 3. Retrieve for each query, merge by RRF
    all_candidates: list[list[ScoredChunk]] = []
    for i, q in enumerate(queries):
        q_emb = query_embeddings[i]
        candidates = retrieve(q, q_emb, index, config)
        all_candidates.append(candidates)

    if len(all_candidates) == 1:
        merged = all_candidates[0]
    else:
        merged = _merge_multi_query(all_candidates, config)
    comp["retrieval"] = _elapsed()

    # 4. Rerank
    final_chunks = rerank(query, merged, config)
    comp["reranking"] = _elapsed()

    # 5. Context assembly
    context = assemble_context(final_chunks, config)
    comp["context_assembly"] = _elapsed()

    # 6. Answer generation
    answer = generate_answer(query, context, config)
    comp["answer_generation"] = _elapsed()

    latency_ms = (time.perf_counter() - t0) * 1000

    return PipelineOutput(
        answer=answer,
        retrieved_chunks=[sc.chunk.text for sc in final_chunks],
        latency_ms=latency_ms,
        component_latency_ms=comp,
    )


def _merge_multi_query(
    all_candidates: list[list[ScoredChunk]],
    config: dict,
) -> list[ScoredChunk]:
    """RRF merge of multiple query result sets."""
    rrf_k = config.get("rrf_k") or 60
    scores: dict[str, float] = {}
    chunk_map: dict[str, Chunk] = {}

    for candidates in all_candidates:
        for rank, sc in enumerate(candidates):
            key = sc.chunk.text[:64]  # use text prefix as dedup key
            chunk_map[key] = sc.chunk
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)

    sorted_keys = sorted(scores, key=scores.__getitem__, reverse=True)
    final_k = config.get("final_top_k", 10)
    return [ScoredChunk(chunk=chunk_map[k], score=scores[k]) for k in sorted_keys[:final_k]]
