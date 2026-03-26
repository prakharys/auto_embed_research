"""
eval_subprocess.py — Subprocess evaluation runner for code_agent.py

Runs the full pipeline + eval in this process and prints a JSON result to stdout.
Designed to be called as a subprocess so code_agent can eval a potentially
modified pipeline.py without contaminating its own process.

Usage:
    python eval_subprocess.py --config '{"chunk_size": 512, ...}' --gts data/gts.jsonl
    python eval_subprocess.py --default-config --gts data/gts.jsonl

Output (stdout, one JSON line):
    {"composite_score": 0.95, "retrieval_relevance": 1.0, ..., "per_item": [...]}

Exit codes:
    0 — success
    1 — error (stderr has details)
"""

from __future__ import annotations

import os
# torch MUST load before faiss (libomp conflict on macOS)
import torch  # noqa: F401

import argparse
import json
import sys
from pathlib import Path

# Load .env if present
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            k, v = _line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

if not os.environ.get("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
    sys.exit(1)

from pipeline import Document, chunk_documents, build_index, run, PipelineOutput
from config import default_config
from eval import evaluate, GTS_PATH
from dataclasses import asdict


DATA_DIR   = Path(__file__).parent / "data"
CORPUS_DIR = DATA_DIR / "corpus"


def build_corpus_index(config: dict):
    txt_files = list(CORPUS_DIR.glob("*.txt"))
    pdf_files = list(CORPUS_DIR.glob("*.pdf"))
    all_files  = txt_files + pdf_files

    if not all_files:
        print(f"ERROR: No documents in {CORPUS_DIR}", file=sys.stderr)
        sys.exit(1)

    docs: list[Document] = []
    for path in txt_files:
        text = path.read_text(errors="replace")
        docs.append(Document(text=text, source=str(path), title=path.stem))
    for path in pdf_files:
        from pipeline import parse_documents
        docs.extend(parse_documents([path], config))

    chunks = chunk_documents(docs, config)
    index  = build_index(chunks, config)
    return index


def main():
    parser = argparse.ArgumentParser(description="Subprocess eval runner")
    parser.add_argument("--config",         type=str,  default=None,
                        help="JSON config dict")
    parser.add_argument("--default-config", action="store_true",
                        help="Use default_config()")
    parser.add_argument("--gts",            type=str,  default=str(GTS_PATH),
                        help="Path to gts.jsonl")
    parser.add_argument("--max-workers",    type=int,  default=4,
                        help="Parallel pipeline calls for eval")
    args = parser.parse_args()

    if args.default_config:
        cfg = default_config()
    elif args.config:
        cfg = json.loads(args.config)
    else:
        print("ERROR: supply --config JSON or --default-config", file=sys.stderr)
        sys.exit(1)

    gts_path = Path(args.gts)
    if not gts_path.exists():
        print(f"ERROR: GTS not found: {gts_path}", file=sys.stderr)
        sys.exit(1)

    index = build_corpus_index(cfg)

    def pipeline_fn(query: str, c: dict) -> PipelineOutput:
        return run(query, c, index)

    result = evaluate(pipeline_fn, cfg, gts_path=gts_path,
                      max_workers=args.max_workers, verbose=False)

    # Print single JSON line to stdout
    print(json.dumps(asdict(result)))


if __name__ == "__main__":
    main()
