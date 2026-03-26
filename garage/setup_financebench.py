"""
setup_financebench.py — Download FinanceBench corpus + ground truth for GARAGE.

Selects 10 well-known companies whose PDFs are hosted on SEC EDGAR
(d18rn0p25nwr6d.cloudfront.net) — these links are stable and reliable.

Downloads:
  garage/data/corpus/<DOC_NAME>.pdf   ← 10-K annual reports
  garage/data/gts.jsonl               ← QA pairs for those docs only

Clears the old toy corpus first.

Usage:
    python setup_financebench.py
    python setup_financebench.py --dry-run    # show plan without downloading
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import requests
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GARAGE_DIR  = Path(__file__).parent
CORPUS_DIR  = GARAGE_DIR / "data" / "corpus"
GTS_PATH    = GARAGE_DIR / "data" / "gts.jsonl"

# Companies to include — chosen for:
#   1. Reliable SEC EDGAR PDF links
#   2. ≥3 QA pairs each
#   3. Sector diversity
SELECTED_COMPANIES = {
    "Boeing",
    "Best Buy",
    "MGM Resorts",
    "Nike",
    "Amazon",
    "General Mills",
    "Pfizer",
    "Corning",
    "CVS Health",
    "Walmart",
}

HEADERS = {
    "User-Agent": "GARAGE-Research/1.0 research@example.com",
    "Accept": "application/pdf",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_pdf(url: str, dest: Path, retries: int = 3) -> bool:
    """Download a PDF to dest. Returns True on success."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=60, stream=True)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "pdf" not in content_type and attempt == retries:
                print(f"  ⚠  unexpected content-type: {content_type}")
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            size_kb = dest.stat().st_size // 1024
            print(f"  ✓  {dest.name}  ({size_kb} KB)")
            return True
        except Exception as e:
            print(f"  ✗  attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(2 * attempt)
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="FinanceBench corpus setup")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show plan without downloading anything")
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Load HuggingFace dataset
    # ------------------------------------------------------------------
    print("Loading FinanceBench from HuggingFace...")
    hf_token = os.environ.get("HF_TOKEN")
    ds = load_dataset("PatronusAI/financebench", split="train", token=hf_token)
    print(f"  {len(ds)} total QA pairs loaded\n")

    # ------------------------------------------------------------------
    # Filter to selected companies
    # ------------------------------------------------------------------
    selected_rows = [r for r in ds if r["company"] in SELECTED_COMPANIES]

    # Group by doc to avoid downloading the same PDF twice
    docs: dict[str, dict] = {}   # doc_name → {link, company, rows}
    for r in selected_rows:
        dn = r["doc_name"]
        if dn not in docs:
            docs[dn] = {"link": r["doc_link"], "company": r["company"], "rows": []}
        docs[dn]["rows"].append(r)

    # Print plan
    print(f"{'='*60}")
    print(f"  Plan: {len(docs)} documents, {len(selected_rows)} QA pairs")
    print(f"{'='*60}")
    for dn, info in sorted(docs.items()):
        n = len(info['rows'])
        print(f"  {info['company']:20s}  {n} QAs  {dn}")
    print()

    if args.dry_run:
        print("[dry-run] Stopping here. Remove --dry-run to actually download.")
        return

    # ------------------------------------------------------------------
    # Clear old toy corpus
    # ------------------------------------------------------------------
    print("Clearing old corpus...")
    if CORPUS_DIR.exists():
        for f in CORPUS_DIR.iterdir():
            f.unlink()
            print(f"  removed {f.name}")
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Download PDFs
    # ------------------------------------------------------------------
    print(f"\nDownloading {len(docs)} PDFs → {CORPUS_DIR}\n")
    downloaded: set[str] = set()
    failed: set[str] = set()

    for doc_name, info in sorted(docs.items()):
        dest = CORPUS_DIR / f"{doc_name}.pdf"
        print(f"  {info['company']} — {doc_name}")
        ok = download_pdf(info["link"], dest)
        if ok:
            downloaded.add(doc_name)
        else:
            failed.add(doc_name)
        time.sleep(0.5)   # polite delay

    print(f"\n  Downloaded: {len(downloaded)}/{len(docs)}")
    if failed:
        print(f"  Failed:     {sorted(failed)}")

    # ------------------------------------------------------------------
    # Write gts.jsonl (only for successfully downloaded docs)
    # ------------------------------------------------------------------
    print(f"\nWriting {GTS_PATH}...")
    kept = 0
    skipped = 0
    with open(GTS_PATH, "w") as f:
        for dn, info in sorted(docs.items()):
            if dn in failed:
                print(f"  skipping QAs for {dn} (PDF failed)")
                skipped += len(info["rows"])
                continue
            for r in info["rows"]:
                record = {
                    "query":           r["question"],
                    "expected_answer": r["answer"],
                    "doc":             r["doc_name"],
                    "company":         r["company"],
                }
                f.write(json.dumps(record) + "\n")
                kept += 1

    print(f"  {kept} QA pairs written  ({skipped} skipped due to failed downloads)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Setup complete!")
    print(f"  Corpus : {CORPUS_DIR}  ({len(list(CORPUS_DIR.glob('*.pdf')))} PDFs)")
    print(f"  GTS    : {GTS_PATH}  ({kept} questions)")
    print(f"\n  Next:")
    print(f"    python bo_agent.py --n-trials 100 --study-name garage_financebench")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
