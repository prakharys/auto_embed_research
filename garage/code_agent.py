"""
code_agent.py — GARAGE Code Improvement Agent (FROZEN — never modified by agents)

Iteratively improves pipeline.py using an LLM backend (Claude or OpenAI).
Each experiment:
  1. Calls the LLM with: pipeline.py + program.md + score history → gets modified pipeline.py
  2. Writes the modified pipeline.py to disk
  3. Evaluates it in a subprocess (eval_subprocess.py) — isolated process so imports are fresh
  4. If score improved → git-commit the change; else → git-revert to previous
  5. Appends result to results/experiments.md

Usage:
    python code_agent.py --n-experiments 20
    python code_agent.py --n-experiments 20 --agent openai --model o3
    python code_agent.py --n-experiments 10 --config results/best_config.json
    python code_agent.py --n-experiments 5  --dry-run   # no git commits

Agent backends:
    --agent claude   Uses Anthropic Claude (default: claude-opus-4-6, adaptive thinking)
                     Requires: ANTHROPIC_API_KEY
    --agent openai   Uses OpenAI (default: o3, streaming)
                     Requires: OPENAI_API_KEY (already set for pipeline)

Requirements:
    pip install anthropic openai
"""

from __future__ import annotations

import os
# torch MUST load before faiss (libomp conflict on macOS)
import torch  # noqa: F401

import argparse
import json
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Load .env if present
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            k, v = _line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# API key checks are deferred to main() after --agent is known

import anthropic
import openai

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

GARAGE_DIR       = Path(__file__).parent
PIPELINE_PATH    = GARAGE_DIR / "pipeline.py"
PROGRAM_PATH     = GARAGE_DIR / "program.md"
RESULTS_DIR      = GARAGE_DIR / "results"
EXPERIMENTS_LOG  = RESULTS_DIR / "experiments.md"
BEST_CONFIG_PATH = RESULTS_DIR / "best_config.json"
GTS_PATH         = GARAGE_DIR / "data" / "gts.jsonl"

RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Experiment record
# ---------------------------------------------------------------------------

@dataclass
class ExperimentRecord:
    number:          int
    hypothesis:      str
    composite_score: float
    delta:           float          # vs previous best
    accepted:        bool
    latency_ms:      float
    timestamp:       str
    error:           str = ""


# ---------------------------------------------------------------------------
# Subprocess evaluation
# ---------------------------------------------------------------------------

def eval_pipeline(config: dict | None = None) -> dict | None:
    """
    Run eval_subprocess.py in a fresh subprocess.
    Returns parsed EvalResult dict on success, None on failure.
    """
    cmd = [sys.executable, str(GARAGE_DIR / "eval_subprocess.py")]
    if config is not None:
        cmd += ["--config", json.dumps(config, default=str)]
    else:
        cmd += ["--default-config"]
    cmd += ["--gts", str(GTS_PATH)]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,          # 5 min max per eval
            cwd=str(GARAGE_DIR),
        )
    except subprocess.TimeoutExpired:
        print("[code_agent] eval subprocess timed out")
        return None

    if proc.returncode != 0:
        print(f"[code_agent] eval subprocess failed (rc={proc.returncode})")
        if proc.stderr:
            print(proc.stderr[-2000:])
        return None

    # Parse the JSON line from stdout
    stdout = proc.stdout.strip()
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    print(f"[code_agent] could not parse eval output:\n{stdout[-1000:]}")
    return None


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git_commit(message: str) -> bool:
    """Stage pipeline.py and commit. Returns True on success."""
    try:
        subprocess.run(
            ["git", "add", str(PIPELINE_PATH)],
            check=True, cwd=str(GARAGE_DIR), capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", message],
            check=True, cwd=str(GARAGE_DIR), capture_output=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"[code_agent] git commit failed: {e.stderr.decode()[:500]}")
        return False


def git_restore_pipeline() -> None:
    """Restore pipeline.py to the last committed version."""
    try:
        subprocess.run(
            ["git", "checkout", "HEAD", "--", str(PIPELINE_PATH)],
            check=True, cwd=str(GARAGE_DIR), capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[code_agent] git restore failed: {e.stderr.decode()[:500]}")


def git_last_commit_hash() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(GARAGE_DIR),
        )
        return r.stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Markdown experiment log
# ---------------------------------------------------------------------------

def append_experiment_log(rec: ExperimentRecord) -> None:
    arrow  = "✅" if rec.accepted else "❌"
    delta  = f"+{rec.delta:.4f}" if rec.delta >= 0 else f"{rec.delta:.4f}"
    header = f"## Experiment {rec.number}  {arrow}  score={rec.composite_score:.4f}  Δ={delta}\n"
    body   = (
        f"**Timestamp**: {rec.timestamp}\n"
        f"**Hypothesis**: {rec.hypothesis}\n"
        f"**Composite score**: {rec.composite_score:.4f}  (Δ {delta})\n"
        f"**Latency**: {rec.latency_ms:.0f} ms\n"
        f"**Accepted**: {rec.accepted}\n"
    )
    if rec.error:
        body += f"**Error**: {rec.error}\n"

    with open(EXPERIMENTS_LOG, "a") as f:
        f.write("\n" + header + body + "\n---\n")


def init_experiments_log(baseline_score: float, agent: str = "claude", model: str = "") -> None:
    if not EXPERIMENTS_LOG.exists():
        with open(EXPERIMENTS_LOG, "w") as f:
            f.write("# GARAGE Code Agent — Experiment Log\n\n")
            f.write(f"**Backend**: {agent}  **Model**: {model}\n\n")
            f.write(f"**Baseline composite_score**: {baseline_score:.4f}\n\n")
            f.write("---\n")


# ---------------------------------------------------------------------------
# Claude API interaction
# ---------------------------------------------------------------------------

def build_system_prompt(program_md: str) -> str:
    return (
        "You are an expert RAG systems engineer.\n\n"
        + program_md
        + "\n\n"
        "## Output format\n\n"
        "Your response MUST contain exactly these two sections and nothing else:\n\n"
        "### HYPOTHESIS\n"
        "<one sentence explaining what you will change and why>\n\n"
        "### PIPELINE_PY\n"
        "```python\n"
        "<complete replacement contents of pipeline.py>\n"
        "```\n\n"
        "Do not include any other text, commentary, or sections outside these two."
    )


def parse_claude_response(text: str) -> tuple[str, str]:
    """
    Extract (hypothesis, pipeline_py_code) from Claude's response.
    Returns ("", "") if parsing fails.
    """
    hypothesis = ""
    pipeline_code = ""

    # Extract hypothesis
    if "### HYPOTHESIS" in text:
        hyp_part = text.split("### HYPOTHESIS", 1)[1]
        if "### PIPELINE_PY" in hyp_part:
            hyp_part = hyp_part.split("### PIPELINE_PY", 1)[0]
        hypothesis = hyp_part.strip()

    # Extract pipeline.py code block
    if "### PIPELINE_PY" in text:
        code_part = text.split("### PIPELINE_PY", 1)[1]
        # Find ```python ... ``` block
        if "```python" in code_part:
            code_inner = code_part.split("```python", 1)[1]
            if "```" in code_inner:
                pipeline_code = code_inner.split("```", 1)[0].strip()
        elif "```" in code_part:
            pipeline_code = code_part.split("```", 1)[1].split("```", 1)[0].strip()
        else:
            # Fallback: everything after the header
            pipeline_code = code_part.strip()

    return hypothesis, pipeline_code


def call_claude(
    client: anthropic.Anthropic,
    messages: list[dict],
    system: str,
    model: str = "claude-opus-4-6",
) -> str:
    """Stream a Claude response and return the full text."""
    full_text = ""
    print("[claude] streaming response", end="", flush=True)

    with client.messages.stream(
        model=model,
        max_tokens=8192,
        thinking={"type": "adaptive"},
        system=system,
        messages=messages,
    ) as stream:
        for event in stream:
            if (
                event.type == "content_block_delta"
                and hasattr(event.delta, "type")
                and event.delta.type == "text_delta"
            ):
                chunk = event.delta.text
                full_text += chunk
                print(".", end="", flush=True)

        final = stream.get_final_message()

    print(f" done ({final.usage.output_tokens} tokens)")
    return full_text


def call_openai(
    client: openai.OpenAI,
    messages: list[dict],
    system: str,
    model: str = "o3",
) -> str:
    """Stream an OpenAI response and return the full text."""
    full_text = ""
    print(f"[openai/{model}] streaming response", end="", flush=True)

    # Prepend system message in OpenAI format
    openai_messages = [{"role": "developer", "content": system}] + messages

    stream = client.chat.completions.create(
        model=model,
        messages=openai_messages,
        stream=True,
    )

    total_tokens = 0
    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            full_text += delta.content
            print(".", end="", flush=True)
        if chunk.usage:
            total_tokens = chunk.usage.completion_tokens or total_tokens

    print(f" done ({total_tokens} tokens)")
    return full_text


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

_AGENT_MODEL_DEFAULTS = {
    "claude": "claude-opus-4-6",
    "openai": "gpt-5.3-codex",
}


def main():
    ap = argparse.ArgumentParser(description="GARAGE Code Agent")
    ap.add_argument("--n-experiments", type=int, default=20,
                    help="Number of code improvement experiments (default: 20)")
    ap.add_argument("--config",        type=str, default=None,
                    help="Path to config JSON (default: use default_config)")
    ap.add_argument("--agent",         type=str, default="openai",
                    choices=["claude", "openai"],
                    help="LLM backend to use: openai (default) or claude")
    ap.add_argument("--model",         type=str, default=None,
                    help="Model name override (default: claude-opus-4-6 for claude, o3 for openai)")
    ap.add_argument("--dry-run",       action="store_true",
                    help="Don't commit changes to git (useful for testing)")
    ap.add_argument("--max-tokens",    type=int, default=8192,
                    help="Max tokens for response (default: 8192, claude only)")
    args = ap.parse_args()

    # Resolve model default based on agent
    if args.model is None:
        args.model = _AGENT_MODEL_DEFAULTS[args.agent]

    # Validate required API keys for the chosen backend
    if args.agent == "claude" and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: ANTHROPIC_API_KEY not set. Required for --agent claude.")
    if args.agent == "openai" and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY not set. Required for --agent openai.")

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    config: dict | None = None
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            sys.exit(f"ERROR: config file not found: {config_path}")
        config = json.loads(config_path.read_text())
        print(f"[code_agent] loaded config from {config_path}")
    elif BEST_CONFIG_PATH.exists():
        config = json.loads(BEST_CONFIG_PATH.read_text())
        print(f"[code_agent] loaded best_config from {BEST_CONFIG_PATH}")
    else:
        print("[code_agent] using default_config")

    # ------------------------------------------------------------------
    # Baseline evaluation
    # ------------------------------------------------------------------
    print("[code_agent] running baseline eval...")
    baseline = eval_pipeline(config)
    if baseline is None:
        sys.exit("ERROR: baseline eval failed — check pipeline.py and data/")

    best_score   = baseline["composite_score"]
    best_latency = baseline["latency_ms"]
    print(f"[code_agent] baseline  composite={best_score:.4f}  latency={best_latency:.0f}ms")

    init_experiments_log(best_score, agent=args.agent, model=args.model)

    # ------------------------------------------------------------------
    # LLM client + static context
    # ------------------------------------------------------------------
    if args.agent == "claude":
        llm_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    else:
        llm_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    print(f"[code_agent] backend={args.agent}  model={args.model}")

    program_md    = PROGRAM_PATH.read_text()
    system_prompt = build_system_prompt(program_md)

    # History of scores for the conversation context
    score_history: list[str] = [
        f"Baseline: composite_score={best_score:.4f}, latency_ms={best_latency:.0f}"
    ]

    # Conversation messages — we keep a sliding window to stay under context
    conversation: list[dict] = []

    # ------------------------------------------------------------------
    # Experiment loop
    # ------------------------------------------------------------------
    for exp_num in range(1, args.n_experiments + 1):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'='*60}")
        print(f"  Experiment {exp_num}/{args.n_experiments}  [{ts}]")
        print(f"  Current best: composite={best_score:.4f}  latency={best_latency:.0f}ms")
        print(f"{'='*60}")

        pipeline_code = PIPELINE_PATH.read_text()

        # Build user message for this turn
        user_content = (
            f"## Current pipeline.py\n\n```python\n{pipeline_code}\n```\n\n"
            f"## Score history\n\n"
            + "\n".join(f"- {s}" for s in score_history[-10:])  # last 10 entries
            + f"\n\n## Task\n\nRun experiment {exp_num}. "
            "Propose your next improvement. Remember: one focused change at a time."
        )

        conversation.append({"role": "user", "content": user_content})

        # Keep conversation from growing too large (sliding window)
        if len(conversation) > 6:
            # Keep only last 6 messages (3 turns)
            conversation = conversation[-6:]

        # Call LLM backend
        try:
            if args.agent == "claude":
                response_text = call_claude(
                    llm_client,
                    conversation,
                    system_prompt,
                    model=args.model,
                )
            else:
                response_text = call_openai(
                    llm_client,
                    conversation,
                    system_prompt,
                    model=args.model,
                )
        except Exception as e:
            print(f"[code_agent] {args.agent} API error: {e}")
            record = ExperimentRecord(
                number=exp_num, hypothesis="(API error)",
                composite_score=best_score, delta=0.0,
                accepted=False, latency_ms=best_latency,
                timestamp=ts, error=str(e),
            )
            append_experiment_log(record)
            continue

        # Append assistant message to conversation
        conversation.append({"role": "assistant", "content": response_text})

        # Parse response
        hypothesis, new_pipeline_code = parse_claude_response(response_text)

        if not new_pipeline_code:
            print("[code_agent] could not parse pipeline.py from response — skipping")
            record = ExperimentRecord(
                number=exp_num, hypothesis=hypothesis or "(parse error)",
                composite_score=best_score, delta=0.0,
                accepted=False, latency_ms=best_latency,
                timestamp=ts, error="Failed to parse pipeline code",
            )
            append_experiment_log(record)
            continue

        print(f"[code_agent] hypothesis: {hypothesis[:120]}")

        # Write the new pipeline.py
        PIPELINE_PATH.write_text(new_pipeline_code)

        # Evaluate
        print("[code_agent] evaluating modified pipeline...")
        result = eval_pipeline(config)

        if result is None:
            print("[code_agent] eval failed — reverting pipeline.py")
            git_restore_pipeline()
            record = ExperimentRecord(
                number=exp_num, hypothesis=hypothesis,
                composite_score=best_score, delta=0.0,
                accepted=False, latency_ms=best_latency,
                timestamp=ts, error="Eval subprocess failed",
            )
            append_experiment_log(record)
            score_history.append(
                f"Exp {exp_num}: FAILED (eval error) | hypothesis: {hypothesis[:60]}"
            )
            continue

        new_score   = result["composite_score"]
        new_latency = result["latency_ms"]
        delta       = new_score - best_score
        accepted    = new_score >= best_score   # accept if equal or better

        print(f"[code_agent] result  composite={new_score:.4f}  "
              f"Δ={delta:+.4f}  latency={new_latency:.0f}ms  "
              f"{'✅ ACCEPT' if accepted else '❌ REVERT'}")

        if accepted:
            best_score   = new_score
            best_latency = new_latency
            if not args.dry_run:
                commit_msg = (
                    f"exp {exp_num}: {hypothesis[:60]} "
                    f"(score {best_score:.4f})"
                )
                git_commit(commit_msg)
        else:
            # Revert to last committed version
            if not args.dry_run:
                git_restore_pipeline()
            else:
                # In dry-run mode, restore from the content we read earlier
                PIPELINE_PATH.write_text(pipeline_code)

        record = ExperimentRecord(
            number=exp_num, hypothesis=hypothesis,
            composite_score=new_score, delta=delta,
            accepted=accepted, latency_ms=new_latency,
            timestamp=ts,
        )
        append_experiment_log(record)

        score_history.append(
            f"Exp {exp_num} ({'✅' if accepted else '❌'}): "
            f"composite={new_score:.4f} Δ={delta:+.4f} "
            f"lat={new_latency:.0f}ms | {hypothesis[:60]}"
        )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Code Agent finished  ({args.n_experiments} experiments)")
    print(f"  Final best composite_score: {best_score:.4f}")
    print(f"  Experiment log: {EXPERIMENTS_LOG}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
