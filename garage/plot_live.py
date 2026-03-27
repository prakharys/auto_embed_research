"""
plot_live.py — Live score dashboard for GARAGE BO runs.

Polls the Optuna SQLite DB every 30 s and redraws:
  - Top-left  : Composite score vs trial (scatter + running best line)
  - Top-right : Pareto front (score vs latency)
  - Bottom    : Score breakdown by key config dimensions

Usage:
    .venv/bin/python plot_live.py
    .venv/bin/python plot_live.py --study garage_financebench --interval 15
    .venv/bin/python plot_live.py --no-window   # headless: just save PNG every interval

Saves results/live_scores.png on every refresh.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

GARAGE_DIR  = Path(__file__).parent
DATA_DIR    = GARAGE_DIR / "data"
RESULTS_DIR = GARAGE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

STORAGE      = f"sqlite:///{DATA_DIR / 'results.db'}"
SAVE_PATH    = RESULTS_DIR / "live_scores.png"
BASELINE     = 0.7629   # sprint 1 best — new floor for sprint 2
LATENCY_XLIM = 25_000   # ms — clip pareto x-axis at 25 s


# ---------------------------------------------------------------------------
# Dark theme helpers
# ---------------------------------------------------------------------------

BG      = "#0f1117"
AX_BG   = "#161b22"
GRID    = "#2d333b"
TEXT    = "#c9d1d9"
ACCENT  = "#58a6ff"
GREEN   = "#3fb950"
ORANGE  = "#d29922"
RED     = "#f85149"
PURPLE  = "#bc8cff"
CYAN    = "#39d353"

def _style_ax(ax):
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.grid(True, color=GRID, linewidth=0.5, linestyle="--", alpha=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("bottom", "left"):
        ax.spines[sp].set_color(GRID)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trials(study_name: str):
    try:
        study = optuna.load_study(study_name=study_name, storage=STORAGE)
    except Exception:
        return [], [], None

    complete = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.values
    ]
    pruned = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.PRUNED
    ]
    return complete, pruned, study


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw(fig, axes, study_name: str):
    ax_prog, ax_pareto, ax_dims = axes

    complete, pruned, study = load_trials(study_name)

    for ax in axes:
        ax.cla()
        _style_ax(ax)

    n_complete = len(complete)
    n_pruned   = len(pruned)
    n_total    = n_complete + n_pruned

    # ------------------------------------------------------------------ #
    # Panel 1 — Score progression
    # ------------------------------------------------------------------ #
    ax_prog.set_title(
        f"Score Progression  |  {n_complete} complete  {n_pruned} pruned  / 100",
        fontsize=10, pad=6,
    )
    ax_prog.set_xlabel("Trial #", fontsize=9)
    ax_prog.set_ylabel("Composite Score", fontsize=9)
    ax_prog.axhline(BASELINE, color=ORANGE, linewidth=1.2,
                    linestyle="--", alpha=0.8, label=f"Baseline ({BASELINE:.3f})")

    if complete:
        nums    = [t.number for t in complete]
        scores  = [t.values[0] for t in complete]

        # running best
        best_so_far = []
        cur_best = -np.inf
        for s in scores:
            cur_best = max(cur_best, s)
            best_so_far.append(cur_best)

        ax_prog.scatter(nums, scores, c=ACCENT, s=30, alpha=0.65,
                        edgecolors="none", zorder=3, label="Trial score")
        best_idx = int(np.argmax(scores))
        best_score = scores[best_idx]
        ax_prog.plot(nums, best_so_far, color=GREEN, linewidth=1.8,
                     zorder=4, label=f"Best: {best_score:.4f}")

        ax_prog.annotate(
            f"  {best_score:.4f}",
            xy=(nums[best_idx], scores[best_idx]),
            color=GREEN, fontsize=8, va="bottom",
        )

    ax_prog.legend(fontsize=8, facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT,
                   loc="upper left")

    # ------------------------------------------------------------------ #
    # Panel 2 — Pareto front
    # ------------------------------------------------------------------ #
    ax_pareto.set_title("Pareto Front  (↑ score, ↓ latency)", fontsize=10, pad=6)
    ax_pareto.set_xlabel("Latency (ms)", fontsize=9)
    ax_pareto.set_ylabel("Composite Score", fontsize=9)
    ax_pareto.axhline(BASELINE, color=ORANGE, linewidth=1.2,
                      linestyle="--", alpha=0.8)

    if complete:
        scores    = [t.values[0] for t in complete]
        latencies = [t.values[1] for t in complete]

        try:
            pareto   = study.best_trials
            p_scores = [t.values[0] for t in pareto]
            p_lats   = [t.values[1] for t in pareto]
        except Exception:
            pareto, p_scores, p_lats = [], [], []

        ax_pareto.scatter(latencies, scores, c=ACCENT, s=25, alpha=0.45,
                          edgecolors="none", label="All trials")
        if p_scores:
            ax_pareto.scatter(p_lats, p_scores, c=GREEN, s=60, zorder=5,
                              edgecolors="none", label=f"Pareto ({len(pareto)})")
            # connect pareto front
            paired = sorted(zip(p_lats, p_scores))
            ax_pareto.plot([x for x, _ in paired], [y for _, y in paired],
                           color=GREEN, linewidth=1.2, linestyle="--", alpha=0.6)

        ax_pareto.set_xlim(0, LATENCY_XLIM)

    ax_pareto.legend(fontsize=8, facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT)

    # ------------------------------------------------------------------ #
    # Panel 3 — Breakdown by key dimensions
    # ------------------------------------------------------------------ #
    ax_dims.set_title("Median Score by Config Dimension", fontsize=10, pad=6)
    ax_dims.set_ylabel("Median Composite Score", fontsize=9)

    if complete:
        dims = {
            "retrieval_mode": None,
            "chunk_strategy": None,
            "reranker":       None,
            "query_strategy": None,
        }

        groups = {}
        for dim, mapping in dims.items():
            buckets: dict[str, list[float]] = {}
            for t in complete:
                val = t.params.get(dim, "?")
                label = (mapping or {}).get(val, val)
                buckets.setdefault(label, []).append(t.values[0])
            groups[dim] = buckets

        # flatten to bar chart
        labels, medians, colors_bar = [], [], []
        color_cycle = [ACCENT, GREEN, ORANGE, PURPLE, CYAN, RED,
                       "#ff7b72", "#ffa657", "#a5d6ff"]
        ci = 0
        sep_positions = []
        pos = 0
        xtick_pos, xtick_lab = [], []

        for dim, buckets in groups.items():
            sep_positions.append(pos - 0.5)
            for label, vals in sorted(buckets.items()):
                labels.append(label)
                medians.append(float(np.median(vals)))
                colors_bar.append(color_cycle[ci % len(color_cycle)])
                ci += 1
                pos += 1
            # dim label in middle of its group
            mid = pos - len(buckets) / 2
            xtick_pos.append(mid - 0.5)
            xtick_lab.append(dim.replace("_", "\n"))

        xs = range(len(labels))
        bars = ax_dims.bar(xs, medians, color=colors_bar, alpha=0.85,
                           edgecolor="none", width=0.7)
        ax_dims.axhline(BASELINE, color=ORANGE, linewidth=1.0,
                        linestyle="--", alpha=0.7)

        # value labels on bars
        for bar, med in zip(bars, medians):
            ax_dims.text(bar.get_x() + bar.get_width() / 2, med + 0.003,
                         f"{med:.3f}", ha="center", va="bottom",
                         fontsize=7, color=TEXT)

        # value sub-labels (bucket names)
        ax_dims.set_xticks(list(xs))
        ax_dims.set_xticklabels(labels, fontsize=7.5, rotation=30, ha="right")

        # vertical separators between dims
        for sep in sep_positions[1:]:
            ax_dims.axvline(sep, color=GRID, linewidth=1.0, alpha=0.8)

        # dim group labels above
        ax2 = ax_dims.twiny()
        ax2.set_xlim(ax_dims.get_xlim())
        ax2.set_facecolor(AX_BG)
        ax2.set_xticks(xtick_pos)
        ax2.set_xticklabels(xtick_lab, fontsize=7.5, color=TEXT)
        ax2.tick_params(colors=TEXT, length=0)
        for sp in ("top", "right", "bottom", "left"):
            ax2.spines[sp].set_color(GRID)

    else:
        ax_dims.text(0.5, 0.5, "Waiting for trials…",
                     ha="center", va="center", color=TEXT,
                     fontsize=12, transform=ax_dims.transAxes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="GARAGE live score dashboard")
    ap.add_argument("--study",     default="garage_financebench_s2")
    ap.add_argument("--interval",  type=int, default=30,
                    help="Refresh interval in seconds (default: 30)")
    ap.add_argument("--no-window", action="store_true",
                    help="Headless mode: save PNG only, don't open a window")
    args = ap.parse_args()

    if args.no_window:
        matplotlib.use("Agg")
    else:
        try:
            matplotlib.use("MacOSX")
        except Exception:
            matplotlib.use("TkAgg")

    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    fig.suptitle(
        f"GARAGE BO — study: {args.study}",
        color=TEXT, fontsize=13, y=0.98,
    )

    # Layout: 2 cols on top, 1 wide on bottom
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.32,
                          left=0.07, right=0.97, top=0.93, bottom=0.10)
    ax_prog   = fig.add_subplot(gs[0, 0])
    ax_pareto = fig.add_subplot(gs[0, 1])
    ax_dims   = fig.add_subplot(gs[1, :])

    axes = (ax_prog, ax_pareto, ax_dims)

    def _update(_frame=None):
        draw(fig, axes, args.study)
        fig.savefig(str(SAVE_PATH), dpi=130, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        if not args.no_window:
            fig.canvas.draw_idle()

    # First draw immediately
    _update()

    if args.no_window:
        import time
        print(f"[plot_live] headless mode — saving to {SAVE_PATH} every {args.interval}s")
        while True:
            time.sleep(args.interval)
            _update()
            print(f"[plot_live] updated {SAVE_PATH}", flush=True)
    else:
        ani = animation.FuncAnimation(
            fig, _update,
            interval=args.interval * 1000,
            cache_frame_data=False,
        )
        plt.show()


if __name__ == "__main__":
    main()
