"""
plot_runs.py — Plot composite scores vs run with experiment labels.
"""
import sqlite3, json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

DB_PATH     = Path(__file__).parent / "data" / "results.db"
OUTPUT_PATH = Path(__file__).parent / "results" / "scores_vs_run.png"

conn = sqlite3.connect(str(DB_PATH))
rows = conn.execute("""
    SELECT run_id, composite_score, retrieval_relevance, answer_relevance,
           groundedness, bertscore_f1, latency_ms, config_json
    FROM runs ORDER BY started_at
""").fetchall()
conn.close()

# ---------------------------------------------------------------------------
# Build label: 2-3 word summary based on distinguishing config choices
# ---------------------------------------------------------------------------

def make_label(cfg, run_id):
    retrieval = cfg.get("retrieval_mode", "?")
    reranker  = cfg.get("reranker", "none")
    query     = cfg.get("query_strategy", "verbatim")
    chunk     = cfg.get("chunk_strategy", "?")

    # Retrieval shorthand
    ret_map = {
        "hybrid_rrf": "HybridRRF",
        "hybrid_cc":  "HybridCC",
        "dense_only": "Dense",
        "bm25_only":  "BM25",
    }
    ret = ret_map.get(retrieval, retrieval)

    # Reranker shorthand
    rer_map = {
        "none": "",
        "cross_encoder_minilm": "+CE-MiniLM",
        "cross_encoder_bge":    "+CE-BGE",
        "rankgpt":              "+RankGPT",
    }
    rer = rer_map.get(reranker, "")

    # Query strategy shorthand
    q_map = {
        "verbatim":    "",
        "hyde":        "+HyDE",
        "step_back":   "+StepBack",
        "decompose":   "+Decompose",
        "multi_query": "+MultiQ",
        "keyword":     "+KW",
    }
    q = q_map.get(query, "")

    # Chunk shorthand
    c_map = {
        "recursive": "recur",
        "fixed":     "fixed",
        "semantic":  "sem",
        "sentence":  "sent",
        "paragraph": "para",
    }
    c = c_map.get(chunk, chunk)

    label = f"{ret}{rer}{q}\n({c})"
    # Special label for baseline
    if "baseline" in run_id:
        label = "Baseline\n(hybrid_rrf)"
    return label

# ---------------------------------------------------------------------------
# Collect data
# ---------------------------------------------------------------------------

run_labels   = []
composites   = []
sub_ret      = []
sub_ans      = []
sub_grnd     = []
sub_bert     = []
latencies    = []
is_baseline  = []

FAILED_SCORE = 0.598   # trials that hit the rrf_k=None bug — mark differently

for run_id, comp, ret, ans, grnd, bert, lat, cfg_json in rows:
    cfg = json.loads(cfg_json) if cfg_json else {}
    label = make_label(cfg, run_id)
    run_labels.append(label)
    composites.append(comp)
    sub_ret.append(ret)
    sub_ans.append(ans)
    sub_grnd.append(grnd)
    sub_bert.append(bert)
    latencies.append(lat / 1000)   # → seconds
    is_baseline.append("baseline" in run_id)

x = np.arange(len(run_labels))
failed = [abs(c - FAILED_SCORE) < 0.001 for c in composites]

# ---------------------------------------------------------------------------
# Figure: two panels — composite scores + sub-scores stacked bar
# ---------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), gridspec_kw={"height_ratios": [2, 1]})
fig.patch.set_facecolor("#0f1117")
for ax in (ax1, ax2):
    ax.set_facecolor("#0f1117")
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#ccc")

# ---------------------------------------------------------------------------
# Panel 1: composite score line + scatter
# ---------------------------------------------------------------------------

# Running best line
running_best = []
cur_best = 0.0
for c in composites:
    cur_best = max(cur_best, c)
    running_best.append(cur_best)

# Color each point
colors = []
for i, (c, fail, base) in enumerate(zip(composites, failed, is_baseline)):
    if base:
        colors.append("#f0a500")       # gold = baseline
    elif fail:
        colors.append("#ff4d4d")       # red = failed (bug)
    elif c == max(composites):
        colors.append("#00e676")       # bright green = best
    elif c >= 0.93:
        colors.append("#69f0ae")       # light green = excellent
    elif c >= 0.90:
        colors.append("#40c4ff")       # blue = good
    else:
        colors.append("#b0bec5")       # grey = average

ax1.plot(x, running_best, color="#555", linewidth=1.2,
         linestyle="--", zorder=1, label="Running best")
ax1.plot(x, composites, color="#333", linewidth=0.8, zorder=2, alpha=0.5)
sc = ax1.scatter(x, composites, c=colors, s=90, zorder=5, edgecolors="#222", linewidths=0.5)

# Horizontal reference line at baseline
ax1.axhline(composites[0], color="#f0a500", linewidth=0.8, linestyle=":", alpha=0.6)

# Annotate each point with score
for i, (xi, yi) in enumerate(zip(x, composites)):
    offset = 0.007 if not failed[i] else -0.022
    ax1.annotate(f"{yi:.3f}", (xi, yi + offset),
                 fontsize=6.5, ha="center", color="#ddd", va="bottom")

ax1.set_ylabel("Composite Score", color="#ccc", fontsize=11)
ax1.set_title("GARAGE — Composite Score per Run", color="#eee", fontsize=14, pad=12)
ax1.set_ylim(0.45, 1.02)
ax1.set_xlim(-0.8, len(x) - 0.2)
ax1.set_xticks(x)
ax1.set_xticklabels(run_labels, fontsize=6.5, rotation=35, ha="right", color="#ccc")
ax1.yaxis.label.set_color("#ccc")

# Legend
legend_patches = [
    mpatches.Patch(color="#f0a500",  label="Baseline"),
    mpatches.Patch(color="#00e676",  label="Best run"),
    mpatches.Patch(color="#69f0ae",  label="≥ 0.93"),
    mpatches.Patch(color="#40c4ff",  label="≥ 0.90"),
    mpatches.Patch(color="#b0bec5",  label="< 0.90"),
    mpatches.Patch(color="#ff4d4d",  label="Failed (bug)"),
]
ax1.legend(handles=legend_patches, fontsize=8, loc="lower right",
           facecolor="#1a1d27", edgecolor="#555", labelcolor="#ccc")

# ---------------------------------------------------------------------------
# Panel 2: stacked sub-score breakdown
# ---------------------------------------------------------------------------

W = [0.25, 0.30, 0.25, 0.20]  # weights
bar_w = 0.65

b_ret  = np.array(sub_ret)  * W[0]
b_ans  = np.array(sub_ans)  * W[1]
b_grnd = np.array(sub_grnd) * W[2]
b_bert = np.array(sub_bert) * W[3]

ax2.bar(x, b_ret,  width=bar_w, label="Retrieval rel (×0.25)", color="#4fc3f7", alpha=0.9)
ax2.bar(x, b_ans,  width=bar_w, bottom=b_ret,
        label="Answer rel (×0.30)", color="#81c784", alpha=0.9)
ax2.bar(x, b_grnd, width=bar_w, bottom=b_ret + b_ans,
        label="Groundedness (×0.25)", color="#ffb74d", alpha=0.9)
ax2.bar(x, b_bert, width=bar_w, bottom=b_ret + b_ans + b_grnd,
        label="BERTScore F1 (×0.20)", color="#ce93d8", alpha=0.9)

ax2.set_ylabel("Weighted contribution", color="#ccc", fontsize=10)
ax2.set_ylim(0, 1.05)
ax2.set_xlim(-0.8, len(x) - 0.2)
ax2.set_xticks(x)
ax2.set_xticklabels(run_labels, fontsize=6.5, rotation=35, ha="right", color="#ccc")
ax2.legend(fontsize=8, loc="lower right", facecolor="#1a1d27",
           edgecolor="#555", labelcolor="#ccc", ncol=2)

fig.tight_layout(h_pad=2.5)
OUTPUT_PATH.parent.mkdir(exist_ok=True)
fig.savefig(str(OUTPUT_PATH), dpi=160, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"Saved → {OUTPUT_PATH}")
plt.show()
