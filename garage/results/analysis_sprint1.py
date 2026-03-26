"""
Sprint 1 — Full Analysis & Plots
Run: .venv/bin/python results/analysis_sprint1.py
"""
import optuna
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT = Path(__file__).parent
GARAGE = Path(__file__).parent.parent

# ── Load data ─────────────────────────────────────────────────────────────────
study = optuna.load_study(
    study_name="garage_financebench",
    storage=f"sqlite:///{GARAGE}/data/results.db",
)
complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
rows = []
for t in complete:
    row = dict(t.params)
    row["trial"] = t.number
    row["score"] = t.values[0]
    row["latency_ms"] = t.values[1]
    rows.append(row)
df = pd.DataFrame(rows).sort_values("trial").reset_index(drop=True)
BASELINE = 0.4996

print(f"Loaded {len(df)} complete trials")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Score over trials + rolling mean
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax = axes[0]
ax.scatter(df.trial, df.score, s=18, alpha=0.5, color="steelblue", zorder=3, label="Trial score")
rolling = df.set_index("trial")["score"].rolling(10, min_periods=3).mean()
ax.plot(rolling.index, rolling.values, color="tomato", lw=2, label="10-trial rolling mean")
cummax = df.set_index("trial")["score"].cummax()
ax.plot(cummax.index, cummax.values, color="green", lw=1.5, ls="--", label="Running best")
ax.axhline(BASELINE, color="gray", lw=1.2, ls=":", label=f"Baseline ({BASELINE})")
ax.set_ylabel("Composite Score")
ax.set_title("Sprint 1 — Score Progression (155 trials)", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.set_ylim(0.25, 0.82)
ax.grid(axis="y", alpha=0.3)

ax2 = axes[1]
ax2.scatter(df.trial, df.latency_ms / 1000, s=14, alpha=0.4, color="mediumpurple")
lat_roll = df.set_index("trial")["latency_ms"].rolling(10, min_periods=3).mean() / 1000
ax2.plot(lat_roll.index, lat_roll.values, color="darkorchid", lw=2)
ax2.set_xlabel("Trial number")
ax2.set_ylabel("Latency (s)")
ax2.set_title("Latency per Trial", fontsize=11)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "plot_score_progression.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_score_progression.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Score by categorical params (violin/box plots)
# ══════════════════════════════════════════════════════════════════════════════
cats = [
    ("retrieval_mode",        ["bm25_only", "dense_only", "hybrid_cc", "hybrid_rrf"]),
    ("embedding_model",       ["text-embedding-3-small", "text-embedding-3-large"]),
    ("parser",                ["pymupdf", "pdfplumber", "unstructured"]),
    ("chunk_strategy",        ["fixed", "sentence", "paragraph", "semantic", "recursive"]),
    ("query_strategy",        ["keyword", "multi_query", "step_back", "decompose", "hyde", "verbatim"]),
    ("reranker",              ["cross_encoder_minilm", "rankgpt", "cross_encoder_bge", "none"]),
    ("system_prompt_variant", ["variant_1", "variant_2", "variant_3"]),
    ("table_extraction_strategy", ["html", "none", "markdown", "text"]),
    ("index_type",            ["IVF", "Flat", "HNSW"]),
    ("metric",                ["cosine", "l2", "ip"]),
    ("context_format",        ["numbered", "cited", "xml_tagged", "plain"]),
    ("answer_format",         ["structured", "bullet", "freeform"]),
]

fig, axes = plt.subplots(4, 3, figsize=(18, 20))
axes = axes.flatten()

for i, (col, order) in enumerate(cats):
    ax = axes[i]
    order = [v for v in order if v in df[col].unique()]
    data = [df[df[col] == v]["score"].values for v in order]
    means = [d.mean() for d in data]

    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", lw=2),
                    whiskerprops=dict(lw=1.2),
                    flierprops=dict(marker=".", markersize=4, alpha=0.5))

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.85, len(order)))
    sorted_means = sorted(enumerate(means), key=lambda x: x[1])
    rank_colors = {idx: colors[rank] for rank, (idx, _) in enumerate(sorted_means)}

    for j, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(rank_colors[j])
        patch.set_alpha(0.75)

    ax.axhline(BASELINE, color="gray", lw=1, ls=":", alpha=0.7)
    ax.set_xticks(range(1, len(order) + 1))
    short_labels = [str(v).replace("text-embedding-3-", "").replace("cross_encoder_", "ce_")
                    .replace("hybrid_", "h_").replace("variant_", "v") for v in order]
    ax.set_xticklabels(short_labels, fontsize=8, rotation=20, ha="right")
    ax.set_title(col, fontsize=10, fontweight="bold")
    ax.set_ylabel("Score", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0.25, 0.82)

    for j, (d, m) in enumerate(zip(data, means)):
        ax.text(j + 1, 0.27, f"n={len(d)}", ha="center", fontsize=7, color="gray")

plt.suptitle("Score Distribution by Configuration Parameter", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUT / "plot_param_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_param_distributions.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Pareto front (score vs latency)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 7))

# Color by retrieval mode
mode_colors = {
    "dense_only": "steelblue",
    "hybrid_rrf": "tomato",
    "hybrid_cc":  "orange",
    "bm25_only":  "gray",
}
for mode, color in mode_colors.items():
    sub = df[df.retrieval_mode == mode]
    ax.scatter(sub.latency_ms / 1000, sub.score, s=30, alpha=0.6,
               color=color, label=mode, zorder=3)

# Mark top 10
top10 = df.nlargest(10, "score")
ax.scatter(top10.latency_ms / 1000, top10.score, s=120, marker="*",
           color="gold", edgecolors="black", lw=0.8, zorder=5, label="Top 10")
for _, r in top10.iterrows():
    ax.annotate(f"T{int(r.trial)}", (r.latency_ms / 1000, r.score),
                xytext=(5, 3), textcoords="offset points", fontsize=7)

ax.axhline(BASELINE, color="gray", ls=":", lw=1.2, label=f"Baseline")
ax.set_xlabel("Latency (s)", fontsize=12)
ax.set_ylabel("Composite Score", fontsize=12)
ax.set_title("Pareto Front — Score vs Latency (colored by retrieval mode)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "plot_pareto_detailed.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_pareto_detailed.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Heatmap: retrieval_mode × system_prompt_variant
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

pairs = [
    ("retrieval_mode", "system_prompt_variant"),
    ("retrieval_mode", "chunk_strategy"),
    ("chunk_strategy", "query_strategy"),
]

for ax, (row_var, col_var) in zip(axes, pairs):
    pivot = df.pivot_table(values="score", index=row_var, columns=col_var,
                           aggfunc="mean")
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0.45, vmax=0.70, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels([str(c).replace("text-embedding-3-","").replace("variant_","v")
                        for c in pivot.columns], rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels([str(r).replace("hybrid_","h_").replace("dense_","d_")
                        .replace("bm25_","b_") for r in pivot.index], fontsize=9)
    ax.set_title(f"{row_var}\n× {col_var}", fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=7.5, color="black")

plt.suptitle("Mean Score Heatmaps — Parameter Interactions", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "plot_interaction_heatmaps.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_interaction_heatmaps.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Parameter importance (FAnova)
# ══════════════════════════════════════════════════════════════════════════════
imp_score = optuna.importance.get_param_importances(study, target=lambda t: t.values[0])
imp_lat   = optuna.importance.get_param_importances(study, target=lambda t: t.values[1])

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, imp, title, color in [
    (axes[0], imp_score, "Composite Score", "steelblue"),
    (axes[1], imp_lat,   "Latency",         "tomato"),
]:
    params = list(imp.keys())
    values = list(imp.values())
    y = range(len(params))
    ax.barh(list(y), values, color=color, alpha=0.75)
    ax.set_yticks(list(y))
    ax.set_yticklabels(params, fontsize=9)
    ax.set_xlabel("Importance (FAnova)", fontsize=11)
    ax.set_title(f"Parameter Importance → {title}", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for i, v in enumerate(values):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8)

plt.tight_layout()
plt.savefig(OUT / "plot_param_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_param_importance.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 6 — Continuous params vs score (scatter grid)
# ══════════════════════════════════════════════════════════════════════════════
conts = ["chunk_size", "chunk_overlap", "temperature", "bm25_k1", "bm25_b",
         "dense_top_k", "bm25_top_k", "final_top_k", "max_context_tokens"]

fig, axes = plt.subplots(3, 3, figsize=(14, 12))
axes = axes.flatten()

for i, col in enumerate(conts):
    ax = axes[i]
    ax.scatter(df[col], df["score"], s=14, alpha=0.5, color="steelblue")
    # trend line
    z = np.polyfit(df[col].dropna(), df.loc[df[col].notna(), "score"], 1)
    p = np.poly1d(z)
    xs = np.linspace(df[col].min(), df[col].max(), 100)
    ax.plot(xs, p(xs), color="tomato", lw=1.5)
    r = df[col].corr(df["score"])
    ax.set_title(f"{col}  (r={r:+.3f})", fontsize=10, fontweight="bold")
    ax.set_xlabel(col, fontsize=8)
    ax.set_ylabel("Score", fontsize=8)
    ax.axhline(BASELINE, color="gray", ls=":", lw=1, alpha=0.6)
    ax.grid(alpha=0.3)

plt.suptitle("Continuous Parameters vs Score", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "plot_continuous_params.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_continuous_params.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 7 — Score distribution histogram + top config summary
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.hist(df["score"], bins=30, color="steelblue", alpha=0.75, edgecolor="white")
ax.axvline(BASELINE, color="red", lw=2, ls="--", label=f"Baseline ({BASELINE})")
ax.axvline(df["score"].mean(), color="orange", lw=2, ls="-", label=f"Mean ({df['score'].mean():.3f})")
ax.axvline(df["score"].max(), color="green", lw=2, ls="-", label=f"Best ({df['score'].max():.4f})")
pct_above = (df["score"] > BASELINE).mean() * 100
ax.set_xlabel("Composite Score", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title(f"Score Distribution\n{pct_above:.0f}% of trials beat baseline", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

# Score percentile buckets
ax2 = axes[1]
buckets = [(0.0, 0.5, "<0.50\n(below baseline)"),
           (0.5, 0.6, "0.50–0.60"),
           (0.6, 0.65, "0.60–0.65"),
           (0.65, 0.70, "0.65–0.70"),
           (0.70, 0.75, "0.70–0.75"),
           (0.75, 1.0, ">0.75\n(top tier)")]
counts = [((df.score >= lo) & (df.score < hi)).sum() for lo, hi, _ in buckets]
labels = [l for _, _, l in buckets]
colors = ["#d73027", "#fc8d59", "#fee090", "#91cf60", "#1a9850", "#004529"]
bars = ax2.bar(range(len(buckets)), counts, color=colors, edgecolor="white", alpha=0.85)
ax2.set_xticks(range(len(buckets)))
ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel("Number of trials", fontsize=11)
ax2.set_title("Trials by Score Bucket", fontsize=12, fontweight="bold")
for bar, cnt in zip(bars, counts):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             str(cnt), ha="center", fontsize=10, fontweight="bold")
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "plot_score_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_score_distribution.png")

print("\nAll plots saved to", OUT)
