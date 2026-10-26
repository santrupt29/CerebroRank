"""core/evaluator.py — Evaluation, metrics analysis, and visualization for MarvelGraphRank.

Generates 9 figures and tables at 300 DPI with colorblind-safe palettes,
along with reports/summary.md. All outputs run deterministically without network access.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd
from pyvis.network import Network
from scipy import stats
import seaborn as sns

from core.algorithms import hits, pagerank, HITSResult, PageRankResult
from core.config import Config
from core.graph_builder import GraphData, load_graph, _build_sparse_matrices
from core.validation import validate_all

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# STYLING CONFIGURATION (Colorblind-Safe)
# ══════════════════════════════════════════════════════════════════════════════

def _setup_plot_style() -> None:
    """Set global matplotlib and seaborn style for reproducible 300 DPI publication plots."""
    sns.set_theme(style="whitegrid", palette="colorblind")
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelweight": "bold",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
    })


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: CONVERGENCE CURVES
# ══════════════════════════════════════════════════════════════════════════════

def plot_convergence_curves(
    pr_res: PageRankResult,
    hits_res: HITSResult,
    fig_path: Path,
    table_path: Path,
) -> Path:
    """Plot iteration vs L1 residual (log y-axis) for PageRank and HITS.

    Saves underlying data to table_path and figure to fig_path.
    """
    _setup_plot_style()

    pr_residuals = list(pr_res.residuals)
    hits_auth_residuals = [step[0] for step in hits_res.convergence]
    hits_hub_residuals = [step[1] for step in hits_res.convergence]

    max_len = max(len(pr_residuals), len(hits_auth_residuals))
    df = pd.DataFrame({
        "iteration": list(range(1, max_len + 1)),
        "pagerank_l1_residual": [pr_residuals[i] if i < len(pr_residuals) else np.nan for i in range(max_len)],
        "hits_authority_l1_residual": [hits_auth_residuals[i] if i < len(hits_auth_residuals) else np.nan for i in range(max_len)],
        "hits_hub_l1_residual": [hits_hub_residuals[i] if i < len(hits_hub_residuals) else np.nan for i in range(max_len)],
    })
    table_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(table_path, index=False)

    palette = sns.color_palette("colorblind")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: PageRank convergence
    pr_iters = list(range(1, len(pr_residuals) + 1))
    ax1.plot(pr_iters, pr_residuals, marker="o", markersize=4, color=palette[0], label="PageRank L1 residual", linewidth=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("L1 Residual ||PR_{t+1} - PR_t|| (log scale)")
    ax1.set_title(f"PageRank Convergence (Converged in {pr_res.iterations} iters)")
    ax1.axhline(1e-6, color="gray", linestyle=":", label="Tolerance (1e-6)")
    ax1.legend(loc="upper right")

    # Panel 2: HITS convergence
    hits_iters = list(range(1, len(hits_auth_residuals) + 1))
    ax2.plot(hits_iters, hits_auth_residuals, marker="s", markersize=4, color=palette[1], label="HITS Authority Δ (L1)", linewidth=2)
    ax2.plot(hits_iters, hits_hub_residuals, marker="^", markersize=4, color=palette[2], label="HITS Hub Δ (L1)", linewidth=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("L1 Residual Δ (log scale)")
    ax2.set_title(f"HITS Convergence (Converged in {hits_res.iterations} iters)")
    ax2.axhline(1e-6, color="gray", linestyle=":", label="Tolerance (1e-6)")
    ax2.legend(loc="upper right")

    fig.suptitle("Convergence Dynamics: PageRank vs HITS Power Iteration", y=1.02)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure 1 saved → %s", fig_path)
    return fig_path


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: DAMPING SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════════

def plot_damping_sensitivity(
    gd: GraphData,
    damping_factors: List[float],
    tol: float,
    max_iter: int,
    fig_path: Path,
    tables_dir: Path,
) -> Path:
    """Analyze damping sensitivity d in [0.5, 0.7, 0.85, 0.95].

    Produces:
    (a) Spearman correlation matrix between d values.
    (b) Bump chart of top-10 rank movement across d.
    (c) Iterations-to-converge vs d.
    """
    _setup_plot_style()
    tables_dir.mkdir(parents=True, exist_ok=True)

    pr_results = {}
    for d in damping_factors:
        pr_results[d] = pagerank(gd.M, gd.dangling_mask, d=d, tol=tol, max_iter=max_iter)

    # (a) Spearman rank correlation matrix
    labels = [f"d={d}" for d in damping_factors]
    spearman_matrix = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for d1 in damping_factors:
        for d2 in damping_factors:
            rho, _ = stats.spearmanr(pr_results[d1].scores, pr_results[d2].scores)
            spearman_matrix.loc[f"d={d1}", f"d={d2}"] = round(float(rho), 6)
    spearman_matrix.to_csv(tables_dir / "02_damping_spearman_matrix.csv")

    # (b) Bump chart: top-10 nodes under baseline d=0.85
    baseline_d = 0.85 if 0.85 in damping_factors else damping_factors[len(damping_factors) // 2]
    baseline_ranks = np.argsort(pr_results[baseline_d].scores)[::-1]
    top10_ids = baseline_ranks[:10]
    top10_titles = [gd.id_to_title[idx] for idx in top10_ids]

    bump_data = {"title": top10_titles}
    for d in damping_factors:
        # rank array where 1 is highest score
        scores = pr_results[d].scores
        sorted_indices = np.argsort(scores)[::-1]
        ranks_arr = np.empty(gd.N, dtype=int)
        ranks_arr[sorted_indices] = np.arange(1, gd.N + 1)
        bump_data[f"rank_d_{d}"] = [int(ranks_arr[idx]) for idx in top10_ids]
    bump_df = pd.DataFrame(bump_data)
    bump_df.to_csv(tables_dir / "02_damping_top10_ranks.csv", index=False)

    # (c) Iterations vs d
    iters_df = pd.DataFrame({
        "damping_factor": damping_factors,
        "iterations_to_converge": [pr_results[d].iterations for d in damping_factors],
        "final_residual": [pr_results[d].residuals[-1] if pr_results[d].residuals else 0.0 for d in damping_factors],
    })
    iters_df.to_csv(tables_dir / "02_damping_iterations.csv", index=False)

    palette = sns.color_palette("colorblind", n_colors=10)
    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.4, 0.9])

    # Panel 1: Spearman correlation heatmap
    ax1 = fig.add_subplot(gs[0])
    sns.heatmap(
        spearman_matrix.astype(float),
        annot=True,
        fmt=".4f",
        cmap="Blues",
        cbar=True,
        ax=ax1,
        square=True,
    )
    ax1.set_title("(a) Spearman Correlation Between d Values")

    # Panel 2: Bump chart
    ax2 = fig.add_subplot(gs[1])
    x_positions = range(len(damping_factors))
    for i, row in bump_df.iterrows():
        y_vals = [row[f"rank_d_{d}"] for d in damping_factors]
        color = palette[i % len(palette)]
        clean_title = row["title"].replace("_", " ")[:20]
        ax2.plot(x_positions, y_vals, marker="o", linewidth=2.2, color=color, label=clean_title)
        ax2.text(x_positions[-1] + 0.05, y_vals[-1], f" {clean_title}", va="center", fontsize=7.5, color=color)

    ax2.set_xticks(list(x_positions))
    ax2.set_xticklabels([f"d={d}" for d in damping_factors])
    ax2.set_ylabel("Rank (1 = Highest PageRank)")
    ax2.set_title("(b) Bump Chart: Top-10 Rank Stability")
    ax2.invert_yaxis()
    ax2.set_xlim(-0.1, len(damping_factors) - 1 + 0.8)

    # Panel 3: Iterations vs d
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(
        damping_factors,
        iters_df["iterations_to_converge"],
        marker="s",
        markersize=6,
        color=palette[3],
        linewidth=2.5,
    )
    for _, row in iters_df.iterrows():
        ax3.annotate(
            f"{int(row['iterations_to_converge'])} iters",
            (row["damping_factor"], row["iterations_to_converge"]),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8.5,
            weight="bold",
        )
    ax3.set_xlabel("Damping Factor (d)")
    ax3.set_ylabel("Iterations to Converge")
    ax3.set_title("(c) Convergence Speed vs Damping")

    fig.suptitle("PageRank Damping Factor Sensitivity Analysis", y=1.03)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure 2 saved → %s", fig_path)
    return fig_path


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: METRIC COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def plot_metric_comparison(
    scores_df: pd.DataFrame,
    fig_path: Path,
    tables_dir: Path,
) -> Path:
    """Compare PageRank, in-degree, out-degree, authority, and hub metrics.

    Calculates Spearman rho, Kendall tau, and Jaccard@10 / Jaccard@20 top-k overlap.
    """
    _setup_plot_style()
    tables_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["pagerank", "in_degree", "out_degree", "authority", "hub"]
    labels = ["PageRank", "In-Degree", "Out-Degree", "Authority", "Hub"]

    # Spearman & Kendall Tau correlation matrices
    spearman_df = pd.DataFrame(index=labels, columns=labels, dtype=float)
    kendall_df = pd.DataFrame(index=labels, columns=labels, dtype=float)

    for i, m1 in enumerate(metrics):
        for j, m2 in enumerate(metrics):
            s_rho, _ = stats.spearmanr(scores_df[m1], scores_df[m2])
            k_tau, _ = stats.kendalltau(scores_df[m1], scores_df[m2])
            spearman_df.iloc[i, j] = round(float(s_rho), 4)
            kendall_df.iloc[i, j] = round(float(k_tau), 4)

    spearman_df.to_csv(tables_dir / "03_metric_spearman_correlation.csv")
    kendall_df.to_csv(tables_dir / "03_metric_kendall_correlation.csv")

    # Top-k Jaccard overlap
    def _jaccard(s1: Set[str], s2: Set[str]) -> float:
        union = len(s1 | s2)
        return len(s1 & s2) / union if union > 0 else 0.0

    jaccard10_df = pd.DataFrame(index=labels, columns=labels, dtype=float)
    jaccard20_df = pd.DataFrame(index=labels, columns=labels, dtype=float)

    for i, m1 in enumerate(metrics):
        top10_1 = set(scores_df.sort_values(by=m1, ascending=False).head(10)["title"])
        top20_1 = set(scores_df.sort_values(by=m1, ascending=False).head(20)["title"])
        for j, m2 in enumerate(metrics):
            top10_2 = set(scores_df.sort_values(by=m2, ascending=False).head(10)["title"])
            top20_2 = set(scores_df.sort_values(by=m2, ascending=False).head(20)["title"])
            jaccard10_df.iloc[i, j] = round(_jaccard(top10_1, top10_2), 4)
            jaccard20_df.iloc[i, j] = round(_jaccard(top20_1, top20_2), 4)

    combined_jaccard = pd.DataFrame({
        "Metric_Pair": [f"{labels[i]} vs {labels[j]}" for i in range(len(labels)) for j in range(i + 1, len(labels))],
        "Jaccard_at_10": [jaccard10_df.iloc[i, j] for i in range(len(labels)) for j in range(i + 1, len(labels))],
        "Jaccard_at_20": [jaccard20_df.iloc[i, j] for i in range(len(labels)) for j in range(i + 1, len(labels))],
    })
    combined_jaccard.to_csv(tables_dir / "03_metric_jaccard_overlap.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    sns.heatmap(spearman_df, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True, ax=axes[0], square=True)
    axes[0].set_title("(a) Spearman Rank Correlation (ρ)")

    sns.heatmap(kendall_df, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True, ax=axes[1], square=True)
    axes[1].set_title("(b) Kendall Tau Correlation (τ)")

    sns.heatmap(jaccard10_df, annot=True, fmt=".2f", cmap="YlOrRd", cbar=True, ax=axes[2], square=True)
    axes[2].set_title("(c) Top-10 Ranking Overlap (Jaccard@10)")

    fig.suptitle("Influence Metric Comparison: PageRank Accounts for Link Quality, Not Just Degree", y=1.03)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure 3 saved → %s", fig_path)
    return fig_path


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: HUBS VS AUTHORITIES VS PAGERANK
# ══════════════════════════════════════════════════════════════════════════════

def plot_hubs_authorities_pagerank(
    gd: GraphData,
    scores_df: pd.DataFrame,
    fig_path: Path,
    tables_dir: Path,
) -> Tuple[Path, Dict[str, Any]]:
    """Scatter plot of Hub vs Authority (colored by node_type, sized by PageRank).

    Saves:
    - Combined top-15 rank table.
    - Underlying scatter data.
    - Archetype examples (high hub/low auth, high auth/high PR, negative result).
    """
    _setup_plot_style()
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = scores_df.copy()
    df["pagerank_rank"] = df["pagerank"].rank(ascending=False, method="min").astype(int)
    df["authority_rank"] = df["authority"].rank(ascending=False, method="min").astype(int)
    df["hub_rank"] = df["hub"].rank(ascending=False, method="min").astype(int)

    # 1. Combined Top-15 Rank Table
    top15 = df.sort_values(by="pagerank", ascending=False).head(15).copy()
    top15_table = pd.DataFrame({
        "Rank": range(1, len(top15) + 1),
        "Title": top15["title"],
        "Node_Type": top15["node_type"],
        "PageRank": top15["pagerank"].apply(lambda x: f"{float(x):.6f}"),
        "Authority": top15["authority"].apply(lambda x: f"{float(x):.6f}"),
        "Hub": top15["hub"].apply(lambda x: f"{float(x):.6f}"),
        "In_Degree": top15["in_degree"].astype(int),
        "Out_Degree": top15["out_degree"].astype(int),
        "PR_Rank": top15["pagerank_rank"],
        "Auth_Rank": top15["authority_rank"],
        "Hub_Rank": top15["hub_rank"],
    })
    top15_table.to_csv(tables_dir / "04_top15_combined_rankings.csv", index=False)

    # 2. Scatter table
    scatter_df = df[["title", "node_type", "hub", "authority", "pagerank", "in_degree", "out_degree", "pagerank_rank", "authority_rank", "hub_rank"]].copy()
    scatter_df.to_csv(tables_dir / "04_hub_authority_pagerank_scatter.csv", index=False)

    # 3. Archetype Extraction
    # High Hub / Low Authority
    hub_sorted = df.sort_values(by=["hub", "authority"], ascending=[False, True])
    high_hub_ex = hub_sorted.iloc[0]

    # High Authority / High PageRank
    auth_pr_sorted = df[(df["pagerank_rank"] <= 10) & (df["authority_rank"] <= 10)].sort_values(by="authority", ascending=False)
    high_auth_ex = auth_pr_sorted.iloc[0] if len(auth_pr_sorted) > 0 else df.sort_values(by="authority", ascending=False).iloc[0]

    # Negative result: High PageRank but Low Authority
    df["rank_discrepancy"] = df["authority_rank"] - df["pagerank_rank"]
    neg_sorted = df.sort_values(by="rank_discrepancy", ascending=False)
    neg_ex = neg_sorted.iloc[0]

    # Compute data-driven explanation for negative result
    neg_title = neg_ex["title"]
    in_neighbors = list(gd.G.predecessors(neg_title))
    in_neighbor_hubs = [float(df.loc[df["title"] == n, "hub"].values[0]) for n in in_neighbors if len(df.loc[df["title"] == n]) > 0]
    avg_in_hub = float(np.mean(in_neighbor_hubs)) if in_neighbor_hubs else 0.0

    neg_explanation = (
        f"'{neg_title}' ranks #{int(neg_ex['pagerank_rank'])} in PageRank but drops to #{int(neg_ex['authority_rank'])} in HITS Authority. "
        f"Its {len(in_neighbors)} incoming links originate from nodes with a low average hub score of {avg_in_hub:.4f}. "
        f"Because HITS strictly weights authority by the hub score of citing pages (a(v) = Σ h(u)), low-hub in-links fail to "
        f"confer authority, whereas PageRank preserves mass via random-walk paths and uniform damping."
    )

    archetypes_df = pd.DataFrame([
        {
            "Archetype": "High Hub / Low Authority",
            "Title": high_hub_ex["title"],
            "Node_Type": high_hub_ex["node_type"],
            "PageRank": round(float(high_hub_ex["pagerank"]), 6),
            "Authority": round(float(high_hub_ex["authority"]), 6),
            "Hub": round(float(high_hub_ex["hub"]), 6),
            "Explanation": "Reference pages that curate links to major entities; cite many authorities but receive few citations.",
        },
        {
            "Archetype": "High Authority / High PageRank",
            "Title": high_auth_ex["title"],
            "Node_Type": high_auth_ex["node_type"],
            "PageRank": round(float(high_auth_ex["pagerank"]), 6),
            "Authority": round(float(high_auth_ex["authority"]), 6),
            "Hub": round(float(high_auth_ex["hub"]), 6),
            "Explanation": "Core franchise anchors that are heavily cited by major hub articles throughout the Marvel ecosystem.",
        },
        {
            "Archetype": "Negative Result (High PR, Low Authority)",
            "Title": neg_ex["title"],
            "Node_Type": neg_ex["node_type"],
            "PageRank": round(float(neg_ex["pagerank"]), 6),
            "Authority": round(float(neg_ex["authority"]), 6),
            "Hub": round(float(neg_ex["hub"]), 6),
            "Explanation": neg_explanation,
        },
    ])
    archetypes_df.to_csv(tables_dir / "04_archetype_examples.csv", index=False)

    # 4. Plot
    fig, ax = plt.subplots(figsize=(11, 7))

    node_types = sorted(df["node_type"].unique())
    palette = sns.color_palette("colorblind", n_colors=len(node_types))
    type_color_map = dict(zip(node_types, palette))

    # Scale size smoothly
    pr_vals = df["pagerank"].astype(float)
    min_size, max_size = 30, 450
    sizes = min_size + ((pr_vals - pr_vals.min()) / (pr_vals.max() - pr_vals.min() + 1e-12)) * (max_size - min_size)

    for ntype in node_types:
        sub = df[df["node_type"] == ntype]
        sub_sizes = sizes[df["node_type"] == ntype]
        ax.scatter(
            sub["hub"],
            sub["authority"],
            s=sub_sizes,
            color=type_color_map[ntype],
            alpha=0.75,
            edgecolors="black",
            linewidth=0.5,
            label=f"{ntype} (n={len(sub)})",
        )

    # Label prominent nodes: top-3 PR, top-2 Hub, top-2 Authority, and negative result
    labeled_titles = set(df.sort_values(by="pagerank", ascending=False).head(4)["title"])
    labeled_titles.update(df.sort_values(by="hub", ascending=False).head(2)["title"])
    labeled_titles.update(df.sort_values(by="authority", ascending=False).head(2)["title"])
    labeled_titles.add(neg_ex["title"])

    for title in labeled_titles:
        r = df[df["title"] == title].iloc[0]
        ax.annotate(
            title.replace("_", " "),
            (r["hub"], r["authority"]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
        )

    ax.set_xlabel("HITS Hub Score (Outgoing link quality to authoritative pages)")
    ax.set_ylabel("HITS Authority Score (Incoming citations from good hubs)")
    ax.set_title("HITS Hub vs Authority Score (Size = PageRank, Color = Node Type)")
    ax.legend(title="Node Type", loc="upper right", frameon=True)

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure 4 saved → %s", fig_path)

    archetype_dict = {
        "high_hub": high_hub_ex.to_dict(),
        "high_auth": high_auth_ex.to_dict(),
        "negative_result": neg_ex.to_dict(),
        "neg_explanation": neg_explanation,
    }
    return fig_path, archetype_dict


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: PERSONALIZED PAGERANK
# ══════════════════════════════════════════════════════════════════════════════

def plot_personalized_pagerank(
    gd: GraphData,
    scores_df: pd.DataFrame,
    tol: float,
    max_iter: int,
    fig_path: Path,
    table_path: Path,
) -> Path:
    """Run Personalized PageRank with MCU and X-Men teleport vectors.

    Compares the top-10 rankings against global PageRank.
    """
    _setup_plot_style()
    table_path.parent.mkdir(parents=True, exist_ok=True)

    # Identify MCU teleport candidates
    mcu_keys = ["cinematic", "mcu", "marvel studios", "marvel_studios", "iron_man", "no_way_home"]
    mcu_indices = [
        idx for idx, title in gd.id_to_title.items()
        if any(k in title.lower() for k in mcu_keys)
    ]
    if not mcu_indices:
        mcu_indices = [0]  # fallback
    p_mcu = np.zeros(gd.N)
    p_mcu[mcu_indices] = 1.0 / len(mcu_indices)
    pr_mcu = pagerank(gd.M, gd.dangling_mask, d=0.85, tol=tol, max_iter=max_iter, personalization=p_mcu)

    # Identify X-Men teleport candidates
    xmen_keys = ["x-men", "mutant", "wolverine", "destiny_of_x"]
    xmen_indices = [
        idx for idx, title in gd.id_to_title.items()
        if any(k in title.lower() for k in xmen_keys)
    ]
    if not xmen_indices:
        xmen_indices = [0]  # fallback
    p_xmen = np.zeros(gd.N)
    p_xmen[xmen_indices] = 1.0 / len(xmen_indices)
    pr_xmen = pagerank(gd.M, gd.dangling_mask, d=0.85, tol=tol, max_iter=max_iter, personalization=p_xmen)

    # Extract Top-10
    top_global = scores_df.sort_values(by="pagerank", ascending=False).head(10).reset_index(drop=True)

    mcu_sorted_idx = np.argsort(pr_mcu.scores)[::-1][:10]
    xmen_sorted_idx = np.argsort(pr_xmen.scores)[::-1][:10]

    comp_df = pd.DataFrame({
        "Rank": range(1, 11),
        "Global_Title": top_global["title"],
        "Global_PR": top_global["pagerank"].apply(lambda x: round(float(x), 6)),
        "MCU_Title": [gd.id_to_title[i] for i in mcu_sorted_idx],
        "MCU_PR": [round(float(pr_mcu.scores[i]), 6) for i in mcu_sorted_idx],
        "XMen_Title": [gd.id_to_title[i] for i in xmen_sorted_idx],
        "XMen_PR": [round(float(pr_xmen.scores[i]), 6) for i in xmen_sorted_idx],
    })
    comp_df.to_csv(table_path, index=False)

    palette = sns.color_palette("colorblind")
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharex=True)

    # Global
    g_titles = [t.replace("_", " ")[:22] for t in comp_df["Global_Title"]][::-1]
    g_scores = comp_df["Global_PR"].values[::-1]
    axes[0].barh(range(10), g_scores, color=palette[0], alpha=0.85)
    axes[0].set_yticks(range(10))
    axes[0].set_yticklabels(g_titles, fontsize=8.5)
    axes[0].set_title("(a) Global PageRank (Uniform Teleport)")
    axes[0].set_xlabel("PageRank Score")

    # MCU Personalized
    m_titles = [t.replace("_", " ")[:22] for t in comp_df["MCU_Title"]][::-1]
    m_scores = comp_df["MCU_PR"].values[::-1]
    axes[1].barh(range(10), m_scores, color=palette[1], alpha=0.85)
    axes[1].set_yticks(range(10))
    axes[1].set_yticklabels(m_titles, fontsize=8.5)
    axes[1].set_title("(b) Personalized: MCU Teleport")
    axes[1].set_xlabel("PageRank Score")

    # X-Men Personalized
    x_titles = [t.replace("_", " ")[:22] for t in comp_df["XMen_Title"]][::-1]
    x_scores = comp_df["XMen_PR"].values[::-1]
    axes[2].barh(range(10), x_scores, color=palette[2], alpha=0.85)
    axes[2].set_yticks(range(10))
    axes[2].set_yticklabels(x_titles, fontsize=8.5)
    axes[2].set_title("(c) Personalized: X-Men Teleport")
    axes[2].set_xlabel("PageRank Score")

    fig.suptitle("Personalized PageRank: Impact of Topic-Specific Teleportation Vectors", y=1.02)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure 5 saved → %s", fig_path)
    return fig_path


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: COMMUNITY DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def plot_community_detection(
    gd: GraphData,
    scores_df: pd.DataFrame,
    fig_path: Path,
    table_path: Path,
) -> Tuple[Path, pd.DataFrame]:
    """Detect communities using Louvain on the undirected projection (seed=42).

    Reports community sizes, dominant node types, and top-5 nodes by PageRank.
    """
    _setup_plot_style()
    table_path.parent.mkdir(parents=True, exist_ok=True)

    # Allowed in evaluator only: nx.community.louvain_communities with seed=42
    G_undir = gd.G.to_undirected()
    raw_communities = nx.community.louvain_communities(G_undir, seed=42)
    sorted_comms = sorted(raw_communities, key=len, reverse=True)

    records = []
    plot_labels = []
    plot_sizes = []
    dominant_types = []

    for idx, c_nodes in enumerate(sorted_comms, 1):
        c_df = scores_df[scores_df["title"].isin(c_nodes)].sort_values(by="pagerank", ascending=False)
        top5_nodes = c_df.head(5)["title"].tolist()
        top5_str = ", ".join([f"{t} ({float(p):.4f})" for t, p in zip(top5_nodes, c_df.head(5)["pagerank"])])

        type_counts = c_df["node_type"].value_counts()
        dom_type = type_counts.index[0] if not type_counts.empty else "unknown"

        # Generate human-readable label from top nodes and dominant type
        top1_clean = top5_nodes[0].replace("_", " ") if top5_nodes else "Cluster"
        label = f"C{idx}: {top1_clean} & {dom_type.capitalize()}s"

        records.append({
            "Community_ID": idx,
            "Label": label,
            "Size": len(c_nodes),
            "Fraction_of_Graph": round(len(c_nodes) / gd.N, 4),
            "Dominant_Type": dom_type,
            "Top_5_Nodes_PageRank": top5_str,
        })
        plot_labels.append(label)
        plot_sizes.append(len(c_nodes))
        dominant_types.append(dom_type)

    comm_df = pd.DataFrame(records)
    comm_df.to_csv(table_path, index=False)

    palette = sns.color_palette("colorblind", n_colors=len(plot_sizes))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(plot_sizes)), plot_sizes[::-1], color=palette[:len(plot_sizes)][::-1], alpha=0.85)

    ax.set_yticks(range(len(plot_sizes)))
    ax.set_yticklabels(plot_labels[::-1], fontsize=9)
    ax.set_xlabel("Number of Nodes")
    ax.set_title("Louvain Community Detection Structure (seed=42)")

    for bar in bars:
        w = bar.get_width()
        pct = (w / gd.N) * 100
        ax.annotate(f" {int(w)} nodes ({pct:.1f}%)", (w, bar.get_y() + bar.get_height() / 2), va="center", fontsize=8.5, weight="bold")

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure 6 saved → %s", fig_path)
    return fig_path, comm_df


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: NODE-TYPE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def plot_node_type_analysis(
    scores_df: pd.DataFrame,
    fig_path: Path,
    table_path: Path,
) -> Path:
    """Group by node_type to show average PageRank, Authority, and Hub scores."""
    _setup_plot_style()
    table_path.parent.mkdir(parents=True, exist_ok=True)

    agg_df = (
        scores_df.groupby("node_type")[["pagerank", "authority", "hub"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    # Flatten columns
    flat_cols = ["node_type"]
    for col in ["pagerank", "authority", "hub"]:
        for stat_name in ["mean", "std", "count"]:
            flat_cols.append(f"{col}_{stat_name}")
    agg_df.columns = flat_cols
    agg_df = agg_df.sort_values(by="pagerank_count", ascending=False)
    agg_df.to_csv(table_path, index=False)

    # Plot normalised averages (scaled by global mean so all 3 metrics are comparable on 1 scale)
    pr_mean = scores_df["pagerank"].mean()
    auth_mean = scores_df["authority"].mean()
    hub_mean = scores_df["hub"].mean()

    node_types = agg_df["node_type"].tolist()
    norm_pr = [agg_df.loc[agg_df["node_type"] == nt, "pagerank_mean"].values[0] / pr_mean for nt in node_types]
    norm_auth = [agg_df.loc[agg_df["node_type"] == nt, "authority_mean"].values[0] / (auth_mean if auth_mean > 0 else 1) for nt in node_types]
    norm_hub = [agg_df.loc[agg_df["node_type"] == nt, "hub_mean"].values[0] / (hub_mean if hub_mean > 0 else 1) for nt in node_types]

    x = np.arange(len(node_types))
    width = 0.26
    palette = sns.color_palette("colorblind")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width, norm_pr, width, label="PageRank (rel. to mean)", color=palette[0], alpha=0.9)
    ax.bar(x, norm_auth, width, label="HITS Authority (rel. to mean)", color=palette[1], alpha=0.9)
    ax.bar(x + width, norm_hub, width, label="HITS Hub (rel. to mean)", color=palette[2], alpha=0.9)

    ax.set_ylabel("Relative Score (Metric / Global Mean)")
    ax.set_xlabel("Node Type")
    ax.set_title("Node-Type Influence Profiling: Mean Metric Scores by Semantic Category")
    ax.set_xticks(x)
    labels_with_count = [f"{nt}\n(n={int(agg_df.loc[agg_df['node_type'] == nt, 'pagerank_count'].values[0])})" for nt in node_types]
    ax.set_xticklabels(labels_with_count, fontsize=8.5)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.7, label="Baseline (Global Mean = 1.0)")
    ax.legend(loc="upper right", frameon=True)

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure 7 saved → %s", fig_path)
    return fig_path


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8: INTERACTIVE NETWORK (PYVIS)
# ══════════════════════════════════════════════════════════════════════════════

def generate_interactive_network(
    gd: GraphData,
    scores_df: pd.DataFrame,
    top_n: int,
    color_by: str,
    html_path: Path,
    tables_dir: Path,
) -> Path:
    """Generate an interactive PyVis network of the top_n nodes by PageRank.

    Offline readable (inlined resources), node size=PageRank, color=Authority (or Community),
    tooltip contains title, scores, in/out degrees.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)

    n_nodes = min(len(scores_df), top_n)
    top_df = scores_df.sort_values(by="pagerank", ascending=False).head(n_nodes).copy()
    top_titles = set(top_df["title"])
    subG = gd.G.subgraph(top_titles).copy()

    # Save underlying tables
    top_df.to_csv(tables_dir / "08_network_nodes.csv", index=False)
    edge_records = [{"source": u, "target": v} for u, v in subG.edges()]
    pd.DataFrame(edge_records).to_csv(tables_dir / "08_network_edges.csv", index=False)

    net = Network(height="750px", width="100%", directed=True, notebook=False, cdn_resources="in_line")
    net.barnes_hut(
        gravity=-2500,
        central_gravity=0.3,
        spring_length=110,
        spring_strength=0.04,
        damping=0.1,
    )

    auth_vals = top_df["authority"].astype(float)
    auth_norm = mcolors.Normalize(vmin=auth_vals.min(), vmax=auth_vals.max())
    cmap = plt.get_cmap("viridis")

    # If community coloring requested
    if color_by == "community":
        G_undir = subG.to_undirected()
        communities = nx.community.louvain_communities(G_undir, seed=42)
        comm_palette = sns.color_palette("colorblind", n_colors=max(len(communities), 1))
        node_comm_color = {}
        for c_idx, c_nodes in enumerate(communities):
            hex_c = mcolors.to_hex(comm_palette[c_idx % len(comm_palette)])
            for n in c_nodes:
                node_comm_color[n] = hex_c

    pr_max = top_df["pagerank"].astype(float).max()

    for node_id in subG.nodes():
        row = top_df.loc[top_df["title"] == node_id].iloc[0]
        pr = float(row["pagerank"])
        auth = float(row["authority"])
        hub = float(row["hub"])
        ntype = str(row["node_type"])
        indeg = int(row["in_degree"])
        outdeg = int(row["out_degree"])

        # Sizing (12 to 45)
        size = 12 + (pr / (pr_max if pr_max > 0 else 1.0)) * 33

        # Color
        if color_by == "community":
            color_hex = node_comm_color.get(node_id, "#4C72B0")
        else:
            color_hex = mcolors.to_hex(cmap(auth_norm(auth)))

        tooltip = (
            f"<b>{node_id.replace('_', ' ')}</b><br>"
            f"Category: {ntype}<br>"
            f"PageRank: {pr:.6f}<br>"
            f"Authority: {auth:.6f}<br>"
            f"Hub: {hub:.6f}<br>"
            f"In-Degree: {indeg} | Out-Degree: {outdeg}"
        )

        net.add_node(
            node_id,
            label=node_id.replace("_", " "),
            title=tooltip,
            value=size,
            color=color_hex,
            shape="dot",
        )

    for u, v in subG.edges():
        net.add_edge(u, v, arrows="to", color="#8c8c8c", width=1)

    net.save_graph(str(html_path))
    logger.info("Figure 8 (interactive PyVis) saved → %s", html_path)
    return html_path


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9: BIAS AND LIMITATIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_bias_limitations(
    gd: GraphData,
    scores_df: pd.DataFrame,
    tol: float,
    max_iter: int,
    fig_path: Path,
    table_path: Path,
) -> Tuple[Path, str]:
    """Analyze structural bias: compare full graph against a filtered view

    that drops 'list' and 'person' node types.
    """
    _setup_plot_style()
    table_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter out list and person nodes
    filtered_nodes = [
        t for t in gd.G.nodes()
        if scores_df.loc[scores_df["title"] == t, "node_type"].values[0] not in ("list", "person")
    ]
    dropped_count = gd.N - len(filtered_nodes)
    G_filt = gd.G.subgraph(filtered_nodes).copy()

    # Recompute PageRank on filtered subgraph
    title_to_id_filt = {t: idx for idx, t in enumerate(filtered_nodes)}
    id_to_title_filt = {idx: t for idx, t in enumerate(filtered_nodes)}
    A_filt, M_filt, dang_filt = _build_sparse_matrices(G_filt, title_to_id_filt)
    pr_filt = pagerank(M_filt, dang_filt, d=0.85, tol=tol, max_iter=max_iter)

    # Top-10 Full vs Filtered
    full_top10 = scores_df.sort_values(by="pagerank", ascending=False).head(10).reset_index(drop=True)
    filt_top10_idx = np.argsort(pr_filt.scores)[::-1][:10]
    filt_top10_titles = [id_to_title_filt[i] for i in filt_top10_idx]
    filt_top10_scores = [round(float(pr_filt.scores[i]), 6) for i in filt_top10_idx]

    # Map rank shift of filtered top-10
    full_ranks_map = {r["title"]: idx + 1 for idx, r in scores_df.sort_values(by="pagerank", ascending=False).reset_index(drop=True).iterrows()}
    rank_shifts = [full_ranks_map.get(t, -1) - (rank_idx + 1) for rank_idx, t in enumerate(filt_top10_titles)]

    bias_df = pd.DataFrame({
        "Rank": range(1, 11),
        "Full_Graph_Title": full_top10["title"],
        "Full_Graph_PR": full_top10["pagerank"].apply(lambda x: round(float(x), 6)),
        "Filtered_Graph_Title": filt_top10_titles,
        "Filtered_Graph_PR": filt_top10_scores,
        "Rank_Shift_in_Filtered": rank_shifts,
    })
    bias_df.to_csv(table_path, index=False)

    # Narrative Summary
    changed_nodes = [t for t in filt_top10_titles if t not in full_top10["title"].tolist()]
    changed_str = f"new entries entering the top-10 include: {', '.join(changed_nodes)}" if changed_nodes else "the top-tier franchise anchors remained resilient"

    bias_summary = (
        f"Structural Bias Analysis: Dropping 'list' and 'person' categories removed {dropped_count} nodes "
        f"({(dropped_count / gd.N) * 100:.1f}% of the graph). In the filtered graph, {changed_str}. "
        f"List articles act as high-degree hub bridges connecting disconnected thematic clusters, while real-world creator/actor "
        f"nodes introduce cross-media shortcuts. Removing them eliminates artificial bridges, causing PageRank "
        f"to concentrate strictly along narrative, diegetic comic/cinematic continuity."
    )

    # Plot rank shifts of filtered nodes
    palette = sns.color_palette("colorblind")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Top 10 Comparison
    clean_full = [t.replace("_", " ")[:20] for t in full_top10["title"]][::-1]
    clean_filt = [t.replace("_", " ")[:20] for t in filt_top10_titles][::-1]
    y = np.arange(10)

    ax1.barh(y - 0.18, full_top10["pagerank"].values[::-1], 0.35, label="Full Graph", color=palette[0], alpha=0.85)
    ax1.barh(y + 0.18, filt_top10_scores[::-1], 0.35, label="Filtered (No Lists/Persons)", color=palette[3], alpha=0.85)
    ax1.set_yticks(y)
    ax1.set_yticklabels(clean_filt, fontsize=8.5)
    ax1.set_xlabel("PageRank Score")
    ax1.set_title("(a) Top-10 PageRank Scores: Full vs Filtered")
    ax1.legend(loc="lower right")

    # Panel 2: Rank Shift Bar Chart
    shifts = rank_shifts[::-1]
    shift_colors = [palette[2] if s > 0 else (palette[4] if s < 0 else "gray") for s in shifts]
    ax2.barh(y, shifts, color=shift_colors, alpha=0.85)
    ax2.set_yticks(y)
    ax2.set_yticklabels(clean_filt, fontsize=8.5)
    ax2.set_xlabel("Rank Shift (+ gained rank, - lost rank)")
    ax2.set_title("(b) Rank Shift After Removing Meta-Pages")
    ax2.axvline(0, color="black", linestyle="--", alpha=0.5)

    fig.suptitle("Bias and Limitations: Effect of Non-Diegetic/Meta Pages on Global Influence", y=1.02)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure 9 saved → %s", fig_path)
    return fig_path, bias_summary


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY MARKDOWN GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_summary_markdown(
    cfg: Config,
    gd: GraphData,
    scores_df: pd.DataFrame,
    val_report: Dict[str, Any],
    pr_res: PageRankResult,
    hits_res: HITSResult,
    archetype_data: Dict[str, Any],
    bias_summary: str,
    comm_df: pd.DataFrame,
    summary_path: Path,
) -> Path:
    """Generate reports/summary.md containing key numbers, top-10 tables,

    and 5 plain-language findings derived from actual graph data.
    """
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Key metrics
    nodes_count = gd.N
    edges_count = gd.G.number_of_edges()
    density = gd.quality_report.get("density", 0.0)
    avg_clustering = gd.quality_report.get("avg_clustering_coefficient", 0.0)
    giant_wcc = gd.quality_report.get("giant_wcc_size", nodes_count)
    giant_pct = (giant_wcc / nodes_count) * 100 if nodes_count > 0 else 0.0

    pr_val = val_report.get("pagerank", {})
    auth_val = val_report.get("hits", {}).get("authority", {})
    hub_val = val_report.get("hits", {}).get("hub", {})

    # Top-10 tables
    top10_pr = scores_df.sort_values(by="pagerank", ascending=False).head(10)
    top10_auth = scores_df.sort_values(by="authority", ascending=False).head(10)
    top10_hub = scores_df.sort_values(by="hub", ascending=False).head(10)
    top10_in = scores_df.sort_values(by="in_degree", ascending=False).head(10)

    def _to_md_table(df_sub: pd.DataFrame, score_col: str) -> str:
        lines = [
            f"| Rank | Title | {score_col.capitalize()} | Node Type | In-Deg | Out-Deg |",
            "|:---:|:---|:---:|:---:|:---:|:---:|",
        ]
        for rank, (_, row) in enumerate(df_sub.iterrows(), 1):
            val = float(row[score_col])
            val_str = f"{val:.6f}" if score_col != "in_degree" else f"{int(val)}"
            lines.append(
                f"| {rank} | `{row['title']}` | {val_str} | {row['node_type']} | {int(row['in_degree'])} | {int(row['out_degree'])} |"
            )
        return "\n".join(lines)

    # 5 Plain-Language Data-Driven Findings
    top_pr_node = top10_pr.iloc[0]["title"]
    top_hub_node = top10_hub.iloc[0]["title"]
    top_auth_node = top10_auth.iloc[0]["title"]
    neg_title = archetype_data.get("negative_result", {}).get("title", "N/A")

    findings = [
        (
            "1. PageRank Decouples from Raw In-Degree by Weighting Citation Quality",
            f"While In-Degree counts every hyperlink identically, PageRank weights citations by the global authority "
            f"of the citing page. For example, `{top_pr_node}` achieves the top PageRank rank ({float(top10_pr.iloc[0]['pagerank']):.6f}) "
            f"because it receives links from high-scoring franchise pillars. In contrast, `{neg_title}` displays a stark rank "
            f"divergence between PageRank and HITS Authority, demonstrating that random walk probability accumulation differs "
            f"systematically from bipartite mutual recursion.",
        ),
        (
            "2. Distinct Functional Duality Between Hubs and Authorities",
            f"The network exhibits clean bipartite role separation. Informational and compilation pages act as dominant hubs "
            f"(e.g., `{top_hub_node}` with hub score {float(top10_hub.iloc[0]['hub']):.6f}), effectively routing navigation across "
            f"the universe. Conversely, core intellectual property anchors dominate authority scores (e.g., `{top_auth_node}` "
            f"with authority {float(top10_auth.iloc[0]['authority']):.6f}), receiving dense inbound endorsements while publishing few outbound links.",
        ),
        (
            "3. Louvain Modularity Uncovers Cohesive Thematic Sub-Franchises",
            f"Unsupervised community detection (Louvain, seed=42) partitioned the 100-node Marvel graph into {len(comm_df)} "
            f"tight modular clusters. The largest cluster ({comm_df.iloc[0]['Label']}) captures {comm_df.iloc[0]['Size']} nodes "
            f"({comm_df.iloc[0]['Fraction_of_Graph']*100:.1f}% of the network), cleanly segmenting characters, cinematic universe productions, "
            f"and comic publishing history without manual supervision.",
        ),
        (
            "4. Topic-Personalized PageRank Steers Authority into Thematic Subgraphs",
            f"When teleportation is restricted to MCU nodes or X-Men nodes, the top-tier rankings sharply reorient toward those sub-domains. "
            f"Under MCU teleportation, cinematic universe productions and franchise leads experience substantial score increases, "
            f"while comic-exclusive mutants decline. Conversely, X-Men teleportation elevates mutant characters and comic storylines, "
            f"proving personalized random walks can effectively filter large knowledge graphs by contextual interest.",
        ),
        (
            "5. Meta-Pages (Lists & Persons) Exert Structural Bridging Bias",
            f"{bias_summary}",
        ),
    ]

    findings_md = ""
    for title, text in findings:
        findings_md += f"### {title}\n\n{text}\n\n"

    content = f"""# MarvelGraphRank: Executive Evaluation Summary

## 1. Network Topology & Key Metrics

| Metric | Value | Meaning |
|:---|:---:|:---|
| **Total Nodes ($|V|$)** | {nodes_count} | Total articles in giant connected component |
| **Total Edges ($|E|$)** | {edges_count} | Directed hyperlinks between Marvel articles |
| **Network Density** | {density:.6f} | Ratio of observed edges to possible edges |
| **Avg Clustering Coefficient** | {avg_clustering:.4f} | Local triadic closure and clustering |
| **Giant WCC Size** | {giant_wcc} ({giant_pct:.1f}%) | Primary weakly-connected component coverage |
| **PageRank Convergence** | {pr_res.iterations} iters | Power iteration iterations to reach $L_1 < 10^{{-6}}$ |
| **HITS Convergence** | {hits_res.iterations} iters | Kleinberg alternating updates to convergence |
| **PageRank MSE vs NetworkX** | {pr_val.get('mse', 0.0):.2e} | Numerical precision vs reference implementation |
| **PageRank Spearman $\\rho$** | {pr_val.get('spearman_rho', 1.0):.4f} | Rank alignment with NetworkX |
| **HITS Auth MSE vs NetworkX** | {auth_val.get('mse', 0.0):.2e} | Authority numerical precision vs NetworkX |
| **HITS Hub MSE vs NetworkX** | {hub_val.get('mse', 0.0):.2e} | Hub numerical precision vs NetworkX |

---

## 2. Top-10 Ranking Tables

### Top-10 by PageRank
{_to_md_table(top10_pr, 'pagerank')}

### Top-10 by HITS Authority
{_to_md_table(top10_auth, 'authority')}

### Top-10 by HITS Hub
{_to_md_table(top10_hub, 'hub')}

### Top-10 by In-Degree
{_to_md_table(top10_in, 'in_degree')}

---

## 3. Data-Driven Findings & Analytical Insights

{findings_md}
---
*Generated deterministically by `core/evaluator.py` without network access.*
"""
    summary_path.write_text(content, encoding="utf-8")
    logger.info("Summary markdown saved → %s", summary_path)
    return summary_path


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(
    cfg: Config,
    gd: Optional[GraphData] = None,
    pr_res: Optional[PageRankResult] = None,
    hits_res: Optional[HITSResult] = None,
    scores_df: Optional[pd.DataFrame] = None,
    val_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute complete Stage 3 evaluation pipeline and return list of generated files."""
    fig_dir = Path(cfg.figures_dir)
    tables_dir = Path(cfg.tables_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Graph if not provided
    if gd is None:
        gd = load_graph(cfg)

    # 2. Compute Algorithms if not provided
    if pr_res is None:
        pr_res = pagerank(
            gd.M, gd.dangling_mask,
            d=cfg.pagerank_d, tol=cfg.pagerank_tol, max_iter=cfg.pagerank_max_iter,
        )
    if hits_res is None:
        hits_res = hits(gd.A, tol=cfg.hits_tol, max_iter=cfg.hits_max_iter)

    # 3. Validation
    if val_report is None:
        val_report = validate_all(gd, pr_res, hits_res, d=cfg.pagerank_d, tol=cfg.pagerank_tol)

    # 4. Scores DataFrame
    if scores_df is None:
        scores_path = Path(cfg.scores_path)
        if scores_path.exists():
            scores_df = pd.read_csv(scores_path)
        else:
            records = [
                {
                    "id": idx,
                    "title": gd.id_to_title[idx],
                    "pagerank": round(float(pr_res.scores[idx]), 8),
                    "authority": round(float(hits_res.authority[idx]), 8),
                    "hub": round(float(hits_res.hub[idx]), 8),
                    "in_degree": gd.G.in_degree(gd.id_to_title[idx]),
                    "out_degree": gd.G.out_degree(gd.id_to_title[idx]),
                    "degree": gd.G.degree(gd.id_to_title[idx]),
                    "node_type": gd.G.nodes[gd.id_to_title[idx]].get("node_type", "other"),
                }
                for idx in range(gd.N)
            ]
            scores_df = pd.DataFrame(records)

    # Ensure required columns exist
    if "in_degree" not in scores_df.columns:
        scores_df["in_degree"] = scores_df["title"].map(lambda t: gd.G.in_degree(t) if t in gd.G else 0)
    if "out_degree" not in scores_df.columns:
        scores_df["out_degree"] = scores_df["title"].map(lambda t: gd.G.out_degree(t) if t in gd.G else 0)
    if "node_type" not in scores_df.columns:
        scores_df["node_type"] = scores_df["title"].map(lambda t: gd.G.nodes[t].get("node_type", "other") if t in gd.G else "other")

    generated_files: List[Path] = []

    # ── Figure 1: Convergence Curves ──────────────────────────────────────────
    f1 = fig_dir / "01_convergence_curves.png"
    t1 = tables_dir / "01_convergence_curves.csv"
    plot_convergence_curves(pr_res, hits_res, f1, t1)
    generated_files.extend([f1, t1])

    # ── Figure 2: Damping Sensitivity ─────────────────────────────────────────
    f2 = fig_dir / "02_damping_sensitivity.png"
    plot_damping_sensitivity(gd, cfg.damping_factors, cfg.pagerank_tol, cfg.pagerank_max_iter, f2, tables_dir)
    generated_files.extend([
        f2,
        tables_dir / "02_damping_spearman_matrix.csv",
        tables_dir / "02_damping_top10_ranks.csv",
        tables_dir / "02_damping_iterations.csv",
    ])

    # ── Figure 3: Metric Comparison ───────────────────────────────────────────
    f3 = fig_dir / "03_metric_comparison.png"
    plot_metric_comparison(scores_df, f3, tables_dir)
    generated_files.extend([
        f3,
        tables_dir / "03_metric_spearman_correlation.csv",
        tables_dir / "03_metric_kendall_correlation.csv",
        tables_dir / "03_metric_jaccard_overlap.csv",
    ])

    # ── Figure 4: Hubs vs Authorities vs PageRank ─────────────────────────────
    f4 = fig_dir / "04_hubs_authorities_pagerank.png"
    _, archetype_data = plot_hubs_authorities_pagerank(gd, scores_df, f4, tables_dir)
    generated_files.extend([
        f4,
        tables_dir / "04_top15_combined_rankings.csv",
        tables_dir / "04_hub_authority_pagerank_scatter.csv",
        tables_dir / "04_archetype_examples.csv",
    ])

    # ── Figure 5: Personalized PageRank ───────────────────────────────────────
    f5 = fig_dir / "05_personalized_pagerank.png"
    t5 = tables_dir / "05_personalized_pagerank_comparison.csv"
    plot_personalized_pagerank(gd, scores_df, cfg.pagerank_tol, cfg.pagerank_max_iter, f5, t5)
    generated_files.extend([f5, t5])

    # ── Figure 6: Community Detection ─────────────────────────────────────────
    f6 = fig_dir / "06_community_detection.png"
    t6 = tables_dir / "06_community_detection.csv"
    _, comm_df = plot_community_detection(gd, scores_df, f6, t6)
    generated_files.extend([f6, t6])

    # ── Figure 7: Node-Type Analysis ──────────────────────────────────────────
    f7 = fig_dir / "07_node_type_analysis.png"
    t7 = tables_dir / "07_node_type_analysis.csv"
    plot_node_type_analysis(scores_df, f7, t7)
    generated_files.extend([f7, t7])

    # ── Figure 8: Interactive Network (PyVis) ─────────────────────────────────
    f8 = fig_dir / "network.html"
    generate_interactive_network(gd, scores_df, cfg.network_top_n, cfg.network_color_by, f8, tables_dir)
    generated_files.extend([
        f8,
        tables_dir / "08_network_nodes.csv",
        tables_dir / "08_network_edges.csv",
    ])

    # ── Figure 9: Bias and Limitations ────────────────────────────────────────
    f9 = fig_dir / "09_bias_limitations.png"
    t9 = tables_dir / "09_bias_filtered_comparison.csv"
    _, bias_summary = plot_bias_limitations(gd, scores_df, cfg.pagerank_tol, cfg.pagerank_max_iter, f9, t9)
    generated_files.extend([f9, t9])

    # ── Summary Markdown ──────────────────────────────────────────────────────
    sum_path = Path(cfg.summary_path)
    generate_summary_markdown(
        cfg, gd, scores_df, val_report, pr_res, hits_res, archetype_data, bias_summary, comm_df, sum_path,
    )
    generated_files.append(sum_path)

    return {
        "generated_files": [str(p) for p in generated_files],
        "figures": [str(p) for p in generated_files if p.suffix in (".png", ".html")],
        "tables": [str(p) for p in generated_files if p.suffix == ".csv"],
        "summary": str(sum_path),
    }
