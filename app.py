"""app.py — Interactive Streamlit Dashboard for MarvelGraphRank.

Stage 4: Multi-tab offline dashboard with live parameter exploration,
interactive network embedding, node inspection, algorithm comparison,
and a toy-graph playground.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from core.algorithms import hits, pagerank, HITSResult, PageRankResult
from core.config import Config, load_config
from core.evaluator import generate_interactive_network
from core.graph_builder import GraphData, load_graph, spider_trap_demo

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION & STYLING
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="MarvelGraphRank Dashboard",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clean typography and compact metrics
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #E23636; /* Marvel Red */
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 8px;
        padding-bottom: 8px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# CACHED DATA & RESOURCE LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_config() -> Config:
    return load_config()


@st.cache_resource
def get_graph_data() -> Optional[GraphData]:
    cfg = get_config()
    edge_p = Path(cfg.edgelist_path)
    node_p = Path(cfg.nodes_path)
    if not edge_p.exists() or not node_p.exists():
        return None
    try:
        return load_graph(cfg)
    except Exception as e:
        st.error(f"Error loading graph: {e}")
        return None


@st.cache_data
def get_hits_results() -> Optional[HITSResult]:
    gd = get_graph_data()
    if gd is None:
        return None
    cfg = get_config()
    return hits(gd.A, tol=cfg.hits_tol, max_iter=cfg.hits_max_iter)


@st.cache_data
def get_live_pagerank(d: float, tol: float = 1e-6, max_iter: int = 100) -> Optional[PageRankResult]:
    gd = get_graph_data()
    if gd is None:
        return None
    return pagerank(gd.M, gd.dangling_mask, d=d, tol=tol, max_iter=max_iter)


@st.cache_data
def get_personalized_pagerank(node_idx: int, d: float = 0.85) -> Optional[PageRankResult]:
    gd = get_graph_data()
    if gd is None:
        return None
    p = np.zeros(gd.N)
    p[node_idx] = 1.0
    return pagerank(gd.M, gd.dangling_mask, d=d, tol=1e-6, personalization=p)


@st.cache_data
def get_louvain_communities() -> List[Set[str]]:
    gd = get_graph_data()
    if gd is None:
        return []
    G_undir = gd.G.to_undirected()
    raw = nx.community.louvain_communities(G_undir, seed=42)
    return sorted(raw, key=len, reverse=True)


def build_unified_scores_df(d: float) -> Optional[pd.DataFrame]:
    """Combine live PageRank, precomputed HITS, and node metadata into a DataFrame."""
    gd = get_graph_data()
    if gd is None:
        return None
    pr_res = get_live_pagerank(d)
    hits_res = get_hits_results()
    if pr_res is None or hits_res is None:
        return None

    comms = get_louvain_communities()
    comm_map: Dict[str, Tuple[int, str]] = {}
    for c_id, c_nodes in enumerate(comms, 1):
        for n in c_nodes:
            comm_map[n] = (c_id, f"Community {c_id}")

    records = []
    for idx in range(gd.N):
        title = gd.id_to_title[idx]
        node_attr = gd.G.nodes[title] if title in gd.G else {}
        ntype = node_attr.get("node_type", "other")
        cid, clabel = comm_map.get(title, (0, "Unassigned"))

        records.append({
            "id": idx,
            "title": title,
            "pagerank": float(pr_res.scores[idx]),
            "authority": float(hits_res.authority[idx]),
            "hub": float(hits_res.hub[idx]),
            "in_degree": int(gd.G.in_degree(title)),
            "out_degree": int(gd.G.out_degree(title)),
            "degree": int(gd.G.degree(title)),
            "node_type": ntype,
            "community_id": cid,
            "community": clabel,
        })

    df = pd.DataFrame(records)
    df["pagerank_rank"] = df["pagerank"].rank(ascending=False, method="min").astype(int)
    df["authority_rank"] = df["authority"].rank(ascending=False, method="min").astype(int)
    df["hub_rank"] = df["hub"].rank(ascending=False, method="min").astype(int)
    df["in_degree_rank"] = df["in_degree"].rank(ascending=False, method="min").astype(int)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.markdown('<div class="main-header">🕸️ MarvelGraphRank Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Influence Ranking in the Marvel Wikipedia Link Graph using PageRank and HITS | SPCE Mumbai</div>',
        unsafe_allow_html=True,
    )

    gd = get_graph_data()
    if gd is None:
        st.error(
            "⚠️ **Graph data not found!**\n\n"
            "Please ensure `data/edgelist.csv` and `data/nodes.csv` are present.\n"
            "Run the following CLI commands to prepare the dataset:\n"
            "```bash\n"
            "python main.py crawl --pilot   # or full crawl: python main.py crawl\n"
            "python main.py build\n"
            "python main.py rank\n"
            "python main.py report\n"
            "```"
        )
        st.stop()

    # ── Sidebar Controls ──────────────────────────────────────────────────────
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/b/b9/Marvel_Logo.svg", width=160)
    st.sidebar.title("🛠️ Navigation & Parameters")

    d_val = st.sidebar.slider(
        "PageRank Damping Factor (d)",
        min_value=0.50,
        max_value=0.99,
        value=0.85,
        step=0.01,
        help="Probability of following outgoing links (1-d is random teleportation).",
    )

    top_k = st.sidebar.slider(
        "Top-K Elements to Display",
        min_value=5,
        max_value=50,
        value=15,
        step=1,
        help="Number of top ranked nodes shown in tables and charts.",
    )

    scores_df = build_unified_scores_df(d_val)
    if scores_df is None:
        st.error("Failed to compute scores. Please check data files.")
        st.stop()

    available_types = sorted(scores_df["node_type"].unique())
    selected_types = st.sidebar.multiselect(
        "Filter Node Types",
        options=available_types,
        default=available_types,
        help="Select which semantic categories of Wikipedia articles to include.",
    )

    exclude_lists = st.sidebar.checkbox(
        "Exclude 'list' pages",
        value=False,
        help="Filter out compilation and indexing articles (e.g., 'Lists of Marvel Comics characters').",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("🌐 **Mode**: Fully Offline (Zero runtime crawling)")
    st.sidebar.caption("📊 **Graph**: Giant WCC Component Loaded")

    # Apply sidebar filters to view dataframe
    view_df = scores_df[scores_df["node_type"].isin(selected_types)].copy()
    if exclude_lists:
        view_df = view_df[view_df["node_type"] != "list"]

    # ── Multi-Tab Layout ──────────────────────────────────────────────────────
    tabs = st.tabs([
        "📊 1. Overview",
        "🏆 2. Rankings",
        "🔍 3. Node Explorer",
        "🕸️ 4. Interactive Network",
        "📈 5. Algorithm Comparison",
        "🧪 6. Toy Graph Playground",
        "📜 7. About & Ethics",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: OVERVIEW
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.subheader("Dataset & Network Topology Overview")
        st.markdown(
            "High-level structural statistics for the English Wikipedia link graph restricted "
            "to the Marvel universe. Nodes represent Wikipedia articles; directed edges represent body hyperlinks."
        )

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        qr = gd.quality_report

        col1.metric("Total Nodes", f"{gd.N}")
        col2.metric("Total Edges", f"{gd.G.number_of_edges()}")
        col3.metric("Density", f"{qr.get('density', 0.0):.4f}")
        col4.metric("Avg Clustering", f"{qr.get('avg_clustering_coefficient', 0.0):.4f}")
        col5.metric("Giant WCC", f"{qr.get('giant_wcc_size', gd.N)} (100%)")
        col6.metric("Dangling Nodes", f"{qr.get('dangling_node_count', 0)}")

        st.markdown("---")
        c_left, c_right = st.columns([1.2, 1])

        with c_left:
            st.markdown("#### 🏷️ Node Type Distribution")
            type_counts = scores_df["node_type"].value_counts().reset_index()
            type_counts.columns = ["Node Type", "Count"]

            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.barh(type_counts["Node Type"][::-1], type_counts["Count"][::-1], color="#4C72B0", alpha=0.85)
            ax.set_xlabel("Number of Articles")
            ax.set_title("Distribution of Article Semantic Categories")
            for bar in bars:
                w = bar.get_width()
                ax.annotate(f" {int(w)}", (w, bar.get_y() + bar.get_height() / 2), va="center", fontsize=9, weight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with c_right:
            st.markdown("#### 📌 Key Scope Highlights")
            st.markdown(
                """
                - **Domain Roots**: `Category:Marvel Comics`, `Category:Marvel Cinematic Universe`, `Category:Marvel Studios`.
                - **Crawling Protocol**: Hybrid BFS gated by MediaWiki category-membership API with robots.txt compliance.
                - **Link Sanitation**: Structural navboxes, reference sections, and cross-namespace links removed prior to graph ingestion.
                - **Component Guarantee**: Weakly-connected giant component isolated to ensure valid stationary distributions.
                - **Storage Policy**: Fully deterministic offline caching; zero runtime network queries.
                """
            )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: RANKINGS
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.subheader(f"🏆 Multi-Metric Influence Leaderboards (Damping d = {d_val:.2f})")
        st.markdown(
            "Side-by-side comparison of **PageRank** (global recursive citation weight), "
            "**HITS Authority** (endorsed by prominent hubs), and **HITS Hub** (directory pages pointing to authorities). "
            "*PageRank updates dynamically as you adjust the damping slider in the sidebar.*"
        )

        r_col1, r_col2, r_col3 = st.columns(3)

        # 1. PageRank
        with r_col1:
            st.markdown("#### 🥇 PageRank (Global Influence)")
            top_pr = view_df.sort_values(by="pagerank", ascending=False).head(top_k)
            display_pr = top_pr[["title", "pagerank", "node_type"]].copy()
            display_pr["title"] = display_pr["title"].str.replace("_", " ")
            display_pr["pagerank"] = display_pr["pagerank"].apply(lambda x: f"{x:.5f}")
            display_pr.columns = ["Article", "Score", "Type"]
            display_pr.index = range(1, len(display_pr) + 1)
            st.dataframe(display_pr, use_container_width=True)

            fig_pr, ax_pr = plt.subplots(figsize=(5, 4))
            ax_pr.barh(top_pr["title"].str.replace("_", " ")[:8][::-1], top_pr["pagerank"][:8][::-1], color="#4C72B0")
            ax_pr.set_xlabel("Score")
            ax_pr.set_title("Top PageRank Scores")
            plt.tight_layout()
            st.pyplot(fig_pr)
            plt.close(fig_pr)

        # 2. HITS Authority
        with r_col2:
            st.markdown("#### ⭐ HITS Authority Score")
            top_auth = view_df.sort_values(by="authority", ascending=False).head(top_k)
            display_auth = top_auth[["title", "authority", "node_type"]].copy()
            display_auth["title"] = display_auth["title"].str.replace("_", " ")
            display_auth["authority"] = display_auth["authority"].apply(lambda x: f"{x:.5f}")
            display_auth.columns = ["Article", "Score", "Type"]
            display_auth.index = range(1, len(display_auth) + 1)
            st.dataframe(display_auth, use_container_width=True)

            fig_auth, ax_auth = plt.subplots(figsize=(5, 4))
            ax_auth.barh(top_auth["title"].str.replace("_", " ")[:8][::-1], top_auth["authority"][:8][::-1], color="#55A868")
            ax_auth.set_xlabel("Score")
            ax_auth.set_title("Top Authority Scores")
            plt.tight_layout()
            st.pyplot(fig_auth)
            plt.close(fig_auth)

        # 3. HITS Hub
        with r_col3:
            st.markdown("#### 🔗 HITS Hub Score")
            top_hub = view_df.sort_values(by="hub", ascending=False).head(top_k)
            display_hub = top_hub[["title", "hub", "node_type"]].copy()
            display_hub["title"] = display_hub["title"].str.replace("_", " ")
            display_hub["hub"] = display_hub["hub"].apply(lambda x: f"{x:.5f}")
            display_hub.columns = ["Article", "Score", "Type"]
            display_hub.index = range(1, len(display_hub) + 1)
            st.dataframe(display_hub, use_container_width=True)

            fig_hub, ax_hub = plt.subplots(figsize=(5, 4))
            ax_hub.barh(top_hub["title"].str.replace("_", " ")[:8][::-1], top_hub["hub"][:8][::-1], color="#C44E52")
            ax_hub.set_xlabel("Score")
            ax_hub.set_title("Top Hub Scores")
            plt.tight_layout()
            st.pyplot(fig_hub)
            plt.close(fig_hub)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3: NODE EXPLORER
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.subheader("🔍 Node Inspector & Personalized Teleportation")
        st.markdown("Examine the complete topological footprint and link neighborhood of any article in the dataset.")

        all_titles = sorted(scores_df["title"].tolist())
        default_idx = all_titles.index("Spider-Man") if "Spider-Man" in all_titles else 0
        selected_title = st.selectbox(
            "Select or Type a Node Title:",
            options=all_titles,
            index=default_idx,
            format_func=lambda s: s.replace("_", " "),
        )

        node_row = scores_df[scores_df["title"] == selected_title].iloc[0]
        node_idx = int(node_row["id"])

        # Metric Badges
        n_col1, n_col2, n_col3, n_col4 = st.columns(4)
        n_col1.metric("PageRank", f"{node_row['pagerank']:.6f}", f"Rank #{node_row['pagerank_rank']} of {gd.N}")
        n_col2.metric("Authority", f"{node_row['authority']:.6f}", f"Rank #{node_row['authority_rank']} of {gd.N}")
        n_col3.metric("Hub Score", f"{node_row['hub']:.6f}", f"Rank #{node_row['hub_rank']} of {gd.N}")
        n_col4.metric("Degrees (In / Out)", f"{node_row['in_degree']} / {node_row['out_degree']}", f"Rank #{node_row['in_degree_rank']} In-Deg")

        st.markdown(
            f"**Semantic Category:** `{node_row['node_type']}` | "
            f"**Cluster:** `{node_row['community']}` | "
            f"**Wikipedia Link:** [View Page](https://en.wikipedia.org/wiki/{selected_title})"
        )

        st.markdown("---")
        neigh_col1, neigh_col2 = st.columns(2)

        # Inbound Neighbors
        with neigh_col1:
            st.markdown(f"#### 📥 Incoming Links (Pages that cite *{selected_title.replace('_', ' ')}*)")
            in_nodes = list(gd.G.predecessors(selected_title))
            if in_nodes:
                in_df = scores_df[scores_df["title"].isin(in_nodes)].sort_values(by="pagerank", ascending=False).head(10)
                in_display = in_df[["title", "pagerank", "node_type"]].copy()
                in_display["title"] = in_display["title"].str.replace("_", " ")
                in_display["pagerank"] = in_display["pagerank"].apply(lambda x: f"{x:.5f}")
                in_display.columns = ["Referring Page", "PageRank", "Type"]
                st.dataframe(in_display, use_container_width=True)
            else:
                st.info("No incoming links from nodes within this dataset.")

        # Outbound Neighbors
        with neigh_col2:
            st.markdown(f"#### 📤 Outgoing Links (Pages cited by *{selected_title.replace('_', ' ')}*)")
            out_nodes = list(gd.G.successors(selected_title))
            if out_nodes:
                out_df = scores_df[scores_df["title"].isin(out_nodes)].sort_values(by="pagerank", ascending=False).head(10)
                out_display = out_df[["title", "pagerank", "node_type"]].copy()
                out_display["title"] = out_display["title"].str.replace("_", " ")
                out_display["pagerank"] = out_display["pagerank"].apply(lambda x: f"{x:.5f}")
                out_display.columns = ["Cited Page", "PageRank", "Type"]
                st.dataframe(out_display, use_container_width=True)
            else:
                st.info("This is a dangling node (no outgoing links within this dataset).")

        # Personalized PageRank Teleportation Section
        st.markdown("---")
        st.markdown("#### 🎯 Personalized PageRank Teleportation")
        st.markdown(
            f"Run a dedicated random-walk where 100% of teleportation restarts jump directly back to "
            f"**{selected_title.replace('_', ' ')}**. This reveals which articles are closest to this entity in the Marvel ontology."
        )

        if st.button(f"Compute Personalized PageRank from '{selected_title.replace('_', ' ')}'"):
            ppr_res = get_personalized_pagerank(node_idx, d=d_val)
            if ppr_res is not None:
                sorted_ppr_idx = np.argsort(ppr_res.scores)[::-1]
                ppr_records = []
                for rank_pos, idx_val in enumerate(sorted_ppr_idx[:12], 1):
                    t = gd.id_to_title[idx_val]
                    if t == selected_title:
                        continue  # Skip query node itself for recommendation view
                    ppr_records.append({
                        "Rank": len(ppr_records) + 1,
                        "Related Article": t.replace("_", " "),
                        "Personalized PR": float(ppr_res.scores[idx_val]),
                        "Global PR": float(scores_df.loc[scores_df["title"] == t, "pagerank"].values[0]),
                    })
                    if len(ppr_records) == 10:
                        break

                ppr_table = pd.DataFrame(ppr_records)
                p_c1, p_c2 = st.columns([1, 1.2])
                with p_c1:
                    st.dataframe(
                        ppr_table.assign(
                            **{
                                "Personalized PR": ppr_table["Personalized PR"].apply(lambda x: f"{x:.5f}"),
                                "Global PR": ppr_table["Global PR"].apply(lambda x: f"{x:.5f}"),
                            }
                        ),
                        use_container_width=True,
                    )
                with p_c2:
                    fig_p, ax_p = plt.subplots(figsize=(6, 4))
                    ax_p.barh(ppr_table["Related Article"][::-1], ppr_table["Personalized PR"][::-1], color="#CCB974")
                    ax_p.set_xlabel("Personalized PageRank Score")
                    ax_p.set_title(f"Entities Closest to {selected_title.replace('_', ' ')}")
                    plt.tight_layout()
                    st.pyplot(fig_p)
                    plt.close(fig_p)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4: INTERACTIVE NETWORK
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.subheader("🕸️ Interactive Knowledge Graph Visualization (PyVis)")
        st.markdown(
            "Interactive physics-directed network rendering. Hover over nodes to view detailed metadata, "
            "drag nodes to inspect local structures, and zoom to explore communities."
        )

        with st.expander("⚙️ Network Generation Controls", expanded=False):
            cfg = get_config()
            net_col1, net_col2, net_col3 = st.columns([1, 1, 1])
            with net_col1:
                custom_n = st.slider("Subnetwork Node Count (Top-N)", min_value=15, max_value=min(100, gd.N), value=min(75, gd.N), step=5)
            with net_col2:
                color_choice = st.radio("Node Coloring Metric", options=["Authority Gradient", "Louvain Community"], index=0)
            with net_col3:
                st.write("")
                st.write("")
                regenerate_btn = st.button("🔄 Regenerate Network View")

            if regenerate_btn:
                chosen_color_key = "community" if "Community" in color_choice else "authority"
                net_out = Path(cfg.figures_dir) / "network.html"
                generate_interactive_network(
                    gd, scores_df, top_n=custom_n, color_by=chosen_color_key,
                    html_path=net_out, tables_dir=Path(cfg.tables_dir)
                )
                st.success(f"Network re-generated with {custom_n} nodes and {color_choice} coloring!")

        # Embed network.html
        cfg = get_config()
        net_path = Path(cfg.figures_dir) / "network.html"
        if net_path.exists():
            html_content = net_path.read_text(encoding="utf-8")
            components.html(html_content, height=750, scrolling=True)
        else:
            st.warning("`reports/figures/network.html` not found. Please run `python main.py report` to generate it.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5: ALGORITHM COMPARISON
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.subheader("📈 Algorithm Benchmarking & Metric Validation")
        st.markdown(
            "Evaluation of mathematical convergence, correlation matrices, and ground-truth validation "
            "against the NetworkX reference implementation."
        )

        # 1. Validation Table against NetworkX
        st.markdown("#### ✅ Ground-Truth Validation vs NetworkX")
        val_path = Path(get_config().validation_path)
        if val_path.exists():
            val_data = json.loads(val_path.read_text(encoding="utf-8"))
            pr_val = val_data.get("pagerank", {})
            auth_val = val_data.get("hits", {}).get("authority", {})
            hub_val = val_data.get("hits", {}).get("hub", {})

            val_summary_df = pd.DataFrame([
                {
                    "Metric": "Mean Squared Error (MSE)",
                    "PageRank vs nx.pagerank": f"{pr_val.get('mse', 0.0):.2e}",
                    "HITS Authority vs nx.hits": f"{auth_val.get('mse', 0.0):.2e}",
                    "HITS Hub vs nx.hits": f"{hub_val.get('mse', 0.0):.2e}",
                },
                {
                    "Metric": "Spearman Rank Correlation (ρ)",
                    "PageRank vs nx.pagerank": f"{pr_val.get('spearman_rho', 1.0):.4f}",
                    "HITS Authority vs nx.hits": f"{auth_val.get('spearman_rho', 1.0):.4f}",
                    "HITS Hub vs nx.hits": f"{hub_val.get('spearman_rho', 1.0):.4f}",
                },
                {
                    "Metric": "Kendall Tau Correlation (τ)",
                    "PageRank vs nx.pagerank": f"{pr_val.get('kendall_tau', 1.0):.4f}",
                    "HITS Authority vs nx.hits": f"{auth_val.get('kendall_tau', 1.0):.4f}",
                    "HITS Hub vs nx.hits": f"{hub_val.get('kendall_tau', 1.0):.4f}",
                },
                {
                    "Metric": "Top-10 Ranking Overlap",
                    "PageRank vs nx.pagerank": f"{pr_val.get('top10_overlap', 10)} / 10",
                    "HITS Authority vs nx.hits": f"{auth_val.get('top10_overlap', 10)} / 10",
                    "HITS Hub vs nx.hits": f"{hub_val.get('top10_overlap', 10)} / 10",
                },
            ])
            st.table(val_summary_df.set_index("Metric"))
        else:
            st.info("Validation report not found. Run `python main.py rank` to generate `reports/validation.json`.")

        st.markdown("---")
        c_fig1, c_fig2 = st.columns(2)

        with c_fig1:
            st.markdown("#### 📉 Algorithm Convergence Curves")
            f1_path = Path("reports/figures/01_convergence_curves.png")
            if f1_path.exists():
                st.image(str(f1_path), use_container_width=True)
            else:
                st.info("Run `python main.py report` to generate Figure 1.")

        with c_fig2:
            st.markdown("#### 🎯 Damping Factor Sensitivity")
            f2_path = Path("reports/figures/02_damping_sensitivity.png")
            if f2_path.exists():
                st.image(str(f2_path), use_container_width=True)
            else:
                st.info("Run `python main.py report` to generate Figure 2.")

        st.markdown("---")
        st.markdown("#### 📊 Metric Divergence & Rank Correlation")
        f3_path = Path("reports/figures/03_metric_comparison.png")
        if f3_path.exists():
            st.image(str(f3_path), use_container_width=True)
        else:
            st.info("Run `python main.py report` to generate Figure 3.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 6: TOY GRAPH PLAYGROUND
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[5]:
        st.subheader("🧪 Interactive Toy Graph & Power Iteration Mechanics")
        st.markdown(
            "Explore how PageRank iterates step-by-step on a small 5-node directed graph. "
            "Modify edge connections, advance the power-iteration step slider, and observe how "
            "teleportation resolves dangling-node leakage and spider traps."
        )

        toy_col1, toy_col2 = st.columns([1, 1.2])

        preset = st.selectbox(
            "Select Graph Scenario Preset:",
            options=["Standard Cycle + Branch", "Spider Trap (C ↔ D)", "Dangling Node (B Dead-End)", "Custom Topology"],
            index=0,
        )

        default_edges = "A -> B\nB -> C\nC -> A\nA -> D\nD -> E\nE -> A"
        if preset == "Spider Trap (C ↔ D)":
            default_edges = "A -> B\nA -> C\nC -> D\nD -> C"
        elif preset == "Dangling Node (B Dead-End)":
            default_edges = "A -> B\nA -> C\nC -> A"

        with toy_col1:
            edge_input = st.text_area(
                "Directed Edges (format: Source -> Target, one per line):",
                value=default_edges,
                height=150,
            )
            d_toy = st.slider("Playground Damping Factor", min_value=0.50, max_value=0.99, value=0.85, step=0.05)
            step = st.slider("Power Iteration Step (t)", min_value=0, max_value=25, value=1, step=1)

        # Parse edges
        parsed_edges = []
        toy_nodes = set(["A", "B", "C", "D", "E"])
        for line in edge_input.strip().split("\n"):
            line = line.strip()
            if "->" in line:
                u, v = line.split("->")
                u, v = u.strip().upper(), v.strip().upper()
                if u and v:
                    parsed_edges.append((u, v))
                    toy_nodes.add(u)
                    toy_nodes.add(v)

        sorted_nodes = sorted(list(toy_nodes))
        N_toy = len(sorted_nodes)
        node_to_idx = {n: i for i, n in enumerate(sorted_nodes)}

        # Build transition matrix M
        M_toy = np.zeros((N_toy, N_toy))
        out_deg = {n: 0 for n in sorted_nodes}
        for u, v in parsed_edges:
            out_deg[u] += 1
        for u, v in parsed_edges:
            M_toy[node_to_idx[v], node_to_idx[u]] = 1.0 / out_deg[u]

        dangling_toy = np.array([out_deg[n] == 0 for n in sorted_nodes])

        # Step-by-step iteration
        uni = np.ones(N_toy) / N_toy
        v_curr = np.ones(N_toy) / N_toy
        residuals_history = []

        for s in range(1, step + 1):
            d_sum = v_curr[dangling_toy].sum()
            v_next = d_toy * (M_toy @ v_curr + d_sum * uni) + (1.0 - d_toy) * uni
            residuals_history.append(float(np.abs(v_next - v_curr).sum()))
            v_curr = v_next

        with toy_col2:
            st.markdown(f"#### 📊 Probability Vector at Step t = {step}")
            res_val = residuals_history[-1] if residuals_history else 0.0
            st.markdown(rf"**Current L1 Residual $\|v_t - v_{{t-1}}\|_1$:** `{res_val:.6f}` | **Vector Sum:** `{v_curr.sum():.6f}`")

            toy_df = pd.DataFrame({
                "Node": sorted_nodes,
                "Probability": v_curr,
                "Out-Degree": [out_deg[n] for n in sorted_nodes],
                "Dangling": ["Yes (Dead-End)" if dangling_toy[i] else "No" for i, n in enumerate(sorted_nodes)],
            })
            st.dataframe(toy_df.style.format({"Probability": "{:.5f}"}), use_container_width=True)

            fig_toy, ax_toy = plt.subplots(figsize=(6, 3))
            ax_toy.bar(sorted_nodes, v_curr, color="#4C72B0", alpha=0.85)
            ax_toy.axhline(1.0 / N_toy, color="gray", linestyle="--", label="Uniform (1/N)")
            ax_toy.set_ylabel("Probability")
            ax_toy.set_title(f"PageRank Distribution (Step {step})")
            ax_toy.set_ylim(0, max(0.5, float(v_curr.max()) + 0.1))
            ax_toy.legend()
            plt.tight_layout()
            st.pyplot(fig_toy)
            plt.close(fig_toy)

        st.markdown("---")
        st.markdown("#### 🔬 Theoretical Analysis: Spider Traps & Dangling Nodes")
        st.markdown(
            r"""
            In naive power iteration ($v_{t+1} = M v_t$), two major structural failure modes arise:
            1. **Dangling Nodes (Dead Ends)**: Nodes with out-degree 0 have all-zero columns in $M$. Mass reaching them leaks out of the system ($\sum v_t \to 0$).
               *Correction*: Dangling mass is gathered at each step and redistributed uniformly across all nodes.
            2. **Spider Traps**: A set of nodes with no outbound edges to the outside world. Repeated random transitions trap all random walks, draining probability from all other pages.
               *Correction*: The damping teleportation parameter $(1-d)$ guarantees that with probability $0.15$, the surfer teleports to an arbitrary node, escaping the trap.
            """
        )

        with st.expander("📖 View Verified 4-Node Proof from core.graph_builder", expanded=False):
            demo = spider_trap_demo()
            st.markdown(f"**Description**: {demo['description']}")
            d_c1, d_c2 = st.columns(2)
            with d_c1:
                st.markdown("**Naive Iteration (No Teleportation):**")
                st.json(demo["plain"])
            with d_c2:
                st.markdown("**Standard PageRank (Teleportation d=0.85):**")
                st.json(demo["fixed"])
    # ══════════════════════════════════════════════════════════════════════════
    # TAB 7: ABOUT & ETHICS
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[6]:
        st.subheader("📜 Academic Context, Ethics, & Module Attribution")
        st.markdown(
            "**Course**: B.Tech Computer Engineering — Web Data Mining — Semester VII  \n"
            "**Institution**: Sardar Patel College of Engineering (SPCE), Mumbai  \n"
            "**Project**: MarvelGraphRank (Influence Ranking in the Marvel Wikipedia Link Graph)"
        )

        st.markdown("---")
        e_col1, e_col2 = st.columns(2)

        with e_col1:
            st.markdown("#### 🛡️ Ethical Crawler & Compliance Record")
            crawl_p = Path(get_config().crawl_report_path)
            if crawl_p.exists():
                c_data = json.loads(crawl_p.read_text(encoding="utf-8"))
                st.markdown(
                    f"""
                    - **Robots.txt Enforced**: Yes (`urllib.robotparser` verification)
                    - **Pages Fetched**: `{c_data.get('pages_fetched', 0)}`
                    - **API Queries Executed**: `{c_data.get('api_calls', 0)}`
                    - **Local Cache Hits**: `{c_data.get('cache_hits', 0)}`
                    - **Robots Disallowed Skips**: `{c_data.get('robots_skipped', 0)}`
                    - **HTTP 429/503 Retries**: `{c_data.get('retries', 0)}`
                    - **Crawl Wall-Clock Time**: `{c_data.get('total_runtime', 0.0):.1f}s`
                    """
                )
            else:
                st.info("Crawl report not found.")

        with e_col2:
            st.markdown("#### 📚 Licensing & Data Attribution")
            st.markdown(
                """
                - **Data Source**: English Wikipedia (`https://en.wikipedia.org`).
                - **License**: Text content is freely available under the **Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)**.
                - **Rate Limiting**: Complies with Wikimedia Foundation API Guidelines using randomized delays ($1.0 - 1.5$s) and declared academic `User-Agent`.
                - **Privacy & Scope**: Public encyclopaedic article links only; no user talk pages, personal data, or private credentials gathered.
                """
            )

        st.markdown("---")
        st.markdown("#### 👥 Team Module Ownership")
        team_df = pd.DataFrame([
            {"Role": "Member 1", "Module": "WikiCrawler + Config", "Key Components": "BFS crawler, category gating, robots.txt, YAML config"},
            {"Role": "Member 2", "Module": "Graph Builder + Sparse Algorithms", "Key Components": "CSR matrix builder, custom PageRank, HITS, spider-trap demo"},
            {"Role": "Member 3", "Module": "Validation + Analysis Figures", "Key Components": "NetworkX validation, convergence curves, damping sensitivity"},
            {"Role": "Member 4", "Module": "Interactive Dashboard + Reports", "Key Components": "Streamlit app.py, PyVis network embedding, executive summary"},
        ])
        st.table(team_df.set_index("Role"))


if __name__ == "__main__":
    main()
