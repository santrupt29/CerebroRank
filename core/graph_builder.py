"""core/graph_builder.py — Build and analyse the Marvel Wikipedia link graph.

Design decisions
----------------
- Self-loops are removed and logged; nx.DiGraph deduplicates parallel edges.
- Giant WCC filter: controlled by keep_giant_wcc in config.yaml.  Dropping
  isolated nodes ensures the transition matrix M is well-behaved.
- Sparse matrix conventions (used everywhere downstream):
    A[i, j] = 1          if directed edge i → j exists     (row=source)
    M[j, i] = 1/outdeg(i) if directed edge i → j exists    (row=dest, col=src)
  M is column-stochastic; dangling columns (outdeg=0) are all-zero and handled
  inside pagerank() via the dangling_mask.
- HITS uses A directly: authority = A^T h,  hub = A a.
- Node index mapping is sorted-alphabetically for reproducibility.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

from core.config import Config

logger = logging.getLogger(__name__)


@dataclass
class GraphData:
    """Container passed between build, rank, and validation stages."""

    G: nx.DiGraph                   # title-keyed NetworkX graph (post-filter)
    A: sp.csr_matrix                # A[i,j]=1 if i→j
    M: sp.csr_matrix                # M[j,i]=1/outdeg(i) — column-stochastic
    id_to_title: Dict[int, str]
    title_to_id: Dict[str, int]
    dangling_mask: np.ndarray       # bool (N,); True where out-degree == 0
    N: int
    quality_report: Dict[str, Any] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def load_graph(cfg: Config) -> GraphData:
    """Read edgelist.csv + nodes.csv, filter, and build sparse matrices.

    Raises FileNotFoundError if edgelist.csv is missing (run crawl first).
    """
    edge_path = Path(cfg.edgelist_path)
    node_path = Path(cfg.nodes_path)

    if not edge_path.exists():
        raise FileNotFoundError(
            f"edgelist.csv not found at {edge_path}. "
            "Run `python main.py crawl` first."
        )

    edges_df = pd.read_csv(edge_path, dtype=str).dropna()
    logger.info("Loaded %d raw edges from %s", len(edges_df), edge_path)

    # Build raw DiGraph, optionally seeding nodes from nodes.csv metadata
    G_raw: nx.DiGraph = nx.DiGraph()
    if node_path.exists():
        for _, row in pd.read_csv(node_path, dtype=str).fillna("").iterrows():
            G_raw.add_node(row["title"], **row.to_dict())

    for _, row in edges_df.iterrows():
        G_raw.add_edge(row["source"], row["target"])

    nodes_raw, edges_raw = G_raw.number_of_nodes(), G_raw.number_of_edges()
    logger.info("Raw graph: %d nodes, %d edges", nodes_raw, edges_raw)

    # Remove self-loops
    loops = list(nx.selfloop_edges(G_raw))
    G_raw.remove_edges_from(loops)
    logger.info("Removed %d self-loop(s)", len(loops))

    # Optionally keep only the giant weakly connected component
    G, nodes_dropped = _filter_giant_wcc(G_raw, cfg)

    report = _compute_quality_report(G, nodes_raw, edges_raw, len(loops), nodes_dropped)

    # Zero-indexed, alphabetically sorted node mapping (reproducible)
    sorted_titles = sorted(G.nodes())
    title_to_id = {t: i for i, t in enumerate(sorted_titles)}
    id_to_title = {i: t for t, i in title_to_id.items()}

    A, M, dangling_mask = _build_sparse_matrices(G, title_to_id)

    logger.info(
        "GraphData ready: N=%d, edges=%d, dangling=%d",
        len(title_to_id), G.number_of_edges(), int(dangling_mask.sum()),
    )
    return GraphData(
        G=G, A=A, M=M,
        id_to_title=id_to_title, title_to_id=title_to_id,
        dangling_mask=dangling_mask, N=len(title_to_id),
        quality_report=report,
    )


# ══════════════════════════════════════════════════════════════════════════════
# FILTERING
# ══════════════════════════════════════════════════════════════════════════════

def _filter_giant_wcc(G: nx.DiGraph, cfg: Config) -> Tuple[nx.DiGraph, int]:
    """Return (filtered_graph, n_nodes_dropped).

    When keep_giant_wcc is True, retains only the largest weakly connected
    component.  This gives a connected topology for meaningful ranking.
    """
    if not cfg.keep_giant_wcc:
        return G, 0
    wccs = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    if len(wccs) <= 1:
        return G, 0
    giant = wccs[0]
    dropped = G.number_of_nodes() - len(giant)
    logger.info(
        "Kept giant WCC: %d nodes (dropped %d from %d smaller components).",
        len(giant), dropped, len(wccs) - 1,
    )
    return G.subgraph(giant).copy(), dropped


# ══════════════════════════════════════════════════════════════════════════════
# DATA QUALITY REPORT
# ══════════════════════════════════════════════════════════════════════════════

def _compute_quality_report(
    G: nx.DiGraph,
    nodes_raw: int,
    edges_raw: int,
    self_loops_removed: int,
    nodes_dropped: int,
) -> Dict[str, Any]:
    """Compute and return the data-quality statistics dict."""
    N, E = G.number_of_nodes(), G.number_of_edges()
    out_degrees = dict(G.out_degree())
    dangling_count = sum(1 for d in out_degrees.values() if d == 0)

    wccs = list(nx.weakly_connected_components(G))
    sccs = list(nx.strongly_connected_components(G))
    giant_wcc_size = max((len(c) for c in wccs), default=0)
    density = E / (N * (N - 1)) if N > 1 else 0.0
    avg_cc = nx.average_clustering(G.to_undirected())

    report: Dict[str, Any] = {
        "nodes_raw": nodes_raw,
        "edges_raw": edges_raw,
        "self_loops_removed": self_loops_removed,
        "nodes_after_filter": N,
        "edges_after_filter": E,
        "nodes_dropped_wcc": nodes_dropped,
        "dangling_node_count": dangling_count,
        "weakly_connected_components": len(wccs),
        "strongly_connected_components": len(sccs),
        "giant_wcc_size": giant_wcc_size,
        "density": round(density, 8),
        "avg_clustering_coefficient": round(avg_cc, 6),
    }
    logger.info("Quality report: %s", json.dumps(report))
    return report


def save_quality_report(report: Dict[str, Any], cfg: Config) -> None:
    """Write data_quality.json to reports/."""
    path = Path(cfg.data_quality_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Quality report → %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# SPARSE MATRIX CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def _build_sparse_matrices(
    G: nx.DiGraph,
    title_to_id: Dict[str, int],
) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """Build CSR adjacency A and column-stochastic transition M.

    A[i, j] = 1            edge i→j
    M[j, i] = 1/outdeg(i)  edge i→j, outdeg(i) > 0
    Dangling columns in M stay all-zero; caller handles them via dangling_mask.

    Column-stochasticity is asserted after construction (non-dangling cols).
    """
    N = len(title_to_id)

    if G.number_of_edges() == 0:
        empty = sp.csr_matrix((N, N), dtype=np.float64)
        return empty, empty, np.ones(N, dtype=bool)

    src_arr = np.array([title_to_id[u] for u, _ in G.edges()], dtype=np.int32)
    dst_arr = np.array([title_to_id[v] for _, v in G.edges()], dtype=np.int32)

    # Adjacency: A[src, dst] = 1
    A = sp.csr_matrix(
        (np.ones(len(src_arr), dtype=np.float64), (src_arr, dst_arr)),
        shape=(N, N),
    )

    # Out-degrees and dangling mask from row sums of A
    out_degrees = np.asarray(A.sum(axis=1), dtype=np.float64).ravel()
    dangling_mask = out_degrees == 0

    # Transition: M[dst, src] = 1/outdeg(src)
    # Suppress divide-by-zero for dangling nodes: np.where already masks them,
    # but numpy evaluates both branches before selecting, triggering a warning.
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_out = np.where(out_degrees > 0, 1.0 / out_degrees, 0.0)
    M = sp.csr_matrix(
        (inv_out[src_arr], (dst_arr, src_arr)),   # (data, (row=dest, col=src))
        shape=(N, N),
    )

    # Sanity: non-dangling columns must sum to 1
    col_sums = np.asarray(M.sum(axis=0), dtype=np.float64).ravel()
    max_col_err = float(np.abs(col_sums[~dangling_mask] - 1.0).max()) if (~dangling_mask).any() else 0.0
    assert max_col_err < 1e-9, f"M not column-stochastic: max_col_err={max_col_err:.2e}"

    return A, M, dangling_mask


# ══════════════════════════════════════════════════════════════════════════════
# SPIDER-TRAP / DANGLING-NODE DEMO
# ══════════════════════════════════════════════════════════════════════════════

def spider_trap_demo() -> Dict[str, Any]:
    """Illustrate dangling nodes and spider traps on a hard-coded 4-node graph.

    Topology
    --------
        A(0) → B(1)           B is a dangling node (no out-edges)
        A(0) → C(2)
        C(2) → D(3) ─┐        C and D form a spider trap
        D(3) → C(2) ─┘

    Plain power iteration (no fixes)
        Mass that reaches B leaks away (B's column in M is zero).
        The remaining mass is absorbed by the C-D trap.
        The PR vector no longer sums to 1.

    Teleportation (d = 0.85)
        dangling mass is redistributed uniformly each step.
        Teleportation prevents trap absorption.
        PR converges to a valid probability distribution.

    Returns a dict suitable for display in the Streamlit app and viva.
    """
    N, d = 4, 0.85

    # M[dest, src] = transition prob (column-stochastic for non-dangling)
    M = np.zeros((N, N), dtype=np.float64)
    M[1, 0] = 0.5   # A → B  (A splits equally)
    M[2, 0] = 0.5   # A → C
    M[3, 2] = 1.0   # C → D
    M[2, 3] = 1.0   # D → C

    dangling = np.array([False, True, False, False])  # only B is dangling

    # ── Plain: just M @ PR repeated (no dangling fix, no teleportation) ────
    pr_plain = np.ones(N) / N
    for _ in range(60):
        pr_plain = M @ pr_plain          # mass leaks at dangling B

    # ── Fixed: standard PageRank formula with dangling correction ──────────
    pr_fixed = np.ones(N) / N
    uni = np.ones(N) / N
    for _ in range(200):
        dsum = pr_fixed[dangling].sum()
        pr_fixed = d * (M @ pr_fixed + dsum * uni) + (1.0 - d) * uni

    return {
        "description": (
            "4-node demo: A→B (B dangling, mass leaks), A→C, C↔D (spider trap). "
            "Plain power iteration: mass leaks at B; rest trapped in C-D. "
            "With teleportation (d=0.85): converges to a valid distribution."
        ),
        "nodes": {"A": 0, "B_dangling": 1, "C_trap": 2, "D_trap": 3},
        "topology": ["A→B", "A→C", "C→D", "D→C"],
        "plain": {
            "scores": [round(x, 6) for x in pr_plain.tolist()],
            "sum": round(float(pr_plain.sum()), 6),
            "issue": "Sum < 1 (mass lost at B); C+D absorb all remaining mass",
        },
        "fixed": {
            "scores": [round(x, 6) for x in pr_fixed.tolist()],
            "sum": round(float(pr_fixed.sum()), 6),
            "note": "Sum ≈ 1; all nodes nonzero; no trap concentration",
        },
    }
