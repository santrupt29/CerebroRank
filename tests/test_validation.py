"""tests/test_validation.py — cross-validate custom algorithms vs NetworkX.

All graphs are generated with fixed random seeds (42, 123) so tests are
fully deterministic.  No network requests are made.

Acceptance criteria (from spec)
--------------------------------
  PageRank MSE vs NetworkX  < 1e-9
  PageRank Spearman rho     > 0.99
  HITS rank correlation     > 0.99 (after renormalisation to sum=1)
"""

from typing import Dict

import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp

from core.algorithms import hits, pagerank
from core.graph_builder import GraphData, _build_sparse_matrices
from core.validation import validate_hits, validate_pagerank


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: build a reproducible random directed graph as GraphData
# ══════════════════════════════════════════════════════════════════════════════

def _make_graph_data(seed: int = 42, n: int = 20, p: float = 0.25) -> GraphData:
    """Erdős-Rényi DiGraph + spanning cycle for guaranteed connectivity.

    The spanning cycle ensures no isolated nodes and that every node has
    at least one in-coming edge, making the HITS comparison meaningful.
    """
    G = nx.erdos_renyi_graph(n, p, seed=seed, directed=True)
    # Add a Hamiltonian cycle so the graph is strongly connected
    for i in range(n):
        G.add_edge(i, (i + 1) % n)

    # Relabel integer nodes as strings so nx_pr_dict keys match id_to_title
    G = nx.relabel_nodes(G, {i: str(i) for i in range(n)})

    title_to_id: Dict[str, int] = {str(i): i for i in range(n)}
    id_to_title: Dict[int, str] = {i: str(i) for i in range(n)}

    A, M, dangling_mask = _build_sparse_matrices(G, title_to_id)
    return GraphData(
        G=G, A=A, M=M,
        id_to_title=id_to_title, title_to_id=title_to_id,
        dangling_mask=dangling_mask, N=n,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGERANK vs NetworkX
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def _gd_seed42() -> GraphData:
    return _make_graph_data(seed=42)


@pytest.fixture(scope="module")
def _pr_seed42(_gd_seed42: GraphData) -> object:
    return pagerank(_gd_seed42.M, _gd_seed42.dangling_mask, d=0.85, tol=1e-10, max_iter=500)


@pytest.fixture(scope="module")
def _hits_seed42(_gd_seed42: GraphData) -> object:
    return hits(_gd_seed42.A, tol=1e-10, max_iter=500)


@pytest.fixture(scope="module")
def _val_pr(_gd_seed42: GraphData, _pr_seed42: object) -> Dict:
    return validate_pagerank(_gd_seed42, _pr_seed42, d=0.85, tol=1e-10)


@pytest.fixture(scope="module")
def _val_hits(_gd_seed42: GraphData, _hits_seed42: object) -> Dict:
    return validate_hits(_gd_seed42, _hits_seed42)


# ══════════════════════════════════════════════════════════════════════════════
# PAGERANK vs NetworkX
# ══════════════════════════════════════════════════════════════════════════════

class TestPageRankVsNetworkX:
    """Custom PageRank must match nx.pagerank to within strict tolerances."""

    def test_mse_below_1e9(self, _val_pr: Dict) -> None:
        assert _val_pr["mse"] < 1e-9, f"MSE={_val_pr['mse']:.2e} exceeds 1e-9"

    def test_spearman_above_099(self, _val_pr: Dict) -> None:
        assert _val_pr["spearman_rho"] > 0.99, f"Spearman={_val_pr['spearman_rho']:.4f}"

    def test_kendall_above_095(self, _val_pr: Dict) -> None:
        assert _val_pr["kendall_tau"] > 0.95, f"Kendall={_val_pr['kendall_tau']:.4f}"

    def test_top10_overlap_high(self, _val_pr: Dict) -> None:
        assert _val_pr["top10_overlap"] >= 8, f"Top-10 overlap={_val_pr['top10_overlap']}"

    def test_custom_sums_to_one(self, _pr_seed42: object) -> None:
        assert abs(_pr_seed42.scores.sum() - 1.0) < 1e-6

    def test_second_seed_still_passes(self) -> None:
        """Robustness check: different random graph, same thresholds."""
        gd = _make_graph_data(seed=123)
        res = pagerank(gd.M, gd.dangling_mask, d=0.85, tol=1e-10, max_iter=500)
        val = validate_pagerank(gd, res, d=0.85, tol=1e-10)
        assert val["mse"] < 1e-9
        assert val["spearman_rho"] > 0.99


# ══════════════════════════════════════════════════════════════════════════════
# HITS vs NetworkX
# ══════════════════════════════════════════════════════════════════════════════

class TestHITSVsNetworkX:
    """Custom HITS (renormalised to sum=1) must match nx.hits rank ordering."""

    def test_authority_spearman_above_099(self, _val_hits: Dict) -> None:
        assert _val_hits["authority"]["spearman_rho"] > 0.99, \
            f"Authority Spearman={_val_hits['authority']['spearman_rho']:.4f}"

    def test_hub_spearman_above_099(self, _val_hits: Dict) -> None:
        assert _val_hits["hub"]["spearman_rho"] > 0.99, \
            f"Hub Spearman={_val_hits['hub']['spearman_rho']:.4f}"

    def test_authority_kendall_above_095(self, _val_hits: Dict) -> None:
        assert _val_hits["authority"]["kendall_tau"] > 0.95

    def test_hub_kendall_above_095(self, _val_hits: Dict) -> None:
        assert _val_hits["hub"]["kendall_tau"] > 0.95

    def test_authority_top10_overlap_high(self, _val_hits: Dict) -> None:
        assert _val_hits["authority"]["top10_overlap"] >= 8

    def test_second_seed_hits(self) -> None:
        gd = _make_graph_data(seed=123)
        res = hits(gd.A, tol=1e-10, max_iter=500)
        val = validate_hits(gd, res)
        assert val["authority"]["spearman_rho"] > 0.99
        assert val["hub"]["spearman_rho"] > 0.99
