"""tests/test_algorithms.py — unit tests for custom PageRank and HITS.

All tests build M/A from scratch (no graph_builder dependency) so they
run fully offline with zero network requests.

Analytical PageRank values for the 3-node chain (d=0.85)
─────────────────────────────────────────────────────────
  Edges: 0→1, 1→2, 2→1  (no dangling nodes)
  p₀ = (1−0.85)/3 = 0.05
  p₁ = 0.85·(p₀ + p₂) + 0.05   [incoming from 0 and 2]
  p₂ = 0.85·p₁ + 0.05           [incoming from 1 only]
  Solving: p₁ ≈ 0.4865, p₂ ≈ 0.4635
"""

from typing import List, Tuple

import numpy as np
import pytest
import scipy.sparse as sp

from core.algorithms import HITSResult, PageRankResult, hits, pagerank


# ══════════════════════════════════════════════════════════════════════════════
# HELPER BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _make_M(edges: List[Tuple[int, int]], N: int) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Build column-stochastic M and dangling_mask from edge list."""
    out_deg = np.zeros(N, dtype=np.float64)
    for s, _ in edges:
        out_deg[s] += 1
    dangling = out_deg == 0

    if not edges:
        return sp.csr_matrix((N, N), dtype=np.float64), dangling

    rows, cols, data = [], [], []
    for s, d in edges:
        rows.append(d)          # M[dest, src]
        cols.append(s)
        data.append(1.0 / out_deg[s])
    M = sp.csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float64)
    return M, dangling


def _make_A(edges: List[Tuple[int, int]], N: int) -> sp.csr_matrix:
    """Build adjacency A[src, dst] = 1."""
    if not edges:
        return sp.csr_matrix((N, N), dtype=np.float64)
    r, c = zip(*edges)
    return sp.csr_matrix(
        (np.ones(len(edges)), (list(r), list(c))), shape=(N, N), dtype=np.float64
    )


# ── common graph definitions ─────────────────────────────────────────────────
CYCLE_EDGES    = [(0, 1), (1, 2), (2, 0)]          # 3-cycle, no dangling
CHAIN_EDGES    = [(0, 1), (1, 2), (2, 1)]          # 0→1→2←→1, no dangling
DANGLING_EDGES = [(0, 1), (0, 2)]                  # 1 and 2 are dangling
BIPARTITE      = [(0, 2), (0, 3), (1, 2), (1, 3)] # H0,H1→A0,A1

# Analytical PageRank for CHAIN_EDGES (d=0.85), see module docstring
CHAIN_EXPECTED = np.array([0.05, 0.4865, 0.4635])


# ══════════════════════════════════════════════════════════════════════════════
# 1. 3-NODE SYMMETRIC CYCLE
# ══════════════════════════════════════════════════════════════════════════════

class TestPageRankCycle:
    """Uniform distribution by symmetry for any d."""

    def test_uniform_scores(self) -> None:
        M, dm = _make_M(CYCLE_EDGES, 3)
        r = pagerank(M, dm)
        assert np.allclose(r.scores, 1 / 3, atol=1e-5)

    def test_sums_to_one(self) -> None:
        M, dm = _make_M(CYCLE_EDGES, 3)
        r = pagerank(M, dm)
        assert abs(r.scores.sum() - 1.0) < 1e-6

    def test_returns_typed_result(self) -> None:
        M, dm = _make_M(CYCLE_EDGES, 3)
        r = pagerank(M, dm)
        assert isinstance(r, PageRankResult)
        assert isinstance(r.scores, np.ndarray)
        assert isinstance(r.residuals, list)

    def test_residuals_decrease(self) -> None:
        M, dm = _make_M(CYCLE_EDGES, 3)
        r = pagerank(M, dm, max_iter=30)
        assert r.residuals[0] >= r.residuals[-1]


# ══════════════════════════════════════════════════════════════════════════════
# 2. CHAIN WITH KNOWN ANALYTICAL VALUES
# ══════════════════════════════════════════════════════════════════════════════

class TestPageRankChain:
    def test_analytical_values(self) -> None:
        M, dm = _make_M(CHAIN_EDGES, 3)
        r = pagerank(M, dm, d=0.85, tol=1e-10)
        assert np.allclose(r.scores, CHAIN_EXPECTED, atol=1e-3)

    def test_node_1_highest(self) -> None:
        M, dm = _make_M(CHAIN_EDGES, 3)
        r = pagerank(M, dm, d=0.85)
        assert np.argmax(r.scores) == 1

    def test_node_0_lowest(self) -> None:
        M, dm = _make_M(CHAIN_EDGES, 3)
        r = pagerank(M, dm, d=0.85)
        assert np.argmin(r.scores) == 0


# ══════════════════════════════════════════════════════════════════════════════
# 3. DANGLING NODES
# ══════════════════════════════════════════════════════════════════════════════

class TestPageRankDangling:
    def test_sums_to_one_with_dangling(self) -> None:
        """Dangling correction must preserve the probability-vector property."""
        M, dm = _make_M(DANGLING_EDGES, 3)
        r = pagerank(M, dm, d=0.85)
        assert abs(r.scores.sum() - 1.0) < 1e-6

    def test_all_scores_positive(self) -> None:
        M, dm = _make_M(DANGLING_EDGES, 3)
        r = pagerank(M, dm, d=0.85)
        assert all(s > 0 for s in r.scores)

    def test_dangling_mask_identifies_sinks(self) -> None:
        _, dm = _make_M(DANGLING_EDGES, 3)
        assert not dm[0]   # node 0 has out-edges
        assert dm[1]       # node 1 is dangling
        assert dm[2]       # node 2 is dangling


# ══════════════════════════════════════════════════════════════════════════════
# 4. MAX ITER CAP
# ══════════════════════════════════════════════════════════════════════════════

class TestPageRankMaxIter:
    def test_stops_at_max_iter(self) -> None:
        M, dm = _make_M(CYCLE_EDGES, 3)
        r = pagerank(M, dm, max_iter=4)
        assert r.iterations <= 4
        assert len(r.residuals) == r.iterations

    def test_result_still_sums_to_one(self) -> None:
        M, dm = _make_M(CYCLE_EDGES, 3)
        r = pagerank(M, dm, max_iter=2)
        assert abs(r.scores.sum() - 1.0) < 1e-6


# ══════════════════════════════════════════════════════════════════════════════
# 5. PERSONALISED PAGERANK
# ══════════════════════════════════════════════════════════════════════════════

class TestPersonalizedPageRank:
    def test_personalisation_boosts_seed_node(self) -> None:
        """Concentrating teleport on node 0 should lift PR[0] above 1/3."""
        M, dm = _make_M(CYCLE_EDGES, 3)
        pers = np.array([1.0, 0.0, 0.0])
        r = pagerank(M, dm, d=0.85, personalization=pers)
        assert r.scores[0] > 1 / 3 + 0.05

    def test_personalised_still_sums_to_one(self) -> None:
        M, dm = _make_M(CYCLE_EDGES, 3)
        r = pagerank(M, dm, personalization=np.array([0.5, 0.3, 0.2]))
        assert abs(r.scores.sum() - 1.0) < 1e-6

    def test_zero_vector_raises(self) -> None:
        M, dm = _make_M(CYCLE_EDGES, 3)
        with pytest.raises(ValueError, match="sums to zero"):
            pagerank(M, dm, personalization=np.zeros(3))

    def test_unnormalised_vector_accepted(self) -> None:
        """A non-unit personalisation vector should be accepted and normalised."""
        M, dm = _make_M(CYCLE_EDGES, 3)
        r = pagerank(M, dm, personalization=np.array([2.0, 4.0, 4.0]))
        assert abs(r.scores.sum() - 1.0) < 1e-6


# ══════════════════════════════════════════════════════════════════════════════
# 6. HITS — BIPARTITE HUB-AUTHORITY GRAPH
# ══════════════════════════════════════════════════════════════════════════════

class TestHITSBipartite:
    """H0(0), H1(1) → A0(2), A1(3): authorities are 2,3; hubs are 0,1."""

    def test_authorities_higher_at_sinks(self) -> None:
        A = _make_A(BIPARTITE, 4)
        r = hits(A)
        assert r.authority[2] > r.authority[0]
        assert r.authority[3] > r.authority[1]

    def test_hubs_higher_at_sources(self) -> None:
        A = _make_A(BIPARTITE, 4)
        r = hits(A)
        assert r.hub[0] > r.hub[2]
        assert r.hub[1] > r.hub[3]

    def test_l2_normalised_authority(self) -> None:
        A = _make_A(BIPARTITE, 4)
        r = hits(A)
        assert abs(np.linalg.norm(r.authority) - 1.0) < 1e-6

    def test_l2_normalised_hub(self) -> None:
        A = _make_A(BIPARTITE, 4)
        r = hits(A)
        assert abs(np.linalg.norm(r.hub) - 1.0) < 1e-6

    def test_symmetric_authorities_equal(self) -> None:
        """By symmetry of the bipartite structure, auth[2] == auth[3]."""
        A = _make_A(BIPARTITE, 4)
        r = hits(A)
        assert np.isclose(r.authority[2], r.authority[3], atol=1e-6)

    def test_symmetric_hubs_equal(self) -> None:
        A = _make_A(BIPARTITE, 4)
        r = hits(A)
        assert np.isclose(r.hub[0], r.hub[1], atol=1e-6)

    def test_returns_typed_result(self) -> None:
        A = _make_A(BIPARTITE, 4)
        r = hits(A)
        assert isinstance(r, HITSResult)


# ══════════════════════════════════════════════════════════════════════════════
# 7. HITS — CONVERGENCE PROPERTIES
# ══════════════════════════════════════════════════════════════════════════════

class TestHITSConvergence:
    def test_convergence_list_populated(self) -> None:
        A = _make_A(BIPARTITE, 4)
        r = hits(A)
        assert len(r.convergence) > 0

    def test_max_iter_respected(self) -> None:
        A = _make_A(BIPARTITE, 4)
        r = hits(A, max_iter=5)
        assert r.iterations <= 5

    def test_empty_graph_no_crash(self) -> None:
        """HITS on a graph with no edges should return zero vectors silently."""
        A = sp.csr_matrix((4, 4), dtype=np.float64)
        r = hits(A)
        assert r.authority.shape == (4,)
        assert r.hub.shape == (4,)

    def test_convergence_tuple_length(self) -> None:
        A = _make_A(BIPARTITE, 4)
        r = hits(A)
        for item in r.convergence:
            assert len(item) == 4    # (auth_l1, hub_l1, cos_a, cos_h)
