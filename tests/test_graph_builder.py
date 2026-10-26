"""tests/test_graph_builder.py — offline unit tests for core/graph_builder.py.

All tests use in-memory CSV fixtures written to tmp_path; no network calls.
"""

from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import scipy.sparse as sp

from core.config import Config
from core.graph_builder import (
    GraphData,
    _build_sparse_matrices,
    load_graph,
    spider_trap_demo,
)


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES & HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_NODES_HEADER = "id,title,url,degree,in_degree,out_degree,node_type,categories\n"

SIMPLE_CSV     = "source,target\nA,B\nB,C\nC,A\n"          # 3-cycle, no dangling
SELF_LOOP_CSV  = "source,target\nA,B\nB,B\nB,C\nC,A\n"     # B has a self-loop
DANGLING_CSV   = "source,target\nA,B\nA,C\n"               # B and C are dangling


def _cfg(tmp_path: Path, edgelist: str) -> Config:
    """Write edgelist CSV to tmp_path and return matching Config."""
    ep = tmp_path / "edgelist.csv"
    np_ = tmp_path / "nodes.csv"
    ep.write_text(edgelist, encoding="utf-8")
    np_.write_text(_NODES_HEADER, encoding="utf-8")
    return Config(
        seeds=[], category_roots=[], category_depth=1,
        max_depth=2, max_nodes=10, max_new_nodes_per_page=5,
        delay_min=0.0, delay_max=0.0, max_retries=1, backoff_base=2,
        user_agent="test", wiki_base_url="https://en.wikipedia.org",
        api_endpoint="", robots_url="",
        cache_dir=str(tmp_path / "cache"),
        edgelist_path=str(ep),
        nodes_path=str(np_),
        crawl_report_path=str(tmp_path / "report.json"),
        state_file=str(tmp_path / "state.json"),
        dropped_namespaces=[], removed_selectors=[], dropped_sections=[],
        keep_giant_wcc=False,           # keep all nodes for deterministic tests
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. SELF-LOOP REMOVAL
# ══════════════════════════════════════════════════════════════════════════════

class TestSelfLoopRemoval:
    def test_no_self_loops_in_graph(self, tmp_path: Path) -> None:
        gd = load_graph(_cfg(tmp_path, SELF_LOOP_CSV))
        for u, v in gd.G.edges():
            assert u != v, f"Self-loop survived: {u}→{v}"

    def test_self_loop_count_reported(self, tmp_path: Path) -> None:
        gd = load_graph(_cfg(tmp_path, SELF_LOOP_CSV))
        assert gd.quality_report["self_loops_removed"] == 1

    def test_zero_self_loops_in_clean_graph(self, tmp_path: Path) -> None:
        gd = load_graph(_cfg(tmp_path, SIMPLE_CSV))
        assert gd.quality_report["self_loops_removed"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# 2. TRANSITION MATRIX COLUMN SUMS
# ══════════════════════════════════════════════════════════════════════════════

class TestTransitionMatrix:
    def test_non_dangling_columns_sum_to_one(self, tmp_path: Path) -> None:
        """Every non-dangling column of M must sum to exactly 1."""
        gd = load_graph(_cfg(tmp_path, SIMPLE_CSV))
        col_sums = np.asarray(gd.M.sum(axis=0)).ravel()
        non_dangling = col_sums[~gd.dangling_mask]
        assert np.allclose(non_dangling, 1.0, atol=1e-9)

    def test_dangling_columns_sum_to_zero(self, tmp_path: Path) -> None:
        """Dangling-node columns in M must be all-zero."""
        gd = load_graph(_cfg(tmp_path, DANGLING_CSV))
        col_sums = np.asarray(gd.M.sum(axis=0)).ravel()
        assert np.allclose(col_sums[gd.dangling_mask], 0.0, atol=1e-12)

    def test_transition_values_correct(self, tmp_path: Path) -> None:
        """A has out-degree 2 in DANGLING_CSV → M[:, A] entries are 0.5."""
        gd = load_graph(_cfg(tmp_path, DANGLING_CSV))
        i_A = gd.title_to_id["A"]
        col_A = np.asarray(gd.M.getcol(i_A).todense()).ravel()
        nz = col_A[col_A > 0]
        assert np.allclose(nz, 0.5, atol=1e-9)

    def test_matrix_is_csr(self, tmp_path: Path) -> None:
        gd = load_graph(_cfg(tmp_path, SIMPLE_CSV))
        assert sp.issparse(gd.M) and gd.M.format == "csr"
        assert sp.issparse(gd.A) and gd.A.format == "csr"


# ══════════════════════════════════════════════════════════════════════════════
# 3. INDEX MAPPING CONSISTENCY
# ══════════════════════════════════════════════════════════════════════════════

class TestIDMapping:
    def test_id_to_title_is_inverse(self, tmp_path: Path) -> None:
        gd = load_graph(_cfg(tmp_path, SIMPLE_CSV))
        for title, idx in gd.title_to_id.items():
            assert gd.id_to_title[idx] == title

    def test_all_graph_nodes_mapped(self, tmp_path: Path) -> None:
        gd = load_graph(_cfg(tmp_path, SIMPLE_CSV))
        assert set(gd.title_to_id.keys()) == set(gd.G.nodes())

    def test_indices_contiguous_zero_based(self, tmp_path: Path) -> None:
        gd = load_graph(_cfg(tmp_path, SIMPLE_CSV))
        assert set(gd.id_to_title.keys()) == set(range(gd.N))

    def test_N_equals_node_count(self, tmp_path: Path) -> None:
        gd = load_graph(_cfg(tmp_path, SIMPLE_CSV))
        assert gd.N == gd.G.number_of_nodes()


# ══════════════════════════════════════════════════════════════════════════════
# 4. ADJACENCY MATRIX VALUES
# ══════════════════════════════════════════════════════════════════════════════

class TestAdjacencyMatrix:
    def test_edge_entry_is_one(self, tmp_path: Path) -> None:
        gd = load_graph(_cfg(tmp_path, SIMPLE_CSV))
        i_A, i_B = gd.title_to_id["A"], gd.title_to_id["B"]
        assert gd.A[i_A, i_B] == 1.0          # A→B exists

    def test_non_edge_entry_is_zero(self, tmp_path: Path) -> None:
        gd = load_graph(_cfg(tmp_path, SIMPLE_CSV))
        i_A, i_C = gd.title_to_id["A"], gd.title_to_id["C"]
        assert gd.A[i_A, i_C] == 0.0          # no A→C in SIMPLE_CSV

    def test_shape_equals_N_by_N(self, tmp_path: Path) -> None:
        gd = load_graph(_cfg(tmp_path, SIMPLE_CSV))
        assert gd.A.shape == (gd.N, gd.N)

    def test_row_sum_equals_out_degree(self, tmp_path: Path) -> None:
        """Row sum of A[i,:] must equal the out-degree of node i."""
        gd = load_graph(_cfg(tmp_path, SIMPLE_CSV))
        row_sums = np.asarray(gd.A.sum(axis=1)).ravel()
        for title, idx in gd.title_to_id.items():
            assert row_sums[idx] == gd.G.out_degree(title)


# ══════════════════════════════════════════════════════════════════════════════
# 5. SPIDER-TRAP DEMO
# ══════════════════════════════════════════════════════════════════════════════

class TestSpiderTrapDemo:
    def test_plain_sum_below_one(self) -> None:
        result = spider_trap_demo()
        assert result["plain"]["sum"] < 1.0 - 1e-4, \
            "Plain PR should lose mass at the dangling node"

    def test_fixed_sum_approx_one(self) -> None:
        result = spider_trap_demo()
        assert abs(result["fixed"]["sum"] - 1.0) < 1e-3

    def test_fixed_all_nodes_nonzero(self) -> None:
        result = spider_trap_demo()
        assert all(s > 1e-6 for s in result["fixed"]["scores"])

    def test_trap_dominates_plain(self) -> None:
        """C(2) and D(3) absorb remaining mass in the plain iteration."""
        r = spider_trap_demo()
        plain = r["plain"]["scores"]
        assert plain[2] + plain[3] > plain[0] + plain[1]

    def test_fixed_has_required_keys(self) -> None:
        r = spider_trap_demo()
        for key in ("description", "nodes", "topology", "plain", "fixed"):
            assert key in r
