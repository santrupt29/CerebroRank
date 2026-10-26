"""core/validation.py — Cross-validate custom algorithms against NetworkX.

══════════════════════════════════════════════════════════════════════════════
IMPORTANT: This is the ONLY file in the project allowed to call
           nx.pagerank() and nx.hits().
══════════════════════════════════════════════════════════════════════════════

HITS renormalisation
    NetworkX normalises HITS scores so they sum to 1.
    Our HITS uses L2 normalisation.
    We renormalise BOTH to sum=1 before computing error metrics so we are
    comparing the shape of the distributions, not the scaling convention.
    Design decision: rank correlation (Spearman/Kendall) is scale-invariant,
    so it would pass either way — but we do it for the MSE comparison too.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import networkx as nx
import numpy as np
from scipy.stats import kendalltau, spearmanr

from core.algorithms import HITSResult, PageRankResult
from core.graph_builder import GraphData

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def validate_all(
    gd: GraphData,
    pr_result: PageRankResult,
    hits_result: HITSResult,
    d: float = 0.85,
    tol: float = 1e-6,
) -> Dict[str, Any]:
    """Run PageRank and HITS validation; return combined report."""
    pr_val = validate_pagerank(gd, pr_result, d=d, tol=tol)
    hits_val = validate_hits(gd, hits_result)
    report = {"pagerank": pr_val, "hits": hits_val}
    logger.info(
        "Validation — PR MSE=%.2e Spearman=%.4f | HITS auth Spearman=%.4f",
        pr_val["mse"], pr_val["spearman_rho"], hits_val["authority"]["spearman_rho"],
    )
    return report


# ══════════════════════════════════════════════════════════════════════════════
# PAGERANK VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_pagerank(
    gd: GraphData,
    pr_result: PageRankResult,
    d: float = 0.85,
    tol: float = 1e-6,
) -> Dict[str, Any]:
    """Compare custom PR against nx.pagerank (same d and tol).

    Both vectors already sum to 1; no renormalisation needed.
    """
    # nx.pagerank uses alpha=damping, tol=convergence threshold
    nx_pr: Dict = nx.pagerank(gd.G, alpha=d, tol=tol, max_iter=1000)

    custom = pr_result.scores
    nx_arr = np.array([nx_pr.get(gd.id_to_title[i], 0.0) for i in range(gd.N)])

    mse = float(np.mean((custom - nx_arr) ** 2))
    max_err = float(np.abs(custom - nx_arr).max())
    sp_rho = float(spearmanr(custom, nx_arr).statistic)
    kt_tau = float(kendalltau(custom, nx_arr).statistic)

    top10_c = set(np.argsort(custom)[-10:])
    top10_n = set(np.argsort(nx_arr)[-10:])

    result = {
        "mse": mse,
        "max_absolute_error": max_err,
        "spearman_rho": sp_rho,
        "kendall_tau": kt_tau,
        "top10_overlap": len(top10_c & top10_n),
        "custom_sum": float(custom.sum()),
        "nx_sum": float(nx_arr.sum()),
    }
    logger.info("PR validation: MSE=%.2e spearman=%.4f kendall=%.4f top10=%d/10",
                mse, sp_rho, kt_tau, result["top10_overlap"])
    return result


# ══════════════════════════════════════════════════════════════════════════════
# HITS VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_hits(
    gd: GraphData,
    hits_result: HITSResult,
) -> Dict[str, Any]:
    """Compare custom HITS against nx.hits, renormalising both to sum=1.

    nx.hits returns (hubs_dict, authorities_dict) with values summing to 1.
    Our HITS uses L2 normalisation → divide by sum before comparison.
    """
    nx_hub_dict, nx_auth_dict = nx.hits(gd.G, max_iter=1000, tol=1e-9)

    # Renormalise our L2-vectors to sum=1
    auth_c = _renorm(hits_result.authority)
    hub_c = _renorm(hits_result.hub)

    auth_nx = np.array([nx_auth_dict.get(gd.id_to_title[i], 0.0) for i in range(gd.N)])
    hub_nx = np.array([nx_hub_dict.get(gd.id_to_title[i], 0.0) for i in range(gd.N)])

    auth_metrics = _score_metrics(auth_c, auth_nx, "authority")
    hub_metrics = _score_metrics(hub_c, hub_nx, "hub")

    return {"authority": auth_metrics, "hub": hub_metrics}


def _renorm(v: np.ndarray) -> np.ndarray:
    """Return v / sum(v); guards against zero-sum vectors."""
    s = v.sum()
    return v / s if s > 1e-15 else v


def _score_metrics(custom: np.ndarray, ref: np.ndarray, label: str) -> Dict[str, Any]:
    """Compute MSE, max-err, Spearman rho, Kendall tau, top-10 overlap."""
    mse = float(np.mean((custom - ref) ** 2))
    max_err = float(np.abs(custom - ref).max())
    sp_rho = float(spearmanr(custom, ref).statistic)
    kt_tau = float(kendalltau(custom, ref).statistic)
    top10_c = set(np.argsort(custom)[-10:])
    top10_r = set(np.argsort(ref)[-10:])
    logger.info("HITS %s: MSE=%.2e spearman=%.4f kendall=%.4f",
                label, mse, sp_rho, kt_tau)
    return {
        "mse": mse,
        "max_absolute_error": max_err,
        "spearman_rho": sp_rho,
        "kendall_tau": kt_tau,
        "top10_overlap": len(top10_c & top10_r),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

def save_validation_report(report: Dict[str, Any], path: str) -> None:
    """Write validation report to JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Validation report → %s", p)
