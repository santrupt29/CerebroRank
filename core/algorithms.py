"""core/algorithms.py — Custom PageRank and HITS implementations.

══════════════════════════════════════════════════════════════════════════════
IMPORTANT: This module must NEVER import nx.pagerank or nx.hits.
           NetworkX built-ins are used only in core/validation.py.
══════════════════════════════════════════════════════════════════════════════

PageRank formula (column-stochastic M, dangling-node correction)
────────────────────────────────────────────────────────────────
    PR_{t+1} = d · (M @ PR_t  +  dangling_sum_t · p)  +  (1 − d) · p

    where  dangling_sum_t = Σᵢ∈D PR_t[i],  D = dangling nodes (outdeg=0)
           p = personalisation vector (sums to 1; uniform by default)

    Convergence: ‖PR_{t+1} − PR_t‖₁ < tol.  Final vector sums to 1.

Personalised PageRank
─────────────────────
    Replace p (uniform) with any probability vector p (sums to 1).
    Both the teleportation term and the dangling redistribution follow p, so
    the stationary distribution is biased towards high-p nodes.
    Design decision: applying p to dangling redistribution (not uniform) is
    consistent with the random-surfer interpretation — even a dangling surfer
    teleports according to the personalisation preference.

HITS (Kleinberg, 1999)
──────────────────────
    A[i,j]=1 if edge i→j.
    h₀ = 1_N / √N  (L2-unit init)
    Loop:
        a_{t+1} = Aᵀ hₜ  →  L2-normalise
        h_{t+1} = A a_{t+1}  →  L2-normalise
    Convergence: max(‖Δa‖₁, ‖Δh‖₁) < tol.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# RESULT DATACLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PageRankResult:
    """Typed result for PageRank."""
    scores: np.ndarray          # shape (N,), sums to 1 within 1e-4
    residuals: List[float]      # L1 ‖PR_next − PR‖ per iteration
    iterations: int             # iterations until convergence (or max_iter)


@dataclass
class HITSResult:
    """Typed result for HITS."""
    authority: np.ndarray       # shape (N,), L2-normalised
    hub: np.ndarray             # shape (N,), L2-normalised
    # Each tuple: (auth_L1_Δ, hub_L1_Δ, auth_cos_dist, hub_cos_dist)
    convergence: List[Tuple[float, float, float, float]] = field(default_factory=list)
    iterations: int = 0


# ══════════════════════════════════════════════════════════════════════════════
# PAGERANK
# ══════════════════════════════════════════════════════════════════════════════

def pagerank(
    M: sp.csr_matrix,
    dangling_mask: np.ndarray,
    d: float = 0.85,
    tol: float = 1e-6,
    max_iter: int = 100,
    personalization: Optional[np.ndarray] = None,
) -> PageRankResult:
    """Power-iteration PageRank on a column-stochastic CSR matrix.

    Args:
        M:               Column-stochastic transition matrix (N×N CSR).
                         M[j,i] = 1/outdeg(i) when edge i→j exists.
        dangling_mask:   Boolean (N,); True for nodes with out-degree 0.
        d:               Damping factor (default 0.85).
        tol:             L1 convergence threshold.
        max_iter:        Hard iteration cap.
        personalization: Optional probability vector p (sums to 1).
                         Defaults to uniform 1/N.

    Returns:
        PageRankResult with scores, residual history, iteration count.
    """
    N = M.shape[0]
    PR = np.ones(N, dtype=np.float64) / N

    # Build (and validate) teleportation distribution
    p = _make_personalization(personalization, N)

    residuals: List[float] = []
    iters = 0

    for iters in range(1, max_iter + 1):
        dangling_sum = float(PR[dangling_mask].sum())
        # Full formula: teleport + dangling redistribution both follow p
        PR_next = d * (M @ PR + dangling_sum * p) + (1.0 - d) * p
        residual = float(np.abs(PR_next - PR).sum())
        residuals.append(residual)
        PR = PR_next

        if residual < tol:
            logger.info("PageRank converged at iter %d (residual=%.2e).", iters, residual)
            break
    else:
        logger.warning("PageRank: max_iter=%d reached; last residual=%.2e.", max_iter, residuals[-1])

    # PR must remain a valid probability vector
    total = float(PR.sum())
    assert abs(total - 1.0) < 1e-4, f"PageRank sum={total:.8f} (expected 1)"

    return PageRankResult(scores=PR, residuals=residuals, iterations=iters)


def _make_personalization(p: Optional[np.ndarray], N: int) -> np.ndarray:
    """Return a validated, L1-normalised personalisation vector."""
    if p is None:
        return np.ones(N, dtype=np.float64) / N
    arr = np.asarray(p, dtype=np.float64).copy()
    total = arr.sum()
    if abs(total) < 1e-12:
        raise ValueError("Personalization vector sums to zero.")
    arr /= total
    return arr


# ══════════════════════════════════════════════════════════════════════════════
# HITS
# ══════════════════════════════════════════════════════════════════════════════

def hits(
    A: sp.csr_matrix,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> HITSResult:
    """Kleinberg HITS via alternating authority/hub updates.

    Args:
        A:        Adjacency matrix A[i,j]=1 if i→j (N×N CSR).
        tol:      Convergence: max(‖Δa‖₁, ‖Δh‖₁) < tol.
        max_iter: Hard iteration cap.

    Returns:
        HITSResult with L2-normalised authority and hub vectors.
    """
    N = A.shape[0]
    At = A.T.tocsr()            # pre-transpose for repeated a = Aᵀ h

    # Initialise hub uniformly (L2-unit)
    h = np.ones(N, dtype=np.float64)
    _l2_norm_inplace(h)

    prev_a = np.zeros(N, dtype=np.float64)
    prev_h = h.copy()
    a = np.zeros(N, dtype=np.float64)

    convergence: List[Tuple[float, float, float, float]] = []
    iters = 0

    for iters in range(1, max_iter + 1):
        # Authority update
        a = At @ h
        _l2_norm_inplace(a)

        # Hub update
        h = A @ a
        _l2_norm_inplace(h)

        # Convergence metrics
        auth_l1 = float(np.abs(a - prev_a).sum())
        hub_l1 = float(np.abs(h - prev_h).sum())
        auth_cos = _cosine_dist(a, prev_a)
        hub_cos = _cosine_dist(h, prev_h)
        convergence.append((auth_l1, hub_l1, auth_cos, hub_cos))

        logger.debug("HITS %d: auth_L1=%.2e hub_L1=%.2e", iters, auth_l1, hub_l1)

        prev_a = a.copy()
        prev_h = h.copy()

        if max(auth_l1, hub_l1) < tol:
            logger.info("HITS converged at iter %d.", iters)
            break
    else:
        logger.warning("HITS: max_iter=%d reached.", max_iter)

    return HITSResult(authority=a, hub=h, convergence=convergence, iterations=iters)


# ══════════════════════════════════════════════════════════════════════════════
# PRIVATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _l2_norm_inplace(v: np.ndarray) -> None:
    """Normalise *v* in-place by its L2 norm; no-op if norm ≈ 0."""
    norm = float(np.linalg.norm(v, 2))
    if norm > 1e-15:
        v /= norm


def _cosine_dist(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine distance ∈ [0, 2]; returns 1.0 if either vector is near-zero."""
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < 1e-15 or nv < 1e-15:
        return 1.0
    return 1.0 - float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
