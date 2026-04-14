"""
Conservative displacement bounds  (Eq. 21-26).

Paper reference: Sec. 3.7.

Each vertex v gets a bound b_v such that, starting from a
penetration-free state X_prev, any X satisfying

    ||x_v − x_prev_v|| ≤ b_v  ∀ v ∈ V          (Eq. 27)

is also penetration-free.

    b_v = γ_p · min(d_min_v,  d_min_e_v,  d_min_t_v)    (Eq. 21)

where:
    d_min_v   = min distance from v to any non-adjacent face      (Eq. 22)
    d_min_e_v = min over v's neighbour edges of d_min_e            (Eq. 23-24)
    d_min_t_v = min over v's neighbour faces of d_min_t            (Eq. 25-26)

All three are already computed as a by-product of contact detection
(Algorithm 1 & 2) and stored in the ContactSets object.
"""

from __future__ import annotations

import numpy as np

from ogc_sim.geometry.mesh     import Mesh
from ogc_sim.contact.detection import ContactSets


def compute_conservative_bounds(
    mesh: Mesh,
    cs: ContactSets,
    gamma_p: float,
) -> dict[int, float]:
    """
    Compute per-vertex conservative displacement bound b_v.  Eq. 21.

    b_v = γ_p · min(d_min_v, d_min_e_v, d_min_t_v)

    Parameters
    ----------
    mesh    : Mesh         provides T_v, E_v adjacency
    cs      : ContactSets  provides d_min_v, d_min_e, d_min_t
    gamma_p : float        relaxation parameter, 0 < γ_p < 0.5
                           (paper uses 0.45)

    Returns
    -------
    dict[int, float]  — b_v for every vertex index
    """
    bounds: dict[int, float] = {}

    for v_idx in range(mesh.num_vertices):
        # Eq. 22 — d_min_v: directly from Algorithm 1
        d_min_v = cs.d_min_v.get(v_idx, float("inf"))

        # Eq. 23 — d_min_e_v: min over v's neighbour edges of d_min_e
        d_min_e_v = float("inf")
        for e_idx in mesh.E_v[v_idx]:
            d_min_e_v = min(d_min_e_v, cs.d_min_e.get(e_idx, float("inf")))

        # Eq. 25 — d_min_t_v: min over v's neighbour faces of d_min_t
        d_min_t_v = float("inf")
        for t_idx in mesh.T_v[v_idx]:
            d_min_t_v = min(d_min_t_v, cs.d_min_t.get(t_idx, float("inf")))

        d_min = min(d_min_v, d_min_e_v, d_min_t_v)
        bounds[v_idx] = gamma_p * d_min

    return bounds


def truncate_displacements(
    X: np.ndarray,
    X_prev: np.ndarray,
    bounds: dict[int, float],
) -> tuple[np.ndarray, int]:
    """
    Clamp each vertex displacement to its conservative bound.  Eq. 27.

    For each v: if ||x_v − x_prev_v|| > b_v, project back to the
    sphere of radius b_v centred at x_prev_v.

    Parameters
    ----------
    X       : (N, 3)  current vertex positions (may be modified)
    X_prev  : (N, 3)  snapshot positions from the last contact detection
    bounds  : dict[int, float]  per-vertex bounds b_v

    Returns
    -------
    X_out       : (N, 3)  positions after truncation
    num_exceed  : int     number of vertices that were truncated
    """
    X_out = X.copy()
    num_exceed = 0

    for v_idx, b_v in bounds.items():
        delta = X[v_idx] - X_prev[v_idx]
        d_n   = float(np.linalg.norm(delta))
        if d_n > b_v:
            X_out[v_idx] = X_prev[v_idx] + (delta / d_n) * b_v
            num_exceed  += 1

    return X_out, num_exceed


def apply_initial_guess_truncation(
    X_init: np.ndarray,
    X_prev: np.ndarray,
    bounds: dict[int, float],
) -> np.ndarray:
    """
    Truncate an initial guess to within the conservative bounds.  Eq. 28.

    Used once at the start of the first solver iteration (Algorithm 3 line 21).
    Identical to truncate_displacements but discards the counter.

    Parameters
    ----------
    X_init  : (N, 3)  proposed initial positions (e.g. inertia extrapolation)
    X_prev  : (N, 3)  penetration-free snapshot (current step's X_t)
    bounds  : dict[int, float]

    Returns
    -------
    (N, 3) truncated positions
    """
    X_out, _ = truncate_displacements(X_init, X_prev, bounds)
    return X_out
