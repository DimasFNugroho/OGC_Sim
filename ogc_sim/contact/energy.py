"""
Contact Energy — 2-stage activation function g(d, r)  (Eq. 18)
and per-contact energy, gradient, and Hessian blocks for VBD.

Paper reference: Sec. 3.6, Eq. 18-20.

The activation function has two stages:

    g(d, r) = (k_c / 2)(r − d)²           if τ ≤ d ≤ r   [quadratic]
              −k′_c log(d) + b             if 0 < d < τ   [logarithmic]

where τ = r/2 is the stitch point.  With τ = r/2 the function is C²
continuous everywhere: C¹ is enforced by matching derivatives at τ (giving
k′_c = k_c τ²) and C² follows automatically because τ = r/2 makes the
second derivatives equal too.

Derivation of k′_c (Eq. 19):
    C¹ at d = τ:  −k_c(r − τ) = −k′_c / τ
    ⟹ k′_c = τ · k_c · (r − τ) = (r/2) · k_c · (r/2) = k_c · τ²

b (Eq. 20) is chosen so the two pieces have equal values at d = τ:
    b = (k_c / 2)(r − τ)² + k′_c log(τ)
"""

from __future__ import annotations

import numpy as np

from ogc_sim.geometry.distance import point_triangle_distance, edge_edge_distance


# ======================================================================
# Scalar activation function and its derivatives
# ======================================================================

def _stitch_params(r: float, k_c: float) -> tuple[float, float, float]:
    """Return (τ, k′_c, b) for the 2-stage activation function."""
    tau       = r / 2.0
    k_c_prime = k_c * tau**2                                    # Eq. 19 (with τ = r/2)
    b         = 0.5 * k_c * (r - tau)**2 + k_c_prime * np.log(tau)  # Eq. 20
    return tau, k_c_prime, b


def activation_g(d: float, r: float, k_c: float) -> float:
    """
    Contact activation energy  g(d, r).  Eq. 18.

    Parameters
    ----------
    d   : float  distance between the two primitives (>= 0)
    r   : float  contact radius
    k_c : float  stiffness of the quadratic stage

    Returns
    -------
    float : energy value
    """
    if d >= r:
        return 0.0
    tau, k_c_prime, b = _stitch_params(r, k_c)
    if d >= tau:                                    # quadratic stage
        return 0.5 * k_c * (r - d)**2
    else:                                           # logarithmic stage
        return -k_c_prime * np.log(d) + b


def activation_dg_dd(d: float, r: float, k_c: float) -> float:
    """
    First derivative of g with respect to d.  dg/dd.

    Returns
    -------
    float
    """
    if d >= r:
        return 0.0
    tau, k_c_prime, _ = _stitch_params(r, k_c)
    if d >= tau:
        return -k_c * (r - d)
    else:
        return -k_c_prime / d


def activation_d2g_dd2(d: float, r: float, k_c: float) -> float:
    """
    Second derivative of g with respect to d.  d²g/dd².

    Returns
    -------
    float
    """
    if d >= r:
        return 0.0
    tau, k_c_prime, _ = _stitch_params(r, k_c)
    if d >= tau:
        return k_c
    else:
        return k_c_prime / d**2


# ======================================================================
# Vertex-facet contact — energy, gradient, Hessian
# ======================================================================
#
# E_vf(x_v, a) = g( dis(x_v, a), r )
#
# where a is a face, edge, or vertex feature of the mesh.
# The closest point c = closest_point(x_v, a).
#
# Chain rule gradient w.r.t. x_v:
#   ∂E/∂x_v = (dg/dd) · ∂d/∂x_v = (dg/dd) · (x_v − c) / d
#
# Chain rule Hessian block ∂²E/∂x_v²:
#   H_vv = (d²g/dd²) · n⊗n + (dg/dd) / d · (I − n⊗n)
#
# where n = (x_v − c) / d  is the contact normal.
#
# Assumption: c is treated as fixed when differentiating (valid away from
# feature boundaries where it varies continuously with x_v).  This is the
# standard approximation used in VBD.


def contact_energy_vf(
    v_pos: np.ndarray,
    a_pos: np.ndarray,
    b_pos: np.ndarray,
    c_pos: np.ndarray,
    r: float,
    k_c: float,
) -> float:
    """
    Contact energy for one vertex-facet pair.

    E = g( dis(v_pos, triangle(a, b, c)), r )

    Parameters
    ----------
    v_pos : (3,)  query vertex position
    a_pos, b_pos, c_pos : (3,)  triangle vertex positions
    r     : float  contact radius
    k_c   : float  contact stiffness

    Returns
    -------
    float
    """
    d, _, _, _ = point_triangle_distance(v_pos, a_pos, b_pos, c_pos)
    return activation_g(d, r, k_c)


def contact_gradient_v_vf(
    v_pos: np.ndarray,
    a_pos: np.ndarray,
    b_pos: np.ndarray,
    c_pos: np.ndarray,
    r: float,
    k_c: float,
) -> np.ndarray:
    """
    Gradient of vertex-facet contact energy w.r.t. the query vertex x_v.

    ∂E/∂x_v = (dg/dd) · (x_v − cp) / d

    Returns (3,) array.
    """
    d, cp, _, _ = point_triangle_distance(v_pos, a_pos, b_pos, c_pos)
    if d >= r or d < 1e-12:
        return np.zeros(3)
    dg_dd = activation_dg_dd(d, r, k_c)
    n     = (v_pos - cp) / d
    return dg_dd * n


def contact_hessian_v_vf(
    v_pos: np.ndarray,
    a_pos: np.ndarray,
    b_pos: np.ndarray,
    c_pos: np.ndarray,
    r: float,
    k_c: float,
) -> np.ndarray:
    """
    Hessian block ∂²E/∂x_v² for the query vertex in a vertex-facet contact.

    The formula depends on which sub-feature the closest point lands on,
    because the curvature of d(v) = ||v − c(v)|| differs by feature:

    FACE_INTERIOR  c(v) = proj(v onto face plane) → d is linear in v
                   Hess(d) = 0
                   H = d²g/dd² · n⊗n

    VERTEX a_face  c(v) = a_face (fixed point)
                   Hess(d) = (I − n⊗n) / d  (standard point-distance curvature)
                   H = d²g/dd² · n⊗n + (dg/dd)/d · (I − n⊗n)

    EDGE  (a_face, b_face)  c(v) = closest point on fixed edge
                   Hess(d) = (I − d̂⊗d̂ − n⊗n) / d  where d̂ = edge direction
                   H = d²g/dd² · n⊗n + (dg/dd)/d · (I − d̂⊗d̂ − n⊗n)

    Returns (3, 3) array.
    """
    from ogc_sim.geometry.distance import ClosestFeature

    d, cp, feature, feat_idx = point_triangle_distance(v_pos, a_pos, b_pos, c_pos)
    if d >= r or d < 1e-12:
        return np.zeros((3, 3))

    dg_dd  = activation_dg_dd(d, r, k_c)
    d2g_dd = activation_d2g_dd2(d, r, k_c)
    n      = (v_pos - cp) / d
    nnt    = np.outer(n, n)

    if feature == ClosestFeature.FACE_INTERIOR:
        # d(v) = |v · n̂_face − const|  (linear) → Hess(d) = 0
        return d2g_dd * nnt

    elif feature == ClosestFeature.VERTEX:
        # d(v) = ||v − a_fixed||, standard point-to-point
        return d2g_dd * nnt + (dg_dd / d) * (np.eye(3) - nnt)

    else:  # ClosestFeature.EDGE
        # Edge direction: feat_idx encodes which edge (a,b)=0, (b,c)=1, (c,a)=2
        verts = [a_pos, b_pos, c_pos]
        e_start = verts[feat_idx]
        e_end   = verts[(feat_idx + 1) % 3]
        edge_vec = e_end - e_start
        d_hat  = edge_vec / np.linalg.norm(edge_vec)
        ddt    = np.outer(d_hat, d_hat)
        return d2g_dd * nnt + (dg_dd / d) * (np.eye(3) - ddt - nnt)


# ======================================================================
# Edge-edge contact — energy, gradient, Hessian
# ======================================================================
#
# E_ee(e1, e2) = g( dis(e1, e2), r )
#
# For VBD, we need ∂E/∂x_v for each vertex x_v that is an endpoint of
# either e1 or e2.
#
# Let cp1 = p + t1·(q−p)  and  cp2 = r + t2·(s−r)  be the closest points.
# d = ||cp1 − cp2||,   n = (cp1 − cp2) / d.
#
# ∂d/∂p = (1 − t1) · n     ∂d/∂q = t1 · n
# ∂d/∂r = −(1 − t2) · n    ∂d/∂s = −t2 · n
#
# (Valid at generic interior-to-interior contacts.  At degenerate configs
# where t1 or t2 is clamped to 0/1, the formula reduces to endpoint
# gradients — the same formula still holds with t1/t2 at their clamped
# values.)


def contact_energy_ee(
    p: np.ndarray,
    q: np.ndarray,
    r_e: np.ndarray,
    s: np.ndarray,
    r: float,
    k_c: float,
) -> float:
    """
    Contact energy for one edge-edge pair.

    E = g( dis(e1=(p,q), e2=(r,s)), r_contact )

    Note: r_e is the first endpoint of edge 2 (named r in the paper);
    r is the contact radius.  Distinct names to avoid shadowing.
    """
    d, *_ = edge_edge_distance(p, q, r_e, s)
    return activation_g(d, r, k_c)


def contact_gradient_v_ee(
    v_pos: np.ndarray,
    v_idx_in_pair: int,
    p: np.ndarray,
    q: np.ndarray,
    r_e: np.ndarray,
    s: np.ndarray,
    r: float,
    k_c: float,
) -> np.ndarray:
    """
    Gradient of edge-edge contact energy w.r.t. one endpoint vertex.

    Parameters
    ----------
    v_pos          : (3,) position of the vertex being differentiated
    v_idx_in_pair  : int  — which endpoint this is:
                      0 = p (e1 start), 1 = q (e1 end),
                      2 = r_e (e2 start), 3 = s (e2 end)
    p, q           : (3,) endpoints of edge e1
    r_e, s         : (3,) endpoints of edge e2
    r              : float  contact radius
    k_c            : float  contact stiffness

    Returns
    -------
    (3,) gradient
    """
    d, cp1, t1, cp2, t2 = edge_edge_distance(p, q, r_e, s)
    if d >= r or d < 1e-12:
        return np.zeros(3)
    dg_dd = activation_dg_dd(d, r, k_c)
    n     = (cp1 - cp2) / d   # direction from e2 toward e1

    if v_idx_in_pair == 0:     # p  (e1 start)
        return dg_dd * (1.0 - t1) * n
    elif v_idx_in_pair == 1:   # q  (e1 end)
        return dg_dd * t1 * n
    elif v_idx_in_pair == 2:   # r_e  (e2 start)
        return dg_dd * -(1.0 - t2) * n
    else:                      # s  (e2 end)
        return dg_dd * -t2 * n


def contact_hessian_v_ee(
    v_idx_in_pair: int,
    p: np.ndarray,
    q: np.ndarray,
    r_e: np.ndarray,
    s: np.ndarray,
    r: float,
    k_c: float,
) -> np.ndarray:
    """
    Hessian block ∂²E/∂x_v² for one endpoint in an edge-edge contact.

    H = d²g/dd² · (α n) ⊗ (α n)

    where α is the weight for this vertex (see contact_gradient_v_ee).

    Note: the (dg/dd)/d · (I − n⊗n) term that appeared in VF contact
    also exists here but requires ∂t1/∂x_v (the derivative of the
    closest-point parameter), which involves the edge direction and
    is zero when the edge direction is fixed.  We omit that term here
    as it contributes a rank-deficient correction that is small when
    the edge is short relative to r.  A full formula is used in M4.

    Returns (3, 3) array.
    """
    d, cp1, t1, cp2, t2 = edge_edge_distance(p, q, r_e, s)
    if d >= r or d < 1e-12:
        return np.zeros((3, 3))
    d2g_dd = activation_d2g_dd2(d, r, k_c)
    n      = (cp1 - cp2) / d

    weights = [1.0 - t1, t1, -(1.0 - t2), -t2]
    alpha   = weights[v_idx_in_pair]
    an      = alpha * n
    return d2g_dd * np.outer(an, an)
