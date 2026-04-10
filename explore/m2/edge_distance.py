"""
edge_distance.py
================
A human-readable implementation of edge-to-edge distance.

This file is self-contained — no imports from ogc_sim.
Read it alongside learn_algorithm2.py so you understand
exactly how the distance query works before trusting it
as a black box.

The one function you need:

    dist, cp_e1, t1, cp_e2, t2, feature, feat_idx = \
        edge_edge_distance(p, q, r, s)

Everything below builds up to that function step by step.
"""

import numpy as np


# ============================================================
# ClosestFeatureOnEdge
# ============================================================
# When the closest point on the TARGET edge is found, we want
# to know WHERE on that edge it landed.
#
# There are exactly two possibilities (simpler than a triangle):
#
#   INTERIOR  — the closest point is somewhere in the middle
#               of the edge (not at either endpoint)
#   VERTEX    — the closest point is right at one of the
#               two endpoints
#
# This matters in Algorithm 2 the same way ClosestFeature
# mattered in Algorithm 1:
#   INTERIOR → always a valid contact (no extra check needed)
#   VERTEX   → need a feasibility check (Eq. 15)
# ============================================================

class ClosestFeatureOnEdge:
    INTERIOR = "INTERIOR"
    VERTEX   = "VERTEX"


# ============================================================
# The main idea: two line segments in 3D
# ============================================================
#
# We have two segments:
#   e1 = the segment from p to q  (the "query" edge)
#   e2 = the segment from r to s  (the "target" edge)
#
# We want to find the pair of points — one on e1, one on e2 —
# that are closest to each other.
#
# We parameterize each segment:
#   point on e1 = p + t1 * (q - p),  t1 ∈ [0, 1]
#   point on e2 = r + t2 * (s - r),  t2 ∈ [0, 1]
#
# So t1 = 0 means we are at p, t1 = 1 means we are at q.
# Likewise for t2 on e2.
#
# We minimize the squared distance between these two points
# over t1, t2 ∈ [0, 1].  The unconstrained minimum is a
# simple 2×2 linear system.  We then clamp t1 and t2 to
# [0, 1] to handle the case where the minimum lies outside
# the segment.
# ============================================================


def _clamp(value, lo, hi):
    """Clamp a scalar to the interval [lo, hi]."""
    return max(lo, min(hi, value))


def edge_edge_distance(p, q, r, s):
    """
    Find the closest points between segment pq and segment rs.

    Parameters
    ----------
    p, q : np.ndarray, shape (3,)   — endpoints of the first  edge (e1)
    r, s : np.ndarray, shape (3,)   — endpoints of the second edge (e2)

    Returns
    -------
    dist      : float
        Minimum distance between the two segments.

    cp_e1     : np.ndarray, shape (3,)
        Closest point ON e1 (the query edge).

    t1        : float   ∈ [0, 1]
        Parameter along e1 where cp_e1 lies.
        t1 = 0 → at p,  t1 = 1 → at q.

    cp_e2     : np.ndarray, shape (3,)
        Closest point ON e2 (the target edge).

    t2        : float   ∈ [0, 1]
        Parameter along e2 where cp_e2 lies.
        t2 = 0 → at r,  t2 = 1 → at s.

    feature   : ClosestFeatureOnEdge
        Whether cp_e2 is at a vertex (VERTEX) or in the
        interior (INTERIOR) of e2.

    feat_idx  : int
        VERTEX   → 0 if cp_e2 is at r,  1 if cp_e2 is at s
        INTERIOR → -1
    """

    d1 = q - p       # direction of e1
    d2 = s - r       # direction of e2

    len1_sq = float(np.dot(d1, d1))   # |e1|²
    len2_sq = float(np.dot(d2, d2))   # |e2|²

    EPS = 1e-10   # treat anything smaller as zero-length

    # ----------------------------------------------------------
    # Special case: both edges are points (zero length)
    # ----------------------------------------------------------
    if len1_sq < EPS and len2_sq < EPS:
        cp_e1 = p.copy()
        cp_e2 = r.copy()
        t1, t2 = 0.0, 0.0
        dist = float(np.linalg.norm(cp_e1 - cp_e2))
        return dist, cp_e1, t1, cp_e2, t2, ClosestFeatureOnEdge.VERTEX, 0

    # ----------------------------------------------------------
    # Special case: e1 is a point (zero length)
    # ----------------------------------------------------------
    if len1_sq < EPS:
        t1 = 0.0
        # Project p onto e2: t2 = dot(p - r, d2) / |d2|²
        t2 = _clamp(float(np.dot(p - r, d2)) / len2_sq, 0.0, 1.0)
        cp_e1 = p.copy()
        cp_e2 = r + t2 * d2
        dist = float(np.linalg.norm(cp_e1 - cp_e2))
        feature, feat_idx = _feature_on_edge(t2)
        return dist, cp_e1, t1, cp_e2, t2, feature, feat_idx

    # ----------------------------------------------------------
    # Special case: e2 is a point (zero length)
    # ----------------------------------------------------------
    if len2_sq < EPS:
        t2 = 0.0
        # Project r onto e1
        t1 = _clamp(float(np.dot(r - p, d1)) / len1_sq, 0.0, 1.0)
        cp_e1 = p + t1 * d1
        cp_e2 = r.copy()
        dist = float(np.linalg.norm(cp_e1 - cp_e2))
        return dist, cp_e1, t1, cp_e2, t2, ClosestFeatureOnEdge.VERTEX, 0

    # ----------------------------------------------------------
    # General case: both edges have positive length
    # ----------------------------------------------------------
    # We want to minimize ||(p + t1*d1) - (r + t2*d2)||²
    # Taking partial derivatives and setting to zero gives:
    #
    #   len1_sq * t1  -  dot(d1,d2) * t2  =  dot(d1, r-p)
    #   dot(d1,d2) * t1  -  len2_sq * t2  =  dot(d2, r-p)
    #
    # This is a 2×2 system in t1 and t2.

    b_vec    = r - p
    d1_dot_d2 = float(np.dot(d1, d2))   # how parallel are the edges
    rhs1     = float(np.dot(d1, b_vec))
    rhs2     = float(np.dot(d2, b_vec))

    denom = len1_sq * len2_sq - d1_dot_d2 * d1_dot_d2

    if abs(denom) > EPS:
        # Edges are not parallel → unique unconstrained solution
        t1 = _clamp((len2_sq * rhs1 - d1_dot_d2 * rhs2) / denom, 0.0, 1.0)
    else:
        # Edges are parallel (or nearly so) → pick t1 = 0 arbitrarily
        t1 = 0.0

    # Now find the best t2 for this t1, then clamp and re-solve if needed
    t2 = (d1_dot_d2 * t1 - rhs2) / len2_sq

    if t2 < 0.0:
        # t2 clamped to 0 → re-solve t1 for t2 = 0
        t2 = 0.0
        t1 = _clamp(rhs1 / len1_sq, 0.0, 1.0)
    elif t2 > 1.0:
        # t2 clamped to 1 → re-solve t1 for t2 = 1
        t2 = 1.0
        t1 = _clamp((rhs1 + d1_dot_d2) / len1_sq, 0.0, 1.0)

    cp_e1 = p + t1 * d1
    cp_e2 = r + t2 * d2
    dist  = float(np.linalg.norm(cp_e1 - cp_e2))

    feature, feat_idx = _feature_on_edge(t2)
    return dist, cp_e1, t1, cp_e2, t2, feature, feat_idx


def _feature_on_edge(t, eps=1e-8):
    """
    Given a parameter t ∈ [0, 1] on an edge, decide whether the
    point is at a vertex endpoint or in the interior.

        t ≈ 0   → VERTEX, index 0   (at the first endpoint)
        t ≈ 1   → VERTEX, index 1   (at the second endpoint)
        else    → INTERIOR, index -1
    """
    if t <= eps:
        return ClosestFeatureOnEdge.VERTEX, 0
    if t >= 1.0 - eps:
        return ClosestFeatureOnEdge.VERTEX, 1
    return ClosestFeatureOnEdge.INTERIOR, -1
