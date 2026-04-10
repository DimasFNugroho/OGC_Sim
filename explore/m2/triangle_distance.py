"""
triangle_distance.py
====================
A human-readable implementation of point-to-triangle distance.

This file is self-contained — no imports from ogc_sim.
It is meant to be read alongside learn_algorithm1.py so you can
understand exactly how the distance query works before trusting
it as a black box.

The one function you need:

    dist, cp, feature, feat_idx = point_triangle_distance(p, a, b, c)

Everything below builds up to that function step by step.
"""

import numpy as np


# ============================================================
# ClosestFeature
# ============================================================
# When the closest point on the triangle is found, we also want
# to know WHERE on the triangle it landed.
#
# There are exactly three possibilities:
#
#   FACE_INTERIOR  — the closest point is somewhere inside the face
#   EDGE           — the closest point is on one of the three edges
#                    (not at a corner)
#   VERTEX         — the closest point is at one of the three corners
#
# This information is used later in Algorithm 1 to decide which
# feasibility check (Eq. 8 or Eq. 9) to apply.
# ============================================================

class ClosestFeature:
    FACE_INTERIOR = "FACE_INTERIOR"
    EDGE          = "EDGE"
    VERTEX        = "VERTEX"


# ============================================================
# The main idea: Voronoi regions
# ============================================================
#
# Imagine projecting point p straight onto the plane of the triangle.
# The triangle divides the plane into 7 zones (Voronoi regions):
#
#        c
#       / \
#      /   \
#     / tri \
#    a-------b
#
#   • 3 corner regions   (one per vertex a, b, c)
#   • 3 edge regions     (one per edge ab, bc, ca)
#   • 1 interior region  (inside the triangle)
#
# The closest point on the triangle to p is determined by which
# region p's projection falls in:
#
#   corner region of a  → closest point = a  (VERTEX, index 0)
#   corner region of b  → closest point = b  (VERTEX, index 1)
#   corner region of c  → closest point = c  (VERTEX, index 2)
#   edge region of ab   → closest point = projection onto ab  (EDGE, index 0)
#   edge region of bc   → closest point = projection onto bc  (EDGE, index 1)
#   edge region of ca   → closest point = projection onto ca  (EDGE, index 2)
#   interior region     → closest point = the projection itself (FACE_INTERIOR)
#
# We test each region using dot products (no square roots until the end).
# ============================================================


def point_triangle_distance(p, a, b, c):
    """
    Find the closest point on triangle (a, b, c) to point p.

    Parameters
    ----------
    p : np.ndarray, shape (3,)   — the query point
    a : np.ndarray, shape (3,)   — first  triangle vertex
    b : np.ndarray, shape (3,)   — second triangle vertex
    c : np.ndarray, shape (3,)   — third  triangle vertex

    Returns
    -------
    dist      : float            — distance from p to the triangle
    cp        : np.ndarray (3,)  — the closest point ON the triangle
    feature   : ClosestFeature   — where cp landed (VERTEX/EDGE/FACE_INTERIOR)
    feat_idx  : int
        VERTEX        → 0, 1, or 2  (which vertex: a, b, or c)
        EDGE          → 0, 1, or 2  (which edge: ab, bc, or ca)
        FACE_INTERIOR → -1  (not needed)

    How to read the code
    --------------------
    We work with DOT PRODUCTS, not coordinates directly.
    The key insight: "p is in the Voronoi region of vertex a"
    means that moving from a toward p makes you go AWAY from
    both edges ab and ac.

    In dot-product terms:
        dot(ab, ap) <= 0   ← p is behind the ab direction
        dot(ac, ap) <= 0   ← p is behind the ac direction
    Both conditions together → p is in vertex a's region.
    """

    # -- Vectors from vertex a to the other two vertices and to p --
    ab = b - a    # edge a→b
    ac = c - a    # edge a→c
    ap = p - a    # vector from a to the query point

    # -- Test: is p in the corner region of vertex a? --
    # This is true when p is "behind" both edges leaving a.
    # "Behind ab" means: if you stand at a and look toward b,
    # p is behind you  →  dot(ab, ap) <= 0
    dot_ab_ap = np.dot(ab, ap)   # d1 in the reference code
    dot_ac_ap = np.dot(ac, ap)   # d2

    if dot_ab_ap <= 0.0 and dot_ac_ap <= 0.0:
        # p is in the corner region of a
        # → closest point is a itself
        cp = a.copy()
        return float(np.linalg.norm(p - cp)), cp, ClosestFeature.VERTEX, 0

    # -- Test: is p in the corner region of vertex b? --
    bp = p - b
    dot_ab_bp = np.dot(ab, bp)   # d3
    dot_ac_bp = np.dot(ac, bp)   # d4

    if dot_ab_bp >= 0.0 and dot_ac_bp <= dot_ab_bp:
        cp = b.copy()
        return float(np.linalg.norm(p - cp)), cp, ClosestFeature.VERTEX, 1

    # -- Test: is p in the corner region of vertex c? --
    cp_vec = p - c
    dot_ab_cp = np.dot(ab, cp_vec)   # d5
    dot_ac_cp = np.dot(ac, cp_vec)   # d6

    if dot_ac_cp >= 0.0 and dot_ab_cp <= dot_ac_cp:
        cp = c.copy()
        return float(np.linalg.norm(p - cp)), cp, ClosestFeature.VERTEX, 2

    # -- Test: is p in the edge region of ab? --
    # The edge region of ab is the strip beside ab, between the
    # perpendiculars at a and at b, on the outside of the triangle.
    #
    # We test this with a "signed area" trick: compute vc, the
    # component of p that is "outside" the ab edge.
    # vc <= 0 means p is on the outside of ab (in the edge strip or beyond).
    vc = dot_ab_ap * dot_ac_bp - dot_ab_bp * dot_ac_ap

    if vc <= 0.0 and dot_ab_ap >= 0.0 and dot_ab_bp <= 0.0:
        # Project p onto the edge ab
        # t is how far along ab the closest point is: 0=at a, 1=at b
        t  = dot_ab_ap / (dot_ab_ap - dot_ab_bp)
        cp = a + t * ab
        return float(np.linalg.norm(p - cp)), cp, ClosestFeature.EDGE, 0

    # -- Test: is p in the edge region of ca? --
    vb = dot_ab_cp * dot_ac_ap - dot_ab_ap * dot_ac_cp

    if vb <= 0.0 and dot_ac_ap >= 0.0 and dot_ac_cp <= 0.0:
        t  = dot_ac_ap / (dot_ac_ap - dot_ac_cp)
        cp = a + t * ac
        return float(np.linalg.norm(p - cp)), cp, ClosestFeature.EDGE, 2

    # -- Test: is p in the edge region of bc? --
    va = dot_ab_bp * dot_ac_cp - dot_ab_cp * dot_ac_bp

    if va <= 0.0 and (dot_ac_bp - dot_ab_bp) >= 0.0 and (dot_ab_cp - dot_ac_cp) >= 0.0:
        t  = (dot_ac_bp - dot_ab_bp) / ((dot_ac_bp - dot_ab_bp) + (dot_ab_cp - dot_ac_cp))
        cp = b + t * (c - b)
        return float(np.linalg.norm(p - cp)), cp, ClosestFeature.EDGE, 1

    # -- Interior: p projects inside the triangle --
    # If none of the above regions matched, p is directly above (or below)
    # the interior of the triangle.  The closest point is its projection.
    #
    # We reconstruct the projection using barycentric coordinates:
    #   cp = a + v*ab + w*ac
    # where v and w are the barycentric weights for b and c.
    total = va + vb + vc
    v  = vb / total
    w  = vc / total
    cp = a + v * ab + w * ac
    return float(np.linalg.norm(p - cp)), cp, ClosestFeature.FACE_INTERIOR, -1
