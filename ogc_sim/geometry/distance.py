"""
Point-triangle and edge-edge distance functions.

Every contact detection query ultimately bottoms out in one of these two
primitives.  Both return not just the distance but also the closest point
and which sub-feature (vertex / edge / face interior) of the triangle or
edge contains that closest point — information needed by the offset-geometry
feasibility checks in contact/offset_geometry.py.

Paper reference: Sec. 3.2–3.5 (closest-point geometry underlies all block
definitions).
"""

from __future__ import annotations
from enum import Enum, auto

import numpy as np


class ClosestFeature(Enum):
    """Which sub-feature of a triangle holds the closest point to a query."""
    VERTEX = auto()        # closest point is at a triangle vertex
    EDGE = auto()          # closest point is on a triangle edge (not at endpoints)
    FACE_INTERIOR = auto() # closest point is strictly inside the triangle


def point_triangle_distance(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> tuple[float, np.ndarray, ClosestFeature, int]:
    """
    Compute the distance from point p to triangle (a, b, c).

    The triangle vertices are given in counter-clockwise order.

    Parameters
    ----------
    p : np.ndarray, shape (3,)
        Query point.
    a, b, c : np.ndarray, shape (3,)
        Triangle vertices (CCW winding).

    Returns
    -------
    dist : float
        Euclidean distance from p to the closest point on the triangle.
    closest_pt : np.ndarray, shape (3,)
        The closest point on the triangle surface to p.
    feature : ClosestFeature
        Whether the closest point lies at a vertex, on an edge, or in the
        face interior.
    feature_index : int
        - VERTEX: 0, 1, or 2 for vertex a, b, c respectively.
        - EDGE:   0, 1, or 2 for edge (a,b), (b,c), (c,a) respectively.
        - FACE_INTERIOR: -1 (unused).

    Notes
    -----
    A robust implementation uses barycentric coordinates.  Project p onto
    the plane of the triangle, express the projection in barycentric coords,
    and clamp to the triangle.  The Voronoi-region approach from
    Ericson "Real-Time Collision Detection" Ch. 5 is a clean reference.

    Steps:
    1. Compute edge vectors ab = b-a, ac = c-a.
    2. Compute barycentric coords (s, t) of the projection of p onto the
       plane spanned by ab and ac.
    3. Clamp (s, t) to the triangle domain and identify the active Voronoi
       region to set `feature` and `feature_index`.
    4. Reconstruct the closest point and compute the distance.
    """
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    # Voronoi region of vertex a
    if d1 <= 0.0 and d2 <= 0.0:
        return np.linalg.norm(p - a), a.copy(), ClosestFeature.VERTEX, 0

    bp = p - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    # Voronoi region of vertex b
    if d3 >= 0.0 and d4 <= d3:
        return np.linalg.norm(p - b), b.copy(), ClosestFeature.VERTEX, 1

    cp = p - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    # Voronoi region of vertex c
    if d6 >= 0.0 and d5 <= d6:
        return np.linalg.norm(p - c), c.copy(), ClosestFeature.VERTEX, 2

    # Voronoi region of edge ab (edge index 0)
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        t = d1 / (d1 - d3)
        closest = a + t * ab
        return np.linalg.norm(p - closest), closest, ClosestFeature.EDGE, 0

    # Voronoi region of edge ac / ca (edge index 2: c->a)
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        t = d2 / (d2 - d6)
        closest = a + t * ac
        return np.linalg.norm(p - closest), closest, ClosestFeature.EDGE, 2

    # Voronoi region of edge bc (edge index 1)
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        closest = b + t * (c - b)
        return np.linalg.norm(p - closest), closest, ClosestFeature.EDGE, 1

    # Interior of triangle
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    closest = a + v * ab + w * ac
    return np.linalg.norm(p - closest), closest, ClosestFeature.FACE_INTERIOR, -1


def edge_edge_distance(
    p: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    s: np.ndarray,
) -> tuple[float, np.ndarray, float, np.ndarray, float]:
    """
    Compute the distance between line segment pq and line segment rs.

    Parameters
    ----------
    p, q : np.ndarray, shape (3,)
        Endpoints of the first edge.
    r, s : np.ndarray, shape (3,)
        Endpoints of the second edge.

    Returns
    -------
    dist       : float          Minimum distance between the two segments.
    closest_pq : np.ndarray     Closest point on pq  (contact point x_c, Algorithm 2 line 9).
    t          : float          Parameter along pq  (0 = at p, 1 = at q).
    closest_rs : np.ndarray     Closest point on rs.
    u          : float          Parameter along rs  (0 = at r, 1 = at s).

    Notes
    -----
    General approach:
    1. Parameterise pq as p + t*(q-p) and rs as r + u*(s-r).
    2. Minimise ||(p + t*d1) - (r + u*d2)||² over t, u ∈ [0, 1].
    3. The unconstrained minimum comes from solving a 2×2 linear system.
    4. Clamp t and u to [0, 1] and handle degenerate (parallel/zero-length)
       segments as special cases.

    The clamping order matters: clamp t first, recompute u; if u is out of
    range clamp u and recompute t.  See Ericson Ch. 5 for the full case
    analysis.
    """
    d1 = q - p
    d2 = s - r
    r_vec = p - r
    a = np.dot(d1, d1)  # |d1|^2
    e = np.dot(d2, d2)  # |d2|^2
    f = np.dot(d2, r_vec)

    EPSILON = 1e-10

    if a <= EPSILON and e <= EPSILON:
        # Both segments degenerate to points
        return np.linalg.norm(p - r), p.copy(), 0.0, r.copy(), 0.0

    if a <= EPSILON:
        # First segment degenerates to a point
        t = 0.0
        u = _clamp(f / e, 0.0, 1.0)
    else:
        c = np.dot(d1, r_vec)
        if e <= EPSILON:
            # Second segment degenerates to a point
            u = 0.0
            t = _clamp(-c / a, 0.0, 1.0)
        else:
            # General non-degenerate case
            b = np.dot(d1, d2)
            denom = a * e - b * b
            if abs(denom) > EPSILON:
                t = _clamp((b * f - c * e) / denom, 0.0, 1.0)
            else:
                t = 0.0  # parallel segments — pick arbitrary t
            # Compute u for this t, then clamp and recompute t if needed
            u = (b * t + f) / e
            if u < 0.0:
                u = 0.0
                t = _clamp(-c / a, 0.0, 1.0)
            elif u > 1.0:
                u = 1.0
                t = _clamp((b - c) / a, 0.0, 1.0)

    closest_pq = p + t * d1
    closest_rs = r + u * d2
    return np.linalg.norm(closest_pq - closest_rs), closest_pq, t, closest_rs, u


# ------------------------------------------------------------------
# Internal helpers (implement these first — point_triangle uses them)
# ------------------------------------------------------------------

def _project_point_onto_plane(
    p: np.ndarray,
    origin: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    """
    Project p onto the plane defined by a point `origin` and unit `normal`.

    Returns
    -------
    np.ndarray, shape (3,)
        The projected point.
    """
    return p - np.dot(p - origin, normal) * normal


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp scalar value to [lo, hi]."""
    return max(lo, min(hi, value))
