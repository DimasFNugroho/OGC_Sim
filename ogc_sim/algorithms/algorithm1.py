"""
Algorithm 1 — Vertex-Facet Contact Detection
=============================================
Paper reference: Sec. 4.1, Algorithm 1.

For a single query vertex v, find all mesh features (faces, edges,
vertices) that are within contact radius r and pass the feasibility
gate (Eq. 8, 9).

Returns
-------
FOGC(v)  — the set of global feature indices in contact with v
d_min_v  — minimum distance from v to any non-adjacent face (<= r_q)

Algorithm structure
-------------------
  1   d_min_v = r_q                                     (line 1)
  2   candidates = BVH sphere query at x(v), radius r_q (line 2)
  3   for each triangle t in candidates:
  4     if v in t: continue                              (line 3 — skip adjacent)
  5     d, cp, feature, feat_idx = distance(v, t)
  6     d_min_v = min(d_min_v, d)                        (line 5-6)
  7     d_min_t[t] = min(d_min_t[t], d)
  8     if d >= r: continue                              (line 7)
  9     global_feat = map_to_global(feature, feat_idx, t)
 10     if global_feat in FOGC: continue                 (line 9 — dedup)
 11     if feasibility_check(feature, direction, global_feat):  (lines 10-19)
 12       FOGC.append(global_feat)
"""

from __future__ import annotations

import numpy as np

from ogc_sim.geometry.mesh import Mesh
from ogc_sim.geometry.bvh import BVH
from ogc_sim.geometry.gauss_map import PolyhedralGaussMap
from ogc_sim.geometry.distance import point_triangle_distance, ClosestFeature
from ogc_sim.contact.detection import (
    vertex_facet_contact_detection as _vf_detect_core,
    run_contact_detection,
    ContactSets,
)


def vertex_facet_contact_detection(
    v_idx: int,
    mesh: Mesh,
    bvh: BVH,
    pgm: PolyhedralGaussMap,
    r: float,
    r_q: float,
    d_min_t: dict[int, float] | None = None,
) -> tuple[list[int], list[int], float]:
    """
    Algorithm 1: vertexFacetContactDetection for a single vertex.

    Parameters
    ----------
    v_idx : int         index of the query vertex in mesh.V
    mesh  : Mesh        the mesh containing both query and target geometry
    bvh   : BVH         broadphase acceleration structure
    pgm   : PolyhedralGaussMap   for feasibility checks (Eq. 8, 9)
    r     : float       contact radius — contacts reported when d < r
    r_q   : float       query radius — BVH query radius; r_q >= r
    d_min_t : dict or None
        Shared mutable dict for per-triangle min distances.
        If None, a fresh dict is created internally.

    Returns
    -------
    fogc_v  : list[int]   global feature indices in contact with v
    vogc_t  : list[int]   triangle indices where v is a contacting vertex
    d_min_v : float       min distance from v to any non-adjacent face
    """
    if d_min_t is None:
        d_min_t = {t: r_q for t in range(mesh.num_triangles)}

    return _vf_detect_core(v_idx, mesh, bvh, pgm, r, r_q, d_min_t)


def run_all_vertices(
    mesh: Mesh,
    bvh: BVH,
    pgm: PolyhedralGaussMap,
    r: float,
    r_q: float,
) -> ContactSets:
    """
    Run Algorithm 1 over all vertices of the mesh.

    This is the vertex-facet portion of run_contact_detection
    (Algorithm 3, lines 9-11).

    Returns a ContactSets with FOGC, VOGC, d_min_v, and d_min_t populated.
    EOGC and d_min_e are left empty (use algorithm2.run_all_edges for those).
    """
    cs = ContactSets()

    # Initialise d_min_t to r_q for all triangles (Algorithm 3, lines 6-8)
    for t_idx in range(mesh.num_triangles):
        cs.d_min_t[t_idx] = r_q

    for v_idx in range(mesh.num_vertices):
        fogc_v, vogc_t, d_min_v = _vf_detect_core(
            v_idx, mesh, bvh, pgm, r, r_q, cs.d_min_t
        )
        cs.FOGC[v_idx] = fogc_v
        cs.d_min_v[v_idx] = d_min_v
        for t_idx in vogc_t:
            cs.VOGC.setdefault(t_idx, []).append(v_idx)

    return cs
