"""
Algorithm 2 — Edge-Edge Contact Detection
==========================================
Paper reference: Sec. 4.2, Algorithm 2.

For a single query edge e, find all other edges that are within
contact radius r and pass the feasibility gate (Eq. 15).

Returns
-------
EOGC(e)  — the set of (e, e') pairs in contact
d_min_e  — minimum distance from e to any non-adjacent edge (<= r_q)

Algorithm structure
-------------------
  1   d_min_e = r_q                                         (line 1)
  2   midpoint = (e.p + e.q) / 2
  3   candidates = BVH sphere query at midpoint, radius r_q + l/2  (line 4)
  4   for each edge e' in candidates:
  5     if e == e': continue
  6     if e and e' share a vertex: continue                 (line 5 — skip adjacent)
  7     d = edge_edge_distance(e, e')
  8     d_min_e = min(d_min_e, d)                            (line 7)
  9     if d >= r: continue                                  (line 8)
 10     global_feat = map_to_global(feature on e')
 11     if global_feat in EOGC: continue                     (line 11 — dedup)
 12     if feasibility_check(feature, direction, global_feat): (lines 12-16)
 13       EOGC.append((e, e'))
"""

from __future__ import annotations

import numpy as np

from ogc_sim.geometry.mesh import Mesh
from ogc_sim.geometry.bvh import BVH
from ogc_sim.geometry.gauss_map import PolyhedralGaussMap
from ogc_sim.contact.detection import (
    edge_edge_contact_detection as _ee_detect_core,
    ContactSets,
)


def edge_edge_contact_detection(
    e_idx: int,
    mesh: Mesh,
    bvh: BVH,
    pgm: PolyhedralGaussMap,
    r: float,
    r_q: float,
) -> tuple[list[tuple[int, int]], float]:
    """
    Algorithm 2: edgeEdgeContactDetection for a single edge.

    Parameters
    ----------
    e_idx : int         index of the query edge in mesh.E
    mesh  : Mesh        the mesh containing both query and target geometry
    bvh   : BVH         broadphase acceleration structure
    pgm   : PolyhedralGaussMap   for feasibility checks (Eq. 15)
    r     : float       contact radius
    r_q   : float       query radius (r_q >= r)

    Returns
    -------
    eogc_e  : list of (e_idx, e'_idx) pairs in contact with e
    d_min_e : float     min distance from e to any non-adjacent edge
    """
    return _ee_detect_core(e_idx, mesh, bvh, pgm, r, r_q)


def run_all_edges(
    mesh: Mesh,
    bvh: BVH,
    pgm: PolyhedralGaussMap,
    r: float,
    r_q: float,
    cs: ContactSets | None = None,
) -> ContactSets:
    """
    Run Algorithm 2 over all edges of the mesh.

    This is the edge-edge portion of run_contact_detection
    (Algorithm 3, lines 12-14).

    Parameters
    ----------
    cs : ContactSets or None
        If provided, EOGC and d_min_e are added to this existing object
        (useful when combining with Algorithm 1 results). If None, a new
        ContactSets is created.

    Returns a ContactSets with EOGC and d_min_e populated.
    """
    if cs is None:
        cs = ContactSets()

    for e_idx in range(mesh.num_edges):
        eogc_e, d_min_e = _ee_detect_core(e_idx, mesh, bvh, pgm, r, r_q)
        cs.EOGC[e_idx] = eogc_e
        cs.d_min_e[e_idx] = d_min_e

    return cs
