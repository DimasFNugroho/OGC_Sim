"""
Contact Detection — Algorithm 1 (vertex-facet) and Algorithm 2 (edge-edge).

Paper reference: Sec. 4.1, 4.2.

This module answers the question: "which surface features are close enough
to be in contact right now?"  It produces two contact sets per vertex/edge:

    FOGC(v)  — the set of faces (vertex / edge / triangle) that are in
               contact with query vertex v               (Algorithm 1)
    VOGC(t)  — the set of vertices that are in contact with triangle t
               (built as a by-product of FOGC)
    EOGC(e)  — the set of face-edges that are in contact with edge e
               (Algorithm 2)

These sets are consumed by the VBD solver (Algorithm 4) to accumulate
contact forces and Hessians.

Key design decisions
--------------------
* Only a BVH over triangles (for vertex-facet) and edges (for edge-edge)
  is needed — not one per offset block.  Paper Sec. 4.
* d_min_v / d_min_t / d_min_e are computed as a by-product of the same
  BVH traversal and feed into the conservative bound (Eq. 21).
* d_min_t is updated atomically in the GPU version; here we use a plain
  dict that is mutated by the single-threaded Python loop.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from ogc_sim.geometry.mesh import Mesh
from ogc_sim.geometry.bvh import BVH
from ogc_sim.geometry.gauss_map import PolyhedralGaussMap
from ogc_sim.geometry.distance import (
    point_triangle_distance,
    edge_edge_distance,
    ClosestFeature,
)
from ogc_sim.contact.offset_geometry import (
    feasible_vf_contact,
    feasible_ee_contact,
)


# ======================================================================
# Data containers
# ======================================================================

@dataclass
class ContactSets:
    """
    All contact sets produced by one round of contact detection.

    Attributes
    ----------
    FOGC : dict[int, list[int]]
        FOGC[v] = list of global feature indices in contact with vertex v.
        A feature index is:
          - a triangle index  if the closest point was on a face interior
          - an edge index     if the closest point was on an edge
          - a vertex index    if the closest point was at a vertex
    VOGC : dict[int, list[int]]
        VOGC[t] = list of vertex indices whose query hit triangle t.
        (Used by the solver to accumulate face-side contact forces.)
    EOGC : dict[int, list[tuple[int,int]]]
        EOGC[e] = list of (e, e') pairs where e' is a contacting edge.
    d_min_v : dict[int, float]
        d_min_v[v] = min distance from v to any non-adjacent face (≤ r_q).
    d_min_t : dict[int, float]
        d_min_t[t] = min distance from triangle t to any non-adjacent vertex.
    d_min_e : dict[int, float]
        d_min_e[e] = min distance from edge e to any non-adjacent edge.
    """

    FOGC:    dict[int, list[int]]            = field(default_factory=dict)
    VOGC:    dict[int, list[int]]            = field(default_factory=dict)
    EOGC:    dict[int, list[tuple[int,int]]] = field(default_factory=dict)
    d_min_v: dict[int, float]                = field(default_factory=dict)
    d_min_t: dict[int, float]                = field(default_factory=dict)
    d_min_e: dict[int, float]                = field(default_factory=dict)


# ======================================================================
# Algorithm 1 — vertex-facet contact detection
# ======================================================================

def vertex_facet_contact_detection(
    v_idx: int,
    mesh: Mesh,
    bvh: BVH,
    pgm: PolyhedralGaussMap,
    r: float,
    r_q: float,
    d_min_t: dict[int, float],
) -> tuple[list[int], list[int], float]:
    """
    Algorithm 1: vertexFacetContactDetection for a single vertex.

    For each triangle t within query radius r_q of vertex v:
      1. Skip adjacent triangles (v is a vertex of t).
      2. Update d_min_v and d_min_t (the "how close is anything" bookkeeping).
      3. If d < r (inside contact radius), find the closest sub-feature a.
      4. Skip if a is already in FOGC(v) (de-duplication).
      5. Apply the Gauss Map feasibility check (Eq. 8, 9, or auto-pass).
      6. Record the contact in FOGC(v) and VOGC(t).

    Parameters
    ----------
    v_idx : int
    mesh : Mesh
    bvh : BVH
    pgm : PolyhedralGaussMap
    r : float    contact radius  — contacts reported when d < r
    r_q : float  query radius    — BVH query radius; r_q >= r
    d_min_t : dict[int, float]
        Shared mutable dict updated here (atomic min in the GPU version).

    Returns
    -------
    fogc_v  : list[int]   global feature indices in contact with v
    vogc_v  : list[int]   triangle indices where v is a contacting vertex
              (caller appends v_idx to VOGC[t] for each t in this list)
    d_min_v : float       min distance from v to any non-adjacent face
    """
    v_pos = mesh.V[v_idx]

    fogc_v: list[int]  = []
    vogc_t: list[int]  = []   # which triangles recorded this vertex
    d_min_v = r_q              # Algorithm 1, line 1: initialise to r_q

    # BVH sphere query: all triangles within radius r_q of v  (line 2)
    candidate_tris = bvh.sphere_query_triangles(v_pos, r_q)

    for t_idx in candidate_tris:
        # line 3: skip triangles that contain v (adjacent faces)
        if v_idx in mesh.T[t_idx]:
            continue

        tri = mesh.T[t_idx]
        a_pos = mesh.V[int(tri[0])]
        b_pos = mesh.V[int(tri[1])]
        c_pos = mesh.V[int(tri[2])]

        dist, cp, feature, local_feat_idx = point_triangle_distance(
            v_pos, a_pos, b_pos, c_pos
        )

        # lines 5–6: update per-vertex and per-triangle minimum distances
        d_min_v = min(d_min_v, dist)
        # d_min_t update (atomic in GPU; plain min here)
        if t_idx not in d_min_t or dist < d_min_t[t_idx]:
            d_min_t[t_idx] = dist

        # line 7: only report as contact if d < r
        if dist >= r:
            continue

        # line 8: identify the closest sub-feature a and its global index
        #   feature == FACE_INTERIOR → a = triangle itself (global: t_idx)
        #   feature == EDGE          → a = global edge index
        #   feature == VERTEX        → a = global vertex index
        if feature == ClosestFeature.FACE_INTERIOR:
            global_feat_idx = t_idx
        elif feature == ClosestFeature.EDGE:
            global_feat_idx = mesh.E_t[t_idx][local_feat_idx]
        else:  # VERTEX
            global_feat_idx = int(tri[local_feat_idx])

        # line 9: de-duplication — skip if a is already in FOGC(v)
        # (two adjacent triangles can both report the same shared edge/vertex)
        if global_feat_idx in fogc_v:
            continue

        # lines 10–19: feasibility check
        direction = v_pos - cp

        if feature == ClosestFeature.FACE_INTERIOR:
            # line 18–20: face interior → always feasible
            fogc_v.append(global_feat_idx)
            vogc_t.append(t_idx)

        elif feature == ClosestFeature.VERTEX:
            # line 10–13: Eq. 8 — vertex normal cone check
            if pgm.is_in_vertex_normal_cone(direction, global_feat_idx):
                fogc_v.append(global_feat_idx)
                vogc_t.append(t_idx)

        else:  # EDGE
            # line 14–17: Eq. 9 — edge normal slab check
            if pgm.is_in_edge_normal_slab(direction, global_feat_idx):
                fogc_v.append(global_feat_idx)
                vogc_t.append(t_idx)

    return fogc_v, vogc_t, d_min_v


# ======================================================================
# Algorithm 2 — edge-edge contact detection
# ======================================================================

def edge_edge_contact_detection(
    e_idx: int,
    mesh: Mesh,
    bvh: BVH,
    pgm: PolyhedralGaussMap,
    r: float,
    r_q: float,
) -> tuple[list[tuple[int,int]], float]:
    """
    Algorithm 2: edgeEdgeContactDetection for a single edge.

    Parameters
    ----------
    e_idx : int
    mesh : Mesh
    bvh : BVH
    pgm : PolyhedralGaussMap
    r : float    contact radius
    r_q : float  query radius

    Returns
    -------
    eogc_e  : list of (e_idx, e'_idx) pairs — edges in contact with e
    d_min_e : float — min distance from e to any non-adjacent edge
    """
    ea = mesh.V[int(mesh.E[e_idx][0])]
    eb = mesh.V[int(mesh.E[e_idx][1])]
    midpoint = (ea + eb) * 0.5
    half_len = float(np.linalg.norm(eb - ea)) * 0.5

    eogc_e: list[tuple[int,int]] = []
    d_min_e = r_q  # Algorithm 2, line 1

    # BVH sphere query centred at midpoint with radius r_q + l/2  (line 4)
    query_radius = r_q + half_len
    candidate_edges = bvh.sphere_query_edges(midpoint, query_radius)

    for e2_idx in candidate_edges:
        if e2_idx == e_idx:
            continue

        # line 5: skip adjacent edges (share a vertex)
        e1_verts = {int(mesh.E[e_idx][0]),  int(mesh.E[e_idx][1])}
        e2_verts = {int(mesh.E[e2_idx][0]), int(mesh.E[e2_idx][1])}
        if e1_verts & e2_verts:
            continue

        ra = mesh.V[int(mesh.E[e2_idx][0])]
        rb = mesh.V[int(mesh.E[e2_idx][1])]
        dist, *_ = edge_edge_distance(ea, eb, ra, rb)

        # line 7: update d_min_e
        d_min_e = min(d_min_e, dist)

        # line 8: only report contact if d < r
        if dist >= r:
            continue

        # line 11: de-duplication check
        if (e_idx, e2_idx) in eogc_e:
            continue

        # Feasibility check (Eq. 15 / Sec. 3.5)
        result = feasible_ee_contact(e_idx, e2_idx, mesh, pgm)
        if result.feasible:
            eogc_e.append((e_idx, e2_idx))

    return eogc_e, d_min_e


# ======================================================================
# Full mesh sweep
# ======================================================================

def run_contact_detection(
    mesh: Mesh,
    bvh: BVH,
    pgm: PolyhedralGaussMap,
    r: float,
    r_q: float,
) -> ContactSets:
    """
    Run Algorithm 1 over all vertices and Algorithm 2 over all edges.

    In the GPU version these loops are fully parallel (Algorithm 3, lines
    9–14). Here they run sequentially; the logic is identical.

    Parameters
    ----------
    mesh : Mesh
    bvh : BVH       pre-built; call bvh.refit() first if vertices moved
    pgm : PolyhedralGaussMap
    r : float       contact radius
    r_q : float     query radius (r_q >= r)

    Returns
    -------
    ContactSets
    """
    cs = ContactSets()

    # Initialise d_min_t to r_q for all triangles  (Algorithm 3, lines 6–8)
    for t_idx in range(mesh.num_triangles):
        cs.d_min_t[t_idx] = r_q

    # Algorithm 1: one call per vertex  (Algorithm 3, lines 9–11)
    for v_idx in range(mesh.num_vertices):
        fogc_v, vogc_t, d_min_v = vertex_facet_contact_detection(
            v_idx, mesh, bvh, pgm, r, r_q, cs.d_min_t
        )
        cs.FOGC[v_idx]    = fogc_v
        cs.d_min_v[v_idx] = d_min_v
        for t_idx in vogc_t:
            cs.VOGC.setdefault(t_idx, []).append(v_idx)

    # Algorithm 2: one call per edge  (Algorithm 3, lines 12–14)
    for e_idx in range(mesh.num_edges):
        eogc_e, d_min_e = edge_edge_contact_detection(
            e_idx, mesh, bvh, pgm, r, r_q
        )
        cs.EOGC[e_idx]    = eogc_e
        cs.d_min_e[e_idx] = d_min_e

    return cs
