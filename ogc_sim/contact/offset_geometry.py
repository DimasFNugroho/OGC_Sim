"""
Offset Geometry — feasibility region checks.

Paper reference: Sec. 3.2–3.5.

The key idea of OGC is that the offset surface of a polyhedral mesh is built
from three types of "blocks," one per topological feature of the surface:

    Face-interior block  U_t :  the region directly above face t within
                                distance r.  Normal is unique → always feasible.
    Edge block           U_e :  the region beside edge e within distance r,
                                restricted to the edge's normal slab.
    Vertex block         U_v :  the region around vertex v within distance r,
                                restricted to the vertex's normal cone.

Together, these blocks tile the offset surface without gaps or overlaps.
A query point x is in exactly one block: the one whose feature's normal set
contains the direction (x - closest_point_on_surface).

Contact detection uses this tiling to assign each contact unambiguously:
  · Point-triangle distance  → closest point cp  and  feature type F
  · Feasibility check (Eq. 8 or 9) → is (query - cp) in the normal set of F?

If the answer is "no," the contact point cp lies at a topological boundary
between features and belongs to a different, adjacent feature's block.  The
query vertex will be picked up when *that* feature is tested.

Paper notation used in this file
---------------------------------
  Eq. 8   vertex-block check    direction ∈ normal_cone(v)
  Eq. 9   edge-block check      direction ∈ normal_slab(e)
  Eq. 15  edge-edge feasibility  both interior + slab test on target edge
  Sec. 3.5  edge-edge contacts
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from ogc_sim.geometry.mesh import Mesh
from ogc_sim.geometry.gauss_map import PolyhedralGaussMap
from ogc_sim.geometry.distance import (
    point_triangle_distance,
    ClosestFeature,
)


# ======================================================================
# Per-feature feasibility checks
# ======================================================================

def check_vertex_feasible_region(
    x: np.ndarray,
    v_idx: int,
    mesh: Mesh,
    pgm: PolyhedralGaussMap,
) -> bool:
    """
    Return True if query point x lies in the offset block of vertex v_idx.

    The vertex block U_v covers all points whose direction from v lies
    inside the normal cone of v:

        direction = x − V[v_idx]
        feasible  ⟺  direction ∈ normal_cone(v)          (Eq. 8)

    Geometric intuition:
      Think of the normal cone as the set of "outward" directions that are
      consistent with the local curvature at v.  A query point that is on the
      same side as the outward normals of all incident faces lies inside the
      vertex block; anything outside the cone naturally falls into an adjacent
      edge or face block.

    Parameters
    ----------
    x : np.ndarray, shape (3,)
        Query point (typically a cloth vertex approaching the mesh surface).
    v_idx : int
        Target vertex index in `mesh`.
    mesh : Mesh
    pgm : PolyhedralGaussMap

    Returns
    -------
    bool
    """
    direction = x - mesh.V[v_idx]
    return pgm.is_in_vertex_normal_cone(direction, v_idx)


def check_edge_feasible_region(
    x: np.ndarray,
    e_idx: int,
    mesh: Mesh,
    pgm: PolyhedralGaussMap,
) -> bool:
    """
    Return True if query point x lies in the offset block of edge e_idx.

    The edge block U_e covers all points whose closest point on the edge is
    *interior* (not at an endpoint) AND whose direction from that closest
    point lies inside the edge's normal slab:

        closest_pt  = projection of x onto edge (t ∈ (0, 1))
        direction   = x − closest_pt
        feasible    ⟺  t ∈ (0, 1)  AND  direction ∈ normal_slab(e)   (Eq. 9)

    Geometric intuition:
      The normal slab is the set of directions that are consistent with both
      adjacent face normals — it is the "wedge" of directions that point away
      from the edge ridge (or into the valley, for concave edges).  The t ∈ (0,1)
      requirement ensures that endpoint-adjacent contacts are handled by the
      vertex blocks of those endpoints instead.

    Parameters
    ----------
    x : np.ndarray, shape (3,)
    e_idx : int
    mesh : Mesh
    pgm : PolyhedralGaussMap

    Returns
    -------
    bool
    """
    a_idx = int(mesh.E[e_idx][0])
    b_idx = int(mesh.E[e_idx][1])
    a, b  = mesh.V[a_idx], mesh.V[b_idx]

    ab    = b - a
    ab_sq = float(np.dot(ab, ab))

    if ab_sq < 1e-20:
        return False  # degenerate zero-length edge

    t = float(np.dot(x - a, ab) / ab_sq)

    # The endpoint guard: if closest point is at or beyond an endpoint,
    # the contact belongs to that endpoint's vertex block, not this edge block.
    _EPS = 1e-8
    if t <= _EPS or t >= 1.0 - _EPS:
        return False

    closest_pt = a + t * ab
    direction  = x - closest_pt

    return pgm.is_in_edge_normal_slab(direction, e_idx)


# ======================================================================
# Vertex-facet contact feasibility
# ======================================================================

@dataclass
class VFContactResult:
    """
    Result of a single vertex-facet feasibility check.

    Attributes
    ----------
    feasible : bool
        True if the query vertex makes a valid contact with this triangle.
    distance : float
        Distance from the query vertex to the closest point on the triangle.
    contact_point : np.ndarray, shape (3,)
        Closest point on the triangle surface.
    feature : ClosestFeature
        Which sub-feature (VERTEX / EDGE / FACE_INTERIOR) owns the contact.
    global_feature_idx : int
        Global index in `mesh` for the owning feature:
          FACE_INTERIOR → triangle index
          EDGE          → edge index in mesh.E
          VERTEX        → vertex index in mesh.V
    """

    feasible: bool
    distance: float
    contact_point: np.ndarray
    feature: ClosestFeature
    global_feature_idx: int


def feasible_vf_contact(
    query_vertex: np.ndarray,
    tri_idx: int,
    mesh: Mesh,
    pgm: PolyhedralGaussMap,
) -> VFContactResult:
    """
    Check whether query_vertex makes a feasible contact with triangle tri_idx.

    Algorithm
    ---------
    1. Compute point-triangle distance → (dist, cp, feature, local_feat_idx).
    2. Map local feature index to global mesh index.
    3. Apply the appropriate feasibility check:
         FACE_INTERIOR → always feasible (unique normal, Sec. 3.2)
         EDGE          → Eq. 9  (edge normal slab)
         VERTEX        → Eq. 8  (vertex normal cone)

    The mapping from local (per-triangle) to global (per-mesh) feature indices:
      local vertex  i  →  global vertex  mesh.T[tri_idx][i]
      local edge    i  →  global edge    mesh.E_t[tri_idx][i]
        where local edge 0 = (v0,v1), 1 = (v1,v2), 2 = (v2,v0)

    Parameters
    ----------
    query_vertex : np.ndarray, shape (3,)
    tri_idx : int
    mesh : Mesh
    pgm : PolyhedralGaussMap

    Returns
    -------
    VFContactResult
    """
    tri_verts = mesh.T[tri_idx]
    a = mesh.V[int(tri_verts[0])]
    b = mesh.V[int(tri_verts[1])]
    c = mesh.V[int(tri_verts[2])]

    dist, cp, feature, local_feat_idx = point_triangle_distance(query_vertex, a, b, c)
    direction = query_vertex - cp

    if feature == ClosestFeature.FACE_INTERIOR:
        # One well-defined outward normal → contact is always valid here.
        global_feat_idx = tri_idx
        feasible = True

    elif feature == ClosestFeature.EDGE:
        # local_feat_idx ∈ {0,1,2}: edge (v0,v1), (v1,v2), (v2,v0)
        # mesh.E_t[tri_idx] lists the three global edge indices in the same order.
        global_feat_idx = mesh.E_t[tri_idx][local_feat_idx]
        feasible = pgm.is_in_edge_normal_slab(direction, global_feat_idx)  # Eq. 9

    else:  # ClosestFeature.VERTEX
        # local_feat_idx ∈ {0,1,2}: vertex a, b, c
        global_feat_idx = int(tri_verts[local_feat_idx])
        feasible = pgm.is_in_vertex_normal_cone(direction, global_feat_idx)  # Eq. 8

    return VFContactResult(
        feasible=feasible,
        distance=dist,
        contact_point=cp,
        feature=feature,
        global_feature_idx=global_feat_idx,
    )


# ======================================================================
# Edge-edge contact feasibility  (Sec. 3.5, Eq. 15)
# ======================================================================

@dataclass
class EEContactResult:
    """
    Result of a single edge-edge feasibility check.

    Attributes
    ----------
    feasible : bool
    distance : float
    contact_point_e1 : np.ndarray, shape (3,)  closest point on e1
    contact_point_e2 : np.ndarray, shape (3,)  closest point on e2
    t1 : float   parameter on e1  (cp_e1 = a1 + t1*(b1-a1))
    t2 : float   parameter on e2  (cp_e2 = a2 + t2*(b2-a2))
    """

    feasible: bool
    distance: float
    contact_point_e1: np.ndarray
    contact_point_e2: np.ndarray
    t1: float
    t2: float


def feasible_ee_contact(
    e1_idx: int,
    e2_idx: int,
    mesh: Mesh,
    pgm: PolyhedralGaussMap,
) -> EEContactResult:
    """
    Check whether edges e1 and e2 make a feasible edge-edge contact.

    Edge-edge contact is geometrically valid when (Sec. 3.5, Eq. 15):

      Condition 1 — both closest points are *interior* to their edges:
                    t1 ∈ (0, 1)  AND  t2 ∈ (0, 1)
        Reason: if a closest point is at an endpoint vertex, the contact
        is assigned to that vertex's vertex-block (handled by VF detection),
        not to the edge-edge block.

      Condition 2 — contact direction in normal slab of e2:
                    direction = cp_e1 − cp_e2 ∈ normal_slab(e2)   (Eq. 15)
        Reason: this is the same slab test as Eq. 9, treating e2 as the
        "face" and e1 as the "query edge."  It ensures the contact is on the
        correct side of the edge ridge / valley.

    Parameters
    ----------
    e1_idx : int   "query" edge (plays the role of the query vertex in VF)
    e2_idx : int   "target" edge (plays the role of the triangle face in VF)
    mesh : Mesh
    pgm : PolyhedralGaussMap

    Returns
    -------
    EEContactResult
    """
    a1 = mesh.V[int(mesh.E[e1_idx][0])]
    b1 = mesh.V[int(mesh.E[e1_idx][1])]
    a2 = mesh.V[int(mesh.E[e2_idx][0])]
    b2 = mesh.V[int(mesh.E[e2_idx][1])]

    dist, cp1, t1, cp2, t2 = _edge_edge_closest(a1, b1, a2, b2)

    _EPS = 1e-8

    # Condition 1: both interior
    if t1 <= _EPS or t1 >= 1.0 - _EPS:
        return EEContactResult(False, dist, cp1, cp2, t1, t2)
    if t2 <= _EPS or t2 >= 1.0 - _EPS:
        return EEContactResult(False, dist, cp1, cp2, t1, t2)

    # Condition 2 (Eq. 15): contact direction in normal slab of e2
    direction = cp1 - cp2  # direction from e2 toward e1
    feasible  = pgm.is_in_edge_normal_slab(direction, e2_idx)

    return EEContactResult(
        feasible=feasible,
        distance=dist,
        contact_point_e1=cp1,
        contact_point_e2=cp2,
        t1=t1,
        t2=t2,
    )


# ======================================================================
# Internal helper
# ======================================================================

def _edge_edge_closest(
    p: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    s: np.ndarray,
) -> tuple[float, np.ndarray, float, np.ndarray, float]:
    """
    Closest points and parameters between segments pq and rs.

    Returns
    -------
    dist   : float       minimum distance between the two segments
    cp_pq  : ndarray     closest point on pq
    t      : float       parameter on pq  (cp_pq = p + t*(q-p))
    cp_rs  : ndarray     closest point on rs
    u      : float       parameter on rs  (cp_rs = r + u*(s-r))
    """
    d1    = q - p
    d2    = s - r
    r_vec = p - r
    a     = float(np.dot(d1, d1))
    e     = float(np.dot(d2, d2))
    f     = float(np.dot(d2, r_vec))

    _EPS = 1e-10

    if a <= _EPS and e <= _EPS:
        return float(np.linalg.norm(p - r)), p.copy(), 0.0, r.copy(), 0.0

    if a <= _EPS:
        t = 0.0
        u = float(np.clip(f / e, 0.0, 1.0))
    else:
        c = float(np.dot(d1, r_vec))
        if e <= _EPS:
            u = 0.0
            t = float(np.clip(-c / a, 0.0, 1.0))
        else:
            b     = float(np.dot(d1, d2))
            denom = a * e - b * b
            t     = float(np.clip((b * f - c * e) / denom, 0.0, 1.0)) if abs(denom) > _EPS else 0.0
            u     = (b * t + f) / e
            if u < 0.0:
                u = 0.0
                t = float(np.clip(-c / a, 0.0, 1.0))
            elif u > 1.0:
                u = 1.0
                t = float(np.clip((b - c) / a, 0.0, 1.0))

    cp_pq = p + t * d1
    cp_rs = r + u * d2
    return float(np.linalg.norm(cp_pq - cp_rs)), cp_pq, t, cp_rs, u
