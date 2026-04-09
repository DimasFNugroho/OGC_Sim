"""
Polyhedral Gauss Map (PGM).

Maps points on a polyhedral surface to their associated set of normals.
On a smooth surface every point has exactly one normal; on a polyhedral
surface, vertices and edges can have a whole *cone* or *slab* of normals.

This information drives the feasibility-region checks in
contact/offset_geometry.py: a query point x belongs to the offset block
of a feature `a` only if (x - closest_point) is in the normal set of `a`.

Paper reference: Sec. 3.1 and Echeverria [2007].

Classification (Sec. 3.1, Fig. 5):
  CONVEX  vertex — all incident face normals point into the same open
                   hemisphere; normal set is a spherical convex polygon.
  CONCAVE vertex — incident normals span more than a hemisphere.
  MIXED   vertex — some incident faces are convex, some concave.

For edges, the normal set is the great-circle arc between the two adjacent
face normals (the "normal slab").
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from .mesh import Mesh


class VertexType(Enum):
    CONVEX = auto()
    CONCAVE = auto()
    MIXED = auto()


@dataclass
class PolyhedralGaussMap:
    """
    Precomputed Gauss Map data for a mesh.

    Attributes
    ----------
    vertex_types : list[VertexType], length N
        Classification of each vertex.
    vertex_normal_cones : list[np.ndarray], length N
        vertex_normal_cones[i] is an array of shape (K_i, 3) containing the
        unit normals of the K_i faces incident to vertex i, ordered CCW
        around the vertex.  Together they span the normal cone.
    edge_normal_slabs : list[tuple[np.ndarray, np.ndarray]], length M
        edge_normal_slabs[j] = (n0, n1) where n0 and n1 are the unit normals
        of the (at most two) faces adjacent to edge j.  The normal slab is
        the set of all convex combinations of n0 and n1.
    """

    mesh: Mesh
    vertex_types: list[VertexType] = field(default=None)
    vertex_normal_cones: list[np.ndarray] = field(default=None)
    edge_normal_slabs: list[tuple[np.ndarray, np.ndarray]] = field(default=None)

    def __post_init__(self) -> None:
        self.vertex_types = self._classify_vertices()
        self.vertex_normal_cones = self._compute_vertex_normal_cones()
        self.edge_normal_slabs = self._compute_edge_normal_slabs()

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify_vertices(self) -> list[VertexType]:
        """
        Classify every vertex as CONVEX, CONCAVE, or MIXED.

        For each edge incident to vertex v, find the two adjacent faces and
        check whether the "extra" vertex of face 1 lies on the negative-normal
        side of face 0.  This is the standard ridge/valley convexity test:

          sign = dot(n0, V[other_in_face1] - edge_midpoint)
          sign < 0  →  convex (ridge)   — faces splay outward
          sign > 0  →  concave (valley) — faces fold inward
          sign ≈ 0  →  flat (coplanar)  — treated as convex

        A vertex is CONVEX if all its incident edges are convex (or flat),
        CONCAVE if all are concave, and MIXED otherwise.

        Boundary edges (one adjacent face) are skipped.

        Returns
        -------
        list[VertexType], length N
        """
        FLAT_TOL = 1e-10

        types: list[VertexType] = []
        for vi in range(self.mesh.num_vertices):
            edge_signs: list[bool] = []  # True = convex

            for ei in self.mesh.E_v[vi]:
                a, b = int(self.mesh.E[ei][0]), int(self.mesh.E[ei][1])
                adj = list(set(self.mesh.T_v[a]) & set(self.mesh.T_v[b]))
                if len(adj) < 2:
                    continue  # boundary edge — skip

                fi0, fi1 = adj[0], adj[1]
                n0 = self.mesh.face_normals[fi0]

                # Extra vertex in face 1 that is not on the shared edge
                other = next(
                    int(v) for v in self.mesh.T[fi1]
                    if int(v) != a and int(v) != b
                )
                mid = (self.mesh.V[a] + self.mesh.V[b]) * 0.5
                sign = float(np.dot(n0, self.mesh.V[other] - mid))

                # Flat edges (|sign| ≈ 0) count as convex
                edge_signs.append(sign <= FLAT_TOL)

            if not edge_signs:
                types.append(VertexType.CONVEX)
            elif all(edge_signs):
                types.append(VertexType.CONVEX)
            elif not any(edge_signs):
                types.append(VertexType.CONCAVE)
            else:
                types.append(VertexType.MIXED)

        return types

    # ------------------------------------------------------------------
    # Normal cones and slabs
    # ------------------------------------------------------------------

    def _compute_vertex_normal_cones(self) -> list[np.ndarray]:
        """
        For each vertex, collect the unit normals of its incident faces,
        ordered CCW around the vertex.

        Ordering procedure:
        1. Compute the average normal at the vertex.
        2. Project each incident triangle centroid relative to the vertex
           onto the tangent plane (perpendicular to the average normal).
        3. Sort by the angle in the tangent plane using a reference direction.

        Returns
        -------
        list of np.ndarray, each shape (K_i, 3)
        """
        cones: list[np.ndarray] = []

        for vi in range(self.mesh.num_vertices):
            tris = self.mesh.T_v[vi]

            if not tris:
                cones.append(np.empty((0, 3)))
                continue

            normals = self.mesh.face_normals[np.array(tris)]  # (K, 3)

            if len(tris) == 1:
                cones.append(normals.copy())
                continue

            # Average normal as tangent-plane reference axis
            avg_n = normals.mean(axis=0)
            avg_n_len = np.linalg.norm(avg_n)
            if avg_n_len < 1e-12:
                cones.append(normals.copy())
                continue
            avg_n = avg_n / avg_n_len

            # Centroid vectors projected onto tangent plane
            v_pos = self.mesh.V[vi]
            centroids = np.array(
                [self.mesh.V[self.mesh.T[ti]].mean(axis=0) for ti in tris]
            )
            vecs = centroids - v_pos                              # (K, 3)
            vecs -= np.dot(vecs, avg_n)[:, None] * avg_n         # project out normal component

            # Reference direction in tangent plane
            ref = vecs[0]
            ref_len = np.linalg.norm(ref)
            if ref_len < 1e-12:
                cones.append(normals.copy())
                continue
            ref = ref / ref_len
            perp = np.cross(avg_n, ref)                           # 90° CCW in tangent plane

            angles = np.arctan2(vecs @ perp, vecs @ ref)         # (K,)
            order = np.argsort(angles)
            cones.append(normals[order].copy())

        return cones

    def _compute_edge_normal_slabs(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        For each edge, return the pair of adjacent face normals.

        The normal slab of an edge is the set of unit vectors that can be
        written as normalise(α·n0 + (1-α)·n1) for α ∈ [0, 1].

        Boundary edges (only one adjacent face) return (n0, n0).

        Returns
        -------
        list of (n0, n1) tuples, length M
        """
        slabs: list[tuple[np.ndarray, np.ndarray]] = []

        for ei in range(self.mesh.num_edges):
            a, b = int(self.mesh.E[ei][0]), int(self.mesh.E[ei][1])
            adj = list(set(self.mesh.T_v[a]) & set(self.mesh.T_v[b]))

            if len(adj) == 0:
                n = np.array([0.0, 0.0, 1.0])
                slabs.append((n, n))
            elif len(adj) == 1:
                n = self.mesh.face_normals[adj[0]].copy()
                slabs.append((n, n))
            else:
                n0 = self.mesh.face_normals[adj[0]].copy()
                n1 = self.mesh.face_normals[adj[1]].copy()
                slabs.append((n0, n1))

        return slabs

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_in_vertex_normal_cone(
        self, direction: np.ndarray, vertex_idx: int
    ) -> bool:
        """
        Return True if `direction` (need not be unit) lies inside the normal
        cone of vertex `vertex_idx`.

        For a convex cone whose boundary generators {n_0, …, n_{K-1}} are
        ordered CCW, a direction d is *inside* the cone iff:

            dot(d, n_i × n_{i+1}) >= 0   for all i (indices mod K)

        This is the "half-space per bounding great-circle arc" test.

        Paper use: Eq. 8 — feasibility check for vertex offset blocks.

        Parameters
        ----------
        direction : np.ndarray, shape (3,)
            Typically x_v - closest_point (un-normalised is fine).
        vertex_idx : int

        Returns
        -------
        bool
        """
        cone = self.vertex_normal_cones[vertex_idx]
        K = len(cone)

        if K == 0:
            return False

        d = direction / (np.linalg.norm(direction) + 1e-12)

        # Necessary condition: d must lie on the positive side of every generator.
        # This also handles degenerate cones where all generators are identical
        # (flat mesh) — the cone degenerates to the half-space above the tangent plane.
        for n in cone:
            if float(np.dot(d, n)) < -1e-10:
                return False

        if K == 1:
            return True

        # For each consecutive pair of generators (ordered CCW), d must be
        # on the positive side of their shared great-circle boundary:
        #   dot(d, n_i × n_{i+1}) >= 0
        # Skip degenerate boundaries where n_i ≈ n_{i+1} (cross product ≈ 0).
        for i in range(K):
            n0 = cone[i]
            n1 = cone[(i + 1) % K]
            boundary_axis = np.cross(n0, n1)
            if np.linalg.norm(boundary_axis) < 1e-10:
                continue  # degenerate boundary — dot-product check above suffices
            if float(np.dot(d, boundary_axis)) < -1e-10:
                return False

        return True

    def is_in_edge_normal_slab(
        self, direction: np.ndarray, edge_idx: int
    ) -> bool:
        """
        Return True if `direction` lies inside the normal slab of edge
        `edge_idx`.

        The slab is the set of directions that make an angle ≤ 90° with
        *both* bounding face normals:

            dot(direction, n0) >= 0   AND   dot(direction, n1) >= 0

        Paper use: Eq. 9 — feasibility check for edge offset blocks.

        Parameters
        ----------
        direction : np.ndarray, shape (3,)
        edge_idx : int

        Returns
        -------
        bool
        """
        n0, n1 = self.edge_normal_slabs[edge_idx]
        return (
            float(np.dot(direction, n0)) >= 0.0
            and float(np.dot(direction, n1)) >= 0.0
        )
