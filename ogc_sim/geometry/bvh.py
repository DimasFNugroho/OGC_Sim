"""
Bounding Volume Hierarchy for broadphase collision queries.

Wraps scipy's cKDTree for point/centroid queries and provides a simple
AABB tree for triangle and edge sphere queries.  At prototype scale a
linear scan against expanded AABBs is acceptable; replace the internals
with a proper BVH tree if performance becomes a bottleneck.

Paper reference: Sec. 4 — "building a BVH of all those blocks … only
requires building a BVH for the faces with the highest dimensionality."
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .mesh import Mesh


class BVH:
    """
    Broadphase structure built over a mesh's triangles and edges.

    After construction, use :meth:`sphere_query_triangles` and
    :meth:`sphere_query_edges` to find candidate primitives within a
    given radius of a query point.

    Parameters
    ----------
    mesh : Mesh
        The mesh to accelerate queries over.
    """

    def __init__(self, mesh: Mesh) -> None:
        self.mesh = mesh
        self._tri_centroids: np.ndarray | None = None   # (F, 3)
        self._tri_half_diags: np.ndarray | None = None  # (F,) max vertex dist to centroid
        self._edge_centroids: np.ndarray | None = None  # (M, 3)
        self._edge_half_lens: np.ndarray | None = None  # (M,) half edge length
        self._tri_tree: cKDTree | None = None
        self._edge_tree: cKDTree | None = None

        self._build()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self) -> None:
        """
        Precompute centroids, bounding radii, and KD-trees for triangles
        and edges.

        For triangles:
          centroid  = mean of the three vertex positions
          half_diag = max distance from centroid to any of its 3 vertices
                      (this is the circumradius upper bound used to expand
                      the query radius conservatively)

        For edges:
          centroid  = midpoint of the two endpoints
          half_len  = half the edge length
                      (Algorithm 2 uses centre x_m and radius r_q + l/2)
        """
        V, T, E = self.mesh.V, self.mesh.T, self.mesh.E

        # --- triangles ---
        tri_verts = V[T]                                        # (F, 3, 3)
        self._tri_centroids  = tri_verts.mean(axis=1)          # (F, 3)
        dists = np.linalg.norm(
            tri_verts - self._tri_centroids[:, None, :], axis=2
        )                                                       # (F, 3)
        self._tri_half_diags = dists.max(axis=1)               # (F,)
        self._tri_tree = cKDTree(self._tri_centroids)

        # --- edges ---
        self._edge_centroids = (V[E[:, 0]] + V[E[:, 1]]) / 2  # (M, 3)
        self._edge_half_lens = (
            np.linalg.norm(V[E[:, 1]] - V[E[:, 0]], axis=1) / 2
        )                                                       # (M,)
        self._edge_tree = cKDTree(self._edge_centroids)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def sphere_query_triangles(
        self, center: np.ndarray, radius: float
    ) -> list[int]:
        """
        Return indices of triangles whose bounding sphere overlaps the
        query sphere (center, radius).

        A triangle's bounding sphere has centre = centroid and radius =
        half_diag.  Two spheres overlap when:
            dist(centres) < radius + half_diag

        So we query the KD-tree with radius = radius + max(half_diag)
        conservatively, then filter by per-triangle half_diag.

        Parameters
        ----------
        center : np.ndarray, shape (3,)
        radius : float
            Query radius r_q (Algorithm 1, line 2).

        Returns
        -------
        list[int]
            Triangle indices that are candidates for contact with `center`.
        """
        # Conservatively expand query radius by the largest half-diagonal,
        # then filter each candidate by its own half-diagonal.
        pad = float(self._tri_half_diags.max()) if len(self._tri_half_diags) else 0.0
        candidates = self._tri_tree.query_ball_point(center, radius + pad)
        return [
            i for i in candidates
            if np.linalg.norm(center - self._tri_centroids[i])
               < radius + self._tri_half_diags[i]
        ]

    def sphere_query_edges(
        self, center: np.ndarray, radius: float
    ) -> list[int]:
        """
        Return indices of edges whose bounding sphere overlaps the query
        sphere (center, radius).

        Algorithm 2 uses a sphere centred at the edge midpoint x_m with
        query radius r_q + l/2 (where l is the edge length).  Here we are
        doing the reverse: given a query point, find all edges within reach.

        An edge's bounding sphere has centre = midpoint and radius = half_len.
        Overlap condition: dist(center, midpoint) < radius + half_len.

        Parameters
        ----------
        center : np.ndarray, shape (3,)
        radius : float

        Returns
        -------
        list[int]
            Edge indices that are candidates.
        """
        pad = float(self._edge_half_lens.max()) if len(self._edge_half_lens) else 0.0
        candidates = self._edge_tree.query_ball_point(center, radius + pad)
        return [
            i for i in candidates
            if np.linalg.norm(center - self._edge_centroids[i])
               < radius + self._edge_half_lens[i]
        ]

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def refit(self) -> None:
        """
        Recompute centroids and rebuild KD-trees after vertex positions
        have changed (called once per contact-detection phase in Algorithm 3).
        """
        self._build()
