"""
Mesh data structure.

Stores vertices, edges, and triangles together with precomputed
adjacency lists and face normals needed throughout the simulation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Mesh:
    """
    Triangle mesh with precomputed adjacency.

    Attributes
    ----------
    V : np.ndarray, shape (N, 3)
        Vertex positions.
    T : np.ndarray, shape (F, 3)
        Triangle indices into V (counter-clockwise winding).
    E : np.ndarray, shape (M, 2)
        Unique directed edge indices into V (each undirected edge stored once).
    face_normals : np.ndarray, shape (F, 3)
        Unit outward normal for each triangle.
    T_v : list[list[int]], length N
        T_v[i] = list of triangle indices that contain vertex i.
    E_v : list[list[int]], length N
        E_v[i] = list of edge indices that contain vertex i.
    E_t : list[list[int]], length F
        E_t[f] = list of edge indices that bound triangle f (up to 3).
    """

    V: np.ndarray                         # (N, 3)
    T: np.ndarray                         # (F, 3)
    E: np.ndarray = field(default=None)   # (M, 2) — built in __post_init__
    face_normals: np.ndarray = field(default=None)  # (F, 3)
    T_v: list[list[int]] = field(default=None)
    E_v: list[list[int]] = field(default=None)
    E_t: list[list[int]] = field(default=None)

    def __post_init__(self) -> None:
        self.V = np.asarray(self.V, dtype=np.float64)
        self.T = np.asarray(self.T, dtype=np.int32)

        if self.E is None:
            self.E = self._build_edges()
        else:
            self.E = np.asarray(self.E, dtype=np.int32)

        if self.face_normals is None:
            self.face_normals = self._compute_face_normals()

        if self.T_v is None:
            self.T_v = self._build_T_v()

        if self.E_v is None:
            self.E_v = self._build_E_v()

        if self.E_t is None:
            self.E_t = self._build_E_t()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_vertices(self) -> int:
        return len(self.V)

    @property
    def num_triangles(self) -> int:
        return len(self.T)

    @property
    def num_edges(self) -> int:
        return len(self.E)

    # ------------------------------------------------------------------
    # Builder helpers  (TODO: implement each one)
    # ------------------------------------------------------------------

    def _build_edges(self) -> np.ndarray:
        """
        Extract the set of unique undirected edges from self.T.

        For each triangle (a, b, c) the three edges are (a,b), (b,c), (c,a).
        Store each undirected edge with the smaller index first so duplicates
        collapse cleanly.

        Returns
        -------
        np.ndarray, shape (M, 2)
        """
        edge_set = set()
        for tri in self.T:
            for i in range(3):
                a, b = int(tri[i]), int(tri[(i + 1) % 3])
                edge_set.add((min(a, b), max(a, b)))
        return np.array(sorted(edge_set), dtype=np.int32)

    def _compute_face_normals(self) -> np.ndarray:
        """
        Compute the unit outward normal for every triangle.

        For triangle (a, b, c) with CCW winding the normal is:
            n = (b - a) × (c - a),  then normalised.

        Returns
        -------
        np.ndarray, shape (F, 3)
        """
        v0 = self.V[self.T[:, 0]]
        v1 = self.V[self.T[:, 1]]
        v2 = self.V[self.T[:, 2]]
        n  = np.cross(v1 - v0, v2 - v0)
        lengths = np.linalg.norm(n, axis=1, keepdims=True)
        return n / np.where(lengths > 0, lengths, 1.0)

    def _build_T_v(self) -> list[list[int]]:
        """
        Build T_v[i] = sorted list of triangle indices that contain vertex i.

        Returns
        -------
        list of length N
        """
        T_v = [[] for _ in range(self.num_vertices)]
        for ti, tri in enumerate(self.T):
            for vi in tri:
                T_v[int(vi)].append(ti)
        return T_v

    def _build_E_v(self) -> list[list[int]]:
        """
        Build E_v[i] = sorted list of edge indices that contain vertex i.

        Returns
        -------
        list of length N
        """
        E_v = [[] for _ in range(self.num_vertices)]
        for ei, (a, b) in enumerate(self.E):
            E_v[int(a)].append(ei)
            E_v[int(b)].append(ei)
        return E_v

    def _build_E_t(self) -> list[list[int]]:
        """
        Build E_t[f] = list of edge indices that bound triangle f.

        Each triangle has exactly 3 boundary edges.

        Returns
        -------
        list of length F
        """
        edge_lookup = {
            (int(a), int(b)): ei for ei, (a, b) in enumerate(self.E)
        }
        E_t = []
        for tri in self.T:
            tri_edges = []
            for i in range(3):
                a, b = int(tri[i]), int(tri[(i + 1) % 3])
                key = (min(a, b), max(a, b))
                tri_edges.append(edge_lookup[key])
            E_t.append(tri_edges)
        return E_t

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_arrays(cls, vertices: np.ndarray, triangles: np.ndarray) -> "Mesh":
        """Convenience constructor — builds all adjacency automatically."""
        return cls(V=vertices, T=triangles)
