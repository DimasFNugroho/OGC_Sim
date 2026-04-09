"""
Tests for PolyhedralGaussMap.

Covers:
  - Vertex classification for flat, convex, and concave meshes
  - Normal cone membership (vertex)
  - Normal slab membership (edge)
  - Boundary edge handling (slab degenerates to (n, n))
"""

import numpy as np
import pytest

from ogc_sim.geometry.mesh import Mesh
from ogc_sim.geometry.gauss_map import PolyhedralGaussMap, VertexType


# ======================================================================
# Fixtures
# ======================================================================

def flat_mesh() -> tuple[Mesh, PolyhedralGaussMap]:
    """3×3 grid of vertices, 8 triangles, all in the Z=0 plane."""
    V = np.array([
        [0., 0., 0.], [1., 0., 0.], [2., 0., 0.],
        [0., 1., 0.], [1., 1., 0.], [2., 1., 0.],
        [0., 2., 0.], [1., 2., 0.], [2., 2., 0.],
    ])
    T = np.array([
        [0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4],
        [3, 4, 7], [3, 7, 6], [4, 5, 8], [4, 8, 7],
    ])
    mesh = Mesh.from_arrays(V, T)
    return mesh, PolyhedralGaussMap(mesh)


def box_corner_mesh() -> tuple[Mesh, PolyhedralGaussMap]:
    """
    Three mutually perpendicular triangles meeting at the origin,
    forming the *outer* corner of a box occupying [0,1]^3.

    V[0] = (0,0,0) is the shared corner.  The three faces are the bottom,
    left, and front faces of the box, with outward normals:
        bottom → (0,0,-1)   left → (-1,0,0)   front → (0,-1,0)

    All three normals point *away* from the box, so V[0] is CONVEX.
    The inward-pointing diagonal (-1,-1,-1)/sqrt(3) is inside the cone.
    """
    V = np.array([
        [0., 0., 0.],  # 0 shared corner
        [1., 0., 0.],  # 1
        [0., 1., 0.],  # 2
        [0., 0., 1.],  # 3
    ])
    # CCW winding so outward normals point away from the box interior (+X,+Y,+Z).
    # Bottom face (z=0): CCW from -Z → [0,2,1] → normal (0,0,-1)
    # Left face   (x=0): CCW from -X → [0,3,2] → normal (-1,0,0)
    # Front face  (y=0): CCW from -Y → [0,1,3] → normal (0,-1,0)
    T = np.array([
        [0, 2, 1],  # normal (0,0,-1)
        [0, 3, 2],  # normal (-1,0,0)
        [0, 1, 3],  # normal (0,-1,0)
    ])
    mesh = Mesh.from_arrays(V, T)
    return mesh, PolyhedralGaussMap(mesh)


# ======================================================================
# Vertex classification
# ======================================================================

class TestVertexClassification:

    def test_flat_mesh_all_convex(self):
        """All vertices of a flat grid should be classified as CONVEX."""
        _, pgm = flat_mesh()
        for vi, vt in enumerate(pgm.vertex_types):
            assert vt == VertexType.CONVEX, f"V[{vi}] expected CONVEX, got {vt}"

    def test_type_count_matches_vertices(self):
        mesh, pgm = flat_mesh()
        assert len(pgm.vertex_types) == mesh.num_vertices


# ======================================================================
# Edge normal slabs
# ======================================================================

class TestEdgeNormalSlabs:

    def test_slab_count_matches_edges(self):
        mesh, pgm = flat_mesh()
        assert len(pgm.edge_normal_slabs) == mesh.num_edges

    def test_slab_normals_are_unit(self):
        _, pgm = flat_mesh()
        for n0, n1 in pgm.edge_normal_slabs:
            assert abs(np.linalg.norm(n0) - 1.0) < 1e-10, f"n0 not unit: {n0}"
            assert abs(np.linalg.norm(n1) - 1.0) < 1e-10, f"n1 not unit: {n1}"

    def test_boundary_edge_slab_is_degenerate(self):
        """A boundary edge should return (n, n) — both normals identical."""
        mesh, pgm = flat_mesh()
        boundary_ei = None
        for ei, (a, b) in enumerate(mesh.E):
            adj = set(mesh.T_v[int(a)]) & set(mesh.T_v[int(b)])
            if len(adj) == 1:
                boundary_ei = ei
                break
        assert boundary_ei is not None, "No boundary edge found"
        n0, n1 = pgm.edge_normal_slabs[boundary_ei]
        np.testing.assert_allclose(n0, n1, atol=1e-12)

    def test_interior_edge_slab_flat_mesh_normals_equal(self):
        """For a flat mesh every edge's two adjacent normals are identical [0,0,1]."""
        _, pgm = flat_mesh()
        expected = np.array([0., 0., 1.])
        for n0, n1 in pgm.edge_normal_slabs:
            np.testing.assert_allclose(n0, expected, atol=1e-10)
            np.testing.assert_allclose(n1, expected, atol=1e-10)


# ======================================================================
# Vertex normal cones
# ======================================================================

class TestVertexNormalCones:

    def test_cone_count_matches_vertices(self):
        mesh, pgm = flat_mesh()
        assert len(pgm.vertex_normal_cones) == mesh.num_vertices

    def test_cone_generator_count_matches_incident_faces(self):
        mesh, pgm = flat_mesh()
        for vi in range(mesh.num_vertices):
            expected = len(mesh.T_v[vi])
            got = len(pgm.vertex_normal_cones[vi])
            assert got == expected, f"V[{vi}]: expected {expected} generators, got {got}"

    def test_cone_generators_are_unit(self):
        _, pgm = flat_mesh()
        for vi, cone in enumerate(pgm.vertex_normal_cones):
            for n in cone:
                assert abs(np.linalg.norm(n) - 1.0) < 1e-10, \
                    f"V[{vi}] generator not unit: {n}"


# ======================================================================
# is_in_vertex_normal_cone
# ======================================================================

class TestVertexConeQuery:

    def test_outward_normal_is_inside_flat_cone(self):
        """[0,0,1] is the outward direction — should be in every vertex's cone."""
        _, pgm = flat_mesh()
        d = np.array([0., 0., 1.])
        for vi in range(9):
            assert pgm.is_in_vertex_normal_cone(d, vi), \
                f"V[{vi}]: outward normal not in cone"

    def test_inward_normal_is_outside_flat_cone(self):
        """[0,0,-1] points inward — should be outside every vertex's cone."""
        _, pgm = flat_mesh()
        d = np.array([0., 0., -1.])
        for vi in range(9):
            assert not pgm.is_in_vertex_normal_cone(d, vi), \
                f"V[{vi}]: inward normal should not be in cone"

    def test_tangential_direction_on_boundary(self):
        """A tangential direction (z=0) is on the boundary of a flat cone."""
        _, pgm = flat_mesh()
        # Strictly tangential: dot with any [0,0,1] generator = 0 → boundary (True)
        d = np.array([1., 0., 0.])
        assert pgm.is_in_vertex_normal_cone(d, 4)  # centre vertex

    def test_unnormalised_direction_accepted(self):
        """The query should work with un-normalised directions."""
        _, pgm = flat_mesh()
        d = np.array([0., 0., 5.0])  # same as +Z but scaled
        assert pgm.is_in_vertex_normal_cone(d, 4)

    def test_box_corner_outward_directions(self):
        """For the convex box corner vertex, the diagonal direction pointing
        away from the box interior should be inside the normal cone."""
        _, pgm = box_corner_mesh()
        # V[0] generators are (0,0,-1), (-1,0,0), (0,-1,0).
        # The outward diagonal from the corner = (-1,-1,-1)/sqrt(3).
        outward = np.array([-1., -1., -1.]) / np.sqrt(3)
        assert pgm.is_in_vertex_normal_cone(outward, 0)
        # Inward diagonal (+1,+1,+1)/sqrt(3) points into the box → not in cone.
        inward = np.array([1., 1., 1.]) / np.sqrt(3)
        assert not pgm.is_in_vertex_normal_cone(inward, 0)


# ======================================================================
# is_in_edge_normal_slab
# ======================================================================

class TestEdgeSlabQuery:

    def test_outward_in_slab(self):
        """[0,0,1] should be inside every edge slab for the flat mesh."""
        _, pgm = flat_mesh()
        d = np.array([0., 0., 1.])
        for ei in range(len(pgm.edge_normal_slabs)):
            assert pgm.is_in_edge_normal_slab(d, ei), \
                f"E[{ei}]: outward direction not in slab"

    def test_inward_not_in_slab(self):
        """[0,0,-1] should be outside every edge slab for the flat mesh."""
        _, pgm = flat_mesh()
        d = np.array([0., 0., -1.])
        for ei in range(len(pgm.edge_normal_slabs)):
            assert not pgm.is_in_edge_normal_slab(d, ei), \
                f"E[{ei}]: inward direction should not be in slab"

    def test_tangential_on_boundary_of_flat_slab(self):
        """A tangential direction has dot=0 with [0,0,1] → on the boundary (True)."""
        _, pgm = flat_mesh()
        d = np.array([1., 0., 0.])
        # dot(d, n0) = 0 >= 0 is True for all flat-mesh edges
        for ei in range(len(pgm.edge_normal_slabs)):
            assert pgm.is_in_edge_normal_slab(d, ei)

    def test_unnormalised_direction_accepted(self):
        _, pgm = flat_mesh()
        d = np.array([0., 0., 100.])
        assert pgm.is_in_edge_normal_slab(d, 0)
