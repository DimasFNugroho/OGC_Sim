"""
Unit tests for geometry/distance.py.

Each test uses a hand-constructed configuration with a known analytic answer.
Implement the functions in distance.py until all tests pass.

Run with:
    pytest tests/test_distance.py -v
"""

import numpy as np
import pytest
from ogc_sim.geometry.distance import (
    ClosestFeature,
    point_triangle_distance,
    edge_edge_distance,
)

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def assert_close(actual, expected, *, atol=1e-10, label=""):
    assert abs(actual - expected) < atol, (
        f"{label}: expected {expected}, got {actual}"
    )

def assert_vec_close(actual, expected, *, atol=1e-10, label=""):
    np.testing.assert_allclose(actual, expected, atol=atol, err_msg=label)


# -----------------------------------------------------------------------
# Triangle setup (shared across point-triangle tests)
#
#   Unit triangle in the XY plane:
#     a = (0, 0, 0),  b = (1, 0, 0),  c = (0, 1, 0)
#   Face normal points in +Z.
# -----------------------------------------------------------------------

A = np.array([0.0, 0.0, 0.0])
B = np.array([1.0, 0.0, 0.0])
C = np.array([0.0, 1.0, 0.0])


class TestPointTriangleDistance:

    def test_above_face_interior(self):
        """Point directly above the centroid — closest feature is face interior."""
        centroid = (A + B + C) / 3.0
        p = centroid + np.array([0.0, 0.0, 2.0])

        dist, closest, feature, idx = point_triangle_distance(p, A, B, C)

        assert_close(dist, 2.0, label="dist")
        assert_vec_close(closest, centroid, label="closest_pt")
        assert feature == ClosestFeature.FACE_INTERIOR
        assert idx == -1

    def test_above_vertex_a(self):
        """Point directly above vertex A — closest feature is vertex 0."""
        p = A + np.array([0.0, 0.0, 3.0])

        dist, closest, feature, idx = point_triangle_distance(p, A, B, C)

        assert_close(dist, 3.0, label="dist")
        assert_vec_close(closest, A, label="closest_pt")
        assert feature == ClosestFeature.VERTEX
        assert idx == 0

    def test_above_vertex_b(self):
        """Point directly above vertex B — closest feature is vertex 1."""
        p = B + np.array([0.0, 0.0, 1.5])

        dist, closest, feature, idx = point_triangle_distance(p, A, B, C)

        assert_close(dist, 1.5, label="dist")
        assert_vec_close(closest, B, label="closest_pt")
        assert feature == ClosestFeature.VERTEX
        assert idx == 1

    def test_above_vertex_c(self):
        """Point directly above vertex C — closest feature is vertex 2."""
        p = C + np.array([0.0, 0.0, 0.5])

        dist, closest, feature, idx = point_triangle_distance(p, A, B, C)

        assert_close(dist, 0.5, label="dist")
        assert_vec_close(closest, C, label="closest_pt")
        assert feature == ClosestFeature.VERTEX
        assert idx == 2

    def test_beside_edge_ab(self):
        """
        Point beside the midpoint of edge AB, offset in -Y (outside the triangle).
        Closest feature should be edge 0 (AB).
        """
        mid_ab = (A + B) / 2.0
        p = mid_ab + np.array([0.0, -2.0, 0.0])

        dist, closest, feature, idx = point_triangle_distance(p, A, B, C)

        assert_close(dist, 2.0, label="dist")
        assert_vec_close(closest, mid_ab, label="closest_pt")
        assert feature == ClosestFeature.EDGE
        assert idx == 0  # edge AB

    def test_beside_edge_bc(self):
        """
        Point beside the midpoint of edge BC, offset outward.
        Edge BC goes from B=(1,0,0) to C=(0,1,0); its outward direction
        is normalised (1, 1, 0) / sqrt(2).
        """
        mid_bc = (B + C) / 2.0
        outward = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
        offset = 3.0
        p = mid_bc + offset * outward

        dist, closest, feature, idx = point_triangle_distance(p, A, B, C)

        assert_close(dist, offset, label="dist", atol=1e-9)
        assert_vec_close(closest, mid_bc, label="closest_pt", atol=1e-9)
        assert feature == ClosestFeature.EDGE
        assert idx == 1  # edge BC

    def test_point_on_triangle(self):
        """Point exactly on the triangle surface — distance should be 0."""
        p = (A + B + C) / 3.0  # centroid, on the face

        dist, closest, feature, idx = point_triangle_distance(p, A, B, C)

        assert_close(dist, 0.0, label="dist")
        assert_vec_close(closest, p, label="closest_pt")
        assert feature == ClosestFeature.FACE_INTERIOR

    def test_below_face(self):
        """Point below the face (negative Z) — distance is still positive."""
        centroid = (A + B + C) / 3.0
        p = centroid + np.array([0.0, 0.0, -4.0])

        dist, closest, feature, idx = point_triangle_distance(p, A, B, C)

        assert_close(dist, 4.0, label="dist")
        assert_vec_close(closest, centroid, label="closest_pt")
        assert feature == ClosestFeature.FACE_INTERIOR


# -----------------------------------------------------------------------
# Edge-edge distance tests
# -----------------------------------------------------------------------

class TestEdgeEdgeDistance:

    def test_parallel_edges_horizontal(self):
        """
        Two horizontal parallel edges separated by 2 units in Y.
          e1: (0,0,0) -> (1,0,0)
          e2: (0,2,0) -> (1,2,0)
        Closest pair: any corresponding points; distance = 2.
        """
        p, q = np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
        r, s = np.array([0.0, 2.0, 0.0]), np.array([1.0, 2.0, 0.0])

        dist, closest = edge_edge_distance(p, q, r, s)

        assert_close(dist, 2.0, label="dist")
        # closest point on pq should have Y=0 and be at the same X as
        # the point on rs (exact X depends on parameterisation — just
        # check Y and Z of the returned point)
        assert_close(closest[1], 0.0, label="closest_pt Y")
        assert_close(closest[2], 0.0, label="closest_pt Z")

    def test_perpendicular_skew_edges(self):
        """
        Two skew edges forming a cross, separated by 1 unit in Z.
          e1: (-1,0,0) -> (1,0,0)   (along X, at Z=0)
          e2: (0,-1,1) -> (0,1,1)   (along Y, at Z=1)
        Closest points: (0,0,0) and (0,0,1) — distance = 1.
        """
        p, q = np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
        r, s = np.array([0.0, -1.0, 1.0]), np.array([0.0, 1.0, 1.0])

        dist, closest = edge_edge_distance(p, q, r, s)

        assert_close(dist, 1.0, label="dist")
        assert_vec_close(closest, np.array([0.0, 0.0, 0.0]), label="closest_pt")

    def test_touching_edges(self):
        """Edges that share an endpoint — distance should be 0."""
        p, q = np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
        r, s = np.array([1.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.0])

        dist, closest = edge_edge_distance(p, q, r, s)

        assert_close(dist, 0.0, label="dist")

    def test_endpoint_to_interior(self):
        """
        e1 endpoint is closest to an interior point of e2.
          e1: (0,0,2) -> (0,0,3)   (short vertical segment)
          e2: (-1,0,0) -> (1,0,0)  (long horizontal segment)
        Closest: e1 endpoint (0,0,2) to e2 point (0,0,0) — distance = 2.
        """
        p, q = np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, 3.0])
        r, s = np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])

        dist, closest = edge_edge_distance(p, q, r, s)

        assert_close(dist, 2.0, label="dist")
        assert_vec_close(closest, np.array([0.0, 0.0, 2.0]), label="closest_pt")
