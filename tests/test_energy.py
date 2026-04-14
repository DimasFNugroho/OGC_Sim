"""
Unit tests for contact/energy.py and contact/bounds.py.

Validation targets (from CLAUDE.md / PLAN.md M3):
  - g is C² at τ = r/2  (value, first, second derivatives match)
  - g is C¹ at r         (value = 0, first derivative = 0)
  - Finite-difference gradient check for contact_gradient_v_vf
  - Finite-difference Hessian  check for contact_hessian_v_vf
  - Finite-difference gradient check for contact_gradient_v_ee
  - compute_conservative_bounds returns correct b_v (Eq. 21)
  - truncate_displacements clamps correctly and counts exceeded vertices

Run with:
    pytest tests/test_energy.py -v
"""

import numpy as np
import pytest

from ogc_sim.contact.energy import (
    _stitch_params,
    activation_g,
    activation_dg_dd,
    activation_d2g_dd2,
    contact_energy_vf,
    contact_gradient_v_vf,
    contact_hessian_v_vf,
    contact_energy_ee,
    contact_gradient_v_ee,
)
from ogc_sim.contact.bounds import (
    compute_conservative_bounds,
    truncate_displacements,
    apply_initial_guess_truncation,
)
from ogc_sim.geometry.mesh     import Mesh
from ogc_sim.contact.detection import ContactSets


# ======================================================================
# Helpers
# ======================================================================

R   = 0.6   # default contact radius for all tests
K_C = 1e3   # default stiffness


def fd_gradient(f, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Central finite-difference gradient of scalar f(x) w.r.t. (N,) array x."""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_p = x.copy(); x_p[i] += eps
        x_m = x.copy(); x_m[i] -= eps
        grad[i] = (f(x_p) - f(x_m)) / (2 * eps)
    return grad


def fd_hessian(grad_fn, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Central finite-difference Hessian from the gradient function."""
    n   = len(x)
    H   = np.zeros((n, n))
    for i in range(n):
        x_p = x.copy(); x_p[i] += eps
        x_m = x.copy(); x_m[i] -= eps
        H[:, i] = (grad_fn(x_p) - grad_fn(x_m)) / (2 * eps)
    return H


# ======================================================================
# activation_g — scalar function tests
# ======================================================================

class TestActivationG:

    def test_zero_outside_radius(self):
        assert activation_g(R,       R, K_C) == 0.0
        assert activation_g(R + 0.1, R, K_C) == 0.0
        assert activation_g(R * 2,   R, K_C) == 0.0

    def test_quadratic_stage(self):
        """At d = τ, both stages should agree."""
        tau, _, _ = _stitch_params(R, K_C)
        g_quad = 0.5 * K_C * (R - tau)**2
        assert abs(activation_g(tau, R, K_C) - g_quad) < 1e-10

    def test_c1_at_r(self):
        """g(r) = 0 and dg/dd|_{d=r} = 0  (C¹ at activation boundary)."""
        assert activation_g(R, R, K_C) == 0.0
        assert activation_dg_dd(R, R, K_C) == 0.0

    def test_c1_at_tau(self):
        """Value and first derivative are continuous at d = τ."""
        tau, k_c_prime, b = _stitch_params(R, K_C)
        eps = 1e-8

        # Value continuity
        g_quad = 0.5 * K_C * (R - tau)**2
        g_log  = -k_c_prime * np.log(tau) + b
        assert abs(g_quad - g_log) < 1e-10, f"value gap at tau: {g_quad - g_log}"

        # Derivative continuity
        dg_quad = -K_C * (R - tau)
        dg_log  = -k_c_prime / tau
        assert abs(dg_quad - dg_log) < 1e-10, f"derivative gap at tau: {dg_quad - dg_log}"

    def test_c2_at_tau(self):
        """Second derivative is continuous at d = τ  (because τ = r/2)."""
        tau, k_c_prime, _ = _stitch_params(R, K_C)

        d2g_quad = K_C
        d2g_log  = k_c_prime / tau**2
        assert abs(d2g_quad - d2g_log) < 1e-10, f"second deriv gap: {d2g_quad - d2g_log}"

    def test_positive_in_contact_zone(self):
        """g > 0 for any d ∈ (0, r)."""
        for d in [0.01, 0.1, R / 3, R / 2 - 0.01, R / 2 + 0.01, R - 0.01]:
            assert activation_g(d, R, K_C) > 0.0, f"g not positive at d={d}"

    def test_monotone_decreasing_in_d(self):
        """g decreases as d increases (more distant → less energy)."""
        ds  = np.linspace(0.01, R - 0.01, 50)
        gs  = [activation_g(d, R, K_C) for d in ds]
        assert all(gs[i] > gs[i + 1] for i in range(len(gs) - 1)), \
            "g is not monotone decreasing in d"

    def test_fd_first_derivative(self):
        """dg/dd matches finite-difference approximation."""
        for d in [0.05, R / 3, R / 2 - 0.02, R / 2 + 0.02, R - 0.05]:
            eps  = 1e-6
            fd   = (activation_g(d + eps, R, K_C) - activation_g(d - eps, R, K_C)) / (2 * eps)
            anal = activation_dg_dd(d, R, K_C)
            assert abs(fd - anal) < 1e-5, f"FD/analytic mismatch at d={d}: fd={fd} anal={anal}"

    def test_fd_second_derivative(self):
        """d²g/dd² matches finite-difference approximation.

        At very small d the log term has large curvature (k'_c/d²), so the
        relative FD error can be larger — we allow a 1% relative tolerance
        alongside a modest absolute floor.
        """
        for d in [0.05, R / 3, R / 2 - 0.02, R / 2 + 0.02, R - 0.05]:
            eps  = 1e-5
            fd   = (activation_dg_dd(d + eps, R, K_C)
                    - activation_dg_dd(d - eps, R, K_C)) / (2 * eps)
            anal = activation_d2g_dd2(d, R, K_C)
            rel_err = abs(fd - anal) / (abs(anal) + 1e-10)
            assert rel_err < 0.01, \
                f"FD/analytic d²g mismatch at d={d}: fd={fd} anal={anal} rel_err={rel_err:.2e}"


# ======================================================================
# Vertex-facet contact energy — gradient and Hessian FD checks
# ======================================================================

# Triangle in the XY-plane: a=(0,0,0), b=(2,0,0), c=(1,2,0)
TRI_A = np.array([0.0, 0.0, 0.0])
TRI_B = np.array([2.0, 0.0, 0.0])
TRI_C = np.array([1.0, 2.0, 0.0])


def _energy_vf(v_pos: np.ndarray) -> float:
    return contact_energy_vf(v_pos, TRI_A, TRI_B, TRI_C, R, K_C)


def _gradient_vf(v_pos: np.ndarray) -> np.ndarray:
    return contact_gradient_v_vf(v_pos, TRI_A, TRI_B, TRI_C, R, K_C)


class TestContactVF:

    def test_gradient_fd_above_interior(self):
        """Vertex directly above face interior — FD gradient matches analytic."""
        v = np.array([0.9, 0.7, 0.2])   # above face interior, d ≈ 0.2
        fd_g   = fd_gradient(_energy_vf, v)
        anal_g = _gradient_vf(v)
        np.testing.assert_allclose(fd_g, anal_g, atol=1e-5,
                                   err_msg="gradient mismatch (above interior)")

    def test_gradient_fd_above_edge(self):
        """Vertex beside an edge — FD gradient matches analytic."""
        v = np.array([1.0, -0.15, 0.1])   # beside bottom edge a-b, d ≈ 0.18
        fd_g   = fd_gradient(_energy_vf, v)
        anal_g = _gradient_vf(v)
        np.testing.assert_allclose(fd_g, anal_g, atol=1e-5,
                                   err_msg="gradient mismatch (beside edge)")

    def test_gradient_fd_near_vertex(self):
        """Vertex near corner a — FD gradient matches analytic."""
        v = np.array([-0.1, -0.1, 0.15])   # near corner a=(0,0,0), d ≈ 0.21
        fd_g   = fd_gradient(_energy_vf, v)
        anal_g = _gradient_vf(v)
        np.testing.assert_allclose(fd_g, anal_g, atol=1e-5,
                                   err_msg="gradient mismatch (near vertex)")

    def test_gradient_zero_outside_radius(self):
        """Gradient is zero when vertex is farther than r from the face."""
        v = np.array([1.0, 0.7, R + 0.1])   # above interior, d > r
        g = _gradient_vf(v)
        np.testing.assert_allclose(g, np.zeros(3), atol=1e-12)

    def test_hessian_fd_above_interior(self):
        """Hessian FD check — vertex above face interior."""
        v = np.array([0.9, 0.7, 0.2])
        fd_H   = fd_hessian(_gradient_vf, v)
        anal_H = contact_hessian_v_vf(v, TRI_A, TRI_B, TRI_C, R, K_C)
        np.testing.assert_allclose(fd_H, anal_H, atol=1e-4,
                                   err_msg="Hessian mismatch (above interior)")

    def test_hessian_fd_beside_edge(self):
        """Hessian FD check — vertex beside an edge."""
        v = np.array([1.0, -0.15, 0.1])
        fd_H   = fd_hessian(_gradient_vf, v)
        anal_H = contact_hessian_v_vf(v, TRI_A, TRI_B, TRI_C, R, K_C)
        np.testing.assert_allclose(fd_H, anal_H, atol=1e-4,
                                   err_msg="Hessian mismatch (beside edge)")

    def test_hessian_zero_outside_radius(self):
        v = np.array([1.0, 0.7, R + 0.1])
        H = contact_hessian_v_vf(v, TRI_A, TRI_B, TRI_C, R, K_C)
        np.testing.assert_allclose(H, np.zeros((3, 3)), atol=1e-12)


# ======================================================================
# Edge-edge contact energy — gradient FD check
# ======================================================================

# Two skew edges
EE_P = np.array([0.0, 0.0, 0.25])   # e1 start
EE_Q = np.array([2.0, 0.0, 0.25])   # e1 end
EE_R = np.array([1.0, -1.0, 0.0])   # e2 start
EE_S = np.array([1.0,  1.0, 0.0])   # e2 end


class TestContactEE:

    def test_energy_positive(self):
        """Energy is positive when edges are within contact radius."""
        E = contact_energy_ee(EE_P, EE_Q, EE_R, EE_S, R, K_C)
        assert E > 0.0

    @pytest.mark.parametrize("v_idx,v_pos", [
        (0, EE_P),
        (1, EE_Q),
        (2, EE_R),
        (3, EE_S),
    ])
    def test_gradient_fd(self, v_idx, v_pos):
        """FD gradient check for each of the 4 edge endpoints."""
        def energy_fn(x):
            verts = [EE_P.copy(), EE_Q.copy(), EE_R.copy(), EE_S.copy()]
            verts[v_idx] = x
            return contact_energy_ee(*verts, R, K_C)

        fd_g   = fd_gradient(energy_fn, v_pos)
        anal_g = contact_gradient_v_ee(v_pos, v_idx, EE_P, EE_Q, EE_R, EE_S, R, K_C)
        np.testing.assert_allclose(fd_g, anal_g, atol=1e-4,
                                   err_msg=f"EE gradient mismatch for v_idx={v_idx}")

    def test_gradient_zero_outside_radius(self):
        """Gradient is zero when edges are farther than r."""
        p = np.array([0.0, 0.0, R + 0.1])
        q = np.array([2.0, 0.0, R + 0.1])
        for v_idx, v in enumerate([p, q, EE_R, EE_S]):
            g = contact_gradient_v_ee(v, v_idx, p, q, EE_R, EE_S, R, K_C)
            np.testing.assert_allclose(g, np.zeros(3), atol=1e-12,
                                       err_msg=f"non-zero gradient outside r for v_idx={v_idx}")


# ======================================================================
# Conservative bounds — compute_conservative_bounds
# ======================================================================

class TestConservativeBounds:

    def _make_simple_scene(self):
        """
        4-vertex, 2-triangle mesh.

           V2 --- V3
           | \\  T1 |
           |  \\    |
           | T0 \\ |
           V0 --- V1
        """
        V = np.array([
            [0., 0., 0.],
            [2., 0., 0.],
            [0., 2., 0.],
            [2., 2., 0.],
        ])
        T = np.array([[0, 1, 2], [1, 3, 2]])
        return Mesh.from_arrays(V, T)

    def test_bounds_use_d_min_v(self):
        """b_v = γ_p * d_min_v when d_min_v is the tightest constraint."""
        mesh = self._make_simple_scene()
        gamma_p = 0.45

        # Build a ContactSets with only d_min_v filled in
        cs = ContactSets()
        d_val = 0.3
        for v in range(mesh.num_vertices):
            cs.d_min_v[v] = d_val
            for e in mesh.E_v[v]:
                cs.d_min_e[e] = 999.0    # very large — not the tightest
            for t in mesh.T_v[v]:
                cs.d_min_t[t] = 999.0

        bounds = compute_conservative_bounds(mesh, cs, gamma_p)

        for v in range(mesh.num_vertices):
            expected = gamma_p * d_val
            assert abs(bounds[v] - expected) < 1e-12, \
                f"V{v}: expected b_v={expected}, got {bounds[v]}"

    def test_bounds_use_minimum_of_three(self):
        """b_v uses min(d_min_v, d_min_e_v, d_min_t_v)."""
        mesh = self._make_simple_scene()
        gamma_p = 0.45
        cs = ContactSets()

        d_min_v_val = 0.5
        d_min_e_val = 0.2    # this is the tightest for vertices that have edges
        d_min_t_val = 0.8

        for v in range(mesh.num_vertices):
            cs.d_min_v[v] = d_min_v_val
        for e in range(mesh.num_edges):
            cs.d_min_e[e] = d_min_e_val
        for t in range(mesh.num_triangles):
            cs.d_min_t[t] = d_min_t_val

        bounds = compute_conservative_bounds(mesh, cs, gamma_p)

        for v in range(mesh.num_vertices):
            # d_min_e_v = min over neighbour edges = 0.2  (tightest)
            expected = gamma_p * d_min_e_val
            assert abs(bounds[v] - expected) < 1e-12, \
                f"V{v}: expected {expected}, got {bounds[v]}"

    def test_bounds_all_positive(self):
        """All bounds must be non-negative."""
        mesh = self._make_simple_scene()
        cs = ContactSets()
        for v in range(mesh.num_vertices):
            cs.d_min_v[v] = 0.1
        for e in range(mesh.num_edges):
            cs.d_min_e[e] = 0.1
        for t in range(mesh.num_triangles):
            cs.d_min_t[t] = 0.1

        bounds = compute_conservative_bounds(mesh, cs, gamma_p=0.45)
        for v, b in bounds.items():
            assert b >= 0.0, f"V{v} has negative bound {b}"


# ======================================================================
# truncate_displacements
# ======================================================================

class TestTruncateDisplacements:

    def test_no_truncation_when_within_bounds(self):
        """Vertices that move less than b_v are left unchanged."""
        X_prev = np.zeros((3, 3))
        X      = np.array([[0.05, 0., 0.],
                            [0., 0.05, 0.],
                            [0., 0., 0.05]])
        bounds = {0: 0.1, 1: 0.1, 2: 0.1}
        X_out, n = truncate_displacements(X, X_prev, bounds)
        np.testing.assert_allclose(X_out, X, atol=1e-12)
        assert n == 0

    def test_truncation_when_exceeding_bounds(self):
        """Vertex that moves too far is clamped to exactly b_v from X_prev."""
        X_prev = np.zeros((3, 3))
        b_v    = 0.1
        # V0 moves by 0.3 — exceeds b_v = 0.1
        X      = np.array([[0.3, 0., 0.],
                            [0., 0.05, 0.],
                            [0., 0., 0.05]])
        bounds = {0: b_v, 1: 0.5, 2: 0.5}
        X_out, n = truncate_displacements(X, X_prev, bounds)

        assert n == 1
        dist = float(np.linalg.norm(X_out[0] - X_prev[0]))
        assert abs(dist - b_v) < 1e-12, f"truncated distance {dist} != b_v {b_v}"

    def test_truncation_direction_preserved(self):
        """After truncation, the vertex moves in the same direction as before."""
        X_prev = np.zeros((3, 3))
        delta  = np.array([0.6, 0.8, 0.0])   # unit direction (0.6, 0.8, 0)
        b_v    = 0.5
        X      = np.vstack([delta[np.newaxis, :], np.zeros((2, 3))])
        bounds = {0: b_v, 1: 1.0, 2: 1.0}
        X_out, _ = truncate_displacements(X, X_prev, bounds)

        direction_before = delta / np.linalg.norm(delta)
        direction_after  = (X_out[0] - X_prev[0])
        direction_after  /= np.linalg.norm(direction_after)
        np.testing.assert_allclose(direction_after, direction_before, atol=1e-12)

    def test_all_vertices_can_be_truncated(self):
        """All vertices truncated → num_exceed equals total count."""
        X_prev = np.zeros((4, 3))
        X      = np.ones((4, 3)) * 10.0   # each moves by ~17.3 >> b_v
        bounds = {v: 0.01 for v in range(4)}
        _, n = truncate_displacements(X, X_prev, bounds)
        assert n == 4

    def test_no_penetration_after_truncation(self):
        """
        Verify the key safety property (Eq. 27):
        after truncation, ||x_v - x_prev_v|| <= b_v for every vertex.
        """
        rng    = np.random.default_rng(42)
        N      = 20
        X_prev = rng.random((N, 3))
        X      = X_prev + rng.random((N, 3)) * 2.0   # large random displacements
        bounds = {v: rng.uniform(0.05, 0.3) for v in range(N)}

        X_out, _ = truncate_displacements(X, X_prev, bounds)

        for v, b_v in bounds.items():
            dist = float(np.linalg.norm(X_out[v] - X_prev[v]))
            assert dist <= b_v + 1e-12, \
                f"V{v}: dist {dist:.6f} > b_v {b_v:.6f} after truncation"
