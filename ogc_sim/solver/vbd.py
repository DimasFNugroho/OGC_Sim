"""
VBD Iteration with Contact — Algorithm 4.

Paper reference: Sec. 4.3, Algorithm 4.

One "VBD iteration" is a single pass of Vertex Block Descent over all
vertices, processed color-by-color so that all same-colored vertices are
independent (no shared triangle or edge) and can be updated in parallel.

Per-vertex update (line 28):
    x_v  ←  x_v  +  H_v⁻¹ f_v

where f_v and H_v accumulate four contributions (lines 3–27):

    1. Inertia           f = -(m/h²)(x - y),    H = (m/h²) I        (line 3)
    2. Elastic/triangle  f = -∂E_t/∂x_v,         H = ∂²E_t/∂x_v²   (lines 4-7)
    3. Elastic/bending   f = -∂E_e/∂x_v,         H = ∂²E_e/∂x_v²   (lines 8-11)
    4. VF contact vertex-side  (FOGC)                                 (lines 12-15)
    5. VF contact face-side    (VOGC)                                 (lines 16-21)
    6. EE contact              (EOGC)                                 (lines 22-27)

Elastic model (M4 placeholder — M5 will replace with StVK):
    Mass-spring over edges: E_edge = (k_s/2)(||x_i - x_j|| - l₀)²
    Used for both "triangle elastic" (stretching) and "edge bending" slots.

Face-side VF contact gradient (lines 16-21):
    ∂E_vf(v', t)/∂x_v  where v is a vertex of triangle t.
    We use a finite-difference approximation here; M5 will add the
    analytic formula once the full cloth material is in place.
"""

from __future__ import annotations

import numpy as np
import networkx as nx

from ogc_sim.geometry.mesh import Mesh
from ogc_sim.contact.detection import ContactSets
from ogc_sim.contact.energy import (
    contact_gradient_v_vf,
    contact_hessian_v_vf,
    contact_gradient_v_ee,
    contact_hessian_v_ee,
)
from ogc_sim.geometry.distance import point_triangle_distance


# ======================================================================
# Graph coloring
# ======================================================================

def graph_color_mesh(mesh: Mesh) -> list[list[int]]:
    """
    Greedy graph-coloring of mesh vertices so that no two vertices of the
    same triangle (or edge) share a color.

    Returns
    -------
    colors : list[list[int]]
        colors[c] = list of vertex indices with color c.
        Process colors[0], colors[1], … in order each VBD pass.
    """
    G = nx.Graph()
    G.add_nodes_from(range(mesh.num_vertices))

    # Two vertices conflict if they share a triangle
    for tri in mesh.T:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        G.add_edges_from([(a, b), (b, c), (a, c)])

    coloring = nx.greedy_color(G, strategy="largest_first")

    num_colors = max(coloring.values()) + 1
    colors: list[list[int]] = [[] for _ in range(num_colors)]
    for v, c in coloring.items():
        colors[c].append(v)
    return colors


# ======================================================================
# Mass-spring elastic energy (M4 placeholder)
# ======================================================================

def compute_rest_lengths(mesh: Mesh) -> np.ndarray:
    """
    Compute rest edge lengths from the mesh's current vertex positions.

    Returns
    -------
    l0 : np.ndarray, shape (num_edges,)
        Rest length for each edge.
    """
    l0 = np.zeros(mesh.num_edges)
    for ei, (a, b) in enumerate(mesh.E):
        l0[ei] = float(np.linalg.norm(mesh.V[a] - mesh.V[b]))
    return l0


def spring_force_hessian(
    v_idx: int,
    X: np.ndarray,
    mesh: Mesh,
    l0: np.ndarray,
    k_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gradient and Hessian of mass-spring elastic energy w.r.t. vertex v_idx.

    Sums over all edges incident to v (i.e., all E_v[v_idx] edges), which
    covers contributions from all neighbouring triangles (lines 4-7 and
    bending lines 8-11 are merged into a single spring sum here).

    Parameters
    ----------
    v_idx : int
    X     : (N, 3) current positions
    mesh  : Mesh
    l0    : (num_edges,) rest lengths
    k_s   : float  spring stiffness

    Returns
    -------
    f_v   : (3,) force on v (= -∂E/∂x_v)
    H_v   : (3, 3) Hessian block
    """
    f_v = np.zeros(3)
    H_v = np.zeros((3, 3))

    xv = X[v_idx]

    for ei in mesh.E_v[v_idx]:
        a, b = int(mesh.E[ei][0]), int(mesh.E[ei][1])
        xo = X[b] if a == v_idx else X[a]   # other endpoint

        diff = xv - xo
        l    = float(np.linalg.norm(diff))
        if l < 1e-12:
            continue
        l_0  = l0[ei]
        d    = diff / l         # unit direction  v → other

        # Gradient of (k_s/2)(l - l_0)² w.r.t. x_v
        # = k_s (l - l_0) * d
        f_v -= k_s * (l - l_0) * d   # force = -grad

        # Hessian
        # ∂²E/∂xv² = k_s * d⊗d + k_s (1 - l_0/l) * (I - d⊗d)
        ddt   = np.outer(d, d)
        H_v  += k_s * ddt + k_s * (1.0 - l_0 / l) * (np.eye(3) - ddt)

    return f_v, H_v


# ======================================================================
# Face-side VF contact gradient (lines 16-21)
# ======================================================================

def _contact_grad_tri_vf_fd(
    v_pos:  np.ndarray,
    a_pos:  np.ndarray,
    b_pos:  np.ndarray,
    c_pos:  np.ndarray,
    r:      float,
    k_c:    float,
    eps:    float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Finite-difference gradient of VF contact energy E_vf(v, triangle)
    w.r.t. each of the three triangle vertices a, b, c.

    Used in lines 16-21 to compute f_t,v' = -∂E_vf(v', t)/∂x_v where
    x_v is a vertex of triangle t.

    Note: M5 will replace this with an analytic formula once the full
    cloth material model is in place.

    Returns
    -------
    (ga, gb, gc) : each (3,) — gradient of E_vf w.r.t. a, b, c respectively.
    """
    from ogc_sim.contact.energy import contact_energy_vf

    def _energy(a, b, c):
        return contact_energy_vf(v_pos, a, b, c, r, k_c)

    def _fd_grad(pt):
        g = np.zeros(3)
        for i in range(3):
            p_plus  = pt.copy(); p_plus[i]  += eps
            p_minus = pt.copy(); p_minus[i] -= eps
            g[i] = (_energy(*(
                (p_plus,  b_pos, c_pos) if np.all(pt == a_pos) else
                (a_pos, p_plus,  c_pos) if np.all(pt == b_pos) else
                (a_pos, b_pos, p_plus )
            )) - _energy(*(
                (p_minus, b_pos, c_pos) if np.all(pt == a_pos) else
                (a_pos, p_minus, c_pos) if np.all(pt == b_pos) else
                (a_pos, b_pos, p_minus)
            ))) / (2.0 * eps)
        return g

    ga = _fd_grad(a_pos)
    gb = _fd_grad(b_pos)
    gc = _fd_grad(c_pos)
    return ga, gb, gc


def _contact_hessian_tri_vf_fd(
    v_pos:  np.ndarray,
    a_pos:  np.ndarray,
    b_pos:  np.ndarray,
    c_pos:  np.ndarray,
    r:      float,
    k_c:    float,
    eps:    float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Finite-difference Hessian ∂²E_vf/∂x_u² for u ∈ {a, b, c}.
    Returns (H_a, H_b, H_c), each (3,3).
    """
    from ogc_sim.contact.energy import contact_gradient_v_vf as grad_v

    def _grad_tri(which_pt, delta):
        """Gradient of E w.r.t. which_pt, perturbed by delta along axis."""
        a, b, c = a_pos.copy(), b_pos.copy(), c_pos.copy()
        if which_pt == 0:
            a = a + delta
        elif which_pt == 1:
            b = b + delta
        else:
            c = c + delta
        # We want ∂E/∂x_u; use the fact that E = g(d(v, tri))
        from ogc_sim.contact.energy import contact_energy_vf
        # Central-difference along each component of x_u
        return contact_energy_vf(v_pos, a, b, c, r, k_c)

    def _H_one(which_pt, pt):
        H = np.zeros((3, 3))
        for i in range(3):
            ei = np.zeros(3); ei[i] = eps
            delta_plus  = ei
            delta_minus = -ei
            gp = _contact_grad_tri_vf_fd(v_pos, a_pos, b_pos, c_pos, r, k_c, eps)[which_pt]
            # Use FD of the FD gradient (2nd order)
            a_p, b_p, c_p = a_pos.copy(), b_pos.copy(), c_pos.copy()
            a_m, b_m, c_m = a_pos.copy(), b_pos.copy(), c_pos.copy()
            if which_pt == 0:
                a_p[i] += eps; a_m[i] -= eps
            elif which_pt == 1:
                b_p[i] += eps; b_m[i] -= eps
            else:
                c_p[i] += eps; c_m[i] -= eps
            gp_plus  = _contact_grad_tri_vf_fd(v_pos, a_p, b_p, c_p, r, k_c, eps)[which_pt]
            gp_minus = _contact_grad_tri_vf_fd(v_pos, a_m, b_m, c_m, r, k_c, eps)[which_pt]
            H[:, i]  = (gp_plus - gp_minus) / (2.0 * eps)
        return H

    return _H_one(0, a_pos), _H_one(1, b_pos), _H_one(2, c_pos)


# ======================================================================
# Algorithm 4 — one VBD iteration
# ======================================================================

def vbd_iteration(
    X:      np.ndarray,
    X_t:    np.ndarray,
    Y:      np.ndarray,
    mesh:   Mesh,
    cs:     ContactSets,
    colors: list[list[int]],
    l0:     np.ndarray,
    dt:     float,
    mass:   float | np.ndarray,
    k_s:    float,
    r:      float,
    k_c:    float,
    n_dof:  int | None = None,
) -> np.ndarray:
    """
    One full VBD pass (Algorithm 4).

    Updates X in-place color-by-color (Gauss-Seidel style).

    Parameters
    ----------
    X       : (N, 3) current positions; modified in place
    X_t     : (N, 3) positions at the start of the time step
    Y       : (N, 3) inertia target  Y = X_t + dt·v + dt²·a_ext
    mesh    : Mesh
    cs      : ContactSets from the most recent contact detection
    colors  : list[list[int]] from graph_color_mesh
    l0      : (num_edges,) rest lengths
    dt      : float  time step
    mass    : float or (N,)  per-vertex mass (scalar → broadcast)
    k_s     : float  spring stiffness
    r       : float  contact radius
    k_c     : float  contact stiffness
    n_dof   : int or None
        If given, only update vertices 0…n_dof-1 (cloth vertices).
        Vertices ≥ n_dof are treated as static obstacles.

    Returns
    -------
    X : (N, 3)  updated positions (same array as input)
    """
    h2   = dt * dt
    M    = np.broadcast_to(np.asarray(mass, dtype=float).reshape(-1), (mesh.num_vertices,))

    for color_group in colors:                          # Algorithm 4, line 1
        for v in color_group:                           # line 2  (parallel in GPU)
            if n_dof is not None and v >= n_dof:
                continue                                # static vertex — skip

            m_v = M[v]
            xv  = X[v]
            yv  = Y[v]

            # ----------------------------------------------------------
            # line 3: inertia term
            # ----------------------------------------------------------
            f_v = -(m_v / h2) * (xv - yv)
            H_v = (m_v / h2) * np.eye(3)

            # ----------------------------------------------------------
            # lines 4-7: elastic from incident triangles (+ lines 8-11:
            # bending from incident edges — merged as mass-spring here)
            # ----------------------------------------------------------
            f_e, H_e = spring_force_hessian(v, X, mesh, l0, k_s)
            f_v += f_e
            H_v += H_e

            # ----------------------------------------------------------
            # lines 12-15: VF contact, vertex side (FOGC)
            # ----------------------------------------------------------
            for a_feat in cs.FOGC.get(v, []):           # line 12
                # Identify which triangle to use for energy evaluation.
                # FOGC stores global feature indices; find which triangle
                # triangle side they come from by searching VOGC.
                # For the energy we use the floor triangle (assumed to be
                # the one with the smallest index not containing v).
                # In a general mesh we would look up the triangle from
                # the BVH result; here we scan T.
                t_idx = _find_parent_triangle(a_feat, mesh)
                if t_idx < 0:
                    continue
                tri = mesh.T[t_idx]
                ap  = X[int(tri[0])]; bp = X[int(tri[1])]; cp = X[int(tri[2])]

                # line 13: f_v,a = -∂E_vf(v, a)/∂x_v
                fva = contact_gradient_v_vf(xv, ap, bp, cp, r, k_c)
                Hva = contact_hessian_v_vf(xv, ap, bp, cp, r, k_c)
                f_v -= fva                              # force = -gradient
                H_v += Hva                              # line 15

            # ----------------------------------------------------------
            # lines 16-21: VF contact, face side (VOGC)
            #   For each triangle t containing v, accumulate force from
            #   each vertex v' that is pressing on t.
            # ----------------------------------------------------------
            for t_idx in mesh.T_v[v]:                   # line 16
                for vp in cs.VOGC.get(t_idx, []):       # line 17
                    if vp == v:
                        continue
                    tri = mesh.T[t_idx]
                    ap  = X[int(tri[0])]; bp = X[int(tri[1])]; cp = X[int(tri[2])]
                    xvp = X[vp]

                    # line 18: grad of E_vf(v', t) w.r.t. x_v (a vertex of t)
                    v_local = [int(tri[0]), int(tri[1]), int(tri[2])].index(v)
                    ga, gb, gc = _contact_grad_tri_vf_fd(xvp, ap, bp, cp, r, k_c)
                    grad_wrt_v = [ga, gb, gc][v_local]
                    Ha, Hb, Hc = _contact_hessian_tri_vf_fd(xvp, ap, bp, cp, r, k_c)
                    H_wrt_v    = [Ha, Hb, Hc][v_local]

                    f_v -= grad_wrt_v                   # line 20
                    H_v += H_wrt_v

            # ----------------------------------------------------------
            # lines 22-27: EE contact (EOGC)
            # ----------------------------------------------------------
            for e_idx in mesh.E_v[v]:                   # line 22
                for (e1, e2) in cs.EOGC.get(e_idx, []): # line 23
                    # Determine which endpoint of e1 is v
                    ea, eb   = int(mesh.E[e1][0]), int(mesh.E[e1][1])
                    ra_, sb_ = int(mesh.E[e2][0]), int(mesh.E[e2][1])

                    if v == ea:
                        v_role = 0
                    elif v == eb:
                        v_role = 1
                    else:
                        # v is an endpoint of e2 in the EOGC pair
                        if v == ra_:
                            v_role = 2
                        elif v == sb_:
                            v_role = 3
                        else:
                            continue

                    p, q   = X[ea], X[eb]
                    re, se = X[ra_], X[sb_]

                    # line 24: f_e,e' = -∂E_ee(e,e')/∂x_v
                    fve = contact_gradient_v_ee(X[v], v_role, p, q, re, se, r, k_c)
                    Hve = contact_hessian_v_ee(v_role, p, q, re, se, r, k_c)
                    f_v -= fve                          # line 26
                    H_v += Hve

            # ----------------------------------------------------------
            # line 28: Newton step  x_v ← x_v + H_v⁻¹ f_v
            # ----------------------------------------------------------
            # Clamp H_v to be PSD (project out negative eigenvalues)
            eigvals, eigvecs = np.linalg.eigh(H_v)
            eigvals_clamped  = np.maximum(eigvals, 1e-9)
            H_psd = eigvecs @ np.diag(eigvals_clamped) @ eigvecs.T
            try:
                delta = np.linalg.solve(H_psd, f_v)
            except np.linalg.LinAlgError:
                delta = np.zeros(3)

            X[v] = xv + delta                           # line 28

    return X


# ======================================================================
# Helper: find any triangle that "owns" a given global feature index
# ======================================================================

def _find_parent_triangle(feat_idx: int, mesh: Mesh) -> int:
    """
    Given a global feature index (could be a triangle index, edge index,
    or vertex index as stored in FOGC), return the index of any triangle
    whose closest sub-feature matches it.

    Strategy:
    - If feat_idx < num_triangles and feat_idx is a valid triangle: return it.
    - If feat_idx is a vertex index: return first triangle containing it.
    - If feat_idx is an edge index: return first triangle containing that edge.

    Returns -1 if nothing found.
    """
    # Try as a triangle index first
    if 0 <= feat_idx < mesh.num_triangles:
        return feat_idx

    # Try as a vertex index
    if 0 <= feat_idx < mesh.num_vertices:
        tv = mesh.T_v[feat_idx]
        if tv:
            return tv[0]

    # Try as an edge index
    if 0 <= feat_idx < mesh.num_edges:
        a, b = int(mesh.E[feat_idx][0]), int(mesh.E[feat_idx][1])
        for t_idx in mesh.T_v[a]:
            if b in mesh.T[t_idx]:
                return t_idx

    return -1
