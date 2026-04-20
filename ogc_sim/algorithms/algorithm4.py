"""
Algorithm 4 — VBD Iteration with Contact (Inner Solver)
=======================================================
Paper reference: Sec. 4.3, Algorithm 4.

One full VBD pass: process vertices color-by-color, accumulating
inertia + elastic + contact forces and Hessians, then apply a
per-vertex 3x3 Newton step.

Algorithm structure
-------------------
  1   for each color c:
  2     for each vertex v in color c  (parallel on GPU):
  3       f_v = -(m/h^2)(x_v - y_v),   H_v = (m/h^2) I       (inertia)
  4-7     for each triangle t in T_v:
            f_v += -dE_t/dx_v,   H_v += d^2E_t/dx_v^2        (elastic)
  8-11    for each edge e in E_v:
            f_v += -dE_e/dx_v,   H_v += d^2E_e/dx_v^2        (bending)
  12-15   for each a in FOGC(v):
            f_v += -dE_vf(v,a)/dx_v,  H_v += d^2E_vf/dx_v^2  (VF contact, vertex side)
  16-21   for each t in T_v, v' in VOGC(t):
            f_v += -dE_vf(v',t)/dx_v,  H_v += d^2E_vf/dx_v^2 (VF contact, face side)
  22-27   for each e in E_v, (e,e') in EOGC(e):
            f_v += -dE_ee(e,e')/dx_v,  H_v += d^2E_ee/dx_v^2 (EE contact)
  28      x_v <- x_v + H_v^{-1} f_v                           (Newton step)
"""

from __future__ import annotations

import numpy as np

from ogc_sim.geometry.mesh import Mesh
from ogc_sim.contact.detection import ContactSets
from ogc_sim.solver.vbd import (
    vbd_iteration as _vbd_core,
    graph_color_mesh,
    compute_rest_lengths,
    spring_force_hessian,
)


def vbd_iteration(
    X: np.ndarray,
    X_t: np.ndarray,
    Y: np.ndarray,
    mesh: Mesh,
    cs: ContactSets,
    colors: list[list[int]],
    l0: np.ndarray,
    dt: float,
    mass: float | np.ndarray,
    k_s: float,
    r: float,
    k_c: float,
    n_dof: int | None = None,
) -> np.ndarray:
    """
    Algorithm 4: one full VBD pass.

    Updates X in-place color-by-color (Gauss-Seidel style).

    Parameters
    ----------
    X       : (N, 3) current positions; modified in place
    X_t     : (N, 3) positions at the start of the time step
    Y       : (N, 3) inertia target  Y = X_t + dt * v + dt^2 * a_ext
    mesh    : Mesh
    cs      : ContactSets from the most recent contact detection
    colors  : list[list[int]] from graph_color_mesh
    l0      : (num_edges,) rest edge lengths
    dt      : float  time step
    mass    : float or (N,)  per-vertex mass (scalar -> broadcast)
    k_s     : float  spring stiffness
    r       : float  contact radius
    k_c     : float  contact stiffness
    n_dof   : int or None
        If given, only update vertices 0..n_dof-1 (cloth vertices).
        Vertices >= n_dof are treated as static obstacles.

    Returns
    -------
    X : (N, 3)  updated positions (same array as input)
    """
    return _vbd_core(
        X, X_t, Y, mesh, cs, colors, l0,
        dt=dt, mass=mass, k_s=k_s, r=r, k_c=k_c, n_dof=n_dof,
    )
