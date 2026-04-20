"""
Algorithm 3 — Simulation Step (Outer Time-Step Loop)
====================================================
Paper reference: Sec. 4.3, Algorithm 3.

Orchestrates one full time step: computes the inertia target Y,
runs contact detection (Algorithms 1 & 2), computes conservative
bounds, calls the inner solver (Algorithm 4), and truncates
displacements.

Algorithm structure
-------------------
  1   collisionDetectionRequired = True
  2   X = X_t
  3   Y = X_t + dt * v_t + dt^2 * a_ext                (inertia target)
  4   for i in 1 .. n_iter:
  5     if collisionDetectionRequired:
  6-8     reset d_min_t -> r_q for every triangle
  9-11    FOGC, d_min_v <- Algorithm 1 (vertex-facet)
  12-14   EOGC, d_min_e <- Algorithm 2 (edge-edge)
  15      X_prev = X
  16      collisionDetectionRequired = False
  17-19   b_v <- computeConservativeBound(v)             (Eq. 21)
  20    if i == 1:
  21      X <- applyInitialGuess(X_t, v_t, a_ext)       (Eq. 28)
  22    X <- simulationIteration(...)                    (Algorithm 4)
  23    numExceed = 0
  24-27 for each v: if ||x_v - x_prev_v|| > b_v -> truncate, numExceed++
  28-29 if numExceed >= gamma_e * K -> collisionDetectionRequired = True
  30    [optional convergence check]
  34  return X
"""

from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import torch

from ogc_sim.geometry.mesh import Mesh
from ogc_sim.geometry.bvh import BVH
from ogc_sim.geometry.gauss_map import PolyhedralGaussMap
from ogc_sim.contact.detection import run_contact_detection, ContactSets
from ogc_sim.contact.bounds import (
    compute_conservative_bounds,
    truncate_displacements,
    apply_initial_guess_truncation,
)
from ogc_sim.solver.vbd import (
    vbd_iteration as _vbd_iteration,
    graph_color_mesh,
    compute_rest_lengths,
)

# GPU modules (imported lazily so CPU-only environments still work)
_CUDA_AVAILABLE = torch.cuda.is_available()
if _CUDA_AVAILABLE:
    from ogc_sim.contact.detection_gpu import detect_contacts_gpu
    from ogc_sim.solver.vbd_gpu import vbd_iteration_gpu, build_gpu_mesh_data, GPUMeshData
    _GPU_DEVICE = torch.device("cuda")
else:
    _GPU_DEVICE = None


@dataclass
class StepResult:
    """Result returned by simulation_step."""
    X: np.ndarray                       # final positions after this time step
    v: np.ndarray                       # updated velocities
    num_detections: int = 0             # how many times contact detection ran
    frames: list[dict] = field(default_factory=list)  # per-iteration snapshots


def simulation_step(
    X_t: np.ndarray,
    v_t: np.ndarray,
    V_floor: np.ndarray,
    T_cloth: np.ndarray,
    T_floor: np.ndarray,
    colors: list[list[int]],
    l0: np.ndarray,
    dt: float,
    a_ext: np.ndarray,
    r: float,
    r_q: float,
    gamma_p: float,
    gamma_e: float,
    n_iter: int,
    mass: float,
    k_s: float,
    k_c: float,
    record_frames: bool = False,
    gpu_mesh_data=None,       # GPUMeshData | None — pre-built in runner.load()
) -> StepResult:
    """
    Algorithm 3: one full simulation time step.

    Parameters
    ----------
    X_t      : (N_cloth, 3) cloth vertex positions at start of step
    v_t      : (N_cloth, 3) cloth vertex velocities at start of step
    V_floor  : (M, 3) static floor vertex positions
    T_cloth  : (K, 3) cloth triangle indices (0-based into X_t)
    T_floor  : (L, 3) floor triangle indices (0-based into V_floor)
    colors   : list[list[int]] graph coloring of the cloth mesh
    l0       : (num_edges,) rest edge lengths for the combined mesh
    dt       : float  time step size
    a_ext    : (3,) external acceleration (e.g. gravity)
    r        : float  contact radius
    r_q      : float  query radius (>= r)
    gamma_p  : float  conservative bound relaxation (0 < gamma_p < 0.5)
    gamma_e  : float  re-detection threshold (fraction of vertices)
    n_iter   : int    number of inner solver iterations
    mass     : float  per-vertex mass
    k_s      : float  spring stiffness
    k_c      : float  contact stiffness
    record_frames : bool  if True, save per-iteration snapshots

    Returns
    -------
    StepResult with final X, v, and optional frame data
    """
    N_cloth = len(X_t)
    N_obs   = len(V_floor)

    # Algorithm 3, line 3: inertia target
    Y = X_t + dt * v_t + dt**2 * a_ext

    # ------------------------------------------------------------------ #
    # Build the combined mesh topology ONCE.  Only V changes each iter.   #
    # ------------------------------------------------------------------ #
    T_all    = np.vstack([T_cloth, T_floor + N_cloth])
    V_scene  = np.vstack([X_t, V_floor])
    mesh     = Mesh.from_arrays(V_scene, T_all)

    # Build obstacle-only mesh for BVH/PGM — obstacle never moves so we
    # build it once and never rebuild it.
    mesh_obs = Mesh.from_arrays(V_floor, T_floor)
    bvh_obs  = BVH(mesh_obs)
    pgm_obs  = PolyhedralGaussMap(mesh_obs)

    X_t_full = np.vstack([X_t, V_floor])
    Y_full   = np.vstack([Y,   V_floor])

    # Algorithm 3, line 1
    collision_detection_required = True
    X_cur  = X_t.copy()
    X_prev = X_t.copy()
    b_v: dict[int, float] = {v: r_q for v in range(N_cloth)}
    cs: ContactSets | None = None
    num_detections = 0
    frames: list[dict] = []

    for i in range(1, n_iter + 1):
        # Sync cloth positions into the shared mesh
        mesh.V[:N_cloth] = X_cur

        # Algorithm 3, lines 5-19: contact detection when required
        if collision_detection_required:
            if _CUDA_AVAILABLE and gpu_mesh_data is not None:
                # GPU brute-force all-pairs: ~4M distance computations in one kernel
                cs = detect_contacts_gpu(X_cur, mesh_obs, r, r_q, _GPU_DEVICE)
            else:
                # CPU BVH-accelerated cloth-vs-obstacle detection
                mesh_cloth_cur = Mesh.from_arrays(X_cur, T_cloth)
                cs = _cloth_vs_obstacle_detection(
                    mesh_cloth_cur, mesh_obs, bvh_obs, pgm_obs, r, r_q, N_cloth
                )

            X_prev = X_cur.copy()                         # line 15
            collision_detection_required = False           # line 16
            num_detections += 1

            # lines 17-19: conservative bounds (cloth vertices only)
            b_v_all = compute_conservative_bounds(mesh, cs, gamma_p)
            b_v = {v: b_v_all.get(v, r_q) for v in range(N_cloth)}

        # Algorithm 3, lines 20-21: initial guess truncation (first iter only)
        if i == 1:
            X_cur = apply_initial_guess_truncation(Y.copy(), X_prev, b_v)
            mesh.V[:N_cloth] = X_cur

        # Algorithm 3, line 22: one VBD iteration (Algorithm 4)
        if _CUDA_AVAILABLE and gpu_mesh_data is not None:
            vbd_iteration_gpu(
                mesh.V, X_t_full, Y_full,
                mesh,
                cs if cs is not None else ContactSets(),
                colors, gpu_mesh_data,
                dt=dt, mass=mass, k_s=k_s, r=r, k_c=k_c, n_dof=N_cloth,
            )
        else:
            _vbd_iteration(
                mesh.V, X_t_full, Y_full,
                mesh,
                cs if cs is not None else ContactSets(),
                colors, l0,
                dt=dt, mass=mass, k_s=k_s, r=r, k_c=k_c, n_dof=N_cloth,
            )
        X_cur = mesh.V[:N_cloth].copy()

        # Algorithm 3, lines 23-28: bound truncation
        X_cur, num_exceed = truncate_displacements(X_cur, X_prev, b_v)

        # Algorithm 3, lines 28-29: re-detection trigger
        K = N_cloth
        if num_exceed >= gamma_e * K and num_exceed > 0:
            collision_detection_required = True

        if record_frames:
            frames.append({
                "X": X_cur.copy(),
                "b_v": dict(b_v),
                "detected": (collision_detection_required or i == 1),
                "outer_iter": i,
                "num_exceed": num_exceed,
            })

    # Update velocity
    v_new = (X_cur - X_t) / dt

    return StepResult(
        X=X_cur,
        v=v_new,
        num_detections=num_detections,
        frames=frames,
    )


def _cloth_vs_obstacle_detection(
    mesh_cloth: Mesh,
    mesh_obs: Mesh,
    bvh_obs: BVH,
    pgm_obs: PolyhedralGaussMap,
    r: float,
    r_q: float,
    cloth_vertex_offset: int,
) -> ContactSets:
    """
    Fast contact detection: cloth vertices vs obstacle faces only.

    Queries each cloth vertex position against the static obstacle BVH.
    Obstacle vertices/edges are never queried — the obstacle never moves.

    Returns ContactSets with FOGC, VOGC, d_min_v populated.
    """
    from ogc_sim.geometry.distance import point_triangle_distance, ClosestFeature
    from ogc_sim.contact.offset_geometry import feasible_vf_contact

    cs = ContactSets()

    for t_idx in range(mesh_obs.num_triangles):
        cs.d_min_t[t_idx] = r_q

    for v_idx in range(mesh_cloth.num_vertices):
        v_pos   = mesh_cloth.V[v_idx]
        d_min_v = r_q
        fogc_v: list[int] = []
        vogc_t: list[int] = []

        # BVH query against obstacle triangles
        candidate_tris = bvh_obs.sphere_query_triangles(v_pos, r_q)

        for t_idx in candidate_tris:
            tri   = mesh_obs.T[t_idx]
            a_pos = mesh_obs.V[int(tri[0])]
            b_pos = mesh_obs.V[int(tri[1])]
            c_pos = mesh_obs.V[int(tri[2])]

            dist, cp, feature, local_feat_idx = point_triangle_distance(
                v_pos, a_pos, b_pos, c_pos
            )

            d_min_v = min(d_min_v, dist)
            if dist < cs.d_min_t.get(t_idx, r_q):
                cs.d_min_t[t_idx] = dist

            if dist >= r:
                continue

            # Map to global feature index (in obstacle mesh coordinate)
            if feature == ClosestFeature.FACE_INTERIOR:
                global_feat_idx = t_idx
            elif feature == ClosestFeature.EDGE:
                global_feat_idx = mesh_obs.E_t[t_idx][local_feat_idx]
            else:
                global_feat_idx = int(tri[local_feat_idx])

            if global_feat_idx in fogc_v:
                continue  # de-duplicate

            # Feasibility check via Gauss Map on the obstacle
            direction = v_pos - cp
            passed = False
            if feature == ClosestFeature.FACE_INTERIOR:
                passed = True
            elif feature == ClosestFeature.VERTEX:
                passed = pgm_obs.is_in_vertex_normal_cone(direction, global_feat_idx)
            else:
                passed = pgm_obs.is_in_edge_normal_slab(direction, global_feat_idx)

            if passed:
                fogc_v.append(global_feat_idx)
                vogc_t.append(t_idx)

        cs.FOGC[v_idx]    = fogc_v
        cs.d_min_v[v_idx] = d_min_v
        for t_idx in vogc_t:
            cs.VOGC.setdefault(t_idx, []).append(v_idx)

    # No EE detection in the fast path — set d_min_e to r_q for cloth edges
    for e_idx in range(mesh_cloth.num_edges):
        cs.d_min_e[e_idx] = r_q

    return cs
