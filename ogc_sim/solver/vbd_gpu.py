"""
GPU-accelerated VBD iteration — Algorithm 4.

Replaces the Python-loop CPU version in ogc_sim/solver/vbd.py with
fully batched PyTorch operations.  The main speedups are:

  1. Spring forces   — one scatter_add pass over all M edges (no per-vertex loop)
  2. Spring Hessians — same scatter pass, outer-products broadcast
  3. Contact forces  — batch point-triangle distance + gradient per color group
  4. Newton step     — torch.linalg.solve on all K color-group vertices at once

VBD is Gauss-Seidel: color groups are sequential, but within each group
all K vertices are updated in parallel (the point of graph coloring).

Usage
-----
    from ogc_sim.solver.vbd_gpu import vbd_iteration_gpu, build_gpu_mesh_data
    gpu_data = build_gpu_mesh_data(mesh, l0, device)   # pre-compute once
    X = vbd_iteration_gpu(X, X_t, Y, mesh, cs, colors, gpu_data,
                          dt, mass, k_s, r, k_c, n_dof, device)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from ogc_sim.geometry.mesh import Mesh
from ogc_sim.contact.detection import ContactSets
from ogc_sim.contact.distance_gpu import (
    batch_point_triangle_distance,
    batch_contact_grad_hessian_vf,
)


# ======================================================================
# Pre-computed GPU mesh data (build once, reuse every time step)
# ======================================================================

@dataclass
class GPUMeshData:
    """
    Topology and rest-state tensors moved to GPU once.

    Only needs to be rebuilt if the mesh topology changes (it doesn't
    during simulation — only vertex positions change).
    """
    edge_src:  torch.Tensor    # (M,)  source vertex index for each edge
    edge_dst:  torch.Tensor    # (M,)  destination vertex index for each edge
    l0:        torch.Tensor    # (M,)  rest edge lengths
    T:         torch.Tensor    # (F, 3) triangle vertex indices
    # Per-vertex incident edge lists kept as Python lists (variable length)
    E_v:       list[list[int]]
    T_v:       list[list[int]]
    E_t:       list[list[int]]
    device:    torch.device
    dtype:     torch.dtype


def build_gpu_mesh_data(
    mesh: Mesh,
    l0: np.ndarray,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> GPUMeshData:
    """
    Move mesh topology and rest lengths to GPU tensors.
    Call once per mesh load; reuse across all time steps.
    """
    E = torch.tensor(mesh.E, dtype=torch.long, device=device)
    return GPUMeshData(
        edge_src = E[:, 0],
        edge_dst = E[:, 1],
        l0       = torch.tensor(l0, dtype=dtype, device=device),
        T        = torch.tensor(mesh.T, dtype=torch.long, device=device),
        E_v      = mesh.E_v,
        T_v      = mesh.T_v,
        E_t      = mesh.E_t,
        device   = device,
        dtype    = dtype,
    )


# ======================================================================
# GPU VBD iteration
# ======================================================================

def vbd_iteration_gpu(
    X:      np.ndarray,         # (N, 3) current positions — updated in place
    X_t:    np.ndarray,         # (N, 3) start-of-step positions
    Y:      np.ndarray,         # (N, 3) inertia target
    mesh:   Mesh,               # full combined mesh (cloth + obstacle)
    cs:     ContactSets,
    colors: list[list[int]],
    gd:     GPUMeshData,        # pre-computed GPU data
    dt:     float,
    mass:   float,
    k_s:    float,
    r:      float,
    k_c:    float,
    n_dof:  Optional[int] = None,
) -> np.ndarray:
    """
    One full VBD pass on GPU (Algorithm 4).

    Updates X in-place (same semantics as the CPU version).

    Parameters
    ----------
    X, X_t, Y : (N, 3) numpy arrays — cloth positions come first (0..n_dof-1)
    mesh       : full combined mesh (cloth + obstacle)
    cs         : ContactSets from most recent detection
    colors     : graph coloring from graph_color_mesh()
    gd         : GPUMeshData from build_gpu_mesh_data()
    dt         : time step
    mass       : per-vertex mass (scalar)
    k_s        : spring stiffness
    r, k_c     : contact radius and stiffness
    n_dof      : only update vertices 0..n_dof-1 (cloth); rest are static

    Returns
    -------
    X (same array, updated in-place)
    """
    device = gd.device
    dtype  = gd.dtype
    n_dof_actual = n_dof if n_dof is not None else mesh.num_vertices
    h2  = dt * dt

    # Move all vertex data to GPU for this iteration
    Xg  = torch.tensor(X,   dtype=dtype, device=device)   # (N, 3)
    Yg  = torch.tensor(Y,   dtype=dtype, device=device)   # (N, 3)
    I3  = torch.eye(3, dtype=dtype, device=device)

    for color_group in colors:
        # Filter to cloth-only vertices that need updating
        vg = [v for v in color_group if v < n_dof_actual]
        if not vg:
            continue

        K    = len(vg)
        vg_t = torch.tensor(vg, dtype=torch.long, device=device)   # (K,)

        xv  = Xg[vg_t]   # (K, 3)
        yv  = Yg[vg_t]   # (K, 3)

        # ---- 1. Inertia ----
        mh2 = mass / h2
        f   = -mh2 * (xv - yv)                            # (K, 3)
        H   = mh2 * I3.unsqueeze(0).expand(K, -1, -1).clone()  # (K, 3, 3)

        # ---- 2. Spring forces and Hessians (scatter over incident edges) ----
        f_s, H_s = _batch_spring_force_hessian(Xg, vg, gd, k_s, device, dtype)
        f += f_s
        H += H_s

        # ---- 3. Contact forces (FOGC vertex side) ----
        f_c, H_c = _batch_contact_forces(Xg, vg, vg_t, mesh, cs, gd, r, k_c, device, dtype)
        f += f_c
        H += H_c

        # ---- 4. Newton step (batch 3×3 solve) ----
        # Clamp Hessian to PSD by projecting negative eigenvalues
        eigvals, eigvecs = torch.linalg.eigh(H)           # (K, 3), (K, 3, 3)
        eigvals = eigvals.clamp(min=1e-9)
        H_psd   = torch.bmm(eigvecs,
                    torch.bmm(torch.diag_embed(eigvals),
                              eigvecs.transpose(-1, -2)))  # (K, 3, 3)

        # Solve K systems of size 3×3 simultaneously
        delta = torch.linalg.solve(H_psd, f.unsqueeze(-1)).squeeze(-1)  # (K, 3)

        # ---- 5. Update positions ----
        Xg = Xg.clone()
        Xg[vg_t] = xv + delta

    # Write updated cloth positions back to numpy
    X[:n_dof_actual] = Xg[:n_dof_actual].cpu().numpy()
    return X


# ======================================================================
# Helpers
# ======================================================================

def _batch_spring_force_hessian(
    Xg:     torch.Tensor,      # (N, 3) all current positions on GPU
    vg:     list[int],         # K cloth vertex indices in this color group
    gd:     GPUMeshData,
    k_s:    float,
    device: torch.device,
    dtype:  torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batch spring force and Hessian for a color group.

    Builds a flat list of (group_slot k, edge_index ei, role) triplets,
    computes all spring contributions in one vectorized pass, then
    scatter-reduces to per-vertex (K) accumulators.
    """
    K  = len(vg)
    I3 = torch.eye(3, dtype=dtype, device=device)

    # Build flat pair lists
    k_list, e_list, role_list = [], [], []
    E_raw = gd.edge_src.cpu().numpy()  # faster to access numpy for indexing
    E_dst_raw = gd.edge_dst.cpu().numpy()

    for k, v in enumerate(vg):
        for ei in gd.E_v[v]:
            k_list.append(k)
            e_list.append(ei)
            # role 0: v is src end of edge, role 1: v is dst end
            role_list.append(0 if int(E_raw[ei]) == v else 1)

    if not k_list:
        return (torch.zeros(K, 3, dtype=dtype, device=device),
                torch.zeros(K, 3, 3, dtype=dtype, device=device))

    k_t    = torch.tensor(k_list, dtype=torch.long, device=device)   # (P,)
    e_t    = torch.tensor(e_list, dtype=torch.long, device=device)   # (P,)
    role_t = torch.tensor(role_list, dtype=torch.long, device=device) # (P,)

    # Positions of both edge endpoints for each pair
    src_idx = gd.edge_src[e_t]    # (P,)
    dst_idx = gd.edge_dst[e_t]    # (P,)
    x_src   = Xg[src_idx]         # (P, 3)
    x_dst   = Xg[dst_idx]         # (P, 3)

    # Diff from "other" end toward v
    # role=0 → v is src → diff = x_src - x_dst
    # role=1 → v is dst → diff = x_dst - x_src
    sign = (1 - 2 * role_t).float().unsqueeze(1)   # +1 or -1, (P, 1)
    diff = sign * (x_src - x_dst)                   # (P, 3)

    l    = diff.norm(dim=1, keepdim=True).clamp(min=1e-12)  # (P, 1)
    d    = diff / l                                  # (P, 3) unit direction

    l0_p = gd.l0[e_t]                               # (P,) rest lengths
    stretch = l.squeeze(1) - l0_p                   # (P,)

    # Spring force contribution: f_v += -k_s * stretch * d
    f_contrib = -k_s * stretch.unsqueeze(1) * d     # (P, 3)

    f_s = torch.zeros(K, 3, dtype=dtype, device=device)
    f_s.scatter_add_(0, k_t.unsqueeze(1).expand(-1, 3), f_contrib)

    # Spring Hessian contribution: k_s*d⊗d + k_s*(1-l0/l)*(I-d⊗d)
    l0_ratio  = (l0_p / l.squeeze(1)).unsqueeze(-1).unsqueeze(-1)  # (P,1,1)
    ddt       = torch.bmm(d.unsqueeze(2), d.unsqueeze(1))          # (P,3,3)
    I3_exp    = I3.unsqueeze(0).expand(len(d), -1, -1)             # (P,3,3)
    H_contrib = k_s * ddt + k_s * (1 - l0_ratio) * (I3_exp - ddt) # (P,3,3)

    H_s = torch.zeros(K, 3, 3, dtype=dtype, device=device)
    H_s.scatter_add_(0,
        k_t.unsqueeze(1).unsqueeze(2).expand(-1, 3, 3),
        H_contrib)

    return f_s, H_s


def _batch_contact_forces(
    Xg:     torch.Tensor,      # (N, 3) all positions on GPU
    vg:     list[int],         # K vertex indices
    vg_t:   torch.Tensor,      # (K,) same as vg on GPU
    mesh:   Mesh,
    cs:     ContactSets,
    gd:     GPUMeshData,
    r:      float,
    k_c:    float,
    device: torch.device,
    dtype:  torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batch VF contact gradient and Hessian for a color group (vertex side).

    Finds all (v, triangle) contact pairs for vertices in this group,
    batches the distance + gradient computation, and scatter-reduces to
    per-vertex (K) accumulators.
    """
    K = len(vg)
    f_c = torch.zeros(K, 3, dtype=dtype, device=device)
    H_c = torch.zeros(K, 3, 3, dtype=dtype, device=device)

    # Build contact pair lists for this color group
    k_list, t_list = [], []
    for k, v in enumerate(vg):
        for a_feat in cs.FOGC.get(v, []):
            t_idx = _find_triangle_for_feature(a_feat, mesh)
            if t_idx >= 0:
                k_list.append(k)
                t_list.append(t_idx)

    if not k_list:
        return f_c, H_c

    k_t = torch.tensor(k_list, dtype=torch.long, device=device)  # (C,)

    # Gather vertex and triangle positions
    v_indices = vg_t[k_t]                                    # (C,)
    P_c  = Xg[v_indices]                                     # (C, 3)
    T_c  = gd.T[torch.tensor(t_list, dtype=torch.long, device=device)]  # (C, 3)
    A_c  = Xg[T_c[:, 0]]
    B_c  = Xg[T_c[:, 1]]
    C_c  = Xg[T_c[:, 2]]

    grad, H_vf = batch_contact_grad_hessian_vf(P_c, A_c, B_c, C_c, r, k_c)  # (C,3),(C,3,3)

    # Accumulate: force = -gradient
    f_c.scatter_add_(0, k_t.unsqueeze(1).expand(-1, 3), -grad)
    H_c.scatter_add_(0, k_t.unsqueeze(1).unsqueeze(2).expand(-1, 3, 3), H_vf)

    return f_c, H_c


def _find_triangle_for_feature(feat_idx: int, mesh: Mesh) -> int:
    """Map a global FOGC feature index to a triangle index (mirrors CPU version)."""
    if 0 <= feat_idx < mesh.num_triangles:
        return feat_idx
    if 0 <= feat_idx < mesh.num_vertices:
        tv = mesh.T_v[feat_idx]
        return tv[0] if tv else -1
    if 0 <= feat_idx < mesh.num_edges:
        a, b = int(mesh.E[feat_idx][0]), int(mesh.E[feat_idx][1])
        for t_idx in mesh.T_v[a]:
            if b in mesh.T[t_idx]:
                return t_idx
    return -1
