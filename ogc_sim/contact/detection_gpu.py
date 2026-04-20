"""
GPU contact detection — cloth vertices vs static obstacle.

Strategy: brute-force all-pairs computation on GPU.
For N_cloth=3957 and N_obs_tri~1000 this is ~4M distance computations
executed in a single batched PyTorch kernel — takes milliseconds on an
RTX 6000 vs seconds in the Python loop.

The Gauss Map feasibility gate is replaced by a simple distance threshold
in the GPU path (Gauss Map requires per-feature graph traversal which is
inherently sequential and rarely changes the result for cloth-body contacts).
"""

from __future__ import annotations

import numpy as np
import torch

from ogc_sim.geometry.mesh import Mesh
from ogc_sim.contact.detection import ContactSets
from ogc_sim.contact.distance_gpu import batch_point_triangle_distance


def detect_contacts_gpu(
    V_cloth: np.ndarray,    # (N_cloth, 3)  current cloth positions
    mesh_obs: Mesh,         # static obstacle mesh
    r: float,
    r_q: float,
    device: torch.device,
    chunk_size: int = 4096,
) -> ContactSets:
    """
    GPU all-pairs contact detection: each cloth vertex vs every obstacle triangle.

    Processes cloth vertices in chunks so the peak VRAM usage is bounded:
        chunk_size * N_obs_tri * 4 tensors * 3 floats * 8 bytes  (float64)

    For chunk_size=4096 and N_obs_tri=1000 that is ~400 MB — well within
    the RTX 6000's 48 GB budget.

    Parameters
    ----------
    V_cloth   : (N_cloth, 3) cloth vertex positions (numpy, CPU)
    mesh_obs  : static obstacle mesh
    r         : contact radius
    r_q       : query radius  (d_min initialized to r_q)
    device    : torch device ('cuda' or 'cpu')
    chunk_size: how many cloth vertices to process per GPU batch

    Returns
    -------
    ContactSets with FOGC, VOGC, d_min_v, d_min_t populated.
    EOGC and d_min_e are left empty (no EE detection in GPU path for now).
    """
    N_cloth  = len(V_cloth)
    N_obs_t  = mesh_obs.num_triangles

    dtype = torch.float64

    # Move obstacle mesh to GPU tensors (done once — obstacle is static)
    V_obs_g = torch.tensor(mesh_obs.V, dtype=dtype, device=device)  # (No, 3)
    T_obs_g = torch.tensor(mesh_obs.T, dtype=torch.long, device=device)  # (M, 3)

    # Obstacle triangle vertices: (M, 3) each
    A_obs = V_obs_g[T_obs_g[:, 0]]   # (M, 3)
    B_obs = V_obs_g[T_obs_g[:, 1]]
    C_obs = V_obs_g[T_obs_g[:, 2]]

    # Move cloth positions to GPU
    V_cloth_g = torch.tensor(V_cloth, dtype=dtype, device=device)  # (N_cloth, 3)

    # Accumulators — kept on CPU to build ContactSets
    d_min_v   = np.full(N_cloth, r_q, dtype=np.float64)
    d_min_t   = np.full(N_obs_t, r_q, dtype=np.float64)
    fogc      = [[] for _ in range(N_cloth)]
    vogc      = [[] for _ in range(N_obs_t)]

    # Process cloth vertices in chunks
    for start in range(0, N_cloth, chunk_size):
        end  = min(start + chunk_size, N_cloth)
        K    = end - start                                # chunk size

        # P: (K, M, 3) — expand cloth chunk × obs triangles
        P_chunk = V_cloth_g[start:end]                   # (K, 3)
        P_exp   = P_chunk.unsqueeze(1).expand(-1, N_obs_t, -1)  # (K, M, 3)
        A_exp   = A_obs.unsqueeze(0).expand(K, -1, -1)
        B_exp   = B_obs.unsqueeze(0).expand(K, -1, -1)
        C_exp   = C_obs.unsqueeze(0).expand(K, -1, -1)

        # Flatten to (K*M, 3) for batch distance
        P_flat  = P_exp.reshape(-1, 3)
        A_flat  = A_exp.reshape(-1, 3)
        B_flat  = B_exp.reshape(-1, 3)
        C_flat  = C_exp.reshape(-1, 3)

        dist_flat, _ = batch_point_triangle_distance(P_flat, A_flat, B_flat, C_flat)
        dist_km = dist_flat.reshape(K, N_obs_t)          # (K, M)

        # Move to CPU for ContactSets bookkeeping
        dist_np = dist_km.cpu().numpy()

        for ki in range(K):
            v_idx = start + ki
            row   = dist_np[ki]                          # (M,)

            dmin = float(row.min()) if N_obs_t > 0 else r_q
            d_min_v[v_idx] = min(dmin, r_q)

            # Contacts: all triangles within radius r
            contact_mask = row < r
            for t_idx in np.where(contact_mask)[0]:
                t_idx = int(t_idx)
                fogc[v_idx].append(t_idx)
                vogc[t_idx].append(v_idx)

            # d_min_t: atomic min across cloth vertices
            np.minimum(d_min_t, row, out=d_min_t)

    # Build ContactSets
    cs = ContactSets()
    for v_idx in range(N_cloth):
        cs.FOGC[v_idx]    = fogc[v_idx]
        cs.d_min_v[v_idx] = d_min_v[v_idx]
    for t_idx in range(N_obs_t):
        cs.VOGC[t_idx]    = vogc[t_idx]
        cs.d_min_t[t_idx] = float(d_min_t[t_idx])
    # No EE detection in GPU fast path
    return cs
