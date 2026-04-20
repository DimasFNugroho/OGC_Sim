"""
Batch point-triangle distance and contact energy derivatives on GPU.

All operations are fully vectorized in PyTorch — no Python loops over
vertices or triangles.  Works on both CUDA and CPU tensors.

Reference: ogc_sim/geometry/distance.py (CPU equivalent)
           ogc_sim/contact/energy.py (scalar activation functions)
"""

from __future__ import annotations

import torch


# ======================================================================
# Batch point-triangle distance
# ======================================================================

def batch_point_triangle_distance(
    P: torch.Tensor,   # (B, 3)
    A: torch.Tensor,   # (B, 3)
    B: torch.Tensor,   # (B, 3)
    C: torch.Tensor,   # (B, 3)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Closest-point distance from B query points to B triangles.

    Uses the Ericson (2005) real-time collision detection method —
    fully vectorized with no Python loops.

    Parameters
    ----------
    P : (B, 3)  query points
    A, B, C : (B, 3)  triangle vertices (CCW winding)

    Returns
    -------
    dist : (B,)   non-negative distances
    cp   : (B, 3) closest points on the triangles
    """
    AB = B - A          # (B, 3)
    AC = C - A
    AP = P - A

    d1 = (AB * AP).sum(-1)      # (B,)
    d2 = (AC * AP).sum(-1)

    BP = P - B
    d3 = (AB * BP).sum(-1)
    d4 = (AC * BP).sum(-1)

    CP = P - C
    d5 = (AB * CP).sum(-1)
    d6 = (AC * CP).sum(-1)

    # Region masks
    mask_A  = (d1 <= 0) & (d2 <= 0)
    mask_B  = (d3 >= 0) & (d4 <= d3)
    mask_C  = (d6 >= 0) & (d5 <= d6)

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    mask_AB = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    mask_AC = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    mask_BC = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)

    # Interior barycentric coordinates
    denom = (va + vb + vc).clamp(min=1e-12)
    v_bary = vb / denom
    w_bary = vc / denom

    # Closest points — start with interior and overwrite with boundary cases
    cp = A + v_bary.unsqueeze(1) * AB + w_bary.unsqueeze(1) * AC

    t_AB = (d1 / (d1 - d3).clamp(min=1e-12)).clamp(0, 1)
    cp = torch.where(mask_AB.unsqueeze(1), A + t_AB.unsqueeze(1) * AB, cp)

    t_AC = (d2 / (d2 - d6).clamp(min=1e-12)).clamp(0, 1)
    cp = torch.where(mask_AC.unsqueeze(1), A + t_AC.unsqueeze(1) * AC, cp)

    t_BC = ((d4 - d3) / ((d4 - d3) + (d5 - d6)).clamp(min=1e-12)).clamp(0, 1)
    cp = torch.where(mask_BC.unsqueeze(1), B + t_BC.unsqueeze(1) * (C - B), cp)

    cp = torch.where(mask_A.unsqueeze(1), A, cp)
    cp = torch.where(mask_B.unsqueeze(1), B, cp)
    cp = torch.where(mask_C.unsqueeze(1), C, cp)

    dist = torch.norm(P - cp, dim=1)
    return dist, cp


# ======================================================================
# Batch activation function and its derivatives
# ======================================================================

def _stitch_params(r: float, k_c: float, device, dtype):
    tau       = r / 2.0
    k_prime   = k_c * tau ** 2
    b         = 0.5 * k_c * (r - tau) ** 2 + k_prime * torch.tensor(tau, dtype=dtype).log().item()
    return tau, k_prime, b


def batch_activation_g(d: torch.Tensor, r: float, k_c: float) -> torch.Tensor:
    """g(d, r)  — contact activation energy  (Eq. 18). Batch version."""
    tau, k_prime, b = _stitch_params(r, k_c, d.device, d.dtype)
    out = torch.zeros_like(d)
    quad = (d >= tau) & (d < r)
    log_ = (d > 0) & (d < tau)
    out = torch.where(quad, 0.5 * k_c * (r - d) ** 2, out)
    out = torch.where(log_, -k_prime * d.clamp(min=1e-12).log() + b, out)
    return out


def batch_dg_dd(d: torch.Tensor, r: float, k_c: float) -> torch.Tensor:
    """dg/dd  — first derivative of activation energy."""
    tau, k_prime, _ = _stitch_params(r, k_c, d.device, d.dtype)
    out = torch.zeros_like(d)
    quad = (d >= tau) & (d < r)
    log_ = (d > 0) & (d < tau)
    out = torch.where(quad, -k_c * (r - d), out)
    out = torch.where(log_, -k_prime / d.clamp(min=1e-12), out)
    return out


def batch_d2g_dd2(d: torch.Tensor, r: float, k_c: float) -> torch.Tensor:
    """d²g/dd²  — second derivative of activation energy."""
    tau, k_prime, _ = _stitch_params(r, k_c, d.device, d.dtype)
    out = torch.zeros_like(d)
    quad = (d >= tau) & (d < r)
    log_ = (d > 0) & (d < tau)
    out = torch.where(quad, torch.full_like(d, k_c), out)
    out = torch.where(log_, k_prime / d.clamp(min=1e-12) ** 2, out)
    return out


# ======================================================================
# Batch VF contact gradient and Hessian (vertex side)
# ======================================================================

def batch_contact_grad_hessian_vf(
    P:   torch.Tensor,   # (C, 3)  query vertex positions
    A:   torch.Tensor,   # (C, 3)  triangle vertex A
    B_t: torch.Tensor,   # (C, 3)  triangle vertex B
    C_t: torch.Tensor,   # (C, 3)  triangle vertex C
    r:   float,
    k_c: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batch gradient and Hessian of VF contact energy w.r.t. each query vertex.

    Parameters
    ----------
    P, A, B_t, C_t : (C, 3)
        One row per contact pair.
    r, k_c : contact radius and stiffness.

    Returns
    -------
    grad : (C, 3)   ∂E/∂x_v  for each pair (zero if dist >= r)
    H    : (C, 3, 3) ∂²E/∂x_v² for each pair
    """
    device, dtype = P.device, P.dtype
    C_n = P.shape[0]

    dist, cp = batch_point_triangle_distance(P, A, B_t, C_t)   # (C,), (C, 3)

    valid  = (dist < r) & (dist > 1e-12)                        # (C,)
    dg     = batch_dg_dd(dist, r, k_c)                          # (C,)
    d2g    = batch_d2g_dd2(dist, r, k_c)                        # (C,)

    n   = (P - cp) / dist.clamp(min=1e-12).unsqueeze(1)        # (C, 3)
    nnt = torch.bmm(n.unsqueeze(2), n.unsqueeze(1))             # (C, 3, 3)
    I3  = torch.eye(3, device=device, dtype=dtype)\
              .unsqueeze(0).expand(C_n, -1, -1)                 # (C, 3, 3)

    # ∂E/∂x_v = (dg/dd) * n
    grad = dg.unsqueeze(1) * n                                  # (C, 3)

    # ∂²E/∂x_v² = d²g/dd² * n⊗n  +  (dg/dd)/d * (I - n⊗n)
    H = (d2g.unsqueeze(-1).unsqueeze(-1) * nnt
         + (dg / dist.clamp(min=1e-12)).unsqueeze(-1).unsqueeze(-1) * (I3 - nnt))

    # Zero out inactive contacts
    vmask = valid.unsqueeze(1)
    grad = grad * vmask.float()
    H    = H    * vmask.unsqueeze(2).float()

    return grad, H
