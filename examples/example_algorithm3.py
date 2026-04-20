"""
Example: Algorithm 3 — Simulation Step (Outer Time-Step Loop)
=============================================================
Run:
    python3 examples/example_algorithm3.py

Demonstrates how to use the Algorithm 3 module to simulate
a cloth triangle falling under gravity toward a floor triangle.

Based on explore/m2/learn_algorithm3.py — see that file for
the step-by-step educational walkthrough.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ogc_sim.geometry.mesh import Mesh
from ogc_sim.geometry.bvh import BVH
from ogc_sim.geometry.gauss_map import PolyhedralGaussMap
from ogc_sim.contact.detection import run_contact_detection
from ogc_sim.contact.energy import contact_gradient_v_vf
from ogc_sim.contact.bounds import (
    compute_conservative_bounds,
    truncate_displacements,
    apply_initial_guess_truncation,
)
from ogc_sim.algorithms.algorithm3 import simulation_step, StepResult
from ogc_sim.solver.vbd import graph_color_mesh, compute_rest_lengths


def pause(title: str) -> None:
    plt.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.show()


def draw_mesh(ax, V, T, face_color="#aed6f1", edge_color="#2980b9", alpha=0.35):
    tris = [[V[T[i, 0]], V[T[i, 1]], V[T[i, 2]]] for i in range(len(T))]
    ax.add_collection3d(Poly3DCollection(
        tris, alpha=alpha, facecolor=face_color, edgecolor=edge_color, lw=1.0
    ))
    for v in V:
        ax.scatter(*v, color=edge_color, s=25, zorder=6)


# ============================================================
# Scene setup
# ============================================================

V_cloth = np.array([
    [0.0, 0.0, 1.0],
    [2.0, 0.0, 1.0],
    [1.0, 2.0, 1.0],
])
T_cloth = np.array([[0, 1, 2]])

V_floor = np.array([
    [-1.0, -1.0, 0.0],
    [ 3.0, -1.0, 0.0],
    [ 1.0,  3.0, 0.0],
])
T_floor = np.array([[0, 1, 2]])

# Build combined mesh for rest lengths and coloring
T_all = np.array([[0, 1, 2], [3, 4, 5]])
V_combined = np.vstack([V_cloth, V_floor])
mesh_rest = Mesh.from_arrays(V_combined, T_all)
mesh_cloth = Mesh.from_arrays(V_cloth, T_cloth)
l0 = compute_rest_lengths(mesh_rest)
colors = graph_color_mesh(mesh_cloth)


# ============================================================
# EXAMPLE 1 — The inertia target Y
# ============================================================

print("=" * 60)
print("Example 1 — The inertia target Y (Algorithm 3, line 3)")
print("=" * 60)

X_t = V_cloth.copy()
v_t = np.zeros_like(X_t)
a_ext = np.array([0.0, 0.0, -9.8])
dt = 0.02

Y = X_t + dt * v_t + dt**2 * a_ext

print(f"\n  dt = {dt},  a_ext = {a_ext}")
print(f"\n  X_t (current positions):")
for i in range(3):
    print(f"    V{i} = {X_t[i]}")
print(f"\n  Y = X_t + dt*v_t + dt^2*a_ext  (free-fall prediction):")
for i in range(3):
    print(f"    V{i} -> {np.round(Y[i], 4)}")

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
draw_mesh(ax, V_floor, T_floor, face_color="#f9e4b7", edge_color="#c0a060")
draw_mesh(ax, X_t, T_cloth, face_color="#aed6f1", edge_color="#2980b9")
draw_mesh(ax, Y, T_cloth, face_color="#a9dfbf", edge_color="#1a8a50", alpha=0.2)

for i, (xt, y) in enumerate(zip(X_t, Y)):
    ax.plot([xt[0], y[0]], [xt[1], y[1]], [xt[2], y[2]],
            color="gray", lw=1.0, linestyle="--")
    ax.text(*xt + np.array([0.04, 0.04, 0.06]),
            f"V{i} X_t", fontsize=8, color="#2980b9")
    ax.text(*y + np.array([0.04, 0.04, -0.1]),
            f"V{i} Y", fontsize=8, color="#1a8a50")

ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_xlim(-0.5, 2.5); ax.set_ylim(-0.5, 2.5); ax.set_zlim(-0.2, 1.4)
ax.view_init(elev=28, azim=-55)
pause("Example 1 — Inertia target Y\n(blue=X_t, green=Y, orange=floor)")


# ============================================================
# EXAMPLE 2 — Single simulation step
# ============================================================

print("\n" + "=" * 60)
print("Example 2 — Single simulation step")
print("=" * 60)

r = 0.35
r_q = 0.50
gamma_p = 0.45
gamma_e = 0.0
n_iter = 5
mass = 1.0
k_s = 200.0
k_c = 500.0

# Start the cloth closer to the floor so contact actually happens
X_t_close = V_cloth.copy()
X_t_close[:, 2] = 0.3
v_t_close = np.tile([0.0, 0.0, -1.0], (3, 1))

result = simulation_step(
    X_t=X_t_close,
    v_t=v_t_close,
    V_floor=V_floor,
    T_cloth=T_cloth,
    T_floor=T_floor,
    colors=colors,
    l0=l0,
    dt=dt,
    a_ext=a_ext,
    r=r,
    r_q=r_q,
    gamma_p=gamma_p,
    gamma_e=gamma_e,
    n_iter=n_iter,
    mass=mass,
    k_s=k_s,
    k_c=k_c,
)

print(f"\n  Parameters: dt={dt}, r={r}, k_c={k_c}, n_iter={n_iter}")
print(f"  Cloth starting z = {X_t_close[0, 2]:.3f}")
print(f"\n  After one simulation step:")
print(f"    Contact detections: {result.num_detections}")
for i in range(3):
    print(f"    V{i}: z = {X_t_close[i, 2]:.3f} -> {result.X[i, 2]:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(12, 5))

draw_mesh(ax1, V_floor, T_floor, face_color="#f9e4b7", edge_color="#c0a060")
draw_mesh(ax1, X_t_close, T_cloth, face_color="#aed6f1", edge_color="#2980b9")
ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
ax1.set_xlim(-1.5, 3.5); ax1.set_ylim(-1.5, 3.5); ax1.set_zlim(-0.2, 0.8)
ax1.view_init(elev=28, azim=-55)
ax1.set_title("Before: cloth at z=0.3", fontsize=9)

draw_mesh(ax2, V_floor, T_floor, face_color="#f9e4b7", edge_color="#c0a060")
draw_mesh(ax2, result.X, T_cloth, face_color="#a9dfbf", edge_color="#1a8a50")
ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
ax2.set_xlim(-1.5, 3.5); ax2.set_ylim(-1.5, 3.5); ax2.set_zlim(-0.2, 0.8)
ax2.view_init(elev=28, azim=-55)
ax2.set_title(f"After: cloth at z={result.X[0, 2]:.3f}", fontsize=9)

pause("Example 2 — Single simulation step\n(blue=before, green=after)")


# ============================================================
# EXAMPLE 3 — Multi-step animation
# ============================================================

print("\n" + "=" * 60)
print("Example 3 — Multi-step animation")
print("=" * 60)

n_outer = 25
dt_anim = 0.04

X_cur = V_cloth.copy()
v_cur = np.tile([0.0, 0.0, -1.2], (3, 1))

print(f"\n  Simulating {n_outer} time steps (dt={dt_anim})")

all_frames = []
all_z = [X_cur[0, 2]]

for step in range(n_outer):
    result = simulation_step(
        X_t=X_cur,
        v_t=v_cur,
        V_floor=V_floor,
        T_cloth=T_cloth,
        T_floor=T_floor,
        colors=colors,
        l0=l0,
        dt=dt_anim,
        a_ext=a_ext,
        r=r,
        r_q=r_q,
        gamma_p=gamma_p,
        gamma_e=gamma_e,
        n_iter=5,
        mass=mass,
        k_s=k_s,
        k_c=k_c,
    )

    X_cur = result.X
    v_cur = result.v
    all_z.append(X_cur[0, 2])

    # Compute contact force arrows
    grad_arrows = []
    for v_idx in range(3):
        g = contact_gradient_v_vf(
            X_cur[v_idx], V_floor[0], V_floor[1], V_floor[2], r, k_c
        )
        if np.linalg.norm(g) > 1e-8:
            grad_arrows.append((v_idx, g))

    contact_active = len(grad_arrows) > 0

    all_frames.append({
        "X": X_cur.copy(),
        "contact_active": contact_active,
        "grad_arrows": grad_arrows,
        "step": step + 1,
    })

print(f"  Final cloth z: {X_cur[0, 2]:.4f}  (floor z=0, contact r={r})")

# Build animation
fig_anim = plt.figure(figsize=(14, 6))
fig_anim.patch.set_facecolor("#1a1a2e")

ax3d = fig_anim.add_subplot(1, 2, 1, projection="3d")
ax3d.set_facecolor("#1a1a2e")

ax_z = fig_anim.add_subplot(1, 2, 2)
ax_z.set_facecolor("#0f0f1e")
ax_z.tick_params(colors="#aaaacc")
for spine in ax_z.spines.values():
    spine.set_edgecolor("#555577")

ax_z.axhline(y=r, color="orange", lw=1.5, linestyle=":", label=f"contact r={r}")
ax_z.axhline(y=0.0, color="#888899", lw=1.0, linestyle="-", label="floor z=0")
ax_z.set_xlabel("Time step", fontsize=9, color="#aaaacc")
ax_z.set_ylabel("V0 z-position", fontsize=9, color="#aaaacc")
ax_z.set_xlim(0, n_outer + 1)
ax_z.set_ylim(-0.1, all_z[0] + 0.1)
ax_z.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#555577", labelcolor="#ccccdd")

(curve_line,) = ax_z.plot([], [], color="#5dade2", lw=1.5)
(cur_dot,) = ax_z.plot([], [], "o", color="#f5cba7", ms=6, zorder=5)


def _setup_3d():
    ax3d.set_xlim(-1.5, 3.5); ax3d.set_ylim(-1.5, 3.5); ax3d.set_zlim(-0.3, 1.3)
    ax3d.set_xlabel("X", color="#aaaacc", fontsize=8)
    ax3d.set_ylabel("Y", color="#aaaacc", fontsize=8)
    ax3d.set_zlabel("Z", color="#aaaacc", fontsize=8)
    ax3d.tick_params(colors="#888899")
    ax3d.view_init(elev=28, azim=-55)


def _update(frame_num):
    fd = all_frames[frame_num]
    X = fd["X"]
    contact_active = fd["contact_active"]

    ax3d.cla()
    _setup_3d()
    draw_mesh(ax3d, V_floor, T_floor, face_color="#f9e4b7", edge_color="#c0a060", alpha=0.6)

    cloth_fc = "#e74c3c" if contact_active else "#5dade2"
    cloth_ec = "#c0392b" if contact_active else "#2980b9"
    draw_mesh(ax3d, X, T_cloth, face_color=cloth_fc, edge_color=cloth_ec, alpha=0.75)

    for v_idx, grad in fd["grad_arrows"]:
        force = -grad
        scale = 0.15 / (np.linalg.norm(force) + 1e-12)
        ax3d.quiver(*X[v_idx], *(force * scale), color="#f39c12",
                    linewidth=2.0, arrow_length_ratio=0.4)

    for v_idx in range(3):
        ax3d.text(*(X[v_idx] + np.array([0.06, 0.06, 0.06])),
                  f"V{v_idx} z={X[v_idx, 2]:.3f}", fontsize=7, color="#ecf0f1")

    state = " [IN CONTACT]" if contact_active else ""
    ax3d.set_title(
        f"Algorithm 3 step {fd['step']}{state}",
        fontsize=9, color="#ecf0f1", pad=6
    )

    z_so_far = all_z[:frame_num + 2]
    curve_line.set_data(range(len(z_so_far)), z_so_far)
    cur_dot.set_data([len(z_so_far) - 1], [z_so_far[-1]])
    ax_z.set_title(f"V0 z-position (step {fd['step']})",
                   fontsize=9, color="#aaaacc")

    return [curve_line, cur_dot]


ani = animation.FuncAnimation(
    fig_anim, _update, frames=len(all_frames),
    interval=150, blit=False, repeat=True,
)

fig_anim.suptitle(
    "Example 3 — Algorithm 3 animated over multiple time steps\n"
    "Close window to finish",
    fontsize=10, color="#ecf0f1", y=1.01
)
plt.tight_layout()
plt.show()

print("""
  Summary:
    Algorithm 3 is the outer time-step loop that orchestrates:
      1. Inertia target Y = X_t + dt*v + dt^2*a_ext       (line 3)
      2. Contact detection (Algorithms 1 & 2) when needed  (lines 5-16)
      3. Conservative bounds b_v = gamma_p * d_min         (lines 17-19)
      4. Initial guess truncation                          (lines 20-21)
      5. Inner solver iteration (Algorithm 4)              (line 22)
      6. Displacement truncation and re-detection trigger  (lines 23-29)
""")
