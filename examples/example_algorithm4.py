"""
Example: Algorithm 4 — VBD Iteration with Contact (Inner Solver)
================================================================
Run:
    python3 examples/example_algorithm4.py

Demonstrates how to use the Algorithm 4 module (VBD iteration)
to solve the per-vertex Newton steps with contact forces.

Based on explore/m2/learn_algorithm4.py — see that file for
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
from ogc_sim.contact.energy import (
    contact_energy_vf, contact_gradient_v_vf, contact_hessian_v_vf,
)
from ogc_sim.contact.bounds import (
    compute_conservative_bounds,
    truncate_displacements,
    apply_initial_guess_truncation,
)
from ogc_sim.algorithms.algorithm4 import vbd_iteration
from ogc_sim.solver.vbd import (
    graph_color_mesh,
    compute_rest_lengths,
    spring_force_hessian,
)


def pause(title: str) -> None:
    plt.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.show()


def draw_mesh(ax, V, T, face_color="#aed6f1", edge_color="#2980b9", alpha=0.35):
    tris = [[V[T[i, 0]], V[T[i, 1]], V[T[i, 2]]] for i in range(len(T))]
    ax.add_collection3d(Poly3DCollection(
        tris, alpha=alpha, facecolor=face_color, edgecolor=edge_color, lw=1.2
    ))
    for v in V:
        ax.scatter(*v, color=edge_color, s=30, zorder=6)


# ============================================================
# Scene setup — 2-triangle cloth quad above a floor triangle
# ============================================================

V_cloth = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
], dtype=float)

T_cloth = np.array([[0, 1, 3], [0, 3, 2]])

V_floor = np.array([
    [-0.5, -0.5, 0.0],
    [ 1.5, -0.5, 0.0],
    [ 0.5,  1.5, 0.0],
], dtype=float)

T_floor = np.array([[0, 1, 2]])

V_all = np.vstack([V_cloth, V_floor])
T_all = np.vstack([T_cloth, T_floor + len(V_cloth)])

N_cloth = len(V_cloth)
r_val = 0.35
k_c_val = 500.0
k_s_val = 150.0
dt_val = 0.04
mass_v = 1.0
a_ext = np.array([0.0, 0.0, -9.8])

mesh_rest = Mesh.from_arrays(V_all, T_all)
l0 = compute_rest_lengths(mesh_rest)
colors = graph_color_mesh(Mesh.from_arrays(V_cloth, T_cloth))

COLOR_MAP = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6"]


# ============================================================
# EXAMPLE 1 — Graph coloring
# ============================================================

print("=" * 60)
print("Example 1 — Graph coloring for VBD")
print("=" * 60)

mesh_cloth = Mesh.from_arrays(V_cloth, T_cloth)
print(f"\n  Cloth mesh: {mesh_cloth.num_vertices} vertices, "
      f"{mesh_cloth.num_triangles} triangles, {mesh_cloth.num_edges} edges")
print(f"  Graph coloring: {len(colors)} colors")
for ci, group in enumerate(colors):
    print(f"    color {ci}: vertices {group}")
print(f"\n  Same-color vertices can be updated in parallel.")

fig = plt.figure(figsize=(10, 5))
ax_col = fig.add_subplot(1, 2, 1)
ax_3d = fig.add_subplot(1, 2, 2, projection="3d")

# 2D coloring diagram
positions_2d = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0), 3: (1.0, 1.0)}
for tri in T_cloth:
    for i in range(3):
        a, b = tri[i], tri[(i + 1) % 3]
        xa, ya = positions_2d[a]
        xb, yb = positions_2d[b]
        ax_col.plot([xa, xb], [ya, yb], color="#888888", lw=1.5, zorder=1)

v_color_map = {}
for ci, group in enumerate(colors):
    for v in group:
        v_color_map[v] = ci

for v_idx in range(4):
    x2, y2 = positions_2d[v_idx]
    ci = v_color_map[v_idx]
    col = COLOR_MAP[ci]
    ax_col.scatter(x2, y2, color=col, s=250, zorder=5)
    ax_col.text(x2 + 0.06, y2 + 0.04, f"V{v_idx}\ncolor {ci}", fontsize=9, color=col)

ax_col.set_xlim(-0.3, 1.5); ax_col.set_ylim(-0.3, 1.5)
ax_col.set_aspect("equal"); ax_col.axis("off")
ax_col.set_title("Graph coloring\n(same color = independent)", fontsize=9)

for ci, group in enumerate(colors):
    for v in group:
        ax_3d.scatter(*V_cloth[v], color=COLOR_MAP[ci], s=100, zorder=6)
        ax_3d.text(*(V_cloth[v] + np.array([0.04, 0.04, 0.04])),
                   f"V{v}", fontsize=8, color=COLOR_MAP[ci])
draw_mesh(ax_3d, V_cloth, T_cloth, face_color="#aed6f1", edge_color="#888", alpha=0.3)
ax_3d.set_xlabel("X"); ax_3d.set_ylabel("Y"); ax_3d.set_zlabel("Z")
ax_3d.set_xlim(-0.3, 1.5); ax_3d.set_ylim(-0.3, 1.5); ax_3d.set_zlim(0.8, 1.2)
ax_3d.view_init(elev=35, azim=-50)
ax_3d.set_title("3-D view", fontsize=9)
pause("Example 1 — Graph coloring (same-color vertices share no triangle)")


# ============================================================
# EXAMPLE 2 — Per-vertex force breakdown
# ============================================================

print("\n" + "=" * 60)
print("Example 2 — Per-vertex force breakdown")
print("=" * 60)

# Place V0 in the contact zone
X_demo = np.array([
    [0.0, 0.0, 0.20],   # V0: inside contact zone
    [1.0, 0.0, 0.60],
    [0.0, 1.0, 0.60],
    [1.0, 1.0, 0.60],
], dtype=float)
Y_demo = X_demo + dt_val * np.tile([0.0, 0.0, -1.5], (4, 1)) + dt_val**2 * a_ext

h2 = dt_val**2
m_h2 = mass_v / h2

V_all_demo = np.vstack([X_demo, V_floor])
mesh_demo = Mesh.from_arrays(V_all_demo, np.vstack([T_cloth, T_floor + N_cloth]))

focus_v = 0  # examine V0

# Inertia
f_in = -(m_h2) * (X_demo[focus_v] - Y_demo[focus_v])

# Elastic
f_el, H_el = spring_force_hessian(focus_v, V_all_demo, mesh_demo, l0, k_s_val)

# Contact
f_ct = np.zeros(3)
g = contact_gradient_v_vf(X_demo[focus_v],
                           V_floor[0], V_floor[1], V_floor[2],
                           r_val, k_c_val)
f_ct = -g  # force = -gradient

f_total = f_in + f_el + f_ct

print(f"\n  V{focus_v} at position {X_demo[focus_v]}:")
print(f"    f_inertia = {np.round(f_in, 4)}")
print(f"    f_elastic = {np.round(f_el, 4)}")
print(f"    f_contact = {np.round(f_ct, 4)}")
print(f"    f_total   = {np.round(f_total, 4)}")

# Bar chart + 3D arrows
fig, (ax_bar, ax_arr) = plt.subplots(1, 2, figsize=(12, 5))
categories = ["inertia", "elastic", "contact"]
magnitudes = [np.linalg.norm(f_in), np.linalg.norm(f_el), np.linalg.norm(f_ct)]
bars = ax_bar.bar(categories, magnitudes, color=["#3498db", "#2ecc71", "#e74c3c"])
ax_bar.set_ylabel("||force||", fontsize=10)
ax_bar.set_title(f"Force magnitudes on V{focus_v}", fontsize=10)
for bar, mag in zip(bars, magnitudes):
    ax_bar.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1, f"{mag:.2f}", ha="center", fontsize=9)

ax_arr = fig.add_subplot(1, 2, 2, projection="3d")
draw_mesh(ax_arr, V_floor, T_floor, face_color="#f9e4b7", edge_color="#c0a060", alpha=0.4)
draw_mesh(ax_arr, X_demo, T_cloth, face_color="#aed6f1", edge_color="#2980b9", alpha=0.5)

pos = X_demo[focus_v]
force_data = [("inertia", f_in, "#3498db"),
              ("elastic", f_el, "#2ecc71"),
              ("contact", f_ct, "#e74c3c"),
              ("TOTAL", f_total, "#f39c12")]

for label, fv, col in force_data:
    fn = np.linalg.norm(fv)
    if fn > 1e-6:
        scale = 0.3 / fn
        ax_arr.quiver(*pos, *(fv * scale), color=col,
                      linewidth=2.5, arrow_length_ratio=0.3, label=label)

ax_arr.scatter(*pos, color="black", s=60, zorder=7)
ax_arr.legend(fontsize=8, loc="upper right")
ax_arr.set_xlabel("X"); ax_arr.set_ylabel("Y"); ax_arr.set_zlabel("Z")
ax_arr.set_xlim(-0.5, 1.5); ax_arr.set_ylim(-0.5, 1.5); ax_arr.set_zlim(-0.05, 0.8)
ax_arr.view_init(elev=22, azim=-55)
ax_arr.set_title(f"Force arrows on V{focus_v}", fontsize=9)
pause("Example 2 — Per-vertex force breakdown\n"
      "(blue=inertia, green=elastic, red=contact, orange=total)")


# ============================================================
# EXAMPLE 3 — Full VBD animated
# ============================================================

print("\n" + "=" * 60)
print("Example 3 — Full VBD animated")
print("=" * 60)

r_anim = 0.35
r_q_anim = 0.50
gamma_p = 0.45
n_outer = 30
n_inner = 6

X_cur = V_cloth.copy()
v_cur = np.tile([0.0, 0.0, -1.0], (N_cloth, 1))

print(f"\n  Simulating {n_outer} time steps x {n_inner} VBD iterations")

frames_anim = []
all_z = {v: [X_cur[v, 2]] for v in range(N_cloth)}

for outer in range(n_outer):
    X_t = X_cur.copy()
    Y = X_t + dt_val * v_cur + dt_val**2 * a_ext

    cdr = True
    X_prev = X_t.copy()
    b_v = {v: r_q_anim for v in range(N_cloth)}
    cs = None

    for inner in range(1, n_inner + 1):
        detected = False

        if cdr:
            V_scene = np.vstack([X_cur, V_floor])
            mesh_i = Mesh.from_arrays(V_scene, T_all)
            pgm_i = PolyhedralGaussMap(mesh_i)
            bvh_i = BVH(mesh_i)
            cs = run_contact_detection(mesh_i, bvh_i, pgm_i, r_anim, r_q_anim)
            X_prev = X_cur.copy()
            cdr = False
            detected = True
            b_all = compute_conservative_bounds(mesh_i, cs, gamma_p)
            b_v = {v: b_all[v] for v in range(N_cloth)}

        if inner == 1:
            X_cur[:N_cloth] = apply_initial_guess_truncation(
                Y.copy(), X_prev, b_v)[:N_cloth]

        # Algorithm 4: one VBD pass
        V_scene = np.vstack([X_cur, V_floor])
        mesh_vbd = Mesh.from_arrays(V_scene, T_all)
        vbd_iteration(
            V_scene,
            np.vstack([X_t, V_floor]),
            np.vstack([Y, V_floor]),
            mesh_vbd, cs, colors, l0,
            dt=dt_val, mass=mass_v, k_s=k_s_val,
            r=r_anim, k_c=k_c_val, n_dof=N_cloth,
        )
        X_cur = V_scene[:N_cloth].copy()

        X_cur, num_exceed = truncate_displacements(X_cur, X_prev, b_v)
        if num_exceed > 0:
            cdr = True

        # Record contact force arrows
        grad_arrows = []
        for v_idx in range(N_cloth):
            g = contact_gradient_v_vf(X_cur[v_idx],
                                       V_floor[0], V_floor[1], V_floor[2],
                                       r_anim, k_c_val)
            if np.linalg.norm(g) > 1e-8:
                grad_arrows.append((v_idx, g))

        contact_active = (cs is not None and
                          any(len(cs.FOGC.get(v, [])) > 0 for v in range(N_cloth)))

        for v_idx in range(N_cloth):
            all_z[v_idx].append(X_cur[v_idx, 2])

        frames_anim.append({
            "X": X_cur.copy(),
            "detected": detected,
            "contact_active": contact_active,
            "grad_arrows": list(grad_arrows),
            "b_v": dict(b_v),
            "outer": outer + 1,
            "inner": inner,
            "z_so_far": {v: list(all_z[v]) for v in range(N_cloth)},
        })

    v_cur = (X_cur - X_t) / dt_val

print(f"  Pre-computed {len(frames_anim)} frames.")
print(f"  Final z: " + ", ".join(f"V{v}={X_cur[v, 2]:.4f}" for v in range(N_cloth)))

# Build animation
COLOR_V = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12"]

fig_anim = plt.figure(figsize=(14, 6))
fig_anim.patch.set_facecolor("#1a1a2e")

ax_3d_a = fig_anim.add_subplot(1, 2, 1, projection="3d")
ax_3d_a.set_facecolor("#1a1a2e")

ax_z_a = fig_anim.add_subplot(1, 2, 2)
ax_z_a.set_facecolor("#0f0f1e")
ax_z_a.tick_params(colors="#aaaacc")

ax_z_a.axhline(y=r_anim, color="orange", lw=1.2, ls=":", label=f"contact r={r_anim}")
ax_z_a.axhline(y=0.0, color="#888899", lw=0.8, ls="-", label="floor z=0")
ax_z_a.set_xlabel("Iteration", fontsize=9, color="#aaaacc")
ax_z_a.set_ylabel("z-position", fontsize=9, color="#aaaacc")
ax_z_a.set_xlim(0, len(frames_anim) + 1)
ax_z_a.set_ylim(-0.1, V_cloth[0, 2] + 0.1)

v_lines = {v: ax_z_a.plot([], [], color=COLOR_V[v], lw=1.5, label=f"V{v}")[0]
            for v in range(N_cloth)}
cur_dots = {v: ax_z_a.plot([], [], "o", color=COLOR_V[v], ms=5)[0]
            for v in range(N_cloth)}
ax_z_a.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="#ccccdd", edgecolor="#555577")


def _setup_3d_anim():
    ax_3d_a.set_xlim(-0.5, 1.5); ax_3d_a.set_ylim(-0.5, 1.5); ax_3d_a.set_zlim(-0.1, 1.2)
    ax_3d_a.set_xlabel("X", color="#aaaacc", fontsize=8)
    ax_3d_a.set_ylabel("Y", color="#aaaacc", fontsize=8)
    ax_3d_a.set_zlabel("Z", color="#aaaacc", fontsize=8)
    ax_3d_a.tick_params(colors="#888899")
    ax_3d_a.view_init(elev=30, azim=-55)


def _update_anim(fi):
    fd = frames_anim[fi]
    X = fd["X"]
    contact_active = fd["contact_active"]

    ax_3d_a.cla()
    _setup_3d_anim()
    draw_mesh(ax_3d_a, V_floor, T_floor, face_color="#f9e4b7", edge_color="#c0a060", alpha=0.5)

    cloth_fc = "#e74c3c" if contact_active else "#5dade2"
    cloth_ec = "#c0392b" if contact_active else "#2980b9"
    draw_mesh(ax_3d_a, X, T_cloth, face_color=cloth_fc, edge_color=cloth_ec, alpha=0.75)

    for v_idx, grad in fd["grad_arrows"]:
        force = -grad
        scale = 0.18 / (np.linalg.norm(force) + 1e-12)
        ax_3d_a.quiver(*X[v_idx], *(force * scale),
                       color="#f39c12", linewidth=2, arrow_length_ratio=0.35)

    theta = np.linspace(0, 2 * np.pi, 40)
    for v_idx in range(N_cloth):
        bv = fd["b_v"].get(v_idx, 0.0)
        if bv > 1e-4:
            cx, cy, cz = X[v_idx]
            ax_3d_a.plot(cx + bv * np.cos(theta), cy + bv * np.sin(theta),
                         np.full(40, cz), color="#a29bfe", lw=0.8, alpha=0.5, ls="--")

    state = "[IN CONTACT]" if contact_active else ""
    ax_3d_a.set_title(
        f"Algorithm 4 (VBD) step {fd['outer']}, iter {fd['inner']} {state}",
        fontsize=8, color="#ecf0f1", pad=5
    )

    z_data = fd["z_so_far"]
    n_pts = len(z_data[0])
    xs = list(range(n_pts))
    for v_idx in range(N_cloth):
        v_lines[v_idx].set_data(xs, z_data[v_idx])
        cur_dots[v_idx].set_data([xs[-1]], [z_data[v_idx][-1]])

    return list(v_lines.values()) + list(cur_dots.values())


ani = animation.FuncAnimation(
    fig_anim, _update_anim, frames=len(frames_anim),
    interval=100, blit=False, repeat=True,
)

fig_anim.suptitle(
    "Example 3 — Algorithm 4 (VBD) animated | Close window to finish",
    fontsize=10, color="#ecf0f1", y=1.01
)
plt.tight_layout()
plt.show()

print("""
  Summary:
    Algorithm 4 is the inner VBD solver that Algorithm 3 calls each iteration.

    Per-vertex Newton step:  x_v <- x_v + H_v^{-1} f_v

    Force contributions:
      1. Inertia:  f = -(m/h^2)(x - y)         (line 3)
      2. Elastic:  springs from incident edges   (lines 4-11)
      3. Contact:  FOGC vertex side              (lines 12-15)
                   VOGC face side                (lines 16-21)
                   EOGC edge-edge                (lines 22-27)

    Key ideas:
      - Graph coloring ensures independent parallel updates
      - PSD projection keeps H_v invertible near contact
      - 3x3 Newton step converges quickly with 2nd-order info
""")
