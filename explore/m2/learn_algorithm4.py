"""
Building Algorithm 4 from scratch — step by step
=================================================
Run this file:
    python3 explore/m2/learn_algorithm4.py

Each step pauses on a plot window.
Close the window → the next step runs.

You can add, change, or comment out anything.
The goal is for you to understand each line before moving on.

How Algorithm 4 relates to Algorithms 1-3
-------------------------------------------
Algorithm 1 asked:   "which triangles are close to this vertex?"
Algorithm 2 asked:   "which edges are close to this edge?"
Algorithm 3 asked:   "how do we move the mesh safely in one time step?"
                      (It calls Algorithm 4 on every inner iteration.)

Algorithm 4 asks:    "given the current positions, which direction should
                      each vertex move to reduce the total energy?"

It is the INNER SOLVER that Algorithm 3 calls every iteration.
The full structure is:

  Algorithm 4: VBD iteration with contact
  ----------------------------------------
  Inputs:  X    (current positions)
           X_t  (positions at start of time step)
           Y    (inertia target  Y = X_t + dt·v + dt²·a_ext)

  1   for each color c  (process one color at a time):
  2     for each vertex v in color c  (parallel on GPU):

  3       f_v = -(m/h²)(x_v − y_v),   H_v = (m/h²) I    ← inertia

  4       for each triangle t ∈ T_v:
  5         f_v += −∂E_t/∂x_v,         H_v += ∂²E_t/∂x_v²  ← elastic

  8       for each edge e ∈ E_v:
  9         f_v += −∂E_e/∂x_v,         H_v += ∂²E_e/∂x_v²  ← bending

  12      for each a ∈ FOGC(v):
  13        f_v += −∂E_vf(v,a)/∂x_v,  H_v += ∂²E_vf/∂x_v²  ← VF contact (vertex)

  16      for each t ∈ T_v:
  17        for each v' ∈ VOGC(t):
  18          f_v += −∂E_vf(v',t)/∂x_v,  H_v += ∂²E_vf/∂x_v²  ← VF contact (face)

  22      for each e ∈ E_v:
  23        for each (e,e') ∈ EOGC(e):
  24          f_v += −∂E_ee(e,e')/∂x_v,  H_v += ∂²E_ee/∂x_v²  ← EE contact

  28      x_v ← x_v + H_v⁻¹ f_v        ← Newton step

  31  return X
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ogc_sim.geometry.mesh      import Mesh
from ogc_sim.geometry.bvh       import BVH
from ogc_sim.geometry.gauss_map import PolyhedralGaussMap
from ogc_sim.contact.detection  import run_contact_detection
from ogc_sim.contact.energy     import (
    activation_g, activation_dg_dd, activation_d2g_dd2,
    contact_energy_vf, contact_gradient_v_vf, contact_hessian_v_vf,
)
from ogc_sim.contact.bounds import (
    compute_conservative_bounds,
    truncate_displacements,
    apply_initial_guess_truncation,
)
from ogc_sim.solver.vbd import (
    graph_color_mesh,
    compute_rest_lengths,
    spring_force_hessian,
    vbd_iteration,
)


def pause(title: str) -> None:
    plt.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.show()


def draw_mesh(ax, V: np.ndarray, T: np.ndarray,
              face_color: str = "#aed6f1",
              edge_color: str = "#2980b9",
              alpha: float = 0.35) -> None:
    tris = [[V[T[i, 0]], V[T[i, 1]], V[T[i, 2]]] for i in range(len(T))]
    ax.add_collection3d(Poly3DCollection(
        tris, alpha=alpha, facecolor=face_color, edgecolor=edge_color, lw=1.2
    ))
    for v in V:
        ax.scatter(*v, color=edge_color, s=30, zorder=6)


# ============================================================
# Scene setup
# ============================================================
# A 2-triangle cloth quad (4 vertices) above a static floor triangle.
#
#   cloth:  V0=(0,0,1.0)  V1=(1,0,1.0)
#           V2=(0,1,1.0)  V3=(1,1,1.0)
#   two cloth triangles:  (V0,V1,V3) and (V0,V3,V2)
#
#   floor:  V4=(-0.5,-0.5,0)  V5=(1.5,-0.5,0)  V6=(0.5,1.5,0)
#
# Using a 2-triangle cloth gives us a more interesting graph coloring
# (3 colors instead of 1) and demonstrates the elastic energy
# coupling between the two triangles.

V_cloth = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
], dtype=float)

T_cloth = np.array([[0, 1, 3], [0, 3, 2]])   # two triangles sharing edge (0,3)

V_floor = np.array([
    [-0.5, -0.5, 0.0],
    [ 1.5, -0.5, 0.0],
    [ 0.5,  1.5, 0.0],
], dtype=float)

T_floor = np.array([[0, 1, 2]])

# Combined mesh (cloth = vertices 0-3, floor = vertices 4-6)
V_all = np.vstack([V_cloth, V_floor])
T_all = np.vstack([T_cloth, T_floor + len(V_cloth)])
mesh_rest = Mesh.from_arrays(V_all, T_all)

N_cloth = len(V_cloth)   # 4 cloth vertices
r_val   = 0.35
k_c_val = 500.0
k_s_val = 200.0
dt_val  = 0.04
mass_v  = 1.0
a_ext   = np.array([0.0, 0.0, -9.8])


# ============================================================
# STEP 1 — What problem does VBD solve?
# ============================================================
# Backward Euler time integration minimises the total energy:
#
#   E_total(X) = Σ_v (m_v / 2h²)||x_v − y_v||²    ← inertia
#              + Σ_t E_t(X)                          ← elastic
#              + Σ_contact E_c(X)                    ← contact
#
# A full Newton solve would update all vertices simultaneously.
# VBD instead updates ONE VERTEX AT A TIME, keeping all others fixed.
# This makes each sub-problem a tiny 3×3 linear solve.
#
# The update for vertex v (with everything else fixed) is:
#
#   x_v  ←  x_v  +  H_v⁻¹ f_v
#
# where f_v = total force on v  and  H_v = total Hessian block.
# ============================================================

print("=" * 55)
print("STEP 1 — What problem does VBD solve?")
print("=" * 55)

# Show the 1-D energy landscape for a single vertex
# to make the Newton step intuitive.
x_vals = np.linspace(-1.5, 2.5, 300)

m_demo, h_demo = 1.0, dt_val
y_demo = 1.0   # inertia target

# Two springs: one connecting to a fixed left neighbour at x=0,
# and one to a fixed right neighbour at x=2.0.
k_demo  = 200.0
x_left, x_right = 0.0, 2.0
l0_left = abs(y_demo - x_left)
l0_right= abs(y_demo - x_right)

def E_total_1d(x):
    E_in  = (m_demo / (2 * h_demo**2)) * (x - y_demo)**2
    E_el  = (k_demo / 2) * (abs(x - x_left)  - l0_left)**2
    E_el += (k_demo / 2) * (abs(x - x_right) - l0_right)**2
    return E_in + E_el

E_vals = np.array([E_total_1d(x) for x in x_vals])

x0 = 1.8   # starting position (off equilibrium)
# gradient = m/h²*(x-y) + springs
dEdx = (m_demo / h_demo**2) * (x0 - y_demo) \
     + k_demo * (x0 - x_left - l0_left) * np.sign(x0 - x_left) \
     + k_demo * (x0 - x_right - l0_right) * np.sign(x0 - x_right)
d2Edx2 = (m_demo / h_demo**2) + k_demo + k_demo
x_newton = x0 - dEdx / d2Edx2   # one Newton step

print(f"\n  1-D demo:  y (inertia target) = {y_demo}")
print(f"  Start position x0 = {x0}   f(x0) = {dEdx:.2f}   H(x0) = {d2Edx2:.2f}")
print(f"  Newton step: Δx = {-dEdx/d2Edx2:.4f}   x_new = {x_newton:.4f}")
print(f"  True minimum (analytic): x* ≈ {y_demo:.4f}  (at equilibrium)")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x_vals, E_vals, color="#2980b9", lw=2, label="E_total(x)")
ax.axvline(y_demo, color="green", lw=1.5, ls="--", label=f"inertia target y={y_demo}")
ax.axvline(x0,     color="red",   lw=1.5, ls=":",  label=f"start x₀={x0}")
ax.axvline(x_newton, color="orange", lw=2.0, ls="-", label=f"Newton step → {x_newton:.3f}")
ax.scatter([x0],      [E_total_1d(x0)],      color="red",    s=60, zorder=5)
ax.scatter([x_newton],[E_total_1d(x_newton)], color="orange", s=60, zorder=5)
ax.set_xlabel("vertex position x"); ax.set_ylabel("Total energy")
ax.set_title("VBD: find x that minimises E_total  (1-D analogy)", fontsize=10)
ax.legend(fontsize=8)
pause("Step 1 — Energy landscape: VBD takes one Newton step per vertex")


# ============================================================
# STEP 2 — Graph coloring
# ============================================================
# Why do we need coloring?
#
# VBD processes one color at a time (line 1 of Algorithm 4).
# Within a color all vertices are independent: they share NO
# triangle and NO edge.  This means their Newton steps don't
# interfere — they can be done in parallel on the GPU.
#
# Coloring rule:  two vertices that share a triangle (or edge)
#                 must get different colors.
#
# For a quad (2 triangles), the shared edge (V0-V3) means V0 and
# V3 must differ.  A greedy algorithm assigns 3 colors here.
# ============================================================

print("\n" + "=" * 55)
print("STEP 2 — Graph coloring")
print("=" * 55)

# Build cloth-only mesh for coloring illustration
mesh_cloth = Mesh.from_arrays(V_cloth, T_cloth)
colors_cloth = graph_color_mesh(mesh_cloth)

print(f"\n  Cloth mesh:  {mesh_cloth.num_vertices} vertices,  "
      f"{mesh_cloth.num_triangles} triangles,  {mesh_cloth.num_edges} edges")
print(f"  Graph coloring: {len(colors_cloth)} colors")
for ci, group in enumerate(colors_cloth):
    print(f"    color {ci}: vertices {group}")
print()
print("  ← All vertices in the same color can be updated in parallel")
print("    because no two of them share a triangle.")

COLOR_MAP = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6"]

fig = plt.figure(figsize=(10, 5))
ax_col = fig.add_subplot(1, 2, 1)
ax_3d2 = fig.add_subplot(1, 2, 2, projection="3d")

# 2-D coloring diagram
positions_2d = {
    0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0), 3: (1.0, 1.0)
}

# Draw triangle edges
for tri in T_cloth:
    for i in range(3):
        a, b = tri[i], tri[(i+1) % 3]
        xa, ya = positions_2d[a]
        xb, yb = positions_2d[b]
        ax_col.plot([xa, xb], [ya, yb], color="#888888", lw=1.5, zorder=1)

# Draw vertices colored
v_color_map = {}
for ci, group in enumerate(colors_cloth):
    for v in group:
        v_color_map[v] = ci

for v_idx in range(4):
    x2, y2 = positions_2d[v_idx]
    ci  = v_color_map[v_idx]
    col = COLOR_MAP[ci]
    ax_col.scatter(x2, y2, color=col, s=250, zorder=5)
    ax_col.text(x2 + 0.06, y2 + 0.04, f"V{v_idx}\ncolor {ci}",
                fontsize=9, color=col)

ax_col.set_xlim(-0.3, 1.5); ax_col.set_ylim(-0.3, 1.5)
ax_col.set_aspect("equal"); ax_col.axis("off")
ax_col.set_title("Cloth graph coloring\n(same color = independent, can run in parallel)",
                 fontsize=9)

# 3-D view
for ci, group in enumerate(colors_cloth):
    for v in group:
        ax_3d2.scatter(*V_cloth[v], color=COLOR_MAP[ci], s=100, zorder=6)
        ax_3d2.text(*(V_cloth[v] + np.array([0.04, 0.04, 0.04])),
                    f"V{v}", fontsize=8, color=COLOR_MAP[ci])

draw_mesh(ax_3d2, V_cloth, T_cloth, face_color="#aed6f1", edge_color="#888888", alpha=0.3)
ax_3d2.set_xlabel("X"); ax_3d2.set_ylabel("Y"); ax_3d2.set_zlabel("Z")
ax_3d2.set_xlim(-0.3, 1.5); ax_3d2.set_ylim(-0.3, 1.5); ax_3d2.set_zlim(0.8, 1.2)
ax_3d2.view_init(elev=35, azim=-50)
ax_3d2.set_title("3-D view of coloring", fontsize=9)

pause("Step 2 — Graph coloring (same-color vertices share no triangle)")


# ============================================================
# STEP 3 — Inertia term  (Algorithm 4, line 3)
# ============================================================
# The inertia energy:
#
#   E_inertia(x_v) = (m_v / 2h²) ||x_v − y_v||²
#
# Gradient (= negative force):
#   ∂E/∂x_v = (m_v / h²)(x_v − y_v)   →   f_v = −(m_v/h²)(x_v − y_v)
#
# Hessian:
#   H_v = (m_v / h²) I
#
# This term pulls x_v toward y_v (the free-fall target).
# Without elastic or contact forces, the Newton step would move
# x_v exactly to y_v in one shot.
# ============================================================

print("\n" + "=" * 55)
print("STEP 3 — Inertia term")
print("=" * 55)

X_step3 = V_cloth.copy()   # start at rest position
Y_step3 = V_cloth + dt_val * np.tile([0.0, 0.0, -1.5], (4, 1)) \
         + dt_val**2 * a_ext  # inertia target (one free-fall step)
h2 = dt_val**2

print(f"\n  Mass = {mass_v}  dt = {dt_val}  m/h² = {mass_v/h2:.2f}")
print(f"  Y (free-fall target):")
for i in range(4):
    print(f"    V{i}: {np.round(Y_step3[i], 4)}")

# Compute inertia force for each cloth vertex
print(f"\n  Inertia force  f_v = −(m/h²)(x_v − y_v):")
for v in range(N_cloth):
    f_in = -(mass_v / h2) * (X_step3[v] - Y_step3[v])
    H_in =  (mass_v / h2) * np.eye(3)
    print(f"    V{v}: f_v = {np.round(f_in, 3)},  H_v (diag) = {(mass_v/h2):.2f}·I")

# Visualise: arrows from x_v toward y_v
fig = plt.figure(figsize=(7, 6))
ax3 = fig.add_subplot(111, projection="3d")
draw_mesh(ax3, X_step3, T_cloth, face_color="#aed6f1", edge_color="#2980b9", alpha=0.4)

for v in range(N_cloth):
    direction = Y_step3[v] - X_step3[v]
    ax3.quiver(*X_step3[v], *direction * 0.7, color="#e74c3c",
               linewidth=2, arrow_length_ratio=0.25)
    ax3.text(*(X_step3[v] + np.array([0.04, 0.04, 0.0])),
             f"V{v}", fontsize=8, color="#2980b9")

ax3.scatter(*Y_step3.T, color="green", s=50, marker="^", zorder=5, alpha=0.6)
ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z")
ax3.set_xlim(-0.3, 1.5); ax3.set_ylim(-0.3, 1.5); ax3.set_zlim(0.7, 1.05)
ax3.view_init(elev=20, azim=-60)
ax3.set_title("Inertia term: red arrows = force toward Y (green triangles)", fontsize=9)
pause("Step 3 — Inertia force pulls each vertex toward its free-fall target Y")


# ============================================================
# STEP 4 — Elastic force from incident triangles  (lines 4-7)
# ============================================================
# E_elastic(e) = (k_s/2)(||x_a − x_b|| − l₀)²   for each edge e=(a,b)
#
# Gradient w.r.t. x_v  (vertex endpoint of edge e):
#   f_v,e = k_s (l − l₀) * (x_v − x_other) / l
#           ↑ positive = stretched (pulls v toward other endpoint)
#
# Hessian w.r.t. x_v:
#   H_v,e = k_s d⊗d  +  k_s (1 − l₀/l)(I − d⊗d)
#           ↑ where d = unit direction from v to other endpoint
#
# If the cloth is at rest (l = l₀), f_v = 0 and H_v = k_s I.
# If the cloth is stretched, f_v points to restore the rest length.
# ============================================================

print("\n" + "=" * 55)
print("STEP 4 — Elastic force (mass-spring)")
print("=" * 55)

# Stretch the cloth: move V3 diagonally outward
X_step4 = V_cloth.copy()
X_step4[3] += np.array([0.3, 0.3, 0.0])   # V3 stretched

l0_all = compute_rest_lengths(mesh_rest)
print(f"\n  Rest lengths (cloth edges only, indices in mesh_rest):")
for ei in range(mesh_rest.num_edges):
    a, b = int(mesh_rest.E[ei][0]), int(mesh_rest.E[ei][1])
    if a < N_cloth and b < N_cloth:
        l_cur = np.linalg.norm(X_step4[a] - X_step4[b])
        print(f"    e{ei}: V{a}–V{b}  l₀={l0_all[ei]:.4f}  l={l_cur:.4f}"
              f"  stretch={l_cur - l0_all[ei]:+.4f}")

print(f"\n  Elastic forces on cloth vertices (k_s={k_s_val}):")
for v in range(N_cloth):
    f_e, H_e = spring_force_hessian(v, X_step4, mesh_rest, l0_all, k_s_val)
    print(f"    V{v}: f_el = {np.round(f_e, 3)}")

# Visualise: elastic force arrows (red) on the stretched cloth
fig = plt.figure(figsize=(8, 6))
ax4 = fig.add_subplot(111, projection="3d")
draw_mesh(ax4, V_cloth, T_cloth, face_color="#aed6f1", edge_color="#2980b9", alpha=0.25)
draw_mesh(ax4, X_step4, T_cloth, face_color="#f9e4b7", edge_color="#c0a060", alpha=0.6)

for v in range(N_cloth):
    f_e, _ = spring_force_hessian(v, X_step4, mesh_rest, l0_all, k_s_val)
    f_norm = np.linalg.norm(f_e)
    if f_norm > 1e-6:
        scale = 0.3 / f_norm
        ax4.quiver(*X_step4[v], *(f_e * scale), color="#e74c3c",
                   linewidth=2, arrow_length_ratio=0.25)
    ax4.text(*(X_step4[v] + np.array([0.04, 0.04, 0.0])),
             f"V{v}", fontsize=8, color="#c0a060")

ax4.set_xlabel("X"); ax4.set_ylabel("Y"); ax4.set_zlabel("Z")
ax4.set_xlim(-0.3, 1.7); ax4.set_ylim(-0.3, 1.5); ax4.set_zlim(0.8, 1.2)
ax4.view_init(elev=28, azim=-60)
ax4.set_title("Elastic forces: blue=rest, gold=stretched, red=restoring force", fontsize=9)
pause("Step 4 — Elastic spring forces pull stretched vertices back to rest length")


# ============================================================
# STEP 5 — Contact force from FOGC(v)  (Algorithm 4, lines 12-15)
# ============================================================
# When vertex v is inside the contact zone (d(v, face) < r):
#
#   f_v,a = −∂E_vf(v, a)/∂x_v    (contact gradient, vertex side)
#   H_v,a = ∂²E_vf(v, a)/∂x_v²   (contact Hessian, vertex side)
#
# These are computed by contact_gradient_v_vf / contact_hessian_v_vf.
# The force pushes v AWAY from the contacting face.
# ============================================================

print("\n" + "=" * 55)
print("STEP 5 — Contact force from FOGC(v)  (vertex side)")
print("=" * 55)

# Place one cloth vertex inside contact zone above the floor
X_step5 = np.array([
    [0.0, 0.0, 0.20],   # V0: well inside r=0.35
    [1.0, 0.0, 0.50],   # V1: outside contact zone
    [0.0, 1.0, 0.50],
    [1.0, 1.0, 0.50],
], dtype=float)

print(f"\n  Contact radius r = {r_val},  k_c = {k_c_val}")
print(f"  V0 is at z=0.20 → d to floor ≈ 0.20 < r={r_val}  ← in contact")
print(f"  V1-V3 at z=0.50 → d to floor ≈ 0.50 > r={r_val}  ← out of contact")

for v in range(N_cloth):
    d_approx = X_step5[v, 2]   # floor is at z=0
    E = contact_energy_vf(X_step5[v],
                          V_floor[0], V_floor[1], V_floor[2],
                          r_val, k_c_val)
    g = contact_gradient_v_vf(X_step5[v],
                               V_floor[0], V_floor[1], V_floor[2],
                               r_val, k_c_val)
    H = contact_hessian_v_vf(X_step5[v],
                              V_floor[0], V_floor[1], V_floor[2],
                              r_val, k_c_val)
    f_contact = -g   # force = -gradient
    print(f"  V{v}: z={X_step5[v,2]:.2f}  E={E:.4f}  "
          f"f_contact={np.round(f_contact, 4)}")

fig = plt.figure(figsize=(7, 6))
ax5 = fig.add_subplot(111, projection="3d")
draw_mesh(ax5, V_floor, T_floor, face_color="#f9e4b7", edge_color="#c0a060", alpha=0.5)
draw_mesh(ax5, X_step5, T_cloth, face_color="#aed6f1", edge_color="#2980b9", alpha=0.6)

for v in range(N_cloth):
    g = contact_gradient_v_vf(X_step5[v],
                               V_floor[0], V_floor[1], V_floor[2],
                               r_val, k_c_val)
    f = -g
    f_n = np.linalg.norm(f)
    if f_n > 1e-6:
        scale = 0.25 / f_n
        ax5.quiver(*X_step5[v], *(f * scale), color="#e74c3c",
                   linewidth=2.5, arrow_length_ratio=0.3)
    ax5.text(*(X_step5[v] + np.array([0.04, 0.04, 0.0])),
             f"V{v}\nz={X_step5[v,2]:.2f}", fontsize=7, color="#2980b9")

# Draw contact radius sphere for V0
theta = np.linspace(0, 2 * np.pi, 40)
ax5.plot(X_step5[0, 0] + r_val * np.cos(theta),
         X_step5[0, 1] + r_val * np.sin(theta),
         np.full(40, 0.0), color="orange", lw=1, ls="--", alpha=0.5)

ax5.set_xlabel("X"); ax5.set_ylabel("Y"); ax5.set_zlabel("Z")
ax5.set_xlim(-0.5, 1.5); ax5.set_ylim(-0.5, 1.5); ax5.set_zlim(-0.1, 0.7)
ax5.view_init(elev=20, azim=-55)
ax5.set_title("FOGC contact: red arrows = contact force on cloth vertices\n"
              "(only V0 is inside r=0.35)", fontsize=9)
pause("Step 5 — Contact force from FOGC(v): pushes cloth vertex away from floor")


# ============================================================
# STEP 6 — Contact force from VOGC(t)  (lines 16-21)
# ============================================================
# When vertex v' hits triangle t, and v is a vertex of t:
#
#   The contact energy E_vf(v', t) depends on the position of t's
#   vertices (including v) through the distance function d(v', t).
#   As t's vertices move, the plane of t shifts, changing d.
#
#   f_t,v' = −∂E_vf(v', t)/∂x_v   ← gradient w.r.t. a vertex of t
#
# This is the "face side" of the contact — the reaction force that
# the contacting vertex v' transmits to triangle t's vertices.
#
# In this demo v' is a "query vertex" pressing on triangle t from
# above, and we show how the force distributes to t's 3 vertices.
# ============================================================

print("\n" + "=" * 55)
print("STEP 6 — Contact force from VOGC(t)  (face side)")
print("=" * 55)

# Toy setup: v' at (0.4, 0.4, 0.15) above the floor triangle
vp = np.array([0.4, 0.4, 0.15])
ta = V_floor[0].copy()
tb = V_floor[1].copy()
tc = V_floor[2].copy()

d_vp, _, _, _ = __import__("ogc_sim.geometry.distance",
                            fromlist=["point_triangle_distance"]
                            ).point_triangle_distance(vp, ta, tb, tc)

print(f"\n  v' = {vp}  (query vertex pressing on floor)")
print(f"  d(v', floor) = {d_vp:.4f}  (r = {r_val})  "
      f"{'IN contact' if d_vp < r_val else 'outside'}")

# Compute face-side gradient via FD (as used in vbd.py)
from ogc_sim.solver.vbd import _contact_grad_tri_vf_fd

ga, gb, gc = _contact_grad_tri_vf_fd(vp, ta, tb, tc, r_val, k_c_val)
print(f"\n  Face-side gradients  ∂E_vf(v', t)/∂x_vertex:")
print(f"    ∂E/∂a  (V4): {np.round(ga, 4)}")
print(f"    ∂E/∂b  (V5): {np.round(gb, 4)}")
print(f"    ∂E/∂c  (V6): {np.round(gc, 4)}")
print(f"  Forces −∂E/∂ = reaction on each floor vertex:")
print(f"    f_a = {np.round(-ga, 4)}")
print(f"    f_b = {np.round(-gb, 4)}")
print(f"    f_c = {np.round(-gc, 4)}")

fig = plt.figure(figsize=(7, 6))
ax6 = fig.add_subplot(111, projection="3d")
draw_mesh(ax6, V_floor, T_floor, face_color="#f9e4b7", edge_color="#c0a060", alpha=0.5)

# v' pressing on floor
ax6.scatter(*vp, color="red", s=80, zorder=8)
ax6.text(*(vp + np.array([0.04, 0.04, 0.0])), "v' (query)", fontsize=8, color="red")

# Force arrows on floor triangle vertices
for i, (pt, grad) in enumerate(zip([ta, tb, tc], [-ga, -gb, -gc])):
    g_n = np.linalg.norm(grad)
    if g_n > 1e-8:
        scale = 0.3 / g_n
        ax6.quiver(*pt, *(grad * scale), color="#27ae60",
                   linewidth=2.5, arrow_length_ratio=0.3)
    ax6.text(*(pt + np.array([0.04, 0.04, 0.0])),
             f"V{4+i}", fontsize=8, color="#c0a060")

# Vertical line from v' to floor
d_vp_obj, cp_vp, _, _ = __import__("ogc_sim.geometry.distance",
    fromlist=["point_triangle_distance"]).point_triangle_distance(
        vp, ta, tb, tc)
ax6.plot([vp[0], cp_vp[0]], [vp[1], cp_vp[1]], [vp[2], cp_vp[2]],
         "r--", lw=1, alpha=0.5)

ax6.set_xlabel("X"); ax6.set_ylabel("Y"); ax6.set_zlabel("Z")
ax6.set_xlim(-0.6, 1.6); ax6.set_ylim(-0.6, 1.6); ax6.set_zlim(-0.05, 0.45)
ax6.view_init(elev=22, azim=-50)
ax6.set_title("VOGC face-side: green arrows = reaction force on floor vertices\n"
              "(face-side gradient of E_vf w.r.t. each triangle vertex)", fontsize=9)
pause("Step 6 — Face-side contact: v' pressing on t distributes force to t's vertices")


# ============================================================
# STEP 7 — Full per-vertex accumulation  (lines 3-27)
# ============================================================
# Put inertia + elastic + contact together for one vertex.
# Show each contribution separately, then the combined f_v and H_v.
# ============================================================

print("\n" + "=" * 55)
print("STEP 7 — Full per-vertex force accumulation")
print("=" * 55)

# Scene: cloth slightly above floor, V0 in contact zone
X_step7 = np.array([
    [0.0, 0.0, 0.20],
    [1.0, 0.0, 0.60],
    [0.0, 1.0, 0.60],
    [1.0, 1.0, 0.60],
], dtype=float)
Y_step7 = X_step7 + dt_val * np.tile([0.0, 0.0, -1.5], (4, 1)) \
         + dt_val**2 * a_ext

V_all7 = np.vstack([X_step7, V_floor])
mesh7  = Mesh.from_arrays(V_all7, np.vstack([T_cloth, T_floor + N_cloth]))
pgm7   = PolyhedralGaussMap(mesh7)
bvh7   = BVH(mesh7)
cs7    = run_contact_detection(mesh7, bvh7, pgm7, r_val, r_val + 0.15)
l0_7   = compute_rest_lengths(mesh7)

focus_v = 0   # examine V0 (inside contact zone)
print(f"\n  Examining vertex V{focus_v} at position {X_step7[focus_v]}")
print(f"  FOGC(V{focus_v}) = {cs7.FOGC.get(focus_v, [])}")

m_h2 = mass_v / h2
# Inertia
f_in = -(m_h2) * (X_step7[focus_v] - Y_step7[focus_v])
H_in = m_h2 * np.eye(3)

# Elastic
f_el, H_el = spring_force_hessian(focus_v, X_step7, mesh7, l0_7, k_s_val)

# Contact (FOGC vertex side)
f_ct = np.zeros(3); H_ct = np.zeros((3, 3))
for a_feat in cs7.FOGC.get(focus_v, []):
    from ogc_sim.solver.vbd import _find_parent_triangle
    t_idx = _find_parent_triangle(a_feat, mesh7)
    if t_idx >= 0:
        tri = mesh7.T[t_idx]
        ap, bp, cp_ = (X_step7 if i < N_cloth else V_floor)[int(tri[k]) - (0 if int(tri[k]) < N_cloth else N_cloth)]
        # Simpler: use X_step7 stacked with V_floor
        V_all7_pos = np.vstack([X_step7, V_floor])
        ap = V_all7_pos[int(tri[0])]; bp = V_all7_pos[int(tri[1])]; cp_ = V_all7_pos[int(tri[2])]
        g  = contact_gradient_v_vf(X_step7[focus_v], ap, bp, cp_, r_val, k_c_val)
        Hc = contact_hessian_v_vf(X_step7[focus_v], ap, bp, cp_, r_val, k_c_val)
        f_ct -= g
        H_ct += Hc

f_total = f_in + f_el + f_ct
H_total = H_in + H_el + H_ct

print(f"\n  Force contributions on V{focus_v}:")
print(f"    f_inertia = {np.round(f_in, 4)}")
print(f"    f_elastic = {np.round(f_el, 4)}")
print(f"    f_contact = {np.round(f_ct, 4)}")
print(f"    ─────────────────────────")
print(f"    f_total   = {np.round(f_total, 4)}")

# Newton step
eigvals, eigvecs = np.linalg.eigh(H_total)
H_psd = eigvecs @ np.diag(np.maximum(eigvals, 1e-9)) @ eigvecs.T
delta = np.linalg.solve(H_psd, f_total)
x_new = X_step7[focus_v] + delta
print(f"\n  Newton step:  Δx = {np.round(delta, 5)}")
print(f"  x_v → {np.round(x_new, 5)}")
print(f"  (negative Δz = contact pushing V0 upward from z={X_step7[focus_v,2]:.3f} "
      f"→ {x_new[2]:.4f})")

# Bar chart of force magnitudes
categories = ["inertia", "elastic", "contact"]
magnitudes = [np.linalg.norm(f_in),
              np.linalg.norm(f_el),
              np.linalg.norm(f_ct)]

fig, (ax_bar, ax_arr) = plt.subplots(1, 2, figsize=(12, 5))
bars = ax_bar.bar(categories, magnitudes,
                  color=["#3498db", "#2ecc71", "#e74c3c"])
ax_bar.set_ylabel("||force||", fontsize=10)
ax_bar.set_title(f"Force magnitudes on V{focus_v}", fontsize=10)
for bar, mag in zip(bars, magnitudes):
    ax_bar.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1, f"{mag:.2f}", ha="center", fontsize=9)

# 3-D arrow diagram
ax_arr = fig.add_subplot(1, 2, 2, projection="3d")
draw_mesh(ax_arr, V_floor, T_floor, face_color="#f9e4b7", edge_color="#c0a060", alpha=0.4)
draw_mesh(ax_arr, X_step7, T_cloth, face_color="#aed6f1", edge_color="#2980b9", alpha=0.5)

pos = X_step7[focus_v]
force_data = [("inertia", f_in, "#3498db"),
              ("elastic", f_el, "#2ecc71"),
              ("contact", f_ct, "#e74c3c"),
              ("TOTAL",   f_total, "#f39c12")]

for label, fv, col in force_data:
    fn = np.linalg.norm(fv)
    if fn > 1e-6:
        scale = 0.3 / fn
        ax_arr.quiver(*pos, *(fv * scale), color=col,
                      linewidth=2.5, arrow_length_ratio=0.3, label=label)

ax_arr.scatter(*pos, color="black", s=60, zorder=7)
ax_arr.text(*pos + np.array([0.04, 0.04, 0.0]), f"V{focus_v}", fontsize=8)
ax_arr.legend(fontsize=8, loc="upper right")
ax_arr.set_xlabel("X"); ax_arr.set_ylabel("Y"); ax_arr.set_zlabel("Z")
ax_arr.set_xlim(-0.5, 1.5); ax_arr.set_ylim(-0.5, 1.5); ax_arr.set_zlim(-0.05, 0.8)
ax_arr.view_init(elev=22, azim=-55)
ax_arr.set_title(f"Force arrows on V{focus_v}\n"
                 f"(inertia=blue, elastic=green, contact=red, total=orange)", fontsize=9)
pause("Step 7 — All forces accumulated; total force drives the Newton step")


# ============================================================
# STEP 8 — Newton step and PSD projection  (line 28)
# ============================================================
# The update rule:
#
#   x_v ← x_v + H_v⁻¹ f_v
#
# For VBD to be stable, H_v must be symmetric positive definite (SPD).
# We guarantee this by clamping negative eigenvalues to a small ε > 0
# before solving.  This is called the "PSD projection".
#
# Without PSD projection, a negative H eigenvalue would make the solver
# move in the wrong direction (energy goes up instead of down).
# ============================================================

print("\n" + "=" * 55)
print("STEP 8 — Newton step and PSD projection")
print("=" * 55)

eigvals_full, eigvecs_full = np.linalg.eigh(H_total)
print(f"\n  H_total eigenvalues (V{focus_v}): {np.round(eigvals_full, 2)}")
print(f"  All positive? {np.all(eigvals_full > 0)}")

# Demonstrate effect of PSD projection
H_bad = H_total.copy()
H_bad[0, 0] -= 2 * eigvals_full[0] + 1.0   # force a negative eigenvalue
eig_bad, _ = np.linalg.eigh(H_bad)
eig_clamped = np.maximum(eig_bad, 1e-9)
print(f"\n  Artificial bad H eigenvalues: {np.round(eig_bad, 2)}")
print(f"  After clamping to ≥ 1e-9:    {np.round(eig_clamped, 2)}")

# Newton step quality
eigvals_psd = np.maximum(eigvals_full, 1e-9)
H_psd2 = eigvecs_full @ np.diag(eigvals_psd) @ eigvecs_full.T
delta_final = np.linalg.solve(H_psd2, f_total)
x_final = X_step7[focus_v] + delta_final

print(f"\n  Newton step Δx   = {np.round(delta_final, 5)}")
print(f"  Final x_v        = {np.round(x_final, 5)}")
print(f"  ΔE (inertia only) = {(mass_v/h2) * 0.5 * np.linalg.norm(x_final - Y_step7[focus_v])**2:.6f}  "
      f"(vs {(mass_v/h2) * 0.5 * np.linalg.norm(X_step7[focus_v] - Y_step7[focus_v])**2:.6f} at start)")

# Visualise eigenvalue spectrum
fig, (ax8a, ax8b) = plt.subplots(1, 2, figsize=(12, 4))
ax8a.bar(range(3), eigvals_full, color="#3498db", label="H_total eigvals")
ax8a.axhline(0, color="red", lw=1)
ax8a.set_xticks(range(3)); ax8a.set_xticklabels([f"λ{i}" for i in range(3)])
ax8a.set_title(f"H_v eigenvalues (V{focus_v})\nAll > 0 → SPD → safe Newton step", fontsize=9)
ax8a.set_ylabel("eigenvalue")

ax8b.bar(range(3), eig_bad, color="#e74c3c", alpha=0.5, label="raw (bad)")
ax8b.bar(range(3), eig_clamped, color="#2ecc71", alpha=0.7, label="clamped")
ax8b.axhline(0, color="black", lw=1)
ax8b.set_xticks(range(3)); ax8b.set_xticklabels([f"λ{i}" for i in range(3)])
ax8b.set_title("PSD projection: clamp negative eigenvalues\n(ensures Newton moves downhill)", fontsize=9)
ax8b.legend(fontsize=9)
ax8b.set_ylabel("eigenvalue")
pause("Step 8 — PSD projection ensures the Newton step reduces energy")


# ============================================================
# STEP 9 — Full VBD pass animated
# ============================================================
# Run Algorithm 4 for multiple iterations integrated with
# Algorithm 3's outer loop.  Animate:
#   LEFT  — 3-D cloth descending; red when in contact, force arrows
#   RIGHT — per-vertex z-position curves
# ============================================================

print("\n" + "=" * 55)
print("STEP 9 — Full VBD pass animated")
print("=" * 55)

r_anim   = 0.35
r_q_anim = 0.50
k_c_anim = 500.0
k_s_anim = 150.0
mass_anim= 1.0
dt_anim  = 0.035
gamma_p  = 0.45
n_outer  = 30
n_inner  = 6

v_init   = np.array([0.0, 0.0, -1.0])
X_cur    = V_cloth.copy()
v_cur    = np.tile(v_init, (N_cloth, 1))

# Build a combined mesh fresh for each outer step
T_all_anim = np.vstack([T_cloth, T_floor + N_cloth])

# Pre-compute rest lengths once on initial cloth + floor
mesh0  = Mesh.from_arrays(np.vstack([X_cur, V_floor]), T_all_anim)
l0_anim = compute_rest_lengths(mesh0)

# Graph coloring (cloth vertices only; floor is static)
colors_anim = graph_color_mesh(Mesh.from_arrays(V_cloth, T_cloth))

print(f"\n  Simulating {n_outer} time steps × {n_inner} VBD iterations")
print(f"  dt={dt_anim}  r={r_anim}  k_c={k_c_anim}  k_s={k_s_anim}")

frames_anim = []
all_z       = {v: [X_cur[v, 2]] for v in range(N_cloth)}
frame_count = 0

for outer in range(n_outer):
    X_t  = X_cur.copy()
    Y    = X_t + dt_anim * v_cur + dt_anim**2 * a_ext

    cdr  = True   # collisionDetectionRequired
    X_prev = X_t.copy()
    b_v  = {v: r_q_anim for v in range(N_cloth)}
    cs   = None

    for inner in range(1, n_inner + 1):
        frame_count += 1
        detected = False

        if cdr:
            V_scene = np.vstack([X_cur, V_floor])
            mesh_i  = Mesh.from_arrays(V_scene, T_all_anim)
            pgm_i   = PolyhedralGaussMap(mesh_i)
            bvh_i   = BVH(mesh_i)
            cs      = run_contact_detection(mesh_i, bvh_i, pgm_i, r_anim, r_q_anim)

            X_prev  = X_cur.copy()
            cdr     = False
            detected= True

            b_all   = compute_conservative_bounds(mesh_i, cs, gamma_p)
            b_v     = {v: b_all[v] for v in range(N_cloth)}

        if inner == 1:
            X_cur[:N_cloth] = apply_initial_guess_truncation(
                Y.copy(), X_prev, b_v)[:N_cloth]

        # --- REAL VBD iteration (Algorithm 4) ---
        V_scene2 = np.vstack([X_cur, V_floor])
        mesh_vbd = Mesh.from_arrays(V_scene2, T_all_anim)
        vbd_iteration(
            V_scene2, X_t_full := np.vstack([X_t, V_floor]),
            np.vstack([Y, V_floor]),
            mesh_vbd, cs if cs is not None else run_contact_detection(
                mesh_vbd, BVH(mesh_vbd), PolyhedralGaussMap(mesh_vbd),
                r_anim, r_q_anim),
            colors_anim,
            l0_anim,
            dt=dt_anim,
            mass=mass_anim,
            k_s=k_s_anim,
            r=r_anim,
            k_c=k_c_anim,
            n_dof=N_cloth,
        )
        X_cur = V_scene2[:N_cloth].copy()

        X_cur, num_exceed = truncate_displacements(X_cur, X_prev, b_v)

        if num_exceed > 0:
            cdr = True

        # Grad arrows
        grad_arrows = []
        for v_idx in range(N_cloth):
            g = contact_gradient_v_vf(X_cur[v_idx],
                                       V_floor[0], V_floor[1], V_floor[2],
                                       r_anim, k_c_anim)
            if np.linalg.norm(g) > 1e-8:
                grad_arrows.append((v_idx, g))

        contact_active = (
            cs is not None and
            any(len(cs.FOGC.get(v, [])) > 0 for v in range(N_cloth))
        )

        for v_idx in range(N_cloth):
            all_z[v_idx].append(X_cur[v_idx, 2])

        frames_anim.append({
            "X":             X_cur.copy(),
            "detected":      detected,
            "contact_active":contact_active,
            "grad_arrows":   list(grad_arrows),
            "b_v":           dict(b_v),
            "outer":         outer + 1,
            "inner":         inner,
            "z_so_far":      {v: list(all_z[v]) for v in range(N_cloth)},
            "frame_idx":     frame_count,
        })

    v_cur = (X_cur - X_t) / dt_anim

print(f"  Pre-computed {len(frames_anim)} frames.  Final z: "
      + ", ".join(f"V{v}={X_cur[v,2]:.4f}" for v in range(N_cloth)))

# ---- Build animation -------------------------------------------
COLOR_V = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12"]

fig_anim = plt.figure(figsize=(14, 6))
fig_anim.patch.set_facecolor("#1a1a2e")

ax_3d_a = fig_anim.add_subplot(1, 2, 1, projection="3d")
ax_3d_a.set_facecolor("#1a1a2e")

ax_z_a = fig_anim.add_subplot(1, 2, 2)
ax_z_a.set_facecolor("#0f0f1e")
ax_z_a.tick_params(colors="#aaaacc")

ax_z_a.axhline(y=r_anim, color="orange", lw=1.2, ls=":", label=f"contact r={r_anim}")
ax_z_a.axhline(y=0.0,    color="#888899", lw=0.8, ls="-",  label="floor z=0")
ax_z_a.set_xlabel("Iteration", fontsize=9, color="#aaaacc")
ax_z_a.set_ylabel("z-position",             fontsize=9, color="#aaaacc")
ax_z_a.set_xlim(0, len(frames_anim) + 1)
ax_z_a.set_ylim(-0.1, V_cloth[0, 2] + 0.1)

v_lines = {v: ax_z_a.plot([], [], color=COLOR_V[v], lw=1.5,
                            label=f"V{v}")[0] for v in range(N_cloth)}
cur_dots = {v: ax_z_a.plot([], [], "o", color=COLOR_V[v], ms=5)[0]
            for v in range(N_cloth)}
ax_z_a.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="#ccccdd",
              edgecolor="#555577")


def _setup_3d_anim():
    ax_3d_a.set_xlim(-0.5, 1.5)
    ax_3d_a.set_ylim(-0.5, 1.5)
    ax_3d_a.set_zlim(-0.1, 1.2)
    ax_3d_a.set_xlabel("X", color="#aaaacc", fontsize=8)
    ax_3d_a.set_ylabel("Y", color="#aaaacc", fontsize=8)
    ax_3d_a.set_zlabel("Z", color="#aaaacc", fontsize=8)
    ax_3d_a.tick_params(colors="#888899")
    ax_3d_a.view_init(elev=30, azim=-55)


def _update_anim(fi: int):
    fd = frames_anim[fi]
    X  = fd["X"]
    contact_active = fd["contact_active"]
    detected       = fd["detected"]

    ax_3d_a.cla()
    _setup_3d_anim()
    draw_mesh(ax_3d_a, V_floor, T_floor,
              face_color="#f9e4b7", edge_color="#c0a060", alpha=0.5)

    cloth_fc = "#e74c3c" if contact_active else "#5dade2"
    cloth_ec = "#c0392b" if contact_active else "#2980b9"
    draw_mesh(ax_3d_a, X, T_cloth, face_color=cloth_fc, edge_color=cloth_ec, alpha=0.75)

    # Contact force arrows
    for v_idx, grad in fd["grad_arrows"]:
        force = -grad
        scale = 0.18 / (np.linalg.norm(force) + 1e-12)
        ax_3d_a.quiver(*X[v_idx], *(force * scale),
                       color="#f39c12", linewidth=2, arrow_length_ratio=0.35)

    # Conservative bound circles
    theta = np.linspace(0, 2 * np.pi, 40)
    for v_idx in range(N_cloth):
        bv = fd["b_v"].get(v_idx, 0.0)
        if bv > 1e-4:
            cx, cy, cz = X[v_idx]
            ax_3d_a.plot(cx + bv * np.cos(theta),
                         cy + bv * np.sin(theta),
                         np.full(40, cz),
                         color="#a29bfe", lw=0.8, alpha=0.5, ls="--")

    state = "[IN CONTACT]" if contact_active else ("[detection]" if detected else "")
    ax_3d_a.set_title(
        f"Algorithm 4 (VBD) — step {fd['outer']}, iter {fd['inner']} {state}\n"
        f"red=contacts  yellow=force  purple=bounds",
        fontsize=8, color="#ecf0f1", pad=5
    )

    # Right panel: update z curves
    z_data = fd["z_so_far"]
    n_pts  = len(z_data[0])
    xs     = list(range(n_pts))
    for v_idx in range(N_cloth):
        v_lines[v_idx].set_data(xs, z_data[v_idx])
        cur_dots[v_idx].set_data([xs[-1]], [z_data[v_idx][-1]])

    ax_z_a.set_title(f"z-positions  (frame {fd['frame_idx']})",
                     fontsize=9, color="#aaaacc")

    return list(v_lines.values()) + list(cur_dots.values())


ani = animation.FuncAnimation(
    fig_anim, _update_anim,
    frames=len(frames_anim),
    interval=100,
    blit=False,
    repeat=True,
)

fig_anim.suptitle(
    "Step 9 — Algorithm 4 (VBD) animated  |  Close window to continue",
    fontsize=10, color="#ecf0f1", y=1.01
)
plt.tight_layout()
plt.show()


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("ALGORITHM 4 — SUMMARY")
print("=" * 60)
print("""
  Algorithm 4 is the inner VBD solver that Algorithm 3 calls
  on every inner iteration.

  ┌──────────────────────────────────────────────────────────┐
  │  Line(s)   What happens                                  │
  ├──────────────────────────────────────────────────────────┤
  │  1         Outer loop over graph colors                  │
  │  2         Inner loop over vertices in that color        │
  │  3         Init with inertia term  f=−(m/h²)(x−y)       │
  │            H = (m/h²) I                                  │
  │  4-7       Add elastic (triangle) gradient & Hessian     │
  │  8-11      Add bending (edge) gradient & Hessian         │
  │            (M4: merged into mass-spring for simplicity)  │
  │  12-15     Add VF contact (vertex side, FOGC)            │
  │  16-21     Add VF contact (face side, VOGC)              │
  │  22-27     Add EE contact (EOGC)                         │
  │  28        Newton step:  x_v ← x_v + H_v⁻¹ f_v         │
  └──────────────────────────────────────────────────────────┘

  Key ideas:
  • Graph coloring ensures no two updated vertices share an element.
  • The 3×3 Newton step is the key: it converges quickly because
    both first and second derivative information is used.
  • PSD projection keeps H_v invertible even near contact.
  • Lines 16-21 (face-side) used FD in M4; M5 will use analytic
    gradients from the StVK cloth material.
""")
