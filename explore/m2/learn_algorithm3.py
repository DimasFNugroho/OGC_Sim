"""
Building Algorithm 3 from scratch — step by step
=================================================
Run this file:
    python3 explore/m2/learn_algorithm3.py

Each step pauses on a plot window.
Close the window → the next step runs.

You can add, change, or comment out anything.
The goal is for you to understand each line before moving on.

How Algorithm 3 relates to Algorithms 1 and 2
-----------------------------------------------
Algorithm 1 asked:  "is this VERTEX  close to any TRIANGLE?"
Algorithm 2 asked:  "is this EDGE    close to any other EDGE?"

Algorithm 3 asks:   "given one time-step, how do we move the mesh
                     safely without penetration?"

It is the OUTER LOOP that orchestrates Algorithms 1 & 2.
The full structure is:

  Algorithm 3: one simulation step
  ----------------------------------
  Inputs:  X_t  (positions from previous step)
           v_t  (velocities from previous step)
           a_ext (external acceleration, e.g. gravity)
           γ_e  (fraction of vertices that may exceed bound before re-detect)
           r, r_q (contact/query radii)

  1   collisionDetectionRequired = True
  2   X = X_t          ← start from previous positions
  3   Y = X_t + dt*v_t + dt²*a_ext  ← inertia target (free-fall guess)

  4   for i in 1 … n_iter:

  5     if collisionDetectionRequired:
  6-8     reset d_min_t → r_q for every triangle
  9-11    FOGC, d_min_v ← vertexFacetContactDetection  (Algorithm 1)
  12-14   EOGC, d_min_e ← edgeEdgeContactDetection    (Algorithm 2)
  15      X_prev = X
  16      collisionDetectionRequired = False
  17-19   b_v ← computeConservativeBound(v)            (Eq. 21)

  20    if i == 1:
  21      X ← applyInitialGuess(X_t, v_t, a_ext)      (Eq. 28)

  22    X ← simulationIteration(…)    ← one VBD pass  (Algorithm 4)

  23    numExceed = 0
  24-27 for each v: if ||x_v - x_prev_v|| > b_v → truncate, numExceed++
  28-29 if numExceed >= γ_e * K → collisionDetectionRequired = True

  30    [optional convergence check]

  34  return X
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
    activation_g, activation_dg_dd,
    contact_energy_vf, contact_gradient_v_vf,
)
from ogc_sim.contact.bounds     import (
    compute_conservative_bounds,
    truncate_displacements,
    apply_initial_guess_truncation,
)


def pause(title: str) -> None:
    plt.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.show()


def draw_mesh(ax, V: np.ndarray, T: np.ndarray,
              face_color: str = "#aed6f1",
              edge_color: str = "#2980b9",
              alpha: float = 0.35) -> None:
    """Draw all triangles of a mesh onto an existing 3D axis."""
    tris = [[V[T[i, 0]], V[T[i, 1]], V[T[i, 2]]] for i in range(len(T))]
    ax.add_collection3d(Poly3DCollection(
        tris, alpha=alpha, facecolor=face_color, edgecolor=edge_color, lw=1.0
    ))
    for v in V:
        ax.scatter(*v, color=edge_color, s=25, zorder=6)


# ============================================================
# Scene setup — a 3-vertex "L-shaped" cloth patch + a static
# floor triangle just below it.
#
#   cloth:  V0=(0,0,1), V1=(2,0,1), V2=(1,2,1)
#            → one triangle, will fall under gravity
#
#   floor:  V3=(−1,−1,0), V4=(3,−1,0), V5=(1,3,0)
#            → one static triangle, acts as obstacle
#
# This is intentionally minimal so that every plot is clear.
# ============================================================

V_cloth = np.array([
    [0.0, 0.0, 1.0],   # V0
    [2.0, 0.0, 1.0],   # V1
    [1.0, 2.0, 1.0],   # V2
])
T_cloth = np.array([[0, 1, 2]])

V_floor = np.array([
    [-1.0, -1.0, 0.0],  # V3
    [ 3.0, -1.0, 0.0],  # V4
    [ 1.0,  3.0, 0.0],  # V5
])
T_floor = np.array([[0, 1, 2]])

# Build a combined mesh (cloth + floor) for the BVH and detection
V_combined = np.vstack([V_cloth, V_floor])           # V0-V5
T_combined = np.array([[0, 1, 2], [3, 4, 5]])        # cloth T0, floor T1

mesh = Mesh.from_arrays(V_combined, T_combined)
pgm  = PolyhedralGaussMap(mesh)
bvh  = BVH(mesh)


# ============================================================
# STEP 1 — The inertia target Y
# ============================================================
# The very first thing Algorithm 3 does is compute Y,
# the "free-fall" prediction of where the cloth would be
# after one time-step if there were NO contact forces.
#
#   Y = X_t + dt * v_t + dt² * a_ext     (Algorithm 3, line 3)
#
# This is just Newton's kinematic law: if we know where we
# are (X_t) and how fast we are moving (v_t), extrapolate.
# a_ext is gravity: a_ext = [0, 0, -9.8]
#
# Y is NOT the answer.  It is the TARGET the solver tries
# to reach while respecting contact constraints.
# ============================================================

print("=" * 50)
print("STEP 1 — The inertia target Y")
print("=" * 50)

# Initial state of the cloth (just the 3 cloth vertices)
X_t = V_cloth.copy()
v_t = np.zeros_like(X_t)              # cloth starts at rest
a_ext = np.array([0.0, 0.0, -9.8])   # gravity pointing down

dt = 0.02   # time-step size in seconds

# Inertia target: Algorithm 3, line 3
Y = X_t + dt * v_t + dt**2 * a_ext

print(f"""
  dt      = {dt} s
  a_ext   = {a_ext}   (gravity)

  X_t (current positions):
    V0 = {X_t[0]}
    V1 = {X_t[1]}
    V2 = {X_t[2]}

  Y = X_t + dt*v_t + dt²*a_ext  (free-fall prediction):
    V0 → {np.round(Y[0], 4)}
    V1 → {np.round(Y[1], 4)}
    V2 → {np.round(Y[2], 4)}

  Each vertex drops {dt**2 * abs(a_ext[2]):.5f} m downward.
  Contact constraints will prevent falling all the way to Y.
""")

fig = plt.figure(figsize=(8, 6))
ax  = fig.add_subplot(111, projection="3d")

draw_mesh(ax, V_floor, T_floor, face_color="#f9e4b7", edge_color="#c0a060")
draw_mesh(ax, X_t, T_cloth, face_color="#aed6f1", edge_color="#2980b9")
draw_mesh(ax, Y,   T_cloth, face_color="#a9dfbf", edge_color="#1a8a50", alpha=0.2)

for i, (xt, y) in enumerate(zip(X_t, Y)):
    ax.plot([xt[0], y[0]], [xt[1], y[1]], [xt[2], y[2]],
            color="gray", lw=1.0, linestyle="--")
    ax.text(*xt + np.array([0.04, 0.04, 0.06]),
            f"V{i}  X_t", fontsize=8, color="#2980b9")
    ax.text(*y + np.array([0.04, 0.04, -0.1]),
            f"V{i}  Y", fontsize=8, color="#1a8a50")

ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_xlim(-0.5, 2.5); ax.set_ylim(-0.5, 2.5); ax.set_zlim(-0.2, 1.4)
ax.view_init(elev=28, azim=-55)
pause("Step 1 — The inertia target Y\n"
      "(blue = X_t, green = Y = free-fall prediction, orange = floor)")


# ============================================================
# STEP 2 — The outer iteration loop structure
# ============================================================
# Algorithm 3 runs n_iter solver iterations, each of which
# may or may not re-run contact detection.
#
# Key flag: collisionDetectionRequired
#   • Starts True  → detection always runs on the first iteration.
#   • Set to False after detection completes.
#   • Reset to True only when enough vertices have exceeded
#     their conservative bounds (line 29-30).
#
# This is the key insight: contact detection is expensive.
# We avoid re-running it every iteration by guaranteeing (via
# the conservative bound b_v) that no penetration can occur as
# long as every vertex stays within its allowed displacement.
#
# Algorithm 3, lines 1, 4-5, 15-16:
#   collisionDetectionRequired = True
#   for i in 1..n_iter:
#     if collisionDetectionRequired: detect, set X_prev = X, reset flag
# ============================================================

print("\n" + "=" * 50)
print("STEP 2 — Outer iteration loop structure")
print("=" * 50)

n_iter = 6
gamma_e = 0.1     # fraction threshold for re-detection

print(f"""
  n_iter  = {n_iter}   (total solver iterations per time-step)
  gamma_e = {gamma_e}  (re-detect when >= gamma_e * K vertices exceed bound)

  Pseudo-run to show which iterations would trigger detection:
""")

collision_detection_required = True
X_prev = None
detection_count = 0

for i in range(1, n_iter + 1):
    if collision_detection_required:
        detection_count += 1
        X_prev = None  # would be current X
        collision_detection_required = False
        print(f"  iter {i:2d}: *** CONTACT DETECTION runs ***  (detection #{detection_count})")
    else:
        print(f"  iter {i:2d}: no detection (flag = False)")

    # Simulate: suppose bound exceeded triggers re-detection at iter 3
    if i == 3:
        collision_detection_required = True
        print(f"           → {int(gamma_e * 3)} vertices exceeded bound → flag = True")

print(f"\n  Total detections in {n_iter} iterations: {detection_count}")
print(f"  Compare to naively detecting every iteration: {n_iter}")

# --- Plot: timeline of detections ---
fig, ax = plt.subplots(figsize=(9, 3))
iters = list(range(1, n_iter + 1))

detect_iters = [1, 4]  # detection happens at iter 1 and iter 4 (after flag reset at 3)
ax.barh([0] * n_iter, [1] * n_iter, left=iters,
        color=["#e74c3c" if i in detect_iters else "#aed6f1" for i in iters],
        edgecolor="white", height=0.4)

for i in iters:
    label = "DETECT" if i in detect_iters else "solve"
    color = "white" if i in detect_iters else "#333"
    ax.text(i + 0.5, 0, label, va="center", ha="center", fontsize=9, color=color)

ax.axvline(x=4, color="orange", lw=1.5, linestyle="--",
           label="flag reset (vertices exceeded bound)")
ax.set_xlim(1, n_iter + 1)
ax.set_yticks([])
ax.set_xticks(np.arange(1, n_iter + 1) + 0.5)
ax.set_xticklabels([f"iter {i}" for i in iters], fontsize=9)
ax.legend(fontsize=8, loc="upper right")
pause("Step 2 — Outer loop: red = contact detection, blue = solver only\n"
      "(detection skipped when vertices stay within conservative bounds)")


# ============================================================
# STEP 3 — Contact detection inside the loop (Algorithms 1 & 2)
# ============================================================
# When collisionDetectionRequired is True, the simulator:
#   (a) Resets all d_min_t to r_q           (lines 6-8)
#   (b) Runs Algorithm 1 for every vertex   (lines 9-11)
#   (c) Runs Algorithm 2 for every edge     (lines 12-14)
#
# This is just run_contact_detection() from detection.py.
# We already learned every detail in learn_algorithm1.py
# and learn_algorithm2.py.  Here we just call it.
#
# After detection:
#   X_prev = X                  (line 15) — snapshot current positions
#   collisionDetectionRequired = False   (line 16)
# ============================================================

print("\n" + "=" * 50)
print("STEP 3 — Contact detection inside the loop")
print("=" * 50)

r   = 0.35   # contact radius
r_q = 0.50   # query radius (> r)

print(f"\n  r = {r},  r_q = {r_q}")
print(f"\n  Cloth vertex positions (V0-V2) currently at z=1.0")
print(f"  Floor triangle       (V3-V5) sits at z=0.0")
print(f"  Cloth-to-floor distance ≈ 1.0 — well outside r={r}, so no contacts yet.\n")

cs_initial = run_contact_detection(mesh, bvh, pgm, r, r_q)

print(f"  Contacts at z=1.0:")
any_contact = False
for v_idx in range(3):    # only cloth vertices
    fogc = cs_initial.FOGC.get(v_idx, [])
    d_min = cs_initial.d_min_v.get(v_idx, r_q)
    if fogc:
        any_contact = True
    print(f"    V{v_idx}: FOGC={fogc}  d_min_v={d_min:.4f}")

if not any_contact:
    print(f"  → No contacts (cloth too far from floor).")

# Now move cloth down close to the floor
V_close = V_combined.copy()
V_close[0:3, 2] = 0.25   # cloth vertices now at z=0.25 (within r_q=0.50 of floor)

mesh_close = Mesh.from_arrays(V_close, T_combined)
pgm_close  = PolyhedralGaussMap(mesh_close)
bvh_close  = BVH(mesh_close)

cs_close = run_contact_detection(mesh_close, bvh_close, pgm_close, r, r_q)

print(f"\n  After moving cloth down to z=0.25:")
for v_idx in range(3):
    fogc  = cs_close.FOGC.get(v_idx, [])
    d_min = cs_close.d_min_v.get(v_idx, r_q)
    print(f"    V{v_idx}: FOGC={fogc}  d_min_v={d_min:.4f}")

# --- Plot: before and after ---
fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(12, 5))

for ax, V_cloth_pos, cs, title_suffix in [
    (ax1, V_combined[0:3], cs_initial, "z=1.0  (no contacts)"),
    (ax2, V_close[0:3],    cs_close,   "z=0.25  (within r_q)"),
]:
    V_full = np.vstack([V_cloth_pos, V_floor])
    draw_mesh(ax, V_floor, T_floor, face_color="#f9e4b7", edge_color="#c0a060")
    draw_mesh(ax, V_cloth_pos, T_cloth, face_color="#aed6f1", edge_color="#2980b9")

    for v_idx in range(3):
        d_min = cs.d_min_v.get(v_idx, r_q)
        fogc  = cs.FOGC.get(v_idx, [])
        color = "#e74c3c" if fogc else "#2980b9"
        ax.text(*V_cloth_pos[v_idx] + np.array([0.04, 0.04, 0.05]),
                f"V{v_idx}\nd={d_min:.2f}", fontsize=7, color=color)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_xlim(-1.5, 3.5); ax.set_ylim(-1.5, 3.5); ax.set_zlim(-0.2, 1.4)
    ax.view_init(elev=28, azim=-55)
    ax.set_title(f"Cloth at {title_suffix}", fontsize=9)

pause("Step 3 — Contact detection inside the loop\n"
      "(d_min_v reported for each vertex; red labels = contact detected)")


# ============================================================
# STEP 4 — Conservative bound b_v (Eq. 21)
# ============================================================
# After detection, the simulator computes a safe displacement
# bound for each vertex:
#
#   b_v = γ_p * min(d_min_v, d_min_e_v, d_min_t_v)   Eq. 21
#
# where γ_p ∈ (0, 0.5) (paper uses 0.45).
#
# Meaning: vertex v may move at most b_v from its CURRENT
# position (X_prev) before the next contact detection is
# needed.  As long as everyone stays within their b_v, NO
# penetration can occur — that is the key theorem.
#
# In this script we use only d_min_v (the vertex-to-face
# minimum distance) to illustrate the concept.  The full
# formula also incorporates d_min_e_v and d_min_t_v (Eq. 22-26).
#
# Algorithm 3, lines 17-19:
#   parallel for each v ∈ V do
#     b_v = computeConservativeBound(v)   ← Eq. 21
# ============================================================

print("\n" + "=" * 50)
print("STEP 4 — Conservative bound b_v  (Eq. 21)")
print("=" * 50)

gamma_p = 0.45   # paper uses 0.45

# Use the "close" positions from Step 3
V_cloth_now = V_close[0:3].copy()

# Real M3 implementation — compute_conservative_bounds uses all three
# d_min values (Eq. 21-26): d_min_v, d_min_e_v (via neighbour edges),
# d_min_t_v (via neighbour faces).  The result is stored in b_v dict.
b_v = compute_conservative_bounds(mesh_close, cs_close, gamma_p)
# Keep only the cloth vertices (indices 0-2)
b_v = {v: b_v[v] for v in range(3)}

print(f"\n  γ_p = {gamma_p}  (Eq. 21 relaxation parameter)")
print(f"\n  For each cloth vertex at z=0.25:")
for v_idx in range(3):
    d_min_v   = cs_close.d_min_v.get(v_idx, r_q)
    d_min_e_v = min((cs_close.d_min_e.get(e, r_q) for e in mesh_close.E_v[v_idx]),
                    default=r_q)
    d_min_t_v = min((cs_close.d_min_t.get(t, r_q) for t in mesh_close.T_v[v_idx]),
                    default=r_q)
    d_min_all = min(d_min_v, d_min_e_v, d_min_t_v)
    print(f"    V{v_idx}: d_min_v={d_min_v:.4f}  d_min_e_v={d_min_e_v:.4f}"
          f"  d_min_t_v={d_min_t_v:.4f}  → min={d_min_all:.4f}"
          f"  b_v={b_v[v_idx]:.4f}")

print(f"""
  Interpretation:
    Each cloth vertex may move at most b_v ≈ {b_v[0]:.3f} m from its current
    position before the next contact detection is needed.

  Why γ_p < 0.5?
    The paper proves that as long as two vertices each move < 0.5 * d_min,
    their combined displacement cannot close the gap entirely.
    γ_p = 0.45 adds a small safety margin for floating-point rounding.
""")

# --- Plot: b_v spheres ---
fig = plt.figure(figsize=(8, 6))
ax  = fig.add_subplot(111, projection="3d")

draw_mesh(ax, V_floor, T_floor, face_color="#f9e4b7", edge_color="#c0a060")
draw_mesh(ax, V_cloth_now, T_cloth, face_color="#aed6f1", edge_color="#2980b9")

# Draw b_v spheres (wireframe)
u  = np.linspace(0, 2 * np.pi, 24)
vv = np.linspace(0, np.pi, 12)
for v_idx in range(3):
    pt = V_cloth_now[v_idx]
    bv = b_v[v_idx]
    ax.plot_wireframe(
        bv * np.outer(np.cos(u), np.sin(vv)) + pt[0],
        bv * np.outer(np.sin(u), np.sin(vv)) + pt[1],
        bv * np.outer(np.ones(24), np.cos(vv)) + pt[2],
        color="#27ae60", alpha=0.2, linewidth=0.5
    )
    ax.scatter(*pt, color="#2980b9", s=50, zorder=9)
    ax.text(*pt + np.array([0.04, 0.04, 0.05]),
            f"V{v_idx}\nb_v={bv:.3f}", fontsize=8, color="#27ae60")

ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_xlim(-1.5, 3.5); ax.set_ylim(-1.5, 3.5); ax.set_zlim(-0.3, 0.6)
ax.view_init(elev=28, azim=-55)
pause("Step 4 — Conservative bound b_v  (Eq. 21)\n"
      "(green sphere = safe displacement region for each vertex)")


# ============================================================
# STEP 5 — Initial guess truncation  (Eq. 28)
# ============================================================
# On the FIRST iteration only (i == 1), we can apply a better
# initial guess than just X_t to improve convergence.
# A natural choice is the inertia position Y.
#
# But Y might move a vertex OUTSIDE its conservative bound!
# So we truncate:
#
#   if ||X_init_v - X_prev_v|| <= b_v  →  keep X_init_v
#   else                               →  clamp to distance b_v
#                                         along the direction
#                                         (X_init_v - X_prev_v)  (Eq. 28)
#
# Algorithm 3, lines 20-21:
#   if i == 1:
#     X = applyInitialGuess(X_t, v_t, a_ext)   ← Eq. 28 applied here
# ============================================================

print("\n" + "=" * 50)
print("STEP 5 — Initial guess truncation  (Eq. 28)")
print("=" * 50)

# X_prev = cloth positions snapshot taken at detection time
X_prev_cloth = V_cloth_now.copy()   # cloth at z=0.25

# The initial guess: here we use a modest inertia extrapolation
# (pretend the cloth was moving downward with velocity 0.5 m/s)
v_guess = np.array([0.0, 0.0, -0.5])
X_init  = X_prev_cloth + dt * v_guess + dt**2 * a_ext

print(f"\n  X_prev (detection snapshot):  z = {X_prev_cloth[0, 2]:.3f}")
print(f"  X_init (inertia guess):       z = {X_init[0, 2]:.4f}")
print(f"  Displacement magnitude:       {np.linalg.norm(X_init[0] - X_prev_cloth[0]):.4f}")
print(f"  b_v[0] = {b_v[0]:.4f}")

# Real M3 implementation — apply_initial_guess_truncation (Eq. 28)
X_init_truncated = apply_initial_guess_truncation(X_init, X_prev_cloth, b_v)

for v_idx in range(3):
    delta      = X_init[v_idx] - X_prev_cloth[v_idx]
    delta_norm = np.linalg.norm(delta)
    bv         = b_v[v_idx]
    if delta_norm > bv:
        print(f"\n  V{v_idx}: ||delta|| = {delta_norm:.4f} > b_v = {bv:.4f}"
              f"  → TRUNCATED to z = {X_init_truncated[v_idx, 2]:.4f}")
    else:
        print(f"\n  V{v_idx}: ||delta|| = {delta_norm:.4f} <= b_v = {bv:.4f}"
              f"  → kept as-is  (z = {X_init_truncated[v_idx, 2]:.4f})")

# --- Plot: before / after truncation ---
fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(12, 5))

for ax, X_plot, title in [
    (ax1, X_init,           "Before truncation  (X_init)"),
    (ax2, X_init_truncated, "After truncation  (Eq. 28)"),
]:
    draw_mesh(ax, V_floor, T_floor, face_color="#f9e4b7", edge_color="#c0a060")
    draw_mesh(ax, X_prev_cloth, T_cloth,
              face_color="#aed6f1", edge_color="#2980b9", alpha=0.4)
    draw_mesh(ax, X_plot, T_cloth,
              face_color="#a9dfbf", edge_color="#1a8a50", alpha=0.6)

    # Draw b_v sphere for V0 only (to keep the plot clear)
    bv = b_v[0]
    pt = X_prev_cloth[0]
    ax.plot_wireframe(
        bv * np.outer(np.cos(u), np.sin(vv)) + pt[0],
        bv * np.outer(np.sin(u), np.sin(vv)) + pt[1],
        bv * np.outer(np.ones(24), np.cos(vv)) + pt[2],
        color="#27ae60", alpha=0.15, linewidth=0.4
    )

    # Draw arrows: X_prev → X_plot
    for v_idx in range(3):
        src = X_prev_cloth[v_idx]
        dst = X_plot[v_idx]
        ax.quiver(*src, *(dst - src), color="#e74c3c", arrow_length_ratio=0.3,
                  lw=1.5, length=1.0, normalize=False)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_xlim(-1.5, 3.5); ax.set_ylim(-1.5, 3.5); ax.set_zlim(-0.4, 0.6)
    ax.view_init(elev=28, azim=-55)
    ax.set_title(title, fontsize=9)

pause("Step 5 — Initial guess truncation  (Eq. 28)\n"
      "(blue = X_prev, green = initial guess, green sphere = b_v safe region,\n"
      " red arrows = displacement, CLIPPED to b_v if out of bounds)")


# ============================================================
# STEP 6 — Simulation iteration  (Algorithm 4 stub)
# ============================================================
# After the initial guess, each iteration calls:
#
#   X = simulationIteration(FOGC, VOGC, EOGC, X, X_t, Y, …)
#
# This is Algorithm 4 (VBD iteration with contact).
# We have NOT yet implemented VBD.  For this script we use
# a simple placeholder: move each cloth vertex toward Y by
# a small step (simulating one gradient-descent step).
#
# The key is NOT what the solver does internally, but what
# happens AFTER: bound truncation (Step 7).
#
# Algorithm 3, line 22:
#   X = simulationIteration(…)
# ============================================================

print("\n" + "=" * 50)
print("STEP 6 — Simulation iteration with real contact forces")
print("=" * 50)

# Now we use real M3 contact energy and gradient.
# We implement ONE simplified gradient-descent step:
#   x_v ← x_v − h * ∇_v E_contact(x_v)
#
# (Real VBD in Algorithm 4 also includes inertia and elastic terms.)
# This shows that the contact gradient pushes cloth vertices AWAY
# from the floor, exactly as the energy intends.

k_c_step6 = 1e4   # contact stiffness
h_step6   = 1e-6  # gradient step size

X_current = X_init_truncated.copy()
Y_cloth    = Y.copy()

# Compute contact forces on each cloth vertex from the floor face
V_scene_6  = np.vstack([X_current, V_floor])
mesh_6     = Mesh.from_arrays(V_scene_6, T_combined)

floor_a = V_floor[0]; floor_b = V_floor[1]; floor_c = V_floor[2]

print(f"\n  Cloth at z={X_current[0, 2]:.4f}  (r={r:.3f},  k_c={k_c_step6:.0e})")
print(f"\n  Contact energy and gradient per cloth vertex:")

X_after_grad = X_current.copy()
for v_idx in range(3):
    v_pos = X_current[v_idx]
    E_contact = contact_energy_vf(v_pos, floor_a, floor_b, floor_c, r, k_c_step6)
    g_contact = contact_gradient_v_vf(v_pos, floor_a, floor_b, floor_c, r, k_c_step6)
    X_after_grad[v_idx] = v_pos - h_step6 * g_contact   # gradient descent step

    print(f"    V{v_idx}  E={E_contact:.3f}  "
          f"∇E_z={g_contact[2]:.3f}  "
          f"(negative = force pushes UP, away from floor)")

print(f"""
  The contact gradient points FROM the floor TOWARD the cloth vertex.
  Since E = g(d, r) and dg/dd < 0 (energy decreases as d grows),
  the force −∇E pushes the cloth vertex away from the floor.

  In real VBD (Algorithm 4), this contact force is combined with
  the inertia and elastic forces to compute the net displacement
  for each vertex per iteration.
""")

# --- Plot ---
fig = plt.figure(figsize=(8, 6))
ax  = fig.add_subplot(111, projection="3d")

draw_mesh(ax, V_floor, T_floor, face_color="#f9e4b7", edge_color="#c0a060")
draw_mesh(ax, X_prev_cloth, T_cloth,
          face_color="#aed6f1", edge_color="#2980b9", alpha=0.3)
draw_mesh(ax, X_current, T_cloth,
          face_color="#d2b4de", edge_color="#7d3c98", alpha=0.4)

# Draw contact force arrows
for v_idx in range(3):
    v_pos    = X_current[v_idx]
    g_vec    = contact_gradient_v_vf(v_pos, floor_a, floor_b, floor_c, r, k_c_step6)
    force    = -g_vec   # force = -gradient
    scale    = 0.3 / (np.linalg.norm(force) + 1e-12)
    ax.quiver(*v_pos, *(force * scale), color="#e74c3c", arrow_length_ratio=0.3,
              lw=2, length=1.0, normalize=False)

# Legend proxies
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
ax.legend(handles=[
    Patch(facecolor="#aed6f1", label="X_prev"),
    Patch(facecolor="#d2b4de", label="X_init* (truncated guess)"),
    Line2D([0],[0], color="#e74c3c", lw=2, label="contact force −∇E"),
], fontsize=8, loc="upper left")

ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_xlim(-1.5, 3.5); ax.set_ylim(-1.5, 3.5); ax.set_zlim(-0.4, 0.6)
ax.view_init(elev=28, azim=-55)
pause("Step 6 — Real contact forces from M3 energy\n"
      "(red arrows = −∇E_contact, pushing cloth away from floor)")


# ============================================================
# STEP 7 — Bound truncation after each iteration  (lines 23-28)
# ============================================================
# After the solver moves the mesh, we check how far each vertex
# has moved FROM X_prev (the position at the last detection).
#
#   Δx_v = x_v − x_prev_v
#   if ||Δx_v|| > b_v:
#     x_v ← (Δx_v / ||Δx_v||) * b_v + x_prev_v   ← project back
#     numVerticesExceedBound++
#
# This is NOT the same as the initial-guess truncation.
# The initial guess is applied ONCE on the first iteration.
# THIS truncation is applied EVERY iteration, measuring the
# total displacement from the last detection snapshot X_prev.
#
# Algorithm 3, lines 23-28.
# ============================================================

print("\n" + "=" * 50)
print("STEP 7 — Bound truncation after each iteration  (lines 23-28)")
print("=" * 50)

# Simulate: X_prev is where detection occurred (z=0.25),
#           X_after_iter is where the gradient step moved us.
X_prev_step = X_prev_cloth.copy()
X_solver    = X_after_grad.copy()   # result of Step 6 gradient step

# Real M3 implementation — truncate_displacements (Algorithm 3 lines 23-28)
X_truncated, num_exceed = truncate_displacements(X_solver, X_prev_step, b_v)

print(f"\n  X_prev (detection snapshot): z = {X_prev_step[0, 2]:.4f}")
print(f"\n  Checking each vertex after iteration:")
for v_idx in range(3):
    delta      = X_solver[v_idx] - X_prev_step[v_idx]
    delta_norm = np.linalg.norm(delta)
    bv         = b_v[v_idx]
    if delta_norm > bv:
        print(f"  V{v_idx}: ||Δx|| = {delta_norm:.4f} > b_v = {bv:.4f}"
              f"  → TRUNCATED to z = {X_truncated[v_idx, 2]:.4f}")
    else:
        print(f"  V{v_idx}: ||Δx|| = {delta_norm:.4f} <= b_v = {bv:.4f}"
              f"  → kept as-is  z = {X_truncated[v_idx, 2]:.4f}")

K = 3   # total cloth vertices
threshold = int(np.ceil(gamma_e * K))
re_detect = num_exceed >= threshold
print(f"""
  numVerticesExceedBound = {num_exceed}
  threshold              = ceil(γ_e * K) = ceil({gamma_e} * {K}) = {threshold}
  collisionDetectionRequired = {re_detect}
  (Algorithm 3 line 29:  if numExceed >= γ_e * K → re-detect)
""")

# --- Plot: before / after truncation ---
fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(12, 5))

for ax, X_plot, title in [
    (ax1, X_solver,    "After solver  (may exceed b_v)"),
    (ax2, X_truncated, "After bound truncation"),
]:
    draw_mesh(ax, V_floor, T_floor, face_color="#f9e4b7", edge_color="#c0a060")
    draw_mesh(ax, X_prev_step, T_cloth,
              face_color="#aed6f1", edge_color="#2980b9", alpha=0.3)
    draw_mesh(ax, X_plot, T_cloth,
              face_color="#a9dfbf", edge_color="#1a8a50", alpha=0.7)

    for v_idx in range(3):
        bv = b_v[v_idx]
        pt = X_prev_step[v_idx]
        ax.plot_wireframe(
            bv * np.outer(np.cos(u), np.sin(vv)) + pt[0],
            bv * np.outer(np.sin(u), np.sin(vv)) + pt[1],
            bv * np.outer(np.ones(24), np.cos(vv)) + pt[2],
            color="#27ae60", alpha=0.12, linewidth=0.4
        )

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_xlim(-1.5, 3.5); ax.set_ylim(-1.5, 3.5); ax.set_zlim(-0.4, 0.6)
    ax.view_init(elev=28, azim=-55)
    ax.set_title(title, fontsize=9)

pause("Step 7 — Bound truncation  (Algorithm 3 lines 23-28)\n"
      "(blue = X_prev, green mesh = cloth, green sphere = b_v bound)")


# ============================================================
# STEP 8 — Re-detection threshold γ_e  (line 29-30)
# ============================================================
# Re-running contact detection every iteration is expensive.
# Algorithm 3 uses a threshold γ_e:
#
#   if numVerticesExceedBound >= γ_e * K → collisionDetectionRequired = True
#
# γ_e = 0 means: detect every time ANY vertex exceeds bound.
# γ_e = 1 means: detect only when ALL vertices have exceeded.
# Paper uses γ_e = 0.01 (1% of vertices).
#
# Why not always re-detect?
#   Vertices that exceed their bound get TRUNCATED.
#   They cannot cause penetration — just stuck at the boundary.
#   As long as fewer than γ_e * K vertices are stuck, the
#   overall solver convergence is not seriously affected.
#   Only when many vertices pile up at their bound do we
#   re-detect to get fresh, tighter bounds.
# ============================================================

print("\n" + "=" * 50)
print("STEP 8 — Re-detection threshold γ_e")
print("=" * 50)

K_total = 100    # pretend we have 100 cloth vertices
gamma_e_values  = [0.0, 0.01, 0.1, 0.5]

print(f"\n  K = {K_total} vertices.  Suppose 5 vertices exceed their bound.\n")
print(f"  {'γ_e':>8}  {'threshold':>12}  {'5 exceed → re-detect?':>22}")
print(f"  {'-'*8}  {'-'*12}  {'-'*22}")
for g in gamma_e_values:
    thresh = g * K_total
    detect = 5 >= thresh
    print(f"  {g:>8.2f}  {thresh:>12.1f}  {'YES' if detect else 'no':>22}")

print(f"""
  The paper uses γ_e = 0.01, meaning: re-detect once 1% of all
  vertices have been truncated.  This strikes a good balance:
    • not too frequent (expensive detection)
    • not too rare    (vertices pile up at boundary, solver stalls)
""")

# --- Plot: detection frequency vs γ_e ---
import numpy as np

n_exceed_range = np.arange(0, K_total + 1)
fig, ax = plt.subplots(figsize=(8, 4))

for g, color in [(0.0, "#e74c3c"), (0.01, "#e67e22"),
                 (0.1, "#3498db"), (0.5, "#2ecc71")]:
    threshold_line = np.full_like(n_exceed_range, g * K_total, dtype=float)
    ax.plot(n_exceed_range, threshold_line, color=color, lw=2,
            linestyle="--", label=f"γ_e = {g:.2f}  (thresh = {g*K_total:.0f})")

ax.axvline(x=5, color="purple", lw=1.5, linestyle=":", label="5 vertices exceed (example)")
ax.fill_between([5, K_total], 0, K_total,
                color="#f9ebea", label="re-detect region for γ_e=0.01")

ax.set_xlabel("numVerticesExceedBound", fontsize=10)
ax.set_ylabel("threshold = γ_e × K", fontsize=10)
ax.set_xlim(0, K_total); ax.set_ylim(0, K_total * 0.6)
ax.legend(fontsize=8)
pause("Step 8 — Re-detection threshold γ_e\n"
      "(detect when curve crosses the purple vertical line)")


# ============================================================
# STEP 9 — Full Algorithm 3 loop wired together
# ============================================================
# Now we put all the pieces together and run Algorithm 3 over
# MULTIPLE TIME STEPS, then watch it unfold as an animation.
#
# Simplified version (no real VBD):
#   - stub_simulation_iteration moves cloth toward Y each inner iter
#   - conservative bound uses real compute_conservative_bounds (Eq. 21)
#   - detection uses run_contact_detection from detection.py
#
# The animation shows:
#   LEFT  — 3-D cloth mesh descending toward the floor;
#            contact force arrows appear once d < r;
#            red flash on the cloth title when detection fires.
#   RIGHT — live z-position curve for V0, updated each inner
#            iteration; orange dotted = contact onset; red dashes
#            mark every time contact detection re-fires.
#
# Watch:
#   • The cloth advances freely until it enters the contact zone (z≈r)
#   • Once inside r the force arrows appear and the descent slows
#   • Every detection event (red dashes) resets X_prev and b_v
#   • The cloth eventually hovers just above the floor
# ============================================================

print("\n" + "=" * 50)
print("STEP 9 — Full Algorithm 3 animated over many time steps")
print("=" * 50)

# ---- Scene parameters ---------------------------------------
T_all        = np.array([[0, 1, 2], [3, 4, 5]])
r_loop       = 0.35
r_q_loop     = 0.50
gamma_p_loop = 0.45
gamma_e_loop = 0.0    # re-detect whenever any vertex exceeds bound
dt_loop      = 0.04
n_outer      = 25     # time steps to simulate
n_inner      = 5      # inner VBD iterations per time step

# Initial state
v_start  = np.array([0.0, 0.0, -1.2])
X_state  = V_cloth.copy()
v_state  = np.tile(v_start, (3, 1))
a_ext    = np.array([0.0, 0.0, -9.8])

print(f"\n  Simulating {n_outer} time steps × {n_inner} inner iterations")
print(f"  dt={dt_loop}  r={r_loop}  r_q={r_q_loop}  γ_p={gamma_p_loop}")

# ---- Pre-compute all frames for the animation ---------------
# Each frame = one inner iteration of one outer step.
# We record everything needed to draw that frame.

frames = []         # list of dicts, one per inner iteration

# Global z-history across ALL inner iterations (x-axis for the curve)
all_z       = [X_state[0, 2]]
all_frame_i = [0]   # cumulative inner iteration index
detection_frame_idx = []  # which frame indices had detection

frame_idx = 0

X_cur = X_state.copy()
v_cur = v_state.copy()

for outer in range(n_outer):
    X_t = X_cur.copy()
    Y   = X_t + dt_loop * v_cur + dt_loop**2 * a_ext

    collision_detection_required = True
    X_prev = X_t.copy()
    b_v    = {v: r_q_loop for v in range(3)}
    cs_saved = None

    for inner in range(1, n_inner + 1):
        frame_idx += 1
        detected_this_iter = False

        if collision_detection_required:
            V_scene = np.vstack([X_cur, V_floor])
            mesh_i  = Mesh.from_arrays(V_scene, T_all)
            pgm_i   = PolyhedralGaussMap(mesh_i)
            bvh_i   = BVH(mesh_i)
            cs_i    = run_contact_detection(mesh_i, bvh_i, pgm_i, r_loop, r_q_loop)
            cs_saved = cs_i

            X_prev = X_cur.copy()
            collision_detection_required = False
            detected_this_iter = True
            detection_frame_idx.append(frame_idx)

            b_v_all = compute_conservative_bounds(mesh_i, cs_i, gamma_p_loop)
            b_v     = {v: b_v_all[v] for v in range(3)}

        # Contact is active when at least one cloth vertex is inside radius r
        # (cs_saved is not None and has non-empty FOGC entries)
        contact_active = (
            cs_saved is not None and
            any(len(cs_saved.FOGC.get(v, [])) > 0 for v in range(3))
        )

        # Initial guess truncation on first inner iter
        if inner == 1:
            X_cur = apply_initial_guess_truncation(Y.copy(), X_prev, b_v)

        # Stub VBD: move 30 % of the way toward Y
        X_cur = X_cur + 0.3 * (Y - X_cur)

        # Truncate displacements
        X_cur, num_exceed = truncate_displacements(X_cur, X_prev, b_v)

        K_loop = 3
        if num_exceed >= gamma_e_loop * K_loop and num_exceed > 0:
            collision_detection_required = True

        # Compute contact force arrows (gradient of energy at cloth verts)
        grad_arrows = []
        if cs_saved is not None:
            for v_idx in range(3):
                V_scene = np.vstack([X_cur, V_floor])
                g = contact_gradient_v_vf(
                    X_cur[v_idx],
                    V_floor[0], V_floor[1], V_floor[2],
                    r_loop, k_c=500.0,
                )
                if np.linalg.norm(g) > 1e-8:
                    grad_arrows.append((v_idx, g))

        all_z.append(X_cur[0, 2])
        all_frame_i.append(frame_idx)

        frames.append({
            "X":             X_cur.copy(),
            "b_v":           dict(b_v),
            "detected":      detected_this_iter,
            "contact_active": contact_active,
            "grad_arrows":   list(grad_arrows),
            "outer":         outer + 1,
            "inner":         inner,
            "frame_idx":     frame_idx,
            "z_so_far":   list(all_z),
            "fi_so_far":  list(all_frame_i),
        })

    # Update velocity (simple Euler for the learning demo)
    v_cur = (X_cur - X_t) / dt_loop

print(f"  Pre-computed {len(frames)} frames ({n_outer} steps × {n_inner} iters)")
print(f"  Final cloth z: {X_cur[0, 2]:.4f}  (floor z=0,  contact onset z≈r={r_loop})")

# ---- Build the figure for the animation ---------------------
fig_anim = plt.figure(figsize=(14, 6))
fig_anim.patch.set_facecolor("#1a1a2e")

# Left — 3-D cloth view
ax3d = fig_anim.add_subplot(1, 2, 1, projection="3d")
ax3d.set_facecolor("#1a1a2e")
for spine in ax3d.spines.values():
    spine.set_color("#555577")

# Right — z-position history
ax_z2 = fig_anim.add_subplot(1, 2, 2)
ax_z2.set_facecolor("#0f0f1e")
ax_z2.tick_params(colors="#aaaacc")
for spine in ax_z2.spines.values():
    spine.set_edgecolor("#555577")

ax_z2.axhline(y=r_loop, color="orange",  lw=1.5, linestyle=":", label=f"contact onset r={r_loop}")
ax_z2.axhline(y=0.0,    color="#888899", lw=1.0, linestyle="-",  label="floor z=0")
ax_z2.set_xlabel("Cumulative inner iteration", fontsize=9, color="#aaaacc")
ax_z2.set_ylabel("V0  z-position",             fontsize=9, color="#aaaacc")
ax_z2.set_xlim(0, len(frames) + 1)
ax_z2.set_ylim(-0.1, all_z[0] + 0.1)
ax_z2.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#555577", labelcolor="#ccccdd")

# Initial curve and current-position dot
(curve_line,)    = ax_z2.plot([], [], color="#5dade2", lw=1.5, label="V0 z")
(cur_dot,)       = ax_z2.plot([], [], "o", color="#f5cba7", ms=6, zorder=5)
# Detection markers (appended dynamically — we'll draw as vlines each frame)
det_vlines = []


def _setup_3d_axes() -> None:
    ax3d.set_xlim(-1.5, 3.5)
    ax3d.set_ylim(-1.5, 3.5)
    ax3d.set_zlim(-0.3, 1.3)
    ax3d.set_xlabel("X", color="#aaaacc", fontsize=8)
    ax3d.set_ylabel("Y", color="#aaaacc", fontsize=8)
    ax3d.set_zlabel("Z", color="#aaaacc", fontsize=8)
    ax3d.tick_params(colors="#888899")
    ax3d.view_init(elev=28, azim=-55)


_setup_3d_axes()

# Draw the static floor once
draw_mesh(ax3d, V_floor, T_floor, face_color="#f9e4b7", edge_color="#c0a060", alpha=0.6)


def _update(frame_num: int):
    """Called by FuncAnimation for each frame."""
    fd = frames[frame_num]
    X  = fd["X"]
    detected       = fd["detected"]
    contact_active = fd["contact_active"]
    outer          = fd["outer"]
    inner          = fd["inner"]

    # ---- 3-D panel: clear and redraw cloth -------------------
    ax3d.cla()
    _setup_3d_axes()

    # Floor (static)
    draw_mesh(ax3d, V_floor, T_floor, face_color="#f9e4b7", edge_color="#c0a060", alpha=0.6)

    # Cloth colour: red = contacts are active (d < r); blue = free fall
    cloth_fc = "#e74c3c" if contact_active else "#5dade2"
    cloth_ec = "#c0392b" if contact_active else "#2980b9"
    draw_mesh(ax3d, X, T_cloth, face_color=cloth_fc, edge_color=cloth_ec, alpha=0.75)

    # Contact force arrows (contact gradient ← pushes cloth away from floor)
    for v_idx, grad in fd["grad_arrows"]:
        origin = X[v_idx]
        # gradient points in the direction of increasing energy (into floor);
        # the force on the cloth is −∇E, pointing away from the floor
        force_dir = -grad
        scale = 0.15 / (np.linalg.norm(force_dir) + 1e-12)
        ax3d.quiver(*origin,
                    *(force_dir * scale),
                    color="#f39c12", linewidth=2.0, arrow_length_ratio=0.4)

    # Conservative bound spheres (simplified: circles in XY plane)
    theta = np.linspace(0, 2 * np.pi, 40)
    for v_idx in range(3):
        bv = fd["b_v"].get(v_idx, 0.0)
        if bv > 1e-4:
            cx, cy, cz = X[v_idx]
            xs = cx + bv * np.cos(theta)
            ys = cy + bv * np.sin(theta)
            zs = np.full_like(theta, cz)
            ax3d.plot(xs, ys, zs, color="#a29bfe", lw=0.8, alpha=0.5, linestyle="--")

    # Vertex labels
    for v_idx in range(3):
        offset = np.array([0.06, 0.06, 0.06])
        ax3d.text(*(X[v_idx] + offset),
                  f"V{v_idx} z={X[v_idx,2]:.3f}", fontsize=7, color="#ecf0f1")

    state_label = "  [IN CONTACT]" if contact_active else ("  [detection ran]" if detected else "")
    ax3d.set_title(
        f"Algorithm 3 — step {outer}, inner iter {inner}{state_label}\n"
        f"red=contacts active  yellow=force  purple=bound",
        fontsize=8, color="#ecf0f1", pad=6
    )

    # ---- Right panel: live z-curve ---------------------------
    z_so_far  = fd["z_so_far"]
    fi_so_far = fd["fi_so_far"]
    curve_line.set_data(fi_so_far, z_so_far)
    cur_dot.set_data([fi_so_far[-1]], [z_so_far[-1]])

    # Add a vertical red line whenever detection fired (once per frame)
    if detected:
        vl = ax_z2.axvline(x=fd["frame_idx"], color="#e74c3c",
                           lw=0.8, linestyle="--", alpha=0.6)
        det_vlines.append(vl)

    ax_z2.set_title(f"V0 z-position  (frame {fd['frame_idx']})",
                    fontsize=9, color="#aaaacc")

    return [curve_line, cur_dot]


ani = animation.FuncAnimation(
    fig_anim,
    _update,
    frames=len(frames),
    interval=120,     # ms between frames
    blit=False,
    repeat=True,
)

fig_anim.suptitle(
    "Step 9 — Algorithm 3 animated\n"
    "Close window to continue",
    fontsize=10, color="#ecf0f1", y=1.01
)
plt.tight_layout()
plt.show()


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("ALGORITHM 3 — SUMMARY")
print("=" * 60)
print("""
  Algorithm 3 is the outer time-step loop.  It ties together:

  ┌─────────────────────────────────────────────────────────┐
  │  Line       What happens                                │
  ├─────────────────────────────────────────────────────────┤
  │  3          Compute inertia target Y (Step 1)           │
  │  5-16       Contact detection (Alg 1 + 2) when needed   │
  │             (Step 3)                                    │
  │  17-19      Compute conservative bounds b_v  (Step 4)   │
  │  20-21      Truncate initial guess to b_v  (Step 5)     │
  │  22         One VBD iteration  (Algorithm 4)  (Step 6)  │
  │  23-28      Truncate displacement to b_v  (Step 7)      │
  │  29-30      Re-detect if γ_e * K vertices exceeded      │
  │             (Step 8)                                    │
  └─────────────────────────────────────────────────────────┘

  Key guarantees:
  • As long as every vertex stays within its b_v ball,
    no penetration can occur  (Theorem, Section 3.8).
  • Contact detection is NOT needed every iteration —
    only when bounds are exhausted.
  • The method works with any solver (VBD, Newton, …) that
    produces per-vertex displacements.
""")
