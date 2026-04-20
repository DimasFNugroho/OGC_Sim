"""
Example: Algorithm 2 — Edge-Edge Contact Detection
===================================================
Run:
    python3 examples/example_algorithm2.py

Demonstrates how to use the Algorithm 2 module to detect contacts
between edges on a simple mesh.

Based on explore/m2/learn_algorithm2.py — see that file for
the step-by-step educational walkthrough.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ogc_sim.geometry.mesh import Mesh
from ogc_sim.geometry.bvh import BVH
from ogc_sim.geometry.gauss_map import PolyhedralGaussMap, VertexType
from ogc_sim.geometry.distance import edge_edge_distance, ClosestFeature
from ogc_sim.algorithms.algorithm2 import (
    edge_edge_contact_detection,
    run_all_edges,
)
from ogc_sim.algorithms.algorithm1 import run_all_vertices


def pause(title: str) -> None:
    plt.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.show()


# ============================================================
# EXAMPLE 1 — Single edge query
# ============================================================

print("=" * 60)
print("Example 1 — Single edge query with Algorithm 2")
print("=" * 60)

# A simple mesh: a square made of two triangles
#
#   V2 --- V3
#   | \  T1 |
#   |  \    |
#   | T0 \  |
#   V0 --- V1
#
V_mesh = np.array([
    [0., 0., 0.],  # V0
    [2., 0., 0.],  # V1
    [0., 2., 0.],  # V2
    [2., 2., 0.],  # V3
])
T_mesh = np.array([[0, 1, 2], [1, 3, 2]])
mesh = Mesh.from_arrays(V_mesh, T_mesh)
bvh  = BVH(mesh)
pgm  = PolyhedralGaussMap(mesh)

r   = 1.5
r_q = 2.0

print(f"\n  Mesh: {mesh.num_vertices} vertices, {mesh.num_edges} edges")
print(f"  r = {r},  r_q = {r_q}")
print(f"\n  Running Algorithm 2 on each edge:")

for e_idx in range(mesh.num_edges):
    ea = int(mesh.E[e_idx][0])
    eb = int(mesh.E[e_idx][1])
    eogc, d_min_e = edge_edge_contact_detection(e_idx, mesh, bvh, pgm, r, r_q)
    print(f"    E[{e_idx}] V{ea}->V{eb}: EOGC = {eogc}  d_min_e = {d_min_e:.3f}")

# --- Plot ---
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")

for ei in range(mesh.num_edges):
    va, vb = mesh.V[int(mesh.E[ei][0])], mesh.V[int(mesh.E[ei][1])]
    ax.plot([va[0], vb[0]], [va[1], vb[1]], [va[2], vb[2]],
            color="#2980b9", lw=2)
    mid = (va + vb) / 2
    ax.text(mid[0], mid[1], mid[2] + 0.06, f"e{ei}", fontsize=8,
            color="#2980b9", ha="center")

for vi in range(mesh.num_vertices):
    ax.scatter(*mesh.V[vi], color="#555", s=40, zorder=6)
    ax.text(*mesh.V[vi] + np.array([0.05, 0.05, 0.04]),
            f"V{vi}", fontsize=8, color="#555")

ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_xlim(-0.3, 2.5); ax.set_ylim(-0.3, 2.5); ax.set_zlim(-0.1, 0.5)
ax.view_init(elev=30, azim=-50)
pause("Example 1 — Algorithm 2 on each mesh edge")


# ============================================================
# EXAMPLE 2 — Adjacent edge skipping
# ============================================================

print("\n" + "=" * 60)
print("Example 2 — Adjacent edge skipping")
print("=" * 60)

query_e_idx = 0
q_a_idx = int(mesh.E[query_e_idx][0])
q_b_idx = int(mesh.E[query_e_idx][1])
query_e_verts = {q_a_idx, q_b_idx}
e_p = mesh.V[q_a_idx]
e_q = mesh.V[q_b_idx]

print(f"\n  Query edge e[{query_e_idx}] = V{q_a_idx}->V{q_b_idx}")
print(f"\n  Checking all other edges:")

for e2_idx in range(mesh.num_edges):
    if e2_idx == query_e_idx:
        continue
    t_a = int(mesh.E[e2_idx][0])
    t_b = int(mesh.E[e2_idx][1])
    is_adj = bool({t_a, t_b} & query_e_verts)

    e_r = mesh.V[t_a]
    e_s = mesh.V[t_b]
    dist, *_ = edge_edge_distance(e_p, e_q, e_r, e_s)

    if is_adj:
        print(f"    e[{e2_idx}] V{t_a}->V{t_b}: SKIPPED (shares vertex)")
    else:
        print(f"    e[{e2_idx}] V{t_a}->V{t_b}: dist={dist:.3f}  "
              f"{'CONTACT' if dist < r else '(too far)'}")

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(12, 5))

for ax, title, skip_adj in [
    (ax1, "WITHOUT skip\n(counts adjacent edges)", False),
    (ax2, "WITH skip (Algorithm 2 line 5)\n(ignores adjacent edges)", True),
]:
    for ei in range(mesh.num_edges):
        va, vb = mesh.V[int(mesh.E[ei][0])], mesh.V[int(mesh.E[ei][1])]
        is_query = (ei == query_e_idx)
        color = "steelblue" if is_query else "#aaa"
        lw = 3 if is_query else 1.5
        ax.plot([va[0], vb[0]], [va[1], vb[1]], [va[2], vb[2]],
                color=color, lw=lw)
        mid = (va + vb) / 2
        ax.text(mid[0], mid[1], mid[2] + 0.06, f"e{ei}", fontsize=8,
                color=color, ha="center")

    for e2_idx in range(mesh.num_edges):
        if e2_idx == query_e_idx:
            continue
        t_a = int(mesh.E[e2_idx][0])
        t_b = int(mesh.E[e2_idx][1])
        is_adj = bool({t_a, t_b} & query_e_verts)
        e_r = mesh.V[t_a]
        e_s = mesh.V[t_b]
        dist_, cp1_, t1_, cp2_, *_ = edge_edge_distance(e_p, e_q, e_r, e_s)
        if skip_adj and is_adj:
            continue
        if dist_ < r:
            c = "red" if is_adj else "green"
            ax.scatter(*cp1_, color=c, s=50, zorder=9)
            ax.scatter(*cp2_, color=c, s=50, zorder=9)
            ax.plot([cp1_[0], cp2_[0]], [cp1_[1], cp2_[1]], [cp1_[2], cp2_[2]],
                    color=c, lw=1.5, linestyle="--")

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_xlim(-0.3, 2.5); ax.set_ylim(-0.3, 2.5); ax.set_zlim(-0.1, 0.5)
    ax.view_init(elev=30, azim=-50)
    ax.set_title(title, fontsize=9)

pause("Example 2 — Adjacent edge skipping\n"
      "(red = wrongly counted adjacent edge, green = real contact)")


# ============================================================
# EXAMPLE 3 — d_min bookkeeping and conservative bound
# ============================================================

print("\n" + "=" * 60)
print("Example 3 — d_min bookkeeping and conservative bound")
print("=" * 60)

d_min_e_vals = {}
for e2_idx in range(mesh.num_edges):
    if e2_idx == query_e_idx:
        continue
    t_a = int(mesh.E[e2_idx][0])
    t_b = int(mesh.E[e2_idx][1])
    if {t_a, t_b} & query_e_verts:
        continue
    e_r = mesh.V[t_a]
    e_s = mesh.V[t_b]
    dist_, *_ = edge_edge_distance(e_p, e_q, e_r, e_s)
    d_min_e_vals[e2_idx] = dist_

if d_min_e_vals:
    d_min_e = min(d_min_e_vals.values())
else:
    d_min_e = r_q

gamma_p = 0.45
b_v_approx = gamma_p * d_min_e

print(f"\n  d_min_e for e[{query_e_idx}] = {d_min_e:.3f}")
print(f"  gamma_p = {gamma_p}")
print(f"  b_v approx = gamma_p * d_min_e = {b_v_approx:.3f}")
print(f"  -> any vertex of e[{query_e_idx}] may move at most {b_v_approx:.3f} units")


# ============================================================
# EXAMPLE 4 — Full sweep: run_all_edges combined with run_all_vertices
# ============================================================

print("\n" + "=" * 60)
print("Example 4 — Full sweep combining Algorithms 1 and 2")
print("=" * 60)

# Run Algorithm 1 first, then add Algorithm 2 results
cs = run_all_vertices(mesh, bvh, pgm, r, r_q)
cs = run_all_edges(mesh, bvh, pgm, r, r_q, cs=cs)

print(f"\n  FOGC (vertex-facet contacts):")
for v_idx, contacts in cs.FOGC.items():
    if contacts:
        print(f"    FOGC[V{v_idx}] = {contacts}")
print(f"\n  EOGC (edge-edge contacts):")
for e_idx, contacts in cs.EOGC.items():
    if contacts:
        print(f"    EOGC[E{e_idx}] = {contacts}")
print(f"\n  d_min_e per edge:")
for e_idx in range(mesh.num_edges):
    d = cs.d_min_e.get(e_idx, r_q)
    print(f"    E[{e_idx}]: d_min_e = {d:.3f}")

print("""
  Summary:
    Algorithm 2 detects edge-edge contacts via:
      1. BVH sphere query for broadphase (centred at edge midpoint)
      2. Exact edge-edge distance for each candidate
      3. d_min_e bookkeeping (always, even if d >= r)
      4. De-duplication of shared features (line 11)
      5. Gauss Map feasibility gate (Eq. 15)

    It parallels Algorithm 1 in structure:
      Alg 1: vertex -> triangle (7 Voronoi regions)
      Alg 2: edge   -> edge    (2 feature types: INTERIOR or VERTEX)
""")
