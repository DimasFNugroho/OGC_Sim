"""
Example: Algorithm 1 — Vertex-Facet Contact Detection
=====================================================
Run:
    python3 examples/example_algorithm1.py

Demonstrates how to use the Algorithm 1 module to detect contacts
between vertices and triangles on a simple mesh.

Based on explore/m2/learn_algorithm1.py — see that file for
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
from ogc_sim.geometry.distance import point_triangle_distance, ClosestFeature
from ogc_sim.algorithms.algorithm1 import (
    vertex_facet_contact_detection,
    run_all_vertices,
)


def pause(title: str) -> None:
    plt.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.show()


# ============================================================
# EXAMPLE 1 — Single vertex query
# ============================================================
# A small flat mesh: 4 vertices, 2 triangles
#
#   V2 --- V3
#   | \  T1 |
#   |  \    |
#   | T0 \  |
#   V0 --- V1
#

print("=" * 60)
print("Example 1 — Single vertex query with Algorithm 1")
print("=" * 60)

V_mesh = np.array([
    [0., 0., 0.],   # V0
    [2., 0., 0.],   # V1
    [0., 2., 0.],   # V2
    [2., 2., 0.],   # V3
])
T_mesh = np.array([
    [0, 1, 2],   # T0
    [1, 3, 2],   # T1
])
mesh = Mesh.from_arrays(V_mesh, T_mesh)
bvh  = BVH(mesh)
pgm  = PolyhedralGaussMap(mesh)

r   = 0.5   # contact radius
r_q = 0.8   # query radius

print(f"\n  Mesh: {mesh.num_vertices} vertices, {mesh.num_triangles} triangles")
print(f"  r = {r},  r_q = {r_q}")
print(f"\n  Running Algorithm 1 on each vertex:")

for v_idx in range(mesh.num_vertices):
    fogc, vogc_t, d_min_v = vertex_facet_contact_detection(
        v_idx, mesh, bvh, pgm, r, r_q
    )
    print(f"    V[{v_idx}] {mesh.V[v_idx]}:  FOGC = {fogc}  d_min_v = {d_min_v:.3f}")

# --- Plot: show each vertex's FOGC ---
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")

tri_colors = ["#aed6f1", "#a9dfbf"]
for t_idx in range(mesh.num_triangles):
    pts = mesh.V[mesh.T[t_idx]]
    ax.add_collection3d(Poly3DCollection(
        [pts], alpha=0.35,
        facecolor=tri_colors[t_idx], edgecolor="#555", lw=1.0
    ))
    c = pts.mean(axis=0)
    ax.text(c[0], c[1], c[2] + 0.03, f"T[{t_idx}]", fontsize=9, color="#333", ha="center")

# Mark vertices with their type
type_color = {VertexType.CONVEX: "#2ecc71", VertexType.CONCAVE: "#e74c3c", VertexType.MIXED: "#f39c12"}
for vi, v in enumerate(mesh.V):
    vtype = pgm.vertex_types[vi]
    ax.scatter(*v, color=type_color[vtype], s=60, zorder=7)
    ax.text(*v + np.array([0.05, 0.05, 0.04]), f"V{vi}", fontsize=8, color=type_color[vtype])

# Face normals
for t_idx in range(mesh.num_triangles):
    cen = mesh.V[mesh.T[t_idx]].mean(axis=0)
    n = mesh.face_normals[t_idx] * 0.25
    ax.quiver(*cen, *n, color="#888", lw=1.2, arrow_length_ratio=0.3)

ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_xlim(-0.3, 2.5); ax.set_ylim(-0.3, 2.5); ax.set_zlim(-0.1, 0.6)
ax.view_init(elev=30, azim=-50)
pause("Example 1 — Algorithm 1 on each mesh vertex\n(green=convex, red=concave, orange=mixed)")


# ============================================================
# EXAMPLE 2 — Full mesh sweep using run_all_vertices
# ============================================================
# Two separate triangles (non-adjacent) with a query vertex between them.

print("\n" + "=" * 60)
print("Example 2 — Full mesh sweep with run_all_vertices")
print("=" * 60)

V2 = np.array([
    [0., 0., 0.],   # 0 — T[0]
    [2., 0., 0.],   # 1 — T[0]
    [1., 2., 0.],   # 2 — T[0]
    [0., 0., 0.6],  # 3 — T[1]
    [2., 0., 0.6],  # 4 — T[1]
    [1., 2., 0.6],  # 5 — T[1]
    [1., 0.8, 0.15], # 6 — query vertex between the two triangles
])
T2 = np.array([
    [0, 1, 2],   # T[0]
    [3, 4, 5],   # T[1]
])
mesh2 = Mesh.from_arrays(V2, T2)
bvh2  = BVH(mesh2)
pgm2  = PolyhedralGaussMap(mesh2)

r2   = 0.25
r_q2 = 0.50

cs = run_all_vertices(mesh2, bvh2, pgm2, r2, r_q2)

print(f"\n  Mesh: {mesh2.num_vertices} vertices, {mesh2.num_triangles} triangles")
print(f"  r = {r2},  r_q = {r_q2}")
print(f"\n  Results from run_all_vertices:")
for v_idx in range(mesh2.num_vertices):
    fogc = cs.FOGC.get(v_idx, [])
    d_min = cs.d_min_v.get(v_idx, r_q2)
    print(f"    V[{v_idx}] {np.round(mesh2.V[v_idx], 2)}:  FOGC = {fogc}  d_min_v = {d_min:.3f}")

print(f"\n  VOGC (which vertices hit each triangle):")
for t_idx, verts in cs.VOGC.items():
    if verts:
        print(f"    VOGC[T{t_idx}] = {verts}")

# --- Plot ---
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")

for t_idx in range(mesh2.num_triangles):
    pts = mesh2.V[mesh2.T[t_idx]]
    ax.add_collection3d(Poly3DCollection(
        [pts], alpha=0.3, facecolor="#aed6f1", edgecolor="#555", lw=1.0
    ))
    c = pts.mean(axis=0)
    ax.text(c[0], c[1], c[2] + 0.03, f"T[{t_idx}]", fontsize=9, ha="center")

# Draw vertices and their contacts
for v_idx in range(mesh2.num_vertices):
    fogc = cs.FOGC.get(v_idx, [])
    color = "#e74c3c" if fogc else "#2980b9"
    ax.scatter(*mesh2.V[v_idx], color=color, s=60, zorder=7)
    ax.text(*mesh2.V[v_idx] + np.array([0.04, 0.04, 0.04]),
            f"V{v_idx}\nFOGC={fogc}", fontsize=7, color=color)

    # Draw contact lines
    if fogc:
        for t_idx in range(mesh2.num_triangles):
            tri = mesh2.T[t_idx]
            a, b, c = mesh2.V[tri[0]], mesh2.V[tri[1]], mesh2.V[tri[2]]
            dist, cp, _, _ = point_triangle_distance(mesh2.V[v_idx], a, b, c)
            if dist < r2:
                ax.scatter(*cp, color="green", s=40, zorder=8)
                ax.plot([mesh2.V[v_idx][0], cp[0]],
                        [mesh2.V[v_idx][1], cp[1]],
                        [mesh2.V[v_idx][2], cp[2]],
                        color="green", lw=2, linestyle="--")

ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_xlim(-0.3, 2.3); ax.set_ylim(-0.3, 2.3); ax.set_zlim(-0.1, 0.8)
ax.view_init(elev=28, azim=-55)
pause("Example 2 — Full mesh sweep with run_all_vertices\n"
      "(red = has contact, green lines = contact pairs)")


# ============================================================
# EXAMPLE 3 — De-duplication and feasibility gate
# ============================================================

print("\n" + "=" * 60)
print("Example 3 — De-duplication and feasibility in action")
print("=" * 60)

# Place a query vertex near a shared edge to trigger de-duplication
query_v = np.array([1.0, 1.0, 0.25])
r3 = 0.5

print(f"\n  Query vertex at {query_v} (near shared edge between T0 and T1)")
print(f"  r = {r3}")
print(f"\n  Brute-force per-triangle breakdown:")

for t_idx in range(mesh.num_triangles):
    tri = mesh.T[t_idx]
    a, b, c = mesh.V[tri[0]], mesh.V[tri[1]], mesh.V[tri[2]]
    dist, cp, feature, local_idx = point_triangle_distance(query_v, a, b, c)

    if feature == ClosestFeature.FACE_INTERIOR:
        gfi = t_idx
    elif feature == ClosestFeature.EDGE:
        gfi = mesh.E_t[t_idx][local_idx]
    else:
        gfi = int(tri[local_idx])

    print(f"    T[{t_idx}]: dist={dist:.3f}  feature={feature.name}  global_feat={gfi}")

print(f"\n  Both triangles report the same shared edge.")
print(f"  Algorithm 1's de-duplication (line 9) keeps it only once.")

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(12, 5))

for ax, title, dedup in [
    (ax1, "WITHOUT de-duplication\n(same edge counted twice)", False),
    (ax2, "WITH de-duplication (Algorithm 1 line 9)\n(each feature counted once)", True),
]:
    for t_idx in range(mesh.num_triangles):
        pts = mesh.V[mesh.T[t_idx]]
        ax.add_collection3d(Poly3DCollection(
            [pts], alpha=0.3, facecolor="#aed6f1", edgecolor="#555", lw=1.0
        ))
    ax.scatter(*query_v, color="purple", s=80, zorder=9)
    ax.text(*query_v + np.array([0.05, 0.05, 0.05]), "query_v", fontsize=9, color="purple")

    # Draw shared edge
    for ei, (ea, eb) in enumerate(mesh.E):
        adj = [t for t in range(mesh.num_triangles) if ei in mesh.E_t[t]]
        if len(adj) == 2:
            p0, p1 = mesh.V[ea], mesh.V[eb]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                    color="red", lw=3, zorder=8)

    # Count contacts
    seen = set()
    n_contacts = 0
    for t_idx in range(mesh.num_triangles):
        tri = mesh.T[t_idx]
        a, b, c = mesh.V[tri[0]], mesh.V[tri[1]], mesh.V[tri[2]]
        dist, cp, feature, local_idx = point_triangle_distance(query_v, a, b, c)
        if dist >= r3:
            continue
        if feature == ClosestFeature.EDGE:
            gfi = mesh.E_t[t_idx][local_idx]
        elif feature == ClosestFeature.VERTEX:
            gfi = int(tri[local_idx])
        else:
            gfi = t_idx
        if dedup and gfi in seen:
            continue
        seen.add(gfi)
        n_contacts += 1
        offset = 0.04 * (n_contacts - 1) if not dedup else 0.0
        cp_draw = cp + np.array([offset, 0, 0])
        ax.scatter(*cp_draw, color="green", s=40, zorder=8)
        ax.plot([query_v[0], cp_draw[0]], [query_v[1], cp_draw[1]],
                [query_v[2], cp_draw[2]], color="green", lw=1.5, linestyle="--")

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_xlim(-0.3, 2.5); ax.set_ylim(-0.3, 2.5); ax.set_zlim(-0.1, 0.6)
    ax.view_init(elev=30, azim=-50)
    ax.set_title(f"{title}\n({n_contacts} contact(s))", fontsize=9)

pause("Example 3 — De-duplication: shared features counted once, not twice")

print("""
  Summary:
    Algorithm 1 detects vertex-facet contacts via:
      1. BVH sphere query for broadphase
      2. Exact point-triangle distance for each candidate
      3. d_min bookkeeping (always, even if d >= r)
      4. De-duplication of shared features (line 9)
      5. Gauss Map feasibility gate (Eq. 8, 9)
""")
