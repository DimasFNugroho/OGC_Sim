import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from triangle_distance import point_triangle_distance, ClosestFeature
from ogc_sim.geometry.mesh import Mesh
from ogc_sim.geometry.gauss_map import PolyhedralGaussMap

def pause(title):
    """Add a title to the current figure and show it."""
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()

# ============================================================
# PUTTING IT ALL TOGETHER
# ============================================================
# Here is the complete Algorithm 1 as a single clean function.
# Every line maps directly to a step above.
# ============================================================

print("============================================================")
print("COMPLETE — Algorithm 1 assembled")
print("============================================================")

def algorithm_1(v_idx, mesh, pgm, r):
    """
    Algorithm 1: vertexFacetContactDetection

    For vertex v_idx, find all faces in contact with it.
    Returns FOGC(v) — the list of global feature indices in contact.
    """
    query_v = mesh.V[v_idx]
    fogc    = []                       # result: contacts for this vertex
    d_min_v = float("inf")             # track closest distance seen
    cp = 0

    for t_idx in range(mesh.num_triangles):
        tri   = mesh.T[t_idx]

        # --- line 3: skip adjacent triangles ---
        if v_idx in tri:
            continue

        a_v = mesh.V[tri[0]]
        b_v = mesh.V[tri[1]]
        c_v = mesh.V[tri[2]]

        dist, cp, feature, local_feat_idx = point_triangle_distance(query_v, a_v, b_v, c_v)

        # --- lines 5–6: update d_min ---
        d_min_v = min(d_min_v, dist)

        # --- line 7: only proceed if close enough ---
        if dist >= r:
            continue

        # --- map local feature to global index ---
        if feature == ClosestFeature.FACE_INTERIOR:
            global_feat_idx = t_idx
        elif feature == ClosestFeature.EDGE:
            global_feat_idx = mesh.E_t[t_idx][local_feat_idx]
        else:  # VERTEX
            global_feat_idx = int(tri[local_feat_idx])

        # --- line 9: de-duplication ---
        if global_feat_idx in fogc:
            continue

        # --- lines 10–19: feasibility gate ---
        direction = query_v - cp

        if feature == ClosestFeature.FACE_INTERIOR:
            feasible = True
        elif feature == ClosestFeature.VERTEX:
            feasible = pgm.is_in_vertex_normal_cone(direction, global_feat_idx)
        else:  # EDGE
            feasible = pgm.is_in_edge_normal_slab(direction, global_feat_idx)

        if feasible:
            fogc.append(global_feat_idx)

    return fogc, d_min_v, cp


# Run it on all vertices of the mesh
print("\n  Running Algorithm 1 on all mesh vertices:\n")
# A small flat mesh: 4 vertices, 2 triangles
#
#   V2 --- V3
#   | \  T1 |
#   |  \    |
#   | T0 \  |
#   V0 --- V1
#
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
print(V_mesh[0])
mesh = Mesh.from_arrays(V_mesh, T_mesh)
pgm = PolyhedralGaussMap(mesh)

# Query vertex above the mesh
v_idx   = None          # we'll use a free-floating query point for now
query_v = np.array([0.0, 0.0, 0.75])
r = 0.15

fogc_final = []

for v_idx in range(mesh.num_vertices):
    fogc_final, d_min_v, cp = algorithm_1(v_idx, mesh, pgm, r=0.5)
    print(f"    V[{v_idx}] {mesh.V[v_idx]}:  FOGC = {fogc_final}  d_min_v = {d_min_v:.3f}")

# --- Plot 1 ---
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")

# Draw the mesh
tri_colors = ["#aed6f1", "#a9dfbf"]
for t_idx in range(mesh.num_triangles):
    pts = mesh.V[mesh.T[t_idx]]
    ax.add_collection3d(Poly3DCollection(
        [pts], alpha=0.35,
        facecolor=tri_colors[t_idx], edgecolor="#555", lw=1.0
    ))
    c = pts.mean(axis=0)
    ax.text(c[0], c[1], c[2] + 0.03, f"T[{t_idx}]", fontsize=9, color="#333", ha="center")

# Draw the query vertex (red dot)
ax.scatter(*query_v, color="red", s=80, zorder=9)
ax.text(*query_v + np.array([0.05, 0.05, 0.05]), "query_v", fontsize=9, color="red")

# Draw the closest point (green dot)
ax.scatter(*cp, color="green", s=60, zorder=9)
ax.text(*cp + np.array([0.05, 0.05, 0.05]), "cp", fontsize=9, color="green")

# Draw the distance line
ax.plot([query_v[0], cp[0]], [query_v[1], cp[1]], [query_v[2], cp[2]],
        color="black", lw=2, linestyle="--")
mid = (query_v + cp) / 2
ax.text(*mid + np.array([0.08, 0, 0]), f"dist={d_min_v:.3f}", fontsize=9, color="black")


ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_xlim(-0.3, 2.3); ax.set_ylim(-0.3, 2.3); ax.set_zlim(-0.1, 0.8)
ax.view_init(elev=28, azim=-55)
pause("Step 1 — Distance from query_v to the triangle\n(dashed line = shortest path)")

'''
# --- Plot 7 - Feasibility ---
fig = plt.figure(figsize=(7, 6))
ax  = fig.add_subplot(111, projection="3d")

for t_idx in range(mesh.num_triangles):
    pts = mesh.V[mesh.T[t_idx]]
    ax.add_collection3d(Poly3DCollection(
        [pts], alpha=0.3, facecolor="#aed6f1", edgecolor="#555", lw=1.0
    ))
    c = pts.mean(axis=0)
    ax.text(c[0], c[1], c[2]+0.03, f"T[{t_idx}]", fontsize=9, ha="center")

# Mark all mesh vertices with their type (convex/concave)
from ogc_sim.geometry.gauss_map import VertexType
type_color = {VertexType.CONVEX: "#2ecc71", VertexType.CONCAVE: "#e74c3c", VertexType.MIXED: "#f39c12"}
for vi, v in enumerate(mesh.V):
    vtype = pgm.vertex_types[vi]
    ax.scatter(*v, color=type_color[vtype], s=60, zorder=7)
    ax.text(*v + np.array([0.05, 0.05, 0.04]), f"V{vi}", fontsize=8, color=type_color[vtype])

# Draw the face normals (small arrows)
for t_idx in range(mesh.num_triangles):
    cen = mesh.V[mesh.T[t_idx]].mean(axis=0)
    n   = mesh.face_normals[t_idx] * 0.25
    ax.quiver(*cen, *n, color="#888", lw=1.2, arrow_length_ratio=0.3)

ax.scatter(*query_v, color="purple", s=80, zorder=9)
ax.text(*query_v + np.array([0.05, 0.05, 0.04]), "query_v", fontsize=9, color="purple")

# Draw contact if found
for t_idx in range(mesh.num_triangles):
    tri = mesh.T[t_idx]
    d, cp, feat, li = point_triangle_distance(query_v, mesh.V[tri[0]], mesh.V[tri[1]], mesh.V[tri[2]])
    if d < r:
        if feat == ClosestFeature.EDGE:
            gfi = mesh.E_t[t_idx][li]
        elif feat == ClosestFeature.VERTEX:
            gfi = int(tri[li])
        else:
            gfi = t_idx
        if gfi in fogc_final:
            ax.scatter(*cp, color="green", s=50, zorder=8)
            ax.plot([query_v[0], cp[0]], [query_v[1], cp[1]], [query_v[2], cp[2]],
                    color="green", lw=2, linestyle="--")

ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_xlim(-0.5, 2.5); ax.set_ylim(-0.5, 2.5); ax.set_zlim(-0.1, 0.6)
ax.view_init(elev=30, azim=-50)
pause("Step 7 — Feasibility gate (Eq. 8/9)\n(green = accepted contact, purple = query vertex)")
'''
