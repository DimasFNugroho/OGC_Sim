"""
Building Algorithm 1 from scratch — step by step
=================================================
Run this file:
    python3 explore/m2/learn_algorithm1.py

Each step pauses on a plot window.
Close the window → the next step runs.

You can add, change, or comment out anything.
The goal is for you to understand each line before moving on.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from triangle_distance import point_triangle_distance, ClosestFeature


def pause(title):
    """Add a title to the current figure and show it."""
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()


# ============================================================
# STEP 1 — One vertex, one triangle, one distance
# ============================================================
# Before anything else: what does "distance from a vertex to
# a triangle" even mean?
#
# point_triangle_distance(p, a, b, c) returns:
#   dist      — the shortest distance from p to the triangle
#   cp        — the closest point ON the triangle to p
#   feature   — WHERE on the triangle that closest point is:
#                 FACE_INTERIOR, EDGE, or VERTEX
#   feat_idx  — which specific edge/vertex (0,1,2) it is
# ============================================================

print("==================================================")
print("STEP 1 — Distance from a vertex to a triangle")
print("==================================================")

# Define one triangle and one query vertex
tri_a = np.array([0.0, 0.0, 0.0])
tri_b = np.array([2.0, 0.0, 0.0])
tri_c = np.array([1.0, 2.0, 0.0])

query_v = np.array([1.0, 0.8, 0.5])   # above the face interior

dist, cp, feature, feat_idx = point_triangle_distance(query_v, tri_a, tri_b, tri_c)

print(f"\n  Triangle vertices:")
print(f"    a = {tri_a}")
print(f"    b = {tri_b}")
print(f"    c = {tri_c}")
print(f"\n  Query vertex: {query_v}")
print(f"\n  Result:")
print(f"    dist    = {dist:.4f}   ← the number we care most about")
print(f"    cp      = {np.round(cp, 3)}   ← closest point ON the triangle")
print(f"    feature = {feature}   ← where cp sits on the triangle")
print(f"    feat_idx= {feat_idx}   ← -1 means interior (no specific edge/vertex)")

# --- Plot ---
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")

# Draw the triangle
ax.add_collection3d(Poly3DCollection(
    [[tri_a, tri_b, tri_c]], alpha=0.3, facecolor="#aed6f1", edgecolor="#2980b9", lw=1.5
))

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
ax.text(*mid + np.array([0.08, 0, 0]), f"dist={dist:.3f}", fontsize=9, color="black")

# Label triangle vertices
for name, pt in [("a", tri_a), ("b", tri_b), ("c", tri_c)]:
    ax.scatter(*pt, color="#2980b9", s=40)
    ax.text(*pt + np.array([0.05, 0.05, 0.03]), name, fontsize=9, color="#2980b9")

ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_xlim(-0.3, 2.3); ax.set_ylim(-0.3, 2.3); ax.set_zlim(-0.1, 0.8)
ax.view_init(elev=28, azim=-55)
pause("Step 1 — Distance from query_v to the triangle\n(dashed line = shortest path)")


# ============================================================
# STEP 2 — The contact radius r
# ============================================================
# "Contact" just means: is the vertex close enough to matter?
# We define a radius r.  If dist < r → contact.  Otherwise → ignore.
#
# This is the key threshold in Algorithm 1, line 7:
#   "if d < r then ..."
# ============================================================

print("\n" + "=" * 50)
print("STEP 2 — The contact radius r")
print("=" * 50)

r = 0.4   # contact radius — try changing this value

is_contact = dist < r
print(f"\n  r = {r}  (contact radius — you can change this)")
print(f"  dist = {dist:.4f}")
print(f"  dist < r ? → {is_contact}")
print(f"\n  {'CONTACT DETECTED' if is_contact else 'No contact (too far away)'}")

# --- Plot ---
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")

ax.add_collection3d(Poly3DCollection(
    [[tri_a, tri_b, tri_c]], alpha=0.3, facecolor="#aed6f1", edgecolor="#2980b9", lw=1.5
))

# Draw the r-sphere around query_v
u = np.linspace(0, 2 * np.pi, 24)
v_ = np.linspace(0, np.pi, 12)
ax.plot_wireframe(
    r * np.outer(np.cos(u), np.sin(v_)) + query_v[0],
    r * np.outer(np.sin(u), np.sin(v_)) + query_v[1],
    r * np.outer(np.ones(24), np.cos(v_)) + query_v[2],
    color="orange", alpha=0.2, linewidth=0.5
)

color = "green" if is_contact else "red"
label = f"CONTACT  (dist={dist:.3f} < r={r})" if is_contact else f"no contact  (dist={dist:.3f} >= r={r})"
ax.scatter(*query_v, color=color, s=100, zorder=9)
ax.text(*query_v + np.array([0.05, 0.05, 0.05]), label, fontsize=8, color=color)
ax.scatter(*cp, color="green", s=50)
ax.plot([query_v[0], cp[0]], [query_v[1], cp[1]], [query_v[2], cp[2]],
        color=color, lw=2, linestyle="--")

ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_xlim(-0.3, 2.3); ax.set_ylim(-0.3, 2.3); ax.set_zlim(-0.1, 0.8)
ax.view_init(elev=28, azim=-55)
pause(f"Step 2 — Contact radius r={r}\n(orange sphere = r, vertex is {'inside' if is_contact else 'outside'})")


# ============================================================
# STEP 3 — Three kinds of closest feature
# ============================================================
# The closest point on the triangle can land in three places:
#
#   FACE_INTERIOR — the point is inside the triangle face
#   EDGE          — the point is on one of the three edges
#   VERTEX        — the point is right at one of the three corners
#
# This matters because each location "belongs" to a different
# feature of the mesh.  We need to know which one so we can
# apply the right feasibility check later.
# ============================================================

print("\n" + "=" * 50)
print("STEP 3 — Three types of closest feature")
print("=" * 50)

cases = {
    "FACE_INTERIOR": np.array([1.0, 0.8, 0.4]),   # directly above the middle
    "EDGE":          np.array([1.0, -0.3, 0.2]),   # beside the bottom edge (a-b)
    "VERTEX":        np.array([-0.3, -0.2, 0.2]),  # near corner a
}

fig = plt.figure(figsize=(14, 5))
feature_colors = {"FACE_INTERIOR": "#2ecc71", "EDGE": "#e67e22", "VERTEX": "#9b59b6"}

for i, (case_name, qv) in enumerate(cases.items()):
    d, closest, feat, fidx = point_triangle_distance(qv, tri_a, tri_b, tri_c)
    print(f"\n  Query {qv} → feature={feat}, dist={d:.3f}, feat_idx={fidx}")

    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    ax.add_collection3d(Poly3DCollection(
        [[tri_a, tri_b, tri_c]], alpha=0.3,
        facecolor=feature_colors[case_name], edgecolor="#555", lw=1.5
    ))
    ax.scatter(*qv, color="red", s=80, zorder=9)
    ax.scatter(*closest, color="black", s=60, zorder=9)
    ax.plot([qv[0], closest[0]], [qv[1], closest[1]], [qv[2], closest[2]],
            color="black", lw=2, linestyle="--")
    ax.text(*qv + np.array([0.05, 0.05, 0.05]), "query", fontsize=8, color="red")
    ax.text(*closest + np.array([0.05, 0.05, 0.05]),
            f"cp\n({feat})", fontsize=7, color="black")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_xlim(-0.5, 2.3); ax.set_ylim(-0.5, 2.3); ax.set_zlim(-0.1, 0.7)
    ax.view_init(elev=28, azim=-55)
    ax.set_title(f"case: {case_name}\ndist={d:.3f}", fontsize=9)

pause("Step 3 — Three types of closest feature\n(FACE_INTERIOR / EDGE / VERTEX)")


# ============================================================
# STEP 4 — Loop over many triangles
# ============================================================
# Now we have multiple triangles.
# We loop over all of them and collect every contact.
#
# One important rule: skip any triangle that CONTAINS vertex v.
# Those are "adjacent" faces — they share a vertex with v,
# so by definition their distance to v is 0.  They are not
# a "collision" — they are just the mesh itself.
# ============================================================

print("\n" + "=" * 50)
print("STEP 4 — Loop over many triangles")
print("=" * 50)

from ogc_sim.geometry.mesh import Mesh

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
mesh = Mesh.from_arrays(V_mesh, T_mesh)

# Query vertex above the mesh
v_idx   = None          # we'll use a free-floating query point for now
query_v = np.array([1.2, 1.0, 0.3])
r = 0.5

print(f"\n  Mesh: {mesh.num_vertices} vertices, {mesh.num_triangles} triangles")
print(f"  Query vertex: {query_v},  r = {r}")
print(f"\n  Looping over all triangles...")

contacts = []   # we will collect contacts here

for t_idx in range(mesh.num_triangles):
    tri    = mesh.T[t_idx]
    a_v    = mesh.V[tri[0]]
    b_v    = mesh.V[tri[1]]
    c_v    = mesh.V[tri[2]]

    dist, cp, feature, feat_idx = point_triangle_distance(query_v, a_v, b_v, c_v)

    within = dist < r
    print(f"    T[{t_idx}]:  dist={dist:.3f}  feature={feature}  "
          f"{'← CONTACT' if within else '(too far)'}")

    if within:
        contacts.append((t_idx, dist, cp, feature))

print(f"\n  Contacts found: {[c[0] for c in contacts]}")

# --- Plot ---
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

ax.scatter(*query_v, color="red", s=80, zorder=9)
ax.text(*query_v + np.array([0.04, 0.04, 0.04]), "query_v", fontsize=9, color="red")

for t_idx, dist, cp, feature in contacts:
    ax.scatter(*cp, color="green", s=50, zorder=8)
    ax.plot([query_v[0], cp[0]], [query_v[1], cp[1]], [query_v[2], cp[2]],
            color="green", lw=2, linestyle="--")
    ax.text(*cp + np.array([0.05, 0.05, 0.04]),
            f"T[{t_idx}]\nd={dist:.2f}", fontsize=8, color="green")

ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_xlim(-0.3, 2.5); ax.set_ylim(-0.3, 2.5); ax.set_zlim(-0.1, 0.7)
ax.view_init(elev=30, azim=-50)
pause("Step 4 — Loop over triangles, collect contacts (green lines)")


# ============================================================
# STEP 5 — Skip adjacent triangles
# ============================================================
# A vertex that IS part of a triangle has distance 0 to it.
# We must not count that as a collision — it is just the mesh.
#
# Algorithm 1 line 3:  "if v ⊂ t then continue"
# In code: if v_idx in mesh.T[t_idx]: skip
#
# Here we place the query vertex AT one of the mesh vertices
# so you can see the problem — and the fix.
# ============================================================

print("\n" + "=" * 50)
print("STEP 5 — Skip adjacent triangles")
print("=" * 50)

# Now the query vertex IS vertex V0 (index 0) of the mesh
# V0 is part of T0 and T1.  Without the skip, both would be
# detected as contacts with dist=0.
v_idx   = 0
query_v = mesh.V[v_idx]   # same position as vertex 0
r       = 0.5

print(f"\n  query_v = mesh.V[{v_idx}] = {query_v}")
print(f"  This vertex belongs to: T{[t for t in range(mesh.num_triangles) if v_idx in mesh.T[t]]}")
print()

contacts_with_skip    = []
contacts_without_skip = []

for t_idx in range(mesh.num_triangles):
    tri   = mesh.T[t_idx]
    a_v   = mesh.V[tri[0]]
    b_v   = mesh.V[tri[1]]
    c_v   = mesh.V[tri[2]]

    dist, cp, feature, feat_idx = point_triangle_distance(query_v, a_v, b_v, c_v)

    # Without skip
    if dist < r:
        contacts_without_skip.append(t_idx)

    # With skip  ← this is Algorithm 1 line 3
    if v_idx in mesh.T[t_idx]:
        print(f"    T[{t_idx}]: SKIPPED (v_idx={v_idx} is a vertex of this triangle)")
        continue

    if dist < r:
        contacts_with_skip.append(t_idx)
        print(f"    T[{t_idx}]: contact, dist={dist:.3f}")
    else:
        print(f"    T[{t_idx}]: dist={dist:.3f}, no contact")

print(f"\n  Without the skip: contacts = {contacts_without_skip}  ← includes self!")
print(f"  With the skip:    contacts = {contacts_with_skip}")

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(12, 5))

for ax, title, contacts in [
    (ax1, "WITHOUT skip\n(counts adjacent triangles as contacts)", contacts_without_skip),
    (ax2, "WITH skip  (Algorithm 1 line 3)\n(correctly ignores adjacent triangles)", contacts_with_skip),
]:
    for t_idx in range(mesh.num_triangles):
        pts   = mesh.V[mesh.T[t_idx]]
        color = "#f1948a" if t_idx in contacts else "#aed6f1"
        ax.add_collection3d(Poly3DCollection(
            [pts], alpha=0.4, facecolor=color, edgecolor="#555", lw=1.0
        ))
        c = pts.mean(axis=0)
        ax.text(c[0], c[1], c[2]+0.03, f"T[{t_idx}]", fontsize=9, ha="center")

    ax.scatter(*query_v, color="red", s=100, zorder=9)
    ax.text(*query_v + np.array([0.05,0.05,0.05]), f"V[{v_idx}]", fontsize=9, color="red")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_xlim(-0.3, 2.5); ax.set_ylim(-0.3, 2.5); ax.set_zlim(-0.1, 0.5)
    ax.view_init(elev=30, azim=-50)
    ax.set_title(title, fontsize=9)

pause("Step 5 — Skip adjacent triangles\n(red = wrongly counted, blue = correctly skipped)")


# ============================================================
# STEP 6 — The duplication problem
# ============================================================
# Consider a vertex near the SHARED EDGE between T0 and T1.
# Both triangles return the same closest sub-feature: that edge.
# Without de-duplication, we record the same contact TWICE.
#
# Algorithm 1 line 9:
#   "if a already in FOGC(v) then continue"
#
# "a" here is the GLOBAL feature index — the same number
# regardless of which triangle reported it.
# ============================================================

print("\n" + "=" * 50)
print("STEP 6 — De-duplication: shared edges appear twice")
print("=" * 50)

# Place the query right above the shared edge (V1-V2 diagonal)
query_v = np.array([1.0, 1.0, 0.25])   # near center, above shared edge
r       = 0.5

print(f"\n  query_v = {query_v}  (near the shared edge between T0 and T1)")
print(f"\n  Loop WITHOUT de-duplication:")

fogc_no_dedup = []   # collect all contacts, even duplicates
fogc_dedup    = []   # collect contacts with de-duplication

for t_idx in range(mesh.num_triangles):
    tri   = mesh.T[t_idx]
    a_v   = mesh.V[tri[0]]
    b_v   = mesh.V[tri[1]]
    c_v   = mesh.V[tri[2]]

    dist, cp, feature, local_feat_idx = point_triangle_distance(query_v, a_v, b_v, c_v)

    if dist >= r:
        continue

    # Map local feature index → global feature index
    if feature == ClosestFeature.FACE_INTERIOR:
        global_feat_idx = t_idx                          # the triangle itself
    elif feature == ClosestFeature.EDGE:
        global_feat_idx = mesh.E_t[t_idx][local_feat_idx]   # global edge index
    else:  # VERTEX
        global_feat_idx = int(tri[local_feat_idx])       # global vertex index

    label = f"global_feat_idx={global_feat_idx}  ({feature})"
    fogc_no_dedup.append(global_feat_idx)
    print(f"    T[{t_idx}]: dist={dist:.3f}  {label}")

    # De-duplication check  ← Algorithm 1 line 9
    if global_feat_idx in fogc_dedup:
        print(f"           → DUPLICATE! already in FOGC. Skipping.")
        continue

    fogc_dedup.append(global_feat_idx)

print(f"\n  FOGC without de-dup: {fogc_no_dedup}  ({len(fogc_no_dedup)} entries)")
print(f"  FOGC with de-dup:    {fogc_dedup}   ({len(fogc_dedup)} entries)")
print(f"\n  The shared edge appeared {fogc_no_dedup.count(fogc_no_dedup[0])} times → "
      f"de-dup reduced it to 1.")

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(12, 5))

for ax, title, fogc in [
    (ax1, f"WITHOUT de-dup: {len(fogc_no_dedup)} contact(s) recorded", fogc_no_dedup),
    (ax2, f"WITH de-dup:    {len(fogc_dedup)} contact(s) recorded", fogc_dedup),
]:
    for t_idx in range(mesh.num_triangles):
        pts = mesh.V[mesh.T[t_idx]]
        ax.add_collection3d(Poly3DCollection(
            [pts], alpha=0.3, facecolor="#aed6f1", edgecolor="#555", lw=1.0
        ))
        c = pts.mean(axis=0)
        ax.text(c[0], c[1], c[2]+0.03, f"T[{t_idx}]", fontsize=9, ha="center")

    # Draw the shared edge in red
    for ei, (ea, eb) in enumerate(mesh.E):
        adj = [t for t in range(mesh.num_triangles) if ei in mesh.E_t[t]]
        if len(adj) == 2:   # shared edge
            p0, p1 = mesh.V[ea], mesh.V[eb]
            ax.plot([p0[0],p1[0]], [p0[1],p1[1]], [p0[2],p1[2]],
                    color="red", lw=3, zorder=8)

    ax.scatter(*query_v, color="purple", s=80, zorder=9)
    ax.text(*query_v + np.array([0.05,0.05,0.05]), "query_v", fontsize=9, color="purple")

    # Draw one line per entry in fogc
    seen_cp = {}
    for t_idx in range(mesh.num_triangles):
        tri   = mesh.T[t_idx]
        a_v   = mesh.V[tri[0]]
        b_v   = mesh.V[tri[1]]
        c_v   = mesh.V[tri[2]]
        dist, cp, feature, local_feat_idx = point_triangle_distance(query_v, a_v, b_v, c_v)
        if dist >= r:
            continue
        if feature == ClosestFeature.EDGE:
            gfi = mesh.E_t[t_idx][local_feat_idx]
        elif feature == ClosestFeature.VERTEX:
            gfi = int(tri[local_feat_idx])
        else:
            gfi = t_idx
        if gfi in fogc:
            key = (round(cp[0],3), round(cp[1],3))
            offset = seen_cp.get(key, 0)
            seen_cp[key] = offset + 1
            cp_draw = cp + np.array([offset * 0.08, 0, 0])
            ax.scatter(*cp_draw, color="green", s=40, zorder=8)
            ax.plot([query_v[0], cp_draw[0]],
                    [query_v[1], cp_draw[1]],
                    [query_v[2], cp_draw[2]],
                    color="green", lw=1.5, linestyle="--")

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_xlim(-0.3, 2.5); ax.set_ylim(-0.3, 2.5); ax.set_zlim(-0.1, 0.6)
    ax.view_init(elev=30, azim=-50)
    ax.set_title(title, fontsize=9)

pause("Step 6 — De-duplication\n(red = shared edge, green lines = contacts recorded)")


# ============================================================
# STEP 7 — The feasibility gate (Eq. 8 and 9)
# ============================================================
# De-duplication removes exact duplicates (same global feature
# from two different triangles).
#
# But there is a subtler problem: when the query vertex is right
# at the BOUNDARY between two feature blocks, point_triangle_distance
# might return different features from two adjacent triangles —
# not the same one.  The feasibility gate (Gauss Map check) resolves
# this by asking: "is this contact direction consistent with this
# feature's normal set?"
#
# Algorithm 1 lines 10–19:
#   if feature is VERTEX:   check Eq. 8 (normal cone)
#   if feature is EDGE:     check Eq. 9 (normal slab)
#   if feature is INTERIOR: always OK
#
# If the check fails, the contact is simply not recorded here —
# it will be picked up by the CORRECT feature when that feature
# is tested.
# ============================================================

print("\n" + "=" * 50)
print("STEP 7 — The feasibility gate (Eq. 8 and 9)")
print("=" * 50)

from ogc_sim.geometry.gauss_map import PolyhedralGaussMap

pgm = PolyhedralGaussMap(mesh)

# Use a query above the corner vertex V0 to demonstrate the vertex cone check
query_v = np.array([-0.2, -0.15, 0.25])  # near corner V0 = (0,0,0)
r       = 0.5

print(f"\n  query_v = {query_v}  (near corner V0)")
print(f"\n  For each candidate triangle:")

fogc_final = []

for t_idx in range(mesh.num_triangles):
    tri   = mesh.T[t_idx]
    a_v   = mesh.V[tri[0]]
    b_v   = mesh.V[tri[1]]
    c_v   = mesh.V[tri[2]]

    dist, cp, feature, local_feat_idx = point_triangle_distance(query_v, a_v, b_v, c_v)

    if dist >= r:
        print(f"    T[{t_idx}]: dist={dist:.3f} >= r  → skip (too far)")
        continue

    # Map to global feature index
    if feature == ClosestFeature.FACE_INTERIOR:
        global_feat_idx = t_idx
    elif feature == ClosestFeature.EDGE:
        global_feat_idx = mesh.E_t[t_idx][local_feat_idx]
    else:
        global_feat_idx = int(tri[local_feat_idx])

    # De-duplication check
    if global_feat_idx in fogc_final:
        print(f"    T[{t_idx}]: global_feat={global_feat_idx}  → duplicate, skip")
        continue

    # Feasibility gate  ← Algorithm 1 lines 10–19
    direction = query_v - cp   # vector from closest point toward the query vertex

    if feature == ClosestFeature.FACE_INTERIOR:
        feasible = True   # always OK, no check needed
        reason   = "face interior → always feasible"

    elif feature == ClosestFeature.VERTEX:
        feasible = pgm.is_in_vertex_normal_cone(direction, global_feat_idx)
        reason   = f"Eq. 8: cone check on V[{global_feat_idx}] → {feasible}"

    else:  # EDGE
        feasible = pgm.is_in_edge_normal_slab(direction, global_feat_idx)
        reason   = f"Eq. 9: slab check on E[{global_feat_idx}] → {feasible}"

    mark = "✓ CONTACT" if feasible else "✗ rejected (wrong block)"
    print(f"    T[{t_idx}]: dist={dist:.3f}  feat={feature}  "
          f"gfeat={global_feat_idx}  {reason}  {mark}")

    if feasible:
        fogc_final.append(global_feat_idx)

print(f"\n  Final FOGC(query_v) = {fogc_final}")

# --- Plot ---
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


# ============================================================
# PUTTING IT ALL TOGETHER
# ============================================================
# Here is the complete Algorithm 1 as a single clean function.
# Every line maps directly to a step above.
# ============================================================

print("\n" + "=" * 50)
print("COMPLETE — Algorithm 1 assembled")
print("=" * 50)

def algorithm_1(v_idx, mesh, pgm, r):
    """
    Algorithm 1: vertexFacetContactDetection

    For vertex v_idx, find all faces in contact with it.
    Returns FOGC(v) — the list of global feature indices in contact.
    """
    query_v = mesh.V[v_idx]
    fogc    = []                       # result: contacts for this vertex
    d_min_v = float("inf")             # track closest distance seen

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

    return fogc, d_min_v


# Run it on all vertices of the mesh
print("\n  Running Algorithm 1 on all mesh vertices:\n")
for v_idx in range(mesh.num_vertices):
    fogc, d_min_v = algorithm_1(v_idx, mesh, pgm, r=0.5)
    print(f"    V[{v_idx}] {mesh.V[v_idx]}:  FOGC = {fogc}  d_min_v = {d_min_v:.3f}")

print("""
  Algorithm 1 is now complete.

  What comes next (not in this script yet):
    - Replace the brute-force triangle loop with a BVH sphere query
      (for performance — the logic is identical)
    - Add d_min_t bookkeeping (per-triangle minimum distance)
    - Wrap this in a full mesh sweep over all vertices
    - Feed the resulting FOGC sets into the VBD solver
""")
