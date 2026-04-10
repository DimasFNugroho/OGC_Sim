"""
Building Algorithm 2 from scratch — step by step
=================================================
Run this file:
    python3 explore/m2/learn_algorithm2.py

Each step pauses on a plot window.
Close the window → the next step runs.

You can add, change, or comment out anything.
The goal is for you to understand each line before moving on.

How Algorithm 2 relates to Algorithm 1
---------------------------------------
Algorithm 1 asks:  "is this VERTEX close to any TRIANGLE?"
Algorithm 2 asks:  "is this EDGE   close to any other EDGE?"

The structure is almost identical:

  Alg 1                         Alg 2
  ------                        ------
  query = vertex v              query = edge e
  loop over triangles t         loop over edges e'
  skip if v ∈ t (adjacent)      skip if e ∩ e' ≠ ∅ (adjacent)
  dist = point_triangle_dist    dist = edge_edge_dist
  update d_min_v, d_min_t       update d_min_e
  if dist < r:                  if dist < r:
    find sub-feature a on t       find sub-feature a on e'
    dedup check                   dedup check
    feasibility gate              feasibility gate
    record in FOGC(v)             record in EOGC(e)

The only real difference:
  • The distance primitive is edge-to-edge instead of point-to-triangle
  • The sub-feature on e' is simpler: only VERTEX or INTERIOR
    (an edge has 2 endpoints and 1 interior — no "face" concept)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from edge_distance import edge_edge_distance, ClosestFeatureOnEdge

from ogc_sim.geometry.mesh      import Mesh
from ogc_sim.geometry.gauss_map import PolyhedralGaussMap, VertexType


def pause(title):
    """Add a title to the current figure and show it."""
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()


# ============================================================
# STEP 1 — One edge, one other edge, one distance
# ============================================================
# Before anything else: what does "distance between two edges"
# mean?
#
# edge_edge_distance(p, q, r, s) returns:
#   dist     — the shortest distance between the two segments
#   cp_e1    — the closest point ON edge e1 (p→q)
#   t1       — where cp_e1 is on e1 (0=at p, 1=at q)
#   cp_e2    — the closest point ON edge e2 (r→s)
#   t2       — where cp_e2 is on e2 (0=at r, 1=at s)
#   feature  — WHERE on e2 the closest point landed:
#                INTERIOR or VERTEX
#   feat_idx — which endpoint (0=r, 1=s) if VERTEX, else -1
# ============================================================

print("=" * 50)
print("STEP 1 — Distance between two edges")
print("=" * 50)

# Two skew (non-parallel, non-intersecting) segments
e1_p = np.array([0.0, 0.0, 0.5])   # edge 1 start
e1_q = np.array([2.0, 0.0, 0.5])   # edge 1 end

e2_r = np.array([1.0, -1.0, 0.0])  # edge 2 start
e2_s = np.array([1.0,  1.0, 0.0])  # edge 2 end

dist, cp_e1, t1, cp_e2, t2, feature, feat_idx = \
    edge_edge_distance(e1_p, e1_q, e2_r, e2_s)

print(f"""
  Edge e1: p={e1_p}  →  q={e1_q}
  Edge e2: r={e2_r}  →  s={e2_s}

  Result:
    dist    = {dist:.4f}   ← the shortest gap between the two edges
    cp_e1   = {np.round(cp_e1, 3)}   ← closest point ON e1
    t1      = {t1:.3f}    ← position along e1  (0=at p, 1=at q)
    cp_e2   = {np.round(cp_e2, 3)}   ← closest point ON e2
    t2      = {t2:.3f}    ← position along e2  (0=at r, 1=at s)
    feature = {feature}   ← where cp_e2 sits on e2
    feat_idx= {feat_idx}   ← -1 means interior
""")

# --- Plot ---
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")

# Draw both edges
ax.plot([e1_p[0], e1_q[0]], [e1_p[1], e1_q[1]], [e1_p[2], e1_q[2]],
        color="steelblue", lw=3, label="e1 (query)")
ax.plot([e2_r[0], e2_s[0]], [e2_r[1], e2_s[1]], [e2_r[2], e2_s[2]],
        color="darkorange", lw=3, label="e2 (target)")

# Label endpoints
for name, pt, color in [("p", e1_p, "steelblue"), ("q", e1_q, "steelblue"),
                         ("r", e2_r, "darkorange"), ("s", e2_s, "darkorange")]:
    ax.scatter(*pt, color=color, s=50)
    ax.text(*pt + np.array([0.04, 0.04, 0.04]), name, fontsize=9, color=color)

# Draw the two closest points
ax.scatter(*cp_e1, color="steelblue", s=80, zorder=9)
ax.scatter(*cp_e2, color="darkorange", s=80, zorder=9)

# Draw the distance line between them
ax.plot([cp_e1[0], cp_e2[0]], [cp_e1[1], cp_e2[1]], [cp_e1[2], cp_e2[2]],
        color="black", lw=2, linestyle="--", label=f"dist = {dist:.3f}")
mid = (cp_e1 + cp_e2) / 2
ax.text(*mid + np.array([0.06, 0, 0]), f"dist={dist:.3f}", fontsize=9)

ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_xlim(-0.3, 2.3); ax.set_ylim(-1.3, 1.3); ax.set_zlim(-0.2, 0.8)
ax.view_init(elev=28, azim=-55)
ax.legend(fontsize=8)
pause("Step 1 — Distance between two edges\n"
      "(dashed line = shortest path, dots = closest points)")


# ============================================================
# STEP 2 — The contact radius r
# ============================================================
# Same idea as Algorithm 1 Step 2.
# An edge-edge contact is reported only when dist < r.
#
# Algorithm 2, line 8:  "if d < r then ..."
# ============================================================

print("\n" + "=" * 50)
print("STEP 2 — The contact radius r")
print("=" * 50)

r = 0.6   # contact radius — try changing this value

is_contact = dist < r
print(f"""
  r = {r}  (contact radius — you can change this)
  dist = {dist:.4f}
  dist < r ? → {is_contact}

  {'CONTACT DETECTED' if is_contact else 'No contact (too far away)'}
""")

# --- Plot ---
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")

color = "green" if is_contact else "red"

ax.plot([e1_p[0], e1_q[0]], [e1_p[1], e1_q[1]], [e1_p[2], e1_q[2]],
        color="steelblue", lw=3, label="e1 (query)")
ax.plot([e2_r[0], e2_s[0]], [e2_r[1], e2_s[1]], [e2_r[2], e2_s[2]],
        color="darkorange", lw=3, label="e2 (target)")

ax.scatter(*cp_e1, color=color, s=80, zorder=9)
ax.scatter(*cp_e2, color=color, s=80, zorder=9)
ax.plot([cp_e1[0], cp_e2[0]], [cp_e1[1], cp_e2[1]], [cp_e1[2], cp_e2[2]],
        color=color, lw=2, linestyle="--",
        label=f"{'CONTACT' if is_contact else 'no contact'}  d={dist:.3f}, r={r}")

ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_xlim(-0.3, 2.3); ax.set_ylim(-1.3, 1.3); ax.set_zlim(-0.2, 0.8)
ax.view_init(elev=28, azim=-55)
ax.legend(fontsize=8)
pause(f"Step 2 — Contact radius r={r}\n"
      f"({'inside' if is_contact else 'outside'} → "
      f"{'contact detected' if is_contact else 'no contact'})")


# ============================================================
# STEP 3 — Two types of closest feature on e2
# ============================================================
# In Algorithm 1 there were THREE feature types on a triangle:
# FACE_INTERIOR, EDGE, VERTEX.
#
# For an edge there are only TWO:
#
#   INTERIOR  — the closest point is somewhere in the middle
#               of e2 (t2 strictly between 0 and 1)
#   VERTEX    — the closest point is at one of the endpoints
#               (t2 ≈ 0 → at r,  t2 ≈ 1 → at s)
#
# Why does this matter?
#   INTERIOR → the edge e2 is the "owner" of this contact.
#              Always valid, no extra check needed.
#   VERTEX   → the endpoint VERTEX is the "owner".
#              Need a feasibility check (Eq. 15) — the same
#              idea as the vertex cone check in Algorithm 1.
# ============================================================

print("\n" + "=" * 50)
print("STEP 3 — Two types of closest feature on e2")
print("=" * 50)

# Three example cases: interior, at endpoint r, at endpoint s
cases = {
    "INTERIOR (t2 in middle)": {
        "p": np.array([0.0, 0.0, 0.5]),
        "q": np.array([2.0, 0.0, 0.5]),
        "r": np.array([1.0, -1.0, 0.0]),
        "s": np.array([1.0,  1.0, 0.0]),
    },
    "VERTEX at r (t2 ≈ 0)": {
        # e2 starts near the query edge (r close) but runs AWAY from it.
        # The unconstrained minimum falls at t2 < 0, so it gets clamped to 0.
        # Geometry: e1 at z=0.5, e2 starts at y=0.4 (above e1's y=0) and
        # runs in +y direction.  dot(r-p, d2)/|d2|² ≈ -0.15 → t2=0 clamped.
        "p": np.array([0.0, 0.0, 0.5]),
        "q": np.array([2.0, 0.0, 0.5]),
        "r": np.array([1.0,  0.4, 0.0]),   # closest endpoint — closest to e1
        "s": np.array([1.0,  3.0, 0.0]),   # far endpoint — runs away
    },
    "VERTEX at s (t2 ≈ 1)": {
        # Mirror of the above: e2 starts far and approaches from the other end.
        "p": np.array([0.0, 0.0, 0.5]),
        "q": np.array([2.0, 0.0, 0.5]),
        "r": np.array([1.0, -3.0, 0.0]),   # far endpoint — runs away
        "s": np.array([1.0, -0.4, 0.0]),   # closest endpoint — closest to e1
    },
}

fig = plt.figure(figsize=(14, 5))
feature_colors = {
    ClosestFeatureOnEdge.INTERIOR: "#2ecc71",
    ClosestFeatureOnEdge.VERTEX:   "#9b59b6",
}

for i, (case_name, pts) in enumerate(cases.items()):
    d, c1, t1_, c2, t2_, feat, fidx = edge_edge_distance(
        pts["p"], pts["q"], pts["r"], pts["s"]
    )
    print(f"  {case_name}")
    print(f"    dist={d:.3f}  t2={t2_:.3f}  feature={feat}  feat_idx={fidx}")

    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    ax.plot(*zip(pts["p"], pts["q"]), color="steelblue", lw=3, label="e1")
    ax.plot(*zip(pts["r"], pts["s"]), color="darkorange", lw=3, label="e2")

    for name, pt, col in [("p", pts["p"], "steelblue"), ("q", pts["q"], "steelblue"),
                           ("r", pts["r"], "darkorange"), ("s", pts["s"], "darkorange")]:
        ax.scatter(*pt, color=col, s=40)
        ax.text(*pt + np.array([0.04, 0.04, 0.04]), name, fontsize=8, color=col)

    f_color = feature_colors[feat]
    ax.scatter(*c1, color=f_color, s=70, zorder=9)
    ax.scatter(*c2, color=f_color, s=70, zorder=9)
    ax.plot([c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]],
            color=f_color, lw=2, linestyle="--")
    ax.text(*c2 + np.array([0.05, 0.05, 0.05]),
            f"cp_e2\n({feat})", fontsize=7, color=f_color)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_xlim(-0.3, 2.3); ax.set_ylim(-2.3, 2.3); ax.set_zlim(-0.1, 0.8)
    ax.view_init(elev=28, azim=-55)
    ax.set_title(f"{case_name}\nt2={t2_:.3f}", fontsize=9)

pause("Step 3 — Two types of closest feature on e2\n"
      "(green = INTERIOR, purple = VERTEX)")


# ============================================================
# STEP 4 — Loop over edges, skip adjacent
# ============================================================
# Now we have a mesh with several edges.
# For each pair of edges we compute their distance.
#
# One important rule: skip any edge that SHARES a vertex with
# the query edge e.  Those edges are "adjacent" — they are
# connected at a corner.  Their distance to e can be very
# small (even 0) just because they share an endpoint.
# That is not a collision — it is just the mesh topology.
#
# Algorithm 2 line 5:  "if e ∩ e' ≠ ∅ then continue"
# In code:  if query_edge_vertices & target_edge_vertices: skip
# ============================================================

print("\n" + "=" * 50)
print("STEP 4 — Loop over edges, skip adjacent")
print("=" * 50)

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

# Pick edge 0 as our query edge
query_e_idx = 0
q_a_idx = int(mesh.E[query_e_idx][0])
q_b_idx = int(mesh.E[query_e_idx][1])
query_e_verts = {q_a_idx, q_b_idx}
e_p = mesh.V[q_a_idx]
e_q = mesh.V[q_b_idx]

r = 1.5   # contact radius

print(f"\n  Mesh: {mesh.num_vertices} vertices, {mesh.num_edges} edges")
print(f"  Query edge e[{query_e_idx}] = V{q_a_idx}→V{q_b_idx}  {e_p} → {e_q}")
print(f"  r = {r}")
print(f"\n  Looping over all other edges...")

contacts = []

for e2_idx in range(mesh.num_edges):
    if e2_idx == query_e_idx:
        continue

    t_a_idx = int(mesh.E[e2_idx][0])
    t_b_idx = int(mesh.E[e2_idx][1])
    target_e_verts = {t_a_idx, t_b_idx}

    e_r = mesh.V[t_a_idx]
    e_s = mesh.V[t_b_idx]

    dist, cp_e1, t1, cp_e2, t2, feature, feat_idx = \
        edge_edge_distance(e_p, e_q, e_r, e_s)

    # Algorithm 2 line 5: skip adjacent edges
    if query_e_verts & target_e_verts:
        print(f"    e[{e2_idx}] V{t_a_idx}→V{t_b_idx}: SKIPPED (shares a vertex)")
        continue

    within = dist < r
    print(f"    e[{e2_idx}] V{t_a_idx}→V{t_b_idx}: dist={dist:.3f}  "
          f"feature={feature}  {'← CONTACT' if within else '(too far)'}")

    if within:
        contacts.append((e2_idx, dist, cp_e1, cp_e2, feature))

print(f"\n  Contacts found: edge indices {[c[0] for c in contacts]}")

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(12, 5))

for ax, title, skip_adj in [
    (ax1, "WITHOUT skip\n(counts adjacent edges)", False),
    (ax2, "WITH skip  (Algorithm 2 line 5)\n(ignores adjacent edges)", True),
]:
    # Draw all mesh edges
    for ei in range(mesh.num_edges):
        va, vb = mesh.V[int(mesh.E[ei][0])], mesh.V[int(mesh.E[ei][1])]
        is_query = (ei == query_e_idx)
        color = "steelblue" if is_query else "#aaa"
        lw    = 3 if is_query else 1.5
        ax.plot([va[0],vb[0]], [va[1],vb[1]], [va[2],vb[2]], color=color, lw=lw)
        mid = (va + vb) / 2
        ax.text(mid[0], mid[1], mid[2]+0.06, f"e{ei}", fontsize=8, color=color, ha="center")

    for v in mesh.V:
        ax.scatter(*v, color="#555", s=20)

    # Draw contacts (or lack of skip)
    for e2_idx in range(mesh.num_edges):
        if e2_idx == query_e_idx:
            continue
        t_a_idx = int(mesh.E[e2_idx][0])
        t_b_idx = int(mesh.E[e2_idx][1])
        target_e_verts = {t_a_idx, t_b_idx}
        e_r = mesh.V[t_a_idx]
        e_s = mesh.V[t_b_idx]
        dist_, cp1_, t1_, cp2_, t2_, feat_, _ = edge_edge_distance(e_p, e_q, e_r, e_s)
        is_adj = bool(query_e_verts & target_e_verts)

        if skip_adj and is_adj:
            continue
        if dist_ < r:
            c = "red" if is_adj else "green"
            ax.scatter(*cp1_, color=c, s=50, zorder=9)
            ax.scatter(*cp2_, color=c, s=50, zorder=9)
            ax.plot([cp1_[0],cp2_[0]],[cp1_[1],cp2_[1]],[cp1_[2],cp2_[2]],
                    color=c, lw=1.5, linestyle="--")

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_xlim(-0.3, 2.5); ax.set_ylim(-0.3, 2.5); ax.set_zlim(-0.1, 0.5)
    ax.view_init(elev=30, azim=-50)
    ax.set_title(title, fontsize=9)

pause("Step 4 — Skip adjacent edges\n"
      "(red = wrongly counted adjacent edge, green = real contact)")


# ============================================================
# STEP 5 — d_min bookkeeping
# ============================================================
# Same idea as Algorithm 1 Step 2 (d_min_v).
#
# Every edge e' that we visit updates d_min_e — the smallest
# distance seen from query edge e to any non-adjacent edge.
# This happens REGARDLESS of whether dist < r.
#
# Algorithm 2 lines 1, 7:
#   d_min_e = r_q      ← start at the upper bound
#   d_min_e = min(d, d_min_e)
#
# d_min_e feeds into the conservative bound b_v (Eq. 21-24):
#   d_min_e_v = min over all edges incident to v of d_min_e
#   b_v = γ_p * min(d_min_v, d_min_e_v, d_min_t_v)
#
# The larger d_min_e is, the more freedom each vertex has to
# move before re-detection is needed.
# ============================================================

print("\n" + "=" * 50)
print("STEP 5 — d_min bookkeeping")
print("=" * 50)

r_q = 2.0   # query radius (r_q >= r)
r   = 1.5

d_min_e = r_q   # Algorithm 2 line 1: initialise to r_q

print(f"\n  r={r},  r_q={r_q}")
print(f"  d_min_e starts at r_q = {r_q}  (the safe upper bound)")
print(f"\n  Updating d_min_e as we visit each non-adjacent edge:")

for e2_idx in range(mesh.num_edges):
    if e2_idx == query_e_idx:
        continue
    t_a_idx = int(mesh.E[e2_idx][0])
    t_b_idx = int(mesh.E[e2_idx][1])
    if {t_a_idx, t_b_idx} & query_e_verts:
        continue

    e_r = mesh.V[t_a_idx]
    e_s = mesh.V[t_b_idx]
    dist_, *_ = edge_edge_distance(e_p, e_q, e_r, e_s)

    old = d_min_e
    d_min_e = min(d_min_e, dist_)
    print(f"    e[{e2_idx}]: dist={dist_:.3f}  "
          f"d_min_e: {old:.3f} → {d_min_e:.3f}")

gamma_p = 0.45
b_v_approx = gamma_p * d_min_e
print(f"\n  Final d_min_e = {d_min_e:.3f}")
print(f"  Conservative bound preview (γ_p={gamma_p}):")
print(f"    b_v ≈ γ_p * d_min_e = {gamma_p} * {d_min_e:.3f} = {b_v_approx:.3f}")
print(f"    → any vertex of e may move at most {b_v_approx:.3f} units")

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(12, 5))

# Left: show all distances from query edge to others
ax1.set_title("d_min_e: distance from e[0] to each non-adjacent edge", fontsize=9)
for ei in range(mesh.num_edges):
    va, vb = mesh.V[int(mesh.E[ei][0])], mesh.V[int(mesh.E[ei][1])]
    color = "steelblue" if ei == query_e_idx else "#aaa"
    lw    = 3 if ei == query_e_idx else 1.5
    ax1.plot([va[0],vb[0]], [va[1],vb[1]], [va[2],vb[2]], color=color, lw=lw)

for e2_idx in range(mesh.num_edges):
    if e2_idx == query_e_idx:
        continue
    t_a_idx = int(mesh.E[e2_idx][0])
    t_b_idx = int(mesh.E[e2_idx][1])
    if {t_a_idx, t_b_idx} & query_e_verts:
        continue
    e_r_ = mesh.V[t_a_idx]
    e_s_ = mesh.V[t_b_idx]
    dist_, cp1_, t1_, cp2_, *_ = edge_edge_distance(e_p, e_q, e_r_, e_s_)
    is_dmin = (abs(dist_ - d_min_e) < 1e-6)
    c = "#e74c3c" if is_dmin else "#95a5a6"
    ax1.plot([cp1_[0],cp2_[0]], [cp1_[1],cp2_[1]], [cp1_[2],cp2_[2]],
             color=c, lw=1.5 if is_dmin else 1.0, linestyle="--")
    mid_ = (cp1_ + cp2_) / 2
    ax1.text(mid_[0]+0.04, mid_[1], mid_[2]+0.05, f"d={dist_:.2f}",
             fontsize=7, color=c)

ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
ax1.set_xlim(-0.3, 2.5); ax1.set_ylim(-0.3, 2.5); ax1.set_zlim(-0.1, 0.5)
ax1.view_init(elev=30, azim=-50)

# Right: show b_v sphere around endpoint vertices of e
ax2.set_title(f"Conservative bound b_v ≈ {b_v_approx:.3f}\n"
              f"(endpoints of e[{query_e_idx}] may each move this far)", fontsize=9)
for ei in range(mesh.num_edges):
    va, vb = mesh.V[int(mesh.E[ei][0])], mesh.V[int(mesh.E[ei][1])]
    color = "steelblue" if ei == query_e_idx else "#aaa"
    ax2.plot([va[0],vb[0]], [va[1],vb[1]], [va[2],vb[2]], color=color, lw=2)

u = np.linspace(0, 2*np.pi, 20)
v_ = np.linspace(0, np.pi, 10)
for pt in [e_p, e_q]:
    ax2.plot_wireframe(
        b_v_approx * np.outer(np.cos(u), np.sin(v_)) + pt[0],
        b_v_approx * np.outer(np.sin(u), np.sin(v_)) + pt[1],
        b_v_approx * np.outer(np.ones(20), np.cos(v_)) + pt[2],
        color="#27ae60", alpha=0.2, linewidth=0.5
    )

ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
ax2.set_xlim(-0.3, 2.5); ax2.set_ylim(-0.3, 2.5); ax2.set_zlim(-0.5, 0.7)
ax2.view_init(elev=30, azim=-50)

pause("Step 5 — d_min bookkeeping\n"
      "(red dashed = d_min_e, green spheres = conservative bound b_v)")


# ============================================================
# STEP 6 — De-duplication
# ============================================================
# In Algorithm 1, the problem was that two TRIANGLES could
# report the same closest feature (a shared edge or vertex).
#
# In Algorithm 2, the same issue appears but from a different
# angle: each edge-edge pair (e, e') is detected TWICE —
# once when e queries e', and once when e' queries e.
#
# BUT there's also the same "shared feature" issue:
# If the closest point on e' lands at one of its endpoints
# (a VERTEX), that vertex is also shared by other edges of
# the mesh.  Multiple target edges e' could report the same
# vertex as the closest feature.
#
# Algorithm 2 line 11:
#   "if {e, a} ∈ EOGC(e) then continue"
#
# 'a' is the closest feature on e'.
# We store it as a global feature index (just like Algorithm 1).
#   INTERIOR → a = e2_idx          (the edge itself)
#   VERTEX   → a = global vertex index
# ============================================================

print("\n" + "=" * 50)
print("STEP 6 — De-duplication")
print("=" * 50)

# Use the SAME 2x2 mesh as before, but put the query edge OUTSIDE it —
# just like Algorithm 1 used a query vertex that wasn't part of the mesh.
#
# The mesh has shared vertex V2 = (0, 2, 0).
# Two mesh edges both end at V2:
#   E1 = V0→V2  (from (0,0,0) to (0,2,0))
#   E4 = V2→V3  (from (0,2,0) to (2,2,0))
#
# We place the query edge at (0, 2, 0.2)→(0, 2, 1.0),
# directly above V2.  Both E1 and E4 will report VERTEX at V2
# as their closest feature.  De-duplication should keep only one.

V_mesh2 = np.array([
    [0., 0., 0.],  # V0
    [2., 0., 0.],  # V1
    [0., 2., 0.],  # V2  ← shared vertex — target of both contacts
    [2., 2., 0.],  # V3
])
T_mesh2 = np.array([[0, 1, 2], [1, 3, 2]])
mesh2 = Mesh.from_arrays(V_mesh2, T_mesh2)

# The query edge is NOT part of the mesh — just two 3D points.
# (No mesh edge index; no adjacency to any mesh vertex.)
ep2 = np.array([0.0, 2.0, 0.2])   # directly above V2
eq2 = np.array([0.0, 2.0, 1.0])
query_verts2 = set()   # external edge — shares no mesh vertices

r2 = 0.4

print(f"\n  Query edge (external): {ep2} → {eq2}")
print(f"  r = {r2}")
print(f"\n  Loop WITHOUT de-duplication:")

eogc_no_dedup = []
eogc_dedup    = []

for e2_idx in range(mesh2.num_edges):
    ta_idx = int(mesh2.E[e2_idx][0])
    tb_idx = int(mesh2.E[e2_idx][1])
    # query_verts2 is empty (external edge) → no adjacency skips

    er = mesh2.V[ta_idx]
    es = mesh2.V[tb_idx]
    dist_, cp1_, t1_, cp2_, t2_, feat_, fidx_ = \
        edge_edge_distance(ep2, eq2, er, es)

    if dist_ >= r2:
        continue

    # Map to global feature index
    if feat_ == ClosestFeatureOnEdge.INTERIOR:
        global_feat = e2_idx
    else:
        # VERTEX: fidx_ is 0 (at er=ta_idx) or 1 (at es=tb_idx)
        global_feat = ta_idx if fidx_ == 0 else tb_idx

    eogc_no_dedup.append(global_feat)

    # De-duplication check — Algorithm 2 line 11
    if global_feat in eogc_dedup:
        print(f"    e[{e2_idx}]: gfeat={global_feat} ({feat_}) "
              f"← DUPLICATE! already in EOGC. Skipping.")
        continue

    eogc_dedup.append(global_feat)
    print(f"    e[{e2_idx}]: dist={dist_:.3f}  gfeat={global_feat}  ({feat_})  ← recorded")

print(f"\n  EOGC without de-dup: {eogc_no_dedup}  ({len(eogc_no_dedup)} entries)")
print(f"  EOGC with de-dup:    {eogc_dedup}   ({len(eogc_dedup)} entries)")

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(12, 5))

for ax, title, eogc in [
    (ax1, f"WITHOUT de-dup: {len(eogc_no_dedup)} contact(s)", eogc_no_dedup),
    (ax2, f"WITH de-dup:    {len(eogc_dedup)} contact(s)",    eogc_dedup),
]:
    # Draw mesh edges
    for ei in range(mesh2.num_edges):
        va, vb = mesh2.V[int(mesh2.E[ei][0])], mesh2.V[int(mesh2.E[ei][1])]
        ax.plot([va[0],vb[0]], [va[1],vb[1]], [va[2],vb[2]], color="#bbb", lw=1.5)

    # Draw the external query edge
    ax.plot([ep2[0],eq2[0]], [ep2[1],eq2[1]], [ep2[2],eq2[2]],
            color="steelblue", lw=3, label="query edge")

    # Draw vertices
    for vi, v in enumerate(mesh2.V):
        ax.scatter(*v, color="#555", s=25, zorder=6)
        ax.text(*v + np.array([0.05,0.05,0.05]), f"V{vi}", fontsize=7, color="#555")

    # Draw contacts
    seen = set()
    for e2_idx in range(mesh2.num_edges):
        ta = int(mesh2.E[e2_idx][0]); tb = int(mesh2.E[e2_idx][1])
        er_ = mesh2.V[ta]; es_ = mesh2.V[tb]
        d_, c1_, _, c2_, _, f_, fi_ = edge_edge_distance(ep2, eq2, er_, es_)
        if d_ >= r2:
            continue
        gf = ta if (f_ == ClosestFeatureOnEdge.VERTEX and fi_ == 0) else \
             tb if (f_ == ClosestFeatureOnEdge.VERTEX and fi_ == 1) else e2_idx
        if gf not in eogc:
            continue
        offset = 0.04 * len([x for x in seen if x == gf])
        seen.add(gf)
        c2_draw = c2_ + np.array([offset, 0, 0])
        ax.scatter(*c2_draw, color="#e67e22", s=50, zorder=9)
        ax.plot([c1_[0],c2_draw[0]],[c1_[1],c2_draw[1]],[c1_[2],c2_draw[2]],
                color="#e67e22", lw=1.5, linestyle="--")

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(-0.3, 2.8); ax.set_zlim(-0.1, 1.2)
    ax.view_init(elev=30, azim=-50)
    ax.set_title(title, fontsize=9)

pause("Step 6 — De-duplication\n"
      "(same shared vertex reported by multiple target edges → keep only one)")


# ============================================================
# STEP 7 — The feasibility gate (Eq. 15)
# ============================================================
# Same role as Eq. 8/9 in Algorithm 1, but for edges.
#
# After de-duplication, Algorithm 2 checks:
#
#   Closest feature on e' is INTERIOR
#     → always valid, record immediately.
#       (e' is the "owner" of this contact region)
#
#   Closest feature on e' is VERTEX v
#     → check Eq. 15: is the contact direction in the
#       normal CONE of vertex v?
#       This is the same normal cone check as Eq. 8 in Alg 1,
#       but now applied from an edge-edge perspective.
#
# If the check fails, a different adjacent edge will correctly
# own this contact — nothing is missed.
#
# Algorithm 2 lines 12–16:
#   if a ∈ V:
#     if checkVertexFeasibleRegionEdgeOffset(x_c, a): record
#   else:
#     record  (interior case, always valid)
# ============================================================

print("\n" + "=" * 50)
print("STEP 7 — The feasibility gate (Eq. 15)")
print("=" * 50)

pgm2 = PolyhedralGaussMap(mesh2)

print(f"""
  The contact direction for edge-edge contact is:

      direction = cp_e1 - cp_e2

  (from the closest point on e2 toward the closest point on e1)

  If the closest feature on e2 is a VERTEX v:
    feasible = pgm.is_in_vertex_normal_cone(direction, v_idx)

  If the closest feature on e2 is INTERIOR:
    feasible = True  (always)

  This is structurally identical to Algorithm 1:
    INTERIOR  → always OK   (like face interior)
    VERTEX    → cone check  (like vertex in Algorithm 1)
    (there is no EDGE case for an edge's sub-features)
""")

print(f"  Query edge (external): {ep2} → {eq2}")
print(f"\n  For each candidate edge e':")

eogc_final = []

for e2_idx in range(mesh2.num_edges):
    ta = int(mesh2.E[e2_idx][0])
    tb = int(mesh2.E[e2_idx][1])
    # query_verts2 is empty — no adjacency skips needed

    er_ = mesh2.V[ta]
    es_ = mesh2.V[tb]
    dist_, cp1_, t1_, cp2_, t2_, feat_, fidx_ = \
        edge_edge_distance(ep2, eq2, er_, es_)

    if dist_ >= r2:
        print(f"    e[{e2_idx}]: dist={dist_:.3f} >= r  → skip")
        continue

    if feat_ == ClosestFeatureOnEdge.INTERIOR:
        global_feat = e2_idx
    else:
        global_feat = ta if fidx_ == 0 else tb

    if global_feat in eogc_final:
        print(f"    e[{e2_idx}]: gfeat={global_feat}  → duplicate, skip")
        continue

    # Feasibility gate — Algorithm 2 lines 12–16
    direction = cp1_ - cp2_   # from e2's closest point toward e1's

    if feat_ == ClosestFeatureOnEdge.INTERIOR:
        feasible = True
        reason   = "INTERIOR → always feasible"
    else:
        feasible = pgm2.is_in_vertex_normal_cone(direction, global_feat)
        reason   = f"Eq. 15: cone check on V[{global_feat}] → {feasible}"

    mark = "CONTACT" if feasible else "rejected (wrong block)"
    print(f"    e[{e2_idx}]: dist={dist_:.3f}  feat={feat_}  "
          f"gfeat={global_feat}  {reason}  {mark}")

    if feasible:
        eogc_final.append(global_feat)

print(f"\n  Final EOGC (external query edge) = {eogc_final}")

# --- Plot ---
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")

# Draw mesh edges
for ei in range(mesh2.num_edges):
    va, vb = mesh2.V[int(mesh2.E[ei][0])], mesh2.V[int(mesh2.E[ei][1])]
    ax.plot([va[0],vb[0]], [va[1],vb[1]], [va[2],vb[2]], color="#bbb", lw=1.5)

# Draw the external query edge
ax.plot([ep2[0],eq2[0]], [ep2[1],eq2[1]], [ep2[2],eq2[2]],
        color="steelblue", lw=3, label="query edge")

# Draw vertex types
type_color = {VertexType.CONVEX: "#2ecc71",
              VertexType.CONCAVE: "#e74c3c",
              VertexType.MIXED: "#f39c12"}
for vi, v in enumerate(mesh2.V):
    if vi >= len(pgm2.vertex_types):
        continue
    vt = pgm2.vertex_types[vi]
    ax.scatter(*v, color=type_color[vt], s=60, zorder=7)
    ax.text(*v + np.array([0.04,0.04,0.04]), f"V{vi}", fontsize=7,
            color=type_color[vt])

# Draw face normals
for ti in range(mesh2.num_triangles):
    cen = mesh2.V[mesh2.T[ti]].mean(axis=0)
    n   = mesh2.face_normals[ti] * 0.3
    ax.quiver(*cen, *n, color="#aaa", lw=1, arrow_length_ratio=0.3)

# Draw accepted contacts
for e2_idx in range(mesh2.num_edges):
    ta = int(mesh2.E[e2_idx][0]); tb = int(mesh2.E[e2_idx][1])
    er_ = mesh2.V[ta]; es_ = mesh2.V[tb]
    d_, c1_, _, c2_, _, f_, fi_ = edge_edge_distance(ep2, eq2, er_, es_)
    if d_ >= r2:
        continue
    gf = ta if (f_==ClosestFeatureOnEdge.VERTEX and fi_==0) else \
         tb if (f_==ClosestFeatureOnEdge.VERTEX and fi_==1) else e2_idx
    if gf in eogc_final:
        ax.scatter(*c1_, color="green", s=50, zorder=9)
        ax.scatter(*c2_, color="green", s=50, zorder=9)
        ax.plot([c1_[0],c2_[0]],[c1_[1],c2_[1]],[c1_[2],c2_[2]],
                color="green", lw=2, linestyle="--")

ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_xlim(-0.5, 2.5); ax.set_ylim(-0.3, 2.8); ax.set_zlim(-0.1, 1.2)
ax.view_init(elev=30, azim=-50)
ax.legend(fontsize=8)
pause("Step 7 — Feasibility gate (Eq. 15)\n"
      "(green = accepted contact, vertex colours = cone type)")


# ============================================================
# PUTTING IT ALL TOGETHER
# ============================================================
# Here is the complete Algorithm 2 as a single clean function.
# Every line maps directly to a step above.
# ============================================================

print("\n" + "=" * 50)
print("COMPLETE — Algorithm 2 assembled")
print("=" * 50)

def algorithm_2(e_idx, mesh, pgm, r):
    """
    Algorithm 2: edgeEdgeContactDetection

    For edge e_idx, find all edges in contact with it.
    Returns EOGC(e) — list of global feature indices in contact.
    """
    ea_idx = int(mesh.E[e_idx][0])
    eb_idx = int(mesh.E[e_idx][1])
    e_verts = {ea_idx, eb_idx}
    e_p = mesh.V[ea_idx]
    e_q = mesh.V[eb_idx]

    eogc    = []
    d_min_e = float("inf")

    for e2_idx in range(mesh.num_edges):
        if e2_idx == e_idx:
            continue

        ta_idx = int(mesh.E[e2_idx][0])
        tb_idx = int(mesh.E[e2_idx][1])

        # --- line 5: skip adjacent edges ---
        if {ta_idx, tb_idx} & e_verts:
            continue

        e_r = mesh.V[ta_idx]
        e_s = mesh.V[tb_idx]

        dist, cp_e1, t1, cp_e2, t2, feature, feat_idx = \
            edge_edge_distance(e_p, e_q, e_r, e_s)

        # --- line 7: update d_min_e ---
        d_min_e = min(d_min_e, dist)

        # --- line 8: only proceed if close enough ---
        if dist >= r:
            continue

        # --- map to global feature index ---
        if feature == ClosestFeatureOnEdge.INTERIOR:
            global_feat = e2_idx
        else:
            global_feat = ta_idx if feat_idx == 0 else tb_idx

        # --- line 11: de-duplication ---
        if global_feat in eogc:
            continue

        # --- lines 12–16: feasibility gate ---
        direction = cp_e1 - cp_e2

        if feature == ClosestFeatureOnEdge.INTERIOR:
            feasible = True
        else:
            feasible = pgm.is_in_vertex_normal_cone(direction, global_feat)

        if feasible:
            eogc.append(global_feat)

    return eogc, d_min_e


# Run it on all edges of the mesh
pgm_mesh = PolyhedralGaussMap(mesh)
print("\n  Running Algorithm 2 on all edges of the small mesh:\n")
for e_idx in range(mesh.num_edges):
    ea = int(mesh.E[e_idx][0])
    eb = int(mesh.E[e_idx][1])
    eogc, d_min_e = algorithm_2(e_idx, mesh, pgm_mesh, r=1.5)
    print(f"    E[{e_idx}] V{ea}→V{eb}: EOGC = {eogc}  d_min_e = {d_min_e:.3f}")

print("""
  Algorithm 2 is now complete.

  How it relates to Algorithm 1:
    Algorithm 1:  vertex  → triangle  (7 Voronoi regions on the triangle)
    Algorithm 2:  edge    → edge      (2 feature types: INTERIOR or VERTEX)

  What comes next:
    - Add the BVH sphere query (centred at edge midpoint, radius r_q + l/2)
    - Add d_min_t bookkeeping for the triangle side
    - Feed EOGC sets into the VBD solver alongside FOGC
""")
