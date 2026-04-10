"""
M2 Detection Explorer — Algorithm 1 (Vertex-Facet Contact Detection)
=====================================================================
Walks through every line of Algorithm 1 step by step, showing exactly
what it does to a concrete pair of triangles.

    python3 explore/m2/m2_detection.py

Steps
-----
  Step 1 — The Setup         : two triangles, query vertex, BVH query
  Step 2 — d_min bookkeeping : how d_min_v and d_min_t are tracked
  Step 3 — De-duplication    : why two triangles can report the same feature
  Step 4 — Feasibility gate  : how Eq. 8/9 decides which triangle "owns" the contact
  Step 5 — Full mesh sweep   : run_contact_detection on a grid mesh

Algorithm 1 (paper, Sec. 4.1) in plain English
-----------------------------------------------
  For each vertex v:
    BVH query → find every triangle t within radius r_q
    For each such triangle t:
      - skip if v is a vertex of t  (they're connected: no self-contact)
      - compute exact distance d and closest sub-feature a
      - update d_min bookkeeping (always, even if d >= r)
      - if d < r (close enough to be a contact):
          - skip if a is already recorded (de-duplicate shared edges/vertices)
          - apply Gauss Map feasibility check (Eq. 8 or 9)
          - if feasible: record (v, a) as a contact pair

The Gauss Map check is what makes OGC different from plain distance-based
contact: it assigns each contact to *exactly one* owner feature, preventing
double-counting near edges and vertices.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D           # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ogc_sim.geometry.mesh      import Mesh
from ogc_sim.geometry.bvh       import BVH
from ogc_sim.geometry.gauss_map import PolyhedralGaussMap
from ogc_sim.geometry.distance  import point_triangle_distance, ClosestFeature
from ogc_sim.contact.detection  import (
    vertex_facet_contact_detection,
    run_contact_detection,
    ContactSets,
)

# ======================================================================
# Shared geometry
# ======================================================================

# Two triangles facing each other across a small gap.
# T[0] lies in the z=0 plane, T[1] is above it at z=0.3.
#
#   T[0]: V0(0,0,0)  V1(2,0,0)  V2(1,2,0)   → normal pointing +z
#   T[1]: V3(0,0,0.3) V4(2,0,0.3) V5(1,2,0.3) → same shape, shifted up
#   (These two triangles share NO vertices, so they are non-adjacent.)
#
# We will query from a vertex V6 placed directly above the shared region.

V = np.array([
    [0., 0., 0.],   # 0 — T[0]
    [2., 0., 0.],   # 1 — T[0]
    [1., 2., 0.],   # 2 — T[0]
    [0., 0., 0.6],  # 3 — T[1]
    [2., 0., 0.6],  # 4 — T[1]
    [1., 2., 0.6],  # 5 — T[1]
    [1., 0.8, 0.15], # 6 — QUERY VERTEX (between the two triangles)
])
T = np.array([
    [0, 1, 2],  # T[0]
    [3, 4, 5],  # T[1]
    # Also add a small "roof" triangle that shares V[6] so T_v is non-trivial
    # (we'll use a separate single-vertex query mesh for clarity)
])

# For the query vertex demo, build a mesh that contains ONLY V[6] as a vertex
# but still has the two triangles as "the surface to query against".
# We use a flat double-triangle mesh for cleaner illustrations.

# Simpler: two separate flat triangles + one isolated query vertex
# Build mesh from all 7 vertices but only the 2 triangles
two_tri_mesh = Mesh.from_arrays(V[:6], T)
two_tri_bvh  = BVH(two_tri_mesh)
two_tri_pgm  = PolyhedralGaussMap(two_tri_mesh)

# The query vertex lives outside the mesh — we just pass its position
QUERY_POS = V[6].copy()
R   = 0.25   # contact radius
R_Q = 0.40   # query radius

# ---- Flat 3×3 grid mesh for the full sweep demo ----
V_grid = np.array([
    [0.,0.,0.],[1.,0.,0.],[2.,0.,0.],
    [0.,1.,0.],[1.,1.,0.],[2.,1.,0.],
    [0.,2.,0.],[1.,2.,0.],[2.,2.,0.],
])
T_grid = np.array([
    [0,1,4],[0,4,3],[1,2,5],[1,5,4],
    [3,4,7],[3,7,6],[4,5,8],[4,8,7],
])
grid_mesh = Mesh.from_arrays(V_grid, T_grid)
grid_bvh  = BVH(grid_mesh)
grid_pgm  = PolyhedralGaussMap(grid_mesh)


# ======================================================================
# Drawing helpers
# ======================================================================

def draw_two_tris(ax, highlight_tris=None, alpha=0.30):
    colors = ["#d5e8d4", "#dae8fc"]
    for ti, tri in enumerate(two_tri_mesh.T):
        pts   = two_tri_mesh.V[tri]
        color = "#f8cecc" if (highlight_tris and ti in highlight_tris) else colors[ti]
        a     = 0.55 if (highlight_tris and ti in highlight_tris) else alpha
        ax.add_collection3d(Poly3DCollection(
            [pts], alpha=a, facecolor=color, edgecolor="#666", linewidth=1.0
        ))
        c = pts.mean(axis=0)
        ax.text(c[0], c[1], c[2]+0.03, f"T[{ti}]", fontsize=8, color="#333", ha="center")
    for vi, v in enumerate(two_tri_mesh.V):
        ax.scatter(*v, color="#555", s=20, zorder=6)
        ax.text(v[0]+0.03, v[1]+0.03, v[2]+0.04, f"V{vi}", fontsize=7, color="#555")


def draw_query_vertex(ax, pos=QUERY_POS, label="V_query", color="#e74c3c"):
    ax.scatter(*pos, color=color, s=80, zorder=9)
    ax.text(pos[0]+0.04, pos[1], pos[2]+0.05, label, fontsize=8, color=color)


def set_ax(ax, title, elev=28, azim=-55):
    ax.set_xlim(-0.2, 2.2); ax.set_ylim(-0.3, 2.3); ax.set_zlim(-0.1, 0.8)
    ax.set_xlabel("X", fontsize=8); ax.set_ylabel("Y", fontsize=8); ax.set_zlabel("Z", fontsize=8)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=9)


# ======================================================================
# Steps
# ======================================================================

def step_1():
    """Setup: BVH query and candidate triangles."""
    print("\n══════════════════════════════════════════════════════")
    print(" Step 1 — Setup: BVH Query and Candidate Triangles")
    print("══════════════════════════════════════════════════════")
    print(f"""
  Scene: two flat triangles facing each other, separated by a small gap.
  Query vertex: {QUERY_POS}

  Parameters:
    r   = {R}   (contact radius  — contacts reported when d < r)
    r_q = {R_Q}  (query radius   — BVH sphere query uses this)

  Algorithm 1, line 1: d_min_v = r_q  (start with the upper bound)
  Algorithm 1, line 2: BVH sphere query at x(v), radius r_q
    → returns all triangles whose bounding sphere overlaps the query sphere

  Note: r_q >= r.  The "extra" range beyond r is not used for contact
  reporting — it is only used so that d_min_v is a tighter bound.
  A larger r_q → tighter conservative bounds → better convergence,
  but more BVH candidates to process.
""")

    candidates = two_tri_bvh.sphere_query_triangles(QUERY_POS, R_Q)
    print(f"  BVH candidates (r_q={R_Q}): triangle indices {candidates}")
    for ti in candidates:
        tri = two_tri_mesh.T[ti]
        a, b, c = two_tri_mesh.V[tri[0]], two_tri_mesh.V[tri[1]], two_tri_mesh.V[tri[2]]
        dist, cp, feature, fidx = point_triangle_distance(QUERY_POS, a, b, c)
        print(f"    T[{ti}]: dist={dist:.4f}  feature={feature.name}  "
              f"feat_idx={fidx}  cp={np.round(cp, 3)}")

    # --- Plot ---
    fig = plt.figure(figsize=(14, 5))

    # Left: full scene with query vertex and r_q sphere
    ax1 = fig.add_subplot(131, projection="3d")
    draw_two_tris(ax1)
    draw_query_vertex(ax1)
    # Draw r_q sphere as a wireframe
    u = np.linspace(0, 2*np.pi, 24)
    v = np.linspace(0, np.pi, 12)
    sx = R_Q * np.outer(np.cos(u), np.sin(v)) + QUERY_POS[0]
    sy = R_Q * np.outer(np.sin(u), np.sin(v)) + QUERY_POS[1]
    sz = R_Q * np.outer(np.ones(24), np.cos(v)) + QUERY_POS[2]
    ax1.plot_wireframe(sx, sy, sz, color="#aaa", alpha=0.15, linewidth=0.4)
    # r sphere (smaller)
    sx2 = R * np.outer(np.cos(u), np.sin(v)) + QUERY_POS[0]
    sy2 = R * np.outer(np.sin(u), np.sin(v)) + QUERY_POS[1]
    sz2 = R * np.outer(np.ones(24), np.cos(v)) + QUERY_POS[2]
    ax1.plot_wireframe(sx2, sy2, sz2, color="#e74c3c", alpha=0.2, linewidth=0.4)
    ax1.text(QUERY_POS[0], QUERY_POS[1] - R_Q - 0.05, QUERY_POS[2],
             f"r_q={R_Q}\n(BVH query)", fontsize=7, color="#aaa", ha="center")
    ax1.text(QUERY_POS[0], QUERY_POS[1] - R - 0.05, QUERY_POS[2] - 0.05,
             f"r={R}\n(contact)", fontsize=7, color="#e74c3c", ha="center")
    set_ax(ax1, "Scene: two triangles + query vertex\n(grey sphere=r_q, red sphere=r)")

    # Middle: highlight triangles within r_q
    ax2 = fig.add_subplot(132, projection="3d")
    draw_two_tris(ax2, highlight_tris=set(candidates))
    draw_query_vertex(ax2)
    for ti in candidates:
        cp_tri = two_tri_mesh.V[two_tri_mesh.T[ti]].mean(axis=0)
        ax2.plot([QUERY_POS[0], cp_tri[0]], [QUERY_POS[1], cp_tri[1]],
                 [QUERY_POS[2], cp_tri[2]], color="#f39c12", lw=1.2, ls="--")
    set_ax(ax2, f"BVH candidates: T{candidates}\n(orange = found by sphere query)")

    # Right: distance lines to closest points
    ax3 = fig.add_subplot(133, projection="3d")
    draw_two_tris(ax3)
    draw_query_vertex(ax3)
    for ti in candidates:
        tri = two_tri_mesh.T[ti]
        a, b, c = two_tri_mesh.V[tri[0]], two_tri_mesh.V[tri[1]], two_tri_mesh.V[tri[2]]
        dist, cp, feature, _ = point_triangle_distance(QUERY_POS, a, b, c)
        color = "#27ae60" if dist < R else "#e67e22" if dist < R_Q else "#aaa"
        ax3.scatter(*cp, color=color, s=40, zorder=8)
        ax3.plot([QUERY_POS[0], cp[0]], [QUERY_POS[1], cp[1]],
                 [QUERY_POS[2], cp[2]], color=color, lw=2)
        mid = (QUERY_POS + cp) / 2
        ax3.text(mid[0]+0.04, mid[1], mid[2], f"d={dist:.3f}", fontsize=7, color=color)
    legend = [
        Line2D([0],[0], color="#27ae60", lw=2, label=f"d < r={R} (contact)"),
        Line2D([0],[0], color="#e67e22", lw=2, label=f"d < r_q={R_Q} (tracked)"),
    ]
    ax3.legend(handles=legend, fontsize=8, loc="upper right")
    set_ax(ax3, "Exact distances to closest points\n(green = contact, orange = d_min only)")

    plt.suptitle("Step 1 — BVH Query: Finding Candidate Triangles", fontsize=11)
    plt.tight_layout()
    plt.show()


def step_2():
    """d_min bookkeeping."""
    print("\n══════════════════════════════════════════════════════")
    print(" Step 2 — d_min Bookkeeping")
    print("══════════════════════════════════════════════════════")
    print(f"""
  Every triangle visited by the BVH query updates two min-distance values,
  regardless of whether d < r (regardless of whether it is a contact):

    d_min_v  ← min(d_min_v, d)   (per vertex)
    d_min_t  ← min(d_min_t, d)   (per triangle — atomic in GPU version)

  Both start at r_q (the "I haven't seen anything closer than r_q" value).
  These are consumed by Eq. 21 to compute the conservative bound b_v:

    b_v = γ_p * min(d_min_v, d_min_e_v, d_min_t_v)

  The larger d_min_v is, the larger the allowed movement b_v — so it pays
  to use a large r_q: you "see" farther and get a tighter (larger) bound.

  Why start at r_q and not infinity?
  Because the BVH query only looks within r_q.  If the nearest triangle is
  at distance r_q + ε, we'd never know — so the safe upper bound is r_q.
""")

    d_min_t_shared: dict[int, float] = {0: R_Q, 1: R_Q}
    fogc, vogc, d_min_v = vertex_facet_contact_detection(
        v_idx=0,                 # pretend vertex 0 is the query
        mesh=two_tri_mesh,
        bvh=two_tri_bvh,
        pgm=two_tri_pgm,
        r=R, r_q=R_Q,
        d_min_t=d_min_t_shared,
    )
    # That used mesh vertex 0, let's also show it for the external query point
    # by building a temporary mesh
    V_ext = np.vstack([two_tri_mesh.V, QUERY_POS])
    T_ext = two_tri_mesh.T.copy()   # T[0], T[1] don't include vertex 6
    ext_mesh = Mesh.from_arrays(V_ext, T_ext)
    ext_bvh  = BVH(ext_mesh)
    ext_pgm  = PolyhedralGaussMap(ext_mesh)

    d_min_t2: dict[int, float] = {0: R_Q, 1: R_Q}
    fogc2, vogc2, d_min_v2 = vertex_facet_contact_detection(
        v_idx=6,
        mesh=ext_mesh,
        bvh=ext_bvh,
        pgm=ext_pgm,
        r=R, r_q=R_Q,
        d_min_t=d_min_t2,
    )

    print(f"  Query at QUERY_POS={QUERY_POS}  (vertex index 6 in ext_mesh):")
    print(f"    d_min_v    = {d_min_v2:.4f}   (smallest dist to any non-adjacent face)")
    for ti, dmt in d_min_t2.items():
        print(f"    d_min_t[{ti}] = {dmt:.4f}   (smallest dist from T[{ti}] to any vertex)")
    print()
    print(f"  Conservative bound preview (γ_p=0.45, d_min_e assumed = r_q):")
    gamma_p = 0.45
    b_v = gamma_p * d_min_v2
    print(f"    b_v ≈ γ_p * d_min_v = {gamma_p} * {d_min_v2:.4f} = {b_v:.4f}")
    print(f"    → the query vertex may move at most {b_v:.4f} units before")
    print(f"      contact detection must be re-run.")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(12, 5))

    # Left: show d_min_v — distance from query vertex to each triangle
    ax1 = axes[0]
    draw_two_tris(ax1)
    draw_query_vertex(ax1, pos=QUERY_POS)
    candidates = ext_bvh.sphere_query_triangles(QUERY_POS, R_Q)
    for ti in candidates:
        if 6 in ext_mesh.T[ti]:
            continue
        tri = ext_mesh.T[ti]
        a, b, c = ext_mesh.V[tri[0]], ext_mesh.V[tri[1]], ext_mesh.V[tri[2]]
        dist, cp, _, _ = point_triangle_distance(QUERY_POS, a, b, c)
        ax1.scatter(*cp, color="#3498db", s=40, zorder=8)
        ax1.plot([QUERY_POS[0], cp[0]], [QUERY_POS[1], cp[1]], [QUERY_POS[2], cp[2]],
                 color="#3498db", lw=1.5)
        mid = (QUERY_POS + cp) / 2
        ax1.text(mid[0]+0.05, mid[1], mid[2]+0.02, f"d={dist:.3f}", fontsize=7, color="#2980b9")
    ax1.text(QUERY_POS[0]+0.05, QUERY_POS[1]+0.1, QUERY_POS[2]+0.05,
             f"d_min_v\n={d_min_v2:.3f}", fontsize=8, color="#e74c3c", ha="left")
    set_ax(ax1, "d_min_v = min distance to any\nnon-adjacent triangle")

    # Right: show b_v as a sphere around the query vertex
    ax2 = axes[1]
    draw_two_tris(ax2)
    draw_query_vertex(ax2, pos=QUERY_POS)
    u = np.linspace(0, 2*np.pi, 20)
    v_ = np.linspace(0, np.pi, 10)
    ax2.plot_wireframe(
        b_v * np.outer(np.cos(u), np.sin(v_)) + QUERY_POS[0],
        b_v * np.outer(np.sin(u), np.sin(v_)) + QUERY_POS[1],
        b_v * np.outer(np.ones(20), np.cos(v_)) + QUERY_POS[2],
        color="#27ae60", alpha=0.3, linewidth=0.6
    )
    ax2.text(QUERY_POS[0], QUERY_POS[1]-b_v-0.05, QUERY_POS[2],
             f"b_v = γ_p·d_min_v\n= {b_v:.3f}", fontsize=8, color="#27ae60", ha="center")
    set_ax(ax2, "Conservative bound b_v (green sphere)\nvertex may move at most b_v before re-detection")

    plt.suptitle("Step 2 — d_min Bookkeeping and Conservative Bounds", fontsize=11)
    plt.tight_layout()
    plt.show()


def step_3():
    """De-duplication: why shared features appear twice."""
    print("\n══════════════════════════════════════════════════════")
    print(" Step 3 — De-duplication: Shared Edges and Vertices")
    print("══════════════════════════════════════════════════════")
    print("""
  Consider a query vertex v near the SHARED EDGE between two triangles T[0]
  and T[1].  The BVH query returns BOTH triangles.

  For each triangle, `point_triangle_distance` identifies the closest
  sub-feature.  If v is closest to the shared edge, BOTH triangles return:
    - feature = EDGE
    - local_feat_idx = the same edge (just accessed via different triangles)
    - global_feat_idx = the same global edge index

  Without de-duplication:
    → T[0] adds the edge to FOGC(v)
    → T[1] adds the SAME edge to FOGC(v) again
    → The contact is counted TWICE → wrong forces!

  Algorithm 1, line 9:
    if global_feat_idx already in FOGC(v): continue

  This check catches the duplicate on the second visit.  The feasibility
  check (Eq. 8/9) is then irrelevant — we skip it entirely.

  The duplicate happens for:
    - Shared EDGES: visible from 2 adjacent triangles
    - Shared VERTICES: a mesh vertex is a corner of several triangles
    - It does NOT happen for face interiors (each face interior is unique)
""")

    # Build a mesh where two triangles share an edge, and place the query
    # vertex near the midpoint of that shared edge.
    V2 = np.array([
        [0., 0., 0.],  # 0
        [2., 0., 0.],  # 1
        [1., 1., 0.],  # 2
        [1.,-1., 0.],  # 3
    ])
    T2 = np.array([
        [0, 1, 2],   # T[0]: shares edge (0,1) with T[1]
        [1, 0, 3],   # T[1]: shares edge (0,1) = (1,0)
    ])
    mesh2  = Mesh.from_arrays(V2, T2)
    bvh2   = BVH(mesh2)
    pgm2   = PolyhedralGaussMap(mesh2)

    # Query vertex: near the midpoint of the shared edge, slightly above
    q2 = np.array([1.0, 0.0, 0.25])
    r2  = 0.5
    r_q2 = 0.6

    # Find the shared edge index
    shared_e = None
    for ei, (a, b) in enumerate(mesh2.E):
        if {int(a), int(b)} == {0, 1}:
            shared_e = ei
            break

    candidates2 = bvh2.sphere_query_triangles(q2, r_q2)

    print(f"  Query vertex: {q2}")
    print(f"  BVH candidates: {candidates2}")
    print()
    print("  Per-triangle closest feature (before de-duplication check):")
    seen_features: list[int] = []
    for ti in candidates2:
        tri = mesh2.T[ti]
        if 0 in tri and 1 in tri and 2 in tri and 3 in tri:  # all verts → skip (shouldn't happen)
            continue
        a, b, c = mesh2.V[tri[0]], mesh2.V[tri[1]], mesh2.V[tri[2]]
        dist, cp, feature, local_idx = point_triangle_distance(q2, a, b, c)
        if feature == ClosestFeature.EDGE:
            gfi = mesh2.E_t[ti][local_idx]
        elif feature == ClosestFeature.VERTEX:
            gfi = int(tri[local_idx])
        else:
            gfi = ti
        dup = "← DUPLICATE! already seen" if gfi in seen_features else ""
        print(f"    T[{ti}]: dist={dist:.4f}  feature={feature.name}  "
              f"global_feat_idx={gfi}  {dup}")
        if gfi not in seen_features:
            seen_features.append(gfi)

    print()
    print(f"  After de-duplication: FOGC(v_query) = {seen_features}")
    if shared_e is not None:
        print(f"  The shared edge is E[{shared_e}] = {tuple(mesh2.E[shared_e])}")
        print(f"  → contact recorded exactly once, not twice.")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(12, 5))

    for axi, (ax, title, show_dup) in enumerate(zip(
        axes,
        ["WITHOUT de-duplication\n(same contact recorded twice)",
         "WITH de-duplication\n(each contact recorded once)"],
        [True, False]
    )):
        # Draw the two triangles
        colors = ["#d5e8d4", "#dae8fc"]
        for ti, tri in enumerate(mesh2.T):
            pts = mesh2.V[tri]
            ax.add_collection3d(Poly3DCollection(
                [pts], alpha=0.4, facecolor=colors[ti], edgecolor="#666", lw=1.0
            ))
            c = pts.mean(axis=0)
            ax.text(c[0], c[1], c[2]+0.03, f"T[{ti}]", fontsize=8, color="#333")
        # Draw the shared edge in red
        e_a, e_b = mesh2.V[0], mesh2.V[1]
        ax.plot([e_a[0], e_b[0]], [e_a[1], e_b[1]], [e_a[2], e_b[2]],
                color="#e74c3c", lw=3, zorder=8)
        ax.text((e_a[0]+e_b[0])/2, (e_a[1]+e_b[1])/2 - 0.12, 0.02,
                f"shared E[{shared_e}]", fontsize=7, color="#c0392b", ha="center")
        # Draw the query vertex
        ax.scatter(*q2, color="#8e44ad", s=80, zorder=9)
        ax.text(q2[0]+0.05, q2[1]+0.05, q2[2]+0.04, "v_query", fontsize=8, color="#8e44ad")
        # Contact lines
        drawn: set[int] = set()
        n_contacts = 0
        for ti in candidates2:
            if shared_e in drawn and show_dup is False:
                continue
            tri = mesh2.T[ti]
            a, b, c = mesh2.V[tri[0]], mesh2.V[tri[1]], mesh2.V[tri[2]]
            dist, cp, feature, local_idx = point_triangle_distance(q2, a, b, c)
            if feature == ClosestFeature.EDGE:
                gfi = mesh2.E_t[ti][local_idx]
            elif feature == ClosestFeature.VERTEX:
                gfi = int(tri[local_idx])
            else:
                gfi = ti
            if not show_dup and gfi in drawn:
                continue
            drawn.add(gfi)
            n_contacts += 1
            ax.scatter(*cp, color="#e67e22", s=40, zorder=8)
            ax.plot([q2[0], cp[0]], [q2[1], cp[1]], [q2[2], cp[2]],
                    color="#e67e22", lw=2)
        ax.text(0.5, -0.8, 0.28, f"{n_contacts} contact(s) recorded",
                fontsize=9, color="#c0392b" if n_contacts > 1 else "#27ae60",
                ha="center", fontweight="bold")
        ax.set_xlim(-0.2, 2.2); ax.set_ylim(-1.2, 1.4); ax.set_zlim(-0.1, 0.5)
        ax.set_xlabel("X", fontsize=8); ax.set_ylabel("Y", fontsize=8); ax.set_zlabel("Z", fontsize=8)
        ax.view_init(elev=28, azim=-55)
        ax.set_title(title, fontsize=9)

    plt.suptitle("Step 3 — De-duplication: Why Shared Features Need the FOGC Check", fontsize=11)
    plt.tight_layout()
    plt.show()


def step_4():
    """Feasibility gate: Eq. 8 / 9 in action."""
    print("\n══════════════════════════════════════════════════════")
    print(" Step 4 — The Feasibility Gate (Eq. 8 and 9)")
    print("══════════════════════════════════════════════════════")
    print("""
  After de-duplication, Algorithm 1 applies one final check:

    Is the contact direction (v - closest_point) in the normal set of
    the closest sub-feature?

    Feature = FACE_INTERIOR  →  always YES   (skip check, Alg. 1 line 18)
    Feature = VERTEX v'      →  check Eq. 8  (direction ∈ normal_cone(v'))
    Feature = EDGE e'        →  check Eq. 9  (direction ∈ normal_slab(e'))

  What happens when it's NO?
    The contact direction points "into" a neighboring feature's block.
    The other feature (the adjacent edge or triangle) will correctly
    pick up this contact when it is tested.  Nothing is missed — it just
    belongs to a different owner.

  Example: query vertex v above the EDGE between T[0] and T[1].
    - T[0] detects closest feature = shared edge, checks slab → YES → records it
    - T[1] detects closest feature = SAME shared edge → duplicate, SKIPPED

  Example: query vertex above a VERTEX shared by many triangles.
    - First triangle: closest feature = shared vertex, checks cone → YES → records it
    - All other triangles: same shared vertex → duplicate, SKIPPED

  So Eq. 8/9 is really about CORRECTNESS near the boundary between blocks.
  Inside a block (far from any feature boundary), the check always passes.
""")

    # Build a mesh with a clear convex vertex to demonstrate Eq. 8
    V_corner = np.array([
        [0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.]
    ])
    T_corner = np.array([[0,2,1],[0,3,2],[0,1,3]])
    corner_mesh = Mesh.from_arrays(V_corner, T_corner)
    corner_bvh  = BVH(corner_mesh)
    corner_pgm  = PolyhedralGaussMap(corner_mesh)

    # Query points at various positions around the corner
    query_pts = [
        ("outward corner (−0.2,−0.2,−0.2)", np.array([-0.2,-0.2,-0.2])),
        ("above z=0 face  (0.3, 0.3,−0.2)", np.array([ 0.3, 0.3,-0.2])),
        ("beside x=0 face (−0.2, 0.3, 0.3)", np.array([-0.2, 0.3, 0.3])),
        ("inward (+0.2, +0.2, +0.2)",        np.array([ 0.2, 0.2, 0.2])),
    ]
    r_c  = 0.4
    r_qc = 0.5

    print("  Corner mesh queries:")
    print(f"  {'Query':<42}  {'dist':>6}  {'feature':>14}  {'global_feat':>11}  {'feasible':>8}  contact")
    print("  " + "-"*95)
    for label, qp in query_pts:
        d_min_t_tmp: dict[int, float] = {0: r_qc, 1: r_qc, 2: r_qc}
        # Use a mesh that treats qp as an isolated vertex
        V_ext2 = np.vstack([corner_mesh.V, qp])
        ext2_mesh = Mesh.from_arrays(V_ext2, T_corner)
        ext2_bvh  = BVH(ext2_mesh)
        ext2_pgm  = PolyhedralGaussMap(ext2_mesh)
        fogc_, _, d_ = vertex_facet_contact_detection(
            v_idx=4, mesh=ext2_mesh, bvh=ext2_bvh,
            pgm=ext2_pgm, r=r_c, r_q=r_qc,
            d_min_t=d_min_t_tmp,
        )
        print(f"  {label:<42}  {d_:>6.3f}  "
              f"{'(see below)':>14}  "
              f"{'FOGC='+str(fogc_):>11}  "
              f"{'YES' if fogc_ else 'NO':>8}  "
              f"{'✓' if fogc_ else '✗'}")

    # --- Plot ---
    fig = plt.figure(figsize=(14, 5))

    ax1 = fig.add_subplot(131, projection="3d")
    # Draw corner mesh
    fc = ["#d5e8d4","#dae8fc","#fff2cc"]
    for ti, tri in enumerate(corner_mesh.T):
        pts = corner_mesh.V[tri]
        ax1.add_collection3d(Poly3DCollection(
            [pts], alpha=0.35, facecolor=fc[ti], edgecolor="#666", lw=1.0
        ))
        c = pts.mean(axis=0)
        ax1.text(c[0],c[1],c[2]+0.03,f"T[{ti}]",fontsize=7,color="#333",ha="center")
    # Corner vertex
    ax1.scatter(*corner_mesh.V[0], color="#8e44ad", s=80, zorder=8)
    ax1.text(-0.07,-0.07,0.06,"V[0]",fontsize=8,color="#8e44ad")
    # Face normals
    for ti, tri in enumerate(corner_mesh.T):
        c = corner_mesh.V[tri].mean(axis=0)
        n = corner_mesh.face_normals[ti]*0.2
        ax1.quiver(*c,*n,color="#888",lw=1.2,arrow_length_ratio=0.3)

    for label, qp in query_pts:
        V_ext2 = np.vstack([corner_mesh.V, qp])
        ext2_mesh = Mesh.from_arrays(V_ext2, T_corner)
        ext2_bvh  = BVH(ext2_mesh)
        ext2_pgm  = PolyhedralGaussMap(ext2_mesh)
        d_min_t_tmp = {0: r_qc, 1: r_qc, 2: r_qc}
        fogc_, _, _ = vertex_facet_contact_detection(
            v_idx=4, mesh=ext2_mesh, bvh=ext2_bvh,
            pgm=ext2_pgm, r=r_c, r_q=r_qc,
            d_min_t=d_min_t_tmp,
        )
        color = "#27ae60" if fogc_ else "#e74c3c"
        ax1.scatter(*qp, color=color, s=60, zorder=9)

    from matplotlib.lines import Line2D
    leg = [Line2D([0],[0],color="#27ae60",lw=2,label="contact detected"),
           Line2D([0],[0],color="#e74c3c",lw=2,label="no contact (infeasible)")]
    ax1.legend(handles=leg, fontsize=8, loc="upper right")
    ax1.set_xlim(-0.4,1.1); ax1.set_ylim(-0.4,1.1); ax1.set_zlim(-0.4,1.1)
    ax1.set_xlabel("X",fontsize=8); ax1.set_ylabel("Y",fontsize=8); ax1.set_zlabel("Z",fontsize=8)
    ax1.view_init(elev=25,azim=35)
    ax1.set_title("Feasibility gate on corner mesh\n(green=contact, red=rejected)", fontsize=9)

    # Middle: the full pipeline as a flowchart in 2D
    ax2 = fig.add_subplot(132)
    ax2.axis("off")
    ax2.set_xlim(0,10); ax2.set_ylim(0,14)
    boxes = [
        (5, 13.0, "BVH query → candidate triangles", "#d6eaf8"),
        (5, 11.2, "For each triangle t:", "#ecf0f1"),
        (5,  9.8, "v ⊂ t ?  → SKIP (adjacent)", "#fadbd8"),
        (5,  8.4, "compute dist, cp, feature, feat_idx", "#d5f5e3"),
        (5,  7.0, "update d_min_v, d_min_t  (always)", "#fef9e7"),
        (5,  5.6, "d < r ?  → if NO, skip", "#fadbd8"),
        (5,  4.2, "global_feat already in FOGC?  → SKIP", "#fadbd8"),
        (5,  2.8, "Feasibility check (Eq. 8 or 9)", "#d6eaf8"),
        (5,  1.4, "→ FOGC(v).append(global_feat)", "#d5f5e3"),
    ]
    for bx, by, text, color in boxes:
        ax2.add_patch(plt.Rectangle((bx-4.5, by-0.5), 9, 0.9,
                      facecolor=color, edgecolor="#aaa", lw=0.8))
        ax2.text(bx, by, text, ha="center", va="center", fontsize=8)
    for i in range(len(boxes)-1):
        ax2.annotate("", xy=(5, boxes[i+1][1]+0.4), xytext=(5, boxes[i][1]-0.5),
                     arrowprops=dict(arrowstyle="->", color="#555", lw=1.0))
    ax2.set_title("Algorithm 1 flowchart", fontsize=9)

    # Right: two triangles with a query near the shared boundary
    ax3 = fig.add_subplot(133, projection="3d")
    draw_two_tris(ax3)
    # Two query points: one clearly inside a face block, one near the boundary
    q_clear = np.array([1.0, 0.8, 0.4])   # directly above face interior
    q_edge  = np.array([1.0, 0.0, 0.13])  # near the base edge of T[0]
    for qp, label in [(q_clear, "face\nblock"), (q_edge, "edge\nboundary")]:
        V_ext3 = np.vstack([two_tri_mesh.V, qp])
        ext3_mesh = Mesh.from_arrays(V_ext3, T)
        ext3_bvh  = BVH(ext3_mesh)
        ext3_pgm  = PolyhedralGaussMap(ext3_mesh)
        d_min_t3  = {0: R_Q, 1: R_Q}
        fogc3, _, _ = vertex_facet_contact_detection(
            v_idx=6, mesh=ext3_mesh, bvh=ext3_bvh,
            pgm=ext3_pgm, r=R, r_q=R_Q,
            d_min_t=d_min_t3,
        )
        color = "#27ae60" if fogc3 else "#e74c3c"
        ax3.scatter(*qp, color=color, s=60, zorder=9)
        ax3.text(qp[0]+0.05, qp[1], qp[2]+0.04, label, fontsize=7, color=color)
    ax3.legend(handles=leg, fontsize=8, loc="upper right")
    set_ax(ax3, "Two query positions\n(green=feasible contact, red=no contact)")

    plt.suptitle("Step 4 — Feasibility Gate: Eq. 8 and 9 in Algorithm 1", fontsize=11)
    plt.tight_layout()
    plt.show()


def step_5():
    """Full mesh sweep with run_contact_detection."""
    print("\n══════════════════════════════════════════════════════")
    print(" Step 5 — Full Mesh Sweep: run_contact_detection")
    print("══════════════════════════════════════════════════════")
    print("""
  run_contact_detection loops over all vertices (Algorithm 1) and all
  edges (Algorithm 2) and returns a ContactSets object containing:

    FOGC[v]   — faces in contact with vertex v
    VOGC[t]   — vertices in contact with triangle t
    EOGC[e]   — edges in contact with edge e
    d_min_v   — per-vertex minimum distance
    d_min_t   — per-triangle minimum distance
    d_min_e   — per-edge minimum distance

  We run it on a flat 3×3 grid mesh to see a realistic output.
  Then we add a "foreign" vertex above the mesh surface to trigger contacts.
""")

    # Build a combined mesh: grid + one foreign vertex above it
    q_above = np.array([1.0, 1.0, 0.08])  # above the center of the grid
    V_combined = np.vstack([grid_mesh.V, q_above])
    # The grid triangles don't include vertex index 9 (the foreign vertex)
    combined_mesh = Mesh.from_arrays(V_combined, T_grid)
    combined_bvh  = BVH(combined_mesh)
    combined_pgm  = PolyhedralGaussMap(combined_mesh)

    r_sweep  = 0.12
    r_q_sweep = 0.20

    cs = run_contact_detection(combined_mesh, combined_bvh, combined_pgm,
                               r=r_sweep, r_q=r_q_sweep)

    print(f"  Mesh: {combined_mesh.num_vertices} vertices, "
          f"{combined_mesh.num_triangles} triangles, "
          f"{combined_mesh.num_edges} edges")
    print(f"  r={r_sweep}, r_q={r_q_sweep}")
    print()
    print("  FOGC (non-empty only):")
    for v_idx, contacts in cs.FOGC.items():
        if contacts:
            print(f"    FOGC[{v_idx}] = {contacts}")
    print()
    print("  VOGC (non-empty only):")
    for t_idx, verts in cs.VOGC.items():
        if verts:
            print(f"    VOGC[{t_idx}] = {verts}")
    print()
    print("  d_min_v for foreign vertex 9:", cs.d_min_v.get(9, "—"))
    print("  d_min_t summary (min across all triangles):",
          min(cs.d_min_t.values()))

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(13, 5))

    for ax, title, show_fogc in zip(
        axes,
        ["Grid mesh — contact detection result\n(vertex 9 = foreign vertex above surface)",
         "d_min_v heatmap\n(vertex brightness = min distance)"],
        [True, False]
    ):
        # Draw grid triangles
        for ti, tri in enumerate(grid_mesh.T):
            pts = grid_mesh.V[tri]
            ax.add_collection3d(Poly3DCollection(
                [pts], alpha=0.25, facecolor="#d5e8d4", edgecolor="#888", lw=0.6
            ))

        if show_fogc:
            # Draw contacts as lines from foreign vertex to closest point
            v9_pos = combined_mesh.V[9]
            ax.scatter(*v9_pos, color="#e74c3c", s=80, zorder=9)
            ax.text(v9_pos[0]+0.04, v9_pos[1]+0.04, v9_pos[2]+0.04,
                    "V9\n(foreign)", fontsize=7, color="#c0392b")
            for feat_idx in cs.FOGC.get(9, []):
                # feat_idx could be tri, edge, or vertex index
                # compute closest point
                for ti in range(combined_mesh.num_triangles):
                    tri = combined_mesh.T[ti]
                    a, b, c = combined_mesh.V[tri[0]], combined_mesh.V[tri[1]], combined_mesh.V[tri[2]]
                    dist, cp, _, _ = point_triangle_distance(v9_pos, a, b, c)
                    if dist < r_sweep:
                        ax.scatter(*cp, color="#27ae60", s=35, zorder=8)
                        ax.plot([v9_pos[0],cp[0]],[v9_pos[1],cp[1]],[v9_pos[2],cp[2]],
                                color="#27ae60", lw=1.5)
                        break
        else:
            # d_min_v heatmap
            import matplotlib.cm as cm
            d_vals = np.array([cs.d_min_v.get(vi, r_q_sweep) for vi in range(grid_mesh.num_vertices)])
            norm_vals = (d_vals - d_vals.min()) / (d_vals.max() - d_vals.min() + 1e-10)
            cmap = cm.RdYlGn
            for vi, v in enumerate(grid_mesh.V):
                color = cmap(norm_vals[vi])
                ax.scatter(*v, color=color, s=50, zorder=7)

        ax.set_xlim(-0.2, 2.2); ax.set_ylim(-0.2, 2.2); ax.set_zlim(-0.05, 0.3)
        ax.set_xlabel("X",fontsize=8); ax.set_ylabel("Y",fontsize=8); ax.set_zlabel("Z",fontsize=8)
        ax.view_init(elev=35, azim=-50)
        ax.set_title(title, fontsize=9)

    plt.suptitle("Step 5 — Full Mesh Sweep: run_contact_detection", fontsize=11)
    plt.tight_layout()
    plt.show()


# ======================================================================
# Main
# ======================================================================

def main():
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║         M2 Detection Explorer — Algorithm 1                   ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  Running all 5 steps in sequence.                             ║")
    print("║  Close each plot window to proceed to the next step.          ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    step_1()   # BVH query and candidates
    step_2()   # d_min bookkeeping
    step_3()   # de-duplication
    step_4()   # feasibility gate
    step_5()   # full sweep

    print("\n✓ All steps complete.")


if __name__ == "__main__":
    main()
