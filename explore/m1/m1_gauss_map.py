"""
M1 Gauss Map Explorer
=====================
Walks through the Polyhedral Gauss Map step by step in a single run.

    python3 explore/m1/m1_gauss_map.py

The process (all steps run in sequence, 1 → 2 → 3 → 4):

  Step 1 — Vertex classification : CONVEX / CONCAVE / MIXED, colour-coded
  Step 2 — Edge normal slabs     : the arc of normals between two adjacent faces
  Step 3 — Vertex normal cones   : the fan of normals around a vertex
  Step 4 — Feasibility queries   : is a contact direction inside the cone/slab?

The key insight:
  A contact pair (query vertex → triangle feature) is only valid if the
  direction (query - closest_point) lies inside the *normal set* of that
  feature.  The Gauss Map stores exactly those normal sets.

Paper reference: Sec. 3.1 — Polyhedral Gauss Map.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D                   # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ogc_sim.geometry.mesh      import Mesh
from ogc_sim.geometry.gauss_map import PolyhedralGaussMap, VertexType

# ======================================================================
# PARAMETERS
# ======================================================================

# <<< CHANGE ME (Step 3 / 4) — which vertex of the flat mesh to inspect
FOCUS_VERTEX = 4   # centre vertex has the most incident faces

# <<< CHANGE ME (Step 2 / 4) — which edge of the flat mesh to inspect
FOCUS_EDGE = 2     # edge (0,4) — shared by two interior triangles

# ======================================================================
# Meshes
# ======================================================================

# ---- Flat 3×3 grid (used throughout M1) ----
V_flat = np.array([
    [0.,0.,0.],[1.,0.,0.],[2.,0.,0.],
    [0.,1.,0.],[1.,1.,0.],[2.,1.,0.],
    [0.,2.,0.],[1.,2.,0.],[2.,2.,0.],
])
T_flat = np.array([
    [0,1,4],[0,4,3],[1,2,5],[1,5,4],
    [3,4,7],[3,7,6],[4,5,8],[4,8,7],
])
flat_mesh = Mesh.from_arrays(V_flat, T_flat)
flat_pgm  = PolyhedralGaussMap(flat_mesh)

# ---- Box-corner mesh (3 faces of a unit cube meeting at the origin)
# Normals: (0,0,-1), (-1,0,0), (0,-1,0) — all pointing away from the box.
# V[0] = origin is a CONVEX corner vertex.
V_box = np.array([
    [0.,0.,0.],  # 0  ← shared convex corner
    [1.,0.,0.],  # 1
    [0.,1.,0.],  # 2
    [0.,0.,1.],  # 3
])
T_box = np.array([
    [0,2,1],   # z=0 face  → normal (0,0,-1)
    [0,3,2],   # x=0 face  → normal (-1,0,0)
    [0,1,3],   # y=0 face  → normal (0,-1,0)
])
box_mesh = Mesh.from_arrays(V_box, T_box)
box_pgm  = PolyhedralGaussMap(box_mesh)

# ======================================================================
# Shared helpers
# ======================================================================

TYPE_COLOR = {
    VertexType.CONVEX:  "#2ecc71",   # green
    VertexType.CONCAVE: "#e74c3c",   # red
    VertexType.MIXED:   "#f39c12",   # orange
}


def draw_flat_mesh(ax, highlight_tris=None, highlight_edges=None,
                   highlight_verts=None, vert_colors=None, alpha=0.18):
    for ti, tri in enumerate(flat_mesh.T):
        pts   = flat_mesh.V[tri]
        color = "#dfe6e9"
        a     = alpha
        if highlight_tris is not None:
            color = "#f39c12" if ti in highlight_tris else "#ecf0f1"
            a     = 0.45 if ti in highlight_tris else 0.08
        ax.add_collection3d(Poly3DCollection(
            [pts], alpha=a, facecolor=color, edgecolor="#95a5a6", linewidth=0.8
        ))
        c = flat_mesh.V[tri].mean(axis=0)
        ax.text(c[0], c[1], c[2]+0.04, f"T{ti}", fontsize=7, color="#555", ha="center")

    for ei, (a, b) in enumerate(flat_mesh.E):
        p0, p1 = flat_mesh.V[a], flat_mesh.V[b]
        color  = "#e74c3c" if (highlight_edges and ei in highlight_edges) else "#bbb"
        lw     = 2.5 if (highlight_edges and ei in highlight_edges) else 0.8
        ax.plot([p0[0],p1[0]], [p0[1],p1[1]], [p0[2],p1[2]], color=color, linewidth=lw)

    for vi, v in enumerate(flat_mesh.V):
        c = vert_colors[vi] if vert_colors else "#555"
        s = 80  if (highlight_verts and vi in highlight_verts) else 20
        ax.scatter(*v, color=c, s=s, zorder=6)
        ax.text(v[0]-0.07, v[1]+0.07, v[2]+0.05, str(vi), fontsize=7, color=c)


def draw_box_mesh(ax):
    colors = ["#dfe6e9", "#d0ece7", "#d6eaf8"]
    for ti, tri in enumerate(box_mesh.T):
        pts = box_mesh.V[tri]
        ax.add_collection3d(Poly3DCollection(
            [pts], alpha=0.35, facecolor=colors[ti], edgecolor="#7f8c8d", linewidth=0.8
        ))
    for a, b in box_mesh.E:
        p0, p1 = box_mesh.V[a], box_mesh.V[b]
        ax.plot([p0[0],p1[0]], [p0[1],p1[1]], [p0[2],p1[2]], color="#aaa", linewidth=0.8)


def unit_sphere_surface(n=30):
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi,   n)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(n), np.cos(v))
    return x, y, z


def draw_unit_sphere(ax, alpha=0.05):
    x, y, z = unit_sphere_surface()
    ax.plot_surface(x, y, z, color="#aaa", alpha=alpha, linewidth=0)


def set_ax_flat(ax, title):
    ax.set_xlim(-0.3, 2.3); ax.set_ylim(-0.3, 2.3); ax.set_zlim(-0.3, 1.2)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.view_init(elev=28, azim=-55)
    ax.set_title(title, fontsize=9)


def set_ax_sphere(ax, title):
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_zlim(-1.4, 1.4)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.view_init(elev=25, azim=-50)
    ax.set_title(title, fontsize=9)


def set_ax_box(ax, title):
    ax.set_xlim(-0.1, 1.2); ax.set_ylim(-0.1, 1.2); ax.set_zlim(-0.1, 1.2)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.view_init(elev=25, azim=35)
    ax.set_title(title, fontsize=9)


# ======================================================================
# Steps
# ======================================================================

def step_1():
    """Vertex classification — CONVEX / CONCAVE / MIXED."""
    print("\n══════════════════════════════════════════════════════")
    print(" Step 1 — Vertex Classification")
    print("══════════════════════════════════════════════════════")
    print("""
  The Gauss Map classifies each vertex by the curvature around it.

  How: for each edge incident to vertex v, check whether the two
  adjacent faces form a ridge (convex) or a valley (concave).
    → Convex ridge : the other face's vertex is on the NEGATIVE
                     side of the first face's normal  (sign < 0)
    → Concave valley: the other face's vertex is on the POSITIVE side
    → Flat edge    : sign ≈ 0, treated as convex

  Vertex is CONVEX if all incident edges are ridges (or flat).
  Vertex is CONCAVE if all are valleys.
  Vertex is MIXED   if there is a mix.
""")

    print("  Flat 3×3 mesh:")
    for vi, vt in enumerate(flat_pgm.vertex_types):
        n_faces = len(flat_mesh.T_v[vi])
        print(f"    V[{vi}] {flat_mesh.V[vi]}  → {vt.name:7s}  ({n_faces} incident faces)")

    print()
    print("  Box-corner mesh:")
    for vi, vt in enumerate(box_pgm.vertex_types):
        n_faces = len(box_mesh.T_v[vi])
        print(f"    V[{vi}] {box_mesh.V[vi]}  → {vt.name:7s}  ({n_faces} incident faces)")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(13, 5))

    # Left: flat mesh coloured by type
    vert_colors = [TYPE_COLOR[vt] for vt in flat_pgm.vertex_types]
    draw_flat_mesh(ax1, vert_colors=vert_colors)
    from matplotlib.patches import Patch
    legend = [Patch(facecolor=c, label=t.name) for t, c in TYPE_COLOR.items()]
    ax1.legend(handles=legend, fontsize=8, loc="upper left")
    set_ax_flat(ax1, "Step 1 — Flat mesh: all vertices CONVEX\n(flat surface, every edge is a flat ridge)")

    # Right: box-corner mesh
    draw_box_mesh(ax2)
    for vi, v in enumerate(box_mesh.V):
        vt = box_pgm.vertex_types[vi]
        ax2.scatter(*v, color=TYPE_COLOR[vt], s=80, zorder=6)
        ax2.text(v[0]-0.04, v[1]+0.04, v[2]+0.05, f"V{vi}\n{vt.name}", fontsize=7,
                 color=TYPE_COLOR[vt], ha="center")
    # Face normals as arrows
    for ti, tri in enumerate(box_mesh.T):
        c = box_mesh.V[tri].mean(axis=0)
        n = box_mesh.face_normals[ti] * 0.25
        ax2.quiver(*c, *n, color="#8e44ad", linewidth=1.2, arrow_length_ratio=0.35)
    set_ax_box(ax2, "Step 1 — Box-corner mesh: V[0] is CONVEX\n(three outward-pointing face normals diverge)")

    plt.suptitle("Step 1 — Vertex Classification", fontsize=11)
    plt.tight_layout()
    plt.show()


def step_2():
    """Edge normal slabs."""
    print("\n══════════════════════════════════════════════════════")
    print(" Step 2 — Edge Normal Slabs")
    print("══════════════════════════════════════════════════════")
    print(f"""
  Each edge's *normal slab* is the arc of unit normals that spans
  between the two adjacent face normals n0 and n1:

      slab(e) = {{ normalize(α·n0 + (1-α)·n1) : α ∈ [0,1] }}

  On the unit sphere this is the shorter great-circle arc between n0 and n1.

  Boundary edges (only one adjacent face) have a degenerate slab: n0 = n1.

  Paper use (Eq. 9): a contact direction d is in the slab iff
      dot(d, n0) ≥ 0   AND   dot(d, n1) ≥ 0
""")

    print(f"  Flat mesh, edge E[{FOCUS_EDGE}] = {flat_mesh.E[FOCUS_EDGE]}:")
    n0, n1 = flat_pgm.edge_normal_slabs[FOCUS_EDGE]
    a_, b_ = [int(x) for x in flat_mesh.E[FOCUS_EDGE]]
    adj    = sorted(set(flat_mesh.T_v[a_]) & set(flat_mesh.T_v[b_]))
    print(f"    Adjacent faces : {adj}")
    print(f"    n0 = {np.round(n0, 3)}  (face {adj[0]})")
    print(f"    n1 = {np.round(n1, 3)}  (face {adj[1] if len(adj)>1 else adj[0]})")
    print(f"    → n0 = n1 for flat mesh (slab degenerates to a single point on sphere)")

    print()
    print("  Box-corner mesh, edges:")
    for ei in range(box_mesh.num_edges):
        n0b, n1b = box_pgm.edge_normal_slabs[ei]
        ea, eb   = [int(x) for x in box_mesh.E[ei]]
        adj_box  = sorted(set(box_mesh.T_v[ea]) & set(box_mesh.T_v[eb]))
        bnd      = "(boundary)" if len(adj_box) < 2 else ""
        print(f"    E[{ei}] ({ea},{eb}): n0={np.round(n0b,2)}  n1={np.round(n1b,2)} {bnd}")

    # --- Plot ---
    fig = plt.figure(figsize=(14, 5))

    # Left: flat mesh, highlight focus edge and its triangles
    ax1 = fig.add_subplot(131, projection="3d")
    adj_tris = sorted(set(flat_mesh.T_v[a_]) & set(flat_mesh.T_v[b_]))
    draw_flat_mesh(ax1, highlight_tris=set(adj_tris),
                   highlight_edges={FOCUS_EDGE})
    # Draw the two face normals
    for ti in adj_tris:
        c = flat_mesh.V[flat_mesh.T[ti]].mean(axis=0)
        n = flat_mesh.face_normals[ti] * 0.4
        ax1.quiver(*c, *n, color="#8e44ad", linewidth=1.5, arrow_length_ratio=0.3)
    set_ax_flat(ax1, f"Flat mesh — E[{FOCUS_EDGE}] highlighted\n(red edge, orange adjacent faces)")

    # Middle: unit sphere — flat mesh slab (degenerate, single point)
    ax2 = fig.add_subplot(132, projection="3d")
    draw_unit_sphere(ax2)
    n0f, n1f = flat_pgm.edge_normal_slabs[FOCUS_EDGE]
    ax2.scatter(*n0f, color="#8e44ad", s=80, zorder=7)
    ax2.text(*n0f*1.15, "n0=n1\n[0,0,1]", fontsize=8, color="#8e44ad", ha="center")
    ax2.set_title(f"Unit sphere — flat slab\n(degenerate: n0=n1, single point)", fontsize=9)
    set_ax_sphere(ax2, f"Unit sphere — flat slab\n(n0=n1: slab degenerates to a point)")

    # Right: unit sphere — box-corner edge slab (genuine arc)
    ax3 = fig.add_subplot(133, projection="3d")
    draw_unit_sphere(ax3)
    # Pick edge 0 of box mesh (between two real faces)
    n0b, n1b = box_pgm.edge_normal_slabs[0]
    # Draw the arc
    alphas = np.linspace(0, 1, 40)
    arc    = np.array([n0b * (1-a) + n1b * a for a in alphas])
    norms  = np.linalg.norm(arc, axis=1, keepdims=True)
    arc    = arc / norms
    ax3.plot(arc[:,0], arc[:,1], arc[:,2], color="#e74c3c", linewidth=2)
    ax3.scatter(*n0b, color="#3498db", s=60, zorder=7)
    ax3.scatter(*n1b, color="#2ecc71", s=60, zorder=7)
    ax3.text(*(n0b*1.2), "n0", fontsize=8, color="#3498db")
    ax3.text(*(n1b*1.2), "n1", fontsize=8, color="#2ecc71")
    set_ax_sphere(ax3, "Unit sphere — box-corner E[0]\n(genuine arc between n0 and n1)")

    plt.suptitle("Step 2 — Edge Normal Slabs", fontsize=11)
    plt.tight_layout()
    plt.show()


def step_3():
    """Vertex normal cones."""
    print("\n══════════════════════════════════════════════════════")
    print(" Step 3 — Vertex Normal Cones")
    print("══════════════════════════════════════════════════════")
    print(f"""
  Each vertex's *normal cone* is the set of directions spanned by the
  face normals of all its incident faces, ordered CCW around the vertex.

  On the unit sphere, the cone is a spherical convex polygon whose
  vertices (generators) are the incident face normals.

  Paper use (Eq. 8): a contact direction d is inside the cone iff
      dot(d, n_i × n_{{i+1}}) ≥ 0   for every consecutive generator pair
  (i.e. d is on the inward side of every bounding great-circle arc).
""")

    cone = flat_pgm.vertex_normal_cones[FOCUS_VERTEX]
    print(f"  Flat mesh, V[{FOCUS_VERTEX}] = {flat_mesh.V[FOCUS_VERTEX]}  "
          f"({len(cone)} generators, type={flat_pgm.vertex_types[FOCUS_VERTEX].name}):")
    for k, n in enumerate(cone):
        print(f"    n[{k}] = {np.round(n, 4)}")
    print(f"  → All generators identical [0,0,1]: cone degenerates to the upper hemisphere.")

    print()
    cone_box = box_pgm.vertex_normal_cones[0]
    print(f"  Box-corner, V[0] = {box_mesh.V[0]}  "
          f"({len(cone_box)} generators, type={box_pgm.vertex_types[0].name}):")
    for k, n in enumerate(cone_box):
        print(f"    n[{k}] = {np.round(n, 4)}")
    print(f"  → Three orthogonal normals span a 1/8-sphere cone (octant corner).")

    # --- Plot ---
    fig = plt.figure(figsize=(14, 5))

    # Left: flat mesh, highlight FOCUS_VERTEX and its incident faces
    ax1 = fig.add_subplot(131, projection="3d")
    inc_tris = set(flat_mesh.T_v[FOCUS_VERTEX])
    inc_edges = set(flat_mesh.E_v[FOCUS_VERTEX])
    draw_flat_mesh(ax1, highlight_tris=inc_tris, highlight_edges=inc_edges,
                   highlight_verts={FOCUS_VERTEX})
    for ti in inc_tris:
        c = flat_mesh.V[flat_mesh.T[ti]].mean(axis=0)
        n = flat_mesh.face_normals[ti] * 0.4
        ax1.quiver(*c, *n, color="#8e44ad", linewidth=1.2, arrow_length_ratio=0.3)
    set_ax_flat(ax1, f"Flat mesh — V[{FOCUS_VERTEX}] (red dot)\n"
                     f"Orange faces = incident faces, their normals shown")

    # Middle: unit sphere — flat cone (degenerate)
    ax2 = fig.add_subplot(132, projection="3d")
    draw_unit_sphere(ax2)
    n_flat = np.array([0., 0., 1.])
    ax2.scatter(*n_flat, color="#8e44ad", s=100, zorder=7)
    ax2.text(*(n_flat*1.2), "[0,0,1]\n(all 6 generators\ncoincide)", fontsize=7,
             color="#8e44ad", ha="center")
    # Draw hemisphere outline to hint at what "upper hemisphere" means
    theta = np.linspace(0, 2*np.pi, 60)
    ax2.plot(np.cos(theta), np.sin(theta), np.zeros(60),
             "--", color="#bbb", linewidth=0.8, label="equator")
    set_ax_sphere(ax2, f"Unit sphere — flat V[{FOCUS_VERTEX}] cone\n"
                       "(all generators identical → upper hemisphere)")

    # Right: unit sphere — box-corner cone (genuine spherical triangle)
    ax3 = fig.add_subplot(133, projection="3d")
    draw_unit_sphere(ax3)
    K = len(cone_box)
    for k, n in enumerate(cone_box):
        ax3.scatter(*n, color="#e74c3c", s=70, zorder=7)
        ax3.text(*(n*1.18), f"n{k}\n{np.round(n,1)}", fontsize=7, color="#c0392b",
                 ha="center")
        # Draw arc to next generator
        n_next  = cone_box[(k+1) % K]
        alphas  = np.linspace(0, 1, 30)
        arc     = np.array([n*(1-a) + n_next*a for a in alphas])
        arc     = arc / np.linalg.norm(arc, axis=1, keepdims=True)
        ax3.plot(arc[:,0], arc[:,1], arc[:,2], color="#e74c3c", linewidth=1.8)
    # Shade the interior of the cone on the sphere
    # (sample points inside the cone)
    pts = []
    for _ in range(600):
        d = np.random.randn(3)
        d /= np.linalg.norm(d)
        if box_pgm.is_in_vertex_normal_cone(d, 0):
            pts.append(d)
    if pts:
        pts = np.array(pts)
        ax3.scatter(pts[:,0], pts[:,1], pts[:,2], color="#e74c3c", s=3, alpha=0.25)
    set_ax_sphere(ax3, "Unit sphere — box-corner V[0] cone\n"
                       "(red boundary arcs, shaded interior)")

    plt.suptitle("Step 3 — Vertex Normal Cones", fontsize=11)
    plt.tight_layout()
    plt.show()


def step_4():
    """Feasibility queries — test directions against the cone and slab."""
    print("\n══════════════════════════════════════════════════════")
    print(" Step 4 — Feasibility Queries")
    print("══════════════════════════════════════════════════════")
    print(f"""
  When a contact detection query finds that the closest point on a
  triangle to a query vertex is at a specific feature (vertex / edge /
  face interior), the Gauss Map tells us whether the contact is *valid*:

    Contact direction d = query_vertex − closest_point

    Feature = face interior:  always valid (one well-defined normal)
    Feature = edge e        :  valid iff is_in_edge_normal_slab(d, e)
    Feature = vertex v      :  valid iff is_in_vertex_normal_cone(d, v)

  Without this check, a naive distance query would double-count contacts
  near edges and vertices (the same contact point would be claimed by
  multiple features).
""")

    # Test several directions against the box-corner vertex cone
    test_dirs = [
        ("outward diagonal  (−1,−1,−1)/√3", np.array([-1.,-1.,-1.])/np.sqrt(3)),
        ("inward  diagonal  (+1,+1,+1)/√3", np.array([ 1., 1., 1.])/np.sqrt(3)),
        ("along −Z                        ", np.array([ 0., 0.,-1.])),
        ("along +X                        ", np.array([ 1., 0., 0.])),
        ("along −X−Y (in-plane diagonal)  ", np.array([-1.,-1., 0.])/np.sqrt(2)),
    ]

    print(f"  Box-corner V[0] normal cone  (generators: {len(box_pgm.vertex_normal_cones[0])} face normals)")
    for label, d in test_dirs:
        inside = box_pgm.is_in_vertex_normal_cone(d, 0)
        mark   = "✓  inside" if inside else "✗  outside"
        print(f"    d = {label}: {mark}")

    print()
    # Test directions against a box-corner edge slab
    ei_box = 0
    n0b, n1b = box_pgm.edge_normal_slabs[ei_box]
    print(f"  Box-corner E[{ei_box}]  slab  n0={np.round(n0b,2)}  n1={np.round(n1b,2)}")
    for label, d in test_dirs:
        inside = box_pgm.is_in_edge_normal_slab(d, ei_box)
        mark   = "✓  inside" if inside else "✗  outside"
        print(f"    d = {label}: {mark}")

    # --- Plot ---
    fig = plt.figure(figsize=(14, 5))

    # Left: unit sphere — box-corner cone with test directions
    ax1 = fig.add_subplot(131, projection="3d")
    draw_unit_sphere(ax1)
    # Shade cone interior
    pts = []
    for _ in range(1000):
        d_ = np.random.randn(3); d_ /= np.linalg.norm(d_)
        if box_pgm.is_in_vertex_normal_cone(d_, 0):
            pts.append(d_)
    if pts:
        pts = np.array(pts)
        ax1.scatter(pts[:,0], pts[:,1], pts[:,2], color="#e74c3c", s=3, alpha=0.2)
    # Draw test directions
    for label, d in test_dirs:
        inside = box_pgm.is_in_vertex_normal_cone(d, 0)
        color  = "#27ae60" if inside else "#e74c3c"
        ax1.quiver(0,0,0, *d*0.9, color=color, linewidth=1.5,
                   arrow_length_ratio=0.2)
    # Legend arrows
    from matplotlib.lines import Line2D
    leg = [
        Line2D([0],[0], color="#27ae60", linewidth=2, label="inside cone"),
        Line2D([0],[0], color="#e74c3c", linewidth=2, label="outside cone"),
    ]
    ax1.legend(handles=leg, fontsize=8, loc="upper left")
    set_ax_sphere(ax1, "Box-corner V[0] cone\n(green = valid contact direction)")

    # Middle: unit sphere — edge slab with test directions
    ax2 = fig.add_subplot(132, projection="3d")
    draw_unit_sphere(ax2)
    # Shade slab interior (directions with dot>=0 for both n0,n1)
    pts_s = []
    for _ in range(1000):
        d_ = np.random.randn(3); d_ /= np.linalg.norm(d_)
        if box_pgm.is_in_edge_normal_slab(d_, ei_box):
            pts_s.append(d_)
    if pts_s:
        pts_s = np.array(pts_s)
        ax2.scatter(pts_s[:,0], pts_s[:,1], pts_s[:,2], color="#3498db", s=3, alpha=0.2)
    for label, d in test_dirs:
        inside = box_pgm.is_in_edge_normal_slab(d, ei_box)
        color  = "#27ae60" if inside else "#e74c3c"
        ax2.quiver(0,0,0, *d*0.9, color=color, linewidth=1.5,
                   arrow_length_ratio=0.2)
    ax2.legend(handles=leg, fontsize=8, loc="upper left")
    set_ax_sphere(ax2, f"Box-corner E[{ei_box}] slab\n(blue = valid slab directions)")

    # Right: 3D scene — box-corner mesh + test directions from V[0]
    ax3 = fig.add_subplot(133, projection="3d")
    draw_box_mesh(ax3)
    ax3.scatter(*box_mesh.V[0], color="#8e44ad", s=100, zorder=7)
    ax3.text(*box_mesh.V[0]+np.array([-0.05,-0.05,0.08]), "V[0]", fontsize=8, color="#8e44ad")
    for label, d in test_dirs:
        inside = box_pgm.is_in_vertex_normal_cone(d, 0)
        color  = "#27ae60" if inside else "#e74c3c"
        tip    = box_mesh.V[0] + d * 0.45
        ax3.quiver(*box_mesh.V[0], *(tip - box_mesh.V[0]),
                   color=color, linewidth=1.2, arrow_length_ratio=0.2)
    ax3.legend(handles=leg, fontsize=8, loc="upper left")
    set_ax_box(ax3, "Box-corner scene\nGreen arrows = valid contact directions from V[0]")

    plt.suptitle("Step 4 — Feasibility Queries (cone and slab membership)", fontsize=11)
    plt.tight_layout()
    plt.show()


# ======================================================================
# Main — run all steps sequentially
# ======================================================================

def main():
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║                  M1 Gauss Map Explorer                       ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  Running all 4 steps in sequence.                            ║")
    print("║  Close each plot window to proceed to the next step.         ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    step_1()   # Vertex classification
    step_2()   # Edge normal slabs
    step_3()   # Vertex normal cones
    step_4()   # Feasibility queries

    print("\n✓ All steps complete.")


if __name__ == "__main__":
    main()
