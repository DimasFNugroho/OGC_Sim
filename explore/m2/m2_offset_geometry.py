"""
M2 Offset Geometry Explorer
============================
Walks through the OGC offset-geometry feasibility checks step by step.

    python3 explore/m2/m2_offset_geometry.py

The process (all steps run in sequence, 1 → 2 → 3 → 4):

  Step 1 — The Offset Surface   : what the three block types look like in 3D
  Step 2 — Vertex Block (Eq. 8) : which query points belong to a vertex block
  Step 3 — Edge Block   (Eq. 9) : which query points belong to an edge block
  Step 4 — VF Contact           : combined feasibility check on real triangles

The core insight
----------------
  When a query vertex is near a triangle surface, the closest point on that
  surface can fall at three types of feature: a face interior, an edge, or a
  vertex.  OGC assigns each query to *exactly one* feature by checking
  whether the contact direction lies in that feature's "normal set":

    Feature = face interior  →  normal set = single face normal  (always OK)
    Feature = edge e         →  normal set = normal slab of e    (Eq. 9)
    Feature = vertex v       →  normal set = normal cone of v    (Eq. 8)

  Without this check, a naive distance query near an edge would fire twice
  (once for each adjacent face) — that double-counting is exactly what the
  Gauss Map feasibility check prevents.

Paper reference: Sec. 3.2–3.5, Eq. 8, 9, 15.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ogc_sim.geometry.mesh      import Mesh
from ogc_sim.geometry.gauss_map import PolyhedralGaussMap
from ogc_sim.geometry.distance  import ClosestFeature
from ogc_sim.contact.offset_geometry import (
    check_vertex_feasible_region,
    check_edge_feasible_region,
    feasible_vf_contact,
    feasible_ee_contact,
)

# ======================================================================
# PARAMETERS  (change these to explore different features)
# ======================================================================

FOCUS_VERTEX = 0   # box corner vertex — 3 incident faces → rich normal cone
FOCUS_EDGE   = 0   # first interior edge in the box mesh

# ======================================================================
# Shared meshes
# ======================================================================

# ---- Box-corner mesh: 3 faces of a unit cube meeting at V[0] = origin ----
#   Face normals all point inward (toward origin) with CCW winding:
#     T[0] = (0,2,1) → normal (0,0,-1)
#     T[1] = (0,3,2) → normal (-1,0,0)
#     T[2] = (0,1,3) → normal (0,-1,0)
V_box = np.array([
    [0., 0., 0.],  # 0 — shared convex corner
    [1., 0., 0.],  # 1
    [0., 1., 0.],  # 2
    [0., 0., 1.],  # 3
])
T_box = np.array([
    [0, 2, 1],   # z=0 face  → outward normal (0,0,-1)
    [0, 3, 2],   # x=0 face  → outward normal (-1,0,0)
    [0, 1, 3],   # y=0 face  → outward normal (0,-1,0)
])
box_mesh = Mesh.from_arrays(V_box, T_box)
box_pgm  = PolyhedralGaussMap(box_mesh)

# ---- Single flat triangle (for a clean face-interior / edge / vertex demo) ----
V_tri = np.array([
    [0., 0., 0.],   # 0
    [2., 0., 0.],   # 1
    [1., 2., 0.],   # 2
])
T_tri = np.array([[0, 1, 2]])
tri_mesh = Mesh.from_arrays(V_tri, T_tri)
tri_pgm  = PolyhedralGaussMap(tri_mesh)

# ======================================================================
# Shared drawing helpers
# ======================================================================

def draw_box(ax, alpha=0.25):
    face_colors = ["#dfe6e9", "#d0ece7", "#d6eaf8"]
    for ti, tri in enumerate(box_mesh.T):
        pts = box_mesh.V[tri]
        ax.add_collection3d(Poly3DCollection(
            [pts], alpha=alpha, facecolor=face_colors[ti], edgecolor="#7f8c8d", linewidth=0.9
        ))
    for a, b in box_mesh.E:
        p0, p1 = box_mesh.V[a], box_mesh.V[b]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color="#aaa", lw=0.8)


def draw_tri(ax, alpha=0.3, color="#d0ece7"):
    pts = tri_mesh.V[tri_mesh.T[0]]
    ax.add_collection3d(Poly3DCollection([pts], alpha=alpha, facecolor=color, edgecolor="#7f8c8d", lw=1.0))
    for a, b in tri_mesh.E:
        p0, p1 = tri_mesh.V[a], tri_mesh.V[b]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color="#888", lw=1.0)
    for vi, v in enumerate(tri_mesh.V):
        ax.scatter(*v, color="#555", s=20, zorder=6)
        ax.text(v[0] - 0.05, v[1] + 0.06, v[2] + 0.05, f"V{vi}", fontsize=7, color="#333")


def draw_unit_sphere(ax, alpha=0.05):
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(30), np.cos(v))
    ax.plot_surface(x, y, z, color="#aaa", alpha=alpha, linewidth=0)


def set_ax(ax, title, xlim=(-0.15, 1.3), ylim=(-0.15, 1.3), zlim=(-0.15, 1.3),
           elev=25, azim=35):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
    ax.set_xlabel("X", fontsize=8); ax.set_ylabel("Y", fontsize=8); ax.set_zlabel("Z", fontsize=8)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=9)


def set_ax_sphere(ax, title):
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_zlim(-1.4, 1.4)
    ax.set_xlabel("X", fontsize=8); ax.set_ylabel("Y", fontsize=8); ax.set_zlabel("Z", fontsize=8)
    ax.view_init(elev=25, azim=-50)
    ax.set_title(title, fontsize=9)


LEGEND_IN  = [Line2D([0],[0], color="#27ae60", lw=2, label="feasible (inside)")]
LEGEND_OUT = [Line2D([0],[0], color="#e74c3c", lw=2, label="infeasible (outside)")]

# ======================================================================
# Steps
# ======================================================================

def step_1():
    """The Offset Surface — three block types."""
    print("\n══════════════════════════════════════════════════════")
    print(" Step 1 — The Offset Surface and Its Three Block Types")
    print("══════════════════════════════════════════════════════")
    print("""
  The offset surface of a mesh at distance r is the set of all points
  within distance r from the mesh surface.  OGC tiles this offset into
  three mutually exclusive block types:

    Face-interior block  U_t :  points closest to the interior of face t
                                (a half-slab above the face, thickness r)

    Edge block           U_e :  points closest to the *interior* of edge e
                                (a wedge/cylinder around the edge ridge)

    Vertex block         U_v :  points closest to vertex v
                                (a cone-shaped region around the corner)

  Why tile it?
  ------------
  A naive offset check "is dist(x, mesh) < r?" would report the same
  contact for a point near an edge from BOTH adjacent faces.  The tiling
  assigns every contact point to exactly one owner feature, eliminating
  all double-counting.

  The Gauss Map tells us which block a direction belongs to:
    direction d = (query - closest_pt)
    → d ∈ normal_cone(v)   ⟹  contact belongs to vertex block  U_v
    → d ∈ normal_slab(e)   ⟹  contact belongs to edge block    U_e
    → d is just face normal ⟹  contact belongs to face block    U_t
""")

    fig = plt.figure(figsize=(15, 5))

    # ---- Left: box mesh with one query point per block type ----
    ax1 = fig.add_subplot(131, projection="3d")
    draw_box(ax1, alpha=0.3)

    # Query in face-interior block: directly above face T[0] (z=0 face)
    q_face = np.array([0.3, 0.3, -0.3])   # below z=0 face (outward = -z)
    # Query in edge block: beside edge E[0] (the shared edge 0-1)
    e0a, e0b = box_mesh.V[int(box_mesh.E[0][0])], box_mesh.V[int(box_mesh.E[0][1])]
    q_edge = (e0a + e0b) * 0.5 + np.array([0., -0.3, -0.2])
    # Query in vertex block: outward from corner V[0]
    q_vert = np.array([-0.3, -0.2, -0.25])

    for q, label, color in [
        (q_face, "face\nblock", "#3498db"),
        (q_edge, "edge\nblock", "#e67e22"),
        (q_vert, "vertex\nblock", "#9b59b6"),
    ]:
        ax1.scatter(*q, color=color, s=60, zorder=8)
        ax1.text(q[0]+0.04, q[1], q[2]+0.04, label, fontsize=7, color=color, ha="left")

    set_ax(ax1, "Three block types\n(one query point each)",
           xlim=(-0.5, 1.2), ylim=(-0.5, 1.2), zlim=(-0.5, 1.2))

    # ---- Middle: show offset "shell" around a single edge ----
    ax2 = fig.add_subplot(132, projection="3d")
    draw_box(ax2, alpha=0.15)

    # Shade the edge block of E[FOCUS_EDGE] by sampling
    ei = FOCUS_EDGE
    ea, eb = box_mesh.V[int(box_mesh.E[ei][0])], box_mesh.V[int(box_mesh.E[ei][1])]
    pts_in = []
    rng = np.random.default_rng(42)
    for _ in range(3000):
        q = rng.uniform([-0.4, -0.4, -0.4], [1.0, 1.0, 1.0])
        if check_edge_feasible_region(q, ei, box_mesh, box_pgm):
            # Also check distance is within r=0.35
            ab = eb - ea
            t_ = float(np.dot(q - ea, ab) / np.dot(ab, ab))
            t_ = float(np.clip(t_, 0., 1.))
            cp_ = ea + t_ * ab
            if np.linalg.norm(q - cp_) < 0.35:
                pts_in.append(q)
    if pts_in:
        pts_in = np.array(pts_in)
        ax2.scatter(pts_in[:,0], pts_in[:,1], pts_in[:,2], color="#e67e22", s=4, alpha=0.25)

    # Draw the focused edge in red
    ax2.plot([ea[0], eb[0]], [ea[1], eb[1]], [ea[2], eb[2]], color="#e74c3c", lw=2.5)
    ax2.text(*(ea+eb)*0.5 + np.array([0.05, 0, 0.05]), f"E[{ei}]", fontsize=8, color="#c0392b")
    set_ax(ax2, f"Edge block of E[{FOCUS_EDGE}]\n(orange = points in the edge's offset region)",
           xlim=(-0.5, 1.2), ylim=(-0.5, 1.2), zlim=(-0.5, 1.2))

    # ---- Right: show vertex block of V[FOCUS_VERTEX] ----
    ax3 = fig.add_subplot(133, projection="3d")
    draw_box(ax3, alpha=0.15)

    vi = FOCUS_VERTEX
    pts_vb = []
    for _ in range(3000):
        q = rng.uniform([-0.5, -0.5, -0.5], [0.8, 0.8, 0.8])
        if check_vertex_feasible_region(q, vi, box_mesh, box_pgm):
            if np.linalg.norm(q - box_mesh.V[vi]) < 0.45:
                pts_vb.append(q)
    if pts_vb:
        pts_vb = np.array(pts_vb)
        ax3.scatter(pts_vb[:,0], pts_vb[:,1], pts_vb[:,2], color="#9b59b6", s=4, alpha=0.25)

    ax3.scatter(*box_mesh.V[vi], color="#8e44ad", s=80, zorder=8)
    ax3.text(*box_mesh.V[vi] + np.array([-0.05, -0.05, 0.06]), f"V[{vi}]", fontsize=8, color="#8e44ad")
    set_ax(ax3, f"Vertex block of V[{FOCUS_VERTEX}]\n(purple = points in the vertex's offset region)",
           xlim=(-0.5, 1.0), ylim=(-0.5, 1.0), zlim=(-0.5, 1.0))

    plt.suptitle("Step 1 — The Offset Surface: Face / Edge / Vertex Blocks", fontsize=11)
    plt.tight_layout()
    plt.show()


def step_2():
    """Vertex block feasibility — Eq. 8."""
    print("\n══════════════════════════════════════════════════════")
    print(" Step 2 — Vertex Block Feasibility (Eq. 8)")
    print("══════════════════════════════════════════════════════")
    print(f"""
  A query point x belongs to the vertex block of vertex v iff:

        direction = x − V[v]
        direction ∈ normal_cone(v)           (Eq. 8)

  The normal cone is the set of directions that are "outward" from v,
  consistent with all incident face normals.  Geometrically it is the
  solid angle on the unit sphere spanned by all incident face normals.

  For V[{FOCUS_VERTEX}] of the box-corner mesh: 3 incident faces with normals
    n0 = (0,0,-1),  n1 = (-1,0,0),  n2 = (0,-1,0)
  → The cone is the negative-x-y-z octant.  Only directions pointing
    away from the box corner (all three components negative) are inside.
""")

    # Test a collection of query directions
    test_pts = [
        ("(−0.3, −0.2, −0.25)  outward corner", np.array([-0.3, -0.2, -0.25])),
        ("(−0.5, −0.5,  0.0 )  edge of cone  ", np.array([-0.5, -0.5,  0.0])),
        ("(+0.3, −0.2, −0.25)  wrong side +x ", np.array([ 0.3, -0.2, -0.25])),
        ("(−0.2, +0.3, −0.1 )  wrong side +y ", np.array([-0.2,  0.3, -0.1])),
        ("(−0.1, −0.1, −0.4 )  deep inside   ", np.array([-0.1, -0.1, -0.4])),
        ("( 0.0,  0.0, +0.5 )  inward +z     ", np.array([ 0.0,  0.0,  0.5])),
    ]

    print(f"  Query points tested against V[{FOCUS_VERTEX}] = {box_mesh.V[FOCUS_VERTEX]} :")
    print()
    for label, q in test_pts:
        feasible = check_vertex_feasible_region(q, FOCUS_VERTEX, box_mesh, box_pgm)
        direction = q - box_mesh.V[FOCUS_VERTEX]
        mark = "✓  feasible" if feasible else "✗  infeasible"
        print(f"    x = {label}  →  d = {np.round(direction, 2)}  →  {mark}")

    # --- Plot ---
    fig = plt.figure(figsize=(14, 5))

    # Left: 3D scene with query points
    ax1 = fig.add_subplot(131, projection="3d")
    draw_box(ax1, alpha=0.2)
    v0 = box_mesh.V[FOCUS_VERTEX]
    ax1.scatter(*v0, color="#8e44ad", s=100, zorder=8)
    ax1.text(*v0 + np.array([-0.05, -0.05, 0.07]), f"V[{FOCUS_VERTEX}]", fontsize=8, color="#8e44ad")

    for label, q in test_pts:
        feasible = check_vertex_feasible_region(q, FOCUS_VERTEX, box_mesh, box_pgm)
        color = "#27ae60" if feasible else "#e74c3c"
        ax1.scatter(*q, color=color, s=55, zorder=7)
        # Arrow from vertex to query point
        ax1.quiver(*v0, *(q - v0) * 0.9, color=color, lw=1.2, arrow_length_ratio=0.2)

    ax1.legend(handles=LEGEND_IN + LEGEND_OUT, fontsize=8, loc="upper right")
    set_ax(ax1, f"Eq. 8 — Vertex block of V[{FOCUS_VERTEX}]\n(green = inside cone = feasible contact)",
           xlim=(-0.6, 1.1), ylim=(-0.6, 1.1), zlim=(-0.6, 1.1))

    # Middle: unit sphere — normal cone with test directions
    ax2 = fig.add_subplot(132, projection="3d")
    draw_unit_sphere(ax2)
    # Shade the cone interior
    pts_cone = []
    rng = np.random.default_rng(7)
    for _ in range(2000):
        d = rng.standard_normal(3)
        d /= np.linalg.norm(d)
        if box_pgm.is_in_vertex_normal_cone(d, FOCUS_VERTEX):
            pts_cone.append(d)
    if pts_cone:
        pts_cone = np.array(pts_cone)
        ax2.scatter(pts_cone[:,0], pts_cone[:,1], pts_cone[:,2], color="#9b59b6", s=4, alpha=0.2)

    # Draw generator normals
    cone = box_pgm.vertex_normal_cones[FOCUS_VERTEX]
    for k, n in enumerate(cone):
        ax2.scatter(*n, color="#e74c3c", s=60, zorder=7)
        ax2.text(*n * 1.2, f"n{k}", fontsize=7, color="#c0392b")

    # Draw test directions
    for label, q in test_pts:
        d = q - box_mesh.V[FOCUS_VERTEX]
        norm = np.linalg.norm(d)
        if norm > 1e-10:
            d_unit = d / norm
            feasible = box_pgm.is_in_vertex_normal_cone(d_unit, FOCUS_VERTEX)
            color = "#27ae60" if feasible else "#e74c3c"
            ax2.quiver(0, 0, 0, *d_unit * 0.85, color=color, lw=1.2, arrow_length_ratio=0.2)

    ax2.legend(handles=LEGEND_IN + LEGEND_OUT, fontsize=8, loc="upper left")
    set_ax_sphere(ax2, f"Unit sphere — V[{FOCUS_VERTEX}] cone\n"
                       "(purple = cone interior, arrows = test directions)")

    # Right: cross-section diagram explaining Eq. 8
    ax3 = fig.add_subplot(133)
    ax3.set_aspect("equal")
    ax3.set_xlim(-1.2, 1.2); ax3.set_ylim(-1.2, 1.2)
    ax3.axis("off")

    # Draw a 2D "wedge" representing the cone cross-section
    theta0, theta1 = np.radians(180 + 30), np.radians(270 + 30)  # two face normals
    thetas = np.linspace(theta0, theta1, 60)
    cone_x = [0] + list(np.cos(thetas) * 0.9) + [0]
    cone_y = [0] + list(np.sin(thetas) * 0.9) + [0]
    ax3.fill(cone_x, cone_y, color="#9b59b6", alpha=0.25, label="normal cone (2D slice)")
    ax3.plot(cone_x, cone_y, color="#8e44ad", lw=1.5)

    # Draw two face normals as arrows
    n0_2d = np.array([np.cos(theta0), np.sin(theta0)])
    n1_2d = np.array([np.cos(theta1), np.sin(theta1)])
    ax3.annotate("", xy=n0_2d * 0.85, xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5))
    ax3.annotate("", xy=n1_2d * 0.85, xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5))
    ax3.text(*n0_2d * 1.0, "n₀", fontsize=9, color="#c0392b")
    ax3.text(*n1_2d * 1.0, "n₁", fontsize=9, color="#c0392b")

    # Draw a "feasible" direction inside the cone
    d_in  = np.array([np.cos(np.radians(230)), np.sin(np.radians(230))])
    d_out = np.array([np.cos(np.radians(30)),  np.sin(np.radians(30))])
    ax3.annotate("", xy=d_in * 0.75, xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color="#27ae60", lw=2))
    ax3.annotate("", xy=d_out * 0.75, xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=2))
    ax3.text(*d_in  * 0.8, "inside\n(feasible)", fontsize=8, color="#27ae60", ha="center")
    ax3.text(*d_out * 0.8, "outside\n(infeasible)", fontsize=8, color="#e74c3c", ha="center")

    ax3.scatter(0, 0, color="#8e44ad", s=80, zorder=6)
    ax3.text(0.05, 0.05, "V", fontsize=10, color="#8e44ad")
    ax3.set_title("Eq. 8 (2D cross-section)\n"
                  "direction ∈ cone → feasible\n"
                  "direction ∉ cone → belongs to another block", fontsize=9)

    plt.suptitle("Step 2 — Vertex Block Feasibility Check (Eq. 8)", fontsize=11)
    plt.tight_layout()
    plt.show()


def step_3():
    """Edge block feasibility — Eq. 9."""
    print("\n══════════════════════════════════════════════════════")
    print(" Step 3 — Edge Block Feasibility (Eq. 9)")
    print("══════════════════════════════════════════════════════")
    print(f"""
  A query point x belongs to the edge block of edge e iff:

        t = parameter of (projection of x onto e) ∈ (0, 1)
        direction = x − (a + t*(b-a))
        direction ∈ normal_slab(e)            (Eq. 9)

  The normal slab is the "wedge" between the two adjacent face normals:
        slab = {{ d : dot(d, n0) ≥ 0  AND  dot(d, n1) ≥ 0 }}

  On a convex edge (ridge), n0 and n1 diverge outward → the slab is the
  region between them, pointing "away" from the edge.
  On a concave edge (valley), n0 and n1 converge → the slab is narrower.

  The t ∈ (0,1) condition ensures that if x is closest to an *endpoint*
  of the edge, that contact is handled by the endpoint's vertex block.
""")

    # Pick a representative edge of the box mesh (a boundary/interior edge)
    ei = FOCUS_EDGE
    ea_idx, eb_idx = int(box_mesh.E[ei][0]), int(box_mesh.E[ei][1])
    ea, eb = box_mesh.V[ea_idx], box_mesh.V[eb_idx]
    n0, n1 = box_pgm.edge_normal_slabs[ei]
    print(f"  Box-mesh E[{ei}] = ({ea_idx}, {eb_idx})")
    print(f"    a = {ea},  b = {eb}")
    print(f"    n0 = {np.round(n0, 3)},  n1 = {np.round(n1, 3)}")

    # Query points: above the edge interior, beside endpoints, wrong side
    mid = (ea + eb) * 0.5
    test_pts = [
        ("mid + outward blend    ", mid + (n0 + n1) * 0.2),
        ("mid − outward blend    ", mid - (n0 + n1) * 0.2),
        ("near endpoint a        ", ea + (n0 + n1) * 0.15),
        ("t < 0 (before a)       ", ea - (eb - ea) * 0.3 + (n0 + n1) * 0.2),
        ("t > 1 (beyond b)       ", eb + (eb - ea) * 0.3 + (n0 + n1) * 0.2),
        ("perpendicular outward  ", mid + np.cross(eb - ea, n0) * 0.3 + n0 * 0.2),
    ]

    print()
    print(f"  Query points tested against E[{ei}]:")
    for label, q in test_pts:
        feasible = check_edge_feasible_region(q, ei, box_mesh, box_pgm)
        mark = "✓  feasible" if feasible else "✗  infeasible"
        print(f"    x = {label}  →  {mark}")

    # --- Plot ---
    fig = plt.figure(figsize=(14, 5))

    # Left: 3D scene
    ax1 = fig.add_subplot(131, projection="3d")
    draw_box(ax1, alpha=0.2)
    ax1.plot([ea[0], eb[0]], [ea[1], eb[1]], [ea[2], eb[2]], color="#e74c3c", lw=3, zorder=7)
    ax1.text(*mid + np.array([0.04, 0.0, 0.06]), f"E[{ei}]", fontsize=8, color="#c0392b")
    # Face normals from edge midpoint
    for ni, n in enumerate([n0, n1]):
        ax1.quiver(*mid, *(n * 0.3), color="#8e44ad", lw=1.5, arrow_length_ratio=0.3)
        ax1.text(*(mid + n * 0.35), f"n{ni}", fontsize=7, color="#8e44ad")

    for label, q in test_pts:
        feasible = check_edge_feasible_region(q, ei, box_mesh, box_pgm)
        color = "#27ae60" if feasible else "#e74c3c"
        ax1.scatter(*q, color=color, s=55, zorder=7)

    ax1.legend(handles=LEGEND_IN + LEGEND_OUT, fontsize=8, loc="upper right")
    set_ax(ax1, f"Eq. 9 — Edge block of E[{ei}]\n(green = inside slab = feasible)",
           xlim=(-0.6, 1.2), ylim=(-0.6, 1.2), zlim=(-0.6, 1.2))

    # Middle: unit sphere — slab region
    ax2 = fig.add_subplot(132, projection="3d")
    draw_unit_sphere(ax2)
    # Shade slab interior (directions in both half-spaces)
    pts_slab = []
    rng = np.random.default_rng(13)
    for _ in range(2000):
        d = rng.standard_normal(3); d /= np.linalg.norm(d)
        if box_pgm.is_in_edge_normal_slab(d, ei):
            pts_slab.append(d)
    if pts_slab:
        pts_slab = np.array(pts_slab)
        ax2.scatter(pts_slab[:,0], pts_slab[:,1], pts_slab[:,2], color="#e67e22", s=4, alpha=0.2)

    # Draw bounding normals
    ax2.scatter(*n0, color="#3498db", s=60, zorder=7)
    ax2.scatter(*n1, color="#2ecc71", s=60, zorder=7)
    ax2.text(*n0 * 1.2, "n0", fontsize=8, color="#2980b9")
    ax2.text(*n1 * 1.2, "n1", fontsize=8, color="#27ae60")

    # Draw arc between n0 and n1
    alphas = np.linspace(0, 1, 40)
    arc = np.array([n0 * (1 - a) + n1 * a for a in alphas])
    arc /= np.linalg.norm(arc, axis=1, keepdims=True)
    ax2.plot(arc[:,0], arc[:,1], arc[:,2], color="#e74c3c", lw=2)

    # Draw test direction arrows
    for label, q in test_pts:
        ab_ = eb - ea
        t_ = float(np.clip(np.dot(q - ea, ab_) / np.dot(ab_, ab_), 0, 1))
        cp_ = ea + t_ * ab_
        d = q - cp_
        norm = np.linalg.norm(d)
        if norm > 1e-10:
            d_unit = d / norm
            feasible = box_pgm.is_in_edge_normal_slab(d_unit, ei)
            color = "#27ae60" if feasible else "#e74c3c"
            ax2.quiver(0, 0, 0, *d_unit * 0.85, color=color, lw=1.2, arrow_length_ratio=0.2)

    ax2.legend(handles=LEGEND_IN + LEGEND_OUT, fontsize=8, loc="upper left")
    set_ax_sphere(ax2, f"Unit sphere — E[{ei}] slab\n(orange = slab, arc = boundary)")

    # Right: cross-section diagram
    ax3 = fig.add_subplot(133)
    ax3.set_aspect("equal")
    ax3.set_xlim(-1.3, 1.3); ax3.set_ylim(-1.3, 1.3)
    ax3.axis("off")

    # Use angle for n0 and n1 in 2D for clarity
    ang0, ang1 = np.radians(120), np.radians(60)
    n0_2d = np.array([np.cos(ang0), np.sin(ang0)])
    n1_2d = np.array([np.cos(ang1), np.sin(ang1)])

    # Slab = region between n0 and n1 half-planes
    thetas_slab = np.linspace(ang1, ang0, 60)
    sx = [0] + list(np.cos(thetas_slab)) + [0]
    sy = [0] + list(np.sin(thetas_slab)) + [0]
    ax3.fill(sx, sy, color="#e67e22", alpha=0.25, label="normal slab")
    ax3.plot(sx, sy, color="#e67e22", lw=1.5)

    for n_2d, label in [(n0_2d, "n₀"), (n1_2d, "n₁")]:
        ax3.annotate("", xy=n_2d * 0.85, xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5))
        ax3.text(*n_2d * 1.05, label, fontsize=9, color="#c0392b")

    # Valid direction (inside slab)
    d_in_2d  = np.array([0.0, 1.0])
    d_out_2d = np.array([0.0, -1.0])
    ax3.annotate("", xy=d_in_2d * 0.7, xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color="#27ae60", lw=2))
    ax3.annotate("", xy=d_out_2d * 0.7, xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=2))
    ax3.text(*d_in_2d  * 0.75, "inside\n(feasible)", fontsize=8, color="#27ae60", ha="center")
    ax3.text(*d_out_2d * 0.78, "outside\n(infeasible)", fontsize=8, color="#e74c3c", ha="center")

    # Edge cross-section dot
    ax3.scatter(0, 0, color="#e74c3c", s=80, zorder=6)
    ax3.text(0.07, 0.07, "e", fontsize=11, color="#c0392b")
    ax3.text(0.0, -0.95,
             "dot(d,n₀) ≥ 0  AND  dot(d,n₁) ≥ 0",
             fontsize=8, ha="center", color="#555")
    ax3.set_title("Eq. 9 (2D cross-section)\n"
                  "direction ∈ slab → feasible\n"
                  "direction ∉ slab → belongs to adjacent block", fontsize=9)

    plt.suptitle("Step 3 — Edge Block Feasibility Check (Eq. 9)", fontsize=11)
    plt.tight_layout()
    plt.show()


def step_4():
    """Combined vertex-facet feasibility check."""
    print("\n══════════════════════════════════════════════════════")
    print(" Step 4 — Full VF Contact Feasibility Check")
    print("══════════════════════════════════════════════════════")
    print("""
  feasible_vf_contact(query_vertex, tri_idx, mesh, pgm) runs the full pipeline:

    1. point_triangle_distance(query, a, b, c)
         → (dist, closest_point, feature_type, local_feat_idx)

    2. Map local_feat_idx → global_feat_idx in the mesh:
         VERTEX i      →  mesh.T[tri_idx][i]     (global vertex index)
         EDGE   i      →  mesh.E_t[tri_idx][i]   (global edge index)
         FACE_INTERIOR →  tri_idx                (trivially feasible)

    3. Apply the appropriate Gauss Map check:
         FACE_INTERIOR →  always True
         EDGE          →  Eq. 9 (slab test)
         VERTEX        →  Eq. 8 (cone test)

  The key insight: for a query near a triangle edge, the closest point
  is "on the edge."  But which face does that contact belong to?
  The Gauss Map check resolves this: the contact belongs to the face
  whose edge block contains the query direction.  The other adjacent face
  will simply return "infeasible" for the same query.

  This is demonstrated below using a single triangle mesh, with query
  points scattered near each sub-feature.
""")

    # Use the single-triangle mesh for cleaner illustrations
    tri_idx = 0
    tri_verts = tri_mesh.T[tri_idx]
    a, b, c = tri_mesh.V[tri_verts[0]], tri_mesh.V[tri_verts[1]], tri_mesh.V[tri_verts[2]]

    # Sample query points above the triangle
    rng = np.random.default_rng(99)
    n_queries = 200
    feature_colors = {
        ClosestFeature.FACE_INTERIOR: "#3498db",
        ClosestFeature.EDGE:          "#e67e22",
        ClosestFeature.VERTEX:        "#9b59b6",
    }

    # Place queries at various heights above the triangle
    queries = []
    for _ in range(n_queries):
        u = rng.random()
        v = rng.random()
        if u + v > 1:
            u, v = 1 - u, 1 - v
        # Barycentric point on triangle
        pt_on_tri = a + u * (b - a) + v * (c - a)
        # Offset upward + small lateral displacement
        lateral = rng.uniform(-0.5, 0.5, 3)
        lateral[2] = 0
        height = rng.uniform(0.05, 0.6)
        q = pt_on_tri + lateral + np.array([0., 0., height])
        queries.append(q)

    # Also add specific queries near each sub-feature
    edge_mid = (a + b) / 2
    vert_a = a.copy()
    face_center = (a + b + c) / 3

    specific = [
        ("above face center   ", face_center + np.array([0., 0., 0.3])),
        ("above edge midpoint ", edge_mid    + np.array([0., 0., 0.3])),
        ("above vertex a      ", vert_a      + np.array([0., 0., 0.3])),
        ("beside edge ab      ", edge_mid    + np.array([0.3, -0.3, 0.1])),
        ("beside vertex a     ", vert_a      + np.array([-0.3, -0.2, 0.1])),
    ]

    print("  Specific query results on single-triangle mesh:")
    for label, q in specific:
        res = feasible_vf_contact(q, tri_idx, tri_mesh, tri_pgm)
        feat_name = res.feature.name
        mark = "✓ feasible" if res.feasible else "✗ infeasible"
        print(f"    x = {label}: dist={res.distance:.3f}  "
              f"feature={feat_name:<14}  gfeat={res.global_feature_idx}  {mark}")

    # --- Plot ---
    fig = plt.figure(figsize=(14, 5))

    # Left: 3D scene — queries coloured by closest feature type
    ax1 = fig.add_subplot(131, projection="3d")
    draw_tri(ax1)
    # Face normal arrow
    fn = tri_mesh.face_normals[0] * 0.5
    fc = face_center
    ax1.quiver(*fc, *fn, color="#888", lw=1.2, arrow_length_ratio=0.25)
    ax1.text(*(fc + fn * 1.1), "n", fontsize=8, color="#555")

    for q in queries:
        res = feasible_vf_contact(q, tri_idx, tri_mesh, tri_pgm)
        color = feature_colors[res.feature]
        ax1.scatter(*q, color=color, s=12, alpha=0.55, zorder=6)

    legend = [Patch(facecolor=feature_colors[f], label=f.name)
              for f in ClosestFeature]
    ax1.legend(handles=legend, fontsize=8, loc="upper right")
    set_ax(ax1, "Query points coloured by closest feature\n(which block each point falls into)",
           xlim=(-0.6, 2.6), ylim=(-0.6, 2.6), zlim=(-0.1, 0.7), elev=30, azim=-60)

    # Middle: 3D scene — queries coloured by feasibility
    ax2 = fig.add_subplot(132, projection="3d")
    draw_tri(ax2)
    for q in queries:
        res = feasible_vf_contact(q, tri_idx, tri_mesh, tri_pgm)
        color = "#27ae60" if res.feasible else "#e74c3c"
        ax2.scatter(*q, color=color, s=12, alpha=0.55, zorder=6)

    ax2.legend(handles=LEGEND_IN + LEGEND_OUT, fontsize=8, loc="upper right")
    set_ax(ax2, "Same queries coloured by feasibility\n(all 1-triangle mesh → all feasible above face)",
           xlim=(-0.6, 2.6), ylim=(-0.6, 2.6), zlim=(-0.1, 0.7), elev=30, azim=-60)

    # Right: box mesh — VF contact from a query vertex toward each face
    ax3 = fig.add_subplot(133, projection="3d")
    draw_box(ax3, alpha=0.25)

    # Query from outside the box: one query per face
    query_pts_box = [
        np.array([ 0.4,  0.3, -0.3]),  # in front of z=0 face  → FACE_INTERIOR
        np.array([-0.3,  0.4,  0.4]),  # in front of x=0 face  → FACE_INTERIOR
        np.array([ 0.5, -0.25, 0.5]),  # in front of y=0 face  → FACE_INTERIOR
        np.array([-0.2, -0.25, -0.2]), # near the corner        → VERTEX (V[0])
        np.array([ 0.5, -0.1, -0.1]),  # near edge              → EDGE
    ]

    for ti in range(box_mesh.num_triangles):
        for q in query_pts_box:
            res = feasible_vf_contact(q, ti, box_mesh, box_pgm)
            if res.feasible:
                color = "#27ae60"
                ax3.scatter(*q, color=color, s=50, zorder=8)
                ax3.plot([q[0], res.contact_point[0]],
                         [q[1], res.contact_point[1]],
                         [q[2], res.contact_point[2]],
                         color=color, lw=1.0, alpha=0.7)
            else:
                ax3.scatter(*q, color="#e74c3c", s=30, alpha=0.3, zorder=7)

    ax3.legend(handles=LEGEND_IN + LEGEND_OUT, fontsize=8, loc="upper right")
    set_ax(ax3, "Box mesh VF contacts\nGreen = feasible (line = contact pair)",
           xlim=(-0.5, 1.1), ylim=(-0.5, 1.1), zlim=(-0.5, 1.1))

    plt.suptitle("Step 4 — Full VF Feasibility Check (Eq. 8 + 9 combined)", fontsize=11)
    plt.tight_layout()
    plt.show()


# ======================================================================
# Main
# ======================================================================

def main():
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║               M2 Offset Geometry Explorer                    ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  Running all 4 steps in sequence.                            ║")
    print("║  Close each plot window to proceed to the next step.         ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    step_1()   # Offset surface block types
    step_2()   # Vertex block feasibility (Eq. 8)
    step_3()   # Edge block feasibility   (Eq. 9)
    step_4()   # Full VF contact check

    print("\n✓ All steps complete.")


if __name__ == "__main__":
    main()
