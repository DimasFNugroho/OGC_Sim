"""
M1 Mesh Explorer
================
Visualise the core data structures that geometry/mesh.py must build.

    python3 explore/m1/m1_mesh.py

Cases
-----
  1 — Mesh structure     : vertices (V), edges (E), triangles (T) labelled
  2 — Face normals       : one arrow per triangle showing its outward normal
  3 — Vertex adjacency   : pick a vertex — highlight its T_v and E_v
  4 — Edge extraction    : step through how unique edges are built from T

The mesh used here is a simple 3×3 grid of quads (2×2 = 4 quads,
each split into 2 triangles → 8 triangles, 9 vertices).
This is small enough to read every index, complex enough to show
non-trivial adjacency.

NOTE: this script does NOT import geometry/mesh.py yet — it builds the
data structures manually so you can see exactly what mesh.py must produce.
Once you implement mesh.py, Case 5 (added later) will cross-check it.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ======================================================================
# PARAMETERS
# ======================================================================

# <<< CHANGE ME (Case 3) — which vertex to highlight (0–8)
FOCUS_VERTEX = 4   # centre vertex — has the most neighbours

# ======================================================================
# Hard-coded mesh: 3×3 grid of vertices, 8 triangles
#
#   6 - 7 - 8
#   |\ | \ |
#   | \|  \|
#   3 - 4 - 5
#   |\ | \ |
#   | \|  \|
#   0 - 1 - 2
#
# Each quad is split along its diagonal (\) into 2 triangles.
# ======================================================================

V = np.array([
    [0.0, 0.0, 0.0],  # 0
    [1.0, 0.0, 0.0],  # 1
    [2.0, 0.0, 0.0],  # 2
    [0.0, 1.0, 0.0],  # 3
    [1.0, 1.0, 0.0],  # 4  ← centre
    [2.0, 1.0, 0.0],  # 5
    [0.0, 2.0, 0.0],  # 6
    [1.0, 2.0, 0.0],  # 7
    [2.0, 2.0, 0.0],  # 8
], dtype=float)

# CCW winding when viewed from +Z
T = np.array([
    [0, 1, 4],  # 0  bottom-left quad, lower triangle
    [0, 4, 3],  # 1  bottom-left quad, upper triangle
    [1, 2, 5],  # 2  bottom-right quad, lower triangle
    [1, 5, 4],  # 3  bottom-right quad, upper triangle
    [3, 4, 7],  # 4  top-left quad, lower triangle
    [3, 7, 6],  # 5  top-left quad, upper triangle
    [4, 5, 8],  # 6  top-right quad, lower triangle
    [4, 8, 7],  # 7  top-right quad, upper triangle
], dtype=int)

# -----------------------------------------------------------------------
# Build the structures that mesh.py must produce
# (this is the reference answer — implement these in mesh.py)
# -----------------------------------------------------------------------

def build_edges(V, T):
    """
    Extract unique undirected edges from triangle soup.
    Each edge stored with smaller index first.
    Returns np.ndarray of shape (M, 2).
    """
    edge_set = set()
    for tri in T:
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i + 1) % 3])
            edge_set.add((min(a, b), max(a, b)))
    return np.array(sorted(edge_set), dtype=int)

def compute_face_normals(V, T):
    """Unit outward normal for each triangle (CCW → +Z for flat mesh)."""
    v0 = V[T[:, 0]]
    v1 = V[T[:, 1]]
    v2 = V[T[:, 2]]
    ab = v1 - v0
    ac = v2 - v0
    n  = np.cross(ab, ac)
    lengths = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.where(lengths > 0, lengths, 1.0)

def build_T_v(V, T):
    """T_v[i] = list of triangle indices that contain vertex i."""
    T_v = [[] for _ in range(len(V))]
    for ti, tri in enumerate(T):
        for vi in tri:
            T_v[vi].append(ti)
    return T_v

def build_E_v(V, E):
    """E_v[i] = list of edge indices that contain vertex i."""
    E_v = [[] for _ in range(len(V))]
    for ei, (a, b) in enumerate(E):
        E_v[a].append(ei)
        E_v[b].append(ei)
    return E_v

def build_E_t(T, E):
    """E_t[f] = list of edge indices that bound triangle f."""
    # Build a lookup: frozenset({a,b}) -> edge index
    edge_lookup = {frozenset(e): ei for ei, e in enumerate(E)}
    E_t = []
    for tri in T:
        tri_edges = []
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i + 1) % 3])
            tri_edges.append(edge_lookup[frozenset({a, b})])
        E_t.append(tri_edges)
    return E_t

# Pre-build everything
E          = build_edges(V, T)
face_norms = compute_face_normals(V, T)
T_v        = build_T_v(V, T)
E_v        = build_E_v(V, E)
E_t        = build_E_t(T, E)

# ======================================================================
# Shared helpers
# ======================================================================

TRI_COLORS = plt.cm.Set3(np.linspace(0, 1, len(T)))

def draw_mesh_base(ax, highlight_tris=None, highlight_edges=None,
                   highlight_verts=None, alpha=0.15):
    """Draw the full mesh with optional highlights."""
    for ti, tri in enumerate(T):
        pts = V[tri]
        color = TRI_COLORS[ti] if highlight_tris is None else (
            "#f39c12" if ti in highlight_tris else "#dfe6e9"
        )
        a = 0.35 if (highlight_tris is None or ti in highlight_tris) else 0.08
        poly = Poly3DCollection([pts], alpha=a,
                                 facecolor=color, edgecolor="#7f8c8d", linewidth=0.8)
        ax.add_collection3d(poly)

    # All edges (thin gray)
    for ei, (a, b) in enumerate(E):
        if highlight_edges and ei in highlight_edges:
            continue   # drawn separately below
        p0, p1 = V[a], V[b]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                color="#95a5a6", linewidth=0.8)

    # Highlighted edges
    if highlight_edges:
        for ei in highlight_edges:
            a, b = E[ei]
            p0, p1 = V[a], V[b]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                    color="#e74c3c", linewidth=3.0)

    # Vertex labels
    for vi, v in enumerate(V):
        color = "#e74c3c" if (highlight_verts and vi in highlight_verts) else "#2c3e50"
        size  = 12 if (highlight_verts and vi in highlight_verts) else 8
        ax.scatter(*v, color=color, s=size*4, zorder=6)
        ax.text(v[0] - 0.08, v[1] + 0.08, v[2] + 0.05,
                str(vi), fontsize=size, color=color, ha="center")

def set_ax(ax):
    margin = 0.3
    ax.set_xlim(-margin, 2 + margin)
    ax.set_ylim(-margin, 2 + margin)
    ax.set_zlim(-0.5, 1.0)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.view_init(elev=30, azim=-60)

# ======================================================================
# Cases
# ======================================================================

def run_case_1():
    """Show V, E, T with labels. Print the arrays."""
    print("\n--- Case 1: Mesh Structure ---")
    print(f"\n  V  (vertices, shape {V.shape}):")
    for i, v in enumerate(V):
        print(f"    V[{i}] = {v}")
    print(f"\n  T  (triangles, shape {T.shape}):")
    for i, t in enumerate(T):
        print(f"    T[{i}] = {t}  → vertices {V[t[0]]}, {V[t[1]]}, {V[t[2]]}")
    print(f"\n  E  (edges, shape {E.shape}):")
    for i, e in enumerate(E):
        print(f"    E[{i}] = {e}")

    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")
    draw_mesh_base(ax)

    # Triangle index labels at centroid
    for ti, tri in enumerate(T):
        c = V[tri].mean(axis=0)
        ax.text(c[0], c[1], c[2] + 0.04, f"T{ti}",
                fontsize=7, color="#8e44ad", ha="center")

    # Edge index labels at midpoint
    for ei, (a, b) in enumerate(E):
        mid = (V[a] + V[b]) / 2
        ax.text(mid[0], mid[1], mid[2] + 0.06, f"e{ei}",
                fontsize=6, color="#e74c3c", ha="center")

    set_ax(ax)
    ax.set_title(
        f"Case 1 — Mesh Structure\n"
        f"{len(V)} vertices (black)   {len(T)} triangles (T, purple)   {len(E)} edges (e, red)",
        fontsize=10
    )
    plt.tight_layout()
    plt.show()


def run_case_2():
    """Show face normals as arrows."""
    print("\n--- Case 2: Face Normals ---")
    for ti, n in enumerate(face_norms):
        print(f"  T[{ti}] normal = {n}")

    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")
    draw_mesh_base(ax)

    scale = 0.35
    for ti, tri in enumerate(T):
        c = V[tri].mean(axis=0)
        n = face_norms[ti]
        ax.quiver(c[0], c[1], c[2],
                  n[0]*scale, n[1]*scale, n[2]*scale,
                  color="#e74c3c", linewidth=1.5, arrow_length_ratio=0.3)
        ax.text(c[0] + n[0]*scale*1.1,
                c[1] + n[1]*scale*1.1,
                c[2] + n[2]*scale*1.1 + 0.05,
                f"n{ti}", fontsize=7, color="#c0392b")

    set_ax(ax)
    ax.set_title(
        "Case 2 — Face Normals\n"
        "Arrow = outward normal per triangle  (all point in +Z for this flat mesh)",
        fontsize=10
    )
    plt.tight_layout()
    plt.show()


def run_case_3():
    """Highlight T_v and E_v for the chosen vertex."""
    v = FOCUS_VERTEX
    tris  = T_v[v]
    edges = E_v[v]

    print(f"\n--- Case 3: Vertex adjacency for V[{v}] = {V[v]} ---")
    print(f"  T_v[{v}] = {tris}   (triangles containing this vertex)")
    for ti in tris:
        print(f"    T[{ti}] = {T[ti]}")
    print(f"  E_v[{v}] = {edges}   (edges containing this vertex)")
    for ei in edges:
        print(f"    E[{ei}] = {E[ei]}")

    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")
    draw_mesh_base(ax, highlight_tris=set(tris),
                       highlight_edges=set(edges),
                       highlight_verts={v})

    set_ax(ax)
    ax.set_title(
        f"Case 3 — Vertex Adjacency  (V[{v}] highlighted in red)\n"
        f"Orange triangles = T_v[{v}] = {tris}\n"
        f"Red edges = E_v[{v}] = {edges}",
        fontsize=10
    )
    plt.tight_layout()
    plt.show()


def run_case_4():
    """Step through edge extraction from triangles."""
    print("\n--- Case 4: Edge Extraction ---")
    print("  For each triangle, collect its 3 half-edges (a→b, b→c, c→a).")
    print("  Normalise each as (min, max) and deduplicate.\n")

    all_half = []
    for ti, tri in enumerate(T):
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i+1) % 3])
            norm = (min(a,b), max(a,b))
            all_half.append((ti, a, b, norm))
            print(f"  T[{ti}] edge {a}→{b}  →  normalised {norm}")

    unique = sorted(set(h[3] for h in all_half))
    print(f"\n  After deduplication: {len(unique)} unique edges")
    for i, e in enumerate(unique):
        print(f"    E[{i}] = {e}")

    # Visualise: colour edges by how many triangles share them
    # (boundary edges = 1, interior edges = 2)
    from collections import Counter
    edge_count = Counter(h[3] for h in all_half)

    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")
    draw_mesh_base(ax)

    for ei, (a, b) in enumerate(E):
        key   = (min(a,b), max(a,b))
        count = edge_count[key]
        color = "#2ecc71" if count == 2 else "#e74c3c"  # green=interior, red=boundary
        p0, p1 = V[a], V[b]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                color=color, linewidth=2.5)

    # Legend proxy
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], color="#2ecc71", linewidth=2.5, label="Interior edge (shared by 2 triangles)"),
        Line2D([0], [0], color="#e74c3c", linewidth=2.5, label="Boundary edge (belongs to 1 triangle)"),
    ]
    ax.legend(handles=legend, fontsize=8, loc="upper left")
    set_ax(ax)
    ax.set_title(
        "Case 4 — Edge Extraction\n"
        "Green = interior edges  |  Red = boundary edges",
        fontsize=10
    )
    plt.tight_layout()
    plt.show()


# ======================================================================
# Menu
# ======================================================================

CASES = {
    "1": ("Mesh structure    — V, E, T with labels",              run_case_1),
    "2": ("Face normals      — outward normal per triangle",       run_case_2),
    "3": (f"Vertex adjacency  — T_v and E_v for vertex {FOCUS_VERTEX}",  run_case_3),
    "4": ("Edge extraction   — how unique edges are built from T", run_case_4),
}


def main():
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║                  M1 Mesh Explorer                        ║")
    print("╠══════════════════════════════════════════════════════════╣")
    for key, (desc, _) in CASES.items():
        print(f"║  {key}  {desc:<54}║")
    print("╚══════════════════════════════════════════════════════════╝")

    choice = input("\nSelect a case (1-4): ").strip()
    if choice not in CASES:
        print(f"Invalid choice '{choice}'. Pick 1-4.")
        return

    desc, fn = CASES[choice]
    print(f"\nRunning: {desc}\n")
    fn()


if __name__ == "__main__":
    main()
