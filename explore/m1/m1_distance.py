"""
M1 Distance Explorer
====================
Run this script and pick which case to visualise from a menu.

    python3 explore/m1_distance.py

The 4 cases correspond directly to the 4 primitive contact pair types
that cover all possible mesh collisions:

  1 — vertex ↔ face interior  (point_triangle_distance → FACE_INTERIOR)
  2 — vertex ↔ edge           (point_triangle_distance → EDGE)
  3 — vertex ↔ vertex         (point_triangle_distance → VERTEX)
  4 — edge   ↔ edge           (edge_edge_distance)

Edit the PARAMETERS block of whichever case you choose, then re-run.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ogc_sim.geometry.distance import (
    point_triangle_distance,
    edge_edge_distance,
    ClosestFeature,
)

# ======================================================================
# PARAMETERS — edit the block for the case you want to run, then re-run
# ======================================================================

# Shared triangle (Cases 1, 2, 3)
# <<< CHANGE ME — try moving vertices to see how regions shift
A = np.array([0.0, 0.0, 0.0])
B = np.array([2.0, 0.0, 0.0])
C = np.array([0.0, 2.0, 0.0])

# Case 1 — vertex ↔ face interior
# <<< CHANGE ME — keep p above the face interior (not past any edge or corner)
QUERY_FACE = np.array([0.5, 0.5, 1.5])

# Case 2 — vertex ↔ edge
# <<< CHANGE ME — place p outside one of the triangle's edges
#   Beside edge ab (bottom):    [ 1.0, -1.0, 0.5]
#   Beside edge bc (diagonal):  [ 1.5,  1.5, 0.5]
#   Beside edge ca (left):      [-1.0,  1.0, 0.5]
QUERY_EDGE = np.array([1.0, -1.0, 0.5])

# Case 3 — vertex ↔ vertex
# <<< CHANGE ME — place p past one of the triangle's corners
#   Past vertex a: [-0.5, -0.5, 0.5]
#   Past vertex b: [ 2.5, -0.5, 0.5]
#   Past vertex c: [-0.5,  2.5, 0.5]
QUERY_VERTEX = np.array([2.5, -0.5, 0.5])

# Case 4 — edge ↔ edge
# <<< CHANGE ME — try parallel, skew, perpendicular, or touching segments
EDGE1_P = np.array([-1.0,  0.0, 0.0])
EDGE1_Q = np.array([ 1.0,  0.0, 0.0])
EDGE2_R = np.array([ 0.0, -1.0, 1.5])
EDGE2_S = np.array([ 0.0,  1.0, 1.5])

# ======================================================================
# Shared drawing helpers
# ======================================================================

FEATURE_COLOR = {
    ClosestFeature.VERTEX:        "#e74c3c",
    ClosestFeature.EDGE:          "#e67e22",
    ClosestFeature.FACE_INTERIOR: "#27ae60",
}
FEATURE_LABEL = {
    ClosestFeature.VERTEX:        "VERTEX",
    ClosestFeature.EDGE:          "EDGE",
    ClosestFeature.FACE_INTERIOR: "FACE INTERIOR",
}


def draw_triangle(ax, a, b, c, alpha=0.18, color="#3498db"):
    poly = Poly3DCollection([[a, b, c]], alpha=alpha,
                             facecolor=color, edgecolor="steelblue", linewidth=1.5)
    ax.add_collection3d(poly)
    for pt, lbl in zip([a, b, c], ["a", "b", "c"]):
        ax.text(pt[0], pt[1], pt[2] + 0.08, lbl, fontsize=9,
                color="steelblue", ha="center")


def draw_query(ax, p, closest, feature, dist):
    color = FEATURE_COLOR[feature]
    ax.scatter(*p, color="black", s=50, zorder=5)
    ax.text(p[0], p[1], p[2] + 0.14, "p", fontsize=9, color="black", ha="center")
    ax.scatter(*closest, color=color, s=80, zorder=5, marker="*")
    ax.plot([p[0], closest[0]], [p[1], closest[1]], [p[2], closest[2]],
            "--", color=color, linewidth=1.5)
    mid = (p + closest) / 2
    ax.text(mid[0], mid[1], mid[2] + 0.1,
            f"d = {dist:.3f}\n{FEATURE_LABEL[feature]}", fontsize=8,
            color=color, ha="center")


def fit_axes(ax, points, margin=0.6):
    pts = np.array(points)
    lo = pts.min(axis=0) - margin
    hi = pts.max(axis=0) + margin
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")


# ======================================================================
# Case implementations
# ======================================================================

def _run_vt_case(case_num, label, p, expected_feature):
    """Shared runner for the three point-triangle cases (1, 2, 3)."""
    dist, closest, feature, feat_idx = point_triangle_distance(p, A, B, C)

    print(f"\n--- Case {case_num}: {label} ---")
    print(f"  query point   : {p}")
    print(f"  distance      : {dist:.6f}")
    print(f"  closest point : {closest}")
    print(f"  feature       : {FEATURE_LABEL[feature]}  (index {feat_idx})")
    if feature != expected_feature:
        print(f"  ⚠  Expected {FEATURE_LABEL[expected_feature]} — "
              f"try adjusting the query point in PARAMETERS.")

    fig = plt.figure(figsize=(7, 6))
    ax  = fig.add_subplot(111, projection="3d")
    draw_triangle(ax, A, B, C)
    draw_query(ax, p, closest, feature, dist)
    fit_axes(ax, [A, B, C, p])
    ax.set_title(
        f"Case {case_num} — {label}\n"
        f"p = {p}\n"
        f"dist = {dist:.4f}   feature = {FEATURE_LABEL[feature]}   idx = {feat_idx}",
        fontsize=10
    )
    plt.tight_layout()
    plt.show()


def run_case_1():
    _run_vt_case(1, "vertex ↔ face interior", QUERY_FACE, ClosestFeature.FACE_INTERIOR)


def run_case_2():
    _run_vt_case(2, "vertex ↔ edge", QUERY_EDGE, ClosestFeature.EDGE)


def run_case_3():
    _run_vt_case(3, "vertex ↔ vertex", QUERY_VERTEX, ClosestFeature.VERTEX)


def run_case_4():
    dist_ee, closest_pq = edge_edge_distance(EDGE1_P, EDGE1_Q, EDGE2_R, EDGE2_S)

    # Recover closest point on edge 2
    d2 = EDGE2_S - EDGE2_R
    t   = np.dot(closest_pq - EDGE2_R, d2) / (np.dot(d2, d2) + 1e-30)
    closest_rs = EDGE2_R + float(np.clip(t, 0.0, 1.0)) * d2

    print("\n--- Case 4: edge ↔ edge ---")
    print(f"  edge 1        : {EDGE1_P} -> {EDGE1_Q}")
    print(f"  edge 2        : {EDGE2_R} -> {EDGE2_S}")
    print(f"  distance      : {dist_ee:.6f}")
    print(f"  closest on e1 : {closest_pq}")
    print(f"  closest on e2 : {closest_rs}")

    fig = plt.figure(figsize=(7, 6))
    ax  = fig.add_subplot(111, projection="3d")

    for pts, color, lbl in [
        ([EDGE1_P, EDGE1_Q], "#2980b9", "Edge 1 (pq)"),
        ([EDGE2_R, EDGE2_S], "#8e44ad", "Edge 2 (rs)"),
    ]:
        xs, ys, zs = zip(*pts)
        ax.plot(xs, ys, zs, "-o", color=color, linewidth=2.5, markersize=7, label=lbl)

    ax.scatter(*closest_pq, color="#e74c3c", s=100, zorder=5, marker="*",
               label="Closest on e1")
    ax.scatter(*closest_rs, color="#e67e22", s=100, zorder=5, marker="*",
               label="Closest on e2")
    ax.plot([closest_pq[0], closest_rs[0]],
            [closest_pq[1], closest_rs[1]],
            [closest_pq[2], closest_rs[2]],
            "--", color="gray", linewidth=1.5)
    mid = (closest_pq + closest_rs) / 2
    ax.text(mid[0], mid[1], mid[2] + 0.12,
            f"d = {dist_ee:.4f}", fontsize=10, color="gray")

    fit_axes(ax, [EDGE1_P, EDGE1_Q, EDGE2_R, EDGE2_S])
    ax.legend(fontsize=8)
    ax.set_title(f"Case 4 — edge ↔ edge\ndist = {dist_ee:.4f}", fontsize=11)
    plt.tight_layout()
    plt.show()


# ======================================================================
# Menu
# ======================================================================

CASES = {
    "1": ("vertex ↔ face interior  (point_triangle → FACE_INTERIOR)", run_case_1),
    "2": ("vertex ↔ edge           (point_triangle → EDGE)         ", run_case_2),
    "3": ("vertex ↔ vertex         (point_triangle → VERTEX)       ", run_case_3),
    "4": ("edge   ↔ edge           (edge_edge_distance)            ", run_case_4),
}


def main():
    print("\n╔═════════════════════════════════════════════════════════════╗")
    print("║                 M1 Distance Explorer                        ║")
    print("╠═════════════════════════════════════════════════════════════╣")
    for key, (desc, _) in CASES.items():
        print(f"║  {key}  {desc:<54}║")
    print("╚═════════════════════════════════════════════════════════════╝")

    choice = input("\nSelect a case (1-4): ").strip()
    if choice not in CASES:
        print(f"Invalid choice '{choice}'. Pick 1-4.")
        return

    desc, fn = CASES[choice]
    print(f"\nRunning: {desc}\n")
    fn()


if __name__ == "__main__":
    main()
