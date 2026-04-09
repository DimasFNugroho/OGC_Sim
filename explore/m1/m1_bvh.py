"""
M1 BVH Explorer
===============
Walks through the BVH contact detection process step by step in a single run.

    python3 explore/m1/m1_bvh.py

The process (all steps run in sequence, 1 → 2 → 3 → 4):

  Step 1 — Setup         : place a query vertex and define the contact radius
  Step 2 — BVH broadphase: fast bounding-sphere filter → candidate triangles
  Step 3 — Exact filter  : run point_triangle_distance on each candidate
  Step 4 — Full pipeline : side-by-side summary of what was skipped vs checked

The key insight:
  Without BVH → check ALL triangles with the expensive distance function.
  With BVH    → cheap sphere test first, exact distance only on candidates.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ogc_sim.geometry.mesh     import Mesh
from ogc_sim.geometry.bvh      import BVH
from ogc_sim.geometry.distance import point_triangle_distance

# ======================================================================
# PARAMETERS  — change these and re-run to see how the pipeline changes
# ======================================================================

# <<< CHANGE ME — the query vertex (the "cloth vertex" being tested)
QUERY_VERTEX = np.array([1.0, 1.0, 0.5])

# <<< CHANGE ME — contact radius r_q (how far we look for triangles)
#   try: 0.3  → few candidates
#        0.8  → moderate
#        1.8  → almost everything
QUERY_RADIUS = 0.8

# ======================================================================
# Mesh (same 3×3 grid used throughout M1)
# ======================================================================

V = np.array([
    [0.,0.,0.],[1.,0.,0.],[2.,0.,0.],
    [0.,1.,0.],[1.,1.,0.],[2.,1.,0.],
    [0.,2.,0.],[1.,2.,0.],[2.,2.,0.],
])
T = np.array([
    [0,1,4],[0,4,3],[1,2,5],[1,5,4],
    [3,4,7],[3,7,6],[4,5,8],[4,8,7],
])

mesh = Mesh.from_arrays(V, T)
bvh  = BVH(mesh)

# ======================================================================
# Shared drawing helpers
# ======================================================================

def draw_mesh(ax, tri_colors=None):
    """Draw mesh with optional per-triangle colours."""
    for ti, tri in enumerate(mesh.T):
        pts   = mesh.V[tri]
        color = tri_colors[ti] if tri_colors else "#dfe6e9"
        alpha = 0.5 if tri_colors else 0.15
        ax.add_collection3d(Poly3DCollection(
            [pts], alpha=alpha, facecolor=color, edgecolor="#7f8c8d", linewidth=0.8
        ))
        c = mesh.V[tri].mean(axis=0)
        ax.text(c[0], c[1], c[2]+0.04, f"T{ti}", fontsize=7,
                color="#555", ha="center")

    for ei, (a, b) in enumerate(mesh.E):
        p0, p1 = mesh.V[a], mesh.V[b]
        ax.plot([p0[0],p1[0]], [p0[1],p1[1]], [p0[2],p1[2]],
                color="#aaa", linewidth=0.8)

    for vi, v in enumerate(mesh.V):
        ax.scatter(*v, color="#555", s=15, zorder=5)


def draw_sphere(ax, center, radius, color, alpha=0.12, n=20):
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi,   n)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(n),  np.cos(v))
    ax.plot_wireframe(x, y, z, color=color, alpha=alpha, linewidth=0.4)


def set_ax(ax, title):
    ax.set_xlim(-0.3, 2.3); ax.set_ylim(-0.3, 2.3); ax.set_zlim(-0.3, 1.5)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.view_init(elev=28, azim=-55)
    ax.set_title(title, fontsize=10)


# ======================================================================
# Steps
# ======================================================================

def step_1(ax):
    """Place the query vertex and show the contact radius sphere."""
    print("\n══════════════════════════════════════")
    print(" Step 1 — Setup")
    print("══════════════════════════════════════")
    print(f"  Query vertex  : {QUERY_VERTEX}")
    print(f"  Contact radius: {QUERY_RADIUS}")
    print(f"\n  Goal: find all triangles within r_q = {QUERY_RADIUS} of this vertex.")
    print(f"  Brute force would check all {len(mesh.T)} triangles with the exact")
    print(f"  distance function. BVH filters this down first.")

    draw_mesh(ax)
    ax.scatter(*QUERY_VERTEX, color="#e74c3c", s=120, zorder=7)
    ax.text(QUERY_VERTEX[0], QUERY_VERTEX[1], QUERY_VERTEX[2]+0.15,
            f"query\n{QUERY_VERTEX}", fontsize=8, color="#e74c3c", ha="center")
    draw_sphere(ax, QUERY_VERTEX, QUERY_RADIUS, color="#3498db", alpha=0.10)
    ax.text(QUERY_VERTEX[0]+QUERY_RADIUS*0.7,
            QUERY_VERTEX[1]+QUERY_RADIUS*0.7,
            QUERY_VERTEX[2]+QUERY_RADIUS*0.5,
            f"r_q = {QUERY_RADIUS}", fontsize=8, color="#3498db")
    set_ax(ax, f"Step 1 — Setup\nQuery vertex (red) + contact radius sphere (blue)")


def step_2(ax):
    """BVH broadphase: show which triangles pass the bounding-sphere test."""
    candidates = bvh.sphere_query_triangles(QUERY_VERTEX, QUERY_RADIUS)
    rejected   = [ti for ti in range(len(mesh.T)) if ti not in candidates]

    print("\n══════════════════════════════════════")
    print(" Step 2 — BVH Broadphase")
    print("══════════════════════════════════════")
    print(f"  For each triangle, check:")
    print(f"    dist(query_vertex, centroid) < r_q + half_diagonal")
    print()
    for ti in range(len(mesh.T)):
        c    = bvh._tri_centroids[ti]
        hd   = bvh._tri_half_diags[ti]
        d    = np.linalg.norm(QUERY_VERTEX - c)
        keep = ti in candidates
        mark = "✓ CANDIDATE" if keep else "✗ rejected "
        print(f"  T[{ti}]: dist={d:.3f}  r_q+hd={QUERY_RADIUS+hd:.3f}  → {mark}")

    print(f"\n  Result: {len(candidates)} candidates from {len(mesh.T)} triangles")
    print(f"  Skipped: {len(rejected)} triangles without any exact distance call")

    tri_colors = ["#f39c12" if ti in candidates else "#ecf0f1"
                  for ti in range(len(mesh.T))]
    draw_mesh(ax, tri_colors=tri_colors)

    for ti in candidates:
        draw_sphere(ax, bvh._tri_centroids[ti], bvh._tri_half_diags[ti],
                    color="#f39c12", alpha=0.08)
        ax.scatter(*bvh._tri_centroids[ti], color="#f39c12", s=25, zorder=5)

    ax.scatter(*QUERY_VERTEX, color="#e74c3c", s=100, zorder=7)
    draw_sphere(ax, QUERY_VERTEX, QUERY_RADIUS, color="#3498db", alpha=0.08)
    set_ax(ax, f"Step 2 — BVH Broadphase\n"
               f"Orange = {len(candidates)} candidates  |  Gray = {len(rejected)} skipped\n"
               f"(cheap sphere test only — no exact distance computed yet)")

    return candidates


def step_3(ax, candidates):
    """Exact filter: run point_triangle_distance on each BVH candidate."""
    print("\n══════════════════════════════════════")
    print(" Step 3 — Exact Distance Filter")
    print("══════════════════════════════════════")
    print(f"  Running point_triangle_distance on {len(candidates)} candidates...\n")

    confirmed = []
    false_pos = []
    for ti in candidates:
        a, b, c = mesh.V[mesh.T[ti][0]], mesh.V[mesh.T[ti][1]], mesh.V[mesh.T[ti][2]]
        dist, closest, feature, _ = point_triangle_distance(QUERY_VERTEX, a, b, c)
        within = dist < QUERY_RADIUS
        if within:
            confirmed.append((ti, dist, closest, feature))
            mark = "✓ within r_q  ← CONTACT"
        else:
            false_pos.append(ti)
            mark = "✗ outside r_q (BVH false positive)"
        print(f"  T[{ti}]: exact dist = {dist:.4f}   {mark}")

    print(f"\n  BVH candidates : {len(candidates)}")
    print(f"  Confirmed      : {len(confirmed)}  (exact dist < r_q)")
    print(f"  False positives: {len(false_pos)}  (passed BVH but too far exactly)")
    print(f"  Skipped entirely: {len(mesh.T) - len(candidates)}  (never computed)")

    tri_colors = []
    for ti in range(len(mesh.T)):
        if any(t == ti for t, *_ in confirmed):
            tri_colors.append("#2ecc71")
        elif ti in false_pos:
            tri_colors.append("#f1c40f")
        else:
            tri_colors.append("#ecf0f1")

    draw_mesh(ax, tri_colors=tri_colors)

    for ti, dist, closest, feature in confirmed:
        ax.plot([QUERY_VERTEX[0], closest[0]],
                [QUERY_VERTEX[1], closest[1]],
                [QUERY_VERTEX[2], closest[2]],
                "--", color="#27ae60", linewidth=1.2)
        ax.scatter(*closest, color="#27ae60", s=40, marker="*", zorder=6)

    ax.scatter(*QUERY_VERTEX, color="#e74c3c", s=100, zorder=7)
    draw_sphere(ax, QUERY_VERTEX, QUERY_RADIUS, color="#3498db", alpha=0.08)

    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor="#2ecc71", label=f"Confirmed contact ({len(confirmed)})"),
        Patch(facecolor="#f1c40f", label=f"BVH false positive ({len(false_pos)})"),
        Patch(facecolor="#ecf0f1", label=f"Skipped ({len(mesh.T)-len(candidates)})"),
    ]
    ax.legend(handles=legend, fontsize=8, loc="upper left")
    set_ax(ax, "Step 3 — Exact Distance Filter\n"
               "Green = contact  |  Yellow = BVH false positive  |  Gray = never checked")

    return confirmed


def step_4(ax1, ax2):
    """Full pipeline summary — brute force vs BVH side by side."""
    candidates = bvh.sphere_query_triangles(QUERY_VERTEX, QUERY_RADIUS)

    brute_contacts = []
    for ti in range(len(mesh.T)):
        a, b, c = mesh.V[mesh.T[ti][0]], mesh.V[mesh.T[ti][1]], mesh.V[mesh.T[ti][2]]
        dist, _, _, _ = point_triangle_distance(QUERY_VERTEX, a, b, c)
        if dist < QUERY_RADIUS:
            brute_contacts.append(ti)

    bvh_contacts = []
    for ti in candidates:
        a, b, c = mesh.V[mesh.T[ti][0]], mesh.V[mesh.T[ti][1]], mesh.V[mesh.T[ti][2]]
        dist, _, _, _ = point_triangle_distance(QUERY_VERTEX, a, b, c)
        if dist < QUERY_RADIUS:
            bvh_contacts.append(ti)

    print("\n══════════════════════════════════════")
    print(" Step 4 — Full Pipeline Summary")
    print("══════════════════════════════════════")
    print(f"  Total triangles       : {len(mesh.T)}")
    print()
    print(f"  Brute force:")
    print(f"    exact distance calls: {len(mesh.T)}  (every triangle)")
    print(f"    contacts found      : {sorted(brute_contacts)}")
    print()
    print(f"  BVH pipeline:")
    print(f"    BVH candidates      : {len(candidates)}  (bounding sphere test)")
    print(f"    exact distance calls: {len(candidates)}  (candidates only)")
    print(f"    contacts found      : {sorted(bvh_contacts)}")
    print()
    print(f"  Exact distance calls saved: {len(mesh.T) - len(candidates)}")
    print(f"  Same result? {sorted(brute_contacts) == sorted(bvh_contacts)}")

    # Left: brute force
    tri_colors_bf = ["#2ecc71" if ti in brute_contacts else "#e8daef"
                     for ti in range(len(mesh.T))]
    draw_mesh(ax1, tri_colors=tri_colors_bf)
    ax1.scatter(*QUERY_VERTEX, color="#e74c3c", s=100, zorder=7)
    draw_sphere(ax1, QUERY_VERTEX, QUERY_RADIUS, color="#3498db", alpha=0.08)
    set_ax(ax1, f"Brute Force\n{len(mesh.T)} exact distance calls\n"
                f"contacts: {sorted(brute_contacts)}")

    # Right: BVH pipeline
    tri_colors_bvh = []
    for ti in range(len(mesh.T)):
        if ti in bvh_contacts:
            tri_colors_bvh.append("#2ecc71")
        elif ti in candidates:
            tri_colors_bvh.append("#f39c12")
        else:
            tri_colors_bvh.append("#ecf0f1")
    draw_mesh(ax2, tri_colors=tri_colors_bvh)
    ax2.scatter(*QUERY_VERTEX, color="#e74c3c", s=100, zorder=7)
    draw_sphere(ax2, QUERY_VERTEX, QUERY_RADIUS, color="#3498db", alpha=0.08)
    set_ax(ax2, f"BVH Pipeline\n{len(candidates)} exact calls  "
                f"(saved {len(mesh.T)-len(candidates)})\n"
                f"contacts: {sorted(bvh_contacts)}")


# ======================================================================
# Main — run all steps sequentially
# ======================================================================

def main():
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║                    M1 BVH Explorer                           ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  Running all 4 steps in sequence.                            ║")
    print("║  Close each plot window to proceed to the next step.         ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # ------------------------------------------------------------------
    # Step 1 — Setup
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection="3d")
    step_1(ax)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Step 2 — BVH Broadphase
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection="3d")
    candidates = step_2(ax)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Step 3 — Exact Filter
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection="3d")
    step_3(ax, candidates)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Step 4 — Full Pipeline Summary (side by side)
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(14, 6))
    step_4(ax1, ax2)
    plt.suptitle("Step 4 — Brute Force vs BVH  (same result, fewer distance calls)",
                 fontsize=11)
    plt.tight_layout()
    plt.show()

    print("\n✓ All steps complete.")


if __name__ == "__main__":
    main()
