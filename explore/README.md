# Explore Scripts

Step-by-step interactive scripts that walk through each concept in the simulation pipeline. Every script runs all steps sequentially — close a plot window to proceed to the next step.

These scripts are meant to be read alongside the implementation. They prioritise **understanding the process** over visual polish.

---

## M1 — Geometry Primitives

### `m1_distance.py` — Distance Functions

```bash
python3 explore/m1/m1_distance.py
```

Walks through the two core distance primitives used in all contact queries.

| Case | What it shows |
|------|---------------|
| 1 | `point_triangle_distance` — query point vs face interior (closest feature = face) |
| 2 | `point_triangle_distance` — query point vs edge (closest feature = edge) |
| 3 | `point_triangle_distance` — query point vs vertex (closest feature = vertex) |
| 4 | `edge_edge_distance` — two skew edges, closest point on each segment |

**Key concept**: any contact between two triangle meshes reduces to either a *vertex–face* query or an *edge–edge* query. There is no face–face case because two intersecting faces always imply a prior edge–edge crossing.

---

### `m1_mesh.py` — Mesh Data Structure

```bash
python3 explore/m1/m1_mesh.py
```

Visualises the data structures that `geometry/mesh.py` builds from a triangle soup.

| Case | What it shows |
|------|---------------|
| 1 | Vertices `V`, edges `E`, triangles `T` with index labels |
| 2 | Per-triangle face normals (CCW winding → outward normal) |
| 3 | Vertex adjacency: `T_v[i]` (incident triangles) and `E_v[i]` (incident edges) |
| 4 | Edge extraction: how unique undirected edges are built, interior vs boundary |

**Changeable parameter**: `FOCUS_VERTEX` (line 38) — pick any vertex index 0–8.

---

### `m1_bvh.py` — BVH Broadphase Pipeline

```bash
python3 explore/m1/m1_bvh.py
```

Demonstrates why a BVH (Bounding Volume Hierarchy) is needed before running expensive exact distance queries, and how the bounding-sphere filter works.

| Step | What it shows |
|------|---------------|
| 1 | Query vertex placed in the scene, contact radius sphere drawn |
| 2 | BVH broadphase: per-triangle bounding sphere test — candidates (orange) vs skipped (gray) |
| 3 | Exact filter: `point_triangle_distance` on candidates only — confirmed contacts (green) vs BVH false positives (yellow) |
| 4 | Side-by-side: brute force (checks all triangles) vs BVH pipeline (checks only candidates) — same result, fewer calls |

**Changeable parameters**: `QUERY_VERTEX` and `QUERY_RADIUS` (lines 37–43).

**Key concept**: the BVH does a cheap sphere-overlap test first. Only triangles whose bounding sphere overlaps the query sphere proceed to the expensive exact distance computation.

---

### `m1_gauss_map.py` — Polyhedral Gauss Map

```bash
python3 explore/m1/m1_gauss_map.py
```

Explains how the Gauss Map assigns a *set of normals* to each geometric feature, and how that set is used to determine which feature "owns" a given contact.

| Step | What it shows |
|------|---------------|
| 1 | Vertex classification: CONVEX / CONCAVE / MIXED, colour-coded on mesh |
| 2 | Edge normal slabs: the arc of normals between two adjacent face normals, shown on the unit sphere |
| 3 | Vertex normal cones: the fan of face normals around a vertex, shown on the unit sphere |
| 4 | Feasibility queries: test directions shown as green (inside) / red (outside) arrows against a cone and a slab |

**Two meshes used**:
- Flat 3×3 grid — all vertices CONVEX, degenerate cones (upper hemisphere)
- Box-corner mesh — proper 3-generator cone (1/8-sphere octant)

**Key concept** (paper Eq. 8–9): a contact is valid for a feature only if the contact direction `d = query − closest_point` lies inside the feature's normal set. This prevents the same physical contact from being double-counted by adjacent features.

**Changeable parameters**: `FOCUS_VERTEX` (line 19) and `FOCUS_EDGE` (line 22).
