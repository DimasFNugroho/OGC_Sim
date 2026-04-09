# OGC Simulation — Implementation Plan

## Goal

Implement the **Offset Geometric Contact (OGC)** method from:

> Anka He Chen, Jerry Hsu, Ziheng Liu, Miles Macklin, Yin Yang, Cem Yuksel.
> *Offset Geometric Contact.* ACM Trans. Graph. 44, 4 (SIGGRAPH 2025).

Purpose: understand, validate, and reproduce the paper's cloth–body contact result,
built from scratch in Python in a modular and maintainable way.

---

## Technology Stack

| Purpose              | Library                                      |
|----------------------|----------------------------------------------|
| Core numerics (CPU)  | NumPy                                        |
| GPU / autograd       | PyTorch (tensors, batched ops, CUDA)         |
| Mesh I/O             | `trimesh` or `meshio`                        |
| Broadphase (CPU)     | `scipy.spatial` KD-tree (BVH fallback)       |
| Visualization        | `polyscope`                                  |
| Testing              | `pytest` + finite-difference gradient checks |

---

## Project Structure

```
ogc_sim/
├── geometry/
│   ├── mesh.py              # Mesh dataclass (V, E, T + adjacency)
│   ├── distance.py          # Point-triangle, edge-edge distances & closest points
│   ├── bvh.py               # BVH for broadphase sphere queries
│   └── gauss_map.py         # Polyhedral Gauss Map (convex/concave/mixed classification)
├── contact/
│   ├── offset_geometry.py   # Offset blocks, feasibility region checks (Eq. 8, 9, 15)
│   ├── detection.py         # Algorithm 1 (v-f) and Algorithm 2 (e-e)
│   ├── energy.py            # 2-stage activation function (Eq. 18), gradient, Hessian
│   └── bounds.py            # Conservative displacement bounds (Eq. 21–26)
├── solver/
│   ├── integrator.py        # Backward Euler simulation loop (Algorithm 3)
│   ├── vbd.py               # VBD iteration with contact (Algorithm 4)
│   └── friction.py          # Lagged friction (Li et al. 2020 + VBD mod)
├── materials/
│   └── cloth.py             # Cloth elastic energy (stretching + bending)
├── scenes/
│   └── cloth_body.py        # Cloth draped over rigid body
├── utils/
│   ├── mesh_io.py           # Load/export meshes (OBJ, PLY)
│   └── visualization.py     # Polyscope-based viewer
└── tests/
    ├── test_distance.py     # Unit tests for geometry primitives
    ├── test_contact.py      # Contact detection correctness
    └── test_energy.py       # Energy/gradient/Hessian finite-difference checks
```

---

## Milestones

### M1 — Geometry Primitives
**Files**: `geometry/mesh.py`, `geometry/distance.py`, `geometry/bvh.py`, `geometry/gauss_map.py`

- `Mesh` dataclass: vertices `V (N×3)`, edges `E`, triangles `T`, precomputed adjacency lists (`T_v`, `E_v`, `E_t`), face normals
- `point_triangle_distance(p, tri)` → `(dist, closest_point, closest_face_type)` where face type ∈ {vertex, edge, interior}
- `edge_edge_distance(e1, e2)` → `(dist, closest_point)`
- Axis-aligned BVH over triangles and edges; sphere queries with radius `r_q`
- Polyhedral Gauss Map: classify each vertex as convex / concave / mixed via dihedral angles; precompute per-edge and per-vertex normal cones

**Validation**: unit tests comparing distances to analytic cases; visualize Gauss Map vertex types on a simple mesh.

---

### M2 — Offset Geometry & Contact Detection
**Files**: `contact/offset_geometry.py`, `contact/detection.py`

- Per-face block types: vertex-block `U_v`, edge-block `U_e`, face-interior-block `U_t`
- `check_vertex_feasible_region(x, a)` — Eq. 8: point is in vertex-block iff it lies in the normal cone of vertex `a`
- `check_edge_feasible_region(x, a)` — Eq. 9: point is in edge-block iff it lies in the normal half-space slab of edge `a`
- Edge-edge offset feasibility checks (Eq. 15, Section 3.5)
- `vertex_facet_contact_detection(v, r, r_q, BVH)` → Algorithm 1: returns `FOGC(v)`, `d_min_v`, updates `d_min_t`
- `edge_edge_contact_detection(e, r, r_q, BVH)` → Algorithm 2: returns `EOGC(e)`, `d_min_e`
- Full sweep over all vertices/edges

**Validation**: two triangles at known distance — verify contact pairs and `d_min`; confirm adjacent faces are excluded; test degenerate contacts.

---

### M3 — Contact Energy & Conservative Bounds
**Files**: `contact/energy.py`, `contact/bounds.py`

- 2-stage activation `g(d, r)` — Eq. 18:
  - Quadratic: `(k_c/2)(r-d)²` for `d ∈ [τ, r]`
  - Logarithmic: `-k'_c log(d) + b` for `d ∈ (0, τ)`
  - `τ = r/2` for C² continuity; `k'_c` and `b` from Eq. 19–20
- `contact_energy(X, FOGC, EOGC)` → scalar
- `contact_gradient(X, ...)` → force vector (analytic)
- `contact_hessian_blocks(X, ...)` → per-vertex 3×3 diagonal blocks for VBD
- `compute_conservative_bound(v, d_min_v, d_min_e_v, d_min_t_v, gamma_p)` — Eq. 21
- `truncate_displacement(X, X_prev, bounds)` — clamp each vertex to `b_v`

**Validation**: finite-difference gradient/Hessian checks; confirm `g` is C² at `τ` and C¹ at `r`; verify bounds prevent penetration on two approaching triangles.

---

### M4 — VBD Solver & Backward Euler Integrator
**Files**: `solver/vbd.py`, `solver/integrator.py`, `solver/friction.py`

- `vbd.py` — Algorithm 4:
  - Graph-coloring of the mesh (greedy via `networkx`)
  - Per-vertex Newton step: inertia term + elastic forces/Hessians + contact forces/Hessians
  - Solve 3×3 local system per vertex

- `integrator.py` — Algorithm 3:
  - Outer loop over `n_iter` iterations
  - Contact detection only when `collisionDetectionRequired = True`
  - Initial guess truncation (Eq. 28)
  - After each VBD pass: bound truncation, count exceeded vertices, trigger re-detection if needed
  - Optional convergence check

- `friction.py`: lagged friction direction from previous step's tangential velocity; VBD stability modification (Chen et al. 2024b)

**Validation**: single falling cloth patch with gravity, no contact — verify momentum/energy; two colliding triangles — verify they stop without penetration.

---

### M5 — Cloth Material Model
**File**: `materials/cloth.py`

- Membrane (stretching): StVK or co-rotational linear elasticity per triangle
- Bending: discrete hinge energy (dihedral angle springs) per shared edge
- Per-element gradient and Hessian (3×3 blocks per vertex)

**Validation**: cloth rests correctly under gravity with no solver divergence.

---

### M6 — Scene: Cloth–Body Contact
**File**: `scenes/cloth_body.py`

- Cloth draped over a static rigid body (sphere or box)
- Body treated as a static volumetric mesh with outward-only offset (Section 3.9)
- DCD for cloth-body contact; OGC for cloth self-contact
- Target: cloth wraps the body without penetration; contact forces orthogonal to body surface

---

## Key Risks & Mitigations

| Risk                                       | Mitigation                                                                 |
|--------------------------------------------|----------------------------------------------------------------------------|
| BVH performance in Python                  | Start with `scipy` KD-tree; profile before optimizing; use PyTorch batched ops for GPU |
| Gauss Map edge cases (mixed vertices)      | Follow Echeverria [2007] classification carefully; unit test each vertex type |
| Contact Hessian accuracy                   | Always FD-check gradients/Hessians before integrating into solver          |
| Graph coloring correctness for VBD         | Use greedy coloring via `networkx`; assert no two same-colored vertices share an element |
| Conservative bound being too tight         | Tune `gamma_p` and `r_q`; verify bound limits per-iteration, not per-step  |
| Cloth-body contact normal direction        | Body treated as static; ensure outward-only offset and consistent DCD depth |

---

## Reference Code

- VBD solver: https://github.com/newton-physics/newton/tree/main/newton/_src/solvers/vbd
- Author's prior codebase (Gaia): https://github.com/AnkaChan/Gaia
