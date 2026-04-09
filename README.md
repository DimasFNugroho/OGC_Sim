# OGC Sim

A from-scratch Python implementation of **Offset Geometric Contact (OGC)** from the SIGGRAPH 2025 paper:

> Anka He Chen, Jerry Hsu, Ziheng Liu, Miles Macklin, Yin Yang, Cem Yuksel.  
> *Offset Geometric Contact.* ACM Trans. Graph. 44, 4 (SIGGRAPH 2025).

**Target scene**: cloth draped over and colliding with a static rigid body.  
**Goal**: understand, validate, and reproduce the paper's cloth–body contact result.

---

## What is OGC?

Traditional contact methods detect penetration and push objects apart. OGC takes a different approach: it offsets each face of a mesh along its normal, defines volumetric *offset blocks* around every geometric feature (vertices, edges, face interiors), and uses the **Polyhedral Gauss Map** to determine which block "owns" each contact point. This eliminates double-counting and produces contact forces that are always orthogonal to the surface — a key property for stable cloth simulation.

---

## Project Structure

```
ogc_sim/
├── geometry/
│   ├── mesh.py          # Mesh dataclass: V, E, T + adjacency (T_v, E_v, E_t)
│   ├── distance.py      # point_triangle_distance, edge_edge_distance
│   ├── bvh.py           # BVH broadphase: bounding-sphere KD-tree queries
│   └── gauss_map.py     # Polyhedral Gauss Map: vertex/edge normal sets
├── contact/             # (M2) Offset geometry & contact detection
├── solver/              # (M4) VBD solver & backward Euler integrator
├── materials/           # (M5) Cloth elastic energy
├── scenes/              # (M6) Cloth–body scene assembly
├── utils/               # Mesh I/O, visualization
explore/
├── m1/                  # Step-by-step explorers for M1 concepts
tests/
├── test_distance.py     # Unit tests for distance primitives
└── test_gauss_map.py    # Unit tests for Gauss Map
```

---

## Milestones

| # | Name | Status | Description |
|---|------|--------|-------------|
| M1 | Geometry Primitives | ✅ Complete | Mesh, distances, BVH, Gauss Map |
| M2 | Contact Detection | 🔲 Next | Offset blocks, FOGC, EOGC algorithms |
| M3 | Contact Energy & Bounds | 🔲 | 2-stage activation, conservative displacement bounds |
| M4 | VBD Solver | 🔲 | Vertex Block Descent + backward Euler integrator |
| M5 | Cloth Material | 🔲 | Membrane stretching + discrete bending |
| M6 | Cloth–Body Scene | 🔲 | Full cloth-over-rigid-body simulation |

---

## Installation

```bash
git clone https://github.com/DimasFNugroho/OGC_Sim.git
cd OGC_Sim
pip install -e .
```

**Dependencies**: NumPy, SciPy, Matplotlib, pytest

---

## Running the Tests

```bash
pytest tests/ -v
```

All 30 tests should pass (12 distance + 18 Gauss Map).

---

## Explore Scripts

Interactive step-by-step visualizations for each concept. See [`explore/README.md`](explore/README.md) for details.

```bash
python3 explore/m1/m1_distance.py   # Point-triangle & edge-edge distance
python3 explore/m1/m1_mesh.py       # Mesh structure & adjacency
python3 explore/m1/m1_bvh.py        # BVH broadphase pipeline
python3 explore/m1/m1_gauss_map.py  # Polyhedral Gauss Map
```

---

## Key Paper Equations

| Symbol | Description | Location |
|--------|-------------|----------|
| `U_a` | Offset block for face `a` | Sec. 3.2 |
| `FOGC(v)` | Vertex-facet contact face set | Eq. 13 |
| `EOGC(e)` | Edge-edge contact face set | Sec. 3.5 |
| `g(d, r)` | 2-stage contact activation function | Eq. 18 |
| `b_v` | Per-vertex conservative displacement bound | Eq. 21 |
| `r_q` | Query radius (r_q ≥ r) | Algorithm 1 |

---

## Reference Code

- VBD solver: https://github.com/newton-physics/newton/tree/main/newton/_src/solvers/vbd
- Author's prior codebase (Gaia): https://github.com/AnkaChan/Gaia
