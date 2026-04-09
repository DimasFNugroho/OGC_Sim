# CLAUDE.md — OGC Simulation Project

## Project Overview

This project implements **Offset Geometric Contact (OGC)** from the SIGGRAPH 2025 paper
(see `Offset_Geometric_Contact-SIGGRAPH2025.pdf`) from scratch in Python.
The full implementation plan is in `PLAN.md`.

**Target scene**: cloth draped over and colliding with a static rigid body.
**Goal**: understand, validate, and reproduce the paper's cloth–body contact result.

---

## Conventions

### Language & Style
- Python only. NumPy for CPU numerics; PyTorch for GPU-accelerated paths.
- Follow PEP 8. Use type hints on all public functions.
- No external simulation frameworks — everything is built from scratch.
- Prefer clarity over premature optimization. Profile before optimizing.

### Math & Notation
- Match the paper's notation in code: `d_min_v`, `b_v`, `gamma_p`, `r_q`, `FOGC`, `EOGC`, etc.
- Vectors are always NumPy arrays of shape `(3,)` or `(N, 3)`. Never use flat 1D arrays for positions.
- Equation references in comments should cite the paper: e.g. `# Eq. 18` or `# Algorithm 1, line 6`.

### Modularity
- Each module has a single, clear responsibility (see project structure in `PLAN.md`).
- No cross-imports between `contact/` and `solver/` — the solver calls into contact, not the reverse.
- Scene files in `scenes/` are the only place that assemble all components together.

---

## Testing Requirements

- Every geometric primitive (`distance.py`, `offset_geometry.py`) must have unit tests with analytic ground truth.
- Every energy function in `contact/energy.py` and `materials/` must pass a **finite-difference gradient check** and a **finite-difference Hessian check** before being used in the solver.
- Contact detection must be tested for:
  - Adjacent faces correctly excluded
  - `d_min` values correct
  - No duplicate or missed contact pairs
- Tests live in `tests/` and are run with `pytest`.

---

## Validation Approach

At each milestone, validate before moving to the next:

| Milestone | Validation target |
|-----------|-------------------|
| M1        | Analytic distance tests; Gauss Map types visualized on simple mesh |
| M2        | Two-triangle contact detection matches known geometry |
| M3        | FD gradient/Hessian checks pass; penetration-free on approaching triangles |
| M4        | Free-falling cloth conserves momentum; two-triangle collision stops cleanly |
| M5        | Cloth rests correctly under gravity with no solver divergence |
| M6        | Cloth wraps body without penetration; visual comparison to paper figures |

---

## Working Approach

- Implement and test one milestone fully before starting the next.
- When implementing an algorithm from the paper, reference the exact algorithm number and line.
- For any function involving gradients or Hessians, write the FD check in the same PR/session.
- Do not add features beyond what the current milestone requires.
- Keep scene files thin — they should configure and run, not contain physics logic.

---

## Key Paper Equations Reference

| Symbol       | Description                                      | Location     |
|--------------|--------------------------------------------------|--------------|
| `U_a`        | Offset block for face `a`                        | Sec. 3.2     |
| `FOGC(v)`    | Vertex-facet contact face set                    | Eq. 13       |
| `EOGC(e)`    | Edge-edge contact face set                       | Sec. 3.5     |
| `g(d, r)`    | 2-stage contact activation function              | Eq. 18       |
| `tau = r/2`  | Stitch point for C² continuity                   | Sec. 3.6     |
| `b_v`        | Per-vertex conservative displacement bound       | Eq. 21       |
| `gamma_p`    | Relaxation parameter for bound (0 < γ_p < 0.5)  | Eq. 21       |
| `r_q`        | Query radius (r_q >= r)                          | Algorithm 1  |
| `d_min_v`    | Min distance from vertex v to non-adjacent faces | Eq. 22       |
| `d_min_e_v`  | Min edge distance for v's neighbor edges         | Eq. 23–24    |
| `d_min_t_v`  | Min vertex distance for v's neighbor facets      | Eq. 25–26    |

---

## Reference Code (for reference only — do not copy directly)

- VBD solver implementation: https://github.com/newton-physics/newton/tree/main/newton/_src/solvers/vbd
- Author's prior codebase: https://github.com/AnkaChan/Gaia
