"""
Scene configuration for OGC simulation.

A plain dataclass that separates *what the scene is* from *how to simulate it*.
Can be serialized to/from JSON for reproducibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class SceneConfig:
    """
    All parameters needed to define and run an OGC simulation.

    Attributes
    ----------
    cloth_path : str
        Path to the cloth mesh file (OBJ, PLY, STL, GLTF, OFF, etc.).
    obstacle_path : str
        Path to the static obstacle mesh file.
    dt : float
        Time step size.
    gravity : tuple[float, float, float]
        External acceleration vector.
    mass : float
        Per-vertex mass for the cloth.
    k_s : float
        Spring stiffness (elastic).
    k_c : float
        Contact stiffness.
    r : float
        Contact radius.
    r_q : float
        Query radius for BVH broadphase (must be >= r).
    gamma_p : float
        Conservative bound relaxation (0 < gamma_p < 0.5). Eq. 21.
    gamma_e : float
        Re-detection threshold (fraction of vertices). Algorithm 3 line 28.
    n_iter : int
        Number of VBD inner iterations per time step.
    n_steps : int
        Total number of outer time steps to simulate.
    cloth_initial_velocity : tuple[float, float, float]
        Initial velocity for all cloth vertices.
    """

    cloth_path: str = ""
    obstacle_path: str = ""
    dt: float = 0.02
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0)  # auto-detected from mesh if zero
    mass: float = 1.0
    k_s: float = 200.0
    k_c: float = 500.0
    r: float = 0.02
    r_q: float = 0.04
    gamma_p: float = 0.45
    gamma_e: float = 0.0
    n_iter: int = 10
    n_steps: int = 100
    cloth_initial_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_json(self, path: str | Path) -> None:
        """Save config to a JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> SceneConfig:
        """Load config from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        # Convert lists back to tuples for tuple fields
        for key in ("gravity", "cloth_initial_velocity"):
            if key in data and isinstance(data[key], list):
                data[key] = tuple(data[key])
        return cls(**data)
