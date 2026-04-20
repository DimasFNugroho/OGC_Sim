"""
Mesh I/O — load any 3D file format via trimesh.

Supports OBJ, PLY, STL, GLTF/GLB, OFF, and anything trimesh can handle.
Returns (V, F) arrays ready for Mesh.from_arrays().
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


def load_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a 3D mesh file and return vertices and triangular faces.

    Handles scenes (multiple meshes) by concatenating them into one.
    Non-triangular faces are automatically triangulated by trimesh.

    Parameters
    ----------
    path : str or Path
        Path to the mesh file. Supports ~ for home directory expansion.

    Returns
    -------
    V : np.ndarray, shape (N, 3), float64
        Vertex positions.
    F : np.ndarray, shape (M, 3), int32
        Triangle face indices.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file contains no geometry.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")

    loaded = trimesh.load(str(path), force="mesh")

    # trimesh.load may return a Scene if the file has multiple objects
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No triangle meshes found in: {path}")
        combined = trimesh.util.concatenate(meshes)
    elif isinstance(loaded, trimesh.Trimesh):
        combined = loaded
    else:
        raise ValueError(f"Unsupported geometry type from {path}: {type(loaded)}")

    V = np.asarray(combined.vertices, dtype=np.float64)
    F = np.asarray(combined.faces, dtype=np.int32)

    if len(V) == 0 or len(F) == 0:
        raise ValueError(f"Empty mesh loaded from: {path}")

    return V, F
