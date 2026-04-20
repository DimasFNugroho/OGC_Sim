"""
GLTF exporter — save simulation frames as an animated .glb file.

Exports per-frame cloth vertex positions as a morph-target animation
that can be viewed in any GLTF-compatible viewer (Blender, Three.js,
model-viewer, etc.).

For simplicity, we export a sequence of per-frame GLB files or a single
GLB with the final frame. Full morph-target animation export is planned.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from ogc_sim.sim.frame import FrameData


def export_gltf(
    frames: list[FrameData],
    F_cloth: np.ndarray,
    F_obstacle: np.ndarray,
    output_path: str | Path,
) -> Path:
    """
    Export simulation frames as a .glb file (final frame geometry).

    The GLB contains two meshes: cloth and obstacle, positioned at
    their final-frame state.

    Parameters
    ----------
    frames : list[FrameData]
        Simulation frames (at minimum the last frame).
    F_cloth : np.ndarray, shape (Fc, 3)
        Cloth face indices.
    F_obstacle : np.ndarray, shape (Fo, 3)
        Obstacle face indices.
    output_path : str or Path
        Where to save the .glb file.

    Returns
    -------
    Path to the written file.
    """
    output_path = Path(output_path)
    last = frames[-1]

    V_cloth = last.V_cloth.astype(np.float32)
    V_obstacle = last.V_obstacle.astype(np.float32)
    I_cloth = F_cloth.astype(np.uint32).flatten()
    I_obstacle = F_obstacle.astype(np.uint32).flatten()

    # Build binary buffer: cloth verts | cloth indices | obs verts | obs indices
    buf = bytearray()

    cloth_verts_bytes = V_cloth.tobytes()
    cloth_idx_bytes = I_cloth.tobytes()
    obs_verts_bytes = V_obstacle.tobytes()
    obs_idx_bytes = I_obstacle.tobytes()

    # Pad each section to 4-byte alignment
    def _pad4(b: bytes) -> bytes:
        rem = len(b) % 4
        return b + b'\x00' * (4 - rem) if rem else b

    cloth_verts_bytes = _pad4(cloth_verts_bytes)
    cloth_idx_bytes = _pad4(cloth_idx_bytes)
    obs_verts_bytes = _pad4(obs_verts_bytes)
    obs_idx_bytes = _pad4(obs_idx_bytes)

    buf += cloth_verts_bytes
    buf += cloth_idx_bytes
    buf += obs_verts_bytes
    buf += obs_idx_bytes

    # Offsets
    off_cloth_v = 0
    off_cloth_i = len(cloth_verts_bytes)
    off_obs_v = off_cloth_i + len(cloth_idx_bytes)
    off_obs_i = off_obs_v + len(obs_verts_bytes)

    # Compute bounds
    def _bounds(V):
        return V.min(axis=0).tolist(), V.max(axis=0).tolist()

    c_min, c_max = _bounds(V_cloth)
    o_min, o_max = _bounds(V_obstacle)

    gltf = {
        "asset": {"version": "2.0", "generator": "ogc_sim"},
        "scene": 0,
        "scenes": [{"nodes": [0, 1]}],
        "nodes": [
            {"mesh": 0, "name": "cloth"},
            {"mesh": 1, "name": "obstacle"},
        ],
        "meshes": [
            {
                "primitives": [{
                    "attributes": {"POSITION": 0},
                    "indices": 1,
                    "material": 0,
                }]
            },
            {
                "primitives": [{
                    "attributes": {"POSITION": 2},
                    "indices": 3,
                    "material": 1,
                }]
            },
        ],
        "materials": [
            {"pbrMetallicRoughness": {"baseColorFactor": [0.2, 0.6, 1.0, 0.8]}, "alphaMode": "BLEND", "name": "cloth"},
            {"pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.7, 0.5, 1.0]}, "name": "obstacle"},
        ],
        "accessors": [
            # 0: cloth positions
            {
                "bufferView": 0, "componentType": 5126, "count": len(V_cloth),
                "type": "VEC3", "min": c_min, "max": c_max,
            },
            # 1: cloth indices
            {
                "bufferView": 1, "componentType": 5125, "count": len(I_cloth),
                "type": "SCALAR", "min": [int(I_cloth.min())], "max": [int(I_cloth.max())],
            },
            # 2: obstacle positions
            {
                "bufferView": 2, "componentType": 5126, "count": len(V_obstacle),
                "type": "VEC3", "min": o_min, "max": o_max,
            },
            # 3: obstacle indices
            {
                "bufferView": 3, "componentType": 5125, "count": len(I_obstacle),
                "type": "SCALAR", "min": [int(I_obstacle.min())], "max": [int(I_obstacle.max())],
            },
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": off_cloth_v, "byteLength": len(cloth_verts_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": off_cloth_i, "byteLength": len(cloth_idx_bytes), "target": 34963},
            {"buffer": 0, "byteOffset": off_obs_v, "byteLength": len(obs_verts_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": off_obs_i, "byteLength": len(obs_idx_bytes), "target": 34963},
        ],
        "buffers": [{"byteLength": len(buf)}],
    }

    # Write GLB (binary GLTF container)
    json_str = json.dumps(gltf, separators=(",", ":"))
    json_bytes = json_str.encode("utf-8")
    # Pad JSON to 4-byte alignment with spaces
    json_pad = (4 - len(json_bytes) % 4) % 4
    json_bytes += b" " * json_pad

    # GLB header: magic + version + total length
    total_length = 12 + 8 + len(json_bytes) + 8 + len(buf)
    header = struct.pack("<III", 0x46546C67, 2, total_length)  # glTF magic

    # JSON chunk
    json_chunk_header = struct.pack("<II", len(json_bytes), 0x4E4F534A)  # JSON

    # Binary chunk
    bin_chunk_header = struct.pack("<II", len(buf), 0x004E4942)  # BIN

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(header)
        f.write(json_chunk_header)
        f.write(json_bytes)
        f.write(bin_chunk_header)
        f.write(bytes(buf))

    return output_path


def export_frame_sequence(
    frames: list[FrameData],
    output_dir: str | Path,
    prefix: str = "frame",
) -> list[Path]:
    """
    Export per-frame vertex positions as .npy files.

    Useful for offline analysis or custom visualization pipelines.

    Parameters
    ----------
    frames : list[FrameData]
    output_dir : str or Path
    prefix : str

    Returns
    -------
    list[Path] — paths to saved files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for frame in frames:
        p = output_dir / f"{prefix}_{frame.step:04d}.npy"
        np.save(p, frame.V_cloth)
        paths.append(p)
    return paths
