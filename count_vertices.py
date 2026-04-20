#!/usr/bin/env python3
"""
Count vertices and faces in a 3D mesh file.

Usage:
    python3 count_vertices.py <path/to/mesh.obj>
"""

import sys
from pathlib import Path

def count_vertices_obj(filepath):
    """Count vertices in an OBJ file by parsing lines."""
    filepath = Path(filepath).expanduser().resolve()

    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        return None, None

    vertices = 0
    faces = 0

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):  # vertex
                vertices += 1
            elif line.startswith('f '):  # face
                faces += 1

    return vertices, faces


def count_vertices_trimesh(filepath):
    """Count vertices using trimesh (handles multiple formats)."""
    try:
        import trimesh
        mesh = trimesh.load(filepath)
        if isinstance(mesh, trimesh.Scene):
            # Multiple meshes in scene
            total_verts = 0
            total_faces = 0
            for geom in mesh.geometry.values():
                if isinstance(geom, trimesh.Trimesh):
                    total_verts += len(geom.vertices)
                    total_faces += len(geom.faces)
            return total_verts, total_faces
        elif isinstance(mesh, trimesh.Trimesh):
            return len(mesh.vertices), len(mesh.faces)
    except Exception as e:
        print(f"Error loading with trimesh: {e}")
        return None, None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 count_vertices.py <path/to/mesh.obj>")
        sys.exit(1)

    filepath = sys.argv[1]

    print(f"Analyzing: {filepath}")
    print()

    # Try OBJ parser first (faster, OBJ-only)
    if filepath.endswith('.obj'):
        print("[Method 1: OBJ parser]")
        verts, faces = count_vertices_obj(filepath)
        if verts is not None:
            print(f"  Vertices: {verts:,}")
            print(f"  Faces:    {faces:,}")
        print()

    # Try trimesh (slower, handles all formats)
    print("[Method 2: trimesh (with merging)]")
    verts, faces = count_vertices_trimesh(filepath)
    if verts is not None:
        print(f"  Vertices: {verts:,}")
        print(f"  Faces:    {faces:,}")

        # Performance estimate
        print()
        print("Performance estimates:")
        if verts > 10000:
            print(f"  ⚠️  Large mesh ({verts:,} verts) - single step may take 10-60s")
        elif verts > 5000:
            print(f"  ⚠️  Medium-large mesh ({verts:,} verts) - single step may take 2-10s")
        elif verts > 1000:
            print(f"  ✓ Medium mesh ({verts:,} verts) - should be manageable")
        else:
            print(f"  ✓ Small mesh ({verts:,} verts) - should be fast")
    else:
        print("  Could not load with trimesh")
