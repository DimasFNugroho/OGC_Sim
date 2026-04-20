"""
OGCSimulator — orchestrates the full simulation loop.

Loads meshes, initialises state, and calls Algorithm 3 (simulation_step)
each time step. Emits FrameData to any registered callback.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from ogc_sim.geometry.mesh import Mesh
from ogc_sim.io.loader import load_mesh
from ogc_sim.sim.config import SceneConfig
from ogc_sim.sim.frame import FrameData
from ogc_sim.algorithms.algorithm3 import simulation_step, StepResult
from ogc_sim.solver.vbd import graph_color_mesh, compute_rest_lengths

_CUDA_AVAILABLE = torch.cuda.is_available()
if _CUDA_AVAILABLE:
    from ogc_sim.solver.vbd_gpu import build_gpu_mesh_data, GPUMeshData
    _GPU_DEVICE = torch.device("cuda")
    print("[runner] CUDA detected — GPU acceleration enabled.", flush=True)
else:
    _GPU_DEVICE = None
    print("[runner] No CUDA — running on CPU.", flush=True)


class OGCSimulator:
    """
    End-to-end OGC simulation runner.

    Usage
    -----
    >>> sim = OGCSimulator()
    >>> sim.load(config)
    >>> sim.run(callback=my_backend.update)

    Or step-by-step:
    >>> sim.load(config)
    >>> for i in range(n):
    ...     frame = sim.step()
    ...     do_something(frame)
    """

    def __init__(self) -> None:
        self.config: SceneConfig | None = None

        # Mesh data
        self.V_cloth_init: np.ndarray | None = None
        self.F_cloth: np.ndarray | None = None
        self.V_obstacle: np.ndarray | None = None
        self.F_obstacle: np.ndarray | None = None

        # Simulation state
        self.X: np.ndarray | None = None       # current cloth positions
        self.v: np.ndarray | None = None       # current cloth velocities
        self.step_num: int = 0

        # Precomputed
        self._colors: list[list[int]] | None = None
        self._l0: np.ndarray | None = None
        self._a_ext: np.ndarray | None = None
        self._gpu_mesh_data = None   # GPUMeshData | None

        self._loaded = False
        self._running = False
        self._paused = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def is_running(self) -> bool:
        return self._running

    def load(self, config: SceneConfig) -> FrameData:
        """
        Load meshes and initialise simulation state.

        Parameters
        ----------
        config : SceneConfig

        Returns
        -------
        FrameData for the initial frame (step=0).
        """
        self.config = config

        # Load meshes via trimesh
        self.V_cloth_init, self.F_cloth = load_mesh(config.cloth_path)
        self.V_obstacle, self.F_obstacle = load_mesh(config.obstacle_path)

        # Initial state
        self.X = self.V_cloth_init.copy()
        self.v = np.tile(
            np.array(config.cloth_initial_velocity, dtype=np.float64),
            (len(self.X), 1),
        )
        self.step_num = 0

        # Precompute graph coloring on cloth mesh
        cloth_mesh = Mesh.from_arrays(self.V_cloth_init, self.F_cloth)
        self._colors = graph_color_mesh(cloth_mesh)

        # Precompute rest lengths on combined mesh
        N_cloth = len(self.V_cloth_init)
        T_all = np.vstack([
            self.F_cloth,
            self.F_obstacle + N_cloth,
        ])
        V_combined = np.vstack([self.V_cloth_init, self.V_obstacle])
        combined_mesh = Mesh.from_arrays(V_combined, T_all)
        self._l0 = compute_rest_lengths(combined_mesh)

        # Auto-detect gravity direction from the scene geometry.
        # Strategy: the "up" axis is the one where the combined scene centroid
        # is furthest from zero — 3D assets are almost always placed above the
        # ground plane (Y>0 for Y-up, Z>0 for Z-up).  This is more robust
        # than bounding-box extent, which fails for box-shaped obstacles.
        config_g = np.array(config.gravity, dtype=np.float64)
        if np.linalg.norm(config_g) < 0.1:
            centroid = V_combined.mean(axis=0)
            up_axis = int(np.argmax(np.abs(centroid)))
            config_g = np.zeros(3)
            config_g[up_axis] = -9.8
            axis_name = ["X", "Y", "Z"][up_axis]
            print(
                f"[runner] Auto-detected up-axis: {axis_name} "
                f"(centroid={centroid.round(3)}) → gravity = {config_g}",
                flush=True,
            )
        self._a_ext = config_g

        # Build GPU mesh data once (topology never changes during simulation)
        if _CUDA_AVAILABLE:
            self._gpu_mesh_data = build_gpu_mesh_data(
                combined_mesh, self._l0, _GPU_DEVICE
            )
            print(
                f"[runner] GPUMeshData built: "
                f"{len(combined_mesh.E)} edges, "
                f"{combined_mesh.num_triangles} tris on {_GPU_DEVICE}",
                flush=True,
            )
        else:
            self._gpu_mesh_data = None

        self._loaded = True
        self._paused = False

        return FrameData(
            step=0,
            V_cloth=self.X.copy(),
            V_obstacle=self.V_obstacle.copy(),
            v_cloth=self.v.copy(),
        )

    def step(self) -> FrameData:
        """
        Run one simulation time step (Algorithm 3).

        Returns
        -------
        FrameData with the new state.
        """
        if not self._loaded:
            raise RuntimeError("Call load() before step().")

        cfg = self.config
        import sys
        print(f"[SIM] Step {self.step_num + 1}: starting Algorithm 3...", flush=True)
        sys.stdout.flush()

        result: StepResult = simulation_step(
            X_t=self.X,
            v_t=self.v,
            V_floor=self.V_obstacle,
            T_cloth=self.F_cloth,
            T_floor=self.F_obstacle,
            colors=self._colors,
            l0=self._l0,
            dt=cfg.dt,
            a_ext=self._a_ext,
            r=cfg.r,
            r_q=cfg.r_q,
            gamma_p=cfg.gamma_p,
            gamma_e=cfg.gamma_e,
            n_iter=cfg.n_iter,
            mass=cfg.mass,
            k_s=cfg.k_s,
            k_c=cfg.k_c,
            gpu_mesh_data=self._gpu_mesh_data,
        )

        self.X = result.X
        self.v = result.v
        self.step_num += 1

        print(f"[SIM] Step {self.step_num} complete: {result.num_detections} detections", flush=True)

        return FrameData(
            step=self.step_num,
            V_cloth=self.X.copy(),
            V_obstacle=self.V_obstacle.copy(),
            v_cloth=self.v.copy(),
            num_contacts=result.num_detections,
        )

    def run(
        self,
        n_steps: int | None = None,
        callback: Callable[[FrameData], None] | None = None,
    ) -> list[FrameData]:
        """
        Run the full simulation loop.

        Parameters
        ----------
        n_steps : int or None
            Number of steps to run. Defaults to config.n_steps.
        callback : callable or None
            Called with each FrameData as it is produced.

        Returns
        -------
        list[FrameData] — all frames including the initial one.
        """
        if not self._loaded:
            raise RuntimeError("Call load() before run().")

        n = n_steps if n_steps is not None else self.config.n_steps
        self._running = True

        frames: list[FrameData] = []

        # Emit initial frame
        initial = FrameData(
            step=0,
            V_cloth=self.X.copy(),
            V_obstacle=self.V_obstacle.copy(),
            v_cloth=self.v.copy(),
        )
        frames.append(initial)
        if callback:
            callback(initial)

        for _ in range(n):
            if self._paused:
                break
            frame = self.step()
            frames.append(frame)
            if callback:
                callback(frame)

        self._running = False
        return frames

    def reset(self) -> FrameData | None:
        """Reset simulation to initial state."""
        if not self._loaded:
            return None
        self.X = self.V_cloth_init.copy()
        self.v = np.tile(
            np.array(self.config.cloth_initial_velocity, dtype=np.float64),
            (len(self.X), 1),
        )
        self.step_num = 0
        self._paused = False
        return FrameData(
            step=0,
            V_cloth=self.X.copy(),
            V_obstacle=self.V_obstacle.copy(),
            v_cloth=self.v.copy(),
        )

    def pause(self) -> None:
        """Pause a running simulation."""
        self._paused = True

    def update_config(self, **kwargs) -> None:
        """
        Update simulation parameters without reloading meshes.

        Only physics parameters can be changed (dt, mass, k_s, k_c, r, r_q,
        gamma_p, gamma_e, n_iter, gravity). Mesh paths are ignored.
        """
        if not self._loaded:
            raise RuntimeError("Call load() before update_config().")

        updatable = {
            "dt", "mass", "k_s", "k_c", "r", "r_q",
            "gamma_p", "gamma_e", "n_iter", "gravity", "n_steps",
        }
        for key, val in kwargs.items():
            if key in updatable and hasattr(self.config, key):
                setattr(self.config, key, val)

        if "gravity" in kwargs:
            self._a_ext = np.array(self.config.gravity, dtype=np.float64)
