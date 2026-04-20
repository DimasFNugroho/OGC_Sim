"""
Frame data emitted by the simulator each time step.

This is the boundary between physics and visualization — the simulator
produces FrameData, and backends consume it. No rendering code here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class FrameData:
    """
    One frame of simulation output.

    Attributes
    ----------
    step : int
        Time step number (0-indexed).
    V_cloth : np.ndarray
        (N, 3) cloth vertex positions after this step.
    V_obstacle : np.ndarray
        (M, 3) obstacle vertex positions (static, included for convenience).
    v_cloth : np.ndarray
        (N, 3) cloth vertex velocities after this step.
    num_contacts : int
        Number of active contact detections this step.
    num_exceed : int
        Number of vertices that exceeded conservative bounds.
    """

    step: int
    V_cloth: np.ndarray
    V_obstacle: np.ndarray
    v_cloth: np.ndarray
    num_contacts: int = 0
    num_exceed: int = 0
