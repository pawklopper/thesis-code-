#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ps5_human_control.py
--------------------
Modular PS5 joystick interface for controlling the human endpoint
in the Albert + Table impedance simulation.

Can be imported into training, evaluation, or demo scripts.
"""

import numpy as np
import pygame
import pybullet as p


# ============================================================
# =============== Controller Setup and Input =================
# ============================================================

def init_ps5_controller():
    """Initialize pygame joystick interface and return the joystick object."""
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise RuntimeError("❌ No controller detected. Plug in your PS5 controller.")
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"✅ Connected to controller: {js.get_name()}")
    return js


def read_joystick_force(js, force_scale=40.0):
    """
    Read the PS5 left stick and map it to a planar force vector.

    Returns (Fx_local, Fy_local) in table-local coordinates:
        Left/right  = ±X
        Forward/back = ±Y
    """
    pygame.event.pump()
    fx_local = js.get_axis(0) * force_scale      # left/right
    fy_local = -js.get_axis(1) * force_scale     # up/down (invert Y for natural motion)
    return np.array([fx_local, fy_local], dtype=np.float32)


# ============================================================
# =============== Frame Transformation =======================
# ============================================================

def compute_world_force_from_table(f_local, table_id):
    """
    Convert local joystick force to world-frame force
    based on the table's yaw (so that +Y local = forward).

    Parameters
    ----------
    f_local : np.ndarray, shape (2,)
        Local force vector [Fx_local, Fy_local].
    table_id : int
        PyBullet ID of the table.

    Returns
    -------
    f_world : np.ndarray, shape (2,)
        Force vector in world frame.
    """
    _, orn = p.getBasePositionAndOrientation(table_id)
    yaw = p.getEulerFromQuaternion(orn)[2]

    # Table faces -Y initially → +π/2 correction
    yaw_corrected = yaw + np.pi / 2.0

    fx_local, fy_local = f_local
    fx_world = fx_local * np.cos(yaw_corrected) - fy_local * np.sin(yaw_corrected)
    fy_world = fx_local * np.sin(yaw_corrected) + fy_local * np.cos(yaw_corrected)

    return np.array([fx_world, fy_world], dtype=np.float32)


# ============================================================
# =============== Force Application ==========================
# ============================================================

def apply_human_force_to_table(sim, f_world):
    """
    Apply the given world-frame force to the table's human handle.

    Parameters
    ----------
    sim : AlbertTableImpedanceSim
        Simulation object (with .table_id and .human_goal_link_idx).
    f_world : np.ndarray, shape (2,)
        Force vector [Fx, Fy] in world coordinates.
    """
    human_link = sim.human_goal_link_idx
    if human_link is None:
        raise RuntimeError("❌ Human handle link not found in table URDF.")

    hs = p.getLinkState(sim.table_id, human_link)
    human_handle_pos = hs[0]

    p.applyExternalForce(
        sim.table_id, human_link,
        [float(f_world[0]), float(f_world[1]), 0.0],
        human_handle_pos,
        flags=p.WORLD_FRAME,
    )


# ============================================================
# =============== Visualization Helper =======================
# ============================================================

def draw_goal_marker(goal_xy, color=(1, 0, 0, 1), label="Goal"):
    """Draw a sphere + label for the goal position."""
    goal_pos = [float(goal_xy[0]), float(goal_xy[1]), 0.05]
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.07,
        rgbaColor=color,
    )
    marker_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=visual_shape,
        basePosition=goal_pos,
    )
    p.addUserDebugText(
        text=label,
        textPosition=[goal_pos[0], goal_pos[1], 0.15],
        textColorRGB=[1, 1, 1],
        textSize=1.2,
    )
    return marker_id
