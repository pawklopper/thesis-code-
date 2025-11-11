#!/usr/bin/env python3
"""
Rule-based human controller for the Albert + Table setup.

This module implements a simple "human partner" for the task.
It decides on *one* behavior per timestep based on:
 - the table’s current translational velocity,
 - the table’s yaw rate,
 - and the direction toward the goal.

The controller then applies a force at the human-side handle of the table.

IMPORTANT ARCHITECTURE NOTE
---------------------------
This class DOES NOT directly know anything about the simulation, robot, or
environment. Instead, it receives three *getter functions* from the
AlbertTableImpedanceSim instance:

    !!! table_id_getter()
        Returns the PyBullet body ID of the table.

    !!! human_link_idx_getter()
        Returns the link index of the table’s *human handle*.

    !!! get_table_state_func()
        Returns (xy, yaw, v_xy, wz) describing the table's world-frame state.

This design decouples the human logic from physics details.
"""

from __future__ import annotations
import numpy as np
import pybullet as p


class RuleBasedHumanController:
    """
    Simple rule-based human partner for the Albert + Table setup.

    The human controller applies a force Fh_xy at the table’s human handle
    using PyBullet's `applyExternalForce`. The rule chosen depends on:

      - table linear velocity (table_vxy)
      - table yaw rate (wz)
      - whether the table is already moving toward or away from the goal

    Parameters
    ----------
    table_id_getter : Callable[[], int | None]
        !!! Provided by AlbertTableImpedanceSim.load_table()
        Function returning the PyBullet body ID of the table.

    human_link_idx_getter : Callable[[], int | None]
        !!! Provided by AlbertTableImpedanceSim.load_table()
        Function returning the link index for the human handle link.

    get_table_state_func : Callable[[], tuple[np.ndarray, float, np.ndarray, float]]
        !!! Provided by AlbertTableImpedanceSim
        Function returning:
            - xy: table position [x, y]
            - yaw: table yaw orientation
            - v_xy: table linear velocity in world frame [vx, vy]
            - wz: table angular velocity around z-axis
    """

    def __init__(self, table_id_getter, human_link_idx_getter, get_table_state_func):
        # Store references to dependency functions.
        # These will be called every step to pull updated simulation data.
        self.get_table_state_world = get_table_state_func     # !!!
        self.get_table_id = table_id_getter                   # !!!
        self.get_human_link_idx = human_link_idx_getter       # !!!

        # Diagnostics: store last chosen force and action name
        self.last_Fh_xy = np.zeros(2, dtype=np.float32)
        self.last_action = "none"

    # ---------------------------------------------------------------------
    #                            MAIN STEP METHOD
    # ---------------------------------------------------------------------
    def step(self, goal_xy: np.ndarray):
        """
        Evaluate and apply a single rule-based human action.

        Parameters
        ----------
        goal_xy : np.ndarray
            The target point the table should move toward. This is passed in
            by the RL environment.

        Returns
        -------
        (np.ndarray, str)
            Fh_xy : 2D human-applied force vector in world frame (Fx, Fy)
            action_name : descriptive string of which rule applied

        !!! Called from:
            AlbertTableEnv.step() on every environment step.
        """

        # --------------------------------------------------
        # Retrieve IDs and handle index through getter functions
        # --------------------------------------------------
        table_id = self.get_table_id()                # !!!
        link_idx = self.get_human_link_idx()          # !!!

        # Safety check: if something isn't set up yet, human is inactive
        if table_id is None or link_idx is None:
            return np.zeros(2, dtype=np.float32), "inactive"

        # --------------------------------------------------
        # Retrieve table state from simulator
        # --------------------------------------------------
        # table_xy : (x,y)
        # table_yaw : yaw orientation
        # table_vxy : linear velocity
        # wz         : yaw rate
        table_xy, table_yaw, table_vxy, wz = self.get_table_state_world()   # !!!

        # Direction toward goal (normalized)
        dir_to_goal = (goal_xy - table_xy)
        dir_to_goal /= np.linalg.norm(dir_to_goal) + 1e-6

        # --------------------------------------------------
        # Default: no force applied
        # --------------------------------------------------
        Fh_xy = np.zeros(2, dtype=np.float32)
        action_name = "none"

        # --------------------------------------------------
        # RULE SET (in descending order of priority)
        # --------------------------------------------------

        # 1) DAMPING — If table is going too fast or spinning too much, slow it down
        if np.linalg.norm(table_vxy) > 0.8 or abs(wz) > 1.0:
            Fh_xy = -10.0 * table_vxy
            action_name = "dampen"

        # 2) TURN LEFT — If table yaw rate is positive, push leftwards
        elif wz > 0.1:
            Fh_xy = np.array([-15.0, 0.0], dtype=np.float32)
            action_name = "turn_left"

        # 3) TURN RIGHT — If table yaw rate is negative, push rightwards
        elif wz < -0.1:
            Fh_xy = np.array([15.0, 0.0], dtype=np.float32)
            action_name = "turn_right"

        # 4) PUSH FORWARD — If progressing toward goal, reinforce
        elif np.dot(table_vxy, dir_to_goal) > 0.02:
            Fh_xy = (40.0 * dir_to_goal).astype(np.float32)
            action_name = "push_forward"

        # 5) PULL BACKWARD — If moving backward relative to goal, oppose it
        elif np.dot(table_vxy, dir_to_goal) < -0.02:
            Fh_xy = (-30.0 * dir_to_goal).astype(np.float32)
            action_name = "pull_backward"

        # --------------------------------------------------
        # APPLY FORCE USING PYBULLET
        # --------------------------------------------------
        # !!! This is where human authority meets the physics engine.
        handle_pos = p.getLinkState(table_id, link_idx)[0]

        p.applyExternalForce(
            objectUniqueId=table_id,
            linkIndex=link_idx,
            forceObj=[float(Fh_xy[0]), float(Fh_xy[1]), 0.0],
            posObj=handle_pos,
            flags=p.WORLD_FRAME,
        )

        # --------------------------------------------------
        # Save diagnostics for visualization, logging, RL features
        # --------------------------------------------------
        self.last_Fh_xy = Fh_xy
        self.last_action = action_name

        return Fh_xy, action_name
