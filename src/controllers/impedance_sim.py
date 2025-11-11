#!/usr/bin/env python3
"""
Impedance simulator for Albert + Table.

This is extracted from the original monolithic script.  
No logic is changed. Only imports and structure are adapted for modular use.
"""

from __future__ import annotations
import os
import time
import numpy as np
import pybullet as p

from gymnasium import spaces  # for type use only
from controllers.rule_based_human_controller import RuleBasedHumanController


# ---------------------------------------------------------------------------
# Utility function (moved here from the monolithic file)
# ---------------------------------------------------------------------------

def wrap_angle(a):
    """Wrap angle to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Main class: AlbertTableImpedanceSim
# ---------------------------------------------------------------------------

class AlbertTableImpedanceSim:
    """Encapsulated Albert + Table impedance controller simulation."""

    def __init__(self, render=True):
        self.render = render
        self.dt = 0.01

        self.arm_joint_indices = [7, 8, 9, 10, 11, 12, 13]
        self.ee_idx = 16

        self.Kp = np.diag([300.0, 300.0, 0.0])
        self.Dp = np.diag([60.0, 60.0, 0.0])
        self.F_max = np.array([70.0, 70.0, 70.0])
        self.tau_max = 25.0

        self.env = None
        self.albert_id = None
        self.table_id = None
        self.goal_link_idx = None
        self.human_goal_link_idx = None

        self.last_F_xy = np.zeros(2)
        self.last_dx_xy = np.zeros(2)
        self.last_Fh_xy = np.zeros(2)

        self.human_controller = None

    # -----------------------------------------------------------------------
    # Helper functions from original file
    # -----------------------------------------------------------------------

    def get_movable_joint_state_arrays(self, body_id):
        """Return joint indices, positions (q), and velocities (dq) for non-fixed joints."""
        joint_indices, q_list, dq_list = [], [], []
        for ji in range(p.getNumJoints(body_id)):
            j_info = p.getJointInfo(body_id, ji)
            if j_info[2] != p.JOINT_FIXED:
                joint_indices.append(ji)
                state = p.getJointState(body_id, ji)
                q_list.append(state[0])
                dq_list.append(state[1])
        return joint_indices, q_list, dq_list


    # -----------------------------------------------------------------------
    # Environment and URDF loading
    # -----------------------------------------------------------------------

    def create_environment(self):
        """Create URDF environment and spawn Albert robot."""
        from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
        from urdfenvs.urdf_common.urdf_env import UrdfEnv

        robot = GenericDiffDriveRobot(
            urdf="albert.urdf",
            mode="vel",
            actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
            castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
            wheel_radius=0.08,
            wheel_distance=0.494,
            spawn_rotation=0,
            facing_direction='-y',
        )

        self.env = UrdfEnv(dt=self.dt, robots=[robot], render=self.render)
        ob0 = self.env.reset(pos=np.zeros(10))
        p.setGravity(0, 0, -9.81)
        return ob0


    def get_albert_body_id(self):
        """Find the body ID of Albert in PyBullet."""
        for i in range(p.getNumBodies()):
            name = p.getBodyInfo(i)[1].decode("utf-8").lower()
            if "albert" in name:
                return i
        raise RuntimeError("Could not find Albert in simulation!")


    def load_table(self):
        """Load the table URDF and identify the handle links."""
        table_path = os.path.expanduser("~/catkin_ws/src/mobile_sim/assets/table/table.urdf")
        if not os.path.exists(table_path):
            raise FileNotFoundError(f"Table URDF not found at {table_path}")

        table_orn = p.getQuaternionFromEuler([0, 0, -np.pi / 2])

        self.table_id = p.loadURDF(
            table_path,
            basePosition=[0.0, -1.15, 0.8],
            baseOrientation=table_orn,
            useFixedBase=False,
        )

        for j in range(p.getNumJoints(self.table_id)):
            name = p.getJointInfo(self.table_id, j)[1].decode().lower()
            if "wheel" in name or "caster" in name:
                p.setJointMotorControl2(self.table_id, j,
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocity=0,
                                        force=0)

        # robot handle
        self.goal_link_idx = None
        self.human_goal_link_idx = None

        for j in range(p.getNumJoints(self.table_id)):
            link_name = p.getJointInfo(self.table_id, j)[12].decode().lower()

            if "table_robot_end" in link_name:
                self.goal_link_idx = j

            if "table_human_end" in link_name:
                self.human_goal_link_idx = j

        if self.goal_link_idx is None:
            raise RuntimeError("Could not find table_robot_end link")

        # Create human controller if human handle exists
        if self.human_goal_link_idx is not None:
            self.human_controller = RuleBasedHumanController(
                table_id_getter=lambda: self.table_id,
                human_link_idx_getter=lambda: self.human_goal_link_idx,
                get_table_state_func=self.get_table_state_world,
            )

        for _ in range(100):
            p.stepSimulation()
            time.sleep(0.01)


    # -----------------------------------------------------------------------
    # Arm configuration (unchanged)
    # -----------------------------------------------------------------------

    def set_arm_initial_pose(self):
        q_current = np.array([p.getJointState(self.albert_id, j)[0]
                             for j in self.arm_joint_indices])
        offset_deg = [0, 90, 0, 25, 0, 40, 38]
        q_target = q_current + np.deg2rad(offset_deg)

        for j, q in zip(self.arm_joint_indices, q_target):
            p.resetJointState(self.albert_id, j, q)


    def disable_arm_motors(self):
        for j in self.arm_joint_indices:
            p.setJointMotorControl2(self.albert_id, j,
                                   controlMode=p.VELOCITY_CONTROL,
                                   targetVelocity=0.0,
                                   force=0.0)


    # -----------------------------------------------------------------------
    # Table state extraction
    # -----------------------------------------------------------------------

    def get_table_state_world(self):
        pos, orn = p.getBasePositionAndOrientation(self.table_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.table_id)

        xy = np.array(pos[:2], dtype=np.float32)
        yaw_raw = p.getEulerFromQuaternion(orn)[2]
        yaw = wrap_angle(yaw_raw + np.pi / 2)
        vxy = np.array(lin_vel[:2], dtype=np.float32)
        wz = float(ang_vel[2])
        return xy, yaw, vxy, wz


    # -----------------------------------------------------------------------
    # Impedance step (full original logic pasted)
    # -----------------------------------------------------------------------

    def impedance_step(self, goal_xy, robot_xy):
        """Run one impedance control step with translational + yaw impedance."""

        # --- EE state ---
        ee_state = p.getLinkState(self.albert_id, self.ee_idx, computeLinkVelocity=1)
        ee_pos, ee_vel, ee_quat = (
            np.array(ee_state[0]),
            np.array(ee_state[6]),
            ee_state[1],
        )

        # --- Handle state ---
        handle_state = p.getLinkState(self.table_id, self.goal_link_idx, computeLinkVelocity=1)
        handle_pos, handle_vel, handle_quat = (
            np.array(handle_state[0]),
            np.array(handle_state[6]),
            handle_state[1],
        )

        # Translational impedance
        dx = handle_pos - ee_pos
        dv = handle_vel - ee_vel

        Fr = -(self.Kp @ dx + self.Dp @ dv)
        Fr = np.clip(Fr, -self.F_max, self.F_max)

        p.applyExternalForce(
            self.table_id,
            self.goal_link_idx,
            Fr.tolist(),
            handle_pos.tolist(),
            flags=p.WORLD_FRAME,
        )

        # Rotational impedance
        h_yaw = p.getEulerFromQuaternion(handle_quat)[2]
        ee_yaw = p.getEulerFromQuaternion(ee_quat)[2]

        yaw_error = ((h_yaw - ee_yaw + np.pi) % (2 * np.pi)) - np.pi

        h_ang_vel = p.getLinkState(self.table_id, self.goal_link_idx, computeLinkVelocity=1)[7][2]
        ee_ang_vel = p.getLinkState(self.albert_id, self.ee_idx, computeLinkVelocity=1)[7][2]
        yaw_vel_error = h_ang_vel - ee_ang_vel

        K_yaw = 60.0
        D_yaw = 4.0
        tau_z = -K_yaw * yaw_error - D_yaw * yaw_vel_error

        p.applyExternalTorque(
            self.table_id,
            self.goal_link_idx,
            [0, 0, tau_z],
            flags=p.WORLD_FRAME,
        )

        # Apply equal/opposite torques to robot arm
        joint_indices, q_list, dq_list = self.get_movable_joint_state_arrays(self.albert_id)

        J_lin, J_ang = p.calculateJacobian(
            self.albert_id,
            self.ee_idx,
            [0, 0, 0],
            q_list,
            dq_list,
            [0.0] * len(q_list),
        )

        J_arm_lin = np.array(J_lin)[:, self.arm_joint_indices]
        J_arm_ang = np.array(J_ang)[:, self.arm_joint_indices]

        tau_lin = J_arm_lin.T @ (-Fr)
        tau_rot = J_arm_ang.T @ np.array([0, 0, -tau_z])

        tau_total = np.clip(tau_lin + tau_rot, -self.tau_max, self.tau_max)

        for j, t in zip(self.arm_joint_indices, tau_total):
            p.setJointMotorControl2(
                self.albert_id,
                j,
                controlMode=p.TORQUE_CONTROL,
                force=float(t),
            )

        # Save last forces
        self.last_F_xy = Fr[:2]
        self.last_dx_xy = dx[:2]

        return Fr[:2], dx[:2], handle_pos[:2], handle_vel[:2]
