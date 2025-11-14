#!/usr/bin/env python3
"""
Impedance simulator for Albert + Table.

This class bundles all physics interactions between:
 - the robot arm end effector (EE),
 - the table's robot handle,
 - PyBullet physics,
 - and (optionally) the rule-based human model.

!!! IMPORTANT ARCHITECTURE LINKAGE
-----------------------------------
This simulator is **owned by the RL environment** (AlbertTableEnv), which:

    - calls create_environment()                  to build world + robot
    - sets albert_id via get_albert_body_id()
    - calls load_table()                          to spawn the table + discover handles
    - calls impedance_step() each timestep        for robot actions
    - calls human_controller.step() through sim   for simulated partner forces

The simulator does NOT:
 - know anything about RL
 - know anything about reward
 - know anything about resets

It purely handles physics + kinematics.
"""

from __future__ import annotations
import os
import time
import numpy as np
import pybullet as p

# gymnasium import kept only for type-checking
from gymnasium import spaces

# !!! Cross-file link: human controller is imported here
from controllers.rule_based_human_controller import RuleBasedHumanController


# ---------------------------------------------------------------------------
#                          SMALL UTILITY FUNCTION
# ---------------------------------------------------------------------------

def wrap_angle(a: float) -> float:
    """
    Wrap angle to (-π, π].

    Used internally to keep robot/table yaw values stable.
    This is NOT changed from original script.
    """
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
#               MAIN CLASS: AlbertTableImpedanceSim
# ---------------------------------------------------------------------------

class AlbertTableImpedanceSim:
    """
    Provides:
        - robot spawning
        - table loading & handle discovery
        - impedance interactions (EE <-> table)
        - optional human forces (via RuleBasedHumanController)
        - helper methods for joint state extraction

    !!! Owned by:
        - AlbertTableEnv (stored as self.sim)
    """

    def __init__(self, render: bool = True):
        # Rendering / physics loop settings
        self.render = render
        self.dt = 0.01   # simulation timestep

        # Arm DOF mapping (matches original urdf)
        self.arm_joint_indices = [7, 8, 9, 10, 11, 12, 13]
        self.ee_idx = 16   # end-effector link index

        # Impedance model parameters (identical to original)
        self.Kp = np.diag([300.0, 300.0, 0.0])   # translational stiffness
        self.Dp = np.diag([60.0, 60.0, 0.0])     # translational damping
        self.F_max = np.array([70.0, 70.0, 70.0]) # translational saturation
        self.tau_max = 25.0                      # max joint torque

        # Simulation entity handles (set later)
        self.env = None
        self.albert_id = None
        self.table_id = None

        self.goal_link_idx = None          # robot-side handle on table
        self.human_goal_link_idx = None    # human-side handle on table

        # Diagnostics
        self.last_F_xy = np.zeros(2)
        self.last_dx_xy = np.zeros(2)
        self.last_Fh_xy = np.zeros(2)      # force applied by human

        # Optional human controller
        self.human_controller: RuleBasedHumanController | None = None

        self.arm_q_rigid = None


    # -----------------------------------------------------------------------
    #         Extract joint state arrays for jacobian / torque calc
    # -----------------------------------------------------------------------

    def get_movable_joint_state_arrays(self, body_id: int):
        """
        Return:
            joint_indices: list[int]
            q_list: list[joint positions]
            dq_list: list[joint velocities]

        Only NON-FIXED joints are included.

        !!! Used by:
            - impedance_step() when computing Jᵀ * F
        """
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
    #                    INITIAL ENV + ROBOT CONSTRUCTION
    # -----------------------------------------------------------------------

    def create_environment(self):
        """
        Build URDFenv environment and spawn the mobile robot.

        !!! Called by:
            - AlbertTableEnv.reset()
        """
        # Deferred import maintains modularity
        from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
        from urdfenvs.urdf_common.urdf_env import UrdfEnv

        robot = GenericDiffDriveRobot(
            urdf="albert.urdf",
            mode="vel",
            actuated_wheels=["wheel_left_joint", "wheel_right_joint"],
            castor_wheels=["rotacastor_left_joint", "rotacastor_right_joint"],
            wheel_radius=0.08,
            wheel_distance=0.494,
            spawn_rotation=0,
            facing_direction='-y',
        )

        # !!! Here the global URDF environment is created
        self.env = UrdfEnv(dt=self.dt, robots=[robot], render=self.render)

        

        # Reset and receive initial robot base state
        ob0 = self.env.reset(pos=np.zeros(10))

        p.setGravity(0, 0, -9.81)
        return ob0


    # -----------------------------------------------------------------------
    #                        DISCOVER ALBERT BODY-ID
    # -----------------------------------------------------------------------

    def get_albert_body_id(self):
        """
        Loop through PyBullet bodies and find one whose name contains 'albert'.

        !!! Called by:
            - AlbertTableEnv.reset()
        """
        for i in range(p.getNumBodies()):
            name = p.getBodyInfo(i)[1].decode("utf-8").lower()
            if "albert" in name:
                return i

        raise RuntimeError("Could not find Albert in simulation!")


    # -----------------------------------------------------------------------
    #                            LOAD TABLE
    # -----------------------------------------------------------------------

    def load_table(self):
        """
        Load the table URDF and identify:
            - robot-end handle (goal_link_idx)
            - human-end handle (human_goal_link_idx)

        If the human handle exists, the rule-based human is constructed.

        !!! Called by:
            - AlbertTableEnv.reset()
        """
        table_path = os.path.expanduser(
            "~/catkin_ws/src/mobile_sim/assets/table/table.urdf"
        )

        if not os.path.exists(table_path):
            raise FileNotFoundError(f"Table URDF not found at {table_path}")

        # Align orientation so robot starts behind table
        table_orn = p.getQuaternionFromEuler([0, 0, -np.pi / 2])

        # Spawn table
        self.table_id = p.loadURDF(
            table_path,
            basePosition=[0.0, -1.15, 0.8],
            baseOrientation=table_orn,
            useFixedBase=False,
        )

        # Disable wheel joints (table rolls passively)
        for j in range(p.getNumJoints(self.table_id)):
            name = p.getJointInfo(self.table_id, j)[1].decode().lower()
            if "wheel" in name or "caster" in name:
                p.setJointMotorControl2(
                    self.table_id, j,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=0,
                    force=0
                )

        # Reset handle indices
        self.goal_link_idx = None
        self.human_goal_link_idx = None

        # Scan links to find the robot and human handle names
        for j in range(p.getNumJoints(self.table_id)):
            link_name = p.getJointInfo(self.table_id, j)[12].decode().lower()

            if "table_robot_end" in link_name:
                self.goal_link_idx = j

            if "table_human_end" in link_name:
                self.human_goal_link_idx = j

        if self.goal_link_idx is None:
            raise RuntimeError("Could not find table_robot_end link!")

        # !!! Create simulated human controller if the handle is present
        if self.human_goal_link_idx is not None:
            self.human_controller = RuleBasedHumanController(
                table_id_getter=lambda: self.table_id,
                human_link_idx_getter=lambda: self.human_goal_link_idx,
                get_table_state_func=self.get_table_state_world,
            )

        # Allow table to settle
        for _ in range(100):
            p.stepSimulation()
            time.sleep(0.01)


    # -----------------------------------------------------------------------
    #              ROBOT ARM SETUP (INITIAL POSE, PASSIVE MODE)
    # -----------------------------------------------------------------------

    def set_arm_initial_pose(self):
        """
        Reset robot arm to a suitable starting configuration.
        Matches original monolithic logic 1:1.
        """
        q_current = np.array([
            p.getJointState(self.albert_id, j)[0]
            for j in self.arm_joint_indices
        ])

        # Relative offset from current pose
        offset_deg = [0, 90, 0, 25, 0, 40, 38]
        q_target = q_current + np.deg2rad(offset_deg)

        # Directly set joint positions
        for j, q in zip(self.arm_joint_indices, q_target):
            p.resetJointState(self.albert_id, j, q)

        self.arm_q_rigid = q_target.copy()

    def disable_arm_motors(self):
        """
        Disable velocity controllers on robot arm so impedance torques
        computed in impedance_step() directly drive the joints.
        """
        for j in self.arm_joint_indices:
            p.setJointMotorControl2(
                self.albert_id, j,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=0.0,
                force=0.0
            )


    # -----------------------------------------------------------------------
    #                        READ TABLE WORLD STATE
    # -----------------------------------------------------------------------

    def get_table_state_world(self):
        """
        Returns:
            xy          : np.array([x, y])
            yaw         : table yaw (adjusted by +90 degrees to align forward)
            vxy         : table linear velocity in world frame
            wz          : angular velocity around z-axis

        !!! Called by:
            - rule_based_human_controller
            - RL observation assembly
        """
        pos, orn = p.getBasePositionAndOrientation(self.table_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.table_id)

        xy = np.array(pos[:2], dtype=np.float32)

        # Yaw conversion matches original script
        yaw_raw = p.getEulerFromQuaternion(orn)[2]
        yaw = wrap_angle(yaw_raw + np.pi / 2)

        vxy = np.array(lin_vel[:2], dtype=np.float32)
        wz = float(ang_vel[2])

        return xy, yaw, vxy, wz


    # -----------------------------------------------------------------------
    #                     THE HEART OF THE SIMULATION
    # -----------------------------------------------------------------------

    def impedance_step(self, goal_xy, robot_xy):
        """
        Perform one impedance step between:
            - robot end-effector (EE)
            - table handle link

        Computes:
            - EE <-> handle translational interaction force (Fr)
            - EE <-> handle yaw torque
            - Jᵀ wrench projection into arm joint torques

        !!! Called by:
            - AlbertTableEnv.step() (EVERY PHYSICS STEP)
        """

        # ------------------------------------------------------------------
        # Get end-effector state
        # ------------------------------------------------------------------
        ee_state = p.getLinkState(self.albert_id, self.ee_idx, computeLinkVelocity=1)
        ee_pos  = np.array(ee_state[0])   # EE position
        ee_vel  = np.array(ee_state[6])   # EE linear velocity
        ee_quat = ee_state[1]             # EE orientation

        # ------------------------------------------------------------------
        # Get table handle (robot-side) state
        # ------------------------------------------------------------------
        handle_state = p.getLinkState(self.table_id, self.goal_link_idx, computeLinkVelocity=1)
        handle_pos  = np.array(handle_state[0])
        handle_vel  = np.array(handle_state[6])
        handle_quat = handle_state[1]

        # ================================================================
        # 1) TRANSLATIONAL IMPEDANCE
        # ================================================================
        dx = handle_pos - ee_pos            # position error
        dv = handle_vel - ee_vel           # velocity error

        # Raw force before clipping
        Fr = -(self.Kp @ dx + self.Dp @ dv)

        # Saturation
        Fr = np.clip(Fr, -self.F_max, self.F_max)

        # Apply translational force to TABLE (world frame)
        p.applyExternalForce(
            self.table_id,
            self.goal_link_idx,
            Fr.tolist(),
            handle_pos.tolist(),
            flags=p.WORLD_FRAME,
        )

        # ================================================================
        # 2) ROTATIONAL IMPEDANCE (YAW ONLY)
        # ================================================================
        h_yaw  = p.getEulerFromQuaternion(handle_quat)[2]
        ee_yaw = p.getEulerFromQuaternion(ee_quat)[2]

        yaw_error = ((h_yaw - ee_yaw + np.pi) % (2 * np.pi)) - np.pi

        # Angular velocities
        h_ang_vel = p.getLinkState(self.table_id, self.goal_link_idx, computeLinkVelocity=1)[7][2]
        ee_ang_vel = p.getLinkState(self.albert_id, self.ee_idx, computeLinkVelocity=1)[7][2]
        yaw_vel_error = h_ang_vel - ee_ang_vel

        # Gains match original script
        K_yaw = 60.0
        D_yaw = 4.0
        tau_z = -K_yaw * yaw_error - D_yaw * yaw_vel_error

        # Apply torque to table
        p.applyExternalTorque(
            self.table_id,
            self.goal_link_idx,
            [0, 0, tau_z],
            flags=p.WORLD_FRAME,
        )

        # ================================================================
        # 3) BACK-PROJECTION: table force/torque -> robot arm joints
        # ================================================================
        # # Compute robot joint Jacobians
        # joint_indices, q_list, dq_list = self.get_movable_joint_state_arrays(self.albert_id)

        # J_lin, J_ang = p.calculateJacobian(
        #     self.albert_id,
        #     self.ee_idx,
        #     [0, 0, 0],
        #     q_list,
        #     dq_list,
        #     [0.0] * len(q_list),
        # )

        # # Restrict to arm joints
        # J_arm_lin = np.array(J_lin)[:, self.arm_joint_indices]
        # J_arm_ang = np.array(J_ang)[:, self.arm_joint_indices]

        # # Compute joint torques
        # tau_lin = J_arm_lin.T @ (-Fr)
        # tau_rot = J_arm_ang.T @ np.array([0, 0, -tau_z])
        # tau_total = np.clip(tau_lin + tau_rot, -self.tau_max, self.tau_max)

        # # Apply torques to robot arm
        # for j, t in zip(self.arm_joint_indices, tau_total):
        #     p.setJointMotorControl2(
        #         self.albert_id,
        #         j,
        #         controlMode=p.TORQUE_CONTROL,
        #         force=float(t),
        #     )

        # Diagnostics returned to environment
        self.last_F_xy = Fr[:2]
        self.last_dx_xy = dx[:2]

        return Fr[:2], dx[:2], handle_pos[:2], handle_vel[:2]
    
    def reset_table(self):
        start_pos = [0.0, -1.15, 0.5]
        start_orn = p.getQuaternionFromEuler([0, 0, -np.pi/2])

        p.resetBasePositionAndOrientation(self.table_id, start_pos, start_orn)
        p.resetBaseVelocity(self.table_id, [0,0,0], [0,0,0])




    def get_joint_id_by_name(self, body_id, joint_name):
        for j in range(p.getNumJoints(body_id)):
            info = p.getJointInfo(body_id, j)
            name = info[1].decode("utf-8")
            if name == joint_name:
                return j
        raise ValueError(f"Joint '{joint_name}' not found in body {body_id}")
    

    def enforce_rigid_arm(self):
        """
        Force the arm joints to stay at arm_q_rigid by hard-resetting their
        state every time this is called. This makes the arm purely kinematic.
        """
        if self.arm_q_rigid is None:
            return

        for j, q in zip(self.arm_joint_indices, self.arm_q_rigid):
            # velocity=0 so nothing accumulates
            p.resetJointState(self.albert_id, j, float(q), 0.0)



