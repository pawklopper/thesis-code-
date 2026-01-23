#!/usr/bin/env python3
"""
Impedance simulator for Albert + Table.

This class bundles all physics interactions between:
 - the robot arm end effector (EE),
 - the table's robot handle,
 - PyBullet physics,
 - and (optionally) the rule-based human model.
"""

from __future__ import annotations
import os
import time
import numpy as np
import pybullet as p
from controllers.rule_based_human_controller import RuleBasedHumanController


def wrap_angle(a: float) -> float:
    """Wrap angle to (-π, π]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class AlbertTableImpedanceSim:
    """
    Manages physics interactions and the Rigid Tow Bar impedance connection.
    """

    def __init__(self, render: bool = True):
        # -------------------------------------------------------------------
        # 1. SIMULATION SETTINGS
        # -------------------------------------------------------------------
        self.render = render
        self.dt = 0.01   # simulation timestep

        # -------------------------------------------------------------------
        # 2. ROBOT CONFIG
        # -------------------------------------------------------------------
        self.arm_joint_indices = [7, 8, 9, 10, 11, 12, 13]
        self.ee_idx = 16   # end-effector link index
        self.tau_max = 25.0

        # -------------------------------------------------------------------
        # 3. IMPEDANCE TUNING (Centralized)
        # -------------------------------------------------------------------
        # Stiffness (Kp): How hard it pulls to correct position errors (N/m)
        self.imp_stiffness = 10000.0 # used to be 300
        
        # Damping (Kd): How much it resists velocity differences (N/(m/s))
        self.imp_damping = 2 * np.sqrt(20 * self.imp_stiffness) # used to be 60

        self.imp_max_force = 1000.0 # used to be 40


        # -------------------------------------------------------------------
        # 4. STATE VARIABLES
        # -------------------------------------------------------------------
        self.env = None
        self.albert_id = None
        self.table_id = None

        self.goal_link_idx = None          # robot-side handle
        self.human_goal_link_idx = None    # human-side handle

        # Diagnostics
        self.last_F_xy = np.zeros(2)
        self.last_dx_xy = np.zeros(2)
        self.last_Fh_xy = np.zeros(2)      

        # Human controller
        self.human_controller: RuleBasedHumanController | None = None
        self.arm_q_rigid = None

        # The vector for the rigid connection (Robot Local Frame)
        # Created in create_connection_impedance()
        self.rest_vector_local = None 


        # -------------------------------------------------------------------
        # 3b. ROTATIONAL (YAW) IMPEDANCE TUNING
        # -------------------------------------------------------------------
        self.rot_stiffness = 300.0   # N*m/rad  (start small; increase carefully)
        self.rot_damping   = 2.0 * np.sqrt(self.rot_stiffness)  # N*m/(rad/s)
        self.rot_max_torque = 60.0   # N*m (saturation)
        self.yaw_rest_offset = None  # stored at connection creation



    # -----------------------------------------------------------------------
    #                    INITIAL ENV + ROBOT CONSTRUCTION
    # -----------------------------------------------------------------------

    def create_environment(self):
        """Build URDFenv environment and spawn the mobile robot."""
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

        self.env = UrdfEnv(dt=self.dt, robots=[robot], render=self.render)

        if self.render:
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

        ob0 = self.env.reset(pos=np.zeros(10))
        p.setGravity(0, 0, -9.81)
        return ob0

    def get_albert_body_id(self):
        """Find robot ID."""
        for i in range(p.getNumBodies()):
            name = p.getBodyInfo(i)[1].decode("utf-8").lower()
            if "albert" in name:
                return i
        raise RuntimeError("Could not find Albert in simulation!")

    # -----------------------------------------------------------------------
    #                            LOAD TABLE
    # -----------------------------------------------------------------------

    def load_table(self):
        """Load the table URDF and identify handles."""
        table_path = os.path.expanduser("~/catkin_ws/src/mobile_sim/assets/table/table.urdf")

        if not os.path.exists(table_path):
            raise FileNotFoundError(f"Table URDF not found at {table_path}")

        table_orn = p.getQuaternionFromEuler([0, 0, -np.pi / 2])

        self.table_id = p.loadURDF(
            table_path,
            basePosition=[0.0, -1.15, 0.1],
            baseOrientation=table_orn,
            useFixedBase=False,
        )

        # Disable wheel joints
        for j in range(p.getNumJoints(self.table_id)):
            name = p.getJointInfo(self.table_id, j)[1].decode().lower()
            if "wheel" in name or "caster" in name:
                p.setJointMotorControl2(
                    self.table_id, j,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=0,
                    force=0
                )

        self.goal_link_idx = None
        self.human_goal_link_idx = None

        for j in range(p.getNumJoints(self.table_id)):
            link_name = p.getJointInfo(self.table_id, j)[12].decode().lower()
            if "table_robot_end" in link_name:
                self.goal_link_idx = j
            if "table_human_end" in link_name:
                self.human_goal_link_idx = j

        if self.goal_link_idx is None:
            raise RuntimeError("Could not find table_robot_end link!")

        if self.human_goal_link_idx is not None:
            self.human_controller = RuleBasedHumanController(
                table_id_getter=lambda: self.table_id,
                human_link_idx_getter=lambda: self.human_goal_link_idx,
                get_table_state_func=self.get_table_state_world,
            )


    # -----------------------------------------------------------------------
    #              ROBOT ARM SETUP
    # -----------------------------------------------------------------------

    def set_arm_initial_pose(self):
        """Reset robot arm to a suitable starting configuration."""
        q_current = np.array([
            p.getJointState(self.albert_id, j)[0]
            for j in self.arm_joint_indices
        ])
        offset_deg = [0, 90, 0, 25, 0, 40, 38]
        q_target = q_current + np.deg2rad(offset_deg)

        for j, q in zip(self.arm_joint_indices, q_target):
            p.resetJointState(self.albert_id, j, q)

        self.arm_q_rigid = q_target.copy()

    def disable_arm_motors(self):
        """Disable velocity controllers on robot arm."""
        for j in self.arm_joint_indices:
            p.setJointMotorControl2(
                self.albert_id, j,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=0.0,
                force=0.0
            )

    def enforce_rigid_arm(self):
        """Force the arm joints to stay at arm_q_rigid."""
        if self.arm_q_rigid is None:
            return
        for j, q in zip(self.arm_joint_indices, self.arm_q_rigid):
            p.resetJointState(self.albert_id, j, float(q), 0.0)

    # -----------------------------------------------------------------------
    #                        READ STATES
    # -----------------------------------------------------------------------

    def get_table_state_world(self):
        """Returns: xy, yaw, vxy, wz"""
        pos, orn = p.getBasePositionAndOrientation(self.table_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.table_id)

        xy = np.array(pos[:2], dtype=np.float32)
        yaw_raw = p.getEulerFromQuaternion(orn)[2]
        yaw = wrap_angle(yaw_raw + np.pi / 2)

        vxy = np.array(lin_vel[:2], dtype=np.float32)
        wz = float(ang_vel[2])

        return xy, yaw, vxy, wz

    def get_joint_id_by_name(self, body_id, joint_name):
        for j in range(p.getNumJoints(body_id)):
            info = p.getJointInfo(body_id, j)
            name = info[1].decode("utf-8")
            if name == joint_name:
                return j
        raise ValueError(f"Joint '{joint_name}' not found in body {body_id}")

    def reset_table(self):
        start_pos = [0.0, -1.15, 0.2]
        start_orn = p.getQuaternionFromEuler([0, 0, -np.pi/2])
        p.resetBasePositionAndOrientation(self.table_id, start_pos, start_orn)
        p.resetBaseVelocity(self.table_id, [0,0,0], [0,0,0])

    # -----------------------------------------------------------------------
    #              RIGID TOW BAR IMPEDANCE LOGIC
    # -----------------------------------------------------------------------

    def create_connection_impedance(self):
        """
        Setup: Measures the initial 'Rest Length' (local_y) of the spring between the
        Robot EE and the Table robot-handle, expressed in the Robot's base frame.

        This version is instrumented to show EXACTLY what is changing episode-to-episode:
        - robot/table base pose (xy, yaw)
        - EE and handle world poses
        - world_vec (handle - EE)
        - local_x/local_y projection (the stored spring rest length)
        - base linear/angular velocities at measurement time

        It also stores a snapshot in `self._last_connection_debug` for later inspection.
        """

        if self.albert_id is None or self.table_id is None:
            return

        # Clean up old constraint (if you ever add one later)
        if hasattr(self, "cid") and self.cid is not None:
            p.removeConstraint(self.cid)
            self.cid = None

        # ---------------------------
        # Read world states (poses)
        # ---------------------------
        ee_pos = np.array(p.getLinkState(self.albert_id, self.ee_idx)[0], dtype=float)
        h_pos  = np.array(p.getLinkState(self.table_id, self.goal_link_idx)[0], dtype=float)

        rpos, rorn = p.getBasePositionAndOrientation(self.albert_id)
        tpos, torn = p.getBasePositionAndOrientation(self.table_id)

        robot_yaw = float(p.getEulerFromQuaternion(rorn)[2])
        table_yaw = float(p.getEulerFromQuaternion(torn)[2])

        # velocities (often the smoking gun)
        r_lin, r_ang = p.getBaseVelocity(self.albert_id)
        t_lin, t_ang = p.getBaseVelocity(self.table_id)

        # ---------------------------
        # Compute projected "rest length" in robot frame
        # ---------------------------
        world_vec = (h_pos - ee_pos)  # (dx, dy, dz) in world frame

        c, s = np.cos(robot_yaw), np.sin(robot_yaw)

        # Robot-local coordinates of the handle relative to EE
        local_x =  c * world_vec[0] + s * world_vec[1]   # lateral
        local_y = -s * world_vec[0] + c * world_vec[1]   # longitudinal (your "spring length")

        # Store rest length
        self.spring_rest_length = float(local_y)

        # ---------------------------
        # Store rest yaw offset (table relative to robot)
        # ---------------------------
        self.yaw_rest_offset = wrap_angle(table_yaw - robot_yaw)

        #print(f"spring zero length: {self.spring_rest_length:3f} m | spring zero rotational: {self.yaw_rest_offset:.3f} rad")


    def impedance_step(self, goal_xy=None, robot_xy=None):
        """
        Physics: Calculates forces based on deviation from Rest Length.
        Uses uniform stiffness (self.imp_stiffness) for both X and Y.
        """
        if not hasattr(self, 'spring_rest_length') or self.spring_rest_length is None:
            return np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2)

        # 1. Get Current State
        ee_state = p.getLinkState(self.albert_id, self.ee_idx, computeLinkVelocity=1)
        h_state = p.getLinkState(self.table_id, self.goal_link_idx, computeLinkVelocity=1)
        
        ee_pos = np.array(ee_state[0])
        ee_vel = np.array(ee_state[6])
        h_pos = np.array(h_state[0])
        h_vel = np.array(h_state[6])
        
        _, robot_orn = p.getBasePositionAndOrientation(self.albert_id)
        robot_yaw = p.getEulerFromQuaternion(robot_orn)[2]

        # 2. Calculate Current Spring Deformation (in Robot Frame)
        world_vec = h_pos - ee_pos
        
        c, s = np.cos(robot_yaw), np.sin(robot_yaw)
        
        # local_x: Lateral deviation (Sideways)
        # local_y: Longitudinal distance (Forward/Back)
        local_x =  c * world_vec[0] + s * world_vec[1]
        local_y = -s * world_vec[0] + c * world_vec[1]
        
        # --- ERROR CALCULATION ---
        # Longitudinal: Difference between Current Length and Rest Length
        y_error = local_y - self.spring_rest_length
        
        # Lateral: Difference between Current X and 0 (Center)
        x_error = local_x - 0.0
        
        # 3. Calculate Local Forces
        rel_vel_world = h_vel - ee_vel
        local_vx =  c * rel_vel_world[0] + s * rel_vel_world[1]
        local_vy = -s * rel_vel_world[0] + c * rel_vel_world[1]

        # Hooke's Law: F = -Kp * x - Kd * v
        # UNIFORM stiffness for both axes as requested
        f_local_x = -(self.imp_stiffness * x_error + self.imp_damping * local_vx)
        f_local_y = -(self.imp_stiffness * y_error + self.imp_damping * local_vy)


        # ==========================================================
        # DEBUG START: AM I PUSHING OR PULLING?
        # ==========================================================
        # Convention based on your coordinate system:
        # local_y is Longitudinal (Forward/Back)
        # f_local_y is the force APPLIED TO THE TABLE.
        
        # If f_local_y is NEGATIVE -> Vector points back to robot -> PULLING (Tension)
        # If f_local_y is POSITIVE -> Vector points away from robot -> PUSHING (Compression)
        
        # action_type = "UNKNOWN"
        # if f_local_y < -10.0:
        #     action_type = f"⬇️ PULLING (Tension) | Force: {f_local_y:.1f} N"
        # elif f_local_y > 10.0:
        #     action_type = f"⬆️ PUSHING (Compression) | Force: {f_local_y:.1f} N"
        # else:
        #     action_type = f"⏺️ NEUTRAL             | Force: {f_local_y:.1f} N"

        # Only print every 20 steps to avoid flooding console
        # (Assuming you add a counter or just print continuously for a short test)
        # print(f"[{action_type}]  Dist Error: {y_error:.3f} m")
        
        # 4. Rotate Force Back to World Frame
        # World Fx = c*Fx - s*Fy
        # World Fy = s*Fx + c*Fy
        f_world_x = c * f_local_x - s * f_local_y
        f_world_y = s * f_local_x + c * f_local_y
        
        force_on_table = np.array([f_world_x, f_world_y, 0.0])

        # 5. Saturation
        force_mag = np.linalg.norm(force_on_table)
        if force_mag > self.imp_max_force:
            force_on_table = force_on_table * (self.imp_max_force / force_mag)


        # --- TABLE YAW + RELATIVE YAW RATE ---
        _, table_orn = p.getBasePositionAndOrientation(self.table_id)
        table_yaw = p.getEulerFromQuaternion(table_orn)[2]

        # angular velocities (world frame); we only use z
        _, robot_ang_vel = p.getBaseVelocity(self.albert_id)
        _, table_ang_vel = p.getBaseVelocity(self.table_id)
        rel_wz = float(table_ang_vel[2] - robot_ang_vel[2])

        # --- YAW ERROR relative to rest offset ---
        if self.yaw_rest_offset is None:
            self.yaw_rest_offset = wrap_angle(table_yaw - robot_yaw)

        yaw_err = wrap_angle((table_yaw - robot_yaw) - self.yaw_rest_offset)

        # torsional spring-damper
        tau_z = -(self.rot_stiffness * yaw_err + self.rot_damping * rel_wz)

        # saturation
        tau_z = float(np.clip(tau_z, -self.rot_max_torque, self.rot_max_torque))


        # 6. Apply Forces
        p.applyExternalForce(self.table_id, self.goal_link_idx, force_on_table, h_pos, p.WORLD_FRAME)
        p.applyExternalForce(self.albert_id, self.ee_idx, -force_on_table, ee_pos, p.WORLD_FRAME)

        # Apply equal and opposite yaw torques (rotational coupling)
        p.applyExternalTorque(self.table_id, -1, [0.0, 0.0, tau_z], flags=p.WORLD_FRAME)
        p.applyExternalTorque(self.albert_id, -1, [0.0, 0.0, -tau_z], flags=p.WORLD_FRAME)


        # 7. Diagnostics
        self.last_F_xy = force_on_table[:2]
        self.last_dx_xy = np.array([x_error, y_error])

        #self.debug_impedance_viz()


        
        
        return self.last_F_xy, self.last_dx_xy, h_pos[:2], h_vel[:2]
    

    def debug_impedance_viz(self):
        """
        Visualizes the Spring logic in PyBullet.
        
        BLUE LINE: The 'Rest Length' reference (Rigidly attached to Robot).
                   Ends at the point where the spring force would be zero.
                   
        RED LINE:  The Deformation (Error).
                   Connects the Zero-Point to the Real Handle.
                   The longer this line, the stronger the force.
        """
        if not hasattr(self, 'spring_rest_length') or self.spring_rest_length is None:
            return

        # 1. Get Robot State
        ee_pos = np.array(p.getLinkState(self.albert_id, self.ee_idx)[0])
        _, robot_orn = p.getBasePositionAndOrientation(self.albert_id)
        robot_yaw = p.getEulerFromQuaternion(robot_orn)[2]
        
        # 2. Calculate 'Ideal' Handle Position (Ghost Point)
        # This is where the handle SHOULD be if the spring was at rest length and centered
        c, s = np.cos(robot_yaw), np.sin(robot_yaw)
        
        # Local Frame: X=0 (Centered), Y=RestLength
        local_x = 0.0
        local_y = self.spring_rest_length
        
        # Rotate to World Frame
        world_x = c * local_x - s * local_y
        world_y = s * local_x + c * local_y
        
        ideal_pos = ee_pos + np.array([world_x, world_y, 0.0])

        # 3. Get Real Handle Position
        h_pos = np.array(p.getLinkState(self.table_id, self.goal_link_idx)[0])

        # -----------------------------------------------------------
        # DRAW LINES
        # -----------------------------------------------------------
        
        # BLUE: The Reference Stick (Robot EE -> Ideal Point)
        # This shows the natural length of the spring
        p.addUserDebugLine(ee_pos, ideal_pos, [0, 0, 1], lineWidth=3, lifeTime=0.1)
        
        # RED: The Spring Deformation (Ideal Point -> Real Handle)
        # This is the visual representation of the Force
        p.addUserDebugLine(ideal_pos, h_pos, [1, 0, 0], lineWidth=4, lifeTime=0.1)


    def hold_table_hover(self, z_target: float = 0.10):
        """
        Hard-hold the table at z_target and keep it flat:
        - roll = 0
        - pitch = 0
        - yaw is preserved
        Also kills vertical velocity and roll/pitch angular velocity.
        """
        pos, orn = p.getBasePositionAndOrientation(self.table_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.table_id)

        # Extract current yaw, discard roll/pitch
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)

        # Force flat orientation (roll=pitch=0), keep yaw
        flat_orn = p.getQuaternionFromEuler([0.0, 0.0, yaw])

        # Clamp Z, keep XY
        new_pos = [pos[0], pos[1], z_target]
        p.resetBasePositionAndOrientation(self.table_id, new_pos, flat_orn)

        # Preserve XY linear velocity and yaw rate; kill vz, wx, wy
        p.resetBaseVelocity(
            self.table_id,
            linearVelocity=[lin_vel[0], lin_vel[1], 0.0],
            angularVelocity=[0.0, 0.0, ang_vel[2]],
        )

        