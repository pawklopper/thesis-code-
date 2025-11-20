#!/usr/bin/env python3
"""
AlbertTableEnv

Gymnasium environment wrapping the entire:
 - PyBullet simulation,
 - Albert robot control,
 - Table physics,
 - Impedance model (via AlbertTableImpedanceSim),
 - Human rule-based controller.

!!! ARCHITECTURE OVERVIEW
-----------------------------------
AlbertTableEnv owns **one instance** of AlbertTableImpedanceSim (self.sim).

The data flow per step is:

    RL-agent  --->  env.step(action)
                     |
                     |---> sim.impedance_step()               (robot ↔ table physics)
                     |---> sim.human_controller.step()        (simulated partner)
                     |---> sim.env.step(robot_velocities)     (URDFenv base movement)

Then the environment:
   - extracts new PyBullet states from sim,
   - computes reward,
   - assembles observation vector,
   - determines termination/truncation,
   - returns (obs, reward, terminated, truncated, info)

Nothing about robot physics happens inside this file. It delegates
all physical interactions to the simulator.
"""

from __future__ import annotations
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import pybullet as p

# !!! CROSS-FILE IMPORT
#     Imports the entire physics + human module
from controllers.impedance_sim import AlbertTableImpedanceSim, wrap_angle


class AlbertTableEnv(gym.Env):
    """
    Simplified Gymnasium environment focusing on table motion + robot/table heading alignment.

    Only responsible for:
        - action interface,
        - observation construction,
        - reward logic,
        - episode termination/reset.

    !!! Important:
        All physics happen in self.sim (AlbertTableImpedanceSim).
    """

    metadata = {"render_modes": ["human"]}


    # ---------------------------------------------------------------------
    #                           CONSTRUCTOR
    # ---------------------------------------------------------------------
    def __init__(self, render=False, max_steps=1000, goals=None):
        """
        Parameters
        ----------
        render : bool
            Whether to launch PyBullet GUI.
        max_steps : int
            Hard truncation limit for an episode.
        goals : list of (float, float)
            List of goal positions; one is sampled each reset().
        """

        super().__init__()

        self.render_mode = render
        self.max_steps = max_steps
        self.step_count = 0

        # goals are stored as np arrays; first one is used initially
        self.goals = [np.array(g, dtype=np.float32) for g in (goals or [(1.5, -2.0)])]
        self.goal = self.goals[0]
        print(f"Goal is: {self.goal}")

        # Action = robot velocity command: (v, omega)
        # These bounds match your original script
        self.action_space = spaces.Box(
            low=np.array([-0.5, -1.0], dtype=np.float32),
            high=np.array([0.5, 1.0], dtype=np.float32)
        )

        # Observation = 12 elements (layout documented below)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )

        # !!! CROSS-FILE CREATION
        #     Create the full impedance physics engine
        self.sim = AlbertTableImpedanceSim(render=self.render_mode)

        # For reward shaping: remember last goal distance
        self.prev_dist = None

        # Latest URDFenv observation (dict structure returned by URDFenv)
        self._last_env_ob = None

        # Toggle: include human/robot forces in observation?
        # True  → real forces (Fh, Fr)
        # False → zeros in observations
        self.use_force_observations = True   # <-- YOU CAN SWITCH THIS


        if self.use_force_observations:
            print("OBSERVATION MODE: REAL FORCES INCLUDED")
        else:
            print("OBSERVATION MODE: FORCES ZEROED")


        




    # ---------------------------------------------------------------------
    #                         OBSERVATION BUFFER
    # ---------------------------------------------------------------------
    def _get_obs(self):
        """
        Constructs the 12D observation vector.

        !!! IMPORTANT:
            All inside-state (EE, table, robot) is read from self.sim,
            because AlbertTableEnv does NOT interface directly with PyBullet.

        Contents:
            0: dx_table_to_goal
            1: dy_table_to_goal
            2: table_vx
            3: table_vy
            4: v_robot
            5: w_robot
            6: cos(heading_error)
            7: sin(heading_error)
            8: Fh_x     (disabled)
            9: Fh_y     (disabled)
           10: Fr_x     (disabled)
           11: Fr_y     (disabled)
        """

        # ---------------------------------------
        # TABLE STATE (from sim)
        # ---------------------------------------
        table_xy, table_yaw, tv_xy, _ = self.sim.get_table_state_world()
        goal_xy = self.goal[:2]

        dx_t, dy_t = goal_xy - table_xy   # position to goal in world frame

        # ---------------------------------------
        # ROBOT BASE STATE (from URDFenv)
        # ---------------------------------------
        # self._last_env_ob is an URDFenv output dict
        base_state = self._last_env_ob[0]["robot_0"]["joint_state"]

        robot_pos = np.array(base_state["position"][:2])
        robot_yaw = base_state["position"][2]

        v_r = base_state["velocity"][0]
        w_r = base_state["velocity"][2]

        # ---------------------------------------
        # HEADING ERROR TABLE ↔ ROBOT
        # ---------------------------------------
        yaw_diff = wrap_angle(table_yaw - robot_yaw)

        # Heading error relative to goal direction
        _, heading_error = self.compute_robot_heading_error(dx_t, dy_t, robot_yaw)
        hcos, hsin = np.cos(heading_error), np.sin(heading_error)

        # ---------------------------------------
        # Forces disabled in your original file
        # ---------------------------------------
        # Human & robot forces
        if self.use_force_observations:
            Fh_x, Fh_y = self.sim.last_Fh_xy
            Fr_x, Fr_y = self.sim.last_F_xy
        else:
            Fh_x = Fh_y = 0.0
            Fr_x = Fr_y = 0.0

        return np.array([
            dx_t, dy_t, # table 
            tv_xy[0], tv_xy[1], # table
            v_r, w_r, # robot
            hcos, hsin, # robot
            Fh_x, Fh_y, Fr_x, Fr_y # forces human and robot
        ], dtype=np.float32)


    # ---------------------------------------------------------------------
    #                            REWARD
    # ---------------------------------------------------------------------
    def compute_reward(self, dist, prev_dist, heading_error_bi, heading_diff):
        """
        Computes the shaping reward based on:
            - table progress
            - table velocity toward goal
            - distance penalty
            - heading penalty

        Logic is 1:1 identical to your monolithic code.
        """

        # progress reward (positive if distance shrinks)
        kpr = 6.0
        progress_reward = kpr * (prev_dist - dist) / 0.01

        # motion reward (velocity toward goal)
        table_xy, _, tv_xy, _ = self.sim.get_table_state_world()
        goal_xy = self.goal[:2]

        dir_to_goal = goal_xy - table_xy
        dir_to_goal /= np.linalg.norm(dir_to_goal) + 1e-6

        speed_toward_goal = np.dot(tv_xy, dir_to_goal)
        kmr = 2.0
        motion_reward = kmr * speed_toward_goal

        # distance penalty for not reaching goal
        distance_penalty = - dist / 10

        # heading penalty squared
        whp = 3.5
        heading_penalty = - (whp * heading_error_bi) ** 2

        # ---------------------------------------------------------
        # GOAL-POWER REWARD  
        # ---------------------------------------------------------
        table_xy, _, tv_xy, _ = self.sim.get_table_state_world()
        goal_xy = self.goal[:2]
        Fh_xy = self.sim.last_Fh_xy
        Fr_xy = self.sim.last_F_xy

        efficiency_reward = self.compute_efficiency_reward(dir_to_goal, tv_xy, Fh_xy, Fr_xy)

        # final reward
        total_reward = (
            progress_reward +
            motion_reward +
            distance_penalty +
            heading_penalty + efficiency_reward
        )
        if self.step_count % 50 == 0: 
            print(f"progress reward {progress_reward}, motion reward {motion_reward}, distance_penalty: {distance_penalty}, heading_penalty: {heading_penalty}, efficiency_reward: {efficiency_reward}")

        return total_reward, progress_reward, distance_penalty, heading_penalty
    


    def compute_efficiency_reward(self, dir_to_goal, tv_xy, Fh_xy, Fr_xy):
            """
            Cooperative Efficiency Reward (Normal Mode)
            
            Measures the ratio of 'Useful Force' to 'Total Effort'.
            
            Returns:
                float: Value roughly between -1.0 and 1.0
                1.0  = Perfect Alignment (Synergy)
                0.0  = Wasted Energy (Orthogonal or Cancellation)
                -1.0 = Counter-Productive (Sabotage)
            """
        

            # 2. Calculate Vectors
            # Team Vector (Resultant)
            F_team = Fr_xy + Fh_xy
            
            # Useful Force (Scalar projection onto goal line)
            # How much of our combined force is actually helping?
            F_useful = np.dot(F_team, dir_to_goal)

            # 3. Calculate Cost
            # Sum of magnitudes (The "fuel cost" of the team)
            # We use magnitudes separately so that fighting forces sum up to high cost
            cost = np.linalg.norm(Fr_xy) + np.linalg.norm(Fh_xy) + 1e-6

            # 4. The Efficiency Ratio
            efficiency = F_useful / cost
            
            # Tuning Gain
            k_collab = 1.0
            
            return k_collab * efficiency
    
    def apply_directional_leash(self, action):
        """
        Simplified Leash:
        Only checks if the commanded linear velocity (v) would increase distance.
        Does not perform complex vector projection.
        """
        
        # 1. Get Spring Vector
        dx_xy = self.sim.last_dx_xy
        spring_len = float(np.linalg.norm(dx_xy))
        
        MAX_STRETCH = 0.25 # used to be 0.25
        
        # If we are in the safe zone, do nothing
        if spring_len <= MAX_STRETCH:
            return action

        # ----------------------------------------------------------
        # 2. Determine alignment (Am I facing the table?)
        # ----------------------------------------------------------
        robot_yaw = self._last_env_ob[0]["robot_0"]["joint_state"]["position"][2]
        
        # Robot Forward Vector (using your coordinate system: 0 is -y)
        robot_forward = np.array([np.sin(robot_yaw), -np.cos(robot_yaw)])
        
        # Vector pointing FROM Robot TO Table
        vec_to_table = dx_xy 
        
        # Dot product tells us alignment:
        # > 0 : Table is mostly IN FRONT (roughly +/- 90 deg)
        # < 0 : Table is mostly BEHIND (roughly +/- 90 deg)
        alignment = float(np.dot(robot_forward, vec_to_table))
        
        # ----------------------------------------------------------
        # 3. Gate the Velocity Command (v)
        # ----------------------------------------------------------
        v_cmd = action[0]
        
        # CASE A: Table is BEHIND me (alignment < 0) AND I try to drive FORWARD (v > 0)
        # -> This moves me away. STOP.
        if alignment < 0 and v_cmd > 0:
            #print("Leash: Stop (Table behind, driving forward)")
            action[0] = 0.0
            
        # CASE B: Table is IN FRONT (alignment > 0) AND I try to drive BACKWARD (v < 0)
        # -> This moves me away. STOP.
        elif alignment > 0 and v_cmd < 0:
            #print("Leash: Stop (Table in front, driving backward)")
            action[0] = 0.0
            
        return action



    # ---------------------------------------------------------------------
    #                               STEP
    # ---------------------------------------------------------------------
    def step(self, action, Fh_override=None):
        """
        Single RL step.

        !!! ORDER OF OPERATIONS (IDENTICAL TO ORIGINAL)
        -------------------------------------------------------
        1) Read robot base pose from URDFenv
        2) sim.impedance_step()                    (robot arm ↔ table physics)
        3) HUMAN FORCE:
            - if Fh_override is provided → external human (PS5)
            - else → rule-based human controller
        4) URDFenv step(action)                    (robot base movement)
        5) get table state -> compute reward
        6) check termination/truncation
        7) build and return observation
        """

        # -------------------------------------------------------
        # 1) ROBOT BASE STATE (URDFenv output)
        # -------------------------------------------------------
        base_state = self._last_env_ob[0]["robot_0"]["joint_state"]

        robot_yaw = base_state["position"][2]
        robot_xy = np.array(base_state["position"][:2])

        # -------------------------------------------------------
        # 2) ROBOT IMPEDANCE STEP
        # -------------------------------------------------------
        # Robot arm impedance applies forces against the table.
        self.sim.impedance_step(self.goal, robot_xy)

        # -------------------------------------------------------
        # 3) HUMAN INTERACTION: RULE-BASED OR EXTERNAL OVERRIDE
        # -------------------------------------------------------

        # 3) HUMAN INTERACTION: EXTERNAL OVERRIDE OR RULE-BASED

        if self._external_force_active and (Fh_override is not None):
            # external human force from PS5
            handle_pos = p.getLinkState(self.sim.table_id, self.sim.human_goal_link_idx)[0]

            p.applyExternalForce(
                self.sim.table_id,
                self.sim.human_goal_link_idx,
                [float(Fh_override[0]), float(Fh_override[1]), float(Fh_override[2])],
                handle_pos,
                flags=p.WORLD_FRAME,
            )

            self.sim.last_Fh_xy = np.array(Fh_override[:2])
            self.sim.human_action = "external_force"

        else:
            # rule-based human controller
            if self.sim.human_controller is not None:
                Fh_xy, human_action = self.sim.human_controller.step(self.goal)
                self.sim.last_Fh_xy = Fh_xy
                self.sim.human_action = human_action
            else:
                self.sim.last_Fh_xy = np.zeros(2)
                self.sim.human_action = "none"


        
        # -------------------------------------------------------
        # 3.5) Directional Leash Gating (prevents runaway robot)
        # -------------------------------------------------------
        action = self.apply_directional_leash(action)


        # -------------------------------------------------------
        # 4) APPLY ROBOT BASE VELOCITY COMMANDS (URDFenv)
        # -------------------------------------------------------
        full = np.zeros(self.sim.env.n(), dtype=float)
        full[0], full[1] = action              # forward velocity and yaw rate

        ob = self.sim.env.step(full)           # physics step for mobile base
        self._last_env_ob = ob

        # force the arm back to its rigid pose after physics integration
        self.sim.enforce_rigid_arm()

        # -------------------------------------------------------
        # 5) GOAL DISTANCE + REWARD
        # -------------------------------------------------------
        table_xy, table_yaw, _, _ = self.sim.get_table_state_world()
        goal_xy = self.goal[:2]
        dx_t, dy_t = goal_xy - table_xy

        dist = float(np.linalg.norm([dx_t, dy_t]))

        heading_error_bi, heading_error = self.compute_robot_heading_error(
            dx_t, dy_t, robot_yaw
        )
        heading_diff = wrap_angle(table_yaw - robot_yaw)

        reward, progress_reward, distance_penalty, heading_penalty = (
            self.compute_reward(dist, self.prev_dist, heading_error_bi, heading_diff)
        )

        self.prev_dist = dist

        # -------------------------------------------------------
        # 6) TERMINATION & TRUNCATION CONDITIONS
        # -------------------------------------------------------
        terminated = dist < 0.4
        if terminated:
            print(f"Reached goal at step: {self.step_count}")
            reward += 50

        truncated = self.step_count >= self.max_steps
        self.step_count += 1

        if self.render_mode:
            time.sleep(self.sim.dt)

        # -------------------------------------------------------
        # 7) INFO DICTIONARY
        # -------------------------------------------------------
        info = {
            "dist_table_to_goal": dist,
            "heading_error": float(heading_error),
            "heading_error_bi": float(heading_error_bi),
            "progress_reward": progress_reward,
            "distance_penalty": distance_penalty,
            "heading_penalty": heading_penalty,
            "human_action": self.sim.human_action,
        }

        # debug wheels: 
        if self.step_count % 10 == 0:
            self.sim.debug_wheel_forces(action[0])

        return self._get_obs(), reward, terminated, truncated, info



    # ---------------------------------------------------------------------
    #                                RESET
    # ---------------------------------------------------------------------
    def reset(self, *, seed=None, options=None, Fh_override=None):
        super().reset(seed=seed)
        self._external_force_active = Fh_override is not None
        self.step_count = 0

        # ---------------------------------------------------
        # 0) Sample goal
        # ---------------------------------------------------
        self.goal = self.goals[np.random.randint(len(self.goals))]
        print(f"Goal is: {self.goal}")

        # ===================================================
        # FIRST-EVER EPISODE
        # ===================================================
        if self.sim.env is None:
            print("=== FIRST EPISODE: full initialization ===")

            # 1) Build URDFenv (spawns robot)
            ob0 = self.sim.create_environment()

            # 2) Find robot ID
            self.sim.albert_id = self.sim.get_albert_body_id()

            # 3) Load the table ONCE
            self.sim.load_table()

            # 4) URDFenv settle BEFORE touching arm
            zero = np.zeros(self.sim.env.n())
            self.sim.env.step(zero)

            # 5) Set arm pose (correct frames)
            self.sim.set_arm_initial_pose()
            #self.sim.disable_arm_motors()

            # 6) Save observation (list-format)
            self._last_env_ob = ob0 

        # ===================================================
        # SUBSEQUENT EPISODES
        # ===================================================
        else:
            print("=== LATER EPISODE: reset positions ===")

            # -----------------------------
            # Reset robot base
            # -----------------------------
            
            p.resetBasePositionAndOrientation(
                self.sim.albert_id,
                [0.0, 0.0, 0.2],
                p.getQuaternionFromEuler([0,0,0])
            )
            p.resetBaseVelocity(self.sim.albert_id, [0,0,0],[0,0,0])


            # Reset wheels
            for wheel_joint in ["wheel_left_joint", "wheel_right_joint"]:
                jid = self.sim.get_joint_id_by_name(self.sim.albert_id, wheel_joint)
                p.resetJointState(self.sim.albert_id, jid, 0.0, 0.0)


          
            

            # -----------------------------
            # Reset table base
            # -----------------------------
            self.sim.reset_table()   # SAFE: table_id guaranteed to exist
         



            # # -----------------------------
            # # Resolve any collisions
            # # -----------------------------
            # for _ in range(5):
            #     p.stepSimulation()

            # -----------------------------
            # Fresh URDFenv observation
            # -----------------------------
            raw = self.sim.env._get_ob()
            self._last_env_ob = [raw]

        # ===================================================
        # COMMON SETTLE STEPS FOR BOTH CASES
        # ===================================================

        # Impedance settle step
        base_state = self._last_env_ob[0]["robot_0"]["joint_state"]
        robot_xy = np.array(base_state["position"][:2])
        self.sim.impedance_step(self.goal, robot_xy)

        # URDFenv settle
        zero = np.zeros(self.sim.env.n())
        self._last_env_ob = self.sim.env.step(zero)

        # Initialize previous distance
        table_xy, _, _, _ = self.sim.get_table_state_world()
        self.prev_dist = float(np.linalg.norm(self.goal - table_xy))

        # Allow everything to settle 
        print("allow everything to settle")
        for _ in range(100):
            p.stepSimulation()
            time.sleep(0.01)

        return self._get_obs(), {}





    # ---------------------------------------------------------------------
    #                             HELPERS
    # ---------------------------------------------------------------------
    def compute_robot_heading_error(self, dx, dy, robot_yaw):
        """
        Converts dx, dy into a world-frame angle,
        converts robot yaw into same reference,
        computes smallest signed angular error.

        heading_error_bi = bi-directional heading error (0 to π/2)
        heading_error = signed heading error

        Used for reward shaping only.
        """
        angle_to_goal_world = np.arctan2(dy, dx)
        angle_to_goal_adj = angle_to_goal_world - np.pi / 2.0

        heading_error = np.arctan2(
            np.sin(angle_to_goal_adj - robot_yaw),
            np.cos(angle_to_goal_adj - robot_yaw)
        )

        err_abs = abs(heading_error)
        heading_error_bi = min(err_abs, np.pi - err_abs)

        return heading_error_bi, heading_error

    def close(self):
        """Close URDFenv"""
        self.sim.env.close()
        print("Environment closed.")


    def _debug_arm(self):
        joints = [p.getJointState(self.sim.albert_id, j)[0] for j in self.sim.arm_joint_indices]

        print(f"\n==== arm pose at step: {self.step_count} ====")
        print("Joint angles:", np.round(joints, 3))



