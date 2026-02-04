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

from rl_env.obstacles_and_sensing import MapMaker



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
    def __init__(
        self,
        render: bool = False,
        max_steps: int = 1000,
        use_obstacles: bool = False,
        goals=None,
    ):
        """
        Parameters
        ----------
        render : bool
            Whether to launch PyBullet GUI.
        max_steps : int
            Hard truncation limit for an episode.
        use_obstacles : bool
            Whether to spawn obstacles (MapMaker) on first episode.
        goals : list of (float, float)
            List of goal positions; one is sampled each reset().
        """

        # --------------------------
        # Core env bookkeeping
        # --------------------------

        super().__init__()

        self.render_mode = render
        self.max_steps = max_steps
        self.step_count = 0

        # Goals
        self.goals = [np.array(g, dtype=np.float32) for g in (goals or [(1.5, -2.0)])]
        self.goal = self.goals[0]
        print(f"Goal is: {self.goal}")

        # Action space: robot base velocity command (v, omega)
        self.action_space = spaces.Box(
            low=np.array([-0.5, -1.0], dtype=np.float32),
            high=np.array([0.5, 1.0], dtype=np.float32),
        )

        # Obstacles (optional)
        self.use_obstacles = use_obstacles
        self.obstacle_pos = None
        self.obstacle_extents = None

        # Physics simulator (PyBullet + robot/table + impedance + human controller)
        self.sim = AlbertTableImpedanceSim(render=self.render_mode)

        # Reward shaping memory
        self.prev_dist = None

        # Latest URDFenv observation (dict structure returned by URDFenv)
        self._last_env_ob = None

        # Toggle: include human/robot forces in observation?
        self.use_force_observations = True  # <-- YOU CAN SWITCH THIS
        if self.use_force_observations:
            print("OBSERVATION MODE: REAL FORCES INCLUDED")
        else:
            print("OBSERVATION MODE: FORCES ZEROED")

        # Robot footprint used by avoidance logic
        self.robot_radius = 0.5  # set to your base footprint radius

      
        # --------------------------
        obs_dim = 10 
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        

        # --------------------------
        # Obstacle penalties (PyBullet-based, robot-only)
        # --------------------------
        self.obstacle_ids: list[int] = []  # filled when MapMaker spawns obstacles

        # Distance band for smooth proximity penalty (meters)
        self.obs_d_enter = 1.0     # start penalizing inside this distance
        self.obs_d_safe  = 0.2    # near-contact zone (strong penalty region)
        self.obs_query_dist = 3.0  # getClosestPoints query radius (>= obs_d_enter)

        # Penalty magnitudes
        self.obs_contact_penalty = 5.0   # per-step penalty when in contact, used to be 5.
        self.obs_impact_penalty  = 0.0   # one-time penalty when contact starts
        self.obs_prox_k          = 5.0    # proximity penalty scale (quadratic)
        self.obs_approach_k      = 0.0    # penalize moving closer (delta distance)

        # Memory for penalties
        self.prev_contact = False
        self.prev_d_obs = None

        # --------------------------
        # Reward hyperparameters (loggable)
        # --------------------------
        self.kpr = 600.0     # progress reward gain
        self.kmr = 2.0       # motion reward gain
        self.whp = 1.5 #3.5  # heading penalty gain
        self.dist_div = 10.0 # distance penalty divisor

        # Termination / success shaping (loggable)
        self.goal_threshold = 0.4
        self.goal_bonus = 500.0

        # --------------------------
        # Admittance hyperparameters (loggable)
        # --------------------------
        self.adm_gain_lin = 0.5 / 40.0
        self.adm_gain_ang = 1.0 / 40.0
        self.adm_deadzone = 1.0  # N (currently unused in code)
        self.adm_v_clip = 1.0
        self.adm_w_clip = 1.5

        # --------------------------
        # Table obstacle penalties (NEW; same logic as robot)
        # --------------------------
        self.table_obs_d_enter = self.obs_d_enter
        self.table_obs_d_safe  = self.obs_d_safe
        self.table_obs_query_dist = self.obs_query_dist

        self.table_obs_contact_penalty =  self.obs_contact_penalty
        self.table_obs_impact_penalty  = self.obs_impact_penalty
        self.table_obs_prox_k          = self.obs_prox_k

        self.prev_contact_table = False
        self.prev_d_obs_table = None



        
    def get_robot_yaw_wf(self):
        """
        Returns the Robot's Yaw in World Frame, corrected for the URDF offset.
        
        Raw URDF: X points to Left Wheel (90 deg offset)
        Corrected: X points to Robot Nose (0 deg offset)
        """
        if self._last_env_ob is None:
            return 0.0
            
        base_state = self._last_env_ob[0]["robot_0"]["joint_state"]
        raw_yaw = base_state["position"][2]
        
        # Apply the -90 degree (-0.5 * pi) correction
        return raw_yaw - (0.5 * np.pi)
    

    def wrap_angle_bidirectional(self, ang: float) -> float:
        """
        Fold an angle so that directions separated by pi are treated as equivalent.
        Output is in [-pi/2, +pi/2].

        This makes bearing relative to a bidirectional axis (a line), not a directed heading.
        """
        a = wrap_angle(ang)  # -> [-pi, pi]
        if a > 0.5 * np.pi:
            a -= np.pi
        elif a < -0.5 * np.pi:
            a += np.pi
        return float(a)
    

    def _motion_dir_angle(self) -> float:
        v = float(self.last_action_executed[0]) if hasattr(self, "last_action_executed") else 0.0
        yaw = float(self.get_robot_yaw_wf())
        return wrap_angle(yaw + (np.pi if v < 0.0 else 0.0))
    

    # --------------------------------------------------------------------
    #                         OBSERVATION BUFFER
    # ---------------------------------------------------------------------
    def _get_obs(self):
        table_xy, table_yaw, tv_xy, _ = self.sim.get_table_state_world()
        goal_xy = self.goal[:2]
        dx_t, dy_t = goal_xy - table_xy

        base_state = self._last_env_ob[0]["robot_0"]["joint_state"]
        robot_yaw = self.get_robot_yaw_wf()
        v_r = base_state["velocity"][0]
        w_r = base_state["velocity"][2]

        _, heading_error = self.compute_robot_heading_error(dx_t, dy_t, robot_yaw)
        hcos, hsin = np.cos(heading_error), np.sin(heading_error)

        if self.use_force_observations:
            Fh_x, Fh_y = self.sim.last_Fh_xy
        else:
            Fh_x = Fh_y = 0.0

        obs_core = np.array([
            dx_t, dy_t,
            tv_xy[0], tv_xy[1],
            v_r, w_r,
            hcos, hsin,
            Fh_x, Fh_y,
        ], dtype=np.float32)
        return obs_core

    def _obstacle_in_contact(self) -> bool:
        """
        True if the ROBOT base is in contact with any obstacle.
        """
        if (not self.use_obstacles) or (len(self.obstacle_ids) == 0):
            return False

        for obs_id in self.obstacle_ids:
            if p.getContactPoints(bodyA=self.sim.albert_id, bodyB=obs_id):
                return True

        return False

    def _min_obstacle_distance(self) -> float:
        """
        Returns minimum separation distance (meters) between ROBOT base and any obstacle.
        If nothing is within query distance, returns +inf.
        """
        if (not self.use_obstacles) or (len(self.obstacle_ids) == 0):
            return float("inf")

        d_query = float(max(self.obs_query_dist, self.obs_d_enter + 0.5))
        min_d = float("inf")

        for obs_id in self.obstacle_ids:
            pts = p.getClosestPoints(bodyA=self.sim.albert_id, bodyB=obs_id, distance=d_query)
            for cp in pts:
                # cp[8] is distance
                min_d = min(min_d, float(cp[8]))

        return min_d
   
    
    # ---------------------------------------------------------------------
    #                            REWARD
    # ---------------------------------------------------------------------

    def compute_reward(self, dist, prev_dist, heading_error_bi):
        """
        Computes the shaping reward based on:
            - table progress
            - table velocity toward goal
            - distance penalty
            - heading penalty

        Logic is 1:1 identical to your monolithic code.
        """

        # progress reward (positive if distance shrinks)
        kpr = self.kpr # used to be 6.0 but with division by 0.01
        progress_reward = kpr * (prev_dist - dist) 

        # motion reward (velocity toward goal)
        table_xy, _, tv_xy, _ = self.sim.get_table_state_world()
        goal_xy = self.goal[:2]

        dir_to_goal = goal_xy - table_xy

        # normalize reward vector so it works for the speed toward goal + efficiency reward
        dir_to_goal /= np.linalg.norm(dir_to_goal) + 1e-6

        speed_toward_goal = np.dot(tv_xy, dir_to_goal) # in this reward
        kmr = self.kmr # used to be 2.0
        motion_reward = kmr * speed_toward_goal

        # distance penalty for not reaching goal
        distance_penalty = - dist / self.dist_div

        # heading penalty squared
        whp = self.whp # used to be 3.5
        heading_penalty = - (whp * heading_error_bi) ** 2

        # ---------------------------------------------------------
        # GOAL-POWER REWARD  
        # ---------------------------------------------------------
        table_xy, _, tv_xy, _ = self.sim.get_table_state_world()
        goal_xy = self.goal[:2]


        base_state = self._last_env_ob[0]["robot_0"]["joint_state"]
        
        # USE THE HELPER (Standardize Yaw to World Frame)
        robot_pos = np.array(base_state["position"][:2])
        obstacle_penalty_robot, is_crash = self.compute_obstacle_penalty_robot()
        obstacle_penalty_table, is_crash = self.compute_obstacle_penalty_table()

        obstacle_penalty = (obstacle_penalty_table + obstacle_penalty_robot) 
        #obstacle_penalty = obstacle_penalty_robot



        # final reward
        total_reward = (
            progress_reward +
            motion_reward +
            distance_penalty +
            heading_penalty + obstacle_penalty
        )

        # if self.step_count % 10 == 0:
        #     print(f"total: {total_reward} | progress: {progress_reward} | motion: {motion_reward} | distance: {distance_penalty} | heading:{heading_penalty} |  obstacle: {obstacle_penalty} |bypass: {bypass_reward}")


        

        return total_reward, progress_reward, motion_reward, distance_penalty, heading_penalty, obstacle_penalty, is_crash
    




    def compute_obstacle_penalty_robot(self) -> tuple[float, bool]:
        """
        Robot-only obstacle penalty using PyBullet:
        1) per-step penalty if robot is in contact with obstacle
        2) one-time impact penalty when contact starts
        3) smooth proximity penalty when robot is close
        4) approach penalty when robot moves closer (within the proximity band)

        Requires:
        - self.obstacle_ids populated in reset() when obstacles are spawned.
        - self.obs_* parameters set in __init__.

        Returns
        -------
        obstacle_penalty : float
            Negative (or zero) penalty.
        is_crash : bool
            Currently always False (do not terminate on contact for now).
        """
        obstacle_penalty = 0.0
        is_crash = False  # keep False; do not terminate on contact for now

        # No obstacles -> no penalty; also reset memory to avoid stale deltas
        if (not self.use_obstacles) or (len(getattr(self, "obstacle_ids", [])) == 0):
            # print("no obstacles")
            self.prev_contact = False
            self.prev_d_obs = None
            return 0.0, False

        # --------------------------
        # (1) Contact / impact penalty (robot-only)
        # --------------------------
        contact_now = False
        for obs_id in self.obstacle_ids:
            if p.getContactPoints(bodyA=self.sim.albert_id, bodyB=obs_id):
                contact_now = True
                break

        if contact_now:
            print("contact now, hit obstacle")
            obstacle_penalty -= float(self.obs_contact_penalty)

        # one-time impact penalty on contact start
        if contact_now and (not getattr(self, "prev_contact", False)):
            obstacle_penalty -= float(self.obs_impact_penalty)

        self.prev_contact = contact_now

        # --------------------------
        # (2) Smooth proximity + approach penalty (robot-only)
        # --------------------------
        d_query = float(max(self.obs_query_dist, self.obs_d_enter + 0.5))
        d_obs = float("inf")

        for obs_id in self.obstacle_ids:
            pts = p.getClosestPoints(bodyA=self.sim.albert_id, bodyB=obs_id, distance=d_query)
            for cp in pts:
                d_obs = min(d_obs, float(cp[8]))  # cp[8] is distance

        if np.isfinite(d_obs):
            d_enter = float(self.obs_d_enter)
            d_safe = float(self.obs_d_safe)

            # proximity shaping inside the band
            if d_obs < d_enter:
                x = (d_enter - d_obs) / max(d_enter - d_safe, 1e-6)
                x = float(np.clip(x, 0.0, 1.0))
                obstacle_penalty -= float(self.obs_prox_k) * (x ** 2)

            # # approach penalty (only when within band)
            # prev_d = getattr(self, "prev_d_obs", None)
            # if (prev_d is not None) and (d_obs < d_enter):
            #     dd = d_obs - float(prev_d)  # <0 means moving closer
            #     obstacle_penalty -= float(self.obs_approach_k) * max(0.0, -dd)

            self.prev_d_obs = d_obs
        else:
            # far away: reset approach memory to avoid big deltas later
            self.prev_d_obs = None


        #print("obstacle penalty", obstacle_penalty)
        return float(obstacle_penalty), bool(is_crash)
    

    def compute_obstacle_penalty_table(self) -> tuple[float, bool]:
        """
        Table-only obstacle penalty using PyBullet:
        1) per-step penalty if table is in contact with obstacle
        2) one-time impact penalty when contact starts
        3) smooth proximity penalty when table is close

        Returns
        -------
        obstacle_penalty : float
            Negative (or zero) penalty.
        is_crash : bool
            Currently always False (do not terminate on contact for now).
        """
        obstacle_penalty = 0.0
        is_crash = False

        if (not self.use_obstacles) or (len(getattr(self, "obstacle_ids", [])) == 0):
            self.prev_contact_table = False
            self.prev_d_obs_table = None
            return 0.0, False

        table_id = getattr(self.sim, "table_id", None)
        if table_id is None:
            # If the table isn't loaded yet, avoid penalizing and avoid stale memory.
            self.prev_contact_table = False
            self.prev_d_obs_table = None
            return 0.0, False

        # --------------------------
        # (1) Contact / impact penalty (table-only)
        # --------------------------
        contact_now = False
        for obs_id in self.obstacle_ids:
            if p.getContactPoints(bodyA=table_id, bodyB=obs_id):
                contact_now = True
                break

        if contact_now:
            print("contact now, hit obstacle")
            obstacle_penalty -= float(self.table_obs_contact_penalty)

        if contact_now and (not getattr(self, "prev_contact_table", False)):
            obstacle_penalty -= float(self.table_obs_impact_penalty)

        self.prev_contact_table = contact_now

        # --------------------------
        # (2) Smooth proximity penalty (table-only)
        # --------------------------
        d_enter = float(self.table_obs_d_enter)
        d_safe  = float(self.table_obs_d_safe)
        d_query = float(max(self.table_obs_query_dist, d_enter + 0.5))

        d_obs = float("inf")
        for obs_id in self.obstacle_ids:
            pts = p.getClosestPoints(bodyA=table_id, bodyB=obs_id, distance=d_query)
            for cp in pts:
                d_obs = min(d_obs, float(cp[8]))

        if np.isfinite(d_obs) and (d_obs < d_enter):
            x = (d_enter - d_obs) / max(d_enter - d_safe, 1e-6)
            x = float(np.clip(x, 0.0, 1.0))
            obstacle_penalty -= float(self.table_obs_prox_k) * (x ** 2)

        return float(obstacle_penalty), bool(is_crash)




    def apply_human_feedback(self, action, Fh_world_xy):
        """
        Modulates the SAC agent's velocity command based on Human Force.
        
        Args:
            action: (v_sac, w_sac) from the RL agent.
            Fh_world_xy: np.array([Fx, Fy]) human force in World Frame.
            
        Returns:
            np.array([v_new, w_new])
        """
        # --- CONFIGURATION (Tune these!) ---
        # 0.0 = Robot ignores human (Infinite mass)
        # 0.05 = Robot is heavy but responsive
        # 0.20 = Robot is very light

        GAIN_LIN = float(self.adm_gain_lin)
        GAIN_ANG = float(self.adm_gain_ang)
        DEADZONE = float(self.adm_deadzone)


        v_sac, w_sac = action

        # 1. Check if force is significant
        # f_mag = np.linalg.norm(Fh_world_xy)
        # if f_mag < DEADZONE:
        #     return action # No change

        # 2. Get Robot Heading
        robot_yaw = self.get_robot_yaw_wf()
        
        # 3. Rotate World Force into Robot Body Frame
        # Rotation Matrix R^T (World -> Body)
        # F_body_x =  F_wx * cos(th) + F_wy * sin(th)
        # F_body_y = -F_wx * sin(th) + F_wy * cos(th)
        
        c, s = np.cos(robot_yaw), np.sin(robot_yaw)
        
        F_long =  Fh_world_xy[0] * c + Fh_world_xy[1] * s
        F_lat  = -Fh_world_xy[0] * s + Fh_world_xy[1] * c
        

        # set to zero for now: 

        # v_sac = 0.0
        # w_sac = 0.0
        # 4. Apply damper Law
        # Longitudinal force adds to linear velocity
        v_adm = v_sac + (GAIN_LIN * F_long)
        

        
        # Heuristic: If human pushes "Right", robot turns "Right".
        # If F_lat is negative (Right), we want w negative.
        w_adm = w_sac - (GAIN_ANG * F_lat)

        # 5. Safety Clipping (Optional but recommended)
        v_adm = np.clip(v_adm, -float(self.adm_v_clip), float(self.adm_v_clip))
        w_adm = np.clip(w_adm, -float(self.adm_w_clip), float(self.adm_w_clip))


        return np.array([v_adm, w_adm])

    # ---------------------------------------------------------------------
    #                               STEP
    # ---------------------------------------------------------------------
    def step(self, action, Fh_override=None):
        """
        Single RL step.

        Order:
        1) Read robot base pose from URDFenv
        2) sim.impedance_step() (robot arm ↔ table physics)
        3) Human force (external override or rule-based)
        4) URDFenv step(action) (robot base movement)
        5) Compute reward + progress-stall bookkeeping
        6) Termination/truncation
        7) Return obs, reward, terminated, truncated, info
        """

        # -------------------------------------------------------
        # 1) ROBOT BASE STATE (URDFenv output)
        # -------------------------------------------------------
        base_state = self._last_env_ob[0]["robot_0"]["joint_state"]
        robot_yaw = self.get_robot_yaw_wf()
        robot_xy = np.array(base_state["position"][:2])

        # -------------------------------------------------------
        # 2) ROBOT IMPEDANCE STEP
        # -------------------------------------------------------
        self.sim.impedance_step(self.goal, robot_xy)
        self.sim.hold_table_hover(z_target=0.10)


        # -------------------------------------------------------
        # 3) HUMAN INTERACTION
        # -------------------------------------------------------
        if self._external_force_active and (Fh_override is not None):
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
            if self.sim.human_controller is not None:
                Fh_xy, human_action = self.sim.human_controller.step(self.goal)
                self.sim.last_Fh_xy = Fh_xy
                self.sim.human_action = human_action
            else:
                self.sim.last_Fh_xy = np.zeros(2)
                self.sim.human_action = "none"

        # -------------------------------------------------------
        # 3.5) ADMITTANCE (Human feedback modulation)
        # -------------------------------------------------------
        Fh_current = self.sim.last_Fh_xy
        action_admitted = self.apply_human_feedback(action, Fh_current)
        # if self.step_count % 50 ==0: 
        #     print("human force", Fh_current)
        #     print("action_admitted", action_admitted)
        v_final, w_final = action_admitted

        #w_final = 0.1

        self.last_action_executed = np.array([v_final, w_final], dtype=np.float32)
        self.last_action_raw = np.array(action, dtype=np.float32)

        # -------------------------------------------------------
        # 4) APPLY ROBOT BASE VELOCITY COMMANDS (URDFenv)
        # -------------------------------------------------------
        full = np.zeros(self.sim.env.n(), dtype=float)
        full[0], full[1] = v_final, w_final

        ob = self.sim.env.step(full)
        self._last_env_ob = ob

        self.sim.enforce_rigid_arm()

        # Build obs once and return it
        obs = self._get_obs()        

        # -------------------------------------------------------
        # 5) GOAL DISTANCE + PROGRESS-STALL BOOKKEEPING + REWARD
        # -------------------------------------------------------
        table_xy, table_yaw, _, _ = self.sim.get_table_state_world()
        goal_xy = self.goal[:2]
        dx_t, dy_t = goal_xy - table_xy
        dist = float(np.linalg.norm([dx_t, dy_t]))


        heading_error_bi, heading_error = self.compute_robot_heading_error(dx_t, dy_t, robot_yaw)



        reward, progress_reward, motion_reward, distance_penalty, heading_penalty, obstacle_penalty, is_crash = (
            self.compute_reward(dist, self.prev_dist, heading_error_bi)
        )

        # Update prev_dist AFTER computing reward
        self.prev_dist = dist

        # -------------------------------------------------------
        # 6) TERMINATION & TRUNCATION
        # -------------------------------------------------------
        terminated = dist < float(self.goal_threshold)
        if terminated:
            print(f"[CHECK] Reached goal at step: {self.step_count}")
            reward += float(self.goal_bonus)


        if is_crash:
            print(f"[COLLISION] Hit obstacle at step: {self.step_count}")
            terminated = True

        truncated = self.step_count >= self.max_steps
        self.step_count += 1

        if self.render_mode:
            time.sleep(self.sim.dt)

        # -------------------------------------------------------
        # 7) INFO
        # -------------------------------------------------------
        info = {
            "dist_table_to_goal": dist,
            "heading_error": float(heading_error),
            "heading_error_bi": float(heading_error_bi),
            "total_reward": reward,
            "progress_reward": progress_reward,
            "motion_reward": motion_reward,
            "distance_penalty": distance_penalty,
            "obstacle_penalty": obstacle_penalty,
            "heading_penalty": heading_penalty,
            "human_action": self.sim.human_action,
        }




        


        return obs, reward, terminated, truncated, info





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

            # 6) Save observation (list-format)
            self._last_env_ob = ob0 

            # ---------------------------------------------------
            # !!! CONDITIONAL OBSTACLE SPAWNING !!!
            # ---------------------------------------------------
            if self.use_obstacles:
                mm = MapMaker()
                mm.create_map_1()  # Spawns the wall
                self.obstacle_ids = list(mm.obstacles)  # <-- THIS LINE

                if len(mm.obstacles) > 0:
                    obs_id = mm.obstacles[-1]
                    pos, _ = p.getBasePositionAndOrientation(obs_id)
                    self.obstacle_pos = np.array(pos[:2], dtype=np.float32)
                    

                    aabb_min, aabb_max = p.getAABB(obs_id)
                    size = np.array(aabb_max) - np.array(aabb_min)
                    self.obstacle_extents = (size[:2] / 2.0).astype(np.float32)
                    print(f"!!! OBSTACLE SPAWNED at {self.obstacle_pos} !!!")
            else:
                print("!!! NO OBSTACLES (Standard Mode) !!!")
                self.obstacle_pos = None
                self.obstacle_extents = None

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
                [0.0, 0.0, 0.1],
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
            self.sim.reset_table() 

            # -----------------------------
            # Fresh URDFenv observation
            # -----------------------------
            raw = self.sim.env._get_ob()
            self._last_env_ob = [raw]

        # ===================================================
        # COMMON SETTLE STEPS (Fixes Rest Length)
        # ===================================================
        # We step physics BEFORE creating the connection.
        # We lock XY but allow Z (falling) to ensure it hits the floor perfectly.

        zero = np.zeros(self.sim.env.n(), dtype=float)
        for _ in range(100):
            self.sim.hold_table_hover(z_target=0.10)
            self.sim.enforce_rigid_arm()
            self._last_env_ob = self.sim.env.step(zero)


        # ===================================================
        # 🔗 CREATE CONNECTION (Measured on settled bodies)
        # ===================================================
        # Now that velocity is 0 and objects are on the floor, measure.

        p.resetBaseVelocity(self.sim.albert_id, [0,0,0], [0,0,0])
        p.resetBaseVelocity(self.sim.table_id, [0,0,0], [0,0,0])
        
        self.sim.create_connection_impedance()

        # Update observation based on new settled position
        zero = np.zeros(self.sim.env.n())
        self._last_env_ob = self.sim.env.step(zero)

        # Initialize diagnostics
        table_xy, _, _, _ = self.sim.get_table_state_world()
        self.prev_dist = float(np.linalg.norm(self.goal - table_xy))
        self.last_action_executed = np.array([0.0, 0.0], dtype=np.float32)

        obs = self._get_obs()
        return obs, {}


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
        
        # Since robot_yaw is now the nose angle (aligned with Forward velocity),
        # we target the angle directly without the 90 degree offset adjustment.
        angle_to_goal_adj = angle_to_goal_world 

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

    