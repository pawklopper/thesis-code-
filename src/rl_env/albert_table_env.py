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
from rl_env.obstacles_and_sensing import LidarMixin



class AlbertTableEnv(LidarMixin, gym.Env):
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
        max_steps: int = 1200,
        use_obstacles: bool = False,
        goals=None,
        # --------------------------
        # LiDAR configuration (forwarded to LidarMixin)
        # --------------------------
        use_lidar: bool = True,
        lidar_num_rays: int = 36,
        lidar_range: float = 3.0,
        lidar_height: float = 0.25,
        lidar_front_half_angle: float = np.deg2rad(60),
        lidar_max_iters: int = 4,
        lidar_eps_adv: float = 1e-3,
        lidar_debug: bool = False,
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

        LiDAR params are forwarded to LidarMixin.__init__().
        """

        # IMPORTANT: cooperative init so LidarMixin (and gym.Env) initialize properly
        super().__init__(
            use_lidar=use_lidar,
            lidar_num_rays=lidar_num_rays,
            lidar_range=lidar_range,
            lidar_height=lidar_height,
            lidar_front_half_angle=lidar_front_half_angle,
            lidar_max_iters=lidar_max_iters,
            lidar_eps_adv=lidar_eps_adv,
            lidar_debug=lidar_debug,
        )

        # --------------------------
        # Core env bookkeeping
        # --------------------------
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
        # Reward / avoidance tuning (non-LiDAR-specific params)
        # --------------------------
        self.lidar_block_dist = 2.0
        self.lidar_clear_margin = 0.20
        self.lidar_escape_k = 4.0
        self.escape_v_min = 0.05

        self.heading_cone_margin = np.deg2rad(10.0)
        self.heading_escape_k = 2.0
        self.heading_escape_scale = np.deg2rad(15.0)

        # --------------------------
        # Observation space
        # --------------------------
        obs_dim = 10 + (int(self.lidar_num_rays) if getattr(self, "use_lidar", False) else 0)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)


        
        
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
    

    def _min_dist_in_cone(self, center_angle, half_angle):
        if self._last_lidar is None:
            return float(self.lidar_range) - float(self.robot_radius)

        dists = self._last_lidar.astype(np.float32) * float(self.lidar_range)
        clear = dists - float(self.robot_radius)

        delta = np.arctan2(np.sin(self.lidar_angles - center_angle),
                        np.cos(self.lidar_angles - center_angle))
        mask = np.abs(delta) <= float(half_angle)
        if not np.any(mask):
            return float(self.lidar_range) - float(self.robot_radius)

        return float(np.min(clear[mask]))
    

    def _motion_dir_angle(self) -> float:
        v = float(self.last_action_executed[0]) if hasattr(self, "last_action_executed") else 0.0
        yaw = float(self.get_robot_yaw_wf())
        return wrap_angle(yaw + (np.pi if v < 0.0 else 0.0))
    
    def _motion_clearance(self) -> float:
        # returns skin-to-skin clearance in meters along motion direction cone
        center = self._motion_dir_angle()
        return self._min_dist_in_cone(center_angle=center, half_angle=float(self.lidar_front_half_angle))
    

    def _get_obstacle_envelope_rf(self):
        """
        Returns obstacle cone in ROBOT frame using:
        - contiguous hit cluster on circular LiDAR ring
        - +1 padding beam on each side (wrap-around)

        Output (or None if no hit):
        center_rf: float in [-pi, pi]
        half_rf:   float >= 0
        obs_dist:  float, center-to-hit distance (meters) for the seed/closest hit
        clearance: float, skin-to-skin (meters) = obs_dist - robot_radius
        """
        if (not getattr(self, "use_lidar", False)) or (self._last_lidar is None):
            return None

        N = int(self.lidar_num_rays)
        R = float(self.lidar_range)
        hit_mask = (self._last_lidar < 0.999)

        if not np.any(hit_mask):
            return None

        # Distances (center-to-hit)
        dists_center = self._last_lidar.astype(np.float32) * R
        hit_idxs = np.where(hit_mask)[0]

        # Seed = closest hit
        i_seed = int(hit_idxs[np.argmin(dists_center[hit_idxs])])

        def prev_i(i): return (i - 1) % N
        def next_i(i): return (i + 1) % N

        # Expand contiguous cluster on circular ring
        i_left = i_seed
        while hit_mask[prev_i(i_left)] and prev_i(i_left) != i_seed:
            i_left = prev_i(i_left)

        i_right = i_seed
        while hit_mask[next_i(i_right)] and next_i(i_right) != i_seed:
            i_right = next_i(i_right)

        # If every ray hits, you don't really have a meaningful "envelope"
        if np.all(hit_mask):
            return None

        # Padding beams: one outside cluster on each side
        pad_left = prev_i(i_left)
        pad_right = next_i(i_right)

        # Ensure padding beams are actually outside (walk outward if necessary)
        steps = 0
        while hit_mask[pad_left] and steps < N:
            pad_left = prev_i(pad_left)
            steps += 1

        steps = 0
        while hit_mask[pad_right] and steps < N:
            pad_right = next_i(pad_right)
            steps += 1

        # Boundary angles in robot frame come directly from padded indices
        ang_L = float(self.lidar_angles[pad_left])
        ang_R = float(self.lidar_angles[pad_right])

        # Define the cone as the shortest arc from ang_R to ang_L
        dtheta = float(np.arctan2(np.sin(ang_L - ang_R), np.cos(ang_L - ang_R)))
        half = 0.5 * abs(dtheta)
        center = float(np.arctan2(np.sin(ang_R + 0.5 * dtheta), np.cos(ang_R + 0.5 * dtheta)))

        obs_dist = float(dists_center[i_seed])
        clearance = float(obs_dist - float(self.robot_radius))

        return center, half, obs_dist, clearance

        


    def visualize_robot_circle(
        self,
        radius: float | None = None,
        *,
        z: float | None = None,
        num_segments: int = 48,
        color=(0.0, 1.0, 1.0),
        line_width: float = 2.0,
        life_time: float = 0.0,
        clear_previous: bool = True,
    ):
        """
        Draw a circle around the robot center in the XY plane for debugging.

        Assumes:
        - self.render_mode exists (truthy when GUI/debug drawing is desired)
        - self._last_env_ob contains robot pose like your other debug functions
        - optional: self.robot_radius (used if radius is None)

        Parameters:
        radius: Circle radius in meters. Defaults to self.robot_radius if available, else 0.25.
        z: Height to draw the circle at. Defaults to self.lidar_height if available, else robot base z.
        num_segments: Polyline resolution.
        color: RGB tuple/list in [0,1].
        line_width: Debug line width.
        life_time: 0.0 = persistent (until removed), >0 seconds = auto-expire.
        clear_previous: If True, removes previously drawn circle segments first.
        """
        if not getattr(self, "render_mode", False):
            return
        if getattr(self, "_last_env_ob", None) is None:
            return

        # --- Clear prior circle lines
        if clear_previous:
            for lid in getattr(self, "_robot_circle_dbg_line_ids", []):
                try:
                    p.removeUserDebugItem(lid)
                except Exception:
                    pass
            self._robot_circle_dbg_line_ids = []

        base_state = self._last_env_ob[0]["robot_0"]["joint_state"]
        pos = base_state["position"]
        cx, cy = float(pos[0]), float(pos[1])
        base_z = float(pos[2]) if len(pos) > 2 else 0.0

        if radius is None:
            radius = float(getattr(self, "robot_radius", 0.25))

        if z is None:
            # Prefer lidar height (consistent with your other debug visuals), else base_z
            z = float(getattr(self, "lidar_height", base_z))

        n = max(8, int(num_segments))
        two_pi = 2.0 * np.pi

        # Build and draw polyline circle
        line_ids = getattr(self, "_robot_circle_dbg_line_ids", [])
        for i in range(n):
            a0 = two_pi * (i / n)
            a1 = two_pi * ((i + 1) / n)

            p0 = [cx + radius * float(np.cos(a0)), cy + radius * float(np.sin(a0)), z]
            p1 = [cx + radius * float(np.cos(a1)), cy + radius * float(np.sin(a1)), z]

            lid = p.addUserDebugLine(
                p0, p1,
                lineColorRGB=list(color),
                lineWidth=float(line_width),
                lifeTime=float(life_time),
            )
            line_ids.append(lid)

        self._robot_circle_dbg_line_ids = line_ids




    # ---------------------------------------------------------------------
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

        if self.use_lidar:
            # use cached lidar; if not available, fill with "no hit"
            if self._last_lidar is None:
                lidar = np.ones(self.lidar_num_rays, dtype=np.float32)
            else:
                lidar = self._last_lidar.astype(np.float32)
            return np.concatenate([obs_core, lidar], axis=0)

        return obs_core

  
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
        kpr = 6.0 # used to be 6.0
        progress_reward = kpr * (prev_dist - dist) / 0.01

        # motion reward (velocity toward goal)
        table_xy, _, tv_xy, _ = self.sim.get_table_state_world()
        goal_xy = self.goal[:2]

        dir_to_goal = goal_xy - table_xy

        # normalize reward vector so it works for the speed toward goal + efficiency reward
        dir_to_goal /= np.linalg.norm(dir_to_goal) + 1e-6

        speed_toward_goal = np.dot(tv_xy, dir_to_goal) # in this reward
        kmr = 2.0 # used to be 2.0
        motion_reward = kmr * speed_toward_goal

        # distance penalty for not reaching goal
        distance_penalty = - dist / 10

        # heading penalty squared
        whp = 3.5 # used to be 3.5
        heading_penalty = - (whp * heading_error_bi) ** 2

        # ---------------------------------------------------------
        # GOAL-POWER REWARD  
        # ---------------------------------------------------------
        table_xy, _, tv_xy, _ = self.sim.get_table_state_world()
        goal_xy = self.goal[:2]


        base_state = self._last_env_ob[0]["robot_0"]["joint_state"]
        
        # USE THE HELPER (Standardize Yaw to World Frame)
        robot_pos = np.array(base_state["position"][:2])
        obstacle_penalty, is_crash = self._calc_obstacle_penalty(robot_pos)


        # ---------------------------------------------------------
        # Option 2: BYPASS SHAPING
        # Reward making obstacle move out of the forward cone when blocked.
        # ---------------------------------------------------------
        # ---------------------------------------------------------
        # LiDAR-based bypass shaping (bidirectional)
        # ---------------------------------------------------------

        bypass_reward = self._avoidance_shaping()

        # final reward
        total_reward = (
            progress_reward +
            motion_reward +
            distance_penalty +
            heading_penalty + obstacle_penalty + bypass_reward
        )

        # if self.step_count % 10 == 0:
        #     print(f"total: {total_reward} | progress: {progress_reward} | motion: {motion_reward} | distance: {distance_penalty} | heading:{heading_penalty} |  obstacle: {obstacle_penalty} |bypass: {bypass_reward}")


        

        return total_reward, progress_reward, motion_reward, distance_penalty, heading_penalty, obstacle_penalty, bypass_reward, is_crash
    

    def _calc_obstacle_penalty(self, _robot_pos_unused):
        """
        Calculates static repulsion based purely on LiDAR observation.
        
        Args:
            _robot_pos_unused: Kept for signature compatibility, but ignored.
            
        Returns:
            (float) penalty, (bool) is_crash
        """
        # 1. Safety / Initialization Checks
        if (not getattr(self, "use_lidar", False)) or (self._last_lidar is None):
            return 0.0, False

        # 2. Get the closest object detected by any ray
        # self._last_lidar values are in [0, 1], representing [0, max_range]
        raw_min_norm = np.min(self._last_lidar)
        
        # If the closest ray is at max range (approx 1.0), we are safe.
        if raw_min_norm >= 0.999:
            return 0.0, False

        # 3. Convert to Real Meters
        # geometric distance from Lidar center to hit point
        dist_to_hit = raw_min_norm * float(self.lidar_range)
        
        # 4. Calculate Skin-to-Skin Clearance
        # We subtract the robot radius to get the distance from the robot's "skin"
        clearance = dist_to_hit - float(self.robot_radius)

        # -------------------------------
        # TUNING PARAMETERS
        # -------------------------------
        CRASH_DIST = 0.0   # Meters. Closer than this = 'Crash' zone. Also tried 0.2 larger crash_dist did not seem to matter or robot, but need to connect this with lidar andactual distance
        SAFE_DIST  = 0.2   # Meters. The "uncomfortable" zone.
        
        # Penalties
        P_CRASH    = -5.0  # Heavy constant penalty for touching # was 7.0
        K_STATIC   = 5.0    # Scaling factor for the approach (shaping)

        # 5. Check Crash Zone (The "Hot Stove")
        # We return is_crash=False so the episode continues, allowing the agent
        # to learn how to back away from the penalty source.
        if clearance <= CRASH_DIST:
            print(f"[CRASH], clearance: {clearance}")
            return P_CRASH, False 
        
        #if self.step_count % 10 == 0:
            #print("clearance in obstacle penalty", clearance)

        # Smooth repulsion inside SAFE_DIST (continuous gradient)
        if clearance < SAFE_DIST:
            # clearance in (CRASH_DIST, SAFE_DIST)
            x = (SAFE_DIST - clearance) / max(SAFE_DIST - CRASH_DIST, 1e-6)  # 0..1
            penalty = -K_STATIC * (x ** 2)   # quadratic ramp
            return float(penalty), False
        
        return 0.0, False
    
    def _avoidance_shaping(self) -> float:
        if (not getattr(self, "use_lidar", False)) or (self._last_lidar is None):
            return 0.0

        v = float(self.last_action_executed[0]) if hasattr(self, "last_action_executed") else 0.0
        if abs(v) < float(self.escape_v_min):
            return 0.0

        envelope = self._get_obstacle_envelope_rf()
        if envelope is None:
            return 0.0

        cone_center_rf, cone_half_rf, obs_dist, clearance = envelope
        travel_rf = 0.0 if v >= 0.0 else float(np.pi)

        # if self.step_count % 10 == 0:
        #     print("clearance in avoidance shaping", clearance)


        # --------------------------
        # NO inflation: use raw LiDAR cone half-angle
        # --------------------------
        half = float(cone_half_rf)

        # Angular difference to cone center
        delta = float(np.arctan2(np.sin(travel_rf - cone_center_rf), np.cos(travel_rf - cone_center_rf)))
        ad = abs(delta)

        # Proximity scaling (skin-to-skin) -- unchanged
        d_enter = float(getattr(self, "lidar_block_dist", 2.0))
        d_safe  = float(getattr(self, "lidar_clear_margin", 0.20))
        prox = (d_enter - float(clearance)) / max(d_enter - d_safe, 1e-6)
        prox = float(np.clip(prox, 0.0, 1.0))

        if prox <= 0.0:
            return 0.0

        # Bonus band outside the cone (default 30 degrees)
        bonus_band = float(getattr(self, "avoid_bonus_band", np.deg2rad(30.0)))

        inside_cone = (ad <= half)
        in_bonus_band = (ad > half) and (ad <= (half + bonus_band))

        k_in  = 6.0 # 6.0 # also tried 10, 3
        k_out = 2.0 # also tried 3 see 1111

        # How far past the cone boundary we are (0 at boundary, 1 at outer edge of bonus band)
        outside_depth = (ad - half) / max(bonus_band, 1e-6)
        outside_depth = float(np.clip(outside_depth, 0.0, 1.0))

        # Optional shaping curve: emphasize steering further away from boundary
        outside_gain = outside_depth   # or keep linear: outside_gain = outside_depth


        if inside_cone:
            #print(f"inside cone penalty: {-k_in * (prox ** 2)}")
            return -k_in * (prox ** 2)
        elif in_bonus_band:
            #print(f"outside cone bonus: {+k_out * ((1.2 - prox))}")
            return +k_out * ((1.0 - prox)) * outside_gain
        else:
            return 0.0







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

        GAIN_LIN = 0.5 / 40  # (m/s) per Newton
        GAIN_ANG = 1.0 / 40  # (rad/s) per Newton (approx torque)

        
        DEADZONE = 1.0   # Newtons (ignore small noise/tremors)

        v_sac, w_sac = action

        # 1. Check if force is significant
        f_mag = np.linalg.norm(Fh_world_xy)
        if f_mag < DEADZONE:
            return action # No change

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
        v_adm = np.clip(v_adm, -1.0, 1.0)
        w_adm = np.clip(w_adm, -1.5, 1.5)

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

        
        # Update sensing caches here (one scan per step)
        self._update_lidar_caches()


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



        reward, progress_reward, motion_reward, distance_penalty, heading_penalty, obstacle_penalty, bypass_reward, is_crash = (
            self.compute_reward(dist, self.prev_dist, heading_error_bi)
        )

        # Update prev_dist AFTER computing reward
        self.prev_dist = dist

        # -------------------------------------------------------
        # 6) TERMINATION & TRUNCATION
        # -------------------------------------------------------
        terminated = dist < 0.4
        if terminated:
            print(f"[CHECK] Reached goal at step: {self.step_count}")
            reward += 150

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
            "bypass_reward": bypass_reward,
        }

        if self.render and (self.step_count % 75) == 0: 
            self.visualize_avoid_envelope()
            self.visualize_robot_circle()
            #self.visualize_lidar()
            #print(self._last_lidar * self.lidar_range)


        


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
            #self.sim.disable_arm_motors()

            # 6) Save observation (list-format)
            self._last_env_ob = ob0 

            # ---------------------------------------------------
            # !!! CONDITIONAL OBSTACLE SPAWNING !!!
            # ---------------------------------------------------
            if self.use_obstacles:
                mm = MapMaker()
                mm.create_map_1()  # Spawns the wall

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


        # ===================================================
        # 🔗 CREATE CONNECTION (The New Step)
        # ===================================================
    
        
        self.sim.create_connection_impedance()

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

        self.prev_obs_dist_surface = None
        self.prev_obs_bearing = None

        if self.use_lidar:
            self.lidar_ignore_ids = {self.sim.albert_id, self.sim.table_id}


        # Initialize last_action_executed so motion clearance is well-defined
        self.last_action_executed = np.array([0.0, 0.0], dtype=np.float32)

        # Compute first scan caches, then return observation
        self._update_lidar_caches()
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

