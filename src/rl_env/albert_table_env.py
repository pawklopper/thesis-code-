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
    def __init__(self, render=False, max_steps=1200, use_obstacles=False, goals=None):
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
        # original action space
        self.action_space = spaces.Box(
            low=np.array([-0.5, -1.0], dtype=np.float32),
            high=np.array([0.5, 1.0], dtype=np.float32)
        )

        self.obstacle_pos = None      
        self.obstacle_extents = None




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


        # another toggle use obstacles: 
        self.use_obstacles = use_obstacles


        if self.use_force_observations:
            print("OBSERVATION MODE: REAL FORCES INCLUDED")
        else:
            print("OBSERVATION MODE: FORCES ZEROED")

        self.robot_radius = 0.5  # set to your base footprint radius

       # --------------------------
        # LiDAR (light) configuration
        # --------------------------
        self.use_lidar = True
        self.lidar_num_rays = 36
        self.lidar_range = 3.0          # max ray length
        self.lidar_height = 0.25        # ray Z origin (adjust if needed)
        self.lidar_front_half_angle = np.deg2rad(60)  # +/- 60 degrees for shaping
        self.lidar_max_iters = 4        # see-through attempts per ray
        self.lidar_eps_adv = 1e-3       # advance beyond ignored hit

        # Will be populated after bodies exist
        self.lidar_ignore_ids = set()

        # Cache last scan metrics (used in reward)
        self._last_lidar = None
        self._last_lidar_min_dist = self.lidar_range
        self._last_lidar_min_dist_front = self.lidar_range


        obs_dim = 10 + (self.lidar_num_rays if self.use_lidar else 0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.lidar_block_dist = 2.0

        self.lidar_angles = np.linspace(-np.pi, np.pi, self.lidar_num_rays, endpoint=False).astype(np.float32)

        self.lidar_clear_margin = 0.20     # skin-to-skin margin required before reward starts
        self.lidar_escape_k = 4.0          # magnitude of positive escape reward
        self.escape_v_min = 0.05           # below this speed, treat as "not moving"


        # --------------------------
        # Space-mode toggle (hysteresis)
        # --------------------------
        self.space_mode_enter = self.lidar_block_dist              # enter when motion clearance < this
        self.space_mode_exit  = 3.0       # exit when motion clearance > this
        self.space_mode_min_hold_steps = 10                        # optional: avoid flicker
        self._space_mode_hold = 0



    def _update_space_mode(self):
        """
        Toggle self.space_mode using clearance in the motion cone with hysteresis.
        Uses self._last_lidar_min_dist_motion (skin-to-skin distance in motion direction).
        """
        if (not getattr(self, "use_lidar", False)) or (self._last_lidar is None):
            self.space_mode = False
            self._space_mode_hold = 0
            return

        clear_motion = float(getattr(self, "_last_lidar_min_dist_motion", self.lidar_range))

        # Optional hold timer to prevent flicker
        if self._space_mode_hold > 0:
            self._space_mode_hold -= 1
            return

        if (not self.space_mode) and (clear_motion < float(self.space_mode_enter)):
            self.space_mode = True
            self._space_mode_hold = int(self.space_mode_min_hold_steps)

        elif self.space_mode and (clear_motion > float(self.space_mode_exit)):
            self.space_mode = False
            self._space_mode_hold = int(self.space_mode_min_hold_steps)



        




    def _ray_cast_ignore_ids(self, start, end, ignore_ids, max_iters=4, eps=1e-3):
        """
        Ray cast start->end. If it hits an ignored body, advance start slightly and retry.
        Returns: (hit_fraction in [0,1], hit_body_id)
        """
        start = np.array(start, dtype=np.float32)
        end   = np.array(end, dtype=np.float32)

        dir_vec = end - start
        dir_len = float(np.linalg.norm(dir_vec))
        if dir_len < 1e-9:
            return 1.0, -1

        dir_unit = dir_vec / dir_len
        cur_start = start.copy()

        for _ in range(max_iters):
            body_id, _, hit_frac, _, hit_pos = p.rayTest(cur_start.tolist(), end.tolist())[0]

            # no hit
            if body_id < 0 or hit_frac >= 1.0:
                return 1.0, -1

            # hit ignored -> see through
            if body_id in ignore_ids:
                hit_pos = np.array(hit_pos, dtype=np.float32)
                cur_start = hit_pos + eps * dir_unit
                if float(np.linalg.norm(end - cur_start)) < 1e-4:
                    return 1.0, -1
                continue

            # valid hit
            return float(hit_frac), int(body_id)

        return 1.0, -1
    
    def _lidar_scan(self, robot_xy, robot_yaw):
        """
        Returns:
        lidar: (N,) float32 in [0,1] (1.0 means no hit within range)
        min_dist: min physical distance (0..range), range if none
        min_dist_front: min distance in +/- front cone
        """
        N = self.lidar_num_rays
        R = self.lidar_range
        z0 = self.lidar_height

        angles = np.linspace(-np.pi, np.pi, N, endpoint=False, dtype=np.float32)
        start = np.array([robot_xy[0], robot_xy[1], z0], dtype=np.float32)

        lidar = np.ones(N, dtype=np.float32)
        ignore_ids = self.lidar_ignore_ids

        for i, a in enumerate(angles):
            aw = float(robot_yaw + a)
            end = np.array([robot_xy[0] + R * np.cos(aw),
                            robot_xy[1] + R * np.sin(aw),
                            z0], dtype=np.float32)

            frac, _ = self._ray_cast_ignore_ids(
                start, end,
                ignore_ids=ignore_ids,
                max_iters=self.lidar_max_iters,
                eps=self.lidar_eps_adv
            )
            lidar[i] = np.clip(frac, 0.0, 1.0)

        dists = lidar * R
        min_dist = float(np.min(dists))

        front_mask = np.abs(angles) <= self.lidar_front_half_angle
        min_dist_front = float(np.min(dists[front_mask])) if np.any(front_mask) else min_dist

        return lidar, min_dist, min_dist_front


    def compute_lidar_scan(self, robot_xy, robot_yaw):
        """
        Computes a 2D LiDAR scan in the plane at z=self.lidar_height.

        Returns:
        angles: (N,) radians in robot frame ([-pi, pi))
        fracs:  (N,) hitFraction in [0,1], 1.0 => no hit within range
        dists:  (N,) physical distance = fracs * range
        hits:   (N,3) world hit points (end point if no hit)
        min_dist: float, minimum distance across all rays
        min_front: float, minimum distance in +/- front cone
        """
        N = self.lidar_num_rays
        R = self.lidar_range
        z0 = self.lidar_height

        angles = np.linspace(-np.pi, np.pi, N, endpoint=False, dtype=np.float32)
        start = np.array([robot_xy[0], robot_xy[1], z0], dtype=np.float32)

        fracs = np.ones(N, dtype=np.float32)
        hits = np.zeros((N, 3), dtype=np.float32)

        ignore_ids = self.lidar_ignore_ids

        for i, a in enumerate(angles):
            aw = float(robot_yaw + float(a))
            end = np.array([robot_xy[0] + R * np.cos(aw),
                            robot_xy[1] + R * np.sin(aw),
                            z0], dtype=np.float32)

            frac, _ = self._ray_cast_ignore_ids(
                start, end,
                ignore_ids=ignore_ids,
                max_iters=self.lidar_max_iters,
                eps=self.lidar_eps_adv
            )
            frac = float(np.clip(frac, 0.0, 1.0))
            fracs[i] = frac
            hits[i, :] = start + frac * (end - start)  # end if frac==1.0

        dists = fracs * R
        min_dist = float(np.min(dists))

        front_mask = np.abs(angles) <= float(self.lidar_front_half_angle)
        min_front = float(np.min(dists[front_mask])) if np.any(front_mask) else min_dist

        return angles, fracs, dists, hits, min_dist, min_front
    

    def visualize_avoid_envelope(self):
        """
        Visualize the envelope used by _lidar_heading_avoid_term():
        - Rays with dist < d_block are 'blocked'
        - Their penalty contribution is closeness * cos^2(delta)
        """
        if (not self.render_mode) or (not getattr(self, "use_lidar", False)):
            return
        if self._last_env_ob is None or self._last_lidar is None:
            return

        # Clear previous items
        for lid in getattr(self, "_avoid_dbg_line_ids", []):
            try:
                p.removeUserDebugItem(lid)
            except Exception:
                pass
        self._avoid_dbg_line_ids = []

        if getattr(self, "_avoid_dbg_text_id", None) is not None:
            try:
                p.removeUserDebugItem(self._avoid_dbg_text_id)
            except Exception:
                pass
            self._avoid_dbg_text_id = None

        # Robot pose
        base_state = self._last_env_ob[0]["robot_0"]["joint_state"]
        robot_xy = np.array(base_state["position"][:2], dtype=np.float32)
        robot_yaw = float(self.get_robot_yaw_wf())
        start = [float(robot_xy[0]), float(robot_xy[1]), float(self.lidar_height)]

        # Params
        R = float(self.lidar_range)
        d_block = float(getattr(self, "lidar_block_dist", 0.60))

        # Motion heading (robot frame): forward only for this term (your function returns early if v<=0)
        v = float(self.last_action_executed[0]) if hasattr(self, "last_action_executed") else 0.0
        w = float(self.last_action_executed[1]) if hasattr(self, "last_action_executed") else 0.0
        motion_heading = 0.0 if v >= 0.0 else float(np.pi)

        # Compute blockedness and contribution per beam
        dists = self._last_lidar.astype(np.float32) * R
        closeness = np.clip((d_block - dists) / max(d_block, 1e-6), 0.0, 1.0)

        delta = np.arctan2(np.sin(self.lidar_angles - motion_heading),
                        np.cos(self.lidar_angles - motion_heading))
        align = (np.cos(delta) ** 2).astype(np.float32)

        contrib = closeness * align
        align_mean = float(np.sum(contrib) / (float(np.sum(closeness)) + 1e-6))

        # Get hit geometry for drawing
        angles, fracs, dists_geom, hits, min_dist, min_front = self.compute_lidar_scan(robot_xy, robot_yaw)

        # Draw motion heading arrow (white)
        aw = robot_yaw + motion_heading
        end_motion = [start[0] + R * float(np.cos(aw)),
                    start[1] + R * float(np.sin(aw)),
                    start[2]]
        self._avoid_dbg_line_ids.append(
            p.addUserDebugLine(start, end_motion, lineColorRGB=[1, 1, 1], lineWidth=3.0, lifeTime=0.0)
        )

        # Draw envelope rays:
        # - only beams with closeness > 0 (within d_block)
        # - color intensity represents contribution (contrib)
        # - red = high contribution (strongly penalized)
        # - yellow = medium contribution
        # - blue = blocked but not aligned (low contribution)
        for i in range(self.lidar_num_rays):
            if float(closeness[i]) <= 1e-6:
                continue

            end = hits[i, :].tolist()
            c = float(contrib[i])  # 0..1-ish

            if c > 0.50:
                color = [1.0, 0.0, 0.0]      # strong penalty contributor
                width = 4.0
            elif c > 0.20:
                color = [1.0, 1.0, 0.0]      # medium
                width = 3.0
            else:
                color = [0.2, 0.2, 1.0]      # blocked but not aligned
                width = 2.5

            self._avoid_dbg_line_ids.append(
                p.addUserDebugLine(start, end, lineColorRGB=color, lineWidth=width, lifeTime=0.0)
            )

        # # HUD text
        # txt = (
        #     f"AVOID ENVELOPE (space_mode={bool(getattr(self,'space_mode',False))})\n"
        #     f"d_block={d_block:.2f} | v={v:.2f} w={w:.2f}\n"
        #     f"align_mean={align_mean:.3f} | max closeness={float(np.max(closeness)):.2f}"
        # )
        # self._avoid_dbg_text_id = p.addUserDebugText(
        #     txt,
        #     [start[0], start[1], start[2] + 0.60],
        #     textColorRGB=[1, 1, 1],
        #     textSize=1.2,
        #     lifeTime=0.0
    #)

    
    def visualize_lidar(self):
        """
        Debug visualization in PyBullet GUI.
        Draws LiDAR rays from robot position with colors based on hit distance.
        Ignores robot base + table via lidar_ignore_ids.

        Call from step() when render_mode is enabled.
        """
        if (not self.render_mode) or (not self.lidar_debug) or (not self.use_lidar):
            return
        if self._last_env_ob is None:
            return

        # Clear previous debug items
        for lid in self._lidar_debug_line_ids:
            try:
                p.removeUserDebugItem(lid)
            except Exception:
                pass
        self._lidar_debug_line_ids.clear()

        if self._lidar_debug_text_id is not None:
            try:
                p.removeUserDebugItem(self._lidar_debug_text_id)
            except Exception:
                pass
            self._lidar_debug_text_id = None

        # Current robot pose
        base_state = self._last_env_ob[0]["robot_0"]["joint_state"]
        robot_xy = np.array(base_state["position"][:2], dtype=np.float32)
        robot_yaw = self.get_robot_yaw_wf()
        start = [float(robot_xy[0]), float(robot_xy[1]), float(self.lidar_height)]

        angles, fracs, dists, hits, min_dist, min_front = self.compute_lidar_scan(robot_xy, robot_yaw)

        # Draw rays
        for i in range(self.lidar_num_rays):
            end = hits[i, :].tolist()
            dist = float(dists[i])
            frac = float(fracs[i])
            in_front = abs(float(angles[i])) <= float(self.lidar_front_half_angle)

            # Color scheme:
            # - no hit => green
            # - close hit => red
            # - mid hit => orange
            if frac >= 0.999:
                color = [0.0, 1.0, 0.0]
            else:
                if dist < 0.30:
                    color = [1.0, 0.0, 0.0]
                else:
                    color = [1.0, 0.6, 0.0]

            # Optional: front-cone tint to distinguish sector
            if in_front and frac < 0.999:
                color = [0.2, 0.2, 1.0]

            line_id = p.addUserDebugLine(
                start,
                end,
                lineColorRGB=color,
                lineWidth=1.5,
                lifeTime=0.0
            )
            self._lidar_debug_line_ids.append(line_id)

        # Draw a text label above robot
        txt = f"LiDAR min={min_dist:.3f} | front min={min_front:.3f} | ignore={list(self.lidar_ignore_ids)}"
        self._lidar_debug_text_id = p.addUserDebugText(
            txt,
            [start[0], start[1], start[2] + 0.35],
            textColorRGB=[1, 1, 1],
            textSize=1.2,
            lifeTime=0.0
        )


    
    def closest_point_on_rect(self, point, rect_center, rect_extents):
        """
        Closest point on an axis-aligned rectangle (AABB) in 2D.
        rect_center: (2,)
        rect_extents: (2,) half-sizes
        """
        rect_min = rect_center - rect_extents
        rect_max = rect_center + rect_extents
        return np.minimum(np.maximum(point, rect_min), rect_max)

        
        
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
    

    def get_surface_dist_to_rect(self, point, rect_center, rect_extents, radius=0.0):
        """
        Returns distance from the ROBOT SURFACE (circle) to the RECTANGLE SURFACE.
        Positive = Safe.
        Negative = Crashed/Penetrating.
        """
        # 1. Get vector from center to point
        diff = np.abs(point - rect_center)
        
        # 2. distance from box edge to point (ignoring radius for a moment)
        outer_dist = np.maximum(diff - rect_extents, 0.0)
        dist_center_to_wall = np.linalg.norm(outer_dist)

        # 3. Check if we are physically inside the box rectangle (center is inside)
        #    If outer_dist is all zeros, the center is inside the box.
        #    (This handles the rare case where the center teleports inside)
        is_center_inside = (outer_dist == 0).all()
        
        if is_center_inside:
            # Deep penetration logic (simplified)
            return -1.0 # Just return a collision value

        # 4. Subtract Robot Radius to get "Skin-to-Skin" distance
        surface_dist = dist_center_to_wall - radius
        
        return surface_dist
    

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
            robot_xy = np.array(base_state["position"][:2], dtype=np.float32)
            lidar, min_dist, min_dist_front = self._lidar_scan(robot_xy, robot_yaw)

            # store for reward/termination if you want
            # store for reward/termination
            self._last_lidar = lidar
            self._last_lidar_min_dist = min_dist
            self._last_lidar_min_dist_front = min_dist_front


            return np.concatenate([obs_core, lidar.astype(np.float32)], axis=0)

        return obs_core
    
  


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
        Fh_xy = self.sim.last_Fh_xy
        Fr_xy = self.sim.last_F_xy

        #collaboration_reward = self.compute_collaboration_reward(dir_to_goal)

        # set to zero for now as motion_reward might already be the collaboration reward you want
        collaboration_reward = 0.0


        base_state = self._last_env_ob[0]["robot_0"]["joint_state"]
        
        # USE THE HELPER (Standardize Yaw to World Frame)
        robot_pos = np.array(base_state["position"][:2])
        obstacle_penalty, is_crash = self._calc_obstacle_penalty(robot_pos)


        # ---------------------------------------------------------
        # Option 2: BYPASS SHAPING
        # Reward making obstacle move out of the forward cone when blocked.
        # ---------------------------------------------------------
        avoid_pen, avoid_dbg = self._lidar_heading_avoid_term()

        # if self.step_count % 10 == 0:
        #     print("heading penalty", avoid_pen)
        #     print("space reward", space_reward)

        space_reward, space_dbg = self._escape_heading_reward()

        bypass_reward = avoid_pen + space_reward

        # if self.step_count % 10 == 0:
        #     print("space_reward", space_reward)
        #     print("avoid pen", avoid_pen)



        # final reward
        total_reward = (
            progress_reward +
            motion_reward +
            distance_penalty +
            heading_penalty + obstacle_penalty + bypass_reward
        )

        

        return total_reward, progress_reward, motion_reward, distance_penalty, heading_penalty, obstacle_penalty, bypass_reward, is_crash
    

    def _calc_obstacle_penalty(self, robot_pos):
        ROBOT_RADIUS = float(self.robot_radius)

        if self.obstacle_pos is None or self.obstacle_extents is None:
            return 0.0, False

        rect_min = self.obstacle_pos - self.obstacle_extents
        rect_max = self.obstacle_pos + self.obstacle_extents
        closest = np.minimum(np.maximum(robot_pos, rect_min), rect_max)

        vec = closest - robot_pos
        dist_center_to_wall = float(np.linalg.norm(vec))

        # Surface (skin-to-skin) distance
        if dist_center_to_wall < 1e-9:
            dist_surface = -1.0
        else:
            dist_surface = dist_center_to_wall - ROBOT_RADIUS

        # --------------------------
        # 1) Crash condition
        # --------------------------
        crash_dist = 0.10  # meters; your current threshold
        if dist_surface <= crash_dist:
            # print(f"[COLISSION] AT STEP {self.step_count}")
            return -1.0, False

        # --------------------------
        # 2) Smooth clearance penalty
        # --------------------------
        # Start penalizing before crash to discourage "corner clipping"
        d_safe = 1.0  # meters; tune (0.25–0.50 is typical)
        if dist_surface < d_safe:
            # normalized violation in [0,1]
            x = (d_safe - dist_surface) / (d_safe - crash_dist + 1e-6)
            x = float(np.clip(x, 0.0, 1.0))

            # quadratic penalty (0 at d_safe, -k at crash_dist)
            k_clear = 2.0  # tune relative to your other reward terms
            penalty = -k_clear * (x ** 2)

            return float(penalty), False

        return 0.0, False

        

    def _lidar_heading_avoid_term(self):
        """
        Secondary guardrail: penalize pushing forward into blocked directions,
        but do not incentivize freezing. Only active when space_mode is ON.
        """
        if (not getattr(self, "use_lidar", False)) or (self._last_lidar is None):
            return 0.0, {}

        # Only apply when we're in "blocked" mode (near obstacle)
        if not getattr(self, "space_mode", False):
            return 0.0, {"active": False, "reason": "space_mode_off"}

        v = float(self.last_action_executed[0])
        w = float(self.last_action_executed[1])

        # Backing up is allowed; don't penalize it
        if v <= 0.0:
            return 0.0, {"active": False, "reason": "v_nonpositive"}

        # Distances in meters
        dists = self._last_lidar.astype(np.float32) * float(self.lidar_range)

        d_block = float(getattr(self, "lidar_block_dist", 0.60))

        closeness = np.clip((d_block - dists) / d_block, 0.0, 1.0)

        # if self.step_count % 10 == 0:
        #     print("closeness", closeness)

        if float(np.max(closeness)) <= 1e-6:
            return 0.0, {"active": False, "reason": "no_blocked"}

        # Motion heading for forward is 0
        v = float(self.last_action_executed[0])
        motion_heading = 0.0 if v >= 0.0 else np.pi


        delta = np.arctan2(np.sin(self.lidar_angles - motion_heading),
                        np.cos(self.lidar_angles - motion_heading))

        align = (np.cos(delta) ** 2).astype(np.float32)

        align_mean = float(np.sum(closeness * align) / (np.sum(closeness) + 1e-6))

        # Reduce penalty when turning (turning is how you open space)
        # turn_scale -> 0 when |w| large, 1 when |w| small
        w_turn = 0.6  # rad/s scale
        turn_scale = float(np.clip(1.0 - abs(w) / w_turn, 0.0, 1.0))

        w_align = 2.0  # keep small; this is now a secondary term
        term = -w_align * align_mean * turn_scale


        return term, {"active": True, "align_mean": align_mean, "turn_scale": turn_scale, "v": v, "w": w}
    

    def _escape_heading_reward(self):
        """
        Positive counterpart to the avoid penalty.

        Active only in space_mode (blocked mode).
        Rewards choosing a motion direction with clearance (in the motion cone),
        scaled by |v| so it doesn't reward freezing.
        """
        if (not getattr(self, "use_lidar", False)) or (self._last_lidar is None):
            return 0.0, {"active": False, "reason": "no_lidar"}
    

        if not getattr(self, "space_mode", False):
            return 0.0, {"active": False, "reason": "space_mode_off"}

        # Need last_action_executed set
        if not hasattr(self, "last_action_executed"):
            return 0.0, {"active": False, "reason": "no_action"}

        v = float(self.last_action_executed[0])

        v_min = float(getattr(self, "escape_v_min", 0.05))
        if abs(v) < v_min:
            return 0.0, {"active": False, "reason": "v_too_small", "v": v}

        # Motion heading in robot frame
        motion_heading = 0.0 if v >= 0.0 else float(np.pi)

        # Clearance in that cone (skin-to-skin) — uses your existing cone helper
        motion_clear = self._min_dist_in_cone(
            center_angle=motion_heading,
            half_angle=float(self.lidar_front_half_angle)
        )

        margin = float(getattr(self, "lidar_clear_margin", 0.20))
        good = max(0.0, float(motion_clear) - margin)

        k = float(getattr(self, "lidar_escape_k", 2.0))
        reward = float(k * abs(v) * good)

        return reward, {
            "active": True,
            "v": v,
            "motion_heading": motion_heading,
            "motion_clear": float(motion_clear),
            "margin": margin,
            "good": good,
            "k": k,
            "reward": reward
        }


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
        obs = self._get_obs()  # updates self._last_lidar for THIS step

        # Compute clearance in the direction of motion (forward/backward)
        if self.use_lidar and (self._last_lidar is not None):
            v_exec = float(self.last_action_executed[0])
            motion_heading = 0.0 if v_exec >= 0.0 else np.pi
            self._last_lidar_min_dist_motion = self._min_dist_in_cone(
                center_angle=motion_heading,
                half_angle=self.lidar_front_half_angle
            )
        else:
            self._last_lidar_min_dist_motion = float(self.lidar_range)

        
        self._update_space_mode()

        # -------------------------------------------------------
        # 5) GOAL DISTANCE + PROGRESS-STALL BOOKKEEPING + REWARD
        # -------------------------------------------------------
        table_xy, table_yaw, _, _ = self.sim.get_table_state_world()
        goal_xy = self.goal[:2]
        dx_t, dy_t = goal_xy - table_xy
        dist = float(np.linalg.norm([dx_t, dy_t]))


        heading_error_bi, heading_error = self.compute_robot_heading_error(dx_t, dy_t, robot_yaw)
        heading_diff = wrap_angle(table_yaw - robot_yaw)



        reward, progress_reward, motion_reward, distance_penalty, heading_penalty, obstacle_penalty, bypass_reward, is_crash = (
            self.compute_reward(dist, self.prev_dist, heading_error_bi, heading_diff)
        )

        # Update prev_dist AFTER computing reward
        self.prev_dist = dist

        # -------------------------------------------------------
        # 6) TERMINATION & TRUNCATION
        # -------------------------------------------------------
        terminated = dist < 0.4
        if terminated:
            print(f"[CHECK] Reached goal at step: {self.step_count}")
            reward += 100

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

        if self.render and (self.step_count % 10) == 0: 
            self.visualize_avoid_envelope()


        


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
                from rl_env.obstacles_and_sensing import MapMaker
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

        # LiDAR ignore list: ignore robot base and table
        if self.use_lidar:
            self.lidar_ignore_ids = {self.sim.albert_id, self.sim.table_id}

        self.space_mode = False
        self.prev_motion_clear = self.lidar_range
        self._last_lidar_min_dist_motion = self.lidar_range







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


    def _debug_arm(self):
        joints = [p.getJointState(self.sim.albert_id, j)[0] for j in self.sim.arm_joint_indices]

        print(f"\n==== arm pose at step: {self.step_count} ====")
        print("Joint angles:", np.round(joints, 3))