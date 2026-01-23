#!/usr/bin/env python3
"""
obstacles_and_sensing.py — Obstacles and Sensing
----------------------------------------------------
- Includes SimpleLidar (for raycasting if needed).
- Includes MapMaker with create_cylinder for the geometric navigation task.
"""

from __future__ import annotations

import numpy as np
import pybullet as p



class LidarMixin:
    """
    Methods assume the parent class defines:
      - self.render_mode, self.use_lidar
      - self._last_env_ob
      - self.lidar_num_rays, lidar_range, lidar_height, lidar_front_half_angle
      - self.lidar_max_iters, self.lidar_eps_adv
      - self.lidar_ignore_ids, self.lidar_angles
      - self.robot_radius
      - self.get_robot_yaw_wf()
      - self._min_dist_in_cone()
      - self.last_action_executed (optional; default handled)
    """

    def __init__(
        self,
        *,
        use_lidar: bool = True,
        lidar_num_rays: int = 36,
        lidar_range: float = 3.0,
        lidar_height: float = 0.25,
        lidar_front_half_angle: float = np.deg2rad(60),
        lidar_max_iters: int = 4,
        lidar_eps_adv: float = 1e-3,
        lidar_debug: bool = False,
        **kwargs,
    ):
        # Critical for multiple inheritance: keep init chain alive
        super().__init__(**kwargs)

        self.use_lidar = bool(use_lidar)
        self.lidar_num_rays = int(lidar_num_rays)
        self.lidar_range = float(lidar_range)
        self.lidar_height = float(lidar_height)
        self.lidar_front_half_angle = float(lidar_front_half_angle)
        self.lidar_max_iters = int(lidar_max_iters)
        self.lidar_eps_adv = float(lidar_eps_adv)
        self.lidar_debug = bool(lidar_debug)

        # Precompute angles and initialize caches
        self.lidar_angles = np.linspace(-np.pi, np.pi, self.lidar_num_rays, endpoint=False).astype(np.float32)
        self.lidar_ignore_ids = set()

        self._last_lidar = None
        self._last_lidar_min_dist = float(self.lidar_range)
        self._last_lidar_min_dist_front = float(self.lidar_range)
        self._last_lidar_min_dist_motion = float(self.lidar_range)

        # Debug handles
        self._lidar_debug_line_ids = []
        self._lidar_debug_text_id = None
        self._avoid_dbg_line_ids = []
        self._avoid_dbg_text_id = None

    def _ray_cast_ignore_ids(self, start, end, ignore_ids, max_iters=4, eps=1e-3):
        start = np.array(start, dtype=np.float32)
        end = np.array(end, dtype=np.float32)

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
        N = int(self.lidar_num_rays)
        R = float(self.lidar_range)
        z0 = float(self.lidar_height)

        # Use precomputed angles to avoid re-allocating each step
        angles = self.lidar_angles
        start = np.array([robot_xy[0], robot_xy[1], z0], dtype=np.float32)

        lidar = np.ones(N, dtype=np.float32)
        ignore_ids = self.lidar_ignore_ids

        for i, a in enumerate(angles):
            aw = float(robot_yaw + float(a))
            end = np.array([robot_xy[0] + R * np.cos(aw),
                            robot_xy[1] + R * np.sin(aw),
                            z0], dtype=np.float32)

            frac, _ = self._ray_cast_ignore_ids(
                start, end,
                ignore_ids=ignore_ids,
                max_iters=int(self.lidar_max_iters),
                eps=float(self.lidar_eps_adv),
            )
            lidar[i] = np.clip(frac, 0.0, 1.0)

        dists = lidar * R
        min_dist = float(np.min(dists))

        front_mask = np.abs(angles) <= float(self.lidar_front_half_angle)
        min_front = float(np.min(dists[front_mask])) if np.any(front_mask) else min_dist

        return lidar, min_dist, min_front

    def compute_lidar_scan(self, robot_xy, robot_yaw):
        N = int(self.lidar_num_rays)
        R = float(self.lidar_range)
        z0 = float(self.lidar_height)

        angles = self.lidar_angles
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
                max_iters=int(self.lidar_max_iters),
                eps=float(self.lidar_eps_adv),
            )
            frac = float(np.clip(frac, 0.0, 1.0))
            fracs[i] = frac
            hits[i, :] = start + frac * (end - start)

        dists = fracs * R
        min_dist = float(np.min(dists))

        front_mask = np.abs(angles) <= float(self.lidar_front_half_angle)
        min_front = float(np.min(dists[front_mask])) if np.any(front_mask) else min_dist

        return angles, fracs, dists, hits, min_dist, min_front

    def _update_lidar_caches(self):
        if (not getattr(self, "use_lidar", False)) or (self._last_env_ob is None):
            self._last_lidar = None
            self._last_lidar_min_dist = float(self.lidar_range)
            self._last_lidar_min_dist_front = float(self.lidar_range)
            self._last_lidar_min_dist_motion = float(self.lidar_range)
            return

        base_state = self._last_env_ob[0]["robot_0"]["joint_state"]
        robot_xy = np.array(base_state["position"][:2], dtype=np.float32)
        robot_yaw = float(self.get_robot_yaw_wf())

        lidar, min_dist, min_front = self._lidar_scan(robot_xy, robot_yaw)
        self._last_lidar = lidar
        self._last_lidar_min_dist = float(min_dist)
        self._last_lidar_min_dist_front = float(min_front)

        v_exec = float(self.last_action_executed[0]) if hasattr(self, "last_action_executed") else 0.0
        motion_heading = 0.0 if v_exec >= 0.0 else float(np.pi)

        self._last_lidar_min_dist_motion = self._min_dist_in_cone(
            center_angle=motion_heading,
            half_angle=float(self.lidar_front_half_angle),
        )


  
    def visualize_avoid_envelope(self):
        if (not self.render_mode) or (not getattr(self, "use_lidar", False)):
            return
        if self._last_env_ob is None or self._last_lidar is None:
            return

        # -----------------------------
        # Clear previous debug items
        # -----------------------------
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

        # -----------------------------
        # Robot pose + config
        # -----------------------------
        base_state = self._last_env_ob[0]["robot_0"]["joint_state"]
        robot_xy = np.array(base_state["position"][:2], dtype=np.float32)
        robot_yaw = float(self.get_robot_yaw_wf())
        start = [float(robot_xy[0]), float(robot_xy[1]), float(self.lidar_height)]

        N = int(self.lidar_num_rays)
        R = float(self.lidar_range)

        # ============================================================
        # Draw robot travel heading (white strip)
        #   - forward: along robot_yaw
        #   - backward: robot_yaw + pi
        # ============================================================
        v_exec = float(self.last_action_executed[0]) if hasattr(self, "last_action_executed") else 0.0
        heading_world = float(robot_yaw + (np.pi if v_exec < 0.0 else 0.0))

        heading_len = R
        end_heading = [
            start[0] + heading_len * float(np.cos(heading_world)),
            start[1] + heading_len * float(np.sin(heading_world)),
            start[2],
        ]
        self._avoid_dbg_line_ids.append(
            p.addUserDebugLine(start, end_heading, lineColorRGB=[1.0, 1.0, 1.0], lineWidth=5.0, lifeTime=0.0)
        )

        # Accurate ray endpoints (for drawing hit beams)
        angles_rf, fracs, dists_geom, hits, min_dist, min_front = self.compute_lidar_scan(robot_xy, robot_yaw)

        # -----------------------------
        # Hit mask
        # -----------------------------
        hit_mask = (self._last_lidar < 0.999)
        if not np.any(hit_mask):
            self._avoid_dbg_text_id = p.addUserDebugText(
                "OBS cone: no hits",
                [start[0], start[1], start[2] + 0.25],
                textColorRGB=[1, 1, 1],
                textSize=1.1,
                lifeTime=0.0
            )
            return

        # Seed = closest hit by center-to-hit distance
        dists_center = self._last_lidar.astype(np.float32) * R
        hit_idxs = np.where(hit_mask)[0]
        i_seed = int(hit_idxs[np.argmin(dists_center[hit_idxs])])

        # ============================================================
        # Find contiguous hit cluster on a circular array
        # ============================================================
        def prev_i(i): return (i - 1) % N
        def next_i(i): return (i + 1) % N

        # Expand left
        i_left = i_seed
        while hit_mask[prev_i(i_left)] and prev_i(i_left) != i_seed:
            i_left = prev_i(i_left)

        # Expand right
        i_right = i_seed
        while hit_mask[next_i(i_right)] and next_i(i_right) != i_seed:
            i_right = next_i(i_right)

        cluster_all = np.all(hit_mask)

        # Padding beams: one outside cluster on each side
        if not cluster_all:
            pad_left = prev_i(i_left)
            pad_right = next_i(i_right)

            steps = 0
            while hit_mask[pad_left] and steps < N:
                pad_left = prev_i(pad_left)
                steps += 1

            steps = 0
            while hit_mask[pad_right] and steps < N:
                pad_right = next_i(pad_right)
                steps += 1
        else:
            pad_left = i_left
            pad_right = i_right

        # Build cluster index list
        cluster_idxs = [i_left]
        i = i_left
        while i != i_right:
            i = next_i(i)
            cluster_idxs.append(i)

        # Colors (as requested)
        PURPLE = [0.7, 0.0, 1.0]  # hit beams
        RED    = [1.0, 0.0, 0.0]  # obstacle cone sides
        BLUE   = [0.0, 0.0, 1.0]  # bonus boundaries

        # 1) Draw hit beams in purple
        for j in cluster_idxs:
            end = hits[j, :].tolist()
            self._avoid_dbg_line_ids.append(
                p.addUserDebugLine(start, end, lineColorRGB=PURPLE, lineWidth=1.0, lifeTime=0.0)
            )

        # ============================================================
        # Compute raw obstacle cone (NO inflation)
        # ============================================================
        ang_L = float(self.lidar_angles[pad_left])    # robot frame
        ang_R = float(self.lidar_angles[pad_right])   # robot frame

        # Shortest arc from ang_R to ang_L
        dtheta = float(np.arctan2(np.sin(ang_L - ang_R), np.cos(ang_L - ang_R)))
        half = 0.5 * abs(dtheta)
        center = float(np.arctan2(np.sin(ang_R + 0.5 * dtheta), np.cos(ang_R + 0.5 * dtheta)))

        # Bonus band outside the cone (default 30 degrees)
        bonus_band = float(getattr(self, "avoid_bonus_band", np.deg2rad(30.0)))

        # Boundaries in robot frame
        red_left_rf   = float(center + half)
        red_right_rf  = float(center - half)
        blue_left_rf  = float(center + half + bonus_band)
        blue_right_rf = float(center - half - bonus_band)

        def _draw_boundary(angle_rf: float, color, width: float):
            aw = float(robot_yaw + angle_rf)  # convert to world frame
            end = [
                start[0] + R * float(np.cos(aw)),
                start[1] + R * float(np.sin(aw)),
                start[2],
            ]
            self._avoid_dbg_line_ids.append(
                p.addUserDebugLine(start, end, lineColorRGB=color, lineWidth=width, lifeTime=0.0)
            )

        # 2) Obstacle cone sides (RED)
        _draw_boundary(red_left_rf,  RED, 3.0)
        _draw_boundary(red_right_rf, RED, 3.0)

        # 3) Bonus boundaries (BLUE)
        _draw_boundary(blue_left_rf,  BLUE, 2.0)
        _draw_boundary(blue_right_rf, BLUE, 2.0)

        # 4) Label
        obs_dist = float(dists_center[i_seed])
        txt = (
            f"seed={i_seed} d_seed={obs_dist:.2f} "
            f"cone_half={np.rad2deg(half):.1f}deg "
            f"bonus_band={np.rad2deg(bonus_band):.1f}deg "
            f"padL={pad_left} padR={pad_right}"
        )
        self._avoid_dbg_text_id = p.addUserDebugText(
            txt,
            [start[0], start[1], start[2] + 0.25],
            textColorRGB=[1, 1, 1],
            textSize=1.1,
            lifeTime=0.0
        )



            

    def visualize_lidar(self):
        if (not self.render_mode) or (not self.use_lidar):
            return
        if self._last_env_ob is None:
            return

        for lid in getattr(self, "_lidar_debug_line_ids", []):
            try:
                p.removeUserDebugItem(lid)
            except Exception:
                pass
        self._lidar_debug_line_ids = []

        if getattr(self, "_lidar_debug_text_id", None) is not None:
            try:
                p.removeUserDebugItem(self._lidar_debug_text_id)
            except Exception:
                pass
            self._lidar_debug_text_id = None

        base_state = self._last_env_ob[0]["robot_0"]["joint_state"]
        robot_xy = np.array(base_state["position"][:2], dtype=np.float32)
        robot_yaw = float(self.get_robot_yaw_wf())
        start = [float(robot_xy[0]), float(robot_xy[1]), float(self.lidar_height)]

        angles, fracs, dists, hits, min_dist, min_front = self.compute_lidar_scan(robot_xy, robot_yaw)

        for i in range(self.lidar_num_rays):
            end = hits[i, :].tolist()
            dist = float(dists[i])
            frac = float(fracs[i])

            if frac >= 0.999:
                color = [0.0, 1.0, 0.0]
            else:
                color = [1.0, 0.0, 0.0] if dist < 0.30 else [1.0, 0.6, 0.0]

            line_id = p.addUserDebugLine(start, end, lineColorRGB=color, lineWidth=1.5, lifeTime=0.0)
            self._lidar_debug_line_ids.append(line_id)

        txt = f"LiDAR min={min_dist:.3f} | front min={min_front:.3f} | ignore={list(self.lidar_ignore_ids)}"
        self._lidar_debug_text_id = p.addUserDebugText(
            txt,
            [start[0], start[1], start[2] + 0.35],
            textColorRGB=[1, 1, 1],
            textSize=1.2,
            lifeTime=0.0
        )



# ============================================================
#                    MAP MAKER
# ============================================================

class MapMaker:
    def __init__(self):
        self.obstacles = []

    def add_box(self, pos, size, yaw=0.0, rgba=(0.3, 0.3, 0.3, 1.0)):
        lx, ly, lz = size
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[lx/2, ly/2, lz/2])
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[lx/2, ly/2, lz/2], rgbaColor=rgba)
        
        box_id = p.createMultiBody(
            0, collision, visual, 
            basePosition=pos, 
            baseOrientation=p.getQuaternionFromEuler([0,0,yaw])
        )
        
        # Group 3 = (Bit 1 | Bit 2). 
        #   - Bit 1: Standard Physics (Collides with Robot/Table)
        #   - Bit 2: Lidar Visibility (Collides with Rays)
        p.setCollisionFilterGroupMask(box_id, -1, collisionFilterGroup=3, collisionFilterMask=1)
        
        self.obstacles.append(box_id)
        return box_id

    def create_cylinder(self, pos_xy, radius=0.5, height=1.0, rgba=(0.8, 0.2, 0.2, 1.0)):
        """
        Creates a vertical cylinder at the specified (x,y).
        Used by the Geometric Navigation Env to spawn the known obstacle.
        """
        x, y = pos_xy
        z = height / 2.0  # Center of cylinder at z-midpoint
        
        # 1. Collision Shape
        col_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
        
        # 2. Visual Shape (PyBullet visual cylinders take 'length' instead of 'height')
        vis_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=rgba)
        
        # 3. Spawn
        obs_id = p.createMultiBody(
            baseMass=0, # Static
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=[x, y, z],
            baseOrientation=[0, 0, 0, 1]
        )
        
        # 4. Set Collision Filter (Same as Box)
        # Ensure it collides with the robot (Group 3, Mask 1)
        p.setCollisionFilterGroupMask(obs_id, -1, collisionFilterGroup=3, collisionFilterMask=1)

        self.obstacles.append(obs_id)
        return obs_id

    def create_map_1(self):
        """
        Creates Map 1: A single gray wall back of the robot.
        """
        self.add_box(
            pos=(0.0, 2.0, 0.1), 
            size=(0.5, 0.1, 2.0), 
            rgba=(0.3, 0.3, 0.3, 1.0) # Gray
        )