#!/usr/bin/env python3
"""
obstacles_and_sensing.py — Obstacles and Sensing
----------------------------------------------------
- Includes SimpleLidar (for raycasting if needed).
- Includes MapMaker with create_cylinder for the geometric navigation task.
"""

import numpy as np
import pybullet as p

# ============================================================
#                  CUSTOM LIDAR CLASS (Filtered)
# ============================================================

class SimpleLidar:
    def __init__(self, robot_id, max_range=6.0, num_rays=8):
        self.robot_id = robot_id
        self.max_range = max_range
        self.num_rays = num_rays 

    def get_readings(self):
        """
        Casts 8 rays: F, FL, L, BL, B, BR, R, FR
        """
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        robot_x, robot_y, robot_z = pos
        _, _, robot_yaw = p.getEulerFromQuaternion(orn)

        angles = np.linspace(-np.pi, np.pi, self.num_rays, endpoint=False)
        
        sensor_height = 0.5 
        ray_starts = []
        ray_ends = []
        
        for angle in angles:
            # We need to offset because your "Front" is -Y (-pi/2)
            global_angle = robot_yaw + angle - (np.pi/2)

            c = np.cos(global_angle)
            s = np.sin(global_angle)

            start = [robot_x, robot_y, sensor_height]
            end = [
                robot_x + self.max_range * c,
                robot_y + self.max_range * s,
                sensor_height
            ]
            ray_starts.append(start)
            ray_ends.append(end)

        # Mask=2 (Visible to Lidar only logic if using filters)
        results = p.rayTestBatch(ray_starts, ray_ends, collisionFilterMask=2)

        dists = []
        for res in results:
            if res[2] < 1.0:
                dists.append(res[2] * self.max_range)
            else:
                dists.append(self.max_range)
                
        return np.array(dists, dtype=np.float32)


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
            pos=(0.0, 2.5, 0.1), 
            size=(0.5, 0.5, 1.0), 
            rgba=(0.3, 0.3, 0.3, 1.0) # Gray
        )