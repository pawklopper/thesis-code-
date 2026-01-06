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
   - computes reward (including geometric obstacle avoidance),
   - assembles observation vector (ego-centric),
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
#     Imports MapMaker to physically spawn the obstacle
from rl_env.obstacles_and_sensing import MapMaker


class AlbertTableEnv(gym.Env):
    """
    Simplified Gymnasium environment focusing on table motion + robot/table heading alignment.
    
    Navigational Strategy: "Geometric/Vector Navigation" with Rectangular Awareness
    -------------------------------------------------------------------------------
    Instead of using raw Lidar rays or simple radius approximations, we feed the robot 
    precise vectors to the nearest point on the rectangular obstacle relative to its 
    own body frame.
    
    - Cos(obs_angle) > 0: Obstacle is in FRONT.
    - Cos(obs_angle) < 0: Obstacle is BEHIND.
    
    This allows the robot to learn bi-directional driving (towing vs pushing) 
    while navigating tightly around rectangular walls.
    """

    metadata = {"render_modes": ["human"]}

    # ---------------------------------------------------------------------
    #                           CONSTRUCTOR
    # ---------------------------------------------------------------------
    def __init__(self, render=False, max_steps=1200, goals=None):
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

        # --- OBSTACLE CONFIGURATION ---
        # We will dynamically retrieve these from MapMaker in reset()
        # obstacle_pos: Center (x, y)
        # obstacle_extents: Half-width, Half-height
        self.obstacle_pos = np.zeros(2, dtype=np.float32)      
        self.obstacle_extents = np.zeros(2, dtype=np.float32)  
        
        # The "Fear Zone" is now just a skin around the rectangle.
        # Robot starts getting penalized if it is within 0.5m of the WALL SURFACE.
        self.safety_margin = 1.2 

        # Action = robot velocity command: (v_linear, w_angular)
        self.action_space = spaces.Box(
            low=np.array([-0.5, -1.0], dtype=np.float32),
            high=np.array([0.5, 1.0], dtype=np.float32)
        )

        # Observation Space = 15 elements
        # 12 (Original State) + 3 (Ego-centric Obstacle Vector)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
        )

        # !!! CROSS-FILE CREATION
        #     Create the full impedance physics engine
        self.sim = AlbertTableImpedanceSim(render=self.render_mode)

        # For reward shaping: remember last goal distance
        self.prev_dist = None

        # Latest URDFenv observation (dict structure returned by URDFenv)
        self._last_env_ob = None

        # Toggle: include human/robot forces in observation?
        self.use_force_observations = True   

        if self.use_force_observations:
            print("OBSERVATION MODE: REAL FORCES INCLUDED")
        else:
            print("OBSERVATION MODE: FORCES ZEROED")

        self.last_tangent_idx = None


    # ---------------------------------------------------------------------
    #                         MATH HELPERS (BOX)
    # ---------------------------------------------------------------------

    def get_query_point(self, robot_pos, v_robot):
        """
        Standardizes the look-ahead logic for both Reward and Observation.
        """
        speed = np.linalg.norm(v_robot)
        # Look ahead 0.3m if moving, otherwise stay put
        if speed > 0.1:
            return robot_pos + (v_robot * 0.3)
        return robot_pos
    
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

    # ---------------------------------------------------------------------
    #                           DEBUG HELPER
    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    #                       DEBUG HELPER
    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    #                       DEBUG VISUALIZER
    # ---------------------------------------------------------------------
    def debug_flow_field(self, robot_pos, goal_pos, v_robot):
        # 1. Get the vectors (Debug Mode = True)
        query_pos = self.get_query_point(robot_pos, v_robot)
        v_final, v_goal, normal, v_slide = self.get_optimal_flow_vector(
            query_pos, goal_pos, v_robot, debug_mode=True
        )

        # 2. Setup Drawing
        p.removeAllUserDebugItems()
        Z = 0.5
        origin = [query_pos[0], query_pos[1], Z]

        # Helper to draw arrows
        def draw(vec, rgb, width, name):
            if np.linalg.norm(vec) < 0.01: return
            end = [origin[0] + vec[0], origin[1] + vec[1], Z]
            p.addUserDebugLine(origin, end, rgb, lineWidth=width, lifeTime=0)
            p.addUserDebugText(name, end, rgb, textSize=1.2, lifeTime=0)

        # 3. Draw The Logic
        draw(v_goal,   [0,0,1], 2, "Goal")    # BLUE: Where you want to go
        draw(normal,   [1,0,0], 2, "Normal")  # RED: The wall pushing back
        draw(v_slide,  [0,1,0], 4, "SLIDE")   # GREEN: The calculated tangent
        draw(v_final,  [1,0,1], 6, "RESULT")  # MAGENTA: The final blend

        print(f"--- DEBUG PAUSE (Dist: {np.linalg.norm(query_pos - self.obstacle_pos):.2f}) ---")
        time.sleep(0.5)
    # ---------------------------------------------------------------------
    #                       FLOW FIELD HELPER (UPDATED)
    # ---------------------------------------------------------------------
    def get_optimal_flow_vector(self, query_pos, goal_pos, v_robot=None, debug_mode=False):
        # --- 1. SETUP GOAL VECTOR ---
        vec_to_goal = goal_pos - query_pos
        dist_goal = np.linalg.norm(vec_to_goal)
        v_goal = vec_to_goal / dist_goal if dist_goal > 1e-3 else np.zeros(2)

        # --- 2. CALCULATE WALL NORMAL (The "Push Back") ---
        # This tells us exactly which direction the wall is facing relative to us.
        c = self.obstacle_pos
        e = self.obstacle_extents
        diff = query_pos - c
        
        # Check which axis we are overlapping with
        # overlap < 0 means we are "in front" of that face
        overlap = np.abs(diff) - e 
        
        # LOGIC: If we are closer to the side (X) than top/bottom (Y), normal is X.
        if overlap[0] > overlap[1]:
            normal = np.array([np.sign(diff[0]), 0.0]) # Pointing Left or Right
        else:
            normal = np.array([0.0, np.sign(diff[1])]) # Pointing Up or Down

        # --- 3. THE SLIDE DECISION (The Core Logic) ---
        
        # "push_factor": Are we driving INTO the wall? 
        # Negative = We are pushing into the wall.
        # Positive = We are already driving away from it.
        push_factor = np.dot(v_goal, normal)
        
        v_slide = np.zeros(2) # Placeholder
        
        if push_factor < 0:
            # === WE NEED TO SLIDE ===
            # We calculate two perpendicular tangents to the Normal.
            # Math: To rotate vector (x, y) 90 degrees, we swap and negate one.
            
            # Option A: Rotate 90 deg Clockwise
            # (x, y) -> (y, -x)
            t_cw = np.array([normal[1], -normal[0]])
            
            # Option B: Rotate 90 deg Counter-Clockwise
            # (x, y) -> (-y, x)
            t_ccw = np.array([-normal[1], normal[0]])
            
            # DECISION: Which tangent is closer to our Goal?
         
            score_cw = np.dot(t_cw, v_goal)
            score_ccw = np.dot(t_ccw, v_goal)

            if self.step_count % 20 == 0:
                print(f"score cw: {score_cw} | score_ccw: {score_ccw}")

            if score_cw > score_ccw:
                v_slide = t_cw
                decision_str = "Clockwise Slide"
            else:
                v_slide = t_ccw
                decision_str = "Counter-Clockwise Slide"
            
            if debug_mode:
                print(f" [LOGIC] Wall Normal: {normal}")
                print(f" [LOGIC] Push Factor: {push_factor:.2f} (Negative means Crash)")
                print(f"  [LOGIC] Decision: {decision_str}")

        else:
            # === NO SLIDE NEEDED ===
            v_slide = v_goal # Just keep going
            if debug_mode: print("   [LOGIC] Safe: Moving away from wall.")

        # --- 4. BLENDING (Final Result) ---
        # Calculate distance to determine how much to respect the slide
        # (Simplified for clarity)
        dist_surf = self.get_surface_dist_to_rect(query_pos, c, e, 0.4)
        
        # 0.0 (at wall) -> 1.0 (far away)
        safety_alpha = np.clip(dist_surf / 1.2, 0.0, 1.0)
        
        # Repulsion: Pure "push away" vector to prevent grazing
        v_repulse = normal * (1.0 - safety_alpha) 

        # Final Blend
        v_final = (v_goal * safety_alpha) + (v_slide * (1.0 - safety_alpha)) + v_repulse
        
        # Normalize
        norm_f = np.linalg.norm(v_final)
        if norm_f > 1e-6: v_final /= norm_f

        if debug_mode:
            return v_final, v_goal, normal, v_slide
            
        return v_final
        
    def get_obstacle_corners_obs(self, robot_pos, robot_yaw):
        """
        Returns a flat array of 8 floats: [x1, y1, x2, y2, x3, y3, x4, y4]
        These are the coordinates of the 4 obstacle corners RELATIVE to the robot.
        """
        # 1. Define the 4 corners in World Frame
        c = self.obstacle_pos
        e = self.obstacle_extents # (half_width, half_height)
        
        # Corner offsets: (++, -+, --, +-)
        offsets = [
            np.array([ e[0],  e[1]]),
            np.array([-e[0],  e[1]]),
            np.array([-e[0], -e[1]]),
            np.array([ e[0], -e[1]])
        ]
        
        world_corners = [c + off for off in offsets]
        
        obs_vecs = []
        
        # 2. Rotation Matrix for Ego-centric transform
        # We want to rotate World -> Robot
        # R^T * (P_world - P_robot)
        cos_a = np.cos(robot_yaw)
        sin_a = np.sin(robot_yaw)
        
        for wc in world_corners:
            dx = wc[0] - robot_pos[0]
            dy = wc[1] - robot_pos[1]
            
            # Rotate
            rx =  dx * cos_a + dy * sin_a
            ry = -dx * sin_a + dy * cos_a
            
            obs_vecs.append(rx)
            obs_vecs.append(ry)
            
        return np.array(obs_vecs, dtype=np.float32)

    def get_closest_point_on_rect(self, point, rect_center, rect_extents):
        """
        Calculates the closest point ON THE RECTANGLE to the given point.
        """
        local_point = point - rect_center
        closest_local = np.clip(local_point, -rect_extents, rect_extents)
        closest_world = closest_local + rect_center
        return closest_world

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


    # ---------------------------------------------------------------------
    #                         OBSERVATION BUFFER
    # ---------------------------------------------------------------------
    def _get_obs(self):
        # 1. STATE RETRIEVAL
        table_xy, table_yaw, tv_xy, _ = self.sim.get_table_state_world()
        goal_xy = self.goal[:2]
        dx_t, dy_t = goal_xy - table_xy

        base_state = self._last_env_ob[0]["robot_0"]["joint_state"]
        robot_pos = np.array(base_state["position"][:2])
        
        # !!! FIX 1: Apply Correction Here !!!
        robot_yaw = self.get_robot_yaw_wf()

        v_r = base_state["velocity"][0] 
        w_r = base_state["velocity"][2] 

        # Now this calculation is correct (X is Forward)
        v_robot_vec = np.array([v_r * np.cos(robot_yaw), v_r * np.sin(robot_yaw)])

        # 2. FLOW VECTOR
        query_pos = self.get_query_point(robot_pos, v_robot_vec)
        v_flow_world = self.get_optimal_flow_vector(query_pos, goal_xy, v_robot_vec)

        # TRANSFORM TO ROBOT LOCAL FRAME
        cos_a = np.cos(robot_yaw)
        sin_a = np.sin(robot_yaw)

        # Standard Rotation Matrix (World -> Body)
        flow_x_local =  v_flow_world[0] * cos_a + v_flow_world[1] * sin_a
        flow_y_local = -v_flow_world[0] * sin_a + v_flow_world[1] * cos_a

        # 3. FORCES & CORNERS
        if self.use_force_observations:
            Fh_x, Fh_y = self.sim.last_Fh_xy
            Fr_x, Fr_y = 0.0, 0.0 
        else:
            Fh_x = Fh_y = Fr_x = Fr_y = 0.0

        # Pass corrected yaw here so corners are rotated correctly!
        corner_obs = self.get_obstacle_corners_obs(robot_pos, robot_yaw)

        base_obs = np.array([
            dx_t, dy_t, tv_xy[0], tv_xy[1], v_r, w_r,
            flow_x_local, flow_y_local,
            Fh_x, Fh_y, Fr_x, Fr_y,
        ], dtype=np.float32)
        
        return np.concatenate([base_obs, corner_obs])


    # ---------------------------------------------------------------------
    #                            REWARD
    # ---------------------------------------------------------------------
    def compute_reward(self, dist, prev_dist):
        # --- 1. GATHER DATA ---
        base_state = self._last_env_ob[0]["robot_0"]["joint_state"]
        
        # USE THE HELPER (Standardize Yaw to World Frame)
        # This ensures 0 radians = World X axis
        robot_yaw = self.get_robot_yaw_wf()
        robot_pos = np.array(base_state["position"][:2])
        
        vx = base_state["velocity"][0] # Local X velocity (Forward/Back)
        vy = base_state["velocity"][1] # Local Y velocity (Slip)
        
        # Calculate Speed magnitude
        v_speed = np.sqrt(vx**2 + vy**2)

        # 1. Robot Nose Vector (Corrected Global Frame)
        # This vector points out of the front of the robot
        robot_nose_vec = np.array([np.cos(robot_yaw), np.sin(robot_yaw)])
        
        # 2. Movement Vector (Global Frame)
        # We derive this from the nose because URDF velocity is local.
        # If vx > 0, we move with nose. If vx < 0, we move against nose.
        if abs(vx) > 0.01:
            v_move_global = robot_nose_vec * np.sign(vx)
        else:
            v_move_global = np.zeros(2)

        goal_xy = self.goal[:2]

        # --- 2. FLOW FIELD REWARD (The Main Driver) ---
        # Get the "Magenta Arrow"
        v_optimal = self.get_optimal_flow_vector(robot_pos, goal_xy)
        
        # PART A: COMPASS REWARD (Orientation - Bidirectional)
        # ----------------------------------------------------
        # 1. Calculate alignment (Cos theta)
        # Range: -1.0 (Opposite) to 1.0 (Aligned)
        alignment_cos = np.dot(robot_nose_vec, v_optimal)

        # 2. Make it Bidirectional
        # We take the absolute value because driving backwards (towing) 
        # is just as valid as driving forwards.
        # Range: 0.0 (Perpendicular/Bad) to 1.0 (Parallel/Good)
        axial_alignment = abs(alignment_cos)

        # 3. Convert to Angular Error (Radians)
        # This effectively recreates 'heading_error_bi' from your old code.
        # arccos(1.0) = 0.0 rad (No error)
        # arccos(0.0) = 1.57 rad (Max error)
        # We clip to ensure numerical stability.
        axial_alignment = np.clip(axial_alignment, 0.0, 1.0)
        heading_error_rad = np.arccos(axial_alignment)

        # 4. Apply Squared Penalty
        # "Punish the robot for not aligning"
        # Logic: - (weight * error)^2
        # This creates a steep penalty for being perpendicular, and near-zero penalty for being aligned.
        w_hp = 2.0  # Weight heuristic (similar to your old 3.5)
        r_compass = - (w_hp * heading_error_rad) ** 2

        # PART B: PROPULSION REWARD (Motion)
        # Reward moving along the vector.
        # We use v_move_global so driving backwards (towing) counts positively
        # as long as the movement direction aligns with the arrow.
        move_alignment = 0.0
        if np.linalg.norm(v_move_global) > 1e-6:
             move_alignment = np.dot(v_move_global / np.linalg.norm(v_move_global), v_optimal)
        
        # Weight: High (5.0). Driving fast in the right direction is the ultimate goal.
        # If move_alignment is negative (wrong way), this becomes a heavy penalty.

        k_flow = 5.0
        r_propulsion = k_flow * move_alignment * v_speed
        
        # setting r_propulsion to zero
        r_propulsion = 0.0


        flow_reward = r_compass + r_propulsion

        # if self.step_count % 5 == 0: 
        #     print(f"move_alignment: {move_alignment} | v_speed: {v_speed}")
        #     print(f"r_compass: {r_compass} | r_propulsion: {r_propulsion} | flow_reward {flow_reward}")

        # --- 3. PROGRESS REWARD (The Milestone) ---
        # Reward getting closer to the goal.
        # We keep this weight moderate so it doesn't fight the flow field near walls.
        kpr = 4.0 
        progress_reward = kpr * (prev_dist - dist) / 0.01


        # --- 5. OBSTACLES & COLLABORATION ---
        # Remove the static distance penalty (-dist/10) as it's redundant with time_penalty
        distance_penalty = - dist / 10
        
        # Collaboration (Keep existing)
        table_xy, _, _, _ = self.sim.get_table_state_world()
        dir_to_goal = goal_xy - table_xy
        dir_to_goal /= (np.linalg.norm(dir_to_goal) + 1e-6)

        #DO NOT FORGET TO CHANGE TO WORK WITH THE OBSTACLE THINGS
        collaboration_reward = self.compute_collaboration_reward(dir_to_goal)

        # Obstacle
        obstacle_reward, is_crash = self._calc_obstacle_penalty(robot_pos)

        # --- TOTAL ---
        total_reward = (
            flow_reward + 
            progress_reward + 
            collaboration_reward + 
            obstacle_reward + distance_penalty 
        )
        
        # Optional: Clip reward to keep training stable (e.g., -10 to +10)
        total_reward = np.clip(total_reward, -20.0, 20.0)

        return total_reward, progress_reward, flow_reward, distance_penalty, collaboration_reward, obstacle_reward, is_crash
    

    def _calc_obstacle_penalty(self, robot_pos):
        ROBOT_RADIUS = 0.4
        
        # Surface Distance
        dist_surface = self.get_surface_dist_to_rect(
            robot_pos, self.obstacle_pos, self.obstacle_extents, radius=ROBOT_RADIUS
        )
        
        # if self.step_count % 5 ==0: 
        #     print("dist to wall", dist_surface)
        # 1. CRASH (Hard Fail)
        if dist_surface <= 0.25:
            return -1000.0, True

        reward = 0.0
        
        # 2. PROXIMITY PENALTY (Exponential)
        # Only punish if REALLY close (< 0.5m). 
        # The Flow Field handles the 0.5m -> 1.5m zone gently.
        danger_zone = 0.5
        
        if dist_surface < danger_zone:
            # 0.0 at edge, 1.0 at wall
            penetration = (danger_zone - dist_surface) / danger_zone
            
            # Simple exponential barrier
            # -25.0 max penalty
            reward = - 2.0 * (penetration ** 2)
            
            # Optional: Add the "TTC" (Time To Collision) penalty here if you
            # find the robot is still driving too fast into the wall.
            # But usually, the Flow Field reward (alignment * speed) 
            # naturally discourages crashing because v_opt points AWAY from wall.

        return reward, False


    def compute_collaboration_reward(self, dir_to_goal):
        """
        Collaboration reward based on:
        - human force direction (intent)
        - robot base velocity direction (motion)
        - only rewarded when human acts toward the goal (helpfulness)
        """

        # Human force (world frame)
        Fh = np.array(self.sim.last_Fh_xy, dtype=np.float32)
        Fh_norm = np.linalg.norm(Fh)

        if Fh_norm < 1e-6:
            return 0.0   # no meaningful human input

        # 1) HUMAN HELPFULNESS (are they pushing toward the goal?)
        Fh_goal = float(np.dot(Fh, dir_to_goal))  # signed projection
        helpfulness = max(0.0, Fh_goal / (Fh_norm + 1e-6))

        if helpfulness <= 1e-6:
            return 0.0   # human is not helping -> no collaboration shaping

        # 2) ROBOT BASE VELOCITY IN WORLD FRAME
        base_state = self._last_env_ob[0]["robot_0"]["joint_state"]
        vx = base_state["velocity"][0]
        vy = base_state["velocity"][1]
        v_world_vec = np.array([vx, vy])

        # Get Robot Heading Vector (The "Nose" of the robot)
        robot_yaw=self.get_robot_yaw_wf()

        heading_unit = np.array([np.cos(robot_yaw), np.sin(robot_yaw)], dtype=np.float32)

        # Calculate Signed Speed (Dot Product)
        v_r = float(np.dot(v_world_vec, heading_unit))
        v_robot_xy = heading_unit * v_r
        v_norm = np.linalg.norm(v_robot_xy)

        if v_norm < 1e-6:
            return 0.0  # robot not moving meaningfully

        # 3) ALIGNMENT BETWEEN HUMAN FORCE AND ROBOT MOTION
        alignment = float(
            np.dot(Fh, v_robot_xy) /
            ((Fh_norm * v_norm) + 1e-6)
        )

        # 4) FINAL COLLABORATION SCORE
        k_collab = 1.0 
        collab_reward = k_collab * helpfulness * alignment
        return float(collab_reward)

    
    def apply_progressive_leash(self, action):
        v_cmd, w_cmd = action
        
        # ... [Keep existing Config & State Reading] ...
        ee_pos = np.array(p.getLinkState(self.sim.albert_id, self.sim.ee_idx)[0])[:2]
        h_pos  = np.array(p.getLinkState(self.sim.table_id, self.sim.goal_link_idx)[0])[:2]
        robot_pos, robot_orn = p.getBasePositionAndOrientation(self.sim.albert_id)
        
        # !!! FIX 3: Apply Correction Here !!!
        robot_yaw = self.get_robot_yaw_wf()

        # Vectors
        r_arm = ee_pos - np.array(robot_pos[:2])
        vec_to_table = h_pos - ee_pos
        current_stretch = float(np.linalg.norm(vec_to_table))

        # --- LOGIC ---
        START_BRAKE_RADIUS = 0.2
        MAX_BRAKE_RADIUS   = 0.4 
        
        if current_stretch < START_BRAKE_RADIUS:
            return action
        
        fraction = (current_stretch - START_BRAKE_RADIUS) / (MAX_BRAKE_RADIUS - START_BRAKE_RADIUS)
        velocity_scale = 1.0 - np.clip(fraction, 0.0, 1.0)

        # Use Standard Heading (Cos, Sin) now that yaw is corrected
        heading_unit = np.array([np.cos(robot_yaw), np.sin(robot_yaw)]) 
        
        # ... [Rest of function is identical] ...
        # (The math below works correctly now because heading_unit is truly "Forward")
        v_linear_vec = heading_unit * v_cmd
        linear_proj = float(np.dot(v_linear_vec, vec_to_table))
        
        v_final = v_cmd
        if linear_proj < 0: 
            v_final = v_cmd * velocity_scale

        torque_direction = (r_arm[0] * vec_to_table[1]) - (r_arm[1] * vec_to_table[0])
        w_final = w_cmd
        if torque_direction > 0.01 and w_cmd > 0.01:
            w_final = w_cmd * velocity_scale
        elif torque_direction < -0.01 and w_cmd < -0.01:
            w_final = w_cmd * velocity_scale

        return np.array([v_final, w_final])


    # ---------------------------------------------------------------------
    #                               STEP
    # ---------------------------------------------------------------------
    def step(self, action, Fh_override=None):
        """
        Single RL step.

        !!! ORDER OF OPERATIONS
        1) Read robot base pose from URDFenv
        2) sim.impedance_step()                   (robot arm ↔ table physics)
        3) HUMAN FORCE logic
        4) apply_progressive_leash()              (safety)
        5) URDFenv step(action)                   (robot base movement)
        6) compute reward (progress + obstacle)
        7) check termination/truncation
        8) build and return observation
        """

        # 1) ROBOT BASE STATE
        base_state = self._last_env_ob[0]["robot_0"]["joint_state"]
        #robot_yaw = base_state["position"][2]
        robot_xy = np.array(base_state["position"][:2])

        # 2) ROBOT IMPEDANCE STEP
        self.sim.impedance_step(self.goal, robot_xy)

        # 3) HUMAN INTERACTION
        if self._external_force_active and (Fh_override is not None):
            # external human force (e.g. from PS5)
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

        # 4) READ DRAG & MODULATE ROBOT (LEASH)
        v_final, w_final = self.apply_progressive_leash(action)

        # 5) APPLY ROBOT BASE VELOCITY COMMANDS
        full = np.zeros(self.sim.env.n(), dtype=float)
        full[0], full[1] = v_final, w_final 

        ob = self.sim.env.step(full)           
        self._last_env_ob = ob
        self.sim.enforce_rigid_arm()

        # 6) GOAL DISTANCE + REWARD
        table_xy, table_yaw, _, _ = self.sim.get_table_state_world()
        goal_xy = self.goal[:2]
        dx_t, dy_t = goal_xy - table_xy
        dist = float(np.linalg.norm([dx_t, dy_t]))

        # heading_error_bi, heading_error = self.compute_robot_heading_error(
        #     dx_t, dy_t, robot_yaw
        # )
        # heading_diff = wrap_angle(table_yaw - robot_yaw)



        # Reward Calculation
        reward, progress_reward, flow_reward, distance_penalty, collaboration_reward, obstacle_reward, is_crash = (
            self.compute_reward(dist, self.prev_dist)
        )

        # check vectors
        base_state = self._last_env_ob[0]["robot_0"]["joint_state"]
        robot_pos = np.array(base_state["position"][:2])
        vx = base_state["velocity"][0]
        vy = base_state["velocity"][1]
        v_robot = np.array([vx, vy])

        # Print debug every 20 steps so you can read it
        if self.step_count % 20 == 0:
            self.debug_flow_field(robot_pos, self.goal[:2], v_robot)

        self.prev_dist = dist



        # 7) TERMINATION & TRUNCATION
        terminated = False
        
        # Goal Reached
        if dist < 0.4:
            print(f"[REACHED GOAL] Reached goal at step: {self.step_count}")
            reward += 50
            terminated = True
        
        # Collision
        if is_crash:
            print(f"[COLLISION] Hit obstacle at step: {self.step_count}")
            # Note: A large negative penalty (-100) was already returned by compute_reward
            terminated = True 

        truncated = self.step_count >= self.max_steps
        self.step_count += 1
        
        if self.render_mode:
            time.sleep(self.sim.dt)

        # 8) INFO DICTIONARY
        info = {
            "dist_table_to_goal": dist,
            "total_reward": reward,
            "progress_reward": progress_reward,
            "flow_reward": flow_reward, 
            "distance_penalty": distance_penalty,
            "obstacle_penalty": obstacle_reward,
            "collaboration_reward": collaboration_reward,
            "human_action": self.sim.human_action,
        }

        return self._get_obs(), reward, terminated, truncated, info


    # ---------------------------------------------------------------------
    #                                RESET
    # ---------------------------------------------------------------------
    def reset(self, *, seed=None, options=None, Fh_override=None):
        super().reset(seed=seed)
        self._external_force_active = Fh_override is not None
        self.step_count = 0

        # Sample goal
        self.goal = self.goals[np.random.randint(len(self.goals))]
        print(f"Goal is: {self.goal}")

        # ===================================================
        # FIRST-EVER EPISODE
        # ===================================================
        if self.sim.env is None:
            print("=== FIRST EPISODE: full initialization ===")
            ob0 = self.sim.create_environment()
            self.sim.albert_id = self.sim.get_albert_body_id()
            self.sim.load_table()

            # --- DYNAMIC RECTANGLE LOADING ---
            mm = MapMaker()
            mm.create_map_1()  # Create the wall using MapMaker.add_box

            if len(mm.obstacles) > 0:
                obs_id = mm.obstacles[-1]

                # 1. Get Position (Center)
                pos, _ = p.getBasePositionAndOrientation(obs_id)
                self.obstacle_pos = np.array(pos[:2], dtype=np.float32)

                # 2. Get Extents (Width/Height)
                # p.getAABB returns (min_x, min_y, min_z) and (max_x, max_y, max_z)
                aabb_min, aabb_max = p.getAABB(obs_id)
                
                # Full Size = Max - Min
                # Half Extents = (Max - Min) / 2
                full_size = np.array(aabb_max) - np.array(aabb_min)
                self.obstacle_extents = (full_size[:2] / 2.0).astype(np.float32)

                print(f"!!! OBSTACLE DETECTED at {self.obstacle_pos} !!!")
                print(f"!!! DIMENSIONS: {full_size[0]:.2f}m x {full_size[1]:.2f}m !!!")
            
            else:
                # Fallback to a default small box if map is empty
                self.obstacle_pos = np.array([0.0, -2.0], dtype=np.float32)
                self.obstacle_extents = np.array([0.5, 0.1], dtype=np.float32)

            # Settle
            zero = np.zeros(self.sim.env.n())
            self.sim.env.step(zero)
            self.sim.set_arm_initial_pose()
            self._last_env_ob = ob0 
            
        # ===================================================
        # LATER EPISODES
        # ===================================================
        else:
            print("=== LATER EPISODE: reset positions ===")
            
            # Reset Robot Base
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

            # Reset Table
            self.sim.reset_table()
            
            # Fresh URDFenv Observation
            raw = self.sim.env._get_ob()
            self._last_env_ob = [raw]

        # ===================================================
        # COMMON SETTLE
        # ===================================================
        self.sim.create_connection_impedance()

        # Impedance settle
        base_state = self._last_env_ob[0]["robot_0"]["joint_state"]
        robot_xy = np.array(base_state["position"][:2])
        self.sim.impedance_step(self.goal, robot_xy)

        # URDFenv settle
        zero = np.zeros(self.sim.env.n())
        self._last_env_ob = self.sim.env.step(zero)

        # Initialize Previous Distance for Delta Reward
        table_xy, _, _, _ = self.sim.get_table_state_world()
        self.prev_dist = float(np.linalg.norm(self.goal - table_xy))

        print("allow everything to settle")
        for _ in range(100):
            p.stepSimulation()
            time.sleep(0.01)

        self.last_tangent_idx = None

        return self._get_obs(), {}


    # ---------------------------------------------------------------------
    #                             HELPERS
    # ---------------------------------------------------------------------
    def compute_robot_heading_error(self, dx, dy, robot_yaw):
        """
        Converts dx, dy into a world-frame angle,
        converts robot yaw into same reference,
        computes smallest signed angular error.
        """
        angle_to_goal_world = np.arctan2(dy, dx)
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
        if self.sim.env is not None:
            self.sim.env.close()
        print("Environment closed.")