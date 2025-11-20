import numpy as np
import pybullet as p
import pygame

class PS5ImpedanceController:
    """
    Interprets Joystick input as moving a virtual 'Ghost Hand'.
    Applies a spring force between this Ghost Hand and the Table Handle.
    
    Includes a 'Leash' mechanism so the ghost cannot move infinitely far
    from the table handle.
    """
    def __init__(self, table_id, link_idx):
        self.table_id = table_id
        self.link_idx = link_idx
        
        # Initialize PyGame for Joystick
        pygame.init()
        pygame.joystick.init()
        
        if pygame.joystick.get_count() > 0:
            self.js = pygame.joystick.Joystick(0)
            self.js.init()
            print(f"✅ PS5 Controller Connected: {self.js.get_name()}")
        else:
            print("⚠️  No Controller Found! Human force will be zero.")
            self.js = None

        # --- TUNING PARAMETERS ---
        self.Kp = np.array([300.0, 300.0]) 
        self.Dp = np.array([60.0, 60.0])
        self.GHOST_SPEED = 3.0  # m/s (How fast the ghost moves)
        
        # --- NEW: TETHER RADIUS ---
        self.MAX_RADIUS = 0.5   # Meters (The leash length)
        
        # Initialize Ghost at current handle position
        start_pos = p.getLinkState(self.table_id, self.link_idx)[0]
        self.ghost_pos = np.array(start_pos[:2], dtype=np.float32)
        
        self.marker_id = None
        self.line_id = None  # To draw the spring line

    def step(self, dt=0.01):
        """
        Returns: F_world (3D np.array) [Fx, Fy, 0.0]
        """
        if self.js is None:
            return np.zeros(3)

        pygame.event.pump()
        
        # 1. READ INPUT (Velocity Command for Ghost)
        dx = self.js.get_axis(0)
        dy = -self.js.get_axis(1) # Invert Y for typical joystick Y-axis mapping
        
        if abs(dx) < 0.1: dx = 0.0
        if abs(dy) < 0.1: dy = 0.0
        
        # 2. ASSIGN INPUT DIRECTLY TO WORLD FRAME VELOCITY
        # This is the key change: we remove the rotation matrix calculation.
        # +X is now dx, +Y is now dy.
        vx_world = dx
        vy_world = dy
        
        # 3. INTEGRATE GHOST POSITION
        self.ghost_pos[0] += vx_world * self.GHOST_SPEED * dt
        self.ghost_pos[1] += vy_world * self.GHOST_SPEED * dt

        # -------------------------------------------------------
        # 4. CLAMP GHOST TO RADIUS (The "Leash" Logic)
        # -------------------------------------------------------
        # Get actual handle state
        state = p.getLinkState(self.table_id, self.link_idx, computeLinkVelocity=1)
        curr_pos = np.array(state[0][:2])
        curr_vel = np.array(state[6][:2])
        
        # Vector from Handle -> Ghost
        vec_to_ghost = self.ghost_pos - curr_pos
        dist = np.linalg.norm(vec_to_ghost)
        
        # If Ghost is too far, pull it back to the rim of the circle
        if dist > self.MAX_RADIUS:
            # Normalize vector and scale to max radius
            direction = vec_to_ghost / dist
            self.ghost_pos = curr_pos + direction * self.MAX_RADIUS

        # -------------------------------------------------------
        # 5. CALCULATE IMPEDANCE FORCE
        # -------------------------------------------------------
        # Recalculate pos_error after clamping
        pos_error = self.ghost_pos - curr_pos
        
        # Note: Impedance force calculation is always in the World frame
        # because curr_pos/curr_vel are in the World frame.
        force_2d = self.Kp * pos_error - self.Dp * curr_vel
        
        # Safety Clip
        force_2d = np.clip(force_2d, -40.0, 40.0)
        
        # 6. VISUALIZE
        self._draw_ghost(curr_pos)
        
        # Return 3D Force (Z=0)
        return np.array([force_2d[0], force_2d[1], 0.0], dtype=np.float32)

    def _draw_ghost(self, handle_pos_2d):
        ghost_3d = [self.ghost_pos[0], self.ghost_pos[1], 0.85]
        handle_3d = [handle_pos_2d[0], handle_pos_2d[1], 0.85]

        # Draw Sphere
        if self.marker_id is None:
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.04, rgbaColor=[0, 1, 0, 0.8])
            self.marker_id = p.createMultiBody(baseVisualShapeIndex=vis, basePosition=ghost_3d)
        else:
            p.resetBasePositionAndOrientation(self.marker_id, ghost_3d, [0,0,0,1])
            
        # Draw Line (The "Spring")
        if self.line_id is not None:
            p.removeUserDebugItem(self.line_id)
            
        self.line_id = p.addUserDebugLine(
            handle_3d, 
            ghost_3d, 
            lineColorRGB=[0, 1, 0], 
            lineWidth=2.0
        )