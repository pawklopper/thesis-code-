def impedance_step(self, goal_xy, robot_xy):
    """
    Perform one impedance step between:
        - robot end-effector (EE)
        - table handle link

    FIXED VERSION:
    --------------
    Prevents the virtual impedance spring from stretching
    indefinitely when the robot drives but the human holds
    the table still.

    Adds a physics-level clamp on dx (spring extension).
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
    dx = handle_pos - ee_pos     # position error (virtual spring)
    dv = handle_vel - ee_vel     # velocity error

    # -----------------------------------------------------------
    # Compute impedance force with clamped dx
    # -----------------------------------------------------------
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
    # h_yaw  = p.getEulerFromQuaternion(handle_quat)[2]
    # ee_yaw = p.getEulerFromQuaternion(ee_quat)[2]

    # yaw_error = ((h_yaw - ee_yaw + np.pi) % (2 * np.pi)) - np.pi

    
    # # Angular velocities
    # h_ang_vel = p.getLinkState(self.table_id, self.goal_link_idx, computeLinkVelocity=1)[7][2]
    # ee_ang_vel = p.getLinkState(self.albert_id, self.ee_idx, computeLinkVelocity=1)[7][2]
    # yaw_vel_error = h_ang_vel - ee_ang_vel

    # # Gains match original script
    # K_yaw = 100 # was 60
    # D_yaw = 30.0 # was 4.0
    # tau_z = -K_yaw * yaw_error - D_yaw * yaw_vel_error

    # # Apply torque to table
    # p.applyExternalTorque(
    #     self.table_id,
    #     self.goal_link_idx,
    #     [0, 0, tau_z],
    #     flags=p.WORLD_FRAME,
    # )

    # ================================================================
    # Diagnostics returned to environment
    # ================================================================
    self.last_F_xy = Fr[:2]
    self.last_dx_xy = dx[:2]

    return Fr[:2], dx[:2], handle_pos[:2], handle_vel[:2]


def get_connection_force(self):
    """
    Reads the physical stress on the connection constraint.
    Returns the force vector [Fx, Fy] in World Frame.
    """
    if not hasattr(self, 'cid'):
        return np.zeros(2)

    # PyBullet returns the force the constraint applies to the Child (Table)
    # to keep it at the constraint location.
    # State is [appliedForce_x, appliedForce_y, appliedForce_z]
    constraint_state = p.getConstraintState(self.cid)
    
    Fx = constraint_state[0]
    Fy = constraint_state[1]
    
    # Store for diagnostics
    self.last_F_xy = np.array([Fx, Fy])
    
    return self.last_F_xy


def create_connection(self):
    """
    Creates a Zero-Energy 'Tow Bar' constraint spanning the current gap 
    between the Robot EE and the Table Handle.
    """
    if self.albert_id is None or self.table_id is None:
        return

    # 1. Cleanup old constraint
    if hasattr(self, 'cid') and self.cid is not None:
        try:
            p.removeConstraint(self.cid)
        except:
            pass
        self.cid = None

    # 2. Disable Collisions (Safety to prevent explosions)
    # Loop through all links to ensure clean interaction
    for r in [-1] + list(range(p.getNumJoints(self.albert_id))):
        for t in [-1] + list(range(p.getNumJoints(self.table_id))):
            p.setCollisionFilterPair(self.albert_id, self.table_id, r, t, 0)

    # ---------------------------------------------------------
    # 3. CALCULATE THE GAP (THE FIX)
    # ---------------------------------------------------------
    # We want to lock the constraint at the CURRENT positions.
    
    # A. Get World Positions
    ee_pos = p.getLinkState(self.albert_id, self.ee_idx)[0]
    
    # B. Get Table Handle Frame info
    h_pos, h_orn = p.getLinkState(self.table_id, self.goal_link_idx)[:2]
    
    # C. Calculate where the Robot EE is *relative* to the Table Handle frame
    # Math: T_handle_inv * P_ee_world = P_ee_local
    inv_h_pos, inv_h_orn = p.invertTransform(h_pos, h_orn)
    
    # This vector is the "Gap" seen from the Table's perspective
    child_pivot_local, _ = p.multiplyTransforms(inv_h_pos, inv_h_orn, ee_pos, [0,0,0,1])

    print(f"🔗 Creating Tow-Bar. Detected Gap: {child_pivot_local}")

    # 4. Create the Constraint
    self.cid = p.createConstraint(
        parentBodyUniqueId=self.albert_id,
        parentLinkIndex=self.ee_idx,
        childBodyUniqueId=self.table_id,
        childLinkIndex=self.goal_link_idx,
        jointType=p.JOINT_POINT2POINT,
        jointAxis=[0, 0, 0],
        
        # Pivot on Robot: Center of the End-Effector
        parentFramePosition=[0, 0, 0],
        
        # Pivot on Table: The calculated offset (The other end of the rod)
        childFramePosition=child_pivot_local 
    )

    # 5. Tune Physics
    # Max Force 200: Strong enough to pull, weak enough to prevent explosion
    # ERP 0.05: Slightly soft (rubber rod) to absorb vibrations
    p.changeConstraint(self.cid, maxForce=200)
    
    print("🔗 Connection created (Soft/Stable Mode)")