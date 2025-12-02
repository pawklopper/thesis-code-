#!/usr/bin/env python3
"""
map_1.py — Minimal environment loader for mobile_sim
----------------------------------------------------
Loads Albert + Table in a static PyBullet scene.
Fixed: Correctly attaches Lidar to the internal physics engine.
"""

import time
import numpy as np
import pybullet as p

# 1. Import Lidar 
from urdfenvs.sensors.lidar import Lidar 


from controllers.impedance_sim import AlbertTableImpedanceSim


# ============================================================
#                     LIDAR SETUP
# ============================================================

def activate_lidar(sim_instance, nb_rays=20, ray_length=5.0):
    """
    Attach a lidar to Albert's base link using the urdfenvs Lidar class.
    """
    link_name = "base_link" # Must match URDF link name

    # 1. Instantiate the Lidar
    sim_instance.lidar = Lidar(
        link_name=link_name,
        nb_rays=nb_rays,
        ray_length=ray_length,
        raw_data=True, 
        physics_engine_name='pybullet' 
    )

    # 2. CRITICAL: Inject the Physics Engine
    # The 'env' created inside AlbertTableImpedanceSim has the engine wrapper.
    if hasattr(sim_instance, 'env') and sim_instance.env is not None:
        
        # TRY/EXCEPT BLOCK TO HANDLE ATTRIBUTE NAMING
        try:
            # Most UrdfEnv implementations use _physics_engine (protected)
            if hasattr(sim_instance.env, '_physics_engine'):
                engine = sim_instance.env._physics_engine
            # Some might use physics_engine (public)
            elif hasattr(sim_instance.env, 'physics_engine'):
                engine = sim_instance.env.physics_engine
            else:
                raise AttributeError("Could not find '_physics_engine' or 'physics_engine' in UrdfEnv.")
            
            # Assign the found engine to the Lidar
            sim_instance.lidar._physics_engine = engine

            # We also need the robot OBJECT (not just ID) for the sense() method
            # UrdfEnv stores robots in a list
            sim_instance.robot_obj = sim_instance.env.robots[0]
            
            sim_instance.use_lidar = True
            print(f"🔧 Lidar activated: {nb_rays} beams, {ray_length} m range")
            
        except AttributeError as e:
            print(f"❌ Lidar Injection Failed: {e}")
            print(f"   Available attributes in env: {dir(sim_instance.env)}")
            sim_instance.use_lidar = False
            
    else:
        print("❌ Error: Environment not initialized. Cannot attach Lidar.")


# ============================================================
#                     MAP MAKER CLASS
# ============================================================

class MapMaker:
    """
    MapMaker — Add obstacles and static objects to your scene.
    """

    def __init__(self):
        self.obstacles = []

    def add_box(self, pos, size, yaw=0.0, rgba=(0.3, 0.3, 0.3, 1.0)):
        """
        Add a static box obstacle.
        pos : (x, y, z)
        size : (length, width, height)
        """
        lx, ly, lz = size
        x, y, z = pos

        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[lx/2, ly/2, lz/2]
        )
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[lx/2, ly/2, lz/2],
            rgbaColor=rgba
        )

        orn = p.getQuaternionFromEuler([0, 0, yaw])

        box_id = p.createMultiBody(
            baseMass=0,  # static (mass = 0)
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[x, y, z],
            baseOrientation=orn,
        )

        self.obstacles.append(box_id)
        return box_id


# ============================================================
#                         MAIN
# ============================================================

def main():
    def spawn_goal_marker(x, y, color=(1, 0, 0, 1)):
        goal_pos = [float(x), float(y), 0.05]
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.08, rgbaColor=color)
        body = p.createMultiBody(baseVisualShapeIndex=vis, basePosition=goal_pos)
        p.addUserDebugText("Goal", [x, y, 0.18], textColorRGB=[1, 1, 1], textSize=1.2)
        return body

    print("\n=== map_1.py — Static Robot + Table Scene ===")

    # -------------------------------------------------------
    # 1. Initialize Simulator
    # -------------------------------------------------------
    sim = AlbertTableImpedanceSim(render=True)
    
    # Build the PyBullet environment (creates sim.env)
    sim.create_environment()
    
    # Identify Robot ID
    sim.albert_id = sim.get_albert_body_id()
    
    # Load Table
    sim.load_table()
    
    # Set Arm to "Ready" pose
    sim.set_arm_initial_pose()
    
    print("Environment loaded.\nRobot + Table spawned.\n")

    # -------------------------------------------------------
    # 2. Add Map Elements
    # -------------------------------------------------------
    spawn_goal_marker(0.0, 5.0)

    map1 = MapMaker()

    # Wall 1
    map1.add_box(
        pos=(0.0, 3.0, 0.25),         
        size=(5.0, 0.25, 0.5),         
        yaw=np.deg2rad(0),           
        rgba=(0.5, 0.5, 0.5, 1.0)     
    )

    # Wall 2
    map1.add_box(
        pos=(0.0, -3.0, 0.25),          
        size=(5.0, 0.25, 0.5),         
        yaw=np.deg2rad(0),           
        rgba=(0.5, 0.5, 0.5, 1.0)     
    )
    
    # -------------------------------------------------------
    # 3. Activate Lidar
    # -------------------------------------------------------
    # Using 40 rays for visualization
    activate_lidar(sim, nb_rays=40, ray_length=6.0)

    print("Obstacles added. Scene ready. Press Ctrl+C to exit.\n")

    # -------------------------------------------------------
    # 4. Physics Loop
    # -------------------------------------------------------
    step_count = 0
    dt = sim.dt

    try:
        while True:
            # Important: Keep the arm stiff
            sim.enforce_rigid_arm()

            p.stepSimulation()

            # --- Lidar Observation ---
            if hasattr(sim, 'use_lidar') and sim.use_lidar:
                
                # The .sense() method requires the robot object, obstacles dict, goals dict, and time.
                # Since we are just testing the sensor, we pass empty dicts for obstacles/goals.
                lidar_data = sim.lidar.sense(
                    robot=sim.robot_obj,
                    obstacles={},
                    goals={},
                    t=step_count * dt
                )
                
                # Print summary every 100 steps
                if step_count % 100 == 0:
                    if lidar_data is not None and lidar_data.size > 0:
                         # Handle Raw Data (Distances)
                         print(f"Step {step_count} | Lidar: Min={np.min(lidar_data):.2f}m, Max={np.max(lidar_data):.2f}m")
                    else:
                        print(f"Step {step_count} | Lidar: No Data")

            time.sleep(dt)
            step_count += 1
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        if p.isConnected():
            p.disconnect()

if __name__ == "__main__":
    main()