#!/usr/bin/env python3
import time
import numpy as np
import rospy
import pybullet as p
import pybullet_data
from geometry_msgs.msg import WrenchStamped


# Toggle: True = dry-test sim only (no GUI), False = run full PyBullet GUI sim
DRY_TEST_SIM = False

# Latest haptic wrench (shared by ROS callback)
_latest_fx = 0.0
_latest_fy = 0.0
_latest_fz = 0.0

def wrench_cb(msg: WrenchStamped):
    global _latest_fx, _latest_fy, _latest_fz
    _latest_fx = msg.wrench.force.x
    _latest_fy = msg.wrench.force.y
    _latest_fz = msg.wrench.force.z


class GhostVisualizer:
    def __init__(self, z=0.15, radius=0.04):
        self.z = z
        self.radius = radius
        self.marker_id = None
        self.line_id = None

    def draw(self, puck_xy, ghost_xy):
        puck_3d = [puck_xy[0], puck_xy[1], self.z]
        ghost_3d = [ghost_xy[0], ghost_xy[1], self.z]
        if self.marker_id is None:
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=self.radius, rgbaColor=[0, 1, 0, 0.7])
            self.marker_id = p.createMultiBody(0, vis, basePosition=ghost_3d)
        else:
            p.resetBasePositionAndOrientation(self.marker_id, ghost_3d, [0, 0, 0, 1])
        if self.line_id is not None:
            p.removeUserDebugItem(self.line_id)
        self.line_id = p.addUserDebugLine(puck_3d, ghost_3d, [0, 1, 0], 2)


def main():
    rospy.init_node("puck_debug", anonymous=False)

    # Subscribe to the haptic force published by the C++ node
    rospy.Subscriber("/sigma7/wrench", WrenchStamped, wrench_cb, queue_size=1, tcp_nodelay=True)

    # Connect PyBullet
    if DRY_TEST_SIM:
        p.connect(p.DIRECT)
    else:
        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    p.setGravity(0, 0, -9.81)

    # Puck
    puck_id = p.createMultiBody(
        baseMass=10.0,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_CYLINDER, radius=0.08, height=0.04),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_CYLINDER, radius=0.08, length=0.04, rgbaColor=[0.2, 0.6, 1.0, 1.0]),
        basePosition=[0, 0, 0.04]
    )

    ghost_vis = GhostVisualizer()
    force_line_id = None

    # Simulation parameters
    sim_hz = 240.0
    dt = 1.0 / sim_hz

    # Clip for stability (match your earlier MAX_FORCE intent)
    MAX_FORCE = 40.0

    print("\n--- Puck simulation running; consuming /sigma7/wrench ---")


    step_count = 0
    try:
        while not rospy.is_shutdown():
            # Read latest force from callback
            Fx = -float(_latest_fx)  # apply opposite if you want puck pushed away
            Fy = -float(_latest_fy)

            # Clip
            mag = (Fx*Fx + Fy*Fy) ** 0.5
            if mag > MAX_FORCE and mag > 1e-12:
                s = MAX_FORCE / mag
                Fx *= s
                Fy *= s

            # Apply to puck
            puck_pos, _ = p.getBasePositionAndOrientation(puck_id)
            p.applyExternalForce(puck_id, -1, [Fx, Fy, 0.0], puck_pos, p.WORLD_FRAME)
            p.stepSimulation()

            # Visuals
            if not DRY_TEST_SIM:
                puck_xy = np.array(puck_pos[:2])
                ghost_xy = puck_xy  # if you have offset too, add it here
                ghost_vis.draw(puck_xy, ghost_xy)

                # Force line
                scale = 0.05
                if force_line_id is not None:
                    p.removeUserDebugItem(force_line_id)
                force_line_id = p.addUserDebugLine(
                    [puck_xy[0], puck_xy[1], 0.1],
                    [puck_xy[0] + Fx*scale, puck_xy[1] + Fy*scale, 0.1],
                    [1, 0, 0], 5
                )
            
            if step_count % 10 == 0: 
                print(f"Applied force [Fx, Fy]: {Fx, Fy} | magnitude: {mag}")
            time.sleep(dt)

    finally:
        p.disconnect()


if __name__ == "__main__":
    main()
