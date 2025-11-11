#!/usr/bin/env python3
import pybullet as p
import numpy as np
import time
import os
import warnings

from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv


# ============================================================
# ===============  MANUAL JOINT CONTROL BOARD  ===============
# ============================================================

def run_albert_manual(render=True):
    """Manual joint viewer and simple control with table and EE-goal distance."""

    # ---------------- Environment & Robot ----------------
    robot = GenericDiffDriveRobot(
        urdf="albert.urdf",
        mode="vel",
        actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
        castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
        wheel_radius=0.08,
        wheel_distance=0.494,
        spawn_rotation=0,
        facing_direction='-y',
    )

    env = UrdfEnv(dt=0.01, robots=[robot], render=render)
    p.setGravity(0, 0, -9.81)
    print("✅ Albert environment created.")
    env.reset(pos=np.zeros(10))

    # ---------------- Locate Albert ----------------
    albert_id = next(
        (i for i in range(p.getNumBodies())
         if "albert" in p.getBodyInfo(i)[1].decode("utf-8").lower()), None
    )
    if albert_id is None:
        raise RuntimeError("❌ Could not find Albert in simulation!")
    print(f"✅ Found Albert body ID: {albert_id}")

    # ---------------- Joint overview ----------------
    num_joints = p.getNumJoints(albert_id)
    print("\n[Joints overview]")
    for j in range(num_joints):
        j_info = p.getJointInfo(albert_id, j)
        j_name = j_info[1].decode()
        j_type = j_info[2]
        lo, hi = j_info[8], j_info[9]
        print(f"{j:2d} | {j_name:28s} | type={j_type:1d} | limits=({lo:.2f},{hi:.2f})")

    # Arm joint indices (from earlier mapping)
    arm_joint_indices = [7, 8, 9, 10, 11, 12, 13]
    ee_idx = 16  # fingertip

    # ---------------- Spawn Table ----------------
    table_path = os.path.expanduser("~/catkin_ws/src/mobile_sim/assets/table/table.urdf")
    if not os.path.exists(table_path):
        raise FileNotFoundError(f"❌ Table URDF not found at {table_path}")

    table_distance = 1.15  # meters forward (Albert faces -Y)
    table_height = 0.8
    table_pos = [0.0, -table_distance, table_height]
    table_orn = p.getQuaternionFromEuler([0, 0, -np.pi / 2])

    # Load the table as a dynamic object
    table_id = p.loadURDF(
        table_path,
        basePosition=[0.0, -table_distance, 0.8],  # initial guess (can be rough)
        baseOrientation=p.getQuaternionFromEuler([0, 0, -np.pi / 2]),
        useFixedBase=False,
    )

    # Let it fall and settle for a short while
    for _ in range(100):
        p.stepSimulation()
        time.sleep(0.01)

    # After table_id = p.loadURDF(...)
    num_joints = p.getNumJoints(table_id)
    for j in range(num_joints):
        j_name = p.getJointInfo(table_id, j)[1].decode()
        if "wheel" in j_name or "rotacastor" in j_name:
            # Apply a constant velocity motor so the wheel rotates visually
            p.setJointMotorControl2(
                bodyUniqueId=table_id,
                jointIndex=j,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=0.0,   # rad/s
                force=0              # max torque
            )


    # Now find the link index of the red sphere
    goal_link_idx = None
    for j in range(p.getNumJoints(table_id)):
        link_name = p.getJointInfo(table_id, j)[12].decode("utf-8")
        if "table_robot_end" in link_name.lower():
            goal_link_idx = j
            break

    if goal_link_idx is None:
        raise RuntimeError("❌ Could not find table_robot_end link")

    # ✅ Get its *final* settled position
    goal_world = np.array(p.getLinkState(table_id, goal_link_idx)[0])
    print(f"🎯 Red sphere (after settling): {np.round(goal_world, 3)}")

    # ---------------- Read current joint angles ----------------
    q_current = np.array([p.getJointState(albert_id, j)[0] for j in arm_joint_indices])
    print("\nCurrent joint angles (rad):", np.round(q_current, 3))
    print("Current joint angles (deg):", np.round(np.rad2deg(q_current), 2))

    # ---------------- Apply small offset to each joint ----------------
    offset_deg = [0, 0, 0, 0, 0, 0, 0]  # adjust degrees for each joint

    # height control
    offset_deg[1] += 90
    offset_deg[3] += 25 
    offset_deg[5] += 40 

    # hand wrist
    offset_deg[6] += 38 

    q_target = q_current + np.deg2rad(offset_deg)


    for j, q in zip(arm_joint_indices, q_target):
        p.resetJointState(albert_id, j, q)
    print("✅ Applied offset (degrees):", offset_deg)


    # ---------------- Create a fixed joint (constraint) between EE and table handle ----------------
    ee_state = p.getLinkState(albert_id, ee_idx)
    ee_pos, ee_orn = ee_state[0], ee_state[1]

    goal_state = p.getLinkState(table_id, goal_link_idx)
    goal_pos, goal_orn = goal_state[0], goal_state[1]

    # Compute relative transform from EE to handle
    parentFramePos, parentFrameOrn = p.invertTransform(ee_pos, ee_orn)
    rel_pos, rel_orn = p.multiplyTransforms(parentFramePos, parentFrameOrn, goal_pos, goal_orn)

    constraint_id = p.createConstraint(
        parentBodyUniqueId=albert_id,
        parentLinkIndex=ee_idx,
        childBodyUniqueId=table_id,
        childLinkIndex=goal_link_idx,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=rel_pos,
        parentFrameOrientation=rel_orn,
        childFramePosition=[0, 0, 0],
        childFrameOrientation=[0, 0, 0, 1],
    )

    print(f"🔗 Fixed joint created between EE(link {ee_idx}) and table handle(link {goal_link_idx}), constraint ID={constraint_id}")


    # # --- Record fixed arm joint angles ---
    # q_fixed = np.array([p.getJointState(albert_id, j)[0] for j in arm_joint_indices])
    # print("⏸️  Freezing arm at:", np.round(np.rad2deg(q_fixed), 1), "deg")

    # # Disable dynamics on those arm links (optional but helps)
    # for j in arm_joint_indices:
    #     p.changeDynamics(albert_id, j, linearDamping=1.0, angularDamping=1.0)

    # ---------------- Simple simulation loop ----------------
    # define action
    action_dim = env.n()
    action = np.zeros(action_dim)
    action[0] = -0.2 # base velocity
    action[1] = 0.2 # base yaw
    print("\n💡 Running simple simulation. Ctrl+C to exit.")

    # --- Capture initial joint angles once ---
    q_init = np.array([p.getJointState(albert_id, j)[0] for j in arm_joint_indices])
    print("\n📏 Initial joint angles:")
    for j, q in zip(arm_joint_indices, q_init):
        print(f"  joint {j:2d}: {q:+.3f} rad  ({np.degrees(q):+.1f}°)")



    step = 0
    while True:
        p.stepSimulation()
        env.step(action)

        # Print joint angles every 50 simulation steps
        # if step % 50 == 0:
        #     q_now = np.array([p.getJointState(albert_id, j)[0] for j in arm_joint_indices])
        #     print(f"\nStep {step:05d} | Joint angles:")
        #     for j, q in zip(arm_joint_indices, q_now):
        #         print(f"  joint {j:2d}: {q:+.3f} rad  ({np.degrees(q):+.1f}°)")

        step += 1
        time.sleep(0.02)


# ============================================================
# ====================  MAIN EXECUTION  =======================
# ============================================================

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        run_albert_manual(render=True)
