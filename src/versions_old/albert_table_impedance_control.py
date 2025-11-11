#!/usr/bin/env python3
"""
Albert Manual Control and Cartesian Impedance Simulation

This script loads the Albert robot and a movable table in PyBullet,
sets up an impedance control loop between Albert’s end-effector and
the table handle, and runs a simple simulation.

Author: [Your Name]
"""

import os
import time
import warnings
import numpy as np
import pybullet as p

from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv


# ============================================================
# =======================  HELPERS  ===========================
# ============================================================

def get_movable_joint_state_arrays(body_id):
    """Return joint indices, positions (q), and velocities (dq) for non-fixed joints."""
    joint_indices, q_list, dq_list = [], [], []
    for ji in range(p.getNumJoints(body_id)):
        j_info = p.getJointInfo(body_id, ji)
        if j_info[2] != p.JOINT_FIXED:  # Non-fixed joints only
            joint_indices.append(ji)
            state = p.getJointState(body_id, ji)
            q_list.append(state[0])
            dq_list.append(state[1])
    return joint_indices, q_list, dq_list


def get_full_state_arrays(body_id):
    """Return full joint position and velocity arrays for a given body."""
    n = p.getNumJoints(body_id)
    q, dq = np.zeros(n), np.zeros(n)
    for ji in range(n):
        s = p.getJointState(body_id, ji)
        q[ji], dq[ji] = s[0], s[1]
    return q.tolist(), dq.tolist()


# ============================================================
# ==================  ENVIRONMENT SETUP  =====================
# ============================================================

def create_albert_environment(render=True):
    """Initialize PyBullet environment with Albert robot."""
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
    env.reset(pos=np.zeros(10))
    p.setGravity(0, 0, -9.81)
    print("✅ Albert environment created.")
    return env


def get_albert_body_id():
    """Find the Albert robot body ID in PyBullet."""
    for i in range(p.getNumBodies()):
        name = p.getBodyInfo(i)[1].decode("utf-8").lower()
        if "albert" in name:
            print(f"✅ Found Albert body ID: {i}")
            return i
    raise RuntimeError("❌ Could not find Albert in simulation!")


def print_joint_overview(body_id):
    """Print all joint information for a given body."""
    print("\n[Joints overview]")
    for j in range(p.getNumJoints(body_id)):
        info = p.getJointInfo(body_id, j)
        name = info[1].decode()
        j_type, lo, hi = info[2], info[8], info[9]
        print(f"{j:2d} | {name:28s} | type={j_type:1d} | limits=({lo:.2f},{hi:.2f})")


# ============================================================
# =====================  TABLE SETUP  ========================
# ============================================================

def load_table():
    """Load a dynamic table URDF and return its body ID and key link indices."""
    table_path = os.path.expanduser("~/catkin_ws/src/mobile_sim/assets/table/table.urdf")
    if not os.path.exists(table_path):
        raise FileNotFoundError(f"❌ Table URDF not found at {table_path}")

    table_distance = 1.15
    table_orn = p.getQuaternionFromEuler([0, 0, -np.pi / 2])

    table_id = p.loadURDF(
        table_path,
        basePosition=[0.0, -table_distance, 0.8],
        baseOrientation=table_orn,
        useFixedBase=False,
    )

    # Disable castor joint motors
    for j in range(p.getNumJoints(table_id)):
        name = p.getJointInfo(table_id, j)[1].decode()
        if ("caster" in name.lower()) or ("wheel" in name.lower()):
            print("wheel found")
            p.setJointMotorControl2(table_id, j, controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=0, force=0)
    
    # Get mass info of the table
    total_mass = 0
    for link_idx in range(-1, p.getNumJoints(table_id)):
        mass = p.getDynamicsInfo(table_id, link_idx)[0]
        total_mass += mass
        print(f"Total table mass = {total_mass:.2f} kg")




    # Let it settle
    for _ in range(100):
        p.stepSimulation()
        time.sleep(0.01)

    # Find red sphere (table handle)
    goal_link_idx = None
    for j in range(p.getNumJoints(table_id)):
        link_name = p.getJointInfo(table_id, j)[12].decode("utf-8")
        if "table_robot_end" in link_name.lower():
            goal_link_idx = j
            break

    if goal_link_idx is None:
        raise RuntimeError("❌ Could not find table_robot_end link")

    goal_world = np.array(p.getLinkState(table_id, goal_link_idx)[0])
    print(f"🎯 Red sphere (after settling): {np.round(goal_world, 3)}")

    return table_id, goal_link_idx, goal_world


# ============================================================
# ==================  ARM INITIALIZATION  ====================
# ============================================================

def set_arm_initial_pose(albert_id, arm_joint_indices):
    """Set Albert’s arm to a starting pose."""
    q_current = np.array([p.getJointState(albert_id, j)[0] for j in arm_joint_indices])
    offset_deg = [0, 90, 0, 25, 0, 40, 38]
    q_target = q_current + np.deg2rad(offset_deg)

    for j, q in zip(arm_joint_indices, q_target):
        p.resetJointState(albert_id, j, q)

    print("✅ Applied arm offset (degrees):", offset_deg)
    return q_target


def disable_arm_motors(albert_id, arm_joint_indices):
    """Disable default motors to enable torque control."""
    for j in arm_joint_indices:
        p.setJointMotorControl2(albert_id, j,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=0.0,
                                force=0.0)


# ============================================================
# ==================  IMPEDANCE CONTROLLER  ==================
# ============================================================

# def impedance_step(albert_id, table_id, ee_idx, goal_link_idx,
#                    arm_joint_indices, Kp, Dp, F_max, tau_max):
#     """Perform one impedance control step between EE and table handle."""
#     # EE state
#     ee_state = p.getLinkState(albert_id, ee_idx, computeLinkVelocity=1)
#     ee_pos, ee_lin_vel = np.array(ee_state[0]), np.array(ee_state[6])

#     # Handle state
#     handle_state = p.getLinkState(table_id, goal_link_idx, computeLinkVelocity=1)
#     handle_pos, handle_lin_vel = np.array(handle_state[0]), np.array(handle_state[6])

#     # Impedance law
#     dx, dv = handle_pos - ee_pos, handle_lin_vel - ee_lin_vel
#     F = -(Kp @ dx + Dp @ dv)
#     F = np.clip(F, -F_max, F_max)

#     #F = np.array([0, 0, 0])

#     # Apply external force on table
#     p.applyExternalForce(table_id, goal_link_idx,
#                          forceObj=F.tolist(),
#                          posObj=handle_pos.tolist(),
#                          flags=p.WORLD_FRAME)

#     # Compute torques τ = Jᵀ(-F)
#     joint_indices, q_list, dq_list = get_movable_joint_state_arrays(albert_id)
#     J_lin, _ = p.calculateJacobian(albert_id, ee_idx, [0, 0, 0],
#                                    q_list, dq_list, [0.0]*len(q_list))
#     J_arm = np.array(J_lin)[:, arm_joint_indices]
#     tau = np.clip(J_arm.T @ (-F), -tau_max, tau_max)

#     # Apply torques
#     for j, t in zip(arm_joint_indices, tau):
#         p.setJointMotorControl2(albert_id, j, controlMode=p.TORQUE_CONTROL, force=float(t))

#     return F, dx

def impedance_step(albert_id, table_id, ee_idx, goal_link_idx,
                   arm_joint_indices, Kp, Dp, F_max, tau_max):
    """Perform one impedance control step between EE and table handle."""

    # EE state
    ee_state = p.getLinkState(albert_id, ee_idx, computeLinkVelocity=1)
    ee_pos, ee_lin_vel = np.array(ee_state[0]), np.array(ee_state[6])
    ee_quat = ee_state[1]

    # Handle state
    handle_state = p.getLinkState(table_id, goal_link_idx, computeLinkVelocity=1)
    handle_pos, handle_lin_vel = np.array(handle_state[0]), np.array(handle_state[6])
    handle_quat = handle_state[1]

    # ==================================================================
    # ✅ Cartesian impedance in position (current behavior)
    # ==================================================================
    dx = handle_pos - ee_pos
    dv = handle_lin_vel - ee_lin_vel
    F = -(Kp @ dx + Dp @ dv)
    F = np.clip(F, -F_max, F_max)

    # Apply force on the table handle
    p.applyExternalForce(table_id, goal_link_idx,
                         forceObj=F.tolist(),
                         posObj=handle_pos.tolist(),
                         flags=p.WORLD_FRAME)

    # ==================================================================
    # ✅ NEW: Rotational impedance around vertical axis (fix push behavior)
    # ==================================================================
    # Extract yaw angles from orientations
    h_yaw = p.getEulerFromQuaternion(handle_quat)[2]
    ee_yaw = p.getEulerFromQuaternion(ee_quat)[2]

    # Wrap yaw error into [-pi, pi]
    yaw_error = ((h_yaw - ee_yaw + np.pi) % (2*np.pi)) - np.pi

    # Angular velocities (only z-axis needed)
    h_ang_vel = p.getLinkState(table_id, goal_link_idx, computeLinkVelocity=1)[7][2]
    ee_ang_vel = p.getLinkState(albert_id, ee_idx, computeLinkVelocity=1)[7][2]
    yaw_vel_error = h_ang_vel - ee_ang_vel

    # Rotational impedance gains
    K_yaw = 80.0    # rotational stiffness
    D_yaw = 4.0     # rotational damping

    # Torque around z-axis
    tau_z = -K_yaw * yaw_error - D_yaw * yaw_vel_error

    # Apply torque to the table handle
    p.applyExternalTorque(table_id, goal_link_idx,
                          [0.0, 0.0, tau_z],
                          flags=p.WORLD_FRAME)

    # ==================================================================
    # ✅ Apply opposite torques on robot arm (Jᵀ * wrench)
    # ==================================================================
    joint_indices, q_list, dq_list = get_movable_joint_state_arrays(albert_id)

    J_lin, J_ang = p.calculateJacobian(albert_id, ee_idx,
                                       [0, 0, 0],
                                       q_list, dq_list,
                                       [0.0] * len(q_list))

    J_arm_lin = np.array(J_lin)[:, arm_joint_indices]
    J_arm_ang = np.array(J_ang)[:, arm_joint_indices]

    # Force-based torques
    tau_lin = J_arm_lin.T @ (-F)

    # Rotation-based torques
    tau_rot = J_arm_ang.T @ np.array([0, 0, -tau_z])

    # Combine and clamp torques
    tau_total = np.clip(tau_lin + tau_rot, -tau_max, tau_max)

    # Apply torque to arm joints
    for j, t in zip(arm_joint_indices, tau_total):
        p.setJointMotorControl2(albert_id, j,
                                controlMode=p.TORQUE_CONTROL,
                                force=float(t))

    return F, dx





def check_table_yaw_alignment(table_id):
    """Check table yaw alignment (raw PyBullet yaw) against world axes."""
    pos, orn = p.getBasePositionAndOrientation(table_id)
    xy = np.array(pos[:2])
    yaw_raw = p.getEulerFromQuaternion(orn)[2]

    print("\n🧭 Table orientation check (RAW yaw only)")
    print(f"Table CoM position: {xy.round(3)}")
    print(f"Raw yaw (PyBullet world frame, 0° = +X) = {np.degrees(yaw_raw):7.2f}°\n")

    # Four reference points in world coordinates
    refs = {
        "+Y axis": np.array([0.0,  2.0]),
        "+X axis": np.array([2.0,  0.0]),
        "-X axis": np.array([-2.0, 0.0]),
        "-Y axis": np.array([0.0, -2.0])
    }

    print(f"{'Axis':8s} | {'Point (x,y)':>14s} | {'World θ [deg]':>12s} | {'Yaw_raw [deg]':>12s} | {'Δ [deg]':>8s}")
    print("-"*65)

    for label, pt in refs.items():
        vec = pt - xy
        world_angle = np.arctan2(vec[1], vec[0])
        diff = ((yaw_raw - world_angle + np.pi) % (2*np.pi)) - np.pi
        print(f"{label:8s} | {np.round(pt,3)} | {np.degrees(world_angle):12.2f} | "
              f"{np.degrees(yaw_raw):12.2f} | {np.degrees(diff):8.2f}")




# ============================================================
# ======================  MAIN LOOP  =========================
# ============================================================

def run_albert_manual(render=True):
    """Run the full Albert manual impedance simulation."""
    env = create_albert_environment(render)
    albert_id = get_albert_body_id()
    print_joint_overview(albert_id)
    

    arm_joint_indices = [7, 8, 9, 10, 11, 12, 13]
    ee_idx = 16

    table_id, goal_link_idx, _ = load_table()

    check_table_yaw_alignment(table_id)

    set_arm_initial_pose(albert_id, arm_joint_indices)
    disable_arm_motors(albert_id, arm_joint_indices)

    # Controller parameters
    Kp = np.diag([300.0, 300.0, 0.0])
    Dp = np.diag([60.0, 60.0, 0.0])
    # Kp = np.diag([1000.0, 1000.0, 0.0])
    # Dp = np.diag([0.0, 0.0, 0.0])



    F_max = np.array([150.0, 150.0, 150.0])
    tau_max = 25.0

    # Base action
    action = np.zeros(env.n())
    action[0], action[1] = 0.3, 0.1
    print("\n💡 Running Cartesian impedance simulation. Ctrl+C to exit.")
    dt, step = 0.01, 0

    while True:
        F, dx = impedance_step(albert_id, table_id, ee_idx, goal_link_idx,
                               arm_joint_indices, Kp, Dp, F_max, tau_max)
        env.step(action)
        if step % 50 == 0:
            print(f"[{step:05d}] Δx={dx.round(3)}, F={F.round(2)}")
        step += 1
        time.sleep(dt)


# ============================================================
# ====================  MAIN EXECUTION  =======================
# ============================================================

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        run_albert_manual(render=True)
