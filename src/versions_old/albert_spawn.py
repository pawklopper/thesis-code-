#!/usr/bin/env python3
import warnings
import gymnasium as gym
import numpy as np
import pybullet as p
import time

from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv


def run_albert(n_steps=1000, render=False, goal=True, obstacles=True):
    robots = [
        GenericDiffDriveRobot(
            urdf="albert.urdf",
            mode="vel",
            actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
            castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
            wheel_radius=0.08,
            wheel_distance=0.494,
            spawn_rotation=0,
            facing_direction='-y',
        ),
    ]

    env: UrdfEnv = UrdfEnv(dt=0.01, robots=robots, render=render)
    robot = robots[0]

    # --- Action: simple forward + turn ---
    action = np.zeros(env.n())
    action[0] = 0.2   # linear velocity [m/s]
    action[1] = 0.2   # yaw rate [rad/s]

    print("Number of actions:", len(action))
    print("Action array:", action)

    # --- Reset environment ---
    ob = env.reset(pos=np.array([0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0]))
    print(f"Initial observation : {ob}")

    # --- Setup goal ---
    history = []
    heading_errors = []
    goal = np.array([-2.0, 2.0])   # ✅ fixed test goal

    # --- Draw goal as red sphere ---
    goal_pos = [goal[0], goal[1], 0.05]
    goal_visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.07,
        rgbaColor=[1, 0, 0, 1],
    )
    goal_body = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=goal_visual,
        basePosition=goal_pos,
    )

    arrow_id = None

    for step in range(n_steps):
        ob, *_ = env.step(action)
        history.append(ob)

        base_state = ob["robot_0"]["joint_state"]
        base_pos = base_state["position"][:3]  # [x, y, yaw]
        x, y, yaw = base_pos

        # --- Compute goal direction + heading error ---
        dx = goal[0] - x
        dy = goal[1] - y

        # (1) World angle to goal (full 360° via atan2)
        angle_to_goal_world = np.arctan2(dy, dx)

        # (2) Translate goal angle into robot yaw-space (+π/2 because facing -y)
        angle_to_goal_adj = angle_to_goal_world + np.pi / 2.0

        # (3) Heading error (robot yaw vs goal)
        heading_error = angle_to_goal_adj - yaw
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        heading_errors.append(heading_error)

        # --- Compute forward direction (arrow endpoint) ---
        arrow_length = 0.5
        arrow_start = [x, y, 0.05]
        arrow_end = [
            x + arrow_length * np.cos(yaw),
            y + arrow_length * np.sin(yaw),
            0.05,
        ]

        # --- Draw / update forward direction arrow ---
        if arrow_id is not None:
            p.removeUserDebugItem(arrow_id)
        arrow_id = p.addUserDebugLine(
            lineFromXYZ=arrow_start,
            lineToXYZ=arrow_end,
            lineColorRGB=[0, 1, 0],
            lineWidth=2.0,
            lifeTime=0.01,
        )

        # --- Print debug info ---
        if step % 20 == 0:
            print(f"Step {step:04d}: x={x:+.3f}, y={y:+.3f}, yaw={yaw:+.3f}")
            print(f"  Goal = ({goal[0]:+.2f}, {goal[1]:+.2f})")
            print(f"  angle_to_goal_world = {angle_to_goal_world:+.3f}")
            print(f"  angle_to_goal_adj   = {angle_to_goal_adj:+.3f}")
            print(f"  heading_error       = {heading_error:+.3f}\n")

        if render:
            time.sleep(0.01)

    env.close()

    # --- Plot heading error after simulation ---
    import matplotlib.pyplot as plt
    plt.plot(heading_errors, label="Heading error [rad]")
    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    plt.xlabel("Step")
    plt.ylabel("Heading Error [rad]")
    plt.legend()
    plt.grid(True)
    plt.show()

    return history


# =====================================
# === MAIN: Run & Plot after finish ===
# =====================================
if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        history = run_albert(render=True)
