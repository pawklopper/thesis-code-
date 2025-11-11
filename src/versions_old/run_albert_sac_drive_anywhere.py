#!/usr/bin/env python3
"""
run_albert_sac_conditional_reset.py
-----------------------------------
Run a trained SAC model continuously:
 - The robot drives toward a goal.
 - If it reaches the goal → environment resets, new random goal is set.
 - If it times out / fails → it keeps running without resetting.
"""

import os
import time
import numpy as np
import pybullet as p
from stable_baselines3 import SAC
from train_albert_sac_drive_anywhere import SimpleAlbertEnv


def run_trained_model_conditional_reset(
    base_log_dir: str,
    model_name: str,
    run_subdir: str,
    start_goal=(1.5, -2.0),
    max_runtime_steps: int = 50_000,
):
    """
    Args:
        base_log_dir (str): Directory containing training runs.
        model_name (str): Model file name (without .zip).
        run_subdir (str): Timestamped subfolder for this model.
        start_goal (tuple): Starting goal coordinates.
        max_runtime_steps (int): Safety limit for total simulation steps.
    """
    print("\n🚀 Running SAC model in conditional continuous mode...")
    run_dir = os.path.join(base_log_dir, run_subdir)
    model_path = os.path.join(run_dir, f"{model_name}.zip")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found: {model_path}")

    # --- Load environment and model ---
    # ✅ Updated: SimpleAlbertEnv now expects 'goals' instead of 'goal'
    env = SimpleAlbertEnv(render=True, goals=[start_goal])
    model = SAC.load(model_path, env=env)
    print(f"✅ Loaded model from: {model_path}")

    # --- Helper: draw and update visual markers ---
    def draw_goal_marker(goal_xy, color=(1, 0, 0, 1), label="Goal"):
        goal_pos = [float(goal_xy[0]), float(goal_xy[1]), 0.05]
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.07,
            rgbaColor=color,
        )
        marker_id = p.createMultiBody(
            baseMass=0,  # ✅ No mass = static body
            baseCollisionShapeIndex=-1,  # ✅ No collision — makes it "ghost"
            baseVisualShapeIndex=visual_shape,
            basePosition=goal_pos,
        )

        p.addUserDebugText(
            text=label,
            textPosition=[goal_pos[0], goal_pos[1], 0.15],
            textColorRGB=[1, 1, 1],
            textSize=1.2,
        )
        return marker_id

    def sample_new_goal(radius_min=1.0, radius_max=3.0):
        """Sample a new random goal within a ring around the origin."""
        theta = np.random.uniform(-np.pi, np.pi)
        r = np.random.uniform(radius_min, radius_max)
        goal_x = r * np.cos(theta)
        goal_y = r * np.sin(theta)
        return np.array([goal_x, goal_y], dtype=np.float32)

    # --- Initialize ---
    obs, _ = env.reset()
    total_steps = 0
    goal = np.array(start_goal)
    goal_marker = draw_goal_marker(goal, color=(1, 0, 0, 1))
    goal_id = 1

    try:
        while total_steps < max_runtime_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_steps += 1

            if total_steps % 200 == 0:
                print(
                    f"Step={total_steps:06d} | dist={info['dist_to_goal']:.3f} | goal=({goal[0]:.2f}, {goal[1]:.2f})"
                )

            # ✅ If goal reached: reset + new goal
            if terminated:
                print(f"\n🎯 Goal #{goal_id} reached at step {total_steps}!")
                p.changeVisualShape(goal_marker, -1, rgbaColor=[0, 1, 0, 1])

                # --- Sample and assign a new goal (ensure reset uses it) ---
                new_goal = sample_new_goal()
                env.goals = [np.array(new_goal, dtype=np.float32)]  # <— key line
                env.goal = env.goals[0]                              # (explicit)

                # --- Reset environment (now starts with new_goal) ---
                obs, _ = env.reset()

                # --- Update visualization and tracking ---
                goal = env.goal
                goal_id += 1
                print(f"🧭 New goal #{goal_id}: {tuple(goal)}")
                goal_marker = draw_goal_marker(goal, color=(1, 0, 0, 1))

            # ❌ If truncated (timeout): DO NOT RESET
            elif (truncated) and (total_steps % 500 == 0):
                print(f"⚠️ Timeout after {env.max_steps} steps — continuing without reset")

            if env.render_mode:
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
    finally:
        env.close()
        print(f"✅ Run finished after {total_steps} total steps.")


# ============================================================
# =============== MAIN CONFIGURATION =========================
# ============================================================

if __name__ == "__main__":
    base_log_dir = "basic_runs_albert"
    model_name = "sac_albert_random_goals_manual_test"
    run_subdir = "20251010-151259_sac_albert_random_goals_manual_test"
    start_goal = (-4.0, 5.0)
    

    # this model can reach any goal in the plane, no obstacle yet
    base_log_dir = "basic_runs_albert"
    model_name = "sac_albert_random_goals_manual_test"
    run_subdir = "20251010-151259_sac_albert_random_goals_manual_test"
  
    run_trained_model_conditional_reset(
        base_log_dir=base_log_dir,
        model_name=model_name,
        run_subdir=run_subdir,
        start_goal=start_goal,
        max_runtime_steps=100_000,  # safety cap
    )
