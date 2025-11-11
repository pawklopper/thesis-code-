#!/usr/bin/env python3
"""
run_albert_table_sac_conditional_reset.py
-----------------------------------------
Run a trained SAC model on the Albert + Table impedance simulation.

✨ Features:
 - Albert drives the table handle toward a goal.
 - When the table reaches the goal → environment resets and a new random goal is set.
 - When time limit is reached → continues simulation (no reset).
 - Includes PyBullet goal visualization markers.

✅ Compatible with:
    train_albert_table_sac_debug.py (14D observation environment)
"""

import os
import time
import numpy as np
import pybullet as p
from stable_baselines3 import SAC

# ✅ Import the correct updated environment
from mobile_sim.src.train_albert_table_sac import AlbertTableEnv


# ============================================================
# =============== Continuous Evaluation Loop =================
# ============================================================

def run_trained_model_conditional_reset(
    base_log_dir: str,
    model_name: str,
    run_subdir: str,
    start_goal=(0.0, -2.0),
    max_runtime_steps: int = 100_000,
):
    """
    Runs a trained SAC policy on the Albert + Table impedance simulation.

    Automatically resets when the table reaches a goal, samples a new random goal,
    and continues simulation otherwise.

    Args:
        base_log_dir (str): Parent folder of training runs (e.g. "runs_albert_table_impedance")
        model_name (str): Model name (without .zip)
        run_subdir (str): Timestamped training run subfolder
        start_goal (tuple): Initial goal coordinates
        max_runtime_steps (int): Max total runtime before exit
    """

    print("\n🚀 Running SAC model on Albert + Table impedance setup...")
    run_dir = os.path.join(base_log_dir, run_subdir)
    model_path = os.path.join(run_dir, f"{model_name}.zip")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found at: {model_path}")

    # === Load environment and trained model ===
    env = AlbertTableEnv(render=True, goals=[start_goal])
    model = SAC.load(model_path, env=env)
    print(f"✅ Loaded model from: {model_path}")

    # --------------------------------------------------------
    # --- Visualization helpers ------------------------------
    # --------------------------------------------------------

    def draw_goal_marker(goal_xy, color=(1, 0, 0, 1), label="Goal"):
        """Draws a sphere and label for the current goal position in PyBullet."""
        goal_pos = [float(goal_xy[0]), float(goal_xy[1]), 0.05]
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.07,
            rgbaColor=color,
        )
        marker_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
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
        """Samples a random new goal within a ring around the origin."""
        theta = np.random.uniform(-np.pi, np.pi)
        r = np.random.uniform(radius_min, radius_max)
        goal_x = r * np.cos(theta)
        goal_y = r * np.sin(theta)
        return np.array([goal_x, goal_y], dtype=np.float32)

    # --------------------------------------------------------
    # --- Main runtime loop ---------------------------------
    # --------------------------------------------------------

    obs, _ = env.reset()
    total_steps = 0
    goal = np.array(start_goal)
    goal_marker = draw_goal_marker(goal, color=(1, 0, 0, 1))
    goal_id = 1

    try:
        while total_steps < max_runtime_steps:
            # Predict next action using trained SAC policy
            action, _ = model.predict(obs, deterministic=True)

            # Step environment forward
            obs, reward, terminated, truncated, info = env.step(action)
            total_steps += 1

            # --- Debug print every 200 steps ---
            if total_steps % 100 == 0:
                dist = info.get("dist_table_to_goal", float("nan"))
                print(
                    f"Step={total_steps:06d} | dist_to_goal={dist:.3f} | goal=({goal[0]:.2f}, {goal[1]:.2f} | reward=({reward}))"
                )

            # --- Goal reached: reset environment & sample new goal ---
            if terminated:
                print(f"\n🎯 Goal #{goal_id} reached at step {total_steps}!")
                p.changeVisualShape(goal_marker, -1, rgbaColor=[0, 1, 0, 1])  # turn marker green

                # Sample and set a new random goal
                new_goal = sample_new_goal()
                env.goals = [np.array(new_goal, dtype=np.float32)]
                env.goal = env.goals[0]

                # Reset environment (recreates Albert + Table)
                obs, _ = env.reset()

                # Draw new goal marker
                goal = env.goal
                goal_id += 1
                print(f"🧭 New goal #{goal_id}: {tuple(goal)}")
                goal_marker = draw_goal_marker(goal, color=(1, 0, 0, 1))

            # --- Timeout: print a message but keep going ---
            elif truncated and total_steps % 500 == 0:
                print(f"⚠️ Timeout after {env.max_steps} steps — continuing without reset")

            # Slow down simulation when rendering is on
            if env.render_mode:
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")

    finally:
        env.close()
        print(f"✅ Run finished after {total_steps} total steps.")


# ============================================================
# ==================== MAIN EXECUTION ========================
# ============================================================

if __name__ == "__main__":
    # this model works well, can drive anywhere while connected for the table, does only account for itself and not table objective yet
    # base_log_dir = "runs_albert_table_robot_only"
    # model_name = "sac_albert_table_robot_only"
    # run_subdir = "20251021-180801_sac_albert_table_robot_only"

    # ⚙️ Adjust these to match your most recent trained model
    base_log_dir = "runs_albert_table_yawalign"
    model_name = "sac_albert_table_yawalign"
    run_subdir = "20251024-150607_sac_albert_table_yawalign"  # Example run folder
    print(f"running: {run_subdir}")
    start_goal = (2.0, -2.0)
    print("start_goal", start_goal)

    run_trained_model_conditional_reset(
        base_log_dir=base_log_dir,
        model_name=model_name,
        run_subdir=run_subdir,
        start_goal=start_goal,
        max_runtime_steps=100_000,
    )
