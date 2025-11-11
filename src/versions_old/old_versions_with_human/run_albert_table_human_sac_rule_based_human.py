#!/usr/bin/env python3
"""
run_albert_table_human_sac.py
-----------------------------------------
Run a trained SAC model on the Albert + Table impedance simulation
with the simulated human partner.

✨ Features:
 - Albert drives the table handle toward a goal.
 - Simulated human exerts impedance-based cooperative forces.
 - Prints robot/human force distribution and gating state.
 - Automatic goal reset upon success.
"""

import os
import time
import numpy as np
import pybullet as p
from stable_baselines3 import SAC
import matplotlib.pyplot as plt

# ✅ Import the updated environment with the simulated human
from train_albert_table_human_sac_rule_based_human import AlbertTableEnv


def plot_human_reference_trajectory(human):
    handle = np.array(human.traj_handle)
    ref = np.array(human.traj_ref)

    plt.figure(figsize=(6,6))
    plt.plot(handle[:,0], handle[:,1], 'b-', label='Human Handle')
    plt.plot(ref[:,0], ref[:,1], 'r--', label='Reference Point')
    plt.scatter(handle[0,0], handle[0,1], c='blue', marker='o', label='Start')
    plt.scatter(ref[-1,0], ref[-1,1], c='red', marker='x', label='End')
    plt.axis('equal')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Human Handle vs. Reference Trajectories')
    plt.legend()
    plt.grid(True)
    plt.show()

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
    Runs a trained SAC policy on the Albert + Table impedance simulation
    (with the simulated human model).
    """
    print("\n🚀 Running SAC model on Albert + Table impedance setup (with human sim)...")
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

            # Step environment forward (includes simulated human forces)
            obs, reward, terminated, truncated, info = env.step(action)
            total_steps += 1

            # Retrieve latest robot and human forces from simulation
            Fr = env.sim.last_F_xy if hasattr(env.sim, "last_F_xy") else np.zeros(2)
            Fh = env.sim.last_Fh_xy if hasattr(env.sim, "last_Fh_xy") else np.zeros(2)
            gate = env.sim.gate if hasattr(env.sim, "gate") else 0.0
            human_action = env.sim.human_action if hasattr(env.sim, "human_action") else "no action found"

            # --- Debug print every 50 steps ---
            if total_steps % 50 == 0:
                dist = info.get("dist_table_to_goal", float("nan"))
                Fr_mag = np.linalg.norm(Fr)
                Fh_mag = np.linalg.norm(Fh)
                Ft_mag = np.linalg.norm(Fr + Fh)
                print(
                    f"Step={total_steps:06d} | dist={dist:6.3f} | "
                    f"Fr=[{Fr[0]:7.2f}, {Fr[1]:7.2f}] | |Fr|={Fr_mag:7.2f} | "
                    f"Fh=[{Fh[0]:7.2f}, {Fh[1]:7.2f}] | |Fh|={Fh_mag:7.2f} | "
                    f"Σ|F|={Ft_mag:7.2f} | human_action= {human_action}"
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

                #plot_human_reference_trajectory(env.sim.human)

            # --- Timeout: print a message but keep going ---
            elif truncated and total_steps % 500 == 0:
                print(f"⚠️ Timeout after {env.max_steps} steps — continuing without reset")

            # Slow down simulation when rendering is on
            if env.render_mode:
                time.sleep(env.sim.dt)

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")

    finally:
        env.close()
        print(f"✅ Run finished after {total_steps} total steps.")


# ============================================================
# ==================== MAIN EXECUTION ========================
# ============================================================

if __name__ == "__main__":

    # ======================================================
    # == Run with rule-based human, trained on all point ===
    # ======================================================
    
    # base_log_dir = "runs_5_nov_test"
    # model_name = "5_nov_test"
    # run_subdir = "20251105-193933_5_nov_test"
    # start_goal = (-3.0, 4.0)


    base_log_dir = "runs_5_nov_test"
    model_name = "5_nov_test"
    run_subdir = "20251105-193933_5_nov_test"
    start_goal = (-2.0, 2.0)


    print(f"▶️ Running trained SAC policy from: {run_subdir}")
    print(f"🎯 Start goal: {start_goal}")

    run_trained_model_conditional_reset(
        base_log_dir=base_log_dir,
        model_name=model_name,
        run_subdir=run_subdir,
        start_goal=start_goal,
        max_runtime_steps=100_000,
    )
