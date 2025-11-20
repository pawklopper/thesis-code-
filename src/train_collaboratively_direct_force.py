#!/usr/bin/env python3
"""
train_collaboratively.py  — FIXED VERSION (PATH A)
---------------------------------------------------
Continue training an already-trained SAC model
with real human PS5 joystick forces.

This version does NOT rebuild a new SAC object.
It loads the pretrained SAC and attaches the live env.

No networks are reinitialized.
Replay buffer is loaded automatically if stored.
Logging is preserved.
"""

import os
import time
import datetime
import numpy as np
import pybullet as p

from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure

from rl_env.albert_table_env import AlbertTableEnv
from controllers.ps5_human_control_direct_force import (
    init_ps5_controller,
    read_joystick_force,
    compute_world_force_from_table,
)


# ============================================================
# Helpers
# ============================================================

def draw_goal_marker(goal_xy, color=(1, 0, 0, 1)):
    goal_pos = [float(goal_xy[0]), float(goal_xy[1]), 0.05]
    visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.07, rgbaColor=color)
    mid = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual, basePosition=goal_pos)
    p.addUserDebugText("Goal", [goal_pos[0], goal_pos[1], 0.15], [1, 1, 1], textSize=1.2)
    return mid

def set_goal_color(goal_id, color=(0, 1, 0, 1)):  
    p.changeVisualShape(goal_id, -1, rgbaColor=color)

def sample_new_goal(rmin=1.0, rmax=3.0):
    theta = np.random.uniform(-np.pi, np.pi)
    r = np.random.uniform(rmin, rmax)
    return np.array([r * np.cos(theta), r * np.sin(theta)], dtype=np.float32)


# ============================================================
# MAIN
# ============================================================

def run_interactive_model(
    base_dir: str,
    load_run_subdir: str,
    model_name: str,
    single_goal_mode: bool,
    start_goal=(0.0, -2.0),
    max_runtime_steps=100_000,
):
    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------
    offline_dir = os.path.join("runs", "offline", base_dir, load_run_subdir)
    pretrained_path = os.path.join(offline_dir, f"{model_name}.zip")
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained model not found: {pretrained_path}")

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    online_root = os.path.join("runs", "online_human", base_dir, f"{ts}_{model_name}")
    tb_dir = os.path.join(online_root, "tb_logs")
    os.makedirs(tb_dir, exist_ok=True)

    print(f"[LOAD] {pretrained_path}")
    print(f"[SAVE ROOT] {online_root}")
    print(f"[TB] {tb_dir}")

    # --------------------------------------------------------
    # Load pretrained SAC (NO NEW MODEL CREATED)
    # --------------------------------------------------------
    model = SAC.load(pretrained_path)

    # --------------------------------------------------------
    # Attach new environment (THIS is the correct SB3 way)
    # --------------------------------------------------------
    env = AlbertTableEnv(render=True, goals=[start_goal])
    model.set_env(env)

    # --------------------------------------------------------
    # Logger setup (same as before)
    # --------------------------------------------------------
    logger = configure(tb_dir, ["stdout", "tensorboard"])
    model.set_logger(logger)

    # --------------------------------------------------------
    # PS5 controller
    # --------------------------------------------------------
    js = init_ps5_controller()

    # --------------------------------------------------------
    # Initial reset
    # --------------------------------------------------------
    obs, _ = env.reset(Fh_override=np.zeros(3))
    total_steps, steps_since_goal, goal_id = 0, 0, 1
    goal = np.array(start_goal)
    goal_marker_id = draw_goal_marker(goal)

    Fprev = np.zeros(3, dtype=np.float32)

    # Rolling windows for logging
    reward_window = []
    actor_loss_window, critic_loss_window = [], []
    ent_coef_window, ent_coef_loss_window = [], []
    learning_rate_window, n_updates_window = [], []

    # =======================================================
    # MAIN TRAINING LOOP
    # =======================================================
    try:
        while total_steps < max_runtime_steps:
            prev_obs = obs.copy()

            # --- Policy action ---
            action, _ = model.predict(obs, deterministic=True)

            # --- Human force via PS5 ---
            f_local = read_joystick_force(js, force_scale=40.0)
            f_world = compute_world_force_from_table(f_local, env.sim.table_id)

            Fh = 0.9 * Fprev + 0.1 * np.array([f_world[0], f_world[1], 0.0], dtype=np.float32)
            Fprev = Fh.copy()

            # --- Step environment ---
            obs, rew, terminated, truncated, info = env.step(action, Fh_override=Fh)
            done = terminated or truncated

            # Forces
            Fr = env.sim.last_F_xy if hasattr(env.sim, "last_F_xy") else np.zeros(2)
            Fh = env.sim.last_Fh_xy if hasattr(env.sim, "last_Fh_xy") else np.zeros(2)

            # Debug info
            if total_steps % 50 == 0:
                dist = info.get("dist_table_to_goal", float("nan"))
                print(
                    f"Step={total_steps:06d} | dist={dist:6.3f} | "
                    f"Fr={Fr} | Fh={Fh} "
                )

            # --- Add transition to replay buffer ---
            model.replay_buffer.add(
                prev_obs,
                obs,
                action,
                np.array([rew], dtype=np.float32),
                np.array([done], dtype=np.float32),
                [{}],
            )

            # --- Train when buffer warm ---
            if model.replay_buffer.size() > 1024:
                model.train(batch_size=64, gradient_steps=1)

                # Collect metrics
                for k, v in model.logger.name_to_value.items():
                    if k == "train/actor_loss": actor_loss_window.append(v)
                    elif k == "train/critic_loss": critic_loss_window.append(v)
                    elif k == "train/ent_coef": ent_coef_window.append(v)
                    elif k == "train/ent_coef_loss": ent_coef_loss_window.append(v)
                    elif k == "train/learning_rate": learning_rate_window.append(v)
                    elif k == "train/n_updates": n_updates_window.append(v)

            # --- Reward window ---
            reward_window.append(rew)

            # --- Logging every 100 steps ---
            if total_steps % 100 == 0 and reward_window:
                def mean_safe(x):
                    return float(np.mean(x)) if len(x) > 0 else 0.0

                model.logger.record("reward/avg_100", mean_safe(reward_window))
                model.logger.record("train/actor_loss", mean_safe(actor_loss_window))
                model.logger.record("train/critic_loss", mean_safe(critic_loss_window))
                model.logger.record("train/ent_coef", mean_safe(ent_coef_window))
                model.logger.record("train/ent_coef_loss", mean_safe(ent_coef_loss_window))
                model.logger.record("train/learning_rate", mean_safe(learning_rate_window))
                model.logger.record("train/n_updates", mean_safe(n_updates_window))
                model.logger.record("train/buffer_size", model.replay_buffer.size())

                model.logger.dump(step=total_steps)
                for of in model.logger.output_formats:
                    if hasattr(of, "writer") and of.writer:
                        of.writer.flush()

                reward_window.clear()
                actor_loss_window.clear()
                critic_loss_window.clear()
                ent_coef_window.clear()
                ent_coef_loss_window.clear()
                learning_rate_window.clear()
                n_updates_window.clear()

            total_steps += 1
            steps_since_goal += 1

            # --- Timeout ---
            if steps_since_goal >= 1000 and not terminated:
                print(f"Timeout: resetting SAME goal #{goal_id}")
                p.removeAllUserDebugItems()
                try:
                    p.removeBody(goal_marker_id)
                except:
                    pass
                obs, _ = env.reset(Fh_override=np.zeros(3))
                steps_since_goal = 0
                goal_marker_id = draw_goal_marker(goal)
                continue

            # --- Goal reached ---
            if terminated:
                print(f"Goal #{goal_id} reached!")
                set_goal_color(goal_marker_id, color=(0, 1, 0, 1))
                time.sleep(0.5)

                p.removeAllUserDebugItems()
                try:
                    p.removeBody(goal_marker_id)
                except:
                    pass

                if not single_goal_mode:
                    new_goal = sample_new_goal()
                    env.goals = [new_goal]
                    env.goal = new_goal
                    goal = new_goal
                    goal_id += 1
                obs, _ = env.reset(Fh_override=np.zeros(3))
                steps_since_goal = 0
                goal_marker_id = draw_goal_marker(goal)

            # --- ESC exit ---
            keys = p.getKeyboardEvents()
            if 27 in keys and keys[27] & p.KEY_WAS_TRIGGERED:
                print("ESC → exit")
                break

            # Toggle single-goal mode
            if ord('g') in keys and keys[ord('g')] & p.KEY_WAS_TRIGGERED:
                single_goal_mode = not single_goal_mode
                print(f"Toggled single_goal_mode → {single_goal_mode}")

            time.sleep(env.sim.dt)

    finally:
        env.close()
        final_zip = os.path.join(online_root, f"{model_name}_final.zip")
        model.save(final_zip)
        print(f"[FINAL SAVE] {final_zip}")
        print(f"[TB LOGDIR]  {tb_dir}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    base_dir = "runs_18_nov_test"
    load_run_subdir = "20251118-120841_18_nov_test"
    model_name = "18_nov_test"

    run_interactive_model(
        base_dir=base_dir,
        load_run_subdir=load_run_subdir,
        model_name=model_name,
        single_goal_mode=False,
        start_goal=(0.0, 2.0),
        max_runtime_steps=100_000,
    )
