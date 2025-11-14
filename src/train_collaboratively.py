#!/usr/bin/env python3
"""
train_collaboratively.py
-----------------------------------------
Run a trained SAC model WITH CONTINUING TRAINING (online),
human PS5 joystick force input, and automatic reset if the robot
fails to reach the goal within 1000 steps.
"""

import os
import time
import datetime
import numpy as np
import pybullet as p

from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure

from rl_env.albert_table_env import AlbertTableEnv
from controllers.ps5_human_control import (
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

def set_goal_color(goal_id, color=(0, 1, 0, 1)):  # default = green
    """Update the goal marker color in PyBullet."""
    p.changeVisualShape(goal_id, -1, rgbaColor=color)



def sample_new_goal(rmin=1.0, rmax=3.0):
    theta = np.random.uniform(-np.pi, np.pi)
    r = np.random.uniform(rmin, rmax)
    return np.array([r * np.cos(theta), r * np.sin(theta)], dtype=np.float32)


# ============================================================
# Main loop – online training
# ============================================================

def run_interactive_model(
    base_dir: str,
    load_run_subdir: str,
    model_name: str,
    single_goal_mode: bool,
    start_goal=(0.0, -2.0),
    max_runtime_steps=100_000, 
):
    # ---------- Paths ----------
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

    # ---------- Load pretrained SAC ----------
    loaded_model = SAC.load(pretrained_path)

    # ---------- Env ----------
    env = AlbertTableEnv(render=True, goals=[start_goal])

    # ---------- Rebuild SAC; copy weights ----------
    model = SAC(
        policy=loaded_model.policy_class,
        env=env,
        learning_rate=loaded_model.learning_rate,
        buffer_size=loaded_model.replay_buffer.buffer_size,
        batch_size=loaded_model.batch_size,
        gamma=loaded_model.gamma,
        tau=loaded_model.tau,
        ent_coef=loaded_model.ent_coef,
        train_freq=loaded_model.train_freq,
        gradient_steps=loaded_model.gradient_steps,
        tensorboard_log=tb_dir,
        verbose=1,
    )
    model.policy.load_state_dict(loaded_model.policy.state_dict())
    model._setup_model()
    model.ep_info_buffer, model.ep_success_buffer = [], []

    # ---------- Logger ----------
    logger = configure(tb_dir, ["stdout", "tensorboard"])
    model.set_logger(logger)

    # ---------- Controller ----------
    js = init_ps5_controller()

    obs, _ = env.reset(Fh_override=np.zeros(3))
    total_steps, steps_since_goal, goal_id = 0, 0, 1
    goal = np.array(start_goal)
    goal_marker_id = draw_goal_marker(goal)

    Fprev = np.zeros(3, dtype=np.float32)

    # ==== Rolling buffers for averaging ====
    reward_window = []
    actor_loss_window, critic_loss_window = [], []
    ent_coef_window, ent_coef_loss_window = [], []
    learning_rate_window, n_updates_window = [], []


    # set the mode for 1 goal or multiple goals once reached
    

    try:
        while total_steps < max_runtime_steps:
            prev_obs = obs.copy()
            action, _ = model.predict(obs, deterministic=True)

            # --- Human force input ---
            f_local = read_joystick_force(js, force_scale=70.0)
            f_world = compute_world_force_from_table(f_local, env.sim.table_id)
            Fh = 0.8 * Fprev + 0.2 * np.array([f_world[0], f_world[1], 0.0], dtype=np.float32)
            Fprev = Fh.copy()

            # --- Step env ---
            obs, rew, terminated, truncated, info = env.step(action, Fh_override=Fh)
            done = terminated or truncated
            model.replay_buffer.add(prev_obs, obs, action,
                                    np.array([rew], dtype=np.float32),
                                    np.array([done], dtype=np.float32),
                                    [{}])

            # --- Train ---
            if model.replay_buffer.size() > 1024:
                model.train(batch_size=64, gradient_steps=1)
                # collect internal metrics
                for k, v in model.logger.name_to_value.items():
                    if k == "train/actor_loss": actor_loss_window.append(v)
                    elif k == "train/critic_loss": critic_loss_window.append(v)
                    elif k == "train/ent_coef": ent_coef_window.append(v)
                    elif k == "train/ent_coef_loss": ent_coef_loss_window.append(v)
                    elif k == "train/learning_rate": learning_rate_window.append(v)
                    elif k == "train/n_updates": n_updates_window.append(v)

            # --- Collect reward ---
            reward_window.append(rew)

            # --- Log averages every 100 steps ---
            if total_steps % 100 == 0 and reward_window:
                def mean_safe(x): return float(np.mean(x)) if len(x) > 0 else 0.0

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

                # reset rolling windows
                reward_window.clear()
                actor_loss_window.clear()
                critic_loss_window.clear()
                ent_coef_window.clear()
                ent_coef_loss_window.clear()
                learning_rate_window.clear()
                n_updates_window.clear()


            total_steps += 1
            steps_since_goal += 1

            # --- Print progress ---
            if total_steps % 300 == 0:
                print(f"Step {total_steps}, reward={rew:.3f}")

            # === DEBUG BLOCK: inspect all bodies every 500 steps ===
            if total_steps % 500 == 0:
                print("\n[DEBUG] ===============================")
                print(f"[DEBUG] Step {total_steps}")
                print(f"[DEBUG] Number of bodies: {p.getNumBodies()}")

                for body_id in range(p.getNumBodies()):
                    pos, orn = p.getBasePositionAndOrientation(body_id)

                    # Visual shape data tells us if rendering is broken
                    vdata = p.getVisualShapeData(body_id)

                    print(f"[DEBUG] Body {body_id}: pos={pos}, visual_shapes={len(vdata)}")

                print("[DEBUG] ===============================\n")


            # --- Timeout or goal reached ---
            if steps_since_goal >= 1000 and not terminated:
                print(f"Timeout: resetting SAME goal #{goal_id}")

                p.removeAllUserDebugItems()
                try:
                    p.removeBody(goal_marker_id)
                except Exception:
                    pass
                obs, _ = env.reset(Fh_override=np.zeros(3))
                steps_since_goal = 0
                goal_marker_id = draw_goal_marker(goal)
               
                continue

            # Goal reached -> new goal
            if terminated:
                print(f"Goal #{goal_id} reached!")

                # 🟢 Turn goal green before resetting
                set_goal_color(goal_marker_id, color=(0, 1, 0, 1))
                time.sleep(0.5)  # optional small pause to visualize success

                p.removeAllUserDebugItems() 

                    # 🧹 Remove old debug items and old goal marker
                try:
                    p.removeBody(goal_marker_id)
                except Exception:
                    pass

                if not single_goal_mode:
                    # Multiple goals mode — sample a new one
                    new_goal = sample_new_goal()
                    env.goals = [new_goal]
                    env.goal = new_goal
                    obs, _ = env.reset(Fh_override=np.zeros(3))
                    goal = new_goal
                    goal_id += 1
                    steps_since_goal = 0
                    goal_marker_id = draw_goal_marker(goal)  # new marker
                    
                else:
                    # Single goal mode — reset same goal
                    obs, _ = env.reset(Fh_override=np.zeros(3))
                    steps_since_goal = 0
                    goal_marker_id = draw_goal_marker(goal)  # redraw same goal


            # --- Exit on ESC ---
            keys = p.getKeyboardEvents()
            if 27 in keys and keys[27] & p.KEY_WAS_TRIGGERED:
                print("ESC → exit")
                break

            # --- Toggle between single/multi-goal mode with 'G' key ---
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
# Entry Point
# ============================================================

if __name__ == "__main__":
    base_dir = "runs_12_nov_test"
    load_run_subdir = "20251112-124408_12_nov_test"
    model_name = "12_nov_test"

    run_interactive_model(
        base_dir=base_dir,
        load_run_subdir=load_run_subdir,
        model_name=model_name,
        single_goal_mode=True,
        start_goal=(0.0, 2.0),
        max_runtime_steps=100_000,
    )
