#!/usr/bin/env python3
"""
train_collaboratively.py — REALISTIC VERSION (Steps A & B)
---------------------------------------------------
1. Loads pretrained SAC.
2. [STEP A] Loads offline Replay Buffer.
3. [STEP B] Warmup before training resumes.
4. Logs custom reward components to TensorBoard.
"""

import os
import time
import datetime
import numpy as np
import pybullet as p

from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure

from rl_env.albert_table_env import AlbertTableEnv
from controllers.ps5_human_control_impedance import PS5ImpedanceController

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
    warmup_steps=1024, 
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

    print(f"[LOAD MODEL]  {pretrained_path}")
    print(f"[SAVE ROOT]   {online_root}")

    # --------------------------------------------------------
    # Load pretrained SAC
    # --------------------------------------------------------
    model = SAC.load(pretrained_path)

    # --------------------------------------------------------
    # Attach new environment
    # --------------------------------------------------------
    # We pass 'options' to ensure reward_weights are set if needed (default is all 1.0)
    env = AlbertTableEnv(render=True, goals=[start_goal])
    model.set_env(env)

    # --------------------------------------------------------
    # Logger setup
    # --------------------------------------------------------
    logger = configure(tb_dir, ["stdout", "tensorboard"])
    model.set_logger(logger)
    
    # --------------------------------------------------------
    # Initial reset
    # --------------------------------------------------------
    obs, _ = env.reset(Fh_override=np.zeros(3))

    # --------------------------------------------------------
    # PS5 controller
    # --------------------------------------------------------
    human_controller = PS5ImpedanceController(
        table_id=env.sim.table_id, 
        link_idx=env.sim.human_goal_link_idx
    )
    
    start_pos = p.getLinkState(env.sim.table_id, env.sim.human_goal_link_idx)[0]
    human_controller.ghost_pos = np.array(start_pos[:2], dtype=np.float32)

    total_steps, steps_since_goal, goal_id = 0, 0, 1
    goal = np.array(start_goal)
    goal_marker_id = draw_goal_marker(goal)

    # --------------------------------------------------------
    # !!! ADDED: Rolling windows for individual rewards
    # --------------------------------------------------------
    reward_window = []
    
    # Windows for specific components
    progress_window = []
    motion_window = []
    dist_window = []
    head_window = []
    eff_window = []

    # Windows for training metrics
    actor_loss_window, critic_loss_window = [], []
    ent_coef_window, ent_coef_loss_window = [], []
    learning_rate_window, n_updates_window = [], []

    print(f"--> Starting Loop. Training will START after {warmup_steps} steps.")

    # =======================================================
    # MAIN LOOP
    # =======================================================
    try:
        while total_steps < max_runtime_steps:
            loop_start_time = time.time()
            
            prev_obs = obs.copy()

            # --- Policy action ---
            action, _ = model.predict(obs, deterministic=True)

            # --- Human force ---
            Fh = human_controller.step(dt=env.sim.dt)

            # --- Step environment ---
            obs, rew, terminated, truncated, info = env.step(action, Fh_override=Fh)
            done = terminated or truncated

            # -------------------------------------------------------
            # !!! ADDED: Collect individual rewards for logging
            # -------------------------------------------------------
            # We use .get() to be safe, defaulting to 0.0 if missing
            progress_window.append(info.get("progress_reward", 0.0))
            motion_window.append(info.get("motion_reward", 0.0))
            dist_window.append(info.get("distance_penalty", 0.0))
            head_window.append(info.get("heading_penalty", 0.0))
            eff_window.append(info.get("collaboration_reward", 0.0))
            reward_window.append(rew)

            # --- Add transition to replay buffer ---
            model.replay_buffer.add(
                prev_obs,
                obs,
                action,
                np.array([rew], dtype=np.float32),
                np.array([done], dtype=np.float32),
                [{}],
            )

            # -------------------------------------------------------
            # Training Logic
            # -------------------------------------------------------
            if model.replay_buffer.size() > warmup_steps:
                TRAIN_FREQ = 10 
                if total_steps % TRAIN_FREQ == 0:
                    model.train(batch_size=64, gradient_steps=TRAIN_FREQ)

                    for k, v in model.logger.name_to_value.items():
                        if k == "train/actor_loss": actor_loss_window.append(v)
                        elif k == "train/critic_loss": critic_loss_window.append(v)
                        elif k == "train/ent_coef": ent_coef_window.append(v)
                        elif k == "train/ent_coef_loss": ent_coef_loss_window.append(v)
                        elif k == "train/learning_rate": learning_rate_window.append(v)
                        elif k == "train/n_updates": n_updates_window.append(v)

            total_steps += 1
            steps_since_goal += 1

            # --- Reset / End of Episode Logic ---
            should_reset = False
            if steps_since_goal >= 1000 and not terminated:
                print(f"Timeout: resetting SAME goal #{goal_id}")
                should_reset = True
            elif terminated:
                print(f"Goal #{goal_id} reached!")
                set_goal_color(goal_marker_id, color=(0, 1, 0, 1))
                time.sleep(0.5)
                if not single_goal_mode:
                    new_goal = sample_new_goal()
                    env.goals = [new_goal]
                    env.goal = new_goal
                    goal = new_goal
                    goal_id += 1
                should_reset = True

            # -------------------------------------------------------
            # Logging & Resetting (At End of Episode)
            # -------------------------------------------------------
            if should_reset:
                # 1. Log stats for the finished episode
                Fr = env.sim.last_F_xy if hasattr(env.sim, "last_F_xy") else np.zeros(2)
                Fh_debug = env.sim.last_Fh_xy if hasattr(env.sim, "last_Fh_xy") else np.zeros(2)
                status = "WARMUP" if total_steps < warmup_steps else "TRAIN"
                
                print(
                    f"[{status}] Episode Finished (Step={total_steps}) | MeanRew={np.mean(reward_window):.2f} | "
                    f"Fr={Fr} | Fh={Fh_debug}"
                )

                if reward_window:
                    def mean_safe(x): return float(np.mean(x)) if len(x) > 0 else 0.0

                    # Standard metrics
                    model.logger.record("reward/episode_mean", mean_safe(reward_window))
                    model.logger.record("train/actor_loss", mean_safe(actor_loss_window))
                    model.logger.record("train/critic_loss", mean_safe(critic_loss_window))
                    model.logger.record("train/ent_coef", mean_safe(ent_coef_window))
                    
                    # Custom Reward Components
                    model.logger.record("rewards/progress", mean_safe(progress_window))
                    model.logger.record("rewards/motion", mean_safe(motion_window))
                    model.logger.record("rewards/dist_penalty", mean_safe(dist_window))
                    model.logger.record("rewards/head_penalty", mean_safe(head_window))
                    model.logger.record("rewards/collaboration_reward", mean_safe(eff_window))

                    model.logger.dump(step=total_steps)

                    # 2. Clear all windows for next episode
                    reward_window.clear()
                    progress_window.clear()
                    motion_window.clear()
                    dist_window.clear()
                    head_window.clear()
                    eff_window.clear()
                    
                    actor_loss_window.clear()
                    critic_loss_window.clear()
                    ent_coef_window.clear()
                    ent_coef_loss_window.clear()
                    learning_rate_window.clear()
                    n_updates_window.clear()

                # 3. Perform Reset
                p.removeAllUserDebugItems()
                try: p.removeBody(goal_marker_id)
                except: pass
                
                obs, _ = env.reset(Fh_override=np.zeros(3))
                start_pos = p.getLinkState(env.sim.table_id, env.sim.human_goal_link_idx)[0]
                human_controller.ghost_pos = np.array(start_pos[:2], dtype=np.float32)

                steps_since_goal = 0
                goal_marker_id = draw_goal_marker(goal)

            # --- Input Handling ---
            keys = p.getKeyboardEvents()
            if 27 in keys and keys[27] & p.KEY_WAS_TRIGGERED: # ESC
                break
            if ord('g') in keys and keys[ord('g')] & p.KEY_WAS_TRIGGERED:
                single_goal_mode = not single_goal_mode
                print(f"Toggled single_goal_mode → {single_goal_mode}")

            # --- Loop Timing ---
            elapsed = time.time() - loop_start_time
            sleep_time = env.sim.dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        env.close()
        final_zip = os.path.join(online_root, f"{model_name}_final.zip")
        final_buffer = os.path.join(online_root, f"{model_name}_final_buffer.pkl")
        
        model.save(final_zip)
        model.save_replay_buffer(final_buffer)
        
        print(f"[FINAL SAVE] {final_zip}")
        print(f"[BUFFER SAVED] {final_buffer}")
        print(f"[TB LOGDIR]  {tb_dir}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    base_dir = "runs_27_nov_test"
    load_run_subdir = "20251127-150342_27_nov_test" # Update this to your latest run!
    model_name = "27_nov_test"

    run_interactive_model(
        base_dir=base_dir,
        load_run_subdir=load_run_subdir,
        model_name=model_name,
        single_goal_mode=False,
        start_goal=(0.0, -3.0),
        max_runtime_steps=100_000,
        warmup_steps=1002 
    )