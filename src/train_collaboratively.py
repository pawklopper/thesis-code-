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

from mobile_sim.src.controllers.ps5_human_control import (
    init_ps5_controller,
    read_joystick_force,
    compute_world_force_from_table,
)


# ============================================================
# Helpers
# ============================================================

def draw_goal_marker(goal_xy, color=(1, 0, 0, 1)):
    goal_pos = [float(goal_xy[0]), float(goal_xy[1]), 0.05]

    visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.07,
        rgbaColor=color,
    )
    mid = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual,
        basePosition=goal_pos,
    )
    p.addUserDebugText(
        "Goal",
        [goal_pos[0], goal_pos[1], 0.15],
        [1, 1, 1],
        textSize=1.2,
    )
    return mid


def sample_new_goal(rmin=1.0, rmax=3.0):
    theta = np.random.uniform(-np.pi, np.pi)
    r = np.random.uniform(rmin, rmax)
    return np.array([r * np.cos(theta), r * np.sin(theta)], dtype=np.float32)


# ============================================================
# Main loop – online training
# ============================================================

def run_interactive_model(
    base_log_dir,
    model_name,
    run_subdir,
    start_goal=(0.0, -2.0),
    max_runtime_steps=100_000,
):

    run_dir = os.path.join(base_log_dir, run_subdir)
    model_path = os.path.join(run_dir, f"{model_name}.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join("runs_interactive_updates", timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Updated models saved to: {save_dir}")

    env = AlbertTableEnv(render=True, goals=[start_goal])

    loaded_model = SAC.load(model_path)

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
        tensorboard_log=os.path.join(run_dir, "online_tb")
    )

    logger = configure(os.path.join(run_dir, "online_tb"), ["stdout", "tensorboard"])
    model.set_logger(logger)

    model.policy.load_state_dict(loaded_model.policy.state_dict())

    model._setup_model()

    print(f"Loaded model: {model_path}")

    js = init_ps5_controller()

    obs, _ = env.reset(Fh_override=np.zeros(3))

    total_steps = 0
    steps_since_goal = 0

    goal = np.array(start_goal)
    goal_id = 1
    draw_goal_marker(goal)

    Fprev = np.zeros(3, dtype=np.float32)


    while total_steps < max_runtime_steps:

        prev_obs = obs.copy()

        action, _ = model.predict(obs, deterministic=False)

        f_local = read_joystick_force(js, force_scale=70.0)
        f_world = compute_world_force_from_table(f_local, env.sim.table_id)
        Fh = np.array([f_world[0], f_world[1], 0.0], dtype=np.float32)
        Fh = 0.8 * Fprev + 0.2 * Fh
        Fprev = Fh.copy()

        obs, rew, terminated, truncated, info = env.step(action, Fh_override=Fh)

        done = terminated or truncated

        model.replay_buffer.add(
            prev_obs,
            obs,
            action,
            np.array([rew], dtype=np.float32),
            np.array([done], dtype=np.float32),
            [{}],
        )

        if model.replay_buffer.size() > 1024:
            model.train(batch_size=64, gradient_steps=1)

        total_steps += 1
        steps_since_goal += 1

        if total_steps % 300 == 0:
            print(f"Step {total_steps}, reward={rew:.3f}")

        if steps_since_goal >= 1000 and not terminated:
            print(f"Timeout: resetting SAME goal #{goal_id}")
            obs, _ = env.reset(Fh_override=np.zeros(3))

            steps_since_goal = 0
            draw_goal_marker(goal)
            continue

        if terminated:
            print(f"Goal #{goal_id} reached!")

            new_goal = sample_new_goal()
            env.goals = [new_goal]
            env.goal = new_goal
            obs, _ = env.reset(Fh_override=np.zeros(3))


            goal = new_goal
            goal_id += 1
            steps_since_goal = 0
            draw_goal_marker(goal)

        keys = p.getKeyboardEvents()
        if 27 in keys and keys[27] & p.KEY_WAS_TRIGGERED:
            print("ESC → exit")
            break

        time.sleep(env.sim.dt)

        if total_steps % 2000 == 0:
            save_path = os.path.join(save_dir, f"updated_{model_name}_{total_steps}.zip")
            model.save(save_path)
            print(f"Saved updated model → {save_path}")

    env.close()
    print(f"Finished after {total_steps} steps.")


# MAIN
if __name__ == "__main__":

    # run to test, forces set to zero in observation space
    base_log_dir = "runs_11_nov_test"
    model_name = "11_nov_test"
    run_subdir = "20251111-220821_11_nov_test"

    # run to test, forces included!
    # base_log_dir = "runs_11_nov_test"
    # model_name = "11_nov_test"
    # run_subdir = "20251111-212814_11_nov_test"

    # run to test, forces set in observation space

    run_interactive_model(
        base_log_dir,
        model_name,
        run_subdir,
        start_goal=(0.0, 2.0),
        max_runtime_steps=100_000,
    )
