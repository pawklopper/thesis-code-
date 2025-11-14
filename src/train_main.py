#!/usr/bin/env python3
"""
Unified SAC training script for Albert + Table + Human.

This file merges:
 - train_sac.py  (training logic)
 - the original main execution
into ONE script without changing logic.

It uses the modular environment:
    rl_env/albert_table_env.py
"""

from __future__ import annotations
import os
import subprocess
import warnings
from datetime import datetime

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from rl_env.albert_table_env import AlbertTableEnv


# ============================================================================
# ============================= TRAINING FUNCTION ============================
# ============================================================================

def train_sac(
    total_timesteps,
    goals,
    model_name,
    base_log_dir,
    learning_rate,
    buffer_size,
    batch_size,
    tau,
    gamma,
    train_freq,
    gradient_steps,
    render=False
):
    """
    Launch SAC training with simplified Albert–table environment.
    Logic preserved 1:1 from the original monolithic file.
    """

    # === Create run directory (ORIGINAL LOGIC) ===
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_{model_name}"
    log_dir = os.path.join(base_log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)

    # === Create environment wrapped with Monitor ===
    env = Monitor(
        AlbertTableEnv(render=render, goals=goals),
        filename=os.path.join(log_dir, "monitor.csv")
    )

    # === SAC MODEL (unchanged) ===
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        verbose=1,
        tensorboard_log=log_dir
    )

    print("\nStarting SAC training...")
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        tb_log_name="SAC"
    )

    # === Save model (unchanged) ===
    model_path = os.path.join(log_dir, f"{model_name}.zip")
    model.save(model_path)

    env.close()
    print(f"\nTraining complete — model saved to {model_path}")

    # === Launch TensorBoard (unchanged) ===
    subprocess.Popen(["tensorboard", "--logdir", base_log_dir, "--port", "6006"])
    print("Open http://localhost:6006 to view live training metrics.")

    return model


# ============================================================================
# =============================== MAIN EXECUTION =============================
# ============================================================================

if __name__ == "__main__":

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
    
    # === ORIGINAL VARIABLE NAMES PRESERVED EXACTLY ===
    model_name = "13_nov_test"
    base_log_dir = "runs/offline/runs_13_nov_test"
    render = False

    # === ORIGINAL GOAL LIST ===
    goal_list = [(0.0, 2.0)]

    # === ORIGINAL TRAINING CALL EXACTLY REPRODUCED ===
    model = train_sac(
        total_timesteps=20000,
        goals=goal_list,
        model_name=model_name,
        base_log_dir=base_log_dir,
        learning_rate=3e-4,
        buffer_size=50_000,
        batch_size=64,
        tau=0.02,
        gamma=0.99,
        train_freq=32,
        gradient_steps=32,
        render=render,
    )
