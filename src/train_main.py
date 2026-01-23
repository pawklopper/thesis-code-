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
# !!! ADDED: Import BaseCallback for TensorBoard logging
from stable_baselines3.common.callbacks import BaseCallback

from rl_env.albert_table_env import AlbertTableEnv


# ============================================================================
# !!! ADDED: CALLBACK CLASS FOR TENSORBOARD
# ============================================================================
class RewardLoggerCallback(BaseCallback):
    """
    Custom callback to extract reward components from 'info'
    and log them to TensorBoard under the 'rewards/' tab.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Access the 'info' dictionary from the environment
        # self.locals['infos'] is a list of info dicts (one per env)
        if "infos" in self.locals:
            info = self.locals["infos"][0]
            
            # List of keys you want to see in TensorBoard
            # (Matches what you calculate in AlbertTableEnv)
            keys_to_log = [
                "progress_reward", 
                "motion_reward", 
                "distance_penalty", 
                "heading_penalty", 
                "obstacle_penalty",
            ]
            
            for key in keys_to_log:
                # We use .get() or check existence to prevent crashes
                if key in info:
                    self.logger.record(f"rewards/{key}", info[key])
                    
        return True


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
    render=False, 
    use_obstacles=False,
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
    # !!! FIX: Added comma after "total_reward" to fix syntax error
    env = Monitor(
        AlbertTableEnv(render=render, goals=goals, use_obstacles=use_obstacles),
        filename=os.path.join(log_dir, "monitor.csv"),
        info_keywords=(
            "total_reward",      # <--- Added comma here
            "progress_reward",
            "motion_reward",
            "distance_penalty",
            "heading_penalty",
            "obstacle_penalty", 
        )
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
    
    # !!! ADDED: Instantiate the callback
    callback = RewardLoggerCallback()

    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        tb_log_name="SAC",
        callback=callback  # !!! ADDED: Pass callback here
    )

    # === Save model (unchanged) ===
    model_path = os.path.join(log_dir, f"{model_name}.zip")
    model.save(model_path)

    env.close()
    print(f"\nTraining complete — model saved to {model_path}")

    # === Launch TensorBoard (unchanged) ===
    # Note: TensorBoard will look recursively. 
    # The logs will be inside log_dir/SAC_1/
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
    model_name = "experiment_16_jan_80000_"
    base_log_dir = "runs/offline/jan/runs_21_jan_test"
    render = False
    use_obstacles=False

    # === ORIGINAL GOAL LIST ===
    # goal_list = [
    # # -----------------------
    # # Near ring (~1.5 m)
    # # -----------------------
    # (0.0, 1.5),    # +Y
    # (1.5, 0.0),    # +X
    # (0.0, -1.5),   # -Y
    # (-1.5, 0.0),   # -X
    # (1.06, 1.06),   # NE
    # (-1.06, 1.06),  # NW
    # (-1.06, -1.06), # SW
    # (1.06, -1.06),  # SE

    # # -----------------------
    # # Mid ring (~2.5 m)
    # # -----------------------
    # (0.0, 2.5),
    # (2.5, 0.0),
    # (0.0, -2.5),
    # (-2.5, 0.0),
    # (1.77, 1.77),
    # (-1.77, 1.77),
    # (-1.77, -1.77),
    # (1.77, -1.77),

    # # -----------------------
    # # Far ring (~3.5 m)
    # # -----------------------
    # (0.0, 3.5),
    # (3.5, 0.0),
    # (0.0, -3.5),
    # (-3.5, 0.0),
    # (2.47, 2.47),
    # (-2.47, 2.47),
    # (-2.47, -2.47),
    # (2.47, -2.47),
# ]
    goal_list = [
    (0.0,  2.5),    # N
    (1.8,  1.8),    # NE
    (2.5,  0.0),    # E
    (1.8, -1.8),    # SE
    (0.0, -3.0),    # S
    (-1.8, -1.8),   # SW
    (-2.5,  0.0),   # W
    (-1.8,  1.8),   # NW
]


    # === ORIGINAL TRAINING CALL EXACTLY REPRODUCED ===
    model = train_sac(
        total_timesteps=80000,
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
        use_obstacles=use_obstacles
    )