#!/usr/bin/env python3
"""
train_albert_sac_simple_manual_full.py
--------------------------------------
Fully manual version of SAC training for the Albert mobile base.
You can configure *all* key parameters (goal, hyperparameters, model name,
and logging directory) directly in the main block below.
"""


#=============================================
#===========REWARD FUNCTION TODOS=============
#=============================================

# 2. Policy can spiral around when the goal is hard -? e.g. in an action space it has not been yet, for example (2, 2) goal it keeps spiralling
# 4. fixed in on (-2, -2) works very decently for that octant
# 5. RL algorithm has trouble when its on a line with the goal but it has to drive backwards. 

import os
import subprocess
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv


# ============================================================
# =============== Simplified Albert Environment ===============
# ============================================================

class SimpleAlbertEnv(gym.Env):
    """Minimal goal-reaching environment for Albert mobile base (random goal from list)."""

    def __init__(self, render: bool = False, max_steps: int = 1000, goals=None):
        super().__init__()
        self.render_mode = render
        self.max_steps = max_steps
        self.step_count = 0

        # If no goals are passed, fall back to a single fixed one
        if goals is None:
            goals = [(1.5, -2.0)]
        self.goals = [np.array(g, dtype=np.float32) for g in goals]
        self.goal = self.goals[0]

        # --- Action and observation spaces ---
        self.action_space = spaces.Box(
            low=np.array([-0.5, -1.0], dtype=np.float32),
            high=np.array([0.5, 1.0], dtype=np.float32),
            dtype=np.float32,
        )


        # observations: [dx, dy, v, w, cos(heading_error), sin(heading_error)]

        self.observation_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )


        # --- Robot + simulation setup ---
        self.robot = GenericDiffDriveRobot(
            urdf="albert.urdf",
            mode="vel",
            actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
            castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
            wheel_radius=0.08,
            wheel_distance=0.494,
            spawn_rotation=0,
            facing_direction='-y',
        )

        self.env = UrdfEnv(dt=0.01, robots=[self.robot], render=self.render_mode)
        self.prev_dist = None
        self._last_env_ob = None  # 👈 Cache for last observation

    # ------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # === Randomly select one of the fixed goals provided ===
        self.goal = self.goals[np.random.randint(len(self.goals))]

        # Fixed start pose
        pos_vec = np.array([0, 0, 0, 0, 0, 0, -1.5, 0, 1.8, 0.5])
        ob = self.env.reset(pos=pos_vec)
        self._last_env_ob = ob  # 👈 Store the observation from reset

        print(f"[INFO] Selected goal: {tuple(self.goal)}")

        obs = self._get_obs()
        self.prev_dist = np.hypot(obs[0], obs[1])
        return obs, {}

    # ------------------------------------------------------------
    def step(self, action):
        full = np.zeros(self.env.n(), dtype=float)
        full[0], full[1] = action
        ob = self.env.step(full)
        self._last_env_ob = ob  # 👈 Always store latest observation

        # --- Extract robot pose ---
        base_state = ob[0]["robot_0"]["joint_state"]
        base_pos = base_state["position"][:3]
        base_vel = base_state["velocity"]
        v, w = base_vel[0], base_vel[2]
        x, y, yaw = base_pos

        # --- Compute relative goal vector ---
        dx = self.goal[0] - x
        dy = self.goal[1] - y
        dist = np.hypot(dx, dy)

        # --- Compute heading error (goal direction vs robot yaw) ---
        angle_to_goal_world = np.arctan2(dy, dx)
        angle_to_goal_adj = angle_to_goal_world + np.pi / 2.0   # because robot faces -y
        heading_error = angle_to_goal_adj - yaw
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        #old ops does not include heading error  
        #obs = np.array([dx, dy, v, w], dtype=np.float32)

        obs = np.array([dx, dy, v, w,
                np.cos(heading_error), np.sin(heading_error)], dtype=np.float32)

    

        #reward = -dist
        #-------------------------------------------
        #-----------------REWARD--------------------
        #-------------------------------------------
        progress_reward = (self.prev_dist - dist) * 100 
        distance_penalty = -dist 
        # older reward did not have heading penalty, worked ok but definetly not perfectly
        heading_penalty = -abs(heading_error) * 0.5

        reward = progress_reward + distance_penalty + heading_penalty
        
        #reset prev_dist to now new distance
        self.prev_dist = dist

        #succes bonus
        if dist < 0.3:
            succes_bonus = 25
            reward += succes_bonus

        terminated = dist < 0.3
        truncated = self.step_count >= self.max_steps

        self.step_count += 1
        info = {"dist_to_goal": dist}
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------
    def _get_obs(self):
        """Use last cached observation (compatible with older urdfenvs)."""
        if self._last_env_ob is None:
            raise RuntimeError("No observation available. Did you call reset()?")

        base_state = self._last_env_ob[0]["robot_0"]["joint_state"]
        base_pos = base_state["position"][:3]
        base_vel = base_state["velocity"]
        v, w = base_vel[0], base_vel[2]
        x, y, yaw = base_pos

        dx = self.goal[0] - x
        dy = self.goal[1] - y

        # --- Compute heading error (same as in step) ---
        angle_to_goal_world = np.arctan2(dy, dx)
        angle_to_goal_adj = angle_to_goal_world + np.pi / 2.0   # robot faces -y
        heading_error = angle_to_goal_adj - yaw
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        # --- Always return consistent shape (6,) ---
        return np.array(
            [dx, dy, v, w, np.cos(heading_error), np.sin(heading_error)],
            dtype=np.float32,
        )



    def close(self):
        self.env.close()


# ============================================================
# =============== Training Function ==========================
# ============================================================

def train_sac(
    total_timesteps: int,
    goals: list,
    model_name: str,
    base_log_dir: str,
    learning_rate: float,
    buffer_size: int,
    batch_size: int,
    tau: float,
    gamma: float,
    train_freq: int,
    gradient_steps: int,
    render: bool = False,
):
    # Timestamped run directory inside base_log_dir
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_{model_name}"
    log_dir = os.path.join(base_log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n🚀 Starting SAC training for Albert")
    print(f" Model name: {model_name}")
    print(f" Goals: {goals}")
    print(f" Logs directory: {log_dir}\n")

    env = Monitor(SimpleAlbertEnv(render=render, goals=goals),
                  filename=os.path.join(log_dir, "monitor.csv"))

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
        tensorboard_log=log_dir,
    )

    model.learn(total_timesteps=total_timesteps, progress_bar=True, tb_log_name="SAC")
    model_path = os.path.join(log_dir, f"{model_name}.zip")
    model.save(model_path)
    env.close()

    # === Launch TensorBoard automatically ===
    print(f"\n✅ Training complete — model saved to {model_path}")
    print("🚀 Launching TensorBoard...")
    subprocess.Popen(["tensorboard", "--logdir", base_log_dir, "--port", "6006"])
    print("🌐 Open http://localhost:6006 to view live training metrics.")

    print(f"\n✅ Training complete — model saved to {model_path}")
    print(f"📁 All logs and checkpoints stored in: {log_dir}")
    return model


# ============================================================
# =============== Main (fully manual config) =================
# ============================================================

if __name__ == "__main__":
    # ----------------------------
    # MANUAL CONFIGURATION SECTION
    # ----------------------------

    # ----- General settings -----
    model_name = "sac_albert_random_goals_manual_test"
    base_log_dir = "basic_runs_albert"        # all logs and models go here
    render = False                      # True = visualize training

    # ----- Goal setup -----
    goal_list = [
        ( 2.0,  2.0),
        ( 2.0, -2.0),
        (-2.0,  2.0),
        (-2.0, -2.0),
    ]
    # ----- Training control -----
    total_timesteps = 60000           # total training steps
    max_steps_per_episode = 1000        # per episode inside env

    # ----- SAC hyperparameters -----
    learning_rate = 3e-4
    buffer_size = 200_000
    batch_size = 256
    tau = 0.02
    gamma = 0.99
    train_freq = 64
    gradient_steps = 64

    # ----------------------------
    # START TRAINING
    # ----------------------------
    model = train_sac(
        total_timesteps=total_timesteps,
        goals=goal_list,
        model_name=model_name,
        base_log_dir=base_log_dir,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        render=render,
    )
