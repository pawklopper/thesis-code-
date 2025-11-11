#!/usr/bin/env python3
"""
albert_sac.py
-------------
Gymnasium-compatible reinforcement learning environment for the Albert mobile base.
Uses Soft Actor-Critic (SAC) from Stable-Baselines3 to learn to drive from A → B.

Action space: [v_forward (m/s), yaw_rate (rad/s)]
Observation: [base_x, base_y, base_yaw]
Reward: + progress towards goal − control penalty
Termination: when close to goal or max steps reached
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import SAC

# URDFEnv / robot imports
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv


# ============================================================
# ==============   Albert Drive Environment   ================
# ============================================================

class AlbertDriveEnv(gym.Env):
    """Minimal Gymnasium environment for controlling Albert’s mobile base."""

    def __init__(self, render: bool = False):
        super().__init__()
        self.render = render

        # Action: [v_forward, yaw_rate]
        self.action_space = spaces.Box(
            low=np.array([-0.5, -1.0], dtype=np.float32),
            high=np.array([0.5, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation: [base_x, base_y, base_yaw]
        self.observation_space = spaces.Box(
            low=np.array([-10, -10, -np.pi], dtype=np.float32),
            high=np.array([10, 10, np.pi], dtype=np.float32),
            dtype=np.float32,
        )

        # Goal and step tracking
        self.goal = np.array([0.0, -2.0], dtype=np.float32)
        self.prev_dist = None
        self.step_count = 0
        self.max_steps = 500

        # --- Create the robot and simulation environment ---
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

        # ✅ Use the same environment setup as the original visualization
        self.env = UrdfEnv(
            dt=0.01,
            robots=[self.robot],
            render=self.render,
        )

    # ------------------------------------------------------------
    #  Reset and step
    # ------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """Reset the simulation and return the initial observation."""
        super().reset(seed=seed)

        # ✅ Same initial position as in run_albert()
        pos_vec = np.array([
            0.0, 1.0, 0.0,  # base x, y, yaw
            0.0, 0.0, 0.0,  # placeholder
            -1.5, 0.0, 1.8, 0.5  # arm joint configuration
        ])

        ob = self.env.reset(pos=pos_vec)
        base_pos = ob[0]["robot_0"]["joint_state"]["position"][:3]

        self.prev_dist = np.linalg.norm(self.goal - base_pos[:2])
        self.step_count = 0
        return np.array(base_pos, dtype=np.float32), {}

    def step(self, action):
        """Apply the action [v_forward, yaw_rate] and return next state + reward."""
        # Build full action vector for PyBullet env
        full = np.zeros(self.env.n(), dtype=float)
        full[0] = float(action[0])  # forward linear velocity
        full[1] = float(action[1])  # yaw rate

        ob = self.env.step(full)
        base_pos = ob[0]["robot_0"]["joint_state"]["position"][:3]

        # Compute distance and progress
        dist = np.linalg.norm(self.goal - base_pos[:2])
        progress = self.prev_dist - dist
        self.prev_dist = dist

        # Reward: positive for moving closer, small penalty for control effort
        control_penalty = 0.01 * np.sum(np.square(action))
        reward = progress - control_penalty

        # Termination conditions
        self.step_count += 1
        terminated = dist < 0.1
        truncated = self.step_count >= self.max_steps

        info = {"dist_to_goal": dist, "progress": progress}

        return np.array(base_pos, dtype=np.float32), reward, terminated, truncated, info

    def close(self):
        """Clean up simulation resources."""
        self.env.close()


# ============================================================
# ====================   SAC Training   =======================
# ============================================================

def train_sac(total_timesteps: int = 50_000):
    """
    Train the SAC algorithm on the AlbertDriveEnv.
    The agent learns to drive from its start to the goal position.
    """
    print("\n🚀 Starting SAC training...")
    env = AlbertDriveEnv(render=False)  # headless training mode

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256,
        tau=0.02,
        gamma=0.99,
        train_freq=64,
        gradient_steps=64,
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps)
    model.save("sac_albert_drive")
    print("\n✅ Training complete. Model saved as sac_albert_drive.zip")
    env.close()
    return model


# ============================================================
# ====================   Evaluation   ========================
# ============================================================

def evaluate_model(model_path: str = "sac_albert_drive", episodes: int = 3):
    """Load the trained model and test it in the simulation."""
    print("\n🎥 Evaluating trained model...")
    env = AlbertDriveEnv(render=True)
    model = SAC.load(model_path, env=env)

    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        print(f"Episode {ep+1}: total_reward={total_reward:.3f}, final_dist={info['dist_to_goal']:.3f}")
    env.close()


# ============================================================
# ====================   Main Script   ========================
# ============================================================

if __name__ == "__main__":
    # --- Quick test: random actions ---
    print("\n🧪 Running random-action test...")
    env = AlbertDriveEnv(render=True)
    obs, _ = env.reset()
    total_reward = 0
    for i in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if i % 20 == 0:
            print(f"Step {i:03d}: pos={obs}, reward={reward:.3f}, dist={info['dist_to_goal']:.3f}")
        if terminated or truncated:
            print("Episode ended early.")
            break
    print(f"Total reward (random actions): {total_reward:.3f}")
    env.close()

    # --- Train the agent ---
    model = train_sac(total_timesteps=50_000)

    # --- Evaluate the trained policy ---
    evaluate_model()
