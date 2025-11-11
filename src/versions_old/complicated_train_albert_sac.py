#!/usr/bin/env python3
"""
train_albert_sac.py
-------------------
Continue (or start) SAC training for the Albert mobile base.
Uses egocentric observations, curriculum learning, and goal-conditioned rewards.
If a previous SAC model is found, it automatically resumes fine-tuning.
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from helper_plots import plot_monitor_csv

from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv


# ============================================================
# ==============   Albert Drive Environment   ================
# ============================================================

class AlbertDriveEnv(gym.Env):
    """Goal-conditioned environment for Albert mobile base (no heading term)."""

    def __init__(self, render: bool = False, max_steps: int = 1000, log_dir: str = "logs/"):
        super().__init__()
        self.render = render
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # -------- Curriculum parameters --------
        self.last_goal = None
        self.reuse_failed_goal = True
        self.last_episode_success = True

        self.global_step = 0
        self.global_episode = 0  # NEW — global episode counter
        self.episode_total_reward = 0.0

        self.curriculum_min_radius = 1.0
        self.curriculum_max_radius = 3.0
        self.curriculum_radius = self.curriculum_min_radius
        self.curriculum_success_window = 20
        self.curriculum_success_target = 0.7
        self.curriculum_backoff_threshold = 0.3
        self.curriculum_step = 0.5
        self.episode_successes = []

        # Action/observation spaces
        self.action_space = spaces.Box(
            low=np.array([-0.5, -1.0], dtype=np.float32),
            high=np.array([0.5, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=np.array([0.0, -2.0, -2.0], dtype=np.float32),
            high=np.array([10.0, 2.0, 2.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Internal state
        self.goal = np.zeros(2, dtype=np.float32)
        self.prev_dist = None
        self.step_count = 0
        self.max_steps = max_steps

        # Robot + sim
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
        self.env = UrdfEnv(dt=0.01, robots=[self.robot], render=self.render)

    # ------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------

    def _compute_obs(self, base_pos, base_yaw, v, w):
        dx = self.goal[0] - base_pos[0]
        dy = self.goal[1] - base_pos[1]
        dist = np.hypot(dx, dy)
        obs = np.array([dist, v, w], dtype=np.float32)
        return obs, dist

    def _compute_reward(self, action, progress, dist):
        control_penalty = 0.001 * np.sum(np.square(action))
        time_penalty = 0.001
        reward = 10.0 * progress - control_penalty - time_penalty

        success_bonus = 0.0
        terminated = False
        if dist < 0.3:
            success_bonus = 30.0
            reward += success_bonus
            terminated = True

        info = {
            "progress": progress,
            "control_penalty": control_penalty,
            "time_penalty": time_penalty,
            "success_bonus": success_bonus,
            "dist_to_goal": dist,
        }
        return float(reward), terminated, info

    # ------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # ---- Curriculum update ----
        rate = None
        if len(self.episode_successes) >= self.curriculum_success_window:
            rate = float(np.mean(self.episode_successes[-self.curriculum_success_window:]))
            if rate >= self.curriculum_success_target and self.curriculum_radius < self.curriculum_max_radius:
                self.curriculum_radius = min(self.curriculum_radius + self.curriculum_step, self.curriculum_max_radius)
            elif rate <= self.curriculum_backoff_threshold and self.curriculum_radius > self.curriculum_min_radius:
                self.curriculum_radius = max(self.curriculum_radius - self.curriculum_step, self.curriculum_min_radius)

        print(f"[DEBUG] step={self.global_step}, radius={self.curriculum_radius:.2f}, "
              f"min={self.curriculum_min_radius:.2f}, max={self.curriculum_max_radius:.2f}, "
              f"rate={rate if rate is not None else 'N/A'}")

        # Fixed start pose
        start_x, start_y, start_yaw = 0.0, 0.0, 0.0
        pos_vec = np.array([start_x, start_y, start_yaw, 0, 0, 0, -1.5, 0, 1.8, 0.5])
        ob = self.env.reset(pos=pos_vec)

        base_pos = ob[0]["robot_0"]["joint_state"]["position"][:3]

        # Goal sampling (reuse or new)
        if self.reuse_failed_goal and not self.last_episode_success and self.last_goal is not None:
            self.goal[:] = self.last_goal
            print(f"[Curriculum] Reusing failed goal {self.goal}")
        else:
            theta = np.random.uniform(-np.pi, np.pi)
            u = np.random.rand()
            r = max(self.curriculum_min_radius, np.sqrt(u) * self.curriculum_radius)
            goal_x = start_x + r * np.cos(theta)
            goal_y = start_y - abs(r * np.sin(theta))  # always in negative Y plane
            self.goal[:] = np.clip([goal_x, goal_y], -10.0, 10.0)

        # Steps budget proportional to goal distance
        goal_dist = float(np.linalg.norm(self.goal - base_pos[:2]))
        v_eff = 0.30
        slack = 1.3
        steps_per_meter = int((1.0 / v_eff) / self.env.dt * slack)
        self.max_steps = int(np.clip(steps_per_meter * goal_dist, 500, 2000))
        self.step_count = 0
        self.episode_total_reward = 0.0  # reset per-episode reward accumulator

        # ----- Curriculum logging -----
        radius_path = os.path.join(self.log_dir, "curriculum_radius.csv")
        if self.global_step == 0 or not os.path.exists(radius_path):
            with open(radius_path, "w") as f:
                f.write("step,radius,success_rate,change\n")

        change = "steady"
        if rate is not None:
            if rate >= self.curriculum_success_target:
                change = "increase"
            elif rate <= self.curriculum_backoff_threshold:
                change = "decrease"

        with open(radius_path, "a") as f:
            f.write(f"{self.global_step},{self.curriculum_radius:.3f},{rate if rate is not None else 'NA'},{change}\n")

        self.prev_obs, self.prev_dist = self._compute_obs(base_pos, start_yaw, 0.0, 0.0)
        return self.prev_obs.copy(), {}

    # ------------------------------------------------------------
    # Step
    # ------------------------------------------------------------

    def step(self, action):
        full = np.zeros(self.env.n(), dtype=float)
        full[0], full[1] = action
        ob = self.env.step(full)

        base_state = ob[0]["robot_0"]["joint_state"]
        base_pos = base_state["position"][:3]
        base_vel = base_state["velocity"]
        v, w = base_vel[0], base_vel[2]

        obs, dist = self._compute_obs(base_pos, base_pos[2], v, w)
        progress = self.prev_dist - dist
        self.prev_dist = dist

        reward, terminated, info = self._compute_reward(action, progress, dist)
        self.episode_total_reward += reward

        # Log actions
        actions_path = os.path.join(self.log_dir, "actions.csv")
        if not os.path.exists(actions_path):
            with open(actions_path, "w") as f:
                f.write("v,w\n")
        with open(actions_path, "a") as f:
            f.write(f"{v},{w}\n")

        truncated = self.step_count >= self.max_steps

        # 🚨 Apply failure penalty at the end if not successful
        if truncated and not terminated:
            failure_penalty = -10.0
            reward += failure_penalty
            self.episode_total_reward += failure_penalty
            info["failure_penalty"] = failure_penalty
        else:
            info["failure_penalty"] = 0.0

        # ---- Episode bookkeeping ----
        if terminated or truncated:
            self.global_episode += 1
            success = 1 if terminated else 0
            self.episode_successes.append(success)
            if len(self.episode_successes) > self.curriculum_success_window:
                self.episode_successes.pop(0)
            success_rate = float(np.mean(self.episode_successes)) if self.episode_successes else 0.0

            self.last_episode_success = bool(terminated)
            if not terminated:
                self.last_goal = self.goal.copy()

            # Log episode summary
            distances_path = os.path.join(self.log_dir, "final_distances.csv")
            if self.global_episode == 1 or not os.path.exists(distances_path):
                with open(distances_path, "w") as f:
                    f.write("episode,final_distance,success,steps,reward_sum,curriculum_radius,success_rate\n")

            with open(distances_path, "a") as f:
                f.write(f"{self.global_episode},{dist:.3f},{success},{self.step_count},"
                        f"{self.episode_total_reward:.3f},{self.curriculum_radius:.3f},{success_rate:.3f}\n")

            print(f"[LOG] Ep {self.global_episode:04d}: dist={dist:.3f}, success={success}, "
                  f"steps={self.step_count}, totalR={self.episode_total_reward:.2f}, "
                  f"radius={self.curriculum_radius:.2f}, rate={success_rate:.2f}")

        self.step_count += 1
        self.global_step += 1
        return obs, reward, terminated, truncated, info


    def close(self):
        self.env.close()


# ============================================================
# ====================   SAC Training   =======================
# ============================================================

def train_sac(total_timesteps: int, resume_training: bool, model_name: str, log_dir: str = "logs/"):
    print("\n🚀 Starting SAC training (curriculum + progress-only reward)...")
    os.makedirs(log_dir, exist_ok=True)

    env = Monitor(AlbertDriveEnv(render=False, log_dir=log_dir),
                  filename=os.path.join(log_dir, "monitor.csv"))
    model_path = f"{model_name}.zip"

    if resume_training and os.path.exists(model_path):
        print(f"✅ Found existing model: {model_path}")
        model = SAC.load(model_path, env=env)
        model.learning_rate = 2e-4
        if hasattr(model.replay_buffer, "clear"):
            model.replay_buffer.clear()
            print("🧹 Cleared replay buffer.")
        reset_num = False
    else:
        print("🆕 Starting FRESH training (resume_training=False).")
        model = SAC(
            "MlpPolicy", env,
            learning_rate=3e-4,
            buffer_size=300_000,
            batch_size=256,
            tau=0.02, gamma=0.99,
            train_freq=64, gradient_steps=64,
            verbose=1,
            tensorboard_log=os.path.join(log_dir, model_name),
        )
        reset_num = True

    model.learn(total_timesteps=total_timesteps, tb_log_name=f"SAC_{model_name}", reset_num_timesteps=reset_num)
    model.save(model_path)

    print(f"\n✅ Training complete — model saved as {model_path}")
    plot_monitor_csv(log_dir)
    env.close()
    return model


# ============================================================
# ====================   Evaluation   ========================
# ============================================================

def evaluate_model(model_path: str, episodes: int = 10):
    print("\n🎥 Evaluating trained model...")
    env = AlbertDriveEnv(render=True)
    model = SAC.load(model_path, env=env)

    start = (0, 0)
    fixed_pairs = [
        (start, (0, -2)),
        (start, (0, -3)),
        (start, (0, -4)),
        (start, (-1, -2)),
        (start, (-1, 2)),
        (start, (-1, 3)),
        (start, (1, 2)),
    ]

    results = []
    for i, (start_xy, goal) in enumerate(fixed_pairs[:episodes]):
        print(f"Episode {i+1}: start={start_xy}, goal={goal}")
        env.goal = np.array(goal, dtype=np.float32)
        env.env.reset(pos=np.array([*start_xy, 0, 0, 0, -1.5, 0, 1.8, 0.5]))
        obs, _ = env.reset()
        total_reward = 0.0
        for _ in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        results.append(info["dist_to_goal"])
        print(f"  total_reward={total_reward:.2f}, final_dist={info['dist_to_goal']:.3f}")

    print(f"\nMean final distance: {np.mean(results):.3f} m")
    env.close()


# ============================================================
# ====================   Main Script   ========================
# ============================================================

if __name__ == "__main__":
    print("\n⚙️  Fine-tuning Albert SAC model (no heading term)...")
    total_timesteps = 180000
    resume_training = False
    model_name = "sac_albert_drive_no_yaw_error_180000"

    model = train_sac(total_timesteps=total_timesteps, resume_training=resume_training, model_name=model_name)

    print("\n📊 Evaluation on fixed set...")
    evaluate_model(model_path=f"{model_name}.zip")
