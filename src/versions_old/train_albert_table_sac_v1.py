#!/usr/bin/env python3
"""
train_albert_table_sac.py
-------------------------------
SAC training for Albert + Table with impedance coupling.

⚙️ Simplified version:
- Observation only includes robot-centric states tied to the reward:
    [dx, dy, v, w, heading_cos_bi, heading_sin_bi]
- Table and impedance remain simulated for realism, but not observed.
"""

import os
import time
import warnings
import subprocess
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
import pybullet as p
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv


# =======================================================
# =================TO DO's ==============================
# =======================================================



# ============================================================
# ============== Albert + Table Impedance Sim ================
# ============================================================

class AlbertTableImpedanceSim:
    """Encapsulated Albert + Table impedance controller simulation."""

    def __init__(self, render=True):
        self.render = render
        self.dt = 0.01
        self.arm_joint_indices = [7, 8, 9, 10, 11, 12, 13]
        self.ee_idx = 16
        self.Kp = np.diag([300.0, 300.0, 0.0])
        self.Dp = np.diag([60.0, 60.0, 0.0])
        self.F_max = np.array([150.0, 150.0, 150.0])
        self.tau_max = 25.0
        self.env = None
        self.albert_id = None
        self.table_id = None
        self.goal_link_idx = None

    def create_environment(self):
        """Create URDF environment and spawn Albert robot."""
        print("\n[DEBUG] Creating URDF environment with Albert robot...")
        robot = GenericDiffDriveRobot(
            urdf="albert.urdf",
            mode="vel",
            actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
            castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
            wheel_radius=0.08,
            wheel_distance=0.494,
            spawn_rotation=0,
            facing_direction='-y',
        )
        self.env = UrdfEnv(dt=self.dt, robots=[robot], render=self.render)
        ob0 = self.env.reset(pos=np.zeros(10))
        p.setGravity(0, 0, -9.81)
        print("✅ Albert environment created. Total bodies:", p.getNumBodies())
        return ob0

    def get_albert_body_id(self):
        """Find the body ID of Albert in PyBullet."""
        for i in range(p.getNumBodies()):
            name = p.getBodyInfo(i)[1].decode("utf-8").lower()
            if "albert" in name:
                return i
        raise RuntimeError("❌ Could not find Albert in simulation!")

    def load_table(self):
        """Load the table URDF and identify the handle link."""
        table_path = os.path.expanduser("~/catkin_ws/src/mobile_sim/assets/table/table.urdf")
        if not os.path.exists(table_path):
            raise FileNotFoundError(f"❌ Table URDF not found at {table_path}")
        table_orn = p.getQuaternionFromEuler([0, 0, -np.pi / 2])
        self.table_id = p.loadURDF(
            table_path,
            basePosition=[0.0, -1.15, 0.8],
            baseOrientation=table_orn,
            useFixedBase=False,
        )
        for j in range(p.getNumJoints(self.table_id)):
            name = p.getJointInfo(self.table_id, j)[1].decode().lower()
            if "wheel" in name or "caster" in name:
                p.setJointMotorControl2(self.table_id, j, controlMode=p.VELOCITY_CONTROL,
                                        targetVelocity=0, force=0)
        for j in range(p.getNumJoints(self.table_id)):
            link_name = p.getJointInfo(self.table_id, j)[12].decode().lower()
            if "table_robot_end" in link_name:
                self.goal_link_idx = j
        if self.goal_link_idx is None:
            raise RuntimeError("❌ Could not find table_robot_end link")
        for _ in range(100):
            p.stepSimulation()
            time.sleep(0.01)

    def set_arm_initial_pose(self):
        q_current = np.array([p.getJointState(self.albert_id, j)[0] for j in self.arm_joint_indices])
        offset_deg = [0, 90, 0, 25, 0, 40, 38]
        q_target = q_current + np.deg2rad(offset_deg)
        for j, q in zip(self.arm_joint_indices, q_target):
            p.resetJointState(self.albert_id, j, q)

    def disable_arm_motors(self):
        for j in self.arm_joint_indices:
            p.setJointMotorControl2(self.albert_id, j, controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=0.0, force=0.0)

    def impedance_step(self):
        """Run one impedance control step (not used for observation now)."""
        ee_state = p.getLinkState(self.albert_id, self.ee_idx, computeLinkVelocity=1)
        ee_pos, ee_vel = np.array(ee_state[0]), np.array(ee_state[6])
        handle_state = p.getLinkState(self.table_id, self.goal_link_idx, computeLinkVelocity=1)
        handle_pos, handle_vel = np.array(handle_state[0]), np.array(handle_state[6])
        dx, dv = handle_pos - ee_pos, handle_vel - ee_vel
        F = -(self.Kp @ dx + self.Dp @ dv)
        F = np.clip(F, -self.F_max, self.F_max)
        p.applyExternalForce(self.table_id, self.goal_link_idx, F.tolist(),
                             handle_pos.tolist(), flags=p.WORLD_FRAME)
        return F[:2], dx[:2], handle_pos[:2], handle_vel[:2]


# ============================================================
# =================== Albert + Table Env =====================
# ============================================================

class AlbertTableEnv(gym.Env):
    """Gymnasium environment using only robot-centric observations."""

    def __init__(self, render=False, max_steps=1000, goals=None):
        super().__init__()
        self.render_mode = render
        self.max_steps = max_steps
        self.step_count = 0
        self.goals = [np.array(g, dtype=np.float32) for g in (goals or [(1.5, -2.0)])]
        self.goal = self.goals[0]
        print(f"Goal is: {self.goal}")
 
        # Action space: [forward velocity, yaw rate]
        self.action_space = spaces.Box(
            low=np.array([-0.5, -1.0], dtype=np.float32),
            high=np.array([0.5, 1.0], dtype=np.float32)
        )

        # ✅ Simplified 6-D observation
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self.sim = AlbertTableImpedanceSim(render=self.render_mode)
        self.prev_dist = None
        self._last_env_ob = None

    def _get_obs(self):
        """Return 6D robot-centric observation."""
        base_state = self._last_env_ob[0]["robot_0"]["joint_state"]
        base_pos = base_state["position"][:3]
        base_vel = base_state["velocity"]
        x, y, yaw = base_pos
        v, w = base_vel[0], base_vel[2]
        dx, dy = self.goal[0] - x, self.goal[1] - y

        # Heading error (Albert faces −Y)
        angle_to_goal_world = np.arctan2(dy, dx)
        angle_to_goal_adj = angle_to_goal_world - np.pi / 2.0
        heading_error = np.arctan2(np.sin(angle_to_goal_adj - yaw),
                                   np.cos(angle_to_goal_adj - yaw))

        hcos = np.cos(heading_error)
        hsin = np.sin(heading_error)
        heading_cos_bi = np.abs(hcos)
        heading_sin_bi = hsin * np.sign(hcos)

        return np.array([dx, dy, v, w, hcos, hsin], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.goal = self.goals[np.random.randint(len(self.goals))]
        print(f"Goal is: {self.goal}")
        if getattr(self.sim, "env", None) is None:
            ob0 = self.sim.create_environment()
        else:
            ob0 = self.sim.env.reset(pos=np.zeros(10))
        self._last_env_ob = ob0

        self.sim.albert_id = self.sim.get_albert_body_id()
        if self.sim.table_id is not None:
            try:
                p.removeBody(self.sim.table_id)
            except Exception:
                pass
        self.sim.table_id = None
        self.sim.load_table()
        self.sim.set_arm_initial_pose()
        self.sim.disable_arm_motors()
        zero = np.zeros(self.sim.env.n(), dtype=float)
        self._last_env_ob = self.sim.env.step(zero)

        base_state = self._last_env_ob[0]["robot_0"]["joint_state"]
        base_pos = base_state["position"][:3]
        self.prev_dist = float(np.linalg.norm(self.goal - base_pos[:2]))

        return self._get_obs(), {}

    def step(self, action):
        self.sim.impedance_step()
        full = np.zeros(self.sim.env.n(), dtype=float)
        full[0], full[1] = action
        ob = self.sim.env.step(full)
        self._last_env_ob = ob

        base_state = ob[0]["robot_0"]["joint_state"]
        base_pos = base_state["position"][:3]
        yaw = base_pos[2]
        goal_vec = self.goal - base_pos[:2]
        dist = float(np.linalg.norm(goal_vec))

        # Heading error
        goal_dir_world = np.arctan2(goal_vec[1], goal_vec[0])
        goal_dir_adj = goal_dir_world - np.pi / 2.0
        heading_error = np.arctan2(np.sin(goal_dir_adj - yaw),
                                   np.cos(goal_dir_adj - yaw))
        err_abs = np.abs(heading_error)
        heading_error_bi = np.minimum(err_abs, np.pi - err_abs)

        print(f"Heading error: {heading_error}, heading error bidirection: {heading_error_bi}")
        
        # Reward
        progress_reward = (self.prev_dist - dist) * 100.0
        #print("progress_reward", progress_reward)
        distance_penalty = - dist / 10 
        #print("distance_penalty", distance_penalty)
        heading_penalty = -0.5 * abs(heading_error_bi)
        #print("heading_penalty", heading_penalty)
        reward = progress_reward + distance_penalty + heading_penalty
        self.prev_dist = dist

        terminated = dist < 0.3
        if terminated:
            reward += 50.0
        truncated = self.step_count >= self.max_steps
        self.step_count += 1
        if self.render_mode:
            time.sleep(self.sim.dt)

        info = {
            "dist_to_goal": dist,
            "heading_error_raw": float(heading_error),
            "heading_error_bi": float(heading_error_bi),
        }
        return self._get_obs(), reward, terminated, truncated, info

    def close(self):
        self.sim.env.close()
        print("✅ Environment closed.")


# ============================================================
# ================== Training Function =======================
# ============================================================

def train_sac(total_timesteps, goals, model_name, base_log_dir, learning_rate,
              buffer_size, batch_size, tau, gamma, train_freq, gradient_steps, render=False):
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_{model_name}"
    log_dir = os.path.join(base_log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)

    env = Monitor(AlbertTableEnv(render=render, goals=goals),
                  filename=os.path.join(log_dir, "monitor.csv"))

    model = SAC("MlpPolicy", env, learning_rate=learning_rate,
                buffer_size=buffer_size, batch_size=batch_size,
                tau=tau, gamma=gamma, train_freq=train_freq,
                gradient_steps=gradient_steps, verbose=1,
                tensorboard_log=log_dir)

    print("\n🚀 Starting SAC training...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True, tb_log_name="SAC")

    model_path = os.path.join(log_dir, f"{model_name}.zip")
    model.save(model_path)
    env.close()

    print(f"\n✅ Training complete — model saved to {model_path}")
    subprocess.Popen(["tensorboard", "--logdir", base_log_dir, "--port", "6006"])
    print("🌐 Open http://localhost:6006 to view live training metrics.")
    return model


# ============================================================
# ==================== MAIN EXECUTION ========================
# ============================================================

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        model_name = "sac_albert_table_robot_only"
        base_log_dir = "runs_albert_table_robot_only"
        render = False
        # each possible "general the algorithm can go as (2.0, 0.0) did not work properly, and with (0.0, 2.0) it drove a bit drunk
        goal_list = [(-2.0, -2.0), (2.0, 2.0), (-2.0, 2.0), (2.0, -2.0), (2.0, 0.0), (0.0, 2.0), (-2.0, 0.0), (0.0, -2.0)]
        #goal_list = [(2.0, 0.0), (-2.0, 0.0)]

        model = train_sac(
            total_timesteps=100000,
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
