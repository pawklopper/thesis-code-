#!/usr/bin/env python3
"""
train_albert_table_sac_debug.py
-------------------------------
SAC training for Albert + Table with impedance coupling.
Now instrumented with detailed debug printouts to verify
that Albert, the table, and the impedance controller
initialize and function correctly.
"""

import os
import time
import warnings
import subprocess
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

import pybullet as p
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv


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
        print("[DEBUG] Searching for Albert body ID...")
        for i in range(p.getNumBodies()):
            name = p.getBodyInfo(i)[1].decode("utf-8").lower()
            print(f"   Body {i}: {name}")
            if "albert" in name:
                print(f"✅ Found Albert body ID: {i}")
                return i
        raise RuntimeError("❌ Could not find Albert in simulation!")

    def load_table(self):
        print("[DEBUG] Loading table URDF...")
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
        print(f"✅ Table loaded with ID {self.table_id}. Total bodies: {p.getNumBodies()}")

        for j in range(p.getNumJoints(self.table_id)):
            name = p.getJointInfo(self.table_id, j)[1].decode().lower()
            if "wheel" in name or "caster" in name:
                print(f"   Disabling wheel joint: {name}")
                p.setJointMotorControl2(self.table_id, j, controlMode=p.VELOCITY_CONTROL,
                                        targetVelocity=0, force=0)

        self.goal_link_idx = None
        for j in range(p.getNumJoints(self.table_id)):
            link_name = p.getJointInfo(self.table_id, j)[12].decode().lower()
            print(f"   Table joint {j}: {link_name}")
            if "table_robot_end" in link_name:
                self.goal_link_idx = j
        if self.goal_link_idx is None:
            raise RuntimeError("❌ Could not find table_robot_end link")
        print(f"✅ Handle link index: {self.goal_link_idx}")

        print("Waiting for table to drop")
        for _ in range(100):
            p.stepSimulation()
            time.sleep(0.01)

    def set_arm_initial_pose(self):
        print("[DEBUG] Setting initial arm pose...")
        q_current = np.array([p.getJointState(self.albert_id, j)[0] for j in self.arm_joint_indices])
        offset_deg = [0, 90, 0, 25, 0, 40, 38]
        q_target = q_current + np.deg2rad(offset_deg)
        for j, q in zip(self.arm_joint_indices, q_target):
            p.resetJointState(self.albert_id, j, q)
        print("✅ Applied arm offset:", offset_deg)

    def disable_arm_motors(self):
        print("[DEBUG] Disabling arm motors...")
        for j in self.arm_joint_indices:
            p.setJointMotorControl2(self.albert_id, j, controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=0.0, force=0.0)
        print("✅ Arm motors disabled (free torque control enabled).")

    def impedance_step(self):
        ee_state = p.getLinkState(self.albert_id, self.ee_idx, computeLinkVelocity=1)
        ee_pos, ee_vel = np.array(ee_state[0]), np.array(ee_state[6])
        handle_state = p.getLinkState(self.table_id, self.goal_link_idx, computeLinkVelocity=1)
        handle_pos, handle_vel = np.array(handle_state[0]), np.array(handle_state[6])
        dx, dv = handle_pos - ee_pos, handle_vel - ee_vel
        F = -(self.Kp @ dx + self.Dp @ dv)
        F = np.clip(F, -self.F_max, self.F_max)
        p.applyExternalForce(self.table_id, self.goal_link_idx, F.tolist(),
                             handle_pos.tolist(), flags=p.WORLD_FRAME)
        joint_indices, q_list, dq_list = self.get_movable_joint_state_arrays(self.albert_id)
        J_lin, _ = p.calculateJacobian(self.albert_id, self.ee_idx, [0, 0, 0],
                                       q_list, dq_list, [0.0]*len(q_list))
        J_arm = np.array(J_lin)[:, self.arm_joint_indices]
        tau = np.clip(J_arm.T @ (-F), -self.tau_max, self.tau_max)
        for j, t in zip(self.arm_joint_indices, tau):
            p.setJointMotorControl2(self.albert_id, j, controlMode=p.TORQUE_CONTROL, force=float(t))
        return F, dx
    
    
    def get_movable_joint_state_arrays(self, body_id):
        """Return joint indices, positions (q), and velocities (dq) for non-fixed joints."""
        joint_indices, q_list, dq_list = [], [], []
        for ji in range(p.getNumJoints(body_id)):
            j_info = p.getJointInfo(body_id, ji)
            if j_info[2] != p.JOINT_FIXED:
                joint_indices.append(ji)
                s = p.getJointState(body_id, ji)
                q_list.append(s[0])
                dq_list.append(s[1])
        return joint_indices, q_list, dq_list



# ============================================================
# =================== Albert + Table Env =====================
# ============================================================

class AlbertTableEnv(gym.Env):
    def __init__(self, render=False, max_steps=1000, goals=None):
        super().__init__()
        self.render_mode = render
        self.max_steps = max_steps
        self.step_count = 0
        if goals is None:
            goals = [(1.5, -2.0)]
        self.goals = [np.array(g, dtype=np.float32) for g in goals]
        self.goal = self.goals[0]
        self.action_space = spaces.Box(low=np.array([-0.5, -1.0], dtype=np.float32),
                                       high=np.array([0.5, 1.0], dtype=np.float32))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.sim = AlbertTableImpedanceSim(render=self.render_mode)
        self.prev_dist = None
        self._last_env_ob = None

    def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.step_count = 0
            self.goal = self.goals[np.random.randint(len(self.goals))]
            # print("\n================ RESET CALLED ================")

            # Close any previous env cleanly
            if getattr(self.sim, "env", None) is not None:
                try:
                    self.sim.env.close()
                except Exception:
                    pass

            # 1) Create env ONCE and cache its initial observation (exactly like the old code)
            ob0 = self.sim.create_environment()
            self._last_env_ob = ob0

            # 2) Spawn table + configure arm + disable arm motors
            self.sim.albert_id = self.sim.get_albert_body_id()
            self.sim.load_table()
            self.sim.set_arm_initial_pose()
            self.sim.disable_arm_motors()

            # 5) One zero-action step to refresh _last_env_ob (no motion, no reset)
            zero = np.zeros(self.sim.env.n(), dtype=float)
            self._last_env_ob = self.sim.env.step(zero)

            # Build obs like before
            obs = self._get_obs()
            self.prev_dist = np.hypot(obs[0], obs[1])
            return obs, {}


    def step(self, action):
        """
        Execute one RL step: apply impedance control + drive base via action.
        """
        # 1) Apply impedance control forces/torques 
        F, dx = self.sim.impedance_step()

        # 2) Execute robot base motion (UrdfEnv steps PyBullet)
        full = np.zeros(self.sim.env.n(), dtype=float)
        full[0], full[1] = action
        ob = self.sim.env.step(full)

        # 3) Cache observation for future access
        self._last_env_ob = ob

        # 4) Extract robot base state
        base_state = ob[0]["robot_0"]["joint_state"]
        base_pos = base_state["position"][:3]
        base_vel = base_state["velocity"]
        v, w = base_vel[0], base_vel[2]
        x, y, yaw = base_pos

        # 5) Compute goal-relative position and heading error
        dxg, dyg = self.goal[0] - x, self.goal[1] - y
        dist = np.hypot(dxg, dyg)
        angle_to_goal_world = np.arctan2(dyg, dxg)
        angle_to_goal_adj = angle_to_goal_world + np.pi / 2.0  # robot faces -y
        heading_error = np.arctan2(np.sin(angle_to_goal_adj - yaw), np.cos(angle_to_goal_adj - yaw))

        # 6) Build observation vector (identical to SimpleAlbertEnv)
        obs = np.array([dxg, dyg, v, w, np.cos(heading_error), np.sin(heading_error)], dtype=np.float32)

        # 7) Compute reward terms
        progress_reward = (self.prev_dist - dist) * 100.0
        distance_penalty = -dist
        heading_penalty = -abs(heading_error) * 0.5
        reward = progress_reward + distance_penalty + heading_penalty
        self.prev_dist = dist

        # 8) Check termination
        terminated = dist < 0.3
        if terminated:
            reward += 25.0
        truncated = self.step_count >= self.max_steps
        self.step_count += 1

        # # 9) Optional debug print every 50 steps
        # if self.step_count % 50 == 0:
        #     print(
        #         f"[STEP {self.step_count:04d}] "
        #         f"dist={dist:.3f}, rew={reward:.2f}, "
        #         f"F={np.round(F, 2)}, dx={np.round(dx, 3)}"
        #     )

        # 10) Real-time sync if rendering
        if self.render_mode:
            time.sleep(self.sim.dt)

        # 11) Return standard Gymnasium tuple
        info = {"dist_to_goal": dist, "force": F, "dx": dx}
        return obs, reward, terminated, truncated, info


    def _get_obs(self):
        base_state = self._last_env_ob[0]["robot_0"]["joint_state"]
        base_pos = base_state["position"][:3]
        base_vel = base_state["velocity"]
        v, w = base_vel[0], base_vel[2]
        x, y, yaw = base_pos
        dx = self.goal[0] - x
        dy = self.goal[1] - y
        angle_to_goal_world = np.arctan2(dy, dx)
        angle_to_goal_adj = angle_to_goal_world + np.pi / 2.0
        heading_error = np.arctan2(np.sin(angle_to_goal_adj - yaw), np.cos(angle_to_goal_adj - yaw))
        return np.array([dx, dy, v, w, np.cos(heading_error), np.sin(heading_error)], dtype=np.float32)

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

    print("\n🚀 Starting SAC training (debug mode)...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True, tb_log_name="SAC")

    model_path = os.path.join(log_dir, f"{model_name}.zip")
    model.save(model_path)
    env.close()
    # print(f"\n✅ Training complete — model saved to {model_path}")
    # subprocess.Popen([
    #     sys.executable, "-m", "tensorboard", "--logdir", base_log_dir, "--port", "6006"
    # ])
    # print("🌐 Open http://localhost:6006 to view training metrics.")

    # === Launch TensorBoard automatically ===
    print(f"\n✅ Training complete — model saved to {model_path}")
    print("🚀 Launching TensorBoard...")
    subprocess.Popen(["tensorboard", "--logdir", base_log_dir, "--port", "6006"])
    print("🌐 Open http://localhost:6006 to view live training metrics.")

    print(f"\n✅ Training complete — model saved to {model_path}")
    print(f"📁 All logs and checkpoints stored in: {log_dir}")
    return model
  

# ============================================================
# ==================== MAIN EXECUTION ========================
# ============================================================



if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        model_name = "sac_albert_table_impedance"
        base_log_dir = "runs_albert_table_impedance"
        render = False
        goal_list = [(0.0, 2.0)]

        model = train_sac(
            total_timesteps=20000,   # short debug run
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


        #run_albert_table_impedance()