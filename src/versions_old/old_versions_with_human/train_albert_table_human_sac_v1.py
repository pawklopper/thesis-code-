#!/usr/bin/env python3

"""
train_albert_table_human_sac_v1.py
-------------------------------

Simplified SAC training for Albert + Table with impedance coupling.

📘 Simplified Table-centric version:
- Observation focuses on table + robot motion and yaw alignment.
- Reward emphasizes table progress and robot–table heading alignment.
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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import pybullet as p

from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv

# ============================================================
# ================= Utility helpers ==========================
# ============================================================

def wrap_angle(a):
    """Wrap angle to (-pi, pi]."""
    return (a + np.pi) % (2*np.pi) - np.pi


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
        self.human_goal_link_idx = None

        self.last_F_xy = np.zeros(2)
        self.last_dx_xy = np.zeros(2)

    def create_environment(self):
        """Create URDF environment and spawn Albert robot."""
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

        return ob0
    
    def get_movable_joint_state_arrays(SELF, body_id):
        """Return joint indices, positions (q), and velocities (dq) for non-fixed joints."""
        joint_indices, q_list, dq_list = [], [], []
        for ji in range(p.getNumJoints(body_id)):
            j_info = p.getJointInfo(body_id, ji)
            if j_info[2] != p.JOINT_FIXED:  # Non-fixed joints only
                joint_indices.append(ji)
                state = p.getJointState(body_id, ji)
                q_list.append(state[0])
                dq_list.append(state[1])
        return joint_indices, q_list, dq_list
    

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

        table_orn = p.getQuaternionFromEuler([0, 0, -np.pi / 2])  # align with -Y

        self.table_id = p.loadURDF(
            table_path,
            basePosition=[0.0, -1.15, 0.8],
            baseOrientation=table_orn,
            useFixedBase=False,
        )

        # disable wheel joints
        for j in range(p.getNumJoints(self.table_id)):
            name = p.getJointInfo(self.table_id, j)[1].decode().lower()
            if "wheel" in name or "caster" in name:
                p.setJointMotorControl2(self.table_id, j, controlMode=p.VELOCITY_CONTROL,
                                        targetVelocity=0, force=0)

        # find robot handle link
        for j in range(p.getNumJoints(self.table_id)):
            link_name = p.getJointInfo(self.table_id, j)[12].decode().lower()
            if "table_robot_end" in link_name:
                self.goal_link_idx = j

        if self.goal_link_idx is None:
            raise RuntimeError("❌ Could not find table_robot_end link")

        # ✅ find human handle link if exists
        for j in range(p.getNumJoints(self.table_id)):
            link_name = p.getJointInfo(self.table_id, j)[12].decode().lower()
            if "table_human_end" in link_name:
                self.human_goal_link_idx = j
                break

        for _ in range(100):
            p.stepSimulation()
            time.sleep(0.01)

    def set_arm_initial_pose(self):
        """Set the robot arm in an initial configuration close to the handle."""
        q_current = np.array([p.getJointState(self.albert_id, j)[0]
                             for j in self.arm_joint_indices])
        offset_deg = [0, 90, 0, 25, 0, 40, 38]
        q_target = q_current + np.deg2rad(offset_deg)

        for j, q in zip(self.arm_joint_indices, q_target):
            p.resetJointState(self.albert_id, j, q)

    def disable_arm_motors(self):
        """Disable velocity control on the arm joints for passive impedance interaction."""
        for j in self.arm_joint_indices:
            p.setJointMotorControl2(self.albert_id, j,
                                   controlMode=p.VELOCITY_CONTROL,
                                   targetVelocity=0.0,
                                   force=0.0)

    def get_table_state_world(self):
        """Return table CoM (x,y), adjusted yaw (0 = facing -Y), v_xy, and ω_z."""
        pos, orn = p.getBasePositionAndOrientation(self.table_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.table_id)

        xy = np.array(pos[:2], dtype=np.float32)
        yaw_raw = p.getEulerFromQuaternion(orn)[2]
        yaw = wrap_angle(yaw_raw + np.pi / 2)
        vxy = np.array(lin_vel[:2], dtype=np.float32)
        wz = float(ang_vel[2])

        return xy, yaw, vxy, wz


    # ============================================================
    # ======================= Impedance step =====================
    # ============================================================



    # NEW IMPEDANCE STEP DOES NOT INCORPORATE HUMAN FORCES YET SO NEED TO THINKG ABOUT THAT ONE
    def impedance_step(self, goal_xy, robot_xy):
        """Run one impedance control step with translational + yaw impedance."""
        
        # --- EE state ---
        ee_state = p.getLinkState(self.albert_id, self.ee_idx, computeLinkVelocity=1)
        ee_pos, ee_vel, ee_quat = (np.array(ee_state[0]),
                                np.array(ee_state[6]),
                                ee_state[1])

        # --- Handle state ---
        handle_state = p.getLinkState(self.table_id, self.goal_link_idx, computeLinkVelocity=1)
        handle_pos, handle_vel, handle_quat = (np.array(handle_state[0]),
                                            np.array(handle_state[6]),
                                            handle_state[1])

        # =======================================================
        # ✅ Translational impedance (existing behavior)
        # =======================================================
        dx = handle_pos - ee_pos
        dv = handle_vel - ee_vel

        F = -(self.Kp @ dx + self.Dp @ dv)
        F = np.clip(F, -self.F_max, self.F_max)

        # Apply to table
        p.applyExternalForce(self.table_id, self.goal_link_idx,
                            F.tolist(), handle_pos.tolist(),
                            flags=p.WORLD_FRAME)

        # =======================================================
        # ✅ Rotational impedance around vertical (yaw)
        # =======================================================
        h_yaw = p.getEulerFromQuaternion(handle_quat)[2]
        ee_yaw = p.getEulerFromQuaternion(ee_quat)[2]

        yaw_error = ((h_yaw - ee_yaw + np.pi) % (2*np.pi)) - np.pi

        h_ang_vel = p.getLinkState(self.table_id, self.goal_link_idx,
                                computeLinkVelocity=1)[7][2]
        ee_ang_vel = p.getLinkState(self.albert_id, self.ee_idx,
                                    computeLinkVelocity=1)[7][2]
        yaw_vel_error = h_ang_vel - ee_ang_vel

        K_yaw = 50.0 # used to be 80, and D 4, quite strict for what we had
        D_yaw = 4.0
        tau_z = -K_yaw * yaw_error - D_yaw * yaw_vel_error

        p.applyExternalTorque(self.table_id, self.goal_link_idx,
                            [0,0,tau_z],
                            flags=p.WORLD_FRAME)

        # =======================================================
        # ✅ Apply equal and opposite torques to robot arm (Jᵀ wrench)
        # =======================================================
        joint_indices, q_list, dq_list = self.get_movable_joint_state_arrays(self.albert_id)

        J_lin, J_ang = p.calculateJacobian(self.albert_id, self.ee_idx,
                                        [0,0,0], q_list, dq_list,
                                        [0.0]*len(q_list))

        J_arm_lin = np.array(J_lin)[:, self.arm_joint_indices]
        J_arm_ang = np.array(J_ang)[:, self.arm_joint_indices]

        tau_lin = J_arm_lin.T @ (-F)
        tau_rot = J_arm_ang.T @ np.array([0,0,-tau_z])

        tau_total = np.clip(tau_lin + tau_rot, -self.tau_max, self.tau_max)

        for j, t in zip(self.arm_joint_indices, tau_total):
            p.setJointMotorControl2(self.albert_id, j,
                                    controlMode=p.TORQUE_CONTROL,
                                    force=float(t))


        # =======================================================
        # ✅ Outputs stored for logging
        # =======================================================
        self.last_F_xy = F[:2]
        self.last_dx_xy = dx[:2]

        return F[:2], dx[:2], handle_pos[:2], handle_vel[:2]



# ============================================================
# =================== Albert + Table Env =====================
# ============================================================

class AlbertTableEnv(gym.Env):
    """Simplified Gymnasium environment focusing on table motion and heading alignment."""

    def __init__(self, render=False, max_steps=1000, goals=None):
        super().__init__()

        self.render_mode = render
        self.max_steps = max_steps
        self.step_count = 0

        self.goals = [np.array(g, dtype=np.float32) for g in (goals or [(1.5, -2.0)])]
        self.goal = self.goals[0]
        print(f"Goal is: {self.goal}")

        self.action_space = spaces.Box(
            low=np.array([-0.5, -1.0], dtype=np.float32),
            high=np.array([0.5, 1.0], dtype=np.float32)
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )

        self.sim = AlbertTableImpedanceSim(render=self.render_mode)

        self.prev_dist = None
        self._last_env_ob = None



    # ======================================================
    # ================ OBSERVATION FUNCTION ================
    # ======================================================

    def _get_obs(self):
        """
        Observation vector (10D):
        [
            dx_table_to_goal, dy_table_to_goal,
            dx_robot_to_table, dy_robot_to_table,
            v_robot, w_robot,
            table_vx, table_vy,
            cos(heading_error_table_to_goal),
            sin(heading_error_table_to_goal)
        ]
        """

        # --- Table state ---
        table_xy, table_yaw, tv_xy, _ = self.sim.get_table_state_world()
        goal_xy = self.goal[:2]
        dx_t, dy_t = goal_xy - table_xy

        # --- Robot state ---
        base_state = self._last_env_ob[0]["robot_0"]["joint_state"]
        robot_pos = np.array(base_state["position"][:2])
        robot_yaw = base_state["position"][2]
        v_r = base_state["velocity"][0]
        w_r = base_state["velocity"][2]


        # --- NEW: robot–table yaw difference for alignment control
        yaw_diff = wrap_angle(table_yaw - robot_yaw)
        yawcos, yawsin = np.cos(yaw_diff), np.sin(yaw_diff)

        # Heading error from robot facing to table→goal direction
        _, heading_error = self.compute_robot_heading_error(dx_t, dy_t, robot_yaw)
        hcos, hsin = np.cos(heading_error), np.sin(heading_error)

        [Fh_x, Fh_y, Fr_x, Fr_y] = [0.0, 0.0, 0.0, 0.0] # placeholders


        return np.array([
            dx_t, dy_t,
            tv_xy[0], tv_xy[1],
            v_r, w_r,
            hcos, hsin, Fh_x, Fh_y, Fr_x, Fr_y
        ], dtype=np.float32)



    # ======================================================
    # ================== REWARD FUNCTION ===================
    # ======================================================

    def compute_reward(self, dist, prev_dist, heading_error_bi, heading_diff):
        """Simplified reward emphasizing goal progress and heading alignment."""

        # basic functionalities
        # basicly times 
        kpr = 6.0 # used to be 18
        progress_reward = kpr * (prev_dist - dist) / 0.01 


        # motion reward for moving the table towards the goal
        table_xy, _, tv_xy, _ = self.sim.get_table_state_world()
        goal_xy = self.goal[:2]

        dir_to_goal = goal_xy - table_xy
        dir_to_goal /= np.linalg.norm(dir_to_goal) + 1e-6

        speed_toward_goal = np.dot(tv_xy, dir_to_goal)
        kmr = 2.0 # used to be 2.0
        motion_reward = kmr * speed_toward_goal
                
        # distance penalty
        distance_penalty = - dist / 10

        # heading penalty
        # increasing heading penalty leads, to slow training, robot tries to optimize for getting as less possible punshment, 
        # therefore, never reaches goal as it learns that within 1000 steps the episode is earlier, so tries to minimize over these steps each episode
        #  
        whp = 3.0
        heading_penalty = - (whp * heading_error_bi)**2

        total_reward = (
            progress_reward + motion_reward + distance_penalty + heading_penalty  
        )

        # if self.step_count % 100 == 0:
        #     print(f"progress reward: {progress_reward}, motion reward: {motion_reward} distance penalty: {distance_penalty}, heading_penalty: {heading_penalty}, alignment_heading_penalty: {alignment_heading_penalty}")

    

        return total_reward, progress_reward, distance_penalty, heading_penalty

    # ======================================================
    # ===================== STEP LOOP ======================
    # ======================================================

    def step(self, action):
        """Perform one step of the SAC environment."""
        # get states of the robot
        base_state = self._last_env_ob[0]["robot_0"]["joint_state"]
        robot_yaw = base_state["position"][2]
        robot_xy = np.array(base_state["position"][:2])

        self.sim.impedance_step(self.goal, robot_xy)

        full = np.zeros(self.sim.env.n(), dtype=float)
        full[0], full[1] = action
        ob = self.sim.env.step(full)
        self._last_env_ob = ob

        table_xy, table_yaw, _, _ = self.sim.get_table_state_world()

        
      

        goal_xy = self.goal[:2]
        dx_t, dy_t = goal_xy - table_xy
        dist = float(np.linalg.norm([dx_t, dy_t]))

        heading_error_bi, heading_error = self.compute_robot_heading_error(dx_t, dy_t, robot_yaw)
        heading_diff = wrap_angle(table_yaw - robot_yaw)

        reward, progress_reward, distance_penalty, heading_penalty = (
            self.compute_reward(dist, self.prev_dist, heading_error_bi, heading_diff)
        )

        self.prev_dist = dist

        terminated = dist < 0.4
        if terminated:
            print(f"✅ Reached goal at step: {self.step_count}")
            reward += 50

        truncated = self.step_count >= self.max_steps
        self.step_count += 1

        if self.render_mode:
            time.sleep(self.sim.dt)

        info = {
            "dist_table_to_goal": dist,
            "heading_error": float(heading_error),
            "heading_error_bi": float(heading_error_bi),
            "progress_reward": progress_reward,
            "distance_penalty": distance_penalty,
            "heading_penalty": heading_penalty
        }

        return self._get_obs(), reward, terminated, truncated, info

    # ======================================================
    # =================== RESET FUNCTION ===================
    # ======================================================

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

        robot_xy = np.zeros(2)  # safe default, geometry-aware logic fades naturally
        self.sim.impedance_step(self.goal, robot_xy)

        zero = np.zeros(self.sim.env.n(), dtype=float)
        self._last_env_ob = self.sim.env.step(zero)

        table_xy, _, _, _ = self.sim.get_table_state_world()
        self.prev_dist = float(np.linalg.norm(self.goal - table_xy))

        return self._get_obs(), {}
    


    def compute_robot_heading_error(self, dx, dy, robot_yaw):
        angle_to_goal_world = np.arctan2(dy, dx)
        angle_to_goal_adj = angle_to_goal_world - np.pi / 2.0
        heading_error = np.arctan2(
            np.sin(angle_to_goal_adj - robot_yaw),
            np.cos(angle_to_goal_adj - robot_yaw)
        )
        err_abs = abs(heading_error)
        heading_error_bi = min(err_abs, np.pi - err_abs)

        return heading_error_bi, heading_error

    def close(self):
        self.sim.env.close()
        print("✅ Environment closed.")


# ============================================================
# ================== Training Function =======================
# ============================================================

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
    """Launch SAC training with simplified Albert–table environment."""
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_{model_name}"
    log_dir = os.path.join(base_log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)

    env = Monitor(AlbertTableEnv(render=render, goals=goals),
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
        tensorboard_log=log_dir
    )

    print("\n🚀 Starting SAC training...")
    model.learn(total_timesteps=total_timesteps,
                progress_bar=True,
                tb_log_name="SAC")

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

    model_name = "3_nov_test_v1"
    base_log_dir = "runs_3_nov_test_v1"
    render = False

    #goal_list = [(-2.0, -2.0), (2.0, 2.0), (-2.0, 2.0), (2.0, -2.0), (2.0, 0.0), (0.0, 2.0), (-2.0, 0.0), (0.0, -2.0)]
    goal_list = [(0.0, 2.0)]

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


