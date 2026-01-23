# utils/parquet_logger.py
from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ParquetStepLogger:
    """
    Chunked Parquet logger:
    - Collect step records in RAM
    - Flush to Parquet every `flush_every` records or at episode end
    - Also writes run metadata once
    """
    out_dir: str
    run_id: str
    flush_every: int = 5000
    compression: str = "zstd"   # "snappy" also good
    engine: str = "pyarrow"

    def __post_init__(self):
        os.makedirs(self.out_dir, exist_ok=True)
        self.records: List[Dict[str, Any]] = []
        self.part_idx: int = 0

        # Store files under run directory
        self.run_dir = os.path.join(self.out_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        self.steps_dir = os.path.join(self.run_dir, "steps")
        os.makedirs(self.steps_dir, exist_ok=True)

        self.meta_path = os.path.join(self.run_dir, "run_meta.json")

    def write_run_meta(self, meta: Dict[str, Any]) -> None:
        """
        Write metadata once (idempotent).
        """
        if os.path.exists(self.meta_path):
            return
        meta = dict(meta)
        meta["run_id"] = self.run_id
        meta["created_wall_time"] = time.time()
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def add(self, rec: Dict[str, Any]) -> None:
        self.records.append(rec)
        if len(self.records) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self.records:
            return

        df = pd.DataFrame(self.records)

        # Ensure stable dtypes for key fields (avoid object columns)
        for col in ["episode", "t", "global_step", "terminated", "truncated", "done"]:
            if col in df.columns:
                # booleans stay bool; ints stay int
                if col in ["terminated", "truncated", "done"]:
                    df[col] = df[col].astype(bool)
                else:
                    df[col] = df[col].astype(np.int64)

        part_path = os.path.join(self.steps_dir, f"part-{self.part_idx:05d}.parquet")
        df.to_parquet(part_path, engine=self.engine, compression=self.compression, index=False)

        self.part_idx += 1
        self.records.clear()

    def close(self) -> None:
        self.flush()



import numpy as np
import pybullet as p


def build_step_record(
    env,
    episode_id: int,
    t_in_episode: int,
    global_step: int,
    wall_time: float,
    episode_start_wall_time: float,
    prev_obs: np.ndarray,
    obs: np.ndarray,
    action_raw: np.ndarray,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict,
    reset_reason: str = "",
) -> dict:

    """
    Produces a flat dict safe for Parquet:
    - scalar columns only
    - obs/action stored as obs_0..obs_9, act_* columns
    """

    done = bool(terminated or truncated)
    episode_elapsed_sec = float(wall_time - float(episode_start_wall_time))


    # --- Table state ---
    table_xy, table_yaw, table_vxy, table_wz = env.sim.get_table_state_world()

        # --- Table angular velocity vector from PyBullet (wx, wy, wz) ---
    try:
        _, table_wxyz = p.getBaseVelocity(env.sim.table_id)
        table_wx = float(table_wxyz[0])
        table_wy = float(table_wxyz[1])
    except Exception:
        table_wx = float("nan")
        table_wy = float("nan")


    # --- Robot base state (URDFenv observation already stored in env) ---
    base_state = env._last_env_ob[0]["robot_0"]["joint_state"]
    robot_x, robot_y = base_state["position"][0], base_state["position"][1]
    robot_yaw = float(env.get_robot_yaw_wf())
    robot_v = float(base_state["velocity"][0])
    robot_w = float(base_state["velocity"][2])

    # --- Forces / impedance diagnostics (from sim) ---
    Fh_xy = getattr(env.sim, "last_Fh_xy", np.zeros(2, dtype=float))
    Fr_xy = getattr(env.sim, "last_F_xy", np.zeros(2, dtype=float))
    dx_xy = getattr(env.sim, "last_dx_xy", np.zeros(2, dtype=float))  # [x_error, y_error]

    # --- Actions ---
    act_raw_v, act_raw_w = float(action_raw[0]), float(action_raw[1])
    act_exec = getattr(env, "last_action_executed", np.array([np.nan, np.nan], dtype=float))
    act_exec_v, act_exec_w = float(act_exec[0]), float(act_exec[1])
    delta_v = act_exec_v - act_raw_v
    delta_w = act_exec_w - act_raw_w

    # --- Reward components (from info; defaults 0.0 if absent) ---
    rec = {
        # Indexing
        "episode": int(episode_id),
        "t": int(t_in_episode),
        "global_step": int(global_step),
        "wall_time": float(wall_time),

        # Goal
        "goal_x": float(env.goal[0]),
        "goal_y": float(env.goal[1]),

        # Table
        "table_x": float(table_xy[0]),
        "table_y": float(table_xy[1]),
        "table_yaw": float(table_yaw),
        "table_vx": float(table_vxy[0]),
        "table_vy": float(table_vxy[1]),
        "table_wz": float(table_wz),
        "table_wx": float(table_wx),
        "table_wy": float(table_wy),


        # Robot base
        "robot_x": float(robot_x),
        "robot_y": float(robot_y),
        "robot_yaw": float(robot_yaw),
        "robot_v": float(robot_v),
        "robot_w": float(robot_w),

        # Observations (current obs is usually what you plot)
        # Optionally store prev_obs too if needed for analysis
        **{f"obs_{i}": float(obs[i]) for i in range(obs.shape[0])},

        # Actions
        "act_raw_v": act_raw_v,
        "act_raw_w": act_raw_w,
        "act_exec_v": act_exec_v,
        "act_exec_w": act_exec_w,
        "delta_v": float(delta_v),
        "delta_w": float(delta_w),

        # Interaction forces
        "Fh_x": float(Fh_xy[0]),
        "Fh_y": float(Fh_xy[1]),
        "Fr_x": float(Fr_xy[0]),
        "Fr_y": float(Fr_xy[1]),
        "spring_x_err": float(dx_xy[0]),
        "spring_y_err": float(dx_xy[1]),

        # Rewards
        "rew_total": float(reward),
        "rew_progress": float(info.get("progress_reward", 0.0)),
        "rew_motion": float(info.get("motion_reward", 0.0)),
        "rew_dist": float(info.get("distance_penalty", 0.0)),
        "rew_heading": float(info.get("heading_penalty", 0.0)),
        "rew_obstacle": float(info.get("obstacle_penalty", 0.0)),

        # Termination
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "done": bool(done),

        # Optional: behavior labels
        "human_action": str(info.get("human_action", "")),
        "dist_table_to_goal": float(info.get("dist_table_to_goal", np.nan)),
        "heading_error": float(info.get("heading_error", np.nan)),
        "heading_error_bi": float(info.get("heading_error_bi", np.nan)),

        "episode_elapsed_sec": float(episode_elapsed_sec),
        "reset_reason": str(reset_reason),

        # --------------------------
        # Experiment hyperparameters / toggles (duplicated per-step by request)
        # --------------------------
        "env_max_steps": int(getattr(env, "max_steps", -1)),
        "use_obstacles": bool(getattr(env, "use_obstacles", False)),
        "use_force_observations": bool(getattr(env, "use_force_observations", False)),
        "robot_radius": float(getattr(env, "robot_radius", np.nan)),

        # Reward params
        "kpr": float(getattr(env, "kpr", np.nan)),
        "kmr": float(getattr(env, "kmr", np.nan)),
        "whp": float(getattr(env, "whp", np.nan)),
        "dist_div": float(getattr(env, "dist_div", np.nan)),
        "goal_threshold": float(getattr(env, "goal_threshold", np.nan)),
        "goal_bonus": float(getattr(env, "goal_bonus", np.nan)),

        # Admittance params
        "adm_gain_lin": float(getattr(env, "adm_gain_lin", np.nan)),
        "adm_gain_ang": float(getattr(env, "adm_gain_ang", np.nan)),
        "adm_deadzone": float(getattr(env, "adm_deadzone", np.nan)),
        "adm_v_clip": float(getattr(env, "adm_v_clip", np.nan)),
        "adm_w_clip": float(getattr(env, "adm_w_clip", np.nan)),

        # Obstacle penalty params (already env attrs)
        "obs_d_enter": float(getattr(env, "obs_d_enter", np.nan)),
        "obs_d_safe": float(getattr(env, "obs_d_safe", np.nan)),
        "obs_query_dist": float(getattr(env, "obs_query_dist", np.nan)),
        "obs_contact_penalty": float(getattr(env, "obs_contact_penalty", np.nan)),
        "obs_impact_penalty": float(getattr(env, "obs_impact_penalty", np.nan)),
        "obs_prox_k": float(getattr(env, "obs_prox_k", np.nan)),
        "obs_approach_k": float(getattr(env, "obs_approach_k", np.nan)),

        # Sim params you already consider run-level, but duplicated here by request
        "sim_dt": float(getattr(env.sim, "dt", np.nan)),
        "imp_stiffness": float(getattr(env.sim, "imp_stiffness", np.nan)) if getattr(env.sim, "imp_stiffness", None) is not None else np.nan,
        "imp_damping": float(getattr(env.sim, "imp_damping", np.nan)) if getattr(env.sim, "imp_damping", None) is not None else np.nan,
        "imp_max_force": float(getattr(env.sim, "imp_max_force", np.nan)) if getattr(env.sim, "imp_max_force", None) is not None else np.nan,
        "rot_stiffness": float(getattr(env.sim, "rot_stiffness", np.nan)) if getattr(env.sim, "rot_stiffness", None) is not None else np.nan,
        "rot_damping": float(getattr(env.sim, "rot_damping", np.nan)) if getattr(env.sim, "rot_damping", None) is not None else np.nan,
        "rot_max_torque": float(getattr(env.sim, "rot_max_torque", np.nan)) if getattr(env.sim, "rot_max_torque", None) is not None else np.nan,


    }

    return rec
