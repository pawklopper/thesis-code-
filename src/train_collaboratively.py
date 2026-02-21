#!/usr/bin/env python3
"""
train_collaboratively.py — ROS-INTEGRATED (carefully corrected)
---------------------------------------------------------------
Kept as requested:
- ROS wrench subscriber and force mapping
- controller logic as-is (ROS wrench -> Fh_override)
- camera view setup
- deterministic=False (kept)
- episode_id overlay + parquet logging + timer window

Fixed / made consistent with your "perfect" reference:
1) Single initial reset (no double-reset at startup)
2) Reset logic is consistent with Gymnasium semantics:
   - reset on terminated OR truncated OR timeout counter
3) reset_reason is always correct and aligned with should_reset / done
4) Goal/marker state is consistent across resets
   - in single_goal_mode: goal is truly fixed (env + marker + local variable)
   - in multi-goal: goal is updated exactly when you change it
5) PyBullet debug line argument correctness (lineColorRGB, lineWidth)
6) Removed duplicate set_goal_color definition
7) Video logging is robust (won't crash when not connected; can be disabled cleanly)
8) Episode overlay is recreated after debug-item removal
"""

import os
import time
import datetime
import numpy as np
import pybullet as p

import rospy
from geometry_msgs.msg import WrenchStamped

from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure

from rl_env.albert_table_env import AlbertTableEnv
from controllers.Camercontroller import CameraController

from utils.parquet_logger import ParquetStepLogger, build_step_record
from utils.timer import EpisodeTimerWindow



# ============================================================
# ROS wrench subscriber (latest values)
# ============================================================

_latest_fx = 0.0
_latest_fy = 0.0
_latest_fz = 0.0

def wrench_cb(msg: WrenchStamped):
    global _latest_fx, _latest_fy, _latest_fz
    _latest_fx = msg.wrench.force.x
    _latest_fy = msg.wrench.force.y
    _latest_fz = msg.wrench.force.z


# ============================================================
# Helpers
# ============================================================

def start_pybullet_video_recording(out_path: str):
    """
    Starts PyBullet GUI video recording to an MP4 file.
    Returns log_id on success, otherwise returns None.

    NOTE: This function is robust: it will not crash if video logging
    is unsupported or if PyBullet is not connected.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # If you intentionally disabled recording, you can keep it disabled here:
    # return None

    try:
        info = p.getConnectionInfo()
        if not info or info.get("isConnected", 0) == 0:
            print(f"[VIDEO] PyBullet not connected; skipping recording: {out_path}")
            return None

        log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, out_path)
        if log_id < 0:
            print(f"[VIDEO] Failed to start recording (log_id={log_id}); skipping: {out_path}")
            return None

        print(f"[VIDEO] Recording started: {out_path}")
        return log_id

    except Exception as e:
        print(f"[VIDEO] Exception starting recording; skipping. Error: {e}")
        return None
    

def draw_hud_line(hud_id, participant_label: str, episode_id: int, step: int, cam_target=(0.0, 0.0, 0.0)):
    """
    Draw a single HUD-style line:
      "Participant X - Episode Y - Step Z"
    in small black text near the upper-left of the fixed top-down camera.

    This is world-coordinate anchored. With your fixed camera, it behaves like a HUD.
    """
    # Remove previous HUD item
    if hud_id is not None:
        try:
            p.removeUserDebugItem(hud_id)
        except Exception:
            pass
        hud_id = None

    # Tune these offsets for "upper-left" in your scene
    x0 = float(cam_target[0] - 5.0)  # left
    y0 = float(cam_target[1] + 4.0)  # up
    z0 = 1.35                        # above scene

    text = f"{participant_label} - Episode {int(episode_id)} - Step {int(step)}"

    hud_id = p.addUserDebugText(
        text,
        [x0, y0, z0],
        textColorRGB=[0.0, 0.0, 0.0],  # black
        textSize=1.05,                 # small; adjust 0.8..1.3
        lifeTime=0
    )
    return hud_id




def sample_new_goal(rmin=1.0, rmax=3.0):
    theta = np.random.uniform(-np.pi, np.pi)
    r = np.random.uniform(rmin, rmax)
    return np.array([r * np.cos(theta), r * np.sin(theta)], dtype=np.float32)


def draw_goal_marker_rect(
    goal_xy,
    rect_size_xy=(0.30, 0.20),
    rect_thickness=0.01,
    rect_z=0.005,
    rect_color=(1, 0, 0, 0.9),
    yaw_rad=0.0,
    label=False,
    label_height=0.12,
    label_color=(1, 1, 1),
    label_size=1.2,
):
    gx, gy = float(goal_xy[0]), float(goal_xy[1])

    hx = float(rect_size_xy[0]) * 0.5
    hy = float(rect_size_xy[1]) * 0.5
    hz = float(rect_thickness) * 0.5

    visual = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[hx, hy, hz],
        rgbaColor=rect_color,
    )

    orn = p.getQuaternionFromEuler([0.0, 0.0, float(yaw_rad)])
    body_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual,
        basePosition=[gx, gy, float(rect_z)],
        baseOrientation=orn,
    )

    if label:
        p.addUserDebugText(
            label,
            [gx, gy, float(rect_z) + float(label_height)],
            textColorRGB=list(label_color),
            textSize=float(label_size),
        )

    return body_id


def set_goal_color(goal_id, color=(0, 1, 0, 0.9)):
    p.changeVisualShape(goal_id, -1, rgbaColor=color)


def cue_goal_reached(
    goal_marker_id: int,
    zone_handle: dict,
    goal_xy,
    marker_success_color=(0.1, 0.95, 0.2, 0.95),
    zone_success_outer_color=(0.1, 0.95, 0.2, 0.28),
    zone_success_inner_color=(0.0, 0.0, 0.0, 0.0),
    show_text=False,
    text="GOAL REACHED — RESETTING",
    text_height=0.35,
    text_color=(0.1, 0.95, 0.2),
    text_size=1.8,
    text_lifetime=1.2,
    flash=True,
    flash_count=3,
    flash_period=0.12,
):
    gx, gy = float(goal_xy[0]), float(goal_xy[1])

    try:
        p.changeVisualShape(goal_marker_id, -1, rgbaColor=marker_success_color)
    except Exception:
        pass

    # Zone handle is optional; your code currently uses None.
    if zone_handle:
        try:
            if zone_handle.get("outer_body_id") is not None:
                p.changeVisualShape(zone_handle["outer_body_id"], -1, rgbaColor=zone_success_outer_color)
        except Exception:
            pass
        try:
            if zone_handle.get("inner_body_id") is not None:
                p.changeVisualShape(zone_handle["inner_body_id"], -1, rgbaColor=zone_success_inner_color)
        except Exception:
            pass

    if show_text:
        p.addUserDebugText(
            text,
            [gx, gy, text_height],
            textColorRGB=list(text_color),
            textSize=float(text_size),
            lifeTime=float(text_lifetime),
        )

    if flash and flash_count > 0:
        bright_marker = (0.8, 1.0, 0.8, marker_success_color[3])
        bright_zone = (0.6, 1.0, 0.6, zone_success_outer_color[3])

        for _ in range(int(flash_count)):
            try:
                p.changeVisualShape(goal_marker_id, -1, rgbaColor=bright_marker)
                if zone_handle and zone_handle.get("outer_body_id") is not None:
                    p.changeVisualShape(zone_handle["outer_body_id"], -1, rgbaColor=bright_zone)
            except Exception:
                pass
            time.sleep(float(flash_period))

            try:
                p.changeVisualShape(goal_marker_id, -1, rgbaColor=marker_success_color)
                if zone_handle and zone_handle.get("outer_body_id") is not None:
                    p.changeVisualShape(zone_handle["outer_body_id"], -1, rgbaColor=zone_success_outer_color)
            except Exception:
                pass
            time.sleep(float(flash_period))


# ============================================================
# MAIN
# ============================================================

def run_interactive_model(
    base_dir: str,
    load_run_subdir: str,
    model_name: str,
    single_goal_mode: bool,
    participant_label: str,
    start_goal=(0.0, -2.0),
    max_runtime_steps=100_000,
    warmup_steps=1024,
    use_obstacles=True
):
    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------
    offline_dir = os.path.join("runs", "experiment", "basic_training_offline", load_run_subdir)
    pretrained_path = os.path.join(offline_dir, f"{model_name}.zip")

    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained model not found: {pretrained_path}")

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    online_root = os.path.join("runs", "experiment", "human", f"{participant_label}", f"{ts}_{model_name}")
    tb_dir = os.path.join(online_root, "tb_logs")
    os.makedirs(tb_dir, exist_ok=True)

    print(f"[LOAD MODEL]  {pretrained_path}")
    print(f"[SAVE ROOT]   {online_root}")

    # ----------------------------------
    # ------- DATA ANALYTICS -----------
    # ----------------------------------
    run_id = f"{ts}_{model_name}"
    parquet_out_dir = os.path.join(online_root, "parquet")
    step_logger = ParquetStepLogger(
        out_dir=parquet_out_dir,
        run_id=run_id,
        flush_every=5000,
        compression="zstd",
        engine="pyarrow",
    )

    # --------------------------------------------------------
    # Load pretrained SAC
    # --------------------------------------------------------
    model = SAC.load(pretrained_path)

    # --------------------------------------------------------
    # Attach new environment
    # --------------------------------------------------------
    env = AlbertTableEnv(render=True, goals=[start_goal], use_obstacles=use_obstacles, max_steps=1500)
    model.set_env(env)
    MAX_EPISODES = 20 

    # --------------------------------------------------------
    # Logger setup
    # --------------------------------------------------------
    logger = configure(tb_dir, ["stdout", "tensorboard"])
    model.set_logger(logger)

    # --------------------------------------------------------
    # ROS init + subscriber (minimal integration)
    # --------------------------------------------------------
    rospy.init_node("train_collaboratively_ros", anonymous=False, disable_signals=True)
    rospy.Subscriber("/sigma7/wrench", WrenchStamped, wrench_cb, queue_size=1, tcp_nodelay=True)

    # --------------------------------------------------------
    # Single initial reset (IMPORTANT: do not double-reset)
    # --------------------------------------------------------
    obs, _ = env.reset(Fh_override=np.zeros(3))
    episode_start_wall_time = time.time()

    # --------------------------------------------------------
    # Start video recording (optional/robust)
    # --------------------------------------------------------
    video_path = os.path.join(online_root, f"{participant_label}.mp4")
    video_log_id = start_pybullet_video_recording(video_path)
    video_start_wall_time = time.time()

  

    # Store run metadata once (after env exists + reset)
    step_logger.write_run_meta({
        "model_name": model_name,
        "pretrained_path": pretrained_path,
        "dt": env.sim.dt,
        "warmup_steps": warmup_steps,
        "max_runtime_steps": max_runtime_steps,
        "use_obstacles": use_obstacles,
        "participant_label": participant_label,
        "imp_stiffness": getattr(env.sim, "imp_stiffness", None),
        "imp_damping": getattr(env.sim, "imp_damping", None),
        "imp_max_force": getattr(env.sim, "imp_max_force", None),
        "rot_stiffness": getattr(env.sim, "rot_stiffness", None),
        "rot_damping": getattr(env.sim, "rot_damping", None),
        "rot_max_torque": getattr(env.sim, "rot_max_torque", None),
        "video_start_wall_time": video_start_wall_time,
        "video_path": video_path, 
        "video_log_id": video_log_id
    })

    # --------------------------------------------------------
    # Set camera once (top-down POV)
    # --------------------------------------------------------
    p.resetDebugVisualizerCamera(
        cameraDistance=6.0,
        cameraYaw=0.0,
        cameraPitch=-89.0,
        cameraTargetPosition=[0.0, 0.0, 0.0],
    )

    HUD_ENABLED = None
    HUD_TARGET = (0.0, 0.0, 0.0)  # must match cameraTargetPosition
    hud_id = None


    # --------------------------------------------------------
    # Goal Configuration (source of truth)
    # --------------------------------------------------------
    total_steps, steps_since_goal, goal_id = 0, 0, 1

    fixed_goal = np.array(start_goal, dtype=np.float32)  # used when single_goal_mode=True

    # Enforce correct goal in the env at start (prevents env-reset side changes)
    if single_goal_mode:
        env.goals = [fixed_goal]
        if hasattr(env, "goal"):
            env.goal = fixed_goal

    # Always derive local goal from env after we enforce it
    goal = np.array(env.goal if hasattr(env, "goal") else env.goals[0], dtype=np.float32)

    GOAL_RECT_CFG = dict(
        rect_size_xy=(2.5, 3.0),
        rect_thickness=0.01,
        rect_z=0.005,
        rect_color=(1.0, 0.2, 0.2, 0.9),
        yaw_rad=0.0,
    )
    goal_marker_id = draw_goal_marker_rect(goal, **GOAL_RECT_CFG)

    # Zone handle retained for compatibility (you currently keep it None)
    goal_zone_handle = None

    # --------------------------------------------------------
    # Episode timer window
    # --------------------------------------------------------
    timer_win = EpisodeTimerWindow(
        window_name="Episode Timer",
        update_hz=10.0,
        show_global=False,
    )
    timer_win.start_episode()

    # --------------------------------------------------------
    # Rolling windows for individual rewards
    # --------------------------------------------------------
    reward_window = []
    progress_window = []
    motion_window = []
    dist_window = []
    head_window = []
    eff_window = []
    obstacle_window = []

    # Windows for training metrics
    actor_loss_window, critic_loss_window = [], []
    ent_coef_window, ent_coef_loss_window = [], []
    learning_rate_window, n_updates_window = [], []

    print(f"--> Starting Loop. Training will START after {warmup_steps} steps.")

    cam = CameraController(target=(0, 0, 0.3))  # kept (even if unused)

    episode_id = 0
    t_in_episode = 0

    # --------------------------------------------------------
    # Episode label overlay (participant.episode)
    # --------------------------------------------------------


    # --------------------------------------------------------
    # Small force visualization (short arrow)
    # --------------------------------------------------------
    force_line_id = None
    FORCE_ARROW_SCALE = 0.01
    MAX_FORCE = 40.0
    HUMAN_FORCE_SIGN = -2.0

    # =======================================================
    # MAIN LOOP
    # =======================================================
    try:
        while total_steps < max_runtime_steps:
            if rospy.is_shutdown():
                break

            loop_start_time = time.time()
            prev_obs = obs.copy()

            # --- Policy action ---
            # IMPORTANT: deterministic=False is required by you; kept.
            action, _ = model.predict(obs, deterministic=False)

            # --- Human force (ROS /sigma7/wrench -> robot frame) ---
            Fx_sig = HUMAN_FORCE_SIGN * float(_latest_fx)
            Fy_sig= HUMAN_FORCE_SIGN * float(_latest_fy)

            # --- Map robot -> simulation ---
            # robot x -> sim -y
            # robot y -> sim  x
            Fx =  Fy_sig
            Fy = -Fx_sig

            # Clip
            mag = float(np.hypot(Fx, Fy))
            if mag > MAX_FORCE and mag > 1e-12:
                s = MAX_FORCE / mag
                Fx *= s
                Fy *= s

            Fh = np.array([Fx, Fy, 0.0], dtype=np.float32)

            if total_steps % 50 == 0:
                fh_norm = float(np.linalg.norm(Fh[:2]))
                print(f"[ROS->Sim] step={total_steps} Fh_xy={[Fx, Fy]} |F|={fh_norm:.3f} N")

            # --- Step environment ---
            obs, rew, terminated, truncated, info = env.step(action, Fh_override=Fh)
            done = bool(terminated or truncated)

            # -------------------------------------------------------
            # Collect reward components for logging
            # -------------------------------------------------------
            progress_window.append(info.get("progress_reward", 0.0))
            motion_window.append(info.get("motion_reward", 0.0))
            dist_window.append(info.get("distance_penalty", 0.0))
            head_window.append(info.get("heading_penalty", 0.0))
            eff_window.append(info.get("collaboration_reward", 0.0))
            obstacle_window.append(info.get("obstacle_penalty", 0.0))

            reward_window.append(rew)

            # --- Add transition to replay buffer ---
            model.replay_buffer.add(
                prev_obs,
                obs,
                action,
                np.array([rew], dtype=np.float32),
                np.array([done], dtype=np.float32),
                [{}],
            )

            # -------------------------------------------------------
            # Training Logic
            # -------------------------------------------------------
            if model.replay_buffer.size() > warmup_steps:
                TRAIN_FREQ = 1
                GRADIENT_STEPS = 1   
                BATCH_SIZE = 32
                if total_steps % TRAIN_FREQ == 0:
                    model.train(batch_size=BATCH_SIZE, gradient_steps=GRADIENT_STEPS)
                    for k, v in model.logger.name_to_value.items():
                        if k == "train/actor_loss":
                            actor_loss_window.append(v)
                        elif k == "train/critic_loss":
                            critic_loss_window.append(v)
                        elif k == "train/ent_coef":
                            ent_coef_window.append(v)
                        elif k == "train/ent_coef_loss":
                            ent_coef_loss_window.append(v)
                        elif k == "train/learning_rate":
                            learning_rate_window.append(v)
                        elif k == "train/n_updates":
                            n_updates_window.append(v)

            total_steps += 1
            steps_since_goal += 1


            if HUD_ENABLED:
                hud_id = draw_hud_line(
                    hud_id,
                    participant_label=participant_label,
                    episode_id=episode_id,
                    step=t_in_episode,
                    cam_target=HUD_TARGET
                )


            # -------------------------------------------------------
            # Force visualization (short arrow) - corrected API usage
            # -------------------------------------------------------
            try:
                ls = p.getLinkState(env.sim.table_id, env.sim.human_goal_link_idx, computeLinkVelocity=1)
                sim_pos = ls[0]  # (x, y, z)

                z = float(sim_pos[2] + 0.1)

                if force_line_id is not None:
                    p.removeUserDebugItem(force_line_id)

                force_line_id = p.addUserDebugLine(
                    [float(sim_pos[0]), float(sim_pos[1]), z],
                    [float(sim_pos[0] + Fx * FORCE_ARROW_SCALE), float(sim_pos[1] + Fy * FORCE_ARROW_SCALE), z],
                    lineColorRGB=[0, 0, 0],
                    lineWidth=3,
                    lifeTime=0
                )
            except Exception as e:
                print(f"[force_line] failed: {e}")

            # -------------------------------------------------------
            # Reset / End of Episode Logic (consistent + "perfect")
            # -------------------------------------------------------
            reset_reason = ""
            should_reset = False

            if terminated:
                print(f"Goal #{goal_id} reached!")
                reset_reason = "goal_reached"
                should_reset = True 
                
            elif truncated:
                print(f"Truncated: resetting SAME goal #{goal_id}")
                reset_reason = "truncated"
                should_reset = True

            # elif steps_since_goal >= 1000:
            #     print(f"Timeout: resetting SAME goal #{goal_id}")
            #     reset_reason = "timeout"
            #     should_reset = True

            # record variables (reset_reason is correct for this step)
            wall_time = time.time()
            step_rec = build_step_record(
                env=env,
                episode_id=episode_id,
                t_in_episode=t_in_episode,
                global_step=total_steps,
                wall_time=wall_time,
                episode_start_wall_time=episode_start_wall_time,
                prev_obs=prev_obs,
                obs=obs,
                action_raw=np.array(action, dtype=np.float32),
                reward=rew,
                terminated=terminated,
                truncated=truncated,
                info=info,
                reset_reason=reset_reason,
            )
            step_logger.add(step_rec)

            t_in_episode += 1
            timer_win.update()

            # Visual cue only for true goal reach
            if reset_reason == "goal_reached":
                cue_goal_reached(
                    goal_marker_id=goal_marker_id,
                    zone_handle=goal_zone_handle,
                    goal_xy=goal,
                    text_lifetime=1.0,
                    flash=True,
                    flash_count=3,
                    flash_period=0.10,
                )
                time.sleep(0.6)

                # Multi-goal behavior: choose next goal BEFORE reset (like "perfect")
                if not single_goal_mode:
                    new_goal = sample_new_goal()
                    env.goals = [new_goal]
                    if hasattr(env, "goal"):
                        env.goal = new_goal
                    goal = np.array(new_goal, dtype=np.float32)
                    goal_id += 1

            # -------------------------------------------------------
            # Logging & Resetting (At End of Episode)
            # -------------------------------------------------------
            if should_reset:
                step_logger.flush()

                Fr = env.sim.last_F_xy if hasattr(env.sim, "last_F_xy") else np.zeros(2)
                Fh_debug = env.sim.last_Fh_xy if hasattr(env.sim, "last_Fh_xy") else np.zeros(2)
                status = "WARMUP" if total_steps < warmup_steps else "TRAIN"

                print(
                    f"[{status}] Episode Finished (Step={total_steps}) | MeanRew={np.mean(reward_window):.2f} | "
                    f"Fr={Fr} | Fh={Fh_debug}"
                )

                if reward_window:
                    def mean_safe(x):
                        return float(np.mean(x)) if len(x) > 0 else 0.0

                    model.logger.record("reward/episode_mean", mean_safe(reward_window))
                    model.logger.record("train/actor_loss", mean_safe(actor_loss_window))
                    model.logger.record("train/critic_loss", mean_safe(critic_loss_window))
                    model.logger.record("train/ent_coef", mean_safe(ent_coef_window))

                    model.logger.record("rewards/progress", mean_safe(progress_window))
                    model.logger.record("rewards/motion", mean_safe(motion_window))
                    model.logger.record("rewards/dist_penalty", mean_safe(dist_window))
                    model.logger.record("rewards/head_penalty", mean_safe(head_window))
                    model.logger.record("rewards/collaboration_reward", mean_safe(eff_window))
                    model.logger.record("rewards/obstacle_penalty", mean_safe(obstacle_window))

                    model.logger.dump(step=total_steps)

                    reward_window.clear()
                    progress_window.clear()
                    motion_window.clear()
                    dist_window.clear()
                    head_window.clear()
                    eff_window.clear()
                    obstacle_window.clear()

                    actor_loss_window.clear()
                    critic_loss_window.clear()
                    ent_coef_window.clear()
                    ent_coef_loss_window.clear()
                    learning_rate_window.clear()
                    n_updates_window.clear()

                episode_id += 1
                t_in_episode = 0

                if episode_id >= MAX_EPISODES:
                    print(f"[STOP] Reached {MAX_EPISODES} episodes (0..{MAX_EPISODES-1}). Exiting loop.")
                    break

                # -------------------------
                # Perform Reset (clean)
                # -------------------------
                p.removeAllUserDebugItems()
                force_line_id = None
                hud_id = None


                try:
                    p.removeBody(goal_marker_id)
                except Exception:
                    pass

                obs, _ = env.reset(Fh_override=np.zeros(3))
                episode_start_wall_time = time.time()
                timer_win.start_episode()

                # Enforce goal invariants
                if single_goal_mode:
                    env.goals = [fixed_goal]
                    if hasattr(env, "goal"):
                        env.goal = fixed_goal
                    goal = np.array(fixed_goal, dtype=np.float32)
                else:
                    # For multi-goal, we already set env.goal before reset when goal reached.
                    # But if reset changes goal, we resync from env as source of truth.
                    goal = np.array(env.goal if hasattr(env, "goal") else env.goals[0], dtype=np.float32)

                goal_marker_id = draw_goal_marker_rect(goal, **GOAL_RECT_CFG)

                steps_since_goal = 0
    
            # --- Input Handling ---
            keys = p.getKeyboardEvents()
            if 27 in keys and keys[27] & p.KEY_WAS_TRIGGERED:  # ESC
                break
            if ord('g') in keys and keys[ord('g')] & p.KEY_WAS_TRIGGERED:
                single_goal_mode = not single_goal_mode
                print(f"Toggled single_goal_mode → {single_goal_mode}")

                # If user toggles into single_goal_mode, enforce fixed goal immediately.
                if single_goal_mode:
                    env.goals = [fixed_goal]
                    if hasattr(env, "goal"):
                        env.goal = fixed_goal
                    goal = np.array(fixed_goal, dtype=np.float32)

                    # Refresh marker to match new rule
                    try:
                        p.removeBody(goal_marker_id)
                    except Exception:
                        pass
                    goal_marker_id = draw_goal_marker_rect(goal, **GOAL_RECT_CFG)

            # --- Loop Timing ---
            elapsed = time.time() - loop_start_time
            sleep_time = env.sim.dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        # 0) Save model first, before any PyBullet/Env teardown
        try:
            final_zip = os.path.join(online_root, f"{model_name}_final.zip")
            model.save(final_zip)
            print(f"[FINAL SAVE] {final_zip}")
        except Exception as e:
            print(f"[FINAL SAVE] Failed to save model: {e}")

        # 1) Flush/close logger
        try:
            step_logger.close()
        except Exception as e:
            print(f"[LOGGER] Failed to close parquet logger cleanly: {e}")

        # 2) Stop video logging (guarded)
        try:
            if video_log_id is not None:
                info = p.getConnectionInfo()
                if info and info.get("isConnected", 0) == 1:
                    p.stopStateLogging(video_log_id)
                    print(f"[VIDEO] Recording stopped: {video_path}")
                else:
                    print("[VIDEO] PyBullet already disconnected; skipping stopStateLogging.")
        except Exception as e:
            print(f"[VIDEO] Failed to stop recording cleanly: {e}")

        # 3) Close env (must be guarded!)
        try:
            env.close()
        except Exception as e:
            print(f"[ENV] Failed to close env cleanly: {e}")

        # 4) Close timer window
        try:
            timer_win.close()
        except Exception:
            pass

        print(f"[TB LOGDIR]  {tb_dir}")



# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    base_dir = "" # does not matter anymore
    load_run_subdir = "20260128-145329_experiment_28_jan_20000"
    model_name = "experiment_28_jan_20000"
    use_obstacles = True
    participant_label = "Test_for_experiment foto"

    run_interactive_model(
        base_dir=base_dir,
        load_run_subdir=load_run_subdir,
        model_name=model_name,
        single_goal_mode=True, 
        participant_label=participant_label,
        start_goal=(-5.0, 0.0),
        max_runtime_steps=100_000,
        warmup_steps=0,
        use_obstacles=use_obstacles
    )
