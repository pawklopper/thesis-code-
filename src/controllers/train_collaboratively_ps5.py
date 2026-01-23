#!/usr/bin/env python3
"""
train_collaboratively_ps5.py — REALISTIC VERSION (Steps A & B)
---------------------------------------------------
1. Loads pretrained SAC.
2. [STEP A] Loads offline Replay Buffer.
3. [STEP B] Warmup before training resumes.
4. Logs custom reward components to TensorBoard.
"""

import os
import time
import datetime
import numpy as np
import pybullet as p

from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure

from rl_env.albert_table_env import AlbertTableEnv
from controllers.ps5_human_control_impedance import PS5ImpedanceController

from controllers.Camercontroller import CameraController

from utils.parquet_logger import ParquetStepLogger
from utils.parquet_logger import build_step_record


from utils.timer import EpisodeTimerWindow


# ============================================================
# Helpers
# ============================================================


def draw_goal_marker(goal_xy, color=(1, 0, 0, 1)):
    goal_pos = [float(goal_xy[0]), float(goal_xy[1]), 0.05]
    visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.07, rgbaColor=color)
    mid = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual, basePosition=goal_pos)
    p.addUserDebugText("Goal", [goal_pos[0], goal_pos[1], 0.15], [1, 1, 1], textSize=1.2)
    return mid

def set_goal_color(goal_id, color=(0, 1, 0, 1)):  
    p.changeVisualShape(goal_id, -1, rgbaColor=color)

def sample_new_goal(rmin=1.0, rmax=3.0):
    theta = np.random.uniform(-np.pi, np.pi)
    r = np.random.uniform(rmin, rmax)
    return np.array([r * np.cos(theta), r * np.sin(theta)], dtype=np.float32)

def draw_goal_marker_rect(
    goal_xy,
    # -------------------------
    # Hyperparameters (tunable)
    # -------------------------
    rect_size_xy=(0.30, 0.20),     # (length_x, length_y) in meters
    rect_thickness=0.01,           # thickness (meters)
    rect_z=0.005,                  # center height above ground plane
    rect_color=(1, 0, 0, 0.9),      # RGBA
    yaw_rad=0.0,                   # rotation about z-axis
    label=False,
    label_height=0.12,
    label_color=(1, 1, 1),
    label_size=1.2,
):
    """
    Draw a rectangular goal marker on the ground plane.
    Returns the body id (so you can recolor/remove it later).
    """
    gx, gy = float(goal_xy[0]), float(goal_xy[1])

    # PyBullet GEOM_BOX expects halfExtents
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
    # Works for any visual shape (sphere, box, cylinder, etc.)
    p.changeVisualShape(goal_id, -1, rgbaColor=color)


def cue_goal_reached(
    goal_marker_id: int,
    zone_handle: dict,
    goal_xy,
    # -------------------------
    # Hyperparameters (tunable)
    # -------------------------
    marker_success_color=(0.1, 0.95, 0.2, 0.95),
    zone_success_outer_color=(0.1, 0.95, 0.2, 0.28),
    zone_success_inner_color=(0.0, 0.0, 0.0, 0.0),
    show_text=True,
    text="GOAL REACHED — RESETTING",
    text_height=0.35,
    text_color=(0.1, 0.95, 0.2),
    text_size=1.8,
    text_lifetime=1.2,         # seconds
    flash=True,
    flash_count=3,
    flash_period=0.12,         # seconds per half-cycle
):
    """
    Visual cue for the human when the goal is reached.
    - Recolors marker + zone
    - Optional banner text
    - Optional flashing for saliency
    """
    gx, gy = float(goal_xy[0]), float(goal_xy[1])

    # 1) Set success colors immediately
    try:
        p.changeVisualShape(goal_marker_id, -1, rgbaColor=marker_success_color)
    except Exception:
        pass

    # Recolor zone bodies if present
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

    # 2) On-screen banner
    if show_text:
        p.addUserDebugText(
            text,
            [gx, gy, text_height],
            textColorRGB=list(text_color),
            textSize=float(text_size),
            lifeTime=float(text_lifetime),
        )

    # 3) Flash effect (toggle between success and a brighter success)
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

def draw_goal_zone(
    goal_xy,
    zone_outer_radius=0.80,
    zone_ring_thickness=0.12,
    zone_height=0.002,
    zone_color=(0.1, 0.8, 1.0, 0.18),
    inner_cutout_color=(0.0, 0.0, 0.0, 0.0),
    add_radial_guides=True,
    guide_count=10,
    guide_length=0.25,
    guide_color=(0.1, 0.9, 1.0),
    guide_line_width=2.0,
):
    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    z = zone_height * 0.5

    outer_r = float(max(0.0, zone_outer_radius))
    inner_r = float(max(0.0, outer_r - float(max(0.0, zone_ring_thickness))))

    created = {"outer_body_id": None, "inner_body_id": None, "guide_debug_ids": []}

    outer_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=outer_r, length=zone_height, rgbaColor=zone_color)
    created["outer_body_id"] = p.createMultiBody(0, outer_vis, basePosition=[gx, gy, z])

    if inner_r > 1e-6:
        inner_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=inner_r, length=zone_height * 1.01, rgbaColor=inner_cutout_color)
        created["inner_body_id"] = p.createMultiBody(0, inner_vis, basePosition=[gx, gy, z + 1e-4])

    if add_radial_guides and guide_count > 0 and outer_r > 1e-6:
        for i in range(int(guide_count)):
            a = 2.0 * np.pi * (i / float(guide_count))
            x1, y1 = gx + outer_r * np.cos(a), gy + outer_r * np.sin(a)
            x2, y2 = gx + (outer_r + guide_length) * np.cos(a), gy + (outer_r + guide_length) * np.sin(a)
            dbg_id = p.addUserDebugLine([x1, y1, z + 1e-3], [x2, y2, z + 1e-3],
                                       lineColorRGB=list(guide_color), lineWidth=float(guide_line_width), lifeTime=0)
            created["guide_debug_ids"].append(dbg_id)

    return created


def remove_goal_zone(zone_handle):
    if not zone_handle:
        return
    try:
        if zone_handle.get("outer_body_id") is not None:
            p.removeBody(zone_handle["outer_body_id"])
    except Exception:
        pass
    try:
        if zone_handle.get("inner_body_id") is not None:
            p.removeBody(zone_handle["inner_body_id"])
    except Exception:
        pass
    for did in zone_handle.get("guide_debug_ids", []):
        try:
            p.removeUserDebugItem(did)
        except Exception:
            pass




# ============================================================
# MAIN
# ============================================================

def run_interactive_model(
    base_dir: str,
    load_run_subdir: str,
    model_name: str,
    single_goal_mode: bool,
    start_goal=(0.0, -2.0),
    max_runtime_steps=100_000,
    warmup_steps=1024, 
    use_obstacles = True
):
    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------
    offline_dir = os.path.join("runs", "offline", "jan",base_dir, load_run_subdir)
    pretrained_path = os.path.join(offline_dir, f"{model_name}.zip")
    
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained model not found: {pretrained_path}")

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    online_root = os.path.join("runs", "online_human", "jan", base_dir, f"{ts}_{model_name}")
    tb_dir = os.path.join(online_root, "tb_logs")
    os.makedirs(tb_dir, exist_ok=True)

    print(f"[LOAD MODEL]  {pretrained_path}")
    print(f"[SAVE ROOT]   {online_root}")

    # ----------------------------------
    # ------- DATA ANALYTICS -----------
    # ----------------------------------

    run_id = f"{ts}_{model_name}"
    parquet_out_dir = os.path.join(online_root, "parquet")  # keep inside the run folder
    step_logger = ParquetStepLogger(
        out_dir=parquet_out_dir,
        run_id=run_id,
        flush_every=5000,          # tune: 2000–20000 typical
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
    # We pass 'options' to ensure reward_weights are set if needed (default is all 1.0)
    env = AlbertTableEnv(render=True, goals=[start_goal], use_obstacles=use_obstacles)
    model.set_env(env)

    # Store run metadata once (high value for research reproducibility)
    step_logger.write_run_meta({
        "model_name": model_name,
        "pretrained_path": pretrained_path,
        "dt": env.sim.dt,
        "warmup_steps": warmup_steps,
        "max_runtime_steps": max_runtime_steps,
        "use_obstacles": use_obstacles,
        "imp_stiffness": getattr(env.sim, "imp_stiffness", None),
        "imp_damping": getattr(env.sim, "imp_damping", None),
        "imp_max_force": getattr(env.sim, "imp_max_force", None),
        "rot_stiffness": getattr(env.sim, "rot_stiffness", None),
        "rot_damping": getattr(env.sim, "rot_damping", None),
        "rot_max_torque": getattr(env.sim, "rot_max_torque", None),
    })


    # --------------------------------------------------------
    # Logger setup
    # --------------------------------------------------------
    logger = configure(tb_dir, ["stdout", "tensorboard"])
    model.set_logger(logger)
    
    # --------------------------------------------------------
    # Initial reset
    # --------------------------------------------------------
    obs, _ = env.reset(Fh_override=np.zeros(3))

    # --------------------------------------------------------
    # PS5 controller
    # --------------------------------------------------------
    human_controller = PS5ImpedanceController(
        table_id=env.sim.table_id, 
        link_idx=env.sim.human_goal_link_idx
    )
    
    start_pos = p.getLinkState(env.sim.table_id, env.sim.human_goal_link_idx)[0]
    human_controller.ghost_pos = np.array(start_pos[:2], dtype=np.float32)


    # --------------------------------------------------------
    # Goal Configuration
    # --------------------------------------------------------

    total_steps, steps_since_goal, goal_id = 0, 0, 1
    goal = np.array(start_goal)
    #goal_marker_id = draw_goal_marker(goal)


    GOAL_RECT_CFG = dict(
    rect_size_xy=(3.0, 2.5),
    rect_thickness=0.01,
    rect_z=0.005,
    rect_color=(1.0, 0.2, 0.2, 0.9),
    yaw_rad=0.0, )



    goal_marker_id = draw_goal_marker_rect(goal, **GOAL_RECT_CFG)


    GOAL_ZONE_CFG = dict(
    zone_outer_radius=0.80,
    zone_ring_thickness=0.12,
    zone_height=0.002,
    zone_color=(0.1, 0.8, 1.0, 0.18),
    inner_cutout_color=(0.0, 0.0, 0.0, 0.0),
    add_radial_guides=True,
    guide_count=10,
    guide_length=0.25,
    guide_color=(0.1, 0.9, 1.0),
    guide_line_width=2.0,
)

    goal_zone_handle = None

    timer_win = EpisodeTimerWindow(
    window_name="Episode Timer",
    update_hz=10.0,     # 5–20 is fine
    show_global=False,  # set True if you also want an overall timer line
    )
    timer_win.start_episode()







    # --------------------------------------------------------
    # !!! ADDED: Rolling windows for individual rewards
    # --------------------------------------------------------
    reward_window = []
    
    # Windows for specific components
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

    cam = CameraController(target=(0, 0, 0.3))

    episode_id = 0
    t_in_episode = 0



    # =======================================================
    # MAIN LOOP
    # =======================================================
    try:
        while total_steps < max_runtime_steps:
            #cam.update()
            loop_start_time = time.time()
            
            prev_obs = obs.copy()

            # --- Policy action ---
            action, _ = model.predict(obs, deterministic=False)

            # --- Human force ---
            Fh = human_controller.step(dt=env.sim.dt)

            # --- Step environment ---
            obs, rew, terminated, truncated, info = env.step(action, Fh_override=Fh)
            done = terminated or truncated

            # record variables
            wall_time = time.time()
            step_rec = build_step_record(
                env=env,
                episode_id=episode_id,
                t_in_episode=t_in_episode,
                global_step=total_steps,
                wall_time=wall_time,
                prev_obs=prev_obs,
                obs=obs,
                action_raw=np.array(action, dtype=np.float32),
                reward=rew,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
            step_logger.add(step_rec)
            t_in_episode += 1
            timer_win.update()



           

            # -------------------------------------------------------
            # !!! ADDED: Collect individual rewards for logging
            # -------------------------------------------------------
            # We use .get() to be safe, defaulting to 0.0 if missing
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
                TRAIN_FREQ = 10 
                if total_steps % TRAIN_FREQ == 0:
                    model.train(batch_size=64, gradient_steps=TRAIN_FREQ)

                    for k, v in model.logger.name_to_value.items():
                        if k == "train/actor_loss": actor_loss_window.append(v)
                        elif k == "train/critic_loss": critic_loss_window.append(v)
                        elif k == "train/ent_coef": ent_coef_window.append(v)
                        elif k == "train/ent_coef_loss": ent_coef_loss_window.append(v)
                        elif k == "train/learning_rate": learning_rate_window.append(v)
                        elif k == "train/n_updates": n_updates_window.append(v)

            total_steps += 1
            steps_since_goal += 1

            # --- Reset / End of Episode Logic ---
            should_reset = False
            if steps_since_goal >= 1000 and not terminated:
                print(f"Timeout: resetting SAME goal #{goal_id}")
                should_reset = True
            elif terminated:
                print(f"Goal #{goal_id} reached!")

                cue_goal_reached(
                    goal_marker_id=goal_marker_id,
                    zone_handle=goal_zone_handle,   # the dict returned by draw_goal_zone(...)
                    goal_xy=goal,
                    # You can tune these per your preference:
                    text_lifetime=1.0,
                    flash=True,
                    flash_count=3,
                    flash_period=0.10,
                )

                # Keep a short pause so the human perceives the cue before reset
                time.sleep(0.6)

                if not single_goal_mode:
                    new_goal = sample_new_goal()
                    env.goals = [new_goal]
                    env.goal = new_goal
                    goal = new_goal
                    goal_id += 1

                should_reset = True


            # -------------------------------------------------------
            # Logging & Resetting (At End of Episode)
            # -------------------------------------------------------
            if should_reset:
                

                step_logger.flush()
                episode_id += 1
                t_in_episode = 0

                # 1. Log stats for the finished episode
                Fr = env.sim.last_F_xy if hasattr(env.sim, "last_F_xy") else np.zeros(2)
                Fh_debug = env.sim.last_Fh_xy if hasattr(env.sim, "last_Fh_xy") else np.zeros(2)
                status = "WARMUP" if total_steps < warmup_steps else "TRAIN"
                
                print(
                    f"[{status}] Episode Finished (Step={total_steps}) | MeanRew={np.mean(reward_window):.2f} | "
                    f"Fr={Fr} | Fh={Fh_debug}"
                )

                if reward_window:
                    def mean_safe(x): return float(np.mean(x)) if len(x) > 0 else 0.0

                    # Standard metrics
                    model.logger.record("reward/episode_mean", mean_safe(reward_window))
                    model.logger.record("train/actor_loss", mean_safe(actor_loss_window))
                    model.logger.record("train/critic_loss", mean_safe(critic_loss_window))
                    model.logger.record("train/ent_coef", mean_safe(ent_coef_window))
                    
                    # Custom Reward Components
                    model.logger.record("rewards/progress", mean_safe(progress_window))
                    model.logger.record("rewards/motion", mean_safe(motion_window))
                    model.logger.record("rewards/dist_penalty", mean_safe(dist_window))
                    model.logger.record("rewards/head_penalty", mean_safe(head_window))
                    model.logger.record("rewards/collaboration_reward", mean_safe(eff_window))
                    model.logger.record("rewards/obstacle_penalty", mean_safe(obstacle_window))


                    model.logger.dump(step=total_steps)

                    # 2. Clear all windows for next episode
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

                # 3. Perform Reset
                # 3. Perform Reset
                p.removeAllUserDebugItems()
                try:
                    p.removeBody(goal_marker_id)
                except:
                    pass

                obs, _ = env.reset(Fh_override=np.zeros(3))

                timer_win.start_episode()


                # IMPORTANT: sync local `goal` with env after reset
                # (covers both single-goal and multi-goal modes)
                if hasattr(env, "goal"):
                    goal = np.array(env.goal, dtype=np.float32)
                else:
                    goal = np.array(env.goals[0], dtype=np.float32)

                # Re-create the rectangle marker EVERY episode
                goal_marker_id = draw_goal_marker_rect(goal, **GOAL_RECT_CFG)

                start_pos = p.getLinkState(env.sim.table_id, env.sim.human_goal_link_idx)[0]
                human_controller.ghost_pos = np.array(start_pos[:2], dtype=np.float32)

                steps_since_goal = 0

                #goal_marker_id = draw_goal_marker(goal)

            # --- Input Handling ---
            keys = p.getKeyboardEvents()
            if 27 in keys and keys[27] & p.KEY_WAS_TRIGGERED: # ESC
                break
            if ord('g') in keys and keys[ord('g')] & p.KEY_WAS_TRIGGERED:
                single_goal_mode = not single_goal_mode
                print(f"Toggled single_goal_mode → {single_goal_mode}")

            # --- Loop Timing ---
            elapsed = time.time() - loop_start_time
            sleep_time = env.sim.dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:

        try:
            step_logger.close()
        except Exception as e:
            print(f"[LOGGER] Failed to close parquet logger cleanly: {e}")

            
        env.close()
        try:
            timer_win.close()
        except Exception:
            pass


        final_zip = os.path.join(online_root, f"{model_name}_final.zip")
        final_buffer = os.path.join(online_root, f"{model_name}_final_buffer.pkl")
        
        model.save(final_zip)
        
        print(f"[FINAL SAVE] {final_zip}")
        print(f"[BUFFER SAVED] {final_buffer}")
        print(f"[TB LOGDIR]  {tb_dir}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    base_dir = "runs_14_jan_test"
    load_run_subdir = "20260114-171544_14_jan_longrun_600000_steps" # Update this to your latest run!
    model_name = "14_jan_longrun_600000_steps"
    use_obstacles = True

    run_interactive_model(
        base_dir=base_dir,
        load_run_subdir=load_run_subdir,
        model_name=model_name,
        single_goal_mode=True,
        start_goal=(0.0, 4.0),
        max_runtime_steps=100_000,
        warmup_steps=1002,
        use_obstacles=use_obstacles
    )