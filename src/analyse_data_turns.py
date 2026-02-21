#!/usr/bin/env python3
from __future__ import annotations

import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    "base_dir": "Participant_B_Karel",          # experiment root folder under runs/experiment/human
    "subdir": "20260205-132107_experiment_28_jan_20000",  # specific run subfolder
    "model": "experiment_28_jan_20000_final",   # model tag for run naming
    "dt": 0.01,                                 # fallback timestep if sim_dt missing in logs
    "table_handle_offset_x_m": 0.5,             # handle offset from COM along table x-axis (meters)
    "min_turn_steps": 10,                       # minimum contiguous signed-wz samples to count as a turn
    "dominance_ratio": 1.3,                     # required ratio for leader vs follower (sum_h vs sum_r)
    "force_norm_eps": 1e-6,                     # epsilon to avoid divide-by-zero in normalization
    "wz_eps": 0.1,                             # deadband for wz to suppress noisy sign flips (rad/s)
    "tau_sum_min": 1e-3,                        # minimum total aligned torque to avoid spurious labels
}

EPS = 1e-9

# ============================================================
# Logic Utilities
# ============================================================

def _estimate_dt(df: pd.DataFrame, cfg: dict) -> float:
    if "sim_dt" in df.columns:
        vals = df["sim_dt"].to_numpy()
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size > 0:
            return float(np.median(vals))
    return float(cfg["dt"])

def _compute_wz(df: pd.DataFrame, dt: float) -> np.ndarray:
    if "table_wz" in df.columns:
        wz = df["table_wz"].to_numpy(dtype=float)
        if np.any(np.isfinite(wz)):
            return wz
    if "table_yaw" in df.columns:
        yaw = df["table_yaw"].to_numpy(dtype=float)
        return np.gradient(np.unwrap(yaw)) / dt
    raise KeyError("Missing table_wz or table_yaw for turn detection.")

def detect_turn_events_by_alpha(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    dt = _estimate_dt(out, cfg)
    wz = _compute_wz(out, dt)
    alpha = np.gradient(wz) / dt

    is_event = np.zeros(len(out), dtype=bool)
    turn_dir = np.zeros(len(out), dtype=int)
    turn_steps = np.zeros(len(out), dtype=int)
    turn_event_id = np.full(len(out), -1, dtype=int)

    next_id = 0
    for ep in np.sort(out["episode"].unique()):
        idx = np.where(out["episode"] == ep)[0]
        if idx.size < 3:
            continue
        wz_ep = wz[idx]
        alpha_ep = alpha[idx]
        wz_eps = float(cfg["wz_eps"])
        sign_wz = np.zeros_like(wz_ep, dtype=int)
        sign_wz[wz_ep > wz_eps] = 1
        sign_wz[wz_ep < -wz_eps] = -1
        min_steps = int(cfg["min_turn_steps"])

        # Identify contiguous non-zero sign segments
        seg_start = None
        seg_sign = 0
        for k in range(len(idx)):
            s = int(sign_wz[k])
            if s == 0:
                if seg_start is not None:
                    seg_end = k
                    seg_idx = np.arange(seg_start, seg_end)
                    if seg_idx.size >= min_steps:
                        seg_mask = idx[seg_idx]
                        is_event[seg_mask] = True
                        turn_dir[seg_mask] = seg_sign
                        turn_steps[seg_mask] = int(seg_idx.size)
                        turn_event_id[seg_mask] = next_id
                        next_id += 1
                    seg_start = None
                    seg_sign = 0
                continue

            if seg_start is None:
                seg_start = k
                seg_sign = s
            elif s != seg_sign:
                seg_end = k
                seg_idx = np.arange(seg_start, seg_end)
                if seg_idx.size >= min_steps:
                    seg_mask = idx[seg_idx]
                    is_event[seg_mask] = True
                    turn_dir[seg_mask] = seg_sign
                    turn_steps[seg_mask] = int(seg_idx.size)
                    turn_event_id[seg_mask] = next_id
                    next_id += 1
                seg_start = k
                seg_sign = s

        if seg_start is not None:
            seg_idx = np.arange(seg_start, len(idx))
            if seg_idx.size >= min_steps:
                seg_mask = idx[seg_idx]
                is_event[seg_mask] = True
                turn_dir[seg_mask] = seg_sign
                turn_steps[seg_mask] = int(seg_idx.size)
                turn_event_id[seg_mask] = next_id
                next_id += 1

    out["table_wz"], out["table_alpha"] = wz, alpha
    out["is_turn_event"], out["turn_dir"], out["turn_steps"] = is_event, turn_dir, turn_steps
    out["turn_event_id"] = turn_event_id
    return out

def _compute_global_force_scales(df: pd.DataFrame, cfg: dict) -> tuple[float, float]:
    fh_mag = np.linalg.norm(df[["Fh_x", "Fh_y"]].to_numpy(dtype=float), axis=1)
    fr_mag = np.linalg.norm(df[["Fr_x", "Fr_y"]].to_numpy(dtype=float), axis=1)
    fh_mag = fh_mag[np.isfinite(fh_mag)]
    fr_mag = fr_mag[np.isfinite(fr_mag)]
    fh_med = float(np.median(fh_mag)) if fh_mag.size > 0 else 0.0
    fr_med = float(np.median(fr_mag)) if fr_mag.size > 0 else 0.0
    eps = float(cfg["force_norm_eps"])
    return fh_med + eps, fr_med + eps

def attribute_turn_initiator_by_torque(
    df: pd.DataFrame,
    cfg: dict,
    human_scale: float,
    robot_scale: float,
) -> pd.DataFrame:
    out = df.copy()
    yaw = out["table_yaw"].to_numpy(dtype=float)
    fhx, fhy = out["Fh_x"].to_numpy(dtype=float), out["Fh_y"].to_numpy(dtype=float)
    frx, fry = out["Fr_x"].to_numpy(dtype=float), out["Fr_y"].to_numpy(dtype=float)
    dominance_ratio = float(cfg["dominance_ratio"])
    force_norm_eps = float(cfg["force_norm_eps"])
    tau_sum_min = float(cfg["tau_sum_min"])

    offset = float(cfg["table_handle_offset_x_m"])
    c, s = np.cos(yaw), np.sin(yaw)

    # Handle positions relative to table COM in world frame
    r_hx, r_hy = offset * c, offset * s
    r_rx, r_ry = -offset * c, -offset * s

    h_scale = float(human_scale) if human_scale > 0.0 else force_norm_eps
    r_scale = float(robot_scale) if robot_scale > 0.0 else force_norm_eps
    fhx_n = fhx / h_scale
    fhy_n = fhy / h_scale
    frx_n = frx / r_scale
    fry_n = fry / r_scale

    tau_h = r_hx * fhy_n - r_hy * fhx_n
    tau_r = r_rx * fry_n - r_ry * frx_n
    tau_h = np.where(np.isfinite(tau_h), tau_h, 0.0)
    tau_r = np.where(np.isfinite(tau_r), tau_r, 0.0)

    labels = np.full(len(out), "", dtype=object)
    score_h = np.zeros(len(out))
    score_r = np.zeros(len(out))
    ev_mask = out["is_turn_event"].to_numpy()
    ev_ids = out["turn_event_id"].to_numpy()
    for eid in np.unique(ev_ids[ev_mask]):
        if eid < 0:
            continue
        seg_idx = np.where(ev_ids == eid)[0]
        d = float(out.loc[seg_idx[0], "turn_dir"])
        if d == 0.0:
            labels[seg_idx] = "unknown"
            continue
        window_idx = seg_idx
        pos_h = np.maximum(0.0, d * tau_h[window_idx])
        pos_r = np.maximum(0.0, d * tau_r[window_idx])
        pos_h = np.where(np.isfinite(pos_h), pos_h, 0.0)
        pos_r = np.where(np.isfinite(pos_r), pos_r, 0.0)
        sum_h = float(np.sum(pos_h))
        sum_r = float(np.sum(pos_r))
        denom = sum_h + sum_r
        if denom <= EPS or denom < tau_sum_min:
            labels[seg_idx] = "unknown"
            continue
        sh = sum_h / denom
        sr = sum_r / denom
        score_h[seg_idx] = sh
        score_r[seg_idx] = sr
        if sum_h >= dominance_ratio * sum_r:
            labels[seg_idx] = "human"
        elif sum_r >= dominance_ratio * sum_h:
            labels[seg_idx] = "robot"
        else:
            labels[seg_idx] = "both"

    out["turn_initiator"] = labels
    out["score_h"] = score_h
    out["score_r"] = score_r
    out["tau_h"] = tau_h
    out["tau_r"] = tau_r
    return out

def resolve_steps_dir(cfg):
    root_parent = os.path.join("runs", "experiment", "human", cfg["base_dir"])
    run_dir = os.path.join(root_parent, f"{cfg['subdir']}_{cfg['model']}")
    if not os.path.isdir(run_dir): run_dir = os.path.join(root_parent, cfg["subdir"])
    pq = os.path.join(run_dir, "parquet")
    sub = [os.path.join(pq, d) for d in os.listdir(pq) if os.path.isdir(os.path.join(pq, d))]
    return os.path.join(max(sub, key=lambda d: os.path.getmtime(d)), "steps")

def main():
    try:
        steps_dir = resolve_steps_dir(CONFIG)
        df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(steps_dir, "part-*.parquet")))])
        df = df.sort_values(["episode", "t"]).reset_index(drop=True)
    except Exception as e:
        print(f"Error: {e}"); return

    df = detect_turn_events_by_alpha(df, CONFIG)
    human_scale, robot_scale = _compute_global_force_scales(df, CONFIG)
    df = attribute_turn_initiator_by_torque(df, CONFIG, human_scale, robot_scale)

    results = []
    raw_scores = []

    for ep in np.sort(df["episode"].unique()):
        ep_data = df[df["episode"] == ep]
        ev = ep_data[ep_data["is_turn_event"]]
        n_turns = int(ev["turn_event_id"].nunique()) if len(ev) > 0 else 0

        fh_mag = np.linalg.norm(ep_data[["Fh_x", "Fh_y"]].to_numpy(), axis=1)
        fr_mag = np.linalg.norm(ep_data[["Fr_x", "Fr_y"]].to_numpy(), axis=1)

        rec = {
            "episode": int(ep),
            "total_ep_steps": len(ep_data),
            "n_turns": n_turns,
            "avg_turn_steps": 0.0,
            "human": 0.0,
            "robot": 0.0,
            "both": 0.0,
            "unknown": 0.0,
            "success": 0,
            "avg_fh_force": float(np.nanmean(fh_mag)) if len(ep_data) > 0 else 0.0,
            "avg_fr_force": float(np.nanmean(fr_mag)) if len(ep_data) > 0 else 0.0,
        }

        if n_turns > 0:
            turn_lengths = ev.groupby("turn_event_id")["turn_steps"].first().to_numpy()
            rec["avg_turn_steps"] = float(np.mean(turn_lengths)) if turn_lengths.size > 0 else 0.0
            vc = ev["turn_initiator"].value_counts(normalize=True)
            rec["human"] = vc.get("human", 0.0)
            rec["robot"] = vc.get("robot", 0.0)
            rec["both"] = vc.get("both", 0.0)
            rec["unknown"] = vc.get("unknown", 0.0)
            for _, seg in ev.groupby("turn_event_id"):
                raw_scores.append({
                    "episode": int(ep),
                    "h_score": float(seg["score_h"].iloc[0]),
                    "r_score": float(seg["score_r"].iloc[0]),
                })

        if "terminated" in ep_data.columns and "truncated" in ep_data.columns:
            term = ep_data["terminated"].to_numpy(dtype=bool)
            trunc = ep_data["truncated"].to_numpy(dtype=bool)
            rec["success"] = int(np.any(term & ~trunc))

        results.append(rec)
    
    summary_df = pd.DataFrame(results)
    scores_df = pd.DataFrame(raw_scores)
    
    if not summary_df.empty:
        fig, (ax1, ax3, ax5) = plt.subplots(3, 1, figsize=(14, 18))
        x = np.arange(len(summary_df))
        episode_labels = summary_df["episode"].to_numpy()

        # --- SUBPLOT 1: ATTRIBUTION ---
        bottom = np.zeros(len(summary_df))
        categories = ["human", "robot", "both", "unknown"]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for cat, color in zip(categories, colors):
            ax1.bar(x, summary_df[cat].to_numpy(), bottom=bottom, label=cat, color=color, alpha=0.7, width=0.8)
            bottom += summary_df[cat].to_numpy()

        ax1.set_ylabel("Fraction of Events", fontsize=11)
        ax1.set_ylim(0, 1.1)
        ax1.set_xticks(x); ax1.set_xticklabels(episode_labels)
        ax1.legend(loc="upper left", bbox_to_anchor=(1.08, 1), title="Leader")

        success_flags = summary_df["success"].to_numpy()
        for i, ok in enumerate(success_flags):
            ax1.text(i, 1.02, "S" if ok else "F", ha="center", va="bottom",
                     fontsize=9, color="green" if ok else "red")

        ax1_count = ax1.twinx()
        n_turns = summary_df["n_turns"].to_numpy()
        ax1_count.plot(x, n_turns, color="black", marker="o", linestyle="--", zorder=5)
        ax1_count.set_zorder(ax1.get_zorder() + 1)
        ax1_count.patch.set_visible(False)
        ax1_count.set_ylabel("Total Turns", color="black")
        ymax = max(1, int(np.ceil(float(np.max(n_turns)))))
        ax1_count.set_ylim(0, max(2, int(np.ceil(ymax * 1.15))))
        ax1_count.yaxis.set_major_locator(MaxNLocator(integer=True))
        turn_label_offset = 0.04 * ax1_count.get_ylim()[1]
        for i, val in enumerate(n_turns):
            ax1_count.text(i, val + turn_label_offset, f"{int(val)}",
                           ha="center", va="bottom", fontsize=8, color="black")

        # --- SUBPLOT 2: DURATION & STEPS ---
        ax3.bar(x, summary_df["avg_turn_steps"].to_numpy(), color="grey", alpha=0.6, width=0.6, label="Avg Steps/Turn")
        ax3.set_xticks(x); ax3.set_xticklabels(episode_labels)
        ax3.set_ylabel("Avg Steps / Turn", color="black", fontsize=11)
        
        ax4 = ax3.twinx()
        # FIX: Convert to numpy
        ax4.plot(x, summary_df["total_ep_steps"].to_numpy(), color="red", marker="s", linestyle="--", label="Total Ep Steps")
        ax4.set_ylabel("Total Episode Steps", color="black", fontsize=11)
        ax4.set_ylim(0, 1800)
        ax3.legend(loc="upper left", bbox_to_anchor=(1.08, 1))
        ax4.legend(loc="upper left", bbox_to_anchor=(1.08, 0.85))
        step_label_offset = 0.04 * ax4.get_ylim()[1]
        for i, val in enumerate(summary_df["total_ep_steps"].to_numpy()):
            ax4.text(i, val + step_label_offset, f"{int(round(val))}",
                     ha="center", va="bottom", fontsize=8, color="red")

        # --- SUBPLOT 3: CONTRIBUTION SCORES ---
        for i, ep_val in enumerate(episode_labels):
            ep_turns = scores_df[scores_df["episode"] == ep_val]
            ax5.scatter([i]*len(ep_turns), ep_turns["h_score"].to_numpy(), color='#1f77b4', marker='o', alpha=0.3, s=30)
            ax5.scatter([i]*len(ep_turns), ep_turns["r_score"].to_numpy(), color='#ff7f0e', marker='x', alpha=0.4, s=40)

        # FIX: Convert grouped data to numpy
        avg_h = scores_df.groupby("episode")["h_score"].mean().to_numpy()
        avg_r = scores_df.groupby("episode")["r_score"].mean().to_numpy()
        ax5.plot(x, avg_h, color='#1f77b4', marker='o', label="Avg Human Alignment", linewidth=2)
        ax5.plot(x, avg_r, color='#ff7f0e', marker='x', label="Avg Robot Alignment", linewidth=2)
        
        ax5.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, label="Equal Contribution")
        ax5.set_ylabel("Contribution Score (0-1)", fontsize=11)
        ax5.set_ylim(0, 1.05)
        ax5.set_xticks(x); ax5.set_xticklabels(episode_labels)
        ax5.set_xlabel("Episode", fontsize=12)
        ax5.legend(loc="upper left", bbox_to_anchor=(1.08, 1))

        plt.suptitle(f"Turn Dynamics & Intent Attribution", fontsize=15, y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

if __name__ == "__main__":
    main()
