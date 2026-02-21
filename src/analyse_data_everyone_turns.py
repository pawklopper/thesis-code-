#!/usr/bin/env python3


# =============================================================================
# DETAILED NOTE ON THE "DOMINANCE OVER NORMALIZED EPISODE PROGRESS" PLOT
# =============================================================================
#
# PURPOSE
# -------
# We want a time-resolved (phase-resolved) summary of "who leads turning" during
# an episode, but episodes can have different lengths (number of steps). Using
# raw step indices (e.g., step=500) is not comparable across episodes of
# different duration. Therefore we map each episode to a normalized progress
# axis p ∈ [0, 1] and aggregate leadership/dominance over that axis.
#
# This plot is still "turn-event based": the basic unit is a detected turn event
# (a contiguous segment of nonzero yaw-rate sign lasting at least min_turn_steps).
# We do NOT claim to estimate instantaneous continuous leadership at every step;
# we estimate leadership dominance conditioned on turn events that are active
# around each normalized progress bin.
#
#
# DEFINITIONS
# -----------
# Let i index turn events (turn_event_id) and let e index episodes.
#
# 1) Episode length
#    Each episode e has length L_e (in simulation steps):
#
#       L_e = total_ep_steps(e)
#
# 2) Turn event interval in steps
#    Each turn event i has:
#
#       start_step_i   (integer step index where the turn starts)
#       duration_i     (integer number of steps the event lasts)
#       end_step_i     = start_step_i + duration_i
#
#    So the event occupies the half-open interval:
#
#       [ start_step_i , end_step_i )
#
# 3) Normalized episode progress
#    We map step indices to a normalized progress coordinate p:
#
#       p = step / L_e
#
#    For each event i inside episode e(i), we define:
#
#       p_start_i = start_step_i / L_{e(i)}
#       p_end_i   = end_step_i   / L_{e(i)}
#
#    and clamp into [0, 1] for safety:
#
#       p_start_i ← clip(p_start_i, 0, 1)
#       p_end_i   ← clip(p_end_i,   0, 1)
#
# 4) Continuous dominance per turn event
#    During each turn event, we compute per-step aligned torques for human and robot
#    (details are in attribute_turn_initiator_by_torque). We accumulate ONLY the
#    torque aligned with the observed turn direction.
#
#    For event i, define:
#
#       sum_h_i = Σ_t max(0, turn_dir_i * τ_h(t))   over steps t in event i
#       sum_r_i = Σ_t max(0, turn_dir_i * τ_r(t))   over steps t in event i
#
#    Let denom_i = sum_h_i + sum_r_i.
#    If denom_i is too small (denom_i < tau_sum_min), we label the event as
#    "unknown" (low evidence) and do not use it to estimate dominance.
#
#    Otherwise define normalized alignment scores:
#
#       score_h_i = sum_h_i / denom_i
#       score_r_i = sum_r_i / denom_i
#
#    NOTE: score_h_i + score_r_i = 1.
#
#    Then define a continuous dominance index in [-1, 1]:
#
#       d_i = score_h_i - score_r_i
#
#    Equivalent form (since score_r_i = 1 - score_h_i):
#
#       d_i = 2*score_h_i - 1
#
#    Interpretation:
#       d_i = +1  -> fully human-dominant (score_h_i=1)
#       d_i =  0  -> equal contribution  (score_h_i=0.5)
#       d_i = -1  -> fully robot-dominant (score_h_i=0)
#
# 5) Event weight (confidence / strength proxy)
#    We optionally weight events by how "strong" they are. In this script, if we
#    do not store denom_i explicitly, we use duration as a proxy:
#
#       w_i = duration_i
#
#    If denom_i (sum_h_i + sum_r_i) is available, it is a better weight:
#
#       w_i = denom_i = sum_h_i + sum_r_i
#
#    because it reflects the total aligned torque evidence rather than time alone.
#
#
# BINNING AND "ACTIVE EVENT" LOGIC
# -------------------------------
# We discretize normalized progress into K bins:
#
#    p_grid = linspace(0, 1, K+1)
#    bin k covers [ p_grid[k], p_grid[k+1] )
#
# An event i is considered "active" in bin k if its interval overlaps the bin:
#
#    active_i(k)  <=>  (p_start_i < p_grid[k+1]) AND (p_end_i > p_grid[k])
#
# This is the standard overlap test for half-open intervals.
#
#
# WHAT WE PLOT
# ------------
# For each bin k, we compute:
#
# (A) Dominance curve D(k)
#     Consider only "informative" active events (i.e., active and leader != "unknown").
#     Let A_k be the set of informative active events in bin k.
#
#     If A_k is empty, D(k) is undefined (NaN).
#
#     Otherwise compute a weighted mean dominance:
#
#        D(k) =  ( Σ_{i in A_k} w_i * d_i ) / ( Σ_{i in A_k} w_i )
#
#     This yields D(k) ∈ [-1, 1].
#
#     Interpretation:
#       D(k) > 0 : human tends to dominate turns around this progress region
#       D(k) < 0 : robot tends to dominate turns around this progress region
#       D(k) ~ 0 : balanced / shared dominance
#
# (B) Confidence curve C(k)
#     Confidence quantifies how many active events are "informative" vs "unknown":
#
#        n_all(k) = number of active events in bin k (including unknown)
#        n_inf(k) = number of informative active events in bin k (excluding unknown)
#
#        C(k) = n_inf(k) / n_all(k)    (if n_all(k) > 0)
#
#     Interpretation:
#       C(k) = 1   -> all active events have enough evidence (not unknown)
#       C(k) small -> many active events are unknown; treat D(k) cautiously
#
# (C) Uncertainty band via bootstrap (95% CI)
#     If n_inf(k) >= min_n, we estimate uncertainty of D(k) by resampling events
#     in A_k with replacement (bootstrap).
#
#     For b = 1..B:
#       - sample n_inf(k) events from A_k with replacement
#       - compute D_b(k) using the same weighted mean formula
#
#     Then compute percentile interval:
#
#       CI_low(k)  = percentile_{2.5%}( {D_b(k)} )
#       CI_high(k) = percentile_{97.5%}( {D_b(k)} )
#
#     If n_inf(k) < min_n, CI is not shown (NaN) because it is too data-poor.
#
#
# IMPORTANT CAVEATS
# -----------------
# 1) Turn-event based (not continuous control):
#    The curve describes dominance among turning events that overlap each progress
#    region; it does not measure instantaneous force dominance at every time step.
#
# 2) Normalized progress assumes phase comparability:
#    Comparing p=0.8 across episodes assumes that "80% through the episode" is a
#    comparable phase across episodes. This is reasonable if episodes share a common
#    structure, but can be misleading if long episodes include stagnation/failure tails.
#
# 3) Weight choice matters:
#    Using duration as weight emphasizes long events. If available, using total aligned
#    torque (sum_h+sum_r) is a better measure of evidence strength.
#
# 4) Unknown events:
#    Unknown events are excluded from D(k) but included in C(k). Thus C(k) is a direct
#    diagnostic of whether D(k) is based on solid evidence in that region.
#
# =============================================================================

from __future__ import annotations

import glob
import os
import shutil
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    "root_dir": os.path.expanduser("~/catkin_ws/runs/experiment/human"),
    "dt": 0.01,                # fallback timestep if sim_dt missing in logs
    "table_handle_offset_x_m": 0.5,
    "min_turn_steps": 10,
    "dominance_ratio": 1.3,
    "force_norm_eps": 1e-6,
    "wz_eps": 0.1,
    "tau_sum_min": 1e-3,
    "turn_bin_steps": 50,
    "lead_kernel_sigma_steps": 50.0,
    # Spatial contexts used to relate progress to map regions.
    "obstacle_center_xy": (-2.5, 0.0),
    "obstacle_size_xy": (0.5, 1.0),
    "obstacle_zone_margin_m": 1.0,  # expand obstacle footprint by 1 m on each side
    "goal_rect_size_xy": (2.5, 3.0),
    # Typography/style knobs for manuscript-ready figures.
    "ieee_font_size_pt": 10,
    "use_latex_text": False,
    "participants_grid_show_titles": True,
    # Participant-grid legend placement: "above" or "below" the subplot area.
    "participants_grid_legend_position": "above",
    # Participant-grid readability controls.
    "participants_grid_axis_label_size": 14,
    "participants_grid_tick_label_size": 13,
    "participants_grid_legend_font_size": 13,
}

EPS = 1e-9


def _configure_ieee_plot_style(cfg: dict) -> None:
    """Set IEEE-like typography (Times family, 10 pt)."""
    base = float(cfg.get("ieee_font_size_pt", 10))
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"],
            "font.size": base,
            "axes.labelsize": base,
            "axes.titlesize": base,
            "xtick.labelsize": base - 1,
            "ytick.labelsize": base - 1,
            "legend.fontsize": base - 1,
            "figure.titlesize": base,
            "mathtext.fontset": "stix",
            "mathtext.rm": "Times New Roman",
        }
    )

    # Optional exact LaTeX text rendering if a LaTeX installation is available.
    if bool(cfg.get("use_latex_text", False)) and shutil.which("latex") is not None:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "text.latex.preamble": r"\usepackage{newtxtext}\usepackage{newtxmath}",
            }
        )


def _apply_plot_grid(ax: plt.Axes) -> None:
    """Apply consistent major-grid styling across plots."""
    ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.35)


def _in_axis_aligned_rect(
    x: float,
    y: float,
    center_xy: tuple[float, float],
    size_xy: tuple[float, float],
) -> bool:
    cx, cy = float(center_xy[0]), float(center_xy[1])
    sx, sy = float(size_xy[0]), float(size_xy[1])
    if not (np.isfinite(x) and np.isfinite(y) and sx > 0 and sy > 0):
        return False
    hx, hy = 0.5 * sx, 0.5 * sy
    return (cx - hx <= x <= cx + hx) and (cy - hy <= y <= cy + hy)


def _classify_spatial_zone(
    table_x: float,
    table_y: float,
    goal_x: float | None,
    goal_y: float | None,
    cfg: dict,
) -> str:
    """Classify table position into Open / Obstacle / Goal."""
    # Obstacle zone is intentionally expanded to include a 1 m neighborhood around the wall.
    margin = float(cfg.get("obstacle_zone_margin_m", 0.0))
    obs_size = tuple(cfg["obstacle_size_xy"])
    obs_size_expanded = (obs_size[0] + 2.0 * margin, obs_size[1] + 2.0 * margin)
    if _in_axis_aligned_rect(
        table_x,
        table_y,
        tuple(cfg["obstacle_center_xy"]),
        obs_size_expanded,
    ):
        return "Obstacle"

    if goal_x is not None and goal_y is not None and np.isfinite(goal_x) and np.isfinite(goal_y):
        if _in_axis_aligned_rect(
            table_x,
            table_y,
            (float(goal_x), float(goal_y)),
            tuple(cfg["goal_rect_size_xy"]),
        ):
            return "Goal"

    return "Open"


# ============================================================
# Logic Utilities (mirrors analyse_data_turns.py)
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
        wz_eps = float(cfg["wz_eps"])
        sign_wz = np.zeros_like(wz_ep, dtype=int)
        sign_wz[wz_ep > wz_eps] = 1
        sign_wz[wz_ep < -wz_eps] = -1
        min_steps = int(cfg["min_turn_steps"])

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
    out["is_turn_event"] = is_event
    out["turn_dir"] = turn_dir
    out["turn_steps"] = turn_steps
    out["turn_event_id"] = turn_event_id
    return out


def _compute_global_force_scales(df: pd.DataFrame, cfg: dict) -> tuple[float, float]:
    fh_mag = np.linalg.norm(df[["Fh_x", "Fh_y"]].to_numpy(dtype=float), axis=1)
    fr_mag = np.linalg.norm(df[["Fr_x", "Fr_y"]].to_numpy(dtype=float), axis=1)
    fh_mag = fh_mag[np.isfinite(fh_mag)]
    fr_mag = fr_mag[np.isfinite(fr_mag)]
    if fh_mag.size == 0:
        fh_mag = np.array([1.0])
    if fr_mag.size == 0:
        fr_mag = np.array([1.0])
    return float(np.percentile(fh_mag, 95)), float(np.percentile(fr_mag, 95))


def attribute_turn_initiator_by_torque(
    df: pd.DataFrame, cfg: dict, human_scale: float, robot_scale: float
) -> pd.DataFrame:
    out = df.copy()
    d = float(cfg["table_handle_offset_x_m"])
    dominance_ratio = float(cfg["dominance_ratio"])
    tau_sum_min = float(cfg["tau_sum_min"])

    fh = out[["Fh_x", "Fh_y"]].to_numpy(dtype=float)
    fr = out[["Fr_x", "Fr_y"]].to_numpy(dtype=float)

    if human_scale <= 0:
        human_scale = 1.0
    if robot_scale <= 0:
        robot_scale = 1.0

    fh_n = fh / (human_scale + float(cfg["force_norm_eps"]))
    fr_n = fr / (robot_scale + float(cfg["force_norm_eps"]))

    tau_h = d * fh_n[:, 1]
    tau_r = d * fr_n[:, 1]

    labels = np.array(["none"] * len(out), dtype=object)
    score_h = np.zeros(len(out), dtype=float)
    score_r = np.zeros(len(out), dtype=float)

    for _, seg in out[out["is_turn_event"]].groupby("turn_event_id"):
        seg_idx = seg.index.to_numpy()
        if seg_idx.size == 0:
            continue
        turn_dir = int(seg["turn_dir"].iloc[0])
        if turn_dir == 0:
            labels[seg_idx] = "unknown"
            continue
        window_idx = seg_idx
        pos_h = np.maximum(0.0, turn_dir * tau_h[window_idx])
        pos_r = np.maximum(0.0, turn_dir * tau_r[window_idx])
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


@dataclass
class RunInfo:
    participant: str
    run_id: str
    steps_dir: str


def _discover_steps_dirs(root_dir: str) -> list[RunInfo]:
    runs: list[RunInfo] = []
    for participant in sorted(os.listdir(root_dir)):
        participant_dir = os.path.join(root_dir, participant)
        if not os.path.isdir(participant_dir):
            continue
        # Normalize participant name: keep up to the second underscore (e.g., Participant_A_Dimi -> Participant_A)
        parts = participant.split("_")
        if len(parts) >= 2:
            participant_label = "_".join(parts[:2])
        else:
            participant_label = participant
        pattern = os.path.join(participant_dir, "*", "parquet", "*", "steps")
        for steps_dir in sorted(glob.glob(pattern)):
            run_id = os.path.basename(os.path.dirname(os.path.dirname(steps_dir)))
            runs.append(RunInfo(participant=participant_label, run_id=run_id, steps_dir=steps_dir))
    return runs


def _load_steps_df(steps_dir: str) -> pd.DataFrame:
    parts = sorted(glob.glob(os.path.join(steps_dir, "part-*.parquet")))
    if not parts:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    return df


def _episode_records_for_run(
    df: pd.DataFrame, run: RunInfo, cfg: dict
) -> tuple[list[dict], list[dict]]:
    if df.empty:
        return [], []
    df = df.sort_values(["episode", "t"]).reset_index(drop=True)
    df = detect_turn_events_by_alpha(df, cfg)
    human_scale, robot_scale = _compute_global_force_scales(df, cfg)
    df = attribute_turn_initiator_by_torque(df, cfg, human_scale, robot_scale)

    records: list[dict] = []
    dominance_records: list[dict] = []
    for ep in np.sort(df["episode"].unique()):
        ep_data = df[df["episode"] == ep]
        ev = ep_data[ep_data["is_turn_event"]]
        n_turns = int(ev["turn_event_id"].nunique()) if len(ev) > 0 else 0
        counts = ev["turn_initiator"].value_counts()
        records.append(
            {
                "participant": run.participant,
                "run_id": run.run_id,
                "episode": int(ep),
                "total_ep_steps": int(len(ep_data)),
                "n_turns": n_turns,
                "human": int(counts.get("human", 0)),
                "robot": int(counts.get("robot", 0)),
                "both": int(counts.get("both", 0)),
                "unknown": int(counts.get("unknown", 0)),
            }
        )
        for _, seg in ev.groupby("turn_event_id"):
            dominance_records.append(
                {
                    "participant": run.participant,
                    "run_id": run.run_id,
                    "episode": int(ep),
                    "dominance": float(seg["score_h"].iloc[0] - seg["score_r"].iloc[0]),
                }
            )
    return records, dominance_records


def _turn_event_timeline_records_for_run(df: pd.DataFrame, run: RunInfo, cfg: dict) -> list[dict]:
    if df.empty:
        return []
    df = df.sort_values(["episode", "t"]).reset_index(drop=True)
    df = detect_turn_events_by_alpha(df, cfg)
    human_scale, robot_scale = _compute_global_force_scales(df, cfg)
    df = attribute_turn_initiator_by_torque(df, cfg, human_scale, robot_scale)

    records: list[dict] = []
    ev_df = df[df["is_turn_event"]]
    for ep in np.sort(ev_df["episode"].unique()):
        ep_ev = ev_df[ev_df["episode"] == ep]
        groups = []
        for turn_id, seg in ep_ev.groupby("turn_event_id"):
            if seg.empty:
                continue
            start_step = int(seg["t"].iloc[0]) if "t" in seg.columns else int(seg.index.min())
            leader = str(seg["turn_initiator"].iloc[0])
            duration = int(seg["turn_steps"].iloc[0]) if "turn_steps" in seg.columns else int(len(seg))
            dominance_event = float(seg["score_h"].iloc[0] - seg["score_r"].iloc[0])
            x0 = float(seg["table_x"].iloc[0]) if "table_x" in seg.columns else float("nan")
            y0 = float(seg["table_y"].iloc[0]) if "table_y" in seg.columns else float("nan")
            gx0 = float(seg["goal_x"].iloc[0]) if "goal_x" in seg.columns else None
            gy0 = float(seg["goal_y"].iloc[0]) if "goal_y" in seg.columns else None
            zone_context = _classify_spatial_zone(x0, y0, gx0, gy0, cfg)
            groups.append((start_step, turn_id, leader, duration, dominance_event, zone_context))
        groups.sort(key=lambda x: x[0])
        for turn_order, (start_step, turn_id, leader, duration, dominance_event, zone_context) in enumerate(groups, start=1):
            records.append(
                {
                    "participant": run.participant,
                    "run_id": run.run_id,
                    "episode": int(ep),
                    "turn_id": int(turn_id),
                    "turn_order": int(turn_order),
                    "start_step": start_step,
                    "duration": duration,
                    "leader": leader,
                    "dominance_event": dominance_event,
                    # If sum_h/sum_r are unavailable, use duration as a proxy weight.
                    "weight_event": float(duration),
                    "zone_context": zone_context,
                }
            )
    return records


def _step_zone_progress_records_for_run(df: pd.DataFrame, run: RunInfo, cfg: dict) -> list[dict]:
    """Create per-step normalized progress records with spatial zone labels."""
    if df.empty:
        return []
    required = {"episode", "table_x", "table_y"}
    if not required.issubset(df.columns):
        return []

    records: list[dict] = []
    sort_cols = [c for c in ("episode", "t") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    for ep in np.sort(df["episode"].dropna().unique()):
        g = df[df["episode"] == ep]
        n = int(len(g))
        if n <= 0:
            continue
        x = pd.to_numeric(g["table_x"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(g["table_y"], errors="coerce").to_numpy(dtype=float)
        gx = (
            pd.to_numeric(g["goal_x"], errors="coerce").to_numpy(dtype=float)
            if "goal_x" in g.columns
            else np.full(n, np.nan, dtype=float)
        )
        gy = (
            pd.to_numeric(g["goal_y"], errors="coerce").to_numpy(dtype=float)
            if "goal_y" in g.columns
            else np.full(n, np.nan, dtype=float)
        )
        # Use centered normalized progress to reduce edge artifacts.
        p = (np.arange(n, dtype=float) + 0.5) / float(n)
        for i in range(n):
            if not (np.isfinite(x[i]) and np.isfinite(y[i]) and np.isfinite(p[i])):
                continue
            zone_context = _classify_spatial_zone(
                table_x=float(x[i]),
                table_y=float(y[i]),
                goal_x=float(gx[i]) if np.isfinite(gx[i]) else None,
                goal_y=float(gy[i]) if np.isfinite(gy[i]) else None,
                cfg=cfg,
            )
            records.append(
                {
                    "participant": run.participant,
                    "run_id": run.run_id,
                    "episode": int(ep),
                    "p": float(np.clip(p[i], 0.0, 1.0)),
                    "zone_context": zone_context,
                }
            )
    return records


def _compute_zone_occupancy_curves(
    zone_steps_df: pd.DataFrame,
    scope: str = "global",
    participant: str | None = None,
    run_id: str | None = None,
    n_bins: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-progress occupancy fractions for Open / Obstacle / Goal."""
    if zone_steps_df.empty:
        return np.array([]), np.array([]), np.array([]), np.array([])

    z = zone_steps_df.copy()
    if scope == "participant":
        if participant is None:
            raise ValueError("participant must be provided when scope='participant'.")
        z = z[z["participant"] == participant]
    elif scope == "run":
        if participant is None or run_id is None:
            raise ValueError("participant and run_id must be provided when scope='run'.")
        z = z[(z["participant"] == participant) & (z["run_id"] == run_id)]
    elif scope != "global":
        raise ValueError(f"Unsupported scope: {scope}")
    if z.empty:
        return np.array([]), np.array([]), np.array([]), np.array([])

    p = pd.to_numeric(z["p"], errors="coerce").to_numpy(dtype=float)
    zone = z["zone_context"].astype(str).to_numpy()
    valid = np.isfinite(p)
    p = np.clip(p[valid], 0.0, 1.0)
    zone = zone[valid]
    if p.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    p_grid = np.linspace(0.0, 1.0, n_bins + 1)
    p_centers = 0.5 * (p_grid[:-1] + p_grid[1:])
    open_frac = np.full(n_bins, np.nan, dtype=float)
    obs_frac = np.full(n_bins, np.nan, dtype=float)
    goal_frac = np.full(n_bins, np.nan, dtype=float)

    for k in range(n_bins):
        lo = p_grid[k]
        hi = p_grid[k + 1]
        mask = (p >= lo) & (p < hi) if k < (n_bins - 1) else (p >= lo) & (p <= hi)
        n = int(np.sum(mask))
        if n == 0:
            continue
        zbin = zone[mask]
        open_frac[k] = float(np.mean(zbin == "Open"))
        obs_frac[k] = float(np.mean(zbin == "Obstacle"))
        goal_frac[k] = float(np.mean(zbin == "Goal"))

    return p_centers, open_frac, obs_frac, goal_frac


def _add_zone_background(
    ax: plt.Axes,
    p_centers: np.ndarray,
    open_frac: np.ndarray,
    obs_frac: np.ndarray,
    goal_frac: np.ndarray,
) -> None:
    """Overlay occupancy-weighted zone background as soft vertical bands."""
    if p_centers.size < 2:
        return
    dp = float(np.nanmedian(np.diff(p_centers)))
    if not np.isfinite(dp) or dp <= 0:
        return
    zone_layers = [
        ("Open", open_frac, "#1f77b4"),
        ("Obstacle", obs_frac, "#d97706"),
        ("Goal", goal_frac, "#2a9d8f"),
    ]
    for i, pc in enumerate(p_centers):
        x0 = float(max(0.0, pc - 0.5 * dp))
        x1 = float(min(1.0, pc + 0.5 * dp))
        for _, frac_arr, color in zone_layers:
            frac = float(frac_arr[i]) if i < frac_arr.size and np.isfinite(frac_arr[i]) else 0.0
            if frac <= 0:
                continue
            # Alpha encodes occupancy; keep subtle so line data remains dominant.
            alpha = 0.22 * float(np.clip(frac, 0.0, 1.0))
            ax.axvspan(x0, x1, color=color, alpha=alpha, lw=0, zorder=0)


def _plot_leader_share_by_participant(agg_df: pd.DataFrame) -> None:
    participants = agg_df["participant"].to_numpy()
    human = agg_df["human_frac"].to_numpy()
    robot = agg_df["robot_frac"].to_numpy()
    both = agg_df["both_frac"].to_numpy()
    unknown = agg_df["unknown_frac"].to_numpy()

    x = np.arange(len(participants))
    width = 0.8

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    bottom = np.zeros(len(participants))
    for label, values, color in [
        ("Human", human, "#1f77b4"),
        ("Robot", robot, "#ff7f0e"),
        ("Both", both, "#2ca02c"),
        ("Unknown", unknown, "#d62728"),
    ]:
        ax.bar(x, values, bottom=bottom, width=width, label=label, color=color, alpha=0.8)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(participants, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of Turn Events")
    ax.set_title("Leader Share by Participant (All Trials)")
    _apply_plot_grid(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()


def _plot_dominance_index(dominance_df: pd.DataFrame) -> None:
    participants = []
    data = []
    for p in sorted(dominance_df["participant"].unique()):
        vals = dominance_df[dominance_df["participant"] == p]["dominance"].to_numpy()
        if vals.size == 0:
            continue
        participants.append(p)
        data.append(vals)

    if not data:
        print("Warning: dominance plot has no data after filtering.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    data_arr = np.array(data, dtype=object)
    parts = ax.violinplot(data_arr, showmeans=True, showextrema=True, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#8ecae6")
        pc.set_edgecolor("#023047")
        pc.set_alpha(0.7)
    parts["cmeans"].set_color("#219ebc")
    parts["cmedians"].set_color("#ffb703")
    parts["cmins"].set_color("#d62828")
    parts["cmaxes"].set_color("#d62828")
    parts["cbars"].set_color("#023047")
    parts["cmins"].set_linewidth(2.0)
    parts["cmaxes"].set_linewidth(2.0)

    legend_items = [
        Line2D([0], [0], color="#219ebc", lw=2, label="Mean"),
        Line2D([0], [0], color="#ffb703", lw=2, label="Median"),
        Line2D([0], [0], color="#d62828", lw=2, label="Min/Max"),
        Line2D([0], [0], color="#023047", lw=2, label="Range Bar"),
    ]
    ax.legend(handles=legend_items, loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.set_xticks(np.arange(1, len(participants) + 1))
    ax.set_xticklabels(participants, rotation=25, ha="right")
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(0, color="#555555", linewidth=1, alpha=0.6)
    ax.set_ylabel("Dominance Index (Human - Robot) / Total")
    ax.set_title("Leader Dominance by Participant (Per Episode)")
    _apply_plot_grid(ax)
    plt.tight_layout()


def _plot_turn_order_vs_steps_by_participant(turn_df: pd.DataFrame) -> None:
    participants = sorted(turn_df["participant"].unique())
    if not participants:
        print("Warning: No turn timeline data to plot.")
        return

    n = len(participants)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), sharex=False, sharey=False)
    axes = np.atleast_1d(axes).ravel()

    leader_colors = {
        "human": "#1f77b4",
        "robot": "#ff7f0e",
        "both": "#2ca02c",
        "unknown": "#7f7f7f",
        "none": "#7f7f7f",
    }

    for ax, p in zip(axes, participants):
        sub = turn_df[turn_df["participant"] == p]
        for leader, color in leader_colors.items():
            sel = sub[sub["leader"] == leader]
            if sel.empty:
                continue
            ax.scatter(
                sel["start_step"].to_numpy(),
                sel["turn_order"].to_numpy(),
                s=16,
                alpha=0.7,
                color=color,
                label=leader,
            )
        ax.set_title(p)
        ax.set_xlabel("Simulation Step (Turn Start)")
        ax.set_ylabel("Turn Index (Per Episode)")
        _apply_plot_grid(ax)

    for ax in axes[len(participants):]:
        ax.set_visible(False)

    legend_items = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=leader_colors["human"], markersize=6, label="Human"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=leader_colors["robot"], markersize=6, label="Robot"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=leader_colors["both"], markersize=6, label="Both"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=leader_colors["unknown"], markersize=6, label="Unknown"),
    ]
    fig.legend(handles=legend_items, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("Turn Order vs Simulation Step (Per Participant)", y=1.02)
    plt.tight_layout()


def _plot_turn_bars_by_step_and_order(
    turn_df: pd.DataFrame, participant: str, max_episode_steps: int | None = None
) -> None:
    sub = turn_df[turn_df["participant"] == participant]
    if sub.empty:
        print(f"Warning: No turn timeline data for {participant}.")
        return

    leader_colors = {
        "human": "#1f77b4",
        "robot": "#ff7f0e",
        "both": "#2ca02c",
        "unknown": "#7f7f7f",
        "none": "#7f7f7f",
    }

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for _, row in sub.iterrows():
        y = float(row["episode"])
        start = float(row["start_step"])
        duration = float(row["duration"])
        color = leader_colors.get(str(row["leader"]), "#7f7f7f")
        if not (np.isfinite(start) and np.isfinite(duration) and np.isfinite(y)):
            continue
        if duration <= 0:
            continue
        ax.broken_barh([(start, duration)], (y - 0.4, 0.8), facecolors=color, alpha=0.8)

    # Smoothed distribution: probability of human/robot leading vs simulation step
    x = sub["start_step"].to_numpy(dtype=float)
    leader = sub["leader"].astype(str).to_numpy()
    duration = sub["duration"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(duration) & (duration > 0)
    x = x[valid]
    leader = leader[valid]
    duration = duration[valid]
    if x.size > 1:
        min_step = int(np.floor(np.min(x)))
        max_turn_step = int(np.ceil(np.max(x + duration)))
        if max_episode_steps is not None and np.isfinite(max_episode_steps):
            max_step = int(max(max_episode_steps, max_turn_step))
        else:
            max_step = max_turn_step
        x_grid = np.arange(min_step, max_step + 1, dtype=float)
        starts = x
        ends = x + duration
        grid = x_grid[:, None]
        active = (grid >= starts[None, :]) & (grid < ends[None, :])
        is_h = (leader == "human")
        is_r = (leader == "robot")
        h_count = np.sum(active & is_h[None, :], axis=1).astype(float)
        r_count = np.sum(active & is_r[None, :], axis=1).astype(float)
        denom = h_count + r_count
        with np.errstate(divide="ignore", invalid="ignore"):
            diff = np.where(denom > 0, (h_count - r_count) / denom, 0.0)

        # Smooth the majority curve with a Gaussian kernel in step-space
        sigma = max(float(CONFIG["lead_kernel_sigma_steps"]), 1.0)
        dx = 1.0
        half_width = int(max(3, np.ceil(3 * sigma / dx)))
        max_half = max(1, (x_grid.size - 1) // 2)
        half_width = min(half_width, max_half)
        kx = np.arange(-half_width, half_width + 1, dtype=float) * dx
        kernel = np.exp(-0.5 * (kx / sigma) ** 2)
        kernel /= np.sum(kernel)
        diff_smooth = np.convolve(diff, kernel, mode="same")

        ax_prob = ax.twinx()
        above = diff_smooth >= 0
        ax_prob.plot(
            x_grid,
            np.where(above, diff_smooth, np.nan),
            color="#000000",
            lw=2.5,
            alpha=0.9,
            linestyle="-",
            label="Human majority (above 0)",
        )
        ax_prob.plot(
            x_grid,
            np.where(~above, diff_smooth, np.nan),
            color="#000000",
            lw=2.5,
            alpha=0.9,
            linestyle="--",
            label="Robot majority (below 0)",
        )
        ax_prob.axhline(0, color="#666666", lw=1.0, alpha=0.7, linestyle="--")
        ax_prob.set_ylabel("Lead Majority (Human − Robot)")
        ax_prob.set_ylim(-1, 1)
        # Single legend on main axis only.

    legend_items = [
        Line2D([0], [0], color=leader_colors["human"], lw=6, label="Human"),
        Line2D([0], [0], color=leader_colors["robot"], lw=6, label="Robot"),
        Line2D([0], [0], color=leader_colors["both"], lw=6, label="Both"),
        Line2D([0], [0], color=leader_colors["unknown"], lw=6, label="Unknown"),
    ]
    ax.legend(handles=legend_items, loc="upper left")
    ax.set_xlabel("Simulation Step (Turn Start)")
    ax.set_ylabel("Episode")
    ax.set_title(f"Turn Bars vs Simulation Step (All Episodes) - {participant}")
    _apply_plot_grid(ax)
    plt.tight_layout()


def _plot_turn_bars_grid(
    turn_df: pd.DataFrame, ep_df: pd.DataFrame, participants: list[str], nrows: int, ncols: int
) -> None:
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows), sharex=False, sharey=False)
    axes = np.atleast_1d(axes).ravel()

    for ax, participant in zip(axes, participants):
        sub = turn_df[turn_df["participant"] == participant]
        if sub.empty:
            ax.set_visible(False)
            continue

        leader_colors = {
            "human": "#1f77b4",
            "robot": "#ff7f0e",
            "both": "#2ca02c",
            "unknown": "#7f7f7f",
            "none": "#7f7f7f",
        }

        for _, row in sub.iterrows():
            y = float(row["episode"])
            start = float(row["start_step"])
            duration = float(row["duration"])
            color = leader_colors.get(str(row["leader"]), "#7f7f7f")
            if not (np.isfinite(start) and np.isfinite(duration) and np.isfinite(y)):
                continue
            if duration <= 0:
                continue
            ax.broken_barh([(start, duration)], (y - 0.4, 0.8), facecolors=color, alpha=0.8)

        x = sub["start_step"].to_numpy(dtype=float)
        leader = sub["leader"].astype(str).to_numpy()
        duration = sub["duration"].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(duration) & (duration > 0)
        x = x[valid]
        leader = leader[valid]
        duration = duration[valid]
        if x.size > 1:
            min_step = int(np.floor(np.min(x)))
            max_turn_step = int(np.ceil(np.max(x + duration)))
            ep_sub = ep_df[ep_df["participant"] == participant]
            max_steps = int(ep_sub["total_ep_steps"].max()) if not ep_sub.empty else None
            if max_steps is not None and np.isfinite(max_steps):
                max_step = int(max(max_steps, max_turn_step))
            else:
                max_step = max_turn_step
            x_grid = np.arange(min_step, max_step + 1, dtype=float)
            starts = x
            ends = x + duration
            grid = x_grid[:, None]
            active = (grid >= starts[None, :]) & (grid < ends[None, :])
            is_h = (leader == "human")
            is_r = (leader == "robot")
            h_count = np.sum(active & is_h[None, :], axis=1).astype(float)
            r_count = np.sum(active & is_r[None, :], axis=1).astype(float)
            denom = h_count + r_count
            with np.errstate(divide="ignore", invalid="ignore"):
                diff = np.where(denom > 0, (h_count - r_count) / denom, 0.0)

            sigma = max(float(CONFIG["lead_kernel_sigma_steps"]), 1.0)
            dx = 1.0
            half_width = int(max(3, np.ceil(3 * sigma / dx)))
            max_half = max(1, (x_grid.size - 1) // 2)
            half_width = min(half_width, max_half)
            kx = np.arange(-half_width, half_width + 1, dtype=float) * dx
            kernel = np.exp(-0.5 * (kx / sigma) ** 2)
            kernel /= np.sum(kernel)
            diff_smooth = np.convolve(diff, kernel, mode="same")

            ax_prob = ax.twinx()
            above = diff_smooth >= 0
            ax_prob.plot(
                x_grid,
                np.where(above, diff_smooth, np.nan),
                color="#000000",
                lw=2.0,
                alpha=0.9,
                linestyle="-",
            )
            ax_prob.plot(
                x_grid,
                np.where(~above, diff_smooth, np.nan),
                color="#000000",
                lw=2.0,
                alpha=0.9,
                linestyle="--",
            )
            ax_prob.axhline(0, color="#666666", lw=1.0, alpha=0.7, linestyle="--")
            ax_prob.set_ylim(-1, 1)

        ax.set_title(participant)
        ax.set_xlabel("Simulation Step")
        ax.set_ylabel("Episode")
        _apply_plot_grid(ax)

    for ax in axes[len(participants):]:
        ax.set_visible(False)

    legend_items = [
        Line2D([0], [0], color="#1f77b4", lw=6, label="Human"),
        Line2D([0], [0], color="#ff7f0e", lw=6, label="Robot"),
        Line2D([0], [0], color="#2ca02c", lw=6, label="Both"),
        Line2D([0], [0], color="#7f7f7f", lw=6, label="Unknown"),
        Line2D([0], [0], color="#000000", lw=2, linestyle="-", label="Human majority (line)"),
        Line2D([0], [0], color="#000000", lw=2, linestyle="--", label="Robot majority (line)"),
    ]
    fig.legend(handles=legend_items, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("Turn Bars and Lead Majority (All Episodes)", y=1.02)
    plt.tight_layout()


def _compute_progress_curves(
    turn_events_df: pd.DataFrame,
    ep_df: pd.DataFrame,
    scope: str = "global",
    participant: str | None = None,
    run_id: str | None = None,
    zone: str | None = None,
    n_bins: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute dominance and confidence curves over normalized episode progress."""
    if turn_events_df.empty:
        print("Warning: No turn event data.")
        return np.array([]), np.array([]), np.array([])

    events = turn_events_df.copy()
    if scope == "participant":
        if participant is None:
            raise ValueError("participant must be provided when scope='participant'.")
        events = events[events["participant"] == participant]
    elif scope == "run":
        if participant is None or run_id is None:
            raise ValueError("participant and run_id must be provided when scope='run'.")
        events = events[(events["participant"] == participant) & (events["run_id"] == run_id)]
    elif scope != "global":
        raise ValueError(f"Unsupported scope: {scope}")

    if zone is not None:
        events = events[events["zone_context"].astype(str) == str(zone)]

    if events.empty:
        print("Warning: No events after scope filtering.")
        return np.array([]), np.array([]), np.array([])

    ep_key_cols = ["participant", "run_id", "episode"]
    ep_lengths = ep_df[ep_key_cols + ["total_ep_steps"]].drop_duplicates()
    events = events.merge(ep_lengths, on=ep_key_cols, how="left")
    events = events[np.isfinite(events["total_ep_steps"]) & (events["total_ep_steps"] > 0)]
    if events.empty:
        print("Warning: No events with valid episode lengths.")
        return np.array([]), np.array([]), np.array([])

    # Normalize time within each episode to align different episode lengths.
    events = events.copy()
    events["end_step"] = events["start_step"] + events["duration"]
    events["p_start"] = events["start_step"] / events["total_ep_steps"]
    events["p_end"] = events["end_step"] / events["total_ep_steps"]
    events = events[np.isfinite(events["p_start"]) & np.isfinite(events["p_end"])]
    events["p_start"] = np.clip(events["p_start"], 0.0, 1.0)
    events["p_end"] = np.clip(events["p_end"], 0.0, 1.0)

    p_grid = np.linspace(0.0, 1.0, n_bins + 1)
    p_centers = 0.5 * (p_grid[:-1] + p_grid[1:])
    D = np.full(n_bins, np.nan, dtype=float)
    C = np.full(n_bins, np.nan, dtype=float)
    p_start = events["p_start"].to_numpy(dtype=float)
    p_end = events["p_end"].to_numpy(dtype=float)
    leader = events["leader"].astype(str).to_numpy()
    d = events["dominance_event"].to_numpy(dtype=float)
    w = events["weight_event"].to_numpy(dtype=float)
    valid = np.isfinite(p_start) & np.isfinite(p_end) & np.isfinite(d) & np.isfinite(w)
    p_start = p_start[valid]
    p_end = p_end[valid]
    leader = leader[valid]
    d = d[valid]
    w = w[valid]

    for k in range(n_bins):
        # Find events active in this progress bin.
        bin_start = p_grid[k]
        bin_end = p_grid[k + 1]
        active = (p_start < bin_end) & (p_end > bin_start)
        if not np.any(active):
            continue
        # Exclude unknown events from dominance but include them in confidence.
        informative = active & (leader != "unknown")
        n_all = int(np.sum(active))
        n_inf = int(np.sum(informative))
        if n_all > 0:
            C[k] = n_inf / n_all
        if n_inf == 0:
            continue
        # Weighted mean dominance for this bin.
        d_inf = d[informative]
        w_inf = w[informative]
        w_sum = float(np.sum(w_inf))
        if w_sum <= 0:
            continue
        D[k] = float(np.sum(w_inf * d_inf) / w_sum)

    return p_centers, D, C


def plot_dominance_over_progress(
    turn_events_df: pd.DataFrame,
    ep_df: pd.DataFrame,
    scope: str = "global",
    participant: str | None = None,
    run_id: str | None = None,
    n_bins: int = 200,
    ax: plt.Axes | None = None,
) -> None:
    """Plot dominance over normalized episode progress."""
    p_centers, D, _ = _compute_progress_curves(
        turn_events_df=turn_events_df,
        ep_df=ep_df,
        scope=scope,
        participant=participant,
        run_id=run_id,
        zone=None,
        n_bins=n_bins,
    )
    if p_centers.size == 0:
        return

    created_ax = ax is None
    if created_ax:
        _, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.plot(p_centers, D, label="Turn Dominance (Human − Robot)")
    ax.axhline(0, linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Episode Progress p")
    ax.set_ylabel("Weighted Mean Turn Dominance")
    ax.set_ylim(-1.0, 1.0)
    _apply_plot_grid(ax)
    lines1, labels1 = ax.get_legend_handles_labels()
    ax.legend(lines1, labels1, loc="upper right")
    if scope == "global":
        title = "Turn Dominance Over Episode Progress (All Participants)"
    elif scope == "participant":
        title = f"Turn Dominance Over Episode Progress ({participant})"
    else:
        title = f"Turn Dominance Over Episode Progress ({participant} | {run_id})"
    ax.set_title(title)
    if created_ax:
        plt.tight_layout()


def plot_confidence_over_progress(
    turn_events_df: pd.DataFrame,
    ep_df: pd.DataFrame,
    scope: str = "global",
    participant: str | None = None,
    run_id: str | None = None,
    n_bins: int = 200,
) -> None:
    """Plot informative-event confidence over normalized episode progress."""
    p_centers, _, C = _compute_progress_curves(
        turn_events_df=turn_events_df,
        ep_df=ep_df,
        scope=scope,
        participant=participant,
        run_id=run_id,
        zone=None,
        n_bins=n_bins,
    )
    if p_centers.size == 0:
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.plot(
        p_centers,
        C,
        linestyle="-",
        linewidth=2.2,
        marker="o",
        markersize=2.2,
        color="#2a9d8f",
        label="Confidence C(p)",
        zorder=3,
    )
    ax.set_xlabel("Episode Progress p")
    ax.set_ylabel("Confidence (Informative Fraction)")
    # Keep headroom so a flat line at 1.0 is not hidden by the top spine.
    ax.set_ylim(0, 1.02)
    _apply_plot_grid(ax)
    finite = np.isfinite(C)
    if np.any(finite):
        cmin = float(np.nanmin(C))
        cmax = float(np.nanmax(C))
        if abs(cmax - cmin) < 1e-10:
            ax.text(
                0.02,
                0.96,
                f"C(p) is constant at {cmin:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color="#0f172a",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cbd5e1", alpha=0.85),
            )
    if scope == "global":
        title = "Confidence Over Episode Progress (All Participants)"
    elif scope == "participant":
        title = f"Confidence Over Episode Progress ({participant})"
    else:
        title = f"Confidence Over Episode Progress ({participant} | {run_id})"
    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.tight_layout()


def plot_scores_over_progress(
    turn_events_df: pd.DataFrame,
    ep_df: pd.DataFrame,
    scope: str = "global",
    participant: str | None = None,
    run_id: str | None = None,
    n_bins: int = 200,
) -> None:
    """Plot weighted mean turn scores (human and robot) over normalized episode progress."""
    p_centers, D, _ = _compute_progress_curves(
        turn_events_df=turn_events_df,
        ep_df=ep_df,
        scope=scope,
        participant=participant,
        run_id=run_id,
        zone=None,
        n_bins=n_bins,
    )
    if p_centers.size == 0:
        return

    score_h = (D + 1.0) / 2.0
    score_r = (1.0 - D) / 2.0

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.plot(p_centers, score_h, color="#1f77b4", linewidth=2.0, label="Human score")
    ax.plot(p_centers, score_r, color="#ff7f0e", linewidth=2.0, label="Robot score")
    ax.set_xlabel("Episode Progress p")
    ax.set_ylabel("Weighted Mean Turn Score")
    ax.set_ylim(0, 1)
    _apply_plot_grid(ax)
    if scope == "global":
        title = "Turn Scores Over Episode Progress (All Participants)"
    elif scope == "participant":
        title = f"Turn Scores Over Episode Progress ({participant})"
    else:
        title = f"Turn Scores Over Episode Progress ({participant} | {run_id})"
    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.tight_layout()


def plot_dominance_over_progress_participants_grid(
    turn_events_df: pd.DataFrame,
    ep_df: pd.DataFrame,
    zone_steps_df: pd.DataFrame,
    participants: list[str],
    nrows: int = 2,
    ncols: int = 3,
) -> None:
    """Plot participant-level dominance curves with zone-occupancy background."""
    if not participants:
        print("Warning: No participants for subplot dominance plot.")
        return

    axis_label_size = float(CONFIG.get("participants_grid_axis_label_size", 12))
    tick_label_size = float(CONFIG.get("participants_grid_tick_label_size", 11))
    legend_font_size = float(CONFIG.get("participants_grid_legend_font_size", 11))

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 9), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for idx, (ax, participant) in enumerate(zip(axes, participants[: nrows * ncols])):
        p_centers, D, _ = _compute_progress_curves(
            turn_events_df=turn_events_df,
            ep_df=ep_df,
            scope="participant",
            participant=participant,
            n_bins=200,
        )
        if p_centers.size == 0:
            ax.set_visible(False)
            continue

        pz, open_frac, obs_frac, goal_frac = _compute_zone_occupancy_curves(
            zone_steps_df=zone_steps_df,
            scope="participant",
            participant=participant,
            n_bins=200,
        )
        if pz.size > 0:
            _add_zone_background(ax, pz, open_frac, obs_frac, goal_frac)

        ax.plot(p_centers, D, label="Dominance", color="#111827", linewidth=1.8, zorder=3)
        ax.axhline(0, linestyle="--", linewidth=1.0, alpha=0.7, color="#4b5563", zorder=2)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=tick_label_size)
        ax.set_ylim(-1.0, 1.0)
        if bool(CONFIG.get("participants_grid_show_titles", False)):
            participant_label = " ".join(str(participant).replace("_", " ").split())
            ax.set_title(participant_label, fontsize=axis_label_size)
        _apply_plot_grid(ax)

    for ax in axes[len(participants[: nrows * ncols]) :]:
        ax.set_visible(False)

    legend_items = [
        Line2D([0], [0], color="#111827", lw=2.0, label="Turn dominance"),
        Line2D([0], [0], color="#1f77b4", lw=6, alpha=0.5, label="Open zone"),
        Line2D([0], [0], color="#d97706", lw=6, alpha=0.5, label="Obstacle zone"),
        Line2D([0], [0], color="#2a9d8f", lw=6, alpha=0.5, label="Goal zone"),
    ]
    legend_pos = str(CONFIG.get("participants_grid_legend_position", "above")).strip().lower()
    if legend_pos == "below":
        fig.legend(
            handles=legend_items,
            loc="lower center",
            ncol=4,
            bbox_to_anchor=(0.5, -0.01),
            fontsize=legend_font_size,
        )
    else:
        fig.legend(
            handles=legend_items,
            loc="upper center",
            ncol=4,
            bbox_to_anchor=(0.5, 0.99),
            fontsize=legend_font_size,
        )
    if hasattr(fig, "supxlabel"):
        fig.supxlabel("Episode Progress p", fontsize=axis_label_size)
    else:
        fig.text(0.5, 0.02, "Episode Progress p", ha="center", va="center", fontsize=axis_label_size)
    if hasattr(fig, "supylabel"):
        fig.supylabel("Weighted Mean Turn Dominance", fontsize=axis_label_size)
    else:
        fig.text(
            0.02,
            0.5,
            "Weighted Mean Turn Dominance",
            ha="center",
            va="center",
            rotation="vertical",
            fontsize=axis_label_size,
        )
    if bool(CONFIG.get("participants_grid_show_titles", False)):
        fig.suptitle("Turn Dominance Over Episode Progress (Per Participant, Zone Background)", y=1.03)
    if legend_pos == "below":
        plt.tight_layout(rect=[0.05, 0.08, 1.0, 1.0])
    else:
        plt.tight_layout(rect=[0.05, 0.04, 1.0, 0.94])


def plot_dominance_over_progress_by_zone(
    turn_events_df: pd.DataFrame,
    ep_df: pd.DataFrame,
    scope: str = "global",
    participant: str | None = None,
    run_id: str | None = None,
    n_bins: int = 200,
) -> None:
    """Plot progress dominance split by spatial context (Open / Obstacle / Goal)."""
    zones = [
        ("Open", "#1f77b4"),
        ("Obstacle", "#d97706"),
        ("Goal", "#2a9d8f"),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    plotted = False
    for zone_name, color in zones:
        p_centers, D, _ = _compute_progress_curves(
            turn_events_df=turn_events_df,
            ep_df=ep_df,
            scope=scope,
            participant=participant,
            run_id=run_id,
            zone=zone_name,
            n_bins=n_bins,
        )
        if p_centers.size == 0:
            continue
        ax.plot(p_centers, D, linewidth=2.0, color=color, label=zone_name)
        plotted = True

    if not plotted:
        print("Warning: No zone-specific dominance data to plot.")
        plt.close(fig)
        return

    ax.axhline(0, linestyle="--", linewidth=1.0, alpha=0.6, color="#4b5563")
    ax.set_xlabel("Episode Progress p")
    ax.set_ylabel("Weighted Mean Turn Dominance")
    ax.set_ylim(-1.0, 1.0)
    _apply_plot_grid(ax)
    if scope == "global":
        title = "Turn Dominance Over Episode Progress by Spatial Zone (All Participants)"
    elif scope == "participant":
        title = f"Turn Dominance Over Episode Progress by Spatial Zone ({participant})"
    else:
        title = f"Turn Dominance Over Episode Progress by Spatial Zone ({participant} | {run_id})"
    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.tight_layout()


def plot_dominance_over_progress_with_zone_background(
    turn_events_df: pd.DataFrame,
    ep_df: pd.DataFrame,
    zone_steps_df: pd.DataFrame,
    scope: str = "global",
    participant: str | None = None,
    run_id: str | None = None,
    n_bins: int = 200,
) -> None:
    """Plot dominance over progress with occupancy-based Open/Obstacle/Goal background."""
    p_centers, D, _ = _compute_progress_curves(
        turn_events_df=turn_events_df,
        ep_df=ep_df,
        scope=scope,
        participant=participant,
        run_id=run_id,
        zone=None,
        n_bins=n_bins,
    )
    if p_centers.size == 0:
        return

    pz, open_frac, obs_frac, goal_frac = _compute_zone_occupancy_curves(
        zone_steps_df=zone_steps_df,
        scope=scope,
        participant=participant,
        run_id=run_id,
        n_bins=n_bins,
    )

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    if pz.size > 0:
        _add_zone_background(ax, pz, open_frac, obs_frac, goal_frac)
    ax.plot(p_centers, D, label="Turn Dominance (Human − Robot)", color="#111827", linewidth=2.0, zorder=3)
    ax.axhline(0, linestyle="--", linewidth=1.0, alpha=0.7, color="#4b5563", zorder=2)
    ax.set_xlabel("Episode Progress p")
    ax.set_ylabel("Weighted Mean Turn Dominance")
    ax.set_ylim(-1.0, 1.0)
    _apply_plot_grid(ax)

    if scope == "global":
        title = "Turn Dominance Over Progress with Occupancy-Based Zone Background"
    elif scope == "participant":
        title = f"Turn Dominance Over Progress with Zone Background ({participant})"
    else:
        title = f"Turn Dominance Over Progress with Zone Background ({participant} | {run_id})"
    ax.set_title(title)

    legend_items = [
        Line2D([0], [0], color="#111827", lw=2.0, label="Dominance"),
        Line2D([0], [0], color="#1f77b4", lw=6, alpha=0.5, label="Open zone"),
        Line2D([0], [0], color="#d97706", lw=6, alpha=0.5, label="Obstacle zone"),
        Line2D([0], [0], color="#2a9d8f", lw=6, alpha=0.5, label="Goal zone"),
    ]
    ax.legend(handles=legend_items, loc="upper right")
    plt.tight_layout()


def main() -> None:
    _configure_ieee_plot_style(CONFIG)
    root_dir = CONFIG["root_dir"]
    if not os.path.isdir(root_dir):
        print(f"Error: root_dir not found: {root_dir}")
        return

    runs = _discover_steps_dirs(root_dir)
    if not runs:
        print(f"No runs found under: {root_dir}")
        return

    all_records: list[dict] = []
    all_dominance: list[dict] = []
    all_turn_timelines: list[dict] = []
    all_zone_steps: list[dict] = []
    for run in runs:
        df = _load_steps_df(run.steps_dir)
        if df.empty:
            continue
        ep_records, dom_records = _episode_records_for_run(df, run, CONFIG)
        all_records.extend(ep_records)
        all_dominance.extend(dom_records)
        all_turn_timelines.extend(_turn_event_timeline_records_for_run(df, run, CONFIG))
        all_zone_steps.extend(_step_zone_progress_records_for_run(df, run, CONFIG))

    if not all_records:
        print("No episode records produced. Check input data.")
        return

    ep_df = pd.DataFrame(all_records)
    dom_df = pd.DataFrame(all_dominance)

    agg = (
        ep_df.groupby("participant")[["human", "robot", "both", "unknown"]]
        .sum()
        .reset_index()
    )
    agg["total"] = agg[["human", "robot", "both", "unknown"]].sum(axis=1).replace(0, np.nan)
    agg["human_frac"] = agg["human"] / agg["total"]
    agg["robot_frac"] = agg["robot"] / agg["total"]
    agg["both_frac"] = agg["both"] / agg["total"]
    agg["unknown_frac"] = agg["unknown"] / agg["total"]
    agg = agg.fillna(0.0)

    # _plot_leader_share_by_participant(agg)
    # if dom_df.empty:
    #     print("Warning: No turn events found for dominance plot.")
    # else:
    #     _plot_dominance_index(dom_df)
    if all_turn_timelines:
        turn_df = pd.DataFrame(all_turn_timelines)
        zone_steps_df = pd.DataFrame(all_zone_steps)
        # Filter episode-length outliers using IQR.
        q1 = ep_df["total_ep_steps"].quantile(0.25)
        q3 = ep_df["total_ep_steps"].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        keep_eps = ep_df[(ep_df["total_ep_steps"] >= lower) & (ep_df["total_ep_steps"] <= upper)]
        if keep_eps.empty:
            print("Warning: All episodes filtered as outliers; using full dataset.")
            keep_eps = ep_df
        keep_keys = set(zip(keep_eps["participant"], keep_eps["run_id"], keep_eps["episode"]))
        turn_df = turn_df[
            turn_df.apply(
                lambda r: (r["participant"], r["run_id"], r["episode"]) in keep_keys, axis=1
            )
        ]
        if not zone_steps_df.empty:
            zone_steps_df = zone_steps_df[
                zone_steps_df.apply(
                    lambda r: (r["participant"], r["run_id"], r["episode"]) in keep_keys, axis=1
                )
            ]
        # plot_dominance_over_progress_with_zone_background(
        #     turn_df,
        #     keep_eps,
        #     zone_steps_df=zone_steps_df,
        #     scope="global",
        # )
        # plot_confidence_over_progress(turn_df, keep_eps, scope="global")
        # plot_scores_over_progress(turn_df, keep_eps, scope="global")
        participants = sorted(turn_df["participant"].astype(str).unique())
        plot_dominance_over_progress_participants_grid(
            turn_df, keep_eps, zone_steps_df, participants, nrows=3, ncols=2
        )
    else:
        print("Warning: No turn duration records found.")
    plt.show()


if __name__ == "__main__":
    main()
