#!/usr/bin/env python3
from __future__ import annotations

import glob
import os
from typing import Optional, Sequence, Tuple
from matplotlib.patches import Patch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




# ============================================================
# Paths (match train_collaboratively layout)
# ============================================================

def resolve_online_root(base_dir: str, load_run_subdir: str, model_name: str) -> str:
    root_parent = os.path.join("runs", "online_human", "jan", base_dir)

    candidate_full = os.path.join(root_parent, load_run_subdir)
    if os.path.isdir(candidate_full):
        return candidate_full

    candidate_ts = os.path.join(root_parent, f"{load_run_subdir}_{model_name}")
    if os.path.isdir(candidate_ts):
        return candidate_ts

    raise FileNotFoundError(
        "Could not resolve online_root. Tried:\n"
        f"  1) {candidate_full}\n"
        f"  2) {candidate_ts}\n"
        "Check base_dir/load_run_subdir/model_name against your run folder names."
    )


def resolve_steps_dir(base_dir: str, load_run_subdir: str, model_name: str) -> str:
    online_root = resolve_online_root(base_dir, load_run_subdir, model_name)
    parquet_root = os.path.join(online_root, "parquet")

    if not os.path.isdir(parquet_root):
        raise FileNotFoundError(f"Parquet root not found: {parquet_root}")

    run_dirs = [
        os.path.join(parquet_root, d)
        for d in os.listdir(parquet_root)
        if os.path.isdir(os.path.join(parquet_root, d))
    ]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under: {parquet_root}")

    run_dir = max(run_dirs, key=lambda d: os.path.getmtime(d))

    steps_dir = os.path.join(run_dir, "steps")
    if not os.path.isdir(steps_dir):
        raise FileNotFoundError(f"Steps dir not found: {steps_dir}")

    part_files = glob.glob(os.path.join(steps_dir, "part-*.parquet"))
    if not part_files:
        raise FileNotFoundError(f"No part-*.parquet files found in: {steps_dir}")

    return steps_dir


# ============================================================
# Loading
# ============================================================

def load_parquet_steps(steps_dir: str, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(steps_dir, "part-*.parquet")))
    if not files:
        raise FileNotFoundError(f"No part-*.parquet files found in: {steps_dir}")

    dfs = []
    for fp in files:
        dfs.append(pd.read_parquet(fp, columns=list(columns) if columns is not None else None))
    df = pd.concat(dfs, ignore_index=True)

    if "global_step" in df.columns:
        df = df.sort_values("global_step", kind="stable").reset_index(drop=True)
    elif "episode" in df.columns and "t" in df.columns:
        df = df.sort_values(["episode", "t"], kind="stable").reset_index(drop=True)

    return df


# ============================================================
# Plot helpers
# ============================================================

def plot_relative_contribution_effort(
    df: pd.DataFrame,
    episode_to_plot: Optional[int] = None,
    dt: Optional[float] = None,
    x_axis: str = "t",                 # "t" or "global_step"
    smooth_seconds: Optional[float] = 0.25,
    min_total_effort: float = 0.0,     # set e.g. 1e-3 to hide near-zero noise
    show_signed_signals: bool = False,
) -> Tuple[pd.DataFrame, Tuple[plt.Figure, ...]]:
    """
    Option (1): relative robot vs human contribution based on command magnitudes.

    Decomposition:
        act_exec = act_raw + delta
    where:
        delta_v = act_exec_v - act_raw_v
        delta_w = act_exec_w - act_raw_w

    Contribution ("effort share") uses magnitudes:
        robot_effort = |act_raw|
        human_effort = |delta|
        human_frac = human_effort / (human_effort + robot_effort + eps)

    Returns:
        dfp: filtered copy of df with additional columns
        figs: tuple of matplotlib Figure objects
    """
    # If delta_* not present, compute it from act_exec_* and act_raw_*
    if "delta_v" not in df.columns:
        if "act_exec_v" not in df.columns or "act_raw_v" not in df.columns:
            raise KeyError("Need either delta_v or (act_exec_v and act_raw_v).")
        df = df.copy()
        df["delta_v"] = df["act_exec_v"] - df["act_raw_v"]

    if "delta_w" not in df.columns:
        if "act_exec_w" not in df.columns or "act_raw_w" not in df.columns:
            raise KeyError("Need either delta_w or (act_exec_w and act_raw_w).")
        df = df.copy()
        df["delta_w"] = df["act_exec_w"] - df["act_raw_w"]

    required = ["act_raw_v", "act_raw_w", "delta_v", "delta_w"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for contribution plot: {missing}")

    # Filter episode (optional)
    if episode_to_plot is not None:
        if "episode" not in df.columns:
            raise KeyError("episode_to_plot was provided but df has no 'episode' column.")
        dfp = df[df["episode"] == episode_to_plot].copy()
    else:
        dfp = df.copy()

    if dfp.empty:
        raise ValueError("No data to plot after filtering (dfp is empty).")

    # Choose x-axis
    if x_axis in dfp.columns:
        x = dfp[x_axis].to_numpy()
    else:
        # fallback
        x = np.arange(len(dfp))

    eps = 1e-9

    # Effort magnitudes
    dfp["robot_effort_v"] = np.abs(dfp["act_raw_v"].to_numpy())
    dfp["human_effort_v"] = np.abs(dfp["delta_v"].to_numpy())
    dfp["robot_effort_w"] = np.abs(dfp["act_raw_w"].to_numpy())
    dfp["human_effort_w"] = np.abs(dfp["delta_w"].to_numpy())

    # Fractions
    den_v = dfp["robot_effort_v"] + dfp["human_effort_v"] + eps
    den_w = dfp["robot_effort_w"] + dfp["human_effort_w"] + eps
    dfp["human_frac_v"] = dfp["human_effort_v"] / den_v
    dfp["robot_frac_v"] = 1.0 - dfp["human_frac_v"]
    dfp["human_frac_w"] = dfp["human_effort_w"] / den_w
    dfp["robot_frac_w"] = 1.0 - dfp["human_frac_w"]

    # Optional smoothing
    if smooth_seconds is not None:
        if dt is None or dt <= 0:
            raise ValueError("smooth_seconds was provided but dt is None/invalid. Provide dt or set smooth_seconds=None.")
        window = max(1, int(round(smooth_seconds / dt)))

        dfp["human_frac_v_s"] = dfp["human_frac_v"].rolling(window, min_periods=1).mean()
        dfp["human_frac_w_s"] = dfp["human_frac_w"].rolling(window, min_periods=1).mean()
        dfp["robot_frac_v_s"] = 1.0 - dfp["human_frac_v_s"]
        dfp["robot_frac_w_s"] = 1.0 - dfp["human_frac_w_s"]

        frac_v_robot = dfp["robot_frac_v_s"].to_numpy()
        frac_v_human = dfp["human_frac_v_s"].to_numpy()
        frac_w_robot = dfp["robot_frac_w_s"].to_numpy()
        frac_w_human = dfp["human_frac_w_s"].to_numpy()

        v_title_suffix = f" (smoothed {smooth_seconds:.3g}s)"
        w_title_suffix = f" (smoothed {smooth_seconds:.3g}s)"
    else:
        frac_v_robot = dfp["robot_frac_v"].to_numpy()
        frac_v_human = dfp["human_frac_v"].to_numpy()
        frac_w_robot = dfp["robot_frac_w"].to_numpy()
        frac_w_human = dfp["human_frac_w"].to_numpy()

        v_title_suffix = ""
        w_title_suffix = ""

    # Optional masking of near-zero total effort
    if min_total_effort > 0.0:
        mask_v = (dfp["robot_effort_v"] + dfp["human_effort_v"]).to_numpy() > float(min_total_effort)
        mask_w = (dfp["robot_effort_w"] + dfp["human_effort_w"]).to_numpy() > float(min_total_effort)
    else:
        mask_v = np.ones(len(dfp), dtype=bool)
        mask_w = np.ones(len(dfp), dtype=bool)

    figs = []

    fig_v = plt.figure()
    plt.stackplot(x[mask_v], frac_v_robot[mask_v], frac_v_human[mask_v], labels=["robot", "human"])
    plt.ylim(0, 1)
    plt.xlabel(x_axis)
    plt.ylabel("fraction of effort")
    plt.title(f"Relative contribution (linear v): |r_v| vs |h_v|{v_title_suffix}")
    plt.legend(loc="upper right")
    figs.append(fig_v)

    fig_w = plt.figure()
    plt.stackplot(x[mask_w], frac_w_robot[mask_w], frac_w_human[mask_w], labels=["robot", "human"])
    plt.ylim(0, 1)
    plt.xlabel(x_axis)
    plt.ylabel("fraction of effort")
    plt.title(f"Relative contribution (angular w): |r_w| vs |h_w|{w_title_suffix}")
    plt.legend(loc="upper right")
    figs.append(fig_w)

    if show_signed_signals:
        fig_v_sig = plt.figure()
        plt.plot(x, dfp["act_raw_v"].to_numpy(), label="raw v (robot)")
        plt.plot(x, dfp["delta_v"].to_numpy(), label="delta v (human)")
        if "act_exec_v" in dfp.columns:
            plt.plot(x, dfp["act_exec_v"].to_numpy(), label="exec v")
        plt.xlabel(x_axis)
        plt.title("Linear velocity decomposition (signed)")
        plt.legend(loc="best")
        figs.append(fig_v_sig)

        fig_w_sig = plt.figure()
        plt.plot(x, dfp["act_raw_w"].to_numpy(), label="raw w (robot)")
        plt.plot(x, dfp["delta_w"].to_numpy(), label="delta w (human)")
        if "act_exec_w" in dfp.columns:
            plt.plot(x, dfp["act_exec_w"].to_numpy(), label="exec w")
        plt.xlabel(x_axis)
        plt.title("Angular velocity decomposition (signed)")
        plt.legend(loc="best")
        figs.append(fig_w_sig)

    return dfp, tuple(figs)


from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_collaboration_conflict(
    df: pd.DataFrame,
    episode_to_plot: Optional[int] = None,
    x_axis: str = "t",                   # "t" or "global_step"
    dt: Optional[float] = None,
    smooth_seconds: Optional[float] = 0.25,
    min_activity: float = 1e-3,          # ignore steps where |raw|+|delta| is tiny
    normalize: bool = True,              # plot fractions (assist vs override) if True, else raw magnitudes
    show_alignment: bool = True,         # also plot alignment in [-1,1]
) -> Tuple[pd.DataFrame, Dict[str, float], Tuple[plt.Figure, ...]]:
    """
    Metric (2): collaboration vs conflict between robot and human, per DOF (v and w).

    Uses 1-D signed alignment between robot command r and human contribution h:
        r = act_raw_*
        h = delta_* = act_exec_* - act_raw_*

    Define per-step:
        prod = r*h
        assist   = max(0,  prod)         # same direction (collaboration)
        override = max(0, -prod)         # opposite direction (conflict)
        alignment = prod / (|r||h| + eps)  in [-1,1] (degenerates to ±1 in 1-D when both nonzero)

    Plots:
        - Assist vs Override (stacked) for v and w (optionally normalized to fractions)
        - Optional alignment traces for v and w

    Returns:
        dfp: filtered dataframe with computed columns
        summary: overall conflict/collaboration rates (effort-weighted) for v and w
        figs: matplotlib figures
    """
    eps = 1e-9

    # Ensure delta_* exists; compute if missing
    df0 = df
    if "delta_v" not in df0.columns:
        if "act_exec_v" not in df0.columns or "act_raw_v" not in df0.columns:
            raise KeyError("Need delta_v or (act_exec_v and act_raw_v).")
        df0 = df0.copy()
        df0["delta_v"] = df0["act_exec_v"] - df0["act_raw_v"]

    if "delta_w" not in df0.columns:
        if "act_exec_w" not in df0.columns or "act_raw_w" not in df0.columns:
            raise KeyError("Need delta_w or (act_exec_w and act_raw_w).")
        df0 = df0.copy()
        df0["delta_w"] = df0["act_exec_w"] - df0["act_raw_w"]

    required = ["act_raw_v", "act_raw_w", "delta_v", "delta_w"]
    missing = [c for c in required if c not in df0.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Filter episode (optional)
    if episode_to_plot is not None:
        if "episode" not in df0.columns:
            raise KeyError("episode_to_plot was provided but df has no 'episode' column.")
        dfp = df0[df0["episode"] == episode_to_plot].copy()
    else:
        dfp = df0.copy()

    if dfp.empty:
        raise ValueError("No data to plot after filtering (dfp is empty).")

    # X-axis
    if x_axis in dfp.columns:
        x = dfp[x_axis].to_numpy()
    else:
        x = np.arange(len(dfp))

    # --- Compute per-step assist/override and alignment ---
    rv = dfp["act_raw_v"].to_numpy()
    hv = dfp["delta_v"].to_numpy()
    rw = dfp["act_raw_w"].to_numpy()
    hw = dfp["delta_w"].to_numpy()

    prod_v = rv * hv
    prod_w = rw * hw

    dfp["assist_v"] = np.maximum(0.0, prod_v)
    dfp["override_v"] = np.maximum(0.0, -prod_v)
    dfp["assist_w"] = np.maximum(0.0, prod_w)
    dfp["override_w"] = np.maximum(0.0, -prod_w)

    dfp["alignment_v"] = prod_v / (np.abs(rv) * np.abs(hv) + eps)
    dfp["alignment_w"] = prod_w / (np.abs(rw) * np.abs(hw) + eps)

    # Activity mask (avoid dividing by tiny values / noisy fractions)
    act_v = (np.abs(rv) + np.abs(hv))
    act_w = (np.abs(rw) + np.abs(hw))
    mask_v = act_v > float(min_activity)
    mask_w = act_w > float(min_activity)

    # Optional smoothing (rolling mean) for readability
    if smooth_seconds is not None:
        if dt is None or dt <= 0:
            raise ValueError("smooth_seconds was provided but dt is None/invalid. Provide dt or set smooth_seconds=None.")
        window = max(1, int(round(smooth_seconds / dt)))

        for col in ["assist_v", "override_v", "assist_w", "override_w", "alignment_v", "alignment_w"]:
            dfp[f"{col}_s"] = dfp[col].rolling(window, min_periods=1).mean()

        assist_v = dfp["assist_v_s"].to_numpy()
        over_v = dfp["override_v_s"].to_numpy()
        assist_w = dfp["assist_w_s"].to_numpy()
        over_w = dfp["override_w_s"].to_numpy()
        align_v = dfp["alignment_v_s"].to_numpy()
        align_w = dfp["alignment_w_s"].to_numpy()
        title_suffix = f" (smoothed {smooth_seconds:.3g}s)"
    else:
        assist_v = dfp["assist_v"].to_numpy()
        over_v = dfp["override_v"].to_numpy()
        assist_w = dfp["assist_w"].to_numpy()
        over_w = dfp["override_w"].to_numpy()
        align_v = dfp["alignment_v"].to_numpy()
        align_w = dfp["alignment_w"].to_numpy()
        title_suffix = ""

    # Normalize to fractions if requested
    if normalize:
        denom_v = assist_v + over_v + eps
        denom_w = assist_w + over_w + eps
        assist_v_plot = np.where(mask_v, assist_v / denom_v, np.nan)
        over_v_plot = np.where(mask_v, over_v / denom_v, np.nan)
        assist_w_plot = np.where(mask_w, assist_w / denom_w, np.nan)
        over_w_plot = np.where(mask_w, over_w / denom_w, np.nan)
        ylab = "fraction"
        stack_title_v = f"Collaboration vs conflict (v): assist vs override{title_suffix}"
        stack_title_w = f"Collaboration vs conflict (w): assist vs override{title_suffix}"
    else:
        assist_v_plot = np.where(mask_v, assist_v, np.nan)
        over_v_plot = np.where(mask_v, over_v, np.nan)
        assist_w_plot = np.where(mask_w, assist_w, np.nan)
        over_w_plot = np.where(mask_w, over_w, np.nan)
        ylab = "magnitude (prod units)"
        stack_title_v = f"Collaboration vs conflict (v): assist vs override magnitudes{title_suffix}"
        stack_title_w = f"Collaboration vs conflict (w): assist vs override magnitudes{title_suffix}"

    # Overall effort-weighted conflict rates (single numbers)
    sum_assist_v = float(np.nansum(dfp.loc[mask_v, "assist_v"]))
    sum_over_v = float(np.nansum(dfp.loc[mask_v, "override_v"]))
    sum_assist_w = float(np.nansum(dfp.loc[mask_w, "assist_w"]))
    sum_over_w = float(np.nansum(dfp.loc[mask_w, "override_w"]))

    conflict_rate_v = sum_over_v / (sum_assist_v + sum_over_v + eps) if (sum_assist_v + sum_over_v) > 0 else float("nan")
    conflict_rate_w = sum_over_w / (sum_assist_w + sum_over_w + eps) if (sum_assist_w + sum_over_w) > 0 else float("nan")

    summary = {
        "conflict_rate_v": conflict_rate_v,
        "collaboration_rate_v": 1.0 - conflict_rate_v if np.isfinite(conflict_rate_v) else float("nan"),
        "conflict_rate_w": conflict_rate_w,
        "collaboration_rate_w": 1.0 - conflict_rate_w if np.isfinite(conflict_rate_w) else float("nan"),
        "sum_assist_v": sum_assist_v,
        "sum_override_v": sum_over_v,
        "sum_assist_w": sum_assist_w,
        "sum_override_w": sum_over_w,
    }

    figs = []

    # --- Plot stacked assist vs override for v ---
    fig1 = plt.figure()
    # stackplot can't handle NaN well; use masked arrays
    xv = x[mask_v]
    plt.stackplot(
        xv,
        np.asarray(assist_v_plot[mask_v], dtype=float),
        np.asarray(over_v_plot[mask_v], dtype=float),
        labels=["assist (collaboration)", "override (conflict)"],
    )
    plt.xlabel(x_axis)
    plt.ylabel(ylab)
    if normalize:
        plt.ylim(0, 1)
    plt.title(stack_title_v)
    plt.legend(loc="upper right")
    figs.append(fig1)

    # --- Plot stacked assist vs override for w ---
    fig2 = plt.figure()
    xw = x[mask_w]
    plt.stackplot(
        xw,
        np.asarray(assist_w_plot[mask_w], dtype=float),
        np.asarray(over_w_plot[mask_w], dtype=float),
        labels=["assist (collaboration)", "override (conflict)"],
    )
    plt.xlabel(x_axis)
    plt.ylabel(ylab)
    if normalize:
        plt.ylim(0, 1)
    plt.title(stack_title_w)
    plt.legend(loc="upper right")
    figs.append(fig2)

    # --- Optional alignment traces ---
    if show_alignment:
        fig3 = plt.figure()
        plt.plot(x[mask_v], align_v[mask_v], label="alignment_v")
        plt.ylim(-1.05, 1.05)
        plt.xlabel(x_axis)
        plt.ylabel("alignment [-1, 1]")
        plt.title(f"Robot–human alignment (v){title_suffix}")
        plt.legend(loc="best")
        figs.append(fig3)

        fig4 = plt.figure()
        plt.plot(x[mask_w], align_w[mask_w], label="alignment_w")
        plt.ylim(-1.05, 1.05)
        plt.xlabel(x_axis)
        plt.ylabel("alignment [-1, 1]")
        plt.title(f"Robot–human alignment (w){title_suffix}")
        plt.legend(loc="best")
        figs.append(fig4)

    return dfp, summary, tuple(figs)



from typing import Optional, Dict
import numpy as np
import pandas as pd


def summarize_overall_authority_shares(
    df: pd.DataFrame,
    episode: Optional[int] = None,
    min_total_effort: float = 1e-3,
) -> Dict[str, float]:
    """
    Computes overall robot vs human dominance for v and w.

    Returns two types of summaries:
      1) Mean-of-fractions (area under fraction curves / duration)
      2) Effort-weighted share (recommended): sum(|delta|) / sum(|raw|+|delta|)

    min_total_effort: filters out near-idle timesteps where fractions are noisy.
    """
    dfp = df[df["episode"] == episode].copy() if episode is not None else df.copy()

    # Ensure required columns exist
    required = ["act_raw_v", "act_raw_w", "delta_v", "delta_w"]
    missing = [c for c in required if c not in dfp.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Magnitudes
    rv = np.abs(dfp["act_raw_v"].to_numpy())
    hv = np.abs(dfp["delta_v"].to_numpy())
    rw = np.abs(dfp["act_raw_w"].to_numpy())
    hw = np.abs(dfp["delta_w"].to_numpy())

    # Filter idle-ish timesteps
    mv = (rv + hv) > float(min_total_effort)
    mw = (rw + hw) > float(min_total_effort)

    eps = 1e-9

    # Fractions per step (only for non-idle steps)
    human_frac_v = hv[mv] / (rv[mv] + hv[mv] + eps)
    human_frac_w = hw[mw] / (rw[mw] + hw[mw] + eps)

    # 1) Mean of fractions (area under fraction curve / duration)
    mean_human_v = float(np.mean(human_frac_v)) if human_frac_v.size else float("nan")
    mean_human_w = float(np.mean(human_frac_w)) if human_frac_w.size else float("nan")

    # 2) Effort-weighted share (recommended)
    weighted_human_v = float(np.sum(hv[mv]) / (np.sum(rv[mv] + hv[mv]) + eps)) if np.any(mv) else float("nan")
    weighted_human_w = float(np.sum(hw[mw]) / (np.sum(rw[mw] + hw[mw]) + eps)) if np.any(mw) else float("nan")

    return {
        # Linear velocity v
        # "mean_human_share_v": mean_human_v,
        # "mean_robot_share_v": 1.0 - mean_human_v if np.isfinite(mean_human_v) else float("nan"),
        "weighted_human_share_v": weighted_human_v,
        "weighted_robot_share_v": 1.0 - weighted_human_v if np.isfinite(weighted_human_v) else float("nan"),

        # Angular velocity w
        # "mean_human_share_w": mean_human_w,
        # "mean_robot_share_w": 1.0 - mean_human_w if np.isfinite(mean_human_w) else float("nan"),
        "weighted_human_share_w": weighted_human_w,
        "weighted_robot_share_w": 1.0 - weighted_human_w if np.isfinite(weighted_human_w) else float("nan"),

    }



# ============================================================
# LF plot 
# ============================================================




def plot_leader_follower_lfi(
    df: pd.DataFrame,
    episode_to_plot: Optional[int] = None,
    x_axis: str = "t",                     # "t" or "global_step"
    dt: Optional[float] = None,
    window_seconds: float = 1.0,           # correlation window
    max_lag_seconds: float = 0.30,         # evaluate lags 1..L where L~max_lag_seconds/dt
    use_goal_directed: bool = True,        # predict v_parallel (toward goal) vs speed
    smooth_seconds: Optional[float] = 0.25,
    min_table_speed: float = 1e-3,         # gate near-stationary steps
    min_confidence: float = 0.10,          # gate low max-correlation
    lfi_threshold: float = 0.05,           # leader only if LFI beyond +/- threshold
    prefer_force_robot_channel: bool = True,  # if Fr_x/Fr_y exist, use them; else fallback to cmd proxy
) -> Tuple[pd.DataFrame, Tuple[plt.Figure, ...]]:
    """
    Leader–Follower Index (LFI) based on *timing* (positive-lag predictability), anchored to the table.

    Table signals:
      v_T[k] = (table_vx, table_vy)

    Human effort channel:
      p_h[k] = Fh[k] · v_T[k]

    Robot effort channel:
      If Fr_x/Fr_y available (and prefer_force_robot_channel=True):
        p_r[k] = Fr[k] · v_T[k]
      Else (fallback):
        u_r_world[k] = act_exec_v[k] * [cos(robot_yaw[k]), sin(robot_yaw[k])]
        p_r[k] = u_r_world[k] · v_T[k]

    Predicted future table motion y:
      y[k] = v_parallel[k] = v_T[k] · ghat[k]   (goal-directed), OR
      y[k] = ||v_T[k]||                          (speed)

    For lags l=1..L, compute corr(p_agent[k-window+1:k], y[k-window+1+l:k+l]) in rolling windows.
    L_agent[k] = max_l corr(...)
    LFI[k] = L_r[k] - L_h[k].

    Plot:
      - LFI curve
      - Fill only area above 0 when robot leads; only area below 0 when human leads
    """
    if dt is None or dt <= 0:
        raise ValueError("dt must be provided and > 0 for LFI timing metric.")

    # Always-required columns
    required = ["table_vx", "table_vy", "Fh_x", "Fh_y", "goal_x", "goal_y", "table_x", "table_y"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for LFI plot: {missing}")

    # Robot channel availability
    has_Fr = ("Fr_x" in df.columns) and ("Fr_y" in df.columns)
    use_Fr = bool(prefer_force_robot_channel and has_Fr)
    if not use_Fr:
        fallback_required = ["act_exec_v", "robot_yaw"]
        miss_fb = [c for c in fallback_required if c not in df.columns]
        if miss_fb:
            raise KeyError(
                "Fr_x/Fr_y not found or not used, and fallback inputs missing. "
                f"Need either (Fr_x, Fr_y) OR ({', '.join(fallback_required)}). Missing: {miss_fb}"
            )

    # Filter episode (optional)
    if episode_to_plot is not None:
        if "episode" not in df.columns:
            raise KeyError("episode_to_plot was provided but df has no 'episode' column.")
        dfp = df[df["episode"] == episode_to_plot].copy()
    else:
        dfp = df.copy()

    if dfp.empty:
        raise ValueError("No data to plot after filtering (dfp is empty).")

    # X-axis
    if x_axis in dfp.columns:
        x = dfp[x_axis].to_numpy()
    else:
        x = np.arange(len(dfp))

    # Core signals
    vx = dfp["table_vx"].to_numpy(dtype=float)
    vy = dfp["table_vy"].to_numpy(dtype=float)
    v = np.stack([vx, vy], axis=1)
    speed = np.linalg.norm(v, axis=1)

    Fh = np.stack([dfp["Fh_x"].to_numpy(dtype=float), dfp["Fh_y"].to_numpy(dtype=float)], axis=1)

    # Goal direction ghat (per step)
    gx = (dfp["goal_x"].to_numpy(dtype=float) - dfp["table_x"].to_numpy(dtype=float))
    gy = (dfp["goal_y"].to_numpy(dtype=float) - dfp["table_y"].to_numpy(dtype=float))
    gnorm = np.sqrt(gx * gx + gy * gy) + 1e-9
    ghat = np.stack([gx / gnorm, gy / gnorm], axis=1)

    # Predicted output y[k]
    if use_goal_directed:
        y = np.sum(v * ghat, axis=1)  # goal-directed table velocity
        y_label = "goal-directed table velocity v_parallel"
    else:
        y = speed
        y_label = "table speed ||v||"

    # Power-like human channel (anchored to table motion)
    ph = np.sum(Fh * v, axis=1)

    # Robot channel: force-based if available; else cmd-based proxy
    if use_Fr:
        print("use Fr")
        Fr = np.stack([dfp["Fr_x"].to_numpy(dtype=float), dfp["Fr_y"].to_numpy(dtype=float)], axis=1)
        pr = np.sum(Fr * v, axis=1)
        robot_channel_label = "robot channel: Fr·v"
    else:
        yaw = dfp["robot_yaw"].to_numpy(dtype=float)
        vcmd = dfp["act_exec_v"].to_numpy(dtype=float)
        u_r = np.stack([vcmd * np.cos(yaw), vcmd * np.sin(yaw)], axis=1)
        pr = np.sum(u_r * v, axis=1)
        robot_channel_label = "robot channel: v_cmd_world·v"

    # Optional smoothing (readability)
    if smooth_seconds is not None and smooth_seconds > 0:
        w_s = max(1, int(round(smooth_seconds / dt)))
        ph = pd.Series(ph).rolling(w_s, min_periods=1).mean().to_numpy()
        pr = pd.Series(pr).rolling(w_s, min_periods=1).mean().to_numpy()
        y = pd.Series(y).rolling(w_s, min_periods=1).mean().to_numpy()

    # Rolling window + lags
    window = max(5, int(round(window_seconds / dt)))
    max_lag = max(1, int(round(max_lag_seconds / dt)))
    lags = np.arange(1, max_lag + 1, dtype=int)

    def corr_1d(a: np.ndarray, b: np.ndarray) -> float:
        aa = a - np.mean(a)
        bb = b - np.mean(b)
        denom = (np.sqrt(np.mean(aa * aa)) * np.sqrt(np.mean(bb * bb))) + 1e-12
        if denom <= 1e-12:
            return 0.0
        return float(np.mean(aa * bb) / denom)

    n = len(dfp)
    Lh = np.full(n, np.nan, dtype=float)
    Lr = np.full(n, np.nan, dtype=float)
    Conf = np.full(n, np.nan, dtype=float)
    BestLagH = np.full(n, np.nan, dtype=float)
    BestLagR = np.full(n, np.nan, dtype=float)

    active = speed > float(min_table_speed)

    # Compute per index k using a trailing window ending at k (needs k+max_lag < n)
    for k in range(window - 1, n - max_lag):
        if not active[k]:
            continue

        i0 = k - window + 1
        i1 = k + 1  # exclusive

        # require enough activity within window
        if np.mean(active[i0:i1]) < 0.5:
            continue

        ph_w = ph[i0:i1]
        pr_w = pr[i0:i1]

        best_h = -1.0
        best_r = -1.0
        best_lh = 1
        best_lr = 1

        for lag in lags:
            y_w = y[i0 + lag : i1 + lag]
            ch = corr_1d(ph_w, y_w)
            cr = corr_1d(pr_w, y_w)
            if ch > best_h:
                best_h = ch
                best_lh = lag
            if cr > best_r:
                best_r = cr
                best_lr = lag

        Lh[k] = best_h
        Lr[k] = best_r
        Conf[k] = max(best_h, best_r)
        BestLagH[k] = float(best_lh) * dt
        BestLagR[k] = float(best_lr) * dt

    LFI = Lr - Lh

    # Store in dfp
    dfp["LFI"] = LFI
    dfp["Lh"] = Lh
    dfp["Lr"] = Lr
    dfp["LFI_conf"] = Conf
    dfp["LFI_bestlag_h_s"] = BestLagH
    dfp["LFI_bestlag_r_s"] = BestLagR

    # Role classification (for masks)
    role = np.full(n, "uncertain", dtype=object)
    valid = np.isfinite(LFI) & np.isfinite(Conf) & (Conf >= float(min_confidence))
    role[valid & (LFI > float(lfi_threshold))] = "robot"
    role[valid & (LFI < -float(lfi_threshold))] = "human"
    dfp["role_lead"] = role

    # ---- Plot (fill only above/below 0 depending on leader) ----
    fig = plt.figure()
    ax = plt.gca()

    # Zero line first (so fills are relative to it)
    ax.axhline(
    0.0,
    color="lightcoral",
    linestyle="--",
    linewidth=1.25,
    zorder=2,
)


    mask_robot = (role == "robot")
    mask_human = (role == "human")

    # Fill ONLY the area between 0 and LFI in the corresponding regions
    ax.fill_between(
        x,
        0.0,
        LFI,
        where=mask_robot,
        interpolate=True,
        color="tab:blue",
        alpha=0.25,
        label="robot leads (area above 0)",
        zorder=1,
    )
    ax.fill_between(
        x,
        0.0,
        LFI,
        where=mask_human,
        interpolate=True,
        color="tab:orange",
        alpha=0.25,
        label="human leads (area below 0)",
        zorder=1,
    )

    mask_unc = (role == "uncertain") & np.isfinite(LFI)

    ax.fill_between(
        x,
        0.0,
        LFI,
        where=mask_unc,
        interpolate=True,
        color="0.75",      # grey
        alpha=0.25,
        label="uncertain",
        zorder=1,
    )


    # LFI curve on top
    ax.plot(x, LFI, linewidth=2.0, label="LFI = Lr - Lh", zorder=3)

    ax.set_xlabel(x_axis)
    ax.set_ylabel("Leader–Follower Index (positive => robot leads)")

    title = f"Leader–Follower timing metric (predicts future {y_label})"
    if smooth_seconds is not None and smooth_seconds > 0:
        title += f" | smoothed {smooth_seconds:.3g}s"
    title += f" | window={window_seconds:.2f}s, max_lag={max_lag_seconds:.2f}s"
    title += f" | {robot_channel_label}"
    ax.set_title(title)

    ax.legend(loc="best")

    return dfp, (fig,)


def plot_obstacle_penalty_per_step(
    df: pd.DataFrame,
    episode_to_plot: Optional[int] = None,
    x_axis: str = "t",                   # "t" or "global_step"
    dt: Optional[float] = None,
    smooth_seconds: Optional[float] = 0.0,
    show_raw: bool = True,
    show_cumulative: bool = False,
) -> Tuple[pd.DataFrame, Tuple[plt.Figure, ...]]:
    """
    Plots the *instantaneous* obstacle penalty per timestep (i.e., rew_obstacle[k]),
    not a running total.

    Requires column: 'rew_obstacle'.

    Options:
      - smooth_seconds: rolling mean for readability (0 or None disables)
      - show_cumulative: additionally plots cumulative sum over time
    """
    if "rew_obstacle" not in df.columns:
        raise KeyError("Missing required column: 'rew_obstacle'")

    # Filter episode (optional)
    if episode_to_plot is not None:
        if "episode" not in df.columns:
            raise KeyError("episode_to_plot was provided but df has no 'episode' column.")
        dfp = df[df["episode"] == episode_to_plot].copy()
    else:
        dfp = df.copy()

    if dfp.empty:
        raise ValueError("No data to plot after filtering (dfp is empty).")

    # X-axis
    if x_axis in dfp.columns:
        x = dfp[x_axis].to_numpy()
    else:
        x = np.arange(len(dfp))

    pen = dfp["rew_obstacle"].to_numpy(dtype=float)
    dfp["obstacle_penalty_step"] = pen

    # Optional smoothing
    if smooth_seconds is not None and smooth_seconds > 0:
        if dt is None or dt <= 0:
            raise ValueError("smooth_seconds was provided but dt is None/invalid. Provide dt or set smooth_seconds=0.")
        w = max(1, int(round(smooth_seconds / dt)))
        pen_s = pd.Series(pen).rolling(w, min_periods=1).mean().to_numpy()
        dfp["obstacle_penalty_step_s"] = pen_s
    else:
        pen_s = None

    figs = []

    # --- Per-step penalty plot ---
    fig1 = plt.figure()
    ax = plt.gca()

    if show_raw:
        ax.plot(x, pen, linewidth=1.0, label="obstacle penalty (per step)")

    if pen_s is not None:
        ax.plot(x, pen_s, linewidth=2.0, label=f"smoothed ({smooth_seconds:.3g}s)")

    ax.axhline(0.0, color="0.6", linestyle="--", linewidth=1.0)
    ax.set_xlabel(x_axis)
    ax.set_ylabel("rew_obstacle (per timestep)")
    ttl = "Obstacle penalty per timestep"
    if episode_to_plot is not None:
        ttl += f" (episode {episode_to_plot})"
    ax.set_title(ttl)
    ax.legend(loc="best")
    figs.append(fig1)

    # --- Optional cumulative penalty plot ---
    if show_cumulative:
        fig2 = plt.figure()
        ax2 = plt.gca()
        cum = np.cumsum(pen)
        dfp["obstacle_penalty_cum"] = cum
        ax2.plot(x, cum, linewidth=2.0, label="cumulative obstacle penalty")
        ax2.set_xlabel(x_axis)
        ax2.set_ylabel("cumulative sum(rew_obstacle)")
        ttl2 = "Cumulative obstacle penalty"
        if episode_to_plot is not None:
            ttl2 += f" (episode {episode_to_plot})"
        ax2.set_title(ttl2)
        ax2.legend(loc="best")
        figs.append(fig2)

    return dfp, tuple(figs)





# ============================================================
# Main runner
# ============================================================

# def main(
#     base_dir: str,
#     load_run_subdir: str,
#     model_name: str,
#     episode_to_plot: Optional[int] = 0,
#     # Match env.apply_human_feedback gains:
#     gain_lin: float = 0.5 / 40.0,
#     gain_ang: float = 1.0 / 40.0,
#     deadzone_force_n: float = 1.0,
#     dt: float = 0.01,
# ) -> None:
#     steps_dir = resolve_steps_dir(base_dir, load_run_subdir, model_name)

#     cols = [
#         "episode", "t", "global_step", "wall_time",
#         "goal_x", "goal_y",
#         "table_x", "table_y", "table_vx", "table_vy", "table_wz",
#         "robot_yaw",
#         "Fh_x", "Fh_y",
#         "Fr_x", "Fr_y",              # <-- add these
#         "act_raw_v", "act_raw_w",
#         "act_exec_v", "act_exec_w",
#         "delta_v", "delta_w",
#         "rew_progress", "rew_obstacle",
#         "dist_table_to_goal",
#         "heading_error",
#     ]

#     df = load_parquet_steps(steps_dir, columns=cols)

#     print(df["rew_obstacle"].sum())

#     # Plot option (1): effort-share robot vs human
#     plot_relative_contribution_effort(
#         df,
#         episode_to_plot=episode_to_plot,
#         dt=dt,
#         x_axis="t",
#         smooth_seconds=0.25,
#         min_total_effort=1e-3,
#         show_signed_signals=False,
#     )

#     summary = summarize_overall_authority_shares(df, episode=episode_to_plot, min_total_effort=1e-3)

#     print("Overall authority shares (episode):", episode_to_plot)
#     for k, v in summary.items():
#         print(f"{k}: {v:.3f}")


#     dfp, summary, figs = plot_collaboration_conflict(
#     df,
#     episode_to_plot=episode_to_plot,
#     x_axis="t",
#     dt=dt,
#     smooth_seconds=0.25,
#     min_activity=1e-3,
#     normalize=True,         # show fractions
#     show_alignment=True,
# )

#     print("Conflict/collaboration summary:")
#     for k, v in summary.items():
#         print(f"{k}: {v:.3f}" if isinstance(v, float) and np.isfinite(v) else f"{k}: {v}")


#    #     --- Leader–Follower plot (timing-based, table-anchored) ---
#     plot_leader_follower_lfi(
#         df,
#         episode_to_plot=episode_to_plot,
#         x_axis="t",
#         dt=dt,
#         window_seconds=1.0,
#         max_lag_seconds=0.30,
#         use_goal_directed=True,
#         smooth_seconds=0.25,
#         min_table_speed=1e-3,
#         min_confidence=0.10,
#         lfi_threshold=0.05,
#     )

#     plot_obstacle_penalty_per_step(
#     df,
#     episode_to_plot=episode_to_plot,
#     x_axis="t",
#     dt=dt,
#     smooth_seconds=0.10,   # try 0.05–0.25; set 0.0 to disable
#     show_raw=True,
#     show_cumulative=False, # set True if you also want the running total
# )






#     plt.show()


# ============================================================
# Entry point
# ============================================================


def open_full_df(steps_dir: str, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Load ALL part-*.parquet from steps_dir into a single pandas DataFrame.
    """
    files = sorted(glob.glob(os.path.join(steps_dir, "part-*.parquet")))
    if not files:
        raise FileNotFoundError(f"No part-*.parquet files found in: {steps_dir}")

    dfs = [pd.read_parquet(fp, columns=list(columns) if columns is not None else None) for fp in files]
    df = pd.concat(dfs, ignore_index=True)

    # Optional: keep your data in a stable order
    if "global_step" in df.columns:
        df = df.sort_values("global_step", kind="stable").reset_index(drop=True)
    elif "episode" in df.columns and "t" in df.columns:
        df = df.sort_values(["episode", "t"], kind="stable").reset_index(drop=True)

    return df


def open_full_df_and_preview(
    steps_dir: str,
    columns: Optional[Sequence[str]] = None,
    n_head: int = 20,
    n_tail: int = 20,
    show_info: bool = True,
) -> pd.DataFrame:
    """
    Loads the full DataFrame and prints a simple preview (head/tail + optional info).
    Returns df so you can inspect it interactively.
    """
    df = open_full_df(steps_dir, columns=columns)

    if show_info:
        print("\n--- df.info() ---")
        print(df.info())

    print("\n--- df.head() ---")
    print(df.head(n_head).to_string(index=False))

    print("\n--- df.tail() ---")
    print(df.tail(n_tail).to_string(index=False))

    return df



def print_all_zero_columns(df: pd.DataFrame) -> list[str]:
    zero_cols = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            s = df[c].dropna()
            if len(s) == 0 or (s == 0).all():

                zero_cols.append(c)

    print("Columns containing ONLY zeros:")
    for c in zero_cols:
        print("  ", c)

    return zero_cols

def main(base_dir: str, load_run_subdir: str, model_name: str, **kwargs):
    steps_dir = resolve_steps_dir(base_dir, load_run_subdir, model_name)

    # Load EVERYTHING (all columns)
    df = open_full_df_and_preview(steps_dir, columns=None, n_head=30, n_tail=30, show_info=True)

    # Now you can manually inspect:
    print(df.columns)
    # print(df.describe(include="all"))
    # print(df[df["episode"] == 0].head())

    zero_cols = print_all_zero_columns(df)


if __name__ == "__main__":
    base_dir = "runs_14_jan_test"
    load_run_subdir = "20260115-102732_14_jan_longrun_600000_steps"
    model_name = "14_jan_longrun_600000_steps_final"

    # Plot configuration
    episode_to_plot = 0  # set None to analyze/plot over full run

    # Gains must match apply_human_feedback()
    gain_lin = 0.5 / 40.0
    gain_ang = 1.0 / 40.0
    deadzone_force_n = 1.0
    dt = 0.01

    main(
        base_dir=base_dir,
        load_run_subdir=load_run_subdir,
        model_name=model_name,
        episode_to_plot=episode_to_plot,
        gain_lin=gain_lin,
        gain_ang=gain_ang,
        deadzone_force_n=deadzone_force_n,
        dt=dt,
    )
