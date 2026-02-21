#!/usr/bin/env python3
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
# CONFIGURATION (mirrors analyse_data_everyone_turns.py layout)
# ============================================================
CONFIG = {
    "root_dir": os.path.expanduser("~/catkin_ws/runs/experiment/human"),
    "rolling_window": 1,
    # Obstacle-centered route classification for top/bottom passing strategy.
    "obstacle_center_xy": (-2.5, 0.0),
    "strategy_x_window_m": 1.0,
    # Typography/style knobs for manuscript-ready figures.
    "ieee_font_size_pt": 10,
    "use_latex_text": False,
    "participants_grid_show_titles": True,
    # Participant-grid legend placement: "above" or "below".
    "participants_grid_legend_position": "above",
    # Participant-grid readability controls.
    "participants_grid_axis_label_size": 14,
    "participants_grid_tick_label_size": 13,
    "participants_grid_legend_font_size": 13,
    # Optional participant-specific y-axis max overrides for readability.
    "participant_ymax_overrides": {"Participant_C": 1450.0},
}

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
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def _compute_ep_final_dist(group: pd.DataFrame) -> float:
    if "dist_table_to_goal" in group.columns:
        vals = group["dist_table_to_goal"].dropna()
        if not vals.empty:
            return float(vals.iloc[-1])
    if {"table_x", "table_y", "goal_x", "goal_y"}.issubset(group.columns):
        row = group.iloc[-1]
        dx = float(row["table_x"]) - float(row["goal_x"])
        dy = float(row["table_y"]) - float(row["goal_y"])
        return float(np.hypot(dx, dy))
    return float("nan")


def _steps_to_goal(group: pd.DataFrame, threshold: float) -> float:
    if "dist_table_to_goal" not in group.columns:
        return float("nan")
    vals = group["dist_table_to_goal"].to_numpy(dtype=float)
    if vals.size == 0 or not np.isfinite(threshold):
        return float("nan")
    idx = np.where(vals <= threshold)[0]
    if idx.size == 0:
        return float("nan")
    t_col = "t" if "t" in group.columns else None
    if t_col is None:
        return float(idx[0])
    return float(group.iloc[idx[0]][t_col])


def _aggregate_episodes(df: pd.DataFrame, run: RunInfo) -> pd.DataFrame:
    if df.empty or "episode" not in df.columns:
        return pd.DataFrame()
    rows = []
    for ep, group in df.groupby("episode"):
        group = group.sort_values("t" if "t" in group.columns else group.index)
        ep_end_global_step = float(group["global_step"].max()) if "global_step" in group.columns else float("nan")
        t_max = float(group["t"].max()) if "t" in group.columns else float(len(group) - 1)
        ep_len = t_max + 1.0

        if "goal_threshold" in group.columns:
            gt = group["goal_threshold"].dropna()
            ep_goal_threshold = float(gt.iloc[-1]) if not gt.empty else 0.0
        else:
            ep_goal_threshold = 0.0

        ep_final_dist = _compute_ep_final_dist(group)
        done_final = bool(group["done"].iloc[-1]) if "done" in group.columns else False
        ep_success = 1 if (np.isfinite(ep_final_dist) and ep_final_dist <= ep_goal_threshold and done_final) else 0

        ep_steps_to_goal = _steps_to_goal(group, ep_goal_threshold)
        ep_pass_strategy = _classify_pass_strategy(group, CONFIG)

        rows.append(
            {
                "participant": run.participant,
                "run_id": run.run_id,
                "episode": int(ep),
                "ep_end_global_step": ep_end_global_step,
                "total_ep_steps": ep_len,
                "ep_goal_threshold": ep_goal_threshold,
                "ep_final_dist": ep_final_dist,
                "ep_success": ep_success,
                "ep_steps_to_goal": ep_steps_to_goal,
                "ep_pass_strategy": ep_pass_strategy,
            }
        )
    return pd.DataFrame(rows)


def _classify_pass_strategy(group: pd.DataFrame, cfg: dict) -> str:
    """Classify obstacle passing strategy as top/bottom/unknown for one episode."""
    if not {"table_x", "table_y"}.issubset(group.columns):
        return "unknown"
    x = pd.to_numeric(group["table_x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(group["table_y"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return "unknown"
    x = x[finite]
    y = y[finite]

    obs_x = float(cfg.get("obstacle_center_xy", (-2.5, 0.0))[0])
    x_win = float(cfg.get("strategy_x_window_m", 1.0))
    near = np.abs(x - obs_x) <= x_win

    if np.any(near):
        y_ref = float(np.nanmean(y[near]))
    else:
        idx = int(np.argmin(np.abs(x - obs_x)))
        y_ref = float(y[idx])

    if y_ref > 0.0:
        return "top"
    if y_ref < 0.0:
        return "bottom"
    return "unknown"


def _plot_steps_to_goal_all_participants(ep_df: pd.DataFrame) -> None:
    participants = sorted(ep_df["participant"].unique())
    if not participants:
        print("Warning: No participants for subplot learning-curve plot.")
        return

    nrows = 3
    ncols = 2
    axis_label_size = float(CONFIG.get("participants_grid_axis_label_size", 12))
    tick_label_size = float(CONFIG.get("participants_grid_tick_label_size", 11))
    legend_font_size = float(CONFIG.get("participants_grid_legend_font_size", 11))
    y_min_fixed = 400.0
    y_tick_step = 100.0
    window = max(int(CONFIG.get("rolling_window", 5)), 1)

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 9), sharex=False, sharey=False)
    axes = np.atleast_1d(axes).ravel()

    for ax, participant in zip(axes, participants[: nrows * ncols]):
        sub = ep_df[ep_df["participant"] == participant].sort_values("episode")
        sub = sub[sub["ep_success"] == 1]
        valid_mask = (
            np.isfinite(sub["episode"].to_numpy(dtype=float))
            & np.isfinite(sub["ep_steps_to_goal"].to_numpy(dtype=float))
        )
        if not np.any(valid_mask):
            if bool(CONFIG.get("participants_grid_show_titles", False)):
                participant_label = " ".join(str(participant).replace("_", " ").split())
                ax.set_title(f"{participant_label} (no successes)", fontsize=tick_label_size)
            ax.set_xticks([])
            ax.tick_params(axis="both", labelsize=tick_label_size)
            _apply_plot_grid(ax)
            continue

        ep_labels = sub.loc[valid_mask, "episode"].to_numpy(dtype=int)
        x = np.arange(1, len(ep_labels) + 1, dtype=float)
        steps = sub.loc[valid_mask, "ep_steps_to_goal"]
        steps_np = steps.to_numpy(dtype=float)
        strategies = sub.loc[valid_mask, "ep_pass_strategy"].astype(str).to_numpy()
        # Only use rolling median when window > 1. With window=1 this is identical to raw data.
        if x.size >= 1 and window > 1:
            med = pd.Series(steps_np).rolling(window=window, min_periods=1).median().to_numpy(dtype=float)
            ax.plot(x, med, linewidth=2.0, color="#1f77b4", label=f"Rolling median (window={window})")
        else:
            ax.plot(x, steps_np, linewidth=1.8, color="#1f77b4", label="Steps to goal")
        if bool(CONFIG.get("participants_grid_show_titles", False)):
            participant_label = " ".join(str(participant).replace("_", " ").split())
            ax.set_title(participant_label, fontsize=tick_label_size)
        ax.tick_params(axis="both", labelsize=tick_label_size)
        p_vals = steps_np[np.isfinite(steps_np)]
        p_max = float(np.max(p_vals)) if p_vals.size > 0 else y_min_fixed
        p_max_rounded = max(1000.0, float(np.ceil(p_max / y_tick_step) * y_tick_step))
        p_override = CONFIG.get("participant_ymax_overrides", {})
        if isinstance(p_override, dict):
            override_val = p_override.get(str(participant))
            if override_val is not None and np.isfinite(float(override_val)):
                p_max_rounded = max(p_max_rounded, float(override_val))
        p_ticks = np.arange(y_min_fixed, p_max_rounded + 0.5 * y_tick_step, y_tick_step)
        ax.set_ylim(y_min_fixed, p_max_rounded)
        ax.set_yticks(p_ticks)
        ax.set_xlim(0.5, len(ep_labels) + 0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([str(e) for e in ep_labels])
        ax.minorticks_off()
        _apply_plot_grid(ax)
        # Show passing strategy under each episode tick (T=top, B=bottom).
        for xi, s in zip(x, strategies):
            if s == "top":
                ax.text(
                    xi,
                    -0.16,
                    "T",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=tick_label_size,
                    color="black",
                )
            elif s == "bottom":
                ax.text(
                    xi,
                    -0.16,
                    "B",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=tick_label_size,
                    color="black",
                )
            else:
                ax.text(
                    xi,
                    -0.16,
                    "?",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=tick_label_size,
                    color="black",
                )

    for ax in axes[len(participants[: nrows * ncols]) :]:
        ax.set_visible(False)

    legend_pos = str(CONFIG.get("participants_grid_legend_position", "above")).strip().lower()
    legend_label = f"Rolling median (window={window})" if window > 1 else "Steps to goal"
    legend_items = [Line2D([0], [0], color="#1f77b4", lw=2.0, label=legend_label)]
    legend = fig.legend(
        handles=legend_items,
        loc="lower center" if legend_pos == "below" else "upper center",
        ncol=1,
        bbox_to_anchor=(0.5, -0.01 if legend_pos == "below" else 0.99),
        fontsize=legend_font_size,
        title="T = Top pass, B = Bottom pass",
        title_fontsize=legend_font_size,
    )
    legend.get_title().set_fontweight("normal")
    if hasattr(fig, "supxlabel"):
        fig.supxlabel("Episode", fontsize=axis_label_size)
    else:
        fig.text(0.5, 0.02, "Episode", ha="center", va="center", fontsize=axis_label_size)
    if hasattr(fig, "supylabel"):
        fig.supylabel("Steps to Goal", fontsize=axis_label_size)
    else:
        fig.text(0.02, 0.5, "Steps to Goal", ha="center", va="center", rotation="vertical", fontsize=axis_label_size)
    if bool(CONFIG.get("participants_grid_show_titles", False)):
        fig.suptitle("Steps-to-Goal by Episode (Successful Episodes)", y=1.03)
    if legend_pos == "below":
        plt.tight_layout(rect=[0.05, 0.12, 1.0, 1.0])
    else:
        plt.tight_layout(rect=[0.05, 0.10, 1.0, 0.94])


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

    all_eps: list[pd.DataFrame] = []
    total_steps = 0
    for run in runs:
        df = _load_steps_df(run.steps_dir)
        if df.empty:
            continue
        total_steps += len(df)
        ep_df = _aggregate_episodes(df, run)
        if not ep_df.empty:
            all_eps.append(ep_df)

    if not all_eps:
        print("No episode records found.")
        return

    ep_df = pd.concat(all_eps, ignore_index=True)
    success_rate = float(ep_df["ep_success"].mean()) if "ep_success" in ep_df.columns else float("nan")
    print(f"Loaded steps: {total_steps}")
    print(f"Episodes: {len(ep_df)}")
    if np.isfinite(success_rate):
        print(f"Overall success rate: {success_rate:.3f}")

    _plot_steps_to_goal_all_participants(ep_df)
    plt.show()


if __name__ == "__main__":
    main()
