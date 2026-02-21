#!/usr/bin/env python3
from __future__ import annotations

import glob
import os
import shutil
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONFIG = {
    "root_dir": os.path.expanduser("~/catkin_ws/runs/experiment/human"),
    # Empty list = include all participants found under root_dir.
    "participant_filters": [],
    # Set to a participant label like "Participant_E" to show one force-distribution subplot.
    # Set to None to disable.
    "force_distribution_participant": "Participant_E",
    # Set to a participant label like "Participant_E" to plot v_exec, w_com, and admittance adjustment per episode.
    # Set to None to disable.
    "robot_action_participant": "Participant_E",
    # Typography/style knobs aligned with analyse_data_everyone_turns.py
    "ieee_font_size_pt": 10,
    "use_latex_text": False,
    "participants_grid_axis_label_size": 16,
    "participants_grid_tick_label_size": 15,
}


@dataclass
class RunInfo:
    participant: str
    run_id: str
    steps_dir: str


def _pretty_participant_label(raw: str) -> str:
    return str(raw).replace("_", " ").strip()


def _configure_ieee_plot_style(cfg: dict) -> None:
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
    if bool(cfg.get("use_latex_text", False)) and shutil.which("latex") is not None:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "text.latex.preamble": r"\usepackage{newtxtext}\usepackage{newtxmath}",
            }
        )


def _apply_plot_grid(ax: plt.Axes) -> None:
    ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.35)


def _discover_steps_dirs(root_dir: str, participant_filters: list[str] | None) -> list[RunInfo]:
    runs: list[RunInfo] = []
    if not os.path.isdir(root_dir):
        return runs
    allowed = set(participant_filters or [])
    for participant in sorted(os.listdir(root_dir)):
        participant_dir = os.path.join(root_dir, participant)
        if not os.path.isdir(participant_dir):
            continue
        parts = participant.split("_")
        participant_label = "_".join(parts[:2]) if len(parts) >= 2 else participant
        if allowed and participant_label not in allowed:
            continue
        pattern = os.path.join(participant_dir, "*", "parquet", "*", "steps")
        for steps_dir in sorted(glob.glob(pattern)):
            run_id = os.path.basename(os.path.dirname(steps_dir))
            runs.append(RunInfo(participant=participant_label, run_id=run_id, steps_dir=steps_dir))
    return runs


def _load_steps_df(steps_dir: str) -> pd.DataFrame:
    parts = sorted(glob.glob(os.path.join(steps_dir, "part-*.parquet")))
    if not parts:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def _episode_action_means_for_run(df: pd.DataFrame, run: RunInfo) -> list[dict]:
    if df.empty or "episode" not in df.columns:
        return []
    needed = {"act_exec_v", "act_exec_w"}
    if not needed.issubset(df.columns):
        missing = sorted(list(needed - set(df.columns)))
        print(f"Warning: missing columns in {run.run_id}: {missing}")
        return []

    rows: list[dict] = []
    for ep in np.sort(df["episode"].dropna().unique()):
        g = df[df["episode"] == ep]
        if g.empty:
            continue

        v = pd.to_numeric(g["act_exec_v"], errors="coerce").to_numpy(dtype=float)
        w = pd.to_numeric(g["act_exec_w"], errors="coerce").to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        w = w[np.isfinite(w)]
        if v.size == 0 and w.size == 0:
            continue

        rows.append(
            {
                "participant": run.participant,
                "run_id": run.run_id,
                "episode": int(ep),
                # Executed speed / yaw-rate magnitudes, averaged over the trial.
                "avg_exec_speed_mps": float(np.mean(np.abs(v))) if v.size > 0 else float("nan"),
                "avg_exec_yaw_rate_radps": float(np.mean(np.abs(w))) if w.size > 0 else float("nan"),
            }
        )
    return rows


def _episode_obstacle_flags_for_run(df: pd.DataFrame, run: RunInfo) -> list[dict]:
    if df.empty or "episode" not in df.columns:
        return []

    rows: list[dict] = []
    for ep in np.sort(pd.to_numeric(df["episode"], errors="coerce").dropna().unique()):
        g = df[pd.to_numeric(df["episode"], errors="coerce") == ep]
        if g.empty:
            continue

        reset_reason = ""
        if "reset_reason" in g.columns:
            rr = g["reset_reason"].dropna().astype(str)
            if not rr.empty:
                reset_reason = str(rr.iloc[-1]).strip()
        rr_low = reset_reason.lower()
        hit_by_reason = any(k in rr_low for k in ("obstacle", "collision", "crash", "contact"))

        hit_by_penalty = False
        min_rew_obstacle = float("nan")
        hit_steps = 0
        total_steps = int(len(g))
        sim_dt_sec = float("nan")
        hit_duration_sec = float("nan")
        if "rew_obstacle" in g.columns:
            rew_obs = pd.to_numeric(g["rew_obstacle"], errors="coerce").to_numpy(dtype=float)
            rew_obs = rew_obs[np.isfinite(rew_obs)]
            if rew_obs.size > 0:
                min_rew_obstacle = float(np.min(rew_obs))
                # From albert_table_env.py:
                # rew_obstacle = -(contact_r + impact_r + prox_r + contact_t + impact_t + prox_t)
                # and prox_r <= obs_prox_k, prox_t <= table_obs_prox_k.
                # So a sufficient contact condition is:
                #   -rew_obstacle > (obs_prox_k + table_obs_prox_k).
                obs_prox_k = 0.0
                if "obs_prox_k" in g.columns:
                    pk = pd.to_numeric(g["obs_prox_k"], errors="coerce").to_numpy(dtype=float)
                    pk = pk[np.isfinite(pk)]
                    if pk.size > 0:
                        obs_prox_k = max(0.0, float(pk[-1]))
                table_obs_prox_k = obs_prox_k
                if "table_obs_prox_k" in g.columns:
                    tpk = pd.to_numeric(g["table_obs_prox_k"], errors="coerce").to_numpy(dtype=float)
                    tpk = tpk[np.isfinite(tpk)]
                    if tpk.size > 0:
                        table_obs_prox_k = max(0.0, float(tpk[-1]))
                prox_penalty_cap = obs_prox_k + table_obs_prox_k
                rew_obs_full = pd.to_numeric(g["rew_obstacle"], errors="coerce").to_numpy(dtype=float)
                rew_obs_full = rew_obs_full[np.isfinite(rew_obs_full)]
                if rew_obs_full.size > 0:
                    # Hit if obstacle penalty magnitude exceeds what proximity alone can explain.
                    # Since penalties are negative, this is: -rew_obstacle > max_proximity_penalty.
                    contact_mask = (-rew_obs_full) > (prox_penalty_cap + 1e-6)
                    hit_steps = int(np.count_nonzero(contact_mask))
                    hit_by_penalty = bool(hit_steps > 0)

                    if "sim_dt" in g.columns:
                        dt_vals = pd.to_numeric(g["sim_dt"], errors="coerce").to_numpy(dtype=float)
                        dt_vals = dt_vals[np.isfinite(dt_vals) & (dt_vals > 0)]
                        if dt_vals.size > 0:
                            sim_dt_sec = float(np.median(dt_vals))
                    if np.isfinite(sim_dt_sec) and sim_dt_sec > 0:
                        hit_duration_sec = float(hit_steps * sim_dt_sec)

        rows.append(
            {
                "participant": run.participant,
                "run_id": run.run_id,
                "episode": int(ep),
                "obstacle_hit": bool(hit_by_reason or hit_by_penalty),
                "hit_signal": (
                    "reset_reason"
                    if hit_by_reason
                    else ("rew_obstacle_beyond_proximity" if hit_by_penalty else "none")
                ),
                "reset_reason": reset_reason,
                "min_rew_obstacle": min_rew_obstacle,
                "hit_steps": int(hit_steps),
                "total_steps": int(total_steps),
                "hit_fraction": (float(hit_steps) / float(total_steps)) if total_steps > 0 else float("nan"),
                "sim_dt_sec": sim_dt_sec,
                "hit_duration_sec": hit_duration_sec,
            }
        )
    return rows


def _plot_obstacle_hit_table(trial_df: pd.DataFrame) -> None:
    if trial_df.empty:
        return

    t = trial_df.copy()
    t["obstacle_hit"] = np.where(t["obstacle_hit"].astype(bool), "yes", "no")
    t["hit_duration_sec"] = pd.to_numeric(t["hit_duration_sec"], errors="coerce").round(3)
    t["hit_fraction"] = pd.to_numeric(t["hit_fraction"], errors="coerce").round(3)
    t = t[t["obstacle_hit"] == "yes"].copy()
    if t.empty:
        print("No obstacle-hit trials.")
        return
    t = t.sort_values(["participant", "episode"])
    t = t[
        [
            "participant",
            "episode",
            "hit_steps",
            "hit_fraction",
            "hit_duration_sec",
        ]
    ].reset_index(drop=True)
    t["participant"] = t["participant"].map(_pretty_participant_label)

    n_rows = max(len(t), 1)
    fig_h = max(4.0, 0.34 * n_rows + 1.6)
    fig, ax = plt.subplots(1, 1, figsize=(12, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=t.astype(str).values.tolist(),
        colLabels=list(t.columns),
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.25)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e9eef5")
        cell.set_edgecolor("#7a8796")

    plt.tight_layout()


def _force_rows_for_run(df: pd.DataFrame, run: RunInfo) -> list[dict]:
    needed = {"episode", "Fh_x", "Fh_y"}
    if df.empty or not needed.issubset(df.columns):
        return []

    ep = pd.to_numeric(df["episode"], errors="coerce").to_numpy(dtype=float)
    if "global_step" in df.columns:
        step = pd.to_numeric(df["global_step"], errors="coerce").to_numpy(dtype=float)
    else:
        step = np.arange(len(df), dtype=float)
    fx = pd.to_numeric(df["Fh_x"], errors="coerce").to_numpy(dtype=float)
    fy = pd.to_numeric(df["Fh_y"], errors="coerce").to_numpy(dtype=float)
    force_mag = np.sqrt(fx * fx + fy * fy)
    valid = np.isfinite(ep) & np.isfinite(step) & np.isfinite(fx) & np.isfinite(fy) & np.isfinite(force_mag)
    if np.count_nonzero(valid) == 0:
        return []

    rows: list[dict] = []
    for e, s, x, y, f in zip(
        ep[valid].astype(int),
        step[valid].astype(int),
        fx[valid],
        fy[valid],
        force_mag[valid],
    ):
        rows.append(
            {
                "participant": run.participant,
                "run_id": run.run_id,
                "episode": int(e),
                "sim_step": int(s),
                "Fh_x": float(x),
                "Fh_y": float(y),
                "human_force_mag": float(f),
            }
        )
    return rows


def _robot_action_rows_for_run(df: pd.DataFrame, run: RunInfo) -> list[dict]:
    needed = {"episode", "act_exec_v", "act_raw_w", "act_exec_w"}
    if df.empty or not needed.issubset(df.columns):
        return []

    ep = pd.to_numeric(df["episode"], errors="coerce").to_numpy(dtype=float)
    if "global_step" in df.columns:
        step = pd.to_numeric(df["global_step"], errors="coerce").to_numpy(dtype=float)
    else:
        step = np.arange(len(df), dtype=float)

    v_exec = pd.to_numeric(df["act_exec_v"], errors="coerce").to_numpy(dtype=float)
    w_com = pd.to_numeric(df["act_raw_w"], errors="coerce").to_numpy(dtype=float)
    w_exec = pd.to_numeric(df["act_exec_w"], errors="coerce").to_numpy(dtype=float)
    if "delta_w" in df.columns:
        delta_w = pd.to_numeric(df["delta_w"], errors="coerce").to_numpy(dtype=float)
    else:
        delta_w = w_exec - w_com

    valid = np.isfinite(ep) & np.isfinite(step) & np.isfinite(v_exec) & np.isfinite(w_com) & np.isfinite(delta_w)
    if np.count_nonzero(valid) == 0:
        return []

    rows: list[dict] = []
    for e, s, v, w, dw in zip(
        ep[valid].astype(int),
        step[valid].astype(int),
        v_exec[valid],
        w_com[valid],
        delta_w[valid],
    ):
        rows.append(
            {
                "participant": run.participant,
                "run_id": run.run_id,
                "episode": int(e),
                "sim_step": int(s),
                "v_exec": float(v),
                "w_com": float(w),
                "adjustment_w": float(dw),
            }
        )
    return rows


def _plot_force_per_simulation_step_per_episode(force_df: pd.DataFrame, participant: str) -> None:
    if force_df.empty:
        print("No force rows available for per-episode per-step force plot.")
        return

    sub = force_df[force_df["participant"] == participant].copy()
    if sub.empty:
        print(f"No force rows found for participant '{participant}'.")
        return

    groups: list[tuple[str, int, pd.DataFrame]] = []
    for (run_id, episode), g in sub.groupby(["run_id", "episode"], sort=True):
        groups.append((str(run_id), int(episode), g.copy()))
    if not groups:
        print(f"No per-episode force rows found for participant '{participant}'.")
        return

    n = len(groups)
    # One column keeps per-episode plots readable when there are many episodes.
    ncols = 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.8 * nrows), sharex=False, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, (run_id, episode, g) in zip(axes, groups):
        g = g.sort_values("sim_step")
        x = pd.to_numeric(g["sim_step"], errors="coerce").to_numpy(dtype=float)
        fx = pd.to_numeric(g["Fh_x"], errors="coerce").to_numpy(dtype=float)
        fy = pd.to_numeric(g["Fh_y"], errors="coerce").to_numpy(dtype=float)
        fm = pd.to_numeric(g["human_force_mag"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(fx) & np.isfinite(fy) & np.isfinite(fm)
        x = x[valid]
        fx = fx[valid]
        fy = fy[valid]
        fm = fm[valid]
        if x.size == 0:
            ax.set_visible(False)
            continue

        # Plot step index local to this episode so each episode keeps its own length.
        x_local = np.arange(x.size, dtype=int)
        ax.plot(x_local, fx, color="#1f77b4", alpha=0.45, linewidth=0.8, label="Fh_x")
        ax.plot(x_local, fy, color="#ff7f0e", alpha=0.45, linewidth=0.8, label="Fh_y")
        ax.plot(x_local, fm, color="#2ca02c", alpha=0.95, linewidth=1.1, label="|Fh|")
        ax.set_title(f"Episode {episode} (steps={x_local.size})", fontsize=10)
        ax.set_xlabel("Simulation Step (within episode)")
        ax.set_ylabel("Force")
        ax.grid(True, which="major", alpha=0.3)

    for ax in axes[len(groups):]:
        ax.set_visible(False)

    if groups:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False)
    fig.suptitle(f"{participant}: Human Force per Simulation Step (Per Episode)", y=0.995)
    fig.tight_layout(rect=[0.03, 0.02, 0.98, 0.985], h_pad=1.0)


def _plot_robot_action_signals_per_episode(action_df: pd.DataFrame, participant: str) -> None:
    if action_df.empty:
        print("No robot action rows available for per-episode action plot.")
        return

    sub = action_df[action_df["participant"] == participant].copy()
    if sub.empty:
        print(f"No robot action rows found for participant '{participant}'.")
        return

    groups: list[tuple[str, int, pd.DataFrame]] = []
    for (run_id, episode), g in sub.groupby(["run_id", "episode"], sort=True):
        groups.append((str(run_id), int(episode), g.copy()))
    if not groups:
        print(f"No per-episode action rows found for participant '{participant}'.")
        return

    n = len(groups)
    ncols = 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.8 * nrows), sharex=False, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, (run_id, episode, g) in zip(axes, groups):
        g = g.sort_values("sim_step")
        v_exec = pd.to_numeric(g["v_exec"], errors="coerce").to_numpy(dtype=float)
        w_com = pd.to_numeric(g["w_com"], errors="coerce").to_numpy(dtype=float)
        adj_w = pd.to_numeric(g["adjustment_w"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(v_exec) & np.isfinite(w_com) & np.isfinite(adj_w)
        v_exec = v_exec[valid]
        w_com = w_com[valid]
        adj_w = adj_w[valid]
        if v_exec.size == 0:
            ax.set_visible(False)
            continue

        x_local = np.arange(v_exec.size, dtype=int)
        ax.plot(x_local, v_exec, color="#1f77b4", linewidth=1.0, alpha=0.9, label="v_exec")
        ax.plot(x_local, w_com, color="#ff7f0e", linewidth=1.0, alpha=0.9, label="w_com (act_raw_w)")
        ax.plot(x_local, adj_w, color="#2ca02c", linewidth=1.0, alpha=0.95, label="adjustment_w")
        ax.set_title(f"Episode {episode} (steps={x_local.size})", fontsize=10)
        ax.set_xlabel("Simulation Step (within episode)")
        ax.set_ylabel("Signal Value")
        ax.grid(True, which="major", alpha=0.3)

    for ax in axes[len(groups):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False)
    fig.suptitle(f"{participant}: v_exec, w_com, and Human-Feedback Adjustment per Episode", y=0.995)
    fig.tight_layout(rect=[0.03, 0.02, 0.98, 0.985], h_pad=1.0)


def _plot_metric_per_trial_grid(ep_mean_df: pd.DataFrame, value_col: str, ylabel: str, title: str) -> None:
    participants = sorted(ep_mean_df["participant"].dropna().astype(str).unique())
    if not participants:
        print(f"No data for plot: {title}")
        return

    n = len(participants)
    axis_label_size = float(CONFIG.get("participants_grid_axis_label_size", 14))
    tick_label_size = float(CONFIG.get("participants_grid_tick_label_size", 13))
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, participant in zip(axes, participants):
        sub = ep_mean_df[ep_mean_df["participant"] == participant].copy()
        sub = (
            sub.groupby(["participant", "episode"], as_index=False)[value_col]
            .mean()
            .sort_values("episode")
        )
        if sub.empty:
            ax.set_visible(False)
            continue
        x = sub["episode"].to_numpy(dtype=float)
        y = sub[value_col].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
        if x.size == 0:
            ax.set_title(f"{_pretty_participant_label(participant)} (no valid data)", fontsize=tick_label_size)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="both", labelsize=tick_label_size, labelbottom=True, labelleft=True)
            _apply_plot_grid(ax)
            continue
        ax.plot(x, y, marker="o", linewidth=1.8, color="#1f77b4")
        xt = np.unique(x.astype(int))
        ax.set_xticks(xt)
        ax.set_xticklabels([str(v) for v in xt])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=tick_label_size, labelbottom=True, labelleft=True)
        ax.set_title(_pretty_participant_label(participant), fontsize=tick_label_size)
        _apply_plot_grid(ax)

    for ax in axes[len(participants):]:
        ax.set_visible(False)

    if hasattr(fig, "supxlabel"):
        fig.supxlabel("Trial (Episode)", fontsize=axis_label_size)
    else:
        fig.text(0.5, 0.02, "Trial (Episode)", ha="center", va="center", fontsize=axis_label_size)
    if hasattr(fig, "supylabel"):
        fig.supylabel(ylabel, fontsize=axis_label_size)
    else:
        fig.text(0.02, 0.5, ylabel, ha="center", va="center", rotation="vertical", fontsize=axis_label_size)
    fig.tight_layout(rect=[0.03, 0.03, 1.0, 0.97])


def _plot_force_distribution_per_episode(force_df: pd.DataFrame, participant: str) -> None:
    if force_df.empty:
        print("No force rows available for force-distribution plot.")
        return

    sub = force_df[force_df["participant"] == participant].copy()
    if sub.empty:
        print(f"No force rows found for participant '{participant}'.")
        return

    episodes = sorted(sub["episode"].dropna().astype(int).unique().tolist())
    data: list[np.ndarray] = []
    valid_episodes: list[int] = []
    for ep in episodes:
        y = pd.to_numeric(sub[sub["episode"] == ep]["human_force_mag"], errors="coerce").to_numpy(dtype=float)
        y = y[np.isfinite(y)]
        if y.size == 0:
            continue
        data.append(y)
        valid_episodes.append(ep)

    if not data:
        print(f"No finite force values for participant '{participant}'.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    ax.boxplot(
        data,
        positions=valid_episodes,
        widths=0.65,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#7fb3d5", "edgecolor": "#1f618d", "linewidth": 1.0},
        medianprops={"color": "#c0392b", "linewidth": 1.2},
        whiskerprops={"color": "#1f618d", "linewidth": 1.0},
        capprops={"color": "#1f618d", "linewidth": 1.0},
    )
    ax.set_title(f"{participant}: Human Force Magnitude Distribution per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("|Fh| = sqrt(Fh_x^2 + Fh_y^2)")
    ax.set_xticks(valid_episodes)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()


def main() -> None:
    _configure_ieee_plot_style(CONFIG)
    root_dir = CONFIG["root_dir"]
    participant_filters = list(CONFIG.get("participant_filters", []))
    runs = _discover_steps_dirs(root_dir, participant_filters)
    filter_desc = ", ".join(participant_filters) if participant_filters else "ALL"
    if not runs:
        print(f"No runs found for participant filters '{filter_desc}' under: {root_dir}")
        return

    total_steps = 0
    all_rows: list[dict] = []
    obstacle_rows: list[dict] = []
    for run in runs:
        df = _load_steps_df(run.steps_dir)
        if df.empty:
            continue
        total_steps += len(df)
        all_rows.extend(_episode_action_means_for_run(df, run))
        obstacle_rows.extend(_episode_obstacle_flags_for_run(df, run))

    if not all_rows:
        print("No action summary records produced.")
        return

    ep_mean_df = pd.DataFrame(all_rows)
    print(f"Participant filters: {filter_desc}")
    print(f"Loaded steps: {total_steps}")
    print(f"Trial summaries: {len(ep_mean_df)}")
    _plot_obstacle_hit_table(pd.DataFrame(obstacle_rows))

    _plot_metric_per_trial_grid(
        ep_mean_df,
        value_col="avg_exec_speed_mps",
        ylabel="Average Executed Speed (m/s)",
        title="Average Trial Executed Speed per Participant",
    )
    _plot_metric_per_trial_grid(
        ep_mean_df,
        value_col="avg_exec_yaw_rate_radps",
        ylabel="Average Executed Yaw Rate (rad/s)",
        title="Average Trial Executed Yaw Rate per Participant",
    )
    # Figure 3 and Figure 4 intentionally skipped: show only Figure 1 and Figure 2.
    plt.show()


if __name__ == "__main__":
    main()
