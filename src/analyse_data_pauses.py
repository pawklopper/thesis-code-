#!/usr/bin/env python3
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONFIG = {
    "root_dir": os.path.expanduser("~/catkin_ws/runs/experiment/human"),
    "participant_filters": ["Participant_A", "Participant_E"],
    # Pause = low commanded speed magnitude |act_exec_v| (m/s).
    "pause_speed_threshold_mps": 0.1,
    "min_pause_steps": 4,
}


@dataclass
class RunInfo:
    participant: str
    run_id: str
    steps_dir: str


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
            run_id = os.path.basename(os.path.dirname(os.path.dirname(steps_dir)))
            runs.append(RunInfo(participant=participant_label, run_id=run_id, steps_dir=steps_dir))
    return runs


def _load_steps_df(steps_dir: str) -> pd.DataFrame:
    parts = sorted(glob.glob(os.path.join(steps_dir, "part-*.parquet")))
    if not parts:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def _contiguous_true_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    if mask.size == 0:
        return []
    m = mask.astype(np.uint8)
    d = np.diff(np.r_[0, m, 0])
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _pause_records_for_run(df: pd.DataFrame, run: RunInfo, cfg: dict) -> tuple[list[dict], list[dict], list[dict]]:
    if df.empty or "episode" not in df.columns:
        return [], [], []
    needed = {"act_exec_v"}
    if not needed.issubset(df.columns):
        missing = sorted(list(needed - set(df.columns)))
        print(f"Warning: missing columns for pause detection in {run.run_id}: {missing}")
        return [], [], []

    episode_rows: list[dict] = []
    pause_rows: list[dict] = []
    step_rows: list[dict] = []
    speed_thr = float(cfg["pause_speed_threshold_mps"])

    for ep in np.sort(df["episode"].unique()):
        g = df[df["episode"] == ep].sort_values("t" if "t" in df.columns else df.index)
        if g.empty:
            continue

        cmd_v = g["act_exec_v"].to_numpy(dtype=float)
        cmd_abs = np.abs(cmd_v)
        valid = np.isfinite(cmd_abs)
        pause_mask = valid & (cmd_abs <= speed_thr)

        segs = _contiguous_true_segments(pause_mask)
        min_steps = int(cfg["min_pause_steps"])
        segs = [(s, e) for s, e in segs if (e - s + 1) >= min_steps]

        t_vals = g["t"].to_numpy(dtype=float) if "t" in g.columns else np.arange(len(g), dtype=float)
        if "sim_dt" in g.columns and np.isfinite(g["sim_dt"].to_numpy(dtype=float)).any():
            dt = float(pd.to_numeric(g["sim_dt"], errors="coerce").dropna().iloc[-1])
            if not np.isfinite(dt) or dt <= 0:
                dt = 1.0
        else:
            dt = 1.0

        total_steps = int(len(g))
        total_pause_steps = int(np.sum([(e - s + 1) for s, e in segs]))
        pause_fraction = float(total_pause_steps / total_steps) if total_steps > 0 else 0.0

        episode_rows.append(
            {
                "participant": run.participant,
                "run_id": run.run_id,
                "episode": int(ep),
                "ep_len_steps": total_steps,
                "pause_count": int(len(segs)),
                "pause_steps_total": total_pause_steps,
                "pause_sec_total": float(total_pause_steps * dt),
                "pause_fraction": pause_fraction,
            }
        )

        for pause_idx, (s, e) in enumerate(segs, start=1):
            start_t = float(t_vals[s]) if s < t_vals.size else float(s)
            steps = int(e - s + 1)
            pause_rows.append(
                {
                    "participant": run.participant,
                    "run_id": run.run_id,
                    "episode": int(ep),
                    "pause_id": pause_idx,
                    "start_t": start_t,
                    "duration_steps": steps,
                    "duration_sec": float(steps * dt),
                }
            )

        for tt, vv in zip(t_vals, cmd_v):
            if not (np.isfinite(tt) and np.isfinite(vv)):
                continue
            step_rows.append(
                {
                    "participant": run.participant,
                    "run_id": run.run_id,
                    "episode": int(ep),
                    "t": float(tt),
                    "act_exec_v": float(vv),
                }
            )

    return episode_rows, pause_rows, step_rows


def _plot_pause_durations_by_episode(pause_df: pd.DataFrame, participant: str) -> None:
    sub = pause_df[pause_df["participant"] == participant]
    if sub.empty:
        print(f"No pause segments to plot for {participant}.")
        return
    sub = sub.sort_values(["episode", "start_t"]).reset_index(drop=True)
    x = sub["episode"].to_numpy(dtype=float)
    y_steps = sub["duration_steps"].to_numpy(dtype=float)
    y_sec = sub["duration_sec"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y_steps) & np.isfinite(y_sec)
    x = x[valid]
    y_steps = y_steps[valid]
    y_sec = y_sec[valid]
    if x.size == 0:
        print(f"No valid pause duration points for {participant}.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.scatter(x, y_steps, s=45, alpha=0.8, color="#1f77b4", label="Pause segment")

    episode_vals = np.unique(x.astype(int))
    ax.set_xticks(episode_vals)
    ax.set_xlabel("Episode Number")
    ax.set_ylabel("Pause Duration (steps)")
    ax.set_title(f"Pause Segment Duration by Episode (from |act_exec_v|) - {participant}")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()


def _plot_pause_count_per_trial(episode_df: pd.DataFrame) -> None:
    participants = sorted(episode_df["participant"].dropna().astype(str).unique())
    if not participants:
        print("No participant pause-count data to plot.")
        return

    n = len(participants)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows), sharex=False, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, participant in zip(axes, participants):
        sub = episode_df[episode_df["participant"] == participant].sort_values("episode")
        if sub.empty:
            ax.set_visible(False)
            continue
        x = sub["episode"].to_numpy(dtype=float)
        y = sub["pause_count"].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
        if x.size == 0:
            ax.set_title(f"{participant} (no valid data)")
            ax.grid(True, which="major", alpha=0.3)
            continue
        ax.plot(x, y, color="#1f77b4", marker="o", linewidth=1.8)
        ax.set_xticks(np.unique(x.astype(int)))
        ax.set_xlabel("Trial (Episode)")
        ax.set_ylabel("Pause Count")
        ax.set_title(participant)
        ax.grid(True, which="major", alpha=0.3)

    for ax in axes[len(participants):]:
        ax.set_visible(False)

    fig.suptitle("Number of Pauses per Trial (Episode) by Participant", y=1.01)
    plt.tight_layout()


def _plot_act_exec_v_full_by_trial(step_df: pd.DataFrame, pause_df: pd.DataFrame, participant: str) -> None:
    sub = step_df[step_df["participant"] == participant].sort_values(["episode", "t"])
    if sub.empty:
        print(f"No per-step act_exec_v data for {participant}.")
        return
    pause_sub = pause_df[pause_df["participant"] == participant].sort_values(["episode", "start_t"])
    episodes = sorted(sub["episode"].unique())
    n = len(episodes)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows), sharex=False, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, ep in zip(axes, episodes):
        g = sub[sub["episode"] == ep]
        x = g["t"].to_numpy(dtype=float)
        y = g["act_exec_v"].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
        if x.size == 0:
            ax.set_title(f"Episode {ep} (no valid data)")
            ax.grid(True, which="major", alpha=0.3)
            continue
        ax.plot(x, y, color="#1f77b4", linewidth=1.2)

        p_ep = pause_sub[pause_sub["episode"] == ep]
        for _, row in p_ep.iterrows():
            start = float(row["start_t"])
            dur = float(row["duration_steps"])
            if np.isfinite(start) and np.isfinite(dur) and dur > 0:
                ax.axvspan(start, start + dur, color="#ff7f0e", alpha=0.20)

        ax.set_title(f"Episode {ep}")
        ax.set_xlabel("Step t")
        ax.set_ylabel("act_exec_v (m/s)")
        ax.grid(True, which="major", alpha=0.3)

    for ax in axes[len(episodes):]:
        ax.set_visible(False)

    fig.suptitle(f"act_exec_v over All Steps per Trial (pause windows shaded) - {participant}", y=1.01)
    plt.tight_layout()


def main() -> None:
    root_dir = CONFIG["root_dir"]
    participant_filters = list(CONFIG.get("participant_filters", []))
    runs = _discover_steps_dirs(root_dir, participant_filters)
    filter_desc = ", ".join(participant_filters) if participant_filters else "ALL"
    if not runs:
        print(f"No runs found for participant filters '{filter_desc}' under: {root_dir}")
        return

    all_episode_rows: list[dict] = []
    all_pause_rows: list[dict] = []
    all_step_rows: list[dict] = []
    total_steps = 0
    for run in runs:
        df = _load_steps_df(run.steps_dir)
        if df.empty:
            continue
        total_steps += len(df)
        ep_rows, pause_rows, step_rows = _pause_records_for_run(df, run, CONFIG)
        all_episode_rows.extend(ep_rows)
        all_pause_rows.extend(pause_rows)
        all_step_rows.extend(step_rows)

    if not all_episode_rows:
        print("No pause records produced.")
        return

    episode_df = pd.DataFrame(all_episode_rows)
    pause_df = pd.DataFrame(all_pause_rows)
    step_df = pd.DataFrame(all_step_rows)
    print(f"Participant filters: {filter_desc}")
    print(f"Loaded steps: {total_steps}")
    print(f"Episodes: {len(episode_df)}")
    print(f"Detected pauses: {len(pause_df)}")
    print(
        "Threshold setup: "
        f"pause if |act_exec_v| <= {CONFIG['pause_speed_threshold_mps']:.3f} m/s "
        f"for at least {CONFIG['min_pause_steps']} step(s)"
    )
    if "pause_fraction" in episode_df.columns and len(episode_df) > 0:
        print(f"Mean pause fraction: {episode_df['pause_fraction'].mean():.3f}")

    _plot_pause_count_per_trial(episode_df)
    selected_participants = sorted(episode_df["participant"].dropna().astype(str).unique())
    for participant in selected_participants:
        _plot_pause_durations_by_episode(pause_df, participant)
        _plot_act_exec_v_full_by_trial(step_df, pause_df, participant)
    plt.show()


if __name__ == "__main__":
    main()
