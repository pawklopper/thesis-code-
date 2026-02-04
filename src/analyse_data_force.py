#!/usr/bin/env python3
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    "base_dir": "Participant_A_Dimi",          # experiment root folder under runs/experiment/human
    "subdir": "20260204-130020_experiment_28_jan_20000",  # specific run subfolder
    "model": "experiment_28_jan_20000_final",   # model tag for run naming
    "rolling_window_steps": 10,                 # rolling average window in steps
}

# ============================================================
# Utilities (mirrors analyse_data_turns.py structure)
# ============================================================

def resolve_steps_dir(cfg):
    root_parent = os.path.join("runs", "experiment", "human", cfg["base_dir"])
    run_dir = os.path.join(root_parent, f"{cfg['subdir']}_{cfg['model']}")
    if not os.path.isdir(run_dir):
        run_dir = os.path.join(root_parent, cfg["subdir"])
    pq = os.path.join(run_dir, "parquet")
    sub = [os.path.join(pq, d) for d in os.listdir(pq) if os.path.isdir(os.path.join(pq, d))]
    return os.path.join(max(sub, key=lambda d: os.path.getmtime(d)), "steps")


def _load_steps_dataframe(steps_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(steps_dir, "part-*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {steps_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files])
    df = df.sort_values(["episode", "t"]).reset_index(drop=True)
    return df


def _compute_human_force_magnitude(df: pd.DataFrame) -> np.ndarray:
    if "Fh_x" not in df.columns or "Fh_y" not in df.columns:
        raise KeyError("Missing Fh_x or Fh_y columns (required for human force magnitude).")
    fh_mag = np.linalg.norm(df[["Fh_x", "Fh_y"]].to_numpy(dtype=float), axis=1)
    fh_mag = np.where(np.isfinite(fh_mag), fh_mag, np.nan)
    return fh_mag


def main():
    try:
        steps_dir = resolve_steps_dir(CONFIG)
        df = _load_steps_dataframe(steps_dir)
    except Exception as e:
        print(f"Error: {e}")
        return

    fh_mag = _compute_human_force_magnitude(df)
    df = df.copy()
    df["fh_mag"] = fh_mag

    episodes = np.sort(df["episode"].unique())
    if episodes.size == 0:
        print("No episodes found in the parquet data.")
        return

    window = int(CONFIG["rolling_window_steps"])
    if window <= 0:
        window = 1

    if episodes.size <= 6:
        fig, axes = plt.subplots(int(episodes.size), 1, figsize=(12, 3 * int(episodes.size)))
        if episodes.size == 1:
            axes = [axes]
        for ax, ep in zip(axes, episodes):
            ep_data = df[df["episode"] == ep]
            steps = ep_data["t"].to_numpy()
            roll = pd.Series(ep_data["fh_mag"]).rolling(window, min_periods=1).mean().to_numpy()
            ax.plot(steps, roll, color="#1f77b4", linewidth=1.5)
            ax.set_ylabel("|Fh| (N)")
            ax.set_title(f"Episode {int(ep)}")
            ax.grid(True, alpha=0.2)
        axes[-1].set_xlabel("Step")
    else:
        plt.figure(figsize=(12, 6))
        for ep in episodes:
            ep_data = df[df["episode"] == ep]
            steps = ep_data["t"].to_numpy()
            roll = pd.Series(ep_data["fh_mag"]).rolling(window, min_periods=1).mean().to_numpy()
            plt.plot(steps, roll, linewidth=1.0, alpha=0.6, label=f"Ep {int(ep)}")
        plt.xlabel("Step")
        plt.ylabel("|Fh| (N)")
        plt.title("Rolling Average Human Force per Episode")
        plt.grid(True, alpha=0.2)
        plt.legend(ncol=3, fontsize=8, loc="upper right")

    plt.suptitle(f"Rolling Avg Human Force (window={window} steps)", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()
