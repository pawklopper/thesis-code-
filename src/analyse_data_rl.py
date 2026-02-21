#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import shutil
from dataclasses import dataclass

import matplotlib


def _can_use_gui_backend() -> bool:
    disp = os.environ.get("DISPLAY", "").strip()
    if not disp:
        return False
    try:
        import tkinter as tk  # pylint: disable=import-outside-toplevel

        root = tk.Tk()
        root.withdraw()
        root.update_idletasks()
        root.destroy()
        return True
    except Exception:
        return False


GUI_AVAILABLE = _can_use_gui_backend()
if not GUI_AVAILABLE:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing TensorBoard parser dependency. Install with: pip install tensorboard\n"
        f"Import error: {exc}"
    )


DEFAULT_ROOT = os.path.expanduser("~/catkin_ws/runs/experiment/human")
DEFAULT_OUT_SUBDIR = "rl_analysis_exports"
PLOT_EXPECTED_EPISODES = 20
PLOT_STYLE_CFG = {
    "ieee_font_size_pt": 10,
    "use_latex_text": False,
    "participants_grid_axis_label_size": 14,
    "participants_grid_tick_label_size": 13,
    "participants_grid_legend_font_size": 13,
}


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


@dataclass
class RunInfo:
    participant: str
    participant_dir: str
    run_id: str
    run_dir: str
    event_files: list[str]


def normalize_participant_name(participant_dir_name: str) -> str:
    parts = participant_dir_name.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else participant_dir_name


def discover_runs(root_dir: str) -> list[RunInfo]:
    runs: list[RunInfo] = []
    if not os.path.isdir(root_dir):
        return runs

    for participant_dir_name in sorted(os.listdir(root_dir)):
        participant_dir = os.path.join(root_dir, participant_dir_name)
        if not os.path.isdir(participant_dir):
            continue
        participant = normalize_participant_name(participant_dir_name)

        for run_id in sorted(os.listdir(participant_dir)):
            run_dir = os.path.join(participant_dir, run_id)
            if not os.path.isdir(run_dir):
                continue
            tb_dir = os.path.join(run_dir, "tb_logs")
            if not os.path.isdir(tb_dir):
                continue
            event_files = sorted(glob.glob(os.path.join(tb_dir, "events.out.tfevents.*")))
            if not event_files:
                continue
            runs.append(
                RunInfo(
                    participant=participant,
                    participant_dir=participant_dir_name,
                    run_id=run_id,
                    run_dir=run_dir,
                    event_files=event_files,
                )
            )
    return runs


def _load_event_accumulator(path: str) -> EventAccumulator:
    ea = EventAccumulator(path, size_guidance={"scalars": 0})
    ea.Reload()
    return ea


def list_all_scalar_tags(runs: list[RunInfo]) -> list[str]:
    tags: set[str] = set()
    for run in runs:
        for event_file in run.event_files:
            try:
                ea = _load_event_accumulator(event_file)
            except Exception:
                continue
            for tag in ea.Tags().get("scalars", []):
                tags.add(tag)
    return sorted(tags)


def collect_scalars(runs: list[RunInfo], include_tags: set[str] | None = None) -> pd.DataFrame:
    rows: list[dict] = []
    for run in runs:
        for event_file in run.event_files:
            try:
                ea = _load_event_accumulator(event_file)
            except Exception as exc:
                print(f"[WARN] Could not read {event_file}: {exc}")
                continue

            for tag in ea.Tags().get("scalars", []):
                if include_tags and tag not in include_tags:
                    continue
                for event in ea.Scalars(tag):
                    rows.append(
                        {
                            "participant": run.participant,
                            "participant_dir": run.participant_dir,
                            "run_id": run.run_id,
                            "event_file": os.path.basename(event_file),
                            "tag": tag,
                            "step": int(event.step),
                            "wall_time": float(event.wall_time),
                            "value": float(event.value),
                        }
                    )

    if not rows:
        return pd.DataFrame(
            columns=[
                "participant",
                "participant_dir",
                "run_id",
                "event_file",
                "tag",
                "step",
                "wall_time",
                "value",
            ]
        )
    return pd.DataFrame(rows).sort_values(["participant", "run_id", "tag", "step"]).reset_index(drop=True)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby(["participant", "run_id", "tag"], as_index=False)
    summary = grouped.agg(
        n_points=("value", "size"),
        step_min=("step", "min"),
        step_max=("step", "max"),
        value_min=("value", "min"),
        value_max=("value", "max"),
        value_mean=("value", "mean"),
    )

    last_idx = grouped["step"].idxmax()["step"].to_numpy()
    last_vals = (
        df.loc[last_idx, ["participant", "run_id", "tag", "value"]]
        .rename(columns={"value": "value_last"})
        .reset_index(drop=True)
    )
    summary = summary.merge(last_vals, on=["participant", "run_id", "tag"], how="left")
    return summary.sort_values(["participant", "run_id", "tag"]).reset_index(drop=True)


def build_wide(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    wide = (
        df.pivot_table(
            index=["participant", "run_id", "step"],
            columns="tag",
            values="value",
            aggfunc="last",
        )
        .sort_index()
        .reset_index()
    )
    wide.columns.name = None
    return wide


def _sanitize_filename(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in s)


def _ema(y: np.ndarray, alpha: float) -> np.ndarray:
    if y.size == 0:
        return y
    out = np.empty_like(y, dtype=float)
    out[0] = float(y[0])
    for i in range(1, y.size):
        out[i] = alpha * float(y[i]) + (1.0 - alpha) * out[i - 1]
    return out


def plot_tensorboard_like_scalars(df: pd.DataFrame, out_dir: str, smoothing: float = 0.6, show: bool = False) -> list[str]:
    if df.empty:
        return []

    os.makedirs(out_dir, exist_ok=True)
    created_paths: list[str] = []
    smoothing = float(np.clip(smoothing, 0.0, 0.999))
    alpha = 1.0 - smoothing

    participants = sorted(df["participant"].unique())
    for participant in participants:
        df_p = df[df["participant"] == participant]
        for run_id in sorted(df_p["run_id"].unique()):
            sub = df_p[df_p["run_id"] == run_id]
            tags = sorted(sub["tag"].unique())
            if not tags:
                continue

            n = len(tags)
            ncols = 3
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.2 * nrows), squeeze=False)
            axes_flat = axes.ravel()

            for ax, tag in zip(axes_flat, tags):
                t = sub[sub["tag"] == tag].sort_values("step")
                x = t["step"].to_numpy(dtype=float)
                y = t["value"].to_numpy(dtype=float)
                y_smooth = _ema(y, alpha=alpha)

                ax.plot(x, y, color="#8fa8ff", linewidth=1.0, alpha=0.45, label="raw")
                ax.plot(x, y_smooth, color="#1f4ed8", linewidth=2.0, label=f"smooth ({smoothing:.2f})")
                ax.set_title(tag, fontsize=10)
                ax.set_xlabel("step")
                ax.set_ylabel("value")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8, frameon=False, loc="best")

            for ax in axes_flat[len(tags):]:
                ax.set_visible(False)

            fig.suptitle(f"TensorBoard-style scalars | {participant} | {run_id}", fontsize=12, y=1.01)
            fig.tight_layout()

            plot_name = f"tb_scalars_{_sanitize_filename(participant)}_{_sanitize_filename(run_id)}.png"
            plot_path = os.path.join(out_dir, plot_name)
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            created_paths.append(plot_path)

            if show:
                plt.show()
            plt.close(fig)

    return created_paths


def _latest_run_df_for_participant(df: pd.DataFrame, participant: str) -> pd.DataFrame:
    """Pick the latest run for a participant by max wall_time."""
    sub = df[df["participant"] == participant]
    if sub.empty:
        return sub
    run_order = (
        sub.groupby("run_id", as_index=False)["wall_time"]
        .max()
        .sort_values("wall_time")
    )
    latest_run_id = str(run_order.iloc[-1]["run_id"])
    return sub[sub["run_id"] == latest_run_id].copy()


def _latest_runinfo_by_participant(runs: list[RunInfo], df: pd.DataFrame) -> dict[str, RunInfo]:
    out: dict[str, RunInfo] = {}
    for participant in sorted(df["participant"].unique().tolist()):
        run_df = _latest_run_df_for_participant(df, participant)
        if run_df.empty:
            continue
        run_id = str(run_df["run_id"].iloc[0])
        for r in runs:
            if r.participant == participant and r.run_id == run_id:
                out[participant] = r
                break
    return out


def _load_steps_parquet(run_dir: str) -> pd.DataFrame:
    pq_root = os.path.join(run_dir, "parquet")
    if not os.path.isdir(pq_root):
        return pd.DataFrame()
    run_dirs = sorted(d for d in glob.glob(os.path.join(pq_root, "*")) if os.path.isdir(d))
    if not run_dirs:
        return pd.DataFrame()
    steps_dir = os.path.join(run_dirs[-1], "steps")
    parts = sorted(glob.glob(os.path.join(steps_dir, "part-*.parquet")))
    if not parts:
        return pd.DataFrame()
    try:
        return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    except Exception:
        return pd.DataFrame()


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
) -> str:
    # Matches analyse_data_turns/everyone_turns defaults.
    obstacle_center_xy = (-2.5, 0.0)
    obstacle_size_xy = (0.5, 1.0)
    obstacle_margin_m = 1.0
    goal_rect_size_xy = (2.5, 3.0)

    obs_size_expanded = (
        obstacle_size_xy[0] + 2.0 * obstacle_margin_m,
        obstacle_size_xy[1] + 2.0 * obstacle_margin_m,
    )
    if _in_axis_aligned_rect(table_x, table_y, obstacle_center_xy, obs_size_expanded):
        return "Obstacle"

    if goal_x is not None and goal_y is not None and np.isfinite(goal_x) and np.isfinite(goal_y):
        if _in_axis_aligned_rect(table_x, table_y, (float(goal_x), float(goal_y)), goal_rect_size_xy):
            return "Goal"
    return "Open"


def _add_zone_background(
    ax: plt.Axes,
    p_centers: np.ndarray,
    open_frac: np.ndarray,
    obs_frac: np.ndarray,
    goal_frac: np.ndarray,
) -> None:
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
            # Match analyse_data_everyone_turns.py exactly.
            alpha = 0.22 * float(np.clip(frac, 0.0, 1.0))
            ax.axvspan(x0, x1, color=color, alpha=alpha, lw=0, zorder=0)


def _add_zone_background_dominant(
    ax: plt.Axes,
    p_centers: np.ndarray,
    open_frac: np.ndarray,
    obs_frac: np.ndarray,
    goal_frac: np.ndarray,
) -> None:
    """
    Cleaner background: one dominant zone per progress bin, merged into
    contiguous spans. This avoids thin mixed stripes.
    """
    if p_centers.size < 2:
        return
    dp = float(np.nanmedian(np.diff(p_centers)))
    if not np.isfinite(dp) or dp <= 0:
        return

    frac_mat = np.vstack([open_frac, obs_frac, goal_frac]).T  # [n,3]
    valid_row = np.any(np.isfinite(frac_mat), axis=1)
    dominant = np.full(len(p_centers), -1, dtype=int)
    if np.any(valid_row):
        safe = np.where(np.isfinite(frac_mat), frac_mat, -1.0)
        dominant[valid_row] = np.argmax(safe[valid_row], axis=1)

    # Smooth bin-wise label jitter with a small majority filter.
    # This preserves large transitions but removes tiny color flicker.
    win = 7
    half = win // 2
    dominant_s = dominant.copy()
    for i in range(len(dominant)):
        lo = max(0, i - half)
        hi = min(len(dominant), i + half + 1)
        vals = dominant[lo:hi]
        vals = vals[vals >= 0]
        if vals.size == 0:
            continue
        counts = np.bincount(vals, minlength=3)
        dominant_s[i] = int(np.argmax(counts))
    dominant = dominant_s

    colors = {0: "#1f77b4", 1: "#d97706", 2: "#2a9d8f"}
    i = 0
    n = len(dominant)
    while i < n:
        if dominant[i] < 0:
            i += 1
            continue
        z = dominant[i]
        j = i + 1
        while j < n and dominant[j] == z:
            j += 1
        x0 = float(max(0.0, p_centers[i] - 0.5 * dp))
        x1 = float(min(1.0, p_centers[j - 1] + 0.5 * dp))
        ax.axvspan(x0, x1, color=colors[z], alpha=0.22, lw=0, zorder=0)
        i = j


def _compute_zone_occupancy_curves_from_progress(
    progress: np.ndarray,
    zones: np.ndarray,
    n_bins: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Equivalent to analyse_data_everyone_turns._compute_zone_occupancy_curves."""
    if progress.size == 0 or zones.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    p = np.asarray(progress, dtype=float)
    z = np.asarray(zones, dtype=object).astype(str)
    valid = np.isfinite(p)
    p = np.clip(p[valid], 0.0, 1.0)
    z = z[valid]
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
        zbin = z[mask]
        open_frac[k] = float(np.mean(zbin == "Open"))
        obs_frac[k] = float(np.mean(zbin == "Obstacle"))
        goal_frac[k] = float(np.mean(zbin == "Goal"))
    return p_centers, open_frac, obs_frac, goal_frac


def _load_parquet_input(parquet_path: str) -> pd.DataFrame:
    """
    Load parquet data from:
      - a single parquet file, or
      - a directory containing parquet files (recursively).
    """
    path = os.path.expanduser(parquet_path)
    if os.path.isfile(path):
        if not path.endswith(".parquet"):
            raise ValueError(f"--parquet points to a file that is not .parquet: {path}")
        return pd.read_parquet(path)
    if os.path.isdir(path):
        parts = sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))
        if not parts:
            raise ValueError(f"No parquet files found under directory: {path}")
        return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    raise ValueError(f"--parquet path does not exist: {path}")


def _resolve_action_columns(
    cols: list[str],
    step_col: str | None = None,
    v_col: str | None = None,
    w_col: str | None = None,
) -> tuple[str, str, str]:
    cset = set(cols)

    def _pick(preferred: list[str], provided: str | None, arg_name: str) -> str:
        if provided:
            if provided not in cset:
                raise ValueError(f"{arg_name}='{provided}' not found in parquet columns.")
            return provided
        for c in preferred:
            if c in cset:
                return c
        raise ValueError(
            f"Could not infer {arg_name}. Provide it explicitly. Available columns: {', '.join(cols)}"
        )

    step = _pick(["global_step", "t"], step_col, "--step-col")
    v = _pick(["act_exec_v", "act_raw_v"], v_col, "--v-col")
    w = _pick(["act_exec_w", "act_raw_w"], w_col, "--w-col")
    return step, v, w


def _phase_names(n_phases: int) -> list[str]:
    base = ["Early", "Mid", "Late"]
    if n_phases <= 3:
        return base[:n_phases]
    out = ["Early"]
    for i in range(1, n_phases - 1):
        out.append(f"Mid-{i}")
    out.append("Late")
    return out


def _mass_contour_threshold(hist2d: np.ndarray, mass: float = 0.68) -> float | None:
    vals = hist2d[np.isfinite(hist2d)].ravel()
    vals = vals[vals > 0]
    if vals.size == 0:
        return None
    s = np.sort(vals)[::-1]
    csum = np.cumsum(s)
    target = mass * float(np.sum(s))
    idx = int(np.searchsorted(csum, target, side="left"))
    idx = min(max(idx, 0), len(s) - 1)
    thr = float(s[idx])
    return thr if thr > 0 else None


def create_action_distribution_over_training_figure(
    parquet_path: str,
    out_base: str,
    step_col: str | None = None,
    v_col: str | None = None,
    w_col: str | None = None,
    bins: int = 48,
    smooth: int = 101,
    phases: int = 3,
    dpi: int = 180,
    robust_limits: bool = False,
    limit_quantile: float = 0.01,
    save: bool = False,
    show: bool = True,
) -> tuple[str | None, str | None]:
    """
    Build and save a publication-friendly multi-panel action-distribution figure.
    Saves both PNG and PDF.
    """
    if phases < 2:
        raise ValueError("--phases must be >= 2")
    if bins < 8:
        raise ValueError("--bins must be >= 8")
    if smooth < 1:
        smooth = 1
    if smooth % 2 == 0:
        smooth += 1  # odd window for centered rolling

    d = _load_parquet_input(parquet_path)
    step_name, v_name, w_name = _resolve_action_columns(
        list(d.columns), step_col=step_col, v_col=v_col, w_col=w_col
    )

    s = d[[step_name, v_name, w_name]].dropna().copy()
    s = s.sort_values(step_name).drop_duplicates(subset=[step_name], keep="last")
    if len(s) < 50:
        raise ValueError(f"Not enough valid rows after cleaning ({len(s)}). Need at least 50.")

    s = s.rename(columns={step_name: "step", v_name: "v", w_name: "w"}).reset_index(drop=True)
    s["step"] = s["step"].astype(float)
    s["v"] = s["v"].astype(float)
    s["w"] = s["w"].astype(float)
    s["mag"] = np.sqrt(s["v"] * s["v"] + s["w"] * s["w"])

    step_min, step_max = float(s["step"].min()), float(s["step"].max())
    denom = (step_max - step_min) if step_max > step_min else 1.0
    s["progress"] = (s["step"] - step_min) / denom

    # Robust optional axis limits.
    if robust_limits:
        q = float(np.clip(limit_quantile, 0.0, 0.2))
        vx0, vx1 = np.nanquantile(s["v"], [q, 1 - q])
        wy0, wy1 = np.nanquantile(s["w"], [q, 1 - q])
    else:
        vx0, vx1 = float(np.nanmin(s["v"])), float(np.nanmax(s["v"]))
        wy0, wy1 = float(np.nanmin(s["w"])), float(np.nanmax(s["w"]))
    if vx1 <= vx0:
        vx0, vx1 = vx0 - 1.0, vx1 + 1.0
    if wy1 <= wy0:
        wy0, wy1 = wy0 - 1.0, wy1 + 1.0
    vpad = 0.04 * (vx1 - vx0)
    wpad = 0.04 * (wy1 - wy0)
    xlim = (vx0 - vpad, vx1 + vpad)
    ylim = (wy0 - wpad, wy1 + wpad)

    # Discrete phase split by step quantiles.
    qbins = np.linspace(0.0, 1.0, phases + 1)
    step_edges = np.quantile(s["step"].to_numpy(dtype=float), qbins)
    # Ensure strictly increasing edges for digitize.
    step_edges = np.unique(step_edges)
    if len(step_edges) <= 2:
        # fallback to equal-count by index
        phase_id = (np.arange(len(s)) * phases) // len(s)
        phase_id = np.minimum(phase_id, phases - 1)
    else:
        phase_id = np.digitize(s["step"].to_numpy(dtype=float), bins=step_edges[1:-1], right=True)
    s["phase"] = phase_id.astype(int)
    phase_labels = _phase_names(phases)
    line_styles = ["solid", "dashed", "dotted", "dashdot"]
    phase_colors = ["#1d4ed8", "#059669", "#b45309", "#7c3aed", "#be123c"]

    # Figure styling
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.titlesize": 13,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.2), squeeze=False)
    ax_mag, ax_vw, ax_vdist, ax_wdist = axes.ravel()

    # Panel 1: magnitude vs progress with smoothing + quantile band in progress bins
    pbins = np.linspace(0.0, 1.0, 31)
    pcent = 0.5 * (pbins[:-1] + pbins[1:])
    med = np.full_like(pcent, np.nan, dtype=float)
    q25 = np.full_like(pcent, np.nan, dtype=float)
    q75 = np.full_like(pcent, np.nan, dtype=float)
    for i in range(len(pcent)):
        m = (s["progress"] >= pbins[i]) & (s["progress"] < pbins[i + 1])
        if m.any():
            vals = s.loc[m, "mag"].to_numpy(dtype=float)
            med[i] = np.nanmedian(vals)
            q25[i], q75[i] = np.nanpercentile(vals, [25, 75])
    med_s = pd.Series(med).rolling(window=min(smooth, len(med)), center=True, min_periods=1).median().to_numpy()
    ax_mag.fill_between(pcent, q25, q75, color="#10b981", alpha=0.24, label="IQR")
    ax_mag.plot(pcent, med_s, color="#047857", lw=2.2, label="Median (smoothed)")
    ax_mag.set_title("Action Magnitude Over Training Progress")
    ax_mag.set_xlabel("Normalized progress (0=start, 1=end)")
    ax_mag.set_ylabel(r"$\sqrt{v^2+w^2}$")
    ax_mag.legend(frameon=False, loc="best")

    # Panel 2 (required): Density + Phase Contours + Time-Ordered Ridge
    H_all, xedges, yedges = np.histogram2d(
        s["v"].to_numpy(dtype=float),
        s["w"].to_numpy(dtype=float),
        bins=bins,
        range=[xlim, ylim],
    )
    H_plot = H_all.T
    norm = LogNorm(vmin=1, vmax=max(1.0, float(np.nanmax(H_plot)))) if np.nanmax(H_plot) >= 1 else None
    im = ax_vw.pcolormesh(xedges, yedges, H_plot, cmap="Greys", shading="auto", norm=norm)
    cbar = fig.colorbar(im, ax=ax_vw, fraction=0.046, pad=0.04)
    cbar.set_label("Count per 2D bin" + (" (log scale)" if norm is not None else ""))

    # Phase contours with per-phase mass threshold.
    xcent = 0.5 * (xedges[:-1] + xedges[1:])
    ycent = 0.5 * (yedges[:-1] + yedges[1:])
    XX, YY = np.meshgrid(xcent, ycent)
    legend_handles = []
    for pid in range(phases):
        pm = s["phase"] == pid
        if not pm.any():
            continue
        ph = s.loc[pm]
        H_p, _, _ = np.histogram2d(
            ph["v"].to_numpy(dtype=float),
            ph["w"].to_numpy(dtype=float),
            bins=[xedges, yedges],
        )
        H_p = H_p.T
        thr = _mass_contour_threshold(H_p, mass=0.68)
        if thr is None:
            continue
        ls = line_styles[pid % len(line_styles)]
        col = phase_colors[pid % len(phase_colors)]
        ax_vw.contour(
            XX,
            YY,
            H_p,
            levels=[thr],
            colors=[col],
            linewidths=1.8,
            linestyles=[ls],
        )
        label = phase_labels[pid] if pid < len(phase_labels) else f"Phase {pid+1}"
        legend_handles.append(
            plt.Line2D(
                [0], [0], color=col, lw=2.0, linestyle=ls, label=f"{label} (n={int(pm.sum())})"
            )
        )

    # Time-ordered ridge/path: median (v,w) per time slice.
    t_edges = np.linspace(step_min, step_max, bins + 1)
    med_v, med_w, cnt, t_prog = [], [], [], []
    for i in range(len(t_edges) - 1):
        m = (s["step"] >= t_edges[i]) & (s["step"] < t_edges[i + 1])
        if i == len(t_edges) - 2:
            m = (s["step"] >= t_edges[i]) & (s["step"] <= t_edges[i + 1])
        if not m.any():
            continue
        sub = s.loc[m]
        med_v.append(float(np.nanmedian(sub["v"])))
        med_w.append(float(np.nanmedian(sub["w"])))
        cnt.append(int(len(sub)))
        t_prog.append(float(np.nanmedian(sub["progress"])))
    med_v = np.asarray(med_v, dtype=float)
    med_w = np.asarray(med_w, dtype=float)
    cnt = np.asarray(cnt, dtype=float)
    t_prog = np.asarray(t_prog, dtype=float)
    if med_v.size >= 2:
        ax_vw.plot(med_v, med_w, color="#111827", lw=1.6, alpha=0.85, zorder=4)
        smin, smax = 30.0, 120.0
        cnt_scaled = (cnt - cnt.min()) / (cnt.max() - cnt.min() + 1e-8)
        sizes = smin + (smax - smin) * cnt_scaled
        ridge = ax_vw.scatter(
            med_v,
            med_w,
            c=t_prog,
            cmap="Greys",
            s=sizes,
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )
        # Mark start/end clearly
        ax_vw.scatter([med_v[0]], [med_w[0]], marker="^", s=130, color="#16a34a", edgecolor="white", linewidth=0.8, zorder=6)
        ax_vw.scatter([med_v[-1]], [med_w[-1]], marker="X", s=130, color="#dc2626", edgecolor="white", linewidth=0.8, zorder=6)
        ax_vw.annotate("start", (med_v[0], med_w[0]), xytext=(6, 6), textcoords="offset points", fontsize=8, color="#14532d")
        ax_vw.annotate("end", (med_v[-1], med_w[-1]), xytext=(6, 6), textcoords="offset points", fontsize=8, color="#7f1d1d")
        for k in range(0, len(med_v), max(1, len(med_v) // 6)):
            ax_vw.annotate(f"{k+1}", (med_v[k], med_w[k]), xytext=(4, -7), textcoords="offset points", fontsize=7, color="#1f2937")

    if legend_handles:
        ax_vw.legend(handles=legend_handles, frameon=False, loc="upper right")
    ax_vw.set_title("Action Space: Density + Phase Contours + Time Ridge")
    ax_vw.set_xlabel("v")
    ax_vw.set_ylabel("w")
    ax_vw.set_xlim(*xlim)
    ax_vw.set_ylim(*ylim)

    # Panel 3/4: phase-wise distributions (violin) for coherence
    v_groups, w_groups, xlabels = [], [], []
    for pid in range(phases):
        pm = s["phase"] == pid
        if not pm.any():
            continue
        label = phase_labels[pid] if pid < len(phase_labels) else f"Phase {pid+1}"
        xlabels.append(f"{label}\nN={int(pm.sum())}")
        v_groups.append(s.loc[pm, "v"].to_numpy(dtype=float))
        w_groups.append(s.loc[pm, "w"].to_numpy(dtype=float))

    if v_groups:
        pos = np.arange(1, len(v_groups) + 1)
        vp = ax_vdist.violinplot(np.array(v_groups, dtype=object), positions=pos, showmedians=True, widths=0.82)
        for i, body in enumerate(vp["bodies"]):
            body.set_facecolor(phase_colors[i % len(phase_colors)])
            body.set_edgecolor("#111827")
            body.set_alpha(0.34)
        vp["cmedians"].set_color("#111827")
        ax_vdist.set_xticks(pos)
        ax_vdist.set_xticklabels(xlabels)
    ax_vdist.set_title("v Distribution by Training Phase")
    ax_vdist.set_ylabel("v")

    if w_groups:
        pos = np.arange(1, len(w_groups) + 1)
        wp = ax_wdist.violinplot(np.array(w_groups, dtype=object), positions=pos, showmedians=True, widths=0.82)
        for i, body in enumerate(wp["bodies"]):
            body.set_facecolor(phase_colors[i % len(phase_colors)])
            body.set_edgecolor("#111827")
            body.set_alpha(0.34)
        wp["cmedians"].set_color("#111827")
        ax_wdist.set_xticks(pos)
        ax_wdist.set_xticklabels(xlabels)
    ax_wdist.set_title("w Distribution by Training Phase")
    ax_wdist.set_ylabel("w")

    for ax in [ax_mag, ax_vw, ax_vdist, ax_wdist]:
        ax.grid(True, alpha=0.22, linestyle="--", linewidth=0.7)

    fig.suptitle(
        f"Robot Action Distribution Over Training | steps={len(s)} | parquet={os.path.basename(os.path.expanduser(parquet_path))}",
        y=0.99,
    )
    fig.text(
        0.5,
        0.01,
        "Caption: Action behavior shifts from exploration to more stable control as training progresses.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=[0.0, 0.03, 1.0, 0.95])

    out_png: str | None = None
    out_pdf: str | None = None
    if save:
        os.makedirs(os.path.dirname(os.path.expanduser(out_base)) or ".", exist_ok=True)
        out_png = os.path.expanduser(out_base)
        root, ext = os.path.splitext(out_png)
        if ext.lower() in {".png", ".pdf"}:
            out_png = root + ".png"
            out_pdf = root + ".pdf"
        else:
            out_png = out_png + ".png"
            out_pdf = out_png[:-4] + ".pdf"
        fig.savefig(out_png, dpi=int(max(72, dpi)), bbox_inches="tight")
        fig.savefig(out_pdf, dpi=int(max(72, dpi)), bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
    return out_png, out_pdf


def plot_core_losses_all_participants(
    df: pd.DataFrame, out_dir: str, show: bool = True
) -> tuple[str | None, plt.Figure | None]:
    """
    One figure with three subplots for all participants:
      - ent_coef
      - actor_loss
      - critic_loss
    """
    if df.empty:
        print("[INFO] No data to plot.")
        return None, None

    participants = sorted(df["participant"].unique().tolist())
    if not participants:
        print("[INFO] No participants found.")
        return None, None

    axis_label_size = float(PLOT_STYLE_CFG.get("participants_grid_axis_label_size", 14))
    tick_label_size = float(PLOT_STYLE_CFG.get("participants_grid_tick_label_size", 13))
    legend_font_size = float(PLOT_STYLE_CFG.get("participants_grid_legend_font_size", 13))
    metric_kinds = [("ent_coef", "Entropy Coef"), ("actor_loss", "Actor Loss"), ("critic_loss", "Critic Loss")]
    fig = plt.figure(figsize=(18, 9))
    # 4-column grid so top axes and bottom-centered axis can share identical size.
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0])
    axes = [
        fig.add_subplot(gs[0, 0:2]),  # top-left (width=2 cols)
        fig.add_subplot(gs[0, 2:4]),  # top-right (width=2 cols)
        fig.add_subplot(gs[1, 1:3]),  # bottom centered (width=2 cols, same as above)
    ]
    cmap = plt.get_cmap("tab10")
    color_map = {p: cmap(i % 10) for i, p in enumerate(participants)}
    alpha = 0.18  # mild smoothing for readability

    for ax, (kind, title) in zip(axes, metric_kinds):
        plotted_any = False
        max_ep = -1
        for participant in participants:
            run_df = _latest_run_df_for_participant(df, participant)
            if run_df.empty:
                continue
            tags = sorted(run_df["tag"].unique().tolist())
            tag = _choose_tag(tags, kind)
            if not tag:
                print(f"[INFO] Missing {kind} for {participant}. Available tags: {', '.join(tags)}")
                continue
            ser = _series_from_run_df(run_df, tag, kind)
            if ser.empty:
                continue
            x = _episode_axis_for_steps(run_df, ser["step"].to_numpy(dtype=float))
            y = ser[kind].to_numpy(dtype=float)
            x, y = _pad_episode_series(x, y, target_points=PLOT_EXPECTED_EPISODES)
            y_s = _ema(y, alpha=alpha)
            col = color_map[participant]
            ax.plot(x, y_s, color=col, alpha=0.95, linewidth=2.0, label=participant)
            max_ep = max(max_ep, _max_episode_index_for_run(run_df, fallback_n=int(len(x))))
            plotted_any = True
        ax.set_title(title, fontsize=tick_label_size)
        ax.set_xlabel("Episode", fontsize=axis_label_size)
        ax.set_ylabel("Value", fontsize=axis_label_size)
        ax.tick_params(axis="both", labelsize=tick_label_size)
        _apply_plot_grid(ax)
        if plotted_any:
            if max_ep >= 0:
                ticks = np.arange(0, int(max_ep) + 1, 1, dtype=int)
                ax.set_xticks(ticks)
                ax.set_xlim(-0.5, float(ticks[-1]) + 0.5)
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")

    legend_items = [
        Line2D([0], [0], color=color_map[p], lw=2.0, label=p.replace("_", " ")) for p in participants
    ]
    if legend_items:
        fig.legend(
            handles=legend_items,
            loc="upper center",
            ncol=min(max(1, len(legend_items)), 6),
            bbox_to_anchor=(0.5, 0.99),
            fontsize=legend_font_size,
            frameon=False,
        )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    if not show:
        plt.close(fig)
        return None, None
    return None, fig


def plot_raw_core_metrics_all_participants(
    df: pd.DataFrame, out_dir: str, show: bool = True
) -> tuple[str | None, plt.Figure | None]:
    """
    One figure with three subplots for all participants (raw, unsmoothed):
      - actor_loss
      - ent_coef
      - critic_loss
    """
    if df.empty:
        print("[INFO] No data to plot.")
        return None, None

    participants = sorted(df["participant"].unique().tolist())
    if not participants:
        print("[INFO] No participants found.")
        return None, None

    axis_label_size = float(PLOT_STYLE_CFG.get("participants_grid_axis_label_size", 14))
    tick_label_size = float(PLOT_STYLE_CFG.get("participants_grid_tick_label_size", 13))
    legend_font_size = float(PLOT_STYLE_CFG.get("participants_grid_legend_font_size", 13))
    metric_kinds = [
        ("actor_loss", "Actor Loss (Raw)"),
        ("ent_coef", "Entropy Coef (Raw)"),
        ("critic_loss", "Critic Loss (Raw)"),
    ]
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0])
    axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[1, 1:3]),
    ]
    cmap = plt.get_cmap("tab10")
    color_map = {p: cmap(i % 10) for i, p in enumerate(participants)}

    for ax, (kind, title) in zip(axes, metric_kinds):
        plotted_any = False
        max_ep = -1
        for participant in participants:
            run_df = _latest_run_df_for_participant(df, participant)
            if run_df.empty:
                continue
            tags = sorted(run_df["tag"].unique().tolist())
            tag = _choose_tag(tags, kind)
            if not tag:
                print(f"[INFO] Missing {kind} for {participant}. Available tags: {', '.join(tags)}")
                continue
            ser = _series_from_run_df(run_df, tag, kind)
            if ser.empty:
                continue
            x = _episode_axis_for_steps(run_df, ser["step"].to_numpy(dtype=float))
            y = ser[kind].to_numpy(dtype=float)
            x, y = _pad_episode_series(x, y, target_points=PLOT_EXPECTED_EPISODES)
            col = color_map[participant]
            ax.plot(x, y, color=col, alpha=0.95, linewidth=1.4, label=participant)
            max_ep = max(max_ep, _max_episode_index_for_run(run_df, fallback_n=int(len(x))))
            plotted_any = True
        ax.set_title(title, fontsize=tick_label_size)
        ax.set_xlabel("Episode", fontsize=axis_label_size)
        ax.set_ylabel("Value", fontsize=axis_label_size)
        ax.tick_params(axis="both", labelsize=tick_label_size)
        _apply_plot_grid(ax)
        if not plotted_any:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        else:
            ticks = np.arange(0, int(max_ep) + 1, 1, dtype=int)
            ax.set_xticks(ticks)
            ax.set_xlim(-0.5, float(ticks[-1]) + 0.5)

    legend_items = [
        Line2D([0], [0], color=color_map[p], lw=2.0, label=p.replace("_", " ")) for p in participants
    ]
    if legend_items:
        fig.legend(
            handles=legend_items,
            loc="upper center",
            ncol=min(max(1, len(legend_items)), 6),
            bbox_to_anchor=(0.5, 0.99),
            fontsize=legend_font_size,
            frameon=False,
        )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    if not show:
        plt.close(fig)
        return None, None
    return None, fig


def _participant_zone_action_df(run_df: pd.DataFrame) -> pd.DataFrame:
    if run_df.empty:
        return pd.DataFrame()
    if not {"episode", "table_x", "table_y"}.issubset(set(run_df.columns)):
        return pd.DataFrame()
    v_col = "act_exec_v" if "act_exec_v" in run_df.columns else ("act_raw_v" if "act_raw_v" in run_df.columns else None)
    w_col = "act_exec_w" if "act_exec_w" in run_df.columns else ("act_raw_w" if "act_raw_w" in run_df.columns else None)
    if not v_col or not w_col:
        return pd.DataFrame()

    goal_x_col = "goal_x" if "goal_x" in run_df.columns else None
    goal_y_col = "goal_y" if "goal_y" in run_df.columns else None
    cols = ["episode", "table_x", "table_y", v_col, w_col]
    if "t" in run_df.columns:
        cols.append("t")
    if goal_x_col:
        cols.append(goal_x_col)
    if goal_y_col:
        cols.append(goal_y_col)
    s = run_df[cols].copy()
    sort_cols = [c for c in ("episode", "t") if c in s.columns]
    if sort_cols:
        s = s.sort_values(sort_cols)
    s = s.dropna(subset=["episode", "table_x", "table_y", v_col, w_col])
    if s.empty:
        return pd.DataFrame()

    recs: list[dict] = []
    for ep in np.sort(s["episode"].dropna().unique()):
        g = s[s["episode"] == ep].copy()
        n = int(len(g))
        if n <= 0:
            continue
        x = pd.to_numeric(g["table_x"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(g["table_y"], errors="coerce").to_numpy(dtype=float)
        v_ep = pd.to_numeric(g[v_col], errors="coerce").to_numpy(dtype=float)
        w_ep = pd.to_numeric(g[w_col], errors="coerce").to_numpy(dtype=float)
        gx = (
            pd.to_numeric(g[goal_x_col], errors="coerce").to_numpy(dtype=float)
            if goal_x_col
            else np.full(n, np.nan, dtype=float)
        )
        gy = (
            pd.to_numeric(g[goal_y_col], errors="coerce").to_numpy(dtype=float)
            if goal_y_col
            else np.full(n, np.nan, dtype=float)
        )
        p = (np.arange(n, dtype=float) + 0.5) / float(n)
        for i in range(n):
            if not (np.isfinite(x[i]) and np.isfinite(y[i]) and np.isfinite(v_ep[i]) and np.isfinite(w_ep[i]) and np.isfinite(p[i])):
                continue
            zone = _classify_spatial_zone(
                float(x[i]),
                float(y[i]),
                (float(gx[i]) if np.isfinite(gx[i]) else None),
                (float(gy[i]) if np.isfinite(gy[i]) else None),
            )
            recs.append(
                {
                    "episode": int(ep),
                    "progress": float(np.clip(p[i], 0.0, 1.0)),
                    "zone": zone,
                    "v": float(v_ep[i]),
                    "w": float(w_ep[i]),
                }
            )
    if not recs:
        return pd.DataFrame()
    dfz = pd.DataFrame(recs)
    q1, q2 = np.quantile(dfz["progress"].to_numpy(dtype=float), [1.0 / 3.0, 2.0 / 3.0])
    stage = np.where(dfz["progress"] <= q1, "Early", np.where(dfz["progress"] <= q2, "Mid", "Late"))
    dfz["stage"] = stage
    return dfz


def plot_action_behavior_all_participants(
    runs: list[RunInfo], df: pd.DataFrame, out_dir: str, show: bool = True
) -> tuple[str | None, list[plt.Figure]]:
    """
    Build a gallery of zone-safe action distribution plots for Participant A.
    """
    if df.empty or not runs:
        return None, []

    latest_runs = _latest_runinfo_by_participant(runs, df)
    if not latest_runs:
        return None, []

    # For now, focus on Participant A to make the behavior figure easier to read.
    participant = "Participant_A"
    if participant not in latest_runs:
        participant = sorted(latest_runs.keys())[0]
    run = latest_runs[participant]
    d = _load_steps_parquet(run.run_dir)
    if d.empty:
        return None, []
    rdf = _participant_zone_action_df(d)
    if rdf.empty:
        return None, []

    zone_order = ["Open", "Obstacle", "Goal"]
    zone_colors = {"Open": "#1f77b4", "Obstacle": "#d97706", "Goal": "#2a9d8f"}
    stage_order = ["Early", "Mid", "Late"]

    figs: list[plt.Figure] = []

    # 1) Combined zone figure
    v_groups = [rdf.loc[rdf["zone"] == z, "v"].to_numpy(dtype=float) for z in zone_order]
    w_groups = [rdf.loc[rdf["zone"] == z, "w"].to_numpy(dtype=float) for z in zone_order]
    n_zone = [int(len(g)) for g in v_groups]
    fig1, axes1 = plt.subplots(1, 3, figsize=(14.5, 4.6), squeeze=False)
    ax_v, ax_w, ax_phase = axes1.ravel()
    vp = ax_v.violinplot(np.array(v_groups, dtype=object), positions=[1, 2, 3], showmedians=True, widths=0.82)
    for i, body in enumerate(vp["bodies"]):
        z = zone_order[i]
        body.set_facecolor(zone_colors[z]); body.set_edgecolor("#111827"); body.set_alpha(0.34)
    vp["cmedians"].set_color("#111827")
    ax_v.set_xticks([1, 2, 3]); ax_v.set_xticklabels([f"{z}\nN={n_zone[i]}" for i, z in enumerate(zone_order)])
    ax_v.set_title("Speed v Distribution by Zone"); ax_v.set_ylabel("v")
    wp = ax_w.violinplot(np.array(w_groups, dtype=object), positions=[1, 2, 3], showmedians=True, widths=0.82)
    for i, body in enumerate(wp["bodies"]):
        z = zone_order[i]
        body.set_facecolor(zone_colors[z]); body.set_edgecolor("#111827"); body.set_alpha(0.34)
    wp["cmedians"].set_color("#111827")
    ax_w.set_xticks([1, 2, 3]); ax_w.set_xticklabels([f"{z}\nN={n_zone[i]}" for i, z in enumerate(zone_order)])
    ax_w.set_title("Yaw Rate w Distribution by Zone"); ax_w.set_ylabel("w")

    # (v,w) density per zone (overlayed hexbins)
    for z, cmap in [("Open", "Blues"), ("Obstacle", "Oranges"), ("Goal", "Greens")]:
        zz = rdf[rdf["zone"] == z]
        if zz.empty:
            continue
        ax_phase.hexbin(
            zz["v"].to_numpy(dtype=float),
            zz["w"].to_numpy(dtype=float),
            gridsize=36,
            mincnt=1,
            cmap=cmap,
            alpha=0.35,
        )
    ax_phase.set_title("Action Space Density by Zone")
    ax_phase.set_xlabel("v")
    ax_phase.set_ylabel("w")
    for ax in [ax_v, ax_w, ax_phase]:
        ax.grid(True, alpha=0.22, linestyle="--", linewidth=0.7)
    zone_handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=zone_colors["Open"], markersize=8, label=f"Open (N={n_zone[0]})"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=zone_colors["Obstacle"], markersize=8, label=f"Obstacle (N={n_zone[1]})"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=zone_colors["Goal"], markersize=8, label=f"Goal (N={n_zone[2]})"),
    ]
    ax_phase.legend(handles=zone_handles, frameon=False, loc="upper right")
    fig1.suptitle(f"Participant A Actions by Zone — {participant} ({run.run_id}) | N={len(rdf)}", y=0.98)
    fig1.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    figs.append(fig1)

    # 2) Zone+stage violins
    fig2, axes2 = plt.subplots(1, 2, figsize=(12.8, 4.6), squeeze=False)
    for ax, val_col, ttl in [(axes2[0, 0], "v", "v by Zone and Stage"), (axes2[0, 1], "w", "w by Zone and Stage")]:
        positions, labels, groups, cols = [], [], [], []
        pos = 1
        for z in zone_order:
            for st in stage_order:
                vals = rdf[(rdf["zone"] == z) & (rdf["stage"] == st)][val_col].to_numpy(dtype=float)
                if vals.size == 0:
                    continue
                positions.append(pos); labels.append(f"{z[:3]}-{st[0]}"); groups.append(vals); cols.append(zone_colors[z]); pos += 1
            pos += 1
        if groups:
            vp2 = ax.violinplot(np.array(groups, dtype=object), positions=positions, showmedians=True, widths=0.8)
            for i, body in enumerate(vp2["bodies"]):
                body.set_facecolor(cols[i]); body.set_edgecolor("#111827"); body.set_alpha(0.35)
            vp2["cmedians"].set_color("#111827")
            ax.set_xticks(positions); ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(ttl); ax.set_ylabel(val_col); ax.grid(True, alpha=0.22, linestyle="--", linewidth=0.7)
    fig2.suptitle("Participant A Zone-Stage Distributions", y=0.98)
    fig2.tight_layout(rect=[0.0, 0.0, 1.0, 0.93]); figs.append(fig2)

    # 3) Step-bin x action-bin heatmaps by zone
    fig3, axes3 = plt.subplots(3, 2, figsize=(12.8, 9.4), squeeze=False)
    for r, z in enumerate(zone_order):
        zz = rdf[rdf["zone"] == z]
        if zz.empty:
            continue
        for c, col_name in enumerate(["v", "w"]):
            H, xed, yed = np.histogram2d(
                zz["progress"].to_numpy(dtype=float),
                zz[col_name].to_numpy(dtype=float),
                bins=[30, 40],
                range=[[0, 1], [np.nanquantile(rdf[col_name], 0.01), np.nanquantile(rdf[col_name], 0.99)]],
            )
            im = axes3[r, c].pcolormesh(xed, yed, H.T, shading="auto", cmap="viridis")
            axes3[r, c].set_title(f"{z}: {col_name} density over progress")
            axes3[r, c].set_xlabel("progress"); axes3[r, c].set_ylabel(col_name)
            fig3.colorbar(im, ax=axes3[r, c], fraction=0.046, pad=0.04)
            axes3[r, c].grid(True, alpha=0.18, linestyle="--", linewidth=0.6)
    fig3.suptitle("Participant A Heatmaps by Zone", y=0.99)
    fig3.tight_layout(rect=[0.0, 0.0, 1.0, 0.96]); figs.append(fig3)

    # 4) Quantile trajectories by zone
    fig4, axes4 = plt.subplots(1, 2, figsize=(12.8, 4.6), squeeze=False)
    p_bins = np.linspace(0.0, 1.0, 31)
    p_cent = 0.5 * (p_bins[:-1] + p_bins[1:])
    for ax, col_name, ttl in [(axes4[0, 0], "v", "v quantiles by zone"), (axes4[0, 1], "w", "w quantiles by zone")]:
        for z in zone_order:
            zz = rdf[rdf["zone"] == z]
            med = np.full_like(p_cent, np.nan); q25 = np.full_like(p_cent, np.nan); q75 = np.full_like(p_cent, np.nan)
            for i in range(len(p_cent)):
                m = (zz["progress"] >= p_bins[i]) & (zz["progress"] < p_bins[i + 1])
                if m.any():
                    vals = zz.loc[m, col_name].to_numpy(dtype=float)
                    med[i] = np.nanmedian(vals); q25[i], q75[i] = np.nanpercentile(vals, [25, 75])
            ax.fill_between(p_cent, q25, q75, color=zone_colors[z], alpha=0.18, lw=0)
            ax.plot(p_cent, med, color=zone_colors[z], lw=1.9, label=z)
        ax.set_title(ttl); ax.set_xlabel("progress"); ax.set_ylabel(col_name)
        ax.grid(True, alpha=0.22, linestyle="--", linewidth=0.7); ax.legend(frameon=False, loc="best")
    fig4.suptitle("Participant A Quantile Trajectories by Zone", y=0.98)
    fig4.tight_layout(rect=[0.0, 0.0, 1.0, 0.93]); figs.append(fig4)

    # 5) CDFs by zone
    fig5, axes5 = plt.subplots(1, 2, figsize=(12.8, 4.4), squeeze=False)
    for ax, col_name, ttl in [(axes5[0, 0], "v", "CDF of v by zone"), (axes5[0, 1], "w", "CDF of w by zone")]:
        for z in zone_order:
            vals = np.sort(rdf[rdf["zone"] == z][col_name].to_numpy(dtype=float))
            if vals.size == 0:
                continue
            yy = np.arange(1, vals.size + 1) / vals.size
            ax.plot(vals, yy, color=zone_colors[z], lw=2.0, label=f"{z} (N={vals.size})")
        ax.set_title(ttl); ax.set_xlabel(col_name); ax.set_ylabel("F(x)")
        ax.grid(True, alpha=0.22, linestyle="--", linewidth=0.7); ax.legend(frameon=False, loc="lower right")
    fig5.suptitle("Participant A Empirical CDFs by Zone", y=0.98)
    fig5.tight_layout(rect=[0.0, 0.0, 1.0, 0.93]); figs.append(fig5)

    if not show:
        for f in figs:
            plt.close(f)
        return None, []
    return None, figs


def _choose_tag(available_tags: list[str], kind: str) -> str | None:
    aset = set(available_tags)
    if kind == "critic_loss":
        for t in ["train/critic_loss", "critic_loss", "loss/critic"]:
            if t in aset:
                return t
        for t in available_tags:
            if "critic_loss" in t.lower():
                return t
        for t in available_tags:
            low = t.lower()
            if "critic" in low and "loss" in low:
                return t
        return None
    if kind == "entropy":
        for t in ["train/entropy", "entropy"]:
            if t in aset:
                return t
        for t in available_tags:
            low = t.lower()
            if "entropy" in low:
                return t
        return None
    if kind == "reward":
        for t in ["reward/episode_mean", "rollout/ep_rew_mean", "episode_reward_mean"]:
            if t in aset:
                return t
        for t in available_tags:
            low = t.lower()
            if "rew" in low and "mean" in low:
                return t
        return None
    if kind == "ent_coef":
        for t in ["train/ent_coef", "ent_coef"]:
            if t in aset:
                return t
        for t in available_tags:
            if "ent_coef" in t.lower():
                return t
        return None
    if kind == "actor_loss":
        for t in ["train/actor_loss", "actor_loss", "loss/actor"]:
            if t in aset:
                return t
        for t in available_tags:
            low = t.lower()
            if "actor" in low and "loss" in low:
                return t
        return None
    return None


def _print_run_missing(
    run: RunInfo,
    message: str,
    available_tags: list[str],
    required_tag_candidates: dict[str, list[str]] | None = None,
) -> None:
    print(f"[INFO] Run '{run.run_id}' skipped for {message} (run_dir={run.run_dir})")
    if required_tag_candidates:
        print("       Required scalar families and candidate tags:")
        for family, candidates in required_tag_candidates.items():
            print(f"       - {family}: {', '.join(candidates)}")
    if available_tags:
        print("       Available scalar tags:")
        for tag in available_tags:
            print(f"       - {tag}")
    else:
        print("       No scalar tags loaded.")


def _series_from_run_df(run_df: pd.DataFrame, tag: str, value_name: str) -> pd.DataFrame:
    d = run_df[run_df["tag"] == tag][["step", "value"]].copy()
    d["step"] = d["step"].astype(int)
    d = d.sort_values("step")
    d = d.drop_duplicates(subset=["step"], keep="last")
    return d.rename(columns={"value": value_name}).reset_index(drop=True)


def _merge_nearest(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    l = left.sort_values("step").copy()
    r = right.sort_values("step").copy()
    l["step"] = l["step"].astype(int)
    r["step"] = r["step"].astype(int)
    return pd.merge_asof(l, r, on="step", direction="nearest")


def _episode_axis_for_steps(run_df: pd.DataFrame, steps: np.ndarray) -> np.ndarray:
    """
    Map global scalar steps to an episode index using nearest reward/episode_mean step.
    Falls back to 0..N-1 when no reward tag is available.
    """
    x_steps = np.asarray(steps, dtype=float)
    if x_steps.size == 0:
        return x_steps

    tags = sorted(run_df["tag"].unique().tolist())
    reward_tag = _choose_tag(tags, "reward")
    if not reward_tag:
        return np.arange(0, x_steps.size, dtype=float)

    reward_df = _series_from_run_df(run_df, reward_tag, "reward_episode_mean")
    if reward_df.empty:
        return np.arange(0, x_steps.size, dtype=float)

    reward_steps = reward_df["step"].to_numpy(dtype=float)
    if reward_steps.size == 0:
        return np.arange(0, x_steps.size, dtype=float)

    idx = np.searchsorted(reward_steps, x_steps, side="left")
    idx = np.clip(idx, 0, reward_steps.size - 1)
    prev_idx = np.maximum(idx - 1, 0)
    choose_prev = np.abs(x_steps - reward_steps[prev_idx]) <= np.abs(x_steps - reward_steps[idx])
    nearest = np.where(choose_prev, prev_idx, idx)
    return nearest.astype(float)


def _max_episode_index_for_run(run_df: pd.DataFrame, fallback_n: int) -> int:
    """
    Return the maximum episode index for axis ticks.
    Prefer reward/episode_mean length; fallback to metric sample count.
    """
    tags = sorted(run_df["tag"].unique().tolist())
    reward_tag = _choose_tag(tags, "reward")
    if reward_tag:
        reward_df = _series_from_run_df(run_df, reward_tag, "reward_episode_mean")
        if not reward_df.empty:
            return max(int(PLOT_EXPECTED_EPISODES) - 1, int(len(reward_df)) - 1)
    return max(int(PLOT_EXPECTED_EPISODES) - 1, int(fallback_n) - 1)


def _pad_episode_series(x: np.ndarray, y: np.ndarray, target_points: int = PLOT_EXPECTED_EPISODES) -> tuple[np.ndarray, np.ndarray]:
    """
    Ensure an episode curve reaches target_points on x-axis (0..target_points-1)
    by extending with the last observed y value when needed.
    """
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    if xx.size == 0 or yy.size == 0 or xx.size != yy.size:
        return xx, yy

    target_points = int(max(1, target_points))
    last_ep = int(np.nanmax(xx))
    target_last_ep = target_points - 1
    if last_ep >= target_last_ep:
        return xx, yy

    x_pad = np.arange(last_ep + 1, target_last_ep + 1, dtype=float)
    y_pad = np.full(x_pad.shape, float(yy[-1]), dtype=float)
    return np.concatenate([xx, x_pad]), np.concatenate([yy, y_pad])


def plot_normalized_critic_error(df_norm: pd.DataFrame, participant_label: str = "Participant A") -> None:
    """Display normalized critic error vs training step."""
    if df_norm.empty:
        print("[INFO] No data to plot for normalized critic error.")
        return
    plot_df = df_norm.sort_values("adjusted_step")
    plt.figure(figsize=(10, 5))
    plt.plot(
        plot_df["adjusted_step"].to_numpy(dtype=float),
        plot_df["normalized_critic_error"].to_numpy(dtype=float),
    )
    plt.title(f"{participant_label} — Normalized Critic Error")
    plt.xlabel("Training step")
    plt.ylabel("critic_loss / (|episode_mean_reward| + eps)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_entropy_reward_corr(
    df_corr: pd.DataFrame,
    window_points: int,
    participant_label: str = "Participant A",
) -> None:
    """Display rolling Pearson corr(entropy, reward) vs training step."""
    if df_corr.empty:
        print("[INFO] No data to plot for rolling entropy-reward correlation.")
        return
    plot_df = df_corr.sort_values("adjusted_step")
    plt.figure(figsize=(10, 5))
    plt.plot(
        plot_df["adjusted_step"].to_numpy(dtype=float),
        plot_df["rolling_entropy_reward_corr"].to_numpy(dtype=float),
    )
    plt.title(f"{participant_label} — Rolling Corr(Entropy, Reward)")
    plt.xlabel("Training step")
    plt.ylabel("Pearson correlation")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_participant_diagnostics(
    runs: list[RunInfo],
    all_df: pd.DataFrame,
    out_dir: str,
    participant_label_for_name: str,
    window_points: int,
) -> list[str]:
    """
    Creates:
      - normalized critic error plot
      - rolling entropy-reward correlation plot
      - aligned CSV used for these plots
    """
    if all_df.empty:
        print("[INFO] No scalar data available for participant diagnostics.")
        return []

    # Chronological run order by earliest wall_time, then concatenate with step offsets.
    run_order = []
    for run in runs:
        run_df = all_df[all_df["run_id"] == run.run_id]
        first_wall = float(run_df["wall_time"].min()) if not run_df.empty else float("inf")
        run_order.append((first_wall, run))
    run_order.sort(key=lambda x: x[0])

    norm_parts: list[pd.DataFrame] = []
    corr_parts: list[pd.DataFrame] = []
    step_offset = 0

    for _, run in run_order:
        run_df = all_df[all_df["run_id"] == run.run_id].copy()
        tags = sorted(run_df["tag"].unique().tolist())
        if not tags:
            _print_run_missing(run, "all diagnostics (no scalars)", tags)
            continue

        critic_tag = _choose_tag(tags, "critic_loss")
        reward_tag = _choose_tag(tags, "reward")
        entropy_tag = _choose_tag(tags, "entropy")

        if critic_tag and reward_tag:
            critic_df = _series_from_run_df(run_df, critic_tag, "critic_loss")
            reward_df = _series_from_run_df(run_df, reward_tag, "reward_episode_mean")
            mcr = _merge_nearest(critic_df, reward_df)
            mcr["normalized_critic_error"] = mcr["critic_loss"] / (mcr["reward_episode_mean"].abs() + 1e-8)
            mcr["run_id"] = run.run_id
            mcr["run_dir"] = run.run_dir
            mcr["adjusted_step"] = mcr["step"].astype(int) + int(step_offset)
            norm_parts.append(mcr)
        else:
            _print_run_missing(
                run,
                "normalized critic error (missing critic_loss or reward tag)",
                tags,
                required_tag_candidates={
                    "critic_loss": ["train/critic_loss", "critic_loss", "loss/critic", "*critic*loss*"],
                    "reward": ["reward/episode_mean", "rollout/ep_rew_mean", "episode_reward_mean", "*rew*mean*"],
                },
            )

        if entropy_tag and reward_tag:
            entropy_df = _series_from_run_df(run_df, entropy_tag, "entropy")
            reward_df = _series_from_run_df(run_df, reward_tag, "reward_episode_mean")
            mer = _merge_nearest(entropy_df, reward_df)
            min_periods = max(5, min(window_points, 20))
            mer["rolling_entropy_reward_corr"] = (
                mer["entropy"].rolling(window=window_points, min_periods=min_periods).corr(mer["reward_episode_mean"])
            )
            mer["run_id"] = run.run_id
            mer["run_dir"] = run.run_dir
            mer["adjusted_step"] = mer["step"].astype(int) + int(step_offset)
            corr_parts.append(mer)
        else:
            _print_run_missing(
                run,
                "rolling entropy-reward correlation (missing entropy or reward tag)",
                tags,
                required_tag_candidates={
                    "entropy": ["train/entropy", "entropy", "*entropy*"],
                    "reward": ["reward/episode_mean", "rollout/ep_rew_mean", "episode_reward_mean", "*rew*mean*"],
                },
            )

        if not run_df.empty:
            step_offset += int(run_df["step"].max()) + 1

    norm_all = pd.concat(norm_parts, ignore_index=True) if norm_parts else pd.DataFrame()
    corr_all = pd.concat(corr_parts, ignore_index=True) if corr_parts else pd.DataFrame()
    if norm_all.empty and corr_all.empty:
        print("[INFO] No participant diagnostic plots could be generated.")
        return []

    os.makedirs(out_dir, exist_ok=True)
    prefix = "participantA" if participant_label_for_name.lower() in {"a", "participant_a"} else _sanitize_filename(participant_label_for_name)
    csv_path = os.path.join(out_dir, f"{prefix}_aligned_data.csv")

    if not norm_all.empty:
        norm_csv = norm_all[
            ["run_id", "run_dir", "step", "adjusted_step", "critic_loss", "reward_episode_mean", "normalized_critic_error"]
        ].copy()
    else:
        norm_csv = pd.DataFrame(
            columns=["run_id", "run_dir", "step", "adjusted_step", "critic_loss", "reward_episode_mean", "normalized_critic_error"]
        )

    if not corr_all.empty:
        corr_csv = corr_all[
            ["run_id", "run_dir", "step", "adjusted_step", "entropy", "reward_episode_mean", "rolling_entropy_reward_corr"]
        ].copy()
        corr_csv = corr_csv.rename(columns={"reward_episode_mean": "reward_for_entropy_corr"})
    else:
        corr_csv = pd.DataFrame(
            columns=["run_id", "run_dir", "step", "adjusted_step", "entropy", "reward_for_entropy_corr", "rolling_entropy_reward_corr"]
        )

    merged_csv = pd.merge(
        norm_csv.rename(columns={"reward_episode_mean": "reward_for_norm_critic"}),
        corr_csv,
        on=["run_id", "run_dir", "step", "adjusted_step"],
        how="outer",
    ).sort_values(["adjusted_step", "run_id"])
    merged_csv.to_csv(csv_path, index=False)
    print(f"Wrote diagnostics CSV: {csv_path}")

    written_paths: list[str] = [csv_path]
    participant_title = "Participant A" if participant_label_for_name.lower() in {"a", "participant_a"} else participant_label_for_name
    if not norm_all.empty:
        plot_normalized_critic_error(norm_all, participant_label=participant_title)
    else:
        print("[INFO] Skipped normalized critic error plot.")

    if not corr_all.empty:
        plot_entropy_reward_corr(corr_all, window_points=window_points, participant_label=participant_title)
    else:
        print("[INFO] Skipped rolling entropy-reward correlation plot.")

    return written_paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Read RL scalar data directly from TensorBoard event files and export CSV.")
    p.add_argument("--root-dir", default=DEFAULT_ROOT, help="Root folder containing participant runs.")
    p.add_argument("--out-dir", default=None, help="Output directory for CSV exports. Defaults to <root-dir>/rl_analysis_exports.")
    p.add_argument("--participant", default="", help="Optional participant filter. Empty = all participants.")
    p.add_argument("--run-id", default="", help="Filter by exact run_id.")
    p.add_argument("--tags", default="", help="Comma-separated scalar tags to keep.")
    p.add_argument("--list-tags-only", action="store_true", help="Only list available scalar tags and exit.")
    p.add_argument(
        "--plot",
        dest="plot",
        action="store_true",
        help="Show combined plots for all participants (smoothed: ent_coef/actor_loss/critic_loss, raw: actor_loss/ent_coef/critic_loss).",
    )
    p.add_argument("--no-plot", dest="plot", action="store_false", help="Disable plot creation.")
    p.set_defaults(plot=True)
    p.add_argument("--show-plots", dest="show_plots", action="store_true", help="Show matplotlib windows while saving files.")
    p.add_argument("--no-show-plots", dest="show_plots", action="store_false", help="Do not open windows; only save files.")
    p.set_defaults(show_plots=GUI_AVAILABLE)
    p.add_argument("--parquet", default="", help="Parquet file or directory for direct action-distribution plotting mode.")
    p.add_argument("--out", default="", help="Output path stem for parquet plotting mode. Saves both .png and .pdf.")
    p.add_argument("--step-col", default="", help="Override step column name for parquet plotting mode.")
    p.add_argument("--v-col", default="", help="Override v action column name for parquet plotting mode.")
    p.add_argument("--w-col", default="", help="Override w action column name for parquet plotting mode.")
    p.add_argument("--bins", type=int, default=48, help="Bins for 2D hist/time slices in parquet plotting mode.")
    p.add_argument("--smooth", type=int, default=101, help="Smoothing window (odd int) for trend curve in parquet plotting mode.")
    p.add_argument("--phases", type=int, default=3, help="Number of quantile phases (default 3: early/mid/late) in parquet plotting mode.")
    p.add_argument("--dpi", type=int, default=180, help="Figure DPI for saved files in parquet plotting mode.")
    p.add_argument(
        "--robust-limits",
        action="store_true",
        help="Use robust quantile-based (v,w) axis limits in parquet plotting mode.",
    )
    p.add_argument("--smoothing", type=float, default=0.6, help="Smoothing factor in [0, 0.999].")
    p.add_argument(
        "--window-points",
        type=int,
        default=200,
        help="Reserved for optional diagnostics; not used in default combined plot.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _configure_ieee_plot_style(PLOT_STYLE_CFG)
    if args.parquet:
        out_base = args.out or os.path.join(os.getcwd(), "action_distribution_over_training")
        try:
            create_action_distribution_over_training_figure(
                parquet_path=args.parquet,
                out_base=out_base,
                step_col=(args.step_col or None),
                v_col=(args.v_col or None),
                w_col=(args.w_col or None),
                bins=args.bins,
                smooth=args.smooth,
                phases=args.phases,
                dpi=args.dpi,
                robust_limits=bool(args.robust_limits),
                save=False,
                show=(args.show_plots and GUI_AVAILABLE),
            )
        except Exception as exc:
            raise SystemExit(f"[ERROR] Could not generate action distribution figure: {exc}") from exc
        if args.show_plots and not GUI_AVAILABLE:
            print("GUI plotting unavailable in this environment.")
        return

    root_dir = os.path.expanduser(args.root_dir)
    out_dir = args.out_dir or os.path.join(root_dir, DEFAULT_OUT_SUBDIR)

    runs = discover_runs(root_dir)
    if args.participant:
        runs = [r for r in runs if r.participant == args.participant]
    if args.run_id:
        runs = [r for r in runs if r.run_id == args.run_id]

    if not runs:
        print(f"No runs found under: {root_dir}")
        return

    tags_filter = {t.strip() for t in args.tags.split(",") if t.strip()} if args.tags else None

    all_tags = list_all_scalar_tags(runs)
    if args.list_tags_only:
        print("Available scalar tags:")
        for tag in all_tags:
            print(f"- {tag}")
        return

    df = collect_scalars(runs, include_tags=tags_filter)
    if df.empty:
        print("No scalar data found for selected filters.")
        return

    summary = build_summary(df)
    wide = build_wide(df)

    os.makedirs(out_dir, exist_ok=True)

    print(f"Runs loaded: {len(runs)}")
    print(f"Participants: {', '.join(sorted(df['participant'].unique()))}")
    print(f"Tags exported ({df['tag'].nunique()}): {', '.join(sorted(df['tag'].unique()))}")
    print(f"Rows (long): {len(df)}")
    print("[INFO] CSV export skipped.")

    if args.plot:
        if args.show_plots and not GUI_AVAILABLE:
            print("GUI plotting unavailable in this environment. Saving plots only.")
        do_show = args.show_plots and GUI_AVAILABLE
        _, fig1 = plot_core_losses_all_participants(df=df, out_dir=out_dir, show=do_show)
        _, fig2 = plot_raw_core_metrics_all_participants(df=df, out_dir=out_dir, show=do_show)
        if do_show and (fig1 is not None):
            plt.show()
            plt.close(fig1)
        if do_show and (fig2 is not None):
            plt.show()
            plt.close(fig2)

    key_tag = "reward/episode_mean"
    if key_tag in set(summary["tag"].unique()):
        key = summary[summary["tag"] == key_tag][["participant", "run_id", "value_last", "step_max"]]
        print("\nLast reward/episode_mean per run:")
        print(key.to_string(index=False))


if __name__ == "__main__":
    main()
