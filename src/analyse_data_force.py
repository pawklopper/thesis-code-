#!/usr/bin/env python3
from __future__ import annotations

import glob
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONFIG = {
    "base_dir": "Participant_A_Dimi",
    "subdir": "20260204-130020_experiment_28_jan_20000",
    "model": "experiment_28_jan_20000_final",
    "eps": 1e-1,
    # Optional uniform downsampling per episode to keep plots responsive.
    # Set to None to disable downsampling.
    "max_points_per_episode": 5000,
    "facet_by_participant": False,
    "bins": 60,
    "smooth_window": 51,
    "bin_size_steps": 20,
    "winsor_quantiles": (0.01, 0.99),
    "outdir": None,
}


def resolve_steps_dir(cfg: dict[str, Any]) -> str:
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
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    sort_cols = [c for c in ("episode", "t") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def _ensure_success_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "success" in out.columns:
        out["success"] = out["success"].astype(bool)
        return out

    if {"episode", "terminated", "truncated"}.issubset(out.columns):
        flags = (
            out.groupby("episode", as_index=False)[["terminated", "truncated"]]
            .max()
            .assign(success=lambda x: x["terminated"].astype(bool) & ~x["truncated"].astype(bool))
        )
        out = out.merge(flags[["episode", "success"]], on="episode", how="left")
        out["success"] = out["success"].fillna(False).astype(bool)
        return out

    out["success"] = False
    return out


def _ensure_participant_column(df: pd.DataFrame, default_participant: str = "pooled") -> pd.DataFrame:
    out = df.copy()
    if "participant" not in out.columns:
        out["participant"] = default_participant
    out["participant"] = out["participant"].astype(str)
    return out


def _uniform_downsample_per_episode(df: pd.DataFrame, max_points_per_episode: int | None) -> pd.DataFrame:
    if max_points_per_episode is None or max_points_per_episode <= 0:
        return df
    if "episode" not in df.columns:
        return df

    parts: list[pd.DataFrame] = []
    for _, g in df.groupby("episode", sort=True):
        if len(g) <= max_points_per_episode:
            parts.append(g)
            continue
        idx = np.linspace(0, len(g) - 1, max_points_per_episode, dtype=int)
        parts.append(g.iloc[idx])
    return pd.concat(parts, ignore_index=True)


def compute_force_alignment(df: pd.DataFrame, eps: float = 1e-3) -> pd.DataFrame:
    """Compute force alignment metrics and return a filtered analysis dataframe.

    Metrics:
      - fh_mag = ||Fh||
      - fr_mag = ||Fr||
      - cosine similarity = (Fh . Fr) / (||Fh|| ||Fr||)
      - angle_deg = arccos(clamp(cos, -1, 1)) in degrees

    Samples with fh_mag < eps or fr_mag < eps are removed.
    """
    required = {"Fh_x", "Fh_y", "Fr_x", "Fr_y"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    out = df.copy()
    fhx = pd.to_numeric(out["Fh_x"], errors="coerce").to_numpy(dtype=float)
    fhy = pd.to_numeric(out["Fh_y"], errors="coerce").to_numpy(dtype=float)
    frx = pd.to_numeric(out["Fr_x"], errors="coerce").to_numpy(dtype=float)
    fry = pd.to_numeric(out["Fr_y"], errors="coerce").to_numpy(dtype=float)

    fh_mag = np.sqrt(fhx * fhx + fhy * fhy)
    fr_mag = np.sqrt(frx * frx + fry * fry)
    dot = fhx * frx + fhy * fry
    denom = fh_mag * fr_mag

    cos = np.full_like(denom, np.nan, dtype=float)
    valid = np.isfinite(denom) & (denom > 0.0)
    cos[valid] = dot[valid] / denom[valid]
    cos = np.clip(cos, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos))

    out["fh_mag"] = fh_mag
    out["fr_mag"] = fr_mag
    out["cos"] = cos
    out["angle_deg"] = angle_deg

    keep = (
        np.isfinite(out["fh_mag"].to_numpy(dtype=float))
        & np.isfinite(out["fr_mag"].to_numpy(dtype=float))
        & np.isfinite(out["cos"].to_numpy(dtype=float))
        & np.isfinite(out["angle_deg"].to_numpy(dtype=float))
        & (out["fh_mag"].to_numpy(dtype=float) >= float(eps))
        & (out["fr_mag"].to_numpy(dtype=float) >= float(eps))
    )
    out = out.loc[keep].reset_index(drop=True)

    out = _ensure_participant_column(out)
    out = _uniform_downsample_per_episode(out, CONFIG["max_points_per_episode"])
    return out


def _density_hist(values: np.ndarray, bins: int, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    hist, edges = np.histogram(values, bins=bins, range=(lo, hi), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist


def _plot_distribution_hist(
    ax: plt.Axes,
    values: np.ndarray,
    bins: int,
    xlim: tuple[float, float],
    xlabel: str,
    title: str,
) -> None:
    color_fill = "#8fb3d9"
    color_line = "#1f4f82"

    vals = values[np.isfinite(values)]
    if vals.size > 0:
        ax.hist(
            vals,
            bins=bins,
            range=xlim,
            density=True,
            alpha=0.35,
            color=color_fill,
            edgecolor="white",
            linewidth=0.4,
            label="Distribution",
        )
        c, h = _density_hist(vals, bins=bins, lo=xlim[0], hi=xlim[1])
        ax.plot(c, h, color=color_line, linewidth=2.4, label="Density profile")

        mean_v = float(np.nanmean(vals))
        med_v = float(np.nanmedian(vals))
        ax.axvline(mean_v, color="#b91c1c", linestyle="-", linewidth=1.4, alpha=0.9, label="Mean")
        ax.axvline(med_v, color="#7c3aed", linestyle="--", linewidth=1.4, alpha=0.9, label="Median")

    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.35)


def _facet_grid(n: int) -> tuple[plt.Figure, np.ndarray]:
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), sharex=True, sharey=True)
    return fig, np.atleast_1d(axes).ravel()


def _save_fig_if_needed(fig: plt.Figure, outdir: str | None, stem: str) -> dict[str, str]:
    paths: dict[str, str] = {}
    if not outdir:
        return paths
    os.makedirs(outdir, exist_ok=True)
    png_path = os.path.join(outdir, f"{stem}.png")
    pdf_path = os.path.join(outdir, f"{stem}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    paths["png"] = png_path
    paths["pdf"] = pdf_path
    return paths


def plot_first_trial_cosine(
    df2: pd.DataFrame,
    participant: str = "Participant_A",
    outdir: str | None = None,
) -> dict[str, Any]:
    """Plot cosine similarity over time for the first trial (episode) of one participant."""
    required = {"cos"}
    missing = sorted(list(required - set(df2.columns)))
    if missing:
        raise KeyError(f"Missing required columns for plotting: {missing}")
    if df2.empty:
        raise ValueError("No rows available for plotting after filtering.")

    d = df2.copy()
    if "participant" in d.columns:
        pcol = d["participant"].astype(str)
        d = d[pcol.str.startswith(participant)]
    if d.empty:
        raise ValueError(f"No rows found for participant prefix: {participant}")

    if "episode" in d.columns:
        first_ep = int(np.nanmin(pd.to_numeric(d["episode"], errors="coerce").to_numpy(dtype=float)))
        d = d[pd.to_numeric(d["episode"], errors="coerce") == first_ep].copy()
    else:
        first_ep = 1
    if d.empty:
        raise ValueError("No rows left for first episode after filtering.")

    if "t" in d.columns:
        d = d.sort_values("t")
        x = pd.to_numeric(d["t"], errors="coerce").to_numpy(dtype=float)
        xlabel = "Time Step (t)"
    else:
        d = d.reset_index(drop=True)
        x = np.arange(len(d), dtype=float)
        xlabel = "Sample Index"

    y = pd.to_numeric(d["cos"], errors="coerce").to_numpy(dtype=float)
    fh = pd.to_numeric(d["fh_mag"], errors="coerce").to_numpy(dtype=float) if "fh_mag" in d.columns else np.full_like(y, np.nan)
    fr = pd.to_numeric(d["fr_mag"], errors="coerce").to_numpy(dtype=float) if "fr_mag" in d.columns else np.full_like(y, np.nan)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    fh = fh[keep]
    fr = fr[keep]
    if x.size == 0:
        raise ValueError("No finite cosine samples to plot for first trial.")

    # Winsorize to suppress isolated spikes before smoothing.
    q_lo, q_hi = CONFIG["winsor_quantiles"]
    lo = float(np.nanquantile(y, q_lo))
    hi = float(np.nanquantile(y, q_hi))
    y_clip = np.clip(y, lo, hi)

    # Robust smoothing.
    smooth_window = int(CONFIG["smooth_window"])
    smooth_window = max(3, smooth_window if smooth_window % 2 == 1 else smooth_window + 1)
    y_smooth = pd.Series(y_clip).rolling(window=smooth_window, min_periods=1, center=True).median().to_numpy(dtype=float)

    # Step-bin medians for an even cleaner trend estimate.
    bin_size = int(CONFIG["bin_size_steps"])
    bin_size = max(2, bin_size)
    x_idx = np.arange(x.size, dtype=int)
    bin_id = x_idx // bin_size
    binned = (
        pd.DataFrame({"bin_id": bin_id, "x": x, "y": y_clip})
        .groupby("bin_id", as_index=False)
        .agg(x_bin=("x", "median"), y_bin=("y", "median"))
    )

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(11.5, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    ax_top.plot(x, y, color="#94a3b8", linewidth=0.9, alpha=0.35, label="Raw cosine")
    ax_top.plot(x, y_smooth, color="#1f4f82", linewidth=2.2, alpha=0.95, label=f"Rolling median ({smooth_window})")
    ax_top.plot(
        binned["x_bin"].to_numpy(dtype=float),
        binned["y_bin"].to_numpy(dtype=float),
        color="#b45309",
        linewidth=2.0,
        marker="o",
        markersize=3.0,
        alpha=0.95,
        label=f"Step-bin median ({bin_size})",
    )
    ax_top.axhline(1.0, color="#065f46", linestyle=":", linewidth=1.0, alpha=0.9)
    ax_top.axhline(0.0, color="#111827", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_top.axhline(-1.0, color="#7f1d1d", linestyle=":", linewidth=1.0, alpha=0.9)
    ax_top.set_ylim(-1.05, 1.05)
    ax_top.set_ylabel("Cosine Similarity")
    ax_top.set_title(f"{participant} - First Trial (Episode {first_ep}) Force Alignment")
    ax_top.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.35)
    ax_top.legend(loc="upper right", frameon=False)

    if np.any(np.isfinite(fh)):
        ax_bot.plot(x, fh, color="#0f766e", linewidth=1.3, alpha=0.9, label="|Fh|")
    if np.any(np.isfinite(fr)):
        ax_bot.plot(x, fr, color="#9333ea", linewidth=1.3, alpha=0.9, label="|Fr|")
    ax_bot.axhline(float(CONFIG["eps"]), color="#111827", linestyle="--", linewidth=1.0, alpha=0.7, label=f"eps={CONFIG['eps']:.2g}")
    ax_bot.set_xlabel(xlabel)
    ax_bot.set_ylabel("Force Magnitude")
    ax_bot.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.35)
    ax_bot.legend(loc="upper right", frameon=False, ncol=3)
    fig.tight_layout()

    saved_paths = _save_fig_if_needed(fig, outdir, "participantA_first_trial_cosine")
    caption = (
        f"Cosine similarity between human and robot force vectors over time for {participant}, "
        f"first trial (episode {first_ep}), n={x.size} samples. "
        f"Shown with rolling-median smoothing (window={smooth_window}) and step-bin medians (bin={bin_size})."
    )
    return {"figures": {"first_trial_cosine": fig}, "caption": caption, "saved_paths": {"first_trial_cosine": saved_paths}}


def plot_force_alignment(
    df2: pd.DataFrame,
    facet_by_participant: bool = False,
    outdir: str | None = None,
) -> dict[str, Any]:
    """Create thesis-quality force alignment figures and return figure handles + caption.

    Figure A:
      Cosine similarity distribution in [-1, 1].
    Figure B:
      Alignment angle distribution in [0, 180] degrees.
    """
    required = {"cos", "angle_deg", "participant"}
    missing = sorted(list(required - set(df2.columns)))
    if missing:
        raise KeyError(f"Missing required columns for plotting: {missing}")
    if df2.empty:
        raise ValueError("No rows available for plotting after filtering.")

    bins = int(CONFIG["bins"])
    dfp = df2.copy()

    figures: dict[str, plt.Figure] = {}
    saved_paths: dict[str, dict[str, str]] = {}

    # Figure A: cosine similarity
    if facet_by_participant:
        participants = sorted(dfp["participant"].dropna().astype(str).unique().tolist())
        fig_cos, axes = _facet_grid(len(participants))
        for ax, p in zip(axes, participants):
            d = dfp[dfp["participant"].astype(str) == p]
            x = d["cos"].to_numpy(dtype=float)
            _plot_distribution_hist(
                ax=ax,
                values=x,
                bins=bins,
                xlim=(-1.0, 1.0),
                xlabel="Force Alignment Cosine Similarity",
                title=str(p),
            )
            ax.axvline(0.0, color="#111827", linestyle="--", linewidth=1.0, alpha=0.7)
            ax.axvline(1.0, color="#065f46", linestyle=":", linewidth=1.0, alpha=0.9)
        for ax in axes[len(participants):]:
            ax.set_visible(False)
        fig_cos.suptitle("Human-Robot Force Alignment Distribution (Cosine Similarity)", y=1.02)
    else:
        fig_cos, ax = plt.subplots(1, 1, figsize=(10, 5.8))
        x = dfp["cos"].to_numpy(dtype=float)
        _plot_distribution_hist(
            ax=ax,
            values=x,
            bins=bins,
            xlim=(-1.0, 1.0),
            xlabel="Force Alignment Cosine Similarity",
            title="Human-Robot Force Alignment Distribution (Cosine Similarity)",
        )
        ax.axvline(0.0, color="#111827", linestyle="--", linewidth=1.0, alpha=0.7, label="cos = 0")
        ax.axvline(1.0, color="#065f46", linestyle=":", linewidth=1.0, alpha=0.9, label="cos = 1")
        ax.legend(loc="upper left", frameon=False, ncol=2)
    fig_cos.tight_layout()
    figures["cosine_similarity"] = fig_cos
    saved_paths["cosine_similarity"] = _save_fig_if_needed(fig_cos, outdir, "force_alignment_cosine")

    # Figure B: angle in degrees
    if facet_by_participant:
        participants = sorted(dfp["participant"].dropna().astype(str).unique().tolist())
        fig_ang, axes = _facet_grid(len(participants))
        for ax, p in zip(axes, participants):
            d = dfp[dfp["participant"].astype(str) == p]
            x = d["angle_deg"].to_numpy(dtype=float)
            _plot_distribution_hist(
                ax=ax,
                values=x,
                bins=bins,
                xlim=(0.0, 180.0),
                xlabel="Alignment Angle (degrees)",
                title=str(p),
            )
            ymax = ax.get_ylim()[1]
            ax.text(5.0, 0.92 * ymax, "Aligned", color="#065f46", fontsize=9, va="top")
            ax.text(175.0, 0.92 * ymax, "Opposed", color="#7f1d1d", fontsize=9, ha="right", va="top")
        for ax in axes[len(participants):]:
            ax.set_visible(False)
        fig_ang.suptitle("Human-Robot Force Alignment Distribution (Angle)", y=1.02)
    else:
        fig_ang, ax = plt.subplots(1, 1, figsize=(10, 5.8))
        x = dfp["angle_deg"].to_numpy(dtype=float)
        _plot_distribution_hist(
            ax=ax,
            values=x,
            bins=bins,
            xlim=(0.0, 180.0),
            xlabel="Alignment Angle (degrees)",
            title="Human-Robot Force Alignment Distribution (Angle)",
        )
        ymax = ax.get_ylim()[1]
        ax.text(5.0, 0.92 * ymax, "Aligned", color="#065f46", fontsize=10, va="top")
        ax.text(175.0, 0.92 * ymax, "Opposed", color="#7f1d1d", fontsize=10, ha="right", va="top")
        ax.legend(loc="upper center", frameon=False, ncol=3)
    fig_ang.tight_layout()
    figures["alignment_angle"] = fig_ang
    saved_paths["alignment_angle"] = _save_fig_if_needed(fig_ang, outdir, "force_alignment_angle")

    n = len(dfp)
    caption = (
        f"Force alignment distributions over {n} valid samples "
        f"(eps={CONFIG['eps']:.1e}, bins={bins}). "
        "Higher cosine and lower angle indicate stronger human-robot force alignment."
    )
    return {"figures": figures, "caption": caption, "saved_paths": saved_paths}


def main() -> None:
    try:
        steps_dir = resolve_steps_dir(CONFIG)
        df = _load_steps_dataframe(steps_dir)
    except Exception as exc:
        print(f"Error: {exc}")
        return

    participant_name = "_".join(str(CONFIG["base_dir"]).split("_")[:2]) or "pooled"
    df = _ensure_participant_column(df, default_participant=participant_name)
    df2 = compute_force_alignment(df, eps=float(CONFIG["eps"]))
    result = plot_first_trial_cosine(df2, participant=participant_name, outdir=CONFIG["outdir"])
    print(result["caption"])
    if not CONFIG["outdir"]:
        plt.show()


if __name__ == "__main__":
    main()
