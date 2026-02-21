#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


DATA_JSON = r'''
{
  "active_negotiations": [
    {
      "participant": "A",
      "rows": [
        {"phase_context": "Open", "total": 0, "yield_h": 0, "yield_r": 0, "yield_n": 0},
        {"phase_context": "Obstacle", "total": 1, "yield_h": 1, "yield_r": 0, "yield_n": 0},
        {"phase_context": "Goal", "total": 1, "yield_h": 0, "yield_r": 1, "yield_n": 0},
        {"phase_context": "Open -> Obstacle", "total": 7, "yield_h": 1, "yield_r": 6, "yield_n": 0},
        {"phase_context": "Obstacle -> Goal", "total": 2, "yield_h": 2, "yield_r": 0, "yield_n": 0},
        {"phase_context": "Collision Obstacle", "total": 0, "yield_h": 0, "yield_r": 0, "yield_n": 0}
      ],
      "total": {"total": 11, "yield_h": 4, "yield_r": 7, "yield_n": 0}
    },
    {
      "participant": "B",
      "rows": [
        {"phase_context": "Open", "total": 0, "yield_h": 0, "yield_r": 0, "yield_n": 0},
        {"phase_context": "Obstacle", "total": 3, "yield_h": 0, "yield_r": 2, "yield_n": 1},
        {"phase_context": "Goal", "total": 0, "yield_h": 0, "yield_r": 0, "yield_n": 0},
        {"phase_context": "Open -> Obstacle", "total": 3, "yield_h": 2, "yield_r": 1, "yield_n": 0},
        {"phase_context": "Obstacle -> Goal", "total": 2, "yield_h": 2, "yield_r": 0, "yield_n": 0},
        {"phase_context": "Collision Obstacle", "total": 3, "yield_h": 1, "yield_r": 0, "yield_n": 2}
      ],
      "total": {"total": 11, "yield_h": 5, "yield_r": 3, "yield_n": 3}
    },
    {
      "participant": "C",
      "rows": [
        {"phase_context": "Open", "total": 0, "yield_h": 0, "yield_r": 0, "yield_n": 0},
        {"phase_context": "Obstacle", "total": 1, "yield_h": 0, "yield_r": 1, "yield_n": 0},
        {"phase_context": "Goal", "total": 0, "yield_h": 0, "yield_r": 0, "yield_n": 0},
        {"phase_context": "Open -> Obstacle", "total": 1, "yield_h": 0, "yield_r": 1, "yield_n": 0},
        {"phase_context": "Obstacle -> Goal", "total": 0, "yield_h": 0, "yield_r": 0, "yield_n": 0},
        {"phase_context": "Collision Obstacle", "total": 3, "yield_h": 0, "yield_r": 2, "yield_n": 1}
      ],
      "total": {"total": 5, "yield_h": 0, "yield_r": 4, "yield_n": 1}
    },
    {
      "participant": "D",
      "rows": [
        {"phase_context": "Open", "total": 4, "yield_h": 0, "yield_r": 4, "yield_n": 0},
        {"phase_context": "Obstacle", "total": 1, "yield_h": 0, "yield_r": 0, "yield_n": 1},
        {"phase_context": "Goal", "total": 5, "yield_h": 5, "yield_r": 0, "yield_n": 0},
        {"phase_context": "Open -> Obstacle", "total": 0, "yield_h": 0, "yield_r": 0, "yield_n": 0},
        {"phase_context": "Obstacle -> Goal", "total": 2, "yield_h": 1, "yield_r": 1, "yield_n": 0},
        {"phase_context": "Collision Obstacle", "total": 1, "yield_h": 0, "yield_r": 0, "yield_n": 1}
      ],
      "total": {"total": 13, "yield_h": 6, "yield_r": 5, "yield_n": 2}
    },
    {
      "participant": "E",
      "rows": [
        {"phase_context": "Open", "total": 2, "yield_h": 1, "yield_r": 1, "yield_n": 0},
        {"phase_context": "Obstacle", "total": 1, "yield_h": 0, "yield_r": 1, "yield_n": 0},
        {"phase_context": "Goal", "total": 5, "yield_h": 3, "yield_r": 2, "yield_n": 0},
        {"phase_context": "Open -> Obstacle", "total": 3, "yield_h": 1, "yield_r": 2, "yield_n": 0},
        {"phase_context": "Obstacle -> Goal", "total": 2, "yield_h": 0, "yield_r": 2, "yield_n": 0},
        {"phase_context": "Collision Obstacle", "total": 0, "yield_h": 0, "yield_r": 0, "yield_n": 0}
      ],
      "total": {"total": 13, "yield_h": 5, "yield_r": 8, "yield_n": 0}
    },
    {
      "participant": "F",
      "rows": [
        {"phase_context": "Open", "total": 1, "yield_h": 0, "yield_r": 1, "yield_n": 0},
        {"phase_context": "Obstacle", "total": 3, "yield_h": 1, "yield_r": 2, "yield_n": 0},
        {"phase_context": "Goal", "total": 4, "yield_h": 4, "yield_r": 0, "yield_n": 0},
        {"phase_context": "Open -> Obstacle", "total": 3, "yield_h": 0, "yield_r": 3, "yield_n": 0},
        {"phase_context": "Obstacle -> Goal", "total": 0, "yield_h": 0, "yield_r": 0, "yield_n": 0},
        {"phase_context": "Collision Obstacle", "total": 0, "yield_h": 0, "yield_r": 0, "yield_n": 0}
      ],
      "total": {"total": 11, "yield_h": 5, "yield_r": 6, "yield_n": 0}
    }
  ],
  "sudden_adaptations": [
    {
      "participant": "A",
      "rows": [
        {"phase_context": "Open", "total": 0, "init_h": 0, "init_r": 0, "init_n": 0},
        {"phase_context": "Obstacle", "total": 2, "init_h": 2, "init_r": 0, "init_n": 0},
        {"phase_context": "Goal", "total": 4, "init_h": 1, "init_r": 3, "init_n": 0},
        {"phase_context": "Open -> Obstacle", "total": 0, "init_h": 0, "init_r": 0, "init_n": 0},
        {"phase_context": "Obstacle -> Goal", "total": 0, "init_h": 0, "init_r": 0, "init_n": 0},
        {"phase_context": "Collision Obstacle", "total": 0, "init_h": 0, "init_r": 0, "init_n": 0},
        {"phase_context": "Strategy change", "total": 3, "init_h": 1, "init_r": 2, "init_n": 0}
      ],
      "total": {"total": 9, "init_h": 4, "init_r": 5, "init_n": 0}
    },
    {
      "participant": "B",
      "rows": [
        {"phase_context": "Open", "total": 0, "init_h": 0, "init_r": 0, "init_n": 0},
        {"phase_context": "Obstacle", "total": 1, "init_h": 0, "init_r": 1, "init_n": 0},
        {"phase_context": "Goal", "total": 4, "init_h": 1, "init_r": 3, "init_n": 0},
        {"phase_context": "Open -> Obstacle", "total": 0, "init_h": 0, "init_r": 0, "init_n": 0},
        {"phase_context": "Obstacle -> Goal", "total": 0, "init_h": 0, "init_r": 0, "init_n": 0},
        {"phase_context": "Collision Obstacle", "total": 5, "init_h": 1, "init_r": 2, "init_n": 2},
        {"phase_context": "Strategy change", "total": 1, "init_h": 0, "init_r": 1, "init_n": 0}
      ],
      "total": {"total": 11, "init_h": 2, "init_r": 7, "init_n": 2}
    },
    {
      "participant": "C",
      "rows": [
        {"phase_context": "Open", "total": 0, "init_h": 0, "init_r": 0, "init_n": 0},
        {"phase_context": "Obstacle", "total": 0, "init_h": 0, "init_r": 0, "init_n": 0},
        {"phase_context": "Goal", "total": 0, "init_h": 0, "init_r": 0, "init_n": 0},
        {"phase_context": "Open -> Obstacle", "total": 1, "init_h": 0, "init_r": 1, "init_n": 0},
        {"phase_context": "Obstacle -> Goal", "total": 1, "init_h": 0, "init_r": 1, "init_n": 0},
        {"phase_context": "Collision Obstacle", "total": 2, "init_h": 1, "init_r": 1, "init_n": 0},
        {"phase_context": "Strategy change", "total": 1, "init_h": 1, "init_r": 0, "init_n": 0}
      ],
      "total": {"total": 5, "init_h": 2, "init_r": 3, "init_n": 0}
    },
    {
      "participant": "D",
      "rows": [
        {"phase_context": "Open", "total": 0, "init_h": 0, "init_r": 0, "init_n": 0},
        {"phase_context": "Obstacle", "total": 1, "init_h": 0, "init_r": 1, "init_n": 0},
        {"phase_context": "Goal", "total": 4, "init_h": 1, "init_r": 3, "init_n": 0},
        {"phase_context": "Open -> Obstacle", "total": 0, "init_h": 0, "init_r": 0, "init_n": 0},
        {"phase_context": "Obstacle -> Goal", "total": 2, "init_h": 0, "init_r": 2, "init_n": 0},
        {"phase_context": "Collision Obstacle", "total": 0, "init_h": 0, "init_r": 0, "init_n": 0},
        {"phase_context": "Strategy change", "total": 1, "init_h": 1, "init_r": 0, "init_n": 0}
      ],
      "total": {"total": 8, "init_h": 2, "init_r": 6, "init_n": 0}
    },
    {
      "participant": "E",
      "rows": [
        {"phase_context": "Open", "total": 1, "init_h": 1, "init_r": 0, "init_n": 0},
        {"phase_context": "Obstacle", "total": 4, "init_h": 1, "init_r": 3, "init_n": 0},
        {"phase_context": "Goal", "total": 4, "init_h": 4, "init_r": 0, "init_n": 0},
        {"phase_context": "Open -> Obstacle", "total": 3, "init_h": 1, "init_r": 1, "init_n": 1},
        {"phase_context": "Obstacle -> Goal", "total": 2, "init_h": 1, "init_r": 0, "init_n": 1},
        {"phase_context": "Collision Obstacle", "total": 0, "init_h": 0, "init_r": 0, "init_n": 0},
        {"phase_context": "Strategy change", "total": 0, "init_h": 0, "init_r": 0, "init_n": 0}
      ],
      "total": {"total": 14, "init_h": 8, "init_r": 4, "init_n": 2}
    },
    {
      "participant": "F",
      "rows": [
        {"phase_context": "Open", "total": 0, "init_h": 0, "init_r": 0, "init_n": 0},
        {"phase_context": "Obstacle", "total": 3, "init_h": 1, "init_r": 2, "init_n": 0},
        {"phase_context": "Goal", "total": 1, "init_h": 0, "init_r": 1, "init_n": 0},
        {"phase_context": "Open -> Obstacle", "total": 0, "init_h": 0, "init_r": 0, "init_n": 0},
        {"phase_context": "Obstacle -> Goal", "total": 0, "init_h": 0, "init_r": 0, "init_n": 0},
        {"phase_context": "Collision Obstacle", "total": 0, "init_h": 0, "init_r": 0, "init_n": 0},
        {"phase_context": "Strategy change", "total": 0, "init_h": 0, "init_r": 0, "init_n": 0}
      ],
      "total": {"total": 4, "init_h": 1, "init_r": 3, "init_n": 0}
    }
  ]
}
'''

# Keep existing fig1/fig2 and optionally add an alternative format.
SHOW_ALTERNATIVE_FORMAT = False
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


def _records(section: list[dict], human_key: str, robot_key: str, neutral_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    totals = []
    for p in section:
        participant = p["participant"]
        t = p["total"]
        totals.append(
            {
                "participant": participant,
                "total": int(t["total"]),
                "human": int(t[human_key]),
                "robot": int(t[robot_key]),
                "neutral": int(t[neutral_key]),
            }
        )
        for r in p["rows"]:
            total = int(r["total"])
            h = int(r[human_key])
            rb = int(r[robot_key])
            n = int(r[neutral_key])
            rows.append(
                {
                    "participant": participant,
                    "phase_context": r["phase_context"],
                    "total": total,
                    "human": h,
                    "robot": rb,
                    "neutral": n,
                    "human_rate": (h / total) if total > 0 else np.nan,
                    "robot_rate": (rb / total) if total > 0 else np.nan,
                    "bias_h_minus_r": ((h - rb) / total) if total > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(totals)


def _plot_stacked(
    ax: plt.Axes,
    totals: pd.DataFrame,
    title: str,
    human_label: str = "Human",
    robot_label: str = "Robot",
    neutral_label: str = "Neutral",
) -> None:
    d = totals.sort_values("participant")
    p = d["participant"].to_numpy()
    t = d["total"].replace(0, np.nan).to_numpy(dtype=float)
    h = (d["human"].to_numpy(dtype=float) / t)
    r = (d["robot"].to_numpy(dtype=float) / t)
    n = (d["neutral"].to_numpy(dtype=float) / t)

    x = np.arange(len(p))
    ax.bar(x, h, color="#1f77b4", label=human_label)
    ax.bar(x, r, bottom=h, color="#ff7f0e", label=robot_label)
    ax.bar(x, n, bottom=h + r, color="#7f7f7f", label=neutral_label)
    ax.set_xticks(x)
    ax.set_xticklabels(p)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Fraction")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)


def _plot_heatmap(ax: plt.Axes, detail: pd.DataFrame, title: str) -> None:
    piv = detail.pivot(index="participant", columns="phase_context", values="bias_h_minus_r")
    piv = piv.reindex(sorted(piv.index), axis=0)
    mat = piv.to_numpy(dtype=float)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=30, ha="right")
    ax.set_title(title)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("(Human - Robot) / Total")


def _plot_totals(ax: plt.Axes, active_negotiations_totals: pd.DataFrame, sudden_adaptations_totals: pd.DataFrame) -> None:
    p = sorted(set(active_negotiations_totals["participant"]).union(set(sudden_adaptations_totals["participant"])))
    x = np.arange(len(p))
    p_map = dict(zip(active_negotiations_totals["participant"], active_negotiations_totals["total"]))
    m_map = dict(zip(sudden_adaptations_totals["participant"], sudden_adaptations_totals["total"]))
    y_active_negotiations = np.array([p_map.get(pp, 0) for pp in p], dtype=float)
    y_sudden_adaptations = np.array([m_map.get(pp, 0) for pp in p], dtype=float)

    ax.plot(x, y_active_negotiations, marker="o", lw=2, color="#4c78a8", label="Active Negotiations")
    ax.plot(x, y_sudden_adaptations, marker="s", lw=2, color="#f58518", label="Sudden Adaptations")
    ax.set_xticks(x)
    ax.set_xticklabels(p)
    ax.set_ylabel("Total Count")
    ax.set_title("Event Totals by Participant")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")


def _plot_phase_totals(ax: plt.Axes, detail: pd.DataFrame, title: str) -> None:
    g = detail.groupby("phase_context", as_index=False)[["human", "robot", "neutral"]].sum()
    g = g.sort_values("phase_context")
    x = np.arange(len(g))
    w = 0.26
    ax.bar(x - w, g["human"], width=w, color="#1f77b4", label="Human")
    ax.bar(x, g["robot"], width=w, color="#ff7f0e", label="Robot")
    ax.bar(x + w, g["neutral"], width=w, color="#7f7f7f", label="Neutral")
    ax.set_xticks(x)
    ax.set_xticklabels(g["phase_context"], rotation=30, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)


def _plot_task_vertical_stacked(detail: pd.DataFrame, title: str, human_label: str, robot_label: str) -> None:
    d = detail.copy()
    participants = sorted(d["participant"].unique())
    phases = sorted(d["phase_context"].unique())

    phase_colors = {
        ph: plt.cm.tab20(i % 20) for i, ph in enumerate(phases)
    }
    x = np.arange(len(participants), dtype=float)
    w = 0.32

    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    bottom_h = np.zeros(len(participants), dtype=float)
    bottom_r = np.zeros(len(participants), dtype=float)

    for ph in phases:
        vals_h = []
        vals_r = []
        for p in participants:
            sub = d[(d["participant"] == p) & (d["phase_context"] == ph)]
            if sub.empty:
                vals_h.append(0.0)
                vals_r.append(0.0)
            else:
                vals_h.append(float(sub["human"].iloc[0]))
                vals_r.append(float(sub["robot"].iloc[0]))
        vals_h = np.array(vals_h, dtype=float)
        vals_r = np.array(vals_r, dtype=float)
        ax.bar(
            x - w / 2.0,
            vals_h,
            width=w,
            bottom=bottom_h,
            color=phase_colors[ph],
            edgecolor="#1f2937",
            linewidth=0.8,
            label=ph,
        )
        ax.bar(
            x + w / 2.0,
            vals_r,
            width=w,
            bottom=bottom_r,
            color=phase_colors[ph],
            edgecolor="#1f2937",
            linewidth=0.8,
        )
        bottom_h += vals_h
        bottom_r += vals_r

    # Annotate totals above both bars.
    for i, p in enumerate(participants):
        ax.text(x[i] - w / 2.0, bottom_h[i] + 0.2, f"{int(bottom_h[i])}", ha="center", va="bottom", fontsize=8)
        ax.text(x[i] + w / 2.0, bottom_r[i] + 0.2, f"{int(bottom_r[i])}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(participants)
    ax.set_xlim(-0.6, len(participants) - 0.4)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)

    # Secondary x labels for left/right bars.
    for i in range(len(participants)):
        ax.text(x[i] - w / 2.0, -0.08, human_label, ha="center", va="top", fontsize=8, transform=ax.get_xaxis_transform())
        ax.text(x[i] + w / 2.0, -0.08, robot_label, ha="center", va="top", fontsize=8, transform=ax.get_xaxis_transform())

    ax.legend(title="Phase", loc="upper right", bbox_to_anchor=(1.28, 1.0))
    plt.tight_layout()


def _plot_participant_dominance_compare(
    ax: plt.Axes, active_negotiations_totals: pd.DataFrame, sudden_adaptations_totals: pd.DataFrame
) -> None:
    p = sorted(set(active_negotiations_totals["participant"]).union(set(sudden_adaptations_totals["participant"])))
    x = np.arange(len(p), dtype=float)
    w = 0.36

    p_map = active_negotiations_totals.set_index("participant")
    m_map = sudden_adaptations_totals.set_index("participant")

    p_balance = []
    m_balance = []
    for pp in p:
        if pp in p_map.index and p_map.loc[pp, "total"] > 0:
            # Active Negotiations encode who yielded: (human_yield - robot_yield) / total_active_negotiations
            p_balance.append((p_map.loc[pp, "human"] - p_map.loc[pp, "robot"]) / p_map.loc[pp, "total"])
        else:
            p_balance.append(np.nan)
        if pp in m_map.index and m_map.loc[pp, "total"] > 0:
            # Sudden Adaptations encodes who initiated: (human_init - robot_init) / total_sudden_adaptations
            m_balance.append((m_map.loc[pp, "human"] - m_map.loc[pp, "robot"]) / m_map.loc[pp, "total"])
        else:
            m_balance.append(np.nan)

    p_balance = np.array(p_balance, dtype=float)
    m_balance = np.array(m_balance, dtype=float)

    ax.bar(
        x - w / 2.0,
        p_balance,
        width=w,
        color="#2563eb",
        alpha=0.85,
        label="Yield balance (Active Negotiations)",
    )
    ax.bar(
        x + w / 2.0,
        m_balance,
        width=w,
        color="#d97706",
        alpha=0.85,
        label="Initiation balance (Sudden Adaptations)",
    )
    ax.axhline(0, color="#666666", lw=1.0, ls="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(p)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("Balance (Human - Robot) / Total")
    ax.set_title("Participant Balance by Concept")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")


def _plot_human_share_scatter(
    ax: plt.Axes,
    active_negotiations_totals: pd.DataFrame,
    sudden_adaptations_totals: pd.DataFrame,
    show_regions: bool = False,
) -> None:
    merged = active_negotiations_totals.merge(
        sudden_adaptations_totals, on="participant", how="inner", suffixes=("_active_negotiations", "_sudden_adaptations")
    )
    if merged.empty:
        return
    p_share = merged["human_active_negotiations"].to_numpy(dtype=float) / merged["total_active_negotiations"].replace(0, np.nan).to_numpy(dtype=float)
    m_share = merged["human_sudden_adaptations"].to_numpy(dtype=float) / merged["total_sudden_adaptations"].replace(0, np.nan).to_numpy(dtype=float)

    ax.scatter(p_share, m_share, s=80, color="#0f766e", alpha=0.85)
    for _, row in merged.iterrows():
        xp = row["human_active_negotiations"] / row["total_active_negotiations"] if row["total_active_negotiations"] > 0 else np.nan
        yp = row["human_sudden_adaptations"] / row["total_sudden_adaptations"] if row["total_sudden_adaptations"] > 0 else np.nan
        if np.isfinite(xp) and np.isfinite(yp):
            ax.text(xp + 0.01, yp + 0.01, row["participant"], fontsize=9)

    ax.plot([0, 1], [0, 1], ls="--", lw=1.0, color="#6b7280")
    if show_regions:
        # Quadrant backgrounds (x: human-yield share, y: human-initiation share)
        ax.axvspan(0.00, 0.50, ymin=0.50, ymax=1.00, color="#d1fae5", alpha=0.35, zorder=0)  # upper-left
        ax.axvspan(0.50, 1.00, ymin=0.50, ymax=1.00, color="#fef3c7", alpha=0.35, zorder=0)  # upper-right
        ax.axvspan(0.00, 0.50, ymin=0.00, ymax=0.50, color="#fee2e2", alpha=0.35, zorder=0)  # lower-left
        ax.axvspan(0.50, 1.00, ymin=0.00, ymax=0.50, color="#dbeafe", alpha=0.35, zorder=0)  # lower-right

        ax.axvline(0.5, color="#6b7280", lw=1.0, ls=":")
        ax.axhline(0.5, color="#6b7280", lw=1.0, ls=":")
        ax.text(
            0.25, 0.88, "Human-Led\n(high initiation, low yielding)",
            ha="center", va="center", fontsize=9, weight="bold", color="#1f2937"
        )
        ax.text(
            0.75, 0.88, "Adaptive Assertor\n(high initiation, high yielding)",
            ha="center", va="center", fontsize=9, weight="bold", color="#1f2937"
        )
        ax.text(
            0.25, 0.12, "Fixed Path Style\n(low initiation, low yielding)",
            ha="center", va="center", fontsize=8, weight="bold", color="#1f2937"
        )
        ax.text(
            0.75, 0.12, "Robot-Led\n(low initiation, high yielding)",
            ha="center", va="center", fontsize=9, weight="bold", color="#1f2937"
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Human-Yield Share in Active Negotiations")
    ax.set_ylabel("Human-Initiation Share in Sudden Adaptations")
    ax.set_title("Participant Consistency: Yielding vs Initiation")
    ax.grid(alpha=0.2)


def _plot_phase_count_heatmap(ax: plt.Axes, detail: pd.DataFrame, title: str) -> None:
    tick_label_size = float(PLOT_STYLE_CFG.get("participants_grid_tick_label_size", 13))
    phase_order = list(dict.fromkeys(detail["phase_context"].tolist()))
    piv = detail.pivot_table(
        index="participant",
        columns="phase_context",
        values="total",
        aggfunc="sum",
        fill_value=0,
    )
    piv = piv.reindex(sorted(piv.index), axis=0).reindex(columns=phase_order, fill_value=0)
    mat = piv.to_numpy(dtype=float)

    im = ax.imshow(mat, cmap="YlGnBu", aspect="auto")
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=30, ha="right")
    ax.set_title(title)

    vmax = np.nanmax(mat) if np.isfinite(mat).any() else 1.0
    threshold = 0.45 * vmax
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = int(mat[i, j])
            if v == 0:
                # De-emphasize empty cells so non-zero counts stand out.
                ax.text(j, i, "0", ha="center", va="center", fontsize=tick_label_size, color="#9ca3af", alpha=0.55)
            else:
                txt_color = "white" if v > threshold else "#111827"
                ax.text(j, i, str(v), ha="center", va="center", fontsize=tick_label_size, color=txt_color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axis_label_size = float(PLOT_STYLE_CFG.get("participants_grid_axis_label_size", 14))
    cbar.set_label("Event count", fontsize=axis_label_size)
    cbar.ax.tick_params(labelsize=tick_label_size)


def _plot_participant_composition(
    ax: plt.Axes,
    totals: pd.DataFrame,
    title: str,
    human_label: str,
    robot_label: str,
    neutral_label: str,
) -> None:
    legend_font_size = float(PLOT_STYLE_CFG.get("participants_grid_legend_font_size", 13))
    tick_label_size = float(PLOT_STYLE_CFG.get("participants_grid_tick_label_size", 13))
    d = totals.sort_values("participant")
    participants = d["participant"].to_numpy()
    total = d["total"].replace(0, np.nan).to_numpy(dtype=float)
    h = d["human"].to_numpy(dtype=float) / total
    r = d["robot"].to_numpy(dtype=float) / total
    n = d["neutral"].to_numpy(dtype=float) / total
    y = np.arange(len(participants))

    ax.barh(y, h, color="#1f77b4", label=human_label)
    ax.barh(y, r, left=h, color="#ff7f0e", label=robot_label)
    ax.barh(y, n, left=h + r, color="#7f7f7f", label=neutral_label)
    ax.set_yticks(y)
    ax.set_yticklabels(participants)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Participant share")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)

    # Show total sample size next to each bar.
    for i, t in enumerate(d["total"].to_numpy(dtype=int)):
        ax.text(1.02, i, f"N={t}", va="center", ha="left", fontsize=tick_label_size, color="#111827")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=3,
        frameon=False,
        fontsize=legend_font_size,
        borderaxespad=0.2,
    )


def _plot_task_overview(
    detail: pd.DataFrame,
    totals: pd.DataFrame,
    event_name: str,
    human_label: str,
    robot_label: str,
    neutral_label: str,
    composition_title: str | None = None,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 10.0), gridspec_kw={"height_ratios": [1.35, 1.0]})
    _plot_yielding_matrix_by_participant_zone(
        axes[0],
        detail,
        f"{event_name}: Participant x Zone (Human/Robot/Event count)",
    )
    _plot_participant_composition(
        axes[1],
        totals,
        composition_title or f"{event_name}: Who Contributed (Share per Participant)",
        human_label=human_label,
        robot_label=robot_label,
        neutral_label=neutral_label,
    )
    axis_label_size = float(PLOT_STYLE_CFG.get("participants_grid_axis_label_size", 14))
    tick_label_size = float(PLOT_STYLE_CFG.get("participants_grid_tick_label_size", 13))
    for ax in axes:
        ax.title.set_fontsize(tick_label_size)
        ax.xaxis.label.set_size(axis_label_size)
        ax.yaxis.label.set_size(axis_label_size)
        ax.tick_params(axis="both", labelsize=tick_label_size)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98], h_pad=2.0)


def _plot_phase_share(
    ax: plt.Axes,
    detail: pd.DataFrame,
    title: str,
    human_label: str,
    robot_label: str,
    neutral_label: str,
) -> None:
    phase_order = _ordered_phases_from_detail(detail)
    g = (
        detail.groupby("phase_context", as_index=False)[["human", "robot", "neutral", "total"]]
        .sum()
        .set_index("phase_context")
        .reindex(phase_order, fill_value=0)
        .reset_index()
    )
    total = g["total"].replace(0, np.nan).to_numpy(dtype=float)
    h = g["human"].to_numpy(dtype=float) / total
    r = g["robot"].to_numpy(dtype=float) / total
    n = g["neutral"].to_numpy(dtype=float) / total
    x = np.arange(len(g), dtype=float)

    ax.bar(x, h, color="#1f77b4", label=human_label)
    ax.bar(x, r, bottom=h, color="#ff7f0e", label=robot_label)
    ax.bar(x, n, bottom=h + r, color="#7f7f7f", label=neutral_label)
    ax.set_xticks(x)
    ax.set_xticklabels(g["phase_context"], rotation=30, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Share")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)

    tick_label_size = float(PLOT_STYLE_CFG.get("participants_grid_tick_label_size", 13))
    for i, t in enumerate(g["total"].to_numpy(dtype=int)):
        ax.text(i, 1.02, f"N={t}", ha="center", va="bottom", fontsize=tick_label_size, color="#111827")

    legend_font_size = float(PLOT_STYLE_CFG.get("participants_grid_legend_font_size", 13))
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=3,
        frameon=False,
        fontsize=legend_font_size,
        borderaxespad=0.2,
    )


def _plot_yielding_matrix_by_participant_zone(
    ax: plt.Axes,
    detail: pd.DataFrame,
    title: str,
) -> None:
    phase_order = _ordered_phases_from_detail(detail)
    participant_order = sorted(detail["participant"].unique().tolist())

    h_piv = detail.pivot_table(
        index="participant",
        columns="phase_context",
        values="human",
        aggfunc="sum",
        fill_value=0,
    ).reindex(index=participant_order, columns=phase_order, fill_value=0)
    r_piv = detail.pivot_table(
        index="participant",
        columns="phase_context",
        values="robot",
        aggfunc="sum",
        fill_value=0,
    ).reindex(index=participant_order, columns=phase_order, fill_value=0)
    t_piv = detail.pivot_table(
        index="participant",
        columns="phase_context",
        values="total",
        aggfunc="sum",
        fill_value=0,
    ).reindex(index=participant_order, columns=phase_order, fill_value=0)

    h = h_piv.to_numpy(dtype=float)
    r = r_piv.to_numpy(dtype=float)
    t = t_piv.to_numpy(dtype=float)

    im = ax.imshow(t, cmap="YlGnBu", aspect="auto")
    ax.set_yticks(np.arange(len(participant_order)))
    ax.set_yticklabels(participant_order)
    ax.set_xticks(np.arange(len(phase_order)))
    ax.set_xticklabels(phase_order, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_xlabel("Zone / Phase")
    ax.set_ylabel("Participant")

    tick_label_size = float(PLOT_STYLE_CFG.get("participants_grid_tick_label_size", 13))
    vmax = np.nanmax(t) if np.isfinite(t).any() else 1.0
    threshold = 0.45 * vmax
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            total_ij = int(t[i, j])
            human_ij = int(h[i, j])
            robot_ij = int(r[i, j])
            txt_color = "white" if total_ij > threshold else "#111827"
            if total_ij == 0:
                txt_color = "#6b7280"
            ax.text(
                j,
                i,
                f"{human_ij}/{robot_ij}/{total_ij}",
                ha="center",
                va="center",
                fontsize=tick_label_size,
                color=txt_color,
            )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axis_label_size = float(PLOT_STYLE_CFG.get("participants_grid_axis_label_size", 14))
    cbar.set_label("Event count", fontsize=axis_label_size)
    cbar.ax.tick_params(labelsize=tick_label_size)

    legend_font_size = float(PLOT_STYLE_CFG.get("participants_grid_legend_font_size", 13))
    guide = Line2D([], [], linestyle="None", label="Cell text = Human/Robot/Event count")
    ax.legend(
        handles=[guide],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        frameon=False,
        fontsize=legend_font_size,
        handlelength=0.0,
        handletextpad=0.2,
    )


def _plot_phase_share_comparison(
    active_negotiations_detail: pd.DataFrame,
    sudden_adaptations_detail: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 10.0))
    _plot_yielding_matrix_by_participant_zone(
        axes[0, 0],
        active_negotiations_detail,
        "Active Negotiations: Yielding Share by Participant x Zone",
    )
    _plot_phase_composition_matrix(
        axes[0, 1],
        active_negotiations_detail,
        "Active Negotiations: Participant x Phase (Yielding Share)",
        human_label="Human yielded",
        robot_label="Robot yielded",
        neutral_label="Unclear",
    )
    _plot_phase_share(
        axes[1, 0],
        sudden_adaptations_detail,
        "Sudden Adaptations: Who Initiated by Phase (Share)",
        human_label="Human initiated",
        robot_label="Robot initiated",
        neutral_label="Unclear",
    )
    _plot_phase_composition_matrix(
        axes[1, 1],
        sudden_adaptations_detail,
        "Sudden Adaptations: Participant x Phase (Initiating Share)",
        human_label="Human initiated",
        robot_label="Robot initiated",
        neutral_label="Unclear",
    )

    axis_label_size = float(PLOT_STYLE_CFG.get("participants_grid_axis_label_size", 14))
    tick_label_size = float(PLOT_STYLE_CFG.get("participants_grid_tick_label_size", 13))
    for ax in axes.ravel():
        ax.title.set_fontsize(tick_label_size)
        ax.xaxis.label.set_size(axis_label_size)
        ax.yaxis.label.set_size(axis_label_size)
        ax.tick_params(axis="both", labelsize=tick_label_size)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98], h_pad=2.0)


def _ordered_phases_from_detail(detail: pd.DataFrame) -> list[str]:
    canonical = [
        "Open",
        "Obstacle",
        "Goal",
        "Open -> Obstacle",
        "Obstacle -> Goal",
        "Collision Obstacle",
        "Strategy change",
    ]
    present = set(detail["phase_context"].tolist())
    ordered = [p for p in canonical if p in present]
    extras = [p for p in detail["phase_context"].unique().tolist() if p not in ordered]
    return ordered + extras


def _plot_phase_composition_matrix(
    ax: plt.Axes,
    detail: pd.DataFrame,
    title: str,
    human_label: str,
    robot_label: str,
    neutral_label: str,
) -> None:
    phase_order = _ordered_phases_from_detail(detail)
    participant_order = (
        detail.groupby("participant", as_index=False)["total"]
        .sum()
        .sort_values("total", ascending=False)["participant"]
        .tolist()
    )

    h_piv = detail.pivot_table(
        index="participant",
        columns="phase_context",
        values="human",
        aggfunc="sum",
        fill_value=0,
    ).reindex(index=participant_order, columns=phase_order, fill_value=0)
    r_piv = detail.pivot_table(
        index="participant",
        columns="phase_context",
        values="robot",
        aggfunc="sum",
        fill_value=0,
    ).reindex(index=participant_order, columns=phase_order, fill_value=0)
    n_piv = detail.pivot_table(
        index="participant",
        columns="phase_context",
        values="neutral",
        aggfunc="sum",
        fill_value=0,
    ).reindex(index=participant_order, columns=phase_order, fill_value=0)
    t_piv = detail.pivot_table(
        index="participant",
        columns="phase_context",
        values="total",
        aggfunc="sum",
        fill_value=0,
    ).reindex(index=participant_order, columns=phase_order, fill_value=0)

    h = h_piv.to_numpy(dtype=float)
    r = r_piv.to_numpy(dtype=float)
    n = n_piv.to_numpy(dtype=float)
    t = t_piv.to_numpy(dtype=float)

    # Add totals row/column so the same matrix supports per-participant and aggregate reading.
    h_ext = np.zeros((h.shape[0] + 1, h.shape[1] + 1), dtype=float)
    r_ext = np.zeros_like(h_ext)
    n_ext = np.zeros_like(h_ext)
    t_ext = np.zeros_like(h_ext)
    h_ext[:-1, :-1] = h
    r_ext[:-1, :-1] = r
    n_ext[:-1, :-1] = n
    t_ext[:-1, :-1] = t

    h_ext[:-1, -1] = h.sum(axis=1)
    r_ext[:-1, -1] = r.sum(axis=1)
    n_ext[:-1, -1] = n.sum(axis=1)
    t_ext[:-1, -1] = t.sum(axis=1)

    h_ext[-1, :-1] = h.sum(axis=0)
    r_ext[-1, :-1] = r.sum(axis=0)
    n_ext[-1, :-1] = n.sum(axis=0)
    t_ext[-1, :-1] = t.sum(axis=0)

    h_ext[-1, -1] = h.sum()
    r_ext[-1, -1] = r.sum()
    n_ext[-1, -1] = n.sum()
    t_ext[-1, -1] = t.sum()

    balance = np.divide(h_ext - r_ext, t_ext, out=np.full_like(h_ext, np.nan), where=t_ext > 0)

    x_labels = phase_order + ["Total"]
    y_labels = participant_order + ["Total"]

    im = ax.imshow(balance, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_xlabel(
        f"Phase (cell text = {human_label}/{robot_label}/{neutral_label}; T=total)"
    )
    ax.set_ylabel("Participant")

    t_int = t_ext.astype(int)
    for i in range(balance.shape[0]):
        for j in range(balance.shape[1]):
            if t_int[i, j] > 0:
                txt_color = "white" if abs(balance[i, j]) > 0.45 else "#111827"
                ax.text(
                    j,
                    i,
                    f"{int(h_ext[i, j])}/{int(r_ext[i, j])}/{int(n_ext[i, j])}\nT={t_int[i, j]}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color=txt_color,
                )
            else:
                ax.text(j, i, "-", ha="center", va="center", fontsize=7, color="#6b7280")

    # Emphasize totals row/column with heavier guide lines.
    ax.axhline(len(y_labels) - 1.5, color="#111827", lw=1.2)
    ax.axvline(len(x_labels) - 1.5, color="#111827", lw=1.2)
    ax.set_xticks(np.arange(-0.5, len(x_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8, alpha=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Balance: ({human_label} - {robot_label}) / T")


def _plot_participant_balance_lollipop(
    ax: plt.Axes,
    detail: pd.DataFrame,
    title: str,
    balance_label: str,
) -> None:
    g = detail.groupby("participant", as_index=False)[["total", "human", "robot"]].sum()
    g = g.assign(
        balance=np.where(g["total"] > 0, (g["human"] - g["robot"]) / g["total"], np.nan)
    ).sort_values("balance", ascending=False)

    y = np.arange(len(g))
    x = g["balance"].to_numpy(dtype=float)
    n = g["total"].to_numpy(dtype=float)
    sizes = 40 + 14 * np.sqrt(n)

    for i in range(len(g)):
        if np.isfinite(x[i]):
            ax.plot([0, x[i]], [y[i], y[i]], color="#9ca3af", lw=1.8, zorder=1)

    colors = np.where(x >= 0, "#1f77b4", "#ff7f0e")
    ax.scatter(x, y, s=sizes, c=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax.axvline(0, color="#6b7280", lw=1.0, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(g["participant"].tolist())
    ax.invert_yaxis()
    ax.set_xlim(-1, 1)
    ax.set_xlabel(balance_label)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)

    for i, t in enumerate(n.astype(int)):
        x_text = x[i] + (0.04 if x[i] >= 0 else -0.04)
        ha = "left" if x[i] >= 0 else "right"
        x_text = max(min(x_text, 0.96), -0.96)
        ax.text(x_text, y[i], f"N={t}", va="center", ha=ha, fontsize=8, color="#111827")


def _plot_task_alternative_format(
    detail: pd.DataFrame,
    _totals: pd.DataFrame,
    event_name: str,
    human_label: str,
    robot_label: str,
    neutral_label: str,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    _plot_phase_composition_matrix(
        ax,
        detail,
        f"{event_name}: Participant x Phase Composition Matrix",
        human_label=human_label,
        robot_label=robot_label,
        neutral_label=neutral_label,
    )
    fig.tight_layout()


def main() -> None:
    _configure_ieee_plot_style(PLOT_STYLE_CFG)
    data = json.loads(DATA_JSON)

    active_negotiations_detail, active_negotiations_totals = _records(data["active_negotiations"], "yield_h", "yield_r", "yield_n")
    sudden_adaptations_detail, sudden_adaptations_totals = _records(data["sudden_adaptations"], "init_h", "init_r", "init_n")

    # fig1: Active Negotiations summary
    _plot_task_overview(
        active_negotiations_detail,
        active_negotiations_totals,
        "Active Negotiations",
        human_label="Human yielded",
        robot_label="Robot yielded",
        neutral_label="Unclear",
        composition_title="Active Negotiations: Yielding Share",
    )
    # fig2: Sudden Adaptations summary
    _plot_task_overview(
        sudden_adaptations_detail,
        sudden_adaptations_totals,
        "Sudden Adaptations",
        human_label="Human initiated",
        robot_label="Robot initiated",
        neutral_label="Unclear",
        composition_title="Sudden Adaptations: Initiating Share",
    )
    # fig3: Phase-wise share comparison (yielding/initiating by where)
    _plot_phase_share_comparison(active_negotiations_detail, sudden_adaptations_detail)

    if SHOW_ALTERNATIVE_FORMAT:
        _plot_task_alternative_format(
            active_negotiations_detail,
            active_negotiations_totals,
            "Active Negotiations",
            human_label="Human yielded",
            robot_label="Robot yielded",
            neutral_label="Neutral",
        )
        _plot_task_alternative_format(
            sudden_adaptations_detail,
            sudden_adaptations_totals,
            "Sudden Adaptations",
            human_label="Human initiated",
            robot_label="Robot initiated",
            neutral_label="Neutral",
        )

    # Keep the code for extra plots, but disable rendering them for now.
    if False:
        fig_cmp, axes_cmp = plt.subplots(2, 2, figsize=(14, 10))
        _plot_participant_dominance_compare(axes_cmp[0, 0], active_negotiations_totals, sudden_adaptations_totals)
        _plot_human_share_scatter(axes_cmp[0, 1], active_negotiations_totals, sudden_adaptations_totals, show_regions=False)
        _plot_stacked(
            axes_cmp[1, 0],
            active_negotiations_totals,
            "Active Negotiations: Share of Who Yielded by Participant",
            human_label="Human yielded",
            robot_label="Robot yielded",
            neutral_label="Unclear",
        )
        _plot_stacked(
            axes_cmp[1, 1],
            sudden_adaptations_totals,
            "Sudden Adaptations: Share of Who Initiated by Participant",
            human_label="Human initiated",
            robot_label="Robot initiated",
            neutral_label="Unclear",
        )
        fig_cmp.suptitle("Participant-Level Comparison", y=1.02)
        fig_cmp.tight_layout()

        fig_phase, axes_phase = plt.subplots(1, 2, figsize=(16, 6))
        _plot_phase_totals(axes_phase[0], active_negotiations_detail, "Active Negotiations: Totals by Phase")
        _plot_phase_totals(axes_phase[1], sudden_adaptations_detail, "Sudden Adaptations: Totals by Phase")
        fig_phase.tight_layout()

        fig_totals, ax_totals = plt.subplots(1, 1, figsize=(10, 5))
        _plot_totals(ax_totals, active_negotiations_totals, sudden_adaptations_totals)
        fig_totals.tight_layout()

        fig_scatter, ax_scatter = plt.subplots(1, 1, figsize=(8, 7))
        _plot_human_share_scatter(ax_scatter, active_negotiations_totals, sudden_adaptations_totals, show_regions=True)
        fig_scatter.tight_layout()

    try:
        plt.show()
    except Exception as exc:
        print(f"Could not open plot window: {exc}")
        print("Run this script in a desktop session with a valid DISPLAY to view plots.")


if __name__ == "__main__":
    main()
