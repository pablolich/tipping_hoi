#!/usr/bin/env python3
"""SI figure: 2x2 boundary-types grid for the all_negative parameterization.

Top row    — bank 2_bank_all_negative_50_models_n_4-10_128_dirs_sA_1.0_sB_1.0
Bottom row — bank ..._stable_throughout
Columns    — Gradual (negative) | Abrupt (fold)

Aesthetic matches figure_3 boundary prevalence panels:
  - seaborn "flare" colormap for n
  - IQR shown as fill_between (alpha=0.18), not errorbars
  - serif/cm typography, lowercase panel labels
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from plot_boundary_types_row_gibbs import (
    DARK_POINT_SIZE,
    DARK_EDGE_WIDTH,
    load_standard_data,
)

PANEL_INFO = [
    ("negative", "Gradual"),
    ("complex",  "Abrupt"),
]

ROW_LABELS = ["Full", "Stable throughout"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bank-top", type=str,
        default="2_bank_all_negative_50_models_n_4-10_128_dirs_sA_1.0_sB_1.0",
    )
    parser.add_argument(
        "--bank-bottom", type=str,
        default="2_bank_all_negative_50_models_n_4-10_128_dirs_sA_1.0_sB_1.0_stable_throughout",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("figures/pdffiles/si_boundary_types_2x2_all_negative.pdf"),
    )
    args = parser.parse_args()

    model_runs = Path("model_runs")
    bank_paths = [model_runs / args.bank_top, model_runs / args.bank_bottom]
    for p in bank_paths:
        if not p.is_dir():
            raise SystemExit(f"Bank directory not found: {p}")

    # ── typography (match figure_3) ──────────────────────────────────────
    plt.rcParams.update({
        "font.family":      "serif",
        "mathtext.fontset": "cm",
        "font.size":        9,
        "axes.labelsize":   10,
        "axes.titlesize":   10,
        "axes.linewidth":   0.7,
        "xtick.direction":  "in",
        "ytick.direction":  "in",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "legend.frameon":   False,
        "legend.fontsize":  9.75,
    })
    label_size = mpl.font_manager.FontProperties(
        size=mpl.rcParams["axes.labelsize"]
    ).get_size_in_points()

    # ── load both banks ───────────────────────────────────────────────────
    print("Loading top bank...")
    df_top = load_standard_data([bank_paths[0]])
    if df_top.empty:
        raise SystemExit(f"No boundary data in {bank_paths[0]}")
    print(f"  {len(df_top) // 4} models, n values = {sorted(df_top['n'].unique())}")

    print("Loading bottom bank...")
    df_bot = load_standard_data([bank_paths[1]])
    if df_bot.empty:
        raise SystemExit(f"No boundary data in {bank_paths[1]}")
    print(f"  {len(df_bot) // 4} models, n values = {sorted(df_bot['n'].unique())}")

    # Shared colormap across both rows (flare palette, matches figure_3)
    all_n = sorted(set(int(v) for v in df_top["n"].unique())
                   | set(int(v) for v in df_bot["n"].unique()))
    n_colors = np.array(sns.color_palette("flare", len(all_n)))
    color_map = {n: n_colors[i] for i, n in enumerate(all_n)}

    # ── layout ────────────────────────────────────────────────────────────
    N_ROWS, N_COLS = 2, len(PANEL_INFO)
    S        = 2.2
    GAP_X    = 0.15
    GAP_Y    = 0.35
    YLABEL_L = 0.75
    MARGIN_L = 0.02
    MARGIN_R = 0.75   # room for n colorbar (pushed further right)
    MARGIN_B = 0.55
    MARGIN_T = 0.30

    FIG_W = MARGIN_L + YLABEL_L + N_COLS * S + (N_COLS - 1) * GAP_X + MARGIN_R
    FIG_H = MARGIN_T + N_ROWS * S + (N_ROWS - 1) * GAP_Y + MARGIN_B

    def nr(x, y, w, h):
        return [x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H]

    fig = plt.figure(figsize=(FIG_W, FIG_H))

    axes_grid = []
    for ri in range(N_ROWS):
        y_phys = MARGIN_B + (N_ROWS - 1 - ri) * (S + GAP_Y)
        row_axes = []
        for ci in range(N_COLS):
            x_phys = MARGIN_L + YLABEL_L + ci * (S + GAP_X)
            row_axes.append(fig.add_axes(nr(x_phys, y_phys, S, S)))
        axes_grid.append(row_axes)

    # ── panel drawer ──────────────────────────────────────────────────────
    def _panel(ax, df, flag, plabel, title, show_yt, show_title, show_xt,
               show_counts):
        m = 0.03
        ax.set_xlim(-m - 0.01, 1.0 + m)
        ax.set_xticks([0.0, 0.5, 1.0])
        if show_xt:
            ax.set_xticklabels(["0", "0.5", "1"])
        else:
            ax.set_xticklabels([])
        ax.set_ylim(-m, 1.0 + m)
        ax.set_yticks([0.0, 0.5, 1.0])
        if show_yt:
            ax.set_yticklabels(["0", "0.5", "1"])
        else:
            ax.set_yticklabels([])
        if show_title:
            ax.set_title(title, fontsize=label_size * 1.15, pad=4)
        ax.text(0.0, 1.01, plabel, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=label_size, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5))

        sub = df[df["boundary_type"] == flag]
        if sub.empty:
            return

        n_values = sorted(int(v) for v in sub["n"].unique())

        # IQR shading (matches figure_3 prevalence panels)
        for nv in n_values:
            sn = sub[sub["n"] == nv]
            if sn.empty:
                continue
            grp = sn.groupby("alpha_value")["fraction"]
            med = grp.median().sort_index()
            q25 = grp.quantile(0.25).reindex(med.index)
            q75 = grp.quantile(0.75).reindex(med.index)
            ax.fill_between(med.index, q25.values, q75.values,
                            color=color_map[nv], alpha=0.18, linewidth=0, zorder=2)

        # median line + dots (+ optional count annotations)
        agg = sub.groupby(["n", "alpha_value"], as_index=False)["fraction"].agg(
            fraction="median", count="size")
        xtick_size = mpl.font_manager.FontProperties(
            size=mpl.rcParams["xtick.labelsize"]
        ).get_size_in_points()
        count_fs = xtick_size * 0.75
        for nv in n_values:
            sn = agg[agg["n"] == nv].sort_values("alpha_value")
            if sn.empty:
                continue
            ax.plot(sn["alpha_value"], sn["fraction"],
                    color=color_map[nv], linewidth=0.9, zorder=2, alpha=0.85)
            ax.scatter(sn["alpha_value"], sn["fraction"],
                       color=color_map[nv], s=DARK_POINT_SIZE * 0.45,
                       edgecolors="black", linewidths=DARK_EDGE_WIDTH * 0.4, zorder=3)
            if show_counts:
                for av, yv, cnt in zip(sn["alpha_value"], sn["fraction"], sn["count"]):
                    ax.annotate(str(int(cnt)), xy=(av, yv),
                                xytext=(2.5, 2.5), textcoords="offset points",
                                fontsize=count_fs, color=color_map[nv],
                                ha="left", va="bottom", zorder=4,
                                path_effects=[mpl.patheffects.withStroke(
                                    linewidth=1.2, foreground="white")])

    # ── fill panels ───────────────────────────────────────────────────────
    row_data = [df_top, df_bot]
    letters = [["a", "b"], ["c", "d"]]
    for ri in range(N_ROWS):
        for ci, (flag, ttl) in enumerate(PANEL_INFO):
            _panel(
                axes_grid[ri][ci], row_data[ri], flag,
                letters[ri][ci], ttl,
                show_yt=(ci == 0),
                show_title=(ri == 0),
                show_xt=(ri == N_ROWS - 1),
                show_counts=(ri == N_ROWS - 1),
            )

    # ── y-axis label per row (includes row name) ──────────────────────────
    for ri in range(N_ROWS):
        y_center_phys = MARGIN_B + (N_ROWS - 1 - ri) * (S + GAP_Y) + S / 2
        fig.text((MARGIN_L + YLABEL_L * 0.35) / FIG_W, y_center_phys / FIG_H,
                 f"Boundary prevalence\n({ROW_LABELS[ri]})",
                 rotation=90, ha="center", va="center",
                 fontsize=label_size * 1.15)

    # ── x-axis label (bottom row only) ────────────────────────────────────
    x_center = (MARGIN_L + YLABEL_L + (N_COLS * S + (N_COLS - 1) * GAP_X) / 2) / FIG_W
    fig.text(x_center, (MARGIN_B * 0.25) / FIG_H,
             r"HOI strength ($\alpha$)",
             ha="center", va="center", fontsize=label_size * 1.15)

    # ── n colorbar ────────────────────────────────────────────────────────
    cmap_obj = mpl.colors.ListedColormap([color_map[n] for n in all_n])
    norm_obj = mpl.colors.BoundaryNorm(np.arange(-0.5, len(all_n) + 0.5, 1),
                                       len(all_n))
    sm = mpl.cm.ScalarMappable(cmap=cmap_obj, norm=norm_obj)
    sm.set_array([])
    pos_top = axes_grid[0][-1].get_position()
    pos_bot = axes_grid[-1][-1].get_position()
    cbar_x = pos_top.x1 + 0.05
    cbar_y = pos_bot.y0 + (pos_top.y1 - pos_bot.y0) * 0.10
    cbar_h = (pos_top.y1 - pos_bot.y0) * 0.80
    cbar_ax = fig.add_axes([cbar_x, cbar_y, 0.012, cbar_h])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks(np.arange(len(all_n)))
    cbar.set_ticklabels([str(n) for n in all_n])
    ytick_size = mpl.font_manager.FontProperties(
        size=mpl.rcParams["ytick.labelsize"]
    ).get_size_in_points()
    cbar.ax.tick_params(labelsize=ytick_size * 0.85)
    cbar.ax.minorticks_off()
    cbar_ax.set_title("n", fontsize=label_size, pad=4, loc="center")

    # ── save ──────────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
