#!/usr/bin/env python3
"""
Three-panel scatter of alpha_eff_taylor_grid (x) vs alpha_eff_hull_grid
(y), one panel per 2_bank_standard_..._muB_* bank (muB in {-0.1, 0.0, 0.1}).

Within each panel, the 11 grid points are plotted per diversity n
(color) with y-axis error bars from the std of alpha_eff_hull_grid over
replicates. The Taylor metric has zero replicate variance by
construction (see the closed-form identity in the supplement), so no
x-axis error bars are drawn.

Usage:
    python alpha_eff_hull_vs_taylor.py [--output STEM] [--dpi INT]
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


plt.rcParams.update({
    "font.family":      "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth":   0.7,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "legend.frameon":   False,
})


BASE = "data/example_runs"
BANKS = [
    (r"$\mu_B = -0.1$", f"{BASE}/2_bank_standard_50_models_n_4-20_128_dirs_muB_-0.1"),
    (r"$\mu_B = 0.0$",  f"{BASE}/2_bank_standard_50_models_n_4-20_128_dirs_muB_0.0"),
    (r"$\mu_B = 0.1$",  f"{BASE}/2_bank_standard_50_models_n_4-20_128_dirs_muB_0.1"),
]


def _fold_frac_grid(scan_results):
    """Per-grid fold fraction = #{flag=='fold'} / #directions, NaN if empty."""
    out = np.full(len(scan_results), np.nan, dtype=float)
    for i, s in enumerate(scan_results):
        directions = s.get("directions") or []
        if not directions:
            continue
        n_fold = sum(1 for row in directions if row.get("flag") == "fold")
        out[i] = n_fold / len(directions)
    return out


def load_bank(directory: str):
    """Return {n: {'taylor_mean','hull_mean','hull_std','fold_mean','fold_std','n_reps'}}."""
    by_n = {}
    for fp in glob.glob(f"{directory}/2_model_*.json"):
        m = re.search(r"n_(\d+)_", Path(fp).name)
        if not m:
            continue
        n = int(m.group(1))
        with open(fp) as fh:
            d = json.load(fh)
        t_grid = d.get("alpha_eff_taylor_grid")
        h_grid = d.get("alpha_eff_hull_grid")
        scan_results = d.get("scan_results")
        if t_grid is None or h_grid is None or scan_results is None:
            continue
        f_grid = _fold_frac_grid(scan_results)
        if f_grid.shape != np.asarray(h_grid).shape:
            continue
        by_n.setdefault(n, {"taylor": [], "hull": [], "fold": []})
        by_n[n]["taylor"].append(np.asarray(t_grid, dtype=float))
        by_n[n]["hull"].append(np.asarray(h_grid, dtype=float))
        by_n[n]["fold"].append(f_grid)

    out = {}
    for n, rec in by_n.items():
        T = np.vstack(rec["taylor"])
        H = np.vstack(rec["hull"])
        F = np.vstack(rec["fold"])
        out[n] = {
            "taylor_mean": np.nanmean(T, axis=0),
            "hull_mean":   np.nanmean(H, axis=0),
            "hull_std":    np.nanstd(H,  axis=0),
            "fold_mean":   np.nanmean(F, axis=0),
            "fold_std":    np.nanstd(F,  axis=0),
            "n_reps":      H.shape[0],
        }
    return out


# ── colormap (matches tipping_prevalence_panels.py) ─────────────────────────

def build_colormaps(n_bins: int):
    colors    = np.array(sns.color_palette("flare", n_bins))
    cmap_base = mpl.colors.ListedColormap(colors)
    norm      = mpl.colors.BoundaryNorm(np.arange(-0.5, n_bins + 0.5, 1), n_bins)
    return cmap_base, norm


def make_color_dict(n_values, cmap_base):
    colors = cmap_base(np.arange(len(n_values)))
    return {n: colors[i] for i, n in enumerate(n_values)}


def _alpha_eff_to_alpha(a_eff):
    """Primary (alpha_eff_taylor) -> secondary (original alpha)."""
    a = np.asarray(a_eff, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return a / (1.0 - 2.0 * a)


def _alpha_to_alpha_eff(a):
    """Secondary (original alpha) -> primary (alpha_eff_taylor)."""
    a = np.asarray(a, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return a / (1.0 + 2.0 * a)


def draw_panel(ax, data, color_map, label_size, tick_label_size,
               title, panel_letter, show_ylabel,
               inset_bbox, show_inset_xlabel, show_inset_ylabel,
               inset_x_key="hull"):
    for n in sorted(data):
        rec = data[n]
        color = color_map[n]
        ax.errorbar(rec["taylor_mean"], rec["hull_mean"],
                    yerr=rec["hull_std"],
                    fmt="-o", color=color, markersize=3.5, linewidth=1.0,
                    elinewidth=0.5, capsize=0, alpha=0.85, zorder=3)
    ax.set_xlabel(r"$\alpha_\mathrm{eff}^{\mathrm{Taylor}}$",
                  fontsize=label_size)
    if show_ylabel:
        ax.set_ylabel(r"$\alpha_\mathrm{eff}^{\mathrm{hull}}$",
                      fontsize=label_size)
    ax.tick_params(axis="both", labelsize=tick_label_size)
    ax.grid(False)

    secax = ax.secondary_xaxis(
        "top", functions=(_alpha_eff_to_alpha, _alpha_to_alpha_eff))
    secax.set_xlabel(r"$\alpha$", fontsize=label_size, labelpad=2)
    secax.tick_params(axis="x", labelsize=tick_label_size,
                      direction="in", length=3.5)
    ax._secax = secax

    ax.text(0.04, 0.96, panel_letter, transform=ax.transAxes,
            fontsize=label_size, fontweight="bold", va="top", ha="left")
    ax.text(0.96, 0.96, title, transform=ax.transAxes,
            fontsize=label_size, va="top", ha="right")

    # ── Inset: fold prevalence vs alpha_eff^{hull|Taylor} (scatter) ────────
    x_mean_key = f"{inset_x_key}_mean"
    inset_xlabel = {
        "hull":   r"$\alpha_\mathrm{eff}^{\mathrm{hull}}$",
        "taylor": r"$\alpha_\mathrm{eff}^{\mathrm{Taylor}}$",
    }[inset_x_key]
    iax = ax.inset_axes(inset_bbox)
    for n in sorted(data):
        rec = data[n]
        iax.scatter(rec[x_mean_key], rec["fold_mean"],
                    color=color_map[n], s=10, alpha=0.9,
                    edgecolors="none", zorder=3)
    iax.grid(False)
    iax.set_facecolor((1.0, 1.0, 1.0, 0.85))
    iax.tick_params(axis="both", labelsize=tick_label_size * 0.8,
                    length=2, pad=1.5)
    for s in iax.spines.values():
        s.set_linewidth(0.5)
    if show_inset_xlabel:
        iax.set_xlabel(inset_xlabel,
                       fontsize=tick_label_size, labelpad=1.5)
    if show_inset_ylabel:
        iax.set_ylabel("Folds", fontsize=tick_label_size, labelpad=1.5)
    ax._inset = iax
    ax._inset_x_key = inset_x_key


def _sync_secondary_ticks_to_primary(ax):
    """Place secondary-axis ticks at the same visual locations as the primary
    axis ticks, but label them with the original alpha value to two decimals.
    Must be called after the primary x-limits are finalized."""
    secax = getattr(ax, "_secax", None)
    if secax is None:
        return
    xlim = ax.get_xlim()
    primary_ticks = np.asarray(ax.get_xticks(), dtype=float)
    visible = primary_ticks[(primary_ticks >= xlim[0]) &
                            (primary_ticks <= xlim[1])]
    alpha_ticks = _alpha_eff_to_alpha(visible)
    secax.set_xticks(alpha_ticks)
    secax.set_xticklabels([f"{a:.2f}" for a in alpha_ticks])


def make_figure(output_stem: Path, dpi: int) -> None:
    label_size = mpl.font_manager.FontProperties(
        size=mpl.rcParams["axes.labelsize"]).get_size_in_points()
    tick_label_size = label_size * 0.8

    bank_data = [(title, load_bank(path)) for title, path in BANKS]

    n_values = sorted({n for _, d in bank_data for n in d})
    if not n_values:
        raise RuntimeError("No banks with alpha_eff_hull_grid and alpha_eff_taylor_grid.")
    cmap, norm = build_colormaps(len(n_values))
    color_map  = make_color_dict(n_values, cmap)

    scale = 0.85 * 0.90
    H = 2.50 * scale
    W = H

    left_m   = 0.17 * 4.45
    right_m  = 0.95
    top_m    = 0.80
    bot_m    = 0.50
    gap      = 0.45

    fig_width  = left_m + 3 * W + 2 * gap + right_m
    fig_height = bot_m + H + top_m

    fig = plt.figure(figsize=(fig_width, fig_height))

    axes = []
    for i in range(3):
        axes.append(fig.add_axes([
            (left_m + i * (W + gap)) / fig_width,
            bot_m / fig_height,
            W / fig_width,
            H / fig_height,
        ]))

    letters = ("a", "b", "c")
    # Per-panel inset placement (axes fractions: [left, bottom, width, height]).
    # a, b: data fills lower half (curves rise to ~0.17), so upper-left half
    #       is the largest empty region.
    # c:    data jumps to ~0.3 plateau at α_Taylor ≈ 0.08, so everything
    #       below the plateau and right of the jump is empty → lower-right.
    inset_bboxes = {
        "a": (0.22, 0.46, 0.44, 0.40),
        "b": (0.22, 0.46, 0.44, 0.40),
        "c": (0.55, 0.24, 0.40, 0.32),
    }
    for ax, letter, (title, data) in zip(axes, letters, bank_data):
        draw_panel(ax, data, color_map, label_size, tick_label_size,
                   title=title, panel_letter=letter,
                   show_ylabel=(letter == "a"),
                   inset_bbox=inset_bboxes[letter],
                   show_inset_xlabel=(letter in ("a", "c")),
                   show_inset_ylabel=(letter in ("a", "c")),
                   inset_x_key=("taylor" if letter == "c" else "hull"))

    # Tight-wrap: compute unified x/y limits directly from plotted data
    # (including error bars), with a small margin. This is more robust than
    # matplotlib's autoscale when some panels are sparse.
    xs, ys_lo, ys_hi = [], [], []
    for _, data in bank_data:
        for rec in data.values():
            xs.extend(np.asarray(rec["taylor_mean"])[np.isfinite(rec["taylor_mean"])].tolist())
            mean = np.asarray(rec["hull_mean"])
            std  = np.asarray(rec["hull_std"])
            lo = mean - std
            hi = mean + std
            mask = np.isfinite(mean) & np.isfinite(std)
            ys_lo.extend(lo[mask].tolist())
            ys_hi.extend(hi[mask].tolist())
    if xs and ys_lo:
        x_lo_d, x_hi_d = min(xs), max(xs)
        y_lo_d, y_hi_d = min(ys_lo), max(ys_hi)
        x_pad = 0.03 * max(x_hi_d - x_lo_d, 1e-9)
        y_pad = 0.03 * max(y_hi_d - y_lo_d, 1e-9)
        x_lo, x_hi = x_lo_d - x_pad, x_hi_d + x_pad
        y_lo, y_hi = y_lo_d - y_pad, y_hi_d + y_pad
        for ax in axes:
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)

    for ax in axes:
        _sync_secondary_ticks_to_primary(ax)

    # Inset y-limits unified across panels (fold fraction is directly
    # comparable across banks); inset x-limits tightened per panel to its
    # own α_eff^hull range, because the facilitative bank (μ_B = 0.1) spans
    # a much larger hull-nonlinearity range than the other two and a
    # unified x-axis would leave the a/b insets mostly empty.
    all_ys = []
    for _, data in bank_data:
        for rec in data.values():
            all_ys.extend(np.asarray(rec["fold_mean"])[np.isfinite(rec["fold_mean"])].tolist())
    if all_ys:
        iy_lo_d, iy_hi_d = min(all_ys), max(all_ys)
        iy_pad = 0.01 * max(iy_hi_d - iy_lo_d, 1e-9)
        iy_lo, iy_hi = iy_lo_d - iy_pad, iy_hi_d + iy_pad
    else:
        iy_lo, iy_hi = 0.0, 1.0

    for ax, (_, data) in zip(axes, bank_data):
        iax = getattr(ax, "_inset", None)
        if iax is None:
            continue
        x_mean_key = f"{getattr(ax, '_inset_x_key', 'hull')}_mean"
        xs = []
        for rec in data.values():
            xs.extend(np.asarray(rec[x_mean_key])[np.isfinite(rec[x_mean_key])].tolist())
        if xs:
            x_lo_d, x_hi_d = min(xs), max(xs)
            x_pad = 0.01 * max(x_hi_d - x_lo_d, 1e-9)
            iax.set_xlim(x_lo_d - x_pad, x_hi_d + x_pad)
        iax.set_ylim(iy_lo, iy_hi)

    # Discrete colorbar (matches tipping_prevalence_panels.py): ticks at {4, 12, 20}
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    last_pos = axes[-1].get_position()
    cbar_left   = last_pos.x1 + 0.02
    cbar_width  = 0.018
    cbar_bottom = last_pos.y0
    cbar_height = last_pos.height
    cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])

    target_ns = {4, 12, 20}
    tick_positions = [i for i, n in enumerate(n_values) if int(n) in target_ns]
    tick_labels    = [str(int(n)) for n in n_values if int(n) in target_ns]
    cb = fig.colorbar(sm, cax=cax, ticks=tick_positions,
                      orientation="vertical", drawedges=True)
    cb.ax.set_yticklabels(tick_labels)
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=tick_label_size - 1, length=2)
    cb.outline.set_linewidth(0.6)
    cb.dividers.set_linewidth(0.4)
    cb.dividers.set_color("white")
    cax.set_title("$n$", fontsize=label_size, pad=4, loc="center")

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"[INFO] Wrote PDF: {pdf_path}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_output = script_dir.parent / "pdffiles" / "si" / "alpha_hull_vs_taylor_by_muB"

    parser = argparse.ArgumentParser(
        description="Scatter: alpha_eff_taylor_grid vs alpha_eff_hull_grid, "
                    "one panel per muB bank, grouped by n and averaged over replicates."
    )
    parser.add_argument("--output", default=str(default_output))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    output_stem = Path(args.output)
    if output_stem.suffix:
        output_stem = output_stem.with_suffix("")

    make_figure(output_stem, args.dpi)


if __name__ == "__main__":
    main()
