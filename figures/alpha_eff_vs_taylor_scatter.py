#!/usr/bin/env python3
"""
Two-panel scatter comparing the three alpha_eff definitions across every
system included in Figure 4 panel B.

Panel A: alpha_eff (y) vs alpha_eff_taylor (x)
Panel B: alpha_eff_hull (y) vs alpha_eff (x)

One marker per JSON file (no aggregation across replicates). Marker shapes
and colors match the Figure 4 panel B legend: Gibbs (Q1/Q2/Q3) diamonds,
Karatayev (FMI/RMI) plus signs, Lever stars, Mougi circles, Aguade
hexagons, Stouffer triangles. Transparency encodes species richness n.

Usage:
    python alpha_eff_vs_taylor_scatter.py [--output STEM] [--dpi INT]
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

import combined_pauls_prevalence_figure as base


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


def load_record(filepath: str):
    """Return (alpha_eff_taylor, alpha_eff, alpha_eff_hull, regime) or None.

    Any of the three alphas may be None if missing from the JSON; the
    caller decides which panel each record can contribute to.
    """
    with open(filepath) as fh:
        data = json.load(fh)
    a_tay  = data.get("alpha_eff_taylor")
    a_eff  = data.get("alpha_eff")
    a_hull = data.get("alpha_eff_hull")
    regime = data.get("metadata", {}).get("regime")
    return (
        None if a_tay  is None else float(a_tay),
        None if a_eff  is None else float(a_eff),
        None if a_hull is None else float(a_hull),
        regime,
    )


def collect_eco(patterns_by_n):
    """Return {n: [(a_tay, a_eff, a_hull), ...]} for a single model family."""
    out = {}
    for n, pattern in patterns_by_n.items():
        pts = []
        for fp in glob.glob(pattern):
            rec = load_record(fp)
            if rec is None:
                continue
            a_tay, a_eff, a_hull, _ = rec
            pts.append((a_tay, a_eff, a_hull))
        if pts:
            out[n] = pts
    return out


def collect_gibbs():
    """Return {n: {'Q1': [...], 'Q2': [...], 'Q3': [...]}}."""
    out = {n: {"Q1": [], "Q2": [], "Q3": []} for n in base.GIBBS_SOURCES}
    for n, pattern in base.GIBBS_SOURCES.items():
        for fp in glob.glob(pattern):
            rec = load_record(fp)
            if rec is None:
                continue
            a_tay, a_eff, a_hull, regime = rec
            if regime in out[n]:
                out[n][regime].append((a_tay, a_eff, a_hull))
    return out


def _filter(pts, ix, iy):
    """Return (xs, ys) keeping only records where both coords are finite."""
    xs, ys = [], []
    for rec in pts:
        x, y = rec[ix], rec[iy]
        if x is None or y is None:
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        xs.append(x)
        ys.append(y)
    return xs, ys


def plot_group(ax, pts, ix, iy, marker, color, size, pt_alpha):
    xs, ys = _filter(pts, ix, iy)
    if not xs:
        return
    import matplotlib.colors as mcolors
    rgba = list(mcolors.to_rgba(color))
    rgba[3] = pt_alpha
    ax.scatter(xs, ys, marker=marker, color=rgba, s=size,
               linewidths=0.5, edgecolors=(0.0, 0.0, 0.0, pt_alpha),
               zorder=3)


def draw_panel(ax, gibbs, karat_fmi, karat_rmi, lever, mougi, aguade, stouffer,
               ix, iy, n_alpha, colors, markers, label_size, tick_label_size,
               xlabel, ylabel, panel_letter):
    marker_gibbs, marker_karat, marker_lever, \
        marker_mougi, marker_aguade, marker_stouffer = markers
    c_gibbs, c_karat, c_lever, c_mougi, c_aguade, c_stouffer = colors

    for n in sorted(gibbs):
        a = n_alpha(n)
        for regime in ("Q1", "Q2", "Q3"):
            plot_group(ax, gibbs[n][regime], ix, iy,
                       marker_gibbs, c_gibbs[regime], 26, a)
    for n, pts in sorted(karat_fmi.items()):
        plot_group(ax, pts, ix, iy, marker_karat, c_karat["FMI"], 26, n_alpha(n))
    for n, pts in sorted(karat_rmi.items()):
        plot_group(ax, pts, ix, iy, marker_karat, c_karat["RMI"], 26, n_alpha(n))
    for n, pts in sorted(lever.items()):
        plot_group(ax, pts, ix, iy, marker_lever, c_lever, 42, n_alpha(n))
    for n, pts in sorted(mougi.items()):
        plot_group(ax, pts, ix, iy, marker_mougi, c_mougi, 22, n_alpha(n))
    for n, pts in sorted(aguade.items()):
        plot_group(ax, pts, ix, iy, marker_aguade, c_aguade, 26, n_alpha(n))
    for n, pts in sorted(stouffer.items()):
        plot_group(ax, pts, ix, iy, marker_stouffer, c_stouffer, 24, n_alpha(n))

    ax.set_xlabel(xlabel, fontsize=label_size)
    ax.set_ylabel(ylabel, fontsize=label_size)
    ax.tick_params(axis="both", labelsize=tick_label_size)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.text(-0.02, 1.05, panel_letter, transform=ax.transAxes,
            fontsize=label_size, fontweight="bold", va="bottom", ha="left")


def make_figure(output_stem: Path, dpi: int) -> None:
    label_size = mpl.font_manager.FontProperties(
        size=mpl.rcParams["axes.labelsize"]).get_size_in_points()
    tick_label_size = label_size * 0.8

    gibbs     = collect_gibbs()
    karat_fmi = collect_eco(base.ECOLOGICAL_SOURCES["karat_fmi"])
    karat_rmi = collect_eco(base.ECOLOGICAL_SOURCES["karat_rmi"])
    lever     = collect_eco(base.ECOLOGICAL_SOURCES["lever"])
    mougi     = collect_eco(base.ECOLOGICAL_SOURCES["mougi_random"])
    aguade    = collect_eco(base.ECOLOGICAL_SOURCES["aguade"])
    stouffer  = collect_eco(base.ECOLOGICAL_SOURCES["stouffer"])

    all_n = sorted({n for d in [gibbs, karat_fmi, karat_rmi, lever,
                                 mougi, aguade, stouffer] for n in d})
    n_min, n_max = all_n[0], all_n[-1]

    def n_alpha(n):
        t = (n - n_min) / max(n_max - n_min, 1)
        return 0.2 + 0.8 * t

    _t10 = [c["color"] for c in plt.rcParams["axes.prop_cycle"]]
    c_gibbs    = {"Q1": _t10[0], "Q2": _t10[1], "Q3": _t10[2]}
    c_karat    = {"FMI": _t10[3], "RMI": _t10[4]}
    c_lever    = _t10[5]
    c_mougi    = _t10[6]
    c_aguade   = _t10[7]
    c_stouffer = _t10[8]
    colors = (c_gibbs, c_karat, c_lever, c_mougi, c_aguade, c_stouffer)

    marker_gibbs    = "d"
    marker_karat    = "P"
    marker_lever    = "*"
    marker_mougi    = "o"
    marker_aguade   = "h"
    marker_stouffer = "^"
    markers = (marker_gibbs, marker_karat, marker_lever,
               marker_mougi, marker_aguade, marker_stouffer)

    # Figure layout: two square panels side by side, shared legend + cbar on the right
    scale = 0.85 * 0.90
    H = 2.50 * scale
    W = H * 1.15

    left_m   = 0.17 * 4.45
    right_m  = 1.80
    top_m    = 0.10
    bot_m    = 0.50
    gap_AB   = 0.70

    fig_width  = left_m + W + gap_AB + W + right_m
    fig_height = bot_m + H + top_m

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax_a = fig.add_axes([
        left_m / fig_width,
        bot_m / fig_height,
        W / fig_width,
        H / fig_height,
    ])
    ax_b = fig.add_axes([
        (left_m + W + gap_AB) / fig_width,
        bot_m / fig_height,
        W / fig_width,
        H / fig_height,
    ])

    # Record tuple layout: (alpha_taylor, alpha_eff, alpha_hull)  →  ix 0,1,2
    draw_panel(ax_a,
               gibbs, karat_fmi, karat_rmi, lever, mougi, aguade, stouffer,
               ix=1, iy=0, n_alpha=n_alpha,
               colors=colors, markers=markers,
               label_size=label_size, tick_label_size=tick_label_size,
               xlabel=r"$\alpha_\mathrm{eff}$",
               ylabel=r"$\alpha_\mathrm{eff}^{\mathrm{Taylor}}$",
               panel_letter="a")

    draw_panel(ax_b,
               gibbs, karat_fmi, karat_rmi, lever, mougi, aguade, stouffer,
               ix=1, iy=2, n_alpha=n_alpha,
               colors=colors, markers=markers,
               label_size=label_size, tick_label_size=tick_label_size,
               xlabel=r"$\alpha_\mathrm{eff}$",
               ylabel=r"$\alpha_\mathrm{eff}^{\mathrm{hull}}$",
               panel_letter="b")

    # Legend anchored to the right panel (mirrors panel B in Figure 4)
    gibbs_proxy    = mlines.Line2D([], [], linestyle="none")
    karat_proxy    = mlines.Line2D([], [], linestyle="none")
    lever_proxy    = mlines.Line2D([], [], linestyle="none")
    mougi_proxy    = mlines.Line2D([], [], linestyle="none")
    aguade_proxy   = mlines.Line2D([], [], linestyle="none")
    stouffer_proxy = mlines.Line2D([], [], linestyle="none")

    legend_handles = [gibbs_proxy, karat_proxy, lever_proxy, mougi_proxy,
                      aguade_proxy, stouffer_proxy]
    legend_labels = [
        "Gibbs et al (2023)",
        "Karatayev et al (2023)",
        "Lever et al (2014)",
        "Mougi (2024)",
        "Aguade-Gorgorio et al (2024)",
        "Stouffer & Bascompte (2010)",
    ]
    legend_handlers = {
        gibbs_proxy:    base.MultiMarkerHandler(
            [marker_gibbs] * 3,
            [c_gibbs["Q1"], c_gibbs["Q2"], c_gibbs["Q3"]], msize=5.7),
        karat_proxy:    base.MultiMarkerHandler(
            [marker_karat, marker_karat],
            [c_karat["FMI"], c_karat["RMI"]], msize=5.7),
        lever_proxy:    base.MultiMarkerHandler([marker_lever],    [c_lever],    msize=7.1),
        mougi_proxy:    base.MultiMarkerHandler([marker_mougi],    [c_mougi],    msize=5.3),
        aguade_proxy:   base.MultiMarkerHandler([marker_aguade],   [c_aguade],   msize=5.7),
        stouffer_proxy: base.MultiMarkerHandler([marker_stouffer], [c_stouffer], msize=5.5),
    }

    ax_b.legend(
        legend_handles,
        legend_labels,
        handler_map=legend_handlers,
        frameon=False,
        fontsize=tick_label_size,
        handlelength=1.5,
        handletextpad=0.3,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
    )

    # Colorbar for n (same construction as panel B in Figure 4)
    c_agu_rgba = np.array(mpl.colors.to_rgba(c_aguade))
    white = np.array([1.0, 1.0, 1.0])
    n_steps = 256
    alphas_grad = np.linspace(0.2, 1.0, n_steps)
    cmap_arr = np.zeros((n_steps, 4))
    cmap_arr[:, :3] = c_agu_rgba[:3] * alphas_grad[:, None] \
                      + white * (1 - alphas_grad[:, None])
    cmap_arr[:, 3] = 1.0
    n_cmap = mpl.colors.ListedColormap(cmap_arr)
    sm = mpl.cm.ScalarMappable(cmap=n_cmap,
                               norm=mpl.colors.Normalize(vmin=n_min, vmax=n_max))
    sm.set_array([])

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    leg_bb = ax_b.get_legend().get_window_extent(renderer)
    fig_w_px, fig_h_px = fig.get_size_inches() * fig.dpi

    ax_pos = ax_b.get_position()
    leg_cx_fig  = (leg_bb.x0 + leg_bb.x1) / 2 / fig_w_px
    cbar_width  = 0.55 * ax_pos.width
    cbar_left   = leg_cx_fig - cbar_width / 2
    cbar_height = 0.055 * ax_pos.height
    gap_fig     = 8 / fig_h_px
    title_h_fig = (tick_label_size - 1) / 72 * fig.dpi / fig_h_px
    cbar_bottom = leg_bb.y0 / fig_h_px - gap_fig - title_h_fig - cbar_height
    cbar_axes = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    cb = fig.colorbar(sm, cax=cbar_axes, orientation="horizontal")
    cb.set_ticks(all_n)
    cb.set_ticklabels([str(n) for n in all_n])
    cb.ax.tick_params(labelsize=tick_label_size - 1, length=2)
    cbar_axes.set_title("Diversity ($n$)", fontsize=tick_label_size, pad=2)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path,           bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"[INFO] Wrote PNG: {png_path}")
    print(f"[INFO] Wrote PDF: {pdf_path}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_output = script_dir / "pdffiles" / "alpha_eff_vs_taylor_scatter"

    parser = argparse.ArgumentParser(
        description="Scatter: alpha_eff vs alpha_eff_taylor and alpha_eff_hull "
                    "vs alpha_eff for all Fig 4 models."
    )
    parser.add_argument("--output", default=str(default_output),
                        help=f"Output stem without extension (default: {default_output})")
    parser.add_argument("--dpi", type=int, default=300,
                        help="PNG resolution (default: 300)")
    args = parser.parse_args()

    output_stem = Path(args.output)
    if output_stem.suffix:
        output_stem = output_stem.with_suffix("")

    make_figure(output_stem, args.dpi)


if __name__ == "__main__":
    main()
