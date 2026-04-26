#!/usr/bin/env python3
"""
Figure 4 panel B rendered as a two-panel figure that compares the two
alternative non-linearity metrics on the x-axis:

    left  panel: alpha_eff_taylor  (Taylor-expansion-based unified metric)
    right panel: alpha_eff_hull    (L² nonlinearity fraction over convex hull)

Both panels share the same y-axis ("Fraction abrupt boundaries") and
a single legend + diversity colorbar drawn once to the right of the
right panel.

Usage:
    python figure_4_panel_b_metrics.py [--output STEM] [--dpi INT] [--log]
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import combined_pauls_prevalence_figure as base
import _panel_b_loglog_helpers as ll


DEFAULT_CACHE_DIR = Path("figures/cache")
CACHE_VERSION      = 1


# ─── Data loaders (parameterized by metric key) ─────────────────────────────

def _fold_fraction(filepath: str, metric_key: str):
    with open(filepath) as fh:
        data = json.load(fh)

    scan_results = data.get("scan_results", [])
    if not scan_results:
        return None
    directions = scan_results[0].get("directions", [])
    if not directions:
        return None

    alpha = data.get(metric_key)
    if alpha is None:
        return None

    n_fold = sum(1 for row in directions if row["flag"] == "fold")
    return float(alpha), n_fold / len(directions)


def _existing_points(pattern: str, metric_key: str):
    pts = []
    for filepath in glob.glob(pattern):
        point = _fold_fraction(filepath, metric_key)
        if point is not None:
            pts.append(point)
    return pts


def build_groups(metric_key: str):
    gibbs_raw = {n: defaultdict(list) for n in base.GIBBS_SOURCES}
    for n, pattern in base.GIBBS_SOURCES.items():
        for filepath in glob.glob(pattern):
            with open(filepath) as fh:
                data = json.load(fh)
            point = _fold_fraction(filepath, metric_key)
            if point is None:
                continue
            regime = data["metadata"]["regime"]
            gibbs_raw[n][regime].append(point)

    def regime_stats(pts):
        if not pts:
            return []
        alphas, ffs = zip(*pts)
        return [(np.mean(alphas), np.std(alphas), np.mean(ffs), np.std(ffs))]

    gibbs = {
        n: {regime: regime_stats(gibbs_raw[n][regime]) for regime in ("Q1", "Q2", "Q3")}
        for n in base.GIBBS_SOURCES
    }

    eco = {}
    for key, n_patterns in base.ECOLOGICAL_SOURCES.items():
        eco[key] = {
            n: ([base.group_stats(pts)] if (pts := _existing_points(pattern, metric_key)) else [])
            for n, pattern in n_patterns.items()
        }

    return {"gibbs": gibbs, **eco}


def default_cache_path(metric_key: str) -> Path:
    return DEFAULT_CACHE_DIR / f"figure_4_panel_b_{metric_key}.pkl"


def load_or_build_groups(metric_key: str, cache_path: Path | None = None,
                         rebuild: bool = False) -> dict:
    if cache_path is None:
        cache_path = default_cache_path(metric_key)
    if cache_path.exists() and not rebuild:
        print(f"Loading cached panel-B groups ({metric_key}) from {cache_path}")
        with cache_path.open("rb") as f:
            payload = pickle.load(f)
        if payload.get("version") == CACHE_VERSION and payload.get("metric_key") == metric_key:
            return payload["groups"]
        print(f"  Cache version/metric mismatch — rebuilding")
    groups = build_groups(metric_key)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(
            {"version": CACHE_VERSION, "metric_key": metric_key, "groups": groups},
            f, protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"Saved cache to {cache_path}")
    return groups


# ─── Panel drawing ──────────────────────────────────────────────────────────

def _draw_panel(ax, groups, xlabel, label_size, tick_label_size,
                log_scale: bool, suppress_legend_cbar: bool,
                panel_letter: str):
    """Draw one panel B variant.  If suppress_legend_cbar, remove the legend
    and colorbar that base.draw_panel_b adds, so only one copy survives in
    the combined figure."""
    fig = ax.get_figure()
    pre_axes_ids = {id(a) for a in fig.axes}

    if log_scale:
        groups_to_plot = ll.clip_groups_for_log(groups)
    else:
        groups_to_plot = groups

    base.draw_panel_b(ax, groups_to_plot, label_size, tick_label_size)

    if suppress_legend_cbar:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        for a in list(fig.axes):
            if id(a) not in pre_axes_ids and a is not ax:
                fig.delaxes(a)

    ax.set_xlabel(xlabel, fontsize=label_size)

    xs, _ys = ll.flatten_groups_xs_ys(groups_to_plot)
    if log_scale:
        ll.apply_loglog_axes(ax, xs)
    else:
        if xs:
            xmin = min(0.0, min(xs))
            xmax = max(1.0, max(xs))
            margin = 0.03 * (xmax - xmin)
            ax.set_xlim(xmin - margin, xmax + margin)

    for t in ax.texts:
        if t.get_text() == "b":
            t.set_text(panel_letter)
            break


def make_combined_figure(groups_taylor, groups_hull, output_stem: Path,
                         dpi: int, log_scale: bool = False) -> None:
    label_size = mpl.font_manager.FontProperties(
        size=mpl.rcParams["axes.labelsize"]
    ).get_size_in_points()
    tick_label_size = label_size * 0.8

    scale = 0.85 * 0.90
    H = 2.50 * scale
    W = H

    left_m = 0.17 * 4.45
    inter_m = 0.55
    right_m = 1.65
    top_m = 0.10
    bot_m = 0.50

    fig_width = left_m + W + inter_m + W + right_m
    fig_height = bot_m + H + top_m

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax_l = fig.add_axes([
        left_m / fig_width,
        bot_m / fig_height,
        W / fig_width,
        H / fig_height,
    ])
    ax_r = fig.add_axes([
        (left_m + W + inter_m) / fig_width,
        bot_m / fig_height,
        W / fig_width,
        H / fig_height,
    ])

    if log_scale:
        restore = ll.monkey_patch_thin_errorbars(elinewidth=0.18,
                                                 ecolor_alpha_mult=0.5,
                                                 y_lower_clip=1e-3)
    else:
        restore = lambda: None
    try:
        _draw_panel(
            ax_l, groups_taylor,
            r"Non-linearity strength ($\alpha_\mathrm{eff}^{\mathrm{Taylor}}$)",
            label_size, tick_label_size, log_scale,
            suppress_legend_cbar=True, panel_letter="b",
        )
        _draw_panel(
            ax_r, groups_hull,
            r"Non-linearity strength ($\alpha_\mathrm{eff}^{\mathrm{hull}}$)",
            label_size, tick_label_size, log_scale,
            suppress_legend_cbar=False, panel_letter="c",
        )
    finally:
        restore()

    ax_r.set_ylabel("")

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"[INFO] Wrote PNG: {png_path}")
    print(f"[INFO] Wrote PDF: {pdf_path}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_output = script_dir / "pdffiles" / "figure_4_panel_b_metrics"

    parser = argparse.ArgumentParser(
        description="Figure 4 panel B with Taylor (left) and Hull (right) metrics.",
    )
    parser.add_argument("--output", default=str(default_output),
                        help=f"Output stem without extension (default: {default_output})")
    parser.add_argument("--dpi", type=int, default=300,
                        help="PNG resolution (default: 300)")
    parser.add_argument("--log", action="store_true",
                        help="Also render a log-log variant at <stem>_loglog.{png,pdf}")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Force rebuild of the panel-B groups cache from JSONs.")
    args = parser.parse_args()

    output_stem = Path(args.output)
    if output_stem.suffix:
        output_stem = output_stem.with_suffix("")

    groups_taylor = load_or_build_groups("alpha_eff_taylor", rebuild=args.rebuild_cache)
    groups_hull   = load_or_build_groups("alpha_eff_hull",   rebuild=args.rebuild_cache)

    make_combined_figure(groups_taylor, groups_hull, output_stem, args.dpi,
                         log_scale=False)
    if args.log:
        stem_log = output_stem.with_name(output_stem.name + "_loglog")
        make_combined_figure(groups_taylor, groups_hull, stem_log, args.dpi,
                             log_scale=True)


if __name__ == "__main__":
    main()
