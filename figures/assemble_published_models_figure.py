#!/usr/bin/env python3
"""
Combined two-panel figure on published multispecies ecological models, with
Panel B on alpha_eff_hull:
  Panel A (left):  bifurcation diagram from lever_and_multimodel_prevalence.py
  Panel B (right): abrupt-boundary scatter with alpha_eff_hull on the x-axis,
                   rendered on log-log axes with thin errorbars (settings
                   inherited from multimodel_alpha_eff_metrics.py / hull variant).

Usage:
    python figures/assemble_published_models_figure.py [--input PATH] [--output STEM] [--dpi INT]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lever_and_multimodel_prevalence as base
import multimodel_alpha_eff_metrics as metrics
import _alpha_eff_loglog_helpers as ll


_orig_draw_panel_b = base.draw_panel_b


def _draw_panel_b_hull_log(ax, groups, label_size, tick_label_size):
    """Panel B with alpha_eff_hull on x-axis, rendered on log-log axes with
    thin errorbars. Mirrors figure_4_panel_b_hull.draw_panel_b_hull
    (log_scale=True branch); calls the ORIGINAL base.draw_panel_b to avoid
    recursion when this function is installed as base.draw_panel_b itself.
    """
    clipped = ll.clip_groups_for_log(groups)
    restore = ll.monkey_patch_thin_errorbars(
        elinewidth=0.18, ecolor_alpha_mult=0.5, y_lower_clip=1e-3,
    )
    try:
        _orig_draw_panel_b(ax, clipped, label_size, tick_label_size)
    finally:
        restore()

    ax.set_xlabel(
        r"Non-linearity strength ($\alpha_\mathrm{eff}$)",
        fontsize=label_size,
    )
    xs, _ys = ll.flatten_groups_xs_ys(clipped)
    ll.apply_loglog_axes(ax, xs)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "data" / "lever_bifurcation_branches.csv"
    default_output = script_dir.parent / "pdffiles" / "main" / "figure_4"

    parser = argparse.ArgumentParser(
        description="Figure 4: Panel A + Panel B (alpha_eff_hull)."
    )
    parser.add_argument("--input", default=str(default_input))
    parser.add_argument("--output", default=str(default_output))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Force rebuild of the panel-B groups cache from JSONs.")
    args = parser.parse_args()

    output_stem = Path(args.output)
    if output_stem.suffix:
        output_stem = output_stem.with_suffix("")

    df = base.load_csv(Path(args.input))
    df = base.validate_csv(df)

    groups = metrics.load_or_build_groups("alpha_eff_hull", rebuild=args.rebuild_cache)

    base.draw_panel_b = _draw_panel_b_hull_log
    try:
        base.make_combined_figure(df, groups, output_stem, args.dpi)
    finally:
        base.draw_panel_b = _orig_draw_panel_b


if __name__ == "__main__":
    main()
