"""Shared primitives for the log-log variants of the alpha_eff vs fold-fraction
panel.

Used by assemble_published_models_figure.py and the log-log mode of
multimodel_alpha_eff_metrics.py.  Keeps the sentinel-row + broken-axis
machinery in one place.
"""

from __future__ import annotations

import math

import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np

import lever_and_multimodel_prevalence as base


# ─── Data clipping ──────────────────────────────────────────────────────────

def clip_groups_for_log(groups, x_floor: float = 1e-7,
                         y_zero_sentinel: float = 1e-3,
                         y_min_nonzero: float = 1e-2,
                         y_zero_tol: float = 1 / 128):
    """Map "effective zero" mean fold fractions (below 1/n_dirs = 1/128,
    i.e. <1 detected fold per 128-direction scan) to a sentinel at
    y_zero_sentinel.  Nonzero group means above the threshold are floored
    at y_min_nonzero so nothing falls into the broken-axis gap."""

    def clip_pts(pts):
        out = []
        for (mx, sx, my, sy) in pts:
            mx_new = max(mx, x_floor)
            if my <= y_zero_tol:
                my_new = y_zero_sentinel
                sy_new = 0.0
            else:
                my_new = max(my, y_min_nonzero)
                sy_new = sy
            out.append((mx_new, sx, my_new, sy_new))
        return out

    clipped = {}
    for key, val in groups.items():
        if isinstance(val, dict):
            new_inner = {}
            for n, inner in val.items():
                if isinstance(inner, dict):
                    new_inner[n] = {k: clip_pts(v) for k, v in inner.items()}
                else:
                    new_inner[n] = clip_pts(inner)
            clipped[key] = new_inner
        else:
            clipped[key] = val
    return clipped


# ─── Errorbar monkey-patch ──────────────────────────────────────────────────

def monkey_patch_thin_errorbars(elinewidth: float = 0.18,
                                 ecolor_alpha_mult: float = 0.55,
                                 y_lower_clip: float = 1e-3):
    """Temporarily replace `base.plot_pts` with a thin-errorbar + clipped
    lower-y version.  Returns a restore callable."""
    _orig = base.plot_pts

    def _thin(ax, pts, marker, color, size, eb_kw, pt_alpha=1.0):
        rgba_eb = list(mcolors.to_rgba(color))
        rgba_eb[3] = pt_alpha * ecolor_alpha_mult
        rgba_eb = tuple(rgba_eb)
        rgba_pt = list(mcolors.to_rgba(color))
        rgba_pt[3] = pt_alpha
        rgba_pt = tuple(rgba_pt)
        new_eb_kw = {**eb_kw, "elinewidth": elinewidth}

        for mean_x, std_x, mean_y, std_y in pts:
            y_low_ext = max(0.0, min(std_y, mean_y - y_lower_clip))
            y_up_ext  = std_y
            yerr_asym = [[y_low_ext], [y_up_ext]]
            ax.errorbar(mean_x, mean_y,
                        xerr=std_x, yerr=yerr_asym,
                        fmt="none", ecolor=rgba_eb, **new_eb_kw)
            ax.scatter([mean_x], [mean_y],
                       marker=marker, color=rgba_pt, s=size,
                       linewidths=0.6,
                       edgecolors=(0.0, 0.0, 0.0, pt_alpha),
                       zorder=3)

    base.plot_pts = _thin
    def restore():
        base.plot_pts = _orig
    return restore


# ─── Axis setup ─────────────────────────────────────────────────────────────

def flatten_groups_xs_ys(groups):
    xs, ys = [], []
    for v in groups.values():
        for inner in v.values():
            if isinstance(inner, dict):
                for pts in inner.values():
                    for tup in pts:
                        xs.append(tup[0]); ys.append(tup[2])
            else:
                for tup in inner:
                    xs.append(tup[0]); ys.append(tup[2])
    return xs, ys


def apply_loglog_axes(ax, xs, y_zero: float = 1e-3, y_log_min: float = 1e-2,
                      y_top_pad: float = 1.2):
    """Apply log-log scales, symmetric y-padding, custom "0"/log-decade
    y-ticks, and broken-axis marks on both y-spines.  Call AFTER drawing
    the points but BEFORE saving."""
    ax.set_xscale("log", nonpositive="clip")
    ax.set_yscale("log", nonpositive="clip")
    if xs:
        xmin = max(min(xs) * 0.5, 1e-8)
        xmax = max(xs) * 2.0
        ax.set_xlim(xmin, xmax)
    else:
        ax.set_xlim(1e-3, 1.0)
    ax.set_ylim(y_zero / y_top_pad, y_top_pad)
    ax.autoscale(enable=False)
    ax.set_aspect("auto")

    yticks = [y_zero, 1e-2, 1e-1, 1.0]
    yticklabels = ["0", r"$10^{-2}$", r"$10^{-1}$", "1"]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    minor_locs = []
    for decade in (1e-2, 1e-1):
        for k in range(2, 10):
            v = decade * k
            if y_log_min <= v <= 1.0:
                minor_locs.append(v)
    ax.yaxis.set_minor_locator(mticker.FixedLocator(minor_locs))

    ylog_axis_min = math.log10(y_zero / y_top_pad)
    ylog_axis_max = math.log10(y_top_pad)
    def yf(y):
        return (math.log10(y) - ylog_axis_min) / (ylog_axis_max - ylog_axis_min)
    yf_break = 0.5 * (yf(y_zero) + yf(y_log_min))

    d_x = 0.012
    d_y = 0.015
    brk_kw = dict(transform=ax.transAxes, color="black",
                  clip_on=False, linewidth=0.9)
    for x_side in (0.0, 1.0):
        ax.plot([x_side - d_x, x_side + d_x],
                [yf_break - d_y, yf_break + d_y], **brk_kw)
        ax.plot([x_side - d_x, x_side + d_x],
                [yf_break - 2 * d_y, yf_break], **brk_kw)
