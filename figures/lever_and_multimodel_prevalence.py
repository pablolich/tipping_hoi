#!/usr/bin/env python3
"""
Combined two-panel figure on published multispecies ecological models:
  Panel A (top):    Lever et al. (2014) plant-pollinator bifurcation diagram
  Panel B (bottom): abrupt-boundary fraction vs. effective non-linearity scatter

Usage:
    python lever_and_multimodel_prevalence.py [--input PATH] [--output STEM] [--dpi INT]
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patheffects as mpe
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase, HandlerTuple
import numpy as np
import pandas as pd


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

# ---------------------------------------------------------------------------
# Constants from pauls_idea_plot.py
# ---------------------------------------------------------------------------

Y_FLOOR = -0.1
COOLWARM_QUAL3_NO_GRAY = ["#3F7F93", "#7B6FAE", "#A8752E"]
FORWARD_COLOR  = "#8B1A1A"
BACKWARD_COLOR = "#1F4E79"
FORWARD_MARKER  = ">"
BACKWARD_MARKER = "<"

BRANCH_LINESTYLES = {
    "pre_fold_1":          "-",
    "pre_fold_2":          "--",
    "post_forward":        "-",
    "post_backward":       "-",
    "extinction_forward":  "-",
    "extinction_backward": "--",
}

BRANCH_ORDER = {
    "pre_fold_1":          0,
    "pre_fold_2":          1,
    "post_forward":        2,
    "post_backward":       3,
    "extinction_forward":  4,
    "extinction_backward": 5,
}


# ---------------------------------------------------------------------------
# Helpers from pauls_idea_plot.py
# ---------------------------------------------------------------------------

def build_species_colors(species_ids: np.ndarray) -> Dict[int, Tuple[float, float, float, float]]:
    species_sorted = sorted(int(x) for x in species_ids)
    n_species = len(species_sorted)
    base_colors = [mpl.colors.to_rgba(c) for c in COOLWARM_QUAL3_NO_GRAY]
    if n_species <= len(base_colors):
        return {species_id: base_colors[i] for i, species_id in enumerate(species_sorted)}
    print(
        f"[WARN] Requested {n_species} species colors, but only {len(base_colors)} "
        "custom colors provided; using tab20 fallback."
    )
    cmap = plt.get_cmap("tab20", n_species)
    return {species_id: cmap(i) for i, species_id in enumerate(species_sorted)}


def darken_color(
    color: Tuple[float, float, float, float], factor: float = 0.78
) -> Tuple[float, float, float, float]:
    r, g, b, a = color
    return (factor * r, factor * g, factor * b, a)


def add_panel_direction_arrow(
    ax: plt.Axes,
    direction: str,
    label_size: float,
    color: str = "0.2",
) -> None:
    if direction == "forward":
        start, end, label = (0.08, 0.12), (0.28, 0.12), "forward"
        label_x = 0.18
    else:
        start, end, label = (0.93, 0.12), (0.73, 0.12), "backward"
        label_x = 0.83

    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=1.4, color=color),
    )
    ax.text(
        label_x,
        0.175,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=label_size,
        color=color,
    )


def first_finite(series: pd.Series) -> float | None:
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    vals = np.unique(vals)
    return float(vals[0])


def extract_event_markers(df: pd.DataFrame, boundary_type: str) -> Tuple[float | None, float | None]:
    top_delta = None
    bottom_delta = None

    if "scrit" in df.columns:
        top_delta = first_finite(df["scrit"])
    if top_delta is None and "delta_post" in df.columns:
        top_delta = first_finite(df["delta_post"])

    if "delta_event" in df.columns:
        if "hc_event" in df.columns:
            hc_event = df["hc_event"].astype(str).str.strip().str.lower()
            bottom_delta = first_finite(df.loc[hc_event == "invasion", "delta_event"])
        if bottom_delta is None:
            bottom_delta = first_finite(df["delta_event"])

    if top_delta is None:
        print(
            f"[WARN] Could not infer top marker for boundary_type='{boundary_type}' "
            "(need 'scrit' or 'delta_post')."
        )
    else:
        print(f"[INFO] Top marker delta ({boundary_type}): {top_delta:.6g}")

    if bottom_delta is None:
        print(
            f"[WARN] Could not infer bottom marker for boundary_type='{boundary_type}' "
            "(need 'delta_event')."
        )
    else:
        print(f"[INFO] Bottom marker delta ({boundary_type}): {bottom_delta:.6g}")

    return top_delta, bottom_delta


def plot_skeleton(
    ax: plt.Axes,
    algebraic: pd.DataFrame,
    species_colors: Dict[int, Tuple[float, float, float, float]],
    unstable_min_delta: float | None = None,
) -> None:
    if algebraic.empty:
        print("[WARN] No algebraic rows found; skeleton lines will be omitted.")
        return

    ordered = algebraic.assign(branch_rank=algebraic["branch_id"].map(BRANCH_ORDER)).sort_values(
        ["species_id", "branch_rank", "delta"], kind="mergesort"
    )

    for (branch_id, species_id), group in ordered.groupby(["branch_id", "species_id"], sort=False):
        if unstable_min_delta is not None and BRANCH_LINESTYLES.get(str(branch_id), "-") == "--":
            group = group[group["delta"] >= unstable_min_delta]
            if group.empty:
                continue
        ax.plot(
            group["delta"],
            group["abundance"].clip(lower=Y_FLOOR),
            color=species_colors[int(species_id)],
            linestyle=BRANCH_LINESTYLES[branch_id],
            linewidth=1.6,
            alpha=0.95,
            zorder=2,
        )


# ---------------------------------------------------------------------------
# I/O helpers (Panel A)
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Loaded: {path}  ({len(df)} rows, {len(df.columns)} cols)")
    return df


def validate_csv(df: pd.DataFrame) -> pd.DataFrame:
    required = ["delta", "species_id", "abundance", "source_type", "pass_direction",
                "branch_id", "scrit", "delta_event"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.copy()
    df["source_type"]    = df["source_type"].astype(str).str.strip().str.lower()
    df["pass_direction"] = df["pass_direction"].astype(str).str.strip().str.lower()
    df["branch_id"]      = df["branch_id"].astype("string").str.strip().str.lower()
    df["branch_id"]      = df["branch_id"].replace({"": pd.NA, "nan": pd.NA, "<na>": pd.NA})
    df["abundance"]      = pd.to_numeric(df["abundance"], errors="coerce")
    df["delta"]          = pd.to_numeric(df["delta"],     errors="coerce")
    df["scrit"]          = pd.to_numeric(df["scrit"],     errors="coerce")
    df["delta_event"]    = pd.to_numeric(df["delta_event"], errors="coerce")
    df["species_id"]     = pd.to_numeric(df["species_id"], errors="coerce").astype(int)
    return df


# ---------------------------------------------------------------------------
# Constants from boundary_prevalence_ecological_models.py
# ---------------------------------------------------------------------------

BASE = "model_runs"
_GIBBS_DIR = f"{BASE}/gibbs_128_dirs_from_gibbs_refgrid_n4to10_50reps_seed12345"
GIBBS_SOURCES = {
    4:  f"{_GIBBS_DIR}/*_n4_*.json",
    6:  f"{_GIBBS_DIR}/*_n6_*.json",
    8:  f"{_GIBBS_DIR}/*_n8_*.json",
    10: f"{_GIBBS_DIR}/*_n10_*.json",
}
_RMI_DIR  = f"{BASE}/karatayev_RMI_n4.6.8.10_50models_128dirs_seed1"
_FMI_DIR  = f"{BASE}/karatayev_FMI_n4.6.8.10_50models_128dirs_seed1"
_STF_DIR  = f"{BASE}/stouffer_random_n4.6.8.10_50models_128dirs_seed1"
_MOU_DIR  = f"{BASE}/mougi_random_n4.6.8.9_50models_128dirs_seed1"
_AGU_DIR  = f"{BASE}/aguade_n4.6.8.10_50models_128dirs_seed1"
_LEV_DIR  = f"{BASE}/lever_n4.6.8.10_50models_128dirs_seed1"

ECOLOGICAL_SOURCES = {
    "karat_fmi":    {4:  f"{_FMI_DIR}/*_n4_*ndirs128.json",
                     6:  f"{_FMI_DIR}/*_n6_*ndirs128.json",
                     8:  f"{_FMI_DIR}/*_n8_*ndirs128.json",
                     10: f"{_FMI_DIR}/*_n10_*ndirs128.json"},
    "karat_rmi":    {4:  f"{_RMI_DIR}/*_n4_*ndirs128.json",
                     6:  f"{_RMI_DIR}/*_n6_*ndirs128.json",
                     8:  f"{_RMI_DIR}/*_n8_*ndirs128.json",
                     10: f"{_RMI_DIR}/*_n10_*ndirs128.json"},
    "lever":        {4:  f"{_LEV_DIR}/*_n4_*ndirs128.json",
                     6:  f"{_LEV_DIR}/*_n6_*ndirs128.json",
                     8:  f"{_LEV_DIR}/*_n8_*ndirs128.json",
                     10: f"{_LEV_DIR}/*_n10_*ndirs128.json"},
    "mougi_random": {4:  f"{_MOU_DIR}/*_n4_*ndirs128.json",
                     6:  f"{_MOU_DIR}/*_n6_*ndirs128.json",
                     8:  f"{_MOU_DIR}/*_n8_*ndirs128.json",
                     9:  f"{_MOU_DIR}/*_n9_*ndirs128.json"},
    "aguade":       {4:  f"{_AGU_DIR}/*_n4_*ndirs128.json",
                     6:  f"{_AGU_DIR}/*_n6_*ndirs128.json",
                     8:  f"{_AGU_DIR}/*_n8_*ndirs128.json",
                     10: f"{_AGU_DIR}/*_n10_*ndirs128.json"},
    "stouffer":     {4:  f"{_STF_DIR}/*_n4_*ndirs128.json",
                     6:  f"{_STF_DIR}/*_n6_*ndirs128.json",
                     8:  f"{_STF_DIR}/*_n8_*ndirs128.json",
                     10: f"{_STF_DIR}/*_n10_*ndirs128.json"},
}


# ---------------------------------------------------------------------------
# Helpers from boundary_prevalence_ecological_models.py
# ---------------------------------------------------------------------------

def fold_fraction(filepath):
    with open(filepath) as fh:
        data = json.load(fh)

    scan_results = data.get("scan_results", [])
    if not scan_results:
        return None

    directions = scan_results[0].get("directions", [])
    if not directions:
        return None

    n_fold = sum(1 for row in directions if row["flag"] == "fold")
    return data["alpha_eff"], n_fold / len(directions)


def existing_points(pattern):
    pts = []
    for filepath in glob.glob(pattern):
        point = fold_fraction(filepath)
        if point is not None:
            pts.append(point)
    return pts


def group_stats(pts):
    alphas, ffs = zip(*pts)
    return np.mean(alphas), np.std(alphas), np.mean(ffs), np.std(ffs)


def summarize_group(pattern):
    pts = existing_points(pattern)
    return [group_stats(pts)] if pts else []


def build_groups():
    # gibbs[n][regime] = list of (alpha, fold_frac) points
    gibbs_raw = {n: defaultdict(list) for n in GIBBS_SOURCES}
    for n, pattern in GIBBS_SOURCES.items():
        for filepath in glob.glob(pattern):
            with open(filepath) as fh:
                data = json.load(fh)
            point = fold_fraction(filepath)
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
        for n in GIBBS_SOURCES
    }

    eco = {}
    for key, n_patterns in ECOLOGICAL_SOURCES.items():
        eco[key] = {
            n: ([group_stats(pts)] if (pts := existing_points(pattern)) else [])
            for n, pattern in n_patterns.items()
        }

    return {"gibbs": gibbs, **eco}


def plot_pts(ax, pts, marker, color, size, eb_kw, pt_alpha=1.0):
    import matplotlib.colors as mcolors
    rgba = list(mcolors.to_rgba(color))
    rgba[3] = pt_alpha
    rgba = tuple(rgba)
    for mean_alpha, std_alpha, mean_ff, std_ff in pts:
        ax.errorbar(mean_alpha, mean_ff, xerr=std_alpha, yerr=std_ff, fmt="none", ecolor=rgba, **eb_kw)
        ax.scatter(
            [mean_alpha],
            [mean_ff],
            marker=marker,
            color=rgba,
            s=size,
            linewidths=0.6,
            edgecolors=(0.0, 0.0, 0.0, pt_alpha),
            zorder=3,
        )


class MultiMarkerHandler(HandlerBase):
    def __init__(self, markers, colors, msize=6):
        self.markers = markers
        self.colors = colors
        self.msize = msize
        super().__init__()

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        artists = []
        n = len(self.markers)
        for i, (marker, color) in enumerate(zip(self.markers, self.colors)):
            x = xdescent + (i + 0.5) * width / n
            y = ydescent + height / 2
            line = mlines.Line2D(
                [x], [y],
                linestyle="none",
                marker=marker,
                color=color,
                markersize=self.msize,
                markeredgewidth=0.8,
                markeredgecolor="black",
                transform=trans,
            )
            artists.append(line)
        return artists


# ---------------------------------------------------------------------------
# Panel drawing functions
# ---------------------------------------------------------------------------

def _add_centered_title(ax: plt.Axes, label_size: float, tick_label_size: float) -> None:
    """Add 'Lever et al (2014) (★)' centered above ax, star matching the scatter plot marker."""
    c_lever = [c["color"] for c in plt.rcParams["axes.prop_cycle"]][5]
    msize = label_size * 1.2 * 0.9 * 0.9 * 0.7 * 1.15  # star marker size

    # Draw text pieces at dummy positions first to measure widths
    fig = ax.get_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bb = ax.get_window_extent(renderer)

    label1_t = ax.text(0, 1.03, "Lever et al (2014, ", transform=ax.transAxes,
                       fontsize=label_size, color="black", ha="left", va="bottom")
    label2_t = ax.text(0, 1.03, ")", transform=ax.transAxes,
                       fontsize=label_size, color="black", ha="left", va="bottom")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bb    = ax.get_window_extent(renderer)
    l1_bb    = label1_t.get_window_extent(renderer)
    l2_bb    = label2_t.get_window_extent(renderer)

    # Star width in display pixels: markersize is in points
    star_px = msize / 72.0 * fig.dpi

    total_w = l1_bb.width + star_px + l2_bb.width
    left_px = ax_bb.x0 + 0.5 * ax_bb.width - 0.5 * total_w

    label1_t.set_position(((left_px - ax_bb.x0) / ax_bb.width, 1.03))
    label2_t.set_position(((left_px + l1_bb.width + star_px - ax_bb.x0) / ax_bb.width, 1.03))

    # Place star marker centred vertically with the text, after redrawing for updated positions
    fig.canvas.draw()
    l1_bb2     = label1_t.get_window_extent(renderer)
    star_cx_px = l1_bb2.x1 + star_px / 2
    star_cy_px = l1_bb2.y0 + l1_bb2.height * 0.55
    star_x_ax  = (star_cx_px - ax_bb.x0) / ax_bb.width
    star_y_ax  = (star_cy_px - ax_bb.y0) / ax_bb.height

    ax.plot([star_x_ax], [star_y_ax],
            marker="*", markersize=msize,
            color=c_lever,
            markeredgecolor="black", markeredgewidth=0.5,
            linestyle="none",
            transform=ax.transAxes,
            clip_on=False)


def draw_panel_a(ax: plt.Axes, df: pd.DataFrame, label_size: float, tick_label_size: float,
                 label_x: float = 0.03) -> None:
    algebraic    = df[df["source_type"] == "algebraic"].copy()
    forward_dyn  = df[(df["source_type"] == "dynamic") & (df["pass_direction"] == "forward")].copy()
    backward_dyn = df[(df["source_type"] == "dynamic") & (df["pass_direction"] == "backward")].copy()

    species_ids    = np.sort(df["species_id"].unique())
    species_colors = build_species_colors(species_ids)

    top_marker, bottom_marker = extract_event_markers(df, "fold")

    all_x   = pd.to_numeric(df["delta"], errors="coerce").dropna()
    x_range = float(all_x.max() - all_x.min()) if len(all_x) > 1 else 1.0
    delta_offset = 0.004 * x_range

    plot_skeleton(ax, algebraic, species_colors, unstable_min_delta=bottom_marker)

    if not forward_dyn.empty:
        forward_dyn = forward_dyn.sort_values(["species_id", "delta"], kind="mergesort")
        ax.scatter(
            forward_dyn["delta"] + delta_offset,
            forward_dyn["abundance"].clip(lower=Y_FLOOR),
            marker=FORWARD_MARKER,
            color=FORWARD_COLOR,
            s=24,
            linewidths=0.8,
            alpha=0.55,
            zorder=3,
        )
    else:
        print("[WARN] No forward dynamic rows; forward scatter omitted.")

    if not backward_dyn.empty:
        backward_dyn = backward_dyn.sort_values(["species_id", "delta"], kind="mergesort")
        ax.scatter(
            backward_dyn["delta"] - delta_offset,
            backward_dyn["abundance"].clip(lower=Y_FLOOR),
            marker=BACKWARD_MARKER,
            color=BACKWARD_COLOR,
            s=24,
            linewidths=0.8,
            alpha=0.55,
            zorder=3,
        )
    else:
        print("[WARN] No backward dynamic rows; backward scatter omitted.")

    if top_marker is not None:
        ax.axvline(top_marker,    color=FORWARD_COLOR,  linestyle="--", linewidth=1.2, zorder=1)
    if bottom_marker is not None:
        ax.axvline(bottom_marker, color=BACKWARD_COLOR, linestyle="--", linewidth=1.2, zorder=1)

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, zorder=1)

    add_panel_direction_arrow(ax, "forward",  tick_label_size, color=FORWARD_COLOR)
    add_panel_direction_arrow(ax, "backward", tick_label_size, color=BACKWARD_COLOR)

    y_vals = pd.to_numeric(df["abundance"], errors="coerce").to_numpy(dtype=float)
    y_vals = y_vals[np.isfinite(y_vals)]
    if y_vals.size > 0:
        y_vals = np.clip(y_vals, Y_FLOOR, None)
        y_max  = max(float(np.max(y_vals)), 3.0)
        y_pad  = 0.03 * (y_max - Y_FLOOR)
        y_top  = y_max + y_pad
    else:
        y_top = 2.0

    ax.set_ylim(Y_FLOOR, y_top)
    ax.set_yticks([0.0, 1.0, 2.0, 3.0])

    x_vals = all_x.to_numpy(dtype=float)
    markers_arr = np.array(
        [m for m in [top_marker, bottom_marker] if m is not None], dtype=float
    )
    if markers_arr.size > 0:
        x_vals = np.concatenate([x_vals, markers_arr])
    if x_vals.size > 0:
        x_min = float(np.min(x_vals))
        x_max = float(np.max(x_vals))
        span  = x_max - x_min
        pad   = 0.015 * span if span > 0 else max(1e-3, 0.015 * max(abs(x_min), 1.0))
        ax.set_xlim(x_min - pad, x_max + pad)

    ax.set_xlabel(r"Driver of pollinator decline ($d_A$)", fontsize=label_size, labelpad=2)
    ax.set_ylabel(r"Abundance of pollinators",             fontsize=label_size)
    ax.tick_params(labelsize=tick_label_size)
    ax.grid(False)

    fwd_marker_handle    = mlines.Line2D([], [], marker=FORWARD_MARKER,  color=FORWARD_COLOR,
                                         linestyle="none", markersize=5)
    bwd_marker_handle    = mlines.Line2D([], [], marker=BACKWARD_MARKER, color=BACKWARD_COLOR,
                                         linestyle="none", markersize=5)
    stable_line_handle   = mlines.Line2D([], [], color="black", linestyle="-",  linewidth=1.6)
    unstable_line_handle = mlines.Line2D([], [], color="black", linestyle="--", linewidth=1.6)
    ax.legend(
        handles=[(bwd_marker_handle, fwd_marker_handle), stable_line_handle, unstable_line_handle],
        labels=["Integration endpoints", "Stable branch", "Unstable branch"],
        handler_map={(bwd_marker_handle, fwd_marker_handle): HandlerTuple(ndivide=2, pad=0)},
        fontsize=tick_label_size,
        loc="upper right",
        frameon=True,
        framealpha=0.6,
        edgecolor="none",
        borderpad=0.5,
        handlelength=1.8,
        handletextpad=0.3,
    )
    ax.text(-0.02, 1.05, "a", transform=ax.transAxes,
            fontsize=label_size, fontweight="bold", va="bottom", ha="left")

    # Title: green star (with black border) + black text, centered over axes
    _add_centered_title(ax, label_size, tick_label_size)


def draw_panel_b(ax: plt.Axes, groups: dict, label_size: float, tick_label_size: float) -> None:
    _t10 = [c["color"] for c in plt.rcParams["axes.prop_cycle"]]

    # Each regime and each other model type gets its own distinct tab10 color
    c_gibbs    = {"Q1": _t10[0], "Q2": _t10[1], "Q3": _t10[2]}
    c_karat    = {"FMI": _t10[3], "RMI": _t10[4]}
    c_lever    = _t10[5]
    c_mougi    = _t10[6]
    c_aguade   = _t10[7]
    c_stouffer = _t10[8]

    # Collect all n values across gibbs and ecological models for a unified alpha scale
    all_n = sorted({n for src in [groups["gibbs"], *[groups[k] for k in
                    ("karat_fmi","karat_rmi","lever","mougi_random","aguade","stouffer")]]
                    for n in src})
    n_min, n_max = all_n[0], all_n[-1]

    def n_alpha(n):
        # linearly from 0.2 (most transparent) at n_min to 1.0 (opaque) at n_max
        t = (n - n_min) / max(n_max - n_min, 1)
        return 0.2 + 0.8 * t

    marker_gibbs    = "d"
    marker_karat    = "P"
    marker_lever    = "*"
    marker_mougi    = "o"
    marker_aguade   = "h"
    marker_stouffer = "^"

    eb_kw = dict(elinewidth=0.5, capsize=0, zorder=2)

    for n in sorted(groups["gibbs"]):
        a = n_alpha(n)
        for regime in ("Q1", "Q2", "Q3"):
            plot_pts(ax, groups["gibbs"][n][regime], marker_gibbs, c_gibbs[regime], 32, eb_kw, pt_alpha=a)

    for n, pts in sorted(groups["karat_fmi"].items()):
        plot_pts(ax, pts, marker_karat, c_karat["FMI"], 32, eb_kw, pt_alpha=n_alpha(n))
    for n, pts in sorted(groups["karat_rmi"].items()):
        plot_pts(ax, pts, marker_karat, c_karat["RMI"], 32, eb_kw, pt_alpha=n_alpha(n))
    for n, pts in sorted(groups["lever"].items()):
        plot_pts(ax, pts, marker_lever, c_lever, 50, eb_kw, pt_alpha=n_alpha(n))
    for n, pts in sorted(groups["mougi_random"].items()):
        plot_pts(ax, pts, marker_mougi, c_mougi, 28, eb_kw, pt_alpha=n_alpha(n))
    for n, pts in sorted(groups["aguade"].items()):
        plot_pts(ax, pts, marker_aguade, c_aguade, 32, eb_kw, pt_alpha=n_alpha(n))
    for n, pts in sorted(groups["stouffer"].items()):
        plot_pts(ax, pts, marker_stouffer, c_stouffer, 30, eb_kw, pt_alpha=n_alpha(n))

    ax.set_xlabel(r"Non-linearity strength ($\alpha_\mathrm{eff}$)", fontsize=label_size)
    ax.set_ylabel("Fraction abrupt boundaries", fontsize=label_size)
    margin = 0.03
    ax.set_xlim(-margin, 1 + margin)
    ax.set_ylim(-margin, 1 + margin)
    ax.tick_params(axis="both", labelsize=tick_label_size)
    ax.set_yticks([0.0, 0.5, 1.0], labels=["0", "0.5", "1"])
    ax.set_aspect("equal")
    ax.text(-0.02, 1.05, "b", transform=ax.transAxes,
            fontsize=label_size, fontweight="bold", va="bottom", ha="left")

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
        gibbs_proxy:    MultiMarkerHandler([marker_gibbs] * 3,
                                           [c_gibbs["Q1"], c_gibbs["Q2"], c_gibbs["Q3"]], msize=5.7),
        karat_proxy:    MultiMarkerHandler([marker_karat, marker_karat],
                                           [c_karat["FMI"], c_karat["RMI"]], msize=5.7),
        lever_proxy:    MultiMarkerHandler([marker_lever],    [c_lever],    msize=7.1),
        mougi_proxy:    MultiMarkerHandler([marker_mougi],    [c_mougi],    msize=5.3),
        aguade_proxy:   MultiMarkerHandler([marker_aguade],   [c_aguade],   msize=5.7),
        stouffer_proxy: MultiMarkerHandler([marker_stouffer], [c_stouffer], msize=5.5),
    }

    ax.legend(
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

    # Horizontal colorbar below legend: transparency encodes n (species richness)
    c_agu_rgba = np.array(mpl.colors.to_rgba(c_aguade))
    white = np.array([1.0, 1.0, 1.0])
    n_steps = 256
    alphas_grad = np.linspace(0.2, 1.0, n_steps)
    cmap_arr = np.zeros((n_steps, 4))
    cmap_arr[:, :3] = c_agu_rgba[:3] * alphas_grad[:, None] + white * (1 - alphas_grad[:, None])
    cmap_arr[:, 3] = 1.0
    n_cmap = mpl.colors.ListedColormap(cmap_arr)
    sm = mpl.cm.ScalarMappable(cmap=n_cmap, norm=mpl.colors.Normalize(vmin=n_min, vmax=n_max))
    sm.set_array([])

    fig = ax.get_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    leg_bb = ax.get_legend().get_window_extent(renderer)
    fig_w_px, fig_h_px = fig.get_size_inches() * fig.dpi

    ax_pos = ax.get_position()
    leg_cx_fig  = (leg_bb.x0 + leg_bb.x1) / 2 / fig_w_px
    cbar_width  = 0.55 * ax_pos.width
    cbar_left   = leg_cx_fig - cbar_width / 2
    cbar_height = 0.055 * ax_pos.height
    gap_fig     = 8 / fig_h_px   # 8 px gap between legend bottom and colorbar top
    # title sits above the bar; account for its height so the bar itself is below the legend
    title_h_fig = (tick_label_size - 1) / 72 * fig.dpi / fig_h_px
    cbar_bottom = leg_bb.y0 / fig_h_px - gap_fig - title_h_fig - cbar_height
    cbar_axes = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    cb = fig.colorbar(sm, cax=cbar_axes, orientation="horizontal")
    cb.set_ticks(all_n)
    cb.set_ticklabels([str(n) for n in all_n])
    cb.ax.tick_params(labelsize=tick_label_size - 1, length=2)
    cbar_axes.set_title("Diversity ($n$)", fontsize=tick_label_size, pad=2)


# ---------------------------------------------------------------------------
# Combined figure
# ---------------------------------------------------------------------------

def _measure_panel_b_extents_in(groups: dict, label_size: float, tick_label_size: float):
    """Render Panel B in a throw-away figure; return (left_overflow, legend_width) in inches.

    left_overflow — how far y-axis content protrudes left of the axes box.
    legend_width  — width of the outside legend.
    """
    fig_t, ax_t = plt.subplots(figsize=(8, 8))
    draw_panel_b(ax_t, groups, label_size, tick_label_size)
    fig_t.canvas.draw()
    renderer  = fig_t.canvas.get_renderer()
    ax_bb     = ax_t.get_window_extent(renderer)
    tight_bb  = ax_t.get_tightbbox(renderer)
    left_ovf  = max(0.0, ax_bb.x0 - tight_bb.x0) / fig_t.dpi
    leg_w     = ax_t.get_legend().get_window_extent(renderer).width / fig_t.dpi
    plt.close(fig_t)
    return left_ovf, leg_w


def make_combined_figure(df: pd.DataFrame, groups: dict, output_stem: Path, dpi: int) -> None:
    label_size      = mpl.font_manager.FontProperties(
        size=mpl.rcParams["axes.labelsize"]
    ).get_size_in_points()
    tick_label_size = label_size * 0.8

    # Measure Panel B's y-axis left overflow so the gap is exact (no overlap)
    left_ovf, _ = _measure_panel_b_extents_in(groups, label_size, tick_label_size)
    print(f"[INFO] Panel B y-axis left overflow: {left_ovf:.3f} in")

    left_m  = 0.17 * 4.45   # 0.757 in — fixed margin for y-axis label
    right_m = 0.10
    top_m   = 0.10
    bot_m   = 0.50

    # Establish the reference total fig_width using the previous scale (0.85)
    # with the minimum non-overlapping gap; new panels are 10 % smaller and the
    # freed space goes entirely into gap_AB so fig_width stays constant.
    _s0     = 0.85
    _gap0   = left_ovf + 0.12
    fig_width = left_m + 3.605 * _s0 + _gap0 + 2.50 * _s0 + right_m

    scale   = 0.85 * 0.90     # 10 % additional reduction on top of previous 0.85
    H       = 2.50  * scale   # common panel height (in)
    W_ax_A  = 3.605 * scale   # Panel A keeps original proportions
    W_B     = H               # Panel B is square

    # gap absorbs the space freed by the 10 % reduction, preserving fig_width
    gap_AB  = fig_width - left_m - W_ax_A - W_B - right_m

    fig_height = bot_m + H + top_m

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Panel A — left
    ax_a = fig.add_axes([
        left_m / fig_width,
        bot_m / fig_height,
        W_ax_A / fig_width,
        H / fig_height,
    ])

    # Panel B — right, square; legend extends past right_m via tight save
    ax_b = fig.add_axes([
        (left_m + W_ax_A + gap_AB) / fig_width,
        bot_m / fig_height,
        W_B / fig_width,
        H / fig_height,
    ])

    draw_panel_a(ax_a, df, label_size, tick_label_size)
    draw_panel_b(ax_b, groups, label_size, tick_label_size)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path,           bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"[INFO] Wrote PNG: {png_path}")
    print(f"[INFO] Wrote PDF: {pdf_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    script_dir    = Path(__file__).resolve().parent
    default_input = script_dir / "data" / "lever_bifurcation_branches.csv"
    default_output = script_dir.parent / "pdffiles" / "si" / "combined_figure"

    parser = argparse.ArgumentParser(
        description="Combined bifurcation + scatter two-panel figure."
    )
    parser.add_argument("--input",  default=str(default_input),
                        help=f"Input CSV for Panel A (default: {default_input})")
    parser.add_argument("--output", default=str(default_output),
                        help=f"Output stem without extension (default: {default_output})")
    parser.add_argument("--dpi",    type=int, default=300,
                        help="PNG resolution (default: 300)")
    args = parser.parse_args()

    output_stem = Path(args.output)
    if output_stem.suffix:
        output_stem = output_stem.with_suffix("")

    df = load_csv(Path(args.input))
    df = validate_csv(df)

    print("[INFO] Building Panel B groups from model runs...")
    groups = build_groups()

    make_combined_figure(df, groups, output_stem, args.dpi)


if __name__ == "__main__":
    main()
