#!/usr/bin/env python3
"""
Reads pauls_idea_branches.csv and produces a single-panel bifurcation diagram
with the aesthetic of panel B ("Abrupt collapse") from hysteresis_panels.py.

Usage:
    python pauls_idea_plot.py [--input PATH] [--output STEM] [--dpi INT]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Style constants (verbatim from hysteresis_panels.py)
# ---------------------------------------------------------------------------

Y_FLOOR = -0.1
COOLWARM_QUAL3_NO_GRAY = ["#3F7F93", "#7B6FAE", "#A8752E"]
FORWARD_COLOR  = "#8B1A1A"   # dark red
BACKWARD_COLOR = "#1F4E79"   # dark blue
FORWARD_MARKER  = ">"
BACKWARD_MARKER = "<"

BRANCH_LINESTYLES = {
    "pre_fold_1":        "-",
    "pre_fold_2":        "--",
    "post_forward":      "-",
    "post_backward":     "-",
    "extinction_forward":  "-",
    "extinction_backward": "--",
}

BRANCH_ORDER = {
    "pre_fold_1":        0,
    "pre_fold_2":        1,
    "post_forward":      2,
    "post_backward":     3,
    "extinction_forward":  4,
    "extinction_backward": 5,
}


# ---------------------------------------------------------------------------
# Utility functions (verbatim from hysteresis_panels.py)
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
# I/O helpers
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
# Main plot
# ---------------------------------------------------------------------------

def make_plot(df: pd.DataFrame, output_stem: Path, dpi: int) -> None:
    algebraic    = df[df["source_type"] == "algebraic"].copy()
    forward_dyn  = df[(df["source_type"] == "dynamic") & (df["pass_direction"] == "forward")].copy()
    backward_dyn = df[(df["source_type"] == "dynamic") & (df["pass_direction"] == "backward")].copy()

    species_ids    = np.sort(df["species_id"].unique())
    species_colors = build_species_colors(species_ids)

    label_size     = mpl.font_manager.FontProperties(
        size=mpl.rcParams["axes.labelsize"]
    ).get_size_in_points()
    main_text_size = label_size * 0.85
    title_size     = main_text_size * 1.10

    fig = plt.figure(figsize=(4.45, 3.20))
    ax  = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.17, top=0.95)

    top_marker, bottom_marker = extract_event_markers(df, "fold")

    all_x   = pd.to_numeric(df["delta"], errors="coerce").dropna()
    x_range = float(all_x.max() - all_x.min()) if len(all_x) > 1 else 1.0
    delta_offset = 0.004 * x_range

    # Skeleton lines
    plot_skeleton(ax, algebraic, species_colors, unstable_min_delta=bottom_marker)

    # Forward dynamic markers
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

    # Backward dynamic markers
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

    # Vertical dashed lines at collapse and recovery
    if top_marker is not None:
        ax.axvline(top_marker,    color=FORWARD_COLOR,  linestyle="--", linewidth=1.2, zorder=1)
    if bottom_marker is not None:
        ax.axvline(bottom_marker, color=BACKWARD_COLOR, linestyle="--", linewidth=1.2, zorder=1)

    # Extinction line
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, zorder=1)

    # Direction arrows
    add_panel_direction_arrow(ax, "forward",  main_text_size, color=FORWARD_COLOR)
    add_panel_direction_arrow(ax, "backward", main_text_size, color=BACKWARD_COLOR)

    # Y-axis
    y_vals = pd.to_numeric(df["abundance"], errors="coerce").to_numpy(dtype=float)
    y_vals = y_vals[np.isfinite(y_vals)]
    if y_vals.size > 0:
        y_vals = np.clip(y_vals, Y_FLOOR, None)
        y_max  = max(float(np.max(y_vals)), 2.0)
        y_pad  = 0.03 * (y_max - Y_FLOOR)
        y_top  = y_max + y_pad
    else:
        y_top = 2.0

    ax.set_ylim(Y_FLOOR, y_top)
    ax.set_yticks([0.0, 1.0, 2.0])

    # X-axis limits
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

    ax.set_xlabel(r"Driver of pollinator decline ($d_A$)", fontsize=title_size, labelpad=2)
    ax.set_ylabel(r"Abundance of pollinators",             fontsize=title_size)
    ax.tick_params(labelsize=main_text_size)
    ax.grid(False)

    # Legend (same HandlerTuple style as hysteresis_panels.py lines 703–719)
    fwd_marker_handle   = mlines.Line2D([], [], marker=FORWARD_MARKER,  color=FORWARD_COLOR,
                                        linestyle="none", markersize=5)
    bwd_marker_handle   = mlines.Line2D([], [], marker=BACKWARD_MARKER, color=BACKWARD_COLOR,
                                        linestyle="none", markersize=5)
    stable_line_handle  = mlines.Line2D([], [], color="black", linestyle="-",  linewidth=1.6)
    unstable_line_handle= mlines.Line2D([], [], color="black", linestyle="--", linewidth=1.6)
    ax.legend(
        handles=[(bwd_marker_handle, fwd_marker_handle), stable_line_handle, unstable_line_handle],
        labels=["Integration endpoints", "Stable branch", "Unstable branch"],
        handler_map={(bwd_marker_handle, fwd_marker_handle): HandlerTuple(ndivide=2, pad=0)},
        fontsize=main_text_size,
        loc="upper left",
        frameon=False,
        borderpad=0.5,
        handlelength=1.8,
        handletextpad=0.3,
    )

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
    script_dir         = Path(__file__).resolve().parent
    default_input      = script_dir / "data" / "pauls_idea_branches.csv"
    default_output_dir = script_dir / "pdffiles"

    parser = argparse.ArgumentParser(
        description="Plot pauls_idea bifurcation diagram from branches CSV."
    )
    parser.add_argument("--input",  default=str(default_input),
                        help=f"Input CSV path (default: {default_input})")
    parser.add_argument("--output", default=None,
                        help="Output stem without extension "
                             f"(default: {default_output_dir}/pauls_idea_branches)")
    parser.add_argument("--dpi",    type=int, default=300,
                        help="PNG resolution (default: 300)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output is None:
        output_stem = default_output_dir / "pauls_idea_branches"
    else:
        output_stem = Path(args.output)
        if output_stem.suffix:
            output_stem = output_stem.with_suffix("")

    df = load_csv(input_path)
    df = validate_csv(df)
    make_plot(df, output_stem, args.dpi)


if __name__ == "__main__":
    main()
