#!/usr/bin/env python3
"""Plot post-boundary properties from JSON pipeline data (Fig 4 equivalent).

Reads model JSON files produced by the new_code pipeline
(generate_bank → boundary_scan → post_boundary_dynamics → backtrack_perturbation)
and produces a three-panel figure identical in layout to plot_fig4_boundary_dynamics.py:

  Panel A — Minimum abundance at fold boundary          (from scan_results)
  Panel B — Boundary distance (delta_c) at fold boundary (from scan_results)
  Panel C — Reversal fraction (%)                       (from backtrack_results)

Usage:
    python figures/plot_post_boundary_properties.py
    python figures/plot_post_boundary_properties.py \\
        --run-dir model_runs/minimal_test \\
        --output figures/pdffiles/post_boundary_properties.pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import LogFormatterSciNotation


# ── visual constants (identical to plot_fig4_boundary_dynamics.py) ───────────
LIGHT_POINT_SIZE = 18 * 0.65
DARK_POINT_SIZE  = 50 * 0.65
DARK_EDGE_WIDTH  = 0.6


# ── data loading ──────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())

def _to_float_or_nan(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if np.isfinite(out) else np.nan


def load_scan_rows(path: Path) -> list[dict]:
    """One row per (model, alpha, ray) from scan_results."""
    data = _load_json(path)
    if "scan_results" not in data:
        return []
    n        = int(data["n"])
    model_id = str(data.get("seed", path.stem))
    rows = []
    for alpha_block in data["scan_results"]:
        alpha = float(alpha_block["alpha"])
        for d in alpha_block["directions"]:
            x_boundary = d.get("x_boundary") or []
            rows.append({
                "model_id": model_id,
                "n":        n,
                "alpha":    alpha,
                "ray_id":   int(d["ray_id"]),
                "flag":     str(d["flag"]).lower(),
                "delta_c":  float(d["delta_c"]),
                "x_boundary_min": float(min(x_boundary)) if x_boundary else np.nan,
            })
    return rows


def load_backtrack_rows(path: Path) -> list[dict]:
    """One row per (model, alpha, ray) from backtrack_results."""
    data = _load_json(path)
    if "backtrack_results" not in data:
        return []
    n        = int(data["n"])
    model_id = str(data.get("seed", path.stem))
    rows = []
    for alpha_block in data["backtrack_results"]:
        alpha = float(alpha_block["alpha"])
        for d in alpha_block["directions"]:
            rows.append({
                "model_id":     model_id,
                "n":            n,
                "alpha":        float(d.get("alpha", alpha)),
                "ray_id":       int(d["ray_id"]),
                "boundary_flag": str(d.get("boundary_flag", "")).lower(),
                "class_label":  str(d.get("class_label", "")),
                "ode_ran":      bool(d.get("ode_ran", False)),
                "returned_n":   bool(d.get("returned_n", False)),
                "reversal_frac": _to_float_or_nan(d.get("reversal_frac")),
            })
    return rows


def collect_data(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all JSON files in run_dir and return (scan_df, backtrack_df)."""
    files = sorted(run_dir.rglob("*.json"))
    if not files:
        raise SystemExit(f"No JSON files found in {run_dir}")

    scan_rows, bt_rows = [], []
    for path in files:
        try:
            scan_rows.extend(load_scan_rows(path))
            bt_rows.extend(load_backtrack_rows(path))
        except Exception as exc:
            print(f"  Warning – skipping {path.name}: {exc}")

    scan_df = pd.DataFrame(scan_rows) if scan_rows else pd.DataFrame()
    bt_df   = pd.DataFrame(bt_rows)   if bt_rows   else pd.DataFrame()
    return scan_df, bt_df


# ── aggregation ───────────────────────────────────────────────────────────────

def aggregate_scan_panel(
    scan_df: pd.DataFrame, boundary: str, metric: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-(model, n, alpha) mean → per-(n, alpha) median.

    metric: "min" uses x_boundary_min; "distance" uses delta_c.
    Returns (sys_df, summary_df) with columns [model_id, n, alpha, value] and
    [n, alpha, median].
    """
    col = "x_boundary_min" if metric == "min" else "delta_c"
    sub = scan_df[(scan_df["flag"] == boundary) & scan_df[col].notna()].copy()
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()

    sys_df = (
        sub.groupby(["model_id", "n", "alpha"], as_index=False)[col]
        .mean()
        .rename(columns={col: "value"})
    )
    summary_df = (
        sys_df.groupby(["n", "alpha"], as_index=False)["value"]
        .median()
        .rename(columns={"value": "median"})
    )
    return sys_df, summary_df


def aggregate_reversal_frac(
    bt_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean reversal_frac on fold rays, including valid non-ODE success_to_zero.

    Returns (sys_df, summary_df) with value in [0, 1].  Only (model, alpha)
    pairs that have at least one included fold ray with finite reversal_frac
    are included.
    """
    fold = bt_df[
        (bt_df["boundary_flag"] == "fold")
        & (
            bt_df["ode_ran"]
            | (bt_df["class_label"] == "success_to_zero")
        )
        & bt_df["reversal_frac"].notna()
    ].copy()
    if fold.empty:
        return pd.DataFrame(), pd.DataFrame()

    agg = (
        fold.groupby(["model_id", "n", "alpha"], as_index=False)["reversal_frac"]
        .mean()
        .rename(columns={"reversal_frac": "value"})
    )
    summary_df = (
        agg.groupby(["n", "alpha"], as_index=False)["value"]
        .median()
        .rename(columns={"value": "median"})
    )
    return agg, summary_df


# ── colormap helpers ──────────────────────────────────────────────────────────

def build_colormaps(n_bins: int):
    colors     = np.asarray(sns.color_palette("flare", n_colors=n_bins))
    cmap_base  = mpl.colors.ListedColormap(colors)
    faded      = np.concatenate([colors.copy(), 0.25 * np.ones((n_bins, 1))], axis=1)
    cmap_faded = mpl.colors.ListedColormap(faded)
    norm       = mpl.colors.BoundaryNorm(np.arange(-0.5, n_bins + 0.5, 1), n_bins)
    return cmap_base, cmap_faded, norm


def make_color_dicts(n_values, cmap_base, cmap_faded):
    colors       = cmap_base (np.arange(len(n_values)))
    colors_faded = cmap_faded(np.arange(len(n_values)))
    color_map       = {n: colors[i]       for i, n in enumerate(n_values)}
    color_map_faded = {n: colors_faded[i] for i, n in enumerate(n_values)}
    return color_map, color_map_faded


# ── plotting helpers ──────────────────────────────────────────────────────────

def _scatter_n(ax, df, y_col, n_values, color_map, color_map_faded, scale=1.0):
    """Scatter light dots (per model) coloured by n."""
    for n_val in n_values:
        sub = df[df["n"] == n_val] if not df.empty else pd.DataFrame()
        if not sub.empty:
            ax.scatter(
                sub["alpha"], sub[y_col] * scale,
                color=color_map_faded[n_val],
                s=LIGHT_POINT_SIZE, linewidths=0.0,
            )


def _scatter_median(ax, df, y_col, n_values, color_map, scale=1.0):
    """Scatter dark median dots coloured by n."""
    for n_val in n_values:
        sub = df[df["n"] == n_val] if not df.empty else pd.DataFrame()
        if not sub.empty:
            sub = sub.sort_values("alpha")
            ax.scatter(
                sub["alpha"], sub[y_col] * scale,
                color=color_map[n_val],
                s=DARK_POINT_SIZE, edgecolors="black",
                linewidths=DARK_EDGE_WIDTH, zorder=3,
            )


def plot_scatter_panel(
    ax, sys_df, summary_df, y_col, ylabel, panel_label,
    n_values, color_map, color_map_faded, label_size, scale=1.0,
):
    _scatter_n     (ax, sys_df,     y_col, n_values, color_map, color_map_faded, scale)
    _scatter_median(ax, summary_df, "median", n_values, color_map, scale)
    ax.set_ylabel(ylabel, fontsize=label_size)
    ax.text(
        0.02, 0.96, panel_label,
        transform=ax.transAxes, ha="left", va="top",
        fontsize=label_size, fontweight="bold",
    )


def set_common_xticks(ax):
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xlim(-0.05, 1.05)


def set_square_aspect(ax):
    ax.set_box_aspect(1)


def add_right_n_colorbar(fig, axes, n_levels, cmap, norm, label_size):
    fig.canvas.draw()
    right_pos = axes[-1].get_position()
    cax = fig.add_axes([right_pos.x1 + 0.02, right_pos.y0, 0.015, right_pos.height])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, ticks=np.arange(len(n_levels)))
    cbar.ax.set_yticklabels([str(int(n)) for n in n_levels])
    cbar.ax.set_title("n", pad=4, fontsize=label_size)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot post-boundary properties (Fig 4 equivalent) from JSON pipeline data."
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=Path("model_runs/minimal_test"),
        help="Directory containing model JSON files (default: model_runs/minimal_test)",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("figures/pdffiles/post_boundary_properties.pdf"),
        help="Output PDF path (default: figures/pdffiles/post_boundary_properties.pdf)",
    )
    args = parser.parse_args()

    if not args.run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {args.run_dir}")

    # ── font scale (identical to plot_fig4_boundary_dynamics.py) ─────────────
    def resolve_font_size(value, key):
        if value is None:
            value = mpl.rcParamsDefault.get(key, mpl.rcParams["font.size"])
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        return mpl.font_manager.FontProperties(size=value).get_size_in_points()

    font_scale = 1.20
    mpl.rcParams.update({
        "font.size":       resolve_font_size(mpl.rcParams["font.size"],       "font.size")       * font_scale,
        "axes.labelsize":  resolve_font_size(mpl.rcParams["axes.labelsize"],  "axes.labelsize")  * font_scale,
        "axes.titlesize":  resolve_font_size(mpl.rcParams["axes.titlesize"],  "axes.titlesize")  * font_scale,
        "xtick.labelsize": resolve_font_size(mpl.rcParams["xtick.labelsize"], "xtick.labelsize") * font_scale,
        "ytick.labelsize": resolve_font_size(mpl.rcParams["ytick.labelsize"], "ytick.labelsize") * font_scale,
        "legend.fontsize": resolve_font_size(mpl.rcParams["legend.fontsize"], "legend.fontsize") * font_scale,
    })
    label_size = mpl.font_manager.FontProperties(
        size=mpl.rcParams["axes.labelsize"]
    ).get_size_in_points()

    # ── load & aggregate ──────────────────────────────────────────────────────
    print(f"Loading data from {args.run_dir} ...")
    scan_df, bt_df = collect_data(args.run_dir)

    if scan_df.empty:
        raise SystemExit("No scan_results data found in any JSON file.")

    n_values = sorted(int(n) for n in scan_df["n"].unique())
    print(f"  n values: {n_values}")
    print(f"  scan rows: {len(scan_df)}  |  backtrack rows: {len(bt_df)}")

    cmap_base, cmap_faded, norm = build_colormaps(len(n_values))
    color_map, color_map_faded = make_color_dicts(n_values, cmap_base, cmap_faded)

    sys_A, sum_A = aggregate_scan_panel(scan_df, "fold", "min")
    sys_B, sum_B = aggregate_scan_panel(scan_df, "fold", "distance")
    sys_C, sum_C = aggregate_reversal_frac(bt_df)

    print(f"  fold rows used – panel A: {len(sys_A)}  panel B: {len(sys_B)}")
    print(
        "  fold model×alpha points used – panel C "
        "(ode_ran or success_to_zero): "
        f"{len(sys_C)}"
    )

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.9), sharex=True, sharey=False)

    # Panel A – minimum abundance at fold boundary
    ax = axes[0]
    plot_scatter_panel(
        ax, sys_A, sum_A, "value",
        r"Minimum abundance ($x_{min}$)", "A",
        n_values, color_map, color_map_faded, label_size,
    )
    ax.set_ylim(-0.085, 1.085)
    ax.set_yticks([0.0, 0.5, 1.0])

    # Panel B – boundary distance (delta_c) at fold boundary
    ax = axes[1]
    plot_scatter_panel(
        ax, sys_B, sum_B, "value",
        r"Boundary distance ($\delta_c$)", "B",
        n_values, color_map, color_map_faded, label_size,
    )
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 10.0)
    ax.set_yticks([1e-2, 1e-1, 1.0, 10.0])
    ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10, labelOnlyBase=False))

    # Panel C – reversal fraction (%)
    ax = axes[2]
    plot_scatter_panel(
        ax, sys_C, sum_C, "value",
        "Reversal fraction (%)", "C",
        n_values, color_map, color_map_faded, label_size,
        scale=100.0,
    )
    ax.set_yticks([0.0, 50.0, 100.0])
    ax.set_ylim(-5.0, 105.0)

    # ── shared formatting ─────────────────────────────────────────────────────
    for ax in axes:
        set_common_xticks(ax)
        set_square_aspect(ax)
        ax.set_xlabel("")
    axes[1].set_xlabel("HOI strength ($\\alpha$)", fontsize=label_size)

    fig.tight_layout(rect=[0.02, 0.05, 0.90, 0.98])
    fig.subplots_adjust(wspace=0.354)
    add_right_n_colorbar(fig, axes, n_values, cmap_base, norm, label_size)

    # ── save ──────────────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
