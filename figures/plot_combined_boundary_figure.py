#!/usr/bin/env python3
"""Combined boundary figure v3.

5-panel layout:
  Left column  — two square main panels:
      A (top)  = fold boundary prevalence
      B (bot)  = negative boundary prevalence
  Right column — three smaller square aux panels (fold only):
      C (top)  = fold robustness (delta_c)
      E (mid)  = fold min abundance
      G (bot)  = fold hysteresis
Callout lines in TikZ from right-center of A to top-left of C and
bottom-left of G.  Y-axis labels on right for right column.

Usage:
    python figures/plot_combined_boundary_figure.py \
        --bank 1_bank_50_models_n_3-8_128_dirs_b_dirichlet \
        --output figures/pdffiles/combined_boundary_figure_v3.pdf
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ── constants ──────────────────────────────────────────────────────────────────

FLAG_REMAP = {"fold": "complex"}
DEFAULT_BOUNDARY_FLAGS = ["negative", "complex"]
ALL_BOUNDARY_FLAGS = ["negative", "complex", "unstable", "success"]

DARK_POINT_SIZE = 50 * 0.65
DARK_EDGE_WIDTH = 0.6


# ── boundary-fraction data loading ────────────────────────────────────────────

def boundary_fractions(df: pd.DataFrame, boundary_types: List[str]) -> pd.DataFrame:
    stable = df[df["stable"]].copy()
    if stable.empty:
        return pd.DataFrame(
            columns=["alpha_value", "boundary_type", "count", "total", "fraction"]
        )
    tmp = stable[["alpha_value", "boundary_type"]].copy()
    tmp["boundary_type"] = tmp["boundary_type"].astype(str).str.lower()
    counts = tmp.groupby(["alpha_value", "boundary_type"]).size().reset_index(name="count")
    totals = (
        counts.groupby("alpha_value", as_index=False)["count"]
        .sum()
        .rename(columns={"count": "total"})
    )
    if boundary_types:
        counts = counts[counts["boundary_type"].isin(boundary_types)]
        full_index = pd.MultiIndex.from_product(
            [totals["alpha_value"].unique(), boundary_types],
            names=["alpha_value", "boundary_type"],
        ).to_frame(index=False)
    else:
        full_index = counts[["alpha_value", "boundary_type"]].copy()
    out = full_index.merge(counts, on=["alpha_value", "boundary_type"], how="left")
    out["count"] = out["count"].fillna(0).astype(int)
    out = out.merge(totals, on="alpha_value", how="left")
    out["fraction"] = np.where(out["total"] > 0, out["count"] / out["total"], 0.0)
    return out


def collect_boundary_fractions(
    roots: List[Path], boundary_types: List[str]
) -> pd.DataFrame:
    frames = []
    for root in roots:
        if not root.is_dir():
            print(f"Missing directory: {root}")
            continue
        files = sorted(root.rglob("*.json"))
        if not files:
            print(f"No JSON files found under {root}")
            continue
        for path in files:
            try:
                data = json.loads(path.read_text())
            except Exception as exc:
                print(f"Skipping {path.name}: {exc}")
                continue
            if "scan_results" not in data:
                continue
            n        = int(data["n"])
            model_id = str(data.get("seed", path.stem))
            rows = []
            for alpha_result in data["scan_results"]:
                alpha = float(alpha_result["alpha"])
                for direction in alpha_result["directions"]:
                    raw_flag = str(direction["flag"]).lower()
                    boundary_type = FLAG_REMAP.get(raw_flag, raw_flag)
                    rows.append({
                        "alpha_value":   alpha,
                        "boundary_type": boundary_type,
                        "stable":        True,
                    })
            if not rows:
                continue
            df   = pd.DataFrame(rows)
            frac = boundary_fractions(df, boundary_types)
            if frac.empty:
                continue
            frac["model_id"] = model_id
            frac["n"]        = n
            frames.append(frac)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── post-boundary data loading ─────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _to_float_or_nan(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if np.isfinite(out) else np.nan


def load_scan_rows(path: Path) -> list:
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
                "model_id":  model_id,
                "n":         n,
                "alpha":     alpha,
                "ray_id":    int(d["ray_id"]),
                "flag":      str(d["flag"]).lower(),
                "delta_c":   float(d["delta_c"]),
                "x_boundary_min": float(min(x_boundary)) if x_boundary else np.nan,
            })
    return rows


def load_backtrack_rows(path: Path) -> list:
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
                "model_id":      model_id,
                "n":             n,
                "alpha":         float(d.get("alpha", alpha)),
                "ray_id":        int(d["ray_id"]),
                "boundary_flag": str(d.get("boundary_flag", "")).lower(),
                "class_label":   str(d.get("class_label", "")),
                "ode_ran":       bool(d.get("ode_ran", False)),
                "returned_n":    bool(d.get("returned_n", False)),
                "reversal_frac": _to_float_or_nan(d.get("reversal_frac")),
            })
    return rows


def collect_data(run_dirs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if isinstance(run_dirs, Path):
        run_dirs = [run_dirs]
    files = []
    for run_dir in run_dirs:
        files.extend(sorted(run_dir.rglob("*.json")))
    if not files:
        raise SystemExit(f"No JSON files found in {run_dirs}")
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
    scan_df: pd.DataFrame, boundary: str, metric: str,
    min_count_filter: dict = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    col = "x_boundary_min" if metric == "min" else "delta_c"
    sub = scan_df[(scan_df["flag"] == boundary) & scan_df[col].notna()].copy()
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()
    if min_count_filter is not None and boundary in min_count_filter:
        passing_df = pd.DataFrame(list(min_count_filter[boundary]), columns=["n", "alpha"])
        sub = sub.merge(passing_df, on=["n", "alpha"])
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
    boundary: str = "fold",
    min_count_filter: dict = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if bt_df.empty or "boundary_flag" not in bt_df.columns:
        return pd.DataFrame(), pd.DataFrame()
    fold = bt_df[
        (bt_df["boundary_flag"] == boundary)
        & (bt_df["ode_ran"] | (bt_df["class_label"] == "success_to_zero"))
        & bt_df["reversal_frac"].notna()
    ].copy()
    if fold.empty:
        return pd.DataFrame(), pd.DataFrame()
    if min_count_filter is not None and boundary in min_count_filter:
        passing_df = pd.DataFrame(list(min_count_filter[boundary]), columns=["n", "alpha"])
        fold = fold.merge(passing_df, on=["n", "alpha"])
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


def build_min_count_filters(scan_df: pd.DataFrame, threshold: float = 0.02) -> dict:
    """Return {flag: set of (n, alpha)} where count >= threshold * total for that (n, alpha)."""
    if scan_df.empty:
        return {}
    totals = scan_df.groupby(["n", "alpha"]).size().reset_index(name="total")
    flag_counts = scan_df.groupby(["n", "alpha", "flag"]).size().reset_index(name="count")
    merged = flag_counts.merge(totals, on=["n", "alpha"])
    merged = merged[merged["count"] >= threshold * merged["total"]]
    result = {}
    for flag, group in merged.groupby("flag"):
        result[flag] = set(zip(group["n"].astype(int), group["alpha"]))
    return result


# ── colormap helpers ──────────────────────────────────────────────────────────

def build_colormaps(n_bins: int) -> Tuple:
    colors     = np.array(sns.color_palette("flare", n_bins))
    cmap_base  = mpl.colors.ListedColormap(colors)
    faded      = np.concatenate([colors.copy(), 0.25 * np.ones((n_bins, 1))], axis=1)
    cmap_faded = mpl.colors.ListedColormap(faded)
    norm       = mpl.colors.BoundaryNorm(np.arange(-0.5, n_bins + 0.5, 1), n_bins)
    return cmap_base, cmap_faded, norm


def make_color_dicts(n_values, cmap_base, cmap_faded):
    colors       = cmap_base(np.arange(len(n_values)))
    colors_faded = cmap_faded(np.arange(len(n_values)))
    color_map       = {n: colors[i]       for i, n in enumerate(n_values)}
    color_map_faded = {n: colors_faded[i] for i, n in enumerate(n_values)}
    return color_map, color_map_faded


# ── scatter / errorbar helpers ────────────────────────────────────────────────

def _scatter_median(ax, df, y_col, n_values, color_map, scale=1.0):
    for n_val in n_values:
        sub = df[df["n"] == n_val] if not df.empty else pd.DataFrame()
        if not sub.empty:
            sub = sub.sort_values("alpha")
            ax.plot(
                sub["alpha"], sub[y_col] * scale,
                color=color_map[n_val], linewidth=0.9, zorder=2, alpha=0.85,
            )
            ax.scatter(
                sub["alpha"], sub[y_col] * scale,
                color=color_map[n_val],
                s=DARK_POINT_SIZE * 0.45, edgecolors="black",
                linewidths=DARK_EDGE_WIDTH * 0.4, zorder=3,
            )


def _errorbar_n(ax, df, y_col, n_values, color_map, scale=1.0):
    """IQR errorbars per (n, alpha) group, replacing faded individual dots."""
    for n_val in n_values:
        sub = df[df["n"] == n_val] if not df.empty else pd.DataFrame()
        if sub.empty:
            continue
        grp  = sub.groupby("alpha")[y_col]
        med  = grp.median()  * scale
        q25  = grp.quantile(0.25) * scale
        q75  = grp.quantile(0.75) * scale
        yerr = np.array([(med - q25).values, (q75 - med).values])
        ax.errorbar(
            med.index, med.values,
            yerr=yerr,
            fmt="none",
            color=color_map[n_val],
            elinewidth=0.9,
            capsize=2.0,
            alpha=0.55,
            zorder=2,
        )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Combined boundary figure v3: 5-panel layout."
    )
    parser.add_argument("--bank", type=str, default=None)
    parser.add_argument("--input-root", type=Path, nargs="+", default=None)
    parser.add_argument("--run-dir",    type=Path, nargs="+", default=None)
    parser.add_argument(
        "--output", type=Path,
        default=Path("figures/pdffiles/combined_boundary_figure.pdf"),
    )
    parser.add_argument("--plot-all", action="store_true")
    args = parser.parse_args()

    model_runs = Path("model_runs")
    if args.bank is not None:
        bank_path = model_runs / args.bank
        if not bank_path.is_dir():
            raise SystemExit(
                f"Bank directory not found: {bank_path}\n"
                f"Available banks: {[d.name for d in sorted(model_runs.iterdir()) if d.is_dir()]}"
            )
        if args.input_root is None:
            args.input_root = [bank_path]
        if args.run_dir is None:
            args.run_dir = [bank_path]
    else:
        if args.input_root is None:
            args.input_root = [model_runs]
        if args.run_dir is None:
            args.run_dir = [model_runs]

    # ── font scale ─────────────────────────────────────────────────────────
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
        "xtick.labelsize": resolve_font_size(mpl.rcParams["xtick.labelsize"], "xtick.labelsize") * font_scale * 0.80,
        "ytick.labelsize": resolve_font_size(mpl.rcParams["ytick.labelsize"], "ytick.labelsize") * font_scale * 0.80,
        "legend.fontsize": resolve_font_size(mpl.rcParams["legend.fontsize"], "legend.fontsize") * font_scale,
    })
    label_size = mpl.font_manager.FontProperties(
        size=mpl.rcParams["axes.labelsize"]
    ).get_size_in_points()

    # ── load data ──────────────────────────────────────────────────────────
    boundary_flags = list(ALL_BOUNDARY_FLAGS if args.plot_all else DEFAULT_BOUNDARY_FLAGS)

    print("Loading boundary fraction data...")
    all_frac = collect_boundary_fractions(args.input_root, boundary_flags)
    if all_frac.empty:
        print("  Warning: no boundary fraction data found; main panels will be empty.")

    print(f"Loading post-boundary data from {args.run_dir} ...")
    scan_df, bt_df = collect_data(list(args.run_dir))
    if scan_df.empty:
        raise SystemExit("No scan_results data found.")

    # ── shared colormap ────────────────────────────────────────────────────
    n_set = set(int(n) for n in scan_df["n"].unique())
    if not all_frac.empty:
        n_set |= set(int(n) for n in all_frac["n"])
    n_values = sorted(n_set)
    print(f"  n values: {n_values}")

    cmap, cmap_faded, norm = build_colormaps(len(n_values))
    n_map = {n: i for i, n in enumerate(n_values)}
    color_map, color_map_faded = make_color_dicts(n_values, cmap, cmap_faded)

    # ── min-count filter ───────────────────────────────────────────────────
    min_count_filters = build_min_count_filters(scan_df, threshold=0.02)

    # ── aggregate data (fold only for aux panels) ──────────────────────────
    sys_fold_dist, sum_fold_dist = aggregate_scan_panel(scan_df, "fold", "distance", min_count_filters)
    sys_fold_min,  sum_fold_min  = aggregate_scan_panel(scan_df, "fold", "min",      min_count_filters)
    sys_fold_rev,  sum_fold_rev  = aggregate_reversal_frac(bt_df,                    min_count_filter=min_count_filters)

    # ── y-limits for aux panels ────────────────────────────────────────────
    def _q75_max(df, y_col, scale=1.0, pad_top=1.0):
        if df.empty:
            return None
        return float(df.groupby(["n", "alpha"])[y_col].quantile(0.75).max()) * scale * pad_top

    def _q25_min(df, y_col, scale=1.0):
        if df.empty:
            return None
        return float(df.groupby(["n", "alpha"])[y_col].quantile(0.25).min()) * scale

    dist_ymax_raw = _q75_max(sys_fold_dist, "value", pad_top=1.15)
    dist_ymin_raw = _q25_min(sys_fold_dist, "value")
    shared_dist_ymin = max(dist_ymin_raw, 1e-6) if dist_ymin_raw is not None else 1e-6
    shared_dist_ymax = dist_ymax_raw if dist_ymax_raw is not None else 10.0

    min_ymax_raw = _q75_max(sys_fold_min, "value", pad_top=1.05)
    min_ymin_raw = _q25_min(sys_fold_min, "value")
    shared_min_ymin = (min(min_ymin_raw, 0.0) - 0.085) if min_ymin_raw is not None else -0.085
    shared_min_ymax = min_ymax_raw if min_ymax_raw is not None else 1.0

    fold_rev_ymax = (_q75_max(sys_fold_rev, "value", scale=100.0) or 100.) * 1.05

    # ── layout constants (inches) ──────────────────────────────────────────
    S       = 2.2     # side of A and B (square)
    gap_AB  = 0.35    # gap between A and B
    gap_CE  = 0.25    # gaps between C/E and E/G
    s       = (2 * S + gap_AB - 2 * gap_CE) / 3   # ≈ 1.4167" side of C, E, G

    YLABEL_L  = 0.85   # y-label space left of left column
    YLABEL_R  = 0.85   # y-label space right of right column
    GAP_COL   = 0.50   # empty gap between columns
    MARGIN_L  = 0.02
    MARGIN_R  = 0.03
    MARGIN_B  = 0.55
    MARGIN_T  = 0.35

    FIG_W = MARGIN_L + YLABEL_L + S + GAP_COL + s + YLABEL_R + MARGIN_R
    FIG_H = MARGIN_T + 2 * S + gap_AB + MARGIN_B

    # ── physical panel positions (inches, bottom-left origin) ──────────────
    x_L = MARGIN_L + YLABEL_L          # left edge of left column
    x_R = MARGIN_L + YLABEL_L + S + GAP_COL  # left edge of right column

    # Left column: A (top) and B (bottom)
    A_x = x_L;  A_y = MARGIN_B + S + gap_AB
    B_x = x_L;  B_y = MARGIN_B

    # Right column: C (top), E (mid), G (bot)
    C_x = x_R;  C_y = MARGIN_B + 2 * s + 2 * gap_CE
    E_x = x_R;  E_y = MARGIN_B + s + gap_CE
    G_x = x_R;  G_y = MARGIN_B

    # ── normalised panel positions ─────────────────────────────────────────
    def norm_rect(x_phys, y_phys, w_phys, h_phys):
        return [x_phys / FIG_W, y_phys / FIG_H, w_phys / FIG_W, h_phys / FIG_H]

    # ── create figure and axes ─────────────────────────────────────────────
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    ax_A = fig.add_axes(norm_rect(A_x, A_y, S, S))
    ax_B = fig.add_axes(norm_rect(B_x, B_y, S, S))
    ax_C = fig.add_axes(norm_rect(C_x, C_y, s, s))
    ax_E = fig.add_axes(norm_rect(E_x, E_y, s, s))
    ax_G = fig.add_axes(norm_rect(G_x, G_y, s, s))

    # Right column: y-ticks on the right
    for ax_r in (ax_C, ax_E, ax_G):
        ax_r.tick_params(left=False, labelleft=False, right=True, labelright=True)

    # ── helper: plot one auxiliary panel ───────────────────────────────────
    def _plot_aux(ax, sys_df, sum_df, y_col, panel_label, scale=1.0):
        _errorbar_n(ax, sys_df, y_col, n_values, color_map, scale)
        _scatter_median(ax, sum_df, "median", n_values, color_map, scale)
        ax.text(
            0.04, 0.96, panel_label,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=label_size, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5),
        )
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_xlim(-0.05, 1.05)

    # ── helper: plot one main boundary-fraction panel ──────────────────────
    def _plot_main(ax, flag, panel_label, title_text):
        margin = 0.03
        ax.set_xlim(-margin - 0.01, 1.0 + margin)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_xticklabels([])
        ax.tick_params(labelbottom=False)
        ax.set_ylim(-margin, 1.0 + margin)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_title(title_text, fontsize=label_size, pad=4)
        ax.text(
            0.0, 1.01, panel_label,
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=label_size, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5),
        )
        if all_frac.empty:
            return
        sub = all_frac[all_frac["boundary_type"] == flag].copy()
        if sub.empty:
            print(f"  No {flag} data; leaving main panel empty.")
            return

        for n_val in n_values:
            sub_n = sub[sub["n"] == n_val]
            if sub_n.empty:
                continue
            grp   = sub_n.groupby("alpha_value")["fraction"]
            med_e = grp.median()
            q25_e = grp.quantile(0.25)
            q75_e = grp.quantile(0.75)
            yerr  = np.array([(med_e - q25_e).values, (q75_e - med_e).values])
            ax.errorbar(
                med_e.index, med_e.values,
                yerr=yerr,
                fmt="none",
                color=color_map[n_val],
                elinewidth=0.9,
                capsize=2.0,
                alpha=0.55,
                zorder=2,
            )
        med = (
            sub.groupby(["n", "alpha_value"], as_index=False)["fraction"]
            .median()
            .reset_index(drop=True)
        )
        for n_val in n_values:
            sub_n = med[med["n"] == n_val].sort_values("alpha_value")
            if sub_n.empty:
                continue
            ax.plot(
                sub_n["alpha_value"], sub_n["fraction"],
                color=color_map[n_val], linewidth=0.9, zorder=2, alpha=0.85,
            )
            ax.scatter(
                sub_n["alpha_value"], sub_n["fraction"],
                color=color_map[n_val],
                s=DARK_POINT_SIZE * 0.45, edgecolors="black",
                linewidths=DARK_EDGE_WIDTH * 0.4, zorder=3,
            )

    # ── LEFT column: A (fold) and B (negative) ─────────────────────────────
    _plot_main(ax_A, "complex",  "A", "Abrupt")
    _plot_main(ax_B, "negative", "B", "Gradual")

    # B shows x-tick labels; A suppresses them
    ax_B.tick_params(labelbottom=True)
    ax_B.set_xticklabels(["0", "0.5", "1"])

    # ── RIGHT column: C (robustness), D (min abundance), E (hysteresis) ────
    _plot_aux(ax_C, sys_fold_dist, sum_fold_dist, "value", "C")
    ax_C.set_ylim(shared_dist_ymin, shared_dist_ymax)
    ax_C.tick_params(which="both", labelbottom=False, left=False, labelleft=False, right=True, labelright=True)

    _plot_aux(ax_E, sys_fold_min, sum_fold_min, "value", "D")
    ax_E.axhline(0, linestyle="--", color="black", linewidth=0.7, zorder=1)
    ax_E.set_ylim(shared_min_ymin, shared_min_ymax)
    ax_E.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, prune="lower"))
    ax_E.tick_params(labelbottom=False)

    _plot_aux(ax_G, sys_fold_rev, sum_fold_rev, "value", "E", scale=100.0)
    ax_G.axhline(0, linestyle="--", color="black", linewidth=0.7, zorder=1)
    ax_G.set_ylim(-5.0, fold_rev_ymax)
    ax_G.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, prune="lower"))
    ax_G.tick_params(labelbottom=True)
    ax_G.set_xticklabels(["0", "0.5", "1"])

    # ── y-axis labels ──────────────────────────────────────────────────────
    YLABEL_EXTRA_PAD = 0.18
    YLABEL_OFFSET    = 0.455 + YLABEL_EXTRA_PAD
    x_ylabel_L = (MARGIN_L + YLABEL_L - YLABEL_OFFSET) / FIG_W

    # Left column: one shared "Boundary prevalence" centered between A and B
    pos_A = ax_A.get_position()
    pos_B = ax_B.get_position()
    y_center_left = (pos_B.y0 + pos_A.y0 + pos_A.height) / 2
    fig.text(
        x_ylabel_L, y_center_left,
        "Boundary prevalence",
        rotation=90, ha="center", va="center", fontsize=label_size,
    )

    # Right column labels (right side, rotation=-90 so text reads top-to-bottom)
    x_ylabel_R = (MARGIN_L + YLABEL_L + S + GAP_COL + s + 0.60) / FIG_W
    for ax_lbl, lbl in [
        (ax_C, r"Robustness ($\delta_c$)"),
        (ax_E, r"Abruptness ($x_\mathrm{min}$)"),
        (ax_G, "Hysteresis (%)"),
    ]:
        pos = ax_lbl.get_position()
        fig.text(
            x_ylabel_R, pos.y0 + pos.height / 2,
            lbl,
            rotation=-90, ha="center", va="center", fontsize=label_size,
        )

    # ── shared x-axis label ────────────────────────────────────────────────
    cols_center_x = (x_L + (x_R + s)) / 2 / FIG_W
    fig.text(
        cols_center_x,
        (MARGIN_B * 0.18) / FIG_H,
        r"HOI strength ($\alpha$)",
        ha="center", va="center", fontsize=label_size,
    )

    # ── colorbar: vertical, inside panel A ────────────────────────────────
    INSET_W   = 0.55 * 1.25   # 0.6875"
    INSET_PAD = 0.09

    A_top = A_y + S
    inset_bottom_A = A_top - INSET_W - INSET_PAD

    _title_overhead = (label_size + 4) / 72
    CBAR_INNER_W = 0.08 * 1.15
    CBAR_INNER_H = 0.75

    cbar_x_phys = x_L + INSET_PAD
    cbar_y_phys = inset_bottom_A - INSET_PAD - _title_overhead - CBAR_INNER_H

    cax = fig.add_axes([
        cbar_x_phys / FIG_W,
        cbar_y_phys / FIG_H,
        CBAR_INNER_W / FIG_W,
        CBAR_INNER_H / FIG_H,
    ])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    target_ns = {3, 11, 20}
    tick_positions   = [i for i, n in enumerate(n_values) if int(n) in target_ns]
    tick_labels_cbar = [str(int(n)) for n in n_values if int(n) in target_ns]
    cbar = fig.colorbar(sm, cax=cax, ticks=tick_positions, orientation="vertical")
    cbar.ax.yaxis.set_ticks_position("right")
    cbar.ax.set_yticklabels(tick_labels_cbar)
    cbar.ax.minorticks_off()
    cbar.ax.set_title("n", pad=4, fontsize=label_size, loc="center")

    # ── save: main PDF then insets + callout lines via LaTeX ──────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Inset positions
    # fold_scheme → top-left corner of A
    x_ins_A = x_L + INSET_PAD
    y_ins_A = A_top - INSET_W - INSET_PAD

    # trans_critical_scheme → bottom-left corner of B
    x_ins_B = x_L + INSET_PAD
    y_ins_B = B_y + INSET_PAD

    # Callout line endpoints (physical inches)
    # Origin: right-center of A
    xo = x_L + S
    yo = A_y + S / 2
    # end C: top-left of C
    xC = x_R
    yC = C_y + s       # top edge of C
    # end G: bottom-left of G
    xG = x_R
    yG = G_y           # bottom edge of G

    scheme_dir = Path("figures/pdffiles").resolve()

    with tempfile.TemporaryDirectory() as texdir:
        tmp_fig = Path(texdir) / "main.pdf"
        fig.savefig(tmp_fig, dpi=300)
        plt.close(fig)

        tex = rf"""\documentclass{{article}}
\usepackage[paperwidth={FIG_W:.4f}in,paperheight={FIG_H:.4f}in,margin=0pt]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{tikz}}
\usetikzlibrary{{calc}}
\pagestyle{{empty}}
\begin{{document}}%
\begin{{tikzpicture}}[remember picture,overlay]
  \node[inner sep=0,anchor=south west] at (current page.south west)
    {{\includegraphics{{{tmp_fig}}}}};
  \node[inner sep=0,anchor=south west]
    at ($(current page.south west)+({x_ins_A:.4f}in,{y_ins_A:.4f}in)$)
    {{\includegraphics[width={INSET_W:.4f}in]{{{scheme_dir}/fold_scheme.pdf}}}};
  \node[inner sep=0,anchor=south west]
    at ($(current page.south west)+({x_ins_B:.4f}in,{y_ins_B:.4f}in)$)
    {{\includegraphics[width={INSET_W:.4f}in]{{{scheme_dir}/trans_critical_scheme.pdf}}}};
  \draw[black, line width=0.8pt]
    ($(current page.south west)+({xo:.4f}in,{yo:.4f}in)$) --
    ($(current page.south west)+({xC:.4f}in,{yC:.4f}in)$);
  \draw[black, line width=0.8pt]
    ($(current page.south west)+({xo:.4f}in,{yo:.4f}in)$) --
    ($(current page.south west)+({xG:.4f}in,{yG:.4f}in)$);
\end{{tikzpicture}}%
\end{{document}}
"""
        tex_path = Path(texdir) / "combined_v3.tex"
        tex_path.write_text(tex)

        for _ in range(2):
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode",
                 "-output-directory", texdir, str(tex_path)],
                capture_output=True, text=True, cwd=texdir,
            )
        if result.returncode != 0:
            print(result.stdout[-1000:])
            raise SystemExit("pdflatex failed — see output above")

        shutil.copy(Path(texdir) / "combined_v3.pdf", args.output)

    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
