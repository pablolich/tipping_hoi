#!/usr/bin/env python3
"""Figure 3: Gradual and Abrupt boundary prevalence as a function of HOI strength
for mu_A=0 and mu_B ∈ {-0.1, 0.0, 0.1} (balanced parameterisation).

Layout:
  Left  3×2 : boundary prevalence panels (rows=mu_B, col0=Gradual, col1=Abrupt)
  Top-right  : A_ij and B_ijk distribution panels (n=4 representative)
  Bot-right  : Robustness / Abruptness / Hysteresis aux panels (pooled banks)
  Insets     : fold_scheme on row-0 Abrupt; trans_critical_scheme on row-0 Gradual
  Callouts   : TikZ lines from bottom-right Abrupt panel to C and G

Usage (run from new_code/):
    python figures/tipping_prevalence_panels.py
    python figures/tipping_prevalence_panels.py --model-runs data/example_runs --output pdffiles/main/figure_3.pdf
"""

import argparse
import json
import pickle
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns


# ── constants ──────────────────────────────────────────────────────────────────

FLAG_REMAP            = {"fold": "complex"}
DEFAULT_BOUNDARY_FLAGS = ["negative", "complex"]

MU_B_ROWS   = [-0.1, 0.0, 0.1]   # top → bottom
REP_N       = 4

BANK_PREFIX = "2_bank_standard_50_models_n_4-20_128_dirs"

DEFAULT_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "figure_cache" / "figure_3_aggregate.pkl"
CACHE_VERSION      = 1

DARK_POINT_SIZE = 50 * 0.65
DARK_EDGE_WIDTH = 0.6

# colours for the three mu_B values in the B distribution panel
MU_B_COLORS = {-0.1: "#2166ac", 0.0: "#4dac26", 0.1: "#d6604d"}
MU_B_LABELS = {-0.1: r"$\mu_B=-0.1$", 0.0: r"$\mu_B=0$", 0.1: r"$\mu_B=0.1$"}


# ── helpers ────────────────────────────────────────────────────────────────────

def _fmt_mu(v: float) -> str:
    return str(v)


def bank_dir_name(mu_b: float) -> str:
    return f"{BANK_PREFIX}_muB_{_fmt_mu(mu_b)}"


# ── data loading ───────────────────────────────────────────────────────────────

def _boundary_fractions_from_rows(rows: list, boundary_types: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["alpha_value", "boundary_type", "count", "total", "fraction"])
    counts = df.groupby(["alpha_value", "boundary_type"]).size().reset_index(name="count")
    totals = (
        counts.groupby("alpha_value", as_index=False)["count"]
        .sum()
        .rename(columns={"count": "total"})
    )
    counts = counts[counts["boundary_type"].isin(boundary_types)]
    full_index = pd.MultiIndex.from_product(
        [totals["alpha_value"].unique(), boundary_types],
        names=["alpha_value", "boundary_type"],
    ).to_frame(index=False)
    out = full_index.merge(counts, on=["alpha_value", "boundary_type"], how="left")
    out["count"] = out["count"].fillna(0).astype(int)
    out = out.merge(totals, on="alpha_value", how="left")
    out["fraction"] = np.where(out["total"] > 0, out["count"] / out["total"], 0.0)
    return out


def collect_boundary_fractions_per_mu_b(
    model_runs: Path, boundary_types: List[str]
) -> Dict[float, pd.DataFrame]:
    result: Dict[float, pd.DataFrame] = {}
    for mu_b in MU_B_ROWS:
        bdir = model_runs / bank_dir_name(mu_b)
        frames = []
        if not bdir.is_dir():
            print(f"  Missing bank: {bdir.name}")
            result[mu_b] = pd.DataFrame()
            continue
        for path in sorted(bdir.rglob("*.json")):
            try:
                data = json.loads(path.read_text())
            except Exception as exc:
                print(f"    Skipping {path.name}: {exc}")
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
                    bt = FLAG_REMAP.get(raw_flag, raw_flag)
                    rows.append({"alpha_value": alpha, "boundary_type": bt})
            if not rows:
                continue
            frac = _boundary_fractions_from_rows(rows, boundary_types)
            if frac.empty:
                continue
            frac["model_id"] = model_id
            frac["n"]        = n
            frames.append(frac)
        result[mu_b] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        n_models = result[mu_b]["model_id"].nunique() if not result[mu_b].empty else 0
        print(f"  mu_B={mu_b}: {n_models} models")
    return result


def _to_float_or_nan(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if np.isfinite(out) else np.nan


def load_scan_rows(path: Path) -> list:
    data = json.loads(path.read_text())
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
                "model_id":       model_id,
                "n":              n,
                "alpha":          alpha,
                "ray_id":         int(d["ray_id"]),
                "flag":           str(d["flag"]).lower(),
                "delta_c":        float(d["delta_c"]),
                "x_boundary_min": float(min(x_boundary)) if x_boundary else np.nan,
            })
    return rows


def load_backtrack_rows(path: Path) -> list:
    data = json.loads(path.read_text())
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
                "delta_post":    _to_float_or_nan(d.get("delta_post")),
                "delta_return":  _to_float_or_nan(d.get("delta_return")),
            })
    return rows


def collect_data_pooled(
    model_runs: Path, mu_b_values: List[float]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scan_rows, bt_rows = [], []
    for mu_b in mu_b_values:
        bdir = model_runs / bank_dir_name(mu_b)
        if not bdir.is_dir():
            continue
        for path in sorted(bdir.rglob("*.json")):
            try:
                scan_rows.extend(load_scan_rows(path))
                bt_rows.extend(load_backtrack_rows(path))
            except Exception as exc:
                print(f"  Warning – skipping {path.name}: {exc}")
    scan_df = pd.DataFrame(scan_rows) if scan_rows else pd.DataFrame()
    bt_df   = pd.DataFrame(bt_rows)   if bt_rows   else pd.DataFrame()
    return scan_df, bt_df


def load_interaction_distributions(
    model_runs: Path, mu_b_values: List[float], rep_n: int
) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
    """Return {mu_b: (A_offdiag, B_offdiag)} for models with n == rep_n."""
    result: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
    for mu_b in mu_b_values:
        bdir = model_runs / bank_dir_name(mu_b)
        a_vals, b_vals = [], []
        if not bdir.is_dir():
            result[mu_b] = (np.array([]), np.array([]))
            continue
        for path in sorted(bdir.rglob("*.json")):
            try:
                data = json.loads(path.read_text())
            except Exception:
                continue
            if int(data.get("n", 0)) != rep_n:
                continue
            A = np.array(data["A"])   # (n, n)
            B = np.array(data["B"])   # (n, n, n)  B[j,k,i]
            n = rep_n
            for i in range(n):
                for j in range(n):
                    if i != j:
                        a_vals.append(A[i, j])
            for j in range(n):
                for k in range(n):
                    for i in range(n):
                        if not (j == k == i):
                            b_vals.append(B[j, k, i])
        result[mu_b] = (np.array(a_vals), np.array(b_vals))
        print(f"  Interactions mu_B={mu_b}: {len(a_vals)} A vals, {len(b_vals)} B vals")
    return result


# ── aggregation ────────────────────────────────────────────────────────────────

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


def build_min_count_filters(scan_df: pd.DataFrame, threshold: float = 0.02) -> dict:
    if scan_df.empty:
        return {}
    totals     = scan_df.groupby(["n", "alpha"]).size().reset_index(name="total")
    flag_counts = scan_df.groupby(["n", "alpha", "flag"]).size().reset_index(name="count")
    merged = flag_counts.merge(totals, on=["n", "alpha"])
    merged = merged[merged["count"] >= threshold * merged["total"]]
    result = {}
    for flag, group in merged.groupby("flag"):
        result[flag] = set(zip(group["n"].astype(int), group["alpha"]))
    return result


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


def aggregate_hysteresis_amount(
    bt_df: pd.DataFrame,
    boundary: str = "fold",
    min_count_filter: dict = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate the absolute hysteresis gap: delta_post - delta_return."""
    if bt_df.empty or "boundary_flag" not in bt_df.columns:
        return pd.DataFrame(), pd.DataFrame()
    fold = bt_df[
        (bt_df["boundary_flag"] == boundary)
        & (bt_df["ode_ran"] | (bt_df["class_label"] == "success_to_zero"))
        & bt_df["delta_post"].notna()
        & bt_df["delta_return"].notna()
    ].copy()
    if fold.empty:
        return pd.DataFrame(), pd.DataFrame()
    if min_count_filter is not None and boundary in min_count_filter:
        passing_df = pd.DataFrame(list(min_count_filter[boundary]), columns=["n", "alpha"])
        fold = fold.merge(passing_df, on=["n", "alpha"])
    if fold.empty:
        return pd.DataFrame(), pd.DataFrame()
    fold["hyst_amount"] = fold["delta_post"] - fold["delta_return"]
    agg = (
        fold.groupby(["model_id", "n", "alpha"], as_index=False)["hyst_amount"]
        .mean()
        .rename(columns={"hyst_amount": "value"})
    )
    summary_df = (
        agg.groupby(["n", "alpha"], as_index=False)["value"]
        .median()
        .rename(columns={"value": "median"})
    )
    return agg, summary_df


# ── scatter / errorbar helpers ─────────────────────────────────────────────────

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
    for n_val in n_values:
        sub = df[df["n"] == n_val] if not df.empty else pd.DataFrame()
        if sub.empty:
            continue
        grp  = sub.groupby("alpha")[y_col]
        med  = grp.median().sort_index()  * scale
        q25  = grp.quantile(0.25).reindex(med.index) * scale
        q75  = grp.quantile(0.75).reindex(med.index) * scale
        ax.fill_between(med.index, q25.values, q75.values,
                        color=color_map[n_val], alpha=0.18, linewidth=0, zorder=2)


# ── panel plotters ─────────────────────────────────────────────────────────────

def _plot_prevalence(
    ax, frac_df: pd.DataFrame, flag: str, n_values: List[int],
    color_map: dict, show_xticks: bool, show_yticks: bool,
    label_size: float,
):
    margin = 0.03
    ax.set_xlim(-margin - 0.01, 1.0 + margin)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_ylim(-margin, 1.0 + margin)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["0", "0.5", "1"] if show_xticks else [])
    ax.set_yticklabels(["0", "0.5", "1"] if show_yticks else [])

    if frac_df.empty:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", color="gray", fontsize=label_size * 0.7)
        return

    sub = frac_df[frac_df["boundary_type"] == flag].copy()
    if sub.empty:
        return

    for n_val in n_values:
        sub_n = sub[sub["n"] == n_val]
        if sub_n.empty:
            continue
        grp  = sub_n.groupby("alpha_value")["fraction"]
        med  = grp.median().sort_index()
        q25  = grp.quantile(0.25).reindex(med.index)
        q75  = grp.quantile(0.75).reindex(med.index)
        ax.fill_between(med.index, q25.values, q75.values,
                        color=color_map[n_val], alpha=0.18, linewidth=0, zorder=2)

    med_df = (
        sub.groupby(["n", "alpha_value"], as_index=False)["fraction"]
        .median()
        .reset_index(drop=True)
    )
    for n_val in n_values:
        sub_n = med_df[med_df["n"] == n_val].sort_values("alpha_value")
        if sub_n.empty:
            continue
        ax.plot(sub_n["alpha_value"], sub_n["fraction"],
                color=color_map[n_val], linewidth=0.9, zorder=2, alpha=0.85)
        ax.scatter(sub_n["alpha_value"], sub_n["fraction"],
                   color=color_map[n_val],
                   s=DARK_POINT_SIZE * 0.45, edgecolors="black",
                   linewidths=DARK_EDGE_WIDTH * 0.4, zorder=3)


def _plot_aux(ax, sys_df, sum_df, y_col, panel_label, n_values, color_map,
              label_size, scale=1.0):
    _errorbar_n(ax, sys_df, y_col, n_values, color_map, scale)
    _scatter_median(ax, sum_df, "median", n_values, color_map, scale)
    ax.text(0.04, 0.96, panel_label,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=label_size, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5))
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xlim(-0.05, 1.05)


def _plot_distribution(ax, vals_list, colors, labels, inner_label,
                       label_size, xlim=(-1.0, 1.0), show_legend=False):
    """KDE plot for a list of (values, color, label) triples."""
    x_grid = np.linspace(xlim[0], xlim[1], 400)
    for vals, color, label in zip(vals_list, colors, labels):
        if vals is None or len(vals) < 2:
            continue
        kde = scipy.stats.gaussian_kde(vals, bw_method="scott")
        y   = kde(x_grid)
        ax.plot(x_grid, y, color=color, linewidth=1.2, label=label, zorder=3)
        ax.fill_between(x_grid, y, alpha=0.15, color=color, zorder=2)
    ax.set_xlim(xlim)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # inner label (A_ij or B_ijk)
    ax.text(0.97, 0.92, inner_label,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=label_size * 1.20, style="italic")
    if show_legend:
        ax.legend(title=r"$\mu_B$",
                  fontsize=label_size * 0.80,
                  title_fontsize=label_size * 0.80 * 1.20,
                  loc="upper left",
                  frameon=False,
                  borderpad=0.4, handlelength=1.2,
                  handletextpad=0.4)


# ── aggregated dataset (load + aggregate, then cache) ─────────────────────────

def build_aggregate(model_runs: Path) -> dict:
    """Run all JSON loading + aggregation. Output is everything plotting needs."""
    print("Loading boundary fraction data per mu_B ...")
    frac_per_mu = collect_boundary_fractions_per_mu_b(
        model_runs, DEFAULT_BOUNDARY_FLAGS
    )

    print("Loading scan/backtrack data (mu_B=0.1) ...")
    scan_df, bt_df = collect_data_pooled(model_runs, [0.1])

    print("Loading scan data (mu_B=0.0) for H panel ...")
    scan_df_0, _ = collect_data_pooled(model_runs, [0.0])

    print("Loading interaction distributions (n=4) ...")
    inter_data = load_interaction_distributions(model_runs, MU_B_ROWS, REP_N)

    n_set = set()
    for df in frac_per_mu.values():
        if not df.empty:
            n_set.update(int(n) for n in df["n"].unique())
    if not scan_df.empty:
        n_set.update(int(n) for n in scan_df["n"].unique())
    n_values = sorted(n for n in n_set if n > 3 and n % 2 == 0) if n_set else list(range(4, 21, 2))

    if not scan_df.empty:
        scan_df = scan_df[scan_df["n"] > 3]
    if not bt_df.empty:
        bt_df = bt_df[bt_df["n"] > 3]
    if not scan_df_0.empty:
        scan_df_0 = scan_df_0[scan_df_0["n"] > 3]

    min_count_filters   = build_min_count_filters(scan_df,   threshold=0.05)
    min_count_filters_0 = build_min_count_filters(scan_df_0, threshold=0.05)
    sys_fold_dist_0, sum_fold_dist_0 = aggregate_scan_panel(scan_df_0, "fold", "distance", min_count_filters_0)
    sys_fold_dist,   sum_fold_dist   = aggregate_scan_panel(scan_df,   "fold", "distance", min_count_filters)
    sys_fold_min,    sum_fold_min    = aggregate_scan_panel(scan_df,   "fold", "min",      min_count_filters)
    sys_fold_rev,    sum_fold_rev    = aggregate_hysteresis_amount(bt_df, min_count_filter=min_count_filters)

    return {
        "version":         CACHE_VERSION,
        "frac_per_mu":     frac_per_mu,
        "inter_data":      inter_data,
        "n_values":        n_values,
        "sys_fold_dist_0": sys_fold_dist_0, "sum_fold_dist_0": sum_fold_dist_0,
        "sys_fold_dist":   sys_fold_dist,   "sum_fold_dist":   sum_fold_dist,
        "sys_fold_min":    sys_fold_min,    "sum_fold_min":    sum_fold_min,
        "sys_fold_rev":    sys_fold_rev,    "sum_fold_rev":    sum_fold_rev,
    }


def load_or_build_aggregate(cache_path: Path, model_runs: Path, rebuild: bool) -> dict:
    if cache_path.exists() and not rebuild:
        print(f"Loading cached aggregate from {cache_path}")
        with cache_path.open("rb") as f:
            agg = pickle.load(f)
        if agg.get("version") == CACHE_VERSION:
            return agg
        print(f"  Cache version mismatch (got {agg.get('version')}, expected {CACHE_VERSION}) — rebuilding")
    agg = build_aggregate(model_runs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(agg, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved cache to {cache_path}")
    return agg


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Figure 3: mu_B prevalence + distributions")
    parser.add_argument("--model-runs", type=Path, default=Path("data/example_runs"))
    parser.add_argument("--output", type=Path,
                        default=Path("pdffiles/main/figure_3.pdf"))
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH,
                        help="Pickle file holding the aggregated dataset. "
                             "Loaded if present; otherwise built from --model-runs and saved.")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Force rebuild of the aggregate cache from JSONs.")
    args = parser.parse_args()

    # ── typography (matches hysteresis_two_routes.py) ──────────────────────
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

    # ── load aggregated dataset (cached) ───────────────────────────────────
    agg = load_or_build_aggregate(args.cache, args.model_runs, args.rebuild_cache)
    frac_per_mu     = agg["frac_per_mu"]
    inter_data      = agg["inter_data"]
    n_values        = agg["n_values"]
    sys_fold_dist_0 = agg["sys_fold_dist_0"]; sum_fold_dist_0 = agg["sum_fold_dist_0"]
    sys_fold_dist   = agg["sys_fold_dist"];   sum_fold_dist   = agg["sum_fold_dist"]
    sys_fold_min    = agg["sys_fold_min"];    sum_fold_min    = agg["sum_fold_min"]
    sys_fold_rev    = agg["sys_fold_rev"];    sum_fold_rev    = agg["sum_fold_rev"]
    print(f"n values: {n_values}")

    cmap, cmap_faded, norm = build_colormaps(len(n_values))
    color_map, _           = make_color_dicts(n_values, cmap, cmap_faded)

    def _q75_max(df, y_col, scale=1.0, pad=1.0):
        if df.empty:
            return None
        return float(df.groupby(["n", "alpha"])[y_col].quantile(0.75).max()) * scale * pad
    def _q25_min(df, y_col, scale=1.0):
        if df.empty:
            return None
        return float(df.groupby(["n", "alpha"])[y_col].quantile(0.25).min()) * scale

    dist_ymax_raw = _q75_max(sys_fold_dist, "value", pad=1.05)
    dist_ymin_raw = _q25_min(sys_fold_dist, "value")
    shared_dist_ymin = max(dist_ymin_raw, 1e-6) if dist_ymin_raw is not None else 1e-6
    shared_dist_ymax = dist_ymax_raw if dist_ymax_raw is not None else 10.0

    dist_ymax_raw_0 = _q75_max(sys_fold_dist_0, "value", pad=1.05)
    dist_ymin_raw_0 = _q25_min(sys_fold_dist_0, "value")
    shared_dist_ymin_0 = max(dist_ymin_raw_0, 1e-6) if dist_ymin_raw_0 is not None else 1e-6
    shared_dist_ymax_0 = dist_ymax_raw_0 if dist_ymax_raw_0 is not None else 10.0

    min_ymax_raw = _q75_max(sys_fold_min, "value", pad=1.05)
    min_ymin_raw = _q25_min(sys_fold_min, "value")
    shared_min_ymin = (min(min_ymin_raw, 0.0) - 0.085) if min_ymin_raw is not None else -0.085
    shared_min_ymax = min_ymax_raw if min_ymax_raw is not None else 1.0

    fold_rev_ymax = (_q75_max(sys_fold_rev, "value") or 1.0) * 1.05

    # ── layout constants (inches) ──────────────────────────────────────────
    S        = 2.2
    gap_row  = 0.25
    gap_col  = 0.25
    gap_CE   = 0.10
    GAP_LR   = 0.50
    YLABEL_L = 0.85
    YLABEL_R = 0.85
    MARGIN_L = 0.02
    MARGIN_R = 0.03
    MARGIN_B = 0.55
    MARGIN_T = 0.35
    INSET_W  = 0.55 * 1.25   # 0.6875"
    INSET_PAD = 0.09

    s = (2 * S + gap_row - 3 * gap_CE) / 4   # height of each aux panel (4 panels)
    s_aux = s + 0.25                          # width of aux panels (wider, shorter ticks free space)

    FIG_W = MARGIN_L + YLABEL_L + 2 * S + gap_col + GAP_LR + s + YLABEL_R + MARGIN_R
    FIG_H = MARGIN_T + 3 * S + 2 * gap_row + MARGIN_B

    x_L = MARGIN_L + YLABEL_L                          # left edge of 3×2 grid
    x_R = x_L + 2 * S + gap_col + GAP_LR              # left edge of right region

    def norm_rect(x, y, w, h):
        return [x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H]

    # ── 3×2 prevalence panel positions ─────────────────────────────────────
    def prev_xy(row, col):
        x = x_L + col * (S + gap_col)
        y = MARGIN_B + (2 - row) * (S + gap_row)
        return x, y

    # ── aux panel positions ─────────────────────────────────────────────────
    H_new_y = MARGIN_B + 3 * s + 3 * gap_CE  # new top panel (Robustness mu_B=0)
    C_y = MARGIN_B + 2 * s + 2 * gap_CE
    E_y = MARGIN_B + s + gap_CE
    G_y = MARGIN_B

    # ── distribution panel positions ───────────────────────────────────────
    # They span the same vertical region as row 0 of the 3×2 grid
    row0_y_bot  = MARGIN_B + 2 * (S + gap_row)   # bottom of row-0 left panels
    row0_height = S                                # height of one left panel
    dist_W      = s + YLABEL_R                    # full right-region width
    gap_dist    = 0.10
    h_dist      = (row0_height - gap_dist) / 2    # height of each distribution panel

    # bottom of the upper distribution panel:
    Adist_y = row0_y_bot + gap_dist + h_dist      # A_ij panel (top)
    Bdist_y = row0_y_bot                           # B_ijk panel (bottom)

    # ── create figure and axes ─────────────────────────────────────────────
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # 3×2 prevalence panels
    ax_prev = {}
    for row in range(3):
        for col in range(2):
            x, y = prev_xy(row, col)
            ax_prev[(row, col)] = fig.add_axes(norm_rect(x, y, S, S))

    # aux panels (right-ticks); width=s_aux, height=s
    ax_H_new = fig.add_axes(norm_rect(x_R, H_new_y, s_aux, s))
    ax_C = fig.add_axes(norm_rect(x_R, C_y, s_aux, s))
    ax_E = fig.add_axes(norm_rect(x_R, E_y, s_aux, s))
    ax_G = fig.add_axes(norm_rect(x_R, G_y, s_aux, s))
    for ax_r in (ax_H_new, ax_C, ax_E, ax_G):
        ax_r.tick_params(left=False, labelleft=False, right=True, labelright=True)

    # distribution panels
    ax_Adist = fig.add_axes(norm_rect(x_R, Adist_y, dist_W, h_dist))
    ax_Bdist = fig.add_axes(norm_rect(x_R, Bdist_y, dist_W, h_dist))

    # ── plot 3×2 prevalence panels ─────────────────────────────────────────
    flags_by_col = {0: "negative", 1: "complex"}
    panel_labels = {(0,0):"a", (0,1):"b", (1,0):"c", (1,1):"d", (2,0):"e", (2,1):"f"}

    for row, mu_b in enumerate(MU_B_ROWS):
        for col, flag in flags_by_col.items():
            ax = ax_prev[(row, col)]
            df = frac_per_mu.get(mu_b, pd.DataFrame())
            show_x = (row == 2)
            show_y = (col == 0)
            _plot_prevalence(ax, df, flag, n_values, color_map,
                             show_xticks=show_x, show_yticks=show_y,
                             label_size=label_size)
            # panel labels: all outside the panel
            ax.text(0.0, 1.01, panel_labels[(row, col)],
                    transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=label_size, fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5))
            # mu_B label inside abrupt panels (col=1)
            if col == 1:
                mub_x = 0.05
                ax.text(mub_x, 0.95, f"$\\mu_B = {mu_b:g}$",
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=label_size * 0.90 * 1.20, fontweight="bold",
                        color="black",
                        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5))

    # column headers on top of row 0
    for col, title in [(0, "Gradual"), (1, "Abrupt")]:
        ax_prev[(0, col)].set_title(title, fontsize=label_size * 1.15, pad=4)

    # ── plot aux panels ─────────────────────────────────────────────────────
    # Fractional distance of lowest tick from y_min — matches boundary prevalence panels
    _prev_margin = 0.03
    _frac_zero   = _prev_margin / (1.0 + 2 * _prev_margin)   # ≈ 0.02830

    def _ymin_for_tick(tick, ymax):
        """y_min such that `tick` sits at _frac_zero of the total range from the bottom."""
        return (tick - _frac_zero * ymax) / (1.0 - _frac_zero)

    _plot_aux(ax_H_new, sys_fold_dist_0, sum_fold_dist_0, "value", "h", n_values, color_map, label_size)
    _h_lo_raw = _q25_min(sys_fold_dist_0, "value") or 0
    _h_span = 8.5 - _h_lo_raw
    _h_lo = _h_lo_raw - 0.02 * _h_span
    ax_H_new.set_ylim(_h_lo, 8.5)
    _h_tick_lo = np.ceil(_h_lo)
    ax_H_new.set_yticks([_h_tick_lo, 7])
    ax_H_new.set_yticklabels([f"{_h_tick_lo:g}", "7"])
    ax_H_new.tick_params(labelbottom=False)

    _plot_aux(ax_C, sys_fold_dist, sum_fold_dist, "value", "i", n_values, color_map, label_size)
    _i_lo = (_q25_min(sys_fold_dist, "value") or 0) * 0.95
    _i_hi = (float(sum_fold_dist["median"].max()) if not sum_fold_dist.empty else shared_dist_ymax) * 1.05
    ax_C.set_ylim(_i_lo, _i_hi)
    _i_tick_lo = np.ceil(_i_lo)
    _i_tick_hi = np.floor(_i_hi)
    ax_C.set_yticks([_i_tick_lo, _i_tick_hi])
    ax_C.set_yticklabels([f"{_i_tick_lo:g}", f"{_i_tick_hi:g}"])
    ax_C.tick_params(labelbottom=False)

    _plot_aux(ax_E, sys_fold_min, sum_fold_min, "value", "j", n_values, color_map, label_size)
    ax_E.set_ylim(_ymin_for_tick(0, shared_min_ymax), shared_min_ymax)
    _min_tick_top = np.round(shared_min_ymax * 0.9, 1)
    ax_E.set_yticks([0, _min_tick_top])
    ax_E.set_yticklabels(["0", f"{_min_tick_top:g}"])
    ax_E.tick_params(labelbottom=False)

    _plot_aux(ax_G, sys_fold_rev, sum_fold_rev, "value", "k", n_values, color_map, label_size)
    _k_hi = fold_rev_ymax
    ax_G.set_ylim(_ymin_for_tick(0, _k_hi), _k_hi)
    _k_tick_hi = np.floor(_k_hi * 10) / 10  # round down to 1 decimal
    ax_G.set_yticks([0, _k_tick_hi])
    ax_G.set_yticklabels(["0", f"{_k_tick_hi:g}"])
    ax_G.tick_params(labelbottom=True)
    ax_G.set_xticklabels(["0", "0.5", "1"])

    # ── plot distribution panels ────────────────────────────────────────────
    # A_ij: pool all mu_B banks (same mu_A → same A distribution); use grey
    a_all = np.concatenate([inter_data[mu_b][0] for mu_b in MU_B_ROWS
                            if len(inter_data[mu_b][0]) > 0])
    _plot_distribution(
        ax_Adist,
        vals_list  = [a_all] if len(a_all) > 1 else [],
        colors     = ["#555555"],
        labels     = [r"$A_{ij}$"],
        inner_label= r"$A_{ij}$",
        label_size = label_size,
        show_legend= False,
    )
    ax_Adist.set_title("Interactions", fontsize=label_size * 1.15, pad=4)
    ax_Adist.set_xticks([-1, 0, 1])
    ax_Adist.tick_params(labelbottom=False)

    # B_ijk: overlay three mu_B values — labels are just numbers for legend
    b_vals_list  = [inter_data[mu_b][1] for mu_b in MU_B_ROWS]
    b_colors     = [MU_B_COLORS[mu_b]   for mu_b in MU_B_ROWS]
    b_labels     = [f"{mu_b:g}"          for mu_b in MU_B_ROWS]
    _plot_distribution(
        ax_Bdist,
        vals_list  = b_vals_list,
        colors     = b_colors,
        labels     = b_labels,
        inner_label= r"$B_{ijk}$",
        label_size = label_size,
        xlim       = (-1.0, 1.0),
        show_legend= True,
    )
    ax_Bdist.axvline(0, linestyle="--", color="grey", linewidth=0.7, zorder=1)
    ax_Bdist.set_xticks([-1, 0, 1])
    ax_Bdist.set_xticklabels(["-1", "0", "1"])

    # Panel label G for the Interactions region (both distribution panels),
    # placed at same height as A and B, aligned with H/I/J labels below
    _x_G_label = (x_R + 0.04 * s_aux) / FIG_W
    _y_G_label = (row0_y_bot + 1.01 * S) / FIG_H
    fig.text(_x_G_label, _y_G_label, "g",
             ha="left", va="bottom",
             fontsize=label_size, fontweight="bold",
             bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5))

    # ── axis labels ────────────────────────────────────────────────────────

    # Shared "Boundary probability" y-label (centred over all 3 rows)
    YLABEL_EXTRA_PAD = 0.06
    YLABEL_OFFSET    = 0.455 + YLABEL_EXTRA_PAD
    x_ylabel_L = (MARGIN_L + YLABEL_L - YLABEL_OFFSET) / FIG_W
    all_prev_axes = list(ax_prev.values())
    pos_top = ax_prev[(0, 0)].get_position()
    pos_bot = ax_prev[(2, 0)].get_position()
    y_center_left = (pos_bot.y0 + pos_top.y0 + pos_top.height) / 2
    fig.text(x_ylabel_L, y_center_left, "Boundary probability",
             rotation=90, ha="center", va="center", fontsize=label_size * 1.15)

    # Shared "HOI strength (α)" x-label (centred under 3×2 grid)
    x_xl = (x_L + S + gap_col / 2) / FIG_W
    fig.text(x_xl, (MARGIN_B * 0.28) / FIG_H,
             r"HOI strength ($\alpha$)",
             ha="center", va="center", fontsize=label_size * 1.15)

    # Right-column y-axis labels
    x_ylabel_R = (x_R + s_aux + 0.45) / FIG_W
    # Shared Robustness label centred between panels H and I
    pos_H = ax_H_new.get_position()
    pos_I = ax_C.get_position()
    y_shared_rob = (pos_I.y0 + pos_H.y0 + pos_H.height) / 2
    fig.text(x_ylabel_R, y_shared_rob, r"Robustness ($\delta_c$)",
             rotation=-90, ha="center", va="center", fontsize=label_size * 1.15)
    for ax_lbl, lbl in [
        (ax_E, "Abruptness"),
        (ax_G, r"Hysteresis ($\delta_c$)"),
    ]:
        pos = ax_lbl.get_position()
        fig.text(x_ylabel_R, pos.y0 + pos.height / 2, lbl,
                 rotation=-90, ha="center", va="center", fontsize=label_size * 1.15)

    # ── colorbar (inside row-0 gradual panel, shifted up, height = INSET_W) ─
    CBAR_INNER_W  = 0.08 * 1.15
    CBAR_INNER_H  = INSET_W          # match inset size
    GAP_CBAR_INSET = 0.38            # gap wide enough to clear colorbar tick labels
    INSET_SHIFT   = 0.11             # shift insets/colorbar upward from panel bottom
    row0_x, row0_y = prev_xy(0, 0)
    cbar_x_phys = row0_x + INSET_PAD
    cbar_y_phys = row0_y + INSET_PAD + INSET_SHIFT
    cax = fig.add_axes([
        cbar_x_phys / FIG_W,
        cbar_y_phys / FIG_H,
        CBAR_INNER_W / FIG_W,
        CBAR_INNER_H / FIG_H,
    ])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    target_ns = {4, 12, 20}
    tick_positions   = [i for i, n in enumerate(n_values) if int(n) in target_ns]
    tick_labels_cbar = [str(int(n)) for n in n_values if int(n) in target_ns]
    cbar = fig.colorbar(sm, cax=cax, ticks=tick_positions, orientation="vertical")
    cbar.ax.yaxis.set_ticks_position("right")
    cbar.ax.set_yticklabels(tick_labels_cbar)
    cbar.ax.minorticks_off()
    cbar.ax.set_title("n", pad=4, fontsize=label_size, loc="center")

    # ── save main PDF then compose with LaTeX/TikZ ────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Inset positions
    row0_abrupt_x, row0_abrupt_y = prev_xy(0, 1)
    row0_gradual_x, row0_gradual_y = prev_xy(0, 0)

    # fold_scheme → bottom-left of abrupt panel, shifted up same as colorbar
    x_ins_fold = row0_abrupt_x + INSET_PAD
    y_ins_fold = row0_abrupt_y + INSET_PAD + INSET_SHIFT

    # transcritical → right of the colorbar, vertically aligned with it
    x_ins_tc = cbar_x_phys + CBAR_INNER_W + GAP_CBAR_INSET
    y_ins_tc = cbar_y_phys

    # Callout endpoints
    # Origin 1: right-centre of bottom-right abrupt panel (row 2, col 1)
    xo = x_L + 2 * S + gap_col
    yo = MARGIN_B + S / 2

    # End C (now panel I): top-left of ax_C
    xC = x_R
    yC = C_y + s

    # End G (now panel K): bottom-left of ax_G
    xG = x_R
    yG = G_y

    # Origin 2: right-centre of panel D (row 1, col 1)
    xo_D = x_L + 2 * S + gap_col
    yo_D = MARGIN_B + (S + gap_row) + S / 2

    # End H_new: left-centre of new H panel
    xH = x_R
    yH = H_new_y + s / 2

    # Grouping bracket for shared Robustness label (H and I)
    x_brk  = x_R + s_aux + 0.27          # bracket spine x (left of y-label)
    x_brk2 = x_brk - 0.05                # serif end (pointing away from y-label → ] shape)
    y_brk_top = H_new_y + s              # top of panel H
    y_brk_bot = C_y                       # bottom of panel I

    scheme_dir = Path(__file__).resolve().parent.parent / "data" / "figure_inputs" / "scheme_pdfs"

    if not shutil.which("pdflatex"):
        print("  (skipping inset composition -- pdflatex not found)")
        fig.savefig(args.output, dpi=300)
        plt.close(fig)
        print(f"Saved {args.output}")
        return

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
    at ($(current page.south west)+({x_ins_fold:.4f}in,{y_ins_fold:.4f}in)$)
    {{\includegraphics[width={INSET_W:.4f}in]{{{scheme_dir}/fold_scheme.pdf}}}};
  \node[inner sep=0,anchor=south west]
    at ($(current page.south west)+({x_ins_tc:.4f}in,{y_ins_tc:.4f}in)$)
    {{\includegraphics[width={INSET_W:.4f}in]{{{scheme_dir}/trans_critical_scheme.pdf}}}};
  \draw[black, line width=0.8pt]
    ($(current page.south west)+({xo:.4f}in,{yo:.4f}in)$) --
    ($(current page.south west)+({xC:.4f}in,{yC:.4f}in)$);
  \draw[black, line width=0.8pt]
    ($(current page.south west)+({xo:.4f}in,{yo:.4f}in)$) --
    ($(current page.south west)+({xG:.4f}in,{yG:.4f}in)$);
  \draw[black, line width=0.8pt]
    ($(current page.south west)+({xo_D:.4f}in,{yo_D:.4f}in)$) --
    ($(current page.south west)+({xH:.4f}in,{yH:.4f}in)$);
  \draw[black, line width=0.6pt]
    ($(current page.south west)+({x_brk:.4f}in,{y_brk_bot:.4f}in)$) --
    ($(current page.south west)+({x_brk:.4f}in,{y_brk_top:.4f}in)$)
    ($(current page.south west)+({x_brk:.4f}in,{y_brk_bot:.4f}in)$) --
    ($(current page.south west)+({x_brk2:.4f}in,{y_brk_bot:.4f}in)$)
    ($(current page.south west)+({x_brk:.4f}in,{y_brk_top:.4f}in)$) --
    ($(current page.south west)+({x_brk2:.4f}in,{y_brk_top:.4f}in)$);
\end{{tikzpicture}}%
\end{{document}}
"""
        tex_path = Path(texdir) / "figure_3.tex"
        tex_path.write_text(tex)

        for _ in range(2):
            result_proc = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode",
                 "-output-directory", texdir, str(tex_path)],
                capture_output=True, text=True, cwd=texdir,
            )
        if result_proc.returncode != 0:
            print(result_proc.stdout[-1000:])
            raise SystemExit("pdflatex failed — see output above")

        shutil.copy(Path(texdir) / "figure_3.pdf", args.output)

    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
