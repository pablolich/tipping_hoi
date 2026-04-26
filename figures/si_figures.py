#!/usr/bin/env python3
"""Supplementary figures for the Nature manuscript on tipping points in HOI systems.

Produces four multi-panel figures covering all four simulation banks
(muB = -0.1, 0.0, 0.1, and the unique_equilibrium negative control):

    SI_boundary_fractions  — prevalence of gradual vs abrupt boundaries   (4×2)
    SI_delta_c             — critical perturbation magnitude               (4×3)
    SI_xmin                — minimum species abundance at boundary         (4×3)
    SI_hysteresis          — hysteresis (reversal_frac) across bnd types   (4×3)

The delta_c / x_min / hysteresis figures each span all three boundary types
(gradual / abrupt / unstable) so regimes can be compared across boundary classes.

Hysteresis data (backtrack_results) is not yet present in the current bank
JSONs, so SI_hysteresis panels show "no data" until the backtrack stage runs.

Run from new_code/:

    python figures/si_figures.py
    python figures/si_figures.py --model-runs model_runs --output-dir figures/si
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse loaders/aggregators/styling from figure_3.py to guarantee identical
# data handling with the main-text figure.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_3 as f3  # noqa: E402


# ── bank registry ──────────────────────────────────────────────────────────────

UNIQUE_EQ_DIR = "2_bank_unique_equilibrium_50_models_n_4-10_128_dirs"

BANKS: List[dict] = [
    {"key": "unique_eq", "dir": UNIQUE_EQ_DIR,          "label": "Unique eq.\n(slice-NDef $B$)"},
    {"key": "muB_-0.1",  "dir": f3.bank_dir_name(-0.1), "label": r"$\mu_B = -0.1$"},
    {"key": "muB_0.0",   "dir": f3.bank_dir_name(0.0),  "label": r"$\mu_B = 0$"},
    {"key": "muB_0.1",   "dir": f3.bank_dir_name(0.1),  "label": r"$\mu_B = 0.1$"},
]

# Boundary types used as row entries in the transposed (3×4) figures.
BOUNDARY_TYPES: List[Tuple[str, str]] = [
    ("fold",     "Abrupt"),
    ("negative", "Gradual"),
    ("unstable", "Unstable"),
]

# Full integer range of n across the banks. unique_equilibrium only populates
# n ∈ {4..10}; the muB banks populate n ∈ {4..20}. Panels auto-skip absent n.
N_VALUES_ALL = list(range(4, 21))


# ── style ──────────────────────────────────────────────────────────────────────

MM = 1.0 / 25.4
DOUBLE_COL_W = 183.0 * MM   # 7.20"
SINGLE_COL_W =  89.0 * MM   # 3.50"

NATURE_RCPARAMS = {
    "font.family":       ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset":  "dejavusans",
    "font.size":         7,
    "axes.labelsize":    7,
    "axes.titlesize":    7,
    "xtick.labelsize":   6,
    "ytick.labelsize":   6,
    "legend.fontsize":   6,
    "axes.linewidth":    0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size":  2.5,
    "ytick.major.size":  2.5,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "legend.frameon":    False,
}


# ── data loading ───────────────────────────────────────────────────────────────

def load_bank(bank_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Walk one bank directory, return (scan_df, bt_df).

    Uses figure_3.load_scan_rows / load_backtrack_rows so the row schema is
    identical to the one used for main-text Figure 3.
    """
    scan_rows: list = []
    bt_rows:   list = []
    if not bank_dir.is_dir():
        print(f"  ! missing bank directory: {bank_dir}")
        return pd.DataFrame(), pd.DataFrame()

    for path in sorted(bank_dir.rglob("*.json")):
        try:
            scan_rows.extend(f3.load_scan_rows(path))
            bt_rows.extend(f3.load_backtrack_rows(path))
        except Exception as exc:
            print(f"    skipping {path.name}: {exc}")
    scan_df = pd.DataFrame(scan_rows) if scan_rows else pd.DataFrame()
    bt_df   = pd.DataFrame(bt_rows)   if bt_rows   else pd.DataFrame()
    return scan_df, bt_df


ALL_BOUNDARY_FLAGS = ["negative", "complex", "unstable", "success"]


def boundary_fractions_from_scan(scan_df: pd.DataFrame) -> pd.DataFrame:
    """Produce prevalence fractions (alpha_value, boundary_type, fraction, n, model_id).

    Applies FLAG_REMAP so `fold → complex` (abrupt), matching figure_3.py, and
    covers all four boundary classes (gradual, abrupt, unstable, success).
    """
    if scan_df.empty:
        return pd.DataFrame()

    frames = []
    for (model_id, n), grp in scan_df.groupby(["model_id", "n"], sort=False):
        rows = [
            {"alpha_value": a, "boundary_type": f3.FLAG_REMAP.get(flag, flag)}
            for a, flag in zip(grp["alpha"], grp["flag"])
        ]
        frac = f3._boundary_fractions_from_rows(rows, ALL_BOUNDARY_FLAGS)
        if frac.empty:
            continue
        frac["model_id"] = model_id
        frac["n"]        = int(n)
        frames.append(frac)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── integrity reporting ────────────────────────────────────────────────────────

def print_integrity_table(scan_df: pd.DataFrame, bank_label: str) -> None:
    print(f"\n── Integrity: {bank_label} ──")
    if scan_df.empty:
        print("  (empty — no JSON files loaded)")
        return

    n_models = scan_df["model_id"].nunique()
    n_dirs   = len(scan_df)
    print(f"  models: {n_models}    total directions: {n_dirs}")
    print(f"  n range: {sorted(scan_df['n'].unique().tolist())}")
    print(f"  alpha range: {sorted(scan_df['alpha'].unique().tolist())}")

    # Per-(n, alpha) flag counts
    pivot = (
        scan_df.assign(flag=scan_df["flag"].astype(str))
        .pivot_table(index=["n", "alpha"], columns="flag",
                     values="ray_id", aggfunc="count", fill_value=0)
        .reset_index()
    )
    pivot["n_replicates"] = (
        scan_df.groupby(["n", "alpha"])["model_id"].nunique().values
    )
    pivot["n_directions"] = pivot.drop(columns=["n", "alpha"]).sum(axis=1) - pivot["n_replicates"]
    # Reorder columns for readability
    flag_cols = [c for c in ["negative", "fold", "unstable", "success"] if c in pivot.columns]
    cols = ["n", "alpha", "n_replicates", "n_directions"] + flag_cols
    cols = [c for c in cols if c in pivot.columns]
    with pd.option_context("display.max_rows", 40,
                           "display.width", 120,
                           "display.float_format", "{:.2f}".format):
        print(pivot[cols].to_string(index=False))


def warn_if_unique_eq_has_abrupt(frac_df: pd.DataFrame, label: str,
                                 threshold: float = 0.05) -> None:
    if frac_df.empty:
        return
    abrupt = frac_df[frac_df["boundary_type"] == "complex"]
    if abrupt.empty:
        return
    agg = abrupt.groupby(["n", "alpha_value"], as_index=False)["fraction"].median()
    violations = agg[agg["fraction"] > threshold]
    if violations.empty:
        print(f"  [ok] {label}: no (n, alpha) cell exceeds {threshold:.0%} abrupt fraction")
        return
    print(f"\n  !! WARNING: {label} has abrupt-boundary fractions above "
          f"{threshold:.0%} in {len(violations)} (n, alpha) cells:")
    with pd.option_context("display.max_rows", 20):
        print(violations.to_string(index=False))


# ── panel plotters ─────────────────────────────────────────────────────────────

def _despine(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.02, 1.08, letter,
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8, fontweight="bold")


def _no_data(ax: plt.Axes) -> None:
    ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
            ha="center", va="center", color="gray", fontsize=6)


def plot_fraction_panel(ax: plt.Axes, frac_df: pd.DataFrame, flag: str,
                        n_values: List[int], color_map: dict) -> None:
    """Median prevalence line + IQR shading, colored by n."""
    _despine(ax)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])

    if frac_df.empty:
        _no_data(ax)
        return
    sub = frac_df[frac_df["boundary_type"] == flag]
    if sub.empty:
        _no_data(ax)
        return

    for n_val in n_values:
        sub_n = sub[sub["n"] == n_val]
        if sub_n.empty:
            continue
        grp = sub_n.groupby("alpha_value")["fraction"]
        med = grp.median().sort_index()
        q25 = grp.quantile(0.25).reindex(med.index)
        q75 = grp.quantile(0.75).reindex(med.index)
        color = color_map[n_val]
        ax.fill_between(med.index, q25.values, q75.values,
                        color=color, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(med.index, med.values, color=color,
                linewidth=1.0, zorder=3)
        ax.scatter(med.index, med.values, color=color,
                   s=7, edgecolors="black", linewidths=0.3, zorder=4)


def plot_metric_panel(ax: plt.Axes, sys_df: pd.DataFrame, summary_df: pd.DataFrame,
                      n_values: List[int], color_map: dict,
                      scale: float = 1.0,
                      logy: bool = False) -> None:
    """Median line + IQR error bars, colored by n. Used for delta_c / x_min / hysteresis."""
    _despine(ax)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks([0.0, 0.5, 1.0])

    if sys_df.empty or summary_df.empty:
        _no_data(ax)
        return

    # IQR error bars
    f3._errorbar_n(ax, sys_df, "value", n_values, color_map, scale=scale)

    # Median line + dots
    for n_val in n_values:
        sub = summary_df[summary_df["n"] == n_val].sort_values("alpha")
        if sub.empty:
            continue
        color = color_map[n_val]
        ax.plot(sub["alpha"], sub["median"] * scale,
                color=color, linewidth=0.9, zorder=3, alpha=0.9)
        ax.scatter(sub["alpha"], sub["median"] * scale,
                   color=color, s=8, edgecolors="black",
                   linewidths=0.3, zorder=4)

    if logy:
        ax.set_yscale("log")


# ── figure builders ────────────────────────────────────────────────────────────

def _add_row_labels(fig: plt.Figure, axes: np.ndarray, labels: List[str]) -> None:
    for row_idx, label in enumerate(labels):
        ax = axes[row_idx, 0]
        pos = ax.get_position()
        y = pos.y0 + 0.5 * pos.height
        fig.text(0.008, y, label, ha="left", va="center",
                 fontsize=7, fontweight="bold", rotation=90)


def _add_shared_legend(fig: plt.Figure, n_values: List[int], color_map: dict) -> None:
    handles = [
        mpl.lines.Line2D([0], [0], color=color_map[n], marker="o",
                         markersize=3.5, markeredgecolor="black",
                         markeredgewidth=0.3, linewidth=1.0,
                         label=f"{n}")
        for n in n_values
    ]
    fig.legend(handles=handles, title=r"$n$",
               loc="lower center", ncol=len(n_values),
               bbox_to_anchor=(0.5, -0.005),
               handlelength=1.2, columnspacing=0.9,
               handletextpad=0.3, borderpad=0.3)


def _abc_letter(row: int, col: int, ncols: int) -> str:
    return chr(ord("A") + row * ncols + col)


def build_fig_boundary_fractions(banks_data: Dict[str, dict],
                                 n_values: List[int], color_map: dict,
                                 out_stem: Path) -> None:
    row_labels = ["Abrupt",  "Gradual",  "Unstable", "Success"]
    row_flags  = ["complex", "negative", "unstable", "success"]
    nrows = len(row_flags)
    ncols = len(BANKS)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(DOUBLE_COL_W, 7.6),
        sharex=True, sharey=True,
        gridspec_kw=dict(left=0.09, right=0.99, bottom=0.09, top=0.94,
                         wspace=0.10, hspace=0.28),
    )

    for row_idx, flag in enumerate(row_flags):
        for col_idx, bank in enumerate(BANKS):
            ax = axes[row_idx, col_idx]
            data = banks_data[bank["key"]]
            frac_df = data["frac_df"]
            plot_fraction_panel(ax, frac_df, flag, n_values, color_map)
            _panel_letter(ax, _abc_letter(row_idx, col_idx, ncols))
            if row_idx == 0:
                ax.set_title(bank["label"], fontsize=7.5, pad=3)
            if row_idx == nrows - 1:
                ax.set_xlabel(r"HOI strength $\alpha$")
            if col_idx == 0:
                ax.set_ylabel("Fraction of directions")

    _add_row_labels(fig, axes, row_labels)
    _add_shared_legend(fig, n_values, color_map)

    _save(fig, out_stem)


def _build_transposed_metric_fig(
    banks_data: Dict[str, dict],
    n_values: List[int],
    color_map: dict,
    out_stem: Path,
    *,
    ylabel: str,
    source: str,          # "scan" or "backtrack"
    metric: str = "delta_c",
    scale: float = 1.0,
    fig_height: float = 6.0,
    isolate_first_col_y: bool = False,
    exclude_points: Dict[Tuple[str, str], List[Tuple[int, float]]] = None,
    boundary_types: List[Tuple[str, str]] = None,
) -> None:
    """Shared rows×4 (boundary × bank) builder for delta_c, x_min, hysteresis.

    source="scan"     → f3.aggregate_scan_panel(scan_df, flag, metric)
    source="backtrack"→ f3.aggregate_reversal_frac(bt_df, flag)
    """
    bt = boundary_types if boundary_types is not None else BOUNDARY_TYPES
    nrows = len(bt)
    ncols = len(BANKS)

    sharey_arg = False if isolate_first_col_y else "row"
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(DOUBLE_COL_W, fig_height),
        sharex=True, sharey=sharey_arg,
        gridspec_kw=dict(left=0.09, right=0.99, bottom=0.12, top=0.92,
                         wspace=0.15, hspace=0.30),
    )

    if isolate_first_col_y:
        # Share y across the muB columns (cols 1..ncols-1) within each row,
        # leaving the unique_eq column (col 0) on its own scale.
        for row_idx in range(nrows):
            anchor = axes[row_idx, 1]
            for col_idx in range(2, ncols):
                axes[row_idx, col_idx].sharey(anchor)
                plt.setp(axes[row_idx, col_idx].get_yticklabels(), visible=False)

    # Pre-compute per-bank min-count filters once.
    bank_filters = {
        bank["key"]: (
            f3.build_min_count_filters(banks_data[bank["key"]]["scan_df"], threshold=0.02)
            if not banks_data[bank["key"]]["scan_df"].empty else {}
        )
        for bank in BANKS
    }

    for row_idx, (flag, flag_label) in enumerate(bt):
        for col_idx, bank in enumerate(BANKS):
            ax = axes[row_idx, col_idx]
            data = banks_data[bank["key"]]
            scan_df = data["scan_df"]
            bt_df   = data["bt_df"]
            min_filter = bank_filters[bank["key"]]

            # Aggregate.
            sys_df, sum_df = pd.DataFrame(), pd.DataFrame()
            empty_src = False
            if source == "scan":
                if scan_df.empty:
                    empty_src = True
                else:
                    sys_df, sum_df = f3.aggregate_scan_panel(
                        scan_df, boundary=flag, metric=metric,
                        min_count_filter=min_filter,
                    )
            elif source == "backtrack":
                if bt_df.empty:
                    empty_src = True
                else:
                    sys_df, sum_df = f3.aggregate_reversal_frac(
                        bt_df, boundary=flag, min_count_filter=min_filter,
                    )
            elif source == "backtrack_amount":
                if bt_df.empty:
                    empty_src = True
                else:
                    sys_df, sum_df = f3.aggregate_hysteresis_amount(
                        bt_df, boundary=flag, min_count_filter=min_filter,
                    )
            else:
                raise ValueError(f"unknown source: {source}")

            # Drop excluded (n, alpha) points if requested.
            if exclude_points and (flag, bank["key"]) in exclude_points:
                for ex_n, ex_a in exclude_points[(flag, bank["key"])]:
                    if not sys_df.empty:
                        sys_df = sys_df[~((sys_df["n"] == ex_n) & (sys_df["alpha"] == ex_a))]
                    if not sum_df.empty:
                        sum_df = sum_df[~((sum_df["n"] == ex_n) & (sum_df["alpha"] == ex_a))]

            if empty_src:
                _despine(ax); _no_data(ax); ax.set_xticks([0.0, 0.5, 1.0])
            else:
                plot_metric_panel(ax, sys_df, sum_df, n_values, color_map, scale=scale)

            _panel_letter(ax, _abc_letter(row_idx, col_idx, ncols))
            if row_idx == 0:
                ax.set_title(bank["label"], fontsize=7.5, pad=3)
            if row_idx == nrows - 1:
                ax.set_xlabel(r"HOI strength $\alpha$")
            if col_idx == 0:
                ax.set_ylabel(ylabel)

    # Row labels on the left margin = boundary types.
    _add_row_labels(fig, axes, [label for _, label in bt])
    _add_shared_legend(fig, n_values, color_map)

    _save(fig, out_stem)


def build_fig_delta_c(banks_data, n_values, color_map, out_stem):
    _build_transposed_metric_fig(
        banks_data, n_values, color_map, out_stem,
        ylabel=r"$\delta_c$",
        source="scan", metric="delta_c",
        isolate_first_col_y=True,
        boundary_types=_BOUNDARY_TYPES_NO_UNSTABLE,
    )


_BOUNDARY_TYPES_NO_UNSTABLE = [bt for bt in BOUNDARY_TYPES if bt[0] != "unstable"]


def build_fig_xmin(banks_data, n_values, color_map, out_stem):
    _build_transposed_metric_fig(
        banks_data, n_values, color_map, out_stem,
        ylabel=r"$x_{\min}$ at boundary",
        source="scan", metric="min",
        boundary_types=_BOUNDARY_TYPES_NO_UNSTABLE,
    )


def build_fig_hysteresis(banks_data, n_values, color_map, out_stem):
    _build_transposed_metric_fig(
        banks_data, n_values, color_map, out_stem,
        ylabel=r"Hysteresis amount ($\delta_{\mathrm{post}} - \delta_{\mathrm{return}}$)",
        source="backtrack_amount",
        exclude_points={("fold", "muB_0.0"): [(5, 0.1)]},
        boundary_types=_BOUNDARY_TYPES_NO_UNSTABLE,
    )


# ── output ─────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {pdf_path}")
    print(f"  wrote {png_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Nature SI figures: four-bank tipping diagnostics")
    parser.add_argument("--model-runs", type=Path, default=Path("model_runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures/pdffiles"))
    args = parser.parse_args()

    plt.rcParams.update(NATURE_RCPARAMS)

    # Load all four banks (per-bank separation).
    banks_data: Dict[str, dict] = {}
    print("\n=== Loading banks ===")
    for bank in BANKS:
        bank_dir = args.model_runs / bank["dir"]
        print(f"Loading {bank['key']}  →  {bank_dir}")
        scan_df, bt_df = load_bank(bank_dir)
        frac_df = boundary_fractions_from_scan(scan_df)
        banks_data[bank["key"]] = {
            "scan_df": scan_df,
            "bt_df":   bt_df,
            "frac_df": frac_df,
        }

    # Integrity tables + negative-control check.
    print("\n=== Data integrity ===")
    for bank in BANKS:
        data = banks_data[bank["key"]]
        print_integrity_table(data["scan_df"], bank["key"])
        if bank["key"] == "unique_eq":
            warn_if_unique_eq_has_abrupt(data["frac_df"], bank["key"])

    # Build shared color map across all banks.
    cmap_base, cmap_faded, _ = f3.build_colormaps(len(N_VALUES_ALL))
    color_map, _ = f3.make_color_dicts(N_VALUES_ALL, cmap_base, cmap_faded)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Building figures ===")
    build_fig_boundary_fractions(
        banks_data, N_VALUES_ALL, color_map,
        args.output_dir / "si_boundary_fractions",
    )
    build_fig_delta_c(
        banks_data, N_VALUES_ALL, color_map,
        args.output_dir / "si_delta_c",
    )
    build_fig_xmin(
        banks_data, N_VALUES_ALL, color_map,
        args.output_dir / "si_xmin",
    )
    build_fig_hysteresis(
        banks_data, N_VALUES_ALL, color_map,
        args.output_dir / "si_hysteresis",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
