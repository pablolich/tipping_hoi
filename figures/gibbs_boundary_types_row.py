#!/usr/bin/env python3
"""Boundary-types row — auto-detects Gibbs vs standard banks.

4-panel row: fold | negative | unstable | success

**Gibbs banks** (have alpha_eff):
  x-axis  = alpha_eff (binned, median per bin)
  colour  = regime  (Q1 blue, Q2 orange, Q3 green, Q4 red)
  IQR errorbars + median line (same style as standard path)

**Standard banks** (no alpha_eff):
  x-axis  = scanned alpha grid
  colour  = n
  IQR errorbars + median line (original plot_boundary_types_row.py behaviour)

Usage:
    python figures/gibbs_boundary_types_row.py \
        --bank gibbs_128_dirs_from_gibbs_figures_n20_seed12345
    python figures/gibbs_boundary_types_row.py \
        --bank 2_bank_balanced_50_models_n_3-20_128_dirs_sA_1.0_sB_1.0_muB_0.5
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib as mpl
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── constants ────────────────────────────────────────────────────────────────

FLAG_REMAP = {"fold": "complex"}
# Kept as-is so per-system fractions still normalise by the TOTAL directions
# scanned (including 'success'). The 'success' panel is simply not rendered.
ALL_BOUNDARY_FLAGS = ["negative", "complex", "unstable", "success"]

PANEL_INFO = [
    ("complex",  "A", "Abrupt (fold)"),
    ("negative", "B", "Gradual (negative)"),
    ("unstable", "C", "Unstable"),
]

DARK_POINT_SIZE = 50 * 0.65
DARK_EDGE_WIDTH = 0.6

REGIME_COLORS = {
    "Q1": "#1f77b4",
    "Q2": "#ff7f0e",
    "Q3": "#2ca02c",
    "Q4": "#d62728",
}

REGIME_MARKERS = {
    "Q1": "o",
    "Q2": "s",
    "Q3": "^",
    "Q4": "D",
}
REGIME_MARKER_DEFAULT = "o"

N_ALPHA_BINS = 15


# ── bank type detection ──────────────────────────────────────────────────────

def detect_bank_type(roots: List[Path]) -> str:
    """Return 'gibbs' if the first scan-result JSON has alpha_eff, else 'standard'."""
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.json")):
            try:
                data = json.loads(path.read_text())
            except Exception:
                continue
            if "scan_results" not in data:
                continue
            return "gibbs" if "alpha_eff" in data else "standard"
    return "standard"


# ── data loading — Gibbs ─────────────────────────────────────────────────────

def load_gibbs_data(roots: List[Path], alpha_field: str = "alpha_eff") -> pd.DataFrame:
    """Load per-model boundary fractions from Gibbs scan JSONs.

    alpha_field selects which effective-alpha scalar to use as x-axis
    (e.g. 'alpha_eff', 'alpha_eff_taylor', 'alpha_eff_hull')."""
    rows = []
    skipped = 0
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.json")):
            try:
                data = json.loads(path.read_text())
            except Exception:
                continue
            if "scan_results" not in data:
                continue
            if alpha_field not in data or data[alpha_field] is None:
                skipped += 1
                continue

            alpha_eff = float(data[alpha_field])
            n = int(data["n"])
            param_id = int(data.get("metadata", {}).get("parameterization_id", 0))
            regime = str(data.get("metadata", {}).get("regime", "Q1"))

            flag_counts = {f: 0 for f in ALL_BOUNDARY_FLAGS}
            total = 0
            for sr in data["scan_results"]:
                for d in sr["directions"]:
                    raw = str(d["flag"]).lower()
                    flag = FLAG_REMAP.get(raw, raw)
                    if flag in flag_counts:
                        flag_counts[flag] += 1
                    total += 1
            if total == 0:
                continue

            for flag, cnt in flag_counts.items():
                rows.append({
                    "alpha_eff": alpha_eff,
                    "n": n,
                    "param_id": param_id,
                    "regime": regime,
                    "boundary_type": flag,
                    "fraction": cnt / total,
                })
    if skipped:
        print(f"  skipped {skipped} system(s) missing '{alpha_field}'")
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── data loading — standard ──────────────────────────────────────────────────

def load_standard_data(roots: List[Path]) -> pd.DataFrame:
    """Load boundary fractions from standard (alpha-grid) scan JSONs."""
    rows = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.json")):
            try:
                data = json.loads(path.read_text())
            except Exception:
                continue
            if "scan_results" not in data:
                continue

            n = int(data["n"])
            model_id = str(data.get("seed", path.stem))

            for sr in data["scan_results"]:
                alpha = float(sr["alpha"])
                flag_counts = {f: 0 for f in ALL_BOUNDARY_FLAGS}
                total = 0
                for d in sr["directions"]:
                    raw = str(d["flag"]).lower()
                    flag = FLAG_REMAP.get(raw, raw)
                    if flag in flag_counts:
                        flag_counts[flag] += 1
                    total += 1
                if total == 0:
                    continue
                for flag, cnt in flag_counts.items():
                    rows.append({
                        "alpha_value": alpha,
                        "n": n,
                        "model_id": model_id,
                        "boundary_type": flag,
                        "fraction": cnt / total,
                    })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── colour helpers ───────────────────────────────────────────────────────────

def make_n_colormap(n_values):
    """Colormap for n (used by standard path)."""
    cmap = mpl.cm.get_cmap("YlOrRd")
    k = len(n_values)
    if k == 1:
        return {n_values[0]: cmap(0.65)}
    return {n: cmap(0.3 + 0.7 * i / (k - 1)) for i, n in enumerate(n_values)}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="4-panel boundary row — auto-detects Gibbs vs standard bank."
    )
    parser.add_argument("--bank", type=str, default=None)
    parser.add_argument("--input-root", type=Path, nargs="+", default=None)
    parser.add_argument(
        "--output", type=Path,
        default=Path("pdffiles/si/boundary_types_row_gibbs.pdf"),
    )
    parser.add_argument("--n-bins", type=int, default=N_ALPHA_BINS,
                        help="Number of alpha_eff bins for Gibbs path")
    parser.add_argument("--alpha-field", type=str, nargs="+",
                        default=["alpha_eff"],
                        choices=["alpha_eff", "alpha_eff_taylor", "alpha_eff_hull"],
                        help="Effective-alpha scalar(s) to use as x-axis (Gibbs only). "
                             "Pass multiple values to produce one row per field.")
    args = parser.parse_args()

    model_runs = Path("model_runs")
    if args.bank is not None:
        bank_path = model_runs / args.bank
        if not bank_path.is_dir():
            raise SystemExit(f"Bank directory not found: {bank_path}")
        if args.input_root is None:
            args.input_root = [bank_path]
    else:
        if args.input_root is None:
            args.input_root = [model_runs]

    # ── detect bank type ──────────────────────────────────────────────────
    bank_type = detect_bank_type(args.input_root)
    print(f"Detected bank type: {bank_type}")

    # ── font scale ────────────────────────────────────────────────────────
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

    # ── load data ─────────────────────────────────────────────────────────
    if bank_type == "gibbs":
        _plot_gibbs(args, label_size)
    else:
        _plot_standard(args, label_size)


# ═══════════════════════════════════════════════════════════════════════════════
#  GIBBS PATH — rows = regimes, columns = boundary types, colour = param_id
# ═══════════════════════════════════════════════════════════════════════════════

def _make_param_colormap(param_ids):
    """Build a distinct colour per parameterization_id using tab20 + tab20b."""
    base_colors = (
        list(mpl.cm.tab20(np.linspace(0, 1, 20)))
        + list(mpl.cm.tab20b(np.linspace(0, 1, 20)))
    )
    return {pid: base_colors[i % len(base_colors)] for i, pid in enumerate(param_ids)}


def _plot_gibbs(args, label_size):
    alpha_fields = list(args.alpha_field)
    N_ROWS = len(alpha_fields)
    print(f"Loading Gibbs boundary data (alpha fields: {alpha_fields})...")
    df_rows = []
    for af in alpha_fields:
        df = load_gibbs_data(args.input_root, alpha_field=af)
        if df.empty:
            raise SystemExit(f"No boundary data found for alpha field '{af}'.")
        df_rows.append(df)
    # Use the first row's df for shared metadata (regimes, params, n)
    df_meta = df_rows[0]

    regimes = sorted(df_meta["regime"].unique())
    n_values = sorted(int(v) for v in df_meta["n"].unique())
    all_param_ids = sorted(int(v) for v in df_meta["param_id"].unique())
    print(f"  n values: {n_values}")
    print(f"  regimes: {regimes}")
    print(f"  parameterizations: {len(all_param_ids)}")
    print(f"  models: {len(df_meta) // len(ALL_BOUNDARY_FLAGS)}")

    # ── colour by parameterization ────────────────────────────────────────
    param_cmap = _make_param_colormap(all_param_ids)

    # ── marker by regime ──────────────────────────────────────────────────
    regime_marker = {r: REGIME_MARKERS.get(r, REGIME_MARKER_DEFAULT) for r in regimes}

    # ── layout: N_ROWS × N_COLS ──────────────────────────────────────────
    N_COLS = len(PANEL_INFO)
    S        = 2.2
    GAP_X    = 0.15
    GAP_Y    = 0.85    # vertical gap between rows (fits top row's xticks + xlabel)
    YLABEL_L = 0.75
    MARGIN_L = 0.02
    MARGIN_R = 0.12
    MARGIN_B = 0.55
    MARGIN_T = 0.30

    FIG_W = MARGIN_L + YLABEL_L + N_COLS * S + (N_COLS - 1) * GAP_X + MARGIN_R
    FIG_H = MARGIN_T + N_ROWS * S + (N_ROWS - 1) * GAP_Y + MARGIN_B

    def nr(x, y, w, h):
        return [x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H]

    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # Place axes row by row (ri=0 is the TOP row visually).
    axes_grid = []
    for ri in range(N_ROWS):
        y_phys = MARGIN_B + (N_ROWS - 1 - ri) * (S + GAP_Y)
        row_axes = []
        for ci in range(N_COLS):
            x_phys = MARGIN_L + YLABEL_L + ci * (S + GAP_X)
            row_axes.append(fig.add_axes(nr(x_phys, y_phys, S, S)))
        axes_grid.append(row_axes)
    # Back-compat alias: `axes` refers to the top row.
    axes = axes_grid[0]

    # ── compute per-(regime, param_id) aggregates (shared helper)
    def _aggregate(df, flag):
        """Return list of (regime, pid, x_mean, x_q25, x_q75, y_mean, y_q25, y_q75)."""
        out = []
        sub = df[df["boundary_type"] == flag]
        if sub.empty:
            return out
        for regime in regimes:
            sub_r = sub[sub["regime"] == regime]
            if sub_r.empty:
                continue
            for pid in sorted(sub_r["param_id"].unique()):
                sub_p = sub_r[sub_r["param_id"] == pid]
                out.append((
                    regime, pid,
                    sub_p["alpha_eff"].mean(),
                    sub_p["alpha_eff"].quantile(0.25),
                    sub_p["alpha_eff"].quantile(0.75),
                    sub_p["fraction"].mean(),
                    sub_p["fraction"].quantile(0.25),
                    sub_p["fraction"].quantile(0.75),
                ))
        return out

    def _draw_scatter(target_ax, agg):
        for regime, pid, x_mean, x_q25, x_q75, y_mean, y_q25, y_q75 in agg:
            mk = regime_marker[regime]
            color = param_cmap[pid]
            target_ax.errorbar(
                x_mean, y_mean,
                xerr=[[max(0, x_mean - x_q25)], [max(0, x_q75 - x_mean)]],
                yerr=[[max(0, y_mean - y_q25)], [max(0, y_q75 - y_mean)]],
                fmt=mk, color=color,
                markersize=4,
                markeredgecolor="black", markeredgewidth=0.3,
                elinewidth=0.8, capsize=2.0,
                alpha=0.75, zorder=3,
            )

    def _style_inset(inset):
        inset.tick_params(length=2, pad=1,
                          labelsize=mpl.rcParams["xtick.labelsize"] * 0.75)
        for spine in inset.spines.values():
            spine.set_linewidth(0.6)
        inset.grid(False)

    # ── panel helper ──────────────────────────────────────────────────────
    def _panel(ax, df, flag, plabel, title, show_yt, show_title, inset_kind):
        m = 0.03
        ax.set_xlim(-m - 0.01, 1.0 + m)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_ylim(-m, 1.0 + m)
        ax.set_yticks([0.0, 0.5, 1.0])
        if not show_yt:
            ax.set_yticklabels([])
        if show_title:
            ax.set_title(title, fontsize=label_size, pad=4)
        ax.text(0.0, 1.01, plabel, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=label_size, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5))

        agg = _aggregate(df, flag)
        if not agg:
            return
        _draw_scatter(ax, agg)

        if flag != "complex" or inset_kind is None:
            return

        # Shared y-range: tight around error bars of the "complex" scatter.
        # Since y stats depend only on boundary-flag counts (not on alpha
        # field), both insets across rows use the same y bounds.
        y_lo = min(y_q25 for _, _, _, _, _, _, y_q25, _ in agg)
        y_hi = max(y_q75 for *_, y_q75 in agg)
        y_pad = max(0.03 * (y_hi - y_lo), 1e-3)
        y_lim = (y_lo - y_pad, y_hi + y_pad)

        def _nice_ticks(lo, hi):
            span = hi - lo
            if span <= 0:
                return [lo]
            step = 10 ** np.floor(np.log10(span / 2))
            for k in (1, 2, 5):
                if span / (k * step) <= 3:
                    step = k * step
                    break
            first = np.ceil(lo / step) * step
            return np.arange(first, hi + step / 2, step)

        yt = _nice_ticks(y_lo, y_hi)

        if inset_kind == "taylor_zoom":
            # Top-left inset, fixed x∈[0.4,0.6]; y shared with hull inset.
            inset = ax.inset_axes([0.12, 0.51, 0.44, 0.44])
            _draw_scatter(inset, agg)
            inset.set_xlim(0.45, 0.55)
            inset.set_ylim(*y_lim)
            inset.set_xticks([0.45, 0.5, 0.55])
            inset.set_yticks(yt)
            inset.set_yticklabels([f"{v:g}" for v in yt])
            _style_inset(inset)

        elif inset_kind == "hull_tight":
            # Top-right inset, dynamic tight bounds around error bars.
            x_lo = min(x_q25 for _, _, _, x_q25, _, _, _, _ in agg)
            x_hi = max(x_q75 for _, _, _, _, x_q75, _, _, _ in agg)
            x_pad = max(0.03 * (x_hi - x_lo), 1e-3)
            inset = ax.inset_axes([0.51, 0.51, 0.44, 0.44])
            _draw_scatter(inset, agg)
            inset.set_xlim(x_lo - x_pad, x_hi + x_pad)
            inset.set_ylim(*y_lim)
            xt = _nice_ticks(x_lo, x_hi)
            inset.set_xticks(xt)
            inset.set_xticklabels([f"{v:g}" for v in xt])
            inset.set_yticks(yt)
            inset.set_yticklabels([f"{v:g}" for v in yt])
            _style_inset(inset)

    # ── fill panels for each row ──────────────────────────────────────────
    row_plabel_offset = ["", "E", "I", "M"]  # extend if N_ROWS grows
    default_labels = [info[1] for info in PANEL_INFO]
    for ri, df_row in enumerate(df_rows):
        # Generate panel labels for this row.
        if ri == 0:
            plabels = default_labels
        else:
            plabels = [chr(ord(default_labels[0]) + ri * len(PANEL_INFO) + ci)
                       for ci in range(len(PANEL_INFO))]
        inset_kind = {
            "alpha_eff_taylor": "taylor_zoom",
            "alpha_eff_hull":   "hull_tight",
        }.get(alpha_fields[ri])
        for ci, (flag, _, ttl) in enumerate(PANEL_INFO):
            _panel(
                axes_grid[ri][ci], df_row, flag, plabels[ci], ttl,
                show_yt=(ci == 0),
                show_title=(ri == 0),
                inset_kind=inset_kind,
            )

    # ── regime legend ─────────────────────────────────────────────────────
    regime_handles = [
        mpl.lines.Line2D([], [], marker=regime_marker[r], linestyle="None",
                         color="gray", markersize=5,
                         markeredgecolor="black", markeredgewidth=0.3,
                         label=r)
        for r in regimes
    ]
    lfs = mpl.rcParams["legend.fontsize"] * 0.75
    tfs = label_size * 0.85
    axes[0].legend(handles=regime_handles, title="Regime",
                   loc="upper right",
                   fontsize=lfs, title_fontsize=tfs,
                   framealpha=0.7, edgecolor="none",
                   handletextpad=0.3, labelspacing=0.4)

    # ── y-axis label per row ──────────────────────────────────────────────
    for ri in range(N_ROWS):
        y_center_phys = MARGIN_B + (N_ROWS - 1 - ri) * (S + GAP_Y) + S / 2
        fig.text((MARGIN_L + YLABEL_L * 0.35) / FIG_W, y_center_phys / FIG_H,
                 "Boundary prevalence", rotation=90, ha="center", va="center",
                 fontsize=label_size)

    # ── x-axis label per row ──────────────────────────────────────────────
    xlabel_map = {
        "alpha_eff":        r"Nonlinearity strength ($\alpha_{\mathrm{eff}}$)",
        "alpha_eff_taylor": r"Nonlinearity strength ($\alpha_{\mathrm{eff}}^{\mathrm{Taylor}}$)",
        "alpha_eff_hull":   r"Nonlinearity strength ($\alpha_{\mathrm{eff}}^{\mathrm{hull}}$)",
    }
    x_center = (MARGIN_L + YLABEL_L + (N_COLS * S + (N_COLS - 1) * GAP_X) / 2) / FIG_W
    for ri, af in enumerate(alpha_fields):
        row_bottom = MARGIN_B + (N_ROWS - 1 - ri) * (S + GAP_Y)
        # Bottom-most row: label sits in the bottom margin.
        # Higher rows: label sits in the gap below that row.
        if ri == N_ROWS - 1:
            y_label_phys = MARGIN_B * 0.25
        else:
            y_label_phys = row_bottom - GAP_Y * 0.55
        fig.text(x_center, y_label_phys / FIG_H,
                 xlabel_map.get(af, r"HOI strength"),
                 ha="center", va="center", fontsize=label_size)

    # ── save ──────────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")


# ═══════════════════════════════════════════════════════════════════════════════
#  STANDARD PATH  (original plot_boundary_types_row.py behaviour)
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_standard(args, label_size):
    print("Loading standard boundary data...")
    df = load_standard_data(args.input_root)
    if df.empty:
        raise SystemExit("No boundary data found.")

    n_values = sorted(int(v) for v in df["n"].unique())
    print(f"  n values: {n_values}")

    color_map = make_n_colormap(n_values)

    # ── layout ────────────────────────────────────────────────────────────
    S, GAP, YLABEL_L = 2.2, 0.15, 0.75
    MARGIN_L, MARGIN_R, MARGIN_B, MARGIN_T = 0.02, 0.12, 0.55, 0.30
    N_PANELS = len(PANEL_INFO)
    FIG_W = MARGIN_L + YLABEL_L + N_PANELS * S + (N_PANELS - 1) * GAP + MARGIN_R
    FIG_H = MARGIN_T + S + MARGIN_B

    def nr(x, y, w, h):
        return [x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H]

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    axes = [fig.add_axes(nr(MARGIN_L + YLABEL_L + i * (S + GAP), MARGIN_B, S, S))
            for i in range(N_PANELS)]

    def _panel(ax, flag, plabel, title, show_yt):
        m = 0.03
        ax.set_xlim(-m - 0.01, 1.0 + m)
        ax.set_xticks([0.0, 0.5, 1.0]); ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_ylim(-m, 1.0 + m)
        ax.set_yticks([0.0, 0.5, 1.0])
        if not show_yt:
            ax.set_yticklabels([])
        ax.set_title(title, fontsize=label_size, pad=4)
        ax.text(0.0, 1.01, plabel, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=label_size, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5))

        sub = df[df["boundary_type"] == flag]
        if sub.empty:
            return

        # IQR errorbars
        for nv in n_values:
            sn = sub[sub["n"] == nv]
            if sn.empty:
                continue
            grp = sn.groupby("alpha_value")["fraction"]
            med = grp.median()
            q25 = grp.quantile(0.25)
            q75 = grp.quantile(0.75)
            yerr = np.array([(med - q25).values, (q75 - med).values])
            ax.errorbar(med.index, med.values, yerr=yerr, fmt="none",
                        color=color_map[nv], elinewidth=0.9, capsize=2.0,
                        alpha=0.55, zorder=2)

        # median line + dots
        agg = sub.groupby(["n", "alpha_value"], as_index=False)["fraction"].agg(
            median="median", count="size")
        count_fs = mpl.rcParams["xtick.labelsize"] * 0.75
        for nv in n_values:
            sn = agg[agg["n"] == nv].sort_values("alpha_value")
            if sn.empty:
                continue
            ax.plot(sn["alpha_value"], sn["median"],
                    color=color_map[nv], linewidth=0.9, zorder=2, alpha=0.85)
            ax.scatter(sn["alpha_value"], sn["median"],
                       color=color_map[nv], s=DARK_POINT_SIZE * 0.45,
                       edgecolors="black", linewidths=DARK_EDGE_WIDTH * 0.4, zorder=3)
            for av, yv, cnt in zip(sn["alpha_value"], sn["median"], sn["count"]):
                ax.annotate(str(int(cnt)), xy=(av, yv),
                            xytext=(2.5, 2.5), textcoords="offset points",
                            fontsize=count_fs, color=color_map[nv],
                            ha="left", va="bottom", zorder=4,
                            path_effects=[mpl.patheffects.withStroke(
                                linewidth=1.2, foreground="white")])

    for i, (flag, lab, ttl) in enumerate(PANEL_INFO):
        _panel(axes[i], flag, lab, ttl, show_yt=(i == 0))

    # ── axis labels ───────────────────────────────────────────────────────
    fig.text((MARGIN_L + YLABEL_L * 0.35) / FIG_W, (MARGIN_B + S / 2) / FIG_H,
             "Boundary prevalence", rotation=90, ha="center", va="center",
             fontsize=label_size)
    fig.text((MARGIN_L + YLABEL_L + (N_PANELS * S + (N_PANELS - 1) * GAP) / 2) / FIG_W,
             (MARGIN_B * 0.18) / FIG_H,
             r"HOI strength ($\alpha$)",
             ha="center", va="center", fontsize=label_size)

    # ── colorbar for n ────────────────────────────────────────────────────
    cmap_obj = mpl.colors.ListedColormap([color_map[n] for n in n_values])
    norm_obj = mpl.colors.BoundaryNorm(np.arange(-0.5, len(n_values) + 0.5, 1),
                                       len(n_values))
    sm = mpl.cm.ScalarMappable(cmap=cmap_obj, norm=norm_obj)
    sm.set_array([])
    pos = axes[-1].get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.005, pos.y0 + pos.height * 0.10,
                            0.012, pos.height * 0.80])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks(np.arange(len(n_values)))
    cbar.set_ticklabels([str(n) for n in n_values])
    cbar.ax.tick_params(labelsize=mpl.rcParams["ytick.labelsize"] * 0.85)
    cbar.set_label("$n$", fontsize=label_size, rotation=0, labelpad=6)

    # ── save ──────────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
