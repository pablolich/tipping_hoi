"""Plot boundary-type prevalence across a (mu_A, mu_B) grid of elegant banks.

Produces four 3x3 figures (one per boundary type: negative, fold, unstable,
success). Each panel is one (mu_A, mu_B) cell; within a panel the x-axis is
alpha and the y-axis is the fraction of directions that hit that boundary
type — IQR shaded band per n with median line + markers, matching the
aesthetics of figures/figure_3.py.

Bank directories are auto-discovered from --input-root by matching
    *_bank_elegant_*_muA_<MUA>_muB_<MUB>

Boundary scan output is required: each model JSON must already contain
`scan_results` (run boundary_scan.jl first).

Usage:
    python figures/plot_mu_grid_panels.py \
        --input-root model_runs \
        --output-dir figures/pdffiles/mu_grid
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


FLAG_REMAP = {"fold": "complex"}
ALL_BOUNDARY_FLAGS = ["negative", "complex", "unstable", "success"]
TITLE_MAP = {
    "negative": "Negative boundary",
    "complex": "Fold boundary",
    "unstable": "Unstable boundary",
    "success": "Success boundary",
}

DARK_POINT_SIZE = 50 * 0.65
DARK_EDGE_WIDTH = 0.6

BANK_RE = re.compile(
    r"_bank_elegant_.*_muA_(?P<muA>-?\d+(?:\.\d+)?)_muB_(?P<muB>-?\d+(?:\.\d+)?)$"
)


# ── data loading ──────────────────────────────────────────────────────────────

def discover_banks(root: Path) -> Dict[Tuple[float, float], Path]:
    banks: Dict[Tuple[float, float], Path] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        m = BANK_RE.search(child.name)
        if not m:
            continue
        banks[(float(m.group("muA")), float(m.group("muB")))] = child
    return banks


def _boundary_fractions_from_rows(rows: list, boundary_types: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=["alpha_value", "boundary_type", "count", "total", "fraction"]
        )
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


def collect_bank_fractions(bank_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(bank_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            print(f"  skipping {path.name}: {exc}")
            continue
        if "scan_results" not in data:
            continue
        n = int(data["n"])
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
        frac = _boundary_fractions_from_rows(rows, ALL_BOUNDARY_FLAGS)
        if frac.empty:
            continue
        frac["model_id"] = model_id
        frac["n"] = n
        frames.append(frac)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── colormap helpers (match figure_3.py) ──────────────────────────────────────

def build_colormaps(n_bins: int):
    colors = np.array(sns.color_palette("flare", n_bins))
    cmap_base = mpl.colors.ListedColormap(colors)
    faded = np.concatenate([colors.copy(), 0.25 * np.ones((n_bins, 1))], axis=1)
    cmap_faded = mpl.colors.ListedColormap(faded)
    norm = mpl.colors.BoundaryNorm(np.arange(-0.5, n_bins + 0.5, 1), n_bins)
    return cmap_base, cmap_faded, norm


def make_color_dicts(n_values, cmap_base, cmap_faded):
    colors = cmap_base(np.arange(len(n_values)))
    colors_faded = cmap_faded(np.arange(len(n_values)))
    color_map = {n: colors[i] for i, n in enumerate(n_values)}
    color_map_faded = {n: colors_faded[i] for i, n in enumerate(n_values)}
    return color_map, color_map_faded


# ── prevalence panel (matches figure_3.py::_plot_prevalence) ──────────────────

def _plot_prevalence(
    ax, frac_df: pd.DataFrame, flag: str, n_values: List[int],
    color_map: dict, show_xticks: bool, show_yticks: bool, label_size: float,
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
        grp = sub_n.groupby("alpha_value")["fraction"]
        med = grp.median().sort_index()
        q25 = grp.quantile(0.25).reindex(med.index)
        q75 = grp.quantile(0.75).reindex(med.index)
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


# ── figure assembly ───────────────────────────────────────────────────────────

def plot_grid_for_flag(
    flag: str,
    panel_data: Dict[Tuple[float, float], pd.DataFrame],
    mu_a_values: List[float],
    mu_b_values: List[float],
    n_values: List[int],
    color_map: dict,
    cmap_base, norm,
    label_size: float,
    output_path: Path,
) -> None:
    nrows = len(mu_a_values)
    ncols = len(mu_b_values)

    # ── physical layout (inches) ──────────────────────────────────────────────
    S = 1.6
    gap_row = 0.18
    gap_col = 0.18
    YLABEL_L = 0.70
    XLABEL_B = 0.55
    MARGIN_L = 0.05
    MURIGHT_W = 0.22   # gap reserved for the μ_A row labels on the right
    MARGIN_R = 0.85    # room for colorbar (placed outside the μ_A label column)
    MARGIN_T = 0.55
    MARGIN_B = 0.10

    FIG_W = (MARGIN_L + YLABEL_L + ncols * S + (ncols - 1) * gap_col
             + MURIGHT_W + MARGIN_R)
    FIG_H = MARGIN_T + nrows * S + (nrows - 1) * gap_row + XLABEL_B + MARGIN_B

    x_L = MARGIN_L + YLABEL_L
    y_B = MARGIN_B + XLABEL_B

    def norm_rect(x, y, w, h):
        return [x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H]

    def panel_xy(row, col):
        x = x_L + col * (S + gap_col)
        y = y_B + (nrows - 1 - row) * (S + gap_row)
        return x, y

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.suptitle(TITLE_MAP.get(flag, flag),
                 fontsize=label_size * 1.30, y=1 - (MARGIN_T * 0.30) / FIG_H)

    axes: Dict[Tuple[int, int], plt.Axes] = {}
    for r, mu_a in enumerate(mu_a_values):
        for c, mu_b in enumerate(mu_b_values):
            x, y = panel_xy(r, c)
            ax = fig.add_axes(norm_rect(x, y, S, S))
            axes[(r, c)] = ax

            df = panel_data.get((mu_a, mu_b), pd.DataFrame())
            _plot_prevalence(
                ax, df, flag, n_values, color_map,
                show_xticks=(r == nrows - 1),
                show_yticks=(c == 0),
                label_size=label_size,
            )

            # μ_B header on top row
            if r == 0:
                ax.set_title(rf"$\mu_B = {mu_b:g}$",
                             fontsize=label_size * 1.10, pad=4)

    # ── μ_A row labels just outside the right y-axis (mirror μ_B title pad) ──
    pad_in = 4 / 72.0   # match ax.set_title(..., pad=4)
    for r, mu_a in enumerate(mu_a_values):
        pos_r = axes[(r, ncols - 1)].get_position()
        x_label = pos_r.x1 + pad_in / FIG_W
        y_label = pos_r.y0 + pos_r.height / 2
        fig.text(x_label, y_label, rf"$\mu_A = {mu_a:g}$",
                 rotation=-90, ha="left", va="center",
                 fontsize=label_size * 1.10)

    # ── shared y-label ────────────────────────────────────────────────────────
    pos_top = axes[(0, 0)].get_position()
    pos_bot = axes[(nrows - 1, 0)].get_position()
    y_center = (pos_bot.y0 + pos_top.y0 + pos_top.height) / 2
    x_ylabel = (MARGIN_L + YLABEL_L * 0.30) / FIG_W
    fig.text(x_ylabel, y_center, "Boundary prevalence",
             rotation=90, ha="center", va="center",
             fontsize=label_size * 1.15)

    # ── shared x-label ────────────────────────────────────────────────────────
    pos_left = axes[(nrows - 1, 0)].get_position()
    pos_right = axes[(nrows - 1, ncols - 1)].get_position()
    x_center = (pos_left.x0 + pos_right.x0 + pos_right.width) / 2
    fig.text(x_center, (MARGIN_B + XLABEL_B * 0.30) / FIG_H,
             r"HOI strength ($\alpha$)",
             ha="center", va="center", fontsize=label_size * 1.15)

    # ── colorbar (right side) ─────────────────────────────────────────────────
    pos_tr = axes[(0, ncols - 1)].get_position()
    pos_br = axes[(nrows - 1, ncols - 1)].get_position()
    cbar_y = pos_br.y0
    cbar_h = pos_tr.y1 - pos_br.y0
    cbar_x = pos_tr.x1 + (MURIGHT_W + 0.10) / FIG_W
    cax = fig.add_axes([cbar_x, cbar_y, 0.018, cbar_h])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_base)
    sm.set_array([])
    n_bins = len(n_values)
    if n_bins <= 6:
        tick_positions = list(range(n_bins))
        tick_labels = [str(int(n)) for n in n_values]
    else:
        idx_lo, idx_mid, idx_hi = 0, n_bins // 2, n_bins - 1
        tick_positions = sorted({idx_lo, idx_mid, idx_hi})
        tick_labels = [str(int(n_values[i])) for i in tick_positions]
    cbar = fig.colorbar(sm, cax=cax, ticks=tick_positions, orientation="vertical")
    cbar.ax.set_yticklabels(tick_labels)
    cbar.ax.minorticks_off()
    cbar.ax.set_title("n", pad=4, fontsize=label_size, loc="center")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("model_runs"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("figures/pdffiles/mu_grid"))
    parser.add_argument("--filename-prefix", type=str, default="mu_grid")
    args = parser.parse_args()

    # ── typography (matches figure_3.py / hysteresis_panels.py) ───────────────
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

    banks = discover_banks(args.input_root)
    if not banks:
        raise SystemExit(f"No elegant bank dirs under {args.input_root}")

    mu_a_values = sorted({k[0] for k in banks})
    mu_b_values = sorted({k[1] for k in banks})
    print(f"Found {len(banks)} bank(s): mu_A={mu_a_values}, mu_B={mu_b_values}")

    panel_data: Dict[Tuple[float, float], pd.DataFrame] = {}
    n_set = set()
    for key, bank_dir in banks.items():
        print(f"Loading {bank_dir.name}")
        df = collect_bank_fractions(bank_dir)
        if df.empty:
            print(f"  WARNING: no scan_results in {bank_dir.name}")
        else:
            n_set.update(int(n) for n in df["n"].unique())
        panel_data[key] = df

    if not n_set:
        raise SystemExit("No models with scan_results found — run boundary_scan.jl first.")
    n_values = sorted(n_set)

    cmap_base, cmap_faded, norm = build_colormaps(len(n_values))
    color_map, _ = make_color_dicts(n_values, cmap_base, cmap_faded)

    for flag in ALL_BOUNDARY_FLAGS:
        out = args.output_dir / f"{args.filename_prefix}_{flag}.pdf"
        plot_grid_for_flag(
            flag, panel_data, mu_a_values, mu_b_values,
            n_values, color_map, cmap_base, norm, label_size, out,
        )


if __name__ == "__main__":
    main()
