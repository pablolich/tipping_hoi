"""Plot boundary-type scatter panels from boundary_scan JSON output.

Reads model JSON files produced by boundary_scan.jl and plots the fraction of
boundary types across alpha values, coloured by n.

Usage:
    python figures/plot_boundary_type_scatter_panels.py \
        --input-root model_runs/minimal_test \
        --output figures/pdffiles/boundary_fraction_scatter_panels.pdf

    # Plot all four boundary types
    python figures/plot_boundary_type_scatter_panels.py \
        --input-root model_runs/minimal_test model_runs/test_verify \
        --output figures/pdffiles/boundary_fraction_scatter_panels.pdf \
        --plot-all
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# In the JSON produced by boundary_scan.jl the fold bifurcation boundary is
# stored as flag="fold".  The original CSV pipeline called it "complex".  We
# remap so that the rest of the plotting logic (titles, boundary_flags list)
# stays identical to the original script.
FLAG_REMAP = {"fold": "complex"}

DEFAULT_BOUNDARY_FLAGS = ["negative", "complex"]
ALL_BOUNDARY_FLAGS = ["negative", "complex", "unstable", "success"]
TITLE_MAP = {
    "negative": "Negative boundary",
    "complex": "Fold boundary",
    "unstable": "Unstable boundary",
    "success": "Success boundary",
}


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
    """Read boundary_scan JSON files and return per-model boundary fractions."""
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
                # File has not yet been processed by boundary_scan.jl
                continue

            n = int(data["n"])
            model_id = str(data.get("seed", path.stem))

            rows = []
            for alpha_result in data["scan_results"]:
                alpha = float(alpha_result["alpha"])
                for direction in alpha_result["directions"]:
                    raw_flag = str(direction["flag"]).lower()
                    boundary_type = FLAG_REMAP.get(raw_flag, raw_flag)
                    rows.append({
                        "alpha_value": alpha,
                        "boundary_type": boundary_type,
                        # All direction results from boundary_scan are valid
                        # (the algorithm always finds a boundary or reports
                        # "success" when none is found within max_pert).
                        "stable": True,
                    })

            if not rows:
                print(f"No scan directions found; skipping {path.name}")
                continue

            df = pd.DataFrame(rows)
            frac = boundary_fractions(df, boundary_types)
            if frac.empty:
                print(f"No stable boundaries; skipping {path.name}")
                continue

            frac["model_id"] = model_id
            frac["n"] = n
            frames.append(frac)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_colormaps(
    n_bins: int,
) -> Tuple[mpl.colors.Colormap, mpl.colors.Colormap, mpl.colors.Normalize]:
    colors = np.asarray(sns.color_palette("flare", n_colors=n_bins))
    cmap_base = mpl.colors.ListedColormap(colors)
    faded = np.concatenate([colors.copy(), 0.25 * np.ones((n_bins, 1))], axis=1)
    cmap_faded = mpl.colors.ListedColormap(faded)
    norm = mpl.colors.BoundaryNorm(np.arange(-0.5, n_bins + 0.5, 1), n_bins)
    return cmap_base, cmap_faded, norm


def plot_boundary_type_scatter_panels(
    all_frac: pd.DataFrame, output_path: Path, boundary_flags: List[str]
) -> None:
    if all_frac.empty:
        print("No boundary fractions available for plotting.")
        return

    n_levels = np.sort(all_frac["n"].unique())
    n_bins = len(n_levels)
    n_map = {n_val: idx for idx, n_val in enumerate(n_levels)}
    all_frac = all_frac.copy()
    all_frac["n_index"] = all_frac["n"].map(n_map)

    cmap, cmap_faded, norm = build_colormaps(n_bins)

    n_panels = len(boundary_flags)
    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_panels,
        figsize=(2.475 * n_panels, 2.475),
        sharey=True,
    )
    if n_panels == 1:
        axes = [axes]
    fig.subplots_adjust(left=0.10, right=0.90, bottom=0.16, top=0.92, wspace=0.20)

    label_size = mpl.font_manager.FontProperties(
        size=mpl.rcParams["axes.labelsize"]
    ).get_size_in_points()
    margin = 0.03
    for idx, flag in enumerate(boundary_flags):
        ax = axes[idx]
        ax.set_title(TITLE_MAP.get(flag, flag), fontsize=label_size)
        ax.set_xlim(0.0 - margin - 0.01, 1.0 + margin)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_ylim(0.0 - margin, 1.0 + margin)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_aspect(1, adjustable="box")

        if idx == 0:
            ax.set_ylabel("Fraction", fontsize=label_size)
            ax.set_yticklabels(["0", "0.5", "1"])
        else:
            ax.tick_params(labelleft=False)

        sub = all_frac[all_frac["boundary_type"] == flag]
        if sub.empty:
            print(f"No {flag} boundary data found; leaving panel empty.")
            continue

        ax.scatter(
            sub["alpha_value"],
            sub["fraction"],
            c=sub["n_index"],
            cmap=cmap_faded,
            norm=norm,
            s=18 * 0.65,
            linewidths=0.0,
        )
        med = (
            sub.groupby(["n", "alpha_value"], as_index=False)["fraction"]
            .median()
            .reset_index(drop=True)
        )
        med["n_index"] = med["n"].map(n_map)
        ax.scatter(
            med["alpha_value"],
            med["fraction"],
            c=med["n_index"],
            cmap=cmap,
            norm=norm,
            s=50 * 0.65,
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
        )

    fig.canvas.draw()
    right_pos = axes[-1].get_position()
    cax = fig.add_axes([right_pos.x1 + 0.02, right_pos.y0, 0.02, right_pos.height])

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, ticks=np.arange(n_bins))
    cbar.ax.set_yticklabels([str(int(n)) for n in n_levels])
    cbar.ax.set_title("n", pad=4, fontsize=label_size)

    if n_panels == 1:
        axes[0].set_xlabel(
            "HOI strength ($\\alpha$)", labelpad=6, fontsize=label_size
        )
    else:
        left_pos = axes[0].get_position()
        right_pos = axes[-1].get_position()
        center_x = (left_pos.x0 + right_pos.x1) / 2
        fig.text(
            center_x,
            0.015,
            "HOI strength ($\\alpha$)",
            ha="center",
            va="bottom",
            fontsize=label_size,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)

    if output_path.suffix.lower() != ".pdf":
        pdf_path = output_path.with_suffix(".pdf")
        fig.savefig(pdf_path)
        print(f"Saved {pdf_path}")
    print(f"Saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot boundary-type scatter panels from boundary_scan JSON output. "
            "Reads model JSON files (with scan_results) from one or more run "
            "directories and plots fraction of each boundary type vs alpha."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        nargs="+",
        default=[Path("model_runs")],
        help=(
            "One or more directories to search for model JSON files. "
            "Each directory is searched recursively. "
            "Files without a 'scan_results' key are skipped. "
            "Default: model_runs/"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures") / "pdffiles" / "boundary_fraction_scatter_panels.pdf",
        help="Output path for the figure.",
    )
    parser.add_argument(
        "--plot-all",
        action="store_true",
        help=(
            "Plot all boundary types (negative, fold, unstable, success). "
            "By default only negative and fold are plotted."
        ),
    )
    args = parser.parse_args()

    boundary_flags = list(ALL_BOUNDARY_FLAGS if args.plot_all else DEFAULT_BOUNDARY_FLAGS)

    all_frac = collect_boundary_fractions(args.input_root, boundary_flags)
    if all_frac.empty:
        print("No model files found for scatter panels.")
        return
    plot_boundary_type_scatter_panels(all_frac, args.output, boundary_flags)


if __name__ == "__main__":
    main()
