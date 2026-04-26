#!/usr/bin/env python3
"""plot_equilibria_counts.py — visualize enumerate_equilibria.jl output.

Produces 3 PDFs in figures/pdffiles/:
  equilibria_count_total.pdf               — mean # feasible equilibria vs alpha, faceted by n
  equilibria_count_stable.pdf              — mean # stable equilibria vs alpha, faceted by n
  equilibria_count_interior_boundary.pdf   — interior vs boundary split, 2 rows x 7 cols

All three figures share a common layout: panels faceted by community size n ∈ {4..10},
four colored lines per panel (one per bank), shaded mean ± SEM across models.

Usage (from new_code/):
  python figures/plot_equilibria_counts.py \
    --input figures/all_equilibria.csv \
    --out-dir figures/pdffiles/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BANK_ORDER = [
    "2_bank_elegant_50_models_n_4-20_128_dirs_muB_-0.1",
    "2_bank_elegant_50_models_n_4-20_128_dirs_muB_0.0",
    "2_bank_elegant_50_models_n_4-20_128_dirs_muB_0.1",
    "2_bank_unique_equilibrium_50_models_n_4-10_128_dirs",
]

BANK_COLORS = {
    "2_bank_elegant_50_models_n_4-20_128_dirs_muB_-0.1": "#2166ac",
    "2_bank_elegant_50_models_n_4-20_128_dirs_muB_0.0":  "#4dac26",
    "2_bank_elegant_50_models_n_4-20_128_dirs_muB_0.1":  "#d6604d",
    "2_bank_unique_equilibrium_50_models_n_4-10_128_dirs": "#6a3d9a",
}

BANK_LABELS = {
    "2_bank_elegant_50_models_n_4-20_128_dirs_muB_-0.1": r"$\mu_B = -0.1$",
    "2_bank_elegant_50_models_n_4-20_128_dirs_muB_0.0":  r"$\mu_B = 0.0$",
    "2_bank_elegant_50_models_n_4-20_128_dirs_muB_0.1":  r"$\mu_B = +0.1$",
    "2_bank_unique_equilibrium_50_models_n_4-10_128_dirs": "unique equilibrium",
}

X_COLS      = [f"x_{i}" for i in range(1, 11)]
SUPPORT_TOL = 1e-6
METRICS     = ["total_count", "stable_count", "interior_count", "boundary_count"]


def load_per_model(csv_path: Path) -> pd.DataFrame:
    """Load CSV and reduce to one row per (bank, model_file, n, alpha) with count metrics.

    Models/alphas with zero feasible equilibria are absent from the raw CSV; we rebuild
    the full cross-product and fill missing counts with 0 so averages are not biased.
    """
    df = pd.read_csv(csv_path)

    support_mask = df[X_COLS] > SUPPORT_TOL
    df["support_size"] = support_mask.sum(axis=1)
    df["is_interior"]  = (df["support_size"] == df["n"]).astype(int)
    df["is_boundary"]  = 1 - df["is_interior"]
    df["stable_int"]   = df["stable"].astype(int)

    per_model = (
        df.groupby(["bank", "n", "alpha", "model_file"], as_index=False)
          .agg(
              total_count    = ("equilibrium_id", "size"),
              stable_count   = ("stable_int",     "sum"),
              interior_count = ("is_interior",    "sum"),
              boundary_count = ("is_boundary",    "sum"),
          )
    )

    index_df = (
        df.drop_duplicates(subset=["bank", "model_file", "n"])
          [["bank", "model_file", "n"]]
    )
    alphas = np.sort(df["alpha"].unique())
    full = (
        index_df.assign(_k=1)
                .merge(pd.DataFrame({"alpha": alphas, "_k": 1}), on="_k")
                .drop(columns="_k")
    )
    per_model = full.merge(
        per_model, on=["bank", "model_file", "n", "alpha"], how="left"
    )
    for col in METRICS:
        per_model[col] = per_model[col].fillna(0).astype(int)
    return per_model


def aggregate(per_model: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        per_model.groupby(["bank", "n", "alpha"], as_index=False)
                 .agg(
                     mean     = (metric, "mean"),
                     sem      = (metric, "sem"),
                     n_models = (metric, "size"),
                 )
    )


def _draw_bank_lines(ax, agg: pd.DataFrame, n_val: int, *,
                     show_label: bool, normalize: bool = False) -> None:
    scale = (2.0 ** n_val) if normalize else 1.0
    for bank in BANK_ORDER:
        sub = (
            agg[(agg["bank"] == bank) & (agg["n"] == n_val)]
               .sort_values("alpha")
        )
        if sub.empty:
            continue
        color = BANK_COLORS[bank]
        label = BANK_LABELS[bank] if show_label else None
        mean = sub["mean"] / scale
        sem  = sub["sem"].fillna(0.0) / scale
        ax.plot(sub["alpha"], mean,
                color=color, marker="o", ms=3, lw=1.5, label=label)
        ax.fill_between(sub["alpha"], mean - sem, mean + sem,
                        color=color, alpha=0.18, linewidth=0)


def plot_metric_faceted(per_model: pd.DataFrame, metric: str, *,
                        title: str, ylabel: str, out_path: Path,
                        log_y: bool = False, normalize: bool = False) -> None:
    agg = aggregate(per_model, metric)
    n_values = sorted(per_model["n"].unique())
    n_panels = len(n_values)
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(2.6 * n_panels + 2.0, 3.2),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes[0]

    for idx, n_val in enumerate(n_values):
        ax = axes[idx]
        _draw_bank_lines(ax, agg, n_val,
                         show_label=(idx == 0), normalize=normalize)
        ax.set_title(f"n = {n_val}", fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.set_xlabel(r"$\alpha$")
        ax.set_xlim(-0.03, 1.03)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        if log_y:
            ax.set_yscale("log", nonpositive="clip")

    axes[0].set_ylabel(ylabel)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="center right", bbox_to_anchor=(1.02, 0.5),
               fontsize=10, frameon=False, title="bank")

    fig.suptitle(title, fontsize=13, y=1.05)
    fig.tight_layout(rect=(0.0, 0.0, 0.92, 1.0))
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_interior_vs_boundary(per_model: pd.DataFrame, out_path: Path,
                               log_y: bool = False,
                               normalize_interior: bool = False,
                               normalize_boundary: bool = False) -> None:
    agg_int = aggregate(per_model, "interior_count")
    agg_bnd = aggregate(per_model, "boundary_count")

    n_values = sorted(per_model["n"].unique())
    n_panels = len(n_values)
    fig, axes = plt.subplots(2, n_panels,
                             figsize=(2.6 * n_panels + 2.0, 6),
                             sharex=True, squeeze=False)

    for col in range(n_panels):
        axes[0, col].sharey(axes[0, 0])
        axes[1, col].sharey(axes[1, 0])

    for col, n_val in enumerate(n_values):
        ax_top = axes[0, col]
        _draw_bank_lines(ax_top, agg_int, n_val,
                         show_label=(col == 0), normalize=normalize_interior)
        ax_top.set_title(f"n = {n_val}", fontsize=10)
        ax_top.grid(alpha=0.3, which="both")
        ax_top.set_xlim(-0.03, 1.03)
        ax_top.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        if log_y:
            ax_top.set_yscale("log", nonpositive="clip")
        if col > 0:
            plt.setp(ax_top.get_yticklabels(), visible=False)

        ax_bot = axes[1, col]
        _draw_bank_lines(ax_bot, agg_bnd, n_val,
                         show_label=False, normalize=normalize_boundary)
        ax_bot.grid(alpha=0.3, which="both")
        ax_bot.set_xlabel(r"$\alpha$")
        ax_bot.set_xlim(-0.03, 1.03)
        ax_bot.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        if log_y:
            ax_bot.set_yscale("log", nonpositive="clip")
        if col > 0:
            plt.setp(ax_bot.get_yticklabels(), visible=False)

    axes[0, 0].set_ylabel("mean # interior eq." + (r" / $2^n$" if normalize_interior else ""))
    axes[1, 0].set_ylabel("mean # boundary eq." + (r" / $2^n$" if normalize_boundary else ""))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="center right", bbox_to_anchor=(1.02, 0.5),
               fontsize=10, frameon=False, title="bank")

    suptitle = "Interior vs boundary equilibria (mean across models)"
    tags = []
    if normalize_interior:
        tags.append("interior")
    if normalize_boundary:
        tags.append("boundary")
    if tags:
        suptitle += r" — " + "/".join(tags) + r" normalized by $2^n$"
    fig.suptitle(suptitle, fontsize=13, y=1.02)
    fig.tight_layout(rect=(0.0, 0.0, 0.92, 1.0))
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",   type=Path,
                        default=Path("figures/all_equilibria.csv"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("figures/pdffiles"))
    parser.add_argument("--log-y",   action="store_true",
                        help="Use log-scale y-axis (clips zero/negative)")
    parser.add_argument("--normalize", action="store_true",
                        help="Divide mean/SEM by 2^n per panel (fraction of 2^n possible supports)")
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"input CSV not found: {args.input}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_model = load_per_model(args.input)
    print(f"Loaded {len(per_model)} (bank, model, n, alpha) rows from {args.input}")

    suffix = ("_norm" if args.normalize else "") + ("_log" if args.log_y else "")
    norm_tag = r" / $2^n$" if args.normalize else ""
    log_tag  = " (log)" if args.log_y else ""
    plot_metric_faceted(
        per_model, "total_count",
        title="Average number of feasible equilibria" + (r" (normalized by $2^n$)" if args.normalize else ""),
        ylabel="mean # feasible equilibria" + norm_tag + log_tag,
        out_path=args.out_dir / f"equilibria_count_total{suffix}.pdf",
        log_y=args.log_y, normalize=args.normalize,
    )
    plot_metric_faceted(
        per_model, "stable_count",
        title="Average number of stable equilibria",
        ylabel="mean # stable equilibria" + log_tag,
        out_path=args.out_dir / f"equilibria_count_stable{'_log' if args.log_y else ''}.pdf",
        log_y=args.log_y, normalize=False,
    )
    ib_suffix = ("_normbnd" if args.normalize else "") + ("_log" if args.log_y else "")
    plot_interior_vs_boundary(
        per_model,
        out_path=args.out_dir / f"equilibria_count_interior_boundary{ib_suffix}.pdf",
        log_y=args.log_y,
        normalize_interior=False,
        normalize_boundary=args.normalize,
    )


if __name__ == "__main__":
    main()
