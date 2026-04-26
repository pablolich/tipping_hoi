#!/usr/bin/env python3
"""Plot inter-extinction gaps (top) and extinction δ values (bottom) by outcome.

Reads sequential_pruning output JSONs and produces a 2×2 figure:
  Top row    — Δδ between consecutive extinction events, split by Fold / Success
  Bottom row — δ at each extinction event,              split by Fold / Success

Usage:
    python figures/plot_gap_and_distance_distribution.py
    python figures/plot_gap_and_distance_distribution.py \\
        --run-dir model_runs/my_run \\
        --output figures/gap_and_distance_distribution.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_RUN_DIR = "model_runs/1_bank_10_models_n_10_10_dirs_b_dirichlet"
DEFAULT_OUTPUT  = "figures/pdffiles/gap_and_distance_distribution.pdf"
SEQ_SUBDIR      = "sequential_pruning"

FOLD_COLOR    = "#E6A817"   # amber
SUCCESS_COLOR = "#4477AA"   # steel blue
N_BINS        = 30


def load_data(run_dir: Path):
    """Return (fold_gaps, success_gaps, fold_deltas, success_deltas)."""
    seq_dir = run_dir / SEQ_SUBDIR
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"Sequential pruning output not found: {seq_dir}")

    fold_gaps, success_gaps       = [], []
    fold_deltas, success_deltas   = [], []

    for fpath in sorted(seq_dir.glob("*.json")):
        model = json.loads(fpath.read_text())
        for alpha_res in model.get("sequential_pruning_results", []):
            for dr in alpha_res.get("directions", []):
                status = dr["status"]
                if status not in ("fold", "success"):
                    continue
                events = dr["extinction_log"]["delta_events"]
                if not events:
                    continue
                deltas = sorted(events)
                gaps   = [deltas[i+1] - deltas[i] for i in range(len(deltas)-1)]
                if status == "fold":
                    fold_deltas.extend(deltas)
                    fold_gaps.extend(gaps)
                else:
                    success_deltas.extend(deltas)
                    success_gaps.extend(gaps)

    return (np.array(fold_gaps), np.array(success_gaps),
            np.array(fold_deltas), np.array(success_deltas))


def make_hist(ax, data, color, xlabel, title):
    if len(data) == 0:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        return
    median = np.median(data)
    mean   = np.mean(data)
    n      = len(data)
    ax.hist(data, bins=N_BINS, color=color, edgecolor="white", linewidth=0.4)
    ax.axvline(median, color=color, linestyle="--", linewidth=1.5,
               label=f"median={median:.2f}")
    ax.axvline(mean,   color=color, linestyle=":",  linewidth=1.5,
               label=f"mean={mean:.2f}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend(fontsize=8, framealpha=0.7)
    ax.text(0.97, 0.95, f"n={n}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--output",  default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    base    = Path(__file__).parent.parent
    run_dir = base / args.run_dir
    out     = base / args.output

    fold_gaps, success_gaps, fold_deltas, success_deltas = load_data(run_dir)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("Extinction δ values (left) and inter-extinction gaps (right) — Success (top), Fold (bottom)",
                 fontsize=11)

    make_hist(axes[0, 0], success_deltas, SUCCESS_COLOR, "δ at extinction",
              "Success — extinction δ values")
    make_hist(axes[1, 0], fold_deltas,    FOLD_COLOR,    "δ at extinction",
              "Fold — extinction δ values")
    make_hist(axes[0, 1], success_gaps,   SUCCESS_COLOR, "Δδ between extinctions",
              "Success — inter-extinction gaps")
    make_hist(axes[1, 1], fold_gaps,      FOLD_COLOR,    "Δδ between extinctions",
              "Fold — inter-extinction gaps")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
