#!/usr/bin/env python3
"""verify_weyl_bounds.py — SI figure: numerical verification of the upper
bounds on the spectral thresholds δ_A and δ_B across regimes.

Two panels:

  Baseline (μ_A = μ_B = 0).  No rank-1 mean contribution; the spectral
    thresholds are bounded by the centered-Wigner Weyl bounds
    √2 + √(2 ln n) (for δ_A) and 1 + √(2 ln n) (for δ_B).  Linear y-axis.

  Competitive (μ_A = μ_B = -0.1).  The slice-sum / row-sum constraints
    in the bank generator concentrate the negative mean on the diagonal
    of the symmetrised parts, producing rank-(n−1) contributions of
    leading order |μ_A|·n to δ_A and |μ_B|·n² to δ_B — exceeding the
    Weyl bounds by polynomial factors.  The plotted bounds add the
    deterministic mean-field contribution to the Weyl correction:
        δ_A ≤ |μ_A|·n      + √2 + √(2 ln n)
        δ_B ≤ |μ_B|·n²     + 1  + √(2 ln n)
    Log y-axis.

Facilitative (μ > 0) is omitted: the bank applies mean-field centering
in that branch, so its δ_A and δ_B distributions match the baseline.

Sampling mirrors sample_parameter_set_standard in
pipeline/generate_bank.jl, without the centering branch (we want the
raw bank thresholds, not the centered Wigner part).

Usage:
    python verify_weyl_bounds.py [--output PDF] [--samples N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


N_VALUES = [4, 8, 16, 32]

REGIMES = [
    {"key": "baseline",    "mu_A":  0.0, "mu_B":  0.0,
     "title": r"Baseline ($\mu_A=\mu_B=0$)", "log": False},
    {"key": "competitive", "mu_A": -0.1, "mu_B": -0.1,
     "title": r"Competitive ($\mu_A=\mu_B=-0.1$)", "log": True},
]

COLOR_A = "#b03a2e"   # red — δ_A
COLOR_B = "#2e6f9e"   # blue — δ_B
BOUND_COLOR = "#d97a3c"   # orange — analytic bound


def hatB_from_tensor(B: np.ndarray) -> np.ndarray:
    """B̂[i, j] = Σ_k (B[j, k, i] + B[k, j, i]).  Mirrors the Julia helper."""
    return (B.sum(axis=0) + B.sum(axis=1)).T


def sample_delta_A(rng: np.random.Generator, n: int, mu_A: float,
                   sigma_A: float = 1.0) -> float:
    """δ_A = max eigenvalue of S_A = (A0 + A0ᵀ)/2 (un-centered)."""
    sd = sigma_A / np.sqrt(n)
    A0 = mu_A + rng.normal(0.0, sd, size=(n, n))
    np.fill_diagonal(A0, 0.0)
    A0[np.diag_indices(n)] = -A0.sum(axis=1)
    S_A = (A0 + A0.T) / 2
    return float(np.linalg.eigvalsh(S_A).max())


def sample_delta_B(rng: np.random.Generator, n: int, mu_B: float,
                   sigma_B: float = 1.0) -> float:
    """δ_B = max eigenvalue of S̃ = B̂_0 + B̂_0ᵀ (un-centered) divided by 4."""
    sd = sigma_B / n
    B0 = mu_B + rng.normal(0.0, sd, size=(n, n, n))
    diag_idx = (np.arange(n),) * 3
    B0[diag_idx] = 0.0
    slice_sums = B0.sum(axis=(0, 1))
    B0[diag_idx] = -slice_sums
    Bhat0 = hatB_from_tensor(B0)
    S_tilde = Bhat0 + Bhat0.T
    return float(np.linalg.eigvalsh(S_tilde).max() / 4.0)


def collect_samples(n_values: list[int], mu_A: float, mu_B: float,
                    n_samples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out: dict[int, dict[str, np.ndarray]] = {}
    for n in n_values:
        dA = np.empty(n_samples)
        dB = np.empty(n_samples)
        for k in range(n_samples):
            dA[k] = sample_delta_A(rng, n, mu_A)
            dB[k] = sample_delta_B(rng, n, mu_B)
        out[n] = {"delta_A": dA, "delta_B": dB}
        print(f"  n={n:3d}: <δ_A>={dA.mean():.3g}  <δ_B>={dB.mean():.3g}")
    return out


def bound_A(n: float | np.ndarray, mu_A: float) -> np.ndarray:
    n = np.asarray(n, dtype=float)
    return abs(mu_A) * n + np.sqrt(2.0) + np.sqrt(2.0 * np.log(n))


def bound_B(n: float | np.ndarray, mu_B: float) -> np.ndarray:
    n = np.asarray(n, dtype=float)
    return abs(mu_B) * n * n + 1.0 + np.sqrt(2.0 * np.log(n))


def plot_panel(ax, n_values, samples, regime, *,
               show_ylabel: bool, show_legend: bool,
               legend_frame: bool = False):
    n_arr = np.asarray(n_values, dtype=float)
    mu_A, mu_B = regime["mu_A"], regime["mu_B"]
    is_log = regime["log"]

    n_smooth = np.linspace(n_arr.min(), n_arr.max(), 400)
    bA_smooth = bound_A(n_smooth, mu_A)
    bB_smooth = bound_B(n_smooth, mu_B)

    means_A = np.array([samples[n]["delta_A"].mean() for n in n_values])
    means_B = np.array([samples[n]["delta_B"].mean() for n in n_values])
    p5_A    = np.array([np.percentile(samples[n]["delta_A"],  5) for n in n_values])
    p95_A   = np.array([np.percentile(samples[n]["delta_A"], 95) for n in n_values])
    p5_B    = np.array([np.percentile(samples[n]["delta_B"],  5) for n in n_values])
    p95_B   = np.array([np.percentile(samples[n]["delta_B"], 95) for n in n_values])

    rng = np.random.default_rng(42)
    for n in n_values:
        x = n * np.exp(rng.normal(0.0, 0.04, size=samples[n]["delta_A"].size))
        ax.scatter(x, samples[n]["delta_A"], s=6, color=COLOR_A,
                   alpha=0.22, linewidths=0, zorder=1.5)
    for n in n_values:
        x = n * np.exp(rng.normal(0.0, 0.04, size=samples[n]["delta_B"].size))
        ax.scatter(x, samples[n]["delta_B"], s=6, color=COLOR_B,
                   alpha=0.22, linewidths=0, zorder=1.5)

    ax.fill_between(n_arr, p5_A, p95_A, color=COLOR_A, alpha=0.18,
                    linewidth=0, zorder=2)
    ax.fill_between(n_arr, p5_B, p95_B, color=COLOR_B, alpha=0.18,
                    linewidth=0, zorder=2)

    ax.plot(n_arr, means_A, "o", color=COLOR_A, markersize=8,
            markeredgecolor="white", markeredgewidth=0.7,
            label=r"$\delta_A$", zorder=4)
    ax.plot(n_arr, means_B, "o", color=COLOR_B, markersize=8,
            markeredgecolor="white", markeredgewidth=0.7,
            label=r"$\delta_B$", zorder=4)

    if is_log:
        label_A = r"$|\mu_A|\,n + \sqrt{2}+\sqrt{2\ln n}\;\;(\delta_A)$"
        label_B = r"$|\mu_B|\,n^{2} + 1+\sqrt{2\ln n}\;\;(\delta_B)$"
    else:
        label_A = r"Weyl bound $\sqrt{2}+\sqrt{2\ln n}$ ($\delta_A$)"
        label_B = r"Weyl bound $1+\sqrt{2\ln n}$ ($\delta_B$)"

    ax.plot(n_smooth, bA_smooth, "--", color=COLOR_A, linewidth=1.5,
            alpha=0.95, zorder=3, label=label_A)
    ax.plot(n_smooth, bB_smooth, "--", color=COLOR_B, linewidth=1.5,
            alpha=0.95, zorder=3, label=label_B)

    ax.set_xscale("log", base=2)
    if is_log:
        ax.set_yscale("log")
    ax.set_xticks(n_values)
    ax.set_xticklabels([str(n) for n in n_values])
    ax.set_xlabel(r"System size $n$")
    if show_ylabel:
        ax.set_ylabel("Threshold value")
    ax.set_xlim(n_arr.min() * 0.85, n_arr.max() * 1.18)
    if show_legend:
        leg = ax.legend(loc="lower right", fontsize=11.5, ncol=1,
                        handlelength=2.2, borderpad=0.4,
                        frameon=legend_frame)
        if legend_frame:
            frame = leg.get_frame()
            frame.set_facecolor("white")
            frame.set_edgecolor("0.6")
            frame.set_linewidth(0.6)
            frame.set_alpha(0.65)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("pdffiles/si/figure_verify_bounds.pdf"),
        help="Output PDF path (default: pdffiles/si/figure_verify_bounds.pdf).",
    )
    parser.add_argument("--samples", type=int, default=80,
                        help="Number of replicate draws per n (default: 80).")
    parser.add_argument("--seed", type=int, default=2026,
                        help="Base RNG seed (default: 2026).")
    args = parser.parse_args()

    plt.rcParams.update({
        "font.family":      "serif",
        "mathtext.fontset": "cm",
        "font.size":        13,
        "axes.labelsize":   14,
        "axes.titlesize":   14,
        "axes.linewidth":   0.9,
        "xtick.labelsize":  12,
        "ytick.labelsize":  12,
        "xtick.direction":  "in",
        "ytick.direction":  "in",
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "legend.frameon":   False,
        "legend.fontsize":  11.0,
    })

    panels = []
    for i, regime in enumerate(REGIMES):
        print(f"Regime: {regime['key']} (μ_A={regime['mu_A']}, μ_B={regime['mu_B']})")
        samples = collect_samples(N_VALUES, regime["mu_A"], regime["mu_B"],
                                  args.samples, seed=args.seed + 1000 * i)
        panels.append((regime, samples))

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.5))
    for i, (regime, samples) in enumerate(panels):
        ax = axes[i]
        ax.set_box_aspect(1.0)
        plot_panel(ax, N_VALUES, samples, regime,
                   show_ylabel=(i == 0), show_legend=True,
                   legend_frame=True)
        ax.set_title(regime["title"], pad=6)
        ax.text(-0.12, 1.04, "ab"[i], transform=ax.transAxes,
                fontsize=18, fontweight="bold", va="bottom", ha="left")

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
