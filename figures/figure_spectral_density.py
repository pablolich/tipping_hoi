#!/usr/bin/env python3
"""Figure: Spectral density of S_B across the three mu_B regimes.

Reproduces data/figure_inputs/figure_spectral_density.pdf (SI Sec. on parameter
sampling protocol). Uses the ensemble described in Eq. (eq:raw)+(eq:zerosum):
    tilde B_{ijk} ~ N(mu_B, 1/n^2)         off-diagonal
    tilde B_{iii} = -sum_{(j,k) != (i,i)} tilde B_{ijk}    (zero slice sums)

The plotted symmetric matrix is M = hat B + hat B^T, with
    hat B_{ij} = sum_k (tilde B_{ijk} + tilde B_{ikj}),
i.e. M_{ij} = sum_k(B_{ijk}+B_{ikj}+B_{jik}+B_{jki}) = 4 S_B in the SI notation.

The three panels span mu_B in {-0.1, 0, 0.1}; the dashed Weyl-bound brackets are
placed at lambda_spike +/- 4(1 + sqrt(2 ln n)), where lambda_spike = -4 n^2 mu_B
is the (n-1)-fold eigenvalue of E[M].

Run:
    python figures/figure_spectral_density.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


N = 20
MU_B_VALUES = (-0.1, 0.0, 0.1)
REGIME_LABELS = {-0.1: "competitive", 0.0: "baseline", 0.1: "facilitative"}
PANEL_COLORS = {-0.1: "#d98477", 0.0: "#6e95c9", 0.1: "#7fae84"}
N_REALIZATIONS = 600
SEED = 0


def sample_M(n: int, mu_B: float, rng: np.random.Generator) -> np.ndarray:
    """Sample tilde B and return M = hat B + hat B^T (n x n symmetric)."""
    sigma = 1.0 / n
    tildeB = rng.normal(loc=mu_B, scale=sigma, size=(n, n, n))
    # Enforce zero slice sums by overwriting B_{iii}
    for i in range(n):
        s = tildeB[i].sum() - tildeB[i, i, i]
        tildeB[i, i, i] = -s
    # hat B_{ij} = sum_k (B_{ijk} + B_{ikj})
    hatB = tildeB.sum(axis=2) + tildeB.sum(axis=1)
    return hatB + hatB.T


def collect_eigenvalues(n: int, mu_B: float, n_realizations: int,
                        rng: np.random.Generator) -> np.ndarray:
    eigs = np.empty(n * n_realizations)
    for r in range(n_realizations):
        M = sample_M(n, mu_B, rng)
        eigs[r * n : (r + 1) * n] = np.linalg.eigvalsh(M)
    return eigs


def weyl_bound(n: int) -> float:
    """Weyl-type bound on lambda_max(tilde S_B), scaled to M = 4 S_B."""
    return 4.0 * (1.0 + np.sqrt(2.0 * np.log(n)))


def main(output: Path) -> None:
    rng = np.random.default_rng(SEED)
    eig_by_mu = {mu: collect_eigenvalues(N, mu, N_REALIZATIONS, rng)
                 for mu in MU_B_VALUES}

    bound = weyl_bound(N)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharey=True)

    for ax, mu in zip(axes, MU_B_VALUES):
        eigs = eig_by_mu[mu]
        color = PANEL_COLORS[mu]
        spike = -4.0 * N**2 * mu  # (n-1)-fold eigenvalue of E[M]

        # Outer plot: full spectrum (shows the isolated lambda_1 outlier near 0).
        full_lo = min(eigs.min(), spike - 1.5 * bound, -2.0 * bound)
        full_hi = max(eigs.max(), spike + 1.5 * bound, 2.0 * bound)
        pad = 0.04 * (full_hi - full_lo)
        ax.hist(eigs, bins=200, range=(full_lo - pad, full_hi + pad),
                density=True, color=color, edgecolor=color, linewidth=0)
        ax.set_xlim(full_lo - pad, full_hi + pad)
        ax.set_ylim(0, 0.085)

        # Weyl-bound dashed brackets around the (n-1)-fold spike.
        for x in (spike - bound, spike + bound):
            ax.axvline(x, color="#7a3f99", linestyle="--", linewidth=1.4)

        if mu == 0.0:
            mu_str = "0"
        else:
            mu_str = f"{mu:+.1f}"
        ax.set_title(rf"$\mu_B = {mu_str}$ ({REGIME_LABELS[mu]})", fontsize=16)
        ax.set_xlabel(r"Eigenvalue $\lambda$", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)

        # Bulk inset for the two regimes that have an isolated outlier.
        if mu != 0.0:
            half = 1.5 * bound
            if mu < 0:
                bbox = (0.13, 0.40, 0.42, 0.52)  # (x0, y0, w, h) in axes fraction
                loc = "upper left"
            else:
                bbox = (0.45, 0.40, 0.42, 0.52)
                loc = "upper right"
            inset = inset_axes(ax, width="100%", height="100%",
                               bbox_to_anchor=bbox,
                               bbox_transform=ax.transAxes,
                               loc=loc, borderpad=0)
            inset.patch.set_alpha(0.0)
            inset.hist(eigs, bins=80, range=(spike - half, spike + half),
                       density=True, color=color, edgecolor=color, linewidth=0)
            inset.set_xlim(spike - half, spike + half)
            inset.tick_params(axis="both", labelsize=10)
            inset.set_ylim(0, 0.085)
            for x in (spike - bound, spike + bound):
                inset.axvline(x, color="#7a3f99", linestyle="--", linewidth=1.0)
        else:
            # Annotate the Weyl bound on the baseline panel.
            ax.annotate("Weyl bound", xy=(bound, 0.072), xytext=(bound * 0.55, 0.078),
                        fontsize=13, color="#7a3f99",
                        arrowprops=dict(arrowstyle="-", color="#7a3f99", lw=0.8))

    axes[0].set_ylabel("Spectral density", fontsize=14)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    print(f"wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    default_out = (Path(__file__).resolve().parent.parent
                   / "pdffiles" / "si" / "figure_spectral_density.pdf")
    parser.add_argument("--output", type=Path, default=default_out)
    args = parser.parse_args()
    main(args.output)
