#!/usr/bin/env python3
"""
Plot interaction coefficient distributions for the clean_balanced banks, n=4.
Two panels:
  Left  – A off-diagonal entries (mu_A = 0.0, same across all banks)
  Right – B non-pure-diagonal entries for mu_B ∈ {-0.1, 0.0, 0.1}
"""

import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BANK_ROOT = os.path.join(os.path.dirname(__file__), "..", "model_runs")
MU_B_VALUES = [-0.1, 0.0, 0.1]
N_TARGET = 4

BANK_TEMPLATE = (
    "2_bank_clean_balanced_50_models_n_4-20_128_dirs_"
    "sA_1.0_sB_1.0_muA_0.0_muB_{muB}"
)

COLORS = {-0.1: "#e07b54", 0.0: "#4c72b0", 0.1: "#55a868"}


def load_entries(mu_b: float):
    """Return (A_offdiag, B_nonpure) arrays for all n=4 models in the bank."""
    tag = f"{mu_b:.1f}" if mu_b != 0 else "0.0"
    bank_dir = os.path.join(BANK_ROOT, BANK_TEMPLATE.format(muB=tag))
    pattern = os.path.join(bank_dir, f"*_n_{N_TARGET}_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No n={N_TARGET} files found in {bank_dir}")

    A_vals, B_vals = [], []
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        n = d["n"]
        A = np.array(d["A"])       # (n,n)
        B = np.array(d["B"])       # (n,n,n); B[j,k,i] in Julia → stored as B_np[j,k,i]

        # A off-diagonal entries (diagonal is constrained, not free)
        mask_A = ~np.eye(n, dtype=bool)
        A_vals.append(A[mask_A].ravel())

        # B non-pure-diagonal entries:
        # pure diagonal = B[i,i,i] for each i (third index = species in Julia)
        for i in range(n):
            slice_i = B[:, :, i].copy()   # all (j,k) affecting species i
            slice_i[i, i] = np.nan        # exclude pure diagonal B[i,i,i]
            B_vals.append(slice_i[~np.isnan(slice_i)].ravel())

    return np.concatenate(A_vals), np.concatenate(B_vals)


# ── Load ─────────────────────────────────────────────────────────────────────
data = {}
for mu_b in MU_B_VALUES:
    data[mu_b] = load_entries(mu_b)
    print(f"mu_B={mu_b:+.1f}  |  A: {len(data[mu_b][0])} entries  "
          f"|  B: {len(data[mu_b][1])} entries")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle(f"Interaction coefficient distributions  (clean_balanced, n={N_TARGET})",
             fontsize=12, y=1.01)

BINS = 60
ALPHA = 0.55

# Left panel: A off-diagonal (mu_A = 0.0, all banks share same A distribution)
ax = axes[0]
A_ref, _ = data[0.0]
counts, edges = np.histogram(A_ref, bins=BINS, density=True)
centers = (edges[:-1] + edges[1:]) / 2
ax.plot(centers, counts, color="#4c72b0", lw=1.8, label="μ_A = 0.0")
ax.set_xlabel("A off-diagonal coefficient", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Pairwise interactions (A)", fontsize=11)
ax.set_xlim(-1, 1)
ax.legend(fontsize=10)
ax.xaxis.set_major_locator(ticker.MaxNLocator(6))

# Right panel: B non-pure-diagonal for each mu_B
ax = axes[1]
for mu_b in MU_B_VALUES:
    _, B_vals = data[mu_b]
    counts, edges = np.histogram(B_vals, bins=BINS, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.plot(centers, counts, color=COLORS[mu_b], lw=1.8, label=f"μ_B = {mu_b:+.1f}")
ax.set_xlim(-1, 1)
ax.set_xlabel("B non-pure-diagonal coefficient", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("HOI coefficients (B)", fontsize=11)
ax.legend(fontsize=10)
ax.xaxis.set_major_locator(ticker.MaxNLocator(6))

fig.tight_layout()
plt.show()
