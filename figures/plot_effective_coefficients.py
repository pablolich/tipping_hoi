import json
import glob
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

BASE = "/Users/pablolechon/Desktop/tipping_points_hoi/new_code/model_runs"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "pdffiles")
os.makedirs(OUTPUT_DIR, exist_ok=True)

REP_N   = 4
MU_VALS = [-0.1, 0.0, 0.1]

# ── load data ────────────────────────────────────────────────────────────────
folders = glob.glob(
    os.path.join(BASE, "2_bank_balanced_50_models_n_3-10_128_dirs_sA_1.0_sB_1.0_muA_*_muB_*")
)
pattern = re.compile(r"muA_([-\d.]+)_muB_([-\d.]+)")

def make_store():
    return {n: {"offdiag_A": [], "offdiag_B": []} for n in range(3, 11)}

data = {}
for muA in MU_VALS:
    for muB in MU_VALS:
        data[(muA, muB)] = make_store()

for folder in folders:
    m = pattern.search(folder)
    if not m:
        continue
    muA = round(float(m.group(1)), 1)
    muB = round(float(m.group(2)), 1)
    if muA not in MU_VALS or muB not in MU_VALS:
        continue

    for jf in glob.glob(os.path.join(folder, "*.json")):
        with open(jf) as f:
            d = json.load(f)
        n     = d["n"]
        if n != REP_N:
            continue
        A     = np.array(d["A"])
        B_raw = np.array(d["B"])

        eye_mask  = np.eye(n, dtype=bool)
        offdiag_A = A[~eye_mask]

        diag_mask = np.zeros((n, n, n), dtype=bool)
        for i in range(n):
            diag_mask[i, i, i] = True
        offdiag_B = B_raw[~diag_mask]

        store = data[(muA, muB)][n]
        store["offdiag_A"].append(offdiag_A)
        store["offdiag_B"].append(offdiag_B)

# ── bin edges ─────────────────────────────────────────────────────────────────
all_A = []
for muB in MU_VALS:
    vals = data[(0.0, muB)][REP_N]["offdiag_A"]
    if vals:
        all_A.extend(np.concatenate(vals).tolist())
all_A  = np.array(all_A)
bins_A = np.linspace(np.percentile(all_A, 0.5), np.percentile(all_A, 99.5), 60)

all_B = []
for muB in MU_VALS:
    vals = data[(0.0, muB)][REP_N]["offdiag_B"]
    if vals:
        all_B.extend(np.concatenate(vals).tolist())
all_B  = np.array(all_B)
bins_B = np.linspace(np.percentile(all_B, 0.5), np.percentile(all_B, 99.5), 60)

# ── colors for mu_B ───────────────────────────────────────────────────────────
cmap_muB   = plt.cm.coolwarm
colors_muB = [cmap_muB(0.1 + 0.4 * i) for i in range(len(MU_VALS))]

# ── figure: 2 side-by-side panels, 2× wider than tall ────────────────────────
FIG_W = 6.0
FIG_H = FIG_W / 2.0

fig, (ax_A, ax_B) = plt.subplots(2, 1, figsize=(FIG_W, FIG_H))
fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.12, hspace=0.55)

# ── Panel 1: A_ij off-diagonal, muA=0, n=REP_N (pooled over muB) ─────────────
pooled_A = []
for muB in MU_VALS:
    vals = data[(0.0, muB)][REP_N]["offdiag_A"]
    if vals:
        pooled_A.append(np.concatenate(vals))
if pooled_A:
    pooled_A = np.concatenate(pooled_A)
    ax_A.hist(pooled_A, bins=bins_A, histtype="step", color="steelblue",
              linewidth=1.6, density=True)

ax_A.axvline(0, color="k", linewidth=0.7, linestyle=":")
ax_A.set_xlabel(r"$A_{ij}$  ($j \neq i$)", fontsize=10)
ax_A.set_ylabel("Density", fontsize=10)
ax_A.set_title(rf"Off-diagonal $A_{{ij}}$  ($\mu_A=0,\; n={REP_N}$)", fontsize=10)
ax_A.tick_params(labelsize=8)

# ── Panel 2: B off-diagonal, muA=0, n=REP_N, three mu_B curves ───────────────
for ci, muB in enumerate(MU_VALS):
    vals = data[(0.0, muB)][REP_N]["offdiag_B"]
    if not vals:
        continue
    pooled = np.concatenate(vals)
    ax_B.hist(pooled, bins=bins_B, histtype="step", color=colors_muB[ci],
              linewidth=1.6, density=True, label=rf"$\mu_B={muB}$")

ax_B.axvline(0, color="k", linewidth=0.7, linestyle=":")
ax_B.set_xlabel(r"$B_{ijk}$  (off-diagonal)", fontsize=10)
ax_B.set_ylabel("Density", fontsize=10)
ax_B.set_title(rf"Off-diagonal $B_{{ijk}}$  ($\mu_A=0,\; n={REP_N}$)", fontsize=10)
ax_B.legend(fontsize=9, framealpha=0.7)
ax_B.tick_params(labelsize=8)

out = os.path.join(OUTPUT_DIR, "effective_coefficients.pdf")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
