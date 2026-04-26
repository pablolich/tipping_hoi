import glob
import json
from collections import defaultdict

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerBase

BASE = "/Users/pablolechon/Desktop/tipping_points_hoi/new_code/model_runs"
OUTPUT = "/Users/pablolechon/Desktop/tipping_points_hoi/new_code/figures/pdffiles/fold_fraction_scatter.pdf"

GIBBS_PATTERN = f"{BASE}/gibbs_n3to5_100models_50dirs/*_n5_*.json"
ECOLOGICAL_PATTERNS = {
    "karat_fmi": f"{BASE}/karatayev_FMI_10models_50dirs_seed42/*.json",
    "karat_rmi": f"{BASE}/karatayev_RMI_10models_50dirs_seed42/*.json",
    "lever": f"{BASE}/lever_Sp3Sa3_10models_50dirs_seed42/*.json",
    "mougi_random": f"{BASE}/mougi_random_n6_10models_50dirs_seed42/*.json",
    "aguade": f"{BASE}/aguade_n6_10models_50dirs_seed42/*.json",
    "stouffer": f"{BASE}/stouffer_random_n6_10models_50dirs_seed1/*.json",
}


def fold_fraction(filepath):
    with open(filepath) as fh:
        data = json.load(fh)

    scan_results = data.get("scan_results", [])
    if not scan_results:
        return None

    directions = scan_results[0].get("directions", [])
    if not directions:
        return None

    n_fold = sum(1 for row in directions if row["flag"] == "fold")
    return data["alpha_eff"], n_fold / len(directions)


def existing_points(pattern):
    pts = []
    for filepath in glob.glob(pattern):
        point = fold_fraction(filepath)
        if point is not None:
            pts.append(point)
    return pts


def group_stats(pts):
    alphas, ffs = zip(*pts)
    return np.mean(alphas), np.std(alphas), np.mean(ffs), np.std(ffs)


def summarize_group(pattern):
    pts = existing_points(pattern)
    return [group_stats(pts)] if pts else []


def build_groups():
    gibbs_by_param = defaultdict(lambda: defaultdict(list))
    for filepath in glob.glob(GIBBS_PATTERN):
        with open(filepath) as fh:
            data = json.load(fh)

        point = fold_fraction(filepath)
        if point is None:
            continue

        regime = data["metadata"]["regime"]
        param_id = data["metadata"]["parameterization_id"]
        gibbs_by_param[regime][param_id].append(point)

    def regime_stats(param_dict):
        per_param = [group_stats(pts)[:3:2] for pts in param_dict.values()]
        alphas, ffs = zip(*per_param)
        return [(np.mean(alphas), np.std(alphas), np.mean(ffs), np.std(ffs))]

    gibbs = {regime: regime_stats(gibbs_by_param[regime]) for regime in ("Q1", "Q2", "Q3")}

    return {
        "gibbs": gibbs,
        "karat_fmi": summarize_group(ECOLOGICAL_PATTERNS["karat_fmi"]),
        "karat_rmi": summarize_group(ECOLOGICAL_PATTERNS["karat_rmi"]),
        "lever": summarize_group(ECOLOGICAL_PATTERNS["lever"]),
        "mougi_random": summarize_group(ECOLOGICAL_PATTERNS["mougi_random"]),
        "aguade": summarize_group(ECOLOGICAL_PATTERNS["aguade"]),
        "stouffer": summarize_group(ECOLOGICAL_PATTERNS["stouffer"]),
    }


def plot_pts(ax, pts, marker, color, size, eb_kw):
    for mean_alpha, std_alpha, mean_ff, std_ff in pts:
        ax.errorbar(mean_alpha, mean_ff, xerr=std_alpha, yerr=std_ff, fmt="none", ecolor=color, **eb_kw)
        ax.scatter(
            [mean_alpha],
            [mean_ff],
            marker=marker,
            color=color,
            s=size,
            linewidths=0.6,
            edgecolors="black",
            zorder=3,
        )


class MultiMarkerHandler(HandlerBase):
    def __init__(self, markers, colors, msize=6):
        self.markers = markers
        self.colors = colors
        self.msize = msize
        super().__init__()

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        artists = []
        n = len(self.markers)
        for i, (marker, color) in enumerate(zip(self.markers, self.colors)):
            x = xdescent + (i + 0.5) * width / n
            y = ydescent + height / 2
            line = mlines.Line2D(
                [x],
                [y],
                linestyle="none",
                marker=marker,
                color=color,
                markersize=self.msize,
                markeredgewidth=0.8,
                markeredgecolor="black",
                transform=trans,
            )
            artists.append(line)
        return artists


def make_plot(groups):
    _t10 = [c["color"] for c in plt.rcParams["axes.prop_cycle"]]

    def _shade(hex_color, factor):
        h = hex_color.lstrip("#")
        r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        if factor >= 1.0:
            r = 1.0 - (1.0 - r) / factor
            g = 1.0 - (1.0 - g) / factor
            b = 1.0 - (1.0 - b) / factor
        else:
            r, g, b = r * factor, g * factor, b * factor
        return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))

    c_gibbs = {
        "Q1": _shade(_t10[0], 2.2),
        "Q2": _t10[0],
        "Q3": _shade(_t10[0], 0.55),
    }
    c_karat = {
        "FMI": _shade(_t10[1], 1.35),
        "RMI": _shade(_t10[1], 0.85),
    }
    c_lever = _t10[2]
    c_mougi = _shade(_t10[3], 1.25)
    c_aguade = _t10[4]
    c_stouffer = _t10[6]

    label_size = mpl.font_manager.FontProperties(size=mpl.rcParams["axes.labelsize"]).get_size_in_points()
    tick_label_size = label_size * 0.8
    fig, ax = plt.subplots(figsize=(2.475, 2.475))

    marker_gibbs = "d"
    marker_karat = "P"
    marker_lever = "*"
    marker_mougi = "o"
    marker_aguade = "h"
    marker_stouffer = "^"

    eb_kw = dict(elinewidth=0.5, capsize=0, zorder=2)

    for regime in ("Q1", "Q2", "Q3"):
        plot_pts(ax, groups["gibbs"][regime], marker_gibbs, c_gibbs[regime], 32, eb_kw)

    plot_pts(ax, groups["karat_fmi"], marker_karat, c_karat["FMI"], 32, eb_kw)
    plot_pts(ax, groups["karat_rmi"], marker_karat, c_karat["RMI"], 32, eb_kw)
    plot_pts(ax, groups["lever"], marker_lever, c_lever, 50, eb_kw)
    plot_pts(ax, groups["mougi_random"], marker_mougi, c_mougi, 28, eb_kw)
    plot_pts(ax, groups["aguade"], marker_aguade, c_aguade, 32, eb_kw)
    plot_pts(ax, groups["stouffer"], marker_stouffer, c_stouffer, 30, eb_kw)

    ax.set_xlabel(r"Non-linearity strength ($\alpha_\mathrm{eff}$)", fontsize=label_size)
    ax.set_ylabel("Fraction fold boundaries", fontsize=label_size)
    margin = 0.03
    ax.set_xlim(-margin, 1 + margin)
    ax.set_ylim(-margin, 1 + margin)
    ax.tick_params(axis="both", labelsize=tick_label_size)
    ax.set_yticks([0.0, 0.5, 1.0], labels=["0", "0.5", "1"])
    ax.set_aspect("equal")

    gibbs_proxy = mlines.Line2D([], [], linestyle="none")
    karat_proxy = mlines.Line2D([], [], linestyle="none")
    lever_proxy = mlines.Line2D([], [], linestyle="none")
    mougi_proxy = mlines.Line2D([], [], linestyle="none")
    aguade_proxy = mlines.Line2D([], [], linestyle="none")
    stouffer_proxy = mlines.Line2D([], [], linestyle="none")

    legend_handles = [gibbs_proxy, karat_proxy, lever_proxy, mougi_proxy, aguade_proxy, stouffer_proxy]
    legend_labels = [
        "Gibbs et al (2023)",
        "Karatayev et al (2023)",
        "Lever et al (2014)",
        "Mougi (2024)",
        "Aguade-Gorgorio et al (2024)",
        "Stouffer & Bascompte (2010)",
    ]
    legend_handlers = {
        gibbs_proxy: MultiMarkerHandler([marker_gibbs] * 3, [c_gibbs["Q1"], c_gibbs["Q2"], c_gibbs["Q3"]], msize=5),
        karat_proxy: MultiMarkerHandler([marker_karat, marker_karat], [c_karat["FMI"], c_karat["RMI"]], msize=5),
        lever_proxy: MultiMarkerHandler([marker_lever], [c_lever], msize=6),
        mougi_proxy: MultiMarkerHandler([marker_mougi], [c_mougi], msize=5),
        aguade_proxy: MultiMarkerHandler([marker_aguade], [c_aguade], msize=5),
        stouffer_proxy: MultiMarkerHandler([marker_stouffer], [c_stouffer], msize=5),
    }

    ax.legend(
        legend_handles,
        legend_labels,
        handler_map=legend_handlers,
        frameon=False,
        fontsize=7,
        handlelength=3.0,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
    )

    fig.subplots_adjust(right=0.75)
    plt.savefig(OUTPUT.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.savefig(OUTPUT, bbox_inches="tight")
    plt.show()

    print(f"Saved {OUTPUT.replace('.pdf', '.png')} and {OUTPUT}")
    print(f"Mougi random points used: {len(existing_points(ECOLOGICAL_PATTERNS['mougi_random']))}")
    print(f"Stouffer points used: {len(existing_points(ECOLOGICAL_PATTERNS['stouffer']))}")


def main():
    groups = build_groups()
    make_plot(groups)


if __name__ == "__main__":
    main()
