#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.feather as feather
import pyarrow.ipc as ipc


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
    "legend.fontsize":  11.7,
})

REQUIRED_COLUMNS = [
    "model_id",
    "n",
    "sys_id",
    "row_idx",
    "alpha",
    "alpha_idx",
    "ray_id",
    "delta",
    "species_id",
    "abundance",
    "source_type",
    "pass_direction",
    "branch_id",
]

ID_COLUMNS = ["model_id", "n", "sys_id", "row_idx", "alpha", "alpha_idx", "ray_id"]
NUMERIC_COLUMNS = ["n", "sys_id", "row_idx", "alpha", "alpha_idx", "ray_id", "delta", "species_id", "abundance"]

VALID_SOURCE_TYPES = {"algebraic", "dynamic"}
VALID_PASS_DIRECTIONS = {"none", "forward", "backward"}
VALID_BRANCHES = {
    "pre_fold_1",
    "pre_fold_2",
    "post_forward",
    "post_backward",
    "extinction_forward",
    "extinction_backward",
}
BOUNDARY_TYPES = ["extinction", "fold"]
BOUNDARY_TITLES = {"fold": "Abrupt collapse", "extinction": "Gradual extinction"}

BRANCH_ORDER = {
    "pre_fold_1": 0,
    "pre_fold_2": 1,
    "post_forward": 2,
    "post_backward": 3,
    "extinction_forward": 4,
    "extinction_backward": 5,
}
BRANCH_LINESTYLES = {
    "pre_fold_1": "-",
    "pre_fold_2": ":",
    "post_forward": "-",
    "post_backward": "-",
    "extinction_forward": "-",
    "extinction_backward": ":",
}

Y_FLOOR = -0.1
COOLWARM_QUAL3_NO_GRAY = ["#0072B2", "#D55E00", "#009E73"]
FORWARD_COLOR = "#D55E00"
BACKWARD_COLOR = "#0072B2"
FORWARD_MARKER = ">"
BACKWARD_MARKER = "<"


def is_stable_value(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value) != 0
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return False
        return float(value) != 0.0
    text = str(value).strip().lower()
    return text in {"true", "t", "1", "yes", "y"}


def read_arrow_table(path: Path) -> pd.DataFrame:
    try:
        with pa.memory_map(path.as_posix(), "r") as source:
            return ipc.open_file(source).read_all().to_pandas()
    except Exception:
        try:
            with pa.memory_map(path.as_posix(), "r") as source:
                return ipc.open_stream(source).read_all().to_pandas()
        except Exception:
            try:
                return feather.read_table(path.as_posix()).to_pandas()
            except Exception:
                return ds.dataset(path.as_posix(), format="ipc").to_table().to_pandas()


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".arrow":
        df = read_arrow_table(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(
            f"Unsupported input extension '{path.suffix}'. Use .arrow (preferred) or .csv."
        )

    print(f"[INFO] Loaded input file: {path}")
    print(f"[INFO] Rows: {len(df)} | Columns: {len(df.columns)}")
    return df


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
            + ". Confirm this was produced by build_hysteresis_table.jl."
        )

    out = df.copy()

    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        n_bad = int(out[col].isna().sum())
        if n_bad > 0:
            raise ValueError(
                f"Column '{col}' has {n_bad} non-numeric or missing values; cannot plot."
            )

    for col in ["n", "sys_id", "row_idx", "alpha_idx", "ray_id", "species_id"]:
        out[col] = out[col].astype(int)

    out["source_type"] = out["source_type"].astype(str).str.strip().str.lower()
    out["pass_direction"] = out["pass_direction"].astype(str).str.strip().str.lower()
    out["branch_id"] = out["branch_id"].astype("string").str.strip().str.lower()
    out["branch_id"] = out["branch_id"].replace({"": pd.NA, "nan": pd.NA, "<na>": pd.NA})
    if "boundary_type" in out.columns:
        out["boundary_type"] = out["boundary_type"].astype(str).str.strip().str.lower()
        out["boundary_type"] = out["boundary_type"].replace({"": "fold", "nan": "fold", "<na>": "fold"})
    else:
        out["boundary_type"] = "fold"
    extra_boundary_types = sorted(set(out["boundary_type"]) - set(BOUNDARY_TYPES))
    if extra_boundary_types:
        print(
            "[WARN] Unrecognized boundary_type values found and ignored for plotting: "
            + ", ".join(extra_boundary_types)
        )

    bad_source = sorted(set(out["source_type"]) - VALID_SOURCE_TYPES)
    if bad_source:
        raise ValueError(
            f"Column 'source_type' has invalid values: {bad_source}. "
            f"Expected values within {sorted(VALID_SOURCE_TYPES)}."
        )

    bad_pass = sorted(set(out["pass_direction"]) - VALID_PASS_DIRECTIONS)
    if bad_pass:
        raise ValueError(
            f"Column 'pass_direction' has invalid values: {bad_pass}. "
            f"Expected values within {sorted(VALID_PASS_DIRECTIONS)}."
        )

    algebraic = out[out["source_type"] == "algebraic"]
    dynamic = out[out["source_type"] == "dynamic"]

    if (algebraic["pass_direction"] != "none").any():
        bad = sorted(set(algebraic.loc[algebraic["pass_direction"] != "none", "pass_direction"]))
        raise ValueError(
            "Algebraic rows must have pass_direction='none', found: " + ", ".join(bad)
        )

    bad_dyn_pass = dynamic.loc[~dynamic["pass_direction"].isin(["forward", "backward"]), "pass_direction"]
    if not bad_dyn_pass.empty:
        bad = sorted(set(bad_dyn_pass))
        raise ValueError(
            "Dynamic rows must have pass_direction in {'forward','backward'}, found: "
            + ", ".join(bad)
        )

    if algebraic["branch_id"].isna().any():
        n_bad = int(algebraic["branch_id"].isna().sum())
        raise ValueError(
            f"Algebraic rows contain {n_bad} missing 'branch_id' values; cannot style skeleton lines."
        )

    bad_branches = sorted(set(algebraic["branch_id"]) - VALID_BRANCHES)
    if bad_branches:
        raise ValueError(
            f"Algebraic rows have invalid branch_id values: {bad_branches}. "
            f"Expected values within {sorted(VALID_BRANCHES)}."
        )

    return out


def log_row_counts(df: pd.DataFrame) -> None:
    counts = (
        df.groupby(["boundary_type", "source_type", "pass_direction"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["boundary_type", "source_type", "pass_direction"], kind="mergesort")
    )
    print("[INFO] Row counts by boundary_type/source_type/pass_direction:")
    for row in counts.itertuples(index=False):
        print(
            f"  - boundary_type={row.boundary_type}, source_type={row.source_type}, "
            f"pass_direction={row.pass_direction}: {row.rows}"
        )


def select_candidates_by_boundary(
    df: pd.DataFrame,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, object]]]:
    selected_dfs: Dict[str, pd.DataFrame] = {}
    selected_ids: Dict[str, Dict[str, object]] = {}

    for boundary_type in BOUNDARY_TYPES:
        sub = df[df["boundary_type"] == boundary_type].copy()
        if sub.empty:
            print(f"[WARN] No rows found for boundary_type='{boundary_type}'.")
            continue

        candidate_ids = (
            sub[ID_COLUMNS]
            .drop_duplicates()
            .sort_values(ID_COLUMNS, kind="mergesort")
            .reset_index(drop=True)
        )
        if candidate_ids.empty:
            print(f"[WARN] No candidate IDs found for boundary_type='{boundary_type}'.")
            continue

        selected = candidate_ids.iloc[0].to_dict()
        if len(candidate_ids) > 1:
            print(
                "[WARN] boundary_type='"
                + boundary_type
                + f"' has multiple candidate IDs ({len(candidate_ids)}); "
                "using the first in deterministic order."
            )

        mask = pd.Series(True, index=sub.index)
        for col, value in selected.items():
            mask &= sub[col] == value

        selected_df = sub.loc[mask].copy()
        if selected_df.empty:
            print(
                f"[WARN] Selected candidate for boundary_type='{boundary_type}' "
                "produced an empty subset."
            )
            continue

        id_text = ", ".join(f"{k}={v}" for k, v in selected.items())
        print(f"[INFO] Selected candidate row ID ({boundary_type}): {id_text}")
        print(f"[INFO] Rows for selected candidate ({boundary_type}): {len(selected_df)}")

        selected_dfs[boundary_type] = selected_df
        selected_ids[boundary_type] = selected

    if not selected_dfs:
        raise ValueError("No candidate rows found for any supported boundary type.")

    return selected_dfs, selected_ids


def find_latest_input(data_dir: Path) -> Path:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    arrow_files = sorted(
        data_dir.glob("*_hysteresis_table.arrow"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if arrow_files:
        return arrow_files[0]

    csv_files = sorted(
        data_dir.glob("*_hysteresis_table.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if csv_files:
        return csv_files[0]

    raise FileNotFoundError(
        "No hysteresis table files found. Expected files matching "
        "'*_hysteresis_table.arrow' (preferred) or '*_hysteresis_table.csv' "
        f"under: {data_dir}"
    )


def resolve_output_stem(input_path: Path, output_arg: str | None, output_dir_default: Path) -> Path:
    if output_arg is None:
        return output_dir_default / f"{input_path.stem}_panels"

    output_path = Path(output_arg)
    if output_path.suffix:
        output_path = output_path.with_suffix("")
    return output_path


def build_species_colors(species_ids: np.ndarray) -> Dict[int, Tuple[float, float, float, float]]:
    species_sorted = sorted(int(x) for x in species_ids)
    n_species = len(species_sorted)
    base_colors = [mpl.colors.to_rgba(c) for c in COOLWARM_QUAL3_NO_GRAY]
    if n_species <= len(base_colors):
        return {species_id: base_colors[i] for i, species_id in enumerate(species_sorted)}

    print(
        f"[WARN] Requested {n_species} species colors, but only {len(base_colors)} "
        "custom colors provided; using tab20 fallback."
    )
    cmap = plt.get_cmap("tab20", n_species)
    return {species_id: cmap(i) for i, species_id in enumerate(species_sorted)}


def darken_color(
    color: Tuple[float, float, float, float], factor: float = 0.78
) -> Tuple[float, float, float, float]:
    r, g, b, a = color
    return (factor * r, factor * g, factor * b, a)


def add_panel_direction_arrow(
    ax: plt.Axes,
    direction: str,
    label_size: float,
    color: str = "0.2",
    show_label: bool = True,
) -> None:
    if direction == "forward":
        start, end, label = (0.08, 0.12), (0.28, 0.12), "forward"
        label_x = 0.18
    else:
        start, end, label = (0.93, 0.12), (0.73, 0.12), "backward"
        label_x = 0.83

    from matplotlib.patches import FancyArrowPatch
    arrow = FancyArrowPatch(
        posA=start,
        posB=end,
        arrowstyle="-|>,head_length=0.3,head_width=0.15",
        mutation_scale=20,
        shrinkA=0,
        shrinkB=0,
        joinstyle="miter",
        capstyle="projecting",
        edgecolor=color,
        facecolor=color,
        linewidth=1.8,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(arrow)
    if show_label:
        ax.text(
            label_x,
            0.195,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=label_size * 0.9,
            color=color,
        )


def first_finite(series: pd.Series) -> float | None:
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    vals = np.unique(vals)
    return float(vals[0])


def extract_event_markers(df: pd.DataFrame, boundary_type: str) -> Tuple[float | None, float | None]:
    top_delta = None
    bottom_delta = None

    if "scrit" in df.columns:
        top_delta = first_finite(df["scrit"])
    if top_delta is None and "delta_post" in df.columns:
        top_delta = first_finite(df["delta_post"])

    if "delta_event" in df.columns:
        if "hc_event" in df.columns:
            hc_event = df["hc_event"].astype(str).str.strip().str.lower()
            bottom_delta = first_finite(df.loc[hc_event == "invasion", "delta_event"])
        if bottom_delta is None:
            bottom_delta = first_finite(df["delta_event"])

    if top_delta is None:
        print(
            f"[WARN] Could not infer top marker for boundary_type='{boundary_type}' "
            "(need 'scrit' or 'delta_post')."
        )
    else:
        print(
            f"[INFO] Top marker delta ({boundary_type}): {top_delta:.6g}"
        )

    if bottom_delta is None:
        print(
            f"[WARN] Could not infer bottom marker for boundary_type='{boundary_type}' "
            "(need 'delta_event')."
        )
    else:
        print(
            f"[INFO] Bottom marker delta ({boundary_type}): {bottom_delta:.6g}"
        )

    return top_delta, bottom_delta


def plot_skeleton(
    ax: plt.Axes,
    algebraic: pd.DataFrame,
    species_colors: Dict[int, Tuple[float, float, float, float]],
    unstable_min_delta: float | None = None,
) -> None:
    if algebraic.empty:
        print("[WARN] No algebraic rows found; skeleton lines will be omitted.")
        return

    ordered = algebraic.assign(branch_rank=algebraic["branch_id"].map(BRANCH_ORDER)).sort_values(
        ["species_id", "branch_rank", "delta"], kind="mergesort"
    )

    for (branch_id, species_id), group in ordered.groupby(["branch_id", "species_id"], sort=False):
        if unstable_min_delta is not None and BRANCH_LINESTYLES.get(str(branch_id), "-") == ":":
            group = group[group["delta"] >= unstable_min_delta]
            if group.empty:
                continue
        ax.plot(
            group["delta"],
            group["abundance"].clip(lower=Y_FLOOR),
            color=species_colors[int(species_id)],
            linestyle=BRANCH_LINESTYLES[branch_id],
            linewidth=1.6,
            alpha=0.95,
            zorder=2,
        )


def plot_boundary_column(
    ax: plt.Axes,
    boundary_df: pd.DataFrame,
    boundary_type: str,
    species_colors: Dict[int, Tuple[float, float, float, float]],
    forward_color: str = FORWARD_COLOR,
    backward_color: str = BACKWARD_COLOR,
) -> List[float]:
    algebraic = boundary_df[boundary_df["source_type"] == "algebraic"].copy()
    forward_dyn = boundary_df[
        (boundary_df["source_type"] == "dynamic") & (boundary_df["pass_direction"] == "forward")
    ].copy()
    backward_dyn = boundary_df[
        (boundary_df["source_type"] == "dynamic") & (boundary_df["pass_direction"] == "backward")
    ].copy()

    if forward_dyn.empty:
        print(
            f"[WARN] No dynamic rows for boundary_type='{boundary_type}', pass_direction='forward'; "
            "panel will show skeleton only."
        )
    if backward_dyn.empty:
        print(
            f"[WARN] No dynamic rows for boundary_type='{boundary_type}', pass_direction='backward'; "
            "panel will show skeleton only."
        )

    top_marker, bottom_marker = extract_event_markers(boundary_df, boundary_type)

    all_x = pd.to_numeric(boundary_df["delta"], errors="coerce").dropna()
    x_range = float(all_x.max() - all_x.min()) if len(all_x) > 1 else 1.0
    delta_offset = 0.004 * x_range

    unstable_cutoff = bottom_marker if boundary_type == "fold" else None
    plot_skeleton(
        ax,
        algebraic,
        species_colors,
        unstable_min_delta=unstable_cutoff,
    )

    def _thin_by_section(dyn_df):
        """Keep specific 1-based marker indices per section (per species).
        Indices are per-section (each section restarts at 1)."""
        if dyn_df.empty or top_marker is None or bottom_marker is None:
            return dyn_df
        lo = min(bottom_marker, top_marker)
        hi = max(bottom_marker, top_marker)
        keep_indices = {1: {1, 3, 5, 7}, 2: {1, 2, 3}, 3: {1, 3, 5}}
        pieces = []
        for _, grp in dyn_df.groupby("species_id", sort=False):
            grp = grp.sort_values("delta", kind="mergesort")
            sec1 = grp[grp["delta"] < lo]
            sec2 = grp[(grp["delta"] >= lo) & (grp["delta"] <= hi)]
            sec3 = grp[grp["delta"] > hi]
            for sec, sec_key in [(sec1, 1), (sec2, 2), (sec3, 3)]:
                for i, idx in enumerate(sec.index, start=1):
                    if i in keep_indices[sec_key]:
                        pieces.append(idx)
        return dyn_df.loc[pieces] if pieces else dyn_df.iloc[0:0]

    if not forward_dyn.empty:
        forward_dyn = forward_dyn.sort_values(["species_id", "delta"], kind="mergesort")
        forward_dyn = _thin_by_section(forward_dyn)
        if top_marker is not None:
            fwd_before = forward_dyn[forward_dyn["delta"] <= top_marker]
            fwd_after = forward_dyn[forward_dyn["delta"] > top_marker]
        else:
            fwd_before = forward_dyn
            fwd_after = forward_dyn.iloc[0:0]
        for chunk, zo in [(fwd_before, 4), (fwd_after, 3)]:
            if not chunk.empty:
                ax.scatter(
                    chunk["delta"] + delta_offset,
                    chunk["abundance"].clip(lower=Y_FLOOR),
                    marker=FORWARD_MARKER,
                    color=forward_color,
                    s=57,
                    linewidths=0,
                    alpha=1.0,
                    zorder=zo,
                )

    if not backward_dyn.empty:
        backward_dyn = backward_dyn.sort_values(["species_id", "delta"], kind="mergesort")
        backward_dyn = _thin_by_section(backward_dyn)
        if top_marker is not None:
            bwd_before = backward_dyn[backward_dyn["delta"] <= top_marker]
            bwd_after = backward_dyn[backward_dyn["delta"] > top_marker]
        else:
            bwd_before = backward_dyn
            bwd_after = backward_dyn.iloc[0:0]
        for chunk, zo in [(bwd_before, 3), (bwd_after, 4)]:
            if not chunk.empty:
                ax.scatter(
                    chunk["delta"] - delta_offset,
                    chunk["abundance"].clip(lower=Y_FLOOR),
                    marker=BACKWARD_MARKER,
                    color=backward_color,
                    s=57,
                    linewidths=0,
                    alpha=1.0,
                    zorder=zo,
                )

    if top_marker is not None:
        ax.axvline(top_marker, color=forward_color, linestyle="--", linewidth=0.9, zorder=1)
    if bottom_marker is not None:
        ax.axvline(bottom_marker, color=backward_color, linestyle="--", linewidth=0.9, zorder=1)

    markers: List[float] = []
    if top_marker is not None:
        markers.append(float(top_marker))
    if bottom_marker is not None:
        markers.append(float(bottom_marker))
    return markers


def plot_panels(
    boundary_dfs: Dict[str, pd.DataFrame],
    output_stem: Path,
    dpi: int,
    ymax_fold: float | None = None,
) -> Tuple[Path, Path]:
    all_selected = pd.concat(list(boundary_dfs.values()), ignore_index=True)
    species_ids = np.sort(all_selected["species_id"].unique())
    species_colors = build_species_colors(species_ids)
    species_cross_colors = {
        species_id: darken_color(color) for species_id, color in species_colors.items()
    }

    # Panel B (fold): all species black.
    black_rgba = mpl.colors.to_rgba("black")
    species_colors_fold = {sid: black_rgba for sid in species_ids}

    # Panel A (extinction): extinct species black, others gray.
    gray_rgba = mpl.colors.to_rgba("0.60")
    extinct_species_id = None
    if "extinction" in boundary_dfs:
        ext_dyn = boundary_dfs["extinction"]
        ext_fwd = ext_dyn[
            (ext_dyn["source_type"] == "dynamic") & (ext_dyn["pass_direction"] == "forward")
        ]
        if not ext_fwd.empty:
            last_abundances = ext_fwd.groupby("species_id")["abundance"].last()
            extinct_species_id = int(last_abundances.idxmin())
    species_colors_ext = {
        sid: (black_rgba if sid == extinct_species_id else gray_rgba)
        for sid in species_ids
    }

    label_size = mpl.font_manager.FontProperties(
        size=mpl.rcParams["axes.labelsize"]
    ).get_size_in_points() * 1.1 * 1.2
    main_text_size = label_size * 0.85
    title_size = main_text_size * 1.10

    fig = plt.figure(figsize=(5.385, 2.311))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.12)
    ax_ext = fig.add_subplot(outer[0])
    ax_fold = fig.add_subplot(outer[1])
    axes = [ax_ext, ax_fold]
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.18, top=0.93)

    # Panel 1: gradual extinction (forward sweep only).
    ax_ext.set_title(BOUNDARY_TITLES["extinction"], fontsize=title_size)
    if "extinction" not in boundary_dfs:
        print("[WARN] boundary_type='extinction' missing; leaving top panel empty.")
    else:
        ext_df = boundary_dfs["extinction"]
        ext_alg = ext_df[ext_df["source_type"] == "algebraic"].copy()
        if "is_stable" in ext_alg.columns:
            n_before = len(ext_alg)
            ext_alg = ext_alg[ext_alg["is_stable"].map(is_stable_value)].copy()
            print(
                f"[INFO] Extinction algebraic rows kept by stability: "
                f"{len(ext_alg)}/{n_before}"
            )
        else:
            print(
                "[WARN] Column 'is_stable' missing for boundary_type='extinction'; "
                "cannot filter algebraic skeleton by stability."
            )
        ext_branch = ext_alg["branch_id"].astype(str).str.lower()
        ext_alg_forward = ext_alg[ext_branch.str.contains("forward", na=False)].copy()
        if ext_alg_forward.empty and not ext_alg.empty:
            print(
                "[WARN] No forward algebraic rows for extinction; "
                "falling back to all stable extinction algebraic rows."
            )
            ext_alg_forward = ext_alg

        plot_skeleton(ax_ext, ext_alg_forward, species_colors_ext)

        ext_dyn_forward = ext_df[
            (ext_df["source_type"] == "dynamic") & (ext_df["pass_direction"] == "forward")
        ].copy()
        ext_all_x = pd.to_numeric(ext_df["delta"], errors="coerce").dropna()
        ext_x_range = float(ext_all_x.max() - ext_all_x.min()) if len(ext_all_x) > 1 else 1.0
        delta_offset = 0.004 * ext_x_range

        ext_top_marker, ext_bottom_marker = extract_event_markers(ext_df, "extinction")
        ext_marker = ext_top_marker if ext_top_marker is not None else ext_bottom_marker

        if ext_dyn_forward.empty:
            print(
                "[WARN] No dynamic forward rows for boundary_type='extinction'; "
                "top panel will show skeleton only."
            )
        else:
            ext_dyn_forward = ext_dyn_forward.sort_values(["species_id", "delta"], kind="mergesort")
            ext_dyn_forward = ext_dyn_forward.groupby("species_id", sort=False).apply(
                lambda g: g.iloc[::2]
            ).reset_index(drop=True)
            if ext_marker is not None:
                fwd_before = ext_dyn_forward[ext_dyn_forward["delta"] <= ext_marker]
                fwd_after = ext_dyn_forward[ext_dyn_forward["delta"] > ext_marker]
            else:
                fwd_before = ext_dyn_forward
                fwd_after = ext_dyn_forward.iloc[0:0]
            for chunk, zo in [(fwd_before, 4), (fwd_after, 3)]:
                if not chunk.empty:
                    ax_ext.scatter(
                        chunk["delta"] + delta_offset,
                        chunk["abundance"].clip(lower=Y_FLOOR),
                        marker=FORWARD_MARKER,
                        color=FORWARD_COLOR,
                        s=57,
                        linewidths=0,
                        alpha=1.0,
                        zorder=zo,
                    )

        ext_dyn_backward = ext_df[
            (ext_df["source_type"] == "dynamic") & (ext_df["pass_direction"] == "backward")
        ].copy()
        if not ext_dyn_backward.empty:
            ext_dyn_backward = ext_dyn_backward.sort_values(["species_id", "delta"], kind="mergesort")
            ext_dyn_backward = ext_dyn_backward.groupby("species_id", sort=False).apply(
                lambda g: g.iloc[::2]
            ).reset_index(drop=True)
            if ext_marker is not None:
                bwd_before = ext_dyn_backward[ext_dyn_backward["delta"] <= ext_marker]
                bwd_after = ext_dyn_backward[ext_dyn_backward["delta"] > ext_marker]
            else:
                bwd_before = ext_dyn_backward
                bwd_after = ext_dyn_backward.iloc[0:0]
            for chunk, zo in [(bwd_before, 3), (bwd_after, 4)]:
                if not chunk.empty:
                    ax_ext.scatter(
                        chunk["delta"] - delta_offset,
                        chunk["abundance"].clip(lower=Y_FLOOR),
                        marker=BACKWARD_MARKER,
                        color=BACKWARD_COLOR,
                        s=57,
                        linewidths=0,
                        alpha=1.0,
                        zorder=zo,
                    )
        ext_markers: List[float] = []
        if ext_marker is not None:
            ax_ext.axvline(ext_marker, color="black", linestyle="--", linewidth=0.9, zorder=1)
            ext_markers.append(float(ext_marker))

        x_vals = pd.to_numeric(ext_df["delta"], errors="coerce").to_numpy(dtype=float)
        x_vals = x_vals[np.isfinite(x_vals)]
        if ext_markers:
            x_vals = np.concatenate([x_vals, np.array(ext_markers, dtype=float)])
        if x_vals.size > 0:
            x_min = float(np.min(x_vals))
            x_max = float(np.max(x_vals))
            span = x_max - x_min
            pad = 0.015 * span if span > 0 else max(1e-3, 0.015 * max(abs(x_min), 1.0))
            ax_ext.set_xlim(x_min - pad, x_max + pad)

        if ext_marker is not None:
            # Force a labeled marker tick and remove nearby ticks to avoid overlap.
            ticks = pd.to_numeric(pd.Series(ax_ext.get_xticks()), errors="coerce").to_numpy(dtype=float)
            x_left, x_right = ax_ext.get_xlim()
            span = x_right - x_left
            min_sep = 0.10 * span if span > 0 else 1e-6

            kept_ticks: List[float] = []
            for tick in ticks:
                if not np.isfinite(tick):
                    continue
                if tick < x_left or tick > x_right:
                    continue
                if abs(float(tick) - float(ext_marker)) < min_sep:
                    continue
                kept_ticks.append(float(tick))

            custom_ticks = np.array(sorted(set(kept_ticks + [float(ext_marker)])), dtype=float)
            marker_tol = max(1e-10, 1e-7 * max(abs(float(ext_marker)), 1.0))
            custom_labels = [
                r"$\mathbf{\delta_c}$" if abs(float(tick) - float(ext_marker)) <= marker_tol else f"{tick:g}"
                for tick in custom_ticks
            ]
            ax_ext.set_xticks(custom_ticks)
            ax_ext.set_xticklabels(custom_labels)

        # Two direction arrows on gradual-extinction panel.
        add_panel_direction_arrow(ax_ext, "forward", main_text_size, color=FORWARD_COLOR)
        add_panel_direction_arrow(ax_ext, "backward", main_text_size, color=BACKWARD_COLOR)

    # Panel 2: abrupt collapse, showing forward and backward sweeps together.
    ax_fold.set_title(BOUNDARY_TITLES["fold"], fontsize=title_size)
    if "fold" not in boundary_dfs:
        print("[WARN] boundary_type='fold' missing; leaving lower panel empty.")
    else:
        fold_df = boundary_dfs["fold"]
        fold_markers = plot_boundary_column(
            ax_fold,
            fold_df,
            "fold",
            species_colors_fold,
            forward_color=FORWARD_COLOR,
            backward_color=BACKWARD_COLOR,
        )

        x_vals = pd.to_numeric(fold_df["delta"], errors="coerce").to_numpy(dtype=float)
        x_vals = x_vals[np.isfinite(x_vals)]
        if fold_markers:
            x_vals = np.concatenate([x_vals, np.array(fold_markers, dtype=float)])
        if x_vals.size > 0:
            x_min = float(np.min(x_vals))
            x_max = max(float(np.max(x_vals)), 1.2)
            span = x_max - x_min
            pad = 0.015 * span if span > 0 else max(1e-3, 0.015 * max(abs(x_min), 1.0))
            ax_fold.set_xlim(x_min - pad, x_max + pad)

        # Add δ_c tick at the dark-red (forward/top) vertical dashed line in panel B.
        fold_top_marker, _ = extract_event_markers(fold_df, "fold")
        if fold_top_marker is not None:
            ticks = pd.to_numeric(pd.Series(ax_fold.get_xticks()), errors="coerce").to_numpy(dtype=float)
            x_left, x_right = ax_fold.get_xlim()
            span = x_right - x_left
            min_sep = 0.10 * span if span > 0 else 1e-6

            kept_ticks: List[float] = []
            for tick in ticks:
                if not np.isfinite(tick):
                    continue
                if tick < x_left or tick > x_right:
                    continue
                if abs(float(tick) - float(fold_top_marker)) < min_sep:
                    continue
                kept_ticks.append(float(tick))

            if x_left <= 1.0 <= x_right and abs(1.0 - float(fold_top_marker)) >= min_sep:
                kept_ticks.append(1.0)
            if x_left <= 1.2 <= x_right and abs(1.2 - float(fold_top_marker)) >= min_sep:
                kept_ticks.append(1.2)

            fold_custom_ticks = np.array(sorted(set(kept_ticks + [float(fold_top_marker)])), dtype=float)
            fold_marker_tol = max(1e-10, 1e-7 * max(abs(float(fold_top_marker)), 1.0))
            fold_custom_labels = [
                r"$\mathbf{\delta_c}$" if abs(float(tick) - float(fold_top_marker)) <= fold_marker_tol else f"{tick:g}"
                for tick in fold_custom_ticks
            ]
            ax_fold.set_xticks(fold_custom_ticks)
            ax_fold.set_xticklabels(fold_custom_labels)

        fwd_marker_handle = mlines.Line2D([], [], marker=FORWARD_MARKER, color=FORWARD_COLOR,
                                          linestyle="none", markersize=12)
        bwd_marker_handle = mlines.Line2D([], [], marker=BACKWARD_MARKER, color=BACKWARD_COLOR,
                                          linestyle="none", markersize=12)
        stable_line_handle = mlines.Line2D([], [], color="black", linestyle="-", linewidth=1.6)
        unstable_line_handle = mlines.Line2D([], [], color="black", linestyle=":", linewidth=1.6)
        # ax_ext.legend(
        #     handles=[(bwd_marker_handle, fwd_marker_handle), stable_line_handle, unstable_line_handle],
        #     labels=["Integration endpoints", "Stable branch", "Unstable branch"],
        #     handler_map={(bwd_marker_handle, fwd_marker_handle): HandlerTuple(ndivide=2, pad=0)},
        #     fontsize=main_text_size,
        #     loc="upper left",
        #     frameon=False,
        #     borderpad=0.5,
        #     handlelength=1.8,
        #     handletextpad=0.3,
        # )

    panel_labels = ["a", "b"]
    for idx, ax in enumerate(axes):
        ax.text(
            -0.02,
            1.05,
            panel_labels[idx],
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=main_text_size,
            fontweight="bold",
        )

    ax_ext.set_ylabel(r"Equilibrium ($\mathbf{x}^*$)", fontsize=title_size)

    for ax in axes:
        ax.grid(False)
        ax.tick_params(labelsize=main_text_size * 0.8 * 1.15)

    # Per-panel tight ylim
    def _tight_ytop(sub_df, pad_frac=0.05):
        yv = pd.to_numeric(sub_df["abundance"], errors="coerce").to_numpy(dtype=float)
        yv = yv[np.isfinite(yv)]
        yv = np.clip(yv, Y_FLOOR, None)
        if yv.size == 0:
            return 2.0
        ym = float(np.max(yv))
        return ym + pad_frac * (ym - Y_FLOOR)

    if "extinction" in boundary_dfs:
        ext_ytop = _tight_ytop(boundary_dfs["extinction"])
    else:
        ext_ytop = 2.0
    ax_ext.set_ylim(Y_FLOOR, ext_ytop)
    ax_ext.set_yticks([t for t in [0, 1, 2] if t <= ext_ytop])

    if "fold" in boundary_dfs:
        fold_ytop = _tight_ytop(boundary_dfs["fold"])
    else:
        fold_ytop = 2.0
    ax_fold.set_ylim(Y_FLOOR, fold_ytop)
    ax_fold.set_yticks([t for t in [0, 1, 2] if t <= fold_ytop])

    # X-axis styling: ticks shown on both panels, shared label centered via supxlabel.
    ax_ext.tick_params(axis="x", bottom=True, labelbottom=True, pad=1)
    ax_fold.tick_params(axis="x", bottom=True, labelbottom=True, pad=1)
    # Enlarge δ_c tick labels AFTER tick_params so they aren't overridden.
    delta_c_size = main_text_size * 0.8 * 1.15 * 1.15 * 1.15 * 1.15
    # x-axis label: "Perturbation magnitude" at title_size, δ scaled up within mathtext
    _delta_rel_pct = int(round(100 * delta_c_size / title_size))
    # Two-part label: text at title_size, δ symbol at delta_c_size, visually centered
    # Single centered label; use delta_c_size for the whole string (δ dominates sizing)
    fig.text(0.555, 0.02,
             r"Perturbation magnitude ($\delta$)",
             ha="center", va="center", fontsize=title_size)

    # Direction arrows for abrupt-collapse panel (no labels — already shown on panel A).
    add_panel_direction_arrow(ax_fold, "forward", main_text_size, color=FORWARD_COLOR, show_label=False)
    add_panel_direction_arrow(ax_fold, "backward", main_text_size, color=BACKWARD_COLOR, show_label=False)

    for ax in axes:
        for lbl in ax.get_xticklabels():
            if r"$\mathbf{\delta_c}$" in lbl.get_text():
                lbl.set_fontsize(delta_c_size)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")

    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate hysteresis panel figures from a table produced by "
            "build_hysteresis_table.jl."
        )
    )
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    data_dir_default = repo_root / "data" / "figure_inputs" / "fig1_hysteresis"
    output_dir_default = repo_root / "pdffiles" / "main"

    parser.add_argument(
        "--input",
        default=str(data_dir_default / "sys_12_alphaidx_5_ray_60_row_460_hysteresis_table.arrow"),
        help=(
            "Path to .arrow (preferred) or .csv table. If omitted, defaults to "
            "sys_12_alphaidx_5_ray_60_row_460_hysteresis_table.arrow in "
            "data/figure_inputs/fig1_hysteresis/."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=str(data_dir_default),
        help=f"Directory searched when --input is omitted (default: {data_dir_default}).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output path stem (without extension). "
            f"Default: {output_dir_default}/<input_stem>_panels"
        ),
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution (default: 300).")
    parser.add_argument(
        "--ymax-fold",
        type=float,
        default=None,
        help="Override the top y-limit for panel B (fold) only. "
        "Unlinks panel B from panel A's shared y-axis.",
    )
    args = parser.parse_args()

    if args.dpi <= 0:
        raise SystemExit("ERROR: --dpi must be a positive integer.")

    data_dir = Path(args.data_dir)
    input_path = Path(args.input)

    output_stem = resolve_output_stem(input_path, args.output, output_dir_default)

    df = load_table(input_path)
    df = validate_schema(df)
    log_row_counts(df)

    selected_by_boundary, _ = select_candidates_by_boundary(df)
    pdf_path = plot_panels(selected_by_boundary, output_stem, args.dpi, ymax_fold=args.ymax_fold)

    print(f"[INFO] Wrote output PDF: {pdf_path}")


if __name__ == "__main__":
    main()
