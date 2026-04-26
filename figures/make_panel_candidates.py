#!/usr/bin/env python3
"""Batch-generate hysteresis panel candidate figures.

For each (n, sys_id) pair found in --results-root, calls
build_hysteresis_table.jl (with a restricted per-system temp dir) and then
hysteresis_panels.py.  Outputs accumulate in:
  figures/data/panel_candidates/   -- Arrow/CSV tables
  figures/panel_candidates/        -- PDF/PNG panels
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple


class Candidate(NamedTuple):
    n: int
    sys_id: int


def discover_candidates(
    results_root: Path,
    n_values: Optional[List[int]],
) -> List[Candidate]:
    """Return sorted (n, sys_id) pairs that have a pass4 fold-backtrack file."""
    found: List[Candidate] = []
    for n_dir in sorted(results_root.iterdir()):
        if not n_dir.is_dir():
            continue
        m = re.match(r"^n_(\d+)$", n_dir.name)
        if not m:
            continue
        n = int(m.group(1))
        if n_values is not None and n not in n_values:
            continue
        for p in sorted(n_dir.glob("sys_*_pass4_fold_backtrack.arrow")):
            m2 = re.match(r"^sys_(\d+)_pass4_fold_backtrack\.arrow$", p.name)
            if m2:
                found.append(Candidate(n=n, sys_id=int(m2.group(1))))
    return found


def make_restricted_results_dir(results_root: Path, cand: Candidate) -> Path:
    """Create a temp dir with only this sys_id's files symlinked, so that
    build_hysteresis_table.jl's internal select_candidate sees only one system."""
    tmpdir = Path(
        tempfile.mkdtemp(prefix=f"panel_cand_n{cand.n}_sys{cand.sys_id}_")
    )
    n_subdir = tmpdir / f"n_{cand.n}"
    n_subdir.mkdir()
    src_dir = results_root / f"n_{cand.n}"
    for src_file in sorted(src_dir.glob(f"sys_{cand.sys_id}_*")):
        os.symlink(src_file.resolve(), n_subdir / src_file.name)
    return tmpdir


def find_newest_arrow(data_dir: Path) -> Optional[Path]:
    arrows = sorted(
        data_dir.glob("*_hysteresis_table.arrow"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return arrows[0] if arrows else None


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Batch-generate hysteresis panel candidate figures."
    )
    parser.add_argument("--config", required=True, help="Pipeline config JSON path.")
    parser.add_argument(
        "--results-root", required=True, help="Legacy results model root."
    )
    parser.add_argument(
        "--bank-root", required=True, help="Legacy bank model root."
    )
    parser.add_argument("--model-id", default=None, help="Model ID label.")
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=10,
        help="Max panels to generate (default: 10).",
    )
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=None,
        help="Restrict to specific n values.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for PDF/PNG panels "
        "(default: <script_dir>/panel_candidates).",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG DPI (default: 300).")
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    bank_root = Path(args.bank_root).resolve()
    config_path = Path(args.config).resolve()

    data_dir = script_dir / "data" / "panel_candidates"
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else script_dir / "panel_candidates"
    )

    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    build_script = script_dir / "build_hysteresis_table.jl"
    plot_script = script_dir / "hysteresis_panels.py"

    candidates = discover_candidates(results_root, args.n_values)
    total = min(len(candidates), args.max_candidates)
    print(
        f"[make_panel_candidates] Discovered {len(candidates)} candidate sys"
        f" (n_values={args.n_values}); will attempt {total}."
    )

    generated: List[Tuple[Candidate, Path, Path]] = []
    failed: List[Candidate] = []

    for i, cand in enumerate(candidates[:args.max_candidates]):
        print(
            f"\n{'='*60}\n"
            f"Candidate {i+1}/{total}: n={cand.n}  sys_id={cand.sys_id}\n"
            f"{'='*60}"
        )

        tmpdir = make_restricted_results_dir(results_root, cand)
        try:
            # Step 1: build hysteresis table
            cmd_build: List[str] = [
                "julia",
                "--startup-file=no",
                str(build_script),
                "--config",
                str(config_path),
                "--results-root",
                str(tmpdir),
                "--bank-root",
                str(bank_root),
                "--output-dir",
                str(data_dir),
            ]
            if args.model_id:
                cmd_build += ["--model-id", args.model_id]

            print(f"[build] {' '.join(cmd_build)}")
            build_result = subprocess.run(cmd_build)
            if build_result.returncode != 0:
                print(
                    f"[WARN] build_hysteresis_table.jl exited {build_result.returncode}"
                    f" for n={cand.n} sys={cand.sys_id}; skipping."
                )
                failed.append(cand)
                continue

            # Step 2: find the arrow just produced
            arrow_path = find_newest_arrow(data_dir)
            if arrow_path is None:
                print(
                    f"[WARN] No *_hysteresis_table.arrow found in {data_dir}"
                    " after build; skipping."
                )
                failed.append(cand)
                continue

            # Step 3: plot panels
            output_stem = output_dir / (arrow_path.stem + "_panels")
            cmd_plot: List[str] = [
                sys.executable,
                str(plot_script),
                "--input",
                str(arrow_path),
                "--output",
                str(output_stem),
                "--dpi",
                str(args.dpi),
            ]
            print(f"[plot] {' '.join(cmd_plot)}")
            plot_result = subprocess.run(cmd_plot)
            if plot_result.returncode != 0:
                print(
                    f"[WARN] hysteresis_panels.py exited {plot_result.returncode}"
                    f" for {arrow_path.name}; skipping."
                )
                failed.append(cand)
                continue

            pdf_path = output_stem.with_suffix(".pdf")
            png_path = output_stem.with_suffix(".png")
            print(f"[OK] n={cand.n} sys={cand.sys_id} -> {pdf_path.name}")
            generated.append((cand, pdf_path, png_path))

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\n{'='*60}")
    print(f"Summary: {len(generated)} generated, {len(failed)} failed/skipped.")
    print(f"Output directory: {output_dir}")
    if generated:
        print("Generated panels:")
        for cand, pdf, png in generated:
            print(f"  n={cand.n} sys={cand.sys_id}: {pdf.name}")
    if failed:
        print("Failed/skipped:")
        for cand in failed:
            print(f"  n={cand.n} sys={cand.sys_id}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
