# Hysteresis Figure Pipeline (new_code)

This folder contains a port of the legacy hysteresis workflow:

- `build_hysteresis_table.jl` builds a tidy table (`.arrow` + `.csv`)
- `plot_hysteresis_panels.py` plots the two-panel hysteresis figure (`.pdf` + `.png`)

## Directory Layout

- Data output: `new_code/figures/data/`
- Figure output: `new_code/figures/pdffiles/`

Data files are written **flat** inside `figures/data` (no `model_*` or `n_*` subfolders).

## 1) Build Table (Legacy pass-file source)

Run from any directory (absolute paths shown):

```bash
julia --startup-file=no /Users/pablolechon/Desktop/tipping_points_hoi/new_code/figures/build_hysteresis_table.jl \
  --config /Users/pablolechon/Desktop/pert_hoi/code/cluster_code/config_mini.json \
  --results-root /Users/pablolechon/Desktop/pert_hoi/code/cluster_code/results_boundary_dynamics/model_d1ee88741ddd31386ccf912d29792176b1759f6c \
  --bank-root /Users/pablolechon/Desktop/pert_hoi/code/cluster_code/parameters_bank/model_d1ee88741ddd31386ccf912d29792176b1759f6c \
  --output-dir /Users/pablolechon/Desktop/tipping_points_hoi/new_code/figures/data
```

Expected output in `figures/data`:

- `sys_*_hysteresis_table.arrow`
- `sys_*_hysteresis_table.csv`

## 2) Plot Panels

### A. Explicit input

```bash
python /Users/pablolechon/Desktop/tipping_points_hoi/new_code/figures/plot_hysteresis_panels.py \
  --input /Users/pablolechon/Desktop/tipping_points_hoi/new_code/figures/data/sys_8_alphaidx_3_ray_7_row_27_hysteresis_table.arrow
```

### B. Auto input (recommended)

Automatically selects newest `*_hysteresis_table.arrow` from `--data-dir` (fallback: newest CSV):

```bash
python /Users/pablolechon/Desktop/tipping_points_hoi/new_code/figures/plot_hysteresis_panels.py
```

Expected figure outputs in `figures/pdffiles`:

- `<input_stem>_panels.pdf`
- `<input_stem>_panels.png`

## CLI Notes

### `build_hysteresis_table.jl`

- `--results-root` and `--bank-root` must be provided together.
- `--output-dir` defaults to `new_code/figures/data`.
- Compatibility mode remains available via `--config` and optional `--model-id`.

### `plot_hysteresis_panels.py`

- `--input` is optional.
- `--data-dir` defaults to `new_code/figures/data`.
- Default output stem is `new_code/figures/pdffiles/<input_stem>_panels`.
