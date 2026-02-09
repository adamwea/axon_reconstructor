# Analysis Step — Part 4: Resume / Overwrite Semantics and `analysis_summary.json`

Scope: this document covers how analysis decides whether to re-render and what it records in the run summary JSON.

Primary code path:
- `axon_reconstructor.pipeline.analysis.runner.analyze_units(...)`

---

## 1. Force restart behavior

Analysis uses `inputs.force_restart` to control overwrite behavior.

At the per-unit grid level:

- if `force_restart=False` and both `unit_summary_grid.png` and `unit_summary_grid.pdf` exist, that unit’s rendering is skipped.

This makes repeated runs cheap.

---

## 2. `analysis_summary.json`

Analysis writes a structured JSON summary at:

- `<well>/analysis_outputs/analysis_summary.json`

It contains:

- input metadata (`h5_path`, `stream_id`, `well_out_dir`)
- output roots
- a `units` list with one entry per unit id, including:
  - unit id
  - key input paths
  - render outputs (`grid_png`, `grid_pdf`)
  - status (`ok` or `error`) and error message/details

This file is designed for downstream automation and quick “what happened?” inspection.
