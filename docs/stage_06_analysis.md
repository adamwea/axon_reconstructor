# Stage 06 — Analysis

This stage generates inspection-friendly summary artifacts that combine outputs from earlier stages.

Current focus: **per-unit summary grids** that montage key panels (templates + reconstruction plots) into a single PNG/PDF per unit.

## Primary APIs

- `axon_reconstructor.pipeline.analysis.AnalysisInputs`
- `axon_reconstructor.pipeline.analysis.analyze_units(inputs=...)`

## Inputs

- `h5_path`: Maxwell `.raw.h5`
- `stream_id`: well/stream id
- `mea_output_root`: output root used for earlier stages (so the per-well directory can be derived)

Optional:

- `unit_ids`: explicit list of unit ids to render
- `unit_limit`: cap how many units to render
- `prefer_curated_waveforms_panels`: if `True`, prefer curated waveforms panels when both exist
- `force_restart`: overwrite existing grids

## What it reads

`analyze_units(...)` is a “join” stage: it looks for artifacts produced earlier. Typical inputs include:

- Templates plots and summary images under `<well_out_dir>/templates_outputs/...`
- Reconstruction plots under `<well_out_dir>/reconstruction_outputs/...`
- (Optionally) waveforms grids when available

Missing inputs are not fatal: the grid renderer will place placeholders for missing panels.

## Outputs

Under the per-well output directory:

- `<well_out_dir>/analysis_outputs/`
  - `analysis_summary.json` (run summary)
  - `by_unit/<unit_id>/unit_summary_grid.png`
  - `by_unit/<unit_id>/unit_summary_grid.pdf`

## Notes

- Rendering uses matplotlib when available; otherwise it falls back to a Pillow montage.
- SVG panels are supported when `cairosvg` is installed (otherwise they render as placeholders).

## Detailed stepwise docs

- Step 01: [methods/methods_analysis_step_01_setup_checkpoint_and_outputs.md](methods/methods_analysis_step_01_setup_checkpoint_and_outputs.md)
- Step 02: [methods/methods_analysis_step_02_unit_discovery_and_panel_contract.md](methods/methods_analysis_step_02_unit_discovery_and_panel_contract.md)
- Step 03: [methods/methods_analysis_step_03_rendering_backend_and_dependencies.md](methods/methods_analysis_step_03_rendering_backend_and_dependencies.md)
- Step 04: [methods/methods_analysis_step_04_resume_overwrite_and_summary_json.md](methods/methods_analysis_step_04_resume_overwrite_and_summary_json.md)
- Step 05: [methods/methods_analysis_step_05_extending_panels_and_custom_runs.md](methods/methods_analysis_step_05_extending_panels_and_custom_runs.md)
