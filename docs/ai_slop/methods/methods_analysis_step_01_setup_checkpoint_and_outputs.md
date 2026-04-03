# Analysis Step — Part 1: Setup, Dedicated Checkpointing, and Output Layout

Scope: this document covers analysis stage setup and output layout.

Primary code path:
- `axon_reconstructor.pipeline.analysis.analyze_units(inputs=...)`

Implementation module:
- `axon_reconstructor.pipeline.analysis.runner`

---

## 0. Stage intent

The analysis stage is a post-hoc “join” stage:

- it reads artifacts produced by earlier stages
- it renders inspection-friendly summary outputs
- it does not re-run heavy compute (sorting, waveforms, reconstruction)

Current focus:
- per-unit summary grids combining templates + reconstruction panels.

---

## 1. Per-well output directory

Analysis uses the same per-well output directory convention as other stages:

- `well_out_dir = _compute_mea_analysis_output_dir(output_root=inputs.mea_output_root, data_file=inputs.h5_path, well=inputs.stream_id)`

---

## 2. Logging

Analysis writes into the per-well pipeline log file:

- `compute_pipeline_log_file(well_out_dir, data_file, stream_id)`
- stage logger name: `{logger_name_prefix}.{stream_id}.analysis`

---

## 3. Output directory layout

Outputs live under:

- `<well_out_dir>/stg6_analysis_outputs/`

Key artifacts:

- `analysis_summary.json`
- `by_unit/unit_<id>/unit_summary_grid.png`
- `by_unit/unit_<id>/unit_summary_grid.pdf`

---

## 4. Dedicated checkpoint file

Analysis uses a dedicated checkpoint file separate from the main pipeline checkpoint:

- it derives from the main checkpoint name but replaces the suffix with `_analysis_checkpoint.json`

This is mainly used to record start/completion and to support `force_restart` behavior.
