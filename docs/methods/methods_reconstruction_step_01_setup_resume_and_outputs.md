# Reconstruction Step — Part 1: Setup, Resume Logic, and Output Layout (Runner)

Scope: this document covers the first portion of the reconstruction stage as implemented in the runner. It makes the run orchestration, resume behavior, and on-disk output layout explicit.

Primary code path:
- `axon_reconstructor.pipeline.reconstruction.reconstruct_from_templates(inputs=...)`

Implementation module:
- `axon_reconstructor.pipeline.reconstruction.runner`

Related modules:
- `axon_reconstructor.pipeline.checkpointing` (checkpoint read/write)
- `axon_reconstructor.pipeline.pipeline_driver` (compute per-well output directory)
- `axon_reconstructor.pipeline.pipeline_logging` (per-well pipeline log)

---

## 0. Stage intent (scientific contract)

Reconstruction runs axon morphology + velocity estimation using `axon_velocity`.

Key contract points:

- **Reconstruction consumes templates outputs**. It does not re-run sorting or waveform extraction.
- **axon_velocity expects a dense full-channel template**. Therefore, this stage is designed to consume the templates-stage *full-channel* template artifacts (see Part 2).
- Outputs are written under `<well>/stg5_reconstruction_outputs/`.

---

## 1. Entry point and per-well output directory

1. Call `reconstruct_from_templates(inputs=inputs, logger_name_prefix="axon_reconstructor")`.

2. Compute the per-well output directory (`well_out_dir`) using the shared pipeline convention:

- `well_out_dir = _compute_mea_analysis_output_dir(output_root=inputs.mea_output_root, data_file=inputs.h5_path, well=inputs.stream_id)`

This keeps reconstruction artifacts co-located with the other stage outputs for the same dataset and well.

---

## 2. Logging

The runner writes a per-well pipeline log:

- `log_file = compute_pipeline_log_file(well_out_dir, data_file, stream_id)`

Logger name:
- `{logger_name_prefix}.{stream_id}.reconstruction`

---

## 3. Output directory structure

Reconstruction writes under:

- `<well>/stg5_reconstruction_outputs/`

Key paths:

- `reconstruction_out_dir = <well>/stg5_reconstruction_outputs/`
- `by_unit_dir = <well>/stg5_reconstruction_outputs/by_unit/`
  - `unit_<id>/branches.json`
  - `unit_<id>/heuristics.json`
  - plus per-unit PDFs (if enabled)

Summary:

- `summary_json = <well>/stg5_reconstruction_outputs/reconstruction_summary.json`

Optional:

- `all_units_overview_pdf = <well>/stg5_reconstruction_outputs/all_units_morphology.pdf`

---

## 4. Dedicated checkpointing for reconstruction

Reconstruction maintains its own checkpoint file alongside other stage checkpoints.

Naming rule:

- If the main checkpoint ends with `_checkpoint.json`, replace that suffix with `_reconstruction_checkpoint.json`.
- Otherwise append `_reconstruction_checkpoint.json`.

The runner loads the checkpoint via:

- `load_checkpoint(checkpoint_file=..., force_restart=inputs.force_restart, ...)`

---

## 5. Resume shortcut (filesystem-based)

If `inputs.force_restart=False` and the following exist:

- `reconstruction_summary.json`
- `by_unit/`
- `all_units_morphology.pdf` (only if `inputs.write_all_units_overview_pdf=True`)

then the runner returns immediately without recomputing.

---

## 6. Stage-start checkpoint write

If we are not resuming, the runner writes a checkpoint update early:

- stage: `ProcessingStage.ANALYZER`
- extra fields: `reconstruction_out_dir`

---

## End of Part 1

At this point, reconstruction has:

- established `well_out_dir`
- prepared output paths
- decided whether it can resume

Next, the stage locates templates outputs and decides which unit ids to process. That is covered in **Part 2**.
