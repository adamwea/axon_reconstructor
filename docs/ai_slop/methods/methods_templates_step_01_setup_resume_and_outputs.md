# Templates Step — Part 1: Setup, Resume Logic, and Output Layout (Runner)

Scope: this document covers the **first chunk** of the templates stage as implemented in the runner. It ends right before we load waveforms-stage analyzers. The goal is to make the run orchestration, resume behavior, and on-disk output layout completely explicit.

Primary code path:
- `axon_reconstructor.pipeline.templates.runner.extract_and_merge_templates(inputs=...)`

Related modules (used in this part):
- `axon_reconstructor.pipeline.templates.runner` (stage runner: directories, resume, checkpoint)
- `axon_reconstructor.pipeline.templates.utils` (checkpoint file naming, JSON helpers)
- `axon_reconstructor.pipeline.checkpointing` (checkpoint read/write)
- `axon_reconstructor.pipeline.pipeline_driver` (compute per-well output directory)
- `axon_reconstructor.pipeline.pipeline_logging` (per-well pipeline log)

---

## 0. Stage intent (scientific contract)

The templates stage is deliberately **not** spike detection/sorting. It is a *template extraction + merging + QC* stage.

Key contract points (enforced by design and docs; not all are “hard” runtime checks):

- **Templates are sourced from waveforms-stage `SortingAnalyzer` artifacts** (specifically the `templates` extension). We do not re-run sorting.
- **Spike-level exclusions belong to the waveforms stage**. Templates does not re-apply per-spike rejection files (e.g., deprecated `wf_exclusions.npz`).
- **“Curation” here means unit selection only**: by default, templates runs only on curated units derived from spikesorting-stage quality metrics (`<well>/stg2_spikesorting_outputs/qm_unfiltered.xlsx`).

---

## 1. Entry point and per-well output directory

1. Call `extract_and_merge_templates(inputs, logger_name_prefix="axon_reconstructor")`.

2. Compute the per-well output directory (`well_out_dir`) using the shared pipeline convention:

   - `well_out_dir = _compute_mea_analysis_output_dir(output_root=inputs.mea_output_root, data_file=inputs.h5_path, well=inputs.stream_id)`

This ensures the templates artifacts land alongside other stage outputs under the well directory.

---

## 2. Logging

1. Compute a per-well pipeline log file path:

   - `log_file = compute_pipeline_log_file(well_out_dir, data_file, stream_id)`

2. Create a stage-specific logger:

   - Logger name: `{logger_name_prefix}.{stream_id}.templates`
   - Logging is configured to write to the per-well pipeline log file.

---

## 3. Output directory structure (declared up front)

The templates stage writes under:

- `<well_out_dir>/stg4_templates_outputs/`

Key paths (created before processing):

- Data outputs
  - `templates_out_dir = <well>/stg4_templates_outputs/`
  - `extracted_templates_dir = <well>/stg4_templates_outputs/extracted_templates/`
    - Per-source template arrays and per-unit meta JSON.
  - `merged_units_dir = <well>/stg4_templates_outputs/merged_units/`
    - Per-unit merged outputs (the canonical handoff to reconstruction).

- Plot outputs (top-level by plot type)
  - `<well>/stg4_templates_outputs/footprints/`
  - `<well>/stg4_templates_outputs/svgs/`
  - `<well>/stg4_templates_outputs/full_chip_maps/`
  - `<well>/stg4_templates_outputs/topo_unit_footprints/` (optional)
  - `<well>/stg4_templates_outputs/propagation_plots/` (optional)
  - `<well>/stg4_templates_outputs/unit_segment_grids/` (optional)
  - `<well>/stg4_templates_outputs/axon_velocity_outputs/` (optional, requires extra deps)

- Summaries
  - `<well>/stg4_templates_outputs/templates_summary.json`

- Main grid PDF
  - `<well>/stg4_templates_outputs/templates_grid.pdf` (if `inputs.plot_templates_grid_pdf=True`)

Notes:
- Earlier versions of the pipeline wrote plots into a dedicated `merged_unit_plots/` directory. Current code writes plots directly under `stg4_templates_outputs/*`.

---

## 4. Dedicated checkpointing for templates

Templates uses a **dedicated checkpoint file**, separate from the main pipeline checkpoint.

- Path computed by `_compute_templates_checkpoint_file(...)` in `axon_reconstructor.pipeline.templates.utils`.

Naming rule:
- If the main checkpoint ends with `_checkpoint.json`, we replace that suffix with `_templates_checkpoint.json`.
- Otherwise we append `_templates_checkpoint.json` to the stem.

Then we load the checkpoint state:
- `ckpt = load_checkpoint(checkpoint_file=ckpt_file, force_restart=inputs.force_restart, output_dir=..., file_path=..., stream_id=...)`

This checkpoint is primarily used to record stage start/completion and assist debugging; the *actual* resume shortcut is filesystem-based (see below).

---

## 5. Resume shortcut (filesystem-based)

Templates has a conservative “resume if outputs exist” behavior.

### 5.1 Canonical outputs it expects

Resume can happen when all of the following are true:

- `inputs.force_restart` is **False**
- `extracted_templates_dir.exists()`
- `summary_json.exists()`
- If grid plotting is enabled, `templates_grid.pdf` exists

### 5.3 What resume returns

If resume succeeds, `extract_and_merge_templates` returns a `TemplateExtractOutputs` pointing at existing outputs:

- `templates_out_dir`
- `extracted_templates_dir`
- `merged_units_dir`
- `summary_json`
- `templates_grid_pdf` (if grid plotting is enabled)
- `multi_source_templates_dir` (if multi-source plotting is enabled)

No recomputation is performed.

---

## 6. Force restart behavior (output hygiene)

When `inputs.force_restart=True`, templates will recompute outputs even if they already exist.

---

## 7. Stage-start checkpoint write

If we are *not* resuming, the runner saves a checkpoint update early:

- Stage: `ProcessingStage.ANALYZER`
- Extra fields: at least `templates_out_dir`

This gives a durable record that templates started and where it intended to write outputs.

---

## 8. Directory creation

The runner ensures the following directories exist before processing units:

- `stg4_templates_outputs/`
- `stg4_templates_outputs/extracted_templates/`
- `stg4_templates_outputs/merged_units/`
- `stg4_templates_outputs/footprints/`
- `stg4_templates_outputs/svgs/`
- `stg4_templates_outputs/full_chip_maps/`
- `stg4_templates_outputs/axon_velocity_outputs/`

Optional directories (created only if enabled):

- `stg4_templates_outputs/full_channels_templates/` if `inputs.save_full_channels_templates=True`
- `stg4_templates_outputs/topo_unit_footprints/` if `inputs.plot_topo_unit_footprints=True`
- `stg4_templates_outputs/propagation_plots/` if `inputs.plot_propagation_plots=True`
- `stg4_templates_outputs/unit_segment_grids/` if `inputs.plot_multi_source_templates_pdf=True`

---

## End of Part 1

At this point, the runner is ready to:

- load waveforms-stage analyzers (concat and/or per-segment)
- select which units to process (spikesorting-stage curation by default)

Those behaviors are covered in **Part 2**.
