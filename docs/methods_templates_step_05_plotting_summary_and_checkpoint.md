# Templates Step — Part 5: Plotting, Summary JSON, and Checkpoint Completion

Scope: this document describes the plotting artifacts produced by the templates stage and how the stage finishes (summary + checkpoint).

Primary code paths:
- `axon_reconstructor.pipeline.templates.processing.process_unit_list(...)` (per-unit plots)
- `axon_reconstructor.pipeline.templates.runner.extract_and_merge_templates(...)` (grid PDF + summary + checkpoint)

Plotting implementation lives in:
- `axon_reconstructor.pipeline.templates.plotting`

---

## 1. Per-unit quick-look plots for `merged_contributing`

If enabled (`TemplateExtractInputs.plot_merged_contributing_footprints_linear_and_log=True`) and the merged-contributing template exists, templates writes quick-look plots:

### 1.1 Footprint PTP maps (linear + log)

Directory:
- `<well>/templates_outputs/footprints/`

Per unit:
- `unit_<uid>_merged_contributing_footprint_ptp_linear.png`
- `unit_<uid>_merged_contributing_footprint_ptp_log.png`

Footprint definition:
- PTP amplitude per channel: `np.ptp(template, axis=0)`

### 1.2 Combined SVG panels (template overlay + footprint)

Directory:
- `<well>/templates_outputs/svgs/`

Per unit:
- `unit_<uid>_merged_contributing_template_footprint_linear.svg`
- `unit_<uid>_merged_contributing_template_footprint_log.svg`

These are intended as compact per-unit “at a glance” QC summaries.

---

## 2. Main grid PDF (`templates_grid.pdf`)

If `TemplateExtractInputs.plot_templates_grid_pdf=True`, templates writes:

- `<well>/templates_outputs/templates_grid.pdf`

Grid entries come from the list returned by `process_unit_list`.

Important behavior:

- The grid is intended to represent the **merged-contributing** template per unit (not concat-only), because merged-contributing is the canonical reconstruction handoff.
- The plotting window uses:
  - sampling rate from analyzers
  - and `ms_before/ms_after` hints from `<well>/waveforms_outputs/waveform_extraction_params.json` when available.

The runner writes the PDF only if it doesn’t exist yet, unless `force_restart=True`.

---

## 3. Optional: per-unit per-source overlay PDFs

If `TemplateExtractInputs.plot_multi_source_templates_pdf=True`, templates creates:

Directory:
- `<well>/templates_outputs/unit_segment_grids/`

Per unit:
- `unit_<uid>_templates.pdf`
  - overlays template waveforms for the unit across sources (concat + segments)
- `unit_<uid>_footprints.pdf` (if enabled in the call site)
  - compares per-source footprints

Scientific purpose:
- diagnose when concat channel intersection dropped important electrodes that appear in some segments
- confirm segment-specific template differences

---

## 4. Optional: topographical footprint plots (3D PTP height map)

If `TemplateExtractInputs.plot_topo_unit_footprints=True` and `save_full_channels_templates=True`, templates writes:

- `<well>/templates_outputs/topo_unit_footprints/unit_<uid>.png`

This plot is generated from the dense `full_template.npy` so that:
- non-contributing electrodes can be treated as zero
- the plot can use a deterministic full-chip geometry when electrode ids indicate Maxwell scheme

---

## 5. Optional: propagation plots (ordered waveforms)

If `TemplateExtractInputs.plot_propagation_plots=True`, templates writes per-unit propagation plots under:

- `<well>/templates_outputs/propagation_plots/`

Implementation detail:
- the call site still uses a legacy function name `write_unit_propagation_plots_pdf`, but current plotting tends to write PNG(s) into the provided directory.

Inputs include:
- `merged_contributing` template
- plotting parameters:
  - `propagation_top_channels`
  - `propagation_n_waveforms`
  - `propagation_channels_per_panel`
  - `propagation_channel_overlap`

It also passes an `ap_timings_json_path` hint:
- `<well>/templates_outputs/merged_units/unit_<uid>/ap_timings.json`

This file may be produced by downstream stages; propagation plotting treats it as optional.

---

## 6. Summary JSON write

After unit processing completes, the runner writes:

- `<well>/templates_outputs/templates_summary.json`

This JSON contains:

- global metadata (inputs, sources, output dirs)
- curation provenance (whether spikesorting-stage curated units derived from `qm_unfiltered.xlsx` were applied)
- per-unit entries appended during persistence

This summary is intended to support:
- replotting
- debugging
- downstream bookkeeping and provenance tracking

---

## 7. Checkpoint completion

Finally, templates saves a checkpoint update:

- Stage: `ProcessingStage.ANALYZER_COMPLETE`
- Extra fields include:
  - `templates_out_dir`
  - `extracted_templates_dir`
  - `merged_units_dir`
  - `templates_summary_json`
  - `templates_grid_pdf` (if enabled)

Then it returns `TemplateExtractOutputs` with:

- `well_out_dir`
- `templates_out_dir`
- `extracted_templates_dir`
- `merged_units_dir`
- `summary_json`
- `templates_grid_pdf` (current filename, if plotting enabled)
- `multi_source_templates_dir` (`unit_segment_grids/` if enabled)

---

## End of templates stage documentation series

At this point, the templates stage has produced:

- per-unit templates from each waveforms analyzer source
- per-unit merged-contributing templates on a contributing channel set
- reconstruction-friendly persisted arrays and optional dense full-channel templates
- QC plots and a JSON summary

If you want, the next natural doc series is the reconstruction stage: what it consumes from `merged_units/` (or `full_channels_templates/`) and what it computes.
