# Waveforms Step — Part 5: Curation, Plotting, and Checkpoint Finalization (Runner)

Scope: This document continues directly after Part 4 ends. It begins at the runner’s curation+plotting phase:

- `curated_units_for_plot = _apply_waveforms_curation(...)`

and ends when the waveforms stage saves its checkpoint at `ProcessingStage.ANALYZER_COMPLETE` and returns `WaveformExtractOutputs`.

Terminology note:
- Channel-set names (all/common/segment/non-common/unique/non-unique) are defined in `methods_waveforms_channel_sets.md`.

Primary code paths:
- `axon_reconstructor.pipeline.waveforms.runner.extract_waveforms(...)` (curation → plotting → checkpoint)
- `axon_reconstructor.pipeline.waveforms.steps._apply_waveforms_curation(...)`
- `axon_reconstructor.pipeline.waveforms.steps._plot_waveforms_outputs(...)`
- `axon_reconstructor.pipeline.waveforms.curation.apply_mea_analysis_curation(...)`
- `axon_reconstructor.pipeline.waveforms.plotting._write_waveforms_grid_pdf(...)`

---

## Preconditions (state coming into Part 5)

At this point, the waveforms runner has already:

- produced the concat analyzer in:
  - `<well_out_dir>/stg3_waveforms_outputs/concat_waveforms/`
- optionally produced per-segment analyzers in:
  - `<well_out_dir>/stg3_waveforms_outputs/segment_waveforms/segXX_<rec_name>/`
- persisted Part 4 audit artifacts:
  - `stg3_waveforms_outputs/channel_groups.json`
  - `stg3_waveforms_outputs/best_channel_sources.xlsx`
  - `stg3_waveforms_outputs/waveform_filtering_summary.json`
  - `stg3_waveforms_outputs/wf_rejection_log.xlsx`

The remaining goal is:
- apply MEA_Analysis-style curation thresholds (using spikesorting metrics),
- generate QC plots,
- and mark the waveforms stage complete via a dedicated waveforms checkpoint.

Important design change:
- The waveforms stage intentionally does **not** compute quality metrics or template metrics.
- Curation is driven by the spikesorting stage’s `qm_unfiltered.xlsx` (under `stg2_spikesorting_outputs/`) to avoid metric drift due to parameterization (notably `presence_ratio.bin_duration_s`).

---

## 1. Apply curation thresholds (from spikesorting metrics)

Runner call:

- `curated_units_for_plot = _apply_waveforms_curation(...)`

Implementation:
- `axon_reconstructor.pipeline.waveforms.steps._apply_waveforms_curation`

What it does:
- Loads spikesorting quality metrics from:
  - `<well_out_dir>/stg2_spikesorting_outputs/qm_unfiltered.xlsx`
- Runs MEA_Analysis-style curation logic on that table:
  - `clean_metrics, rejection_log = apply_mea_analysis_curation(q_metrics=qm, user_thresholds=None)`

Where the curation logic comes from:
- Preferred path: if MEA_Analysis is importable, it calls MEA_Analysis’ internal curation method.
- Fallback path: if not importable, it uses built-in default thresholds (presence ratio, RP contamination, firing rate, amplitude, amplitude CV).

Artifacts written:
- None in the waveforms stage. Curation here is used only to decide which units appear in the curated plotting permutations.
- If you need curated metrics tables / rejection logs on disk, those should come from the spikesorting stage outputs.

Return value:
- `curated_units_for_plot` is a list of unit IDs (index of `clean_metrics`).
- If curation fails, it returns `None` and plotting proceeds in “uncurated-only” mode.

---

## 2. Plot waveforms grids for human QC

Runner call:

- `waveforms_grid_pdf, spikesorting_waveforms_grid_pdf = _plot_waveforms_outputs(...)`

Implementation:
- `axon_reconstructor.pipeline.waveforms.steps._plot_waveforms_outputs`
- which uses `axon_reconstructor.pipeline.waveforms.plotting._write_waveforms_grid_pdf`

This phase produces MEA_Analysis-style waveform grid PDFs by loading waveforms from the on-disk analyzers.

### 2.1 Segment overlay behavior

If per-segment analyzers exist, `_plot_waveforms_outputs(...)` passes them as `segment_waveforms_folders`.

In `_write_waveforms_grid_pdf(...)`:
- For each unit, it tries to load waveforms from:
  - concat analyzer waveforms extension, and
  - each segment analyzer waveforms extension.
- It then chooses a single “best” channel across the union of available channel IDs (best negative mean deflection; tie-break by channel presence across sources).
- It overlays waveform snippets from concat + segments on that selected channel.

This means the PDF is explicitly “cross-source” when per-segment analyzers are available.

### 2.2 Grid outputs (`grids/`) and panel outputs (`panels/`)

To reduce clutter, the waveforms stage writes all grid artifacts under:

- `<waveforms_out_dir>/grids/`

and all per-unit panel SVGs under:

- `<waveforms_out_dir>/panels/`

Each grid permutation produces:

- a multi-page PDF: `<waveforms_out_dir>/grids/<name>.pdf`
- per-page debug images (so you can inspect without opening the PDF):
  - `<waveforms_out_dir>/grids/<name>_pages/page_000.png` (and `.svg`)
- per-unit SVG panels:
  - `<waveforms_out_dir>/panels/<name>/unit_<unit_id>.svg`

Grid permutations currently written:

- `concat_uncurated`
- `concat_curated` (only if curation succeeds)
- `uncurated_old_bestchan`
- `uncurated_new_best_chan`
- `curated_old_bestchan` (only if curation succeeds)
- `curated_new_bestchan` (only if curation succeeds)

Where:
- `concat_*` means concat analyzer only (no segment overlay).
- `*_old_bestchan` means the per-unit channel is chosen using the concat-only heuristic.
- `*_new_bestchan` means the per-unit channel is chosen across the union of available channels from concat + segments.

Plotting resume/restart semantics:
- Each permutation is only re-written if its PDF does not exist, or if `inputs.force_restart=True`.

Note on `spikesorting_waveforms_grid_pdf`:
- The current implementation returns `None` for this output.

---

## 3. Save checkpoint and return outputs

After metrics, curation, and plotting complete, the runner saves a waveforms-specific checkpoint:

- `save_checkpoint(..., stage=ProcessingStage.ANALYZER_COMPLETE, ...)`

Key design note:
- This is a separate checkpoint file from the main pipeline checkpoint to avoid regressing global stage numbers.

The checkpoint includes `extra_fields` pointing at key output locations:
- `waveforms_out_dir`
- `concat_waveforms_dir`
- `segment_waveforms_dir`
- `waveforms_params_json`
- `waveforms_filtering_json`
- `waveforms_grid_pdf`

Finally, the runner logs `"Waveform extraction complete"` and returns a `WaveformExtractOutputs` object with the same paths.

---

## What comes next

After the waveforms stage finishes, later pipeline stages typically consume:
- the `concat_waveforms` analyzer,
- any per-segment analyzers,
- and the plotting artifacts for QC and downstream decisions.

For quality metrics, later stages should use the spikesorting outputs:
- `<well_out_dir>/stg2_spikesorting_outputs/qm_unfiltered.xlsx` (and any curated variants produced there)
