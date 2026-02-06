# Waveforms Step — Part 5: Metrics Merge, Curation, Plotting, and Checkpoint Finalization (Runner)

Scope: This document continues directly after Part 4 ends. It begins at the runner’s metrics phase:

- `merged_qm, merged_tm = _compute_and_merge_waveforms_metrics(...)`

and ends when the waveforms stage saves its checkpoint at `ProcessingStage.ANALYZER_COMPLETE` and returns `WaveformExtractOutputs`.

Terminology note:
- Channel-set names (all/common/segment/non-common/unique/non-unique) are defined in `docs/methods_waveforms_channel_sets.md`.

Primary code paths:
- `axon_reconstructor.pipeline.waveforms.runner.extract_waveforms(...)` (metrics → curation → plotting → checkpoint)
- `axon_reconstructor.pipeline.waveforms.steps._compute_and_merge_waveforms_metrics(...)`
- `axon_reconstructor.pipeline.waveforms.steps._apply_waveforms_curation(...)`
- `axon_reconstructor.pipeline.waveforms.steps._plot_waveforms_outputs(...)`
- `axon_reconstructor.pipeline.waveforms.metrics` (metric load/merge/recompute logic)
- `axon_reconstructor.pipeline.waveforms.curation.apply_mea_analysis_curation(...)`
- `axon_reconstructor.pipeline.waveforms.plotting._write_waveforms_grid_pdf(...)`

---

## Preconditions (state coming into Part 5)

At this point, the waveforms runner has already:

- produced the concat analyzer in:
  - `<well_out_dir>/waveforms_outputs/concat_waveforms/`
- optionally produced per-segment analyzers in:
  - `<well_out_dir>/waveforms_outputs/segment_waveforms/segXX_<rec_name>/`
- persisted Part 4 audit artifacts:
  - `waveforms_outputs/channel_groups.json`
  - `waveforms_outputs/best_channel_sources.xlsx`
  - `waveforms_outputs/waveform_filtering_summary.json`
  - `waveforms_outputs/wf_rejection_log.xlsx`

The remaining goal is:
- compute per-source quality/template metrics,
- merge them into a single, consistent metrics view,
- apply MEA_Analysis-style curation thresholds,
- generate QC plots,
- and mark the waveforms stage complete via a dedicated waveforms checkpoint.

---

## 1. Compute per-source metrics and merge to a single table

Runner call:

- `merged_qm, merged_tm = _compute_and_merge_waveforms_metrics(...)`

Implementation:
- `axon_reconstructor.pipeline.waveforms.steps._compute_and_merge_waveforms_metrics`

Output directory convention:
- A per-source metrics tree is created under:
  - `<well_out_dir>/waveforms_outputs/metrics_sources/`

### 1.1 Load metrics from existing analyzers

The metrics computation does **not** recompute waveforms; it loads analyzers from disk and reads extensions.

- Concat metrics are loaded from:
  - `concat_waveforms_dir = <waveforms_out_dir>/concat_waveforms/`

- Segment metrics are loaded (best-effort) from:
  - `<waveforms_out_dir>/segment_waveforms/segXX_<rec_name>/`
  - Segment folders are derived by iterating `epochs.concat_epochs` and using `_parse_concat_epoch_segment(...)`.

How metrics are loaded:
- `load_and_compute_metrics(analyzer_dir=..., ...)` in `axon_reconstructor.pipeline.waveforms.metrics` loads the analyzer and requires:
  - `quality_metrics` extension
  - `template_metrics` extension
- It also tries to add `unit_locations` into the quality-metrics DataFrame as `loc_x` and `loc_y`.

Important dependency note:
- If an analyzer folder is missing required extensions (e.g. `quality_metrics`), the loader raises a helpful error suggesting re-running waveforms with `force_restart=True`.

### 1.2 Per-source file outputs (concat + segments)

For concat, files are written under:

- `<waveforms_out_dir>/metrics_sources/concat/`
  - `qm_unfiltered.xlsx`
  - `tm_unfiltered.xlsx`

Additionally, the code applies MEA_Analysis-style curation *as a per-source diagnostic* and writes:

- `<waveforms_out_dir>/metrics_sources/concat/`
  - `metrics_curated.xlsx`
  - `rejection_log.xlsx`
  - `tm_curated.xlsx` (best-effort)

For each segment source (name like `seg02_<rec_name>`), unfiltered metrics are written under:

- `<waveforms_out_dir>/metrics_sources/segXX_<rec_name>/`
  - `qm_unfiltered.xlsx`
  - `tm_unfiltered.xlsx`

Notes:
- Segment metrics are best-effort; a failure in one segment logs a warning and the pipeline continues.

### 1.3 Merge policy: concat + segments → merged metrics

After per-source metrics are collected, the pipeline merges them into two “merged” tables:

- `merged_qm = merge_quality_metrics(concat_qm, segment_qm_by_source)`
- `merged_tm = merge_template_metrics(concat_tm, segment_tm_by_source)`

Quality-metrics merge policy (high level):
- Column-wise aggregation with safe defaults (intended to align with MEA_Analysis threshold semantics):
  - “Higher-is-better” columns use a min across sources (worst-case)
  - “Lower-is-better” columns use a max across sources (worst-case)
  - certain columns (notably `loc_x`, `loc_y`) prefer concat
  - unknown numeric columns default to concat-preferred if present, else mean fallback

Template-metrics merge policy (high level):
- Concat-preferred per column when concat has that metric; otherwise mean across segments for numeric columns.

### 1.4 Optional recomputation from a deduplicated spike+amplitude representation

After the initial merges, the code attempts a higher-fidelity recomputation step:

- `recompute_merged_quality_metrics_from_deduplicated_spikes(...)`

Goal:
- Avoid purely heuristic scalar merges by recomputing key metrics on a deduplicated view of spikes (using concat-time samples, respecting segment boundaries).

Key idea:
- Build per-unit maps of `sample_index (concat time) -> amplitude` from:
  - concat analyzer spike amplitudes
  - per-segment analyzer spike amplitudes (offset into concat time using each segment’s `start_sample_concat`)
- Where the same spike time appears in multiple sources, choose the amplitude with larger magnitude $|\mathrm{amp}|$.

If successful:
- The recomputed columns are inserted/overwritten in `merged_qm` for the recomputed unit IDs.

If it fails:
- A warning is logged and the pipeline continues using merge-only metrics.

### 1.5 Merged metrics file outputs

Merged metrics are written both under the “merged” subfolder and at the waveforms root:

- `<waveforms_out_dir>/metrics_sources/merged/`
  - `qm_merged.xlsx`
  - `tm_merged.xlsx`

- `<waveforms_out_dir>/`
  - `qm_merged_unfiltered.xlsx`
  - `tm_merged_unfiltered.xlsx`

---

## 2. Apply curation thresholds to merged metrics

Runner call:

- `curated_units_for_plot = _apply_waveforms_curation(...)`

Implementation:
- `axon_reconstructor.pipeline.waveforms.steps._apply_waveforms_curation`

What it does:
- Runs MEA_Analysis-style curation logic on the merged quality metrics table:
  - `clean_metrics, rejection_log = apply_mea_analysis_curation(q_metrics=merged_qm, user_thresholds=None)`

Where the curation logic comes from:
- Preferred path: if MEA_Analysis is importable, it calls MEA_Analysis’ internal curation method.
- Fallback path: if not importable, it uses built-in default thresholds (presence ratio, RP contamination, firing rate, amplitude, amplitude CV).

Artifacts written (only if curation succeeds):

- `<waveforms_out_dir>/metrics_curated.xlsx`
- `<waveforms_out_dir>/rejection_log.xlsx`
- `<waveforms_out_dir>/tm_curated.xlsx` (best-effort subset of `merged_tm`)

Return value:
- `curated_units_for_plot` is a list of unit IDs (index of `clean_metrics`).
- If curation fails, it returns `None` and plotting proceeds in “uncurated-only” mode.

---

## 3. Plot waveforms grids for human QC

Runner call:

- `waveforms_grid_pdf, spikesorting_waveforms_grid_pdf = _plot_waveforms_outputs(...)`

Implementation:
- `axon_reconstructor.pipeline.waveforms.steps._plot_waveforms_outputs`
- which uses `axon_reconstructor.pipeline.waveforms.plotting._write_waveforms_grid_pdf`

This phase produces MEA_Analysis-style waveform grid PDFs by loading waveforms from the on-disk analyzers.

### 3.1 Segment overlay behavior

If per-segment analyzers exist, `_plot_waveforms_outputs(...)` passes them as `segment_waveforms_folders`.

In `_write_waveforms_grid_pdf(...)`:
- For each unit, it tries to load waveforms from:
  - concat analyzer waveforms extension, and
  - each segment analyzer waveforms extension.
- It then chooses a single “best” channel across the union of available channel IDs (best negative mean deflection; tie-break by channel presence across sources).
- It overlays waveform snippets from concat + segments on that selected channel.

This means the PDF is explicitly “cross-source” when per-segment analyzers are available.

### 3.2 Grid outputs (`grids/`) and panel outputs (`panels/`)

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

## 4. Save checkpoint and return outputs

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
- and the merged/curated metrics + plotting artifacts for QC and downstream decisions.
