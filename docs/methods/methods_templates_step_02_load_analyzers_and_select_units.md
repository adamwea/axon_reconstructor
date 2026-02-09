# Templates Step — Part 2: Load Waveforms Analyzers, Select Units, and Initialize Summary

Scope: this document covers the portion of the templates runner that loads waveforms-stage analyzers, determines which unit ids will be processed, infers plotting time-window metadata, and initializes the `templates_summary.json` payload.

Primary code path:
- `axon_reconstructor.pipeline.templates.runner.extract_and_merge_templates(inputs=...)`

Related helper modules:
- `axon_reconstructor.pipeline.templates.multi_source_utils` (load analyzers; load curated unit list; ID normalization)
- `axon_reconstructor.pipeline.templates.processing` (apply unit curation; infer plot window)

---

## 1. Load waveforms-stage analyzers (multi-source)

Templates is a multi-source stage: it can combine templates computed from:

- the **concat** waveforms analyzer (stable channel set used for sorting)
- optional **per-segment** waveforms analyzers (can recover channels not present in concat intersection)

### 1.1 Analyzer locations

Templates expects the waveforms stage to have written analyzers under:

- `<well>/waveforms_outputs/concat_waveforms/`
- `<well>/waveforms_outputs/segment_waveforms/<segment_name>/` (0+ segment folders)

### 1.2 Loader behavior

`_load_waveforms_analyzers(well_out_dir, include_concat, include_segments, logger)` returns:

- `analyzers: list[tuple[str, SortingAnalyzer]]`

Rules:

- If `inputs.include_concat=True`, concat analyzer **must exist** or we raise `FileNotFoundError`.
- If `inputs.include_segments=True` and `segment_waveforms/` exists, we attempt to load each directory:
  - unreadable segment analyzers are **skipped** with a warning.
- If no analyzers were successfully loaded, we raise `RuntimeError`.

The analyzer list is ordered as:

- `("concat", analyzer)` first (if included)
- then segment analyzers in sorted directory-name order

This ordering matters because later code:
- chooses `analyzers[0]` as a default “universe” for unit ids
- prefers the source named `concat` when selecting a reference analyzer

---

## 2. Determine which units to process

Templates uses the following unit selection rules.

### 2.1 If the user passes explicit `unit_ids`

If `inputs.unit_ids is not None`:

- `unit_ids = list(inputs.unit_ids)`
- This list is treated as the full unit set to process (no waveforms-stage curation filtering is applied).

### 2.2 Default behavior: use spikesorting-stage curation (quality metrics)

If `inputs.unit_ids is None`:

1. Start from the “universe” of unit ids in the first analyzer’s sorting:

   - `unit_ids = list(analyzers[0][1].sorting.unit_ids)`

2. Apply spikesorting-stage unit curation via `_apply_spikesorting_stage_unit_curation(...)`.

  - It reads spikesorting quality metrics from:
    - `<well>/spikesorting_outputs/qm_unfiltered.xlsx`
  - It applies `waveforms.curation.apply_mea_analysis_curation(q_metrics=qm, user_thresholds=None)` to derive the curated set.

3. If `qm_unfiltered.xlsx` is found and pandas is available:

  - Read it using `pd.read_excel(..., index_col=0)` (best-effort)
  - Ensure `unit_id` is the dataframe index (best-effort)
  - Normalize IDs via `_normalize_id_for_compare` to handle numpy scalars / float-as-int IDs robustly.

4. Filtering logic:

   - Convert curated IDs into a set (`curated_set`)
   - Keep only units from the “universe” whose normalized ID is in that set

  This makes spikesorting-stage curation authoritative for what proceeds into templates.

### 2.3 Require curated units (default)

If `inputs.require_curated_units=True` (default) and curated units cannot be derived:

- templates raises a `RuntimeError` explaining that it expected `<well>/spikesorting_outputs/qm_unfiltered.xlsx`.

Ways to override:

- run spikesorting first so `<well>/spikesorting_outputs/qm_unfiltered.xlsx` exists
- pass `TemplateExtractInputs(unit_ids=[...])`
- or set `require_curated_units=False`

### 2.4 Unit limit (debugging control)

If `inputs.unit_limit is not None`:

- truncate: `unit_ids = unit_ids[:unit_limit]`

This is intended as a development/debug knob.

---

## 3. Infer plotting time window metadata

Templates plots need to know the sampling rate and (optionally) the waveform window `ms_before/ms_after`.

This stage does **not** recompute or enforce the waveforms window; it uses best-effort metadata.

### 3.1 Sampling frequency

`_infer_template_plot_window(well_out_dir, analyzers, read_json)`:

- tries `fs_hz = analyzers[0][1].recording.get_sampling_frequency()`
- falls back to `fs_hz = 10_000.0` on failure

### 3.2 Window duration in milliseconds

If available, it reads the waveforms-stage params JSON:

- `<well>/waveforms_outputs/waveform_extraction_params.json`

It attempts to extract:

- `ms_before`
- `ms_after`

If anything fails, both are left as `None` and plots fall back to a simple time axis derived from `fs_hz`.

---

## 4. Initialize the templates summary payload

The runner builds a `summary` dict that will be written to:

- `<well>/templates_outputs/templates_summary.json`

Important fields include:

- dataset pointers: `h5_path`, `stream_id`, `well_out_dir`, `templates_out_dir`
- sources: list of analyzer names (e.g. `["concat", "seg_0", "seg_1", ...]`)
- output pointers (paths as strings) for:
  - `templates_grid_pdf`
  - `multi_source_templates_dir` (current `unit_segment_grids/`)
  - optional `full_channels_templates_dir`, `topo_unit_footprints_dir`, `propagation_plots_dir`
  - plots dirs (`footprints`, `svgs`, `full_chip_maps`, `axon_velocity_outputs_root_dir`)
- curation section:
  - `qm_unfiltered_xlsx` (path if used)
  - `applied` boolean
  - `n_curated_units`
- a hint to waveforms best-channel provenance:
  - `waveforms_best_channel_sources_xlsx` if `<well>/waveforms_outputs/best_channel_sources.xlsx` exists
- `units: []` (to be populated per unit)

This `summary` is passed down into `process_unit_list(..., summary=summary)` so that per-unit persistence can append unit entries.

---

## End of Part 2

At this point we have:

- loaded analyzers (concat and/or per-segment)
- decided on the unit list to process
- inferred plotting metadata
- initialized `templates_summary.json` content

Next, the pipeline enters the per-unit processing loop: gather per-source templates, build the `merged_contributing` template, resolve overlaps, and decide what goes in the main grid.

That is covered in **Part 3**.
