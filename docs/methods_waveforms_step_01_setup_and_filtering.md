# Waveforms Step — Part 1: Setup + Epoch-Aware Filtering (Runner)

Scope: This document covers the **first chunk** of the waveforms stage as implemented in the runner. It ends right after we produce `filtered_sorting` (the spike trains we will actually use for waveform extraction).

Terminology note:
- Channel-set names (all/common/segment/non-common/unique/non-unique) are defined in `docs/methods_waveforms_channel_sets.md`.

Primary code path:
- `axon_reconstructor.pipeline.waveforms.runner.extract_waveforms(...)`

Related helper modules (called during this part):
- `axon_reconstructor.pipeline.waveforms.run_context` (initialize run context, resolve waveform window, load epoch markers)
- `axon_reconstructor.pipeline.waveforms.utils` (load preprocessed recording; load sorter output)
- `axon_reconstructor.pipeline.waveforms.filtering` (epoch-aware spike filtering + rejection logging scaffolding)
- `axon_reconstructor.pipeline.waveforms.qm_config` (quality-metric extension parameters)

---

## 1. Entry + run context initialization

1. Call `extract_waveforms(inputs=..., logger_name_prefix=...)`.

2. Initialize a waveforms-specific run context via `_initialize_run_context(inputs, logger_name_prefix)`.

   1. Compute `well_out_dir` (the MEA_Analysis-style output folder for this well).
      - Uses the same directory computation used throughout the pipeline so waveforms artifacts land under the same well directory.

   2. Create a pipeline logger.
      - The logger is configured to write into the per-well pipeline log file.
      - Name is `{logger_name_prefix}.{stream_id}`.

   3. Create/resolve output directories:
      1. `waveforms_out_dir = <well_out_dir>/waveforms_outputs/`
      2. `concat_waveforms_dir = <waveforms_out_dir>/concat_waveforms/`
      3. `segment_waveforms_dir = <waveforms_out_dir>/segment_waveforms/` (only if `inputs.per_segment=True`)

   4. Define waveforms bookkeeping artifact paths:
      1. `params_json = <waveforms_out_dir>/waveform_extraction_params.json`
      2. `filtering_json = <waveforms_out_dir>/waveform_filtering_summary.json`

   5. Resolve a **dedicated waveforms checkpoint file**:
      - Purpose: avoid interfering with the main pipeline checkpoint (whose stage machine does not include waveforms).
      - Name is derived from the main checkpoint, but ends in `_waveforms_checkpoint.json`.

   6. Load the waveforms checkpoint state (`load_checkpoint(...)`).
      - If `inputs.force_restart=True`, the checkpoint loader can reset state to allow a clean rerun.

---

## 2. Fast-path resume (optional)

1. Try `_resume_if_possible(inputs, ctx)`.

   1. Condition: if `inputs.force_restart == False` AND `ctx.concat_waveforms_dir` already exists.

   2. Behavior:
      - Log that we are resuming.
      - Return a best-effort `WaveformExtractOutputs` object that points to existing folders.
      - Attempt to point to `waveforms_grid_uncurated.pdf` if present.

2. If resume succeeded: **return immediately** (no recomputation).

Notes:
- This resume path is intentionally conservative: it doesn’t validate that all extensions exist; it assumes the prior run produced usable artifacts.

---

## 3. Stage checkpoint write (pre-run)

1. Persist a checkpoint update indicating the waveforms stage is entering analyzer work.

   - Stage stored: `ProcessingStage.ANALYZER`.
   - Extra fields include `waveforms_out_dir`.

Purpose:
- If the run crashes mid-way, there is a durable record that waveforms started and where outputs were intended to go.

---

## 4. Load the concat recording (authoritative time base)

1. Load the **preprocessed concatenated recording** from preprocessing outputs:

   1. `recording = _load_preprocessed_recording(well_out_dir=ctx.well_out_dir)`
      - Reads from: `<well_out_dir>/<preprocess_outputs_dir>/preprocessed_recording/`
      - Uses SpikeInterface loader (`si.load` / `si.load_extractor`).

2. Define the authoritative sampling rate:
   - `fs_hz = recording.get_sampling_frequency()`

Scientific rationale:
- All spike times in the sorter output are assumed to be in the **concatenated** sample index space.
- This concat recording defines the only globally consistent time base for this waveforms step.

---

## 5. Resolve waveform cutout window (ms → samples)

1. Resolve waveform window parameters into a `_WaveformWindow` object:

   1. `window = _resolve_waveform_window(inputs, fs_hz)`

2. Inside `_resolve_waveform_window`:

   1. Determine `ms_before` and `ms_after`:
      - If user provided them in `inputs`, use them.
      - Otherwise infer them from Maxwell trigger settings via `_infer_cutout_ms(...)`.

   2. Convert to samples:
      - `pre_samples = ceil(ms_before * fs_hz / 1000)`
      - `post_samples = ceil(ms_after * fs_hz / 1000)`

Why this matters:
- Every subsequent boundary check (epoch edges, segment edges) uses `pre_samples` and `post_samples`.
- Any spike too close to a boundary (by this window) must be excluded to prevent extracting a waveform window that spans a discontinuity.

---

## 6. Load spikesorting output (concat coordinates)

1. Resolve sorter output directory for this well:

   1. `sorter_output_dir = _resolve_mea_sorter_output_dir(well_out_dir)`
      - Prefers: `<well_out_dir>/spikesorting_outputs/sorter_output/`
      - Falls back to legacy: `<well_out_dir>/sorter_output/`

2. Load sorting extractor:

   1. `sorting = _load_sorting_from_sorter_output_dir(sorter_output_dir, inputs.sorter)`
      - Uses `si.read_sorter_folder(...)` when available.
      - Falls back to `si.load_extractor(...)`.

Assumption:
- `sorting` spike trains are in the concat sample index space.

---

## 7. Safety cleanup: remove out-of-range spikes, drop empty units (best-effort)

1. Attempt to clean up sorting vs recording length:

   1. `sorting = si.remove_excess_spikes(sorting, recording)`
      - Removes spikes that refer to samples outside `[0, recording.num_samples)`.
      - Protects downstream analyzer computations from indexing errors.

   2. `sorting = sorting.remove_empty_units()`
      - Some units may become empty after removing excess spikes.
      - Empty units can create confusing downstream artifacts and metrics.

2. If this cleanup fails, log debug info and continue (best-effort).

Why this exists:
- In practice, sorter output can include spikes beyond the final sample (depending on how the sorter handles edges or rounding). This prevents hard failures.

---

## 8. Load epoch marker JSONs from preprocessing

1. Load epoch markers:

   1. `epochs = _load_epoch_markers(well_out_dir, stream_id)`

2. Epoch marker files (when present) are expected under `<well_out_dir>/<preprocess_outputs_dir>/`:

   1. `maxwell_contiguous_epochs_<stream_id>.json`
      - Defines contiguous “valid signal” intervals in concat coordinates.
      - Used to exclude spikes that fall in discontinuities or too close to epoch edges.

   2. `concatenation_stitch_epochs_<stream_id>.json`
      - Defines how raw rec segments map into concat time.
      - Used later for per-segment waveform extraction and segment-local coordinate transforms.

3. Convert maxwell epochs into numeric concat intervals:
   - `_epochs_to_intervals` converts `[{'start_sample', 'end_sample'}, ...]` into `[(start, end), ...]`.

---

## 9. Configure quality metric extension parameters (for later analyzer compute)

1. Compute a conservative `min_duration_s` for robust binning:

   1. If `inputs.per_segment=True` and we have `epochs.concat_epochs`:
      - Compute each segment duration `(end-start)/fs_hz`.
      - Take the minimum.

   2. Else:
      - Fall back to `recording.get_total_duration()`.

2. Build params:
   - `quality_metrics_params = build_quality_metrics_extension_params(min_duration_s, logger)`

Important detail:
- This config primarily tunes `presence_ratio.bin_duration_s` so short segments don’t produce warnings or degenerate results.

---

## 10. Initialize filtering summaries + rejection log scaffolding

1. Initialize a structured `filtering_summary` dict:
   - `_init_filtering_summary(inputs, window, epochs)`

   Contains:
   - window parameters (ms + samples)
   - epoch marker paths
   - counts for kept/removed spikes
   - a nested section for per-segment bookkeeping

2. Initialize the spike-level rejection row list + base fields:

   1. `(wf_rejection_rows, base_rej_fields) = _init_wf_rejection_log_fields(inputs, window)`

   `base_rej_fields` includes:
   - h5_path, stream_id, sorter
   - fs_hz
   - ms_before/ms_after and sample equivalents

Purpose:
- Every excluded spike can later be written as a row in `wf_rejection_log.xlsx`.
- Rows are intended to be joinable downstream (scope/source/unit_id/spike_sample + reason).

---

## 11. Filter spikes by Maxwell contiguous epochs (produces `filtered_sorting`)

1. Run epoch-aware filtering:

   1. `filtered_sorting = _filter_sorting_by_maxwell_epochs(...)`

2. Conditions:

   1. Filtering runs only when:
      - `inputs.filter_by_maxwell_epochs=True` AND
      - `epochs.maxwell_intervals` is non-empty.

   2. If epoch markers are missing or filtering is disabled:
      - `filtered_sorting` is left equal to `sorting`.
      - The summary tries to record total spikes as “kept”.

3. What `_filter_sorting_by_maxwell_epochs` does (concat scope):

   For each unit `u` in `sorting.get_unit_ids()`:

   1. Pull the entire spike train in concat samples:
      - `st = sorting.get_unit_spike_train(u)`
      - Convert to sorted Python ints.

   2. For each spike time `t`:
      1. Define the waveform window bounds:
         - `t0 = t - pre_samples`
         - `t1 = t + post_samples`

      2. Determine whether `t` is inside a maxwell interval.

      3. Keep the spike only if the **entire waveform window** is within a single interval:
         - `t0 >= interval_start` AND `t1 < interval_end`

      4. Otherwise, reject the spike as one of:
         - `outside_maxwell_epoch` (spike not within any interval)
         - `waveform_window_crosses_epoch_edge` (spike in interval but window crosses boundary)

   3. Append a rejection row per excluded spike:
      - Includes `scope='concat'`, `source_name='concat'`, `unit_id`, `spike_sample_concat`, `spike_time_s`, and `reason`.

4. Construct `filtered_sorting`:

   1. Build `unit_trains: {unit_id -> kept_spike_times}`.
   2. Convert to a SpikeInterface `NumpySorting` via `_to_numpy_sorting(unit_trains, fs_hz)`.
      - Sorting is built from times+labels with a stable sort.

5. Update `filtering_summary` totals:
   - kept total
   - removed total
   - removed outside vs removed by edge

Notes / implications:
- This filtering is the *authoritative* spike exclusion mechanism for waveform extraction.
- It prevents boundary artifacts caused by Maxwell discontinuities.
- It can remove different spikes for different units depending on spike timing.

---

## Output of Part 1

At the end of this part, we have:

- `recording`: the concat preprocessed recording (authoritative time base)
- `window`: waveform cutout parameters in ms and samples
- `sorting`: original sorter output (best-effort cleaned)
- `epochs`: epoch markers from preprocessing
- `filtering_summary`: initialized and partially populated
- `wf_rejection_rows`: a list of per-spike exclusions (concat scope)
- `filtered_sorting`: the sorting that will be used for waveform extraction

Next part (not covered here):
- Persist params JSON
- Extract concat waveforms analyzer
- Optionally extract per-segment waveforms analyzers
- Cross-source best-channel reporting
- Persist filtering artifacts
- Compute/merge metrics, curate, plot
