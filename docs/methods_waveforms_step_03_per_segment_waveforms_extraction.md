# Waveforms Step — Part 3: Per-Segment Waveforms on Raw Recordings (Runner)

Scope: This document covers the **per-segment** waveform extraction branch in the waveforms runner. It begins right after Part 2 ends (concat analyzer exists and we have `concat_best`) and ends once `_extract_per_segment_waveforms(...)` returns `seg_best` (best-channel-by-PTP summary per unit across segment analyzers).

Primary code paths:
- `axon_reconstructor.pipeline.waveforms.runner.extract_waveforms(...)` (per-segment branch)
- `axon_reconstructor.pipeline.waveforms.extraction._extract_per_segment_waveforms(...)`

Key helper modules involved:
- `axon_reconstructor.pipeline.waveforms.segments` (segment spec parsing + filtering summary updates)
- `axon_reconstructor.pipeline.waveforms.utils` (load raw segment recordings full channels; NumpySorting helper)

---

## Preconditions (state coming into Part 3)

From earlier parts, the runner has:

- `recording`: concat preprocessed recording (common/intersection channel set)
- `filtered_sorting`: concat-time spike trains after epoch-aware filtering
- `sorting`: concat-time sorter output (cleaned best-effort)
- `epochs.concat_epochs`: list of concatenation stitch segments (each has `segment_index`, `rec_name`, `start_sample`, `end_sample`, ...)
- `window`: waveform window in ms + samples (`pre_samples`, `post_samples`)
- `quality_metrics_params`: params for SpikeInterface `quality_metrics`
- `common_channel_ids`: set of channel ids present in concat recording
- `filtering_summary`: mutable dict initialized earlier
- `wf_rejection_rows`: list that accumulates spike-level rejections across concat + segments
- `base_rej_fields`: fields that are merged into every rejection row
- `ctx.segment_waveforms_dir`: `<waveforms_out_dir>/segment_waveforms/` (exists only if `inputs.per_segment=True`)

---

## 1. Runner entry into per-segment extraction

1. In `runner.extract_waveforms(...)`, per-segment extraction runs only if:

   - `inputs.per_segment == True` AND
   - `epochs.concat_epochs` is non-empty

2. The runner calls:

   - `seg_best = _extract_per_segment_waveforms(...)`

   with:
   - `recording` (concat recording; used mainly for logging channel counts)
   - `sorting_unfiltered = sorting` (pre epoch-filter)
   - `filtered_sorting` (post epoch-filter; used to define which spikes are eligible at all)
   - `epochs`, `window`, `segment_waveforms_dir`
   - `common_channel_ids`
   - `filtering_summary`, `wf_rejection_rows`, `base_rej_fields`
   - `quality_metrics_params`

---

## 2. High-level goal of per-segment analyzers

Per-segment analyzers exist to recover waveforms on electrodes that are not present in the concat/common intersection.

Key design choice:
- We **reuse** the concat-time spike trains (from the sorter output) and map them into segment-local time.
- We do **not** re-run detection/sorting per segment.

Scientific rationale:
- The sorter’s spike times/units are defined on the concatenated, preprocessed recording.
- Re-sorting per segment would generally change spike trains and unit identities.

---

## 3. Iterate concat stitch segments and create per-segment analyzer folders

Inside `_extract_per_segment_waveforms(...)`:

1. Initialize a cross-segment best-channel accumulator:

   - `best_by_unit: {unit_id -> (best_ptp_uv, best_channel_id, source_name)}`

2. For each `seg` dict in `epochs.concat_epochs`:

   1. Parse a segment spec:

      - `spec = _parse_concat_epoch_segment(seg=seg, segment_waveforms_dir=segment_waveforms_dir)`

      This extracts:
      - `segment_index`
      - `rec_name`
      - `start_sample_concat`
      - `end_sample_concat`
      - `seg_dir = <segment_waveforms_dir>/segXX_<rec_name>/`

      If parsing fails, skip the segment.

   2. Resolve segment-local variables:

      - `seg_index = spec.segment_index`
      - `rec_name = spec.rec_name`
      - `start = spec.start_sample_concat`
      - `end = spec.end_sample_concat`
      - `seg_dir = spec.seg_dir`

   3. Handle force-restart semantics per segment:

      - If `seg_dir` exists and `inputs.force_restart=True`, delete it before recomputing.

---

## 4. Load raw segment recording (full channels) and optionally preprocess

1. Load the raw Maxwell segment recording with all available channels:

   - `seg_rec = _load_raw_segment_recording_full_channels(
       h5_path=inputs.h5_path,
       stream_id=inputs.stream_id,
       rec_name=rec_name,
       center_chunk_size=10_000,
       preprocess_like_mea_analysis=inputs.per_segment_preprocess_like_mea_analysis,
     )`

2. What `_load_raw_segment_recording_full_channels(...)` does (high level):

   1. Reads the Maxwell rec (`rec_name`) for this well from the raw H5.
   2. Applies `si.center(...)` to remove DC offsets.
   3. When possible, renames channels to Maxwell electrode IDs (via `contact_vector['electrode']`).

3. If `inputs.per_segment_preprocess_like_mea_analysis=True` (default), it then tries to mimic MEA_Analysis preprocessing:

   1. `unsigned_to_signed` (only if dtype is unsigned)
   2. high-pass filter at 300 Hz
   3. common median reference
      - tries local median CMR with `local_radius=(0, 250)`
      - falls back to global median if local fails
   4. `annotate(is_filtered=True)`
   5. cast to float32

Important nuance:
- This MEA-like preprocessing is applied to **per-segment** recordings only.
- The concat analyzer already operates on the preprocessed concat recording produced in preprocessing.

---

## 5. Optionally restrict per-segment extraction to “additional channels only”

This branch is controlled by:

- `inputs.per_segment_only_additional_channels` (default True)

Goal:
- Avoid duplicating work on channels already present in the concat/common intersection.
- Focus per-segment analyzers on channels that were dropped by concatenation.

How it’s implemented:

1. If `inputs.per_segment_only_additional_channels=True` and `common_channel_ids` is non-empty:

   1. Attempt to read per-channel electrode IDs from `seg_rec.get_property("contact_vector")`.
   2. Build a keep-mask for channels whose electrode id is **not** in `common_channel_ids`.
   3. Select only those channels:
      - `seg_rec = seg_rec.select_channels(keep_channel_ids)`
   4. Track bookkeeping counts for filtering_summary:
      - `raw_channels_total`
      - `excluded_common_channels_total`
      - `kept_additional_channels_total`
   5. Best-effort: rename remaining channels to their electrode IDs.

2. If `per_segment_only_additional_channels=True` and selection yields 0 channels:

   1. Log that the segment has no additional channels.
   2. Append a per-segment skip record via `_append_segment_skip_summary(...)` with reason `"no_additional_channels"`.
   3. `continue` to the next segment (no analyzer produced for this segment).

---

## 6. Validate segment length expectations (best-effort)

1. Compute expected segment length from concat stitch epochs:

   - `seg_len_expected = end - start`

2. Compare to raw segment recording length:

   - `seg_len = seg_rec.get_num_samples()`

3. If mismatch, log a warning and proceed.

Rationale:
- Stitch metadata is the authority for mapping concat time to segments, but the raw extractor may report slightly different lengths. The pipeline chooses to warn rather than hard-fail.

---

## 7. Log spike counts in this segment window (debug/QC)

This is informational logging to help audit filtering decisions.

1. Count spikes in this concat-time window for:

   - the unfiltered sorting (`sorting_unfiltered`)
   - the concat-epoch-filtered sorting (`filtered_sorting`)

2. Log:

- unfiltered spikes
- filtered spikes
- excluded-at-concat count (difference)

---

## 8. Compute segment-local Maxwell intervals (optional)

This is computed for every segment, but is only used by a *deprecated* post-hoc random_spikes flagging mode.

1. The extractor calls:

   - `seg_maxwell_intervals = _compute_segment_maxwell_intervals(...)`

   which returns a list of `(start_local, end_local)` intervals in **segment-local** sample coordinates.

2. Sanity check: warn if any interval lies outside `[0, seg_len]`.

Note:
- These intervals are not used for the primary per-segment spike filtering in the current design (edge filtering is segment-bounds based).

---

## 9. Build segment-local spike trains from concat-time filtered sorting

Core idea:
- Take the concat-time spike trains from `filtered_sorting`.
- Restrict them to `[start, end)` in concat coordinates.
- Convert into segment-local sample coordinates by subtracting `start`.
- Drop spikes too close to segment edges given the waveform window.

Step-by-step:

1. Initialize per-segment unit trains:

   - `unit_trains_seg: {unit_id -> [t_local, ...]}`

2. Initialize per-segment counters:

   - `seg_spikes_total` (spikes in window prior to segment-edge filtering)
   - `seg_removed_edge_total` (removed due to segment-edge window violations)
   - `seg_kept_total` (kept after edge filtering)

3. For each unit `u` in `filtered_sorting.get_unit_ids()`:

   1. Read concat-time spikes:
      - `st = filtered_sorting.get_unit_spike_train(u)`

   2. Restrict to this segment’s concat window and convert to local coordinates:
      - `local_all = [t - start for t in st if start <= t < end]`
      - sort `local_all`

   3. Edge filter by waveform cutout:

      Keep only `t_local` such that:

      - `t_local - pre_samples >= 0`
      - `t_local + post_samples < seg_len`

      Those that fail are excluded because the waveform window would extend outside the segment recording.

   4. For each excluded-by-edge spike, append a rejection row (best-effort):

      - `scope = "segment"`
      - `source_name = f"seg{seg_index:02d}_{rec_name}"`
      - `segment_index`, `rec_name`
      - `unit_id`
      - `spike_sample_local = t_local`
      - `spike_sample_concat = t_local + start`
      - `reason = "waveform_window_outside_segment_bounds"`

   5. Update counters and store kept spikes:

      - `unit_trains_seg[int(u)] = kept_edges`

4. Create a SpikeInterface sorting for this segment:

   - `seg_sort = _to_numpy_sorting(unit_trains=unit_trains_seg, fs_hz=window.fs_hz)`

Notes:
- This segment sorting is intentionally single-segment and uses segment-local sample indices.

---

## 10. Segment-local safety cleanup (remove_excess_spikes, drop empty units)

Even though we already attempted to keep only in-bounds spikes, we do a second best-effort safety pass:

1. `seg_sort = si.remove_excess_spikes(seg_sort, seg_rec)`
2. `seg_sort = seg_sort.remove_empty_units()`

Then, best-effort attach the recording handle:

3. `seg_sort.register_recording(seg_rec)` (wrapped in `try/except`)

Rationale:
- Prevent analyzer-time indexing errors if any spike time still violates recording bounds.
- Ensure the per-segment analyzer doesn’t contain units with zero spikes.

---

## 11. Create per-segment SortingAnalyzer and compute extensions

1. Create the analyzer folder for this segment:

- `seg_analyzer = si.create_sorting_analyzer(
     seg_sort,
     seg_rec,
     format="binary_folder",
     folder=seg_dir,
     return_in_uV=True,
   )`

2. Compute `random_spikes` + `waveforms`:

- same parameterization as concat:
  - uniform random spikes
  - bounded by `inputs.max_spikes_per_unit`
  - `seed=0`
  - waveform window uses the same `window.ms_before/ms_after`

- `n_jobs` uses `max(1, inputs.n_jobs)` to avoid invalid values.

3. Compute additional extensions:

- `spike_amplitudes`
- `templates`
- `noise_levels`
- `quality_metrics` (with `quality_metrics_params`)
- `template_metrics`
- `unit_locations` (`monopolar_triangulation`)

---

## 12. Track best PTP channel per unit for this segment (cross-source QC)

1. After templates exist, compute best channel by PTP for this segment analyzer:

- `seg_best = _best_ptp_channel_by_unit_from_templates(analyzer=seg_analyzer)`

2. For each unit, update a cross-segment winner map:

- `best_by_unit[unit_id] = (ptp_uv, channel_id, source_name)`

but only if:

- unit not seen before, or
- `ptp_uv` is greater than the previously stored value.

Result:
- after iterating all segments, `best_by_unit` records the single strongest observed per-segment template channel per unit (and which segment it came from).

---

## 13. Deprecated: post-hoc flagging of segment random_spikes by epochs (optional)

This code path is gated by:

- `inputs.deprecated_flag_segment_random_spikes_by_epochs=True`

and also requires:

- `inputs.filter_by_maxwell_epochs=True`
- `seg_maxwell_intervals` non-empty

What it does:

1. Reads the random spike sample indices selected by the analyzer.
2. Classifies each selected spike as:

- `outside_maxwell_epoch` or
- `waveform_window_crosses_epoch_edge`

3. Appends rejection rows for those classifications.

Important note:
- This is explicitly marked deprecated in-code. The authoritative filtering is the concat-time filtering and the segment-edge filtering described earlier.

---

## 14. Update per-segment filtering summary bookkeeping

For each segment, append a structured summary record via `_append_segment_summary(...)` containing:

- segment identity (`segment_index`, `rec_name`)
- spikes in segment window
- removed by epoch (deprecated path only)
- removed by segment edge (window out of bounds)
- kept spikes total
- number of segment-local maxwell intervals
- channel counts: raw total, excluded common, kept additional

Also update per-segment aggregate totals in `filtering_summary["per_segment"]`.

---

## 15. Return value of Part 3

`_extract_per_segment_waveforms(...)` returns:

- `best_by_unit` (called `seg_best` in the runner):

  `seg_best: {unit_id: (best_ptp_uv, best_channel_id, source_name)}`

where:
- `best_ptp_uv` is the max PTP observed across all per-segment analyzers for that unit
- `best_channel_id` is the channel/electrode id for that best PTP template
- `source_name` identifies which segment analyzer produced the winner, formatted like `segXX_<rec_name>`

Next part (Part 4):
- cross-source QC comparing `seg_best` vs `concat_best`
- write `best_channel_sources.xlsx`
- persist filtering summary + rejection log
- compute/merge metrics, curation, plotting, checkpoint completion
