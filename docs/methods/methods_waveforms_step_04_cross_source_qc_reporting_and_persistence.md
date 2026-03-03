# Waveforms Step — Part 4: Cross-Source QC, Reporting, and Persistence (Runner)

Scope: This document continues directly after Part 3 ends. It begins after the runner has:
- computed `concat_best` from the concat analyzer and (optionally) `seg_best` from per-segment analyzers, and
- finished enriching `filtering_summary` and `wf_rejection_rows` during waveform extraction.

It ends immediately after the runner persists the waveforms-stage “audit artifacts”:
- `stg3_waveforms_outputs/channel_groups.json`
- `stg3_waveforms_outputs/best_channel_sources.xlsx`
- `stg3_waveforms_outputs/waveform_filtering_summary.json`
- `stg3_waveforms_outputs/wf_rejection_log.xlsx`

Terminology note:
- Channel-set names (all/common/segment/non-common/unique/non-unique) are defined in `methods_waveforms_channel_sets.md`.

Primary code paths:
- `axon_reconstructor.pipeline.waveforms.runner.extract_waveforms(...)` (post-extraction block)
- `axon_reconstructor.pipeline.waveforms.reporting._write_best_channel_sources_xlsx(...)`
- `axon_reconstructor.pipeline.waveforms.artifacts._persist_filtering_and_exclusions(...)`
- `axon_reconstructor.pipeline.waveforms.artifacts._persist_channel_groups_json(...)`

---

## Preconditions (state coming into Part 4)

From earlier parts, the runner has already produced (best-effort):

- `ctx.waveforms_out_dir` which is `<well_out_dir>/stg3_waveforms_outputs/`
- `ctx.concat_waveforms_dir` which is `<waveforms_out_dir>/concat_waveforms/`
- `ctx.segment_waveforms_dir` which is `<waveforms_out_dir>/segment_waveforms/` (only if `inputs.per_segment=True`)

And the runner already has these in memory:

- `common_channel_ids`: the channel set present in the concat recording (intended to correspond to **common channels**)
- `channel_groups`: a dict initialized earlier with at least `common_channel_ids` and an empty `segments` sub-dict
  - If per-segment extraction ran, `channel_groups` may now also include union/intersection diagnostics and per-segment breakdowns.
- `concat_best`: best-channel-by-template-PTP summary for the concat analyzer
  - shape: `{unit_id -> (ptp_uv, channel_id)}`
- `seg_best`: best-channel-by-template-PTP summary *across segment analyzers*
  - shape: `{unit_id -> (ptp_uv, channel_id, source_name)}`
  - This is empty if per-segment extraction was disabled or nothing succeeded.
- `filtering_summary`: aggregate counters for the waveforms-stage filters
- `wf_rejection_rows`: spike-level rejection rows accumulated during concat + per-segment extraction

---

## 1. Persist channel-set bookkeeping: `channel_groups.json`

Immediately after per-segment extraction returns (or is skipped), the runner persists the channel bookkeeping dict:

- Output path:
  - `<well_out_dir>/stg3_waveforms_outputs/channel_groups.json`

Why this exists:
- This file makes the “channel sets” concrete for a run (which channels were treated as **common channels**, what each segment’s **segment channels** were, and what the **non-common segment channels** were when exclusion succeeded).
- It’s intentionally plain JSON so you can inspect it without loading SpikeInterface analyzers.

Notes:
- This persistence is best-effort and wrapped in a try/except; failures here do not fail the waveforms stage.
- The runner may log a quick count summary if `channel_groups["counts"]` is available.

---

## 2. Cross-source QC: “does any segment beat concat for best channel?”

If per-segment analyzers ran and produced any `seg_best`, the runner performs a lightweight QC check:

Goal:
- Detect units for which a **segment analyzer** contains a much stronger best-channel template (higher PTP) than the **concat analyzer**.
- This is a strong hint that the concat/common-channel intersection may have dropped the unit’s true strongest electrode.

Exact logic (runner defaults):
- `ratio_thr = 1.10` (segment best must be ≥ 10% larger than concat best)
- `abs_thr_uv = 2.0` (segment best must exceed concat best by > 2 µV)

For each `unit_id`:
- Pull `ptp_seg_uv, ch_seg, src_name` from `seg_best`.
- Pull `ptp_concat_uv, ch_concat` from `concat_best` (defaults to 0 if missing).
- If both the absolute and ratio thresholds are exceeded, emit a warning log with:
  - unit id
  - segment source name
  - segment channel + PTP
  - concat channel + PTP

Interpretation guidance:
- This warning does not mean the sorting is “wrong”; it means the concat representation (common channels only) may be a weaker view of the unit than at least one segment’s view.
- It’s primarily useful when you later merge/propagate best-channel-dependent quantities across sources (templates, footprints, etc.).

---

## 3. Persist a cross-source provenance report: `best_channel_sources.xlsx`

Whether or not warnings were emitted, the runner next writes an XLSX report that records the best-channel candidate(s) across sources.

- Output path:
  - `<well_out_dir>/stg3_waveforms_outputs/best_channel_sources.xlsx`

Implementation entry point:
- `axon_reconstructor.pipeline.waveforms.reporting._write_best_channel_sources_xlsx(...)`

Important behavior:
- The function **loads analyzers from disk**:
  - `si.load_sorting_analyzer(ctx.concat_waveforms_dir)`
  - for each segment directory under `ctx.segment_waveforms_dir`, it tries `si.load_sorting_analyzer(seg_dir)`

What it computes:
- For each source analyzer (concat and each segment), it computes:
  - `ptp_uv`: best PTP (peak-to-peak amplitude) from the analyzer’s templates
  - `channel_id`: the channel id where that best PTP occurs
  - `channel_x`, `channel_y`: best-effort channel location lookup (when available)

Workbook sheets:
- `best_by_unit`
  - winner row per unit (max PTP across all sources)
  - includes convenience columns for:
    - the concat best
    - the best segment (if any)
  - includes a `segment_beats_concat` boolean when computable
- `by_source`
  - all rows for all sources, useful for debugging or plotting

Force-restart semantics:
- If the XLSX exists and `inputs.force_restart` is false, it will not overwrite.

---

## 4. Persist filtering summaries and spike-level rejection log

Finally, the runner persists the filtering audit artifacts via:

- `axon_reconstructor.pipeline.waveforms.artifacts._persist_filtering_and_exclusions(...)`

### 4.1 `waveform_filtering_summary.json`

- Output path:
  - `<well_out_dir>/stg3_waveforms_outputs/waveform_filtering_summary.json`

Content:
- A JSON-serializable dict of aggregate counts produced during waveforms extraction.
- This is where you can confirm (at a glance) that:
  - concat filtering removed expected out-of-epoch spikes,
  - per-segment extraction skipped segments when appropriate,
  - and (when enabled) per-segment channel restriction to **non-common segment channels** took effect.

### 4.2 `wf_rejection_log.xlsx`

- Output path:
  - `<well_out_dir>/stg3_waveforms_outputs/wf_rejection_log.xlsx`

Purpose:
- A spike-level rejection log designed to be joinable to downstream waveforms/template computations.

Key columns (when present):
- identity: `scope`, `source_name`, `segment_index`, `rec_name`, `unit_id`
- timing: `spike_sample_local`, `spike_sample_concat`, `spike_time_s`
- reason: `reason`
- provenance: `stream_id`, `sorter`, `h5_path`
- window: `fs_hz`, `ms_before`, `ms_after`, `pre_samples`, `post_samples`

Sheets:
- `summary`: a compact count summary (e.g., by reason)
- `unit_counts`: grouped counts by `(scope, source_name, segment_index, rec_name, unit_id, reason)`
- `rejections_000`, `rejections_001`, …: the raw rejection rows (chunked to keep sheet sizes manageable)

Failure tolerance:
- Writing this XLSX is best-effort; failures only emit a warning and do not fail the waveforms stage.

### 4.3 Legacy optional artifact: `wf_exclusions.npz` (deprecated)

There is an escape hatch controlled by `inputs.write_wf_exclusions_npz`.

- If enabled, `_persist_filtering_and_exclusions(...)` will attempt to write a legacy `wf_exclusions.npz`.
- This is explicitly marked deprecated in logs, and the preferred downstream interface is to consume the analyzers directly.

---

## 5. What comes next (Part 5)

After Part 4 completes, the runner proceeds to:

- compute quality/template metrics for concat + segments,
- merge those metrics into a single table,
- apply curation thresholds,
- plot QC PDFs,
- and finalize the waveforms checkpoint.

Those steps are covered in Part 5.
