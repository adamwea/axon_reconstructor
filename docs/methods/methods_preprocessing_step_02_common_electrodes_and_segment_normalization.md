# Preprocessing Step — Part 2: Segment Loading, Channel Normalization, and Common-Electrode Intersection

Scope: this document covers how preprocessing turns per-segment Maxwell recordings into a set of per-segment SpikeInterface `Recording`s that all share the **same physical electrode set**.

Primary code paths:
- `axon_reconstructor.pipeline.raw_preprocessing.runner.build_concatenated_recording(...)`

Related modules:
- `axon_reconstructor.pipeline.raw_preprocessing.concatenation`
  - `find_common_electrodes_from_segments(...)`
  - `_process_rec_segment_for_concatenation(...)`
- `axon_reconstructor.pipeline.raw_preprocessing.utils` (`_ensure_maxwell_hdf5_plugin_path`)

---

## 0. Why electrode intersection exists

Maxwell `.raw.h5` wells often contain multiple `rec_name` segments. Between segments, the routed electrode set can differ (e.g. due to routing changes or acquisition configuration). If you concatenate segments without normalizing/slicing, you can end up with:

- channel count changes mid-recording
- ambiguous channel identities
- invalid comparisons across segments

The pipeline solves this by taking the **intersection** electrode id set across all segments, then slicing every segment to that set.

---

## 1. HDF5 plugin path safety

Before touching the `.raw.h5`, preprocessing calls:

- `_ensure_maxwell_hdf5_plugin_path()`

Purpose:
- Maxwell HDF5 often depends on a compression filter plugin (e.g. `libcompression.so`). This helper tries to make that plugin discoverable via environment configuration.

---

## 2. Segment discovery + common electrode intersection

1. Preprocessing calls:

   - `rec_names, common_el = find_common_electrodes_from_segments(h5_path, stream_id)`

2. `rec_names` is the ordered set of segment identifiers (e.g. `rec0000`, `rec0001`, ...).

3. `common_el` is the set of physical electrode ids shared by all segments.

This intersection step is the “channel identity contract” for downstream stages.

---

## 3. Optional: channel layout diagnostics

If `plot_output_dir` is provided, preprocessing writes channel-layout plots under:

- `<plot_output_dir>/channel_layouts/`

via:

- `_save_channel_layout_plots(h5_path, stream_id, rec_names, common_electrodes, out_dir=...)`

These plots are intended to quickly show per-segment routing differences and confirm that intersection electrodes look reasonable.

---

## 4. Per-segment loading + conditioning

Segments are processed concurrently (thread pool) via:

- `_process_rec_segment_for_concatenation(...)`

Each segment is expected to:

1. Load a per-segment recording using the Maxwell extractor.
2. Normalize channel identity so channel ids reflect physical electrode ids (via `contact_vector["electrode"]`).
3. Slice down to `common_el` (the intersection electrode list).
4. Center each channel (bounded chunk) using SpikeInterface centering, to stabilize downstream processing.

Implementation detail:
- Concurrency is deliberately modest (`max_workers = min(len(rec_names), n_jobs)`) because the extractors are I/O heavy.

---

## 5. Sampling-rate consistency check

After per-segment processing, preprocessing performs a best-effort check that all segment sampling rates match.

- If sampling rates differ, a warning is emitted.

Downstream, the concatenated recording uses the sampling rate as reported by SpikeInterface; significant segment-to-segment mismatch is treated as a data integrity concern.
