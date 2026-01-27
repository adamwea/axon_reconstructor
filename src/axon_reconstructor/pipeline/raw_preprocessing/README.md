# Raw preprocessing

Builds a concatenated SpikeInterface recording from a Maxwell `.h5` stream and emits metadata/diagnostics that later steps use to avoid Maxwell snippet boundaries.

## Primary API

- `axon_reconstructor.pipeline.raw_preprocessing.build_concatenated_recording(...)`
- Planning helpers (optional): `build_preprocess_plan(...)`, `discover_cfg_files(...)`, `parse_cfg_channel_locations(...)`

## Inputs

- `h5_path`: path to Maxwell `.h5`
- `stream_id`: well/stream identifier (e.g. `"well000"`)

## Outputs (artifacts)

This step is mostly “in-memory” (returns a `Recording` object), but it can write useful diagnostics when you pass output dirs:

- `assay_stats_<stream_id>.txt`
  - best-effort metadata dump (assay + `/data_store` timing)
- `channel_layouts/`
  - `layout_<stream>_<rec>.png` (shared electrodes highlighted)
  - `layout_shared_<stream>.png`
  - `layout_combined_<stream>.png`
- Epoch marker JSONs (if `epoch_markers_output_dir` provided)
  - `maxwell_contiguous_epochs_<stream_id>.json`
  - `concatenation_stitch_epochs_<stream_id>.json`
- Concatenation diagnostics (if `plot_output_dir` provided)
  - `concat_cluster_reps_<stream_id>.png`
  - `segment_traces/segment_trace_<stream_id>_<rec>.png`

## What it returns

- `(multirecording, common_electrodes)`
  - `multirecording`: concatenated recording (channel ids are normalized to stable identities)
  - `common_electrodes`: the intersection electrode ids across all segments

## Mermaid flow

```mermaid
flowchart TD
  A[Maxwell .h5 + stream_id] --> B[build_concatenated_recording]
  B -->|returns| C[SpikeInterface recording (concatenated)]
  B -->|returns| D[common_electrodes]
  B --> E[assay_stats_*.txt]
  B --> F[channel_layouts/*.png]
  B --> G[maxwell_contiguous_epochs_*.json]
  B --> H[concatenation_stitch_epochs_*.json]
```

## Notes

- Uses the Maxwell HDF5 compression plugin if available (see `utils._ensure_maxwell_hdf5_plugin_path`).
- The epoch JSONs are designed for the waveforms stage’s `filter_by_maxwell_epochs` option.
