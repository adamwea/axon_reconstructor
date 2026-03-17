# Stage 01 — Preprocessing (raw → stable concatenated recording)

This stage turns a Maxwell `.raw.h5` well/stream into a **single SpikeInterface Recording** with a stable channel identity and useful timing metadata.

## Why this stage exists

Maxwell “axon tracking” acquisitions commonly produce multiple `rec_name` segments per well. Across segments:

- the available electrode set can differ
- channel ordering can differ
- recordings can be **triggered/snippet** style (stored samples are discontinuous in wall-clock time)

Downstream stages (sorting, waveforms/templates, reconstruction) assume a consistent channel set and benefit from explicit epoch markers at snippet boundaries.

## Primary APIs

- High-level: `axon_reconstructor.pipeline.pipeline_driver.AxonReconstructor.preprocess_for_spikesorting(...)`
- Core builder: `axon_reconstructor.pipeline.preprocessing.build_concatenated_recording(...)`

## Inputs

- `h5_path`: path to Maxwell `.raw.h5`
- `stream_id`: well id, e.g. `"well000"`
- `n_jobs`: used for I/O heavy segment loads and (optionally) saving

Optional (temporal interpolation / upsampling):

- `temporal_resample_factor` (e.g. `10`) or `temporal_resample_rate_hz` (explicit target Hz)
- `temporal_resample_margin_ms` (default `100.0`): resampling edge padding to reduce artifacts
- `temporal_resample_dtype` (optional)

## What it does (stepwise)

1) **Discover segments** for the stream (`rec_name`s inside the H5)
2) **Load each segment** as a SpikeInterface Recording (Maxwell extractor)
3) **Normalize channel identity**
   - Uses the extractor’s `contact_vector["electrode"]` so channel ids become the physical electrode ids.
4) **Intersection of shared electrodes**
   - Computes the intersection of electrode ids across all segments.
   - Slices every segment down to this shared set.
5) **Segment conditioning**
   - Applies `spikeinterface.full.center(...)` (bounded chunk) so sorting/filtering is stable.
6) **Concatenate segments**
   - `spikeinterface.full.concatenate_recordings(rec_list)`
7) **Reconstruct accurate time vectors (when possible)**
   - Reads `/wells/<stream>/<rec>/groups/routed/frame_nos` to build an “absolute-ish” time axis for the concatenated recording.
   - This preserves *gaps* for triggered/snippet acquisitions.
8) **Emit epoch marker JSONs**
   - `maxwell_contiguous_epochs_<stream_id>.json`: contiguous sample runs within snippet/trigger recordings
   - `concatenation_stitch_epochs_<stream_id>.json`: segment stitch boundaries in concatenated coordinates

9) **Optional: temporal resampling (upsampling)**
   - Uses `spikeinterface.preprocessing.resample(...)` after concatenation.
   - Scales epoch marker sample indices by the resampling ratio so later stages remain consistent.

## Outputs on disk (when run via `preprocess_for_spikesorting`)

When `mea_analysis_output_root` is configured, outputs live under the MEA_Analysis-style per-well folder:

- `stages.mea_analysis.phases.preprocessing.execution.save_binary`
   - `true`: persist the preprocessed recording folder using SpikeInterface binary format
   - `false`: run preprocessing in-memory without writing the recording artifacts
- `stages.mea_analysis.phases.preprocessing.outputs.preprocessed_recording` controls the relative destination path for that saved recording when `save_binary=true`.

- `<well_out_dir>/stg1_preprocess_outputs/`
  - `preprocessed_recording/` (SpikeInterface binary recording; optional but recommended)
  - `common_electrodes.npy`
  - `preprocess_config.json` (cache-safety: prevents accidental resume with different resampling settings)
  - `maxwell_contiguous_epochs_<stream_id>.json`
  - `concatenation_stitch_epochs_<stream_id>.json`
  - `channel_layouts/*.png` and other diagnostics (if enabled)

## Cache/resume behavior

`preprocess_for_spikesorting(...)` will resume from `preprocessed_recording/` when:

- the checkpoint indicates preprocessing is complete
- `overwrite_saved_recording=False`
- and `preprocess_config.json` (if present) matches the requested temporal resampling settings

If resampling is requested but no `preprocess_config.json` exists, preprocessing is re-run (to avoid silent mismatch).

## Debugging entrypoints

- Canonical CLI command: `python -m axon_reconstructor.cli stage preprocess --h5-path ... --stream-id ... --mea-output-root ...`
- Project-local script entrypoint (example): `projects/.../run_preprocess.py` calling package CLI/module APIs.

## Detailed stepwise docs

- Step 01: [methods/methods_preprocessing_step_01_setup_plan_and_output_layout.md](methods/methods_preprocessing_step_01_setup_plan_and_output_layout.md)
- Step 02: [methods/methods_preprocessing_step_02_common_electrodes_and_segment_normalization.md](methods/methods_preprocessing_step_02_common_electrodes_and_segment_normalization.md)
- Step 03: [methods/methods_preprocessing_step_03_concatenation_times_and_epoch_markers.md](methods/methods_preprocessing_step_03_concatenation_times_and_epoch_markers.md)
- Step 04: [methods/methods_preprocessing_step_04_temporal_resampling_and_epoch_scaling.md](methods/methods_preprocessing_step_04_temporal_resampling_and_epoch_scaling.md)
- Step 05: [methods/methods_preprocessing_step_05_persistence_resume_and_debugging.md](methods/methods_preprocessing_step_05_persistence_resume_and_debugging.md)

## Common failure modes

- **HDF5 compression plugin missing**: Maxwell recordings may require `libcompression.so`.
  - In many Shifter/Docker environments this is already configured; on the host environment you may need `HDF5_PLUGIN_PATH`.
- **Triggered/snippet recordings**: `get_num_samples()/fs` underestimates wall-clock span.
  - This is why we use `frame_nos` to build times and epoch markers.
- **Resampling edge artifacts**: if you see ringing near boundaries, increase `temporal_resample_margin_ms`.
