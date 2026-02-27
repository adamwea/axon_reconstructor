# Preprocessing Step — Part 5: Persistence, Cache Safety, and Debugging Entry Points

Scope: this document covers how preprocessing artifacts are saved, how cache-safety works for resampling, and how to debug preprocessing in isolation.

Primary code path:
- `axon_reconstructor.pipeline.pipeline_driver.AxonReconstructor.preprocess_for_spikesorting(...)`

Related scripts:
- `tools/debug/debug_preprocessing_step.py`
- project-local wrappers (example): `projects/.../debug_preprocessing_step.py`

---

## 1. Saved recording persistence

When `save_recording=True` and `mea_analysis_output_root` is configured, preprocessing saves the concatenated recording into:

- `<well_out_dir>/preprocess_outputs/preprocessed_recording/`

This saved recording is the contract used by spikesorting debug harnesses and waveforms stage.

---

## 2. Cache-safety via `preprocess_config.json`

Problem:
- it’s easy to accidentally “resume” from a cached `preprocessed_recording/` that was produced with different temporal resampling settings.

Solution:
- preprocessing writes a small config file:

  - `<well_out_dir>/preprocess_outputs/preprocess_config.json`

It stores the “requested resampling config” (factor/rate/margin/dtype).

Resume behavior:

- If `overwrite_saved_recording=False` and a saved recording exists, the driver checks that the saved config matches the requested config.
- If resampling is requested but the config file is missing, preprocessing is forced to re-run.

This keeps downstream sample-coordinate assumptions stable.

---

## 3. Epoch JSON persistence (downstream contract)

Preprocessing writes:

- `maxwell_contiguous_epochs_<stream_id>.json`
- `concatenation_stitch_epochs_<stream_id>.json`

These files are read later by waveforms stage (`filter_by_maxwell_epochs`) to exclude spikes too close to snippet boundaries.

---

## 4. Diagnostics / plotting outputs

If diagnostics are enabled (via `plot_layouts=True` and output dirs are available), preprocessing can write:

- channel layout plots
- concatenation representative traces
- per-segment trace plots

These are intended as sanity checks and do not affect downstream correctness.

---

## 5. Debugging entry points

Two common ways to debug preprocessing:

1. Repo-local harness:
   - `tools/debug/debug_preprocessing_step.py`

2. Project-local wrapper scripts (convenience for specific datasets):
   - `projects/.../debug_preprocessing_step.py`

These are designed to run preprocessing as an isolated step and emit artifacts into the same per-well output folder that later stages consume.
