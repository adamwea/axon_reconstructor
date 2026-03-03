# Preprocessing Step — Part 1: Setup, Planning, and Output Layout

Scope: this document covers how preprocessing is *invoked* and how it decides where outputs go. It does **not** cover the detailed signal/time/epoch logic (those are in later parts).

Primary code paths:
- User entry: `axon_reconstructor.pipeline.pipeline_driver.AxonReconstructor.preprocess_for_spikesorting(...)`
- Core builder: `axon_reconstructor.pipeline.preprocessing.runner.build_concatenated_recording(...)`

Related modules:
- `axon_reconstructor.pipeline.preprocessing.planning` (cfg discovery / plan object)
- `axon_reconstructor.pipeline.pipeline_driver` (per-well output dir contract, cache-safety)
- `axon_reconstructor.pipeline.checkpointing` (checkpoint read/write)
- `axon_reconstructor.pipeline.pipeline_logging` (per-well log)

---

## 0. Stage intent (contract)

Preprocessing turns a Maxwell `.raw.h5` **stream/well** (e.g. `well000`) into a single SpikeInterface `Recording` that downstream stages can treat as:

- having a **stable channel identity** (channel ids correspond to physical electrode ids)
- having a **consistent channel set across segments** (via electrode intersection)
- being aligned to explicit **epoch markers** that capture snippet boundaries and concatenation stitch points
- optionally being **temporally upsampled** (temporal resampling / interpolation) in a way that keeps epoch markers consistent

---

## 1. Entry point: `preprocess_for_spikesorting(...)`

1. The driver constructs a `RawPreprocessPlan` via:
   - `raw_preprocessing.build_preprocess_plan(h5_path=..., stream_id=...)`

2. It attempts to compute a per-well output folder using the MEA_Analysis-style convention:
   - `_compute_mea_analysis_output_dir(output_root=<mea_output_root>, data_file=<h5_path>, well=<stream_id>)`

3. If a per-well output dir exists (and we are going to emit artifacts there), the driver configures a per-well log file:
   - `compute_pipeline_log_file(well_out_dir=..., data_file=..., stream_id=...)`
   - `setup_pipeline_logger(...)`

---

## 2. Output directory layout

When `mea_analysis_output_root` is set, preprocessing writes into:

- `<well_out_dir>/stg1_preprocess_outputs/`

Key artifacts in that folder:

- `preprocessed_recording/`
  - SpikeInterface saved recording folder (binary format)
- `common_electrodes.npy`
  - the intersection electrode id list used for all segments
- `maxwell_contiguous_epochs_<stream_id>.json`
  - contiguous sample runs inside triggered/snippet recordings (concat sample coordinates)
- `concatenation_stitch_epochs_<stream_id>.json`
  - segment stitch boundaries (concat sample coordinates)
- `preprocess_config.json`
  - cache-safety: persists the caller’s requested temporal resampling settings
- Optional diagnostics (when enabled):
  - `channel_layouts/*.png`
  - `concat_cluster_reps_<stream_id>.png`
  - `segment_traces/*.png`

---

## 3. Checkpointing + resume logic

Preprocessing uses the main pipeline checkpoint file (per well):

- `compute_checkpoint_file(output_dir=well_out_dir, file_path=h5_path, stream_id=stream_id)`

Resume shortcut (filesystem-based) can happen when all of the following are true:

- the checkpoint stage indicates preprocessing is complete
- `overwrite_saved_recording=False`
- `<well>/stg1_preprocess_outputs/preprocessed_recording/` exists
- `<well>/stg1_preprocess_outputs/common_electrodes.npy` exists

Additionally, cache-safety for temporal resampling:

- If `<well>/stg1_preprocess_outputs/preprocess_config.json` exists, it must match the requested resampling options.
- If resampling is requested but `preprocess_config.json` is missing, preprocessing re-runs (to avoid silently mixing cached outputs with incompatible sample coordinates).

---

## 4. Handoff to the core builder

If preprocessing does not resume, `preprocess_for_spikesorting(...)` calls:

- `raw_preprocessing.build_concatenated_recording(...)`

and passes:

- `plot_output_dir=<well>/stg1_preprocess_outputs/` (when diagnostics are enabled)
- `epoch_markers_output_dir=<well>/stg1_preprocess_outputs/`
- any temporal resampling parameters (see Part 4)

The returned `Recording` + common electrode list are then optionally persisted to disk under `preprocessed_recording/`.
