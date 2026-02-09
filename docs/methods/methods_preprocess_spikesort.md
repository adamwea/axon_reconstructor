# Methods: Raw preprocessing + spikesorting (WIP / legacy)

This document is kept for historical context.

For the current stepwise documentation (with ordered `stage_XX_` filenames), start here:

- [../stage_01_preprocessing.md](../stage_01_preprocessing.md)
- [../stage_02_spikesorting.md](../stage_02_spikesorting.md)

This document describes the **scientific / data-processing methods** used by `axon_reconstructor` to prepare Maxwell recordings for spikesorting and to interface with MEA_Analysis for GPU sorting.

## Scope

- **Input**: Maxwell `.raw.h5` recordings
- **Goal**: create a stable, reproducible recording representation suitable for spikesorting, despite recordings being split into multiple segments/configurations.
- **Current milestone**: rebuild and improve the steps that
  1) interpret per-segment configuration / channel-location metadata (often via sibling `.cfg` files)
  2) determine the set of channels shared across all segments
  3) concatenate the segments into one continuous recording for spikesorting

## Preprocessing: shared-channel intersection

### Why intersection?

In axon tracking experiments, a single “recording” can be split into multiple acquisition segments (or “configs”). Channel availability and/or ordering may differ across segments. Many downstream algorithms (spikesorting, template extraction, axon recon) assume a consistent channel set.

A conservative and reproducible approach is to compute the **intersection** of electrodes available across all segments and restrict analysis to those shared channels.

### How we compute shared channels (current implementation)

The rebuild currently uses SpikeInterface’s Maxwell extractor metadata:

- Each segment is loaded via `spikeinterface.full.MaxwellRecordingExtractor(..., rec_name=...)`.
- We read `contact_vector["electrode"]` and treat those electrode ids as the channel identity.
- The shared set is the set-intersection across all segments.

This logic lives in:

- `axon_reconstructor.pipeline.raw_preprocessing.find_common_electrodes_from_segments`

### `.cfg` files

Some datasets provide `.cfg` files adjacent to the `.raw.h5` that describe channel locations.

- The rebuild treats these as **optional** inputs.
- We currently *discover* them and store them in a preprocessing plan, but the schema-aware parser is still pending because we need to inspect real `.cfg` examples from the target datasets.

Relevant code:

- `axon_reconstructor.pipeline.raw_preprocessing.discover_cfg_files`
- `axon_reconstructor.pipeline.raw_preprocessing.parse_cfg_channel_locations` (conservative placeholder)

## Preprocessing: concatenation

### Steps

Once we have the shared electrode set, we build a concatenated recording:

1) Load each segment for a given well/stream.
2) Apply per-segment centering (`spikeinterface.full.center`) to remove DC offsets (a standard conditioning step for sorting).
3) Slice each segment down to the shared electrode set.
4) Concatenate the segments using `spikeinterface.full.concatenate_recordings`.

Relevant code:

- `axon_reconstructor.pipeline.raw_preprocessing.build_concatenated_recording`

### Notes / assumptions

- **Centering window**: we use a bounded chunk size (default 10k samples) to keep preprocessing stable and fast.
- **Concurrency**: segments are processed with a small thread pool because extraction is I/O heavy.

## Spikesorting execution model (NERSC)

On Perlmutter, sorting is performed via MEA_Analysis inside Shifter. `axon_reconstructor` does **not** attempt to run Kilosort directly on the host environment.

The user-facing path is:

- request an interactive GPU allocation
- run MEA_Analysis’s `IPNAnalysis/run_pipeline_driver.py` inside Shifter
- write all outputs under a configured MEA output root

The CLI `axon-reconstructor gpu-interact` is currently the canonical entry point.

## Known gaps / next steps

- Inspect real `.cfg` files and implement a schema-aware parser that yields electrode ids + (x,y) locations.
- Define a stable mapping between `.cfg` files and segments (`rec_name`) if the dataset requires it.
- Decide how/where to persist the preprocessing plan (JSON/TOML) alongside MEA outputs for full reproducibility.
