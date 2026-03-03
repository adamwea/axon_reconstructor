# Radivojevic 2023 Strategy: Step-by-Step Process

This document describes the current implementation in:
- src/axon_reconstructor/pipeline/stg5_reconstruction/strategies/radivojevic_2023.py

It is intended for review before wiring this strategy into the active reconstruction runner.

## Scope and status

- Strategy name: Radivojevic2023Reconstructor
- Status: standalone module, not wired into reconstruction runner
- Input compatibility: full-template unit artifacts (preferred), merged contributing templates (compatibility)
- Core algorithm stages implemented:
  1) Adaptive thresholding (3 steps)
  2) Trajectory tracking (3 steps)
  3) Velocity-consistency filtering

## Paper-aligned constants used

From the 2023 eLife description, current defaults are:

- Frame interval: 50 microseconds
- Expected frame count (strict mode): 400

Adaptive thresholding:
- Step 1 threshold: 9 x noise std
- Step 2 threshold: 2 x noise std, spatial radius 50 micrometers, temporal window t-1, t, t+1
- Step 3 threshold: 1 x noise std, spatial radius 100 micrometers, temporal window t-1, t, t+1

Tracking:
- Step 1 direct linking max distance: 100 micrometers
- Step 2 skeleton-assisted max distance: 200 micrometers
- Step 3 indirect every-other-frame max distance: 400 micrometers

Velocity filtering:
- Reject links deviating more than 50% from median previously estimated velocity

## Input pathways

### A) Precomputed dV/dt frames

Provide Radivojevic2023Input with:
- dvdt_frames_uv_per_us: array shape [n_frames, n_channels]
- channel_locations_um: array shape [n_channels, 2]
- noise_std_uv_per_us: positive scalar
- sampling_frequency_hz
- source label

### B) Build input from full-template artifacts (preferred)

The helper make_input_from_full_template reads from one unit folder:
- full_template.npy
- full_channel_locations_xy.npy
- full_template_meta.json
- full_channel_ids.npy (if present)
- contributing_full_channel_indices.npy (if present)

Pipeline in helper:
1. Load full template and channel geometry.
2. Keep informative/contributing channels (full template is zero-padded on non-contributing channels).
3. Read sampling_frequency_hz from meta JSON.
4. Compute time derivative dV/dt in microvolt per microsecond via finite difference.
5. Convert derivative samples to 50 microsecond frames by block reduction (minimum per block).
6. Estimate noise std from derivative signal unless user supplies one (optional quiet-period path available).

### C) Build input from merged contributing template artifacts (compatibility)

The helper make_input_from_merged_contributing reads from one unit folder:
- merged_contributing_template.npy
- merged_contributing_channel_locations.npy
- merged_contributing_template_meta.json

Pipeline in helper:
1. Load template samples and channel locations.
2. Read sampling_frequency_hz from meta JSON.
3. Compute time derivative dV/dt in microvolt per microsecond via finite difference.
4. Convert derivative samples to 50 microsecond frames by block reduction (minimum per block).
5. Estimate noise std from derivative signal (robust MAD-based estimator) unless user supplies one.

## Full run flow

Main call:
- Radivojevic2023Reconstructor.run(inputs)

Internal sequence:

1. Validate dimensions and values
   - frames must be 2D
   - locations must be [n_channels, 2]
   - channels must match between frames and locations
   - noise std must be finite and positive
   - optional strict mode enforces exactly 400 frames

2. Precompute pairwise channel distances
   - Used for local minima checks and confined neighborhood queries

3. Adaptive thresholding

   3.1 Step 1: global high-threshold detection
   - Threshold = -9 x noise std (negative peak polarity)
   - Candidate channels per frame are those below threshold
   - Keep only local minima within local_peak_radius_um neighborhood

   3.2 Step 2: confined lower-threshold detection
   - Threshold = -2 x noise std
   - Use peaks from step 1 as seeds
   - Search around seed channels within 50 micrometers
   - Search temporally across seed frame neighbors t-1, t, t+1
   - Keep local minima only

   3.3 Merge step 1 and step 2 peaks
   - Deduplicate by (frame, channel)
   - Prefer detections from later step on collisions
   - Tie-break by more negative dV/dt

   3.4 Step 3: broader confined low-threshold detection
   - Threshold = -1 x noise std
   - Use merged peaks from prior stage as seeds
   - Same temporal window, broader radius 100 micrometers
   - Keep local minima only

   3.5 Final peak merge
   - Same dedup rules
   - Reindex peak_id sequentially

4. Three-phase trajectory tracking

   4.1 Phase 1: direct interconnection
   - Consecutive frames only
   - Link closest unmatched destination within 100 micrometers
   - Compute velocity from distance and frame interval
   - Apply velocity consistency check

   4.2 Phase 2: skeleton-assisted interconnection
   - Consecutive frames, up to 200 micrometers
   - Build support image from 2-frame average at low support threshold
   - Interpolate to raster map, apply binary morphology, extract skeleton
   - Require connected skeleton path between candidate endpoints
   - Apply velocity consistency check

   4.3 Phase 3: indirect interconnection
   - Every other frame (frame t to t+2), up to 400 micrometers
   - Build support image from 3-frame average
   - Interpolate to raster map, apply binary morphology, extract skeleton
   - Require connected skeleton path between candidate endpoints
   - Compute velocity using 2-frame time delta
   - Store predicted midpoint coordinates for middle frame
   - Apply velocity consistency check

5. Build trajectories
   - Convert directed links into path lists starting from nodes with zero indegree
   - Enumerate branches via DFS-style expansion
   - Sort trajectories by length descending

6. Summarize result
   - Collect peaks, links, trajectories
   - Compute median link velocity over positive finite velocities
   - Return Radivojevic2023Result

## Output schema

Radivojevic2023Result includes:
- params
- n_frames
- n_channels
- peaks list (DetectedPeak)
- links list (PeakLink)
- trajectories list of peak_id paths
- velocity_m_per_s_median

Serialization helper:
- to_jsonable() returns plain dict/list structures for JSON export

## Important implementation notes for review

1. Negative-polarity assumption
- Peak detection currently assumes axonal signal peaks correspond to negative dV/dt extrema.

2. Noise estimation source
- If not provided, noise std is estimated from the available derivative signal using robust MAD scaling.
- Optional quiet-period estimation is available from spike-free windows in preprocessed concat recording.

3. Skeletonization mode
- The skeleton-assisted step uses interpolated image maps with morphology + skeleton extraction and connectivity checks.

4. Multi-link constraints
- Current linking is greedy with unmatched source and destination constraints per tracking stage.

5. Not wired into runner
- No active reconstruction stage calls this strategy yet.

## Minimal usage example

Use in ad hoc experiments:

1. Instantiate parameters and reconstructor.
2. Build input from one full-template unit directory.
3. Run strategy.
4. Save result via to_jsonable.

Pseudo-steps:
- recon = Radivojevic2023Reconstructor(Radivojevic2023Params())
- inp = recon.make_input_from_full_template(full_unit_dir=...)
- res = recon.run(inp)
- json.dump(res.to_jsonable(), ...)

## Files relevant to this strategy

- src/axon_reconstructor/pipeline/stg5_reconstruction/strategies/radivojevic_2023.py
- src/axon_reconstructor/pipeline/stg5_reconstruction/strategies/__init__.py
- src/axon_reconstructor/pipeline/stg5_reconstruction/README.md
