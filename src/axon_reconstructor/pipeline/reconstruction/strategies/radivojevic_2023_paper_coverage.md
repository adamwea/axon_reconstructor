# Radivojevic 2023 Paper Coverage Checklist

This checklist maps method claims from:
- Radivojevic & Rostedt Punga, eLife 2023;12:e86512

to the current implementation in:
- src/axon_reconstructor/pipeline/reconstruction/strategies/radivojevic_2023.py

Legend:
- IMPLEMENTED: directly represented in code
- APPROXIMATED: represented, but not with the exact procedure described in paper
- PENDING: not implemented yet

## A) Inputs and signal framing

- [x] IMPLEMENTED: Uses dV/dt electrical images for detection.
  - Code: `_time_derivative_uv_per_us`, `_frame_from_derivative`

- [x] IMPLEMENTED: 50 us frame interval.
  - Code: `Radivojevic2023Params.interframe_us = 50.0`

- [x] IMPLEMENTED: 400-frame expectation available.
  - Code: `n_timeframes_expected = 400`, optional enforcement via `strict_paper_mode`

- [x] IMPLEMENTED: Optional noise estimated from inactive periods for each neuron.
  - Code: `_estimate_quiet_noise_std_uv_per_us(...)` with `use_quiet_period_noise_estimation=True`
  - Behavior: samples spike-free windows (unit-specific by default; configurable to all-units exclusion), computes channelwise dV/dt noise, and uses median channel sigma.
  - Fallback: default path remains robust estimate from available derivative signal (`_robust_noise_std`) when quiet-period mode is disabled.

## B) Adaptive thresholding (Figure 3 flow)

- [x] IMPLEMENTED: Step 1 global threshold at 9 STD.
  - Code: `_detect_global_step(... threshold_std=threshold_std_step1)`

- [x] IMPLEMENTED: Step 2 threshold at 2 STD, spatial confinement 50 um, temporal t-1/t/t+1.
  - Code: `_detect_confined_step(...)`, params `threshold_std_step2`, `confined_radius_step2_um`, `temporal_neighbor_half_window_frames=1`

- [x] IMPLEMENTED: Step 3 threshold at 1 STD, broader spatial confinement 100 um.
  - Code: `_detect_confined_step(...)`, params `threshold_std_step3`, `confined_radius_step3_um`

- [x] IMPLEMENTED: Greedy progression from high-amplitude to lower-amplitude detections.
  - Code: step order in `run()`, plus `_merge_peaks` between stages

- [ ] APPROXIMATED: Exact local peak logic used in original Matlab implementation.
  - Current: local minimum in radius-limited channel neighborhood (`_is_local_minimum`).
  - Gap: potential differences from original peak operator and neighborhood topology.

## C) Three-step tracking (Figure 4 flow)

- [x] IMPLEMENTED: Step I direct interconnection, consecutive frames, 100 um range.
  - Code: `_track_three_steps`, `link_kind='direct'`, `direct_link_max_distance_um=100`

- [x] IMPLEMENTED: Step II skeletonization-assisted interconnection, consecutive frames, 200 um range.
  - Code: `_track_three_steps`, `link_kind='skeleton_assisted'`, `skeleton_link_max_distance_um=200`

- [x] IMPLEMENTED: Step III indirect interconnection over every other frame, 400 um range.
  - Code: `_track_three_steps`, `link_kind='indirect'`, `dt_frames=2`, `indirect_link_max_distance_um=400`

- [x] IMPLEMENTED: Middle-frame prediction artifact stored for indirect links.
  - Code: `predicted_midpoint_xy_um` in `PeakLink`

- [x] IMPLEMENTED: Velocity consistency rejection at >50% deviation.
  - Code: `_velocity_ok`, `max_velocity_deviation_fraction=0.5`

- [ ] APPROXIMATED: True image skeletonization from 2-frame / 3-frame averaged maps.
  - Current: graph-based support path check on thresholded support nodes (`_support_indices`, `_support_path_exists`).
  - Gap: no explicit binary morphology + skeleton image operations equivalent to paper figure workflow.

- [ ] APPROXIMATED: “No false links” operating point reported in Figure 6.
  - Current: algorithmic constraints are present, but no evaluation harness reproducing paper benchmark yet.

## D) Validation and performance reporting (Figure 5/6 concepts)

- [ ] PENDING: Bayes optimal template-matching ground truth validation (>70% match criteria).
  - Needed: BOTM/single-trial matching evaluation pipeline connected to detected peaks.

- [ ] PENDING: Reproduce reported detection percentages (45/74/98%) and tracking percentages (70/85/91%).
  - Needed: benchmarking scripts and matched datasets/protocols.

- [ ] PENDING: Reproduce stimulation-direction validation workflow used to confirm trajectories.
  - Needed: stimulation metadata integration + dedicated analysis routine.

## E) Integration status

- [x] IMPLEMENTED: Strategy is packaged and importable.
  - Files: `strategies/radivojevic_2023.py`, `strategies/__init__.py`

- [x] IMPLEMENTED: Intentionally not wired into active reconstruction runner.
  - Safety: no impact on current production/batch runs.

- [ ] PENDING: Add strategy selection interface in reconstruction runner.
  - Suggested future: `ReconstructionInputs.reconstruction_strategy` with default `axon_velocity`.

## F) Priority gap-closing plan (recommended)

1. Implement paper-faithful skeletonization path extraction
   - Replace graph-support approximation in step II/III with explicit map thresholding + skeleton extraction.
2. Verify inactive-window noise estimation fidelity
  - Compare quiet-window estimator outputs against BOTM validation noise stats on the same units.
3. Add BOTM-based peak truth evaluation module
   - Compute true/false peak labels and detection metrics using >70% matching rule.
4. Add benchmark script
   - Emit paper-style coverage metrics for thresholding and tracking phases.

## Reviewer quick summary

Current implementation is method-structured and parameter-faithful for the main thresholds/distances/timing, with one major area still approximation-level versus the paper’s exact workflow:
- skeletonization procedure details

Validation sections (BOTM ground truth and reported percentages) are still pending.
