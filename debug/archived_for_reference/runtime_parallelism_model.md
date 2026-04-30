# Runtime Parallelism Model (Debug)

This document describes the runtime parallelism knobs used by `tools/debug/debug.runtime.yml`.

## Knobs

- `resources.max_workers`
  - Global runtime cap.
  - Must be `<= logical_cores_available` on the current machine.
  - The CLI throws an error if this is exceeded.
  - The CLI warns when this value is `>= 75%` of logical cores.

- `stages.<stage>.resources.max_stage_workers`
  - Per-stage cap.
  - Auto-clamped to `max_workers` with a warning if set higher.

- `stages.<stage>.resources.well_workers`
  - Number of wells run concurrently for that stage.
  - Auto-clamped to `max_stage_workers` with a warning if set higher.

## Derived Internal Parallelism

For stages that use SpikeInterface parallel job controls (`preprocess`, `spikesort`, `waveforms`, `templates`), the effective `n_jobs` is derived as:

`n_jobs = floor(max_stage_workers / well_workers)`

With a minimum of `1`.

This keeps stage-wide compute bounded so you do not accidentally oversubscribe with `well_workers * n_jobs`.

## Reconstruction Parallelism

`reconstruct` no longer uses a direct runtime `unit_workers` knob.

Instead, unit-level reconstruction workers are derived from the same stage formula:

`unit_workers = floor(max_stage_workers / well_workers)` (minimum `1`)

This keeps reconstruction bounded by stage-level limits and avoids having a separate independent worker knob.

## Practical Tuning

- CPU-heavy stage, one well at a time:
  - `max_stage_workers: 24`, `well_workers: 1` -> `n_jobs: 24`

- Multiple wells in parallel:
  - `max_stage_workers: 24`, `well_workers: 6` -> `n_jobs: 4`

- Memory-heavy reconstruction:
  - `max_stage_workers: 8`, `well_workers: 4` -> `unit_workers: 2`

## Notes

- Stage barriers remain unchanged for the `stages` command: each stage completes before the next begins.
- This file is in the debug folder for iteration and can be moved to formal docs later.
