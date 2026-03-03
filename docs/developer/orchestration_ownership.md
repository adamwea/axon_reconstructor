# Orchestration Ownership (Phase 3.7)

This note defines canonical ownership after orchestration consolidation.

## Canonical orchestration entrypoints

- `axon-reconstructor scope-run`
  - Owns multi-target stage-barrier orchestration across datasets/wells.
  - Uses `run_scope_stage_barriers(...)` in `src/axon_reconstructor/pipeline/pipeline_driver.py`.

- `axon-reconstructor stage <stage>`
  - Owns single-target stage execution.
  - Resolves CLI/env args and dispatches via shared stage executor.

## Shared execution registry

- `src/axon_reconstructor/pipeline/pipeline_driver.py`
  - Canonical stage dispatch for preprocess/spikesort/waveforms/templates/reconstruct/analysis.
  - Canonical scope stage-barrier orchestration runtime for `scope-run`.
  - Canonical stage CLI argument registration helpers.
  - Called by both `stage` CLI and `scope-run` orchestration flow.

## Stage service modules (implementation ownership)

- Preprocess service:
  - `run_preprocess_stage(...)` in `pipeline/stg1_preprocessing/main.py`.
- Spikesort service:
  - `run_spikesorting_stage(...)` in `pipeline/stg2_spikesorting/runner.py`.
- Waveforms service:
  - `extract_waveforms(...)` in `pipeline/stg3_waveforms/runner.py`.
- Templates service:
  - `extract_and_merge_templates(...)` in `pipeline/stg4_templates/runner.py`.
- Reconstruction service:
  - `reconstruct_from_templates(...)` in `pipeline/stg5_reconstruction/runner.py`.
- Analysis service:
  - `analyze_units(...)` in `pipeline/stg6_analysis/runner.py`.

## Deprecated/removed orchestration ownership

- Legacy orchestration via `pipeline_driver.run_pipeline(...)` is removed.
- Legacy preprocess owner `pipeline_driver.AxonReconstructor.preprocess_for_spikesorting(...)` is removed.
- Deprecated top-level CLI orchestration surfaces (`run`, `pipeline`) are removed in Phase 3.7 cleanup.

## Test parity coverage

- `tests/test_stage_execution_parity.py`
  - verifies scope orchestrator and stage CLI both dispatch through shared stage executor.
- `tests/test_stage_orchestrator_runtime.py`
  - verifies barrier semantics (`fail_fast`, transition gates).
