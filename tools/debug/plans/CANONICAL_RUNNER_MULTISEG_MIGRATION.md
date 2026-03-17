# Canonical Runner + Multiseg Migration Plan

Status: MOSTLY COMPLETE (integration validation + stage-1 consolidation follow-ups pending)
Last updated: 2026-03-16

## Scope

This plan tracks migration work to:
- establish a canonical pipeline runner module in axon_reconstructor,
- migrate multisegment preprocessing policy into MEA_Analysis stage-1 preprocessing,
- enforce strict fail-hard topology guards for single-vs-multisegment expectations.

## Phase A - Canonical runner (axon_reconstructor)

- [x] Add canonical module `src/axon_reconstructor/pipeline/runner.py`.
- [x] Switch CLI imports to canonical runner module.
- [x] Keep backward compatibility by re-exporting existing pipeline_driver APIs.
- [x] Move implementation bodies from `pipeline_driver.py` into `runner.py`.
- [x] Reduce `pipeline_driver.py` to compatibility shim.

## Phase B - Multiseg preprocessing seam (MEA_Analysis)

- [x] Add `IPNAnalysis/multiseg_utils/preprocess_multiseg_h5/` package.
- [x] Add strict `expect_multisegment` fail-hard policy gate.
- [x] Add `multiseg_mode` support (`none`, `concatenate`).
- [x] Wire utility into `MEAPipeline.run_preprocessing()` immediately after recording load.
- [x] Add CLI/config plumbing for `expect_multisegment` + `multiseg_mode`.

## Phase B.1 - Debug entrypoint support

- [x] Add preprocessing-only runner script through canonical `stage preprocess` entrypoint.
- [x] Add config-driven python interpreter resolution in `tools/debug/run_mea_phase_preprocess.sh`.
- [x] Add config template entries in `tools/debug/debug.config.yml` for `python.executable_path` and `python.conda_env_name`.
- [x] Fix YAML parsing for nested `python.*` keys in preprocess-only runner wrapper.

## Phase C - Validation

- [x] Add focused tests for topology mismatch failures and concatenate mode.
- [x] Convert focused tests to stdlib `unittest` so validation does not require `pytest`.
- [x] Validate preprocess-only smoke run (`--force-restart`) with configured `axon_recon` interpreter.
- [ ] Add integration smoke run for `stage spikesort` + resume paths. (deferred)
- [ ] Document final configuration examples in docs.

## Phase D - Stg1 runtime ownership migration

- [x] Add MEA-owned minimized stg1 runtime module `IPNAnalysis/multiseg_utils/stg1_runtime.py`.
- [x] Export migrated stg1 runtime interfaces via `IPNAnalysis/multiseg_utils/__init__.py`.
- [x] Convert axon stg1 helper/runtime modules to compatibility wrappers that delegate to MEA-owned implementations:
	- `src/axon_reconstructor/pipeline/stg1_preprocessing/planning.py`
	- `src/axon_reconstructor/pipeline/stg1_preprocessing/concatenation.py`
	- `src/axon_reconstructor/pipeline/stg1_preprocessing/utils.py`
	- `src/axon_reconstructor/pipeline/stg1_preprocessing/h5_helpers.py`
	- `src/axon_reconstructor/pipeline/stg1_preprocessing/runner.py`
- [x] Validate preprocess-only end-to-end run against MEA-backed stg1 wrappers (`run_mea_phase_preprocess.sh --force-restart`).
- [x] Migrate/trim remaining diagnostics and plotting ownership (`stg1_preprocessing/plotting.py`) into MEA or remove as dead code.
- [x] Move stg1 checkpoint state ownership from axon wrapper into MEA stage scaffold.

## Phase D.1 - h5_utils test layout cleanup

- [x] Move `IPNAnalysis/h5_utils/test_plugin.py` into `IPNAnalysis/h5_utils/tests/test_plugin.py`.
- [x] Move `IPNAnalysis/h5_utils/test_streams.py` into `IPNAnalysis/h5_utils/tests/test_streams.py`.
- [x] Move `IPNAnalysis/h5_utils/test_timing.py` into `IPNAnalysis/h5_utils/tests/test_timing.py`.
- [x] Add `IPNAnalysis/h5_utils/tests/__init__.py`.
- [x] Update test invocation module paths to `IPNAnalysis.h5_utils.tests.*`.

## Phase E - Stage-1 MEA consolidation (spikesort/waveforms collapse)

Status: PLANNED
Updated: 2026-03-16

- [x] Create `IPNAnalysis/multiseg_utils/spikesort_multiseg_h5/` as optional multisegment spikesort mechanics package.
- [x] Create `IPNAnalysis/multiseg_utils/extract_multiseg_wfs/` as optional multisegment waveform extraction package.
- [x] Wire initial stage-1 no-op hook integration for optional multiseg spikesort/waveform planning in `mea_analysis_routine`.
- [x] Extract initial UnitMatch artifact-path + summary-sync mechanics into `multiseg_utils/spikesort_multiseg_h5` and consume from stage-1 merge flow.
- [x] Make UnitMatch output/throughput/report subdir selection phase-override-aware with default-safe fallback behavior.
- [x] Extract initial waveform sorting-source resolution helpers into `multiseg_utils/extract_multiseg_wfs` and wire analyzer migration hook metadata/logging.
- [x] Add default-off analyzer sorting-source loading options (`waveform_prefer_merged_sorting`, `waveform_merged_sorting_dir`) wired through MEA + axon config plumbing.
- [x] Extract epoch-interval filtering primitives into `multiseg_utils/extract_multiseg_wfs` and add focused unit tests.
- [x] Delegate axon stage-3 epoch helper callsites to MEA extracted implementations with local fallback.
- [x] Delegate axon stage-3 sorting-source resolve/load helper callsites to MEA extracted implementations with local fallback.
- [x] Delegate axon stage-3 preprocessed-recording loader callsite to MEA extracted implementation with local fallback.
- [x] Delegate axon stage-2 merged sorting artifact path callsite to MEA extracted UnitMatch path resolver with local fallback.
- [x] Propagate multiseg waveform/sorting + phase output override options through axon stage-2 debug config builder.
- [x] Delegate waveform epoch-marker path resolution (MEA stage-1 + axon stage-3) to shared MEA extracted helper with fallback.
- [x] Add focused unit tests for shared waveform epoch-marker path resolution helper.
- [x] Delegate stage-3 sorting cleanup + debug-unit limiting mechanics to shared MEA waveform helper with local fallback.
- [x] Add focused unit tests for shared waveform sorting harmonization helper.
- [x] Extract phase output coercion/resolution/build mechanics from `mea_analysis_routine` into new `IPNAnalysis/phase_path_utils` module and delegate callsites.
- [x] Remove non-default waveform/report/curation artifact defaults from resolved phase-path map; include them only when explicitly configured.
- [x] Enforce explicit multiseg-only phase output paths when `multiseg_mode` is enabled (missing required paths now error early).
- [x] Simplify `mea_analysis_routine` internal multiseg execution mode usage to a single `multiseg_mode` signal.
- [ ] Move mechanical stage logic from axon `stg2_spikesorting` into MEA stage-1 sorting/merge hooks where multisegment behavior is required.
- [ ] Move mechanical stage logic from axon `stg3_waveforms` into MEA stage-1 analyzer hooks where multisegment behavior is required.
- [ ] Remove standalone spikesort/waveforms stage behavior (no compatibility aliases).

## Phase E.1 - Debug config rewiring (stage-1 phase-centric)

- [x] Reframe debug config to make MEA stage-1 phases explicit: preprocessing, spikesorting, merge, analyzer, reports, curation.
- [x] Add explicit relative output path controls per phase and reports artifact group.
- [x] Keep all phase artifact outputs relative to the overall MEA stage output directory.
- [x] Emit resolved phase output paths in logs/checkpoints for debug traceability.
- [x] Remove standalone `stages.preprocess`, `stages.spikesort`, and `stages.waveforms` config blocks from debug config.
- [x] Rewire CLI stage config lookups to `stages.mea_analysis.phases.*` + `stages.mea_analysis.resources.*` for spikesort/waveforms settings.
- [x] Remove deprecated per-subphase multiseg CLI plumbing (`expect_multisegment`, `multiseg_spikesort_mode`, `multiseg_waveforms_mode`) in favor of `multiseg_mode`.
- [ ] Validate resume-from-merge and phase path contracts with integration smoke runs.
