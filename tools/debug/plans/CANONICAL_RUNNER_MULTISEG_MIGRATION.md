# Canonical Runner + Multiseg Migration Plan

Status: IN PROGRESS (spikesort resume validation deferred)
Last updated: 2026-03-15

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

- [x] Add `IPNAnalysis/multiseg_utils/preprocess_multiseg_h5.py`.
- [x] Add strict `expect_multisegment` fail-hard policy gate.
- [x] Add `multiseg_mode` support (`none`, `concatenate`).
- [x] Wire utility into `MEAPipeline.run_preprocessing()` immediately after recording load.
- [x] Add CLI/config plumbing for `expect_multisegment` + `multiseg_mode`.

## Phase B.1 - Debug entrypoint support

- [x] Add preprocessing-only runner script through canonical `stage preprocess` entrypoint.
- [x] Add config-driven python interpreter resolution in `tools/debug/run_preprocess_only_stage.sh`.
- [x] Add config template entries in `tools/debug/debug.config.yml` for `python.executable_path` and `python.conda_env_name`.
- [x] Fix YAML parsing for nested `python.*` keys in preprocess-only runner wrapper.

## Phase C - Validation

- [x] Add focused tests for topology mismatch failures and concatenate mode.
- [x] Convert focused tests to stdlib `unittest` so validation does not require `pytest`.
- [x] Validate preprocess-only smoke run (`--force-restart`) with configured `axon_recon` interpreter.
- [ ] Add integration smoke run for `stage spikesort` + resume paths. (deferred)
- [ ] Document final configuration examples in docs.

## Phase D - Stg1 runtime ownership migration (in progress)

- [x] Add MEA-owned minimized stg1 runtime module `IPNAnalysis/multiseg_utils/stg1_runtime.py`.
- [x] Export migrated stg1 runtime interfaces via `IPNAnalysis/multiseg_utils/__init__.py`.
- [x] Convert axon stg1 helper/runtime modules to compatibility wrappers that delegate to MEA-owned implementations:
	- `src/axon_reconstructor/pipeline/stg1_preprocessing/planning.py`
	- `src/axon_reconstructor/pipeline/stg1_preprocessing/concatenation.py`
	- `src/axon_reconstructor/pipeline/stg1_preprocessing/utils.py`
	- `src/axon_reconstructor/pipeline/stg1_preprocessing/h5_helpers.py`
	- `src/axon_reconstructor/pipeline/stg1_preprocessing/runner.py`
- [x] Validate preprocess-only end-to-end run against MEA-backed stg1 wrappers (`run_preprocess_only_stage.sh --force-restart`).
- [x] Migrate/trim remaining diagnostics and plotting ownership (`stg1_preprocessing/plotting.py`) into MEA or remove as dead code.
- [x] Move stg1 checkpoint state ownership from axon wrapper into MEA stage scaffold.
