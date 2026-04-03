# Resume + checkpoint inventory (Phase 3.5.3 follow-up)

This document inventories resume/restart behavior from CLI args down to stage runtime and checkpoint persistence.

## 1) CLI resume/restart surface

| Command | Resume/restart args | Effective source | Notes |
|---|---|---|---|
| `axon-reconstructor run` | `--force-restart`, `--no-checkpoint` | CLI flags only | Routes through `AxonReconstructor` driver path. |
| `axon-reconstructor pipeline` | `--force-restart`, `--no-checkpoint` | CLI flags only | Primarily preprocess-focused path. |
| `axon-reconstructor stage <stage>` | `--force-restart` | CLI → env fallback (`AXON_RECON_FORCE_RESTART`) | Canonical stage entrypoint for all stages. |
| `axon-reconstructor analysis-deck` | `--force-restart` | CLI → env fallback (`AXON_RECON_FORCE_RESTART`) | Calls analysis stage first, then deck rendering. |
| `axon-reconstructor scope-run` | *(none directly)* | Config payload | Uses `scope_config.force_restart` parsed from config file. |
| `axon-reconstructor scope-config-build` | `--force-restart` | CLI → env fallback (`AXON_RECON_FORCE_RESTART`) | Writes `force_restart` into generated scope config. |

Primary arg definitions:
- `src/axon_reconstructor/cli.py`
- `src/axon_reconstructor/pipeline/pipeline_driver.py`

## 2) Stage runtime checkpointing + resume logic

| Stage | Checkpoint file | Checkpoint transitions | Resume gate (non-force) | Status |
|---|---|---|---|---|
| preprocess | main checkpoint (`*_checkpoint.json`) | `PREPROCESSING` -> `PREPROCESSING_COMPLETE` (+ failure) | Requires checkpoint stage complete + saved recording + `common_electrodes.npy` + config parity | Covered |
| spikesort | MEA_Analysis checkpoint under `stg2_spikesorting_outputs/checkpoints` | MEA pipeline internal stages | Resume controlled by MEA_Analysis checkpoint/state; no axon stage-checkpoint wrapper yet | Partial |
| waveforms | stage checkpoint (`*_waveforms_checkpoint.json`) | `ANALYZER` -> `ANALYZER_COMPLETE` (+ failure) | `concat_waveforms_dir` existence shortcut, optional replot pass | Covered |
| templates | stage checkpoint (`*_templates_checkpoint.json`) | `ANALYZER` -> `ANALYZER_COMPLETE` | `extracted_templates_dir` + `templates_summary.json` (+ grid if enabled) | Mostly covered |
| reconstruct | stage checkpoint (`*_reconstruction_checkpoint.json`) | `ANALYZER` -> `ANALYZER_COMPLETE` | summary + by-unit outputs (+ overview if enabled), disabled in replot/branches-only modes | Mostly covered |
| analysis | stage checkpoint (`*_analysis_checkpoint.json`) | `REPORTS` -> `REPORTS_COMPLETE` (+ failure) | No top-level stage early-return; per-artifact skip behavior controlled by `force_restart` | Covered (no global early-return) |

Primary stage files:
- `src/axon_reconstructor/pipeline/stg1_preprocessing/main.py` (preprocess)
- `src/axon_reconstructor/pipeline/stg2_spikesorting/runner.py`
- `src/axon_reconstructor/pipeline/stg3_waveforms/runner.py`
- `src/axon_reconstructor/pipeline/stg4_templates/runner.py`
- `src/axon_reconstructor/pipeline/stg5_reconstruction/runner.py`
- `src/axon_reconstructor/pipeline/stg6_analysis/runner.py`
- `src/axon_reconstructor/pipeline/checkpointing.py`

## 3) Scope orchestration behavior

`scope-run` (`src/axon_reconstructor/pipeline/pipeline_driver.py`) propagates `force_restart` from config to stage inputs, and maintains a barrier checkpoint artifact for stage-level progress. Transition gates (`unit_match`, `merge_update`) are readiness-gate checks.

## 4) Low-hanging fruit (obvious + easy)

1. **[DONE] Spikesort stage-checkpoint wrapper in axon runtime (easy-medium)**
   - Add a dedicated axon stage checkpoint (`*_spikesort_checkpoint.json`) around `run_spikesorting_stage`.
   - Keep MEA_Analysis checkpointing intact; this is additive for consistency with other stages.
   - Wire transitions: started (`SORTING`) -> completed (`SORTING_COMPLETE`/`ANALYZER_COMPLETE`/`REPORTS_COMPLETE`) + failed.

2. **[DONE] Failure checkpoint parity in templates/reconstruct (easy)**
   - Add stage-level failure writes via `save_stage_failed(...)` on unhandled exceptions.
   - `analysis`/`waveforms` already do this; templates/reconstruct should match.

3. **[DONE] Optional analysis global resume shortcut (easy)**
   - Add early return in analysis when summary + by-unit outputs are complete and `force_restart=False`.
   - Preserve current per-artifact skip behavior for partial reruns.

4. **[DONE] Optional analysis-deck checkpoint (easy)**
   - If desired, add tiny command-level checkpoint/state artifact for deck generation (`deck_started/deck_complete`), separate from stage analysis checkpoint.

5. **[DONE] Orchestrator barrier-level resume artifact (medium)**
   - Add optional scope-run barrier checkpoint file (completed stages + failed targets) to resume long multi-dataset runs without restarting from stage 1.

## 5) Suggested implementation order

1. Spikesort stage checkpoint wrapper.
2. Templates/reconstruct failure checkpoint parity.
3. Analysis global resume shortcut.
4. (Optional) analysis-deck checkpoint.
5. (Optional) scope-run barrier checkpoint.

## 6) Validation coverage

Added in this phase:
- `tests/test_stage_checkpointing.py`
- `tests/test_stage_orchestrator_runtime.py`

Adjacent regression checks run:
- `tests/test_checkpointing.py`
- `tests/test_pipeline_logging.py`
