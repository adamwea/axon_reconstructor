# Spikesorting Stage Simplification Plan

## Goal
Reduce the custom wrapper in axon_reconstructor stage 2 and move control points into MEA_Analysis so integration is more robust to MEA_Analysis updates.

## Direct Answers To The Five Points

### 1) Can we keep full control without such a large wrapper?
Short answer: Mostly yes.

Confirmed from MEA_Analysis:
- MEAPipeline already has native checkpointing, stage progression, resume behavior, and logging.
- MEAPipeline already supports per-well execution and writes per-well outputs.
- MEAPipeline already exposes several controls in constructor arguments (sorter, docker image, sorter kwargs, merge options, analyzer rerun, n_jobs/chunk_duration hints).

Denied in strict form:
- A few controls currently used by stage 2 are not first-class in MEAPipeline (explicit env thread controls).
- Decision: implement env controls natively in MEA_Analysis as optional inputs with defaults when not specified.
- Decision: remove the custom post_merge_4x4 path from axon_reconstructor stage 2.

Conclusion:
- A much smaller adapter is feasible.
- Full removal of wrapper is feasible if remaining custom features are either dropped or upstreamed.

### 2) Prefer patching mea_analysis_routine to accept preprocessed recording directly
This is a strong path and is feasible.

Current MEAPipeline behavior already supports the concept indirectly:
- run_sorting assumes self.recording exists.
- run_preprocessing is the step that currently populates self.recording from raw input.

Recommended MEA_Analysis patch:
- Add an official input mode for preprocessed recording.
- Accept either:
  - recording object, or
  - recording folder path (binary folder / extractor path).
- Add a run control flag to skip Phase 1 preprocessing when preprocessed recording is provided.
- Ensure checkpoint stage is set to PREPROCESSING_COMPLETE in this mode.

This removes the current stage-2 pattern of manually mutating pipeline.recording and pipeline.state.

### 3) Could resource control be handled in MEA_Analysis instead?
Yes, mostly.

Already in MEA_Analysis:
- n_jobs and chunk_duration exist and are used in parts of analyzer/merge flow.
- sorter kwargs override exists.

Needed upstream improvements:
- Use n_jobs/chunk_duration consistently across preprocessing save and all compute calls.
- Add optional env thread controls in MEAPipeline init (OMP, MKL, OPENBLAS, NUMEXPR, CUDA_VISIBLE_DEVICES), with current behavior preserved when unset.
- Add a startup runtime summary log block in MEAPipeline itself.

Then stage 2 can pass one config object and avoid custom environment handling logic.

### 4) Can merge behavior be simplified/upstreamed (or removed)?
Yes.

Already in MEA_Analysis:
- run_optional_merge_phase exists.
- UnitMatch and auto_merge options exist.

Custom behavior currently outside MEA_Analysis:
- post_merge_4x4 optional merge path.

Decision options:
- Chosen option: Drop post_merge_4x4 support in axon_reconstructor stage 2.
- Rationale: this path adds compatibility and maintenance burden and is not required for core pipeline goals.

### 5) Can MEA_Analysis run one well at a time with unified logs/checkpoints?
Yes.

Confirmed:
- CLI and MEAPipeline are single-well oriented via well and stream_id.
- Per-well output directory and per-well pipeline log are native.

If desired, axon_reconstructor can still keep a thin stage-level checkpoint layer for cross-stage orchestration, but not re-implement MEA internals.

## What Actually Creates Compatibility Risk Today
Risk comes from stage 2 mutating MEAPipeline internals directly:
- Assigning pipeline.recording manually.
- Writing pipeline.state stage manually.
- Overriding output_dir/checkpoint_file/logger handlers after object construction.
- Calling specific method sequences that may change as MEA_Analysis evolves.

Any internal refactor in MEA_Analysis can break this adapter unexpectedly.

## Recommended Target Architecture

### Principle
Use MEA_Analysis public interfaces only. No direct mutation of internal attributes from axon_reconstructor.

### Desired integration boundary
- axon_reconstructor stage 2 should provide:
  - inputs (well, file_path metadata, output root, runtime controls)
  - optional preprocessed recording reference
  - stage selection (sort only, sort+analyzer, full reports)
- MEA_Analysis should own:
  - loading/injecting recording
  - stage transitions and checkpoint updates
  - logger setup and file location decisions
  - merge/analyzer/report internals

## Concrete Patch Plan

### Phase A: Minimal upstream changes in MEA_Analysis
1. Add official preprocessed recording input mode.
2. Add skip_preprocessing_when_recording_provided behavior.
3. Add optional env/resource controls with defaults when unset:
  - OMP_NUM_THREADS
  - MKL_NUM_THREADS
  - OPENBLAS_NUM_THREADS
  - NUMEXPR_NUM_THREADS
  - CUDA_VISIBLE_DEVICES
4. Add consistent n_jobs/chunk_duration usage across preprocessing/sorting/analyzer paths.
5. Add runtime summary logging at pipeline start.
6. Add optional explicit log_file override (or logger injection hook).

### Phase B: Shrink stage-2 wrapper in axon_reconstructor
1. Stop mutating pipeline internals (recording/state/output_dir/checkpoint_file/logger).
2. Replace with one MEAPipeline construction call plus explicit public stage calls.
3. Remove custom post_merge_4x4 code path and related metadata/artifact handling.
4. Keep only:
   - outer stage checkpoint start/complete/fail
   - argument translation
   - optional policy validation

### Phase C: Remove post_merge_4x4
1. Delete post_merge_4x4 flag handling from stage-2 inputs and debug config.
2. Delete _run_post_merge_4x4 and helper plotting/location functions that only serve this path.
3. Remove merged_sorter_output_dir and related checkpoint fields tied only to this path.
4. Update docs and CLI help to reflect that this optional merge mode is no longer supported.

## Proposed End State For axon_reconstructor stage 2
A compact adapter with responsibilities limited to:
- translating stage args to MEAPipeline args
- requesting desired phase range
- recording outer stage checkpoint result

No direct handling of:
- MEA internal checkpoint state
- MEA logger handlers
- MEA output_dir rewrites
- manual pipeline.state mutation

## Migration Safety Checklist
- Add integration tests that run one representative well in:
  - full run
  - resume run
  - force restart
  - preprocessed input mode
- Validate parity of key artifacts:
  - sorter output folder
  - analyzer output folder
  - qm_unfiltered and report plots
  - checkpoint stage at completion
- Validate log expectations:
  - info and error in terminal with normal python execution
  - same messages persisted in per-well pipeline log

## Bottom Line
Your direction is sound.
- Most current stage-2 custom logic can be moved into MEA_Analysis or removed.
- The largest simplification win is adding an official preprocessed-recording mode in MEA_Analysis.
- The agreed simplifications for this pass are:
  - upstream optional env controls to MEA_Analysis with defaults
  - remove post_merge_4x4 support from stage 2
- After that, stage 2 can be reduced to a thin, stable adapter with much lower compatibility risk.
