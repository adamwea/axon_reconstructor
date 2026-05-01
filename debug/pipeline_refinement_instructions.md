# Pipeline Refinement Autopilot Instructions

Draft status: reviewable working instructions for future AI-assisted refinement of the `preprocess`, `spikesort`, and `reconstruct` stages.

These instructions are for iterative repo work after the basic pipeline behavior is already in place. The goal is to treat this as the first real version of the pipeline: simplify aggressively, make configured behavior reliable, and remove code that only exists to preserve unused legacy paths.

## Scope

Primary stages:

- `preprocess`
- `spikesort`
- `reconstruct`

Related shared systems are in scope when needed:

- runtime config parsing and validation
- target selection and debug limits
- stage and phase orchestration
- parallel execution and worker allocation
- logging, progress, summaries, and failure reporting
- smoke-test tooling and CLI flags used to validate these stages
- CLI stage/phase selection and dispatch
- active tests that protect the v2 pipeline behavior

Out of scope unless explicitly needed:

- broad scientific algorithm changes
- large UI/documentation rewrites
- compatibility with old config layouts that are not actively used
- preserving aliases, fallbacks, or legacy knobs solely because they once existed

Repository boundary:

- Only edit the `axon_reconstructor` repository during this refinement process.
- Do not modify sibling workspaces such as `spikeinterface`, `MEA_Analysis`, `axon_velocity`, `UnitMatch`, `SLAy`, `projects`, or scratch data directories unless Adam explicitly asks.
- If a dependency behavior looks wrong outside `axon_reconstructor`, document the finding and work around it cleanly from this repo.

## North Star

Make the pipeline boring, predictable, and easy to reason about.

Each stage should do exactly what its YAML says:

- A phase runs only when it is both enabled and present in the active `phase_sequence`.
- A phase that is disabled must not run.
- A phase that is omitted from `phase_sequence` must not run.
- Stage-level defaults should be clear, typed, and tested.
- Hidden fallbacks, legacy aliases, and redundant knobs should be removed or consolidated.
- Runtime logs and summary artifacts should make it obvious what ran, what skipped, what failed, and why.

Prefer smaller, cleaner code over preserving historical flexibility. No one besides Adam has used this pipeline, so breaking old internal config shapes is acceptable when it makes the current system simpler.

## Pipeline Purpose

Keep the high-level scientific workflow in view while refactoring. The pipeline exists to:

1. Preprocess Maxwell `.h5` data.
2. Concatenate/prep recordings for downstream processing.
3. Spike sort the concatenated well-level recording.
4. Register segment-level spike/sort information back from the concat sort.
5. Build templates for each unit using segment analyzers only.
6. Generate reconstructed unit branch morphology, including GTR artifacts.
7. Eventually feed a rewritten analysis stage that can inspect template and morphology reconstruction data within each well and across whole datasets.

Refinement should preserve this direction. Avoid optimizing for historical module boundaries if they conflict with this end-to-end workflow.

## Operating Loop

For each refinement slice:

1. Inspect this instruction file, `debug/pipeline_refinement_commit_notes.md`, the active runtime YAML, and the enabled phase sequence before editing.
2. Identify the smallest coherent behavior to simplify or fix.
3. Remove redundant code and fallbacks while keeping the active path working.
4. Add or update focused tests for the behavior being changed.
5. Run focused pytest validation and, whenever behavior touches real data, CLI dispatch, phase wiring, paths, logging, or parallelism, run a minimal smoke test too.
6. Review `git diff` and commit a green, coherent slice with an AI prefix when commit authorization is active.
7. Append a concise commit entry to `debug/pipeline_refinement_commit_notes.md` with summary, acceptance criteria, validation, storage/cache impact, CLI impact, and follow-ups.
8. Move to the next slice.

Do not batch unrelated refactors together. If a cleanup reveals another problem, either make it the next slice or write it down for later.

Slice discipline:

- Work on one primary intent at a time.
- Each commit should have one clear purpose, such as `retire templates import path`, `simplify CLI registry`, or `move analyzer loading into reconstruct core`.
- Do not mix broad cleanup, behavior changes, test deletion, and CLI rewiring in the same commit unless they are tightly coupled and documented in the notes.
- Acceptance criteria must include what should run and what must not run, especially for phase-sequence or enabled/disabled behavior.

Instruction-file rule:

- Before Adam says to start iterating, edits to this instruction file are allowed when Adam asks for instruction changes.
- After Adam says to start iterating, do not edit this instruction file unless Adam explicitly asks.
- During iteration, update only `debug/pipeline_refinement_commit_notes.md` for running notes unless Adam instructs otherwise.
- Review both markdown files after every change slice so the current work remains aligned with the operating contract.

## Architecture Rules

### Retire Old Pipeline Modules

The v2 pipeline is the only target design. Move toward deleting old modules rather than preserving compatibility shims.

Retirement targets:

- Delete the v1 pipeline package at `src/axon_reconstructor` after confirming no active v2 code, CLI path, tests, or docs still import it.
- Delete the `templates` stage module after moving any still-needed analyzer/template logic into the updated `reconstruct` stage.
- Delete the `analysis` stage module after disconnecting it from current CLI/stage dispatch; it will be rewritten later after this refinement pass.

Retirement procedure:

1. Search imports and CLI dispatch for references before removing a module.
2. Move only the logic that the active v2 pipeline truly needs.
3. Prefer moving core logic into the destination stage's `core/` modules and exposing it through dedicated phase orchestrators.
4. Delete old wrappers, tests, and config paths that only protected retired modules.
5. Run focused tests plus a minimal smoke test after each module retirement slice.

Do not leave compatibility imports from `axon_recon` back into `axon_reconstructor`. If code still needs the old module, migrate that dependency first.

Pre-delete audit requirements:

- Run import/reference searches before deleting retired modules, including searches for `axon_reconstructor`, `stages.templates`, `stages.analysis`, `run_templates`, and `run_analysis`.
- Audit CLI registry/dispatch, pyproject entry points, tests, docs, and debug scripts for old module references.
- Run a pytest collection or focused import test after import-path removals.
- Run a smoke command after deleting or disconnecting anything that affects CLI dispatch, stage wiring, imports, or real-data IO.
- Record what was checked and what no longer imports the retired module in `debug/pipeline_refinement_commit_notes.md`.

Migration-before-deletion rule:

- For the `templates` stage, migrate needed analyzer/template behavior into `reconstruct` first.
- Switch active imports and callers to the reconstructed v2 path.
- Confirm the reconstruct path works with focused tests and a smoke test.
- Only then delete the old `templates` module and obsolete tests.

### Keep Runners Minimal

Stage runners should be thin. They should coordinate phase calls, build inputs, handle lifecycle/logging boundaries, and return results. They should not contain core business logic.

Runners may:

- select and order phases
- bind logging/progress context
- call phase orchestrators
- collect and return structured results
- handle skip/failure lifecycle events

Runners should not:

- implement SpikeInterface processing details
- implement plotting internals
- perform heavy file transformations directly
- contain long nested helper stacks
- duplicate config interpretation already handled in config parsing

### Put Core Logic In Core Modules

Core logic belongs in `core/` or similarly focused modules under the stage package. Examples:

- recording/file transformations
- analyzer discovery and loading
- template extraction/building
- sort result inspection
- plotting data preparation
- merge/report computations

Core functions should be directly testable without running the whole pipeline.

### Use Dedicated Phase Orchestrators

Each real phase should have a dedicated orchestrator entry point. The preferred shape is:

```text
axon_recon/pipeline/stages/<stage>/
  config.py
  runner.py
  api.py
  models/
  core/
  orchestrators/
    <phase_name>.py
```

Each phase orchestrator should:

- accept typed inputs or explicit keyword arguments
- call focused core functions
- emit clean lifecycle events through shared logging helpers
- return a structured result object
- avoid parsing raw YAML directly

If a phase currently lives as a large block inside a runner, extract it into an orchestrator and move heavy logic into `core/`.

### Minimize Lines And Duplication

Use shared helpers for repeated patterns:

- path resolution
- output-root handling
- debug limits
- phase-sequence filtering
- skip result construction
- JSON writing
- lifecycle logging
- target distribution
- SpikeInterface analyzer loading policies

Delete unused compatibility wrappers, dead aliases, and redundant helper variants after confirming the active YAML and tests do not need them.

## YAML And Knob Rules

YAML is the user-facing contract. Make every active knob work exactly as written.

Rules:

- Prefer one canonical knob for one behavior.
- Remove or consolidate duplicate knobs.
- Avoid multi-path fallback chains unless they are actively needed by current configs.
- Parse config once into typed stage config objects.
- Avoid interpreting raw config dictionaries deep inside runners or core logic.
- Add parser tests for every meaningful knob.
- Add behavior tests for knobs that change execution, not just parsed values.
- Keep debug knobs consistent across stages where practical.

Phase behavior rules:

- Active `phase_sequence` controls execution order.
- `enabled: false` prevents execution even if the phase appears in `phase_sequence`.
- A configured phase block does not imply execution unless the phase is also in sequence.
- A phase omitted from `phase_sequence` must not run due to a default enabled block.
- Skips should be explicit in result/summary artifacts when a phase is considered but disabled.
- Avoid quiet surprise execution.

## CLI Contract

The CLI should assume the pipeline is organized as stages and phases. Avoid manually wiring every possible `stage.phase` command when a declarative stage/phase registry can generate accepted commands and dispatch targets.

Required behavior:

- `axon_recon stages all` runs `preprocess`, `spikesort`, and `reconstruct` in series.
- Prefer positional selectors as the concrete shell syntax: `axon_recon stages spikesort reconstruct` runs only those stages in the requested order.
- `axon_recon stages spikesort reconstruct.analyzers` runs the full `spikesort` stage, then the `analyzers` phase under `reconstruct`.
- Bracketed examples such as `axon_recon stages [spikesort, reconstruct]` are conceptual shorthand only unless the CLI explicitly implements that syntax.
- Running any combination of stages and phases directly should be possible when the stage/phase names exist in the registry.
- Stage and phase selectors should respect the same runtime config, debug limits, force flags, and logging setup as normal full-stage execution.
- Unknown stage or phase names should fail fast with a helpful list of valid selectors.
- `all` should be explicit and stable; it should not accidentally include retired or experimental stages.

Preferred implementation:

- Maintain one registry of active stages and their phases.
- Generate CLI choices, help text, and dispatch from that registry.
- Let each registry entry point to a stage runner or dedicated phase orchestrator/runtime wrapper.
- Keep aliases minimal and remove them when they only preserve old command shapes.
- Keep the active stage list to `preprocess`, `spikesort`, and `reconstruct` until Adam explicitly adds the rewritten analysis stage.

Validate CLI changes with both parser/unit tests and at least one real smoke invocation using debug limits.

CLI validation must confirm both what runs and what does not run. For mixed selectors, record in the notes which stages/phases were expected to execute and which configured-but-unselected phases were confirmed not to execute.

## Stage-Specific Refinement Goals

### Preprocess

Focus on making segment handling, scratch/source copying, metadata saving, preprocessing, concatenation, and plotting phases cleanly separated.

Priorities:

- Make active preprocess phases obey `phase_sequence` exactly.
- Keep source/scratch/final path policy centralized.
- Ensure segment limits apply before expensive work.
- Keep plotting phases optional and quiet when disabled.
- Remove old aliases and fallback path guesses that are not used by current runtime YAML.

### Spikesort

Focus on clear phase boundaries around bootstrap concat binary, sort, summarize, bombcell label, merge phases, and cleanup.

Priorities:

- Preserve well-worker parallelism for independent non-sort phases.
- Keep the `sort` phase single-well gated when `resources.force_single_well_sort: true`.
- Do not globally reduce `well_workers` to serialize sort.
- Ensure `n_jobs`, `chunk_duration`, and sorter-specific options are applied to the correct internal operation.
- Make bombcell/analyzer knobs actually wire into analyzer creation and extension computation.
- Keep merge phases separate and do not run them unless enabled and present in sequence.
- Remove stale fallback behavior copied from old spikesorting scripts.

### Reconstruct

Focus on making template/analyzer preparation, unit selection, reconstruction, plotting, and reporting clean and sequence-driven.

Priorities:

- Ensure embedded template phases under reconstruct obey the reconstruct phase sequence.
- Keep unit limits and segment limits applied before expensive analyzer/template work.
- Avoid recomputing analyzers/templates when configured cache paths already satisfy the phase.
- Keep plotting/report phases optional and non-invasive.
- Move reconstruction math and template handling out of runners into core modules.

## Storage And Cache Discipline

Minimize permanent storage of very large data objects. The pipeline needs to run across entire datasets, so every phase should be intentional about what it persists.

General rules:

- Never modify source `.h5` input data. All writes must go to configured scratch, cache, or output roots.
- Treat large binaries, analyzers, waveform/template workspaces, and merged intermediate artifacts as caches unless they are explicitly final outputs.
- Make caches stage-owned, phase-aware, and safe to clear at the end of the relevant stage.
- Prefer compact durable artifacts that support later analysis or regeneration.
- Keep direct stage/phase CLI execution working so expensive artifacts can be regenerated on demand instead of stored forever.
- Do not preserve duplicate copies of large artifacts just for convenience.
- Add cleanup knobs only when they map to real workflow needs; avoid creating many redundant cache-retention flags.

Default artifact policy:

- Preserve as final or final-like artifacts: sort outputs, compact summaries, cluster labels, per-phase reports, GTRs, reconstruction summaries, and other compact metadata needed for later analysis.
- Treat as cache-like artifacts: concat binaries, sort analyzers, segment analyzers, waveform folders, template workspaces, merged templates, temporary report workspaces, and intermediate SpikeInterface analyzer folders.
- If an artifact is large and regenerable, prefer cache treatment unless Adam explicitly marks it as durable.

Spikesort storage expectations:

- The concat binary is a cache and should be clearable after the sort-dependent phases finish.
- SpikeInterface analyzers created for sorting, labeling, or reporting should be treated as cache unless explicitly promoted to a final artifact.
- Sort outputs and compact summaries/labels should be preserved according to the active output contract.

Reconstruct storage expectations:

- Segment analyzers and merged templates are cache-like and should be clearable after reconstruction outputs are generated.
- GTRs are relatively small and should be retained because they carry much of the morphology data needed for later analysis.
- Larger template/analyzer artifacts should be regenerable by rerunning specific reconstruct phases.

When changing storage behavior, record the storage/cache impact in `debug/pipeline_refinement_commit_notes.md` and run a smoke test that confirms expected files are created and cleaned. Include observable before/after checks such as relevant directory listings or `du -h --max-depth=...` summaries when practical.

## Resume And Force-Restart Behavior

Implement or refine resume logic inside each phase wherever practical. Long-running phases should be able to continue from useful existing artifacts instead of repeating expensive work after interruption.

Resume rules:

- A phase should detect complete, valid prior outputs and skip or reuse them when not force-restarting.
- Resume checks should be explicit and validated with manifests, summaries, or expected file sets rather than vague directory-exists checks.
- Partial outputs should be either safely resumed or clearly cleaned/rebuilt; avoid silently treating partial artifacts as complete.
- Log whether a phase is running fresh, resuming, reusing complete outputs, or rebuilding incomplete outputs.
- Notes for each commit should mention resume behavior when touched.

Force-restart rules:

- `--force-restart` should clear all targeted artifacts for the selected stage/phase before execution starts.
- Clearing targeted artifacts a priori is preferred because it makes generated outputs easier to inspect and avoids confusing stale artifacts.
- Force-restart cleanup must be scoped to the selected dataset/well/stage/phase targets and must never delete source `.h5` inputs.
- After force-restart changes, run a smoke test that confirms targeted artifacts were removed and regenerated.

## Parallelism Rules

General behavior:

- Stages run one at a time across the selected data scope.
- Within a stage, well workers can run independently through their phase chains.
- Unit workers and segment workers should be derived and applied consistently from runtime resources.
- Worker allocation should be logged clearly enough to debug interleaved execution.
- Shared mutable state across well workers should be avoided unless guarded deliberately.

Specific exception:

- `spikesort.sort` may be gated by `resources.force_single_well_sort: true` so only one well is inside the sort phase at a time.
- Other spikesort phases should continue to use well parallelism.
- The sort gate should be phase-local, not a global stage serialization hack.

Parallelism tests:

- Use two wells when verifying well-worker concurrency.
- Use up to two datasets with two wells per dataset when dataset/well interaction matters.
- Confirm both positive behavior and absence of unwanted gating.

## Smoke Testing Rules

Use the smallest useful validation for each change.

Default smoke limits:

- 1 dataset
- 1 well
- 2 segments
- 1 unit

Parallelism smoke limits:

- up to 2 datasets
- up to 2 wells per dataset
- 2 segments
- 1 unit unless the test specifically needs more

Prefer CLI/runtime limits when available:

- `--limit-segments 2`
- `--limit-units 1`
- stage or phase debug limits in runtime YAML
- temporary debug runtime overrides when they avoid expensive full runs

Smoke timeout and stop conditions:

- Smoke tests should usually complete quickly because they must use limited datasets, wells, segments, and units.
- Use a 20-minute maximum runtime for smoke tests unless Adam explicitly approves a longer run.
- Kill or stop a smoke test that exceeds the 20-minute cap, record the timeout in `debug/pipeline_refinement_commit_notes.md`, and inspect logs/artifacts before continuing.
- Carveout: if log/artifact inspection shows the smoke was healthy and simply needed more time, rerun it once with up to a 1-hour maximum runtime.
- Record the reason for extending beyond 20 minutes, the logs inspected, and the final outcome in `debug/pipeline_refinement_commit_notes.md`.
- For Kilosort/container runs, inspect sorter/container logs in the sort output directory before deciding whether to extend, retry, or change the smoke scope.
- If a limited smoke unexpectedly starts full Kilosort, large analyzer compute, broad template extraction, or whole-dataset work, stop it and report the scope mismatch rather than letting it run unattended.
- Prefer making the smoke smaller or fixing limit propagation before trying the same expensive command again.

Expected limited-data caveats:

- Smoke tests using limited segments will produce sparse or lower-density templates. That is expected and should not be treated as a template-quality failure by itself.
- If reconstruct units fail only because channel selection is under-supported by too few segments, rerun the reconstruct smoke with more segments before treating it as a code failure.
- Record when a failure appears to be a limited-data artifact rather than a pipeline bug.

Testing cadence:

- Run focused pytest tests for pure config/core changes.
- Run a minimal runtime smoke for phase sequencing, path, logging, or parallelism changes.
- Do not rely on pytest alone when a change could behave differently with real data.
- Use smoke tests to catch issues that mocks miss: real file layouts, SpikeInterface object behavior, HDF5 segment metadata, sorter/analyzer side effects, CLI argument plumbing, and log/output paths.
- Avoid full expensive SpikeInterface/Kilosort/template runs unless explicitly requested or truly necessary.
- If a smoke test is expected to touch external tools, explain what it will run before launching it.

## Test Cleanup Rules

Tests should protect the active v2 pipeline, not fossilize retired behavior.

Rules:

- Delete tests that only cover v1 pipeline behavior, retired modules, removed aliases, or obsolete fallback config layouts.
- Move useful fixtures/helpers out of deleted test modules before deleting them.
- Consolidate duplicated tests around shared helpers for phase sequencing, debug limits, CLI selectors, and target distribution.
- Prefer fewer high-signal tests over many near-duplicate tests that make refactors noisy.
- Keep smoke-test documentation or scripts aligned with the active runtime YAML and CLI.
- When deleting tests, make sure the active behavior is still covered by focused unit tests or a smoke test.
- Do not keep tests solely to preserve old command names or old module import paths.

## Git Discipline

Before editing:

- Check `git status`.
- Identify user changes and do not revert them.
- Keep unrelated dirty files out of the current slice.

Commit discipline when autopilot commit authorization is active:

- Commit frequently after each green coherent slice.
- Use a clear AI prefix, preferably `ai:`.
- Example: `ai: simplify spikesort phase sequencing`.
- Auto-commit only; do not auto-push.
- Never run `git push` during autopilot refinement unless Adam explicitly asks for a push.
- Include tests/smoke results in the commit message body when useful.
- After every commit, append an entry to `debug/pipeline_refinement_commit_notes.md` so Adam can review what happened after a long unattended run.
- Use git history to backtrack by making new corrective commits or explicit reverts.
- Do not use destructive commands such as `git reset --hard` unless Adam explicitly requests them.

If a fix does not work:

- Inspect the diff and logs.
- Prefer a small corrective patch over broad rewrites.
- Use the previous AI-prefixed commits as checkpoints.

## Logging Refinement Target

Build a reusable professional logging subsystem for the full multi-stage pipeline. Do not solve logging by duplicating manual log calls everywhere.

Core principle:

- Emit each event once with rich context.
- Route or filter that event into the appropriate physical logs.
- Use contextual logging, not string-prefix hacks.

Desired context hierarchy:

```text
Run
└── Dataset
    └── Recording
        └── Well
            └── Stage
                └── Phase
```

Every log record should support as much context as available:

- `run_id`
- `dataset_id`
- `dataset_name`
- `recording_id`
- `chip_id`
- `date`
- `assay`
- `well_id`
- `stage`
- `phase`
- `event`
- `pid`
- `worker`
- elapsed time when useful
- relevant output paths
- exception information and tracebacks when failures occur

Lifecycle events to add around major execution units:

- `run_started`, `run_completed`, `run_failed`
- `dataset_started`, `dataset_completed`, `dataset_failed`
- `recording_started`, `recording_completed`, `recording_failed`
- `well_started`, `well_completed`, `well_failed`
- `stage_started`, `stage_completed`, `stage_failed`
- `phase_started`, `phase_completed`, `phase_failed`, `phase_skipped`

These lifecycle events should feed `summary.json`.

### Required Log Outputs

Each full run should produce:

- `logs/pipeline.log`: human-readable chronological full run log.
- `logs/pipeline.jsonl`: canonical structured log stream; one JSON object per event.
- `logs/errors.log`: warnings, errors, critical failures, and exception tracebacks.
- `logs/summary.json`: final run summary by dataset, recording, well, stage, and phase when available.

Strongly recommended filtered logs:

- `logs/datasets/<dataset_id>/dataset.log`
- `logs/datasets/<dataset_id>/recordings/<recording_id>/recording.log`
- `logs/datasets/<dataset_id>/recordings/<recording_id>/wells/<well_id>/well.log`
- `logs/datasets/<dataset_id>/recordings/<recording_id>/wells/<well_id>/phases/<stage>__<phase>.log`

Do not create physical phase logs when `phase_logs.enabled: false`. Still include phase context in `pipeline.log` and `pipeline.jsonl`.

Do not create physical well logs when `well_logs.enabled: false`. Still include well context in `pipeline.jsonl`.

Avoid physical stage-level logs by default unless they naturally fall out of the filtering system. The structured JSONL stream should make stage filtering possible.

### Logging Subsystem Shape

Implement logging as a reusable subsystem, for example:

```text
axon_recon/pipeline/logging/
  config.py
  setup.py
  context.py
  formatters.py
  filters.py
  summary.py
  multiprocessing.py
```

Adapt the exact package structure if the repo already has a better location, but keep the logging system shared and out of stage-specific runners.

Preferred API shape:

```python
with log_context(dataset_id=dataset_id):
    ...

with log_context(recording_id=recording_id, well_id=well_id):
    ...

with phase_logger(stage="spikesort", phase="run_kilosort4"):
    ...
```

or:

```python
logger = get_logger().bind(
    dataset_id=dataset_id,
    recording_id=recording_id,
    well_id=well_id,
    stage="spikesort",
    phase="run_kilosort4",
)
```

Normal logging calls should stay clean and automatically carry context.

### Logging Backend Rules

- Use Loguru and/or Rich where useful.
- Prioritize correctness and multiprocessing safety over terminal aesthetics.
- Rich is appropriate for console readability.
- File logs must be plain and parseable.
- If stdlib `logging` with `QueueHandler` and `QueueListener` is safer for multiprocessing, use it underneath or bridge Loguru into it.
- Workers should not write directly to shared physical log files when a central queue/listener can be used.
- Worker exceptions must include tracebacks and route to `errors.log`, the relevant well log, and the relevant phase log when enabled.
- Logging setup should initialize once near pipeline startup.
- Avoid duplicate handlers when pipeline entry points are invoked repeatedly in the same Python process.
- If logging is disabled globally, the pipeline must still run.

Preferred human-readable format:

```text
2026-04-28 15:23:41.102 | INFO | run=debug01 dataset=Media_Density_T5 recording=000031 well=well000 stage=spikesort phase=run_kilosort4 pid=418984 | Starting Kilosort4
```

Preferred JSONL event shape:

```json
{
  "time": "2026-04-28T15:23:41.102",
  "level": "INFO",
  "message": "Starting Kilosort4",
  "run_id": "debug01",
  "dataset_id": "Media_Density_T5",
  "recording_id": "000031",
  "chip_id": "M08073",
  "assay": "AxonTracking",
  "well_id": "well000",
  "stage": "spikesort",
  "phase": "run_kilosort4",
  "event": "phase_started",
  "pid": 418984,
  "worker": "ForkPoolWorker-3"
}
```

### Runtime Logging Knobs

Add or align top-level runtime YAML knobs with this functionality:

```yaml
logging:
  enabled: true

  level: INFO
  console:
    enabled: true
    rich: true
    level: INFO

  structured:
    enabled: true
    path: logs/pipeline.jsonl
    level: DEBUG

  run_log:
    enabled: true
    path: logs/pipeline.log
    level: INFO

  error_log:
    enabled: true
    path: logs/errors.log
    level: WARNING

  dataset_logs:
    enabled: true
    level: INFO

  recording_logs:
    enabled: true
    level: INFO

  well_logs:
    enabled: true
    level: DEBUG

  phase_logs:
    enabled: false
    level: DEBUG

  summary:
    enabled: true
    path: logs/summary.json

  include_external_stdout: true
  include_external_stderr: true
  capture_warnings: true
  capture_uncaught_exceptions: true

  multiprocessing:
    enabled: true
    use_queue_listener: true
```

The exact names can be adjusted to match existing config style, but preserve the behavior.

### Logging Acceptance Criteria

1. Logs are not mislabeled as per-well logs when they contain the full run.
2. Multiprocessing logs are readable and attributable to the correct dataset, recording, well, stage, and phase.
3. A full run produces `pipeline.log`, `pipeline.jsonl`, `errors.log`, and `summary.json`.
4. Per-dataset, per-recording, per-well, and per-phase physical logs can be toggled independently.
5. Disabling physical filtered logs does not remove context from the canonical structured log.
6. Exceptions from worker processes are captured with tracebacks.
7. Re-running the pipeline does not attach duplicate handlers.
8. Existing active stages and phases keep working while using the new logging foundation.

## Cleanup And Deletion Rules

Delete code when it exists only to support inactive legacy behavior.

Good deletion candidates:

- the v1 package at `src/axon_reconstructor` after migration is complete
- the old `templates` stage after required logic is moved into `reconstruct`
- the old `analysis` stage after it is disconnected from active dispatch
- duplicate config aliases
- fallback branches for old YAML layouts
- commented-out execution paths
- unused wrappers around wrappers
- runner helper functions that can become core/orchestrator functions
- old debug print controls superseded by centralized logging
- broad exception fallbacks that hide real failures

Before deleting:

- Search for references.
- For module retirement, include searches like `rg "axon_reconstructor|stages\.templates|stages\.analysis|run_templates|run_analysis"` from the repo root.
- Confirm the active debug/runtime YAML does not rely on it.
- Add or update tests that lock in the intended simpler behavior.
- Run a real smoke test when deletion affects CLI dispatch, imports, stage wiring, path resolution, or real-data IO.
- Prefer one focused deletion commit over mixing deletion with unrelated feature work.

## Failure Handling Rules

- Fail early on invalid config.
- Prefer clear validation errors over silent fallback behavior.
- Include phase, dataset, recording, well, and output path context in errors.
- Preserve tracebacks in logs.
- Return structured failed target results where the pipeline is designed to continue across targets.
- Do not swallow exceptions just to keep a stage appearing successful.

## Definition Of Done For A Refinement Slice

A slice is done when:

- The active runtime YAML behavior is simpler or more correct.
- Enabled and sequenced phases are the only phases that run.
- Config knobs touched by the slice are parsed and tested.
- Runners are no more complex than before, preferably simpler.
- Core logic is direct-testable where practical.
- Minimal pytest validation passes.
- A real smoke test has been run whenever the slice touches CLI dispatch, phase wiring, paths, logging, parallelism, or real-data IO.
- CLI selectors still behave as expected for `all`, stage-only selections, and mixed stage/phase selections when relevant.
- Acceptance criteria and notes identify expected-to-run phases and confirmed-not-run phases when relevant.
- No active imports or CLI paths still depend on retired v1, `templates`, or `analysis` modules unless the current slice explicitly documents why retirement is not complete yet.
- Storage/cache behavior is intentional and documented when touched.
- `debug/pipeline_refinement_commit_notes.md` has an entry for the slice when a commit is made.
- Logs are no noisier than before and ideally more contextual.
- A clean AI-prefixed commit exists when commit authorization is active.
