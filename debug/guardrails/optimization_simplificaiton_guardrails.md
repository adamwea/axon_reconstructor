# Optimization And Simplification Guardrails

Status: guardrail document. Once agentic development begins, treat this file as locked. Do not edit it unless Adam explicitly asks for guardrail changes.

This document defines broad goals for keeping the `axon_recon` pipeline small, fast, understandable, and high-throughput while avoiding legacy drift and accidental full-scope work. The filename intentionally preserves Adam's requested spelling.

## Operating Contract

- Commit frequently after each coherent accepted slice, using an `ai:` prefix in the commit subject.
- Update `debug/commit_log.md` after every AI commit.
- Test frequently with focused tests and real-data smoke tests using CLI debug flags.
- Use 1 dataset, 1 well, 2 segments, and a few units by default.
- Expand datasets or wells only when logging, parallelism, H5 contention, or cross-target behavior requires it.
- Do not run full-scope tests unless Adam explicitly requests them.

## North Star

Make the pipeline boring, predictable, and efficient.

Prefer:

- fewer active code paths
- fewer config aliases
- less duplicated runner logic
- shared helpers for lifecycle, target selection, resource budgeting, logging, and summaries
- stage-owned core modules for heavy behavior
- compact durable artifacts
- regenerable caches for large intermediates
- explicit failures over silent fallbacks

Avoid:

- broad rewrites without a narrow acceptance target
- preserving old config layouts that are not active
- hidden fallback chains
- global serialization to avoid fixing resource gating
- large artifacts duplicated only for convenience
- unit tests that fossilize retired behavior
- full data runs as routine validation

## Repository Footprint Guardrails

- Delete code that only supports retired v1, inactive templates/analysis surfaces, obsolete aliases, or unused fallback paths.
- Before deleting, search imports, CLI dispatch, tests, docs, debug scripts, and active runtime YAML.
- Move still-needed logic into active v2 `axon_recon` modules before deleting old surfaces.
- Keep runners thin: orchestration, lifecycle, context binding, phase calls, and result collection only.
- Move heavy scientific or file-transform logic into focused `core/` modules or phase orchestrators.
- Do not add new dependencies unless they clearly reduce complexity, improve throughput, or are already part of the container/runtime strategy.
- Keep sibling repositories out of scope unless Adam explicitly asks.

## Throughput Guardrails

- Apply debug limits and target flags before expensive source loading, materialization, analyzer generation, sorting, plotting, or reporting.
- Resource classes should describe phase demand; machine profiles should describe capacity.
- Use keyed resources for real contention, especially same-source H5 reads.
- Prefer phase-local gates over global worker reductions.
- Avoid recomputing analyzers/templates when complete valid artifacts can be reused.
- Make cache reuse and cache misses visible in logs.
- Use resource tuning/calibration observations before changing broad resource-class estimates.
- Treat tiny debug runs as insufficient evidence for lowering resource estimates unless there are enough observations.

## Storage And Cache Guardrails

- Never modify source `.h5` files.
- Keep large regenerable artifacts under stage/phase-owned cache paths.
- Preserve compact durable artifacts needed for later analysis: sort outputs, labels, compact summaries, GTRs, reconstruction summaries, and final reports.
- Treat concat binaries, analyzers, waveform folders, template workspaces, merged-template caches, and temporary report workspaces as cache-like unless Adam marks them durable.
- Cleanup phases must delete only owned cache paths.
- When changing storage behavior, record created/modified/deleted paths and any size checks in `debug/commit_log.md`.

## YAML And CLI Simplification Guardrails

- Prefer one canonical knob for one behavior.
- Retire aliases after migration when they no longer serve active workflows.
- Parse config once into typed config objects.
- Avoid raw config dictionary interpretation deep inside runners or core modules.
- Add parser tests for meaningful knobs.
- Add behavior tests for knobs that change execution.
- Keep CLI debug flags consistent across stages and direct phases.

## Performance Validation Guardrails

Performance changes must be measured enough to justify their complexity.

For small changes, record:

- expected improvement
- limited real-data smoke command
- elapsed time before/after if practical
- resource usage before/after if relevant
- output equivalence checks
- residual risk

For larger optimization work, include:

- representative limited scope
- reason the scope represents the target bottleneck
- logs inspected
- artifacts inspected
- storage impact
- rollback notes

Do not optimize by weakening correctness, hiding logs, dropping summaries, or silently skipping configured work.

## Required Tests And Acceptance Criteria

### Simplification slice test

Acceptance criteria:

- Removed code is not referenced by active imports, CLI dispatch, tests, docs, or runtime YAML.
- Active behavior has focused coverage after cleanup.
- The diff reduces or clarifies code paths without adding unrelated behavior.
- Commit notes identify what was confirmed not to run anymore.

### Runtime smoke after behavior cleanup

Example:

```bash
axon-recon-container stages preprocess \
  --config debug/debug.runtime.yml \
  --limit-datasets 1 \
  --limit-wells 1 \
  --limit-segments 2 \
  --force-restart
```

Acceptance criteria:

- Selected stage/phase behavior remains correct.
- Logs and summaries are no noisier or less contextual than before.
- Expected artifacts are created or reused.
- No full-scope work starts.

### Storage optimization test

Acceptance criteria:

- Large cache artifacts are created only when needed.
- Cleanup removes owned cache paths and preserves durable artifacts.
- Downstream stages can regenerate or resolve required artifacts after cleanup.
- Size/path observations are recorded in commit notes when practical.

### Throughput optimization test

Acceptance criteria:

- Limited real-data smoke completes successfully.
- Output artifacts are equivalent for the behavior being optimized.
- Elapsed time or resource usage improves, or the change is justified by simpler resource behavior.
- Parallelism/logging remain correct under the selected smoke scope.

### Config simplification test

Acceptance criteria:

- Canonical config names are documented and tested.
- Removed aliases no longer appear in active runtime YAML.
- Invalid old config fails clearly or is migrated by an explicit, logged bridge.
- Active CLI smoke still works with the canonical config.
