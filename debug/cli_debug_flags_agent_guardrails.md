# CLI Debug Flags Agent Guardrails

Status: guardrail document. Once agentic development begins, treat this file as locked. Do not edit it unless Adam explicitly asks for guardrail changes.

This document defines the expected behavior for CLI debug and target flags across the active `axon_recon` v2 pipeline. These flags are the main way future agents should run frequent real-data smoke tests without accidentally starting a full-scope run.

## Operating Contract

- Work in the `axon_reconstructor` repository only unless Adam explicitly asks otherwise.
- Commit frequently after each coherent accepted slice, using an `ai:` prefix in the commit subject.
- Do not push unless Adam explicitly asks.
- Update `debug/agent_guardrails_commit_notes.md` after every AI commit, including acceptance criteria, tests, real-data smokes, logs inspected, and residual risk.
- Run focused tests plus real-data smoke tests whenever a change touches CLI parsing, target selection, stage dispatch, phase dispatch, force-restart behavior, paths, logging, or parallelism.
- Prefer smoke scopes like 1 dataset, 1 well, 2 segments, and a few units. Expand datasets or wells only when the behavior under test requires it, especially for logging or parallelism.
- Do not run full-scope tests unless Adam explicitly requests them or a narrow smoke cannot validate the behavior.

## Stage Scope Contract

Every debug and target flag must work with all CLI selector scopes:

- Full pipeline: `stages all`
- Stage subset: `stages preprocess spikesort`, `stages reconstruct`, or any valid ordered subset
- Phase subset: `stages preprocess.preprocess_segments`, `stages spikesort.sort`, `stages reconstruct.build_templates`
- Mixed selectors: `stages spikesort reconstruct.plot_templates`
- Container wrapper: `axon-recon-container ...` must pass the same flags through unchanged

Shared flags must be registered once on the shared stage parser and passed through nested stage or phase handlers. Do not add a flag only to one stage CLI path unless that flag is truly stage-local and not part of this contract.

## Canonical Flags

### `--limit-segments`

Limit the total number of segments per dataset in the selected data scope.

Required behavior:

- Applies before expensive segment materialization, not only after target enumeration.
- Preprocess: limits segment loading, segment metadata, and segment preprocessing work.
- Spikesort: limits segments consumed by `spikesort.bootstrap_concat_binary` before the bootstrapped concat binary is written.
- Reconstruct: limits segment analyzers and template-building inputs before expensive analyzer/template work.
- Direct phase selectors must receive the same value as full-stage selectors.

### `--limit-datasets`

Limit the total number of datasets in the selected data scope.

Required behavior:

- Applies consistently before per-dataset target expansion.
- The selected dataset count must be visible in logs or summary artifacts.
- Direct phase selectors must not bypass this limit.

### `--limit-wells`

Limit wells per dataset in the selected data scope.

Required behavior:

- This is the preferred canonical flag name.
- If `--limit-wells-per-dataset` remains supported during migration, it must be an alias to the same behavior and must not drift.
- Logs or summaries must make clear how many wells were selected per dataset.

### `--limit-units`

Limit units per well in the selected data scope.

Required behavior:

- Applies before expensive per-unit work.
- Reconstruct template, GTR, plot, report, and summary phases must obey it.
- Spikesort and merge phases that operate on units must either obey it or log clearly that the phase has no unit-scope concept.

### `--target-segments`

Target one or more specific segment indices per dataset in the selected data scope.

Required behavior:

- Segment indices are explicit and deterministic.
- Targeted segments must be resolved before expensive source loading or concat materialization.
- If both target and limit flags are supplied, targets select the candidate set first and limits may further cap it only if that behavior is explicitly logged and tested.

### `--target-datasets`

Target one or more datasets in the data scope.

Required behavior:

- Dataset identifiers must match the runtime/data config identifiers or documented aliases.
- Unknown dataset targets fail fast with a helpful list or context.
- Direct phase selectors must receive the targeted dataset set.

### `--target-wells`

Target one or more wells per dataset in the data scope.

Required behavior:

- Well IDs are deterministic and normalized consistently, for example `well000` versus `0` only if the CLI documents that normalization.
- Unknown well targets fail fast before heavy work starts.
- Multi-dataset runs must apply well targets per dataset unless a future syntax explicitly supports dataset-specific well maps.

### `--target-units`

Target one or more units per well by unit ID.

Required behavior:

- This is the canonical explicit-unit flag for data-scope targeting.
- Existing `--unit-id` and `--unit-ids` flags may remain as aliases, but new work should validate `--target-units`.
- Reconstruct unit phases must process only targeted units and must log selected IDs.

### `--force-restart`

Delete target artifacts, if any, and start the targeted stages and/or phases from scratch.

Required behavior:

- Cleanup must be scoped to the selected datasets, wells, stages, phases, segments, and units.
- Never delete source `.h5` files.
- Direct phase selectors must clear only the artifacts owned by that phase unless the phase has an explicitly documented dependency cleanup.
- Logs must show what was deleted, what was preserved, and where regenerated artifacts were written.

## Implementation Guardrails

- Debug flags must be data-scope controls, not cosmetic parser knobs.
- Every flag must reach the first heavy operation it is meant to limit.
- Do not rely on old stage-level `debug_mode` fields when a CLI override is present.
- Do not implement separate target-selection logic for container and non-container commands.
- Do not let `axon-recon-container` parse stage or phase semantics; it should resolve container execution details and forward pipeline args unchanged.
- Stage and phase summaries must include applied debug flags so smoke results are auditable.

## Required Tests And Acceptance Criteria

### Parser and dispatch test

Run focused parser tests after flag changes.

Acceptance criteria:

- All canonical flags parse on `stages` and `stage` commands.
- Full-stage, direct-phase, mixed-selector, and `all` selectors carry the same parsed flag values into nested handlers.
- Container wrapper dry-run forwards all flags unchanged.
- Invalid target values fail before any real data work starts.

### Limited preprocess smoke

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

- Exactly the limited dataset/well/segment scope is selected.
- `copy_src_to_scratch`, `save_rec_metadata`, and `preprocess_segments` logs show the limited scope.
- Phase summaries and `pipeline.jsonl` record the applied limits.
- The run completes without starting unselected datasets or wells.

### Limited spikesort smoke

Example:

```bash
axon-recon-container stages spikesort \
  --config debug/debug.runtime.yml \
  --limit-datasets 1 \
  --limit-wells 1 \
  --limit-segments 2 \
  --force-restart
```

Acceptance criteria:

- `bootstrap_concat_binary` builds from the limited segment set before sorting.
- `sort` uses the bootstrapped limited recording, not a full recording.
- `bombcell_label` and cleanup operate only on the selected well.
- Logs identify sort engine, selected scope, and force-restart cleanup.

### Limited reconstruct smoke

Example:

```bash
axon-recon-container stages reconstruct \
  --config debug/debug.runtime.yml \
  --limit-datasets 1 \
  --limit-wells 1 \
  --limit-segments 2 \
  --limit-units 3 \
  --force-restart
```

Acceptance criteria:

- Segment analyzers/template build inputs are capped before expensive analyzer/template work.
- Unit phases process no more than the selected unit count.
- Unit IDs processed are visible in logs or summaries.
- Plot/report phases do not expand back to full unit scope.

### Direct phase smoke

Example:

```bash
axon-recon-container stages reconstruct.plot_templates \
  --config debug/debug.runtime.yml \
  --limit-datasets 1 \
  --limit-wells 1 \
  --target-units 1,3 \
  --force-restart
```

Acceptance criteria:

- Only the requested phase runs.
- Upstream phases are not silently run unless the command explicitly requests dependency preparation.
- Missing required inputs fail with actionable messages.
- Force-restart cleanup is phase-scoped.

### Logging or parallelism expansion smoke

When the change affects logging visibility or concurrent execution, expand scope intentionally.

Example:

```bash
axon-recon-container stages preprocess \
  --config debug/debug.runtime.yml \
  --limit-datasets 2 \
  --limit-wells 2 \
  --limit-segments 2 \
  --force-restart
```

Acceptance criteria:

- Logs show every selected dataset and well started and completed.
- Progress output remains visible and does not hide completion or resource-usage logs.
- No unselected dataset, well, segment, or unit work starts.
