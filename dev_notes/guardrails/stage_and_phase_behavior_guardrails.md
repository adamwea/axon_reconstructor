# Stage And Phase Behavior Guardrails

Status: guardrail document. Once agentic development begins, treat this file as locked. Do not edit it unless Adam explicitly asks for guardrail changes.

This document gives high-level behavior expectations for the active v2 pipeline stages and phases. Use it to decide acceptance criteria before changing stage dispatch, phase sequencing, direct phase execution, resume behavior, force-restart cleanup, outputs, or summaries.

## Operating Contract

- Commit frequently after each coherent accepted slice, using an `ai:` prefix in the commit subject.
- Update `debug/commit_log.md` after every AI commit.
- Test frequently with real-data smokes using CLI debug flags.
- Use 1 dataset, 1 well, 2 segments, and a few units by default.
- Expand datasets or wells only when behavior under test requires it, especially logging and parallelism.
- Do not run full-scope tests unless Adam explicitly requests them.

## Universal Stage And Phase Rules

- The active pipeline stages are `preprocess`, `spikesort`, and `reconstruct`.
- `stages all` runs active stages in canonical order.
- Stage subsets run only selected stages in the requested order.
- Direct phase selectors run only the selected phase unless the command explicitly requests dependency preparation.
- A phase runs only when it is enabled and present in the active `phase_sequence`, except for explicit direct-phase execution where the phase handler itself is selected.
- Disabled phases must not run silently.
- Phases omitted from `phase_sequence` must not run just because their config block exists.
- Every phase must have clear start, completion, skip, failure, summary, and resource-usage behavior where applicable.
- Resume checks must validate expected artifacts, not vague directory existence.
- `--force-restart` cleanup must be scoped to selected targets and must never delete source `.h5` data.

## Preprocess Stage

Purpose: turn Maxwell `.h5` data into validated metadata and per-segment preprocessed recordings suitable for spikesort and reconstruct downstream work.

Expected behavior:

- Select datasets, recordings, wells, and segments according to runtime config and CLI debug/target flags.
- Optionally copy or stage source H5 data to scratch.
- Save recording metadata, segment epochs, sampling metadata, and common electrode artifacts.
- Preprocess selected segments in lazy or binary mode according to config.
- Write manifests and summaries that downstream stages can resolve deterministically.
- Optional plotting/report/cleanup phases must be quiet when disabled.

Known preprocess phases:

- `copy_src_to_scratch`: copy/stage source H5 data into scratch when configured.
- `save_rec_metadata`: inspect selected recordings and write metadata/context artifacts.
- `prepare_raw_binaries`: optional raw binary preparation phase.
- `preprocess_segments`: load selected segments, preprocess them, and write segment recording manifests.
- `plot_segment_traces`: optional per-segment trace diagnostics.
- `plot_segment_channel_layouts`: optional per-segment channel layout diagnostics.
- `concat_segments`: optional concatenated recording build.
- `plot_concat_traces`: optional concat trace diagnostics.
- `plot_concat_channel_layout`: optional concat channel layout diagnostics.
- `plot_raster_threshold`: optional threshold/raster diagnostics.
- `report_preprocessing`: optional preprocessing report.
- `cleanup_preprocessing_outputs`: optional cleanup of generated preprocessing intermediates.
- `wipe_src_scratch`: optional scratch-source cleanup, dry-run by default unless explicitly changed.

Preprocess acceptance cues:

- Logs identify whether data came from source H5 or scratch H5.
- Segment limits apply before segment loads and preprocess writes.
- `preprocess_segments` completion and resource usage are visible in terminal and file logs.
- Manifests contain enough provenance for spikesort and reconstruct to consume segment outputs.

## Spikesort Stage

Purpose: build a selected well-level recording for sorting, run the configured sorter, label units, optionally merge units, and clean regenerable concat caches.

Expected behavior:

- Consume preprocess segment outputs or configured concat sources.
- Build a bootstrapped concat binary/cache from selected segments before sorting.
- Run the configured sort engine.
- Preserve local/container/HPC compatibility by using `local_spikeinterface` for container/HPC paths.
- Label units with Bombcell when enabled.
- Run merge phases only when enabled and sequenced.
- Clean concat caches only when cleanup is selected/enabled and safe.

Known spikesort phases:

- `bootstrap_concat_binary`: materialize limited selected segments into the concat binary/cache used for sort.
- `sort`: run Kilosort-backed sorting through the selected engine.
- `summarize_sort`: optional compact sort summary.
- `bombcell_label`: compute and write Bombcell-style labels and summaries.
- `merge`: umbrella/legacy merge selector where supported.
- `merge_SLAy`: optional SLAy merge recommendation/application path.
- `merge_si_auto`: optional SpikeInterface automerge path.
- `merge_unitmatch`: optional UnitMatch-based merge path.
- `cleanup_concat_binary`: remove regenerable concat binary/cache artifacts after dependent phases finish.

Spikesort acceptance cues:

- Segment limits reach `bootstrap_concat_binary` before the sort recording is built.
- `sort` uses the intended engine and never invokes nested Docker in container/HPC mode.
- Sort gating is phase-local; other phases keep allowed well parallelism.
- Downstream label/merge phases consume the selected sorter output and do not rerun unselected phases.

## Reconstruct Stage

Purpose: prepare analyzers/templates from selected units and segments, build merged templates, generate GTR/reconstruction artifacts, produce plots, write reports, and clear regenerable template caches.

Expected behavior:

- Consume spikesort outputs, labels, and preprocess segment recordings.
- Resolve required sources explicitly.
- Build or reuse segment analyzers according to config.
- Build merged templates for selected units from selected segment sources.
- Generate GTRs and reconstruction artifacts for selected units.
- Produce plots and reports without expanding beyond selected unit scope.
- Clear template caches only when cleanup is selected/enabled and safe.

Known reconstruct phases:

- `resolve_sources`: inspect candidate upstream artifacts and write source-resolution summaries.
- `analyzers`: build or reuse concat/segment analyzers.
- `build_templates`: build merged per-unit templates and template metadata.
- `compute_template_similarity`: optional template similarity/candidate comparison.
- `plot_templates`: render per-unit template plots.
- `report_templates`: write template reports.
- `reports`: aggregate reconstruct reports where supported.
- `generate_gtrs`: generate GTR/reconstruction artifacts.
- `plot_recons`: plot reconstructed units.
- `plot_branch_propagations`: plot branch propagation diagnostics.
- `plot_branch_velocities`: plot branch velocity diagnostics.
- `plot_unit_summary`: write per-unit summary plots.
- `report_recons`: write reconstruction reports.
- `report_recon_grid`: write reconstruction grid reports.
- `report_full_chip_layout`: write full-chip layout reports.
- `report_summaries`: write summary report artifacts.
- `clear_templates_cache`: clear regenerable template/analyzer caches.

Reconstruct acceptance cues:

- Unit and segment limits apply before analyzer/template/GTR/plot/report work.
- Unit label filters are respected.
- Direct unit-targeted reruns do not overwrite unrelated unit outputs unless explicitly configured.
- Plot/report phases are optional and do not recompute heavy analyzer/template artifacts without a documented reason.

## Required Tests And Acceptance Criteria

### Phase sequence test

Acceptance criteria:

- Enabled phases in `phase_sequence` run in order.
- Disabled phases in `phase_sequence` are skipped with a reason.
- Configured phases omitted from `phase_sequence` do not run.
- Results and summaries identify run, skipped, and failed phases.

### Direct phase test

Acceptance criteria:

- Selecting `stage.phase` runs only that phase.
- Required inputs are checked before heavy work.
- Missing inputs fail clearly and do not trigger broad upstream execution.
- CLI debug and target flags are honored.

### Mixed selector test

Example:

```bash
axon-recon-container stages spikesort reconstruct.plot_templates \
  --config debug/debug.runtime.yml \
  --limit-datasets 1 \
  --limit-wells 1 \
  --limit-segments 2 \
  --limit-units 3
```

Acceptance criteria:

- Full `spikesort` runs for the selected scope.
- Only `reconstruct.plot_templates` runs after spikesort.
- Unselected reconstruct phases do not run.
- Logs and summaries make the selector behavior obvious.

### Resume and force-restart smoke

Acceptance criteria:

- A second run without `--force-restart` reuses complete valid artifacts or logs explicit skip reasons.
- A run with `--force-restart` deletes only selected target artifacts.
- Source `.h5` files are never deleted.
- Regenerated artifacts have fresh logs/summaries.

### Output contract smoke

Acceptance criteria:

- Each selected phase writes its summary artifact.
- Downstream phases can resolve upstream artifacts by manifest/summary paths.
- Large regenerable caches are not promoted to final durable outputs without a documented reason.
- Cleanup phases remove only their owned cache paths.
