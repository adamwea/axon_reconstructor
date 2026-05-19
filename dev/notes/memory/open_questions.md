# Open questions

TBD decisions awaiting user input or empirical data. Each entry has a clear resolution criterion. When resolved, move the conclusion to `current_state.md`, `guardrails/`, or a plan; delete the entry from here.

## Awaiting empirical data (deferred until plan reaches the relevant slice)

- **Two-halves split granularity**: temporal midpoint is what UMPy expects. Could split finer for more same-neuron pairs per unit, but UMPy shape is hardcoded to `(..., 2)`. Decide after `unitlink` v1 results.
- **Per-chip match threshold tuning**: default `match_threshold: 0.5` from UMPy may be too permissive for HD-MEA. Add per-group calibration in `unitlink` v3? Wait for v1 + v2 empirical data.
- **Network-scan inclusion as default**: decide after `unitmatch_phase_plan.md` slice 7's measurement of marginal gain.
- **Both network-scan types in unitmatch v2**: v2 picks ONE type (lean clustered variant). The sparse variant may join in v3 if marginal gain measurement justifies it.
- **DeepUnitMatch HD-MEA training**: `unitlink` v2 wrapper supports it; training a HD-MEA model is its own project. Defer.
- **`force_replot` final fate**: RESOLVED — deleted entirely. `--replot` (without "force") replaces it. See `phase_roster_cleanup_plan` slice 11.
- **`init` / `cleanup` stage scope**: v1 = one phase each (`copy_src_to_scratch` / `wipe_src_scratch`); grow organically. Stages disabled by default for now but must work.
- **`concat_binary` resource class** after consolidation: keep spikesort-side budget. Plan §6 §3.
- **bombcell / SLAy code deletion timing**: never delete from spikesort code, just disable. User: "I think in the future we will only use the recon stage versions if we successfully implement them as I imagine, maybe then we delete them. but for now, just disable them."
- **`plot_raster_threshold` quality fix design**: needs design-doc-level thinking about colormap / per-segment channel toggling visualization. Defer.

## Empirically TBD per slice

- **unitlink slice 5 (UnitMatchPy wrapper) blocked on UMPy install**: `UnitMatchPy` is not installed in the `axon_recon` conda env (`~/.conda/envs/axon_recon/`). The package source lives at `/global/homes/a/adammwea/dev/pkgs/UnitMatch/UnitMatchPy/` but `pip install -e .` from that directory has not been run. Slice 5 (`backends/classical.py`) wraps UMPy's `extract_parameters → extract_metric_scores → bayes → threshold → UIDs` spine — without the install, tests can only exercise the import-failure path. Resolution criterion: install UMPy (`pip install -e /global/homes/a/adammwea/dev/pkgs/UnitMatch/UnitMatchPy/` or via PyPI when available), then implement slice 5 with real-call tests. Unblocks slice 7 (api.match orchestrator) and slice 9 (axon_recon unitmatch phase).

- **Slice 14c integration shape for preprocess/spikesort/reconstruct/analysis**: slice 14b's pattern (call `find_first_broken_phase` at the per-stage `run_<stage>_stage` entry, skip phases before broken index, run from broken onwards) is clean for init + cleanup because their stage runners are simple per-phase dispatch loops. For preprocess/spikesort/reconstruct/analysis, the full-stage runtime entry (`run_<stage>_from_runtime` in `pipeline/runner.py`) calls a MONOLITHIC per-target stage runner that runs the entire phase_sequence as one block — there's no convenient per-phase dispatch point to instrument. Two viable approaches:
  - **(A) Target-level skip only**: at the `run_<stage>_from_runtime` level, walk each target's summaries before dispatch. If `find_first_broken_phase` returns None, skip the target entirely; otherwise dispatch normally (the monolithic stage runner re-runs the whole sequence). Cheap to implement (~30 LoC per stage); achieves the most common case (re-run after success → no-op) but loses the "force-restart from broken phase onwards" semantic.
  - **(B) Phase-level granularity**: refactor the monolithic stage runners to accept a `skip_phases_before_index` parameter (or equivalent), and instrument each existing phase dispatch with a pre-check. Higher refactoring cost but matches the guardrail's full semantic.
  - **Resolution criterion**: pick when implementing. (A) is the right scope for "ship soon and iterate"; (B) is the right scope if the auto-restart contract needs to be airtight before tier 3 work.

- **resources.profiles elimination — per-srun-flag fallback when slot is missing**: when `--cpus-per-task` is passed on the command line, does the resolver use that directly, or compute from `os.sched_getaffinity(0)`? Both are reasonable; pick during the implementation slice. See `trackers/tech_debt.md` §"Minimize / eliminate `resources.profiles`".

- **Auto-restart with chip-well-group phase scope** (`unitmatch` phase): the auto-restart logic walks `phase_sequence` per target. For chip-well groups, the "target" is a group, not a (dataset, well) pair. Verify the logic generalizes when the unitmatch phase lands.

- **Pre-existing test failures surfaced during phase_roster_cleanup slice 1** (2026-05-18): originally 15 failures; mostly resolved during overnight iteration 2026-05-19. Remaining failures: 5× `test_spikeinterface_extract.py` (parallelism slice 9 scope). Root cause analysis from this iteration: tests assert that `build_unit_source_payload(max_spikes_per_unit=None)` expands the analyzer's waveforms extension from a limited cached set (3 spikes) to all available (5 spikes from `_full_waveforms`). Production at `templates/integrations/spikeinterface_extract.py:_prepare_analyzer_for_payload_extraction` short-circuits when `_loaded_analyzer_extensions_satisfy_requested_payload(normalized_max=None, has_waveforms=True, has_templates=True)` returns True — meaning "nothing requested + already have something → don't recompute". This is a contract design choice: does `max_spikes=None` mean "use whatever's cached" (current production) or "expand to maximum available" (what tests expect)? Both are reasonable. Resolution criterion: confirm the intended semantic with the user, then either (a) flip `_loaded_analyzer_extensions_satisfy_requested_payload` to NOT satisfy when `_full_waveforms` count exceeds cached count, OR (b) update the test contract to reflect "no recompute when nothing requested". The `test_build_unit_source_payload_retries_without_compat_only_kwargs` test (passes ms_before/ms_after/dtype) needs a separate diagnosis since `needs_prepare=True` should kick in there — likely a recompute path issue. Resolved this iteration:
  - ~~3× `test_reconstruct_combined_phase_sequence_*`~~ **RESOLVED 2026-05-19** by parallelism slice 7 (`7a277d6`).
  - ~~`test_reconstruct_phase_worker_allocation_uses_resource_class_cpu_for_downstream_phases`~~ **RESOLVED 2026-05-19** by parallelism slice 8 sub-item A (YAML fixture migrated to nested-profile + phase_budgets shape).
  - ~~`test_load_templates_config_parses_plot_templates_v2_phase_block`~~ **RESOLVED 2026-05-19** by parallelism slice 8 sub-item E2.
  - ~~`test_run_reconstruct_report_full_chip_layout_phase_writes_outputs`~~ **RESOLVED 2026-05-19** by parallelism slice 8 sub-item D (unit_ids fixture widened to match production unit-selection semantics).
  - ~~`test_write_unit_circle_recon_plot_branches_only_scope_uses_raw_and_remaps`~~ **RESOLVED 2026-05-19** by adding `invert_y_axis` propagation from `display_cfg` to `v2_cfg` in `core/unit_plots.py:write_unit_circle_recon_plot`.
  - ~~2× `test_run_reconstruct_templates_pipeline_*`~~ **RESOLVED 2026-05-19** by parallelism slice 8 sub-item E1 (flipped `include_concat is True` → `False` to match concat-analyzer-disabled production contract).
  - ~~`test_load_config_reconstruct_populates_templates_inputs_from_debug_local_runtime`~~ **RESOLVED 2026-05-19** by parallelism slice 8 sub-item C (phase_sequence updated for plot_templates_v2 insertion; dpi assertion now plumbing-only per plan).
  - ~~`test_run_preprocess_stage_logs_phase_start_per_well`~~ **RESOLVED 2026-05-19** by parallelism slice 8 sub-item B (phase_n_jobs assertion bumped to match production clamp).
  - ~~5× `test_container_cli` failures~~ **RESOLVED 2026-05-19** by softening `_build_docker_run_command` config-load on `--dry-run` (test mode shouldn't require the inner `--config` file to exist).

## Resolved 2026-05-18 (kept here briefly for context; delete on next prune)

All Q1-Q24 of the pre-loop scoping Q&A resolved. Decisions encoded in:
- `CLAUDE.md` (loop protocol, model selection, smoke-test ladder, commit cadence + restore rules, working-data scope)
- `guardrails/stage_phase_architecture.md` (checkpoint status enum + auto-restart-from-first-broken)
- `guardrails/force_restart.md` (three invocation modes: no-flag auto-restart, `--force-restart`, `--replot`; `--force-replot` eliminated)
- `guardrails/scope_flags.md` (flag table updates: `--replot` replaces `--force-replot`; new `--output-root`)
- `guardrails/dry_run.md` (dry-run interaction with checkpoint status)
- `plans/active/phase_roster_cleanup_plan.md` (slices 11-14 added for `--force-replot` deletion, `--output-root`, checkpoint markers, auto-restart logic)
- `plans/active/ks_synthesizer_package_plan.md` (name locked: `kssynth`)
- `plans/active/unitmatch_runner_package_plan.md` (name locked: `unitlink`)
- `plans/active/unitmatch_phase_plan.md` (v2 picks one network-scan type; data path noted)
- `memory/current_state.md` (working-data scope, queued tier order, known-good baselines)
