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

- **resources.profiles elimination — per-srun-flag fallback when slot is missing**: when `--cpus-per-task` is passed on the command line, does the resolver use that directly, or compute from `os.sched_getaffinity(0)`? Both are reasonable; pick during the implementation slice. See `trackers/tech_debt.md` §"Minimize / eliminate `resources.profiles`".

- **Auto-restart with chip-well-group phase scope** (`unitmatch` phase): the auto-restart logic walks `phase_sequence` per target. For chip-well groups, the "target" is a group, not a (dataset, well) pair. Verify the logic generalizes when the unitmatch phase lands.

- **Pre-existing test failures surfaced during phase_roster_cleanup slice 1** (2026-05-18): 15 failures noted by sub-commit agents, all confirmed pre-existing via `git stash` comparison (not caused by the deletion). Worth a dedicated triage slice:
  - `parses_plot_templates_v2_phase_block` — resource_class `'plot_unit'` not in budgets registry. Likely needs the resource budget yaml to register it, or the test fixture to provide a registered class.
  - `test_load_config_reconstruct_populates_templates_inputs_from_debug_local_runtime` — test expects a specific phase_sequence including `templates_report_templates` and `clear_templates_cache`; the live `debug_local/debug.runtime.yml` already differs. Test assertion is stale; refresh in follow-up.
  - 3× `test_reconstruct_combined_phase_sequence_*` — `dataclasses.replace()` called on a `SimpleNamespace`. Tests scaffold `templates_inputs` as SimpleNamespace; production code path now requires a real `TemplatesInputs` dataclass. Fixture upgrade needed.
  - `test_reconstruct_phase_worker_allocation_uses_resource_class_cpu_for_downstream_phases` — likely related to phase_budgets schema drift.
  - `test_run_reconstruct_report_full_chip_layout_phase_writes_outputs` — unit count assertion drift.
  - `test_write_unit_circle_recon_plot_branches_only_scope_uses_raw_and_remaps` — pre-existing.
  - 5× `test_spikeinterface_extract.py` — separate spikeinterface API drift.
  - 2× `test_runner` upsampling/spikeinterface fallback tests — pre-existing.

  Resolution: not blocking phase_roster_cleanup slices 2-14, but worth a dedicated cleanup slice after the destructive cleanups settle.

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
