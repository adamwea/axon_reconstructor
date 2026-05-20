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
- **Dashboard slice 7 tertiary-grouping UX**: third grouping dropdown for box/bar plots. Two reasonable UX paths:
  - (A) **Faceted small-multiples** — one plot per tertiary value. Scales visually with up to ~6-9 facet values; breaks down for high-cardinality tertiary (e.g. DIV ∈ {6,8,10,...,32}). Easy to read individual sub-plots; harder to compare values across sub-plots.
  - (B) **Nested hierarchical X-axis** — render primary × secondary × tertiary as compound x-tick labels (e.g. `wt | DIV10 | media_a`, `wt | DIV10 | media_b`, …). Scales to higher cardinality; keeps the comparison axis intact; visually busier. Plotly supports this via `category_orders` + multi-level group_col.
  - **Recommendation**: ship (B) first because it's the lower-risk extension of the existing box-plot rendering (slice 6 already renders nested groups for secondary; tertiary is just another dimension to fold into the category sort). Add (A) later as an opt-in `tertiary_mode: facet` knob if users want it. **User input wanted**: confirm (B)-first, or override to (A) if you want small-multiples as the default. **Resolution criterion**: user picks one; loop ships that slice 7.

## Empirically TBD per slice

- **unitlink slice 5 (UnitMatchPy wrapper) blocked on UMPy install** — **RESOLVED 2026-05-19, REDO SHIPPED 2026-05-19**: UMPy + mat73 installed in conda env. `unitlink` commit `9a3b331` ships the redo: `backends/classical.py` now drives the real UMPy spine — `extract_parameters → extract_metric_scores → get_parameter_kernels → apply_naive_bayes → assign_unique_id` — with 2 real-call tests that exercise UMPy end-to-end on synthetic 82-sample 16-channel waveforms. The earlier mock-only impl had called a non-existent `bayes_functions.run_bayes`; that bug would have surfaced at the first real call. 53/53 unitlink tests green. Next gating items: `kssynth_plan` slice 9 (axon_recon recon-stage integration) and `unitmatch_phase_plan` slice 5 (enable + login-node smoke).

- **Slice 14c integration shape for preprocess/spikesort/reconstruct/analysis** — **RESOLVED 2026-05-19**: Approach **(A) target-level skip** in all four monolithic stages now. Book tracker entry for approach (B) phase-level refactor as a future tech-debt item. Implementation directive in `current_state.md` USER INJECTIONS entry [2026-05-19] item 2.

- **resources.profiles elimination — per-srun-flag fallback when slot is missing**: when `--cpus-per-task` is passed on the command line, does the resolver use that directly, or compute from `os.sched_getaffinity(0)`? Both are reasonable; pick during the implementation slice. See `trackers/tech_debt.md` §"Minimize / eliminate `resources.profiles`".

- **Auto-restart with chip-well-group phase scope** (`unitmatch` phase): the auto-restart logic walks `phase_sequence` per target. For chip-well groups, the "target" is a group, not a (dataset, well) pair. Verify the logic generalizes when the unitmatch phase lands.

- **Pre-existing test failures surfaced during phase_roster_cleanup slice 1** (2026-05-18): originally 15 failures; mostly resolved during overnight iteration 2026-05-19. Remaining failures: 5× `test_spikeinterface_extract.py` (parallelism slice 9 scope). **Semantic decision RESOLVED 2026-05-19 — test wins**: `max_spikes_per_unit=None` means "expand to maximum available". Implementation directive: flip `_loaded_analyzer_extensions_satisfy_requested_payload` to return False when `normalized_max is None` AND cached waveforms count < `_full_waveforms` count (recompute needed to expand). Diagnose `test_build_unit_source_payload_retries_without_compat_only_kwargs` separately. Full implementation note in `current_state.md` USER INJECTIONS entry [2026-05-19] item 3. Resolved this iteration:
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
