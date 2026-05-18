# Open questions

TBD decisions awaiting user input or empirical data. Each entry has a clear resolution criterion. When resolved, move the conclusion to `current_state.md`, `guardrails/`, or a plan; delete the entry from here.

## Package naming

- **`kssynth` vs `unitprep` vs other**: package working name is `kssynth` per `plans/active/ks_synthesizer_package_plan.md` §10. User to confirm or pick alternative before slice 1.
- **`unitlink` vs `unitmatch_runner` vs `unittrack`**: working name is `unitlink` per `plans/active/unitmatch_runner_package_plan.md` §10. User to confirm or pick alternative before slice 1.
- **Repo hosting**: GitHub under user's own org? Both plans note this is TBD.

## Phase roster cleanup (`plans/active/phase_roster_cleanup_plan.md`)

- **`init` stage scope**: is `copy_src_to_scratch` the only phase that belongs there, or are there future "once per data config" setup phases worth scaffolding for now? Defer to plan §6 §1.
- **`cleanup` stage scope**: the plan envisions consolidating `cleanup_concat_binary` / `cleanup_analyzers` / `clear_templates_cache` into the new `cleanup` stage eventually. Decide ordering with the broader phase-roster work.
- **`concat_binary` resource class** after consolidation: keep spikesort-side budget? Plan §6 §3.
- **bombcell / SLAy code deletion timing**: after disabling in spikesort phase_sequence, the phase implementations stay in code until the recon-side migration lands. When do we fully delete? Plan §6 §5.
- **`plot_raster_threshold` quality fix design**: needs design-doc-level thinking about colormap / per-segment channel toggling visualization. Defer.

## UnitMatch / unitlink

- **Two-halves split granularity**: temporal midpoint is what UMPy expects. Could split finer for more same-neuron pairs per unit, but `UMPy` shape is hardcoded to `(..., 2)`. Decide after v1 results.
- **Per-chip match threshold tuning**: default `match_threshold: 0.5` from UMPy may be too permissive for HD-MEA. Add per-group calibration in unitlink v3? Wait for v1 + v2 empirical data.
- **Network-scan inclusion as default**: decide after `unitmatch_phase_plan.md` slice 7's measurement of marginal gain.
- **DeepUnitMatch HD-MEA training**: wrapper supports it; training a HD-MEA model is its own project. Defer.

## --force-restart collapse

- **`force_replot` final fate**: collapse into `--force-restart` with a flag, or eliminate entirely? Plan and tracker note both options. Decide during the collapse-implementation slice.
- **Stage-vs-phase scope inference**: `axon-recon stages spikesort --force-restart` is clearly stage-level. `axon-recon stages spikesort.merge_SLAy --force-restart` is clearly phase-level. Make sure the CLI dispatch is unambiguous; document in `guardrails/force_restart.md` once the collapse lands.

## resources.profiles elimination

- **Per-srun-flag fallback when slot is missing**: when `--cpus-per-task` is passed on the command line, does the resolver use that directly, or compute from `os.sched_getaffinity(0)`? Both are reasonable; pick during the implementation slice. See `trackers/tech_debt.md` §"Minimize / eliminate `resources.profiles`".

## Iteration scope (current cycle)

- **Iteration output dir naming convention**: `/pscratch/sd/a/adammwea/dev_outputs/<feature>/...` — `<feature>` is the plan slice or feature branch name? Confirm convention as Claude lands the first iteration commit.
