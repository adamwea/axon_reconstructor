# Tech debt — cleanup / refactor / repo-footprint backlog

Working code that we want to improve: dead code, obsolete config, duplicate logic,
oversized files, missing tests, slow paths. Distinct from `issues.md` (which is
broken behavior) and `roadmap.md` (which is new ambitions).

## Format

```
### <Short title>
- **Status**: open | in-flight | landed-in <commit> | wont-fix
- **Tags**: dead-code, duplicate-logic, obsolete-config, oversized-file, slow-path, missing-test, repo-footprint, …
- **Where**: pointer to file/symbol/dir
- **Why it's debt**: 1 paragraph
- **Suggested cleanup**: rough shape; rough touch size (S/M/L)
- **See also**: related commits, plans, prior discussion
```

---

## Open entries

### ~~Scaffold `tools/bootstrap_editable_deps.sh`~~ — SUPERSEDED 2026-05-19
- **Status**: superseded by `plans/active/env_install_unification_plan.md` slice 4 (`tools/install_dev_siblings.{sh,py}`)
- **Notes**: original entry called for a single bootstrap script to install sibling editables. The unification plan absorbs this into a flag-driven entry point (`tools/setup_env.sh --editable-siblings`) plus a sub-helper (`tools/install_dev_siblings.{sh,py}`), aligning sibling installs with the broader migration of pip deps onto `pyproject.toml` extras. Tracker entry kept for audit; remove on next prune.

### Phase-level auto-restart granularity in monolithic stage runners
- **Status**: SHIPPED (slice 14c approach A — target-level skip — landed for all 4 stages: **preprocess** `ee2ec7c`, **analysis** `2093977`, **reconstruct** `847e197`, **spikesort** `2d88217`). Approach (B) — per-phase index passing — remains as long-term debt; see "Suggested approach (B) cleanup" below.
- **Tags**: refactor, contract-tightening, parallelism
- **Where**: `pipeline/runner.py:run_<stage>_from_runtime` for spikesort, reconstruct, analysis (preprocess shipped 2026-05-19). Each calls a monolithic per-target stage runner that executes the entire phase_sequence as one block; the slice 14c skip happens before that monolithic call.
- **Per-stage status**:
  - `preprocess`: SHIPPED (`ee2ec7c`). New helper `target_all_preprocess_phases_succeeded(inputs)` in `stages/preprocess/runner.py`; `_worker` returns a `{stage: preprocess, status: skipped, reason: all_phases_ok}` sentinel before dispatching `run_preprocess(inputs)`. 5 new tests.
  - `analysis`: SHIPPED (`2093977`). Analysis dispatches each phase separately rather than monolithically, so the slice 14c here is a per-phase per-target short-circuit: `target_analysis_phase_summary_ok(phase_name, target, stage_config)` in `stages/analysis/runner.py` checks `manifest.json` (compute_metrics) or `context/unitmatch_summary.json` (unitmatch) and `_run_analysis_phase_from_runtime`'s `_worker` returns a skip sentinel when the summary shows ok/skipped/noop. 9 new tests.
  - `reconstruct`: SHIPPED (`847e197`). New helper `target_all_reconstruct_phases_succeeded(inputs)` in `stages/reconstruct/runner.py` walks `inputs.phase_sequence` and reads each phase's `summary_json_relpath` off `inputs.phases.<phase_name>`. `_run_reconstruct_substage_from_runtime`'s `_worker` short-circuits only when `stage_name == "reconstruct"` (full-stage dispatch); individual phase substage entry points keep their slice 13 per-phase idempotency. 6 new tests.
  - `spikesort`: SHIPPED (`2d88217`, partial). New helper `target_all_spikesort_phases_succeeded(target, stage_config, phase_plan)` covers the convention-named-relpath phases (concat_binary, plot_concat_*, cleanup_concat_binary, cleanup_analyzers, bombcell_label*) via the `_SPIKESORT_PHASE_SUMMARY_RELPATH_ATTRS` mapping. Phase plans containing sort/summarize_sort/merge_*/snapshot_sorter_output/concat_analyzer (the expensive sort-path phases) fall through to normal dispatch; those phases manage their state via slice 13's `with_checkpoint_marker` but don't yet surface a convention-named relpath on stage_config. The full target-level skip for sort-anchored phase plans waits on approach B (per-phase index passing). 6 new tests.
- **Why slice 14c is partial**: spikesort/reconstruct have ~10K-line runners with many phases and heterogeneous summary path conventions. Per-stage helper takes more like 60-90 LoC + careful test coverage rather than the ~30 LoC the user-injection estimated. Shipping preprocess first to lock down the pattern; remaining stages will follow as smaller targeted commits.
- **Why approach (B) is still here as long-term debt**: even after approach (A) ships for all 4 stages, when SOME phases are broken the monolithic runner still re-runs the WHOLE phase_sequence rather than only from the broken phase forward. For init + cleanup (per-phase dispatch loops), the full semantic is in effect via slice 14b. The discrepancy is contained but real.
- **Suggested approach (B) cleanup**: refactor each monolithic per-target stage runner to accept a `skip_phases_before_index` parameter (or equivalent), and instrument each existing phase dispatch with a pre-check. Then thread `find_first_broken_phase` into `run_<stage>_from_runtime` and pass the resolved index downward. Touch size: M (per stage).
- **See also**: `dev/notes/brain/open_questions.md` slice-14c entry; `phase_roster_cleanup_plan.md` slices 14a/14b/14c; `brain/guardrails/force_restart.md`

### Remove `debug_mode` YAML blocks across all stages
- **Status**: open
- **Tags**: obsolete-config, duplicate-logic, repo-footprint
- **Where**: `debug/debug.runtime.yml` — every stage's `phases.<phase>.debug_mode:
  {enabled, limit_datasets, limit_wells, limit_wells_per_dataset}` block.
  Parsed into ~20 `<phase>_debug_mode_*` fields per stage in
  `src/axon_recon/pipeline/stages/<stage>/config.py`.
- **Why it's debt**: the CLI args `--target-dataset`, `--limit-datasets`,
  `--limit-wells`, `--limit-wells-per-dataset` are now the source of truth for
  scoping a run. The YAML debug_mode blocks are dead duplication that clutter
  the runtime config and the stage-config dataclasses.
- **Suggested cleanup**: drop YAML blocks → drop dataclass fields → drop parser
  logic → drop tests asserting the fields → grep audit confirms zero residual
  hits. One commit per stage for bisectability; total touch is M (large mechanical
  diff, no behavior change once the YAML is gone). Apply to all four stages
  (preprocess, spikesort, reconstruct, analysis). User-confirmed ordering: do
  this BEFORE the `--force-restart` audit (see issues.md), because it shrinks
  the surface area that the force-restart contract has to reason about.
- **See also**: discussed 2026-05-11; partial commit
  `62568c1 commented out some debug_mode leftovers` started in this direction.

### Container runtime alignment with NERSC shifter + MPI backend
- **Status**: open
- **Tags**: infra, container, repo-footprint
- **Where**: `src/axon_recon/pipeline/container_cli.py` (the `axon-recon-container`
  wrapper); `debug/guardrails/container_mpi4py_NERSC_optimization_guardrails.md`;
  `debug/guardrails/container_mpi_strategy_note.md`;
  `debug/plans/active/nersc_shaped_local_affinity_plan.md` (probably
  the active home when this work resumes — verify status).
- **Why it's debt**: local `axon-recon-container` uses Docker-shaped bind mounts,
  root user, isolated network, isolated `/tmp`+`/home`. NERSC uses `shifter`:
  host UID passthrough, host networking, host paths. Behavior divergence between
  local dev and NERSC deploy means bugs surface late. MPI backend works
  transparently under shifter but needs explicit fabric / network-mode handling
  under Docker.
- **Suggested cleanup**: opt-in `--runtime=shifter-mimic` flag on
  `axon-recon-container` that flips `--user $(id -u):$(id -g)`, `--network=host`,
  passes `$HOME` and `/tmp` through. Align local `mpirun` invocation with the
  NERSC-side `srun` shape. Add a 2-rank MPI smoke that exercises a stage
  end-to-end under the mimic mode. L touch — likely earns its own plan doc.
  Decide later whether mimic mode becomes the default.
- **See also**: discussed 2026-05-11.

### `recon_outputs_` sibling directory cleanup
- **Status**: open
- **Tags**: dead-code, repo-footprint
- **Where**: in the wild under
  `/mnt/disk15tb/adamm/scratch/axon_recon_scratch/outputs/.../well*/recon_outputs_/`
  (note trailing underscore). The canonical recon output dir is `recon_outputs/`;
  the `_` suffix is an older snapshot from an earlier pipeline version.
- **Why it's debt**: scratch-level clutter that confuses agents and humans alike
  ("which one is current?"). No code references it — it's purely on-disk drift.
- **Suggested cleanup**: scratch-side `find -name recon_outputs_ -type d | xargs
  rm -rf` once we've verified nothing depends on it. S touch. Not a code change,
  more of a one-shot cleanup. Could become a `debug/scripts/` helper.

### `spikesort/runner.py` is ~10K lines
- **Status**: open
- **Tags**: oversized-file
- **Where**: `src/axon_recon/pipeline/stages/spikesort/runner.py`.
- **Why it's debt**: searching this file is a pain, even with grep-first
  discipline. The recent spikesort-merge-cleanup work split out
  `core/concat_analyzer.py`, `core/post_merge_view.py`, `core/pre_merge_cache.py`,
  `core/derive_post_merge.py`, `core/snapshot_sorter_output.py` — which helped —
  but the merge orchestrator + report writers + extension helpers still live in
  `runner.py`.
- **Suggested cleanup**: extract `_write_merge_unit_location_reports` +
  `_write_merge_template_heatmap_reports` (the two largest report writers) into
  `core/merge_reports/` modules. Extract the SLAy orchestrator
  (`_run_slay_merge_method` and friends) into `core/slay_orchestrator.py`. Each
  extraction is one commit, mechanical (move + update imports). M-L total touch.
  Earns its own plan when scoped.

### Soften container preflight when `publish_outputs: false`
- **Status**: SHIPPED 2026-05-19 (commit ~latest, see commit_log)
- **Tags**: infra, container, ergonomics
- **Where**: `src/axon_recon/pipeline/container_cli.py:_resolve_config_mounts`.
- **Why it's debt**: the preflight tries to `mkdir -p` the data config's
  `output_root` even when the runtime sets `publish_outputs: false` (i.e., the
  run will never actually write to `output_root`). Forces users to know the
  `--no-config-mounts` bypass even when the unavailable path is irrelevant to
  the run.
- **Suggested cleanup**: detect `publish_outputs: false` in
  `_resolve_config_mounts` and skip the `output_root` mkdir check (still mount
  it `:rw` if the path exists, but don't fail when it doesn't). S touch — single
  function edit + test. Eliminates the need for `--no-config-mounts` in 90%+
  of the cases where it's currently required.
- **See also**: see `debug/trackers/issues.md` "NAS mount … stale".

### Remove concat analyzer plumbing from recon stage
- **Status**: open (production path already force-disabled this turn)
- **Tags**: dead-code, repo-footprint, recon
- **Where**: 287 references across:
  - `src/axon_recon/pipeline/stages/reconstruct/templates/integrations/spikeinterface_extract.py` (~132 hits — `concat_analyzer_relpath`, `concat_sorting_relpath`, `concat_policy`, `concat_use_existing_analyzer`, `concat_build_if_missing`, `_build_concat_analyzer_from_sorting_and_recording`, `concat_analyzer_obj`, every "concat" branch in `load_spikeinterface_analyzers`)
  - `templates/runner.py` (~49 hits)
  - `templates/config.py` (~44 hits — `concat_phase_cfg`, `legacy_include_concat`, every `concat` sub-block parse)
  - `templates/core/merge.py` (~15 hits — `materialize_templates_from_spikeinterface`'s `concat_analyzer` selection)
  - `templates/models/inputs.py` (~6 hits — dataclass fields `include_concat`, `require_concat_analyzer`, `concat_*_relpath`, `preprocessed_concat_reldir`)
  - `phases/analyzers.py` (~33 hits — phase config builder for the `concat` sub-block)
  - `phases/build_templates.py` (~8 hits)
  - tests under `templates/tests/` (~11+ tests that still pin `include_concat=True` and assert concat-loading behavior)
- **Why it's debt**: the concat analyzer was built by `spikesort.concat_analyzer` BEFORE `merge_SLAy` mutated `sorter_output` in place, so its embedded sorting captured pre-SLAy unit IDs. When recon's templates phase used `analyzers[0].sorting.unit_ids` as the authoritative unit list, it iterated pre-SLAy IDs, and the downstream label filter against the post-SLAy `cluster_KSLabel.tsv` silently dropped absorbed-by-merge IDs → systematic post-merge template undercount. This turn (commit pending) force-disables `include_concat` at every recon production wrapper (`_iter_templates_phase_analyzers`, `_load_templates_phase_analyzers`, `materialize_templates_from_spikeinterface` call sites) and flips the YAML legacy default from `True` to `False`, but leaves the parameter plumbing in place so the tests still pass.
- **Suggested cleanup**: delete every concat-related parameter, dataclass field, YAML config builder, and code branch listed above. Update tests: keep ones that exercise segment-only behavior (drop the `include_concat=True` kwarg), delete ones whose entire purpose is asserting concat-loading outcomes (`test_load_spikeinterface_analyzers_builds_dense_concat_from_sorting_and_preprocessed_concat`, `test_load_spikeinterface_analyzers_rebuilds_concat_without_reusing_existing_analyzer`, etc.). The `spikesort.concat_analyzer` PHASE upstream of recon is a separate concern — that phase still builds a concat analyzer file on disk, even though recon ignores it; whether to also delete the spikesort.concat_analyzer phase belongs in the "Finalize the phase roster" decision doc above. Touch is L (a few hundred mechanical line deletions + ~6 tests dropped, no algorithmic change). Worth its own scoped plan.
- **See also**: the bombcell pass2 KS-extractor inner-join bug in `roadmap.md` ("Post-templates bombcell + SLAy pass") describes a related artifact-management issue: SLAy mutates `cluster_KSLabel.tsv` / `cluster_group.tsv` but not `cluster_Amplitude.tsv` / `cluster_ContamPct.tsv`, so loaders that inner-join those TSVs end up with the pre-merge unit set. Both bugs trace to the same root: assuming the `cluster_*.tsv` family is a coherent post-merge view when SLAy only updates some of it.

### Minimize / eliminate `resources.profiles` in favor of srun / MPI / native affinity
- **Status**: open
- **Tags**: obsolete-config, duplicate-logic, infra
- **Where**: `src/axon_recon/pipeline/resources.py` (`ResourcesConfig.profiles`,
  `active_profile`, `parse_resources_config`); `src/axon_recon/pipeline/config.py`
  (`parse_resources_config_for_bundle`, `_ACTIVE_PROFILE_OVERRIDE`); `cli.py`
  (`--profile` / `--task-profile`); `dev/debug_NERSC/debug.runtime.yml`
  (`resources.profiles.perlmutter_cpu` / `perlmutter_gpu` etc.).
- **Why it's debt**: YAML-defined task profiles (`cpus_per_task`,
  `tasks_per_node`, slot caps, …) duplicate information slurm/MPI/cgroups
  already enforce on the running process. Keeping them in sync with the actual
  srun flags (`-c`, `--gpus`, `--mem`) is a constant footgun: the spikesort_full
  gate-deadlock (gpu_sort_slots=0) and the bootstrap `n_jobs=1` regression
  (resolved-from-profile=16 but slot ContextVar not propagated through the MPI
  worker) both trace to profile/CLI drift. Every new sbatch has to remember to
  pass `--profile perlmutter_gpu`, and every YAML edit has to be cross-checked
  against the sbatch flags. The profile concept also imposes a process-wide
  override pattern that we now have to thread through every fresh
  `parse_resources_config` call site.
- **Suggested cleanup**: collapse to a single runtime that reads its own
  affinity (`os.sched_getaffinity(0)`), `SLURM_CPUS_PER_TASK`, `SLURM_GPUS_*`,
  cgroup limits, and MPI rank/size, then derives `cpus_per_task` / slot caps
  from those. Drop the YAML `profiles` block, drop `active_profile` / `--profile`,
  drop `_ACTIVE_PROFILE_OVERRIDE` and the bundle's `active_profile_override`
  field. Resource classes stay (they encode phase-level RAM/slot DEMANDS, which
  isn't redundant with srun), but the profile-level supply side comes from the
  environment. Add a native local-affinity reader for non-slurm runs (already
  partially present in `cpu_allocation._default_affinity_getter`). Touch is L
  — the change is mechanical but spans the resource-gate, the CPU allocation
  resolver, every sbatch script, and the YAML schema. Worth scoping a plan
  before starting.
- **See also**: commit `c205d14` (added `--profile` override as a workaround);
  `dev/debug_NERSC/jobs/sans_bombcell_rerun/sbatches/*.sbatch` (every sbatch
  currently has to redundantly pass `--profile perlmutter_gpu`).

### Finalize the phase roster before any destructive cleanups land
- **Status**: open
- **Tags**: ordering, scoping, blocker
- **Where**: `stages.*.phase_sequence` in every runtime YAML (canonical:
  `src/axon_recon/default.runtime.yml`; downstream: `dev/debug_NERSC/debug.runtime.yml`).
  Each `phase_sequence` references phases defined under
  `stages.<stage>.phases.<phase>:` in the same YAML and implemented by a
  `_run_<stage>_<phase>_target` runner in `src/axon_recon/pipeline/stages/<stage>/runner.py`.
- **Why it's debt**: several upcoming cleanups (the force-restart collapse
  below, the `debug_mode` YAML purge above, the `resources.profiles` removal
  above, the `--force-restart` issues.md entry, and the
  `_cleanup_spikesort_outputs_for_force_restart` allowlist this turn) all
  touch the same surface: per-phase YAML blocks, per-phase output dirs, and
  per-phase cleanup helpers. Each phase we keep is dozens of lines of YAML
  + code to migrate; each phase we kill is dozens of lines deleted with zero
  migration cost. Doing those cleanups BEFORE the roster is settled means
  we pay the migration tax on phases we're about to delete anyway, and we
  re-touch the same files twice.
  Known candidates that need a keep/kill/reshape decision before the
  destructive passes land:
  - `spikesort.summarize_sort` — currently skipped in `phase_sequence` (yml
    line ~603); has its own resource class entry and output file. Keep or
    fold into `sort`?
  - `spikesort.bombcell_label` / `spikesort.bombcell_label_pass2` — pass2 is
    disabled-by-default because of the KiloSortSortingExtractor inner-join
    bug (see `roadmap.md` "Post-templates bombcell + SLAy pass"). Decision:
    keep pass1 + reshape pass2 into a templates-aware post-recon phase, or
    delete the existing pass2 wiring entirely and rebuild from scratch when
    the templates-aware version lands.
  - `spikesort.cleanup_concat_binary` / `spikesort.cleanup_analyzers` —
    once force-restart wipes the stage dir uniformly, are these still
    earning their keep, or do they become trivial helpers folded into the
    stage runner?
  - `reconstruct.analyzers` — collapsed to segment-only this week (commit
    `c205d14`); the concat-analyzer half of the phase should be physically
    deleted from code, not just disabled.
  - Any `_pass2`-style phase across stages that we previously prototyped
    and shipped disabled.
- **Suggested cleanup**: produce one document under `dev/notes/plans/` that
  lists every phase across the four stages, marks each as keep / kill /
  reshape, and (for reshapes) links to the relevant roadmap/issues entry.
  Drop the kill list FIRST (one commit per phase removed: YAML block +
  stage runner branch + tests + output-dir cleanup logic + any
  cross-references). Then proceed to the force-restart collapse,
  `debug_mode` purge, and `resources.profiles` removal with confidence
  that we're only migrating phases that will survive. S touch for the
  planning doc; M-L per phase removed. Order: this → debug_mode YAML
  purge → force-restart collapse → resources.profiles removal.
- **See also**: this turn's commits on `_cleanup_spikesort_outputs_for_force_restart`
  expanded the cleanup allowlist as a stopgap; one entry per phase added or
  removed will eventually rewrite that helper. `roadmap.md` "Post-templates
  bombcell + SLAy pass" is the main reshape candidate. `issues.md`
  "`--force-restart` does not reliably clean prior artifacts" is the
  user-facing motivation.

### Collapse `--force-restart` semantics to "delete the relevant output folder, full stop"
- **Status**: open
- **Tags**: duplicate-logic, obsolete-config, repo-footprint, ergonomics
- **Where**: every stage runner's force-restart handling. Today's hotspots:
  - `src/axon_recon/pipeline/stages/spikesort/runner.py` —
    `_cleanup_spikesort_outputs_for_force_restart` (explicit allowlist of
    dirs/files), plus per-phase `*_delete_outputs_on_force_restart` flags
    (`bootstrap_concat_binary_overwrite_on_force_restart`,
    `sort_delete_outputs_on_force_restart`,
    `bombcell_label_delete_outputs_on_force_restart`,
    `slay_delete_outputs_on_force_restart`,
    `merge_delete_outputs_on_force_restart`,
    `slay_force_restart_retrain_model`, …) scattered across `runner.py` and
    the YAML.
  - Equivalent per-phase flags + per-phase cleanup heuristics in
    `stages/preprocess/runner.py`, `stages/reconstruct/runner.py`,
    `stages/analysis/runner.py`.
  - ~~`force_replot` is a sibling escape hatch with its own partial-cleanup
    semantics; same blast radius.~~ (SHIPPED 2026-05-19 in phase_roster slice 11,
    commit `2a7a7e4`: `--force-replot` deleted, `--replot` is the replacement.)
- **Why it's debt**: today, `--force-restart` means N different things
  depending on which phase reads which YAML flag. Some phases delete only
  some sub-paths, some preserve "partial" caches (model caches, snapshot dirs,
  per-segment intermediates) and try to resume from them, some honor the flag
  only when a sibling phase also writes a marker. The complexity has produced
  a stream of "stale leftover artifact" bugs (e.g. `bombcell_label_outputs/`
  surviving a `--force-restart` and contaminating the SLAy status reporter)
  and the per-phase YAML flags duplicate information the stage's
  `phase_sequence` already implies. Net result: users can't trust
  `--force-restart` to actually start over, and contributors have to read
  ~5 different cleanup helpers to know what a flag actually does.
- **Suggested cleanup**: enforce the simplest possible rule everywhere:
  - **Stage force-restart for a well** → `rmtree(<well>/<stage_output_root>/)`
    before anything else runs, then proceed. No exceptions, no "preserve
    cache".
  - **Phase force-restart for a well** → `rmtree(<well>/<stage_output_root>/<phase_output_dir>/)`
    before the phase runs. The phase output dir is whatever the phase's
    `rel_output_root` / `relpath` resolves to.
  - Anything more clever (partial restarts, per-step caches kept across
    force-restart, "force-restart but only the model retrain") gets DELETED
    from code, the YAML, and the tests. No fallback paths. No backward-compat
    shims. Phases that need a fresh cache just rebuild it from scratch every
    time — disk I/O is the right tax for the simplicity dividend.
  - ~~`force_replot` likely gets the same treatment (collapse into
    force-restart-with-a-flag or remove entirely).~~ (SHIPPED — removed entirely
    in `2a7a7e4`; `--replot` is the replacement, which is orthogonal to
    `--force-restart` rather than being absorbed into it.)
  - Delete every `*_delete_outputs_on_force_restart` / `*_overwrite_on_force_restart`
    YAML knob and every `getattr(stage_config, "..._on_force_restart", ...)`
    call site. The phase output dir IS the unit of restart granularity.
- **Touch size**: L. Hits 4 stage runners + all stage YAMLs + tests. Tests
  asserting partial-restart behavior should be deleted, not migrated.
- **Ordering**: BLOCKED on "Finalize the phase roster before any destructive
  cleanups land" (above) — every phase we kill in that step is one fewer
  phase whose cleanup helper we have to migrate here. Then ordered AFTER
  the `debug_mode` YAML cleanup so we're not editing the same YAML blocks
  twice. Worth its own scoped plan when those prerequisites are clear.
- **See also**: today's commit on `_cleanup_spikesort_outputs_for_force_restart`
  expanded the allowlist as a stopgap; that whole function should go away when
  this lands. The stale-bombcell SLAy mtime fallback in
  `status._read_slay_label_column` (`bc_is_fresh`) becomes dead code once
  force-restart actually cleans the stage dir — drop it too.

## Device-agnostic source data (low priority, design-only)

**Problem**: `axon_recon` currently runs on MaxWell HD-MEA devices (MaxOne @ 20 kHz, MaxTwo @ 10 kHz). The pipeline already does the RIGHT thing in the sense that `metadata_get` + `debug.data.yml` are authoritative for per-recording sample rate and channel count — but there are scattered assumptions across phases that effectively encode "MaxWell-shaped data":
- Some plot phases assume 26,400 electrodes' worth of channel-map shape
- Some YAML defaults assume ~17.5 μm electrode pitch
- Recording-loader code paths are MaxWell-flavored (file format, channel-mapping conventions)

**Why it matters for the future**: Pipeline will eventually consume data from ThreeBrain and Sony HD-MEA devices. Those will have:
- Different electrode counts (TBD per device)
- Different pitches (TBD)
- Different sample rates (TBD)
- Different raw file formats (likely)
- Different channel-mapping / activity-scan conventions

**Goal**: Every phase that consumes recording data should:
1. Read device parameters from the analyzer manifest at runtime — never hardcode 26400, 17.5 μm, or 20 kHz anywhere in src/.
2. Scale derived quantities (radii in electrodes, pitch-dependent thresholds, upsample factors) as ratios of manifest-reported values, not as absolutes.
3. Tolerate any HD-MEA-family device whose analyzer manifest reports valid `(sample_rate_hz, n_channels, electrode_locations[n,2], pitch_um)` quadruple.
4. Where a phase legitimately can't be device-agnostic (e.g. a plot's grid layout is shaped to a specific array), gate it on a `supported_devices` declaration and skip cleanly on others.

**Suggested cleanup approach (when this becomes a priority)**:
- Audit pass: `grep -rn '26400\|17.5\|20.0\|20_000\|0.05.*1000' src/` — any literal that matches a MaxWell spec is suspect.
- Introduce `DeviceProfile` dataclass (device_name, n_channels, pitch_um, sample_rate_hz_default, raw_file_format, ...) read from the analyzer manifest. Every phase consumes the profile, not raw constants.
- For per-device YAML calibration sets, use `device_profile: maxone | maxtwo | threebrain | sony` as a top-level YAML key selecting which sub-block of hyperparameters applies. Defer until empirical results show different devices need different calibrations.

**Touch size**: M-L. Distributed across many phases. Best done incrementally — touch one phase per slice as it's audited for new-device support.

**Ordering**: Strictly post-Era-3 (kssynth + unitlink + Radivojevic integration). No new devices in scope for v1 iteration; this is a v2+ concern. Document the principle now (per user 2026-05-21) so future contributors don't bake in new MaxWell assumptions; defer execution until a non-MaxWell device actually enters scope.

**See also**: `project-axon-recon-device-diversity` auto-memory captures the per-device sample-rate awareness rule that's already in effect.
