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
- **Status**: open (cross-listed in issues.md as workaround)
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
  - `force_replot` is a sibling escape hatch with its own partial-cleanup
    semantics; same blast radius.
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
  - `force_replot` likely gets the same treatment (collapse into
    force-restart-with-a-flag or remove entirely).
  - Delete every `*_delete_outputs_on_force_restart` / `*_overwrite_on_force_restart`
    YAML knob and every `getattr(stage_config, "..._on_force_restart", ...)`
    call site. The phase output dir IS the unit of restart granularity.
- **Touch size**: L. Hits 4 stage runners + all stage YAMLs + tests. Tests
  asserting partial-restart behavior should be deleted, not migrated. Worth
  scoping a plan; ordering hint: do this AFTER the `debug_mode` YAML cleanup
  (above) so we're not editing the same YAML blocks twice.
- **See also**: today's commit on `_cleanup_spikesort_outputs_for_force_restart`
  expanded the allowlist as a stopgap; that whole function should go away when
  this lands. The stale-bombcell SLAy mtime fallback in
  `status._read_slay_label_column` (`bc_is_fresh`) becomes dead code once
  force-restart actually cleans the stage dir — drop it too.
