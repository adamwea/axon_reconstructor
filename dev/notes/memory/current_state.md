# Current state

Snapshot of what's shipped, in-flight, and queued. Updated as state changes; old facts get deleted, not commented out.

## ⚡ USER INJECTIONS

User-authored directives that override plan / tier order until satisfied. Read FIRST each iteration. Apply at the earliest applicable slice; when reliably internalized, promote the rule to a guardrail / CLAUDE.md slice protocol / plan and delete the entry here.

### Active

- **[2026-05-18] YAML hygiene as you go**: Every slice that touches phase code, CLI flags, config schema, or phase wiring must update `dev/debug_NERSC/debug.runtime.yml` AND `dev/debug_NERSC/debug.data.yml` so they stay an accurate mechanical source of truth for what the pipeline runs. The future default `runtime.yml` will derive from `debug.runtime.yml`, so stale entries propagate forward. Remove dead phase blocks, dead `resource_class` entries, dead CLI flag defaults; add new keys for new phases/flags; clear `# TODO Claude:` annotations once their target is resolved.
  - **Audit pass [done 2026-05-19 after slice 7]**: confirmed all six already-deleted phases (`reports`, `plot_templates` v1, `per_unit_processing`, `prepare_raw_binaries`, `report_preprocessing`, `cleanup_preprocessing_outputs`) are gone from `phases:` / `phase_sequence:` / `resource_class:` in both YAMLs. The remaining `reports:` keys (debug_local:631,678,846; debug_NERSC:884,945,1115) are sub-keys inside `bombcell_label` / `merge_SLAy` / `bombcell_label_pass2` phase blocks, not the deleted top-level `reports` phase. `--force-replot` references stay because slice 11 (force_replot deletion) hasn't shipped — they'll be addressed there.
  - **Promote when stable**: once this is reliably part of every slice for two+ tiers, promote the rule to the CLAUDE.md slice protocol (alongside "run tests") and delete this entry.

## Shipped this week (2026-05-12 → 2026-05-18)

### axon_recon repo
- **Phase roster cleanup slice 1 — legacy reconstruct phases deleted** (commits `34bb353`, `2c9e1d3`, `17ca304` + commit_log companions):
  - `reports` mega-phase deleted (phase impl, config dataclass, CLI dispatch, YAML, tests).
  - `plot_templates` v1 deleted (phase impl, config field, CLI aliases, YAML, tests). `report_templates.consume` normalizer now canonicalizes on `plot_templates_v2` and rejects v1 strings.
  - `per_unit_processing` deleted (phase impl, the four nested sub-config dataclasses, monolithic-pipeline helpers `_report_scope_config` + `_disable_reports_config`, YAML, tests). `_run_reconstruct_templates_pipeline_monolithic` KEPT (still called by `run_reconstruct_templates_pipeline` when `phase_sequence is None`; ~70 template tests rely on it).
  - 0 new test failures across the slice; 15 pre-existing failures noted (see open_questions).
- **Reconstruct `--force-restart` wipes the whole stage output** (no more templates-cache stash-and-restore). Commit `5e2b883`.
- **Concat analyzer disabled at every recon production wrapper.** `legacy_include_concat` default flipped `True → False`; `_iter/_load_templates_phase_analyzers` hard-set `include_concat=False`; materialize call sites pass `include_concat=False`. Loader low-level path still honors `include_concat=True` for tests only. Tracker entry "Remove concat analyzer plumbing from recon stage" tracks full removal.
- **CLI scope flags landed**: `--profile`/`--task-profile`, `--target-wells` accepts ints, `--targets <ds>:<well>` per-pair filter. Process-wide overrides in `pipeline/config.py`. Tests pass.
- **summary.json per-pid tmp rename fix** (`logging/summary.py`). No more `FileNotFoundError` floods from concurrent srun writers.
- **`resolve_inner_worker_count` honors yaml/phase hints when slot is None.** No more silent `n_jobs=1` collapse in MPI workers. Test rewritten to lock in corrected fallback.

### SLAy repo
- **`accept_merge` assertion relaxed** (commit `426ba71`): `>=` instead of `==` for same-time intra-cluster collisions. Big wells (>700 KS units) no longer crash at the auto-merge step.
- **`accept_all_merges` aux-tsv sync**: cluster_KSLabel.tsv / cluster_Amplitude.tsv / cluster_ContamPct.tsv stay coherent with cluster_group.tsv after merges. Resolves the KS-extractor inner-join bug that was dropping new merged unit IDs in downstream consumers.

### Shifter image
- Latest digest in shifterimg: `62c8a06e6b` (built 2026-05-18 03:24:35, with concat-disabled wrappers).
- A later build `32638ea26b` (2026-05-18 04:57:38) includes the SLAy aux-tsv sync. **Both are present in shifterimg**; the 04:57 build is the most current.
- Docker.io path: `adammwea/axon-recon:pipeline-v2`.

### Validation runs
- **Smoke test for 260326/M08073/000208/well000** (DIV 36, 80k DMEM): recon stage with `--force-restart` produced 176 merged templates + 176 per-unit recon outputs, matching the post-SLAy good-label count from status replay. The concat-analyzer rip-out + aux-tsv sync together fixed the systematic post-merge undercount.
- **Job 53089489** (4-node interactive GPU, 16 wells) completed with ds4 spikesort_full results + 4 merge_SLAy retries. ds6/well002 host-OOM'd; others completed.

## In-flight

- **Phase roster cleanup plan** (`plans/active/phase_roster_cleanup_plan.md`): slices 1-9 + 11-13 + 14a-14b shipped (slice 10 plot_raster_threshold quality fix deferred per plan §3). Slice 14c still queued (blocker documented in open_questions: the monolithic per-target stage runner in preprocess/spikesort/reconstruct/analysis needs a refactoring decision before integration).
- **Parallelism post-migration cleanup plan** (`plans/active/parallelism_post_migration_cleanup_plan.md`): slices 3 + 5 + 7 + 8 (sub-items A/B/C/D/E1/E2) shipped this iteration. Also `+` a production-code fix to `core/unit_plots.py` (invert_y_axis propagation) and a softening to `container_cli` on `--dry-run`. Pre-existing failures went from 21 → 5 (remaining 5 are all spikeinterface_extract — parallelism slice 9 scope). Slices 1, 2, 4, 6, 9, 10 still queued.
- **kssynth + unitlink + unitmatch_phase plans** drafted in `plans/active/`. `~/dev/pkgs/kssynth/` now exists locally — slice 1 of its plan landed (scaffold + stub `api.synthesize` + 2 tests + CI matrix; commit `7e236f3` in the kssynth repo). User will push to GitHub when ready. `~/dev/pkgs/unitlink/` still TBD. axon_recon analysis-stage `unitmatch` phase still TBD.

## Locked decisions from 2026-05-18 pre-loop Q&A

Run-semantics:
- **Three invocation modes** for any stage: no-flag (auto-restart-from-first-broken), `--force-restart` (rmtree everything), `--replot` (run plot/report phases only, orthogonal to auto-restart).
- `--force-replot` is DELETED (`--replot` replaces it).
- **Checkpoint status enum**: `{missing, in_progress, ok, error, stale, skipped}`. `stale`/`error`/`in_progress`/`missing` all trigger force-restart of that phase + downstream in auto-restart mode. `skipped` stays skipped only if still configured that way; otherwise re-evaluated like `missing`.
- **`in_progress` marker**: phase writes a stub summary_json (`status: in_progress`, `started_at`, `pid`) BEFORE its main work begins, overwrites with `ok`/`error` on completion. Stranded `in_progress` = crashed process; auto-restart catches it.
- `dry_run` is its own status (`dry_run_ok`) and is treated like `missing` by auto-restart (dry-run doesn't actually run the phase).

Output locations:
- Iteration outputs: `/pscratch/sd/a/adammwea/dev_outputs/<plan_slice_name>/...` — one subdir per plan slice, granular.
- Reference data: `/pscratch/sd/a/adammwea/analyzed_data/...` — read-only. Never mutated during iteration.
- New `--output-root` CLI flag overrides `data_config.output_root` for iteration runs.

Working-data scope (re-confirmed):
- 80k DMEM well000 of M08073 chips, all DIVs, for v1 iteration.
- well001 OK as an additional sample if needed for parallelism / diversity smoke tests.
- Network scans: path pattern `/pscratch/sd/a/adammwea/raw_data/.../<date>/<chip>/Network/`. Two types per DIV (clustered groups of 4/9 channels, and fully sparse). v1 picks ONE type when network scans enter scope; the other is a future lever. Reserved for unitmatch v2 — out of v1 iteration scope.

Test policy:
- Delete tests that no longer make sense after a slice.
- Morph tests that should remain into the current mental model — done inline with the slice that requires it, not as a dedicated cleanup pass.

Process control:
- Model: default Opus 4.7; Sonnet 4.6 only for very concrete mechanical work.
- Mid-slice ambiguity: low-stakes → best-guess + note in `memory/notes.md`; high-stakes → ask user. If user is unresponsive: pause that slice, document blocker in `open_questions.md`, switch to another unblocked slice. Keep the loop moving.
- Login-node smokes: pass `--task-backend local_affinity` every time; cap 64 procs; use `--limit-*` flags to keep scope tiny. Bigger smokes → ask user to run on a real allocation.
- New repos (`kssynth`, `unitlink`) get local `git init` at slice 1; remote pushed by user when they create the GitHub repo.

## Queued / not started

Listed in tier-order from `dev/notes/plans/active/phase_roster_cleanup_plan.md` §"Execution order" (and the related response):

1. `phase_roster_cleanup_plan.md` (Tier 1)
2. `trackers/tech_debt.md` §"Remove `debug_mode` YAML blocks" + §"Remove concat analyzer plumbing from recon stage" (Tier 2, parallel)
3. `ks_synthesizer_package_plan.md` (Tier 3) — create the kssynth sibling repo + ship slices 1-7
4. `unitmatch_runner_package_plan.md` (Tier 3) — create the unitlink sibling repo + ship slices 1-7 (parallel with kssynth once APIs settle)
5. kssynth slice 9 (axon_recon recon-stage integration) + `unitmatch_phase_plan.md` slices 1-5
6. `trackers/tech_debt.md` §"Collapse `--force-restart` semantics" (Tier 4)
7. `trackers/tech_debt.md` §"Minimize / eliminate `resources.profiles`" (Tier 4)

Tier 5 (chip away anytime): parallelism_post_migration_cleanup_plan, spikesort/runner.py 10K-line decomposition, container preflight softening, container/MPI alignment notes.

## Environment state

- Working on Perlmutter (NERSC). Logged in as `adammwea`.
- Shifter container: `adammwea/axon-recon:pipeline-v2`.
- Code lives at `/global/homes/a/adammwea/dev/pkgs/axon_recon/` (also accessible via `/global/u2/...` symlink — they're the same dir).
- Sibling packages at `/global/homes/a/adammwea/dev/pkgs/SLAy/`, `/global/homes/a/adammwea/dev/pkgs/UnitMatch/`.
- Pscratch for I/O: `/pscratch/sd/a/adammwea/{raw_data,analyzed_data,run_logs,smoke_logs}/`.

## Working-data scope (current iteration cycle)

**Locked to ONE cohort**: 80k DMEM well000 (M08073 chip family, well000 of each DIV's AxonTracking recording). Use this and only this for iteration development; ask the user before expanding scope.

- **Reference (read-only)**: `/pscratch/sd/a/adammwea/analyzed_data/Media_Density_T5_02182026_AR_axon_analysis_AW/Media_Density_T5_02182026_AR/<date>/M08073/AxonTracking/<rec>/well000/`. Existing per-stage outputs are the ground-truth reference for desired behavior. **Do not mutate these.** Compare against them.
- **Iteration output dir**: `/pscratch/sd/a/adammwea/dev_outputs/<feature>/...` — one subdir per feature or plan-slice. New run outputs go here; pscratch retention is fine since these are disposable.
- **Raw network scans**: available under the raw_data tree. Single-segment, non-concatenable. Need preprocess stage before any downstream work. Reserved for `unitmatch_phase_plan.md` slice 6 — DO NOT touch them during v1 iteration.
- **Smoke-test discipline**: targeted login-node only (`--task-backend local_affinity`, cap 64 procs, use `--limit-*` flags). Bigger than that → ask user to run on a real allocation.

## Known good baseline

For regression-check purposes, the following counts on `260326/M08073/000208/well000` (DIV 36, 80k DMEM) post-2026-05-18 fixes:
- post-SLAy unit IDs in `spike_clusters.npy`: 377
- post-SLAy cluster_KSLabel.tsv (after aux-tsv sync): 287 good + 312 mua = 599 rows
- recon merged templates: 176
- recon per-unit outputs: 176 (matches good count of post-merge units in spike_clusters; mua not reconstructed by default per `unit_label_filter`)
