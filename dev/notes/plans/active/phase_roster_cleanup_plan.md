# Phase roster cleanup — preprocess / spikesort / reconstruct

Status: implementation plan. Realizes the `tech_debt.md` entry **"Finalize the phase roster before any destructive cleanups land"** (section §163 of that file). Sibling to the active parallelism / kssynth / unitlink / unitmatch-phase plans. Same operating contract: one slice at a time, `claude:` commit prefix, append a line to `debug/commit_log.md` after every commit.

The user marked specific phases in `dev/debug_NERSC/debug.runtime.yml` with `# TODO Claude: …` annotations describing what to do with each. This plan consolidates those, identifies dependencies, and orders the work into commits.

**See also:**
- `tech_debt.md` §"Finalize the phase roster before any destructive cleanups land" — the parent tech_debt item this plan realizes.
- `tech_debt.md` §"Collapse `--force-restart` semantics" — explicitly BLOCKED on this plan (per its own ordering note).
- `tech_debt.md` §"Remove `debug_mode` YAML blocks across all stages" — touches the same YAML files; should overlap with this plan's commits to avoid editing the same blocks twice.
- `tech_debt.md` §"Remove concat analyzer plumbing from recon stage" — independent cleanup; can land in parallel.
- `ks_synthesizer_package_plan.md` — enables the bombcell / merge_SLAy migration to recon stage (one of the items below).
- `roadmap.md` §"Post-templates bombcell + SLAy pass" — the long-term home for bombcell / SLAy is in recon stage on full templates; this plan begins that migration by disabling the spikesort-stage copies.

---

## 0. End-state, in one paragraph

After this plan lands, the preprocess / spikesort / reconstruct phase rosters reflect the user's intended shape: short, current, no legacy phases, no half-decided phases. Stage moves are completed (the `init` and `cleanup` stages exist; concat-binary logic lives in spikesort; bombcell + SLAy are gone from spikesort's default sequence pending the recon migration that kssynth enables). Legacy plot / report / per-unit phases that were superseded by `_v2` variants are physically deleted from code, YAML, and tests. One phase rename for naming hygiene (`generate_gtrs` → `axon_velocity_gtrs`). The roster is now small enough that the next two destructive cleanups (`--force-restart` collapse, `resources.profiles` elimination) can land with confidence they're not migrating phases that are about to be deleted anyway.

---

## 1. Inventory — what the user annotated, what to do with each

Read from `dev/debug_NERSC/debug.runtime.yml` `# TODO Claude:` annotations as of 2026-05-18. Quoting the user verbatim is preserved where useful.

### 1.A — preprocess stage (line 365–576)

| Phase | Line | Action | Notes |
|---|---|---|---|
| `copy_src_to_scratch` | 420 | **Move to a new `init` stage** | "No need to force restart this when testing preprocess iterations, and it adds unnecessary overhead to the critical path of preprocess when enabled." |
| `prepare_raw_binaries` | 440 | **Delete entirely** | "No backwards compat. No fallbacks. No legacy code. Just delete it." Currently `enabled: false` already. |
| `concat_segments` | 510 | **Move to spikesort stage; rename to `concat_binary`** | "bootstrap_concat_binary phase in spikesort already does this." Consolidate; this preprocess copy becomes dead. |
| `plot_concat_traces` | 526 | **Move to spikesort stage** | "Keep trace plotting and raster plotting disabled by default, but these are useful diagnostics to have in spikesort stage." |
| `plot_concat_channel_layout` | 539 | **Move to spikesort stage** | Same as above. |
| `plot_raster_threshold` | 548 | **Keep in preprocess; fix the plot** | "Ideally we see all the channels across all the segments, and in each plotted segment division on the raster, we should see channels turn on and off as they drop in and out of the recording." Not blocking; quality fix, schedule after the structural cleanup. |
| `report_preprocessing` | 556 | **Delete** | "I dont think this actually does anyting right now." |
| `cleanup_preprocessing_outputs` | 564 | **Delete** | "I also dont really know what this does." |
| `wipe_src_scratch` | 570 | **Move to a new `cleanup` stage** | "We will eventually optionally run all clean up phases from this stage." |

After all moves + deletions, preprocess shrinks to: `save_rec_metadata`, `preprocess_segments`, `plot_segment_traces`, `plot_segment_channel_layouts`, `plot_raster_threshold` (kept, fix pending).

### 1.B — spikesort stage (line 578+)

| Phase | Line | Action | Notes |
|---|---|---|---|
| `bootstrap_concat_binary` | 627 | **Rename to `concat_binary`; consolidate with the moved-in preprocess concat logic** | "Keep the behavior of this stage, just rename it and consolidate it with the concat_segments phase in processing stage." |
| `bombcell_label` | 797 | **Disable + remove from default sequence** | "we're migrating this basic idea to recon stage to be done with merged templates if possible … re-enable and iterate in the future as needed." Already `enabled: false` in the current YAML. Long-term home is in recon stage post-kssynth. |
| `merge_SLAy` | 928 | **Disable + remove from default sequence** | Same migration path as bombcell_label. The spikesort-stage version stays in code (its phase implementation isn't deleted) but exits the default phase_sequence. |

After this: spikesort phase_sequence is `concat_binary` (renamed), `sort`, `snapshot_sorter_output`, `concat_analyzer` (still here pending the broader concat-analyzer removal in §T9 of tech_debt), `cleanup_concat_binary`, `cleanup_analyzers`. bombcell and SLAy are wired in code but not in the default sequence.

### 1.C — reconstruct stage (line 1156+)

| Phase | Line | Action | Notes |
|---|---|---|---|
| `plot_templates` | 1459 | **Delete legacy phase** | "We're adopting plot_templates_v2 now." Phase implementation + YAML block + tests + any caller references. |
| `per_unit_processing` | 1704 | **Delete legacy phase** | "This is a legacy phase, please delete." |
| `reports` | 1749 | **Delete legacy phase** | "This is a legacy phase, please delete." |
| `generate_gtrs` | 1892 | **Rename to `axon_velocity_gtrs`** | "in the future we aim to introduce more than one reconstruction method, so it will be helpful to have the phase name reflect the specific method used for GTR generation." Touches references in YAML, the runner, tests, and any downstream phase that references the gtrs output paths. |

### 1.D — new stages introduced by this plan

| New stage | Phases | Purpose |
|---|---|---|
| `init` | `copy_src_to_scratch` (moved from preprocess) | Once-per-data-config setup work that doesn't need to be re-run when iterating on preprocess. Resource class: `h5_to_binary` (existing). |
| `cleanup` | `wipe_src_scratch` (moved from preprocess), eventually other per-stage cleanup phases | Optional final-stage scratch wipe / dataset disk reclaim. "We will eventually optionally run all clean up phases from this stage." Resource class: `disk_cleanup` (existing). |

---

## 2. Dependencies and ordering between items

Not all items are independent. Some constraints:

1. **The concat_binary consolidation is a single logical change.** Move preprocess `concat_segments` to spikesort, rename spikesort's `bootstrap_concat_binary` to `concat_binary`, fold them together. This is one commit.

2. **The new `init` stage has to exist before its first phase moves in.** Two commits: (a) scaffold the `init` stage with no phases; (b) move `copy_src_to_scratch` in. Same shape for `cleanup` stage.

3. **The plot_* moves to spikesort can wait** until after the concat_binary consolidation lands — they consume what concat_binary produces.

4. **The legacy reconstruct deletions (plot_templates, per_unit_processing, reports) are independent** of everything else. Can land in parallel commits.

5. **The bombcell/SLAy spikesort-side disable** is a YAML-only change (remove from `phase_sequence`, set `enabled: false`). Does NOT delete the phase implementations. The recon-side migration is its own followup work (gated on kssynth's slice 9 landing, per the `ks_synthesizer_package_plan.md`).

6. **The generate_gtrs rename** is a mechanical search-and-replace across YAML, runner, tests, and any phase whose outputs reference the gtrs path. One commit.

7. **The plot_raster_threshold quality fix** isn't structural — it's a "do a better job on the existing plot". Defer to its own commit; not blocking the destructive cleanups that depend on the roster being settled.

---

## 3. Implementation slices

One commit per slice unless noted. `claude:` prefix.

### Slice 1 — pure deletions (legacy reconstruct phases)
- Delete `plot_templates`, `per_unit_processing`, `reports` phases from `stages/reconstruct/templates/runner.py`, the YAML, tests, and any downstream callers. Use `grep -rn "plot_templates\b\|per_unit_processing\b\|reports[. ]" src/ dev/` to find references.
- Each deleted phase is its own commit (3 commits in this slice) — bisectability.
- No new behavior; mechanical removal.

### Slice 2 — preprocess pure deletions
- Delete `prepare_raw_binaries`, `report_preprocessing`, `cleanup_preprocessing_outputs` phases entirely. (1 commit each.)
- These are already `enabled: false`. Removing the phase entirely is the destructive step.

### Slice 3 — `generate_gtrs` → `axon_velocity_gtrs` rename
- Single mechanical commit. Touches:
  - YAML phase block + phase_sequence references
  - The phase's runner function name
  - Output relpaths that reference "gtrs" (search & decide whether they get renamed too — the YAML output knobs probably keep the `gtrs` substring since they refer to a generic concept; the phase NAME is what changes)
  - Tests
- Verify with a grep audit.

### Slice 4 — scaffold the `init` stage
- New stage `init` registered in `pipeline/stages/`. Empty phase_sequence + empty phases dict initially.
- `stages.init` block added to YAML with `enabled: false` to start; flip to `true` once a phase moves in.
- Resource class entries: none new (init phases reuse existing classes).
- Tests: `tests/test_init_stage_disabled_is_noop.py`.

### Slice 5 — move `copy_src_to_scratch` from preprocess to init
- Move the phase implementation, its config dataclass, its YAML block.
- Update the data config / runtime YAML where the phase is referenced.
- Tests: pre-existing `copy_src_to_scratch` tests move with it; add one new test asserting the phase runs in the init stage's phase_sequence.

### Slice 6 — scaffold the `cleanup` stage + move `wipe_src_scratch`
- Same shape as slices 4 + 5 collapsed (cleanup stage doesn't have a separate "fill it later" use case — `wipe_src_scratch` is its first and currently only phase).
- Document in the stage docstring that more cleanup phases (currently per-stage cleanup_* phases scattered across stages) will eventually consolidate here. Tracked in `tech_debt.md` §"Finalize the phase roster" as part of the longer-term work.

### Slice 7 — concat_binary consolidation
- Move preprocess `concat_segments` phase implementation into spikesort.
- Rename spikesort `bootstrap_concat_binary` → `concat_binary`.
- Fold the two implementations together (they share most logic; the user's annotation says "this phase has been working well in the spikesort stage, it basically leverages preprocessing stage concat segments logic").
- Update YAML phase_sequence + phase block references.
- Update all callers (anything reading `bootstrap_concat_binary_*_relpath` keys from stage config).
- Tests: `concat_segments` tests migrate to spikesort's test dir + get merged with `bootstrap_concat_binary` tests.

### Slice 8 — move plot_concat_traces + plot_concat_channel_layout to spikesort
- Move the phase implementations + YAML blocks.
- Both stay `enabled: false` by default — they're diagnostic phases. No behavior change for default runs.
- Tests: phase wiring tests move with them.

### Slice 9 — disable bombcell_label + merge_SLAy in spikesort phase_sequence
- YAML-only change: remove both from `stages.spikesort.phase_sequence`. Both already have `enabled: false`; the sequence removal makes that the canonical state.
- The phase implementations stay in code (`stages/spikesort/runner.py`) — re-enabling them is a YAML flip if a user wants to. Code deletion is gated on the recon-side migration landing (see `ks_synthesizer_package_plan.md` slice 9 followup, eventually a separate plan).
- Update tests that exercise the default phase_sequence's contents.

### Slice 10 — `plot_raster_threshold` quality fix
- Not structural; can land anytime after the structural cleanups settle. Defer to its own future commit; this plan calls it out so it doesn't get lost but doesn't sequence it inline.
- Goal: "ideally we see all the channels across all the segments, and in each plotted segment division on the raster, we should see channels turn on and off as they drop in and out of the recording."

### Slice 11 — `--force-replot` deletion + `--replot` rename
- Mechanical: replace `--force-replot` with `--replot` everywhere. Delete `_FORCE_REPLOT_OVERRIDE` / `set_force_replot_override` if they exist; introduce `_REPLOT_OVERRIDE` if needed (only if a process-wide override is currently used).
- Grep audit: `grep -rn "force.replot\|force_replot" src/ dev/` — zero hits in non-archive code after the slice.
- Tests: update any test asserting `--force-replot` behavior to assert `--replot` instead. Delete tests that test the OLD semantic of "reuse computed outputs but rebuild plots only when stale" — the new `--replot` is "always rebuild plot phases, orthogonal to staleness". Some test rewriting is needed; lean on `guardrails/force_restart.md` for the post-change contract.

### Slice 12 — `--output-root` CLI flag
- New CLI flag at the shared argparse layer (`pipeline/cli.py`) that overrides `data_config.output_root`.
- Plumbing: when set, the bundle's `data_config["output_root"]` is replaced before `select_execution_targets` reads it.
- Use case: Claude (or any user) writes iteration outputs to `/pscratch/sd/a/adammwea/dev_outputs/<plan_slice>/...` without editing the YAML or mutating the reference `analyzed_data/` tree.
- Tests: assert override is applied; assert no leak between invocations (clear in `finally`).

### Slice 13 — Checkpoint status enum + `in_progress` marker mechanic
- Each phase's target runner writes a stub `summary_json` with `status: in_progress`, `started_at: <iso8601>`, `pid: <int>` BEFORE its main work begins. The marker overwrites whatever was at the same path (a previous run's `ok` / `error`).
- On successful completion, the runner overwrites with the final `status: ok` summary.
- On exception, the runner overwrites with `status: error` + a `traceback` field (and re-raises).
- If the process dies (kernel-killed, OOM, kill -9), the marker stays as `in_progress` — which the next invocation's auto-restart logic catches.
- New helper module `src/axon_recon/pipeline/checkpoint.py`:
  - `write_in_progress_marker(summary_json_path, phase_name, …) -> None`
  - `read_checkpoint_status(summary_json_path) -> Literal["missing","in_progress","ok","error","stale","skipped"]`
  - `is_stale(summary_json_path, *input_mtimes) -> bool`
- Tests: per-status unit tests; one integration test asserting the marker is written before work begins and overwritten correctly.

### Slice 14 — Auto-restart-from-first-broken logic
- New helper in `pipeline/runner.py` (or `checkpoint.py`): `find_first_broken_phase(phase_sequence, target_paths, yaml_skip_state) -> (phase_index, status)` returning the index in `phase_sequence` of the first phase that is NOT `ok` and NOT (legitimately) `skipped`.
- Stage runner updated: when NEITHER `--force-restart` NOR `--replot` is set, call `find_first_broken_phase`; if it returns a non-None phase index, treat that phase (and everything downstream) as if `--force-restart` was set for them. If it returns None (everything is `ok` or `skipped`), the stage is a no-op for that target.
- `--force-restart` bypass: skip the find-first-broken check entirely; rmtree + run all phases as before.
- `--replot` bypass: skip the find-first-broken check; run plot/report phases only, regardless of statuses.
- Tests: 3-phase fixture with phase 2 in various states (`ok`, `in_progress`, `error`, `stale`, `skipped`); assert correct restart point in each.

Slices 13 and 14 together realize the auto-restart contract documented in `guardrails/stage_phase_architecture.md` §"Checkpoint status enum + auto-restart-from-first-broken". They're substantial enough to potentially split into their own dedicated plan once they start landing; tracked here for now since they touch the same stage runner code as the phase roster cleanup.

Total estimated touch: ~600-900 LoC across 9-10 commits. Mostly deletion and mechanical move; ~150-200 net new lines (the new stage scaffolds + the consolidated `concat_binary` after dedup).

---

## 4. What this UNBLOCKS

After this plan lands:

- **`tech_debt.md` §"Collapse `--force-restart` semantics"** can proceed safely. Per its ordering note: "BLOCKED on 'Finalize the phase roster' — every phase we kill in that step is one fewer phase whose cleanup helper we have to migrate". The roster is now small, settled, and free of legacy phases that would have cost migration effort.

- **`tech_debt.md` §"Remove `debug_mode` YAML blocks"** can land in parallel or right after — same YAML files, fewer phase blocks to touch.

- **The recon-stage bombcell / SLAy migration** (their long-term home) can now move forward without legacy duplicates standing in the way. Gated on `kssynth` shipping (slice 9 in `ks_synthesizer_package_plan.md`).

- **Future destructive YAML cleanups** (eventually eliminating `resources.profiles`, consolidating per-stage cleanup phases into the new `cleanup` stage) are unblocked by the same logic.

---

## 5. Tests + verification

- Per-slice: existing tests for the affected phase migrate with the phase. New tests for new wiring (`init` / `cleanup` stages, the consolidated `concat_binary` phase).
- After slice 9 lands: a smoke run of `axon-recon stages preprocess` + `axon-recon stages spikesort` + `axon-recon stages reconstruct` on one well, end-to-end, asserting the new phase_sequence + the renamed phases produce the expected outputs.
- Grep-audit after each commit for stale references to the deleted / renamed phases. The audit is mechanical: `grep -rn "plot_templates\b\|generate_gtrs\b\|prepare_raw_binaries\b" src/ dev/`. Any hits in non-test code = followup commit.

---

## 6. Open questions

1. **`init` stage scope.** Is `copy_src_to_scratch` the only phase that belongs there, or should other "once per data config" setup phases (e.g. anything from the eventual data-fetch / globus workflow) live there too? Document the design intent in the stage's docstring; defer adding more phases until they're actually needed.

2. **`cleanup` stage scope.** Same question. The user's annotation says "We will eventually optionally run all clean up phases from this stage". For v1: just `wipe_src_scratch`. Future: consolidate `cleanup_concat_binary`, `cleanup_analyzers`, `clear_templates_cache` (whichever survive the broader cleanup) into this stage.

3. **`concat_binary` resource class.** Currently spikesort's `bootstrap_concat_binary` resource class is its own entry; preprocess's `concat_segments` reuses a separate class. After consolidation, one resource class entry; merge the budgets sensibly (probably keep the spikesort version since that's the one actually being used).

4. **Cross-references to `bootstrap_concat_binary` in code.** A grep audit during slice 7 will surface any external code (sbatch scripts, dashboards, scripts under `dev/`) that hardcodes the old phase name. Each gets updated in the same commit.

5. **bombcell_label / merge_SLAy code deletion timing.** After this plan disables them in spikesort phase_sequence, the phase implementations stay in `stages/spikesort/runner.py` as dead-but-wired-up code. Full code deletion is gated on the recon-stage replacement landing. Tracked as a follow-up in a future plan once `kssynth` + recon-stage migration ship.

6. **`plot_raster_threshold` quality fix scope.** The "channels turn on and off as they drop in and out of the recording" requirement implies the plot iterates per-segment channel sets and visualizes the difference. Reasonable change; needs design-doc-level thinking about colormap / legend choices. Defer to its own commit + light design note.
