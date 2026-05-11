# Spikesort merge cleanup plan — finish the SLAy/bombcell repair

Follow-up to `debug/spikesort_label_merge_repair_plan.md` (the 7-slice repair that landed on
`claude-migration` as commits `72fe271`..`6c78e7f`). That repair stopped intentionally with
`SPIKESORT REPAIR PARTIAL` because the remaining work was too entangled to land in one
plan-slice without a merge-orchestrator rewrite. This plan does that rewrite.

The user has decided we do not need `merge_si_auto` or `merge_unitmatch` going forward.
Dropping them collapses most of the interlock and lets the rest of the cleanup fall out
cleanly.

This plan assumes a fresh branch off `claude-migration` (suggestion: `spikesort-merge-cleanup`).
Test baseline at slice 0: 200 spikesort tests passed, 457 pipeline tests passed
(test_progress.py excluded).

---

## 0. Goal And End State

After this plan lands:

- The pipeline keeps **two** post-sort label/merge phases: `bombcell_label` and `merge_SLAy`.
- `merge_si_auto` and `merge_unitmatch` are gone from code, CLI, aliases, YAML, tests, and the
  default phase sequence.
- `_run_merge_methods_for_target` is replaced by a SLAy-only orchestrator. The
  pre-merge / post-merge / replot scaffolding that survives is keyed to SLAy's actual needs,
  not to a generic method-dispatch abstraction.
- The 4 legacy cache helpers (`_cache_sorting_outputs_before_merge`,
  `_cache_canonical_sorter_output_for_merge`, `_prepare_replot_workspace_analyzer`,
  `_restore_sorting_outputs_from_pre_merge_cache`) are deleted. Mutation safety is
  delivered by slice 1's `snapshot_sorter_output` phase + slice 4's per-run SLAy scratch
  copy.
- The legacy analyzer recompute path
  (`_load_or_recompute_spikesort_analyzer` → `_recompute_spikesort_analyzer` →
  `_recompute_sorting_analyzer_to_dir`) is deleted. All analyzer loads go through
  `_load_concat_analyzer_for_phase` (the slice-2 helper).
- `debug/debug.runtime.yml` has no `working_cache:` blocks, no `merge_si_auto:` block, no
  `merge_unitmatch:` block, and no `cache_sorting_outputs_before_merge_*` /
  `pre_merge_workspace_*` flat config fields.
- §6 cleanup-grep checks from the previous plan all return 0 hits.
- §3 smoke matrix runs end-to-end with `dry_run=true` and `dry_run=false` round trips
  (gated on a real sort fixture being available on the target machine).

The non-goal is restructuring `_run_merge_methods_for_target` further than necessary to
remove the dispatch and cache scaffolding; we leave the per-well merge metadata, replot,
unit-diff, and report-generation code alone except where SLAy-only inlining is trivial.

---

## 1. Why removing `merge_si_auto` + `merge_unitmatch` unlocks the rest

The previous plan's slice-6 audit identified five interlocked artifacts that could not
be deleted in isolation:

1. `_run_auto_merge_method` (`runner.py:8262`) is the only caller of
   `_load_or_recompute_spikesort_analyzer`. As long as it exists, the legacy analyzer
   recompute family cannot be deleted. Once `merge_si_auto` and `merge_unitmatch` are gone,
   `_run_auto_merge_method` has no caller and can be deleted, which strands the recompute
   family — so it can be deleted too.

2. `_run_merge_methods_for_target` (`runner.py:9000ish`–`9900ish`) is a method-dispatch
   loop that handles all three methods. Its cache scaffolding (`working_cache`,
   `pre_merge_cache`, `merge_workspace`) exists because each method may want to roll the
   sorter_output back on failure. Slice 1 superseded this with `snapshot_sorter_output` +
   `restore_sorter_output`, and slice 4 made SLAy mutation-safe via per-run scratch copy.
   With only SLAy remaining, the cache helpers have no remaining job.

3. The 4 cache helpers (`_cache_sorting_outputs_before_merge`,
   `_cache_canonical_sorter_output_for_merge`, `_prepare_replot_workspace_analyzer`,
   `_restore_sorting_outputs_from_pre_merge_cache`) have 5 call sites all inside
   `_run_merge_methods_for_target`. They go away when the orchestrator is rewritten
   SLAy-only.

4. The `working_cache:` YAML blocks under `phases.merge_SLAy`, `phases.merge_si_auto`,
   `phases.merge_unitmatch` in `debug/debug.runtime.yml` exist because the orchestrator
   parses them. With the orchestrator rewritten and the merge_si_auto / merge_unitmatch
   phases deleted, all three blocks become dead YAML.

5. The `cache_sorting_outputs_before_merge_*` and `pre_merge_workspace_*` flat config
   fields on `SpikesortStageConfig` are parsed from the YAML blocks above. They go away
   when the YAML blocks go away.

The interlock is therefore: (merge_si_auto + merge_unitmatch removal) → unlocks
(auto-merge method removal) → unlocks (analyzer recompute family removal) → unlocks
(cache helper removal) → unlocks (YAML and flat config removal). The plan executes
in that order.

---

## 2. Discovery Targets (Read These First)

Before starting any slice, an iteration should grep/read these to confirm the current state
matches what the plan describes. The `_partial_` repair shifted lines, so use grep, not
line numbers.

### 2.1 Source modules touching merge_si_auto / merge_unitmatch

- `src/axon_recon/pipeline/stages/spikesort/orchestrators/merge_si_auto.py` — delete entirely.
- `src/axon_recon/pipeline/stages/spikesort/orchestrators/merge_unitmatch.py` — delete entirely.
- `src/axon_recon/pipeline/stages/spikesort/orchestrators/merge_units.py` — inspect; the
  top-level `run_spikesort_merge_units` may dispatch to the three methods.
- `src/axon_recon/pipeline/stages/spikesort/orchestrators/__init__.py` — drop the two
  imports/exports.
- `src/axon_recon/pipeline/stages/spikesort/__init__.py` — drop the two re-exports.
- `src/axon_recon/pipeline/stages/spikesort/cli.py` — drop `_run_merge_si_auto_from_args`,
  `_run_merge_unitmatch_from_args`.
- `src/axon_recon/pipeline/cli.py` — drop the two imports (~lines 109, 113), drop the two
  short aliases (~lines 179, 180), drop the four full aliases (~lines 190-197), drop the
  two dispatch entries (~lines 293, 294).
- `src/axon_recon/pipeline/stages/spikesort/config.py`:
  - `DEFAULT_SPIKESORT_PHASE_SEQUENCE` — drop the two entries.
  - `_SPIKESORT_PHASE_ALIASES` — drop entries for `merge_si_auto`, `merge_auto`,
    `merge_auto_merge`, `auto_merge`, `si_auto`, `merge_unitmatch`, `unitmatch`.
  - Flat fields on `SpikesortStageConfig`: `merge_si_auto_*` (~25 fields),
    `merge_unitmatch_*` (~25 fields), `merge_si_auto_dry_run`, `merge_unitmatch_dry_run`,
    `merge_si_auto_resource_class`, `merge_unitmatch_resource_class`.

### 2.2 Source modules touching the merge orchestrator + cache helpers

- `src/axon_recon/pipeline/stages/spikesort/runner.py`:
  - Cache helpers: `_cache_sorting_outputs_before_merge`,
    `_cache_canonical_sorter_output_for_merge`, `_prepare_replot_workspace_analyzer`,
    `_restore_sorting_outputs_from_pre_merge_cache`.
  - Method bodies: `_run_slay_merge_method`, `_run_auto_merge_method`,
    `_run_slay_analyzer_recompute`.
  - Orchestrator: `_run_merge_methods_for_target` (long; ~600 lines).
  - Legacy analyzer family: `_load_or_recompute_spikesort_analyzer`,
    `_recompute_spikesort_analyzer`, `_recompute_sorting_analyzer_to_dir`.
  - Concat-analyzer accessor: `_load_concat_analyzer_for_phase` (keep).

The runner is ~10K lines; grep, do not read whole.

### 2.3 YAML

- `debug/debug.runtime.yml`:
  - `phases.merge_si_auto:` whole block — delete.
  - `phases.merge_unitmatch:` whole block — delete.
  - `phases.merge_SLAy.working_cache:` sub-block — delete.
  - `phases.merge_SLAy.merge_units:` sub-block — inspect; may carry parameters that need
    relocating onto `phases.merge_SLAy` directly or are dead.
  - `stages.spikesort.phase_sequence:` (if set explicitly) — remove the two phase names.

### 2.4 Tests

- `src/axon_recon/pipeline/stages/spikesort/tests/test_runner.py` — find all
  `merge_si_auto` / `merge_unitmatch` / `auto_merge` / `unitmatch` tests; delete or
  update. The slice-5 `test_run_auto_merge_method_dry_run_skips_canonical_analyzer_writeback`
  goes away because the method itself goes away.
- `src/axon_recon/pipeline/stages/spikesort/tests/test_spikesort_config.py` — drop merge_si_auto
  and merge_unitmatch flat-field defaults / parsing tests.
- `src/axon_recon/pipeline/stages/spikesort/tests/_mutation_safety.py` — verify
  references to the two phases are not present; if present, drop.
- `src/axon_recon/pipeline/tests/test_cli_stage_sequence.py` — drop CLI alias coverage
  for the two phases.
- `src/axon_recon/pipeline/tests/test_resources.py` — drop resource-class coverage for
  the two phases.
- `src/axon_recon/pipeline/tests/test_spikesort_target_status.py` — drop direct-phase
  coverage for the two phases.

### 2.5 Documentation / notes

- `debug/spikesort_label_merge_repair_plan.md` — leave intact (historical record).
- `debug/agent_guardrails_commit_notes.md` — append a new top-of-section entry after each
  slice (mandatory; see operating contract below).

---

## 3. Smoke Test Matrix

Smokes are gated on a real post-`spikesort.sort` `<well>/spikesort_outputs/sorter_output`
directory and a built `concat_analyzer/` on the target server. If unavailable when a slice
runs, mark `BLOCKED-SMOKE` and lean on unit tests.

S0. **Baseline** — full spikesort suite green on the new branch before slice 1.
    Command: `conda run -n axon_recon python -m pytest src/axon_recon/pipeline/stages/spikesort/tests/ -q`.

S1. **Snapshot → SLAy dry_run → restore round-trip** — runs
    `spikesort.snapshot_sorter_output` then `spikesort.merge_SLAy` with `dry_run=true`,
    confirms sha256(sorter_output) is byte-identical to the snapshot, then
    `spikesort.restore_sorter_output` round-trips to the snapshot.
    Command: see CLI examples in the previous plan §3 (`spikesort_label_merge_repair_plan.md`).

S2. **SLAy apply** — `dry_run=false` followed by a sha256 check that confirms
    sorter_output *did* change as expected, plus that `merge_metadata_summary.json` was
    written.

S3. **bombcell apply after concat_analyzer** — `spikesort.concat_analyzer` then
    `spikesort.bombcell_label` with `dry_run=false`; confirms labels land.

S4. **CLI sanity** — `axon-recon stages spikesort.merge_si_auto --config debug/debug.runtime.yml`
    fails with an unknown-phase error. Same for `merge_unitmatch`. Same for the aliases.

S5. **YAML sanity** — `debug/debug.runtime.yml` parses cleanly. No keys related to
    `merge_si_auto`, `merge_unitmatch`, or `working_cache` remain.

S6. **Full sort → label → SLAy end-to-end** — `axon-recon stages spikesort` for a
    `--limit-datasets 1 --limit-wells-per-dataset 1 --limit-segments 2 --force-restart`
    scope runs to completion, all enabled phases succeed, no warnings about missing
    `working_cache` / `pre_merge_cache` / `merge_workspace` knobs.

If any of S1–S6 are blocked, the new unit tests in §5 carry the contract.

---

## 4. Migration Slices

Each slice is exactly **one** `claude:` commit. Commit message format:
`claude: spikesort-merge-cleanup, <slice description> (slice N)`.

Tests gate commits: spikesort suite must be green (or strict-subset failures vs. baseline)
before commit. Append commit notes after every commit.

---

### Slice 1 — Delete `merge_si_auto` and `merge_unitmatch` phases

**Files:**
- Delete `orchestrators/merge_si_auto.py`, `orchestrators/merge_unitmatch.py`.
- Edit `orchestrators/__init__.py`: drop the two imports + exports.
- Edit `stages/spikesort/__init__.py`: drop the two re-exports.
- Edit `stages/spikesort/cli.py`: drop `_run_merge_si_auto_from_args`,
  `_run_merge_unitmatch_from_args`.
- Edit `pipeline/cli.py`: drop the two import lines, the two short aliases, the four
  full aliases (`spikesort.merge.automerge`, `spikesort.merge.auto_merge`,
  `spikesort.merge_si_auto`, `spikesort.merge.unitmatch`, `spikesort.merge_unitmatch`,
  `spikesort.merge_units.auto_merge`, `spikesort.merge_units.unitmatch`), the two
  dispatch entries.
- Edit `stages/spikesort/config.py`:
  - Drop `"merge_si_auto"` and `"merge_unitmatch"` from `DEFAULT_SPIKESORT_PHASE_SEQUENCE`.
  - Drop the merge_si_auto/merge_unitmatch entries from `_SPIKESORT_PHASE_ALIASES`.
  - Drop all `merge_si_auto_*` and `merge_unitmatch_*` flat fields from
    `SpikesortStageConfig` (≈50 fields total).
  - Drop the corresponding parse logic in the YAML→config conversion.
- Edit `debug/debug.runtime.yml`:
  - Delete the `phases.merge_si_auto:` block (~lines 820–959 today).
  - Delete the `phases.merge_unitmatch:` block (~lines 971–end-of-stage).
  - Update the `phase_sequence:` list (if explicit) to drop the two entries.

**Tests:**
- Delete tests in `test_runner.py` and `test_spikesort_config.py` that exercise
  `merge_si_auto` / `merge_unitmatch`. Examples: `test_run_auto_merge_method_*`,
  `test_run_unitmatch_*`, any config-default test naming the dropped fields.
- Delete CLI alias coverage in `test_cli_stage_sequence.py`.
- Delete resource-class coverage for the two phases in `test_resources.py`.
- Delete direct-phase coverage in `test_spikesort_target_status.py`.

**Acceptance:**
- `git grep -nE "\bmerge_si_auto\b|\bmerge_unitmatch\b" src/axon_recon/` → 0 hits.
- `git grep -nE "\bmerge_si_auto\b|\bmerge_unitmatch\b" debug/*.yml` → 0 hits
  (except inside the historical `spikesort_label_merge_repair_plan.md`).
- `conda run -n axon_recon python -m pytest src/axon_recon/pipeline/stages/spikesort/tests/ -q`
  green (or strict-subset vs. baseline).
- `axon-recon stages spikesort.merge_si_auto --config debug/debug.runtime.yml` fails with
  unknown-phase error (S4).
- `debug/debug.runtime.yml` parses cleanly (S5).

**Commit:** `claude: spikesort-merge-cleanup, drop merge_si_auto and merge_unitmatch phases (slice 1)`

---

### Slice 2 — Collapse `_run_merge_methods_for_target` to SLAy-only

**Goal:** delete the method-dispatch abstraction inside the merge orchestrator. After this
slice, the merge-target code path is linear: prepare → run SLAy → write merge metadata →
generate reports → done.

**Files:**
- `src/axon_recon/pipeline/stages/spikesort/runner.py`:
  - Delete `_run_auto_merge_method`, `_run_slay_analyzer_recompute` if it is now dead
    (check; it is called by `_run_merge_methods_for_target` only).
  - Rename `_run_merge_methods_for_target` → `_run_merge_slay_for_target` (or inline
    into its single caller, whichever yields a cleaner diff). Delete the methods loop;
    keep the SLAy branch only.
  - Drop the per-method `method_name` plumbing, `methods_enabled` config reads, and
    `existing_summary["methods"]` list handling. Replace with a single SLAy report.
  - Drop the imports and references for the deleted methods.

**Tests:**
- Update `test_runner.py` tests that exercise `_run_merge_methods_for_target` to
  the new shape. Tests that asserted on the methods loop become single-call assertions.

**Acceptance:**
- `git grep -n "_run_auto_merge_method\|_run_slay_analyzer_recompute" src/axon_recon/` →
  0 hits (unless `_run_slay_analyzer_recompute` is still used by `_run_slay_merge_method`).
- `git grep -n "_run_merge_methods_for_target" src/axon_recon/` → 0 hits.
- Spikesort tests green.

**Commit:** `claude: spikesort-merge-cleanup, merge orchestrator is SLAy-only (slice 2)`

---

### Slice 3 — Delete cache helpers and `working_cache:` plumbing

**Goal:** Remove the four cache helpers and all of their config plumbing. Mutation safety
is delivered by `snapshot_sorter_output` (slice 1 of the prior plan) and SLAy's per-run
scratch (slice 4). The cache helpers are now redundant.

**Files:**
- `src/axon_recon/pipeline/stages/spikesort/runner.py`:
  - Delete `_cache_sorting_outputs_before_merge`, `_cache_canonical_sorter_output_for_merge`,
    `_restore_sorting_outputs_from_pre_merge_cache`.
  - Delete the 3 cache-helper call sites inside the SLAy-only orchestrator
    (`cache_restore_summary`, `cache_summary`, `canonical_workspace_summary`).
  - Delete the `pre_merge_workspace_*` and `working_cache_*` flat config reads.
  - Delete the assertion that points users at
    `stages.spikesort.phases.merge_units.working_cache.<knob>` (`runner.py:~3155, ~3165`).
- `src/axon_recon/pipeline/stages/spikesort/config.py`:
  - Delete `cache_sorting_outputs_before_merge_*`, `pre_merge_workspace_*`,
    `working_cache_*` flat fields on `SpikesortStageConfig`.
  - Delete the corresponding YAML→config parsing.
- `debug/debug.runtime.yml`:
  - Delete the `phases.merge_SLAy.working_cache:` sub-block.

**Tests:**
- Update `test_runner.py` to remove cache-helper coverage.
- Add (if not already present) a SLAy-after-snapshot smoke unit test asserting that
  with no cache helpers in the orchestrator, sorter_output is preserved under dry_run
  and the snapshot/restore round-trip still works.

**Acceptance:**
- `git grep -nE "_cache_sorting_outputs_before_merge|_cache_canonical_sorter_output_for_merge|_restore_sorting_outputs_from_pre_merge_cache" src/axon_recon/` → 0 hits.
- `git grep -nE "working_cache:|pre_merge_cache:|merge_workspace:" debug/*.yml` → 0 hits.
- `git grep -nE "cache_sorting_outputs_before_merge|pre_merge_workspace" src/axon_recon/` → 0 hits.
- Spikesort tests green.
- S1 + S2 smokes pass (or BLOCKED-SMOKE).

**Commit:** `claude: spikesort-merge-cleanup, delete cache helpers + working_cache plumbing (slice 3)`

---

### Slice 4 — Migrate replot analyzer building to `concat_analyzer`

**Goal:** Delete `_prepare_replot_workspace_analyzer` and the legacy analyzer family that
backed it. Pre-merge analyzer load goes through `_load_concat_analyzer_for_phase`.
Post-merge analyzer is built fresh from the merged sorter_output via a small dedicated
helper (or, if the replot use case is dead, deleted entirely).

**Files:**
- `src/axon_recon/pipeline/stages/spikesort/runner.py`:
  - Delete `_prepare_replot_workspace_analyzer`.
  - Replace the 2 call sites (pre-merge replot, post-merge replot) inside the SLAy-only
    orchestrator with: pre-merge → `_load_concat_analyzer_for_phase` (returns the
    canonical analyzer that lives at `<well>/spikesort_outputs/concat_analyzer/`).
    Post-merge → a new tiny helper that builds an analyzer from the post-merge sorter_output
    directly. The helper can live in `core/concat_analyzer.py` if appropriate, otherwise
    inline.
  - Delete `_load_or_recompute_spikesort_analyzer`, `_recompute_spikesort_analyzer`,
    `_recompute_sorting_analyzer_to_dir`.
  - If `_run_slay_analyzer_recompute` survived slice 2 because it called
    `_recompute_spikesort_analyzer`, rewrite it to load the canonical concat_analyzer (or
    delete if SLAy no longer needs an analyzer recompute artifact).

**Tests:**
- Update `test_runner.py` replot tests to expect the new code path.
- Drop `test_runner.py` tests for the deleted analyzer family.

**Acceptance:**
- `git grep -nE "_prepare_replot_workspace_analyzer|_load_or_recompute_spikesort_analyzer|_recompute_spikesort_analyzer|_recompute_sorting_analyzer_to_dir" src/axon_recon/` → 0 hits.
- `git grep -c "create_sorting_analyzer\|SortingAnalyzer.create" src/axon_recon/pipeline/stages/spikesort/runner.py` → at most 1 (the concat_analyzer integration). Ideally 0 (the only construction site lives in `core/concat_analyzer.py`).
- Spikesort tests green.
- S3 smoke (bombcell after concat_analyzer) passes; S1+S2 still pass.

**Commit:** `claude: spikesort-merge-cleanup, replot uses concat_analyzer, legacy analyzer family deleted (slice 4)`

---

### Slice 5 — Final config / YAML sweep + DoD audit

**Goal:** Strip every remaining cache-related config field and YAML key. Run the prior
plan's §6 cleanup grep at 0 hits.

**Files:**
- `src/axon_recon/pipeline/stages/spikesort/config.py`:
  - Audit `SpikesortStageConfig` for any remaining `cache_*`, `working_cache_*`,
    `pre_merge_*`, `merge_workspace_*`, `auto_merge_*`, `unitmatch_*` flat fields. Delete.
  - Audit the YAML→config parser for the corresponding reads. Delete.
- `debug/debug.runtime.yml`:
  - Final pass: ensure no `working_cache:`, `pre_merge_cache:`, `merge_workspace:`,
    `merge_si_auto:`, `merge_unitmatch:`, `auto_merge:`, `unitmatch:` keys survive.
- `src/axon_recon/pipeline/stages/spikesort/tests/test_spikesort_config.py`:
  - Drop tests for the deleted fields.

**Tests:**
- The full spikesort suite must remain green.

**Acceptance — the previous plan's §6 cleanup checklist, all 0 hits:**
- `git grep -nE "cache_sorter_output_before_analyzer_gen|publish_cached_sorter_output|cleanup_cached_sorter_output" -- src/axon_recon/` → 0 hits.
- `git grep -nE "cache_sorter_output_before_analyzer_gen:|publish_cached_sorter_output|cleanup_cached_sorter_output|cache_sorting_outputs_before_merge:|pre_merge_cache:|merge_workspace:|working_cache:" -- '*.yml' '*.yaml'` → 0 hits.
- `git grep -n "create_sorting_analyzer\|SortingAnalyzer.create" src/axon_recon/pipeline/stages/spikesort/` → at most 1 hit (concat_analyzer integration).
- `git grep -nE "\bmerge_si_auto\b|\bmerge_unitmatch\b|\bauto_merge\b" src/axon_recon/` → 0 hits.

**Commit:** `claude: spikesort-merge-cleanup, final config and YAML sweep (slice 5)`

---

### Slice 6 — End-to-end smoke matrix + mutation-safety regression suite refresh

**Goal:** Run S1–S6 against a real fixture (gated). Update the mutation-safety suite to
cover the new orchestrator shape.

**Steps:**
1. If a post-`spikesort.sort` fixture exists on the target server: run S1–S6 verbatim,
   capture stdout to `/tmp/smoke_slice6_<S>.log`, attach to commit notes.
2. If unavailable: mark each smoke `BLOCKED-SMOKE` in commit notes, naming the missing
   precondition exactly (same wording as the prior plan).
3. Update `tests/test_mutation_safety.py` (slice 7 of the prior repair) so the
   "must-not-mutate-on-build" check is asserted for the new SLAy-only orchestrator (not the
   deleted method dispatch).
4. Verify `tests/_mutation_safety.py` helpers still work; no behavior change needed.

**Acceptance:**
- All previous spikesort tests remain green.
- New mutation-safety assertions cover the SLAy-only orchestrator on the snapshot/restore
  contract.
- If smokes ran: byte-identical sha256 in S1; expected change in S2; expected labels in S3.

**Commit:** `claude: spikesort-merge-cleanup, smoke matrix + mutation-safety suite refresh (slice 6)`

---

## 5. Validation Matrix

| Check | Slice landed | Cmd |
|---|---|---|
| Spikesort tests green | 1–6 | `pytest src/axon_recon/pipeline/stages/spikesort/tests/ -q` |
| Pipeline tests green (excl. test_progress.py) | 1–6 | `pytest src/axon_recon/pipeline/tests/ --ignore=src/axon_recon/pipeline/tests/test_progress.py -q` |
| No merge_si_auto / merge_unitmatch in code | 1 | `git grep -nE "\bmerge_si_auto\b|\bmerge_unitmatch\b" src/axon_recon/` |
| No merge_si_auto / merge_unitmatch in YAML | 1 | `git grep -nE "\bmerge_si_auto\b|\bmerge_unitmatch\b" debug/*.yml` |
| No method-dispatch loop | 2 | `git grep -n "_run_auto_merge_method\|_run_merge_methods_for_target" src/axon_recon/` |
| No cache helpers | 3 | `git grep -nE "_cache_sorting_outputs_before_merge|_cache_canonical_sorter_output_for_merge|_restore_sorting_outputs_from_pre_merge_cache" src/axon_recon/` |
| No working_cache YAML | 3 | `git grep -nE "working_cache:" debug/*.yml` |
| No legacy analyzer family | 4 | `git grep -nE "_prepare_replot_workspace_analyzer|_load_or_recompute_spikesort_analyzer|_recompute_spikesort_analyzer|_recompute_sorting_analyzer_to_dir" src/axon_recon/` |
| `create_sorting_analyzer` only in concat_analyzer integration | 4 | `git grep -c "create_sorting_analyzer\|SortingAnalyzer.create" src/axon_recon/pipeline/stages/spikesort/runner.py` ≤ 1 |
| §6 cleanup grep 0 hits | 5 | listed above |

---

## 6. Cleanup Checklist (post-Slice 5)

All checks must return 0 hits at the end of slice 5:

```
git grep -nE "\bmerge_si_auto\b|\bmerge_unitmatch\b" src/axon_recon/
git grep -nE "\bmerge_si_auto\b|\bmerge_unitmatch\b" debug/*.yml
git grep -nE "_run_auto_merge_method|_run_merge_methods_for_target" src/axon_recon/
git grep -nE "_cache_sorting_outputs_before_merge|_cache_canonical_sorter_output_for_merge|_restore_sorting_outputs_from_pre_merge_cache|_prepare_replot_workspace_analyzer" src/axon_recon/
git grep -nE "_load_or_recompute_spikesort_analyzer|_recompute_spikesort_analyzer|_recompute_sorting_analyzer_to_dir" src/axon_recon/
git grep -nE "working_cache:|pre_merge_cache:|merge_workspace:" debug/*.yml
git grep -nE "cache_sorting_outputs_before_merge|pre_merge_workspace|cache_sorter_output_before_analyzer_gen|publish_cached_sorter_output|cleanup_cached_sorter_output" src/axon_recon/
```

And the `create_sorting_analyzer` site count in `runner.py`:

```
git grep -c "create_sorting_analyzer\|SortingAnalyzer.create" src/axon_recon/pipeline/stages/spikesort/runner.py
```

Expected ≤ 1.

---

## 7. Risks And Out-Of-Scope

**Risks:**

1. The replot pre/post-merge analyzer use case may require shape parity with the deleted
   `_prepare_replot_workspace_analyzer`. If the new post-merge helper diverges, replot
   reports could change. Mitigation: keep extensions list identical, run S6 end-to-end
   if a fixture is available, otherwise note as a known divergence in commit notes.

2. The `_run_slay_merge_method` still imports/uses `_recompute_spikesort_analyzer`
   (via `_run_slay_analyzer_recompute`). Slice 4 must either rewrite that to use
   concat_analyzer or delete `_run_slay_analyzer_recompute` if it has no real consumer.
   Risk: SLAy's recompute artifact (`slay_analyzer_recompute_summary.json`) may be
   referenced downstream. Check with grep before deleting.

3. CLI alias removal (slice 1) is a breaking change for any user script invoking the old
   names. Mitigation: callout in commit notes; the script can be updated to `merge_SLAy`
   or just `spikesort` (full stage).

4. YAML edits inside slice 1 and 3 touch the canonical debug runtime config. If a
   concurrent branch references the merge_si_auto / working_cache sub-trees, a merge
   conflict is likely. Mitigation: slice 1 should be the first commit on the new branch,
   making the rename obvious to reviewers.

**Out of scope:**

- Refactoring the SLAy orchestrator beyond removing method-dispatch and cache scaffolding.
- Changing the snapshot_sorter_output or concat_analyzer phases (locked from the prior repair).
- Touching reconstruct, preprocess, or templates phases.
- Changing the smoke matrix beyond what the prior plan defined.

---

## 8. Definition Of Done

1. All 6 slices merged in order, each its own `claude:` commit and commit-notes entry.
2. `merge_si_auto` and `merge_unitmatch` phases gone from code, CLI, aliases, YAML, tests,
   and the default phase sequence.
3. `_run_merge_methods_for_target` and `_run_auto_merge_method` deleted.
4. The 4 cache helpers deleted.
5. The legacy analyzer family (`_load_or_recompute_spikesort_analyzer`,
   `_recompute_spikesort_analyzer`, `_recompute_sorting_analyzer_to_dir`) deleted.
6. `debug/debug.runtime.yml` has no `working_cache:`, no `merge_si_auto:`, no
   `merge_unitmatch:` blocks. No `cache_sorting_outputs_before_merge_*` or
   `pre_merge_workspace_*` flat config survives.
7. §6 cleanup-grep checks all 0 hits; `create_sorting_analyzer` site count ≤ 1.
8. Spikesort test suite green; pipeline-level tests at baseline or better.
9. Smoke matrix S1–S6 either passes or is marked `BLOCKED-SMOKE` with the exact missing
   precondition recorded.
10. Mutation-safety regression suite covers the new SLAy-only orchestrator shape.

When 1–10 hold, write a `SPIKESORT MERGE CLEANUP COMPLETE` entry at the top of
`debug/agent_guardrails_commit_notes.md` summarizing test counts and any BLOCKED-SMOKE
items. Merge the branch back to `claude-migration`.

---

## 9. Operating Contract (if running this under /loop)

If this plan is driven by an autonomous /loop session, reuse the contract from
`debug/spikesort_label_merge_repair_loop_prompt.md` with these substitutions:

- Plan file: `debug/spikesort_merge_cleanup_plan.md` (this file).
- Number of slices: 6 (not 7).
- Branch: `spikesort-merge-cleanup` (off `claude-migration`).
- Halt condition: §8 DoD satisfied + final commit notes entry titled
  `SPIKESORT MERGE CLEANUP COMPLETE`.
- Stop signals, failure handling, commit prefix, conda env, mutation-safety check rules
  unchanged.

Commit prefix on every slice: `claude: spikesort-merge-cleanup, <slice description> (slice N)`.
