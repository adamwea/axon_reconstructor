# Spikesort Repair Plan — Fix `bombcell_label` And `merge_SLAy`

Status: implementation plan. Sibling to `parallelism_migration_plan.md`. Hand off one slice at a time. Same operating contract: each slice is self-contained, tests must pass, commits prefixed `claude:` when Claude does the work, update `debug/agent_guardrails_commit_notes.md` after each commit.

This plan does not depend on the parallelism migration. Order with respect to that plan: this work and the parallelism work touch overlapping files (`spikesort/runner.py` mostly), so do not run them in parallel. Pick one and finish before the other. Recommended order: parallelism first (it deletes legacy plumbing), then this plan (it touches phase-specific orchestration that's largely orthogonal to the parallelism cleanup).

---

## 0. Goal And End State

### Today

`spikesort.bombcell_label` and `spikesort.merge_SLAy` are tangled, mutate `sorter_output` in place, and have no dry-run. Iterating on either of them requires re-running `spikesort.sort` (slow, GPU-bound). Each phase builds its own SortingAnalyzer from scratch even though both want the same concat-level analyzer. Around them lives a maze of bespoke cache/publish/cleanup config knobs that are hard to reason about.

Quoting the survey:
- `bombcell_label` mutates `<ks_dir>/cluster_KSLabel.tsv` and `<ks_dir>/cluster_group.tsv` directly (`runner.py:2622-2676`). It has its own pre-cache (`runner.py:1454-1494`), publish-back (`runner.py:1497-1516`), and cleanup (`runner.py:1519-1543`) helpers, all gated by config flags like `bombcell_label_cache_sorter_output_before_analyzer_gen`, `bombcell_label_publish_cached_sorter_output_on_success`, `bombcell_label_cleanup_cached_sorter_output_on_success`.
- `merge_SLAy` runs SLAy's `run_slay()` against the kilosort folder in place (`runner.py:7611-7800+`). Its caching lives one level higher in `_cache_sorting_outputs_before_merge` (`runner.py:187-225`), `_cache_canonical_sorter_output_for_merge` (`runner.py:256-290`), `_prepare_replot_workspace_analyzer` (`runner.py:1330+`), and `_restore_sorting_outputs_from_pre_merge_cache` (`runner.py:227-253`).
- `merge_unitmatch` already has `dry_run` (defaulting True) wired through `um_kwargs`. `bombcell_label` and `merge_SLAy` do not.
- No standalone `concat_analyzer` phase exists. Both phases reference the YAML `spikeinterface_analyzer_concat` resource class — the intent is there, the implementation is not.
- No tests assert that `sorter_output` is byte-identical before/after a phase that should not mutate it.

### Target

```
spikesort phase sequence (canonical, after this plan):

  bootstrap_concat_binary
  sort
  summarize_sort
  snapshot_sorter_output           ← NEW: cheap recursive copy, runs once
  concat_analyzer                  ← NEW: builds the canonical SortingAnalyzer
  bombcell_label                   ← consumes concat_analyzer; honors dry_run
  merge_SLAy                       ← consumes concat_analyzer; honors dry_run
  merge_si_auto                    ← consumes concat_analyzer
  merge_unitmatch                  ← consumes concat_analyzer
  cleanup_concat_binary
```

Two new phases (`snapshot_sorter_output`, `concat_analyzer`) replace the bespoke per-phase caching. Every mutating phase honors `dry_run`. Restoring sorter_output to its post-sort state is one CLI flag away (no re-run of sort).

### What is being deleted

- `_prepare_bombcell_sorter_output_workspace` (`runner.py:1454-1494`) and the `bombcell_out_dir/cache/sorter_output/` concept.
- `_publish_bombcell_cached_workspace_outputs` (`runner.py:1497-1516`).
- `_cleanup_bombcell_success_outputs` parts that handle sorter_output (the analyzer-cleanup parts merge into the new `concat_analyzer` phase if needed).
- `_cache_sorting_outputs_before_merge` (`runner.py:187-225`) and the `pre_merge_cache/` concept.
- `_cache_canonical_sorter_output_for_merge` (`runner.py:256-290`) and the `cache/merge_workspace/` concept.
- `_prepare_replot_workspace_analyzer` (`runner.py:1330+`) and its freshness-check logic. The new `concat_analyzer` phase owns this.
- `_restore_sorting_outputs_from_pre_merge_cache` (`runner.py:227-253`) — replaced by a much simpler snapshot-restore CLI command.
- All YAML knobs under `bombcell_label` and `merge_SLAy` named like `cache_sorter_output_*`, `publish_cached_sorter_output_*`, `cleanup_cached_sorter_output_*`, `working_cache.*`, `cache_sorting_outputs_before_merge*`, `analyzer_output` subblocks owned per-phase. The new shape: `dry_run`, `enabled`, `relpath`, plus phase-specific behavioral knobs (e.g., `params`, `apply_to_sorter_output`).

### What stays

- The phase entry shapes (`run_spikesort_bombcell_label_stage`, `run_spikesort_merge_stage`) — they still take inputs, return result objects, write summary JSONs.
- The kilosort-folder file format (`cluster_KSLabel.tsv`, `cluster_group.tsv`) — that's an external contract.
- SLAy's own `run_slay()` API — the phase wraps it; we don't change SLAy.
- The merge-stage orchestration that runs multiple merge methods in sequence.

---

## 1. End-State Schema (YAML Sketch)

```yaml
stages:
  spikesort:
    phases:
      sort:
        enabled: true
        # ...

      summarize_sort:
        enabled: true

      snapshot_sorter_output:                  # NEW
        enabled: true
        relpath: spikesort_outputs/sorter_output_snapshot
        # If true and a snapshot already exists, compare-and-skip; if false, always re-snapshot.
        skip_if_exists: true

      concat_analyzer:                         # NEW
        enabled: true
        relpath: spikesort_outputs/concat_analyzer
        analyzer:
          format: binary_folder
          extensions:
            random_spikes: { max_spikes_per_unit: 500 }
            waveforms: { ms_before: 1.0, ms_after: 2.0 }
            templates: {}
            # ...other extensions exactly once, here, not duplicated per-phase.
        # If true and an analyzer already exists matching current sorter_output,
        # skip rebuild; otherwise rebuild.
        rebuild_on_sorter_output_change: true

      bombcell_label:
        enabled: true
        relpath: spikesort_outputs/bombcell_label
        dry_run: true                          # NEW: defaults true while broken; flip to false when stable
        params:
          label_non_somatic: false
          split_non_somatic_good_mua: false
        apply_to_sorter_output: true           # writes cluster_*.tsv ONLY if dry_run=false
        write_cluster_group: true
        fail_on_error: true
        reports:
          summary_json_relpath: bombcell_label_summary.json
        # NO MORE: cache_sorter_output_before_analyzer_gen, publish_cached_*, cleanup_cached_*,
        # analyzer subblock (analyzer comes from concat_analyzer phase).

      merge_SLAy:
        enabled: true
        relpath: spikesort_outputs/merge_SLAy
        dry_run: true                          # NEW: defaults true; flip when ready to apply merges
        slay_params:
          auto_accept_merges: false
          plot_merges: true
          allow_numpy_fallback: false
        # NO MORE: working_cache, cache_sorting_outputs_before_merge, pre_merge_cache, etc.

      merge_si_auto:
        enabled: true
        dry_run: true
        # ...

      merge_unitmatch:
        enabled: true
        dry_run: true
        # ...

      cleanup_concat_binary:
        enabled: true
```

CLI additions:
```bash
# Restore sorter_output from snapshot (cheap; bypasses re-running sort)
axon-recon stages spikesort.restore_sorter_output --config debug/debug.runtime.yml \
  --target-dataset 11,12 --limit-wells 1
```

---

## 2. Discovery Targets (Read These First)

```text
src/axon_recon/pipeline/stages/spikesort/runner.py                     # 10.7K lines; grep, do not read whole
src/axon_recon/pipeline/stages/spikesort/config.py                      # phase sequence, defaults
src/axon_recon/pipeline/stages/spikesort/orchestrators/bombcell_label.py
src/axon_recon/pipeline/stages/spikesort/orchestrators/merge_slay.py
src/axon_recon/pipeline/stages/spikesort/models/                        # phase config dataclasses
src/axon_recon/pipeline/stages/spikesort/tests/test_runner.py
src/axon_recon/pipeline/cli.py                                          # _STAGE_HANDLERS dict
debug/debug.runtime.yml                                                 # phases.bombcell_label, phases.merge_SLAy
```

### Key code references (verified by survey)

| Concern | File:Line | What it does |
|---|---|---|
| bombcell_label entry | `spikesort/orchestrators/bombcell_label.py:13-29` | thin wrapper |
| bombcell_label runner | `spikesort/runner.py:3137-3175` | orchestrates the phase |
| bombcell_label sorter_output mutation | `spikesort/runner.py:2622-2676` | `_apply_bombcell_labels_to_kilosort_outputs` writes cluster_*.tsv |
| bombcell pre-cache | `spikesort/runner.py:1454-1494` | `_prepare_bombcell_sorter_output_workspace` (delete) |
| bombcell publish-back | `spikesort/runner.py:1497-1516` | `_publish_bombcell_cached_workspace_outputs` (delete) |
| bombcell cleanup | `spikesort/runner.py:1519-1543` | `_cleanup_bombcell_success_outputs` (slim down or delete) |
| bombcell call-site | `spikesort/runner.py:2906-2976, 3057-3068` | invokes the cache helpers |
| merge_SLAy entry | `spikesort/orchestrators/merge_slay.py:17-41` | wrapper |
| merge_SLAy method | `spikesort/runner.py:7611-7800+` | `_run_slay_merge_method` (in-place mutation) |
| merge stage runner | `spikesort/runner.py:8206` | `run_spikesort_merge_stage` |
| merge pre-cache | `spikesort/runner.py:187-225` | `_cache_sorting_outputs_before_merge` (delete) |
| merge canonical workspace | `spikesort/runner.py:256-290` | `_cache_canonical_sorter_output_for_merge` (delete) |
| merge analyzer prep | `spikesort/runner.py:1330+` | `_prepare_replot_workspace_analyzer` (delete; replaced by concat_analyzer phase) |
| merge restore-from-cache | `spikesort/runner.py:227-253` | `_restore_sorting_outputs_from_pre_merge_cache` (delete; replaced by snapshot CLI) |
| canonical sorter_output resolution | `spikesort/runner.py:9520-9553` | resolves `resolved_sorter_output_dir` |
| phase sequence | `spikesort/config.py:267-276` | `DEFAULT_SPIKESORT_PHASE_SEQUENCE` |
| CLI handlers | `pipeline/cli.py:267,270` | spikesort.bombcell_label, spikesort.merge_SLAy |
| YAML bombcell block | `debug/debug.runtime.yml:538-621` | knobs to delete |
| YAML merge_SLAy block | `debug/debug.runtime.yml:623+` | knobs to delete |
| existing dry_run reference | `spikesort/legacy_runner.py` (um_kwargs) | merge_unitmatch dry_run pattern to mirror |

### Tests for the targeted phases

- `spikesort/tests/test_runner.py:7987` — `test_run_bombcell_label_phase_updates_kilosort_label_files`
- `spikesort/tests/test_runner.py:8110` — `test_run_bombcell_label_phase_uses_cached_sorter_output_workspace` (will become obsolete in Slice 5)
- `spikesort/tests/test_runner.py:8246` — `test_run_bombcell_label_phase_publishes_cached_workspace_outputs_on_success` (obsolete, Slice 5)
- `spikesort/tests/test_runner.py:8376` — `test_run_spikesort_merge_stage_does_not_invoke_bombcell_when_enabled`
- `spikesort/tests/test_runner.py:4194` — merge_SLAy in merge_stage tests

Run before AND after every slice:
```bash
conda run -n axon_recon python -m pytest src/axon_recon/pipeline/stages/spikesort/tests/ -q
```

---

## 3. Smoke Test Matrix

CLI flag shape matches the parallelism plan and `debug/mpirun.sh`. Use 2 datasets × 1 well unless a specific scenario calls for same-source contention.

```bash
# Smoke S0: full sort once (slow; do not repeat between slices unless --force-restart needed)
axon-recon stages spikesort.bootstrap_concat_binary spikesort.sort spikesort.summarize_sort \
  --config debug/debug.runtime.yml \
  --target-dataset 11,12 --limit-wells 1 --limit-segments 2 --force-restart

# Smoke S1: snapshot only (cheap)
axon-recon stages spikesort.snapshot_sorter_output \
  --config debug/debug.runtime.yml \
  --target-dataset 11,12 --limit-wells 1

# Smoke S2: build canonical concat_analyzer
axon-recon stages spikesort.concat_analyzer \
  --config debug/debug.runtime.yml \
  --target-dataset 11,12 --limit-wells 1

# Smoke S3: bombcell_label DRY-RUN (sorter_output must NOT change)
axon-recon stages spikesort.bombcell_label \
  --config debug/debug.runtime.yml \
  --target-dataset 11,12 --limit-wells 1 \
  --override 'stages.spikesort.phases.bombcell_label.dry_run=true'

# Smoke S4: bombcell_label APPLY (cluster_*.tsv get written)
axon-recon stages spikesort.bombcell_label \
  --config debug/debug.runtime.yml \
  --target-dataset 11,12 --limit-wells 1 \
  --override 'stages.spikesort.phases.bombcell_label.dry_run=false'

# Smoke S5: restore snapshot, undoing S4
axon-recon stages spikesort.restore_sorter_output \
  --config debug/debug.runtime.yml \
  --target-dataset 11,12 --limit-wells 1

# Smoke S6: merge_SLAy DRY-RUN
axon-recon stages spikesort.merge_SLAy \
  --config debug/debug.runtime.yml \
  --target-dataset 11,12 --limit-wells 1 \
  --override 'stages.spikesort.phases.merge_SLAy.dry_run=true'

# Smoke S7: merge_SLAy APPLY
axon-recon stages spikesort.merge_SLAy \
  --config debug/debug.runtime.yml \
  --target-dataset 11,12 --limit-wells 1 \
  --override 'stages.spikesort.phases.merge_SLAy.dry_run=false'

# Smoke S8: full spikesort end-to-end after the rewrite
axon-recon stages spikesort \
  --config debug/debug.runtime.yml \
  --target-dataset 11,12 --limit-wells 1 --limit-segments 2
```

If `--override` is not yet supported by the CLI, edit the YAML inline for the smoke run and revert. Verify the override flag's actual spelling with `axon-recon stages spikesort.bombcell_label --help`.

### Mutation-safety check (used by Slice 3 and Slice 4 acceptance)

```bash
# Capture sorter_output checksum, run dry-run phase, re-checksum, diff.
SORTER_OUT="<well_out_dir>/spikesort_outputs/sorter_output"
sha256sum "$SORTER_OUT"/* > /tmp/before.sha
axon-recon stages spikesort.bombcell_label --override '...dry_run=true' ...
sha256sum "$SORTER_OUT"/* > /tmp/after.sha
diff /tmp/before.sha /tmp/after.sha   # MUST be empty
```

---

## 4. Migration Slices

### Slice 1 — Add `snapshot_sorter_output` phase

**Goal**: A cheap, deterministic snapshot of `sorter_output` taken once after `summarize_sort`. Restoration is a separate CLI handler (`spikesort.restore_sorter_output`), not a phase. Replaces all the bespoke pre-cache logic in `bombcell_label` and `merge_SLAy` that's about to be deleted.

**A. Add config**:
- New phase config dataclass `SpikesortSnapshotSorterOutputPhaseConfig` in `spikesort/models/<wherever phase configs live; grep>`. Fields:
  - `enabled: bool = True`
  - `relpath: str = "spikesort_outputs/sorter_output_snapshot"`
  - `skip_if_exists: bool = True`
- Add it to `SpikesortPhasesConfig` (search for `bombcell_label_phase: ...` to find the parent dataclass).
- Add `"snapshot_sorter_output"` to `DEFAULT_SPIKESORT_PHASE_SEQUENCE` in `spikesort/config.py:267-276`, immediately after `"summarize_sort"`, before `"bombcell_label"`.

**B. Add core**: Create `src/axon_recon/pipeline/stages/spikesort/core/snapshot_sorter_output.py` (or co-locate near other small helpers — match existing pattern):
```python
def run_snapshot_sorter_output_phase(
    *,
    sorter_output_dir: Path,
    snapshot_dir: Path,
    skip_if_exists: bool,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Recursively copy sorter_output_dir to snapshot_dir.

    Returns a summary dict. Idempotent when skip_if_exists=True.
    """
```
- Use `shutil.copytree(..., dirs_exist_ok=False)` and write a one-line `snapshot_summary.json` containing source path, snapshot path, file count, total bytes, ISO timestamp.
- If `skip_if_exists` and snapshot exists, return `{"skipped": True, "reason": "snapshot_exists"}`.

**C. Add runner shim**: In `spikesort/runner.py`, add `run_spikesort_snapshot_sorter_output_phase(inputs)` that resolves paths and calls the core function. Wire it into the phase resolver helpers (search for `_normalize_spikesort_stage_phase_name`, `_spikesort_stage_phase_runner`, `_spikesort_stage_phase_enabled` — match the existing patterns).

**D. Add `restore_sorter_output` CLI handler** (it is NOT a phase; it's a pure CLI utility):
- New handler `_run_spikesort_restore_sorter_output_from_args` in `spikesort/cli.py` or `pipeline/cli.py` that:
  - Resolves the snapshot path and the canonical sorter_output path for each selected target.
  - For each target: `shutil.rmtree(canonical, ignore_errors=False)` then `shutil.copytree(snapshot, canonical)`.
  - Refuses to run if the snapshot does not exist.
  - Refuses to run unless the user passes `--confirm` (since it overwrites canonical state).
- Register `"spikesort.restore_sorter_output"` in `_STAGE_HANDLERS` in `pipeline/cli.py` (next to `spikesort.bombcell_label` at line 267).
- Document in the handler's argparse help: "Restore <well>/spikesort_outputs/sorter_output from snapshot. Requires --confirm. Useful for iterating on bombcell_label or merge_SLAy without re-running sort."

**E. YAML**: Add the `phases.snapshot_sorter_output` block to `debug/debug.runtime.yml` after `summarize_sort`.

**F. Tests**:
- `test_snapshot_sorter_output_creates_snapshot`: build a fake sorter_output dir, run the phase, assert snapshot is byte-identical (compare `sha256sum -c`).
- `test_snapshot_sorter_output_skips_when_exists`: run twice; second call returns `{"skipped": True}`.
- `test_restore_sorter_output_round_trip`: create snapshot, mutate canonical (touch a file), call restore, assert canonical matches snapshot byte-for-byte.
- `test_restore_sorter_output_refuses_without_confirm`: assert non-zero exit / clear error.

**Acceptance**:
- Smoke S1 produces `<well_out_dir>/spikesort_outputs/sorter_output_snapshot/` with file count + total bytes matching the source.
- Smoke S5 (restore) succeeds and produces a byte-identical canonical sorter_output.
- All existing spikesort tests still pass.
- `git grep -n "snapshot_sorter_output" src/axon_recon/pipeline/stages/spikesort/` lights up the new code.

**Commit**: `claude: add snapshot_sorter_output phase and restore_sorter_output CLI (slice 1)`

---

### Slice 2 — Add `concat_analyzer` phase

**Goal**: One canonical `SortingAnalyzer` shared by every downstream label/merge phase. Built once, consumed read-only. Replaces the per-phase analyzer-build code currently duplicated across `bombcell_label` and the merge stage's `_prepare_replot_workspace_analyzer` (`runner.py:1330+`).

**A. Inventory before extracting**: identify the shared seam.
```bash
grep -n "SortingAnalyzer\|create_sorting_analyzer\|load_sorting_analyzer" \
  src/axon_recon/pipeline/stages/spikesort/runner.py | head -40
grep -n "analyzer_output\|analyzer_relpath" \
  src/axon_recon/pipeline/stages/spikesort/config.py
```
Find the union of analyzer kwargs used by `bombcell_label` (currently `runner.py:1429-1451`, `2906-2976`) and the merge stage (`runner.py:1330+`). They should be near-identical — note any divergence and resolve toward the bombcell shape (the user indicated bombcell is the more complete of the two).

**B. Add config** `SpikesortConcatAnalyzerPhaseConfig`:
- `enabled: bool = True`
- `relpath: str = "spikesort_outputs/concat_analyzer"`
- `format: str = "binary_folder"`
- `rebuild_on_sorter_output_change: bool = True` — if True, write a fingerprint file (sha256 of sorter_output dir state) inside the analyzer dir and rebuild when fingerprint changes.
- `extensions: dict[str, dict]` — extension name → kwargs (random_spikes, waveforms, templates, noise_levels, etc.). Default to the union from bombcell+merge.
- `n_jobs: int | None = None` (post-parallelism-migration this comes from the phase budget; pre-migration, it's a YAML knob).

Add the dataclass next to other phase configs. Add `concat_analyzer` to `SpikesortPhasesConfig`. Add `"concat_analyzer"` to `DEFAULT_SPIKESORT_PHASE_SEQUENCE` between `"snapshot_sorter_output"` and `"bombcell_label"`.

**C. Extract analyzer-build code into a core module**: `src/axon_recon/pipeline/stages/spikesort/core/concat_analyzer.py`:
```python
def run_concat_analyzer_phase(
    *,
    sorter_output_dir: Path,
    recording: Any,                  # the bootstrap concat recording
    analyzer_dir: Path,
    extensions: dict[str, dict],
    rebuild_on_sorter_output_change: bool,
    n_jobs: int,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Build (or load) the canonical concat-level SortingAnalyzer.

    Writes a fingerprint file 'sorter_output_fingerprint.json' inside analyzer_dir.
    Returns {"analyzer_dir": str, "rebuilt": bool, "extension_count": int, ...}.
    """
```
Lift the working analyzer-creation code from `_prepare_replot_workspace_analyzer` (most complete) — copy, don't move yet. Remove the freshness-via-mtime logic and replace with the explicit fingerprint comparison. Keep the call sites in bombcell/merge intact for now; they'll be migrated in slices 3-4.

**D. Add runner shim**: `run_spikesort_concat_analyzer_phase(inputs)` in `spikesort/runner.py`. Resolves recording from the bootstrap_concat_binary outputs, resolves sorter_output and analyzer paths, calls the core function. Wire phase resolver/enabled/runner helpers as in Slice 1.

**E. CLI**: Register `"spikesort.concat_analyzer"` in `_STAGE_HANDLERS`. Add `_run_spikesort_concat_analyzer_from_args` that resolves runtime config and invokes the shim.

**F. YAML**: Add `phases.concat_analyzer` block to `debug/debug.runtime.yml`. Set `extensions` to the union of what bombcell and merge currently configure (consult their existing analyzer subblocks at debug.runtime.yml:540-560 area and 626-640 area).

**G. Tests**:
- `test_concat_analyzer_builds_once_and_skips_unchanged`: synthetic sorter_output + recording → run → assert analyzer dir created, fingerprint written. Run again → assert `rebuilt=False`.
- `test_concat_analyzer_rebuilds_when_sorter_output_changes`: run → mutate a file in sorter_output → run again → assert `rebuilt=True`.
- `test_concat_analyzer_extensions_present`: assert all configured extensions land in the analyzer dir.

**Acceptance**:
- Smoke S2 produces `<well_out_dir>/spikesort_outputs/concat_analyzer/` with all configured extensions and a `sorter_output_fingerprint.json`.
- Smoke S2 re-run is fast (skip).
- `git grep -n "concat_analyzer" src/axon_recon/pipeline/stages/spikesort/` lights up.

**Commit**: `claude: add concat_analyzer phase as canonical shared analyzer (slice 2)`

---

### Slice 3 — Refactor `bombcell_label` to consume `concat_analyzer` and honor `dry_run`

**Goal**: `bombcell_label` reads the canonical analyzer from `spikesort_outputs/concat_analyzer/`. It produces labels into its own out_dir always. It writes `cluster_KSLabel.tsv` / `cluster_group.tsv` into the canonical sorter_output ONLY when `dry_run=false`. All bespoke caching/publish/cleanup logic is deleted.

**A. Update phase config** `bombcell_label`:
- ADD: `dry_run: bool = true` (default true while the phase is being stabilized; the user flips to false when ready to apply).
- REMOVE: `cache_sorter_output_before_analyzer_gen`, `publish_cached_sorter_output_on_success`, `publish_cached_analyzer_on_success`, `cleanup_analyzer_on_success`, `cleanup_cached_sorter_output_on_success`, the `analyzer:` subblock (analyzer comes from concat_analyzer phase).
- KEEP: `enabled`, `relpath`, `delete_outputs_on_force_restart`, `params`, `apply_to_sorter_output`, `write_cluster_group`, `fail_on_error`, `reports`.

**B. Edit code**:
- Delete `_prepare_bombcell_sorter_output_workspace` (`runner.py:1454-1494`).
- Delete `_publish_bombcell_cached_workspace_outputs` (`runner.py:1497-1516`).
- Delete the sorter_output portions of `_cleanup_bombcell_success_outputs` (`runner.py:1519-1543`); keep any non-sorter-output cleanup.
- Delete `_load_or_recompute_bombcell_sorting_analyzer` (`runner.py:1429-1451`) — replace with a one-liner that loads the analyzer from `<well_out_dir>/spikesort_outputs/concat_analyzer/`.
- Update the bombcell call site (`runner.py:2906-2976` → labeling logic; `runner.py:3057-3068` → cleanup invocation):
  - Remove all calls into the deleted helpers.
  - Load analyzer from concat_analyzer dir; raise a clear error if missing ("run spikesort.concat_analyzer first").
  - Wrap `_apply_bombcell_labels_to_kilosort_outputs` (`runner.py:2622-2676`) with `if not phase_cfg.dry_run: <existing call>`.
  - When `dry_run=true`, still write the dry-run preview: dump the proposed label dict to `<bombcell_out_dir>/dry_run/proposed_cluster_KSLabel.tsv` and `proposed_cluster_group.tsv` so the user can inspect what would be applied.
- Update the orchestrator (`spikesort/orchestrators/bombcell_label.py:13-29`) signatures only if needed; the entry-point shape should be unchanged.

**C. YAML**: Edit `debug/debug.runtime.yml:538-621` per §1 schema. Set `dry_run: true` for now.

**D. Tests**:
- Update `test_run_bombcell_label_phase_updates_kilosort_label_files` (`test_runner.py:7987`) to use a fixture where the concat_analyzer phase has already been run. Assert cluster_*.tsv updates only when `dry_run=false`.
- DELETE `test_run_bombcell_label_phase_uses_cached_sorter_output_workspace` (`test_runner.py:8110`) — obsolete.
- DELETE `test_run_bombcell_label_phase_publishes_cached_workspace_outputs_on_success` (`test_runner.py:8246`) — obsolete.
- ADD `test_run_bombcell_label_phase_dry_run_does_not_mutate_sorter_output`:
  - Run sort fixture, snapshot sorter_output checksum.
  - Run bombcell_label with `dry_run=true`.
  - Re-checksum sorter_output → assert byte-identical.
  - Assert dry-run preview files exist under `<bombcell_out_dir>/dry_run/`.
- ADD `test_run_bombcell_label_phase_apply_writes_cluster_files`:
  - Same fixture, `dry_run=false`.
  - Assert cluster_KSLabel.tsv and cluster_group.tsv updated.
- ADD `test_run_bombcell_label_phase_requires_concat_analyzer`:
  - Run bombcell_label without first running concat_analyzer; assert clear error.

**Acceptance**:
- Smoke S3 (dry-run): `sha256sum` of sorter_output before/after is identical (per §3 mutation-safety check).
- Smoke S3 produces `<bombcell_out_dir>/dry_run/proposed_cluster_*.tsv`.
- Smoke S4 (apply): cluster_*.tsv updated; `bombcell_labels.json`, `bombcell_labels.tsv`, summary all present.
- Smoke S5 (restore) brings sorter_output back to post-S0 state cleanly.
- `grep -rn "_prepare_bombcell_sorter_output_workspace\|_publish_bombcell_cached_workspace_outputs\|_load_or_recompute_bombcell_sorting_analyzer" src/axon_recon/` returns 0 hits.

**Commit**: `claude: bombcell_label consumes concat_analyzer; honors dry_run (slice 3)`

---

### Slice 4 — Refactor `merge_SLAy` to consume `concat_analyzer` and honor `dry_run`

**Goal**: Mirror Slice 3 for merge_SLAy. Strip the merge-stage-level caching tangle; rely on `snapshot_sorter_output` for restore and on `concat_analyzer` for the analyzer.

**A. Update phase config** `merge_SLAy`:
- ADD: `dry_run: bool = true`.
- REMOVE: `working_cache.*`, `cache_sorting_outputs_before_merge`, `cache_sorting_outputs_before_merge_use_canonical_workspace`, the `analyzer:` subblock.
- KEEP: `enabled`, `rel_output_root`, `delete_outputs_on_force_restart`, `force_restart`, `force_replot`, `slay_enabled`, `slay_params`.

**B. Edit code**:
- Delete `_cache_sorting_outputs_before_merge` (`runner.py:187-225`).
- Delete `_cache_canonical_sorter_output_for_merge` (`runner.py:256-290`).
- Delete `_prepare_replot_workspace_analyzer` (`runner.py:1330+`) — its replacement is the `concat_analyzer` phase from Slice 2.
- Delete `_restore_sorting_outputs_from_pre_merge_cache` (`runner.py:227-253`) — replacement is the `restore_sorter_output` CLI from Slice 1.
- In `_run_slay_merge_method` (`runner.py:7611-7800+`):
  - Remove resolution of `resolved_sorter_output_dir` against the working cache (lines 9520-9553); the canonical sorter_output dir is the one and only target.
  - Load analyzer from `<well_out_dir>/spikesort_outputs/concat_analyzer/`; raise if missing.
  - Branch on `dry_run`:
    - **dry_run=true**: invoke SLAy with its built-in dry-run-equivalent path if it exists (check SLAy's API; `auto_accept_merges=False` may already mean "don't apply"). If SLAy cannot honor a true dry-run, copy sorter_output to a per-run scratch dir, point SLAy at that copy, and discard the scratch on completion. Either way, write the SLAy report and `run-output.json` into `<merge_out_dir>/dry_run/`.
    - **dry_run=false**: invoke SLAy against canonical sorter_output (current behavior).
- Update merge_stage runner (`runner.py:8206`) to remove every call into the deleted helpers.

**C. YAML**: Edit `debug/debug.runtime.yml:623+` per §1 schema. Set `dry_run: true`.

**D. Tests**:
- Migrate any merge_stage tests that referenced `pre_merge_cache` or `working_cache` (likely `test_runner.py:4194` and nearby) — delete obsolete assertions.
- ADD `test_run_merge_slay_phase_dry_run_does_not_mutate_sorter_output` (mirror of bombcell test).
- ADD `test_run_merge_slay_phase_apply_mutates_sorter_output_via_slay`:
  - `dry_run=false`. After running, assert sorter_output has been modified (some cluster_* file has changed).
- ADD `test_run_merge_slay_phase_requires_concat_analyzer`.
- ADD `test_run_merge_slay_phase_dry_run_preview_outputs_exist`:
  - Assert `<merge_out_dir>/dry_run/run-output.json` and `slay_method_summary.json` are written even in dry-run.

**Acceptance**:
- Smoke S6 (dry-run): sorter_output checksum unchanged.
- Smoke S7 (apply): sorter_output is modified by SLAy; `<merge_out_dir>/run-output.json` written.
- After Smoke S7, Smoke S5 (restore) reverts cleanly.
- `grep -rn "_cache_sorting_outputs_before_merge\|_cache_canonical_sorter_output_for_merge\|_prepare_replot_workspace_analyzer\|_restore_sorting_outputs_from_pre_merge_cache\|pre_merge_cache\|working_cache" src/axon_recon/pipeline/stages/spikesort/` returns 0 hits in non-test code.

**Commit**: `claude: merge_SLAy consumes concat_analyzer; honors dry_run (slice 4)`

---

### Slice 5 — Migrate `merge_si_auto` and `merge_unitmatch` to consume `concat_analyzer`

**Goal**: Eliminate the last two analyzer-build sites. Existing `merge_unitmatch.dry_run` semantics preserved; `merge_si_auto` gains `dry_run` to match.

**A. Inventory**:
```bash
grep -n "merge_si_auto\|run_merge_si_auto\|merge_unitmatch\|run_merge_unitmatch" \
  src/axon_recon/pipeline/stages/spikesort/runner.py | head -30
```
Locate each method's analyzer-build site.

**B. Edit code**:
- For each of `_run_merge_si_auto_method` and `_run_merge_unitmatch_method` (find the actual function names): replace the analyzer construction with a load from `<well_out_dir>/spikesort_outputs/concat_analyzer/`. Raise clearly if missing.
- Add `dry_run: bool = true` to `merge_si_auto`'s phase config (mirror merge_SLAy).
- For merge_unitmatch: ensure the existing `dry_run` lives at the phase config level (not buried in `um_kwargs`). Migrate if needed.

**C. YAML**: Update `phases.merge_si_auto` and `phases.merge_unitmatch` to expose `dry_run` directly.

**D. Tests**:
- ADD `test_run_merge_si_auto_phase_dry_run_does_not_mutate_sorter_output`.
- ADD `test_run_merge_unitmatch_phase_dry_run_does_not_mutate_sorter_output`.
- KEEP existing merge_unitmatch dry_run tests; verify they still pass.

**Acceptance**:
- Both phases load the canonical analyzer; no per-phase analyzer build code remains.
- Dry-run preserves sorter_output for both phases.
- `grep -rn "create_sorting_analyzer\b" src/axon_recon/pipeline/stages/spikesort/runner.py` returns at most one hit (inside `concat_analyzer` core).

**Commit**: `claude: merge_si_auto and merge_unitmatch consume concat_analyzer; uniform dry_run (slice 5)`

---

### Slice 6 — Strip dead code, dead config, dead docs

**Goal**: With Slices 1–5 in place, sweep the codebase and YAML for the now-unused caching scaffolding.

**A. Sweep**:
```bash
grep -rn "cache_sorter_output_before_analyzer_gen\|publish_cached_sorter_output\|cleanup_cached_sorter_output\|cache_sorting_outputs_before_merge\|pre_merge_cache\|merge_workspace\|working_cache\|_cleanup_bombcell_success_outputs" \
  src/axon_recon/ debug/ docs/
```
Every hit should be deleted (code, comments, YAML keys, tests, docs).

**B. Tighten phase config dataclasses**: remove the now-unused fields. Any field that survived Slices 3-4-5 only because it was harmless (e.g., default-False) — if the field never affected behavior, delete it.

**C. Update operator documentation**: scan `debug/*.md` for references to the old caching concepts. Update any that mention `pre_merge_cache`, `working_cache`, `bombcell.cache.*`, or "publish back".

**D. Final test pass**: make sure the spikesort test suite has no skipped/xfailed tests left over from the old behavior. Delete tests that asserted properties of the deleted caches.

**Acceptance**:
- The grep above returns 0 hits.
- `conda run -n axon_recon python -m pytest src/axon_recon/pipeline/stages/spikesort/tests/ -q` runs to completion with no skips related to this work.
- Smokes S0–S8 all pass end-to-end.

**Commit**: `claude: remove dead spikesort caching scaffolding (slice 6)`

---

### Slice 7 — Add mutation-safety regression tests

**Goal**: Lock in the new contract. After this slice, any future regression that re-introduces silent sorter_output mutation will fail in CI.

**A. Add test helpers**: `src/axon_recon/pipeline/stages/spikesort/tests/_mutation_safety.py`:
```python
def hash_directory(path: Path) -> dict[str, str]:
    """Return {relative_path: sha256_hex} for every file under path. Stable, sorted."""

def assert_directory_unchanged(path: Path, baseline: dict[str, str]) -> None:
    """Re-hash and assert equality; print diff on failure."""
```

**B. Add tests** in `tests/test_mutation_safety.py`:
- For each phase that may mutate sorter_output (`bombcell_label`, `merge_SLAy`, `merge_si_auto`, `merge_unitmatch`):
  - `test_<phase>_dry_run_preserves_sorter_output`: build fixture, snapshot, run with dry_run=true, assert directory unchanged.
- For each phase that should never mutate sorter_output (`snapshot_sorter_output`, `concat_analyzer`, `summarize_sort`):
  - `test_<phase>_never_mutates_sorter_output`: same shape, no dry_run knob — phase must never touch sorter_output.

**C. Wire into CI hooks** (if pre-commit / CI exists; check `agent_guardrails_commit_notes.md` for the hook contract). Otherwise just run as part of the standard test suite.

**Acceptance**:
- All new mutation-safety tests pass.
- Deliberately reverting Slice 3's `if not dry_run:` guard makes `test_bombcell_label_dry_run_preserves_sorter_output` fail — confirms the test catches the regression. (Re-apply the guard before committing.)

**Commit**: `claude: lock mutation-safety contract for label and merge phases (slice 7)`

---

## 5. Validation Matrix

| Slice | Tests must pass | Smokes |
|---|---|---|
| 1 | test_snapshot_*, test_restore_* | S0 (existing), S1, S5 |
| 2 | test_concat_analyzer_* | S0, S1, S2 |
| 3 | new bombcell tests; obsolete cache tests deleted | S3 (dry-run mutation-safety), S4 (apply), S5 (restore) |
| 4 | new merge_SLAy tests; obsolete cache tests deleted | S6 (dry-run), S7 (apply), S5 (restore) |
| 5 | new merge_si_auto + merge_unitmatch dry-run tests | S8 (full spikesort end-to-end) |
| 6 | full spikesort suite passes; no skips | S8 |
| 7 | new mutation-safety suite passes | S8 |

Run before each slice (baseline) and after each slice (regression):
```bash
conda run -n axon_recon python -m pytest src/axon_recon/pipeline/stages/spikesort/tests/ -q
```

---

## 6. Cleanup Checklist (post-Slice 7)

```bash
# (a) No legacy caching scaffolding left
git grep -nE "_prepare_bombcell_sorter_output_workspace|_publish_bombcell_cached_workspace_outputs|_load_or_recompute_bombcell_sorting_analyzer|_cache_sorting_outputs_before_merge|_cache_canonical_sorter_output_for_merge|_prepare_replot_workspace_analyzer|_restore_sorting_outputs_from_pre_merge_cache" src/

# (b) No legacy YAML knobs
git grep -nE "cache_sorter_output_before_analyzer_gen:|publish_cached_sorter_output|cleanup_cached_sorter_output|cache_sorting_outputs_before_merge:|pre_merge_cache:|merge_workspace:|working_cache:" -- '*.yml' '*.yaml'

# (c) The only remaining SortingAnalyzer construction is in concat_analyzer.py
git grep -n "create_sorting_analyzer\|SortingAnalyzer.create" src/axon_recon/pipeline/stages/spikesort/

# (d) Every label/merge phase has a dry_run knob
conda run -n axon_recon python -c "
from axon_recon.pipeline.config import load_pipeline_runtime_bundle
from axon_recon.pipeline.stages.spikesort.config import parse_spikesort_inputs
b = load_pipeline_runtime_bundle(config_path='debug/debug.runtime.yml')
inputs = parse_spikesort_inputs(b)
for phase in ('bombcell_label','merge_SLAy','merge_si_auto','merge_unitmatch'):
    cfg = getattr(inputs.phases, phase)
    assert hasattr(cfg, 'dry_run'), f'{phase} missing dry_run'
    print(f'{phase:20s} dry_run={cfg.dry_run}')
"

# (e) End-to-end with all four label/merge phases in dry-run mode preserves sorter_output
SORTER_OUT="<well_out_dir>/spikesort_outputs/sorter_output"
sha256sum "$SORTER_OUT"/* > /tmp/before.sha
axon-recon stages spikesort.bombcell_label spikesort.merge_SLAy spikesort.merge_si_auto spikesort.merge_unitmatch \
  --config debug/debug.runtime.yml --target-dataset 11,12 --limit-wells 1
sha256sum "$SORTER_OUT"/* > /tmp/after.sha
diff /tmp/before.sha /tmp/after.sha   # MUST be empty
```

(a), (b), (e) must be empty / clean. (c) returns at most one hit (the helper in `concat_analyzer.py`). (d) prints `dry_run=True` for all four phases by default.

---

## 7. Risks And Out-Of-Scope

### Risks

- **SLAy may not natively support a dry-run mode.** If `run_slay()` always mutates the input directory, Slice 4's dry-run uses a scratch-copy fallback (copy sorter_output to a temp dir, run SLAy there, discard). This costs disk I/O but avoids modifying the SLAy upstream.
- **Analyzer schema drift.** If different downstream phases want different analyzer extensions, the unified concat_analyzer must include the union. If that bloats the analyzer significantly, consider parameterizing extensions per-phase via `phase_budgets` rather than per-phase config — but only if it becomes a real problem.
- **Snapshot disk usage.** A snapshot doubles `sorter_output` disk footprint per well. On the lab server this is fine; on NERSC scratch it may not be. Document in `parallelism_agent_guardrails.md` (Storage section) and consider adding a `--no-snapshot` toggle on `spikesort.sort` if scratch pressure becomes real.
- **Concurrent writes.** If two phases ever ran in parallel against the same sorter_output, dry-run guarantees would race. Today the spikesort phase sequence is serial within a well; do not introduce intra-well phase parallelism without revisiting this contract.
- **External tools (Phy, etc.) reading sorter_output during a run.** Out of scope; assume no external readers during a phase.

### Out of scope

- Changing SLAy's behavior or API.
- Changing Bombcell's metric computation.
- Refactoring the merge orchestration that runs multiple methods in sequence — only the per-method internals change.
- Changing the kilosort folder file format (`cluster_KSLabel.tsv`, `cluster_group.tsv`).
- Changing `merge_unitmatch`'s existing dry-run semantics beyond exposing the flag at the phase config level.

---

## 8. Definition Of Done

The repair is complete when:

1. The 7 slices above are merged in order, each with passing acceptance checks.
2. `spikesort.snapshot_sorter_output` and `spikesort.concat_analyzer` exist as standalone phases in `DEFAULT_SPIKESORT_PHASE_SEQUENCE`.
3. `spikesort.restore_sorter_output` exists as a CLI utility (not a phase) that round-trips sorter_output cleanly.
4. `bombcell_label`, `merge_SLAy`, `merge_si_auto`, `merge_unitmatch` all expose a `dry_run: bool` config knob defaulting `true`. All four refuse to mutate sorter_output when dry_run=true; all four mutate it when dry_run=false. Mutation-safety regression tests cover every case.
5. Only one place in the codebase constructs a `SortingAnalyzer` for spikesort: `core/concat_analyzer.py`. All four label/merge phases load the analyzer from `<well_out_dir>/spikesort_outputs/concat_analyzer/` and refuse to run if it is absent.
6. The cleanup checklist (§6) commands all return clean.
7. Iterating on `bombcell_label` or `merge_SLAy` no longer requires re-running `spikesort.sort`. The user can: (a) snapshot once after first sort, (b) dry-run a phase to inspect proposed changes, (c) apply with dry_run=false, (d) restore from snapshot, (e) repeat (b)–(d) without ever re-running sort.

If any acceptance check fails, the slice does not land. Each slice is a single commit prefixed `claude:`. Update `debug/agent_guardrails_commit_notes.md` after every commit.
