# `--dry-run` rollout across every phase

Status: implementation plan. Realizes the `guardrails/dry_run.md` contract: every phase exposes `--dry-run` and short-circuits at input resolution. Same operating contract: one slice at a time, `claude:` commit prefix, append a line to `dev/notes/commit_log.md` after every commit.

**See also:**
- `guardrails/dry_run.md` — the contract this plan realizes.
- `guardrails/stage_phase_architecture.md` — the architectural invariants every phase satisfies; `--dry-run` joins the list of universal phase capabilities once this plan lands.
- `phase_roster_cleanup_plan.md` — the phase roster needs to be settled before broad changes touch every phase. Dry-run rollout can either run AFTER the roster cleanup, or in parallel for phases that are clearly staying.

---

## 0. End-state, in one paragraph

`axon-recon stages <stage>[.<phase>] --dry-run --config <yaml> [--targets ...]` works on every stage and every phase. It resolves inputs, validates prerequisites, writes a `<phase>_summary.json` with `status: dry_run_ok`, and exits in seconds. The user gets a fast wiring-only smoke check for any scope they want to verify before kicking off a real run.

---

## 1. Where the flag lives

Single `--dry-run` argparse entry at the shared CLI level (`pipeline/cli.py`), inherited by every stage / phase subcommand. Parsed into a process-wide override pattern (matching the existing `_TARGET_WELLS_OVERRIDE`, `_NO_PLOT_OVERRIDE`, `_ACTIVE_PROFILE_OVERRIDE`):

```python
# pipeline/config.py
_DRY_RUN_OVERRIDE: bool | None = None
def set_dry_run_override(enabled: bool | None) -> None: ...
def get_dry_run_override() -> bool | None: ...
```

Carried into `stage_config.dry_run` via the same plumbing the other overrides use. Cleared in the `finally` block of CLI main().

Phase implementations check the attribute at the top of their work block:
```python
if getattr(stage_config, "dry_run", False):
    return _write_dry_run_summary(
        phase_name="...",
        well_out_dir=...,
        stage_output_root_dir=...,
        inputs_resolved=[...],
        outputs_would_produce=[...],
    )
```

---

## 2. Shared helper

A single utility writes the dry-run summary. New module:

```
src/axon_recon/pipeline/dry_run.py
```

Exposes:

```python
def write_dry_run_summary(
    *,
    phase_name: str,
    well_out_dir: Path,
    stage_output_root_dir: Path,
    summary_json_path: Path,
    inputs_resolved: list[dict],
    outputs_would_produce: list[dict],
    validation: dict | None = None,
) -> Path:
    """Writes the standard dry-run summary JSON to summary_json_path.
    Returns the path. Used by every phase's dry-run short-circuit."""
```

Summary schema is fixed at the contract level (see `guardrails/dry_run.md` §3); the helper enforces it. New phases can extend with phase-specific fields but the base shape stays.

---

## 3. Implementation slices

One commit per slice. `claude:` prefix.

### Slice 1 — CLI flag + plumbing (no behavior)

#### Slice 1a — SHIPPED 2026-05-21 — process-wide override + CLI flag
- Added `--dry-run` argparse arg to the shared `stages` subparser
  (`pipeline/cli.py`).
- Added `_DRY_RUN_OVERRIDE` + `set/get_dry_run_override` in
  `pipeline/config.py` mirroring the existing override pattern
  (`--no-plot`, `--profile`, `--scratch-output`, `--output-root`,
  `--force-enable`).
- Wired into CLI main: set the override before dispatching, clear in
  `finally`.
- 7 tests in `test_dry_run_override.py` cover setter/getter invariants
  (None default, True/False distinct from None, clears with None),
  CLI flag parsing (present + absent), and combinability with
  `--force-enable` (the natural slice 3b smoke command).
- Phases read `get_dry_run_override()` directly when they implement
  their short-circuit (deferred to slices 3-7).

#### Slice 1b — pending — stage_config dataclass attribute
- Add `dry_run: bool = False` field to each stage_config dataclass
  (preprocess, spikesort, reconstruct, analysis).
- Thread `dry_run_override` kwarg through `parse_*_stage_config` and
  the substage runners.
- Optional — phases can read the process-wide override directly via
  `get_dry_run_override()` per slice 1a. Slice 1b adds the dataclass
  attribute for phases that prefer to consume from stage_config (more
  testable: tests can set `stage_config.dry_run=True` without
  manipulating a process-wide global).

### Slice 2 — `write_dry_run_summary` helper + base test — SHIPPED 2026-05-21

- New module `src/axon_recon/pipeline/dry_run.py` exports
  `write_dry_run_summary(*, phase_name, well_out_dir, stage_output_root_dir,
  summary_json_path, inputs_resolved, outputs_would_produce, validation=None,
  extra_fields=None) -> Path`. Enforces the schema from `guardrails/dry_run.md`
  §3 (status=dry_run_ok, base fields well_out_dir / stage_output_root_dir /
  phase / inputs_resolved / outputs_would_produce / validation).
- Defaults: empty `validation={missing_prerequisites:[], warnings:[]}` when
  caller passes None.
- Phase-specific extras allowed via `extra_fields=` — merged AFTER base
  fields so base shape always wins on conflict.
- Tests (7 in `tests/test_dry_run_summary.py`): base schema, default empty
  validation, validation carried through, parent-dir creation, extras
  accepted, extras can't shadow base, input-items normalized to the canonical
  `{name, path, exists}` shape.
- Phases adopting the dry-run short-circuit (slices 3-7) consume this
  helper. Combined with slice 1a's `get_dry_run_override()`, the per-phase
  short-circuit looks like:
  ```python
  if get_dry_run_override():
      return write_dry_run_summary(phase_name=..., ...)
  ```

### Slice 3 — Preprocess stage phase short-circuits — SHIPPED 2026-05-21

Single-intercept implementation: all preprocess phases dispatch through
`_run_preprocess_selected_phase` in `preprocess/runner.py`, so the
dry-run check goes at the top of that helper — one code path, every
phase covered. Each phase still writes to its standard per-phase
summary_json relpath; the dry-run summary preserves that contract.

Phases covered (5 of 5):
- ~~`save_rec_metadata`~~ — SHIPPED
- ~~`preprocess_segments`~~ — SHIPPED
- ~~`plot_segment_traces`~~ — SHIPPED
- ~~`plot_segment_channel_layouts`~~ — SHIPPED
- ~~`plot_raster_threshold`~~ — SHIPPED

Tests in `preprocess/tests/test_dry_run.py` (7 tests, parametrized
across 5 phases + h5-missing + h5-existing warning paths). Heavy
`_run_preprocess_phase_sequence` is stubbed to raise so any leak fails
the test.

### Slice 4 — Spikesort stage phase short-circuits
Same pattern. Phases (post-cleanup shape):
- `concat_binary` (renamed from `bootstrap_concat_binary`)
- `sort`
- `snapshot_sorter_output`
- `concat_analyzer`
- `cleanup_concat_binary`
- `cleanup_analyzers`

For `sort` specifically: dry-run must NOT load Kilosort, NOT load CUDA, NOT load the recording into memory. Just verify the recording manifest exists, the sorter_output dir is writable, params look sane, then write the summary.

### Slice 5 — Reconstruct stage phase short-circuits

**Progress (sub-slices landing out-of-order with the other plans):**

✅ **14 of 17 recon phases now have dry-run** (as of 2026-05-21):
- `kssynth` — SHIPPED via `kssynth_recon_integration_plan` slice 4e.
- `analyzers` — SHIPPED.
- `axon_velocity_gtrs` — SHIPPED.
- `plot_templates_v2` — SHIPPED.
- `report_templates` — SHIPPED.
- `plot_recons` + `plot_branch_propagations` + `plot_branch_velocities`
  + `plot_unit_summary` — SHIPPED via shared helper
  `reconstruct_phase_dry_run_short_circuit` in `stages/reconstruct/runner.py`.
- `report_recons` + `report_recon_grid` + `report_full_chip_layout`
  + `report_summaries` — SHIPPED via same helper.
- `clear_templates_cache` — SHIPPED. Phase short-circuits before the
  rmtree; reports the cache dir as a would-be-removed output. Phase
  also refactored to resolve `core.clear_templates_cache.run_clear_templates_cache_phase`
  via submodule attribute lookup at call time (instead of import-time
  binding) so monkeypatch-on-core is observed.

**Remaining**: 3 phases (`resolve_sources` + the two
slated-for-deletion phases below).

- `resolve_sources` — TODO
- ~~`analyzers`~~ — SHIPPED
- `extract_partial_templates` (slated for deletion by
  `kssynth_recon_integration` slice 5 — skip dry-run for it; it's
  going away)
- `build_templates` (same — slated for deletion by kssynth slice 5)
- ~~`kssynth`~~ — SHIPPED (the replacement phase)
- ~~`plot_templates_v2`~~ — SHIPPED
- ~~`report_templates`~~ — SHIPPED
- ~~`axon_velocity_gtrs`~~ — SHIPPED
- ~~`plot_recons`~~ — SHIPPED (via shared helper)
- ~~`plot_branch_propagations`~~ — SHIPPED (via shared helper)
- ~~`plot_branch_velocities`~~ — SHIPPED (via shared helper)
- ~~`plot_unit_summary`~~ — SHIPPED (via shared helper)
- ~~`report_recons`~~ — SHIPPED (via shared helper)
- ~~`report_recon_grid`~~ — SHIPPED (via shared helper)
- ~~`report_full_chip_layout`~~ — SHIPPED (via shared helper)
- ~~`report_summaries`~~ — SHIPPED (via shared helper)
- ~~`clear_templates_cache`~~ — SHIPPED

This is the biggest slice. Consider sub-slices grouped by sub-domain (analyzers/templates, plots, reports).

### Slice 6 — Analysis stage phase short-circuits

Progress:
- ~~`compute_metrics`~~ — SHIPPED 2026-05-21 (dry-run intercept in
  `run_analysis_compute_metrics_stage` writes a dry_run_ok manifest
  without scanning recon_outputs/units).
- `propagation_video` — ALREADY HAS dry-run via the old
  `stage_config.dry_run` field (different mechanism; pre-dates the
  process-wide override). Could be retrofitted to also honor
  `get_dry_run_override()` for consistency; not blocking.
- `unitmatch` (when it lands per `unitmatch_phase_plan.md`) — TODO

### Slice 7 — New stages from the phase roster cleanup
Once `init` and `cleanup` stages exist:
- `init.copy_src_to_scratch`
- `cleanup.wipe_src_scratch`

These get dry-run as part of their respective creation commits in `phase_roster_cleanup_plan.md` slices 4-6 — fold the dry-run short-circuit into the move-the-phase commit so they ship with dry-run from day one.

### Slice 8 — Integration test
- New test: `tests/test_dry_run_universal.py` enumerates every phase in every stage's default `phase_sequence`, runs `axon-recon stages <stage>.<phase> --dry-run --config <fixture>`, asserts return code 0 + summary JSON exists with `status: dry_run_ok`. This is the regression gate for any future phase added without dry-run support.

### Slice 9 — Documentation
- Update `dev/notes/guardrails/dry_run.md` to remove the "Dry-run does not exist today" caveat in §"Open exceptions / follow-ups". Replace with "Universal across all phases as of <commit-hash>; new phases must include the short-circuit per slice-8 enforcement test."
- Update `dev/notes/memory/current_state.md` noting dry-run is live.

Total estimated touch: ~600-1000 LoC + ~400 LoC tests across ~20 commits (one per phase + helpers + integration). Heavily mechanical once the helper + pattern are established. Each phase commit is 5-15 LoC of source + 20-40 LoC of test.

---

## 4. Tests / verification

Per-phase tests in their respective stage's test dir. The universal integration test (slice 8) is the regression gate.

Smoke verification after slice 8: run `axon-recon stages preprocess --dry-run --config dev/debug_NERSC/debug.runtime.yml --targets 13:0` and confirm:
- Return code is 0
- Each phase's `<phase>_summary.json` lands with `status: dry_run_ok`
- Total runtime is single-digit seconds (NOT minutes — if it's minutes, a phase is doing too much work in dry-run mode)

Repeat for spikesort, reconstruct, analysis on the same target.

---

## 5. Ordering with other plans

- **Independent of**: `ks_synthesizer_package_plan.md` (kssynth package work), `unitmatch_runner_package_plan.md` (unitlink package work). These plans introduce new phases that ALSO need dry-run, but they can include the dry-run short-circuit as part of their own slices using this plan's `write_dry_run_summary` helper.
- **Coordinated with**: `phase_roster_cleanup_plan.md`. Either:
  - Run AFTER phase cleanup (cleanest — don't add dry-run to phases about to be deleted), OR
  - Run IN PARALLEL for phases that are clearly staying.
  Pragmatic: do this plan's slice 1 (CLI flag + plumbing) and slice 2 (helper) right away — they're zero-risk; then sequence per-stage slices after the corresponding phase-cleanup-plan slices for that stage land.
- **Unblocks**: future smoke-test workflow. Once dry-run is universal, the smoke-test scoping ladder in `CLAUDE.md` always has step 1 (`--dry-run`) available without exception.

---

## 6. Open questions

1. **Does `--dry-run` skip resource-gate acquisition?** Probably yes — gate acquisition is part of "starting the phase"; dry-run short-circuits before that. But if the gate's contention is itself a wiring concern, maybe we WANT dry-run to acquire-then-release the gate to verify the slot demands are satisfiable. Lean toward "skip the gate entirely" for v1 — the gate's behavior is verified by the resource-gate's own tests, not by phase dry-runs.

2. **Where does the `dry_run` attribute live on `stage_config`?** All existing overrides flow through the stage_config dataclass; dry-run follows the same pattern. The attribute is `bool` (not `bool | None`) — default `False`, set `True` when override is active.

3. **What about `--dry-run --force-restart`?** Per `guardrails/dry_run.md` §sub-rule 7: dry-run lists what would be wiped in the summary's `outputs_would_produce`, but does NOT actually rmtree. The combination is useful for "what's about to get nuked" inspection before a real `--force-restart` run.

4. **Multi-rank dry-run under MPI backend?** Each rank does its own dry-run short-circuit independently; only rank 0 writes the summary JSON (matching the existing rank-0-only summary writer guard in `logging/summary.py`). Verify this works during the spikesort-stage slice (slice 4) since that's the first one where MPI matters.
