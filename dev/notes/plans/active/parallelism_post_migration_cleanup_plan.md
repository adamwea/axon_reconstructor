# Parallelism Migration — Post-Slice-11 Cleanup Plan

Status: implementation plan. Sibling to `parallelism_migration_plan.md` (slices 1–11 landed) and `spikesort_label_merge_repair_plan.md`. Same operating contract: one slice at a time, `claude:` commit prefix, append a line to `debug/commit_log.md` after every commit.

This plan does **not** introduce any new behavior. It harvests the dead/stale residue left behind after the parallelism migration plan finished landing, and bring the repo's vocabulary in line with the guardrails doc. Order it before the spikesort plan (touches some of the same files: `pipeline/runner.py`, `pipeline/config.py`, spikesort `runner.py`).

Baseline (HEAD of `claude-migration` after the post-merge fix-ups):
- `83ab6f5` docs: backfill commit_notes entry for the cpus_per_task cleanup
- `32f76cb` claude: collapse per-phase cpu_cores onto cpus_per_task; remove dead worker-count helpers
- `82ed42c` claude: fix logger context collision in phase_parallelism event
- `371f702` claude: lock new resources schema, remove back-compat parser (slice 11)

`pipeline/tests/` runs clean: 457 passed. `stages/` has 14 pre-existing failures (stale fixtures using flat profile schema, plus an unrelated `replace()` bug in `reconstruct/runner.py:1509`); these are out of scope here but tagged in §6.

---

## 0. Why This Plan Exists

The migration plan's §0 end state included this line (vocabulary contract):

> The end state is a guardrails doc whose vocabulary matches the new model exactly: `nested_shape`, `phase_budgets`, `profiles[<name>]`, `task_slot.cpu_count`, **no `well_workers` / `max_stage_workers` / `divide_stage_workers_by_wells`**.

The runtime model now matches that contract. The **code** does not, in three areas:

1. The legacy `resolve_stage_parallelism` / `StageParallelism` plumbing is still wired into `pipeline/runner.py`, complete with a defensive try/except cascade that swallows old-shape signature errors. It computes `unit_workers = stage_workers // well_workers` — the exact `divide_stage_workers_by_wells` pattern the plan banned.
2. Several reconstruct phases still pull worker count from `inputs.n_jobs` directly instead of going through `resolve_inner_worker_count` (this is what slice 5 was supposed to finish).
3. A pile of legacy nouns remain: `phase_resource_classes` parameter name (the YAML key was renamed to `phase_budgets` in slice 3), `_legacy_keyed_resource_limits`, `legacy_runner.py`, the `inputs.n_jobs` source label in the budget manager.

Cleaning these is mechanical, but it's load-bearing for future readers — every legacy term is one extra cognitive step when reading code, and the defensive try/except cascade actively misleads agents about which call shape is canonical.

---

## 1. Inventory (Read These First)

### 1.1 Forbidden-vocabulary code residue

| File:Line | What it is | Verdict |
|---|---|---|
| `pipeline/config.py:329-386` | `resolve_stage_parallelism` returns `StageParallelism(well_workers, unit_workers, …)`; computes `derived_unit_workers = max(1, int(cpu_demand or max(1, stage_workers // well_workers)))` (line 374) | DELETE the function or reduce to a thin shim that returns only `unit_workers` |
| `pipeline/config.py:334` | parameter `phase_resource_classes: list[str]` | RENAME to `phase_budget_classes` if the function survives, else delete |
| `pipeline/runner.py:280-313` | `_resolve_runtime_stage_parallelism` defensive try/except cascade catching `phase_resource_classes`/`target_count` shape errors | DELETE the fallback branches; keep one canonical call |
| `pipeline/runner.py:283` | param `phase_resource_classes` mirrored in caller | rename per above |
| `execution/context.py:24-32` | `StageParallelism` dataclass with `well_workers`, `unit_workers`, `task_allocation_plan`, etc. | KEEP only if `_attach_task_allocation_plan` still needs the bundle; otherwise replace with a slimmer record (or fold into `TaskAllocationPlan`) |
| `pipeline/resource_budget.py:331,336` | source label `"inputs.n_jobs"` returned by `current_phase_worker_allocation` fallback | RELABEL to `"slot.cpu_count"` (the actual fallback source today) |
| `pipeline/resource_budget.py:67`, `resources.py:127,524,527` | `_legacy_keyed_resource_limits` field/path | VERIFY no parser path still populates it; if dead, delete the field, the two `get_keyed_resource_limit_config` branches, and the resource_budget guard |
| `stages/spikesort/legacy_runner.py` (700 lines) | name says legacy, two active importers (`stages/reconstruct/templates/core/unit_labels.py:8`, `stages/spikesort/runner.py:20`) | INVESTIGATE — either rename to `runner_helpers.py` (it's not legacy, just misnamed) or carve out the truly stale parts |

### 1.2 Stale `inputs.n_jobs` reads (slice-5 incomplete)

These should route through `resolve_inner_worker_count(phase_cpus_per_task=getattr(_budget, "cpus_per_task", None), …)` per the plan §1.D. They currently read `inputs.n_jobs` directly:

| File:Line | Phase |
|---|---|
| `stages/reconstruct/phases/plot_templates_v2.py:90` | plot_templates_v2 |
| `stages/reconstruct/phases/plot_templates.py:137` | plot_templates |
| `stages/reconstruct/phases/generate_gtrs.py:85` | generate_gtrs |
| `stages/reconstruct/templates/runner.py:940,1953,2071,2165` | various templates phases |
| `stages/reconstruct/runner.py:851` | reconstruct stage thread pool |
| `stages/spikesort/runner.py:10627,10708` | spikesort run kwargs |
| `stages/spikesort/legacy_runner.py:340-568` (multiple) | spikesort legacy runner |
| `stages/preprocess/runner.py:1039,3530` | preprocess phase telemetry |
| `stages/spikesort/core/local_spikeinterface.py:341` | sort job kwargs |

Some of these are legitimately "pass `inputs.n_jobs` to an SI kwargs dict" rather than "decide my fanout from `inputs.n_jobs`" — those stay. Each line above needs a one-line classification before being touched.

### 1.3 YAML residue in `debug/debug.runtime.yml`

Quick audit (current state):
- `phase_budgets` block uses canonical schema (`nested_shape`, `cpus_per_task`, slot-typed gates). ✓
- Profile capacity uses `cpu_cores`, `ram_gb`, etc. ✓
- `resources.profiles.<name>.task_allocation.ram_gb_per_task: null` (lines 51, 76 area) — the field is still parsed; verify it's still consulted, otherwise drop. (Slice 6 covers this.)
- The `defaults` block at `resources.defaults.chunk_duration: 1s` (line 179) — still consumed via `resources.py:672-680`. Keep.
- `debug/debug.runtime.yml:1223` — commented-out `#n_jobs: 4` under `phases.analyzers.defaults`. **HIGH stale**: banned vocabulary even as a comment. Delete in Slice 8's vocabulary sweep.
- `debug/debug.runtime.yml:322,335,393` — `print_n_jobs_used: true` toggles under `outputs:` of `prepare_raw_binaries`, `preprocess_segments`, `concat_segments`. Logging-only switch; verify a consumer still references it before keeping. **MEDIUM**.

No `cpu_cores` keys remain under `phase_budgets`. No `phase_resource_classes` block. No `well_workers`/`max_stage_workers` keys. The YAML is the cleanest of the three layers.

### 1.4 Stale model fields (banned vocabulary still on input dataclasses)

- `pipeline/stages/reconstruct/models/inputs.py:502` — `max_plotting_concurrency: int | None = None` field on `ReconstructionInputs`. Parser at `stages/reconstruct/config.py:887-894` warns and ignores YAML usage, but the field stays so legacy tests can still construct the dataclass. **HIGH stale**: delete the field, delete the parser warn-and-ignore branch, delete the test that asserts on it (`test_run_reconstruct_generate_gtrs_phase_ignores_max_plotting_concurrency` is one of the 8 stage-test failures — see §1.6 C2).
- `pipeline/stages/preprocess/models/inputs.py:307` — `runtime_well_workers: int | None = None`. Slice 1.B already covers deletion.

### 1.5 Dual-format parallelism logging

The new `phase_parallelism` event from migration-plan Slice 10 (emitted by `cpu_allocation.py:185`) co-exists with old log lines that still print `well_workers=… n_jobs=… n_jobs_source=…`. Cleanup is folded into cleanup-plan Slice 1 (most sites) and cleanup-plan Slice 10 (final allowlist drop):

| File:Line | Old format still emitted |
|---|---|
| `pipeline/runner.py:261, 752, 808, 844-848, 910-913, 2173, 2839, 2842` | preprocess/spikesort `parallelism.well_workers`/`unit_workers` log lines |
| `pipeline/runner.py:914-919` | `n_jobs_source` literals `"derived"`, `"configured"`, `"serial"` (canonical: `slot.cpu_count` / `phase_cap` / `yaml_n_jobs_override`) |
| `pipeline/resource_budget.py:235-245` | phase-gate warning emits `well_workers=%d` |
| `stages/preprocess/runner.py:1031-1041` | `Preprocess phase worker allocation … well_workers=%d n_jobs=%d n_jobs_source=%s phase_n_jobs=%d` |
| `pipeline/logging/formatters.py:69` | `well_workers` in JSON allowlist (drop after the emit sites are dead) |

### 1.6 Pre-existing stage-test failures (14 total — broader-sweep classification)

Grouped by root cause and by which slice fixes each. Categories C1–C4 came from the focused reconstruct/preprocess sweep; the templates subtree adds another 6 to the picture once you run the broader test sweep.

**Resolved by Slice 7 (replace() guard + max_plotting_concurrency deletion)** — 4 tests
- `test_reconstruct_combined_phase_sequence_runs_in_order` — C2
- `test_reconstruct_combined_phase_sequence_skips_clear_templates_cache_when_disabled` — C2
- `test_reconstruct_configured_copied_template_phase_sequence_runs_requested_order` — C2
- `test_run_reconstruct_generate_gtrs_phase_ignores_max_plotting_concurrency` — C2 + dead field. **Test deleted with the field.**

**Resolved by Slice 8 (stage test fixture migration)** — 5 tests
- `stages/reconstruct/tests/test_runner.py::test_reconstruct_phase_worker_allocation_uses_resource_class_cpu_for_downstream_phases` — C1: flat profile fixture; also retarget `"resource_class.cpu_cores"` → `"resource_class.cpus_per_task"` source string.
- `stages/preprocess/tests/test_runner.py::test_run_preprocess_stage_logs_phase_start_per_well:843-847` — C3: asserts `phase_n_jobs=12`; runtime now clamps to 1. Update assertion (or delete if Slice 1 retired the log line).
- `stages/reconstruct/tests/test_config.py::test_load_config_reconstruct_populates_templates_inputs_from_debug_runtime` — C4: asserts `v2.dpi == 220.0` against live `debug.runtime.yml:1537` which now ships `420`. Rewrite to use a pinned fixture YAML instead of the live runtime.
- `stages/reconstruct/tests/test_runner.py::test_run_reconstruct_report_full_chip_layout_phase_writes_outputs:2013` — C4 follow-up: asserts `units_successful == 2`, gets 1. Verify this clears after Slice 7 lands; if not, fix in Slice 8.
- `stages/reconstruct/templates/tests/test_runner.py::test_run_reconstruct_templates_build_templates_phase_uses_unit_workers` — phase-sequence fixture drift; needs `nested_shape: unit_workers` vocabulary.
- `stages/reconstruct/templates/tests/test_config.py::test_load_templates_config_parses_plot_templates_v2_phase_block` — YAML parser test for the v2 block; either parser changed shape or fixture is stale.

**Resolved by Slice 9 (spikeinterface_extract compat audit)** — 4 tests
- `stages/reconstruct/templates/tests/test_spikeinterface_extract.py::test_build_unit_source_payload_expands_to_all_waveforms_when_unlimited`
- `stages/reconstruct/templates/tests/test_spikeinterface_extract.py::test_build_unit_source_payload_forwards_waveform_window_on_recompute`
- `stages/reconstruct/templates/tests/test_spikeinterface_extract.py::test_build_unit_source_payload_forwards_random_spikes_policy_on_recompute`
- `stages/reconstruct/templates/tests/test_spikeinterface_extract.py::test_build_unit_source_payload_retries_without_compat_only_kwargs`

These all exercise the `compat_only_kwargs` retry path against `SortingAnalyzer.compute(...)`. Likely root cause: spikeinterface upstream API drift; production retry guard needs an audit. **Don't delete** — `build_unit_source_payload` is load-bearing for analyzers and extract_partial_templates.

After Slices 7–9, all 14 should be green. Verify with `pytest src/axon_recon/pipeline/stages/ -q`.

---

## 2. Discovery Targets

```text
src/axon_recon/pipeline/config.py                        # resolve_stage_parallelism
src/axon_recon/pipeline/runner.py                        # _resolve_runtime_stage_parallelism + try/except cascade
src/axon_recon/pipeline/execution/context.py             # StageParallelism dataclass
src/axon_recon/pipeline/resource_budget.py               # current_phase_worker_allocation, _legacy_keyed_resource_limits guard
src/axon_recon/pipeline/resources.py                     # _legacy_keyed_resource_limits field + access helpers
src/axon_recon/pipeline/stages/spikesort/legacy_runner.py
src/axon_recon/pipeline/stages/reconstruct/phases/{plot_templates,plot_templates_v2,generate_gtrs}.py
src/axon_recon/pipeline/stages/reconstruct/templates/runner.py
src/axon_recon/pipeline/stages/preprocess/models/inputs.py    # runtime_well_workers field
debug/guardrails/parallelism_agent_guardrails.md                    # contract doc to update alongside slices
debug/debug.runtime.yml                                  # YAML residue
```

Run between slices:
```bash
conda run -n axon_recon python -m pytest src/axon_recon/pipeline/tests/ -q --ignore=src/axon_recon/pipeline/tests/test_progress.py
```
Baseline target: **457 passed**, no new failures.

---

## 3. Slices

### Slice 1 — Retire `resolve_stage_parallelism` and `StageParallelism.well_workers`

**Goal**: kill the legacy stage-parallelism plumbing. The runtime now derives parallel concurrency from the live `ResourceBudgetManager` gate, not from a precomputed `well_workers`. Anything that consumes `parallelism.well_workers` either reads it for telemetry (move to budget manager) or for execution decisions (replace with a budget-manager call).

**A. Inventory** (run before editing):
```bash
git grep -nE "well_workers\b|StageParallelism\(|resolve_stage_parallelism\(|\.well_workers\b" src/ debug/ docs/
```
Classify each hit as: `delete`, `replace-with-budget-manager`, `keep-as-runtime-metric`.

**B. Edits**:
- `pipeline/config.py:329-386` — delete `resolve_stage_parallelism` entirely. If something needs `unit_workers` for fallback, expose a one-liner `default_inner_workers(stage_config)` that returns 1.
- `pipeline/runner.py:280-313` — delete `_resolve_runtime_stage_parallelism` (or simplify to just `_attach_task_allocation_plan`). Remove the three-branch try/except.
- `pipeline/runner.py:248-274, 752, 808, 844-848, 910-913, 2173, 2839, 2842` — strip log lines that emit `parallelism.well_workers`/`unit_workers`. Migration-plan Slice 10's `phase_parallelism` event already covers the structured telemetry; these are dead duplicates.
- `pipeline/runner.py:914-919` — drop the `n_jobs_source` literal mapping (`"derived"`/`"configured"`/`"serial"`); replace with the canonical `slot.cpu_count` / `phase_cap` / `yaml_n_jobs_override` source from `current_phase_worker_allocation` (see Slice 5).
- `execution/context.py:24-32` — drop `well_workers` and `unit_workers` from `StageParallelism`. If only `task_allocation_plan` survives, fold into `TaskAllocationPlan`.
- `pipeline/resources.py:643` — delete `get_max_phase_resource_demands` if no caller remains after the above.
- `pipeline/resource_budget.py:235-245` — phase-gate warning still prints `well_workers=%d`; rename log key to `task_slot_count` (keep the value; it really is the rank's well_workers, but that's a budget-manager concept now, not a stage parallelism concept).
- `pipeline/logging/formatters.py:69` — remove `well_workers` from the JSON allowlist after the emit sites above are dead.
- `stages/preprocess/models/inputs.py:307` — remove `runtime_well_workers: int | None = None` field.
- `stages/preprocess/runner.py:1031-1041, 2892` — drop the `runtime_well_workers` plumbing AND the old-format `Preprocess phase worker allocation … well_workers=%d n_jobs=%d n_jobs_source=%s phase_n_jobs=%d` log line; the new `phase_parallelism` event already covers it.

**C. Tests**:
- Update or delete the `~10` test fixtures in `pipeline/tests/test_spikesort_target_status.py` that monkeypatch `resolve_stage_parallelism` and instantiate `StageParallelism(well_workers=…)`. Where the assertion is "the stage ran with N parallel wells", replace with "the budget manager ran with planned_target_count=N".
- Drop `test_resolve_stage_parallelism_*` tests if they exist.

**Acceptance**:
- `git grep -nE "resolve_stage_parallelism\(|StageParallelism\(.*well_workers"` returns 0 hits in non-test code.
- `pipeline/tests/` still 457 passed.
- A targeted grep of the migration's banned vocabulary is clean:
  ```bash
  git grep -nE "max_stage_workers|divide_stage_workers_by_wells" src/
  ```

**Commit**: `claude: retire resolve_stage_parallelism and StageParallelism.well_workers (cleanup slice 1)`

---

### Slice 2 — Route remaining `inputs.n_jobs` reads through `resolve_inner_worker_count`

**Goal**: finish what slice 5 of the original migration started — every fanout decision goes through the budget-aware helper.

**A. Inventory** (already done in §1.2). For each line:
- If the call is **passing `n_jobs` to a third-party SI kwargs dict** (e.g. `job_kwargs["n_jobs"] = inputs.n_jobs`), keep — that's an external-library boundary.
- If the call is **deciding "how many threads should I spawn"** (`derived_unit_workers = max(1, int(inputs.n_jobs))`, `ThreadPoolExecutor(max_workers=…)`), replace with:
  ```python
  _budget = current_phase_budget("<stage>", "<phase>")
  workers = resolve_inner_worker_count(
      nested_shape=str(getattr(_budget, "nested_shape", "<canonical>") or "<canonical>"),
      phase_cpus_per_task=getattr(_budget, "cpus_per_task", None) if _budget else None,
      yaml_n_jobs_override=int(inputs.n_jobs) if getattr(inputs, "n_jobs", None) is not None else None,
      work_item_count=<len(items) if known>,
      stage_name="<stage>",
      phase_name="<phase>",
  )
  ```
  Use the canonical `nested_shape` per the plan's §1.D table.

**B. Files to touch** (one edit per phase):
- `stages/reconstruct/phases/plot_templates.py:137` (nested_shape: unit_workers, clamp 4)
- `stages/reconstruct/phases/plot_templates_v2.py:90` (unit_workers, clamp 4)
- `stages/reconstruct/phases/generate_gtrs.py:85` (unit_workers)
- `stages/reconstruct/templates/runner.py:940,1953,2071,2165` (mostly templates phase fanouts)
- `stages/reconstruct/runner.py:851` (downstream reconstruct thread pool)
- Spikesort `inputs.n_jobs` reads at `runner.py:10627,10708` and `core/local_spikeinterface.py:341` — these look like SI kwargs forwarding; classify before changing.
- Preprocess telemetry sites (`runner.py:1039,3530`) — these are log-only; replace with the resolved `workers` value or drop.

**C. Acceptance**:
- `git grep -nE "int\(inputs\.n_jobs\)" src/axon_recon/` should return only the SI-kwargs forwarding sites you classified as "keep".
- `pipeline/tests/` still 457 passed.
- Smoke (must run in container):
  ```bash
  axon-recon-container --no-build --gpus all stages reconstruct.analyzers \
    --config debug/debug.runtime.yml --target-dataset 11,12 --limit-wells 1 --limit-segments 2
  ```
  Logs should show `phase_parallelism event=…` lines for analyzers, plot_templates, generate_gtrs, and the worker counts should match what the table predicts (analyzers: slot.cpu_count = 10; plot_templates: 4 because clamp).

**Commit**: `claude: route remaining inputs.n_jobs reads through resolve_inner_worker_count (cleanup slice 2)`

---

### Slice 3 — Delete `_legacy_keyed_resource_limits` if dead

**Goal**: confirm/delete the legacy-YAML compatibility shim for `keyed_resource_limits` declared at the top level (outside `profiles.<name>`).

**A. Verify it's dead**:
- Slice 11 (`371f702`) locked the resources schema and rejects flat profiles with a clear error. Check whether the parser ever populates `_legacy_keyed_resource_limits` after that change. Read `_parse_resources_config` in `resources.py` and confirm there is no path that writes to that field.
- If a path exists: trace it. It might be pre-`profiles` YAMLs from external configs.

**B. If dead, delete**:
- `resources.py:127` — `_legacy_keyed_resource_limits` field on `ResourcesConfig`.
- `resources.py:524,527` — the two `_legacy_keyed_resource_limits.get(resolved, None)` branches in `get_keyed_resource_limit_config`.
- `resource_budget.py:67` — the `_keyed_limits = self.resources._legacy_keyed_resource_limits` fallback.
- Any tests that exercise the legacy shape.

**C. If still alive (some external config relies on flat keyed_resource_limits)**: rename the field to drop the `_legacy_` prefix (`top_level_keyed_resource_limits`) and document why it remains.

**Acceptance**:
- `git grep -nE "_legacy_keyed_resource_limits" src/` returns 0 hits (or the rename is clean).
- `pipeline/tests/` still 457 passed.

**Commit**: `claude: drop _legacy_keyed_resource_limits shim (cleanup slice 3)` — title varies if rename instead of delete.

---

### Slice 4 — Audit `legacy_runner.py` and either rename or carve

**Goal**: the file at `stages/spikesort/legacy_runner.py` is 700 lines and has two active importers (`reconstruct/templates/core/unit_labels.py`, `spikesort/runner.py`). The "legacy" name implies dead code, but it's not. This slice clarifies the picture.

**A. Inventory** what each importer actually consumes:
```bash
grep -nE "from .*spikesort.legacy_runner import" -A 5 src/axon_recon/pipeline/stages/reconstruct/templates/core/unit_labels.py
grep -nE "from .*spikesort.legacy_runner import" -A 5 src/axon_recon/pipeline/stages/spikesort/runner.py
```
List the imported symbols. For each, decide:
- Active production helper that happens to live in a misnamed file → **rename** the file.
- Truly stale (no caller, kept for historical reasons) → **delete**.
- Mixed → **split**: move active helpers to `runner.py` or a new `runner_helpers.py`, delete the rest.

**B. Execute the chosen path**.

**C. Acceptance**:
- `git grep -nE "from .*legacy_runner|import legacy_runner"` returns 0 hits (file is gone or renamed).
- Spikesort suite passes:
  ```bash
  conda run -n axon_recon python -m pytest src/axon_recon/pipeline/stages/spikesort/tests/ -q
  ```

**Commit**: `claude: clarify or delete spikesort legacy_runner.py (cleanup slice 4)`

---

### Slice 5 — Relabel and tighten `current_phase_worker_allocation`

**Goal**: the function returns `(workers, source)` where `source` is `"inputs.n_jobs"` for the no-budget-manager fallback. After Slice 2, fewer phases will hit that branch — but the label is also wrong: the actual fallback path returns `fallback_workers`, which callers typically derive from `slot.cpu_count`, not from `inputs.n_jobs`.

**A. Edits** (`pipeline/resource_budget.py:324-336`):
- Relabel `"inputs.n_jobs"` → `"slot.cpu_count"` (or `"fallback"` if the caller controls the fallback source).
- Update callers' log lines that switch on the source string (`stages/reconstruct/templates/runner.py:940` area, `stages/reconstruct/runner.py:1495`).

**B. Acceptance**:
- `git grep -nE "\"inputs.n_jobs\"|inputs\.n_jobs source" src/` returns 0 hits.

**Commit**: `claude: relabel current_phase_worker_allocation fallback source (cleanup slice 5)`

---

### Slice 6 — Drop unused `task_allocation.ram_gb_per_task` / `shm_gb_per_task` fields

**Status**: SHIPPED 2026-05-19 via **path C** (consumer exists; documented).
`cpu_allocation.py:670, 674` does consume both fields via `_capacity_limit_from_float`,
gating `effective_task_limit` per node alongside CPU capacity (`cpu_allocation.py:682-685`).
Documented in `guardrails/parallelism.md` "Open exceptions / follow-ups" with cross-refs
to the field defs, parser, and consumer tests. Fields and YAML keys remain. No code change.

**Goal**: `TaskAllocationConfig` carries `ram_gb_per_task` and `shm_gb_per_task` fields (`resources.py:102-103`). YAML sets them to `null`. Verify whether any code reads them; if not, drop.

**A. Verify**:
```bash
git grep -nE "ram_gb_per_task|shm_gb_per_task" src/axon_recon/
```
Classify each: parser write, YAML config doc, actual consumer.

**B. If only writer/parser, no consumer**: delete the fields, their parsing, and the YAML keys.

**C. If consumer exists** but always sees `None` in current YAML: leave alone, document the consumer in `parallelism_agent_guardrails.md` so it's discoverable.

**Acceptance**:
- Either: `ram_gb_per_task` removed everywhere except the parser-rejection error message that still names it as a valid alternative.
- Or: a guardrails note explains where it's read and why it's optional.

**Commit**: `claude: drop unused task_allocation per-task RAM/SHM fields (cleanup slice 6)` — skip if Slice 6.C path was taken.

---

### Slice 7 — Fix `replace()` bug at `reconstruct/runner.py:1509` and retire `max_plotting_concurrency` field

**Goal**: pick up the highest-impact stage-test failures (4 of the 8) in one focused slice. The two are coupled: `test_run_reconstruct_generate_gtrs_phase_ignores_max_plotting_concurrency` only constructs `ReconstructionInputs(max_plotting_concurrency=…)` because the field still exists on the dataclass; once the field is gone, the test goes with it.

**A. Fix the `replace()` bug** (`pipeline/stages/reconstruct/runner.py:1509`):
```python
# Before:
phase_inputs = replace(phase_inputs, templates_inputs=replace(inputs.templates_inputs, n_jobs=int(workers)))
# After:
templates_inputs = inputs.templates_inputs
if templates_inputs is not None and is_dataclass(templates_inputs):
    templates_inputs = replace(templates_inputs, n_jobs=int(workers))
phase_inputs = replace(phase_inputs, templates_inputs=templates_inputs)
```
Add `from dataclasses import is_dataclass` to the imports if missing. This unblocks 4 stage-test failures (C2 group).

**B. Retire `max_plotting_concurrency`**:
- Delete `max_plotting_concurrency: int | None = None` from `pipeline/stages/reconstruct/models/inputs.py:502`.
- Delete the parser warn-and-ignore branch at `stages/reconstruct/config.py:887-894`.
- Delete `test_run_reconstruct_generate_gtrs_phase_ignores_max_plotting_concurrency` (the only behavior it tested is "the field is ignored" — once the field is gone, the test is moot).
- Grep the rest of the codebase for `max_plotting_concurrency` to confirm zero hits.

**C. Acceptance**:
- `git grep -nE "max_plotting_concurrency" src/` returns 0 hits.
- `git grep -nE "replace\(inputs\.templates_inputs" src/` returns 0 hits (or only the guarded form above).
- The 4 C2 tests now pass; the C2-tagged generate_gtrs test is deleted (no longer applicable).
- `pipeline/tests/` still 457 passed.

**Commit**: `claude: guard reconstruct templates_inputs replace; drop max_plotting_concurrency field (cleanup slice 7)`

---

### Slice 8 — Stage test fixture migration (mechanical drift fixes)

**Goal**: clear the mechanical drift in stage tests so the fail list is honest about what's actually broken. After Slice 7 lands, ~9 stage-test failures remain; this slice fixes the ones that are pure fixture/assertion drift (no production-code change needed).

**A. Schema-nest fix (1 test, C1)**:
- `stages/reconstruct/tests/test_runner.py::test_reconstruct_phase_worker_allocation_uses_resource_class_cpu_for_downstream_phases` — change the inline YAML fixture so `profiles.<name>.cpu_cores: N` becomes `profiles.<name>.capacity.cpu_cores: N`. Also retarget the source-string assertions: `"resource_class.cpu_cores"` → `"resource_class.cpus_per_task"` (matches `current_phase_worker_allocation`'s relabel from `32f76cb`).

**B. Assertion drift (1 test, C3)**:
- `stages/preprocess/tests/test_runner.py::test_run_preprocess_stage_logs_phase_start_per_well` — assertion `phase_n_jobs=12` is stale; runtime correctly clamps to 1 because `slot_cpus=1` in the test env. Update to `phase_n_jobs=1`.
  - **Note**: if cleanup plan Slice 1 retires the old `Preprocess phase worker allocation … well_workers=… n_jobs=… n_jobs_source=… phase_n_jobs=…` log line entirely, this test should be deleted instead. Land Slice 1 first; if the log line is gone, drop this test in Slice 8 instead of patching the assertion.

**C. Live-config dependency (1 test, C4)**:
- `stages/reconstruct/tests/test_config.py::test_load_config_reconstruct_populates_templates_inputs_from_debug_runtime` — currently asserts `v2.dpi == 220.0` against `debug/debug.runtime.yml:1537` which now ships `dpi: 420`. Don't just bump the assertion to 420 — that re-rots the next time anyone tunes the live config. Rewrite the test to either (a) load a stable fixture YAML pinned in the test directory with known values, or (b) bind the dpi-comparison value via `RuntimeConfig.load(debug.runtime.yml).get_templates_dpi()` so the test verifies plumbing, not a specific number.

**D. C4 follow-up (1 test, verify after Slice 7)**:
- `stages/reconstruct/tests/test_runner.py::test_run_reconstruct_report_full_chip_layout_phase_writes_outputs:2013` — asserts `units_successful == 2`, gets 1. Probably the same `replace()` root cause as C2; re-run after Slice 7 lands. If still red, then it's a separate worker-clamp side effect — investigate and fix in this slice. If it passes after Slice 7, no action needed.

**E. Vocabulary/fixture drift (2 tests in templates/tests)**:
- `stages/reconstruct/templates/tests/test_runner.py::test_run_reconstruct_templates_build_templates_phase_uses_unit_workers` — likely needs the test's phase-sequence fixture updated to the new `nested_shape: unit_workers` vocabulary. Read the test, identify what assumption broke, fix.
- `stages/reconstruct/templates/tests/test_config.py::test_load_templates_config_parses_plot_templates_v2_phase_block` — YAML parser test for the v2 block. Either the parser changed shape post-migration or the fixture is stale; one read of each should make it obvious.

**Acceptance**:
- After Slice 7 + Slice 8, `pytest src/axon_recon/pipeline/stages/ -q` shows at most the 4 spikeinterface_extract failures (handled in Slice 9).
- `pipeline/tests/` still 457 passed.
- Each fixture change is a one-test commit unit (don't bundle); 5–6 small commits or one commit if all share the same shape — author's choice.

**Commit**: `claude: migrate stale stage test fixtures to post-migration schema (cleanup slice 8)`

---

### Slice 9 — Templates `spikeinterface_extract` compat audit

**Goal**: the four `test_build_unit_source_payload_*` failures in `stages/reconstruct/templates/tests/test_spikeinterface_extract.py` exercise the `compat_only_kwargs` retry path that guards against version drift in `spikeinterface.SortingAnalyzer.compute(...)`. They've been failing since before slice 11. Don't delete — the `build_unit_source_payload` path is load-bearing for analyzers and extract_partial_templates. This slice diagnoses and fixes.

**A. Inventory the 4 tests**:
- `test_build_unit_source_payload_expands_to_all_waveforms_when_unlimited`
- `test_build_unit_source_payload_forwards_waveform_window_on_recompute`
- `test_build_unit_source_payload_forwards_random_spikes_policy_on_recompute`
- `test_build_unit_source_payload_retries_without_compat_only_kwargs`

**B. Diagnose**:
```bash
conda run -n axon_recon python -m pytest \
  src/axon_recon/pipeline/stages/reconstruct/templates/tests/test_spikeinterface_extract.py -q --tb=short
conda run -n axon_recon python -c "import spikeinterface; print(spikeinterface.__version__)"
conda run -n axon_recon python -c "from spikeinterface.core.sortinganalyzer import SortingAnalyzer; help(SortingAnalyzer.compute)" | head -40
```
Identify which kwarg the production retry path branches on and what spikeinterface accepts now. Possible root causes:

- Fixture stale — bump expected kwargs to match current SI signature.
- Real compat path is broken — the `compat_only_kwargs` retry guards against a SI version we no longer support; reduce or update the guard set in `templates/core/spikeinterface_extract.py` (or wherever the retry lives).
- SI changed its API mid-release and the production code needs an update too.

**C. Fix**:
- If fixture-only: update test fixtures + assertions, no production change.
- If production path needs to evolve: pinch out the dead branch (the SI version it guarded against is no longer supported) and simplify, or update the retry to match new SI shape.

**D. Acceptance**:
- All 4 `test_build_unit_source_payload_*` tests pass.
- A targeted real-data smoke (analyzer build + extract_partial_templates) confirms no regression:
  ```bash
  axon-recon-container --no-build --gpus all stages reconstruct.analyzers reconstruct.extract_partial_templates \
    --config debug/debug.runtime.yml --target-dataset 11,12 --limit-wells 1 --limit-segments 2
  ```

**Commit**: `claude: audit and repair spikeinterface_extract compat path (cleanup slice 9)`

---

### Slice 10 — Update `parallelism_agent_guardrails.md`

**Goal**: lock the contract doc to the post-cleanup vocabulary. Each prior slice was supposed to update this doc inline, but several slices are pending (slices 1–4 of *this* plan introduce vocabulary changes, e.g. removing `well_workers`).

**A. Edits** in `debug/guardrails/parallelism_agent_guardrails.md`:
- Strike any remaining mention of `well_workers`, `max_stage_workers`, `divide_stage_workers_by_wells`, `resolve_stage_parallelism`, `StageParallelism`, `phase_resource_classes`, `inputs.n_jobs source`.
- Add a "Banned Vocabulary" appendix listing those terms, why each was removed, and the replacement.
- Cross-link the post-cleanup commit SHAs in the "Slices Landed" section.

**B. Acceptance**:
- The grep `git grep -nE "well_workers|max_stage_workers|divide_stage_workers|resolve_stage_parallelism" debug/guardrails/parallelism_agent_guardrails.md` returns only `## Banned Vocabulary` mentions.

**Commit**: `claude: lock guardrails doc to post-cleanup vocabulary (cleanup slice 10)`

---

## 4. Smoke Matrix (rerun after slices 1, 2, 4, 7, 9)

```bash
# A. Direct preprocess via mpirun (already passing post-82ed42c)
bash debug/mpirun.sh

# B. Bootstrap-only spikesort (cheapest container exercise)
axon-recon-container --no-build --gpus all stages spikesort.bootstrap_concat_binary \
  --config debug/debug.runtime.yml --target-dataset 11,12 --limit-wells 1 --limit-segments 2

# C. Sort end-to-end (requires GPU + kilosort4 in container)
axon-recon-container --no-build --gpus all stages spikesort.sort \
  --config debug/debug.runtime.yml --target-dataset 11,12 --limit-wells 1 --limit-segments 2 --force-restart

# D. Reconstruct.analyzers (validates si_njobs nested_shape after slice 2)
axon-recon-container --no-build --gpus all stages reconstruct.analyzers \
  --config debug/debug.runtime.yml --target-dataset 11,12 --limit-wells 1 --limit-segments 2

# E. Generate_gtrs (validates unit_workers nested_shape after slice 2)
axon-recon-container --no-build --gpus all stages reconstruct.generate_gtrs \
  --config debug/debug.runtime.yml --target-dataset 11,12 --limit-wells 1 --limit-segments 2
```

Verify in each: `phase_parallelism` log lines exist for every fanout site; `effective` worker count matches the §1.D table; no `KeyError` from logging or `replace()` errors.

---

## 5. Cleanup Checklist (post-Slice 10)

```bash
# (a) No banned vocabulary in non-test code
git grep -nE "well_workers|max_stage_workers|divide_stage_workers_by_wells|resolve_stage_parallelism|StageParallelism" src/axon_recon/ -- ':!*/tests/*'

# (b) No phase_resource_classes anywhere
git grep -nE "phase_resource_classes" src/ debug/

# (c) No legacy keyed-resource shim
git grep -nE "_legacy_keyed_resource_limits" src/

# (d) No legacy_runner imports
git grep -nE "legacy_runner" src/

# (e) No bare inputs.n_jobs reads except SI-kwargs forwarding sites
git grep -nE "int\(inputs\.n_jobs\)" src/axon_recon/

# (f) Guardrails doc clean
git grep -nE "well_workers|max_stage_workers|divide_stage_workers" debug/guardrails/parallelism_agent_guardrails.md
```

(a)–(d) must be empty. (e) must match only the documented forwarding sites. (f) must return only `## Banned Vocabulary` mentions.

---

## 6. Out of Scope (filed for the next planner)

- ~~**Reconstruct stage test rot**~~ — superseded by Slice 8 (mechanical fixture/assertion drift) and Slice 9 (spikeinterface_extract compat audit).
- ~~**`reconstruct/runner.py:1509` `replace()` bug**~~ — superseded by Slice 7 (the guard goes in alongside the `max_plotting_concurrency` field deletion).
- **Container + mpirun "runs double"**. Each `mpirun -np 2 axon-recon-container` spawns two independent containers each with `MPI_COMM_WORLD` of size 1. Either fix MPI passthrough (`--ipc=host`, PMIx mounts) or document that container is single-rank only and use mpirun directly with the host axon-recon for multi-rank work. Strategy options captured in `debug/guardrails/container_mpi_strategy_note.md`; recommendation is to stay on Option C (single-rank container for sort, host mpirun for everything else) until lab patterns force a change.

- **Retire `plot_templates` (v1) phase**. The `plot_templates_v2` phase replaces it; v1 is commented out of `debug/debug.runtime.yml` phase_sequence (line 1125) but the YAML block at lines 1406+, the parser, and the runner all still exist for v1. A future cleanup slice should:
  1. Confirm no other `phase_sequence` (in any sibling runtime YAML or test fixture) still references `plot_templates` (the v1 name).
  2. Delete the `phases.plot_templates:` YAML block in `debug/debug.runtime.yml`.
  3. Delete `stages/reconstruct/phases/plot_templates.py` and the parser branch in `stages/reconstruct/templates/config.py` for `plot_templates` (keep `plot_templates_v2`).
  4. Delete the `plot_templates`-specific aliases in `_RECONSTRUCT_PHASE_ALIASES` (or wherever the alias map lives) — keep only `plot_templates_v2` and its synonyms.
  5. Update tests that exercise the v1 path (`test_run_reconstruct_templates_plot_templates_phase_writes_circle_plots_only_and_cleans_stale_artifacts` is one such; there are likely more).
  6. Update guardrails doc to mention only v2.

---

## 7. Definition of Done

This cleanup is complete when:

1. The 10 slices above are merged in order, each with passing acceptance checks.
2. `pipeline/tests/` still passes 457 tests after every slice.
3. `parallelism_agent_guardrails.md` matches the post-cleanup vocabulary exactly.
4. The cleanup checklist (§5) commands all return clean.
5. The §3 smoke matrix is green on the user's lab server with the live container.

If any acceptance check fails, the slice does not land. Each slice is a single commit prefixed `claude:`. Append one line to `debug/commit_log.md` after every commit.
