# Stage / phase architecture guardrail

## Contract

Every stage in axon_recon's pipeline conforms to the following structure. Every phase inside every stage conforms to the same per-phase structure. Adding a new stage or phase means producing all the listed artifacts; removing one means cleaning all of them up. Half-wired stages or half-wired phases are bugs.

### Stage shape

A stage `<stage_name>`:
- Lives under `src/axon_recon/pipeline/stages/<stage_name>/`.
- Has a runner module that exposes `run_<stage_name>_stage(bundle, …) -> StageResult` (or equivalent — see `runner.py` in each existing stage for the canonical signature).
- Declares its phases under YAML `stages.<stage_name>.phases.<phase_name>:` and its execution order under `stages.<stage_name>.phase_sequence:`.
- Has a `resources` block under YAML `stages.<stage_name>.resources:` if it needs stage-level resource overrides (rare; most stages don't).
- Has an `output_rel_root` under YAML (e.g. `spikesort_outputs`, `recon_outputs`, `preprocess_outputs`). All phase outputs land under `<well>/<output_rel_root>/<phase_output_rel_path>/`.
- Has tests under `src/axon_recon/pipeline/stages/<stage_name>/tests/`.
- Has a `__init__.py` plus typically `runner.py`, `config.py`, `api.py`, `cli.py`, `core/`, `orchestrators/`, `models/` — match the existing stages.

### Phase shape

A phase `<phase_name>` under stage `<stage_name>`:
- Has a YAML block under `stages.<stage_name>.phases.<phase_name>:` with at least:
  ```yaml
  <phase_name>:
    enabled: true
    resource_class: <name in resources.phase_budgets>
    summary_json_relpath: context/<phase_name>_summary.json    # or similar
    # phase-specific config below
  ```
- Has a corresponding target runner function (typically `_run_<stage>_<phase>_target` in `stages/<stage>/runner.py`) with signature:
  ```python
  def _run_<stage>_<phase>_target(*, target: ExecutionTarget, stage_config: Any, unit_workers: int) -> <Stage>Result: ...
  ```
- Writes its own `summary_json` at the configured relpath. The summary at minimum has: `status` (`ok` / `error` / `skipped` / `dry_run_ok`), `well_out_dir`, `applied_debug_limits`, `inputs`, `outputs`. Phase-specific fields beyond that are free.
- Either appears in the YAML `phase_sequence` (= runs by default) or is excluded from it (= manual invocation only via `axon-recon stages <stage>.<phase>`).
- Supports `--force-restart` per `guardrails/force_restart.md`.
- Supports `--dry-run` per `guardrails/dry_run.md`.
- Respects `--target-*`, `--targets`, `--limit-*` per `guardrails/scope_flags.md` — usually automatically, by consuming the filtered ExecutionTarget list from `select_execution_targets`.

### Resource class

Every phase has a `resource_class` referencing an entry under `resources.phase_budgets.<class>` (a top-level YAML block, not per-stage). The class declares the phase's RAM, slot, and CPU demands. The resource-gate manager enforces these against the active profile's capacity.

Reusing existing classes is fine and encouraged — `disk_cleanup`, `plot_unit`, `bootstrap_concat_binary`, `kilosort4`, `spikeinterface_analyzer_concat`, etc. Add a new class only when an existing one genuinely doesn't fit. Document new classes in the YAML alongside the existing entries.

## Why

This structure is what makes the pipeline introspectable. `axon-recon status` reads phase summary_json files to compute completeness. `--force-restart` reads phase output_rel_root to know what to rmtree. The resource gate reads phase resource_class to know what slots to reserve. The status reporter's chip-well groups + cross-stage marker check works because every phase emits a marker at a predictable path.

Phases that don't conform — phases without a summary_json, phases that mutate sibling phases' output dirs, phases with no resource_class — are silently broken in one or more of these tools. The post-SLAy aux-tsv bug, the bombcell-stale-labels contamination, the spikesort-stage cleanup allowlist gaps: all of them trace to phases that drifted off the architectural contract.

## Sub-rules

1. **No silent cross-phase side effects.** A phase writes to its own output_rel_root + the canonical sorter_output (when explicitly contracted, like merge_SLAy). It does NOT write into sibling phases' output dirs.

2. **Per-target idempotence.** Running the same phase twice on the same target with the same inputs produces the same outputs (modulo file mtimes). If a phase has internal state (e.g. trained ML model cache), the cache invalidation must respect `--force-restart`.

3. **Status-reportable.** Every phase that lands in `phase_sequence` must have its summary_json picked up by `axon-recon status` — meaning the relpath under the well's output dir must be in `STAGE_PHASES` in `src/axon_recon/pipeline/status.py`.

4. **No magic side files.** If a phase produces an output the rest of the pipeline depends on, it goes in the YAML's `outputs:` block with an explicit `relpath`. Don't hardcode paths in code.

5. **Forced default state**: phases default to `enabled: true` only when they're in the default `phase_sequence`. If a phase is in code but not in the default sequence, its YAML block has `enabled: false` so a user enabling it accidentally is caught.

6. **Phase-name = directory-name = function-name root.** A phase called `bombcell_label`:
   - YAML key: `bombcell_label`
   - Output dir: `<output_rel_root>/bombcell_label_outputs/` (suffix optional, but the prefix matches the phase name)
   - Runner function: `_run_<stage>_bombcell_label_target` / `run_bombcell_label_stage`
   - Resource class: typically `bombcell_label` (or a class that's clearly its budget)
   - Tests: `test_bombcell_label*.py`

   If a phase is renamed (e.g. `generate_gtrs` → `axon_velocity_gtrs` per the phase-cleanup plan), ALL of these get renamed in the same commit. A grep audit afterwards confirms zero orphan references.

## Tests / verification

- New stages need: stage-level smoke test (phase_sequence runs end-to-end on a tiny fixture), `--force-restart` test (write garbage, restart, assert clean), `--dry-run` test (every phase short-circuits at input resolution).
- New phases need: targeted unit tests for the phase's core logic, a wiring test asserting the phase shows up in the stage's `phase_sequence` (or is correctly listed as opt-in), and a summary_json schema test.
- `axon-recon status --config <yaml>` should always produce a coherent output — no `<phase>_missing` for a phase that's been disabled, no `<phase>_stale` for a phase whose upstream artifact predates its output (per the SLAy aux-tsv fix).

## Open exceptions / follow-ups

- Several legacy phases violate this contract today (`plot_templates`, `per_unit_processing`, `reports` in reconstruct stage; the un-enabled `prepare_raw_binaries` / `report_preprocessing` / `cleanup_preprocessing_outputs` in preprocess). The `phase_roster_cleanup_plan` deletes them.
- A planned `init` stage will hold once-per-data-config setup phases (`copy_src_to_scratch`). A planned `cleanup` stage will hold all cleanup phases (`wipe_src_scratch`, eventually consolidating the per-stage `cleanup_*` variants). Tracked in `phase_roster_cleanup_plan`.
- The chip-well-group phase scope (analysis-stage `unitmatch` phase) is a new pattern: ONE phase invocation processes N (dataset, well) pairs as a group. Document the group-mode invariants in this guardrail once the unitmatch phase lands.
