# axon_reconstructor Development Roadmap

This roadmap is the active navigation document for development work.

## Working rules

1. Complete items in phase order unless explicitly reprioritized.
2. After each **major item**, perform a **Docs + Roadmap Update Checkpoint**:
   - Update relevant docs (README + methods/developer docs).
   - Update this roadmap status and next actions.
   - Record key design decisions and migration notes.
3. Keep project-level analysis code minimal; move reusable logic into package `src/`.

Status legend:
- `TODO` not started
- `IN PROGRESS` active
- `DONE` completed
- `BLOCKED` requires decision/dependency

---

## Phase 0 — Planning Baseline (Current)

### 0.1 Establish roadmap + docs baseline
- Status: `DONE`
- Scope:
  - Create this roadmap.
  - Update base README for roadmap-first workflow.
  - Remove contribution guide for now.

### 0.2 Freeze implementation to roadmap-led execution
- Status: `DONE`
- Scope:
  - Execute one roadmap item at a time.
  - Require explicit handoff after each major item.

---

## Phase 1 — Debug Harness Migration + Packaging

### 1.1 Move debug harness into package-owned developer location
- Status: `DONE`
- Goal:
  - Copy relevant debug harness code from project debug folder into package docs/tools location that supports both developers and future users.
- Deliverables:
  - Package-owned debug entrypoint location (proposed: `tools/debug/` and `docs/debugging/`).
  - Developer-facing debug usage docs.
  - Clean boundary between package logic vs local experiment glue.

### 1.2 Add default `.env` and example cross-well config
- Status: `DONE`
- Goal:
  - Provide package examples for runtime defaults and large-run cross-well analysis config.
- Deliverables:
  - `docs/examples/debug.env.example` (or equivalent).
  - `docs/examples/cross_well_config.example.yml` (or equivalent).
  - Field-by-field explanation in docs.

### 1.3 Docs + Roadmap Update Checkpoint
- Status: `DONE`

---

## Phase 2 — Modular Real Analysis Project in `/projects`

### 2.1 Create dedicated analysis project for Media Density dataset
- Status: `DONE`
- Goal:
  - Create a new project under `/home/adamm/dev/projects` isolating real-analysis workflow for Media Density runs.
- Deliverables:
  - Minimal project structure for configuration + orchestration + outputs.
  - Clear dependency on `axon_reconstructor` package entrypoints.

### 2.2 Minimize project glue; maximize package reuse
- Status: `DONE`
- Goal:
  - Project code should be thin wrappers around package functionality.
- Deliverables:
  - Remove duplicated helpers from project-level scripts.
  - Replace with imports/calls into package modules.

### 2.3 Docs + Roadmap Update Checkpoint
- Status: `DONE`

---

## Phase 3 — Core Refactor for Stage Efficiency

### 3.0 Debug harness minimization + migration audit
- Status: `DONE`
- Goal:
  - Reduce `tools/debug` to thin wrappers by migrating reusable logic into `src/axon_reconstructor/pipeline`.
- Deliverables:
  - [DONE] Move assets from `src/axon_reconstructor/devtools` into `tools/debug`, then remove `src/axon_reconstructor/devtools`.
  - [DONE] Remove compatibility shims under `tools/stepwise_debug_scripts`.
  - [DONE] Move spikesorting stage logic into `src/axon_reconstructor/pipeline/stg2_spikesorting/runner.py`.
  - [DONE] Remove preprocessing debug intermediary (`tools/debug/preprocessing_debug.py`); debug entrypoint now calls pipeline stage API directly.
  - [DONE] Canonicalize smoke-test location to `tools/smoke_tests` (remove `src/axon_reconstructor/smoke` wrapper package and `axon-recon-smoke` script entrypoint).
  - [DONE] Inventory each `tools/debug` script by function ownership:
    - keep as wrapper,
    - migrate to stage module,
    - migrate to shared pipeline utility,
    - archive/remove from package-owned debug area.
  - [DONE] Explicit de-scope note executed: `prune_to_axontracking.py` moved out of package-owned debug workflows into `tools/project_local`.

#### 3.0 Inventory (initial pass)
- Keep as wrapper (target state: thin CLI/env adapters only):
  - `debug_steps.py`
  - `debug_preprocessing_step.py`
  - `debug_spikesorting_step.py`
  - `debug_waveforms_step.py`
  - `debug_templates_step.py`
  - `debug_reconstruction_step.py`
  - `debug_analysis_step.py`
  - `debug_analysis_deck.py`
- Migrate to stage module (`src/axon_reconstructor/pipeline/...`):
  - [DONE] `cross_well_analysis.py` → `pipeline/stg6_analysis/cross_well.py` (debug script now thin wrapper)
  - [DONE] `spikesorting_debug.py` → `pipeline/stg2_spikesorting/runner.py`
  - [DONE] `debug_preprocessing_step.py` now calls `AxonReconstructor.preprocess_for_spikesorting(...)` directly (no intermediary module)
- Migrate to shared utility:
  - [DONE] `debug_env.py` → shared config/env utility module (`src/axon_reconstructor/env_utils.py`); `tools/debug/debug_env.py` kept as thin compatibility adapter
- Archive/remove from package-owned debug area:
  - [DONE] `prune_to_axontracking.py` moved to `tools/project_local/prune_to_axontracking.py` (project-local)
  - [DONE] `regen_template_movies_from_raw.py` moved to `tools/project_local/regen_template_movies_from_raw.py` (project-local)
- Script de-bloating candidates once orchestration is pipeline-native:
  - [DEFERRED → 3.2c] `run_multidataset_preprocessing.sh`
  - [DEFERRED → 3.2c] `run_multidataset_spikesorting.sh`
  - [DEFERRED → 3.2c] `run_multidataset_waveforms.sh`
  - [DEFERRED → 3.2c] `run_multidataset_templates.sh`
  - [DEFERRED → 3.2c] `run_multidataset_reconstruction.sh`
  - [DEFERRED → 3.2c] `run_multidataset_stages.sh`
  - [DEFERRED → 3.2c] `run_multidataset_waveforms_templates_reconstruction.sh`

#### 3.0 Canonical smoke-test location decision
- Canonical location: `tools/smoke_tests`.
- Removed non-canonical wrapper package: `src/axon_reconstructor/smoke`.
- Removed package script entrypoint: `axon-recon-smoke` from `pyproject.toml`.

### 3.1 Stage-by-stage optimization pass
- Status: `DONE`
- Scope:
  - Preprocess
  - Spikesort
  - Waveforms/templates
  - Reconstruction
  - Analysis/cross-well
- Goals:
  - Break up overlong functions.
  - Remove dead/redundant paths.
  - Tighten interfaces and reduce side effects.

### 3.1a Analysis-stage integration (priority: `cross_well_analysis`)
- Status: `DONE`
- Goal:
  - Refactor `tools/debug/cross_well_analysis.py` into pipeline-owned analysis modules.
- Deliverables:
  - [DONE] Move reusable analysis logic into `src/axon_reconstructor/pipeline/stg6_analysis/...`.
  - [DONE] Keep debug script as thin CLI adapter that imports pipeline code.
  - [DONE] Split plotting/statistics/deck generation into testable units:
    - `src/axon_reconstructor/pipeline/stg6_analysis/cross_well_stats.py`
    - `src/axon_reconstructor/pipeline/stg6_analysis/cross_well_plotting.py`
    - `src/axon_reconstructor/pipeline/stg6_analysis/cross_well_decks.py`
    - `cross_well.py` now acts as orchestration and calls these modules.

### 3.1b Stage-owned integration for other debug scripts
- Status: `DONE`
- Goal:
  - Move non-analysis reusable logic from debug scripts into corresponding pipeline stages.
- Deliverables:
  - [DONE] Preprocess helpers moved into preprocessing stage modules:
    - `src/axon_reconstructor/pipeline/stg1_preprocessing/debug_stage.py`
  - [DONE] Spikesort helpers moved into spikesorting stage modules:
    - `src/axon_reconstructor/pipeline/stg2_spikesorting/debug_stage.py`
  - [DONE] Waveforms/templates/reconstruction helpers moved into stage modules:
    - `src/axon_reconstructor/pipeline/stg3_waveforms/debug_stage.py`
    - `src/axon_reconstructor/pipeline/stg4_templates/debug_stage.py`
    - `src/axon_reconstructor/pipeline/stg5_reconstruction/debug_stage.py`
  - [DONE] Debug entrypoints de-duplicated and reduced to thin adapters:
    - `tools/debug/debug_preprocessing_step.py`
    - `tools/debug/debug_spikesorting_step.py`
    - `tools/debug/debug_waveforms_step.py`
    - `tools/debug/debug_templates_step.py`
    - `tools/debug/debug_reconstruction_step.py`
  - [DONE] Reconstruction stage now owns axon-velocity import fallback/runtime path resolution via `ReconstructionInputs.axon_velocity_repo_root` in:
    - `src/axon_reconstructor/pipeline/stg5_reconstruction/runner.py`

### 3.2 Extract shared helpers across stages
- Status: `DONE`
- Goal:
  - Identify common utilities and move into shared modules.
- Deliverables:
  - [DONE] Pipeline-native scope config schema + validation helpers:
    - `src/axon_reconstructor/pipeline/scope_config.py`
  - [DONE] Stage-barrier orchestration helper with per-well parallel execution:
    - `src/axon_reconstructor/pipeline/stage_orchestrator.py`
  - [DONE] Reduced stage-invocation duplication by centralizing cross-stage orchestration and kwargs merging.
  - [DEFERRED → 3.5.1] Continue extracting lower-level reusable IO/logging/plot/metrics helpers from stage internals.

### 3.2a Pipeline-native scope config for multi-dataset execution
- Status: `DONE`
- Goal:
  - Define full analysis scope up front (datasets, wells, assay/timepoints, stage controls) in config rather than shell-script orchestration.
- Deliverables:
  - [DONE] Config schema supporting global stage execution across all datasets/wells:
    - `mea_output_root`, `datasets[]`, `wells[]`, `stage_order`, stage kwargs, parallelism/fail-fast controls.
  - [DONE] Validation/normalization utilities for scope config:
    - `load_scope_config`, `validate_scope_config`, `summarize_scope_config` in `src/axon_reconstructor/pipeline/scope_config.py`.

### 3.2b Pipeline-native stage orchestration across full scope
- Status: `DONE`
- Goal:
  - Run each stage across all configured datasets/wells before advancing to next stage.
- Deliverables:
  - [DONE] Built-in stage scheduler/executor in pipeline:
    - `run_scope_stage_barriers(...)` in `src/axon_reconstructor/pipeline/stage_orchestrator.py`.
  - [DONE] Stage barriers (`all preprocess` → `all spikesort` → `all waveforms` ...):
    - barrier semantics implemented from `stage_order` in scope config.
  - [DONE] Built-in per-well parallel processing controls in pipeline (not dependent on external shell loops):
    - `per_well_parallelism` support in orchestrator.
  - [DONE] CLI integration for execution and dry-run planning:
    - `axon-reconstructor scope-run --config <scope.json|yml> [--dry-run]`.

### 3.2c Multi-dataset script de-bloating/deprecation
- Status: `DONE`
- Goal:
  - Minimize or retire `run_multidataset_*.sh` scripts.
- Deliverables:
  - [DONE] Converted multi-dataset shell scripts to thin `scope-run` compatibility wrappers:
    - `tools/debug/run_multidataset_preprocessing.sh`
    - `tools/debug/run_multidataset_spikesorting.sh`
    - `tools/debug/run_multidataset_waveforms.sh`
    - `tools/debug/run_multidataset_templates.sh`
    - `tools/debug/run_multidataset_reconstruction.sh`
    - `tools/debug/run_multidataset_stages.sh`
    - `tools/debug/run_multidataset_waveforms_templates_reconstruction.sh`
  - [DONE] Added shared wrapper/conversion helpers:
    - `tools/debug/run_scope_stage_barrier.sh`
    - `tools/debug/build_scope_config.py`
  - [DONE] Documented migration path from script-driven to config-driven runs:
    - `docs/debugging/README.md`
    - `docs/examples/scope_config.example.json`
    - `docs/examples/README.md`

### 3.2d UnitMatch readiness gate in orchestration flow
- Status: `DONE`
- Goal:
  - Ensure downstream unit-matching prerequisites are satisfied by stage ordering.
- Deliverables:
  - [DONE] Guarantee all spikesorting artifacts across configured timepoints are complete before unit matching.
    - `scope-run` now enforces `unit_match` readiness by checking each target's expected `spikesorting_outputs/sorter_output` path.
  - [DONE] Add explicit transition points for: unit match → merge updates → waveform/template/reconstruction continuation.
    - Added stage-order support + validation for `unit_match` and `merge_update` in scope config.
    - Added transition-stage execution in scope orchestrator with fail-fast semantics.

### 3.2e Debug-arg centralization into package CLI
- Status: `DONE`
- Goal:
  - Move debug-stage argument definitions out of `tools/debug` scripts and into package-owned CLI/library code so there is one canonical argument surface.
- Implementation steps:
  - [DONE] Create package-owned debug CLI registry in `src/axon_reconstructor/cli.py` for stage debug commands (consolidated single-file CLI ownership).
  - [DONE] Register centralized debug subcommands in top-level CLI (`src/axon_reconstructor/cli.py`).
  - [DONE] Convert stage debug scripts into thin wrappers with no local arg definitions:
    - `tools/debug/debug_preprocessing_step.py`
    - `tools/debug/debug_spikesorting_step.py`
    - `tools/debug/debug_waveforms_step.py`
    - `tools/debug/debug_templates_step.py`
    - `tools/debug/debug_reconstruction_step.py`
  - [DONE] Convert staged runner wrapper to CLI-owned args:
    - `tools/debug/debug_steps.py` → delegates to `axon-reconstructor debug-steps`.
  - [DONE] Fold remaining non-stage debug utilities into package-owned CLI commands:
    - `debug_analysis_step.py` → `axon-reconstructor debug-analysis`
    - `debug_analysis_deck.py` → `axon-reconstructor debug-analysis-deck`

### 3.2f Stage-first CLI argument model consolidation (remove debug-named CLI surface)
- Status: `DONE`
- Goal:
  - Re-orient CLI architecture around stage and substage ownership (not debug wrappers), so debug scripts are thin launchers that apply defaults and call canonical stage commands.
- Design intent:
  - Canonical CLI commands should describe pipeline function (`stage`, stage/substage verbs), not debugging mode.
  - Debug-only behavior should come from optional flags and/or wrapper-provided env defaults, not a parallel debug command tree.
  - Argument definitions should be owned by stage modules and reused by all entrypoints.
- Implementation steps (proposed):
  - [DONE] Build a complete argument inventory:
    - enumerate args from `src/axon_reconstructor/cli.py`, all `pipeline/*/debug_stage.py` builders, and `tools/debug/*.py` wrappers.
    - include env-var defaults from `tools/debug/debug.env` and `docs/examples/debug.env.example`.
    - produce a source-of-truth matrix: `arg -> stage -> current owners -> env key -> behavior/side effects`.
    - v0 artifact created: `docs/developer/cli_arg_inventory.md`.
    - machine-parsed appendix expanded to full parser surface (`14` commands / `101` unique args-positionals).
    - Gate A checklist added and canonical non-debug command-only arg rows have now been added to inventory.
    - owner/path columns for newly added canonical non-debug command rows have been filled with concrete target registries.
    - env/default provenance for canonical non-debug rows has been verified and corrected in inventory notes.
    - legacy/debug row provenance pass completed (debug builders + analysis wrappers), with provisional sign-off markers added.
  - [DONE] Classify and assign ownership for each arg:
    - stage runtime args (required for execution),
    - stage tuning/resource args (performance/runtime controls),
    - diagnostic args (useful for stepping/troubleshooting),
    - wrapper convenience args (kept only in debug wrappers, not canonical stage CLI).
    - Gate B command-family ownership matrix + provisional approval checklist added in `docs/developer/cli_arg_inventory.md`.
    - row-level Gate B decision tagging applied (`provisional-approved` default + explicit exception tags).
    - Gate B reviewer sign-off template added in `docs/developer/cli_arg_inventory.md` (required to close Gate B).
    - Gate B approved (named sign-off captured in inventory).
    - Compatibility decision updated during 3.2g execution: retire `debug-*` aliases immediately (no extra wrapper window).
    - Debug-focused flag decision: keep `--break-before-run`, `--debug-max-units`, `--debug-max-segments` canonical on stage commands.
  - [DONE] Define target CLI hierarchy by stage/substage:
    - canonical interface is stage-first: `axon-reconstructor stage <stage> [--stage-args...]`.
    - approved substage shape for mode-specific flows: `axon-reconstructor stage <stage> <substage> [--substage-args...]`.
    - legacy `debug-*` aliases retired as part of package-first cleanup.
  - [DONE] Introduce reusable stage arg registries/builders:
    - each stage module exposes parser-registration helpers and argument normalization/build functions,
    - top-level CLI composes these helpers,
    - wrappers call canonical stage commands (no local arg definitions).
    - first implementation tranche complete: canonical `stage` parser now composes reusable helpers from `src/axon_reconstructor/pipeline/stage_cli_args.py`.
    - second implementation tranche (historical): debug alias command family was composed via `src/axon_reconstructor/pipeline/debug_cli_args.py` before retirement.
    - canonical `stage` now supports `--env-file` and env-var fallback for common runtime args with CLI-over-env precedence.
  - [DONE] Refactor wrappers to be defaults-only launchers:
    - wrappers load `debug_env` and pass stage args to canonical stage commands,
    - retain VS Code debugger-friendly single-stage scripts,
    - keep wrapper code minimal and free of duplicated parser logic.
    - wrapper tranche complete (historical): single-stage wrapper launchers were composed via `tools/debug/_stage_wrapper.py` before retirement.
    - `scope-run` now accepts `--env-file` for shared runtime/logging environment defaults.
  - [DONE] Migrate and deprecate safely:
    - add temporary compatibility aliases for `debug-*` commands with explicit deprecation messaging,
    - update docs and examples to canonical stage/substage commands,
    - remove deprecated aliases after agreed transition point.
    - deprecation phase completed; compatibility alias tree was then retired and removed.
    - canonical usage docs/examples updated in `docs/debugging/README.md` and `docs/examples/README.md`.
    - project-local env template added at `docs/examples/project.env.example`.
    - alias removal executed during 3.2g cleanup.
  - [DONE] Validate parity and remove redundancy:
    - golden help-surface snapshots for each stage command,
    - smoke parity checks between wrapper invocations and canonical CLI invocations,
    - delete superseded parsing/normalization code paths.
    - repeatable parity check script added: `tools/debug/validate_stage_cli_parity.sh`.
    - parity report artifact added: `docs/developer/cli_parity_validation.md`.

#### 3.2f.1 Argument inventory template (to fill before code changes)

Canonical artifact to populate during this item:
- `docs/developer/cli_arg_inventory.md` (new; source-of-truth for consolidation decisions)

Required table schema:

| Arg | Stage | Substage/Scope | Current CLI owner | Current builder/normalizer | Wrapper usage | Env key(s) | Default source | Behavior / side effects | Classification | Target owner | Target command path | Compatibility alias needed? | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `--h5-path` | preprocess/spikesort/waveforms/templates/reconstruct/analysis | common stage input | `src/axon_reconstructor/cli.py` | `pipeline/*/debug_stage.py` | yes | `AXON_RECON_H5_PATH` | env/wrapper | selects dataset file | runtime | stage-common arg registry | `axon-reconstructor stage <name>` | maybe | normalize path once |

Classification vocabulary (required values):
- `runtime` — required execution inputs/identifiers
- `tuning` — performance/resource or algorithm tuning controls
- `diagnostic` — troubleshooting/inspection controls useful in debugger
- `wrapper-only` — convenience flags only for wrapper defaults UX

#### 3.2f.2 CLI target-shape (approved)

Approved target command structure:
- `axon-reconstructor stage preprocess [args...]`
- `axon-reconstructor stage spikesort [args...]`
- `axon-reconstructor stage waveforms [args...]`
- `axon-reconstructor stage templates [args...]`
- `axon-reconstructor stage reconstruct [args...]`
- `axon-reconstructor stage analysis [args...]`

Approved substage grouping (when needed):
- `axon-reconstructor stage <stage> <substage> [args...]`

Compatibility/deprecation policy (approved):
- Retire `debug-*` command aliases and wrapper compatibility surfaces once canonical stage/scope commands are validated.

#### 3.2f.3 Execution gates (must pass in order)

- Gate A (Inventory complete):
  - every current arg from CLI + debug stage builders + wrappers mapped in inventory table.
- Gate B (Ownership approved):
  - each arg has approved classification + target owner + command path.
- Gate C (Parser architecture approved):
  - shared stage arg registries/builders design reviewed before migration edits.
- Gate D (Migration verified):
  - wrapper parity checks and help-surface snapshots pass.
- Gate E (Cleanup complete):
  - deprecated/duplicate parser logic removed per approved transition window.

### 3.2g Package-first harness consolidation (post-3.2f cleanup)
- Status: `DONE`
- Goal:
  - Align the debug/project harness with package-first ownership so `tools/debug` behaves like a thin user-project layer and all major capabilities live in `axon_reconstructor`.
  - Keep `tools/debug/debug.env` and `tools/debug/cross_well_config.yml` as stable project-template assets (not package feature implementations).
- Deliverables:
  - Reduce temporary debug alias surface in `src/axon_reconstructor/cli.py` after the one-release compatibility window:
    - [DONE] removed `debug-*` subcommand registrations and handlers.
    - [DONE] removed deprecated compatibility mapping used by `debug-steps` normalization.
  - Revisit parser helper placement after alias sunset:
    - [DONE] removed `src/axon_reconstructor/pipeline/debug_cli_args.py` with alias subcommand retirement.
    - keep/adjust only the canonical `stage` arg registry structure needed for maintainability.
  - Migrate remaining heavy debug analysis utilities into package modules/commands:
    - move reusable logic from `tools/debug/debug_analysis_step.py` and `tools/debug/debug_analysis_deck.py` into `src/axon_reconstructor/pipeline/stg6_analysis/...`,
    - [DONE] analysis execution moved into canonical `stage analysis` with package-owned analysis args/env resolution; retired `tools/debug/debug_analysis_step.py`.
    - [DONE] `analysis-deck` promoted as canonical package CLI command with package-owned implementation in `src/axon_reconstructor/pipeline/stg6_analysis/analysis_deck.py`.
  - Move scope-config conversion utility into package CLI:
    - [DONE] promoted cross-well to scope config conversion into package-owned `scope-config-build` command.
    - [DONE] retired `tools/debug/build_scope_config.py` compatibility wrapper.
  - Preserve editable project-style stage orchestration script in debug layer:
    - [DONE] added `tools/debug/run_stage_combo.sh` for running any chosen combination of canonical `stage` commands.
    - script contract: `--env-file` defaults from `tools/debug/debug.env`, optional stage list via `STAGES_CSV`, editable per-stage args in-script.
  - Re-assess wrapper helper lifetime:
    - [DONE] retired `tools/debug/_stage_wrapper.py` and removed deleted stage wrapper scripts.
  - Update docs/examples to treat `tools/debug` as optional project templates, not primary feature location.
    - [DONE] debug docs updated to canonical `stage`/`analysis-deck`/`scope-run`/`scope-config-build` entrypoints and retained template assets.
    - [DONE] added editable scope orchestration script docs for `tools/debug/run_scope_combo.sh`.

### 3.3 Docs + Roadmap Update Checkpoint
- Status: `DONE`

---

## Phase 3.5 — Cross-Stage Runtime Hardening

### 3.5.1 Deferred helper extraction from Phase 3.2
- Status: `DONE`
- Goal:
  - Complete deferred shared-helper extraction across all stages before reconstruction-backend expansion.
- Deliverables:
  - Extract reusable IO helper functions into shared pipeline utilities (path resolution, artifact discovery, structured output writes).
  - Extract reusable plotting helper functions into shared pipeline utilities (common figure setup/output conventions).
  - Extract reusable metrics helper functions into shared pipeline utilities (common metric serialization/aggregation utilities).
  - Replace duplicated per-stage inline helper logic with shared utility imports while preserving behavior parity.
  - Add a short developer note listing moved helpers and module ownership boundaries.

### 3.5.2 Logging harmonization across pipeline stages
- Status: `DONE`
- Goal:
  - Centralize and standardize logging behavior across preprocess/spikesort/waveforms/templates/reconstruct/analysis/scope orchestration.
- Deliverables:
  - Define a single logging contract (logger naming, levels, message shape, summary lines, error/traceback handling).
  - Implement shared logging setup/helpers used by all stage runners and orchestration entrypoints.
  - Remove stage-specific ad hoc logging differences where they do not provide clear value.
  - Ensure scope summaries and stage outputs are consistently structured for CLI and file logs.

### 3.5.3 Checkpointing architecture review + optimization
- Status: `DONE`
- Goal:
  - Review checkpoint behavior end-to-end and harden resume/restart semantics across all stages.
- Deliverables:
  - Inventory checkpoint state ownership and transitions per stage and orchestrated scope runs.
  - Identify and resolve gaps in restart semantics (`force_restart`, partial failures, fail-fast boundaries).
  - Standardize checkpoint read/write/update patterns and failure handling.
  - Add targeted validation scenarios for resume/retry behavior (single-stage + scope barrier contexts).
  - [DONE] Added targeted runtime validation tests:
    - `tests/test_stage_checkpointing.py`
    - `tests/test_stage_orchestrator_runtime.py`
  - [DONE] Added CLI-to-stage resume/checkpoint inventory + low-hanging follow-up plan:
    - `docs/developer/resume_checkpoint_inventory.md`

### 3.5.4 Docs + Roadmap Update Checkpoint
- Status: `DONE`

### 3.6 Resume/checkpoint low-hanging completion pass
- Status: `DONE`
- Goal:
  - Complete additive, non-conflicting checkpoint/resume hardening identified in the 3.5.3 inventory.
- Deliverables:
  - 3.6.1 Spikesort stage wrapper checkpoint (axon-level, additive to MEA_Analysis checkpoints): `DONE`
  - 3.6.2 Templates/reconstruct stage failure checkpoint parity (`save_stage_failed` on unhandled exceptions): `DONE`
  - 3.6.3 Analysis global resume shortcut when complete outputs exist and `force_restart=False`: `DONE`
  - 3.6.4 Analysis-deck command checkpoint artifact (`deck_started`/`deck_complete`/`deck_failed`): `DONE`
  - 3.6.5 Scope-run barrier resume artifact for stage-level barrier progress: `DONE`
  - 3.6.6 Network-scan runtime caveats (single-segment preprocess fast path + spikesort profile handling): `DONE`

### 3.7 Orchestration consolidation + legacy deletion
- Status: `DONE`
- Goal:
  - Make one canonical orchestration path (`scope-run`/`stage_orchestrator`) and remove legacy orchestration codepaths after parity is verified.
- Policy:
  - No long-lived compatibility window for deprecated orchestration paths.
  - Deprecated orchestration surfaces should be deleted in this phase once replacement parity checks pass.
- Deliverables:
  - 3.7.1 Introduce a shared stage-execution registry/API used by both `stage` CLI and `scope-run` (`DONE`)
    - eliminate duplicate stage dispatch logic across `cli.py` and `stage_orchestrator.py`.
  - 3.7.2 Rewire `run` and `pipeline` commands to canonical orchestrator semantics (`DONE`)
    - either invoke scope orchestration directly (single-target scope) or be removed if redundant.
  - 3.7.3 Eliminate `pipeline_driver.py` and move remaining responsibilities to canonical stage services (`DONE`)
    - path/layout helpers moved into `pipeline/output_paths.py`;
    - preprocess runtime ownership moved into `pipeline/preprocessing_service.py`;
    - legacy driver module removed after in-repo callsite migration.
  - 3.7.4 Delete deprecated orchestration codepaths and CLI surfaces (`DONE`)
    - remove legacy command handlers and switches that are no longer part of canonical stage/scope flow;
    - remove superseded docs/examples in the same change set.
  - 3.7.5 Add consolidation parity checks and migration notes (`DONE`)
    - targeted runtime tests for stage CLI vs scope-run stage execution parity;
    - developer note documenting final ownership boundaries (`stage_orchestrator` vs stage service modules).
    - parity tests added: `tests/test_stage_execution_parity.py`.
    - ownership note added: `docs/developer/orchestration_ownership.md`.

#### 3.7 implementation order (active)
1. 3.7.1 Shared stage executor extraction (DONE).
2. 3.7.5 Parity tests for stage CLI vs scope-run execution (DONE).
3. 3.7.2 Rewire `run`/`pipeline` to canonical orchestrator semantics (DONE).
4. 3.7.3 Eliminate `pipeline_driver.py` via service extraction + module deletion (DONE).
5. 3.7.4 Delete deprecated orchestration CLI/codepaths and update docs in same changeset (DONE).

### 3.7.6 Docs + Roadmap Update Checkpoint
- Status: `DONE`

### 3.7.7 Top-level pipeline ownership cleanup (post-consolidation)
- Status: `DONE`
- Goal:
  - Minimize top-level `pipeline/` module surface by pushing stage-specific logic into stage-owned packages and keeping only true cross-stage orchestration/config contracts at the top level.
- Deliverables:
  - 3.7.7a Preprocess service ownership alignment (`DONE`)
    - move preprocess stage service entrypoint out of top-level `pipeline/preprocessing_service.py` into `pipeline/raw_preprocessing` package (`main.py`/`runner.py` ownership).
    - remove deprecated top-level preprocess service module after import rewiring.
  - 3.7.7b Stage orchestration file-shape review (`DONE`)
    - review `stage_checkpointing.py`, `stage_cli_args.py`, `stage_execution.py`, `stage_orchestrator.py` boundaries and merge only where cohesion improves maintainability without reducing testability.
    - if merged, introduce a canonical `stage_driver` module and keep legacy module paths as short-lived shims until same changeset cleanup.
  - 3.7.7c Scope config layering review (`DONE`)
    - review split between `scope_config.py` and `scope_config_builder.py`; merge or retain split based on parser/model vs CLI-conversion responsibility boundary.
  - 3.7.7d Shared path helper ownership decision (`DONE`)
    - confirm whether MEA output path computation is cross-stage shared contract or preprocess-only concern; relocate if preprocess-only.
  - 3.7.7e Remove temporary compatibility shims (`DONE`)
    - delete `stage_cli_args.py`, `stage_execution.py`, `stage_orchestrator.py`, and `scope_config_builder.py` shims after callers/tests/docs are fully migrated to canonical owners (`stage_driver.py`, `scope_config.py`).

#### 3.7.7 implementation order (active)
1. 3.7.7a Move preprocess service into `pipeline/raw_preprocessing` package and delete top-level module.
2. 3.7.7d Re-evaluate output path helper ownership after preprocess move (DONE).
3. 3.7.7b Review/implement stage driver consolidation with tests (DONE).
4. 3.7.7c Review/implement scope config consolidation with tests (DONE).
5. 3.7.7e Remove temporary compatibility shims after full migration (DONE).

### 3.7.8 Pipeline nomenclature normalization (stage numbering + naming)
- Status: `DONE`
- Goal:
  - Normalize pipeline naming to reduce ambiguity from mixed `stage_*`, unnumbered stage package names, and inconsistent output folder labels.
  - Adopt canonical numbered stage package names while preserving ergonomic imports (`preprocessing`, `spikesorting`, etc.).
- Proposed canonical naming:
  - top-level stage runtime modules:
    - `pipeline/pipeline_driver.py` → `pipeline/pipeline_driver.py`
    - `pipeline/pipeline_checkpointing.py` → `pipeline/checkpointing.py`
  - stage package directories:
    - `pipeline/stg1_preprocessing/` → `pipeline/stg1_preprocessing/`
    - `pipeline/stg2_spikesorting/` → `pipeline/stg2_spikesorting/`
    - `pipeline/stg3_waveforms/` → `pipeline/stg3_waveforms/`
    - `pipeline/stg4_templates/` → `pipeline/stg4_templates/`
    - `pipeline/stg5_reconstruction/` → `pipeline/stg5_reconstruction/`
    - `pipeline/stg6_analysis/` → `pipeline/stg6_analysis/`
  - import ergonomics (public aliases):
    - `pipeline.preprocessing` → forwards to `pipeline.stg1_preprocessing`
    - `pipeline.spikesorting` → forwards to `pipeline.stg2_spikesorting`
    - `pipeline.waveforms` → forwards to `pipeline.stg3_waveforms`
    - `pipeline.templates` → forwards to `pipeline.stg4_templates`
    - `pipeline.reconstruction` → forwards to `pipeline.stg5_reconstruction`
    - `pipeline.analysis` → forwards to `pipeline.stg6_analysis`

- Deliverables:
  - 3.7.8a Rename top-level stage runtime module files (`DONE`)
    - rename `stage_driver.py` and `stage_checkpointing.py` to `pipeline_*` equivalents.
    - rewire all in-repo imports and update `pipeline/__init__.py` exports.
    - add short-lived compatibility shim files at old paths only if needed for one migration tranche.
  - 3.7.8b Rename all stage package directories to numbered `stgN_*` names (`DONE`)
    - move each existing stage package directory to canonical `stgN_*` path.
    - update intra-package relative imports and cross-stage imports.
  - 3.7.8c Add ergonomic public import aliases (`DONE`)
    - create alias packages/modules for `preprocessing`, `spikesorting`, `waveforms`, `templates`, `reconstruction`, `analysis`.
    - ensure alias imports are stable for callers while canonical ownership remains `stgN_*`.
  - 3.7.8d Rewire CLI + orchestrator to canonical `pipeline_*` + `stgN_*` ownership (`DONE`)
    - update CLI/runtime imports, dispatch registry imports, and stage-service callsites.
    - verify resume/checkpoint paths and scope-run behavior unchanged.
  - 3.7.8e Output path nomenclature update to stage numbering (`DONE`)
    - rename stage output directory labels to include `stg1`, `stg2`, etc. where stage-owned outputs are written.
    - keep migration-safe handling for existing output layouts where needed (read old, write new, or explicit one-time break).
  - 3.7.8f Tests and fixtures migration (`DONE`)
    - update test imports/monkeypatch targets/fixtures for renamed modules and packages.
    - update any hard-coded expected output paths for new stage-numbered directory names.
  - 3.7.8g Docs and examples migration (`DONE`)
    - update developer docs, ownership docs, and command examples referencing old module/package names.
  - 3.7.8h Compatibility shim retirement (`DONE`)
    - after parity verification, remove temporary shims for renamed modules and alias forwarders not intended to remain.

#### 3.7.8 implementation order (proposed)
1. 3.7.8a Rename top-level `stage_*.py` runtime modules to `pipeline_*.py`.
2. 3.7.8b Rename stage package directories to `stgN_*` canonical names.
3. 3.7.8c Add/verify ergonomic public import aliases (`preprocessing`, `spikesorting`, etc.).
4. 3.7.8d Rewire CLI/orchestration imports and registry wiring.
5. 3.7.8e Migrate stage output naming to explicit `stgN` labeling.
6. 3.7.8f Update tests/fixtures; run targeted then broader regression.
7. 3.7.8g Update docs/examples.
8. 3.7.8h Remove short-lived compatibility shims.

---

## Phase 4 — Reconstruction Engines Alignment

### 4.0 Single-neuron footprinting + reconstruction debug workflow
- Status: `IN PROGRESS`
- Goal:
  - Create a repeatable, minimal runner to debug footprinting and reconstruction behavior for a single selected unit, then use it to confirm/fix unit-level issues.
- Deliverables:
  - 4.0a Add one-command single-unit stage runner in debug tools (`DONE`)
    - added `tools/debug/run_single_unit_reconstruct.sh`.
    - supports selecting a target unit via `UNIT_DIR`/`UNIT_ID`, optional `FORCE_RESTART=1`, and stage-minimal single-unit kwargs for templates/reconstruct.
    - defaults GIF generation off (`ENABLE_GIFS=0`; opt-in via `ENABLE_GIFS=1`).
  - 4.0b Footprinting/reconstruction validation on target neuron (`TODO`)
    - run the single-unit workflow on the target unit and capture stage-by-stage artifacts/logs.
    - confirm expected unit outputs under current naming (`stg5_reconstruction_outputs/by_unit/unit_<id>`).
    - [DONE] Added template footprint panel improvements for batch QC:
      - per-panel sizebars on footprint maps;
      - optional shared global color scale for zoomed merged-contributing footprints across units in a templates run (`zoomed_footprints_global_color_scale`).
  - 4.0c Fix pass for confirmed unit-level footprinting/reconstruction issues (`TODO`)
    - implement targeted fixes for root-cause issues found in 4.0b.
    - rerun single-unit debug workflow + targeted regression tests.
  - 4.0d Waveforms runtime optimization pass (concurrent with 4.0b/4.0c) (`TODO`)
    - Scope:
      - apply high- and medium-impact reductions to Stage 3 redundant compute while preserving current scientific outputs.
    - Deliverables:
      - 4.0d.1 Early per-segment reuse short-circuit (`DONE`)
        - in `stg3_waveforms/extraction.py`, check for a loadable existing segment analyzer at the top of each segment loop.
        - when present, skip expensive pre-work (`seg_rec` load/preprocess, per-unit spike filtering/edge checks, `seg_sort` construction).
      - 4.0d.2 Apriori spike index cache + segment binning (`DONE`)
        - precompute per-unit spike arrays once per run and derive segment-local slices via boundary indexing (e.g., searchsorted) instead of rescanning all spikes per segment.
        - reuse that cache for both extraction and summary accounting paths.
      - 4.0d.3 Apriori unfiltered/filtered count derivation (`DONE`)
        - replace repeated per-segment calls to `_count_spikes_in_concat_window` with counts derived from the pre-binned spike indices for unfiltered and filtered sortings.
      - 4.0d.4 Segment cleanup gating (`DONE`)
        - make `remove_excess_spikes` / `remove_empty_units` optional behind a strict safety/debug flag when in-window/edge filtering has already guaranteed bounds.
      - 4.0d.5 Channel bookkeeping skip on checkpoint-complete segments (`DONE`)
        - when a segment analyzer is checkpoint-complete and loadable, skip expensive segment channel/electrode bookkeeping unless channel-group recomputation is explicitly requested.
      - 4.0d.6 Segment manifest + selective execution (`DONE`)
        - build a stage-start manifest (segment bounds, expected analyzer folder, checkpoint state, validation/loadability status).
        - execute compute only for missing/invalid segments; treat completed segments as metadata-only unless forced restart.
      - 4.0d.7 Runtime instrumentation + acceptance report (`IN PROGRESS`)
        - add timing logs for per-segment phases (load/preprocess, spike prep, analyzer compute, postprocess) and emit aggregate stage timing summary.
        - validate no scientific-output regressions and document expected wall-time improvement on the single-unit debug workflow.

### 4.1 Wire up and test partially implemented Radivojevic-style reconstruction
- Status: `TODO`
- Deliverables:
  - Executable submodule path.
  - Feature parity checks with current pipeline interfaces.
  - Basic validation outputs integrated into analysis stage.

### 4.2 Isolate axon_velocity reconstruction in parallel submodule structure
- Status: `TODO`
- Goal:
  - Make `axon_velocity` and Radivojevic-style reconstructions first-class, swappable backends.
- Deliverables:
  - Unified reconstruction backend interface.
  - Backend selector in config/runtime.

### 4.3 Implement BOTM validation/scoring (Radivojevic 2023-aligned)
- Status: `TODO`
- Goal:
  - Add BOTM as formal validation/scoring pathway.
- Deliverables:
  - BOTM metric computation hooks.
  - Comparative reporting against existing reconstruction outputs.

### 4.4 Docs + Roadmap Update Checkpoint
- Status: `TODO`

---

## Phase 5 — Longitudinal Unit Matching + Merge Assist (From Brainstorming)

### 5.1 Cross-session unit matching foundation
- Status: `TODO`
- Scope:
  - Canonical tracked unit IDs across sessions.
  - Similarity matrix + constrained assignment.
  - Confidence and provenance outputs.

### 5.2 Motion-tolerant matching strategy
- Status: `TODO`
- Scope:
  - Session registration/drift handling.
  - Partial-observation matching robustness.
  - Birth/death/unmatched track handling.

### 5.3 Multi-assay bridge (network assays as identity anchors)
- Status: `TODO`
- Goal:
  - Use frequent network assays to improve continuity between less frequent axon assays.

### 5.4 Merge-candidate scoring informed by longitudinal evidence
- Status: `TODO`
- Goal:
  - Improve within-session unit merge decisions using cross-session support.

### 5.5 Docs + Roadmap Update Checkpoint
- Status: `TODO`

---

## Phase 6 — Longitudinal Reconstruction Enrichment

### 6.1 Time-indexed template trajectory model
- Status: `TODO`
- Scope:
  - Recency-weighted template fusion priors.
  - Day-specific reconstruction preservation.

### 6.2 Reconstruction-aware longitudinal priors
- Status: `TODO`
- Scope:
  - Electrode support union constraints.
  - Latency/topology continuity priors.
  - Uncertainty-aware borrowing from adjacent DIVs.

### 6.3 Validation framework for low-ground-truth regime
- Status: `TODO`
- Scope:
  - BOTM and Bayesian template-matching benchmark integration.
  - Secondary metrics (stability/held-out consistency/branch continuity).
  - Method comparisons (`axon_velocity` vs Radivojevic-style vs longitudinal enriched).

### 6.4 Docs + Roadmap Update Checkpoint
- Status: `TODO`

---

## Phase 7 — Device Generalization and Scale

### 7.1 Abstraction for multi-device support
- Status: `TODO`
- Scope:
  - Maxwell (current)
  - 3Brain (planned)
  - Sony high-channel platform (planned)

### 7.2 Assay/channel model abstraction
- Status: `TODO`
- Goal:
  - Decouple platform-specific acquisition/channel assumptions from core matching and reconstruction logic.

### 7.3 Docs + Roadmap Update Checkpoint
- Status: `TODO`

---

## Phase 8 — Branch-Type Tracking Expansion (Axon + Dendrite)

### 8.1 Axon-vs-dendrite branch classification strategy
- Status: `TODO`
- Goal:
  - Define a robust, auditable method to classify tracked branches as axonal vs dendritic within the existing unit-level reconstruction framework.
- Deliverables:
  - Short literature review of current methods for branch-type classification from extracellular morphology + propagation signatures.
  - Candidate feature inventory (latency profile, propagation directionality, branch geometry, signal amplitude decay, neighborhood context).
  - Decision framework for deterministic rules vs probabilistic classifier vs hybrid approach.

### 8.2 Dendritic tracking feasibility + method design
- Status: `TODO`
- Goal:
  - Determine what is needed to extend current tracking from axon-focused outputs to explicit dendritic tracking.
- Deliverables:
  - Gap analysis of current pipeline outputs/artifacts for dendritic inference readiness.
  - Proposed pipeline insertion points (templates/reconstruction/analysis) for dendrite-specific processing.
  - Minimal prototype plan with acceptance criteria and failure modes.

### 8.3 Longitudinal axon development via unit matching
- Status: `TODO`
- Goal:
  - Quantify axonal development trajectories across DIVs using `unit_match` continuity.
- Deliverables:
  - Per-unit longitudinal morphology metric set (e.g., branch count, path length proxies, covered area, conduction-related summaries).
  - Time-series aggregation and visualization plan across matched units.
  - Validation plan for biological plausibility and robustness under partial tracking.

### 8.4 Docs + Roadmap Update Checkpoint
- Status: `TODO`

---

## Immediate Next Item (for execution)

`Phase 4.1` — Wire up and test partially implemented Radivojevic-style reconstruction.