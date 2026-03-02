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
  - [DONE] Move spikesorting stage logic into `src/axon_reconstructor/pipeline/spikesorting/runner.py`.
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
  - [DONE] `cross_well_analysis.py` → `pipeline/analysis/cross_well.py` (debug script now thin wrapper)
  - [DONE] `spikesorting_debug.py` → `pipeline/spikesorting/runner.py`
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
- Status: `IN PROGRESS`
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
  - [DONE] Move reusable analysis logic into `src/axon_reconstructor/pipeline/analysis/...`.
  - [DONE] Keep debug script as thin CLI adapter that imports pipeline code.
  - [DONE] Split plotting/statistics/deck generation into testable units:
    - `src/axon_reconstructor/pipeline/analysis/cross_well_stats.py`
    - `src/axon_reconstructor/pipeline/analysis/cross_well_plotting.py`
    - `src/axon_reconstructor/pipeline/analysis/cross_well_decks.py`
    - `cross_well.py` now acts as orchestration and calls these modules.

### 3.1b Stage-owned integration for other debug scripts
- Status: `DONE`
- Goal:
  - Move non-analysis reusable logic from debug scripts into corresponding pipeline stages.
- Deliverables:
  - [DONE] Preprocess helpers moved into preprocessing stage modules:
    - `src/axon_reconstructor/pipeline/raw_preprocessing/debug_stage.py`
  - [DONE] Spikesort helpers moved into spikesorting stage modules:
    - `src/axon_reconstructor/pipeline/spikesorting/debug_stage.py`
  - [DONE] Waveforms/templates/reconstruction helpers moved into stage modules:
    - `src/axon_reconstructor/pipeline/waveforms/debug_stage.py`
    - `src/axon_reconstructor/pipeline/templates/debug_stage.py`
    - `src/axon_reconstructor/pipeline/reconstruction/debug_stage.py`
  - [DONE] Debug entrypoints de-duplicated and reduced to thin adapters:
    - `tools/debug/debug_preprocessing_step.py`
    - `tools/debug/debug_spikesorting_step.py`
    - `tools/debug/debug_waveforms_step.py`
    - `tools/debug/debug_templates_step.py`
    - `tools/debug/debug_reconstruction_step.py`
  - [DONE] Reconstruction stage now owns axon-velocity import fallback/runtime path resolution via `ReconstructionInputs.axon_velocity_repo_root` in:
    - `src/axon_reconstructor/pipeline/reconstruction/runner.py`

### 3.2 Extract shared helpers across stages
- Status: `IN PROGRESS`
- Goal:
  - Identify common utilities and move into shared modules.
- Deliverables:
  - [DONE] Pipeline-native scope config schema + validation helpers:
    - `src/axon_reconstructor/pipeline/scope_config.py`
  - [DONE] Stage-barrier orchestration helper with per-well parallel execution:
    - `src/axon_reconstructor/pipeline/stage_orchestrator.py`
  - [DONE] Reduced stage-invocation duplication by centralizing cross-stage orchestration and kwargs merging.
  - [TODO] Continue extracting lower-level reusable IO/logging/plot/metrics helpers from stage internals.

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
- Status: `TODO`
- Goal:
  - Ensure downstream unit-matching prerequisites are satisfied by stage ordering.
- Deliverables:
  - Guarantee all spikesorting artifacts across configured timepoints are complete before unit matching.
  - Add explicit transition points for: unit match → merge updates → waveform/template/reconstruction continuation.

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
- Status: `IN PROGRESS`
- Goal:
  - Align the debug/project harness with package-first ownership so `tools/debug` behaves like a thin user-project layer and all major capabilities live in `axon_reconstructor`.
- Deliverables:
  - Reduce temporary debug alias surface in `src/axon_reconstructor/cli.py` after the one-release compatibility window:
    - [DONE] removed `debug-*` subcommand registrations and handlers.
    - [DONE] removed deprecated compatibility mapping used by `debug-steps` normalization.
  - Revisit parser helper placement after alias sunset:
    - [DONE] removed `src/axon_reconstructor/pipeline/debug_cli_args.py` with alias subcommand retirement.
    - keep/adjust only the canonical `stage` arg registry structure needed for maintainability.
  - Migrate remaining heavy debug analysis utilities into package modules/commands:
    - move reusable logic from `tools/debug/debug_analysis_step.py` and `tools/debug/debug_analysis_deck.py` into `src/axon_reconstructor/pipeline/analysis/...`,
    - [DONE] analysis execution moved into canonical `stage analysis` with package-owned analysis args/env resolution; retired `tools/debug/debug_analysis_step.py`.
    - [DONE] `analysis-deck` promoted as canonical package CLI command with package-owned implementation in `src/axon_reconstructor/pipeline/analysis/analysis_deck.py`.
  - Move scope-config conversion utility into package CLI:
    - [IN PROGRESS] promote cross-well to scope config conversion into package-owned command.
    - [DONE] retired `tools/debug/build_scope_config.py` compatibility wrapper.
  - Re-assess wrapper helper lifetime:
    - [DONE] retired `tools/debug/_stage_wrapper.py` and removed deleted stage wrapper scripts.
  - Update docs/examples to treat `tools/debug` as optional project templates, not primary feature location.
    - [IN PROGRESS] debug docs updated to canonical `stage`/`analysis-deck`/`scope-run` entrypoints.

### 3.3 Docs + Roadmap Update Checkpoint
- Status: `TODO`

---

## Phase 4 — Reconstruction Engines Alignment

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

## Immediate Next Item (for execution)

`Phase 3.2d` — UnitMatch readiness gate in orchestration flow.