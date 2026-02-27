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
- Status: `IN PROGRESS`
- Goal:
  - Reduce `tools/debug` to thin wrappers by migrating reusable logic into `src/axon_reconstructor/pipeline`.
- Deliverables:
  - [DONE] Move assets from `src/axon_reconstructor/devtools` into `tools/debug`, then remove `src/axon_reconstructor/devtools`.
  - [DONE] Remove compatibility shims under `tools/stepwise_debug_scripts`.
  - [DONE] Move spikesorting stage logic into `src/axon_reconstructor/pipeline/spikesorting/runner.py`.
  - [DONE] Remove preprocessing debug intermediary (`tools/debug/preprocessing_debug.py`); debug entrypoint now calls pipeline stage API directly.
  - [DONE] Canonicalize smoke-test location to `tools/smoke_tests` (remove `src/axon_reconstructor/smoke` wrapper package and `axon-recon-smoke` script entrypoint).
  - Inventory each `tools/debug` script by function ownership:
    - keep as wrapper,
    - migrate to stage module,
    - migrate to shared pipeline utility,
    - archive/remove from package-owned debug area.
  - Explicit de-scope note: `prune_to_axontracking.py` is one-time/project-local and should remain outside package-owned debug workflows.

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
  - `cross_well_analysis.py` → `pipeline/analysis/...` (priority)
  - [DONE] `spikesorting_debug.py` → `pipeline/spikesorting/runner.py`
  - [DONE] `debug_preprocessing_step.py` now calls `AxonReconstructor.preprocess_for_spikesorting(...)` directly (no intermediary module)
- Migrate to shared utility:
  - `debug_env.py` → shared config/env utility module (pipeline-adjacent or common util package)
- Archive/remove from package-owned debug area:
  - `prune_to_axontracking.py` (de-scope; keep project-local only)
  - `regen_template_movies_from_raw.py` (evaluate if still needed after stage-owned migration)
- Script de-bloating candidates once orchestration is pipeline-native:
  - `run_multidataset_preprocessing.sh`
  - `run_multidataset_spikesorting.sh`
  - `run_multidataset_waveforms.sh`
  - `run_multidataset_templates.sh`
  - `run_multidataset_reconstruction.sh`
  - `run_multidataset_stages.sh`
  - `run_multidataset_waveforms_templates_reconstruction.sh`

#### 3.0 Canonical smoke-test location decision
- Canonical location: `tools/smoke_tests`.
- Removed non-canonical wrapper package: `src/axon_reconstructor/smoke`.
- Removed package script entrypoint: `axon-recon-smoke` from `pyproject.toml`.

### 3.1 Stage-by-stage optimization pass
- Status: `TODO`
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
- Status: `TODO`
- Goal:
  - Refactor `tools/debug/cross_well_analysis.py` into pipeline-owned analysis modules.
- Deliverables:
  - Move reusable analysis logic into `src/axon_reconstructor/pipeline/analysis/...`.
  - Keep debug script as thin CLI adapter that imports pipeline code.
  - Split plotting/statistics/deck generation into testable units.

### 3.1b Stage-owned integration for other debug scripts
- Status: `TODO`
- Goal:
  - Move non-analysis reusable logic from debug scripts into corresponding pipeline stages.
- Deliverables:
  - Preprocess helpers into preprocessing stage modules.
  - Spikesort helpers into spikesorting stage modules.
  - Waveforms/templates/reconstruction helpers into their stage modules.
  - Remove duplicated logic currently copied across debug entrypoints.

### 3.2 Extract shared helpers across stages
- Status: `TODO`
- Goal:
  - Identify common utilities and move into shared modules.
- Deliverables:
  - Reusable IO/config/logging/plot/metrics helpers.
  - Reduced duplication across stage implementations.

### 3.2a Pipeline-native scope config for multi-dataset execution
- Status: `TODO`
- Goal:
  - Define full analysis scope up front (datasets, wells, assay/timepoints, stage controls) in config rather than shell-script orchestration.
- Deliverables:
  - Config schema supporting global stage execution across all datasets/wells.
  - Validation/normalization utilities for scope config.

### 3.2b Pipeline-native stage orchestration across full scope
- Status: `TODO`
- Goal:
  - Run each stage across all configured datasets/wells before advancing to next stage.
- Deliverables:
  - Built-in stage scheduler/executor in pipeline.
  - Stage barriers (`all preprocess` → `all spikesort` → `all waveforms` ...).
  - Built-in per-well parallel processing controls in pipeline (not dependent on external shell loops).

### 3.2c Multi-dataset script de-bloating/deprecation
- Status: `TODO`
- Goal:
  - Minimize or retire `run_multidataset_*.sh` scripts.
- Deliverables:
  - Keep only thin wrappers (or remove) once pipeline orchestration is native.
  - Document migration path from script-driven to config-driven runs.

### 3.2d UnitMatch readiness gate in orchestration flow
- Status: `TODO`
- Goal:
  - Ensure downstream unit-matching prerequisites are satisfied by stage ordering.
- Deliverables:
  - Guarantee all spikesorting artifacts across configured timepoints are complete before unit matching.
  - Add explicit transition points for: unit match → merge updates → waveform/template/reconstruction continuation.

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

`Phase 3.0` + `Phase 3.1a` — Debug harness migration audit and `cross_well_analysis` integration into pipeline-owned analysis modules.