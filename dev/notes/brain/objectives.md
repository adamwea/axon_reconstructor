# Objectives — current iteration cycle

The persistent goal slot. Loop reads this FIRST every iteration. Survives interruptions, context compactions, re-fires of the /loop prompt. Do NOT scroll past — this is the stable attractor.

**Reading rule for the loop**: if the next-slice candidate doesn't advance one of these objectives, the slice is OUT-OF-SCOPE and gets surfaced via `open_questions.md` for user approval, not executed.

**Editing rule**: loop NEVER edits this file without explicit user direction. User-anchored.

---

## Current cycle: v1 axon_recon end-to-end on M08073 80k DMEM well000

**Stated**: 2026-05-18 (working-data scope locked).
**Anchor cohort**: 80k DMEM well000 of M08073 chips, all DIVs (MaxTwo 10 kHz). Validated baseline: `260326/M08073/000208/well000` (DIV 36, 176 reconstructed templates post-SLAy aux-tsv sync).

### O1 — kssynth integrated into recon stage
**What**: `reconstruct.kssynth` phase produces per-unit `merged_template.npy` + `merged_channel_locations.npy` matching `build_templates`' layout; downstream phases (plot_templates_v2, report_templates, axon_velocity_gtrs) consume them transparently.
**Status**: phase code shipped through `kssynth_recon_integration_plan` slices 0-4 (+ 4c/4d/4e). Slice 5 (enable in YAML + retire `extract_partial_templates` + `build_templates` predecessors) pending; HARD-gate regression (176 templates) blocks slice 5.
**Definition of done (DOD)**:
1. kssynth-produced merged_template files exist for all post-merge units on M08073/well000/DIV 36.
2. Full reconstruct stage run on that well produces 176 reconstructed templates (matches pre-kssynth baseline; trusted-output regression).
3. Plot_recons output on a high-branch unit is qualitatively comparable to the pre-kssynth output (USER review).
4. `extract_partial_templates.py` + `build_templates.py` deleted from src/; tests migrated.

### O2 — unitlink (cross-session unit matching) integrated as analysis phase
**What**: `analysis.unitmatch` phase consumes kssynth outputs from a (chip, well) group of DIVs and writes a per-group match table (`unitmatch/<chip>/<well>/...`).
**Status**: phase scaffolded through `unitmatch_phase_plan` slices 1-4. Slice 5 (enable + login-node smoke) gated on O1 plus a salloc-allocation smoke.
**DOD**:
1. unitmatch phase enabled in YAML, runs end-to-end on a (M08073, well000) group of ≥3 DIVs from kssynth outputs.
2. Match table written; row counts + UID chain sanity-checked against trusted-output invariants.
3. Existing unitlink test suite green; no soft-skip fallbacks trigger.

### O3 — radivojevic_recon as alternative recon algorithm
**What**: clean-room implementation of Radivojevic 2023 axon-reconstruction algorithm as sibling pkg + `reconstruct.radivojevic_recon` phase; comparable outputs to `axon_velocity_gtrs` for the same merged_template input.
**Status**: sibling package Stages 1+2+3 shipped (`radivojevic2023_recon_algo.reconstruct()` callable end-to-end). axon_recon-side `radivojevic_recon` phase + plot_recons adapter NOT yet shipped. First real-data smoke ran on a substituted kilosort cluster (NOT a high-branch post-merge unit) — diagnostic was rejected by user; apples-to-apples comparison gated on O1's heavy smoke producing merged_template files.
**DOD**:
1. Radivojevic phase ships as a recon-stage phase consuming the SAME `merged_template.npy` shape that `axon_velocity_gtrs` consumes.
2. Outputs include a gtr-shaped result object: branches (count, lengths), conduction velocities, selected channels, soma localization.
3. Side-by-side comparison PNG via `plot_recons` (NOT a new renderer) on a high-branch post-merge unit (target: unit_0598 or equivalent, ≥8 inter-branch segments).
4. Numerical comparison invariants pass (see `brain/trusted_outputs.md` "Radivojevic vs axon_velocity_gtrs differential checks"):
   - Identical merged_template inputs (byte-for-byte).
   - Similar SIZE (area covered): within tolerance TBD by triage.
   - Similar BRANCHING (n_branches within tolerance).
   - Similar VELOCITY (median + distribution comparable).
   - Similar SOMA LOCALIZATION (xy within μm tolerance).
5. USER visual review of the comparison PNG approves the result.

### O4 — Dashboard UI refinement
**What**: `src/axon_recon/dashboard/` reads existing analyzed_data and presents pipeline outputs without broken-empty-state UI; consistent styling; box↔bar toggle; tertiary grouping.
**Status**: plan complete (all 9 slices SHIPPED). No formal DOD verification yet; runs interactively in the user's browser.
**DOD**: USER interactive review approves the dashboard renders cleanly on current analyzed_data.

### O5 — Resource-profile elimination (correctness of worker counts in smokes)
**What**: replace YAML `resources.profiles.*` with env-only supply resolver; cpus_per_task / tasks_per_node read from `SLURM_CPUS_PER_TASK` / `sched_getaffinity` / `CUDA_VISIBLE_DEVICES` directly; YAML profile concept deleted.
**Status**: plan written (`resources_profiles_elimination_plan.md`, 5 slices). Slice 0 reverts 2026-05-21 paper-overs. Sequencing locked: resources_profiles slices 0-2 BEFORE `parallelism_post_migration_cleanup_plan` slice 2 (see `open_questions.md` Finding #1 execution-order table).
**DOD**:
1. `resources.profiles.*` deleted from YAMLs + `src/`; `--profile` CLI flag removed.
2. Smoke under `srun -n 1 -c 128 ...` shows actual cpus_per_task = 128 in `phase_parallelism event=` logs (NOT clamped to YAML default).
3. Existing test suite passes after profile removal.
4. All sbatch scripts have `--profile` references removed.

---

## Phase zero (precedes O1-O5 work resuming)

The current pause exists because we can't safely resume O1-O5 without a verifier scaffold. Phase zero is the loop-allowed work to build that scaffold.

### Z1 — Pipeline dependency graph (autonomous; loop produces, populates `brain/dependency_graph.md`)
- Read existing code + active plans; produce DAG of phases / modules / plans / contracts.
- Identify junctions (nodes that many things flow through).
- Output: ranked verification checkpoints — "if you certify these N outputs, you transitively cover M% of the pipeline."
- **No real-data smokes during this work** — it's pure code-reading + graph construction.

### Z2 — User triages verification checkpoint list (USER)
- Bucket Z1's ranked list into already-trusted (pin) / cheap-to-acquire (eyeball + approve) / expensive (judgment call, no reference yet).
- Knock out the cheap bucket inline.

### Z3 — Loop generates invariant assertions from trusted outputs (loop, then USER approval)
- For each pinned trusted output, loop authors invariant-based tests: schema, counts, value ranges, reconcilements.
- USER approves the ASSERTIONS (not the data — fast review).
- Promoted assertions become the trusted gate.

### Z4 — User lifts the pause via new USER INJECTION
- When Z1-Z3 are far enough to anchor the verifier, the user authors an injection saying so.
- Loop resumes O1-O5 work gated on the new scaffold.

---

## What's OUT of scope this cycle

- Network scans (raw_data tree, single-segment recordings) — reserved for unitmatch v2 / `unitmatch_phase_plan` slice 6+.
- Other devices (ThreeBrain, Sony HD-MEA) — tracked in `trackers/tech_debt.md` §"Device-agnostic source data"; design-only.
- bombcell/SLAy code deletion — keep them disabled per user, not deleted.
- DeepUnitMatch HD-MEA training — its own project, deferred.

---

## How to read this when context drifts

If the loop finds itself uncertain whether a slice candidate is in scope: re-read this file. The 5 objectives + Phase zero are the entire current target. Anything that doesn't advance one of them or build the verifier scaffold is OUT-OF-SCOPE and gets surfaced as a multiple-choice question, not executed.
