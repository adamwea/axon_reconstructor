# Brain — protected goal/context modules

This directory is the loop's **persistent goal slot** + **trust anchor** + **dependency map**. Read FIRST every iteration, before guardrails / memory / plans. Designed to survive context-window drift over long autonomous runs.

## Why this exists

A long autonomous loop running on this codebase has repeatedly drifted: forgot user-explicit constraints (use plot_recons not new code), substituted easier targets (kilosort cluster instead of high-branch unit), self-graded its own work, built parallel renderers that satisfied no comparison goal, and shipped paper-overs that the next plan immediately reverted. Each failure mapped to a missing brain-theory module: goal-holding under interference, monitor-actor separation, trust hierarchy, dependency graph.

The fix isn't more rules — it's a SMALL set of protected files the loop consults before doing anything else, that hold:
- **What we're trying to do** (objectives + definition-of-done)
- **What's trustworthy enough to verify against** (trusted outputs)
- **What depends on what** (dependency graph, change-propagation)
- **What "better" means + the rollback trigger** (metrics)
- **What each shipped slice committed to** (slice contracts — compressed returns)

Together these are the brain's "stable goal-attractor at the top biasing a hierarchical controller, with a monitor running prediction-error against progress, an inhibitory shield protecting it from interference."

## Files

### Mode router (read BEFORE backbone — determines what loop is allowed to do this iteration)

| File | Module (brain-theory) | What it holds |
|---|---|---|
| `mode.md` | **Inhibition / interference control** | Active loop mode (`extended_autonomous` / `collaborative` / `paused`) + permitted action set per mode. The "selection mechanism" that decides what gets into the limited workspace. |

### Stable backbone (read every iteration, change slowly)

| File | Module (brain-theory) | What it holds |
|---|---|---|
| `objectives.md` | **Goal representation** | Top-level objectives for the current iteration cycle + definition-of-done per objective. The "stable attractor" that biases everything downstream. |
| `dependency_graph.md` | **Persistent DAG + propagation tracking** | What phases/modules/plans depend on what. When a slice changes a contract, this is what tells the loop what to re-verify. Populated by phase-zero mapping mission. |
| `trusted_outputs.md` | **Verifier anchor (tiered trust)** | Three-tier registry: trusted (user-pinned) > advisory (existing tests, demoted) > provisional (new outputs without reference). Anchors the whole verification chain. |
| `metrics.md` | **Value/motivation + termination** | What "better" means per refinement target + baseline measurements + rollback trigger. Without this, refinement has no natural endpoint. |
| `slice_contracts.md` | **Compressed returns w/ contracts** | Per shipped slice: produces / assumes / propagates. The "subtask returns the interface, not the trace" discipline. |
| `escalation.md` | **Flexible updating / perseveration limit** | After N=3 same-fix retries / M=5 stalled-slice iterations / K=20 stalled-plan / T=50 loop-wide thrash → MUST pop up the tree. Counters in `notes.md`. |

### Working layer (read every iteration, change continuously)

| File | What it holds |
|---|---|
| `current_state.md` | What's shipped / in-flight / queued. USER INJECTIONS at the top are authoritative. Folded from former `memory/`. |
| `open_questions.md` | Pending decisions awaiting user input or empirical data. Plan-audit findings. PRE-DIAGNOSTIC GATEs. Multiple-choice format per the stop-and-ask discipline. |
| `diagnostics_to_review.md` | Visual / tabular diagnostics filed during slices awaiting user review (soft- or hard-gate). |
| `notes.md` | Free-form scratch — debugging trails, half-formed thoughts. Prune aggressively. |

### Reference layer (read as needed, not every iteration)

| Subdir | What it holds |
|---|---|
| `guardrails/` | Locked code contracts — env_parity, force_restart, loop_cadence, output_locations, package_contracts, parallelism, scope_flags, stage_phase_architecture, dry_run. Treat as authoritative; user-approval to change. |
| `refs/` | Reference docs — paper summaries, audit docs, usage docs (radivojevic2023, kilosort4_base_audit, dashboard_audit, propagation_video, etc.). External-spec mining lives here. |

## Reading discipline (loop)

Every loop iteration, in this order:

0. **Mode FIRST**: `brain/mode.md` — confirms what kinds of actions are permitted this iteration. If a planned action is outside the active mode's permitted set, HALT + surface multiple-choice.
1. **Backbone**: `brain/objectives.md` (goal slot) → `brain/trusted_outputs.md` top (trust state) → `brain/dependency_graph.md` for any node the current slice touches (propagation) → `brain/metrics.md` for any metric the slice surface affects (rollback triggers) → `brain/escalation.md` counters in `brain/notes.md` (am I about to hit a perseveration limit?).
2. **Working layer**: `brain/current_state.md` USER INJECTIONS (authoritative) → `brain/open_questions.md` (gates, plan-audit findings) → `brain/diagnostics_to_review.md` (anything blocking).
3. **Reference layer as needed**: `brain/guardrails/<topic>.md` re-read whenever a slice's surface area maps to one of them (including `critic_separation.md` for code-shipping slices). `brain/refs/<doc>.md` consulted when implementing a new algorithm / phase that's been previously researched.
4. **Plan + commit log**: `plans/active/<current-plan>.md` cover-to-cover before starting a slice. `commit_log.md` for what shipped recently.

The point: mode + backbone (goal + trust + DAG + metrics + escalation) survives in **active context every iteration**, not just in scrolled-past history. The working + reference layers are read on a needs-basis.

## Writing discipline (loop)

- `mode.md` — USER-ANCHORED. Loop NEVER edits autonomously. Loop CAN propose mode changes via `open_questions.md` multiple-choice.
- `objectives.md` — loop NEVER edits without explicit user approval. Only the user sets objectives + definition-of-done. Loop can propose changes via `open_questions.md` multiple-choice.
- `trusted_outputs.md` — loop NEVER promotes an output to "trusted" tier without explicit user approval (the user-anchoring of the trust chain is the whole point). Loop CAN add candidates to a "proposed for promotion" section.
- `dependency_graph.md` — loop UPDATES whenever a slice changes a contract (adds an edge, modifies a node's interface). The graph is auto-maintained, not user-blessed; user reviews on audit-pass.
- `metrics.md` — loop UPDATES baselines after each smoke that establishes one. Adding a NEW metric (= new refinement target) requires user approval via `open_questions.md`.
- `slice_contracts.md` — loop APPENDS one entry per shipped slice. Append-only; never edits prior entries (they're history). Slice that doesn't produce a contract entry is INCOMPLETE per CLAUDE.md slice protocol.
- `escalation.md` — rules require user approval. The numeric LIMITS (N=3, M=5, K=20, T=50) can be tuned by user; loop maintains the counters in `notes.md`.
- `notes.md` — loop AUTO-UPDATES perseveration counters per iteration (see `escalation.md`). User can prune freely.

## What this dir is NOT

- NOT a plan registry — plans live in `plans/active/`
- NOT a tracker — long-form backlog (tech_debt, issues, roadmap, smoke_log, salloc_smokes_queued) lives in `trackers/`
- NOT a TODO list — that's `dev/notes/TODO.md`
- NOT a commit log — `commit_log.md` is the append-only audit trail

Brain holds what the loop uses to **self-regulate**: things the user doesn't want to manually review every iteration but does want the loop to consult. Trackers, plans, TODOs, and the commit log are user-facing or audit-facing artifacts that live alongside.
