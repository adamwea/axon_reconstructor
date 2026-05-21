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

| File | Module (brain-theory) | What it holds |
|---|---|---|
| `objectives.md` | **Goal representation** | Top-level objectives for the current iteration cycle + definition-of-done per objective. The "stable attractor" that biases everything downstream. |
| `dependency_graph.md` | **Persistent DAG + propagation tracking** | What phases/modules/plans depend on what. When a slice changes a contract, this is what tells the loop what to re-verify. Populated by phase-zero mapping mission. |
| `trusted_outputs.md` | **Verifier anchor (tiered trust)** | Three-tier registry: trusted (user-pinned) > advisory (existing tests, demoted) > provisional (new outputs without reference). Anchors the whole verification chain. |
| `metrics.md` | **Value/motivation + termination** | What "better" means per refinement target + baseline measurements + rollback trigger. Without this, refinement has no natural endpoint. |
| `slice_contracts.md` | **Compressed returns w/ contracts** | Per shipped slice: produces / assumes / propagates. The "subtask returns the interface, not the trace" discipline. |

## Reading discipline (loop)

Every loop iteration, before reading `current_state.md` or `open_questions.md` or any plan:

1. Read `brain/objectives.md` — what we're actually trying to do
2. Glance at `brain/trusted_outputs.md` top section — current trust state
3. Read `brain/dependency_graph.md` for any node the current slice touches — what re-verification this slice triggers
4. Check `brain/metrics.md` for any metric the current slice's surface area affects
5. Then proceed to `current_state.md` USER INJECTIONS + the rest of the entry protocol

The point: the goal + trust state survive in **active context every iteration**, not just in scrolled-past history.

## Writing discipline (loop)

- `objectives.md` — loop NEVER edits without explicit user approval. Only the user sets objectives + definition-of-done. Loop can propose changes via `open_questions.md` multiple-choice.
- `trusted_outputs.md` — loop NEVER promotes an output to "trusted" tier without explicit user approval (the user-anchoring of the trust chain is the whole point). Loop CAN add candidates to a "proposed for promotion" section.
- `dependency_graph.md` — loop UPDATES whenever a slice changes a contract (adds an edge, modifies a node's interface). The graph is auto-maintained, not user-blessed; user reviews on audit-pass.
- `metrics.md` — loop UPDATES baselines after each smoke that establishes one. Adding a NEW metric (= new refinement target) requires user approval via `open_questions.md`.
- `slice_contracts.md` — loop APPENDS one entry per shipped slice. Append-only; never edits prior entries (they're history).

## What this dir is NOT

- NOT a plan registry — plans live in `plans/active/`
- NOT a status log — current state lives in `memory/current_state.md`
- NOT an issues tracker — that's `trackers/issues.md`
- NOT a TODO list — that's `dev/notes/TODO.md`

These files are the **stable backbone** the others reference. They change slowly. If you find yourself editing them every iteration, something's wrong.
