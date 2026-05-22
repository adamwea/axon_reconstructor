# Guardrails

Locked code contracts for axon_recon and its sibling packages. Treat each topic file here as **the source of truth** for the rule it documents. If a slice's edits violate a guardrail, the guardrail wins; if the guardrail is wrong, fix the guardrail first (in its own `claude:` commit), then proceed.

## When to read

- At the start of every session (per `CLAUDE.md`'s entry protocol).
- Before starting any slice — re-read the topic files whose surface area the slice touches.
- After any conversation segment where a new contract surfaces — confirm whether it goes in a guardrail or somewhere else (see §"What goes where" below).

## When to update

Update a guardrail when:
- A new invariant has been agreed and you want it enforced going forward.
- An existing invariant has been clarified or sharpened (e.g. an edge case got pinned down).
- A contract has been retired (delete the entry, don't leave dead rules).

Don't update a guardrail when:
- You're recording in-flight work (→ `brain/current_state.md`).
- You're noting an open question (→ `brain/open_questions.md`).
- You're scoping a refactor (→ a `plans/active/*.md` plan, or a `trackers/tech_debt.md` entry).

## What goes where

| Kind of statement | Lives in |
|---|---|
| "The code must do X" — locked contract | `brain/guardrails/<topic>.md` |
| "We're currently doing Y" — present state | `brain/current_state.md` |
| "We haven't decided Z" — pending question | `brain/open_questions.md` |
| "We should refactor W someday" — backlog | `trackers/tech_debt.md` / `roadmap.md` / `issues.md` |
| "Here's the multi-slice plan to do W" — execution roadmap | `plans/active/<plan>.md` |

## Topic files

| File | Scope |
|---|---|
| [`parallelism.md`](parallelism.md) | n_jobs resolution, slot.cpu_count, MPI worker behavior, the `resolve_inner_worker_count` contract |
| [`scope_flags.md`](scope_flags.md) | `--target-datasets`, `--target-wells`, `--targets`, `--limit-*`, `--profile`, `--task-backend` semantics |
| [`force_restart.md`](force_restart.md) | Three invocation modes: no-flag (auto-restart-from-first-broken), `--force-restart` (rmtree everything), `--replot` (plot phases only, orthogonal). No fallbacks. |
| [`stage_phase_architecture.md`](stage_phase_architecture.md) | Stage / phase invariants — phase_sequence, phases dict, summary_json, resource_class, output_rel_root |
| [`package_contracts.md`](package_contracts.md) | Sibling-repo layout, SI-convention compliance for shared packages, no axon_recon-specific deps in `kssynth` / `unitlink` / `SLAy` / `UnitMatchPy` |
| [`output_locations.md`](output_locations.md) | Logs → pscratch; code → `/global/homes`; what's tracked in repo vs gitignored |
| [`dry_run.md`](dry_run.md) | Every phase exposes `--dry-run` and short-circuits at input resolution. Universal cheap-smoke-test path. |
| [`env_parity.md`](env_parity.md) | `axon_recon` conda env and the shifter image must have equivalent capabilities, EXCEPT Kilosort+CUDA stack and NERSC/HPC/SLURM plumbing. New conda dep mirrors into Dockerfile in the same slice, or surfaces a "rebuild needed" USER INJECTION. |
| [`loop_cadence.md`](loop_cadence.md) | `ScheduleWakeup.reason` format ("Next iteration in {N}s — {sentence}") + cadence ladder (90s actively iterating / 120s between slices / 300s audit / 600-1200s blocked). Promoted from a 2026-05-21 USER INJECTION. |
| [`critic_separation.md`](critic_separation.md) | Every code-shipping slice MUST run a separate Explore subagent as critic before commit. Narrow scope (diff + trusted-output fixture + invariants only). Actor doesn't grade its own work. Promoted from 2026-05-21 brain-build session. |

## Format (light)

Each topic file has:
- A one-paragraph **Contract** at the top: the rule in plain English.
- A **Why** section: enough context to remember why this is the rule.
- A **Tests / verification** section: what should fail if the contract is violated.
- A **Open exceptions / follow-ups** section: anything that doesn't quite conform yet, tracked in a plan or tracker.

Concise > comprehensive. Link out to plans/trackers/code for details.
