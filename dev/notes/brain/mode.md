# Mode — active loop mode + permitted action set

The mode router. Read FIRST every iteration, before `objectives.md`. Determines what kinds of actions the loop is allowed to take this iteration.

**Editing rule**: USER-ANCHORED. Loop may PROPOSE a mode change via multiple-choice in `open_questions.md` but NEVER edits this file without explicit user direction. Mode transitions are user-initiated.

---

## Active mode

```
ACTIVE_MODE: collaborative
SINCE: 2026-05-21
SET_BY: user
NEXT_MODE_HINT: collaborative (per user 2026-05-21 plan)
```

To switch modes, the user edits the `ACTIVE_MODE` line above to one of: `extended_autonomous` | `collaborative` | `paused`. The `SINCE` + `SET_BY` lines should be updated alongside.

---

## Mode definitions

### `extended_autonomous`

The loop ships code, runs smokes, advances plans, opens PRs (sibling repos), triggers shifter rebuilds — everything currently in the standing `/loop` prompt's authorization block (per `loop_prompts/extended_autonomous.md`).

**Permitted action set**:
- ✅ Read/write any file in the repo (except axon_recon git push — still prohibited)
- ✅ `src/` code changes
- ✅ Login-node smokes on REAL DATA (per smoke_log discipline)
- ✅ Shifter rebuilds (`podman build` → push → `shifterimg pull`)
- ✅ Sibling-repo git pushes on well-named branches; `gh repo create`
- ✅ Update brain/ files as side-effect of slice work (slice_contracts append, dependency_graph propagation, metrics baseline updates)
- ❌ axon_recon git push (always prohibited)
- ❌ slurm sbatch submissions (always prohibited)
- ❌ `gh pr merge` (user-only)

**Iteration cadence**: per `brain/guardrails/loop_cadence.md` ladder (90s active / 120s between slices / 270s audit / 600-1200s blocked).

**Goal per iteration**: advance the highest-priority unblocked slice from the queued plan tier.

### `collaborative`

The loop reads, audits, asks. It does NOT ship code or run side-effecting commands without an explicit user pick first.

**Permitted action set**:
- ✅ Read any file in the repo (code, plans, commits, brain)
- ✅ Write to `brain/open_questions.md` (multiple-choice questions per B1/B2)
- ✅ Write to `brain/current_state.md` (state updates, prune stale entries)
- ✅ Write to `brain/notes.md` (scratch / debugging trails)
- ✅ Write to `dev/notes/TODO.md` (append new tasks; mark items resolved)
- ✅ Write to `dev/notes/plans/active/*.md` (refine plan text, mark slices SHIPPED/SUPERSEDED, add See-also sequencing notes)
- ✅ Audit-pass work (plan-coherence findings, dependency_graph updates, retire stale injections)
- ✅ Update brain backbone files (objectives, dependency_graph, trusted_outputs, metrics, slice_contracts) AS A SIDE-EFFECT of executing a user-picked answer
- ❌ Any `src/` code change
- ❌ Any smoke run (real-data OR dry-run)
- ❌ Any sbatch / srun / shifter rebuild
- ❌ Any git push (sibling repos included — defer to autonomous mode)
- ❌ New test additions (tests are code changes)
- ❌ Bumping the /loop prompt round

**Iteration cadence**: slower — typical iteration is "read state → identify next decision point → write multiple-choice question → ScheduleWakeup to wait for user response". When the user responds, the next iteration processes the pick. Cadence ladder still applies but the "actively iterating" tier means "actively drafting questions", not shipping code.

**Goal per iteration**: surface ONE high-leverage decision point as a multiple-choice question in `open_questions.md`, OR process a user-picked answer from the previous iteration. The loop is a question-curator + answer-executor (where the answer is non-destructive), NOT a code-shipper.

**When to use**: when there are open design questions worth resolving before committing to autonomous execution. The current state (2026-05-21) is the prototypical case: plans have churn-risk relationships, brain components are being added, the user wants to solidify before letting the loop loose.

### `paused`

The loop does nothing user-action-related. It MAY do meta work but must NOT touch any plan / phase / smoke surface area.

**Permitted action set**:
- ✅ Brain refinement (this file, READMEs, schema cleanup) — explicitly user-directed
- ✅ Audit / inventory tasks (plan coherence, brain self-audit, dependency graph updates from prior commits)
- ✅ Test additions ONLY if explicitly user-directed and limited to refining existing test infrastructure
- ❌ Real-data smokes
- ❌ New phase implementations
- ❌ /loop prompt round bumps
- ❌ Mode transitions (user lifts the pause by editing `ACTIVE_MODE` above)

**When to use**: when refinement work is needed before either autonomous or collaborative work can proceed safely. Typically a short window.

---

## Mode-check discipline (CLAUDE.md slice protocol step 0)

Every loop iteration, before doing anything else, the loop checks `ACTIVE_MODE` above. If the planned action for this iteration is NOT in the active mode's permitted set, the loop:

1. STOPS the planned action
2. Surfaces a multiple-choice question in `brain/open_questions.md`: "Action X requires mode Y; current mode is Z. Options: (1) user transitions to mode Y, (2) loop picks a different action allowed in mode Z, (3) defer."
4. Waits for user direction

This is the inhibition / interference-control module from the brain theory — the "selection mechanism" that decides what gets into the limited workspace.

---

## Mode transition rules

- **User-initiated only**: loop never edits `ACTIVE_MODE` autonomously.
- **Wholesale transitions**: no mid-iteration mode flip. The loop's current iteration completes (or halts cleanly) before a new mode takes effect on the next iteration.
- **Pause is absorbing**: from autonomous or collaborative, anyone can drop to paused. From paused, only the user can lift.
- **Promotion criteria** (informal): collaborative → autonomous when the user is confident the open-question backlog is processed + brain components are stable. autonomous → paused when context drift or coherence issues warrant a refinement stop (as happened 2026-05-21).
