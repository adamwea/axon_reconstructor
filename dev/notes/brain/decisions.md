# Decisions — rationale log

Append-only record of decisions made + WHY. The semantic-memory layer for choices: lets future iterations (and fresh loop instances) answer "why did we pick X over Y" without rederiving from scratch OR blindly perpetuating without checking whether the rationale still holds.

**Reading rule for the loop**: when a current question's answer feels like it might revisit a past decision, search this file FIRST. If a decision here addresses your question, check whether its `Expected failure mode` has materialized — if yes, the decision is invalidated and you can propose revisiting; if no, the decision still holds.

**Writing rule**: append-only. Loop APPENDS one entry per multiple-choice resolution that locks in a non-trivial design decision (mode-switches, architectural picks, scope choices, sequencing locks, trust-anchor changes). Do NOT append for: routine slice-execution picks, typo fixes, doc refinements. The bar: would a future iteration benefit from knowing the rationale? If yes, append.

---

## Schema

```
### D-NNN — <one-line decision title>
- **Date**: YYYY-MM-DD
- **Source**: user msg / open_questions.md gate / commit hash / plan doc
- **Context**: 1-2 sentences — what surfaced this
- **Options considered**:
  1. <label> — <touch-size> — <tradeoff>
  2. <label> — <touch-size> — <tradeoff>
  (etc.)
- **Picked**: option N — <one-line rationale from user OR loop>
- **Expected failure mode**: "this decision is invalidated if X happens"
- **Resolution**: commit hash(es), plan slice ID, or "discussion only — no commit"
- **Status**: active | superseded by D-MMM (date) | invalidated (date, reason)
```

---

## Active entries

### D-001 — Use existing `plot_recons` phase for Radivojevic comparison (NOT a new renderer)
- **Date**: 2026-05-21
- **Source**: user msg + commit `4fb6957` + `a5996ea`
- **Context**: First Radivojevic SOFT-gate diagnostic landed with a renderer the loop invented in the sibling repo. User pushed back: comparison must use the SAME rendering code for both algorithms to isolate algorithmic differences from rendering differences.
- **Options considered**:
  1. Build adapter that converts `ReconstructionResult` → plot_recons-expected on-disk shape, then invoke plot_recons (Recommended)
  2. Invoke plot_recons' plotting helpers directly with both methods' outputs
  3. Accept loop's invented `render_reconstruction_png()` (REJECTED)
- **Picked**: option 1 — apples-to-apples requires identical rendering code path
- **Expected failure mode**: invalidated if plot_recons turns out to be fundamentally incompatible with radivojevic's output shape (e.g., hardcoded to `gtr.pkl` shape). At that point: STOP AND ASK rather than fall back to option 3.
- **Resolution**: discussion only — implementation pending; spec in `brain/open_questions.md` PRE-DIAGNOSTIC GATE 1
- **Status**: active

### D-002 — Reference data tree is BLANKET-pinned as Tier 1 trusted
- **Date**: 2026-05-21
- **Source**: user msg "pretty much anything in the reference data is pinned. New outputs should be identical or at least similar in shape to the current reference data."
- **Context**: T2 triage of 6 ranked trusted-output candidates. User generalized beyond the specific candidates to the whole reference tree.
- **Options considered**:
  1. Pin TR-CAND-001 + 003 + 006 individually (the cheap bucket)
  2. Blanket-pin the entire reference data tree (Recommended by user)
  3. Pin only TR-001 (the narrowest baseline)
- **Picked**: option 2 — captures the user's actual trust model; lets the loop derive structural invariants for ANY new output against the analogous reference
- **Expected failure mode**: invalidated if a specific reference-data file is later found to be wrong (e.g., a bug shipped into the reference). At that point: explicit downgrade of the file's TR-NNN entry to advisory, leaving the rest of the tree pinned.
- **Resolution**: commit `5ddf64d` (TR-000 added, TR-002/003/004 promoted)
- **Status**: active

### D-003 — resources_profiles_elimination slices 0-2 ship BEFORE parallelism_post_migration_cleanup slice 2
- **Date**: 2026-05-21
- **Source**: open_questions.md Finding #1 resolution + commit `488628f`
- **Context**: Two open plans touch J8 (resource budget). Parallelism slice 2 wires call sites that read `_budget.cpus_per_task` — which today comes from YAML profile (the user-flagged clamping bug), but will come from env after resources_profiles slice 2 ships. Order matters.
- **Options considered**:
  1. Ship resources_profiles 0-2 first; then parallelism 2 (Recommended)
  2. Ship parallelism 2 first; accept clamped behavior; fix later
  3. Combine into one plan
- **Picked**: option 1 — option 2 has the loop shipping work atop a known-broken behavior; option 3 risks scope blowup
- **Expected failure mode**: invalidated if resources_profiles plan turns out to need a parallelism-slice-2-shaped change in its own slice 2 (i.e., the dependency is bidirectional). At that point: pause + replan order.
- **Resolution**: commit `488628f` (See-also markers added to both plans)
- **Status**: active

### D-004 — Brain folder structure: fold memory/, guardrails/, refs/ into brain/
- **Date**: 2026-05-21
- **Source**: user msg "fold memory and guardrails as concepts into the brain scaffold... refs can live in there too"
- **Context**: After the brain skeleton was first built, user reviewed and recognized that existing memory/guardrails/refs dirs were all "self-regulation" content that conceptually belonged inside the brain.
- **Options considered**:
  1. Flatten all into brain/ as siblings of objectives/metrics/etc
  2. Fold but preserve subdirs: brain/memory/, brain/guardrails/, brain/refs/
  3. Leave memory/guardrails/refs separate; brain/ stays small
- **Picked**: hybrid — flat for memory (only 4 files, no need for subdir); subdir for guardrails + refs (cohesive sets) — per loop's read of user's "fold" framing
- **Expected failure mode**: invalidated if the brain/ directory becomes hard to navigate. Re-split if so.
- **Resolution**: commit `77d38fc` (41-file rename)
- **Status**: active

### D-005 — Two-mode loop architecture: extended_autonomous + collaborative + paused
- **Date**: 2026-05-21
- **Source**: user msg "I want to have two looper modes going forward. Extended Autonomous and Collaborative"
- **Context**: Recurring context drift in autonomous mode led to wanting a "ask-before-doing" mode for periods when many open questions need user attention before code shipping is safe.
- **Options considered**:
  1. Two modes (autonomous + collaborative) with paused as a transient state (Recommended)
  2. Mode spectrum (continuous "boldness" parameter)
  3. Single mode with finer-grained per-action permissions
- **Picked**: option 1 — discrete modes are easier to reason about; user can transition explicitly
- **Expected failure mode**: invalidated if collaborative mode turns out to need sub-modes (e.g., "collaborative-design" vs "collaborative-execute"). At that point: add modes; the file structure supports it.
- **Resolution**: commit `240908c` (brain/mode.md + loop_prompts/collaborative.md + critic_separation + escalation)
- **Status**: active

### D-015 — Lift the pause; transition collaborative → extended_autonomous (post-2-audit-rounds)
- **Date**: 2026-05-21
- **Source**: QZ11 user pick "option 1" via AskUserQuestion
- **Context**: After phase zero (Z1+Z2+Z3) + GATE 1 approval + 2 plan-audit rounds (6 findings shipped), backlog was genuinely drained. User picked lift-now vs. risking audit-pass drift.
- **Picked**: option 1 — lift to extended_autonomous; loop's first autonomous work = GATE 1 per D-010
- **Expected failure mode**: invalidated if a friction point in GATE 1 reveals the gate spec is wrong AND requires a return to collaborative. At that point: stop-and-ask surfaces it; if user transitions back to collaborative, the cycle restarts.
- **Resolution**: (a) loop_prompts/extended_autonomous.md PAUSE NOTICE removed inline (not a round bump — stale-block cleanup); (b) user edits brain/mode.md ACTIVE_MODE to extended_autonomous + fires fresh /loop session
- **Status**: active

### D-014 — Execute all 3 round-2 audit findings (F#5 TODO refresh + F#6 dashboard_audit historicize + F#7 AP-013/014 anti_patterns)
- **Date**: 2026-05-21
- **Source**: QZ10 user pick multi-select [F#5, F#6, F#7] via AskUserQuestion
- **Picked**: act on all 3; touch S each; addresses stale references + missing tribal knowledge captures
- **Resolution**: TODO.md refreshed; dashboard_audit historicized; anti_patterns AP-013+014 appended
- **Status**: active

### D-013 — Run second audit pass before lifting pause (refs/, anti_patterns, TODO.md surfaces)
- **Date**: 2026-05-21
- **Source**: QZ9 user pick "option 2" via AskUserQuestion
- **Context**: After QZ8 shipped 3 plan-audit findings, user wanted one more audit pass on different surface areas before lifting pause.
- **Picked**: option 2 — second audit catches finding gaps the first didn't cover
- **Resolution**: surfaced 3 more findings (F#5 stale TODO.md, F#6 stale dashboard_audit cross-ref, F#7 missing AP-013+AP-014 in anti_patterns); QZ10 batched multi-select pending
- **Status**: active

### D-012 — Execute all 3 plan-audit findings (F#2 plan-completion sweep + F#3 kssynth slice 5 spec amend + F#4 radivojevic slice ref fix)
- **Date**: 2026-05-21
- **Source**: QZ8 user pick multi-select [F#2, F#3, F#4] via AskUserQuestion
- **Context**: D-011's audit surfaced 3 findings. User picked all 3 to act on (vs deferring).
- **Picked**: act on all 3 findings — minimal touch sizes; addresses real coherence drift before pause lifts; sets autonomous loop up to find correct cross-plan references.
- **Expected failure mode**: invalidated if F#2's plan move breaks a cross-reference the loop didn't catch (the sed audit was scoped to `dev/notes/` + `CLAUDE.md`; commit_log references are intentionally left as historical). If a future iteration hits a broken plan ref, restore + fix.
- **Resolution**: 4 plans moved active→completed; kssynth slice 5 amended; radivojevic kick-off trigger fixed
- **Status**: active

### D-011 — Run proactive plan audit before lifting pause
- **Date**: 2026-05-21
- **Source**: QZ7 user pick "option 4" via AskUserQuestion
- **Context**: After QZ6 approved GATE 1, backlog appeared drained. Loop offered to lift the pause but user wanted a fresh proactive plan audit first (per A1-A5 discipline) to ensure backlog was truly drained vs just appears drained.
- **Options considered**:
  1. Lift now → extended_autonomous
  2. Lift but pin a different first task
  3. Stay collaborative — surface new items
  4. Run a proactive plan audit first (user picked)
- **Picked**: option 4 — checks that no coherence issues are about to bite after the pause lifts
- **Expected failure mode**: invalidated if audit surfaces NO findings (in which case option 1 was the right call). Audit-iteration time is the cost.
- **Resolution**: audit walked plans/active/*.md; 3 findings surfaced as QZ8 batched multi-select (Finding #2 plan-completion sweep; Finding #3 kssynth slice 5 GATE 1 dep; Finding #4 radivojevic stale slice ref)
- **Status**: active

### D-010 — PRE-DIAGNOSTIC GATE 1 (radivojevic apples-to-apples) approved as-written for post-pause autonomous execution
- **Date**: 2026-05-21
- **Source**: QZ6 user pick "option 1" via AskUserQuestion
- **Context**: GATE 1 was the biggest unresolved item in `brain/open_questions.md`. Its 9-step plan body was spec'd 2026-05-21; one prereq (--input-root analyzers extension, commit `c8b8b11`) already shipped. With phase zero complete + Z3 specs approved, GATE 1 became actionable.
- **Options considered**:
  1. Approve as-written; execute post-pause (user picked)
  2. Refine plan first (specify which step)
  3. Slim down to numerical-only comparison
  4. Table indefinitely
- **Picked**: option 1 — plan is well-spec'd; STOP-AND-ASK guards on the 3 anticipated friction points (unit-ID mapping, plot_recons adapter, empty kssynth output) prevent silent improvisation
- **Expected failure mode**: invalidated if a friction point hits + the loop's STOP-AND-ASK response surfaces a question that requires re-spec'ing significantly more of the plan than the friction point implies. At that point: re-open GATE 1 as a multi-part question.
- **Resolution**: PRE-DIAGNOSTIC GATE 1 section in `brain/open_questions.md` marked APPROVED inline; awaits autonomous execution post-pause-lift
- **Status**: active

### D-009 — Stay in collaborative mode after phase zero closes; resolve backlog before lifting pause
- **Date**: 2026-05-21
- **Source**: QZ5 user pick "option 2" via AskUserQuestion
- **Context**: Phase zero closed with Z3 specs approved (D-008). Natural moment to lift the pause and transition to extended_autonomous. User chose to drain the open_questions backlog first instead.
- **Options considered**:
  1. Lift now → extended_autonomous (resume autonomous execution)
  2. Stay collaborative — resolve backlog first (user picked)
  3. Lift with specific first task pinned
  4. Lift in 'audit-only' sub-mode (would require defining a new mode)
- **Picked**: option 2 — backlog items (PRE-DIAGNOSTIC GATE 1 being the biggest) deserve explicit user picks before autonomous execution touches them. Avoids the loop self-interpreting plan specs that haven't been formally blessed.
- **Expected failure mode**: invalidated if the collaborative-mode question backlog turns out to never drain (each resolution surfaces more questions ad infinitum). At that point: surface a meta-question asking whether the user wants to skip remaining backlog + lift anyway.
- **Resolution**: brain/mode.md ACTIVE_MODE remains `collaborative`; loop continues surfacing one question at a time via AskUserQuestion
- **Status**: active

### D-008 — Z3 invariant specs approved 2026-05-21 as-written
- **Date**: 2026-05-21
- **Source**: QZ4 user pick "option 1" via AskUserQuestion
- **Context**: After QZ3 authored 5 Z3-TR-xxx invariant spec blocks, user reviewed + approved all 5 without pushback (no per-block adjustments, no format change request, no defer-to-pytest).
- **Options considered**:
  1. Approve all 5 as-written (user picked)
  2. Approve some, push back on others
  3. Defer batch, request different spec format
  4. Defer until pytest-first
- **Picked**: option 1 — fast path to closing Z3; user trusts spec quality to be sufficient
- **Expected failure mode**: invalidated if a future autonomous-mode pytest implementation discovers a Z3-TR-xxx spec is ambiguous OR materially wrong on real data. At that point: amend the spec via a new decision; pytest gets the fixed version.
- **Resolution**: commit (this commit) marks each Z3-TR-xxx block APPROVED. Z3 phase-zero step closed.
- **Status**: active

### D-007 — Z3 invariants authored as markdown specs in `brain/trusted_outputs.md` (NOT as pytest yet)
- **Date**: 2026-05-21
- **Source**: QZ3 user pick "option 1"
- **Context**: Z3 of the phase-zero mission says "loop authors invariant-based tests" but collaborative mode forbids new test additions. Reconciliation question surfaced as QZ3; user picked option 1.
- **Options considered**:
  1. Spec-first as markdown (Recommended; user picked)
  2. Audit existing tests first
  3. Mode transition: collaborative → autonomous to author pytest directly
  4. Hybrid (spec + audit in parallel)
- **Picked**: option 1 — Loop authored Z3-TR-000 / Z3-TR-001 / Z3-TR-002 / Z3-TR-003 / Z3-TR-004 invariant specs as structured markdown sections in `brain/trusted_outputs.md`. User reviews + approves SPECS (fast — read claims, not data). Pytest implementation deferred to a follow-up autonomous-mode session.
- **Expected failure mode**: invalidated if (a) markdown specs turn out to be ambiguous when the autonomous-mode session tries to convert them to pytest (i.e. the spec → code translation needs more detail), OR (b) the specs reveal a reference-data inconsistency the user wants to fix before pinning. At that point: revise specs OR adjust the reference data.
- **Resolution**: commit (this commit) authors specs; QZ4 surfaces the approval gate
- **Status**: active

### D-006 — Add 4 reasoning-consistency brain components (decisions, glossary, anti_patterns, slice_contracts Prediction field)
- **Date**: 2026-05-21
- **Source**: user msg "Any other brain pieces we could add for better reasoning consistency?"
- **Context**: After mode + escalation + critic_separation landed, gap analysis identified 4 more pieces specifically targeting cross-session reasoning consistency (rather than just executive function or data-pipeline soundness).
- **Options considered**:
  1. Build all 4 (Recommended; user picked)
  2. Just decisions.md + glossary.md (the two most critical)
  3. Defer all — see if existing brain is sufficient
- **Picked**: option 1 — small files, mostly schema + initial population; high value for fresh-loop handoff
- **Expected failure mode**: invalidated if any of the 4 turns out to fragment more than consolidate (e.g., decisions.md duplicates commit_log effort). Prune the redundant file if so.
- **Resolution**: commit (this commit)
- **Status**: active

---

## Superseded / invalidated entries

*(empty — when a decision gets superseded by a new one, mark the old one's Status field with the link to the new decision; keep the body for history)*
