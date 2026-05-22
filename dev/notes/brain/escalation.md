# Escalation — perseveration limits + flexible-updating rules

Addresses the brain-theory "flexible updating / set-shifting" module. Without explicit escalation limits, the loop perseverates (retrying variations of the same failing approach indefinitely). With them, the loop has a defined "this isn't working, step back up" trigger.

**Reading rule for the loop**: at the END of every iteration that didn't make forward progress, check the rules below. If a limit is hit, escalate per the spec — do NOT continue with the same approach.

**Writing rule**: loop UPDATES the per-slice perseveration counters in `brain/notes.md` as it tracks attempts. New escalation rules require user approval via `open_questions.md` multiple-choice.

---

## Rule 1 — Same-fix perseveration limit (N=3)

**Trigger**: 3 consecutive iterations all attempting the SAME fix (same file edits, same approach, different parameters or off-by-one variations).

**Required response**: pop up to the slice's parent plan level. Surface a multiple-choice question:
- (a) Try a fundamentally DIFFERENT approach (loop proposes 1-2 alternatives based on what failed)
- (b) Split the slice into smaller pieces (the current slice may be too big)
- (c) Escalate to user (the loop has tried; user judgment needed)
- (d) Defer this slice; pick a different unblocked slice

**Recommended default**: (c) — escalate. The loop is the worst judge of "is this approach fundamentally wrong"; the user is the best judge.

**Counter location**: `brain/notes.md` under a heading like `## Perseveration counters` — per-slice, reset on first successful forward-progress iteration.

---

## Rule 2 — Stalled-slice limit (M=5)

**Trigger**: 5 iterations on the SAME slice without a commit landing.

**Required response**: brain self-audit + surface findings as multiple-choice:
- (a) The slice as specified is wrong — refine the plan spec
- (b) The slice is right but a prerequisite is missing — surface the prerequisite as its own slice
- (c) The slice is right but the loop is stuck on tooling/infra — surface the tooling blocker
- (d) Defer; pick a different slice

**Counter location**: `brain/notes.md` (same as Rule 1).

---

## Rule 3 — Stalled-plan limit (K=20)

**Trigger**: 20 iterations advancing slices in the SAME plan without a milestone landing (no SHIPPED slice that advances a DOD item in `brain/objectives.md`).

**Required response**: plan-coherence audit (per A1-A5 proactive plan audit injection). Surface as multiple-choice:
- (a) Plan is on track but slow — continue; recalibrate ETA
- (b) Plan is scope-creeping — propose splitting OR re-scoping
- (c) Plan is blocked on something not in the plan — surface the external blocker
- (d) Plan is wrong — abandon + replace

**Counter location**: `brain/notes.md` (top-of-file `## Plan progress counters` — per-plan).

---

## Rule 4 — Loop-wide thrash limit (T=50)

**Trigger**: 50 iterations across all plans without ANY objective DOD item landing.

**Required response**: alert the user. This is the "the loop has been busy but hasn't made meaningful progress" signal — likely indicates a fundamental misalignment between what the loop thinks it's doing and what advances the objectives.

The escalation surfaces as a top-of-`open_questions.md` `🛑 LOOP THRASH ALERT` block:
- Quantitative: which plans got iterated, what slices shipped, what objectives moved (or didn't)
- Qualitative: loop's best-guess diagnosis of why nothing's advancing
- Multiple-choice on next move: typically (a) pause loop, (b) drop low-priority plans from the queue, (c) re-spec objectives.

**Counter location**: `brain/notes.md` `## Loop-wide progress counter`. Resets when an objective DOD item completes.

---

## Rule 5 — Critic-rejection limit (per slice)

**Trigger**: critic subagent (per `brain/guardrails/critic_separation.md`) rejects 3 consecutive proposed commits for the same slice.

**Required response**: same as Rule 1 (the actor is stuck producing things the critic doesn't accept). Escalate.

This rule is what keeps the actor from rationalizing weaker and weaker fixes to satisfy the critic — at N=3 the loop concedes the design is wrong, not the implementation.

---

## How counters work mechanically

The loop maintains a small section in `brain/notes.md` like:

```
## Perseveration counters (auto-updated by loop)

### Active slices
- kssynth_recon_integration slice 3b heavy smoke: attempt 2/3 (Rule 1); slice iteration 4/5 (Rule 2)
- parallelism_post_migration slice 2: attempt 0/3; iteration 0/5

### Active plans
- kssynth_recon_integration: plan iteration 8/20 (Rule 3); last DOD movement: 2026-05-21 (slice 4e SHIPPED)
- resources_profiles_elimination: plan iteration 0/20; last DOD movement: none yet

### Loop-wide
- Iterations since last objective DOD: 12/50 (Rule 4)
- Last objective DOD: O1 milestone (slice 5 not yet reached)
```

Loop reads these at iteration start; updates at end. When a counter hits the limit, the loop's NEXT action MUST be the escalation per the rule, not the next slice attempt.

---

## Counter resets

| Event | Resets |
|---|---|
| Slice commits forward-progress code | Rule 1 + Rule 2 counters for that slice |
| Slice marked SHIPPED in plan | Rule 1 + Rule 2 counters; Rule 3 counter advances "last DOD movement" if relevant |
| Objective DOD item completes | Rule 4 counter resets |
| User overrides escalation (e.g. "keep trying, I think you're close") | The triggered rule's counter is bumped +5 grace iterations |

---

## When NOT to apply these rules

- **Long-running smoke**: a slice that's actively executing a multi-hour smoke run isn't "perseverating"; it's waiting. Counters should advance based on iteration count, not wall-clock.
- **User-explicitly-blessed approach**: if the user has greenlit a multi-step approach via a multiple-choice answer, the loop counts iterations against that approach as ONE attempt (not N attempts at "the same fix"). Counter advances when the user-blessed approach is itself rejected or abandoned.
- **Audit-pass iterations**: refinement work (prune stale entries, update brain files) does NOT count against the slice/plan counters. Only slice-advancement iterations count.

---

## Promotion when stable

After 5+ successful escalations resolved without user complaints about false-positive triggers, promote the numeric limits (N/M/K/T) into `brain/guardrails/loop_cadence.md` and reference here. The escalation file itself stays — the rules + responses are the load-bearing content.
