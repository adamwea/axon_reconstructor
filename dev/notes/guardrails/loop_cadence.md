# Loop cadence + heartbeat reason format guardrail

## Contract

The loop's `ScheduleWakeup.reason` field is the cadence message the user
sees between iterations. It must answer two questions at a glance:
**how long** until the next iteration, and **what's queued** (or being
watched) for that iteration. The cadence itself must match what's
actually happening — short when actively iterating, longer when truly
idle, never the old 1200-1800s default unless genuinely blocked.

## Why

Opaque reason strings like "heartbeat armed" or "idle" don't tell the
user anything actionable. The cadence delay is a free signal: pairing
it with one specific sentence about what's queued lets the user pace
their own attention to the loop without checking logs. Shorter cadence
also lowers per-iteration cost when there's real work to ship.

Promoted from `current_state.md` USER INJECTION 2026-05-21 after the
loop applied the format reliably for ≥2 consecutive iterations
(commits `8a0b289`, `321c75f`, `df6a895`, `7dabc1a`, `2ce4829`).

## Concrete sub-rules

### 1. Reason format (mandatory)

`"Next iteration in {N}s — {one short sentence on what's queued or being watched}"`

Examples (all real, from the rollout iterations):

- `"Next iteration in 90s — finishing dry_run_rollout slice 4 spikesort sweep"`
- `"Next iteration in 120s — slice 4 COMPLETE; next slice TBD (plan sync or pivot to tracker)"`
- `"Next iteration in 120s — slice 3b SHORT-PATH smoke confirmed; next slice TBD after empirical validation"`
- `"Next iteration in 600s — queue empty, doing audit pass on plans/active/"`
- `"Next iteration in 1200s — genuinely idle, no actionable slices until user reviews Radivojevic gate"`

The sentence after the dash must name a concrete next-iteration anchor
— a plan + slice, a phase + smoke, a tracker entry, a gated user
decision. Avoid jargon-free phrases like "heartbeat armed" or "armed
for next" — those don't tell the user what's coming.

### 2. Default cadence ladder

| Loop state | delaySeconds | When to pick |
|---|---|---|
| Actively iterating (mid-slice + ready to commit) | **90s** | Within the 5-minute cache window. Low per-iteration cost. Pick this when the next slice is concrete + ready to start. |
| Between slices, queue full | **120s** | Just shipped a slice; next slice candidate identified but not yet started. Still cache-window-friendly. |
| Queue empty, audit-pass mode | **300s** | Nothing concrete is queued; doing plan/memory cleanup. Cache miss is OK because work isn't pressing. |
| Genuinely blocked on user gate | **600-1200s** | Plan-level gate (e.g. Radivojevic slice 3 needs user review) — no actionable slices. Let the user breathe. |
| Polling external state (CI, allocation, deploy) | **270s** | Stay inside the cache window. Don't poll at 60s if state changes minute-to-minute. |

### 3. NEVER use the bare 1200-1800s as a default

That was the old idle-tick cadence. The loop should only reach 1200s+
when truly blocked (user gate, allocation wait, CI run). Reaching that
without a specific reason is a smell — pick a shorter cadence and
let the next iteration's state inform the cadence after.

### 4. Cadence reflects iteration intent, not arbitrary preference

If you just shipped a slice and the next slice is ~10 min of work,
the FIRST follow-up wakeup is the 90s "actively iterating" cadence.
The user's mental model is that the loop iteration corresponds to a
slice of progress — short cadence between slices means the loop is
moving.

If the next slice is gated on something external (data-routing
decision, user gate), bump to 600-1200s with a reason that names
the gate.

## Tests / verification

This guardrail isn't directly testable in code — it's a behavioral
contract on `ScheduleWakeup` calls. Verification is by review of
recent loop iterations:

- Last 5 ScheduleWakeup calls all match the reason format ✓
- Cadence chosen matches the loop state ✓
- No calls with delaySeconds in [1200, 1800] without a "blocked on
  X" reason ✓

## Open exceptions / follow-ups

- The `<task-notification>` path in the /loop skill says to reset the
  wakeup with the SAME parameters. With this new cadence ladder, that
  guidance should be: keep the same `prompt`, but pick a new
  `delaySeconds` + `reason` matching the current loop state (the
  notification itself may have changed what's queued).
- Per the original 2026-05-21 injection's "Promote when stable" note,
  the loop-prompt template the user pastes when invoking `/loop` should
  also be updated to bake in this cadence ladder. That's a user-side
  edit; surface as a "User actions queued" item until done.
