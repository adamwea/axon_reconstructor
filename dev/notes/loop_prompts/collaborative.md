# Collaborative loop prompt

The "ask before doing" `/loop` invocation prompt. Use this when there are open design questions worth resolving before committing to autonomous execution. Defined as a brain mode in `brain/mode.md`.

**How to use:**
1. Open Claude Code in this repo.
2. Type `/loop` and paste the fenced block below as the loop instructions.
3. The loop reads `CLAUDE.md` → `brain/mode.md` (sets `ACTIVE_MODE: collaborative` if not already) → `brain/objectives.md` → identifies the highest-leverage open decision → surfaces it as multiple-choice in `brain/open_questions.md` → ScheduleWakeup'd to wait for user response. When the user answers, the next iteration executes the pick (if non-destructive) + identifies the next decision.

**Mode-mode**: this prompt corresponds to `ACTIVE_MODE: collaborative` in `brain/mode.md`. The loop checks the mode at iteration start and refuses actions outside the collaborative permitted-action-set.

**When to switch from collaborative → autonomous**: when the open-question backlog is small + the brain components feel stable + the user is confident the loop has the verifier scaffold to ship code safely.

**Revision history**: tracked by `git log dev/notes/loop_prompts/collaborative.md`.

---

## Current — Round 1 (2026-05-21)

```
COLLABORATIVE MODE (round 1) — question-curator + answer-executor.

Read /global/homes/a/adammwea/dev/pkgs/axon_recon/CLAUDE.md and follow its
entry protocol. First file you read: brain/mode.md. If ACTIVE_MODE is not
`collaborative`, surface a multiple-choice question asking the user
whether to transition the mode, and HALT until they answer. Do NOT
proceed with collaborative-mode work until ACTIVE_MODE confirms.

Permitted action set (per brain/mode.md `collaborative` definition):
- ✅ Read any file in the repo (code, plans, commits, brain)
- ✅ Write to brain/open_questions.md (multiple-choice questions per B1/B2)
- ✅ Write to brain/current_state.md (state updates, prune stale entries)
- ✅ Write to brain/notes.md (scratch, debugging trails, perseveration counters)
- ✅ Write to dev/notes/TODO.md (append new tasks; mark items resolved)
- ✅ Write to dev/notes/plans/active/*.md (refine plan text, mark slices SHIPPED/SUPERSEDED, add See-also sequencing notes)
- ✅ Audit-pass work (plan-coherence findings, dependency_graph updates from prior commits, retire stale injections)
- ✅ Update brain backbone files (objectives, dependency_graph, trusted_outputs, metrics, slice_contracts) AS A SIDE-EFFECT of executing a user-picked answer
- ❌ Any src/ code change
- ❌ Any smoke run (real-data OR dry-run)
- ❌ Any sbatch / srun / shifter rebuild / podman build
- ❌ Any git push (sibling repos included — defer to autonomous mode)
- ❌ New test additions (tests are code changes)
- ❌ Bumping the /loop prompt round
- ❌ Editing brain/mode.md or brain/objectives.md without user direction

Iteration goal: surface ONE high-leverage decision point as a
multiple-choice question in brain/open_questions.md (label / touch-size /
tradeoff per option + a recommended option), OR process a user-picked
answer from a prior iteration. The loop is a question-curator + non-
destructive answer-executor. NOT a code-shipper.

Question selection heuristic (which decision to surface each iteration):
1. Resolved blockers > new questions: if open_questions.md has a question
   the user has marked `✅ USER APPROVED option N`, EXECUTE that pick
   first (if non-destructive) before surfacing new questions.
2. Highest-leverage > most-recent: prefer questions that unblock the
   most downstream work. Use brain/dependency_graph.md §5 (ranked
   verification checkpoints) + plan tier order to estimate leverage.
3. Coherence findings > forward planning: if proactive plan-audit
   (per A1-A5) surfaces a contradiction or stale assumption, that
   trumps forward-planning questions until resolved.
4. One at a time: surface ONE question per iteration, not a batch.
   The user's cognitive load is the bottleneck, not the loop's
   throughput.

Question format (mandatory — per stop-and-ask + multiple-choice
discipline B1/B2):
```
### <question id> — <one-line title>
- **Why now**: <one sentence on what surfaced this>
- **Context**: <one paragraph of background; link to relevant brain/* + plan files>

**Pick an option:**

1. **<label> (Recommended)** — <touch-size> — <tradeoff>
2. **<label>** — <touch-size> — <tradeoff>
3. **<label>** — <touch-size> — <tradeoff>
4. (optional) **<label>** — <touch-size> — <tradeoff>

**Loop recommendation**: option N because <one sentence>.
```

Cadence: per brain/guardrails/loop_cadence.md, but the "actively
iterating" tier (90s) means "actively drafting one question" — not
shipping code. After surfacing a question, expect to ScheduleWakeup
in the 600-1200s "blocked on user gate" tier and wait for response.

Stop conditions:
- User says "stop" / "wake up" / "wrap up"
- Open-questions backlog is genuinely empty (no unresolved questions
  AND no obvious next-question to surface). At this point, the loop
  surfaces ONE question: "Backlog appears clear. Options: (1) switch
  to autonomous mode, (2) do an audit pass on a specific plan, (3)
  pause." User picks.

Standing constraints (NOT relaxed):
- Working-data scope: 80k DMEM well000 of M08073, all DIVs.
- Reference data tree is read-only (per brain/trusted_outputs.md TR-000).
- All guardrails apply. Read brain/guardrails/<topic>.md whenever a
  question's surface area touches one of them.

Default to Opus 4.7; Sonnet 4.6 only for very concrete mechanical
question-drafting. Continue iterating until I explicitly stop or
ACTIVE_MODE changes.
```

---

## Promotion criteria

After 10+ collaborative-mode iterations without process complaints from the user, consider promoting the prompt to v2 with refinements based on what worked + what didn't. Until then, this is round 1.
