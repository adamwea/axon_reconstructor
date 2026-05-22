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
tradeoff per option + a recommended option) AND output the same question
to chat (so the user sees it without having to scroll the file), OR
process a user-picked answer from a prior iteration. The loop is a
question-curator + non-destructive answer-executor. NOT a code-shipper.

**Chat output discipline (CRITICAL)**: every iteration that surfaces a
question MUST do BOTH:
1. Write the full multiple-choice block to brain/open_questions.md (with
   label / touch-size / tradeoff / recommended fields). This is for
   persistence across sessions.
2. Call the `AskUserQuestion` tool with the SAME question. This renders
   as a clickable UI in Claude Code — the user picks via the UI, not by
   typing "option 1". Map open_questions.md's full block to the tool's
   option format: each option becomes `{label: <short label>, description:
   <touch-size + tradeoff one-line>}`. Put the Recommended option FIRST
   with "(Recommended)" suffix per the tool's convention.
   AskUserQuestion constraints: 2-4 options per question; label ≤ ~5
   words; description = one short paragraph (touch-size + tradeoff).
   The tool auto-adds an "Other" option for the user to type custom
   responses, so don't add one yourself.

Both writes happen in the same iteration. Then end the iteration
(no ScheduleWakeup needed — the AskUserQuestion answer fires the next
iteration automatically when the user picks).

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

Cadence (CRITICAL — different from autonomous mode):
- After surfacing a question to brain/open_questions.md, ALSO output
  the question text to the chat (with the multiple-choice block visible)
  AND DO NOT ScheduleWakeup. End the iteration cleanly. The user is at
  the keyboard; they will respond in chat, which fires the next
  iteration automatically via the normal /loop turn-handling. The
  600-1200s "blocked on user gate" cadence from loop_cadence.md is for
  autonomous mode (when the user is AWAY); in collaborative mode, the
  user is THERE and a 10-minute sleep defeats the purpose.
- ScheduleWakeup is ONLY appropriate in collaborative mode when:
  (a) the loop is mid-processing a user-picked answer and needs to come
      back to itself (e.g. read a few more files then continue) — use 60s
  (b) the user has explicitly stepped away ("brb 20 min") AND there's
      no question pending — use 600-1200s
  Default behavior is: end iteration after surfacing a question; wait
  for chat response.

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
