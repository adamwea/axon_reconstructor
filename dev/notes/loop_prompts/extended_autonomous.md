# Extended autonomous loop prompt

The fully-autonomous `/loop` invocation prompt. Paste the fenced block below into `/loop` when the user is genuinely AWAY (overnight, multi-hour unsupervised runs) and wants the loop to maximize forward motion without interrupting them.

**Key behavioral difference from `autonomous_collaborative.md`**:
- On friction with explicit user instructions: **autonomous_collaborative** asks the user via `AskUserQuestion` and waits.
- **extended_autonomous** (this prompt): **PIVOTS** to another unblocked slice from another plan; files the block to `brain/open_questions.md` as multi-choice for the user to see whenever they next check in; does NOT block on `AskUserQuestion`. The user discovers blocks during their next status check, not via real-time interruption.

The "extended" framing comes from: the loop runs EXTENDED periods unsupervised without needing the user's keyboard. It still respects every other brain discipline (anti-improvising, critic-separation, slice-contracts, trusted-output anchors, etc.) — just doesn't pause for synchronous user input.

**Sibling prompt**: `loop_prompts/autonomous_collaborative.md` — the asks-when-blocked variant. Use that one when user is nearby.

**How to use:**
1. Open Claude Code in this repo.
2. Type `/loop` and paste the fenced block below as the loop instructions.
3. The loop reads CLAUDE.md → `brain/objectives.md` → backbone files → current_state → relevant guardrails → plan, then advances the highest-leverage unblocked slice. On friction it pivots; on HARD-gates it pauses + files a diagnostic for review; on objective DOD completion it surfaces a summary.

**Revision history**: tracked by `git log dev/notes/loop_prompts/extended_autonomous.md`.

**When to bump the round**: when authorizations change OR when the pivot-instead-of-ask discipline needs refinement based on observed failure modes.

---

## Current — Round 1 (2026-05-21)

```
EXTENDED AUTONOMOUS MODE (round 1) — fully autonomous; pivots on friction.

Read /global/homes/a/adammwea/dev/pkgs/axon_recon/CLAUDE.md and follow its
entry protocol. The mode IS this prompt: you're in extended-autonomous
mode because the user pasted THIS prompt (not autonomous_collaborative).
No central mode file to check.

Permitted action set (extended_autonomous — full autonomy):
- ✅ Read/write any file in the repo (except axon_recon git push — still prohibited)
- ✅ src/ code changes
- ✅ Login-node smokes on REAL DATA (per smoke_log discipline)
- ✅ Shifter rebuilds (podman build → push → shifterimg pull)
- ✅ Sibling-repo git pushes on well-named branches; `gh repo create`
- ✅ Update brain/ files as side-effect of slice work (slice_contracts
  append, dependency_graph propagation, metrics baseline updates)
- ❌ axon_recon git push (always prohibited)
- ❌ slurm sbatch submissions (always prohibited)
- ❌ `gh pr merge` (user-only)
- ❌ Editing brain/objectives.md without user direction
- ❌ Editing brain/trusted_outputs.md Tier 1 entries (user-anchored)

Friction-handling discipline (KEY DIFFERENCE from autonomous_collaborative):
- When a slice hits friction with an EXPLICIT user instruction
  (e.g. "use existing X", "pick high-branch unit", "produce comparison"),
  DO NOT call AskUserQuestion. Instead:
  1. File a multi-choice block under brain/open_questions.md (same
     format as autonomous_collaborative would — label / touch-size /
     tradeoff / Recommended). User will see it on next status check.
  2. PIVOT to another unblocked slice from any queued plan. The
     friction-bearing slice stays blocked + filed; loop resumes
     forward motion elsewhere.
  3. Use the proactive-plan-audit ranking (brain/dependency_graph.md
     §5) to pick the next-best slice.
- PRE-DIAGNOSTIC GATEs (B2 from earlier injections): in extended_autonomous,
  these are NO LONGER pre-gated. Loop generates diagnostics autonomously
  per the strict R1-R5 visual-diagnostics rule + critic_separation
  for code-shipping slices. HARD-gate diagnostics still pause for user
  review (file in brain/diagnostics_to_review.md); SOFT-gate diagnostics
  ship without pausing.
- Anti-improvising rule (AP-001) STILL APPLIES: when friction hits, do
  NOT substitute (different unit, different tool, different approach).
  File the block + pivot to a DIFFERENT slice, not a substitute approach
  for the same slice.

Relaxed stance (applies until user explicitly says stop):
- NEVER stop on a blocker within ~15 min. git restore, write block to
  brain/open_questions.md, PIVOT.
- Stop only when: (a) user says "stop"/"wake up"/"wrap up", or
  (b) ALL plans/trackers genuinely exhausted, or (c) a HARD-gate
  diagnostic lands (pauses for user review per visual-diagnostics R2).
- Queue empty? Audit pass (refine plans, prune stale notes, write
  next-tier tracker entries, surface proactive plan-audit findings).
  Don't idle.
- Login-node smokes on REAL DATA encouraged. Limits: --task-backend
  local_affinity, cap 64 procs, --limit-* flags. NO slurm submissions.
  Bigger smokes → salloc command into trackers/salloc_smokes_queued.md
  + keep going.
- Commit often + restore often.

Loop cadence (per brain/guardrails/loop_cadence.md):
- Heartbeat ScheduleWakeup.reason MUST be:
  "Next iteration in {N}s — {one sentence on what's queued/being watched}"
  NEVER bare "heartbeat armed".
- Cadence ladder:
  - Actively iterating mid-slice: 90s
  - Between slices, next candidate identified: 120s
  - Queue empty / audit-pass mode: 270s
  - HARD-gate diagnostic landed, waiting for review: 600–1200s
- In extended_autonomous, the "blocked on user gate" tier is rarer
  (most blocks → pivot, not wait). 600-1200s is reserved for genuine
  HARD-gate pauses + the audit-pass-without-actionable-work state.

Phase-enable for tests rule (USER INJECTION 2026-05-21):
- Smoke-testing a YAML-disabled phase MUST enable it for the smoke —
  `--force-enable PHASE` CLI flag is the canonical mechanism. Never
  validate a phase by running it disabled.

Real-data smoke log discipline (USER INJECTION 2026-05-21):
- After every smoke on REAL DATA (login-node OR user-initiated salloc),
  append an entry to dev/notes/trackers/smoke_log.md per its schema —
  including bug→fix-commit chains.
- Do NOT log dry-run smokes / unit tests / synthetic-fixture smokes.
- Worker-count validation: BEFORE each smoke, record EXPECTED worker
  counts; DURING/AFTER scan logs for actual; mismatch = smoke_log MUST
  flag it. Env over-request → FAIL FAST.

Diagnostic discipline (USER INJECTIONS 2026-05-21):
- ANY slice changing user-visible rendering MUST file a diagnostic
  entry in brain/diagnostics_to_review.md. Multi-stage algorithms file
  at EACH stage transition.
- Diagnostics MUST include a rendered image (PNG/SVG/PDF) as the
  user-visible artifact. npy / tsv / parquet ALONE is data, not a
  diagnostic.
- SOFT-gate diagnostics ship without pausing the loop.
- HARD-gate diagnostics PAUSE the loop for user review (file to
  diagnostics_to_review.md; ScheduleWakeup to 600-1200s).

Critic separation (brain/guardrails/critic_separation.md):
- Every code-shipping slice MUST spawn a separate Explore subagent
  as critic before commit. Narrow scope (diff + trusted-output fixture
  + invariants). Pass/fail/concerns. Block commit on fail.

Slice contract discipline (per CLAUDE.md slice protocol step 9):
- Every shipped slice MUST append an entry to brain/slice_contracts.md
  with Produces / Assumes / Propagates / Prediction / Actual / Delta.
  Prediction filled BEFORE actor work starts.

Perseveration escalation (brain/escalation.md):
- N=3 same-fix retries / M=5 stalled-slice / K=20 stalled-plan /
  T=50 loop-wide thrash → escalate (file to open_questions.md;
  pivot if possible; only HARD-stop if truly nothing to pivot to).
- Critic-rejection limit (Rule 5): N=3 critic rejections same slice
  → escalate (filing + pivot).

Standing constraints (NOT relaxed):
- Working-data scope: 80k DMEM well000 of M08073, all DIVs.
- /pscratch/.../analyzed_data/ is read-only reference data (TR-000
  blanket-pinned). Iteration outputs go to
  /pscratch/sd/a/adammwea/dev_outputs/<slice>/.
- Code lives in /global/homes. Podman GraphRoot lives in
  /pscratch/sd/a/adammwea/podman_storage/.
- Sample rate is per-recording (MaxTwo 10 kHz current cohort,
  MaxOne 20 kHz, ThreeBrain/Sony TBD future). NEVER hardcode sample
  rate — read from analyzer manifest via metadata_get.
- All guardrails apply.

Default to Opus 4.7; Sonnet 4.6 only for very concrete mechanical work.
Continue iterating until user explicitly stops.
```

---

## Promotion criteria

After 10+ unsupervised iterations without the user-discovering-a-bad-pivot complaint, consider promoting to round 2 with refinements. Until then, this is round 1.

If a fresh failure mode emerges where pivot-on-friction produces accumulating low-priority work while a higher-priority slice silently sits blocked, the fix might be a "max simultaneously-blocked slices" rule. File as a finding via the proactive plan audit if observed.
