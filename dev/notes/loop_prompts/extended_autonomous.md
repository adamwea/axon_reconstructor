# Extended autonomous loop prompt

The standing `/loop` invocation prompt for this repo. Paste the fenced block
below into the `/loop` command (Claude Code) to fire an autonomous iteration
on the queued plan tier.

**How to use:**
1. Open Claude Code in this repo.
2. Type `/loop` and paste the fenced block below as the loop instructions.
3. The loop reads `CLAUDE.md` → `dev/notes/memory/current_state.md` → relevant
   guardrails, then advances the next slice on its own. Stops only on
   "stop"/"wake up"/"wrap up" or genuine queue exhaustion.

**Revision history**: tracked by `git log dev/notes/loop_prompts/extended_autonomous.md`.
Earlier paste-cache copies of rounds 1-3 live in `~/.claude/paste-cache/` (don't
edit those; they're auto-managed by Claude Code).

**When to bump the round**: when authorizations change, when major
directives ship (and can be removed as first-iteration priorities), or when
new USER INJECTIONS warrant in-prompt visibility instead of relying solely
on `current_state.md` discovery.

---

## Current — Round 6 (2026-05-21)

```
EXTENDED AUTONOMOUS MODE (round 6) — relaxed stop conditions.

Read /global/homes/a/adammwea/dev/pkgs/axon_recon/CLAUDE.md and follow its
entry protocol. Read dev/notes/memory/current_state.md ⚡ USER INJECTIONS
FIRST — directives there are authoritative over this prompt and over plan
tier order. Read dev/notes/guardrails/ — `loop_cadence.md` and
`env_parity.md` have hard rules baked in. Advance the queued plan tier
one slice at a time.

Authorizations (override any conflicting line in this prompt):
- SHIFTER REBUILDS authorized. podman build + podman push docker.io/... +
  shifterimg pull are loop-doable. Pre-check `podman login --get-login
  docker.io` (returns `adammwea` via persistent authfile per env_parity).
  If login missing, post blocker under USER INJECTIONS and pick another
  slice instead of attempting a build that would fail at push.
- SIBLING-REPO GIT PUSHES authorized (SLAy, kssynth, unitlink, UnitMatchPy)
  on WELL-NAMED BRANCHES (not main, except inaugural publish of a fresh
  repo where main IS the named branch). `gh repo create` authorized.
  Loop NEVER `gh pr merge`s — user-only.
- AXON_RECON git push still PROHIBITED. Slurm submission still PROHIBITED.

Relaxed stance (applies until user explicitly says stop):
- NEVER stop on a blocker within ~15 min. git restore, write blocker into
  dev/notes/memory/open_questions.md, PICK UP ANOTHER UNBLOCKED SLICE.
- Stop only when: (a) user says "stop"/"wake up"/"wrap up", or
  (b) ALL plans/trackers genuinely exhausted.
- Queue empty? Audit pass (refine plans, prune stale notes, write next-
  tier tracker entries). Don't idle.
- Login-node smokes on REAL DATA encouraged. Limits: --task-backend
  local_affinity, cap 64 procs, --limit-* flags. NO slurm submissions.
  Bigger smokes → salloc command into "📝 User actions queued" + keep going.
- Commit often + restore often.

Loop cadence (per guardrails/loop_cadence.md):
- Heartbeat ScheduleWakeup.reason MUST be:
  "Next iteration in {N}s — {one sentence on what's queued/being watched}"
  NEVER bare "heartbeat armed".
- Cadence ladder:
  - Actively iterating mid-slice: 90s
  - Between slices, next candidate identified: 120s
  - Queue empty / audit-pass mode: 270s (stay in cache window)
  - Blocked on user gate / external state: 600–1200s
  - Never default to bare 1200-1800s without naming a specific gate.

Phase-enable for tests rule (USER INJECTION 2026-05-21):
- Smoke-testing a YAML-disabled phase (enabled: false) MUST enable it for
  the smoke — `--force-enable PHASE` CLI flag (shipped, commit 5d28561) is
  the canonical mechanism. Never validate a phase by running it disabled.

Real-data smoke log discipline (USER INJECTION 2026-05-21):
- After every smoke on REAL DATA (login-node OR user-initiated salloc),
  append an entry to dev/notes/trackers/smoke_log.md per its schema —
  including bug→fix-commit chains.
- Do NOT log dry-run smokes / unit tests / synthetic-fixture smokes.
  Those go in commit messages.
- HARD-gate visual diagnostics: when reviewed/approved/rejected, also
  append a smoke_log entry.

Diagnostic discipline (USER INJECTIONS 2026-05-21):
- ANY slice changing user-visible rendering MUST file a diagnostic entry
  in dev/notes/memory/diagnostics_to_review.md. Multi-stage algorithms
  file at EACH stage transition (not just the final end-to-end gate).
- Diagnostics MUST include a rendered image (PNG/SVG/PDF) as the
  user-visible artifact. npy / tsv / parquet ALONE is data, not a
  diagnostic. Data files can accompany the image for downstream consumers.

Stop-and-ask discipline + multiple-choice format (USER INJECTIONS 2026-05-21):
- When blocked on or facing friction with an EXPLICIT user instruction
  (e.g. "use existing X", "pick high-branch unit", "produce comparison"),
  STOP. Do NOT improvise a substitute approach. The substitute may look
  like progress but answers a question the user didn't ask.
- The "ask" form MUST be 2-4 numbered options in open_questions.md, each
  with `label / touch-size / tradeoff`, AND a "recommended" option called
  out. NOT a bare prose halt. Format mirrors the assistant's
  AskUserQuestion tool calls: do the analysis, curate options, let the
  user pick a number.
- For the NEXT 3 diagnostic-generation iterations: PAUSE BEFORE each
  diagnostic and surface a multiple-choice PRE-DIAGNOSTIC GATE in
  open_questions.md (full plan as option 1, alternatives 2-4). Loop
  does NOT execute until user marks `✅ USER APPROVED <date>: option N`.
  After 3 successful pre-gated diagnostics ship without user complaints
  about deviation, this discipline relaxes back to "file as you go."

Proactive plan audit (USER INJECTION 2026-05-21):
- Regularly audit dev/notes/plans/active/*.md + trackers/* for logical
  inconsistencies, stale assumptions, dead slices, redundant work
  across plans, scope drift, inefficient orderings. Audit cadence:
  opportunistic (1-2 min scan before starting a new slice from a plan)
  AND during audit-pass mode (queue empty — deeper pass on 1-2 plans).
  NOT every iteration; that's overhead.
- File findings to open_questions.md under "## 🔎 Plan-audit findings
  (loop-surfaced)" as multiple-choice questions (same format as
  stop-and-ask): 2-4 options + a recommended. Cap at 5 open findings
  — close one before filing a 6th.
- When the user asks for status ("any blocks?", "any questions?"),
  surface a digest of plan-audit findings alongside the active gates.
- Tone: neutral observations, not blame. Most stale-assumption findings
  are about prior loop iterations.

Standing pre-cleared decisions:
- SLAy PR merge policy: USER-ONLY merge. Loop pushes branch + opens PR via
  `gh pr create` but never `gh pr merge`.
- kssynth slice 3b heavy-smoke data-routing: PATH 2 (--input-root
  plumbing). Loop ships the analyzers-loader plumbing slice; then heavy
  smoke uses --input-root /pscratch/sd/a/adammwea/analyzed_data/...
  combined with --output-root /pscratch/sd/a/adammwea/dev_outputs/<slice>/.
- Radivojevic plan slice 1 USER GATE: RESOLVED 2026-05-21. All 6 Qs
  answered. Phase name = `radivojevic_recon`. Hyperparams locked
  (Step 1 = 9 STD, Step 2 = 2 STD / 50 μm, Step 3 = 1 STD / 100 μm,
  direct interconnect = 100 μm, skeleton-assisted = 200 μm, upsample
  parameterized as `upsample_factor: 10` integer ratio NOT absolute Hz).
  Add minimum-n_spikes ≥50 sanity check at slice 3 start.

Standing constraints (NOT relaxed):
- Working-data scope: 80k DMEM well000 of M08073, all DIVs. Do not expand.
- /pscratch/.../analyzed_data/ is read-only reference data. Iteration
  outputs go to /pscratch/sd/a/adammwea/dev_outputs/<slice>/.
- Code lives in /global/homes. Podman GraphRoot lives in
  /pscratch/sd/a/adammwea/podman_storage/ (DIRECTIVE A scoped override;
  no other pscratch overlays).
- Sample rate is per-recording (MaxTwo 10 kHz current cohort,
  MaxOne 20 kHz, ThreeBrain/Sony TBD future). NEVER hardcode sample rate
  in any analysis/recon phase — read from analyzer manifest via
  metadata_get. See project-axon-recon-device-diversity auto-memory.
- All guardrails apply. Visual diagnostics filed in
  memory/diagnostics_to_review.md as you go.

Default to Opus 4.7; Sonnet 4.6 only for very concrete mechanical work.
Continue iterating until I explicitly stop.
```
