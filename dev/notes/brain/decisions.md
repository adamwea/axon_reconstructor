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
