# TODO — running list of next actions

User-facing list of the most useful next actions, in rough order. Updated as items resolve. Loop appends new TODOs at the bottom; user prunes.

**Status as of 2026-05-21**: planning/refinement pause is in effect (`brain/current_state.md` 🛑 PAUSED header). Items in this list survive the pause; they describe what's queued for when the pause lifts AND what user-side work would enable the pause to lift.

---

## 🔴 Currently the blockers to lifting the pause

These are the items the brain-skeleton build (2026-05-21) identified as the work needed before autonomous refinement can resume safely.

### T1 — Phase-zero mapping mission [✅ COMPLETED 2026-05-21 by assistant manually]
Dependency graph populated in `brain/dependency_graph.md` (6 stages, 4 active stage phase_sequences, 13 data junctions J1-J13, plan→junction touch matrix, 6 ranked verification checkpoints). Ranked checkpoint candidates mirrored into `brain/trusted_outputs.md` "Proposed for promotion" as TR-CAND-001..006. Ready for T2.

### T2 — Triage trusted-output candidates [USER, current action]
Read `brain/trusted_outputs.md` "Proposed for promotion" section. Six candidates ranked by leverage. Bucket each into:
- ✅ **PIN** (eyeball + approve now)
- 💵 **ACQUIRE LATER** (cheap but needs a smoke / new code first)
- ❓ **DEFER** (expensive or needs other work to land first)
- ❌ **SKIP** (don't bother)

Loop's recommendation: tackle TR-CAND-001 + 003 + 006 first (cheap bucket). Together with TR-001 (already pinned) they cover J1/J2/J3/J4/J6/J11/J12/J13 transitively — substantial coverage before any heavy work runs.

### T3 — Approve invariant assertions [USER, after T2]
For each trusted output, loop proposes a small set of invariant-based assertions (schema / counts / value-ranges / reconcilements). You approve the ASSERTIONS (not the data). These become the trusted gate the autonomous loop runs against.

### T4 — Lift pause [USER]
When T1-T3 are far enough along that the verifier scaffold actually exists, lift the pause via a new USER INJECTION saying so. Loop resumes execution gated on the new scaffold.

---

## ⚪ Standing non-blocking user actions (do whenever)

- **Delete the smoke-test repo** `adamwea/__gh_auth_smoke_test` via GitHub web UI (`gh` token lacks `delete_repo` scope). Clutter only.
- **Merge SLAy PR `claude/merge-fixes-2026-05`** via GitHub web UI when convenient. Loop pinned the feature-branch SHA in pyproject so this isn't blocking image builds; keeps SLAy main tidy.
- **Run queued salloc smokes** in `trackers/salloc_smokes_queued.md` — NOTE: don't run the radivojevic-comparison-gating ones until the pause lifts; running them now produces work that the new verifier scaffold may invalidate.

---

## 🧠 Brain-build deliverables (post-T4, for tracking)

These get built by the loop as it resumes; tracked here so you can see the system grow.

- **Slice contract discipline** (`brain/slice_contracts.md`): every shipped slice records produces + assumes + propagation list. Started 2026-05-21, populated by loop as it ships.
- **Actor/critic separation per slice**: loop spawns a separate verifier subagent (Explore-type) per slice. Sees only diff + relevant trusted fixture + invariants. Returns pass/fail/concerns.
- **Metric module + auto-rollback** (`brain/metrics.md`): refinement that degrades a baseline metric → `git restore` + escalate.

---

## 📋 How this list is maintained

- User prunes resolved items (don't leave 'DONE' clutter; link forward via commit hashes if needed).
- Loop appends new TODOs at the bottom — never inserts mid-list without flagging.
- Ordering: T-numbers are stable IDs; you can reorder priority without changing the ID.
- When a TODO becomes more than 2-3 lines, promote it to its own file under `brain/` or `plans/active/` and leave a 1-line pointer here.
