# TODO — running list of next actions

User-facing list of the most useful next actions, in rough order. Updated as items resolve. Loop appends new TODOs at the bottom; user prunes.

**Status as of 2026-05-21**: planning/refinement pause is in effect (`brain/current_state.md` 🛑 PAUSED header). Items in this list survive the pause; they describe what's queued for when the pause lifts AND what user-side work would enable the pause to lift.

---

## 🔴 Currently the blockers to lifting the pause

These are the items the brain-skeleton build (2026-05-21) identified as the work needed before autonomous refinement can resume safely.

### T1 — Phase-zero mapping mission [✅ COMPLETED 2026-05-21 by assistant manually]
Dependency graph populated in `brain/dependency_graph.md` (6 stages, 4 active stage phase_sequences, 13 data junctions J1-J13, plan→junction touch matrix, 6 ranked verification checkpoints). Ranked checkpoint candidates mirrored into `brain/trusted_outputs.md` "Proposed for promotion" as TR-CAND-001..006. Ready for T2.

### T2 — Triage trusted-output candidates [✅ DONE 2026-05-21 by user blanket pin]
User: "I already trust those pinned things. pretty much anything in the reference data is pinned. New outputs should be identical or at least similar in shape to the current reference data." Result:
- **TR-000** (blanket pin on the entire reference data tree at `/pscratch/sd/a/adammwea/analyzed_data/Media_Density_T5_02182026_AR_axon_analysis_AW/`) added to Tier 1
- **TR-002, TR-003, TR-004** (was TR-CAND-001, 003, 006) all promoted to Tier 1
- Meta-rule recorded in `brain/trusted_outputs.md`: new outputs must match reference shape OR be a justified differential
- Heavy candidates (TR-CAND-002, 004, 005) remain "proposed" since the artifacts don't exist in the reference tree yet — they'll become Tier 3 provisional when generated and only promote after user review

### T3 — Derive invariant assertions from the pinned reference tree [LOOP, allowed during pause]
With TR-000 + TR-001..004 pinned, the loop can now author invariant assertions automatically:
- Per-stage output-dir structure (file paths + naming conventions)
- Numpy shape + dtype invariants per file
- JSON schema invariants per `*_summary.json`
- TSV column invariants per `cluster_*.tsv` / `branches.json` etc.
- Count invariants where applicable (e.g. TR-001's 176 templates / 287 good + 312 mua)

Loop writes these to a `brain/invariants/` subdir or appends to `brain/trusted_outputs.md` per-entry (TBD by loop in execution). User reviews the ASSERTIONS (fast — read claims, not data) before they become the trusted gate.

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
