# Slice contracts — compressed returns with interface

The "subtask returns the contract, not the trace" discipline. Each shipped slice gets ONE entry here recording what interface it exposed, what it assumed about its inputs, and what downstream now needs re-verification.

**Reading rule for the loop**: when starting a new slice, scan this file for slices that touched related code; their `Produces` + `Propagates` fields tell you what's safe to assume vs needs re-verification.

**Writing rule**: loop APPENDS one entry per shipped slice. Append-only; never edits prior entries (they're history; if a later slice supersedes them, the LATER entry records that and points back).

---

## Schema

```
### <commit-hash> — <slice name> (<plan + slice number>, <date>)
- **Produces**: <interface signature OR output shape OR file layout this slice now exposes>
- **Assumes**: <preconditions on inputs this slice requires>
- **Propagates**: <list of files / nodes downstream that should be re-verified because this slice landed>
- **Trusted-output impact**: <any TR-xxx entries in brain/trusted_outputs.md this slice's behavior affects>
- **Metric impact**: <any M-xxx entries in brain/metrics.md this slice's behavior affects>
- **Prediction** (filled BEFORE execution): what the loop expects to observe after the slice ships — concrete numbers / shapes / pass-or-fail conditions that the critic + smoke can check against
- **Actual** (filled POST-commit by loop or critic): what was actually observed
- **Delta**: matches | diverges-as-expected | diverges-unexpectedly | not-yet-verified
```

**Why Prediction matters** (2026-05-21 brain-build): the critic subagent (per `brain/guardrails/critic_separation.md`) checks invariants against the diff. With invariants alone, the critic can only verify "does this match the spec." With a Prediction, the critic can also verify "does this match what the loop EXPECTED" — which catches the failure mode where the slice technically passes invariants but the outcome surprises the loop. A surprise outcome usually signals a deeper model error (e.g. the loop's mental model of how the code behaves is wrong) — exactly the thing the actor-critic separation is supposed to catch.

**Discipline**: Prediction MUST be filled before the slice's actor work starts. If the loop can't articulate a prediction, the slice is under-spec'd — propose splitting OR add the spec work as its own prior slice.

---

## Entries

### STATUS — populated forward from 2026-05-21

Earlier slices (kssynth integration 1a-4e, parallelism 9.5, etc.) were shipped before this file existed. They're not retroactively backfilled — that's archeology, not signal. Going forward, every loop-shipped slice gets an entry here.

The earlier slices' contracts are recoverable from:
- `dev/notes/commit_log.md` — what shipped + why
- Per-plan `plans/active/*.md` — slice spec + acceptance criteria
- `git log -p` — the actual diff

---

## (Forward entries land below this line as slices ship)

*(append-only; do NOT insert in the middle)*

### 2026-05-21 — kssynth slice 3b: analyzer LOAD-path `--input-root` plumbing extension (PATH 2)

- **Surface**: `pipeline/stages/reconstruct/templates/integrations/spikeinterface_extract.py` — `load_spikeinterface_analyzers` + `iter_spikeinterface_analyzers` fallback recursions.
- **Intent**: when fallback recursion fires at `fallback_well_out_dir` (the alternate `--input-root` path), re-derive `analyzer_cache_dir` relative to that alternate so cache lookup follows the recursion instead of pointing at the primary's empty cache.
- **Produces**: behavior change — analyzers load successfully from reference data when `--output-root` is a fresh dev path AND `--input-root` provides the reference. No new public API.
- **Assumes**: `analyzer_cache_dir` when provided lives under `well_out_dir`; if not (caller-supplied absolute path outside primary tree), falls through unchanged via `ValueError` catch.
- **Propagates**: kssynth recon-stage integration (`kssynth_recon_integration_plan.md` slice 3b) — the heavy-smoke prereq is now unblocked. Downstream PRE-DIAGNOSTIC GATE 1 (radivojevic apples-to-apples comparison) is gated on this smoke succeeding.
- **Trusted-output impact**: TR-001 (176 templates) unchanged — this fix is upstream of unit-manifest generation. Z3-TR-001 schema invariants unchanged. TR-000 reference-data read-only invariant respected (load only; no writes to alternate path).
- **Metric impact**: enables a new metric baseline once smoke runs — `source_count > 0` + `units_ok > 0` on M08073/000208/well000 DIV 36 with `--input-root` set.
- **Critic verdict**: concerns (1 caught: `.expanduser()` asymmetry between the two fallback sites) — fixed before commit; re-test green.

---

## How the loop uses this file

When the loop is about to start a slice from any plan:

1. Identify the surface area the slice touches (which phases, modules, files).
2. Search this file for prior entries whose `Produces` or `Propagates` involves the same surface.
3. Cross-check: does the slice's plan still align with what those prior entries committed to? If a prior `Produces` says "X returns dict keyed by Y" and the new slice assumes "X returns list of Y", THAT'S the silent-downstream-break the dependency graph + this file together are designed to catch.
4. Surface mismatches as a `🛑 CONTRACT DRIFT` entry in `open_questions.md` multiple-choice, NOT improvise around it.

This is the verifier module's `Compressed return must carry the contract, not just "done"` from the brain-theory: the loop reads contracts not summaries, propagates change-impact down the dependency graph, and avoids the leaky-module failure mode where two plans drift apart in their assumptions about a shared interface.
