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
```

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

---

## How the loop uses this file

When the loop is about to start a slice from any plan:

1. Identify the surface area the slice touches (which phases, modules, files).
2. Search this file for prior entries whose `Produces` or `Propagates` involves the same surface.
3. Cross-check: does the slice's plan still align with what those prior entries committed to? If a prior `Produces` says "X returns dict keyed by Y" and the new slice assumes "X returns list of Y", THAT'S the silent-downstream-break the dependency graph + this file together are designed to catch.
4. Surface mismatches as a `🛑 CONTRACT DRIFT` entry in `open_questions.md` multiple-choice, NOT improvise around it.

This is the verifier module's `Compressed return must carry the contract, not just "done"` from the brain-theory: the loop reads contracts not summaries, propagates change-impact down the dependency graph, and avoids the leaky-module failure mode where two plans drift apart in their assumptions about a shared interface.
