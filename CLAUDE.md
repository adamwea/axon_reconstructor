# Claude — axon_recon repo entry point + loop protocol

**Read this first every session.** This file is the canonical entry point. It points at the rules (`dev/notes/guardrails/`), the working memory (`dev/notes/memory/`), the active plans (`dev/notes/plans/active/`), and the trackers (`dev/notes/trackers/`). It also encodes the loop protocol — when to compact, fork, spawn Agents; when to use which model; when smoke tests are required.

## Entry protocol (every session)

1. Read this file.
2. Read `dev/notes/guardrails/README.md` and the topic files it lists. Treat these as locked contracts unless the user explicitly asks for a guardrail change.
3. Read `dev/notes/memory/current_state.md` to absorb what's shipped, what's in-flight, what's queued. **The §"⚡ USER INJECTIONS" section at the top is authoritative — apply those directives at the earliest applicable slice before falling back to plan tier order. Promote resolved injections to guardrails / slice protocol / plans, then delete the entry.**
4. Read `dev/notes/memory/open_questions.md` for pending decisions.
5. Glance at `dev/notes/plans/active/` to know what plans exist; read the one you're working on cover-to-cover before starting a slice from it.

## Slice protocol (every commit-sized unit of work)

1. **Re-read the relevant guardrail** for the slice's surface area (parallelism / scope flags / force_restart / stage_phase / package / output_locations / dry_run).
2. **Plan the diff** before editing: list files touched, anticipated test impact, smoke-test requirement (see §"When a smoke test is required" below), whether the slice will produce visual diagnostics that need user review.
3. **Edit in scope.** Don't refactor adjacent code, don't add features the slice didn't authorize, don't add error handling for impossible cases. The `Doing tasks` section of the system prompt is the authority on this.
4. **Run targeted tests** for the modules touched. Pre-existing failures don't count against the slice; new failures block the commit.
5. **Run a smoke test** when required (see §below). Use `--dry-run` for fast wiring confirmation when applicable (see `guardrails/dry_run.md`).
6. **Generate visual diagnostics when the slice's claim depends on them** (see §"Visual diagnostics" below). Save under `/pscratch/sd/a/adammwea/dev_outputs/<plan_slice>/diagnostics/`. Add an entry to `dev/notes/memory/diagnostics_to_review.md` BEFORE commit.
7. **Commit** with a `claude:` subject prefix; descriptive body explaining what changed and why. Include a `Co-Authored-By: Claude Opus 4.7 …` line. If the slice added a diagnostic entry, reference it in the commit body.
8. **Append a line to `dev/notes/commit_log.md`** noting the slice + the plan/tracker it advanced.
9. **Update memory** if the slice changed anything in `current_state.md` or resolved an `open_questions.md` item.
10. **Update guardrails** if you discovered a new invariant or clarified an existing one — same `claude:` commit prefix.

## Visual diagnostics

A passing test suite is necessary but not sufficient. Some things can only be confirmed visually (waveform shapes, template overlays, sort-quality plots, before/after comparison figures, channel-grid sanity, …). When a slice's claim of correctness depends on "the picture looks right" rather than "the count is N", produce a diagnostic artifact and flag it for the user.

**Generate a diagnostic when:**
- The slice introduces a new phase that produces a visual output (a plot, a heatmap, a footprint figure). Save its FIRST real-data output as a diagnostic; the user verifies the visual shape is what they expect.
- A bug fix produces visually different output than before, even when tests pass. Save a before-fix and after-fix figure side-by-side. The user sanity-checks the picture matches the intended behavior.
- A regression check on a known-good baseline needs a before/after comparison (e.g. ds4/well000 templates pre- and post-concat-rip-out).
- A new computed table is being introduced (match tables, UID assignments). Save the first one for inspection — schema sanity, sensible row counts.

**Don't generate a diagnostic for:**
- Every routine plot the pipeline emits during a normal run. Those are pipeline outputs, not Claude-flagged artifacts.
- Pure computation slices whose validation is numerical (test asserts count == N).
- Anything covered by an automated assertion that the user trusts.

**Gate levels:**
- **Soft gate**: the user reviews when they can; downstream slices proceed in the meantime. Default for most diagnostics.
- **Hard gate**: downstream slices on the same code path WAIT for user approval before starting. Use sparingly — only when getting the picture wrong would invalidate everything downstream (e.g. a new template extraction algorithm's first output: if it's wrong, every plot that uses it is wrong).

**Format**: entries in `dev/notes/memory/diagnostics_to_review.md`. See that file for the schema.

## Commit cadence & rollback

**Commit often.** Small commits beat big ones for bisectability, for rollback, and for review. The rule of thumb: every coherent unit of work — even mid-plan — gets its own commit. Don't accumulate "I'll commit at the end" diffs. If you find yourself with >5 files changed and no commit since you started a slice, stop and commit what's stable before continuing.

**Restore often when fixes break things.** If a change you made breaks something — tests fail in a way you didn't anticipate, the smoke-test produces wrong output, a downstream phase that was passing now isn't — the FIRST instinct should be `git restore <file>` (or `git reset --hard HEAD` for uncommitted scratch work). Don't pile additional fixes on top of a broken state hoping to recover. Restore to the last known-good commit, re-think, try again. The whole point of committing often is to make this cheap.

Specifically:
- If the test suite was green before your slice and is red now, and you've spent more than ~15 minutes trying to fix it: restore. The slice as-conceived may be wrong, not just buggy.
- If a smoke test produces obviously-wrong output (counts off by an order of magnitude, NaNs everywhere, file format unreadable): restore. Don't tweak the smoke fixture; tweak the code-or-plan.
- If you accidentally edited files outside the slice's scope: `git restore <out-of-scope-file>`. Stay surgical.
- If `git status` shows uncommitted changes you don't remember making (system reminder reset state, conversation forking, etc.): inspect with `git diff` first. If they're not from the current slice's intent, restore them.

The point isn't to be timid. It's to never let an in-progress slice's debris become next session's mystery.

## Context window management

| Situation | Action |
|---|---|
| Conversation hits ~50% of context window AND current slice is complete | `/compact` — preserve plan / current state, drop debugging chatter |
| Starting a discrete subtask with bounded scope (e.g. mechanical phase deletion, single test write-up) | Spawn an Agent (Explore for read-only, general-purpose for write-able). Keeps the main conversation clean. |
| Need to investigate multiple files in parallel | Single message with multiple parallel `Agent` tool calls |
| About to do a long-running build / rebuild (shifter rebuild, multi-minute test runs) | Use `Bash` with `run_in_background: true`; let `Monitor` watch for done/error patterns. Don't burn cache window waiting. |
| Output of a single Bash call is going to be huge (logs > 5KB) | Pipe to `tee` to a file; read only the lines you need via `Read` with offset/limit |
| Cache miss is imminent (5+ min sleep planned) | Don't sleep. Either schedule a `Monitor` to wake on a signal, or hand back to the user with the question. |
| Working on independent slices from different plans | Consider forking: finish current slice, commit, then start a fresh session for the next plan to keep context tight |

## Model selection

**Default to Opus 4.7. User prefers quality over speed.** Use Sonnet 4.6 only when the work is very concrete, very mechanical, and very low-risk.

**Sonnet 4.6 — use ONLY for:**
- Pure rename / grep-driven find-and-replace where the diff is mechanical and a `grep -rn` audit can confirm completeness.
- Running tests / inspecting log files / summarizing outputs from a single Bash invocation.
- Drafting docstrings, comments, CHANGELOG entries against already-locked code.
- File moves between directories with no content changes.
- `--dry-run` smoke runs whose only purpose is "did the wiring work" (not "is this correct").

**Opus 4.7 — use for everything else, including:**
- Any code change that affects more than one file, OR touches more than ~20 lines of one file.
- Diagnosing bugs of any kind (even ones that look simple — the SLAy `==` → `>=` assertion looked simple, the n_jobs resolver fix looked simple, both took careful reading).
- Plan / tracker / guardrail edits (design work).
- Smoke-test result interpretation (looking at logs and deciding "is this the right behavior?").
- Anything where two reasonable options exist and the choice has downstream consequences.
- Anything where the user is going to look at the diff and ask "why this and not that?".

If in doubt, choose Opus. The cost is small; the cost of a wrong Sonnet decision is large.

## When a smoke test is required

A smoke test = run the affected code path end-to-end on real data (or a tight synthetic equivalent) and confirm outputs match expectations. Distinct from unit tests, which run fast and are always required.

**Smoke test REQUIRED when the slice touches:**
- Parallelism / resource allocation (n_jobs, slot.cpu_count, MPI worker count, resource budget manager)
- Force-restart contract (any change to what gets `rmtree`'d when)
- Phase signatures or phase wiring (a phase moved between stages, renamed, added, deleted from a default sequence)
- CLI flag semantics (`--targets`, `--target-wells`, `--profile`, `--task-backend`, `--force-restart`, …)
- YAML config schema (new top-level keys, removed keys, changed defaults)
- Output schema (anything that downstream consumers — bombcell, SLAy, UnitMatch, kssynth, dashboards — reads)
- Cross-stage contracts (recon-stage output that analysis-stage consumes; spikesort-stage output that recon consumes)

**Smoke test NOT REQUIRED for:**
- Pure deletions of dead code that's already disabled (grep-audit-clean)
- Doc / comment / commit-log edits
- Renaming a symbol when grep -rn confirms no orphan references
- Adding a new test
- Refactoring within a function where the function's contract is unchanged

**Smoke test scoping ladder** — try cheapest first:
1. **`--dry-run`** on the affected phase. Should complete in seconds. Confirms wiring + input resolution. See `guardrails/dry_run.md`.
2. **Targeted login-node smoke** — a few wells / a few units / a few segments at most. Use `--task-backend local_affinity` (NOT mpi, NOT srun). Cap at **64 processes total**. Use `--limit-wells`, `--limit-wells-per-dataset`, `--limit-units`, `--limit-segments` to keep the scope small. The login node is shared infrastructure; don't sit on it for hours. The point of this tier is "did the algorithm actually run on real data on one or two units" — NOT "did it converge".
3. **Bigger than that — stop and hand back to the user.** Larger smoke tests need a real allocation. If the slice needs an interactive GPU allocation (`spikesort.sort`, `merge_SLAy`, anything GPU), a multi-node CPU allocation (full reconstruct, full preprocess sweep), or just more time than the login node should hold, surface the exact `salloc` + `srun` command and let the user start the allocation and kick off the run themselves. Don't silently start interactive jobs.
4. **Full-scope sbatch run.** User-initiated only. Claude doesn't submit sbatch jobs.

If step 1 fails, debug at that level before climbing the ladder. If step 1 passes but step 2 fails, the problem is in the expensive code path, not the wiring.

### Working-data scope for the current iteration cycle

- **One cohort only**: the **80k DMEM well000** — i.e. M08073 chip family, well000 of each DIV's AxonTracking recording. Plenty of completed data is available under `/pscratch/sd/a/adammwea/analyzed_data/Media_Density_T5_02182026_AR_axon_analysis_AW/...` for this cohort.
- **Reference (do NOT mutate)**: existing `analyzed_data/...` outputs are the ground-truth reference for desired behavior. Compare against them; don't overwrite them. Any iteration output goes to a SEPARATE pscratch dir.
- **Iteration output dir**: write new run outputs (anything Claude produces while developing) under `/pscratch/sd/a/adammwea/dev_outputs/<feature>/...`. Conventionally one subdir per feature or plan-slice. Auto-purgeable; pscratch retention is fine.
- **Raw network scans available**: under the relevant `raw_data/` tree. Single-segment recordings (no concat); preprocess stage must run on them before any downstream work. Used for the eventual `unitmatch` v2 scope (`unitmatch_phase_plan.md` slice 6) — out of scope for v1 iteration.

## Update protocol for this file

CLAUDE.md is the contract for how Claude works in this repo. Update it when:
- A new directory convention is added (e.g. a new top-level docs section)
- The loop protocol changes (model selection heuristic shifts, new context-window pattern emerges)
- The user articulates a new rule that doesn't fit a single guardrail topic

Don't update it when:
- A specific code contract is established → that's a `guardrails/*.md` change
- A specific item is in-flight or shipped → that's a `memory/current_state.md` change
- A new bug or open question surfaces → that's `memory/open_questions.md` or `trackers/issues.md`

## Pointers

| Resource | Path | Update frequency |
|---|---|---|
| **Guardrails** (locked code contracts) | `dev/notes/guardrails/*.md` | Rarely; only when a contract changes |
| **Working memory** (current state, open questions, scratch) | `dev/notes/memory/*.md` | Often; refined as Claude works |
| **Active plans** | `dev/notes/plans/active/*.md` | Touched per-slice during execution |
| **Trackers** (tech debt, issues, roadmap) | `dev/notes/trackers/*.md` | When new items surface |
| **Salloc smokes queued** | `dev/notes/trackers/salloc_smokes_queued.md` | When the loop needs to surface a smoke that requires an interactive allocation (user-only to run). Loop appends; user prunes when the smoke completes. |
| **Real-data smoke log** | `dev/notes/trackers/smoke_log.md` | Append-only log of completed real-data smokes (login-node or salloc) with bug→fix chains. |
| **Commit log** | `dev/notes/commit_log.md` | After every `claude:` commit |
| **Loop prompts** (autonomous-mode `/loop` invocations) | `dev/notes/loop_prompts/*.md` | When authorizations change or directives ship |
| **Archive** (old guardrails, handoffs) | `dev/notes/archive/` | Don't touch unless rescuing context |
