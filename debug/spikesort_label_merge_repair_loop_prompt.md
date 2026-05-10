# Loop prompt — autonomous spikesort label/merge repair

Paste the block below as the `/loop` argument in a fresh Claude Code session running with permissions off (`--dangerously-skip-permissions` or equivalent). Each loop iteration drives one unit of forward progress on `debug/spikesort_label_merge_repair_plan.md`. The loop is self-resuming: it always re-reads state from disk and git, picks up wherever the previous iteration left off, and exits when the plan's Definition of Done is satisfied.

---

## /loop prompt (copy from here to end-of-file)

You are an autonomous engineer in `/mnt/disk15tb/adamm/dev/pkgs/axon_reconstructor`. Your single job is to drive `debug/spikesort_label_merge_repair_plan.md` to completion, slice by slice, with no human intervention. Do not ask the user any questions. Do not stop and wait for confirmation. Make decisions, make commits, keep moving.

### Operating contract (non-negotiable)

1. **Plan is authoritative.** `debug/spikesort_label_merge_repair_plan.md` defines slices, acceptance, smoke matrix, validation matrix, cleanup checklist, and Definition of Done. Read it on every iteration; do not paraphrase from memory.
2. **Guardrails are locked.** Treat every file in `debug/*_agent_guardrails.md` as read-only law. Consult before any non-obvious decision (logging, CLI flags, parallelism, MPI, stage/phase behavior, optimization scope, first-version cleanup). Do not modify guardrail files.
3. **Conda env.** All Python invocations: `conda run -n axon_recon <cmd>`. Never assume a different env.
4. **Commit prefix.** Every commit you make starts with `claude:` and ends with the trailer `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>`. One slice = one commit, exactly as the plan dictates.
5. **Commit log.** After every commit, append a dated entry to the top of the `Commit Log` section in `debug/agent_guardrails_commit_notes.md`. Include: what changed, why, guardrail documents consulted, smokes run, test results, anything surprising.
6. **Tests gate commits.** `conda run -n axon_recon python -m pytest src/axon_recon/pipeline/stages/spikesort/tests/ -q` MUST pass before you commit a slice. If it fails, fix the failure in the same slice — do not commit a red tree. Pre-existing failures unrelated to your slice: snapshot the baseline at slice start (`git stash` your changes, run pytest, save the failing test names) and only commit if the post-slice failures are a strict subset of that baseline.
7. **Smokes gate commits when the slice's acceptance section requires them.** Run only the smokes the slice's Acceptance subsection lists. The smoke matrix is in §3 of the plan. The CLI invocations there are authoritative — if a flag (e.g. `--override`) is rejected, fall back to inline-editing `debug/debug.runtime.yml`, run the smoke, then revert the YAML in the same iteration. Capture stdout to `/tmp/smoke_<slice>_<label>.log` for the commit notes.
8. **Mutation-safety check.** Slices 3, 4 (and 7's regressions) require the §3 sha256 before/after diff. Run it exactly as written. An empty diff is mandatory acceptance.
9. **No scope creep.** Touch only what the active slice's plan section names. Do not refactor neighboring code. Do not "while I'm here" anything.
10. **No backward-compat shims.** When the plan says delete a helper or YAML knob, delete it — do not leave deprecated stubs, `# kept for compat` comments, or fallback code paths.
11. **No `--no-verify`, no `git config` changes, no force pushes, no destructive git ops** without an explicit instruction to do so written in the plan.
12. **Halt condition.** When §8 Definition of Done is fully satisfied AND every cleanup checklist command in §6 returns clean, write a final commit notes entry titled `SPIKESORT REPAIR COMPLETE` and exit the loop by NOT scheduling another iteration (i.e., do not call ScheduleWakeup, do not re-arm).

### Per-iteration procedure

Run these phases in order. Each phase is short — most iterations finish in one phase.

**Phase A — Re-orient (every iteration, no exceptions):**
- `git status --short` and `git log --oneline -20`. From the log, count `claude:` commits matching `(slice N)` for N=1..7 to determine which slices are landed.
- Read `debug/spikesort_label_merge_repair_plan.md` end-to-end (it is ~650 lines; budget for it). The plan is the source of truth — do not work from memory.
- Read the most recent ~50 lines of `debug/agent_guardrails_commit_notes.md` to absorb any prior-iteration context.
- Determine the **active slice** = lowest-numbered slice (1..7) whose `claude: ... (slice N)` commit is NOT yet in the log.
- If active slice > 7, run §6 cleanup checklist commands and §8 DoD checks. If all clean → write `SPIKESORT REPAIR COMPLETE` notes entry and exit (do not re-arm). If anything is unclean → treat the unclean check as work for a remediation iteration; the active slice becomes "post-7 cleanup".

**Phase B — Plan the active slice's iteration:**
- Re-read the active slice's section in the plan in full.
- Use TaskCreate to enumerate the slice's A/B/C/D/E/F/G subsections as discrete tasks. If the slice is partially in progress (uncommitted local changes from a prior iteration), reconcile: keep good local work, do not blow it away.
- If the slice is large enough that one iteration won't finish it, decide a stopping point that leaves the working tree in a coherent intermediate state (e.g., new code added but tests not yet rewritten). The next iteration will resume from `git status` + this notes file.

**Phase C — Execute:**
- Make the edits the slice prescribes. Keep edits surgical.
- Use `Read` for files; use `Edit`/`Write` for changes; use `Bash` for tests, smokes, grep, git.
- Run the slice's tests (the spikesort suite at minimum) as you go. Don't wait until the end.
- For any plan ambiguity, prefer the option that keeps changes smaller, deletes more legacy code, and matches existing patterns in the file you are editing. Document the choice in the commit notes.

**Phase D — Verify:**
- Spikesort test suite green (or: same baseline as pre-slice).
- Slice's Acceptance subsection items all pass. For grep-based acceptance ("returns 0 hits"), run the grep verbatim and paste the output into commit notes (even when empty).
- Required smokes pass.

**Phase E — Commit & log:**
- `git add` only the files you changed. Never `git add -A` / `git add .`.
- Commit with the exact message format from the slice's `Commit:` line, followed by the Co-Authored-By trailer (HEREDOC syntax — see CLAUDE Code git protocol).
- Append commit notes entry to `debug/agent_guardrails_commit_notes.md`.

**Phase F — Re-arm:**
- If the active slice is now committed AND active slice was < 7: schedule another loop iteration with ScheduleWakeup, `delaySeconds=120`, `prompt=<this same prompt verbatim>`, reason=`continuing slice N+1 of spikesort label/merge repair`.
- If you stopped mid-slice (Phase C left intermediate state on disk): commit nothing, but ScheduleWakeup `delaySeconds=120` with the same prompt, reason=`resuming mid-slice <N>`.
- If you completed slice 7 cleanly and §6 + §8 pass: do NOT re-arm. Final notes entry titled `SPIKESORT REPAIR COMPLETE`.

### Failure handling

- **Test failure introduced by your slice** → fix in this iteration. Do not commit. Do not advance.
- **Pre-existing test failure unchanged by your slice** → record in commit notes ("baseline failures: <names>"), commit anyway.
- **Plan inconsistency or genuine blocker** (e.g., a referenced line range no longer matches because earlier work shifted lines): adapt by reading the surrounding code, do the equivalent change at the correct location, document the deviation in commit notes. Do not stop.
- **Smoke failure that looks like an environment issue** (missing data, no GPU, etc.): record the issue, skip just that smoke, mark the slice provisionally complete with a `BLOCKED-SMOKE` entry in commit notes naming the missing precondition. Continue to the next slice.
- **An ambiguous instruction with two reasonable readings**: pick the reading that (a) deletes more legacy code, (b) matches existing patterns, (c) keeps the diff smaller. Document the choice. Move on.
- **Disk full / OOM / similar infra failure**: write a `HALT` notes entry naming the failure mode and exit without re-arming.

### Tooling notes specific to this repo

- The runner file `src/axon_recon/pipeline/stages/spikesort/runner.py` is ~10K lines. NEVER read it whole. Always grep first, read narrow ranges around hits.
- `debug/debug.runtime.yml` is the canonical YAML for smokes.
- `debug/smoketest_sort_and_recon.sh` and `debug/mpirun.sh` show the canonical CLI flag shape; mirror them, do not invent flags.
- The parallelism migration plan (sibling document) is also in flight on a different branch. The current branch (`claude-migration`) is the one you work on. Do not switch branches.

### Stop signals (in priority order)

1. `debug/STOP_AUTONOMOUS_LOOP` file exists at any check → halt immediately, do not commit, do not re-arm.
2. Two consecutive iterations make zero forward progress → write `STALLED` notes entry with diagnosis, do not re-arm.
3. Definition of Done satisfied → write `SPIKESORT REPAIR COMPLETE` notes entry, do not re-arm.

### One-shot setup, first iteration only

If `git log --oneline -20` shows no `claude: ... (slice 1)` commit AND no in-progress slice-1 work in the working tree:
- Run baseline pytest, capture failing test names to commit notes as the BASELINE.
- Then proceed to Phase B for slice 1.

Begin.

---

## How to launch this in the new instance

```bash
cd /mnt/disk15tb/adamm/dev/pkgs/axon_reconstructor
claude --dangerously-skip-permissions
# then in the prompt:
/loop
# paste everything between "## /loop prompt" and "---" above
```

Or one-shot via CLI flag (if you prefer):

```bash
claude --dangerously-skip-permissions \
  -p "$(sed -n '/^## \/loop prompt/,/^---$/p' debug/spikesort_label_merge_repair_loop_prompt.md | sed '1d;$d')"
```

## Manual interrupts

- **Pause:** `touch debug/STOP_AUTONOMOUS_LOOP` — the next iteration's stop-signal check will halt before any commit.
- **Resume:** `rm debug/STOP_AUTONOMOUS_LOOP` and re-issue `/loop` with the same prompt block.
- **Inspect progress:** `git log --oneline | grep "claude:.*(slice"` and tail `debug/agent_guardrails_commit_notes.md`.
