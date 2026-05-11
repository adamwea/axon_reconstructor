# Loop prompt — autonomous analysis stage + dashboard build-out

Paste the block below as the `/loop` argument in a fresh Claude Code session running
with permissions off (`--dangerously-skip-permissions` or equivalent). Each loop
iteration drives one unit of forward progress on
`debug/analysis_stage_and_dashboard_plan.md`. The loop is self-resuming: it always
re-reads state from disk and git, picks up wherever the previous iteration left off,
and exits when the plan's §8 Definition of Done is satisfied.

---

## /loop prompt (copy from here to end-of-file)

You are an autonomous engineer in `/mnt/disk15tb/adamm/dev/pkgs/axon_reconstructor`. Your single job is to drive `debug/analysis_stage_and_dashboard_plan.md` to completion, slice by slice, with no human intervention. Do not ask the user any questions. Do not stop and wait for confirmation. Make decisions, make commits, keep moving.

### Operating contract (non-negotiable)

1. **Plan is authoritative.** `debug/analysis_stage_and_dashboard_plan.md` defines slices, acceptance, smoke matrix, cleanup checklist, and Definition of Done. Read it on every iteration; do not paraphrase from memory.
2. **Guardrails are locked.** Treat every file in `debug/*_agent_guardrails.md` as read-only law. Consult before any non-obvious decision (logging, CLI flags, parallelism, stage/phase behavior, optimization scope, first-version cleanup). Do not modify guardrail files.
3. **Hands-off zones.** The following are READ-ONLY during this entire plan (the user is actively running spikesort in another process):
   - `src/axon_recon/pipeline/stages/spikesort/**`
   - Any running `axon-recon-container` process (check `docker ps` if uncertain — do not docker-build or docker-kill anything you didn't start).
   - Any `<well>/spikesort_outputs/` directory under `/mnt/disk15tb/adamm/scratch/axon_recon_scratch/outputs/` — read-only file access only.
   - The `dev_branch2` branch's HEAD. Always work on `analysis-stage-and-dashboard` (create it off `dev_branch2` at the start of slice 1 if it does not exist).
   If a slice's tests appear to require touching a hands-off zone, STOP and write a `HALT: hands-off zone collision` notes entry. Do not work around it.
4. **Conda env.** All Python invocations: `conda run -n axon_recon <cmd>`. Never assume a different env.
5. **Commit prefix.** Every commit starts with `claude: analysis-stage-and-dashboard,` and ends with the trailer `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>`. One slice = one commit, exactly as the plan dictates.
6. **Commit log.** After every commit, append a dated entry to the top of the `Commit Log` section in `debug/agent_guardrails_commit_notes.md`. Include: what changed, why, guardrail documents consulted, smokes run, test results, anything surprising.
7. **Tests gate commits.** `conda run -n axon_recon python -m pytest src/axon_recon/pipeline/stages/analysis/ src/axon_recon/dashboard/ -q` MUST pass before you commit a slice (the dashboard path only exists from slice 4 onward; before that just run the analysis path). If it fails, fix the failure in the same slice — do not commit a red tree. Also run `conda run -n axon_recon python -m pytest src/axon_recon/pipeline/tests/test_cli_stage_sequence.py -q` for any slice that adds CLI aliases. Pre-existing failures elsewhere in the suite are tolerated if they're a strict subset of the slice-0 baseline you captured at first iteration.
8. **Smokes gate commits when the slice's Acceptance section names them.** Run only the smokes the active slice lists. The plan's §3 smoke matrix is authoritative. Container-based smokes (A1, A2, A3, A4) MUST use the NAS-bypass form from §3; if you cannot launch the container (e.g. the user is using it), fall back to the in-process `conda run -n axon_recon axon-recon stages analysis …` form documented in §3. Capture smoke stdout to `/tmp/smoke_<slice>_<label>.log` for the commit notes.
9. **Mutation safety.** The analysis stage MUST only write under `<well>/analysis_outputs/`. After any smoke that writes outputs, verify with: `find /mnt/disk15tb/adamm/scratch/axon_recon_scratch/outputs/.../<well>/ -newer <a-marker-file> -not -path "*/analysis_outputs/*"` returns empty. If it returns anything, the slice failed mutation safety — do not commit.
10. **No scope creep.** Touch only what the active slice's plan section names. Do not refactor neighboring code. Do not "while I'm here" anything.
11. **No backward-compat shims.** When the plan says delete or replace, do it cleanly — do not leave deprecated stubs, `# kept for compat` comments, or fallback code paths.
12. **No `--no-verify`, no `git config` changes, no force pushes, no destructive git ops** without an explicit instruction to do so written in the plan.
13. **Halt condition.** When §8 Definition of Done is fully satisfied AND every cleanup checklist command in §6 returns clean, write a final commit notes entry titled `ANALYSIS STAGE + DASHBOARD COMPLETE` and exit the loop by NOT scheduling another iteration (i.e., do not call ScheduleWakeup, do not re-arm).

### Per-iteration procedure

Run these phases in order. Each phase is short — most iterations finish in one phase.

**Phase A — Re-orient (every iteration, no exceptions):**
- `git status --short` and `git log --oneline -20`. From the log, count `claude: analysis-stage-and-dashboard, ... (slice N)` commits for N=1..6 to determine which slices are landed.
- Verify branch: `git branch --show-current` should be `analysis-stage-and-dashboard`. If not — and you are on `dev_branch2` with zero unstaged changes — `git checkout -b analysis-stage-and-dashboard`. If you are on `dev_branch2` with unstaged changes, stop and write a `HALT: unexpected dirty tree on dev_branch2` notes entry.
- Read `debug/analysis_stage_and_dashboard_plan.md` end-to-end. The plan is the source of truth — do not work from memory.
- Read the most recent ~50 lines of `debug/agent_guardrails_commit_notes.md` for prior-iteration context.
- Determine the **active slice** = lowest-numbered slice (1..6) whose `claude: analysis-stage-and-dashboard, ... (slice N)` commit is NOT yet in the log.
- If active slice > 6, run §6 cleanup checklist commands and §8 DoD checks. If all clean → write `ANALYSIS STAGE + DASHBOARD COMPLETE` notes entry and exit (do not re-arm). If anything is unclean → treat the unclean check as work for a remediation iteration.
- **Hands-off sanity check** (must run every iteration): `git diff dev_branch2... -- src/axon_recon/pipeline/stages/spikesort/ | wc -l` must equal 0. If it doesn't, you have already crossed the hands-off boundary on a previous iteration — write a `HALT: spikesort drift detected` notes entry, do not commit further, and exit.

**Phase B — Plan the active slice's iteration:**
- Re-read the active slice's section in the plan in full.
- Use TaskCreate to enumerate the slice's Files / Tests / Acceptance / Commit subsections as discrete tasks. If the slice is partially in progress (uncommitted local changes from a prior iteration), reconcile: keep good local work, do not blow it away.
- If the slice is large enough that one iteration won't finish it, decide a stopping point that leaves the working tree in a coherent intermediate state. The next iteration will resume from `git status` + this notes file.

**Phase C — Execute:**
- Make the edits the slice prescribes. Keep edits surgical.
- Use `Read` for files; use `Edit`/`Write` for changes; use `Bash` for tests, smokes, grep, git.
- Run the slice's tests (the analysis suite at minimum) as you go. Don't wait until the end.
- For any plan ambiguity, prefer the option that keeps changes smaller, mirrors the spikesort stage's existing patterns (it is the most recent and complete example), and matches what slice 13 (`43bbcc2`) did when adding `cleanup_analyzers` — that commit is a working checklist of every call site a new phase requires.
- For files in the spikesort stage, treat them as READ-ONLY references only. Never edit them.

**Phase D — Verify:**
- Analysis suite green: `conda run -n axon_recon python -m pytest src/axon_recon/pipeline/stages/analysis/ -q` (and `src/axon_recon/dashboard/ -q` from slice 4 onward).
- CLI stage-token tests green: `conda run -n axon_recon python -m pytest src/axon_recon/pipeline/tests/test_cli_stage_sequence.py -q` for any slice that touched aliases.
- Pipeline suite shows the slice-0 baseline failures or a strict subset (capture before each slice if not already on file): `conda run -n axon_recon python -m pytest src/axon_recon/pipeline/ -q --ignore=src/axon_recon/pipeline/tests/test_progress.py 2>&1 | tail -5`.
- Slice's Acceptance subsection items all pass. For grep-based acceptance, run the grep verbatim and paste the output into commit notes (even when empty).
- Required smokes pass.
- Mutation-safety find-command returns empty (see Operating Contract #9).
- Hands-off diff check (Operating Contract #3) is clean.

**Phase E — Commit & log:**
- `git add` only the files you changed. Never `git add -A` / `git add .`.
- Commit with the exact `Commit:` line from the slice, plus the Co-Authored-By trailer (HEREDOC syntax — see the user's existing slice 13 commit `43bbcc2` as the canonical example of acceptable format).
- Append a commit notes entry to `debug/agent_guardrails_commit_notes.md`.

**Phase F — Re-arm:**
- If the active slice is now committed AND active slice was < 6: schedule another loop iteration with ScheduleWakeup, `delaySeconds=180`, `prompt=<this same prompt verbatim>`, `reason="continuing slice <N+1> of analysis-stage-and-dashboard"`.
- If you stopped mid-slice (Phase C left intermediate state on disk): commit nothing, but ScheduleWakeup `delaySeconds=180` with the same prompt, `reason="resuming mid-slice <N>"`.
- If you completed slice 6 cleanly and §6 + §8 pass: do NOT re-arm. Final notes entry titled `ANALYSIS STAGE + DASHBOARD COMPLETE`.

### Failure handling

- **Test failure introduced by your slice** → fix in this iteration. Do not commit. Do not advance.
- **Pre-existing test failure unchanged by your slice** → record in commit notes ("baseline failures: <names>"), commit anyway.
- **Plan inconsistency or genuine blocker** (e.g., a referenced symbol name has drifted): adapt by reading the surrounding code, do the equivalent change at the correct location, document the deviation in commit notes. Do not stop.
- **Smoke failure that looks like an environment issue** (NAS unavailable, container collision with the user's instance, no GPU): record the issue, drop to the in-process `conda run` fallback for the smoke if available; if no fallback is possible, mark the slice provisionally complete with a `BLOCKED-SMOKE` entry in commit notes naming the missing precondition. Continue to the next slice.
- **An ambiguous instruction with two reasonable readings**: pick the reading that (a) deletes more legacy code or avoids adding new abstractions, (b) matches the spikesort stage's existing patterns, (c) keeps the diff smaller. Document the choice. Move on.
- **Hands-off zone collision detected** (operating contract #3 / phase A check fails): write `HALT: hands-off zone collision` notes entry. Do not commit. Do not re-arm. Exit.
- **Disk full / OOM / similar infra failure**: write a `HALT: <reason>` notes entry naming the failure mode and exit without re-arming.

### Tooling notes specific to this repo

- `src/axon_recon/pipeline/stages/spikesort/runner.py` is ~10K lines. NEVER read whole. Always grep first, read narrow ranges around hits. (Read-only here anyway per hands-off contract.)
- `src/axon_recon/pipeline/stages/spikesort/config.py` is the cleanest reference for stage-config dataclasses + YAML parsers + alias maps. Mirror its patterns in `pipeline/stages/analysis/config.py`.
- `debug/debug.runtime.yml` is the canonical YAML for smokes. To add the `analysis:` stage block, append after the existing `spikesort:` block; do not reorder existing blocks.
- `debug/debug.data.yml` carries per-recording `DIV` and per-well `attributes` (genotype/media/plating_density). Slice 1 reads these via the existing runtime-config helpers.
- The repo has TWO equivalent paths (`/mnt/disk15tb/adamm/dev/pkgs/axon_reconstructor/` and `/home/adamm/dev/pkgs/axon_reconstructor/`) — `readlink -f` resolves both to disk15tb. Always work from the disk15tb path (which is what `pwd` returns from either entry).
- Recon fixture for end-to-end smokes: `--target-dataset 11 --limit-wells 1` (dataset 11 well000 has full recon outputs under `Media_Density_T5_02182026_AR/260326/M08073/AxonTracking/000208/well000/recon_outputs/`).
- The NAS mount `/mnt/ben-shalom_nas/` may be stale during the user's debugging session — every container invocation needs `--no-config-mounts` plus explicit `--mount /mnt/disk15tb/adamm/scratch:/mnt/disk15tb/adamm/scratch:rw` and `--mount /mnt/disk15tb/adamm/dev/pkgs/axon_reconstructor/debug:/mnt/disk15tb/adamm/dev/pkgs/axon_reconstructor/debug:ro`. The plan's §3 has the canonical form.

### Stop signals (in priority order)

1. `debug/STOP_AUTONOMOUS_LOOP` file exists at any check → halt immediately, do not commit, do not re-arm.
2. Two consecutive iterations make zero forward progress → write `STALLED` notes entry with diagnosis, do not re-arm.
3. Definition of Done satisfied → write `ANALYSIS STAGE + DASHBOARD COMPLETE` notes entry, do not re-arm.
4. Hands-off zone collision detected → write `HALT: hands-off zone collision` notes entry, do not re-arm.

### One-shot setup, first iteration only

If `git branch --show-current` is NOT `analysis-stage-and-dashboard`:
- Verify `git status` is clean (no unstaged changes).
- Verify HEAD is at a healthy point on `dev_branch2` (or whatever the current main branch is — check `debug/spikesort_merge_cleanup_plan.md` if uncertain; that's the most recently completed plan and its tip is the right baseline).
- `git checkout -b analysis-stage-and-dashboard`.

If `git log --oneline -20` shows no `claude: analysis-stage-and-dashboard, ... (slice 1)` commit AND no in-progress slice-1 work in the working tree:
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

Or one-shot via CLI flag:

```bash
claude --dangerously-skip-permissions \
  -p "$(sed -n '/^## \/loop prompt/,/^---$/p' debug/analysis_stage_and_dashboard_loop_prompt.md | sed '1d;$d')"
```

## Manual interrupts

- **Pause:** `touch debug/STOP_AUTONOMOUS_LOOP` — the next iteration's stop-signal check will halt before any commit.
- **Resume:** `rm debug/STOP_AUTONOMOUS_LOOP` and re-issue `/loop` with the same prompt block.
- **Inspect progress:** `git log --oneline | grep "claude: analysis-stage-and-dashboard.*(slice"` and tail `debug/agent_guardrails_commit_notes.md`.
- **Hard stop:** Ctrl-C the claude session; nothing on disk is left in an unreviewable state because every slice ends in a single coherent commit.
