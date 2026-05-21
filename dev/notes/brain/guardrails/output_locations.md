# Output location guardrail

## Contract

| Kind of artifact | Where it lives | Tracked in repo? |
|---|---|---|
| Source code | `/global/homes/a/adammwea/dev/pkgs/<package>/` | ✅ Yes — that's the canonical home |
| Pipeline YAML config (`debug.runtime.yml`, `debug.data.yml`) | `dev/debug_NERSC/` | ✅ Yes |
| Sbatch scripts, rerun helpers, smoke scripts | `dev/debug_NERSC/jobs/<subdir>/` | ✅ Yes (the `.sh` and `.sbatch` files) |
| Plans, trackers, guardrails, memory, commit log | `dev/notes/` | ✅ Yes |
| Raw / preprocessed / analyzed recording data (production / reference) | `/pscratch/sd/a/adammwea/{raw_data,analyzed_data}/` | ❌ Never |
| **Iteration outputs** (anything Claude produces while developing) | `/pscratch/sd/a/adammwea/dev_outputs/<feature>/...` | ❌ Never |
| Per-run logs (anything produced by an axon-recon invocation) | `/pscratch/sd/a/adammwea/run_logs/<stage>/` or `/pscratch/sd/a/adammwea/smoke_logs/<stage>/` | ❌ Never |
| Job-specific log dirs (`logs_<timestamp>/` created by rerun scripts) | Inside `dev/debug_NERSC/jobs/<subdir>/` is acceptable PROVIDED gitignored | ❌ Never (gitignored) |
| Shifter / podman image cache | `/tmp/105000_hpc/` (NERSC-managed) | ❌ Never (ephemeral) |
| Container scratch overlay | Forbidden as a permanent location — see `package_contracts.md` §4 | N/A |

## Why

- pscratch is the high-throughput scratch filesystem on Perlmutter. It has 8-week auto-purge for unused files; it's designed for the I/O hot path. Code in pscratch is fragile (gets purged); outputs there are exactly what pscratch is for.
- `/global/homes` is the durable user home. Source code, configs, plans — anything we want to survive across sessions — lives here.
- Mixing log output into the repo tree fights `git status`, bloats clones, and makes diffs noisy. It also makes accidental commits of personal log content trivial. Gitignore patterns are explicit and audited.
- The shifter bind-mount allowlist (per the user memory note `reference-nersc-shifter-bindmounts`) constrains what paths CAN be mounted into the container. `/global/homes` is NOT mountable; `/pscratch` IS. That's a NERSC infrastructure constraint, not a preference.

## Reference vs iteration outputs — the hard rule

Existing per-stage outputs under `/pscratch/sd/a/adammwea/analyzed_data/...` are the **ground-truth reference** for desired behavior. They were produced by prior runs; downstream analysis depends on them; users may have inspected them. **Claude does not mutate them during iteration work.**

When developing a new feature or iterating on a fix, write all run outputs under a SEPARATE pscratch tree:
- `/pscratch/sd/a/adammwea/dev_outputs/<feature_or_slice_name>/...`
- Mirror the `analyzed_data` directory structure under there (i.e. `<date>/<chip>/AxonTracking/<rec>/<well>/...`) so paths line up for diffing.

The user reviews iteration outputs against the reference. If a behavior change is desired and validated, the user explicitly promotes the iteration outputs to the reference dir; Claude does not.

## Concrete sub-rules

1. **Scripts that produce logs write them under `/pscratch/sd/a/adammwea/run_logs/<scope>/<timestamp>/`** by default. Override-able via env var or CLI arg for one-off debugging, but the default never lands in the repo.

2. **Rerun helpers (e.g. `rerun_again.sh`) that have a `LOGDIR` variable** default to a pscratch path. If a contributor has a strong reason to keep logs adjacent to the script, the path goes under `dev/debug_NERSC/jobs/<subdir>/logs_<timestamp>/` AND that subpath must be in `.gitignore`.

3. **Never commit log files.** `git status` should never show a `dev/**/logs*` entry as untracked. If it does, either the gitignore is missing a pattern or the run is writing to the wrong location. Fix the gitignore OR move the log location; don't `git add` it.

4. **Outputs of axon_recon stages always land under the well's pscratch output dir.** The YAML `output_root: /pscratch/sd/a/adammwea/analyzed_data/...` is the canonical home. Stages do NOT write into the repo tree.

5. **Shifter image is the source-of-record for code at runtime.** No bind-mount workarounds for "fast iteration" or "shifterimg pull is broken right now". When shifterimg is misbehaving, wait or escalate to NERSC — see `package_contracts.md` §4.

## Gitignore patterns

The repo's root `.gitignore` carries the following patterns (or equivalent — verify periodically):

```gitignore
# Per-run log directories created by axon-recon stages, rerun scripts, smoke runs
dev/debug_NERSC/logs/
dev/debug_NERSC/jobs/**/logs/
dev/debug_NERSC/jobs/**/logs_*/

# Loose runtime log files
axon_reconstruction*.log
*.log

# Any pscratch checkout overlay that accidentally appears as a sibling
# (shouldn't happen — but belt-and-suspenders)
axon_recon_overlay/
```

A `find dev/ -name 'logs_*' -o -name '*.log' | xargs git check-ignore` audit should show every match is ignored.

## Tests / verification

- `git status -s` after a smoke run should show no untracked log files. If it does, the rerun script wrote to the wrong location.
- The `.gitignore` patterns are tested implicitly by every commit. CI could grow a "no logs in tree" check if drift becomes a problem.

## Open exceptions / follow-ups

- A few historical logs are in `dev/debug_NERSC/logs/` (`restore_*.log`) that pre-date this guardrail. They get gitignored (already done at root .gitignore) but not committed.
- The job-specific `logs_<timestamp>/` dirs from the `sans_bombcell_rerun` work are currently in the working tree; they're gitignored by patterns above but never staged. Cleanup is optional — they'll get auto-purged from pscratch (if symlinked there) or just sit as dead files.
- Future improvement: update `rerun_again.sh` (and any similar scripts) to default to a pscratch `LOGDIR`. Currently they default to `dev/debug_NERSC/jobs/<subdir>/logs_<timestamp>/` which works as long as gitignore is right, but the pscratch default would be cleaner.
