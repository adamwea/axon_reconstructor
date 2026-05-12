# Loop prompt — autonomous container shifter-shape + NERSC affinity finish

Paste the block below as the `/loop` argument in a fresh Claude Code session running with permissions off (`--dangerously-skip-permissions` or equivalent). Each loop iteration drives forward progress on TWO plans plus a README sweep, in this order:

1. `debug/plans/active/container_shifter_shape_plan.md` (7 slices) — bring up `--mpi-ranks N` inside one container so the local invocation shape mirrors `srun -n N shifter …` at NERSC.
2. `debug/plans/active/nersc_shaped_local_affinity_plan.md` — finish slices 10 (container readiness), 11 (MPI backend validation), and 12 (Slurm/srun shape). Slices 1–9 are already landed (verified by the `task_affinity_applied`, `thread_env_applied`, and `task_allocation_target_assigned` events visible in current run logs).
3. README sweep — `containers/axon-recon/README.md` documents every supported run mode (local host, local host + mpirun, local container, local container + `--mpi-ranks`, NERSC interactive `srun shifter`, NERSC sbatch `srun shifter` + multi-rank). Today's README only covers two of those.

The loop is self-resuming: it always re-reads state from disk and git, picks up wherever the previous iteration left off, and exits when all three goals are satisfied.

---

## /loop prompt (copy from here to end-of-file)

You are an autonomous engineer in `/mnt/disk15tb/adamm/dev/pkgs/axon_reconstructor`. Your job is to drive `debug/plans/active/container_shifter_shape_plan.md` to completion, then finish the remaining slices of `debug/plans/active/nersc_shaped_local_affinity_plan.md` (slices 10, 11, 12), then do a final README sweep, all with no human intervention. Do not ask the user any questions. Do not stop and wait for confirmation. Make decisions, make commits, keep moving.

### Operating contract (non-negotiable)

1. **Plans are authoritative.** Both plans define slices, acceptance criteria, smoke matrices, decisions, and non-goals. Read the relevant plan section on every iteration; do not paraphrase from memory.
2. **Guardrails are locked.** Treat every file in `debug/guardrails/*` as read-only law. The two most relevant for this loop are `container_mpi_strategy_note.md` (Option B is the chosen path; Option A and Option C are explicitly out of scope) and `container_mpi4py_NERSC_optimization_guardrails.md` (NERSC validation stays deferred — do not claim NERSC-ready from local Docker tests). Consult before any non-obvious decision. Do not modify guardrail files unless an active slice explicitly tells you to (e.g., container_shifter_shape slice 7 flips a status line in `container_mpi_strategy_note.md`).
3. **Conda env.** All host Python invocations: `conda run -n axon_recon <cmd>`. Container Python invocations go through `axon-recon-container` or `docker run --rm axon-recon:local …`. Never assume a different env.
4. **Commit prefix.** Every commit you make in `axon_reconstructor` starts with `claude:` and ends with the trailer `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>`. One slice = one commit. If you also need to commit in the sibling `SLAy/` repo (unlikely for this loop), use that repo's existing prefix style (`fix (component): …`) and the same Co-Authored-By trailer.
5. **Commit log.** After every commit, append a dated entry to the top of `debug/commit_log.md` (find the existing format by reading the file). Include: what slice landed, what tests/smokes ran, what guardrails were consulted, anything surprising.
6. **Tests gate commits.** `conda run -n axon_recon python -m pytest src/axon_recon/pipeline/tests/ -q -x --ignore=src/axon_recon/pipeline/tests/test_progress.py` MUST pass before any commit. For slices that touch spikesort code paths, also run `conda run -n axon_recon python -m pytest src/axon_recon/pipeline/stages/spikesort/tests/ -q`. If anything fails, fix it in the same slice — do not commit a red tree. Snapshot the pre-slice baseline if you suspect a pre-existing failure: `git stash && pytest && git stash pop`.
7. **Smokes gate commits when the slice's acceptance section requires them.** Run only the smokes the slice's Acceptance subsection lists. The container_shifter_shape smoke matrix is §5; the affinity plan's initial smoke is at the bottom of the plan. Capture stdout to `/tmp/smoke_<slice>_<label>.log` for commit notes. Real-data smokes (the ones that use `--target-dataset 11,12 --limit-wells 1`) are slow (5–10 min); only run them when the plan section requires.
8. **Image rebuilds.** Whenever a slice modifies `containers/axon-recon/Dockerfile`, `containers/axon-recon/entrypoint.sh`, or any installed source the image bakes, run `containers/axon-recon/build_local_image.sh --image axon-recon:local` once before the smoke. `axon-recon-container` will also rebuild automatically when its fingerprint detects source drift, but explicit rebuilds make the failure surface clearer.
9. **No scope creep.** Touch only what the active slice's plan section names. Do not refactor neighboring code. Do not "while I'm here" anything. The README sweep is its own dedicated stage at the end — do not start it early.
10. **No backward-compat shims, no `--no-verify`, no `git config` changes, no force pushes, no destructive git ops** without explicit instruction written in the active plan. The strategy note (`container_mpi_strategy_note.md`) explicitly forbids pursuing Option A (host mpirun → container ranks). Do not implement Option A under any circumstance, even if it looks easy.
11. **Halt condition.** When the goals in `### Goal hierarchy` below are fully satisfied AND the README has all six run-mode sections documented, write a final commit notes entry titled `CONTAINER SHIFTER + NERSC AFFINITY COMPLETE` and exit the loop by NOT scheduling another iteration (do not call ScheduleWakeup, do not re-arm).

### Goal hierarchy

This loop has three goals. Do not start a later goal until the earlier ones are satisfied.

**Goal 1 — container_shifter_shape_plan.md to DoD.** All 7 slices landed, each with its own `claude:` commit. The plan's §1 end-state checklist must all be checkable:
- `axon-recon-container --mpi-ranks N stages …` runs one container with N ranks inside
- Default behavior (no flag) byte-for-byte identical to today
- `--dry-run` shows resolved mpirun line
- Image has working `mpirun` (verified, not assumed)
- Entrypoint passes mpirun through
- Per-rank `CUDA_VISIBLE_DEVICES` partitioning in `mpi_adapter`
- Smoke matrix §5 rows A–H all pass
- `debug/mpirun.sh` updated to a working example
- `containers/axon-recon/README.md` documents `--mpi-ranks`
- `debug/guardrails/container_mpi_strategy_note.md` Option B status flipped to "implemented locally on <date>"

**Goal 2 — nersc_shaped_local_affinity_plan.md slices 10, 11, 12.** Slices 1–9 are already landed (don't redo them). Read the plan; pick up at slice 10.

- **Slice 10 (Container readiness)**: this overlaps with container_shifter_shape's smoke matrix. The affinity check inside the container is already exercised by the existing logs (`task_affinity_applied cpus=0-9` proves it works). Slice 10's acceptance is: container smoke with 1 dataset + 2 wells (or 2 datasets × 1 well) shows two distinct CPU sets assigned. Run `axon-recon-container --mpi-ranks 2 stages preprocess --config debug/debug.runtime.yml --target-dataset 11,12 --limit-wells 1 --limit-segments 2 --force-restart` and grep the log for `task_affinity_applied` lines — there should be two with distinct `cpus=` ranges. Commit `claude: validate task allocation inside axon-recon container (affinity slice 10)`. This may be a no-op-commit slice (no code change, just a validation + commit log entry).

- **Slice 11 (MPI backend)**: the plan calls for a `backend: mpi` value in `resources.task_allocation.backend`. The MPI partition mechanism already exists in `partition_targets_by_mpi_rank` (`pipeline/execution/distributor.py:156`) and is exercised by host `mpirun -np 2 axon-recon …` runs. What's missing is: (a) wiring `backend: mpi` so that when MPI is detected, the allocation plan partitions targets across ranks BEFORE local CPU fanout; (b) fake-MPI tests; (c) rank metadata in logs/summaries. Read slice 11's acceptance criteria in full before designing. The wrapper-side `--mpi-ranks` from Goal 1 supplies the rank count; this slice connects it to the existing task allocation plan. Commit `claude: wire backend=mpi into task allocation (affinity slice 11)`.

- **Slice 12 (Slurm/NERSC backend)**: same allocation vocabulary, just translated to Slurm directives. Acceptance is documentation-heavy. Deliverables:
  - One example sbatch script under `debug/perlmutter_preprocess.sbatch.example` showing CPU-only stages with `srun -n N shifter axon-reconstructor stages preprocess … --task-backend mpi`.
  - One example sbatch script under `debug/perlmutter_spikesort.sbatch.example` showing the GPU sort case (likely `-n 1` because of the GPU contention rule from container_shifter_shape slice 6 — re-state that rule in the example's comments).
  - A `backend: slurm` parser in the task allocation config that reads `SLURM_NTASKS`, `SLURM_CPUS_PER_TASK`, `SLURM_NODELIST` if present. It does NOT need to run a real Slurm job to test; fake-Slurm env injection in unit tests is sufficient. NERSC validation stays deferred.
  - Commit `claude: add slurm backend + perlmutter sbatch examples (affinity slice 12)`.

**Goal 3 — README sweep (single big commit).** Edit `containers/axon-recon/README.md` so it documents EVERY supported run mode in a single "Run modes" section, in this order:

1. **Local host, single-process** — `axon-reconstructor stages …` (no container, no MPI). Existing default for `axon-recon` users on the lab server.
2. **Local host + mpirun (multi-rank, no container)** — `/usr/bin/mpirun -np N axon-reconstructor stages … --task-backend mpi`. The validated path from `debug/mpirun.sh`.
3. **Local container, single rank** — `axon-recon-container stages …`. Today's default container path.
4. **Local container + multi-rank (NEW from Goal 1)** — `axon-recon-container --mpi-ranks N stages … --task-backend mpi`. The Shifter-shape pivot.
5. **NERSC interactive (Shifter)** — `salloc … --image=…`, then `srun -n N shifter axon-reconstructor stages …`. The interactive-node form.
6. **NERSC sbatch (Shifter, multi-rank)** — full `#SBATCH` script with `--image=`, `--module=gpu,cuda-mpich`, `srun shifter axon-reconstructor stages … --task-backend mpi`. Point at the example scripts created in affinity slice 12.

For each mode, give: one-line description, the exact command, what stages are appropriate, GPU/CPU constraints, and a "see also" link to the plan or guardrail that justifies it. End the section with a quick-reference table mapping (machine context, scale) → mode number. Commit `claude: README sweep documents all six run modes`.

### Per-iteration procedure

Run these phases in order. Each phase is short — most iterations finish in one phase.

**Phase A — Re-orient (every iteration, no exceptions):**
- `git status --short` and `git log --oneline -30`. From the log, count `claude:` commits matching slice labels (e.g. `(slice 1)`, `(slice 2)`, … and `(affinity slice 10)`, `(affinity slice 11)`, `(affinity slice 12)`, plus the `README sweep` final commit) to determine which goals are landed.
- Read the active plan's slice section end-to-end. The plan is the source of truth.
- Read the most recent ~30 lines of `debug/commit_log.md` to absorb prior-iteration context.
- Determine the **active work unit** using this priority order:
  1. Lowest-numbered unlanded slice in `container_shifter_shape_plan.md` (slices 1–7).
  2. Then affinity slice 10, 11, 12 in order.
  3. Then the README sweep (Goal 3).
  4. If all three goals' commits are present → run the halt check (next bullet).
- If all goals are landed, run the §1 end-state checklist of `container_shifter_shape_plan.md` AND the slice-12 acceptance of the affinity plan AND a visual scan of the README's "Run modes" section. If anything is unchecked → treat it as a remediation work unit; the active slice becomes "post-goals cleanup". If everything is checked → write `CONTAINER SHIFTER + NERSC AFFINITY COMPLETE` notes entry and exit (do not re-arm).

**Phase B — Plan the active slice's iteration:**
- Re-read the active slice's section in the relevant plan in full.
- Use TaskCreate to enumerate the slice's subsections as discrete tasks (typically 3–6 tasks per slice). Mark each `in_progress` when started, `completed` when done. Update TodoWrite after every commit.
- If the slice is partially in progress (uncommitted local changes from a prior iteration), reconcile: keep good local work, do not blow it away.
- If the slice is large enough that one iteration won't finish it, decide a stopping point that leaves the working tree in a coherent intermediate state. The next iteration will resume from `git status` + commit log notes.

**Phase C — Execute:**
- Make the edits the slice prescribes. Keep edits surgical.
- Use `Read` for files; `Edit`/`Write` for changes; `Bash` for tests, smokes, grep, git, docker.
- Run the slice's tests (pipeline + spikesort suites) as you go. Don't wait until the end.
- For any plan ambiguity, prefer the option that keeps changes smaller, deletes more legacy code, and matches existing patterns. Document the choice in the commit notes.
- When in doubt about an MPI/container/CUDA decision, consult `debug/guardrails/container_mpi_strategy_note.md` (Option B is the only path) and `debug/guardrails/container_mpi4py_NERSC_optimization_guardrails.md`.

**Phase D — Verify:**
- Pipeline test suite green (or: same baseline as pre-slice).
- Slice's Acceptance subsection items all pass. For grep-based acceptance, run the grep verbatim and paste the output into commit notes (even when empty).
- For container slices, dry-run BOTH `--mpi-ranks 1` (parity) and `--mpi-ranks 2` (new) before committing. The dry-run docker command difference is your acceptance evidence.
- For real-data smokes (5–10 min), only run when the plan section explicitly requires. Capture stdout to `/tmp/smoke_<slice>_<label>.log`.

**Phase E — Commit + log:**
- `git add` only the files the slice's plan section names. Use specific paths, never `git add -A`.
- Write a focused commit message: subject line `claude: <slice label> — <one-line what>`, body 1–3 sentences on the why and a reference to the plan path.
- After commit, append the dated entry to `debug/commit_log.md` (match existing format; read the file to confirm).
- `git status` after commit should be clean.

**Phase F — Re-arm or halt:**
- If the halt condition (Phase A bullet) is satisfied, write `CONTAINER SHIFTER + NERSC AFFINITY COMPLETE` notes entry and exit.
- Otherwise, schedule the next iteration. Default cadence: 60s for slices in active flow (cache stays warm), 300s+ if you're between slices waiting on a long smoke. Pass the same `/loop` prompt back via ScheduleWakeup `prompt`.

### Common pitfalls (read once, internalize)

- **CUDA import order**: per-rank `CUDA_VISIBLE_DEVICES` partitioning (container_shifter_shape slice 4) MUST happen before any torch/cupy/kilosort import. If `cli.py`'s stage dispatcher imports a stage module that does `import torch` at top level, your partitioning is too late. Verify with `python -c "import sys; print('torch' in sys.modules)"` right after the `mpi_adapter` call. If torch is already loaded, find the import that pulled it in and either defer it or move the partition call earlier.
- **OpenMPI vs Cray MPICH**: local container uses OpenMPI for local emulation. NERSC Shifter swaps in Cray MPICH at runtime via `--module=gpu,cuda-mpich`. Do not pin a Cray MPICH version in the Dockerfile — that breaks local emulation and isn't what NERSC uses.
- **GPU contention**: container_shifter_shape slice 6 enforces fail-fast when ranks > visible GPUs for `spikesort.sort`. Do not work around this with MPS or by silently round-robining — the strategy note's open question explicitly recommends serializing through one rank for sort.
- **`--bind-to none`**: in-container `mpirun` uses `--bind-to none` so the existing `local_affinity` machinery (already landed in slices 1–9) keeps owning CPU pinning. Do not fight it by adding `--bind-to core` or `--map-by ppr:…`.
- **Affinity slice 11 ≠ container_shifter_shape slice 4**: they touch overlapping code (`mpi_adapter.py`). Slice 4 of container_shifter_shape adds the CUDA partition function and wires it in `cli.py`. Slice 11 of affinity wires `backend: mpi` so the existing target-partition logic is gated on it being explicitly chosen. Different files, related concerns — handle in plan order (container slice 4 first, then affinity slice 11).
- **NERSC sbatch examples are not tested**: slice 12 of affinity ships `.example` files. They are documentation; they cannot be smoked locally. The guardrail document explicitly defers NERSC validation. Do not pretend you've run them.

### Required reads before first iteration

- `debug/plans/active/container_shifter_shape_plan.md` (full)
- `debug/plans/active/nersc_shaped_local_affinity_plan.md` slices 10, 11, 12 + non-goals
- `debug/guardrails/container_mpi_strategy_note.md` (full)
- `debug/guardrails/container_mpi4py_NERSC_optimization_guardrails.md` (full)
- `containers/axon-recon/README.md` (current state — you'll be rewriting it in Goal 3)
- `containers/axon-recon/Dockerfile` (knowledge baseline before slice 1)
- `containers/axon-recon/entrypoint.sh` (knowledge baseline before slice 2)
- `src/axon_recon/pipeline/container_cli.py` (knowledge baseline before slice 3)
- `src/axon_recon/pipeline/mpi_adapter.py` (knowledge baseline before slice 4 and affinity slice 11)
- `debug/commit_log.md` (last 30 lines, every iteration)

### Halt acknowledgment

When you exit, the final `debug/commit_log.md` entry should be a single-line confirmation:

```
CONTAINER SHIFTER + NERSC AFFINITY COMPLETE — container_shifter_shape_plan.md (slices 1–7), nersc_shaped_local_affinity_plan.md (slices 10–12), README sweep all landed. Plans moved to debug/plans/completed/.
```

Move the two plan files from `debug/plans/active/` to `debug/plans/completed/` in your last commit, alongside this loop prompt file.

### Begin

Start with Phase A. If your `git log` shows zero `claude:` commits matching this loop's slice labels, your first action is to start container_shifter_shape slice 1 (image MPI baseline verified — run the three `docker run` probes from §2.3 of that plan).
