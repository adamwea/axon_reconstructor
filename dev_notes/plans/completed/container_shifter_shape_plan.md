# Container Shifter-Shape Plan — N ranks inside one container

Status: implementation plan. Sibling to `nersc_shaped_local_affinity_plan.md` (the
target-allocation backbone) and `parallelism_post_migration_cleanup_plan.md`. Reference
material: `debug/guardrails/container_mpi_strategy_note.md` (Option B), and
`debug/guardrails/container_mpi4py_NERSC_optimization_guardrails.md` (acceptance contract).

Same operating contract as the active plans: one slice at a time, `claude:` commit prefix,
append a line to `debug/commit_log.md` after every commit. Validate locally; treat NERSC
behavior as deferred until tested on Perlmutter.

Baseline: `pipeline_v2` HEAD. The `axon-recon-container` wrapper today wraps a single
`docker run` and the entrypoint exec's `axon-reconstructor "$@"`. Multi-rank execution
exists only via host `mpirun -np N axon-recon …` (validated for preprocess) or via
`mpirun -np N axon-recon-container …` (broken — spawns N containers, "runs double",
documented in `debug/mpirun.sh` and the strategy note).

---

## 0. Why This Plan Exists

The strategy note in `debug/guardrails/container_mpi_strategy_note.md` lays out three
options for combining the wrapper with MPI:

- **Option A**: host `mpirun` reaches across the docker namespace into ranks. Requires
  `--ipc=host --pid=host --network=host`, host/container OpenMPI parity, and PMIx wire-
  up. Coupling-prone. Strategy note says "do not pursue first."
- **Option B**: ONE `docker run`, with `mpirun -np N` *inside* the container. Ranks share
  the container's namespaces. No host/container MPI coupling. **Generalizes to NERSC**:
  `srun -n N shifter axon-reconstructor …` is structurally identical to
  `docker run … mpirun -np N axon-reconstructor …` — one image instance, N ranks living
  inside it. Strategy note's medium-term recommendation; estimated "1 day, mostly
  testing."
- **Option C**: status quo — single-rank container for `spikesort.sort`, host mpirun for
  everything else. Two binaries, two mental models, doesn't transfer to NERSC.

This plan delivers **Option B locally**, so that:

1. The lab-server wrapper invocation shape (`axon-recon-container --mpi-ranks N stages …`)
   becomes the same one-line invocation users will type on Perlmutter
   (`srun -n N shifter axon-reconstructor stages …`) modulo the `shifter` swap. The CLI
   tail (`stages …`, `--config`, `--target-dataset`, `--limit-wells`, `--task-backend
   mpi`, etc.) is identical.
2. `mpi_adapter` inside the container sees `MPI.COMM_WORLD.size == N` correctly — so
   `partition_targets_by_mpi_rank` (`distributor.py:156`) actually splits work across
   ranks instead of each container thinking it's rank 0 of size 1 and "running double."
3. The `FileExistsError` at `kilosort4.py:127` that today's
   `mpirun -np N axon-recon-container …` trips is structurally impossible — ranks share
   the same filesystem and the same target-partition decision.

NERSC validation (Shifter pull, `srun shifter`, Cray MPICH swap, CUDA-aware MPI,
multi-node) stays deferred per the guardrails doc. This plan only delivers the local
emulation.

---

## 1. Goal And End State

After this plan lands:

- `axon-recon-container --mpi-ranks N stages …` runs **one** docker container, with
  `mpirun -np N --bind-to none axon-reconstructor stages …` as the inner command.
- `axon-recon-container` without `--mpi-ranks` (or with `--mpi-ranks 1`) is byte-for-byte
  behavior-equivalent to today: single rank, no `mpirun` invocation, no entrypoint
  changes user-visible.
- `--dry-run` prints the resolved `docker run … image mpirun -np N …` line so the
  command is auditable before execution.
- The `containers/axon-recon` image contains a working `mpirun` (OpenMPI 4.x, already
  shipped by the kilosort4 base — verified, not assumed). `mpi4py` already installed.
- The entrypoint passes `mpirun` through as a recognized leader, so the cache/plugin
  preflight checks still happen exactly once before `mpirun` forks ranks.
- Per-rank `CUDA_VISIBLE_DEVICES` partitioning happens inside `mpi_adapter` (rank reads
  `MPI.COMM_WORLD.Get_rank()`, sets `CUDA_VISIBLE_DEVICES = rank % visible_gpu_count`
  before any CUDA-loading import). Generalizes to Shifter without wrapper-side
  partitioning.
- Smoke matrix in §6 passes: 2-rank preprocess on `--target-dataset 11,12 --limit-wells
  1` places each dataset on a distinct rank with no `FileExistsError`; 1-rank parity
  preserved.
- `debug/mpirun.sh` is updated: the commented-out `mpirun … axon-recon-container …`
  block is replaced with a working `axon-recon-container --mpi-ranks N …` example.
- `containers/axon-recon/README.md` documents `--mpi-ranks` and explicitly states the
  one-rank-per-container invariant (host `mpirun -np N axon-recon-container …` is
  unsupported; the wrapper itself owns the rank count via `--mpi-ranks`).
- `debug/guardrails/container_mpi_strategy_note.md` is annotated: "Option B implemented
  locally on <date>, see `container_shifter_shape_plan.md`. NERSC validation deferred."

**Non-goal**: cross-node MPI. **Non-goal**: NERSC Shifter validation. **Non-goal**: MPS
or GPU sharing across ranks on the lab server's single GPU. **Non-goal**: removing the
host `mpirun -np N axon-recon …` (no container) path — that one works and stays.

---

## 2. Inventory (Read These First)

### 2.1 Files we touch

| File:Line | What it is | Action |
|---|---|---|
| `containers/axon-recon/Dockerfile` | image build | VERIFY `openmpi-bin` present (likely already via kilosort4 base); only modify if `mpirun --version` fails |
| `containers/axon-recon/entrypoint.sh:36-42` | recognizer that auto-exec's leading arg | EXTEND to also recognize `mpirun` as a passthrough leader |
| `src/axon_recon/pipeline/container_cli.py:40-95` | `WrapperOptions` + arg parser | ADD `--mpi-ranks N` (default 1), `-n` short alias |
| `src/axon_recon/pipeline/container_cli.py:159-308` | `_parse_options` | parse the new flag into `WrapperOptions.mpi_ranks: int` |
| `src/axon_recon/pipeline/container_cli.py:759-836` | `_build_docker_run_command` | when `mpi_ranks > 1`, insert `mpirun -np N --bind-to none --allow-run-as-root` between `options.image` and `options.container_args` |
| `src/axon_recon/pipeline/container_cli.py:66-95` | `usage()` text | document `--mpi-ranks` |
| `src/axon_recon/pipeline/mpi_adapter.py` | rank/size detection + (new) GPU partitioning | ADD `apply_per_rank_cuda_visible_devices()` helper invoked at startup |
| `src/axon_recon/pipeline/cli.py` (near the existing `current_mpi_context` import) | startup wiring | CALL the GPU-partitioning helper before any stage import that might pull torch/kilosort |
| `containers/axon-recon/README.md` | docs | document `--mpi-ranks`, explicit "host mpirun + container is unsupported" |
| `debug/mpirun.sh` | example commands | replace the broken `mpirun … axon-recon-container …` lines with `axon-recon-container --mpi-ranks N …` |
| `debug/guardrails/container_mpi_strategy_note.md` | strategy note | flip Option B's recommendation status to "implemented locally" |
| `debug/commit_log.md` | per-slice trail | append one line per slice commit |

### 2.2 Files we do NOT touch

- `src/axon_recon/pipeline/runner.py` — already calls `partition_targets_by_mpi_rank`
  via `distributor.py:156`; once `MPI.COMM_WORLD.size == N` inside the container, the
  existing partition logic works unchanged.
- `src/axon_recon/pipeline/execution/distributor.py` — same. No code change; just
  benefits from the corrected `COMM_WORLD`.
- `src/axon_recon/pipeline/cpu_allocation.py` and the `local_affinity` slot machinery —
  per-rank CPU pinning continues to work; we just ask `mpirun` not to fight it
  (`--bind-to none`).
- Stage runners (`stages/preprocess/runner.py`, `stages/spikesort/runner.py`,
  `stages/reconstruct/runner.py`) — no change.

### 2.3 Compatibility check before slice 1

Run once on `pipeline_v2` HEAD to verify the baseline assumption that OpenMPI is
already in the image:

```bash
docker run --rm axon-recon:local which mpirun
docker run --rm axon-recon:local mpirun --version
docker run --rm axon-recon:local python -c "from mpi4py import MPI; print(MPI.Get_library_version())"
```

Expected: `mpirun` resolves under `/usr/bin/` or `/usr/local/bin/`, OpenMPI 4.x reported,
`mpi4py` reports a matching OpenMPI library version. If any of those three fail, slice 1
adds the `apt-get install -y openmpi-bin libopenmpi-dev` line to the Dockerfile and
rebuilds. If they pass, slice 1 collapses to "VERIFY only, no Dockerfile change."

---

## 3. Sequential Slices

### Slice 1 — Image MPI baseline verified

**Goal**: confirm the image can launch `mpirun -np N --allow-run-as-root` and that the
ranks see a sized COMM_WORLD.

Steps:

1. Run the three `docker run` probes from §2.3. Record the OpenMPI version in
   `debug/commit_log.md`.
2. If `mpirun` is missing or `mpi4py` reports a mismatched library version, add
   `RUN apt-get update && apt-get install -y openmpi-bin libopenmpi-dev` to the
   Dockerfile in a location where the apt cache layer is reusable; rebuild
   `axon-recon:local`.
3. Smoke: `docker run --rm axon-recon:local mpirun -np 2 --allow-run-as-root python -c
   "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size())"`.
   Expected output: two lines, one each from rank 0 and rank 1, both reporting size 2.

Acceptance:

- `mpirun --version` works inside the image.
- `mpi4py` reports OpenMPI library version matching the system `mpirun`.
- 2-rank smoke prints both ranks with `size=2`.

Commit: `claude: verify (or install) openmpi-bin in axon-recon image`. Append commit_log.

### Slice 2 — Entrypoint passes `mpirun` through

**Goal**: keep the entrypoint's cache/plugin preflight, but let `mpirun` be the actual
process invoked.

Today (`entrypoint.sh:36-42`):

```bash
if [[ $# -gt 0 && "${1}" == "axon-reconstructor" ]]; then
  exec "$@"
fi
if [[ $# -gt 0 && "${1}" != -* ]] && command -v "${1}" >/dev/null 2>&1; then
  exec "$@"
fi
exec axon-reconstructor "$@"
```

The second branch (any command resolvable on PATH) *already* covers `mpirun` — but only
incidentally. Make it explicit so future readers don't accidentally tighten it:

```bash
if [[ $# -gt 0 && ( "${1}" == "axon-reconstructor" || "${1}" == "mpirun" ) ]]; then
  exec "$@"
fi
if [[ $# -gt 0 && "${1}" != -* ]] && command -v "${1}" >/dev/null 2>&1; then
  exec "$@"
fi
exec axon-reconstructor "$@"
```

Acceptance:

- `docker run --rm axon-recon:local mpirun -np 2 --allow-run-as-root axon-reconstructor
  --help` prints help text from two ranks (or once, depending on whether `--help`
  short-circuits before MPI init — either is fine; the point is the entrypoint passed
  through cleanly).
- All existing smoke commands (`axon-recon-smoke-cli`, plain
  `axon-reconstructor stages …`) still work.

Commit: `claude: entrypoint accepts mpirun as passthrough leader`. Append commit_log.

### Slice 3 — Wrapper `--mpi-ranks` flag

**Goal**: add the flag to `axon-recon-container` and route it through to the inner
command.

Code changes in `src/axon_recon/pipeline/container_cli.py`:

1. Extend `WrapperOptions` (line 40-65) with `mpi_ranks: int = 1`.
2. In `_parse_options` (line 159+), add handlers for `--mpi-ranks N` and `-n N`.
   Validate `N >= 1`; reject `N < 1` with a clear `SystemExit`.
3. In `_build_docker_run_command` (line 759), when `options.mpi_ranks > 1`, replace the
   tail:

   ```python
   cmd.extend([options.image, *options.container_args])
   ```

   with:

   ```python
   if options.mpi_ranks > 1:
       cmd.extend([
           options.image,
           "mpirun", "-np", str(options.mpi_ranks),
           "--allow-run-as-root",
           "--bind-to", "none",
           "axon-reconstructor",
           *options.container_args,
       ])
   else:
       cmd.extend([options.image, *options.container_args])
   ```

   (Explicit `axon-reconstructor` because we're inserting between the entrypoint and the
   args; the entrypoint will see `mpirun` as `$1` and `exec` it.)
4. Update `usage()` (line 66) to document `--mpi-ranks` (and `-n`).
5. Update the dry-run print format if needed (it already calls `shlex.join(cmd)`, so the
   resolved mpirun line is automatically visible — verify by running
   `--dry-run --mpi-ranks 2 stages preprocess --config debug/debug.runtime.yml`).

Acceptance:

- `axon-recon-container --dry-run --mpi-ranks 2 stages preprocess --config debug/debug.runtime.yml`
  prints a docker command whose tail is
  `… axon-recon:local mpirun -np 2 --allow-run-as-root --bind-to none axon-reconstructor stages preprocess --config debug/debug.runtime.yml`.
- `axon-recon-container --dry-run stages preprocess --config debug/debug.runtime.yml`
  (no flag) prints today's command unchanged.
- `axon-recon-container --mpi-ranks 0 …` errors with a clear message.
- Existing wrapper tests (if any) under `pipeline/tests/test_container_cli*.py` still
  pass.

Commit: `claude: add --mpi-ranks to axon-recon-container wrapper`. Append commit_log.

### Slice 4 — Per-rank CUDA partitioning in `mpi_adapter`

**Goal**: when `MPI.COMM_WORLD.size > 1`, each rank claims a deterministic subset of
visible GPUs by setting `CUDA_VISIBLE_DEVICES` before any CUDA-loading import.

This belongs in `mpi_adapter` (not the wrapper) because:

- It generalizes to Shifter at NERSC without wrapper-side logic.
- `mpi_adapter` is already the single source of truth for rank context.
- The wrapper passes `--gpus all` (or whatever); axon-recon-side decides per-rank
  visibility.

Code changes:

1. In `src/axon_recon/pipeline/mpi_adapter.py`, add:

   ```python
   def apply_per_rank_cuda_visible_devices() -> str | None:
       """If running under MPI with size>1, partition visible GPUs across ranks.

       Reads the rank from the existing context detector; reads the currently-visible
       GPU set from CUDA_VISIBLE_DEVICES (or NVML if available); sets
       CUDA_VISIBLE_DEVICES = str(rank % visible_count) before any torch/cuda import.

       Returns the set value (or None if no partitioning was applied).
       Idempotent: skips if size==1 or if rank==0 already sees a single GPU.
       """
   ```

   The function must be called before any module that imports torch/cupy/kilosort.
2. In `src/axon_recon/pipeline/cli.py`, find the existing `current_mpi_context` call site
   (line 20 import, and wherever it's invoked at startup). Add a call to
   `apply_per_rank_cuda_visible_devices()` immediately after rank detection and
   immediately *before* the stage dispatcher imports stage modules.
3. Add fake-MPI unit tests in
   `src/axon_recon/pipeline/tests/test_mpi_adapter.py`:
   - 2-rank, 2-GPU host → rank 0 gets "0", rank 1 gets "1".
   - 2-rank, 1-GPU host → both ranks get "0" (log a warning about contention).
   - 1-rank → no mutation of `CUDA_VISIBLE_DEVICES`.
   - 4-rank, 2-GPU host → ranks 0,2 get "0"; ranks 1,3 get "1" (log warning).

Acceptance:

- Tests pass under fake MPI (`OMPI_COMM_WORLD_RANK` / `_SIZE` env injection — the
  existing `test_mpi_adapter.py` pattern at lines 194-207 already does this).
- Real smoke: `axon-recon-container --gpus all --mpi-ranks 2 --dry-run stages spikesort
  --config debug/debug.runtime.yml --target-dataset 11,12 --limit-wells 1` shows the
  mpirun line; a follow-up real run (not in this slice) would confirm GPU visibility
  per rank in logs.
- Behavior unchanged when `--mpi-ranks` is absent (size==1, function is a no-op).

Commit: `claude: partition CUDA_VISIBLE_DEVICES across MPI ranks in mpi_adapter`. Append
commit_log.

### Slice 5 — Real-data smoke (preprocess, 2 ranks, 2 datasets)

**Goal**: end-to-end validation that the local Option B shape correctly partitions
targets and produces clean outputs.

Smoke:

```bash
axon-recon-container --mpi-ranks 2 stages preprocess \
    --config debug/debug.runtime.yml \
    --target-dataset 11,12 --limit-wells 1 --limit-segments 2 \
    --task-backend mpi --force-restart
```

Expected:

- Logs show two ranks (`rank=0/2`, `rank=1/2`).
- Rank 0 processes dataset 11, rank 1 processes dataset 12 (or vice versa — the
  `partition_targets_by_mpi_rank` policy decides; verify deterministic and disjoint).
- No `FileExistsError`.
- Both datasets' preprocess outputs land on disk under their respective output roots.
- Total wall time roughly half of single-rank serial execution (rough check, not a
  hard gate).

Then verify the 1-rank parity smoke:

```bash
axon-recon-container --mpi-ranks 1 stages preprocess \
    --config debug/debug.runtime.yml \
    --target-dataset 11 --limit-wells 1 --limit-segments 2 \
    --force-restart
```

Expected: byte-for-byte same logs and outputs as today's `axon-recon-container stages
preprocess …` invocation (no `mpirun` in the resolved command; verify with `--dry-run`).

Acceptance:

- 2-rank preprocess completes; targets partitioned; no error.
- 1-rank preprocess identical to today's behavior.
- `debug/commit_log.md` records the wall-time observation and target partition map.

Commit: `claude: smoke validation of --mpi-ranks 2 preprocess on dataset 11+12`. Append
commit_log.

### Slice 6 — Spikesort behavior with `--mpi-ranks` and GPU bound

**Goal**: define and enforce the spikesort.sort policy under multi-rank.

The lab server has 1 GPU. Two ranks both wanting to run Kilosort on it is bad:

- Without MPS, CUDA contention on a single device is slow and can OOM.
- For 2 datasets × 1 well, serializing through 1 rank is empirically faster than
  splitting one GPU.

Decision (recorded in §4 below): when `spikesort.sort` (or any phase whose
`engine == "kilosort4"` and `device == "cuda"`) runs with `MPI.COMM_WORLD.size >
visible_gpu_count`, fail fast with a clear error message pointing at this plan and at
the `--mpi-ranks` flag.

Code changes:

1. In `src/axon_recon/pipeline/stages/spikesort/runner.py` (sort phase entry — find via
   `grep -n "kilosort4\|engine.*sort" runner.py`), add a precondition check that reads
   `current_mpi_context()` and the visible CUDA device count; raise a typed exception
   with the actionable message:

   ```
   spikesort.sort: MPI size N=2 exceeds visible GPU count G=1. Kilosort4 cannot
   safely share a single GPU across ranks. Re-run with --mpi-ranks 1 for the sort
   stage, or use stage-split jobs (CPU preprocess multi-rank, GPU sort single-rank,
   CPU reconstruct multi-rank). See container_shifter_shape_plan.md §4.
   ```
2. CPU-only stages (`preprocess`, `reconstruct`, `analysis`) skip this check — they're
   fine with `N > visible_gpu_count`.

Acceptance:

- `axon-recon-container --gpus all --mpi-ranks 2 stages spikesort.sort …` errors with
  the message above (not a hang, not a CUDA OOM).
- `axon-recon-container --gpus all --mpi-ranks 1 stages spikesort.sort …` runs as today.
- `axon-recon-container --mpi-ranks 2 stages preprocess …` (no GPU) passes through.
- Unit test for the precondition with fake MPI + monkey-patched GPU count.

Commit: `claude: fail fast when spikesort.sort ranks exceed visible GPUs`. Append
commit_log.

### Slice 7 — Docs, README, strategy-note status flip

**Goal**: make the new shape discoverable and document the unsupported-host-mpirun
boundary.

Edits:

1. `containers/axon-recon/README.md`:
   - Add a new section "Multi-rank inside one container" after "Local Wrapper",
     showing `--mpi-ranks N` usage and the rule "host `mpirun -np N axon-recon-container
     …` is not supported; the wrapper owns rank count via `--mpi-ranks`."
   - Document the NERSC parallel: same CLI tail under `srun -n N shifter
     axon-reconstructor …`.
   - Document the GPU policy from Slice 6.
2. `debug/mpirun.sh`: delete the commented-out `mpirun … axon-recon-container …`
   blocks; replace with a working `axon-recon-container --mpi-ranks 2 stages preprocess
   …` example. Keep the host `/usr/bin/mpirun -np 2 axon-recon …` example (that one
   still works for the no-container path).
3. `debug/guardrails/container_mpi_strategy_note.md`: append a status line to the
   "Option B" section: `**Status (2026-MM-DD)**: implemented locally by
   container_shifter_shape_plan.md. NERSC validation deferred per guardrails doc.`
4. `debug/plans/active/container_shifter_shape_plan.md`: append a "Completed" marker
   when all slices land; move to `debug/plans/completed/` after the final commit.

Acceptance:

- README mentions `--mpi-ranks` and `srun shifter`.
- `debug/mpirun.sh` example actually runs (smoke per Slice 5 still passes).
- Strategy note's Option B status reflects local implementation.

Commit: `claude: docs sweep for --mpi-ranks shifter-shape pivot`. Append commit_log.

---

## 4. Decisions

| Question | Choice | Rationale |
|---|---|---|
| Flag name | `--mpi-ranks N` with `-n N` short alias | `--mpi-ranks` is verbose and unambiguous (no collision with `-n` of any existing wrapper option; verify in slice 3). Mirrors NERSC's `srun -n` for muscle-memory parity. |
| Default rank count | `1` | Zero behavior change for existing users. Multi-rank is opt-in. |
| GPU partitioning location | `mpi_adapter.apply_per_rank_cuda_visible_devices()` | Generalizes to Shifter without wrapper-side logic. Single source of truth for rank-aware setup. |
| Partition policy | `CUDA_VISIBLE_DEVICES = str(rank % visible_gpu_count)` | Round-robin; deterministic; sane for both balanced (N==G) and oversubscribed (N>G) cases. Warn on oversubscription. |
| `spikesort.sort` with N > visible GPUs | Hard error before any CUDA allocation | Avoids slow CUDA contention failures and confusing OOMs. User explicitly opts in to single-rank for sort. |
| `mpirun --bind-to` | `none` | Let `local_affinity` (from `nersc_shaped_local_affinity_plan.md`) own CPU pinning. Avoid double-binding fights between mpirun and `os.sched_setaffinity`. |
| `--allow-run-as-root` | Always set inside container | The container may run as root or as a non-root mapped user; OpenMPI refuses to launch as root unless told it's OK. Setting unconditionally is safe because there's no privileged operation here. |
| Host `mpirun -np N axon-recon-container …` | Documented as unsupported | The "runs double" problem; Option A is the only way to make it work and we explicitly chose Option B. |
| OpenMPI version coupling to NERSC Cray MPICH | None | Local container uses OpenMPI for local emulation. At NERSC, Shifter's `--module=gpu,cuda-mpich` swaps Cray MPICH at runtime, fully replacing the in-image MPI. No coupling. |

---

## 5. Smoke Matrix

Run all of these locally on the lab server. Each row is a separate command; expected
behavior is the "Outcome" column.

| # | Command | Outcome |
|---|---|---|
| A | `axon-recon-container --dry-run stages preprocess --config debug/debug.runtime.yml` | Today's docker command, no `mpirun` in tail |
| B | `axon-recon-container --dry-run --mpi-ranks 2 stages preprocess --config debug/debug.runtime.yml` | Docker tail includes `mpirun -np 2 --allow-run-as-root --bind-to none axon-reconstructor stages preprocess …` |
| C | `axon-recon-container --mpi-ranks 1 stages preprocess --config debug/debug.runtime.yml --target-dataset 11 --limit-wells 1 --limit-segments 2 --force-restart` | Same logs/outputs as today's no-flag invocation |
| D | `axon-recon-container --mpi-ranks 2 stages preprocess --config debug/debug.runtime.yml --target-dataset 11,12 --limit-wells 1 --limit-segments 2 --task-backend mpi --force-restart` | Two ranks, disjoint datasets, no FileExistsError |
| E | `axon-recon-container --gpus all --mpi-ranks 1 stages spikesort.sort --config debug/debug.runtime.yml --target-dataset 11 --limit-wells 1 --force-restart` | Kilosort runs as today, single rank |
| F | `axon-recon-container --gpus all --mpi-ranks 2 stages spikesort.sort --config debug/debug.runtime.yml --target-dataset 11,12 --limit-wells 1 --force-restart` | Fail fast with the message from Slice 6 |
| G | `pytest src/axon_recon/pipeline/tests/test_mpi_adapter.py` | Pass, including new GPU-partition tests |
| H | `axon-recon-container --mpi-ranks 0 stages preprocess …` | SystemExit with clear message |

Smoke A, B, G, H are CI-friendly (fast, deterministic, no real data). C–F require the
debug runtime config and a real dataset on the lab server.

---

## 6. Non-Goals

- **Cross-node MPI**. Single-node Option B only.
- **Host `mpirun` → container ranks (Option A)**. Out of scope per strategy note.
- **NERSC Shifter validation**. Deferred until run on Perlmutter.
- **CUDA-aware MPI (`MPICH_GPU_SUPPORT_ENABLED=1`)**. NERSC-only concern.
- **MPS / multi-process service for GPU sharing**. Lab server has 1 GPU; not worth the
  complexity for 2 datasets.
- **Removing the host-binary `mpirun -np N axon-recon …` path**. It works for non-GPU
  stages and stays as an alternative when users want to avoid the container.
- **Auto-derivation of `--mpi-ranks` from CPU/GPU topology**. The user explicitly
  chooses. The `local_affinity` plan handles per-rank CPU sizing; rank count stays
  manual.
- **Slurm/sbatch templates**. Documented as part of the NERSC plan, not here.

---

## 7. Main Risks

1. **OpenMPI version mismatch inside the image**. If the kilosort4 base bumps OpenMPI in
   a future rebuild and `mpi4py` was built against an older version, ranks may fail with
   `PMIx_Init failed` or initialize a singleton COMM_WORLD silently. Mitigation: slice 1
   verifies the version pair and pins both in the Dockerfile if needed. Watch the
   commit_log entry from slice 1.
2. **CUDA imported before `apply_per_rank_cuda_visible_devices()` runs**. If any module
   imported by `axon-reconstructor`'s entry point pulls torch/cupy/kilosort at top-level
   before `mpi_adapter` runs, the rank-0 view of GPUs becomes shared across all ranks.
   Mitigation: slice 4 places the call in `cli.py` *before* the stage dispatcher imports
   stage modules. Verify with `python -c "import sys; print('torch' in sys.modules)"`
   after `mpi_adapter` initialization.
3. **`--allow-run-as-root` security note**. We unconditionally pass `--allow-run-as-root`
   to in-container `mpirun`. This is only an OpenMPI safety lock, not a real privilege
   escalation. Documented but worth noting.
4. **`/dev/shm` pressure from N ranks sharing the container**. Spikeinterface's shared-
   memory recording uses `/dev/shm`. With 2 ranks each potentially holding a shared-
   memory recording, `--shm-size` (currently set via runtime YAML `container_caps`) may
   need to grow. Mitigation: slice 5 smoke watches for `OSError: [Errno 28] No space
   left on device` and recommends a `--shm-size` bump if seen.
5. **`mpirun --bind-to none` + `local_affinity` interaction**. If
   `nersc_shaped_local_affinity_plan.md`'s slice 5 (worker affinity) hasn't landed yet,
   ranks may end up unbound and stomp each other's CPU caches. Mitigation: this plan
   does not assume the affinity plan has landed; ranks remain functionally correct just
   slower without affinity. Document the dependency in the README addition.
6. **Dry-run drift**. If `_build_docker_run_command` and the actual exec path drift, dry-
   run output stops matching reality. Mitigation: there's only one builder; `main()`
   calls it for both dry-run and exec (`container_cli.py:847-852`). No drift possible.

---

## 8. Dependencies / Sequencing

- **No hard dependency** on `nersc_shaped_local_affinity_plan.md`. They compose: this
  plan delivers the *rank* layer; the affinity plan delivers the *thread/CPU pinning*
  layer. They're independent because `mpirun --bind-to none` cleanly cedes binding to
  whatever lives inside.
- **No dependency** on `parallelism_post_migration_cleanup_plan.md`. Pure addition.
- This plan's slices 1–4 are mechanically independent and could in principle land out
  of order; recommended order is sequential because slice 4's tests are easier to write
  once the wrapper exists (slice 3) and the entrypoint passes mpirun (slice 2). Slices
  5–6 are validation; slice 7 is docs.

---

## 9. Followups (Not In Plan Scope)

- **Slurm/Shifter sbatch templates** under `debug/perlmutter_*.sbatch.example`,
  authored after first successful NERSC test run.
- **`--mpi-ranks auto`** mode that consults `nersc_shaped_local_affinity_plan.md`'s
  `tasks_per_node` derivation. Wait until the affinity plan lands.
- **Multi-node MPI** via `mpirun --host` or `srun` on a Slurm-managed cluster. Requires
  Option A or a Singularity port; out of scope today.
- **CUDA-aware MPI smoke** (`MPICH_GPU_SUPPORT_ENABLED=1`). NERSC-only.
- **Replace `mpirun` with `srun` shape inside the container** for closer NERSC parity.
  Not needed locally; in-container `mpirun` is fine.
