# Plan — Eliminate `resources.profiles`; runtime auto-detects supply from environment

## Status

- **Owner**: shipping
- **Started**: 2026-05-21
- **Promoted from**: `dev/notes/trackers/tech_debt.md` §"Minimize / eliminate `resources.profiles` in favor of srun / MPI / native affinity"
- **Trigger**: 2026-05-21 smoke run where `srun -n 1 -c 128 --hint=nomultithread axon-recon stages reconstruct.analyzers --task-backend local_affinity ...` was supposed to give one task with one thread per physical core, but actual `n_jobs=16` (= the YAML profile's `task_allocation.cpus_per_task: 16`). User: *"I basically want profiles to go away."* See `open_questions.md` Finding #1 (option 2 selected).

## Why this plan exists

`resources.profiles.<name>` in the YAML duplicates information that the runtime can read directly from the environment. Every YAML profile entry has a counterpart somewhere on the shell side:
- `task_allocation.cpus_per_task` ↔ `srun -c K` / `SLURM_CPUS_PER_TASK`
- `task_allocation.tasks_per_node` ↔ `srun -n N` / `SLURM_NTASKS` / MPI rank count
- `capacity.cpu_cores` ↔ `os.sched_getaffinity(0)` / cgroup CPU limit
- `capacity.ram_gb` ↔ `/sys/fs/cgroup/memory/memory.limit_in_bytes` / `SLURM_MEM_PER_NODE`
- `capacity.gpu_sort_slots` ↔ `len(CUDA_VISIBLE_DEVICES.split(","))` / `SLURM_GPUS_*`
- `--profile <name>` ↔ which sbatch you submitted

When the YAML and the shell disagree, the YAML wins silently. Every sbatch script has to remember to pass `--profile perlmutter_gpu`. Every YAML edit has to be cross-checked against the sbatch flags. Two recent incidents traced to profile/CLI drift (spikesort_full gate-deadlock with `gpu_sort_slots=0`; bootstrap `n_jobs=1` regression with the slot ContextVar not propagating).

Resource **classes** (per-phase RAM / slot DEMANDS like `analyzer_slots: 1`, `ram_gb: 14`) are NOT redundant with the environment and stay. Only the profile-level **supply** block goes away.

The 2026-05-21 paper-over commits (`1982a31` + `322e8fe`) added a SLURM env precedence + topology clamp on top of the profile concept. After this plan ships, both become moot — the resolver IS env-only, no precedence ladder needed.

## Inventory (read these first)

```bash
# 136 source references to the profile concept
grep -rnE "active_profile|--profile|--task-profile|_ACTIVE_PROFILE_OVERRIDE|get_active_profile" src/axon_recon/

# 2 YAMLs with the profiles block
grep -lE "^  profiles:|^  active_profile:" dev/debug_NERSC/*.yml src/axon_recon/*.yml

# 1 sbatch script with --profile (only one!)
grep -rl "\-\-profile" dev/debug_NERSC/jobs/
```

Key call sites already mapped (from the tech-debt entry):
- `src/axon_recon/pipeline/resources.py` — `ResourcesConfig.profiles`, `active_profile`, `parse_resources_config`, `get_active_profile`
- `src/axon_recon/pipeline/config.py` — `parse_resources_config_for_bundle`, `_ACTIVE_PROFILE_OVERRIDE`, `set_active_profile_override`
- `src/axon_recon/pipeline/cli.py` — `--profile` / `--task-profile` flag
- `src/axon_recon/pipeline/cpu_allocation.py` — `_limit_units_by_profile_cpu`, `_derive_cpus_per_task` (the 2026-05-21 paper-over here gets reverted in slice 1)
- `dev/debug_NERSC/debug.runtime.yml` lines 69-156 (perlmutter_cpu + perlmutter_gpu blocks)
- `src/axon_recon/default.runtime.yml` (also has profile blocks)

## Slices

### Slice 0 — Revert the 2026-05-21 paper-overs

**Goal**: clear the workspace. The SLURM env precedence (`1982a31`) and the topology clamp (`322e8fe`) bolted env-awareness ON TOP of the profile resolver. After slice 2 below, the resolver IS env-only by construction — those paper-overs become moot AND would actively conflict with the new resolver.

**Edits**:
- `git revert --no-commit 322e8fe 1982a31` — restores `_derive_cpus_per_task` to its pre-paper-over shape.
- Drop the new tests added in those commits (they'll be replaced by the env-only resolver tests in slice 2).
- Keep the `parallelism_post_migration_cleanup_plan.md` slice 9.5 entry as a historical record but mark it `superseded by resources_profiles_elimination_plan.md`.

**Acceptance**: clean `git diff` against pre-`1982a31` for `cpu_allocation.py` and `test_cpu_allocation.py`. Existing test suite still green.

**Commit**: `claude: revert env-precedence + clamp paper-overs (resources.profiles elim slice 0)`

---

### Slice 1 — Introduce env-only supply resolver (no call sites switched yet)

**Goal**: stand up the function that reads supply from the environment. Don't wire it in yet — slice 2 does the swap.

**New module/function** (location TBD; sketch: `pipeline/env_supply.py`):
```python
@dataclass(frozen=True)
class EnvSupplyBudget:
    cpus_per_task: int           # from SLURM_CPUS_PER_TASK or sched_getaffinity(0) count
    cpus_per_task_source: str    # "slurm_env" | "sched_getaffinity" | "default"
    tasks_per_node: int          # from SLURM_NTASKS_PER_NODE / SLURM_NTASKS / 1
    tasks_per_node_source: str
    ram_gb_per_node: float | None  # from cgroup memory.limit_in_bytes / SLURM_MEM_PER_NODE / None
    gpu_count: int               # len(CUDA_VISIBLE_DEVICES) / 0
    gpu_indices: tuple[int, ...]

def detect_env_supply() -> EnvSupplyBudget:
    ...
```

**Tests** (`pipeline/tests/test_env_supply.py`, new):
- SLURM_CPUS_PER_TASK=64 → `cpus_per_task=64`, source `slurm_env`.
- SLURM_CPUS_PER_TASK unset, monkeypatched `sched_getaffinity` returns {0..127} → `cpus_per_task=128`, source `sched_getaffinity`.
- Both unset → `cpus_per_task=1`, source `default`.
- SLURM_MEM_PER_NODE=480000 → `ram_gb_per_node=480.0`.
- CUDA_VISIBLE_DEVICES="0,2,3" → `gpu_count=3, gpu_indices=(0,2,3)`.
- No GPU env → `gpu_count=0`.

**Acceptance**: new file ships with full test coverage; nothing else changes; pre-existing tests still green.

**Commit**: `claude: introduce env-supply resolver (resources.profiles elim slice 1)`

---

### Slice 2 — Make `build_task_allocation_plan` env-only; remove profile clamping

**See also (sequencing — added 2026-05-21 refinement pass)**: `parallelism_post_migration_cleanup_plan.md` slice 2 ships AFTER this slice. Reason: that slice wires `inputs.n_jobs` call sites through `resolve_inner_worker_count(phase_cpus_per_task=_budget.cpus_per_task)`. Once this slice lands, `_budget.cpus_per_task` is env-derived — so the parallelism slice 2 wiring then works correctly out of the box. Order reverse would mean the parallelism slice ships first using profile-clamped budgets (the user-flagged problem) and then this slice changes the budget source under it.

**Goal**: the resolver becomes the SOLE source of supply truth. Profile-keyed capacity clipping (`_limit_units_by_profile_cpu`) gets deleted.

**Edits**:
- `pipeline/cpu_allocation.py::_derive_cpus_per_task`: rewrite to call `detect_env_supply().cpus_per_task` directly. YAML fallback removed (the YAML's `task_allocation.cpus_per_task` field gets the deletion treatment in slice 3).
- `pipeline/cpu_allocation.py::_limit_units_by_profile_cpu`: DELETE. The topology is the supply ceiling, not a profile-keyed `cpu_cores` value.
- `pipeline/cpu_allocation.py::build_task_allocation_plan`: drop the `resource_profile` parameter and all consumers. The call site (`runner.py::_attach_task_allocation_plan`) stops passing it.
- Tests: rewrite `test_cpu_allocation.py` to monkeypatch `sched_getaffinity` instead of constructing fake `ResourceProfileConfig`s. Drop tests that assert on profile-keyed `cpu_cores` clipping.

**Acceptance**: with monkeypatched `SLURM_CPUS_PER_TASK=128` + 128-core topology → plan has 1 slot of 128 cores, source `slurm_env`. With unset env + 128-core affinity → same, source `sched_getaffinity`. With unset env + 64-core affinity (the 2026-05-21 smoke case) → 1 slot of 64 cores. NO "produced no available task slots" error — the resolver just uses what's actually there.

**Commit**: `claude: env-only build_task_allocation_plan; remove profile clamping (resources.profiles elim slice 2)`

---

### Slice 3 — Delete the YAML `profiles` block + `active_profile` + `--profile`/`--task-profile` + `_ACTIVE_PROFILE_OVERRIDE`

**Goal**: physically remove the profile concept from code + YAML + CLI. After this slice, `grep -rn "active_profile\|--profile" src/` returns 0 hits.

**A. Code deletions**:
- `pipeline/resources.py`: drop `ResourcesConfig.active_profile`, `ResourcesConfig.profiles`, `Profile` dataclass, `get_active_profile`, profile parsing in `parse_resources_config`. Resource classes (`PhaseResourceClassConfig` keyed by phase) STAY — they're per-phase demands, not profile-level supply.
- `pipeline/config.py`: drop `_ACTIVE_PROFILE_OVERRIDE`, `set_active_profile_override`, `get_active_profile_override`. Drop `active_profile_override` from the bundle.
- `pipeline/cli.py`: drop `--profile` and `--task-profile` argparse args + the override-setting/clearing in `_run_stage_sequence_from_args`.
- `pipeline/runner.py`: `_attach_task_allocation_plan` no longer accepts `resource_profile`. `available_shm_gb` / capacity-driven logic switches to reading from the env-supply budget if it's still needed.
- `pipeline/resource_budget.py`: any `Profile`-keyed budget construction switches to env-supply or per-phase resource_class declarations.

**B. YAML deletions**:
- `dev/debug_NERSC/debug.runtime.yml`: delete `resources.profiles:` block entirely (lines 14-156 approximately). Delete `resources.active_profile:` if present. Resource classes (per-phase declarations, line ~258 area) stay.
- `src/axon_recon/default.runtime.yml`: same surgery.

**C. Sbatch surgery**:
- The 1 sbatch script with `--profile` (per inventory grep) loses the flag.

**Tests**:
- Drop tests that monkeypatch `set_active_profile_override` or pass `active_profile_override=`.
- Drop tests asserting profile-keyed capacity behavior.
- `pipeline/tests/test_resources.py` (if present) gets a profile-block-rejection test (parse_resources_config should raise on `profiles:` key, with a clear "removed — supply is read from environment" error message — fail loudly during a migration window, then we can delete the rejection in a later cleanup).

**Acceptance**:
```bash
grep -rnE "active_profile|--profile|--task-profile|_ACTIVE_PROFILE_OVERRIDE|get_active_profile|ResourcesConfig\.profiles|class Profile\b" src/axon_recon/
# expected: 0 hits (or only the migration-rejection-test reference)

grep -rn "^  profiles:" dev/debug_NERSC/*.yml src/axon_recon/*.yml
# expected: 0 hits
```

**Commit**: `claude: delete resources.profiles + active_profile + --profile CLI + _ACTIVE_PROFILE_OVERRIDE (resources.profiles elim slice 3)`

---

### Slice 4 — Sbatch + docs re-audit

**Goal**: sweep up the trailing surface — examples, sbatch templates, and any documentation that referenced the profile concept.

**Edits**:
- `dev/debug_NERSC/examples/` (if exists): drop `--profile` from any example commands.
- `dev/debug_NERSC/jobs/sans_bombcell_rerun/sbatches/*.sbatch` (per tech-debt entry, "every sbatch currently has to redundantly pass `--profile perlmutter_gpu`"): the actual count from inventory was 1, but a final sweep confirms.
- `dev/notes/brain/guardrails/parallelism_agent_guardrails.md` (if present): codify "supply comes from the environment, not from YAML profiles" as a contract.
- `dev/notes/trackers/tech_debt.md`: mark the "Minimize / eliminate `resources.profiles`" entry RESOLVED with a pointer to this plan.
- Update CLAUDE.md if it referenced `--profile` or profiles.

**Acceptance**: full repo grep `--profile` returns no recommended-usage hits; only migration-historical notes (commit log entries, archived plans) remain.

**Commit**: `claude: sbatch + docs sweep, mark profile-elim tech-debt resolved (resources.profiles elim slice 4)`

---

### Slice 5 — Real-data smoke verification

**Goal**: validate the whole stack with the 2026-05-21 use case that triggered this plan.

**Smoke** (from `salloc_smokes_queued.md`'s kssynth slice 3b heavy entry, updated with the post-plan command shape — likely identical, just no `--profile` to worry about):

```bash
HDF5_PLUGIN_PATH=/global/homes/a/adammwea/hdf5_plugin_path_maxwell \
srun -n 1 -c 128 --cpu-bind=cores --hint=nomultithread \
shifter --image=adammwea/axon-recon:pipeline-v2 \
  env PYTHONPATH=/global/u2/a/adammwea/dev/pkgs/axon_recon/src \
  python -m axon_recon.pipeline.cli stages reconstruct.analyzers \
  --config dev/debug_NERSC/debug.runtime.yml \
  --target-dataset 13 --limit-wells 1 --task-backend local_affinity \
  --force-restart \
  --input-root /pscratch/sd/a/adammwea/analyzed_data/Media_Density_T5_02182026_AR_axon_analysis_AW/ \
  --output-root /pscratch/sd/a/adammwea/dev_outputs/kssynth_slice3b/
```

**Expected log lines** (verify each):
- `cpus_per_task_source=slurm_env` OR `cpus_per_task_source=sched_getaffinity` (whichever the env supplied).
- `Applied task CPU affinity task_slot=0 cpus=...` showing ALL cores the rank actually has (no profile-keyed narrowing).
- `compute_waveforms (workers: N processes fork)` where N == the rank's physical-core count.
- Per-segment runtime drops proportionally vs the 16-worker run.

If srun is exposing only 64 physical cores instead of 128, that's a SEPARATE issue (an srun flag investigation, not an axon_recon code question). Surface as a follow-up rather than re-introducing profile clamping.

**Smoke log entry**: append to `dev/notes/trackers/smoke_log.md` with full bug→fix chain.

**Commit**: `claude: kssynth slice 3b heavy smoke validates env-only supply (resources.profiles elim slice 5)`

---

## Out of scope

- Re-architecting `PhaseResourceClassConfig` — per-phase demands stay as-is. Only profile-level supply changes.
- MPI rank discovery via `mpi4py` — the existing `mpi_adapter._context_from_mpi_env` already handles this. Untouched.
- GPU partitioning logic for `spikesort.sort` — the `CUDA_VISIBLE_DEVICES` partitioning still happens via `mpi_adapter`. Just stop reading `profiles.<x>.capacity.gpu_sort_slots` for the slot count.
- The "srun gives only 64 physical cores instead of 128" question. That's a slurm/Perlmutter flag question, not an axon_recon question. Document the right `srun -c` / `--cpu-bind` / `--threads-per-core` invocation in the smoke entry after the plan ships.

## Definition of Done

- Single-source-of-truth: every supply-side decision flows from `detect_env_supply()`. No YAML can override what slurm / cgroup / sched_getaffinity says.
- `grep -rn "active_profile\|--profile\|_ACTIVE_PROFILE_OVERRIDE\|ResourcesConfig\.profiles" src/` returns 0 hits.
- `dev/debug_NERSC/debug.runtime.yml` and `src/axon_recon/default.runtime.yml` have no `profiles:` block.
- 2026-05-21 smoke verifies the resolver respects whatever srun actually exposed.
- `tech_debt.md` entry marked RESOLVED with a commit-SHA pointer to slice 3.
- `parallelism_post_migration_cleanup_plan.md` slice 9.5 marked SUPERSEDED (the env-precedence concept lives here now).
