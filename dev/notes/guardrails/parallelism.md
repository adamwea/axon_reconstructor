# Parallelism guardrail

## Contract

Inner-worker count (`n_jobs` for SpikeInterface, OMP / numba thread counts, autoencoder DataLoader batch counts, etc.) is **always** computed via `resolve_inner_worker_count(...)` from `src/axon_recon/pipeline/cpu_allocation.py`. Direct reads of `inputs.n_jobs` or hardcoded values are forbidden. The resolver's fallback when the `_CURRENT_TASK_SLOT` ContextVar isn't bound is: `base = min(slot.cpu_count if slot else ∅, yaml_n_jobs_override, phase_cpus_per_task, work_item_count)` — picking the smallest positive hint, NOT silently defaulting to `1`.

## Why

The MPI task backend spawns worker processes via `srun → shifter → python`. Python ContextVars don't cross process boundaries. So `_CURRENT_TASK_SLOT` is `None` in every worker. The naïve "if slot is None, set base = 1" path used to silently make every MPI worker single-threaded, even when the YAML asked for parallelism — the bootstrap_concat_binary regression (`engine=process - n_jobs=1`) and the spikesort.sort `SpikeInterface global job kwargs: {'n_jobs': 1, …}` log line both traced to this bug. The fix (commit `c205d14` and subsequent test rewrite) makes the resolver honor the explicit yaml/phase hints when slot is missing.

Per the YAML active profile (`perlmutter_gpu` has `cpus_per_task: 16`, `perlmutter_cpu` has `cpus_per_task: 16` too), under MPI workers we should see `n_jobs = 16` per rank, not `n_jobs = 1`.

## Concrete sub-rules

1. **n_jobs source for any new phase**: read from the phase's resource_class budget via `current_phase_budget(stage, phase).cpus_per_task`, pass into `resolve_inner_worker_count` as `phase_cpus_per_task`. Optionally accept a yaml-side `n_jobs` override at the phase config; if present, pass as `yaml_n_jobs_override`.

2. **MPI rank scheduling**: a single srun produces N MPI ranks; each rank's Python process gets its own `n_jobs` budget via the resolver. The MPI world's RANK COUNT is set by `srun -n N`, not by anything in the YAML — the YAML's `cpus_per_task` only controls per-rank threading.

3. **No `n_jobs=1` defaults in YAML**. If `n_jobs: null` in the YAML, the resolver derives from the profile's `cpus_per_task`. Setting `n_jobs: 1` in YAML for a non-debug reason is a bug.

4. **OMP/numba/MKL thread env**: matched to the resolved `n_jobs` via `thread_env_applied event=thread_env_applied policy=match_cpus_per_task` (see logs). Don't manually set `OMP_NUM_THREADS` in CLI or scripts; let the resolver do it.

5. **GPU phase budget**: phases that consume GPU (`spikesort.sort`, `spikesort.merge_SLAy`) have `gpu_sort_slots` or `analyzer_slots` declarations in `resources.phase_budgets`. Resource-gate must show `gpu_sort_slots: 1` available before the phase starts; if it shows `0`, the active profile is wrong (this was the ds4 timeout root cause — `active_profile: perlmutter_cpu` set, but the sbatch was a GPU node).

## Tests / verification

- `tests/test_cpu_allocation.py` covers the `resolve_inner_worker_count` contract including the slot-is-None fallback.
- Smoke-test trigger: any change to `cpu_allocation.py`, `runner.py`'s phase worker allocation, or `resources.py` requires a real-data smoke test (one well, login node) that grep-confirms `n_jobs=<expected>` in the log and NOT `n_jobs=1`.
- Visual log markers to watch for:
  - `Spikesort phase worker allocation stage=spikesort phase=<P> target=<T> well_workers=1 n_jobs=<N>` — N should match `cpus_per_task` of the active profile.
  - `write_binary_recording engine=process - n_jobs=<N>` — same N as above.
  - `SpikeInterface global job kwargs: {'n_jobs': <N>, …}` — same N.
  - If any of these read `n_jobs=1`, parallelism is broken; do NOT proceed with a heavier smoke test until it's restored.

## Open exceptions / follow-ups

- `resources.profiles` itself is on the kill list (see `trackers/tech_debt.md` §"Minimize / eliminate `resources.profiles`"). Once it's gone, the resolver's `phase_cpus_per_task` hint comes from srun/cgroup-derived state instead of YAML. The contract above is written to survive that transition: it just reads from a different source.

- **`task_allocation.{ram_gb_per_task, shm_gb_per_task}` are LIVE, just unset by default.** Both fields are `None` in `default.runtime.yml:61-62` and the active debug YAMLs, but they ARE consumed in `cpu_allocation.py:670, 674` via `_capacity_limit_from_float(budget=…, demand=…)`. When set to a positive float they cap `effective_task_limit` per node alongside `cpu_capacity_tasks` (`cpu_allocation.py:682-685`):
  - `ram_gb_per_task` is per-task RAM demand; budget is `resource_profile.ram_gb`.
  - `shm_gb_per_task` is per-task `/dev/shm` demand; budget is the node's available shm (queried at allocation time, with a warning at `cpu_allocation.py:905` if unavailable).
  Do NOT delete these fields as "dead" — they're the only knob for RAM-bound or SHM-bound phases (multi-binary writers, large preprocess buffers) to be throttled below `cpu_capacity_tasks`. If YAML sets them positively, the limit applies; if not, only CPU capacity bounds tasks-per-node. Cross-ref: `pipeline/resources.py:106-107` (field defs), `pipeline/resources.py:397-405` (parser), `tests/test_cpu_allocation.py:252-253, 651` (consumer tests pin the cap path + warning).
