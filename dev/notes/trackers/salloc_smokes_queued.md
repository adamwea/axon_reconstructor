# Salloc smokes queued

Smoke runs that need an **interactive Slurm allocation** (the loop's authorization caps at login-node smokes with `--task-backend local_affinity`). User runs each entry manually from an `salloc` shell; loop picks up from the produced outputs in the next iteration.

**Conventions**:
- Each entry is one self-contained smoke: title + context + prereqs + the exact ready-to-run command block + expected outputs + "next loop iteration picks up from..." pointer.
- When the smoke completes, EITHER delete the entry (loop has consumed the result) OR move it to `dev/notes/trackers/smoke_log.md` with the real-data smoke schema filled in.
- One-time env setup steps (e.g. plugin installs) belong in `brain/guardrails/env_parity.md`, not here. Each entry can reference them by name.
- Loop appends entries here as needed (replaces the older pattern of stuffing salloc commands into `current_state.md`'s 📝 User actions queued section).

**Default `salloc` shell pattern** (Perlmutter cpu queue):

```
salloc -A m4408 -N 1 -t 60 --qos=interactive -C cpu
```

Adjust `-t` for longer runs; `-C gpu` for GPU-bound smokes (e.g. `spikesort.sort`).

**Inside `salloc`, wrap each pipeline invocation in `srun`** so the allocation actually parallelizes across the node's cores instead of running serially in the login-like shell. For a 1-well / 1-target smoke (most entries here), the right shape is one rank with the full node's CPU budget:

```
srun -n 1 -c 128 --cpu-bind=cores --hint=nomultithread <command...>
```

(Perlmutter cpu node = 128 physical cores; `-c 128` + `--cpu-bind=cores --hint=nomultithread` pins one task to all physical cores without hyperthread oversubscription. The pipeline's `--task-backend local_affinity` picks the budget up from `slot.cpu_count` automatically.)

For multi-target smokes that benefit from MPI fan-out across targets, use `srun -n N -c $((128/N)) --task-backend mpi` instead.

---

## Queued

### ~~2026-05-21 — kssynth slice 3b HEAVY (analyzers + kssynth) on M08073/well000 DIV 36~~ **(SUPERSEDED 2026-05-21 by GATE 1 entry below — shifter image now contains the --input-root fix; no PYTHONPATH overlay needed.)**

---

## GATE 1 step 1+1b — kssynth on unit_0598 only (M08073/well000 DIV 36)

- **Why salloc not login**: login-node attempts (3 of them, latest at 2026-05-21 19:51) consistently OOM-killed (exit 137) within ~3 min, even scoped to `--unit-ids 598`. The per-segment SpikeInterface analyzer build (100% random_spikes_percentage) blows past login-node cgroup limits. Verified shifter env + PATH 2 fix work (smoke #5: source_count 0→2 with --limit-segments 2). Salloc gives full-node RAM.
- **Prereqs**: shifter image `adammwea/axon-recon:pipeline-v2` rebuilt 2026-05-21 with commit d824d4a (PATH 2 LOAD-path fix). Reference data at `/pscratch/.../analyzed_data/Media_Density_T5_02182026_AR_axon_analysis_AW/`.
- **Allocation**: `salloc -A m4408 -N 1 -t 60 --qos=interactive -C cpu`
- **Inside salloc, run** (one command — kssynth's internal analyzer-build covers step 1; `--unit-ids 598` scopes to the single high-branch unit):

```bash
srun -n 1 -c 128 --cpu-bind=cores --hint=nomultithread \
  shifter --image=adammwea/axon-recon:pipeline-v2 -- \
  axon-recon stages reconstruct.kssynth \
    --config dev/debug_NERSC/debug.runtime.yml \
    --target-dataset 13 --limit-wells 1 --task-backend local_affinity \
    --unit-ids 598 \
    --force-enable kssynth \
    --input-root /pscratch/sd/a/adammwea/analyzed_data/Media_Density_T5_02182026_AR_axon_analysis_AW/ \
    --output-root /pscratch/sd/a/adammwea/dev_outputs/radivojevic_apples_to_apples/
```

- **Expected outputs**:
  - `/pscratch/sd/a/adammwea/dev_outputs/radivojevic_apples_to_apples/Media_Density_T5_02182026_AR/260326/M08073/AxonTracking/000208/well000/recon_outputs/synth_sorter_output/kssynth_summary.json` — `status: ok`, `per_unit_n_units_written: 1`.
  - `.../synth_sorter_output/per_unit/unit_0598/merged_template.npy` + `merged_channel_locations.npy` — the merged template radivojevic will consume.
- **ETA**: ~5-15 min (5 segments × analyzer build + tiny kssynth synth).
- **Next loop iteration picks up from**: loading `unit_0598/merged_template.npy` + running `radivojevic2023_recon_algo.reconstruct(...)` on it → adapter design question (plot_recons is hardcoded to consume `gtr.pkl`; loop will surface adapter-strategy multi-choice once kssynth output exists). See GATE 1 step 4 in dev/notes/brain/open_questions.md.
