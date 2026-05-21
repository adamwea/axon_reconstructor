# Salloc smokes queued

Smoke runs that need an **interactive Slurm allocation** (the loop's authorization caps at login-node smokes with `--task-backend local_affinity`). User runs each entry manually from an `salloc` shell; loop picks up from the produced outputs in the next iteration.

**Conventions**:
- Each entry is one self-contained smoke: title + context + prereqs + the exact ready-to-run command block + expected outputs + "next loop iteration picks up from..." pointer.
- When the smoke completes, EITHER delete the entry (loop has consumed the result) OR move it to `dev/notes/trackers/smoke_log.md` with the real-data smoke schema filled in.
- One-time env setup steps (e.g. plugin installs) belong in `guardrails/env_parity.md`, not here. Each entry can reference them by name.
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

### 2026-05-21 — kssynth slice 3b HEAVY (analyzers + kssynth) on M08073/well000 DIV 36

**Context**: Apples-to-apples radivojevic-vs-axon_velocity_gtrs diagnostic (per PRE-DIAGNOSTIC GATE 1) gates on this. Loop's tried twice on the login node; both runs got SIGTERM'd at ~3-5 min into the HDF5 chunk read (likely login-node walltime policy). Heavy smoke needs an interactive allocation.

**Plan + slice**: `kssynth_recon_integration_plan` slice 3b HEAVY → unlocks `radivojevic_recon_algo_plan` slice 3 sub-step 9's apples-to-apples diagnostic.

**Prereqs**:
- MaxWell HDF5 plugin already installed at `~/hdf5_plugin_path_maxwell/libcompression.so` (loop ran the `auto_install_maxwell_hdf5_compression_plugin()` step this session; documented as one-time setup in `guardrails/env_parity.md`).
- `--input-root` analyzers plumbing fix shipped (commits `f33821b` + `c8b8b11`). PYTHONPATH-overlay needed because the May-19 shifter image's `axon-recon` is pinned pre-`--input-root`.

**Command** (run inside `salloc`):

```bash
cd /global/u2/a/adammwea/dev/pkgs/axon_recon

# Analyzers (5-15 min). Builds preprocessed_segments analyzer cache from
# the reference path's input data, writes to dev_outputs/.
# --force-restart clears the partial state from the prior login-node
# SIGTERM kills (per user 2026-05-21).
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

# kssynth synthesize (~30s after analyzers complete). No --force-restart
# here because the analyzers step above produces a fresh cache and
# kssynth has nothing to clean.
HDF5_PLUGIN_PATH=/global/homes/a/adammwea/hdf5_plugin_path_maxwell \
srun -n 1 -c 128 --cpu-bind=cores --hint=nomultithread \
shifter --image=adammwea/axon-recon:pipeline-v2 \
  env PYTHONPATH=/global/u2/a/adammwea/dev/pkgs/axon_recon/src \
  python -m axon_recon.pipeline.cli stages reconstruct.kssynth \
  --config dev/debug_NERSC/debug.runtime.yml \
  --target-dataset 13 --limit-wells 1 --task-backend local_affinity \
  --force-enable kssynth \
  --input-root /pscratch/sd/a/adammwea/analyzed_data/Media_Density_T5_02182026_AR_axon_analysis_AW/ \
  --output-root /pscratch/sd/a/adammwea/dev_outputs/kssynth_slice3b/
```

**Expected outputs**:
- `/pscratch/sd/a/adammwea/dev_outputs/kssynth_slice3b/Media_Density_T5_02182026_AR/260326/M08073/AxonTracking/000208/well000/recon_outputs/cache/analyzers/segments/{000_rec0000,001_rec0001,...}/` — per-segment SpikeInterface analyzer dirs.
- `/pscratch/sd/a/adammwea/dev_outputs/kssynth_slice3b/Media_Density_T5_02182026_AR/260326/M08073/AxonTracking/000208/well000/recon_outputs/synth_sorter_output/per_unit/unit_<N>/{merged_template.npy, merged_channel_locations.npy, ...}` — per-unit merged templates ready for radivojevic consumption.

**Next loop iteration picks up from**: scan the `synth_sorter_output/per_unit/` dir for the 9-branch unit (`unit_0598` per reference data's `branches.json`; cross-check the kssynth unit-ID numbering before substituting), run `radivojevic2023_recon_algo.reconstruct(...)` on its `merged_template.npy`, then file PRE-DIAGNOSTIC GATE 1 as a multi-choice question per the Round 5 STOP-AND-ASK discipline.
