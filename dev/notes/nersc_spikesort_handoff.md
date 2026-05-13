# NERSC spikesort handoff — GPU node, 2026-05-12

Picks up after `dev/notes/nersc_smoke_handoff.md`. Preprocess is green on
Perlmutter CPU. You are now inside a Perlmutter GPU alloc (1 node, 4× A100,
64 cores, 256 GB) running the same shifter image `adammwea/axon-recon:pipeline-v2`.

## What changed vs the preprocess handoff

- **Active profile**: must be `perlmutter_gpu` (not `perlmutter_cpu`). The
  profile defines `cpu_cores: 64`, `ram_gb: 224`, `gpu_sort_slots: 4`,
  `cpus_per_task: 16` — so 4 ranks/node is the natural mapping (1 rank per A100).
- **Phase sequence** (from `dev/debug_NERSC/debug.runtime.yml` `stages.spikesort.phase_sequence`):
  1. `bootstrap_concat_binary` (CPU)
  2. `sort` (**GPU** — Kilosort4)
  3. `snapshot_sorter_output` (CPU)
  4. `concat_analyzer` (CPU)
  5. `bombcell_label` (CPU)
  6. `merge_SLAy` (CPU)
  7. `cleanup_concat_binary` (CPU)
  8. `cleanup_analyzers` (CPU)

- **GPU oversubscription rule** (hard-coded in `run_spikesort_stage`):
  `SpikesortGpuOversubscriptionError` raises **before any CUDA allocation**
  if MPI/Slurm ranks > visible GPU count. So for `sort` itself, `-n <= 4`
  on a 4-GPU node, full stop. No MPS, no half-rank GPU sharing.

## Profile switch — one-line edit before any run

```bash
sed -i 's/^  active_profile:.*/  active_profile: perlmutter_gpu/' dev/debug_NERSC/debug.runtime.yml
grep "active_profile:" dev/debug_NERSC/debug.runtime.yml
```

Flip back to `perlmutter_cpu` before the next reconstruct alloc (CPU node).

## Required env for cuda-aware MPI inside the image

The image bakes OpenMPI 4.x (not Cray MPICH), so set:

```bash
export MPICH_GPU_SUPPORT_ENABLED=1
```

`--module=gpu` is required to expose the NVIDIA driver to shifter; **don't**
add `--module=cuda-mpich` (that's for host Cray MPICH integration, which is
not our path since we use the container's bundled OpenMPI).

## srun shape

```bash
# Sanity check the image sees CUDA inside shifter
srun --module=gpu \
  shifter --image=adammwea/axon-recon:pipeline-v2 \
  python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"
```

Expect `CUDA: True count: 4`. If `False`, the alloc didn't reserve GPUs or
`--module=gpu` is missing.

```bash
# Allocation preview (dry-run) for the whole spikesort stage
shifter --image=adammwea/axon-recon:pipeline-v2 \
  axon-recon stages spikesort \
    --config dev/debug_NERSC/debug.runtime.yml \
    --target-dataset 0 --limit-wells 1 \
    --alloc
```

```bash
# Smoke: 1 dataset, 1 well, multi-rank via srun, 1 rank per GPU
srun --cpu-bind=cores --module=gpu -N 1 -n 4 -c 16 \
  shifter --image=adammwea/axon-recon:pipeline-v2 \
  axon-recon stages spikesort \
    --config dev/debug_NERSC/debug.runtime.yml \
    --target-dataset 0 --limit-wells 1 \
    --task-backend mpi --force-restart
```

Notes:
- `-n 4 -c 16` → 4 ranks × 16 cpus = 64 cores (full GPU node).
- The `sort` phase honors the oversubscription gate at `-n <= 4`.
- All other phases also run at `-n 4`; that's fine since they're CPU-bound.

## Smoke escalation (spikesort)

Mirror the preprocess escalation but with `spikesort` as the stage. Run
each step; debug+iterate before escalating.

1. **CUDA sanity** — `torch.cuda.is_available()` returns True, 4 devices.
2. **Image smoke** — `axon-recon-smoke-cli` inside the GPU alloc.
3. **Allocation preview** — `--alloc` with 1 dataset, 1 well.
4. **Single-phase isolation, sort only** — run `spikesort.sort` alone first
   to verify GPU correctness before chaining:
   ```bash
   srun --cpu-bind=cores --module=gpu -N 1 -n 4 -c 16 \
     shifter --image=adammwea/axon-recon:pipeline-v2 \
     axon-recon stages spikesort.sort \
       --config dev/debug_NERSC/debug.runtime.yml \
       --target-dataset 0 --limit-wells 1 \
       --task-backend mpi --force-restart
   ```
5. **Phase chain, 1 well** — full `spikesort` stage, 1 dataset, 1 well.
6. **All 6 wells, 1 dataset**.
7. **3 datasets, all wells** (`--limit-datasets 3`).
8. **Full scope** — drop all `--limit-*`. Hand this command back to the user.

## Debug surfaces

- `<output_root>/<dataset>/<well>/spikesort_outputs/sorter_output/` — Kilosort raw output
- `<output_root>/<dataset>/<well>/spikesort_outputs/logs/spikesort_pipeline.log` — text log
- `<output_root>/<dataset>/<well>/spikesort_outputs/concat_analyzer/` — SortingAnalyzer
- `<output_root>/<dataset>/<well>/spikesort_outputs/bombcell_label_outputs/bombcell_label_summary.json`
- `<output_root>/<dataset>/<well>/spikesort_outputs/merge_SLAy/run-output.json` + `recommended_merge_groups.json`
- `<output_root>/<dataset>/<well>/spikesort_outputs/sorter_output_snapshot/` — pre-merge backup
- Structured pipeline log: `<output_root>/<dataset>/<well>/logs/pipeline.jsonl`

## GPU-specific failure modes (watch for these)

- **`SpikesortGpuOversubscriptionError`**: you used `-n 5` or more on a 4-GPU
  node. Drop to `-n 4`.
- **`CUDA out of memory`** during Kilosort: Kilosort4 default settings target
  ~16 GB VRAM headroom. If a well has very dense spike rates this can OOM.
  Look at the `kilosort` block under `phases.sort.sorter.kilosort` in the
  runtime yml; `batch_duration_s` can be reduced if needed.
- **CUDA visible to one rank but not others**: `mpi_adapter` partitions
  `CUDA_VISIBLE_DEVICES` at startup. If you see "no CUDA device" in a rank,
  check that mpi_adapter ran (it's imported by the pipeline entry point).
- **`spikesort.sort` hangs after Kilosort prints "GPU memory"**: usually one
  rank crashed silently. Look at `spikesort_pipeline.log` per rank for the
  loser. SIGTERM the srun and re-run with `--limit-wells 1` to isolate.
- **`bombcell_label` / `merge_SLAy` dry_run leaves stale dry_run/ dirs**: the
  config has `dry_run: false` in NERSC by default. If the snapshot is missing
  for a re-run, set `force_restart` to true for the phase.
- **Concat binary cache pollution between iterations**: `bootstrap_concat_binary`
  honors `--force-restart` to drop the cache. Use it freely during smokes.

## Definition of done (spikesort)

- Steps 1–8 of the escalation pass without error.
- Step 8 produces spikesort outputs for every included well.
- Each well shows: `sorter_output/` populated, `concat_analyzer/` written,
  `bombcell_label_summary.json` with non-zero unit counts, `merge_SLAy/run-output.json`
  present. `cleanup_*` phases successfully deleted intermediate caches.
- Final shareable command for the user is the step-8 invocation.

## After spikesort: reconstruct (back to CPU node)

End this GPU alloc. Have the user `salloc -N 1 -t 2:00:00 -C cpu -q interactive -A m2043 --image=adammwea/axon-recon:pipeline-v2`.

Inside, flip the profile back:

```bash
sed -i 's/^  active_profile:.*/  active_profile: perlmutter_cpu/' dev/debug_NERSC/debug.runtime.yml
```

Reconstruct stage phase sequence is in `stages.reconstruct.phase_sequence`
(in the runtime yml — `templates_analyzers`, `extract_partial_templates`,
`build_templates`, `report_templates`, `generate_gtrs`, plot phases, report phases).
Use the same escalation pattern. There's no GPU work in reconstruct, but it's
plot-heavy — watch for `plot_unit_summary` runtime per well (slowest phase).

## Things NOT to do (spikesort-specific)

- Don't run `spikesort.sort` at `-n > 4` on a 4-GPU node. The pipeline will
  refuse, but it costs you a debug round-trip.
- Don't set `engine: mea_analysis` for `spikesort.sort` inside shifter —
  it would try to launch a nested Docker container which doesn't exist
  inside the image. Keep `engine: local_spikeinterface` (the default).
- Don't `rm -rf` the `sorter_output_snapshot/` dirs between sort and merge.
  bombcell_label_pass2 (if you later enable it) reads from this snapshot.
- Don't forget to flip `active_profile` back to `perlmutter_cpu` when you
  move to the reconstruct alloc — perlmutter_gpu's 64-core capacity will
  underclaim a 128-core CPU node.
