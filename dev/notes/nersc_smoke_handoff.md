# NERSC smoke handoff — first axon-recon runs on Perlmutter

You (the in-alloc Claude) are inside a NERSC Perlmutter compute node allocation.
Shifter image `adammwea/axon-recon:pipeline-v2` is already pulled and staged.
Your job: drive smoke tests for one stage at a time, debug until clean, and
hand back a "run the full scope" command once a stage is green.

## State of the world (set by the login-node session)

- **Repo**: `/global/u2/a/adammwea/dev/pkgs/axon_recon` on branch `pipeline_v2`.
  Three commits ahead of `origin/pipeline_v2` (rename + perlmutter_login profile
  + pscratch data paths) — not pushed yet; the user is reviewing.
- **Active configs**: `dev/debug_NERSC/debug.runtime.yml` + `dev/debug_NERSC/debug.data.yml`.
  - `active_profile: perlmutter_cpu` (switch to `perlmutter_gpu` for `spikesort.sort`).
  - `use_scratch_root: false` — raw data IS already on pscratch, no copy step.
  - `output_root: /pscratch/sd/a/adammwea/analyzed_data/Media_Density_T5_02182026_AR_axon_analysis_AW`
  - `scratch_root: /pscratch/sd/a/adammwea/scratch` (only used if `use_scratch_root: true` again later)
  - 13 raw data .h5 files referenced, all confirmed present at
    `/pscratch/sd/a/adammwea/raw_data/Media_Density_T5_02182026_AR/...`
- **Profiles defined** in the runtime yml: `lab_server_safe`, `perlmutter_cpu`,
  `perlmutter_gpu`, `perlmutter_login`.
- **NERSC account**: `m2043`. CFS dir `/global/cfs/cdirs/m2043/roybens/ben-shalom_nas`
  is the long-term destination (not needed for these smoke runs — outputs stay on pscratch).

## Shifter automounts

NERSC Shifter automounts these paths inside the container with the host's UID:
`/pscratch`, `/global/common`, `/global/cfs`, `/global/u1`, `/global/u2`,
`/global/homes`. **No explicit `--volume=` needed** for the configs as written.

## Invocation pattern

Always cd to the repo root first so relative config paths resolve. The image's
ENTRYPOINT is `axon-recon-entrypoint`; default CMD is `--help`. Pass `axon-recon`
as the first arg to override.

### Sanity check the image (no srun needed)

```bash
cd /global/u2/a/adammwea/dev/pkgs/axon_recon
shifter --image=adammwea/axon-recon:pipeline-v2 axon-recon --help
shifter --image=adammwea/axon-recon:pipeline-v2 axon-recon-smoke-cli
```

### Allocation preview (no work — just prints task layout)

```bash
shifter --image=adammwea/axon-recon:pipeline-v2 \
  axon-recon stages preprocess \
    --config dev/debug_NERSC/debug.runtime.yml \
    --target-dataset 0 --limit-wells 1 --limit-segments 2 \
    --alloc
```

### Multi-rank smoke (preprocess, 1 well, 2 segments, 1 dataset)

```bash
srun --cpu-bind=cores --threads-per-core=1 -N 1 -n 8 -c 16 \
  shifter --image=adammwea/axon-recon:pipeline-v2 \
  axon-recon stages preprocess \
    --config dev/debug_NERSC/debug.runtime.yml \
    --target-dataset 0 --limit-wells 1 --limit-segments 2 \
    --task-backend mpi --force-restart
```

Notes:
- `-N 1 -n 8 -c 16 --threads-per-core=1` → 1 node, 8 ranks/node, 16 PHYSICAL
  cores/rank → 128 of 128 physical cores used (full CPU node, 4 ranks per
  socket, 2 CCX per rank). The `--threads-per-core=1` flag matches
  `perlmutter_cpu.use_hyperthreads: false` so the profile's cpus_per_task=16
  and Slurm's -c 16 mean the same thing (physical cores). See
  `examples/README.md` "Truth table" for the full matrix.
- `--task-backend mpi` is the pipeline-side backend; `mpi_adapter` detects
  `SLURM_PROCID`/`SLURM_NTASKS` and partitions work. `--task-backend slurm` is
  equivalent and matches what the sbatch examples document.
- `--force-restart` clears prior outputs for the targeted scope so re-runs are deterministic.

## Smoke escalation sequence (preprocess)

Run each step. If it fails, debug, fix, re-run before escalating. Capture stdout
to a per-step log under `/pscratch/sd/a/adammwea/smoke_logs/preprocess/`.

1. **Image smoke** — `axon-recon --help` and `axon-recon-smoke-cli` inside shifter.
2. **Allocation dry-run** — `--alloc` with `--target-dataset 0 --limit-wells 1`.
   Confirms profile + budget plan looks right; no I/O.
3. **Tiny smoke** — 1 dataset, 1 well, 2 segments. Validates the full stage path
   (copy_src_to_scratch skip → save_rec_metadata → preprocess_segments) end-to-end.
4. **1 well, full segments** — drop `--limit-segments`. Catches per-segment I/O issues.
5. **All 6 wells, 1 dataset** — drop `--limit-wells`. Catches well-fanout + keyed
   resource limit behavior (only one rank reads each source_h5 at a time).
6. **3 datasets, all wells** — `--limit-datasets 3`. Catches inter-dataset state.
7. **Full scope** — drop all `--limit-*`. This is the answer to hand back.

## Debug surfaces

- Per-phase logs (text): `<output_root>/<rel_path>/preprocess_outputs/logs/preprocess_pipeline.log`
- Structured pipeline log: `<output_root>/<rel_path>/logs/pipeline.jsonl`
- Run manifest + event timeline: `<output_root>/<rel_path>/preprocess_outputs/run_metadata/`
- Per-phase summary JSON: `<output_root>/<rel_path>/preprocess_outputs/<phase>/context/<phase>_summary.json`
- Slurm job stderr/stdout: wherever `srun` writes (usually the controlling terminal in an interactive alloc).

## Expected failure modes (watch for these)

- **HDF5 plugin missing**: entrypoint fails early with a specific message
  about `libcompression.so` not in `HDF5_PLUGIN_PATH`. Image baked it at
  `/usr/local/lib/plugin/`; should be fine.
- **Rank desync / hang on a phase boundary**: usually indicates one rank
  errored without surfacing. Look at the structured log for an unhandled
  exception; check `preprocess_pipeline.log` per rank.
- **OOM**: `perlmutter_cpu` profile budgets `ram_gb_per_task` via phase_budgets.
  If you see SIGKILL/9, lower `ntasks-per-node` (drop from 6 → 4) or lower
  `cpus_per_task`. Don't touch the profile capacity; tune the srun shape.
- **CUDA visible but unused**: harmless for preprocess (no GPU phases active).
  `mpi_adapter` partitions `CUDA_VISIBLE_DEVICES` even when phases don't use GPU;
  this is by design.
- **File ownership weirdness**: Shifter runs as host UID by default at NERSC.
  Output files should be owned by `adammwea` and readable from the login node
  after the alloc ends.

## Definition of done (preprocess)

- All 7 escalation steps pass without error.
- Step 7 produces preprocess outputs for **every** included well in
  `dev/debug_NERSC/debug.data.yml`.
- Final shareable command for the user is the step-7 invocation (a single
  `srun ... shifter ... axon-recon stages preprocess ...` line — no `--limit-*` flags).

## After preprocess: next stages

### Spikesort.sort (requires GPU node)

End this CPU allocation. Have the user `salloc -C gpu -G 4 -N 1 -t 2:00:00 -q interactive -A m2043 --image=adammwea/axon-recon:pipeline-v2`.
Flip `active_profile: perlmutter_gpu` in the runtime yml (or override per
invocation if the pipeline supports `--active-profile`). Run smoke at
`--target-dataset 0 --limit-wells 1` first. `spikesort.bootstrap_concat_binary`
and `spikesort.summarize_sort` can stay on the GPU node or move back to a
CPU alloc; `sort` itself is the only strictly-GPU phase.

### Reconstruct (back to CPU node)

End the GPU alloc. Re-salloc on `-C cpu`. Flip back to `active_profile: perlmutter_cpu`.
Reconstruct phases mirror what `examples/smoketest_sort_and_recon.sh` runs on
the lab server — same escalation strategy: 1 well → 1 dataset → full.

## Things NOT to do

- Don't push to `origin/pipeline_v2` from inside the alloc unless the user
  asks. The login session has unpushed commits the user wants to review.
- Don't edit `dev/debug_local/debug.runtime.yml` — that's the lab-server config.
- Don't switch `use_scratch_root: true` unless raw data moves off pscratch.
- Don't run on login nodes after the alloc ends. Use the alloc's compute time
  fully or `scancel` cleanly.
- Don't fudge the `Definition of done` criteria. If step 7 partially succeeded,
  say so; don't hand back a "full scope" command that wasn't actually validated.
