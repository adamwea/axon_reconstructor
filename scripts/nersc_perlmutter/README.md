# NERSC Perlmutter quick test scripts

These scripts are meant for **interactive node testing** with your example dataset.

## Files

- `00_config.sh`: edit paths (raw data, pscratch output root, repo roots).
- `10_gpu_spikesort_interactive.sh`: run Mandar’s driver on an interactive **GPU** node. Uses `--scratch-dir $SLURM_TMPDIR` when available and stages back `sorter_output` into `OUT_ROOT`.
- `20_cpu_postprocess_skip_sort.sh`: run Mandar’s driver on a **CPU** node with `--skip-spikesorting`.
- `30_run_axon_reconstructor.sh`: run axon reconstruction, **loading** sorter outputs from `OUT_ROOT`.

## Typical usage

1) Edit `00_config.sh` once.

2) GPU interactive (spikesorting):

- Allocate:
  - `salloc -A <acct> -C gpu -q interactive -t 02:00:00 -N 1 --gpus=1 --cpus-per-task=32`
- Run:
  - `./scripts/nersc_perlmutter/10_gpu_spikesort_interactive.sh`

3) CPU interactive (rest of MEA pipeline):

- Allocate:
  - `salloc -A <acct> -C cpu -q interactive -t 02:00:00 -N 1 --cpus-per-task=32`
- Run:
  - `./scripts/nersc_perlmutter/20_cpu_postprocess_skip_sort.sh`

4) Run axon reconstruction:

- `./scripts/nersc_perlmutter/30_run_axon_reconstructor.sh`

If `30_run_axon_reconstructor.sh` fails with “sorter_output not found”, it means step (2) didn’t stage results back under `OUT_ROOT` (or the output-root/path contract doesn’t match where the data file lives).
