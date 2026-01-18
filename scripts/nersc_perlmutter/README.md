# NERSC Perlmutter quick test scripts

These scripts are meant for **interactive node testing** with your example dataset.

## Files

- `00_config.sh`: edit paths (raw data, pscratch output root, repo roots).
- `05_login_smoketest_no_sort.sh`: safe login-node smoke test (runs `--skip-spikesorting`; verifies imports + path contract; does not require GPU). Also writes full per-well routine stdout/stderr logs under `${OUT_ROOT}/subprocess_logs/login_smoketest_no_sort`.
- `06_login_smoketest_force_restart.sh`: login-node smoke test with `--force-restart` (verifies clean re-run behavior).
- `07_gpu_node_smoketest_no_sort_container_plugin_default.sh`: GPU-node smoke test *inside Shifter* that does **not** set `HDF5_PLUGIN_PATH` (expects container plugin defaults) and still runs `--skip-spikesorting`.
- `08_gpu_require_gpu_dry_check.sh`: fast GPU gate check *inside Shifter* using driver `--require-gpu --dry` (fails fast if CUDA isn’t visible).
- `09_gpu_readiness_check.sh`: GPU-node readiness check inside Shifter (torch import + CUDA visibility); writes a log under `${OUT_ROOT}/gpu_checks`.
- `10_gpu_spikesort_interactive.sh`: run Mandar’s driver on an interactive **GPU** node. Uses `--scratch-dir $SLURM_TMPDIR` when available and stages back `sorter_output` into `OUT_ROOT`.
- `11_shifter_image_inventory.sh`: Shifter image inventory (python version, torch/h5py presence, CUDA visibility); writes a log under `${OUT_ROOT}/gpu_checks`.
- `20_cpu_postprocess_skip_sort.sh`: run Mandar’s driver on a **CPU** node with `--skip-spikesorting`.
- `30_run_axon_reconstructor.sh`: run axon reconstruction, **loading** sorter outputs from `OUT_ROOT`.

## Typical usage

1) Edit `00_config.sh` once.

0) Optional login-node smoke test (recommended while debugging deps/paths):

- `./scripts/nersc_perlmutter/05_login_smoketest_no_sort.sh`

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

## Common GPU failure mode: torch missing

If `09_gpu_readiness_check.sh` reports `ModuleNotFoundError: torch` (or shows `Python 3.6.x`), run `./scripts/nersc_perlmutter/11_shifter_image_inventory.sh` and confirm that `/opt/conda` exists inside the container and that `torch` imports.

If it doesn’t, the Shifter step is likely not using the intended image snapshot (or the image wasn’t pulled/updated). On Perlmutter, try setting `SHIFTER_IMAGE` to the fully-qualified form `docker:<repo>:<tag>` and/or pulling the image explicitly with `shifterimg pull ...`, then re-run the inventory.
