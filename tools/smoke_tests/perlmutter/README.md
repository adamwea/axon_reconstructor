# NERSC Perlmutter smoke tests

These scripts are meant for **interactive node validation** on Perlmutter using your example dataset.

## Layout

- `_shared/`
  - `00_config.sh`: edit paths and defaults (raw data, output root, repo roots, Shifter image, wrapper allocation knobs).
  - `_nersc_shifter_helpers.sh`: small helper functions for module/shifter availability.
- `login_node/`: safe tests runnable on a login node (no GPU, no spikesorting).
- `interactive_gpu_node/`: tests runnable inside a GPU allocation (primarily inside Shifter).
- `interactive_cpu_node/`: CPU-node postprocessing (expects sorter outputs already exist under `OUT_ROOT`).
- `axon_reconstructor/`: runs axon reconstruction by **loading** MEA outputs from `OUT_ROOT`.

## Quick start

1) Create a local config (recommended):

- Copy `tools/smoke_tests/perlmutter/smoke_tests.local.toml.example` to `tools/smoke_tests/perlmutter/smoke_tests.local.toml`.
- Fill in at least:
  - `paths.raw_h5`
  - `paths.out_root`
  - `gpu_salloc.account`

Environment variables override both TOML files.

2) Run all login-node smoke tests:

- `bash tools/smoke_tests/perlmutter/login_node/run_all_login_node_smoke_tests.sh`

3) Run all GPU-node smoke tests from a login node (single allocation, runs suite inside it):

- `bash tools/smoke_tests/perlmutter/interactive_gpu_node/run_all_gpu_node_smoke_tests_from_login.sh`

4) When ready for real spikesorting, allocate a GPU and run:

- `bash tools/smoke_tests/perlmutter/interactive_gpu_node/10_gpu_spikesort_interactive.sh`

## Common GPU failure mode: torch missing

If the GPU checks show `ModuleNotFoundError: torch` (or you see host Python like `3.6.x`), run:

- `bash tools/smoke_tests/perlmutter/interactive_gpu_node/11_shifter_image_inventory.sh`

and confirm the intended image is being used and imports succeed.
