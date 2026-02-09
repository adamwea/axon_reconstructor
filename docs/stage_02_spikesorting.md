# Stage 02 — Spikesorting (MEA_Analysis integration)

This stage is intentionally a thin integration layer around **MEA_Analysis**.

`axon_reconstructor` does not (currently) run Kilosort directly in-process. Instead, spikesorting is executed via MEA_Analysis (often on GPU nodes via Shifter), and axon_reconstructor focuses on:

- producing/validating the expected per-well folder structure
- locating the sorter outputs for downstream SpikeInterface consumers (waveforms/templates)

## Primary APIs

This stage exposes two layers:

- Command building + path contract helpers (pure integration): `axon_reconstructor.integrations.mea_analysis`
  - `build_run_pipeline_driver_cmd(...)`
  - `compute_spikesorting_output_dir(...)`
  - `compute_sorter_output_dir(...)`
  - `validate_sorter_output_dir(...)`

- In-process debug harness (imports MEA_Analysis and injects the saved recording):
  - `tools/stepwise_debug_scripts/spikesorting_debug.py` (`run_spikesorting_only(...)`)

## Inputs

- `data_file`: Maxwell `.raw.h5`
- `mea_output_root`: MEA_Analysis output root
- `well` / `stream_id`: well identifier (e.g. `well000`)
- `sorter`: sorter name (default `kilosort4`)

## Execution model

### NERSC / Perlmutter (typical)

- Allocate a GPU node (interactive or batch)
- Run MEA_Analysis’ `IPNAnalysis/run_pipeline_driver.py` inside the Shifter image

User-facing helpers:

- `axon-reconstructor mea-sort-cmd ...`
  - prints the `run_pipeline_driver.py` command-line for your environment
- `axon-reconstructor gpu-interact ...`
  - requests an interactive GPU allocation and runs spikesorting inside Shifter

(See `axon_reconstructor/cli.py` for flag details; the docs here stay high-level on purpose.)

### Lab server / local (optional)

In non-NERSC setups, you can still use `mea-sort-cmd` to build a driver invocation. The exact container/runtime varies by lab environment.

## Outputs on disk

Outputs are produced by MEA_Analysis under the per-well output folder (computed from `mea_output_root`, `data_file`, and `stream_id`). The key directory later stages consume is typically:

- `<well_out_dir>/sorter_output/`
  - sorter-specific outputs (spike times, templates, logs, binaries)

`axon_reconstructor` primarily treats this folder as a contract: it resolves/validates it, then waveforms/templates use it to construct `Sorting` objects.

## Detailed stepwise docs

- Step 01: [methods/methods_spikesorting_step_01_path_contract_and_output_layout.md](methods/methods_spikesorting_step_01_path_contract_and_output_layout.md)
- Step 02: [methods/methods_spikesorting_step_02_building_driver_commands.md](methods/methods_spikesorting_step_02_building_driver_commands.md)
- Step 03: [methods/methods_spikesorting_step_03_in_process_debug_harness.md](methods/methods_spikesorting_step_03_in_process_debug_harness.md)
- Step 04: [methods/methods_spikesorting_step_04_outputs_validation_and_handoff.md](methods/methods_spikesorting_step_04_outputs_validation_and_handoff.md)
- Step 05: [methods/methods_spikesorting_step_05_troubleshooting_and_hpc_notes.md](methods/methods_spikesorting_step_05_troubleshooting_and_hpc_notes.md)

## Troubleshooting

- If later stages can’t find sorting outputs, first confirm the per-well folder matches the MEA_Analysis path contract and that `sorter_output/` exists.
- When switching sorters or changing MEA_Analysis driver options, re-run spikesorting (or ensure you’re pointing to the correct output root).
