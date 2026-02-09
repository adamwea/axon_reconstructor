# Spikesorting Step — Part 2: Building MEA_Analysis Driver Commands (CLI)

Scope: this document covers how axon_reconstructor constructs reproducible MEA_Analysis invocations for spikesorting.

Primary code paths:
- CLI: `axon_reconstructor.cli` subcommands:
  - `axon-reconstructor mea-sort-cmd ...`
  - `axon-reconstructor gpu-interact ...`
- Command builder: `axon_reconstructor.integrations.mea_analysis.build_run_pipeline_driver_cmd(spec)`

---

## 0. The MEA_Analysis entrypoint we target

The integration targets the MEA_Analysis driver:

- `MEA_Analysis/IPNAnalysis/run_pipeline_driver.py`

axon_reconstructor does not shell out to this by default in NERSC mode; instead it prints or runs commands to do so.

---

## 1. `axon-reconstructor mea-sort-cmd`

This subcommand prints a runnable command to execute MEA_Analysis for sorting.

Key behavior:

- Requires `--mea-analysis-repo-root` so it can point at the driver script.
- Requires `--mea-output-root` (passed as `--output-dir` to MEA_Analysis).
- Passes `--sorter` through.

Environment presets:

- `--mea-environment=nersc`
  - assumes you will run inside Shifter or a NERSC-friendly environment
  - may default `--scratch-dir` to `$SLURM_TMPDIR` if present

- `--mea-environment=lab`
  - expects a Docker-based workflow
  - requires `--docker-image`

HPC carveouts supported by the driver command builder:

- `--cuda-visible-devices`
- `--require-gpu`
- `--n-jobs`
- `--chunk-duration`
- `--scratch-dir` + `--stage-back` + `--stage-back-mode`

---

## 2. `axon-reconstructor gpu-interact`

This subcommand is a user-facing convenience for NERSC:

- requests an interactive GPU allocation (via `salloc` + `srun`)
- runs MEA_Analysis inside Shifter
- injects `$SLURM_TMPDIR` as MEA_Analysis scratch when available

It intentionally avoids smoke-test debug knobs. It is meant to be the “typical user” path.

---

## 3. What the builder actually returns

`build_run_pipeline_driver_cmd(spec)` returns an `argv` list resembling:

- `python3 <mea_repo_root>/IPNAnalysis/run_pipeline_driver.py <data_path> --output-dir <output_root> --sorter <sorter> ...`

The caller decides whether to print this, run it directly, or embed it in a container execution wrapper.
