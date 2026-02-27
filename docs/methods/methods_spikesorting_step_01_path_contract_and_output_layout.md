# Spikesorting Step — Part 1: Path Contract and Output Layout (MEA_Analysis Integration)

Scope: this document covers how axon_reconstructor and MEA_Analysis agree on where spikesorting outputs live on disk.

Primary code paths:
- CLI command builder: `axon_reconstructor.integrations.mea_analysis.build_run_pipeline_driver_cmd(...)`
- Path helpers (installed MEA_Analysis required):
  - `axon_reconstructor.integrations.mea_analysis.compute_mea_output_dir(...)`
  - `axon_reconstructor.integrations.mea_analysis.compute_spikesorting_output_dir(...)`
  - `axon_reconstructor.integrations.mea_analysis.compute_sorter_output_dir(...)`
- In-process stage runner (imports MEA_Analysis): `axon_reconstructor.pipeline.spikesorting.run_spikesorting_stage(...)`

---

## 0. Stage intent

In this pipeline, spikesorting is treated as an **integration boundary**:

- MEA_Analysis runs sorting (Kilosort4 by default), analyzer computation, and report generation.
- axon_reconstructor focuses on:
  - producing the input recording (Stage 01)
  - making it easy to run MEA_Analysis consistently
  - locating and validating the resulting `sorter_output/` for downstream stages

---

## 1. Per-well folder structure

All stages share the “per-well output directory” concept:

- `<output_root>/<relative_pattern>/<well>/`

Where:

- `output_root` is the user-specified analysis output root.
- `relative_pattern` is computed from the raw data file path using MEA_Analysis’ path contract.
- `well` is the stream id (e.g. `well000`).

Within that per-well folder, spikesorting outputs live under:

- `<well_out_dir>/spikesorting_outputs/`

Within `spikesorting_outputs/`, the key contract directory is:

- `<well_out_dir>/spikesorting_outputs/sorter_output/`

This is the folder later stages load (via SpikeInterface reader helpers).

---

## 2. Two path-contract implementations

You will see two implementations in the codebase:

1. **Dependency-free reimplementation** (used by the pipeline driver):
   - `axon_reconstructor.pipeline.pipeline_driver._compute_mea_analysis_output_dir(...)`
   - This is designed to be robust even if MEA_Analysis isn’t importable.

2. **Source-of-truth MEA_Analysis contract** (used by integration helpers):
   - `axon_reconstructor.integrations.mea_analysis.compute_mea_relative_pattern(...)`
   - This requires MEA_Analysis installed such that `IPNAnalysis.path_contract` is importable.

In practice they should match; the split exists so axon_reconstructor remains usable without MEA_Analysis installed.

---

## 3. Minimal validation of sorter output

For lightweight sanity checks without importing SpikeInterface, axon_reconstructor provides:

- `axon_reconstructor.integrations.mea_analysis.validate_sorter_output_dir(sorter_output_dir)`

This checks that:

- the folder exists, and
- it contains at least one common expected file (or is non-empty)

Downstream stages do more concrete validation by actually loading the sorter output.
