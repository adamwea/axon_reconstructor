# Example Configs Reference

This folder contains editable templates for stage-first CLI and cross-well/scope orchestration.

## Files

- [debug.env.example](debug.env.example): default runtime variables used by `tools/debug/*.py` scripts.
- [project.env.example](project.env.example): project-local env template for canonical `stage` / `scope-run` commands.
- [cross_well_config.example.yml](cross_well_config.example.yml): multi-dataset cross-well analysis spec.
- [scope_config.example.json](scope_config.example.json): canonical multi-dataset pipeline orchestration spec for `scope-run`.

## Recommended config layering

- Keep project defaults in a project-local env file (for example `./.env` or `./project.env`).
- Invoke canonical CLI commands with `--env-file <path>`.
- Override any env default at runtime with explicit CLI flags.

Example:

```bash
python -m axon_reconstructor.cli stage waveforms \
  --env-file docs/examples/project.env.example \
  --n-jobs 16
```

Generate a scope config directly from retained project-template assets:

```bash
python -m axon_reconstructor.cli scope-config-build \
  --cross-well-config tools/debug/cross_well_config.yml \
  --env-file tools/debug/debug.env \
  --out tools/debug/logs/generated_scope_config.json \
  --stage-order preprocess,spikesort,unit_match,merge_update,waveforms,templates,reconstruct,analysis
```

Then execute with `scope-run` (or use `tools/debug/run_scope_combo.sh` for an
editable, script-first flow).

## `debug.env.example` key groups

- **Paths**: `AXON_RECON_H5_PATH`, `AXON_RECON_STREAM_ID`, `AXON_RECON_MEA_OUTPUT_ROOT`
- **Cross-well controls**: `AXON_RECON_CROSS_WELL_*`
- **Preprocess/Spikesort/Waveforms/Templates/Reconstruction knobs**
- **AV overrides**: `AXON_RECON_AV_*` (optional fine tuning)
- **Analysis toggles**: `AXON_RECON_ANALYSIS_*`, `AXON_RECON_ANALYSIS_BOTM_*`

## `cross_well_config.example.yml` fields

- `analysis_name`: label used in output naming
- `runtime_env_file`: path to env file loaded by analysis script
- `out_dir`: output folder for CSV/plots/decks
- `electrode_pitch_um`: probe pitch used in geometry metrics
- `datasets`: list of recordings to aggregate
  - `raw_data_h5_path`: source recording file
  - `DIV`: days in vitro for grouping/ordering
  - `wells`: wells and their condition metadata
    - `well_id`: stream id / well identifier
    - `condition`: grouping label (e.g., density)
    - `plating_density_nbp`: numeric density for ordered grouping
    - `genotype`: genotype label

## `scope_config.example.json` fields

- `mea_output_root`: output root consumed by pipeline stages.
- `datasets[]`: dataset/well targets for execution.
  - `h5_path`: source recording path.
  - `wells[].stream_id`: well/stream id.
- `stage_order`: global stage barrier order (supports transition gates like `unit_match` and `merge_update`).
- `per_well_parallelism`: number of wells processed concurrently within a stage barrier.
- `stage_kwargs`: optional stage-specific keyword overrides.

Migration note:
- Use `scope_config.example.json` + `axon-reconstructor scope-run` for new runs.
- Prefer project-local scripts that call canonical package commands directly.
