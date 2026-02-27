# Example Configs Reference

This folder contains editable templates for debug and cross-well analysis.

## Files

- [debug.env.example](debug.env.example): default runtime variables used by `tools/debug/*.py` scripts.
- [cross_well_config.example.yml](cross_well_config.example.yml): multi-dataset cross-well analysis spec.

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
