# axon_reconstructor

`axon_reconstructor` is actively under development for multi-stage axon reconstruction and longitudinal HD-MEA analysis workflows.

The active runtime package is `axon_recon`. The older `axon_reconstructor` Python package has been retired during the v2 pipeline cleanup.

The active stages are:

- `preprocess`
- `spikesort`
- `reconstruct`

The installed command is still named `axon-reconstructor`, but it now dispatches to `axon_recon.pipeline.cli`.

Useful development references:

- Environment spec: [environment.yml](environment.yml)
- Runtime config: [debug/debug.runtime.yml](debug/debug.runtime.yml)
- Data config: [debug/debug.data.yml](debug/debug.data.yml)
- Refinement notes: [debug/pipeline_refinement_commit_notes.md](debug/pipeline_refinement_commit_notes.md)

## Environment setup

Create the base conda env first:

```bash
conda env create -f environment.yml
conda activate axon_recon
```

Sibling editable dependencies usually live next to this checkout and are installed separately:

- `axon_velocity` from `adamwea/axon_velocity` on `main`
- `UnitMatch` from `adamwea/UnitMatch` on `enable_hdmea`
- `SLAy` from `adamwea/SLAy` on `main`
- `MEA_Analysis` from `roybens/MEA_Analysis` on `aw_dev`

Run tests with:

```bash
python -m pytest
```

AI coding agents have been used during development, mostly GPT-5.2-Codex and GPT-5.3-Codex.
