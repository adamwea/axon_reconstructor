# axon_recon

`axon_recon` is actively under development for multi-stage axon reconstruction and longitudinal HD-MEA analysis workflows.

The active stages are:

- `preprocess`
- `spikesort`
- `reconstruct`

The installed CLI is `axon-recon` and dispatches to `axon_recon.pipeline.cli`.

Useful development references:

- Environment spec: [environment.yml](environment.yml)
- Default runtime config (hermetic template): [src/axon_recon/default.runtime.yml](src/axon_recon/default.runtime.yml)
- Example data config (schema only): [examples/example.data.yml](examples/example.data.yml)
- Lab-server runtime config: [dev/debug_local/debug.runtime.yml](dev/debug_local/debug.runtime.yml)
- Lab-server data config: [dev/debug_local/debug.data.yml](dev/debug_local/debug.data.yml)
- NERSC runtime config: [dev/debug_NERSC/debug.runtime.yml](dev/debug_NERSC/debug.runtime.yml)
- NERSC data config: [dev/debug_NERSC/debug.data.yml](dev/debug_NERSC/debug.data.yml)
- Example run/launch wrappers: [examples/](examples/)
- Dev notes (plans, guardrails, trackers, commit log): [dev/notes/](dev/notes/)
- Refinement notes: [dev/notes/archive/pipeline_refinement_commit_notes.md](dev/notes/archive/pipeline_refinement_commit_notes.md)

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
