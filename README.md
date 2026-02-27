# axon_reconstructor

`axon_reconstructor` is the primary package for multi-stage axon reconstruction and longitudinal HD-MEA analysis workflows.

## Current priority: roadmap-driven development

- Primary planning + execution tracker: [DEVELOPMENT_ROADMAP.md](DEVELOPMENT_ROADMAP.md)
- Developer setup guide: [docs/developer_setup.md](docs/developer_setup.md)
- Environment spec: [environment.yml](environment.yml)
- Debug harness docs: [docs/debugging/README.md](docs/debugging/README.md)
- Example configs: [docs/examples/README.md](docs/examples/README.md)

All major work should follow the roadmap sequence and include a docs/roadmap checkpoint after each major item.

## Install (editable)

```bash
pip install -e ".[dev]"
```

## Quick sanity check

```bash
python -c "import axon_reconstructor; print(axon_reconstructor.__version__)"
axon-reconstructor
pytest
```

## Methods docs

- Preprocess + spikesorting: [docs/methods/methods_preprocess_spikesort.md](docs/methods/methods_preprocess_spikesort.md)

## Current implementation notes

- Package-owned debug harness scripts now live under `tools/debug`.
- A minimal project wrapper for Media Density analysis lives at `/home/adamm/dev/projects/260227_media_density_analysis_project` and delegates execution to package-owned scripts.
