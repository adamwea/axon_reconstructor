# axon_reconstructor

`axon_reconstructor` is the primary package for multi-stage axon reconstruction and longitudinal HD-MEA analysis workflows.

## Current priority: roadmap-driven development

- Primary planning + execution tracker: [DEVELOPMENT_ROADMAP.md](DEVELOPMENT_ROADMAP.md)
- Developer setup guide: [docs/developer_setup.md](docs/developer_setup.md)
- Environment spec: [environment.yml](environment.yml)

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

## Near-term documentation targets

- Add package-owned debug harness docs and examples (env defaults + cross-well config template).
- Add modular analysis project guidance for real dataset runs.
