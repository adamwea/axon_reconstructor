# axon_reconstructor

Rebuilding `axon_reconstructor` with cleaner structure, documentation, and reproducible tooling.

## Developer setup (start here)

- Create the conda env: [environment.yml](environment.yml)
- Follow the walkthrough: [docs/developer_setup.md](docs/developer_setup.md)

## Methods

- Raw preprocessing + spikesorting: [docs/methods/methods_preprocess_spikesort.md](docs/methods/methods_preprocess_spikesort.md)

## Install (editable)

```bash
pip install -e ".[dev]"
```

## Quick check

```bash
python -c "import axon_reconstructor; print(axon_reconstructor.__version__)"
axon-reconstructor
pytest
```
