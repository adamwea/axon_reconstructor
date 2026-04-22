# axon_reconstructor

`axon_reconstructor` is actively under development for multi-stage axon reconstruction and longitudinal HD-MEA analysis workflows.

This project is not currently user-friendly and should be treated as work-in-progress. Documentation is currently messy and overly verbose.

For now, the most relevant references are:

- Development roadmap: [DEVELOPMENT_ROADMAP.md](DEVELOPMENT_ROADMAP.md)
- Environment spec: [environment.yml](environment.yml)
- Debug harness docs: [docs/debugging/README.md](docs/debugging/README.md)

## Environment setup

Create the base conda env first:

```bash
conda env create -f environment.yml
conda activate axon_recon
```

Then bootstrap the sibling editable dependencies that live in adjacent Git checkouts:

```bash
bash tools/bootstrap_editable_deps.sh --python "${CONDA_PREFIX}/bin/python"
```

The bootstrap script clones missing repos into the parent directory of this checkout using HTTPS GitHub remotes and installs them editable with `--no-deps` from the branches currently used in `pkgs/`:

- `axon_velocity` from `adamwea/axon_velocity` on `main`
- `UnitMatch` from `adamwea/UnitMatch` on `enable_hdmea`
- `SLAy` from `adamwea/SLAy` on `main`
- `MEA_Analysis` from `roybens/MEA_Analysis` on `aw_dev`

Use `bash tools/bootstrap_editable_deps.sh --dry-run` to inspect the planned git and pip commands before making changes.

AI coding agents have been used during development, mostly GPT-5.2-Codex and GPT-5.3-Codex.
