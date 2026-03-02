# Developer setup

This repo is being rebuilt with a modern `src/` layout, tests, and reproducible environments.

## Step 1: create the conda environment

From the repo root:

```bash
conda env create -f environment.yml
conda activate axon_recon
```

If you created the environment manually (e.g. `conda create -n axon_recon ...`), you still need to install this repo into that environment:

```bash
conda activate axon_recon
python -m pip install -e ".[dev]"
```

If you already created the env and want to update it:

```bash
conda env update -f environment.yml --prune
```

## Step 2: editable install

The goal is to have the project installed in editable mode so `import axon_reconstructor` works from anywhere:

```bash
python -c "import axon_reconstructor; print(axon_reconstructor.__version__)"
```

## Step 3: run tests

```bash
python -m pytest -q
```

## Notes (HPC / NERSC)

- Prefer running tools via `python -m ...` to ensure you’re using the active environment.
- Heavy scientific/runtime dependencies will be added incrementally as we rebuild each pipeline step.

## Pipeline helper ownership (Phase 3.5)

- Stage-independent JSON/value conversion helpers live in `src/axon_reconstructor/pipeline/shared_io.py`.
- Stage-specific checkpoint filename/transition helpers live in `src/axon_reconstructor/pipeline/stage_checkpointing.py`.
- Shared stage logger and stage lifecycle logging helpers live in `src/axon_reconstructor/pipeline/pipeline_logging.py`.
- Stage runners should call shared helpers instead of re-implementing `_read_json`/`_write_json`, stage checkpoint naming, or stage start/complete/failure log formatting.
