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

Three documented install paths, in order of recommendation:

### 1. Shifter container (recommended for production / cluster runs)

The shipped shifter image bakes the full pipeline + every pip dep + the
Kilosort CUDA stack. Pull / load per the cluster's shifter docs; no
local pip installs required.

### 2. `./tools/setup_env.sh` (recommended for new contributors)

One-command developer install. Creates the conda env, installs
axon_recon with the `[dev,full]` extras (production-grade dep set
including UnitMatchPy from its public upstream), and (optionally)
re-installs siblings editable from local clones for active hacking.

```bash
# Fresh user: clone + setup
git clone <axon_recon-repo-url>
cd axon_recon
./tools/setup_env.sh                          # standard dev install
./tools/setup_env.sh --editable-siblings      # + editable siblings
                                              #   from $HOME/dev/pkgs/
./tools/setup_env.sh --editable-siblings \
  --from-local /custom/path/to/sibling/clones  # override sibling search dir
```

The `--editable-siblings` flag uses `tools/install_dev_siblings.sh` to
re-install UnitMatchPy + SLAy in editable mode (overriding the
git-URL versions that `[full]` installs by default). Missing siblings
get cloned on demand into a gitignored `deps/` directory.

### 3. `conda env create -f environment.yml` (manual, for advanced users)

Same outcome as path 2 but skips `setup_env.sh`'s convenience wrapping:

```bash
conda env create -f environment.yml
conda activate axon_recon
```

`environment.yml`'s `pip: - -e .[dev,full]` line pulls everything from
`pyproject.toml`'s `[dev]` + `[full]` extras — the single source of
truth for pip deps. The conda layer keeps the scientific stack (numpy,
h5py, scipy, spikeinterface, …) on conda-forge for binary-friendly
builds.

### Sibling registry

The pipeline depends on these external siblings:

- **`axon_velocity`** (PyPI: `axon_velocity==0.1.2`) — installed by `[full]`.
- **`UnitMatchPy`** — sibling of the parent `EnnyvanBeest/UnitMatch`
  repo, installed by `[full]` as a git URL pin
  (`git+https://github.com/EnnyvanBeest/UnitMatch.git#subdirectory=UnitMatchPy`).
- **`SLAy`** (`saikoukunt/SLAy`) — Dockerfile-uninstalled today; will
  be added to `[full]` once its install story is finalized. Editable
  install path: `./tools/install_dev_siblings.sh --siblings SLAy`.
- **`kssynth`**, **`unitlink`** — local-only sibling packages used by
  the (in-progress) cross-DIV unitmatch integration. GH remotes pending;
  use local editable installs from `~/dev/pkgs/{kssynth,unitlink}/`.
- **`MEA_Analysis`** (`roybens/MEA_Analysis` on `aw_dev`) — supporting
  utilities, manual editable install when needed.

## Running tests

```bash
python -m pytest                      # full default test scope
python -m pytest tools/tests/         # tooling smoke tests (out of default scope)
```

AI coding agents have been used during development, mostly GPT-5.2-Codex and GPT-5.3-Codex.
