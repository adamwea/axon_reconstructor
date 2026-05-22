# env_install_unification slice 1 — install-paths audit

Inventory of every pip / conda / sibling dep referenced across the
three install artifacts as of 2026-05-19. Slice 1 of
`env_install_unification_plan.md`.

Slices 2-3 will collapse this list onto `pyproject.toml` extras
(`[dev]` + `[full]`) and shrink `environment.yml` to conda-only. Slice
6 collapses the Dockerfile onto `pip install -e .[full]`. This audit
locks the current shape so the migration commits don't silently drop a
dep or downgrade a pin.

## Artifact 1: `pyproject.toml`

Source: `pyproject.toml` (root).

| Section | Deps |
|---|---|
| `[project].dependencies` | `rich>=15`, `tqdm>=4`, `tomli>=2; python_version < '3.11'` |
| `[project.optional-dependencies].dev` | `pytest>=7`, `ruff>=0.4` |

No `[full]` extra exists yet. Production deps (numpy, h5py, scipy,
spikeinterface, …) currently live in `environment.yml` + Dockerfile,
not here. Slice 2 introduces the `[full]` extra.

## Artifact 2: `environment.yml`

Source: `environment.yml` (root).

| Channel / section | Deps |
|---|---|
| conda-forge / defaults | `python=3.11`, `pip`, `ipykernel`, `nodejs` |
| conda — dev/test tooling | `pytest`, `ruff` |
| conda — scientific stack | `numpy`, `h5py`, `tqdm`, `requests`, `psutil`, `spikeinterface=0.104.3`, `matplotlib`, `pandas`, `pyarrow`, `plotly`, `dash`, `dash-ag-grid`, `statsmodels`, `kaleido`, `openpyxl`, `scipy`, `pyyaml` |
| conda — commented out | `torch` (commented; explicitly excluded from base spec) |
| pip section | `docker`, `pypdf`, `-e .[dev]` |

Comment in file: "Keep sibling editable repos out of the base env spec.
Install them after env creation with tools/bootstrap_editable_deps.sh."
**The referenced bootstrap script does not exist** (tracker entry under
"Add a tools/bootstrap_editable_deps.sh helper" — superseded by this
plan).

## Artifact 3: `containers/axon-recon/Dockerfile`

Source: `containers/axon-recon/Dockerfile`.

Base image: `docker.io/spikeinterface/kilosort4-base:4.0.38_cuda-12.0.0`.

| ARG | Default | Role |
|---|---|---|
| `AXON_RECON_INSTALL_EXTRAS` | `""` | `[extras]` passed to the final `pip install .` of axon_recon itself |
| `AXON_RECON_OBSERVABILITY_APT_PACKAGES` | `"time sysstat procps"` | apt packages |
| `AXON_RECON_MPI_APT_PACKAGES` | `"openmpi-bin libopenmpi-dev"` | apt packages |
| `AXON_RECON_RUNTIME_SPEC` | (see below) | pip specs for the full scientific stack |
| `AXON_VELOCITY_SPEC` | `"axon_velocity==0.1.2"` | pip pin for axon_velocity |
| `SPIKEINTERFACE_SPEC` | `"spikeinterface==0.104.3"` | pip pin for spikeinterface |
| `UNITMATCH_SPEC` | `""` (empty — sibling not auto-installed) | git-url-or-path for UnitMatchPy |
| `UNITMATCH_RUNTIME_SPEC` | `"joblib mat73 mtscomp"` | pip deps installed only if UNITMATCH_SPEC is set |
| `UNITMATCH_INSTALL_ARGS` | `"--no-deps"` | flags passed to the final UNITMATCH_SPEC install |
| `SLAY_SPEC` | `""` (empty) | git-url-or-path for SLAy |
| `SLAY_RUNTIME_SPEC` | `"marshmallow"` | pip deps installed only if SLAY_SPEC is set |
| `SLAY_INSTALL_ARGS` | `"--no-deps"` | flags passed to the final SLAY_SPEC install |
| `MPI4PY_SPEC` | `"mpi4py"` | pip pin for mpi4py |
| `MAXWELL_HDF5_PLUGIN_URL` | (mxwbio cloud URL) | h5 plugin binary |
| `MAXWELL_HDF5_PLUGIN_DIR` | `"/usr/local/lib/plugin"` | h5 plugin install dir |

Expanded `AXON_RECON_RUNTIME_SPEC` (single string):
```
numpy<2.0 h5py<4 tqdm<5 requests<3 psutil matplotlib<4 pandas<3.0
pyarrow plotly>=5.18 dash>=2.14 dash-ag-grid statsmodels kaleido
openpyxl scipy<2.0 pyyaml docker pypdf scikit-learn<2.0 nvidia-ml-py
```

Versions sometimes differ from `environment.yml` — see "drift table"
below.

## Drift table — same dep, different pin

These are the spots where the Dockerfile and `environment.yml` disagree
on a version, OR where a dep is in one and not the other. Slice 2 must
choose authoritative pins for each.

| Dep | environment.yml | Dockerfile | Authoritative source (slice-2 choice) |
|---|---|---|---|
| `numpy` | unpinned | `numpy<2.0` | TBD — defer to slice 2; the upper bound is real for downstream deps |
| `h5py` | unpinned | `h5py<4` | TBD |
| `tqdm` | unpinned (`tqdm>=4` in pyproject) | `tqdm<5` | TBD |
| `requests` | unpinned | `requests<3` | TBD |
| `matplotlib` | unpinned | `matplotlib<4` | TBD |
| `pandas` | unpinned | `pandas<3.0` | TBD |
| `scipy` | unpinned | `scipy<2.0` | TBD |
| `plotly` | unpinned | `plotly>=5.18` | TBD |
| `dash` | unpinned | `dash>=2.14` | TBD |
| `spikeinterface` | `spikeinterface=0.104.3` (conda) | `spikeinterface==0.104.3` (pip) | env.yml conda authoritative (newer); Dockerfile pip mirrors |
| `pyyaml` | conda | (in RUNTIME_SPEC) | both — Dockerfile keeps for non-conda layer |
| `docker` | pip | (in RUNTIME_SPEC) | both — used by container_cli |
| `pypdf` | pip | (in RUNTIME_SPEC) | both — used by report tooling |
| `scikit-learn` | not in env.yml | `scikit-learn<2.0` | Dockerfile-only — investigate why; likely a downstream dep that should be explicit |
| `nvidia-ml-py` | not in env.yml | `nvidia-ml-py` (unpinned) | Dockerfile-only — GPU observability; conda env doesn't have GPU access on dev machines, OK to skip |
| `axon_velocity` | not in env.yml | `axon_velocity==0.1.2` | Dockerfile-only — conda env didn't install it; gap that the env-parity guardrail catches |
| `mpi4py` | not in env.yml | `mpi4py` | Dockerfile-only — conda env doesn't need MPI (login-node dev) |
| `joblib` | implied via SI conda | `joblib` (in UNITMATCH_RUNTIME_SPEC) | both — explicit |
| `mat73` | not in env.yml | `mat73` (in UNITMATCH_RUNTIME_SPEC) | Manually installed in conda env on 2026-05-19 per USER INJECTION #1; gap documented in env_parity guardrail |
| `mtscomp` | not in env.yml | `mtscomp` (in UNITMATCH_RUNTIME_SPEC) | Dockerfile-only — UMPy dep; conda env doesn't load it |
| `marshmallow` | not in env.yml | `marshmallow` (in SLAY_RUNTIME_SPEC) | Dockerfile-only — SLAy dep |
| `ipykernel` | conda | not in Dockerfile | env.yml-only — Jupyter dev convenience |
| `nodejs` | conda | not in Dockerfile | env.yml-only — Jupyter dev convenience |

## Sibling editable deps

| Sibling | Local path | Public GH remote? | Editable in Docker? | Editable in env.yml? | Conda env editable today? |
|---|---|---|---|---|---|
| `UnitMatchPy` | `/global/homes/a/adammwea/dev/pkgs/UnitMatch/UnitMatchPy/` | yes (parent repo) | optional (`UNITMATCH_SPEC` ARG) | no | yes (manual `pip install -e` on 2026-05-19 per USER INJECTION #1) |
| `SLAy` | (varies — user clone) | yes (parent repo) | optional (`SLAY_SPEC` ARG) | no | varies (manual) |
| `kssynth` | `/global/homes/a/adammwea/dev/pkgs/kssynth/` | **no** (held — per USER INJECTION #4) | no | no | yes (manual `pip install -e`) |
| `unitlink` | `/global/homes/a/adammwea/dev/pkgs/unitlink/` | **no** (held — per USER INJECTION #4) | no | no | yes (manual `pip install -e`) |

Slice 4 (`tools/install_dev_siblings.{sh,py}`) targets initially only
SLAy + UnitMatchPy. kssynth + unitlink wait until their GH remotes are
created (which is itself gated on a successful end-to-end smoke per
USER INJECTION #4).

## Findings to act on in slices 2-3

1. **Choose authoritative version pins** for every drifted dep (table
   above). Slice 2 takes the Dockerfile's pins by default — they're the
   ones that have been tested in production-shape — UNLESS the conda
   side has a stronger reason to differ (the spikeinterface case is a
   tie because both are `0.104.3`).
2. **Migrate the dev-machine-irrelevant deps** (mpi4py, nvidia-ml-py,
   axon_velocity, mat73, mtscomp, marshmallow, scikit-learn) into
   `[full]` extras with the right environment markers OR keep them
   Dockerfile-only. **Important**: `mat73` should NOT be Dockerfile-only
   given the manual workaround on the conda env — surface it through
   `[full]` so a future conda env rebuild picks it up automatically.
3. **Drop `ipykernel` + `nodejs`** from `environment.yml`? They're
   developer convenience for Jupyter; could live in `[dev]` instead so
   contributors who don't want them can skip. Decision deferred to
   slice 3.
4. **Eliminate `UNITMATCH_SPEC` + `SLAY_SPEC` ARGs** in slice 6 — the
   `[full]` extra will pull these from git URLs (UnitMatchPy public repo;
   SLAy public repo). `UNITMATCH_RUNTIME_SPEC` (joblib mat73 mtscomp) +
   `SLAY_RUNTIME_SPEC` (marshmallow) become regular pip deps under
   `[full]`.
5. **Decision on `--no-deps`**: the Dockerfile installs UnitMatchPy and
   SLAy with `--no-deps` and supplies their runtime deps separately.
   `[full]` would lose this isolation unless we pin every transitive
   dep. Slice 6 needs to decide whether to use `--no-deps` via a
   special install command, or to trust the siblings' own dep declarations.

## Out-of-scope for this audit

- The CUDA + Kilosort base image carve-out stays. The conda env doesn't
  get GPU; the Dockerfile inherits its CUDA layer from the base image.
- The Maxwell HDF5 plugin install (binary download) stays a
  Dockerfile-only step. Adding it to `environment.yml` would require a
  conda-forge package which doesn't exist.
- Apt packages stay Dockerfile-only (no conda equivalent for openmpi-bin).
