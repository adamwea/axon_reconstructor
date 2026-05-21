# Environment-parity guardrail

## Contract

The `axon_recon` conda env (`~/.conda/envs/axon_recon/`) and the
`adammwea/axon-recon:pipeline-v2` shifter image must have **equivalent
capabilities** — anything that runs in one should run in the other —
**except** for these explicitly allowed differences:

1. **Kilosort + its CUDA stack**: lives in the shifter image only. The
   shifter bundles KS plus the CUDA toolkit, cuDNN, NCCL, etc. that the
   conda env doesn't (or doesn't fully) carry.
2. **NERSC / HPC / SLURM runtime plumbing**: the shifter image is
   configured to run cleanly under NERSC's MPI / SLURM srun / cgroup
   limits — `mpi4py` linked against `cray-mpich`, shifter-specific
   entrypoint shimming, container preflight scripts. That stack lives
   in the shifter image only.

Outside those two carve-outs, anything installable in one should be
present in the other, at the same major+minor version where feasible.

## The artifacts that encode capability

Parity is maintained across a set of artifacts; a slice that touches
capabilities touches some subset of them in the SAME commit range. Don't
leave one stale.

**Target shape** (per `plans/active/env_install_unification_plan.md` — under
execution; some artifacts don't exist yet):

| Artifact | What it covers | Sibling editables? |
|---|---|---|
| **`pyproject.toml` extras** | Single source of truth for pip deps. Extras: `[dev]` (testing/linting), `[full]` (production siblings as git URL pins where they have public GH presence). | Production siblings: git URL pins under `[full]` |
| **`environment.yml`** | Conda-only deps (python, conda-forge-preferred scientific stack). Ends in `pip: - -e .[dev,full]` (or `.[dev]` only when editable-siblings is used). | No — excluded by design |
| **`tools/setup_env.sh [--editable-siblings] [--from-local PATH]`** | One user-facing install command. Wraps `conda env create` + `pip install -e .[dev,full]` + optional editable-siblings sub-step. **Does not exist yet** — slice 5 of the plan. | Optional via `--editable-siblings` |
| **`tools/install_dev_siblings.{sh,py}`** | Editable install of siblings. Prefers `~/dev/pkgs/<name>/` via `--from-local`; falls back to gitignored `deps/` inside repo. **Does not exist yet** — slice 4 of the plan. | Yes (editable) |
| **`containers/axon-recon/Dockerfile`** | Shifter image — mirrors via `pip install -e .[full]` once plan slice 6 lands. Kilosort + CUDA + NERSC/HPC plumbing remain shifter-only. | Production via `[full]` |

**Current shape (pre-plan)** — what's actually in the repo today:

- `environment.yml`: main conda deps + a `pip:` list with `docker`, `pypdf`, `-e .[dev]`. Sibling editables explicitly excluded ("see comment line 31-32").
- `pyproject.toml`: exists but has no `[full]` extra yet — pip deps still scattered across env.yml and the Dockerfile.
- `containers/axon-recon/Dockerfile`: ARG-driven per-sibling spec strings (`AXON_RECON_RUNTIME_SPEC`, `UNITMATCH_SPEC`, `UNITMATCH_RUNTIME_SPEC`, `SLAY_SPEC`, `SLAY_RUNTIME_SPEC`, `MPI4PY_SPEC`).
- `tools/setup_env.sh`, `tools/install_dev_siblings.{sh,py}`: do not exist.

Until the plan ships, slices interpret the guardrail against the current
shape; once plan slices land, the artifacts they touch move under the
target-shape contract.

## Why

- The conda env is the **development surface**: login-node smokes (cap
  64 procs, local_affinity), real-call pytest runs, laptop iteration.
- The shifter image is the **production surface**: real allocations on
  Perlmutter GPU/CPU nodes, KS sorting, SLURM-managed multi-node runs.
- Tests that pass in conda but fail in shifter (or vice versa) cost
  an entire iteration cycle — the divergence only shows up after
  rebuild → docker push → `shifterimg pull`, which is slow. Parity
  collapses that loop.
- Cross-env contract bugs ("the sibling package imports in conda but
  not in shifter") are the most common silent-failure surface — they
  pass tests, pass login smokes, then break on the first
  real-allocation run.
- Anchors the workflow: ship a working install in conda first, then
  mirror in the Dockerfile, then rebuild + push + `shifterimg pull`.

## Sub-rules

These rules apply against whichever shape (current or target) is live for
the artifact in question. As the unification plan
(`plans/active/env_install_unification_plan.md`) ships slice by slice, the
governing artifact for a given concern shifts.

1. **New runtime Python dep** (e.g. `mat73`, or any non-sibling package):
   - **Target shape**: add to `pyproject.toml`'s `[full]` extra (or `[dev]`
     if it's a testing/linting tool). Conda-installable packages preferred
     via conda-forge can additionally live in `environment.yml`'s conda
     deps; everything else flows through pyproject.
   - **Current shape (pre-plan)**: add to BOTH `environment.yml` (under
     `pip:` or main conda deps) AND the Dockerfile's appropriate
     `*_RUNTIME_SPEC` ARG default. If hidden behind an inactive ARG (e.g.
     `mat73` is in `UNITMATCH_RUNTIME_SPEC` but gated by `UNITMATCH_SPEC`),
     fix the activation gap or move the dep to an always-installed ARG.

   **The loop IS authorized to trigger shifter rebuilds (as of
   2026-05-19, per user authorization)** via the local container
   toolchain on Perlmutter (`podman build` → `podman push docker.io/...`
   → `shifterimg pull docker:...`). Before attempting a rebuild the
   loop runs `podman login --get-login docker.io` to confirm credentials;
   if not logged in, it falls back to posting a "shifter rebuild
   blocked: docker.io login required" line under
   `dev/notes/memory/current_state.md` §"⚡ USER INJECTIONS" instead of
   attempting a build that will fail at push time. NEVER use `docker` —
   Perlmutter has `podman` only.

   **Docker.io authfile MUST live at `$HOME/.config/containers/auth.json`**
   on NERSC. The default rootless authfile path
   (`$XDG_RUNTIME_DIR/containers/auth.json` = `/run/user/<uid>/containers/`)
   is tmpfs and gets garbage-collected by systemd-logind when all SSH
   sessions to a login node disconnect, OR when the user switches between
   login nodes (each has its own `/run/user/`). The persistent fix
   (DIRECTIVE C, shipped 2026-05-19): `~/.config/containers/auth.json`
   exists at `chmod 600`; `~/.bashrc` exports
   `REGISTRY_AUTH_FILE="$HOME/.config/containers/auth.json"`;
   `~/.config/containers/containers.conf` has
   `[engine] auth_file = "/global/homes/<user>/.config/containers/auth.json"`
   as a fallback for non-interactive shells. With this setup,
   `podman login --get-login docker.io` survives session disconnects;
   user-initiated `podman login docker.io` only needs to happen on actual
   password change.

2. **Sibling-package install** (`SLAy`, `UnitMatchPy`, `kssynth`, `unitlink`):
   - **Target shape, production (non-editable)**: add to
     `pyproject.toml`'s `[full]` extra as a git URL pin (e.g.
     `unitlink @ git+https://github.com/<owner>/unitlink.git@<sha>`). Only
     possible once the sibling has a public GH remote.
   - **Target shape, editable for dev**: add a `pip install -e <path>` line
     to `tools/install_dev_siblings.{sh,py}`. The script prefers
     `~/dev/pkgs/<name>/` clones via `--from-local`; falls back to cloning
     into gitignored `deps/`.
   - **Current shape (pre-plan)**: editables documented in
     `current_state.md` §"⚡ USER INJECTIONS" with the explicit
     `pip install -e` command (the bootstrap script doesn't exist yet —
     superseded by the plan). Dockerfile: relevant `<NAME>_SPEC` ARG
     default is set so the build installs the sibling; new siblings
     (kssynth, unitlink) get new ARG blocks alongside `UNITMATCH_SPEC` /
     `SLAY_SPEC` until plan slice 6 simplifies this.

3. **Tool / binary additions** (rare): same parity rule. Conda env via
   `environment.yml` if conda-installable, otherwise an apt step. Mirror
   in Dockerfile.

4. **CUDA-stack or NERSC-stack additions**: carve-out — shifter only.
   Document the carve-out reason on the Dockerfile line so the next reader
   understands why the conda env doesn't mirror it. NEVER add to
   `environment.yml`.

5. **Removing a dep**: remove from BOTH halves — target shape:
   `pyproject.toml` extras AND Dockerfile (which mirrors via
   `pip install -e .[full]` once plan slice 6 lands); pre-plan:
   `environment.yml` AND Dockerfile. No silent skew either direction.

6. **Version drift**: when a dep exists in both envs and a slice updates
   its pinned version, update the matching artifact too. If it's not
   feasible (e.g. shifter is stuck on an older version for a
   container-level reason), document the divergence in §"Open exceptions"
   below.

7. **Slice-level commit discipline**: any slice that adds/removes/upgrades
   a conda dep MUST touch the appropriate artifact pair in the same commit
   range. Absence of one is a regression of this guardrail.

## Tests / verification

- **Slice-level check**: any slice that installs / upgrades / removes
  a conda dep MUST either (a) touch the Dockerfile in the same commit
  range, or (b) post a "shifter rebuild needed: X" line under USER
  INJECTIONS. Absence of both = regression of this guardrail. Reviewer
  (or next-iteration audit pass) flags as such.
- **Parity audit script** (proposed, not yet implemented):
  `tools/audit_env_parity.py` — diff `pip list` between conda env and
  inside shifter; report any non-carve-out delta. Run as part of the
  shifter rebuild workflow. Scheduled for the slice when container/MPI
  alignment work lands (`trackers/tech_debt.md`).
- **Smoke-test gate**: any integration slice that touches kssynth /
  unitlink / a new Python dep requires verification in BOTH conda env
  (login-node smoke) AND shifter (after rebuild). Parity isn't proved
  until both pass.

## Conda-env-only setup steps (NOT covered by `pip install`)

Some capabilities the conda env needs aren't installable via PyPI — they need a manual setup step. The shifter image bakes these into the Dockerfile build; the conda env requires them to be run once after `pip install -e .[dev]`. Each step here is a one-shot, idempotent install.

- **MaxWell HDF5 decompression plugin** (required for loading reference data's preprocessed_segments via spikeinterface):

  ```
  python -c "from neo.rawio.maxwellrawio import auto_install_maxwell_hdf5_compression_plugin; auto_install_maxwell_hdf5_compression_plugin(force_download=False)"
  ```

  Plugin lands at `~/hdf5_plugin_path_maxwell/libcompression.so`. The neo helper also sets `HDF5_PLUGIN_PATH=~/hdf5_plugin_path_maxwell` at module import; for explicit-shell invocations (login-node smokes, etc.), export it manually:

  ```
  export HDF5_PLUGIN_PATH=$HOME/hdf5_plugin_path_maxwell
  ```

  Skipping this step makes `spikeinterface.load(...)` of MaxWell-compressed `.json` recordings fail with `Can't synchronously read data (can't open directory (/usr/local/hdf5/lib/plugin). Please verify its existence)`. Discovered during kssynth slice 3b heavy smoke prep, 2026-05-21.

## Open exceptions / follow-ups

- **Unification plan in flight**: `plans/active/env_install_unification_plan.md`
  is the destination spec for the artifact set described in §"The artifacts
  that encode capability". Until that plan ships (slices 1-8), the
  guardrail's sub-rules cover BOTH the current shape (env.yml + Dockerfile)
  and the target shape (pyproject.toml extras + setup_env.sh +
  install_dev_siblings + Dockerfile). Each slice of the plan flips one
  artifact pair from current to target; this guardrail is amended to match.
- **`tools/bootstrap_editable_deps.sh` doesn't get built**: the
  `environment.yml` comment (line 31-32) references this script, but per
  the unification plan it's superseded by
  `tools/install_dev_siblings.{sh,py}` (slice 4) — same role, fits the
  flag-driven `setup_env.sh` entry point cleanly. The `environment.yml`
  comment should be updated as part of plan slice 3.
- **Current gap (2026-05-19)**: this iteration's UMPy unblocker landed
  `pip install -e /global/homes/a/adammwea/dev/pkgs/UnitMatch/UnitMatchPy/`
  + `pip install mat73` into the conda env directly, without updating any
  capability artifact. The unification plan resolves this cleanly:
  - **Plan slice 2** moves `mat73` into `pyproject.toml`'s `[full]` extra,
    and `UnitMatchPy` too (as git URL pin) — UMPy has a public GH remote.
  - **Plan slice 4** adds the editable install of `UnitMatchPy` (and
    `SLAy`) to `tools/install_dev_siblings.{sh,py}` so dev workflows can
    edit in place.
  - **Plan slice 6** rebuilds the Dockerfile to install via
    `pip install -e .[full]`, closing the shifter side.

  Short-term (until plan slices land), the existing
  `current_state.md` USER INJECTIONS [2026-05-19] entry documents the
  manual `pip install -e ...` + `pip install mat73` workaround.

- **kssynth + unitlink (v1)**: both live at
  `~/dev/pkgs/{kssynth,unitlink}/` with their own `git init` history;
  no GH remote yet (per "hold until real-data validation" directive).
  Once GH remotes exist:
  - they get added to `pyproject.toml`'s `[full]` extra as git URL pins
    (production), plus
  - `tools/install_dev_siblings.{sh,py}` (editable for dev), plus
  - Dockerfile's `pip install -e .[full]` picks them up automatically
    post-plan-slice-6.

  Until then, when the loop hits kssynth slice 9 (axon_recon recon-stage
  integration), that slice documents the editable install command under
  USER INJECTIONS and uses the pre-plan Dockerfile ARG approach
  (`KSSYNTH_SPEC` + `UNITLINK_SPEC` blocks alongside `UNITMATCH_SPEC`).
  These ARG blocks become unnecessary once plan slice 6 ships.
- **mpi4py version**: conda env's pip-installed `mpi4py` will differ from
  shifter's NERSC-compatible build (which links against cray-mpich). This
  is the canonical "NERSC carve-out" case; acceptable but worth recording
  the conda version too for awareness.
