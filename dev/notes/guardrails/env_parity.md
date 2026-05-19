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

## The three artifacts that encode capability

Parity is maintained across three files; a slice that touches capabilities
touches some subset of these in the SAME commit range. Don't leave one
stale.

| Artifact | What it covers | Sibling editables? |
|---|---|---|
| `environment.yml` (repo root) | Conda env source-of-truth: main conda deps + a `pip:` list for non-sibling pip packages. Recreating the env from this file is the contract. | NO — sibling editables are explicitly excluded; see env.yml's comment. |
| `tools/bootstrap_editable_deps.sh` | Editable installs of sibling packages (`SLAy`, `UnitMatchPy`, `kssynth`, `unitlink`, etc.) into the conda env post-`conda env create`. **DOES NOT YET EXIST** — see Open exceptions. | YES — exclusive home. |
| `containers/axon-recon/Dockerfile` | Shifter image — ARG-driven: `AXON_RECON_RUNTIME_SPEC` (pip deps), `UNITMATCH_SPEC` / `UNITMATCH_RUNTIME_SPEC`, `SLAY_SPEC` / `SLAY_RUNTIME_SPEC`, `MPI4PY_SPEC`, etc. All capability deltas land here, default-args or via `--build-arg` at build time. | YES — via the per-sibling `<NAME>_SPEC` ARG. |

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

1. **New runtime Python dep** (e.g. `pip install mat73`, or any non-sibling
   package): the dep must land in BOTH:
   - **`environment.yml`** — added under the `pip:` list (or main conda deps
     if conda-forge has it) so `conda env create -f environment.yml` reproduces
     the capability.
   - **`containers/axon-recon/Dockerfile`** — added to the appropriate
     `*_RUNTIME_SPEC` ARG default (usually `AXON_RECON_RUNTIME_SPEC` for
     general deps, or a sibling-specific spec if the dep is for that sibling).
     If the dep is hidden behind an inactive ARG (e.g. `mat73` lives in
     `UNITMATCH_RUNTIME_SPEC` but is only installed when `UNITMATCH_SPEC` is
     non-empty), the slice fixes that activation gap or moves the dep to an
     always-installed ARG.

   The loop never triggers the shifter rebuild itself — instead it posts a
   "shifter rebuild needed: X" line under
   `dev/notes/memory/current_state.md` §"⚡ USER INJECTIONS" so the user
   knows what to bake into the next rebuild.

2. **Sibling-package editable install** (`kssynth`, `unitlink`, `SLAy`,
   `UnitMatchPy`): editables are kept OUT of `environment.yml` by design.
   They land in BOTH:
   - **`tools/bootstrap_editable_deps.sh`** — a `pip install -e <path>` line
     for the sibling. Until this script exists (see Open exceptions), the
     slice that needs the install documents the explicit `pip install -e`
     command in `dev/notes/memory/current_state.md` §"⚡ USER INJECTIONS".
   - **`containers/axon-recon/Dockerfile`** — the relevant `<NAME>_SPEC` ARG
     default is set so the build installs the sibling. If a sibling doesn't
     have an ARG block yet (kssynth, unitlink), add one alongside the
     existing UNITMATCH_SPEC / SLAY_SPEC blocks.

   Both artifacts must reflect the install before any axon_recon integration
   slice can claim parity.

3. **Tool / binary additions** (rare): same parity rule. Conda env via
   `environment.yml` if conda-installable, otherwise an apt or pip install
   in a bootstrap step. Mirror in Dockerfile.

4. **CUDA-stack or NERSC-stack additions**: carve-out — shifter only.
   Document the carve-out reason on the Dockerfile line so the next reader
   understands why the conda env doesn't mirror it. NEVER add to
   `environment.yml`.

5. **Removing a dep from conda env**: remove from BOTH `environment.yml`
   (or `bootstrap_editable_deps.sh` for siblings) AND the Dockerfile in the
   same slice. No silent skew in either direction.

6. **Version drift**: when a dep exists in both envs and a slice updates the
   conda version, update the corresponding Dockerfile ARG too. If it's not
   feasible (e.g. shifter is stuck on an older version for a container-level
   reason), document the divergence in §"Open exceptions" below.

7. **Slice-level commit discipline**: any slice that adds/removes/upgrades a
   conda dep MUST touch `environment.yml` (or `bootstrap_editable_deps.sh`)
   AND the Dockerfile in the same commit range. Absence of one is a
   regression of this guardrail.

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

## Open exceptions / follow-ups

- **`tools/bootstrap_editable_deps.sh` doesn't exist yet**: `environment.yml`
  references it ("Install them after env creation with
  tools/bootstrap_editable_deps.sh") but the script has never been written.
  Until it lands, sibling editable installs are documented in
  `dev/notes/memory/current_state.md` §"⚡ USER INJECTIONS" with the explicit
  `pip install -e` command per sibling. **Action**: book a tracker item to
  scaffold this script + populate it with `SLAy`, `UnitMatchPy`, `kssynth`,
  `unitlink` editable installs. Run it in CI / smoke-test after `conda env
  create` to verify it works.
- **Current gap (2026-05-19)**: this iteration's UMPy unblocker landed
  `pip install -e /global/homes/a/adammwea/dev/pkgs/UnitMatch/UnitMatchPy/`
  + `pip install mat73` into the conda env directly, without updating
  `environment.yml` or the Dockerfile. To close:
  - **`environment.yml`**: add `mat73` to the `pip:` list. UnitMatchPy stays
    out (sibling editable) but the install command is documented under USER
    INJECTIONS until the bootstrap script exists.
  - **Dockerfile**: `mat73` is already in `UNITMATCH_RUNTIME_SPEC` (line 15)
    but is gated behind `if [ -n "${UNITMATCH_SPEC}" ]` (line 50). Either set
    `UNITMATCH_SPEC` default to a UMPy install spec, OR move `mat73` to
    `AXON_RECON_RUNTIME_SPEC` (always installed), OR pass
    `--build-arg UNITMATCH_SPEC=...` at build time. Pick the cleanest
    approach when doing the rebuild.
  - Tracked under `current_state.md` USER INJECTIONS [2026-05-19].
- **kssynth + unitlink (v1)**: both live at
  `~/dev/pkgs/{kssynth,unitlink}/` with their own `git init` history;
  neither is installed in conda env yet (pytest runs from each package's
  own dir). When kssynth slice 9 (axon_recon recon-stage integration) ships,
  THAT slice MUST:
  - add the editable install commands to (the future)
    `bootstrap_editable_deps.sh` or document under USER INJECTIONS,
  - add new ARG blocks to the Dockerfile (`KSSYNTH_SPEC` + `UNITLINK_SPEC`
    alongside `UNITMATCH_SPEC` / `SLAY_SPEC`), each set to install the
    package editable from `/opt/axon_recon/.../` post-`COPY . /opt/axon_recon`,
  - post a "shifter rebuild needed: kssynth + unitlink editable installs"
    line under USER INJECTIONS for the user to action.
- **mpi4py version**: conda env's pip-installed `mpi4py` will differ from
  shifter's NERSC-compatible build (which links against cray-mpich). This
  is the canonical "NERSC carve-out" case; acceptable but worth recording
  the conda version too for awareness.
