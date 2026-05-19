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

1. **New runtime Python dep added to conda env** (e.g.
   `pip install mat73`): the same dep MUST be added to the shifter
   Dockerfile in the same slice/branch. The loop never triggers the
   rebuild itself — instead it posts a "shifter rebuild needed: X" line
   under `dev/notes/memory/current_state.md` §"⚡ USER INJECTIONS" so the
   user knows what to bake into the next rebuild.
2. **Sibling-package install** (`kssynth`, `unitlink`, `SLAy`,
   `UnitMatchPy`): editable `pip install -e` in conda env; matching
   `RUN pip install -e <path>` in the shifter Dockerfile, same source
   path layout. Both must be present before any axon_recon integration
   slice can claim parity.
3. **Tool / binary additions** (rare): same parity rule.
4. **CUDA-stack or NERSC-stack additions**: carve-out — shifter only.
   Document the carve-out reason on the Dockerfile line so the next
   reader understands why the conda env doesn't mirror it.
5. **Removing a dep from conda env**: remove it from the Dockerfile in
   the same slice. No silent skew either direction.
6. **Version drift**: when a dep already exists in both envs and a
   slice updates the conda version, update the Dockerfile pin too. If
   it's not feasible (e.g. shifter is stuck on an older version for a
   container-level reason), document the divergence in this guardrail's
   §"Open exceptions" section.

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

- **Current gap (2026-05-19)**: `UnitMatchPy` (editable, from
  `/global/homes/a/adammwea/dev/pkgs/UnitMatch/UnitMatchPy/`) plus
  `mat73` installed in conda env. Shifter image does not yet have
  them. **Action**: user adds both to Dockerfile + rebuilds + pushes
  + `shifterimg pull`. Tracked under `current_state.md` USER
  INJECTIONS.
- **kssynth + unitlink (v1)**: both live at
  `~/dev/pkgs/{kssynth,unitlink}/` with their own `git init` history;
  neither is installed in conda env yet (pytest runs from each
  package's own dir). Once an axon_recon slice imports either, both
  envs need an editable install. **Action**: when kssynth slice 9
  (axon_recon recon-stage integration) ships, the same slice installs
  kssynth into conda env AND adds it to the Dockerfile.
- **mpi4py version**: conda env's pip-installed `mpi4py` will differ
  from shifter's NERSC-compatible build (which links against
  cray-mpich). This is the canonical "NERSC carve-out" case;
  acceptable but worth recording the conda version too for awareness.
