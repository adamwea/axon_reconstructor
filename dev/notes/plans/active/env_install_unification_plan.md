# Env-install unification plan

> **Status (2026-05-19)**: ✅ **v1 COMPLETE — slices 1-8 SHIPPED + shifter rebuild SHIPPED.**
>
> Shipped commits:
> - Slice 1 audit: `274b62e` (`env_install_unification_audit.md`)
> - Slice 2 pyproject.toml `[full]`: `dbd8ae7`
> - Slice 3 environment.yml shrink: `b571824`
> - Slice 4 `install_dev_siblings.sh`: `2939194`
> - Slice 5 `setup_env.sh`: `ba290a8`
> - Slice 6 Dockerfile collapse: `7d7ced1`
> - Slice 7 `deps/` .gitignore: `03ce49c`
> - Slice 8 README docs: `f743abe`
>
> Shifter rebuild result (2026-05-19 21:56 PDT, commit `8c3e7c2`): image
> `adammwea/axon-recon:pipeline-v2` now READY in NERSC shifter registry
> at hash `9cdca44d9b` (down from 17-20 GiB projected to 12.8 GiB actual,
> per the DIRECTIVE B `[full]` → `[full-cuda]` split). In-container smoke
> verified UMPy/torch+CUDA/spikeinterface/axon_velocity/transitives all
> import. See `current_state.md` USER INJECTIONS for the full record.
>
> The remaining open items are:
> (a) ~~user-initiated shifter rebuild~~ — DONE (loop-driven 2026-05-19).
> (b) lift the `kssynth` / `unitlink` GH-remotes hold (USER INJECTION #4)
>     and add them to `[full]`.
> (c) decide on SLAy's install path. The new shifter image is MISSING
>     SLAy — `env_install_unification` slice 6's Dockerfile collapse
>     dropped the `SLAY_SPEC` ARG without adding SLAy to `[full]`/`[full-cuda]`.
>     `merge_SLAy` will ImportError on the new image until this is closed.
>     SLAy has a public remote at `git@github.com:adamwea/SLAy.git`; adding
>     `"SLAy @ git+https://github.com/adamwea/SLAy.git"` to both extras +
>     rebuilding the shifter image closes the gap. Same shape as (b).

## Motivation

Today's install story has three+ artifacts to keep in sync:
`environment.yml`, `containers/axon-recon/Dockerfile`, and a
referenced-but-missing `tools/bootstrap_editable_deps.sh` for sibling
editables. Sibling installs land via user-specific paths or build-time
ARGs, and a contributor who clones the repo has no single command to get
a working dev env. The `env_parity` guardrail (`guardrails/env_parity.md`)
holds the contract together with documentation, but the underlying
mechanics don't yet live in a single source of truth.

This plan unifies the install story onto **`pyproject.toml` as the
single source of truth for pip deps**, with **a small entry-point script
plus `--editable-siblings` flag** for the development path (the
modularity the user asked for). The shifter image stays the recommended
default for production use; editable siblings are a development-phase
affordance, not a long-term repo feature.

Aligned with user direction (2026-05-19):
- "If installing the conda env could just be done with a flag so the
  siblings are editable that'd be nice. So the operation is the same,
  just modular."
- "Recommended use will always be to use the docker / shifter image since
  using the conda env requires multiple cloning operations."
- "Having them install as editable while we're in this development phase
  is a nice option, but shouldnt be a long-term feature of the repo
  necessarily."
- "Perhaps we should eventually have build cloning sibling repos somewhere
  within the repo for easy install by other users."

## Target shape (post-plan)

| Artifact | Role | Sibling editables? |
|---|---|---|
| **`pyproject.toml`** extras | Single source of truth for pip deps. Extras: `[dev]` (testing/linting), `[full]` (siblings as git URL pins for production, non-editable). Grows as siblings reach public GH. | Production: git URL pins under `[full]` |
| **`environment.yml`** | Conda-only deps (python, conda-forge-preferred scientific stack). Final line: `pip: - -e .[dev,full]` (or `.[dev]` only when the editable-siblings flag is used). | No (excluded by design) |
| **`tools/setup_env.sh [--editable-siblings] [--from-local PATH]`** | One user-facing install command. Wraps `conda env create -f environment.yml` + activation + `pip install -e .[dev,full]` + optional editable-siblings sub-step. | Optional, via `--editable-siblings` |
| **`tools/install_dev_siblings.{sh,py}`** | Editable install of siblings. Prefers an existing clone at `~/dev/pkgs/<name>/` (via `--from-local`); falls back to cloning into gitignored `deps/` inside the repo. Idempotent. Sub-script invoked by `setup_env.sh`. | Yes (editable) |
| **`containers/axon-recon/Dockerfile`** | Shifter image — mirrors via `pip install -e .[full]` (same mechanism), eliminating per-sibling `<NAME>_SPEC` ARGs where possible. Kilosort + CUDA + NERSC/HPC plumbing remain shifter-only carve-outs per `env_parity` guardrail. | Production via `[full]` |

`deps/` is a gitignored directory inside the repo that the script populates
lazily when no local clone is available. **NOT git submodules** —
submodules have UX hazards (detached HEADs, sync drift) and the script
approach is reproducible without them.

## Install paths (post-plan)

1. **Shifter image (recommended for production use)** — no local deps;
   one container, everything baked in.
2. **`pip install -e .[full]`** — for users who want a local install
   without hacking on siblings. Pulls siblings from git URL pins.
3. **`./tools/setup_env.sh --editable-siblings`** — for contributors
   hacking on siblings. Clones / reuses local sibling clones, installs
   editable.

## Slices (rough — refine as we go)

1. **Audit current install paths**: inventory every pip dep across
   `environment.yml`, the Dockerfile ARGs (`AXON_RECON_RUNTIME_SPEC`,
   `UNITMATCH_SPEC`, `UNITMATCH_RUNTIME_SPEC`, `SLAY_SPEC`, etc.), and
   any ad-hoc install commands in docs/scripts. Document existing
   version pins. Output: `dev/notes/plans/active/env_install_unification_audit.md`
   or in-place section here.

2. **Migrate pip deps to `pyproject.toml` extras**: collapse the audit
   into `[dev]` (testing/linting) and `[full]` (production-grade incl.
   siblings as git URL pins where they have public GH presence —
   initially `UnitMatchPy` only; kssynth/unitlink wait until their
   remotes are created per the existing "hold on GH remotes"
   directive). Don't change install behavior yet — both env.yml and
   Dockerfile still drive their own installs. Smoke: `pip install -e
   .[dev]` and `pip install -e .[full]` both work from a scratch
   conda env.

3. **Shrink `environment.yml`**: remove pip deps that moved to
   `pyproject.toml`. Replace with `pip: - -e .[dev,full]` line. Keep
   conda-only deps (python, numpy, scipy, etc.). Smoke: `conda env
   create -f environment.yml` produces a working dev env from scratch
   that can `pytest` cleanly.

4. **Write `tools/install_dev_siblings.{sh,py}`**: idempotent
   editable-install helper. Args: `--from-local PATH` (prefer local
   clones), default fall-back to `deps/` inside repo. Targets initially:
   SLAy, UnitMatchPy. kssynth + unitlink added once GH remotes exist
   (per user's "hold on GH remotes" directive). Tests: dry-run mode +
   idempotency check.

5. **Write `tools/setup_env.sh`**: one user-facing entry point. Args:
   `--editable-siblings`, `--from-local PATH`. Wraps:
   `conda env create -f environment.yml` (skip if exists),
   `conda activate axon_recon`, `pip install -e .[dev]` (production
   extras `[full]` if NOT `--editable-siblings`, else skipped),
   optional `./tools/install_dev_siblings.{sh,py}`. Idempotent. Smoke:
   fresh user clone + `./tools/setup_env.sh` works.

6. **Migrate Dockerfile to `pip install -e .[full]`**: eliminate the
   per-sibling `<NAME>_SPEC` ARGs where possible. Kilosort + CUDA +
   NERSC/HPC plumbing stay in the Dockerfile. The Dockerfile becomes
   noticeably shorter. Smoke: shifter rebuild + smoke-test inside
   container that the import chain works.

7. **Add `deps/` to `.gitignore`** and document its role in README.

8. **README + docs**: update install instructions. Three paths
   documented: shifter (recommended), `pip install .[full]`
   (production-style local), `./tools/setup_env.sh --editable-siblings`
   (dev with hacking). Cross-reference the env_parity guardrail.

## Smoke tests

- After slice 3: `conda env create -f environment.yml` on a fresh
  machine path produces a working axon_recon dev env (`pytest` runs).
- After slice 5: `./tools/setup_env.sh` from a fresh user clone produces
  a working env. `./tools/setup_env.sh --editable-siblings --from-local
  ~/dev/pkgs/` produces an editable dev env.
- After slice 6: shifter image built from the new Dockerfile has
  equivalent capabilities to the conda env (parity audit via `pip list`
  diff inside container vs. outside).

## Done criteria

- One `pip install` command (`pip install -e .[full]`) is the canonical
  production install.
- `./tools/setup_env.sh --editable-siblings` is the canonical dev install.
- `pyproject.toml` is the single source of truth for pip deps.
- `environment.yml` is conda-only.
- The Dockerfile is shorter and uses the same `pip install -e .[full]`
  command for production deps (except Kilosort + CUDA + NERSC carve-outs).
- The `env_parity` guardrail's "three-artifact" model is refreshed to
  match the new shape.

## Open decisions to resolve during execution

- **Versioning strategy for sibling git URL pins**: branch (`@main`) vs.
  tag (`@v0.1.0`) vs. commit SHA (`@<sha>`). Probably commit SHA for
  reproducibility; tags once siblings have proper releases.
- **`--from-local` semantics when sibling is missing at PATH**: error
  vs. fall back to clone. Error is safer (catches typos).
- **`deps/` location**: inside repo (current proposal) vs. strictly
  outside (`~/dev/pkgs/`). Compromise: script supports both via
  `--from-local`; default to `deps/`.
- **`[full]` extras vs. `[siblings]` extras**: do we want a separate
  `[siblings]` extra for "all sibling packages" so users can opt out
  of siblings without losing the rest of `[full]`? Probably yes; keep
  scope flexible.
- **Conda-vs-pip choice for numpy/scipy/pyarrow**: conda-forge usually
  ships better wheels for these on HPC. Decide per-dep during slice 3.

## Relationship to existing artifacts

- **Supersedes** `trackers/tech_debt.md` §"Scaffold
  `tools/bootstrap_editable_deps.sh`" — that script's role is taken by
  `tools/install_dev_siblings.{sh,py}` here. Tracker entry amended in
  same commit range as this plan.
- **Amends** `guardrails/env_parity.md`'s "three-artifact" model to a
  new five-artifact model (pyproject.toml + environment.yml +
  setup_env.sh + install_dev_siblings + Dockerfile). Guardrail amended
  in same commit range as this plan.
- **Resolves** the [2026-05-19] USER INJECTIONS gap for UMPy + mat73
  cleanly:
  - `mat73` moves into `pyproject.toml` `[full]` extra in slice 2.
  - UnitMatchPy moves into `pyproject.toml` `[full]` extra (as git URL
    pin) in slice 2 once its GH remote spec is locked in; editable
    install for dev happens via `install_dev_siblings.sh` from slice 4.
  - Dockerfile rebuild after slice 6 closes the parity gap for both.
  - The current short-term injection (manual `pip install -e ...` +
    `pip install mat73` until plan lands) remains valid in the meantime.

## Out of scope

- Migrating from conda to pure pip (no — conda env stays the dev front
  door per user direction; shifter is the recommended default).
- Publishing siblings to PyPI (separate workstream; the git URL pin
  approach is a stepping stone that survives PyPI publication later —
  the URLs just become version specs).
- Renaming the conda env from `axon_recon`.
- Container registry / push automation (separate concern).
