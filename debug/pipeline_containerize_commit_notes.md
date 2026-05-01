# Pipeline Containerization Commit Notes

Living review log for AI-assisted containerization commits.

Use this file after Adam starts the containerization implementation pass. The instruction files are `debug/pipeline_containerize_instructions.md` and `debug/pipeline_refinement_instructions.md`; review both before each change slice. Once iteration begins, do not edit either instruction file unless Adam explicitly asks. Update this notes file after each AI containerization commit and whenever Adam asks for running notes.

## How To Use

For each commit, append a new entry at the top of `Commit Log` or directly below the most recent entry.

Each entry should let Adam quickly review what changed after an unattended run:

- what was changed
- why it was changed
- acceptance criteria used
- self-checks performed before commit
- what was expected to run and what was confirmed not to run
- tests and smoke checks run
- smoke timeout decisions and log inspection when a smoke is extended
- container build/run impact
- Shifter/NERSC impact
- storage/cache/mount impact
- resume and force-restart impact
- CLI impact inside and outside the container
- any risks, follow-ups, or rollback notes

## Entry Template

```markdown
## YYYY-MM-DD HH:MM - <short_sha> - ai: <commit subject>

Status: accepted | needs follow-up | reverted

Summary:
-

Acceptance Criteria:
-

Self-Check:
- Diff reviewed:
- Unrelated/user edits excluded from commit:
- Instruction files re-read:
- Residual risk:

Expected To Run:
-

Confirmed Not Run:
-

Validation:
- Pytest:
- Smoke:
- Container build/run:
- Logs inspected:
- Not run:

Container / Shifter Impact:
- Local Docker behavior:
- Shifter/NERSC behavior:
- Image size/cache impact:

CLI Impact:
- Normal CLI:
- Container CLI:

Resume / Force-Restart Impact:
- Resume behavior:
- Force-restart behavior:

Storage / Mount Impact:
- Created:
- Modified:
- Required mounts:

Rollback Notes:
-
```

## Commit Log

## 2026-05-01 15:05 - pending - ai: install sibling packages in container

Status: accepted

Summary:
- Added explicit compatibility install knobs for sibling UnitMatchPy and SLAy package builds inside the container.
- Kept sibling installs as normal package installs from the temporary build context, but used `--no-deps` so UnitMatch and SLAy metadata conflicts do not replace the Kilosort4 base image runtime stack.
- Pinned the default runtime dependency set under NumPy 2/Pandas 3 where needed for UnitMatch compatibility, and added the small runtime extras used by the current import paths: `mat73`, `mtscomp`, `joblib`, and `marshmallow`.
- Updated import smoke so UnitMatchPy is validated as an installed distribution/module spec rather than by importing its GUI-heavy top-level package, and SLAy is validated through the pipeline's `slay.run` import path with NumPy fallback for missing `cupy`.
- Updated the container entrypoint so explicit helper commands such as `axon-recon-smoke-cli` run directly while normal stage arguments still forward through `axon-reconstructor`.

Acceptance Criteria:
- Full local probe image builds with sibling UnitMatchPy and SLAy copied into the temporary build context and installed as packages.
- Container import smoke passes for `axon_recon`, `spikeinterface`, `kilosort`, `mpi4py`, UnitMatchPy package discovery, and the SLAy pipeline import path.
- Normal container CLI and wrapper-forwarded CLI still behave like the host CLI.

Self-Check:
- Diff reviewed: yes.
- Unrelated/user edits excluded from commit: yes; only container install/smoke/entrypoint/docs files and these notes are modified.
- Instruction files re-read: yes, `debug/pipeline_containerize_instructions.md` and `debug/pipeline_refinement_instructions.md` before this slice.
- Residual risk: top-level `import UnitMatchPy` still pulls GUI/Tk and fails in this headless image with missing `libX11.so.6`; the smoke validates installed package discovery instead. Top-level `import slay` still fails without `cupy`; the pipeline import path succeeds via the existing NumPy fallback. Real merge-stage data smokes are still needed.

Expected To Run:
- `containers/axon-recon/build_local_image.sh --image axon-recon:full-probe` should build with copied sibling package sources when the sibling checkouts are present.
- `docker run --rm axon-recon:full-probe axon-recon-smoke-cli` should run the container smoke helper directly through the entrypoint.
- `docker run --rm axon-recon:full-probe --help`, `docker run --rm axon-recon:full-probe stages --help`, and wrapper-forwarded `stages --help` should work.

Confirmed Not Run:
- Real data pipeline stages, top-level UnitMatch GUI import, native CUDA/GPU Kilosort execution, and Shifter validation were not run in this slice.

Validation:
- Pytest: not run; this slice only changes container dependency installation, smoke scripts, entrypoint behavior, docs, and notes.
- Smoke: `bash -n containers/axon-recon/build_local_image.sh containers/axon-recon/entrypoint.sh containers/axon-recon/smoke_cli.sh tools/axon-recon-container` passed.
- Smoke: host `containers/axon-recon/smoke_imports.py --allow-missing` passed with expected host-only misses for packages not installed in the host env.
- Container build/run: full sibling build completed successfully as `axon-recon:full-probe`; first patched build completed in `45.1s`, rebuild after entrypoint adjustment completed in `37.4s`.
- Container smoke: `docker run --rm axon-recon:full-probe axon-recon-smoke-cli` passed.
- Container smoke: `docker run --rm axon-recon:full-probe --help`, `docker run --rm axon-recon:full-probe stages --help`, and `tools/axon-recon-container --image axon-recon:full-probe --no-tty stages --help` passed.
- Container smoke: strict import JSON confirmed `axon_recon 0.1.0`, `kilosort 4`, `mpi4py 4.1.1`, `spikeinterface 0.103.2`, UnitMatchPy distribution `3.3.0`, and SLAy distribution `0.1.0` via the pipeline import path.
- Container size: `docker image inspect axon-recon:full-probe` reported about `10.04 GB`.
- Logs inspected: Docker build output, `axon-recon-smoke-cli` output, import smoke JSON, wrapper/CLI smoke output, image size output.
- Diff hygiene: `git diff --check` passed.

Container / Shifter Impact:
- Local Docker behavior: full builds now include sibling packages from copied build context sources without runtime sibling mounts.
- Shifter/NERSC behavior: still uses normal installed Python packages and writable cache env vars; no new NERSC-only runtime assumptions added.
- Image size/cache impact: full sibling image measured about `10.04 GB`, only slightly above the no-sibling runtime-deps image and still below the practical 20 GB concern noted in the instructions.

CLI Impact:
- Normal CLI: no change to host CLI behavior.
- Container CLI: normal arguments still forward through `axon-reconstructor`; explicit commands already on `PATH` can now be run directly through the entrypoint.

Resume / Force-Restart Impact:
- Resume behavior: unchanged.
- Force-restart behavior: unchanged.

Storage / Mount Impact:
- Created: no runtime storage paths.
- Modified: container image Python environment at build time only.
- Required mounts: unchanged; data/config/output/cache mounts are still caller responsibility through the wrapper or Shifter job.

Rollback Notes:
- Revert this commit to return sibling installs to naive `pip install` behavior. The previous no-sibling image path remains available with `--no-unitmatch --no-slay` if a sibling package conflict blocks future work.

## 2026-05-01 14:50 - pending - ai: install container runtime deps by default

Status: accepted

Summary:
- Validated that the initial scaffold Dockerfile can build on top of `spikeinterface/kilosort4-base:4.0.6_cuda-12.0.0`.
- Added default installation of the active pipeline runtime dependency set and `spikeinterface==0.103.2`; the Kilosort4 base image already provided `kilosort` but did not provide `spikeinterface`.
- Updated the container README to document the default runtime dependency installation.

Acceptance Criteria:
- A no-sibling probe image builds successfully from the local Dockerfile.
- The probe image can run normal CLI help through the container entrypoint.
- The host wrapper can forward `stages --help` through Docker to the probe image.
- Container import smoke passes for the dependencies expected in the no-sibling image: `axon_recon`, `kilosort`, `mpi4py`, and `spikeinterface`.

Self-Check:
- Diff reviewed: yes.
- Unrelated/user edits excluded from commit: yes; only the Dockerfile, container README, and these notes are modified.
- Instruction files re-read: yes, `debug/pipeline_containerize_instructions.md` and `debug/pipeline_refinement_instructions.md` before this slice.
- Residual risk: full sibling UnitMatch/SLAy installation is still pending; SLAy and UnitMatch have dependency and import-shape conflicts that should be handled in their own focused slice.

Expected To Run:
- `containers/axon-recon/build_local_image.sh --image axon-recon:probe --no-unitmatch --no-slay` should build a runnable image.
- `docker run --rm axon-recon:probe --help` and `docker run --rm axon-recon:probe stages --help` should work.
- `tools/axon-recon-container --image axon-recon:probe --no-tty stages --help` should forward to the container CLI.

Confirmed Not Run:
- Full sibling build with UnitMatch/SLAy, real data pipeline stages, and Shifter validation were not run in this slice.

Validation:
- Pytest: not run; this slice only changes container dependency installation defaults and docs.
- Smoke: `bash -n containers/axon-recon/build_local_image.sh containers/axon-recon/entrypoint.sh containers/axon-recon/smoke_cli.sh tools/axon-recon-container` passed.
- Smoke: `docker --version` reported Docker `29.1.5`.
- Container build/run: first no-sibling build completed in `239.8s`; rebuilt after the dependency patch in `35.3s` with cached base layers.
- Container smoke: `docker run --rm axon-recon:probe --help`, `docker run --rm axon-recon:probe stages --help`, and wrapper-forwarded `stages --help` all passed.
- Container smoke: `python smoke_imports.py --allow-missing` inside `axon-recon:probe` confirmed `axon_recon 0.1.0`, `kilosort 4`, `mpi4py 4.1.1`, and `spikeinterface 0.103.2`; `UnitMatchPy` and `slay` remain missing by design in the no-sibling build.
- Container size: `docker image inspect axon-recon:probe` reported about `10.01 GB` after runtime deps.
- Logs inspected: Docker build output, container smoke import JSON, wrapper/CLI smoke commands.
- Not run: full test suite, full sibling build, real pipeline stage execution, Shifter validation.

Container / Shifter Impact:
- Local Docker behavior: no-sibling images now include SpikeInterface and the base runtime Python packages needed by the pipeline CLI.
- Shifter/NERSC behavior: image remains below the documented 20 GB risk threshold in this no-sibling probe; CUDA-aware `mpi4py` remains NERSC-deferred despite import success.
- Image size/cache impact: no-sibling probe image is about `10.01 GB`; first pull is heavy because the base image contains a 5.91 GB layer, but rebuilds are fast with cached layers.

CLI Impact:
- Normal CLI: unchanged.
- Container CLI: `axon-reconstructor` help and wrapper-forwarded `stages --help` work inside the probe image.

Resume / Force-Restart Impact:
- Resume behavior: unchanged.
- Force-restart behavior: unchanged.

Storage / Mount Impact:
- Created: local Docker image tag `axon-recon:probe` during validation.
- Modified: `containers/axon-recon/Dockerfile`, `containers/axon-recon/README.md`, and `debug/pipeline_containerize_commit_notes.md`.
- Required mounts: unchanged from the scaffold slice.

Rollback Notes:
- Revert the Dockerfile runtime dependency defaults and README note if the full sibling dependency strategy needs a different base install model.

## 2026-05-01 14:39 - pending - ai: add container scaffold and wrapper

Status: accepted

Summary:
- Added the first container scaffold under `containers/axon-recon`: Dockerfile, entrypoint, smoke import/CLI scripts, README, and a local build helper that can copy sibling UnitMatch/SLAy checkouts into a temporary build context.
- Added `.dockerignore` to keep caches, build outputs, scratch data, and VCS metadata out of Docker contexts.
- Added `tools/axon-recon-container`, a host wrapper that forwards normal `axon-reconstructor` arguments through Docker without parsing pipeline stage selectors.

Acceptance Criteria:
- Normal CLI argument shape is preserved inside the image through the entrypoint and wrapper.
- The wrapper supports dry-run inspection, image override, repo/cache mounts, extra mounts/env vars, and defaults to read-only repo mounts.
- Sibling packages can be installed as normal packages from copied temporary-context paths rather than imported from workspace paths.
- Container smoke assets check CLI help and required imports without requiring local host installs for container-only dependencies.

Self-Check:
- Diff reviewed: yes.
- Unrelated/user edits excluded from commit: yes; only container scaffold files, wrapper, `.dockerignore`, and these notes are modified.
- Instruction files re-read: yes, `debug/pipeline_containerize_instructions.md` and `debug/pipeline_refinement_instructions.md` before this slice.
- Residual risk: the image was not built in this slice; SLAy/UnitMatch dependency resolution and `mpi4py` linkage still need real container and NERSC validation.

Expected To Run:
- `tools/axon-recon-container --image axon-recon:local stages reconstruct --config debug/debug.runtime.yml` should resolve to a Docker run command that calls the same stage selector inside the image.
- `containers/axon-recon/build_local_image.sh --image axon-recon:local` should create a temporary context and include sibling UnitMatch/SLAy paths when present.

Confirmed Not Run:
- Docker build, container runtime smoke, Shifter import, and NERSC jobs were not run in this slice.

Validation:
- Pytest: not run; this slice only adds shell/container assets and a host-side Python smoke helper.
- Smoke: `bash -n containers/axon-recon/build_local_image.sh containers/axon-recon/entrypoint.sh containers/axon-recon/smoke_cli.sh tools/axon-recon-container` passed.
- Smoke: `tools/axon-recon-container --dry-run --no-tty --image axon-recon:local stages reconstruct --config debug/debug.runtime.yml` printed the expected Docker command with repo/cache mounts and forwarded CLI args.
- Smoke: `containers/axon-recon/build_local_image.sh --dry-run --image axon-recon:local` detected sibling UnitMatch/SLAy paths and added the corresponding build args.
- Smoke: `/home/adamm/miniconda3/envs/axon_recon/bin/python containers/axon-recon/smoke_imports.py --allow-missing` ran successfully; host environment lacks expected container-only imports for `UnitMatchPy`, `kilosort`, and `mpi4py`.
- Smoke: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m axon_recon.pipeline.cli --help` and `... stages --help` passed.
- Container build/run: not run; first actual image build remains pending because dependency resolution/image size should be inspected as its own slice.
- Logs inspected: shell validation, dry-run output, host import smoke output.
- Not run: full test suite, Docker build, real pipeline stage execution, Shifter validation.

Container / Shifter Impact:
- Local Docker behavior: adds a Dockerfile and host wrapper for local container execution; wrapper defaults to the configured `docker` CLI.
- Shifter/NERSC behavior: README records CPU/GPU job shapes and leaves CUDA-aware `mpi4py` validation as NERSC-deferred.
- Image size/cache impact: `.dockerignore` and build helper avoid copying VCS metadata, caches, scratch data, and outputs; real image size still unmeasured.

CLI Impact:
- Normal CLI: unchanged.
- Container CLI: new `tools/axon-recon-container` wrapper forwards all remaining args to `axon-reconstructor` inside the image; use `--wrapper-help` for wrapper options.

Resume / Force-Restart Impact:
- Resume behavior: unchanged; wrapper does not inspect or modify runtime config/resume flags.
- Force-restart behavior: unchanged; forwarded to the normal pipeline CLI when provided.

Storage / Mount Impact:
- Created: `.dockerignore`, `containers/axon-recon/*`, `tools/axon-recon-container`.
- Modified: `debug/pipeline_containerize_commit_notes.md`.
- Required mounts: repo/config path is mounted at the same absolute path; extra data/output roots must be supplied with `--mount` or NERSC volume directives.

Rollback Notes:
- Revert the scaffold/wrapper files and this notes entry; no runtime pipeline code is changed in this slice.

## 2026-05-01 14:33 - pending - ai: add local spikeinterface sort backend

Status: accepted

Summary:
- Added a focused `spikesort.core.local_spikeinterface` backend that runs SpikeInterface `run_sorter` in the current Python environment instead of invoking MEA_Analysis or Docker.
- Wired `sort_engine: local_spikeinterface` dispatch through `run_spikesort_stage` while preserving the existing `mea_analysis` route for legacy configs.
- Added tests for Kilosort parameter translation, local sorter/analyzer calls, rejecting container kwargs, and runner dispatch that does not call MEA_Analysis.

Acceptance Criteria:
- Existing active debug runtime still resolves to `mea_analysis` and remains compatible.
- Explicit `local_spikeinterface` runner dispatch calls the new local backend and not the legacy MEA_Analysis route.
- The local backend uses normal installed SpikeInterface imports and `run_sorter` without `docker_image` or other container execution kwargs.
- Local backend returns the same output shape consumed by summarize, bombcell, merge, and runner summary logic.

Self-Check:
- Diff reviewed: yes.
- Unrelated/user edits excluded from commit: yes; only local backend code/tests, runner dispatch, and containerization notes are modified.
- Instruction files re-read: yes, `debug/pipeline_containerize_instructions.md` and `debug/pipeline_refinement_instructions.md`.
- Residual risk: tests use fake SpikeInterface modules; real Kilosort4 execution still needs a data smoke and later container/GPU validation.

Expected To Run:
- Existing `mea_analysis` configs should continue to run through the legacy route.
- New `local_spikeinterface` configs should run SpikeInterface/Kilosort in-process when dependencies and GPU/runtime inputs are available.

Confirmed Not Run:
- Real Kilosort4 sorting, Docker/container build, and Shifter validation were not run in this slice.

Validation:
- Pytest: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/stages/spikesort/tests/test_local_spikeinterface.py src/axon_recon/pipeline/stages/spikesort/tests/test_runner.py src/axon_recon/pipeline/stages/spikesort/tests/test_spikesort_config.py -q` passed.
- Smoke: active `debug/debug.runtime.yml` parsed through `parse_spikesort_stage_config`; output confirmed `mea_analysis`, `kilosort4`, current Docker image, `local_spikeinterface_enabled=False`, and local helper importability.
- Smoke: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m axon_recon.pipeline.cli stages --help` passed.
- Container build/run: not run; container assets are not implemented yet.
- Logs inspected: pytest and smoke command output.
- Not run: full test suite, real data sorting, Docker build, Shifter validation.

Container / Shifter Impact:
- Local Docker behavior: unchanged for `mea_analysis`; local engine rejects container kwargs so it cannot request nested Docker/Singularity through SpikeInterface.
- Shifter/NERSC behavior: adds the in-process sort path needed for future Shifter use, but not yet validated on NERSC/GPU.
- Image size/cache impact: none.

CLI Impact:
- Normal CLI: existing `mea_analysis` behavior preserved; explicit local engine can now dispatch to the new backend.
- Container CLI: unchanged; wrapper not implemented yet.

Resume / Force-Restart Impact:
- Resume behavior: unchanged for legacy `mea_analysis`; local backend has no resume checkpoint beyond output reuse yet.
- Force-restart behavior: local backend forwards `remove_existing_folder` based on force restart and `local_spikeinterface.remove_existing_on_force_restart`.

Storage / Mount Impact:
- Created: `src/axon_recon/pipeline/stages/spikesort/core/` and `test_local_spikeinterface.py`.
- Modified: spikesort runner, runner tests, this notes file.
- Required mounts: local engine writes sorter/analyzer outputs under the configured spikesort stage output root.

Rollback Notes:
- Revert this commit to remove the local SpikeInterface backend and return `local_spikeinterface` to the parser-only guard from the previous commit.

## 2026-05-01 14:27 - pending - ai: parse spikesort sort engines

Status: accepted

Summary:
- Added typed spikesort sort-engine parsing for `mea_analysis` and `local_spikeinterface`.
- Parsed the target sectioned sort layout: `source`, `sorter.name`, `sorter.kilosort`, `local_spikeinterface`, and `mea_analysis`.
- Preserved legacy flat sort config behavior by defaulting missing `engine` to `mea_analysis` and keeping existing Docker/MEA_Analysis fields effective.
- Added a runner guard so explicit `local_spikeinterface` does not silently fall through to the legacy MEA_Analysis/Docker route before the local backend is implemented.

Acceptance Criteria:
- Existing flat `stages.spikesort.phases.sort` YAML keeps resolving to `mea_analysis` with the current sorter, Docker image, and source flags.
- The new sectioned YAML shape parses local SpikeInterface settings and nested Kilosort params.
- Unknown sort engines fail clearly during config parsing.
- Explicit `local_spikeinterface` dispatch does not call the legacy MEA_Analysis route.

Self-Check:
- Diff reviewed: yes.
- Unrelated/user edits excluded from commit: yes; only parser/model/runner tests and containerization notes are modified.
- Instruction files re-read: yes, `debug/pipeline_containerize_instructions.md` and `debug/pipeline_refinement_instructions.md`.
- Residual risk: local SpikeInterface execution is intentionally not implemented in this slice; explicit local engine raises until the next backend slice.

Expected To Run:
- Existing `mea_analysis` spikesort sort configs should run as before.
- New `local_spikeinterface` configs should parse but fail before running legacy sort until the local backend exists.

Confirmed Not Run:
- Real sorting, Docker/container build, and Shifter validation were not run for this parser/dispatch seam.

Validation:
- Pytest: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/stages/spikesort/tests/test_spikesort_config.py src/axon_recon/pipeline/stages/spikesort/tests/test_runner.py -q` passed.
- Smoke: active `debug/debug.runtime.yml` parsed through `parse_spikesort_stage_config`; output confirmed `mea_analysis`, `kilosort4`, current Docker image, and source flags `True False False`.
- Smoke: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m axon_recon.pipeline.cli stages --help` passed.
- Container build/run: not run; no container files exist yet.
- Logs inspected: pytest and smoke command output.
- Not run: full test suite, real data sorting, Docker build, Shifter validation.

Container / Shifter Impact:
- Local Docker behavior: unchanged for existing `mea_analysis` engine configs.
- Shifter/NERSC behavior: explicit `local_spikeinterface` is now a recognized config value but guarded until implemented, preventing accidental nested legacy Docker use.
- Image size/cache impact: none.

CLI Impact:
- Normal CLI: existing config remains compatible; help smoke still works.
- Container CLI: unchanged; wrapper not implemented yet.

Resume / Force-Restart Impact:
- Resume behavior: `mea_analysis.resume_from` can now feed the legacy resume field when using the sectioned layout.
- Force-restart behavior: unchanged for existing legacy route; local backend cleanup knobs parse but are not executed yet.

Storage / Mount Impact:
- Created: none.
- Modified: spikesort config parser, input model, runner guard, parser/runner tests, this notes file.
- Required mounts: none.

Rollback Notes:
- Revert this commit to remove the sort-engine config seam and local-engine guard.

## 2026-05-01 14:18 - pending - ai: add nersc handoff resource notes

Status: accepted

Summary:
- Added explicit handoff context for a future AI agent that may continue the work inside NERSC without access to this chat.
- Clarified that GPU resources are expected only for Kilosort-backed spikesort work; CPU-capable stages should remain runnable on CPU nodes, with high-memory CPU tuning deferred to profiling.
- Split Shifter examples into CPU-stage and GPU-spikesort shapes and warned that `stages all` should request GPU only because it includes spikesort.

Acceptance Criteria:
- Containerization instructions preserve full-pipeline parity while distinguishing image contents from per-stage NERSC resource requests.
- Future agents are told which repo/branch/package/CLI/instruction files matter.
- NERSC guidance states CPU-only selectors should not require GPU module flags.
- NERSC guidance states Kilosort-backed spikesort selectors should request GPU resources.

Self-Check:
- Diff reviewed: yes.
- Unrelated/user edits excluded from commit: yes; keep `debug/debug.runtime.yml` unstaged.
- Instruction files re-read: yes, `debug/pipeline_containerize_instructions.md` and `debug/pipeline_refinement_instructions.md`.
- Residual risk: docs-only change; actual CPU/GPU scheduling behavior still needs implementation and NERSC validation.

Expected To Run:
- Future agents use the new handoff/resource notes when implementing wrappers, Shifter scripts, and validation plans.

Confirmed Not Run:
- No pipeline code, Docker build, or smoke execution is expected from this docs-only change.

Validation:
- Pytest: not run; docs-only change.
- Smoke: not run; docs-only change.
- Container build/run: not run; docs-only change.
- Logs inspected: none.
- Not run: runtime tests, container build, Shifter validation.

Container / Shifter Impact:
- Local Docker behavior: unchanged.
- Shifter/NERSC behavior: documented CPU/GPU resource expectations only; no scripts changed.
- Image size/cache impact: none.

CLI Impact:
- Normal CLI: unchanged.
- Container CLI: unchanged.

Resume / Force-Restart Impact:
- Resume behavior: unchanged.
- Force-restart behavior: unchanged.

Storage / Mount Impact:
- Created: none.
- Modified: `debug/pipeline_containerize_instructions.md`, `debug/pipeline_containerize_commit_notes.md`.
- Required mounts: none.

Rollback Notes:
- Revert the docs commit to remove these NERSC handoff and CPU/GPU resource notes.

## 2026-05-01 14:14 - pending - ai: document containerization operating loop

Status: accepted

Summary:
- Added the containerization operating-loop guardrails Adam requested: read both instruction files before each slice, smoke test whenever possible, self-check before commits, commit every accepted containerization slice after the pass starts, and use this file for containerization commit notes.
- Seeded the containerization commit-notes file with a template modeled after the refinement notes, expanded for Docker/Shifter, mounts, CLI parity, and smoke validation.

Acceptance Criteria:
- Containerization instructions point to `debug/pipeline_containerize_commit_notes.md` for running notes.
- Future containerization work is instructed to re-read both instruction files before each slice.
- Future containerization work is instructed not to edit either instruction file after Adam says to start unless Adam explicitly asks.
- Future containerization commits require acceptance criteria, self-checks, tests, and smoke notes when possible.

Self-Check:
- Diff reviewed: yes.
- Unrelated/user edits excluded from commit: yes; keep `debug/debug.runtime.yml` unstaged.
- Instruction files re-read: yes, `debug/pipeline_containerize_instructions.md` and `debug/pipeline_refinement_instructions.md`; commit-note context also checked in `debug/pipeline_refinement_commit_notes.md`.
- Residual risk: docs-only change; no runtime behavior changed.

Expected To Run:
- Future containerization slices use this file for running notes.

Confirmed Not Run:
- No pipeline code, Docker build, or smoke execution is expected from this docs-only change.

Validation:
- Pytest: not run; docs-only change.
- Smoke: not run; docs-only change.
- Container build/run: not run; docs-only change.
- Logs inspected: none.
- Not run: runtime tests, container build, Shifter validation.

Container / Shifter Impact:
- Local Docker behavior: unchanged.
- Shifter/NERSC behavior: unchanged.
- Image size/cache impact: none.

CLI Impact:
- Normal CLI: unchanged.
- Container CLI: unchanged.

Resume / Force-Restart Impact:
- Resume behavior: unchanged.
- Force-restart behavior: unchanged.

Storage / Mount Impact:
- Created: `debug/pipeline_containerize_commit_notes.md` content.
- Modified: `debug/pipeline_containerize_instructions.md` operating loop.
- Required mounts: none.

Rollback Notes:
- Revert the docs commit to remove these operating-loop and commit-note-template additions.
