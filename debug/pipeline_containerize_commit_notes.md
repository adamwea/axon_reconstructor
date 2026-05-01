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

## 2026-05-01 16:00 - pending - ai: add auto-building container wrapper

Status: accepted

Summary:
- Replaced the large Bash-only host wrapper with an installable `axon-recon-container` console script backed by `axon_recon.pipeline.container_cli`.
- Kept `tools/axon-recon-container` as a thin source-tree launcher into the same Python implementation so installed and repo-local wrapper behavior cannot drift.
- Made the wrapper build/update the default `axon-recon:local` image when it is missing or its source fingerprint label is stale; added `--build`, `--no-build`, and `--rebuild` controls.
- Preserved wrapper pass-through semantics for all pipeline commands and selectors; the wrapper handles only image, mount, cache, user, and Docker execution concerns.
- Added shared stage parser flags for `--limit-segments`, `--limit-datasets`, `--limit-wells-per-dataset`, and `--limit-units` so the requested real-data smoke command shapes parse consistently.
- Applied dataset and wells-per-dataset overrides at runtime target selection for preprocess, spikesort, and reconstruct; mapped preprocess `--limit-segments` to the existing per-well segment throttle, spikesort `--limit-segments` to the `bootstrap_concat_binary` source segment manifest before binary materialization, and reconstruct `--limit-segments` to the established analyzer/template segment limit path.
- Documented the simple wrapper UX, real-data smoke commands, DockerHub tags, and the environment strategy.
- Compared Adam's live host `axon_recon` conda env with the built container: the container uses the base image's existing `/home/miniconda3` conda stack plus explicit repo/runtime specs rather than copying the host env verbatim.

Acceptance Criteria:
- `axon-recon-container stages reconstruct --config debug/debug.runtime.yml` is the expected user-facing shape.
- Container wrapper behavior remains command-agnostic and does not whitelist only the smoke examples.
- Requested smoke flags parse through the shared stage parser and affect runtime target/unit/segment limits where the stage has those concepts; spikesort segment limiting happens before bootstrap concat writes the binary used by downstream sort phases.
- Active debug runtime defaults to `local_spikeinterface` sorting for container compatibility.
- Docs/instructions explain the wrapper contract, smoke commands, DockerHub tags, and conda environment decision for future NERSC handoff.

Self-Check:
- Diff reviewed: yes.
- Unrelated/user edits excluded from commit: Dockerfile formatting churn was removed from the diff before validation.
- Instruction files re-read: yes, `debug/pipeline_containerize_instructions.md` and the container README were re-read/updated during this slice.
- Segment-limit follow-up: `--limit-segments` now applies to spikesort by limiting `spikesort.bootstrap_concat_binary` before the bootstrapped concatenated binary is materialized.

Expected To Run:
- Installed package command `axon-recon-container ...` should behave the same as `tools/axon-recon-container ...`.
- Any pipeline CLI selector accepted by `axon-reconstructor` should pass through the wrapper unchanged when it appears after wrapper options.
- Default local image builds/updates through `containers/axon-recon/build_local_image.sh` with an image source-fingerprint label.

Confirmed Not Run:
- Real data preprocess/spikesort/reconstruct, local Docker image rebuild from this exact source, DockerHub push, Shifter, Slurm, GPU Kilosort execution, and MPI execution were not run in this slice.

Validation:
- Pytest: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/tests/test_cli_stage_sequence.py src/axon_recon/pipeline/tests/test_container_cli.py -q` passed.
- Syntax: Pylance syntax checks passed for `container_cli.py`, `cli.py`, `runner.py`, and `test_container_cli.py`.
- Runtime parse smoke: active `debug/debug.runtime.yml` resolves `sort_engine=local_spikeinterface`, `local_spikeinterface_enabled=True`, and `mea_analysis_enabled=False`.
- Smoke: `bash -n tools/axon-recon-container containers/axon-recon/build_local_image.sh containers/axon-recon/entrypoint.sh containers/axon-recon/smoke_cli.sh` passed.
- Wrapper dry-run: preprocess, spikesort, and reconstruct real-data smoke command shapes resolved to Docker commands with config-derived output/scratch/raw-data mounts.
- Wrapper dry-run: arbitrary pass-through form `axon-recon-container axon-reconstructor stages all --config debug/debug.runtime.yml` resolved without wrapper stage parsing.
- Diff hygiene: `git diff --check` passed after the wrapper/path changes.

Container / Shifter Impact:
- Local Docker behavior: default local image now self-updates from the source tree when the source fingerprint label is missing or stale; `--no-build` preserves no-build dry-run/manual image behavior.
- Shifter/NERSC behavior: no Shifter runtime change, but docs now identify the DockerHub tags and the local wrapper contract future NERSC jobs should mirror with `srun shifter axon-reconstructor ...`.
- Image size/cache impact: no image rebuild in this slice; source fingerprint labels are added only when the wrapper invokes the build helper.

CLI Impact:
- Normal CLI: shared `stages`/`stage` parser now accepts `--limit-datasets` and `--limit-wells-per-dataset` in addition to existing segment/unit limits.
- Container CLI: `axon-recon-container` is an installed console script and the repo-local tool is a thin launcher; all forwarded args remain unchanged after wrapper options.

Resume / Force-Restart Impact:
- Resume behavior: unchanged except smoke-size target selection can now be reduced from CLI flags.
- Force-restart behavior: unchanged.

Storage / Mount Impact:
- Created: no new runtime storage paths beyond the existing wrapper cache directory behavior.
- Modified: wrapper still mounts the repo read-only by default and auto-mounts config-derived raw/data/scratch roots.
- Required mounts: unchanged for normal config-based runs.

Rollback Notes:
- Revert the console script/module plus `tools/axon-recon-container` launcher changes to return to the previous Bash-only wrapper. Keep the CLI limit flag changes independently if smoke-size runs remain desired outside the container.

## 2026-05-01 15:55 - pending - ai: guard MEA sort inside container

Status: accepted

Summary:
- Marked axon_recon images with `AXON_RECON_IN_CONTAINER=1`.
- Added a spikesort runner guard that blocks `sort.engine: mea_analysis` inside the axon_recon container because the legacy MEA_Analysis path can launch nested Docker.
- Kept the host/local MEA_Analysis route available and added `AXON_RECON_ALLOW_CONTAINER_MEA_ANALYSIS=1` as an explicit override for intentional nested-container debugging.
- Documented that container/HPC runs should use `engine: local_spikeinterface`.

Acceptance Criteria:
- Host behavior remains backward-compatible when `AXON_RECON_IN_CONTAINER` is unset.
- In-container MEA_Analysis sorting fails before calling the legacy runner unless the explicit override is set.
- Local SpikeInterface dispatch remains unaffected.
- Container smoke still passes after the image marker is added.

Self-Check:
- Diff reviewed: yes.
- Unrelated/user edits excluded from commit: yes; only the Dockerfile, spikesort runner/test, container README, and these notes are modified.
- Instruction files re-read: yes, `debug/pipeline_containerize_instructions.md` and `debug/pipeline_refinement_instructions.md` before this slice.
- Residual risk: the active debug runtime still defaults to `mea_analysis` for local compatibility, so containerized data runs using that config must switch the sort engine to `local_spikeinterface` or set the explicit override.

Expected To Run:
- Host `mea_analysis` dispatch remains available when not in the container.
- Container `mea_analysis` dispatch raises a clear error before legacy MEA_Analysis can launch nested Docker.
- Container `local_spikeinterface` dispatch remains the intended HPC path.

Confirmed Not Run:
- Real data sorting, native CUDA/GPU Kilosort execution, Shifter, Slurm, and MPI execution were not run in this slice.

Validation:
- Pytest: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/stages/spikesort/tests/test_runner.py -q -k 'local_engine or rejects_mea_analysis_inside_container or summarize_sort_writes_summary'` passed.
- Runtime parse smoke: active `debug/debug.runtime.yml` still resolves `sort_engine=mea_analysis` for local compatibility.
- Smoke: `bash -n containers/axon-recon/build_local_image.sh containers/axon-recon/entrypoint.sh containers/axon-recon/smoke_cli.sh tools/axon-recon-container` passed.
- Container build/run: rebuilt full sibling image as `axon-recon:full-probe` in `40.3s`.
- Container smoke: `docker run --rm axon-recon:full-probe axon-recon-smoke-cli` passed.
- Container smoke: `docker run --rm --entrypoint python axon-recon:full-probe -c '...'` confirmed `AXON_RECON_IN_CONTAINER=1` and `_mea_analysis_allowed_in_container()` is `False` by default.
- Container smoke: setting `AXON_RECON_ALLOW_CONTAINER_MEA_ANALYSIS=1` made `_mea_analysis_allowed_in_container()` return `True`.
- Container size: `docker image inspect axon-recon:full-probe` reported about `9.94 GB`.
- Diff hygiene: `git diff --check` passed.

Container / Shifter Impact:
- Local Docker behavior: legacy MEA_Analysis sorting is now blocked inside the image by default to prevent nested Docker surprises.
- Shifter/NERSC behavior: prevents a known Shifter-incompatible sort path and points users toward the in-process SpikeInterface engine.
- Image size/cache impact: no meaningful size change; full probe image remained about `9.94 GB`.

CLI Impact:
- Normal CLI: unchanged outside the container.
- Container CLI: `mea_analysis` sort engine now requires explicit `AXON_RECON_ALLOW_CONTAINER_MEA_ANALYSIS=1`; `local_spikeinterface` remains the intended container sort engine.

Resume / Force-Restart Impact:
- Resume behavior: unchanged for allowed engines.
- Force-restart behavior: unchanged for allowed engines.

Storage / Mount Impact:
- Created: none.
- Modified: no runtime storage layout changes.
- Required mounts: unchanged.

Rollback Notes:
- Revert this commit to allow the container image to run the legacy MEA_Analysis route without an explicit override, with the known risk of nested Docker attempts inside Shifter/container mode.

## 2026-05-01 15:38 - pending - ai: auto-mount runtime config paths

Status: accepted

Summary:
- Added config-derived mount discovery to `tools/axon-recon-container` so normal `--config debug/debug.runtime.yml` invocations automatically mount the runtime config's output root, scratch root, and raw H5 data root.
- Kept the mount scanner lightweight and PyYAML-free; it follows the runtime `data:` pointer and recognizes scalar YAML keys used by the active pipeline config, including list-item `- raw_data_h5_path:` entries.
- Added `--no-config-mounts` and `AXON_RECON_CONTAINER_CONFIG_MOUNTS=0` escape hatches so manual `--mount` behavior remains available.
- Documented the automatic mount behavior in the container README.

Acceptance Criteria:
- Wrapper dry-run with the active runtime config shows configured output and scratch roots mounted read-write.
- Wrapper dry-run with the active runtime config shows the common raw H5 root mounted read-only.
- `--no-config-mounts` suppresses config-derived data/output/scratch mounts.
- Existing explicit `--mount` and container argument forwarding behavior remains available.

Self-Check:
- Diff reviewed: yes.
- Unrelated/user edits excluded from commit: yes; only the host wrapper, container README, and these notes are modified.
- Instruction files re-read: yes, `debug/pipeline_containerize_instructions.md` and `debug/pipeline_refinement_instructions.md` were re-read before continuing this container wrapper slice.
- Residual risk: the scanner is deliberately narrow and validates the current YAML shape without a PyYAML dependency; unusual YAML constructs or future schema changes may require extending it.

Expected To Run:
- `tools/axon-recon-container --image axon-recon:full-probe --current-user --no-tty --dry-run stages preprocess --config debug/debug.runtime.yml` should include the data/output/scratch mounts.
- `tools/axon-recon-container --image axon-recon:full-probe --current-user --no-config-mounts --no-tty --dry-run stages preprocess --config debug/debug.runtime.yml` should omit config-derived mounts.

Confirmed Not Run:
- Real data stages, Docker build, Shifter, Slurm, GPU execution, and MPI execution were not run in this slice.

Validation:
- Pytest: not run; this slice only changes the Bash host wrapper, docs, and notes.
- Smoke: initial dry-run showed output/scratch mounts but missed raw paths; scanner was corrected to handle list-item `- raw_data_h5_path:` syntax.
- Smoke: `bash -n tools/axon-recon-container` passed.
- Smoke: wrapper dry-run with `--config debug/debug.runtime.yml` included `/mnt/ben-shalom_nas/analysis/Media_Density_T5_02182026_AR_axon_analysis_AW:rw`, `/mnt/disk15tb/adamm/scratch:rw`, and the common active raw-data root under `/mnt/ben-shalom_nas/raw_data/.../Media_Density_T5_02182026_AR:ro`.
- Smoke: wrapper dry-run with `--no-config-mounts` omitted the raw-data config mount.
- Diff hygiene: `git diff --check` passed.

Container / Shifter Impact:
- Local Docker behavior: users no longer need to manually specify active runtime data/output/scratch mounts for normal config-based runs.
- Shifter/NERSC behavior: documents the intended mount set for later Slurm/Shifter volume translation but does not validate Shifter itself.
- Image size/cache impact: no image changes and no build required.

CLI Impact:
- Normal CLI: no change to host `axon-reconstructor` behavior.
- Container CLI: wrapper still forwards all selectors unchanged; it now inspects forwarded `--config` only to construct Docker mounts.

Resume / Force-Restart Impact:
- Resume behavior: unchanged, but resumed container runs now get the same configured roots mounted by default.
- Force-restart behavior: unchanged, but force-restart writes now land on auto-mounted scratch/output roots when configured.

Storage / Mount Impact:
- Created: writable output/scratch roots may be created by the wrapper if absent.
- Modified: Docker command receives config-derived `-v` mounts before any explicit user mounts.
- Required mounts: raw H5 common root is mounted read-only; output and scratch roots are mounted read-write.

Rollback Notes:
- Revert this commit to return to explicit-only `--mount` behavior. Users can also pass `--no-config-mounts` to bypass auto mounts without reverting.

## 2026-05-01 15:21 - pending - ai: support non-root container runs

Status: accepted

Summary:
- Fixed a Shifter-style/root-squash readiness bug where the image preserved a root-owned `/tmp/axon-recon-cache` directory from build time, causing non-root container runs to fail before the CLI started.
- Added entrypoint checks that create and verify writable cache/HOME directories before forwarding to either helper commands or `axon-reconstructor`.
- Added wrapper support for `--current-user`, `--user UID:GID`, and `AXON_RECON_CONTAINER_USER` so local Docker smoke tests can run with the same UID/GID that will own host outputs.
- Documented current-user wrapper usage in the container README.

Acceptance Criteria:
- Direct Docker execution with `--user $(id -u):$(id -g)` can run the full container smoke helper.
- Wrapper execution with `--current-user` can forward normal CLI selectors through the container.
- Default/root container smoke remains working.
- Runtime cache paths resolve to writable mounted storage rather than relying on image-owned directories.

Self-Check:
- Diff reviewed: yes.
- Unrelated/user edits excluded from commit: yes; only container runtime/cache/wrapper docs and these notes are modified.
- Instruction files re-read: yes, `debug/pipeline_containerize_instructions.md`, `debug/pipeline_refinement_instructions.md`, current container notes, current refinement notes, and the active spikesort runtime block before this slice.
- Residual risk: this validates local Docker non-root behavior, not NERSC Shifter root squashing itself. Real Shifter runtime still needs NERSC-side validation.

Expected To Run:
- `docker run --rm --user "$(id -u):$(id -g)" axon-recon:full-probe axon-recon-smoke-cli` should pass.
- `tools/axon-recon-container --image axon-recon:full-probe --current-user --no-tty stages --help` should pass.
- Default `docker run --rm axon-recon:full-probe axon-recon-smoke-cli` should still pass.

Confirmed Not Run:
- Real data stages, Shifter, Slurm, GPU execution, and MPI execution were not run in this slice.

Validation:
- Pytest: not run; this slice only changes container entrypoint/cache behavior, wrapper options, docs, and notes.
- Smoke: precheck `docker run --rm --user "$(id -u):$(id -g)" axon-recon:full-probe axon-recon-smoke-cli` failed before the fix with permission denied creating `/tmp/axon-recon-cache/*`, confirming the root-owned cache bug.
- Smoke: `bash -n containers/axon-recon/build_local_image.sh containers/axon-recon/entrypoint.sh containers/axon-recon/smoke_cli.sh tools/axon-recon-container` passed.
- Smoke: wrapper dry-run with `--current-user` showed Docker `--user 1010:1010`, a writable cache mount at `/tmp/axon-recon-cache`, and HOME/cache env vars pointing inside that mount.
- Container build/run: rebuilt full sibling image as `axon-recon:full-probe` in `38.3s` after the cache fix.
- Container smoke: direct non-root `axon-recon-smoke-cli` passed and confirmed `axon_recon 0.1.0`, `kilosort 4`, `mpi4py 4.1.1`, `spikeinterface 0.103.2`, UnitMatchPy distribution `3.3.0`, and SLAy distribution `0.1.0` via the pipeline import path.
- Container smoke: `tools/axon-recon-container --image axon-recon:full-probe --current-user --no-tty stages --help` passed.
- Container smoke: default/root `docker run --rm axon-recon:full-probe axon-recon-smoke-cli` passed.
- Container size: `docker image inspect axon-recon:full-probe` reported about `9.94 GB`.
- Logs inspected: failing non-root precheck, rebuilt Docker output, direct non-root smoke JSON, wrapper smoke output, default smoke output, image size output.
- Diff hygiene: `git diff --check` passed.

Container / Shifter Impact:
- Local Docker behavior: wrapper can now opt into non-root UID/GID execution and defaults HOME/cache paths to the writable cache mount.
- Shifter/NERSC behavior: improves readiness for read-only/root-squashed image execution by avoiding baked writable cache assumptions.
- Image size/cache impact: removing the build-time cache root from the final image reduced the full probe image to about `9.94 GB`.

CLI Impact:
- Normal CLI: no change to host CLI behavior.
- Container CLI: no selector changes; wrapper gains `--current-user`, `--user`, and `AXON_RECON_CONTAINER_USER` controls.

Resume / Force-Restart Impact:
- Resume behavior: unchanged.
- Force-restart behavior: unchanged.

Storage / Mount Impact:
- Created: wrapper creates `home`, `xdg`, `matplotlib`, `numba`, and `pycache` subdirectories under the configured host cache dir.
- Modified: runtime HOME/cache env vars now explicitly point into `/tmp/axon-recon-cache`.
- Required mounts: unchanged for pipeline data; non-root wrapper runs require the cache mount to be writable by the selected UID/GID.

Rollback Notes:
- Revert this commit to remove non-root wrapper controls and return to image-default cache directory behavior. Non-root Docker smoke will again fail if `/tmp/axon-recon-cache` is root-owned inside the image.

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
