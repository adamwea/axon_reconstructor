# Pipeline Containerization Instructions

Draft status: review-only instructions for the next AI-assisted implementation pass. Do not begin these changes until Adam explicitly says to start.

This document extends `debug/pipeline_refinement_instructions.md`. All refinement guardrails still apply unless this file narrows them for container/HPC work.

## Goal

Prepare `axon_reconstructor` to run the active v2 pipeline from a self-contained image that can later be pulled into NERSC Shifter. The container is a full-pipeline execution environment for every supported stage and phase, not a special-purpose image for sorting or reconstruction only.

The local developer-facing contract is:

```bash
python -m axon_recon.pipeline.cli stages reconstruct --config debug/debug.runtime.yml
```

should have a containerized equivalent shaped like:

```bash
axon-recon-container stages reconstruct --config debug/debug.runtime.yml
```

The wrapper must pass through the same stage and phase selectors supported by the normal CLI. It must not special-case only `reconstruct`.

Containerized and non-containerized runs should be behaviorally equivalent for every supported selector, including `stages all`, `preprocess`, `spikesort`, `reconstruct`, multiple stages, and stage-phase selectors such as `reconstruct.analyzers`.

## Handoff Context For Future AI Agents

Adam may move this work into the NERSC environment with a different AI agent. Leave enough context in this repo for that agent to continue without relying on this chat transcript.

Current high-level state:

- Repository: `axon_reconstructor`, branch `pipeline_v2`.
- Active package/import namespace: `axon_recon`.
- Existing console script: `axon-reconstructor` points to `axon_recon.pipeline.cli:main`.
- Planned host wrapper: `axon-recon-container`, which should forward to the installed pipeline CLI inside the container.
- Active stages to preserve: `preprocess`, `spikesort`, and `reconstruct`.
- Container goal: one installable full-pipeline image whose behavior matches the normal CLI; job resource requests may differ by stage.
- First blocker: the current MEA_Analysis sorting path can launch Docker, which cannot be the only path when the pipeline itself runs inside Shifter/container mode.
- Key instruction files: this file, `debug/pipeline_refinement_instructions.md`, `debug/pipeline_containerize_commit_notes.md`, and `debug/pipeline_refinement_commit_notes.md`.

Future agents should keep this document updated with verified facts, unresolved NERSC checks, image tags, build commands, mount assumptions, and smoke commands. Do not assume the next agent can inspect the previous conversation.

## Non-Goals For This First Pass

- Do not require NERSC access to complete the local container work.
- Do not push images to Docker Hub, registry.nersc.gov, or any remote registry unless Adam explicitly asks.
- Do not convert to a Shifter image locally; prepare a Docker-compatible image and scripts/docs that can later be pulled by Shifter at NERSC.
- Do not delete the existing MEA_Analysis sort path yet; keep it selectable for local/backward-compatible runs.
- Do not edit sibling workspaces such as `MEA_Analysis`, `UnitMatch`, `SLAy`, `spikeinterface`, or `axon_velocity` unless Adam explicitly asks.

## External Guidance Already Reviewed

- Base image candidate: `spikeinterface/kilosort4-base:4.0.6_cuda-12.0.0`, digest `sha256:e6c8919a949d96460fd52b91ba092c2d498984670eef548a79e49844d35da0cf`, `linux/amd64`, about 6.21 GB compressed.
- NERSC Shifter images should be compact; images larger than about 20 GB can time out during registry upload/import.
- Shifter images are read-only at runtime and root is squashed. Installed software must be executable by a non-root user and all outputs/caches must go to mounted writable paths.
- Shifter reserves/replaces `/etc`, `/var`, and `/tmp`. Do not rely on persistent image state in those paths; mount writable scratch/cache locations explicitly.
- For multi-node runs, the image is requested with `#SBATCH --image=...`, and `srun shifter ...` launches commands inside it.
- GPU Shifter jobs need the Shifter `gpu` module; CUDA-aware MPI needs `cuda-mpich` plus `MPICH_GPU_SUPPORT_ENABLED=1`.
- NERSC recommends GPU-aware Cray MPICH for `mpi4py` on Perlmutter GPU nodes. Their non-container examples build `mpi4py` from source with `MPICC="cc -shared"`; Shifter examples instead build MPICH in the image so Shifter can swap in Cray MPICH at runtime.
- NERSC warns that `fork()`/subprocess use inside MPI processes can produce undefined MPI behavior. Treat nested subprocess/container launches as incompatible with future MPI rank execution unless proven otherwise.

## NERSC Resource Model

GPU support is only expected to be required for the spikesort stage, specifically the Kilosort-backed `spikesort.sort` phase. Other active stages and phases should remain runnable on CPU nodes unless future profiling proves otherwise.

Rules:

- Use the same container image for CPU and GPU jobs when practical, but do not require GPU resources to launch CPU-only selectors.
- `preprocess`, `reconstruct`, reconstruct template/analyzer/report/plot phases, and non-Kilosort bookkeeping phases should be treated as CPU-capable.
- Some CPU stages may eventually need high-memory CPU nodes or adjusted worker counts. Record those needs after NERSC profiling; do not assume GPU just because a stage is heavy.
- If a selector includes Kilosort-backed `spikesort.sort`, the NERSC job should request GPU resources and the Shifter `gpu` module.
- If a selector excludes Kilosort-backed sorting, the NERSC job should be able to run without GPU module flags.
- `stages all` in a single NERSC job will include spikesort and therefore may need GPU resources for the whole job. Prefer splitting production workflows into CPU preprocess, GPU spikesort, and CPU reconstruct jobs once checkpoint/artifact boundaries are validated.
- The wrapper should fail clearly if a GPU-required selector is launched without visible GPU support, but it should not reject CPU-only selectors on CPU nodes.
- MPI/GPU-aware MPI work should not make CPU-only stages require CUDA-aware MPI.

## Operating Loop

For each containerization slice:

1. Re-read this file, `debug/pipeline_refinement_instructions.md`, `debug/pipeline_containerize_commit_notes.md`, `debug/pipeline_refinement_commit_notes.md`, and the active `debug/debug.runtime.yml` stage and phase config before editing.
2. Work in the `axon_reconstructor` repo only.
3. Make one coherent change at a time: sort-engine selection, local Kilosort path, Dockerfile, wrapper, MPI preparation, docs, or tests.
4. Preserve current CLI behavior for non-container runs.
5. Add tests for every new YAML knob and dispatch path.
6. Validate normal Python execution and containerized execution separately.
7. Smoke test after every change whenever a meaningful smoke is possible. If a smoke is not possible, record why and run the closest focused substitute.
8. Self-check before every commit: inspect `git diff`, confirm unrelated/user edits are not staged, confirm acceptance criteria are met, and record residual risk.
9. Commit every coherent accepted slice with an `ai:` prefix once Adam has started the containerization pass.
10. Update `debug/pipeline_containerize_commit_notes.md` after each containerization commit. Use `debug/pipeline_refinement_commit_notes.md` only for non-container refinement commits.

Instruction-file rule:

- Before Adam says to start containerization, edits to this file are allowed when Adam asks for instruction changes.
- After Adam says to start containerization, do not edit `debug/pipeline_containerize_instructions.md` or `debug/pipeline_refinement_instructions.md` unless Adam explicitly asks.
- During containerization iteration, keep running notes in `debug/pipeline_containerize_commit_notes.md` and re-read both instruction files before each slice.

## Priority Order

1. Preserve full normal CLI parity for every active pipeline stage and phase.
2. Split the spikesort `sort` phase into selectable engines because nested MEA_Analysis/Docker sorting is the first known blocker for running the full pipeline inside one container.
3. Implement a local, in-process SpikeInterface/Kilosort4 sort engine that does not launch a nested Docker container.
4. Keep the MEA_Analysis engine available as an explicit option.
5. Add container build assets using the Kilosort4 CUDA base image.
6. Add the `axon-recon-container` host wrapper and container entrypoint behavior for all current CLI selectors.
7. Add import/runtime validation for `axon_recon`, `spikeinterface`, Kilosort4, patched UnitMatch, SLAy, and `mpi4py`.
8. Prepare MPI/Shifter notes and any code hooks needed for future MPI rank-aware execution.

## Spikesort Engine Split

The current sort phase mixes source selection, Kilosort params, MEA_Analysis params, Docker image settings, plotting, and report settings in one block. Replace that with explicit subsections under `stages.spikesort.phases.sort`.

Target YAML shape:

```yaml
stages:
  spikesort:
    phases:
      sort:
        enabled: true
        engine: local_spikeinterface  # local_spikeinterface|mea_analysis

        source:
          use_bootstrapped_concat_binary: true
          use_lazy_source: false
          assert_one_source: false

        sorter:
          name: kilosort4
          kilosort:
            batch_duration_s: 0.75
            thresholds:
              universal: 8
              learned: 7
              single_ch: 5
            clustering:
              downsampling: 15
            channels:
              nearest: 12
              max_distance: 40

        local_spikeinterface:
          enabled: true
          output_relpath: sorter_output
          remove_existing_on_force_restart: true
          run_sorter_kwargs: {}
          analyzer:
            enabled: true
            output_relpath: analyzer_output

        mea_analysis:
          enabled: true
          docker_image: adammwea/benshalomlab_spikesorter_pythonpatch:v3
          resume_from: null
          plot:
            enabled: true
            mode: separate
            plot_debug: false
            raster_sort: null
            fixed_y: false
          report:
            enabled: true
            no_curation: true
            export_to_phy: false
```

Rules:

- `engine` is the only selector for the implementation used by the `sort` phase.
- `engine: local_spikeinterface` must run Kilosort4 from the current Python environment/container, not by calling Docker or `run_sorter_container`.
- `engine: mea_analysis` may keep using the current MEA_Analysis routine and its Docker image path for local/non-HPC compatibility.
- MEA_Analysis-only knobs must live under `mea_analysis`.
- Local SpikeInterface-only knobs must live under `local_spikeinterface`.
- Sorter algorithm params shared by both engines should live under `sorter` and be mapped deliberately into each backend.
- If a legacy top-level sort knob remains temporarily, parse it only as a migration bridge and log/deprecate it. Do not add new behavior to the legacy layout.
- Tests must prove that each engine receives only its own params and that the unselected engine is not called.

## Local SpikeInterface Sort Engine

The local engine should be a first-class spikesort phase implementation, not a thin wrapper around MEA_Analysis.

Implementation expectations:

- Add a focused orchestrator/core module for local sorting rather than growing `legacy_runner.py`.
- Use normal installed-package imports such as `import spikeinterface as si` and SpikeInterface submodules. Do not import from the sibling `spikeinterface` workspace path.
- Use SpikeInterface sorter APIs that execute inside the current environment. Do not call another Docker container from inside the pipeline container.
- Keep output layout compatible with downstream summarize, bombcell-label, merge, and reconstruct inputs, or update those downstream resolvers with tests.
- Preserve force-restart cleanup semantics for local sort outputs.
- Preserve source selection behavior for bootstrapped concat binaries versus preprocess concat recordings.
- Emit logs that identify `sort_engine`, `sorter.name`, effective `n_jobs`, output dirs, source recording, and whether force-restart cleanup ran.
- Ensure Kilosort parameter names are translated once in config/core code, not scattered through runner logic.

Acceptance tests:

- Parser test for the new YAML structure.
- Sort runner dispatch test for `engine: local_spikeinterface`.
- Sort runner dispatch test for `engine: mea_analysis`.
- Test that `local_spikeinterface` does not call MEA_Analysis or Docker container helpers.
- Test that `mea_analysis` continues to call the existing MEA_Analysis route with isolated params.
- Test that downstream sorter-output resolution can read the local engine output layout.

## Container Assets

Add container files under a dedicated directory, for example:

```text
containers/axon-recon/
  Dockerfile
  README.md
  entrypoint.sh
  smoke_imports.py
  smoke_cli.sh
```

Add a repo-level host wrapper, for example:

```text
tools/axon-recon-container
```

Container build rules:

- Use `spikeinterface/kilosort4-base:4.0.6_cuda-12.0.0` as the initial base image. Prefer pinning the digest once the Dockerfile is stable.
- Keep the image `linux/amd64` compatible for Perlmutter/Shifter.
- Install the active `axon_reconstructor` package into the image as an installable package, not by relying on runtime bind-mounted source imports.
- Install dependencies from `environment.yml` or an explicit container lock/spec derived from it. Record any intentional deviations in the container README.
- Use one Python environment in the image and make it active by `PATH`/entrypoint so users do not need to `conda activate` manually.
- Install patched UnitMatch from the edited workspace package at build time, but make it importable as a normal installed package at runtime. Do not rely on `/home/adamm/dev/pkgs/UnitMatch` being mounted.
- Install SLAy normally with pip as package `slay` unless Adam asks to use the sibling checkout.
- Install SpikeInterface normally through package management, not by importing from the sibling checkout.
- Keep package import names normal in source code. No `sys.path` hacks, workspace-relative imports, or editable sibling imports in runtime code.
- Include `mpi4py` in the image in a way compatible with Shifter's MPICH swapping plan. If this cannot be fully validated before NERSC, document the unresolved validation step clearly.
- Remove package caches, build dirs, and temporary tarballs in the same Docker layer where they are created.
- Do not bake raw data, scratch outputs, caches, credentials, SSH keys, or local absolute config paths into the image.

Suggested local build command shape after the wrapper/build script exists:

```bash
docker build -f containers/axon-recon/Dockerfile -t axon-recon:local .
```

If patched UnitMatch cannot be included from the default repo build context, add a build helper that creates a temporary build context or wheelhouse from the sibling checkout. Keep that helper inside `axon_reconstructor` and avoid modifying UnitMatch itself.

## `axon-recon-container` Wrapper Contract

The wrapper should make containerized execution feel like the normal CLI.

Required behavior:

- `axon-recon-container stages reconstruct --config debug/debug.runtime.yml` forwards arguments unchanged to the pipeline CLI inside the image.
- `axon-recon-container stages all --config debug/debug.runtime.yml`, `axon-recon-container stages preprocess spikesort --config debug/debug.runtime.yml`, and stage-phase selectors forward through the same path.
- The same wrapper supports every stage and phase selector accepted by the normal CLI. New selectors added to the normal CLI should work through the wrapper without wrapper code changes.
- The wrapper mounts the repo/config path read-only by default unless a development mode intentionally mounts source writable.
- The wrapper mounts data roots, scratch roots, and output roots writable based on runtime config or explicit flags.
- The wrapper sets a writable cache/temp location. Do not let Kilosort, Matplotlib, SpikeInterface, or Python caches try to write inside the read-only image.
- The wrapper fails with a clear message if required host paths from the config are not mounted into the container.
- The wrapper exposes a `--image` override so Adam can test different local tags.
- The wrapper has a dry-run/debug mode that prints the resolved container command and mounts without running the pipeline.

Do not make the wrapper parse stage/phase semantics itself. It should only resolve container execution details and pass the remaining args to the installed pipeline CLI.

## Shifter Preparation

This pass should prepare for Shifter without requiring NERSC runtime validation.

Document examples in `containers/axon-recon/README.md` for later NERSC use:

```bash
shifterimg -v pull docker:<registry>/<image>:<tag>
```

and batch-script shape:

CPU-stage example shape:

```bash
#SBATCH --image=docker:<registry>/<image>:<tag>
#SBATCH --constraint=cpu

srun shifter axon-reconstructor stages preprocess reconstruct --config /mounted/path/debug.runtime.yml
```

GPU spikesort example shape:

```bash
#SBATCH --image=docker:<registry>/<image>:<tag>
#SBATCH --constraint=gpu
#SBATCH --module=gpu,cuda-mpich

export MPICH_GPU_SUPPORT_ENABLED=1
srun shifter axon-reconstructor stages spikesort --config /mounted/path/debug.runtime.yml
```

Use these only as example shapes. The same Shifter invocation should accept any selector that the normal CLI accepts. If `stages all` is used as one job, request GPU resources because it includes spikesort; for efficient production runs, prefer stage-split jobs once resume/artifact boundaries are proven.

Guardrails:

- In Slurm scripts, request the image with `#SBATCH --image`; do not rely on passing `--image` only to `shifter` for multi-node jobs.
- Use absolute NERSC paths in `#SBATCH --volume` lines because environment variables are not expanded there.
- Use `/global/cfs/...` for Community File System paths, not `/cfs`.
- Keep large data on mounted filesystems, not inside the image.
- Make all runtime output locations writable mounts.
- Test the Docker image as a non-root user locally before declaring it Shifter-ready.

## MPI And Parallelism Preparation

Do not rewrite the whole pipeline around MPI in the first container slice. Prepare clean seams.

Expected future direction:

- Add an optional runtime execution mode that detects `mpi4py` rank/size when enabled.
- Use MPI ranks to partition independent execution targets before local thread/process fanout.
- Keep existing local `well_workers` and `unit_workers` behavior for non-MPI runs.
- In MPI mode, avoid nested `ProcessPoolExecutor`, `subprocess`, and Docker calls inside ranks unless deliberately tested on NERSC.
- Ensure only rank 0 performs global summary writes unless outputs are rank-scoped or synchronized safely.
- Include rank metadata in logs and summaries when MPI mode is active.
- Add tests with a fake MPI adapter before requiring real `mpirun`/`srun` in CI.

Validation to defer until NERSC:

- Confirm Shifter `cuda-mpich` swaps the image MPI correctly.
- Confirm `MPICH_GPU_SUPPORT_ENABLED=1` is set in the job environment.
- Confirm multi-rank target partitioning works with real Slurm `srun`.
- Confirm Kilosort/CUDA sees the expected GPU per task.

## Validation Matrix

Minimum local validation before Adam reviews a containerization code slice:

- Normal non-container CLI still works for parser/dispatch tests.
- `python -m pytest` focused tests for changed config and runner code pass.
- `docker build` for the local image completes.
- Container smoke import passes for:
  - `axon_recon`
  - `spikeinterface`
  - `kilosort` or the import name required by SpikeInterface Kilosort4
  - patched UnitMatch import path, expected to include `UnitMatchPy` unless verified otherwise
  - `slay`
  - `mpi4py`
- Container CLI help works.
- `axon-recon-container stages --help` or equivalent forwards to the pipeline CLI.
- A dry-run or minimal no-heavy-data command works with mounted `debug/debug.runtime.yml`.
- A CPU-only container selector can be dry-run or smoke-tested without GPU flags.
- If local sorting is touched, run tests proving it does not invoke Docker/MEA_Analysis when `engine: local_spikeinterface` is selected.

Validation that can be documented but not completed before NERSC:

- `shifterimg pull` of the final pushed image.
- `srun shifter ...` on Perlmutter.
- CUDA-aware `mpi4py` with Cray MPICH.
- CPU-node execution for CPU-only selectors.
- GPU-node execution for Kilosort-backed spikesort selectors.
- Multi-node/multi-GPU execution for any active stage or phase that supports distributed execution.

## Risk Register

- Kilosort4 base image is already large. Extra conda/pip dependencies may approach Shifter/Docker Hub practical limits; keep layers small and clean caches.
- SLAy currently declares CUDA/PyTorch-heavy dependencies. Installing it normally may materially increase image size and dependency conflicts.
- UnitMatch packaging may not be standard at the workspace root. Verify the install command and import path before baking it into the Dockerfile.
- The current MEA_Analysis sort path can launch Docker. That is incompatible with Shifter/local container nesting and should be guarded when running in container mode.
- `mpi4py` compatibility cannot be fully proven off NERSC. Keep the implementation explicit about what is locally validated versus NERSC-deferred.
- Shifter images are read-only and root-squashed. Hidden writes to package directories, home directories, `/tmp`, or matplotlib/cache dirs will fail unless redirected.
- MPI plus Python subprocess/fork behavior is risky on Perlmutter. Favor rank-level target partitioning over process spawning inside ranks.
- Do not over-request GPUs for CPU-only stages by default; GPU allocation should follow selector requirements, not image contents.

## Stop Conditions

Pause and ask Adam before continuing if:

- The Kilosort4 base image cannot support the needed Python/CUDA package set.
- Local SpikeInterface Kilosort4 requires a nested container to work.
- UnitMatch cannot be installed as a normal package without modifying its repo.
- The image is likely to exceed practical Shifter import size.
- A required NERSC-only behavior blocks further local progress.
- A proposed fix requires editing sibling repositories.
