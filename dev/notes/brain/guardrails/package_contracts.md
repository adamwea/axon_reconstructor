# Package contracts guardrail

## Contract

axon_recon's sibling packages — `SLAy`, `UnitMatchPy`, and (planned) `kssynth`, `unitlink` — each live as **independent git repos at `~/dev/pkgs/<package>/`**, NOT as submodules, NOT vendored into `axon_recon/external/`. The container build script (`containers/axon-recon/build_local_image.sh`) stages siblings into the build context at image-build time; nothing else assumes a particular on-disk layout in `axon_recon` itself.

Each shared package stays **axon_recon-agnostic**: takes generic inputs (SpikeInterface analyzers, KS folders, sorter_output paths), produces generic outputs (SI-convention-compliant folders, tidy tabular files). No axon_recon dataclasses, no axon_recon YAML schemas, no Maxwell-HD-MEA-specific assumptions baked into core code.

Outputs that consumers read MUST be loadable via SpikeInterface's standard reader functions (`si.read_sorter_folder`, `si.extractors.read_kilosort`, etc.). If a downstream consumer can't load the output via SI, the package is broken.

## Why

The R&D iteration loop is fast: we patch SLAy weekly, we expect to patch `kssynth` / `unitlink` similarly. Submodules add 5 git steps per patch (edit → commit submodule → push submodule → cd up → bump SHA → commit parent → push parent) where sibling repos add 2 (edit → commit → push). Vendoring would make upstream contributions impossible. Both alternatives lose features we use.

The "axon_recon-agnostic" rule matters because:
- We want other labs to use these packages (`kssynth` and `unitlink` especially are general SI utilities)
- We want to push patches upstream to the source repos (`SLAy`, `UnitMatchPy`) without first untangling axon_recon-specific code from them
- Tests are simpler when the package's API surface is "give me generic inputs, get generic outputs"

See `dev/notes/plans/active/ks_synthesizer_package_plan.md` §1, `unitmatch_runner_package_plan.md` §1 for the full reasoning, and the user memory note `reference-nersc-shifter-bindmounts` for the bind-mount allowlist constraint that drives the sibling-overlay-vs-not decision on NERSC.

## Concrete sub-rules

1. **No `import axon_recon` in any sibling package.** Period. If you find yourself wanting to share a utility between axon_recon and a sibling, the utility belongs in the sibling (and axon_recon imports it from there), not the reverse.

2. **No axon_recon YAML schema awareness in sibling packages.** A sibling consumes Python-level config dataclasses or kwargs, not YAML dicts.

3. **SI-convention output**: sibling packages that produce sorter_output-style folders must emit `spikeinterface_log.json`, `params.py`, and the standard `cluster_*.tsv` files in coherent state (every cluster_id in `spike_clusters.npy` must have a row in every `cluster_*.tsv`). The kssynth plan §6 enumerates the full file list.

4. **No bind-mount workarounds for sibling source.** Per `memory/notes` from earlier session: do NOT propose `--volume=<src>:<dst>` bind mounts of host source code into the shifter container as a way to bypass `shifterimg pull` issues. The canonical path is `rebuild_shifter.sh → docker.io push → shifterimg pull`. Wait, retry, or escalate to NERSC. NEVER overlay source from `/pscratch` or `/global/cfs` as a "fast iteration" hack.

5. **License**: all shared packages use MIT (matching `SLAy` and `UnitMatchPy`). Don't change licenses without explicit user approval.

6. **PyPI publication is downstream of stabilization.** Don't publish v0.x packages to PyPI until the API has shipped a few real-user-feedback iterations. The sibling-checkout pattern works without PyPI for axon_recon's own use.

## Sibling repos as of 2026-05-18

| Package | Path | Role | Status |
|---|---|---|---|
| `axon_recon` | `~/dev/pkgs/axon_recon/` | This repo. Orchestrates the pipeline. | Active |
| `SLAy` | `~/dev/pkgs/SLAy/` | Spike-sort merge tool. Patched twice this week (assert relax, aux-tsv sync). | Active |
| `UnitMatch` (containing `UnitMatchPy`) | `~/dev/pkgs/UnitMatch/UnitMatchPy/` | Cross-session unit matching. Mostly read-only consumer for us. | Active |
| `kssynth` | `~/dev/pkgs/kssynth/` (planned) | Synthetic sorter_output builder from analyzers. See `plans/active/ks_synthesizer_package_plan.md`. | Planned |
| `unitlink` | `~/dev/pkgs/unitlink/` (planned) | UnitMatch / DeepUnitMatch wrapper. See `plans/active/unitmatch_runner_package_plan.md`. | Planned |
| `axon_velocity` | `~/dev/pkgs/axon_velocity/` | GTR generation algorithm. Active sibling. | Active |
| Others (`MEA_Analysis`, `axon_reconstructor`, `RBS_network_models`, etc.) | `~/dev/pkgs/*/` | Various lab tooling. Not all on this pipeline's critical path. | Mixed |

## Tests / verification

- A sibling package's tests run against pip-installed-or-editable that package in isolation, NOT against a checkout that happens to also have axon_recon nearby. If a sibling test imports axon_recon, it's a bug.
- Smoke-test trigger for any sibling change: after `bash containers/axon-recon/rebuild_shifter.sh`, verify the sibling's code is live in the shifter image (`stat -c %y` on the installed file inside shifter) and run a 1-well smoke test of the affected pipeline path.
- Build script test: `build_local_image.sh --help` works without error; the default lookup for siblings (`../UnitMatch/UnitMatchPy`, `../SLAy`) resolves to existing paths.

## Open exceptions / follow-ups

- `kssynth` and `unitlink` don't exist yet — their plans (`plans/active/ks_synthesizer_package_plan.md`, `plans/active/unitmatch_runner_package_plan.md`) scope the new repos. Slice 1 of each creates the repo with the contracts in this guardrail in mind.
- The bombcell pass2 KS-extractor inner-join bug (`roadmap.md` §"Post-templates bombcell + SLAy pass") is a SLAy contract violation by another tool; `kssynth` is the structural fix.
- The build_local_image.sh script currently has the sibling list hardcoded for `UnitMatchPy` + `SLAy`. When `kssynth` and `unitlink` ship, the script grows two more sibling entries — small, mechanical, per their respective integration slices.
