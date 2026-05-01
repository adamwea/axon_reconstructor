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
