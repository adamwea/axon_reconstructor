# Agent Guardrails Commit Notes

Living review log for AI-assisted work governed by the guardrail documents in this directory.

The guardrail documents are:

- `debug/cli_debug_flags_agent_guardrails.md`
- `debug/logging_agent_guardrails.md`
- `debug/parallelism_agent_guardrails.md`
- `debug/container_mpi4py_NERSC_optimization_guardrails.md`
- `debug/stage_and_phase_behavior_guardrails.md`
- `debug/optimization_simplificaiton_guardrails.md`
- `debug/first_version_pipeline_guardrails.md` temporary first-version cleanup guardrail

Once agentic development begins, treat those guardrail documents as locked. Use this file for mutable running notes unless Adam explicitly asks to change a guardrail document.

## How To Use

After each AI commit, append a new entry at the top of `Commit Log` or directly below the most recent entry.

Every entry should include:

- what changed
- why it changed
- guardrail documents consulted
- acceptance criteria used
- focused tests run
- real-data smoke command and scope
- logs and artifacts inspected
- what was expected to run and confirmed not to run
- storage/cache impact
- CLI/debug-flag impact
- logging/parallelism impact when relevant
- container/NERSC/MPI impact when relevant
- residual risk, follow-ups, and rollback notes

Commits should be frequent, coherent, and prefixed with `ai:`. Do not push unless Adam explicitly asks.

## Entry Template

```markdown
## YYYY-MM-DD HH:MM - <short_sha> - ai: <commit subject>

Status: accepted | needs follow-up | reverted

Summary:
-

Guardrails Consulted:
-

Acceptance Criteria:
-

Expected To Run:
-

Confirmed Not Run:
-

Validation:
- Focused tests:
- Real-data smoke:
- Logs inspected:
- Artifacts inspected:
- Not run:

CLI / Debug Flag Impact:
-

Logging / Parallelism Impact:
-

Storage / Cache Impact:
- Created:
- Modified:
- Removed:

Container / NERSC / MPI Impact:
-

Resume / Force-Restart Impact:
-

Residual Risk And Follow-Ups:
-

Rollback Notes:
-
```

## Commit Log

## 2026-05-03 - pending - ai: add agent guardrail documents

Status: accepted

Summary:
- Added locked guardrail documents for CLI debug flags, logging, parallelism, container/mpi4py/NERSC preparation, stage/phase behavior, and optimization/simplification.
- Added a separate temporary first-version pipeline guardrail for minimizing active YAML knobs, removing undesired fallback code, deleting unused aliases, and eliminating unused legacy code before rollout.
- Added this mutable commit-notes file for future AI implementation slices.

Guardrails Consulted:
- Source notes under `debug/ai_notes`.
- Existing refinement and container commit-note templates.
- Repo memories for debug limits, console/progress logging, resource telemetry, and process lifecycle.

Acceptance Criteria:
- Each requested guardrail file exists in `debug/`.
- Each guardrail emphasizes frequent `ai:` commits, real-data smoke tests, CLI debug flags, limited smoke scope, expanded scope for logging/parallelism, acceptance criteria, and locked-doc treatment.
- A separate first-version pipeline guardrail exists because those cleanup priorities may be less true after rollout.
- A separate commit-notes Markdown file exists in `debug/`.

Validation:
- Focused tests: not run; docs-only change.
- Real-data smoke: not run; docs-only change.
- Logs inspected: not applicable.
- Artifacts inspected: created Markdown files.

Residual Risk And Follow-Ups:
- Future implementation passes should update this file after every AI commit and leave the guardrail documents locked unless Adam asks for changes.
