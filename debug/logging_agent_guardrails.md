# Logging Agent Guardrails

Status: guardrail document. Once agentic development begins, treat this file as locked. Do not edit it unless Adam explicitly asks for guardrail changes.

This document defines logging expectations for the active `axon_recon` pipeline: terminal logs, file logs, JSONL events, progress bars, summaries, worker errors, and resource usage. Logging changes must be validated with real data, not only unit tests.

## Operating Contract

- Commit frequently after each coherent accepted slice, using an `ai:` prefix in the commit subject.
- Update `debug/agent_guardrails_commit_notes.md` after every AI commit.
- Test frequently with real-data smoke runs using CLI debug flags.
- Use the smallest useful scope by default: 1 dataset, 1 well, 2 segments, and a few units.
- When terminal visibility, progress behavior, multiprocessing, or worker interleaving is in question, expand to multiple datasets and/or wells.
- Full-scope runs are not required for logging validation unless Adam explicitly requests one.

## Logging Goals

The logging system should make it obvious what ran, what skipped, what failed, what resources were used, and where artifacts landed.

Required outputs:

- `logs/pipeline.log`: plain chronological human-readable run log
- `logs/pipeline.jsonl`: canonical structured event stream
- `logs/errors.log`: warnings, errors, exceptions, and tracebacks
- `logs/summary.json`: run, stage, phase, dataset, recording, and well summary
- Optional filtered logs under `logs/datasets/...`, `recordings/...`, `wells/...`, and `phases/...` when enabled

Terminal output is a curated live view. It must never be the only source of truth.

## Context Contract

Every log record should carry available context through logging context helpers, not hand-built string prefixes:

- `run_id`
- `dataset_id`
- `recording_id`
- `chip_id`
- `assay`
- `date`
- `well_id`
- `stage`
- `phase`
- `event`
- `pid`
- `worker`
- elapsed time when useful
- artifact paths when created, reused, deleted, or skipped
- resource usage when available

Required lifecycle events:

- `run_started`, `run_completed`, `run_failed`
- `dataset_started`, `dataset_completed`, `dataset_failed`
- `recording_started`, `recording_completed`, `recording_failed`
- `well_started`, `well_completed`, `well_failed`, `well_skipped`
- `stage_started`, `stage_completed`, `stage_failed`, `stage_skipped`
- `phase_started`, `phase_completed`, `phase_failed`, `phase_skipped`
- `phase_resource_usage` for phases with resource monitoring enabled

Completion events must include elapsed time. Skips must include a reason. Failures must include exception type, message, traceback in file logs, and target context.

## Terminal And Progress Guardrails

- Keep Rich/tqdm output readable and progress-aware.
- Do not solve terminal output by replacing structured logging with `print()` calls.
- Do not use process-global `contextlib.redirect_stdout` or `redirect_stderr` inside worker threads or worker callbacks. Those redirects mutate global `sys.stdout` and `sys.stderr` and can swallow unrelated terminal logs from other active workers.
- If noisy third-party stdout/stderr must be suppressed, prefer library-specific knobs, subprocess capture, file capture, or main-thread-only suppression with explicit tests.
- When debugging terminal behavior, validate with a real TTY capture, for example `script`, because non-TTY test captures can miss Rich/tqdm failures.
- Logs must remain visible while progress bars are active. In particular, phase completion and phase resource-usage logs must not disappear behind progress rendering.

## File And Structured Log Guardrails

- `pipeline.jsonl` is the canonical machine-readable event stream.
- Disabling physical per-well or per-phase logs must not remove context from `pipeline.jsonl`.
- File logs must be plain text or JSONL, not Rich markup.
- Logging setup must initialize once and avoid duplicate handlers when entry points are invoked repeatedly in one process.
- Worker exceptions must route to `errors.log`, `pipeline.log`, and the relevant filtered log when enabled.
- Resource usage should never crash the pipeline. Missing metrics should be logged as unavailable, not treated as pipeline failures.

## Resource Usage Logging

Phase resource logs must distinguish planned pipeline resources from raw process observations.

Required semantics:

- `max_threads` means pipeline-declared current-process concurrency.
- `observed_process_max_threads` preserves raw psutil/native-library thread counts for ambient pools such as OpenBLAS.
- Resource warning logic should use pipeline-owned counts for pipeline planning warnings.
- Raw observations should remain visible for debugging unexpected native-library behavior.

## Stage Logging Expectations

At normal verbosity, terminal logs should show:

- run start/completion
- stage start/completion
- phase start/completion for major phases
- per-well start/completion for long-running well jobs
- warnings/errors
- progress bars
- final summary

At detailed verbosity, logs should add:

- all phase boundaries
- key input and output paths
- cache hits/misses
- applied debug flags
- resource usage at phase boundaries
- external command summaries

At trace verbosity, logs may add:

- per-segment and per-unit details
- detailed artifact generation
- detailed external stdout/stderr routing
- branch decisions that affect outputs

## Required Tests And Acceptance Criteria

### Focused logging unit tests

Acceptance criteria:

- Logging setup does not attach duplicate handlers on repeated initialization.
- JSONL records include context fields and event names.
- Error logs capture tracebacks and worker context.
- Progress-aware console handlers preserve logging calls while progress contexts are active.

### Minimal real-data smoke

Example:

```bash
axon-recon-container stages preprocess \
  --config debug/debug.runtime.yml \
  --limit-datasets 1 \
  --limit-wells 1 \
  --limit-segments 2 \
  --force-restart
```

Acceptance criteria:

- Terminal shows `run_started`, `stage_started`, phase starts, phase completions, `phase_resource_usage`, `stage_completed`, and `run_completed`.
- `pipeline.log`, `pipeline.jsonl`, `errors.log`, and `summary.json` are created or updated.
- Phase logs are created only if enabled.
- No unselected datasets or wells appear in logs.

### Multi-target terminal visibility smoke

Use this when progress, Rich, multiprocessing, or interleaved worker logs are touched.

```bash
script -qefc 'axon-recon-container stages preprocess --config debug/debug.runtime.yml --limit-datasets 2 --limit-wells 2 --limit-segments 2 --force-restart' /tmp/axon-recon-terminal-smoke.txt
```

Acceptance criteria:

- The normalized transcript contains completion and resource-usage logs for every selected well.
- Later datasets/wells remain visible in the terminal transcript.
- Progress bars do not overwrite or hide final lifecycle logs.
- File logs and terminal logs agree on success/failure status.

### Worker-output suppression regression test

Acceptance criteria:

- Worker-thread code does not install process-global stdout/stderr redirects.
- Noisy external output suppression, when requested, still suppresses the intended library output.
- Concurrent worker logs from unrelated wells are not swallowed.

### Failure-path smoke or focused test

Acceptance criteria:

- A forced or simulated phase failure records `phase_failed`, `well_failed` or equivalent target failure, and `run_failed` when appropriate.
- `errors.log` includes traceback and context.
- `summary.json` reports failed target and failed phase.
- The terminal shows a concise failure message without dumping enormous stdout/stderr blocks.
