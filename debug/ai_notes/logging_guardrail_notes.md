````markdown id="aa66am"
# Logging Refinement Guardrails for Agentic Coding Agent

## Goal

Refine logging across the full `axon_recon` multi-stage pipeline so that terminal logs, file logs, structured logs, progress bars, resource tracking, and artifact tracking are consistent, professional, and useful for debugging.

The logging foundation already exists and currently prefers **Rich logging**. Do **not** redesign the logging system from scratch unless necessary. The main task is to normalize logging behavior across every stage and phase.

Primary priority:

- Clean, readable, professional **terminal INFO logs**.
- Progress bars remain visible and usable.
- Logs are ordered and contextual enough to understand pipeline state during execution.

Secondary priority:

- Rich enough file/structured logs that a coding agent can inspect behavior before/after refactors.
- Good artifact tracking.
- Optional resource tracking.
- Debug-level and trace-level visibility when needed.

---

# Expected Logging Stack

The pipeline should use this general structure:

```text
Python logging / project logger
        ↓
RichHandler for terminal logs
        ↓
File handlers for persistent human-readable logs
        ↓
JSONL structured handler for machine-readable event logs
        ↓
Optional per-dataset/per-recording/per-well/per-phase filtered handlers
````

Use **Rich** for terminal display.

Use either:

```text
stdlib logging + rich.logging.RichHandler
```

or:

```text
Loguru bridged into stdlib/Rich
```

depending on the current implementation.

Since the pipeline already likes Rich logging, prefer:

```python
from rich.logging import RichHandler
```

for console output.

Do not use raw `print()` for normal pipeline messages. Progress bars may use Rich, tqdm, SpikeInterface’s own progress display, or existing project progress utilities.

---

# High-Level Logging Architecture

The logging system should distinguish between:

```text
1. Terminal logs
2. Human-readable file logs
3. Structured JSONL logs
4. Error logs
5. Optional filtered physical logs
6. Final summary artifacts
```

## 1. Terminal Logs

Purpose:

* Human monitoring during a run.
* Show what is happening now.
* Keep output readable.
* Preserve progress bars.

Expected terminal sink:

```text
RichHandler / Rich console
```

Terminal logs should be controlled primarily by:

```yaml
logging:
  verbosity: quiet | normal | detailed | trace
  console:
    enabled: true
    rich: true
```

Terminal output should **not** be the source of truth. It is a curated live view.

---

## 2. Human-Readable File Logs

Purpose:

* Persistent full-run debugging.
* More complete than terminal logs.
* Plain text, grep-friendly.
* No Rich markup required.

Expected file outputs:

```text
logs/pipeline.log
logs/errors.log
logs/datasets/<dataset_id>/dataset.log
logs/datasets/<dataset_id>/recordings/<recording_id>/recording.log
logs/datasets/<dataset_id>/recordings/<recording_id>/wells/<well_id>/well.log
logs/datasets/<dataset_id>/recordings/<recording_id>/wells/<well_id>/phases/<stage>__<phase>.log
```

Not all physical logs need to be enabled by default. They should be independently configurable.

---

## 3. Structured JSONL Logs

Purpose:

* Canonical event stream.
* Source of truth for post-hoc analysis.
* Machine-readable debugging.
* Useful for coding agents, future dashboards, summaries, comparisons, and regression checks.

Expected file:

```text
logs/pipeline.jsonl
```

Each line should be a complete JSON object.

Example:

```json
{
  "time": "2026-05-03T14:21:33.102",
  "level": "INFO",
  "event": "phase_started",
  "message": "Starting Kilosort4",
  "run_id": "debug01",
  "dataset_id": "Media_Density_T5",
  "recording_id": "000031",
  "chip_id": "M08073",
  "assay": "AxonTracking",
  "well_id": "well000",
  "stage": "spike_sorting",
  "phase": "run_kilosort4",
  "pid": 418984,
  "worker": "ForkPoolWorker-3"
}
```

Structured logs should preserve context even when terminal verbosity is low or physical per-well/per-phase logs are disabled.

---

## 4. Error Logs

Purpose:

* Fast failure triage.
* Warnings, errors, critical failures, and tracebacks.
* Avoid searching the full pipeline log for failures.

Expected file:

```text
logs/errors.log
```

Should include:

```text
WARNING
ERROR
CRITICAL
exception type
exception message
traceback
dataset/recording/well/stage/phase context
external command failure details
stdout/stderr file paths when applicable
```

---

## 5. Optional Filtered Physical Logs

Purpose:

* Debug one dataset, recording, well, or phase without scanning the whole run.

Expected optional outputs:

```text
logs/datasets/<dataset_id>/dataset.log
logs/datasets/<dataset_id>/recordings/<recording_id>/recording.log
logs/datasets/<dataset_id>/recordings/<recording_id>/wells/<well_id>/well.log
logs/datasets/<dataset_id>/recordings/<recording_id>/wells/<well_id>/phases/<stage>__<phase>.log
```

Rules:

* Per-dataset logs contain only that dataset.
* Per-recording logs contain only that recording.
* Per-well logs contain only that well.
* Per-phase logs contain only that well/stage/phase.
* These logs should be generated by filtering contextual log records, not by duplicating logging calls manually.

---

## 6. Summary Artifact

Purpose:

* Final run-level status report.
* Easy to inspect success/failure/skipped state without reading logs.

Expected file:

```text
logs/summary.json
```

Should include:

```text
run_id
start time
end time
elapsed time
datasets processed
recordings processed
wells processed
stages completed
phases completed
failed stages/phases
skipped stages/phases
artifact summary
resource summary if enabled
```

---

# Expected Log Directory Layout

Use this as the target structure unless the project already has a stronger convention.

```text
<run_output_dir>/
└── logs/
    ├── pipeline.log
    ├── pipeline.jsonl
    ├── errors.log
    ├── summary.json
    └── datasets/
        └── <dataset_id>/
            ├── dataset.log
            └── recordings/
                └── <recording_id>/
                    ├── recording.log
                    └── wells/
                        └── <well_id>/
                            ├── well.log
                            └── phases/
                                ├── preprocessing__save_raw_binary.log
                                ├── spike_sorting__run_kilosort4.log
                                ├── templates__extract_waveforms.log
                                ├── reconstruction__reconstruct_units.log
                                └── reporting__generate_reports.log
```

Physical phase logs may be disabled by default because they can create many files.

---

# Logging Context Hierarchy

The pipeline hierarchy is:

```text
Run
└── Dataset
    └── Recording
        └── Well
            └── Stage
                └── Phase
```

Every log record should carry context when available:

```text
run_id
dataset_id / dataset_name
recording_id
chip_id
assay
date
well_id
stage
phase
pid
worker/process name
event type
elapsed time
resource metrics, when enabled
artifact paths, when generated
```

Do not manually format context into message strings everywhere.

Avoid this:

```python
logger.info(f"[{dataset_id}][{well_id}][{stage}][{phase}] Starting phase")
```

Prefer a contextual adapter/helper:

```python
logger = get_logger().bind(
    dataset_id=dataset_id,
    recording_id=recording_id,
    well_id=well_id,
    stage="spike_sorting",
    phase="run_kilosort4",
)

logger.info("Starting Kilosort4")
```

If using stdlib logging, use a `LoggerAdapter`, context variables, a custom filter, or the project’s existing context manager.

If using Loguru, use `.bind(...)`.

If using Rich through stdlib logging, use contextual `extra` fields or a project-specific context manager.

---

# Recommended Internal Module Structure

Use or adapt something like:

```text
axon_recon/pipeline/logging/
├── __init__.py
├── config.py
├── setup.py
├── context.py
├── formatters.py
├── filters.py
├── handlers.py
├── summary.py
├── resources.py
├── artifacts.py
└── multiprocessing.py
```

Responsibilities:

## `config.py`

* Parse logging settings from `runtime.yml`.
* Validate verbosity mode.
* Resolve output paths.
* Provide defaults.

## `setup.py`

* Initialize logging once.
* Attach Rich terminal handler.
* Attach run log handler.
* Attach JSONL handler.
* Attach error handler.
* Attach optional filtered handlers.
* Avoid duplicate handlers on repeated pipeline invocation.

## `context.py`

* Provide context binding helpers.
* Maintain current run/dataset/recording/well/stage/phase context.
* Support multiprocessing-safe propagation where possible.

## `formatters.py`

* Human-readable text formatter.
* JSONL formatter.
* Rich terminal formatting options.

## `filters.py`

* Filter records by dataset.
* Filter records by recording.
* Filter records by well.
* Filter records by stage/phase.
* Filter records by verbosity.

## `handlers.py`

* Create file handlers.
* Create Rich console handler.
* Create JSONL handler.
* Create error handler.
* Create dynamic per-well/per-phase handlers if enabled.

## `summary.py`

* Track lifecycle events.
* Write `summary.json`.
* Track success/failure/skipped state.
* Track elapsed times.

## `resources.py`

* Optional resource snapshots.
* CPU/memory/GPU/disk metrics where available.
* Resource logging should never crash the pipeline.

## `artifacts.py`

* Standard helpers for logging artifact events:

  * created
  * reused
  * overwritten
  * deleted
  * missing
  * validated
  * invalid

## `multiprocessing.py`

* Queue/listener setup if applicable.
* Worker logging initialization.
* Safe shutdown/flush behavior.
* Exception capture from workers.

---

# Rich Logging Expectations

Use Rich for terminal readability.

Expected behavior:

* Colored levels.
* Clean timestamps.
* No excessive file path/module spam unless enabled.
* Progress bars should coexist with logs.
* Exceptions should render nicely in terminal when appropriate.
* File logs should remain plain text or JSONL, not Rich markup.

Recommended Rich setup conceptually:

```python
from rich.console import Console
from rich.logging import RichHandler
import logging

console = Console(stderr=True)

rich_handler = RichHandler(
    console=console,
    rich_tracebacks=True,
    markup=False,
    show_time=True,
    show_level=True,
    show_path=False,
)

logging.basicConfig(
    level=logging.INFO,
    handlers=[rich_handler],
)
```

Adapt to the actual project implementation.

Do not let every worker process create its own unmanaged Rich console handler if that causes jumbled terminal output. In multiprocessing, prefer the parent process to own terminal rendering through a queue/listener architecture.

---

# Runtime YAML Controls

Add or normalize a logging section in `runtime.yml`.

Exact names can be adapted, but preserve the functionality.

```yaml
logging:
  enabled: true

  verbosity: normal
  # allowed:
  #   quiet
  #   normal
  #   detailed
  #   trace

  console:
    enabled: true
    rich: true
    show_timestamps: true
    show_context: true
    show_elapsed: true
    level: INFO

  progress:
    enabled: true
    preserve_spikeinterface_progress: true
    preserve_pipeline_progress: true

  structured:
    enabled: true
    path: logs/pipeline.jsonl
    level: DEBUG

  run_log:
    enabled: true
    path: logs/pipeline.log
    level: INFO

  error_log:
    enabled: true
    path: logs/errors.log
    level: WARNING

  dataset_logs:
    enabled: true
    level: INFO

  recording_logs:
    enabled: true
    level: INFO

  well_logs:
    enabled: true
    level: DEBUG

  phase_logs:
    enabled: false
    level: DEBUG

  resource_tracking:
    enabled: true
    terminal:
      enabled: true
      min_verbosity: detailed
    file:
      enabled: true
      min_verbosity: detailed
    include_cpu: true
    include_memory: true
    include_gpu: false
    include_disk_io: false
    interval_seconds: 30

  artifacts:
    log_created_artifacts: true
    log_existing_artifacts: false
    log_deleted_artifacts: true
    log_cache_hits: true
    log_cache_misses: true

  external_tools:
    capture_stdout: true
    capture_stderr: true
    terminal_stdout_min_verbosity: trace
    terminal_stderr_min_verbosity: detailed
    file_stdout_min_verbosity: detailed
    file_stderr_min_verbosity: detailed

  multiprocessing:
    enabled: true
    use_queue_listener: true
    include_pid: true
    include_worker_name: true

  exceptions:
    capture_warnings: true
    capture_uncaught_exceptions: true
    include_tracebacks: true
```

---

# Verbosity Modes

## `quiet`

Purpose:

Mostly silent execution.

Terminal should show:

* Progress bars.
* Critical failures.
* Final summary.

Terminal should not show:

* Routine INFO logs.
* Per-stage chatter.
* Per-phase chatter.
* Per-well chatter unless failure occurs.

File logs may still contain full records if enabled.

---

## `normal`

Purpose:

Professional default terminal output.

Terminal should show:

* Run start/completion.
* Dataset start/completion.
* Recording start/completion when applicable.
* Stage start/completion.
* Major phase start/completion.
* Per-well start/completion for long-running well jobs.
* Progress bars.
* Warnings/errors.
* Final summary.

Terminal should not show:

* Per-segment logs.
* Per-unit logs.
* Large configs.
* Raw arrays.
* Huge stdout dumps.
* Repeated redundant context.

This should be the default.

---

## `detailed`

Purpose:

Inspect per-well and per-phase behavior.

Terminal/file logs should show:

* Everything from `normal`.
* All phase start/completion events.
* Per-well phase boundaries.
* Important artifact paths.
* Important parameter choices.
* Cache hits/misses.
* Resource usage at stage/phase boundaries.
* External tool command summaries.
* Skipped optional outputs.

Terminal should still avoid:

* Massive arrays.
* Huge configs.
* Unit-by-unit spam unless specifically relevant.

---

## `trace`

Purpose:

Maximum debugging.

Terminal/file logs may show:

* Everything from `detailed`.
* Every meaningful step inside each phase.
* Per-segment behavior.
* Per-unit behavior.
* Detailed resource snapshots.
* Detailed external stdout/stderr routing.
* Detailed object dimensions and resolved paths.
* Branch decisions that affect outputs.

This mode can be noisy. It is for debugging, not routine runs.

---

# Terminal Formatting Requirements

Preferred normal terminal style:

```text
14:21:33 | INFO | well000 | spike_sorting/run_kilosort4 | Starting Kilosort4
```

Alternative full context style:

```text
2026-05-03 14:21:33 | INFO | dataset=Media_Density_T5 recording=000031 well=well000 stage=spike_sorting phase=run_kilosort4 | Starting Kilosort4
```

Completion logs should include elapsed time:

```text
14:42:36 | INFO | well000 | spike_sorting/run_kilosort4 | Completed Kilosort4 in 00:21:03
```

Skip logs should include a reason:

```text
14:42:36 | INFO | well000 | preprocessing/save_raw_binary | Skipping raw binary write | reason=exists and overwrite=false
```

Failure logs should include the exception summary:

```text
14:42:36 | ERROR | well000 | spike_sorting/run_kilosort4 | Failed Kilosort4 | RuntimeError: sorter exited with nonzero status
```

---

# Lifecycle Events

Every major unit of work should emit lifecycle events.

Required events:

```text
run_started
run_completed
run_failed

dataset_started
dataset_completed
dataset_failed

recording_started
recording_completed
recording_failed

well_started
well_completed
well_failed
well_skipped

stage_started
stage_completed
stage_failed
stage_skipped

phase_started
phase_completed
phase_failed
phase_skipped
```

Every completion event should include elapsed time.

Every skipped event should include a reason.

Every failure event should include:

```text
exception type
exception message
traceback in file logs
context fields
failed artifact if applicable
```

---

# Stage-Level Acceptance Criteria

The pipeline has five broad stages:

```text
1. preprocessing
2. spike_sorting
3. waveform/template extraction
4. reconstruction
5. reporting
```

Adapt names to the actual codebase.

Each stage should satisfy this contract.

At `normal` verbosity:

* Stage started.
* Stage completed.
* Stage failed, if applicable.
* Stage skipped, if applicable.
* Number of datasets/recordings/wells affected where useful.
* Major phase start/completion events.
* Progress bars for long-running well/unit/segment loops.

At `detailed` verbosity:

* All phase start/completion events.
* Per-well start/completion events.
* Key input paths.
* Key output paths.
* Major config decisions.
* Cache behavior.
* Resource usage at stage and phase boundaries.

At `trace` verbosity:

* Per-segment and per-unit behavior where relevant.
* Detailed artifact generation.
* Detailed resource snapshots.
* External tool stdout/stderr according to config.
* Internal branch decisions that affect output.

---

# Phase-Level Acceptance Criteria

Every phase should follow the same pattern.

## Start

Log:

```text
Starting phase: <phase_name>
```

Include:

```text
stage
phase
dataset/recording/well context if available
important input path
important output path if known
important counts if known
```

## Completion

Log:

```text
Completed phase: <phase_name> in <elapsed>
```

Include:

```text
elapsed time
created artifacts
important output path
summary counts
resource usage if enabled
```

## Skip

Log:

```text
Skipping phase: <phase_name> | reason=<reason>
```

Include:

```text
skip reason
config toggle or condition responsible
existing artifact path if relevant
```

## Failure

Log:

```text
Failed phase: <phase_name> | error=<ExceptionType>: <message>
```

Include:

```text
exception type
exception message
traceback in file logs
external command if relevant
stdout/stderr path if relevant
partial artifact path if relevant
```

---

# Stage-Specific Logging Requirements

## 1. Preprocessing

At `normal`, terminal should show:

* Stage start/completion.
* Major phases:

  * loading recording metadata
  * saving/validating raw binary
  * writing preprocessing metadata
  * copy/staging/cache behavior
* Progress bars for long operations.
* Warnings for missing or suspicious metadata.

At `detailed`, logs should include:

* Source recording path.
* Destination binary path.
* Whether data is copied, staged, referenced, cached, or skipped.
* Recording duration.
* Number of channels.
* Sampling frequency.
* Number of segments.
* Well IDs discovered.
* Metadata artifact paths.
* Resource usage at phase boundaries.
* Cache hit/miss behavior.

At `trace`, logs should include:

* Per-segment preprocessing details.
* Per-segment duration/channel counts.
* Detailed validation checks.
* Binary write/read verification steps.
* Disk I/O metrics if enabled.

Acceptance criteria:

* User can tell whether preprocessing used original `.h5`, an existing binary, or wrote a new binary.
* User can tell which wells/segments were discovered.
* User can tell where the definitive downstream binary/reference was saved.
* User can tell why preprocessing skipped work.
* Missing metadata produces a warning with enough context to locate the issue.

---

## 2. Spike Sorting

At `normal`, terminal should show:

* Stage start/completion.
* Per-well sorting start/completion.
* Sorter used, e.g. Kilosort4.
* SpikeInterface/sorter progress bars when available.
* Failures by well.
* Whether sorting strategy is concatenated/bootstrap, per-segment, or other.

At `detailed`, logs should include:

* Sorter name and version if available.
* Sorting strategy.
* Input recording path/reference.
* Output sorting folder.
* Docker/container usage if applicable.
* GPU-related notes if available.
* Number of channels.
* Duration.
* Number of detected units after sorting.
* External command summary.
* External stdout/stderr capture paths.
* Resource usage before/after sorter execution.
* Per-well success/failure.

At `trace`, logs should include:

* Detailed sorter parameters.
* Detailed environment/container command.
* More complete stdout/stderr routing.
* Segment-level sorting/registering behavior.
* Intermediate artifact paths.
* Unit merge/split decisions if they occur here.

Acceptance criteria:

* User can tell which well is currently sorting.
* User can tell whether sorting is per-well, per-segment, or concatenated.
* User can tell where sorter output was written.
* User can tell how many units were produced.
* User can diagnose external sorter failure from logs.
* Multiprocessing does not jumble well logs beyond readability.

---

## 3. Waveform / Template Extraction

At `normal`, terminal should show:

* Stage start/completion.
* Per-well analyzer/template extraction start/completion.
* SpikeInterface progress bars.
* Number of units processed.
* Major skipped extraction due to existing analyzer/cache.

At `detailed`, logs should include:

* Sorting input path.
* Recording input path.
* Analyzer output path.
* Number of units.
* Number of channels.
* Number of spikes sampled per unit.
* Window sizes / waveform extraction parameters.
* Template extraction parameters.
* Cache hits/misses.
* Resource usage at phase boundaries.

At `trace`, logs should include:

* Per-unit extraction summaries.
* Per-extension computation events if using SpikeInterface analyzers.
* Detailed analyzer extension list.
* Intermediate artifact paths.
* Failed units/extensions.

Acceptance criteria:

* User can tell when SpikeInterface computes waveforms/templates versus loading cached results.
* User can tell how many units are being analyzed.
* User can tell where analyzer output lives.
* User can see SpikeInterface progress bars.
* User can diagnose whether slowness is due to unit count, channel count, spike count, or cache misses.

---

## 4. Reconstruction

At `normal`, terminal should show:

* Stage start/completion.
* Per-well reconstruction start/completion.
* Number of units reconstructed.
* Number of units skipped/failed.
* Reconstruction method used.
* Progress bars for unit-level reconstruction.

At `detailed`, logs should include:

* Input template/analyzer path.
* Reconstruction output directory.
* Method/algorithm selected.
* Major thresholds/parameters.
* Number of candidate units.
* Number of successfully reconstructed units.
* Number of failed/skipped units.
* Artifact paths for reconstruction summaries.
* Resource usage at phase boundaries.

At `trace`, logs should include:

* Per-unit reconstruction decisions.
* Per-unit channel counts.
* Per-unit branch candidate counts.
* Per-unit pruning/cleaning decisions.
* Per-unit artifact paths.
* Detailed failure reasons for skipped units.

Acceptance criteria:

* User can tell how many units entered reconstruction.
* User can tell how many produced usable morphology outputs.
* User can tell why units were skipped.
* User can locate per-unit reconstruction artifacts.
* Trace logs are detailed enough to compare reconstruction behavior before/after algorithm refactors.

---

## 5. Reporting

At `normal`, terminal should show:

* Stage start/completion.
* Major reports being generated.
* Per-well report completion.
* Final report/output location.
* Full-chip layout report generation if applicable.
* Warnings for missing expected inputs.

At `detailed`, logs should include:

* Input artifact paths.
* Output report paths.
* Number of units included in reports.
* Number of branches/objects plotted.
* Plot dimensions or geometry source when relevant.
* Skipped plots/reports and why.
* Resource usage for expensive report generation.

At `trace`, logs should include:

* Per-unit plotting/reporting behavior.
* Per-figure artifact paths.
* Detailed geometry/layout calculations.
* Fallback behavior for missing metadata.

Acceptance criteria:

* User can tell which reports were generated.
* User can locate report artifacts from logs.
* User can tell if a report is incomplete due to missing reconstruction/template artifacts.
* Full-chip plots log chip geometry source and number of included units/branches.

---

# Artifact Logging Requirements

Whenever a phase creates, reuses, modifies, validates, invalidates, or deletes an important artifact, log it.

Artifact event types:

```text
artifact_created
artifact_reused
artifact_overwritten
artifact_deleted
artifact_missing
artifact_validated
artifact_invalid
```

Artifact logs should include:

```text
artifact_type
path
stage
phase
well_id if applicable
size if cheap to compute
reason if skipped/reused/deleted
```

Examples:

```text
Created artifact: raw_binary | path=/...
Reused artifact: sorting_output | path=/... | reason=exists and overwrite=false
Deleted artifact: temporary_binary | path=/... | reason=cleanup enabled
Missing artifact: analyzer | path=/... | action=recomputing
```

---

# Resource Tracking Requirements

Resource tracking should be optional and controlled by runtime config.

At minimum, support:

```text
CPU percent
memory RSS
available memory
disk usage for relevant output location
GPU utilization/memory if implemented and available
elapsed time
```

Resource logs should appear:

* At stage start/completion.
* At phase start/completion.
* Periodically during long-running phases when enabled.
* Around external tool execution.
* In file logs at `detailed` or `trace`.
* In terminal only when verbosity allows.

Acceptance criteria:

* Resource tracking can be turned off.
* Resource logs do not spam `normal` terminal output.
* Long-running phases can emit periodic resource snapshots.
* Resource data is structured enough to compare performance across runs.
* Resource tracking failure does not fail the pipeline.

---

# Multiprocessing Logging Requirements

Multiprocessing logs must remain attributable and readable.

Acceptance criteria:

* Every worker log includes PID and worker/process name in structured/file logs.
* Every worker log includes dataset/recording/well/stage/phase context when available.
* File writes are not corrupted by concurrent workers.
* Terminal logs are not randomly interleaved in a way that destroys readability.
* Worker exceptions are captured with tracebacks.
* Parent process logs a clear summary of worker successes/failures.
* Per-well logs contain only that well’s events.
* Per-phase logs contain only that phase’s events when phase logs are enabled.
* Rich terminal output should preferably be handled by the parent process, not independently by each worker.

---

# Testing Requirements

Focus tests primarily on terminal logging behavior, but also verify file/structured logs where practical.

## Unit Tests

Add tests for:

* Verbosity mode resolution.
* Context binding.
* Rich terminal handler setup.
* Log formatting.
* Lifecycle event helpers.
* Artifact logging helpers.
* Resource tracking helpers with mocked resource data.
* Duplicate-handler prevention.
* Config parsing from `runtime.yml`.

Acceptance criteria:

* `quiet`, `normal`, `detailed`, and `trace` modes map to expected terminal/file behavior.
* Context fields are preserved in structured logs.
* Disabled physical logs do not create files.
* Enabled physical logs do create files.
* Duplicate initialization does not duplicate log lines.
* Rich logging can be initialized without breaking file logs.

---

## Terminal Log Tests

Use `pytest` capture tools or equivalent.

For each verbosity mode, test a small fake run with:

```text
1 dataset
1 recording
2 wells
2 stages
2 phases per stage
```

Acceptance criteria:

### Quiet

* Progress-compatible output allowed.
* No routine INFO phase logs.
* Errors are visible.
* Final summary is visible.

### Normal

* Stage start/completion visible.
* Major phase start/completion visible.
* Well start/completion visible.
* Per-unit/per-segment logs absent.
* Output order is readable.

### Detailed

* All phase start/completion visible.
* Artifact paths visible.
* Resource summary visible if enabled.
* Per-well details visible.
* Per-unit trace spam absent.

### Trace

* Per-segment/per-unit logs visible where emitted.
* Detailed artifact/resource logs visible.
* Context remains readable.

---

## File Log Tests

Test that:

* `pipeline.log` contains the whole run.
* `pipeline.jsonl` contains structured events.
* `errors.log` contains warnings/errors only.
* Per-dataset logs contain only that dataset.
* Per-recording logs contain only that recording.
* Per-well logs contain only that well.
* Per-phase logs contain only that well/stage/phase when enabled.
* Disabling per-phase logs prevents phase log files from being created.
* Disabling per-well logs prevents well log files from being created.
* Structured logs still contain well/phase context even when physical well/phase logs are disabled.

---

## Multiprocessing Tests

Create a small multiprocessing test or mocked equivalent.

Acceptance criteria:

* Logs from two wells do not get mislabeled.
* Per-well files are not contaminated with the other well’s records.
* Parent summary correctly records success/failure per worker.
* Worker exceptions are logged with context.
* Queue/listener shutdown flushes all logs before process exit.
* Rich terminal output remains readable.

---

# Refactor Rules

When improving logging across stages/phases:

1. Do not add random `print()` calls.
2. Do not manually prefix every message with context.
3. Do not duplicate the same log event in multiple places.
4. Use existing logging helpers/context managers where possible.
5. If existing helpers are insufficient, improve the helper layer first.
6. Keep terminal logs concise at `normal` verbosity.
7. Put noisy diagnostics behind `detailed` or `trace`.
8. Ensure every long-running phase has start/completion/failure logs.
9. Ensure every important artifact is logged.
10. Ensure every skip has a reason.
11. Ensure every failure has enough context to debug.
12. Preserve SpikeInterface and pipeline progress bars.
13. Prefer Rich for terminal display.
14. Keep file logs plain and grep-friendly.
15. Keep structured logs machine-readable and complete.

---

# Definition of Done

This logging refinement is complete when:

* All stages and phases use the same logging conventions.
* Terminal output at `normal` verbosity looks professional and readable.
* Rich logging is used consistently for terminal output.
* File logs and JSONL logs are generated in predictable locations.
* `quiet`, `normal`, `detailed`, and `trace` modes behave distinctly.
* Progress bars remain visible.
* Stage/phase/well lifecycle events are consistently logged.
* Artifact creation/reuse/deletion is consistently logged.
* Resource tracking is available and configurable.
* Multiprocessing logs remain attributable to the correct well/phase.
* File logs contain enough detail for a coding agent to compare behavior before/after refactors.
* Tests verify terminal logging behavior for each verbosity mode.
* Tests verify physical logs are correctly filtered by dataset/recording/well/phase.
* Tests verify disabling physical logs does not remove structured context.

```
```
