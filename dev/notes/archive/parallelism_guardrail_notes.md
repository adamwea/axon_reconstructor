Add this section to the guardrail file.

````markdown id="p4pdgw"
---

## Resource Tuning / Calibration Mode

The pipeline should support a resource tuning mode for empirically estimating realistic `phase_budget` values before large full runs.

The goal is to measure actual resource usage for representative phase executions, then provide recommendations for CPU/RAM/slot estimates and safe parallelism. This should help tune:

```yaml
resources.phase_budgets.<class>.cpu_cores
resources.phase_budgets.<class>.ram_gb
resources.phase_budgets.<class>.h5_read_slots
resources.phase_budgets.<class>.disk_heavy_slots
resources.phase_budgets.<class>.gpu_sort_slots
resources.phase_budgets.<class>.plot_slots
resources.phase_budgets.<class>.analyzer_slots
````

Resource tuning should be observational and advisory. It should not silently rewrite the runtime YAML unless explicitly requested.

---

## Why Resource Tuning Exists

Initial resource-class estimates are conservative guesses. Real usage depends on:

```text
dataset size
recording duration
number of segments
number of units
number of channels
SpikeInterface n_jobs
Kilosort behavior
available disk speed
whether data is on NAS / SSD / NVMe
plot/report complexity
debug limits
```

But after measuring representative cases, most phase estimates should become reasonably portable across machines. RAM demand and CPU demand are mostly properties of the phase, data shape, and inner parallelism. Machine profiles should describe available capacity; phase resource classes should describe expected demand.

The calibration workflow should therefore measure phase usage on one or more representative wells and recommend updated resource-class estimates with safety margins.

---

## Calibration Execution Modes

Support direct execution for calibration at both stage and phase level.

### Stage calibration

Run a stage with limits and resource logging enabled:

```bash
axon-recon tune-resources \
  --runtime runtime.debug.yml \
  --stage preprocess \
  --limit-datasets 1 \
  --limit-wells-per-dataset 1 \
  --limit-segments 2
```

Expected behavior:

```text
run selected stage
run selected wells through the stage phase_sequence
collect resource usage for every phase
summarize observed usage by phase and resource_class
emit recommendations
```

### Phase calibration

Run one or more individual phases directly, using existing upstream artifacts where required:

```bash
axon-recon tune-resources \
  --runtime runtime.debug.yml \
  --stage reconstruct \
  --phase plot_templates \
  --limit-datasets 1 \
  --limit-wells-per-dataset 1 \
  --limit-units 5
```

Expected behavior:

```text
resolve the selected dataset / recording / well / phase context
verify required inputs already exist
run only the selected phase for the limited scope
collect resource usage
emit recommendations for that phase's resource_class
```

If required upstream artifacts are missing, fail clearly with an actionable message:

```text
Cannot calibrate reconstruct.plot_templates because required template outputs are missing.
Run reconstruct.build_templates first, or run calibration at stage level.
```

Do not silently run unrelated upstream phases unless the user explicitly requests dependency preparation.

---

## Calibration Scope Controls

Calibration commands must respect normal limit flags.

Required supported limit concepts:

```text
limit_datasets
limit_recordings
limit_wells_per_dataset
limit_wells
limit_segments
limit_units
unit_limit
debug_mode
```

The exact flag names should follow the project’s real CLI conventions.

Calibration logs must clearly show the selected scope:

```text
Resource tuning scope:
  stage=reconstruct
  phase=plot_templates
  datasets=1
  wells=1
  segments<=2
  units<=5
  debug_mode=true
```

Do not allow calibration commands to accidentally run full datasets unless explicitly requested.

If no limits are supplied, require either:

```text
--confirm-full-scope
```

or fail with a clear message recommending limit flags.

---

## Calibration Data Collection

For every calibrated phase execution, record:

```text
stage
phase
resource_class
dataset
recording
well
source_h5_path
limits applied
wall_time_s
process_peak_rss_gb
child_peak_rss_gb
total_peak_rss_gb
cpu_time_user_s
cpu_time_system_s
max_threads
child_process_count_max
disk_read_gb
disk_write_gb
gpu_peak_memory_gb
gpu_utilization_max_pct
exit status
exception type if failed
```

Resource usage must include child processes when enabled, because phases may spawn SpikeInterface workers, Kilosort subprocesses, multiprocessing pools, plotting workers, or analyzer workers.

If a metric is unavailable, report `null`.

Write calibration results to a machine-readable artifact, for example:

```text
resource_tuning/resource_usage_observations.jsonl
resource_tuning/resource_tuning_summary.json
resource_tuning/resource_tuning_report.md
```

Use the project’s output directory conventions.

---

## Calibration Recommendation Logic

For each `resource_class`, aggregate observations across calibrated phase executions.

Recommended RAM estimate:

```text
recommended_ram_gb =
  ceil(max_observed_total_peak_rss_gb * ram_safety_factor)
```

Default:

```yaml
ram_safety_factor: 1.5
```

Recommended CPU estimate should be based on observed CPU/thread behavior, but it should be advisory because CPU utilization is harder to infer from process samples.

Use something like:

```text
recommended_cpu_cores =
  ceil(min(max_threads_observed, cpu_time_parallelism_estimate) * cpu_safety_factor)
```

or, if robust CPU parallelism estimation is not implemented yet:

```text
recommended_cpu_cores =
  current_resource_class.cpu_cores
```

with a tuning note.

Recommended slot behavior:

```text
gpu_sort_slots:
  keep 1 for kilosort4 unless explicit benchmarking shows concurrent sort improves wall time

plot_slots:
  keep 1 for plot/report classes unless repeated runs show plotting is stable in parallel

h5_read_slots:
  keep keyed source_h5_path limit at 1 unless repeated benchmark shows same-file concurrent reads are safe

disk_heavy_slots:
  increase only if disk wait is low and wall time improves with concurrency
```

Example recommendation output:

```text
Resource tuning recommendation: reconstruct.plot_templates
  resource_class=plot_unit
  observations=6
  max_total_peak_rss_gb=13.2
  current_class_ram_gb=24
  recommended_class_ram_gb=24
  current_class_cpu_cores=2
  recommended_class_cpu_cores=2
  note=Current RAM estimate is reasonable with 1.8x safety margin.
```

Overuse example:

```text
Resource tuning recommendation: reconstruct.report_full_chip_layout
  resource_class=plot_report_grid
  observations=3
  max_total_peak_rss_gb=41.7
  current_class_ram_gb=48
  recommended_class_ram_gb=64
  warning=Observed peak is too close to configured estimate.
```

Underuse example:

```text
Resource tuning recommendation: preprocess.save_rec_metadata
  resource_class=h5_metadata
  observations=12
  max_total_peak_rss_gb=0.8
  current_class_ram_gb=4
  recommended_class_ram_gb=2
  note=Class may be overestimated, but keep conservative value if measurements were from limited/debug runs.
```

Underuse recommendations should be conservative. Do not recommend lowering class estimates from tiny/debug runs unless enough observations exist.

---

## Calibration Config

Add optional config knobs:

```yaml
resources:
  tuning:
    enabled: false
    output_relpath: resource_tuning
    ram_safety_factor: 1.5
    cpu_safety_factor: 1.25
    min_observations_for_underuse: 5
    require_limits_unless_confirmed: true
    write_recommendations: true
    update_runtime_yml: false
    include_children: true
    include_gpu: true
    include_disk_io: true
```

`update_runtime_yml` must default to `false`.

If a future implementation supports writing patched YAML, it must write to a new file, not mutate the original by default:

```text
runtime.tuned.yml
```

---

## Calibration Logging

At calibration start:

```text
Starting resource tuning run
stage=reconstruct
phase=plot_templates
active_profile=lab_server_safe
limits={datasets:1, wells_per_dataset:1, units:5}
```

At each phase:

```text
Calibrating phase: reconstruct.plot_templates [well=..., resource_class=plot_unit]
```

At calibration end:

```text
Finished resource tuning run
observations_written=...
summary_path=...
recommendations_path=...
```

Also emit a concise recommendation block in the terminal logs:

```text
Resource tuning summary:
  plot_unit:
    current_ram_gb=24
    recommended_ram_gb=24
    observations=6
  spikeinterface_analyzer:
    current_ram_gb=48
    recommended_ram_gb=64
    observations=2
```

---

## Calibration Acceptance Criteria

Resource tuning changes are acceptable only if:

1. There is a CLI or runtime path to run calibration for a single phase.
2. There is a CLI or runtime path to run calibration for a whole stage.
3. Calibration respects limit flags.
4. Calibration refuses full-scope runs unless explicitly confirmed.
5. Calibration records resource usage for each well-local phase execution.
6. Calibration includes child processes in resource usage when configured.
7. Calibration writes machine-readable observation output.
8. Calibration writes or logs human-readable recommendations.
9. Calibration does not mutate runtime YAML by default.
10. Calibration clearly reports missing upstream artifacts for direct phase runs.
11. Calibration logs include stage, phase, well, resource_class, and active_profile.
12. Calibration can be used to tune resource class estimates without changing normal pipeline execution.
13. Calibration recommendations are advisory, not automatic enforcement.
14. Smoke tests demonstrate at least one limited phase calibration run.
15. Smoke tests inspect logs and output artifacts, not only process exit code.

---

## Calibration Smoke Test Requirement

When modifying resource tuning code, run at least one real smoke test.

Example phase calibration smoke test:

```bash
axon-recon tune-resources \
  --runtime runtime.debug.yml \
  --stage reconstruct \
  --phase plot_templates \
  --limit-datasets 1 \
  --limit-wells-per-dataset 1 \
  --limit-units 3
```

Inspect logs for:

```text
Starting resource tuning run
Calibrating phase: reconstruct.plot_templates
Phase resource usage: reconstruct.plot_templates
Resource tuning recommendation
Finished resource tuning run
```

Inspect output artifacts for:

```text
resource_usage_observations.jsonl
resource_tuning_summary.json
resource_tuning_report.md
```

Example stage calibration smoke test:

```bash
axon-recon tune-resources \
  --runtime runtime.debug.yml \
  --stage preprocess \
  --limit-datasets 1 \
  --limit-wells-per-dataset 1 \
  --limit-segments 2
```

Inspect logs for:

```text
Starting resource tuning run
Starting stage: preprocess
Calibrating phase: preprocess.save_rec_metadata
Calibrating phase: preprocess.preprocess_segments
Resource tuning recommendation
Finished resource tuning run
```

Do not accept calibration work that only adds pytest coverage without a real limited CLI smoke test.

```

One wording tweak I’d make to the core document too: define calibration as measuring **phase demand under a specified inner-parallelism setting**. CPU/RAM usage is not fully phase-intrinsic; `n_jobs`, channel count, unit count, segments, and recording duration matter. The report should include those context values so recommendations do not look more universal than they are.
```
