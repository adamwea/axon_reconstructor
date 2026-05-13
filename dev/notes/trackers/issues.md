# Issues — known bugs / specific fixes

Defects against current pipeline behavior, with reproduction + impact + suggested
fix shape. Distinct from `roadmap.md` (which is about new ambitions) and
`tech_debt.md` (which is about cleanup of working-but-ugly code).

## Format

```
### <Short title>
- **Status**: open | in-flight | fixed-in <commit> | wont-fix
- **Tags**: stage/area tags (spikesort, recon, analysis, dashboard, container, infra)
- **Repro**: how to observe the bug
- **Impact**: what breaks because of it
- **Workaround**: temporary mitigation if any
- **Suggested fix**: rough shape of the fix (will grow into a plan if large)
- **See also**: related commits, plans, prior discussion
```

---

## Open issues

### `--force-restart` does not reliably clean prior artifacts
- **Status**: open
- **Tags**: spikesort, preprocess, reconstruct, analysis, infra
- **Repro**: run a stage with `--force-restart` after a previous run wrote partial
  outputs; observe residual files from the prior run persisting in the well's
  output dirs, producing hybrid-state runs on rerun.
- **Impact**: reruns are not actually clean restarts. Hard-to-diagnose state
  bleeding between iterations. User has hit this in multiple places already.
- **Workaround**: manually `rm -rf <well>/<stage>_outputs/` before rerunning.
- **Suggested fix**: standardize the contract — every stage runner, at entry with
  `force_restart=True`, calls a uniform pre-clean helper that wipes the stage's
  output directory(ies). Per-phase opt-outs are gone; if you need to preserve
  something (e.g., concat_analyzer cache, SLAy AE model), re-run only the
  specific phases you want to preserve via the `--phase` arg. All-or-nothing
  semantics across all four stages (preprocess, spikesort, reconstruct, analysis).
  Each phase needs a test asserting the contract (write garbage to output dir,
  run with force_restart=True, assert garbage gone). Will earn its own
  fix-plan when scoped.
- **See also**: discussed 2026-05-11; ordering decision: do
  `debug_mode` YAML cleanup (see tech_debt.md) FIRST to shrink the audit surface,
  then this.

### merge_SLAy unit_locations plots scope and channel overlay
- **Status**: open
- **Tags**: spikesort, plots
- **Repro**: run `axon-recon stages spikesort.merge_SLAy --target-dataset N
  --limit-wells 1`; inspect `<well>/spikesort_outputs/merge_SLAy/
  unit_locations_before_after_merge.png` (and the single-axis `_before_merge.png` /
  `_after_merge.png`). The plot bbox is shrunk to just the units actively being
  merged, not to the full set of pre-merge units passing the quality filter.
- **Impact**: pre/post comparison is visually misleading — you can't see where
  the merged units sit relative to the rest of the good population. The post-merge
  panel may also show units displaced slightly outside the bbox due to
  weighted-average locations.
- **Workaround**: none currently.
- **Suggested fix**:
  - Drop the `zoom_to_affected_units` branch in
    `_write_merge_unit_location_reports` (`spikesort/runner.py`); bbox always
    derived from the filtered pre-merge set with a small margin.
  - Both panels share the same `(x_limits, y_limits)` (the same bbox the
    pre-merge filtered set produces). Post-merge weighted-avg locations that
    drift outside the bbox stay visible but the axis doesn't expand. Log how
    many post-merge units land outside the pre-merge bbox as a QC diagnostic
    in the report payload.
  - Single-axis plots share the same bbox but stay as separate files.
  - Probe-dimension inheritance branch (`merge_reports_2panel_probe_dim_x_um/y_um`)
    stays available as opt-in, off by default.
  - **New: channel-square overlay.** Plot the union of `merged_contributing_electrode_ids.json`
    channel xy positions across the displayed units as small grey squares below
    the unit dots. Each panel uses its own channel set (pre-panel = pre-units'
    union; post-panel = post-units' union). Config knobs default on:
    `merge_reports_2panel_plot_template_channels`,
    `merge_reports_2panel_template_channel_marker_size`,
    `merge_reports_2panel_template_channel_color`.
- **See also**: discussed 2026-05-11; quality-filter fix landed in
  `6a6e8c6 filter unit_locations plots to good/non_soma_good`.

### NAS mount `/mnt/ben-shalom_nas/` periodically stale
- **Status**: open (environment, not code)
- **Tags**: infra, container
- **Repro**: `ls /mnt/ben-shalom_nas/` returns "Resource temporarily unavailable";
  `axon-recon-container` preflight fails with `cannot create writable output_root
  path /mnt/ben-shalom_nas/analysis/...`.
- **Impact**: container launches blocked when the share is stale. Affects every
  stage invocation that flows through the wrapper.
- **Workaround**: use the documented bypass form:
  ```
  axon-recon-container \
    --no-config-mounts \
    --mount /mnt/disk15tb/adamm/scratch:/mnt/disk15tb/adamm/scratch:rw \
    --mount /mnt/disk15tb/adamm/dev/pkgs/axon_reconstructor/debug:/mnt/disk15tb/adamm/dev/pkgs/axon_reconstructor/debug:ro \
    --gpus all \
    stages <stage> --config debug/debug.runtime.yml ...
  ```
- **Suggested fix**: lower-priority since `publish_outputs: false` means the run
  doesn't actually need the NAS path — only the preflight check does. Soften
  the preflight so unreachable `output_root` is a warning-and-skip rather than
  a hard failure when `publish_outputs: false` is set in the runtime config.
- **See also**: workaround validated in conversation 2026-05-11.


### Multi-rank stage summary shows per-rank slice only
- **Status**: open
- **Tags**: infra, logging, mpi
- **Repro**: run any stage with backend `mpi` at `-n N > 1`. Each rank prints its
  own `stage_aggregate_summary_lines(agg)` block at the end of the stage,
  reflecting only that rank's ~1/N partition of targets (round-robin from
  `partition_targets_by_mpi_rank` in `src/axon_recon/pipeline/mpi_adapter.py:177`).
  No MPI gather of `target_results` happens before the printer fires.
- **Impact**: misleading summary output in multi-rank runs. A user inspecting the
  end-of-stage block sees `targets_total: M/N` from rank 0 and may think only
  a fraction of work was attempted. Per-target listings that follow only show
  the same rank's slice. The pipeline still does the full work — only the
  summary is wrong.
- **Workaround**: read the full stage log (pipeline.jsonl + per-well manifests)
  instead of trusting the terminal summary in MPI mode.
- **Suggested fix**: in `_distribute_runtime_targets` (or earlier, inside
  `distribute_targets`), MPI-gather `target_results` to rank 0 after the
  per-rank work finishes; build the canonical `MultiTargetStageResult` on rank
  0 only; have non-root ranks return an empty result so their printers no-op.
  Add a test that runs a synthetic 2-rank pipeline and asserts the rank-0 agg
  contains both ranks' targets.
- **See also**: `stage_aggregate_summary_lines` introduced in commit 90e17e3.


### `keyed_resource_limits.source_h5_path` does not gate across MPI ranks
- **Status**: open, low priority (GPU node is fast enough that the contention
  is not currently noticeable in real runs)
- **Tags**: infra, mpi, resource_budget, scheduling
- **Repro**: run any stage that reads source H5 (preprocess, spikesort) under
  `--task-backend mpi` with `-n N > 1` and a config that includes multiple
  datasets. With 4 ranks × 4 datasets, observe the first scheduling slot:
  all 4 ranks open well000–well003 of the **same** earliest dataset's H5,
  rather than well000 of each of the 4 datasets. Confirmed empirically on
  Perlmutter GPU node, 2026-05-12, spikesort.bootstrap_concat_binary phase.
- **Impact**: concurrent reads of the same `data.raw.h5` across ranks create
  filesystem contention (Lustre metadata + per-file bandwidth) and partially
  defeat the purpose of `max_concurrent: 1`. Currently low impact on Perlmutter
  because pscratch + A100 throughput dominate the cost. Higher impact on
  slower storage or when scaling to many ranks/datasets.
- **Workaround**: none clean. Manually narrow `--target-dataset` to one dataset
  at a time, or accept the contention.
- **Root cause** (two layers):
  1. **Partition is index-blind.** `partition_targets_by_mpi_rank` in
     `src/axon_recon/pipeline/mpi_adapter.py:177-184` slices the target list
     by `i % size`. The target list is ordered (dataset, well), so ranks 0..3
     all start on the first dataset. A partition strategy that strides by
     dataset first (rank `r` starts at dataset `r mod ndatasets`) would
     spread the initial fan-out without any runtime coordination.
  2. **Runtime gate is per-process.** The keyed-resource counter in
     `src/axon_recon/pipeline/resource_budget.py:117-132` lives inside one
     process's resource-budget manager. Each MPI rank instantiates its own
     manager with its own counter, so `max_concurrent: 1` reads as
     "1 concurrent per process" rather than "1 concurrent globally". Even
     with a smarter partition, later scheduling moments can still re-collide
     (rank 0 finishes dataset0/well000 and walks to dataset0/well001 while
     rank 2 is still on dataset0/well002).
- **Suggested fix shape**:
  - **Cheap, partition-only**: rewrite `partition_targets_by_mpi_rank` to be
    h5-key-aware. Group targets by `source_h5_path`, then deal them out
    round-robin across ranks so no rank gets two same-h5 targets adjacent in
    its queue and ranks start on distinct h5 keys whenever possible. Pure
    scheduling change; no IPC. Doesn't fully solve later-scheduling-step
    overlaps but eliminates the worst case (first slot collision) cheaply.
  - **Robust, backend-agnostic gate** (user's stated preference): replace the
    in-process counter with a filesystem-level lock keyed on
    `sha1(source_h5_path)`. The gate code already runs per-rank inside each
    process; instead of incrementing an in-memory counter, take an exclusive
    `flock` on `<scratch_root>/locks/h5/<sha1>.lock` (or per-stage-output
    lock dir) for the duration of the read context. flock works across
    processes on the same node and across nodes on shared filesystems
    (Lustre/pscratch supports it). This gives one mechanism that enforces
    the rule for `local_affinity`, `mpi`, and `slurm` backends uniformly,
    which matches the "ideally global" goal.
  - **Combine both**: smart partition for the common case, flock-based gate
    for correctness when partition can't help. The gate also covers the
    case where two separate `axon-recon` invocations from different shells
    happen to target overlapping wells — currently undefined behavior.
- **Test idea**: synthetic 4-rank pipeline against 4 fixture h5 files;
  assert from the structured log that no two ranks issued an h5_metadata or
  preprocess_segments phase for the same `source_h5_path` within the same
  time window > a few ms.
- **See also**: discussion 2026-05-12 (spikesort smoke on GPU node, last
  4 datasets). User noted GPU compute dominates the cost on Perlmutter so
  this is a polish item, not a blocker.
