# Time budget — sans_bombcell_rerun

Track requested walltime vs observed elapsed per stage, so we can tighten
sbatch `--time` for future reruns. Shorter walltime requests improve
backfill eligibility on NERSC's shared QoS → shorter queue waits.

## Empirical per-phase elapsed (from prior summary JSONs)

Mined from `*_summary.json` files across the analyzed_data tree
(1,940 phase summaries, 74 wells minus some early failures). Falls back
through `phase_elapsed_s` → max(`phase_timing_s`) → `timing.duration_seconds`
→ `resource_usage.wall_time_s`.

### preprocess phases
| phase                | n  | median (s) | p95 (s) | max (s) |
|----------------------|----|-----------:|--------:|--------:|
| preprocess_segments  | 79 |     20.47  |   29.86 |   53.05 |
| save_rec_metadata    | 82 |      3.94  |   11.18 |   20.74 |
| **stage rollup**     |    |   **24.4** |  **41** |  **74** |

### spikesort phases
| phase                   | n  | median (s) |  p95 (s) |  max (s) |
|-------------------------|----|-----------:|---------:|---------:|
| bootstrap_concat_binary | 74 |      99.6  |    155.6 |    271.7 |
| sort                    | 74 |     385.4  |    654.8 |   1232.7 |
| snapshot_sorter_output  | 74 |       0.2  |      0.2 |      0.2 |
| concat_analyzer         | 74 |      72.7  |    135.2 |    174.8 |
| merge_SLAy              | 38 |    4537.4  |   8860.5 |  11478.0 |
| restore_sorter_output   | 74 |       2.0  |      4.0 |      6.3 |
| **stage rollup w/ slay**|    |  **~85 min** | **~163 min** | **~213 min** |
| **stage rollup no slay**|    |  **~9.6 min** | **~16 min** | **~28 min** |

Note: merge_SLAy ran on only 38/74 prior wells; for any "full" job the
slay phase dominates. Median ~76 min, p95 ~148 min, max ~191 min.

### reconstruct phases
| phase                     | n  | median (s) | p95 (s) | max (s) |
|---------------------------|----|-----------:|--------:|--------:|
| analyzers                 | 75 |     435.5  |   651.9 |   781.8 |
| extract_partial_templates | 72 |     377.4  |   887.9 |  1117.6 |
| build_templates           | 71 |      14.1  |    31.9 |    38.5 |
| plot_templates_v2         | 71 |       7.0  |    14.5 |    18.2 |
| report_templates          | 71 |      28.0  |    63.0 |    81.3 |
| generate_gtrs             | 71 |      88.3  |   129.5 |   146.4 |
| plot_recons               | 71 |     231.3  |   385.6 |   422.1 |
| plot_branch_propagations  | 71 |       4.7  |    10.6 |    11.1 |
| plot_branch_velocities    | 71 |       3.7  |     6.2 |     7.0 |
| plot_unit_summary         | 71 |      12.9  |    21.0 |    23.2 |
| report_recons             | 63 |       8.7  |    12.7 |    13.7 |
| report_recon_grid         | 63 |      15.4  |    22.5 |    24.0 |
| report_full_chip_layout   | 63 |       1.2  |     1.6 |     2.3 |
| report_summaries          | 63 |      29.0  |    41.4 |    45.3 |
| clear_templates_cache     | 63 |       5.0  |     8.9 |    16.2 |
| **stage rollup**          |    |  **~21 min** | **~38 min** | **~50 min** |

## Recommended sbatch walltimes (rule: round-up p95 × 1.5)

| Stage           | Template                | Currently | **Recommended** | Rationale |
|-----------------|-------------------------|-----------|-----------------|-----------|
| preproc         | `preproc.sbatch`        | 04:00:00  | **00:30:00**    | p95 = 41 s + container start + srun overhead → 30 min is very safe |
| spikesort_full  | `spikesort_full.sbatch` | 06:00:00  | **04:00:00**    | p95 ≈ 163 min (merge dominated); 4 h gives ~50% headroom |
| merge_SLAy      | `merge_SLAy.sbatch`     | 06:00:00  | **04:00:00**    | merge p95 ≈ 148 min, max 191 min → 4 h fits max + 25% |
| recon           | `recon.sbatch`          | 06:00:00  | **02:00:00**    | p95 ≈ 38 min → 2 h gives 3× safety |

Backfill bonus: NERSC's shared qos backfill picks shorter jobs to slot
between large reservations. Dropping preproc 4 h → 30 min materially
improves their start time.

## Observed runtimes (this rerun)

Format: `<stage> ds<idx>/<well> queued=<wait> elapsed=<run>`. Pull from
`sacct -u adammwea -S today -X -P -o JobID,JobName,State,Submit,Start,Elapsed`.

### preproc (2026-05-15)
- ds4/well000 queued≈60 min  elapsed=00:01:35
- ds4/well001 queued≈63 min  elapsed=00:01:28
- ds4/well002 queued≈100 min elapsed=00:01:24
- ds4/well003 queued≈102 min elapsed=00:01:18
- _ds4/well004 pending_
- _ds4/well005 pending_

Observation: matches empirical median (20 s pipeline work + ~60 s
container/srun overhead). Confirms 30 min walltime is plenty.

### spikesort_full (GPU)
- _all pending_

### merge_SLAy (GPU)
- _all pending_

### recon (CPU)
- _all pending_

## Shrink-walltime helpers

`scontrol update` accepts `TimeLimit` on PENDING jobs (it does NOT accept
`ThreadsPerCore`, `ReqCPUs`, `CpusPerTask` — those are fixed at submit).

Drop the whole pending queue to the recommended times:

```bash
shrink_pending() {
  # usage: shrink_pending <name_glob> <HH:MM:SS>
  local pat="$1" newtime="$2"
  squeue -u "$USER" -h -t PD -o "%i %j" \
    | awk -v p="$pat" '$2 ~ p {print $1}' \
    | xargs -r -I{} scontrol update jobid={} TimeLimit="$newtime"
}

shrink_pending '^preproc_'    00:30:00
shrink_pending '^spikesort_'  04:00:00
shrink_pending '^merge_SLAy_' 04:00:00
shrink_pending '^recon_'      02:00:00
```

Verify:
```bash
squeue -u adammwea -o "%.10i %.20j %.10l %.10T %r"
```

## Quick query helpers

```bash
# all completed today with submit/start/elapsed
sacct -u adammwea -S today -X -P -o JobID,JobName,State,Submit,Start,Elapsed | grep COMPLETED

# walltime efficiency for a single job
seff <jobid>
```

## Future-template fixes (CANNOT be applied to pending jobs)

If/when this rerun completes and you re-submit, the templates should also:
- Add `#SBATCH --threads-per-core=1` so the allocation reserves physical
  cores instead of logical (avoids the SMT2 / `-c N` ambiguity entirely).
- Consider bumping merge_SLAy / spikesort_full memory to `--mem=120G
  --cpus-per-task=64` (½-node slice, same 1 GPU) if observed peak RSS
  approaches the 56 G ceiling. Check with `seff` after first slay run.
