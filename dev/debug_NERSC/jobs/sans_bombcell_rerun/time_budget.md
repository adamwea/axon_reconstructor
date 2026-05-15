# Time budget — sans_bombcell_rerun

Track requested walltime vs observed elapsed per stage, so we can tighten
sbatch `--time` for future reruns. Shorter walltime requests improve
backfill eligibility on NERSC's shared QoS → shorter queue waits.

## Current walltime requests

| Stage         | Template                 | Requested | Notes |
|---------------|--------------------------|-----------|-------|
| preproc       | `preproc.sbatch`         | 04:00:00  | CPU shared |
| spikesort_full| `spikesort_full.sbatch`  | 06:00:00  | GPU shared, sort + analyzer + merge_SLAy |
| merge_SLAy    | `merge_SLAy.sbatch`      | 06:00:00  | GPU shared, slay phase only |
| recon         | `recon.sbatch`           | 06:00:00  | CPU shared, --force-restart |

## Observed runtimes

Format: `<stage> ds<idx>/<well> queued=<wait> elapsed=<run>`. Pull from
`sacct -u adammwea -S today -X -P -o JobID,JobName,State,Submit,Start,Elapsed`.

### preproc (2026-05-15)
- ds4/well000 queued≈60 min  elapsed=00:01:35
- ds4/well001 queued≈63 min  elapsed=00:01:28
- ds4/well002 queued≈100 min elapsed=00:01:24
- ds4/well003 queued≈102 min elapsed=00:01:18
- _ds4/well004 pending_
- _ds4/well005 pending_

Observation: preproc on a fresh ds4 (DIV 12) well completes in <2 min when
the YAML's enabled phases are just `save_rec_metadata + preprocess_segments`.
NERSC's 1h AccrueTime gate dominates wait; first two ran the instant they
became eligible.

### spikesort_full (GPU)
- _all pending — first one will populate this section once it finishes_

### merge_SLAy (GPU)
- _all pending_

### recon (CPU)
- _all pending_

## Recommendation when filling this in

After each new finished job:
1. Append `dsX/wellY queued=<wait> elapsed=<run>` to the relevant section.
2. If a stage's max observed elapsed << requested walltime, propose a new
   `--time` value in this table:
   - request = round-up(max_observed * 1.5, nearest 30 min)
   - i.e. preproc has run <2 min so far; if that holds across all 6 ds4
     wells, drop `preproc.sbatch` to `--time=00:30:00`.

## Quick query helpers

```bash
# all completed today with submit/start/elapsed
sacct -u adammwea -S today -X -P -o JobID,JobName,State,Submit,Start,Elapsed | grep COMPLETED

# walltime efficiency for a single job
seff <jobid>
```
