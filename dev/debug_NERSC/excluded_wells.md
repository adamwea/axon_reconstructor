# Wells excluded from runtime (Media_Density_T5_02182026_AR)

Running ledger of `(dataset, well)` pairs whose `include_in_runtime` is
deliberately `false` in `debug.data.yml`, plus the reason why and the
unblock criterion. Pair this with `axon-recon status --config
dev/debug_NERSC/debug.runtime.yml` to confirm at a glance that the
excluded wells no longer appear as `missing_wells`.

Each entry is meant to be revisitable: when the unblock criterion is
met, flip `include_in_runtime` back to `true` in `debug.data.yml`,
re-run the affected stages, and delete the corresponding entry here.

---

## Dataset 1 — `260224/M06804/000032` (DIV 6)

### well003 — excluded 2026-05-13
- **Symptom**: SLAy crashed with
  `ValueError: With n_samples=0, test_size=0.2 and train_size=None,
  the resulting train set will be empty.` See SLAy
  `autoencoder.py:300` (`train_test_split`).
- **Root cause**: bombcell labeled all 63 sorted clusters as
  `noise` (59) or `mua` (4). SLAy filters to bombcell-qualifying
  units (`good` + `non_soma_good`); with zero such units, the
  autoencoder receives no training samples. Spikesort produced
  ~25k spikes; the data isn't empty, just doesn't pass bombcell's
  default thresholds at DIV 6.
- **Pipeline now short-circuits gracefully**: as of commit `d537ff5`,
  merge_SLAy writes a `status=skipped, reason=no_qualifying_units`
  stub when bombcell reports 0 qualifying units (this well would
  now show up under `axon-recon status --verbose` as a
  `merge_SLAy=no_qualifying_units(ok)` skip rather than a crash).
  We're still excluding it from runtime because **the well has
  zero analysis-eligible units** — letting it flow through
  reconstruct + analysis would just produce empty downstream
  artifacts and confuse the dashboard.
- **Unblock criterion**: bombcell parameters retuned for young
  (DIV ≤ ~10) cultures and the well produces ≥ 1 qualifying unit,
  OR we decide DIV 6 wells with sparse signal are not part of the
  scientific scope and stay excluded permanently.
- **See also**: roadmap entry "Bombcell tuning for young / sparse
  cultures".

### well005 — excluded 2026-05-13
- **Symptom**: identical SLAy crash to well003.
- **Root cause**: bombcell labeled all 539 sorted clusters as
  `noise` (530), `mua` (5), or `non_soma_mua` (4). **1.5 M spikes
  in this well** — there's lots of activity, but the quality
  metrics bombcell uses (refractory violations, presence ratio,
  somatic amplitude, etc.) reject every cluster at this DIV.
- **Pipeline now short-circuits gracefully** — same `d537ff5` fix
  as well003 above.
- **Unblock criterion**: same as well003.

## Dataset 8 — `260316/M08073/000165` (DIV 26)

### well004 — excluded 2026-05-13
- **Symptom**: spikesort never completed for this well. Multiple
  rerun attempts over 2 days; the only output is
  `cache/bootstrap_concat_binary/` from the very first sbatch and
  no `sorter_output/`, no `spikesort_summary.json`. The well is
  simply skipped by every run we attempt.
- **Root cause**: suspected interaction between `--force-restart`'s
  known partial-cleanup gap (see tracker entry
  "`--force-restart` does not reliably clean prior artifacts") and
  the leftover `bootstrap_concat_binary` from the very first
  failed sbatch. Possibly also a per-rank crash that orphans this
  well in the MPI work assignment — we never observed a real
  error from the spikesort run for this well, just absence of
  output.
- **Unblock criterion**: a clean rerun of just this well that
  actually produces `sorter_output/` and `merge_SLAy/merge_stage_summary.json`.
  Probably needs `rm -rf
  /pscratch/sd/a/adammwea/analyzed_data/Media_Density_T5_02182026_AR_axon_analysis_AW/
  Media_Density_T5_02182026_AR/260316/M08073/AxonTracking/000165/well004/spikesort_outputs/`
  first to bypass the `--force-restart` partial-cleanup issue,
  then `axon-recon stages spikesort --target-dataset 8
  --target-wells well004 --task-backend mpi`. Hold until after the
  current reconstruct backlog finishes — no reason to compete for
  GPU time with productive wells.

---

## Procedural notes

- Excluding via `include_in_runtime: false` removes the well from
  scope for ALL stages (preprocess, spikesort, reconstruct,
  analysis, dashboard). It does not delete on-disk artifacts;
  re-enabling the well later picks up any existing outputs.
- The `axon-recon status` default table will still list these
  wells in "skipped_wells" if their existing stage markers say
  `status=skipped` — that's expected. Once they're excluded from
  the data config they no longer count toward the
  `wells (ok/total)` denominator.
- Keep this file in sync with the actual `include_in_runtime: false`
  entries in `debug.data.yml`. If they drift, the data.yml is
  authoritative.
