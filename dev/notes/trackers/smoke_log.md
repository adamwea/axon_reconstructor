# Real-data smoke log

Append-only log of smoke tests against **real data** — the bugs they revealed and how those got resolved. Distinct from:
- **Unit / integration tests** (in `tests/`) — run on every commit, work on synthetic fixtures
- **Dry-run smokes** (wired by `dry_run_rollout_plan`) — mechanical wiring confirmations, no real I/O
- **CI pipeline runs** — these are the actual prod runs; this log captures what claude-driven smokes found

**Purpose:**
1. Track what's been actually validated against real cohorts — so the loop doesn't pretend "tests pass" means "feature works on data."
2. Record bug→fix chains so similar regressions can be spotted faster next time.
3. Establish regression baselines (counts, timings, output sizes) that future smokes verify against.

## Entry format

```
### YYYY-MM-DD — <short title>
- **Smoke command**: `axon-recon ...` (or `srun ... axon-recon ...` / `sbatch <script>`)
- **Cohort / data**: `<chip>/<recording-id>/well<NNN>` (DIV / size descriptor) — path under `analyzed_data/` or `dev_outputs/`
- **Plan + slice**: `<plan>.<slice>` — what this smoke is validating
- **Commit at time of smoke**: `<axon_recon hash>` (+ sibling hashes if relevant)
- **Outcome**: ✅ pass | ⚠️ pass-with-findings | ❌ fail
- **What ran end-to-end**: one sentence describing the pipeline path that executed
- **Quantitative result** (when applicable): templates / units / spikes counts, runtime, output size
- **Bugs revealed**:
  - <bug 1> → fixed in commit `<hash>` (<description>)
  - <bug 2> → still pending, tracked in `trackers/issues.md` / `brain/open_questions.md`
- **Diagnostics**: `dev_outputs/<slice>/diagnostics/...` (link to `brain/diagnostics_to_review.md` entry if HARD-gate)
- **Regression baseline established**: <yes/no — if yes, this is the count future smokes target>
- **Status**: closed | followup-pending (<link>)
```

## When to add an entry

**Add when:**
- A login-node smoke on real data completes (pass OR fail)
- A user-initiated `salloc` / `sbatch` run completes and the loop has access to its output dir
- A regression check on a known-good baseline is run (even if uneventful — the "no-change" entry is the value)
- A HARD-gate visual diagnostic was reviewed and approved/rejected

**Don't add for:**
- Dry-run smokes — those are wiring checks, not data smokes
- Unit-test runs — those live in commit messages
- Smokes on synthetic fixtures — those are integration tests

## Pruning

Don't prune. This is a historical record. If an entry becomes superseded, link forward to the entry that replaced it; don't delete.

---

## Entries

### 2026-05-18 — Concat-analyzer rip-out + SLAy aux-tsv sync regression check (known-good baseline)
- **Smoke command**: `axon-recon stages reconstruct --config dev/debug_NERSC/debug.runtime.yml --target-dataset 260326 --target-wells 000 --force-restart` (login-node + interactive GPU mix)
- **Cohort / data**: `260326/M08073/000208/well000` (DIV 36, 80k DMEM)
- **Plan + slice**: phase_roster_cleanup_plan + SLAy assertion / aux-tsv fixes
- **Commit at time of smoke**: ~`5e2b883` (reconstruct `--force-restart` semantics) + SLAy `f7c2173` (aux-tsv sync) + `426ba71` (assertion relax)
- **Outcome**: ✅ pass — established regression baseline
- **What ran end-to-end**: preprocess → spikesort → SLAy merge → recon (with `--force-restart` wiping the templates cache properly) → 176 reconstructed templates on the well
- **Quantitative result**:
  - `spike_clusters.npy`: 377 post-SLAy unit IDs
  - `cluster_KSLabel.tsv` after aux-tsv sync: 287 good + 312 mua = 599 rows
  - Recon merged templates: 176
  - Recon per-unit outputs: 176 (matches good-label count; mua not reconstructed per `unit_label_filter`)
- **Bugs revealed (and pre-existing fixes confirmed working)**:
  - Concat-analyzer plumbing was producing under-counted templates → fixed by flipping `legacy_include_concat: True → False` + hard-set `include_concat=False` in materialize call sites (pre-this-smoke)
  - SLAy `accept_merge` `==`-assertion crashed on big wells (>700 KS units) for same-time intra-cluster collisions → fixed via `>=` relax (commit `426ba71`)
  - SLAy `accept_all_merges` was leaving aux-tsvs stale, causing KS-extractor inner-join to drop merged unit IDs → fixed by syncing `cluster_KSLabel.tsv` / `cluster_Amplitude.tsv` / `cluster_ContamPct.tsv` to match `cluster_group.tsv` post-merge (commit `f7c2173`)
- **Diagnostics**: pre-`brain/diagnostics_to_review.md` schema; record lives in commit messages + `current_state.md` "Known good baseline" section
- **Regression baseline established**: **YES — 176 reconstructed templates on this well/DIV is the regression target for any future recon-stage change.**
- **Status**: closed. Re-validated by all subsequent recon-stage changes via the count check.

### 2026-05-18 — Job 53089489 multi-well GPU sweep (parallelism stress test)
- **Smoke command**: `salloc -N 4 -C gpu -q interactive -t 4:00:00 ...` + `srun ... axon-recon ...` (user-initiated interactive)
- **Cohort / data**: 16 wells across the M08073 80k DMEM family (mix of DIVs)
- **Plan + slice**: parallelism_post_migration_cleanup_plan (slices 3-9 mid-cycle validation)
- **Commit at time of smoke**: pre-2026-05-19 (exact hash not captured at time of run)
- **Outcome**: ⚠️ pass-with-findings — 15 of 16 wells completed cleanly; 1 host-OOM
- **What ran end-to-end**: ds4 full spikesort sweep + 4 SLAy merge retries across the cohort
- **Quantitative result**: 15 wells completed. ds6/well002 host-OOM'd (NOT GPU-OOM — host RAM exhaustion during a merge step)
- **Bugs revealed**:
  - `summary.json` per-pid tmp-rename was producing `FileNotFoundError` floods from concurrent srun writers → fixed in `logging/summary.py`
  - `resolve_inner_worker_count` silently collapsed to `n_jobs=1` in MPI workers when slot was None → fixed to honor yaml/phase hints (test rewritten to lock in fallback)
  - ds6/well002 host-OOM during merge → **NOT YET FIXED** — tracked in `trackers/issues.md` ("merge_SLAy host-RAM ceiling for largest wells"); workaround: skip that well in the dev cohort
- **Diagnostics**: run logs at `/pscratch/sd/a/adammwea/run_logs/53089489_*` (not pruned)
- **Regression baseline established**: NO (but the 15-of-16 success rate is the target for future multi-well sweeps on this cohort family)
- **Status**: closed for the 2 fixed bugs; **followup-pending** for the host-OOM (see `trackers/issues.md`)

### 2026-05-21 — radivojevic_recon first real-data smoke (cluster 67, M08073 well000 DIV 36)

- **Smoke command**: `python` invocation against `radivojevic2023_recon_algo.reconstruct(...)` (sibling repo at `~/dev/pkgs/radivojevic2023_recon_algo/`). NOT axon-recon CLI yet (no axon_recon adapter phase exists yet — that's slice 5 of `radivojevic_recon_algo_plan.md`).
- **Cohort / data**: M08073 / 000208 (DIV 36, MaxTwo, 10 kHz raw) / well000. Used `templates.npy` directly from `spikesort_outputs/sorter_output_snapshot/`. Cluster 67 picked as the highest-amplitude kilosort cluster with ≥200 spikes (amp 73.3 μV, n_spikes=1328).
- **Plan + slice**: `radivojevic_recon_algo_plan` slice 3 sub-step 9 (real-data smoke per USER GATE 3).
- **Commits at time of smoke**: axon_recon `aa2445f`; radivojevic2023_recon_algo `e02d566`.
- **Outcome**: ⚠️ pass-with-findings — algorithm core CALLABLE on real HD-MEA shape data; found a real algorithmic bug along the way (MAD noise estimator collapses on sparse templates).
- **What ran end-to-end**: `reconstruct(trace[266,61], channel_positions_um, sampling_rate_hz=10_000, upsample_factor=2, pixel_um=10.0, noise_estimator='window', noise_window=(0, 15), n_spikes=1328)` → Stage1 → Stage2 → Stage3 → ReconstructionResult.
- **Quantitative result**:
  - Wall time: 0.80 seconds (after tuning Stage 2 to upsample_factor=2 + pixel_um=10.0).
  - noise_std: 0.0015 μV/μs (window-based on first 15 frames of upsampled derivative).
  - Thresholds: 0.0135 / 0.0030 / 0.0015 μV/μs (9/2/1 STD).
  - Peaks detected: step1=97, step2=66, step3=9 → 172 total. 12 of 266 channels have at least one peak (expected — kilosort templates are sparse: only channels near the unit's center are active).
  - Stage 2 skeleton: 10949 pixels across all 120 frames; 933 unique (x, y) pixels in the union.
  - Stage 3 links: direct=52, skeleton_assisted=1, indirect=19 → 72 total inter-frame edges.
- **Bugs revealed**:
  - **MAD noise estimator collapses on sparse kilosort templates** → still pending design follow-up. For sparse inputs where most channels are near zero, `np.median(|x - median(x)|)` → 0, making thresholds 0, making EVERY sample a "peak" (got 659 step-1 peaks in the bad run). WORKAROUND adopted: switched to window-based estimator on quiescent first frames. **Real fix candidates** (next slice):
    1. Per-channel MAD then take median of non-zero per-channel STDs.
    2. Auto-pick a quiescent time window via lowest-energy frames.
    3. Document `noise_estimator="window"` as the recommended default for sparse-template input (kilosort) and `"mad"` for dense input (full-recording STA).
  - **GATE 3 spec assumed `merged_template.npy` files exist on disk** → they don't. Per-unit STAs are inside `gtr.pkl` (axon_velocity-pickled, shifter-only). Loop substituted with raw kilosort `templates.npy`. Documented in `brain/open_questions.md` as 3 questions for user.
- **Diagnostics**: `/pscratch/sd/a/adammwea/dev_outputs/radivojevic_first_run/cluster_67/radivojevic_recon/`
  - `run_summary.json` — full result metadata
  - `peaks_all.tsv` — per-step peak list (channel_idx, time_idx, amplitude)
  - `links.tsv` — per-edge inter-frame links (method, channel_a/b, time_a/b, distance_um, dt_us)
  - `skeleton_union_xy.npy` — union-of-all-frames binary skeleton (183x202 bool)
  - `skeleton_pixels_per_frame.npy` — frame-wise pixel counts
  - `per_channel_peak_counts.npy` — per-channel peak histogram
- **Regression baseline established**: YES — cluster 67 on this DIV / well = 172 peaks / 72 links / 933 unique skeleton xy / 0.80s runtime. Future regressions on same input + same knobs should match within ±5%.
- **Status**: closed (algorithm core proven runnable on real HD-MEA shape data); follow-ups:
  - **MAD vs window noise estimator** design decision pending (next sub-step or USER GATE 4).
  - **Side-by-side plot_recons comparison** still blocked on user's clarification of the `merged_template.npy` data-layout question (open_questions.md).

---

## Smoke #5 — kssynth slice 3b PATH 2 verify (analyzers phase --input-root LOAD path)

- **Date**: 2026-05-21
- **Command**: `shifter --image=adammwea/axon-recon:pipeline-v2 -- axon-recon stages reconstruct.analyzers --config dev/debug_NERSC/debug.runtime.yml --target-dataset 13 --limit-wells 1 --task-backend local_affinity --limit-segments 2 --input-root /pscratch/sd/a/adammwea/analyzed_data/Media_Density_T5_02182026_AR_axon_analysis_AW/ --output-root /pscratch/sd/a/adammwea/dev_outputs/kssynth_slice3b_path2_verify_shifter/`
- **Cohort**: M08073/000208/well000 (DIV 36, 80k DMEM)
- **Commit**: d824d4a (analyzer LOAD-path --input-root extension) + 5580ca2a25fb shifter image rebuild this iteration
- **Outcome**: ✅ SUCCESS
- **Quantitative result**:
  - `discovered_source_count: 2` (discovery side, was already working via commit f33821b)
  - `source_count: 2` (LOAD side — **was 0 before the fix**, now 2 ✅)
  - `recordings_loaded: 2`, `segments.built: 2`, `cache.persisted: 2`
  - `generated_manifest_count: 2`, `source_unit_manifest_count: 2`
  - `recordings_source_dir`: reference path (alternate) — confirms fallback recursion reached alt root
  - `cache_root`: dev_outputs (primary) — confirms cache writes go to dev_outputs, not reference
  - Duration: 271s (~4.5 min) for analyzers build on 2 segments
  - Resource usage: pss_gb=23.99 (estimated 14; flagged as 1.71x over-estimate)
- **Worker count**: cpus_per_task=64 (login-node `--task-backend local_affinity`), max_threads=64, observed_process_max_threads=66. Matches expected for login node with default budget; not a worker-count concern.
- **Bug→fix chain**:
  - **Bug 1**: LOAD-side fallback recursion preserved primary `analyzer_cache_dir` → cache lookup pointed at empty primary cache → recursive load returned source_count=0.
  - **Fix**: commit d824d4a — re-derive `analyzer_cache_dir` relative to `fallback_well_out_dir` in both fallback recursion sites (`load_spikeinterface_analyzers` line 2890 + `iter_spikeinterface_analyzers` line 3198).
  - **Bug 2 (env-only, not code)**: conda env lacks Maxwell HDF5 plugin → spikeinterface recording-load failed when first smoke ran outside shifter. Workaround: run in shifter (which has the plugin baked in).
- **Diagnostics**: `/pscratch/sd/a/adammwea/dev_outputs/kssynth_slice3b_path2_verify_shifter/...` analyzer caches + per-source unit manifests.
- **Baseline established**: NO — this was a verification smoke (--limit-segments 2). Full-segments baseline pending GATE 1 step 1 full run.
- **Status**: closed (PATH 2 fix verified end-to-end in shifter); follow-up: PRE-DIAGNOSTIC GATE 1 brought back to user for full-segments-run approval.

---

## Smoke #6 — Radivojevic apples-to-apples diagnostic on unit_0598 (kssynth-merged template)

- **Date**: 2026-05-21
- **Command**: `python -c "..."` (radivojevic.reconstruct on kssynth's unit_598 merged_template) — login-node, conda env (axon_recon), 1 worker.
- **Cohort**: M08073/000208/well000 DIV 36 unit_0598 (9-branch reference)
- **Commit**: d824d4a (PATH 2 LOAD-path) + 5ec8764/3f34405 (salloc setup) + this iteration's diagnostic-rendering work
- **Outcome**: ✅ SUCCESS
- **Quantitative result**:
  - Input: kssynth merged_template (13439, 70) at 10 kHz raw + merged_channel_locations (13439, 2)
  - Stage 1: 5443 + 8142 + 31624 = 45209 total peaks
  - Stage 2: skeleton built (see result.pkl)
  - Stage 3: links built (see result.pkl)
  - Wall: 162.9s
- **Worker count**: n_jobs=1 (radivojevic algorithm is single-threaded). NOT a worker-count concern for this smoke.
- **Diagnostics**: HARD-gate filed in `dev/notes/brain/diagnostics_to_review.md` — `comparison_unit_598.png` side-by-side composite at `dev_outputs/radivojevic_apples_to_apples/unit_598_radivojevic/`.
- **Status**: closed (algorithm + comparison shipped end-to-end); user reviewing HARD-gate diagnostic. Follow-ups: kssynth slice 7+8 (analyzer-policy parity + upsample integration); resources_profiles slice 6 (64-vs-128 procs); plot_recons adapter for true apples-to-apples renderer parity (deferred).

---

## Smoke #7 — Radivojevic 2-stage paper-alignment refactor verification on unit_598

- **Date**: 2026-05-22
- **Command**: `python /pscratch/sd/a/adammwea/dev_outputs/radivojevic_paper_2stage/run_smoke.py` — login-node, conda env (axon_recon), 1 worker.
- **Cohort**: M08073/000208/well000 DIV 36 unit_598 (9-branch reference; same as smoke #6)
- **Commit**: radivojevic sibling repo `1c22138` (paper-alignment refactor; collapse stage_2+stage_3 → paper Stage 2; default `use_pair_averaged_skeleton=True`; add Stage1Trace / Stage2Trace)
- **Outcome**: ✅ SUCCESS — diagnostic plot written.
- **Quantitative result**:
  - Input: kssynth merged_template (13439, 70) at 10 kHz raw + merged_channel_locations (13439, 2)
  - Stage 1 (paper Step 1/2/3 — n_std 90/20/10 stopgap scale, paper geometry 50μm/100μm): 173 + 424 + 881 = 1478 cumulative peaks
  - Stage 2 (paper Step 1/2/3 — direct/skel-assisted/indirect, 100/200/400 μm radii, on-demand pair-averaged skel): 991 + 12 + 31 = 1034 links
  - Wall: 180.12s
- **Worker count**: n_jobs=1 (radivojevic single-threaded; numpy ops fan out internally; observed ~210% CPU).
- **Diagnostics**: SOFT-gate filed in `brain/diagnostics_to_review.md` — `unit_598_two_stage_diagnostic.png` at `dev_outputs/radivojevic_paper_2stage/diagnostics/`. Plot shows 4×2 grid (paper Stage 1 / Step 1/2/3/all on top, paper Stage 2 / Step 1/2/3/all on bottom).
- **Status**: closed (refactor verified; structural plot delivered). Open known issue: selected channels still cluster centrally (noise-underestimate symptom). Follow-up: axon_recon slice 11 raw-recording noise wiring (next session) → re-run with paper n_std 9/2/1 against per-channel raw noise.
