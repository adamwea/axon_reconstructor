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
  - <bug 2> → still pending, tracked in `trackers/issues.md` / `memory/open_questions.md`
- **Diagnostics**: `dev_outputs/<slice>/diagnostics/...` (link to `memory/diagnostics_to_review.md` entry if HARD-gate)
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
- **Diagnostics**: pre-`memory/diagnostics_to_review.md` schema; record lives in commit messages + `current_state.md` "Known good baseline" section
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
