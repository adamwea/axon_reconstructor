# Metrics — what "better" means + rollback triggers

The value/motivation surrogate. Per the brain-theory: refinement has no natural endpoint unless we define one — agents either fiddle forever or quit arbitrarily. Each refinement target needs an explicit metric + baseline + improvement direction + rollback trigger.

**Reading rule for the loop**: before claiming a slice "shipped" against a refinement target, measure the relevant metric. If it WORSENED versus baseline, the slice MUST be rolled back (git restore) + escalated, not committed.

**Writing rule**: loop UPDATES the "current value" field after each smoke that establishes a measurement. NEW metric entries (= new refinement target) require user approval via `open_questions.md` multiple-choice.

---

## Schema

Each metric is:
```
### <metric-id> — <human label>
- **Refinement target**: <which objective from brain/objectives.md this serves>
- **Definition**: <one sentence specifying exactly what is measured>
- **Direction**: minimize | maximize | exact-match | within-tolerance
- **Baseline**: <value> (established <date>, anchored by <trusted output / commit>)
- **Current**: <value> (measured <date>, smoke ID)
- **Target**: <value or "monotonic improvement" or "stay within X of baseline">
- **Rollback trigger**: <condition that triggers automatic git restore + escalate>
- **Measurement command**: <how to reproduce; the smoke command or script that computes it>
```

---

## Active metrics

### M-001 — Reconstructed-template count (recon-stage regression)
- **Refinement target**: O1 (kssynth integration) — protects the recon-stage baseline as kssynth replaces predecessors
- **Definition**: count of per-unit `merged_template.npy` files produced by a full reconstruct stage run on the anchor well
- **Direction**: exact-match (regression check)
- **Baseline**: **176** (2026-05-18, anchor TR-001, commits `5e2b883` + SLAy `f7c2173` + SLAy `426ba71`)
- **Current**: 176 (unchanged; last measured 2026-05-18)
- **Target**: 176 (exact) on `260326/M08073/000208/well000` DIV 36 with no `--limit-*` flags
- **Rollback trigger**: count drops below 176. Whatever slice caused the drop gets `git restore`d; failure escalated.
- **Measurement command**:
  ```bash
  axon-recon stages reconstruct --config dev/debug_NERSC/debug.runtime.yml \
    --target-dataset 260326 --target-wells 000 --force-restart
  # then: count files at <output>/recon_outputs/units/*/merged_template.npy
  ```

### M-002 — Test-suite passing rate (broad regression)
- **Refinement target**: all O1-O5 (any refactor)
- **Definition**: `pytest src/axon_recon/pipeline/tests/` pass count
- **Direction**: monotonic non-decrease (allowed exception: tests deleted as part of an explicit cleanup slice)
- **Baseline**: 457 passed, 14 pre-existing failures noted (parallelism_post_migration_cleanup_plan §6 documents the 14)
- **Current**: 457 (last broad run 2026-05-19)
- **Target**: monotonic improvement OR explicit-deletion-with-rationale
- **Rollback trigger**: net NEW failures (any test that previously passed now fails) AND the slice didn't authorize the deletion. Roll back + escalate.
- **Measurement command**: `pytest src/axon_recon/pipeline/tests/ -q`

### M-003 — Worker-count fidelity (resources.profiles elimination)
- **Refinement target**: O5
- **Definition**: in a smoke under `srun -n 1 -c K ...`, the logged `phase_parallelism event` actual `cpus_per_task` == K (NOT the YAML clamp)
- **Direction**: exact-match (per phase per smoke)
- **Baseline**: currently INCORRECT (user-flagged 2026-05-21 — profile clamps at YAML default 16 even when srun gives 128). The baseline is the bug; the target is the fix.
- **Current**: BUG ACTIVE — profile clamping still occurs (resources_profiles_elimination_plan slice 0+1+2 will fix)
- **Target**: actual cpus_per_task in `phase_parallelism event=` matches the srun-supplied value, no clamping at YAML default, with no `--profile` flag in the invocation
- **Rollback trigger**: post-fix, any smoke that shows clamping returning. Roll back the offending slice.
- **Measurement command**: run any reconstruct smoke under srun + grep logs for `phase_parallelism event=` + compare `cpus_per_task` field against `SLURM_CPUS_PER_TASK` env

### M-004 — Cross-algorithm reconstruction similarity (Radivojevic vs axon_velocity_gtrs)
- **Refinement target**: O3
- **Definition**: 5-tuple of soft-invariant pass/fail per unit (size, branching, velocity, soma localization, channel-set overlap) per `brain/trusted_outputs.md` DC-001
- **Direction**: maximize the fraction of units where all 5 soft invariants pass
- **Baseline**: NOT YET MEASURED — gated on the apples-to-apples comparison shipping (O3 DOD step 3+4)
- **Current**: N/A
- **Target**: ≥70% of units pass all 5 soft invariants OR user explicitly accepts a lower bar per the algorithm's known divergence
- **Rollback trigger**: Radivojevic algorithm changes that cause the % to drop AND the change wasn't a user-approved algorithm tuning. Roll back.
- **Measurement command**: TBD — will be authored during O3 DOD step 4. Loop produces `comparison.json` per unit with all 5 invariant computations; aggregator script reports the pass-rate.

### (Proposed metrics — phase zero / user adds)

*(empty — Z1 may surface additional refinement targets that warrant metrics. User adds via `open_questions.md` multiple-choice + writes the entry.)*

---

## Retired / superseded metrics

*(empty — keep this section as history when metrics get superseded by better measurements; don't delete superseded entries inline)*

---

## How auto-rollback works

When the loop ships a slice + measures a metric + finds it degraded:

1. **Stop**. Don't continue to the next slice.
2. **`git restore` the slice's changes** (the commit is fine to keep as history; restore the working tree to HEAD~1 OR the last green-metric commit).
3. **Append a smoke_log entry** documenting: which slice, which metric, baseline vs degraded value, why this was unexpected.
4. **File a `🛑 METRIC REGRESSION` entry in `open_questions.md`** with multiple-choice options for how to proceed (retry with different approach / accept regression with user-documented rationale / pause for user input).
5. **PAUSE** until user resolves.

This is the brain's "checkpoint and rollback" discipline — the only thing that prevents refinement from accreting subtle degradations into the baseline.
