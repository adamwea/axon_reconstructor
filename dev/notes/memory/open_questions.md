# Open questions

TBD decisions awaiting user input or empirical data. Each entry has a clear resolution criterion. When resolved, move the conclusion to `current_state.md`, `guardrails/`, or a plan; delete the entry from here.

## Awaiting empirical data (deferred until plan reaches the relevant slice)

- **Two-halves split granularity**: temporal midpoint is what UMPy expects. Could split finer for more same-neuron pairs per unit, but UMPy shape is hardcoded to `(..., 2)`. Decide after `unitlink` v1 results.
- **Per-chip match threshold tuning**: default `match_threshold: 0.5` from UMPy may be too permissive for HD-MEA. Add per-group calibration in `unitlink` v3? Wait for v1 + v2 empirical data.
- **Network-scan inclusion as default**: decide after `unitmatch_phase_plan.md` slice 7's measurement of marginal gain.
- **Both network-scan types in unitmatch v2**: v2 picks ONE type (lean clustered variant). The sparse variant may join in v3 if marginal gain measurement justifies it.
- **DeepUnitMatch HD-MEA training**: `unitlink` v2 wrapper supports it; training a HD-MEA model is its own project. Defer.
- **`init` / `cleanup` stage scope**: v1 = one phase each (`copy_src_to_scratch` / `wipe_src_scratch`); grow organically. Stages disabled by default for now but must work.
- **`concat_binary` resource class** after consolidation: keep spikesort-side budget. Plan §6 §3.
- **bombcell / SLAy code deletion timing**: never delete from spikesort code, just disable. User: "I think in the future we will only use the recon stage versions if we successfully implement them as I imagine, maybe then we delete them. but for now, just disable them."
- **`plot_raster_threshold` quality fix design**: needs design-doc-level thinking about colormap / per-segment channel toggling visualization. Defer.
- ~~**Dashboard slice 7 tertiary-grouping UX**~~ — RESOLVED 2026-05-21
  via the pre-overnight clearance ("YAML-configurable tertiary mode,
  default small-multiples"). Slice 7 SHIPPED with both render modes
  available (`small_multiples` + `hierarchical_labels`) via radio
  control + YAML default. Download path threads the choice through
  (commit `ab56f64`). Both modes are reachable from the UI and the
  image export. **Marked for deletion** at next audit-pass.

## Per-slice empirical findings

- **resources.profiles elimination — per-srun-flag fallback when slot is missing**: when `--cpus-per-task` is passed on the command line, does the resolver use that directly, or compute from `os.sched_getaffinity(0)`? Both are reasonable; pick during the implementation slice. See `trackers/tech_debt.md` §"Minimize / eliminate `resources.profiles`".

- **Auto-restart with chip-well-group phase scope** (`unitmatch` phase): the auto-restart logic walks `phase_sequence` per target. For chip-well groups, the "target" is a group, not a (dataset, well) pair. Verify the logic generalizes when the unitmatch phase lands.

## Radivojevic slice 1 user-gate review — ✅ RESOLVED 2026-05-21

User answered all 6 questions in the 2026-05-21 walkthrough. Slice 3 (core algorithm impl) is UNGATED — loop can proceed when its queue reaches the Radivojevic plan.

**Answers locked in:**

1. **Paper identity**: ✅ confirmed — *Radivojevic & Rostedt Punga (2023), Functional imaging of conduction dynamics in cortical and spinal axons, eLife 12:e86512, DOI 10.7554/eLife.86512*.
2. **Clean-room approach**: ✅ confirmed — no public code; clean-room re-implementation from methods + figures.
3. **🔴 HIGH-IMPACT — averaged template sufficient**: ✅ confirmed via methods-mining doc evidence: paper's Steps IV+V operate on the averaged "axonal electrical image" only. axon_recon's `merged_template.npy` is the input-equivalent. Add a minimum-n_spikes sanity check at slice 3 start (skip units with <50 spikes, mirroring paper's implicit 100-200-trials averaging assumption).
4. **Input compat map**: ✅ confirmed (modulo Q3 RESOLVED). Inputs = `merged_template.npy` + `merged_channel_locations.npy` + `sampling_rate_hz` (read from analyzer manifest, NEVER hardcoded).
5. **Hyperparameter defaults**: ✅ confirmed — Step 1 = 9 STD, Step 2 = 2 STD / 50 μm, Step 3 = 1 STD / 100 μm, Direct interconnect = 100 μm, Skeleton-assisted = 200 μm. **AMENDED**: upsampling is parameterized as `upsample_factor: 10` (integer ratio), NOT an absolute Hz target. Paper's "200 kHz" was 10× their 20 kHz input; our pipeline runs on multiple devices so we scale per-recording: MaxTwo (10 kHz raw → 100 kHz upsampled), MaxOne (20 kHz → 200 kHz, matches paper). All 7 thresholds exposed as YAML knobs; slice 6 will do an empirical sweep on Step 1 ∈ {7, 9, 11} STD and Step 3 ∈ {0.5, 1, 1.5} STD if first-pass results warrant tuning.
6. **Phase + sibling-package name**: ✅ `radivojevic_recon` — matches sibling-package dir; parallels `axon_velocity_gtrs` naming convention.

**Critical project-level clarification logged from this walkthrough**: axon_recon runs on multiple MaxWell devices (MaxTwo @ 10 kHz, MaxOne @ 20 kHz, others future). Sample rate is per-recording; the `metadata_get` preprocess step + `dev/debug_NERSC/debug.data.yml` are the authoritative sources of truth. **NEVER hardcode sample rate** in any analysis/recon phase. The Radivojevic algorithm summary doc (`refs/radivojevic2023_algorithm_summary.md`) updated to reflect this. Captured as auto-memory under `project-axon-recon-device-diversity` for future sessions.
