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

## Radivojevic slice 3 USER GATE 2 — ✅ RESOLVED 2026-05-21

**User chose option (B)**: ship `core/derivatives.py` with `compute_time_derivative(trace, *, dt_us)` returning μV/μs as a thin helper now; promote to a stage-1 orchestrator `detect_step1_peaks(trace, *, sampling_rate_hz, upsample_factor=10, noise_estimator='mad'|'window', n_std=9.0)` AFTER Steps 2 + 3 land and the orchestrator's full API is clear. Loop's recommendation accepted verbatim.

Loop can now proceed with sub-step 4 (Step 2 — confined 2-STD thresholding within 50 μm radius of step-1 peaks) and continue accumulating Stage 1 sub-steps. After Steps 2 + 3 land, loop opens a small follow-up to refactor the three thresholding helpers into the orchestrator + flag this entry for deletion.

**Original gate body preserved below for archeology:**

## Radivojevic slice 3 USER GATE 2 — original body

Per the slice 3 plan ("every 2-3 sub-steps; switch to another plan
until user reviews"), the loop has shipped 3 concrete algorithm
sub-steps in the sibling repo at `~/dev/pkgs/radivojevic2023_recon_algo/`:

1. **`core/upsampling.py`** (commit `5b06fbf`) — Whittaker-Shannon
   sinc-kernel interpolation. Default `upsample_factor=10` matches
   the paper's 10x ratio; device-agnostic via `compute_upsampled_rate_hz`
   helper. 15 tests.
2. **`core/noise_estimation.py`** (commit `3f363d7`) — two estimators:
   paper-faithful window-based + robust MAD (Median Absolute Deviation
   / 0.6745). 16 tests.
3. **`core/adaptive_thresholding.py`** (commit `6403403`) — Step 1 of
   stage 1: planar |signal| >= 9*noise_std cutoff + per-electrode
   local-max detection. Returns list[PeakDetection(channel_idx,
   time_idx, amplitude)]. Sanity test confirms zero false positives on
   10k Gaussian samples (matches paper Fig 5B). 17 tests.

Package now 53 tests total, all green.

**Open question for user review** (the only design ambiguity surfaced
so far): the algorithm operates on the **time derivative** of the
upsampled trace (μV/μs). I have NOT folded the derivative step into
any single utility — the working assumption is that the caller computes
`np.diff(upsampled, axis=-1)` before calling
`find_local_peaks_above_threshold`. Three plausible places to put it:

(A) **Caller computes** — current design. Pros: simple, explicit,
keeps each utility single-purpose. Cons: easy to forget the unit
conversion.

(B) **Single-purpose helper** in `core/derivatives.py` exposing
`compute_time_derivative(trace, *, dt_us)` returning μV/μs. Pros:
encapsulates the unit conversion; one place to verify sign convention.
Cons: trivial wrapper around np.diff.

(C) **Fold into a stage-1 orchestrator** — `core/stage_1.py` exposing
`detect_step1_peaks(trace, *, sampling_rate_hz, upsample_factor=10,
noise_std=...)`. Pros: callers don't need to remember the upsample →
diff → threshold pipeline. Cons: hides the noise estimation step that
the caller should also customize.

**Recommendation**: (B) for now (a thin helper that makes the unit
conversion explicit), upgrade to (C) only after Step 2 + Step 3 land
and the orchestrator's API is clear.

When ready to resume slice 3 sub-step 4+, the loop will pick up Step 2
(confined 2-STD thresholding within 50 μm radius of step-1 peaks).
Stage 1 step 3 + stages 2 + 3 follow.

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
