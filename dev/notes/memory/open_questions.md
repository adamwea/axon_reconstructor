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

## 🛑 PRE-DIAGNOSTIC GATE 1 — confirm plan before next radivojevic diagnostic attempt (USER MANDATE 2026-05-21)

**Per USER INJECTION 2026-05-21 (B2) — the next 3 diagnostic-generation iterations are GATED.** Loop does NOT begin diagnostic generation until the user explicitly confirms the plan below.

**Background**: First Radivojevic SOFT-gate diagnostic failed on 3 user-explicit requirements because the loop improvised around friction instead of stopping to ask. User feedback: "yea i think for generating these next few diagnostics, i need the loop to stop and ask more carefully i guess. Include all your notes for the loop to read, and we'll try again." Full root-cause analysis is in `current_state.md` USER INJECTION (B3).

**Proposed execution plan for the next diagnostic attempt** (apples-to-apples radivojevic vs axon_velocity_gtrs comparison via existing plot_recons):

1. **Run kssynth slice 3b heavy** on M08073/well000:
   ```
   axon-recon stages reconstruct.analyzers --config dev/debug_NERSC/debug.runtime.yml \
     --target-dataset 260326 --limit-wells 1 --task-backend local_affinity \
     --input-root /pscratch/sd/a/adammwea/analyzed_data/Media_Density_T5_02182026_AR_axon_analysis_AW/ \
     --output-root /pscratch/sd/a/adammwea/dev_outputs/radivojevic_apples_to_apples/
   axon-recon stages reconstruct.kssynth --config dev/debug_NERSC/debug.runtime.yml \
     --target-dataset 260326 --limit-wells 1 --task-backend local_affinity \
     --force-enable kssynth \
     --input-root /pscratch/sd/a/adammwea/analyzed_data/Media_Density_T5_02182026_AR_axon_analysis_AW/ \
     --output-root /pscratch/sd/a/adammwea/dev_outputs/radivojevic_apples_to_apples/
   ```
   ETA ~5-15 min for analyzers cache + ~30s for kssynth. Produces `synth_sorter_output/per_unit/unit_<N>/merged_template.npy` + `merged_channel_locations.npy` for every post-merge unit.

2. **Identify the high-branch unit**. Earlier scan identified `unit_0598` as the 9-branch reference. The kssynth-produced per_unit/ directory's unit ID may or may not match the reference's `unit_0598` numbering — needs cross-check. **If the mapping is unclear, STOP AND ASK** rather than substituting a different unit.

3. **Run radivojevic on the chosen unit's merged_template**:
   ```python
   merged_template = np.load(".../per_unit/unit_<N>/merged_template.npy")
   channel_positions = np.load(".../per_unit/unit_<N>/merged_channel_locations.npy")
   result = radivojevic2023_recon_algo.reconstruct(
       merged_template,  # shape (n_active_channels, n_samples)
       channel_positions_um=channel_positions,
       sampling_rate_hz=10_000.0,  # MaxTwo
       upsample_factor=2,
       pixel_um=10.0,
       noise_estimator='window',
       noise_window=(0, 15),
   )
   ```
   Tune knobs if runtime requires (empirical baseline: 0.8s with these settings on cluster 67).

4. **Build an ADAPTER (NOT a new renderer)** that converts `ReconstructionResult` → the on-disk artifact shape `plot_recons` (axon_recon's existing phase) reads. **First step: audit plot_recons** — locate it under `pipeline/stages/reconstruct/phases/`, read its input contract, identify whether it consumes (a) `gtr.pkl`-like axon_velocity output, (b) a structured numpy/parquet layout, (c) something else. Document the adapter design BEFORE writing code. **If the shape mismatch is fundamental (e.g. plot_recons hardcoded to consume gtr.pkl), STOP AND ASK** — do not invent a new renderer.

5. **Invoke plot_recons twice** — once on axon_velocity_gtrs's existing reference output for the same unit, once on radivojevic_recon's adapted output. Two PNGs, SAME rendering code.

6. **Compose `comparison.png` side-by-side**.

7. **File HARD-gate diagnostic** in `diagnostics_to_review.md` with: radivojevic PNG / axon_velocity PNG / composite + a `run_summary.json` documenting unit choice, knobs, runtimes.

8. **File real-data smoke entry** in `smoke_log.md` (entry #4 — first apples-to-apples comparison).

9. **PAUSE for user review.** That's GATE 1 satisfied + the actual user-review gate.

**Anticipated friction points** (where the loop is most likely to face the kind of friction that previously triggered improvisation):
- A) Unit-ID mapping (step 2): kssynth unit IDs may not align with the existing `unit_0598` reference numbering. The temptation to "just pick any unit" or "use the first high-channel-count unit" is exactly what step B1 forbids.
- B) plot_recons input shape (step 4): plot_recons may expect a specific on-disk structure that radivojevic doesn't natively produce. The temptation to "write a quick renderer in the sibling repo" is exactly what B1 forbids.
- C) Empty or unexpected kssynth output (step 1): if kssynth produces fewer units than expected, or the per_unit/ dir is empty, the temptation to "fall back to kilosort templates" is exactly what B1 forbids.

For each, the correct response is: write a focused question to this file under "PRE-DIAGNOSTIC GATE 1 — friction encountered" and PAUSE.

**Resolution criterion**: user reads this plan and either (a) greenlights as-is, (b) tweaks specific steps, or (c) redirects entirely. Loop does not execute step 1 until this gate has a `✅ USER APPROVED <date>` mark above the gate's title.

## 🔴 IMMEDIATE — Radivojevic diagnostic MUST include PNG renderings (USER FEEDBACK 2026-05-21)

User feedback on the first SOFT-gate filing: "I see the recon output but its npy and tsv files." The diagnostic landed with npy + tsv only — that's not a visual diagnostic, that's data dumps. The strict diagnostic rule (USER INJECTION 2026-05-21) requires user-visible rendering — and rendering means PNG, not arrays.

**MANDATORY for the loop's next iteration on radivojevic**:
1. Add a `render_reconstruction_png(result: ReconstructionResult, channel_positions_um, *, output_path: Path)` function to `radivojevic2023_recon_algo`. Renders:
   - Channel positions as light-gray dots (background)
   - Stage 1 detected peaks as colored markers at (x, y) of their channel, colored by step (step1=red, step2=orange, step3=yellow) — quick visual proxy for "are peaks where the eye expects signal"
   - Stage 2 skeleton union overlaid (binary mask → light blue pixels)
   - Stage 3 links as line segments connecting peak xy positions, colored by method (direct=solid green, skeleton-assisted=dashed green, indirect=dotted green)
   - Title with unit ID + stage counts (n_peaks, n_skeleton_pixels, n_links)
2. Re-file the SOFT-gate diagnostic with the PNG included (NOT replacing the npy/tsv — those stay for downstream consumers; the PNG is the user-visible artifact).
3. **Going-forward rule (amend strict diagnostic injection)**: ANY diagnostic with claimed visual content MUST include a rendered image format (PNG / SVG / PDF). npy / tsv / parquet alone is data, not a diagnostic. The visual file is the audit-trail artifact. Data files can accompany it for re-rendering / downstream consumption.

This unblocks itself in the next loop iteration — no user gate needed. Loop runs it on the kilosort-cluster-67 smoke that already ran; the rendered PNG becomes the SOFT-gate diagnostic content.

## Radivojevic real-data smoke (sub-step 9) — partial findings 2026-05-21

Loop attempted the first real-data smoke per USER GATE 3 spec. Findings:

**Data layout differs from GATE 3 spec:**
- GATE 3 spec said: "run the chosen unit's `merged_template.npy` +
  `merged_channel_locations.npy` through Stage 1 → Stage 2 → Stage 3."
- These files **do NOT exist on disk** for the reference cohort. The
  per-unit STAs live inside `gtr.pkl` (axon_velocity-pickled object)
  which requires `axon_velocity` to unpickle — shifter-only.
- `templates.npy` at `spikesort_outputs/sorter_output_snapshot/`
  (shape `(502, 61, 266)`) DOES exist and provides the raw per-cluster
  kilosort templates. Channel positions also available at
  `channel_positions.npy` (shape `(266, 2)`).

**Unit→cluster mapping is unclear:**
- Reference recon-stage unit dirs are named `0001`..`0626` (sparse,
  176 IDs total).
- Kilosort cluster IDs span `0..501` (max=501 in spike_clusters.npy).
- So `unit_0598` (9-branch unit picked from GATE 3 high-branch-count
  scan) IS NOT kilosort cluster 598. The mapping likely lives in
  `branches.json`'s `unit_id` field which references the POST-MERGE
  axon_velocity_gtrs unit numbering. Without `axon_velocity`, the
  mapping can't be inverted from the loop.

**Loop's pragmatic substitute:**
- Switched to selecting a high-amplitude + high-spike-count
  KILOSORT cluster directly from `templates.npy`. Cluster 67 picked
  (amp=73.3 μV, n_spikes=1328 — well above the `min_n_spikes=50` gate).
- Ran `radivojevic2023_recon_algo.reconstruct(...)` on
  `templates[67].T` (shape `(266, 61)`).
- **Stage 1 timing UNKNOWN**: the 5-minute timeout fired before the
  output landed. Stage 2 is suspected slow with default pixel_um=1 +
  upsample_factor=10 + 600 timeframes — that's ~600 scipy.griddata
  calls on a 266-point sparse input with potentially 10k+ target pixels.
- **Next iteration MUST tune knobs**: reduce upsample_factor to 2-3
  (cuts timeframes to 121-181), or raise pixel_um to 5-10 (cuts target
  pixels by 25-100x), or both. Then re-run + capture timing baseline.

**Open questions — ✅ RESOLVED 2026-05-21**:
1. User CONFIRMS: `merged_template.npy` files are produced by kssynth's per-unit postprocess (slice 4 SHIPPED — `_write_per_unit_templates_from_synth_output`). They don't exist in the reference data because the reference was built with the older pipeline; they'll exist in `dev_outputs/kssynth_slice3b/...` once kssynth slice 3b HEAVY runs.
2. User CHOSE: **APPLES-TO-APPLES via kssynth heavy** — loop runs kssynth slice 3b heavy on M08073/well000 (~5-15 min for analyzers cache build + 30s kssynth), then re-runs radivojevic on the SAME high-branch unit (unit_0598 or whichever post-merge unit the user originally identified as the 9-branch reference). Two PNGs of the same axon, rendered by each algorithm. Direct head-to-head.
3. plot_recons side-by-side comparison gates on the same — kssynth heavy unblocks it.

**Concrete next loop iteration sequence**:
1. Run kssynth slice 3b heavy smoke (PATH 2 — `--input-root` plumbing already shipped):
   ```bash
   axon-recon stages reconstruct.analyzers --config dev/debug_NERSC/debug.runtime.yml \
     --target-dataset 260326 --limit-wells 1 --task-backend local_affinity \
     --input-root /pscratch/sd/a/adammwea/analyzed_data/Media_Density_T5_02182026_AR_axon_analysis_AW/ \
     --output-root /pscratch/sd/a/adammwea/dev_outputs/kssynth_slice3b/
   axon-recon stages reconstruct.kssynth --config dev/debug_NERSC/debug.runtime.yml \
     --target-dataset 260326 --limit-wells 1 --task-backend local_affinity \
     --force-enable kssynth \
     --input-root /pscratch/sd/a/adammwea/analyzed_data/Media_Density_T5_02182026_AR_axon_analysis_AW/ \
     --output-root /pscratch/sd/a/adammwea/dev_outputs/kssynth_slice3b/
   ```
2. Locate the post-merge unit dir for the 9-branch unit (loop previously identified `unit_0598` via branches.json scan — that's the target). Confirm `merged_template.npy` + `merged_channel_locations.npy` exist there.
3. Run radivojevic_recon on that unit's merged_template. Generate PNG.
4. Generate axon_velocity_gtrs PNG via `plot_recons` (existing phase) on the SAME unit's reference recon output. Note: reference data has axon_velocity_gtrs output, so `plot_recons` can read it directly without rerun.
5. Compose side-by-side `comparison.png`.
6. File HARD-gate diagnostic in `diagnostics_to_review.md`.
7. File real-data smoke entry in `smoke_log.md` (entry #4).
8. PAUSE for user review.

**Status**: UNBLOCKED. Loop's next sequence is fully spec'd; estimated ~20-30 min wall-time including kssynth heavy + radivojevic + PNG composition.

## Per-slice empirical findings

- **resources.profiles elimination — per-srun-flag fallback when slot is missing**: when `--cpus-per-task` is passed on the command line, does the resolver use that directly, or compute from `os.sched_getaffinity(0)`? Both are reasonable; pick during the implementation slice. See `trackers/tech_debt.md` §"Minimize / eliminate `resources.profiles`".

- **Auto-restart with chip-well-group phase scope** (`unitmatch` phase): the auto-restart logic walks `phase_sequence` per target. For chip-well groups, the "target" is a group, not a (dataset, well) pair. Verify the logic generalizes when the unitmatch phase lands.

## Radivojevic slice 3 USER GATE 3 — ✅ RESOLVED 2026-05-21 (PROCEED FULLY AUTONOMOUS through Stage 2 AND Stage 3)

**User authorized**: Loop proceeds through Stage 2 (image skeletonization: sub-steps 6a + 6b + 6c) AND Stage 3 (multi-step tracking / peak interlinking) **without intermediate gates**. No pause between stages. First end-to-end real-data run becomes the natural next gate.

**HOWEVER** (per tightened diagnostic rule USER INJECTION 2026-05-21): **each stage transition MUST file its OWN diagnostic entry** during execution, not just the final end-to-end gate. Loop produces and files:
- **Stage 1 diagnostic (SOFT-gate)**: first real-data peak-detection output overlaid on the chosen high-branch-count unit's STA. Visualizes peak distribution sanity — are detected peaks where the eye sees signal? Soft gate: downstream stages can proceed; user reviews when convenient.
- **Stage 2 diagnostic (SOFT-gate)**: first real-data electrical-image rendering + skeleton thinning overlay. Visualizes whether the skeletonized footprint resembles the underlying axon arbor. Soft gate.
- **Stage 3 diagnostic (HARD-gate)**: first real-data interconnect / final axon trajectory + the side-by-side comparison vs axon_velocity_gtrs. THIS is the existing GATE 3 spec below — also the gate the loop pauses on.

Filing each transition keeps the audit trail clean and lets the user spot upstream stage errors before they poison downstream review.

**Concrete spec for the next gate trigger** (the first real-data smoke + side-by-side method comparison):
- **Unit selection**: pick a unit from M08073 80k DMEM well000 (`260326/M08073/000208/well000`, DIV 36, known-good baseline) **WITH PLENTY OF BRANCHES** per its existing `axon_velocity_gtrs` reconstruction. Loop should scan the reference data's existing axon_velocity_gtrs outputs (under `analyzed_data/.../well000/recon_outputs/`), look at per-unit branch counts or visualization complexity, and pick a high-branch-count unit (target: ≥ ~8-12 inter-branch segments, comparable to the paper's example cell with 23 axon terminals if possible). The high-branch-count unit makes algorithmic differences between axon_velocity_gtrs and radivojevic_recon visually obvious; a 2-branch unit would be too easy / hide differences.
- **Reuse the existing `plot_recons` phase** for visualization — DO NOT build new plotting code. plot_recons takes a reconstruction output and renders it; the comparison comes from running plot_recons on BOTH outputs:
  - **A**: existing `axon_velocity_gtrs` output (already exists at the reference path; no rerun needed)
  - **B**: new `radivojevic_recon` output (run the chosen unit's `merged_template.npy` + `merged_channel_locations.npy` through Stage 1 → Stage 2 → Stage 3)
  - plot_recons rendering of BOTH side-by-side isolates the algorithmic difference (not the plotting difference).
- **Output layout**:
  - `/pscratch/sd/a/adammwea/dev_outputs/radivojevic_first_run/<unit_id>/axon_velocity_gtrs/...` — A's plot_recons rendering
  - `/pscratch/sd/a/adammwea/dev_outputs/radivojevic_first_run/<unit_id>/radivojevic_recon/...` — B's plot_recons rendering
  - `/pscratch/sd/a/adammwea/dev_outputs/radivojevic_first_run/<unit_id>/comparison.png` — side-by-side composite for the HARD-gate review
- **MANDATORY**: HARD-gate entry in `dev/notes/memory/diagnostics_to_review.md` pointing at the comparison composite. User compares the two reconstructions: same axon visible? Different number of detected branches? Velocity estimates qualitatively similar? Either method visually wrong on this unit?
- **MANDATORY**: real-data smoke entry in `dev/notes/trackers/smoke_log.md` per the smoke-log discipline rule, capturing unit_id + branch counts from both methods + runtime.
- **Reusing plot_recons**: per loop's audit of the recon stage's existing phase code, `plot_recons` is in `src/.../phases/plot_recons.py` (or equivalent) and reads from `<well>/recon_outputs/<unit>/merged_template.npy` + a reconstruction artifact. For the radivojevic_recon variant, the loop needs to: (1) ship a thin adapter that writes a radivojevic-recon's `Stage3Result` into the same on-disk shape `plot_recons` expects, OR (2) call `plot_recons`'s plotting helpers directly with both methods' outputs. (1) is cleaner; (2) is faster to draft. Loop picks at execution time.
- Then PAUSE for user review — that's the next user gate.

**Original gate body preserved below for archeology:**

## Radivojevic slice 3 USER GATE 3 — original body (2026-05-21)

Per the slice 3 plan ("every 2-3 sub-steps; switch to another plan
until user reviews"), the loop has shipped 3 more sub-steps since
GATE 2 resolved:

- **sub-step 3b** (commit `3e97770`): `core/derivatives.py` — option-B
  thin helper for μV/μs conversion. 12 tests.
- **sub-step 4** (commit `bb3b3d4`): `core/adaptive_thresholding.py` —
  Step 2 confined 2-STD thresholding (50 μm spatial + ±1 temporal).
  Same `find_confined_peaks_step_n` function serves Step 3. 7 new tests.
- **sub-step 5** (commit `8a76a7b`): `core/stage_1.py` — the
  promised option-C orchestrator. `detect_axon_peaks(trace, *,
  channel_positions_um, sampling_rate_hz, ...)` composes the full
  6-stage pipeline (upsample → derivative → noise → step 1 → step 2 →
  step 3). Returns `Stage1Result` with per-step peak lists, sorted
  `all_peaks` union, noise STD, per-step thresholds, upsampled rate +
  dt_us for downstream stages. **STAGE 1 IS COMPLETE end-to-end** —
  callable on real (channel x time) STAs with one function. 9 new tests.

Package now **81 tests total**, all green. Stage 1 paper-faithful
defaults all locked.

**No new design ambiguities surfaced** — the option-C upgrade unified
the per-step thresholding APIs cleanly. The only previously-flagged
empirical-tuning question (temporal_radius_frames default) remains
in the adaptive_thresholding docstring; slice 6 sweep proposal stands.

**Ready to start stage 2 (image skeletonization)** when the user
greenlights. The natural sub-steps for stage 2:
- (6a) `core/electrical_image.py` — build 2D electrical images
  (interpolated voltage map at each timeframe; uses channel positions
  + a chosen grid resolution).
- (6b) `core/skeletonization.py` — apply morphological thinning
  (likely via `skimage.morphology.skeletonize`).
- (6c) Stage-2 orchestrator + tests on synthetic patterns.

If user wants the loop to proceed without explicit gating, this entry
can be marked RESOLVED with a "proceed to stage 2" note; otherwise
mark sub-step 6 as the next-iteration target after review.

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
