# Radivojevic-2023 reconstruction algorithm plan

## Motivation

Radivojevic et al. 2023 describe a reconstruction algorithm for axon
tracking on high-density MEAs. The user wants it implemented as an
**alternative and/or comparison to axon_recon's current `axon_velocity`
GTRS-based approach** in the recon stage. Sibling-package scaffold has
been started by the user at
`/global/homes/a/adammwea/dev/pkgs/radivojevic2023_recon_algo/` with
`literature/` and `notes/` subdirs already populated (see slice 1).

Reference code is reportedly NOT publicly available. Plan starts with
an exhaustive literature read + code-availability search (slice 1)
before committing to a clean-room re-implementation.

User direction (2026-05-19):
- "I want to implement [Radivojevic 2023's algorithm] in the recon stage
  as an alternative and/or comparison to axon_recon's algo."
- "Inputs are ideally comparable to what axon_velocity needs to generate
  gtrs but unsure."
- "I would eventually want to produce all the same plots and metrics that
  axon_velocity gtrs enable, if not more."
- "I'd later want to ship this with full attribution to the authors."
- "If you can find the code online, and avoid reinventing the wheel,
  obviously that'd be ideal."
- "This plan will certainly require some back and forth with me."

## Pre-populated literature (user-provided)

In `/global/homes/a/adammwea/dev/pkgs/radivojevic2023_recon_algo/literature/`:

- **`elife-86512-figures-v1.pdf`** — eLife article 86512 (almost certainly
  the primary "Radivojevic 2023" paper). Read in slice 1; confirm with user
  during slice 1 user-gate.
- **`Radivojevic and Rostedt Punga - Functional imaging of conduction
  dynamics in cortical...`** — possible second 2023 paper. May provide
  complementary algorithmic detail. Read in slice 1.
- `Radivojevic et al. - 2017 - Tracking individual action potentials...` —
  earlier methods paper; useful background for algorithm lineage.
- `Radivojevic et al. - 2016 - Electrical Identification and Selective
  Microstimulation...` — earlier methods; deeper background.
- `Franke et al. - 2015 - Bayes optimal template matching for spike sorting`
  — adjacent / not directly the target algorithm but useful context.

## Out of scope (v1)

- Replacing `axon_velocity` GTRS as the recon-stage default. New algo
  lives alongside, opt-in via YAML. Comparison work decides which (if
  either) becomes default later.
- Improving on or modifying the published method. Stay faithful to the
  paper for v1; deviations recorded in slice 9 as v2 questions.
- Publishing the sibling package to PyPI. Stays local + private per
  existing GH-remotes-hold policy.

## High-uncertainty plan structure

This plan has significantly more open questions than typical
sibling-package plans because:

- Source code may not be public — clean-room reimplementation carries
  inherent risk of misinterpretation.
- Input compatibility with `axon_velocity_gtrs` is assumed but not
  verified.
- "Same plots and metrics" may expand scope as we discover what
  Radivojevic's output supports.
- Implementing the method may imply new phases in OTHER stages
  (preprocess / spikesort / reconstruct) if input formats differ.

**Explicit user check-in gates** are built into the slice ladder. Loop
posts questions to `open_questions.md` at each gate and switches to
another plan until the user answers. Gates at slices 1, 3, 4, 6, 9.

## Slices

### Slice 1 — Literature + code search (RESEARCH ONLY, no code changes) — SHIPPED 2026-05-21

**Deliverables:**
- ✅ `dev/notes/brain/refs/radivojevic2023_paper.md` — citation (DOI 10.7554/eLife.86512, published 2023-08-22) + data availability (Dryad doi:10.5061/dryad.gxd2547r1) + code availability finding (**NONE** — clean-room required) + lineage of related papers (Buccino 2022 → axon_velocity, Bullmann 2019 → hana, Radivojevic 2016/2017 → earlier methods).
- ✅ `dev/notes/brain/refs/radivojevic2023_algorithm_summary.md` — algorithm spec (3 stages: adaptive thresholding + skeletonization + multi-step tracking), input/output spec, hardware assumptions (HD-MEA ~17.5 μm pitch, 20 kHz), input compat table vs `axon_velocity_gtrs`, list of slice-3 hyperparameter unknowns flagged for tuning.
- ✅ USER GATE 1 questions logged to `dev/notes/brain/open_questions.md` under "Radivojevic slice 1 user-gate review" (6 questions; question 3 — averaged-vs-per-spike STA — is the highest-impact one).

**Note on PDF reading**: the loop's environment lacks `pdftoppm` /
`pdftotext`; the pre-populated literature/ PDFs couldn't be read in
this iteration. All paper info gathered via WebFetch on the eLife
article page + WebSearch. The figures-only PDF in literature/ is
useful visual reference but doesn't carry the methods detail
(suffix `-figures-v1` = supplementary figures only). When PDF
extraction is available in a future env, slice 3's algorithm
implementation should re-read the full methods section to verify
hyperparameter defaults + answer question 3 above (raw per-spike STA
vs averaged template).

**Pre-overnight clearance**: per current_state.md "PRE-OVERNIGHT
CLEARANCES" item #2, loop proceeds INTO slice 2 (sibling-package
scaffold) without pausing. Slice 3 (core algorithm impl) DOES gate
on user review of the open_questions.md entries.

#### Original spec (preserved for slice 3 reference)


**Goal**: pin the exact paper; exhaust code-availability search;
produce a written algorithm summary; identify input requirements.

**Actions**:
- **Read the pre-populated PDFs** under
  `~/dev/pkgs/radivojevic2023_recon_algo/literature/`. Primary
  focus on `elife-86512-figures-v1.pdf` (likely the target paper);
  also read the Radivojevic + Rostedt Punga paper if it covers
  algorithm detail.
- **Identify the paper precisely** — record DOI + title + author list
  in this plan + in a new `dev/notes/brain/refs/radivojevic2023_paper.md`.
- **Search for public code** via WebSearch + WebFetch tools:
  - GitHub: search "Radivojevic", "Hierlemann" (likely ETH Zurich lab
    affiliation), "MaxWell Biosystems".
  - Zenodo + FigShare: paper supplementary code archives.
  - The paper's supplementary materials / data availability section.
  - Lab pages: Hierlemann group at ETHZ.
  - Document findings in this plan (positive OR negative).
- **Write `dev/notes/brain/refs/radivojevic2023_algorithm_summary.md`** — a
  structured summary: algorithm name, inputs, processing steps,
  outputs, hyperparameters, evaluation metrics, figure-level plot
  inventory. This is the source-of-truth doc the rest of the plan
  reads against.
- **Map inputs to `axon_velocity_gtrs`**: side-by-side table comparing
  what Radivojevic needs vs. what `axon_velocity_gtrs` currently
  consumes from the recon stage. Identify deltas.

**Output**:
- `dev/notes/brain/refs/radivojevic2023_paper.md` (citation)
- `dev/notes/brain/refs/radivojevic2023_algorithm_summary.md` (algorithm doc)
- Updated paper-id + code-search-result sections in this plan
- List of user check-in questions for slice 2 kickoff

**USER GATE 1**: pause + post questions to `open_questions.md`:
- Confirm paper identity (paper title + DOI).
- Review the algorithm summary doc.
- Confirm clean-room re-implementation path (OR, if code found,
  approve attribution + license strategy for direct use).
- Confirm or revise the input-delta map (which prerequisite data we
  need to compute that we don't already have).

### Slice 2 — Sibling-package scaffold — SHIPPED 2026-05-21

`~/dev/pkgs/radivojevic2023_recon_algo/` is now a proper sibling
package matching the kssynth / unitlink shape. Initial commit on the
local `main` branch (no remote yet — pre-approved via DIRECTIVE D's
sibling-repo authorization, but the loop didn't `gh repo create` since
the algorithm core isn't yet implemented; user should confirm whether
to publish the empty scaffold now or wait until slice 3 has shipped
something runnable).

**Deliverables:**
- ✅ `git init` (initial commit landed on local `main`).
- ✅ `pyproject.toml` — name=`radivojevic2023_recon_algo`, version
  0.1.0, deps (numpy, scipy, scikit-image for morphological
  skeletonization); `[project.optional-dependencies]` for dev +
  spikeinterface. `[project.urls]` link the paper + Dryad data.
- ✅ Package layout: `src/radivojevic2023_recon_algo/{__init__,api,
  core/,io/}.py` matching the kssynth/unitlink convention.
- ✅ `api.reconstruct()` exists as a stub raising NotImplementedError
  (slice 3 wires the real impl).
- ✅ `core/` + `io/` packages are empty stubs with module docstrings
  for the future stage modules.
- ✅ Empty `tests/__init__.py` + `tests/test_scaffold.py` (5 sanity
  tests: import, version, api symbol, NotImplementedError, sub-package
  imports). All 5 pass under `pip install -e .` + pytest.
- ✅ `LICENSE` (MIT, matching kssynth/unitlink).
- ✅ `README.md` with scaffold disclaimer + paper citation +
  attribution boilerplate + package-layout overview.
- ✅ `.gitignore` (Python project boilerplate).

**Hyperparameter values updated in slice 1's algorithm-summary doc**:
discovered pre-extracted PDF text in
`notes/archive/old_ai_notes_for_reference/radivojevic_2023_methods_mining.txt`
(596 lines) which the user had populated previously. This carries
the concrete hyperparameter defaults — 9/2/1 STD noise thresholds,
50/100/200 μm radii, 200 kHz Whittaker-Shannon up-sampling — that
slice 3 will use as starting points. Open-questions Q5 updated to
reflect this.

#### Original spec (preserved for reference)


**Actions**:
- `git init` (skip if user already did)
- `pyproject.toml` with core deps; specific deps TBD from slice 1's
  summary.
- Package layout:
  `radivojevic2023_recon_algo/{__init__,core,io,api}.py` (or whatever
  fits the algorithm's structure).
- Empty test scaffolding under `tests/`.
- README with paper citation + clean-room re-implementation
  disclaimer + attribution paragraph.
- `LICENSE` (decision in slice 9 user-gate).
- 2-3 sanity tests (import, version).

**Compliance**: per `brain/guardrails/package_contracts.md` — SI-compliant
where it touches SpikeInterface conventions; no axon_recon-specific
deps.

**No remote**: hold per existing GH-remotes-hold policy.

### Slice 3 — Core algorithm implementation (most complex slice)

**Status (2026-05-21)**: STAGE 1 COMPLETE end-to-end. 6 sub-steps
shipped in the sibling repo at `~/dev/pkgs/radivojevic2023_recon_algo/`:

- **3.1** (commit `5b06fbf`): `core/upsampling.py` — Whittaker-Shannon
  sinc interpolation; 15 tests.
- **3.2** (commit `3f363d7`): `core/noise_estimation.py` — paper-
  faithful window-based + robust MAD estimators; 16 tests.
- **3.3** (commit `6403403`): `core/adaptive_thresholding.py` —
  Step 1 planar 9-STD threshold + local-max peak detection. Sanity
  test confirms zero false positives on 10k Gaussian samples
  (matches Fig 5B). 17 tests.
- **3.3b** (commit `3e97770`): `core/derivatives.py` — μV/μs time-
  derivative helper, per USER GATE 2 option B resolution. 12 tests.
- **3.4** (commit `bb3b3d4`): `core/adaptive_thresholding.py` —
  Step 2 confined 2-STD thresholding (50 μm spatial + ±1 temporal
  frame). Same `find_confined_peaks_step_n` function serves Step 3.
  7 new tests.
- **3.5** (commit `8a76a7b`): `core/stage_1.py` —
  `detect_axon_peaks` orchestrator composing
  upsample → derivative → noise estimation → step 1 → step 2 →
  step 3. Returns `Stage1Result` dataclass with per-step peak lists,
  sorted union, noise STD, per-step thresholds, upsampled_rate_hz,
  dt_us. STAGE 1 callable end-to-end on real STAs with one function.
  9 tests.

Package now **81 tests total**, all green. USER GATE 3 surfaced
2026-05-21 in `brain/open_questions.md`. Stage 2 (image
skeletonization) is the next milestone — proposed sub-steps:
- (6a) `core/electrical_image.py` — build 2D voltage maps per timeframe
- (6b) `core/skeletonization.py` — morphological thinning of the maps
- (6c) `core/stage_2.py` — stage-2 orchestrator

USER GATES landed so far:
- USER GATE 1: ✅ RESOLVED 2026-05-21 (paper identity, clean-room
  approach, averaged template sufficient, input compat, hyperparams,
  phase name — all 6 answers locked in).
- USER GATE 2: ✅ RESOLVED 2026-05-21 (chose option B for derivative
  helper; promote to option-C orchestrator after Steps 2 + 3 land).
- USER GATE 3: PENDING — surfaced after STAGE 1 COMPLETE.

**Goal**: best-effort implementation of the algorithm's core processing
chain per the slice-1 summary doc.

**Actions**:
- Implement per the summary doc, step by step. One commit per logical
  step where possible.
- Heavy use of `open_questions.md` for ambiguities in the paper —
  every unresolved spec question gets logged with paper section
  reference + the two/three reasonable interpretations.
- Test each step with synthetic input where feasible.
- **USER GATE 2 every 2-3 sub-steps**: post sub-step results +
  ambiguity questions to `open_questions.md`; switch to another plan
  until user reviews.

**Risk note**: this slice may take many iterations and produce a v1
that diverges from the paper in subtle ways. Divergences documented
in slice 9.

### Slice 4 — Input compatibility check vs `axon_velocity_gtrs`

**Goal**: confirm whether the algorithm's inputs match what
`axon_velocity_gtrs` already consumes, or whether new prerequisite
phases are needed.

**Actions**:
- Diff input requirements (from slice 1 summary) against
  `axon_velocity_gtrs`'s actual inputs (templates, channel positions,
  sample rate, waveform extracts, …).
- **Compatible**: no new phases needed.
- **Incompatible**: document required prerequisite data. Could be new
  phases in preprocess / spikesort / reconstruct, OR new computations
  inside the sibling package, OR both. Surface for user decision.

**USER GATE 3**: surface any required new phases for user approval
BEFORE slice 5. New phases imply additional plan injections for those
stages.

### Slice 5 — Recon-stage phase wire-in

**Goal**: new `radivojevic_recon` phase (name TBD; could also be
`radivojevic_2023_recon` for explicit attribution) in
`pipeline/stages/reconstruct/` that calls into the sibling package.

**Actions**:
- Per the existing recon-stage phase template (compare to
  `axon_velocity_gtrs`).
- `enabled: false` default — opt-in per run.
- All scope flags + `--dry-run` + checkpoint markers per guardrails.
- YAML scaffolding in `debug.runtime.yml` + `debug.data.yml` per the
  YAML-hygiene injection.
- Per `env_parity` guardrail: editable install added to (the future)
  `tools/install_dev_siblings.{sh,py}` once env_install_unification
  plan ships slice 4; until then, document the editable install
  command under USER INJECTIONS.

### Slice 6 — First real-data smoke + HARD-gate correctness review

**Goal**: run on **one (dataset, well, unit)** of the 80k DMEM well000
cohort. Compare side-by-side against `axon_velocity_gtrs`.

**Actions**:
- Login-node smoke per standing constraints.
- Save outputs side-by-side under
  `/pscratch/sd/a/adammwea/dev_outputs/radivojevic_recon_smoke/`.
- **HARD-gate visual diagnostic**: produce comparison plots
  (Radivojevic vs `axon_velocity_gtrs`) — branch overlay,
  propagation timing comparison, every figure type slice 1 surfaced.
- Entry in `brain/diagnostics_to_review.md`.

**USER GATE 4**: user reviews + approves the first real-data output
before any downstream slices ship. **Most important user check-in of
the plan.** If outputs look obviously wrong vs. the paper figures,
slice 3 reopens.

### Slice 7 — Comparable plots + metrics

**Goal**: ship every plot + metric that `axon_velocity_gtrs` produces,
but sourced from the Radivojevic algorithm. Plus any additional
outputs the Radivojevic data supports (per slice 1 summary).

**Actions**:
- Inventory `axon_velocity_gtrs`'s plot + metric outputs.
- Implement equivalents using Radivojevic-output data.
- Tests for each output type.
- If the new algo can drive the analysis-stage propagation-video plan
  (separate plan), expose that path here too.

### Slice 8 — Comparison analysis phase

**Goal**: new analysis-stage phase that compares
`axon_velocity_gtrs` vs `radivojevic_recon` outputs for units where
both have run.

**Actions**:
- Per-unit comparison: agreement metrics, divergence metrics,
  computational-cost diff.
- Comparison plot suite (side-by-side figures, scatter, residuals).
- Lives in `pipeline/stages/analysis/comparison/` or similar.

### Slice 9 — Attribution + docs + divergences

**Goal**: ship the package + recon phase with full attribution and a
clear "what we did differently from the paper" doc.

**Actions**:
- Sibling package README: citation, author list, attribution
  paragraph, clean-room re-implementation disclaimer.
- `DIVERGENCES.md`: every place the implementation deviates from the
  paper (or where the paper was ambiguous and a judgment call was
  made).
- axon_recon main README: cite Radivojevic 2023 alongside
  axon_velocity.
- License compatibility check.

**USER GATE 5**: review attribution wording before declaring done.

## Smoke tests

- After slice 3: each algorithm step passes its synthetic unit test.
- After slice 5: dry-run on a small target list reports correct paths.
- After slice 6: real login-node smoke on one unit produces output;
  HARD-gate visual diagnostic flagged.
- After slice 7: every `axon_velocity_gtrs` plot/metric has a
  Radivojevic equivalent.

## Done criteria

- `axon-recon recon radivojevic_recon --targets <ds>:<well>` produces
  recon outputs.
- Phase wired into recon stage as `enabled: false`.
- Side-by-side comparison plots vs `axon_velocity_gtrs` available
  (analysis-stage comparison phase).
- Full attribution + DIVERGENCES doc shipped.
- All 5 USER GATEs approved.

## Tier + sequencing

**Tier 4 (gated)** — substantive work behind in-flight integration.
Per user direction (2026-05-19): "priority-wise, this can come after
implementing unitlink and kssynth."

Kick-off trigger: **`kssynth_recon_integration_plan.md` slice 5** (the
"enable + retire predecessors" gate that completes the kssynth integration
into the recon stage; this is the slice formerly tracked as "kssynth
slice 9" before the work migrated to its own dedicated plan in mid-2026-05)
AND **`unitmatch_phase_plan.md` slice 5** (enable + login-node
smoke) BOTH shipped. Those are the natural "unitlink + kssynth
integration" markers — first real end-to-end smoke through axon_recon
using the new sibling packages. Until both land, this plan sits idle.
(Cross-plan slice-numbering corrected 2026-05-21 via plan-audit Finding #4.)

Once unblocked, **slice 1 (research-only)** is a good first pick — it's
bounded research that produces written docs + user-gate questions
without touching code. Slices 2-9 sequence behind slice 1 + user gates.

## Open questions (initial — will grow during execution)

- **Paper identity confirmation** (slice 1 user-gate): the literature
  folder strongly implies eLife 86512 is the target, but user should
  confirm.
- **Code availability** (slice 1): is the algorithm actually
  unavailable, or buried in supplementary materials / a GitHub repo we
  haven't found?
- **Input compatibility** (slice 4 user-gate): does the algorithm
  consume the same templates/footprints `axon_velocity` does? Or new
  preprocessing required?
- **New prerequisite phases**: if input compatibility fails, which
  stages need new phases? Implications for tier ordering of those
  ancillary plans.
- **License** (slice 9 user-gate): which license for the sibling
  package? Compatible with axon_recon's?
- **Author contact**: should we reach out to Radivojevic et al. for
  algorithm-detail questions, or attempt clean-room only? (User
  decision; loop doesn't email people.)
- **Plotting parity scope** (slice 7): does "all the same plots and
  metrics that axon_velocity gtrs enable" include the
  propagation-video work, or is that strictly future?

---

## Algorithm-correctness fix slices (added 2026-05-21 after first real-data smoke audit)

**Context**: Slice 6 first real-data smoke (commit `081ea98` HARD-gate diagnostic on unit_0598) revealed algorithm mismatches with the paper. Audit done in `brain/refs/old_pipeline_analyzer_window.md` + detailed in-chat read of the 2023 paper (eLife-86512) Methods + Figures 3, 4. Filing slices 10-15 to bring the radivojevic sibling package into paper-spec compliance.

### Paper two-stage structure (the high-level architecture our code MUST match)

The paper has two CLEANLY SEPARATED stages with DIFFERENT euclidean radii:

**Paper Stage 1 — Adaptive thresholding peak detection (Figure 3)** — radii 50/100μm, NO rider/chain logic. Just finds peaks via thresholding.
- Step 1A: 9 STD planar (global, per-frame, all electrodes)
- Step 1B: 2 STD within **50μm** + ±1 frame of step1A peaks
- Step 1C: 1 STD within **100μm** + ±1 frame of step1A+1B peaks

**Paper Stage 2 — Trajectory reconstruction (Figure 4)** — radii 100/200/400μm, RIDER logic across consecutive frames.
- Direct (Fig 4A): peaks in two consecutive frames within **100μm** → direct line link
- Skeleton-assisted (Fig 4B): peaks in two consecutive frames within **200μm** → link via skeletonization of the **(frame_t + frame_t+1)/2 AVERAGED** electrical image (Δt=100μs at paper's 20kHz raw)
- Indirect (Fig 4C): peaks in EVERY-OTHER frame within **400μm** → link via skeletonization of (frame_t-1 + frame_t + frame_t+1)/3 AVERAGED image (Δt=150μs)

Our package's current implementation:
- `core/stage_1.py` = paper Stage 1 (peak detection) — ✓ structurally correct, radii 50/100μm
- `core/stage_2.py` (skeletonization) — ✗ BUG: skeletonizes EVERY frame independently, not pair/triple-averaged per-link on-demand
- `core/stage_3.py` = paper Stage 2 (trajectory reconstruction) — ✗ BUG: `max_distance_um_indirect=200.0` default (paper says 400μm)

### Slice 10 — Fix `max_distance_um_indirect` to 400μm + add per-paper validation tests

**Scope**: 1-line default change in `core/stage_3.py:70` from `200.0` to `400.0`. Add a regression test that asserts paper-default values match Fig 4 spec (100/200/400μm).

**Tests added**:
- `test_stage_3_defaults_match_paper`: assert `max_distance_um_direct == 100.0`, `max_distance_um_skeleton == 200.0`, `max_distance_um_indirect == 400.0`.

**Touch**: S (1-line + test). Independent of all other slices.

### Slice 11 — Raw-recording-inactive-period noise estimator (paper-spec)

**Why**: noise from template (current MAD or window estimator) is **√n_spikes smaller** than raw recording noise. Paper estimates noise from "background noise was sampled across all electrodes during periods when the observed neuron was inactive, and the noise was estimated for each neuron separately" (2023 page 6).

**Per-channel scale**: paper estimates per-electrode noise (not a single global value). Our current estimators return scalar `noise_std`. Change to `noise_std_per_channel: np.ndarray(n_channels,)`.

**Implementation plan**:
1. Add `noise_estimator='from_inactive_periods'` to radivojevic API. Accept new params:
   - `recording_inactive_samples: np.ndarray(n_channels, n_inactive_samples)` OR
   - `recording_extension`: SpikeInterface recording + spike_times for THIS unit; loader extracts inactive-period samples.
2. Per-channel `noise_std[c] = STD(recording_inactive_samples[c, :])`.
3. Update `adaptive_thresholding.py` to accept per-channel noise (broadcast over time when computing threshold).
4. Update Stage 2's binarization to use per-channel noise.

**Tests added**:
- `test_noise_estimator_from_inactive_periods_scalar_baseline`: synthetic recording with known noise → estimator recovers ground truth.
- `test_noise_per_channel_shape`: per-channel noise array has correct shape.
- `test_adaptive_thresholding_with_per_channel_noise`: detection respects per-channel thresholds (channel with low noise → easier detection).

**Integration**: axon_recon's `reconstruct.radivojevic_recon` phase (when shipped per slice 5) wires up the noise estimator from SpikeInterface analyzer's `compute_noise_levels` extension.

**Touch**: M (sibling pkg API change + axon_recon wire-up).

### Slice 12 — Separate analysis frame interval from upsample factor

**Why**: paper upsamples to 200 kHz for waveform smoothness (10× from 20kHz raw) but analyzes at **50μs frame intervals** (= 20 kHz effective for peak detection / thresholding). Our code's frames are AT the upsampled rate, so analysis frames are 10× more frequent than paper.

**Implementation plan**:
1. Add `analysis_frame_us: float = 50.0` to radivojevic API (default per paper).
2. After upsampling for waveform smoothness, decimate (or average groups of upsampled samples) to the analysis_frame_us interval for the threshold + skeleton steps.
3. Frame counts in Stage 1 + Stage 2 + Stage 3 then refer to ANALYSIS frames, not raw upsampled frames.

**Tests added**:
- `test_analysis_frame_interval_decimation`: input 10× upsampled trace + analysis_frame_us=50 → analyzed at every-5th-frame.
- `test_paper_frame_count_for_20ms_template`: paper "400 frames at 50μs" matches our derived count for a 20ms-wide input.
- `test_analysis_frame_independent_of_upsample`: upsample_factor=20 and =10 produce same step1 peak count given the same analysis_frame_us.

**Touch**: M (sibling pkg algorithm change). Cleanest if done after slice 11 (noise) — frame interval affects thresholding semantics.

### Slice 13 — Stage 2 architecture fix: on-demand pair-averaged skeletonization

**Why**: paper Stage 2 (Fig 4B-C) skeletonizes the AVERAGED-PAIR (or AVERAGED-TRIPLE) electrical image PER LINK CANDIDATE, not every frame independently. Our `core/skeletonization.py:122` does `skeletons[t] = skeletonize(frame, method=...)` — per-frame, no averaging → dense per-frame skeletons (12% of every grid pixel) regardless of where peaks actually exist.

**Implementation plan**:
1. Move skeletonization OUT of pre-computed Stage 2 stack.
2. New helper `skeletonize_for_link(electrical_image_t, electrical_image_t1, ...)` that averages then skeletonizes the pair.
3. Stage 3's `link_peaks_skeleton_assisted` calls this helper for EACH candidate pair (peak_a at frame t, peak_b at frame t+1 within 200μm).
4. Stage 3's `link_peaks_indirect` calls a triple-averaging skeletonize helper.
5. Stage 2's role becomes optional — `electrical_image_t` pre-computation only (just the interpolated 2D voltage maps), no binarization/skeletonization globally.

**Tests added**:
- `test_pair_averaged_skeleton_matches_per_link_call`: avg(frame_t, frame_t+1) skeleton = on-demand result.
- `test_indirect_triple_averaged_skeleton`: avg(frame_t-1, frame_t, frame_t+1) skeleton = on-demand result for indirect linking.
- `test_skeleton_density_per_link_is_local`: per-link skeleton has ~tens of pixels (one wavefront width), NOT thousands.
- `test_stage_3_with_pair_skeleton_match_paper_fig4`: regression test on a hand-built synthetic 3-channel propagating signal.

**Touch**: M-L (architectural refactor of stage_2.py + stage_3.py + tests).

### Slice 14 — Recursive step 2/3 expansion (rider chain logic)

**Why**: paper's "moving object tracking" / user's "rider" model suggests chains extend frame-by-frame as long as continuation peaks exist. Our current code applies steps 1B + 1C once each — temporal extent of rider is bounded by step1's initial frame range + ±1 frame (per `temporal_radius_frames=1`).

**Implementation plan**:
1. Wrap step 1B + 1C in a convergence loop:
   ```
   peaks_step_1B = []
   seeds = peaks_step_1A
   while True:
       new = find_peaks_in_neighborhood(seeds, 50μm, 2 STD, ±1 frame)
       if not new: break
       peaks_step_1B.extend(new)
       seeds = new  # next iteration's seeds = THIS iteration's new peaks (chain logic)
   ```
2. Same for step 1C (with 100μm + 1 STD).
3. Add `max_iterations` safety cap (e.g. 50) to prevent runaway in pathological inputs.
4. Track per-iteration peak counts in a diagnostic field for debugging.

**Tests added**:
- `test_step_1B_iterates_until_no_new_peaks`: synthetic propagation across 10 frames → step 1B chains through all 10 (without iteration, bounded to ±1 frame).
- `test_recursion_terminates_no_new_peaks`: chain stops when noise floor reached.
- `test_max_iterations_safety_cap`: pathological input doesn't infinite-loop.

**Touch**: M (algorithm change to stage_1.py + tests). Independent of slice 11/12/13 but BENEFITS from them (proper noise + paper frame interval → cleaner iteration).

### Slice 15 — Integration smoke + diagnostic regression on unit_0598

**Why**: validate slices 10-14 together produce paper-spec output on real data.

**Smoke**:
1. Re-run radivojevic on kssynth's unit_598 merged_template with paper defaults (n_std 9/2/1, k_stage2=1, no scaling hacks).
2. Generate per-stage diagnostics (channels-only, skeleton-union, comparison vs axon_velocity).
3. File HARD-gate diagnostic update.

**Acceptance criteria**:
- Selected channels (step1+1B+1C) trace the axon arbor shape (NOT central blob).
- Per-link pair-averaged skeleton is local (~tens of pixels per link), NOT global (thousands per frame).
- Stage 2 trajectory (Direct + Skel-assisted + Indirect links) visually matches axon_velocity's branch structure on the same unit.

**Tests added**:
- Regression on synthetic 3-branch propagating signal: known channels along each branch → algorithm recovers each branch.

**Touch**: S (smoke + diagnostic file; tests added per slice 10-14).

### Sequencing for slices 10-15

| Order | Slice | Why |
|---|---|---|
| 1 | Slice 10 | Quick fix; independent. |
| 2 | Slice 11 | Noise estimator is foundational — affects EVERY threshold. |
| 3 | Slice 12 | Frame interval — affects step counts + skeleton timing. |
| 4 | Slice 13 | Stage 2 architecture — depends on slices 11+12 for correct per-link skeletons. |
| 5 | Slice 14 | Recursive rider — benefits from proper noise + frames. |
| 6 | Slice 15 | Integration smoke — validates slices 10-14 together. |

Per-slice tests run in CI (radivojevic2023_recon_algo's `tests/`). Real-data smoke (slice 15) goes to `dev/notes/trackers/smoke_log.md`.

---

## Paper-fidelity Stage-2 refinement-criteria slices (added 2026-05-22 from re-read of Methods + Discussion)

**Context**: After the 2-stage paper-alignment refactor (sibling `1c22138`) shipped, the user pointed at the persistent central-blob symptom and said "the issue is clearly skeletonization." Re-reading the paper Methods section on tracking confirmed three Stage-2-side gaps in our impl that govern skeletonization quality:

> "Trajectories whose velocities deviated from previously estimated values by more than 50% were discarded." (Step 2 — skel-assisted)
>
> "Conduction velocities estimated in the previous steps were used as criteria for selecting optimal propagation trajectories and predicting spatial coordinates of data for the second timeframe." (Step 3 — indirect, both filter + predict)

The paper's mechanism: Step 1 (direct) anchors the velocity prior; Step 2 + Step 3 use that prior as a strict ±50% gate; Step 3 additionally reconstructs the missing intermediate-frame peak position. Our impl had ±100% on Step 3 and no velocity gate at all on Step 2 → skeleton hits between distant peaks pass freely, contributing to over-linking.

### Slice 16 — Step 2 (skel-assisted) velocity filter (±50% from Step-1 median)

**Scope**: Add `velocity_tolerance` parameter (default 1.5 = ±50%) to `link_peaks_skeleton_assisted`. Compute reference velocity from `already_linked` (the direct links passed in); discard skel-assisted candidates whose implied velocity (`distance_um / dt_us`) deviates by >50% of the median.

**Tests added**:
- `test_skel_assisted_velocity_filter_discards_outliers`: synthetic 3-pair setup where one candidate is ±60% of median velocity → discarded; another at ±30% → kept.
- `test_skel_assisted_no_filter_when_no_direct_links`: when `already_linked=[]` (no velocity prior available), accept all candidates that pass the geometry + skeleton check.

**Touch**: S-M (parameter addition + filter logic + 2 tests). Independent of other slices.

### Slice 17 — Step 3 (indirect) tighten velocity tolerance from 2.0 → 1.5

**Scope**: Change `velocity_tolerance` default in `link_peaks_indirect` (and the corresponding pass-through in `link_peaks_all_strategies` + `api.reconstruct`) from 2.0 (= ±100%) to 1.5 (= ±50%) per paper. Update `test_stage_2_defaults_match_paper` to lock the value.

**Tests added**:
- Update existing `test_stage_2_defaults_match_paper` to also assert `velocity_tolerance == 1.5`.

**Touch**: S (default change + test update). Independent.

### Slice 18 — Step 3 (indirect) predict intermediate-frame peak position

**Scope**: Per paper: *"predicting spatial coordinates of data for the second timeframe."* When an indirect candidate (peak_a at frame t, peak_b at frame t+2) clears the velocity + skeleton checks, predict the intermediate peak's xy position at frame t+1 using the velocity. Simplest approach: midpoint along the Bresenham line between peak_a and peak_b (since velocity is constant under the model). Store on `PeakLink` as a new optional field `predicted_intermediate_xy_um: tuple[float, float] | None`.

**Tests added**:
- `test_indirect_predicts_intermediate_xy_midpoint`: synthetic indirect pair → predicted intermediate is the midpoint.
- `test_indirect_no_prediction_for_other_methods`: direct + skel-assisted links have `predicted_intermediate_xy_um = None`.

**Touch**: M (new field on PeakLink + prediction logic + 2 tests).

### Slice 19 — Smoke + per-iteration Stage-2-Step-1 diagnostic on unit_598 (paper defaults)

**Scope**: re-run radivojevic on unit_598 with paper defaults (`n_std_step1=9, n_std_step2=2, n_std_step3=1` — no scaling), slice 16/17/18 changes in place. Produce:
- (a) Updated 2-stage 4×2 grid (refresh of the smoke-#7 plot) showing new link counts under the tightened Stage-2 criteria.
- (b) Per-iteration panels for **Stage 2 Step 1 (direct)**: one panel per frame-pair (t, t+1) with link activity. Each panel shows electrode positions + frame-t peaks + frame-{t+1} peaks (different colors) + accepted direct links + the pair-averaged skeleton overlay (so the user can see what skeletonization would consider for Step 2 if direct misses).

**Acceptance criteria**:
- Stage 2 link counts under tighter criteria are LOWER than the pre-slice-16 baseline (over-linking is reduced).
- Per-iteration panels show clear frame-by-frame progression of the trajectory build.

**Touch**: S (smoke + diagnostic file).

### Sequencing for slices 16-19

| Order | Slice | Why |
|---|---|---|
| 1 | Slice 17 | Trivial default tweak; ships first as a no-risk warm-up. |
| 2 | Slice 16 | Step 2 velocity filter — the biggest expected behavior change. |
| 3 | Slice 18 | Step 3 intermediate-peak prediction — additive (new field). |
| 4 | Slice 19 | Integration smoke + diagnostic — validates 16+17+18 together. |
