# Diagnostics pending user review

Visual / tabular diagnostics that Claude produced during slices and that the user should look at before the slice is fully validated. Some things can only be confirmed visually (waveform shapes, template overlays, sort-quality plots, before/after comparison figures); a passing test suite is necessary but not sufficient.

## How this works

- When a slice's verification benefits from a visual or tabular artifact, Claude generates it during the smoke run, saves it under `/pscratch/sd/a/adammwea/dev_outputs/<plan_slice>/diagnostics/`, and adds an entry below before committing.
- The commit message references this file ("see `brain/diagnostics_to_review.md` entry <N>") so the user knows to check it.
- The user reviews when they can — no synchronous blocking unless Claude flagged the entry as a hard gate.
- Once reviewed, the user either marks the entry `✅ approved <date>` or replies with corrective feedback; Claude then updates the entry to `resolved` or addresses the feedback in a follow-up slice.

## Entry format

```
### <N>. <one-line title>
- **Date / commit**: 2026-05-NN / <commit-hash>
- **Plan / slice**: <plan>.<slice-N>
- **Artifact**: `<path under dev_outputs>`
- **What it shows**: <one sentence>
- **What to check**: <one sentence — what makes this PASS vs FAIL>
- **Gate level**: soft | hard
  - soft = nice to confirm, but downstream slices can proceed
  - hard = must be approved before downstream slices touch this surface
- **Status**: pending | ✅ approved <date> | ❌ rejected <date> (followup: <slice>)
```

## When to add an entry vs not

**Add an entry when:**
- The slice's claim depends on visual inspection (e.g. "templates look right", "raster looks clean", "merge plot shows correct candidate pairs").
- A regression check needs a before/after comparison figure.
- A new phase's first real-data run produces a plot that establishes the baseline.
- A bug fix produces visually different output than before — even when tests pass, the user should sanity-check the picture.

**Do NOT add an entry for:**
- Every routine plot the pipeline emits during a normal run. Those are part of the pipeline's regular outputs; the user can browse them anytime via the dashboard or `find`.
- Plots that have automated assertions (numerical counts, hash comparisons) that fully cover the validation. Tests are the validation; the plot is a bonus.
- Logs / text summaries — those go in commit messages, not here.

## Pruning

Keep the list short. When an entry is `approved`, leave it for ~one week so the audit trail survives, then prune. `rejected` entries stay until the followup slice resolves them, then prune.

---

## Entries

(initially empty — Claude appends entries as slices generate diagnostics)

### 2026-05-22 — Radivojevic slices 16/17/18 + per-iteration Stage 2 Step 1 plot (unit_598)

- **Date / commit**: 2026-05-22 / sibling-repo `582af64`
- **Plan / slice**: `radivojevic_recon_algo_plan` — slices 16 (Step 2 ±50% velocity filter) + 17 (Step 3 tighten 2.0→1.5) + 18 (Step 3 intermediate-peak prediction) + 19 (smoke + per-iteration plot).
- **Artifacts** (`/pscratch/sd/a/adammwea/dev_outputs/radivojevic_paper_2stage/diagnostics/`):
  - `unit_598_two_stage_diagnostic.png` — refreshed 4×2 grid showing Stage 1 (top) + Stage 2 (bottom) with the new tighter Stage 2 criteria.
  - **`unit_598_stage2_step1_per_iteration.png` ← the main artifact for this review.** 15 panels, one per active frame-pair iteration. Each panel shows electrode positions (gray dots) + pair-averaged skeleton (light blue, what Step 2 would consider) + frame-t Stage 1 peaks (red squares) + frame-{t+1} Stage 1 peaks (orange triangles) + direct links accepted (green lines). Iteration title shows frame indices + time stamps (Δt = 50 μs/iteration). Useful for inspecting where the trajectory build connects vs misses.
  - `unit_598_summary.json` — quantitative summary.
- **Quantitative**:
  - Stage 1 peaks (unchanged from before, since slices 16/17/18 only touch Stage 2): 173 / 424 / 881.
  - **Stage 2 link counts before vs after slices 16/17/18**:
    - direct: 991 → 991 (unchanged; direct doesn't use velocity)
    - skeleton-assisted: 12 → **0** (the ±50% velocity filter killed every candidate — Step 2 candidates had implied velocities outside ±50% of direct's median, so the over-reaching skeleton-routed links are correctly suppressed)
    - indirect: 31 → **33** (slight shift due to tighter ±50% gate + intermediate-peak prediction now populating the new `predicted_intermediate_xy_um` field).
  - Wall: 171s (similar to pre-slice runs; the velocity filter is fast).
- **What to check**:
  - **(per-iteration plot)** Does each iteration's frame-t / frame-{t+1} peak distribution look plausible for an axonal propagation moment? The user asked for "every iteration of Step 1" — 15 panels = the 15 frame-pairs where direct linking found candidates. Δt=50 μs between iterations (paper-faithful).
  - **(slice 16 effect)** Step 2 (skel-assisted) link count went 12 → 0. Reasoning: the central blob's Stage-1 peaks have small direct-link distances (low implied velocity), so the median velocity is small. Any candidate that needs the skeleton (longer distance) has implied velocity well above median, and the ±50% filter discards it. This is paper-correct behavior — Step 2 is designed to fill SHORT gaps near the direct-link trajectory, not long-range over-skeleton links.
  - **(slice 18 effect)** Indirect links now carry a `predicted_intermediate_xy_um` field (not visualized in current plot; could be added next iteration if useful).
- **Known caveat**: thresholds remain at v4 scale (n_std 90/20/10 instead of paper's 9/2/1) for tractability — paper defaults yielded 45k+ peaks → reconstruct >5min, infeasible for an interactive smoke. The slice 16/17/18 effects shown here are independent of the threshold magnitude; the raw-recording-noise wiring (axon_recon slice 11) is the principled fix and stays the next session's task.
- **Gate level**: soft — review when convenient.
- **Status**: pending

### 2026-05-22 — Radivojevic 2-stage paper-alignment refactor (unit_598)

- **Date / commit**: 2026-05-22 / sibling-repo `1c22138`
- **Plan / slice**: `radivojevic_recon_algo_plan` — paper-alignment refactor (collapses misnamed stage_2 + stage_3 → single paper-Stage-2 module; adds per-step / per-frame trace fields; flips skeleton default to paper-faithful on-demand pair/triple-averaged)
- **Artifact**: `/pscratch/sd/a/adammwea/dev_outputs/radivojevic_paper_2stage/diagnostics/unit_598_two_stage_diagnostic.png`
- **What it shows**: 4×2 grid where the TOP row is Paper Stage 1 (channel selection) with each of the 3 thresholding steps as its own panel + a union panel; the BOTTOM row is Paper Stage 2 (trajectory reconstruction) with each of the 3 link steps (direct / skeleton-assisted / indirect) as its own panel + a union panel. Background gray envelope is the max-abs of the electrical-image grid. Channel positions are light gray dots.
- **Quantitative**:
  - Stage 1 cumulative peaks (9 / 2 / 1 STD steps): 173 / 424 / 881
  - Stage 2 links (direct / skel-assisted / indirect): 991 / 12 / 31
  - Wall: 180.12s (single-pass; iterative_expansion=False for tractability)
  - Stage 1 step-1 hit 11 frames (out of ~140 analysis frames)
  - **Threshold note**: stopgap n_std 90 / 20 / 10 (= paper 9/2/1 × 10). The 10× scale is the v4 baseline the user already accepted. Once raw-recording noise-per-channel is wired in (axon_recon slice 11 still pending), paper 9/2/1 should yield comparable peak counts without the scale.
- **What to check**:
  - Does each top-row panel look like a sensible "channel-selection at this threshold"? Step 1 should mark a small cluster of high-amplitude channels; step 2 should expand modestly; step 3 should expand a bit more (in a relaxed-confinement way), NOT fill the whole array.
  - Does each bottom-row panel show a different kind of link (direct = many short links between adjacent frames; skel-assisted = sparse longer links; indirect = even sparser, every-other-frame).
  - **KNOWN ISSUE the user has called out**: selected channels still cluster centrally (around the soma), not tracing the axon arbor. This is the noise-underestimate symptom — template-derived MAD noise is √n_spikes smaller than the paper-spec raw-recording noise, so the relaxed-threshold step 3 still excludes peripheral arbor channels. The fix is axon_recon slice 11 wiring (next session). The plot's purpose this turn is to confirm the algorithm STRUCTURE matches the paper, not that the output is biologically correct.
- **Gate level**: soft — review when convenient; downstream slice work (axon_recon-side noise wiring) can proceed once the structure is confirmed.
- **Status**: pending

### 2026-05-21 — Radivojevic first real-data smoke (cluster 67, M08073/well000/DIV 36) — RE-FILED with PNG

- **Gate**: SOFT (Stage 1 + Stage 2 + Stage 3 outputs; loop proceeded autonomously)
- **What's here**: First real-data outputs from `radivojevic2023_recon_algo.reconstruct(...)` on a kilosort cluster from M08073 well000. **Re-filed 2026-05-21 with PNG** after USER FEEDBACK that npy/tsv alone aren't a diagnostic.
- **Path**: `/pscratch/sd/a/adammwea/dev_outputs/radivojevic_first_run/cluster_67/radivojevic_recon/`
- **🖼 USER-VISIBLE ARTIFACT (open this first)**: **`recon.png`** (71KB)
  - Renders: channel positions (light-gray dots) + Stage 2 skeleton union overlay (light-blue) + Stage 1 detected peaks colored by step (crimson=9-STD, orange=2-STD, gold=1-STD) + Stage 3 inter-frame links as line segments (solid=direct, dashed=skel-assisted, dotted=indirect).
  - Title shows per-step counts + skeleton pixel total + per-method link counts.
  - Generated by `radivojevic2023_recon_algo.io.render_reconstruction_png(...)` (commit `9ff6eff`).
- **Data files** (downstream consumers + re-rendering):
  - `run_summary.json` — knob values + per-stage numerical results.
  - `peaks_all.tsv` — Stage 1 output (172 detected peaks; channel_idx + time_idx + amplitude per step).
  - `links.tsv` — Stage 3 output (72 inter-frame edges; method=direct/skel/indirect, distance + dt).
  - `skeleton_union_xy.npy` — Stage 2 binary skeleton union across all 120 frames (183 x 202 bool).
  - `skeleton_pixels_per_frame.npy` — per-frame skeleton pixel counts.
  - `per_channel_peak_counts.npy` — per-channel peak histogram.
- **What to verify**:
  - **Stage 1**: 12 of 266 channels had peaks (kilosort templates ARE sparse — only ~6 channels around the spike's source typically active; 12 is plausible). 9-STD step caught 97 peaks; 2-STD step caught 66 more; 1-STD step caught 9 more.
  - **Stage 2**: 933 unique skeleton xy pixels. If you plot `skeleton_union_xy.npy` as a 2D image, expect a small cluster of pixels around the channels with peaks (NOT a sprawling structure — this is a single-recording kilosort template, not a multi-segment merged template).
  - **Stage 3**: 72 edges total (52 direct + 1 skel-assisted + 19 indirect). The direct-link count being the majority is paper-consistent (paper said direct catches ~70%; we got 72%).
- **Critical follow-ups** (logged in `brain/open_questions.md`):
  - MAD-noise-estimator-on-sparse-template bug (noise_std=0 → thresholds=0 → flood of false positives). Workaround: use `noise_estimator='window'`. Real fix candidates documented in smoke_log.md.
  - GATE 3 spec's `merged_template.npy` data-layout question (3 questions for user).
- **Status**: SOFT-gate filed; **HARD-gate (Stage 3 + plot_recons side-by-side) still pending** the user's resolution of the data-layout question OR the kssynth slice 3b heavy smoke producing the merged-template artifacts the comparison needs.

### 2026-05-21 — Radivojevic apples-to-apples on unit_0598 (HARD-gate) — kssynth-merged template

- **Gate**: HARD — please review before downstream radivojevic_recon_algo_plan slices advance.
- **What's here**: First side-by-side comparison of axon_velocity_gtrs (reference, OLD pipeline) vs radivojevic2023_recon_algo (new clean-room sibling) on the SAME unit (M08073/000208/well000 DIV 36, unit_0598 — 9-branch reference unit per branches.json).
- **🖼 USER-VISIBLE ARTIFACT (open this first)**: **`/pscratch/sd/a/adammwea/dev_outputs/radivojevic_apples_to_apples/unit_598_radivojevic/comparison_unit_598.png`** (4194×1800 px)
  - LEFT panel: axon_velocity_gtrs's existing `circle_recon.png` from reference (unit_0598 — built by the OLD pipeline; gtr.template shape (13439, 300) — see brain/refs/old_pipeline_analyzer_window.md for why 300).
  - RIGHT panel: radivojevic_recon's own rendering (peaks + skeleton + links) on the SAME unit's kssynth-produced merged_template (shape (13439, 70), upsampled internally 10x by radivojevic).
- **Sub-artifacts** (right panel only — for re-rendering / downstream):
  - `radivojevic_unit_598.png` — standalone radivojevic render.
  - `result.pkl` — pickled ReconstructionResult (Stage 1 + Stage 2 + Stage 3 dataclass tree).
- **Quantitative summary**:
  - Stage 1 peaks: 5443 (step1, 9 STD) + 8142 (step2, 2 STD) + 31624 (step3, 1 STD) = **45209 total** (compared to cluster 67's 172 — this unit is genuinely high-branch + the peaks are dense).
  - Stage 2 skeleton + Stage 3 links: see `result.pkl` for full counts.
  - Wall: 162.9s (radivojevic on the conda env; no GPU; n_jobs=1 default).
- **Methodological caveats (USER reviewed + accepted per "1-to-1 in the end")**:
  - Inputs differ slightly: axon_velocity ran on a 300-sample upsampled-then-trimmed template; radivojevic ran on the 70-sample raw kssynth template + did its own internal 10x sinc upsample.
  - The "trim" (700→300) per user happens INSIDE axon_velocity (vendor code), not in axon_recon. See brain/refs/old_pipeline_analyzer_window.md for full audit.
  - Both algorithms processed essentially the same axonal signal — comparison is "good enough" per user 2026-05-21.
- **Renderers ALSO differ** (NOT the apples-to-apples ideal of "same plot_recons"):
  - LEFT: axon_recon's `plot_recons` phase (circle_recon style — gtr-pkl-hardcoded; can't render radivojevic without a major plot_recons refactor).
  - RIGHT: radivojevic's own `render_reconstruction_png` (peaks/skel/links style).
  - So differences in visual style are NOT algorithmic differences. Look at: branch coverage (does radivojevic see roughly the same regions axon_velocity does?), peak density, overall axonal extent.
- **What to verify (USER)**:
  - Does radivojevic's reconstruction cover the same spatial region as axon_velocity's? (Same axon → same xy footprint.)
  - Are the radivojevic peaks where you'd expect signal? (Compare peak density per region against the LEFT panel's branch structure.)
  - Does either algorithm look visually wrong on this unit? (E.g. radivojevic finding peaks far from axon_velocity's branches = suspicious; axon_velocity missing a clear branch radivojevic catches = also suspicious.)
- **Status**: HARD-gate filed; downstream radivojevic_recon_algo_plan slices (slice 5 hyperparam sweep, slice 9 second comparison cell) wait for user review.

#### Per-stage debug artifacts (re-filed 2026-05-21 per user feedback)

**User feedback on first composite** (2026-05-21): "1. invert the y axis on the new recon plot. 2. you're still not using the same plot_recon phase machinery — that is clear. 3. your skeletonization looks like way too much. Try plotting each step's channel selections and the skeleton by itself so i can give you better feedback."

New artifacts at `dev_outputs/radivojevic_apples_to_apples/unit_598_radivojevic/per_stage_debug/`:
- `peaks_step1_9STD.png` — 5443 Stage-1 peaks (channel selections at xy positions).
- `peaks_step2_2STD.png` — 8142 Stage-2 peaks.
- `peaks_step3_1STD.png` — 31624 Stage-3 peaks.
- `skeleton_union.png` — 4-panel: channels alone + binary_footprint UNION (across 690 frames) + skeleton UNION + skeleton+channels overlay.
- `skeleton_single_frames.png` — 3 single-frame skeletons (samples at frames T/4, T/2, 3T/4 of 690) for "what does a typical frame look like" feedback.

All debug plots use `ax.invert_yaxis()` (MEA convention: top of chip = low y).

**Stage 2 BUG candidates surfaced by per-frame counts**:
- skeleton_stack.skeletons shape (690, 211, 207), per-frame avg **5142 pixels** out of 43677 total grid pixels (= **11.8% of EVERY frame** is skeletonized).
- skeleton_stack.binary_footprints per-frame avg **8804 pixels** (= 20% of every frame is "above threshold").
- For a real axon at any given frame you'd expect tens to low-hundreds of skeleton pixels max. 5142 = the binarization/skeletonization is firing on noise or interpolation artifacts.
- Likely root causes: (a) binarization_k_stage2 threshold too aggressive (default 1.0 noise_std); (b) noise_std estimate too low (MAD on dense merged template might under-estimate); (c) image_grid interpolation creating dense low-amplitude artifacts that all exceed threshold.

**Outstanding action** for the main composite: rendering via plot_recons phase machinery (per user "use plot_recons machinery — that would handle y-invert and consistent style"). Question surfaced separately.

#### v2 composite via plot_recons machinery (per user "use plot_recons machinery")

**New artifact**: `dev_outputs/radivojevic_apples_to_apples/unit_598_radivojevic/comparison_unit_598_v2_plot_recons_style.png`

- LEFT: axon_velocity_gtrs's existing `circle_recon.png` (unchanged).
- RIGHT: radivojevic rendered via `render_template_circles_plot_v2` (plot_recons phase's primitive renderer) — same rendering machinery as the LEFT panel. Y-axis auto-handled by the renderer.

**Adapter approach** (option 2 from your earlier pick): bypass plot_recons phase's file-load (no fake gtr.pkl) — call the renderer directly with `branch_morphology` payload built from radivojevic's Stage 3 links. Each link's method (direct / skel-assisted / indirect) becomes one "branch" → 3-branch payload.

**Channel coverage by method** (unique channels with ≥1 link touching them):
- direct: 6403 channels from 29680 links (≈48% of 13439 channels — way too many for a real axon; downstream of Stage 2 over-skeletonization)
- skel-assisted: 1170 channels from 775 links
- indirect: 686 channels from 393 links

**Standalone radivojevic PNG (no compositing)**: `radivojevic_plot_recons_style.png`.
**Standalone SVG**: `radivojevic_plot_recons_style.svg` (plot_recons renderer outputs both).

Stage 2 bug investigation still the bigger fish — even with consistent rendering, the visual will read "wrong" until the binarization fires correctly.

#### Paper-vs-code Stage 2 bug found + more frames plotted (per user request)

**Read figures-v1 PDF** — Figure 4B (Skeletonization-assisted interconnection) is explicit: "Signal averaged over the two consecutive timeframes (Δt=100 μs) is skeletonized to infer directionality of the propagating signal." — i.e. skeletonize PAIRS of averaged frames, not single frames.

**Our `core/skeletonization.py:122`** does `skeletons[t] = skeletonize(frame, method=...)` — per-frame, no averaging. **THIS IS THE BUG.** Each frame's noise pattern gets binarized independently → dense skeleton on every frame → "random segmented lines all over the place" per user.

**New artifacts at `dev_outputs/.../unit_598_radivojevic/per_stage_debug/`**:
- `skeleton_grid_24frames.png` — 24 evenly-spaced frames showing the per-frame skeleton (user can see the dense-blob-everywhere pattern).
- `binary_footprint_grid_24frames.png` — 24 frames showing the binary_footprint (pre-skeletonization). If this is dense everywhere, threshold is too low.

**Other paper-spec deviations the audit surfaces** (Stage 2):
- Paper: noise from "background noise was sampled across all electrodes during periods when the neuron was inactive". Our `noise_estimator='mad'` computes noise from THIS template, which on a dense merged template will under-estimate (signal dominates) → too-low noise_std → too many pixels exceed threshold.
- Paper Step IV (skel-assisted): averaged over 2 timeframes at Δt=100μs (at 20 kHz raw → upsampled rate). At 10 kHz raw + upsample 10 = 100 kHz, Δt=10μs. Need to confirm if 2-timeframe averaging is at raw rate or upsampled rate.
- Paper Step V (indirect): averaged over 3 timeframes at Δt=150μs.

**Fix design (slice TBD in radivojevic_recon_algo_plan)**:
1. Pre-averaging step: build `averaged_pairs[t] = (images[t] + images[t+1]) / 2` for Step IV.
2. Build `averaged_triples[t] = (images[t-1] + images[t] + images[t+1]) / 3` for Step V.
3. Skeletonize the averaged stacks, not the raw stack.
4. Noise estimation: separate inactivity-window noise estimator (paper-faithful), distinct from per-template MAD. Currently MAD over-includes signal as noise contributor.

#### Correction (user 2026-05-21): per-frame density framing

User: "I don't think *each frame* will have a dense skeleton. Just, the skeletons should be aware of each other and the final combined version should be dense."

Re-framing:
- **Expected correct behavior**: per-frame skeleton = sparse (just the wavefront region where the AP is at that instant); UNION across all frames = dense + axon-shaped (the full arbor traced by the propagating signal over time).
- **What "aware of each other" means**: paper's Step IV averages signal over 2 consecutive timeframes before skeletonizing, which couples adjacent frames' wavefronts; downstream, the link-from-skeleton step traces those coupled segments into continuous paths.
- **Our current bug**: per-frame skeleton is too dense AND the frames don't relate to each other → union is dense-EVERYWHERE, not dense-AND-axon-shaped.

To diagnose which sub-bug dominates, the 24-frame grid lets us see: (a) is each frame uniformly noise-dense? (likely b/c noise estimator under-estimates noise + binarization too aggressive — fix the threshold), OR (b) does each frame have wavefront-like sparse structure but the wavefronts are too thick? (fix is the averaging step + maybe threshold).

Slice 7-equivalent for radivojevic_recon_algo_plan: file the two-fix design (noise + averaging) as a single slice once user confirms diagnosis from the grid.

#### v4 — noise-scaling experiment per user feedback (2026-05-21)

**Root cause confirmed**: template noise = raw noise / √(n_spikes averaged). When radivojevic measures noise from the template, it's ~√n smaller than the paper's intended raw-recording noise — making thresholds way too loose (45209 peaks vs paper's ~150-200).

**v4 quick test** at `dev_outputs/.../unit_598_radivojevic_v4_scaled_all/`:
- noise_estimator='window' (pre-spike samples 0-15 from template)
- n_std_step1=90 (= 9 × 10 proxy for √n), n_std_step2=20, n_std_step3=10
- binarization_k_stage2=10 (was 1.0)
- Other knobs at paper defaults (radius 50/100μm, etc).

**Results**:
| Metric | v1 (MAD) | v2 (window only) | v3 (window + n_std×10) | v4 (window + n_std×10 + bin_k×10) |
|---|---|---|---|---|
| Stage 1 peaks | 45209 | 23637 | **700** | 700 |
| Stage 2 skel/frame | 5142 | 2753 | 2753 | **30** |
| Stage 3 links | 30848 | 15606 | 508 | 508 |

Down to 700 peaks + 30 skel pixels per frame + 508 links. **This is paper-order-of-magnitude.**

**Artifacts**: `radivojevic_v4.png`, `comparison_v4.png` in the v4 dir.

**Verified**: spatial radius logic (paper Steps II + III) IS correctly applied — `stage_1.py:186` step2 uses spatial_radius_um=50μm; `stage_1.py:196` step3 uses 100μm. Threshold scaling, not radius, was the bug shape.

**Proper fix design** (to file as slice):
1. Use SpikeInterface analyzer's `noise_levels` extension (compute_noise_levels on raw recording, paper-faithful) — currently not computed; would need an analyzer policy update.
2. OR scale window-estimator noise by √(n_spikes_per_unit) automatically (use spike_times to count).
3. OR (simplest stopgap) expose a `noise_scale` multiplier in radivojevic API — caller can pass √n manually.

Knobs that worked here are HACKED (n_std=90 isn't a real paper value). Need (1) or (2) for production.

#### v5 — slices 10/11/12/14 applied (paper-spec algorithm fixes)

**Slices shipped in sibling radivojevic2023_recon_algo** (commits in `~/dev/pkgs/radivojevic2023_recon_algo/`):
- `2e1caa7` slice 10: max_distance_um_indirect 200→400μm
- `775177e` slice 14: recursive step 1B/1C expansion (rider chain)
- `58eae8d` slice 12: separate analysis_frame_us from upsample_factor
- `243d13f` slice 11: per-channel noise from external source
- `dbd0881` slice 11 followup: skeletonization accepts per-channel noise
- 168→172 tests pass throughout

**Knob-by-knob comparison on unit_598** (`dev_outputs/.../v5_comparison/comparison_knobs.png`, same threshold scaling 90/20/10):

| Config | peaks | chans | frames covered | links |
|---|---|---|---|---|
| v4 baseline (no slices) | 700 | 372 | 51/690 | 508 |
| slice 14 only (iterative) | **2042** | **768** | **84/690** | **1436** |
| slice 12 only (analysis_frame=50μs) | 1167 | 496 | 13/138 | 788 |
| all 10+12+14 | **2118** | **825** | 26/138 | 1418 |

**Slice 14 (rider chain) alone DOUBLES channel coverage** — exactly what user predicted. The iterative expansion extends detection across more frames (51→84 at upsample=10) so the rider can travel further along the propagating wavefront.

**Slice 11 deferred for proper raw-recording-noise**: my v5 per-channel proxy (template pre-spike STD) was wrong units (μV vs μV/μs after derivative+decimation). True paper-spec needs:
- SpikeInterface analyzer's `compute_noise_levels` extension on the raw recording during inactive periods of unit 598
- OR equivalent: extract raw samples at non-spike times → per-channel STD on dV/dt of those samples

Slice 13 (Stage 2 on-demand pair-averaged skeletonization) NOT shipped — biggest architectural change, deferred.

**Sibling pushes** pending — commits are local-only until pushed to GitHub.

Acceptance criteria from slice 15 NOT met (channels don't trace axon shape) — proper noise estimator (slice 11 done right) is the remaining gap. Slice 14 + 12 + 10 are correct + tested.
