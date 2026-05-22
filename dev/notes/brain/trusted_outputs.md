# Trusted outputs — the verifier's anchor

The root of the verification trust chain. User-anchored: only the USER pins outputs to "trusted"; loop NEVER promotes without explicit approval. Everything downstream — tests, smoke validations, refinement-gate checks — inherits trust from this file.

**Reading rule for the loop**: before accepting any refinement as "shipped", verify the relevant trusted output (full match OR invariant assertions, per the entry) still holds. If it doesn't: refinement is REJECTED + auto-rollback + escalate.

**Writing rule**: loop appends candidates to "Proposed for promotion" only. User edits the "Tier 1 — TRUSTED" section.

---

## Trust tiers

| Tier | What it is | Loop behavior |
|---|---|---|
| **Tier 1 — TRUSTED** | USER has eyeballed the output OR the invariant assertions, marked them correct. The reference. | Loop MUST verify against this. Failure = blocking. |
| **Tier 2 — ADVISORY** | Existing tests + outputs whose correctness wasn't formally user-verified. Demoted per the trust-anchor-on-outputs decision (2026-05-21). | Loop runs them; failure ≠ blocking, pass ≠ green. Surfaces failures to user as flags. NEVER edits to make passing. |
| **Tier 3 — PROVISIONAL** | NEW outputs from new code, no reference yet. | Loop tags every downstream artifact as "rests on provisional output X". Refinement built on provisional outputs inherits the uncertainty. User promotes to trusted when ready. |

---

## Tier 1 — TRUSTED outputs (USER-ANCHORED)

### TR-001 — Reconstruct stage on M08073/000208/well000 (DIV 36)
- **Source cohort**: `260326/M08073/AxonTracking/000208/well000` (DIV 36, 80k DMEM, MaxTwo @10 kHz)
- **Reference path**: `/pscratch/sd/a/adammwea/analyzed_data/Media_Density_T5_02182026_AR_axon_analysis_AW/.../260326/M08073/AxonTracking/000208/well000/`
- **Established**: 2026-05-18 via `smoke_log.md` entry "Concat-analyzer rip-out + SLAy aux-tsv sync regression check"
- **What's trusted (USER-eyeballed)**:
  - `spike_clusters.npy`: 377 post-SLAy unit IDs
  - `cluster_KSLabel.tsv` after aux-tsv sync: 287 good + 312 mua = 599 rows
  - Recon merged templates: **176**
  - Recon per-unit outputs: 176 (matches good-label count; mua not reconstructed per `unit_label_filter`)
- **Invariant assertions** (loop should run these against any pipeline change touching recon stage):
  - `len(spike_clusters_unique) == 377` — post-SLAy unit count
  - `len(merged_templates) == 176` — reconstructed-template count (THE primary regression baseline)
  - `cluster_group.tsv good_count == 287` AND `mua_count == 312`
  - Per-unit `merged_template.npy` exists for each of the 176 good units; shape sparse-channels × n_samples
- **Failure mode**: any refinement that drops the merged-template count from 176 OR breaks the per-unit existence invariant gets ROLLED BACK + escalated.

### (Proposed for promotion — user reviews + pins)

Z1 ranked 2026-05-21 (see `brain/dependency_graph.md` §5 for the full justification including coverage estimates). Triage each into: ✅ PIN (you eyeball + approve now) / 💵 ACQUIRE LATER (cheap but needs a smoke first) / ❓ DEFER (expensive or needs other work to land first) / ❌ SKIP.

#### TR-CAND-001 — `axon_velocity_gtrs` output for unit_0598 (9-branch reference)
- **Cohort**: `260326/M08073/000208/well000` (DIV 36)
- **Path**: existing reference data at `/pscratch/sd/a/adammwea/analyzed_data/.../well000/recon_outputs/units/0598/{gtr.pkl, branches.json, ...}` + the existing `plot_recons` rendering for that unit
- **Covers**: J2 (gtr-shape) + 11 recon-stage visualization phases downstream
- **Cost**: cheap — file exists; you eyeball the plot_recons rendering for unit_0598 + approve
- **Trust handle**: "yes this is what an axon arbor reconstruction should look like"
- **Why high-leverage**: this is the COMPARATOR side of the Radivojevic DC-001 differential checks. Without this pinned, the Radivojevic comparison has no reference.

#### TR-CAND-002 — kssynth `synth_sorter_output/` per-unit dirs on M08073/well000/DIV 36
- **Path**: `<dev_outputs>/kssynth_slice3b/.../well000/recon_outputs/synth_sorter_output/per_unit/unit_<id>/{merged_template.npy, merged_channel_locations.npy}`
- **Covers**: J5 + kssynth slice 5 retirement gate
- **Cost**: HEAVY — requires kssynth slice 3b heavy on a real allocation (already queued in `trackers/salloc_smokes_queued.md`)
- **Trust handle**: per-unit dir count matches TR-001 (176); sha256(kssynth merged_template) == sha256(build_templates merged_template) for the same unit (loop computes automatically once both exist)

#### TR-CAND-003 — `save_rec_metadata` output for M08073 (sampling rate + device info)
- **Path**: existing reference data; metadata.json under one preprocess_outputs dir for any DIV
- **Covers**: J11 (sample rate authoritative source)
- **Cost**: trivial — file exists; confirm `sampling_rate_hz: 10000` for MaxTwo
- **Trust handle**: device + sample rate confirmation

#### TR-CAND-004 — `analysis.unitmatch` match table for (M08073, well000, ≥3 DIVs)
- **Path**: `<output>/analysis/unitmatch/<chip>/well000/match_table.parquet`
- **Covers**: O2 DOD + cross-session color anchor for chip_layout_phase_split + future propagation_video
- **Cost**: HEAVY — requires kssynth slice 5 + unitmatch_phase slice 5 both shipped
- **Trust handle**: row counts in sensible bands; UID chains consistent; spot-check a few matches by hand

#### TR-CAND-005 — `radivojevic2023_recon_algo.reconstruct()` output for unit_0598
- **Path**: TBD when radivojevic_recon phase ships
- **Covers**: O3 DOD (paired with TR-CAND-001 as DC-001 differential check input)
- **Cost**: gated on TR-CAND-001 pinned AND TR-CAND-002 acquired (needs the merged_template input)
- **Trust handle**: NOT standalone — it's the differential-similarity check vs TR-CAND-001 per DC-001

#### TR-CAND-006 — `preprocess_outputs/segments/<seg>/recording.bin` integrity
- **Path**: existing reference data; pick any `260326/M08073/.../000_rec0000/recording.bin`
- **Covers**: J12 + preprocess-stage integrity (catches silent preprocess breakage)
- **Cost**: trivial — sha256 + file-size check on the existing file
- **Trust handle**: file-level integrity; not semantic.

**Triage recommendation** (loop's): start with **TR-CAND-001 + 003 + 006** (the cheap bucket) — together with TR-001 they cover J1+J2+J3+J4+J6+J11+J12+J13 transitively. That's a substantial chunk of the recon-side pipeline pinned BEFORE any heavy work runs. TR-CAND-002 and TR-CAND-004 are heavy and gated on smokes; TR-CAND-005 is gated on radivojevic phase implementation. Working through 001/003/006 first front-loads the user's time to maximum-coverage points.

---

## Tier 2 — ADVISORY (existing tests, demoted)

The repository's existing test suite. Loop runs them; failures flagged not blocking; passes are not equivalent to verification.

- `src/axon_recon/pipeline/tests/` (457 passed at last full run, 14 pre-existing failures noted in parallelism_post_migration_cleanup_plan §6)
- `src/axon_recon/pipeline/stages/*/tests/` (per-stage test dirs)
- Sibling-package test suites:
  - kssynth (58 tests)
  - unitlink (53 tests)
  - radivojevic2023_recon_algo (81 tests, all on synthetic fixtures)
  - SLAy
  - UnitMatchPy

Pre-existing failures are listed in plan/tracker docs, not promoted here. Loop SHOULD NOT spend cycles "fixing" pre-existing failures unless the user explicitly authorizes it via an injection.

---

## Tier 3 — PROVISIONAL (new outputs awaiting promotion)

### PR-001 — Radivojevic Stage 1+2+3 output on kilosort cluster 67 (M08073/000208/well000 DIV 36)
- **Path**: `/pscratch/sd/a/adammwea/dev_outputs/radivojevic_first_run/cluster_67/radivojevic_recon/`
- **Why provisional**: input was a kilosort cluster substitute (not the high-branch post-merge unit the user asked for); rendering was loop-invented PNG (not via plot_recons); no comparator. User rejected this as a verification artifact.
- **Promotion path**: superseded entirely once the apples-to-apples comparison ships (radivojevic on the actual high-branch unit_0598's merged_template, rendered via plot_recons, compared against axon_velocity_gtrs). DO NOT promote this PR-001 entry; it'll be retired when its replacement lands.

---

## Differential / invariant checks for cross-algorithm comparisons

For comparing two algorithms producing similar-but-not-identical outputs from the SAME input. Approved by user 2026-05-21: "merged_template inputs should be identical. Outputs should be something similar to the gtr object that specifies branches, lengths, velocities, selected channels, etc. Since the algorithms are different, they are liable to produce different reconstructions, but they should generally be similar in size (area covered by the neuron), similar in branching, similar in velocity, similar in soma localization."

### DC-001 — Radivojevic vs axon_velocity_gtrs (same merged_template input)

**Hard invariants** (these MUST hold; failure = bug):
- Input identity: `sha256(merged_template_radivojevic_input) == sha256(merged_template_av_gtrs_input)` — byte-for-byte
- Input identity: `sha256(merged_channel_locations_radivojevic) == sha256(merged_channel_locations_av_gtrs)` — byte-for-byte
- Both algorithms produced non-empty output

**Soft invariants** (similarity bands; tolerance TBD by triage; failure = surface to user not auto-block):
- **SIZE (area covered)**: `|area_R - area_av| / area_av < TBD` (suggested starting tolerance: 0.5 = ±50%; calibrate after first ~3-5 unit comparisons)
- **BRANCHING (branch count)**: `|n_branches_R - n_branches_av| <= max(2, 0.5 * n_branches_av)` — within 50% OR within 2 absolute, whichever is more permissive
- **VELOCITY (median + spread)**: `|median_velocity_R - median_velocity_av| / median_velocity_av < TBD` (suggested 0.3 = ±30%); plus KL-divergence of velocity distributions reported as a soft signal (no hard threshold yet)
- **SOMA LOCALIZATION (xy of detected soma channel)**: `euclidean(soma_R, soma_av) < TBD` (suggested 35 μm = 2 electrode pitches; the paper's algorithm-degradation threshold)
- **SELECTED CHANNELS (set overlap)**: `jaccard(channels_R, channels_av) > TBD` (suggested 0.3 = 30% overlap; very loose since algorithms select differently)

**What this enables**:
- Per-unit "comparison.png" via plot_recons (manual visual review)
- Per-unit "comparison.json" with all 5 numerical invariants computed (automated; logged to smoke_log; flagged-to-user if soft invariants fail)
- Across N units, aggregate "% of units where Radivojevic stays within similarity bands" — a single-number health metric for the algorithm port

**Manual review tier** (user-only, per 2026-05-21): plotting style, "does it look like an axon" qualitative read, soma detection correctness in edge cases. These can't be auto-asserted.

**Calibration plan** (to size the TBDs):
- First 3 units worth of comparisons → loop computes the 5 metrics → user picks tolerances by inspecting distributions.
- After ~5-10 units, the tolerances harden.

---

## Where this file gets read

Loop reads this file early in its entry protocol (per `brain/README.md`). When the loop's current slice touches:
- A recon-stage phase → read TR-001 + check it's still met
- A radivojevic comparison → read DC-001 + apply the differential checks
- Any new phase producing output → propose a PR-xxx provisional entry; never claim "trusted" without user approval
