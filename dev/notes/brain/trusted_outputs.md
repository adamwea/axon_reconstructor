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

## Meta-rule (USER 2026-05-21)

> *"Pretty much anything in the reference data is pinned. New outputs should be identical or at least similar in shape to the current reference data."*

**Implications**:
1. **The entire reference data tree is Tier 1 trusted by default** — `/pscratch/sd/a/adammwea/analyzed_data/Media_Density_T5_02182026_AR_axon_analysis_AW/` and its subtree of per-DIV / per-well pipeline outputs. Specific entries below pin the high-leverage invariants; the rule generalizes to anything else in the tree.
2. **New outputs must match reference shape unless explicitly justified.** Loop's default behavior when shipping a new output (kssynth merged_template, radivojevic gtr-equivalent, etc.): compare on-disk structure / numpy shapes / JSON schemas / TSV columns against the analogous reference output. Mismatch = STOP AND ASK (multiple-choice per B1/B2), do NOT silently produce a different shape.
3. **For genuinely new outputs without a reference analog** (e.g. unitmatch match tables, propagation videos): differential-check framework (DC-xxx) applies — derive invariants from related-but-not-identical reference artifacts.
4. **What's NOT reference data**: `dev_outputs/` (iteration sandbox, disposable), `raw_data/` (input not output), anything the loop generates post-2026-05-21 (provisional until promoted).

---

## Tier 1 — TRUSTED outputs (USER-ANCHORED)

### TR-000 — The entire reference data tree (BLANKET PIN per user 2026-05-21)
- **Reference root**: `/pscratch/sd/a/adammwea/analyzed_data/Media_Density_T5_02182026_AR_axon_analysis_AW/`
- **What's trusted**: the on-disk structure + file shapes + JSON/TSV schemas + numpy dtypes/dimensions of every pipeline output under this tree, for every (date, chip, well, DIV) combo.
- **What the loop derives from this as invariants** (these are the structural invariants enforced on any NEW output that claims to be the same kind of artifact):
  - Per-stage output dir layout (e.g. `preprocess_outputs/segments/<seg-id>/`, `spikesort_outputs/sorter_output_snapshot/`, `recon_outputs/cache/analyzers/segments/<seg-id>/`, `recon_outputs/units/unit_<id>/`)
  - File naming conventions (e.g. `merged_template.npy`, `merged_channel_locations.npy`, `gtr.pkl`, `branches.json`, `spike_clusters.npy`)
  - Numpy shapes + dtypes per file (per-unit merged_template = `(n_active_channels, n_samples)` sparsified; merged_channel_locations = `(n_active_channels, 2)`; spike_clusters = `(n_spikes,)` int64; etc.)
  - JSON schemas for `*_summary.json` artifacts
  - TSV column layouts (`cluster_KSLabel.tsv` columns; `cluster_group.tsv` columns; etc.)
- **What's NOT trusted by blanket**: anything OUTSIDE this tree (dev_outputs/, raw_data/ inputs, anything generated post-2026-05-21 by the loop).
- **Failure mode**: any new pipeline output that doesn't match the analogous reference's shape → STOP AND ASK; do NOT silently produce a divergent shape.
- **Why so broad**: per user 2026-05-21 "pretty much anything in the reference data is pinned. New outputs should be identical or at least similar in shape to the current reference data." The reference data IS the contract; the trust anchor is the whole tree, not just a curated subset.

### TR-001 — Reconstruct stage on M08073/000208/well000 (DIV 36) — specific count invariants
- **Sub-anchor of TR-000** for the most-frequently-validated well. Established 2026-05-18 via smoke_log "Concat-analyzer rip-out + SLAy aux-tsv sync regression check".
- **Source cohort**: `260326/M08073/AxonTracking/000208/well000` (DIV 36, 80k DMEM, MaxTwo @10 kHz)
- **Specific count invariants** (anchored regression checks for recon-stage changes):
  - `spike_clusters.npy`: 377 post-SLAy unit IDs
  - `cluster_KSLabel.tsv` (post aux-tsv sync): 287 good + 312 mua = 599 rows
  - Recon merged templates: **176**
  - Recon per-unit outputs: 176 (matches good-label count; mua not reconstructed per `unit_label_filter`)
- **Failure mode**: count drop = ROLLED BACK + escalated.

### TR-002 — axon_velocity_gtrs output for unit_0598 (was TR-CAND-001; pinned by user 2026-05-21)
- **Path**: `<TR-000 root>/.../260326/M08073/AxonTracking/000208/well000/recon_outputs/units/0598/{gtr.pkl, branches.json, ...}` + the existing plot_recons rendering for that unit
- **Covers**: J2 (gtr-shape) + the comparator side of DC-001 (Radivojevic differential checks)
- **Trust handle**: "this is what a 9-branch axon arbor reconstruction should look like"
- **Use**: the reference output that radivojevic_recon must produce a similar-in-shape result against

### TR-003 — save_rec_metadata output (sampling rate + device info) — was TR-CAND-003
- **Path**: `<TR-000 root>/.../<any DIV>/preprocess_outputs/metadata.json` (or equivalent per-recording metadata artifact)
- **Covers**: J11 (sample rate authoritative source) per `feedback-axon-recon-device-diversity` auto-memory
- **Trust handle**: `sampling_rate_hz: 10000` for MaxTwo cohort confirmed
- **Use**: every analysis phase reading sample rate must consume this file, NEVER hardcode

### TR-004 — preprocess binary integrity (was TR-CAND-006)
- **Path**: any `<TR-000 root>/.../preprocess_outputs/segments/<seg>/recording.bin`
- **Covers**: J12 (preprocess-stage binary output integrity)
- **Trust handle**: sha256 + file-size of the first per-segment binary on M08073/well000/DIV 36 (or any reference well). Loop computes the sha + records it as the regression target.
- **Use**: any preprocess-stage change must reproduce the same binary content (within filter-parameter expectations)

### (Proposed for promotion — user reviews + pins)

Z1 ranked 2026-05-21 (see `brain/dependency_graph.md` §5 for the full justification including coverage estimates). Triage each into: ✅ PIN (you eyeball + approve now) / 💵 ACQUIRE LATER (cheap but needs a smoke first) / ❓ DEFER (expensive or needs other work to land first) / ❌ SKIP.

#### TR-CAND-001 — ✅ PINNED as TR-002 (2026-05-21)
*(promoted; see Tier 1 above)*

#### TR-CAND-002 — kssynth `synth_sorter_output/` per-unit dirs on M08073/well000/DIV 36
- **Path**: `<dev_outputs>/kssynth_slice3b/.../well000/recon_outputs/synth_sorter_output/per_unit/unit_<id>/{merged_template.npy, merged_channel_locations.npy}`
- **Covers**: J5 + kssynth slice 5 retirement gate
- **Cost**: HEAVY — requires kssynth slice 3b heavy on a real allocation (already queued in `trackers/salloc_smokes_queued.md`)
- **Trust handle**: per-unit dir count matches TR-001 (176); sha256(kssynth merged_template) == sha256(build_templates merged_template) for the same unit (loop computes automatically once both exist)

#### TR-CAND-003 — ✅ PINNED as TR-003 (2026-05-21)
*(promoted; see Tier 1 above)*

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

#### TR-CAND-006 — ✅ PINNED as TR-004 (2026-05-21)
*(promoted; see Tier 1 above)*

**Status post-2026-05-21 user pin**: TR-CAND-001, 003, 006 all promoted to Tier 1 as TR-002, TR-003, TR-004. Plus TR-000 added (the blanket pin of the entire reference data tree per user's broader statement). TR-CAND-002, 004, 005 remain in this section — they're new outputs that don't exist in the reference tree yet, so they become Tier 3 provisional when generated and only promote to Tier 1 after user review of the actual artifacts.

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
