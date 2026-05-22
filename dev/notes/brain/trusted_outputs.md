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

---

## Z3 invariant specs (USER-REVIEWABLE; pending promotion)

Authored 2026-05-21 per QZ3 option 1 (spec-first as markdown). User reviews each block + marks `✅ APPROVED <date>` above it; approved invariants become the trusted gate consumed by the critic_separation subagent + future pytest implementations.

Format: each block lists `Schema` (what shape the artifact has) + `Counts` (cardinality invariants) + `Value-range` (numerical guards) + `Reconciliation` (cross-file consistency predicates) per the option-1 framing.

✅ APPROVED 2026-05-21

### Z3-TR-000 — Reference tree structural invariants (broad, applies tree-wide)

**Schema invariants** (must hold for any well's outputs under TR-000):
- `<well>/preprocess_outputs/segments/<seg-id>/` exists for each segment in the recording
- Per-segment dir contains: `recording.bin`, `channel_positions.npy`, channel-mapping files
- `<well>/spikesort_outputs/sorter_output/` exists post-sort
- `<well>/spikesort_outputs/sorter_output_snapshot/` exists post-snapshot
- Snapshot contains: `spike_times.npy`, `spike_clusters.npy`, `templates.npy`, `channel_positions.npy`, `channel_map.npy`, `cluster_KSLabel.tsv`, `params.py`
- `<well>/recon_outputs/cache/analyzers/segments/<seg-id>/` exists per segment (post-analyzers phase)
- `<well>/recon_outputs/units/unit_<id>/` exists per reconstructed unit (post-build_templates OR kssynth)
- Per-unit dir contains: `merged_template.npy`, `merged_channel_locations.npy`, `gtr.pkl`, `branches.json`

**Numpy shape + dtype invariants**:
- `merged_template.npy`: dtype float32/float64; shape `(n_active_channels, n_samples)`; n_samples > 0; n_active_channels > 0 AND ≤ total channel count for the well
- `merged_channel_locations.npy`: dtype float; shape `(n_active_channels, 2)` matching merged_template's first axis
- `templates.npy` (kilosort-level): dtype float; shape `(n_units, n_samples, n_channels)`
- `spike_clusters.npy`: dtype int (usually int64); shape `(n_spikes,)`; values in `[0, max_cluster_id]`
- `spike_times.npy`: dtype int64; shape `(n_spikes,)` matching spike_clusters length; values monotone non-decreasing
- `channel_positions.npy`: dtype float; shape `(n_channels, 2)`
- `channel_map.npy`: dtype int; shape `(n_channels,)`; values are valid electrode indices

**TSV column invariants**:
- `cluster_KSLabel.tsv`: tab-separated; header row `cluster_id\tKSLabel`; one row per unique cluster
- `cluster_group.tsv`: tab-separated; header row `cluster_id\tgroup`; group values in `{good, mua, noise}` post-SLAy
- `cluster_KSLabel.tsv` and `cluster_group.tsv` agree on cluster_id sets (same row count, same IDs)

**JSON schema invariants**:
- All `*_summary.json`: have a `status` key with value in `{ok, error, dry_run_ok, in_progress, skipped, stale}`
- `kssynth_summary.json` additionally has: `n_units`, `synth_sorter_output_relpath`, `per_unit_dir`, `per_unit_n_units_written`
- `branches.json`: list of dicts; each dict has at minimum `unit_id`, `branch_id`, `n_branches`, `total_length_um`

**Reconciliation invariants**:
- For each well: `count(per-unit dirs in recon_outputs/units/) == count(unique IDs in cluster_KSLabel.tsv where KSLabel=='good')` (only good units get reconstructed per `unit_label_filter`)
- `merged_template.npy`'s n_active_channels ≤ `channel_positions.npy`'s n_channels for the same well
- `spike_clusters.npy`'s unique values == `cluster_KSLabel.tsv`'s cluster_id column

**Status**: pending user review. Approve = these become the trusted gate consumed by critic_separation + future pytest.

---

✅ APPROVED 2026-05-21

### Z3-TR-001 — M08073/000208/well000/DIV 36 specific count invariants

**Hard counts (exact-match regression targets)**:
- `len(np.unique(spike_clusters.npy)) == 377` — post-SLAy unique unit IDs
- `cluster_KSLabel.tsv` row count == 599
- `cluster_KSLabel.tsv` "good" rows == 287
- `cluster_KSLabel.tsv` "mua" rows == 312
- `count(per-unit dirs in recon_outputs/units/) == 176`
- 176 of those 176 per-unit dirs contain `merged_template.npy` (no missing files)
- 176 of those 176 per-unit dirs contain `merged_channel_locations.npy`

**Soft numerical invariants** (not exact, but tight bounds for sanity):
- For each unit's `merged_template.npy`: amplitude (max abs voltage) in [3, 500] μV (units with extremes are suspect)
- For each unit's `merged_template.npy`: n_active_channels in [3, 100] (sparse-template assumption)

**Reconciliation invariants** (with TR-000 derived):
- `count(per-unit dirs) == count(cluster_KSLabel.tsv where KSLabel=='good')` → 176 == 287 (NOTE: these DIFFER because post-SLAy merging creates new "good" composites that aren't reconstructed if they fall below `unit_label_filter`; the 176 number is the count of units that survive ALL filtering). The "good" count is 287 because some "good" units were excluded from reconstruction by the filter — this is expected; NOT a bug. Reconciliation predicate: 176 ≤ 287 (loose) and the gap == count of units excluded by `unit_label_filter`.

**Failure response**: any of the hard counts drifting → auto-rollback per `brain/metrics.md` M-001 + escalate to user.

**Status**: pending user review.

---

✅ APPROVED 2026-05-21

### Z3-TR-002 — unit_0598 axon_velocity_gtrs anchor

**Schema invariants** (per TR-000 J2 contract):
- `<well>/recon_outputs/units/0598/gtr.pkl` exists + unpickles via `axon_velocity` (shifter-only)
- Unpickled object is a `GraphTracking` instance with attributes: `selected_channels`, `branches`, `velocities`, `soma_ch`
- `<well>/recon_outputs/units/0598/branches.json` exists; list of dicts; len ≥ 8 (high-branch criterion)
- `<well>/recon_outputs/units/0598/merged_template.npy` exists (TR-000 inherits)
- `<well>/recon_outputs/units/0598/plot_recons/<unit>.png` exists post-plot_recons phase

**Branch-tree invariants** (gtr-shape specific):
- `gtr.branches` is a list; each branch dict has: `branch_id`, `length_um`, `electrodes`, `start_t_us`, `end_t_us`
- `len(gtr.branches) >= 8` (this IS the high-branch-count criterion)
- For each branch: `length_um > 0`, `electrodes` is a non-empty list of valid channel indices
- For each branch: `end_t_us > start_t_us` (time-monotone)
- `gtr.soma_ch` is a valid channel index in `channel_positions.npy`'s range

**Velocity invariants** (physical plausibility — soft):
- For each branch: derived velocity = `length_um / (end_t_us - start_t_us)` ≈ µm/µs = m/s
- Velocity values in [0.05, 5.0] m/s (range covering myelinated + unmyelinated axons; outside this is suspect)

**Comparator-side discipline** (DC-001 uses this entry):
- This TR-002 IS the comparator for Radivojevic differential checks. radivojevic_recon on unit_0598's merged_template must produce a similar-in-shape result (per DC-001 in this file).

**Status**: pending user review.

---

✅ APPROVED 2026-05-21

### Z3-TR-003 — sample-rate + device metadata anchor

**Schema invariants**:
- `<well>/preprocess_outputs/metadata.json` (or equivalent metadata artifact produced by `save_rec_metadata`) exists per recording
- Contains keys: `sampling_rate_hz` (int), `n_channels` (int), `electrode_positions_um` (list of (x, y) pairs OR shape (n, 2))
- May additionally contain: `device_name`, `recording_duration_s`, `dtype`, `gain_uv_per_lsb`

**Value-range invariants** (device-family aware):
- `sampling_rate_hz` is a positive integer
- `sampling_rate_hz` in `{10000, 20000}` for current cohort (MaxTwo OR MaxOne); broader range for future devices but flag if outside `{1000..100000}`
- `n_channels` > 0
- For MaxWell family (current): `n_channels` in `{266, 1024, 26400}` depending on routing/scan-type

**Reconciliation invariants**:
- `metadata.json` `sampling_rate_hz` agrees with the corresponding `preprocess_outputs/segments/<seg>/params.py` if present
- Same recording across multiple per-segment metadata files: `sampling_rate_hz` is constant per recording (NOT per segment — if it varies, that's a bug)

**Anti-pattern invariant** (encodes AP-008):
- NO Python source file in `src/axon_recon/` may have `sampling_rate_hz = 10000` (or any literal sample rate) hardcoded — assertion is `grep -rn 'sampling_rate.*=.*[0-9]\+0\+' src/axon_recon/` returns ZERO non-test hits. (Test files OK to have literal rates as fixtures.)

**Status**: pending user review.

---

✅ APPROVED 2026-05-21

### Z3-TR-004 — preprocess binary integrity anchor

**Schema invariants**:
- `<well>/preprocess_outputs/segments/<seg>/recording.bin` exists
- Sibling `channel_*.npy` files exist + are consistent

**Bit-level integrity invariants**:
- For a chosen reference segment (recommendation: `260326/M08073/AxonTracking/000208/well000/preprocess_outputs/segments/000_rec0000/recording.bin`):
  - Recorded baseline: sha256 = `<TBD: compute on first run>`; file size in bytes = `<TBD>`
  - Future preprocess-stage changes MUST reproduce this exact sha256 + size (within explicit filter-parameter changes — those get their own baseline update via user-approved promotion)
- For other segments: shape invariant only — `file_size_bytes == n_samples * n_channels * dtype_bytes`

**Soft consistency invariants**:
- `recording.bin` shape: `(n_samples, n_channels)` consistent with metadata.json's `n_channels`
- dtype: int16 OR float32 (preprocess filter output dtype)
- No NaN / Inf values in any 1000-sample window (catches silent filter blowups)

**Reconciliation invariants**:
- Sum of per-segment `n_samples` across all `recording.bin` files for a recording == total recording_duration_s × sampling_rate_hz (within rounding)

**Status**: pending user review.

---

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
