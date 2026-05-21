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
- **Dashboard slice 7 tertiary-grouping UX**: third grouping dropdown for box/bar plots. Two reasonable UX paths:
  - (A) **Faceted small-multiples** — one plot per tertiary value. Scales visually with up to ~6-9 facet values; breaks down for high-cardinality tertiary (e.g. DIV ∈ {6,8,10,...,32}). Easy to read individual sub-plots; harder to compare values across sub-plots.
  - (B) **Nested hierarchical X-axis** — render primary × secondary × tertiary as compound x-tick labels (e.g. `wt | DIV10 | media_a`, `wt | DIV10 | media_b`, …). Scales to higher cardinality; keeps the comparison axis intact; visually busier. Plotly supports this via `category_orders` + multi-level group_col.
  - **Recommendation**: ship (B) first because it's the lower-risk extension of the existing box-plot rendering (slice 6 already renders nested groups for secondary; tertiary is just another dimension to fold into the category sort). Add (A) later as an opt-in `tertiary_mode: facet` knob if users want it. **User input wanted**: confirm (B)-first, or override to (A) if you want small-multiples as the default. **Resolution criterion**: user picks one; loop ships that slice 7.

## Per-slice empirical findings

- **resources.profiles elimination — per-srun-flag fallback when slot is missing**: when `--cpus-per-task` is passed on the command line, does the resolver use that directly, or compute from `os.sched_getaffinity(0)`? Both are reasonable; pick during the implementation slice. See `trackers/tech_debt.md` §"Minimize / eliminate `resources.profiles`".

- **Auto-restart with chip-well-group phase scope** (`unitmatch` phase): the auto-restart logic walks `phase_sequence` per target. For chip-well groups, the "target" is a group, not a (dataset, well) pair. Verify the logic generalizes when the unitmatch phase lands.

## Radivojevic slice 1 user-gate review

`radivojevic_recon_algo_plan` slice 1 SHIPPED 2026-05-21. Full deliverables in:
- `dev/notes/refs/radivojevic2023_paper.md` (citation + data/code availability)
- `dev/notes/refs/radivojevic2023_algorithm_summary.md` (algorithm spec + input compat map)

Per current_state.md "PRE-OVERNIGHT CLEARANCES" item #2, the loop pre-approval lets it proceed into slice 2 (sibling-package scaffold) WITHOUT pausing. Slice 3 (core algorithm impl) DOES gate on user review of these questions:

1. **Paper identity confirmed?** Target paper is *Radivojevic & Rostedt Punga (2023), Functional imaging of conduction dynamics in cortical and spinal axons, eLife 12:e86512, DOI 10.7554/eLife.86512*. Please verify.

2. **Clean-room approach confirmed?** Methodical search (eLife article page, bioRxiv preprint, ResearchGate, GitHub author search) found NO public code for the 2023 algorithm. Dryad deposit `doi:10.5061/dryad.gxd2547r1` contains DATA only (no code). The adjacent code repos (`axon_velocity` by Buccino 2022 — already used by axon_recon; `hana` by Bullmann 2019) implement DIFFERENT algorithms. Clean-room re-implementation is the only path. Please confirm OR identify a code source the search missed.

3. **🔴 HIGH-IMPACT**: Does the algorithm need the RAW spike-triggered-average per-spike (the SPREAD of arrival times across electrodes), or just the AVERAGED template? The 2023 paper's multi-step tracking may use per-spike data to refine velocity estimates — this is what would distinguish it from Buccino 2022's graph-based approach which only consumes averaged templates. If RAW per-spike STA is needed, we'd need either (a) a new analyzers-phase output that caches per-spike STA arrays, or (b) accept averaged-only as a v1 limitation and document a v2 enhancement path. Please confirm by reading the methods section of the full paper text (PDF rendering not available in the loop's environment — the loop only had access to the abstract + algorithm overview via WebFetch).

4. **Input compat map verification**: see the table in `radivojevic2023_algorithm_summary.md` §"Input compat with axon_velocity_gtrs". The proposed claim is that recon-stage `merged_template.npy` + `merged_channel_locations.npy` + sampling_rate (from analyzer manifest) are sufficient inputs for Radivojevic's algorithm (modulo question 3 above). Please verify.

5. **Hyperparameter defaults — UPDATED 2026-05-21**: ✅ Concrete values FOUND in pre-extracted `notes/archive/old_ai_notes_for_reference/radivojevic_2023_methods_mining.txt` in the sibling-package dir. Defaults captured in `radivojevic2023_algorithm_summary.md`: Step 1 = 9 STD noise, Step 2 = 2 STD / 50 μm radius, Step 3 = 1 STD / 100 μm radius. Direct interconnection = 100 μm, skeleton-assisted = 200 μm. Up-sampling = 200 kHz (Whittaker-Shannon). These will be the slice-3 defaults; please flag if any need adjustment for our data (HD-MEA1k 20 kHz sample rate matches paper).

6. **Algorithm/phase name preference**: paper doesn't pick a nickname. Options for the sibling-package + phase name:
   - `radivojevic_recon` (current sibling-package dir name; descriptive of WHO not WHAT)
   - `electrical_imaging_recon` (descriptive of WHAT — matches the paper's framing of "electrical visualization")
   - `axon_skeleton_recon` (descriptive of the KEY ALGORITHMIC STEP)
   Pick one to lock in for slice 4 onward.
