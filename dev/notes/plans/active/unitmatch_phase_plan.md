# UnitMatch-based cross-DIV unit matching — analysis stage phase

Status: implementation plan (idea → scoped). Sibling to the active parallelism / spikesort plans. Same operating contract: one slice at a time, `claude:` commit prefix, append a line to `debug/commit_log.md` after every commit.

This plan adds a new phase to the **analysis** stage that runs UnitMatch across all DIVs of a single chip-well, producing a unique-unit-id (UID) mapping that links the same biological neuron tracked over multiple recording days. The result feeds a longitudinal view of per-unit metrics that downstream dashboard / compute_metrics consumers can use.

**See also (companion plans):**
- `ks_synthesizer_package_plan.md` — kssynth (sibling pip package) produces SI-compliant synthetic sorter_output folders from segment analyzers. Runs as a recon-stage phase; each `(dataset, well)` ends with a `synth_sorter_output/` directory.
- `unitmatch_runner_package_plan.md` — unitlink (sibling pip package) wraps UnitMatchPy / DeepUnitMatch. Takes N sorter_output paths in, returns a cross-session match table.

This phase is the thin glue between those two packages and the axon_recon data config. Most of the heavy lifting moved out of this plan into the package plans; what remains is **chip-well group discovery + path resolution + one library call per group**.

The plan is layered: v1 ships axon-tracking-only; v2 expands to ingest network scans as additional sessions per `(chip, well)` for better channel-set stability across DIVs. Each layer is an independently testable quality lever.

---

## 0. End-state, in one paragraph

`axon-recon stages analysis.unitmatch` walks the data config, groups recordings by `(chip_id, well_id)`, and for each group with ≥2 DIVs calls `unitlink.match()` over the group's per-DIV synthetic sorter_outputs (produced by the recon-stage `kssynth` phase). The phase writes one match table per group, a UID assignment per `(DIV, original-unit-id)`, and a manifest summary — these are unitlink's outputs landed under the analysis stage's output tree. The phase contributes about 100 lines of axon_recon-specific code; every other line is in the two sibling packages. No changes to the spikesort stage; small change to the reconstruct stage (the `kssynth` phase). Reruns are stage-restartable per the agreed `--force-restart` contract.

---

## 1. Where this phase lives

- **Stage:** `analysis` (existing — currently has only `compute_metrics`).
- **Phase name:** `unitmatch`.
- **Resource class:** new entry `unitmatch` under `resources.phase_budgets`. UnitMatch is CPU-bound (numpy + Naive Bayes), memory-heavy on big well rosters. Sensible budget: `ram_gb: 32`, `analyzer_slots: 1`, `disk_heavy_slots: 1`, `nested_shape: si_njobs`. Sits next to the existing `disk_cleanup` class.
- **Task unit:** **chip-well group**, not per-(dataset, well). One phase invocation processes all DIVs of one chip-well in a single Python process; UnitMatch needs all sessions in memory simultaneously to compute cross-session score matrices.
- **Granularity for `--targets`:** since the unit is a chip-well group, extend `--targets` parsing to accept `chip-well:<chip>:<well>` forms, OR keep `--targets <ds>:<well>` and have the phase fan out to "every group that contains this (ds, well)". Decide in slice 3.

---

## 2. What this phase actually does

The phase is a thin orchestrator. End-to-end runtime sequence:

1. **Discover chip-well groups.** Walk the data config's enabled datasets, group by `(chip_id, well_id)` (chip extracted from h5 path via `_chip_from_short_label` from `status.py`). Filter to groups with ≥2 DIVs. Single-DIV groups get a `skipped: insufficient_sessions` status.

2. **For each group, validate prerequisites.** Each DIV must have completed the recon-stage `kssynth` phase (output directory exists at the canonical path). Each DIV's segment analyzers must be discoverable (so unitlink can compute two-halves waveforms). Fail-fast with actionable error if any DIV is missing either.

3. **Resolve per-session inputs for `unitlink.match()`.** For each DIV in the group:
   - `sorter_output_path` → `<well>/recon_outputs/synth_sorter_output/` (the kssynth output from the recon stage)
   - `analyzers_path` → `<well>/recon_outputs/cache/analyzers/segments/` (the segment analyzers from the recon-stage `analyzers` phase)

4. **Call the library.**
   ```python
   import unitlink
   unitlink.match(
       sessions=[
           {"sorter_output": Path(...), "analyzers": Path(...)}
           for ds in group_members
       ],
       out_folder=group_out_dir,
       backend="classical",
       match_threshold=phase_config.match_threshold,
       param_overrides=phase_config.param_overrides,
   )
   ```
   The library handles: per-session two-halves waveform computation, union channel grid, UnitMatch invocation, output writing.

5. **Aggregate index.** After all groups process, write `<analysis_outputs>/unitmatch/_groups_summary.json` listing every group with `(chip, well, n_divs, n_units_total, n_matched_pairs, n_uids_intermediate, status)`.

Everything substantive — rasterize, two-halves, channel grid, UMPy spine, output schemas — lives in unitlink / kssynth. Search for it there if you're debugging algorithmic behavior.

---

## 3. CLI / YAML schema

YAML add under `stages.analysis.phases`:

```yaml
unitmatch:
  enabled: true
  resource_class: unitmatch
  rel_output_root: unitmatch
  match_threshold: 0.5                 # forwarded to unitlink.match()
  good_units_only: true                # forwarded to unitlink.match()
  backend: classical                   # "classical" (v1) | "deep" (v2)
  include_network_scans: false         # v1 false; v2 flips to true (see §5)
  network_scan_dataset_role: network_scan   # how network-scan datasets are tagged in data config (v2)
  param_overrides: {}                  # forwarded to unitlink as UnitMatchPy params
  outputs:
    match_table:    {relpath: match_table.tsv}
    uid_assignment: {relpath: uid_assignment.tsv}
    prob_matrix:    {relpath: output_prob_matrix.npy, write: true}
    summary_json:   {relpath: summary.json}
```

YAML add `stages.analysis.phase_sequence`:
```yaml
phase_sequence:
  - compute_metrics
  - unitmatch
```

And as a hard requirement, `stages.reconstruct.phases.kssynth.enabled: true` so the synth sorter_outputs this phase consumes actually exist. The phase fails fast with a clear error if any group member is missing its kssynth output.

CLI: no new flags needed for v1 — existing `--targets`, `--target-datasets`, `--target-wells`, `--profile`, `--task-backend`, `--force-restart` cover scoping. v2 may want `--chip <id>` for the group unit; defer until needed.

---

## 4. Implementation slices

One commit per slice. `claude:` prefix. Log to `debug/commit_log.md`.

### Slice 1 — scaffolding (no behavior, no shipped result)

**Status**: SHIPPED in commit `86672e2`. Phase scaffolded with enabled-by-default-false + analysis-stage wiring + dry-run support (dry_run_rollout slice 6, commit `d40684c`).

- `stages/analysis/orchestrators/unitmatch.py` runner shell that returns `{"status": "noop"}`.
- YAML: register the phase under `default.runtime.yml` and `debug.runtime.yml`, **enabled: false** initially.
- Resource class entry under `resources.phase_budgets.unitmatch`.
- Test: `tests/test_unitmatch_phase_disabled_is_skipped.py` proves the phase wires into the analysis stage and is a no-op when `enabled: false`.

### Slice 2 — group discovery + path resolution
**Status**: SHIPPED in commit `a18cb4b`. `core/unitmatch_groups.py` with discover_chip_well_groups + resolve_session_inputs.
- `stages/analysis/core/unitmatch_groups.py`:
  - `discover_chip_well_groups(data_cfg) -> dict[(chip_id, well_id), list[DatasetIndex]]`.
  - `resolve_session_inputs(dataset_index, well_id, output_root) -> {"sorter_output": Path, "analyzers": Path}`.
- Validate that each session's `sorter_output_path` exists; raise with actionable suggestion ("run reconstruct.kssynth first").
- Tests: tmp-path fixture with mixed chip-well configurations; assert group discovery + path resolution.

### Slice 3 — `unitlink.match()` invocation + output landing
**Status**: SHIPPED in commit `575e08b`. Orchestrator invokes `unitlink.match` once per (chip, well) group with idempotent skip on subsequent group targets; output lands at `<output_root>/unitmatch/<chip>/<well>/`.

- The phase's main loop: for each group, call `unitlink.match(...)`, capture the returned result, write the aggregate `_groups_summary.json` after all groups process.
- Outputs land at `<analysis_outputs>/unitmatch/<chip>/<well>/{match_table.tsv, uid_assignment.tsv, summary.json, …}` (unitlink writes these — the phase just chooses the output directory).
- Tests: synthetic 2-group scenario with mocked `unitlink.match()` (the real library is exercised in unitlink's own test suite); assert the phase calls match() once per group, lands files at expected paths, writes _groups_summary.json.

### Slice 4 — wire `--targets` to chip-well groups
**Status**: SHIPPED in commit `a7f8c51`. `--targets chip-well:<chip>:<well>` group form expands against the data config.

- Extend `_parse_targets_pairs_from_args` (or add a parallel `--targets-groups`) to accept `chip-well:<chip>:<well>` forms.
- Update the phase runner to map (dataset, well) targets to their owning groups.
- Tests: `--targets 13:0` → identifies the group containing dataset 13 well 0, runs the phase on the FULL group, not just that one (dataset, well).

### Slice 5 — enable in YAML + smoke run on real data + close-out
- Flip `unitmatch.enabled: true` in `debug.runtime.yml`.
- Prerequisite: kssynth phase landed in recon stage (see `ks_synthesizer_package_plan.md` slice 9) and run for at least one chip-well group.
- Smoke-test on one chip-well group with ≥3 DIVs from the active dataset.
- Record baseline counts (n_units, n_matched_pairs, UID-chain lengths) in `commit_log.md` as the regression target for v2.

### Slice 6 (v2) — ingest network scans as additional same-day sessions per group
Quality-improvement lever, separable from v1. Motivation: same-day same-neuron pairs (network scan + axon tracking on the same chip-well-DIV) provide a high-density anchor in unitlink's classifier candidate set, sitting between the within-session two-halves anchor (~zero drift) and cross-DIV pairs (~days of drift / biology). Network scan channel sets are also more stable across DIVs than axon-tracking routings, which helps unitlink's union channel grid.

**Network scan types** (per user's 2026-05-18 clarification): each DIV typically has TWO network scans — one with channels clustered into groups of 4 or 9 (similar to concat channel sets after slicing+concatenating axon-tracking segments), and one completely sparse with no clustering. Both overlap with the axon-tracking channel set across segments and merged-template outputs. **v2 picks ONE network-scan type for now (lean toward the clustered variant, since channel-grid overlap with axon-tracking is higher).** The other type stays as a future v3 lever — note in `memory/open_questions.md` once the v2 marginal-gain measurement lands.

- Data config: add `dataset_role` field per dataset. Existing axon-tracking entries get `dataset_role: axon_tracking` (default). Network scan entries get `dataset_role: network_scan`. Existing data configs continue to work (default == axon_tracking).
- Data path convention (per user 2026-05-18): `/pscratch/sd/a/adammwea/raw_data/Media_Density_T5_02182026_AR/Media_Density_T5_02182026_AR/<date>/<chip>/Network/` — same date/chip nesting as axon-tracking, with `Network` instead of `AxonTracking`. Network scans are single-segment (cannot be concatenated).
- Slice-2 update: `discover_chip_well_groups` includes both roles in the group when `include_network_scans: true`.
- Prerequisite: preprocess + spikesort + recon (including kssynth) must have run for the network-scan datasets too. Network scans are short, so the upstream pipeline cost is modest.
- Tests: synthetic group of 2 DIVs × 2 roles each; assert all 4 session paths flow through to `unitlink.match()`, UID chains tighten relative to the slice-5 baseline.

### Slice 7 (v2 validation, not code) — measure marginal gain
- Compare the slice-5 baseline (axon-tracking only) UID outputs against slice-6 (with network scans) UID outputs for the same chip-well group.
- Compare: # of UIDs in intermediate mode (lower = tighter matching), average UID-chain length (higher = more matches across DIVs per neuron), fraction of cross-DIV pairs that have a same-day intermediate support, # of singletons (units with no match anywhere).
- Record findings in `commit_log.md`. If marginal gain is large and network-scan upstream cost is bearable, recommend `include_network_scans: true` as the default.

Total estimated touch: S+. ~100-150 lines of new code in axon_recon (mostly slice 2 — group discovery + path resolution) + ~100 lines of tests. The previous draft of this plan estimated ~400-600 LoC because the work hadn't been extracted into kssynth/unitlink yet. Now those packages own the heavy lifting; this plan stays thin.

---

## 5. Tests

- `tests/test_unitmatch_groups.py`: chip-well group discovery against a synthetic data config (3 DIVs same chip+well; 1 DIV different chip; mixed roles in v2).
- `tests/test_unitmatch_path_resolution.py`: per-session input path resolution; asserts missing-kssynth-output raises with the actionable suggestion.
- `tests/test_unitmatch_phase_disabled_is_skipped.py`: phase no-ops when `enabled: false`.
- `tests/test_unitmatch_phase_calls_unitlink_per_group.py`: mocks `unitlink.match`; asserts one call per group, with the expected per-session inputs and out_folder.
- `tests/test_unitmatch_phase_skipped_for_single_div_group.py`: chip-wells with only 1 DIV get `status: skipped reason: insufficient_sessions`.
- `tests/test_unitmatch_phase_includes_network_scans_when_enabled.py` (v2): asserts both roles appear in the per-session inputs handed to unitlink.

Algorithmic correctness tests (two-halves split, channel-grid union, UMPy invocation, output schemas) live in unitlink's test suite, not here. The axon_recon tests check wiring, not algorithm behavior — that separation keeps both test suites focused.

---

## 6. Dependencies / blockers

- **kssynth (sibling pip package)** — produces the synth sorter_output folders this phase consumes. Must be installed in the shifter image. Pipeline must run `reconstruct.kssynth` before this phase. See `ks_synthesizer_package_plan.md`.
- **unitlink (sibling pip package)** — does the actual matching. Must be installed in the shifter image. See `unitmatch_runner_package_plan.md`.
- **Aux-tsv coherence on input sorter_outputs.** kssynth guarantees this for its synthetic outputs (`cluster_*.tsv` files are coherent by construction). If a session's sorter_output came from somewhere else (e.g. a non-kssynth path), unitlink will fail; clear error message points users at kssynth.
- **Recon stage completion.** Every DIV in a group must have completed reconstruct.kssynth + reconstruct.analyzers. The phase fails fast if any group member is missing inputs.
- **v2 prerequisite:** preprocess + spikesort + reconstruct must run on network scan datasets. This is upstream pipeline work that's not currently scoped; defer slice 6 until network scans are part of the normal pipeline run.

---

## 7. Open questions / follow-ups

The two-halves / channel-grid / UMPy-spine open questions migrated to the package plans. What remains here:

1. **Group key.** `(chip_id, well_id)` is the natural key for "same biological well across days". Alternative: include treatment/media so different conditions of the same chip-well get separate UID sets. Data config has `treatment` + `media` fields per well; consider whether group key should be `(chip_id, well_id, treatment, media)`. Lean toward `(chip_id, well_id)` for v1 (treatment changes between DIVs are rare in our protocol); revisit if the assumption is wrong.

2. **Network-scan inclusion as default.** Decide after slice 7's measurement. If marginal UID-chain quality gain is large and upstream cost is bearable, make it the default. Otherwise leave it as an opt-in YAML knob.

3. **Backend selection.** v1 hardcodes classical. When unitlink ships its v2 deep backend, this phase can expose `backend: classical | deep` in YAML — straightforward but blocked on unitlink slice 8 + an HD-MEA-trained DUM model.

4. **GUI curation handoff.** unitlink's `write_intermediate=True` mode persists the files UnitMatchPy's GUI needs. The phase could expose a YAML knob `write_intermediate: false` (off by default to save disk; flip per-run for hand-curation work). Defer until users ask.

5. **Per-chip threshold tuning.** unitlink's v3 slice adds `recommend_threshold()`. Once that ships, this phase can default to `match_threshold: null` (= use library recommendation) and surface the recommendation in summary.json. Wait until both v1 layers (axon-tracking and network-scan) are validated and we have empirical match-yield numbers.
