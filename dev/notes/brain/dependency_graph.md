# Dependency graph — phases, modules, plans, contracts

The persistent DAG. Lets the loop answer: "I touched X — what now needs re-verification?" Without this, contract drift goes silent and downstream code breaks invisibly.

**Reading rule for the loop**: when ANY slice ships that changes a node's contract (output shape, function signature, file layout), the loop MUST update the affected node's edges + push the dependent nodes onto a re-verify list.

**Writing rule**: loop UPDATES this file as a side-effect of shipping slices. Append edges + node entries; modify existing entries when a contract changes. Mark superseded edges struck-through but don't delete (history matters for audits).

---

## Status — Z1 mapped 2026-05-21

Initial population of the graph by the assistant in chat (manual Z1 mission). Captures the state as of commit `77d38fc` (post-reorg). Loop maintains forward from here: every shipped slice that changes a node's contract MUST update this file.

---

## 1. Stages + active phase_sequence (from `dev/debug_NERSC/debug.runtime.yml`)

### init
**phase_sequence**: `copy_src_to_scratch`
- `copy_src_to_scratch` — copies source recordings to scratch root for nodes that require local I/O. Resource: `h5_to_binary`.

### preprocess
**phase_sequence**: `save_rec_metadata` → `preprocess_segments`
**Disabled-in-sequence** (still defined): `plot_segment_traces`, `plot_segment_channel_layouts`, `plot_raster_threshold`
- `save_rec_metadata` — extracts per-recording metadata including **sampling_rate_hz** (J11 producer). Authoritative source per `feedback-axon-recon-device-diversity` auto-memory.
- `preprocess_segments` — per-segment SI recording filter + binarize. Produces per-segment preprocessed binaries (J12 below).

### spikesort
**phase_sequence**: `concat_binary` → `sort` → `snapshot_sorter_output` → `concat_analyzer` → `cleanup_concat_binary` → `cleanup_analyzers`
**Disabled-in-sequence** (still defined): `plot_concat_traces`, `plot_concat_channel_layout`, `summarize_sort`, `bombcell_label`, `merge_SLAy`, `bombcell_label_pass2`
- `concat_binary` — concatenates per-segment binaries into a single SI recording. Required by Kilosort.
- `sort` — runs Kilosort4 (`spikeinterface.sorters.run_sorter("kilosort4")`). Produces kilosort `sorter_output/` (J13 — see below).
- `snapshot_sorter_output` — freezes a copy of sorter_output as `sorter_output_snapshot/` (immutable post-bombcell/SLAy reference).
- `concat_analyzer` — builds a concat-level SI SortingAnalyzer (post-sort, pre-template-extraction).
- `bombcell_label` (DISABLED) — BombCell QC labeling. Re-enable: TBD per `chip_layout_phase_split_plan` / `kssynth_recon_integration_plan` slice 5 retirement decision.
- `merge_SLAy` (DISABLED) — SLAy automerge (consumes `slay.run`). Re-enable: same TBD.
- `bombcell_label_pass2` (DISABLED) — post-merge BombCell. KS-extractor inner-join bug; deferred to templates-aware redesign.
- `cleanup_concat_binary` / `cleanup_analyzers` — frees scratch after the heavy steps complete.

### reconstruct
**phase_sequence**: `analyzers` → `extract_partial_templates` → `build_templates` → `plot_templates_v2` → `report_templates` → `axon_velocity_gtrs` → `plot_recons` → `plot_branch_propagations` → `plot_branch_velocities` → `plot_unit_summary` → `report_recons` → `report_recon_grid` → `report_full_chip_layout` → `report_summaries` → `clear_templates_cache`
**Defined but DISABLED**: `kssynth` (replaces `extract_partial_templates` + `build_templates` per slice 5), `resolve_sources`, `compute_template_similarity`
- `analyzers` — builds per-segment SI analyzers from `preprocessed_segments` + `snapshot_sorter_output`. Produces analyzer cache (J3). **The `--input-root` extension shipped `c8b8b11` lets this run on a dev_outputs/ well while reading inputs from a reference path.**
- `extract_partial_templates` (LEGACY, slated for deletion by kssynth slice 5) — extracts per-unit waveforms from analyzers.
- `build_templates` (LEGACY, slated for deletion by kssynth slice 5) — assembles per-unit merged_template.npy + merged_channel_locations.npy (J1+J4 producer).
- `kssynth` (SHIPPED but DISABLED) — REPLACES extract_partial+build. Synthesizes KS-style folder via `kssynth.synthesize`; produces J5 (synth_sorter_output) AND J1+J4 via slice 4 per-unit postprocess (`_write_per_unit_templates_from_synth_output`).
- `plot_templates_v2` — per-unit waveform plots from J1.
- `report_templates` — per-unit waveform reports from J1.
- `axon_velocity_gtrs` — runs `axon_velocity` (Buccino) on per-unit templates. Produces J2 (gtr.pkl) + per-unit branches.json.
- `plot_recons`, `plot_branch_propagations`, `plot_branch_velocities`, `plot_unit_summary` — per-unit recon visualizations consuming J2.
- `report_recons`, `report_recon_grid`, `report_full_chip_layout`, `report_summaries` — aggregate report renderings.
- `clear_templates_cache` — disposal phase post-recon.
- `radivojevic_recon` (PLANNED — `radivojevic_recon_algo_plan` slice 4 onward) — alternative to `axon_velocity_gtrs`; consumes J1; will produce J2-equivalent (gtr-shape) per O3 DOD.

### analysis
**phase_sequence**: `compute_metrics` → `unitmatch` → `propagation_video`
- `compute_metrics` — reads JSON artifacts from recon stage; writes a per-well summary manifest.
- `unitmatch` (DISABLED, scaffolded slices 1-4) — invokes `unitlink.match` per (chip, well) group. Slice 5 enables + smokes. Consumes J5 (kssynth output).
- `propagation_video` — per-unit branch-propagation video / GIF (axon_velocity wrapping). Opt-in via YAML. Consumes J2.

### cleanup
**phase_sequence**: `wipe_src_scratch`
- `wipe_src_scratch` — DISABLED + dry_run=true by default for safety.

---

## 2. Sibling packages (consumed by phases above)

| Package | Consumer (phase or module) | Status |
|---|---|---|
| `axon_velocity` (Buccino 2022) | `reconstruct.axon_velocity_gtrs` | Upstream pin; PyPI |
| `spikeinterface` | preprocess, spikesort, reconstruct (analyzers) | Upstream pin; PyPI 0.104.3 |
| `SLAy` | `spikesort.merge_SLAy` (via `importlib.import_module("slay.run")` in `stages/spikesort/runner.py:2008,2018`) | GitHub: `adamwea/SLAy` (public); pinned by SHA in pyproject |
| `UnitMatchPy` | `unitlink` (via classical backend) | GitHub external; pinned via `--no-deps` install in Dockerfile |
| `kssynth` | `reconstruct.kssynth` (slice 5 enables) | GitHub: `adamwea/kssynth` (public); v1 feature-complete (58 tests green) |
| `unitlink` | `analysis.unitmatch` | GitHub: `adamwea/unitlink` (public); v1 feature-complete (53 tests green) |
| `radivojevic2023_recon_algo` | (planned) `reconstruct.radivojevic_recon` | LOCAL only (`~/dev/pkgs/radivojevic2023_recon_algo/`); Stages 1+2+3 shipped, 81 tests green |
| `bombcell` (matlab interop) | `spikesort.bombcell_label` | DISABLED phases; deferred per `chip_layout_phase_split_plan` |

---

## 3. Data contracts (the JUNCTIONS — highest-leverage)

Each junction is the on-disk shape (or in-memory shape) that many phases reach for. A change to a junction's contract propagates to every consumer — and these are where silent-downstream-break happens most often.

### J1 — Per-unit merged template (the biggest junction)
- **Location**: `<well>/recon_outputs/units/unit_<id>/merged_template.npy`
- **Shape**: `(n_active_channels, n_samples)` — sparsified (only channels with non-zero samples survive)
- **Produced by**: `build_templates` (LEGACY) OR `kssynth` (NEW via slice 4 postprocess `_write_per_unit_templates_from_synth_output`)
- **Consumed by**: `plot_templates_v2`, `report_templates`, `axon_velocity_gtrs`, `radivojevic_recon` (planned), `plot_recons` (indirectly via J2), `plot_branch_*` (via J2), report phases
- **Trusted output anchor**: `brain/trusted_outputs.md` TR-001 (176 templates on 260326/M08073/000208/well000 DIV 36)
- **Re-verify trigger**: any change to build_templates OR kssynth slice 4 postprocess; any change to the analyzer cache contract (J3); any change to spikesort sort/merge output (J6)

### J2 — Gtr-shaped reconstruction result
- **Location**: `<well>/recon_outputs/units/unit_<id>/gtr.pkl` + `branches.json`
- **Shape**: `axon_velocity`'s pickled GraphTracking result (axon trajectory, branch tree, velocity per segment, soma channel, selected channels)
- **Produced by**: `axon_velocity_gtrs`
- **Consumed by**: `plot_recons`, `plot_branch_propagations`, `plot_branch_velocities`, `plot_unit_summary`, `report_recons`, `report_recon_grid`, `report_summaries`, `analysis.propagation_video`, dashboard (read-only)
- **PLANNED**: `radivojevic_recon` (per O3 DOD) must produce a J2-equivalent shape so plot_recons can render BOTH via the SAME code path. This is the apples-to-apples requirement.
- **Re-verify trigger**: axon_velocity upstream change; axon_velocity_gtrs phase config change; any plot_recons-side input-shape change

### J3 — Analyzer cache
- **Location**: `<well>/recon_outputs/cache/analyzers/segments/{000_rec0000, 001_rec0001, ...}/`
- **Shape**: per-segment SpikeInterface `SortingAnalyzer` saved dir
- **Produced by**: `reconstruct.analyzers`
- **Consumed by**: `extract_partial_templates`, `build_templates`, `kssynth`, `compute_template_similarity`, anything that needs per-segment analyzers
- **Re-verify trigger**: preprocess_segments output shape change (J12); spikesort sort output change (J13); analyzers phase config change; spikeinterface upstream

### J4 — Per-unit merged channel locations
- **Location**: `<well>/recon_outputs/units/unit_<id>/merged_channel_locations.npy`
- **Shape**: `(n_active_channels, 2)` — xy coords matching J1's channel axis
- **Produced by**: same as J1
- **Consumed by**: same as J1 (paired)

### J5 — kssynth synth_sorter_output
- **Location**: `<well>/recon_outputs/synth_sorter_output/`
- **Contents**: `spike_times.npy`, `spike_clusters.npy`, `cluster_KSLabel.tsv`, `cluster_group.tsv`, `channel_map.npy`, `channel_positions.npy`, `params.py`, `templates.npy`, `kssynth_summary.json`, plus `per_unit/unit_<id>/{merged_template.npy, merged_channel_locations.npy}` from slice 4 postprocess
- **Produced by**: `reconstruct.kssynth`
- **Consumed by**: downstream of J1 (via the per_unit/ subset); `analysis.unitmatch` (via the spike-sort-shape portion)
- **Re-verify trigger**: kssynth phase config / postprocess change; kssynth sibling pkg version change; analyzer cache contract change (J3)

### J6 — Spikesort canonical outputs
- **Location**: `<well>/spikesort_outputs/sorter_output/` and `sorter_output_snapshot/`
- **Contents**: `spike_times.npy`, `spike_clusters.npy`, `cluster_KSLabel.tsv` (good/mua/noise), `cluster_group.tsv` (post-SLAy if enabled), `templates.npy`, `channel_positions.npy`
- **Produced by**: `spikesort.sort` (raw kilosort); `spikesort.merge_SLAy` (post-merge clusters, when enabled)
- **Consumed by**: `reconstruct.analyzers` (consumes the snapshot), kssynth (channel positions + cluster TSV sync)
- **Trusted output anchor**: TR-001 partial (377 post-SLAy unit IDs; 287 good + 312 mua = 599 rows in cluster_KSLabel.tsv)

### J7 — Phase config dataclass shapes (Python module contract)
- **Location**: `src/axon_recon/pipeline/stages/<stage>/config.py` + per-phase `<phase>_config` dataclasses
- **Consumed by**: per-phase parser functions in `pipeline/config.py`; runner call sites; YAML parser
- **Re-verify trigger**: any add/remove/rename of a dataclass field; tests in `pipeline/tests/` typically catch via fixture failures

### J8 — Resource budget contract
- **Producer**: `pipeline/resource_budget.py::current_phase_budget` → returns `_budget` with `cpus_per_task`, `nested_shape`, etc.
- **Currently sourced from**: `pipeline/resources.py` YAML profile system (`active_profile` + `task_allocation.cpus_per_task`)
- **WILL CHANGE to**: env-only via `resources_profiles_elimination_plan` slices 1-3 (new `pipeline/env_supply.py::EnvSupplyBudget`)
- **Consumed by**: every phase that calls `resolve_inner_worker_count` (parallelism cleanup slice 2 routes more call sites here)
- **Re-verify trigger**: any change to `_derive_cpus_per_task`, `build_task_allocation_plan`, or the env-supply resolver. Currently the user-flagged "incorrect clamping" bug (metric M-003).

### J9 — CLI dispatch tree
- **Location**: `pipeline/cli.py` + `pipeline/stages/<stage>/cli.py` per-stage subparsers
- **Consumed by**: user-facing `axon-recon stages <stage>.<phase>` invocations; sbatch scripts; smoke tests
- **Re-verify trigger**: phase add/remove/rename; flag add/remove; `--force-enable`-like overrides

### J10 — Runtime YAML schema
- **Location**: `dev/debug_NERSC/debug.runtime.yml` + `dev/debug_NERSC/debug.data.yml` (+ `src/axon_recon/default.runtime.yml`)
- **Consumed by**: every parser in `pipeline/config.py`; every `phase_sequence` consumer
- **Hygiene rule**: CLAUDE.md slice protocol step 7 (any code change touching CLI/phase/config MUST update both YAMLs)
- **Re-verify trigger**: same as J7

### J11 — Sample rate (per-recording, device-dependent)
- **Producer**: `preprocess.save_rec_metadata` (extracts from raw H5 header)
- **Consumed by**: every analysis phase needing time-alignment (radivojevic `upsample_factor`, axon_velocity_gtrs velocity computation, etc.)
- **Authoritative source per**: `feedback-axon-recon-device-diversity` auto-memory + `brain/refs/radivojevic2023_algorithm_summary.md` "Hardware assumptions"
- **Re-verify trigger**: any new device support (ThreeBrain, Sony — see `trackers/tech_debt.md` §"Device-agnostic source data")

### J12 — Preprocessed segment binaries
- **Location**: `<well>/preprocess_outputs/segments/<seg-id>/{recording.bin, channel_*.npy}`
- **Produced by**: `preprocess.preprocess_segments`
- **Consumed by**: `spikesort.concat_binary`, `reconstruct.analyzers`
- **Re-verify trigger**: spikeinterface upstream change; preprocess filter parameter change

### J13 — Kilosort raw sorter_output
- **Location**: `<well>/spikesort_outputs/sorter_output/` (pre-snapshot)
- **Produced by**: `spikesort.sort` (via spikeinterface `run_sorter("kilosort4")`)
- **Consumed by**: `spikesort.snapshot_sorter_output`, then downstream of J6

---

## 4. Plan → junction touch matrix

Which plans touch which contracts. The biggest red flag is two plans touching the same junction — that's where coherence drift (the kssynth/parallelism/resources_profiles tangle of 2026-05-21) comes from.

| Plan | Junctions touched | Notes |
|---|---|---|
| `kssynth_recon_integration_plan` | J1 (becomes co-producer), J3 (consumer), J5 (NEW producer), J8 (resource budget consumer) | Slice 5 retires `extract_partial_templates` + `build_templates` as J1 producers; kssynth takes over |
| `ks_synthesizer_package_plan` | (sibling pkg only — produces `kssynth` package consumed by J5 producer) | v1 feature-complete; SHIPPED |
| `unitmatch_phase_plan` | J5 (consumer), produces `analysis.unitmatch/<chip>/<well>/match_table` | Slice 5 enables + smokes; gated on kssynth slice 5 + unitlink v1 (both done) |
| `unitmatch_runner_package_plan` | (sibling pkg only — produces `unitlink` package consumed by `analysis.unitmatch`) | v1 feature-complete; SHIPPED |
| `radivojevic_recon_algo_plan` | J1 (consumer), will produce J2-equivalent | Sibling pkg Stages 1-3 SHIPPED; phase implementation + plot_recons adapter NOT YET shipped |
| `parallelism_post_migration_cleanup_plan` | J8 (consumer — wires `inputs.n_jobs` call sites through `resolve_inner_worker_count`) | Slices 3-9 + 9.5 SHIPPED; slice 1+2+10 queued. **Slice 2 MUST ship AFTER `resources_profiles_elimination` slice 2.** |
| `resources_profiles_elimination_plan` | J8 (producer — replaces YAML-profile source with env-only resolver), J9 (removes `--profile` flag), J10 (deletes YAML `profiles:` block) | 5 slices; slice 0 reverts `1982a31` + `322e8fe` paper-overs |
| `phase_roster_cleanup_plan` | J7 (deletes per-phase config classes), J9 (deletes phase CLI dispatchers), J10 (removes phase blocks from YAML) | Slices 1-9 + 11-13 + 14a-14b SHIPPED; slice 14c blocked on per-target stage runner refactor decision |
| `env_install_unification_plan` | (install-side only — pyproject extras + Dockerfile; no runtime contracts) | v1 COMPLETE (slices 1-8 shipped) |
| `analysis_propagation_video_plan` | J1 (consumer), J2 (consumer); produces `analysis/propagation_video/<unit-id>/*.mp4` | Tier 5 chip-away; NOT BLOCKING anything |
| `chip_layout_phase_split_plan` | J2 (consumer for cross-session colors via unitmatch match table); moves `report_full_chip_layout` from recon to analysis | Tier 4 gated on kssynth slice 5 + unitmatch_phase slice 5 |
| `dashboard_ui_refinement_plan` | (consumes existing `analyzed_data/` outputs read-only; no runtime contracts touched) | PLAN COMPLETE (all 9 slices SHIPPED) |
| `dry_run_rollout_plan` | Transparent short-circuit across every phase (additive; doesn't change output contract when dry-run is off) | Slices 1-3 + 5 + 6 SHIPPED + most of phase coverage |

**Junction coherence findings** (the multi-plan-touching-same-surface flags):

- **J1 producer migration**: `build_templates` (LEGACY) → `kssynth` (NEW). Single owner pre-slice-5; dual-producer mid-flight; single owner (kssynth) post-slice-5. Risk if slice 5 ships incompletely (kssynth-as-producer + extract_partial+build retained = 3 producers, ambiguity).
- **J2 producer expansion**: `axon_velocity_gtrs` (current sole producer) → ALSO `radivojevic_recon` (planned). Both must produce same shape. This is the apples-to-apples comparison anchor.
- **J8 source change**: YAML-profile (current) → env-only (post-resources_profiles_elim). `parallelism_post_migration_cleanup_plan` slice 2 wires call sites that read J8 — sequencing matters; see existing See-also markers in both plans (commits `488628f`).
- **J7+J9+J10 simultaneous change**: `phase_roster_cleanup_plan` retires phases; `kssynth_recon_integration_plan` slice 5 deletes 2 phases. Both must coordinate to avoid leaving stale YAML blocks or stale CLI dispatchers.

---

## 5. Ranked verification checkpoints (T2 input for user triage)

Per the phase-zero design: identify the MINIMAL set of trusted outputs that maximally constrains the rest. Loop proposes; user triages into pin / cheap-acquire / expensive buckets.

### Already pinned (Tier 1)
- **TR-001** (already in `brain/trusted_outputs.md`) — 176 merged templates on `260326/M08073/000208/well000` DIV 36. Covers J1 + J3 + J6 + parts of J8 transitively. THE single highest-leverage anchor.

### Proposed for promotion (loop appends; user triages)

**Rank 1 — covers J2 (recon visualization anchor)**
- **Candidate**: `axon_velocity_gtrs` output (`gtr.pkl` + `branches.json` + `plot_recons` rendering) for unit_0598 (9-branch reference) on `260326/M08073/000208/well000` DIV 36
- **Why**: J2 is the second-biggest junction. Every recon-stage visualization phase consumes it. Pinning a single high-branch unit's gtr output as trusted anchors the entire recon-stage visualization chain. Also: this is the comparator side of the Radivojevic DC-001 differential check (`brain/trusted_outputs.md`).
- **Coverage** (estimated): all 11 recon-stage visualization/report phases that consume J2.
- **Acquisition cost**: cheap — file exists at the reference path; user eyeballs the plot_recons rendering for unit_0598 + approves.
- **Trust handle**: "yes this is what an axon arbor reconstruction should look like."

**Rank 2 — covers J5 (kssynth output regression)**
- **Candidate**: `kssynth` `synth_sorter_output/` for `260326/M08073/000208/well000` DIV 36 after slice 5 lands
- **Why**: J5 is the new J1-producer path. Critical to know kssynth's J1+J4 per-unit dirs MATCH `build_templates`' contract.
- **Coverage**: kssynth phase + slice 5 retirement of extract_partial+build_templates.
- **Acquisition cost**: HEAVY — requires running kssynth slice 3b heavy on a real allocation (already queued in `trackers/salloc_smokes_queued.md`).
- **Trust handle**: per-unit dir count matches TR-001's 176; each unit dir has merged_template.npy + merged_channel_locations.npy; shapes match build_templates' output for the same unit (loop can compute a sha256 comparison automatically once both exist).

**Rank 3 — covers J11 (sample rate / device metadata)**
- **Candidate**: `save_rec_metadata` output for any one M08073 recording (e.g. dataset 13's metadata.json)
- **Why**: every analysis phase that time-aligns consumes J11. Cheap to pin since metadata is small + already-extracted.
- **Coverage**: implicit for everything downstream that uses sample rate. Plus protects against silent device misidentification.
- **Acquisition cost**: trivial — file exists; user confirms `sampling_rate_hz: 10000` for MaxTwo.

**Rank 4 — covers analysis.unitmatch output**
- **Candidate**: `unitmatch/<chip>/<well>/match_table.parquet` for a (M08073, well000) group of ≥3 DIVs
- **Why**: O2 DOD requires this; cross-session unit matching is critical for chip_layout_phase_split + future propagation_video work.
- **Coverage**: analysis.unitmatch + downstream consumers (chip_layout split slice 3 cross-session colors).
- **Acquisition cost**: HEAVY — requires kssynth slice 5 + unitmatch_phase slice 5 both shipped + real-data smoke.
- **Trust handle**: row counts in sensible bands; UID chains internally consistent; spot-check a few matches by hand against unit signatures.

**Rank 5 — covers Radivojevic comparison (DC-001)**
- **Candidate**: `radivojevic2023_recon_algo.reconstruct()` output for unit_0598's merged_template (same input as Rank-1 candidate)
- **Why**: pairs with Rank 1 to anchor the DC-001 differential checks.
- **Coverage**: radivojevic_recon phase implementation + plot_recons adapter (NOT yet shipped).
- **Acquisition cost**: gated on Rank 1 + Rank 2 (needs J1 input that matches what axon_velocity_gtrs consumed).
- **Trust handle**: NOT a standalone trusted output — it's a differential-similarity check against Rank 1.

**Rank 6 — covers preprocess output sanity (J12)**
- **Candidate**: `preprocess_outputs/segments/000_rec0000/recording.bin` checksum + row count for one M08073 recording
- **Why**: catches silent preprocess-stage breakage that would invalidate everything downstream.
- **Coverage**: preprocess_segments + everything downstream of J12.
- **Acquisition cost**: cheap — already exists at reference path; sha256 + file-size check.
- **Trust handle**: file-level integrity check, not semantic.

### Coverage estimate

If user pins Rank 1 + Rank 3 + Rank 6 (the cheap-acquire bucket): TR-001 + Ranks 1/3/6 transitively cover J1 / J2 / J3 / J4 / J6 / J11 / J12 / J13 + most of the visualization chain. That's a substantial chunk of the recon-side pipeline before any heavy work runs.

Adding Rank 2 (kssynth heavy) covers J5 + closes the kssynth integration loop.

Adding Rank 4 + Rank 5 closes O2 + O3 respectively, both contingent on heavier work.

---

## 6. Schema (for future entries)

When a slice changes a node's contract or adds a new node, append an entry like:

```
### <node-id> — <human label>
- **Kind**: phase | module | plan | sibling-package | YAML-key | CLI-flag | guardrail | junction
- **Owns**: <list of file paths that implement this node>
- **Produces (contract)**: <interface signature OR output shape OR file layout>
- **Consumes (deps)**: [<list of node-ids this depends on>]
- **Depended-on-by**: [<list of node-ids whose contract assumes this one>]
- **Trusted outputs**: [<list of brain/trusted_outputs.md TR-xxx IDs that anchor this node>]
- **Last contract change**: <commit hash> on <date>
- **Re-verify when**: <one sentence on what kind of change triggers downstream re-verification>
```

Z1 mapped 2026-05-21 as prose (above) rather than this schema; the schema is for FORWARD entries as the loop ships new contract-changing slices.
