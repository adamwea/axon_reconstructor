# kssynth — synthetic sorter-output builder from segment analyzers

Status: scoped (idea → in-plan). Same operating contract: one slice at a time, `claude:` commit prefix.

This plan extracts the "build a coherent, fused-template sorter-output folder from a set of registered SpikeInterface analyzers" capability into its own Python package. The package lives as a sibling to `axon_recon` under `~/dev/pkgs/kssynth/` (separate git repo, separate pyproject.toml, separate pip install). axon_recon depends on it via the same build-time sibling-staging pattern we already use for SLAy / UnitMatchPy.

**See also (companion plans):**
- `unitmatch_runner_package_plan.md` — unitlink, the sibling pip package that consumes kssynth's output (and per-session SpikeInterface analyzers) to run UnitMatch / DeepUnitMatch across N sessions. UnitMatch-specific shape work (two-halves split for the `(spike_width, n_channels, 2)` axis) lives there, deliberately kept OUT of kssynth.
- `unitmatch_phase_plan.md` — the thin axon_recon analysis-stage phase that calls kssynth (in recon stage) + unitlink (in analysis stage) in sequence.

---

## 0. End-state, in one paragraph

A small, thin Python package — working name `kssynth` — that takes a **list of registered SpikeInterface SortingAnalyzers** (the natural output of the recon stage's `analyzers` phase, or anyone's per-segment / per-block analyzers) and produces ONE synthetic `sorter_output/` folder whose contents follow SpikeInterface conventions and are consumable by any KS-folder-aware downstream tool: UnitMatchPy, Bombcell, SLAy, SI's `read_sorter_folder` / `read_kilosort`. The package internally handles partial-template extraction (per analyzer, per unit) and merged-template building (cross-analyzer fusion) — the two recon-stage phases that currently own this become a single `kssynth` library call. The package is intentionally NOT axon-recon-specific: any SpikeInterface user with multi-segment analyzers benefits.

---

## 1. Why a separate package

| Reason | Detail |
|---|---|
| **Other tools want it** | Bombcell, UnitMatch, SLAy, and any KS-folder consumer all need a coherent KS-shape directory with cross-segment-fused templates. Solve it once, generically. |
| **General SpikeInterface utility** | The "fuse per-segment analyzers into one synthetic sorter output" operation is independent of axon_recon. It belongs in the broader SI ecosystem. |
| **Clean boundaries** | axon_recon's recon stage shrinks to "build analyzers" → "call kssynth" → "downstream phases consume synth folder". Three sharp stages instead of a tangle. |
| **Independent versioning / upstreaming** | Same model as SLAy / UnitMatchPy — separate git repo, separate pip install, can be PR'd to / used by labs outside this codebase. |
| **Test surface** | Library-level tests against synthetic analyzers are simpler than pipeline-integrated tests. |

What this is NOT solving:
- Not a spike sorter.
- Not a label / curation tool — produces inputs FOR those tools.
- Not specific to HD-MEA or axon tracking.

---

## 2. Package scope

### In scope (v1)

The package internalizes everything between "I have registered analyzers" and "I have a sorter_output folder ready for downstream consumption":

1. **Channel grid resolution.** Given N analyzers, compute the union (default) or intersection of their channel sets via xy matching, producing one global channel grid for the synthetic output.
2. **Partial template extraction (per analyzer, per unit).** For each analyzer in the input list, pull each unit's mean waveform (from the analyzer's `templates` or `waveforms` extension) and rasterize it onto the global channel grid. This is what axon_recon's `extract_partial_templates` phase currently does.
3. **Merged template building (cross-analyzer fusion).** For each unit, fuse its per-analyzer partial templates into one global template. Default policy: spike-count-weighted mean. Optional: median, time-aligned average (drift-correcting). This is what axon_recon's `build_templates` phase currently does.
4. **Synthetic sorter_output folder writer.** Emit a directory that comports with SpikeInterface conventions for a kilosort-shape sorter folder:
   - `spike_times.npy`, `spike_clusters.npy` — from the input sorting (taken from the analyzers — they all share one sorting object)
   - `templates.npy` — cross-analyzer-fused mean templates (one row per unit, dense on the global channel grid)
   - `mean_waveforms.npy` — alias of `templates.npy` for tools that read it
   - `cluster_KSLabel.tsv`, `cluster_group.tsv`, `cluster_Amplitude.tsv`, `cluster_ContamPct.tsv` — coherent across all cluster_ids in `spike_clusters.npy` (every TSV has every id, stub-filled where data is missing)
   - `channel_positions.npy`, `channel_map.npy` — from the global channel grid
   - `params.py` — sampling_frequency, n_channels_dat, dtype, etc., per kilosort conventions
   - `spikeinterface_log.json` — the marker SI's `read_sorter_folder` looks for (so the folder is loadable without falling back to `read_kilosort`)
   - `automerge/` (optional, only if caller provided a merge map) — for SLAy compatibility
5. **Coherence invariant.** Every `cluster_id` present in `spike_clusters.npy` has a row in EVERY `cluster_*.tsv` file. SI's `read_kilosort` inner-join won't drop units. This is the structural fix that motivated the SLAy patch this week; here it's a library-level invariant of the output.

The package does NOT produce `RawWaveforms/Unit{N}_RawSpikes.npy` with a two-halves axis, and does NOT do per-unit temporal-midpoint waveform splits. Those are UnitMatch-specific shape conversions that don't belong in a generic sorter_output. The UnitMatch consumer adapter (in axon_recon's analysis stage, see `unitmatch_phase_plan.md`) layers that on top of kssynth's output without needing kssynth to know about it.

### Out of scope (v1)

- Spike sorting.
- Label assignment.
- Drift correction (downstream tools — UnitMatchPy's `extract_metric_scores` — handle drift natively; we just expose pre-aligned per-segment templates as an opt-in fusion policy).
- Plot / report generation.
- Raw-waveform re-extraction from binary recordings — we consume cached analyzer data. If the analyzer didn't compute it, this package doesn't compute it from scratch.

### Future scope (v2+, deferred)

- Multi-session synthesis (combine analyzers across multiple recordings of the same well into one synthetic folder — useful for UnitMatch-on-merged-sessions).
- Aggregation across analyzers from different SpikeInterface sortings (currently we assume all input analyzers share one sorting).
- DeepUnitMatch-compatible feature extraction.

---

## 3. Inputs / Outputs

### Inputs

The public API takes a **list of registered SortingAnalyzer objects**:

```python
analyzers: list[SortingAnalyzer]
```

Each analyzer must:
- Be a SpikeInterface `SortingAnalyzer` (or compatible — duck-typed on `.sorting`, `.recording`, `.get_extension`).
- Share its `.sorting` object with the other analyzers in the list (same unit_ids, same spike assignments — the spec asserts this; raises if not).
- Have a `templates` or `waveforms` extension computed (so per-unit per-analyzer template data is available without re-extraction).
- Have a `.recording` with `.get_channel_locations()` and `.get_sampling_frequency()`.

A single analyzer is a degenerate-but-valid input (just one segment / no fusion across analyzers; templates come straight from that analyzer).

Optional inputs:
- `merges` — a dict `{new_unit_id: [old_unit_id, …]}` describing post-merge unit lineage, so the synthetic folder's `automerge/` directory + auxiliary stub rows are filled in for SLAy/UnitMatch consumers that want it. When the caller has already run a merger (e.g. SLAy), they pass this; otherwise it's `None`.
- `dtype`, `sampling_frequency` overrides — pulled from the analyzers by default.
- `out_folder` — where to write.

### Output

A single directory at `<out_folder>/` shaped per SpikeInterface conventions for a kilosort-style sorter_output folder. Loadable via:

- `spikeinterface.read_sorter_folder(<out_folder>)` — the canonical path (needs `spikeinterface_log.json`, which we write).
- `spikeinterface.extractors.read_kilosort(<out_folder>)` — fallback for tools that bypass the SI wrapper.
- `UnitMatchPy.utils.paths_from_KS([<out_folder>, …])` — works once the consumer (UnitMatch adapter) layers a `RawWaveforms/Unit{N}_RawSpikes.npy` dir on top of kssynth's output. kssynth provides everything except the UnitMatch-specific two-halves waveforms.
- Bombcell's KS-folder reader — schema matches.

---

## 4. Module layout

Modular and thin: each step in §2 has its own module, usable independently. The `api.synthesize()` orchestrator is the convenience entry point but isn't required.

```
~/dev/pkgs/kssynth/                        (sibling to axon_recon, SLAy, UnitMatch)
├── README.md
├── LICENSE                                (MIT, matching SLAy)
├── pyproject.toml
├── src/kssynth/
│   ├── __init__.py
│   ├── api.py                             (synthesize() orchestrator)
│   ├── policies.py                        (config dataclasses: AggregationPolicy, ChannelGridPolicy, …)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── channel_grid.py                (union / intersection across analyzer channel sets)
│   │   ├── partial_templates.py           (per-analyzer per-unit partial template extraction)
│   │   ├── merge_templates.py             (cross-analyzer template fusion policies)
│   │   ├── rasterize.py                   (sparse-to-dense onto a channel grid)
│   │   └── cluster_tsv_sync.py            (aux-tsv coherence; generalized from the SLAy patch)
│   ├── io/
│   │   ├── __init__.py
│   │   ├── analyzer_reader.py             (read from SortingAnalyzer)
│   │   └── ks_folder_writer.py            (writes the synthetic sorter_output folder + SI metadata)
│   └── cli.py                             (kssynth synthesize <args>)
└── tests/
    ├── test_channel_grid.py
    ├── test_partial_templates.py
    ├── test_merge_templates.py
    ├── test_rasterize.py
    ├── test_cluster_tsv_sync.py
    ├── test_ks_folder_writer.py
    └── test_api_end_to_end.py
```

Every `core/` module exposes a small, pure function or two. `api.synthesize()` just composes them. Other consumers (e.g. someone who only wants the `cluster_tsv_sync` function in their own pipeline) can import that one module and use it standalone.

---

## 5. Public API

### Library

```python
import kssynth

# Most common case: list of segment analyzers all built against the same sorting.
out = kssynth.synthesize(
    analyzers=segment_analyzers,      # list[SortingAnalyzer]
    out_folder="/path/to/synth_sorter_output",
    channel_grid="union",             # or "intersection"
    aggregation="spike_count_weighted_mean",   # or "mean", "median", "time_aligned_average"
    merges=None,                      # optional: {new_id: [old_ids, …]}
)
# returns a small results dataclass: out.folder, out.unit_ids, out.channel_grid, out.summary
```

### Direct module use

```python
from kssynth.core.channel_grid import compute_channel_grid
from kssynth.core.partial_templates import extract_partial_templates_for_analyzer
from kssynth.core.merge_templates import merge_partial_templates
from kssynth.core.cluster_tsv_sync import ensure_cluster_tsv_coherence
from kssynth.io.ks_folder_writer import write_synthetic_sorter_output

# Anyone using SpikeInterface can pick the one piece they want without
# pulling in the rest. cluster_tsv_sync in particular is a single-function
# import that other pipelines (and SLAy itself, see slice 9) can call to
# fix their KS folder's TSV coherence without buying into the whole package.
```

### CLI

```bash
kssynth synthesize \
    --analyzers /path/to/analyzer_dir_or_glob \
    --out /path/to/synth_sorter_output \
    [--aggregation spike_count_weighted_mean] \
    [--channel-grid union]

kssynth ensure-coherence /path/to/sorter_output
```

---

## 6. SpikeInterface convention compliance

Specific things we do to make the output "feel like" a normal SI/kilosort folder:

1. **`spikeinterface_log.json`** — written at the root. This is the marker SI's `read_sorter_folder` checks for; without it consumers fall back to `read_kilosort`. We write a minimal valid log with the sorter name (e.g. `"kssynth"`), version, sampling frequency, and a note that the folder is a synthetic-template output.
2. **`params.py`** — kilosort-conventional Python file with `sample_rate`, `n_channels_dat`, `dtype`, `offset`, `hp_filtered` fields. Some tools (Bombcell, GUI tools) read it.
3. **`templates.npy` shape** — `(n_units, spike_width, n_channels)`, matching what SI's `Templates` object expects when loaded.
4. **`spike_clusters.npy` integer dtype** — int64, matching SI's reader.
5. **`channel_positions.npy` shape** — `(n_channels, 2)` xy coordinates. (`channel_map.npy` is also written for kilosort compatibility, as `np.arange(n_channels)`.)
6. **TSV files with header + integer cluster_id column** — standard kilosort layout.
7. **Coherence**: every cluster_id present in `spike_clusters.npy` has a row in EVERY `cluster_*.tsv` (this is the structural invariant from §2 #6).
8. **`automerge/` directory** (when `merges` is passed): contains `new2old.json` and `old2new.json` mirroring SLAy's format, so SLAy can re-read the folder and recognize its own merge history.

Loadability test (part of slice 8): run `si.read_sorter_folder(<out_folder>)` and assert `unit_ids` match, recording info matches, channel locations match.

---

## 7. Implementation slices

One commit per slice. `claude:` prefix. Logged in the new repo's `commit_log.md` (which will mirror the axon_recon convention).

**Status (2026-05-21)**: Slices 1-7 SHIPPED. The kssynth package at
`~/dev/pkgs/kssynth/` has all v1 modules + tests:
- `src/kssynth/core/` — channel_grid.py, rasterize.py, partial_templates.py,
  merge_templates.py, cluster_tsv_sync.py
- `src/kssynth/io/` — ks_folder_writer.py
- `src/kssynth/api.py` — orchestrator composing all of the above
- `src/kssynth/cli.py` — `kssynth` CLI entry
- Tests for each module + end-to-end test (`tests/test_api_end_to_end.py`).
- `pyproject.toml` + `pip install -e .` works.

Slice 8 (SLAy soft-import) and slice 9 (axon_recon integration) are
tracked separately:
- Slice 9 is in `kssynth_recon_integration_plan.md` (slices 1-4e SHIPPED;
  slice 5 destructive enable gated on data-routing decision in
  current_state.md).
- Slice 8 is upstream — a SLAy-side commit to soft-import kssynth's
  cluster_tsv_sync. Not yet picked up.

### Slice 1 — scaffolding (no behavior)
**Status**: SHIPPED — repo created at ~/dev/pkgs/kssynth/ with all
artifacts.
- New git repo at `~/dev/pkgs/kssynth/` with `pyproject.toml`, `README.md`, `LICENSE` (MIT), `commit_log.md`, basic CI skeleton (GitHub Actions matrix).
- Empty `src/kssynth/` package with `__init__.py` and a stub `api.synthesize()` raising `NotImplementedError`.
- `pip install -e .` works.
- Single test asserts the stub raises. Single CI run green.

### Slice 2 — `cluster_tsv_sync.py` (generalize the SLAy patch)
- Lift `_sync_auxiliary_cluster_tsvs` from `SLAy/src/slay/stages.py` into `core/cluster_tsv_sync.py`. Make it accept a generic sorter_output path and OPTIONAL `merges` dict. When no merges dict is provided, infer cluster_id divergence by comparing the cluster_id sets across `cluster_group.tsv`, `cluster_KSLabel.tsv`, `cluster_Amplitude.tsv`, `cluster_ContamPct.tsv` and fill stubs for any missing IDs.
- Default fill values: KSLabel from cluster_group's `label` column if present else empty string; Amplitude / ContamPct from a configurable fallback (NaN, 0, or first-parent via `merges` map).
- Tests: synthetic sorter_output with deliberately-divergent cluster_*.tsv files; asserts coherent output. Tests against the actual SLAy-mutated folder we have on disk as a regression case.

### Slice 3 — `channel_grid.py`
- Function: `compute_channel_grid(analyzers, mode="union", tolerance_um=1.0) -> ChannelGrid` where `ChannelGrid` carries xy positions, an ordering, and per-analyzer index mapping (the mapping is the heart of slice 4's rasterizer).
- Tests: 3 analyzers with partially-overlapping channel sets; asserts union size, per-analyzer index lookup, tolerance-based xy matching.

### Slice 4 — `rasterize.py`
- Function: `rasterize_to_grid(template, source_channel_indices, grid, fill_value=0.0) -> np.ndarray`.
- Tests: sparse template (shape `(n_active, spike_width)`); assert output shape, zero-fill, channel order matches grid.

### Slice 5 — `partial_templates.py` (per-analyzer per-unit extraction)
- Function: `extract_partial_templates_for_analyzer(analyzer, grid) -> dict[unit_id, np.ndarray]` returning per-unit dense templates on the global channel grid. Reads the analyzer's `templates` extension; falls back to per-unit waveform mean if only `waveforms` is available.
- Tests: synthetic analyzer with known per-unit waveforms; assert extracted templates match.

### Slice 6 — `merge_templates.py` (cross-analyzer fusion)
- Function: `merge_partial_templates(per_analyzer_partials, spike_counts, policy="spike_count_weighted_mean") -> dict[unit_id, np.ndarray]`. Supported policies: `mean`, `spike_count_weighted_mean` (default), `median`, `time_aligned_average` (uses cross-correlation to align per-analyzer templates before averaging — useful for cross-segment drift).
- Tests: synthetic 2-analyzer 2-unit case with known per-segment spike counts; assert weighted-mean fusion matches closed-form expected result.

### Slice 7 — `ks_folder_writer.py` + `api.synthesize()` orchestration + CLI
- `write_synthetic_sorter_output(out_path, sorting, merged_templates, channel_grid, params=..., merges=None)`. Writes all files per §6. Always calls `ensure_cluster_tsv_coherence` on the output as the last step.
- `api.synthesize(analyzers, out_folder, …)` orchestrates: channel_grid → per-analyzer partials → merge_templates → writer.
- `kssynth` CLI entry point in `pyproject.toml`.
- End-to-end test: 2 synthetic analyzers, 5 units each, call `kssynth.synthesize(...)`, verify:
  - Output is loadable by `si.read_sorter_folder()` (this is the key compliance check).
  - Unit ids match expected.
  - `cluster_*.tsv` files all have the same cluster_id set.
- This is the v1-feature-complete commit. Tag `v0.1.0`.

### Slice 8 — Upstream SLAy soft-imports kssynth (optional, opt-in)
- SLAy-side change: in `accept_all_merges`, try to `import kssynth.core.cluster_tsv_sync` and call its function. Fall back to SLAy's inline implementation if the import fails (keeps SLAy zero-deps by default).
- A separate SLAy commit, not a kssynth commit.

### Slice 9 — axon_recon integration (separate plan, in axon_recon repo)
Tracked here briefly; full implementation lives in axon_recon's recon-stage plan:
- Replace recon-stage `extract_partial_templates` and `build_templates` phases with a single `kssynth` phase that calls `kssynth.synthesize(segment_analyzers, …)`.
- The recon-stage `analyzers` phase still builds the segment analyzers; nothing changes there. It publishes per-segment SortingAnalyzers under `cache/analyzers/segments/`.
- The new `kssynth` phase reads those, calls the library, writes the synthetic `sorter_output/` folder under `recon_outputs/synth_sorter_output/`.
- Downstream recon phases (`plot_templates_v2`, `report_templates`, etc.) get repointed at the new folder.
- **Downstream of THAT**: the analysis stage's `unitmatch` phase (see `unitmatch_phase_plan.md`) consumes the per-DIV `synth_sorter_output/` directories as inputs to `unitlink.match()`. So kssynth's output is the contract bridge between recon and analysis.
- This is also the unlock for the bombcell/SLAy migration: bombcell_label and merge_SLAy phases can move OUT of the spikesort stage and INTO the recon stage, AFTER kssynth, so they operate on the cross-segment-fused templates instead of the raw KS templates.
- Update `containers/axon-recon/build_local_image.sh` to stage `~/dev/pkgs/kssynth` into the build context (same pattern as SLAy / UnitMatchPy).

Total estimated touch for slices 1-7 (kssynth repo): ~500-700 LoC + ~350 LoC tests.

---

## 8. The cascading wins

Once kssynth exists and slice 10 lands, the recon stage's phase_sequence collapses significantly:

```yaml
# Before:
stages.reconstruct.phase_sequence:
  - analyzers
  - extract_partial_templates
  - build_templates
  - plot_templates_v2
  - report_templates
  - ...

# After:
stages.reconstruct.phase_sequence:
  - analyzers
  - kssynth                  # ← internalizes extract_partial_templates + build_templates
  - plot_templates_v2
  - report_templates
  - ...

# And further down the line (once bombcell+SLAy migrate out of spikesort):
stages.reconstruct.phase_sequence:
  - analyzers
  - kssynth
  - bombcell_label           # operates on synth sorter_output → richer feature inputs
  - merge_SLAy               # operates on synth sorter_output → cross-segment-aware merges
  - plot_templates_v2
  - report_templates
  - unitmatch                # consumes the same synth sorter_output across DIVs
  - ...

# spikesort stage shrinks correspondingly:
stages.spikesort.phase_sequence:
  - bootstrap_concat_binary
  - sort
  - snapshot_sorter_output
  - cleanup_concat_binary
  # bombcell_label and merge_SLAy migrated out
```

This is exactly the kind of consolidation the phase-roster tracker entry calls for. kssynth is the enabling primitive.

---

## 9. Dependencies / blockers

- Python 3.10+ (match SpikeInterface's modern baseline).
- `spikeinterface >= 0.100` for the SortingAnalyzer API.
- `numpy`, `pandas`, `scipy` — all already in the axon_recon container.
- Nothing axon_recon-specific. Package is consumable in any environment with the deps above.
- For axon_recon integration: needs to be staged into shifter. `build_local_image.sh` extension is small (mirrors the SLAy / UnitMatchPy lines).

---

## 10. Open questions

1. **Package name.** **Locked: `kssynth`** (Kilosort synthesizer; user-confirmed 2026-05-18). Repo at `~/dev/pkgs/kssynth/`. Local `git init` only at slice 1; user creates the GitHub remote when ready.

2. **Default aggregation policy.** Options:
   - `mean` — simple average across analyzers.
   - `spike_count_weighted_mean` — weight each analyzer's per-unit contribution by its spike count for that unit. Recommended default.
   - `median` — robust to outlier analyzers.
   - `time_aligned_average` — cross-correlation-align per-analyzer templates before averaging; handles cross-segment drift.

3. **License.** MIT to match SLAy and remove friction for other research labs.

4. **Repo hosting.** GitHub under your own org (same as SLAy / UnitMatch). Decide before slice 1.

5. **PyPI publication.** Eventually yes — that's the point. Push after the API stabilizes (post-slice 7, real-user feedback). Sibling-checkout pattern works without PyPI for the axon_recon use case.

6. **CI / tests.** GitHub Actions matrix across Python 3.10/3.11/3.12 + Linux. Standard pytest layout. Coverage tracked but not gated. Skip macOS/Windows unless someone asks — SI's macOS story is rocky.

7. **Documentation.** Thorough `README.md` with `synthesize()` examples and a `CHANGELOG.md` are sufficient through v1. Sphinx + readthedocs once external users start asking.

8. **What happens if analyzers don't share a sorting?** v1: raise a clear error. v2 (deferred): support cross-sorting fusion with a caller-supplied unit-id mapping. The common case (segment analyzers from the same concat sort) is v1 territory.

9. **Drift-correcting fusion as default vs opt-in?** `time_aligned_average` is the principled default for axon_recon's use case (different recordings have different drift) but is more expensive and assumes the templates are similar enough to cross-correlate-align. Make it opt-in for v1, evaluate as a default candidate after slice 7 with real data.
