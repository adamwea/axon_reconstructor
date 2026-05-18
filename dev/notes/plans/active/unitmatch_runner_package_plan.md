# unitmatch-runner — UnitMatch / DeepUnitMatch wrapper for cross-session unit tracking

Status: scoped (idea → in-plan). Same operating contract: one slice at a time, `claude:` commit prefix, separate git repo at `~/dev/pkgs/<name>/`.

This plan extracts the "given N sorter outputs, decide which units are the same neuron across them" capability into its own Python package. Wraps UnitMatchPy in v1; adds a DeepUnitMatch backend in v2 (gated on having an HD-MEA-trained model). axon_recon's analysis stage gets a thin phase that calls this library.

**See also (companion plans):**
- `ks_synthesizer_package_plan.md` — kssynth, the sibling pip package that produces SI-compliant sorter_output folders from segment analyzers. Typical input source for unitlink, but unitlink also accepts any kilosort-shape sorter_output (e.g. vanilla KS runs).
- `unitmatch_phase_plan.md` — the thin axon_recon analysis-stage phase that calls unitlink once per chip-well group, with per-DIV synth sorter_outputs (from kssynth) + segment analyzers as inputs.

---

## 0. End-state, in one paragraph

A small Python package — working name `unitlink` (alt: `unitmatch_runner`, `unittrack`) — whose public API is essentially: **give me a list of sorter_output folders, get back a table describing which units reappear across them**. Inputs are KS-shape sorter_output folders, synthetic or otherwise (produced by kssynth, a normal kilosort run, or anyone else). Outputs are a flat tabular match summary suitable for spreadsheets, dashboards, and downstream analysis. The package is intentionally NOT axon-recon-specific; any researcher running multi-session spike sorting can use it.

---

## 1. Why a separate package

| Reason | Detail |
|---|---|
| **Right-sized abstraction** | "Take some sorter outputs, get a unit-correspondence table" is exactly one operation. It deserves its own ~one-thing-well module. |
| **Independent backend story** | Classical UnitMatch and DeepUnitMatch share the SAME input shape and produce the SAME output structure but differ in inference path. A single wrapper that selects backend is more useful than two separate consumer integrations. |
| **Generally useful to SpikeInterface users** | Anyone with multi-session spike sorting + wanting cross-session unit identity benefits. Not specific to axon_recon, HD-MEA, or any one pipeline. |
| **Clean boundary with axon_recon** | axon_recon's analysis stage owns "which (chip, well) groups have multiple sessions"; this package owns "match the units across those sessions". One library call per group. |
| **Versioning** | Wrapper logic can evolve (better defaults, new backends, threshold calibration policies) independently of either the matchers or axon_recon. |

What this is NOT solving:
- Not a spike sorter.
- Not a sorter_output builder — that's kssynth's job. This package just consumes them.
- Not a curation GUI — UnitMatchPy ships its own; that's a separate concern.

---

## 2. Package scope

### In scope (v1, classical UnitMatch backend)

1. **Input adapter.** Take a list of paths to sorter_output-shape directories (synthetic from kssynth, or vanilla kilosort outputs). For each:
   - Resolve the canonical files (`spike_times.npy`, `spike_clusters.npy`, `templates.npy`, `cluster_KSLabel.tsv`, `channel_positions.npy`).
   - Validate basic schema; raise actionable errors when something's wrong (e.g. missing cluster_*.tsv → suggest running `kssynth.ensure_cluster_tsv_coherence`).
2. **Per-unit two-halves waveform computation.** UnitMatch's input shape requires `(spike_width, n_channels, 2)` per unit, where the last axis is first-half / second-half mean waveforms over the recording. This package owns that computation: split each unit's spike times at the temporal midpoint, average each half's waveforms, rasterize onto the union channel grid across all input sessions. Writes `RawWaveforms/Unit{N}_RawSpikes.npy` per session as a side effect (UnitMatchPy reads from there).
3. **Union channel grid across sessions.** The N input sessions can have different channel sets (channel-set drift across DIVs, etc.). Compute the union, rasterize all per-unit two-halves waveforms onto it, use as the global grid handed to UnitMatch.
4. **Classical UnitMatch invocation.** Wrap UnitMatchPy's algorithmic spine (extract_parameters → extract_metric_scores → Naive Bayes → threshold → UID assignment) behind one function call. Reasonable defaults for all UMPy parameters; per-call overrides via a config dict.
5. **Output writer.** Produce a tidy match table and UID assignment in a stable, documented schema (see §3).
6. **CLI.** `unitlink match --inputs <path>... --out <path>` as the one-shot entry point.

### Out of scope (v1)

- DeepUnitMatch backend (v2).
- Manual GUI curation. The user can hand the outputs to UnitMatchPy's GUI separately.
- Drift correction beyond what UMPy does natively.
- Per-pair score-distribution plots / reports — flat output files are enough for v1.

### v2 — DeepUnitMatch backend

- Optional `backend="deep"` flag. Loads a DUM model checkpoint, runs DUM's feature-extraction + inference pipeline, returns the SAME output schema as v1 (match table + UID assignment).
- **Blocker for HD-MEA users**: DUM's released model is trained on Neuropixels 1.0 / 2.0 (per UnitMatch README). Users with other probes have to train their own model. Not a blocker for shipping v2; it's a blocker for HD-MEA users actually USING the deep backend. The wrapper supports the `--model-checkpoint` arg as a v2 parameter so once a HD-MEA model exists it slots in.
- Same input/output schema as v1 — switching backends doesn't change the consumer's contract.

### v3 — Sensible threshold calibration

- Per-pair-of-sessions threshold sweep, with a metric for "yield stability" — pick a threshold where small changes don't move match yields dramatically. Surface as a recommendation in summary.json.

---

## 3. Inputs / Outputs

### Inputs

```python
unitlink.match(
    sorter_outputs: list[Path],         # one path per session
    out_folder: Path,
    backend: Literal["classical", "deep"] = "classical",
    model_checkpoint: Path | None = None,    # required if backend == "deep"
    good_units_only: bool = True,
    match_threshold: float | None = None,    # None = use backend default
    param_overrides: dict | None = None,     # forwarded to UMPy params
    write_intermediate: bool = False,        # set True to keep RawWaveforms/, prob_matrix.npy
)
```

Each `sorter_outputs[i]` is a path to a directory loadable by `spikeinterface.read_sorter_folder(...)` (or vanilla kilosort layout). All paths must share consistent channel-position semantics (same xy units, same physical layout) — the union-grid logic assumes physical xy positions across sessions are comparable.

### Outputs

All under `<out_folder>/`:

| File | Format | Schema |
|---|---|---|
| `match_table.tsv` | tab-separated, header row | `session_a_index`, `session_a_path`, `unit_a_id`, `session_b_index`, `session_b_path`, `unit_b_id`, `match_prob`, `total_score`, `match_method` (`classical` or `deep`) |
| `uid_assignment.tsv` | tab-separated, header row | `session_index`, `session_path`, `original_unit_id`, `uid_conservative`, `uid_intermediate`, `uid_liberal` |
| `summary.json` | json | counts (sessions, units per session, candidate pairs, matched pairs, UIDs per mode), wall time, params used, channel grid size + per-session coverage fraction, backend name + model checkpoint (deep) |
| `prob_matrix.npy` (optional) | numpy | full pairwise probability matrix (n_units × n_units, stacked across sessions); only when `write_intermediate=True` |
| `clus_info.json` (optional) | json | session_switch boundaries, per-session unit rosters; only when `write_intermediate=True` |
| `cache/RawWaveforms/<session>/Unit{N}_RawSpikes.npy` (optional) | per session | only when `write_intermediate=True` — useful for hand-running UMPy's GUI on borderline cases |

The first two files are the primary outputs. They're the "table describing which units are reappearing" — consumable by spreadsheets, pandas, dashboards, downstream analyses, manual review.

---

## 4. Module layout

```
~/dev/pkgs/unitlink/                       (sibling to axon_recon, kssynth, SLAy, UnitMatch)
├── README.md
├── LICENSE                                (MIT, matching SLAy / kssynth)
├── pyproject.toml
├── src/unitlink/
│   ├── __init__.py
│   ├── api.py                             (match() top-level)
│   ├── policies.py                        (config dataclasses: TwoHalvesPolicy, ChannelGridPolicy, …)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── two_halves.py                  (per-unit temporal midpoint split + rasterize)
│   │   ├── union_grid.py                  (channel union across N sorter_outputs)
│   │   ├── sorter_output_reader.py        (read sorter_output files via SI; validate schema)
│   │   └── output_writer.py               (writes match_table.tsv, uid_assignment.tsv, summary.json)
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── classical.py                   (UnitMatchPy wrapper — v1)
│   │   └── deep.py                        (DeepUnitMatch wrapper — v2, stub in v1)
│   └── cli.py                             (unitlink match <args>)
└── tests/
    ├── test_two_halves.py
    ├── test_union_grid.py
    ├── test_sorter_output_reader.py
    ├── test_classical_backend.py
    ├── test_output_writer.py
    └── test_api_end_to_end.py
```

The `backends/` split keeps the classical and deep paths cleanly separated. `api.match()` selects the backend and hands off; everything before (input prep, two-halves, union grid) and after (output writing) is shared.

---

## 5. Public API

### Library

```python
import unitlink

result = unitlink.match(
    sorter_outputs=[
        "/path/to/div04/synth_sorter_output",
        "/path/to/div07/synth_sorter_output",
        "/path/to/div12/synth_sorter_output",
        # …
    ],
    out_folder="/path/to/output/M08073_well000",
    backend="classical",
    match_threshold=0.5,
)
# result.match_table:    Path to match_table.tsv
# result.uid_assignment: Path to uid_assignment.tsv
# result.summary:        Path to summary.json
# result.n_matches:      int
# result.n_uids_intermediate: int
```

### Direct module use

```python
from unitlink.core.two_halves import compute_two_halves_for_session
from unitlink.core.union_grid import compute_session_union_grid
from unitlink.backends.classical import run_classical_unitmatch
```

### CLI

```bash
unitlink match \
    --inputs /path/to/div04 /path/to/div07 /path/to/div12 \
    --out /path/to/output/M08073_well000 \
    --backend classical \
    [--match-threshold 0.5] \
    [--write-intermediate]
```

---

## 6. Two-halves and union grid — what was scoped out of kssynth lives here

The two-halves split (temporal midpoint, per-half mean waveform, axis-2 stack) is the UnitMatch-specific shape conversion we deliberately kept OUT of kssynth. That logic lives in this package as `core/two_halves.py`. Operationally:

For each input sorter_output:
1. Read `spike_times.npy` + `spike_clusters.npy` to get per-unit spike-time arrays.
2. Determine the recording's temporal midpoint (sample index = total_samples / 2; reads from `params.py` or asks the user to supply via CLI).
3. For each good unit, partition its spike times at the midpoint into `times_a` and `times_b`.
4. For each half, compute the per-unit mean waveform. For a synthetic sorter_output from kssynth, the templates.npy already has the cross-segment-fused per-unit mean; we'd need per-half data from the underlying analyzer — which we don't have access to from inside the synth folder. **So the per-half computation needs raw waveform access.**

This surfaces a design tension. Two paths:

**Path A — accept the limitation.** When the input is a kssynth synth folder (no raw waveform access), use the cross-segment-fused `templates.npy` for BOTH halves. This is the "duplicate" cheat we explicitly rejected for calibration purposes. Bad.

**Path B — operate on richer inputs.** Require the caller to supply per-session SpikeInterface analyzers alongside the sorter_output, so per-unit waveform data is available for the two-halves computation. The signature becomes:

```python
unitlink.match(
    sessions=[
        {"sorter_output": Path("…"), "analyzers": [...] | "auto"},
        {"sorter_output": Path("…"), "analyzers": [...] | "auto"},
    ],
    out_folder=Path("…"),
    backend="classical",
)
```

Where `analyzers="auto"` tries to discover them from the sorter_output's neighboring directories (e.g. `<sorter_output>/../cache/analyzers/segments/`). If neither is supplied / discoverable, raise a clear error.

**Path B is correct.** It preserves the "real two-halves = real calibration" invariant we established. Documented in the package as: this is a UnitMatch wrapper, not a magic-data-recovery tool — it needs the same inputs UnitMatchPy needs.

Decision in §10 open questions: is `analyzers="auto"` discovery worth the complexity for v1, or do we just require explicit paths? Lean toward explicit for v1; add auto-discovery in v1.1 once the call sites stabilize.

---

## 7. Implementation slices

One commit per slice. `claude:` prefix. Logged in the new repo's `commit_log.md`.

### Slice 1 — scaffolding (no behavior)
- New git repo at `~/dev/pkgs/unitlink/` with `pyproject.toml`, `README.md`, `LICENSE` (MIT), `commit_log.md`, basic CI.
- Empty `src/unitlink/` with stub `api.match()` raising `NotImplementedError`.
- Pip-installable, CI green on the stub test.

### Slice 2 — `sorter_output_reader.py`
- Function: `read_sorter_output(path) -> SorterOutputView` returning a small dataclass with the canonical files' paths + lazy-loaders for the arrays.
- Validates: spike_times.npy, spike_clusters.npy, channel_positions.npy, cluster_KSLabel.tsv exist; raises with actionable suggestions if not.
- Optional `via_spikeinterface=True` path: load via `si.read_sorter_folder()` for richer schema validation.
- Tests against a synthetic sorter_output folder fixture.

### Slice 3 — `union_grid.py`
- Function: `compute_session_union_grid(sessions, tolerance_um=1.0) -> ChannelGrid` (could re-use kssynth's ChannelGrid if useful; for v1 the package keeps a local copy so deps stay minimal).
- Tests: 3 sessions with partially-overlapping channel sets → assert union size, per-session index lookup.

### Slice 4 — `two_halves.py`
- Function: `compute_two_halves_for_session(sorter_output, analyzers, midpoint_sample, grid) -> dict[unit_id, np.ndarray]` returning per-unit `(spike_width, n_global_channels, 2)` arrays. Reads per-unit spike times from sorter_output, per-spike waveforms from the supplied analyzers' waveforms extension.
- Tests: synthetic 2-unit session with known spike-time distribution; assert split-by-sample-index correctness, non-degenerate halves, correct rasterization.

### Slice 5 — `backends/classical.py` (UnitMatchPy wrapper)
- Function: `run_classical_unitmatch(per_session_waveforms, channel_grid, params) -> ClassicalResult` returning the prob_matrix, candidate_pairs, UID assignment via `aid.assign_unique_id`. Internally calls UMPy's `extract_parameters → extract_metric_scores → bayes → threshold → UIDs` spine.
- Tests: synthetic 2-session 3-units-each, non-degenerate halves; assert match_table is non-empty and UIDs have the expected three-mode columns.

### Slice 6 — `output_writer.py`
- Function: `write_outputs(result, sessions, out_folder, write_intermediate=False)` writes `match_table.tsv`, `uid_assignment.tsv`, `summary.json`, and optionally `prob_matrix.npy` + `clus_info.json` + cached `RawWaveforms/` dirs.
- Tests: round-trip via pandas — read each file back, assert schema + counts match the input result.

### Slice 7 — `api.match()` orchestration + CLI
- Wire core modules behind the public `match()` function from §5.
- CLI entry point.
- End-to-end test: synthetic 2-session data → run `unitlink.match(...)` → verify all output files exist and conform to schema.
- v1-feature-complete. Tag `v0.1.0`.

### Slice 8 (v2) — `backends/deep.py` (DeepUnitMatch wrapper)
- Function: `run_deep_unitmatch(per_session_waveforms, channel_grid, model_checkpoint, params) -> DeepResult`. Loads the DUM model from the checkpoint, runs feature extraction + inference, returns the same shape as ClassicalResult.
- `api.match(backend="deep", model_checkpoint=...)` selects this backend.
- Tests: ship a tiny stub model checkpoint or mock the inference call; assert the wrapper produces a result with the expected schema.
- Document the HD-MEA-model-blocker prominently in the README.

### Slice 9 (v3) — threshold calibration
- Function: `recommend_threshold(prob_matrix, session_switch) -> {"threshold": float, "yield_curve": [(t, n_matches)]}` reports the match-yield curve over a range of thresholds and recommends a stable point.
- Surface in `summary.json` whenever the user doesn't pin a threshold.

### Slice 10 — axon_recon integration (separate plan, in axon_recon repo)
The analysis-stage `unitmatch` phase (see `unitmatch_phase_plan.md`) collapses to:

```python
unitlink.match(
    sessions=[
        {"sorter_output": ds_synth_sorter_path, "analyzers": ds_segment_analyzers}
        for ds in chip_well_group
    ],
    out_folder=group_out_dir,
    backend="classical",
)
```

The phase becomes ~20 lines of glue. All the heavy lifting is in this package.

Update `containers/axon-recon/build_local_image.sh` to stage `~/dev/pkgs/unitlink` into the build context.

Total estimated touch for slices 1-7 (unitlink repo): ~400-600 LoC + ~250 LoC tests.

---

## 8. Dependencies

- Python 3.10+ (match SI baseline).
- `spikeinterface >= 0.100` for sorter_output reading.
- `UnitMatchPy >= 3.3.0` for classical backend.
- `numpy`, `pandas`, `scipy`.
- Optional: `kssynth` — only needed if the consumer wants to call `kssynth.ensure_cluster_tsv_coherence(...)` to fix up its inputs before matching. unitlink itself doesn't require kssynth at import time.
- v2: DeepUnitMatch's torch dependencies (torch >= 2.0, lightning).

Nothing axon-recon-specific.

---

## 9. Cascading wins

Once both kssynth and unitlink exist, axon_recon's recon + analysis stages become:

```yaml
stages.reconstruct.phase_sequence:
  - analyzers                # build segment analyzers
  - kssynth                  # fuse → synth sorter_output (one per dataset, per well)
  - plot_templates_v2
  - report_templates
  - ...

stages.analysis.phase_sequence:
  - compute_metrics
  - unitmatch                # call unitlink.match() once per chip-well group
```

The recon stage produces one synth sorter_output per (dataset, well). The analysis stage's unitmatch phase walks chip-well groups, calls `unitlink.match()` with the group's N synth sorter_outputs as inputs, writes the result under `<analysis_outputs>/unitmatch/<chip>/<well>/`. ~20 lines of glue in the analysis phase runner; everything substantive is in the two sibling packages.

---

## 10. Open questions

1. **Package name.** Working: `unitlink`. Alternatives: `unitmatch_runner` (explicit), `unittrack` (descriptive), `umrun` (terse), `crosssort` (alt-concept). Decide before slice 1.
2. **Analyzers input shape.** Path B in §6: require the caller to supply per-session SpikeInterface analyzers alongside each sorter_output, for the two-halves waveform computation. `analyzers="auto"` discovery (look in `<sorter_output>/../cache/analyzers/segments/`) is convenient but adds implicit-path complexity. Lean toward explicit for v1; add auto-discovery as v1.1 once call sites stabilize.
3. **License.** MIT, matching SLAy / kssynth / UnitMatchPy.
4. **Repo hosting.** GitHub under your own org. Same as the other sibling packages.
5. **PyPI publication.** Eventually; not blocking axon_recon integration.
6. **DeepUnitMatch HD-MEA training.** v2 ships the wrapper; users of non-Neuropixels probes need to train their own model. Add a stub `unitlink.training` module pointing at UnitMatchPy's DUM training notebook for users who want to train? Defer.
7. **Threshold default.** UnitMatchPy's `match_threshold=0.5` is the documented default. Should the wrapper override to something more conservative for HD-MEA's overlapping-template regime? Pin at 0.5 in v1, evaluate at slice 7 against real-data runs, adjust if data warrants.
8. **GUI handoff.** UnitMatchPy ships an interactive GUI. The `write_intermediate=True` mode writes the artifacts the GUI needs (`output_prob_matrix.npy`, `clus_info.json`, `RawWaveforms/`). Document that calling `unitlink.match(..., write_intermediate=True)` makes the outputs immediately GUI-loadable. Not a code change in this package — just docs.
9. **Multi-backend output schema unification.** Classical and deep backends produce slightly different intermediate structures (deep emits a continuous similarity score, classical emits Naive Bayes posteriors). The wrapper's `match_table.tsv` schema needs to be the same regardless of backend; deep's "score" maps to `match_prob` and a `match_method` column carries the backend label. Lock this in at slice 8 design.

---

## 11. Relationship to the axon_recon unitmatch phase plan

The companion `unitmatch_phase_plan.md` is the consumer of this package on the axon_recon side. After this plan landed, that phase plan was trimmed to its glue-layer scope (~100 LoC for group discovery + path resolution + one `unitlink.match()` call per group). The two-halves split, union channel grid, UnitMatch invocation, output schemas — all the substantive algorithm — live here. The phase plan's tests check wiring (was `unitlink.match` called once per group with the right inputs?); algorithmic correctness tests live in this package's test suite. The boundary is clean: unitlink is generally useful for anyone matching units across spike-sorting sessions; axon_recon's phase is just one consumer of it.
