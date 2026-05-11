# Analysis stage + dashboard plan

Add a final `analysis` stage that consumes recon (and later spikesort/preprocess) outputs,
emits a per-well metrics artifact (Parquet tables + manifest.json), and a separate
top-level `axon-recon dashboard` command that serves an interactive Plotly Dash app over
those artifacts.

Branch suggestion: `analysis-stage-and-dashboard` off `dev_branch2` (the main branch).

Test baseline at slice 0: capture the current `pytest src/axon_recon/pipeline/ -q
--ignore=src/axon_recon/pipeline/tests/test_progress.py` failing-test list before any
slice 1 work. Pre-existing failures unrelated to this plan are allowed to stay red;
slices must not introduce *new* red tests.

---

## 0. Goal And End State

After this plan lands:

- A new `analysis` stage runs per-well like every other stage. Default phase sequence is
  intentionally minimal (one phase, `compute_metrics`). Invocation:
  ```
  axon-recon stages analysis --config debug/debug.runtime.yml [--target-dataset N --limit-wells K]
  axon-recon-container ... stages analysis ...
  ```
- Each well writes a self-describing artifact directory:
  ```
  <well>/analysis_outputs/
    manifest.json
    tables/
      units.parquet
      well_summary.parquet
  ```
  `manifest.json` records `schema_version`, `pipeline_version`, identity fields
  (project, recording_date, chip_id, scan_type, run_id, well_id, DIV, genotype,
  media, plating_density), and the list of emitted tables.
- `units.parquet` carries one row per reconstructed unit with the 4 MVP metrics plus
  all per-unit identity / quality / filter columns.
- `well_summary.parquet` carries one row per well with aggregates.
- A separate top-level command `axon-recon dashboard --config <yaml>` walks the same
  target scope as `stages` (respecting `--target-dataset` / `--limit-wells` /
  `--limit-datasets`), discovers every `<well>/analysis_outputs/manifest.json`,
  concatenates the parquet tables, and serves a Dash app on `localhost:8050`
  (overridable via `--port`).
- The dashboard's left rail exposes **modular, independent** inclusion/exclusion filters
  (no precomputed `qc_pass` boolean — see §4). The main pane renders one starter plot
  + a table view in slice 4, with additional plot types (box-with-significance,
  scatter+facet, export buttons) bolted on in slices 5–6.

Non-goals for this plan:

- No spikesort/preprocess-derived metrics yet (the analysis stage is recon-only at MVP).
  Adding them is a follow-on slice plan.
- No GTR-pickle unpickling at MVP — the four starter metrics are derivable from the JSON
  artifacts already on disk (`branches.json`, `unit_reconstruction_summary.json`,
  `merged_contributing_electrode_ids.json`, `unit_templates_summary.json`). Slice 2's
  metric implementations must not import `pickle` against `gtr.pkl`.
- No PDF/HTML report-export pipeline beyond per-figure download buttons.
- No multi-user auth, no cloud hosting.

---

## 1. Operating Notes (read before every slice)

- **Conda env:** `conda run -n axon_recon <cmd>`. Never assume another env.
- **DO NOT TOUCH** any spikesort stage code (`src/axon_recon/pipeline/stages/spikesort/`)
  or the running container. The user is currently re-running spikesort on dataset 11
  and the analysis stage must stand entirely on existing recon outputs. If a slice
  needs a spikesort field later, defer it.
- **DO NOT TOUCH** any active spikesort scratch outputs under
  `/mnt/disk15tb/adamm/scratch/axon_recon_scratch/outputs/.../spikesort_outputs/`
  except by reading them read-only. The analysis stage writes only under
  `<well>/analysis_outputs/`.
- **Test fixture wells available now (recon already produced):**
  - `/mnt/disk15tb/adamm/scratch/axon_recon_scratch/outputs/Media_Density_T5_02182026_AR/260326/M08073/AxonTracking/000208/well000/recon_outputs/`
    has unit dirs under `units/<unit_id>/` with all the JSONs the metrics need.
  - The recon `_` sibling (`recon_outputs_/`) is an older snapshot — use the
    canonical `recon_outputs/` for fixtures.
- **Path-key convention** (single source of truth, used in every identity column,
  config field, dashboard pivot):
  ```
  raw_data_h5_path = .../<project>/<recording_date>/<chip_id>/<scan_type>/<run_id>/data.raw.h5
  e.g. Media_Density_T5_02182026_AR/260319/M06804/AxonTracking/000174/data.raw.h5
        ^project                     ^date  ^chip   ^scan_type     ^run
  ```
  `recording_date` is parsed as YYMMDD → ISO `YYYY-MM-DD`. `scan_type` defaults to the
  literal string from the path. `run_id` stays the zero-padded string (`"000174"`).
- **Mutation safety:** analysis writes only under `<well>/analysis_outputs/`. Never write
  to `recon_outputs/`, `spikesort_outputs/`, `preprocess_outputs/`.
- **Per-well independence:** the analysis stage must run wells in parallel via the
  existing harness, same pattern as preprocess/spikesort/reconstruct.

---

## 2. Discovery Targets (read these before slice 1)

Confirm current state before touching anything.

### 2.1 Stage package conventions

Read the spikesort stage as the most current pattern reference (every surface area we
need: orchestrator, config dataclass, YAML parser, target runner, runtime phase plan,
CLI dispatcher, aliases, api.py shim, __init__ re-exports, pipeline/runner.py
registration):

- `src/axon_recon/pipeline/stages/spikesort/__init__.py`
- `src/axon_recon/pipeline/stages/spikesort/api.py`
- `src/axon_recon/pipeline/stages/spikesort/cli.py`
- `src/axon_recon/pipeline/stages/spikesort/config.py` (sections: `SpikesortStageConfig`,
  `DEFAULT_SPIKESORT_PHASE_SEQUENCE`, `_SPIKESORT_PHASE_ALIASES`,
  `parse_spikesort_stage_config`)
- `src/axon_recon/pipeline/stages/spikesort/runner.py` (just the
  `run_spikesort_cleanup_concat_binary_stage` function — small, clean, parallel pattern
  for our `run_analysis_compute_metrics_stage`)
- `src/axon_recon/pipeline/stages/spikesort/orchestrators/cleanup_concat_binary.py`
  (clean orchestrator pattern)
- `src/axon_recon/pipeline/stages/spikesort/orchestrators/__init__.py`
- `src/axon_recon/pipeline/cli.py`: `_STAGE_ALIASES`, `_STAGE_HANDLERS`, imports
- `src/axon_recon/pipeline/runner.py`: search `_run_spikesort_cleanup_concat_binary_target`
  + `run_spikesort_cleanup_concat_binary_from_runtime` + the
  `_SPIKESORT_DIRECT_PHASE_LABELS` map + the `resource_attr_by_phase_label` map.
- `slice 13 commit` (`43bbcc2 claude: spikesort-merge-cleanup, add cleanup_analyzers
  phase (slice 13)`) is the most recent end-to-end "added a brand new phase" diff —
  the agent should `git show 43bbcc2 --stat` and read the diff as a checklist of every
  call site that needs to be touched for a new stage.

### 2.2 Recon outputs schema (the analysis stage's input)

For one unit directory
`<well>/recon_outputs/units/<unit_id>/`:

- `unit_reconstruction_summary.json` — `status` ("ok" / others), `outputs` (paths),
  `grid_sort_metrics` containing already-computed values including
  `template_density`, `max_amplitude`, `max_ptp`, `max_delay`, `max_negative_peak`.
  Always read this first; skip units with `status != "ok"`.
- `branches.json` — dict with `unit_id` + `branches` (list). Each branch dict has:
  `branch_index`, `channels` (list of channel indices in path order), `polyline_xy`
  (list of `[x, y]` per node), `velocity` (m/s, fitted), `offset`, `r2`, `pval`,
  `distances` (between successive nodes), `peak_times`. **Branch count = `len(branches)`.
  Branch length = `sum(distances)`.**
- `merged_contributing_electrode_ids.json` — dict with `electrode_ids` (list of
  electrode IDs contributing to the merged reconstruction). **Count = recon's selected
  channel count.**
- `unit_templates_summary.json` — has `unit_location`, `channel_scope`,
  `selected_template_source`, `grid_sort_metrics` (overlap with the recon summary —
  prefer reading `template_density` from `unit_reconstruction_summary.json`).
- `gtr.pkl` — pickled GraphTracker object. **Do not load at MVP.** Reserve for slice
  beyond this plan when a metric truly requires a graph walk not exposed in JSON.

The recon stage's per-well summary contexts under
`<well>/recon_outputs/context/*.json` (`generate_gtrs_summary.json`,
`build_templates_summary.json`, etc.) give the well-level provenance — slice 2 reads
these for the manifest.

### 2.3 Data config (the analysis stage's identity columns)

`debug/debug.data.yml` carries per-dataset top-level `DIV` and per-well
`attributes: {plating_density, media, genotype}`. The analysis stage threads those
into every output row + the manifest. The runtime-config helpers that already read
these (`axon_recon.runtime_config.RuntimeConfig` + the
`build_spikesort_inputs_for_target` family) are the reference for how to plumb
attributes through to a per-well runner.

### 2.4 Plotly / Dash deps (don't yet exist in this repo)

- `environment.yml` has `pandas` + `scipy`. Needs: `pyarrow` (parquet),
  `plotly>=5.18`, `dash>=2.14`, `dash-ag-grid`, `statsmodels`.
- `containers/axon-recon/Dockerfile` has the runtime spec on line 8 — bump it to
  include the same.
- Slice 4 (the first dashboard slice) is the right place to add these. Don't add them
  in slice 1–3 (avoids cross-coupling slice tests with new deps).

---

## 3. Smoke Matrix

All smokes use the NAS-bypass wrapper pattern the user proved out:

```bash
axon-recon-container \
  --no-config-mounts \
  --mount /mnt/disk15tb/adamm/scratch:/mnt/disk15tb/adamm/scratch:rw \
  --mount /mnt/disk15tb/adamm/dev/pkgs/axon_reconstructor/debug:/mnt/disk15tb/adamm/dev/pkgs/axon_reconstructor/debug:ro \
  --gpus all \
  stages analysis \
  --config debug/debug.runtime.yml \
  --target-dataset 11 --limit-wells 1
```

If the autonomous agent cannot launch the container (the user noted they are also
running it), it MUST instead run the stage in-process via the conda env:

```bash
conda run -n axon_recon axon-recon stages analysis --config debug/debug.runtime.yml \
  --target-dataset 11 --limit-wells 1
```

(no `--force-restart` needed for a first-time analysis run; outputs are new).

Smokes:

- **A1 — analysis stage skeleton end-to-end (after slice 1):** stage runs, writes
  `<well>/analysis_outputs/manifest.json` with `schema_version=axon_analysis_v1`,
  no tables yet (or empty tables). No crash on units with `status != "ok"`.
- **A2 — units.parquet populated (after slice 2):** the parquet exists, has one row
  per recon unit with `status="ok"`, the 4 starter metric columns are numeric and
  non-null for those units, identity columns match what the data config declares.
- **A3 — well_summary.parquet populated (after slice 3):** one row per well with
  unit-count aggregates and metric means/medians.
- **A4 — dashboard boots (after slice 4):** `axon-recon dashboard --config ... --port
  8051 --no-browser` starts, replies 200 on `/_dash-layout`, exits cleanly on SIGTERM.
- **A5 — significance brackets render (after slice 5):** box plot with `genotype` on
  x-axis renders without raising in the Dash callback test.

Each slice's Acceptance section names exactly which smokes apply.

---

## 4. Modular filter contract (binding on slices 2 + 4)

The dashboard composes inclusion masks from independent toggles. The stage emits all
the raw columns the dashboard needs; **do not bake a `qc_pass` boolean into
`units.parquet`**.

Required raw columns on `units.parquet` (in addition to identity + metrics):

- `recon_status` ("ok" / other) — always emit, dashboard hides non-ok by default.
- `bombcell_label` — string ("good" / "non_soma_good" / "mua" / "noise" / "unsorted" /
  None). Read from `<well>/spikesort_outputs/sorter_output/cluster_group.tsv` (or
  `cluster_KSLabel.tsv` fallback) keyed by `unit_id`. **Read-only.** If the file is
  missing (e.g., bombcell hasn't run), set to None and continue.
- `num_spikes` — int. Read from the same TSV's neighborhood (or
  `<well>/spikesort_outputs/sorter_output_snapshot/spike_clusters.npy` — count per
  cluster). If unavailable, set None.
- `num_branches` — int (already a metric).
- `recon_quality_score` — float, optional. If recon doesn't emit one, leave None;
  dashboard's threshold filter will treat None as "include".

The dashboard left rail in slice 4 surfaces each of these as an independent control:

- `bombcell_label` allowlist (multi-select)
- `num_spikes` >= threshold (numeric input)
- `num_branches` >= threshold (numeric input)
- `recon_quality_score` >= threshold (numeric input, blank = no filter)
- `recon_status` == "ok" (checkbox, default on)

Plus the identity-pivot multi-selects:

- `project`, `chip_id`, `well_id`, `DIV` (continuous via slider or numeric range),
  `genotype`, `media`, `plating_density`, `scan_type`, `run_id`.

`DIV` stays **continuous** (numeric range slider, no pre-binning).

---

## 5. Slice plan

Each slice = one commit, prefixed `claude: analysis-stage-and-dashboard, <description>
(slice N)`. Co-Authored-By trailer mandatory.

### Slice 1 — Analysis stage skeleton

**Goal:** stage package exists, registers as `analysis`, runs per-well, writes an
empty-but-valid artifact dir + `manifest.json` per well. No metrics yet.

**Files (new):**
- `src/axon_recon/pipeline/stages/analysis/__init__.py`
- `src/axon_recon/pipeline/stages/analysis/api.py`
- `src/axon_recon/pipeline/stages/analysis/cli.py`
- `src/axon_recon/pipeline/stages/analysis/config.py` (with `AnalysisStageConfig`
  dataclass + `parse_analysis_stage_config` + `DEFAULT_ANALYSIS_PHASE_SEQUENCE =
  ("compute_metrics",)` + `_ANALYSIS_PHASE_ALIASES`)
- `src/axon_recon/pipeline/stages/analysis/runner.py` (with
  `run_analysis_compute_metrics_stage`)
- `src/axon_recon/pipeline/stages/analysis/orchestrators/__init__.py`
- `src/axon_recon/pipeline/stages/analysis/orchestrators/compute_metrics.py`
- `src/axon_recon/pipeline/stages/analysis/models/__init__.py`
- `src/axon_recon/pipeline/stages/analysis/models/results.py` (`AnalysisResult` dataclass)
- `src/axon_recon/pipeline/stages/analysis/tests/__init__.py`
- `src/axon_recon/pipeline/stages/analysis/tests/test_runner.py`
- `src/axon_recon/pipeline/stages/analysis/tests/test_config.py`

**Files (edited):**
- `src/axon_recon/pipeline/cli.py`: import + register `_STAGE_ALIASES["analysis"]`,
  `_STAGE_ALIASES["analysis.compute_metrics"]`, plus a few short aliases
  ("metrics", "compute_metrics"); add to `_STAGE_HANDLERS`.
- `src/axon_recon/pipeline/runner.py`: add `_run_analysis_compute_metrics_target`,
  `run_analysis_compute_metrics_from_runtime`, `_ANALYSIS_DIRECT_PHASE_LABELS`,
  resource-class map entry (`"compute_metrics": "compute_metrics_resource_class"`),
  full-stage `run_analysis_from_runtime` analogous to `run_spikesort_from_runtime`,
  and registration into the top-level stage dispatcher (search the file for how
  spikesort is registered and mirror).
- `debug/debug.runtime.yml`: add a top-level `analysis:` stage block with
  `enabled: true`, `output_rel_root: analysis_outputs`, `phase_sequence:
  [compute_metrics]`, and `phases.compute_metrics: {enabled: true, resource_class:
  cpu_light}`.

**Manifest schema (`manifest.json`):**
```json
{
  "artifact_type": "axon_recon_well_analysis",
  "schema_version": "axon_analysis_v1",
  "pipeline_version": "<from importlib.metadata or VERSION constant>",
  "project": "Media_Density_T5_02182026_AR",
  "recording_date": "2026-03-26",
  "chip_id": "M08073",
  "scan_type": "AxonTracking",
  "run_id": "000208",
  "well_id": "well000",
  "dataset_id": "<canonical key matching how spikesort/recon report it>",
  "DIV": 36,
  "well_attributes": {"genotype": "WT", "media": "DMEM", "plating_density": 80000},
  "written_at": "<ISO-8601 UTC>",
  "tables": {}
}
```
Slice 1 writes `tables: {}` (empty). Slice 2 adds `units.parquet`. Slice 3 adds
`well_summary.parquet`.

**Tests:**
- `test_config.py`: defaults + YAML round-trip (mirror
  `test_spikesort_config.py` structure).
- `test_runner.py`: writes manifest with all identity fields populated from a
  fixture data-config; respects `analysis_outputs` relpath; idempotent on rerun.

**Acceptance:**
- `git grep -nE "analysis_outputs|axon_analysis_v1" src/axon_recon/` → multiple hits
  (sanity that the new code is present, not just imports).
- `conda run -n axon_recon python -m pytest src/axon_recon/pipeline/stages/analysis/tests/
  -q` green.
- `conda run -n axon_recon python -m pytest src/axon_recon/pipeline/tests/test_cli_stage_sequence.py
  -q` green (the stage-alias parser tests pick up the new `analysis*` aliases —
  add parallel tests to that file).
- Smoke **A1** passes on `--target-dataset 11 --limit-wells 1`.

**Commit:** `claude: analysis-stage-and-dashboard, analysis stage skeleton + per-well manifest (slice 1)`

---

### Slice 2 — Starter metrics → `units.parquet`

**Goal:** the `compute_metrics` phase reads each recon unit dir, computes the four
starter metrics + raw filter columns, and writes `<well>/analysis_outputs/tables/units.parquet`.

**Metric definitions (must match these exactly):**

- `branch_count`: `len(branches.json["branches"])`. NaN if `branches.json` missing or
  `recon_status != "ok"`.
- `total_branch_length_um`: `sum(sum(branch["distances"]) for branch in
  branches.json["branches"])`. Distances are already euclidean per-segment in microns.
  Falls back to `sum(np.linalg.norm(np.diff(branch["polyline_xy"], axis=0), axis=1))`
  if `distances` is missing. NaN if no branches.
- `template_density`: `unit_reconstruction_summary.json["grid_sort_metrics"]["template_density"]`
  (already precomputed by recon — do NOT re-derive). NaN if missing.
- `recon_density`: `len(merged_contributing_electrode_ids.json["electrode_ids"]) /
  recon_bbox_area_um2`, where `recon_bbox_area_um2` = (max_x - min_x) * (max_y - min_y)
  over the union of all `polyline_xy` points across all branches in `branches.json`.
  If bbox area is 0 or branches list is empty, NaN.

**Required identity columns on every row:**

`project`, `recording_date`, `chip_id`, `scan_type`, `run_id`, `well_id`, `dataset_id`,
`unit_id`, `DIV`, `genotype`, `media`, `plating_density`.

**Required filter columns** (see §4):

`recon_status`, `bombcell_label`, `num_spikes`, `num_branches`, `recon_quality_score`.

**Additional handy passthroughs** (already on disk — costs nothing to include):

`max_amplitude_uv`, `max_ptp_uv`, `max_delay_ms` (from
`grid_sort_metrics`), `unit_location_x_um`, `unit_location_y_um` (from
`unit_templates_summary.json["unit_location"]`).

**Files (new):**
- `src/axon_recon/pipeline/stages/analysis/core/__init__.py`
- `src/axon_recon/pipeline/stages/analysis/core/metrics.py` — pure functions:
  `compute_unit_metrics(unit_dir, identity_cols) -> dict` plus the four helpers.
- `src/axon_recon/pipeline/stages/analysis/core/recon_io.py` — pure JSON readers
  with strict status checks and "missing file" fallbacks. NO pickle reading.
- `src/axon_recon/pipeline/stages/analysis/core/labels_io.py` — read
  `<well>/spikesort_outputs/sorter_output/cluster_group.tsv` (fallback
  `cluster_KSLabel.tsv`) for `bombcell_label` and per-cluster spike counts from
  `spike_clusters.npy` if present. Read-only; tolerate missing files.

**Files (edited):**
- `src/axon_recon/pipeline/stages/analysis/runner.py`: `run_analysis_compute_metrics_stage`
  enumerates `<well>/recon_outputs/units/*` dirs, calls `compute_unit_metrics` per
  unit, builds a `pandas.DataFrame`, writes `tables/units.parquet`, updates manifest
  `tables.units = "tables/units.parquet"` and an inline `unit_count: int`.
- `environment.yml`: add `pyarrow` (do not add plotly/dash here — defer to slice 4 to
  keep this slice's test surface small).
- `containers/axon-recon/Dockerfile`: bump the runtime spec to include `pyarrow`.

**Tests:**
- `test_runner.py`: synthetic fixture wells with stub JSONs → assert the 4 metrics
  compute to expected values, identity columns are stamped correctly, missing-file
  units round-trip as NaN rows (not exceptions).
- `core/tests/test_metrics.py`: unit tests for each of the 4 metric helpers with
  edge cases (empty branches list, 1-node branch, zero bbox area, missing distances).
- `core/tests/test_labels_io.py`: TSV parsing with edge cases (missing file,
  header-only, comments).

**Acceptance:**
- `units.parquet` written for the dataset-11 well000 fixture (`A2` smoke passes).
- Schema: `pa.read_table(units.parquet).schema` matches the documented column list
  (identity + 4 metrics + filter columns + handy passthroughs).
- All four metrics non-null for all `recon_status == "ok"` rows.
- Spikesort stage untouched (`git diff --stat HEAD~1` shows zero spikesort/* edits).

**Commit:** `claude: analysis-stage-and-dashboard, starter metrics + units.parquet (slice 2)`

---

### Slice 3 — `well_summary.parquet`

**Goal:** one row per well with aggregates over the unit table.

**Aggregations:**

- `unit_count_total` (all rows)
- `unit_count_recon_ok` (`recon_status == "ok"`)
- `unit_count_bombcell_good` (`bombcell_label == "good"`)
- `unit_count_bombcell_non_soma_good` (`bombcell_label == "non_soma_good"`)
- `mean_branch_count` / `median_branch_count` (over `recon_status == "ok"` units only)
- `mean_total_branch_length_um` / `median_total_branch_length_um` (same)
- `mean_template_density` / `median_template_density` (same)
- `mean_recon_density` / `median_recon_density` (same)
- All identity columns (`project`, `recording_date`, `chip_id`, `scan_type`,
  `run_id`, `well_id`, `dataset_id`, `DIV`, `genotype`, `media`, `plating_density`).

**Files (edited):**
- `src/axon_recon/pipeline/stages/analysis/core/metrics.py`: add
  `compute_well_summary(units_df, identity_cols) -> dict`.
- `src/axon_recon/pipeline/stages/analysis/runner.py`: after writing
  `units.parquet`, derive + write `well_summary.parquet` (one-row table for the
  current well; concat happens dashboard-side across wells).
- Manifest gets `tables.well_summary = "tables/well_summary.parquet"`.

**Tests:** mirror slice 2's fixtures, assert aggregate values.

**Acceptance:** smoke **A3** passes.

**Commit:** `claude: analysis-stage-and-dashboard, well_summary.parquet aggregates (slice 3)`

---

### Slice 4 — `axon-recon dashboard` CLI + minimal Dash app

**Goal:** top-level CLI that walks the runtime target scope, loads every
`<well>/analysis_outputs/manifest.json` under it, concatenates the parquet tables,
and serves a Dash app on `localhost:8050`.

**Files (new):**
- `src/axon_recon/dashboard/__init__.py`
- `src/axon_recon/dashboard/cli.py` — argparse parser: `--config <yaml>`,
  `--target-dataset N [N ...]`, `--limit-wells K`, `--limit-datasets K`,
  `--limit-wells-per-dataset K`, `--port P` (default 8050), `--host H` (default
  "127.0.0.1"), `--no-browser` (skip auto-open; mandatory in CI), `--debug`
  (Dash debug mode).
- `src/axon_recon/dashboard/discovery.py` — walks the runtime's target scope using
  the same scope helpers `axon-recon stages` uses; returns a `list[Path]` of
  manifest.json files. Read-only; tolerate missing manifests.
- `src/axon_recon/dashboard/data.py` — `load_all(manifests) -> dict[str,
  pandas.DataFrame]` keyed by table name. Concatenates per-well parquets, stamping
  manifest-derived identity columns onto every row (defense in depth, in case the
  parquet schema drifts).
- `src/axon_recon/dashboard/app.py` — `build_app(units_df, well_summary_df) ->
  dash.Dash`. Layout: left rail (filters), main pane (one histogram + a
  `dash_ag_grid` table view in this slice). Callbacks update on filter change.
- `src/axon_recon/dashboard/filters.py` — pure helpers that translate filter UI
  state → pandas masks. No Dash imports in this module (testable headless).
- `src/axon_recon/dashboard/tests/test_discovery.py`
- `src/axon_recon/dashboard/tests/test_data.py`
- `src/axon_recon/dashboard/tests/test_filters.py`

**Files (edited):**
- `pyproject.toml` or `setup.py` (whichever drives the entry points): add
  `axon-recon-dashboard = "axon_recon.dashboard.cli:main"` console script. Alias
  short-form `axon-recon dashboard` lives in `pipeline/cli.py` if there's a
  top-level dispatcher there (check `pipeline/cli.py`'s argparse structure first;
  if the existing dispatcher is `axon-recon stages …`, add a sibling
  `axon-recon dashboard …`).
- `environment.yml`: add `plotly`, `dash`, `dash-ag-grid`, `statsmodels`.
- `containers/axon-recon/Dockerfile`: bump runtime spec.

**Dashboard UX (slice 4 only — slices 5/6 add more):**

- Left rail (always visible):
  - "Project" multi-select (populated from `units.project.unique()`)
  - "Chip" multi-select
  - "Well" multi-select
  - "DIV" range slider (continuous numeric)
  - "Genotype" multi-select
  - "Media" multi-select
  - "Plating density" multi-select
  - "Scan type" multi-select
  - "Bombcell label allowlist" multi-select (good / non_soma_good / mua / noise /
    None) — default `{good, non_soma_good}`.
  - "Min num_spikes" numeric (default 0)
  - "Min num_branches" numeric (default 0)
  - "Min recon_quality_score" numeric (blank = no filter)
  - "Require recon_status=='ok'" checkbox (default on)
- Main pane:
  - Histogram with axis dropdown (any numeric column from `units.parquet`) and
    `color` dropdown (any categorical column).
  - Below: `dash_ag_grid.AgGrid` view of the filtered table.

**Tests:**
- `test_discovery.py`: synthetic scratch tree → asserts the correct manifests are
  found respecting `--target-dataset` / `--limit-wells`.
- `test_data.py`: concat handles missing tables, dtype consistency.
- `test_filters.py`: each filter knob produces the expected pandas mask.
- `test_app.py` (light): build_app returns a dash.Dash whose `app.layout` has the
  expected component ids. No browser test.

**Acceptance:**
- Smoke **A4** passes: dashboard starts, replies on `/_dash-layout`, exits cleanly.
- `conda run -n axon_recon python -m pytest src/axon_recon/dashboard/ -q` green.

**Commit:** `claude: analysis-stage-and-dashboard, dashboard CLI + minimal Dash app (slice 4)`

---

### Slice 5 — Box plot + significance brackets

**Goal:** add a box plot to the main pane with optional significance markers
(`*`/`**`/`***`) computed from pairwise tests with configurable correction.

**Files (new):**
- `src/axon_recon/dashboard/significance.py`:
  - `compute_pairwise_pvalues(df, group_col, value_col, test) -> dict[tuple[str,
    str], float]`. Tests supported: `mann_whitney`, `welch_t`, `tukey_hsd`,
    `kruskal_wallis` (omnibus; for omnibus, return single p-value at the figure
    level instead of pairwise).
  - `apply_correction(pvalues, method) -> dict`. Methods: `none`, `bonferroni`,
    `holm`, `bh` (Benjamini-Hochberg FDR). Uses `statsmodels.stats.multitest`.
  - `significance_brackets(fig, pvalues_corrected, thresholds=(0.05, 0.01, 0.001))
    -> fig`. Adds Plotly `shapes` (bracket lines) + `annotations` (asterisk text)
    at staggered y-offsets above the box plot.

**Files (edited):**
- `src/axon_recon/dashboard/app.py`: add a "Box plot" tab in the main pane. UI
  controls: `value_col` (numeric), `group_col` (categorical), `color` (categorical
  or none), `test` (radio), `correction` (radio), `show_significance` (toggle).
- `src/axon_recon/dashboard/tests/test_significance.py`: synthetic numpy arrays
  with known-significant + known-null groups; assert correct p-value ordering;
  correction sanity checks (Bonferroni-corrected p > raw p; BH preserves
  ordering).

**Acceptance:**
- Smoke **A5** passes.
- Visual sanity (manual, capture screenshot to commit notes): a 2-group box plot
  with known-different groups renders with `***`.

**Commit:** `claude: analysis-stage-and-dashboard, box plot + significance brackets (slice 5)`

---

### Slice 6 — Scatter, facet, export

**Goal:** add scatter plot with facet support and per-figure download buttons
(PDF/SVG/PNG/CSV). This finishes the MVP dashboard UX.

**Files (edited):**
- `src/axon_recon/dashboard/app.py`: add "Scatter" tab with `x`, `y`, `color`,
  `facet_col`, `facet_row` dropdowns. Add a "Download" button group on every plot
  (PDF, SVG, PNG via `kaleido`; CSV of the filtered table; JSON of the current
  filter+plot spec for provenance).
- `environment.yml`: add `kaleido` (Plotly's static-image exporter).
- `containers/axon-recon/Dockerfile`: bump runtime spec.

**Tests:**
- `test_app.py`: each download endpoint returns the expected MIME type with
  non-zero content.
- `test_filters.py`: filter+plot spec round-trips through the JSON serializer.

**Acceptance:**
- All previous smokes still pass.
- Manual visual sanity (commit notes screenshot): scatter with `color=genotype,
  facet_col=DIV` renders.

**Commit:** `claude: analysis-stage-and-dashboard, scatter + facet + export buttons (slice 6)`

---

## 6. Cleanup Checklist (post-slice 6)

Each command should return zero hits / clean exit.

```bash
# No legacy pre-MVP analysis scaffolding left behind
git grep -nE "qc_pass\s*:|qc_pass\s*=" src/axon_recon/pipeline/stages/analysis/

# No accidental GTR-pickle reads at MVP
git grep -n "gtr\.pkl\|GraphTracker\|pickle.load" src/axon_recon/pipeline/stages/analysis/

# Spikesort untouched
git diff dev_branch2... -- src/axon_recon/pipeline/stages/spikesort/ | wc -l   # expect 0

# Dashboard tests fully isolated from analysis stage
git grep -n "from axon_recon.pipeline.stages.analysis" src/axon_recon/dashboard/

# No hardcoded localhost:8050 outside dashboard/cli.py default
git grep -n "localhost:8050\|127.0.0.1:8050" src/axon_recon/

# Plotly / Dash deps recorded in BOTH environment.yml AND Dockerfile
grep -E "plotly|dash|pyarrow|statsmodels|kaleido" environment.yml
grep -E "plotly|dash|pyarrow|statsmodels|kaleido" containers/axon-recon/Dockerfile
```

---

## 7. Risks And Out-Of-Scope

**Risks:**

1. The data config's `attributes` block uses different keys than `unit_node_templates.parquet`-style metadata.
   Mitigation: slice 1 reads `attributes` literally and stamps them as `well_attributes.*` columns in the
   manifest; slice 2 promotes the keys the dashboard needs (`genotype`, `media`, `plating_density`) to
   top-level columns on `units.parquet`. Unknown attribute keys travel through unchanged.

2. `bombcell_label` reads depend on the spikesort stage having completed (`cluster_group.tsv` exists). If a
   user runs `stages analysis` before spikesort finishes, slice 2's metric writer must tolerate the missing
   file (write `bombcell_label = None`). The dashboard's filter then treats None as "include" or "exclude"
   per the user's toggle.

3. `pyarrow` already ships with the conda `pandas` install on most Linux setups, but Dockerfile ships pip-only.
   Slice 2 must add `pyarrow` to the Dockerfile's RUNTIME_SPEC explicitly (do not rely on transitive deps).

4. The running container the user launched might conflict with our container builds if we rebuild the image
   while the user's instance is running. Slices 2/4/6 (which bump environment.yml + Dockerfile) MUST commit the
   spec change without triggering a rebuild — the user will rebuild on their next launch.

**Out of scope:**

- Spikesort-derived metrics, preprocess-derived metrics (future slice plan).
- GTR-pickle metrics (future slice plan once a metric truly needs the graph object).
- Multi-table relational schema beyond `units` + `well_summary` (the GPT-suggested
  `branches`, `branch_nodes`, `unit_node_templates`, `node_summary` tables are deferred).
- Dashboard auth / multi-user / cloud hosting.
- Dashboard's full-report PDF export (per-figure downloads only at MVP).

---

## 8. Definition Of Done

1. Slices 1–6 each committed in order, every commit's tests green at HEAD of that commit.
2. `axon-recon stages analysis --config debug/debug.runtime.yml --target-dataset 11 --limit-wells 1`
   succeeds end-to-end and writes:
   - `<well>/analysis_outputs/manifest.json` with all identity fields populated.
   - `<well>/analysis_outputs/tables/units.parquet` with one row per `recon_status=="ok"` unit and the
     four starter metrics non-null.
   - `<well>/analysis_outputs/tables/well_summary.parquet` with a single-row aggregate.
3. `axon-recon dashboard --config debug/debug.runtime.yml --port 8051 --no-browser` boots, serves
   `/_dash-layout`, and the filter rail + 3 plot tabs (histogram, box, scatter) all render against the
   dataset-11 well000 artifact.
4. `pytest src/axon_recon/pipeline/stages/analysis/ src/axon_recon/dashboard/ -q` green.
5. `pytest src/axon_recon/pipeline/ -q --ignore=src/axon_recon/pipeline/tests/test_progress.py` shows the
   slice-0 baseline failures or a strict subset (no new failures introduced by this plan).
6. §6 cleanup checklist passes.
7. `debug/agent_guardrails_commit_notes.md` has an entry titled
   `ANALYSIS STAGE + DASHBOARD COMPLETE` summarizing test counts and any deviations from the slice plan.

When 1–7 hold, merge the branch back to `dev_branch2`.

---

## 9. Operating Contract (if running this under /loop)

See `debug/analysis_stage_and_dashboard_loop_prompt.md` for the autonomous-loop prompt.
Same conventions as the prior loops:

- Plan file: `debug/analysis_stage_and_dashboard_plan.md` (this file).
- Number of slices: 6.
- Branch: `analysis-stage-and-dashboard` (off `dev_branch2`).
- Halt condition: §8 DoD satisfied + `ANALYSIS STAGE + DASHBOARD COMPLETE` notes entry.
- Stop signals, failure handling, commit prefix, conda env, mutation-safety check rules unchanged.

Commit prefix on every slice: `claude: analysis-stage-and-dashboard, <slice description> (slice N)`.
