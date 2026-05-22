# Dashboard usage — `axon-recon-dashboard`

User-facing docs for the Plotly Dash dashboard that visualizes the
analysis-stage `manifest.json` + per-table parquet artifacts. Slice 9
of `dashboard_ui_refinement_plan.md`.

## Quickstart

```bash
# Make sure compute_metrics has run for the wells you want to visualize.
axon-recon stages analysis.compute_metrics \
  --config dev/debug_NERSC/debug.runtime.yml \
  --target-wells well000 --limit-datasets 1

# Launch the dashboard.
axon-recon-dashboard --config dev/debug_NERSC/debug.runtime.yml \
  --limit-datasets 1 --limit-wells 5
```

The dashboard opens at `http://127.0.0.1:8050/` by default (loopback
only). Use `--lan` or `--host 0.0.0.0` to serve over the local network
(needed when SSH'd into a head node and connecting from a workstation).

## CLI flags

| Flag | Default | Purpose |
|---|---|---|
| `--config` | required | Path to runtime YAML/JSON config. |
| `--target-dataset`, `--target-datasets` | none (all) | 0-based dataset indices to load (`0 2 11`). |
| `--limit-wells` | unlimited | Cap total wells across all datasets. |
| `--limit-datasets` | unlimited | Cap number of datasets loaded. |
| `--limit-wells-per-dataset` | unlimited | Cap wells per dataset. |
| `--port` | 8050 | Bind port. |
| `--host` | 127.0.0.1 | Bind host. |
| `--lan` | off | Shortcut for `--host 0.0.0.0` + prints LAN URLs. |
| `--debug` | off | Dash debug toolbar. |

## Plot types

The dashboard's main pane has tabs for each plot type. All four follow
the same uniform style (`dashboard/style.py` slice 8 of
`dashboard_ui_refinement_plan.md`) — plotly_white template, system-ui
font, consistent grid/axis colors, slice-2 empty-state messages when
data is missing.

### Histogram

Distribution of one numeric metric, optionally faceted/colored by a
categorical dimension.

Controls:
- **X-axis**: any numeric metric in the loaded `units` table.
- **Color**: optional categorical column for stacking/grouping.
- **Log transform** (slice 5): drop non-positive rows of X then plot
  `log10(X)`. Useful for spike-count-like distributions.
- **Facet col / row** (slice 5): split into small-multiples by a
  categorical column.

### Box plot

Per-group distribution comparison with optional significance brackets.

Controls:
- **Value column**: numeric metric to plot.
- **Group column**: categorical column for primary X-axis grouping.
- **Color column**: optional secondary grouping (side-by-side boxes
  within each primary cluster).
- **Significance test**: Mann-Whitney U / Kruskal-Wallis / t-test.
- **Correction**: none / Bonferroni / BH-FDR / Holm.
- **Show significance**: toggle bracket overlay on/off.
- **Points mode**: off (outliers only) / jittered_side / over_box.
- **Log transform**: drop non-positive rows then plot `log10(value)`.
- **Data source**: units table vs. well_summary aggregates.

### Scatter

Two-numeric relationship plot with optional faceting + jitter.

Controls:
- **X / Y**: two numeric metrics.
- **Color**: optional categorical.
- **Facet col / row**: split into small-multiples.
- **Jitter**: add small Gaussian noise to numeric axes (helps when
  discrete-valued axes stack points).
- **Log X / Log Y** (slice 5): per-axis log10 transform (drops
  non-positive rows of that axis).

### Table

dash_ag_grid view of the filtered `units` DataFrame. CSV export available.

## Filters (left rail)

Modular inclusion/exclusion filters drive every plot AND the table.
Plotting/table contents always reflect the current filter state.

Filter dimensions populated dynamically from the loaded data:
- Identity: project, chip, well, scan_type, genotype, media,
  plating_density, treatment, DIV range.
- Quality: bombcell label allowlist, min num spikes / branches /
  recon quality score, require `recon_status == ok`.

When a column is absent from the loaded data (e.g. a manifest didn't
write `bombcell_label`), the corresponding filter row is hidden — no
silent "every row" pass-through.

## Empty-state UX (slice 2)

When no data matches the current filters / no rows in the selected
column / a required column is absent, the plot renders a centered
"No data available" annotation with a smaller sub-message explaining
WHY (e.g. `Column 'isolation_metric' is not present in any selected
recording's output`). Axes are hidden in this state — no confusing
blank gridlines with a `_` axis.

## Discovery snapshot (slice 3)

The `dashboard.discovery.discover_available(manifest_paths) → DataDiscovery`
helper aggregates the SUPERSET of tables + columns + numeric vs
categorical splits across loaded manifests. The dashboard's filter
dropdown population reads from this so:
- New tables added by future analysis-phase slices show up
  automatically.
- A column missing from some manifests but present in others still
  appears in the dropdown (with empty cells in the missing rows).

## Export

Per-plot:
- PNG / SVG / PDF download buttons under each plot tab.
- Filter spec JSON download (capture the current filter state for
  reproducible replay).

Per-table:
- CSV download of the filtered `units` DataFrame.

## Known gaps

Per the slice-1 audit (`dev/notes/refs/dashboard_audit.md`):
- **Slice 4 not yet shipped**: shared plot abstraction
  (`PlotConfig` dataclass + per-plot-type modules under
  `dashboard/plots/`) is still pending. Until then, app.py keeps the
  plot builders inline.
- **Slice 6 (box ↔ bar toggle)**: needs slice 4's PlotConfig first.
- **Slice 7 (tertiary grouping)**: needs a UX decision (facet
  small-multiples vs nested hierarchical X-axis).
- **Login-node smoke**: not yet automated; user-domain when first
  needed.

## See also

- Plan: `dev/notes/plans/completed/dashboard_ui_refinement_plan.md`
- Slice-1 audit: `dev/notes/refs/dashboard_audit.md`
- Style module: `src/axon_recon/dashboard/style.py`
- Discovery API: `src/axon_recon/dashboard/discovery.py`
