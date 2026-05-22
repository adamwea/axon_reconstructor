# Dashboard UI audit — dashboard_ui_refinement_plan slice 1 (HISTORICAL)

> **Status (post-2026-05-21)**: the parent plan SHIPPED all 9 slices and moved to `plans/completed/dashboard_ui_refinement_plan.md`. This audit doc is kept as a historical snapshot of the dashboard's pre-refinement state — useful when comparing pre/post-refinement behavior or planning v2 work. Original "env_install_unification" title was a misnomer; the audit was always for the dashboard plan's slice 1.

Inventory of `src/axon_recon/dashboard/` as of 2026-05-19 for slice 1
of `dashboard_ui_refinement_plan.md` (now at `plans/completed/`).
Maps original state to the plan's nine refinement slices so slices 2-9 could act surgically.

## Module map

| File | LoC | Role |
|---|---:|---|
| `__init__.py` | 7 | Package init |
| `app.py` | 2128 | Dash app builder, layout, callbacks, plot builders |
| `cli.py` | 285 | `axon-recon-dashboard` CLI entry point |
| `data.py` | 100 | Manifest → DataFrame loader (`load_all`) |
| `discovery.py` | 90 | Walks runtime targets → `manifest.json` paths |
| `filters.py` | 176 | Pure-pandas filter helpers (mask builders) |
| `significance.py` | 455 | Stat-test + bracket overlay logic |
| `tests/` | — | Component-level tests for each module |

Total: ~3241 LoC, dominated by `app.py`. The `data.py` / `discovery.py` /
`filters.py` / `significance.py` are already modular; `app.py` is where
the UI / layout / styling lives and where most slices 2-9 will land.

## Plot type inventory

Three plot types currently rendered, all via `plotly.express`:

1. **Histogram** — `_build_histogram(df, x_column, color_column)` at
   `app.py:1811`. Empty-state fallback: `px.histogram(pd.DataFrame({"_": []}), x="_")`.
2. **Box plot** — `build_box_plot(...)` at `app.py:1835`. Optional
   significance brackets via `significance.py`. Three empty-state
   fallbacks at `app.py:1896`, `1900`, `1904`, `1912`. `boxgap` /
   `boxgroupgap` is the only `update_layout` call in the file.
3. **Scatter** — `build_scatter(...)` at `app.py:2064`. Empty-state
   fallback at `app.py:2086`, `2090`. Supports facet_col / facet_row /
   jitter / opacity=0.7.

Plus one **dash_ag_grid table view** of the filtered units DataFrame
(rendered via `dash_ag_grid as dag` — see `ID_UNITS_TABLE`).

Plan slice 6 (Box ↔ bar plot toggle) implies a fourth plot type
("bar") will be added. Plan slice 4 (shared plot abstraction) implies
collapsing the four `px.*` calls behind a common abstraction.

## Feature parity table (slice 5 scope)

| Feature | Histogram | Box | Scatter | Table |
|---|:---:|:---:|:---:|:---:|
| X-axis dropdown | ✅ (`ID_HIST_X_AXIS`) | ✅ (`ID_BOX_VALUE_COL`) | ✅ (`ID_SCATTER_X`) | n/a |
| Y-axis dropdown | n/a | ✅ (implied via value_col) | ✅ (`ID_SCATTER_Y`) | n/a |
| Color-by dropdown | ✅ (`ID_HIST_COLOR`) | ✅ (`ID_BOX_COLOR`) | ✅ (`ID_SCATTER_COLOR`) | n/a |
| Group-by dropdown | n/a | ✅ (`ID_BOX_GROUP_COL`) | n/a (via facet) | n/a |
| Facet col / row | ❌ | ❌ | ✅ (`ID_SCATTER_FACET_*`) | n/a |
| Log transform | ❌ | ✅ (`ID_BOX_LOG_TRANSFORM`) | ❌ | n/a |
| Points mode | n/a | ✅ (`ID_BOX_POINTS_MODE`) | ❌ | n/a |
| Jitter | n/a | n/a | ✅ (`ID_SCATTER_JITTER`) | n/a |
| Significance brackets | n/a | ✅ (`ID_BOX_SHOW_SIGNIFICANCE` + ttest/correction + bracket offset/step) | n/a | n/a |
| Data source toggle | n/a | ✅ (`ID_BOX_DATA_SOURCE`) | n/a | n/a |
| PNG / SVG / PDF export | ✅ (3) | ✅ (3) | ✅ (3) | ❌ |
| CSV export | n/a | n/a | n/a | ✅ (`ID_DOWNLOAD_CSV`) |
| Spec JSON export | ✅ (filter spec, all plots share) | ✅ | ✅ | ✅ |

**Slice 5 gaps to backfill** (one-direction parity asks):
- Histogram: no log transform; no facet.
- Scatter: no log transform; no points mode (since scatter is already
  one-point-per-row).
- All plots: no boxgap/boxgroupgap or equivalent uniform-styling
  control surface.

## Hardcoded data source inventory (slice 3 scope)

✅ **Already dynamic** (no hardcoded literals to migrate):
- `discovery.py:iter_manifest_paths` walks `select_execution_targets`
  results — same well set the analysis stage would compute. No
  hardcoded plates/wells.
- `data.py:_IDENTITY_COLUMNS_FROM_MANIFEST` is a fixed tuple of column
  names, NOT data — that's schema, not values.
- `filters.py:filter_*` operates on columns by name; no value lists
  hardcoded.

⚠️ **Implicit hardcoding** (worth a slice 3 follow-up):
- `data.py:99` ensures `units` + `well_summary` table names exist as
  empty DataFrames if missing from any manifest. That's a hardcoded
  pair of expected table names; the dashboard would silently miss any
  new table type added in a future analysis-phase slice.
- `app.py` references `units_df` and `well_summary_df` directly via
  `build_app(units_df, well_summary_df)` — these are the only two data
  surfaces the UI knows about.
- Identity columns enumerated in `app.py:35-46` (`ID_FILTER_*` for
  project/chip/well/scan_type/genotype/media/plating/treatment) match
  `_IDENTITY_COLUMNS_FROM_MANIFEST` plus the well-attribute fields.
  Adding a new identity dimension (e.g. shaker speed) requires a
  matching `ID_FILTER_*` constant + filter callback.

Slice 3's "one source of truth" recommendation maps cleanly: add a
`discovery.discover_available_metrics(<output_root>)` (or extend
`load_all` to surface column-set metadata) so the UI can build filter
dropdowns from data without static lists.

## Empty-state handling (slice 2 scope)

Current pattern, repeated across the three plot builders:
```python
if df is None or df.empty:
    return px.<plot>(pd.DataFrame({"_": []}), x="_", ...)
```

What that renders: an empty plotly figure with the X-axis literally
labeled `_`. No "no data" message, no tooltip explaining why, no
loading-state visual.

Slice 2's plan §86-96 calls for:
- "No data available" placeholder with tooltip ("metric `X` not present
  in any selected recording's output").
- Missing-plate / missing-recording entries faded in UI with tooltip.
- Loading-state spinner.

Action items for slice 2:
1. Wrap `px.<plot>(...)` empty-state returns in a small helper that
   renders an annotation rather than an empty `_` axis.
2. Surface the metric / column name in the tooltip so users know why
   the plot is empty.
3. Add `dcc.Loading` wrappers around the graph components (cheap; just
   layout changes around the existing graphs).

## Styling drift (slice 8 scope)

Plotly defaults applied throughout — no global template / theme call.
Per-plot tweaks limited to:
- `update_layout(boxgap=..., boxgroupgap=...)` once at `app.py:1948`
  (box plot only).
- `barmode="overlay" if color else "relative"` on histogram.
- `opacity=0.7` on scatter.
- Empty-state fallbacks use the default plotly template.

Slice 8 ("style parity") would add a uniform call site: every plot
runs through a `_apply_dashboard_style(fig)` that sets `template=...`,
font, gridline color, color-discrete-map, hover background, etc.

## Slice → action mapping summary

| Slice | Scope ready? | Notes |
|---|:---:|---|
| 1 — Audit | ✅ DONE (this doc) | — |
| 2 — Empty-state UX | ✅ Ready | Targets the 6+ `px.<plot>(pd.DataFrame({"_": []}))` call sites + add dcc.Loading |
| 3 — Dynamic data discovery | ⚠️ Mostly already done | Extend discovery to surface available-metrics; add new-table-type handling |
| 4 — Shared plot abstraction | ✅ Ready | Collapse 3 px.<plot> calls behind a common helper that takes plot_type + axis/color spec |
| 5 — Feature parity backfill | ✅ Ready | Add log transform to histogram + scatter; facet to histogram |
| 6 — Box ↔ bar toggle | ✅ Ready | New plot_type + dropdown; reuses slice 4's abstraction |
| 7 — Tertiary grouping | ✅ Ready | Add a third group-by dropdown; mostly an `ID_BOX_*` + callback addition |
| 8 — Style parity | ✅ Ready | Single `_apply_dashboard_style(fig)` applied uniformly |
| 9 — Tests + smoke + docs | ✅ Ready | Existing test files at `tests/test_*.py` cover the modules |

## Out-of-scope follow-ups (NOT this plan)

Things noticed during the audit that belong elsewhere:

1. **No table CSV export of well_summary** — only the units table has
   `ID_DOWNLOAD_CSV`. If well_summary becomes user-relevant, mirror
   the export.
2. **No multi-plot side-by-side / dashboard-of-dashboards view** —
   one main tab per plot type. If the user wants comparative
   side-by-side rendering, that's a layout-level slice not in this
   plan.
3. **No URL-state encoding** — refreshing the page resets all
   selections. Could be a separate UX slice if the user wants
   shareable links.

These are user-domain decisions; flagging here so they're tracked.
