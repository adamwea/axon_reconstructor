# Dashboard UI refinement plan

> **Status (2026-05-19)**: 8 of 9 slices SHIPPED. Only slice 7
> (tertiary grouping — needs UX decision) remains.
>
> Shipped commits:
> - Slice 1 audit: `76333d3` (`dev/notes/refs/dashboard_audit.md`)
> - Slice 2 empty-state UX: `5ee230b`
> - Slice 3 discover_available: `0682fc0`
> - Slice 4 PlotConfig scaffold: `3fe87cb`
>   (`src/axon_recon/dashboard/plots/__init__.py`)
> - Slice 5 feature parity backfill: `8c72abf`
> - Slice 6 box↔bar toggle (render path): `fc3c45f`
>   (`build_bar_plot` + `aggregate_by_group`)
> - Slice 8 style parity (`dashboard/style.py`): `28e1ff3`
> - Slice 9 usage docs (partial): `74ea39a`
>   (`dev/notes/refs/dashboard_usage.md`)
>
> **NEXT** (slice 7): tertiary grouping. UX decision still open:
> faceted small-multiples vs nested hierarchical X-axis. User input
> needed before implementation can proceed.
>
> **UI wiring follow-up** (post-slice-6): app.py's Dash callback layer
> doesn't yet expose a `mode: box | bar` dropdown — the new
> `build_bar_plot` is callable but not yet user-toggleable. That
> callback wiring is the natural close-out commit alongside slice 7.

## Motivation

The dashboard (`src/axon_recon/dashboard/`, ~2128-line `app.py` + 6
sibling modules) has grown organically and shows pain points in real
use:

- **Empty metrics render as broken UI** rather than as "no data
  available" placeholders.
- **Some plates are missing** from the UI — suggests hardcoded plate
  lists somewhere instead of filesystem-driven discovery (despite
  having a `discovery.py` module that should own this).
- **Feature parity gap across plot types**: some features available on
  box plots aren't available on histograms or scatter plots.
- **No bar-plot view** of the same data box plots can render — user
  wants a toggle between box and bar modes.
- **No tertiary grouping** for box plots (only primary + secondary).
- **Styling drift** between plot types — colors, fonts, axis treatment
  not consistent across the dashboard.

User direction (2026-05-19):
- "I'm noticing many empty metrics, some plates missing, so I guess
  that's not dynamic."
- "Features on box plots not on histogram or scatter."
- "Box plots should be able to switched to bar plot mode."
- "Make more dynamic, modular in general."
- "Tertiary grouping would be a nice option in box plots as well."
- "Generally make sure all plot types have parity in data presentation
  and styling in general."

## Out of scope (v1)

- Migrating off Dash (no — Dash + dash-ag-grid stays).
- New scientific metrics (this plan is presentation, not new
  computations — though slice 1 may surface metrics that exist in code
  but never produce output, which is its own follow-up).
- Interactive editing of underlying recon data (read-only dashboard).
- Multi-user / auth (still single-user dev tool).

## Target shape

A modular dashboard where:

- **All data sources are filesystem-discovered** via `discovery.py` —
  no hardcoded plate / well / metric lists.
- **All plot types share a common rendering abstraction** — common
  features (filtering, grouping, color-by, faceting, log/linear axes,
  significance overlays, export) live in one place; per-plot-type
  modules only implement the specific render logic.
- **Empty-state UX is explicit** — empty metric → "no data" placeholder;
  missing plate / recording → faded entry with explanatory tooltip; no
  silent broken-UI states.
- **Box plot ↔ bar plot toggle** — UI control switches between
  per-point / per-distribution (box) and per-group-aggregate (bar) views
  on the same data.
- **Tertiary grouping** available for box / bar plots (third grouping
  dropdown). Optional: color-by-tertiary on scatter, facet-by-tertiary
  on histogram where it makes sense.
- **Style parity** — one palette module + one style config consumed by
  all plot types. Fonts, axis colors, legend behavior, gridlines,
  defaults aligned.

## Slices

### Slice 1 — Audit (no code changes)

Inventory the current dashboard. Output: a section in this plan +
optionally `dev/notes/refs/dashboard_audit.md`.

- Locate every plot type rendered (likely box, scatter, histogram;
  also any tables / heatmaps).
- For each plot type, list the features currently available (filters,
  grouping levels, color-by, export, significance overlays, axis
  options, etc.).
- Identify hardcoded data sources: grep for plate / well / metric
  literals, hardcoded paths, hardcoded lists in `data.py`, `filters.py`,
  `app.py`.
- Identify empty-state handling gaps — every place a missing /
  empty metric is silently rendered as empty UI.
- Identify styling drift: extract every place colors, fonts, gridlines,
  axis treatment are set; flag inconsistencies.
- Categorize each finding by: (a) presentation fix (this plan) vs.
  (b) pipeline-output gap (separate follow-up).

### Slice 2 — Empty-state UX

Graceful empty-state handling everywhere. One pass through the dashboard:

- Empty metric → "No data available" placeholder with brief tooltip
  ("metric `X` not present in any selected recording's output").
- Missing plate / recording → entry stays in UI but rendered as faded
  with a tooltip explaining why it's missing.
- Loading state → spinner or progress indicator while data is being
  resolved.
- Tests for each empty state.

### Slice 3 — Dynamic data source discovery

Replace every hardcoded plate / well / metric list with
filesystem-driven discovery via `discovery.py`. Where the discovery
module already does this, audit + extend to cover the gaps.

- One source of truth: `discovery.discover_available(<output_root>)`
  returns the populated set of (plate, recording, well, metric)
  tuples.
- All filter dropdowns and plot data sources read from this discovery
  result.
- Cache discovery output per request lifecycle.
- Tests: stub a fake filesystem tree; assert dropdowns populate
  correctly.

### Slice 4 — Shared plot abstraction

Pull common plot infrastructure into `dashboard/plots/__init__.py`
(or similar) as a shared module:

- `PlotConfig` dataclass: filters, grouping (primary / secondary /
  tertiary), color-by, log/linear axes, significance overlays, export
  format.
- Shared filter-application + grouping + facet logic.
- Per-plot-type modules (`plots/box.py`, `plots/scatter.py`,
  `plots/histogram.py`, …) implement only the specific render logic;
  they consume `PlotConfig` from the shared module.
- `app.py` thins out as logic moves into per-plot-type modules.

### Slice 5 — Feature parity audit + backfill

Produce a feature matrix: rows = features (filters, grouping levels,
color-by, log/linear, significance overlays, export formats, hover
tooltips, …), columns = plot types (box, scatter, histogram, bar
[after slice 6], …). Identify gaps from slice 1's audit.

Implement each gap as its own commit. Tests per backfill.

### Slice 6 — Box ↔ bar plot toggle

UI control on box plot widgets that switches the render mode between
box (per-distribution, IQR + whiskers + outliers) and bar (per-group
mean/median + error bars). Same `PlotConfig`; render branches on the
mode setting.

- Single shared `aggregate_by_group(...)` for the bar path.
- Default error-bar = std (configurable: std / sem / 95% CI).
- Tests: golden render of each mode on the same fixture data.

### Slice 7 — Tertiary grouping

Third grouping dropdown for box / bar plots. Primary = X-axis,
secondary = within-X grouping (already exists), tertiary = additional
faceting / sub-grouping axis. Decide UX during slice 7:

- Faceted small-multiples for tertiary (one plot per tertiary value).
- OR nested grouping on X-axis (primary × secondary × tertiary as
  hierarchical X labels).

For other plot types where tertiary makes sense:

- Scatter: color-by-tertiary (alongside existing color-by-secondary
  if present) — decide whether this requires a 2D color encoding or
  switches the color-by axis.
- Histogram: facet-by-tertiary as small-multiples.

### Slice 8 — Style parity

One palette module + one style config consumed by all plot types:

- `dashboard/style.py`: palettes (categorical, sequential, diverging),
  axis defaults, legend defaults, font stack, gridline behavior.
- All plot modules import from `style.py`; no inline color / font /
  style overrides.
- Match the cross-session color palette planned for slice 3 of
  `chip_layout_phase_split_plan.md` — share the palette module if it
  makes sense, so dashboard and recon plots align.

### Slice 9 — Tests + smoke + docs

- Golden tests for empty-state, dynamic discovery, feature parity
  matrix, style parity.
- Login-node smoke: launch dashboard on the analyzed_data reference
  cohort, verify no crashes + every feature works + the empty states
  trigger correctly (intentionally point at a recording with missing
  metrics to verify).
- README + dashboard docs update.

## Smoke tests

- After slice 3: dashboard launches against a fresh `analyzed_data/`
  tree and discovers all plates/wells/metrics without any source
  edits.
- After slice 5: every plot type accepts the same feature set per the
  feature matrix.
- After slice 9: full real-data smoke against the 80k DMEM cohort.

## Done criteria

- No hardcoded plate / well / metric lists anywhere in dashboard
  source.
- Empty metrics + missing plates / recordings render as graceful
  placeholders, not broken UI.
- Every plot type shares the common `PlotConfig` abstraction and
  feature surface (where the feature makes sense for that plot type).
- Box plot widgets have a working box ↔ bar toggle.
- Box / bar plots have a tertiary grouping option.
- Styling is consistent across plot types — palette, fonts, axes
  match.

## Open decisions

- **Tertiary grouping UX**: small-multiples vs. hierarchical X labels
  (slice 7).
- **Bar plot error-bar default**: std vs. sem vs. 95% CI (slice 6).
- **Color palette source**: shared with `chip_layout_phase_split_plan`
  slice 3 helper, or independent (slice 8).
- **Scatter tertiary encoding**: color-by-tertiary vs.
  size-by-tertiary vs. separate facet (slice 7).
- **Whether to split `app.py` (2128 lines)** as part of this plan or
  defer to a follow-up: probably defer — too much surface in one plan.
  But slice 4's plot abstraction will naturally shrink `app.py`
  meaningfully.

## Tier + sequencing

**Tier 4** — substantive user-facing UX work; not blocking science
work but user-stated priority. No dependency on the Era 3 integration
gate (the dashboard reads existing `analyzed_data/` outputs).

Can start any time the loop has bandwidth. Slice 1 (audit, no code
changes) is the natural first pick — it surfaces the feature matrix +
empty-state inventory + hardcoded-list inventory that the rest of the
plan executes against.

## Relationship to other plans

- **Independent of**: Era 3 integration work (kssynth slice 9 +
  unitmatch_phase 5). Dashboard reads `analyzed_data/` directly.
- **Palette synergy with**: `chip_layout_phase_split_plan.md` slice 3
  cross-session colors helper. Shared `style.py` module possibly the
  right host for both.
- **May surface follow-ups**: slice 1 may identify metrics that exist
  in dashboard code but never produce data — those are pipeline-output
  gaps, booked as separate tracker entries (not this plan's scope).
