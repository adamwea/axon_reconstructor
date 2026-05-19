"""Shared plot abstraction for the dashboard.

Slice 4 of `dashboard_ui_refinement_plan.md`: pulls common plot
infrastructure into one module so the three plot builders in `app.py`
(`_build_histogram`, `build_box_plot`, `build_scatter`) share a
uniform config surface.

This slice scaffolds the abstraction without migrating the existing
builders. Subsequent slices 6 (box↔bar toggle) and 7 (tertiary
grouping) consume `PlotConfig` to thread their new features through
one place instead of three. The existing `_build_histogram` /
`build_box_plot` / `build_scatter` continue to work unchanged; their
migration is incremental.

Exports:
- `PlotConfig`: dataclass capturing the union of options across plot
  types. Each plot-type builder reads only the fields it cares about.
- `PlotMode`: enum-like literal type tagging the render mode (box vs
  bar etc) — slice 6 expands this.
- `apply_filters_to_dataframe(df, config) -> DataFrame`: shared
  filter application so each plot type doesn't reimplement the
  filter→DataFrame pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# Slice 4: literal-string render mode. Slice 6 will add "bar" for the
# box↔bar toggle.
PlotMode = Literal["histogram", "box", "scatter", "bar"]


@dataclass(frozen=True)
class PlotConfig:
	"""Unified config for every plot type the dashboard renders.

	Each plot builder reads only the fields it cares about; extra
	fields are ignored. This lets slices 5-7 add features without
	rippling new parameters through three separate function signatures.

	Field groups:
	- **identity**: which plot type to render (``mode``) and which
	  table to pull from (``data_source``).
	- **axes**: x_column, y_column (when applicable).
	- **grouping**: primary group (X for box/bar), secondary
	  (color/within-group), tertiary (faceting axis — slice 7).
	- **transforms**: per-axis log10 toggle.
	- **facets**: small-multiple split columns (slice 5 added these to
	  histogram + scatter).
	- **significance**: box-plot-only stat-test config.
	- **render**: visual tweaks (jitter, points mode, opacity, …).
	"""

	# Identity.
	mode: PlotMode = "histogram"
	data_source: Literal["units", "well_summary"] = "units"

	# Axes.
	x_column: str | None = None
	y_column: str | None = None
	color_column: str | None = None

	# Grouping.
	group_column: str | None = None  # primary (box/bar X-axis)
	secondary_group_column: str | None = None  # alias for color_column in box plots
	tertiary_group_column: str | None = None  # slice 7 — facet/3rd axis

	# Faceting.
	facet_col: str | None = None
	facet_row: str | None = None

	# Transforms.
	log_x: bool = False
	log_y: bool = False
	log_transform: bool = False  # box-plot single-axis log (uses value_col)

	# Significance overlays (box plot).
	significance_test: str = "mann_whitney"
	significance_correction: str = "none"
	show_significance: bool = True
	bracket_y_offset_frac: float = 0.10
	bracket_step_frac: float = 0.08

	# Render tweaks.
	points_mode: str = "off"
	exclude_nulls: bool = False
	jitter: bool = False
	opacity: float = 0.7
	boxgap: float = 0.3
	boxgroupgap: float = 0.3
	point_size: float = 6.0
	point_opacity: float = 0.6

	# Bar-mode (slice 6 future): aggregate + error-bar config.
	bar_aggregate: Literal["mean", "median"] = "mean"
	bar_error: Literal["std", "sem", "ci95", "none"] = "std"

	# Export.
	export_format: Literal["png", "svg", "pdf", "none"] = "none"


def apply_filters_to_dataframe(df: Any, config: PlotConfig) -> Any:
	"""Apply config-encoded filters to the DataFrame.

	Slice 4 scaffold: this function currently no-ops (returns the df
	unchanged). The dashboard's filter callback layer in `app.py`
	already applies filters before passing the DataFrame to the plot
	builders, so this helper is a future-facing seam for when filter
	application moves into the shared abstraction.

	Returning the df unchanged lets slice 4 ship without disrupting
	the existing app.py filter wiring.
	"""

	_ = config  # reserved for future filter encoding
	return df


__all__ = [
	"PlotConfig",
	"PlotMode",
	"apply_filters_to_dataframe",
]
