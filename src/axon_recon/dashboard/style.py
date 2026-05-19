"""Dashboard style module — single source of truth for palettes, axis
defaults, fonts, and gridline behavior consumed by every plot builder.

Slice 8 of `dashboard_ui_refinement_plan.md`. Plot builders should never
declare inline colors / fonts / templates / paper backgrounds; they
either pass a column name through to plotly and let it use these
defaults, OR call `apply_dashboard_style(fig)` to apply the uniform
overlay.

Cross-references:
- `chip_layout_phase_split_plan.md` slice 3 will eventually share a
  cross-session color palette with this module so dashboard + recon
  plots align. For now, this module holds dashboard-only defaults.
"""

from __future__ import annotations

from typing import Any

# Categorical palette — Plotly's "Plotly" qualitative palette extended
# with a few visually-distinct fallback colors. Long enough to handle
# the typical dashboard's grouping dimensions (10 wells, ~5 media
# conditions, etc) without recycling.
CATEGORICAL_PALETTE: tuple[str, ...] = (
	"#636EFA",  # blue
	"#EF553B",  # red
	"#00CC96",  # green
	"#AB63FA",  # purple
	"#FFA15A",  # orange
	"#19D3F3",  # cyan
	"#FF6692",  # pink
	"#B6E880",  # lime
	"#FF97FF",  # magenta
	"#FECB52",  # yellow
	"#1F77B4",  # darker blue (fallback)
	"#FF7F0E",  # darker orange (fallback)
)

# Sequential palette — perceptually uniform; used for ordered group
# rendering (e.g. DIV order, plating density).
SEQUENTIAL_PALETTE: tuple[str, ...] = (
	"#FFF7BC",
	"#FEE391",
	"#FEC44F",
	"#FE9929",
	"#EC7014",
	"#CC4C02",
	"#993404",
	"#662506",
)

# Diverging palette — symmetric around zero; used for differential
# metrics (e.g. effect-size visualizations).
DIVERGING_PALETTE: tuple[str, ...] = (
	"#5E3C99",
	"#B2ABD2",
	"#F7F7F7",
	"#FDB863",
	"#E66101",
)

# Plot template name — Plotly's built-in. "plotly_white" gives a clean
# white background; gridlines stay light enough not to fight the data.
TEMPLATE = "plotly_white"

# Default font stack. system-ui keeps things native on every OS;
# Helvetica/Arial fallbacks for Plotly's SVG export.
FONT_FAMILY = "system-ui, -apple-system, 'Helvetica Neue', Helvetica, Arial, sans-serif"
FONT_SIZE_BASE = 12
FONT_SIZE_TITLE = 16

# Gridline / axis defaults.
GRID_COLOR = "#E5E5E5"
AXIS_LINE_COLOR = "#999999"
ZEROLINE_COLOR = "#CCCCCC"

# Margin defaults — slim so the figure uses its area for data, not
# whitespace.
MARGIN_LEFT = 60
MARGIN_RIGHT = 30
MARGIN_TOP = 40
MARGIN_BOTTOM = 60


def apply_dashboard_style(fig: Any) -> Any:
	"""Apply the dashboard's uniform style to a Plotly figure in place
	and return it.

	Idempotent: re-applying yields the same result. Each plot builder
	calls this on its final figure (after the px.<plot> call) so
	downstream tweaks (annotations, traces) survive.

	Doesn't touch the figure's color sequence — that's set by passing
	``color_discrete_sequence=CATEGORICAL_PALETTE`` at the px.<plot>
	call site. Callers can also pass ``apply_axis_lines=False`` to skip
	the axis-line overlay for empty-state figures whose axes are
	already hidden.
	"""

	fig.update_layout(
		template=TEMPLATE,
		font={"family": FONT_FAMILY, "size": FONT_SIZE_BASE},
		title_font={"family": FONT_FAMILY, "size": FONT_SIZE_TITLE},
		margin={
			"l": MARGIN_LEFT,
			"r": MARGIN_RIGHT,
			"t": MARGIN_TOP,
			"b": MARGIN_BOTTOM,
		},
	)
	# Axis lines + gridlines applied via update_xaxes / update_yaxes so
	# the settings propagate to every subplot in faceted figures.
	fig.update_xaxes(
		gridcolor=GRID_COLOR,
		linecolor=AXIS_LINE_COLOR,
		zerolinecolor=ZEROLINE_COLOR,
	)
	fig.update_yaxes(
		gridcolor=GRID_COLOR,
		linecolor=AXIS_LINE_COLOR,
		zerolinecolor=ZEROLINE_COLOR,
	)
	return fig


__all__ = [
	"CATEGORICAL_PALETTE",
	"SEQUENTIAL_PALETTE",
	"DIVERGING_PALETTE",
	"TEMPLATE",
	"FONT_FAMILY",
	"FONT_SIZE_BASE",
	"FONT_SIZE_TITLE",
	"GRID_COLOR",
	"AXIS_LINE_COLOR",
	"ZEROLINE_COLOR",
	"apply_dashboard_style",
]
