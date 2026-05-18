from __future__ import annotations

from typing import Any


def propagation_outputs_requested(propagation_plots: Any) -> bool:
	return any(
		bool(getattr(propagation_plots, field_name, False))
		for field_name in (
			"write_pdf",
			"write_png",
			"write_svg",
			"write_circles_template_numbered_png",
			"write_circles_template_numbered_svg",
			"write_propagation_2panel_png",
			"write_propagation_2panel_svg",
		)
	)
