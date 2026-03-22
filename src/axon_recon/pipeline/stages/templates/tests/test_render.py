from __future__ import annotations

from pathlib import Path

import numpy as np  # type: ignore[import-not-found]

from axon_recon.pipeline.stages.templates.core.render import render_propagation_plot
from axon_recon.pipeline.stages.templates.models.inputs import PropagationPlotConfig


def test_render_propagation_plot_respects_panel_chunk_knobs(tmp_path: Path) -> None:
	n_channels = 18
	n_samples = 80
	x = np.linspace(-1.0, 1.0, n_samples)
	t = np.vstack([
		np.sin((i + 2) * x) * (1.0 - (0.03 * i))
		for i in range(n_channels)
	]).astype(float)
	locs = np.column_stack([
		np.linspace(0.0, 170.0, n_channels),
		np.linspace(0.0, 50.0, n_channels),
	])

	png_path = tmp_path / "propagation.png"
	pdf_path = tmp_path / "propagation.pdf"

	outputs = render_propagation_plot(
		template=t,
		locations_xy=locs,
		config=PropagationPlotConfig(
			write_pdf=True,
			pdf_relpath="propagation.pdf",
			write_png=True,
			png_relpath="propagation.png",
			top_channels=16,
			channels_per_panel=5,
			channel_overlap=2,
			background="white",
			show_electrode_ids=True,
		),
		pdf_path=pdf_path,
		png_path=png_path,
	)

	assert png_path.exists()
	assert pdf_path.exists()
	assert outputs.get("propagation_plot_png") == str(png_path)
	assert outputs.get("propagation_plot_pdf") == str(pdf_path)
