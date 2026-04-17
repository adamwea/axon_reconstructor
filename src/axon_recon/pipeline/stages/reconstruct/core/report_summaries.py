from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from ..models.inputs import ReconstructionInputs
from ..models.results import UnitReconstructionResult


def _existing_slide(*, title: str, image_path: Any) -> dict[str, str] | None:
	if image_path is None:
		return None
	path = Path(str(image_path))
	if not path.exists():
		return None
	return {"title": str(title), "image_path": str(path)}


def write_reconstruct_summary_slides_pdf(
	*,
	slides: list[dict[str, str]],
	pdf_path: Path,
) -> dict[str, str]:
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]
	from matplotlib.backends.backend_pdf import PdfPages  # type: ignore[import-not-found]

	page_width = 13.333
	page_height = 7.5
	margin_x = 0.03
	margin_y = 0.05
	image_y0 = 0.08
	image_height = 0.82
	image_width = 1.0 - (2.0 * margin_x)
	pdf_path.parent.mkdir(parents=True, exist_ok=True)
	with PdfPages(pdf_path) as pdf:
		for slide in slides:
			image_path = Path(str(slide["image_path"]))
			image = plt.imread(image_path)
			image_height_px = max(1, int(image.shape[0]))
			image_width_px = max(1, int(image.shape[1]))
			image_aspect = float(image_width_px) / float(image_height_px)
			available_aspect = float(image_width * page_width) / float(image_height * page_height)
			if image_aspect >= available_aspect:
				axes_width = image_width
				axes_height = float((image_width * page_width) / (image_aspect * page_height))
				axes_x0 = margin_x
				axes_y0 = image_y0 + float((image_height - axes_height) * 0.5)
			else:
				axes_height = image_height
				axes_width = float((image_height * page_height * image_aspect) / page_width)
				axes_x0 = margin_x + float((image_width - axes_width) * 0.5)
				axes_y0 = image_y0

			fig = plt.figure(figsize=(page_width, page_height), facecolor="black")
			fig.text(
				0.5,
				0.965,
				str(slide["title"]),
				color="white",
				fontsize=20,
				horizontalalignment="center",
				verticalalignment="top",
			)
			ax = fig.add_axes([axes_x0, axes_y0, axes_width, axes_height])
			ax.imshow(image)
			ax.axis("off")
			pdf.savefig(fig, facecolor=fig.get_facecolor())
			plt.close(fig)
	return {"report_summaries_pdf": str(pdf_path)}


def run_report_summaries_phase(
	*,
	inputs: ReconstructionInputs,
	reconstruction_out_dir: Path,
	unit_results: list[UnitReconstructionResult],
	unit_results_for_reports: list[UnitReconstructionResult],
	preserve_stage_reports: bool,
	existing_stage_outputs: dict[str, str],
	resolve_report_output_paths_fn: Callable[..., dict[str, Path]],
	write_reconstruct_summary_slides_pdf_fn: Callable[..., dict[str, str]],
	logger: logging.Logger | None = None,
) -> dict[str, str]:
	active_logger = logger or logging.getLogger("axon_recon.reconstruct.report_summaries")
	phase_cfg = inputs.phases.report_summaries
	stage_outputs: dict[str, str] = dict(existing_stage_outputs)
	if not bool(getattr(phase_cfg, "write_pdf", True)):
		return stage_outputs
	if bool(preserve_stage_reports):
		return stage_outputs

	report_paths = resolve_report_output_paths_fn(
		reconstruction_out_dir=reconstruction_out_dir,
		reports=inputs.reports,
		report_recons_phase=inputs.phases.report_recons,
		report_full_chip_layout_phase=inputs.phases.report_full_chip_layout,
		report_summaries_phase=phase_cfg,
	)
	slides: list[dict[str, str]] = []
	for title, candidate in (
		(str(getattr(inputs.phases.report_full_chip_layout.display, "title", "Full-chip reconstructed branch layout")), report_paths.get("full_chip_layout_png")),
		("Reconstruct circle recon grid", report_paths.get("circle_recon_grid_png")),
	):
		slide = _existing_slide(title=title, image_path=candidate)
		if slide is not None:
			slides.append(slide)

	missing_unit_summaries: list[Any] = []
	unit_summary_slides = 0
	for item in unit_results_for_reports:
		image_path = item.outputs.get("plot_unit_summary_png") if isinstance(item.outputs, dict) else None
		slide = _existing_slide(title=f"Unit {item.unit_id} summary", image_path=image_path)
		if slide is None:
			missing_unit_summaries.append(item.unit_id)
			continue
		slides.append(slide)
		unit_summary_slides += 1

	if not slides:
		raise FileNotFoundError(
			"Missing unit summary images required for reconstruct.report_summaries PDF slide deck; run reconstruct.plot_unit_summary first"
		)

	active_logger.info(
		"Reconstruct summary slide deck inputs=%d unit_summary_slides=%d missing_unit_summaries=%d",
		len(slides),
		unit_summary_slides,
		len(missing_unit_summaries),
	)
	stage_outputs.update(
		write_reconstruct_summary_slides_pdf_fn(
			slides=slides,
			pdf_path=report_paths["report_summaries_pdf"],
		)
	)
	return stage_outputs


__all__ = ["run_report_summaries_phase", "write_reconstruct_summary_slides_pdf"]