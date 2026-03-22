from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models.inputs import PerUnitTemplatesOutputsConfig


def read_json(path: Path) -> Any:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def write_json(path: Path, payload: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2)


def format_unit_reldir(unit_reldir: str, unit_id: Any) -> Path:
	try:
		unit_id_int = int(unit_id)
		rendered = str(unit_reldir).format(unit_id=unit_id_int)
	except Exception:
		rendered = str(unit_reldir).format(unit_id=unit_id)
	return Path(rendered)


def _render_template_paths(unit_dir: Path, relpath: str) -> tuple[Path, Path]:
	raw = Path(str(relpath)).expanduser()
	if raw.suffix.lower() in {".png", ".svg"}:
		base = raw.with_suffix("")
	else:
		base = raw
	return unit_dir / base.with_suffix(".png"), unit_dir / base.with_suffix(".svg")


def _render_pdf_png_paths(base_dir: Path, *, pdf_relpath: str, png_relpath: str) -> tuple[Path, Path]:
	pdf = base_dir / Path(str(pdf_relpath)).expanduser()
	png = base_dir / Path(str(png_relpath)).expanduser()
	return pdf, png


def _render_npy_path(base_dir: Path, *, npy_relpath: str) -> Path:
	raw = Path(str(npy_relpath)).expanduser()
	if raw.suffix.lower() == ".npy":
		return base_dir / raw
	return base_dir / raw.with_suffix(".npy")


def resolve_unit_output_paths(
	*,
	templates_out_dir: Path,
	unit_id: Any,
	per_unit_outputs: PerUnitTemplatesOutputsConfig,
) -> dict[str, Path]:
	unit_rel = format_unit_reldir(per_unit_outputs.unit_reldir, unit_id)
	unit_dir = templates_out_dir / unit_rel
	merged_template_npy = _render_npy_path(unit_dir, npy_relpath=per_unit_outputs.merged_template.npy_relpath)
	square_template_npy = _render_npy_path(unit_dir, npy_relpath=per_unit_outputs.square_template.npy_relpath)
	scan_template_npy = _render_npy_path(unit_dir, npy_relpath=per_unit_outputs.scan_template.npy_relpath)
	full_template_npy = _render_npy_path(unit_dir, npy_relpath=per_unit_outputs.full_template.npy_relpath)
	template_png, template_svg = _render_template_paths(unit_dir, per_unit_outputs.template.relpath)
	template_circles_png, template_circles_svg = _render_template_paths(unit_dir, per_unit_outputs.template_circles.relpath)
	overlay_pdf, overlay_png = _render_pdf_png_paths(
		unit_dir,
		pdf_relpath=per_unit_outputs.template_wf_overlay.pdf_relpath,
		png_relpath=per_unit_outputs.template_wf_overlay.png_relpath,
	)
	amp_png, amp_svg = _render_template_paths(unit_dir, per_unit_outputs.footprint_plots.amplitude_map.relpath)
	lat_png, lat_svg = _render_template_paths(unit_dir, per_unit_outputs.footprint_plots.latency_map.relpath)
	topo_amp_png, topo_amp_svg = _render_template_paths(unit_dir, per_unit_outputs.topographical_footprints.amplitude.relpath)
	topo_lat_png, topo_lat_svg = _render_template_paths(unit_dir, per_unit_outputs.topographical_footprints.latency.relpath)
	propagation_pdf, propagation_png = _render_pdf_png_paths(
		unit_dir,
		pdf_relpath=per_unit_outputs.propagation_plots.pdf_relpath,
		png_relpath=per_unit_outputs.propagation_plots.png_relpath,
	)

	return {
		"unit_dir": unit_dir,
		"unit_summary_json": unit_dir / "unit_templates_summary.json",
		"merged_template_npy": merged_template_npy,
		"square_template_npy": square_template_npy,
		"scan_template_npy": scan_template_npy,
		"full_template_npy": full_template_npy,
		"template_png": template_png,
		"template_svg": template_svg,
		"template_circles_png": template_circles_png,
		"template_circles_svg": template_circles_svg,
		"template_wf_overlay_pdf": overlay_pdf,
		"template_wf_overlay_png": overlay_png,
		"footprint_amplitude_map_png": amp_png,
		"footprint_amplitude_map_svg": amp_svg,
		"footprint_latency_map_png": lat_png,
		"footprint_latency_map_svg": lat_svg,
		"topographical_amplitude_footprint_png": topo_amp_png,
		"topographical_amplitude_footprint_svg": topo_amp_svg,
		"topographical_latency_footprint_png": topo_lat_png,
		"topographical_latency_footprint_svg": topo_lat_svg,
		"propagation_plot_pdf": propagation_pdf,
		"propagation_plot_png": propagation_png,
	}


def resolve_report_output_paths(*, templates_out_dir: Path, reports: Any) -> dict[str, Path]:
	wf_grid_pdf, wf_grid_png = _render_pdf_png_paths(
		templates_out_dir,
		pdf_relpath=reports.wf_overlay_grid.pdf_relpath,
		png_relpath=reports.wf_overlay_grid.png_relpath,
	)
	amp_grid_pdf, amp_grid_png = _render_pdf_png_paths(
		templates_out_dir,
		pdf_relpath=reports.footprint_grids.amplitude_map_grid.pdf_relpath,
		png_relpath=reports.footprint_grids.amplitude_map_grid.png_relpath,
	)
	lat_grid_pdf, lat_grid_png = _render_pdf_png_paths(
		templates_out_dir,
		pdf_relpath=reports.footprint_grids.latency_map_grid.pdf_relpath,
		png_relpath=reports.footprint_grids.latency_map_grid.png_relpath,
	)
	multi_source_pdf = templates_out_dir / Path(str(reports.plot_multi_source_pdf.pdf_relpath)).expanduser()
	return {
		"wf_overlay_grid_pdf": wf_grid_pdf,
		"wf_overlay_grid_png": wf_grid_png,
		"footprint_amplitude_map_grid_pdf": amp_grid_pdf,
		"footprint_amplitude_map_grid_png": amp_grid_png,
		"footprint_latency_map_grid_pdf": lat_grid_pdf,
		"footprint_latency_map_grid_png": lat_grid_png,
		"multi_source_pdf": multi_source_pdf,
	}
