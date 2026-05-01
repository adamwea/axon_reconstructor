from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from .models.inputs import PerUnitTemplatesOutputsConfig


MATERIALIZED_TEMPLATES_CACHE_RELPATH = Path("cache/templates")


def read_json(path: Path) -> Any:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def write_json(path: Path, payload: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(_json_compatible_value(payload), f, indent=2)


def _json_compatible_value(value: Any) -> Any:
	if isinstance(value, np.generic):
		return value.item()
	if isinstance(value, list):
		return [_json_compatible_value(item) for item in value]
	if isinstance(value, tuple):
		return [_json_compatible_value(item) for item in value]
	if isinstance(value, dict):
		return {key: _json_compatible_value(item) for key, item in value.items()}
	return value


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


def _render_pdf_png_svg_paths(base_dir: Path, *, pdf_relpath: str, png_relpath: str, svg_relpath: str) -> tuple[Path, Path, Path]:
	pdf = base_dir / Path(str(pdf_relpath)).expanduser()
	png = base_dir / Path(str(png_relpath)).expanduser()
	svg = base_dir / Path(str(svg_relpath)).expanduser()
	return pdf, png, svg


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
	merged_template_channel_locations_npy = (
		None
		if per_unit_outputs.merged_template.channel_locations_npy_relpath is None
		else _render_npy_path(unit_dir, npy_relpath=per_unit_outputs.merged_template.channel_locations_npy_relpath)
	)
	square_template_npy = _render_npy_path(unit_dir, npy_relpath=per_unit_outputs.square_template.npy_relpath)
	square_template_channel_locations_npy = (
		None
		if per_unit_outputs.square_template.channel_locations_npy_relpath is None
		else _render_npy_path(unit_dir, npy_relpath=per_unit_outputs.square_template.channel_locations_npy_relpath)
	)
	scan_template_npy = _render_npy_path(unit_dir, npy_relpath=per_unit_outputs.scan_template.npy_relpath)
	scan_template_channel_locations_npy = (
		None
		if per_unit_outputs.scan_template.channel_locations_npy_relpath is None
		else _render_npy_path(unit_dir, npy_relpath=per_unit_outputs.scan_template.channel_locations_npy_relpath)
	)
	full_template_npy = _render_npy_path(unit_dir, npy_relpath=per_unit_outputs.full_template.npy_relpath)
	full_template_channel_locations_npy = (
		None
		if per_unit_outputs.full_template.channel_locations_npy_relpath is None
		else _render_npy_path(unit_dir, npy_relpath=per_unit_outputs.full_template.channel_locations_npy_relpath)
	)
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
	propagation_svg = propagation_png.with_suffix(".svg")
	circles_numbered_relpath = per_unit_outputs.propagation_plots.circles_template_numbered_relpath
	if (
		str(circles_numbered_relpath) == "circles_template_numbered"
		and str(per_unit_outputs.propagation_plots.right_panel_png_relpath) != "propagation_plot__right_temp.png"
	):
		circles_numbered_relpath = per_unit_outputs.propagation_plots.right_panel_png_relpath
	circles_numbered_png, circles_numbered_svg = _render_template_paths(
		unit_dir,
		str(circles_numbered_relpath),
	)
	propagation_2panel_png, propagation_2panel_svg = _render_template_paths(
		unit_dir,
		per_unit_outputs.propagation_plots.propagation_2panel_relpath,
	)
	propagation_left_temp_svg = propagation_png.with_name(f"{propagation_png.stem}__left_temp.svg")
	_, propagation_right_temp_svg = _render_template_paths(
		unit_dir,
		per_unit_outputs.propagation_plots.right_panel_svg_relpath,
	)
	propagation_right_temp_png, _ = _render_template_paths(
		unit_dir,
		per_unit_outputs.propagation_plots.right_panel_png_relpath,
	)
	qc_cfg = per_unit_outputs.quality_checks.check_for_multiple_peaks_at_channel_templates
	qc_plot_png, qc_plot_svg = _render_template_paths(unit_dir, qc_cfg.plot.relpath)
	qc_json = unit_dir / Path(str(qc_cfg.json_relpath)).expanduser()

	return {
		"unit_dir": unit_dir,
		"unit_summary_json": unit_dir / "unit_templates_summary.json",
		"merged_contributing_electrode_ids_json": unit_dir / "merged_contributing_electrode_ids.json",
		"overlay_top_channel_meta_json": unit_dir / "overlay_top_channel_meta.json",
		"merged_template_npy": merged_template_npy,
		"merged_template_channel_locations_npy": merged_template_channel_locations_npy,
		"square_template_npy": square_template_npy,
		"square_template_channel_locations_npy": square_template_channel_locations_npy,
		"scan_template_npy": scan_template_npy,
		"scan_template_channel_locations_npy": scan_template_channel_locations_npy,
		"full_template_npy": full_template_npy,
		"full_template_channel_locations_npy": full_template_channel_locations_npy,
		"template_png": template_png,
		"template_svg": template_svg,
		"template_circles_png": template_circles_png,
		"template_circles_svg": template_circles_svg,
		"template_wf_overlay_pdf": overlay_pdf,
		"template_wf_overlay_png": overlay_png,
		"extremum_ch_wf_overlay_pdf": overlay_pdf,
		"extremum_ch_wf_overlay_png": overlay_png,
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
		"propagation_plot_svg": propagation_svg,
		"circles_template_numbered_png": circles_numbered_png,
		"circles_template_numbered_svg": circles_numbered_svg,
		"propagation_2panel_png": propagation_2panel_png,
		"propagation_2panel_svg": propagation_2panel_svg,
		"propagation_plot_left_temp_svg": propagation_left_temp_svg,
		"propagation_plot_right_temp_svg": propagation_right_temp_svg,
		"propagation_plot_right_temp_png": propagation_right_temp_png,
		"quality_checks_multiple_negative_peaks_json": qc_json,
		"quality_checks_multiple_negative_peaks_plot_png": qc_plot_png,
		"quality_checks_multiple_negative_peaks_plot_svg": qc_plot_svg,
	}


def resolve_report_output_paths(*, templates_out_dir: Path, reports: Any) -> dict[str, Path]:
	unit_locations_json = templates_out_dir / Path(str(reports.locations.json_relpath)).expanduser()
	unit_locations_png = templates_out_dir / Path(str(reports.locations.png_relpath)).expanduser()
	unit_locations_svg = templates_out_dir / Path(str(reports.locations.svg_relpath)).expanduser()
	wf_grid_pdf, wf_grid_png, wf_grid_svg = _render_pdf_png_svg_paths(
		templates_out_dir,
		pdf_relpath=reports.wf_overlay_grid.pdf_relpath,
		png_relpath=reports.wf_overlay_grid.png_relpath,
		svg_relpath=reports.wf_overlay_grid.svg_relpath,
	)
	amp_grid_pdf, amp_grid_png, amp_grid_svg = _render_pdf_png_svg_paths(
		templates_out_dir,
		pdf_relpath=reports.footprint_grids.amplitude_map_grid.pdf_relpath,
		png_relpath=reports.footprint_grids.amplitude_map_grid.png_relpath,
		svg_relpath=reports.footprint_grids.amplitude_map_grid.svg_relpath,
	)
	circles_grid_pdf, circles_grid_png, circles_grid_svg = _render_pdf_png_svg_paths(
		templates_out_dir,
		pdf_relpath=reports.footprint_grids.circles_map_grid.pdf_relpath,
		png_relpath=reports.footprint_grids.circles_map_grid.png_relpath,
		svg_relpath=reports.footprint_grids.circles_map_grid.svg_relpath,
	)
	lat_grid_pdf, lat_grid_png, lat_grid_svg = _render_pdf_png_svg_paths(
		templates_out_dir,
		pdf_relpath=reports.footprint_grids.latency_map_grid.pdf_relpath,
		png_relpath=reports.footprint_grids.latency_map_grid.png_relpath,
		svg_relpath=reports.footprint_grids.latency_map_grid.svg_relpath,
	)
	wf_grid_temp_svg = templates_out_dir / Path(str(reports.wf_overlay_grid.temp_svg_relpath)).expanduser()
	circles_grid_temp_svg = templates_out_dir / Path(str(reports.footprint_grids.circles_map_grid.temp_svg_relpath)).expanduser()
	amp_grid_temp_svg = templates_out_dir / Path(str(reports.footprint_grids.amplitude_map_grid.temp_svg_relpath)).expanduser()
	lat_grid_temp_svg = templates_out_dir / Path(str(reports.footprint_grids.latency_map_grid.temp_svg_relpath)).expanduser()
	multi_source_pdf = templates_out_dir / Path(str(reports.plot_multi_source_pdf.pdf_relpath)).expanduser()
	return {
		"unit_locations_json": unit_locations_json,
		"unit_locations_png": unit_locations_png,
		"unit_locations_svg": unit_locations_svg,
		"wf_overlay_grid_pdf": wf_grid_pdf,
		"wf_overlay_grid_png": wf_grid_png,
		"wf_overlay_grid_svg": wf_grid_svg,
		"wf_overlay_grid_temp_svg": wf_grid_temp_svg,
		"template_circles_map_grid_pdf": circles_grid_pdf,
		"template_circles_map_grid_png": circles_grid_png,
		"template_circles_map_grid_svg": circles_grid_svg,
		"template_circles_map_grid_temp_svg": circles_grid_temp_svg,
		"footprint_amplitude_map_grid_pdf": amp_grid_pdf,
		"footprint_amplitude_map_grid_png": amp_grid_png,
		"footprint_amplitude_map_grid_svg": amp_grid_svg,
		"footprint_amplitude_map_grid_temp_svg": amp_grid_temp_svg,
		"footprint_latency_map_grid_pdf": lat_grid_pdf,
		"footprint_latency_map_grid_png": lat_grid_png,
		"footprint_latency_map_grid_svg": lat_grid_svg,
		"footprint_latency_map_grid_temp_svg": lat_grid_temp_svg,
		"multi_source_pdf": multi_source_pdf,
	}


def resolve_similarity_output_paths(*, templates_out_dir: Path, similarity: Any) -> dict[str, Path]:
	scores_json = templates_out_dir / Path(str(similarity.scores_json_relpath)).expanduser()
	candidate_pairs_json = templates_out_dir / Path(str(similarity.candidate_pairs_json_relpath)).expanduser()
	pair_plots_dir = templates_out_dir / Path(str(similarity.pair_plots.relpath_root)).expanduser()
	matrix_png = templates_out_dir / Path(str(similarity.matrix.png_relpath)).expanduser()
	matrix_svg = templates_out_dir / Path(str(similarity.matrix.svg_relpath)).expanduser()
	return {
		"template_similarity_scores_json": scores_json,
		"template_similarity_candidate_pairs_json": candidate_pairs_json,
		"template_similarity_candidate_pair_plots_dir": pair_plots_dir,
		"template_similarity_matrix_png": matrix_png,
		"template_similarity_matrix_svg": matrix_svg,
	}


def resolve_materialized_templates_dirs(*, templates_out_dir: Path) -> tuple[Path, Path]:
	templates_root = templates_out_dir / MATERIALIZED_TEMPLATES_CACHE_RELPATH
	merged_units_dir = templates_root / "merged"
	full_channels_templates_dir = templates_root / "full"
	merged_units_dir.mkdir(parents=True, exist_ok=True)
	full_channels_templates_dir.mkdir(parents=True, exist_ok=True)
	return merged_units_dir, full_channels_templates_dir


def resolve_materialized_source_payload_unit_dir(
	*,
	templates_out_dir: Path,
	output_rel_root: str,
	source_name: str,
	unit_id: Any,
) -> Path:
	root = templates_out_dir / Path(str(output_rel_root)).expanduser()
	return root / str(source_name) / f"unit_{unit_id}"


def write_materialized_source_payload(
	*,
	templates_out_dir: Path,
	output_rel_root: str,
	source_name: str,
	unit_id: Any,
	template_c_by_t: np.ndarray,
	locations_xy: np.ndarray,
	electrode_ids: list[Any] | None,
	channel_ids: list[Any] | None,
	waveform_count: int,
	sampling_rate_hz: float | None,
	overlay_waveforms: np.ndarray | None,
	top_electrode_id: Any,
	total_waveforms_at_channel: int | None,
) -> Path:
	unit_dir = resolve_materialized_source_payload_unit_dir(
		templates_out_dir=templates_out_dir,
		output_rel_root=output_rel_root,
		source_name=source_name,
		unit_id=unit_id,
	)
	unit_dir.mkdir(parents=True, exist_ok=True)
	np.save(unit_dir / "template.npy", np.asarray(template_c_by_t, dtype=float))
	np.save(unit_dir / "channel_locations_xy.npy", np.asarray(locations_xy, dtype=float))
	meta = {
		"unit_id": _json_compatible_value(unit_id),
		"source_name": str(source_name),
		"waveform_count": int(waveform_count),
		"sampling_rate_hz": (None if sampling_rate_hz is None else float(sampling_rate_hz)),
		"electrode_ids": (None if electrode_ids is None else _json_compatible_value(list(electrode_ids))),
		"channel_ids": (None if channel_ids is None else _json_compatible_value(list(channel_ids))),
		"top_electrode_id": _json_compatible_value(top_electrode_id),
		"total_waveforms_at_channel": (None if total_waveforms_at_channel is None else int(total_waveforms_at_channel)),
	}
	write_json(unit_dir / "payload_meta.json", meta)
	if overlay_waveforms is not None:
		np.save(unit_dir / "overlay_top_channel_waveforms.npy", np.asarray(overlay_waveforms, dtype=float))
	return unit_dir


def load_materialized_source_payload(
	*,
	source_payload_unit_dir: Path,
) -> tuple[np.ndarray, np.ndarray, list[Any] | None, list[Any] | None, int, float | None, np.ndarray | None, Any, int | None] | None:
	template_path = source_payload_unit_dir / "template.npy"
	locations_path = source_payload_unit_dir / "channel_locations_xy.npy"
	meta_path = source_payload_unit_dir / "payload_meta.json"
	if (not template_path.exists()) or (not locations_path.exists()) or (not meta_path.exists()):
		return None
	meta = read_json(meta_path)
	if not isinstance(meta, dict):
		return None
	overlay_waveforms_path = source_payload_unit_dir / "overlay_top_channel_waveforms.npy"
	overlay_waveforms = np.load(overlay_waveforms_path) if overlay_waveforms_path.exists() else None
	return (
		np.asarray(np.load(template_path), dtype=float),
		np.asarray(np.load(locations_path), dtype=float),
		(None if meta.get("electrode_ids", None) is None else list(meta.get("electrode_ids", []))),
		(None if meta.get("channel_ids", None) is None else list(meta.get("channel_ids", []))),
		int(meta.get("waveform_count", 0)),
		(None if meta.get("sampling_rate_hz", None) is None else float(meta.get("sampling_rate_hz"))),
		(None if overlay_waveforms is None else np.asarray(overlay_waveforms, dtype=float)),
		meta.get("top_electrode_id", None),
		(None if meta.get("total_waveforms_at_channel", None) is None else int(meta.get("total_waveforms_at_channel"))),
	)


def write_materialized_unit_templates(
	*,
	merged_units_dir: Path,
	full_channels_templates_dir: Path,
	unit_id: Any,
	merged_template: np.ndarray,
	merged_locations_xy: np.ndarray,
	full_template: np.ndarray,
	full_locations_xy: np.ndarray,
	write_full_template: bool = True,
) -> None:
	merged_dir = merged_units_dir / f"unit_{unit_id}"
	merged_dir.mkdir(parents=True, exist_ok=True)
	np.save(merged_dir / "merged_contributing_template.npy", np.asarray(merged_template, dtype=float))
	np.save(merged_dir / "merged_contributing_channel_locations.npy", np.asarray(merged_locations_xy, dtype=float))

	full_dir = full_channels_templates_dir / f"unit_{unit_id}"
	if not bool(write_full_template):
		if full_dir.exists():
			shutil.rmtree(full_dir)
		return
	full_dir.mkdir(parents=True, exist_ok=True)
	np.save(full_dir / "full_template.npy", np.asarray(full_template, dtype=float))
	np.save(full_dir / "full_channel_locations_xy.npy", np.asarray(full_locations_xy, dtype=float))


def write_materialized_overlay_waveforms(
	*,
	merged_units_dir: Path,
	unit_id: Any,
	waveforms_by_t: np.ndarray,
	top_electrode_id: Any,
	total_waveforms_at_channel: int,
	metadata_json_path: Path | None = None,
) -> None:
	if isinstance(top_electrode_id, np.generic):
		top_electrode_id = top_electrode_id.item()
	merged_dir = merged_units_dir / f"unit_{unit_id}"
	merged_dir.mkdir(parents=True, exist_ok=True)
	np.save(merged_dir / "overlay_top_channel_waveforms.npy", np.asarray(waveforms_by_t, dtype=float))
	meta_path = metadata_json_path or (merged_dir / "overlay_top_channel_meta.json")
	write_json(
		meta_path,
		{
			"top_electrode_id": top_electrode_id,
			"top_channel_id": top_electrode_id,
			"total_waveforms_at_channel": int(max(0, int(total_waveforms_at_channel))),
		},
	)


def write_materialized_merged_electrode_ids(
	*,
	merged_units_dir: Path,
	unit_id: Any,
	electrode_ids: list[Any] | None,
	metadata_json_path: Path | None = None,
) -> None:
	merged_dir = merged_units_dir / f"unit_{unit_id}"
	merged_dir.mkdir(parents=True, exist_ok=True)
	meta_path = metadata_json_path or (merged_dir / "merged_contributing_electrode_ids.json")
	if electrode_ids is None:
		write_json(meta_path, {"electrode_ids": None})
		return
	serialized: list[Any] = []
	for eid in list(electrode_ids):
		if isinstance(eid, np.generic):
			serialized.append(eid.item())
		else:
			serialized.append(eid)
	write_json(meta_path, {"electrode_ids": serialized})


def load_materialized_merged_electrode_ids(*, merged_unit_dir: Path, metadata_dir: Path | None = None) -> list[Any] | None:
	meta_candidates = []
	if metadata_dir is not None:
		meta_candidates.append(metadata_dir / "merged_contributing_electrode_ids.json")
	meta_candidates.append(merged_unit_dir / "merged_contributing_electrode_ids.json")
	meta_path = next((path for path in meta_candidates if path.exists()), None)
	if meta_path is None:
		return None
	try:
		meta = read_json(meta_path)
		if not isinstance(meta, dict):
			return None
		electrode_ids = meta.get("electrode_ids", None)
		if electrode_ids is None:
			return None
		if not isinstance(electrode_ids, list):
			return None
		return list(electrode_ids)
	except Exception:
		return None


def load_materialized_overlay_waveforms(
	*,
	merged_unit_dir: Path,
	metadata_dir: Path | None = None,
) -> tuple[np.ndarray, Any, int] | None:
	wf_path = merged_unit_dir / "overlay_top_channel_waveforms.npy"
	meta_candidates = []
	if metadata_dir is not None:
		meta_candidates.append(metadata_dir / "overlay_top_channel_meta.json")
	meta_candidates.append(merged_unit_dir / "overlay_top_channel_meta.json")
	meta_path = next((path for path in meta_candidates if path.exists()), None)
	if (not wf_path.exists()) or meta_path is None:
		return None
	try:
		waveforms = np.asarray(np.load(wf_path), dtype=float)
		meta = read_json(meta_path)
		if waveforms.ndim != 2 or not isinstance(meta, dict):
			return None
		top_electrode_id = meta.get("top_electrode_id", meta.get("top_channel_id", None))
		total = int(max(0, int(meta.get("total_waveforms_at_channel", int(waveforms.shape[0])))))
		return waveforms, top_electrode_id, total
	except Exception:
		return None
