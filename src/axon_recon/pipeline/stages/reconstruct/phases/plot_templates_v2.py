from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from axon_recon.pipeline.stages.reconstruct.templates import runner as templates_runner
from axon_recon.pipeline.stages.reconstruct.templates.core.render import render_template_circles_plot_v2
from axon_recon.pipeline.stages.reconstruct.templates.io import read_json, resolve_unit_output_paths, write_json
from axon_recon.pipeline.stages.reconstruct.templates.models.inputs import TemplatesInputs


def _render_v2_paths(unit_dir: Path, relpath: str) -> tuple[Path, Path]:
	raw = Path(str(relpath)).expanduser()
	base = raw.with_suffix("") if raw.suffix.lower() in {".png", ".svg"} else raw
	return unit_dir / base.with_suffix(".png"), unit_dir / base.with_suffix(".svg")


def _requested_v2_outputs(*, png_path: Path, svg_path: Path, write_png: bool, write_svg: bool) -> dict[str, Path]:
	outputs: dict[str, Path] = {}
	if bool(write_png):
		outputs["template_circles_v2_png"] = png_path
	if bool(write_svg):
		outputs["template_circles_v2_svg"] = svg_path
	return outputs


def _load_unit_summary_payload(*, unit_summary_json: Path, unit_id: Any) -> dict[str, Any]:
	if unit_summary_json.exists():
		try:
			payload = read_json(unit_summary_json)
			if isinstance(payload, dict):
				payload.setdefault("unit_id", unit_id)
				payload.setdefault("status", "ok")
				payload.setdefault("outputs", {})
				return payload
		except Exception:
			pass
	return {"unit_id": unit_id, "status": "ok", "error": None, "outputs": {}}


def _persist_unit_v2_outputs(
	*,
	unit_summary_json: Path,
	unit_id: Any,
	outputs: dict[str, str],
	selected_template_source: str,
	status: str = "ok",
	error: str | None = None,
) -> None:
	payload = _load_unit_summary_payload(unit_summary_json=unit_summary_json, unit_id=unit_id)
	unit_outputs = payload.get("outputs", {})
	if not isinstance(unit_outputs, dict):
		unit_outputs = {}
	unit_outputs.update(outputs)
	payload["outputs"] = unit_outputs
	selected_sources = payload.get("selected_template_sources", {})
	if not isinstance(selected_sources, dict):
		selected_sources = {}
	selected_sources["template_circles_v2"] = selected_template_source
	payload["selected_template_sources"] = selected_sources
	payload["plot_templates_v2"] = {
		"status": status,
		"error": error,
		"selected_template_source": selected_template_source,
		"outputs": dict(outputs),
	}
	write_json(unit_summary_json, payload)


def run_reconstruct_templates_plot_templates_v2_phase(inputs: TemplatesInputs) -> dict[str, Any]:
	phase_started = perf_counter()
	phase_cfg = inputs.phases.plot_templates_v2
	well_out_dir, _, templates_out_dir, _ = templates_runner._resolve_templates_phase_environment(inputs)
	try:
		merged_units_dir, full_channels_templates_dir = templates_runner._resolve_templates_dirs(
			well_out_dir=well_out_dir,
			templates_out_dir=templates_out_dir,
		)
	except FileNotFoundError as exc:
		raise FileNotFoundError(
			"Missing built template artifacts for plot_templates_v2; run templates.build_templates first"
		) from exc

	unit_ids = templates_runner._build_unit_ids(inputs, merged_units_dir)
	unit_ids = templates_runner._apply_unit_label_filter(inputs, unit_ids, well_out_dir, context="plot_templates_v2")
	if not unit_ids:
		raise FileNotFoundError(
			f"No built template artifacts found under {merged_units_dir}; run templates.build_templates first"
		)

	force_replot_requested = bool(inputs.force_restart) or bool(inputs.force_replot) or bool(inputs.force_replot_per_unit)
	rendered_units: list[Any] = []
	skipped_units: list[Any] = []
	failed_units: list[dict[str, Any]] = []
	outputs_by_unit: dict[str, dict[str, str]] = {}
	requested_output_keys: set[str] = set()

	templates_runner.LOGGER.info(
		"templates.plot_templates_v2 start: templates_out_dir=%s units=%d force_restart=%s output_relpath=%s",
		str(templates_out_dir),
		len(unit_ids),
		bool(inputs.force_restart),
		str(phase_cfg.output_relpath),
	)
	with templates_runner._quiet_unexpected_plot_logs(inputs):
		for unit_id in unit_ids:
			paths = resolve_unit_output_paths(
				templates_out_dir=templates_out_dir,
				unit_id=unit_id,
				per_unit_outputs=inputs.per_unit_outputs,
			)
			png_path, svg_path = _render_v2_paths(paths["unit_dir"], str(phase_cfg.output_relpath))
			requested_outputs = _requested_v2_outputs(
				png_path=png_path,
				svg_path=svg_path,
				write_png=bool(phase_cfg.write_png),
				write_svg=bool(phase_cfg.write_svg),
			)
			requested_output_keys.update(requested_outputs.keys())
			existing_outputs = {
				key: str(path)
				for key, path in requested_outputs.items()
				if path.exists()
			}
			if requested_outputs and (not force_replot_requested) and len(existing_outputs) == len(requested_outputs):
				_persist_unit_v2_outputs(
					unit_summary_json=paths["unit_summary_json"],
					unit_id=unit_id,
					outputs=existing_outputs,
					selected_template_source="existing",
				)
				skipped_units.append(unit_id)
				outputs_by_unit[str(unit_id)] = existing_outputs
				continue
			if not requested_outputs:
				skipped_units.append(unit_id)
				outputs_by_unit[str(unit_id)] = {}
				continue

			try:
				merged_dir = merged_units_dir / f"unit_{unit_id}"
				full_dir = full_channels_templates_dir / f"unit_{unit_id}"
				templates_runner.LOGGER.info(
					"Templates v2 unit render: unit_id=%s merged_dir=%s",
					unit_id,
					str(merged_dir),
				)
				merged_template, merged_locs = templates_runner._load_merged_unit(merged_dir)
				full_payload = templates_runner._load_full_unit(full_dir) if full_channels_templates_dir.exists() else None
				template, locations, selected_source = templates_runner._select_template_for_scope(
					merged_template=merged_template,
					merged_locs=merged_locs,
					full_payload=full_payload,
					channel_scope=str(phase_cfg.channel_scope),
				)
				outputs = render_template_circles_plot_v2(
					template=template,
					locations_xy=locations,
					config=phase_cfg,
					png_path=png_path,
					svg_path=svg_path,
					probe_geometry=inputs.probe_geometry,
					unit_id=unit_id,
				)
				_persist_unit_v2_outputs(
					unit_summary_json=paths["unit_summary_json"],
					unit_id=unit_id,
					outputs=outputs,
					selected_template_source=selected_source,
				)
				rendered_units.append(unit_id)
				outputs_by_unit[str(unit_id)] = dict(outputs)
			except Exception as exc:
				error = str(exc)
				failed_units.append({"unit_id": unit_id, "error": error})
				_persist_unit_v2_outputs(
					unit_summary_json=paths["unit_summary_json"],
					unit_id=unit_id,
					outputs={},
					selected_template_source="error",
					status="error",
					error=error,
				)
				templates_runner.LOGGER.exception("Templates v2 unit failed: unit_id=%s", unit_id)

	summary = {
		"phase": "plot_templates_v2",
		"templates_out_dir": str(templates_out_dir),
		"merged_units_dir": str(merged_units_dir),
		"full_channels_templates_dir": str(full_channels_templates_dir),
		"unit_count": len(unit_ids),
		"rendered_units": rendered_units,
		"skipped_units": skipped_units,
		"failed_units": failed_units,
		"requested_outputs": sorted(requested_output_keys),
		"outputs_by_unit": outputs_by_unit,
		"force_restart": bool(inputs.force_restart),
		"force_replot": bool(inputs.force_replot),
		"duration_seconds": float(perf_counter() - phase_started),
		"applied_debug_limits": templates_runner._templates_applied_debug_limits(inputs),
	}
	summary_path = templates_out_dir / str(phase_cfg.summary_json_relpath)
	write_json(summary_path, summary)
	summary["summary_json"] = str(summary_path)
	templates_runner.LOGGER.info("templates.plot_templates_v2 wrote summary output: %s", str(summary_path))
	templates_runner.LOGGER.info(
		"templates.plot_templates_v2 run stats: duration_seconds=%.3f unit_count=%d rendered_units=%d skipped_units=%d failed_units=%d",
		float(summary["duration_seconds"]),
		int(summary["unit_count"]),
		len(rendered_units),
		len(skipped_units),
		len(failed_units),
	)
	return summary