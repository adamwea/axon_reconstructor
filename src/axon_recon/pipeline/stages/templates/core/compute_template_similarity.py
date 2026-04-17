from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from ..io import resolve_unit_output_paths, write_json
from ..models.inputs import PerUnitTemplatesOutputsConfig, ProbeGeometryConfig, TemplateComputeSimilarityPhaseConfig
from .render import render_template_circles_plot
from .template_similarity_methods import (
	TemplateSimilarityFeatures,
	build_template_similarity_features,
	build_global_amplitude_matrix,
	compute_ptp_cosine_similarity_matrix,
	compute_pairwise_template_similarity,
	normalize_template_similarity_method,
)


LOGGER = logging.getLogger("axon_recon.templates.compute_template_similarity")


@dataclass(frozen=True)
class TemplateSimilarityUnitInput:
	unit_id: Any
	template_c_by_t: np.ndarray
	locations_xy: np.ndarray


def _safe_unit_token(unit_id: Any) -> str:
	token = str(unit_id).strip().replace("/", "_")
	try:
		return f"{int(unit_id):04d}"
	except Exception:
		return token or "unknown"


def _score_text_color(score: float) -> str:
	return "black" if float(score) >= 0.72 else "white"


def _apply_dark_axes_style(ax: Any) -> None:
	ax.set_facecolor("black")
	for spine in ax.spines.values():
		spine.set_color("white")
	ax.tick_params(colors="white")
	ax.xaxis.label.set_color("white")
	ax.yaxis.label.set_color("white")
	ax.title.set_color("white")


def _resample_waveform(trace: np.ndarray, *, target_samples: int) -> np.ndarray:
	waveform = np.asarray(trace, dtype=float).reshape(-1)
	if int(waveform.size) == 0:
		return np.zeros((target_samples,), dtype=float)
	if int(waveform.size) == int(target_samples):
		return waveform
	if int(target_samples) <= 1:
		return np.asarray([float(waveform[0])], dtype=float)
	source_x = np.linspace(0.0, 1.0, int(waveform.size), dtype=float)
	target_x = np.linspace(0.0, 1.0, int(target_samples), dtype=float)
	return np.asarray(np.interp(target_x, source_x, waveform), dtype=float)


def _plot_waveform_overlay(
	ax: Any,
	*,
	features_a: TemplateSimilarityFeatures,
	features_b: TemplateSimilarityFeatures,
	pair_score: dict[str, Any],
) -> None:
	_apply_dark_axes_style(ax)
	target_samples = int(max(len(features_a.dominant_waveform), len(features_b.dominant_waveform), 1))
	waveform_a = _resample_waveform(features_a.dominant_waveform, target_samples=target_samples)
	waveform_b = _resample_waveform(features_b.dominant_waveform, target_samples=target_samples)
	x_values = np.arange(target_samples, dtype=float)
	ax.plot(x_values, waveform_a, color="#4cc9f0", linewidth=2.0, label=f"u{features_a.unit_id}")
	ax.plot(x_values, waveform_b, color="#f72585", linewidth=2.0, label=f"u{features_b.unit_id}")
	ax.axhline(0.0, color="#aaaaaa", linewidth=0.8, alpha=0.6)
	ax.set_title("Dominant-channel waveform")
	ax.set_xlabel("sample")
	ax.set_ylabel("norm amp")
	legend = ax.legend(frameon=False, loc="best", fontsize=8)
	for text in legend.get_texts():
		text.set_color("white")
	metrics = dict(pair_score.get("metrics", {}))
	annotation_lines = [
		f"score: {float(pair_score.get('score', 0.0)):.3f}",
		f"ptp cosine: {float(metrics.get('ptp_cosine', 0.0)):.3f}",
		f"weighted jaccard: {float(metrics.get('amplitude_weighted_jaccard', 0.0)):.3f}",
		f"occupied jaccard: {float(metrics.get('occupied_channel_jaccard', 0.0)):.3f}",
	]
	ax.text(
		0.02,
		0.98,
		"\n".join(annotation_lines),
		transform=ax.transAxes,
		ha="left",
		va="top",
		fontsize=8,
		color="white",
		bbox={"facecolor": "#111111", "alpha": 0.72, "edgecolor": "#444444"},
	)


def _write_similarity_matrix_plot(
	*,
	unit_ids: list[Any],
	matrix: np.ndarray,
	config: TemplateComputeSimilarityPhaseConfig,
	png_path: Path,
	svg_path: Path,
) -> dict[str, str]:
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]

	n_units = len(unit_ids)
	fig_size = max(5.5, min(18.0, 2.0 + (0.36 * float(max(1, n_units)))))
	fig, ax = plt.subplots(figsize=(fig_size, fig_size), constrained_layout=True)
	fig.patch.set_facecolor("black")
	_apply_dark_axes_style(ax)
	image = ax.imshow(
		np.asarray(matrix, dtype=float),
		vmin=0.0,
		vmax=1.0,
		cmap=str(config.matrix.color_map),
		interpolation="nearest",
	)
	ax.set_title("Template Similarity Matrix")
	show_tick_labels = bool(config.matrix.show_tick_labels) and n_units <= 60
	if show_tick_labels:
		labels = [str(unit_id) for unit_id in unit_ids]
		ax.set_xticks(np.arange(n_units, dtype=int))
		ax.set_yticks(np.arange(n_units, dtype=int))
		ax.set_xticklabels(labels, rotation=90, fontsize=float(config.matrix.tick_fontsize), color="white")
		ax.set_yticklabels(labels, fontsize=float(config.matrix.tick_fontsize), color="white")
	else:
		ax.set_xticks([])
		ax.set_yticks([])
	ax.set_xlabel("unit")
	ax.set_ylabel("unit")
	color_bar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.02)
	color_bar.ax.tick_params(colors="white")
	color_bar.outline.set_edgecolor("white")
	color_bar.set_label("similarity", color="white")
	if bool(config.matrix.annotate_values) and n_units <= 24:
		for row_idx in range(n_units):
			for col_idx in range(n_units):
				score = float(matrix[row_idx, col_idx])
				ax.text(
					col_idx,
					row_idx,
					f"{score:.2f}",
					ha="center",
					va="center",
					fontsize=float(config.matrix.annotation_fontsize),
					color=_score_text_color(score),
				)
	outputs: dict[str, str] = {}
	if bool(config.matrix.write_png):
		png_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(png_path, dpi=float(config.matrix.dpi), facecolor=fig.get_facecolor(), bbox_inches="tight")
		outputs["template_similarity_matrix_png"] = str(png_path)
	if bool(config.matrix.write_svg):
		svg_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(svg_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
		outputs["template_similarity_matrix_svg"] = str(svg_path)
	plt.close(fig)
	return outputs


def _write_candidate_pair_plot(
	*,
	templates_out_dir: Path,
	features_a: TemplateSimilarityFeatures,
	features_b: TemplateSimilarityFeatures,
	pair_score: dict[str, Any],
	output_path: Path,
	dpi: float,
	per_unit_outputs: PerUnitTemplatesOutputsConfig,
	probe_geometry: ProbeGeometryConfig | None,
) -> str:
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]
	import matplotlib.image as mpimg  # type: ignore[import-not-found]

	def _resolve_template_circles_panel_png(
		*,
		features: TemplateSimilarityFeatures,
		cache_dir: Path,
	) -> Path:
		unit_paths = resolve_unit_output_paths(
			templates_out_dir=templates_out_dir,
			unit_id=features.unit_id,
			per_unit_outputs=per_unit_outputs,
		)
		canonical_png = unit_paths["template_circles_png"]
		if canonical_png.exists():
			return canonical_png
		cache_dir.mkdir(parents=True, exist_ok=True)
		cache_png = cache_dir / f"unit_{_safe_unit_token(features.unit_id)}__template_circles.png"
		if cache_png.exists():
			return cache_png
		render_config = replace(
			per_unit_outputs.template_circles,
			write_png=True,
			write_svg=False,
		)
		render_template_circles_plot(
			template=features.template_c_by_t,
			locations_xy=features.locations_xy,
			config=render_config,
			png_path=cache_png,
			svg_path=cache_png.with_suffix(".svg"),
			probe_geometry=probe_geometry,
			unit_id=features.unit_id,
		)
		return cache_png

	fig = plt.figure(figsize=(13.5, 9.0), constrained_layout=True)
	grid_spec = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.75])
	left_ax = fig.add_subplot(grid_spec[0, 0])
	right_ax = fig.add_subplot(grid_spec[0, 1])
	waveform_ax = fig.add_subplot(grid_spec[1, :])
	fig.patch.set_facecolor("black")
	panel_cache_dir = output_path.parent / "_template_circles_cache"
	panel_a_png = _resolve_template_circles_panel_png(features=features_a, cache_dir=panel_cache_dir)
	panel_b_png = _resolve_template_circles_panel_png(features=features_b, cache_dir=panel_cache_dir)
	for ax, panel_path in ((left_ax, panel_a_png), (right_ax, panel_b_png)):
		ax.set_facecolor("black")
		ax.imshow(mpimg.imread(panel_path), interpolation="nearest")
		ax.axis("off")
	_plot_waveform_overlay(
		waveform_ax,
		features_a=features_a,
		features_b=features_b,
		pair_score=pair_score,
	)
	fig.suptitle(
		f"Template similarity candidate: unit {features_a.unit_id} vs unit {features_b.unit_id}",
		color="white",
		fontsize=12,
	)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=float(dpi), facecolor=fig.get_facecolor(), bbox_inches="tight")
	plt.close(fig)
	return str(output_path)


def _candidate_pair_plot_path(*, output_dir: Path, unit_id_a: Any, unit_id_b: Any) -> Path:
	return output_dir / f"pair_u{_safe_unit_token(unit_id_a)}__u{_safe_unit_token(unit_id_b)}.png"


def _progress_interval(total_items: int, *, target_updates: int = 10, minimum: int = 1) -> int:
	if total_items <= 0:
		return int(max(1, minimum))
	return int(max(minimum, total_items // max(1, target_updates)))


def _build_pair_scores_from_matrix(
	*,
	unit_ids: list[Any],
	matrix: np.ndarray,
	method: str,
) -> list[dict[str, Any]]:
	unit_count = len(unit_ids)
	if unit_count <= 1:
		return []
	progress_interval = _progress_interval(unit_count, target_updates=8, minimum=8)
	pair_scores: list[dict[str, Any]] = []
	for row_idx in range(unit_count - 1):
		row_scores = np.asarray(matrix[row_idx, row_idx + 1 :], dtype=float)
		unit_a = unit_ids[row_idx]
		for offset, score in enumerate(row_scores, start=row_idx + 1):
			pair_scores.append(
				{
					"unit_a": unit_a,
					"unit_b": unit_ids[offset],
					"method": method,
					"score": float(score),
				}
			)
		if ((row_idx + 1) % progress_interval == 0) or (row_idx == (unit_count - 2)):
			LOGGER.info(
				"templates.compute_template_similarity score assembly progress: rows=%d/%d accumulated_pairs=%d",
				int(row_idx + 1),
				int(unit_count - 1),
				int(len(pair_scores)),
			)
	return pair_scores


def _enrich_candidate_pairs(
	*,
	candidate_pairs: list[dict[str, Any]],
	feature_lookup: dict[str, TemplateSimilarityFeatures],
	method: str,
) -> list[dict[str, Any]]:
	if not candidate_pairs:
		return []
	enriched_pairs: list[dict[str, Any]] = []
	progress_interval = _progress_interval(len(candidate_pairs), target_updates=6, minimum=1)
	for idx, candidate_pair in enumerate(candidate_pairs, start=1):
		features_a = feature_lookup[str(candidate_pair["unit_a"])]
		features_b = feature_lookup[str(candidate_pair["unit_b"])]
		pairwise = compute_pairwise_template_similarity(
			features_a=features_a,
			features_b=features_b,
			method=method,
		)
		enriched_pairs.append(
			{
				**candidate_pair,
				"metrics": dict(pairwise.metrics),
				"shared_channel_count": int(pairwise.shared_channel_count),
				"union_channel_count": int(pairwise.union_channel_count),
			}
		)
		if (idx % progress_interval == 0) or (idx == len(candidate_pairs)):
			LOGGER.info(
				"templates.compute_template_similarity candidate metric progress: %d/%d pairs enriched",
				int(idx),
				int(len(candidate_pairs)),
			)
	return enriched_pairs


def _select_candidate_pairs(
	*,
	unit_ids: list[Any],
	pair_scores: list[dict[str, Any]],
	config: TemplateComputeSimilarityPhaseConfig,
) -> list[dict[str, Any]]:
	threshold = float(max(0.0, min(1.0, config.candidate_selection.min_similarity)))
	top_k_per_unit = int(max(1, config.candidate_selection.top_k_per_unit))
	max_pairs = int(max(1, config.candidate_selection.max_pairs))
	best_by_unit: dict[str, list[dict[str, Any]]] = {str(unit_id): [] for unit_id in unit_ids}
	for pair_score in pair_scores:
		if float(pair_score.get("score", 0.0)) < threshold:
			continue
		for side in ("unit_a", "unit_b"):
			best_by_unit[str(pair_score[side])].append(pair_score)
	for unit_key, rows in best_by_unit.items():
		rows.sort(key=lambda row: (-float(row.get("score", 0.0)), str(row.get("unit_a")), str(row.get("unit_b"))))
		best_by_unit[unit_key] = rows[:top_k_per_unit]
	selected_keys: set[tuple[str, str]] = set()
	selected_rows: list[dict[str, Any]] = []
	for pair_score in sorted(pair_scores, key=lambda row: (-float(row.get("score", 0.0)), str(row.get("unit_a")), str(row.get("unit_b")))):
		if float(pair_score.get("score", 0.0)) < threshold:
			continue
		unit_a = str(pair_score.get("unit_a"))
		unit_b = str(pair_score.get("unit_b"))
		is_top_for_a = pair_score in best_by_unit.get(unit_a, [])
		is_top_for_b = pair_score in best_by_unit.get(unit_b, [])
		if (not is_top_for_a) and (not is_top_for_b):
			continue
		pair_key = tuple(sorted((unit_a, unit_b)))
		if pair_key in selected_keys:
			continue
		selected_keys.add(pair_key)
		selected_rows.append(
			{
				**pair_score,
				"selected": True,
				"mutual_top_k": bool(is_top_for_a and is_top_for_b),
				"top_k_hit_for_units": [
					value
					for value, hit in ((pair_score.get("unit_a"), is_top_for_a), (pair_score.get("unit_b"), is_top_for_b))
					if hit
				],
			}
		)
		if len(selected_rows) >= max_pairs:
			break
	return selected_rows


def build_template_similarity_phase_summary(
	*,
	unit_payloads: list[TemplateSimilarityUnitInput],
	templates_out_dir: Path,
	config: TemplateComputeSimilarityPhaseConfig,
	output_paths: dict[str, Path],
	per_unit_outputs: PerUnitTemplatesOutputsConfig,
	probe_geometry: ProbeGeometryConfig | None,
	missing_units: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
	resolved_method = normalize_template_similarity_method(config.method)
	missing = list(missing_units or [])
	feature_start = perf_counter()
	LOGGER.info(
		"templates.compute_template_similarity feature build start: units=%d method=%s",
		len(unit_payloads),
		resolved_method,
	)
	features_by_unit: list[TemplateSimilarityFeatures] = [
		build_template_similarity_features(
			unit_id=payload.unit_id,
			template_c_by_t=payload.template_c_by_t,
			locations_xy=payload.locations_xy,
		)
		for payload in unit_payloads
	]
	unit_ids = [features.unit_id for features in features_by_unit]
	unit_count = len(features_by_unit)
	if unit_count <= 0:
		raise FileNotFoundError("No merged template artifacts available for similarity computation")
	LOGGER.info(
		"templates.compute_template_similarity feature build complete: units=%d duration_seconds=%.3f",
		int(unit_count),
		float(perf_counter() - feature_start),
	)
	bulk_matrix_start = perf_counter()
	amplitude_matrix, ordered_locations = build_global_amplitude_matrix(features_by_unit)
	LOGGER.info(
		"templates.compute_template_similarity amplitude matrix prepared: shape=%s unique_locations=%d",
		tuple(amplitude_matrix.shape),
		int(len(ordered_locations)),
	)
	if resolved_method == "ptp_cosine":
		matrix = compute_ptp_cosine_similarity_matrix(amplitude_matrix)
		LOGGER.info(
			"templates.compute_template_similarity bulk cosine matrix complete: units=%d pair_count=%d duration_seconds=%.3f",
			int(unit_count),
			int((unit_count * (unit_count - 1)) // 2),
			float(perf_counter() - bulk_matrix_start),
		)
	else:
		LOGGER.info(
			"templates.compute_template_similarity falling back to pairwise scoring loop for method=%s",
			resolved_method,
		)
		matrix = np.eye(unit_count, dtype=float)
		progress_interval = _progress_interval(unit_count, target_updates=8, minimum=6)
		for row_idx in range(unit_count):
			for col_idx in range(row_idx + 1, unit_count):
				pairwise = compute_pairwise_template_similarity(
					features_a=features_by_unit[row_idx],
					features_b=features_by_unit[col_idx],
					method=resolved_method,
				)
				score = float(pairwise.score)
				matrix[row_idx, col_idx] = score
				matrix[col_idx, row_idx] = score
			if ((row_idx + 1) % progress_interval == 0) or (row_idx == (unit_count - 1)):
				LOGGER.info(
					"templates.compute_template_similarity pairwise fallback progress: rows=%d/%d",
					int(row_idx + 1),
					int(unit_count),
				)
	pair_scores = _build_pair_scores_from_matrix(unit_ids=unit_ids, matrix=matrix, method=resolved_method)
	candidate_pairs = _select_candidate_pairs(unit_ids=unit_ids, pair_scores=pair_scores, config=config)
	feature_lookup = {str(features.unit_id): features for features in features_by_unit}
	candidate_pairs = _enrich_candidate_pairs(
		candidate_pairs=candidate_pairs,
		feature_lookup=feature_lookup,
		method=resolved_method,
	)
	LOGGER.info(
		"templates.compute_template_similarity candidate selection complete: candidates=%d threshold=%.3f top_k=%d max_pairs=%d",
		int(len(candidate_pairs)),
		float(config.candidate_selection.min_similarity),
		int(config.candidate_selection.top_k_per_unit),
		int(config.candidate_selection.max_pairs),
	)
	outputs: dict[str, str] = {}
	LOGGER.info("templates.compute_template_similarity writing similarity matrix plot")
	outputs.update(
		_write_similarity_matrix_plot(
			unit_ids=unit_ids,
			matrix=matrix,
			config=config,
			png_path=output_paths["template_similarity_matrix_png"],
			svg_path=output_paths["template_similarity_matrix_svg"],
		)
	)
	scores_payload = {
		"method": resolved_method,
		"unit_ids": [unit_id for unit_id in unit_ids],
		"matrix": np.asarray(matrix, dtype=float).tolist(),
		"unique_location_count": int(len(ordered_locations)),
		"pair_scores": pair_scores,
		"missing_units": missing,
	}
	write_json(output_paths["template_similarity_scores_json"], scores_payload)
	outputs["template_similarity_scores_json"] = str(output_paths["template_similarity_scores_json"])
	pair_plot_records: list[dict[str, Any]] = []
	if bool(config.pair_plots.write_png):
		pair_plot_dir = output_paths["template_similarity_candidate_pair_plots_dir"]
		pair_plot_dir.mkdir(parents=True, exist_ok=True)
		outputs["template_similarity_candidate_pair_plots_dir"] = str(pair_plot_dir)
		progress_interval = _progress_interval(len(candidate_pairs), target_updates=6, minimum=1)
		for idx, candidate_pair in enumerate(candidate_pairs, start=1):
			features_a = feature_lookup[str(candidate_pair["unit_a"])]
			features_b = feature_lookup[str(candidate_pair["unit_b"])]
			plot_path = _candidate_pair_plot_path(
				output_dir=pair_plot_dir,
				unit_id_a=candidate_pair["unit_a"],
				unit_id_b=candidate_pair["unit_b"],
			)
			pair_plot_path = _write_candidate_pair_plot(
				templates_out_dir=templates_out_dir,
				features_a=features_a,
				features_b=features_b,
				pair_score=candidate_pair,
				output_path=plot_path,
				dpi=float(config.pair_plots.dpi),
				per_unit_outputs=per_unit_outputs,
				probe_geometry=probe_geometry,
			)
			candidate_pair["pair_plot_png"] = pair_plot_path
			pair_plot_records.append(
				{
					"unit_a": candidate_pair["unit_a"],
					"unit_b": candidate_pair["unit_b"],
					"pair_plot_png": pair_plot_path,
				}
			)
			if (idx % progress_interval == 0) or (idx == len(candidate_pairs)):
				LOGGER.info(
					"templates.compute_template_similarity pair plot progress: %d/%d plots written",
					int(idx),
					int(len(candidate_pairs)),
				)
	candidate_payload = {
		"method": resolved_method,
		"candidate_count": int(len(candidate_pairs)),
		"candidates": candidate_pairs,
		"pair_plot_records": pair_plot_records,
		"selection": {
			"min_similarity": float(config.candidate_selection.min_similarity),
			"top_k_per_unit": int(config.candidate_selection.top_k_per_unit),
			"max_pairs": int(config.candidate_selection.max_pairs),
		},
	}
	write_json(output_paths["template_similarity_candidate_pairs_json"], candidate_payload)
	outputs["template_similarity_candidate_pairs_json"] = str(output_paths["template_similarity_candidate_pairs_json"])
	return {
		"phase": "compute_template_similarity",
		"templates_out_dir": str(templates_out_dir),
		"method": resolved_method,
		"unit_count": int(unit_count),
		"unit_ids": [unit_id for unit_id in unit_ids],
		"pair_count": int(len(pair_scores)),
		"candidate_pair_count": int(len(candidate_pairs)),
		"candidate_pairs": candidate_pairs,
		"missing_units": missing,
		"outputs": outputs,
	}