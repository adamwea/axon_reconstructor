"""Per-stage / per-dataset / per-well processing-status scan.

Generalization of ``examples/detect_incomplete_spikesort.py``: walks a
runtime yml + its data yml, and for every ``(stage, dataset, well)`` checks
whether the stage's completion marker exists on disk. The CLI subcommand
``axon-recon status`` renders the result as a per-stage table.

Stage completion is detected by the presence of a single well-level marker
file per stage (chosen as the natural "stage done" artifact each runner
writes). Verbose mode additionally checks per-phase markers under each
stage's per-well output dir.

Marker paths reflect the on-disk conventions used by current runners
(spikesort/runner.py, preprocess/runner.py, reconstruct/runner.py,
analysis/runner.py). If a runtime config overrides ``summary_json_relpath``
for a phase, the per-phase row may show that phase as missing even when it
ran — the per-stage well marker (which doesn't depend on relpath overrides)
is still the authoritative answer for "did the well finish this stage."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml


STAGE_ORDER: tuple[str, ...] = ("preprocess", "spikesort", "reconstruct", "analysis")


# Marker that signals "this stage finished for this well." Path is relative
# to the per-well output dir (compute_mea_analysis_output_dir output).
STAGE_WELL_MARKER: dict[str, tuple[str, ...]] = {
	"preprocess": ("preprocess_outputs", "preprocess_summary.json"),
	"spikesort": ("spikesort_outputs", "merge_SLAy", "merge_stage_summary.json"),
	"reconstruct": ("recon_outputs", "context", "report_summaries_summary.json"),
	"analysis": ("analysis_outputs", "manifest.json"),
}


# Per-phase markers, relative to the per-well output dir. Ordered to mirror
# the canonical phase sequence.
STAGE_PHASES: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
	"preprocess": (
		("copy_src_to_scratch", ("preprocess_outputs", "context", "copy_src_to_scratch_summary.json")),
		("save_rec_metadata", ("preprocess_outputs", "context", "recording_metadata_summary.json")),
		("prepare_raw_binaries", ("preprocess_outputs", "context", "prepare_raw_binaries_summary.json")),
		("preprocess_segments", ("preprocess_outputs", "context", "segment_recordings_summary.json")),
		("plot_segment_traces", ("preprocess_outputs", "context", "plot_segment_traces_summary.json")),
		("plot_segment_channel_layouts", ("preprocess_outputs", "context", "plot_segment_channel_layouts_summary.json")),
		("concat_segments", ("preprocess_outputs", "context", "concat_segments_summary.json")),
		("plot_concat_traces", ("preprocess_outputs", "context", "plot_concat_traces_summary.json")),
		("plot_concat_channel_layout", ("preprocess_outputs", "context", "plot_concat_channel_layout_summary.json")),
		("plot_raster_threshold", ("preprocess_outputs", "context", "plot_raster_threshold_summary.json")),
		("wipe_src_scratch", ("preprocess_outputs", "context", "wipe_src_scratch_summary.json")),
	),
	"spikesort": (
		("bootstrap_concat_binary", ("spikesort_outputs", "cache", "bootstrap_concat_binary", "bootstrap_concat_binary_summary.json")),
		("sort", ("spikesort_outputs", "spikesort_summary.json")),
		("summarize_sort", ("spikesort_outputs", "summarize_sort_summary.json")),
		("snapshot_sorter_output", ("spikesort_outputs", "snapshot_sorter_output_summary.json")),
		("concat_analyzer", ("spikesort_outputs", "concat_analyzer_summary.json")),
		("bombcell_label", ("spikesort_outputs", "bombcell_label_outputs", "bombcell_label_summary.json")),
		("merge_SLAy", ("spikesort_outputs", "merge_SLAy", "merge_stage_summary.json")),
		("bombcell_label_pass2", ("spikesort_outputs", "bombcell_label_outputs", "bombcell_label_pass2_summary.json")),
	),
	"reconstruct": (
		("analyzers", ("recon_outputs", "context", "analyzers_summary.json")),
		("extract_partial_templates", ("recon_outputs", "context", "extract_partial_templates_summary.json")),
		("build_templates", ("recon_outputs", "context", "build_templates_summary.json")),
		("plot_templates_v2", ("recon_outputs", "context", "plot_templates_v2_summary.json")),
		("report_templates", ("recon_outputs", "context", "report_templates_summary.json")),
		("generate_gtrs", ("recon_outputs", "context", "generate_gtrs_summary.json")),
		("plot_recons", ("recon_outputs", "context", "plot_recons_summary.json")),
		("plot_branch_propagations", ("recon_outputs", "context", "plot_branch_propagations_summary.json")),
		("plot_branch_velocities", ("recon_outputs", "context", "plot_branch_velocities_summary.json")),
		("plot_unit_summary", ("recon_outputs", "context", "plot_unit_summary_summary.json")),
		("report_recons", ("recon_outputs", "context", "report_recons_summary.json")),
		("report_recon_grid", ("recon_outputs", "context", "report_recon_grid_summary.json")),
		("report_full_chip_layout", ("recon_outputs", "context", "report_full_chip_layout_summary.json")),
		("report_summaries", ("recon_outputs", "context", "report_summaries_summary.json")),
	),
	"analysis": (
		("compute_metrics", ("analysis_outputs", "manifest.json")),
	),
}


@dataclass(frozen=True)
class WellStatus:
	well_id: str
	stage_done: bool
	phase_done: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetStatus:
	index: int
	div: int | None
	short_label: str
	rel_pattern: str
	wells: list[WellStatus] = field(default_factory=list)


@dataclass(frozen=True)
class StageStatus:
	stage: str
	datasets: list[DatasetStatus] = field(default_factory=list)


@dataclass(frozen=True)
class StatusReport:
	runtime_yml: Path
	data_yml: Path
	output_root: Path
	stages: list[StageStatus] = field(default_factory=list)


def _resolve_data_yml(runtime_yml: Path, data_ref: str | Path) -> Path:
	data_path = Path(data_ref)
	if data_path.is_absolute():
		return data_path
	return (runtime_yml.parent / data_path).resolve()


def _rel_pattern_from_h5(raw_h5: Path) -> str:
	"""Mirror compute_mea_analysis_output_dir's rel pattern: last 5 path parts
	of the raw H5 file (excluding ``data.raw.h5`` itself)."""
	parts = str(raw_h5).split("/")
	if len(parts) > 5:
		return "/".join(parts[-6:-1])
	return raw_h5.name


def _short_h5(raw_h5: Path) -> str:
	parts = str(raw_h5).split("/")
	if len(parts) >= 5:
		return f"{parts[-5]}/{parts[-4]}/{parts[-2]}"
	return raw_h5.name


def _iter_included_datasets(data_cfg: dict, target_datasets: set[int] | None) -> Iterable[tuple[int, dict]]:
	for i, dataset in enumerate(data_cfg.get("datasets", []) or []):
		if not isinstance(dataset, dict):
			continue
		if not dataset.get("include_in_runtime", True):
			continue
		if target_datasets is not None and i not in target_datasets:
			continue
		yield i, dataset


def _included_wells(dataset: dict, target_wells: set[str] | None = None) -> list[str]:
	out: list[str] = []
	for well in dataset.get("wells", []) or []:
		if not isinstance(well, dict):
			continue
		if not well.get("include_in_runtime", True):
			continue
		well_id = str(well["well_id"])
		if target_wells is not None and well_id not in target_wells:
			continue
		out.append(well_id)
	return out


def scan_status(
	runtime_yml: Path,
	*,
	target_datasets: Iterable[int] | None = None,
	target_wells: Iterable[str] | None = None,
	stages: Iterable[str] | None = None,
	collect_phases: bool = False,
) -> StatusReport:
	"""Walk the runtime + data configs and check completion markers on disk."""
	runtime_yml = Path(runtime_yml).expanduser().resolve()
	runtime_cfg = yaml.safe_load(runtime_yml.read_text())
	data_yml = _resolve_data_yml(runtime_yml, runtime_cfg.get("data", "./debug.data.yml"))
	data_cfg = yaml.safe_load(data_yml.read_text())
	output_root = Path(data_cfg["output_root"]).expanduser()

	target_set: set[int] | None = None
	if target_datasets is not None:
		target_set = {int(value) for value in target_datasets}
	target_wells_set: set[str] | None = None
	if target_wells is not None:
		target_wells_set = {str(w) for w in target_wells if str(w).strip()}

	stage_list = list(stages) if stages is not None else list(STAGE_ORDER)
	for stage_name in stage_list:
		if stage_name not in STAGE_WELL_MARKER:
			raise ValueError(f"Unknown stage for status: {stage_name!r}")

	stages_out: list[StageStatus] = []
	for stage_name in stage_list:
		well_marker = STAGE_WELL_MARKER[stage_name]
		phase_markers = STAGE_PHASES.get(stage_name, ()) if collect_phases else ()

		datasets_out: list[DatasetStatus] = []
		for i, dataset in _iter_included_datasets(data_cfg, target_set):
			raw_h5 = Path(dataset["raw_data_h5_path"])
			rel_pattern = _rel_pattern_from_h5(raw_h5)
			div_raw = dataset.get("DIV", None)
			div: int | None
			try:
				div = int(div_raw) if div_raw is not None else None
			except Exception:
				div = None
			wells_out: list[WellStatus] = []
			for well_id in _included_wells(dataset, target_wells_set):
				well_root = output_root / rel_pattern / well_id
				stage_done = (well_root.joinpath(*well_marker)).is_file()
				phase_done: dict[str, bool] = {}
				for phase_name, relparts in phase_markers:
					phase_done[phase_name] = (well_root.joinpath(*relparts)).is_file()
				wells_out.append(WellStatus(well_id=well_id, stage_done=stage_done, phase_done=phase_done))
			datasets_out.append(
				DatasetStatus(
					index=int(i),
					div=div,
					short_label=_short_h5(raw_h5),
					rel_pattern=rel_pattern,
					wells=wells_out,
				)
			)
		stages_out.append(StageStatus(stage=stage_name, datasets=datasets_out))

	return StatusReport(
		runtime_yml=runtime_yml,
		data_yml=data_yml,
		output_root=output_root,
		stages=stages_out,
	)


def format_default_tables(report: StatusReport) -> str:
	"""Per-stage table: dataset × wells (ok/total + missing well list)."""
	lines: list[str] = []
	lines.append(f"# status scan vs {report.output_root}")
	lines.append(f"# runtime: {report.runtime_yml}")
	lines.append(f"# data:    {report.data_yml}")

	for stage in report.stages:
		lines.append("")
		marker_path = "/".join(STAGE_WELL_MARKER[stage.stage])
		lines.append(f"=== {stage.stage} ===")
		lines.append(f"# marker: <well>/{marker_path}")
		lines.append(f"{'idx':>3}  {'DIV':>3}  {'dataset':<30}  {'wells (ok/total)':<18}  missing_wells")
		lines.append(f"{'---':>3}  {'---':>3}  {'-'*30:<30}  {'-'*18:<18}  -------------")
		total_ok = 0
		total_wells = 0
		incomplete_count = 0
		for dataset in stage.datasets:
			total = len(dataset.wells)
			ok = sum(1 for well in dataset.wells if well.stage_done)
			missing = [well.well_id for well in dataset.wells if not well.stage_done]
			div_str = str(dataset.div) if dataset.div is not None and dataset.div >= 0 else "-"
			missing_str = ",".join(missing) if missing else "-"
			tag = "  COMPLETE" if not missing else ""
			lines.append(
				f"{dataset.index:>3}  {div_str:>3}  {dataset.short_label:<30}  {ok}/{total:<16}  {missing_str}{tag}"
			)
			total_ok += ok
			total_wells += total
			if missing:
				incomplete_count += 1
		lines.append(f"# {incomplete_count} of {len(stage.datasets)} datasets incomplete · {total_ok}/{total_wells} wells done")
	return "\n".join(lines)


def format_verbose_tables(report: StatusReport) -> str:
	"""Per-stage table expanded to one row per (dataset, well) with a numbered
	phase legend. Each row's ``phases`` column is a compact glyph string where
	position N corresponds to phase N in the legend (1-indexed).
	"""
	lines: list[str] = []
	lines.append(f"# status scan vs {report.output_root}")
	lines.append(f"# runtime: {report.runtime_yml}")
	lines.append(f"# data:    {report.data_yml}")

	for stage in report.stages:
		phase_sequence = STAGE_PHASES.get(stage.stage, ())
		phase_names = [name for name, _ in phase_sequence]
		lines.append("")
		lines.append(f"=== {stage.stage} phases ===")
		if not phase_names:
			lines.append("(no phase markers configured)")
			continue
		lines.append("# phase legend (column position → phase name):")
		for idx, phase_name in enumerate(phase_names, start=1):
			lines.append(f"#   {idx:>2}. {phase_name}")
		lines.append("# glyphs: ✓ = marker exists  · = marker missing  (left-to-right matches legend order)")
		phases_header = "".join(f"{i % 10}" for i in range(1, len(phase_names) + 1))
		header = f"{'idx':>3}  {'DIV':>3}  {'dataset':<30}  {'well':<8}  done  phases({phases_header})"
		lines.append(header)
		lines.append("-" * min(len(header), 200))
		for dataset in stage.datasets:
			div_str = str(dataset.div) if dataset.div is not None and dataset.div >= 0 else "-"
			for well in dataset.wells:
				done_glyph = "✓" if well.stage_done else "·"
				phase_glyphs = "".join(
					("✓" if well.phase_done.get(name, False) else "·") for name in phase_names
				)
				lines.append(
					f"{dataset.index:>3}  {div_str:>3}  {dataset.short_label:<30}  {well.well_id:<8}  {done_glyph:>4}  {phase_glyphs}"
				)
	return "\n".join(lines)


def incomplete_dataset_indices(report: StatusReport, *, stage: str) -> list[int]:
	"""Indices of datasets with at least one well missing the stage marker."""
	for stage_status in report.stages:
		if stage_status.stage != stage:
			continue
		return [
			dataset.index
			for dataset in stage_status.datasets
			if any(not well.stage_done for well in dataset.wells)
		]
	return []
