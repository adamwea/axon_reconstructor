"""Per-stage / per-dataset / per-well processing-status scan.

Generalization of ``examples/detect_incomplete_spikesort.py``: walks a
runtime yml + its data yml, and for every ``(stage, dataset, well)`` checks
whether the stage's completion marker exists on disk. The CLI subcommand
``axon-recon status`` renders the result as a per-stage table.

Stage completion is detected by the presence of a single well-level marker
file per stage (chosen as the natural "stage done" artifact each runner
writes). Verbose mode additionally checks per-phase markers under each
stage's per-well output dir.

The scan also reads each existing marker's `status` / `reason` fields to
detect *skipped-but-complete* wells (e.g. spikesort's merge_SLAy writing a
stub with `reason="no_qualifying_units"` when bombcell rejected every
unit). Known-benign skip reasons surface as "OK skips"; anything else
shows as a flagged warning so the user can quickly spot wells that may
warrant exclusion from downstream analysis.

Marker paths reflect the on-disk conventions used by current runners
(spikesort/runner.py, preprocess/runner.py, reconstruct/runner.py,
analysis/runner.py). If a runtime config overrides ``summary_json_relpath``
for a phase, the per-phase row may show that phase as missing even when it
ran — the per-stage well marker (which doesn't depend on relpath overrides)
is still the authoritative answer for "did the well finish this stage."
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml


# Skip reasons that represent normal-but-not-an-error outcomes. Wells with
# only these reasons in their marker files are flagged as "OK skips" — the
# pipeline did the right thing given the inputs. Anything outside this set
# (or anything with status="error" / status="failed") is flagged as a
# warning so the operator can decide whether to exclude the well.
ACCEPTABLE_SKIP_REASONS: frozenset[str] = frozenset(
	{
		"no_qualifying_units",
		"slay_disabled",
		"merge_units_disabled",
		"merge_disabled",
		"phase_disabled",
		"bombcell_label_not_invoked_by_merge_stage",
	}
)


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
class SkipRecord:
	"""One skipped phase / well-marker entry."""
	name: str  # phase name, or "<stage>" for a stage-level marker
	reason: str | None
	acceptable: bool


@dataclass(frozen=True)
class WellStatus:
	well_id: str
	stage_done: bool
	phase_done: dict[str, bool] = field(default_factory=dict)
	skip_records: tuple[SkipRecord, ...] = field(default_factory=tuple)

	@property
	def has_skips(self) -> bool:
		return bool(self.skip_records)

	@property
	def has_unacceptable_skip(self) -> bool:
		return any(not record.acceptable for record in self.skip_records)


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


def _classify_skip_reason(reason: str | None) -> bool:
	"""True if a skip reason is in the known-acceptable set."""
	if reason is None:
		return False
	return str(reason).strip() in ACCEPTABLE_SKIP_REASONS


def _read_marker_skip(marker_path: Path, *, label: str) -> SkipRecord | None:
	"""Return a SkipRecord if the marker file declares a non-ok status.

	Returns None when:
	  - the marker doesn't exist (caller already knows: stage_done=False),
	  - the marker exists but the JSON has status missing or status="ok",
	  - the JSON is malformed (treat as ok-status to avoid false positives).
	"""
	if not marker_path.is_file():
		return None
	try:
		with marker_path.open("r", encoding="utf-8") as fh:
			payload = json.load(fh)
	except (json.JSONDecodeError, OSError):
		return None
	if not isinstance(payload, dict):
		return None
	status = payload.get("status")
	if status is None or str(status).strip().lower() == "ok":
		return None
	reason_raw = payload.get("reason")
	reason = str(reason_raw).strip() if reason_raw is not None else None
	# status=="error" / "failed" / anything else non-"ok" non-"skipped" → not acceptable.
	if str(status).strip().lower() == "skipped":
		acceptable = _classify_skip_reason(reason)
	else:
		acceptable = False
	return SkipRecord(name=label, reason=reason, acceptable=acceptable)


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
				well_marker_path = well_root.joinpath(*well_marker)
				stage_done = well_marker_path.is_file()
				phase_done: dict[str, bool] = {}
				skip_records: list[SkipRecord] = []

				# Stage-level skip (status="skipped" inside the well marker, with a reason).
				stage_skip = _read_marker_skip(well_marker_path, label=stage_name)
				if stage_skip is not None:
					skip_records.append(stage_skip)

				# Per-phase skip (only checked in verbose mode where we already walk the phase markers).
				for phase_name, relparts in phase_markers:
					phase_marker_path = well_root.joinpath(*relparts)
					phase_done[phase_name] = phase_marker_path.is_file()
					phase_skip = _read_marker_skip(phase_marker_path, label=phase_name)
					if phase_skip is not None:
						skip_records.append(phase_skip)
				wells_out.append(
					WellStatus(
						well_id=well_id,
						stage_done=stage_done,
						phase_done=phase_done,
						skip_records=tuple(skip_records),
					)
				)
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


def _format_skip_annotation(well: WellStatus) -> str:
	"""Compact skip annotation for default-mode tables.

	Examples:
	  - "merge_SLAy=no_qualifying_units(ok)"
	  - "merge_SLAy=manual_block(!!)"      # unacceptable skip → !!
	"""
	if not well.skip_records:
		return ""
	parts: list[str] = []
	for record in well.skip_records:
		reason = record.reason if record.reason else "(no reason)"
		tag = "ok" if record.acceptable else "!!"
		parts.append(f"{record.name}={reason}({tag})")
	return "; ".join(parts)


def format_default_tables(report: StatusReport) -> str:
	"""Per-stage table: dataset × wells (ok/total + missing well list + skip flags).

	The trailing `skipped_wells` column lists wells whose marker file says
	`status="skipped"`. An `(ok)` tag follows known-acceptable reasons
	(e.g. `no_qualifying_units`); `(!!)` flags reasons outside the
	allowlist so the operator can decide whether to exclude the well.
	"""
	lines: list[str] = []
	lines.append(f"# status scan vs {report.output_root}")
	lines.append(f"# runtime: {report.runtime_yml}")
	lines.append(f"# data:    {report.data_yml}")
	lines.append(f"# acceptable skip reasons: {', '.join(sorted(ACCEPTABLE_SKIP_REASONS))}")

	for stage in report.stages:
		lines.append("")
		marker_path = "/".join(STAGE_WELL_MARKER[stage.stage])
		lines.append(f"=== {stage.stage} ===")
		lines.append(f"# marker: <well>/{marker_path}")
		lines.append(
			f"{'idx':>3}  {'DIV':>3}  {'dataset':<30}  {'wells (ok/total)':<18}  "
			f"{'missing_wells':<28}  skipped_wells"
		)
		lines.append(
			f"{'---':>3}  {'---':>3}  {'-'*30:<30}  {'-'*18:<18}  {'-'*28:<28}  -------------"
		)
		total_ok = 0
		total_wells = 0
		incomplete_count = 0
		skip_count = 0
		unacceptable_skip_count = 0
		for dataset in stage.datasets:
			total = len(dataset.wells)
			ok = sum(1 for well in dataset.wells if well.stage_done)
			missing = [well.well_id for well in dataset.wells if not well.stage_done]
			skipped_wells = [well for well in dataset.wells if well.has_skips]
			div_str = str(dataset.div) if dataset.div is not None and dataset.div >= 0 else "-"
			missing_str = ",".join(missing) if missing else "-"
			if skipped_wells:
				skip_parts = [
					f"{well.well_id}[{_format_skip_annotation(well)}]" for well in skipped_wells
				]
				skip_str = "; ".join(skip_parts)
				skip_count += len(skipped_wells)
				unacceptable_skip_count += sum(
					1 for well in skipped_wells if well.has_unacceptable_skip
				)
			else:
				skip_str = "-"
			complete_tag = "" if missing else "  COMPLETE"
			lines.append(
				f"{dataset.index:>3}  {div_str:>3}  {dataset.short_label:<30}  "
				f"{ok}/{total:<16}  {missing_str:<28}  {skip_str}{complete_tag}"
			)
			total_ok += ok
			total_wells += total
			if missing:
				incomplete_count += 1
		summary = (
			f"# {incomplete_count} of {len(stage.datasets)} datasets incomplete · "
			f"{total_ok}/{total_wells} wells done"
		)
		if skip_count:
			summary += (
				f" · {skip_count} skipped-but-complete wells "
				f"({unacceptable_skip_count} flagged !!)"
			)
		lines.append(summary)
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
		lines.append(
			"# skips column: '<phase>=<reason>(ok|!!)' for each phase whose marker has status=skipped"
		)
		phases_header = "".join(f"{i % 10}" for i in range(1, len(phase_names) + 1))
		header = (
			f"{'idx':>3}  {'DIV':>3}  {'dataset':<30}  {'well':<8}  done  "
			f"phases({phases_header})  skips"
		)
		lines.append(header)
		lines.append("-" * min(len(header), 200))
		for dataset in stage.datasets:
			div_str = str(dataset.div) if dataset.div is not None and dataset.div >= 0 else "-"
			for well in dataset.wells:
				done_glyph = "✓" if well.stage_done else "·"
				phase_glyphs = "".join(
					("✓" if well.phase_done.get(name, False) else "·") for name in phase_names
				)
				skip_str = _format_skip_annotation(well) if well.has_skips else "-"
				lines.append(
					f"{dataset.index:>3}  {div_str:>3}  {dataset.short_label:<30}  "
					f"{well.well_id:<8}  {done_glyph:>4}  {phase_glyphs}  {skip_str}"
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
