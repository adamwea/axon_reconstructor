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
from collections import Counter
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
		"plots_disabled",
	}
)

# Phase summary "status" values that mean the phase completed normally and
# should NOT appear in the skips column. Different phases historically emit
# different verbs ("ok", "success", "completed"); accept all of them.
_HEALTHY_PHASE_STATUSES: frozenset[str] = frozenset({"ok", "success", "completed"})


STAGE_ORDER: tuple[str, ...] = ("preprocess", "spikesort", "reconstruct", "analysis")


# KS-raw labels: pull from the snapshot dir rather than the canonical
# sorter_output because bombcell + merge_SLAy both mutate the canonical
# directory in place. The snapshot is the only pristine record of KS4's
# original cluster_KSLabel.tsv that survives those mutations.
SORTER_OUTPUT_KS_LABEL_TSV: tuple[str, ...] = (
	"spikesort_outputs",
	"sorter_output_snapshot",
	"cluster_KSLabel.tsv",
)

# Bombcell persists its labeling as a json artifact independent of the
# canonical cluster_group.tsv it writes. We read counts from here so the
# numbers survive any subsequent restore_sorter_output operation.
BOMBCELL_LABELS_JSON: tuple[str, ...] = (
	"spikesort_outputs",
	"bombcell_label_outputs",
	"bombcell_labels.json",
)

# Sort-stage completion marker — used as the freshness reference for
# downstream bombcell / merge_SLAy artifacts.
SPIKESORT_SUMMARY_JSON: tuple[str, ...] = (
	"spikesort_outputs",
	"spikesort_summary.json",
)

# merge_SLAy persists per-merge groupings here. Combined with the bombcell
# per-unit labels, this is enough to reconstruct what SLAy's post-merge
# labels would be (via accept_merge()'s mode-of-input rule) without
# depending on the canonical cluster_group.tsv that SLAy mutates.
MERGE_SLAY_UNIT_DIFF_FLAT_JSON: tuple[str, ...] = (
	"spikesort_outputs",
	"merge_SLAy",
	"unit_diff_map_flat.json",
)

# Labels SLAy treats as "good for downstream analysis" — used to count
# merges that consume a good unit but produce a non-good merged unit.
SLAY_GOOD_LIKE_LABELS: frozenset[str] = frozenset({"good", "non_soma_good"})


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
		("axon_velocity_gtrs", ("recon_outputs", "context", "axon_velocity_gtrs_summary.json")),
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
class LabelColumn:
	"""Per-well label counts for one stage in the pipeline (KS / bombcell / SLAy).

	``status`` is "ok" when counts are populated and current, or a short token
	like ``snapshot_missing`` / ``bombcell_stale`` describing why ``counts`` is
	empty. ``extras`` carries per-column scalars that aren't label counts (used
	by the SLAy column to surface merge counts).
	"""
	counts: dict[str, int] = field(default_factory=dict)
	status: str = "ok"
	extras: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class WellStatus:
	well_id: str
	stage_done: bool
	phase_done: dict[str, bool] = field(default_factory=dict)
	skip_records: tuple[SkipRecord, ...] = field(default_factory=tuple)
	ks_labels: LabelColumn = field(default_factory=LabelColumn)
	bombcell_labels: LabelColumn = field(default_factory=LabelColumn)
	slay_labels: LabelColumn = field(default_factory=LabelColumn)

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


def _chip_from_short_label(short_label: str) -> str:
	"""Extract the chip id (middle component) from a `<date>/<chip>/<rec>` label.

	Returns the original label when the layout doesn't match — keeps sort
	keys total without erroring on non-canonical paths.
	"""
	parts = str(short_label).split("/")
	if len(parts) >= 3:
		return parts[1]
	return str(short_label)


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


def _read_labels_from_tsv(tsv_path: Path) -> dict[str, str]:
	"""Parse a TSV with header + `<id><TAB><label>` rows into a unit→label map."""
	out: dict[str, str] = {}
	try:
		with tsv_path.open("r", encoding="utf-8") as fh:
			next(fh, None)  # header row
			for line in fh:
				parts = line.rstrip("\n").split("\t")
				if len(parts) < 2:
					continue
				unit_id = parts[0].strip()
				label = parts[1].strip()
				if not unit_id or not label:
					continue
				out[unit_id] = label
	except OSError:
		return {}
	return out


def _count_labels_from_tsv(tsv_path: Path) -> dict[str, int]:
	"""Parse a TSV with header + `<id><TAB><label>` rows into per-label counts."""
	counts: dict[str, int] = {}
	for label in _read_labels_from_tsv(tsv_path).values():
		counts[label] = counts.get(label, 0) + 1
	return counts


def _mode_alphabetical(labels: list[str]) -> str:
	"""First-alphabetically among the most-frequent labels — mirrors pandas
	``mode().values[0]`` semantics that SLAy's ``accept_merge`` relies on."""
	if not labels:
		return ""
	counter = Counter(labels)
	max_count = max(counter.values())
	modes = sorted(label for label, count in counter.items() if count == max_count)
	return modes[0]


def _read_ks_label_column(well_root: Path) -> LabelColumn:
	"""KS-raw labels from sorter_output_snapshot/cluster_KSLabel.tsv.

	The snapshot is the only pristine record once bombcell / merge_SLAy have
	mutated the canonical sorter_output. If the snapshot is missing the
	column surfaces ``snapshot_missing`` rather than empty counts so the
	operator knows the data is not just absent of clusters.
	"""
	tsv_path = well_root.joinpath(*SORTER_OUTPUT_KS_LABEL_TSV)
	if not tsv_path.is_file():
		return LabelColumn(status="snapshot_missing")
	return LabelColumn(counts=_count_labels_from_tsv(tsv_path))


def _read_bombcell_label_column(well_root: Path) -> LabelColumn:
	"""Bombcell label counts from bombcell_labels.json's ``counts_by_label``.

	Stale flag fires when the sort-stage marker is newer than bombcell's
	output (i.e. spikesort_summary.json mtime > bombcell_labels.json mtime),
	which means a re-sort happened after the bombcell pass and the cached
	counts no longer reflect the current sorter_output.
	"""
	bc_path = well_root.joinpath(*BOMBCELL_LABELS_JSON)
	if not bc_path.is_file():
		return LabelColumn(status="bombcell_missing")
	sort_summary = well_root.joinpath(*SPIKESORT_SUMMARY_JSON)
	if (
		sort_summary.is_file()
		and sort_summary.stat().st_mtime > bc_path.stat().st_mtime
	):
		return LabelColumn(status="bombcell_stale")
	try:
		payload = json.loads(bc_path.read_text(encoding="utf-8"))
	except (json.JSONDecodeError, OSError):
		return LabelColumn(status="bombcell_unreadable")
	counts_raw = payload.get("counts_by_label", {}) if isinstance(payload, dict) else {}
	counts: dict[str, int] = {
		str(label): int(count) for label, count in counts_raw.items()
	}
	return LabelColumn(counts=counts)


def _read_slay_label_column(well_root: Path) -> LabelColumn:
	"""Reconstruct post-SLAy label counts + per-merge stats.

	SLAy mutates the canonical cluster_group.tsv in place, so we can't trust
	whatever is on disk now (a restore_sorter_output between SLAy runs would
	wipe the post-SLAy labels). Instead we replay SLAy's ``accept_merge``
	rule: for each merge group in ``unit_diff_map_flat.json``, the post-merge
	unit's label is the mode (first-alphabetical) of its input labels;
	absorbed input units drop out of the post-merge unit set.

	Input-label source priority:
	  1. bombcell_labels.json's labels_by_unit, but ONLY when the bombcell
	     artifact is newer than the sort snapshot (i.e. bombcell ran in the
	     current sort cycle and actually fed labels into SLAy).
	  2. sorter_output_snapshot/cluster_KSLabel.tsv — used when bombcell did
	     not run in the current cycle, or when bombcell_labels.json is older
	     than the snapshot (orphaned artifact from a previous run where
	     bombcell was enabled — SLAy in the current run actually saw KS labels
	     off cluster_group.tsv, so the snapshot is the faithful proxy).

	``extras`` reports:
	  - ``merges``: total merge groups SLAy applied
	  - ``good_loss``: count of merges where at least one good/non_soma_good
	    input did NOT survive to the merged output's label

	Stale fires when either the source label artifact or the sort marker is
	newer than SLAy's flat map.
	"""
	flat_path = well_root.joinpath(*MERGE_SLAY_UNIT_DIFF_FLAT_JSON)
	if not flat_path.is_file():
		return LabelColumn(status="slay_missing")
	sort_summary = well_root.joinpath(*SPIKESORT_SUMMARY_JSON)
	flat_mtime = flat_path.stat().st_mtime
	sort_mtime = sort_summary.stat().st_mtime if sort_summary.is_file() else 0

	bc_path = well_root.joinpath(*BOMBCELL_LABELS_JSON)
	snapshot_tsv = well_root.joinpath(*SORTER_OUTPUT_KS_LABEL_TSV)

	labels_by_unit: dict[str, str] = {}
	upstream_mtime = sort_mtime
	# Only trust bombcell as the label source when it post-dates the sort snapshot.
	# A bombcell artifact older than the snapshot is a stale leftover from a previous
	# run whose bombcell phase has since been removed from the phase_sequence;
	# SLAy in the current run actually read KS labels off cluster_group.tsv, so
	# replaying with the stale bombcell labels would inject phantom "noise" units.
	bc_is_fresh = False
	if bc_path.is_file():
		bc_mtime = bc_path.stat().st_mtime
		snapshot_mtime = snapshot_tsv.stat().st_mtime if snapshot_tsv.is_file() else 0
		bc_is_fresh = bc_mtime >= snapshot_mtime
	if bc_is_fresh:
		try:
			bc_payload = json.loads(bc_path.read_text(encoding="utf-8"))
		except (json.JSONDecodeError, OSError):
			return LabelColumn(status="slay_unreadable")
		raw = bc_payload.get("labels_by_unit", {}) if isinstance(bc_payload, dict) else {}
		labels_by_unit = {str(uid): str(label) for uid, label in raw.items()}
		upstream_mtime = max(upstream_mtime, bc_path.stat().st_mtime)
	elif snapshot_tsv.is_file():
		labels_by_unit = _read_labels_from_tsv(snapshot_tsv)
		upstream_mtime = max(upstream_mtime, snapshot_tsv.stat().st_mtime)
	else:
		# No bombcell artifact and no snapshot → can't determine input labels.
		return LabelColumn(status="slay_no_labels")

	if upstream_mtime > flat_mtime:
		return LabelColumn(status="slay_stale")
	try:
		flat_payload = json.loads(flat_path.read_text(encoding="utf-8"))
	except (json.JSONDecodeError, OSError):
		return LabelColumn(status="slay_unreadable")
	groups = flat_payload.get("groups", []) if isinstance(flat_payload, dict) else []
	if not isinstance(groups, list):
		groups = []

	absorbed_unit_ids: set[str] = set()
	new_merged_labels: dict[str, str] = {}
	n_merges = 0
	n_merges_with_good_loss = 0
	for group in groups:
		if not isinstance(group, dict):
			continue
		pre_ids = [str(x) for x in (group.get("primary_pre_unit_ids", []) or [])]
		post_id = group.get("final_post_unit_id", None)
		if post_id is None or not pre_ids:
			continue
		post_id_str = str(post_id)
		pre_labels = [labels_by_unit.get(uid, "") for uid in pre_ids]
		pre_labels = [lbl for lbl in pre_labels if lbl]
		if not pre_labels:
			# Can't replay SLAy's mode rule without input labels; skip this group.
			continue
		n_merges += 1
		absorbed_unit_ids.update(pre_ids)
		mode_label = _mode_alphabetical(pre_labels)
		new_merged_labels[post_id_str] = mode_label
		# "good_loss" = the merge had at least one good/non_soma_good input but
		# the merged unit's inferred label is no longer good/non_soma_good.
		# We don't count the natural N->1 reduction (e.g. merging two good
		# units into one good unit) since that is the expected outcome of
		# merging duplicate templates.
		had_good_input = any(lbl in SLAY_GOOD_LIKE_LABELS for lbl in pre_labels)
		post_is_good = mode_label in SLAY_GOOD_LIKE_LABELS
		if had_good_input and not post_is_good:
			n_merges_with_good_loss += 1

	# Post-SLAy label distribution = surviving non-absorbed pre-units + the
	# new merged units' inferred labels.
	counts: dict[str, int] = {}
	for unit_id, label in labels_by_unit.items():
		if unit_id in absorbed_unit_ids:
			continue
		counts[label] = counts.get(label, 0) + 1
	for label in new_merged_labels.values():
		counts[label] = counts.get(label, 0) + 1

	return LabelColumn(
		counts=counts,
		extras={"merges": n_merges, "good_loss": n_merges_with_good_loss},
	)


def _read_marker_skip(marker_path: Path, *, label: str) -> SkipRecord | None:
	"""Return a SkipRecord if the marker file declares a non-ok status.

	Returns None when:
	  - the marker doesn't exist (caller already knows: stage_done=False),
	  - the marker exists but the JSON has status missing or status in
	    {"ok", "success", "completed"} (all treated as healthy completion),
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
	if status is None or str(status).strip().lower() in _HEALTHY_PHASE_STATUSES:
		return None
	reason_raw = payload.get("reason")
	reason = str(reason_raw).strip() if reason_raw is not None else None
	# status=="error" / "failed" / anything else non-healthy non-"skipped" → not acceptable.
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
				# Label columns only make sense for the spikesort stage. Other
				# stages keep the default-empty LabelColumns and the formatter
				# elides the three columns for them.
				if stage_name == "spikesort":
					ks_labels = _read_ks_label_column(well_root)
					bombcell_labels = _read_bombcell_label_column(well_root)
					slay_labels = _read_slay_label_column(well_root)
				else:
					ks_labels = LabelColumn()
					bombcell_labels = LabelColumn()
					slay_labels = LabelColumn()
				wells_out.append(
					WellStatus(
						well_id=well_id,
						stage_done=stage_done,
						phase_done=phase_done,
						skip_records=tuple(skip_records),
						ks_labels=ks_labels,
						bombcell_labels=bombcell_labels,
						slay_labels=slay_labels,
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


def _format_label_column(col: LabelColumn) -> str:
	"""Compact label-column summary.

	  - ok    → ``good:194,mua:132|t=326`` (and ``|merges=N,good_loss=M`` if extras)
	  - non-ok status (snapshot_missing / bombcell_stale / etc.) → the status
	    token verbatim so the operator sees *why* the counts are absent.
	"""
	if col.status != "ok":
		return col.status
	if not col.counts:
		return "-"
	parts = [f"{label}:{col.counts[label]}" for label in sorted(col.counts)]
	base = ",".join(parts) + f"|t={sum(col.counts.values())}"
	if col.extras:
		extras_str = ",".join(f"{k}={col.extras[k]}" for k in sorted(col.extras))
		base = f"{base}|{extras_str}"
	return base


def _aggregate_label_column(
	wells: Iterable[WellStatus], attr: str
) -> tuple[LabelColumn, dict[str, int]]:
	"""Sum counts/extras across wells; tally non-ok statuses separately.

	Returns ``(combined, status_tally)`` where ``status_tally`` maps each
	non-ok status to the number of wells in that state. Wells with status="ok"
	contribute to the combined counts; others only show up in the tally.
	"""
	counts: dict[str, int] = {}
	extras: dict[str, int] = {}
	status_tally: dict[str, int] = {}
	for well in wells:
		col: LabelColumn = getattr(well, attr)
		if col.status != "ok":
			status_tally[col.status] = status_tally.get(col.status, 0) + 1
			continue
		for label, count in col.counts.items():
			counts[label] = counts.get(label, 0) + count
		for key, value in col.extras.items():
			extras[key] = extras.get(key, 0) + value
	return LabelColumn(counts=counts, extras=extras), status_tally


def _format_label_column_aggregate(col: LabelColumn, status_tally: dict[str, int]) -> str:
	"""Aggregate format: ``<counts>[ + N {status}, M {status}]``."""
	base = _format_label_column(col)
	if not status_tally:
		return base
	tally_parts = [f"{count} {status}" for status, count in sorted(status_tally.items())]
	suffix = " [" + ", ".join(tally_parts) + "]"
	if base in {"-", *status_tally}:
		# When no wells were ok at all, the base is "-"; just show the tally.
		return suffix.strip(" []")
	return base + suffix


def _render_aligned_table(
	headers: list[str],
	rows: list[list[str]],
	*,
	aligns: list[str] | None = None,
	gap: int = 2,
) -> list[str]:
	"""Render a table sized to each column's widest cell.

	``aligns`` is one of ``"L"`` (left) or ``"R"`` (right) per column; defaults
	to all-left. Returns the lines: header, rule, and one row per ``rows``.
	"""
	n = len(headers)
	if aligns is None:
		aligns = ["L"] * n
	widths = [len(h) for h in headers]
	for row in rows:
		for i in range(min(n, len(row))):
			cell_len = len(row[i])
			if cell_len > widths[i]:
				widths[i] = cell_len

	def _fmt(cells: list[str]) -> str:
		parts: list[str] = []
		for i in range(n):
			cell = cells[i] if i < len(cells) else ""
			width = widths[i]
			if aligns[i] == "R":
				parts.append(cell.rjust(width))
			else:
				parts.append(cell.ljust(width))
		return (" " * gap).join(parts)

	out = [_fmt(headers), (" " * gap).join("-" * w for w in widths)]
	for row in rows:
		out.append(_fmt(row))
	return out


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


def format_default_tables(report: StatusReport, *, sort_by: str = "dataset") -> str:
	"""Per-stage table: dataset × wells (ok/total + missing well list + skip flags).

	The trailing `skipped_wells` column lists wells whose marker file says
	`status="skipped"`. An `(ok)` tag follows known-acceptable reasons
	(e.g. `no_qualifying_units`); `(!!)` flags reasons outside the
	allowlist so the operator can decide whether to exclude the well.

	``sort_by``: ``"dataset"`` (default — preserves the data-config order) or
	``"chip-well"`` (groups datasets by chip id, then orders within a chip
	by DIV and dataset index — useful for reading down one chip's evolution
	over time without hopping rows).
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
		label_cols = stage.stage == "spikesort"
		if label_cols:
			lines.append(
				"# ks_labels: from sorter_output_snapshot/cluster_KSLabel.tsv (KS4 raw)"
			)
			lines.append(
				"# bombcell: from bombcell_label_outputs/bombcell_labels.json "
				"(stale if sort newer than bombcell)"
			)
			lines.append(
				"# slay: post-merge labels reconstructed from merge_SLAy/unit_diff_map_flat.json "
				"+ input labels (bombcell if present, else KS-raw snapshot) via SLAy's mode-of-input rule; "
				"merges=N, good_loss=M (merges where a good/non_soma_good was absorbed but the merged label isn't)"
			)
		headers = ["idx", "DIV", "dataset", "wells", "missing_wells", "skipped_wells"]
		aligns = ["R", "R", "L", "L", "L", "L"]
		if label_cols:
			headers += ["ks_labels(agg)", "bombcell_labels(agg)", "slay_labels(agg)"]
			aligns += ["L", "L", "L"]
		# Trailing "status" column carries the COMPLETE tag when every well is done.
		headers.append("status")
		aligns.append("L")

		rows: list[list[str]] = []
		total_ok = 0
		total_wells = 0
		incomplete_count = 0
		skip_count = 0
		unacceptable_skip_count = 0
		datasets_iter = stage.datasets
		if str(sort_by) == "chip-well":
			datasets_iter = sorted(
				stage.datasets,
				key=lambda d: (
					_chip_from_short_label(d.short_label),
					d.div if d.div is not None and d.div >= 0 else -1,
					int(d.index),
				),
			)
		for dataset in datasets_iter:
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
			row: list[str] = [
				str(dataset.index),
				div_str,
				dataset.short_label,
				f"{ok}/{total}",
				missing_str,
				skip_str,
			]
			if label_cols:
				ks_col, ks_tally = _aggregate_label_column(dataset.wells, "ks_labels")
				bc_col, bc_tally = _aggregate_label_column(dataset.wells, "bombcell_labels")
				sl_col, sl_tally = _aggregate_label_column(dataset.wells, "slay_labels")
				row += [
					_format_label_column_aggregate(ks_col, ks_tally),
					_format_label_column_aggregate(bc_col, bc_tally),
					_format_label_column_aggregate(sl_col, sl_tally),
				]
			row.append("" if missing else "COMPLETE")
			rows.append(row)
			total_ok += ok
			total_wells += total
			if missing:
				incomplete_count += 1
		lines.extend(_render_aligned_table(headers, rows, aligns=aligns))
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


def format_verbose_tables(report: StatusReport, *, sort_by: str = "dataset") -> str:
	"""Per-stage table expanded to one row per (dataset, well) with a numbered
	phase legend. Each row's ``phases`` column is a compact glyph string where
	position N corresponds to phase N in the legend (1-indexed).

	``sort_by``: ``"dataset"`` (default — dataset-major, all wells of dataset 0
	then dataset 1, etc.) or ``"chip-well"`` (well-major within chip — group
	rows by ``(chip_id, well_id)`` and order each group by DIV / dataset
	index, so each chip-well's evolution over time reads down consecutive
	rows). Useful for visually watching unit counts grow/shrink for a single
	chip-well as the recording sequence progresses.
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
		label_cols = stage.stage == "spikesort"
		headers = ["idx", "DIV", "dataset", "well", "done", f"phases({phases_header})", "skips"]
		aligns = ["R", "R", "L", "L", "R", "L", "L"]
		if label_cols:
			headers += ["ks_labels", "bombcell_labels", "slay_labels"]
			aligns += ["L", "L", "L"]

		# Flatten to one row per (dataset, well) so we can re-sort the whole
		# stage by chip/well when --sort-by chip-well is requested.
		flat: list[tuple[DatasetStatus, WellStatus]] = [
			(dataset, well)
			for dataset in stage.datasets
			for well in dataset.wells
		]
		if str(sort_by) == "chip-well":
			flat.sort(
				key=lambda pair: (
					_chip_from_short_label(pair[0].short_label),
					str(pair[1].well_id),
					pair[0].div if pair[0].div is not None and pair[0].div >= 0 else -1,
					int(pair[0].index),
				)
			)
		rows: list[list[str]] = []
		for dataset, well in flat:
			div_str = str(dataset.div) if dataset.div is not None and dataset.div >= 0 else "-"
			done_glyph = "✓" if well.stage_done else "·"
			phase_glyphs = "".join(
				("✓" if well.phase_done.get(name, False) else "·") for name in phase_names
			)
			skip_str = _format_skip_annotation(well) if well.has_skips else "-"
			row: list[str] = [
				str(dataset.index),
				div_str,
				dataset.short_label,
				well.well_id,
				done_glyph,
				phase_glyphs,
				skip_str,
			]
			if label_cols:
				row += [
					_format_label_column(well.ks_labels),
					_format_label_column(well.bombcell_labels),
					_format_label_column(well.slay_labels),
				]
			rows.append(row)
		lines.extend(_render_aligned_table(headers, rows, aligns=aligns))
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
