from __future__ import annotations

import csv
import importlib
import json
import logging
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from axon_reconstructor.pipeline.output_paths import compute_mea_analysis_output_dir
from axon_reconstructor.pipeline.stg2_spikesorting.runner import (
	SpikeSortingInputs as LegacySpikeSortingInputs,
	run_spikesorting_stage as run_legacy_spikesorting_stage,
)

from .models.inputs import SpikesortInputs
from .models.results import SpikesortMergeResult, SpikesortResult


LOGGER = logging.getLogger("axon_recon.spikesort")


def _write_json(path: Path, payload: dict) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cleanup_spikesort_outputs_for_force_restart(*, stage_output_root_dir: Path, um_kwargs: dict[str, Any] | None) -> list[str]:
	cleanup_names: set[str] = {
		"sorter_output",
		"analyzer_output",
	}

	if isinstance(um_kwargs, dict):
		output_subdir_name = str(um_kwargs.get("output_subdir_name", "")).strip().lstrip("/")
		throughput_subdir_name = str(um_kwargs.get("throughput_subdir_name", "")).strip().lstrip("/")
		if output_subdir_name:
			cleanup_names.add(output_subdir_name)
		if throughput_subdir_name:
			cleanup_names.add(throughput_subdir_name)

	removed_paths: list[str] = []
	for rel_name in sorted(cleanup_names):
		target = (stage_output_root_dir / rel_name).resolve()
		if not target.exists():
			continue
		shutil.rmtree(target, ignore_errors=True)
		removed_paths.append(str(target))

	return removed_paths


def _resolve_under_well(*, well_out_dir: Path, relpath: str) -> Path:
	candidate = Path(str(relpath).strip()).expanduser()
	if candidate.is_absolute():
		return candidate.resolve()
	return (well_out_dir / str(relpath).lstrip("/")).resolve()


def _resolve_under_spikesort_output_root(*, well_out_dir: Path, output_rel_root: str, relpath: str) -> Path:
	candidate = Path(str(relpath).strip()).expanduser()
	if candidate.is_absolute():
		return candidate.resolve()

	stage_root_rel = str(output_rel_root).strip().lstrip("/") or "spikesort_outputs"
	resolved_stage_root = (well_out_dir / stage_root_rel).resolve()
	rel = str(relpath).strip().lstrip("/")

	# Backward-compat: allow relpaths that still include output_rel_root prefix.
	if rel == stage_root_rel:
		rel = ""
	elif rel.startswith(f"{stage_root_rel}/"):
		rel = rel[len(stage_root_rel) + 1 :]

	if not rel:
		return resolved_stage_root
	return (resolved_stage_root / rel).resolve()


def _install_numpy_cupy_fallback_module() -> None:
	import numpy as np

	shim = ModuleType("cupy")
	shim.array = np.array  # type: ignore[attr-defined]
	shim.asarray = np.asarray  # type: ignore[attr-defined]
	shim.asnumpy = np.asarray  # type: ignore[attr-defined]
	shim.mean = np.mean  # type: ignore[attr-defined]
	shim.zeros = np.zeros  # type: ignore[attr-defined]
	shim.float32 = np.float32  # type: ignore[attr-defined]
	shim.ndarray = np.ndarray  # type: ignore[attr-defined]
	sys.modules["cupy"] = shim


def _import_slay_run_function(*, package_root: str | None, allow_numpy_fallback: bool) -> Callable[[dict[str, Any]], None]:
	def _patch_parse_kilosort_params(module: Any) -> None:
		original = getattr(module, "parse_kilosort_params", None)
		if not callable(original):
			return

		def _patched(args: dict[str, Any]) -> dict[str, Any]:
			import os

			ks_folder = str(args.get("KS_folder", "")).strip()
			if not ks_folder:
				return original(args)

			ksparam_path = os.path.join(ks_folder, "params.py")
			ksparams: dict[str, Any] = {}
			with open(ksparam_path, "r", encoding="utf-8") as f:
				for line in f:
					if "=" not in line:
						continue
					key, value = line.split("=", 1)
					ksparams[str(key).strip()] = eval(str(value).strip())

			dat_path = ksparams.pop("dat_path", None)
			if isinstance(dat_path, (list, tuple)):
				dat_path = (dat_path[0] if len(dat_path) > 0 else None)
			if dat_path is not None:
				dat_path_s = str(dat_path)
				if os.path.isabs(dat_path_s):
					ksparams["data_filepath"] = dat_path_s
				else:
					ksparams["data_filepath"] = os.path.join(ks_folder, dat_path_s)
			if "n_channels_dat" in ksparams:
				ksparams["n_chan"] = ksparams.pop("n_channels_dat")
			args.update(ksparams)
			return args

		setattr(module, "parse_kilosort_params", _patched)

	search_paths: list[str] = []
	if package_root:
		base = Path(package_root).expanduser().resolve()
		search_paths.extend([str(base / "src"), str(base)])
	for path in reversed(search_paths):
		if path and path not in sys.path:
			sys.path.insert(0, path)

	try:
		module = importlib.import_module("slay.run")
		_patch_parse_kilosort_params(module)
		run_slay = getattr(module, "run_slay", None)
		if callable(run_slay):
			return run_slay
		raise RuntimeError("SLAy import succeeded but slay.run.run_slay is not callable")
	except ModuleNotFoundError as exc:
		if str(getattr(exc, "name", "")) != "cupy" or not bool(allow_numpy_fallback):
			raise
		_install_numpy_cupy_fallback_module()
		module = importlib.import_module("slay.run")
		_patch_parse_kilosort_params(module)
		run_slay = getattr(module, "run_slay", None)
		if callable(run_slay):
			return run_slay
		raise RuntimeError("SLAy import succeeded with numpy fallback but run_slay is not callable")


def _normalize_cluster_id(raw: Any) -> str:
	try:
		return str(int(raw))
	except Exception:
		return str(raw)


def _load_metrics_lookup(metrics_tsv_path: Path) -> dict[tuple[str, str], dict[str, str]]:
	if not metrics_tsv_path.exists():
		return {}
	lookup: dict[tuple[str, str], dict[str, str]] = {}
	with metrics_tsv_path.open("r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f, delimiter="\t")
		for row in reader:
			a = _normalize_cluster_id(row.get("Cluster 1", ""))
			b = _normalize_cluster_id(row.get("Cluster 2", ""))
			if not a or not b:
				continue
			key = tuple(sorted((a, b)))
			lookup[key] = dict(row)
	return lookup


def _build_candidate_pair_rows(
	*,
	merge_groups: dict[str, list[Any]],
	metrics_lookup: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, str]]:
	rows: list[dict[str, str]] = []
	for group_id, members_raw in merge_groups.items():
		if not isinstance(members_raw, list):
			continue
		members = [_normalize_cluster_id(m) for m in members_raw]
		if len(members) < 2:
			continue
		members_sorted = sorted(members, key=lambda token: int(token) if token.isdigit() else token)
		group_members = "|".join(members_sorted)
		for idx_a in range(len(members_sorted)):
			for idx_b in range(idx_a + 1, len(members_sorted)):
				a = members_sorted[idx_a]
				b = members_sorted[idx_b]
				metrics = metrics_lookup.get(tuple(sorted((a, b))), {})
				rows.append(
					{
						"group_id": str(group_id),
						"group_size": str(len(members_sorted)),
						"group_members": group_members,
						"cluster_a": a,
						"cluster_b": b,
						"final_metric": str(metrics.get("Final Metric", "")),
						"similarity": str(metrics.get("Similarity", "")),
						"xcorr_significance": str(metrics.get("Cross-correlation Significance", "")),
						"refractory_penalty": str(metrics.get("Refractory Period Penalty", "")),
					}
				)
	return rows


def _write_candidate_pairs_tsv(path: Path, rows: list[dict[str, str]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	fieldnames = [
		"group_id",
		"group_size",
		"group_members",
		"cluster_a",
		"cluster_b",
		"final_metric",
		"similarity",
		"xcorr_significance",
		"refractory_penalty",
	]
	with path.open("w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
		writer.writeheader()
		for row in rows:
			writer.writerow(row)


def run_spikesort_merge_stage(
	*,
	h5_path: Path,
	stream_id: str,
	mea_output_root: Path,
	output_rel_root: str,
	stage_config: Any,
	force_restart: bool,
) -> SpikesortMergeResult:
	well_out_dir = compute_mea_analysis_output_dir(
		output_root=mea_output_root,
		data_file=h5_path,
		well=stream_id,
	)
	stage_output_root_dir = _resolve_under_well(well_out_dir=well_out_dir, relpath=str(output_rel_root).strip() or "spikesort_outputs")
	merge_out_dir = _resolve_under_spikesort_output_root(
		well_out_dir=well_out_dir,
		output_rel_root=output_rel_root,
		relpath=str(getattr(stage_config, "slay_relpath", "SLAy_outputs")),
	)
	slay_delete_outputs_on_force_restart = bool(getattr(stage_config, "slay_delete_outputs_on_force_restart", True))
	if bool(force_restart) and bool(slay_delete_outputs_on_force_restart) and merge_out_dir.exists():
		shutil.rmtree(merge_out_dir, ignore_errors=True)
	merge_out_dir.mkdir(parents=True, exist_ok=True)
	summary_json = merge_out_dir / "slay_merge_summary.json"

	slay_enabled = bool(getattr(stage_config, "slay_enabled", False))
	if not slay_enabled:
		_write_json(
			summary_json,
			{
				"status": "skipped",
				"reason": "slay_disabled",
				"well_out_dir": str(well_out_dir),
				"stage_output_root_dir": str(stage_output_root_dir),
				"merge_out_dir": str(merge_out_dir),
			},
		)
		return SpikesortMergeResult(
			well_out_dir=well_out_dir,
			merge_out_dir=merge_out_dir,
			summary_json=summary_json,
			outputs={
				"summary_json": str(summary_json),
			},
		)

	configured_ks_relpath = getattr(stage_config, "slay_sorter_output_relpath", None)
	default_ks_relpath = "sorter_output"
	ks_dir = _resolve_under_spikesort_output_root(
		well_out_dir=well_out_dir,
		output_rel_root=output_rel_root,
		relpath=(str(configured_ks_relpath) if configured_ks_relpath else default_ks_relpath),
	)
	if not (ks_dir / "params.py").exists() and (ks_dir / "sorter_output" / "params.py").exists():
		ks_dir = (ks_dir / "sorter_output").resolve()
	if not (ks_dir / "params.py").exists():
		raise FileNotFoundError(
			"SLAy requires a Kilosort folder containing params.py. "
			f"Resolved path: {ks_dir}. Configure stages.spikesort.phases.merge_units.SLAy.sorter_output_relpath if needed."
		)

	run_slay = _import_slay_run_function(
		package_root=getattr(stage_config, "slay_package_root", None),
		allow_numpy_fallback=bool(getattr(stage_config, "slay_allow_numpy_fallback", True)),
	)

	run_output_json = merge_out_dir / str(getattr(stage_config, "slay_output_json_relpath", "run-output.json"))
	run_output_json.parent.mkdir(parents=True, exist_ok=True)
	run_args: dict[str, Any] = {
		"KS_folder": str(ks_dir),
		"auto_accept_merges": bool(getattr(stage_config, "slay_auto_accept_merges", False)),
		"plot_merges": bool(getattr(stage_config, "slay_plot_merges", False)),
		"output_json": str(run_output_json),
	}
	extra_params = getattr(stage_config, "slay_params", None)
	if isinstance(extra_params, dict):
		run_args.update(dict(extra_params))
	run_args["KS_folder"] = str(ks_dir)
	run_args["auto_accept_merges"] = bool(getattr(stage_config, "slay_auto_accept_merges", False))
	run_args["plot_merges"] = bool(getattr(stage_config, "slay_plot_merges", False))
	run_args["output_json"] = str(run_output_json)

	run_slay(run_args)

	automerge_dir = (ks_dir / "automerge").resolve()
	automerge_snapshot_dir = merge_out_dir / "automerge"
	if bool(getattr(stage_config, "slay_copy_automerge_artifacts", True)) and automerge_dir.exists():
		if automerge_snapshot_dir.exists():
			shutil.rmtree(automerge_snapshot_dir, ignore_errors=True)
		shutil.copytree(automerge_dir, automerge_snapshot_dir)

	merge_groups_src = automerge_dir / "new2old.json"
	merge_groups_payload: dict[str, list[Any]] = {}
	if merge_groups_src.exists():
		merge_groups_payload = json.loads(merge_groups_src.read_text(encoding="utf-8"))

	merge_groups_out = merge_out_dir / str(getattr(stage_config, "slay_merge_groups_relpath", "recommended_merge_groups.json"))
	_write_json(
		merge_groups_out,
		{
			"n_groups": int(len(merge_groups_payload)),
			"merge_groups": merge_groups_payload,
		},
	)

	metrics_lookup = _load_metrics_lookup(automerge_dir / "metrics.tsv")
	candidate_rows = _build_candidate_pair_rows(
		merge_groups=merge_groups_payload,
		metrics_lookup=metrics_lookup,
	)
	candidates_out = merge_out_dir / str(getattr(stage_config, "slay_candidate_pairs_relpath", "recommended_merge_candidates.tsv"))
	_write_candidate_pairs_tsv(candidates_out, candidate_rows)

	outputs: dict[str, str] = {
		"summary_json": str(summary_json),
		"slay.run_output_json": str(run_output_json),
		"slay.recommended_merge_groups_json": str(merge_groups_out),
		"slay.recommended_merge_candidates_tsv": str(candidates_out),
	}
	if automerge_dir.exists():
		outputs["slay.automerge_dir"] = str(automerge_dir)
	if automerge_snapshot_dir.exists():
		outputs["slay.automerge_snapshot_dir"] = str(automerge_snapshot_dir)

	_write_json(
		summary_json,
		{
			"status": "ok",
			"well_out_dir": str(well_out_dir),
			"stage_output_root_dir": str(stage_output_root_dir),
			"merge_out_dir": str(merge_out_dir),
			"ks_dir": str(ks_dir),
			"slay_enabled": bool(slay_enabled),
			"n_merge_groups": int(len(merge_groups_payload)),
			"n_candidate_pairs": int(len(candidate_rows)),
			"run_args": {
				"KS_folder": str(ks_dir),
				"auto_accept_merges": bool(run_args.get("auto_accept_merges", False)),
				"plot_merges": bool(run_args.get("plot_merges", False)),
				"output_json": str(run_output_json),
			},
			"force_restart": bool(force_restart),
			"slay_delete_outputs_on_force_restart": bool(slay_delete_outputs_on_force_restart),
			"outputs": outputs,
		},
	)

	return SpikesortMergeResult(
		well_out_dir=well_out_dir,
		merge_out_dir=merge_out_dir,
		summary_json=summary_json,
		outputs=outputs,
	)


def run_spikesort_stage(inputs: SpikesortInputs) -> SpikesortResult:
	well_out_dir = compute_mea_analysis_output_dir(
		output_root=inputs.mea_output_root,
		data_file=inputs.h5_path,
		well=inputs.stream_id,
	)

	stage_output_root_dir = _resolve_under_well(
		well_out_dir=well_out_dir,
		relpath=(str(inputs.output_rel_root).strip() or "spikesort_outputs"),
	)
	stage_output_root_dir.mkdir(parents=True, exist_ok=True)
	summary_json = stage_output_root_dir / "spikesort_summary.json"
	effective_force_restart = bool(inputs.force_restart or inputs.force_replot)

	if not bool(inputs.sort_enabled):
		_write_json(
			summary_json,
			{
				"status": "skipped",
				"reason": "sort_disabled",
				"h5_path": str(inputs.h5_path),
				"stream_id": str(inputs.stream_id),
				"well_out_dir": str(well_out_dir),
				"spikesort_out_dir": str(stage_output_root_dir),
				"output_rel_root": str(inputs.output_rel_root),
				"inputs": {
					"sort_enabled": bool(inputs.sort_enabled),
					"sort_delete_outputs_on_force_restart": bool(inputs.sort_delete_outputs_on_force_restart),
					"force_restart": bool(inputs.force_restart),
					"force_replot": bool(inputs.force_replot),
				},
				"outputs": {
					"summary_json": str(summary_json),
				},
			},
		)
		return SpikesortResult(
			well_out_dir=well_out_dir,
			spikesort_out_dir=stage_output_root_dir,
			summary_json=summary_json,
			outputs={
				"summary_json": str(summary_json),
			},
		)

	removed_on_force_restart: list[str] = []
	if bool(effective_force_restart) and bool(inputs.sort_delete_outputs_on_force_restart):
		removed_on_force_restart = _cleanup_spikesort_outputs_for_force_restart(
			stage_output_root_dir=stage_output_root_dir,
			um_kwargs=inputs.um_kwargs,
		)

	legacy_inputs = LegacySpikeSortingInputs(
		h5_path=inputs.h5_path,
		stream_id=inputs.stream_id,
		mea_output_root=inputs.mea_output_root,
		output_subdir_after_well=(str(inputs.output_rel_root).strip() or "spikesort_outputs"),
		preprocess_concat_recording_relpath=inputs.preprocess_concat_recording_relpath,
		log_enabled=bool(inputs.logging_enabled),
		log_verbose=bool(inputs.logging_verbose),
		log_file_override=inputs.logging_file_relpath,
		sorter=inputs.sorter,
		docker_image=inputs.docker_image,
		recording_num=inputs.recording_num,
		verbose=inputs.verbose,
		ks_batch_duration_s=inputs.ks_batch_duration_s,
		ks_batch_size=inputs.ks_batch_size,
		ks_th_universal=inputs.ks_th_universal,
		ks_th_learned=inputs.ks_th_learned,
		ks_th_single_ch=inputs.ks_th_single_ch,
		ks_cluster_downsampling=inputs.ks_cluster_downsampling,
		ks_nearest_chans=inputs.ks_nearest_chans,
		ks_max_channel_distance=inputs.ks_max_channel_distance,
		n_jobs=inputs.n_jobs,
		chunk_duration=inputs.chunk_duration,
		cuda_visible_devices=inputs.cuda_visible_devices,
		run_analyzer=inputs.run_analyzer,
		run_reports=inputs.run_reports,
		plot_mode=inputs.plot_mode,
		plot_debug=inputs.plot_debug,
		raster_sort=inputs.raster_sort,
		fixed_y=inputs.fixed_y,
		no_curation=inputs.no_curation,
		export_to_phy=inputs.export_to_phy,
		force_rerun_analyzer=inputs.force_rerun_analyzer,
		um_kwargs=(dict(inputs.um_kwargs) if isinstance(inputs.um_kwargs, dict) else None),
		am_kwargs=(dict(inputs.am_kwargs) if isinstance(inputs.am_kwargs, dict) else None),
		option_kwargs=(dict(inputs.option_kwargs) if isinstance(inputs.option_kwargs, dict) else None),
		force_restart=bool(effective_force_restart),
		resume_from=inputs.resume_from,
	)
	legacy_outputs = run_legacy_spikesorting_stage(inputs=legacy_inputs, logger=LOGGER)

	legacy_out_dir = Path(legacy_outputs.output_dir)
	spikesort_out_dir = legacy_out_dir
	summary_json = spikesort_out_dir / "spikesort_summary.json"

	outputs: dict[str, str] = {
		"legacy.spikesort_out_dir": str(legacy_out_dir),
		"recording_dir": str(legacy_outputs.recording_dir),
		"sorter_output_dir": str(legacy_outputs.sorter_output_dir),
		"analyzer_dir": str(legacy_outputs.analyzer_dir),
	}
	if legacy_outputs.merged_sorting_dir is not None:
		outputs["merged_sorting_dir"] = str(legacy_outputs.merged_sorting_dir)
	if legacy_outputs.merged_sorter_output_dir is not None:
		outputs["merged_sorter_output_dir"] = str(legacy_outputs.merged_sorter_output_dir)

	_write_json(
		summary_json,
		{
			"h5_path": str(inputs.h5_path),
			"stream_id": str(inputs.stream_id),
			"well_out_dir": str(well_out_dir),
			"spikesort_out_dir": str(spikesort_out_dir),
			"legacy_spikesort_out_dir": str(legacy_out_dir),
			"output_rel_root": str(inputs.output_rel_root),
			"inputs": {
				"preprocess_concat_recording_relpath": inputs.preprocess_concat_recording_relpath,
				"logging_enabled": bool(inputs.logging_enabled),
				"logging_verbose": bool(inputs.logging_verbose),
				"logging_file_relpath": inputs.logging_file_relpath,
				"sorter": str(inputs.sorter),
				"docker_image": inputs.docker_image,
				"recording_num": str(inputs.recording_num),
				"verbose": bool(inputs.verbose),
				"n_jobs": inputs.n_jobs,
				"chunk_duration": inputs.chunk_duration,
				"cuda_visible_devices": inputs.cuda_visible_devices,
				"run_analyzer": bool(inputs.run_analyzer),
				"run_reports": bool(inputs.run_reports),
				"sort_enabled": bool(inputs.sort_enabled),
				"sort_delete_outputs_on_force_restart": bool(inputs.sort_delete_outputs_on_force_restart),
				"plot_enabled": bool(inputs.plot_enabled),
				"plot_mode": str(inputs.plot_mode),
				"plot_debug": bool(inputs.plot_debug),
				"raster_sort": inputs.raster_sort,
				"fixed_y": bool(inputs.fixed_y),
				"no_curation": bool(inputs.no_curation),
				"export_to_phy": bool(inputs.export_to_phy),
				"force_restart": bool(inputs.force_restart),
				"force_replot": bool(inputs.force_replot),
				"effective_force_restart": bool(effective_force_restart),
				"resume_from": inputs.resume_from,
			},
			"cleanup": {
				"removed_on_force_restart": list(removed_on_force_restart),
			},
			"outputs": outputs,
		},
	)

	return SpikesortResult(
		well_out_dir=well_out_dir,
		spikesort_out_dir=spikesort_out_dir,
		summary_json=summary_json,
		outputs=outputs,
	)
