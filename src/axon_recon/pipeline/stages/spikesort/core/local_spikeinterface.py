from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import shutil
from typing import Any

from axon_recon.pipeline.stages.preprocess.constants import (
	LEGACY_PREPROCESS_OUTPUTS_DIRNAME,
	PREPROCESS_OUTPUTS_DIRNAME,
)
from .debug_outputs import suppress_spikesort_external_debug_output

from ..models.inputs import SpikesortInputs


_FORBIDDEN_RUN_SORTER_KWARGS = {
	"container_image",
	"docker_image",
	"singularity_image",
	"use_docker",
	"use_singularity",
}


@dataclass(frozen=True)
class LocalSpikeInterfaceSortOutputs:
	recording_dir: Path
	sorter_output_dir: Path
	output_dir: Path
	analyzer_dir: Path
	merged_sorting_dir: Path | None = None
	merged_sorter_output_dir: Path | None = None


def _resolve_preprocess_dir(*, well_out_dir: Path) -> Path:
	canonical_dir = well_out_dir / PREPROCESS_OUTPUTS_DIRNAME
	if canonical_dir.exists():
		return canonical_dir
	legacy_stage_dir = well_out_dir / LEGACY_PREPROCESS_OUTPUTS_DIRNAME
	if legacy_stage_dir.exists():
		return legacy_stage_dir
	return canonical_dir


def _resolve_recording_source_candidate(*, well_out_dir: Path, relpath: str) -> Path:
	candidate = Path(str(relpath)).expanduser()
	if candidate.is_absolute():
		candidate = candidate.resolve()
	else:
		candidate = (well_out_dir / str(relpath).lstrip("/")).resolve()
	if candidate.name in {"preprocessed_recording", "recording"}:
		return candidate
	nested_recording_dir = candidate / "preprocessed_recording"
	if nested_recording_dir.exists():
		return nested_recording_dir
	return candidate


def _recording_source_relpaths_for_assertion(*, inputs: SpikesortInputs) -> list[str]:
	relpaths: list[str] = []
	for value in (
		inputs.sort_original_preprocess_concat_recording_relpath,
		inputs.sort_bootstrapped_concat_recording_relpath,
	):
		token = str(value or "").strip()
		if token and token not in relpaths:
			relpaths.append(token)
	if not relpaths:
		selected = str(inputs.preprocess_concat_recording_relpath or "").strip()
		if selected:
			relpaths.append(selected)
	return relpaths


def _recording_dir_has_materialized_traces(recording_dir: Path) -> bool:
	if not recording_dir.exists() or not recording_dir.is_dir():
		return False
	trace_suffixes = {".raw", ".dat", ".bin"}
	try:
		for child in recording_dir.rglob("*"):
			if not child.is_file():
				continue
			suffix = child.suffix.lower()
			if suffix in trace_suffixes:
				return True
			if suffix == ".npy" and "trace" in child.stem.lower():
				return True
	except Exception:
		return False
	return False


def _resolve_source_recording_dir(*, inputs: SpikesortInputs, well_out_dir: Path) -> tuple[Path, Path]:
	if bool(inputs.sort_assert_one_source):
		valid_sources: list[Path] = []
		for source_relpath in _recording_source_relpaths_for_assertion(inputs=inputs):
			source_dir = _resolve_recording_source_candidate(well_out_dir=well_out_dir, relpath=source_relpath)
			if source_dir.exists() and source_dir.is_dir() and source_dir not in valid_sources:
				valid_sources.append(source_dir)
		if len(valid_sources) != 1:
			raise RuntimeError(
				"Expected exactly one valid spikesort source recording because sort.assert_one_source=true; "
				f"found={len(valid_sources)} candidates={[str(path) for path in valid_sources]}"
			)

	preprocess_relpath = str(inputs.preprocess_concat_recording_relpath or "").strip()
	if preprocess_relpath:
		recording_dir = _resolve_recording_source_candidate(well_out_dir=well_out_dir, relpath=preprocess_relpath)
		preprocess_dir = recording_dir.parent
	else:
		preprocess_dir = _resolve_preprocess_dir(well_out_dir=well_out_dir)
		recording_dir = preprocess_dir / "preprocessed_recording"

	if not recording_dir.exists():
		raise FileNotFoundError(
			"Saved preprocessed recording folder not found. "
			"Run preprocessing first with `mea_output_root` set. "
			f"Expected: {recording_dir}"
		)
	if not bool(inputs.sort_use_lazy_source) and not _recording_dir_has_materialized_traces(recording_dir):
		raise FileNotFoundError(
			"Materialized spikesort source recording not found because sort.use_lazy_source=false. "
			f"Expected binary trace files under: {recording_dir}"
		)
	return recording_dir, preprocess_dir


def _import_spikeinterface_modules() -> tuple[Any, Any]:
	import spikeinterface as si  # type: ignore[import-not-found]

	try:
		import spikeinterface.sorters as sorters  # type: ignore[import-not-found]
	except Exception:
		sorters = si
	return si, sorters


def _load_recording(*, si_module: Any, recording_dir: Path) -> Any:
	load = getattr(si_module, "load", None)
	if callable(load):
		try:
			return load(recording_dir)
		except Exception:
			pass
	load_extractor = getattr(si_module, "load_extractor", None)
	if not callable(load_extractor):
		raise RuntimeError("SpikeInterface must provide load or load_extractor to load saved recordings")
	return load_extractor(recording_dir)


def _set_global_job_kwargs(*, si_module: Any, inputs: SpikesortInputs, logger: logging.Logger) -> None:
	job_kwargs: dict[str, Any] = {}
	if inputs.n_jobs is not None:
		job_kwargs["n_jobs"] = int(inputs.n_jobs)
	if inputs.chunk_duration is not None:
		job_kwargs["chunk_duration"] = str(inputs.chunk_duration)
	job_kwargs["progress_bar"] = bool(inputs.progress_bar)
	if not job_kwargs:
		return
	set_global_job_kwargs = getattr(si_module, "set_global_job_kwargs", None)
	if not callable(set_global_job_kwargs):
		return
	try:
		set_global_job_kwargs(**job_kwargs)
		logger.info("SpikeInterface global job kwargs: %s", job_kwargs)
	except Exception:
		logger.debug("Could not set SpikeInterface global job kwargs", exc_info=True)


def build_local_kilosort_kwargs(*, inputs: SpikesortInputs, recording: Any) -> dict[str, Any]:
	sorter_kwargs: dict[str, Any] = {}
	if inputs.ks_batch_size is not None:
		sorter_kwargs["batch_size"] = int(inputs.ks_batch_size)
	elif inputs.ks_batch_duration_s is not None:
		fs = float(recording.get_sampling_frequency())
		sorter_kwargs["batch_size"] = int(round(fs * float(inputs.ks_batch_duration_s)))

	if inputs.ks_th_universal is not None:
		sorter_kwargs["Th_universal"] = float(inputs.ks_th_universal)
	if inputs.ks_th_learned is not None:
		sorter_kwargs["Th_learned"] = float(inputs.ks_th_learned)
	if inputs.ks_th_single_ch is not None:
		sorter_kwargs["Th_single_ch"] = float(inputs.ks_th_single_ch)
	if inputs.ks_cluster_downsampling is not None:
		sorter_kwargs["cluster_downsampling"] = int(inputs.ks_cluster_downsampling)
	if inputs.ks_nearest_chans is not None:
		sorter_kwargs["nearest_chans"] = int(inputs.ks_nearest_chans)
	if inputs.ks_max_channel_distance is not None:
		sorter_kwargs["max_channel_distance"] = float(inputs.ks_max_channel_distance)
	return sorter_kwargs


def _local_run_sorter_kwargs(inputs: SpikesortInputs) -> dict[str, Any]:
	run_sorter_kwargs = dict(inputs.local_spikeinterface_run_sorter_kwargs or {})
	for forbidden_key in sorted(_FORBIDDEN_RUN_SORTER_KWARGS):
		if forbidden_key in run_sorter_kwargs:
			raise ValueError(
				"local_spikeinterface.run_sorter_kwargs must not request container execution; "
				f"remove {forbidden_key!r}"
			)
	return run_sorter_kwargs


def _run_sorter(
	*,
	sorters_module: Any,
	si_module: Any,
	inputs: SpikesortInputs,
	recording: Any,
	sorter_output_dir: Path,
	sorter_kwargs: dict[str, Any],
) -> Any:
	run_sorter = getattr(sorters_module, "run_sorter", None)
	if not callable(run_sorter):
		run_sorter = getattr(si_module, "run_sorter", None)
	if not callable(run_sorter):
		raise RuntimeError("SpikeInterface run_sorter is required for local_spikeinterface sorting")

	call_kwargs = dict(sorter_kwargs)
	call_kwargs.update(_local_run_sorter_kwargs(inputs))
	call_kwargs.setdefault(
		"remove_existing_folder",
		bool(inputs.force_restart and inputs.local_spikeinterface_remove_existing_on_force_restart),
	)
	call_kwargs.setdefault("verbose", bool(inputs.verbose))
	try:
		return run_sorter(
			sorter_name=str(inputs.sorter),
			recording=recording,
			folder=sorter_output_dir,
			**call_kwargs,
		)
	except TypeError:
		return run_sorter(str(inputs.sorter), recording, sorter_output_dir, **call_kwargs)


def _read_sorting_from_output(*, si_module: Any, sorters_module: Any, sorter_output_dir: Path, sorter_name: str) -> Any:
	for module in (si_module, sorters_module):
		read_sorter_folder = getattr(module, "read_sorter_folder", None)
		if not callable(read_sorter_folder):
			continue
		for kwargs in ({}, {"sorter_name": sorter_name}):
			try:
				return read_sorter_folder(sorter_output_dir, **kwargs)
			except TypeError:
				continue
			except Exception:
				break
	return None


def _create_sorting_analyzer(
	*,
	si_module: Any,
	recording: Any,
	sorting: Any,
	analyzer_dir: Path,
) -> Any:
	create_sorting_analyzer = getattr(si_module, "create_sorting_analyzer", None)
	if not callable(create_sorting_analyzer):
		raise RuntimeError("spikeinterface.create_sorting_analyzer is required for local analyzer output")
	if analyzer_dir.exists():
		shutil.rmtree(analyzer_dir, ignore_errors=True)
	try:
		return create_sorting_analyzer(
			sorting=sorting,
			recording=recording,
			format="binary_folder",
			folder=analyzer_dir,
		)
	except TypeError:
		return create_sorting_analyzer(sorting=sorting, recording=recording, folder=analyzer_dir)


def _resolve_stage_child_path(*, stage_output_root_dir: Path, relpath: str) -> Path:
	candidate = Path(str(relpath).strip()).expanduser()
	if candidate.is_absolute():
		return candidate.resolve()
	return (stage_output_root_dir / str(relpath).strip().lstrip("/")).resolve()


def run_local_spikeinterface_sort_stage(
	*,
	inputs: SpikesortInputs,
	well_out_dir: Path,
	stage_output_root_dir: Path,
	logger: logging.Logger,
	si_module: Any | None = None,
	sorters_module: Any | None = None,
) -> LocalSpikeInterfaceSortOutputs:
	if not bool(inputs.local_spikeinterface_enabled):
		raise ValueError("spikesort sort engine local_spikeinterface selected but local_spikeinterface.enabled is false")

	stage_output_root_dir = Path(stage_output_root_dir).resolve()
	sorter_output_dir = _resolve_stage_child_path(
		stage_output_root_dir=stage_output_root_dir,
		relpath=inputs.local_spikeinterface_output_relpath,
	)
	analyzer_dir = _resolve_stage_child_path(
		stage_output_root_dir=stage_output_root_dir,
		relpath=inputs.local_spikeinterface_analyzer_output_relpath,
	)
	if inputs.cuda_visible_devices is not None:
		os.environ["CUDA_VISIBLE_DEVICES"] = str(inputs.cuda_visible_devices)

	if si_module is None or sorters_module is None:
		loaded_si_module, loaded_sorters_module = _import_spikeinterface_modules()
		si_module = si_module or loaded_si_module
		sorters_module = sorters_module or loaded_sorters_module

	recording_dir, preprocess_dir = _resolve_source_recording_dir(inputs=inputs, well_out_dir=Path(well_out_dir))
	logger.info(
		"Local SpikeInterface source policy: use_bootstrapped_concat_binary=%s use_lazy_source=%s assert_one_source=%s selected_relpath=%s original_relpath=%s bootstrapped_relpath=%s",
		bool(inputs.sort_use_bootstrapped_concat_binary),
		bool(inputs.sort_use_lazy_source),
		bool(inputs.sort_assert_one_source),
		str(inputs.preprocess_concat_recording_relpath or "") or None,
		inputs.sort_original_preprocess_concat_recording_relpath,
		inputs.sort_bootstrapped_concat_recording_relpath,
	)
	logger.info("Resolved preprocessing outputs dir: %s", preprocess_dir)
	logger.info("Loading saved preprocessed recording from %s", recording_dir)
	recording = _load_recording(si_module=si_module, recording_dir=recording_dir)
	_set_global_job_kwargs(si_module=si_module, inputs=inputs, logger=logger)

	sorter_kwargs = build_local_kilosort_kwargs(inputs=inputs, recording=recording)
	logger.info(
		"Starting local SpikeInterface sort: sorter=%s sorter_output_dir=%s analyzer_dir=%s n_jobs=%s progress_bar=%s sorter_kwargs=%s",
		str(inputs.sorter),
		sorter_output_dir,
		analyzer_dir,
		inputs.n_jobs,
		bool(inputs.progress_bar),
		sorter_kwargs,
	)
	with suppress_spikesort_external_debug_output(enabled=bool(inputs.debug_outputs)):
		sorting = _run_sorter(
			sorters_module=sorters_module,
			si_module=si_module,
			inputs=inputs,
			recording=recording,
			sorter_output_dir=sorter_output_dir,
			sorter_kwargs=sorter_kwargs,
		)
		if sorting is None:
			sorting = _read_sorting_from_output(
				si_module=si_module,
				sorters_module=sorters_module,
				sorter_output_dir=sorter_output_dir,
				sorter_name=str(inputs.sorter),
			)
		if sorting is None and bool(inputs.local_spikeinterface_analyzer_enabled and inputs.run_analyzer):
			raise RuntimeError("Local SpikeInterface sorter did not return a sorting and it could not be reloaded")

		if bool(inputs.local_spikeinterface_analyzer_enabled and inputs.run_analyzer):
			_create_sorting_analyzer(
				si_module=si_module,
				recording=recording,
				sorting=sorting,
				analyzer_dir=analyzer_dir,
			)

	return LocalSpikeInterfaceSortOutputs(
		recording_dir=recording_dir,
		sorter_output_dir=sorter_output_dir,
		output_dir=stage_output_root_dir,
		analyzer_dir=analyzer_dir,
	)