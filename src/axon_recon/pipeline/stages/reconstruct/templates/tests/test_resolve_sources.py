from __future__ import annotations

import json
from pathlib import Path

from axon_recon.pipeline.output_paths import compute_mea_analysis_output_dir
from axon_recon.pipeline.stages.reconstruct.templates.models.inputs import ResolveSourcesPhaseConfig, TemplatesInputs
from axon_recon.pipeline.stages.reconstruct.templates.runner import run_reconstruct_templates_resolve_sources_phase


def _prepare_source_dirs(*, well_out_dir: Path) -> dict[str, Path]:
	concat_analyzer = well_out_dir / "spikesort_outputs" / "analyzer_output"
	concat_sorting = well_out_dir / "spikesort_outputs" / "sorter_output"
	preprocessed_concat = well_out_dir / "preprocess_outputs" / "preprocessed_recording"
	preprocessed_segments = well_out_dir / "preprocess_outputs" / "per_segment_preprocessed"

	concat_analyzer.mkdir(parents=True, exist_ok=True)
	concat_sorting.mkdir(parents=True, exist_ok=True)
	preprocessed_concat.mkdir(parents=True, exist_ok=True)
	preprocessed_segments.mkdir(parents=True, exist_ok=True)

	return {
		"concat_analyzer": concat_analyzer.resolve(),
		"concat_sorting": concat_sorting.resolve(),
		"preprocessed_concat": preprocessed_concat.resolve(),
		"preprocessed_segments": preprocessed_segments.resolve(),
	}


def test_run_reconstruct_templates_resolve_sources_phase_resolves_expected_sources(tmp_path: Path) -> None:
	output_root = tmp_path / "outputs"
	h5_path = tmp_path / "dataset.h5"
	h5_path.write_text("", encoding="utf-8")

	well_out_dir = compute_mea_analysis_output_dir(output_root=output_root, data_file=h5_path, well="well000")
	resolved = _prepare_source_dirs(well_out_dir=well_out_dir)

	inputs = TemplatesInputs(
		h5_path=h5_path,
		stream_id="well000",
		mea_output_root=output_root,
		output_rel_root="templates_outputs",
		unit_label_filter_required=False,
		resolve_sources_phase=ResolveSourcesPhaseConfig(
			enabled=True,
			log_candidates=False,
			include_alternate_well_dirs=False,
			max_candidates_per_source=4,
			write_json=False,
		),
	)

	summary = run_reconstruct_templates_resolve_sources_phase(inputs)
	sources = summary["sources"]

	assert sources["concat_analyzer"]["first_existing"] == str(resolved["concat_analyzer"])
	assert sources["concat_sorting"]["first_existing"] == str(resolved["concat_sorting"])
	assert sources["preprocessed_concat"]["first_existing"] == str(resolved["preprocessed_concat"])
	assert sources["preprocessed_segments"]["first_existing"] == str(resolved["preprocessed_segments"])
	assert summary["unit_scope"]["unit_label_filter"]["probe_attempted"] is True
	assert summary["unit_scope"]["unit_label_filter"]["available"] is False


def test_run_reconstruct_templates_resolve_sources_phase_writes_summary_json_when_enabled(tmp_path: Path) -> None:
	output_root = tmp_path / "outputs"
	h5_path = tmp_path / "dataset.h5"
	h5_path.write_text("", encoding="utf-8")

	well_out_dir = compute_mea_analysis_output_dir(output_root=output_root, data_file=h5_path, well="well000")
	_prepare_source_dirs(well_out_dir=well_out_dir)

	inputs = TemplatesInputs(
		h5_path=h5_path,
		stream_id="well000",
		mea_output_root=output_root,
		output_rel_root="templates_outputs",
		unit_label_filter_required=False,
		resolve_sources_phase=ResolveSourcesPhaseConfig(
			enabled=True,
			log_candidates=False,
			include_alternate_well_dirs=False,
			write_json=True,
			json_relpath="context/resolve_sources_test.json",
		),
	)

	summary = run_reconstruct_templates_resolve_sources_phase(inputs)
	summary_json = Path(str(summary["summary_json"]))
	assert summary_json.exists()

	payload = json.loads(summary_json.read_text(encoding="utf-8"))
	assert payload["phase"] == "resolve_sources"
	assert payload["stream_id"] == "well000"


def test_run_reconstruct_templates_resolve_sources_phase_logs_header_when_enabled(tmp_path: Path, caplog) -> None:
	output_root = tmp_path / "outputs"
	h5_path = tmp_path / "dataset.h5"
	h5_path.write_text("", encoding="utf-8")

	well_out_dir = compute_mea_analysis_output_dir(output_root=output_root, data_file=h5_path, well="well000")
	_prepare_source_dirs(well_out_dir=well_out_dir)

	inputs = TemplatesInputs(
		h5_path=h5_path,
		stream_id="well000",
		mea_output_root=output_root,
		output_rel_root="templates_outputs",
		unit_label_filter_required=False,
		resolve_sources_phase=ResolveSourcesPhaseConfig(
			enabled=True,
			show_header=True,
			log_candidates=False,
			include_alternate_well_dirs=False,
			write_json=False,
		),
	)

	with caplog.at_level("INFO"):
		run_reconstruct_templates_resolve_sources_phase(inputs)

	assert "templates.resolve_sources [well000]" in caplog.text
