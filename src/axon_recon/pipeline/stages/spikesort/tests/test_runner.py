from __future__ import annotations

import json
from pathlib import Path

from axon_recon.pipeline.stages.spikesort.models.inputs import SpikesortInputs
from axon_recon.pipeline.stages.spikesort.runner import run_spikesort_stage


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_run_spikesort_stage_propagates_logging_debug_plot_report_inputs(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    well_out_dir = tmp_path / "well001"
    captured_legacy_inputs: dict[str, object] = {}

    def _fake_compute_mea_analysis_output_dir(*, output_root: Path, data_file: Path, well: str) -> Path:
        return well_out_dir

    class _LegacyOutputs:
        def __init__(self) -> None:
            self.recording_dir = well_out_dir / "preprocess_outputs" / "preprocessed_recording"
            self.sorter_output_dir = well_out_dir / "stg2_spikesorting_outputs" / "sorter_output"
            self.output_dir = well_out_dir / "stg2_spikesorting_outputs"
            self.analyzer_dir = well_out_dir / "stg2_spikesorting_outputs" / "analyzer_output"
            self.merged_sorting_dir = None
            self.merged_sorter_output_dir = None

    def _fake_run_legacy_spikesorting_stage(*, inputs, logger):
        captured_legacy_inputs.update(
            {
                "log_enabled": bool(getattr(inputs, "log_enabled")),
                "log_verbose": bool(getattr(inputs, "log_verbose")),
                "log_file_override": getattr(inputs, "log_file_override"),
                "limit_segments_per_well": getattr(inputs, "limit_segments_per_well"),
                "plot_mode": getattr(inputs, "plot_mode"),
                "plot_debug": bool(getattr(inputs, "plot_debug")),
                "raster_sort": getattr(inputs, "raster_sort"),
                "fixed_y": bool(getattr(inputs, "fixed_y")),
                "run_reports": bool(getattr(inputs, "run_reports")),
                "no_curation": bool(getattr(inputs, "no_curation")),
                "export_to_phy": bool(getattr(inputs, "export_to_phy")),
            }
        )
        return _LegacyOutputs()

    monkeypatch.setattr(spikesort_runner, "compute_mea_analysis_output_dir", _fake_compute_mea_analysis_output_dir)
    monkeypatch.setattr(spikesort_runner, "run_legacy_spikesorting_stage", _fake_run_legacy_spikesorting_stage)

    inputs = SpikesortInputs(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        output_rel_root="spikesort_outputs_v2",
        logging_enabled=False,
        logging_verbose=True,
        logging_file_relpath="logs/custom_spikesort.log",
        debug_limit_segments_per_well=2,
        run_reports=False,
        plot_mode="merged",
        plot_debug=True,
        raster_sort="unit_id",
        fixed_y=True,
        no_curation=True,
        export_to_phy=True,
    )

    result = run_spikesort_stage(inputs)
    summary = _read_json(result.summary_json)

    assert captured_legacy_inputs.get("log_enabled") is False
    assert captured_legacy_inputs.get("log_verbose") is True
    assert captured_legacy_inputs.get("log_file_override") == "logs/custom_spikesort.log"
    assert captured_legacy_inputs.get("limit_segments_per_well") == 2
    assert captured_legacy_inputs.get("plot_mode") == "merged"
    assert captured_legacy_inputs.get("plot_debug") is True
    assert captured_legacy_inputs.get("raster_sort") == "unit_id"
    assert captured_legacy_inputs.get("fixed_y") is True
    assert captured_legacy_inputs.get("run_reports") is False
    assert captured_legacy_inputs.get("no_curation") is True
    assert captured_legacy_inputs.get("export_to_phy") is True

    assert summary.get("inputs", {}).get("logging_enabled") is False
    assert summary.get("inputs", {}).get("logging_verbose") is True
    assert summary.get("inputs", {}).get("logging_file_relpath") == "logs/custom_spikesort.log"
    assert summary.get("inputs", {}).get("debug_limit_segments_per_well") == 2
    assert summary.get("inputs", {}).get("plot_mode") == "merged"
    assert summary.get("inputs", {}).get("plot_debug") is True
    assert summary.get("inputs", {}).get("raster_sort") == "unit_id"
    assert summary.get("inputs", {}).get("fixed_y") is True
    assert summary.get("inputs", {}).get("run_reports") is False
    assert summary.get("inputs", {}).get("no_curation") is True
    assert summary.get("inputs", {}).get("export_to_phy") is True
