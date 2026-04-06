from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from axon_recon.pipeline.stages.spikesort.models.inputs import SpikesortInputs
from axon_recon.pipeline.stages.spikesort.runner import run_spikesort_merge_stage, run_spikesort_stage


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_run_spikesort_stage_propagates_logging_debug_plot_report_inputs(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    well_out_dir = tmp_path / "well001"
    captured_legacy_inputs: dict[str, object] = {}

    def _fake_compute_mea_analysis_output_dir(*, output_root: Path, data_file: Path, well: str) -> Path:
        return well_out_dir

    class _LegacyOutputs:
        def __init__(self, output_subdir_after_well: str) -> None:
            self.recording_dir = well_out_dir / "preprocess_outputs" / "preprocessed_recording"
            self.sorter_output_dir = well_out_dir / output_subdir_after_well / "sorter_output"
            self.output_dir = well_out_dir / output_subdir_after_well
            self.analyzer_dir = well_out_dir / output_subdir_after_well / "analyzer_output"
            self.merged_sorting_dir = None
            self.merged_sorter_output_dir = None

    def _fake_run_legacy_spikesorting_stage(*, inputs, logger):
        captured_legacy_inputs.update(
            {
                "preprocess_concat_recording_relpath": getattr(inputs, "preprocess_concat_recording_relpath", None),
                "log_enabled": bool(getattr(inputs, "log_enabled")),
                "log_verbose": bool(getattr(inputs, "log_verbose")),
                "log_file_override": getattr(inputs, "log_file_override"),
                "output_subdir_after_well": getattr(inputs, "output_subdir_after_well"),
                "plot_mode": getattr(inputs, "plot_mode"),
                "plot_debug": bool(getattr(inputs, "plot_debug")),
                "raster_sort": getattr(inputs, "raster_sort"),
                "fixed_y": bool(getattr(inputs, "fixed_y")),
                "run_reports": bool(getattr(inputs, "run_reports")),
                "no_curation": bool(getattr(inputs, "no_curation")),
                "export_to_phy": bool(getattr(inputs, "export_to_phy")),
            }
        )
        return _LegacyOutputs(str(getattr(inputs, "output_subdir_after_well")))

    monkeypatch.setattr(spikesort_runner, "compute_mea_analysis_output_dir", _fake_compute_mea_analysis_output_dir)
    monkeypatch.setattr(spikesort_runner, "run_legacy_spikesorting_stage", _fake_run_legacy_spikesorting_stage)

    inputs = SpikesortInputs(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        output_rel_root="spikesort_outputs_v2",
        preprocess_concat_recording_relpath="preprocess_outputs/preprocessed_recording",
        logging_enabled=False,
        logging_verbose=True,
        logging_file_relpath="logs/custom_spikesort.log",
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
    assert captured_legacy_inputs.get("preprocess_concat_recording_relpath") == "preprocess_outputs/preprocessed_recording"
    assert captured_legacy_inputs.get("output_subdir_after_well") == "spikesort_outputs_v2"
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
    assert summary.get("inputs", {}).get("preprocess_concat_recording_relpath") == "preprocess_outputs/preprocessed_recording"
    assert summary.get("inputs", {}).get("plot_mode") == "merged"
    assert summary.get("inputs", {}).get("plot_debug") is True
    assert summary.get("inputs", {}).get("raster_sort") == "unit_id"
    assert summary.get("inputs", {}).get("fixed_y") is True
    assert summary.get("inputs", {}).get("run_reports") is False
    assert summary.get("inputs", {}).get("no_curation") is True
    assert summary.get("inputs", {}).get("export_to_phy") is True
    assert summary.get("inputs", {}).get("sort_enabled") is True
    assert summary.get("inputs", {}).get("sort_delete_outputs_on_force_restart") is False
    assert result.spikesort_out_dir == well_out_dir / "spikesort_outputs_v2"


def test_run_spikesort_stage_deletes_sort_outputs_on_force_restart_when_enabled(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    well_out_dir = tmp_path / "well001"
    stage_out_dir = well_out_dir / "spikesort_outputs"
    sorter_out_dir = stage_out_dir / "sorter_output"
    analyzer_out_dir = stage_out_dir / "analyzer_output"
    sorter_out_dir.mkdir(parents=True, exist_ok=True)
    analyzer_out_dir.mkdir(parents=True, exist_ok=True)
    (sorter_out_dir / "stale.txt").write_text("old", encoding="utf-8")
    (analyzer_out_dir / "stale.txt").write_text("old", encoding="utf-8")

    def _fake_compute_mea_analysis_output_dir(*, output_root: Path, data_file: Path, well: str) -> Path:
        return well_out_dir

    class _LegacyOutputs:
        def __init__(self) -> None:
            self.recording_dir = well_out_dir / "preprocess_outputs" / "preprocessed_recording"
            self.sorter_output_dir = stage_out_dir / "sorter_output"
            self.output_dir = stage_out_dir
            self.analyzer_dir = stage_out_dir / "analyzer_output"
            self.merged_sorting_dir = None
            self.merged_sorter_output_dir = None

    def _fake_run_legacy_spikesorting_stage(*, inputs, logger):
        assert not sorter_out_dir.exists()
        assert not analyzer_out_dir.exists()
        return _LegacyOutputs()

    monkeypatch.setattr(spikesort_runner, "compute_mea_analysis_output_dir", _fake_compute_mea_analysis_output_dir)
    monkeypatch.setattr(spikesort_runner, "run_legacy_spikesorting_stage", _fake_run_legacy_spikesorting_stage)

    inputs = SpikesortInputs(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        output_rel_root="spikesort_outputs",
        force_restart=True,
        sort_delete_outputs_on_force_restart=True,
    )

    result = run_spikesort_stage(inputs)
    summary = _read_json(result.summary_json)

    removed = summary.get("cleanup", {}).get("removed_on_force_restart", [])
    assert any(path.endswith("/spikesort_outputs/sorter_output") for path in removed)
    assert any(path.endswith("/spikesort_outputs/analyzer_output") for path in removed)


def test_run_spikesort_stage_skips_when_sort_disabled(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    well_out_dir = tmp_path / "well001"

    def _fake_compute_mea_analysis_output_dir(*, output_root: Path, data_file: Path, well: str) -> Path:
        return well_out_dir

    def _never_call_legacy(*, inputs, logger):
        raise AssertionError("legacy spikesort should not run when sort is disabled")

    monkeypatch.setattr(spikesort_runner, "compute_mea_analysis_output_dir", _fake_compute_mea_analysis_output_dir)
    monkeypatch.setattr(spikesort_runner, "run_legacy_spikesorting_stage", _never_call_legacy)

    inputs = SpikesortInputs(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        output_rel_root="spikesort_outputs",
        sort_enabled=False,
    )

    result = run_spikesort_stage(inputs)
    summary = _read_json(result.summary_json)

    assert summary.get("status") == "skipped"
    assert summary.get("reason") == "sort_disabled"
    assert result.spikesort_out_dir == well_out_dir / "spikesort_outputs"
    assert result.spikesort_out_dir.exists()


def test_run_spikesort_merge_stage_writes_recommended_candidate_outputs(tmp_path: Path, monkeypatch) -> None:
    from axon_reconstructor.pipeline.output_paths import compute_mea_analysis_output_dir
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    h5_path = tmp_path / "raw_data" / "input.raw.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"")

    output_rel_root = "spikesort_outputs"
    stream_id = "well001"
    well_out_dir = compute_mea_analysis_output_dir(
        output_root=tmp_path,
        data_file=h5_path,
        well=stream_id,
    )
    ks_wrapper_dir = well_out_dir / output_rel_root / "sorter_output"
    ks_dir = ks_wrapper_dir / "sorter_output"
    ks_dir.mkdir(parents=True, exist_ok=True)
    (ks_dir / "params.py").write_text(
        "dat_path = 'data.bin'\n"
        "n_channels_dat = 4\n"
        "dtype = 'int16'\n"
        "sample_rate = 30000\n",
        encoding="utf-8",
    )

    def _fake_import_slay_run_function(*, package_root, allow_numpy_fallback):
        assert package_root == "/tmp/slay"
        assert allow_numpy_fallback is True

        def _fake_run_slay(args):
            assert str(args["KS_folder"]) == str(ks_dir)
            automerge_dir = Path(args["KS_folder"]) / "automerge"
            automerge_dir.mkdir(parents=True, exist_ok=True)
            (automerge_dir / "new2old.json").write_text(
                json.dumps({"100": [1, 2, 3]}),
                encoding="utf-8",
            )
            (automerge_dir / "metrics.tsv").write_text(
                "Cluster 1\tCluster 2\tSimilarity\tCross-correlation Significance\tRefractory Period Penalty\tFinal Metric\n"
                "1\t2\t0.91\t0.11\t0.01\t0.73\n"
                "1\t3\t0.92\t0.12\t0.02\t0.74\n"
                "2\t3\t0.93\t0.13\t0.03\t0.75\n",
                encoding="utf-8",
            )
            Path(args["output_json"]).write_text(json.dumps({"num_merges": 1}), encoding="utf-8")

        return _fake_run_slay

    monkeypatch.setattr(spikesort_runner, "_import_slay_run_function", _fake_import_slay_run_function)

    stage_cfg = SimpleNamespace(
        slay_enabled=True,
        slay_relpath="SLAy_outputs",
        slay_package_root="/tmp/slay",
        slay_sorter_output_relpath=None,
        slay_output_json_relpath="run-output.json",
        slay_candidate_pairs_relpath="recommended_merge_candidates.tsv",
        slay_merge_groups_relpath="recommended_merge_groups.json",
        slay_allow_numpy_fallback=True,
        slay_plot_merges=False,
        slay_auto_accept_merges=False,
        slay_copy_automerge_artifacts=True,
        slay_delete_outputs_on_force_restart=True,
        slay_params={"max_spikes": 123},
    )

    result = run_spikesort_merge_stage(
        h5_path=h5_path,
        stream_id=stream_id,
        mea_output_root=tmp_path,
        output_rel_root=output_rel_root,
        stage_config=stage_cfg,
        force_restart=True,
    )

    summary = _read_json(result.summary_json)
    groups = _read_json(result.merge_out_dir / "recommended_merge_groups.json")
    candidates_tsv = (result.merge_out_dir / "recommended_merge_candidates.tsv").read_text(encoding="utf-8")

    assert result.merge_out_dir == well_out_dir / output_rel_root / "SLAy_outputs"
    assert summary.get("status") == "ok"
    assert summary.get("stage_output_root_dir") == str(well_out_dir / output_rel_root)
    assert summary.get("n_merge_groups") == 1
    assert summary.get("n_candidate_pairs") == 3
    assert groups.get("n_groups") == 1
    assert groups.get("merge_groups", {}).get("100") == [1, 2, 3]
    assert "cluster_a\tcluster_b" in candidates_tsv
    assert "1\t2" in candidates_tsv
    assert "1\t3" in candidates_tsv
    assert "2\t3" in candidates_tsv


def test_run_spikesort_merge_stage_skips_when_disabled(tmp_path: Path) -> None:
    h5_path = tmp_path / "raw_data" / "input.raw.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"")

    stage_cfg = SimpleNamespace(
        slay_enabled=False,
        slay_relpath="SLAy_outputs",
        slay_delete_outputs_on_force_restart=True,
    )

    result = run_spikesort_merge_stage(
        h5_path=h5_path,
        stream_id="well001",
        mea_output_root=tmp_path,
        output_rel_root="spikesort_outputs",
        stage_config=stage_cfg,
        force_restart=False,
    )

    summary = _read_json(result.summary_json)
    assert result.merge_out_dir == result.well_out_dir / "spikesort_outputs" / "SLAy_outputs"
    assert summary.get("status") == "skipped"
    assert summary.get("reason") == "slay_disabled"
    assert summary.get("stage_output_root_dir") == str(result.well_out_dir / "spikesort_outputs")


def test_run_spikesort_merge_stage_preserves_existing_outputs_when_delete_disabled(tmp_path: Path, monkeypatch) -> None:
    from axon_reconstructor.pipeline.output_paths import compute_mea_analysis_output_dir
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    h5_path = tmp_path / "raw_data" / "input.raw.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"")

    output_rel_root = "spikesort_outputs"
    stream_id = "well001"
    well_out_dir = compute_mea_analysis_output_dir(
        output_root=tmp_path,
        data_file=h5_path,
        well=stream_id,
    )
    ks_dir = well_out_dir / output_rel_root / "sorter_output" / "sorter_output"
    ks_dir.mkdir(parents=True, exist_ok=True)
    (ks_dir / "params.py").write_text(
        "dat_path = 'data.bin'\n"
        "n_channels_dat = 4\n"
        "dtype = 'int16'\n"
        "sample_rate = 30000\n",
        encoding="utf-8",
    )

    merge_out_dir = well_out_dir / output_rel_root / "SLAy_outputs"
    merge_out_dir.mkdir(parents=True, exist_ok=True)
    sentinel = merge_out_dir / "keep_me.txt"
    sentinel.write_text("persist", encoding="utf-8")

    def _fake_import_slay_run_function(*, package_root, allow_numpy_fallback):
        def _fake_run_slay(args):
            automerge_dir = Path(args["KS_folder"]) / "automerge"
            automerge_dir.mkdir(parents=True, exist_ok=True)
            (automerge_dir / "new2old.json").write_text(json.dumps({}), encoding="utf-8")
            (automerge_dir / "metrics.tsv").write_text(
                "Cluster 1\tCluster 2\tSimilarity\tCross-correlation Significance\tRefractory Period Penalty\tFinal Metric\n",
                encoding="utf-8",
            )
            Path(args["output_json"]).write_text(json.dumps({"num_merges": 0}), encoding="utf-8")

        return _fake_run_slay

    monkeypatch.setattr(spikesort_runner, "_import_slay_run_function", _fake_import_slay_run_function)

    stage_cfg = SimpleNamespace(
        slay_enabled=True,
        slay_relpath="SLAy_outputs",
        slay_package_root="/tmp/slay",
        slay_sorter_output_relpath=None,
        slay_output_json_relpath="run-output.json",
        slay_candidate_pairs_relpath="recommended_merge_candidates.tsv",
        slay_merge_groups_relpath="recommended_merge_groups.json",
        slay_allow_numpy_fallback=True,
        slay_plot_merges=False,
        slay_auto_accept_merges=False,
        slay_copy_automerge_artifacts=True,
        slay_delete_outputs_on_force_restart=False,
        slay_params={"max_spikes": 10},
    )

    run_spikesort_merge_stage(
        h5_path=h5_path,
        stream_id=stream_id,
        mea_output_root=tmp_path,
        output_rel_root=output_rel_root,
        stage_config=stage_cfg,
        force_restart=True,
    )

    assert sentinel.exists()
