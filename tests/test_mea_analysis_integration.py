from __future__ import annotations

from pathlib import Path

import pytest

from axon_reconstructor.integrations.mea_analysis import (
    MEAAnalysisRunSpec,
    build_run_pipeline_driver_cmd,
    compute_mea_output_dir,
    compute_mea_relative_pattern,
    compute_spikesorting_output_dir,
    compute_sorter_output_dir,
    validate_sorter_output_dir,
)


def test_compute_mea_relative_pattern_deep_path():
    # The contract is owned by MEA_Analysis (installed as IPNAnalysis.path_contract).
    pc = pytest.importorskip("IPNAnalysis.path_contract")

    data_file = Path("/tmp/ProjectA/2026-01-01/Chip123/Network/123456/data.raw.h5")
    assert pc.compute_relative_pattern(data_file) == "ProjectA/2026-01-01/Chip123/Network/123456"


def test_compute_mea_relative_pattern_prefers_installed_contract(monkeypatch, tmp_path: Path):
    pc = pytest.importorskip("IPNAnalysis.path_contract")

    monkeypatch.setattr(pc, "compute_relative_pattern", lambda _: "SINGLE_SOURCE_OF_TRUTH")

    data_file = tmp_path / "whatever" / "data.raw.h5"
    rel = compute_mea_relative_pattern(data_file)
    assert rel == "SINGLE_SOURCE_OF_TRUTH"

    out = compute_mea_output_dir(output_root=tmp_path / "out", data_file=data_file, well="well000")
    assert str(out).endswith("/out/SINGLE_SOURCE_OF_TRUTH/well000")


def test_compute_output_and_sorter_output_dirs():
    pytest.importorskip("IPNAnalysis.path_contract")

    output_root = Path("/tmp/outputs")
    data_file = Path("/tmp/ProjectA/2026-01-01/Chip123/Network/123456/data.raw.h5")
    well = "well000"

    out = compute_mea_output_dir(output_root=output_root, data_file=data_file, well=well)
    assert out.as_posix().endswith(
        "/outputs/ProjectA/2026-01-01/Chip123/Network/123456/well000"
    )

    sorter_out = compute_sorter_output_dir(output_root=output_root, data_file=data_file, well=well)
    assert sorter_out.name == "sorter_output"
    assert sorter_out.parent == compute_spikesorting_output_dir(
        output_root=output_root,
        data_file=data_file,
        well=well,
    )


def test_validate_sorter_output_dir(tmp_path: Path):
    sorter_output = tmp_path / "sorter_output"
    assert validate_sorter_output_dir(sorter_output) is False

    sorter_output.mkdir()
    assert validate_sorter_output_dir(sorter_output) is False

    # Create a plausible artifact
    (sorter_output / "spike_times.npy").write_bytes(b"fake")
    assert validate_sorter_output_dir(sorter_output) is True


def test_build_run_pipeline_driver_cmd_includes_hpc_flags(tmp_path: Path):
    mea_repo = tmp_path / "MEA_Analysis"
    (mea_repo / "IPNAnalysis").mkdir(parents=True)

    spec = MEAAnalysisRunSpec(
        mea_analysis_repo_root=mea_repo,
        path=tmp_path / "raw",
        output_dir=tmp_path / "out",
        sorter="kilosort4",
        scratch_dir=tmp_path / "scratch",
        stage_back="sorter",
        stage_back_mode="move",
        require_gpu=True,
        cuda_visible_devices="0",
        n_jobs=8,
        chunk_duration="1s",
        debug=True,
    )

    argv = build_run_pipeline_driver_cmd(spec)

    # Basic structure
    assert argv[0] == "python3"
    assert argv[1].endswith("IPNAnalysis/run_pipeline_driver.py")

    # Core args
    assert "--output-dir" in argv
    assert "--sorter" in argv

    # HPC/GPU carveouts
    assert "--scratch-dir" in argv
    assert "--stage-back" in argv
    assert "--stage-back-mode" in argv
    assert "--require-gpu" in argv
    assert "--cuda-visible-devices" in argv
    assert "--n-jobs" in argv
    assert "--chunk-duration" in argv

    # Values present
    assert "move" in argv
    assert "0" in argv
    assert "8" in argv
    assert "1s" in argv
