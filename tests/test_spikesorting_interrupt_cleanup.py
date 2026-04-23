from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from axon_reconstructor.pipeline.stg2_spikesorting.runner import (
    SpikeSortingInputs,
    _cleanup_interrupted_sorter_containers,
    run_spikesorting_stage,
)


def test_cleanup_interrupted_sorter_containers_removes_matching_mount(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    removed: list[tuple[str, bool]] = []
    mount_source = (tmp_path / "spikesort_outputs").resolve()

    class _FakeContainer:
        def __init__(self, name: str, container_id: str, mounts: list[dict[str, str]]):
            self.name = name
            self.id = container_id
            self.short_id = container_id[:12]
            self.attrs = {"Mounts": mounts}

        def remove(self, force: bool = False):
            removed.append((self.name, force))

    class _FakeContainerManager:
        def list(self, all: bool = False, filters: dict[str, str] | None = None):
            assert all is True
            assert filters == {"ancestor": "dummy/image:latest"}
            return [
                _FakeContainer("matching", "abcdef1234567890", [{"Source": str(mount_source)}]),
                _FakeContainer("other-mount", "fedcba0987654321", [{"Source": str(tmp_path / 'other')}]),
            ]

    class _FakeDockerClient:
        def __init__(self):
            self.containers = _FakeContainerManager()

        def close(self):
            return None

    fake_docker = ModuleType("docker")
    fake_docker.from_env = lambda timeout=300: _FakeDockerClient()
    monkeypatch.setitem(sys.modules, "docker", fake_docker)

    labels = _cleanup_interrupted_sorter_containers(
        docker_image="dummy/image:latest",
        mount_source=mount_source,
        logger=logging.getLogger("test.cleanup"),
    )

    assert removed == [("matching", True)]
    assert labels == ["matching(abcdef123456)"]


def test_run_spikesorting_stage_cleans_sorter_containers_on_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import axon_reconstructor.pipeline.output_paths as output_paths
    import axon_reconstructor.pipeline.stg2_spikesorting.runner as runner_mod

    well_out_dir = tmp_path / "outputs" / "well000"
    recording_dir = well_out_dir / "preprocess_outputs" / "preprocessed_recording"
    recording_dir.mkdir(parents=True)

    class _DummyRecording:
        def get_num_segments(self):
            return 1

        def get_channel_ids(self):
            return [0, 1]

        def get_sampling_frequency(self):
            return 10_000.0

    fake_si_pkg = ModuleType("spikeinterface")
    fake_si = ModuleType("spikeinterface.full")
    fake_si.set_global_job_kwargs = lambda **kwargs: None
    fake_si.load = lambda path: _DummyRecording()
    fake_si.load_extractor = lambda path: _DummyRecording()
    fake_si_pkg.full = fake_si
    monkeypatch.setitem(sys.modules, "spikeinterface", fake_si_pkg)
    monkeypatch.setitem(sys.modules, "spikeinterface.full", fake_si)

    fake_ipn_pkg = ModuleType("IPNAnalysis")
    fake_mea = ModuleType("IPNAnalysis.mea_analysis_routine")
    fake_mea.MEARunOptions = lambda **kwargs: SimpleNamespace(**kwargs)
    fake_mea.ProcessingStage = SimpleNamespace(
        REPORTS_COMPLETE=SimpleNamespace(value=10),
        ANALYZER_COMPLETE=SimpleNamespace(value=8),
        SORTING_COMPLETE=SimpleNamespace(value=4),
    )
    fake_mea.run_mea_pipeline = lambda run_options: (_ for _ in ()).throw(KeyboardInterrupt())
    fake_ipn_pkg.mea_analysis_routine = fake_mea
    monkeypatch.setitem(sys.modules, "IPNAnalysis", fake_ipn_pkg)
    monkeypatch.setitem(sys.modules, "IPNAnalysis.mea_analysis_routine", fake_mea)

    cleanup_calls: list[tuple[str | None, Path]] = []

    monkeypatch.setattr(output_paths, "compute_mea_analysis_output_dir", lambda **kwargs: well_out_dir)
    monkeypatch.setattr(runner_mod, "compute_stage_checkpoint_file", lambda **kwargs: tmp_path / "spikesort.ckpt.json")
    monkeypatch.setattr(runner_mod, "load_checkpoint", lambda **kwargs: SimpleNamespace(stage=0))
    monkeypatch.setattr(runner_mod, "save_stage_started", lambda **kwargs: SimpleNamespace(stage=1))
    monkeypatch.setattr(runner_mod, "save_stage_completed", lambda **kwargs: SimpleNamespace(stage=2))
    monkeypatch.setattr(runner_mod, "save_stage_failed", lambda **kwargs: SimpleNamespace(stage=-1))
    monkeypatch.setattr(runner_mod, "log_stage_start", lambda **kwargs: None)
    monkeypatch.setattr(runner_mod, "log_stage_complete", lambda **kwargs: None)
    monkeypatch.setattr(runner_mod, "log_stage_failure", lambda **kwargs: None)
    monkeypatch.setattr(
        runner_mod,
        "_cleanup_interrupted_sorter_containers",
        lambda *, docker_image, mount_source, logger: cleanup_calls.append((docker_image, mount_source)) or ["container(abc123)"],
    )

    inputs = SpikeSortingInputs(
        h5_path=tmp_path / "data.raw.h5",
        stream_id="well000",
        mea_output_root=tmp_path / "outputs_root",
        docker_image="dummy/image:latest",
        log_enabled=False,
    )

    with pytest.raises(KeyboardInterrupt):
        run_spikesorting_stage(inputs=inputs, logger=logging.getLogger("test.spikesort"))

    assert cleanup_calls == [("dummy/image:latest", (well_out_dir / "spikesort_outputs").resolve())]