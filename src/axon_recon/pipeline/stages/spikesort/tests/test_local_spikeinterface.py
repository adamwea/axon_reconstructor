from __future__ import annotations

import io
import logging
from pathlib import Path
import sys

import pytest

from axon_recon.pipeline.stages.spikesort.core.debug_outputs import (
    suppress_spikesort_external_debug_output,
)
from axon_recon.pipeline.stages.spikesort.core.local_spikeinterface import (
    build_local_kilosort_kwargs,
    run_local_spikeinterface_sort_stage,
)
from axon_recon.pipeline.stages.spikesort.models.inputs import SpikesortInputs


class _FakeRecording:
    def get_sampling_frequency(self) -> float:
        return 20_000.0


class _FakeSpikeInterface:
    def __init__(self) -> None:
        self.global_job_kwargs: dict[str, object] = {}
        self.analyzer_kwargs: dict[str, object] = {}

    def load(self, recording_dir: Path) -> _FakeRecording:
        assert Path(recording_dir).exists()
        return _FakeRecording()

    def set_global_job_kwargs(self, **kwargs) -> None:
        self.global_job_kwargs.update(kwargs)

    def create_sorting_analyzer(self, **kwargs):
        self.analyzer_kwargs.update(kwargs)
        Path(kwargs["folder"]).mkdir(parents=True, exist_ok=True)
        return object()


class _FakeSorters:
    def __init__(self) -> None:
        self.run_sorter_kwargs: dict[str, object] = {}

    def run_sorter(self, **kwargs):
        self.run_sorter_kwargs.update(kwargs)
        Path(kwargs["folder"]).mkdir(parents=True, exist_ok=True)
        return object()


def test_suppress_spikesort_external_debug_output_keeps_terminal_streams_and_file_handlers(tmp_path: Path, capsys) -> None:
    logger = logging.getLogger("kilosort")
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate

    raw_stream = io.StringIO()
    raw_handler = logging.StreamHandler(raw_stream)
    file_path = tmp_path / "kilosort4.log"
    file_handler = logging.FileHandler(file_path)
    logger.handlers = [raw_handler, file_handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False

    try:
        with suppress_spikesort_external_debug_output(enabled=False):
            print("stdout noise")
            print("stderr noise", file=sys.stderr)
            assert any(type(handler) == logging.StreamHandler for handler in logger.handlers)
            logger.info("kilosort file line")
        captured = capsys.readouterr()
        assert "stdout noise" in captured.out
        assert "stderr noise" in captured.err
        assert raw_stream.getvalue() == ""
        assert "kilosort file line" in file_path.read_text(encoding="utf-8")
    finally:
        for handler in logger.handlers:
            try:
                handler.close()
            except Exception:
                pass
        logger.handlers = original_handlers
        logger.setLevel(original_level)
        logger.propagate = original_propagate


def test_suppress_spikesort_external_debug_output_preserves_rich_root_logs(tmp_path: Path) -> None:
    logger = logging.getLogger("kilosort")
    root = logging.getLogger()
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate
    original_root_handlers = list(root.handlers)
    original_root_level = root.level
    messages: list[str] = []

    class RichHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(self.format(record))

    raw_stream = io.StringIO()
    raw_handler = logging.StreamHandler(raw_stream)
    file_path = tmp_path / "kilosort4.log"
    file_handler = logging.FileHandler(file_path)
    rich_handler = RichHandler()
    rich_handler.setFormatter(logging.Formatter("%(message)s"))
    root.handlers = [rich_handler]
    root.setLevel(logging.INFO)
    logger.handlers = [raw_handler, file_handler]
    logger.setLevel(logging.INFO)
    logger.propagate = True

    try:
        with suppress_spikesort_external_debug_output(enabled=False):
            logger.info("kilosort rich line")
        assert raw_stream.getvalue() == ""
        assert messages == ["kilosort rich line"]
        assert "kilosort rich line" in file_path.read_text(encoding="utf-8")
    finally:
        for handler in logger.handlers:
            try:
                handler.close()
            except Exception:
                pass
        for handler in root.handlers:
            try:
                handler.close()
            except Exception:
                pass
        logger.handlers = original_handlers
        logger.setLevel(original_level)
        logger.propagate = original_propagate
        root.handlers = original_root_handlers
        root.setLevel(original_root_level)


def test_build_local_kilosort_kwargs_translates_shared_sorter_params() -> None:
    inputs = SpikesortInputs(
        h5_path=Path("test.h5"),
        stream_id="well001",
        mea_output_root=Path("/tmp/out"),
        ks_batch_duration_s=0.5,
        ks_th_universal=8,
        ks_th_learned=7,
        ks_th_single_ch=5,
        ks_cluster_downsampling=15,
        ks_nearest_chans=12,
        ks_max_channel_distance=40,
    )

    assert build_local_kilosort_kwargs(inputs=inputs, recording=_FakeRecording()) == {
        "batch_size": 10_000,
        "Th_universal": 8.0,
        "Th_learned": 7.0,
        "Th_single_ch": 5.0,
        "cluster_downsampling": 15,
        "nearest_chans": 12,
        "max_channel_distance": 40.0,
    }


def test_run_local_spikeinterface_sort_stage_uses_in_process_sorter(tmp_path: Path) -> None:
    well_out_dir = tmp_path / "well001"
    recording_dir = well_out_dir / "preprocess_outputs/preprocessed_recording"
    recording_dir.mkdir(parents=True)
    (recording_dir / "traces_cached_seg0.raw").write_bytes(b"raw")
    stage_output_root_dir = well_out_dir / "spikesort_outputs"
    fake_si = _FakeSpikeInterface()
    fake_sorters = _FakeSorters()

    inputs = SpikesortInputs(
        h5_path=tmp_path / "test.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        sort_engine="local_spikeinterface",
        local_spikeinterface_enabled=True,
        preprocess_concat_recording_relpath="preprocess_outputs/preprocessed_recording",
        sort_use_lazy_source=False,
        sorter="kilosort4",
        ks_batch_size=4096,
        ks_th_universal=8,
        n_jobs=2,
        chunk_duration="1s",
        verbose=False,
        progress_bar=True,
        debug_outputs=True,
        force_restart=True,
        local_spikeinterface_remove_existing_on_force_restart=True,
        local_spikeinterface_run_sorter_kwargs={"delete_output_folder": False},
    )

    outputs = run_local_spikeinterface_sort_stage(
        inputs=inputs,
        well_out_dir=well_out_dir,
        stage_output_root_dir=stage_output_root_dir,
        logger=logging.getLogger("test.local_spikeinterface"),
        si_module=fake_si,
        sorters_module=fake_sorters,
    )

    assert outputs.output_dir == stage_output_root_dir.resolve()
    assert outputs.recording_dir == recording_dir.resolve()
    assert outputs.sorter_output_dir == (stage_output_root_dir / "sorter_output").resolve()
    assert outputs.analyzer_dir == (stage_output_root_dir / "analyzer_output").resolve()
    assert fake_si.global_job_kwargs == {"n_jobs": 2, "chunk_duration": "1s", "progress_bar": True}
    assert fake_sorters.run_sorter_kwargs["sorter_name"] == "kilosort4"
    assert fake_sorters.run_sorter_kwargs["verbose"] is False
    assert fake_sorters.run_sorter_kwargs["recording"].__class__ is _FakeRecording
    assert fake_sorters.run_sorter_kwargs["folder"] == outputs.sorter_output_dir
    assert fake_sorters.run_sorter_kwargs["batch_size"] == 4096
    assert fake_sorters.run_sorter_kwargs["Th_universal"] == 8.0
    assert fake_sorters.run_sorter_kwargs["remove_existing_folder"] is True
    assert fake_sorters.run_sorter_kwargs["delete_output_folder"] is False
    assert "docker_image" not in fake_sorters.run_sorter_kwargs
    assert "output_folder" not in fake_sorters.run_sorter_kwargs
    assert fake_si.analyzer_kwargs["folder"] == outputs.analyzer_dir


def test_run_local_spikeinterface_sort_stage_rejects_container_kwargs(tmp_path: Path) -> None:
    well_out_dir = tmp_path / "well001"
    recording_dir = well_out_dir / "preprocess_outputs/preprocessed_recording"
    recording_dir.mkdir(parents=True)
    (recording_dir / "traces_cached_seg0.raw").write_bytes(b"raw")

    inputs = SpikesortInputs(
        h5_path=tmp_path / "test.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        sort_engine="local_spikeinterface",
        local_spikeinterface_enabled=True,
        preprocess_concat_recording_relpath="preprocess_outputs/preprocessed_recording",
        local_spikeinterface_run_sorter_kwargs={"docker_image": "should/not:run"},
    )

    with pytest.raises(ValueError, match="container execution"):
        run_local_spikeinterface_sort_stage(
            inputs=inputs,
            well_out_dir=well_out_dir,
            stage_output_root_dir=well_out_dir / "spikesort_outputs",
            logger=logging.getLogger("test.local_spikeinterface"),
            si_module=_FakeSpikeInterface(),
            sorters_module=_FakeSorters(),
        )