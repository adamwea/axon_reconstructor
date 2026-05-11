from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from axon_recon.pipeline.stages.spikesort.models.inputs import SpikesortInputs
from axon_recon.pipeline.stages.spikesort.runner import (
    run_spikesort_bootstrap_concat_binary_stage,
    run_spikesort_cleanup_concat_binary_stage,
    run_spikesort_merge_stage,
    run_spikesort_stage,
)
from axon_recon.pipeline.cpu_allocation import TaskSlot, task_slot_context

_TEST_TASK_SLOT = TaskSlot(
    slot_id=0,
    logical_cpus=tuple(range(10)),
    core_ids=tuple(range(10)),
    package_ids=(0,),
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_unit_helpers_prefer_spikeinterface_methods_for_counts_and_ids() -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _FakeSorting:
        def get_unit_ids(self):
            return [101, 102, 102]

        def get_num_units(self):
            return 2

    class _FakeAnalyzer:
        def __init__(self) -> None:
            self.sorting = _FakeSorting()

    analyzer = _FakeAnalyzer()
    assert spikesort_runner._unit_ids_from_obj(analyzer) == ["101", "102"]
    assert spikesort_runner._unit_count(analyzer) == 2


def test_unit_helpers_handle_bool_ambiguous_unit_id_sequences() -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _BoolAmbiguousSequence:
        def __init__(self, values):
            self._values = list(values)

        def __iter__(self):
            return iter(self._values)

        def __len__(self):
            return len(self._values)

        def __bool__(self):
            raise ValueError("ambiguous truth value")

    class _FakeSorting:
        def get_unit_ids(self):
            return _BoolAmbiguousSequence([201, 202])

    class _FakeAnalyzer:
        def __init__(self) -> None:
            self.sorting = _FakeSorting()

    analyzer = _FakeAnalyzer()
    assert spikesort_runner._unit_ids_from_obj(analyzer) == ["201", "202"]
    assert spikesort_runner._unit_count(analyzer) == 2


def test_extract_unit_locations_from_analyzer_computes_missing_extension() -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _FakeExtension:
        def get_data(self):
            return [[10.0, 20.0], [30.0, 40.0]]

    class _FakeSorting:
        def get_unit_ids(self):
            return [11, 12]

    class _FakeAnalyzer:
        def __init__(self) -> None:
            self.sorting = _FakeSorting()
            self._has_unit_locations = False
            self.compute_calls: list[object] = []

        def has_extension(self, name: str) -> bool:
            return bool(name == "unit_locations" and self._has_unit_locations)

        def compute(self, extension_name):
            self.compute_calls.append(extension_name)
            self._has_unit_locations = True

        def get_extension(self, name: str):
            assert name == "unit_locations"
            return _FakeExtension()

    analyzer = _FakeAnalyzer()
    locations, error = spikesort_runner._extract_unit_locations_from_analyzer(analyzer=analyzer)

    assert error is None
    assert analyzer.compute_calls == ["unit_locations"]
    assert locations == {
        "11": {"x_um": 10.0, "y_um": 20.0},
        "12": {"x_um": 30.0, "y_um": 40.0},
    }


def test_extract_unit_locations_from_analyzer_computes_dependency_chain() -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _FakeExtension:
        def get_data(self):
            return [[100.0, 200.0], [300.0, 400.0]]

    class _FakeSorting:
        def get_unit_ids(self):
            return [101, 202]

    class _FakeAnalyzer:
        def __init__(self) -> None:
            self.sorting = _FakeSorting()
            self._computed: set[str] = set()
            self.compute_calls: list[str] = []

        def has_extension(self, name: str) -> bool:
            return bool(name in self._computed)

        def compute(self, extension_name):
            if isinstance(extension_name, (list, tuple)):
                if len(extension_name) != 1:
                    raise AssertionError("expected single extension")
                extension_name = extension_name[0]

            name = str(extension_name)
            self.compute_calls.append(name)
            if name == "unit_locations" and not {"random_spikes", "waveforms", "templates"}.issubset(self._computed):
                raise AssertionError("Extension unit_locations requires templates to be computed first")
            if name == "templates" and not {"random_spikes", "waveforms"}.issubset(self._computed):
                raise AssertionError("Extension templates requires random_spikes|waveforms to be computed first")
            if name == "waveforms" and "random_spikes" not in self._computed:
                raise AssertionError("Extension waveforms requires random_spikes")
            self._computed.add(name)

        def get_extension(self, name: str):
            assert name == "unit_locations"
            if "unit_locations" not in self._computed:
                return None
            return _FakeExtension()

    analyzer = _FakeAnalyzer()
    locations, error = spikesort_runner._extract_unit_locations_from_analyzer(analyzer=analyzer)

    assert error is None
    assert locations == {
        "101": {"x_um": 100.0, "y_um": 200.0},
        "202": {"x_um": 300.0, "y_um": 400.0},
    }
    assert {"random_spikes", "waveforms", "templates", "unit_locations"}.issubset(set(analyzer.compute_calls))


def test_run_spikesort_bootstrap_concat_binary_stage_materializes_binary(monkeypatch, tmp_path: Path) -> None:
    from axon_recon.pipeline.stages.preprocess.core import concat_segments as concat_segments_module
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    well_out_dir = tmp_path / "well001"
    monkeypatch.setattr(spikesort_runner, "compute_mea_analysis_output_dir", lambda **kwargs: well_out_dir)

    captured: dict[str, object] = {}

    def _fake_run_concat_segments_core(**kwargs):
        captured.update(kwargs)
        recording_dir = Path(kwargs["recording_dir"])
        recording_dir.mkdir(parents=True, exist_ok=True)
        (recording_dir / "traces_cached_seg0.raw").write_bytes(b"raw")
        concat_manifest_path = Path(kwargs["concat_manifest_path"])
        concat_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        concat_manifest_path.write_text("{}\n", encoding="utf-8")
        return {
            "output_mode": "binary",
            "materialized_recording": True,
            "saved": True,
            "reused_existing": False,
        }

    monkeypatch.setattr(concat_segments_module, "run_concat_segments_core", _fake_run_concat_segments_core)

    stage_config = SimpleNamespace(
        output_rel_root="spikesort_outputs",
        bootstrap_concat_binary_enabled=True,
        bootstrap_concat_binary_cache_relpath="cache/bootstrap_concat_binary",
        bootstrap_concat_binary_recording_relpath="cache/bootstrap_concat_binary/recording",
        bootstrap_concat_binary_manifest_relpath="cache/bootstrap_concat_binary/concat_segments_manifest.json",
        bootstrap_concat_binary_summary_json_relpath="cache/bootstrap_concat_binary/bootstrap_concat_binary_summary.json",
        bootstrap_concat_binary_source_segment_manifest_relpath="preprocess_outputs/preprocessed_segments/manifest.json",
        bootstrap_concat_binary_debug_limit_segments_per_well=2,
        debug_limit_segments_per_well=None,
        bootstrap_concat_binary_overwrite_existing=False,
        bootstrap_concat_binary_overwrite_on_force_restart=True,
        bootstrap_concat_binary_n_jobs=2,
        bootstrap_concat_binary_chunk_duration="1s",
        bootstrap_concat_binary_progress_bar=False,
        n_jobs=4,
        chunk_duration="2s",
    )

    with task_slot_context(_TEST_TASK_SLOT):
        result = run_spikesort_bootstrap_concat_binary_stage(
            h5_path=tmp_path / "test.h5",
            stream_id="well001",
            mea_output_root=tmp_path,
            output_rel_root="spikesort_outputs",
            stage_config=stage_config,
            force_restart=True,
        )

    assert Path(captured["recording_dir"]) == well_out_dir / "spikesort_outputs/cache/bootstrap_concat_binary/recording"
    assert captured["output_mode"] == "binary"
    assert captured["overwrite_saved_recording"] is True
    assert captured["n_jobs"] == 2
    assert captured["limit_segments_per_well"] == 2
    assert result.summary_json.exists()
    payload = _read_json(result.summary_json)
    assert payload["status"] == "ok"
    assert payload["limit_segments_per_well"] == 2
    assert payload["recording_dir"].endswith("spikesort_outputs/cache/bootstrap_concat_binary/recording")


def test_run_spikesort_cleanup_concat_binary_stage_removes_cache(monkeypatch, tmp_path: Path) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    well_out_dir = tmp_path / "well001"
    cache_dir = well_out_dir / "spikesort_outputs/cache/bootstrap_concat_binary"
    cache_dir.mkdir(parents=True)
    (cache_dir / "traces_cached_seg0.raw").write_bytes(b"raw")
    monkeypatch.setattr(spikesort_runner, "compute_mea_analysis_output_dir", lambda **kwargs: well_out_dir)

    stage_config = SimpleNamespace(
        cleanup_concat_binary_enabled=True,
        cleanup_concat_binary_relpath="cache/bootstrap_concat_binary",
        cleanup_concat_binary_summary_json_relpath="cache/bootstrap_concat_binary_cleanup_summary.json",
    )

    result = run_spikesort_cleanup_concat_binary_stage(
        h5_path=tmp_path / "test.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        output_rel_root="spikesort_outputs",
        stage_config=stage_config,
        force_restart=False,
    )

    assert not cache_dir.exists()
    assert result.summary_json == well_out_dir / "spikesort_outputs/cache/bootstrap_concat_binary_cleanup_summary.json"
    payload = _read_json(result.summary_json)
    assert payload["status"] == "ok"
    assert payload["removed_paths"] == [str(cache_dir.resolve())]


def test_run_spikesort_stage_generates_sort_summary_artifacts(monkeypatch, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _FakeSorting:
        def get_unit_ids(self):
            return [1, 2, 3]

        def get_num_segments(self):
            return 1

        def get_unit_spike_train(self, unit_id=None, segment_index=0):
            mapping = {
                1: [0, 1, 2],
                2: list(range(7)),
                3: list(range(2)),
            }
            return mapping[unit_id]

    class _LegacyOutputs:
        def __init__(self, output_dir: Path):
            self.output_dir = output_dir
            self.recording_dir = output_dir / "recording"
            self.sorter_output_dir = output_dir / "sorter_output"
            self.analyzer_dir = output_dir / "analyzer_output"
            self.merged_sorting_dir = None
            self.merged_sorter_output_dir = None

    well_out_dir = tmp_path / "well001"
    stage_output_root_dir = well_out_dir / "spikesort_outputs"
    sorter_output_dir = stage_output_root_dir / "sorter_output"
    nested_kilosort_dir = sorter_output_dir / "sorter_output"
    nested_kilosort_dir.mkdir(parents=True)
    (nested_kilosort_dir / "spike_times.npy").write_bytes(b"fake")
    (nested_kilosort_dir / "spike_clusters.npy").write_bytes(b"fake")
    (nested_kilosort_dir / "cluster_KSLabel.tsv").write_text(
        "cluster_id\tKSLabel\n1\tgood\n2\tmua\n3\tgood\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(spikesort_runner, "compute_mea_analysis_output_dir", lambda **kwargs: well_out_dir)
    monkeypatch.setattr(
        spikesort_runner,
        "run_legacy_spikesorting_stage",
        lambda **kwargs: _LegacyOutputs(stage_output_root_dir),
    )
    monkeypatch.setattr(
        spikesort_runner,
        "_load_sorting_from_sorter_output_dir",
        lambda **kwargs: _FakeSorting(),
    )
    monkeypatch.setattr(spikesort_runner, "_import_spikeinterface_full_module", lambda: object())

    inputs = SpikesortInputs(
        h5_path=tmp_path / "test.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        summarize_sort_enabled=True,
        summarize_sort_emit_logs=True,
        summarize_sort_generate_artifacts=True,
    )

    with caplog.at_level("INFO"):
        result = run_spikesort_stage(inputs)

    payload = _read_json(result.summary_json)
    summarize_payload = payload["summarize_sort"]
    assert summarize_payload["status"] == "ok"
    assert summarize_payload["unit_count"] == 3
    assert summarize_payload["counts_by_label"] == {"good": 2, "mua": 1}
    assert summarize_payload["spike_count_stats"] == {"min": 2, "max": 7}
    assert [unit["unit_id"] for unit in summarize_payload["units"]] == ["1", "2", "3"]
    assert summarize_payload["sorter_output_dir"].endswith("sorter_output/sorter_output")
    assert summarize_payload["label_sources"]["cluster_kslabel_tsv"].endswith("sorter_output/sorter_output/cluster_KSLabel.tsv")
    assert result.outputs["summarize_sort.summary_json"].endswith("summarize_sort_summary.json")
    assert result.outputs["summarize_sort.units_tsv"].endswith("summarize_sort_units.tsv")
    assert "Sort summary [stream=well001] units=3" in caplog.text


def test_run_spikesort_stage_dispatches_local_engine_without_legacy(monkeypatch, tmp_path: Path) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner
    from axon_recon.pipeline.stages.spikesort.core.local_spikeinterface import LocalSpikeInterfaceSortOutputs

    well_out_dir = tmp_path / "well001"
    monkeypatch.setattr(spikesort_runner, "compute_mea_analysis_output_dir", lambda **kwargs: well_out_dir)

    def _fail_legacy_call(**kwargs):
        raise AssertionError("legacy MEA_Analysis route should not run for local_spikeinterface")

    monkeypatch.setattr(spikesort_runner, "run_legacy_spikesorting_stage", _fail_legacy_call)

    captured: dict[str, object] = {}

    def _fake_local_run(**kwargs):
        captured.update(kwargs)
        stage_output_root_dir = Path(kwargs["stage_output_root_dir"])
        sorter_output_dir = stage_output_root_dir / "sorter_output"
        analyzer_dir = stage_output_root_dir / "analyzer_output"
        sorter_output_dir.mkdir(parents=True, exist_ok=True)
        analyzer_dir.mkdir(parents=True, exist_ok=True)
        return LocalSpikeInterfaceSortOutputs(
            recording_dir=well_out_dir / "preprocess_outputs/preprocessed_recording",
            sorter_output_dir=sorter_output_dir,
            output_dir=stage_output_root_dir,
            analyzer_dir=analyzer_dir,
        )

    monkeypatch.setattr(spikesort_runner, "run_local_spikeinterface_sort_stage", _fake_local_run)

    inputs = SpikesortInputs(
        h5_path=tmp_path / "test.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        sort_engine="local_spikeinterface",
        local_spikeinterface_enabled=True,
    )

    result = run_spikesort_stage(inputs)

    assert captured["inputs"] is inputs
    assert result.outputs["local_spikeinterface.spikesort_out_dir"].endswith("spikesort_outputs")
    assert "legacy.spikesort_out_dir" not in result.outputs
    payload = _read_json(result.summary_json)
    assert payload["sort_engine"] == "local_spikeinterface"


def test_run_spikesort_stage_rejects_mea_analysis_inside_container(monkeypatch, tmp_path: Path) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    well_out_dir = tmp_path / "well001"
    monkeypatch.setattr(spikesort_runner, "compute_mea_analysis_output_dir", lambda **kwargs: well_out_dir)
    monkeypatch.setenv("AXON_RECON_IN_CONTAINER", "1")
    monkeypatch.delenv("AXON_RECON_ALLOW_CONTAINER_MEA_ANALYSIS", raising=False)

    def _fail_legacy_call(**kwargs):
        raise AssertionError("legacy MEA_Analysis route should be blocked inside the container")

    monkeypatch.setattr(spikesort_runner, "run_legacy_spikesorting_stage", _fail_legacy_call)

    inputs = SpikesortInputs(
        h5_path=tmp_path / "test.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        sort_engine="mea_analysis",
    )

    with pytest.raises(RuntimeError, match="local_spikeinterface"):
        run_spikesort_stage(inputs)


def test_run_spikesort_summarize_sort_writes_summary_when_artifacts_disabled(monkeypatch, tmp_path: Path) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _FakeSorting:
        def get_unit_ids(self):
            return [10]

        def get_num_segments(self):
            return 1

        def get_unit_spike_train(self, unit_id=None, segment_index=0):
            return [0, 1, 2, 3]

    well_out_dir = tmp_path / "well001"
    stage_output_root_dir = well_out_dir / "spikesort_outputs"
    sorter_output_dir = stage_output_root_dir / "sorter_output"
    sorter_output_dir.mkdir(parents=True)
    (sorter_output_dir / "cluster_group.tsv").write_text(
        "cluster_id\tgroup\n10\tunsorted\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(spikesort_runner, "compute_mea_analysis_output_dir", lambda **kwargs: well_out_dir)
    monkeypatch.setattr(
        spikesort_runner,
        "_load_sorting_from_sorter_output_dir",
        lambda **kwargs: _FakeSorting(),
    )
    monkeypatch.setattr(spikesort_runner, "_import_spikeinterface_full_module", lambda: object())

    inputs = SpikesortInputs(
        h5_path=tmp_path / "test.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        summarize_sort_enabled=True,
        summarize_sort_emit_logs=False,
        summarize_sort_generate_artifacts=False,
    )

    result = spikesort_runner.run_spikesort_summarize_sort(inputs)

    payload = _read_json(result.summary_json)
    assert payload["status"] == "ok"
    assert payload["counts_by_label"] == {"unsorted": 1}
    assert result.outputs["summarize_sort.summary_json"].endswith("summarize_sort_summary.json")


def test_print_spikesort_summarize_aggregate_includes_counts_by_label(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from axon_recon.pipeline.stages.spikesort.orchestrators.summarize_sort import _print_spikesort_summarize_aggregate

    summary_json = tmp_path / "summarize_sort_summary.json"
    summary_json.write_text(
        json.dumps({"counts_by_label": {"good": 12, "mua": 3, "noise": 1}}),
        encoding="utf-8",
    )

    aggregate = SimpleNamespace(
        stage="spikesort.summarize_sort",
        total_targets=1,
        succeeded_targets=1,
        failed_targets=0,
        target_results=[
            SimpleNamespace(
                target=SimpleNamespace(dataset_index=0, stream_id="well000"),
                status="ok",
                result=SimpleNamespace(summary_json=summary_json),
                error=None,
            )
        ],
    )

    exit_code = _print_spikesort_summarize_aggregate(aggregate)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "labels={'good': 12, 'mua': 3, 'noise': 1}" in captured.out


def test_ensure_merge_analyzer_extensions_uses_all_random_spikes_method() -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _FakeAnalyzer:
        def __init__(self) -> None:
            self._computed: set[str] = set()
            self.compute_calls: list[tuple[object, dict[str, object]]] = []

        def has_extension(self, name: str) -> bool:
            return bool(name in self._computed)

        def compute(self, extension_name, **kwargs):
            self.compute_calls.append((extension_name, dict(kwargs)))
            if isinstance(extension_name, dict):
                raise TypeError("dict compute signature not supported in test fake")
            if isinstance(extension_name, (list, tuple)):
                if len(extension_name) != 1:
                    raise AssertionError("expected single extension")
                extension_name = extension_name[0]
            self._computed.add(str(extension_name))

    analyzer = _FakeAnalyzer()
    with task_slot_context(_TEST_TASK_SLOT):
        computed = spikesort_runner._ensure_merge_analyzer_extensions(
            analyzer=analyzer,
            stage_config=SimpleNamespace(
                merge_template_random_spikes_method="all",
                merge_template_random_spikes_max_spikes_per_unit=321,
                merge_template_random_spikes_margin_size=17,
                merge_template_random_spikes_seed=42,
                merge_analyzer_n_jobs=2,
                merge_analyzer_chunk_duration="0.5s",
                merge_analyzer_waveforms_ms_before=0.75,
                merge_analyzer_waveforms_ms_after=1.5,
                merge_analyzer_waveforms_dtype="float32",
                n_jobs=None,
                chunk_duration=None,
            ),
            include_unit_locations=True,
        )

    assert computed == ["random_spikes", "waveforms", "templates", "unit_locations"]
    assert analyzer.compute_calls[0] == (
        {"random_spikes": {"method": "all", "max_spikes_per_unit": 321, "margin_size": 17, "seed": 42}},
        {"n_jobs": 2, "chunk_duration": "0.5s", "progress_bar": True},
    )
    assert analyzer.compute_calls[1] == (
        "random_spikes",
        {
            "method": "all",
            "max_spikes_per_unit": 321,
            "margin_size": 17,
            "seed": 42,
            "n_jobs": 2,
            "chunk_duration": "0.5s",
            "progress_bar": True,
        },
    )
    assert analyzer.compute_calls[2] == (
        "waveforms",
        {
            "n_jobs": 2,
            "chunk_duration": "0.5s",
            "ms_before": 0.75,
            "ms_after": 1.5,
            "dtype": "float32",
            "progress_bar": True,
        },
    )
    assert analyzer.compute_calls[3] == (
        "templates",
        {
            "n_jobs": 2,
            "chunk_duration": "0.5s",
            "ms_before": 0.75,
            "ms_after": 1.5,
            "progress_bar": True,
        },
    )


def test_ensure_merge_analyzer_extensions_uses_percentage_random_spikes_method() -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _FakeAnalyzer:
        def __init__(self) -> None:
            self._computed: set[str] = set()
            self.compute_calls: list[tuple[object, dict[str, object]]] = []

        def has_extension(self, name: str) -> bool:
            return bool(name in self._computed)

        def compute(self, extension_name, **kwargs):
            self.compute_calls.append((extension_name, dict(kwargs)))
            if isinstance(extension_name, dict):
                raise TypeError("dict compute signature not supported in test fake")
            if isinstance(extension_name, (list, tuple)):
                if len(extension_name) != 1:
                    raise AssertionError("expected single extension")
                extension_name = extension_name[0]
            self._computed.add(str(extension_name))

    analyzer = _FakeAnalyzer()
    with task_slot_context(_TEST_TASK_SLOT):
        computed = spikesort_runner._ensure_merge_analyzer_extensions(
            analyzer=analyzer,
            stage_config=SimpleNamespace(
                merge_template_random_spikes_method="percentage",
                merge_template_random_spikes_percentage=0.75,
                merge_template_random_spikes_min_spikes_per_unit=1000,
                merge_template_random_spikes_log_before_after_spike_counts=True,
                merge_template_random_spikes_max_spikes_per_unit=5000,
                merge_template_random_spikes_margin_size=17,
                merge_template_random_spikes_seed=42,
                merge_analyzer_n_jobs=2,
                merge_analyzer_chunk_duration="0.5s",
                merge_analyzer_waveforms_ms_before=0.75,
                merge_analyzer_waveforms_ms_after=1.5,
                merge_analyzer_waveforms_dtype="float32",
                n_jobs=None,
                chunk_duration=None,
            ),
            include_unit_locations=False,
        )

    assert computed == ["random_spikes", "waveforms", "templates"]
    assert analyzer.compute_calls[0] == (
        "random_spikes",
        {
            "method": "percentage",
            "percentage": 0.75,
            "min_spikes_per_unit": 1000,
            "log_before_after_spike_counts": True,
            "max_spikes_per_unit": 5000,
            "margin_size": 17,
            "seed": 42,
            "n_jobs": 2,
            "chunk_duration": "0.5s",
            "progress_bar": True,
        },
    )


def test_ensure_merge_analyzer_extensions_omits_max_cap_for_percentage_mode_when_unset() -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _FakeAnalyzer:
        def __init__(self) -> None:
            self._computed: set[str] = set()
            self.compute_calls: list[tuple[object, dict[str, object]]] = []

        def has_extension(self, name: str) -> bool:
            return bool(name in self._computed)

        def compute(self, extension_name, **kwargs):
            self.compute_calls.append((extension_name, dict(kwargs)))
            if isinstance(extension_name, (list, tuple)):
                extension_name = extension_name[0]
            self._computed.add(str(extension_name))

    analyzer = _FakeAnalyzer()
    with task_slot_context(_TEST_TASK_SLOT):
        spikesort_runner._ensure_merge_analyzer_extensions(
            analyzer=analyzer,
            stage_config=SimpleNamespace(
                merge_template_random_spikes_method="percentage",
                merge_template_random_spikes_percentage=0.75,
                merge_template_random_spikes_max_spikes_per_unit=None,
                merge_template_random_spikes_min_spikes_per_unit=None,
                merge_template_random_spikes_log_before_after_spike_counts=False,
                merge_template_random_spikes_margin_size=None,
                merge_template_random_spikes_seed=42,
                merge_analyzer_n_jobs=2,
                merge_analyzer_chunk_duration="0.5s",
                merge_analyzer_waveforms_ms_before=0.75,
                merge_analyzer_waveforms_ms_after=1.5,
                merge_analyzer_waveforms_dtype="float32",
                n_jobs=None,
                chunk_duration=None,
            ),
            include_unit_locations=False,
        )

    assert analyzer.compute_calls[0] == (
        "random_spikes",
        {
            "method": "percentage",
            "percentage": 0.75,
            "seed": 42,
            "n_jobs": 2,
            "chunk_duration": "0.5s",
            "progress_bar": True,
        },
    )


def test_release_loaded_analyzer_extensions_clears_loaded_extensions_without_touching_disk() -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _FakeExtension:
        def __init__(self) -> None:
            self.data = {"value": [1, 2, 3]}
            self._some_spikes = [4, 5, 6]

    class _FakeAnalyzer:
        def __init__(self) -> None:
            self.extensions = {
                "random_spikes": _FakeExtension(),
                "templates": _FakeExtension(),
            }

        def get_loaded_extension_names(self):
            return list(self.extensions.keys())

    analyzer = _FakeAnalyzer()
    released = spikesort_runner._release_loaded_analyzer_extensions(analyzer=analyzer)

    assert set(released) == {"random_spikes", "templates"}
    assert analyzer.extensions == {}


def test_ensure_bombcell_metric_extensions_uses_bombcell_job_kwargs() -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _FakeTable:
        def __init__(self, columns: list[str]) -> None:
            self.columns = list(columns)

    class _FakeExtension:
        def __init__(self, columns: list[str]) -> None:
            self._table = _FakeTable(columns)

        def get_data(self):
            return self._table

    class _FakeAnalyzer:
        def __init__(self) -> None:
            self._computed: set[str] = set()
            self.compute_calls: list[tuple[object, dict[str, object]]] = []
            self._extensions: dict[str, _FakeExtension] = {}

        def has_extension(self, name: str) -> bool:
            return bool(name in self._computed)

        def get_extension(self, name: str):
            return self._extensions.get(name)

        def compute(self, extension_name, **kwargs):
            self.compute_calls.append((extension_name, dict(kwargs)))
            if isinstance(extension_name, dict):
                raise TypeError("dict compute signature not supported in test fake")
            if isinstance(extension_name, (list, tuple)):
                if len(extension_name) != 1:
                    raise AssertionError("expected single extension")
                extension_name = extension_name[0]
            extension_name = str(extension_name)
            self._computed.add(extension_name)
            if extension_name == "template_metrics":
                self._extensions[extension_name] = _FakeExtension(list(kwargs.get("metric_names", [])))
            elif extension_name == "quality_metrics":
                self._extensions[extension_name] = _FakeExtension(list(spikesort_runner._BOMBCELL_QUALITY_METRIC_COLUMNS))
            else:
                self._extensions[extension_name] = _FakeExtension([])

    analyzer = _FakeAnalyzer()
    with task_slot_context(_TEST_TASK_SLOT):
        computed = spikesort_runner._ensure_bombcell_metric_extensions(
            analyzer=analyzer,
            stage_config=SimpleNamespace(
                bombcell_label_template_random_spikes_method="all",
                bombcell_label_template_random_spikes_max_spikes_per_unit=500,
                bombcell_label_template_random_spikes_margin_size=None,
                bombcell_label_template_random_spikes_seed=None,
                bombcell_label_analyzer_n_jobs=3,
                bombcell_label_analyzer_chunk_duration="0.25s",
                bombcell_label_analyzer_waveforms_ms_before=1.0,
                bombcell_label_analyzer_waveforms_ms_after=2.0,
                bombcell_label_analyzer_waveforms_dtype=None,
                bombcell_label_analyzer_compute_sparsity=True,
                bombcell_label_analyzer_sparsity_method="radius",
                bombcell_label_analyzer_sparsity_radius_um=100.0,
                bombcell_label_analyzer_sparsity_num_channels=5,
                bombcell_label_analyzer_sparsity_threshold=5.0,
                bombcell_label_analyzer_sparsity_peak_sign="neg",
                bombcell_label_analyzer_sparsity_num_spikes_for_sparsity=100,
                bombcell_label_analyzer_sparsity_by_property=None,
                merge_template_random_spikes_method="default",
                merge_analyzer_n_jobs=99,
                merge_analyzer_chunk_duration="9s",
                n_jobs=None,
                chunk_duration=None,
            ),
        )

    assert computed == [
        "random_spikes",
        "waveforms",
        "templates",
        "noise_levels",
        "spike_amplitudes",
        "spike_locations",
        "template_metrics",
        "quality_metrics",
    ]
    assert analyzer.compute_calls[0][1].get("n_jobs") == 3
    assert analyzer.compute_calls[0][1].get("chunk_duration") == "0.25s"
    assert analyzer.compute_calls[1] == (
        "random_spikes",
        {
            "method": "all",
            "max_spikes_per_unit": 500,
            "n_jobs": 3,
            "chunk_duration": "0.25s",
            "progress_bar": True,
        },
    )
    assert analyzer.compute_calls[2] == (
        "waveforms",
        {"n_jobs": 3, "chunk_duration": "0.25s", "ms_before": 1.0, "ms_after": 2.0, "progress_bar": True},
    )
    assert ("noise_levels", {"n_jobs": 3, "chunk_duration": "0.25s", "progress_bar": True}) in analyzer.compute_calls
    assert ("spike_amplitudes", {"n_jobs": 3, "chunk_duration": "0.25s", "progress_bar": True}) in analyzer.compute_calls
    assert ("spike_locations", {"n_jobs": 3, "chunk_duration": "0.25s", "progress_bar": True}) in analyzer.compute_calls
    assert (
        "template_metrics",
        {
            "n_jobs": 3,
            "chunk_duration": "0.25s",
            "metric_names": list(spikesort_runner._BOMBCELL_TEMPLATE_METRIC_NAMES),
            "include_multi_channel_metrics": True,
            "progress_bar": True,
        },
    ) in analyzer.compute_calls
    assert (
        "quality_metrics",
        {
            "n_jobs": 3,
            "chunk_duration": "0.25s",
            "metric_names": list(spikesort_runner._BOMBCELL_QUALITY_METRIC_NAMES),
            "skip_pc_metrics": True,
            "progress_bar": True,
        },
    ) in analyzer.compute_calls


def test_ensure_bombcell_metric_extensions_recomputes_incomplete_metric_extensions() -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _FakeTable:
        def __init__(self, columns: list[str]) -> None:
            self.columns = list(columns)

    class _FakeExtension:
        def __init__(self, columns: list[str]) -> None:
            self._table = _FakeTable(columns)

        def get_data(self):
            return self._table

    class _FakeAnalyzer:
        def __init__(self) -> None:
            self._computed = {
                "random_spikes",
                "waveforms",
                "templates",
                "noise_levels",
                "spike_amplitudes",
                "spike_locations",
                "template_metrics",
                "quality_metrics",
            }
            self.compute_calls: list[tuple[str, dict[str, object]]] = []
            self._extensions = {
                "template_metrics": _FakeExtension(["peak_to_trough_duration"]),
                "quality_metrics": _FakeExtension(["num_spikes"]),
            }

        def has_extension(self, name: str) -> bool:
            return bool(name in self._computed)

        def get_extension(self, name: str):
            return self._extensions.get(name)

        def compute(self, extension_name, **kwargs):
            extension_name = str(extension_name)
            self.compute_calls.append((extension_name, dict(kwargs)))
            if extension_name == "template_metrics":
                self._extensions[extension_name] = _FakeExtension(list(spikesort_runner._BOMBCELL_TEMPLATE_METRIC_COLUMNS))
            elif extension_name == "quality_metrics":
                self._extensions[extension_name] = _FakeExtension(list(spikesort_runner._BOMBCELL_QUALITY_METRIC_COLUMNS))

    analyzer = _FakeAnalyzer()
    computed = spikesort_runner._ensure_bombcell_metric_extensions(
        analyzer=analyzer,
        stage_config=SimpleNamespace(
            bombcell_label_template_random_spikes_method="all",
            bombcell_label_analyzer_n_jobs=3,
            bombcell_label_analyzer_chunk_duration="0.25s",
            bombcell_label_analyzer_waveforms_ms_before=1.0,
            bombcell_label_analyzer_waveforms_ms_after=2.0,
            bombcell_label_analyzer_waveforms_dtype=None,
            bombcell_label_analyzer_compute_sparsity=False,
            merge_template_random_spikes_method="default",
            merge_analyzer_n_jobs=99,
            merge_analyzer_chunk_duration="9s",
            n_jobs=None,
            chunk_duration=None,
        ),
    )

    assert computed == ["template_metrics", "quality_metrics"]
    assert [name for name, _ in analyzer.compute_calls] == ["template_metrics", "quality_metrics"]


def test_extract_bombcell_label_mapping_accepts_bombcell_label_column() -> None:
    import pandas as pd

    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    labels = pd.DataFrame(
        {"bombcell_label": ["non_soma_good", "mua", ""]},
        index=[1, 2, 3],
    )

    extracted = spikesort_runner._extract_bombcell_label_mapping(labels)

    assert extracted == {"1": "non_soma_good", "2": "mua"}


def test_write_merge_unit_location_reports_inverts_y_axis(tmp_path: Path, monkeypatch) -> None:
    import sys
    from types import SimpleNamespace as _SimpleNamespace

    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _FakeAxis:
        def __init__(self) -> None:
            self.inverted = False

        def set_title(self, title):
            return None

        def set_xlabel(self, label):
            return None

        def set_ylabel(self, label):
            return None

        def set_xlim(self, limits):
            return None

        def set_ylim(self, limits):
            return None

        def invert_yaxis(self):
            self.inverted = True

        def set_aspect(self, aspect, adjustable=None):
            return None

        def grid(self, enabled, alpha=None):
            return None

        def scatter(self, xs, ys, s=None, alpha=None, c=None):
            return None

        def text(self, x, y, text, ha=None, va=None, transform=None):
            return None

        @property
        def transAxes(self):
            return object()

    class _FakeFigure:
        def savefig(self, path, dpi=None):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("fake", encoding="utf-8")

    class _FakePyplot:
        def __init__(self) -> None:
            self.axes_created: list[_FakeAxis] = []

        def subplots(self, nrows=1, ncols=1, figsize=None, constrained_layout=False):
            fig = _FakeFigure()
            if int(nrows) == 1 and int(ncols) == 1:
                ax = _FakeAxis()
                self.axes_created.append(ax)
                return fig, ax
            axes = [_FakeAxis() for _ in range(int(nrows) * int(ncols))]
            self.axes_created.extend(axes)
            return fig, axes

        def close(self, fig):
            return None

    fake_plt = _FakePyplot()
    fake_matplotlib = _SimpleNamespace(pyplot=fake_plt)

    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_plt)

    before_snapshot = {
        "analyzer": {
            "unit_locations_by_unit": {
                "1": {"x_um": 10.0, "y_um": 20.0},
                "2": {"x_um": 30.0, "y_um": 40.0},
            }
        }
    }
    after_snapshot = {
        "analyzer": {
            "unit_locations_by_unit": {
                "1": {"x_um": 11.0, "y_um": 21.0},
            }
        }
    }
    stage_cfg = SimpleNamespace(
        merge_reports_2panel_before_write_png=True,
        merge_reports_2panel_before_write_svg=False,
        merge_reports_2panel_before_relpath="unit_locations_before_merge.png",
        merge_reports_2panel_after_write_png=True,
        merge_reports_2panel_after_write_svg=False,
        merge_reports_2panel_after_relpath="unit_locations_after_merge.png",
        merge_reports_2panel_write_png=True,
        merge_reports_2panel_write_svg=False,
        merge_reports_2panel_relpath="unit_locations_before_after_merge.png",
    )

    payload = spikesort_runner._write_merge_unit_location_reports(
        merge_out_dir=tmp_path,
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
        stage_config=stage_cfg,
    )

    assert payload.get("status") == "ok"
    assert payload.get("before_unit_locations_count") == 2
    assert payload.get("after_unit_locations_count") == 1
    assert all(ax.inverted for ax in fake_plt.axes_created)


def test_write_merge_unit_location_reports_labels_unit_ids_when_enabled(tmp_path: Path, monkeypatch) -> None:
    import sys
    from types import SimpleNamespace as _SimpleNamespace

    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _FakeAxis:
        def __init__(self) -> None:
            self.text_calls: list[dict[str, object]] = []

        def set_title(self, title):
            return None

        def set_xlabel(self, label):
            return None

        def set_ylabel(self, label):
            return None

        def set_xlim(self, limits):
            return None

        def set_ylim(self, limits):
            return None

        def invert_yaxis(self):
            return None

        def set_aspect(self, aspect, adjustable=None):
            return None

        def grid(self, enabled, alpha=None):
            return None

        def scatter(self, xs, ys, s=None, alpha=None, c=None):
            return None

        def text(self, x, y, text, ha=None, va=None, transform=None):
            self.text_calls.append(
                {
                    "x": x,
                    "y": y,
                    "text": str(text),
                    "ha": ha,
                    "va": va,
                    "transform": transform,
                }
            )
            return None

        @property
        def transAxes(self):
            return object()

    class _FakeFigure:
        def savefig(self, path, dpi=None):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("fake", encoding="utf-8")

    class _FakePyplot:
        def __init__(self) -> None:
            self.axes_created: list[_FakeAxis] = []

        def subplots(self, nrows=1, ncols=1, figsize=None, constrained_layout=False):
            fig = _FakeFigure()
            if int(nrows) == 1 and int(ncols) == 1:
                ax = _FakeAxis()
                self.axes_created.append(ax)
                return fig, ax
            axes = [_FakeAxis() for _ in range(int(nrows) * int(ncols))]
            self.axes_created.extend(axes)
            return fig, axes

        def close(self, fig):
            return None

    fake_plt = _FakePyplot()
    fake_matplotlib = _SimpleNamespace(pyplot=fake_plt)

    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_plt)

    before_snapshot = {
        "analyzer": {
            "unit_locations_by_unit": {
                "1": {"x_um": 10.0, "y_um": 20.0},
                "2": {"x_um": 30.0, "y_um": 40.0},
            }
        }
    }
    after_snapshot = {
        "analyzer": {
            "unit_locations_by_unit": {
                "10": {"x_um": 11.0, "y_um": 21.0},
            }
        }
    }
    stage_cfg = SimpleNamespace(
        merge_reports_2panel_label_pre_and_post_units=True,
        merge_reports_2panel_before_write_png=False,
        merge_reports_2panel_before_write_svg=False,
        merge_reports_2panel_after_write_png=False,
        merge_reports_2panel_after_write_svg=False,
        merge_reports_2panel_write_png=True,
        merge_reports_2panel_write_svg=False,
        merge_reports_2panel_relpath="unit_locations_before_after_merge.png",
    )

    payload = spikesort_runner._write_merge_unit_location_reports(
        merge_out_dir=tmp_path,
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
        stage_config=stage_cfg,
    )

    assert payload.get("status") == "ok"
    assert len(fake_plt.axes_created) == 2
    before_texts = [str(call.get("text", "")) for call in fake_plt.axes_created[0].text_calls]
    after_texts = [str(call.get("text", "")) for call in fake_plt.axes_created[1].text_calls]
    assert "1" in before_texts
    assert "2" in before_texts
    assert "10" in after_texts


def test_write_merge_unit_location_reports_plots_highlights_after_other_units_when_enabled(
    tmp_path: Path, monkeypatch
) -> None:
    import sys
    from types import SimpleNamespace as _SimpleNamespace

    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _FakeAxis:
        def __init__(self) -> None:
            self.scatter_calls: list[dict[str, object]] = []

        def set_title(self, title):
            return None

        def set_xlabel(self, label):
            return None

        def set_ylabel(self, label):
            return None

        def set_xlim(self, limits):
            return None

        def set_ylim(self, limits):
            return None

        def invert_yaxis(self):
            return None

        def set_aspect(self, aspect, adjustable=None):
            return None

        def grid(self, enabled, alpha=None):
            return None

        def scatter(self, xs, ys, s=None, alpha=None, c=None):
            self.scatter_calls.append(
                {
                    "xs": list(xs),
                    "ys": list(ys),
                    "colors": list(c),
                }
            )
            return None

        def text(self, x, y, text, ha=None, va=None, transform=None):
            return None

        @property
        def transAxes(self):
            return object()

    class _FakeFigure:
        def savefig(self, path, dpi=None):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("fake", encoding="utf-8")

    class _FakePyplot:
        def __init__(self) -> None:
            self.axes_created: list[_FakeAxis] = []

        def subplots(self, nrows=1, ncols=1, figsize=None, constrained_layout=False):
            fig = _FakeFigure()
            if int(nrows) == 1 and int(ncols) == 1:
                ax = _FakeAxis()
                self.axes_created.append(ax)
                return fig, ax
            axes = [_FakeAxis() for _ in range(int(nrows) * int(ncols))]
            self.axes_created.extend(axes)
            return fig, axes

        def close(self, fig):
            return None

    fake_plt = _FakePyplot()
    fake_matplotlib = _SimpleNamespace(pyplot=fake_plt)

    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_plt)

    before_snapshot = {
        "analyzer": {
            "unit_locations_by_unit": {
                "1": {"x_um": 10.0, "y_um": 20.0},
                "2": {"x_um": 30.0, "y_um": 40.0},
                "3": {"x_um": 50.0, "y_um": 60.0},
            }
        }
    }
    after_snapshot = {"analyzer": {"unit_locations_by_unit": {}}}
    stage_cfg = SimpleNamespace(
        merge_reports_2panel_before_write_png=True,
        merge_reports_2panel_before_write_svg=False,
        merge_reports_2panel_before_relpath="unit_locations_before_merge.png",
        merge_reports_2panel_after_write_png=False,
        merge_reports_2panel_after_write_svg=False,
        merge_reports_2panel_write_png=False,
        merge_reports_2panel_write_svg=False,
        merge_reports_2panel_before_point_color="#7a7a7a",
        merge_reports_2panel_highlight_merges_enabled=True,
        merge_reports_2panel_highlight_merges_linked=False,
        merge_reports_2panel_highlight_before_color="#ff7f0e",
        merge_reports_2panel_highlight_plot_after_other_units=True,
    )

    payload = spikesort_runner._write_merge_unit_location_reports(
        merge_out_dir=tmp_path,
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
        applied_unit_mappings=[{"pre_unit_ids": ["2", "3"], "post_unit_id": None}],
        stage_config=stage_cfg,
    )

    assert payload.get("status") == "ok"
    assert len(fake_plt.axes_created) == 1
    scatter_calls = fake_plt.axes_created[0].scatter_calls
    assert len(scatter_calls) == 2
    assert scatter_calls[0].get("xs") == [10.0]
    assert scatter_calls[0].get("colors") == ["#7a7a7a"]
    assert scatter_calls[1].get("xs") == [30.0, 50.0]
    assert scatter_calls[1].get("colors") == ["#ff7f0e", "#ff7f0e"]


def test_write_merge_unit_location_reports_labels_only_affected_units_when_enabled(
    tmp_path: Path, monkeypatch
) -> None:
    import sys
    from types import SimpleNamespace as _SimpleNamespace

    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _FakeAxis:
        def __init__(self) -> None:
            self.text_calls: list[dict[str, object]] = []

        def set_title(self, title):
            return None

        def set_xlabel(self, label):
            return None

        def set_ylabel(self, label):
            return None

        def set_xlim(self, limits):
            return None

        def set_ylim(self, limits):
            return None

        def invert_yaxis(self):
            return None

        def set_aspect(self, aspect, adjustable=None):
            return None

        def grid(self, enabled, alpha=None):
            return None

        def scatter(self, xs, ys, s=None, alpha=None, c=None):
            return None

        def text(self, x, y, text, ha=None, va=None, transform=None):
            self.text_calls.append(
                {
                    "x": x,
                    "y": y,
                    "text": str(text),
                    "ha": ha,
                    "va": va,
                    "transform": transform,
                }
            )
            return None

        @property
        def transAxes(self):
            return object()

    class _FakeFigure:
        def savefig(self, path, dpi=None):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("fake", encoding="utf-8")

    class _FakePyplot:
        def __init__(self) -> None:
            self.axes_created: list[_FakeAxis] = []

        def subplots(self, nrows=1, ncols=1, figsize=None, constrained_layout=False):
            fig = _FakeFigure()
            if int(nrows) == 1 and int(ncols) == 1:
                ax = _FakeAxis()
                self.axes_created.append(ax)
                return fig, ax
            axes = [_FakeAxis() for _ in range(int(nrows) * int(ncols))]
            self.axes_created.extend(axes)
            return fig, axes

        def close(self, fig):
            return None

    fake_plt = _FakePyplot()
    fake_matplotlib = _SimpleNamespace(pyplot=fake_plt)

    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_plt)

    before_snapshot = {
        "analyzer": {
            "unit_locations_by_unit": {
                "1": {"x_um": 10.0, "y_um": 20.0},
                "2": {"x_um": 30.0, "y_um": 40.0},
                "3": {"x_um": 50.0, "y_um": 60.0},
            }
        }
    }
    after_snapshot = {"analyzer": {"unit_locations_by_unit": {}}}
    stage_cfg = SimpleNamespace(
        merge_reports_2panel_label_pre_and_post_units=False,
        merge_reports_2panel_before_write_png=True,
        merge_reports_2panel_before_write_svg=False,
        merge_reports_2panel_before_relpath="unit_locations_before_merge.png",
        merge_reports_2panel_after_write_png=False,
        merge_reports_2panel_after_write_svg=False,
        merge_reports_2panel_write_png=False,
        merge_reports_2panel_write_svg=False,
        merge_reports_2panel_highlight_merges_enabled=True,
        merge_reports_2panel_highlight_merges_linked=False,
        merge_reports_2panel_highlight_before_color="#ff7f0e",
        merge_reports_2panel_highlight_label_affected_units=True,
    )

    payload = spikesort_runner._write_merge_unit_location_reports(
        merge_out_dir=tmp_path,
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
        applied_unit_mappings=[{"pre_unit_ids": ["2", "3"], "post_unit_id": None}],
        stage_config=stage_cfg,
    )

    assert payload.get("status") == "ok"
    assert len(fake_plt.axes_created) == 1
    before_texts = [str(call.get("text", "")) for call in fake_plt.axes_created[0].text_calls]
    assert "1" not in before_texts
    assert "2" in before_texts
    assert "3" in before_texts


def test_write_merge_unit_location_reports_infers_after_highlight_when_post_unit_missing(tmp_path: Path, monkeypatch) -> None:
    import sys
    from types import SimpleNamespace as _SimpleNamespace

    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _FakeAxis:
        def __init__(self) -> None:
            self.scatter_calls: list[dict[str, object]] = []

        def set_title(self, title):
            return None

        def set_xlabel(self, label):
            return None

        def set_ylabel(self, label):
            return None

        def set_xlim(self, limits):
            return None

        def set_ylim(self, limits):
            return None

        def invert_yaxis(self):
            return None

        def set_aspect(self, aspect, adjustable=None):
            return None

        def grid(self, enabled, alpha=None):
            return None

        def scatter(self, xs, ys, s=None, alpha=None, c=None):
            self.scatter_calls.append(
                {
                    "xs": list(xs),
                    "ys": list(ys),
                    "colors": list(c),
                }
            )
            return None

        def text(self, x, y, text, ha=None, va=None, transform=None):
            return None

        @property
        def transAxes(self):
            return object()

    class _FakeFigure:
        def savefig(self, path, dpi=None):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("fake", encoding="utf-8")

    class _FakePyplot:
        def __init__(self) -> None:
            self.axes_created: list[_FakeAxis] = []

        def get_cmap(self, name: str):
            def _cmap(_value: float):
                return (0.1, 0.2, 0.9, 1.0)

            return _cmap

        def subplots(self, nrows=1, ncols=1, figsize=None, constrained_layout=False):
            fig = _FakeFigure()
            if int(nrows) == 1 and int(ncols) == 1:
                ax = _FakeAxis()
                self.axes_created.append(ax)
                return fig, ax
            axes = [_FakeAxis() for _ in range(int(nrows) * int(ncols))]
            self.axes_created.extend(axes)
            return fig, axes

        def close(self, fig):
            return None

    fake_plt = _FakePyplot()
    fake_matplotlib = _SimpleNamespace(pyplot=fake_plt)

    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_plt)

    before_snapshot = {
        "analyzer": {
            "unit_locations_by_unit": {
                "190": {"x_um": 2653.33, "y_um": 2079.95},
                "195": {"x_um": 2653.45, "y_um": 2080.30},
            }
        }
    }
    after_snapshot = {
        "analyzer": {
            "unit_locations_by_unit": {
                "205": {"x_um": 2653.69, "y_um": 2079.56},
                "120": {"x_um": 100.0, "y_um": 100.0},
            }
        }
    }
    stage_cfg = SimpleNamespace(
        merge_reports_2panel_before_write_png=False,
        merge_reports_2panel_before_write_svg=False,
        merge_reports_2panel_after_write_png=True,
        merge_reports_2panel_after_write_svg=False,
        merge_reports_2panel_after_relpath="unit_locations_after_merge.png",
        merge_reports_2panel_write_png=False,
        merge_reports_2panel_write_svg=False,
        merge_reports_2panel_highlight_merges_enabled=True,
        merge_reports_2panel_highlight_merges_linked=True,
        merge_reports_2panel_highlight_palette="tab20",
        merge_reports_2panel_after_point_color="#7a7a7a",
    )

    payload = spikesort_runner._write_merge_unit_location_reports(
        merge_out_dir=tmp_path,
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
        applied_unit_mappings=[{"pre_unit_ids": ["190", "195"], "post_unit_id": None}],
        stage_config=stage_cfg,
    )

    assert payload.get("status") == "ok"
    assert payload.get("after_highlighted_units_count") == 1
    assert payload.get("after_highlighted_inferred_units_count") == 1
    assert len(fake_plt.axes_created) == 1
    scatter_colors = list(fake_plt.axes_created[0].scatter_calls[0].get("colors", []))
    assert any(color != "#7a7a7a" for color in scatter_colors)


def test_write_merge_unit_location_reports_links_reused_post_ids_blocks_gray_and_applies_legend_knobs(
    tmp_path: Path, monkeypatch
) -> None:
    import sys
    from types import SimpleNamespace as _SimpleNamespace

    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _FakeLine2D:
        def __init__(self, *args, **kwargs) -> None:
            self.label = str(kwargs.get("label", ""))

    class _FakeColors:
        @staticmethod
        def to_rgba(value):
            if isinstance(value, tuple) and len(value) >= 3:
                if len(value) >= 4:
                    return value
                return (value[0], value[1], value[2], 1.0)
            raise ValueError("unsupported color")

    class _FakeAxis:
        def __init__(self) -> None:
            self.scatter_calls: list[dict[str, object]] = []
            self.legend_calls: list[dict[str, object]] = []

        def set_title(self, title):
            return None

        def set_xlabel(self, label):
            return None

        def set_ylabel(self, label):
            return None

        def set_xlim(self, limits):
            return None

        def set_ylim(self, limits):
            return None

        def invert_yaxis(self):
            return None

        def set_aspect(self, aspect, adjustable=None):
            return None

        def grid(self, enabled, alpha=None):
            return None

        def scatter(self, xs, ys, s=None, alpha=None, c=None):
            self.scatter_calls.append(
                {
                    "xs": list(xs),
                    "ys": list(ys),
                    "colors": list(c),
                }
            )
            return None

        def legend(self, *args, **kwargs):
            labels: list[str] = []
            handles = kwargs.get("handles", None)
            if isinstance(handles, list):
                labels = [str(getattr(handle, "label", "")) for handle in handles]
            elif args:
                try:
                    labels = [str(label) for label in list(args[0])]
                except Exception:
                    labels = []
            self.legend_calls.append({"labels": labels, "kwargs": dict(kwargs)})
            return None

        def text(self, x, y, text, ha=None, va=None, transform=None):
            return None

        @property
        def transAxes(self):
            return object()

    class _FakeFigure:
        def savefig(self, path, dpi=None):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("fake", encoding="utf-8")

    class _FakePyplot:
        def __init__(self) -> None:
            self.axes_created: list[_FakeAxis] = []

        def get_cmap(self, name: str):
            def _cmap(_value: float):
                return (0.45, 0.45, 0.45, 1.0)

            return _cmap

        def subplots(self, nrows=1, ncols=1, figsize=None, constrained_layout=False):
            fig = _FakeFigure()
            if int(nrows) == 1 and int(ncols) == 1:
                ax = _FakeAxis()
                self.axes_created.append(ax)
                return fig, ax
            axes = [_FakeAxis() for _ in range(int(nrows) * int(ncols))]
            self.axes_created.extend(axes)
            return fig, axes

        def close(self, fig):
            return None

    fake_plt = _FakePyplot()
    fake_matplotlib = _SimpleNamespace(pyplot=fake_plt, colors=_FakeColors)

    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_plt)
    monkeypatch.setitem(sys.modules, "matplotlib.colors", _FakeColors)
    monkeypatch.setitem(sys.modules, "matplotlib.lines", _SimpleNamespace(Line2D=_FakeLine2D))

    before_snapshot = {
        "analyzer": {
            "unit_locations_by_unit": {
                "2": {"x_um": 10.0, "y_um": 10.0},
                "7": {"x_um": 20.0, "y_um": 20.0},
                "10": {"x_um": 30.0, "y_um": 30.0},
            }
        }
    }
    after_snapshot = {
        "analyzer": {
            "unit_locations_by_unit": {
                "2": {"x_um": 10.5, "y_um": 10.5},
                "7": {"x_um": 20.5, "y_um": 20.5},
                "99": {"x_um": 90.0, "y_um": 90.0},
            }
        }
    }
    stage_cfg = SimpleNamespace(
        merge_reports_2panel_before_write_png=False,
        merge_reports_2panel_before_write_svg=False,
        merge_reports_2panel_after_write_png=False,
        merge_reports_2panel_after_write_svg=False,
        merge_reports_2panel_write_png=True,
        merge_reports_2panel_write_svg=False,
        merge_reports_2panel_relpath="unit_locations_before_after_merge.png",
        merge_reports_2panel_before_point_color="#7a7a7a",
        merge_reports_2panel_after_point_color="#7a7a7a",
        merge_reports_2panel_highlight_merges_enabled=True,
        merge_reports_2panel_highlight_merges_linked=True,
        merge_reports_2panel_highlight_show_legend=True,
        merge_reports_2panel_highlight_palette="tab20",
        merge_reports_2panel_highlight_legend_position="center left",
        merge_reports_2panel_highlight_legend_x=-0.25,
        merge_reports_2panel_highlight_legend_y=0.4,
        merge_reports_2panel_highlight_sort_pre_legend_by_groups=True,
    )

    payload = spikesort_runner._write_merge_unit_location_reports(
        merge_out_dir=tmp_path,
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
        applied_unit_mappings=[
            {"pre_unit_ids": ["10", "2"], "post_unit_id": "2"},
            {"pre_unit_ids": ["7"], "post_unit_id": "7"},
        ],
        stage_config=stage_cfg,
    )

    assert payload.get("status") == "ok"
    assert len(fake_plt.axes_created) == 2

    before_colors = list(fake_plt.axes_created[0].scatter_calls[0].get("colors", []))
    after_colors = list(fake_plt.axes_created[1].scatter_calls[0].get("colors", []))

    # before ordered_uids are [2, 7, 10]; after ordered_uids are [2, 7, 99]
    assert before_colors[0] == after_colors[0]
    assert before_colors[0] != "#7a7a7a"
    assert before_colors[0] == "#e41a1c"

    before_legend = fake_plt.axes_created[0].legend_calls[0]
    assert before_legend.get("labels") == ["10", "2", "7"]
    legend_kwargs = dict(before_legend.get("kwargs", {}))
    assert legend_kwargs.get("loc") == "center left"
    assert legend_kwargs.get("bbox_to_anchor") == (-0.25, 0.4)


def test_write_merge_unit_location_reports_writes_highlight_linkage_debug_json(tmp_path: Path, monkeypatch) -> None:
    import sys
    from types import SimpleNamespace as _SimpleNamespace

    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _FakeAxis:
        def set_title(self, title):
            return None

        def set_xlabel(self, label):
            return None

        def set_ylabel(self, label):
            return None

        def set_xlim(self, limits):
            return None

        def set_ylim(self, limits):
            return None

        def invert_yaxis(self):
            return None

        def set_aspect(self, aspect, adjustable=None):
            return None

        def grid(self, enabled, alpha=None):
            return None

        def scatter(self, xs, ys, s=None, alpha=None, c=None):
            return None

        def text(self, x, y, text, ha=None, va=None, transform=None):
            return None

        @property
        def transAxes(self):
            return object()

    class _FakeFigure:
        def savefig(self, path, dpi=None):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("fake", encoding="utf-8")

    class _FakePyplot:
        def subplots(self, nrows=1, ncols=1, figsize=None, constrained_layout=False):
            fig = _FakeFigure()
            if int(nrows) == 1 and int(ncols) == 1:
                return fig, _FakeAxis()
            return fig, [_FakeAxis() for _ in range(int(nrows) * int(ncols))]

        def get_cmap(self, name: str):
            def _cmap(_value: float):
                return (0.2, 0.6, 0.2, 1.0)

            return _cmap

        def close(self, fig):
            return None

    class _FakeColors:
        @staticmethod
        def to_rgba(value):
            if isinstance(value, tuple) and len(value) >= 4:
                return value
            if isinstance(value, tuple) and len(value) == 3:
                return (value[0], value[1], value[2], 1.0)
            if isinstance(value, str) and value.startswith("#") and len(value) in {7, 9}:
                return (0.1, 0.2, 0.3, 1.0)
            raise ValueError("unsupported")

        @staticmethod
        def to_hex(value, keep_alpha=False):
            return "#123456ff" if keep_alpha else "#123456"

    fake_plt = _FakePyplot()
    fake_matplotlib = _SimpleNamespace(pyplot=fake_plt, colors=_FakeColors)

    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_plt)
    monkeypatch.setitem(sys.modules, "matplotlib.colors", _FakeColors)

    before_snapshot = {
        "analyzer": {
            "unit_locations_by_unit": {
                "10": {"x_um": 10.0, "y_um": 20.0},
                "20": {"x_um": 20.0, "y_um": 30.0},
            }
        }
    }
    after_snapshot = {
        "analyzer": {
            "unit_locations_by_unit": {
                "20": {"x_um": 20.0, "y_um": 30.0},
            }
        }
    }
    stage_cfg = SimpleNamespace(
        merge_reports_2panel_before_write_png=False,
        merge_reports_2panel_before_write_svg=False,
        merge_reports_2panel_after_write_png=False,
        merge_reports_2panel_after_write_svg=False,
        merge_reports_2panel_write_png=False,
        merge_reports_2panel_write_svg=False,
        merge_reports_2panel_highlight_merges_enabled=True,
        merge_reports_2panel_highlight_merges_linked=False,
        merge_reports_2panel_highlight_before_color="#ff7f0e",
        merge_reports_2panel_highlight_after_color="#2ca02c",
        merge_reports_2panel_highlight_debug_json_enabled=True,
        merge_reports_2panel_highlight_debug_json_relpath="reports/highlight_linkage_debug.json",
    )

    payload = spikesort_runner._write_merge_unit_location_reports(
        merge_out_dir=tmp_path,
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
        applied_unit_mappings=[{"pre_unit_ids": ["10", "20"], "post_unit_id": "20"}],
        stage_config=stage_cfg,
    )

    assert payload.get("status") == "ok"
    debug_json = Path(str(payload.get("outputs", {}).get("merge.report.unit_locations_highlight_linkage_json")))
    assert debug_json.exists()
    debug_payload = _read_json(debug_json)
    assert debug_payload.get("n_mappings") == 1
    rows = list(debug_payload.get("linkage_rows", []))
    assert len(rows) == 1
    assert rows[0].get("requested_post_unit_id") == "20"
    assert rows[0].get("resolved_post_unit_id") == "20"


def test_write_template_amplitude_heatmap_asset_uses_shrunk_right_side_colorbar(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import numpy as np

    from matplotlib.figure import Figure

    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    colorbar_calls: list[dict[str, object]] = []

    def _fake_colorbar(self, mappable, *args, **kwargs):
        colorbar_calls.append(dict(kwargs))
        return object()

    monkeypatch.setattr(Figure, "colorbar", _fake_colorbar)

    ok, error = spikesort_runner._write_template_amplitude_heatmap_asset(
        template_ch_by_t=np.asarray([[-1.0, 2.0, 0.5], [-0.2, 0.8, -0.4]], dtype=float),
        locations_xy=np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=float),
        out_path=tmp_path / "heatmap.png",
        title="Example heatmap",
        cmap="viridis",
        marker_size=12.0,
        show_colorbar=True,
        relative_color_bar_height=0.5,
    )

    assert ok is True
    assert error is None
    assert (tmp_path / "heatmap.png").exists()
    assert len(colorbar_calls) == 1
    assert colorbar_calls[0].get("ax") is not None
    assert "cax" not in colorbar_calls[0]
    assert colorbar_calls[0].get("shrink") == pytest.approx(0.5)
    assert colorbar_calls[0].get("pad") == pytest.approx(0.02)


def test_write_merge_template_heatmap_reports_outputs_panel_and_debug_json(tmp_path: Path, monkeypatch) -> None:
    import numpy as np

    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    monkeypatch.setattr(spikesort_runner, "_import_spikeinterface_full_module", lambda: object())
    monkeypatch.setattr(
        spikesort_runner,
        "_load_sorting_analyzer_from_snapshot",
        lambda si_module, snapshot, stage_config=None: (object(), None),
    )
    monkeypatch.setattr(
        spikesort_runner,
        "_extract_template_and_locations_for_unit",
        lambda analyzer, unit_id, stage_config=None: (
            np.asarray([[-1.0, 2.0, 0.5], [-0.2, 0.8, -0.4]], dtype=float),
            np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=float),
            None,
        ),
    )

    asset_calls: list[dict[str, object]] = []

    def _fake_write_asset(
        template_ch_by_t,
        locations_xy,
        out_path,
        title,
        cmap,
        marker_size,
        show_colorbar,
        relative_color_bar_height=1.0,
        color_vmin=None,
        color_vmax=None,
        color_scale_mode="linear",
        log_epsilon=1e-3,
        magnitude_mode="ptp",
        x_limits=None,
        y_limits=None,
    ):
        asset_calls.append(
            {
                "title": str(title),
                "marker_size": float(marker_size),
                "relative_color_bar_height": float(relative_color_bar_height),
                "vmin": color_vmin,
                "vmax": color_vmax,
                "color_scale_mode": str(color_scale_mode),
                "log_epsilon": float(log_epsilon),
                "magnitude_mode": str(magnitude_mode),
                "x_limits": x_limits,
                "y_limits": y_limits,
            }
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"fake")
        return True, None

    monkeypatch.setattr(spikesort_runner, "_write_template_amplitude_heatmap_asset", _fake_write_asset)
    monkeypatch.setattr(
        spikesort_runner,
        "_stack_rendered_images_horizontally",
        lambda image_paths: (np.ones((16, 16, 3), dtype=float), None),
    )

    before_snapshot = {"analyzer": {"source_dir": str(tmp_path)}}
    after_snapshot = {"analyzer": {"source_dir": str(tmp_path)}}
    stage_cfg = SimpleNamespace(
        merge_reports_template_heatmaps_relpath="reports/template_heatmaps",
        merge_reports_template_heatmaps_assets_reldir="assets",
        merge_reports_template_heatmaps_write_png=True,
        merge_reports_template_heatmaps_write_svg=False,
        merge_reports_template_heatmaps_write_assets_png=True,
        merge_reports_template_heatmaps_write_assets_svg=False,
        merge_reports_template_heatmaps_panel_width_in=8.0,
        merge_reports_template_heatmaps_panel_height_in=4.0,
        merge_reports_template_heatmaps_marker_size=12.0,
        merge_reports_template_heatmaps_cmap="viridis",
        merge_reports_template_heatmaps_show_colorbar=False,
        merge_reports_template_heatmaps_relative_color_bar_height=0.5,
        merge_reports_template_heatmaps_color_scale="log",
        merge_reports_template_heatmaps_log_epsilon=0.01,
        merge_reports_template_heatmaps_magnitude_mode="abs_peak",
        merge_reports_template_heatmaps_max_merges=None,
        merge_reports_template_heatmaps_debug_json_relpath="reports/template_heatmap_debug.json",
        merge_reports_template_heatmaps_inherit_probe_dimensions=True,
        merge_reports_template_heatmaps_probe_dim_x_um=3850.0,
        merge_reports_template_heatmaps_probe_dim_y_um=2100.0,
        merge_reports_template_heatmaps_probe_pitch_um=17.5,
        merge_reports_template_heatmaps_probe_electrode_size_um_x=12.0,
        merge_reports_template_heatmaps_probe_electrode_size_um_y=8.8,
    )
    expected_marker_size, expected_marker_size_source = spikesort_runner._probe_relative_marker_size_points2(
        probe_dim_x_um=3850.0,
        probe_dim_y_um=2100.0,
        probe_pitch_um=17.5,
        electrode_size_um_x=12.0,
        electrode_size_um_y=8.8,
    )
    assert expected_marker_size is not None
    assert expected_marker_size_source == "probe_geometry_electrode_size"

    payload = spikesort_runner._write_merge_template_heatmap_reports(
        merge_out_dir=tmp_path,
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
        applied_unit_mappings=[{"group_id": "g1", "pre_unit_ids": ["10", "11"], "post_unit_id": "20"}],
        stage_config=stage_cfg,
    )

    assert payload.get("status") == "ok"
    assert payload.get("n_mappings_processed") == 1
    debug_json = Path(str(payload.get("debug_json")))
    assert debug_json.exists()
    debug_payload = _read_json(debug_json)
    assert debug_payload.get("n_mappings_processed") == 1
    assert len(list(debug_payload.get("rows", []))) == 1
    rows = list(debug_payload.get("rows", []))
    assert debug_payload.get("panel_layout") == "single_row"
    assert rows[0].get("pre_color_scale", {}).get("mode") == "dynamic_per_merge_group"
    assert rows[0].get("pre_color_scale", {}).get("scale") == "log"
    assert rows[0].get("pre_color_scale", {}).get("magnitude_mode") == "abs_peak"
    assert debug_payload.get("magnitude_mode") == "abs_peak"
    assert debug_payload.get("inherit_probe_dimensions") is True
    assert debug_payload.get("probe_dim_x_um") == pytest.approx(3850.0)
    assert debug_payload.get("probe_dim_y_um") == pytest.approx(2100.0)
    assert debug_payload.get("probe_pitch_um") == pytest.approx(17.5)
    assert debug_payload.get("probe_electrode_size_um_x") == pytest.approx(12.0)
    assert debug_payload.get("probe_electrode_size_um_y") == pytest.approx(8.8)
    assert debug_payload.get("marker_size_requested") == pytest.approx(12.0)
    assert debug_payload.get("marker_size_effective") == pytest.approx(expected_marker_size)
    assert debug_payload.get("marker_size_source") == expected_marker_size_source
    assert debug_payload.get("relative_color_bar_height") == pytest.approx(0.5)
    output_values = list(payload.get("outputs", {}).values())
    assert any(str(path).endswith(".png") for path in output_values)

    pre_calls = [c for c in asset_calls if str(c.get("title", "")).startswith("Pre unit")]
    post_calls = [c for c in asset_calls if str(c.get("title", "")).startswith("Post unit")]
    assert len(pre_calls) == 2
    assert len(post_calls) == 1

    pre_vmins = {c.get("vmin") for c in pre_calls}
    pre_vmaxs = {c.get("vmax") for c in pre_calls}
    assert len(pre_vmins) == 1
    assert len(pre_vmaxs) == 1
    assert next(iter(pre_vmins)) == pytest.approx(0.8)
    assert next(iter(pre_vmaxs)) == pytest.approx(2.0)
    assert {c.get("color_scale_mode") for c in pre_calls} == {"log"}
    assert {c.get("color_scale_mode") for c in post_calls} == {"log"}
    assert {c.get("log_epsilon") for c in pre_calls} == {0.01}
    assert {c.get("log_epsilon") for c in post_calls} == {0.01}
    assert {c.get("magnitude_mode") for c in pre_calls} == {"abs_peak"}
    assert {c.get("magnitude_mode") for c in post_calls} == {"abs_peak"}
    assert {c.get("relative_color_bar_height") for c in pre_calls} == {0.5}
    assert {c.get("relative_color_bar_height") for c in post_calls} == {0.5}
    for call in pre_calls:
        assert float(call.get("marker_size")) == pytest.approx(expected_marker_size)
    for call in post_calls:
        assert float(call.get("marker_size")) == pytest.approx(expected_marker_size)
    assert {c.get("x_limits") for c in pre_calls} == {(0.0, 3850.0)}
    assert {c.get("y_limits") for c in pre_calls} == {(0.0, 2100.0)}
    assert {c.get("x_limits") for c in post_calls} == {(0.0, 3850.0)}
    assert {c.get("y_limits") for c in post_calls} == {(0.0, 2100.0)}
    assert post_calls[0].get("vmin") is None
    assert post_calls[0].get("vmax") is None


def test_write_merge_template_heatmap_reports_uses_provided_analyzers_without_snapshot_reload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import numpy as np

    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    monkeypatch.setattr(
        spikesort_runner,
        "_load_sorting_analyzer_from_snapshot",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("snapshot analyzer load should not be used")),
    )
    monkeypatch.setattr(
        spikesort_runner,
        "_extract_template_and_locations_for_unit",
        lambda analyzer, unit_id, stage_config=None: (
            np.asarray([[-1.0, 2.0, 0.5], [-0.2, 0.8, -0.4]], dtype=float),
            np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=float),
            None,
        ),
    )
    monkeypatch.setattr(
        spikesort_runner,
        "_stack_rendered_images_horizontally",
        lambda image_paths: (np.ones((16, 16, 3), dtype=float), None),
    )

    class _FakeAnalyzer:
        def __init__(self) -> None:
            self.extensions = {}

        def get_loaded_extension_names(self):
            return []

        def has_extension(self, name):
            return True

        def get_extension(self, name):
            return object()

        @property
        def recording(self):
            return SimpleNamespace(get_channel_locations=lambda: np.asarray([[10.0, 20.0], [30.0, 40.0]]))

        def get_unit_ids(self):
            return [10, 20]

    asset_calls: list[dict[str, object]] = []

    def _fake_write_asset(
        template_ch_by_t,
        locations_xy,
        out_path,
        title,
        cmap,
        marker_size,
        show_colorbar,
        relative_color_bar_height=1.0,
        color_vmin=None,
        color_vmax=None,
        color_scale_mode="linear",
        log_epsilon=1e-3,
        magnitude_mode="ptp",
        x_limits=None,
        y_limits=None,
    ):
        asset_calls.append({"title": str(title)})
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"fake")
        return True, None

    monkeypatch.setattr(spikesort_runner, "_write_template_amplitude_heatmap_asset", _fake_write_asset)

    payload = spikesort_runner._write_merge_template_heatmap_reports(
        merge_out_dir=tmp_path,
        before_snapshot={"analyzer": {"source_dir": str(tmp_path / "before")}},
        after_snapshot={"analyzer": {"source_dir": str(tmp_path / "after")}},
        applied_unit_mappings=[{"group_id": "g1", "pre_unit_ids": ["10"], "post_unit_id": "20"}],
        stage_config=SimpleNamespace(
            merge_reports_template_heatmaps_relpath="reports/template_heatmaps",
            merge_reports_template_heatmaps_assets_reldir="assets",
            merge_reports_template_heatmaps_write_png=True,
            merge_reports_template_heatmaps_write_svg=False,
            merge_reports_template_heatmaps_write_assets_png=True,
            merge_reports_template_heatmaps_write_assets_svg=False,
            merge_reports_template_heatmaps_panel_width_in=8.0,
            merge_reports_template_heatmaps_panel_height_in=4.0,
            merge_reports_template_heatmaps_marker_size=12.0,
            merge_reports_template_heatmaps_cmap="viridis",
            merge_reports_template_heatmaps_show_colorbar=False,
            merge_reports_template_heatmaps_relative_color_bar_height=0.5,
            merge_reports_template_heatmaps_color_scale="linear",
            merge_reports_template_heatmaps_log_epsilon=0.01,
            merge_reports_template_heatmaps_magnitude_mode="abs_peak",
            merge_reports_template_heatmaps_max_merges=None,
            merge_reports_template_heatmaps_debug_json_relpath="reports/template_heatmap_debug.json",
            merge_reports_template_heatmaps_inherit_probe_dimensions=False,
        ),
        before_analyzer=_FakeAnalyzer(),
        after_analyzer=_FakeAnalyzer(),
    )

    assert payload.get("status") == "ok"
    assert len(asset_calls) == 2


def test_write_merge_unit_location_reports_does_not_highlight_premerge_ids_on_after_panel(
    tmp_path: Path, monkeypatch
) -> None:
    import sys
    from types import SimpleNamespace as _SimpleNamespace

    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _FakeAxis:
        def __init__(self) -> None:
            self.scatter_calls: list[dict[str, object]] = []

        def set_title(self, title):
            return None

        def set_xlabel(self, label):
            return None

        def set_ylabel(self, label):
            return None

        def set_xlim(self, limits):
            return None

        def set_ylim(self, limits):
            return None

        def invert_yaxis(self):
            return None

        def set_aspect(self, aspect, adjustable=None):
            return None

        def grid(self, enabled, alpha=None):
            return None

        def scatter(self, xs, ys, s=None, alpha=None, c=None):
            self.scatter_calls.append(
                {
                    "xs": list(xs),
                    "ys": list(ys),
                    "colors": list(c),
                }
            )
            return None

        def text(self, x, y, text, ha=None, va=None, transform=None):
            return None

        @property
        def transAxes(self):
            return object()

    class _FakeFigure:
        def savefig(self, path, dpi=None):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("fake", encoding="utf-8")

    class _FakePyplot:
        def __init__(self) -> None:
            self.axes_created: list[_FakeAxis] = []

        def subplots(self, nrows=1, ncols=1, figsize=None, constrained_layout=False):
            fig = _FakeFigure()
            if int(nrows) == 1 and int(ncols) == 1:
                ax = _FakeAxis()
                self.axes_created.append(ax)
                return fig, ax
            axes = [_FakeAxis() for _ in range(int(nrows) * int(ncols))]
            self.axes_created.extend(axes)
            return fig, axes

        def close(self, fig):
            return None

    fake_plt = _FakePyplot()
    fake_matplotlib = _SimpleNamespace(pyplot=fake_plt)

    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_plt)

    before_snapshot = {
        "analyzer": {
            "unit_locations_by_unit": {
                "10": {"x_um": 10.0, "y_um": 10.0},
                "20": {"x_um": 20.0, "y_um": 20.0},
            }
        }
    }
    after_snapshot = {
        "analyzer": {
            "unit_locations_by_unit": {
                "20": {"x_um": 20.0, "y_um": 20.0},
                "31": {"x_um": 31.0, "y_um": 31.0},
            }
        }
    }
    stage_cfg = SimpleNamespace(
        merge_reports_2panel_before_write_png=False,
        merge_reports_2panel_before_write_svg=False,
        merge_reports_2panel_after_write_png=True,
        merge_reports_2panel_after_write_svg=False,
        merge_reports_2panel_after_relpath="unit_locations_after_merge.png",
        merge_reports_2panel_write_png=False,
        merge_reports_2panel_write_svg=False,
        merge_reports_2panel_highlight_merges_enabled=True,
        merge_reports_2panel_highlight_merges_linked=False,
        merge_reports_2panel_highlight_after_color="#2ca02c",
        merge_reports_2panel_after_point_color="#7a7a7a",
    )

    payload = spikesort_runner._write_merge_unit_location_reports(
        merge_out_dir=tmp_path,
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
        applied_unit_mappings=[{"pre_unit_ids": ["10", "20"], "post_unit_id": "20"}],
        stage_config=stage_cfg,
    )

    assert payload.get("status") == "ok"
    assert payload.get("after_highlighted_units_count") == 1
    assert len(fake_plt.axes_created) == 1
    scatter_colors = list(fake_plt.axes_created[0].scatter_calls[0].get("colors", []))
    # ordered_uids are [20, 31]; the resolved post unit 20 should be highlighted.
    assert scatter_colors == ["#2ca02c", "#7a7a7a"]


def test_write_merge_unit_location_reports_zoom_to_affected_units_uses_affected_extent(
    tmp_path: Path, monkeypatch
) -> None:
    import sys
    from types import SimpleNamespace as _SimpleNamespace

    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _FakeAxis:
        def __init__(self) -> None:
            self.x_limits: list[tuple[float, float]] = []
            self.y_limits: list[tuple[float, float]] = []

        def set_title(self, title):
            return None

        def set_xlabel(self, label):
            return None

        def set_ylabel(self, label):
            return None

        def set_xlim(self, limits):
            self.x_limits.append((float(limits[0]), float(limits[1])))
            return None

        def set_ylim(self, limits):
            self.y_limits.append((float(limits[0]), float(limits[1])))
            return None

        def invert_yaxis(self):
            return None

        def set_aspect(self, aspect, adjustable=None):
            return None

        def grid(self, enabled, alpha=None):
            return None

        def scatter(self, xs, ys, s=None, alpha=None, c=None):
            return None

        def text(self, x, y, text, ha=None, va=None, transform=None):
            return None

        @property
        def transAxes(self):
            return object()

    class _FakeFigure:
        def savefig(self, path, dpi=None):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("fake", encoding="utf-8")

    class _FakePyplot:
        def __init__(self) -> None:
            self.axes_created: list[_FakeAxis] = []

        def subplots(self, nrows=1, ncols=1, figsize=None, constrained_layout=False):
            fig = _FakeFigure()
            if int(nrows) == 1 and int(ncols) == 1:
                ax = _FakeAxis()
                self.axes_created.append(ax)
                return fig, ax
            axes = [_FakeAxis() for _ in range(int(nrows) * int(ncols))]
            self.axes_created.extend(axes)
            return fig, axes

        def close(self, fig):
            return None

    fake_plt = _FakePyplot()
    fake_matplotlib = _SimpleNamespace(pyplot=fake_plt)

    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_plt)

    before_snapshot = {
        "analyzer": {
            "unit_locations_by_unit": {
                "10": {"x_um": 10.0, "y_um": 10.0},
                "20": {"x_um": 20.0, "y_um": 20.0},
                "99": {"x_um": 1000.0, "y_um": 1000.0},
            }
        }
    }
    after_snapshot = {
        "analyzer": {
            "unit_locations_by_unit": {
                "101": {"x_um": 110.0, "y_um": 130.0},
                "150": {"x_um": 1500.0, "y_um": 1500.0},
            }
        }
    }
    stage_cfg = SimpleNamespace(
        merge_reports_2panel_before_write_png=False,
        merge_reports_2panel_before_write_svg=False,
        merge_reports_2panel_after_write_png=False,
        merge_reports_2panel_after_write_svg=False,
        merge_reports_2panel_write_png=True,
        merge_reports_2panel_write_svg=False,
        merge_reports_2panel_relpath="unit_locations_before_after_merge.png",
        merge_reports_2panel_highlight_merges_enabled=True,
        merge_reports_2panel_highlight_merges_linked=False,
        merge_reports_2panel_highlight_after_color="#2ca02c",
        merge_reports_2panel_zoom_to_affected_units=True,
    )

    payload = spikesort_runner._write_merge_unit_location_reports(
        merge_out_dir=tmp_path,
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
        applied_unit_mappings=[{"pre_unit_ids": ["10", "20"], "post_unit_id": "101"}],
        stage_config=stage_cfg,
    )

    assert payload.get("status") == "ok"
    assert payload.get("zoom_to_affected_units") is True
    assert payload.get("zoom_to_affected_units_applied") is True
    assert len(fake_plt.axes_created) == 2
    for ax in fake_plt.axes_created:
        assert ax.x_limits
        assert ax.y_limits
        x_min, x_max = ax.x_limits[0]
        y_min, y_max = ax.y_limits[0]
        assert x_min > 0.0
        assert x_max < 500.0
        assert y_min > 0.0
        assert y_max < 500.0


def test_build_applied_unit_mappings_uses_post_unit_hint_even_if_missing_from_post_snapshot() -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    mappings = spikesort_runner._build_applied_unit_mappings(
        applied_operations=[
            {
                "method": "slay",
                "group_id": "202",
                "pre_unit_ids": ["190", "195"],
                "post_unit_id_hint": "202",
            }
        ],
        pre_analyzer_payload={
            "unit_ids": ["190", "195"],
            "unit_locations_by_unit": {
                "190": {"x_um": 2653.33, "y_um": 2079.95},
                "195": {"x_um": 2653.45, "y_um": 2080.30},
            },
        },
        post_analyzer_payload={
            "unit_ids": ["105", "120"],
            "unit_locations_by_unit": {
                "105": {"x_um": 2653.69, "y_um": 2079.56},
                "120": {"x_um": 100.0, "y_um": 100.0},
            },
        },
    )

    assert len(mappings) == 1
    assert mappings[0].get("post_unit_id") == "202"
    assert mappings[0].get("resolution") == "post_unit_hint_missing_in_post_snapshot"
    assert mappings[0].get("post_unit_id_in_post_snapshot") is False


def test_build_merge_metadata_summary_uses_before_after_set_delta_for_added_ids() -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    stage_cfg = SimpleNamespace(
        slay_enabled=True,
        merge_slay_dry_run=False,
        slay_auto_accept_merges=True,
        auto_merge_enabled=False,
        auto_merge_auto_accept_merges=False,
    )

    summary = spikesort_runner._build_merge_metadata_summary(
        requested_sequence_raw=["SLAy"],
        stage_config=stage_cfg,
        pre_snapshot={
            "sorter": {"available": True, "unit_ids": ["0", "22", "105"]},
            "analyzer": {"available": True, "unit_ids": ["0", "22", "105"]},
        },
        post_snapshot={
            "sorter": {"available": True, "unit_ids": ["105"]},
            "analyzer": {"available": True, "unit_ids": ["105"]},
        },
        applied_operations=[
            {
                "method": "slay",
                "group_id": "202",
                "pre_unit_ids": ["0", "22"],
                "post_unit_id_hint": "105",
            }
        ],
    )

    analyzer_delta = summary.get("delta", {}).get("analyzer", {})
    sorter_delta = summary.get("delta", {}).get("sorter", {})

    assert analyzer_delta.get("added_unit_ids_set_delta") == []
    assert analyzer_delta.get("added_unit_ids") == []
    assert analyzer_delta.get("added_unit_ids_source") == "set_delta"
    assert set(analyzer_delta.get("removed_unit_ids", [])) == {"0", "22"}
    assert analyzer_delta.get("merge_target_unit_ids_from_mappings") == ["105"]
    assert analyzer_delta.get("merge_tracking_validation", {}).get("added_ids_cover_mapping_targets") is False
    assert analyzer_delta.get("merge_tracking_validation", {}).get("targets_present_in_post_snapshot") is True

    assert sorter_delta.get("added_unit_ids_set_delta") == []
    assert sorter_delta.get("added_unit_ids") == []
    assert sorter_delta.get("added_unit_ids_source") == "set_delta"


def test_build_merge_metadata_summary_keeps_mapping_target_gaps_as_diagnostics_only() -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    stage_cfg = SimpleNamespace(
        slay_enabled=True,
        merge_slay_dry_run=False,
        slay_auto_accept_merges=True,
        auto_merge_enabled=False,
        auto_merge_auto_accept_merges=False,
    )

    summary = spikesort_runner._build_merge_metadata_summary(
        requested_sequence_raw=["SLAy"],
        stage_config=stage_cfg,
        pre_snapshot={
            "sorter": {"available": True, "unit_ids": ["0", "22", "190", "195"]},
            "analyzer": {"available": True, "unit_ids": ["0", "22", "190", "195"]},
        },
        post_snapshot={
            "sorter": {"available": True, "unit_ids": ["1", "2"]},
            "analyzer": {"available": True, "unit_ids": ["1", "2"]},
        },
        applied_operations=[
            {
                "method": "slay",
                "group_id": "202",
                "pre_unit_ids": ["190", "195"],
                "post_unit_id_hint": "202",
            },
            {
                "method": "slay",
                "group_id": "203",
                "pre_unit_ids": ["0", "22"],
                "post_unit_id_hint": "203",
            },
        ],
    )

    change_validation = summary.get("change_validation", {})
    analyzer_delta = summary.get("delta", {}).get("analyzer", {})

    assert change_validation.get("passes") is True
    assert change_validation.get("reason") == "ok"
    assert change_validation.get("merge_target_unit_ids") == ["202", "203"]
    assert set(change_validation.get("merge_target_ids_missing_from_analyzer_post", [])) == {"202", "203"}
    assert analyzer_delta.get("added_unit_ids") == ["1", "2"]


def test_compute_snapshot_unit_delta_uses_snapshot_unit_count_when_ids_missing() -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    delta = spikesort_runner._compute_snapshot_unit_delta(
        before_payload={"available": True, "unit_count": 7, "unit_ids": []},
        after_payload={"available": True, "unit_count": 5, "unit_ids": []},
    )

    assert delta.get("compared") is True
    assert delta.get("changed") is True
    assert delta.get("before_unit_count") == 7
    assert delta.get("after_unit_count") == 5
    assert delta.get("before_unit_ids") == []
    assert delta.get("after_unit_ids") == []


def test_build_unit_diff_map_and_flat_map_supports_chained_merges() -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    unit_diff_payload = {
        "before": {
            "analyzer": {
                "unit_ids": ["10", "11", "12", "13"],
                "unit_locations_by_unit": {
                    "10": {"x_um": 10.0, "y_um": 10.0},
                    "11": {"x_um": 11.0, "y_um": 11.0},
                    "12": {"x_um": 12.0, "y_um": 12.0},
                    "13": {"x_um": 13.0, "y_um": 13.0},
                },
            }
        },
        "after": {
            "analyzer": {
                "unit_ids": ["25"],
                "unit_locations_by_unit": {
                    "25": {"x_um": 25.0, "y_um": 25.0},
                },
            }
        },
        "applied_merge_operations": [
            {
                "method": "slay",
                "group_id": "g1",
                "pre_unit_ids": ["10", "11"],
            },
            {
                "method": "auto_merge",
                "group_id": "g2",
                "pre_unit_ids": ["12", "20"],
                "iteration": 1,
            },
            {
                "method": "auto_merge",
                "group_id": "g3",
                "pre_unit_ids": ["21", "13"],
                "iteration": 2,
            },
        ],
        "applied_unit_mappings": [
            {
                "method": "slay",
                "group_id": "g1",
                "pre_unit_ids": ["10", "11"],
                "post_unit_id": "20",
                "resolution": "post_unit_hint",
            },
            {
                "method": "auto_merge",
                "group_id": "g2",
                "pre_unit_ids": ["12", "20"],
                "iteration": 1,
                "post_unit_id": "21",
                "resolution": "post_unit_hint",
            },
            {
                "method": "auto_merge",
                "group_id": "g3",
                "pre_unit_ids": ["21", "13"],
                "iteration": 2,
                "post_unit_id": "25",
                "resolution": "post_unit_hint",
            },
        ],
    }

    op_map = spikesort_runner._build_unit_diff_map_payload(unit_diff_payload=unit_diff_payload)
    flat_map = spikesort_runner._build_unit_diff_map_flat_payload(
        unit_diff_map_payload=op_map,
        unit_diff_payload=unit_diff_payload,
    )

    assert op_map.get("summary", {}).get("n_operations") == 3
    assert flat_map.get("summary", {}).get("n_flat_groups") == 1
    groups = flat_map.get("groups", [])
    assert len(groups) == 1
    assert groups[0].get("final_post_unit_id") == "25"
    assert groups[0].get("primary_pre_unit_ids") == ["10", "11", "12", "13"]


def test_extract_plot_inputs_from_unit_diff_report_prefers_flattened_groups() -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    unit_diff_payload = {
        "before": {"analyzer": {"unit_locations_by_unit": {"10": {"x_um": 10.0, "y_um": 10.0}}}},
        "after": {"analyzer": {"unit_locations_by_unit": {"25": {"x_um": 25.0, "y_um": 25.0}}}},
        "applied_unit_mappings": [
            {"method": "slay", "group_id": "g1", "pre_unit_ids": ["10"], "post_unit_id": "20"}
        ],
        "unit_diff_map_flat": {
            "groups": [
                {
                    "group_id": "flat_g1",
                    "final_post_unit_id": "25",
                    "primary_pre_unit_ids": ["10"],
                }
            ]
        },
    }

    before_snapshot, after_snapshot, mappings = spikesort_runner._extract_plot_inputs_from_unit_diff_report(
        unit_diff_payload=unit_diff_payload
    )

    assert isinstance(before_snapshot, dict)
    assert isinstance(after_snapshot, dict)
    assert len(mappings) == 1
    assert mappings[0].get("post_unit_id") == "25"
    assert mappings[0].get("pre_unit_ids") == ["10"]


def test_build_post_merge_unit_locations_payload_includes_ids_and_locations() -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    payload = spikesort_runner._build_post_merge_unit_locations_payload(
        post_snapshot={
            "analyzer": {
                "unit_ids": ["21", "22"],
                "unit_locations_by_unit": {
                    "21": {"x_um": 21.0, "y_um": 22.0},
                    "22": {"x_um": 22.0, "y_um": 23.0},
                },
            }
        }
    )

    assert payload.get("summary", {}).get("n_unit_ids") == 2
    assert payload.get("summary", {}).get("n_locations") == 2
    assert payload.get("unit_ids") == ["21", "22"]
    assert payload.get("unit_locations_by_unit", {}).get("21", {}).get("x_um") == 21.0


def test_capture_merge_state_snapshot_uses_analyzer_sorting_if_sorter_load_fails(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _FakeSorting:
        def get_unit_ids(self):
            return [11, 12, 13]

        def get_num_units(self):
            return 3

    class _FakeAnalyzer:
        def __init__(self) -> None:
            self.sorting = _FakeSorting()

    well_out_dir = tmp_path / "well001"
    stage_output_root_dir = well_out_dir / "spikesort_outputs"
    stage_output_root_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(spikesort_runner, "_import_spikeinterface_full_module", lambda: object())
    monkeypatch.setattr(
        spikesort_runner,
        "_load_sorting_from_sorter_output_dir",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("cannot load sorter")),
    )
    monkeypatch.setattr(
        spikesort_runner,
        "_load_concat_analyzer_for_phase",
        lambda **kwargs: (
            _FakeAnalyzer(),
            stage_output_root_dir / "concat_analyzer",
        ),
    )

    snapshot = spikesort_runner._capture_merge_state_snapshot(
        well_out_dir=well_out_dir,
        stage_output_root_dir=stage_output_root_dir,
        output_rel_root="spikesort_outputs",
        stage_config=SimpleNamespace(sorter="kilosort4", slay_sorter_output_relpath=None),
        capture_label="after_merge",
        include_unit_locations=False,
        allow_analyzer_recompute=True,
    )

    assert snapshot.get("sorter", {}).get("available") is True
    assert snapshot.get("sorter", {}).get("load_error") is None
    assert snapshot.get("sorter", {}).get("unit_count") == 3
    assert snapshot.get("sorter", {}).get("unit_ids") == ["11", "12", "13"]


def test_install_spikeinterface_random_spikes_percentage_compatibility_preserves_percentage(monkeypatch) -> None:
    import sys
    from types import ModuleType

    import numpy as np

    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    spikeinterface_module = ModuleType("spikeinterface")
    core_module = ModuleType("spikeinterface.core")
    analyzer_extension_core_module = ModuleType("spikeinterface.core.analyzer_extension_core")
    sorting_tools_module = ModuleType("spikeinterface.core.sorting_tools")

    class _FakeComputeRandomSpikes:
        def _set_params(
            self,
            method="uniform",
            max_spikes_per_unit=500,
            margin_size=None,
            seed=None,
        ):
            return {
                "method": method,
                "max_spikes_per_unit": max_spikes_per_unit,
                "margin_size": margin_size,
                "seed": seed,
            }

        def _run(self, verbose=False):
            return None

    analyzer_extension_core_module.ComputeRandomSpikes = _FakeComputeRandomSpikes
    sorting_tools_module.random_spikes_selection = lambda *args, **kwargs: np.array([], dtype=np.int64)
    sorting_tools_module.spike_vector_to_indices = lambda *args, **kwargs: []

    monkeypatch.setitem(sys.modules, "spikeinterface", spikeinterface_module)
    monkeypatch.setitem(sys.modules, "spikeinterface.core", core_module)
    monkeypatch.setitem(sys.modules, "spikeinterface.core.analyzer_extension_core", analyzer_extension_core_module)
    monkeypatch.setitem(sys.modules, "spikeinterface.core.sorting_tools", sorting_tools_module)

    spikesort_runner._install_spikeinterface_random_spikes_percentage_compatibility()

    params = _FakeComputeRandomSpikes()._set_params(
        method="percentage",
        percentage=0.75,
        min_spikes_per_unit=1000,
        log_before_after_spike_counts=True,
        max_spikes_per_unit=5000,
    )
    assert params["method"] == "percentage"
    assert params["max_spikes_per_unit"] == 5000
    assert params["percentage"] == pytest.approx(0.75)
    assert params["min_spikes_per_unit"] == 1000
    assert params["log_before_after_spike_counts"] is True


def test_install_spikeinterface_random_spikes_percentage_compatibility_runs_percentage_selection_without_upstream_support(
    monkeypatch,
    caplog,
) -> None:
    import sys
    from types import ModuleType

    import numpy as np

    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    spikeinterface_module = ModuleType("spikeinterface")
    core_module = ModuleType("spikeinterface.core")
    analyzer_extension_core_module = ModuleType("spikeinterface.core.analyzer_extension_core")
    sorting_tools_module = ModuleType("spikeinterface.core.sorting_tools")

    class _FakeSorting:
        unit_ids = np.array(["u0", "u1"])

        def get_num_segments(self):
            return 1

        def to_spike_vector(self, concatenated=False):
            spikes = np.array(
                [
                    (10, 0, 0),
                    (20, 0, 0),
                    (30, 0, 0),
                    (40, 0, 0),
                    (15, 1, 0),
                    (25, 1, 0),
                ],
                dtype=[("sample_index", "int64"), ("unit_index", "int64"), ("segment_index", "int64")],
            )
            if concatenated:
                return spikes
            return [spikes]

    def _fake_spike_vector_to_indices(spikes, unit_ids, absolute_index=False):
        return [{"u0": np.array([0, 1, 2, 3]), "u1": np.array([4, 5])}]

    def _old_random_spikes_selection(
        sorting,
        num_samples=None,
        method="uniform",
        max_spikes_per_unit=500,
        margin_size=None,
        seed=None,
    ):
        if method == "percentage":
            raise AssertionError("compat path should bypass old random_spikes_selection for percentage mode")
        return np.array([], dtype=np.int64)

    class _FakeComputeRandomSpikes:
        def __init__(self, sorting_analyzer=None):
            self.sorting_analyzer = sorting_analyzer
            self.data = {}
            self.params = {}

        def _set_params(
            self,
            method="uniform",
            max_spikes_per_unit=500,
            margin_size=None,
            seed=None,
        ):
            return {
                "method": method,
                "max_spikes_per_unit": max_spikes_per_unit,
                "margin_size": margin_size,
                "seed": seed,
            }

        def _run(self, verbose=False):
            raise AssertionError("compat path should replace _run for percentage mode")

    analyzer_extension_core_module.ComputeRandomSpikes = _FakeComputeRandomSpikes
    sorting_tools_module.random_spikes_selection = _old_random_spikes_selection
    sorting_tools_module.spike_vector_to_indices = _fake_spike_vector_to_indices

    monkeypatch.setitem(sys.modules, "spikeinterface", spikeinterface_module)
    monkeypatch.setitem(sys.modules, "spikeinterface.core", core_module)
    monkeypatch.setitem(sys.modules, "spikeinterface.core.analyzer_extension_core", analyzer_extension_core_module)
    monkeypatch.setitem(sys.modules, "spikeinterface.core.sorting_tools", sorting_tools_module)

    caplog.set_level("INFO")
    spikesort_runner._install_spikeinterface_random_spikes_percentage_compatibility()

    sorting_analyzer = SimpleNamespace(
        sorting=_FakeSorting(),
        rec_attributes={"num_samples": [100]},
    )
    extension = _FakeComputeRandomSpikes(sorting_analyzer=sorting_analyzer)
    extension.params = extension._set_params(
        method="percentage",
        percentage=0.5,
        min_spikes_per_unit=2,
        log_before_after_spike_counts=True,
        max_spikes_per_unit=1,
        seed=0,
    )

    extension._run()

    selected = extension.data["random_spikes_indices"]
    assert selected.ndim == 1
    assert selected.size == 3
    assert np.all(np.isin(np.array([4, 5], dtype=np.int64), selected))
    assert np.all(np.isin(selected, np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)))
    assert any(
        "Merge analyzer random_spikes counts method=percentage" in record.getMessage()
        for record in caplog.records
    )
    assert any(
        "Merge analyzer random_spikes unit=u0 total_spikes=4 eligible_spikes=4 selected_spikes=1 selection=sampled_percentage_capped_by_max"
        in record.getMessage()
        for record in caplog.records
    )
    assert any(
        "Merge analyzer random_spikes unit=u1 total_spikes=2 eligible_spikes=2 selected_spikes=2 selection=all_by_min_spikes_threshold"
        in record.getMessage()
        for record in caplog.records
    )


def test_install_spikeinterface_random_spikes_percentage_compatibility_applies_min_spikes_floor(
    monkeypatch,
    caplog,
) -> None:
    import sys
    from types import ModuleType

    import numpy as np

    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    spikeinterface_module = ModuleType("spikeinterface")
    core_module = ModuleType("spikeinterface.core")
    analyzer_extension_core_module = ModuleType("spikeinterface.core.analyzer_extension_core")
    sorting_tools_module = ModuleType("spikeinterface.core.sorting_tools")

    class _FakeSorting:
        unit_ids = np.array(["u0", "u1"])

        def get_num_segments(self):
            return 1

        def to_spike_vector(self, concatenated=False):
            spikes = np.array(
                [
                    (10, 0, 0),
                    (20, 0, 0),
                    (30, 0, 0),
                    (40, 0, 0),
                    (15, 1, 0),
                    (25, 1, 0),
                ],
                dtype=[("sample_index", "int64"), ("unit_index", "int64"), ("segment_index", "int64")],
            )
            if concatenated:
                return spikes
            return [spikes]

    def _fake_spike_vector_to_indices(spikes, unit_ids, absolute_index=False):
        return [{"u0": np.array([0, 1, 2, 3]), "u1": np.array([4, 5])}]

    def _old_random_spikes_selection(
        sorting,
        num_samples=None,
        method="uniform",
        max_spikes_per_unit=500,
        margin_size=None,
        seed=None,
    ):
        if method == "percentage":
            raise AssertionError("compat path should bypass old random_spikes_selection for percentage mode")
        return np.array([], dtype=np.int64)

    class _FakeComputeRandomSpikes:
        def __init__(self, sorting_analyzer=None):
            self.sorting_analyzer = sorting_analyzer
            self.data = {}
            self.params = {}

        def _set_params(
            self,
            method="uniform",
            max_spikes_per_unit=500,
            margin_size=None,
            seed=None,
        ):
            return {
                "method": method,
                "max_spikes_per_unit": max_spikes_per_unit,
                "margin_size": margin_size,
                "seed": seed,
            }

        def _run(self, verbose=False):
            raise AssertionError("compat path should replace _run for percentage mode")

    analyzer_extension_core_module.ComputeRandomSpikes = _FakeComputeRandomSpikes
    sorting_tools_module.random_spikes_selection = _old_random_spikes_selection
    sorting_tools_module.spike_vector_to_indices = _fake_spike_vector_to_indices

    monkeypatch.setitem(sys.modules, "spikeinterface", spikeinterface_module)
    monkeypatch.setitem(sys.modules, "spikeinterface.core", core_module)
    monkeypatch.setitem(sys.modules, "spikeinterface.core.analyzer_extension_core", analyzer_extension_core_module)
    monkeypatch.setitem(sys.modules, "spikeinterface.core.sorting_tools", sorting_tools_module)

    caplog.set_level("INFO")
    spikesort_runner._install_spikeinterface_random_spikes_percentage_compatibility()

    sorting_analyzer = SimpleNamespace(
        sorting=_FakeSorting(),
        rec_attributes={"num_samples": [100]},
    )
    extension = _FakeComputeRandomSpikes(sorting_analyzer=sorting_analyzer)
    extension.params = extension._set_params(
        method="percentage",
        percentage=0.25,
        min_spikes_per_unit=2,
        log_before_after_spike_counts=True,
        seed=0,
    )

    assert extension.params["max_spikes_per_unit"] is None

    extension._run()

    selected = extension.data["random_spikes_indices"]
    assert selected.ndim == 1
    assert selected.size == 4
    assert np.count_nonzero(np.isin(selected, np.array([0, 1, 2, 3], dtype=np.int64))) == 2
    assert np.all(np.isin(np.array([4, 5], dtype=np.int64), selected))
    assert any(
        "Merge analyzer random_spikes unit=u0 total_spikes=4 eligible_spikes=4 selected_spikes=2 selection=min_spikes_floor"
        in record.getMessage()
        for record in caplog.records
    )
    assert any(
        "Merge analyzer random_spikes unit=u1 total_spikes=2 eligible_spikes=2 selected_spikes=2 selection=all_by_min_spikes_threshold"
        in record.getMessage()
        for record in caplog.records
    )


def test_install_spikeinterface_random_spikes_percentage_compatibility_does_not_apply_implicit_default_cap(
    monkeypatch,
) -> None:
    import sys
    from types import ModuleType

    import numpy as np

    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    spikeinterface_module = ModuleType("spikeinterface")
    core_module = ModuleType("spikeinterface.core")
    analyzer_extension_core_module = ModuleType("spikeinterface.core.analyzer_extension_core")
    sorting_tools_module = ModuleType("spikeinterface.core.sorting_tools")

    class _FakeSorting:
        unit_ids = np.array(["u0"])

        def get_num_segments(self):
            return 1

        def to_spike_vector(self, concatenated=False):
            spikes = np.array(
                [
                    (10, 0, 0),
                    (20, 0, 0),
                    (30, 0, 0),
                    (40, 0, 0),
                ],
                dtype=[("sample_index", "int64"), ("unit_index", "int64"), ("segment_index", "int64")],
            )
            if concatenated:
                return spikes
            return [spikes]

    def _fake_spike_vector_to_indices(spikes, unit_ids, absolute_index=False):
        return [{"u0": np.array([0, 1, 2, 3])}]

    def _old_random_spikes_selection(
        sorting,
        num_samples=None,
        method="uniform",
        max_spikes_per_unit=500,
        margin_size=None,
        seed=None,
    ):
        if method == "percentage":
            raise AssertionError("compat path should bypass old random_spikes_selection for percentage mode")
        return np.array([], dtype=np.int64)

    class _FakeComputeRandomSpikes:
        def __init__(self, sorting_analyzer=None):
            self.sorting_analyzer = sorting_analyzer
            self.data = {}
            self.params = {}

        def _set_params(
            self,
            method="uniform",
            max_spikes_per_unit=500,
            margin_size=None,
            seed=None,
        ):
            return {
                "method": method,
                "max_spikes_per_unit": max_spikes_per_unit,
                "margin_size": margin_size,
                "seed": seed,
            }

        def _run(self, verbose=False):
            raise AssertionError("compat path should replace _run for percentage mode")

    analyzer_extension_core_module.ComputeRandomSpikes = _FakeComputeRandomSpikes
    sorting_tools_module.random_spikes_selection = _old_random_spikes_selection
    sorting_tools_module.spike_vector_to_indices = _fake_spike_vector_to_indices

    monkeypatch.setitem(sys.modules, "spikeinterface", spikeinterface_module)
    monkeypatch.setitem(sys.modules, "spikeinterface.core", core_module)
    monkeypatch.setitem(sys.modules, "spikeinterface.core.analyzer_extension_core", analyzer_extension_core_module)
    monkeypatch.setitem(sys.modules, "spikeinterface.core.sorting_tools", sorting_tools_module)

    spikesort_runner._install_spikeinterface_random_spikes_percentage_compatibility()

    sorting_analyzer = SimpleNamespace(
        sorting=_FakeSorting(),
        rec_attributes={"num_samples": [100]},
    )
    extension = _FakeComputeRandomSpikes(sorting_analyzer=sorting_analyzer)
    extension.params = extension._set_params(
        method="percentage",
        percentage=0.75,
        seed=0,
    )

    assert extension.params["max_spikes_per_unit"] is None

    extension._run()

    selected = extension.data["random_spikes_indices"]
    assert selected.ndim == 1
    assert selected.size == 3
    assert np.all(np.isin(selected, np.array([0, 1, 2, 3], dtype=np.int64)))


def test_install_spikeinterface_random_spikes_percentage_compatibility_uses_threshold_even_with_upstream_percentage_support(
    monkeypatch,
) -> None:
    import sys
    from types import ModuleType

    import numpy as np

    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    spikeinterface_module = ModuleType("spikeinterface")
    core_module = ModuleType("spikeinterface.core")
    analyzer_extension_core_module = ModuleType("spikeinterface.core.analyzer_extension_core")
    sorting_tools_module = ModuleType("spikeinterface.core.sorting_tools")

    class _FakeSorting:
        unit_ids = np.array(["u0", "u1"])

        def get_num_segments(self):
            return 1

        def to_spike_vector(self, concatenated=False):
            spikes = np.array(
                [
                    (10, 0, 0),
                    (20, 0, 0),
                    (30, 0, 0),
                    (40, 0, 0),
                    (15, 1, 0),
                    (25, 1, 0),
                ],
                dtype=[("sample_index", "int64"), ("unit_index", "int64"), ("segment_index", "int64")],
            )
            if concatenated:
                return spikes
            return [spikes]

    def _fake_spike_vector_to_indices(spikes, unit_ids, absolute_index=False):
        return [{"u0": np.array([0, 1, 2, 3]), "u1": np.array([4, 5])}]

    def _upstream_random_spikes_selection(
        sorting,
        num_samples=None,
        method="uniform",
        max_spikes_per_unit=500,
        margin_size=None,
        seed=None,
        percentage=None,
    ):
        if method == "percentage":
            raise AssertionError("thresholded percentage mode should use the local compat selector")
        return np.array([], dtype=np.int64)

    class _FakeComputeRandomSpikes:
        def __init__(self, sorting_analyzer=None):
            self.sorting_analyzer = sorting_analyzer
            self.data = {}
            self.params = {}

        def _set_params(
            self,
            method="uniform",
            max_spikes_per_unit=500,
            margin_size=None,
            seed=None,
            percentage=None,
        ):
            return {
                "method": method,
                "max_spikes_per_unit": max_spikes_per_unit,
                "margin_size": margin_size,
                "seed": seed,
                "percentage": percentage,
            }

        def _run(self, verbose=False):
            raise AssertionError("compat path should replace _run when min_spikes_per_unit is set")

    analyzer_extension_core_module.ComputeRandomSpikes = _FakeComputeRandomSpikes
    sorting_tools_module.random_spikes_selection = _upstream_random_spikes_selection
    sorting_tools_module.spike_vector_to_indices = _fake_spike_vector_to_indices

    monkeypatch.setitem(sys.modules, "spikeinterface", spikeinterface_module)
    monkeypatch.setitem(sys.modules, "spikeinterface.core", core_module)
    monkeypatch.setitem(sys.modules, "spikeinterface.core.analyzer_extension_core", analyzer_extension_core_module)
    monkeypatch.setitem(sys.modules, "spikeinterface.core.sorting_tools", sorting_tools_module)

    spikesort_runner._install_spikeinterface_random_spikes_percentage_compatibility()

    sorting_analyzer = SimpleNamespace(
        sorting=_FakeSorting(),
        rec_attributes={"num_samples": [100]},
    )
    extension = _FakeComputeRandomSpikes(sorting_analyzer=sorting_analyzer)
    extension.params = extension._set_params(
        method="percentage",
        percentage=0.5,
        min_spikes_per_unit=2,
        max_spikes_per_unit=1,
        seed=0,
    )

    extension._run()

    selected = extension.data["random_spikes_indices"]
    assert selected.ndim == 1
    assert selected.size == 3
    assert np.all(np.isin(np.array([4, 5], dtype=np.int64), selected))


def test_capture_merge_state_snapshot_uses_provided_analyzer_object(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _FakeSorting:
        def get_unit_ids(self):
            return [31, 32]

        def get_num_units(self):
            return 2

    class _FakeAnalyzer:
        def __init__(self) -> None:
            self.sorting = _FakeSorting()
            self.sparsity = None

    well_out_dir = tmp_path / "well001"
    stage_output_root_dir = well_out_dir / "spikesort_outputs"
    analyzer_output_dir = stage_output_root_dir / "pre_merge_analyzer_output"
    analyzer_output_dir.mkdir(parents=True, exist_ok=True)

    analyzer = _FakeAnalyzer()
    spikesort_runner._attach_merge_analyzer_policy_info(
        analyzer,
        spikesort_runner._describe_merge_analyzer_policy_info(
            analyzer=analyzer,
            stage_config=SimpleNamespace(
                merge_analyzer_compute_sparsity=False,
                merge_template_random_spikes_method="all",
            ),
            reused_cached_analyzer=True,
        ),
    )

    monkeypatch.setattr(spikesort_runner, "_import_spikeinterface_full_module", lambda: object())
    monkeypatch.setattr(
        spikesort_runner,
        "_load_sorting_from_sorter_output_dir",
        lambda **kwargs: _FakeSorting(),
    )
    monkeypatch.setattr(
        spikesort_runner,
        "_load_concat_analyzer_for_phase",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not reload analyzer")),
    )

    snapshot = spikesort_runner._capture_merge_state_snapshot(
        well_out_dir=well_out_dir,
        stage_output_root_dir=stage_output_root_dir,
        output_rel_root="spikesort_outputs",
        stage_config=SimpleNamespace(
            sorter="kilosort4",
            merge_analyzer_compute_sparsity=False,
            merge_template_random_spikes_method="all",
        ),
        sorter_output_dir=stage_output_root_dir / "sorter_output",
        analyzer_source_dir=analyzer_output_dir,
        analyzer_obj=analyzer,
        capture_label="before_merge",
        include_unit_locations=False,
        allow_analyzer_recompute=False,
    )

    assert snapshot.get("analyzer", {}).get("available") is True
    assert snapshot.get("analyzer", {}).get("source_dir") == str(analyzer_output_dir.resolve())
    assert snapshot.get("analyzer", {}).get("unit_ids") == ["31", "32"]


def test_resolve_sorter_output_dir_prefers_wrapper_with_spikeinterface_markers(tmp_path: Path) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    well_out_dir = tmp_path / "well001"
    wrapper_dir = well_out_dir / "spikesort_outputs" / "sorter_output"
    nested_dir = wrapper_dir / "sorter_output"
    nested_dir.mkdir(parents=True, exist_ok=True)
    (nested_dir / "params.py").write_text("n_channels_dat=1\n", encoding="utf-8")
    (wrapper_dir / "spikeinterface_params.json").write_text("{}", encoding="utf-8")

    resolved = spikesort_runner._resolve_sorter_output_dir(
        well_out_dir=well_out_dir,
        output_rel_root="spikesort_outputs",
        stage_config=SimpleNamespace(slay_sorter_output_relpath=None),
    )

    assert resolved == wrapper_dir.resolve()


def test_resolve_sorter_output_dir_prefers_stage_root_when_merge_root_also_exists(tmp_path: Path) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    well_out_dir = tmp_path / "well001"
    stage_ks_dir = well_out_dir / "spikesort_outputs" / "sorter_output" / "sorter_output"
    stage_ks_dir.mkdir(parents=True, exist_ok=True)
    (stage_ks_dir / "params.py").write_text("n_channels_dat=4\n", encoding="utf-8")

    merge_ks_dir = well_out_dir / "spikesort_outputs" / "merge_outputs" / "sorter_output" / "sorter_output"
    merge_ks_dir.mkdir(parents=True, exist_ok=True)
    (merge_ks_dir / "params.py").write_text("n_channels_dat=8\n", encoding="utf-8")

    resolved = spikesort_runner._resolve_sorter_output_dir(
        well_out_dir=well_out_dir,
        output_rel_root="spikesort_outputs",
        stage_config=SimpleNamespace(
            merge_rel_output_root="merge_outputs",
            slay_sorter_output_relpath=None,
        ),
    )

    assert resolved == stage_ks_dir.resolve()


def test_load_sorting_from_sorter_output_dir_tries_wrapper_parent(tmp_path: Path) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    wrapper_dir = tmp_path / "sorter_output"
    nested_dir = wrapper_dir / "sorter_output"
    nested_dir.mkdir(parents=True, exist_ok=True)

    class _FakeSI:
        def read_sorter_folder(self, folder, sorter_name=None):
            if Path(folder).resolve() == wrapper_dir.resolve():
                return "loaded_from_wrapper"
            raise RuntimeError("wrong folder")

        def load_extractor(self, folder):
            raise RuntimeError("not used")

    loaded = spikesort_runner._load_sorting_from_sorter_output_dir(
        si_module=_FakeSI(),
        sorter_output_dir=nested_dir,
        sorter_name="kilosort4",
    )

    assert loaded == "loaded_from_wrapper"


def test_load_sorting_from_sorter_output_dir_prefers_kilosort_raw_ids(tmp_path: Path) -> None:
    import numpy as np

    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    sorter_dir = tmp_path / "sorter_output"
    sorter_dir.mkdir(parents=True, exist_ok=True)
    np.save(sorter_dir / "spike_times.npy", np.array([0, 1, 2, 3], dtype=np.int64))
    np.save(sorter_dir / "spike_clusters.npy", np.array([1, 2, 210, 210], dtype=np.int32))

    class _ReadKiloSorting:
        def get_unit_ids(self):
            return [1, 2]

        def get_sampling_frequency(self):
            return 10000.0

    class _FullSorting:
        def __init__(self, unit_ids):
            self._unit_ids = list(unit_ids)

        def get_unit_ids(self):
            return list(self._unit_ids)

    calls: dict[str, object] = {}

    class _FakeNumpySorting:
        @staticmethod
        def from_times_labels(*, times_list, labels_list, sampling_frequency, unit_ids=None):
            calls["from_times_labels"] = {
                "sampling_frequency": float(sampling_frequency),
                "unit_ids": list(unit_ids or []),
                "n_times": int(len(times_list[0])),
                "n_labels": int(len(labels_list[0])),
            }
            return _FullSorting(unit_ids or [])

    class _FakeSI:
        NumpySorting = _FakeNumpySorting

        def read_kilosort(self, folder, keep_good_only=False, remove_empty_units=False):
            calls["read_kilosort"] = {
                "folder": str(Path(folder).resolve()),
                "keep_good_only": bool(keep_good_only),
                "remove_empty_units": bool(remove_empty_units),
            }
            return _ReadKiloSorting()

        def read_sorter_folder(self, folder, sorter_name=None):
            raise AssertionError("read_sorter_folder should not be used for this kilosort path")

        def load_extractor(self, folder):
            raise AssertionError("load_extractor should not be used for this kilosort path")

    loaded = spikesort_runner._load_sorting_from_sorter_output_dir(
        si_module=_FakeSI(),
        sorter_output_dir=sorter_dir,
        sorter_name="kilosort2_5",
    )

    assert calls.get("read_kilosort") == {
        "folder": str(sorter_dir.resolve()),
        "keep_good_only": False,
        "remove_empty_units": False,
    }
    assert calls.get("from_times_labels") == {
        "sampling_frequency": 10000.0,
        "unit_ids": [1, 2, 210],
        "n_times": 4,
        "n_labels": 4,
    }
    assert loaded.get_unit_ids() == [1, 2, 210]


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
                "sort_original_preprocess_concat_recording_relpath": getattr(
                    inputs,
                    "sort_original_preprocess_concat_recording_relpath",
                    None,
                ),
                "sort_bootstrapped_concat_recording_relpath": getattr(
                    inputs,
                    "sort_bootstrapped_concat_recording_relpath",
                    None,
                ),
                "sort_use_bootstrapped_concat_binary": bool(
                    getattr(inputs, "sort_use_bootstrapped_concat_binary")
                ),
                "sort_use_lazy_source": bool(getattr(inputs, "sort_use_lazy_source")),
                "sort_assert_one_source": bool(getattr(inputs, "sort_assert_one_source")),
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
                "limit_segments_per_well": getattr(inputs, "limit_segments_per_well"),
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
        sort_original_preprocess_concat_recording_relpath="preprocess_outputs/concatenated_recording",
        sort_bootstrapped_concat_recording_relpath="spikesort_outputs_v2/cache/bootstrap_concat_binary/recording",
        sort_use_bootstrapped_concat_binary=True,
        sort_use_lazy_source=False,
        sort_assert_one_source=True,
        logging_enabled=False,
        logging_verbose=True,
        logging_file_relpath="logs/custom_spikesort.log",
        debug_outputs=True,
        run_reports=False,
        plot_mode="merged",
        plot_debug=True,
        raster_sort="unit_id",
        fixed_y=True,
        no_curation=True,
        export_to_phy=True,
        debug_limit_segments_per_well=2,
    )

    result = run_spikesort_stage(inputs)
    summary = _read_json(result.summary_json)

    assert captured_legacy_inputs.get("log_enabled") is False
    assert captured_legacy_inputs.get("log_verbose") is True
    assert captured_legacy_inputs.get("log_file_override") == "logs/custom_spikesort.log"
    assert captured_legacy_inputs.get("preprocess_concat_recording_relpath") == "preprocess_outputs/preprocessed_recording"
    assert captured_legacy_inputs.get("sort_original_preprocess_concat_recording_relpath") == "preprocess_outputs/concatenated_recording"
    assert captured_legacy_inputs.get("sort_bootstrapped_concat_recording_relpath") == "spikesort_outputs_v2/cache/bootstrap_concat_binary/recording"
    assert captured_legacy_inputs.get("sort_use_bootstrapped_concat_binary") is True
    assert captured_legacy_inputs.get("sort_use_lazy_source") is False
    assert captured_legacy_inputs.get("sort_assert_one_source") is True
    assert captured_legacy_inputs.get("output_subdir_after_well") == "spikesort_outputs_v2"
    assert captured_legacy_inputs.get("plot_mode") == "merged"
    assert captured_legacy_inputs.get("plot_debug") is True
    assert captured_legacy_inputs.get("raster_sort") == "unit_id"
    assert captured_legacy_inputs.get("fixed_y") is True
    assert captured_legacy_inputs.get("run_reports") is False
    assert captured_legacy_inputs.get("no_curation") is True
    assert captured_legacy_inputs.get("export_to_phy") is True
    assert captured_legacy_inputs.get("limit_segments_per_well") == 2

    assert summary.get("inputs", {}).get("logging_enabled") is False
    assert summary.get("inputs", {}).get("logging_verbose") is True
    assert summary.get("inputs", {}).get("logging_file_relpath") == "logs/custom_spikesort.log"
    assert summary.get("inputs", {}).get("debug_outputs") is True
    assert summary.get("inputs", {}).get("preprocess_concat_recording_relpath") == "preprocess_outputs/preprocessed_recording"
    assert summary.get("inputs", {}).get("sort_original_preprocess_concat_recording_relpath") == "preprocess_outputs/concatenated_recording"
    assert summary.get("inputs", {}).get("sort_bootstrapped_concat_recording_relpath") == "spikesort_outputs_v2/cache/bootstrap_concat_binary/recording"
    assert summary.get("inputs", {}).get("sort_use_bootstrapped_concat_binary") is True
    assert summary.get("inputs", {}).get("sort_use_lazy_source") is False
    assert summary.get("inputs", {}).get("sort_assert_one_source") is True
    assert summary.get("inputs", {}).get("plot_mode") == "merged"
    assert summary.get("inputs", {}).get("plot_debug") is True
    assert summary.get("inputs", {}).get("raster_sort") == "unit_id"
    assert summary.get("inputs", {}).get("fixed_y") is True
    assert summary.get("inputs", {}).get("run_reports") is False
    assert summary.get("inputs", {}).get("no_curation") is True
    assert summary.get("inputs", {}).get("export_to_phy") is True
    assert summary.get("inputs", {}).get("sort_enabled") is True
    assert summary.get("inputs", {}).get("sort_delete_outputs_on_force_restart") is False
    assert summary["applied_debug_limits"]["limit_segments_per_well"] == 2
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
    from axon_recon.pipeline.output_paths import compute_mea_analysis_output_dir
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    h5_path = tmp_path / "raw_data" / "input.raw.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"")
    well_out_dir = tmp_path / "well001"
    replot_workspace_analyzer_dir = (
        tmp_path
        / "well001"
        / "spikesort_outputs"
        / "merge_output"
        / "cache"
        / "merge_workspace"
        / "pre_merge_analyzer_output"
    )
    replot_workspace_analyzer_dir.mkdir(parents=True, exist_ok=True)

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
    (ks_dir / "data.bin").write_bytes(b"0")

    stale_merge_out_dir = well_out_dir / output_rel_root / "SLAy_outputs"
    stale_merge_out_dir.mkdir(parents=True, exist_ok=True)
    (stale_merge_out_dir / "stale.txt").write_text("old", encoding="utf-8")

    def _fake_import_slay_run_function(*, allow_numpy_fallback):
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
        merge_slay_dry_run=False,
        slay_relpath="SLAy_outputs",
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
    slay_method = next((m for m in summary.get("methods", []) if str(m.get("name")) == "slay"), {})
    assert str(stale_merge_out_dir) in list(slay_method.get("removed_on_force_restart", []))
    assert not (stale_merge_out_dir / "stale.txt").exists()
    assert groups.get("n_groups") == 1
    assert groups.get("merge_groups", {}).get("100") == [1, 2, 3]
    assert "cluster_a\tcluster_b" in candidates_tsv
    assert "1\t2" in candidates_tsv
    assert "1\t3" in candidates_tsv
    assert "2\t3" in candidates_tsv


def test_run_spikesort_merge_stage_reports_plot_generation_note_when_auto_accept_enabled(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.output_paths import compute_mea_analysis_output_dir
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
    (ks_dir / "data.bin").write_bytes(b"0")

    def _fake_import_slay_run_function(*, allow_numpy_fallback):
        def _fake_run_slay(args):
            automerge_dir = Path(args["KS_folder"]) / "automerge"
            automerge_dir.mkdir(parents=True, exist_ok=True)
            (automerge_dir / "new2old.json").write_text(
                json.dumps({"100": [1, 2]}),
                encoding="utf-8",
            )
            (automerge_dir / "metrics.tsv").write_text(
                "Cluster 1\tCluster 2\tSimilarity\tCross-correlation Significance\tRefractory Period Penalty\tFinal Metric\n"
                "1\t2\t0.91\t0.11\t0.01\t0.73\n",
                encoding="utf-8",
            )
            Path(args["output_json"]).write_text(json.dumps({"num_merges": 1}), encoding="utf-8")

        return _fake_run_slay

    monkeypatch.setattr(spikesort_runner, "_import_slay_run_function", _fake_import_slay_run_function)

    stage_cfg = SimpleNamespace(
        slay_enabled=True,
        merge_slay_dry_run=False,
        slay_relpath="SLAy_outputs",
        slay_sorter_output_relpath=None,
        slay_output_json_relpath="run-output.json",
        slay_candidate_pairs_relpath="recommended_merge_candidates.tsv",
        slay_merge_groups_relpath="recommended_merge_groups.json",
        slay_allow_numpy_fallback=True,
        slay_plot_merges=True,
        slay_auto_accept_merges=True,
        slay_copy_automerge_artifacts=True,
        slay_delete_outputs_on_force_restart=True,
        slay_params=None,
    )

    result = run_spikesort_merge_stage(
        h5_path=h5_path,
        stream_id=stream_id,
        mea_output_root=tmp_path,
        output_rel_root=output_rel_root,
        stage_config=stage_cfg,
        force_restart=False,
    )

    slay_summary = _read_json(result.merge_out_dir / "slay_method_summary.json")

    assert slay_summary.get("plot_files_generated") == 0
    assert slay_summary.get("plot_files_generated_in_snapshot") == 0
    assert "plot_generation_note" in slay_summary
    assert "auto_accept_merges=true" in str(slay_summary.get("plot_generation_note", ""))


def test_run_spikesort_merge_stage_releases_pre_merge_analyzer_before_slay(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.output_paths import compute_mea_analysis_output_dir
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
    sorter_output_dir = well_out_dir / output_rel_root / "sorter_output" / "sorter_output"
    sorter_output_dir.mkdir(parents=True, exist_ok=True)
    (sorter_output_dir / "params.py").write_text(
        "dat_path = 'data.bin'\n"
        "n_channels_dat = 4\n"
        "dtype = 'int16'\n"
        "sample_rate = 30000\n",
        encoding="utf-8",
    )
    (sorter_output_dir / "data.bin").write_bytes(b"0")

    class _FakeAnalyzer:
        pass

    pre_analyzer = _FakeAnalyzer()
    release_called = False

    monkeypatch.setattr(
        spikesort_runner,
        "_resolve_sorter_output_dir",
        lambda **kwargs: sorter_output_dir,
    )
    monkeypatch.setattr(
        spikesort_runner,
        "_run_bombcell_label_phase",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("merge stage should not invoke bombcell")),
    )
    monkeypatch.setattr(
        spikesort_runner,
        "_import_spikeinterface_full_module",
        lambda: object(),
    )
    monkeypatch.setattr(
        spikesort_runner,
        "_load_concat_analyzer_for_phase",
        lambda **kwargs: (pre_analyzer, sorter_output_dir.parent.parent / "concat_analyzer"),
    )
    monkeypatch.setattr(
        spikesort_runner,
        "_ensure_merge_analyzer_extensions",
        lambda **kwargs: [],
    )

    def _fake_capture(**kwargs):
        if kwargs.get("capture_label") == "before_merge":
            assert kwargs.get("analyzer_obj") is pre_analyzer
        return {
            "sorter": {"available": True, "unit_ids": ["1"], "unit_count": 1},
            "analyzer": {
                "available": True,
                "unit_ids": ["1"],
                "unit_count": 1,
                "source_dir": str(kwargs.get("analyzer_source_dir", sorter_output_dir.parent / "analyzer_output")),
                "unit_locations_by_unit": {},
            },
        }

    monkeypatch.setattr(spikesort_runner, "_capture_merge_state_snapshot", _fake_capture)

    def _fake_release(*, analyzer, extension_names=None):
        nonlocal release_called
        assert analyzer is pre_analyzer
        release_called = True
        return ["waveforms", "templates"]

    monkeypatch.setattr(spikesort_runner, "_release_loaded_analyzer_extensions", _fake_release)

    def _fake_slay(**kwargs):
        assert release_called is True
        return {
            "name": "slay",
            "status": "ok",
            "reason": None,
            "out_dir": str((well_out_dir / output_rel_root / "SLAy_outputs").resolve()),
            "summary_json": None,
            "outputs": {},
            "ks_dir": str(sorter_output_dir),
            "applied_merges": False,
        }

    monkeypatch.setattr(spikesort_runner, "_run_slay_merge_method", _fake_slay)

    result = run_spikesort_merge_stage(
        h5_path=h5_path,
        stream_id=stream_id,
        mea_output_root=tmp_path,
        output_rel_root=output_rel_root,
        stage_config=SimpleNamespace(
            merge_units_enabled=True,
            merge_sequence=["slay"],
            slay_enabled=True,
            merge_slay_dry_run=False,
            slay_relpath="SLAy_outputs",
            slay_auto_accept_merges=False,
            slay_recompute_analyzer=False,
            pre_merge_metadata_enabled=True,
            pre_merge_metadata_write_json=True,
            pre_merge_metadata_include_unit_locations=True,
            merge_reports_enabled=False,
            bombcell_label_enabled=True,
        ),
        force_restart=False,
    )

    assert result.summary_json.exists()
    assert release_called is True



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


def test_run_spikesort_merge_stage_uses_merge_rel_output_root_for_stage_outputs_when_disabled(tmp_path: Path) -> None:
    h5_path = tmp_path / "raw_data" / "input.raw.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"")

    stage_cfg = SimpleNamespace(
        merge_units_enabled=False,
        merge_rel_output_root="merge_outputs",
        slay_relpath="SLAy_outputs",
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
    assert result.merge_out_dir == result.well_out_dir / "spikesort_outputs" / "merge_outputs"
    assert summary.get("merge_rel_output_root") == "merge_outputs"
    assert summary.get("merge_output_rel_root") == "spikesort_outputs/merge_outputs"


def test_run_slay_merge_method_writes_outputs_under_merge_rel_output_root(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    output_rel_root = "spikesort_outputs"
    well_out_dir = tmp_path / "well001"
    stage_output_root_dir = well_out_dir / output_rel_root
    ks_dir = stage_output_root_dir / "sorter_output" / "sorter_output"
    ks_dir.mkdir(parents=True, exist_ok=True)
    (ks_dir / "params.py").write_text(
        "dat_path = 'data.bin'\n"
        "n_channels_dat = 4\n"
        "dtype = 'int16'\n"
        "sample_rate = 30000\n",
        encoding="utf-8",
    )
    (ks_dir / "data.bin").write_bytes(b"0")

    def _fake_import_slay_run_function(*, allow_numpy_fallback):
        def _fake_run_slay(args):
            automerge_dir = Path(args["KS_folder"]) / "automerge"
            automerge_dir.mkdir(parents=True, exist_ok=True)
            (automerge_dir / "new2old.json").write_text(json.dumps({"100": [1, 2]}), encoding="utf-8")
            (automerge_dir / "metrics.tsv").write_text(
                "Cluster 1\tCluster 2\tSimilarity\tCross-correlation Significance\tRefractory Period Penalty\tFinal Metric\n"
                "1\t2\t0.91\t0.11\t0.01\t0.73\n",
                encoding="utf-8",
            )
            Path(args["output_json"]).write_text(json.dumps({"num_merges": 1}), encoding="utf-8")

        return _fake_run_slay

    monkeypatch.setattr(spikesort_runner, "_import_slay_run_function", _fake_import_slay_run_function)

    stage_cfg = SimpleNamespace(
        merge_rel_output_root="merge_outputs",
        slay_enabled=True,
        merge_slay_dry_run=False,
        slay_relpath="SLAy_outputs",
        slay_sorter_output_relpath=None,
        slay_output_json_relpath="run-output.json",
        slay_candidate_pairs_relpath="recommended_merge_candidates.tsv",
        slay_merge_groups_relpath="recommended_merge_groups.json",
        slay_allow_numpy_fallback=True,
        slay_plot_merges=False,
        slay_auto_accept_merges=False,
        slay_copy_automerge_artifacts=True,
        slay_delete_outputs_on_force_restart=True,
        slay_params=None,
    )

    report = spikesort_runner._run_slay_merge_method(
        well_out_dir=well_out_dir,
        stage_output_root_dir=stage_output_root_dir,
        output_rel_root=output_rel_root,
        stage_config=stage_cfg,
        force_restart=False,
    )

    expected_out_dir = (well_out_dir / output_rel_root / "merge_outputs" / "SLAy_outputs").resolve()
    assert Path(str(report.get("out_dir"))).resolve() == expected_out_dir
    assert (expected_out_dir / "slay_method_summary.json").exists()


def test_run_slay_merge_method_dry_run_preserves_canonical_sorter_output(tmp_path: Path, monkeypatch) -> None:
    """SLAy with dry_run=true must NOT mutate canonical sorter_output. SLAy is
    pointed at a per-run scratch copy, and artifacts land under
    <merge_out_dir>/dry_run/.
    """
    import hashlib

    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    output_rel_root = "spikesort_outputs"
    well_out_dir = tmp_path / "well001"
    stage_output_root_dir = well_out_dir / output_rel_root
    canonical_ks_dir = stage_output_root_dir / "sorter_output" / "sorter_output"
    canonical_ks_dir.mkdir(parents=True, exist_ok=True)
    (canonical_ks_dir / "params.py").write_text(
        "dat_path = 'data.bin'\n"
        "n_channels_dat = 4\n"
        "dtype = 'int16'\n"
        "sample_rate = 30000\n",
        encoding="utf-8",
    )
    (canonical_ks_dir / "data.bin").write_bytes(b"\x00" * 64)
    (canonical_ks_dir / "spike_clusters.npy").write_bytes(b"\x01" * 32)

    captured_ks_folders: list[Path] = []

    def _fake_import_slay_run_function(*, allow_numpy_fallback):
        def _fake_run_slay(args):
            ks_folder = Path(args["KS_folder"])
            captured_ks_folders.append(ks_folder.resolve())
            # Mutate the KS_folder as SLAy would in auto_accept_merges mode.
            (ks_folder / "spike_clusters.npy").write_bytes(b"\x99" * 32)
            (ks_folder / "params.py").write_text("MUTATED\n", encoding="utf-8")
            automerge_dir = ks_folder / "automerge"
            automerge_dir.mkdir(parents=True, exist_ok=True)
            (automerge_dir / "new2old.json").write_text(
                json.dumps({"100": [1, 2]}), encoding="utf-8"
            )
            (automerge_dir / "metrics.tsv").write_text(
                "Cluster 1\tCluster 2\tSimilarity\tCross-correlation Significance\tRefractory Period Penalty\tFinal Metric\n"
                "1\t2\t0.91\t0.11\t0.01\t0.73\n",
                encoding="utf-8",
            )
            Path(args["output_json"]).write_text(
                json.dumps({"num_merges": 1}), encoding="utf-8"
            )

        return _fake_run_slay

    monkeypatch.setattr(spikesort_runner, "_import_slay_run_function", _fake_import_slay_run_function)

    def _hash_dir(root: Path) -> dict[str, str]:
        out: dict[str, str] = {}
        for p in sorted(root.rglob("*")):
            if p.is_file():
                out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
        return out

    pre_hashes = _hash_dir(canonical_ks_dir)
    assert "spike_clusters.npy" in pre_hashes  # sanity

    stage_cfg = SimpleNamespace(
        merge_rel_output_root="merge_outputs",
        slay_enabled=True,
        merge_slay_dry_run=True,  # the knob under test
        slay_relpath="SLAy_outputs",
        slay_sorter_output_relpath=None,
        slay_output_json_relpath="run-output.json",
        slay_candidate_pairs_relpath="recommended_merge_candidates.tsv",
        slay_merge_groups_relpath="recommended_merge_groups.json",
        slay_allow_numpy_fallback=True,
        slay_plot_merges=False,
        slay_auto_accept_merges=True,  # SLAy WOULD mutate; dry_run protects us
        slay_copy_automerge_artifacts=True,
        slay_delete_outputs_on_force_restart=True,
        slay_params=None,
    )

    report = spikesort_runner._run_slay_merge_method(
        well_out_dir=well_out_dir,
        stage_output_root_dir=stage_output_root_dir,
        output_rel_root=output_rel_root,
        stage_config=stage_cfg,
        force_restart=False,
    )

    # 1. Canonical sorter_output must be byte-identical.
    post_hashes = _hash_dir(canonical_ks_dir)
    assert post_hashes == pre_hashes, "dry_run must not mutate canonical sorter_output"

    # 2. SLAy was pointed at the scratch, not the canonical KS_folder.
    expected_merge_out_dir = (
        well_out_dir / output_rel_root / "merge_outputs" / "SLAy_outputs"
    ).resolve()
    expected_scratch_dir = (expected_merge_out_dir / "dry_run" / "sorter_output_scratch").resolve()
    assert captured_ks_folders == [expected_scratch_dir]

    # 3. Artifacts land under <merge_out_dir>/dry_run/.
    dry_run_dir = (expected_merge_out_dir / "dry_run").resolve()
    assert (dry_run_dir / "run-output.json").exists()
    assert (dry_run_dir / "recommended_merge_groups.json").exists()
    assert (dry_run_dir / "automerge" / "new2old.json").exists()

    # 4. Report surface fields show dry_run state.
    assert report.get("dry_run") is True
    assert Path(str(report.get("ks_dir"))).resolve() == expected_scratch_dir
    assert Path(str(report.get("canonical_ks_dir"))).resolve() == canonical_ks_dir.resolve()
    assert report.get("applied_merges") is False  # canonical un-mutated even with auto_accept_merges=true


def test_run_slay_merge_method_normalizes_wrapper_sorter_output_path(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    output_rel_root = "spikesort_outputs"
    well_out_dir = tmp_path / "well001"
    stage_output_root_dir = well_out_dir / output_rel_root
    wrapper_dir = stage_output_root_dir / "sorter_output"
    ks_dir = wrapper_dir / "sorter_output"
    ks_dir.mkdir(parents=True, exist_ok=True)
    (ks_dir / "params.py").write_text("n_channels_dat=4\n", encoding="utf-8")

    def _fake_import_slay_run_function(*, allow_numpy_fallback):
        def _fake_run_slay(args):
            automerge_dir = Path(args["KS_folder"]) / "automerge"
            automerge_dir.mkdir(parents=True, exist_ok=True)
            (automerge_dir / "new2old.json").write_text(json.dumps({}), encoding="utf-8")

        return _fake_run_slay

    monkeypatch.setattr(spikesort_runner, "_import_slay_run_function", _fake_import_slay_run_function)

    stage_cfg = SimpleNamespace(
        slay_enabled=True,
        merge_slay_dry_run=False,
        slay_relpath="SLAy_outputs",
        slay_sorter_output_relpath=None,
        slay_output_json_relpath="run-output.json",
        slay_candidate_pairs_relpath="recommended_merge_candidates.tsv",
        slay_merge_groups_relpath="recommended_merge_groups.json",
        slay_allow_numpy_fallback=True,
        slay_plot_merges=False,
        slay_auto_accept_merges=False,
        slay_copy_automerge_artifacts=False,
        slay_delete_outputs_on_force_restart=True,
        slay_params=None,
    )

    report = spikesort_runner._run_slay_merge_method(
        well_out_dir=well_out_dir,
        stage_output_root_dir=stage_output_root_dir,
        output_rel_root=output_rel_root,
        stage_config=stage_cfg,
        force_restart=False,
        sorter_output_dir=wrapper_dir,
    )

    assert Path(str(report.get("ks_dir"))).resolve() == ks_dir.resolve()


def test_run_slay_merge_method_disables_model_cache_read_and_write_when_knobs_false(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    output_rel_root = "spikesort_outputs"
    well_out_dir = tmp_path / "well001"
    stage_output_root_dir = well_out_dir / output_rel_root
    ks_dir = stage_output_root_dir / "sorter_output" / "sorter_output"
    ks_dir.mkdir(parents=True, exist_ok=True)
    (ks_dir / "params.py").write_text("n_channels_dat=4\n", encoding="utf-8")

    captured_args: dict[str, object] = {}

    def _fake_import_slay_run_function(*, allow_numpy_fallback):
        def _fake_run_slay(args):
            captured_args.update(dict(args))
            automerge_dir = Path(args["KS_folder"]) / "automerge"
            automerge_dir.mkdir(parents=True, exist_ok=True)
            (automerge_dir / "new2old.json").write_text(json.dumps({}), encoding="utf-8")
            Path(args["output_json"]).write_text(json.dumps({"num_merges": 0}), encoding="utf-8")

        return _fake_run_slay

    monkeypatch.setattr(spikesort_runner, "_import_slay_run_function", _fake_import_slay_run_function)

    stage_cfg = SimpleNamespace(
        slay_enabled=True,
        merge_slay_dry_run=False,
        slay_relpath="SLAy_outputs",
        slay_sorter_output_relpath=None,
        slay_output_json_relpath="run-output.json",
        slay_candidate_pairs_relpath="recommended_merge_candidates.tsv",
        slay_merge_groups_relpath="recommended_merge_groups.json",
        slay_allow_numpy_fallback=True,
        slay_plot_merges=False,
        slay_auto_accept_merges=False,
        slay_copy_automerge_artifacts=True,
        slay_delete_outputs_on_force_restart=True,
        slay_model_cache_relpath="cache/slay_model/ae.pt",
        slay_model_cache_use_cached_model=False,
        slay_model_cache_write_model=False,
        slay_params=None,
    )

    report = spikesort_runner._run_slay_merge_method(
        well_out_dir=well_out_dir,
        stage_output_root_dir=stage_output_root_dir,
        output_rel_root=output_rel_root,
        stage_config=stage_cfg,
        force_restart=False,
    )

    summary = _read_json(Path(str(report.get("summary_json"))))
    assert "model_path" not in captured_args
    assert summary.get("run_args", {}).get("model_path") is None
    assert summary.get("slay_model_cache_use_cached_model") is False
    assert summary.get("slay_model_cache_write_model") is False
    assert "slay.model_cache_path" not in dict(report.get("outputs", {}))


def test_run_slay_merge_method_retrains_without_using_existing_cached_model_when_disabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    output_rel_root = "spikesort_outputs"
    well_out_dir = tmp_path / "well001"
    stage_output_root_dir = well_out_dir / output_rel_root
    ks_dir = stage_output_root_dir / "sorter_output" / "sorter_output"
    ks_dir.mkdir(parents=True, exist_ok=True)
    (ks_dir / "params.py").write_text("n_channels_dat=4\n", encoding="utf-8")

    model_cache_path = stage_output_root_dir / "cache" / "slay_model" / "ae.pt"
    model_cache_path.parent.mkdir(parents=True, exist_ok=True)
    model_cache_path.write_text("old-model", encoding="utf-8")

    captured_args: dict[str, object] = {}

    def _fake_import_slay_run_function(*, allow_numpy_fallback):
        def _fake_run_slay(args):
            captured_args.update(dict(args))
            model_path = Path(str(args["model_path"]))
            assert not model_path.exists()
            model_path.write_text("new-model", encoding="utf-8")

            automerge_dir = Path(args["KS_folder"]) / "automerge"
            automerge_dir.mkdir(parents=True, exist_ok=True)
            (automerge_dir / "new2old.json").write_text(json.dumps({}), encoding="utf-8")
            Path(args["output_json"]).write_text(json.dumps({"num_merges": 0}), encoding="utf-8")

        return _fake_run_slay

    monkeypatch.setattr(spikesort_runner, "_import_slay_run_function", _fake_import_slay_run_function)

    stage_cfg = SimpleNamespace(
        slay_enabled=True,
        merge_slay_dry_run=False,
        slay_relpath="SLAy_outputs",
        slay_sorter_output_relpath=None,
        slay_output_json_relpath="run-output.json",
        slay_candidate_pairs_relpath="recommended_merge_candidates.tsv",
        slay_merge_groups_relpath="recommended_merge_groups.json",
        slay_allow_numpy_fallback=True,
        slay_plot_merges=False,
        slay_auto_accept_merges=False,
        slay_copy_automerge_artifacts=True,
        slay_delete_outputs_on_force_restart=True,
        slay_model_cache_relpath="cache/slay_model/ae.pt",
        slay_model_cache_use_cached_model=False,
        slay_model_cache_write_model=True,
        slay_params=None,
    )

    report = spikesort_runner._run_slay_merge_method(
        well_out_dir=well_out_dir,
        stage_output_root_dir=stage_output_root_dir,
        output_rel_root=output_rel_root,
        stage_config=stage_cfg,
        force_restart=False,
    )

    summary = _read_json(Path(str(report.get("summary_json"))))
    assert str(captured_args.get("model_path")) == str(model_cache_path.resolve())
    assert model_cache_path.read_text(encoding="utf-8") == "new-model"
    assert summary.get("slay_model_deleted_to_disable_cache_use") is True
    assert summary.get("slay_model_cache_use_cached_model") is False
    assert summary.get("slay_model_cache_write_model") is True


def test_resolve_sorter_output_dir_resolves_slay_sorter_relpath_from_merge_root(tmp_path: Path) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    well_out_dir = tmp_path / "well001"
    ks_dir = well_out_dir / "spikesort_outputs" / "sorter_output" / "sorter_output"
    ks_dir.mkdir(parents=True, exist_ok=True)
    (ks_dir / "params.py").write_text("n_channels_dat=4\n", encoding="utf-8")

    resolved = spikesort_runner._resolve_sorter_output_dir(
        well_out_dir=well_out_dir,
        output_rel_root="spikesort_outputs",
        stage_config=SimpleNamespace(
            merge_rel_output_root="merge_outputs",
            slay_sorter_output_relpath="../sorter_output/sorter_output",
        ),
    )

    assert resolved == ks_dir.resolve()


def test_run_spikesort_merge_stage_skips_entire_phase_when_merge_units_disabled(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    h5_path = tmp_path / "raw_data" / "input.raw.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"")

    def _never_run_slay(**kwargs):
        raise AssertionError("SLAy method should not execute when merge_units is disabled")

    monkeypatch.setattr(spikesort_runner, "_run_slay_merge_method", _never_run_slay)

    stage_cfg = SimpleNamespace(
        merge_units_enabled=False,
        slay_enabled=True,
        merge_slay_dry_run=False,
        slay_relpath="SLAy_outputs",
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

    assert summary.get("status") == "skipped"
    assert summary.get("reason") == "merge_units_disabled"
    assert summary.get("methods") == []
    assert summary.get("merge_units_enabled") is False
    assert result.outputs.get("summary_json") == str(result.summary_json)


def test_run_spikesort_merge_stage_preserves_existing_outputs_when_delete_disabled(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.output_paths import compute_mea_analysis_output_dir
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
    (ks_dir / "data.bin").write_bytes(b"0")

    merge_out_dir = well_out_dir / output_rel_root / "SLAy_outputs"
    merge_out_dir.mkdir(parents=True, exist_ok=True)
    sentinel = merge_out_dir / "keep_me.txt"
    sentinel.write_text("persist", encoding="utf-8")

    def _fake_import_slay_run_function(*, allow_numpy_fallback):
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
        merge_slay_dry_run=False,
        slay_relpath="SLAy_outputs",
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


def test_run_spikesort_merge_stage_force_replot_uses_unit_diff_json_as_2panel_source_of_truth(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    h5_path = tmp_path / "raw_data" / "input.raw.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"")

    well_out_dir = tmp_path / "well001"
    merge_out_dir = well_out_dir / "spikesort_outputs" / "SLAy_outputs"
    merge_out_dir.mkdir(parents=True, exist_ok=True)
    replot_workspace_analyzer_dir = (
        well_out_dir
        / "spikesort_outputs"
        / "merge_output"
        / "cache"
        / "merge_workspace"
        / "pre_merge_analyzer_output"
    )
    replot_workspace_analyzer_dir.mkdir(parents=True, exist_ok=True)

    metadata_json = merge_out_dir / "merge_metadata_summary.json"
    metadata_json.write_text(
        json.dumps(
            {
                "before": {
                    "analyzer": {
                        "unit_locations_by_unit": {
                            "9": {"x_um": 90.0, "y_um": 90.0},
                        }
                    }
                },
                "after": {
                    "analyzer": {
                        "unit_locations_by_unit": {
                            "9": {"x_um": 91.0, "y_um": 91.0},
                        }
                    }
                },
                "applied_unit_mappings": [
                    {
                        "pre_unit_ids": ["9"],
                        "post_unit_id": "9",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    unit_diff_json = merge_out_dir / "unit_diffs_after_merge.json"
    unit_diff_json.write_text(
        json.dumps(
            {
                "before": {
                    "analyzer": {
                        "unit_locations_by_unit": {
                            "1": {"x_um": 10.0, "y_um": 20.0},
                            "2": {"x_um": 30.0, "y_um": 40.0},
                        }
                    }
                },
                "after": {
                    "analyzer": {
                        "unit_locations_by_unit": {
                            "1": {"x_um": 11.0, "y_um": 21.0},
                        }
                    }
                },
                "applied_unit_mappings": [
                    {
                        "pre_unit_ids": ["1", "2"],
                        "post_unit_id": "1",
                    }
                ],
                "applied_merge_group_count": 1,
            }
        ),
        encoding="utf-8",
    )

    summary_json = merge_out_dir / "merge_stage_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "status": "ok",
                "methods": [
                    {"name": "slay", "status": "ok"},
                ],
                "merge_metadata_summary_json": str(metadata_json),
                "merge_unit_diff_json": str(unit_diff_json),
                "outputs": {
                    "merge.metadata_summary_json": str(metadata_json),
                    "merge.report.unit_diff_json": str(unit_diff_json),
                    "merge.replot_workspace_analyzer_output_dir": str(replot_workspace_analyzer_dir),
                },
            }
        ),
        encoding="utf-8",
    )

    def _fail_if_called(**kwargs):
        raise AssertionError("merge methods must not run in force_replot-only mode")

    report_calls: list[dict[str, object]] = []

    def _fake_write_reports(*, merge_out_dir, before_snapshot, after_snapshot, applied_unit_mappings, stage_config):
        report_calls.append(
            {
                "before_count": len(before_snapshot.get("analyzer", {}).get("unit_locations_by_unit", {})),
                "after_count": len(after_snapshot.get("analyzer", {}).get("unit_locations_by_unit", {})),
                "applied_mappings_count": len(list(applied_unit_mappings or [])),
            }
        )
        return {
            "status": "ok",
            "before_unit_locations_count": 2,
            "after_unit_locations_count": 1,
            "outputs": {
                "merge.report.unit_locations_before_after_png": str(merge_out_dir / "unit_locations_before_after_merge.png"),
            },
        }

    monkeypatch.setattr(spikesort_runner, "compute_mea_analysis_output_dir", lambda **kwargs: well_out_dir)
    monkeypatch.setattr(spikesort_runner, "_run_slay_merge_method", _fail_if_called)
    monkeypatch.setattr(spikesort_runner, "_write_merge_unit_location_reports", _fake_write_reports)

    stage_cfg = SimpleNamespace(
        merge_sequence=("SLAy",),
        merge_units_enabled=True,
        slay_relpath="SLAy_outputs",
        merge_metadata_enabled=True,
        merge_metadata_write_json=True,
        merge_metadata_json_relpath="merge_metadata_summary.json",
        merge_reports_enabled=True,
        merge_reports_unit_diff_json_enabled=True,
        merge_reports_unit_diff_json_relpath="unit_diffs_after_merge.json",
        merge_reports_2panel_enabled=True,
    )

    result = run_spikesort_merge_stage(
        h5_path=h5_path,
        stream_id="well001",
        mea_output_root=tmp_path,
        output_rel_root="spikesort_outputs",
        stage_config=stage_cfg,
        force_restart=False,
        force_replot=True,
    )

    summary = _read_json(result.summary_json)

    assert summary.get("status") == "ok"
    assert summary.get("replot_only") is True
    assert len(report_calls) == 1
    assert report_calls[0].get("before_count") == 2
    assert report_calls[0].get("after_count") == 1
    assert report_calls[0].get("applied_mappings_count") == 1
    assert result.outputs.get("merge.report.unit_diff_json") == str(unit_diff_json)


def test_run_spikesort_merge_stage_force_replot_only_uses_existing_metadata(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    h5_path = tmp_path / "raw_data" / "input.raw.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"")

    well_out_dir = tmp_path / "well001"
    merge_out_dir = well_out_dir / "spikesort_outputs" / "SLAy_outputs"
    merge_out_dir.mkdir(parents=True, exist_ok=True)
    replot_workspace_analyzer_dir = (
        well_out_dir
        / "spikesort_outputs"
        / "merge_output"
        / "cache"
        / "merge_workspace"
        / "pre_merge_analyzer_output"
    )
    replot_workspace_analyzer_dir.mkdir(parents=True, exist_ok=True)

    metadata_json = merge_out_dir / "merge_metadata_summary.json"
    metadata_json.write_text(
        json.dumps(
            {
                "before": {
                    "analyzer": {
                        "unit_locations_by_unit": {
                            "1": {"x_um": 10.0, "y_um": 20.0},
                        }
                    }
                },
                "after": {
                    "analyzer": {
                        "unit_locations_by_unit": {
                            "1": {"x_um": 11.0, "y_um": 21.0},
                        }
                    }
                },
                "applied_unit_mappings": [
                    {
                        "pre_unit_ids": ["1"],
                        "post_unit_id": "1",
                    }
                ],
                "applied_merge_group_count": 1,
                "change_validation": {"passes": True},
            }
        ),
        encoding="utf-8",
    )

    summary_json = merge_out_dir / "merge_stage_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "status": "ok",
                "methods": [
                    {"name": "slay", "status": "ok"},
                ],
                "merge_metadata_summary_json": str(metadata_json),
                "outputs": {
                    "merge.metadata_summary_json": str(metadata_json),
                    "merge.replot_workspace_analyzer_output_dir": str(replot_workspace_analyzer_dir),
                },
            }
        ),
        encoding="utf-8",
    )

    def _fail_if_called(**kwargs):
        raise AssertionError("merge methods must not run in force_replot-only mode")

    report_calls: list[dict[str, object]] = []

    def _fake_write_reports(*, merge_out_dir, before_snapshot, after_snapshot, applied_unit_mappings, stage_config):
        report_calls.append(
            {
                "before_count": len(before_snapshot.get("analyzer", {}).get("unit_locations_by_unit", {})),
                "after_count": len(after_snapshot.get("analyzer", {}).get("unit_locations_by_unit", {})),
                "applied_mappings_count": len(list(applied_unit_mappings or [])),
            }
        )
        return {
            "status": "ok",
            "before_unit_locations_count": 1,
            "after_unit_locations_count": 1,
            "outputs": {
                "merge.report.unit_locations_before_after_png": str(merge_out_dir / "unit_locations_before_after_merge.png"),
            },
        }

    monkeypatch.setattr(spikesort_runner, "compute_mea_analysis_output_dir", lambda **kwargs: well_out_dir)
    monkeypatch.setattr(spikesort_runner, "_run_slay_merge_method", _fail_if_called)
    monkeypatch.setattr(spikesort_runner, "_write_merge_unit_location_reports", _fake_write_reports)

    stage_cfg = SimpleNamespace(
        merge_sequence=("SLAy", "auto_merge"),
        merge_units_enabled=True,
        slay_relpath="SLAy_outputs",
        merge_metadata_enabled=True,
        merge_metadata_write_json=True,
        merge_metadata_json_relpath="merge_metadata_summary.json",
        merge_reports_enabled=True,
        merge_reports_2panel_enabled=True,
    )

    result = run_spikesort_merge_stage(
        h5_path=h5_path,
        stream_id="well001",
        mea_output_root=tmp_path,
        output_rel_root="spikesort_outputs",
        stage_config=stage_cfg,
        force_restart=False,
        force_replot=True,
    )

    summary = _read_json(result.summary_json)

    assert summary.get("status") == "ok"
    assert summary.get("replot_only") is True
    assert summary.get("force_replot") is True
    assert len(report_calls) == 1
    assert report_calls[0].get("before_count") == 1
    assert report_calls[0].get("after_count") == 1
    assert report_calls[0].get("applied_mappings_count") == 1
    assert "merge.report.unit_locations_before_after_png" in result.outputs


def test_run_spikesort_merge_stage_force_replot_only_does_not_fallback_to_applied_operations(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    h5_path = tmp_path / "raw_data" / "input.raw.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"")

    well_out_dir = tmp_path / "well001"
    merge_out_dir = well_out_dir / "spikesort_outputs" / "SLAy_outputs"
    merge_out_dir.mkdir(parents=True, exist_ok=True)
    replot_workspace_analyzer_dir = (
        well_out_dir
        / "spikesort_outputs"
        / "merge_output"
        / "cache"
        / "merge_workspace"
        / "pre_merge_analyzer_output"
    )
    replot_workspace_analyzer_dir.mkdir(parents=True, exist_ok=True)

    metadata_json = merge_out_dir / "merge_metadata_summary.json"
    metadata_json.write_text(
        json.dumps(
            {
                "before": {
                    "analyzer": {
                        "unit_locations_by_unit": {
                            "190": {"x_um": 2653.33, "y_um": 2079.95},
                            "195": {"x_um": 2653.45, "y_um": 2080.30},
                        }
                    }
                },
                "after": {
                    "analyzer": {
                        "unit_locations_by_unit": {
                            "105": {"x_um": 2653.69, "y_um": 2079.56},
                        }
                    }
                },
                "applied_merge_operations": [
                    {
                        "method": "slay",
                        "group_id": "202",
                        "pre_unit_ids": ["190", "195"],
                    }
                ],
                "applied_merge_group_count": 1,
                "change_validation": {"passes": True},
            }
        ),
        encoding="utf-8",
    )

    summary_json = merge_out_dir / "merge_stage_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "status": "ok",
                "methods": [
                    {"name": "slay", "status": "ok"},
                ],
                "merge_metadata_summary_json": str(metadata_json),
                "outputs": {
                    "merge.metadata_summary_json": str(metadata_json),
                    "merge.replot_workspace_analyzer_output_dir": str(replot_workspace_analyzer_dir),
                },
            }
        ),
        encoding="utf-8",
    )

    def _fail_if_called(**kwargs):
        raise AssertionError("merge methods must not run in force_replot-only mode")

    report_calls: list[dict[str, object]] = []

    def _fake_write_reports(*, merge_out_dir, before_snapshot, after_snapshot, applied_unit_mappings, stage_config):
        report_calls.append(
            {
                "before_count": len(before_snapshot.get("analyzer", {}).get("unit_locations_by_unit", {})),
                "after_count": len(after_snapshot.get("analyzer", {}).get("unit_locations_by_unit", {})),
                "applied_mappings_count": len(list(applied_unit_mappings or [])),
            }
        )
        return {
            "status": "ok",
            "before_unit_locations_count": 2,
            "after_unit_locations_count": 1,
            "outputs": {
                "merge.report.unit_locations_before_after_png": str(merge_out_dir / "unit_locations_before_after_merge.png"),
            },
        }

    monkeypatch.setattr(spikesort_runner, "compute_mea_analysis_output_dir", lambda **kwargs: well_out_dir)
    monkeypatch.setattr(spikesort_runner, "_run_slay_merge_method", _fail_if_called)
    monkeypatch.setattr(spikesort_runner, "_write_merge_unit_location_reports", _fake_write_reports)

    stage_cfg = SimpleNamespace(
        merge_sequence=("SLAy",),
        merge_units_enabled=True,
        slay_relpath="SLAy_outputs",
        merge_metadata_enabled=True,
        merge_metadata_write_json=True,
        merge_metadata_json_relpath="merge_metadata_summary.json",
        merge_reports_enabled=True,
        merge_reports_2panel_enabled=True,
    )

    result = run_spikesort_merge_stage(
        h5_path=h5_path,
        stream_id="well001",
        mea_output_root=tmp_path,
        output_rel_root="spikesort_outputs",
        stage_config=stage_cfg,
        force_restart=False,
        force_replot=True,
    )

    summary = _read_json(result.summary_json)

    assert summary.get("status") == "ok"
    assert summary.get("replot_only") is True
    assert len(report_calls) == 1
    assert report_calls[0].get("before_count") == 2
    assert report_calls[0].get("after_count") == 1
    assert report_calls[0].get("applied_mappings_count") == 0
    assert "merge.report.unit_locations_before_after_png" in result.outputs


def _seed_bombcell_test_fixture(*, tmp_path: Path):
    """Set up sorter_output + concat_analyzer dir for bombcell phase tests."""
    import numpy as np

    well_out_dir = tmp_path / "well001"
    stage_output_root_dir = well_out_dir / "spikesort_outputs"
    sorter_output_dir = stage_output_root_dir / "sorter_output"
    ks_dir = sorter_output_dir / "sorter_output"
    ks_dir.mkdir(parents=True, exist_ok=True)

    (ks_dir / "params.py").write_text("sample_rate = 30000\n", encoding="utf-8")
    np.save(ks_dir / "spike_times.npy", np.array([0, 1, 2, 3], dtype=np.int64))
    np.save(ks_dir / "spike_clusters.npy", np.array([1, 1, 2, 3], dtype=np.int64))
    (ks_dir / "cluster_KSLabel.tsv").write_text(
        "cluster_id\tKSLabel\n"
        "1\tgood\n"
        "2\tmua\n"
        "3\tgood\n",
        encoding="utf-8",
    )
    (ks_dir / "cluster_group.tsv").write_text(
        "cluster_id\tgroup\n"
        "1\tgood\n"
        "2\tmua\n"
        "3\tgood\n",
        encoding="utf-8",
    )

    concat_analyzer_dir = stage_output_root_dir / "concat_analyzer"
    concat_analyzer_dir.mkdir(parents=True, exist_ok=True)
    (concat_analyzer_dir / "marker.txt").write_text("analyzer", encoding="utf-8")

    return well_out_dir, stage_output_root_dir, sorter_output_dir, ks_dir, concat_analyzer_dir


def _bombcell_test_stage_cfg(**overrides):
    base = dict(
        sorter="kilosort4",
        merge_rel_output_root=None,
        bombcell_label_enabled=True,
        bombcell_label_relpath="bombcell_label_outputs",
        bombcell_label_delete_outputs_on_force_restart=True,
        bombcell_label_dry_run=True,
        bombcell_label_thresholds=None,
        bombcell_label_thresholds_path=None,
        bombcell_label_label_non_somatic=True,
        bombcell_label_split_non_somatic_good_mua=True,
        bombcell_label_apply_to_sorter_output=True,
        bombcell_label_write_cluster_group=True,
        bombcell_label_reports_enabled=True,
        bombcell_label_reports_summary_json_enabled=True,
        bombcell_label_reports_summary_json_relpath="bombcell_label_summary.json",
        concat_analyzer_relpath="concat_analyzer",
        preprocess_concat_recording_relpath="preprocess_outputs/preprocessed_recording",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _install_bombcell_test_monkeypatches(*, monkeypatch, spikesort_runner, concat_analyzer_dir):
    class _FakeAnalyzer:
        def has_extension(self, name: str) -> bool:
            return name in {"quality_metrics", "template_metrics"}

        def compute(self, extension_name, **kwargs):
            return None

    monkeypatch.setattr(spikesort_runner, "_import_spikeinterface_full_module", lambda: object())
    monkeypatch.setattr(
        spikesort_runner,
        "_load_concat_analyzer_for_phase",
        lambda **kwargs: (_FakeAnalyzer(), Path(concat_analyzer_dir).resolve()),
    )

    class _FakeLabels:
        def iterrows(self):
            yield 1, {"bombcell_label": "non_soma_good"}
            yield 2, {"bombcell_label": "mua"}

    class _FakeCurationModule:
        @staticmethod
        def bombcell_label_units(
            sorting_analyzer=None,
            thresholds=None,
            label_non_somatic=True,
            split_non_somatic_good_mua=False,
            external_metrics=None,
        ):
            return _FakeLabels()

    original_import_module = spikesort_runner.importlib.import_module

    def _fake_import_module(name: str):
        if name == "spikeinterface.curation":
            return _FakeCurationModule()
        return original_import_module(name)

    monkeypatch.setattr(spikesort_runner.importlib, "import_module", _fake_import_module)


def test_run_bombcell_label_phase_apply_writes_cluster_files(tmp_path: Path, monkeypatch) -> None:
    """Phase with dry_run=False mutates cluster_KSLabel.tsv / cluster_group.tsv."""
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    well_out_dir, stage_output_root_dir, sorter_output_dir, ks_dir, concat_analyzer_dir = (
        _seed_bombcell_test_fixture(tmp_path=tmp_path)
    )
    _install_bombcell_test_monkeypatches(
        monkeypatch=monkeypatch,
        spikesort_runner=spikesort_runner,
        concat_analyzer_dir=concat_analyzer_dir,
    )

    stage_cfg = _bombcell_test_stage_cfg(
        bombcell_label_dry_run=False,
        bombcell_label_reports_summary_json_relpath="reports/custom_bombcell_summary.json",
    )

    report = spikesort_runner._run_bombcell_label_phase(
        well_out_dir=well_out_dir,
        stage_output_root_dir=stage_output_root_dir,
        output_rel_root="spikesort_outputs",
        stage_config=stage_cfg,
        force_restart=False,
        sorter_output_dir=sorter_output_dir,
    )

    assert report.get("status") == "ok"
    assert report.get("n_units_labeled") == 2
    assert report.get("dry_run") is False
    summary_json = Path(str(report.get("summary_json")))
    assert summary_json.name == "custom_bombcell_summary.json"
    assert summary_json.exists()
    assert report.get("outputs", {}).get("bombcell_label.summary_json") == str(summary_json)
    assert str(report.get("analyzer_dir", "")).endswith("concat_analyzer")

    kslabel_text = (ks_dir / "cluster_KSLabel.tsv").read_text(encoding="utf-8")
    group_text = (ks_dir / "cluster_group.tsv").read_text(encoding="utf-8")
    assert "1\tnon_soma_good" in kslabel_text
    assert "2\tmua" in kslabel_text
    assert "3\tgood" in kslabel_text
    assert "1\tnon_soma_good" in group_text


def test_run_bombcell_label_phase_dry_run_preserves_sorter_output(tmp_path: Path, monkeypatch) -> None:
    """Phase with dry_run=true must NOT mutate sorter_output and must write a preview."""
    import hashlib

    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    well_out_dir, stage_output_root_dir, sorter_output_dir, ks_dir, concat_analyzer_dir = (
        _seed_bombcell_test_fixture(tmp_path=tmp_path)
    )
    _install_bombcell_test_monkeypatches(
        monkeypatch=monkeypatch,
        spikesort_runner=spikesort_runner,
        concat_analyzer_dir=concat_analyzer_dir,
    )

    def _hash_dir(root: Path) -> dict[str, str]:
        out: dict[str, str] = {}
        for p in sorted(root.rglob("*")):
            if p.is_file():
                out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
        return out

    pre_hashes = _hash_dir(sorter_output_dir)
    stage_cfg = _bombcell_test_stage_cfg(bombcell_label_dry_run=True)

    report = spikesort_runner._run_bombcell_label_phase(
        well_out_dir=well_out_dir,
        stage_output_root_dir=stage_output_root_dir,
        output_rel_root="spikesort_outputs",
        stage_config=stage_cfg,
        force_restart=False,
        sorter_output_dir=sorter_output_dir,
    )

    assert report.get("status") == "ok"
    assert report.get("dry_run") is True
    post_hashes = _hash_dir(sorter_output_dir)
    assert post_hashes == pre_hashes, "dry_run must not mutate sorter_output"

    bombcell_out_dir = Path(str(report.get("out_dir")))
    preview_kslabel = bombcell_out_dir / "dry_run" / "proposed_cluster_KSLabel.tsv"
    preview_group = bombcell_out_dir / "dry_run" / "proposed_cluster_group.tsv"
    assert preview_kslabel.exists()
    assert preview_group.exists()
    preview_text = preview_kslabel.read_text(encoding="utf-8")
    assert "1\tnon_soma_good" in preview_text
    assert "2\tmua" in preview_text
    outputs = dict(report.get("outputs", {}))
    assert outputs.get("bombcell_label.dry_run_cluster_kslabel_tsv") == str(preview_kslabel)
    assert outputs.get("bombcell_label.dry_run_cluster_group_tsv") == str(preview_group)


def test_run_bombcell_label_phase_requires_concat_analyzer(tmp_path: Path, monkeypatch) -> None:
    """When concat_analyzer dir is absent, the phase must fail with a clear error."""
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    well_out_dir, stage_output_root_dir, sorter_output_dir, _ks_dir, concat_analyzer_dir = (
        _seed_bombcell_test_fixture(tmp_path=tmp_path)
    )
    # Remove the analyzer dir so the loader raises.
    import shutil

    shutil.rmtree(concat_analyzer_dir)

    monkeypatch.setattr(
        spikesort_runner, "_import_spikeinterface_full_module", lambda: object()
    )
    stage_cfg = _bombcell_test_stage_cfg(
        bombcell_label_dry_run=True,
        bombcell_label_fail_on_error=True,
    )

    report = spikesort_runner._run_bombcell_label_phase(
        well_out_dir=well_out_dir,
        stage_output_root_dir=stage_output_root_dir,
        output_rel_root="spikesort_outputs",
        stage_config=stage_cfg,
        force_restart=False,
        sorter_output_dir=sorter_output_dir,
    )

    assert report.get("status") == "error"
    assert "concat_analyzer" in str(report.get("error", ""))


def test_run_spikesort_merge_stage_does_not_invoke_bombcell_when_enabled(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    h5_path = tmp_path / "raw_data" / "input.raw.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"")

    well_out_dir = tmp_path / "well001"
    stage_output_root_dir = well_out_dir / "spikesort_outputs"
    sorter_output_dir = stage_output_root_dir / "sorter_output" / "sorter_output"
    sorter_output_dir.mkdir(parents=True, exist_ok=True)
    (sorter_output_dir / "params.py").write_text("n_channels_dat=4\n", encoding="utf-8")

    monkeypatch.setattr(
        spikesort_runner,
        "_run_bombcell_label_phase",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("merge stage should not invoke bombcell")),
    )
    monkeypatch.setattr(spikesort_runner, "compute_mea_analysis_output_dir", lambda **kwargs: well_out_dir)
    monkeypatch.setattr(spikesort_runner, "_resolve_sorter_output_dir", lambda **kwargs: sorter_output_dir)
    monkeypatch.setattr(spikesort_runner, "_import_spikeinterface_full_module", lambda: object())
    monkeypatch.setattr(
        spikesort_runner,
        "_load_concat_analyzer_for_phase",
        lambda **kwargs: (object(), stage_output_root_dir / "concat_analyzer"),
    )
    monkeypatch.setattr(spikesort_runner, "_release_loaded_analyzer_extensions", lambda **kwargs: [])
    monkeypatch.setattr(
        spikesort_runner,
        "_run_slay_merge_method",
        lambda **kwargs: {
            "name": "slay",
            "status": "skipped",
            "reason": "slay_disabled",
            "out_dir": str(stage_output_root_dir / "SLAy_outputs"),
            "summary_json": None,
            "outputs": {},
            "ks_dir": str(sorter_output_dir),
            "applied_merges": False,
        },
    )

    stage_cfg = SimpleNamespace(
        merge_units_enabled=True,
        merge_sequence=("SLAy",),
        slay_enabled=False,
        merge_reports_enabled=False,
        cache_sorting_outputs_before_merge=False,
        bombcell_label_enabled=True,
        bombcell_label_fail_on_error=True,
    )

    result = run_spikesort_merge_stage(
        h5_path=h5_path,
        stream_id="well001",
        mea_output_root=tmp_path,
        output_rel_root="spikesort_outputs",
        stage_config=stage_cfg,
        force_restart=False,
        force_replot=False,
    )

    summary = _read_json(result.summary_json)
    assert summary.get("bombcell_label", {}).get("status") == "skipped"
    assert summary.get("bombcell_label", {}).get("reason") == "bombcell_label_not_invoked_by_merge_stage"
