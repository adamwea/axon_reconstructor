from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import axon_recon.pipeline.cli as pipeline_cli

ACTIVE_STAGE_ORDER = ["preprocess", "spikesort", "reconstruct", "analysis"]


def _write_runtime_cfg(path: Path) -> None:
    path.write_text("{}\n", encoding="utf-8")


def test_parse_stage_list_tokens_supports_aliases_and_comma_spacing() -> None:
    parsed = pipeline_cli._parse_stage_list_tokens(["preproc,", "sort"])
    assert parsed == ["preprocess", "spikesort"]


def test_parse_stage_list_tokens_supports_all_keyword() -> None:
    parsed = pipeline_cli._parse_stage_list_tokens(["all"])
    assert parsed == ACTIVE_STAGE_ORDER


def test_canonical_all_selector_excludes_retired_stages() -> None:
    assert list(pipeline_cli._CANONICAL_STAGE_ORDER) == ACTIVE_STAGE_ORDER
    assert "templates" not in pipeline_cli._CANONICAL_STAGE_ORDER


@pytest.mark.parametrize("raw_token", ["analyse", "analyze"])
def test_parse_stage_list_tokens_rejects_misspelled_analysis_selector(raw_token: str) -> None:
    with pytest.raises(SystemExit, match="Unsupported stage token"):
        pipeline_cli._parse_stage_list_tokens([raw_token])


@pytest.mark.parametrize(
    ("raw_token", "expected"),
    [
        ("analysis", "analysis"),
        ("analysis.compute_metrics", "analysis.compute_metrics"),
        ("analysis.metrics", "analysis.compute_metrics"),
        ("analysis.compute", "analysis.compute_metrics"),
        ("metrics", "analysis.compute_metrics"),
        ("compute_metrics", "analysis.compute_metrics"),
    ],
)
def test_parse_stage_list_tokens_supports_analysis_aliases(raw_token: str, expected: str) -> None:
    parsed = pipeline_cli._parse_stage_list_tokens([raw_token])
    assert parsed == [expected]


@pytest.mark.parametrize(
    "raw_token",
    [
        "templates",
        "template",
        "templates.resolve_sources",
        "templates.analyzers",
        "templates.per_unit_processing.extract_template_segments",
        "templates.build_templates",
        "templates.compute_template_similarity",
        "templates.plot_templates",
        "templates.report_templates",
        "templates.reports",
        "template.reports",
    ],
)
def test_parse_stage_list_tokens_rejects_retired_templates_selector(raw_token: str) -> None:
    with pytest.raises(SystemExit, match="Unsupported stage token"):
        pipeline_cli._parse_stage_list_tokens([raw_token])


@pytest.mark.parametrize(
    "raw_token",
    [
        "reconstruct.extract_template_segments",
        "reconstruct.templates_extract_template_segments",
        "recon.extract_template_segments",
        "recon.templates_extract_template_segments",
        "reconstruction.extract_template_segments",
        "reconstruction.templates_extract_template_segments",
    ],
)
def test_parse_stage_list_tokens_rejects_removed_extract_template_segments(raw_token: str) -> None:
    with pytest.raises(SystemExit, match="Unsupported stage token"):
        pipeline_cli._parse_stage_list_tokens([raw_token])


@pytest.mark.parametrize(
    ("raw_token", "expected"),
    [
        ("preprocess.copy_src_to_scratch", "preprocess.copy_src_to_scratch"),
        ("preprocess.save_rec_metadata", "preprocess.save_rec_metadata"),
        ("preprocess.prepare_raw_binaries", "preprocess.prepare_raw_binaries"),
        ("preprocess.wipe_src_scratch", "preprocess.wipe_src_scratch"),
        ("preprocess.preprocess_segments", "preprocess.preprocess_segments"),
        ("preprocess.plot_segment_traces", "preprocess.plot_segment_traces"),
        ("preprocess.plot_segment_channel_layouts", "preprocess.plot_segment_channel_layouts"),
        ("preprocess.concat_segments", "preprocess.concat_segments"),
        ("preprocess.plot_concat_traces", "preprocess.plot_concat_traces"),
        ("preprocess.plot_concat_channel_layout", "preprocess.plot_concat_channel_layout"),
        ("preprocess.plot_raster_threshold", "preprocess.plot_raster_threshold"),
        ("preproc.save_rec_metadata", "preprocess.save_rec_metadata"),
        ("preproc.prepare_raw_binaries", "preprocess.prepare_raw_binaries"),
        ("preproc.wipe_src_scratch", "preprocess.wipe_src_scratch"),
        ("preproc.plot_segment_traces", "preprocess.plot_segment_traces"),
        ("preproc.plot_segment_channel_layouts", "preprocess.plot_segment_channel_layouts"),
        ("preproc.concat_segments", "preprocess.concat_segments"),
        ("preproc.plot_concat_traces", "preprocess.plot_concat_traces"),
        ("preproc.plot_concat_channel_layout", "preprocess.plot_concat_channel_layout"),
        ("preproc.plot_raster_threshold", "preprocess.plot_raster_threshold"),
    ],
)
def test_parse_stage_list_tokens_supports_preprocess_phase_tokens(raw_token: str, expected: str) -> None:
    parsed = pipeline_cli._parse_stage_list_tokens([raw_token])
    assert parsed == [expected]


@pytest.mark.parametrize(
    "raw_token",
    [
        "preprocess.concatenate_recordings",
        "preprocess.concatenate_preprocessed_recordings",
        "preprocess.save_common_electrodes",
        "pre.build_preprocessed_recording",
        "pre.save_concatenated_recording",
        "pre.save_segment_recordings",
    ],
)
def test_parse_stage_list_tokens_rejects_removed_preprocess_aliases(raw_token: str) -> None:
    with pytest.raises(SystemExit, match="Unsupported stage token"):
        pipeline_cli._parse_stage_list_tokens([raw_token])


@pytest.mark.parametrize(
    "raw_token",
    [
        "templates.per_unit_processing.build_templates",
        "templates.per_unit_processing.plot_templates",
        "template.compute_template_similarity",
        "template.report_templates",
    ],
)
def test_parse_stage_list_tokens_rejects_legacy_templates_aliases(raw_token: str) -> None:
    with pytest.raises(SystemExit, match="Unsupported stage token"):
        pipeline_cli._parse_stage_list_tokens([raw_token])


@pytest.mark.parametrize(
    ("raw_token", "expected"),
    [
        ("reconstruct.generate_gtrs", "reconstruct.generate_gtrs"),
        ("reconstruct.plot_recons", "reconstruct.plot_recons"),
        ("reconstruct.plot_branch_propagations", "reconstruct.plot_branch_propagations"),
        ("reconstruct.plot_branch_velocities", "reconstruct.plot_branch_velocities"),
        ("reconstruct.plot_unit_summary", "reconstruct.plot_unit_summary"),
        ("reconstruct.report_recons", "reconstruct.report_recons"),
        ("reconstruct.report_recon_grid", "reconstruct.report_recon_grid"),
        ("reconstruct.report_full_chip_layout", "reconstruct.report_full_chip_layout"),
        ("reconstruct.analyzers", "reconstruct.analyzers"),
        ("reconstruct.build_templates", "reconstruct.build_templates"),
        ("reconstruct.compute_template_similarity", "reconstruct.compute_template_similarity"),
        ("reconstruct.plot_templates", "reconstruct.plot_templates"),
        ("reconstruct.plot_templates_v2", "reconstruct.plot_templates_v2"),
        ("reconstruct.report_templates", "reconstruct.report_templates"),
        ("recon.generate_gtrs", "reconstruct.generate_gtrs"),
        ("recon.analyzers", "reconstruct.analyzers"),
        ("recon.templates_analyzers", "reconstruct.analyzers"),
        ("reconstruct.templates_build_templates", "reconstruct.build_templates"),
        ("reconstruction.plot_recons", "reconstruct.plot_recons"),
        ("reconstruction.build_templates", "reconstruct.build_templates"),
        ("reconstruction.templates_plot_templates", "reconstruct.plot_templates"),
        ("reconstruction.templates_plot_templates_v2", "reconstruct.plot_templates_v2"),
        ("recon.plot_branch_propagations", "reconstruct.plot_branch_propagations"),
        ("reconstruction.plot_branch_velocities", "reconstruct.plot_branch_velocities"),
        ("reconstruction.plot_unit_summary", "reconstruct.plot_unit_summary"),
        ("reconstruction.report_recons", "reconstruct.report_recons"),
        ("recon.report_recon_grid", "reconstruct.report_recon_grid"),
        ("reconstruction.report_recon_grid", "reconstruct.report_recon_grid"),
    ],
)
def test_parse_stage_list_tokens_supports_reconstruct_phase_tokens(raw_token: str, expected: str) -> None:
    parsed = pipeline_cli._parse_stage_list_tokens([raw_token])
    assert parsed == [expected]


def test_parse_stage_list_tokens_supports_spikesort_sort_substage_alias() -> None:
    parsed = pipeline_cli._parse_stage_list_tokens(["spikesort.sort"])
    assert parsed == ["spikesort.sort"]


@pytest.mark.parametrize(
    ("raw_token", "expected"),
    [
        ("bootstrap_concat_binary", "spikesort.bootstrap_concat_binary"),
        ("spikesort.bootstrap_concat_binary", "spikesort.bootstrap_concat_binary"),
        ("cleanup_concat_binary", "spikesort.cleanup_concat_binary"),
        ("clear_concat_binary", "spikesort.cleanup_concat_binary"),
        ("spikesort.clear_concat_binary", "spikesort.cleanup_concat_binary"),
    ],
)
def test_parse_stage_list_tokens_supports_spikesort_concat_binary_phase_tokens(raw_token: str, expected: str) -> None:
    parsed = pipeline_cli._parse_stage_list_tokens([raw_token])
    assert parsed == [expected]


@pytest.mark.parametrize(
    ("raw_token", "expected"),
    [
        ("cleanup_analyzers", "spikesort.cleanup_analyzers"),
        ("cleanup_analyzer", "spikesort.cleanup_analyzers"),
        ("clear_analyzers", "spikesort.cleanup_analyzers"),
        ("clear_analyzer", "spikesort.cleanup_analyzers"),
        ("spikesort.cleanup_analyzers", "spikesort.cleanup_analyzers"),
        ("spikesort.cleanup_analyzer", "spikesort.cleanup_analyzers"),
        ("spikesort.clear_analyzers", "spikesort.cleanup_analyzers"),
        ("spikesort.clear_analyzer", "spikesort.cleanup_analyzers"),
        ("spikesort.cleanup_concat_analyzer", "spikesort.cleanup_analyzers"),
    ],
)
def test_parse_stage_list_tokens_supports_spikesort_cleanup_analyzers_phase_tokens(raw_token: str, expected: str) -> None:
    parsed = pipeline_cli._parse_stage_list_tokens([raw_token])
    assert parsed == [expected]


def test_parse_stage_list_tokens_supports_spikesort_merge_substage() -> None:
    parsed = pipeline_cli._parse_stage_list_tokens(["spikesort.merge"])
    assert parsed == ["spikesort.merge"]


def test_parse_stage_list_tokens_supports_spikesort_merge_slay_substage() -> None:
    parsed = pipeline_cli._parse_stage_list_tokens(["spikesort.merge.slay"])
    assert parsed == ["spikesort.merge_SLAy"]


def test_parse_stage_list_tokens_supports_spikesort_merge_slay_phase_name() -> None:
    parsed = pipeline_cli._parse_stage_list_tokens(["spikesort.merge_SLAy"])
    assert parsed == ["spikesort.merge_SLAy"]


def test_build_parser_supports_stage_command_alias() -> None:
    parser = pipeline_cli.build_parser()
    args = parser.parse_args([
        "stage",
        "preproc,",
        "sort",
        "--config",
        "/tmp/runtime.yml",
        "--force-restart",
    ])

    assert args.command == "stage"
    assert args.stages == ["preproc,", "sort"]
    assert args.force_restart is True


def test_build_parser_supports_phase_tune_flags() -> None:
    parser = pipeline_cli.build_parser()
    args = parser.parse_args(
        [
            "stages",
            "reconstruct.plot_templates",
            "--config",
            "/tmp/runtime.yml",
            "--phase-tune",
            "--confirm-full-scope",
        ]
    )

    assert args.phase_tune is True
    assert args.confirm_full_scope is True


def test_build_parser_supports_alloc_flag() -> None:
    parser = pipeline_cli.build_parser()
    args = parser.parse_args([
        "stages",
        "reconstruct.report_templates",
        "--config",
        "/tmp/runtime.yml",
        "--alloc",
    ])

    assert args.alloc is True


def test_run_stage_sequence_alloc_prints_preview_without_running_handlers(monkeypatch) -> None:
    preview_calls: list[dict[str, object]] = []

    def fake_preview(**kwargs: object) -> None:
        preview_calls.append(kwargs)

    def fail_handler(_args: argparse.Namespace) -> int:
        raise AssertionError("stage handler should not run for --alloc")

    monkeypatch.setattr(pipeline_cli, "print_stage_allocation_preview", fake_preview)
    monkeypatch.setitem(pipeline_cli._STAGE_HANDLERS, "preprocess", fail_handler)

    args = argparse.Namespace(
        stages=["preprocess"],
        config="default.runtime.yml",
        alloc=True,
        target_datasets=["0,2", "8"],
        unit_id=None,
        unit_ids=None,
        limit_units=None,
        limit_segments=2,
        limit_datasets=None,
        limit_wells_per_dataset=1,
        force_restart=False,
        force_replot=False,
        phase_tune=False,
        confirm_full_scope=False,
    )

    rc = pipeline_cli._run_stage_sequence_from_args(args)

    assert rc == 0
    assert len(preview_calls) == 1
    assert preview_calls[0]["stages"] == ["preprocess"]
    assert preview_calls[0]["target_datasets_override"] == [0, 2, 8]
    assert preview_calls[0]["limit_segments_override"] == 2
    assert preview_calls[0]["limit_wells_per_dataset_override"] == 1


def test_run_stage_sequence_alloc_skips_non_root_rank_for_mpi_override(monkeypatch) -> None:
    preview_calls: list[dict[str, object]] = []

    def fake_preview(**kwargs: object) -> None:
        preview_calls.append(kwargs)

    class _Ctx:
        rank = 1
        size = 2

    monkeypatch.setattr(pipeline_cli, "print_stage_allocation_preview", fake_preview)
    monkeypatch.setattr(pipeline_cli, "current_mpi_context", lambda: _Ctx())

    args = argparse.Namespace(
        stages=["preprocess"],
        config="default.runtime.yml",
        alloc=True,
        target_datasets=["11,12"],
        unit_id=None,
        unit_ids=None,
        limit_units=None,
        limit_segments=2,
        limit_datasets=None,
        limit_wells_per_dataset=1,
        force_restart=False,
        force_replot=False,
        phase_tune=False,
        confirm_full_scope=False,
        task_allocation_backend="mpi",
        task_allocation_tasks_per_node=None,
        task_allocation_cpus_per_task=None,
        task_allocation_bind=None,
        task_allocation_use_hyperthreads=None,
        task_allocation_reserve_cpus=None,
    )

    rc = pipeline_cli._run_stage_sequence_from_args(args)

    assert rc == 0
    assert len(preview_calls) == 0


def test_build_parser_supports_systopo_command() -> None:
    parser = pipeline_cli.build_parser()
    args = parser.parse_args(["systopo"])

    assert args.command == "systopo"
    assert args.handler is pipeline_cli._run_system_topology_from_args


def test_build_parser_supports_target_datasets_flag() -> None:
    parser = pipeline_cli.build_parser()
    args = parser.parse_args(
        [
            "stages",
            "reconstruct.build_templates",
            "--config",
            "/tmp/runtime.yml",
            "--target-datasets",
            "0,",
            "2,",
            "8",
        ]
    )

    assert args.target_datasets == ["0,", "2,", "8"]


def test_build_parser_supports_singular_target_dataset_alias() -> None:
    parser = pipeline_cli.build_parser()
    args = parser.parse_args(
        [
            "stages",
            "reconstruct.build_templates",
            "--config",
            "/tmp/runtime.yml",
            "--target-dataset",
            "12",
        ]
    )

    assert args.target_datasets == ["12"]


def test_build_parser_supports_reconstruct_subparser_dataset_limit_flags() -> None:
    from axon_recon.pipeline.stages.reconstruct import cli as reconstruct_cli

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    reconstruct_cli.register_reconstruct_subparser(subparsers)
    args = parser.parse_args(
        [
            "reconstruct",
            "--config",
            "/tmp/runtime.yml",
            "--limit-segments",
            "2",
            "--limit-datasets",
            "1",
            "--target-dataset",
            "0,",
            "2,",
            "8",
            "--limit-wells-per-dataset",
            "1",
            "--limit-units",
            "3",
        ]
    )

    assert args.limit_segments == 2
    assert args.limit_datasets == 1
    assert args.target_datasets == ["0,", "2,", "8"]
    assert args.limit_wells_per_dataset == 1
    assert args.limit_units == 3


@pytest.mark.parametrize(
    ("command", "register_module"),
    [
        ("preprocess", "axon_recon.pipeline.stages.preprocess.cli"),
        ("spikesort", "axon_recon.pipeline.stages.spikesort.cli"),
    ],
)
def test_build_parser_supports_target_datasets_for_other_stage_subparsers(
    command: str,
    register_module: str,
) -> None:
    module = __import__(register_module, fromlist=["register_preprocess_subparser", "register_spikesort_subparser"])
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    if command == "preprocess":
        module.register_preprocess_subparser(subparsers)
    else:
        module.register_spikesort_subparser(subparsers)

    args = parser.parse_args(
        [
            command,
            "--config",
            "/tmp/runtime.yml",
            "--target-dataset",
            "1,",
            "3",
        ]
    )

    assert args.target_datasets == ["1,", "3"]


def test_reconstruct_cli_runtime_kwargs_include_dataset_and_well_limits() -> None:
    from axon_recon.pipeline.stages.reconstruct import cli as reconstruct_cli

    args = argparse.Namespace(
        config="/tmp/runtime.yml",
        unit_id=None,
        unit_ids=None,
        limit_units=5,
        limit_segments=2,
        limit_datasets=1,
        target_datasets=["0,", "2,", "8"],
        limit_wells_per_dataset=1,
        force_restart=False,
        force_replot=False,
    )

    kwargs = reconstruct_cli._reconstruct_runtime_kwargs(args)

    assert kwargs["limit_segments_override"] == 2
    assert kwargs["limit_datasets_override"] == 1
    assert kwargs["target_datasets_override"] == [0, 2, 8]
    assert kwargs["limit_wells_per_dataset_override"] == 1
    assert kwargs["unit_limit_override"] == 5


def test_main_runs_selected_stages_in_order(monkeypatch, tmp_path: Path) -> None:
    runtime_cfg = tmp_path / "runtime.yml"
    _write_runtime_cfg(runtime_cfg)

    calls: list[tuple[str, str, bool]] = []

    def _mk_handler(stage_name: str):
        def _handler(args):
            calls.append((stage_name, str(getattr(args, "stage", "")), bool(getattr(args, "force_restart", False))))
            return 0

        return _handler

    monkeypatch.setitem(pipeline_cli._STAGE_HANDLERS, "preprocess", _mk_handler("preprocess"))
    monkeypatch.setitem(pipeline_cli._STAGE_HANDLERS, "spikesort", _mk_handler("spikesort"))

    rc = pipeline_cli.main(
        [
            "stages",
            "preproc,",
            "sort",
            "--config",
            str(runtime_cfg),
            "--force-restart",
        ]
    )

    assert rc == 0
    assert calls == [
        ("preprocess", "preprocess", True),
        ("spikesort", "spikesort", True),
    ]


def test_main_runs_systopo_command(monkeypatch, capsys) -> None:
    fake_topology = object()

    monkeypatch.setattr(pipeline_cli, "detect_cpu_topology", lambda: fake_topology)
    monkeypatch.setattr(
        pipeline_cli,
        "format_cpu_topology",
        lambda topology: "CPU topology\nsource: sysfs\nvisible_cpus: 0-3",
    )

    rc = pipeline_cli.main(["systopo"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "CPU topology" in captured.out
    assert "source: sysfs" in captured.out
    assert "visible_cpus: 0-3" in captured.out


def test_phase_tune_rejects_unlimited_scope_before_running_stage(monkeypatch, tmp_path: Path) -> None:
    runtime_cfg = tmp_path / "runtime.yml"
    _write_runtime_cfg(runtime_cfg)
    calls: list[str] = []

    def _handler(args):
        calls.append(str(getattr(args, "stage", "")))
        return 0

    monkeypatch.setitem(pipeline_cli._STAGE_HANDLERS, "preprocess", _handler)

    rc = pipeline_cli.main(["stages", "preprocess", "--config", str(runtime_cfg), "--phase-tune"])

    assert rc == 2
    assert calls == []


def test_stage_sequence_allows_target_datasets_for_preprocess_and_spikesort(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime_cfg = tmp_path / "runtime.yml"
    _write_runtime_cfg(runtime_cfg)
    calls: list[tuple[str, list[str]]] = []

    def _mk_handler(stage_name: str):
        def _handler(args):
            calls.append((stage_name, list(getattr(args, "target_datasets", []) or [])))
            return 0

        return _handler

    monkeypatch.setitem(pipeline_cli._STAGE_HANDLERS, "preprocess", _mk_handler("preprocess"))
    monkeypatch.setitem(pipeline_cli._STAGE_HANDLERS, "spikesort", _mk_handler("spikesort"))

    rc = pipeline_cli.main(
        [
            "stages",
            "preprocess",
            "spikesort",
            "--config",
            str(runtime_cfg),
            "--target-datasets",
            "1",
        ]
    )

    assert rc == 0
    assert calls == [("preprocess", ["1"]), ("spikesort", ["1"])]


def test_phase_tune_runs_stage_then_emits_recommendations(monkeypatch, tmp_path: Path) -> None:
    from axon_recon.pipeline import phase_tuning

    runtime_cfg = tmp_path / "runtime.yml"
    _write_runtime_cfg(runtime_cfg)
    calls: list[str] = []
    emitted: list[tuple[str, tuple[str, ...]]] = []

    def _handler(args):
        calls.append(str(getattr(args, "stage", "")))
        return 0

    def _fake_emit_phase_tuning_recommendations(*, config_path: str, selected_stages):
        emitted.append((str(config_path), tuple(selected_stages)))
        return {"summary": {}, "paths": {}}

    monkeypatch.setitem(pipeline_cli._STAGE_HANDLERS, "preprocess", _handler)
    monkeypatch.setattr(phase_tuning, "emit_phase_tuning_recommendations", _fake_emit_phase_tuning_recommendations)

    rc = pipeline_cli.main(
        [
            "stages",
            "preprocess",
            "--config",
            str(runtime_cfg),
            "--phase-tune",
            "--limit-datasets",
            "1",
        ]
    )

    assert rc == 0
    assert calls == ["preprocess"]
    assert emitted == [(str(runtime_cfg), ("preprocess",))]


def test_main_runs_mixed_stage_and_reconstruct_phase_selector(monkeypatch, tmp_path: Path) -> None:
    runtime_cfg = tmp_path / "runtime.yml"
    _write_runtime_cfg(runtime_cfg)

    calls: list[str] = []

    def _mk_handler(stage_name: str):
        def _handler(args):
            calls.append(str(getattr(args, "stage", stage_name)))
            return 0

        return _handler

    monkeypatch.setitem(pipeline_cli._STAGE_HANDLERS, "spikesort", _mk_handler("spikesort"))
    monkeypatch.setitem(
        pipeline_cli._STAGE_HANDLERS,
        "reconstruct.analyzers",
        _mk_handler("reconstruct.analyzers"),
    )

    rc = pipeline_cli.main(["stages", "spikesort", "reconstruct.analyzers", "--config", str(runtime_cfg)])

    assert rc == 0
    assert calls == ["spikesort", "reconstruct.analyzers"]


def test_main_stage_alias_supports_all(monkeypatch, tmp_path: Path) -> None:
    runtime_cfg = tmp_path / "runtime.yml"
    _write_runtime_cfg(runtime_cfg)

    calls: list[str] = []

    for stage_name in pipeline_cli._CANONICAL_STAGE_ORDER:
        def _handler(args, _stage=stage_name):
            calls.append(_stage)
            return 0

        monkeypatch.setitem(pipeline_cli._STAGE_HANDLERS, stage_name, _handler)

    rc = pipeline_cli.main(["stage", "all", "--config", str(runtime_cfg), "--force-restart"])

    assert rc == 0
    assert calls == ACTIVE_STAGE_ORDER


def test_main_stops_after_first_failure(monkeypatch, tmp_path: Path) -> None:
    runtime_cfg = tmp_path / "runtime.yml"
    _write_runtime_cfg(runtime_cfg)

    calls: list[str] = []

    def _preprocess(args):
        calls.append("preprocess")
        return 0

    def _spikesort(args):
        calls.append("spikesort")
        return 3

    def _reconstruct(args):
        calls.append("reconstruct")
        return 0

    monkeypatch.setitem(pipeline_cli._STAGE_HANDLERS, "preprocess", _preprocess)
    monkeypatch.setitem(pipeline_cli._STAGE_HANDLERS, "spikesort", _spikesort)
    monkeypatch.setitem(pipeline_cli._STAGE_HANDLERS, "reconstruct", _reconstruct)

    rc = pipeline_cli.main(["stages", "preprocess,spikesort,reconstruct", "--config", str(runtime_cfg)])

    assert rc == 3
    assert calls == ["preprocess", "spikesort"]


@pytest.mark.parametrize(
    ("stage_token", "handler_key"),
    [
        ("preprocess.copy_src_to_scratch", "preprocess.copy_src_to_scratch"),
        ("preprocess.save_rec_metadata", "preprocess.save_rec_metadata"),
        ("preprocess.wipe_src_scratch", "preprocess.wipe_src_scratch"),
        ("preprocess.preprocess_segments", "preprocess.preprocess_segments"),
        ("preprocess.plot_segment_traces", "preprocess.plot_segment_traces"),
        ("preprocess.concat_segments", "preprocess.concat_segments"),
        ("preprocess.plot_concat_traces", "preprocess.plot_concat_traces"),
        ("preprocess.plot_raster_threshold", "preprocess.plot_raster_threshold"),
        ("preproc.save_rec_metadata", "preprocess.save_rec_metadata"),
        ("preproc.wipe_src_scratch", "preprocess.wipe_src_scratch"),
        ("preproc.plot_segment_traces", "preprocess.plot_segment_traces"),
        ("preproc.concat_segments", "preprocess.concat_segments"),
        ("preproc.plot_concat_traces", "preprocess.plot_concat_traces"),
        ("preproc.plot_raster_threshold", "preprocess.plot_raster_threshold"),
    ],
)
def test_main_runs_preprocess_phase_substages(
    monkeypatch,
    tmp_path: Path,
    stage_token: str,
    handler_key: str,
) -> None:
    runtime_cfg = tmp_path / "runtime.yml"
    _write_runtime_cfg(runtime_cfg)

    calls: list[str] = []

    def _handler(args):
        calls.append(str(getattr(args, "stage", "")))
        return 0

    monkeypatch.setitem(pipeline_cli._STAGE_HANDLERS, handler_key, _handler)

    rc = pipeline_cli.main(["stages", stage_token, "--config", str(runtime_cfg)])

    assert rc == 0
    assert calls == [handler_key]


@pytest.mark.parametrize(
    ("stage_token", "handler_key"),
    [
        ("reconstruct.generate_gtrs", "reconstruct.generate_gtrs"),
        ("reconstruct.plot_recons", "reconstruct.plot_recons"),
        ("reconstruct.plot_branch_propagations", "reconstruct.plot_branch_propagations"),
        ("reconstruct.plot_branch_velocities", "reconstruct.plot_branch_velocities"),
        ("reconstruct.plot_unit_summary", "reconstruct.plot_unit_summary"),
        ("reconstruct.report_recons", "reconstruct.report_recons"),
        ("reconstruct.report_recon_grid", "reconstruct.report_recon_grid"),
        ("reconstruct.report_full_chip_layout", "reconstruct.report_full_chip_layout"),
    ],
)
def test_main_runs_reconstruct_phase_substages(
    monkeypatch,
    tmp_path: Path,
    stage_token: str,
    handler_key: str,
) -> None:
    runtime_cfg = tmp_path / "runtime.yml"
    _write_runtime_cfg(runtime_cfg)

    calls: list[str] = []

    def _handler(args):
        calls.append(str(getattr(args, "stage", "")))
        return 0

    monkeypatch.setitem(pipeline_cli._STAGE_HANDLERS, handler_key, _handler)

    rc = pipeline_cli.main(["stages", stage_token, "--config", str(runtime_cfg)])

    assert rc == 0
    assert calls == [handler_key]


def test_stage_sequence_parser_accepts_debug_limit_flags(monkeypatch, tmp_path: Path) -> None:
    runtime_cfg = tmp_path / "runtime.yml"
    _write_runtime_cfg(runtime_cfg)

    seen: dict[str, int | None] = {}

    def _reconstruct(args):
        seen["limit_segments"] = getattr(args, "limit_segments", None)
        seen["limit_datasets"] = getattr(args, "limit_datasets", None)
        seen["limit_wells_per_dataset"] = getattr(args, "limit_wells_per_dataset", None)
        seen["limit_units"] = getattr(args, "limit_units", None)
        return 0

    monkeypatch.setitem(pipeline_cli._STAGE_HANDLERS, "reconstruct", _reconstruct)

    rc = pipeline_cli.main(
        [
            "stages",
            "reconstruct",
            "--config",
            str(runtime_cfg),
            "--limit-segments",
            "2",
            "--limit-datasets",
            "4",
            "--limit-wells-per-dataset",
            "1",
            "--limit-units",
            "3",
        ]
    )

    assert rc == 0
    assert seen == {
        "limit_segments": 2,
        "limit_datasets": 4,
        "limit_wells_per_dataset": 1,
        "limit_units": 3,
    }


def test_stage_sequence_parser_accepts_limit_wells_alias(monkeypatch, tmp_path: Path) -> None:
    runtime_cfg = tmp_path / "runtime.yml"
    _write_runtime_cfg(runtime_cfg)

    seen: dict[str, int | None] = {}

    def _preprocess_phase(args):
        seen["limit_wells_per_dataset"] = getattr(args, "limit_wells_per_dataset", None)
        return 0

    monkeypatch.setitem(pipeline_cli._STAGE_HANDLERS, "preprocess.preprocess_segments", _preprocess_phase)

    rc = pipeline_cli.main(
        [
            "stages",
            "preprocess.preprocess_segments",
            "--config",
            str(runtime_cfg),
            "--limit-wells",
            "2",
        ]
    )

    assert rc == 0
    assert seen == {"limit_wells_per_dataset": 2}


def test_main_runs_spikesort_sort_substage_alias(monkeypatch, tmp_path: Path) -> None:
    runtime_cfg = tmp_path / "runtime.yml"
    _write_runtime_cfg(runtime_cfg)

    calls: list[str] = []

    def _spikesort_sort(args):
        calls.append(str(getattr(args, "stage", "")))
        return 0

    monkeypatch.setitem(pipeline_cli._STAGE_HANDLERS, "spikesort.sort", _spikesort_sort)

    rc = pipeline_cli.main(["stages", "spikesort.sort", "--config", str(runtime_cfg), "--force-restart"])

    assert rc == 0
    assert calls == ["spikesort.sort"]


def test_main_runs_spikesort_summarize_sort_substage(monkeypatch, tmp_path: Path) -> None:
    runtime_cfg = tmp_path / "runtime.yml"
    _write_runtime_cfg(runtime_cfg)

    calls: list[str] = []

    def _spikesort_summarize_sort(args):
        calls.append(str(getattr(args, "stage", "")))
        return 0

    monkeypatch.setitem(pipeline_cli._STAGE_HANDLERS, "spikesort.summarize_sort", _spikesort_summarize_sort)

    rc = pipeline_cli.main(["stages", "spikesort.summarize_sort", "--config", str(runtime_cfg), "--force-restart"])

    assert rc == 0
    assert calls == ["spikesort.summarize_sort"]


def test_main_runs_spikesort_merge_substage(monkeypatch, tmp_path: Path) -> None:
    runtime_cfg = tmp_path / "runtime.yml"
    _write_runtime_cfg(runtime_cfg)

    calls: list[str] = []

    def _spikesort_merge(args):
        calls.append(str(getattr(args, "stage", "")))
        return 0

    monkeypatch.setitem(pipeline_cli._STAGE_HANDLERS, "spikesort.merge", _spikesort_merge)

    rc = pipeline_cli.main(["stages", "spikesort.merge", "--config", str(runtime_cfg), "--force-restart"])

    assert rc == 0
    assert calls == ["spikesort.merge"]


def test_main_runs_spikesort_merge_slay_substage(monkeypatch, tmp_path: Path) -> None:
    runtime_cfg = tmp_path / "runtime.yml"
    _write_runtime_cfg(runtime_cfg)

    calls: list[str] = []

    def _spikesort_merge_slay(args):
        calls.append(str(getattr(args, "stage", "")))
        return 0

    monkeypatch.setitem(pipeline_cli._STAGE_HANDLERS, "spikesort.merge_SLAy", _spikesort_merge_slay)

    rc = pipeline_cli.main(["stages", "spikesort.merge.slay", "--config", str(runtime_cfg), "--force-restart"])

    assert rc == 0
    assert calls == ["spikesort.merge_SLAy"]


def test_parse_stage_list_tokens_rejects_unknown_stage() -> None:
    with pytest.raises(SystemExit):
        pipeline_cli._parse_stage_list_tokens(["bogus_stage"])
