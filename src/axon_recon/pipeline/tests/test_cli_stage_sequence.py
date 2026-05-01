from __future__ import annotations

from pathlib import Path

import pytest

import axon_recon.pipeline.cli as pipeline_cli

ACTIVE_STAGE_ORDER = ["preprocess", "spikesort", "reconstruct"]


def _write_runtime_cfg(path: Path) -> None:
    path.write_text("global_logger: {}\n", encoding="utf-8")


def test_parse_stage_list_tokens_supports_aliases_and_comma_spacing() -> None:
    parsed = pipeline_cli._parse_stage_list_tokens(["preproc,", "sort"])
    assert parsed == ["preprocess", "spikesort"]


def test_parse_stage_list_tokens_supports_all_keyword() -> None:
    parsed = pipeline_cli._parse_stage_list_tokens(["all"])
    assert parsed == ACTIVE_STAGE_ORDER


def test_canonical_all_selector_excludes_retired_stages() -> None:
    assert list(pipeline_cli._CANONICAL_STAGE_ORDER) == ACTIVE_STAGE_ORDER
    assert "templates" not in pipeline_cli._CANONICAL_STAGE_ORDER
    assert "analysis" not in pipeline_cli._CANONICAL_STAGE_ORDER


@pytest.mark.parametrize("raw_token", ["analysis", "analyse", "analyze"])
def test_parse_stage_list_tokens_rejects_retired_analysis_selector(raw_token: str) -> None:
    with pytest.raises(SystemExit, match="Unsupported stage token"):
        pipeline_cli._parse_stage_list_tokens([raw_token])


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
        ("reconstruct.report_full_chip_layout", "reconstruct.report_full_chip_layout"),
        ("reconstruct.analyzers", "reconstruct.analyzers"),
        ("reconstruct.build_templates", "reconstruct.build_templates"),
        ("reconstruct.compute_template_similarity", "reconstruct.compute_template_similarity"),
        ("reconstruct.plot_templates", "reconstruct.plot_templates"),
        ("reconstruct.report_templates", "reconstruct.report_templates"),
        ("recon.generate_gtrs", "reconstruct.generate_gtrs"),
        ("recon.analyzers", "reconstruct.analyzers"),
        ("recon.templates_analyzers", "reconstruct.analyzers"),
        ("reconstruct.templates_build_templates", "reconstruct.build_templates"),
        ("reconstruction.plot_recons", "reconstruct.plot_recons"),
        ("reconstruction.build_templates", "reconstruct.build_templates"),
        ("reconstruction.templates_plot_templates", "reconstruct.plot_templates"),
        ("recon.plot_branch_propagations", "reconstruct.plot_branch_propagations"),
        ("reconstruction.plot_branch_velocities", "reconstruct.plot_branch_velocities"),
        ("reconstruction.plot_unit_summary", "reconstruct.plot_unit_summary"),
        ("reconstruction.report_recons", "reconstruct.report_recons"),
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


def test_parse_stage_list_tokens_supports_spikesort_merge_substage() -> None:
    parsed = pipeline_cli._parse_stage_list_tokens(["spikesort.merge"])
    assert parsed == ["spikesort.merge"]


def test_parse_stage_list_tokens_supports_spikesort_merge_slay_substage() -> None:
    parsed = pipeline_cli._parse_stage_list_tokens(["spikesort.merge.slay"])
    assert parsed == ["spikesort.merge_SLAy"]


def test_parse_stage_list_tokens_supports_spikesort_merge_slay_phase_name() -> None:
    parsed = pipeline_cli._parse_stage_list_tokens(["spikesort.merge_SLAy"])
    assert parsed == ["spikesort.merge_SLAy"]


def test_parse_stage_list_tokens_supports_spikesort_merge_auto_merge_substage() -> None:
    parsed = pipeline_cli._parse_stage_list_tokens(["spikesort.merge.auto_merge"])
    assert parsed == ["spikesort.merge_si_auto"]


def test_parse_stage_list_tokens_supports_spikesort_merge_si_auto_phase_name() -> None:
    parsed = pipeline_cli._parse_stage_list_tokens(["spikesort.merge_si_auto"])
    assert parsed == ["spikesort.merge_si_auto"]


def test_parse_stage_list_tokens_supports_spikesort_merge_automerge_alias() -> None:
    parsed = pipeline_cli._parse_stage_list_tokens(["spikesort.merge.automerge"])
    assert parsed == ["spikesort.merge_si_auto"]


def test_parse_stage_list_tokens_supports_spikesort_merge_unitmatch_phase_name() -> None:
    parsed = pipeline_cli._parse_stage_list_tokens(["spikesort.merge_unitmatch"])
    assert parsed == ["spikesort.merge_unitmatch"]


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


def test_main_runs_spikesort_merge_si_auto_substage(monkeypatch, tmp_path: Path) -> None:
    runtime_cfg = tmp_path / "runtime.yml"
    _write_runtime_cfg(runtime_cfg)

    calls: list[str] = []

    def _spikesort_merge_si_auto(args):
        calls.append(str(getattr(args, "stage", "")))
        return 0

    monkeypatch.setitem(pipeline_cli._STAGE_HANDLERS, "spikesort.merge_si_auto", _spikesort_merge_si_auto)

    rc = pipeline_cli.main(["stages", "spikesort.merge.auto_merge", "--config", str(runtime_cfg), "--force-restart"])

    assert rc == 0
    assert calls == ["spikesort.merge_si_auto"]


def test_main_runs_spikesort_merge_unitmatch_substage(monkeypatch, tmp_path: Path) -> None:
    runtime_cfg = tmp_path / "runtime.yml"
    _write_runtime_cfg(runtime_cfg)

    calls: list[str] = []

    def _spikesort_merge_unitmatch(args):
        calls.append(str(getattr(args, "stage", "")))
        return 0

    monkeypatch.setitem(pipeline_cli._STAGE_HANDLERS, "spikesort.merge_unitmatch", _spikesort_merge_unitmatch)

    rc = pipeline_cli.main(["stages", "spikesort.merge_unitmatch", "--config", str(runtime_cfg), "--force-restart"])

    assert rc == 0
    assert calls == ["spikesort.merge_unitmatch"]


def test_parse_stage_list_tokens_rejects_unknown_stage() -> None:
    with pytest.raises(SystemExit):
        pipeline_cli._parse_stage_list_tokens(["bogus_stage"])
