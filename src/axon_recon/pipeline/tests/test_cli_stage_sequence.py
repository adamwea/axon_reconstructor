from __future__ import annotations

from pathlib import Path

import pytest

import axon_recon.pipeline.cli as pipeline_cli


def _write_runtime_cfg(path: Path) -> None:
    path.write_text("global_logger: {}\n", encoding="utf-8")


def test_parse_stage_list_tokens_supports_aliases_and_comma_spacing() -> None:
    parsed = pipeline_cli._parse_stage_list_tokens(["preproc,", "sort"])
    assert parsed == ["preprocess", "spikesort"]


def test_parse_stage_list_tokens_supports_all_keyword() -> None:
    parsed = pipeline_cli._parse_stage_list_tokens(["all"])
    assert parsed == list(pipeline_cli._CANONICAL_STAGE_ORDER)


def test_parse_stage_list_tokens_supports_templates_resolve_sources_substage() -> None:
    parsed = pipeline_cli._parse_stage_list_tokens(["templates.resolve_sources"])
    assert parsed == ["templates.resolve_sources"]


def test_parse_stage_list_tokens_supports_spikesort_sort_substage_alias() -> None:
    parsed = pipeline_cli._parse_stage_list_tokens(["spikesort.sort"])
    assert parsed == ["spikesort"]


def test_parse_stage_list_tokens_supports_spikesort_merge_substage() -> None:
    parsed = pipeline_cli._parse_stage_list_tokens(["spikesort.merge"])
    assert parsed == ["spikesort.merge"]


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
    assert calls == list(pipeline_cli._CANONICAL_STAGE_ORDER)


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

    def _templates(args):
        calls.append("templates")
        return 0

    monkeypatch.setitem(pipeline_cli._STAGE_HANDLERS, "preprocess", _preprocess)
    monkeypatch.setitem(pipeline_cli._STAGE_HANDLERS, "spikesort", _spikesort)
    monkeypatch.setitem(pipeline_cli._STAGE_HANDLERS, "templates", _templates)

    rc = pipeline_cli.main(["stages", "preprocess,spikesort,templates", "--config", str(runtime_cfg)])

    assert rc == 3
    assert calls == ["preprocess", "spikesort"]


def test_main_runs_templates_resolve_sources_substage(monkeypatch, tmp_path: Path) -> None:
    runtime_cfg = tmp_path / "runtime.yml"
    _write_runtime_cfg(runtime_cfg)

    calls: list[str] = []

    def _resolve_sources(args):
        calls.append(str(getattr(args, "stage", "")))
        return 0

    monkeypatch.setitem(pipeline_cli._STAGE_HANDLERS, "templates.resolve_sources", _resolve_sources)

    rc = pipeline_cli.main(["stages", "templates.resolve_sources", "--config", str(runtime_cfg)])

    assert rc == 0
    assert calls == ["templates.resolve_sources"]


def test_main_runs_spikesort_sort_substage_alias(monkeypatch, tmp_path: Path) -> None:
    runtime_cfg = tmp_path / "runtime.yml"
    _write_runtime_cfg(runtime_cfg)

    calls: list[str] = []

    def _spikesort(args):
        calls.append(str(getattr(args, "stage", "")))
        return 0

    monkeypatch.setitem(pipeline_cli._STAGE_HANDLERS, "spikesort", _spikesort)

    rc = pipeline_cli.main(["stages", "spikesort.sort", "--config", str(runtime_cfg), "--force-restart"])

    assert rc == 0
    assert calls == ["spikesort"]


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


def test_parse_stage_list_tokens_rejects_unknown_stage() -> None:
    with pytest.raises(SystemExit):
        pipeline_cli._parse_stage_list_tokens(["bogus_stage"])
