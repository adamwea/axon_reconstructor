from __future__ import annotations

from pathlib import Path

from axon_recon.pipeline import container_cli


def test_container_wrapper_defaults_to_current_user_on_posix(monkeypatch) -> None:
    monkeypatch.delenv("AXON_RECON_CONTAINER_USER", raising=False)
    monkeypatch.setattr(container_cli, "_host_uid_gid_user_spec", lambda: "1010:2020")

    options = container_cli._parse_options(["stages", "reconstruct"])

    assert options.container_user == "1010:2020"


def test_container_wrapper_explicit_user_overrides_default_current_user(monkeypatch) -> None:
    monkeypatch.setattr(container_cli, "_host_uid_gid_user_spec", lambda: "1010:2020")

    options = container_cli._parse_options(["--user", "3030:4040", "stages", "reconstruct"])

    assert options.container_user == "3030:4040"


def test_container_wrapper_forwards_pipeline_args_without_stage_whitelist(tmp_path: Path) -> None:
    forwarded = [
        "stages",
        "preprocess",
        "spikesort",
        "reconstruct.analyzers",
        "--config",
        "debug/debug.runtime.yml",
        "--limit-segments",
        "2",
        "--limit-datasets",
        "2",
        "--limit-wells-per-dataset",
        "1",
        "--limit-units",
        "5",
    ]
    options = container_cli._parse_options(["--no-build", "--dry-run", "--no-config-mounts", *forwarded])
    assert options.container_args == forwarded

    cmd = container_cli._build_docker_run_command(repo_root=tmp_path, options=options)
    assert cmd[-(len(forwarded) + 1) :] == [container_cli.DEFAULT_IMAGE, *forwarded]


def test_container_wrapper_stops_parsing_at_first_pipeline_arg() -> None:
    options = container_cli._parse_options([
        "stages",
        "reconstruct",
        "--image",
        "this-is-a-pipeline-arg-now",
    ])

    assert options.image == container_cli.DEFAULT_IMAGE
    assert options.container_args == [
        "stages",
        "reconstruct",
        "--image",
        "this-is-a-pipeline-arg-now",
    ]


def test_container_wrapper_option_before_pipeline_command_changes_image() -> None:
    options = container_cli._parse_options(["--image", "custom:tag", "stages", "all", "--config", "cfg.yml"])

    assert options.image == "custom:tag"
    assert options.container_args == ["stages", "all", "--config", "cfg.yml"]


def test_container_wrapper_gpus_option_is_forwarded_to_docker_run(tmp_path: Path) -> None:
    options = container_cli._parse_options([
        "--no-build",
        "--dry-run",
        "--no-config-mounts",
        "--gpus",
        "all",
        "stages",
        "spikesort",
        "--config",
        "debug/debug.runtime.yml",
    ])

    cmd = container_cli._build_docker_run_command(repo_root=tmp_path, options=options)

    assert "--gpus" in cmd
    gpu_idx = cmd.index("--gpus")
    assert cmd[gpu_idx + 1] == "all"


def test_container_wrapper_no_gpus_overrides_environment(monkeypatch) -> None:
    monkeypatch.setenv("AXON_RECON_CONTAINER_GPUS", "all")

    options = container_cli._parse_options(["--no-gpus", "stages", "spikesort"])

    assert options.gpu_request is None


def test_source_fingerprint_preserves_logical_sibling_symlink(tmp_path: Path) -> None:
    workspace = tmp_path / "pkgs"
    repo_root = workspace / "axon_reconstructor"
    repo_root.mkdir(parents=True)
    (repo_root / "pyproject.toml").write_text("[project]\nname = 'axon-reconstructor'\n", encoding="utf-8")

    external_unitmatch = tmp_path / "external" / "UnitMatchPy"
    external_unitmatch.mkdir(parents=True)
    (external_unitmatch / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")

    sibling_parent = workspace / "UnitMatch"
    sibling_parent.mkdir()
    (sibling_parent / "UnitMatchPy").symlink_to(external_unitmatch, target_is_directory=True)

    assert len(container_cli._source_fingerprint(repo_root)) == 64