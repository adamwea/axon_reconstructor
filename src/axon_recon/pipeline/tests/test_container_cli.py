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


def test_container_wrapper_forwards_config_free_systopo_command(tmp_path: Path) -> None:
    options = container_cli._parse_options(["--no-build", "--dry-run", "systopo"])

    assert options.container_args == ["systopo"]

    cmd = container_cli._build_docker_run_command(repo_root=tmp_path, options=options)

    assert cmd[-2:] == [container_cli.DEFAULT_IMAGE, "systopo"]


def test_container_wrapper_forwards_alloc_flag(tmp_path: Path) -> None:
    forwarded = [
        "stages",
        "reconstruct.report_templates",
        "--config",
        "debug/debug.runtime.yml",
        "--alloc",
    ]
    options = container_cli._parse_options(["--no-build", "--dry-run", "--no-config-mounts", *forwarded])

    assert options.container_args == forwarded

    cmd = container_cli._build_docker_run_command(repo_root=tmp_path, options=options)

    assert cmd[-(len(forwarded) + 1) :] == [container_cli.DEFAULT_IMAGE, *forwarded]


def test_container_wrapper_forwards_singular_target_dataset_flag(tmp_path: Path) -> None:
    forwarded = [
        "stages",
        "preprocess",
        "--config",
        "debug/debug.runtime.yml",
        "--target-dataset",
        "12",
        "--limit-wells",
        "2",
    ]
    options = container_cli._parse_options(["--no-build", "--dry-run", "--no-config-mounts", *forwarded])

    assert options.container_args == forwarded

    cmd = container_cli._build_docker_run_command(repo_root=tmp_path, options=options)

    assert cmd[-(len(forwarded) + 1) :] == [container_cli.DEFAULT_IMAGE, *forwarded]


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


def test_container_wrapper_reads_container_caps_from_runtime_config(tmp_path: Path) -> None:
    config_dir = tmp_path / "debug"
    config_dir.mkdir()
    runtime_cfg = config_dir / "debug.runtime.yml"
    runtime_cfg.write_text(
        """
resources:
  container_caps:
    shm_size: 32g
    memory: 48g
    memory_reservation: 40g
    memory_swap: 64g
    ipc: host
""".strip()
        + "\n",
        encoding="utf-8",
    )

    options = container_cli._parse_options([
        "--no-build",
        "--dry-run",
        "--no-config-mounts",
        "stages",
        "reconstruct.analyzers",
        "--config",
        str(runtime_cfg),
    ])

    cmd = container_cli._build_docker_run_command(repo_root=tmp_path, options=options)

    assert "--shm-size" in cmd
    assert cmd[cmd.index("--shm-size") + 1] == "32g"
    assert "--memory" in cmd
    assert cmd[cmd.index("--memory") + 1] == "48g"
    assert "--memory-reservation" in cmd
    assert cmd[cmd.index("--memory-reservation") + 1] == "40g"
    assert "--memory-swap" in cmd
    assert cmd[cmd.index("--memory-swap") + 1] == "64g"
    assert "--ipc" in cmd
    assert cmd[cmd.index("--ipc") + 1] == "host"


def test_container_wrapper_reads_container_caps_from_yaml_without_host_pyyaml(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "debug"
    config_dir.mkdir()
    runtime_cfg = config_dir / "debug.runtime.yml"
    runtime_cfg.write_text(
        """
resources:
  container_caps:
    shm_size: 32g
    memory: 48g
    memory_reservation: 40g
    memory_swap: 64g
    ipc: host
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        container_cli.RuntimeConfig,
        "load",
        classmethod(
            lambda cls, path: (_ for _ in ()).throw(
                RuntimeError("YAML runtime config requires PyYAML (`pip install pyyaml`).")
            )
        ),
    )

    options = container_cli._parse_options([
        "--no-build",
        "--dry-run",
        "--no-config-mounts",
        "stages",
        "reconstruct.analyzers",
        "--config",
        str(runtime_cfg),
    ])

    cmd = container_cli._build_docker_run_command(repo_root=tmp_path, options=options)

    assert "--shm-size" in cmd
    assert cmd[cmd.index("--shm-size") + 1] == "32g"
    assert "--memory" in cmd
    assert cmd[cmd.index("--memory") + 1] == "48g"
    assert "--memory-reservation" in cmd
    assert cmd[cmd.index("--memory-reservation") + 1] == "40g"
    assert "--memory-swap" in cmd
    assert cmd[cmd.index("--memory-swap") + 1] == "64g"
    assert "--ipc" in cmd
    assert cmd[cmd.index("--ipc") + 1] == "host"


def test_container_wrapper_cli_shm_size_overrides_runtime_config(tmp_path: Path) -> None:
    config_dir = tmp_path / "debug"
    config_dir.mkdir()
    runtime_cfg = config_dir / "debug.runtime.yml"
    runtime_cfg.write_text(
        """
resources:
  container_caps:
    shm_size: 32g
""".strip()
        + "\n",
        encoding="utf-8",
    )

    options = container_cli._parse_options([
        "--no-build",
        "--dry-run",
        "--no-config-mounts",
        "--shm-size",
        "12g",
        "stages",
        "reconstruct.analyzers",
        "--config",
        str(runtime_cfg),
    ])

    cmd = container_cli._build_docker_run_command(repo_root=tmp_path, options=options)

    assert "--shm-size" in cmd
    assert cmd[cmd.index("--shm-size") + 1] == "12g"


def test_container_wrapper_runtime_config_can_disable_shm_override(tmp_path: Path) -> None:
    config_dir = tmp_path / "debug"
    config_dir.mkdir()
    runtime_cfg = config_dir / "debug.runtime.yml"
    runtime_cfg.write_text(
        """
resources:
  container_caps:
    shm_size: null
""".strip()
        + "\n",
        encoding="utf-8",
    )

    options = container_cli._parse_options([
        "--no-build",
        "--dry-run",
        "--no-config-mounts",
        "stages",
        "reconstruct.analyzers",
        "--config",
        str(runtime_cfg),
    ])

    cmd = container_cli._build_docker_run_command(repo_root=tmp_path, options=options)

    assert "--shm-size" not in cmd


def test_container_wrapper_warns_when_parallel_analyzers_have_small_shm(tmp_path: Path) -> None:
    config_dir = tmp_path / "debug"
    config_dir.mkdir()
    runtime_cfg = config_dir / "debug.runtime.yml"
    runtime_cfg.write_text(
        """
resources:
  active_profile: lab_server_safe
  profiles:
    lab_server_safe:
      capacity:
        analyzer_slots: 2
  container_caps:
    shm_size: 8g
""".strip()
        + "\n",
        encoding="utf-8",
    )

    options = container_cli._parse_options([
        "--no-build",
        "--dry-run",
        "--no-config-mounts",
        "stages",
        "reconstruct.analyzers",
        "--config",
        str(runtime_cfg),
    ])

    warnings = container_cli._container_preflight_warnings(options)

    assert len(warnings) == 1
    assert "analyzer_slots=2" in warnings[0]
    assert "16g" in warnings[0]


def test_container_wrapper_warns_when_parallel_analyzers_have_small_shm_without_host_pyyaml(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "debug"
    config_dir.mkdir()
    runtime_cfg = config_dir / "debug.runtime.yml"
    runtime_cfg.write_text(
        """
resources:
  active_profile: lab_server_safe
  profiles:
    lab_server_safe:
      analyzer_slots: 2
  container_caps:
    shm_size: 8g
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        container_cli.RuntimeConfig,
        "load",
        classmethod(
            lambda cls, path: (_ for _ in ()).throw(
                RuntimeError("YAML runtime config requires PyYAML (`pip install pyyaml`).")
            )
        ),
    )

    options = container_cli._parse_options([
        "--no-build",
        "--dry-run",
        "--no-config-mounts",
        "stages",
        "reconstruct.analyzers",
        "--config",
        str(runtime_cfg),
    ])

    warnings = container_cli._container_preflight_warnings(options)

    assert len(warnings) == 1
    assert "analyzer_slots=2" in warnings[0]
    assert "16g" in warnings[0]


def test_container_wrapper_skips_parallel_analyzer_shm_warning_when_ipc_is_host(tmp_path: Path) -> None:
    config_dir = tmp_path / "debug"
    config_dir.mkdir()
    runtime_cfg = config_dir / "debug.runtime.yml"
    runtime_cfg.write_text(
        """
resources:
  active_profile: lab_server_safe
  profiles:
    lab_server_safe:
      capacity:
        analyzer_slots: 2
  container_caps:
    shm_size: 8g
    ipc: host
""".strip()
        + "\n",
        encoding="utf-8",
    )

    options = container_cli._parse_options([
        "--no-build",
        "--dry-run",
        "--no-config-mounts",
        "stages",
        "reconstruct.analyzers",
        "--config",
        str(runtime_cfg),
    ])

    assert container_cli._container_preflight_warnings(options) == []


def test_container_wrapper_cpuset_cpus_is_forwarded_to_docker_run(tmp_path: Path) -> None:
    options = container_cli._parse_options([
        "--no-build",
        "--dry-run",
        "--no-config-mounts",
        "--cpuset-cpus",
        "0-7",
        "stages",
        "preprocess",
        "--config",
        "debug/debug.runtime.yml",
    ])

    assert options.cpuset_cpus == "0-7"

    cmd = container_cli._build_docker_run_command(repo_root=tmp_path, options=options)

    assert "--cpuset-cpus" in cmd
    assert cmd[cmd.index("--cpuset-cpus") + 1] == "0-7"


def test_container_wrapper_cpuset_cpus_defaults_to_none() -> None:
    options = container_cli._parse_options(["stages", "preprocess"])

    assert options.cpuset_cpus is None


def test_container_wrapper_cpuset_cpus_not_in_run_command_when_unset(tmp_path: Path) -> None:
    options = container_cli._parse_options([
        "--no-build",
        "--dry-run",
        "--no-config-mounts",
        "stages",
        "preprocess",
    ])

    cmd = container_cli._build_docker_run_command(repo_root=tmp_path, options=options)

    assert "--cpuset-cpus" not in cmd