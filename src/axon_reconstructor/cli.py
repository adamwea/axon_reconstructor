from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from axon_reconstructor import env_utils
from axon_reconstructor.pipeline.output_paths import compute_mea_analysis_output_dir
from axon_reconstructor.pipeline.pipeline_logging import compute_pipeline_log_file, setup_pipeline_logger
from axon_reconstructor.runtime_config import RuntimeConfig
from axon_reconstructor.pipeline.runner import (
    add_stage_analysis_args,
    add_stage_common_required_args,
    add_stage_execution_args,
    add_stage_kwargs_args,
    add_stage_reconstruct_args,
    add_stage_selector_arg,
    add_stage_spikesort_args,
    add_stage_waveforms_args,
)


def _load_explicit_env_file(*, args: argparse.Namespace) -> None:
    env_file = getattr(args, "env_file", None)
    if env_file is None:
        return
    env_utils.load_env_files_into_os(env_files=[Path(env_file)], override_existing=False)


def _resolve_required_str(*, cli_value: str | None, env_key: str, cli_flag: str) -> str:
    if cli_value is not None and str(cli_value).strip() != "":
        return str(cli_value)
    env_value = env_utils.env_str(env_key, default=None)
    if env_value is not None:
        return str(env_value)
    raise SystemExit(f"{cli_flag} is required (or set {env_key} in --env-file/shell environment).")


def _resolve_required_path(*, cli_value: str | Path | None, env_key: str, cli_flag: str) -> Path:
    if cli_value is not None and str(cli_value).strip() != "":
        return Path(cli_value).expanduser().resolve()
    env_value = env_utils.env_path(env_key, default=None)
    if env_value is not None:
        return Path(env_value).expanduser().resolve()
    raise SystemExit(f"{cli_flag} is required (or set {env_key} in --env-file/shell environment).")


def _resolve_optional_path(*, cli_value: str | Path | None, env_key: str) -> Path | None:
    if cli_value is not None and str(cli_value).strip() != "":
        return Path(cli_value).expanduser().resolve()
    env_value = env_utils.env_path(env_key, default=None)
    if env_value is None:
        return None
    return Path(env_value).expanduser().resolve()


def _resolve_optional_str(*, cli_value: str | None, env_key: str, default: str | None = None) -> str | None:
    if cli_value is not None and str(cli_value).strip() != "":
        return str(cli_value)
    return env_utils.env_str(env_key, default=default)


def _resolve_bool(*, cli_value: bool | None, env_key: str, default: bool) -> bool:
    if cli_value is not None:
        return bool(cli_value)
    return bool(env_utils.env_bool(env_key, default=default))


def _resolve_int(*, cli_value: int | None, env_key: str, default: int) -> int:
    if cli_value is not None:
        return int(cli_value)
    env_value = env_utils.env_int(env_key, default=None)
    if env_value is None:
        return int(default)
    return int(env_value)


def _resolve_optional_int(*, cli_value: int | None, env_key: str, default: int | None = None) -> int | None:
    if cli_value is not None:
        return int(cli_value)
    parsed = env_utils.env_typed(env_key, default=default)
    if parsed is None:
        return None
    try:
        return int(parsed)
    except Exception as e:
        raise SystemExit(f"Invalid integer for {env_key}: {parsed!r}") from e


def _resolve_optional_float(*, cli_value: float | None, env_key: str, default: float | None = None) -> float | None:
    if cli_value is not None:
        return float(cli_value)
    parsed = env_utils.env_typed(env_key, default=default)
    if parsed is None:
        return None
    try:
        return float(parsed)
    except Exception as e:
        raise SystemExit(f"Invalid float for {env_key}: {parsed!r}") from e


def _parse_int_or_none_token(raw: str | int | None) -> int | None:
    if raw is None:
        return None
    token = str(raw).strip().lower()
    if token in {"", "none", "null", "all"}:
        return None
    try:
        return int(token)
    except Exception as e:
        raise SystemExit(f"Invalid int-or-none token: {raw!r}") from e


def _parse_int_or_unlimited_to_optional(raw: str | int | None) -> int | None:
    if raw is None:
        return None
    token = str(raw).strip().lower()
    if token in {"", "none", "null", "all", "unlimited", "inf", "infinite"}:
        return None
    try:
        value = int(token)
    except Exception as e:
        raise SystemExit(f"Invalid int-or-unlimited token: {raw!r}") from e
    return None if value < 0 else int(value)


def _resolve_required_str_cfg(
    *,
    cli_value: str | None,
    cfg: RuntimeConfig,
    cfg_path: str,
    env_key: str,
    cli_flag: str,
) -> str:
    if cli_value is not None and str(cli_value).strip() != "":
        return str(cli_value)
    cfg_value = cfg.get_str(cfg_path, default=None)
    if cfg_value is not None:
        return str(cfg_value)
    env_value = env_utils.env_str(env_key, default=None)
    if env_value is not None:
        return str(env_value)
    raise SystemExit(f"{cli_flag} is required (CLI/YAML/env)")


def _resolve_required_path_cfg(
    *,
    cli_value: str | Path | None,
    cfg: RuntimeConfig,
    cfg_path: str,
    env_key: str,
    cli_flag: str,
) -> Path:
    if cli_value is not None and str(cli_value).strip() != "":
        return Path(cli_value).expanduser().resolve()
    cfg_value = cfg.get_path(cfg_path, default=None)
    if cfg_value is not None:
        return cfg_value
    env_value = env_utils.env_path(env_key, default=None)
    if env_value is not None:
        return Path(env_value).expanduser().resolve()
    raise SystemExit(f"{cli_flag} is required (CLI/YAML/env)")


def _resolve_bool_cfg(
    *,
    cli_value: bool | None,
    cfg: RuntimeConfig,
    cfg_path: str,
    env_key: str,
    default: bool,
) -> bool:
    if cli_value is not None:
        return bool(cli_value)
    cfg_value = cfg.get_bool(cfg_path, default=None)
    if cfg_value is not None:
        return bool(cfg_value)
    return bool(env_utils.env_bool(env_key, default=default))


def _resolve_stage_bool_cfg(
    *,
    stage: str,
    flag_name: str,
    cli_value: bool | None,
    cfg: RuntimeConfig,
    env_key: str,
    default: bool,
    global_fallback_path: str | None,
) -> bool:
    if cli_value is not None:
        return bool(cli_value)
    stage_value = cfg.get_bool(f"stages.{stage}.execution.{flag_name}", default=None)
    if stage_value is not None:
        return bool(stage_value)
    if global_fallback_path:
        global_value = cfg.get_bool(global_fallback_path, default=None)
        if global_value is not None:
            return bool(global_value)
    return bool(env_utils.env_bool(env_key, default=default))


def _resolve_stage_log_level_cfg(
    *,
    stage: str,
    cfg: RuntimeConfig,
    target: str,
    debug_enabled: bool,
) -> str:
    cfg_paths = [
        f"stages.{stage}.logging.{target}_level",
        f"stages.mea_analysis.logging.{target}_level",
        f"global.logging.{target}_level",
    ]
    env_key = f"AXON_RECON_LOG_{target.upper()}_LEVEL"

    raw: object | None = None
    for path in cfg_paths:
        if cfg.has(path):
            raw = cfg.get(path, None)
            break
    if raw is None:
        raw = env_utils.env_str(env_key, default=None)

    if raw is None:
        return "DEBUG" if bool(debug_enabled) else "INFO"

    token = str(raw).strip()
    if token == "":
        return "DEBUG" if bool(debug_enabled) else "INFO"
    if token.isdigit():
        return token

    level_name = token.upper()
    resolved = logging.getLevelName(level_name)
    if not isinstance(resolved, int):
        raise SystemExit(
            f"Invalid logging level for {target}_level: {raw!r}. Use DEBUG/INFO/WARNING/ERROR/CRITICAL or integer level."
        )
    return level_name


def _resolve_int_cfg(
    *,
    cli_value: int | None,
    cfg: RuntimeConfig,
    cfg_path: str,
    env_key: str,
    default: int,
) -> int:
    if cli_value is not None:
        return int(cli_value)
    cfg_value = cfg.get_int(cfg_path, default=None)
    if cfg_value is not None:
        return int(cfg_value)
    env_value = env_utils.env_int(env_key, default=None)
    if env_value is None:
        return int(default)
    return int(env_value)


def _resolve_optional_int_cfg(
    *,
    cli_value: int | None,
    cfg: RuntimeConfig,
    cfg_path: str,
    env_key: str,
    default: int | None = None,
) -> int | None:
    if cli_value is not None:
        return int(cli_value)
    cfg_value = cfg.get_int(cfg_path, default=None)
    if cfg_value is not None:
        return int(cfg_value)
    parsed = env_utils.env_typed(env_key, default=default)
    if parsed is None:
        return None
    try:
        return int(parsed)
    except Exception as e:
        raise SystemExit(f"Invalid integer for {env_key}: {parsed!r}") from e


def _resolve_optional_float_cfg(
    *,
    cli_value: float | None,
    cfg: RuntimeConfig,
    cfg_path: str,
    env_key: str,
    default: float | None = None,
) -> float | None:
    if cli_value is not None:
        return float(cli_value)
    cfg_value = cfg.get_float(cfg_path, default=None)
    if cfg_value is not None:
        return float(cfg_value)
    parsed = env_utils.env_typed(env_key, default=default)
    if parsed is None:
        return None
    try:
        return float(parsed)
    except Exception as e:
        raise SystemExit(f"Invalid float for {env_key}: {parsed!r}") from e


def _resolve_optional_path_cfg(
    *,
    cli_value: str | Path | None,
    cfg: RuntimeConfig,
    cfg_path: str,
    env_key: str,
) -> Path | None:
    if cli_value is not None and str(cli_value).strip() != "":
        return Path(cli_value).expanduser().resolve()
    cfg_value = cfg.get_path(cfg_path, default=None)
    if cfg_value is not None:
        return cfg_value
    env_value = env_utils.env_path(env_key, default=None)
    if env_value is None:
        return None
    return Path(env_value).expanduser().resolve()


def _resolve_optional_str_cfg(
    *,
    cli_value: str | None,
    cfg: RuntimeConfig,
    cfg_path: str,
    env_key: str,
    default: str | None = None,
) -> str | None:
    if cli_value is not None and str(cli_value).strip() != "":
        return str(cli_value)
    cfg_value = cfg.get_str(cfg_path, default=None)
    if cfg_value is not None:
        return str(cfg_value)
    return env_utils.env_str(env_key, default=default)


def _first_cfg_int(cfg: RuntimeConfig, paths: list[str]) -> int | None:
    for path in paths:
        parsed = cfg.get_int(path, default=None)
        if parsed is not None:
            return int(parsed)
    return None


def _first_cfg_str(cfg: RuntimeConfig, paths: list[str]) -> str | None:
    for path in paths:
        parsed = cfg.get_str(path, default=None)
        if parsed is not None and str(parsed).strip() != "":
            return str(parsed)
    return None


def _logical_cores() -> int:
    return max(1, int(os.cpu_count() or 1))


def _clamp_worker_count(*, value: int, label: str, stage: str, logger: logging.Logger) -> int:
    if int(value) < 1:
        raise SystemExit(f"Invalid {label} for stage '{stage}': {value}. Expected >= 1.")
    max_workers = _logical_cores()
    if int(value) > max_workers:
        logger.warning(
            "Clamping %s for stage=%s from %d to logical core limit %d",
            label,
            stage,
            int(value),
            int(max_workers),
        )
        return int(max_workers)
    return int(value)


def _resolve_stage_resource_int(
    *,
    cfg: RuntimeConfig,
    stage: str,
    stage_paths: list[str],
    global_path: str | None,
    env_key: str | None,
    cli_value: int | None,
    default: int,
    clamp: bool,
    label: str,
    logger: logging.Logger,
) -> int:
    if cli_value is not None:
        out = int(cli_value)
    else:
        stage_value = _first_cfg_int(cfg, stage_paths)
        if stage_value is not None:
            out = int(stage_value)
        else:
            global_value = cfg.get_int(global_path, default=None) if global_path else None
            if global_value is not None:
                out = int(global_value)
            else:
                env_value = env_utils.env_int(env_key, default=None) if env_key else None
                out = int(env_value) if env_value is not None else int(default)
    if clamp:
        return _clamp_worker_count(value=int(out), label=label, stage=stage, logger=logger)
    return int(out)


def _resolve_stage_resource_str(
    *,
    cfg: RuntimeConfig,
    stage_paths: list[str],
    global_path: str | None,
    env_key: str | None,
    cli_value: str | None,
    default: str | None,
) -> str | None:
    if cli_value is not None and str(cli_value).strip() != "":
        return str(cli_value)
    stage_value = _first_cfg_str(cfg, stage_paths)
    if stage_value is not None:
        return str(stage_value)
    if global_path:
        global_value = cfg.get_str(global_path, default=None)
        if global_value is not None and str(global_value).strip() != "":
            return str(global_value)
    if env_key:
        env_value = env_utils.env_str(env_key, default=None)
        if env_value is not None and str(env_value).strip() != "":
            return str(env_value)
    return default


def _cmd_analysis_deck(args: argparse.Namespace) -> int:
    _load_explicit_env_file(args=args)
    from axon_reconstructor.pipeline.analysis.analysis_deck import run_with_args

    return int(run_with_args(args))


def _add_mea_common_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mea-environment",
        choices=["nersc", "lab"],
        default="nersc",
        help="Execution environment presets. 'nersc' expects sorting done externally; 'lab' can use --docker.",
    )
    parser.add_argument(
        "--mea-output-root",
        required=True,
        help="Output root used by MEA_Analysis (this is what axon_reconstructor will read from).",
    )
    parser.add_argument(
        "--mea-analysis-repo-root",
        default=None,
        help="Path to the MEA_Analysis repo root (needed to build an executable driver command).",
    )
    parser.add_argument(
        "--sorter",
        default="kilosort4",
        help="Sorter name passed to MEA_Analysis (default: kilosort4).",
    )


def _cmd_mea_sort_cmd(args: argparse.Namespace) -> int:
    from axon_reconstructor.integrations.mea_analysis import MEAAnalysisRunSpec, build_run_pipeline_driver_cmd

    repo_root = args.mea_analysis_repo_root
    if repo_root is None:
        raise SystemExit("--mea-analysis-repo-root is required to print a runnable command")

    docker = None
    if args.mea_environment == "lab":
        docker = args.docker_image
        if docker is None:
            raise SystemExit("--docker-image is required when --mea-environment=lab")

    scratch_dir = args.scratch_dir
    if scratch_dir is None and args.mea_environment == "nersc":
        scratch_dir = os.environ.get("SLURM_TMPDIR")

    spec = MEAAnalysisRunSpec(
        mea_analysis_repo_root=Path(repo_root),
        path=Path(args.data_path),
        output_dir=Path(args.mea_output_root),
        sorter=args.sorter,
        docker=docker,
        cuda_visible_devices=args.cuda_visible_devices,
        require_gpu=bool(args.require_gpu),
        n_jobs=args.n_jobs,
        chunk_duration=args.chunk_duration,
        scratch_dir=Path(scratch_dir) if scratch_dir else None,
        stage_back=args.stage_back,
        stage_back_mode=args.stage_back_mode,
    )
    argv = build_run_pipeline_driver_cmd(spec)
    print(" ".join(argv))
    return 0


def _first_cuda_visible_device() -> str:
    """Return a single-device CUDA_VISIBLE_DEVICES value.

    On some systems CUDA_VISIBLE_DEVICES is a comma-separated list; we prefer the first.
    """

    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not value:
        return "0"
    return value.split(",", 1)[0].strip() or "0"


@dataclass(frozen=True)
class _GpuInteractDefaults:
    account: str
    qos: str
    constraint: str
    time: str


def _gpu_interact_defaults() -> _GpuInteractDefaults:
    return _GpuInteractDefaults(
        account=os.environ.get("GPU_SMOKE_SALLOC_ACCOUNT", ""),
        qos=os.environ.get("GPU_SMOKE_SALLOC_QOS", "interactive"),
        constraint=os.environ.get("GPU_SMOKE_SALLOC_CONSTRAINT", "gpu"),
        time=os.environ.get("GPU_SMOKE_SALLOC_TIME", "02:00:00"),
    )


def _cmd_gpu_interact(args: argparse.Namespace) -> int:
    """Request an interactive GPU allocation and run MEA_Analysis spikesorting inside Shifter.

    Intentionally avoids smoke-test debug knobs (no --debug, no DEBUG console env).
    """

    from axon_reconstructor.integrations.mea_analysis import MEAAnalysisRunSpec, build_run_pipeline_driver_cmd

    defaults = _gpu_interact_defaults()

    account = args.account or defaults.account
    if not account:
        raise SystemExit(
            "--account is required (or set env GPU_SMOKE_SALLOC_ACCOUNT)."
        )

    mea_repo_root = args.mea_analysis_repo_root
    if mea_repo_root is None:
        raise SystemExit("--mea-analysis-repo-root is required")

    shifter_image = args.shifter_image or os.environ.get("SHIFTER_IMAGE")
    if not shifter_image:
        raise SystemExit("--shifter-image is required (or set env SHIFTER_IMAGE)")

    data_path = Path(args.data_path).expanduser().resolve()
    if not data_path.exists():
        raise SystemExit(f"data_path not found: {data_path}")

    output_root = Path(args.mea_output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    cuda_visible_devices = args.cuda_visible_devices or _first_cuda_visible_device()

    # Build a base driver argv (no scratch here; scratch is optionally injected on the compute node).
    spec = MEAAnalysisRunSpec(
        mea_analysis_repo_root=Path(mea_repo_root).expanduser().resolve(),
        path=data_path,
        output_dir=output_root,
        sorter=args.sorter,
        require_gpu=True,
        cuda_visible_devices=cuda_visible_devices,
        n_jobs=args.n_jobs,
        chunk_duration=args.chunk_duration,
        scratch_dir=None,
        stage_back=args.stage_back,
        stage_back_mode=args.stage_back_mode,
    )
    driver_argv = build_run_pipeline_driver_cmd(spec)

    # The driver argv starts with python3 + absolute driver path. We want to run inside the container
    # with unbuffered stdout like the smoke tests.
    # Example: python3 -u /path/to/run_pipeline_driver.py ...
    if len(driver_argv) >= 2 and driver_argv[0] == "python3":
        driver_argv = ["python3", "-u", *driver_argv[1:]]

    # Inside the compute node + container, optionally use SLURM_TMPDIR as scratch.
    driver_cmd_str = " ".join(shlex.quote(x) for x in driver_argv)
    container_script = (
        "set -euo pipefail\n"
        f"cd {shlex.quote(str(Path(mea_repo_root).expanduser().resolve()))}\n"
        "SCRATCH_DIR=\"${SLURM_TMPDIR:-}\"\n"
        "CMD=(" + driver_cmd_str + ")\n"
        "if [[ -n \"$SCRATCH_DIR\" ]]; then\n"
        "  CMD+=(--scratch-dir \"$SCRATCH_DIR\" --stage-back "
        + shlex.quote(args.stage_back)
        + " --stage-back-mode "
        + shlex.quote(args.stage_back_mode)
        + ")\n"
        "fi\n"
        "echo \"Running (in container): ${CMD[*]}\" >&2\n"
        "${CMD[@]}\n"
    )

    # Build the salloc+srun command.
    qos = args.qos or defaults.qos
    constraint = args.constraint or defaults.constraint
    time_limit = args.time or defaults.time

    srun_cmd = [
        "srun",
        "--ntasks=1",
        "--gpus=1",
        f"--cpus-per-task={int(args.n_jobs)}" if args.n_jobs else "--cpus-per-task=16",
        "bash",
        "-lc",
        # Attempt to load shifter module if available, then run shifter.
        "module load shifter >/dev/null 2>&1 || true; "
        + "shifter "
        + f"--image={shlex.quote(shifter_image)} "
        + f"--env=CUDA_VISIBLE_DEVICES={shlex.quote(cuda_visible_devices)} "
        + "/bin/bash -lc "
        + shlex.quote(container_script),
    ]

    salloc_cmd = [
        "salloc",
        "-C",
        constraint,
        "-q",
        qos,
        "-t",
        time_limit,
        "-A",
        account,
        *srun_cmd,
    ]

    if args.dry_run:
        print(" ".join(shlex.quote(x) for x in salloc_cmd))
        return 0

    proc = subprocess.run(salloc_cmd)
    return int(proc.returncode)


def _load_stage_kwargs(args: argparse.Namespace) -> dict:
    kwargs: dict = {}
    if getattr(args, "stage_kwargs_file", None):
        payload = json.loads(Path(args.stage_kwargs_file).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise SystemExit("--stage-kwargs-file must contain a JSON object")
        kwargs.update(payload)
    if getattr(args, "stage_kwargs", None):
        payload = json.loads(args.stage_kwargs)
        if not isinstance(payload, dict):
            raise SystemExit("--stage-kwargs must be a JSON object")
        kwargs.update(payload)
    return kwargs


def _cmd_stage(args: argparse.Namespace) -> int:
    _load_explicit_env_file(args=args)
    config_path = getattr(args, "config", None)
    if config_path is None:
        env_file = getattr(args, "env_file", None)
        candidates: list[Path] = []
        if env_file is not None:
            candidates.append(Path(env_file).expanduser().resolve().with_name("debug.config.yml"))
            candidates.append(Path(env_file).expanduser().resolve().with_name("debug.config.yaml"))
        candidates.append(Path("tools/debug/debug.config.yml").expanduser().resolve())
        candidates.append(Path("tools/debug/debug.config.yaml").expanduser().resolve())
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                config_path = candidate
                break

    runtime_config = RuntimeConfig.load(config_path)
    stage_logger = logging.getLogger("axon_reconstructor.stage")

    from axon_reconstructor.pipeline.runner import StageExecutionContext, execute_stage

    stage = str(args.stage)
    stage_kwargs = _load_stage_kwargs(args)

    debug_enabled = _resolve_stage_bool_cfg(
        stage=stage,
        flag_name="debug",
        cli_value=getattr(args, "debug", None),
        cfg=runtime_config,
        env_key="AXON_RECON_DEBUG",
        default=False,
        global_fallback_path="global.debug",
    )
    force_restart = _resolve_stage_bool_cfg(
        stage=stage,
        flag_name="force_restart",
        cli_value=getattr(args, "force_restart", None),
        cfg=runtime_config,
        env_key="AXON_RECON_FORCE_RESTART",
        default=False,
        global_fallback_path="global.force_restart",
    )
    force_replot = _resolve_stage_bool_cfg(
        stage=stage,
        flag_name="force_replot",
        cli_value=getattr(args, "force_replot", None),
        cfg=runtime_config,
        env_key="AXON_RECON_FORCE_REPLOT",
        default=False,
        global_fallback_path="global.force_replot",
    )
    stage_workers = _resolve_stage_resource_int(
        cfg=runtime_config,
        stage=stage,
        stage_paths=[
            "stages.mea_analysis.resources.stage_workers",
            "stages.mea_analysis.resources.workers_total",
            "stages.mea_analysis.resources.n_jobs",
            f"stages.{stage}.resources.stage_workers",
            f"stages.{stage}.resources.workers_total",
            f"stages.{stage}.resources.n_jobs",
        ],
        global_path="resources.n_jobs",
        env_key="AXON_RECON_N_JOBS",
        cli_value=getattr(args, "n_jobs", None),
        default=8,
        clamp=True,
        label="stage_workers",
        logger=stage_logger,
    )
    stage_well_workers = _resolve_stage_resource_int(
        cfg=runtime_config,
        stage=stage,
        stage_paths=[
            "stages.mea_analysis.resources.well_workers",
            f"stages.{stage}.resources.well_workers",
        ],
        global_path=None,
        env_key=None,
        cli_value=None,
        default=1,
        clamp=True,
        label="well_workers",
        logger=stage_logger,
    )
    if int(stage_well_workers) > 1:
        stage_logger.info(
            "stage resources.well_workers=%d configured for stage=%s; ignored for direct `stage` command (used by scope-run orchestration).",
            int(stage_well_workers),
            stage,
        )
    if "n_jobs" in stage_kwargs:
        stage_kwargs["n_jobs"] = _clamp_worker_count(
            value=int(stage_kwargs["n_jobs"]),
            label="stage_kwargs.n_jobs",
            stage=stage,
            logger=stage_logger,
        )
    sorter = _resolve_optional_str_cfg(
        cli_value=getattr(args, "sorter", None), env_key="AXON_RECON_SORTER", default="kilosort4"
        , cfg=runtime_config, cfg_path="stages.mea_analysis.phases.spikesorting.sorter"
    )
    docker_image = _resolve_optional_str_cfg(
        cli_value=getattr(args, "docker_image", None),
        cfg=runtime_config,
        cfg_path="stages.mea_analysis.phases.spikesorting.docker_image",
        env_key="AXON_RECON_DOCKER_IMAGE",
    )
    chunk_duration = _resolve_stage_resource_str(
        cfg=runtime_config,
        stage_paths=[
            "stages.mea_analysis.resources.chunk_duration",
            f"stages.{stage}.resources.chunk_duration",
        ],
        global_path="resources.chunk_duration",
        env_key="AXON_RECON_CHUNK_DURATION",
        cli_value=getattr(args, "chunk_duration", None),
        default=None,
    )
    mea_analysis_repo_root = _resolve_optional_path_cfg(
        cli_value=getattr(args, "mea_analysis_repo_root", None), env_key="AXON_RECON_MEA_ANALYSIS_REPO_ROOT"
        , cfg=runtime_config, cfg_path="paths.mea_analysis_repo_root"
    )
    debug_max_units = _resolve_optional_int_cfg(
        cli_value=getattr(args, "debug_max_units", None), env_key="AXON_RECON_WF_DEBUG_MAX_UNITS", default=None
        , cfg=runtime_config, cfg_path="stages.mea_analysis.phases.analyzer.waveforms.debug.max_units"
    )
    debug_max_segments = _resolve_optional_int_cfg(
        cli_value=getattr(args, "debug_max_segments", None), env_key="AXON_RECON_WF_DEBUG_MAX_SEGMENTS", default=None
        , cfg=runtime_config, cfg_path="stages.mea_analysis.phases.analyzer.waveforms.debug.max_segments"
    )

    if bool(debug_enabled):
        logging.basicConfig(level=logging.DEBUG, format="[%(name)s] [%(levelname)s] %(message)s", force=True)

    h5_path = _resolve_required_path_cfg(
        cli_value=args.h5_path,
        cfg=runtime_config,
        cfg_path="paths.h5_path",
        env_key="AXON_RECON_H5_PATH",
        cli_flag="--h5-path",
    )
    if not h5_path.exists():
        raise SystemExit(f"h5 path not found: {h5_path}")

    stream_id = _resolve_required_str_cfg(
        cli_value=args.stream_id,
        cfg=runtime_config,
        cfg_path="paths.stream_id",
        env_key="AXON_RECON_STREAM_ID",
        cli_flag="--stream-id",
    )
    mea_output_root = _resolve_required_path_cfg(
        cli_value=args.mea_output_root,
        cfg=runtime_config,
        cfg_path="paths.mea_output_root",
        env_key="AXON_RECON_MEA_OUTPUT_ROOT",
        cli_flag="--mea-output-root",
    )

    console_log_level = _resolve_stage_log_level_cfg(
        stage=stage,
        cfg=runtime_config,
        target="terminal",
        debug_enabled=bool(debug_enabled),
    )
    file_log_level = _resolve_stage_log_level_cfg(
        stage=stage,
        cfg=runtime_config,
        target="file",
        debug_enabled=bool(debug_enabled),
    )
    well_out_dir = compute_mea_analysis_output_dir(
        output_root=mea_output_root,
        data_file=h5_path,
        well=stream_id,
    )
    stage_log_file = compute_pipeline_log_file(
        well_out_dir=well_out_dir,
        data_file=h5_path,
        stream_id=stream_id,
    )
    stage_logger = setup_pipeline_logger(
        log_file=stage_log_file,
        logger_name=f"{__name__}[stream={stream_id}][stage={stage}]",
        verbose=bool(debug_enabled),
        console_level=console_log_level,
        file_level=file_log_level,
        stream=sys.__stdout__,
    )
    stage_logger.info(
        "Logger levels: terminal=%s file=%s",
        str(console_log_level),
        str(file_log_level),
    )

    if bool(force_replot) and stage not in {"waveforms", "preprocess"}:
        stage_logger.info(
            "force_replot is currently applied only by the waveforms stage; ignoring for stage=%s",
            stage,
        )

    if stage == "preprocess":
        preprocess_force_replot = _resolve_bool_cfg(
            cli_value=getattr(args, "force_replot", None),
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.preprocessing.execution.force_replot",
            env_key="AXON_RECON_PREPROCESS_FORCE_REPLOT",
            default=bool(force_replot),
        )
        save_binary = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.preprocessing.execution.save_binary",
            env_key="AXON_RECON_PREPROCESS_SAVE_BINARY",
            default=True,
        )
        preprocess_root = _resolve_optional_str_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.preprocessing.outputs.preprocess_root",
            env_key="AXON_RECON_PREPROCESS_ROOT",
            default=None,
        )
        if preprocess_root is None:
            preprocess_root = _resolve_optional_str_cfg(
                cli_value=None,
                cfg=runtime_config,
                cfg_path="stages.mea_analysis.phases.preprocessing.outputs.multiseg.multiseg_preprocess_outputs",
                env_key="AXON_RECON_PREPROCESS_MULTISEG_OUTPUTS",
                default=None,
            )

        centered_in_root = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.preprocessing.outputs.centered.in_root",
            env_key="AXON_RECON_PREPROCESS_CENTERED_IN_ROOT",
            default=True,
        )
        preprocessed_in_root = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.preprocessing.outputs.preprocessed.in_root",
            env_key="AXON_RECON_PREPROCESS_PREPROCESSED_IN_ROOT",
            default=True,
        )
        reports_in_root = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.preprocessing.outputs.reports.in_root",
            env_key="AXON_RECON_PREPROCESS_REPORTS_IN_ROOT",
            default=False,
        )
        data_in_root = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.preprocessing.outputs.data.in_root",
            env_key="AXON_RECON_PREPROCESS_DATA_IN_ROOT",
            default=True,
        )

        concat_recording = _resolve_optional_str_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.preprocessing.outputs.centered.concat_recording",
            env_key="AXON_RECON_PREPROCESSED_RECORDING",
            default=None,
        )
        if concat_recording is None:
            concat_recording = _resolve_optional_str_cfg(
                cli_value=None,
                cfg=runtime_config,
                cfg_path="stages.mea_analysis.phases.preprocessing.outputs.multiseg.concat_recording",
                env_key="AXON_RECON_PREPROCESSED_RECORDING",
                default=None,
            )
        if concat_recording is None:
            concat_recording = _resolve_optional_str_cfg(
                cli_value=None,
                cfg=runtime_config,
                cfg_path="stages.mea_analysis.phases.preprocessing.outputs.preprocessed_recording",
                env_key="AXON_RECON_PREPROCESSED_RECORDING",
                default=None,
            )
        preprocessed_concat = _resolve_optional_str_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.preprocessing.outputs.preprocessed.concat_recording",
            env_key="AXON_RECON_PREPROCESSED_RECORDING",
            default=None,
        )
        if preprocessed_concat is None:
            preprocessed_concat = _resolve_optional_str_cfg(
                cli_value=None,
                cfg=runtime_config,
                cfg_path="stages.mea_analysis.phases.preprocessing.outputs.multiseg.preprocessed_concat",
                env_key="AXON_RECON_PREPROCESSED_RECORDING",
                default=None,
            )
        common_electrodes = _resolve_optional_str_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.preprocessing.outputs.data.common_electrodes",
            env_key="AXON_RECON_COMMON_ELECTRODES",
            default=None,
        )
        if common_electrodes is None:
            common_electrodes = _resolve_optional_str_cfg(
                cli_value=None,
                cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.preprocessing.outputs.preprocessed.common_electrodes",
            env_key="AXON_RECON_COMMON_ELECTRODES",
            default=None,
            )
        if common_electrodes is None:
            common_electrodes = _resolve_optional_str_cfg(
                cli_value=None,
                cfg=runtime_config,
                cfg_path="stages.mea_analysis.phases.preprocessing.outputs.multiseg.common_electrodes",
                env_key="AXON_RECON_COMMON_ELECTRODES",
                default=None,
            )
        preprocess_config = _resolve_optional_str_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.preprocessing.outputs.reports.preprocess_config",
            env_key="AXON_RECON_PREPROCESS_CONFIG",
            default=None,
        )
        if preprocess_config is None:
            preprocess_config = _resolve_optional_str_cfg(
                cli_value=None,
                cfg=runtime_config,
                cfg_path="stages.mea_analysis.phases.preprocessing.outputs.preprocess_config",
                env_key="AXON_RECON_PREPROCESS_CONFIG",
                default=None,
            )

        assay_stats = _resolve_optional_str_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.preprocessing.outputs.reports.assay_stats",
            env_key="AXON_RECON_PREPROCESS_ASSAY_STATS",
            default=None,
        )
        multiseg_preprocess_outputs = preprocess_root
        centered_segments = _resolve_optional_str_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.preprocessing.outputs.centered.segments_dir",
            env_key="AXON_RECON_PREPROCESS_CENTERED_SEGMENTS",
            default=None,
        )
        if centered_segments is None:
            centered_segments = _resolve_optional_str_cfg(
                cli_value=None,
                cfg=runtime_config,
                cfg_path="stages.mea_analysis.phases.preprocessing.outputs.multiseg.centered_segments",
                env_key="AXON_RECON_PREPROCESS_CENTERED_SEGMENTS",
                default=None,
            )
        preprocessed_segments = _resolve_optional_str_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.preprocessing.outputs.preprocessed.segments_dir",
            env_key="AXON_RECON_PREPROCESS_PREPROCESSED_SEGMENTS",
            default=None,
        )
        if preprocessed_segments is None:
            preprocessed_segments = _resolve_optional_str_cfg(
                cli_value=None,
                cfg=runtime_config,
                cfg_path="stages.mea_analysis.phases.preprocessing.outputs.multiseg.preprocessed_segments",
                env_key="AXON_RECON_PREPROCESS_PREPROCESSED_SEGMENTS",
                default=None,
            )

        maxwell_epochs = _resolve_optional_str_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.preprocessing.outputs.reports.maxwell_epochs",
            env_key="AXON_RECON_PREPROCESS_MAXWELL_EPOCHS",
            default=None,
        )
        if maxwell_epochs is None:
            maxwell_epochs = _resolve_optional_str_cfg(
                cli_value=None,
                cfg=runtime_config,
                cfg_path="stages.mea_analysis.phases.preprocessing.outputs.multiseg.maxwell_epochs",
                env_key="AXON_RECON_PREPROCESS_MAXWELL_EPOCHS",
                default=None,
            )
        concat_epochs = _resolve_optional_str_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.preprocessing.outputs.reports.concat_epochs",
            env_key="AXON_RECON_PREPROCESS_CONCAT_EPOCHS",
            default=None,
        )
        if concat_epochs is None:
            concat_epochs = _resolve_optional_str_cfg(
                cli_value=None,
                cfg=runtime_config,
                cfg_path="stages.mea_analysis.phases.preprocessing.outputs.multiseg.concat_epochs",
                env_key="AXON_RECON_PREPROCESS_CONCAT_EPOCHS",
                default=None,
            )
        channels_report = _resolve_optional_str_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.preprocessing.outputs.reports.channels",
            env_key="AXON_RECON_PREPROCESS_CHANNELS_JSON",
            default=None,
        )
        if channels_report is None:
            channels_report = _resolve_optional_str_cfg(
                cli_value=None,
                cfg=runtime_config,
                cfg_path="stages.mea_analysis.phases.preprocessing.outputs.channels",
                env_key="AXON_RECON_PREPROCESS_CHANNELS_JSON",
                default=None,
            )
        channel_layouts = _resolve_optional_str_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.preprocessing.outputs.reports.channel_layouts_dir",
            env_key="AXON_RECON_PREPROCESS_CHANNEL_LAYOUTS",
            default=None,
        )
        if channel_layouts is None:
            channel_layouts = _resolve_optional_str_cfg(
                cli_value=None,
                cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.preprocessing.outputs.centered.channel_layouts_dir",
            env_key="AXON_RECON_PREPROCESS_CHANNEL_LAYOUTS",
            default=None,
            )
        if channel_layouts is None:
            channel_layouts = _resolve_optional_str_cfg(
                cli_value=None,
                cfg=runtime_config,
                cfg_path="stages.mea_analysis.phases.preprocessing.outputs.diagnostics.channel_layouts",
                env_key="AXON_RECON_PREPROCESS_CHANNEL_LAYOUTS",
                default=None,
            )
        channel_layout_heatmap = _resolve_optional_str_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.preprocessing.outputs.reports.channel_layout_heatmap",
            env_key="AXON_RECON_PREPROCESS_CHANNEL_LAYOUT_HEATMAP_PATH",
            default=None,
        )
        segment_traces = _resolve_optional_str_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.preprocessing.outputs.centered.segment_traces_dir",
            env_key="AXON_RECON_PREPROCESS_SEGMENT_TRACES",
            default=None,
        )
        if segment_traces is None:
            segment_traces = _resolve_optional_str_cfg(
                cli_value=None,
                cfg=runtime_config,
                cfg_path="stages.mea_analysis.phases.preprocessing.outputs.diagnostics.segment_traces",
                env_key="AXON_RECON_PREPROCESS_SEGMENT_TRACES",
                default=None,
            )
        preprocessed_segment_traces = _resolve_optional_str_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.preprocessing.outputs.preprocessed.segment_traces_dir",
            env_key="AXON_RECON_PREPROCESS_PREPROCESSED_SEGMENT_TRACES",
            default=None,
        )
        if preprocessed_segment_traces is None:
            preprocessed_segment_traces = _resolve_optional_str_cfg(
                cli_value=None,
                cfg=runtime_config,
                cfg_path="stages.mea_analysis.phases.preprocessing.outputs.multiseg.preprocessed_segment_traces",
                env_key="AXON_RECON_PREPROCESS_PREPROCESSED_SEGMENT_TRACES",
                default=None,
            )
        concat_cluster_reps = _resolve_optional_str_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.preprocessing.outputs.centered.concat_trace_plot",
            env_key="AXON_RECON_PREPROCESS_CONCAT_CLUSTER_REPS",
            default=None,
        )
        if concat_cluster_reps is None:
            concat_cluster_reps = _resolve_optional_str_cfg(
                cli_value=None,
                cfg=runtime_config,
                cfg_path="stages.mea_analysis.phases.preprocessing.outputs.diagnostics.concat_cluster_reps",
                env_key="AXON_RECON_PREPROCESS_CONCAT_CLUSTER_REPS",
                default=None,
            )
        preprocessed_concat_cluster_reps = _resolve_optional_str_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.preprocessing.outputs.preprocessed.concat_trace_plot",
            env_key="AXON_RECON_PREPROCESS_PREPROCESSED_CONCAT_CLUSTER_REPS",
            default=None,
        )
        if preprocessed_concat_cluster_reps is None:
            preprocessed_concat_cluster_reps = _resolve_optional_str_cfg(
                cli_value=None,
                cfg=runtime_config,
                cfg_path="stages.mea_analysis.phases.preprocessing.outputs.multiseg.preprocessed_concat_cluster_reps",
                env_key="AXON_RECON_PREPROCESS_PREPROCESSED_CONCAT_CLUSTER_REPS",
                default=None,
            )
        n_trace_channels = _resolve_optional_int_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.preprocessing.multiseg.plot.n_trace_channels",
            env_key="AXON_RECON_PREPROCESS_N_TRACE_CHANNELS",
            default=24,
        )
        if n_trace_channels is None:
            n_trace_channels = _resolve_optional_int_cfg(
                cli_value=None,
                cfg=runtime_config,
                cfg_path="stages.mea_analysis.phases.preprocessing.outputs.diagnostics.n_trace_channels",
                env_key="AXON_RECON_PREPROCESS_N_TRACE_CHANNELS",
                default=24,
            )
        plot_centered_traces = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.preprocessing.multiseg.plot.centered_traces",
            env_key="AXON_RECON_PREPROCESS_PLOT_CENTERED",
            default=True,
        )
        plot_preprocessed_traces = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.preprocessing.multiseg.plot.preprocessed_traces",
            env_key="AXON_RECON_PREPROCESS_PLOT_PREPROCESSED",
            default=True,
        )
        plot_channel_layouts = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.preprocessing.multiseg.plot.channel_layouts",
            env_key="AXON_RECON_PREPROCESS_PLOT_CHANNEL_LAYOUTS",
            default=True,
        )
        plot_channel_layout_heatmap = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.preprocessing.multiseg.plot.channel_layout_heatmap",
            env_key="AXON_RECON_PREPROCESS_PLOT_CHANNEL_LAYOUT_HEATMAP",
            default=False,
        )
        centered_outputs_enabled = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.preprocessing.multiseg.enable_outputs",
            env_key="AXON_RECON_PREPROCESS_CENTERED_ENABLE",
            default=True,
        )
        if not runtime_config.has("stages.mea_analysis.phases.preprocessing.multiseg.enable_outputs"):
            centered_outputs_enabled = _resolve_bool_cfg(
                cli_value=None,
                cfg=runtime_config,
                cfg_path="stages.mea_analysis.phases.preprocessing.multiseg.centered.enable_outputs",
                env_key="AXON_RECON_PREPROCESS_CENTERED_ENABLE",
                default=True,
            )
        centered_recordings_enabled = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.preprocessing.multiseg.centered_recordings",
            env_key="AXON_RECON_PREPROCESS_CENTERED_RECORDINGS",
            default=bool(centered_outputs_enabled),
        )
        if not runtime_config.has("stages.mea_analysis.phases.preprocessing.multiseg.centered_recordings"):
            centered_recordings_enabled = _resolve_bool_cfg(
                cli_value=None,
                cfg=runtime_config,
                cfg_path="stages.mea_analysis.phases.preprocessing.multiseg.centered.centered_recordings",
                env_key="AXON_RECON_PREPROCESS_CENTERED_RECORDINGS",
                default=bool(centered_outputs_enabled),
            )
        if not runtime_config.has("stages.mea_analysis.phases.preprocessing.multiseg.centered_recordings") and not runtime_config.has("stages.mea_analysis.phases.preprocessing.multiseg.centered.centered_recordings"):
            centered_recordings_enabled = _resolve_bool_cfg(
                cli_value=None,
                cfg=runtime_config,
                cfg_path="stages.mea_analysis.phases.preprocessing.multiseg.centered.recordings",
                env_key="AXON_RECON_PREPROCESS_CENTERED_RECORDINGS",
                default=bool(centered_outputs_enabled),
            )
        preprocessed_recordings_enabled = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.preprocessing.multiseg.preprocessed_recordings",
            env_key="AXON_RECON_PREPROCESS_PREPROCESSED_RECORDINGS",
            default=True,
        )
        if not runtime_config.has("stages.mea_analysis.phases.preprocessing.multiseg.preprocessed_recordings"):
            preprocessed_recordings_enabled = _resolve_bool_cfg(
                cli_value=None,
                cfg=runtime_config,
                cfg_path="stages.mea_analysis.phases.preprocessing.multiseg.centered.preprocessed_recordings",
                env_key="AXON_RECON_PREPROCESS_PREPROCESSED_RECORDINGS",
                default=True,
            )
        centered_plots_enabled = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.preprocessing.multiseg.plots",
            env_key="AXON_RECON_PREPROCESS_CENTERED_PLOTS",
            default=bool(centered_outputs_enabled),
        )
        if not runtime_config.has("stages.mea_analysis.phases.preprocessing.multiseg.plots"):
            centered_plots_enabled = _resolve_bool_cfg(
                cli_value=None,
                cfg=runtime_config,
                cfg_path="stages.mea_analysis.phases.preprocessing.multiseg.centered.plots",
                env_key="AXON_RECON_PREPROCESS_CENTERED_PLOTS",
                default=bool(centered_outputs_enabled),
            )

        hdmea_cfg = runtime_config.get("HDMEA", {})
        hdmea_geometry: dict[str, object] = {}
        if isinstance(hdmea_cfg, dict):
            brand = hdmea_cfg.get("brand")
            device = hdmea_cfg.get("device")
            if brand is not None and str(brand).strip() != "":
                hdmea_geometry["brand"] = str(brand).strip()
            if device is not None and str(device).strip() != "":
                hdmea_geometry["device"] = str(device).strip()

            pitch_raw = hdmea_cfg.get("pitch_um", hdmea_cfg.get("channel_pitch_um"))
            if pitch_raw is not None:
                try:
                    hdmea_geometry["pitch_um"] = float(pitch_raw)
                except Exception:
                    pass

            count_raw = hdmea_cfg.get("electrode_count", hdmea_cfg.get("n_electrodes"))
            if count_raw is not None:
                try:
                    hdmea_geometry["electrode_count"] = int(count_raw)
                except Exception:
                    pass
            per_well_raw = hdmea_cfg.get("electrodes_per_well")
            if per_well_raw is not None:
                try:
                    hdmea_geometry["electrodes_per_well"] = int(per_well_raw)
                except Exception:
                    pass
            sampling_rate_raw = hdmea_cfg.get("sampling_rate_hz")
            if sampling_rate_raw is not None:
                try:
                    hdmea_geometry["sampling_rate_hz"] = float(sampling_rate_raw)
                except Exception:
                    pass

            phys = hdmea_cfg.get("physical_size_mm", hdmea_cfg.get("dimensions_mm"))
            if isinstance(phys, dict):
                w_raw = phys.get("x", phys.get("width"))
                h_raw = phys.get("y", phys.get("height"))
            else:
                w_raw = hdmea_cfg.get("width_mm")
                h_raw = hdmea_cfg.get("height_mm")

            active_area = hdmea_cfg.get("active_sensing_area_mm")
            if isinstance(active_area, dict):
                aw_raw = active_area.get("x", active_area.get("width"))
                ah_raw = active_area.get("y", active_area.get("height"))
                if aw_raw is not None:
                    try:
                        hdmea_geometry["active_width_mm"] = float(aw_raw)
                    except Exception:
                        pass
                if ah_raw is not None:
                    try:
                        hdmea_geometry["active_height_mm"] = float(ah_raw)
                    except Exception:
                        pass

            if w_raw is not None:
                try:
                    hdmea_geometry["physical_width_mm"] = float(w_raw)
                except Exception:
                    pass
            if h_raw is not None:
                try:
                    hdmea_geometry["physical_height_mm"] = float(h_raw)
                except Exception:
                    pass

            grid = hdmea_cfg.get("grid", hdmea_cfg.get("dimensions"))
            if isinstance(grid, dict):
                gx_raw = grid.get("x", grid.get("width"))
                gy_raw = grid.get("y", grid.get("height"))
                if gx_raw is not None:
                    try:
                        hdmea_geometry["grid_x"] = int(gx_raw)
                    except Exception:
                        pass
                if gy_raw is not None:
                    try:
                        hdmea_geometry["grid_y"] = int(gy_raw)
                    except Exception:
                        pass

            el_size = hdmea_cfg.get("electrode_size_um")
            if isinstance(el_size, dict):
                esx_raw = el_size.get("x", el_size.get("width"))
                esy_raw = el_size.get("y", el_size.get("height"))
                if esx_raw is not None:
                    try:
                        hdmea_geometry["electrode_size_x_um"] = float(esx_raw)
                    except Exception:
                        pass
                if esy_raw is not None:
                    try:
                        hdmea_geometry["electrode_size_y_um"] = float(esy_raw)
                    except Exception:
                        pass

        if "preprocess_root_relpath" not in stage_kwargs and preprocess_root is not None:
            stage_kwargs["preprocess_root_relpath"] = str(preprocess_root)
        if "centered_in_root" not in stage_kwargs:
            stage_kwargs["centered_in_root"] = bool(centered_in_root)
        if "preprocessed_in_root" not in stage_kwargs:
            stage_kwargs["preprocessed_in_root"] = bool(preprocessed_in_root)
        if "reports_in_root" not in stage_kwargs:
            stage_kwargs["reports_in_root"] = bool(reports_in_root)
        if "data_in_root" not in stage_kwargs:
            stage_kwargs["data_in_root"] = bool(data_in_root)
        if "concat_recording_relpath" not in stage_kwargs and concat_recording is not None:
            stage_kwargs["concat_recording_relpath"] = str(concat_recording)
        if "preprocessed_recording_relpath" not in stage_kwargs and preprocessed_concat is not None:
            stage_kwargs["preprocessed_recording_relpath"] = str(preprocessed_concat)
        if "preprocessed_recording_relpath" not in stage_kwargs and concat_recording is not None:
            stage_kwargs["preprocessed_recording_relpath"] = str(concat_recording)
        if "preprocessed_concat_relpath" not in stage_kwargs and preprocessed_concat is not None:
            stage_kwargs["preprocessed_concat_relpath"] = str(preprocessed_concat)
        if "common_electrodes_relpath" not in stage_kwargs and common_electrodes is not None:
            stage_kwargs["common_electrodes_relpath"] = str(common_electrodes)
        if "preprocess_config_relpath" not in stage_kwargs and preprocess_config is not None:
            stage_kwargs["preprocess_config_relpath"] = str(preprocess_config)
        if "multiseg_preprocess_outputs_relpath" not in stage_kwargs and multiseg_preprocess_outputs is not None:
            stage_kwargs["multiseg_preprocess_outputs_relpath"] = str(multiseg_preprocess_outputs)
        if "centered_segments_relpath" not in stage_kwargs and centered_segments is not None:
            stage_kwargs["centered_segments_relpath"] = str(centered_segments)
        if "preprocessed_segments_relpath" not in stage_kwargs and preprocessed_segments is not None:
            stage_kwargs["preprocessed_segments_relpath"] = str(preprocessed_segments)
        if "assay_stats_relpath" not in stage_kwargs and assay_stats is not None:
            stage_kwargs["assay_stats_relpath"] = str(assay_stats)
        if "maxwell_epochs_relpath" not in stage_kwargs and maxwell_epochs is not None:
            stage_kwargs["maxwell_epochs_relpath"] = str(maxwell_epochs)
        if "concat_epochs_relpath" not in stage_kwargs and concat_epochs is not None:
            stage_kwargs["concat_epochs_relpath"] = str(concat_epochs)
        if "channels_relpath" not in stage_kwargs and channels_report is not None:
            stage_kwargs["channels_relpath"] = str(channels_report)
        if "channel_layouts_relpath" not in stage_kwargs and channel_layouts is not None:
            stage_kwargs["channel_layouts_relpath"] = str(channel_layouts)
        if "channel_layout_heatmap_relpath" not in stage_kwargs and channel_layout_heatmap is not None:
            stage_kwargs["channel_layout_heatmap_relpath"] = str(channel_layout_heatmap)
        if "segment_traces_relpath" not in stage_kwargs and segment_traces is not None:
            stage_kwargs["segment_traces_relpath"] = str(segment_traces)
        if "preprocessed_segment_traces_relpath" not in stage_kwargs and preprocessed_segment_traces is not None:
            stage_kwargs["preprocessed_segment_traces_relpath"] = str(preprocessed_segment_traces)
        if "concat_cluster_reps_relpath" not in stage_kwargs and concat_cluster_reps is not None:
            stage_kwargs["concat_cluster_reps_relpath"] = str(concat_cluster_reps)
        if "preprocessed_concat_cluster_reps_relpath" not in stage_kwargs and preprocessed_concat_cluster_reps is not None:
            stage_kwargs["preprocessed_concat_cluster_reps_relpath"] = str(preprocessed_concat_cluster_reps)
        if "n_trace_channels" not in stage_kwargs and n_trace_channels is not None:
            stage_kwargs["n_trace_channels"] = int(n_trace_channels)
        if "plot_centered_traces" not in stage_kwargs:
            stage_kwargs["plot_centered_traces"] = bool(plot_centered_traces)
        if "plot_preprocessed_traces" not in stage_kwargs:
            stage_kwargs["plot_preprocessed_traces"] = bool(plot_preprocessed_traces)
        if "plot_channel_layouts" not in stage_kwargs:
            stage_kwargs["plot_channel_layouts"] = bool(plot_channel_layouts)
        if "channel_layout_heatmap_enabled" not in stage_kwargs:
            stage_kwargs["channel_layout_heatmap_enabled"] = bool(plot_channel_layout_heatmap)
        if "enable_centered_outputs" not in stage_kwargs:
            stage_kwargs["enable_centered_outputs"] = bool(centered_outputs_enabled)
        if "enable_centered_recordings" not in stage_kwargs:
            stage_kwargs["enable_centered_recordings"] = bool(centered_recordings_enabled)
        if "enable_centered_plots" not in stage_kwargs:
            stage_kwargs["enable_centered_plots"] = bool(centered_plots_enabled)
        if "enable_preprocessed_recordings" not in stage_kwargs:
            stage_kwargs["enable_preprocessed_recordings"] = bool(preprocessed_recordings_enabled)
        if "hdmea_geometry" not in stage_kwargs and hdmea_geometry:
            stage_kwargs["hdmea_geometry"] = dict(hdmea_geometry)
        if "save_recording" not in stage_kwargs:
            stage_kwargs["save_recording"] = bool(save_binary)
        if "force_replot" not in stage_kwargs:
            stage_kwargs["force_replot"] = bool(preprocess_force_replot)
        if bool(preprocess_force_replot):
            stage_logger.info("Preprocess force_replot enabled: diagnostics plots will be refreshed")
        stage_kwargs.setdefault("console_log_level", console_log_level)
        stage_kwargs.setdefault("file_log_level", file_log_level)

    stage_logger.info(
        "Effective stage resources: stage=%s stage_workers=%d well_workers=%d chunk_duration=%s",
        stage,
        int(stage_workers),
        int(stage_well_workers),
        str(chunk_duration),
    )

    if stage == "spikesort":
        legacy_flat_spikesort_keys = {
            "unitmatch_merge_units",
            "unitmatch_dry_run",
            "unitmatch_scored_dry_run",
            "unitmatch_output_subdir_name",
            "unitmatch_throughput_subdir_name",
            "unitmatch_max_candidate_pairs",
            "unitmatch_oversplit_min_probability",
            "unitmatch_oversplit_max_suggestions",
            "unitmatch_apply_merges",
            "unitmatch_recursive",
            "unitmatch_max_iterations",
            "unitmatch_max_spikes_per_unit",
            "unitmatch_keep_all_iterations",
            "auto_merge_units",
            "auto_merge_template_diff_thresh",
        }
        legacy_present = sorted([k for k in legacy_flat_spikesort_keys if k in stage_kwargs])
        if legacy_present:
            raise ValueError(
                "Spikesort stage kwargs now require grouped keys (um_kwargs/am_kwargs/option_kwargs); "
                f"legacy flat keys are not supported: {', '.join(legacy_present)}"
            )

        resume_from = _resolve_optional_str_cfg(
            cli_value=getattr(args, "resume_from", None),
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.spikesorting.resume_from",
            env_key="AXON_RECON_SPIKESORT_RESUME_FROM",
            default=None,
        )
        if "resume_from" not in stage_kwargs and resume_from is not None:
            stage_kwargs["resume_from"] = str(resume_from)

        target_phase = _resolve_optional_str_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.spikesorting.target_phase",
            env_key="AXON_RECON_SPIKESORT_TARGET_PHASE",
            default=None,
        )
        if "target_phase" not in stage_kwargs and target_phase is not None:
            stage_kwargs["target_phase"] = str(target_phase)

        run_analyzer = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.analyzer.enabled",
            env_key="AXON_RECON_SPIKESORT_RUN_ANALYZER",
            default=True,
        )
        if "run_analyzer" not in stage_kwargs:
            stage_kwargs["run_analyzer"] = bool(run_analyzer)

        run_reports = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.reports.enabled",
            env_key="AXON_RECON_SPIKESORT_RUN_REPORTS",
            default=True,
        )
        if "run_reports" not in stage_kwargs:
            stage_kwargs["run_reports"] = bool(run_reports)

        um_kwargs = stage_kwargs.get("um_kwargs")
        if um_kwargs is None:
            um_kwargs = {}
        elif not isinstance(um_kwargs, dict):
            raise ValueError("stage_kwargs.um_kwargs must be a mapping")

        am_kwargs = stage_kwargs.get("am_kwargs")
        if am_kwargs is None:
            am_kwargs = {}
        elif not isinstance(am_kwargs, dict):
            raise ValueError("stage_kwargs.am_kwargs must be a mapping")

        option_kwargs = stage_kwargs.get("option_kwargs")
        if option_kwargs is None:
            option_kwargs = {}
        elif not isinstance(option_kwargs, dict):
            raise ValueError("stage_kwargs.option_kwargs must be a mapping")

        unitmatch_merge_units = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.merge.unitmatch.merge_units",
            env_key="AXON_RECON_SPIKESORT_UNITMATCH_MERGE_UNITS",
            default=False,
        )
        if "merge_units" not in um_kwargs:
            um_kwargs["merge_units"] = bool(unitmatch_merge_units)

        unitmatch_dry_run = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.merge.unitmatch.dry_run",
            env_key="AXON_RECON_SPIKESORT_UNITMATCH_DRY_RUN",
            default=True,
        )
        if "dry_run" not in um_kwargs:
            um_kwargs["dry_run"] = bool(unitmatch_dry_run)

        unitmatch_scored_dry_run = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.merge.unitmatch.scored_dry_run",
            env_key="AXON_RECON_SPIKESORT_UNITMATCH_SCORED_DRY_RUN",
            default=True,
        )
        if "scored_dry_run" not in um_kwargs:
            um_kwargs["scored_dry_run"] = bool(unitmatch_scored_dry_run)

        unitmatch_output_subdir_name = _resolve_optional_str_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.merge.unitmatch.output_subdir_name",
            env_key="AXON_RECON_SPIKESORT_UNITMATCH_OUTPUT_SUBDIR_NAME",
            default="unitmatch_outputs",
        )
        if "output_subdir_name" not in um_kwargs and unitmatch_output_subdir_name is not None:
            um_kwargs["output_subdir_name"] = str(unitmatch_output_subdir_name)

        unitmatch_throughput_subdir_name = _resolve_optional_str_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.merge.unitmatch.throughput_subdir_name",
            env_key="AXON_RECON_SPIKESORT_UNITMATCH_THROUGHPUT_SUBDIR_NAME",
            default="unitmatch_throughput",
        )
        if "throughput_subdir_name" not in um_kwargs and unitmatch_throughput_subdir_name is not None:
            um_kwargs["throughput_subdir_name"] = str(unitmatch_throughput_subdir_name)

        unitmatch_oversplit_min_probability = _resolve_optional_float_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.merge.unitmatch.oversplit_min_probability",
            env_key="AXON_RECON_SPIKESORT_UNITMATCH_OVERSPLIT_MIN_PROBABILITY",
            default=None,
        )
        if "oversplit_min_probability" not in um_kwargs and unitmatch_oversplit_min_probability is not None:
            um_kwargs["oversplit_min_probability"] = float(unitmatch_oversplit_min_probability)

        unitmatch_apply_merges = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.merge.unitmatch.apply_merges",
            env_key="AXON_RECON_SPIKESORT_UNITMATCH_APPLY_MERGES",
            default=False,
        )
        if "apply_merges" not in um_kwargs:
            um_kwargs["apply_merges"] = bool(unitmatch_apply_merges)

        unitmatch_recursive = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.merge.unitmatch.recursive",
            env_key="AXON_RECON_SPIKESORT_UNITMATCH_RECURSIVE",
            default=False,
        )
        if "recursive" not in um_kwargs:
            um_kwargs["recursive"] = bool(unitmatch_recursive)

        if runtime_config.has("stages.spikesort.unitmatch.uncapped_iterations"):
            stage_logger.warning(
                "Deprecated config key stages.spikesort.unitmatch.uncapped_iterations is ignored; use stages.mea_analysis.phases.merge.unitmatch.iterations.max=-1 for uncapped recursion."
            )

        unitmatch_keep_all_iterations = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.merge.unitmatch.keep_all_iterations",
            env_key="AXON_RECON_SPIKESORT_UNITMATCH_KEEP_ALL_ITERATIONS",
            default=True,
        )
        if "keep_all_iterations" not in um_kwargs:
            um_kwargs["keep_all_iterations"] = bool(unitmatch_keep_all_iterations)

        max_candidate_pairs_cfg = runtime_config.get_int_or_unlimited(
            "stages.mea_analysis.phases.merge.unitmatch.max_candidate_pairs", default=None
        )
        if max_candidate_pairs_cfg is None:
            max_candidate_pairs_cfg = runtime_config.get_int_or_unlimited(
                "stages.mea_analysis.phases.merge.unitmatch.limits.max_candidate_pairs", default=None
            )
        max_candidate_pairs_env = env_utils.env_str("AXON_RECON_SPIKESORT_UNITMATCH_MAX_CANDIDATE_PAIRS", default=None)
        max_candidate_pairs_env_parsed = None
        if max_candidate_pairs_env is not None:
            max_candidate_pairs_env_parsed = RuntimeConfig({"v": max_candidate_pairs_env}).get_int_or_unlimited("v", default=None)
        max_candidate_pairs = max_candidate_pairs_cfg if max_candidate_pairs_cfg is not None else max_candidate_pairs_env_parsed
        if "max_candidate_pairs" not in um_kwargs and max_candidate_pairs is not None:
            um_kwargs["max_candidate_pairs"] = int(max_candidate_pairs)

        oversplit_max_suggestions_cfg = runtime_config.get_int_or_unlimited(
            "stages.mea_analysis.phases.merge.unitmatch.oversplit_max_suggestions", default=None
        )
        if oversplit_max_suggestions_cfg is None:
            oversplit_max_suggestions_cfg = runtime_config.get_int_or_unlimited(
                "stages.mea_analysis.phases.merge.unitmatch.limits.oversplit_max_suggestions", default=None
            )
        oversplit_max_suggestions_env = env_utils.env_str("AXON_RECON_SPIKESORT_UNITMATCH_OVERSPLIT_MAX_SUGGESTIONS", default=None)
        oversplit_max_suggestions_env_parsed = None
        if oversplit_max_suggestions_env is not None:
            oversplit_max_suggestions_env_parsed = RuntimeConfig({"v": oversplit_max_suggestions_env}).get_int_or_unlimited("v", default=None)
        oversplit_max_suggestions = (
            oversplit_max_suggestions_cfg
            if oversplit_max_suggestions_cfg is not None
            else oversplit_max_suggestions_env_parsed
        )
        if "oversplit_max_suggestions" not in um_kwargs and oversplit_max_suggestions is not None:
            um_kwargs["oversplit_max_suggestions"] = int(oversplit_max_suggestions)

        max_iterations_cfg = runtime_config.get_int_or_unlimited(
            "stages.mea_analysis.phases.merge.unitmatch.max_iterations", default=None
        )
        if max_iterations_cfg is None:
            max_iterations_cfg = runtime_config.get_int_or_unlimited(
                "stages.mea_analysis.phases.merge.unitmatch.iterations.max", default=None
            )
        max_iterations_env = env_utils.env_str("AXON_RECON_SPIKESORT_UNITMATCH_MAX_ITERATIONS", default=None)
        max_iterations_env_parsed = None
        if max_iterations_env is not None:
            max_iterations_env_parsed = RuntimeConfig({"v": max_iterations_env}).get_int_or_unlimited("v", default=None)
        max_iterations = max_iterations_cfg if max_iterations_cfg is not None else max_iterations_env_parsed
        if "max_iterations" not in um_kwargs and max_iterations is not None:
            um_kwargs["max_iterations"] = int(max_iterations)

        max_spikes_per_unit_cfg = runtime_config.get_int_or_unlimited(
            "stages.mea_analysis.phases.merge.unitmatch.max_spikes_per_unit", default=None
        )
        max_spikes_per_unit_env = env_utils.env_str("AXON_RECON_SPIKESORT_UNITMATCH_MAX_SPIKES_PER_UNIT", default=None)
        max_spikes_per_unit_env_parsed = None
        if max_spikes_per_unit_env is not None:
            max_spikes_per_unit_env_parsed = RuntimeConfig({"v": max_spikes_per_unit_env}).get_int_or_unlimited("v", default=None)
        max_spikes_per_unit = (
            max_spikes_per_unit_cfg if max_spikes_per_unit_cfg is not None else max_spikes_per_unit_env_parsed
        )
        if "max_spikes_per_unit" not in um_kwargs and max_spikes_per_unit is not None:
            um_kwargs["max_spikes_per_unit"] = int(max_spikes_per_unit)

        unitmatch_generate_reports = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.merge.unitmatch.generate_reports",
            env_key="AXON_RECON_SPIKESORT_UNITMATCH_GENERATE_REPORTS",
            default=True,
        )
        if "generate_reports" not in um_kwargs:
            um_kwargs["generate_reports"] = bool(unitmatch_generate_reports)

        unitmatch_report_subdir_name = _resolve_optional_str_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.merge.unitmatch.report_subdir_name",
            env_key="AXON_RECON_SPIKESORT_UNITMATCH_REPORT_SUBDIR_NAME",
            default="unitmatch_reports",
        )
        if "report_subdir_name" not in um_kwargs and unitmatch_report_subdir_name is not None:
            um_kwargs["report_subdir_name"] = str(unitmatch_report_subdir_name)

        unitmatch_report_max_heatmap_units = _resolve_optional_int_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.merge.unitmatch.report_max_heatmap_units",
            env_key="AXON_RECON_SPIKESORT_UNITMATCH_REPORT_MAX_HEATMAP_UNITS",
            default=200,
        )
        if "report_max_heatmap_units" not in um_kwargs and unitmatch_report_max_heatmap_units is not None:
            um_kwargs["report_max_heatmap_units"] = int(unitmatch_report_max_heatmap_units)

        auto_merge_units = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.spikesorting.auto_merge_units",
            env_key="AXON_RECON_SPIKESORT_AUTO_MERGE_UNITS",
            default=False,
        )
        if "enabled" not in am_kwargs:
            am_kwargs["enabled"] = bool(auto_merge_units)

        auto_merge_template_diff_thresh = _resolve_optional_str_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.spikesorting.auto_merge_template_diff_thresh",
            env_key="AXON_RECON_SPIKESORT_AUTO_MERGE_TEMPLATE_DIFF_THRESH",
            default="0.05,0.15,0.25",
        )
        if "template_diff_thresh" not in am_kwargs and auto_merge_template_diff_thresh is not None:
            am_kwargs["template_diff_thresh"] = str(auto_merge_template_diff_thresh)

        force_rerun_analyzer = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.spikesorting.rerun_analyzer",
            env_key="AXON_RECON_SPIKESORT_RERUN_ANALYZER",
            default=False,
        )
        if "force_rerun_analyzer" not in option_kwargs:
            option_kwargs["force_rerun_analyzer"] = bool(force_rerun_analyzer)

        multiseg_mode = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.multiseg_mode",
            env_key="AXON_RECON_SPIKESORT_MULTISEG_MODE",
            default=False,
        )
        if "multiseg_mode" not in option_kwargs and multiseg_mode is not None:
            option_kwargs["multiseg_mode"] = bool(multiseg_mode)

        waveform_prefer_merged_sorting = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.analyzer.waveforms.prefer_merged_sorting",
            env_key="AXON_RECON_WF_PREFER_MERGED_SORTING",
            default=False,
        )
        if "waveform_prefer_merged_sorting" not in option_kwargs:
            option_kwargs["waveform_prefer_merged_sorting"] = bool(waveform_prefer_merged_sorting)

        waveform_merged_sorting_dir = _resolve_optional_str_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.analyzer.waveforms.merged_sorting_dir",
            env_key="AXON_RECON_WF_MERGED_SORTING_DIR",
            default=None,
        )
        if "waveform_merged_sorting_dir" not in option_kwargs and waveform_merged_sorting_dir is not None:
            option_kwargs["waveform_merged_sorting_dir"] = str(waveform_merged_sorting_dir)

        enable_phase_output_overrides = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.enable_phase_output_overrides",
            env_key="AXON_RECON_MEA_ENABLE_PHASE_OUTPUT_OVERRIDES",
            default=False,
        )
        if bool(enable_phase_output_overrides) and "phase_output_paths" not in option_kwargs:
            phase_output_paths_cfg = runtime_config.get("stages.mea_analysis.phases", default=None)
            if isinstance(phase_output_paths_cfg, dict) and phase_output_paths_cfg:
                option_kwargs["phase_output_paths"] = dict(phase_output_paths_cfg)
                stage_logger.info("Enabled MEA phase output path overrides from stages.mea_analysis.phases")

        stage_kwargs["um_kwargs"] = um_kwargs
        stage_kwargs["am_kwargs"] = am_kwargs
        stage_kwargs["option_kwargs"] = option_kwargs

        ks_overrides = {
            "ks_th_universal": _resolve_optional_float(
                cli_value=getattr(args, "ks_th_universal", None),
                env_key="AXON_RECON_KS_TH_UNIVERSAL",
                default=None,
            ),
            "ks_th_learned": _resolve_optional_float(
                cli_value=getattr(args, "ks_th_learned", None),
                env_key="AXON_RECON_KS_TH_LEARNED",
                default=None,
            ),
            "ks_th_single_ch": _resolve_optional_float(
                cli_value=getattr(args, "ks_th_single_ch", None),
                env_key="AXON_RECON_KS_TH_SINGLE_CH",
                default=None,
            ),
            "ks_cluster_downsampling": _resolve_optional_int(
                cli_value=getattr(args, "ks_cluster_downsampling", None),
                env_key="AXON_RECON_KS_CLUSTER_DOWNSAMPLING",
                default=None,
            ),
            "ks_nearest_chans": _resolve_optional_int(
                cli_value=getattr(args, "ks_nearest_chans", None),
                env_key="AXON_RECON_KS_NEAREST_CHANS",
                default=None,
            ),
            "ks_max_channel_distance": _resolve_optional_float(
                cli_value=getattr(args, "ks_max_channel_distance", None),
                env_key="AXON_RECON_KS_MAX_CHANNEL_DISTANCE",
                default=None,
            ),
        }
        for key, value in ks_overrides.items():
            if key not in stage_kwargs and value is not None:
                stage_kwargs[key] = value

    if stage == "waveforms":
        max_spikes_per_unit_raw: str | int | None = getattr(args, "max_spikes_per_unit", None)
        if max_spikes_per_unit_raw is None:
            max_spikes_per_unit_cfg = runtime_config.get_int_or_unlimited(
                "stages.mea_analysis.phases.analyzer.waveforms.max_spikes_per_unit",
                default=None,
            )
            max_spikes_per_unit_raw = max_spikes_per_unit_cfg
        if max_spikes_per_unit_raw is None:
            max_spikes_per_unit_raw = env_utils.env_str("AXON_RECON_WF_MAX_SPIKES_PER_UNIT", default=None)
        max_spikes_per_unit = _parse_int_or_unlimited_to_optional(max_spikes_per_unit_raw)

        if "debug_max_units" not in stage_kwargs and debug_max_units is not None:
            stage_kwargs["debug_max_units"] = int(debug_max_units)
        if "debug_max_segments" not in stage_kwargs and debug_max_segments is not None:
            stage_kwargs["debug_max_segments"] = int(debug_max_segments)
        if "force_replot" not in stage_kwargs:
            stage_kwargs["force_replot"] = bool(force_replot)
        if "max_spikes_per_unit" not in stage_kwargs and max_spikes_per_unit is not None:
            stage_kwargs["max_spikes_per_unit"] = int(max_spikes_per_unit)

        prefer_merged_sorting = _resolve_bool(
            cli_value=getattr(args, "prefer_merged_sorting", None),
            env_key="AXON_RECON_WF_PREFER_MERGED_SORTING",
            default=False,
        )
        if "prefer_merged_sorting" not in stage_kwargs:
            stage_kwargs["prefer_merged_sorting"] = bool(prefer_merged_sorting)

        merged_sorting_dir = _resolve_optional_path(
            cli_value=getattr(args, "merged_sorting_dir", None),
            env_key="AXON_RECON_WF_MERGED_SORTING_DIR",
        )
        if "merged_sorting_dir" not in stage_kwargs and merged_sorting_dir is not None:
            stage_kwargs["merged_sorting_dir"] = str(merged_sorting_dir)

        filter_by_maxwell_epochs = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.analyzer.waveforms.filter_by_maxwell_epochs",
            env_key="AXON_RECON_WF_FILTER_BY_MAXWELL_EPOCHS",
            default=True,
        )
        if "filter_by_maxwell_epochs" not in stage_kwargs:
            stage_kwargs["filter_by_maxwell_epochs"] = bool(filter_by_maxwell_epochs)

        filter_by_segment_bounds = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.analyzer.waveforms.filter_by_segment_bounds",
            env_key="AXON_RECON_WF_FILTER_BY_SEGMENT_BOUNDS",
            default=True,
        )
        if "filter_by_segment_bounds" not in stage_kwargs:
            stage_kwargs["filter_by_segment_bounds"] = bool(filter_by_segment_bounds)

        segment_sort_safety_cleanup = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.analyzer.waveforms.segment_sort_safety_cleanup",
            env_key="AXON_RECON_WF_SEGMENT_SORT_SAFETY_CLEANUP",
            default=True,
        )
        if "segment_sort_safety_cleanup" not in stage_kwargs:
            stage_kwargs["segment_sort_safety_cleanup"] = bool(segment_sort_safety_cleanup)

        recompute_channel_groups_for_reused_segments = _resolve_bool(
            cli_value=None,
            env_key="AXON_RECON_WF_RECOMPUTE_CHANNEL_GROUPS_FOR_REUSED_SEGMENTS",
            default=False,
        )
        if "recompute_channel_groups_for_reused_segments" not in stage_kwargs:
            stage_kwargs["recompute_channel_groups_for_reused_segments"] = bool(
                recompute_channel_groups_for_reused_segments
            )

        waveforms_variant_name = _resolve_optional_str_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.mea_analysis.phases.analyzer.waveforms.variant_name",
            env_key="AXON_RECON_WF_VARIANT_NAME",
            default=None,
        )
        if "waveforms_variant_name" not in stage_kwargs and waveforms_variant_name is not None:
            stage_kwargs["waveforms_variant_name"] = str(waveforms_variant_name)

    if stage == "reconstruct":
        if "unit_workers" in stage_kwargs:
            stage_kwargs["unit_workers"] = _clamp_worker_count(
                value=int(stage_kwargs["unit_workers"]),
                label="stage_kwargs.unit_workers",
                stage=stage,
                logger=stage_logger,
            )
        else:
            unit_workers = _resolve_stage_resource_int(
                cfg=runtime_config,
                stage=stage,
                stage_paths=[
                    "stages.reconstruct.resources.unit_workers",
                    "stages.reconstruct.unit_workers",
                ],
                global_path=None,
                env_key="AXON_RECON_RECON_UNIT_WORKERS",
                cli_value=None,
                default=1,
                clamp=True,
                label="unit_workers",
                logger=stage_logger,
            )
            stage_kwargs["unit_workers"] = int(unit_workers)

    if stage == "templates":
        template_unit_workers = _first_cfg_int(
            runtime_config,
            [
                "stages.templates.resources.unit_workers",
                "stages.templates.resources.template_unit_workers",
            ],
        )
        if template_unit_workers is not None and int(template_unit_workers) > 1:
            stage_logger.info(
                "templates resources unit worker settings are not yet implemented; configured value=%d is currently no-op.",
                int(template_unit_workers),
            )

        templates_variant_name = _resolve_optional_str(
            cli_value=getattr(args, "recon_templates_variant_name", None),
            env_key="AXON_RECON_RECON_TEMPLATES_VARIANT_NAME",
            default=None,
        )
        if "templates_variant_name" not in stage_kwargs and templates_variant_name is not None:
            stage_kwargs["templates_variant_name"] = str(templates_variant_name)

        reconstruction_variant_name = _resolve_optional_str(
            cli_value=getattr(args, "recon_variant_name", None),
            env_key="AXON_RECON_RECON_VARIANT_NAME",
            default=None,
        )
        if "reconstruction_variant_name" not in stage_kwargs and reconstruction_variant_name is not None:
            stage_kwargs["reconstruction_variant_name"] = str(reconstruction_variant_name)

        top_n_density_raw = getattr(args, "recon_top_n_density_requested", None)
        if top_n_density_raw is None:
            top_n_density_raw = env_utils.env_str("AXON_RECON_RECON_TOP_N_DENSITY_REQUESTED", default=None)
        top_n_density_requested = _parse_int_or_none_token(top_n_density_raw)
        if "top_n_density_requested" not in stage_kwargs and top_n_density_requested is not None:
            stage_kwargs["top_n_density_requested"] = top_n_density_requested

        write_top_density_grid = _resolve_bool(
            cli_value=getattr(args, "recon_write_top_density_grid", None),
            env_key="AXON_RECON_RECON_WRITE_TOP_DENSITY_GRID",
            default=True,
        )
        if "write_top_density_grid" not in stage_kwargs:
            stage_kwargs["write_top_density_grid"] = bool(write_top_density_grid)

        show_density_scale_debug_text = _resolve_bool(
            cli_value=getattr(args, "recon_show_density_scale_debug_text", None),
            env_key="AXON_RECON_RECON_SHOW_DENSITY_SCALE_DEBUG_TEXT",
            default=False,
        )
        if "show_density_scale_debug_text" not in stage_kwargs:
            stage_kwargs["show_density_scale_debug_text"] = bool(show_density_scale_debug_text)

        show_density_scale_global_debug_text = _resolve_bool(
            cli_value=getattr(args, "recon_show_density_scale_global_debug_text", None),
            env_key="AXON_RECON_RECON_SHOW_DENSITY_SCALE_GLOBAL_DEBUG_TEXT",
            default=False,
        )
        if "show_density_scale_global_debug_text" not in stage_kwargs:
            stage_kwargs["show_density_scale_global_debug_text"] = bool(show_density_scale_global_debug_text)

        show_density_scale_local_debug_text = _resolve_bool(
            cli_value=getattr(args, "recon_show_density_scale_local_debug_text", None),
            env_key="AXON_RECON_RECON_SHOW_DENSITY_SCALE_LOCAL_DEBUG_TEXT",
            default=False,
        )
        if "show_density_scale_local_debug_text" not in stage_kwargs:
            stage_kwargs["show_density_scale_local_debug_text"] = bool(show_density_scale_local_debug_text)

        replot_top_density_grid_only = _resolve_bool(
            cli_value=getattr(args, "recon_replot_top_density_grid_only", None),
            env_key="AXON_RECON_RECON_REPLOT_TOP_DENSITY_GRID_ONLY",
            default=False,
        )
        if "replot_top_density_grid_only" not in stage_kwargs:
            stage_kwargs["replot_top_density_grid_only"] = bool(replot_top_density_grid_only)

    if stage == "analysis":
        analysis_unit_workers = _first_cfg_int(
            runtime_config,
            [
                "stages.analysis.resources.unit_workers",
                "stages.analysis.resources.analysis_unit_workers",
            ],
        )
        if analysis_unit_workers is not None and int(analysis_unit_workers) > 1:
            stage_logger.info(
                "analysis resources unit worker settings are not yet implemented; configured value=%d is currently no-op.",
                int(analysis_unit_workers),
            )

        if args.unit_ids:
            unit_ids = [int(value) for value in args.unit_ids]
        else:
            unit_ids = env_utils.env_int_list("AXON_RECON_UNIT_IDS")

        unit_limit = _parse_int_or_none_token(args.unit_limit)
        if args.unit_limit is None:
            unit_limit = _parse_int_or_none_token(env_utils.env_str("AXON_RECON_UNIT_LIMIT", default=None))

        prefer_curated_waveforms_panels = _resolve_bool(
            cli_value=getattr(args, "prefer_curated_waveforms_panels", None),
            env_key="AXON_RECON_ANALYSIS_PREFER_CURATED_WAVEFORMS_PANELS",
            default=True,
        )

        compute_botm_validation = _resolve_bool(
            cli_value=getattr(args, "botm_enable", None),
            env_key="AXON_RECON_ANALYSIS_BOTM_ENABLE",
            default=False,
        )
        botm_n_events = _resolve_int(
            cli_value=getattr(args, "botm_n_events", None),
            env_key="AXON_RECON_ANALYSIS_BOTM_N_SPIKE",
            default=200,
        )
        botm_n_noise_windows = _resolve_int(
            cli_value=getattr(args, "botm_n_noise_windows", None),
            env_key="AXON_RECON_ANALYSIS_BOTM_N_NOISE",
            default=2000,
        )
        botm_seed = _resolve_int(
            cli_value=getattr(args, "botm_seed", None),
            env_key="AXON_RECON_ANALYSIS_BOTM_SEED",
            default=0,
        )
        botm_prior_signal = float(
            getattr(args, "botm_prior_signal", None)
            if getattr(args, "botm_prior_signal", None) is not None
            else (env_utils.env_float("AXON_RECON_ANALYSIS_BOTM_CHANNEL_MATCH_PRIOR_SIGNAL", default=0.5) or 0.5)
        )
        botm_match_fraction_threshold = float(
            getattr(args, "botm_match_fraction_threshold", None)
            if getattr(args, "botm_match_fraction_threshold", None) is not None
            else (env_utils.env_float("AXON_RECON_ANALYSIS_BOTM_CHANNEL_MATCH_FRACTION_THRESHOLD", default=0.70) or 0.70)
        )
        botm_sorter = _resolve_optional_str(
            cli_value=getattr(args, "botm_sorter", None),
            env_key="AXON_RECON_ANALYSIS_BOTM_SORTER",
            default="kilosort4",
        )

        stage_kwargs.update(
            {
            "unit_ids": unit_ids,
            "unit_limit": unit_limit,
            "prefer_curated_waveforms_panels": bool(prefer_curated_waveforms_panels),
            "compute_botm_validation": bool(compute_botm_validation),
            "botm_n_events": int(botm_n_events),
            "botm_n_noise_windows": int(botm_n_noise_windows),
            "botm_seed": int(botm_seed),
            "botm_prior_signal": float(botm_prior_signal),
            "botm_match_fraction_threshold": float(botm_match_fraction_threshold),
            "botm_sorter": str(botm_sorter),
            }
        )

    context = StageExecutionContext(
        h5_path=h5_path,
        stream_id=stream_id,
        mea_output_root=mea_output_root,
        force_restart=bool(force_restart),
        n_jobs=int(stage_workers),
        sorter=str(sorter or "kilosort4"),
        docker_image=docker_image,
        chunk_duration=chunk_duration,
        mea_analysis_repo_root=mea_analysis_repo_root,
        verbose=bool(debug_enabled),
    )
    try:
        result = execute_stage(
            stage=stage,
            context=context,
            stage_kwargs=stage_kwargs,
            logger=stage_logger,
        )
    except Exception:
        stage_logger.exception("Stage execution failed: stage=%s stream=%s", stage, stream_id)
        raise

    if stage == "preprocess":
        n_common = result.artifacts.get("n_common_electrodes")
        stage_logger.info("preprocess complete: stream=%s common_electrodes=%s", stream_id, n_common)
        return 0
    if stage == "spikesort":
        stage_logger.info("spikesort complete: sorter_output=%s", result.artifacts.get("sorter_output_dir"))
        return 0
    if stage == "waveforms":
        stage_logger.info("waveforms complete: out_dir=%s", result.artifacts.get("waveforms_out_dir"))
        return 0
    if stage == "templates":
        stage_logger.info("templates complete: out_dir=%s", result.artifacts.get("templates_out_dir"))
        return 0
    if stage == "reconstruct":
        stage_logger.info("reconstruction complete: out_dir=%s", result.artifacts.get("reconstruction_out_dir"))
        return 0
    if stage == "analysis":
        stage_logger.info("analysis complete: out_dir=%s", result.artifacts.get("analysis_out_dir"))
        return 0

    raise SystemExit(f"Unsupported stage: {stage}")


def _cmd_scope_run(args: argparse.Namespace) -> int:
    _load_explicit_env_file(args=args)

    from axon_reconstructor.pipeline.scope_config import load_scope_config, summarize_scope_config, validate_scope_config
    from axon_reconstructor.pipeline.runner import run_scope_stage_barriers, write_scope_run_summary

    scope_config = load_scope_config(Path(args.config))
    errors = validate_scope_config(scope_config)
    if errors:
        msg = "\n".join(f"- {e}" for e in errors)
        raise SystemExit(f"Invalid scope config:\n{msg}")

    debug_enabled = _resolve_bool(cli_value=getattr(args, "debug", None), env_key="AXON_RECON_DEBUG", default=False)
    if bool(debug_enabled):
        logging.basicConfig(level=logging.DEBUG, format="[%(name)s] [%(levelname)s] %(message)s", force=True)

    logger = logging.getLogger("axon_reconstructor.scope")
    logger.info("Scope summary: %s", summarize_scope_config(scope_config))

    summary = run_scope_stage_barriers(
        config=scope_config,
        dry_run=bool(args.dry_run),
        logger=logger,
    )

    out_path = Path(args.summary_out).expanduser() if args.summary_out else (scope_config.mea_output_root / "scope_run_summary.json")
    out_path = write_scope_run_summary(summary=summary, out_path=out_path)

    failed_total = 0
    for stage_block in summary.get("stages", []):
        failed_total += int(stage_block.get("failed", 0) or 0)

    print(f"scope-run summary: {out_path}")
    print(f"stages_executed={len(summary.get('stages', []))} failed_targets={failed_total} dry_run={bool(summary.get('dry_run'))}")
    return 0 if failed_total == 0 else 1


def _cmd_scope_config_build(args: argparse.Namespace) -> int:
    from axon_reconstructor.pipeline.scope_config import run_scope_config_build

    return int(run_scope_config_build(args))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="axon-reconstructor")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cmd = sub.add_parser(
        "mea-sort-cmd",
        help="Print a MEA_Analysis run_pipeline_driver.py command for spikesorting (NERSC or lab presets).",
    )
    p_cmd.add_argument("data_path", help="Path to an MEA data file or directory.")
    _add_mea_common_flags(p_cmd)
    p_cmd.add_argument("--docker-image", default=None, help="Docker image for lab server execution.")
    p_cmd.add_argument("--scratch-dir", default=None, help="Scratch dir for NERSC (defaults to $SLURM_TMPDIR if set).")
    p_cmd.add_argument("--stage-back", default="sorter", choices=["none", "sorter", "all"])
    p_cmd.add_argument("--stage-back-mode", default="copy", choices=["copy", "move"])
    p_cmd.add_argument("--cuda-visible-devices", default=None)
    p_cmd.add_argument("--require-gpu", action="store_true")
    p_cmd.add_argument("--n-jobs", type=int, default=None)
    p_cmd.add_argument("--chunk-duration", default=None)
    p_cmd.set_defaults(func=_cmd_mea_sort_cmd)

    p_gpu = sub.add_parser(
        "gpu-interact",
        help=(
            "Request an interactive GPU allocation and run MEA_Analysis spikesorting inside Shifter "
            "(user-facing; no smoke-test debug knobs)."
        ),
    )
    p_gpu.add_argument("data_path", help="Path to an MEA data file or directory.")
    _add_mea_common_flags(p_gpu)
    p_gpu.add_argument("--account", default=None, help="Slurm project/account (e.g. m2043_g).")
    p_gpu.add_argument("--qos", default=None, help="Slurm qos (default: interactive).")
    p_gpu.add_argument("--constraint", default=None, help="Slurm constraint (default: gpu).")
    p_gpu.add_argument("--time", default=None, help="Time limit for salloc (default: 02:00:00).")
    p_gpu.add_argument(
        "--shifter-image",
        default=None,
        help="Shifter image URI (e.g. docker:adammwea/benshalomlab_spikesorter_shifter:v5).",
    )
    p_gpu.add_argument("--cuda-visible-devices", default=None)
    p_gpu.add_argument("--n-jobs", type=int, default=None)
    p_gpu.add_argument("--chunk-duration", default=None)
    p_gpu.add_argument("--stage-back", default="sorter", choices=["none", "sorter", "all"])
    p_gpu.add_argument("--stage-back-mode", default="copy", choices=["copy", "move"])
    p_gpu.add_argument("--dry-run", action="store_true", help="Print the salloc command and exit.")
    p_gpu.set_defaults(func=_cmd_gpu_interact)

    p_stage = sub.add_parser(
        "stage",
        help="Run an individual axon_reconstructor pipeline stage directly.",
    )
    p_stage.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional .env file to load before resolving stage args (CLI flags override env values).",
    )
    p_stage.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML/JSON runtime config file. Precedence: CLI > config > env > defaults.",
    )
    add_stage_selector_arg(p_stage)
    add_stage_common_required_args(p_stage)
    add_stage_spikesort_args(p_stage)
    add_stage_waveforms_args(p_stage)
    add_stage_execution_args(p_stage)
    add_stage_reconstruct_args(p_stage)
    add_stage_analysis_args(p_stage)
    add_stage_kwargs_args(p_stage)
    p_stage.set_defaults(func=_cmd_stage)

    p_analysis_deck = sub.add_parser(
        "analysis-deck",
        help="Build analysis unit-grid deck from stage outputs.",
    )
    p_analysis_deck.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional .env file to load before resolving command args (CLI flags override env values).",
    )
    p_analysis_deck.add_argument("--debug", action=argparse.BooleanOptionalAction, default=None)
    p_analysis_deck.add_argument("--h5-path", type=Path, default=None)
    p_analysis_deck.add_argument("--stream-id", type=str, default=None)
    p_analysis_deck.add_argument("--mea-output-root", type=Path, default=None)
    p_analysis_deck.add_argument("--force-restart", action=argparse.BooleanOptionalAction, default=None)
    p_analysis_deck.add_argument("--unit-limit", type=str, default=None)
    p_analysis_deck.add_argument("--unit-ids", nargs="*", default=None)
    p_analysis_deck.add_argument("--require-complete", action=argparse.BooleanOptionalAction, default=None)
    p_analysis_deck.set_defaults(func=_cmd_analysis_deck)

    p_scope = sub.add_parser(
        "scope-run",
        help=(
            "Run pipeline-native stage barriers across all datasets/wells defined in a scope config. "
            "Executes each stage globally before advancing to the next stage."
        ),
    )
    p_scope.add_argument("--config", required=True, help="Path to scope config (.json/.yml/.yaml)")
    p_scope.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional .env file to load before scope execution (useful for shared runtime flags).",
    )
    p_scope.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan-only mode: validate config and emit execution plan without running stages.",
    )
    p_scope.add_argument(
        "--summary-out",
        default=None,
        help="Optional output path for scope run summary JSON (default: <mea_output_root>/scope_run_summary.json)",
    )
    p_scope.add_argument("--debug", action=argparse.BooleanOptionalAction, default=None, help="Enable debug logging")
    p_scope.set_defaults(func=_cmd_scope_run)

    p_scope_build = sub.add_parser(
        "scope-config-build",
        help="Build a scope-run JSON config from a cross-well config and env defaults.",
    )
    p_scope_build.add_argument("--cross-well-config", required=True, type=Path)
    p_scope_build.add_argument("--env-file", default=None, type=Path)
    p_scope_build.add_argument("--out", required=True, type=Path)
    p_scope_build.add_argument("--stage-order", required=True, help="Comma-separated stage order")
    p_scope_build.add_argument("--per-well-parallelism", type=int, default=1)
    p_scope_build.add_argument("--fail-fast", nargs="?", const="1", default=None)
    p_scope_build.add_argument("--force-restart", nargs="?", const="1", default=None)
    p_scope_build.add_argument("--n-jobs", type=int, default=None)
    p_scope_build.add_argument("--chunk-duration", default=None)
    p_scope_build.add_argument("--mea-output-root", type=Path, default=None)
    p_scope_build.add_argument("--mea-analysis-repo-root", type=Path, default=None)
    p_scope_build.add_argument("--sorter", default=None)
    p_scope_build.add_argument("--docker-image", default=None)
    p_scope_build.add_argument("--recon-unit-workers", type=int, default=None)
    p_scope_build.add_argument("--recon-json-only", action="store_true")
    p_scope_build.set_defaults(func=_cmd_scope_config_build)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except subprocess.CalledProcessError as e:
        return int(getattr(e, "returncode", 1) or 1)
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
