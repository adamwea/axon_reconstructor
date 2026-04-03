from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from axon_reconstructor import env_utils
from axon_reconstructor.runtime_config import RuntimeConfig
from axon_reconstructor.pipeline.pipeline_driver import (
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


def _configure_cli_logging(*, debug_enabled: bool) -> None:
    level = logging.DEBUG if bool(debug_enabled) else logging.INFO
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format="[%(levelname)s] %(message)s", force=True)
        return

    root.setLevel(level)
    for handler in root.handlers:
        try:
            handler.setLevel(level)
        except Exception:
            continue


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


def _first_cfg_path(cfg: RuntimeConfig, paths: list[str]) -> Path | None:
    for path in paths:
        parsed = cfg.get_path(path, default=None)
        if parsed is not None:
            return Path(parsed).expanduser().resolve()
    return None


def _as_bool_token(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return bool(default)


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


def _resolve_runtime_max_workers(
    *,
    cfg: RuntimeConfig,
    cli_value: int | None,
    logger: logging.Logger,
) -> int:
    if cli_value is not None:
        out = int(cli_value)
    else:
        cfg_value = _first_cfg_int(cfg, ["resources.max_workers", "resources.n_jobs"])
        if cfg_value is not None:
            out = int(cfg_value)
        else:
            env_value = env_utils.env_int("AXON_RECON_MAX_WORKERS", default=None)
            if env_value is None:
                env_value = env_utils.env_int("AXON_RECON_N_JOBS", default=None)
            out = int(env_value) if env_value is not None else 8

    if int(out) < 1:
        raise SystemExit(f"Invalid resources.max_workers: {out}. Expected >= 1.")

    logical_cores = _logical_cores()
    if int(out) > int(logical_cores):
        raise SystemExit(
            "Invalid resources.max_workers: "
            f"{int(out)} exceeds available logical cores ({int(logical_cores)})."
        )

    warn_threshold = max(1.0, 0.75 * float(logical_cores))
    if float(out) >= warn_threshold:
        logger.warning(
            "resources.max_workers=%d is >= 75%% of logical cores (%d). High utilization may increase contention.",
            int(out),
            int(logical_cores),
        )

    return int(out)


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


def _discover_runtime_config_path(*, args: argparse.Namespace) -> Path | None:
    explicit = getattr(args, "config", None)
    if explicit is not None:
        return Path(explicit).expanduser().resolve()

    env_file = getattr(args, "env_file", None)
    candidates: list[Path] = []
    if env_file is not None:
        env_path = Path(env_file).expanduser().resolve()
        candidates.append(env_path.with_name("debug.runtime.yml"))
        candidates.append(env_path.with_name("debug.runtime.yaml"))
        candidates.append(env_path.with_name("debug.config.yml"))
        candidates.append(env_path.with_name("debug.config.yaml"))
    candidates.append(Path("tools/debug/debug.runtime.yml").expanduser().resolve())
    candidates.append(Path("tools/debug/debug.runtime.yaml").expanduser().resolve())
    candidates.append(Path("tools/debug/debug.config.yml").expanduser().resolve())
    candidates.append(Path("tools/debug/debug.config.yaml").expanduser().resolve())
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _resolve_mea_python_cmd(*, args: argparse.Namespace, logger: logging.Logger) -> list[str]:
    config_path = _discover_runtime_config_path(args=args)
    if config_path is None:
        return ["python3"]

    try:
        cfg = RuntimeConfig.load(config_path)
    except Exception as e:
        logger.warning("Failed to load runtime config at %s; using python3 (%s)", str(config_path), str(e))
        return ["python3"]

    executable_token = cfg.get_str("env.python.executable_path", default=None)
    if executable_token:
        expanded = os.path.expandvars(str(executable_token))
        executable_path = Path(expanded).expanduser()
        if executable_path.exists() and executable_path.is_file():
            return [str(executable_path.resolve())]
        logger.warning(
            "Configured env.python.executable_path not found at %s; falling back to conda env/python3",
            str(executable_path),
        )

    conda_env_name = cfg.get_str("env.python.conda_env_name", default=None)
    if conda_env_name:
        return ["conda", "run", "-n", str(conda_env_name), "python"]

    return ["python3"]


def _insert_python_unbuffered_flag(argv: list[str]) -> list[str]:
    py_idx = -1
    for i, token in enumerate(argv):
        if "python" in str(token).strip().lower():
            py_idx = i
            break

    if py_idx < 0:
        return list(argv)

    if py_idx + 1 < len(argv) and str(argv[py_idx + 1]).strip() == "-u":
        return list(argv)

    return [*argv[: py_idx + 1], "-u", *argv[py_idx + 1 :]]


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
        "--sorter",
        default="kilosort4",
        help="Sorter name passed to MEA_Analysis (default: kilosort4).",
    )


def _cmd_mea_sort_cmd(args: argparse.Namespace) -> int:
    _load_explicit_env_file(args=args)
    from axon_reconstructor.integrations.mea_analysis import MEAAnalysisRunSpec, build_run_pipeline_driver_cmd

    logger = logging.getLogger("axon_reconstructor.mea_sort_cmd")

    docker = None
    if args.mea_environment == "lab":
        docker = args.docker_image
        if docker is None:
            raise SystemExit("--docker-image is required when --mea-environment=lab")

    scratch_dir = args.scratch_dir
    if scratch_dir is None and args.mea_environment == "nersc":
        scratch_dir = os.environ.get("SLURM_TMPDIR")

    spec = MEAAnalysisRunSpec(
        path=Path(args.data_path),
        output_dir=Path(args.mea_output_root),
        python_cmd=_resolve_mea_python_cmd(args=args, logger=logger),
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

    _load_explicit_env_file(args=args)

    from axon_reconstructor.integrations.mea_analysis import MEAAnalysisRunSpec, build_run_pipeline_driver_cmd

    logger = logging.getLogger("axon_reconstructor.gpu_interact")

    defaults = _gpu_interact_defaults()

    account = args.account or defaults.account
    if not account:
        raise SystemExit(
            "--account is required (or set env GPU_SMOKE_SALLOC_ACCOUNT)."
        )

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
        path=data_path,
        output_dir=output_root,
        python_cmd=_resolve_mea_python_cmd(args=args, logger=logger),
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

    # Run with unbuffered stdout like the smoke tests.
    driver_argv = _insert_python_unbuffered_flag(driver_argv)

    # Inside the compute node + container, optionally use SLURM_TMPDIR as scratch.
    driver_cmd_str = " ".join(shlex.quote(x) for x in driver_argv)
    container_script = (
        "set -euo pipefail\n"
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
    selected_by_default_discovery = False
    stage_logger = logging.getLogger("axon_reconstructor.stage")
    if config_path is None:
        env_file = getattr(args, "env_file", None)
        candidates: list[Path] = []
        if env_file is not None:
            candidates.append(Path(env_file).expanduser().resolve().with_name("debug.runtime.yml"))
            candidates.append(Path(env_file).expanduser().resolve().with_name("debug.runtime.yaml"))
            candidates.append(Path(env_file).expanduser().resolve().with_name("debug.config.yml"))
            candidates.append(Path(env_file).expanduser().resolve().with_name("debug.config.yaml"))
        candidates.append(Path("tools/debug/debug.runtime.yml").expanduser().resolve())
        candidates.append(Path("tools/debug/debug.runtime.yaml").expanduser().resolve())
        candidates.append(Path("tools/debug/debug.config.yml").expanduser().resolve())
        candidates.append(Path("tools/debug/debug.config.yaml").expanduser().resolve())
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                config_path = candidate
                selected_by_default_discovery = True
                break

    merge_paths: list[Path] = []
    runtime_config: RuntimeConfig
    if config_path is not None:
        config_path_resolved = Path(config_path).expanduser().resolve()
        merge_paths.append(config_path_resolved)

        # First load runtime config to inspect explicit companion data file path.
        runtime_cfg_only = RuntimeConfig.load(config_path_resolved)

        data_cfg_token = runtime_cfg_only.get_str("data", None)
        if data_cfg_token:
            data_cfg_path = Path(str(data_cfg_token)).expanduser()
            if not data_cfg_path.is_absolute():
                data_cfg_path = (config_path_resolved.parent / data_cfg_path).resolve()
            else:
                data_cfg_path = data_cfg_path.resolve()
            if not data_cfg_path.exists() or not data_cfg_path.is_file():
                raise FileNotFoundError(
                    f"Runtime config data file not found: {data_cfg_path} (from data={data_cfg_token!r} in {config_path_resolved})"
                )
            merge_paths.append(data_cfg_path)

        # Backward-compatible fallback: auto-merge sibling debug.data.yml when using split runtime name.
        if len(merge_paths) < 2:
            config_name = config_path_resolved.name.lower()
            using_split_runtime_name = config_name in {"debug.runtime.yml", "debug.runtime.yaml"}
            if using_split_runtime_name:
                data_candidate_yml = config_path_resolved.with_name("debug.data.yml")
                data_candidate_yaml = config_path_resolved.with_name("debug.data.yaml")
                if data_candidate_yml.exists() and data_candidate_yml.is_file():
                    merge_paths.append(data_candidate_yml)
                elif data_candidate_yaml.exists() and data_candidate_yaml.is_file():
                    merge_paths.append(data_candidate_yaml)

    if len(merge_paths) >= 2:
        runtime_config = RuntimeConfig.load_multiple([str(p) for p in merge_paths])
    elif len(merge_paths) == 1:
        runtime_config = RuntimeConfig.load(merge_paths[0])
    else:
        runtime_config = RuntimeConfig.load(None)

    selected_runtime_datasets: list[dict[str, Any]] = []
    datasets_block = runtime_config.get("datasets", None)

    if isinstance(datasets_block, list):
        for item in datasets_block:
            if not isinstance(item, dict):
                continue
            include_token = item.get("include_in_runtime", item.get("enabled", item.get("include", False)))
            try:
                include_it = bool(include_token)
            except Exception:
                include_it = False
            if not include_it:
                continue

            ds_h5 = item.get("raw_data_h5_path", None)
            if ds_h5 is None:
                ds_h5 = item.get("h5_path", None)
            if ds_h5 is None:
                continue
            ds_h5_str = str(ds_h5).strip()
            if not ds_h5_str:
                continue

            selected_runtime_datasets.append(item)

    # Backward compatibility: if no explicit selection given and no datasets matched,
    # still honor legacy datasets.active/datasets.h5_path keys when present.
    ds0 = runtime_config.get("datasets.active.h5_path", None)
    if ds0 is None:
        ds0 = runtime_config.get("datasets.h5_path", None)

    if ds0 is None and selected_runtime_datasets:
        selected_dataset = selected_runtime_datasets[0]
        candidate = selected_dataset.get("raw_data_h5_path", None)
        if candidate is None:
            candidate = selected_dataset.get("h5_path", None)
        if candidate is not None:
            ds0 = candidate

    # Normalize useful compatibility paths from selected dataset + top-level output/scratch roots.
    if (
        ds0 is not None
        or runtime_config.get("output_root", None) is not None
        or runtime_config.get("scratch_root", None) is not None
        or runtime_config.get("scratch_output_root", None) is not None
    ):
        payload = dict(getattr(runtime_config, "_payload", {}) or {})
        paths_payload = payload.get("paths")
        if not isinstance(paths_payload, dict):
            paths_payload = {}

        if ds0 is not None and paths_payload.get("h5_path") is None:
            paths_payload["h5_path"] = ds0

        if selected_runtime_datasets and paths_payload.get("stream_id") is None:
            wells_block = selected_runtime_datasets[0].get("wells", None)
            if isinstance(wells_block, list):
                for w in wells_block:
                    if isinstance(w, dict) and w.get("well_id") is not None:
                        token = str(w.get("well_id")).strip()
                        if token:
                            paths_payload["stream_id"] = token
                            break

        if paths_payload.get("mea_output_root") is None:
            output_root = runtime_config.get("output_root", None)
            if output_root is not None:
                paths_payload["mea_output_root"] = output_root

        if paths_payload.get("use_scratch_root") is None:
            use_scratch_root_raw = runtime_config.get("use_scratch_root", None)
            if use_scratch_root_raw is not None:
                paths_payload["use_scratch_root"] = bool(_as_bool_token(use_scratch_root_raw, True))

        use_scratch_root_effective = _as_bool_token(
            paths_payload.get("use_scratch_root", runtime_config.get("use_scratch_root", True)),
            True,
        )

        if paths_payload.get("scratch_output_root") is None and bool(use_scratch_root_effective):
            scratch_root = runtime_config.get("scratch_root", None)
            if scratch_root is None:
                scratch_root = runtime_config.get("scratch_output_root", None)
            if scratch_root is not None:
                paths_payload["scratch_output_root"] = scratch_root

        if not bool(use_scratch_root_effective):
            paths_payload["scratch_output_root"] = None

        payload["paths"] = paths_payload
        runtime_config = RuntimeConfig(payload)

    if selected_by_default_discovery and merge_paths:
        merged_display = " + ".join(str(p) for p in merge_paths[1:])
        stage_logger.info(
            "Loaded runtime config from %s%s",
            merge_paths[0],
            (f" + {merged_display}" if len(merge_paths) > 1 else ""),
        )

    from axon_reconstructor.pipeline.pipeline_driver import StageExecutionContext, execute_stage

    stage = str(args.stage)
    if stage == "all":
        all_stages = ("preprocess", "spikesort", "waveforms", "templates", "reconstruct", "analysis")
        for stage_name in all_stages:
            stage_logger.info("stage all: starting %s", stage_name)
            nested_args = argparse.Namespace(**vars(args))
            nested_args.stage = stage_name
            rc = int(_cmd_stage(nested_args))
            if rc != 0:
                stage_logger.error("stage all: stage %s failed with code %d", stage_name, rc)
                return rc
            stage_logger.info("stage all: completed %s", stage_name)
        return 0

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
    runtime_max_workers = _resolve_runtime_max_workers(
        cfg=runtime_config,
        cli_value=getattr(args, "n_jobs", None),
        logger=stage_logger,
    )

    stage_max_workers = _resolve_stage_resource_int(
        cfg=runtime_config,
        stage=stage,
        stage_paths=[
            f"stages.{stage}.resources.max_stage_workers",
            f"stages.{stage}.resources.stage_workers",
            f"stages.{stage}.resources.workers_total",
            f"stages.{stage}.resources.n_jobs",
        ],
        global_path=None,
        env_key=None,
        cli_value=None,
        default=int(runtime_max_workers),
        clamp=True,
        label="max_stage_workers",
        logger=stage_logger,
    )
    if int(stage_max_workers) > int(runtime_max_workers):
        stage_logger.warning(
            "Clamping max_stage_workers for stage=%s from %d to max_workers=%d",
            stage,
            int(stage_max_workers),
            int(runtime_max_workers),
        )
        stage_max_workers = int(runtime_max_workers)

    stage_well_workers = _resolve_stage_resource_int(
        cfg=runtime_config,
        stage=stage,
        stage_paths=[f"stages.{stage}.resources.well_workers"],
        global_path=None,
        env_key=None,
        cli_value=None,
        default=1,
        clamp=True,
        label="well_workers",
        logger=stage_logger,
    )
    if int(stage_well_workers) > int(stage_max_workers):
        stage_logger.warning(
            "Clamping well_workers for stage=%s from %d to max_stage_workers=%d",
            stage,
            int(stage_well_workers),
            int(stage_max_workers),
        )
        stage_well_workers = int(stage_max_workers)

    derived_stage_n_jobs = max(1, int(stage_max_workers) // int(stage_well_workers))

    if int(stage_well_workers) > 1:
        stage_logger.info(
            "stage resources.well_workers=%d configured for stage=%s; used when multiple dataset/well runs are selected.",
            int(stage_well_workers),
            stage,
        )
    if "n_jobs" in stage_kwargs:
        stage_logger.warning(
            "Ignoring stage_kwargs.n_jobs=%s for stage=%s; using derived n_jobs=%d from max_stage_workers/well_workers",
            str(stage_kwargs.get("n_jobs")),
            stage,
            int(derived_stage_n_jobs),
        )
    stage_kwargs["n_jobs"] = int(derived_stage_n_jobs)
    sorter = _resolve_optional_str_cfg(
        cli_value=getattr(args, "sorter", None), env_key="AXON_RECON_SORTER", default="kilosort4"
        , cfg=runtime_config, cfg_path="stages.spikesort.sorter"
    )
    docker_image = _resolve_optional_str_cfg(
        cli_value=getattr(args, "docker_image", None),
        cfg=runtime_config,
        cfg_path="stages.spikesort.docker_image",
        env_key="AXON_RECON_DOCKER_IMAGE",
    )
    chunk_duration = _resolve_stage_resource_str(
        cfg=runtime_config,
        stage_paths=[f"stages.{stage}.resources.chunk_duration"],
        global_path="resources.chunk_duration",
        env_key="AXON_RECON_CHUNK_DURATION",
        cli_value=getattr(args, "chunk_duration", None),
        default=None,
    )
    debug_max_units = _resolve_optional_int_cfg(
        cli_value=getattr(args, "debug_max_units", None), env_key="AXON_RECON_WF_DEBUG_MAX_UNITS", default=None
        , cfg=runtime_config, cfg_path="stages.waveforms.debug.max_units"
    )
    debug_max_segments = _resolve_optional_int_cfg(
        cli_value=getattr(args, "debug_max_segments", None), env_key="AXON_RECON_WF_DEBUG_MAX_SEGMENTS", default=None
        , cfg=runtime_config, cfg_path="stages.waveforms.debug.max_segments"
    )

    _configure_cli_logging(debug_enabled=bool(debug_enabled))

    h5_path_optional = _resolve_optional_path_cfg(
        cli_value=args.h5_path,
        cfg=runtime_config,
        cfg_path="paths.h5_path",
        env_key="AXON_RECON_H5_PATH",
    )
    stream_id_optional = _resolve_optional_str_cfg(
        cli_value=args.stream_id,
        cfg=runtime_config,
        cfg_path="paths.stream_id",
        env_key="AXON_RECON_STREAM_ID",
        default=None,
    )
    mea_output_root_optional = _resolve_optional_path_cfg(
        cli_value=args.mea_output_root,
        cfg=runtime_config,
        cfg_path="paths.mea_output_root",
        env_key="AXON_RECON_MEA_OUTPUT_ROOT",
    )
    scratch_output_root_optional = _resolve_optional_path_cfg(
        cli_value=getattr(args, "scratch_output_root", None),
        cfg=runtime_config,
        cfg_path="paths.scratch_output_root",
        env_key="AXON_RECON_SCRATCH_OUTPUT_ROOT",
    )
    if scratch_output_root_optional is None:
        scratch_output_root_optional = _first_cfg_path(runtime_config, ["paths.scratch_root"])
    if scratch_output_root_optional is None:
        scratch_top = runtime_config.get("scratch_root", None)
        if scratch_top is None:
            scratch_top = runtime_config.get("scratch_output_root", None)
        if scratch_top is not None and str(scratch_top).strip() != "":
            scratch_output_root_optional = Path(str(scratch_top)).expanduser().resolve()

    use_scratch_root_cfg = runtime_config.get("paths.use_scratch_root", None)
    if use_scratch_root_cfg is None:
        use_scratch_root_cfg = runtime_config.get("use_scratch_root", None)
    use_scratch_root_enabled = _as_bool_token(use_scratch_root_cfg, True)
    if not bool(use_scratch_root_enabled):
        scratch_output_root_optional = None

    run_multi_dataset = (
        len(selected_runtime_datasets) >= 1
        and getattr(args, "h5_path", None) is None
    )

    if (
        not run_multi_dataset
        and getattr(args, "h5_path", None) is None
        and isinstance(datasets_block, list)
        and len(datasets_block) > 0
        and len(selected_runtime_datasets) == 0
    ):
        raise SystemExit(
            "No datasets selected for runtime. Set datasets[].include_in_runtime=true in data config, or pass --h5-path explicitly."
        )

    if run_multi_dataset:
        stage_logger.info(
            "Running stage=%s across %d datasets (well_workers=%d)",
            stage,
            len(selected_runtime_datasets),
            int(stage_well_workers),
        )
    else:
        if h5_path_optional is None:
            raise SystemExit("--h5-path is required (CLI/YAML/env)")
        h5_path = Path(h5_path_optional)
        if not h5_path.exists():
            raise SystemExit(f"h5 path not found: {h5_path}")
        if stream_id_optional is None:
            raise SystemExit("--stream-id is required (CLI/YAML/env)")
        stream_id = str(stream_id_optional)
        if mea_output_root_optional is None:
            raise SystemExit("--mea-output-root is required (CLI/YAML/env)")
        mea_output_root = Path(mea_output_root_optional)
        scratch_output_root = Path(scratch_output_root_optional) if scratch_output_root_optional is not None else None
        active_output_root = scratch_output_root or mea_output_root

    if bool(force_replot) and stage != "waveforms":
        stage_logger.info(
            "force_replot is currently applied only by the waveforms stage; ignoring for stage=%s",
            stage,
        )

    if stage == "preprocess":
        plot_segment_traces = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.preprocess.plot.segment_traces",
            env_key="AXON_RECON_PREPROCESS_PLOT_SEGMENT_TRACES",
            default=True,
        )
        if "plot_segment_traces" not in stage_kwargs:
            stage_kwargs["plot_segment_traces"] = bool(plot_segment_traces)

    stage_logger.info(
        "Effective stage resources: stage=%s max_workers=%d max_stage_workers=%d well_workers=%d derived_n_jobs=%d chunk_duration=%s",
        stage,
        int(runtime_max_workers),
        int(stage_max_workers),
        int(stage_well_workers),
        int(derived_stage_n_jobs),
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
            cfg_path="stages.spikesort.resume_from",
            env_key="AXON_RECON_SPIKESORT_RESUME_FROM",
            default=None,
        )
        if "resume_from" not in stage_kwargs and resume_from is not None:
            stage_kwargs["resume_from"] = str(resume_from)

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
            cfg_path="stages.spikesort.unitmatch.merge_units",
            env_key="AXON_RECON_SPIKESORT_UNITMATCH_MERGE_UNITS",
            default=False,
        )
        if "merge_units" not in um_kwargs:
            um_kwargs["merge_units"] = bool(unitmatch_merge_units)

        unitmatch_dry_run = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.spikesort.unitmatch.dry_run",
            env_key="AXON_RECON_SPIKESORT_UNITMATCH_DRY_RUN",
            default=True,
        )
        if "dry_run" not in um_kwargs:
            um_kwargs["dry_run"] = bool(unitmatch_dry_run)

        unitmatch_scored_dry_run = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.spikesort.unitmatch.scored_dry_run",
            env_key="AXON_RECON_SPIKESORT_UNITMATCH_SCORED_DRY_RUN",
            default=True,
        )
        if "scored_dry_run" not in um_kwargs:
            um_kwargs["scored_dry_run"] = bool(unitmatch_scored_dry_run)

        unitmatch_output_subdir_name = _resolve_optional_str_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.spikesort.unitmatch.output_subdir_name",
            env_key="AXON_RECON_SPIKESORT_UNITMATCH_OUTPUT_SUBDIR_NAME",
            default="unitmatch_outputs",
        )
        if "output_subdir_name" not in um_kwargs and unitmatch_output_subdir_name is not None:
            um_kwargs["output_subdir_name"] = str(unitmatch_output_subdir_name)

        unitmatch_throughput_subdir_name = _resolve_optional_str_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.spikesort.unitmatch.throughput_subdir_name",
            env_key="AXON_RECON_SPIKESORT_UNITMATCH_THROUGHPUT_SUBDIR_NAME",
            default="unitmatch_throughput",
        )
        if "throughput_subdir_name" not in um_kwargs and unitmatch_throughput_subdir_name is not None:
            um_kwargs["throughput_subdir_name"] = str(unitmatch_throughput_subdir_name)

        unitmatch_oversplit_min_probability = _resolve_optional_float_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.spikesort.unitmatch.oversplit_min_probability",
            env_key="AXON_RECON_SPIKESORT_UNITMATCH_OVERSPLIT_MIN_PROBABILITY",
            default=None,
        )
        if "oversplit_min_probability" not in um_kwargs and unitmatch_oversplit_min_probability is not None:
            um_kwargs["oversplit_min_probability"] = float(unitmatch_oversplit_min_probability)

        unitmatch_apply_merges = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.spikesort.unitmatch.apply_merges",
            env_key="AXON_RECON_SPIKESORT_UNITMATCH_APPLY_MERGES",
            default=False,
        )
        if "apply_merges" not in um_kwargs:
            um_kwargs["apply_merges"] = bool(unitmatch_apply_merges)

        unitmatch_recursive = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.spikesort.unitmatch.recursive",
            env_key="AXON_RECON_SPIKESORT_UNITMATCH_RECURSIVE",
            default=False,
        )
        if "recursive" not in um_kwargs:
            um_kwargs["recursive"] = bool(unitmatch_recursive)

        if runtime_config.has("stages.spikesort.unitmatch.uncapped_iterations"):
            stage_logger.warning(
                "Deprecated config key stages.spikesort.unitmatch.uncapped_iterations is ignored; use stages.spikesort.unitmatch.iterations.max=-1 for uncapped recursion."
            )

        unitmatch_keep_all_iterations = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.spikesort.unitmatch.keep_all_iterations",
            env_key="AXON_RECON_SPIKESORT_UNITMATCH_KEEP_ALL_ITERATIONS",
            default=True,
        )
        if "keep_all_iterations" not in um_kwargs:
            um_kwargs["keep_all_iterations"] = bool(unitmatch_keep_all_iterations)

        max_candidate_pairs_cfg = runtime_config.get_int_or_unlimited(
            "stages.spikesort.unitmatch.max_candidate_pairs", default=None
        )
        if max_candidate_pairs_cfg is None:
            max_candidate_pairs_cfg = runtime_config.get_int_or_unlimited(
                "stages.spikesort.unitmatch.limits.max_candidate_pairs", default=None
            )
        max_candidate_pairs_env = env_utils.env_str("AXON_RECON_SPIKESORT_UNITMATCH_MAX_CANDIDATE_PAIRS", default=None)
        max_candidate_pairs_env_parsed = None
        if max_candidate_pairs_env is not None:
            max_candidate_pairs_env_parsed = RuntimeConfig({"v": max_candidate_pairs_env}).get_int_or_unlimited("v", default=None)
        max_candidate_pairs = max_candidate_pairs_cfg if max_candidate_pairs_cfg is not None else max_candidate_pairs_env_parsed
        if "max_candidate_pairs" not in um_kwargs and max_candidate_pairs is not None:
            um_kwargs["max_candidate_pairs"] = int(max_candidate_pairs)

        oversplit_max_suggestions_cfg = runtime_config.get_int_or_unlimited(
            "stages.spikesort.unitmatch.oversplit_max_suggestions", default=None
        )
        if oversplit_max_suggestions_cfg is None:
            oversplit_max_suggestions_cfg = runtime_config.get_int_or_unlimited(
                "stages.spikesort.unitmatch.limits.oversplit_max_suggestions", default=None
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
            "stages.spikesort.unitmatch.max_iterations", default=None
        )
        if max_iterations_cfg is None:
            max_iterations_cfg = runtime_config.get_int_or_unlimited(
                "stages.spikesort.unitmatch.iterations.max", default=None
            )
        max_iterations_env = env_utils.env_str("AXON_RECON_SPIKESORT_UNITMATCH_MAX_ITERATIONS", default=None)
        max_iterations_env_parsed = None
        if max_iterations_env is not None:
            max_iterations_env_parsed = RuntimeConfig({"v": max_iterations_env}).get_int_or_unlimited("v", default=None)
        max_iterations = max_iterations_cfg if max_iterations_cfg is not None else max_iterations_env_parsed
        if "max_iterations" not in um_kwargs and max_iterations is not None:
            um_kwargs["max_iterations"] = int(max_iterations)

        max_spikes_per_unit_cfg = runtime_config.get_int_or_unlimited(
            "stages.spikesort.unitmatch.max_spikes_per_unit", default=None
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
            cfg_path="stages.spikesort.unitmatch.generate_reports",
            env_key="AXON_RECON_SPIKESORT_UNITMATCH_GENERATE_REPORTS",
            default=True,
        )
        if "generate_reports" not in um_kwargs:
            um_kwargs["generate_reports"] = bool(unitmatch_generate_reports)

        unitmatch_report_subdir_name = _resolve_optional_str_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.spikesort.unitmatch.report_subdir_name",
            env_key="AXON_RECON_SPIKESORT_UNITMATCH_REPORT_SUBDIR_NAME",
            default="unitmatch_reports",
        )
        if "report_subdir_name" not in um_kwargs and unitmatch_report_subdir_name is not None:
            um_kwargs["report_subdir_name"] = str(unitmatch_report_subdir_name)

        unitmatch_report_max_heatmap_units = _resolve_optional_int_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.spikesort.unitmatch.report_max_heatmap_units",
            env_key="AXON_RECON_SPIKESORT_UNITMATCH_REPORT_MAX_HEATMAP_UNITS",
            default=200,
        )
        if "report_max_heatmap_units" not in um_kwargs and unitmatch_report_max_heatmap_units is not None:
            um_kwargs["report_max_heatmap_units"] = int(unitmatch_report_max_heatmap_units)

        auto_merge_units = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.spikesort.auto_merge_units",
            env_key="AXON_RECON_SPIKESORT_AUTO_MERGE_UNITS",
            default=False,
        )
        if "enabled" not in am_kwargs:
            am_kwargs["enabled"] = bool(auto_merge_units)

        auto_merge_template_diff_thresh = _resolve_optional_str_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.spikesort.auto_merge_template_diff_thresh",
            env_key="AXON_RECON_SPIKESORT_AUTO_MERGE_TEMPLATE_DIFF_THRESH",
            default="0.05,0.15,0.25",
        )
        if "template_diff_thresh" not in am_kwargs and auto_merge_template_diff_thresh is not None:
            am_kwargs["template_diff_thresh"] = str(auto_merge_template_diff_thresh)

        force_rerun_analyzer = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.spikesort.rerun_analyzer",
            env_key="AXON_RECON_SPIKESORT_RERUN_ANALYZER",
            default=False,
        )
        if "force_rerun_analyzer" not in option_kwargs:
            option_kwargs["force_rerun_analyzer"] = bool(force_rerun_analyzer)

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
        per_segment = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.waveforms.per_segment",
            env_key="AXON_RECON_WF_PER_SEGMENT",
            default=True,
        )
        if "per_segment" not in stage_kwargs:
            stage_kwargs["per_segment"] = bool(per_segment)

        per_segment_preprocess_like_mea_analysis = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.waveforms.per_segment_preprocess_like_mea_analysis",
            env_key="AXON_RECON_WF_PER_SEGMENT_PREPROCESS_LIKE_MEA_ANALYSIS",
            default=True,
        )
        if "per_segment_preprocess_like_mea_analysis" not in stage_kwargs:
            stage_kwargs["per_segment_preprocess_like_mea_analysis"] = bool(
                per_segment_preprocess_like_mea_analysis
            )

        per_segment_only_additional_channels = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.waveforms.per_segment_only_additional_channels",
            env_key="AXON_RECON_WF_PER_SEGMENT_ONLY_ADDITIONAL_CHANNELS",
            default=True,
        )
        if "per_segment_only_additional_channels" not in stage_kwargs:
            stage_kwargs["per_segment_only_additional_channels"] = bool(
                per_segment_only_additional_channels
            )

        max_spikes_per_unit_raw: str | int | None = getattr(args, "max_spikes_per_unit", None)
        if max_spikes_per_unit_raw is None:
            max_spikes_per_unit_cfg = runtime_config.get_int_or_unlimited("stages.waveforms.max_spikes_per_unit", default=None)
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
            cfg_path="stages.waveforms.filter_by_maxwell_epochs",
            env_key="AXON_RECON_WF_FILTER_BY_MAXWELL_EPOCHS",
            default=True,
        )
        if "filter_by_maxwell_epochs" not in stage_kwargs:
            stage_kwargs["filter_by_maxwell_epochs"] = bool(filter_by_maxwell_epochs)

        filter_by_segment_bounds = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.waveforms.filter_by_segment_bounds",
            env_key="AXON_RECON_WF_FILTER_BY_SEGMENT_BOUNDS",
            default=True,
        )
        if "filter_by_segment_bounds" not in stage_kwargs:
            stage_kwargs["filter_by_segment_bounds"] = bool(filter_by_segment_bounds)

        segment_sort_safety_cleanup = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.waveforms.segment_sort_safety_cleanup",
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
            cfg_path="stages.waveforms.variant_name",
            env_key="AXON_RECON_WF_VARIANT_NAME",
            default=None,
        )
        if "waveforms_variant_name" not in stage_kwargs and waveforms_variant_name is not None:
            stage_kwargs["waveforms_variant_name"] = str(waveforms_variant_name)

    if stage == "reconstruct":
        if "n_jobs" in stage_kwargs:
            stage_logger.warning(
                "Ignoring stage_kwargs.n_jobs=%s for reconstruct; using derived n_jobs=%d from max_stage_workers/well_workers",
                str(stage_kwargs.get("n_jobs")),
                int(derived_stage_n_jobs),
            )
        stage_kwargs["n_jobs"] = int(derived_stage_n_jobs)

        av_params = stage_kwargs.get("axon_velocity_params")
        if av_params is None:
            av_params = {}
        elif not isinstance(av_params, dict):
            raise ValueError("stage_kwargs.axon_velocity_params must be a mapping")

        for cfg_path in ("stages.reconstruct.av", "stages.reconstruct.axon_velocity"):
            av_cfg = runtime_config.get(cfg_path, default=None)
            if av_cfg is None:
                continue
            if not isinstance(av_cfg, dict):
                raise ValueError(f"{cfg_path} must be a mapping/object")
            for key, value in av_cfg.items():
                av_params[str(key)] = value

        # Optional env-level overrides for AV parameters.
        av_param_keys = [
            "upsample",
            "init_delay",
            "detect_threshold",
            "kurt_threshold",
            "peak_std_threshold",
            "peak_std_distance",
            "remove_isolated",
            "detection_type",
            "min_selected_points",
            "min_path_length",
            "min_path_points",
            "min_points_after_branching",
            "r2_threshold",
            "max_distance_for_edge",
            "max_distance_to_init",
            "mad_threshold",
            "n_neighbors",
            "init_amp_peak_ratio",
            "edge_dist_amp_ratio",
            "distance_exp",
            "max_peak_latency_for_splitting",
            "r2_threshold_for_outliers",
            "min_outlier_tracking_error",
            "theilsen_maxiter",
            "neighbor_radius",
            "split_paths",
        ]
        for key in av_param_keys:
            env_key = f"AXON_RECON_AV_{str(key).upper()}"
            raw = env_utils.env_str(env_key, default=None)
            if raw is None:
                continue
            av_params[key] = env_utils.parse_typed_value(raw)

        if av_params:
            stage_kwargs["axon_velocity_params"] = dict(av_params)

    if stage == "reconstruct":
        cli_unit_ids: list[int] = []
        unit_id_single = getattr(args, "unit_id", None)
        if unit_id_single is not None:
            cli_unit_ids.append(int(unit_id_single))
        if getattr(args, "unit_ids", None):
            cli_unit_ids.extend(int(value) for value in list(getattr(args, "unit_ids", []) or []))
        if cli_unit_ids and ("unit_ids" not in stage_kwargs):
            dedup_ids = list(dict.fromkeys(int(x) for x in cli_unit_ids))
            stage_kwargs["unit_ids"] = dedup_ids

        force_restart_per_unit = _resolve_bool_cfg(
            cli_value=getattr(args, "force_restart_per_unit", None),
            cfg=runtime_config,
            cfg_path="stages.reconstruct.execution.force_restart_per_unit",
            env_key="AXON_RECON_RECON_FORCE_RESTART_PER_UNIT",
            default=False,
        )
        if "force_restart_per_unit" not in stage_kwargs:
            stage_kwargs["force_restart_per_unit"] = bool(force_restart_per_unit)

        force_replot_per_unit = _resolve_bool_cfg(
            cli_value=getattr(args, "force_replot_per_unit", None),
            cfg=runtime_config,
            cfg_path="stages.reconstruct.execution.force_replot_per_unit",
            env_key="AXON_RECON_RECON_FORCE_REPLOT_PER_UNIT",
            default=False,
        )
        if "force_replot_per_unit" not in stage_kwargs:
            stage_kwargs["force_replot_per_unit"] = bool(force_replot_per_unit)

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

        write_template_movie_gif = _resolve_bool_cfg(
            cli_value=None,
            cfg=runtime_config,
            cfg_path="stages.reconstruct.write_template_movie_gif",
            env_key="AXON_RECON_RECON_WRITE_TEMPLATE_MOVIE_GIF",
            default=False,
        )
        if "write_template_movie_gif" not in stage_kwargs:
            stage_kwargs["write_template_movie_gif"] = bool(write_template_movie_gif)

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

        if bool(stage_kwargs.get("force_replot_per_unit", False)):
            # Per-unit replot loops should avoid expensive/global artifacts.
            stage_kwargs["write_top_density_grid"] = False
            stage_kwargs["replot_top_density_grid_only"] = False

            # Preserve existing reconstruction data artifacts used for plotting.
            stage_kwargs["per_unit_write_branches_raw_json"] = False
            stage_kwargs["per_unit_write_branches_json"] = False
            stage_kwargs["per_unit_write_heuristics_json"] = False

        stage_logger.info(
            "Reconstruct options: unit_ids=%s force_restart_per_unit=%s force_replot_per_unit=%s write_top_density_grid=%s replot_top_density_grid_only=%s per_unit_write_template=%s",
            str(stage_kwargs.get("unit_ids")),
            str(bool(stage_kwargs.get("force_restart_per_unit", False))).lower(),
            str(bool(stage_kwargs.get("force_replot_per_unit", False))).lower(),
            str(bool(stage_kwargs.get("write_top_density_grid", True))).lower(),
            str(bool(stage_kwargs.get("replot_top_density_grid_only", False))).lower(),
            str(bool(stage_kwargs.get("per_unit_write_template", True))).lower(),
        )

        grids_cfg = runtime_config.get("stages.reconstruct.grids", default=None)
        if grids_cfg is not None and not isinstance(grids_cfg, dict):
            raise ValueError("stages.reconstruct.grids must be a mapping/object")
        grids_cfg = grids_cfg if isinstance(grids_cfg, dict) else {}

        def _cfg_or_default(key: str, default: Any) -> Any:
            return grids_cfg.get(key, default) if isinstance(grids_cfg, dict) else default

        if "grid_output_subdir" not in stage_kwargs:
            stage_kwargs["grid_output_subdir"] = str(_cfg_or_default("output_subdir", "grids"))
        if "grid_write_pdf" not in stage_kwargs:
            stage_kwargs["grid_write_pdf"] = bool(_cfg_or_default("write_pdf", True))
        if "grid_write_png" not in stage_kwargs:
            stage_kwargs["grid_write_png"] = bool(_cfg_or_default("write_png", True))
        if "grid_write_ranking_json" not in stage_kwargs:
            stage_kwargs["grid_write_ranking_json"] = bool(_cfg_or_default("write_ranking_json", True))
        if "grid_log_basename" not in stage_kwargs:
            stage_kwargs["grid_log_basename"] = str(
                _cfg_or_default("log_basename", "raw_branch_log_footprint_top_density_grid")
            )
        if "grid_linear_basename" not in stage_kwargs:
            stage_kwargs["grid_linear_basename"] = str(
                _cfg_or_default("linear_basename", "raw_branch_linear_footprint_top_density_grid")
            )
        if "grid_ranking_filename" not in stage_kwargs:
            stage_kwargs["grid_ranking_filename"] = str(
                _cfg_or_default("ranking_filename", "raw_branch_log_footprint_top_density_ranking.json")
            )
        if "grid_ncols" not in stage_kwargs:
            stage_kwargs["grid_ncols"] = int(_cfg_or_default("ncols", 5))
        if "grid_dpi" not in stage_kwargs:
            stage_kwargs["grid_dpi"] = int(_cfg_or_default("dpi", 220))
        if "grid_draw_zoom_range_box" not in stage_kwargs:
            stage_kwargs["grid_draw_zoom_range_box"] = bool(_cfg_or_default("draw_zoom_range_box", False))
        if "grid_panel_background_color" not in stage_kwargs:
            stage_kwargs["grid_panel_background_color"] = str(_cfg_or_default("panel_background_color", "black"))
        if "grid_branch_color" not in stage_kwargs:
            stage_kwargs["grid_branch_color"] = str(_cfg_or_default("branch_color", "red"))
        if "grid_branch_outline_color" not in stage_kwargs:
            stage_kwargs["grid_branch_outline_color"] = str(_cfg_or_default("branch_outline_color", "white"))
        if "grid_node_radius_um" not in stage_kwargs:
            stage_kwargs["grid_node_radius_um"] = float(_cfg_or_default("node_radius_um", 5.0))
        if "grid_soma_node_radius_um" not in stage_kwargs:
            stage_kwargs["grid_soma_node_radius_um"] = float(_cfg_or_default("soma_node_radius_um", 10.0))
        if "grid_soma_node_color" not in stage_kwargs:
            stage_kwargs["grid_soma_node_color"] = str(_cfg_or_default("soma_node_color", "yellow"))
        if "grid_zoom_priority" not in stage_kwargs:
            stage_kwargs["grid_zoom_priority"] = str(_cfg_or_default("zoom_priority", "branches"))
        if "grid_zoom_padding_percent" not in stage_kwargs:
            stage_kwargs["grid_zoom_padding_percent"] = float(
                _cfg_or_default("zoom_padding_percent", _cfg_or_default("zoom_padding_um", 20.0))
            )
        if "grid_force_soma_centering" not in stage_kwargs:
            stage_kwargs["grid_force_soma_centering"] = bool(_cfg_or_default("force_soma_centering", False))
        soma_xy_cfg = grids_cfg.get("soma_xy_coords", None) if isinstance(grids_cfg, dict) else None
        if soma_xy_cfg is not None and not isinstance(soma_xy_cfg, dict):
            raise ValueError("stages.reconstruct.grids.soma_xy_coords must be a mapping/object")
        soma_xy_cfg = soma_xy_cfg if isinstance(soma_xy_cfg, dict) else {}

        def _soma_xy_or_default(key: str, default: Any) -> Any:
            return soma_xy_cfg.get(key, default) if isinstance(soma_xy_cfg, dict) else default

        if "grid_soma_xy_show" not in stage_kwargs:
            stage_kwargs["grid_soma_xy_show"] = bool(_soma_xy_or_default("show", False))
        if "grid_soma_xy_color" not in stage_kwargs:
            stage_kwargs["grid_soma_xy_color"] = str(_soma_xy_or_default("color", "white"))
        if "grid_soma_xy_fontsize" not in stage_kwargs:
            stage_kwargs["grid_soma_xy_fontsize"] = float(_soma_xy_or_default("fontsize", 5.0))
        if "grid_soma_xy_location" not in stage_kwargs:
            stage_kwargs["grid_soma_xy_location"] = str(_soma_xy_or_default("location", "bottom left"))
        if "grid_sort_by" not in stage_kwargs:
            stage_kwargs["grid_sort_by"] = str(_cfg_or_default("sort_by", "density"))
        if "grid_show_unit_id_in_plot" not in stage_kwargs:
            stage_kwargs["grid_show_unit_id_in_plot"] = bool(_cfg_or_default("show_unit_id_in_plot", True))
        if "grid_unit_id_fontsize" not in stage_kwargs:
            stage_kwargs["grid_unit_id_fontsize"] = float(_cfg_or_default("unit_id_fontsize", 6.0))
        if "grid_unit_id_color" not in stage_kwargs:
            stage_kwargs["grid_unit_id_color"] = str(_cfg_or_default("unit_id_color", "white"))
        if "grid_show_minimap" not in stage_kwargs:
            stage_kwargs["grid_show_minimap"] = bool(_cfg_or_default("show_minimap", True))
        if "grid_minimap_position" not in stage_kwargs:
            stage_kwargs["grid_minimap_position"] = str(_cfg_or_default("minimap_position", "bottomright"))
        if "grid_minimap_size" not in stage_kwargs:
            stage_kwargs["grid_minimap_size"] = float(_cfg_or_default("minimap_size", 0.20))
        if "grid_minimap_outline_color" not in stage_kwargs:
            stage_kwargs["grid_minimap_outline_color"] = str(_cfg_or_default("minimap_outline_color", "white"))
        if "grid_minimap_chip_width_mm" not in stage_kwargs:
            stage_kwargs["grid_minimap_chip_width_mm"] = float(_cfg_or_default("minimap_chip_width_mm", 3.85))
        if "grid_minimap_chip_height_mm" not in stage_kwargs:
            stage_kwargs["grid_minimap_chip_height_mm"] = float(_cfg_or_default("minimap_chip_height_mm", 2.10))
        if "grid_minimap_inner_box_linestyle" not in stage_kwargs:
            stage_kwargs["grid_minimap_inner_box_linestyle"] = str(
                _cfg_or_default("minimap_inner_box_linestyle", "dotted")
            )
        if "grid_minimap_inner_box_linewidth" not in stage_kwargs:
            stage_kwargs["grid_minimap_inner_box_linewidth"] = float(
                _cfg_or_default("minimap_inner_box_linewidth", 0.8)
            )
        if "grid_minimap_include_footprint" not in stage_kwargs:
            stage_kwargs["grid_minimap_include_footprint"] = bool(
                _cfg_or_default("minimap_include_footprint", False)
            )
        if "grid_minimap_prevent_occlusions" not in stage_kwargs:
            stage_kwargs["grid_minimap_prevent_occlusions"] = bool(
                _cfg_or_default("minimap_prevent_occlusions", False)
            )
        if "grid_minimap_clearance_um" not in stage_kwargs:
            stage_kwargs["grid_minimap_clearance_um"] = float(
                _cfg_or_default("minimap_clearance_um", 2.0)
            )
        if "grid_minimap_linewidth_buffer_pt" not in stage_kwargs:
            stage_kwargs["grid_minimap_linewidth_buffer_pt"] = float(
                _cfg_or_default("minimap_linewidth_buffer_pt", 0.5)
            )
        if "grid_minimap_occlusion_max_iters" not in stage_kwargs:
            stage_kwargs["grid_minimap_occlusion_max_iters"] = int(
                _cfg_or_default("minimap_occlusion_max_iters", 8)
            )
        if "grid_minimap_occlusion_growth_factor" not in stage_kwargs:
            stage_kwargs["grid_minimap_occlusion_growth_factor"] = float(
                _cfg_or_default("minimap_occlusion_growth_factor", 1.20)
            )
        legend_cfg = grids_cfg.get("legend", None) if isinstance(grids_cfg, dict) else None
        if legend_cfg is not None and not isinstance(legend_cfg, dict):
            raise ValueError("stages.reconstruct.grids.legend must be a mapping/object")
        legend_cfg = legend_cfg if isinstance(legend_cfg, dict) else {}

        def _legend_or_default(key: str, default: Any) -> Any:
            return legend_cfg.get(key, default) if isinstance(legend_cfg, dict) else default

        if "grid_legend_show" not in stage_kwargs:
            stage_kwargs["grid_legend_show"] = bool(_legend_or_default("show", False))
        if "grid_legend_location" not in stage_kwargs:
            stage_kwargs["grid_legend_location"] = str(_legend_or_default("location", "first_empty_panel"))
        if "grid_legend_fontsize" not in stage_kwargs:
            stage_kwargs["grid_legend_fontsize"] = float(_legend_or_default("fontsize", 6.0))
        if "grid_legend_fontcolor" not in stage_kwargs:
            stage_kwargs["grid_legend_fontcolor"] = str(_legend_or_default("fontcolor", "white"))
        if "grid_legend_marker_size" not in stage_kwargs:
            stage_kwargs["grid_legend_marker_size"] = float(_legend_or_default("marker_size", 3.0))
        if "grid_legend_show_nodes_in_legend" not in stage_kwargs:
            stage_kwargs["grid_legend_show_nodes_in_legend"] = bool(
                _legend_or_default("show_nodes_in_legend", True)
            )
        if "grid_legend_show_footprint_in_legend" not in stage_kwargs:
            stage_kwargs["grid_legend_show_footprint_in_legend"] = bool(
                _legend_or_default("show_footprint_in_legend", True)
            )

        local_cb_cfg = grids_cfg.get("local_color_bars", None) if isinstance(grids_cfg, dict) else None
        if local_cb_cfg is not None and not isinstance(local_cb_cfg, dict):
            raise ValueError("stages.reconstruct.grids.local_color_bars must be a mapping/object")
        local_cb_cfg = local_cb_cfg if isinstance(local_cb_cfg, dict) else {}

        def _local_cb_or_default(key: str, default: Any) -> Any:
            return local_cb_cfg.get(key, default) if isinstance(local_cb_cfg, dict) else default

        if "grid_local_color_bars_show" not in stage_kwargs:
            stage_kwargs["grid_local_color_bars_show"] = bool(_local_cb_or_default("show", False))
        if "grid_local_color_bars_location" not in stage_kwargs:
            stage_kwargs["grid_local_color_bars_location"] = str(_local_cb_or_default("location", "topright"))
        if "grid_local_color_bars_fontsize" not in stage_kwargs:
            stage_kwargs["grid_local_color_bars_fontsize"] = float(_local_cb_or_default("fontsize", 5.0))
        if "grid_local_color_bars_fontcolor" not in stage_kwargs:
            stage_kwargs["grid_local_color_bars_fontcolor"] = str(_local_cb_or_default("fontcolor", "white"))
        if "grid_local_color_bars_length_fraction" not in stage_kwargs:
            stage_kwargs["grid_local_color_bars_length_fraction"] = float(
                _local_cb_or_default("length_fraction", 0.26)
            )
        if "grid_local_color_bars_pad_fraction" not in stage_kwargs:
            stage_kwargs["grid_local_color_bars_pad_fraction"] = float(_local_cb_or_default("pad_fraction", 0.01))
        if "grid_local_color_bars_show_ticks" not in stage_kwargs:
            stage_kwargs["grid_local_color_bars_show_ticks"] = _local_cb_or_default("show_ticks", None)

        global_cb_cfg = grids_cfg.get("global_color_bar", None) if isinstance(grids_cfg, dict) else None
        if global_cb_cfg is not None and not isinstance(global_cb_cfg, dict):
            raise ValueError("stages.reconstruct.grids.global_color_bar must be a mapping/object")
        global_cb_cfg = global_cb_cfg if isinstance(global_cb_cfg, dict) else {}

        def _global_cb_or_default(key: str, default: Any) -> Any:
            return global_cb_cfg.get(key, default) if isinstance(global_cb_cfg, dict) else default

        if "grid_global_color_bar_show" not in stage_kwargs:
            stage_kwargs["grid_global_color_bar_show"] = bool(_global_cb_or_default("show", True))
        if "grid_global_color_bar_location" not in stage_kwargs:
            stage_kwargs["grid_global_color_bar_location"] = str(_global_cb_or_default("location", "topright"))
        if "grid_global_color_bar_fontsize" not in stage_kwargs:
            stage_kwargs["grid_global_color_bar_fontsize"] = float(_global_cb_or_default("fontsize", 6.0))
        if "grid_global_color_bar_fontcolor" not in stage_kwargs:
            stage_kwargs["grid_global_color_bar_fontcolor"] = str(_global_cb_or_default("fontcolor", "white"))
        if "grid_global_color_bar_length_fraction" not in stage_kwargs:
            stage_kwargs["grid_global_color_bar_length_fraction"] = float(
                _global_cb_or_default("length_fraction", 0.30)
            )
        if "grid_global_color_bar_pad_fraction" not in stage_kwargs:
            stage_kwargs["grid_global_color_bar_pad_fraction"] = float(_global_cb_or_default("pad_fraction", 0.02))
        if "grid_global_color_bar_low_color" not in stage_kwargs:
            stage_kwargs["grid_global_color_bar_low_color"] = str(_global_cb_or_default("low_color", "blue"))
        if "grid_global_color_bar_mid_color" not in stage_kwargs:
            stage_kwargs["grid_global_color_bar_mid_color"] = str(_global_cb_or_default("mid_color", "white"))
        if "grid_global_color_bar_high_color" not in stage_kwargs:
            stage_kwargs["grid_global_color_bar_high_color"] = str(_global_cb_or_default("high_color", "red"))
        if "grid_global_color_bar_force_low_value" not in stage_kwargs:
            stage_kwargs["grid_global_color_bar_force_low_value"] = _global_cb_or_default("force_low_value", None)
        if "grid_global_color_bar_force_high_value" not in stage_kwargs:
            stage_kwargs["grid_global_color_bar_force_high_value"] = _global_cb_or_default("force_high_value", None)
        if "grid_global_color_bar_show_ticks" not in stage_kwargs:
            stage_kwargs["grid_global_color_bar_show_ticks"] = _global_cb_or_default("show_ticks", None)
        if "grid_global_color_bar_percentile_low" not in stage_kwargs:
            stage_kwargs["grid_global_color_bar_percentile_low"] = float(
                _global_cb_or_default("percentile_low", 5.0)
            )
        if "grid_global_color_bar_percentile_high_linear" not in stage_kwargs:
            stage_kwargs["grid_global_color_bar_percentile_high_linear"] = float(
                _global_cb_or_default("percentile_high_linear", 99.0)
            )
        if "grid_global_color_bar_percentile_high_log" not in stage_kwargs:
            stage_kwargs["grid_global_color_bar_percentile_high_log"] = float(
                _global_cb_or_default("percentile_high_log", 99.5)
            )
        if "grid_global_color_bar_knot_anchor_values" not in stage_kwargs:
            stage_kwargs["grid_global_color_bar_knot_anchor_values"] = _global_cb_or_default(
                "knot_anchor_values", [1.0, 10.0]
            )
        if "grid_global_color_bar_knot_y1_min" not in stage_kwargs:
            stage_kwargs["grid_global_color_bar_knot_y1_min"] = float(
                _global_cb_or_default("knot_y1_min", 0.02)
            )
        if "grid_global_color_bar_knot_y1_max" not in stage_kwargs:
            stage_kwargs["grid_global_color_bar_knot_y1_max"] = float(
                _global_cb_or_default("knot_y1_max", 0.90)
            )
        if "grid_global_color_bar_knot_y2_min" not in stage_kwargs:
            stage_kwargs["grid_global_color_bar_knot_y2_min"] = float(
                _global_cb_or_default("knot_y2_min", 0.07)
            )
        if "grid_global_color_bar_knot_y2_max" not in stage_kwargs:
            stage_kwargs["grid_global_color_bar_knot_y2_max"] = float(
                _global_cb_or_default("knot_y2_max", 0.98)
            )
        if "grid_global_color_bar_knot_min_gap" not in stage_kwargs:
            stage_kwargs["grid_global_color_bar_knot_min_gap"] = float(
                _global_cb_or_default("knot_min_gap", 0.05)
            )
        if "grid_global_color_bar_linear_cap_rounding_mode" not in stage_kwargs:
            stage_kwargs["grid_global_color_bar_linear_cap_rounding_mode"] = str(
                _global_cb_or_default("linear_cap_rounding_mode", "ceil_step")
            )
        if "grid_global_color_bar_linear_cap_rounding_step" not in stage_kwargs:
            stage_kwargs["grid_global_color_bar_linear_cap_rounding_step"] = float(
                _global_cb_or_default("linear_cap_rounding_step", 10.0)
            )
        if "grid_global_color_bar_linear_cap_min_vmax" not in stage_kwargs:
            stage_kwargs["grid_global_color_bar_linear_cap_min_vmax"] = float(
                _global_cb_or_default("linear_cap_min_vmax", 11.0)
            )

        if "grid_emit_debug_logs" not in stage_kwargs:
            stage_kwargs["grid_emit_debug_logs"] = bool(_cfg_or_default("emit_debug_logs", False))

        per_unit_cfg = runtime_config.get("stages.reconstruct.per_unit_outputs", default=None)
        if per_unit_cfg is not None and not isinstance(per_unit_cfg, dict):
            raise ValueError("stages.reconstruct.per_unit_outputs must be a mapping/object")
        per_unit_cfg = per_unit_cfg if isinstance(per_unit_cfg, dict) else {}

        if "per_unit_outputs_schema" not in stage_kwargs:
            stage_kwargs["per_unit_outputs_schema"] = dict(per_unit_cfg)

        def _per_unit_cfg_or_default(key: str, default: Any) -> Any:
            return per_unit_cfg.get(key, default) if isinstance(per_unit_cfg, dict) else default

        if "per_unit_write_branches_raw_json" not in stage_kwargs:
            stage_kwargs["per_unit_write_branches_raw_json"] = bool(
                _per_unit_cfg_or_default("write_branches_raw_json", True)
            )
        if "per_unit_write_branches_json" not in stage_kwargs:
            stage_kwargs["per_unit_write_branches_json"] = bool(
                _per_unit_cfg_or_default("write_branches_json", True)
            )
        if "per_unit_write_heuristics_json" not in stage_kwargs:
            stage_kwargs["per_unit_write_heuristics_json"] = bool(
                _per_unit_cfg_or_default("write_heuristics_json", True)
            )
        if "per_unit_branches_raw_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_branches_raw_relpath"] = str(
                _per_unit_cfg_or_default("branches_raw_relpath", "branches_raw.json")
            )
        if "per_unit_branches_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_branches_relpath"] = str(
                _per_unit_cfg_or_default("branches_relpath", "branches.json")
            )
        if "per_unit_heuristics_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_heuristics_relpath"] = str(
                _per_unit_cfg_or_default("heuristics_relpath", "heuristics.json")
            )
        if "per_unit_branches_root_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_branches_root_relpath"] = str(
                _per_unit_cfg_or_default("branches_root_relpath", "branches")
            )
        if "per_unit_branches_clean_dir_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_branches_clean_dir_relpath"] = str(
                _per_unit_cfg_or_default("branches_clean_dir_relpath", "branches/clean")
            )
        if "per_unit_branches_raw_dir_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_branches_raw_dir_relpath"] = str(
                _per_unit_cfg_or_default("branches_raw_dir_relpath", "branches/raw")
            )
        if "per_unit_morphology_dir_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_morphology_dir_relpath"] = str(
                _per_unit_cfg_or_default("morphology_dir_relpath", "morphology")
            )
        if "per_unit_heuristics_dir_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_heuristics_dir_relpath"] = str(
                _per_unit_cfg_or_default("heuristics_dir_relpath", "heuristics")
            )
        if "per_unit_maps_dir_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_maps_dir_relpath"] = str(
                _per_unit_cfg_or_default("maps_dir_relpath", "maps")
            )
        if "per_unit_template_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_template_relpath"] = str(
                _per_unit_cfg_or_default("template_relpath", "template.png")
            )
        if "per_unit_template_zoom_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_template_zoom_relpath"] = str(
                _per_unit_cfg_or_default("template_zoom_relpath", "template_zoom.png")
            )
        if "per_unit_template_movie_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_template_movie_relpath"] = str(
                _per_unit_cfg_or_default("template_movie_relpath", "template_movie.gif")
            )
        if "per_unit_summary_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_summary_relpath"] = str(
                _per_unit_cfg_or_default("summary_relpath", "summary.png")
            )
        if "per_unit_summary_clean_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_summary_clean_relpath"] = str(
                _per_unit_cfg_or_default("summary_clean_relpath", "summary_clean.png")
            )
        if "per_unit_summary_raw_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_summary_raw_relpath"] = str(
                _per_unit_cfg_or_default("summary_raw_relpath", "summary_raw.png")
            )
        if "per_unit_amplitude_map_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_amplitude_map_relpath"] = str(
                _per_unit_cfg_or_default("amplitude_map_relpath", "maps/amplitude_map.png")
            )
        if "per_unit_amplitude_map_zoom_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_amplitude_map_zoom_relpath"] = str(
                _per_unit_cfg_or_default("amplitude_map_zoom_relpath", "maps/amplitude_map_zoom.png")
            )
        if "per_unit_peak_latency_map_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_peak_latency_map_relpath"] = str(
                _per_unit_cfg_or_default("peak_latency_map_relpath", "maps/peak_latency_map.png")
            )
        if "per_unit_peak_latency_map_zoom_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_peak_latency_map_zoom_relpath"] = str(
                _per_unit_cfg_or_default("peak_latency_map_zoom_relpath", "maps/peak_latency_map_zoom.png")
            )
        if "per_unit_peak_std_map_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_peak_std_map_relpath"] = str(
                _per_unit_cfg_or_default("peak_std_map_relpath", "maps/peak_std_map.png")
            )
        if "per_unit_peak_std_map_zoom_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_peak_std_map_zoom_relpath"] = str(
                _per_unit_cfg_or_default("peak_std_map_zoom_relpath", "maps/peak_std_map_zoom.png")
            )
        if "per_unit_channel_selection_detect_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_channel_selection_detect_relpath"] = str(
                _per_unit_cfg_or_default("channel_selection_detect_relpath", "maps/channel_selection_detect.png")
            )
        if "per_unit_channel_selection_kurt_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_channel_selection_kurt_relpath"] = str(
                _per_unit_cfg_or_default("channel_selection_kurt_relpath", "maps/channel_selection_kurt.png")
            )
        if "per_unit_channel_selection_delay_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_channel_selection_delay_relpath"] = str(
                _per_unit_cfg_or_default("channel_selection_delay_relpath", "maps/channel_selection_delay.png")
            )
        if "per_unit_channel_selection_all_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_channel_selection_all_relpath"] = str(
                _per_unit_cfg_or_default("channel_selection_all_relpath", "maps/channel_selection_all.png")
            )
        if "per_unit_graph_nodes_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_graph_nodes_relpath"] = str(
                _per_unit_cfg_or_default("graph_nodes_relpath", "maps/graph_nodes.png")
            )
        if "per_unit_graph_edges_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_graph_edges_relpath"] = str(
                _per_unit_cfg_or_default("graph_edges_relpath", "maps/graph_edges.png")
            )
        if "per_unit_graph_heuristics_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_graph_heuristics_relpath"] = str(
                _per_unit_cfg_or_default("graph_heuristics_relpath", "heuristics/graph_heuristics.png")
            )
        if "per_unit_morphology_pdf_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_morphology_pdf_relpath"] = str(
                _per_unit_cfg_or_default("morphology_pdf_relpath", "morphology/morphology.pdf")
            )
        if "per_unit_morphology_zoom_pdf_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_morphology_zoom_pdf_relpath"] = str(
                _per_unit_cfg_or_default("morphology_zoom_pdf_relpath", "morphology/morphology_zoom.pdf")
            )
        if "per_unit_branches_raw_clean_pdf_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_branches_raw_clean_pdf_relpath"] = str(
                _per_unit_cfg_or_default("branches_raw_clean_pdf_relpath", "branches/raw/branches_raw_clean.pdf")
            )
        if "per_unit_branches_raw_pdf_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_branches_raw_pdf_relpath"] = str(
                _per_unit_cfg_or_default("branches_raw_pdf_relpath", "branches/raw/branches_raw.pdf")
            )
        if "per_unit_branches_raw_zoom_pdf_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_branches_raw_zoom_pdf_relpath"] = str(
                _per_unit_cfg_or_default("branches_raw_zoom_pdf_relpath", "branches/raw/branches_raw_zoom.pdf")
            )
        if "per_unit_branches_clean_pdf_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_branches_clean_pdf_relpath"] = str(
                _per_unit_cfg_or_default("branches_clean_pdf_relpath", "branches/clean/branches_clean.pdf")
            )
        if "per_unit_branches_clean_zoom_pdf_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_branches_clean_zoom_pdf_relpath"] = str(
                _per_unit_cfg_or_default("branches_clean_zoom_pdf_relpath", "branches/clean/branches_clean_zoom.pdf")
            )
        if "per_unit_branch_velocities_pdf_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_branch_velocities_pdf_relpath"] = str(
                _per_unit_cfg_or_default("branch_velocities_pdf_relpath", "branches/clean/branch_velocities.pdf")
            )
        if "per_unit_branch_velocities_raw_overlay_pdf_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_branch_velocities_raw_overlay_pdf_relpath"] = str(
                _per_unit_cfg_or_default(
                    "branch_velocities_raw_overlay_pdf_relpath",
                    "branches/raw/branch_velocities_overlay.pdf",
                )
            )
        if "per_unit_branch_velocities_overlay_pdf_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_branch_velocities_overlay_pdf_relpath"] = str(
                _per_unit_cfg_or_default(
                    "branch_velocities_overlay_pdf_relpath",
                    "branches/clean/branch_velocities_overlay.pdf",
                )
            )
        if "per_unit_branch_velocity_template_relpath" not in stage_kwargs:
            stage_kwargs["per_unit_branch_velocity_template_relpath"] = str(
                _per_unit_cfg_or_default(
                    "branch_velocity_template_relpath",
                    "branches/clean/branch_{index:02d}_velocity",
                )
            )
        if "per_unit_write_template" not in stage_kwargs:
            stage_kwargs["per_unit_write_template"] = bool(
                _per_unit_cfg_or_default("write_template", True)
            )
        if "per_unit_write_template_zoom" not in stage_kwargs:
            stage_kwargs["per_unit_write_template_zoom"] = bool(
                _per_unit_cfg_or_default("write_template_zoom", False)
            )
        if "per_unit_write_template_movie" not in stage_kwargs:
            stage_kwargs["per_unit_write_template_movie"] = bool(
                _per_unit_cfg_or_default("write_template_movie", True)
            )
        if "per_unit_write_summary" not in stage_kwargs:
            stage_kwargs["per_unit_write_summary"] = bool(
                _per_unit_cfg_or_default("write_summary", True)
            )
        if "per_unit_write_summary_clean" not in stage_kwargs:
            stage_kwargs["per_unit_write_summary_clean"] = bool(
                _per_unit_cfg_or_default("write_summary_clean", True)
            )
        if "per_unit_write_summary_raw" not in stage_kwargs:
            stage_kwargs["per_unit_write_summary_raw"] = bool(
                _per_unit_cfg_or_default("write_summary_raw", True)
            )
        if "per_unit_write_amplitude_map" not in stage_kwargs:
            stage_kwargs["per_unit_write_amplitude_map"] = bool(
                _per_unit_cfg_or_default("write_amplitude_map", True)
            )
        if "per_unit_write_amplitude_map_zoom" not in stage_kwargs:
            stage_kwargs["per_unit_write_amplitude_map_zoom"] = bool(
                _per_unit_cfg_or_default("write_amplitude_map_zoom", True)
            )
        if "per_unit_write_peak_latency_map" not in stage_kwargs:
            stage_kwargs["per_unit_write_peak_latency_map"] = bool(
                _per_unit_cfg_or_default("write_peak_latency_map", True)
            )
        if "per_unit_write_peak_latency_map_zoom" not in stage_kwargs:
            stage_kwargs["per_unit_write_peak_latency_map_zoom"] = bool(
                _per_unit_cfg_or_default("write_peak_latency_map_zoom", True)
            )
        if "per_unit_write_peak_std_map" not in stage_kwargs:
            stage_kwargs["per_unit_write_peak_std_map"] = bool(
                _per_unit_cfg_or_default("write_peak_std_map", True)
            )
        if "per_unit_write_peak_std_map_zoom" not in stage_kwargs:
            stage_kwargs["per_unit_write_peak_std_map_zoom"] = bool(
                _per_unit_cfg_or_default("write_peak_std_map_zoom", True)
            )
        if "per_unit_write_channel_selection_detect" not in stage_kwargs:
            stage_kwargs["per_unit_write_channel_selection_detect"] = bool(
                _per_unit_cfg_or_default("write_channel_selection_detect", True)
            )
        if "per_unit_write_channel_selection_kurt" not in stage_kwargs:
            stage_kwargs["per_unit_write_channel_selection_kurt"] = bool(
                _per_unit_cfg_or_default("write_channel_selection_kurt", True)
            )
        if "per_unit_write_channel_selection_delay" not in stage_kwargs:
            stage_kwargs["per_unit_write_channel_selection_delay"] = bool(
                _per_unit_cfg_or_default("write_channel_selection_delay", True)
            )
        if "per_unit_write_channel_selection_all" not in stage_kwargs:
            stage_kwargs["per_unit_write_channel_selection_all"] = bool(
                _per_unit_cfg_or_default("write_channel_selection_all", True)
            )
        if "per_unit_write_graph_nodes" not in stage_kwargs:
            stage_kwargs["per_unit_write_graph_nodes"] = bool(
                _per_unit_cfg_or_default("write_graph_nodes", True)
            )
        if "per_unit_write_graph_edges" not in stage_kwargs:
            stage_kwargs["per_unit_write_graph_edges"] = bool(
                _per_unit_cfg_or_default("write_graph_edges", True)
            )
        if "per_unit_write_graph_heuristics" not in stage_kwargs:
            stage_kwargs["per_unit_write_graph_heuristics"] = bool(
                _per_unit_cfg_or_default("write_graph_heuristics", True)
            )
        if "per_unit_write_morphology_pdf" not in stage_kwargs:
            stage_kwargs["per_unit_write_morphology_pdf"] = bool(
                _per_unit_cfg_or_default("write_morphology_pdf", True)
            )
        if "per_unit_write_morphology_zoom_pdf" not in stage_kwargs:
            stage_kwargs["per_unit_write_morphology_zoom_pdf"] = bool(
                _per_unit_cfg_or_default("write_morphology_zoom_pdf", True)
            )
        if "per_unit_write_branches_raw_clean_pdf" not in stage_kwargs:
            stage_kwargs["per_unit_write_branches_raw_clean_pdf"] = bool(
                _per_unit_cfg_or_default("write_branches_raw_clean_pdf", True)
            )
        if "per_unit_write_branches_raw_pdf" not in stage_kwargs:
            stage_kwargs["per_unit_write_branches_raw_pdf"] = bool(
                _per_unit_cfg_or_default("write_branches_raw_pdf", True)
            )
        if "per_unit_write_branches_raw_zoom_pdf" not in stage_kwargs:
            stage_kwargs["per_unit_write_branches_raw_zoom_pdf"] = bool(
                _per_unit_cfg_or_default("write_branches_raw_zoom_pdf", True)
            )
        if "per_unit_write_branches_clean_pdf" not in stage_kwargs:
            stage_kwargs["per_unit_write_branches_clean_pdf"] = bool(
                _per_unit_cfg_or_default("write_branches_clean_pdf", True)
            )
        if "per_unit_write_branches_clean_zoom_pdf" not in stage_kwargs:
            stage_kwargs["per_unit_write_branches_clean_zoom_pdf"] = bool(
                _per_unit_cfg_or_default("write_branches_clean_zoom_pdf", True)
            )
        if "per_unit_write_branch_velocities_pdf" not in stage_kwargs:
            stage_kwargs["per_unit_write_branch_velocities_pdf"] = bool(
                _per_unit_cfg_or_default("write_branch_velocities_pdf", True)
            )
        if "per_unit_write_branch_velocities_raw_overlay_pdf" not in stage_kwargs:
            stage_kwargs["per_unit_write_branch_velocities_raw_overlay_pdf"] = bool(
                _per_unit_cfg_or_default("write_branch_velocities_raw_overlay_pdf", True)
            )
        if "per_unit_write_branch_velocities_overlay_pdf" not in stage_kwargs:
            stage_kwargs["per_unit_write_branch_velocities_overlay_pdf"] = bool(
                _per_unit_cfg_or_default("write_branch_velocities_overlay_pdf", True)
            )
        if "per_unit_write_branch_velocity_template" not in stage_kwargs:
            stage_kwargs["per_unit_write_branch_velocity_template"] = bool(
                _per_unit_cfg_or_default("write_branch_velocity_template", True)
            )

    if stage == "analysis":
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

    if run_multi_dataset:
        from copy import deepcopy

        runs: list[tuple[str, str, StageExecutionContext]] = []
        data_output_root = runtime_config.get("output_root", None)
        data_scratch_root = runtime_config.get("scratch_root", None)
        if data_scratch_root is None:
            data_scratch_root = runtime_config.get("scratch_output_root", None)
        if not bool(use_scratch_root_enabled):
            data_scratch_root = None
        for idx, ds in enumerate(selected_runtime_datasets, start=1):
            ds_h5_raw = ds.get("raw_data_h5_path")
            if ds_h5_raw is None:
                ds_h5_raw = ds.get("h5_path")
            if ds_h5_raw is None:
                raise SystemExit(f"Dataset #{idx}: missing raw_data_h5_path")
            ds_h5 = Path(str(ds_h5_raw)).expanduser().resolve()
            ds_id = str(ds.get("dataset_id") or f"dataset_{idx}:{ds_h5.name}")
            if not ds_h5.exists():
                raise SystemExit(f"Dataset {ds_id}: h5 path not found: {ds_h5}")

            wells_block = ds.get("wells", None)
            available_well_ids: list[str] = []
            if isinstance(wells_block, list):
                for w in wells_block:
                    if not isinstance(w, dict):
                        continue
                    wid = w.get("well_id", None)
                    if wid is None:
                        continue
                    token = str(wid).strip()
                    if token:
                        available_well_ids.append(token)

            selected_well_ids: list[str] = []
            if getattr(args, "stream_id", None) is not None:
                selected_well_ids = [str(getattr(args, "stream_id")).strip()]
            elif available_well_ids:
                selected_well_ids = list(available_well_ids)
            elif stream_id_optional is not None:
                selected_well_ids = [str(stream_id_optional)]

            if not selected_well_ids:
                raise SystemExit(
                    f"Dataset {ds_id}: no selected well_ids available (define dataset wells[].well_id or pass --stream-id)."
                )

            ds_out_root_raw = ds.get("mea_output_root")
            if ds_out_root_raw is None:
                ds_out_root_raw = data_output_root
            if ds_out_root_raw is None:
                ds_out_root_raw = mea_output_root_optional
            if ds_out_root_raw is None:
                raise SystemExit(
                    f"Dataset {ds_id}: missing mea_output_root (set top-level output_root in data config or --mea-output-root)"
                )
            ds_out_root = Path(str(ds_out_root_raw)).expanduser().resolve()

            ds_scratch_root_raw = ds.get("scratch_output_root")
            if ds_scratch_root_raw is None:
                ds_scratch_root_raw = ds.get("scratch_root")
            ds_use_scratch_root = _as_bool_token(ds.get("use_scratch_root", use_scratch_root_enabled), use_scratch_root_enabled)
            if not bool(ds_use_scratch_root):
                ds_scratch_root_raw = None
            if ds_scratch_root_raw is None:
                ds_scratch_root_raw = data_scratch_root
            if ds_scratch_root_raw is None:
                ds_scratch_root_raw = scratch_output_root_optional
            ds_scratch_root = (
                Path(str(ds_scratch_root_raw)).expanduser().resolve() if ds_scratch_root_raw is not None else None
            )
            ds_active_root = ds_scratch_root or ds_out_root

            for ds_stream in selected_well_ids:
                ds_ctx = StageExecutionContext(
                    h5_path=ds_h5,
                    stream_id=str(ds_stream),
                    mea_output_root=ds_active_root,
                    force_restart=bool(force_restart),
                    final_output_root=ds_out_root,
                    scratch_output_root=ds_scratch_root,
                    n_jobs=int(derived_stage_n_jobs),
                    sorter=str(sorter or "kilosort4"),
                    docker_image=docker_image,
                    chunk_duration=chunk_duration,
                    verbose=bool(debug_enabled),
                )
                runs.append((ds_id, str(ds_stream), ds_ctx))

        failures = 0
        if int(stage_well_workers) <= 1:
            for ds_id, ds_stream, ds_ctx in runs:
                stage_logger.info("[dataset] starting %s stream=%s", ds_id, ds_stream)
                result = execute_stage(
                    stage=stage,
                    context=ds_ctx,
                    stage_kwargs=deepcopy(stage_kwargs),
                    logger=logging.getLogger(f"axon_reconstructor.stage.{stage}.{ds_stream}"),
                )
                if stage == "preprocess":
                    n_common = result.artifacts.get("n_common_electrodes")
                    print(f"preprocess complete [{ds_id}]: stream={ds_stream} common_electrodes={n_common}")
                if stage == "spikesort":
                    print(f"spikesort complete [{ds_id}]: sorter_output={result.artifacts.get('sorter_output_dir')}")
                if stage == "waveforms":
                    print(f"waveforms complete [{ds_id}]: out_dir={result.artifacts.get('waveforms_out_dir')}")
                if stage == "templates":
                    print(f"templates complete [{ds_id}]: out_dir={result.artifacts.get('templates_out_dir')}")
                if stage == "reconstruct":
                    print(f"reconstruction complete [{ds_id}]: out_dir={result.artifacts.get('reconstruction_out_dir')}")
                if stage == "analysis":
                    print(f"analysis complete [{ds_id}]: out_dir={result.artifacts.get('analysis_out_dir')}")
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=int(stage_well_workers)) as pool:
                fut_to_meta: dict[Any, tuple[str, str]] = {}
                for ds_id, ds_stream, ds_ctx in runs:
                    fut = pool.submit(
                        execute_stage,
                        stage=stage,
                        context=ds_ctx,
                        stage_kwargs=deepcopy(stage_kwargs),
                        logger=logging.getLogger(f"axon_reconstructor.stage.{stage}.{ds_stream}"),
                    )
                    fut_to_meta[fut] = (ds_id, ds_stream)

                for fut in concurrent.futures.as_completed(fut_to_meta):
                    ds_id, ds_stream = fut_to_meta[fut]
                    try:
                        result = fut.result()
                        if stage == "preprocess":
                            n_common = result.artifacts.get("n_common_electrodes")
                            print(f"preprocess complete [{ds_id}]: stream={ds_stream} common_electrodes={n_common}")
                        if stage == "spikesort":
                            print(f"spikesort complete [{ds_id}]: sorter_output={result.artifacts.get('sorter_output_dir')}")
                        if stage == "waveforms":
                            print(f"waveforms complete [{ds_id}]: out_dir={result.artifacts.get('waveforms_out_dir')}")
                        if stage == "templates":
                            print(f"templates complete [{ds_id}]: out_dir={result.artifacts.get('templates_out_dir')}")
                        if stage == "reconstruct":
                            print(f"reconstruction complete [{ds_id}]: out_dir={result.artifacts.get('reconstruction_out_dir')}")
                        if stage == "analysis":
                            print(f"analysis complete [{ds_id}]: out_dir={result.artifacts.get('analysis_out_dir')}")
                    except Exception as e:
                        failures += 1
                        stage_logger.error("[dataset] failed %s stream=%s: %s", ds_id, ds_stream, e)

        if failures > 0:
            raise SystemExit(f"{failures} dataset run(s) failed")
        return 0

    context = StageExecutionContext(
        h5_path=h5_path,
        stream_id=stream_id,
        mea_output_root=active_output_root,
        force_restart=bool(force_restart),
        final_output_root=mea_output_root,
        scratch_output_root=scratch_output_root,
        n_jobs=int(derived_stage_n_jobs),
        sorter=str(sorter or "kilosort4"),
        docker_image=docker_image,
        chunk_duration=chunk_duration,
        verbose=bool(debug_enabled),
    )
    result = execute_stage(
        stage=stage,
        context=context,
        stage_kwargs=stage_kwargs,
        logger=logging.getLogger(f"axon_reconstructor.stage.{stage}"),
    )

    if stage == "preprocess":
        n_common = result.artifacts.get("n_common_electrodes")
        print(f"preprocess complete: stream={stream_id} common_electrodes={n_common}")
        return 0
    if stage == "spikesort":
        print(f"spikesort complete: sorter_output={result.artifacts.get('sorter_output_dir')}")
        return 0
    if stage == "waveforms":
        print(f"waveforms complete: out_dir={result.artifacts.get('waveforms_out_dir')}")
        return 0
    if stage == "templates":
        print(f"templates complete: out_dir={result.artifacts.get('templates_out_dir')}")
        return 0
    if stage == "reconstruct":
        print(f"reconstruction complete: out_dir={result.artifacts.get('reconstruction_out_dir')}")
        return 0
    if stage == "analysis":
        print(f"analysis complete: out_dir={result.artifacts.get('analysis_out_dir')}")
        return 0

    raise SystemExit(f"Unsupported stage: {stage}")


def _parse_stage_list_tokens(raw_tokens: list[str]) -> list[str]:
    text = " ".join(str(t) for t in list(raw_tokens or [])).strip()
    if not text:
        raise SystemExit("No stages provided. Example: stages templates reconstruct")

    # Accept flexible forms such as:
    #   templates reconstruct
    #   templates,reconstruct
    #   [templates, recon]
    text = text.strip().strip("[]")
    if not text:
        raise SystemExit("No stages provided. Example: stages [templates, recon]")

    prelim: list[str] = []
    for chunk in text.split(","):
        for tok in chunk.strip().split():
            if tok:
                prelim.append(tok)

    alias = {
        "pre": "preprocess",
        "prep": "preprocess",
        "sort": "spikesort",
        "spike": "spikesort",
        "spikesorting": "spikesort",
        "waveform": "waveforms",
        "template": "templates",
        "recon": "reconstruct",
        "reconstruction": "reconstruct",
        "analyse": "analysis",
        "analyze": "analysis",
    }
    canonical_order = ["preprocess", "spikesort", "waveforms", "templates", "reconstruct", "analysis"]
    valid = set(canonical_order)

    out: list[str] = []
    for raw in prelim:
        t = str(raw).strip().lower()
        t = alias.get(t, t)
        if t == "all":
            out.extend(canonical_order)
            continue
        if t not in valid:
            raise SystemExit(
                f"Unsupported stage token: {raw}. Supported: {', '.join(canonical_order)} (plus alias 'recon')."
            )
        out.append(t)

    # De-duplicate while preserving order.
    dedup: list[str] = []
    seen: set[str] = set()
    for s in out:
        if s in seen:
            continue
        dedup.append(s)
        seen.add(s)
    return dedup


def _cmd_stages(args: argparse.Namespace) -> int:
    _load_explicit_env_file(args=args)
    stage_list = _parse_stage_list_tokens(list(getattr(args, "stages", []) or []))
    logger = logging.getLogger("axon_reconstructor.stage")

    for stage_name in stage_list:
        logger.info("stages: starting %s", stage_name)
        nested_args = argparse.Namespace(**vars(args))
        nested_args.stage = stage_name
        rc = int(_cmd_stage(nested_args))
        if rc != 0:
            logger.error("stages: stage %s failed with code %d", stage_name, rc)
            return rc
        logger.info("stages: completed %s", stage_name)
    return 0


def _cmd_scope_run(args: argparse.Namespace) -> int:
    _load_explicit_env_file(args=args)

    from axon_reconstructor.pipeline.scope_config import load_scope_config, summarize_scope_config, validate_scope_config
    from axon_reconstructor.pipeline.pipeline_driver import run_scope_stage_barriers, write_scope_run_summary

    scope_config = load_scope_config(Path(args.config))
    scratch_override = _resolve_optional_path(
        cli_value=getattr(args, "scratch_output_root", None),
        env_key="AXON_RECON_SCRATCH_OUTPUT_ROOT",
    )
    if scratch_override is not None:
        scope_config = replace(scope_config, scratch_output_root=Path(scratch_override).expanduser().resolve())
    errors = validate_scope_config(scope_config)
    if errors:
        msg = "\n".join(f"- {e}" for e in errors)
        raise SystemExit(f"Invalid scope config:\n{msg}")

    debug_enabled = _resolve_bool(cli_value=getattr(args, "debug", None), env_key="AXON_RECON_DEBUG", default=False)
    _configure_cli_logging(debug_enabled=bool(debug_enabled))

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
    p_cmd.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional .env file to load before resolving command args (CLI flags override env values).",
    )
    p_cmd.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional runtime config file; used for env.python launcher fallback.",
    )
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
    p_gpu.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional .env file to load before resolving command args (CLI flags override env values).",
    )
    p_gpu.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional runtime config file; used for env.python launcher fallback.",
    )
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

    p_stages = sub.add_parser(
        "stages",
        help="Run multiple pipeline stages in sequence (e.g. 'stages [templates, recon]').",
    )
    p_stages.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional .env file to load before resolving stage args (CLI flags override env values).",
    )
    p_stages.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML/JSON runtime config file. Precedence: CLI > config > env > defaults.",
    )
    p_stages.add_argument(
        "stages",
        nargs="+",
        help="Stages list. Accepts forms like: templates reconstruct | templates,reconstruct | [templates, recon]",
    )
    add_stage_common_required_args(p_stages)
    add_stage_spikesort_args(p_stages)
    add_stage_waveforms_args(p_stages)
    add_stage_execution_args(p_stages)
    add_stage_reconstruct_args(p_stages)
    add_stage_analysis_args(p_stages)
    add_stage_kwargs_args(p_stages)
    p_stages.set_defaults(func=_cmd_stages)

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
    p_scope.add_argument(
        "--scratch-output-root",
        default=None,
        help="Optional override for scope-level scratch output root (env: AXON_RECON_SCRATCH_OUTPUT_ROOT).",
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
    p_scope_build.add_argument("--scratch-output-root", type=Path, default=None)
    p_scope_build.add_argument("--sorter", default=None)
    p_scope_build.add_argument("--docker-image", default=None)
    p_scope_build.add_argument("--recon-n-jobs", type=int, default=None)
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
