from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any

from axon_recon.runtime_config import RuntimeConfig


@dataclass(frozen=True)
class ConsoleLoggingConfig:
    enabled: bool = True
    rich: bool = True
    level: int = logging.INFO
    blank_line_after_phase: bool = False


@dataclass(frozen=True)
class FileLoggingConfig:
    enabled: bool = True
    path: Path | None = None
    level: int = logging.INFO


@dataclass(frozen=True)
class HierarchyLoggingConfig:
    enabled: bool = True
    level: int = logging.INFO


@dataclass(frozen=True)
class ResourceUsageLoggingConfig:
    enabled: bool = False
    level: int = logging.INFO
    include_children: bool = True
    sample_interval_s: float = 0.5
    include_gpu: bool = True
    include_disk_io: bool = False
    write_to_phase_summary: bool = True
    warnings: "ResourceUsageWarningConfig" | None = None


@dataclass(frozen=True)
class ResourceUsageWarningConfig:
    enabled: bool = True
    level: int = logging.WARNING
    plan_fraction_threshold: float = 0.8
    observed_ram_warn_fraction: float = 1.25
    observed_thread_warn_fraction: float = 1.5
    underuse_fraction: float = 0.25
    underuse_observation_count: int = 5


@dataclass(frozen=True)
class PipelineLoggingConfig:
    enabled: bool
    run_id: str
    run_root: Path
    logs_dir: Path
    level: int
    console: ConsoleLoggingConfig
    structured: FileLoggingConfig
    run_log: FileLoggingConfig
    error_log: FileLoggingConfig
    dataset_logs: HierarchyLoggingConfig
    recording_logs: HierarchyLoggingConfig
    well_logs: HierarchyLoggingConfig
    phase_logs: HierarchyLoggingConfig
    summary: FileLoggingConfig
    resource_usage: ResourceUsageLoggingConfig
    include_external_stdout: bool = False
    include_external_stderr: bool = False
    capture_warnings: bool = True
    capture_uncaught_exceptions: bool = True
    multiprocessing_enabled: bool = True
    use_queue_listener: bool = False


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_bool(value: Any, default: bool) -> bool:
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


def _level(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    if isinstance(value, int):
        return int(value)
    name = str(value).strip().upper()
    if name.isdigit():
        return int(name)
    return int(getattr(logging, name, default))


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_run_id(config_path: str | Path | None) -> str:
    stem = "pipeline"
    if config_path is not None:
        try:
            stem = Path(config_path).expanduser().resolve().stem
        except Exception:
            stem = Path(str(config_path)).stem or stem
    started = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stem}-{started}"


def _resolve_path(*, run_root: Path, raw: Any, default: str) -> Path:
    token = str(raw if raw is not None else default).strip() or str(default)
    path = Path(token).expanduser()
    if path.is_absolute():
        return path
    return (run_root / path).resolve()


def _resolve_scratch_outputs_root(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    if path.name == "outputs" and path.parent.name == "axon_recon_scratch":
        return path
    if path.name == "inputs" and path.parent.name == "axon_recon_scratch":
        return path.parent / "outputs"
    if path.name == "axon_recon_scratch":
        return path / "outputs"
    return path / "axon_recon_scratch" / "outputs"


def _resolve_run_root(*, runtime_config: RuntimeConfig, config_path: str | Path | None) -> Path:
    data_raw = runtime_config.get("data", None)
    data_payload: dict[str, Any] = {}
    data_path: Path | None = None
    if data_raw is not None:
        data_path = Path(str(data_raw)).expanduser()
        if not data_path.is_absolute() and config_path is not None:
            data_path = Path(config_path).expanduser().resolve().parent / data_path
        try:
            import yaml  # type: ignore[import-not-found]

            loaded = yaml.safe_load(data_path.read_text(encoding="utf-8"))
            data_payload = dict(loaded) if isinstance(loaded, dict) else {}
        except Exception:
            data_payload = {}

    root_raw = runtime_config.get("logging.root_dir", None)
    if root_raw is not None and str(root_raw).strip():
        root_path = Path(str(root_raw)).expanduser()
        if root_path.is_absolute():
            return root_path.resolve()
        base = Path(config_path).expanduser().resolve().parent if config_path is not None else Path.cwd()
        return (base / root_path).resolve()

    output_root_raw = data_payload.get("output_root", None)
    output_root = Path(output_root_raw).expanduser() if output_root_raw is not None else Path.cwd()
    if not output_root.is_absolute() and data_path is not None:
        output_root = data_path.parent / output_root
    output_root = output_root.resolve()

    use_scratch = _as_bool(data_payload.get("use_scratch_root", False), False)
    scratch_root_raw = data_payload.get("scratch_root", None)
    if use_scratch and scratch_root_raw is not None and str(scratch_root_raw).strip():
        scratch_root = Path(str(scratch_root_raw)).expanduser()
        if not scratch_root.is_absolute() and data_path is not None:
            scratch_root = data_path.parent / scratch_root
        return _resolve_scratch_outputs_root(scratch_root)
    return output_root


def _file_config(
    *,
    block: dict[str, Any],
    run_root: Path,
    default_path: str,
    default_level: int,
    default_enabled: bool = True,
) -> FileLoggingConfig:
    enabled = _as_bool(block.get("enabled", default_enabled), default_enabled)
    return FileLoggingConfig(
        enabled=bool(enabled),
        path=_resolve_path(run_root=run_root, raw=block.get("path", None), default=default_path),
        level=_level(block.get("level", None), default_level),
    )


def _hierarchy_config(*, block: dict[str, Any], default_level: int, default_enabled: bool) -> HierarchyLoggingConfig:
    return HierarchyLoggingConfig(
        enabled=_as_bool(block.get("enabled", default_enabled), default_enabled),
        level=_level(block.get("level", None), default_level),
    )


def parse_pipeline_logging_config(
    *,
    runtime_config: RuntimeConfig,
    config_path: str | Path | None = None,
) -> PipelineLoggingConfig:
    logging_raw = runtime_config.get("logging", None)
    logging_block_defined = isinstance(logging_raw, dict)
    logging_block = _as_mapping(logging_raw)
    top_level = _level(logging_block.get("level", None), logging.INFO)
    enabled = _as_bool(logging_block.get("enabled", True), True)
    run_root = _resolve_run_root(runtime_config=runtime_config, config_path=config_path)

    run_id_raw = logging_block.get("run_id", None)
    run_id = str(run_id_raw).strip() if run_id_raw is not None and str(run_id_raw).strip() else _safe_run_id(config_path)

    console_block = _as_mapping(logging_block.get("console", {}))
    structured_block = _as_mapping(logging_block.get("structured", {}))
    run_log_block = _as_mapping(logging_block.get("run_log", {}))
    error_block = _as_mapping(logging_block.get("error_log", {}))
    summary_block = _as_mapping(logging_block.get("summary", {}))
    resource_usage_block = _as_mapping(logging_block.get("resource_usage", {}))
    resource_warning_block = _as_mapping(resource_usage_block.get("warnings", {}))

    run_log = _file_config(
        block=run_log_block,
        run_root=run_root,
        default_path="logs/pipeline.log",
        default_level=top_level,
        default_enabled=bool(logging_block_defined),
    )
    structured = _file_config(
        block=structured_block,
        run_root=run_root,
        default_path="logs/pipeline.jsonl",
        default_level=top_level,
        default_enabled=bool(logging_block_defined),
    )
    error_log = _file_config(
        block=error_block,
        run_root=run_root,
        default_path="logs/errors.log",
        default_level=logging.WARNING,
        default_enabled=bool(logging_block_defined),
    )
    summary = _file_config(
        block=summary_block,
        run_root=run_root,
        default_path="logs/summary.json",
        default_level=logging.DEBUG,
        default_enabled=bool(logging_block_defined),
    )
    logs_dir = (run_log.path.parent if run_log.path is not None else run_root / "logs").resolve()

    multiprocessing_block = _as_mapping(logging_block.get("multiprocessing", {}))

    return PipelineLoggingConfig(
        enabled=bool(enabled),
        run_id=run_id,
        run_root=run_root,
        logs_dir=logs_dir,
        level=top_level,
        console=ConsoleLoggingConfig(
            enabled=_as_bool(console_block.get("enabled", True), True),
            rich=_as_bool(console_block.get("rich", True), True),
            level=_level(console_block.get("level", None), top_level),
            blank_line_after_phase=_as_bool(console_block.get("blank_line_after_phase", False), False),
        ),
        structured=structured,
        run_log=run_log,
        error_log=error_log,
        dataset_logs=_hierarchy_config(
            block=_as_mapping(logging_block.get("dataset_logs", {})),
            default_level=top_level,
            default_enabled=bool(logging_block_defined),
        ),
        recording_logs=_hierarchy_config(
            block=_as_mapping(logging_block.get("recording_logs", {})),
            default_level=top_level,
            default_enabled=bool(logging_block_defined),
        ),
        well_logs=_hierarchy_config(
            block=_as_mapping(logging_block.get("well_logs", {})),
            default_level=top_level,
            default_enabled=bool(logging_block_defined),
        ),
        phase_logs=_hierarchy_config(
            block=_as_mapping(logging_block.get("phase_logs", {})),
            default_level=top_level,
            default_enabled=bool(logging_block_defined),
        ),
        summary=summary,
        resource_usage=ResourceUsageLoggingConfig(
            enabled=_as_bool(resource_usage_block.get("enabled", False), False),
            level=_level(resource_usage_block.get("level", None), top_level),
            include_children=_as_bool(resource_usage_block.get("include_children", True), True),
            sample_interval_s=max(0.05, _as_float(resource_usage_block.get("sample_interval_s", 0.5), 0.5)),
            include_gpu=_as_bool(resource_usage_block.get("include_gpu", True), True),
            include_disk_io=_as_bool(resource_usage_block.get("include_disk_io", False), False),
            write_to_phase_summary=_as_bool(resource_usage_block.get("write_to_phase_summary", True), True),
            warnings=ResourceUsageWarningConfig(
                enabled=_as_bool(resource_warning_block.get("enabled", True), True),
                level=_level(resource_warning_block.get("level", None), logging.WARNING),
                plan_fraction_threshold=max(
                    0.0,
                    min(1.0, _as_float(resource_warning_block.get("plan_fraction_threshold", 0.8), 0.8)),
                ),
                observed_ram_warn_fraction=max(
                    1.0,
                    _as_float(resource_warning_block.get("observed_ram_warn_fraction", 1.25), 1.25),
                ),
                observed_thread_warn_fraction=max(
                    1.0,
                    _as_float(resource_warning_block.get("observed_thread_warn_fraction", 1.5), 1.5),
                ),
                underuse_fraction=max(
                    0.0,
                    min(1.0, _as_float(resource_warning_block.get("underuse_fraction", 0.25), 0.25)),
                ),
                underuse_observation_count=max(
                    1,
                    _as_int(resource_warning_block.get("underuse_observation_count", 5), 5),
                ),
            ),
        ),
        include_external_stdout=_as_bool(logging_block.get("include_external_stdout", False), False),
        include_external_stderr=_as_bool(logging_block.get("include_external_stderr", False), False),
        capture_warnings=_as_bool(logging_block.get("capture_warnings", True), True),
        capture_uncaught_exceptions=_as_bool(logging_block.get("capture_uncaught_exceptions", True), True),
        multiprocessing_enabled=_as_bool(multiprocessing_block.get("enabled", True), True),
        use_queue_listener=_as_bool(multiprocessing_block.get("use_queue_listener", False), False),
    )