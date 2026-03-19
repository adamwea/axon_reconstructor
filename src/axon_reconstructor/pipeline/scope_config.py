from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from axon_reconstructor import env_utils


STAGE_NAMES: tuple[str, ...] = (
    "preprocess",
    "spikesort",
    "unit_match",
    "merge_update",
    "waveforms",
    "templates",
    "reconstruct",
    "analysis",
)


def _read_scope_payload(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    raw = path.read_text(encoding="utf-8")

    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(raw)
    elif suffix in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore[import-not-found]
        except Exception as e:
            raise RuntimeError("YAML scope config requires PyYAML (`pip install pyyaml`).") from e
        payload = yaml.safe_load(raw)
    else:
        # Best-effort: JSON first, then YAML if available.
        try:
            payload = json.loads(raw)
        except Exception:
            try:
                import yaml  # type: ignore[import-not-found]
            except Exception as e:
                raise RuntimeError(
                    f"Unsupported scope config extension for {path}. Use .json/.yml/.yaml."
                ) from e
            payload = yaml.safe_load(raw)

    if not isinstance(payload, dict):
        raise ValueError("Scope config root must be a mapping/object")
    return payload


def _as_stage_kwargs(x: Any, *, field_name: str) -> dict[str, dict[str, Any]]:
    if x is None:
        return {}
    if not isinstance(x, dict):
        raise ValueError(f"{field_name} must be an object mapping stage->kwargs")

    out: dict[str, dict[str, Any]] = {}
    for stage, kwargs in x.items():
        stage_name = str(stage)
        if stage_name not in STAGE_NAMES:
            raise ValueError(f"Unknown stage '{stage_name}' in {field_name}. Valid stages: {', '.join(STAGE_NAMES)}")
        if kwargs is None:
            out[stage_name] = {}
            continue
        if not isinstance(kwargs, dict):
            raise ValueError(f"{field_name}.{stage_name} must be an object")
        out[stage_name] = dict(kwargs)
    return out


@dataclass(frozen=True)
class ScopeWellSpec:
    stream_id: str
    enabled: bool = True
    stage_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class ScopeDatasetSpec:
    h5_path: Path
    wells: list[ScopeWellSpec]
    dataset_id: str | None = None
    enabled: bool = True
    stage_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class ScopeConfig:
    mea_output_root: Path
    sorter: str
    docker_image: str | None
    n_jobs: int
    chunk_duration: str | None
    force_restart: bool
    per_well_parallelism: int
    fail_fast: bool
    stage_order: list[str]
    stage_kwargs: dict[str, dict[str, Any]]
    datasets: list[ScopeDatasetSpec]

    def effective_stage_order(self) -> list[str]:
        return [stage for stage in self.stage_order if stage in STAGE_NAMES]


def _parse_well(raw: dict[str, Any]) -> ScopeWellSpec:
    stream_id = raw.get("stream_id", raw.get("well"))
    if stream_id is None:
        raise ValueError("Each well entry must define 'stream_id' (or 'well')")

    return ScopeWellSpec(
        stream_id=str(stream_id),
        enabled=bool(raw.get("enabled", True)),
        stage_kwargs=_as_stage_kwargs(raw.get("stage_kwargs"), field_name="datasets[].wells[].stage_kwargs"),
    )


def _parse_dataset(raw: dict[str, Any], *, index: int) -> ScopeDatasetSpec:
    h5_path_raw = raw.get("h5_path", raw.get("raw_h5"))
    if h5_path_raw is None:
        raise ValueError(f"datasets[{index}] missing required 'h5_path'")

    wells_raw = raw.get("wells")
    if not isinstance(wells_raw, list) or len(wells_raw) == 0:
        raise ValueError(f"datasets[{index}] must provide non-empty 'wells' list")

    wells = [_parse_well(w) for w in wells_raw if isinstance(w, dict)]
    if len(wells) == 0:
        raise ValueError(f"datasets[{index}] has no valid well objects")

    return ScopeDatasetSpec(
        h5_path=Path(h5_path_raw).expanduser().resolve(),
        wells=wells,
        dataset_id=(str(raw.get("dataset_id")) if raw.get("dataset_id") is not None else None),
        enabled=bool(raw.get("enabled", True)),
        stage_kwargs=_as_stage_kwargs(raw.get("stage_kwargs"), field_name="datasets[].stage_kwargs"),
    )


def load_scope_config(path: Path) -> ScopeConfig:
    payload = _read_scope_payload(Path(path))

    mea_output_root_raw = payload.get("mea_output_root")
    if mea_output_root_raw is None:
        raise ValueError("Scope config missing required key: mea_output_root")

    stage_order_raw = payload.get("stage_order", list(STAGE_NAMES))
    if not isinstance(stage_order_raw, list) or len(stage_order_raw) == 0:
        raise ValueError("stage_order must be a non-empty list")
    stage_order = [str(s) for s in stage_order_raw]
    bad = [s for s in stage_order if s not in STAGE_NAMES]
    if bad:
        raise ValueError(f"Invalid stage names in stage_order: {bad}; valid: {list(STAGE_NAMES)}")

    datasets_raw = payload.get("datasets")
    if not isinstance(datasets_raw, list) or len(datasets_raw) == 0:
        raise ValueError("Scope config must define non-empty 'datasets' list")

    datasets = [_parse_dataset(d, index=i) for i, d in enumerate(datasets_raw) if isinstance(d, dict)]
    if len(datasets) == 0:
        raise ValueError("No valid dataset entries parsed from 'datasets'")

    cfg = ScopeConfig(
        mea_output_root=Path(mea_output_root_raw).expanduser(),
        sorter=str(payload.get("sorter", "kilosort4")),
        docker_image=(str(payload["docker_image"]) if payload.get("docker_image") is not None else None),
        n_jobs=int(payload.get("n_jobs", 8)),
        chunk_duration=(str(payload["chunk_duration"]) if payload.get("chunk_duration") is not None else None),
        force_restart=bool(payload.get("force_restart", False)),
        per_well_parallelism=max(1, int(payload.get("per_well_parallelism", 1))),
        fail_fast=bool(payload.get("fail_fast", True)),
        stage_order=stage_order,
        stage_kwargs=_as_stage_kwargs(payload.get("stage_kwargs"), field_name="stage_kwargs"),
        datasets=datasets,
    )

    return cfg


def validate_scope_config(cfg: ScopeConfig) -> list[str]:
    errors: list[str] = []

    for dataset in cfg.datasets:
        if not dataset.enabled:
            continue
        if not dataset.h5_path.exists():
            errors.append(f"dataset h5_path does not exist: {dataset.h5_path}")
        for well in dataset.wells:
            if not well.enabled:
                continue
            if not str(well.stream_id).strip():
                errors.append(f"dataset {dataset.h5_path} has empty stream_id")

    if "unit_match" in cfg.stage_order:
        idx_unit_match = cfg.stage_order.index("unit_match")
        if "spikesort" not in cfg.stage_order:
            errors.append("unit_match stage requested but spikesort stage is missing from stage_order")
        elif cfg.stage_order.index("spikesort") > idx_unit_match:
            errors.append("unit_match stage must run after spikesort")

        for downstream in ("waveforms", "templates", "reconstruct", "analysis"):
            if downstream in cfg.stage_order and cfg.stage_order.index(downstream) < idx_unit_match:
                errors.append(f"{downstream} must run after unit_match when unit_match is present")

    if "merge_update" in cfg.stage_order:
        idx_merge_update = cfg.stage_order.index("merge_update")
        if "unit_match" not in cfg.stage_order:
            errors.append("merge_update stage requested but unit_match stage is missing from stage_order")
        elif cfg.stage_order.index("unit_match") > idx_merge_update:
            errors.append("merge_update stage must run after unit_match")

        for downstream in ("waveforms", "templates", "reconstruct", "analysis"):
            if downstream in cfg.stage_order and cfg.stage_order.index(downstream) < idx_merge_update:
                errors.append(f"{downstream} must run after merge_update when merge_update is present")

    return errors


def summarize_scope_config(cfg: ScopeConfig) -> dict[str, Any]:
    datasets = 0
    wells = 0
    for d in cfg.datasets:
        if not d.enabled:
            continue
        datasets += 1
        wells += sum(1 for w in d.wells if w.enabled)

    return {
        "mea_output_root": str(cfg.mea_output_root),
        "stage_order": list(cfg.stage_order),
        "datasets_enabled": int(datasets),
        "wells_enabled": int(wells),
        "per_well_parallelism": int(cfg.per_well_parallelism),
        "fail_fast": bool(cfg.fail_fast),
    }


def _env_bool(name: str, default: bool) -> bool:
    return bool(env_utils.env_bool(name, default=default))


def _env_str(name: str, default: str | None = None) -> str | None:
    return env_utils.env_str(name, default=default)


def _env_int(name: str, default: int) -> int:
    value = env_utils.env_int(name, default=None)
    if value is None:
        return int(default)
    return int(value)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-not-found]
    except Exception as e:
        raise RuntimeError("PyYAML is required to read cross_well config (pip install pyyaml)") from e

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("cross_well config root must be an object")
    return payload


def _parse_stage_order(raw: str) -> list[str]:
    stage_order = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not stage_order:
        raise ValueError("stage_order cannot be empty")
    return stage_order


def _bool_arg(value: str | None) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def run_scope_config_build(args: argparse.Namespace) -> int:
    if args.env_file is not None:
        env_utils.load_env_file_into_os(env_file=Path(args.env_file), override_existing=False)

    cross_cfg = _read_yaml(Path(args.cross_well_config).expanduser().resolve())
    stage_order = _parse_stage_order(args.stage_order)

    mea_output_root = (
        Path(args.mea_output_root).expanduser()
        if args.mea_output_root is not None
        else Path(_env_str("AXON_RECON_MEA_OUTPUT_ROOT", "")).expanduser()
    )
    if str(mea_output_root).strip() == "." or str(mea_output_root).strip() == "":
        raise ValueError("mea_output_root must be provided via --mea-output-root or AXON_RECON_MEA_OUTPUT_ROOT")

    sorter = args.sorter or _env_str("AXON_RECON_SORTER", "kilosort4") or "kilosort4"
    docker_image = args.docker_image or _env_str("AXON_RECON_DOCKER_IMAGE", None)
    n_jobs = int(args.n_jobs) if args.n_jobs is not None else int(_env_int("AXON_RECON_N_JOBS", 8))
    chunk_duration = args.chunk_duration if args.chunk_duration is not None else _env_str("AXON_RECON_CHUNK_DURATION", None)

    force_restart = (
        _bool_arg(args.force_restart)
        if args.force_restart is not None
        else _env_bool("AXON_RECON_FORCE_RESTART", False)
    )
    fail_fast = _bool_arg(args.fail_fast) if args.fail_fast is not None else True

    datasets_raw = cross_cfg.get("datasets", [])
    datasets: list[dict[str, Any]] = []
    for dataset in datasets_raw:
        if not isinstance(dataset, dict):
            continue
        raw_h5 = dataset.get("raw_data_h5_path") or dataset.get("h5_path")
        if raw_h5 is None:
            continue

        wells_out: list[dict[str, Any]] = []
        for well in dataset.get("wells", []):
            if not isinstance(well, dict):
                continue
            well_id = well.get("well_id") or well.get("stream_id")
            if well_id is None:
                continue
            wells_out.append({"stream_id": str(well_id), "enabled": bool(well.get("enabled", True))})

        if not wells_out:
            continue

        ds: dict[str, Any] = {
            "h5_path": str(Path(raw_h5).expanduser().resolve()),
            "wells": wells_out,
            "enabled": bool(dataset.get("enabled", True)),
        }
        if dataset.get("dataset_id") is not None:
            ds["dataset_id"] = str(dataset.get("dataset_id"))
        datasets.append(ds)

    if not datasets:
        raise ValueError("No datasets/wells were parsed from cross_well config")

    stage_kwargs: dict[str, dict[str, Any]] = {}
    if args.recon_n_jobs is not None or args.recon_json_only:
        recon_kwargs: dict[str, Any] = {}
        if args.recon_n_jobs is not None:
            recon_kwargs["n_jobs"] = int(args.recon_n_jobs)
        if args.recon_json_only:
            recon_kwargs["write_unit_pdfs"] = False
            recon_kwargs["write_all_units_overview_pdf"] = False
        stage_kwargs["reconstruct"] = recon_kwargs

    scope_payload: dict[str, Any] = {
        "mea_output_root": str(mea_output_root),
        "sorter": str(sorter),
        "docker_image": docker_image,
        "n_jobs": int(n_jobs),
        "chunk_duration": chunk_duration,
        "force_restart": bool(force_restart),
        "per_well_parallelism": max(1, int(args.per_well_parallelism)),
        "fail_fast": bool(fail_fast),
        "stage_order": stage_order,
        "datasets": datasets,
    }
    if stage_kwargs:
        scope_payload["stage_kwargs"] = stage_kwargs

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scope_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_path)
    return 0
