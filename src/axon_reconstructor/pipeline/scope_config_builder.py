from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from axon_reconstructor import env_utils


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

    mea_analysis_repo_root_value: str | None = None
    if args.mea_analysis_repo_root is not None:
        mea_analysis_repo_root_value = str(Path(args.mea_analysis_repo_root).expanduser().resolve())
    else:
        env_repo = _env_str("AXON_RECON_MEA_ANALYSIS_REPO_ROOT", None)
        if env_repo is not None:
            mea_analysis_repo_root_value = str(Path(env_repo).expanduser().resolve())

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
    if args.recon_unit_workers is not None or args.recon_json_only:
        recon_kwargs: dict[str, Any] = {}
        if args.recon_unit_workers is not None:
            recon_kwargs["unit_workers"] = int(args.recon_unit_workers)
        if args.recon_json_only:
            recon_kwargs["write_unit_pdfs"] = False
            recon_kwargs["write_all_units_overview_pdf"] = False
        stage_kwargs["reconstruct"] = recon_kwargs

    scope_payload: dict[str, Any] = {
        "mea_output_root": str(mea_output_root),
        "mea_analysis_repo_root": mea_analysis_repo_root_value,
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
