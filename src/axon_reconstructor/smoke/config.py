from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    # Python 3.11+ has tomllib in stdlib; for 3.10 we use tomli.
    try:
        import tomllib  # type: ignore[attr-defined]

        loader = tomllib
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as loader  # type: ignore[no-redef]

    with path.open("rb") as f:
        return loader.load(f)


def _deep_update(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    for key, value in overlay.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _get_nested(config: dict[str, Any], path: tuple[str, ...], default: Any = "") -> Any:
    cur: Any = config
    for part in path:
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


@dataclass(frozen=True)
class SmokePaths:
    repo_root: Path
    suite_root: Path
    default_config: Path
    default_local_config: Path


def infer_paths(cwd: Path | None = None) -> SmokePaths:
    cwd = cwd or Path.cwd()

    # Find repo root by walking up until we see pyproject.toml.
    repo_root = cwd.resolve()
    while True:
        if (repo_root / "pyproject.toml").exists():
            break
        if repo_root.parent == repo_root:
            raise RuntimeError(
                "Could not locate repo root (pyproject.toml not found). "
                "Run from within the axon_reconstructor checkout or pass --repo-root."
            )
        repo_root = repo_root.parent

    suite_root = repo_root / "tools" / "smoke_tests" / "perlmutter"
    default_config = suite_root / "smoke_tests.toml"
    default_local_config = suite_root / "smoke_tests.local.toml"

    return SmokePaths(
        repo_root=repo_root,
        suite_root=suite_root,
        default_config=default_config,
        default_local_config=default_local_config,
    )


def load_smoke_env(
    *,
    config_path: Path | None,
    local_config_path: Path | None,
    overrides: dict[str, str],
    cwd: Path | None = None,
) -> tuple[SmokePaths, dict[str, str]]:
    paths = infer_paths(cwd)

    config_path = (config_path or paths.default_config).resolve()
    local_config_path = (local_config_path or paths.default_local_config).resolve()

    config: dict[str, Any] = {}
    _deep_update(config, _read_toml(config_path))
    _deep_update(config, _read_toml(local_config_path))

    def pick(env_key: str, toml_path: tuple[str, ...], fallback: str = "") -> str:
        # 1) env var overrides both files
        import os

        env_val = os.environ.get(env_key)
        if env_val:
            return env_val

        # 2) config files
        value = _get_nested(config, toml_path, fallback)
        if value is None:
            return ""
        return str(value)

    def pick_int(env_key: str, toml_path: tuple[str, ...], fallback: int) -> str:
        val = pick(env_key, toml_path, str(fallback))
        return str(val)

    # Apply CLI overrides last.
    # (We still map via env var names to keep everything simple.)
    env: dict[str, str] = {}

    # Paths
    raw_h5 = overrides.get("RAW_H5") or pick("RAW_H5", ("paths", "raw_h5"), "")
    out_root = overrides.get("OUT_ROOT") or pick("OUT_ROOT", ("paths", "out_root"), "")
    mea_repo = overrides.get("MEA_REPO") or pick("MEA_REPO", ("paths", "mea_repo"), "")

    axon_repo = overrides.get("AXON_REPO") or pick("AXON_REPO", ("paths", "axon_repo"), "")
    if not axon_repo:
        axon_repo = str(paths.repo_root)

    maxwell_dir = overrides.get("MAXWELL_HDF5_PLUGIN_DIR") or pick(
        "MAXWELL_HDF5_PLUGIN_DIR", ("paths", "maxwell_hdf5_plugin_dir"), ""
    )

    env["RAW_H5"] = raw_h5
    env["OUT_ROOT"] = out_root
    env["MEA_REPO"] = mea_repo
    env["AXON_REPO"] = axon_repo
    if maxwell_dir:
        env["MAXWELL_HDF5_PLUGIN_DIR"] = maxwell_dir

    # Run
    env["SORTER"] = overrides.get("SORTER") or pick("SORTER", ("run", "sorter"), "kilosort4")
    env["N_JOBS"] = overrides.get("N_JOBS") or pick_int("N_JOBS", ("run", "n_jobs"), 16)

    # Shifter
    env["SHIFTER_IMAGE"] = overrides.get("SHIFTER_IMAGE") or pick(
        "SHIFTER_IMAGE", ("shifter", "image"), ""
    )
    env["SHIFTER_CONTAINER_PATH"] = overrides.get("SHIFTER_CONTAINER_PATH") or pick(
        "SHIFTER_CONTAINER_PATH", ("shifter", "container_path"), ""
    )
    env["SHIFTER_PYTHON"] = overrides.get("SHIFTER_PYTHON") or pick(
        "SHIFTER_PYTHON", ("shifter", "python"), "python3"
    )
    env["SHIFTER_LAUNCH_MODE"] = overrides.get("SHIFTER_LAUNCH_MODE") or pick(
        "SHIFTER_LAUNCH_MODE", ("shifter", "launch_mode"), "shifter_cmd"
    )
    env["SHIFTER_MODULES"] = overrides.get("SHIFTER_MODULES") or pick(
        "SHIFTER_MODULES", ("shifter", "modules"), ""
    )

    # GPU salloc
    env["GPU_SMOKE_SALLOC_ACCOUNT"] = overrides.get("GPU_SMOKE_SALLOC_ACCOUNT") or pick(
        "GPU_SMOKE_SALLOC_ACCOUNT", ("gpu_salloc", "account"), ""
    )
    env["GPU_SMOKE_SALLOC_QOS"] = overrides.get("GPU_SMOKE_SALLOC_QOS") or pick(
        "GPU_SMOKE_SALLOC_QOS", ("gpu_salloc", "qos"), "interactive"
    )
    env["GPU_SMOKE_SALLOC_CONSTRAINT"] = overrides.get("GPU_SMOKE_SALLOC_CONSTRAINT") or pick(
        "GPU_SMOKE_SALLOC_CONSTRAINT", ("gpu_salloc", "constraint"), "gpu"
    )
    env["GPU_SMOKE_SALLOC_TIME"] = overrides.get("GPU_SMOKE_SALLOC_TIME") or pick(
        "GPU_SMOKE_SALLOC_TIME", ("gpu_salloc", "time"), "00:45:00"
    )
    env["GPU_SMOKE_SALLOC_NODES"] = overrides.get("GPU_SMOKE_SALLOC_NODES") or pick_int(
        "GPU_SMOKE_SALLOC_NODES", ("gpu_salloc", "nodes"), 1
    )
    env["GPU_SMOKE_SALLOC_GPUS"] = overrides.get("GPU_SMOKE_SALLOC_GPUS") or pick_int(
        "GPU_SMOKE_SALLOC_GPUS", ("gpu_salloc", "gpus"), 1
    )
    env["GPU_SMOKE_SALLOC_CPUS_PER_TASK"] = overrides.get(
        "GPU_SMOKE_SALLOC_CPUS_PER_TASK"
    ) or pick_int("GPU_SMOKE_SALLOC_CPUS_PER_TASK", ("gpu_salloc", "cpus_per_task"), 16)

    env["GPU_SMOKE_SALLOC_EXTRA_ARGS"] = overrides.get("GPU_SMOKE_SALLOC_EXTRA_ARGS") or pick(
        "GPU_SMOKE_SALLOC_EXTRA_ARGS", ("gpu_salloc", "extra_args"), ""
    )

    # MEA_Analysis auto-update knobs (used by some scripts/entrypoints)
    env["MEA_ANALYSIS_REPO_URL"] = overrides.get("MEA_ANALYSIS_REPO_URL") or pick(
        "MEA_ANALYSIS_REPO_URL", ("mea_analysis", "repo_url"), ""
    )
    env["MEA_ANALYSIS_BRANCH"] = overrides.get("MEA_ANALYSIS_BRANCH") or pick(
        "MEA_ANALYSIS_BRANCH", ("mea_analysis", "branch"), ""
    )

    # Filter empty values (so bash config can apply its own safe defaults)
    env = {k: v for k, v in env.items() if v != ""}

    return paths, env
