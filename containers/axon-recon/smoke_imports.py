from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import sys
from typing import Any

from packaging import version as packaging_version


IMPORTS: tuple[tuple[str, str], ...] = (
    ("axon_recon", "axon_recon"),
    ("spikeinterface", "spikeinterface"),
    ("kilosort", "kilosort"),
    ("mpi4py", "mpi4py"),
)


def _module_version(module: Any) -> str | None:
    value = getattr(module, "__version__", None)
    if value is None:
        return None
    return str(value)


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _check_module_import(module_name: str) -> dict[str, str | bool | None]:
    module = importlib.import_module(module_name)
    return {
        "ok": True,
        "module": module_name,
        "check": "import",
        "error": None,
        "version": _module_version(module),
    }


def _check_unitmatch_package() -> dict[str, str | bool | None]:
    module_name = "UnitMatchPy"
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise ModuleNotFoundError(f"No module named {module_name!r}", name=module_name)
    return {
        "ok": True,
        "module": module_name,
        "check": "find_spec",
        "error": None,
        "version": _distribution_version("UnitMatchPy"),
    }


def _check_kilosort_version_floor() -> dict[str, str | bool | None]:
    minimum_version = "4.0.16"
    module = importlib.import_module("kilosort")
    resolved_version = _distribution_version("kilosort") or _module_version(module)
    if resolved_version is None:
        raise RuntimeError("kilosort version could not be resolved")
    if packaging_version.parse(resolved_version) < packaging_version.parse(minimum_version):
        raise RuntimeError(
            "kilosort version floor check failed: "
            f"found {resolved_version}, need >= {minimum_version}"
        )
    return {
        "ok": True,
        "module": "kilosort",
        "check": "version_floor",
        "error": None,
        "version": resolved_version,
    }


def _check_pynvml_available() -> dict[str, str | bool | None]:
    importlib.import_module("pynvml")
    return {
        "ok": True,
        "module": "pynvml",
        "check": "import",
        "error": None,
        "version": _distribution_version("nvidia-ml-py") or _distribution_version("pynvml"),
    }


def _check_slay_pipeline_import() -> dict[str, str | bool | None]:
    runner = importlib.import_module("axon_recon.pipeline.stages.spikesort.runner")
    import_run_slay = getattr(runner, "_import_slay_run_function")
    run_slay = import_run_slay(allow_numpy_fallback=True)
    if not callable(run_slay):
        raise RuntimeError("SLAy import succeeded but run_slay is not callable")
    return {
        "ok": True,
        "module": "slay.run",
        "check": "pipeline_import_with_numpy_cupy_fallback",
        "error": None,
        "version": _distribution_version("slay"),
    }


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    allow_missing = "--allow-missing" in args
    results: dict[str, dict[str, str | bool | None]] = {}
    failures: list[str] = []
    for label, module_name in IMPORTS:
        try:
            results[label] = _check_module_import(module_name)
        except Exception as exc:
            results[label] = {
                "ok": False,
                "module": module_name,
                "check": "import",
                "error": f"{type(exc).__name__}: {exc}",
                "version": None,
            }
            failures.append(label)
    if bool(results.get("kilosort", {}).get("ok")):
        try:
            results["kilosort_version_floor"] = _check_kilosort_version_floor()
        except Exception as exc:
            results["kilosort_version_floor"] = {
                "ok": False,
                "module": "kilosort",
                "check": "version_floor",
                "error": f"{type(exc).__name__}: {exc}",
                "version": results["kilosort"].get("version"),
            }
            failures.append("kilosort_version_floor")
        try:
            results["pynvml"] = _check_pynvml_available()
        except Exception as exc:
            results["pynvml"] = {
                "ok": False,
                "module": "pynvml",
                "check": "import",
                "error": f"{type(exc).__name__}: {exc}",
                "version": _distribution_version("nvidia-ml-py") or _distribution_version("pynvml"),
            }
            failures.append("pynvml")
    special_checks = (
        ("UnitMatchPy", _check_unitmatch_package),
        ("slay", _check_slay_pipeline_import),
    )
    for label, check in special_checks:
        try:
            results[label] = check()
        except Exception as exc:
            results[label] = {
                "ok": False,
                "module": label,
                "check": getattr(check, "__name__", "special_check"),
                "error": f"{type(exc).__name__}: {exc}",
                "version": None,
            }
            failures.append(label)
    print(json.dumps(results, indent=2, sort_keys=True))
    if failures and not allow_missing:
        print(f"missing imports: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())