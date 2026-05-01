from __future__ import annotations

import importlib
import json
import sys
from typing import Any


IMPORTS: tuple[tuple[str, str], ...] = (
    ("axon_recon", "axon_recon"),
    ("spikeinterface", "spikeinterface"),
    ("kilosort", "kilosort"),
    ("UnitMatchPy", "UnitMatchPy"),
    ("slay", "slay"),
    ("mpi4py", "mpi4py"),
)


def _module_version(module: Any) -> str | None:
    value = getattr(module, "__version__", None)
    if value is None:
        return None
    return str(value)


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    allow_missing = "--allow-missing" in args
    results: dict[str, dict[str, str | bool | None]] = {}
    failures: list[str] = []
    for label, module_name in IMPORTS:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            results[label] = {
                "ok": False,
                "module": module_name,
                "error": f"{type(exc).__name__}: {exc}",
                "version": None,
            }
            failures.append(label)
            continue
        results[label] = {
            "ok": True,
            "module": module_name,
            "error": None,
            "version": _module_version(module),
        }
    print(json.dumps(results, indent=2, sort_keys=True))
    if failures and not allow_missing:
        print(f"missing imports: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())