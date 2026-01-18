from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_smoke_env
from .run import SmokeRunError, run_bash_script


def _add_common_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None, help="Path to smoke_tests.toml")
    parser.add_argument(
        "--local-config",
        type=Path,
        default=None,
        help="Path to smoke_tests.local.toml (overrides defaults; typically gitignored)",
    )

    parser.add_argument("--raw-h5", dest="RAW_H5", default=None, help="Override RAW_H5")
    parser.add_argument("--out-root", dest="OUT_ROOT", default=None, help="Override OUT_ROOT")
    parser.add_argument("--mea-repo", dest="MEA_REPO", default=None, help="Override MEA_REPO")
    parser.add_argument("--axon-repo", dest="AXON_REPO", default=None, help="Override AXON_REPO")

    parser.add_argument("--sorter", dest="SORTER", default=None, help="Override SORTER")
    parser.add_argument("--n-jobs", dest="N_JOBS", default=None, help="Override N_JOBS")

    parser.add_argument(
        "--shifter-image", dest="SHIFTER_IMAGE", default=None, help="Override SHIFTER_IMAGE"
    )
    parser.add_argument(
        "--account",
        dest="GPU_SMOKE_SALLOC_ACCOUNT",
        default=None,
        help="Override GPU_SMOKE_SALLOC_ACCOUNT (Perlmutter project/account)",
    )


def _collect_overrides(args: argparse.Namespace) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for key, value in vars(args).items():
        if key.isupper() and value is not None:
            overrides[key] = str(value)
    return overrides


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="axon-recon-smoke",
        description=(
            "Run Perlmutter/NERSC smoke tests for the MEA_Analysis + axon_reconstructor workflow. "
            "This is an integration/system validation harness (not unit tests)."
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_login = subparsers.add_parser("login", help="Run all login-node smoke tests")
    _add_common_overrides(p_login)

    p_gpu = subparsers.add_parser(
        "gpu", help="Run all GPU-node smoke tests from a login node (single allocation)"
    )
    _add_common_overrides(p_gpu)

    p_all = subparsers.add_parser("all", help="Run login then GPU smoke suites")
    _add_common_overrides(p_all)

    args = parser.parse_args(argv)
    overrides = _collect_overrides(args)

    paths, env = load_smoke_env(
        config_path=getattr(args, "config", None),
        local_config_path=getattr(args, "local_config", None),
        overrides=overrides,
    )

    suite_root = paths.suite_root

    login_script = suite_root / "login_node" / "run_all_login_node_smoke_tests.sh"
    gpu_script = suite_root / "interactive_gpu_node" / "run_all_gpu_node_smoke_tests_from_login.sh"

    try:
        if args.command in {"login", "all"}:
            # Login suite requires a raw file path.
            if "RAW_H5" not in env:
                raise SmokeRunError(
                    "RAW_H5 is not set. Provide it via --raw-h5, env var RAW_H5, "
                    "or tools/smoke_tests/perlmutter/smoke_tests.local.toml."
                )
            run_bash_script(script=login_script, cwd=paths.repo_root, env_overrides=env)

        if args.command in {"gpu", "all"}:
            # GPU suite wrapper requires an account.
            if "GPU_SMOKE_SALLOC_ACCOUNT" not in env:
                raise SmokeRunError(
                    "GPU_SMOKE_SALLOC_ACCOUNT is not set. Provide it via --account, env var "
                    "GPU_SMOKE_SALLOC_ACCOUNT, or tools/smoke_tests/perlmutter/smoke_tests.local.toml."
                )
            if "RAW_H5" not in env:
                raise SmokeRunError(
                    "RAW_H5 is not set. Provide it via --raw-h5, env var RAW_H5, "
                    "or tools/smoke_tests/perlmutter/smoke_tests.local.toml."
                )
            run_bash_script(script=gpu_script, cwd=paths.repo_root, env_overrides=env)

    except SmokeRunError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    return 0
