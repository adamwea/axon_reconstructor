from __future__ import annotations

import argparse
import os
from pathlib import Path


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


def _cmd_run_reconstruction(args: argparse.Namespace) -> int:
    # Import heavy pipeline only for the run subcommand.
    from axon_reconstructor.pipeline.reconstructor import AxonReconstructor

    recon = AxonReconstructor(
        h5_parent_dirs=[args.h5_parent_dir],
        mea_environment=args.mea_environment,
        mea_analysis_output_root=args.mea_output_root,
        mea_analysis_repo_root=args.mea_analysis_repo_root,
        mea_analysis_docker_image=args.docker_image,
        mea_auto_run_driver=bool(args.auto_run_driver),
    )

    recon.run_pipeline(
        concatenate_switch=bool(args.concatenate),
        sort_switch=True,
        waveform_switch=bool(args.waveforms),
        template_switch=bool(args.templates),
        recon_switch=bool(args.reconstruct),
        only_load_sortings=bool(args.only_load_sortings),
    )

    return 0


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

    p_run = sub.add_parser(
        "run",
        help="Run axon reconstruction, loading MEA_Analysis sorter outputs from --mea-output-root.",
    )
    p_run.add_argument("h5_parent_dir", help="Directory to scan for raw .h5 files.")
    _add_mea_common_flags(p_run)
    p_run.add_argument("--docker-image", default=None, help="Docker image for lab auto-run (optional).")
    p_run.add_argument(
        "--auto-run-driver",
        action="store_true",
        help="If set, may invoke MEA_Analysis driver (lab mode) when sorter_output is missing.",
    )
    p_run.add_argument("--only-load-sortings", action="store_true", help="Legacy flag; kept for compatibility.")
    p_run.add_argument("--no-concatenate", dest="concatenate", action="store_false", default=True)
    p_run.add_argument("--no-waveforms", dest="waveforms", action="store_false", default=True)
    p_run.add_argument("--no-templates", dest="templates", action="store_false", default=True)
    p_run.add_argument("--no-reconstruct", dest="reconstruct", action="store_false", default=True)
    p_run.set_defaults(func=_cmd_run_reconstruction)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
