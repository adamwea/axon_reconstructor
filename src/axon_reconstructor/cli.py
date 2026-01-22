from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
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


def _cmd_run_reconstruction(args: argparse.Namespace) -> int:
    # Import heavy pipeline only for the run subcommand.
    from axon_reconstructor.pipeline.pipeline_driver import AxonReconstructor

    recon = AxonReconstructor(
        h5_parent_dirs=[args.h5_parent_dir],
        mea_environment=args.mea_environment,
        mea_analysis_output_root=args.mea_output_root,
        mea_analysis_repo_root=args.mea_analysis_repo_root,
        mea_analysis_docker_image=args.docker_image,
        mea_auto_run_driver=bool(args.auto_run_driver),
        force_restart=bool(getattr(args, "force_restart", False)),
        enable_checkpointing=bool(getattr(args, "enable_checkpointing", True)),
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


def _cmd_pipeline(args: argparse.Namespace) -> int:
    """Run the rebuilt pipeline on a targeted raw dataset.

    This command is intended for iterative development/debugging of the new
    preprocessing + spikesorting preparation steps.
    """

    from axon_reconstructor.pipeline.pipeline_driver import AxonReconstructor

    data_path = Path(args.data_path).expanduser().resolve()
    if not data_path.exists():
        raise SystemExit(f"data_path not found: {data_path}")

    recon = AxonReconstructor(
        h5_parent_dirs=[data_path],
        mea_environment=args.mea_environment,
        mea_analysis_output_root=args.mea_output_root,
        mea_analysis_repo_root=args.mea_analysis_repo_root,
        mea_analysis_docker_image=args.docker_image,
        mea_auto_run_driver=bool(args.auto_run_driver),
        force_restart=bool(getattr(args, "force_restart", False)),
        enable_checkpointing=bool(getattr(args, "enable_checkpointing", True)),
    )

    if args.list_streams:
        try:
            import h5py
        except Exception as e:
            raise SystemExit("--list-streams requires h5py") from e

        if not data_path.is_file():
            raise SystemExit("--list-streams requires data_path to be a single .h5 file")
        with h5py.File(data_path, "r") as h5:
            streams = list(h5["wells"].keys())
        for s in streams:
            print(s)
        return 0

    if args.stream_id:
        if not data_path.is_file():
            raise SystemExit("--stream-id requires data_path to be a single .h5 file")

        recon.preprocess_for_spikesorting(
            h5_path=data_path,
            stream_id=args.stream_id,
            n_jobs=int(args.n_jobs) if args.n_jobs else 8,
        )
        return 0

    # Fallback: run the driver entrypoint (currently WIP for downstream steps).
    recon.run_pipeline(
        concatenate_switch=bool(args.concatenate),
        sort_switch=bool(args.sort),
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
    p_run.add_argument(
        "--force-restart",
        action="store_true",
        help="Ignore existing axon_reconstructor checkpoint state (re-run stages).",
    )
    p_run.add_argument(
        "--no-checkpoint",
        dest="enable_checkpointing",
        action="store_false",
        default=True,
        help="Disable axon_reconstructor JSON checkpointing.",
    )
    p_run.add_argument("--only-load-sortings", action="store_true", help="Legacy flag; kept for compatibility.")
    p_run.add_argument("--no-concatenate", dest="concatenate", action="store_false", default=True)
    p_run.add_argument("--no-waveforms", dest="waveforms", action="store_false", default=True)
    p_run.add_argument("--no-templates", dest="templates", action="store_false", default=True)
    p_run.add_argument("--no-reconstruct", dest="reconstruct", action="store_false", default=True)
    p_run.set_defaults(func=_cmd_run_reconstruction)

    p_pipe = sub.add_parser(
        "pipeline",
        help=(
            "Run the rebuilt pipeline on a targeted dataset (development/debug command; "
            "focuses on preprocessing + spikesorting preparation)."
        ),
    )
    p_pipe.add_argument("data_path", help="Path to an MEA .raw.h5 file or a directory containing .h5 files.")
    _add_mea_common_flags(p_pipe)
    p_pipe.add_argument("--docker-image", default=None, help="Docker image for lab auto-run (optional).")
    p_pipe.add_argument(
        "--auto-run-driver",
        action="store_true",
        help="If set, may invoke MEA_Analysis driver (lab mode) when sorter_output is missing.",
    )
    p_pipe.add_argument(
        "--force-restart",
        action="store_true",
        help="Ignore existing axon_reconstructor checkpoint state (re-run stages).",
    )
    p_pipe.add_argument(
        "--no-checkpoint",
        dest="enable_checkpointing",
        action="store_false",
        default=True,
        help="Disable axon_reconstructor JSON checkpointing.",
    )
    p_pipe.add_argument("--only-load-sortings", action="store_true")
    p_pipe.add_argument("--no-concatenate", dest="concatenate", action="store_false", default=True)
    p_pipe.add_argument("--no-sort", dest="sort", action="store_false", default=True)
    p_pipe.add_argument("--no-waveforms", dest="waveforms", action="store_false", default=True)
    p_pipe.add_argument("--no-templates", dest="templates", action="store_false", default=True)
    p_pipe.add_argument("--no-reconstruct", dest="reconstruct", action="store_false", default=True)
    p_pipe.add_argument(
        "--stream-id",
        default=None,
        help="If set, runs preprocessing for a specific well/stream (e.g. well000) and exits.",
    )
    p_pipe.add_argument(
        "--list-streams",
        action="store_true",
        help="Print available stream ids (wells) in the given .h5 and exit.",
    )
    p_pipe.add_argument("--n-jobs", type=int, default=None)
    p_pipe.set_defaults(func=_cmd_pipeline)

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
