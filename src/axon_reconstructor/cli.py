from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from axon_reconstructor import env_utils
from axon_reconstructor.pipeline.stage_cli_args import (
    add_stage_analysis_args,
    add_stage_common_required_args,
    add_stage_debug_controls,
    add_stage_execution_args,
    add_stage_kwargs_args,
    add_stage_selector_arg,
    add_stage_spikesort_args,
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

    stage = str(args.stage)
    stage_kwargs = _load_stage_kwargs(args)

    debug_enabled = _resolve_bool(cli_value=getattr(args, "debug", None), env_key="AXON_RECON_DEBUG", default=False)
    force_restart = _resolve_bool(
        cli_value=getattr(args, "force_restart", None), env_key="AXON_RECON_FORCE_RESTART", default=False
    )
    break_before_run = _resolve_bool(
        cli_value=getattr(args, "break_before_run", None), env_key="AXON_RECON_BREAK_BEFORE_RUN", default=False
    )
    n_jobs = _resolve_int(cli_value=getattr(args, "n_jobs", None), env_key="AXON_RECON_N_JOBS", default=8)
    sorter = _resolve_optional_str(
        cli_value=getattr(args, "sorter", None), env_key="AXON_RECON_SORTER", default="kilosort4"
    )
    docker_image = _resolve_optional_str(cli_value=getattr(args, "docker_image", None), env_key="AXON_RECON_DOCKER_IMAGE")
    chunk_duration = _resolve_optional_str(
        cli_value=getattr(args, "chunk_duration", None), env_key="AXON_RECON_CHUNK_DURATION"
    )
    mea_analysis_repo_root = _resolve_optional_path(
        cli_value=getattr(args, "mea_analysis_repo_root", None), env_key="AXON_RECON_MEA_ANALYSIS_REPO_ROOT"
    )
    debug_max_units = _resolve_optional_int(
        cli_value=getattr(args, "debug_max_units", None), env_key="AXON_RECON_WF_DEBUG_MAX_UNITS", default=None
    )
    debug_max_segments = _resolve_optional_int(
        cli_value=getattr(args, "debug_max_segments", None), env_key="AXON_RECON_WF_DEBUG_MAX_SEGMENTS", default=None
    )

    if bool(debug_enabled):
        logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s", force=True)

    h5_path = _resolve_required_path(cli_value=args.h5_path, env_key="AXON_RECON_H5_PATH", cli_flag="--h5-path")
    if not h5_path.exists():
        raise SystemExit(f"h5 path not found: {h5_path}")

    stream_id = _resolve_required_str(cli_value=args.stream_id, env_key="AXON_RECON_STREAM_ID", cli_flag="--stream-id")
    mea_output_root = _resolve_required_path(
        cli_value=args.mea_output_root,
        env_key="AXON_RECON_MEA_OUTPUT_ROOT",
        cli_flag="--mea-output-root",
    )

    if bool(break_before_run) and stage in {"preprocess", "spikesort"}:
        import pdb

        print(f"break-before-run enabled for stage '{stage}'", file=sys.stderr)
        pdb.set_trace()

    if stage == "preprocess":
        from axon_reconstructor.pipeline.pipeline_driver import AxonReconstructor

        recon = AxonReconstructor(
            h5_parent_dirs=[h5_path],
            mea_analysis_output_root=str(mea_output_root),
            force_restart=bool(force_restart),
        )
        multirec, common_el = recon.preprocess_for_spikesorting(
            h5_path=h5_path,
            stream_id=stream_id,
            n_jobs=int(n_jobs),
            overwrite_saved_recording=bool(force_restart),
            **stage_kwargs,
        )
        print(f"preprocess complete: stream={stream_id} common_electrodes={len(common_el)}")
        _ = multirec
        return 0

    if stage == "spikesort":
        if mea_analysis_repo_root is None:
            raise SystemExit("--mea-analysis-repo-root is required for stage 'spikesort'")

        from axon_reconstructor.pipeline.spikesorting import SpikeSortingInputs, run_spikesorting_stage

        inputs = SpikeSortingInputs(
            h5_path=h5_path,
            stream_id=stream_id,
            mea_output_root=mea_output_root,
            mea_analysis_repo_root=mea_analysis_repo_root,
            sorter=str(sorter),
            docker_image=docker_image,
            n_jobs=int(n_jobs) if n_jobs else None,
            chunk_duration=chunk_duration,
            force_restart=bool(force_restart),
            verbose=bool(debug_enabled),
            **stage_kwargs,
        )
        outputs = run_spikesorting_stage(inputs=inputs, logger=logging.getLogger("axon_reconstructor.stage.spikesort"))
        print(f"spikesort complete: sorter_output={outputs.sorter_output_dir}")
        return 0

    if stage == "waveforms":
        from axon_reconstructor.pipeline.waveforms import WaveformExtractInputs, extract_waveforms

        inputs = WaveformExtractInputs(
            h5_path=h5_path,
            stream_id=stream_id,
            mea_output_root=mea_output_root,
            sorter=str(sorter),
            n_jobs=int(n_jobs),
            force_restart=bool(force_restart),
            debug_max_units=debug_max_units,
            debug_max_segments=debug_max_segments,
            **stage_kwargs,
        )
        outputs = extract_waveforms(inputs=inputs)
        print(f"waveforms complete: out_dir={outputs.waveforms_out_dir}")
        return 0

    if stage == "templates":
        from axon_reconstructor.pipeline.templates import TemplateExtractInputs, extract_and_merge_templates

        inputs = TemplateExtractInputs(
            h5_path=h5_path,
            stream_id=stream_id,
            mea_output_root=mea_output_root,
            n_jobs=int(n_jobs),
            force_restart=bool(force_restart),
            **stage_kwargs,
        )
        outputs = extract_and_merge_templates(inputs=inputs)
        print(f"templates complete: out_dir={outputs.templates_out_dir}")
        return 0

    if stage == "reconstruct":
        from axon_reconstructor.pipeline.reconstruction import ReconstructionInputs, reconstruct_from_templates

        inputs = ReconstructionInputs(
            h5_path=h5_path,
            stream_id=stream_id,
            mea_output_root=mea_output_root,
            force_restart=bool(force_restart),
            **stage_kwargs,
        )
        outputs = reconstruct_from_templates(inputs=inputs)
        print(f"reconstruction complete: out_dir={outputs.reconstruction_out_dir}")
        return 0

    if stage == "analysis":
        from axon_reconstructor.pipeline.analysis import AnalysisInputs, analyze_units

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

        analysis_fields = {
            "h5_path": h5_path,
            "stream_id": stream_id,
            "mea_output_root": mea_output_root,
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
            "force_restart": bool(force_restart),
        }
        analysis_fields.update(stage_kwargs)

        inputs = AnalysisInputs(**analysis_fields)
        outputs = analyze_units(inputs=inputs)
        print(f"analysis complete: out_dir={outputs.analysis_out_dir}")
        return 0

    raise SystemExit(f"Unsupported stage: {stage}")


def _cmd_scope_run(args: argparse.Namespace) -> int:
    _load_explicit_env_file(args=args)

    from axon_reconstructor.pipeline.scope_config import load_scope_config, summarize_scope_config, validate_scope_config
    from axon_reconstructor.pipeline.stage_orchestrator import run_scope_stage_barriers, write_scope_run_summary

    scope_config = load_scope_config(Path(args.config))
    errors = validate_scope_config(scope_config)
    if errors:
        msg = "\n".join(f"- {e}" for e in errors)
        raise SystemExit(f"Invalid scope config:\n{msg}")

    debug_enabled = _resolve_bool(cli_value=getattr(args, "debug", None), env_key="AXON_RECON_DEBUG", default=False)
    if bool(debug_enabled):
        logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s", force=True)

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
    add_stage_selector_arg(p_stage)
    add_stage_common_required_args(p_stage)
    add_stage_spikesort_args(p_stage)
    add_stage_execution_args(p_stage)
    add_stage_debug_controls(p_stage)
    add_stage_analysis_args(p_stage)
    add_stage_kwargs_args(p_stage)
    p_stage.set_defaults(func=_cmd_stage)

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
    p_scope.add_argument("--debug", action=argparse.BooleanOptionalAction, default=None, help="Enable debug logging")
    p_scope.set_defaults(func=_cmd_scope_run)

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
