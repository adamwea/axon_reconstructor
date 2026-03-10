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
from axon_reconstructor.pipeline.pipeline_driver import (
    add_stage_analysis_args,
    add_stage_common_required_args,
    add_stage_debug_controls,
    add_stage_execution_args,
    add_stage_kwargs_args,
    add_stage_reconstruct_args,
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


def _resolve_optional_float(*, cli_value: float | None, env_key: str, default: float | None = None) -> float | None:
    if cli_value is not None:
        return float(cli_value)
    parsed = env_utils.env_typed(env_key, default=default)
    if parsed is None:
        return None
    try:
        return float(parsed)
    except Exception as e:
        raise SystemExit(f"Invalid float for {env_key}: {parsed!r}") from e


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

    from axon_reconstructor.pipeline.pipeline_driver import StageExecutionContext, execute_stage

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

    if stage == "spikesort":
        ks_overrides = {
            "ks_th_universal": _resolve_optional_float(
                cli_value=getattr(args, "ks_th_universal", None),
                env_key="AXON_RECON_KS_TH_UNIVERSAL",
                default=None,
            ),
            "ks_th_learned": _resolve_optional_float(
                cli_value=getattr(args, "ks_th_learned", None),
                env_key="AXON_RECON_KS_TH_LEARNED",
                default=None,
            ),
            "ks_th_single_ch": _resolve_optional_float(
                cli_value=getattr(args, "ks_th_single_ch", None),
                env_key="AXON_RECON_KS_TH_SINGLE_CH",
                default=None,
            ),
            "ks_cluster_downsampling": _resolve_optional_int(
                cli_value=getattr(args, "ks_cluster_downsampling", None),
                env_key="AXON_RECON_KS_CLUSTER_DOWNSAMPLING",
                default=None,
            ),
            "ks_nearest_chans": _resolve_optional_int(
                cli_value=getattr(args, "ks_nearest_chans", None),
                env_key="AXON_RECON_KS_NEAREST_CHANS",
                default=None,
            ),
            "ks_max_channel_distance": _resolve_optional_float(
                cli_value=getattr(args, "ks_max_channel_distance", None),
                env_key="AXON_RECON_KS_MAX_CHANNEL_DISTANCE",
                default=None,
            ),
        }
        for key, value in ks_overrides.items():
            if key not in stage_kwargs and value is not None:
                stage_kwargs[key] = value

        post_merge_4x4 = _resolve_bool(
            cli_value=None,
            env_key="AXON_RECON_SPIKESORT_POST_MERGE_4X4",
            default=False,
        )
        if "post_merge_4x4_units" not in stage_kwargs:
            stage_kwargs["post_merge_4x4_units"] = bool(post_merge_4x4)

        post_merge_block_size = _resolve_int(
            cli_value=None,
            env_key="AXON_RECON_SPIKESORT_POST_MERGE_BLOCK_SIZE_CHANNELS",
            default=4,
        )
        if "post_merge_block_size_channels" not in stage_kwargs:
            stage_kwargs["post_merge_block_size_channels"] = int(post_merge_block_size)

        post_merge_recursive = _resolve_bool(
            cli_value=None,
            env_key="AXON_RECON_SPIKESORT_POST_MERGE_RECURSIVE",
            default=True,
        )
        if "post_merge_recursive" not in stage_kwargs:
            stage_kwargs["post_merge_recursive"] = bool(post_merge_recursive)

        post_merge_max_iterations = _resolve_int(
            cli_value=None,
            env_key="AXON_RECON_SPIKESORT_POST_MERGE_MAX_ITERATIONS",
            default=8,
        )
        if "post_merge_max_iterations" not in stage_kwargs:
            stage_kwargs["post_merge_max_iterations"] = int(post_merge_max_iterations)

        post_merge_pitch_um = _resolve_optional_float(
            cli_value=None,
            env_key="AXON_RECON_SPIKESORT_POST_MERGE_CHANNEL_PITCH_UM",
            default=17.5,
        )
        if "post_merge_channel_pitch_um" not in stage_kwargs and post_merge_pitch_um is not None:
            stage_kwargs["post_merge_channel_pitch_um"] = float(post_merge_pitch_um)

    if stage == "waveforms":
        if "debug_max_units" not in stage_kwargs and debug_max_units is not None:
            stage_kwargs["debug_max_units"] = int(debug_max_units)
        if "debug_max_segments" not in stage_kwargs and debug_max_segments is not None:
            stage_kwargs["debug_max_segments"] = int(debug_max_segments)

        filter_by_maxwell_epochs = _resolve_bool(
            cli_value=None,
            env_key="AXON_RECON_WF_FILTER_BY_MAXWELL_EPOCHS",
            default=True,
        )
        if "filter_by_maxwell_epochs" not in stage_kwargs:
            stage_kwargs["filter_by_maxwell_epochs"] = bool(filter_by_maxwell_epochs)

        filter_by_segment_bounds = _resolve_bool(
            cli_value=None,
            env_key="AXON_RECON_WF_FILTER_BY_SEGMENT_BOUNDS",
            default=True,
        )
        if "filter_by_segment_bounds" not in stage_kwargs:
            stage_kwargs["filter_by_segment_bounds"] = bool(filter_by_segment_bounds)

        segment_sort_safety_cleanup = _resolve_bool(
            cli_value=None,
            env_key="AXON_RECON_WF_SEGMENT_SORT_SAFETY_CLEANUP",
            default=True,
        )
        if "segment_sort_safety_cleanup" not in stage_kwargs:
            stage_kwargs["segment_sort_safety_cleanup"] = bool(segment_sort_safety_cleanup)

        recompute_channel_groups_for_reused_segments = _resolve_bool(
            cli_value=None,
            env_key="AXON_RECON_WF_RECOMPUTE_CHANNEL_GROUPS_FOR_REUSED_SEGMENTS",
            default=False,
        )
        if "recompute_channel_groups_for_reused_segments" not in stage_kwargs:
            stage_kwargs["recompute_channel_groups_for_reused_segments"] = bool(
                recompute_channel_groups_for_reused_segments
            )

        use_merged_spikesorting_4x4 = _resolve_bool(
            cli_value=None,
            env_key="AXON_RECON_WF_USE_MERGED_SPIKESORTING_4X4",
            default=False,
        )
        if "use_merged_spikesorting_4x4" not in stage_kwargs:
            stage_kwargs["use_merged_spikesorting_4x4"] = bool(use_merged_spikesorting_4x4)

        waveforms_variant_name = _resolve_optional_str(
            cli_value=None,
            env_key="AXON_RECON_WF_VARIANT_NAME",
            default=None,
        )
        if "waveforms_variant_name" not in stage_kwargs and waveforms_variant_name is not None:
            stage_kwargs["waveforms_variant_name"] = str(waveforms_variant_name)

    if stage == "reconstruct":
        templates_variant_name = _resolve_optional_str(
            cli_value=getattr(args, "recon_templates_variant_name", None),
            env_key="AXON_RECON_RECON_TEMPLATES_VARIANT_NAME",
            default=None,
        )
        if "templates_variant_name" not in stage_kwargs and templates_variant_name is not None:
            stage_kwargs["templates_variant_name"] = str(templates_variant_name)

        reconstruction_variant_name = _resolve_optional_str(
            cli_value=getattr(args, "recon_variant_name", None),
            env_key="AXON_RECON_RECON_VARIANT_NAME",
            default=None,
        )
        if "reconstruction_variant_name" not in stage_kwargs and reconstruction_variant_name is not None:
            stage_kwargs["reconstruction_variant_name"] = str(reconstruction_variant_name)

        top_n_density_raw = getattr(args, "recon_top_n_density_requested", None)
        if top_n_density_raw is None:
            top_n_density_raw = env_utils.env_str("AXON_RECON_RECON_TOP_N_DENSITY_REQUESTED", default=None)
        top_n_density_requested = _parse_int_or_none_token(top_n_density_raw)
        if "top_n_density_requested" not in stage_kwargs and top_n_density_requested is not None:
            stage_kwargs["top_n_density_requested"] = top_n_density_requested

        write_top_density_grid = _resolve_bool(
            cli_value=getattr(args, "recon_write_top_density_grid", None),
            env_key="AXON_RECON_RECON_WRITE_TOP_DENSITY_GRID",
            default=True,
        )
        if "write_top_density_grid" not in stage_kwargs:
            stage_kwargs["write_top_density_grid"] = bool(write_top_density_grid)

        show_density_scale_debug_text = _resolve_bool(
            cli_value=getattr(args, "recon_show_density_scale_debug_text", None),
            env_key="AXON_RECON_RECON_SHOW_DENSITY_SCALE_DEBUG_TEXT",
            default=False,
        )
        if "show_density_scale_debug_text" not in stage_kwargs:
            stage_kwargs["show_density_scale_debug_text"] = bool(show_density_scale_debug_text)

        show_density_scale_global_debug_text = _resolve_bool(
            cli_value=getattr(args, "recon_show_density_scale_global_debug_text", None),
            env_key="AXON_RECON_RECON_SHOW_DENSITY_SCALE_GLOBAL_DEBUG_TEXT",
            default=False,
        )
        if "show_density_scale_global_debug_text" not in stage_kwargs:
            stage_kwargs["show_density_scale_global_debug_text"] = bool(show_density_scale_global_debug_text)

        show_density_scale_local_debug_text = _resolve_bool(
            cli_value=getattr(args, "recon_show_density_scale_local_debug_text", None),
            env_key="AXON_RECON_RECON_SHOW_DENSITY_SCALE_LOCAL_DEBUG_TEXT",
            default=False,
        )
        if "show_density_scale_local_debug_text" not in stage_kwargs:
            stage_kwargs["show_density_scale_local_debug_text"] = bool(show_density_scale_local_debug_text)

        replot_top_density_grid_only = _resolve_bool(
            cli_value=getattr(args, "recon_replot_top_density_grid_only", None),
            env_key="AXON_RECON_RECON_REPLOT_TOP_DENSITY_GRID_ONLY",
            default=False,
        )
        if "replot_top_density_grid_only" not in stage_kwargs:
            stage_kwargs["replot_top_density_grid_only"] = bool(replot_top_density_grid_only)

    if stage == "analysis":
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

        stage_kwargs.update(
            {
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
            }
        )

    context = StageExecutionContext(
        h5_path=h5_path,
        stream_id=stream_id,
        mea_output_root=mea_output_root,
        force_restart=bool(force_restart),
        n_jobs=int(n_jobs),
        sorter=str(sorter or "kilosort4"),
        docker_image=docker_image,
        chunk_duration=chunk_duration,
        mea_analysis_repo_root=mea_analysis_repo_root,
        verbose=bool(debug_enabled),
    )
    result = execute_stage(
        stage=stage,
        context=context,
        stage_kwargs=stage_kwargs,
        logger=logging.getLogger(f"axon_reconstructor.stage.{stage}"),
    )

    if stage == "preprocess":
        n_common = result.artifacts.get("n_common_electrodes")
        print(f"preprocess complete: stream={stream_id} common_electrodes={n_common}")
        return 0
    if stage == "spikesort":
        print(f"spikesort complete: sorter_output={result.artifacts.get('sorter_output_dir')}")
        return 0
    if stage == "waveforms":
        print(f"waveforms complete: out_dir={result.artifacts.get('waveforms_out_dir')}")
        return 0
    if stage == "templates":
        print(f"templates complete: out_dir={result.artifacts.get('templates_out_dir')}")
        return 0
    if stage == "reconstruct":
        print(f"reconstruction complete: out_dir={result.artifacts.get('reconstruction_out_dir')}")
        return 0
    if stage == "analysis":
        print(f"analysis complete: out_dir={result.artifacts.get('analysis_out_dir')}")
        return 0

    raise SystemExit(f"Unsupported stage: {stage}")


def _cmd_scope_run(args: argparse.Namespace) -> int:
    _load_explicit_env_file(args=args)

    from axon_reconstructor.pipeline.scope_config import load_scope_config, summarize_scope_config, validate_scope_config
    from axon_reconstructor.pipeline.pipeline_driver import run_scope_stage_barriers, write_scope_run_summary

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


def _cmd_scope_config_build(args: argparse.Namespace) -> int:
    from axon_reconstructor.pipeline.scope_config import run_scope_config_build

    return int(run_scope_config_build(args))


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
    add_stage_reconstruct_args(p_stage)
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

    p_scope_build = sub.add_parser(
        "scope-config-build",
        help="Build a scope-run JSON config from a cross-well config and env defaults.",
    )
    p_scope_build.add_argument("--cross-well-config", required=True, type=Path)
    p_scope_build.add_argument("--env-file", default=None, type=Path)
    p_scope_build.add_argument("--out", required=True, type=Path)
    p_scope_build.add_argument("--stage-order", required=True, help="Comma-separated stage order")
    p_scope_build.add_argument("--per-well-parallelism", type=int, default=1)
    p_scope_build.add_argument("--fail-fast", nargs="?", const="1", default=None)
    p_scope_build.add_argument("--force-restart", nargs="?", const="1", default=None)
    p_scope_build.add_argument("--n-jobs", type=int, default=None)
    p_scope_build.add_argument("--chunk-duration", default=None)
    p_scope_build.add_argument("--mea-output-root", type=Path, default=None)
    p_scope_build.add_argument("--mea-analysis-repo-root", type=Path, default=None)
    p_scope_build.add_argument("--sorter", default=None)
    p_scope_build.add_argument("--docker-image", default=None)
    p_scope_build.add_argument("--recon-unit-workers", type=int, default=None)
    p_scope_build.add_argument("--recon-json-only", action="store_true")
    p_scope_build.set_defaults(func=_cmd_scope_config_build)

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
