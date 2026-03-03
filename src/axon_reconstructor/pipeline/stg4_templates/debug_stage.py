from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from axon_reconstructor.pipeline.output_paths import compute_mea_analysis_output_dir

from .replot import replot_templates_outputs_from_disk
from .runner import TemplateExtractInputs, TemplateExtractOutputs, extract_and_merge_templates


def _parse_int_or_none(raw: str) -> int | None:
    v = str(raw).strip().lower()
    if v in {"none", "null", "all"}:
        return None
    return int(v)


def _env_int_or_none(*, env: Any, name: str, default: int | None) -> int | None:
    raw = env.env_str(name, default=None)
    if raw is None:
        return default
    return _parse_int_or_none(raw)


@dataclass(frozen=True)
class TemplatesDebugConfig:
    inputs: TemplateExtractInputs
    well_out_dir: Path
    templates_out_dir: Path
    do_replot: bool
    force: bool
    rebuild_full_channels_templates: bool
    propagation_top_channels: int
    propagation_channels_per_panel: int
    propagation_channel_overlap: int
    unit_ids: Optional[list[object]]


def build_templates_debug_config(*, args: Any, env: Any) -> TemplatesDebugConfig:
    h5_path = Path(args.h5_path) if args.h5_path is not None else env.env_required_path("AXON_RECON_H5_PATH")
    stream_id = str(args.stream_id) if args.stream_id is not None else env.env_required_str("AXON_RECON_STREAM_ID")
    mea_output_root = (
        Path(args.mea_output_root)
        if args.mea_output_root is not None
        else env.env_required_path("AXON_RECON_MEA_OUTPUT_ROOT")
    )
    n_jobs = int(args.n_jobs) if args.n_jobs is not None else int(env.env_int("AXON_RECON_N_JOBS", default=8) or 8)

    include_concat = env.env_bool("AXON_RECON_INCLUDE_CONCAT", default=True) if args.include_concat is None else bool(args.include_concat)
    include_segments = env.env_bool("AXON_RECON_INCLUDE_SEGMENTS", default=True) if args.include_segments is None else bool(args.include_segments)

    plot_templates_grid_pdf = (
        env.env_bool("AXON_RECON_TEMPLATES_PLOT_GRID_PDF", default=True)
        if args.plot_templates_grid_pdf is None
        else bool(args.plot_templates_grid_pdf)
    )
    plot_multi_source_templates_pdf = (
        env.env_bool("AXON_RECON_TEMPLATES_PLOT_MULTI_SOURCE_PDF", default=True)
        if args.plot_multi_source_templates_pdf is None
        else bool(args.plot_multi_source_templates_pdf)
    )

    require_curated_units = (
        env.env_bool("AXON_RECON_TEMPLATES_REQUIRE_CURATED_UNITS", default=True)
        if args.require_curated_units is None
        else bool(args.require_curated_units)
    )

    top_channels_per_template = (
        int(env.env_int("AXON_RECON_TEMPLATES_TOP_CHANNELS_PER_TEMPLATE", default=0) or 0)
        if args.top_channels_per_template is None
        else int(args.top_channels_per_template)
    )

    template_time_upsample_factor = (
        int(env.env_int("AXON_RECON_TEMPLATES_TIME_UPSAMPLE_FACTOR", default=1) or 1)
        if args.template_time_upsample_factor is None
        else int(args.template_time_upsample_factor)
    )
    template_time_upsample_method = (
        str(env.env_str("AXON_RECON_TEMPLATES_TIME_UPSAMPLE_METHOD", default="sinc") or "sinc").strip()
        if args.template_time_upsample_method is None
        else str(args.template_time_upsample_method).strip()
    )

    if args.force_restart is not None:
        force_restart = bool(args.force_restart)
    elif bool(args.force):
        force_restart = True
    else:
        force_restart = env.env_bool("AXON_RECON_FORCE_RESTART", default=True)

    unit_limit = _env_int_or_none(
        env=env,
        name="AXON_RECON_TEMPLATES_UNIT_LIMIT",
        default=_env_int_or_none(env=env, name="AXON_RECON_UNIT_LIMIT", default=None),
    )
    if args.unit_limit is not None:
        unit_limit = _parse_int_or_none(args.unit_limit)

    unit_ids = None
    if args.unit_ids:
        parsed: list[object] = []
        for x in list(args.unit_ids):
            try:
                parsed.append(int(x))
            except Exception:
                parsed.append(x)
        unit_ids = parsed

    if bool(args.replot_from_disk):
        do_replot = True
    elif bool(args.run_templates):
        do_replot = False
    else:
        do_replot = env.env_bool("AXON_RECON_TEMPLATES_REPLOT_FROM_DISK", default=False)

    well_out_dir = compute_mea_analysis_output_dir(
        output_root=mea_output_root,
        data_file=h5_path,
        well=stream_id,
    )
    templates_out_dir = well_out_dir / "stg4_templates_outputs"

    inputs = TemplateExtractInputs(
        h5_path=h5_path,
        stream_id=stream_id,
        mea_output_root=mea_output_root,
        include_concat=include_concat,
        include_segments=include_segments,
        unit_ids=unit_ids,
        require_curated_units=bool(require_curated_units),
        plot_templates_grid_pdf=plot_templates_grid_pdf,
        plot_multi_source_templates_pdf=plot_multi_source_templates_pdf,
        top_channels_per_template=top_channels_per_template,
        template_time_upsample_factor=int(template_time_upsample_factor),
        template_time_upsample_method=str(template_time_upsample_method),
        propagation_show_electrode_ids=True,
        propagation_trace_gain=5.0,
        n_jobs=n_jobs,
        unit_limit=unit_limit,
        force_restart=force_restart,
    )

    return TemplatesDebugConfig(
        inputs=inputs,
        well_out_dir=well_out_dir,
        templates_out_dir=templates_out_dir,
        do_replot=bool(do_replot),
        force=bool(args.force),
        rebuild_full_channels_templates=bool(args.rebuild_full_channels_templates),
        propagation_top_channels=int(args.propagation_top_channels),
        propagation_channels_per_panel=int(args.propagation_channels_per_panel),
        propagation_channel_overlap=int(args.propagation_channel_overlap),
        unit_ids=unit_ids,
    )


def run_templates_debug_stage(*, config: TemplatesDebugConfig, logger: Any) -> Optional[TemplateExtractOutputs]:
    if config.do_replot:
        logger.info("Replotting templates outputs from disk: %s", config.templates_out_dir)
        replot_templates_outputs_from_disk(
            well_out_dir=config.well_out_dir,
            unit_ids=config.unit_ids,
            force=bool(config.force),
            rebuild_full_channels_templates=bool(config.rebuild_full_channels_templates),
            propagation_top_channels=int(config.propagation_top_channels),
            propagation_channels_per_panel=int(config.propagation_channels_per_panel),
            propagation_channel_overlap=int(config.propagation_channel_overlap),
        )
        logger.info("Replot complete.")
        return None

    if bool(config.inputs.force_restart) and config.templates_out_dir.exists():
        logger.info("Removing existing stg4_templates_outputs: %s", config.templates_out_dir)
        shutil.rmtree(config.templates_out_dir)
        logger.info("Removed existing stg4_templates_outputs.")

    return extract_and_merge_templates(inputs=config.inputs)
