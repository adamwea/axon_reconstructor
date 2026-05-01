from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from axon_recon.pipeline.stages.reconstruct.config import load_reconstruction_inputs_from_runtime
from axon_recon.pipeline.stages.reconstruct.templates.config import load_templates_inputs_from_runtime


def _write_common_data_config(tmp_path: Path) -> Path:
    data_path = tmp_path / "data.yml"
    data_path.write_text(
        dedent(
            """
            output_root: /tmp/out
            datasets:
              - raw_data_h5_path: /tmp/input.raw.h5
                include_in_runtime: true
                wells:
                  - well_id: well001
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return data_path


def test_reconstruct_amplitude_map_precedence_output_over_stage_over_global(tmp_path: Path) -> None:
    data_path = _write_common_data_config(tmp_path)
    runtime_path = tmp_path / "runtime.yml"
    runtime_path.write_text(
        dedent(
            f"""
            data: {data_path}
            global_heatmap_defaults:
              default:
                color_bar:
                  scale: log
                  low_color: navy
                  show_ticks: [1, 5, dynamic_high]
            stages:
              reconstruct:
                outputs:
                  output_rel_root: recon_outputs
                  amplitude_map:
                    write_png: true
                    relpath: maps/stage_amp
                    color_bar:
                      low_color: cyan
                      location: bottomleft
                  per_unit_outputs:
                    amplitude_map:
                      color_bar:
                        show_ticks: [2, 4, dynamic_high]
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    inputs = load_reconstruction_inputs_from_runtime(config_path=str(runtime_path))
    heat = inputs.per_unit_outputs.amplitude_map_heatmap

    # Derived from stage-level amplitude_map write_png + relpath because explicit per-unit write/path absent.
    assert inputs.per_unit_outputs.write_amplitude_map_png is True
    assert inputs.per_unit_outputs.amplitude_map_png_relpath == "maps/stage_amp.png"

    # Global defaults are inherited when not overridden.
    assert heat.scale == "log"

    # Stage-level override applies.
    assert heat.low_color == "cyan"
    assert heat.colorbar_location == "bottomleft"

    # Output-level (per-unit) override applies over stage/global.
    assert heat.show_ticks == (2, 4, "dynamic_high")


def test_templates_footprint_precedence_stage_over_global(tmp_path: Path) -> None:
    data_path = _write_common_data_config(tmp_path)
    runtime_path = tmp_path / "runtime.yml"
    runtime_path.write_text(
        dedent(
            f"""
            data: {data_path}
            global_heatmap_defaults:
              default:
                background: white
                color_bar:
                  show: false
              footprint_plots:
                amplitude_map:
                  scale: log
            stages:
              templates:
                outputs:
                  per_unit_outputs:
                    footprint_plots:
                      amplitude_map:
                        color_map: viridis
                        color_bar:
                          show: true
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    inputs = load_templates_inputs_from_runtime(config_path=str(runtime_path))
    amp = inputs.per_unit_outputs.footprint_plots.amplitude_map

    assert amp.background == "white"
    assert amp.scale == "log"
    assert amp.color_map == "viridis"
    assert amp.show_color_bar is True
