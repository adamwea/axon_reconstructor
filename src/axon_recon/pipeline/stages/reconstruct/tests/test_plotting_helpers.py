from __future__ import annotations

import numpy as np  # type: ignore[import-not-found]

from axon_recon.pipeline.stages.reconstruct.plotting.helpers import reconstruct_colorbar_bounds
from axon_recon.pipeline.stages.reconstruct.plotting.helpers import reconstruct_colorbar_ticks
from axon_recon.pipeline.stages.reconstruct.plotting.helpers import reconstruct_corner_location
from axon_recon.pipeline.stages.reconstruct.plotting.helpers import reconstruct_draw_footprint_squares
from axon_recon.pipeline.stages.reconstruct.plotting.helpers import reconstruct_prepare_mapping
from axon_recon.pipeline.stages.reconstruct.plotting.helpers import reconstruct_tick_spec
from axon_recon.pipeline.stages.reconstruct.plotting.helpers import reconstruct_value_limits


def test_reconstruct_tick_spec_parses_dynamic_tokens() -> None:
    parsed = reconstruct_tick_spec("1,10,top")
    assert parsed == [1.0, 10.0, "dynamic_high"]


def test_reconstruct_colorbar_ticks_emits_dynamic_high_label() -> None:
    ticks, labels = reconstruct_colorbar_ticks(
        tick_spec=[1, 10, "dynamic_high"],
        vmin=0,
        vmax=12,
        detected_max=20,
    )
    assert ticks == [1.0, 10.0, 12.0]
    assert labels == ["1", "10", ">12"]


def test_reconstruct_colorbar_ticks_accepts_detected_amp_max_alias() -> None:
    ticks, labels = reconstruct_colorbar_ticks(
        tick_spec=["dynamic_high"],
        vmin=0,
        vmax=5,
        detected_amp_max=8,
    )
    assert ticks == [5.0]
    assert labels == [">5"]


def test_reconstruct_corner_location_normalizes_aliases() -> None:
    assert reconstruct_corner_location("top right") == "topright"


def test_reconstruct_colorbar_bounds_bottom_left() -> None:
    left, bottom, width, height = reconstruct_colorbar_bounds(
        location="bottom left",
        length_fraction=0.3,
        pad_fraction=0.02,
    )
    assert left < 0.2
    assert bottom < 0.2
    assert width > 0.0
    assert height > 0.0


def test_reconstruct_value_limits_linear() -> None:
    vals = np.asarray([0.1, 1.0, 2.0, 4.0], dtype=float)
    vmin, vmax = reconstruct_value_limits(
        values=vals,
        scale="linear",
        percentile_low=5.0,
        percentile_high_linear=99.0,
        percentile_high_log=99.5,
        linear_cap_rounding_mode="ceil_step",
        linear_cap_rounding_step=1.0,
    )
    assert vmin <= vmax
    assert np.isclose(vmax % 1.0, 0.0)


def test_reconstruct_prepare_mapping_log_clips_values() -> None:
    vals = np.asarray([-1.0, 0.0, 0.2, 1.0], dtype=float)
    mapped, norm, lo, hi = reconstruct_prepare_mapping(values=vals, scale="log", vmin=0.0, vmax=1.0)
    assert norm is not None
    assert lo > 0.0
    assert hi > lo
    assert np.min(mapped) >= lo


def test_reconstruct_draw_footprint_squares_returns_norm_bundle() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]

    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111)
    locs = np.asarray([[0.0, 0.0], [17.5, 0.0], [35.0, 0.0]], dtype=float)
    amp = np.asarray([0.5, 2.0, 4.0], dtype=float)

    result = reconstruct_draw_footprint_squares(
        ax,
        locs_xy=locs,
        amp=amp,
        scale_mode="linear",
    )
    assert result is not None
    assert "norm" in result
    assert "cmap" in result
    assert result["vmax"] >= result["vmin"]
    plt.close(fig)
