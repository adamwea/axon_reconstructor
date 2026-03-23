from __future__ import annotations

import numpy as np  # type: ignore[import-not-found]

from axon_recon.pipeline.shared.plotting import parse_show_ticks_spec
from axon_recon.pipeline.shared.plotting import normalize_corner_location
from axon_recon.pipeline.shared.plotting import colorbar_axes_bounds
from axon_recon.pipeline.shared.plotting import compute_value_limits
from axon_recon.pipeline.shared.plotting import build_footprint_norm_and_cmap
from axon_recon.pipeline.shared.plotting import resolve_colorbar_ticks
from axon_recon.pipeline.shared.plotting import prepare_linear_or_log_mapping
from axon_recon.pipeline.shared.plotting import ticks_ending_in_0_or_5_with_max


def test_parse_show_ticks_spec_normalizes_dynamic_high_tokens() -> None:
    parsed = parse_show_ticks_spec([1, "10", "top"])
    assert parsed == [1.0, 10.0, "dynamic_high"]


def test_resolve_colorbar_ticks_marks_clipped_dynamic_high() -> None:
    ticks, labels = resolve_colorbar_ticks(
        tick_spec=[1, 10, "dynamic_high"],
        vmin=0,
        vmax=12,
        detected_amp_max=20,
    )
    assert ticks == [1.0, 10.0, 12.0]
    assert labels == ["1", "10", ">12"]


def test_ticks_ending_in_0_or_5_with_max_includes_exact_max() -> None:
    ticks = ticks_ending_in_0_or_5_with_max(vmin=0.001, vmax=0.023, decimal_places=3, target_count=6)
    assert np.isclose(float(ticks[-1]), 0.023)
    for t in ticks[:-1]:
        thousandths_digit = int(round(abs(float(t)) * 1000.0)) % 10
        assert thousandths_digit in (0, 5)


def test_normalize_corner_location_aliases() -> None:
    assert normalize_corner_location("top right") == "topright"
    assert normalize_corner_location("bottom_left") == "bottomleft"


def test_colorbar_axes_bounds_places_bottom_left() -> None:
    left, bottom, width, height = colorbar_axes_bounds(
        location="bottomleft",
        length_fraction=0.3,
        pad_fraction=0.02,
    )
    assert left < 0.2
    assert bottom < 0.2
    assert width > 0.0
    assert height > 0.0


def test_compute_value_limits_linear_applies_ceil_step_rounding() -> None:
    vals = np.asarray([0.5, 2.0, 4.0, 5.5, 9.2], dtype=float)
    vmin, vmax = compute_value_limits(
        values=vals,
        scale="linear",
        percentile_low=5.0,
        percentile_high_linear=99.0,
        percentile_high_log=99.5,
        linear_cap_rounding_mode="ceil_step",
        linear_cap_rounding_step=2.0,
        linear_cap_min_vmax=4.0,
    )
    assert vmin <= vmax
    assert np.isclose(vmax % 2.0, 0.0)


def test_compute_value_limits_log_uses_positive_percentiles() -> None:
    vals = np.asarray([-2.0, 0.0, 0.1, 1.0, 10.0], dtype=float)
    vmin, vmax = compute_value_limits(
        values=vals,
        scale="log",
        percentile_low=5.0,
        percentile_high_linear=99.0,
        percentile_high_log=99.5,
        positive_only_for_percentiles=True,
    )
    assert vmin > 0.0
    assert vmax > vmin


def test_prepare_linear_or_log_mapping_linear_returns_no_norm() -> None:
    vals = np.asarray([0.0, 1.0, 2.0], dtype=float)
    mapped, norm, vmin_eff, vmax_eff = prepare_linear_or_log_mapping(
        values=vals,
        scale="linear",
        vmin=0.0,
        vmax=2.0,
    )
    np.testing.assert_allclose(mapped, vals)
    assert norm is None
    assert vmin_eff == 0.0
    assert vmax_eff == 2.0


def test_prepare_linear_or_log_mapping_log_clips_nonpositive_values() -> None:
    vals = np.asarray([-1.0, 0.0, 0.01, 1.0], dtype=float)
    mapped, norm, vmin_eff, vmax_eff = prepare_linear_or_log_mapping(
        values=vals,
        scale="log",
        vmin=0.0,
        vmax=1.0,
    )
    assert norm is not None
    assert vmin_eff > 0.0
    assert vmax_eff > vmin_eff
    assert np.min(mapped) >= vmin_eff


def test_build_footprint_norm_and_cmap_linear_returns_finite_limits() -> None:
    norm, cmap, amp_render, vmin, vmax = build_footprint_norm_and_cmap(
        amp=np.asarray([0.0, 1.0, 2.0, 5.0], dtype=float),
        scale_mode="linear",
    )
    assert cmap is not None
    assert norm is not None
    assert np.isfinite(vmin)
    assert np.isfinite(vmax)
    assert vmax > vmin
    assert np.max(amp_render) >= 5.0


def test_build_footprint_norm_and_cmap_log_clamps_nonpositive() -> None:
    norm, _cmap, amp_render, vmin, vmax = build_footprint_norm_and_cmap(
        amp=np.asarray([-2.0, 0.0, 0.1, 4.0], dtype=float),
        scale_mode="log",
    )
    assert norm is not None
    assert vmin > 0.0
    assert vmax > vmin
    assert np.min(amp_render) > 0.0
