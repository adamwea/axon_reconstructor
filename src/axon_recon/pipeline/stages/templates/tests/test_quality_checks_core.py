from __future__ import annotations

import numpy as np

from axon_recon.pipeline.stages.templates.core.quality_checks import detect_multiple_negative_peaks


def test_detect_multiple_negative_peaks_flags_two_significant_minima() -> None:
	template = np.asarray(
		[
			[0.0, -1.0, -5.0, -1.5, -4.2, -1.0, 0.0],
			[0.0, -0.3, -0.8, -0.2, -0.2, -0.1, 0.0],
		],
		dtype=float,
	)
	out = detect_multiple_negative_peaks(
		template_c_by_t=template,
		channel_labels=[101, 102],
		prominence_fraction=0.30,
		min_separation_samples=2,
	)

	assert out["detected"] is True
	assert out["violation_count"] == 1
	assert out["violations"][0]["channel_index"] == 0
	assert out["violations"][0]["channel_label"] == "101"
	assert out["violations"][0]["peak_count"] == 2


def test_detect_multiple_negative_peaks_respects_min_separation() -> None:
	template = np.asarray(
		[
			[0.0, -4.2, -4.1, -0.3, 0.0],
		],
		dtype=float,
	)
	out = detect_multiple_negative_peaks(
		template_c_by_t=template,
		channel_labels=[11],
		prominence_fraction=0.20,
		min_separation_samples=3,
	)

	assert out["detected"] is False
	assert out["violation_count"] == 0


def test_detect_multiple_negative_peaks_caps_to_strongest_two() -> None:
	template = np.asarray(
		[
			[0.0, -5.0, -0.4, -4.0, -0.5, -3.0, 0.0],
		],
		dtype=float,
	)
	out = detect_multiple_negative_peaks(
		template_c_by_t=template,
		channel_labels=[7],
		prominence_fraction=0.20,
		min_separation_samples=1,
	)

	assert out["detected"] is True
	assert out["violation_count"] == 1
	assert out["max_peaks_per_channel"] == 2
	violation = out["violations"][0]
	assert violation["peak_count"] == 2
	assert violation["peak_indices"] == [1, 3]
