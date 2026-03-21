from __future__ import annotations

from axon_recon.pipeline.stages.reconstruct.io import format_unit_reldir


def test_format_unit_reldir() -> None:
	p = format_unit_reldir("units/{unit_id:04d}/", 94)
	assert str(p) == "units/0094"

