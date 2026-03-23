from __future__ import annotations

from pathlib import Path

from axon_recon.pipeline.stages.reconstruct.io import format_unit_reldir
from axon_recon.pipeline.stages.reconstruct.io import resolve_unit_output_paths
from axon_recon.pipeline.stages.reconstruct.models.inputs import PerUnitOutputsConfig


def test_format_unit_reldir() -> None:
	p = format_unit_reldir("units/{unit_id:04d}/", 94)
	assert str(p) == "units/0094"


def test_resolve_unit_output_paths_includes_amplitude_map() -> None:
	paths = resolve_unit_output_paths(
		reconstruction_out_dir=Path("/tmp/recon"),
		unit_id=1,
		per_unit_outputs=PerUnitOutputsConfig(
			write_amplitude_map_png=True,
			amplitude_map_png_relpath="maps/amplitude_map.png",
		),
	)
	assert paths["amplitude_map_png"] == Path("/tmp/recon") / "units/0001" / "maps/amplitude_map.png"

