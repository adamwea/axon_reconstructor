from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TemplatePlotConfig:
	write_png: bool = True
	write_svg: bool = False
	relpath: str = "template"
	channel_scope: str = "contributing_channels"
	background: str = "black"
	signal_color: str = "white"
	force_center_soma: bool = False
	force_square_aspect: bool = True
	show_scale_bar: bool = True
	scale_bar_color: str = "white"
	scale_bar_text_offset_frac: float = 0.02
	scale_bar_y_offset_frac: float = 0.06
	scale_bar_fontsize: float = 6.0
	scale_bar_linewidth: float = 1.8
	scale_bar_length_um: float | None = None


@dataclass(frozen=True)
class PerUnitTemplatesOutputsConfig:
	unit_reldir: str = "units/{unit_id:04d}/"
	template: TemplatePlotConfig = field(default_factory=TemplatePlotConfig)


@dataclass(frozen=True)
class TemplatesInputs:
	h5_path: Path
	stream_id: str
	mea_output_root: Path

	output_rel_root: str = "templates_outputs"
	per_unit_outputs: PerUnitTemplatesOutputsConfig = field(default_factory=PerUnitTemplatesOutputsConfig)

	unit_ids: list[Any] | None = None
	unit_limit: int | None = None

	force_restart: bool = False
	force_replot: bool = False
	n_jobs: int = 1
