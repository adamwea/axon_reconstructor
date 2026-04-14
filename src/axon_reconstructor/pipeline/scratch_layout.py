from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


CANONICAL_SCRATCH_DIRNAME = "axon_recon_scratch"
CANONICAL_SCRATCH_INPUTS_DIRNAME = "inputs"
CANONICAL_SCRATCH_OUTPUTS_DIRNAME = "outputs"


@dataclass(frozen=True)
class ScratchLayout:
	scratch_root: Path
	canonical_root: Path
	inputs_root: Path
	outputs_root: Path


def resolve_optional_path(raw: str | Path | None) -> Path | None:
	if raw is None:
		return None
	token = str(raw).strip()
	if token == "":
		return None
	return Path(token).expanduser().resolve()


def resolve_scratch_layout(raw: str | Path | None) -> ScratchLayout | None:
	path = resolve_optional_path(raw)
	if path is None:
		return None

	if path.name == CANONICAL_SCRATCH_OUTPUTS_DIRNAME and path.parent.name == CANONICAL_SCRATCH_DIRNAME:
		canonical_root = path.parent
		scratch_root = canonical_root.parent
		outputs_root = path
		inputs_root = canonical_root / CANONICAL_SCRATCH_INPUTS_DIRNAME
	elif path.name == CANONICAL_SCRATCH_INPUTS_DIRNAME and path.parent.name == CANONICAL_SCRATCH_DIRNAME:
		canonical_root = path.parent
		scratch_root = canonical_root.parent
		inputs_root = path
		outputs_root = canonical_root / CANONICAL_SCRATCH_OUTPUTS_DIRNAME
	elif path.name == CANONICAL_SCRATCH_DIRNAME:
		canonical_root = path
		scratch_root = canonical_root.parent
		inputs_root = canonical_root / CANONICAL_SCRATCH_INPUTS_DIRNAME
		outputs_root = canonical_root / CANONICAL_SCRATCH_OUTPUTS_DIRNAME
	else:
		scratch_root = path
		canonical_root = scratch_root / CANONICAL_SCRATCH_DIRNAME
		inputs_root = canonical_root / CANONICAL_SCRATCH_INPUTS_DIRNAME
		outputs_root = canonical_root / CANONICAL_SCRATCH_OUTPUTS_DIRNAME

	return ScratchLayout(
		scratch_root=scratch_root,
		canonical_root=canonical_root,
		inputs_root=inputs_root,
		outputs_root=outputs_root,
	)


def resolve_canonical_scratch_input_root(raw: str | Path | None) -> Path | None:
	layout = resolve_scratch_layout(raw)
	return None if layout is None else layout.inputs_root


def resolve_canonical_scratch_output_root(raw: str | Path | None) -> Path | None:
	layout = resolve_scratch_layout(raw)
	return None if layout is None else layout.outputs_root


__all__ = [
	"CANONICAL_SCRATCH_DIRNAME",
	"CANONICAL_SCRATCH_INPUTS_DIRNAME",
	"CANONICAL_SCRATCH_OUTPUTS_DIRNAME",
	"ScratchLayout",
	"resolve_canonical_scratch_input_root",
	"resolve_canonical_scratch_output_root",
	"resolve_optional_path",
	"resolve_scratch_layout",
]