from __future__ import annotations

from pathlib import Path
from typing import Any


def write_reconstruct_report_markdown(
	*,
	output_md: Path,
	h5_path: Path,
	stream_id: str,
	reconstruction_out_dir: Path,
	stage_outputs: dict[str, str],
	unit_rows: list[dict[str, Any]],
) -> Path:
	lines: list[str] = []
	lines.append("# Reconstruct Report")
	lines.append("")
	lines.append(f"- h5_path: {h5_path}")
	lines.append(f"- stream_id: {stream_id}")
	lines.append(f"- reconstruction_out_dir: {reconstruction_out_dir}")
	lines.append("")
	lines.append("## Stage Outputs")
	lines.append("")
	if stage_outputs:
		for key in sorted(stage_outputs):
			lines.append(f"- {key}: {stage_outputs[key]}")
	else:
		lines.append("- none")
	lines.append("")
	lines.append("## Units")
	lines.append("")
	lines.append("| unit_id | status | outputs | error |")
	lines.append("|---|---|---|---|")
	for row in unit_rows:
		unit_id = row.get("unit_id")
		status = row.get("status", "")
		error = row.get("error") or ""
		outputs = row.get("outputs", {})
		if isinstance(outputs, dict) and outputs:
			outputs_text = "; ".join(f"{k}={v}" for k, v in sorted(outputs.items()))
		else:
			outputs_text = ""
		lines.append(f"| {unit_id} | {status} | {outputs_text} | {error} |")

	output_md = Path(output_md)
	output_md.parent.mkdir(parents=True, exist_ok=True)
	output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
	return output_md


__all__ = ["write_reconstruct_report_markdown"]
