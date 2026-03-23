from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Iterable


def write_amplitude_map_summary_png(
	*,
	entries: Iterable[tuple[Any, Path]],
	output_png: Path,
	ncols: int = 5,
	title: str = "Reconstruct Amplitude Maps",
) -> bool:
	import matplotlib

	matplotlib.use("Agg", force=True)
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]

	items = [(unit_id, Path(path)) for unit_id, path in entries if Path(path).exists()]
	if not items:
		return False

	ncols = max(1, int(ncols))
	nrows = (len(items) + ncols - 1) // ncols
	fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2.8 * ncols, 2.6 * nrows), dpi=170)
	if nrows == 1 and ncols == 1:
		axes_list = [axes]
	else:
		try:
			axes_list = list(axes.ravel())
		except Exception:
			axes_list = [axes]

	for ax in axes_list:
		ax.axis("off")

	for idx, (unit_id, image_path) in enumerate(items):
		if idx >= len(axes_list):
			break
		ax = axes_list[idx]
		img = plt.imread(str(image_path))
		ax.imshow(img)
		ax.set_title(f"unit {unit_id}", fontsize=8)
		ax.axis("off")

	fig.suptitle(str(title), fontsize=10)
	fig.tight_layout()
	output_png = Path(output_png)
	output_png.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_png, dpi=170, bbox_inches="tight")
	plt.close(fig)
	return True


__all__ = ["write_amplitude_map_summary_png"]
