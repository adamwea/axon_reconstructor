from __future__ import annotations

import logging
from pathlib import Path


def run_save_common_electrodes_core(
	*,
	common_electrodes: list[int],
	output_path: Path,
	logger: logging.Logger | None,
) -> dict[str, object]:
	import numpy as np

	output_path.parent.mkdir(parents=True, exist_ok=True)
	np.save(output_path, np.asarray([int(value) for value in list(common_electrodes)], dtype=np.int64))
	if logger is not None:
		logger.info("Preprocess common-electrode save wrote %d electrodes: %s", len(common_electrodes), output_path)
	return {
		"common_electrodes_path": str(output_path),
		"electrode_count": int(len(common_electrodes)),
	}