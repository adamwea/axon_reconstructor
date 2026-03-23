from __future__ import annotations

from pathlib import Path

import numpy as np

from axon_recon.pipeline.stages.reconstruct.core.summary_plots import write_amplitude_map_summary_png


def test_write_amplitude_map_summary_png_writes_file(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]

    img1 = np.zeros((20, 20, 3), dtype=float)
    img1[..., 0] = 1.0
    img2 = np.zeros((20, 20, 3), dtype=float)
    img2[..., 2] = 1.0

    p1 = tmp_path / "u1.png"
    p2 = tmp_path / "u2.png"
    plt.imsave(str(p1), img1)
    plt.imsave(str(p2), img2)

    out = tmp_path / "summary_grid.png"
    wrote = write_amplitude_map_summary_png(
        entries=[(1, p1), (2, p2)],
        output_png=out,
        ncols=2,
    )

    assert wrote is True
    assert out.exists()
    assert out.stat().st_size > 0


def test_write_amplitude_map_summary_png_no_inputs_returns_false(tmp_path: Path) -> None:
    out = tmp_path / "summary_grid.png"
    wrote = write_amplitude_map_summary_png(entries=[], output_png=out, ncols=2)
    assert wrote is False
    assert not out.exists()
