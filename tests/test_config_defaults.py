from __future__ import annotations

from pathlib import Path

from axon_reconstructor.pipeline.templates import TemplateExtractInputs
from axon_reconstructor.pipeline.waveforms import WaveformExtractInputs


def test_waveforms_inputs_accept_overrides() -> None:
    inputs = WaveformExtractInputs(
        h5_path=Path("/tmp/data.raw.h5"),
        stream_id="well000",
        mea_output_root=Path("/tmp/out"),
        n_jobs=5,
        per_segment=False,
        filter_by_maxwell_epochs=False,
    )

    assert inputs.n_jobs == 5
    assert inputs.per_segment is False
    assert inputs.filter_by_maxwell_epochs is False


def test_templates_inputs_accept_overrides() -> None:
    inputs = TemplateExtractInputs(
        h5_path=Path("/tmp/data.raw.h5"),
        stream_id="well000",
        mea_output_root=Path("/tmp/out"),
        run_unit_merging=False,
        include_segments=False,
        unit_limit=10,
    )

    assert inputs.run_unit_merging is False
    assert inputs.include_segments is False
    assert inputs.unit_limit == 10
