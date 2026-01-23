from __future__ import annotations

from pathlib import Path

from axon_reconstructor.pipeline.templates import TemplateExtractInputs
from axon_reconstructor.pipeline.waveforms import WaveformExtractInputs


def test_waveforms_inputs_defaults() -> None:
    inputs = WaveformExtractInputs(
        h5_path=Path("/tmp/data.raw.h5"),
        stream_id="well000",
        mea_output_root=Path("/tmp/out"),
    )

    assert inputs.sorter == "kilosort4"
    assert inputs.ms_before is None
    assert inputs.ms_after is None
    assert inputs.per_segment is True
    assert inputs.per_segment_only_additional_channels is True
    assert inputs.filter_by_maxwell_epochs is True


def test_templates_inputs_defaults() -> None:
    inputs = TemplateExtractInputs(
        h5_path=Path("/tmp/data.raw.h5"),
        stream_id="well000",
        mea_output_root=Path("/tmp/out"),
    )

    assert inputs.include_concat is True
    assert inputs.include_segments is True
    assert inputs.unit_ids is None
    assert inputs.unit_limit is None

    assert inputs.run_unit_merging is True
    assert inputs.merge_presets is None
    assert inputs.merge_recursive is False
    assert inputs.n_jobs == 8
