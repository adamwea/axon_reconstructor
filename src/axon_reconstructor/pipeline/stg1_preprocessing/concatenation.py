from __future__ import annotations

from pathlib import Path
from typing import Optional

from .utils import _ensure_maxwell_hdf5_plugin_path


def find_common_electrodes_from_segments(*, h5_path: Path, stream_id: str) -> tuple[list[str], list[int]]:
    """Compute the shared electrode set across all rec segments for a stream."""

    try:
        import h5py
        import spikeinterface.extractors as se
    except Exception as e:  # pragma: no cover
        raise RuntimeError("raw preprocessing requires `h5py` and `spikeinterface` installed") from e

    _ensure_maxwell_hdf5_plugin_path()

    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as h5:
        rec_names = list(h5["wells"][stream_id].keys())

    common: Optional[set[int]] = None
    for rec_name in rec_names:
        if hasattr(se, "read_maxwell"):
            rec = se.read_maxwell(file_path=str(h5_path), stream_id=stream_id, rec_name=rec_name)
        else:  # pragma: no cover
            rec = se.MaxwellRecordingExtractor(str(h5_path), stream_id=stream_id, rec_name=rec_name)
        electrodes = rec.get_property("contact_vector")["electrode"]
        electrode_set = set(int(x) for x in electrodes)
        if common is None:
            common = electrode_set
        else:
            common &= electrode_set

    return rec_names, sorted(common or set())


def _process_rec_segment_for_concatenation(
    *,
    h5_path: Path,
    stream_id: str,
    rec_name: str,
    common_el: list[int],
    center_chunk_size: int,
    expected_xy_by_electrode: Optional[dict[int, tuple[float, float]]] = None,
    expected_xy_atol: float = 0.0,
):
    """Load a segment, center, select common electrodes, validate, and normalize channel ids."""

    try:
        import numpy as np
        import spikeinterface.full as si
        import spikeinterface.extractors as se
    except Exception as e:  # pragma: no cover
        raise RuntimeError("raw preprocessing requires `numpy` and `spikeinterface` installed") from e

    _ensure_maxwell_hdf5_plugin_path()

    if hasattr(se, "read_maxwell"):
        rec = se.read_maxwell(file_path=str(h5_path), stream_id=stream_id, rec_name=rec_name)
    else:  # pragma: no cover
        rec = se.MaxwellRecordingExtractor(str(h5_path), stream_id=stream_id, rec_name=rec_name)

    fs = float(rec.get_sampling_frequency())
    n_samples = int(rec.get_num_samples())
    chunk = min(center_chunk_size, rec.get_num_samples()) - 100
    chunk = max(chunk, 100)
    rec_centered = si.center(rec, chunk_size=chunk)

    rec_el = np.asarray(rec.get_property("contact_vector")["electrode"], dtype=int)
    if int(np.unique(rec_el).size) != int(rec_el.size):
        raise RuntimeError(
            f"Duplicate electrode ids found in contact_vector for segment {rec_name}; cannot map electrodes reliably"
        )
    el_to_idx = {int(el): int(i) for i, el in enumerate(rec_el)}
    try:
        chan_idx = [el_to_idx[int(el)] for el in common_el]
    except KeyError as e:
        raise RuntimeError(
            f"Segment {rec_name} is missing expected electrode id={e.args[0]} from the common electrode set"
        ) from e

    sel_channels = np.asarray(rec.get_channel_ids(), dtype=object)[chan_idx]

    processed = rec_centered.select_channels(list(sel_channels))

    processed_ch = np.asarray(processed.get_channel_ids(), dtype=object)
    if processed_ch.shape != sel_channels.shape or not np.array_equal(processed_ch, sel_channels):
        raise RuntimeError(
            f"Selected channel_ids mismatch for segment {rec_name}. "
            "This may indicate a channel-id ordering issue during selection."
        )

    processed_el = np.asarray(processed.get_property("contact_vector")["electrode"], dtype=int)
    expected_el = np.asarray(common_el, dtype=int)
    if processed_el.shape != expected_el.shape or not np.array_equal(processed_el, expected_el):
        raise RuntimeError(
            f"Selected electrodes mismatch for segment {rec_name}. "
            f"Expected {expected_el.shape[0]} electrodes matching common set; got {processed_el.shape[0]} "
            f"and/or different ordering."
        )

    if expected_xy_by_electrode is not None:
        cv = processed.get_property("contact_vector")
        from .plotting import _extract_xy_from_contact_vector

        x, y = _extract_xy_from_contact_vector(cv)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        expected_x = np.asarray([expected_xy_by_electrode[int(el)][0] for el in expected_el], dtype=float)
        expected_y = np.asarray([expected_xy_by_electrode[int(el)][1] for el in expected_el], dtype=float)
        if not (
            np.allclose(x, expected_x, atol=float(expected_xy_atol))
            and np.allclose(y, expected_y, atol=float(expected_xy_atol))
        ):
            raise RuntimeError(
                f"Electrode x/y locations differ from reference for segment {rec_name}. "
                "This suggests inconsistent layouts across segments; refusing to concatenate."
            )

    processed = processed.rename_channels([int(el) for el in expected_el])

    renamed_ch = np.asarray(processed.get_channel_ids(), dtype=object)
    if renamed_ch.shape != expected_el.shape or not np.array_equal(renamed_ch.astype(int), expected_el):
        raise RuntimeError(f"Failed to rename channel ids to electrode ids for segment {rec_name}")

    return processed, {
        "rec_name": rec_name,
        "fs": fs,
        "n_samples": n_samples,
        "n_channels": int(processed.get_num_channels()),
    }


__all__ = ["find_common_electrodes_from_segments", "_process_rec_segment_for_concatenation"]
