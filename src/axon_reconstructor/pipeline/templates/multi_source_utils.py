from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ..checkpointing import compute_checkpoint_file


def _try_get_recording_property(recording: Any, key: str):
    try:
        if hasattr(recording, "get_property_keys"):
            keys = set(recording.get_property_keys())
            if key not in keys:
                return None
        if hasattr(recording, "get_property"):
            return recording.get_property(key)
    except Exception:
        return None
    return None


def _try_get_electrode_ids(recording: Any) -> Optional[list[Any]]:
    """Best-effort extraction of an 'electrode id' per channel.

    This intentionally supports multiple SpikeInterface/Maxwell conventions.
    Returns a Python list or None.
    """

    for key in (
        "electrode_id",
        "electrode",
        "contact_id",
        "contact_ids",
        "contact",
        "site_id",
        "site",
    ):
        vals = _try_get_recording_property(recording, key)
        if vals is not None:
            try:
                return list(vals)
            except Exception:
                return None

    # Maxwell-style contact_vector.
    try:
        cv = _try_get_recording_property(recording, "contact_vector")
        if isinstance(cv, dict) and "electrode" in cv:
            return list(cv["electrode"])
    except Exception:
        pass

    return None


def _sparsity_unit_channel_indices(*, sparsity, unit_id: Any):
    """Return per-unit channel indices from a `ChannelSparsity` object.

    SpikeInterface API differs across versions:
    - Some versions expose `unit_id_to_channel_indices` as a callable.
    - Others expose it as a dict.
    """

    if sparsity is None:
        return None

    try:
        mapping = getattr(sparsity, "unit_id_to_channel_indices", None)
        if callable(mapping):
            return mapping(unit_id)
        if isinstance(mapping, dict):
            if unit_id in mapping:
                return mapping[unit_id]
            # Best-effort normalized match.
            try:
                from ..waveforms.exclusions import normalize_unit_id

                uid_norm = normalize_unit_id(unit_id)
                for k, v in mapping.items():
                    if normalize_unit_id(k) == uid_norm:
                        return v
            except Exception:
                pass
    except Exception:
        pass

    for attr in ("get_channel_indices", "get_channel_indices_for_unit"):
        if hasattr(sparsity, attr):
            fn = getattr(sparsity, attr)
            if callable(fn):
                try:
                    return fn(unit_id)
                except Exception:
                    pass

    return None


def _normalize_id_for_compare(x: Any) -> Any:
    """Normalize ids so np scalars / floats round-trip consistently."""

    try:
        if hasattr(x, "item"):
            x = x.item()
    except Exception:
        pass

    if isinstance(x, bool):
        return x
    if isinstance(x, int):
        return int(x)
    if isinstance(x, float) and x.is_integer():
        return int(x)
    return str(x)


def _load_curated_unit_ids_from_waveforms_outputs(*, well_out_dir: Path, logger) -> tuple[Optional[list[Any]], Optional[Path]]:
    """Load curated (kept) unit ids from waveforms outputs, if available."""

    metrics_curated_xlsx = well_out_dir / "waveforms_outputs" / "metrics_curated.xlsx"
    if not metrics_curated_xlsx.exists():
        return None, None

    try:
        import pandas as pd  # type: ignore[import-not-found]
    except Exception:
        logger.warning("Found %s but pandas is unavailable; cannot apply unit curation", metrics_curated_xlsx)
        return None, metrics_curated_xlsx

    try:
        df = pd.read_excel(metrics_curated_xlsx, index_col=0)
        curated = [_normalize_id_for_compare(x) for x in list(df.index.values)]

        seen: set[Any] = set()
        curated_unique: list[Any] = []
        for u in curated:
            if u in seen:
                continue
            seen.add(u)
            curated_unique.append(u)
        return curated_unique, metrics_curated_xlsx
    except Exception as e:
        logger.warning("Failed reading curated unit list from %s: %s", metrics_curated_xlsx, e)
        return None, metrics_curated_xlsx


def _ensure_analyzer_extensions(*, analyzer, extension_names: list[str], logger, n_jobs: int) -> None:
    missing = [name for name in extension_names if not analyzer.has_extension(name)]
    if not missing:
        return
    logger.info("Computing extensions: %s", ", ".join(missing))
    analyzer.compute(missing, verbose=False, n_jobs=max(1, int(n_jobs)))


def _load_waveforms_analyzers(
    *,
    well_out_dir: Path,
    include_concat: bool,
    include_segments: bool,
    logger,
):
    """Load analyzers produced by the waveforms stage.

    Returns a list of (source_name, analyzer).
    """

    import spikeinterface.full as si  # type: ignore[import-not-found]

    waveforms_out_dir = well_out_dir / "waveforms_outputs"
    concat_waveforms_dir = waveforms_out_dir / "concat_waveforms"
    segment_waveforms_dir = waveforms_out_dir / "segment_waveforms"

    analyzers: list[tuple[str, Any]] = []

    if include_concat:
        if not concat_waveforms_dir.exists():
            raise FileNotFoundError(f"Missing concat waveforms analyzer at {concat_waveforms_dir}")
        logger.info("Loading concat analyzer: %s", concat_waveforms_dir)
        analyzers.append(("concat", si.load_sorting_analyzer(concat_waveforms_dir)))

    if include_segments and segment_waveforms_dir.exists():
        seg_dirs = sorted([p for p in segment_waveforms_dir.iterdir() if p.is_dir()])
        logger.info("Found %d segment analyzers", len(seg_dirs))
        for p in seg_dirs:
            try:
                analyzers.append((p.name, si.load_sorting_analyzer(p)))
            except Exception:
                logger.warning("Skipping unreadable segment analyzer: %s", p)

    if not analyzers:
        raise RuntimeError("No analyzers available for templates")

    return analyzers


def _get_unit_template_from_extension(*, analyzer, templates_ext, unit_id: Any):
    """Compatibility helper for SpikeInterface templates extension."""

    try:
        if hasattr(analyzer, "sorting") and hasattr(analyzer.sorting, "id_to_index"):
            analyzer.sorting.id_to_index(unit_id)
    except Exception:
        return None

    if hasattr(templates_ext, "get_unit_template"):
        try:
            return templates_ext.get_unit_template(unit_id=unit_id)
        except Exception:
            return None

    if hasattr(templates_ext, "get_templates"):
        try:
            all_templates = templates_ext.get_templates()
            unit_index = list(analyzer.sorting.unit_ids).index(unit_id)
            return all_templates[unit_index]
        except Exception:
            return None

    return None


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _compute_templates_checkpoint_file(*, well_out_dir: Path, h5_path: Path, stream_id: str) -> Path:
    main_ckpt = compute_checkpoint_file(output_dir=well_out_dir, file_path=h5_path, stream_id=stream_id)
    name = main_ckpt.name
    if name.endswith("_checkpoint.json"):
        name = name[: -len("_checkpoint.json")] + "_templates_checkpoint.json"
    else:
        name = main_ckpt.stem + "_templates_checkpoint.json"
    return main_ckpt.with_name(name)
