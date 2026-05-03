from __future__ import annotations

from pathlib import Path
from typing import Any, Hashable


def _path_key(value: Any) -> str:
    try:
        return str(Path(value).expanduser())
    except Exception:
        return str(value)


def target_read_group_key(target: Any) -> Hashable:
    source_h5_path = getattr(target, "source_h5_path", None)
    if source_h5_path is not None and str(source_h5_path).strip() != "":
        return ("source_h5_path", _path_key(source_h5_path))

    h5_path = getattr(target, "h5_path", None)
    if h5_path is not None and str(h5_path).strip() != "":
        return ("h5_path", _path_key(h5_path))

    dataset_index = getattr(target, "dataset_index", None)
    if dataset_index is not None:
        try:
            return ("dataset_index", int(dataset_index))
        except Exception:
            return ("dataset_index", str(dataset_index))

    dataset_id = getattr(target, "dataset_id", None)
    if dataset_id is not None and str(dataset_id).strip() != "":
        return ("dataset_id", str(dataset_id))

    return ("target", id(target))


def count_target_read_groups(targets: list[Any]) -> int:
    return len({target_read_group_key(target) for target in targets})