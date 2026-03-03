from __future__ import annotations

from importlib import import_module

_target = import_module("axon_reconstructor.pipeline.stg2_spikesorting")

__all__ = list(getattr(_target, "__all__", []))
__doc__ = getattr(_target, "__doc__", __doc__)
__path__ = _target.__path__

for _name in __all__:
    globals()[_name] = getattr(_target, _name)


def __getattr__(name: str):
    return getattr(_target, name)
