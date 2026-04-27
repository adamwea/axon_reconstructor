from .context import ExecutionTarget, StageParallelism
from .distributor import distribute_targets
from .lifecycle import install_linux_parent_death_signal, install_process_lifecycle
from .phase_chain import PhaseChainError, PhaseChainResult, PhaseDescriptor, PhaseOutcome, run_phase_chain
from .results import MultiTargetStageResult, TargetStageResult

__all__ = [
    "ExecutionTarget",
    "StageParallelism",
    "PhaseDescriptor",
    "PhaseOutcome",
    "PhaseChainResult",
    "PhaseChainError",
    "TargetStageResult",
    "MultiTargetStageResult",
    "distribute_targets",
    "run_phase_chain",
    "install_linux_parent_death_signal",
    "install_process_lifecycle",
]
