"""Pipeline state machine.

Deliberately minimal and observation-only. There is no state or transition
here that represents or enables actuation, targeting, or engagement — the
only states are about whether the observation pipeline itself is running.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class PipelineState(Enum):
    IDLE = auto()
    RUNNING = auto()
    STOPPED = auto()
    ERROR = auto()


_VALID_TRANSITIONS: dict[PipelineState, set[PipelineState]] = {
    PipelineState.IDLE: {PipelineState.RUNNING},
    PipelineState.RUNNING: {PipelineState.STOPPED, PipelineState.ERROR},
    PipelineState.STOPPED: {PipelineState.RUNNING, PipelineState.IDLE},
    PipelineState.ERROR: {PipelineState.IDLE},
}


@dataclass
class StateMachine:
    state: PipelineState = PipelineState.IDLE
    history: list[PipelineState] = field(default_factory=lambda: [PipelineState.IDLE])

    def transition(self, new_state: PipelineState) -> None:
        if new_state not in _VALID_TRANSITIONS[self.state]:
            raise ValueError(f"Invalid transition: {self.state.name} -> {new_state.name}")
        self.state = new_state
        self.history.append(new_state)

    def is_running(self) -> bool:
        return self.state is PipelineState.RUNNING
