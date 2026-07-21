import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from control.state import PipelineState, StateMachine


def test_starts_idle():
    machine = StateMachine()
    assert machine.state is PipelineState.IDLE
    assert not machine.is_running()


def test_idle_to_running_is_valid():
    machine = StateMachine()
    machine.transition(PipelineState.RUNNING)
    assert machine.is_running()


def test_idle_to_error_is_valid():
    # Regression test: a source that fails to open must be able to move
    # straight from IDLE to ERROR without the state machine itself raising.
    machine = StateMachine()
    machine.transition(PipelineState.ERROR)
    assert machine.state is PipelineState.ERROR


def test_error_to_idle_recovers():
    machine = StateMachine()
    machine.transition(PipelineState.ERROR)
    machine.transition(PipelineState.IDLE)
    assert machine.state is PipelineState.IDLE


def test_running_to_stopped_is_valid():
    machine = StateMachine()
    machine.transition(PipelineState.RUNNING)
    machine.transition(PipelineState.STOPPED)
    assert not machine.is_running()


def test_invalid_transition_raises():
    machine = StateMachine()
    with pytest.raises(ValueError, match="Invalid transition"):
        machine.transition(PipelineState.STOPPED)


def test_history_records_every_transition():
    machine = StateMachine()
    machine.transition(PipelineState.RUNNING)
    machine.transition(PipelineState.STOPPED)
    assert machine.history == [PipelineState.IDLE, PipelineState.RUNNING, PipelineState.STOPPED]
