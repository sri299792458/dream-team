from typing import Literal

from .state import ExperimentState


def route_after_check_evolution(state: ExperimentState) -> Literal["evolve", "plan"]:
    """Route to evolution or next iteration based on evolution check."""
    return "evolve" if state.evolution.triggered else "plan"


def route_after_check_continue(state: ExperimentState) -> Literal["complete", "check_evolution"]:
    """Route to completion or evolution check based on continue check."""
    return "complete" if state.should_stop else "check_evolution"


def route_after_bootstrap(state: ExperimentState) -> Literal["init_math", "plan"]:
    """Route from bootstrap to math init or directly to planning."""
    return "init_math" if not state.mathematical_state.iteration_count else "plan"
