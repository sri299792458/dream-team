"""Unit tests for routing helpers used in the experiment graph."""

from __future__ import annotations

from typing import Callable

from dream_team.experiment.graph import (
    AgentConfig,
    ExperimentConfig,
    ExperimentState,
    TeamConfig,
)
from dream_team.experiment.graph.routing import (
    route_after_bootstrap,
    route_after_check_continue,
    route_after_check_evolution,
)


def _make_state(configure: Callable[[ExperimentState], None] | None = None) -> ExperimentState:
    """Create a minimal ExperimentState for routing tests."""

    team_lead = AgentConfig(
        title="Principal Investigator",
        expertise="ml",
        goal="optimize",
        role="lead",
    )
    coding_agent = AgentConfig(
        title="Engineer",
        expertise="python",
        goal="implement",
        role="code",
    )
    team = TeamConfig(team_lead=team_lead, coding_agent=coding_agent, team_members=[])
    config = ExperimentConfig(problem_statement="problem", target_metric="metric")
    state = ExperimentState(team=team, config=config)

    if configure:
        configure(state)

    return state


def test_route_after_bootstrap_runs_math_init_when_not_initialized() -> None:
    state = _make_state()
    assert route_after_bootstrap(state) == "init_math"


def test_route_after_bootstrap_skips_math_init_when_initialized() -> None:
    state = _make_state(
        lambda s: setattr(
            s,
            "mathematical_state",
            s.mathematical_state.model_copy(update={"iteration_count": 1}),
        )
    )
    assert route_after_bootstrap(state) == "plan"


def test_route_after_check_continue_goes_to_complete_when_should_stop() -> None:
    state = _make_state(lambda s: setattr(s, "should_stop", True))
    assert route_after_check_continue(state) == "complete"


def test_route_after_check_continue_loops_when_not_stopping() -> None:
    state = _make_state()
    assert route_after_check_continue(state) == "check_evolution"


def test_route_after_check_evolution_returns_evolve_when_triggered() -> None:
    state = _make_state(
        lambda s: setattr(
            s,
            "evolution",
            s.evolution.model_copy(update={"triggered": True}),
        )
    )
    assert route_after_check_evolution(state) == "evolve"


def test_route_after_check_evolution_returns_plan_when_not_triggered() -> None:
    state = _make_state()
    assert route_after_check_evolution(state) == "plan"

