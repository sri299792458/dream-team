"""
Centralized state management for experiment orchestration.

This module defines the single source of truth for experiment state,
replacing scattered dicts and parameters with an explicit Pydantic model.

The ExperimentState will be used by LangGraph to manage the experiment lifecycle.
"""

from typing import Literal, Dict, List, Optional, Any
from pydantic import BaseModel, Field
from pathlib import Path


class AgentConfig(BaseModel):
    """Configuration for a single agent"""
    title: str
    expertise: str
    goal: str
    role: str
    model: str = "gemini-2.0-flash-exp"
    specialization_depth: int = 0


class TeamConfig(BaseModel):
    """Team composition and configuration"""
    team_lead: AgentConfig
    team_members: List[AgentConfig] = Field(default_factory=list)
    coding_agent: AgentConfig


class IterationSummary(BaseModel):
    """Summary of a single iteration"""
    iteration: int
    phase: str
    approach: Optional[str] = None
    code: Optional[str] = None
    results: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)
    agents_snapshot: List[str] = Field(default_factory=list)
    notes: Optional[str] = None

    # Bootstrap-specific fields
    recruitment_plan: Optional[str] = None
    recruited_agents: Optional[List[Dict[str, Any]]] = None


class MathematicalState(BaseModel):
    """Mathematical framework state for team dynamics"""
    problem_concepts: List[str] = Field(default_factory=list)
    team_diversity: float = 0.0
    iteration_count: int = 0

    # Agent-level mathematical state
    agent_states: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class EvolutionState(BaseModel):
    """State related to team evolution"""
    decision: Optional[Literal["NO_CHANGE", "ADD_AGENT", "REMOVE_AGENT", "DEEPEN_AGENT", "REPLAN", "STOP"]] = None
    reason: Optional[str] = None
    triggered: bool = False
    trigger_names: List[str] = Field(default_factory=list)


class ExperimentConfig(BaseModel):
    """Configuration for the experiment run"""
    problem_statement: str
    target_metric: str
    minimize_metric: bool = True
    max_iterations: int = 5
    target_score: Optional[float] = None
    resume: bool = True


class ExperimentState(BaseModel):
    """
    Unified state for the entire experiment.

    This replaces all the scattered state in ExperimentOrchestrator:
    - iteration, best_metric, experiment_history
    - team_lead, team_members, coding_agent
    - problem_statement, target_metric, etc.
    - bootstrap_completed, column_schemas
    - problem_graph, team (mathematical framework)

    All state is serializable for checkpointing.
    """

    # Core iteration state
    iteration: int = 0
    phase: Literal[
        "init",
        "bootstrap",
        "init_math",
        "plan",
        "code",
        "execute",
        "evaluate",
        "check_continue",
        "check_evolution",
        "evolve",
        "complete"
    ] = "init"

    # Team configuration
    team: TeamConfig

    # Experiment configuration
    config: ExperimentConfig

    # Execution history
    history: List[IterationSummary] = Field(default_factory=list)

    # Current iteration state
    current_approach: Optional[str] = None
    current_code: Optional[str] = None
    current_results: Optional[Dict[str, Any]] = None
    current_metrics: Dict[str, float] = Field(default_factory=dict)

    # Best metrics tracking
    best_metric: Optional[float] = None
    best_iteration: Optional[int] = None

    # Bootstrap state
    bootstrap_completed: bool = False
    column_schemas: Dict[str, List[str]] = Field(default_factory=dict)

    # Mathematical framework
    mathematical_state: MathematicalState = Field(default_factory=MathematicalState)

    # Evolution state
    evolution: EvolutionState = Field(default_factory=EvolutionState)

    # Data context (non-serializable data references)
    # Note: We store metadata about data, not the data itself
    data_context_keys: List[str] = Field(default_factory=list)

    # Paths and outputs
    results_dir: str = "results"

    # Goal achievement
    goal_achieved: bool = False
    should_stop: bool = False

    class Config:
        arbitrary_types_allowed = True

    def get_iteration_summary(self) -> IterationSummary:
        """Create summary of current iteration"""
        return IterationSummary(
            iteration=self.iteration,
            phase=self.phase,
            approach=self.current_approach,
            code=self.current_code,
            results=self.current_results or {},
            metrics=self.current_metrics,
            agents_snapshot=[
                self.team.team_lead.title
            ] + [m.title for m in self.team.team_members]
        )

    def update_best_metric(self) -> bool:
        """
        Update best metric if current is better.
        Returns True if new best found.
        """
        if not self.current_metrics or self.config.target_metric not in self.current_metrics:
            return False

        current = self.current_metrics[self.config.target_metric]

        record_iteration = self.iteration if self.iteration > 0 else 1

        if self.best_metric is None:
            self.best_metric = current
            self.best_iteration = record_iteration
            return True

        is_better = (
            (self.config.minimize_metric and current < self.best_metric) or
            (not self.config.minimize_metric and current > self.best_metric)
        )

        if is_better:
            self.best_metric = current
            self.best_iteration = record_iteration
            return True

        return False

    def check_goal_achieved(self) -> bool:
        """Check if target score has been achieved"""
        if self.config.target_score is None:
            return False

        if self.config.target_metric not in self.current_metrics:
            return False

        current = self.current_metrics[self.config.target_metric]

        if self.config.minimize_metric:
            return current <= self.config.target_score
        else:
            return current >= self.config.target_score

    def should_evolve(self) -> bool:
        """Check if evolution should be triggered"""
        # Need at least 3 iterations of history
        if len(self.history) < 3:
            return False

        # Get recent metric history
        recent_metrics = []
        for h in self.history[-3:]:
            if self.config.target_metric in h.metrics:
                recent_metrics.append(h.metrics[self.config.target_metric])

        if len(recent_metrics) < 3:
            return False

        # Simple plateau detection: no improvement in last 3 iterations
        if self.config.minimize_metric:
            # Check if metrics aren't decreasing
            improvements = [recent_metrics[i] - recent_metrics[i+1] for i in range(len(recent_metrics)-1)]
        else:
            # Check if metrics aren't increasing
            improvements = [recent_metrics[i+1] - recent_metrics[i] for i in range(len(recent_metrics)-1)]

        # If all improvements are near zero, we've plateaued
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0
        return abs(avg_improvement) < 0.01

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentState':
        """Load from dictionary"""
        return cls(**data)


def create_initial_state(
    team_lead: AgentConfig,
    coding_agent: AgentConfig,
    problem_statement: str,
    target_metric: str,
    minimize_metric: bool = True,
    max_iterations: int = 5,
    results_dir: str = "results",
    team_members: Optional[List[AgentConfig]] = None,
    target_score: Optional[float] = None,
    resume: bool = True,
) -> ExperimentState:
    """
    Factory function to create initial experiment state.

    Args:
        team_lead: Lead agent configuration
        coding_agent: Coding agent configuration
        problem_statement: Description of the problem
        target_metric: Metric to optimize
        minimize_metric: Whether lower is better
        max_iterations: Maximum iterations
        results_dir: Directory for results
        team_members: Optional initial team members
        target_score: Optional target score to achieve
        resume: Whether to allow resuming

    Returns:
        Initialized ExperimentState
    """
    team_config = TeamConfig(
        team_lead=team_lead,
        team_members=team_members or [],
        coding_agent=coding_agent
    )

    experiment_config = ExperimentConfig(
        problem_statement=problem_statement,
        target_metric=target_metric,
        minimize_metric=minimize_metric,
        max_iterations=max_iterations,
        target_score=target_score,
        resume=resume
    )

    return ExperimentState(
        team=team_config,
        config=experiment_config,
        results_dir=results_dir,
        bootstrap_completed=len(team_members or []) > 0
    )
