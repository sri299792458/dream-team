from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ...agent import Agent
from ...evolution import EvolutionEngine
from ...executor import CodeExecutor
from ...knowledge_state import KnowledgeGraph
from ...research import get_research_assistant
from ...team import Team


class ExecutionContext:
    """Container for non-serializable execution dependencies."""

    def __init__(
        self,
        data_context: Dict[str, Any],
        results_dir: Path,
        research_api: Optional[Any] = None,
    ) -> None:
        self.data_context = data_context
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        artifacts_dir = self.results_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        data_context["artifacts_dir"] = artifacts_dir

        self.executor = CodeExecutor(data_context=data_context)

        if research_api is None:
            # In offline/test environments we may not have access to real research APIs.
            self.research_api = None
        else:
            self.research_api = research_api

        self.team_lead: Optional[Agent] = None
        self.team_members: List[Agent] = []
        self.coding_agent: Optional[Agent] = None
        self.all_agents: List[Agent] = []

        self.evolution_engine = EvolutionEngine()

        self.problem_graph: Optional[KnowledgeGraph] = None
        self.team: Optional[Team] = None

    def create_agents_from_state(self, state: "ExperimentState") -> None:
        """Create agent instances from the serialized state."""
        from .state import AgentConfig, ExperimentState  # local import to avoid cycle

        lead_cfg: AgentConfig = state.team.team_lead
        self.team_lead = Agent(
            title=lead_cfg.title,
            expertise=lead_cfg.expertise,
            goal=lead_cfg.goal,
            role=lead_cfg.role,
            model=lead_cfg.model,
            specialization_depth=lead_cfg.specialization_depth,
        )

        self.team_members = [
            Agent(
                title=member.title,
                expertise=member.expertise,
                goal=member.goal,
                role=member.role,
                model=member.model,
                specialization_depth=member.specialization_depth,
            )
            for member in state.team.team_members
        ]

        coding_cfg = state.team.coding_agent
        self.coding_agent = Agent(
            title=coding_cfg.title,
            expertise=coding_cfg.expertise,
            goal=coding_cfg.goal,
            role=coding_cfg.role,
            model=coding_cfg.model,
            specialization_depth=coding_cfg.specialization_depth,
        )

        self.all_agents = [self.team_lead] + self.team_members

    def update_state_from_agents(self, state: "ExperimentState") -> None:
        """Write agent attributes back to the serialized state."""
        from .state import AgentConfig, ExperimentState  # local import to avoid cycle

        state.team.team_lead = AgentConfig(
            title=self.team_lead.title,
            expertise=self.team_lead.expertise,
            goal=self.team_lead.goal,
            role=self.team_lead.role,
            model=self.team_lead.model,
            specialization_depth=self.team_lead.specialization_depth,
        )

        state.team.team_members = [
            AgentConfig(
                title=agent.title,
                expertise=agent.expertise,
                goal=agent.goal,
                role=agent.role,
                model=agent.model,
                specialization_depth=agent.specialization_depth,
            )
            for agent in self.team_members
        ]

        state.team.coding_agent = AgentConfig(
            title=self.coding_agent.title,
            expertise=self.coding_agent.expertise,
            goal=self.coding_agent.goal,
            role=self.coding_agent.role,
            model=self.coding_agent.model,
            specialization_depth=self.coding_agent.specialization_depth,
        )
