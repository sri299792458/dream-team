"""LangGraph-native orchestration for the Dream Team framework.

This module builds a self-contained LangGraph workflow that does **not**
wrap the legacy `ExperimentOrchestrator` loop. The graph owns its own state,
context propagation, mathematical signals, and evolution triggers so that
we don't carry forward the drift and coupling from the original iterative
loop. The workflow remains faithful to the desired PI-led flow:

1) PI kicks off with data exploration + literature search and recruits an
   initial roster of experts and a deepening generalist.
2) Team meets with ReAct + grounded search to propose a plan with full
   execution history context and artifact awareness.
3) Coding agent implements the plan, code executes with retries/safety,
   metrics are evaluated, and the knowledge graph is updated.
4) Evolution is considered after every iteration using the mathematical
   state (K, θ, δ, overlap, Gini) to determine whether to specialize,
   diversify, or broaden the roster.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph

from .agent import Agent
from .executor import CodeExecutor
from .knowledge_state import (
    AttentionDistribution,
    DepthMap,
    KnowledgeGraph,
    extract_concepts_from_text,
)
from .meetings import IndividualMeeting, TeamMeeting
from .evolution import EvolutionEngine
from .research import get_research_assistant
from .utils import save_json


@dataclass
class RunConfig:
    problem_statement: str
    data_context: Dict[str, Any]
    target_metric: str
    minimize_metric: bool = True
    max_iterations: int = 5
    target_score: Optional[float] = None
    resume: bool = True


@dataclass
class ArtifactRegistry:
    base_dir: Path
    artifacts_dir: Path
    code_dir: Path
    meetings_dir: Path
    agents_dir: Path


@dataclass
class MathematicalState:
    problem_graph: KnowledgeGraph
    attention: AttentionDistribution
    depth: DepthMap
    metric_history: List[float] = field(default_factory=list)

    def summarize_for_prompt(self) -> str:
        gini = self.depth.gini_coefficient()
        max_depth = self.depth.max_depth()
        top_depths = sorted(
            self.depth.depths.items(), key=lambda x: x[1], reverse=True
        )[:3]
        top_attention = self.attention.top_concepts()
        lines = [
            f"Specialization (Gini): {gini:.2f}",
            f"Max depth: {max_depth:.2f}",
        ]
        if top_depths:
            lines.append("Deep expertise:")
            lines.extend([f"- {c}: {d:.2f}" for c, d in top_depths])
        if top_attention:
            lines.append("Current focus (θ):")
            lines.extend([f"- {c}: {w:.2f}" for c, w in top_attention])
        return "\n".join(lines)


@dataclass
class ContextSnapshot:
    problem_summary: str
    data_schema: Dict[str, Any]
    artifact_manifest: Dict[str, str]
    prior_metrics: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_notes: List[str] = field(default_factory=list)
    math_state: str = ""


@dataclass
class IterationRecord:
    iteration: int
    plan: str
    code: str
    execution: Dict[str, Any]
    metrics: Dict[str, Any]
    agents_snapshot: List[str]


@dataclass
class GraphState:
    config: RunConfig
    artifacts: ArtifactRegistry
    roster: List[Agent]
    coding_agent: Agent
    executor: CodeExecutor
    evolution_engine: EvolutionEngine
    research_api: Any
    math_state: MathematicalState
    iteration: int = 1
    best_metric: Optional[float] = None
    plan: Optional[str] = None
    code: Optional[str] = None
    execution: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    history: List[IterationRecord] = field(default_factory=list)
    context: Optional[ContextSnapshot] = None


class LangGraphExperimentOrchestrator:
    """Self-contained LangGraph orchestrator."""

    def __init__(
        self,
        team_lead: Agent,
        team_members: List[Agent],
        coding_agent: Agent,
        results_dir: Path,
        evolution_engine: Optional[EvolutionEngine] = None,
    ):
        self.team_lead = team_lead
        self.initial_team_members = list(team_members)
        self.coding_agent = coding_agent
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.evolution_engine = evolution_engine or EvolutionEngine()
        self.research = get_research_assistant()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, **kwargs) -> Dict[str, Any]:
        config = RunConfig(**kwargs)
        data_context = config.data_context or {}

        artifacts_dir = self.results_dir / "artifacts"
        code_dir = self.results_dir / "code"
        meetings_dir = self.results_dir / "meetings"
        agents_dir = self.results_dir / "agents"
        for path in [artifacts_dir, code_dir, meetings_dir, agents_dir]:
            path.mkdir(parents=True, exist_ok=True)

        data_context["artifacts_dir"] = artifacts_dir
        executor = CodeExecutor(data_context=data_context)

        problem_graph = self._build_problem_graph(config.problem_statement)
        math_state = MathematicalState(
            problem_graph=problem_graph,
            attention=AttentionDistribution(),
            depth=DepthMap(),
        )

        state = GraphState(
            config=config,
            artifacts=ArtifactRegistry(
                base_dir=self.results_dir,
                artifacts_dir=artifacts_dir,
                code_dir=code_dir,
                meetings_dir=meetings_dir,
                agents_dir=agents_dir,
            ),
            roster=[self.team_lead] + list(self.initial_team_members),
            coding_agent=self.coding_agent,
            executor=executor,
            evolution_engine=self.evolution_engine,
            research_api=self.research.ss_api if hasattr(self.research, "ss_api") else None,
            math_state=math_state,
            context=self._build_context(config.problem_statement, data_context, math_state, []),
        )

        graph = self._build_graph()
        app = graph.compile()
        final_state = app.invoke(state)
        return self._final_summary(final_state)

    # ------------------------------------------------------------------
    # Graph wiring
    # ------------------------------------------------------------------
    def _build_graph(self) -> StateGraph:
        graph = StateGraph(GraphState)

        graph.add_node("kickoff", self._kickoff)
        graph.add_node("plan", self._team_meeting)
        graph.add_node("implement", self._implement)
        graph.add_node("execute", self._execute)
        graph.add_node("evaluate", self._evaluate)
        graph.add_node("evolve", self._evolve)

        graph.set_entry_point("kickoff")
        graph.add_edge("kickoff", "plan")
        graph.add_edge("plan", "implement")
        graph.add_edge("implement", "execute")
        graph.add_edge("execute", "evaluate")
        graph.add_conditional_edges(
            "evaluate",
            self._termination_check,
            {"continue": "evolve", "end": END},
        )
        graph.add_edge("evolve", "plan")

        return graph

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------
    def _kickoff(self, state: GraphState) -> GraphState:
        self._bootstrap_exploration(state)
        state.context = self._build_context(
            state.config.problem_statement,
            state.config.data_context,
            state.math_state,
            state.history,
        )
        return state

    def _team_meeting(self, state: GraphState) -> GraphState:
        agenda = self._meeting_agenda(state)
        meeting = TeamMeeting(save_dir=str(state.artifacts.meetings_dir), research_api=state.research_api)
        summary = meeting.run(
            team_lead=self.team_lead,
            team_members=state.roster,
            agenda=agenda,
            num_rounds=2,
            temperature=0.7,
        )
        state.plan = summary.get("summary", "")
        state.context = self._build_context(
            state.config.problem_statement,
            state.config.data_context,
            state.math_state,
            state.history,
        )
        return state

    def _implement(self, state: GraphState) -> GraphState:
        code_task = self._implementation_prompt(state)
        coder = IndividualMeeting(save_dir=str(state.artifacts.meetings_dir))
        code_output = coder.run(
            agent=state.coding_agent,
            task=code_task,
            num_iterations=1,
            use_react_coding=True,
        )
        state.code = code_output
        return state

    def _execute(self, state: GraphState) -> GraphState:
        description = (state.plan or "Generated plan")[:200]
        result = state.executor.execute_with_retry(
            code=state.code or "", description=description, max_retries=2
        )
        state.execution = result
        return state

    def _evaluate(self, state: GraphState) -> GraphState:
        state.metrics = self._extract_metrics(state.execution or {}, state.config.target_metric)
        record = IterationRecord(
            iteration=state.iteration,
            plan=state.plan or "",
            code=state.code or "",
            execution=self._strip_nonserializable(state.execution or {}),
            metrics=state.metrics,
            agents_snapshot=[a.title for a in state.roster],
        )
        state.history.append(record)

        self._update_best_metric(
            metrics=state.metrics,
            target_metric=state.config.target_metric,
            minimize_metric=state.config.minimize_metric,
            state=state,
        )
        state.math_state.metric_history.append(state.metrics.get(state.config.target_metric))
        save_json(
            {
                "iteration": record.iteration,
                "plan": record.plan,
                "metrics": record.metrics,
                "agents": record.agents_snapshot,
            },
            state.artifacts.base_dir / f"iteration_{state.iteration:02d}.json",
        )
        return state

    def _evolve(self, state: GraphState) -> GraphState:
        context = {
            "metric_history": state.math_state.metric_history,
            "minimize_metric": state.config.minimize_metric,
            "problem_description": state.config.problem_statement,
            "last_meeting_transcript": state.plan or "",
        }
        triggers = state.evolution_engine.check_triggers(context)
        if triggers:
            weakest = min(state.roster, key=lambda a: a.specialization_depth)
            papers = self.research.research_topic(
                query=state.config.problem_statement,
                context=state.plan or state.config.problem_statement,
                num_papers=3,
            )
            evolved = state.evolution_engine.evolve_agent(
                agent=weakest,
                context=context,
                papers=papers,
                trigger_reason=", ".join(r for _, r in triggers),
            )
            state.roster = [a for a in state.roster if a != weakest] + [evolved]
        state.iteration += 1
        state.context = self._build_context(
            state.config.problem_statement,
            state.config.data_context,
            state.math_state,
            state.history,
        )
        return state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_problem_graph(self, problem_statement: str) -> KnowledgeGraph:
        graph = KnowledgeGraph()
        for concept in extract_concepts_from_text(problem_statement):
            graph.add_concept(concept)
        return graph

    def _build_context(
        self,
        problem_statement: str,
        data_context: Dict[str, Any],
        math_state: MathematicalState,
        history: List[IterationRecord],
    ) -> ContextSnapshot:
        schema = {}
        for name, value in data_context.items():
            if hasattr(value, "columns"):
                schema[name] = list(value.columns)
        artifacts = {}
        if "artifacts_dir" in data_context:
            artifacts["artifacts_dir"] = str(data_context["artifacts_dir"])
        prior_metrics = [h.metrics for h in history]
        return ContextSnapshot(
            problem_summary=problem_statement,
            data_schema=schema,
            artifact_manifest=artifacts,
            prior_metrics=prior_metrics,
            math_state=math_state.summarize_for_prompt(),
        )

    def _meeting_agenda(self, state: GraphState) -> str:
        prior = "\n".join(
            [f"Iteration {h.iteration}: {h.metrics}" for h in state.history[-3:]]
        )
        return f"""Problem: {state.config.problem_statement}

Context:
- Target metric: {state.config.target_metric} ({'minimize' if state.config.minimize_metric else 'maximize'})
- Artifacts: {state.context.artifact_manifest}
- Data schema: {state.context.data_schema}
- Math state: {state.context.math_state}
- Recent metrics: {prior or 'None yet'}

Agenda:
1. Review what worked/failed so far.
2. Propose the next plan with grounded evidence (search allowed).
3. Keep prompts benchmark-agnostic and avoid food-specific phrasing.
"""

    def _implementation_prompt(self, state: GraphState) -> str:
        return f"""You are the coding agent translating the plan into code.

## Plan to implement
{state.plan}

## Problem statement
{state.config.problem_statement}

## Data available in execution context
{list(state.config.data_context.keys())}

## Requirements
- Use variables available in the execution context directly.
- Save important artifacts to {state.artifacts.artifacts_dir} when helpful.
- Print metrics clearly and ensure the target metric {state.config.target_metric} is computed.
- Output ONLY Python code wrapped in ```python``` fences.
"""

    def _strip_nonserializable(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        serializable = {}
        for k, v in payload.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                serializable[k] = v
        serializable["code"] = payload.get("code", "")
        serializable["description"] = payload.get("description", "")
        return serializable

    def _extract_metrics(self, results: Dict[str, Any], target_metric: str) -> Dict[str, float]:
        metrics = {}
        if not results:
            return metrics
        if "metrics" in results and isinstance(results["metrics"], dict):
            for k, v in results["metrics"].items():
                if isinstance(v, (int, float)):
                    metrics[k] = float(v)
        if target_metric in results and isinstance(results[target_metric], (int, float)):
            metrics[target_metric] = float(results[target_metric])
        return metrics

    def _update_best_metric(
        self,
        metrics: Dict[str, Any],
        target_metric: str,
        minimize_metric: bool,
        state: GraphState,
    ):
        if target_metric not in metrics:
            return
        current = metrics[target_metric]
        if state.best_metric is None:
            state.best_metric = current
            return
        if minimize_metric and current < state.best_metric:
            state.best_metric = current
        elif not minimize_metric and current > state.best_metric:
            state.best_metric = current

    def _termination_check(self, state: GraphState) -> str:
        goal_met = False
        if state.config.target_score is not None and state.metrics:
            metric = state.metrics.get(state.config.target_metric)
            if metric is not None:
                goal_met = (
                    metric <= state.config.target_score
                    if state.config.minimize_metric
                    else metric >= state.config.target_score
                )
        if goal_met:
            return "end"
        if state.iteration >= state.config.max_iterations:
            return "end"
        return "continue"

    # ------------------------------------------------------------------
    # Bootstrap flow
    # ------------------------------------------------------------------
    def _bootstrap_exploration(self, state: GraphState):
        exploration_prompt = f"""You are the Principal Investigator preparing a new project.

Problem: {state.config.problem_statement}

Data handles: {list(state.config.data_context.keys())}

Tasks:
1) Outline a quick data exploration plan (schemas, distributions, target sanity checks).
2) Identify expertise to recruit (diverse specialists + deepening generalist).
Keep it concise and benchmark-agnostic.
"""
        pi_meeting = IndividualMeeting(
            save_dir=str(state.artifacts.meetings_dir),
            research_api=state.research_api,
        )
        exploration_plan = pi_meeting.run(
            agent=self.team_lead,
            task=exploration_prompt,
            num_iterations=1,
            use_react=True,
        )

        coder = IndividualMeeting(save_dir=str(state.artifacts.meetings_dir))
        code_task = f"""Write Python code for the PI's exploration plan.

## PI Exploration Plan
{exploration_plan}

## Data available
{list(state.config.data_context.keys())}

Rules:
- Use pandas/numpy already imported.
- Only print observations (info/head/describe) and avoid speculative conclusions.
- Save any helpful artifacts to {state.artifacts.artifacts_dir}.
- Return ONLY Python code inside ```python``` fences.
"""
        code_output = coder.run(
            agent=state.coding_agent,
            task=code_task,
            num_iterations=1,
            use_react_coding=True,
        )
        execution = state.executor.execute_with_retry(
            code=code_output,
            description="Bootstrap exploration",
            max_retries=2,
        )
        bootstrap_record = IterationRecord(
            iteration=0,
            plan=exploration_plan,
            code=code_output,
            execution=self._strip_nonserializable(execution),
            metrics=self._extract_metrics(execution, state.config.target_metric),
            agents_snapshot=[a.title for a in [self.team_lead, state.coding_agent]],
        )
        state.history.append(bootstrap_record)
        save_json(
            {
                "iteration": 0,
                "phase": "bootstrap",
                "plan": exploration_plan,
                "metrics": bootstrap_record.metrics,
            },
            state.artifacts.base_dir / "iteration_00.json",
        )
        self._recruit_from_bootstrap(exploration_plan, state)

    def _recruit_from_bootstrap(self, exploration_plan: str, state: GraphState):
        recruitment_prompt = f"""Based on the PI's exploration and research, recruit 2-3 agents.

Problem: {state.config.problem_statement}
Exploration findings: {exploration_plan[:1500]}

For each recruit, provide title, expertise, and role. Balance specialists from diverse fields with a deepening generalist.
Return a bullet list.
"""
        meeting = IndividualMeeting(
            save_dir=str(state.artifacts.meetings_dir),
            research_api=state.research_api,
        )
        roster_text = meeting.run(
            agent=self.team_lead,
            task=recruitment_prompt,
            num_iterations=1,
            use_react=True,
        )
        new_agents: List[Agent] = []
        for line in roster_text.splitlines():
            if not line.strip().startswith("-"):
                continue
            payload = line.replace("-", "").strip()
            if not payload:
                continue
            new_agents.append(
                Agent(
                    title=payload,
                    expertise=payload,
                    goal="contribute expertise to the project",
                    role=payload,
                )
            )
        if new_agents:
            state.roster.extend(new_agents)

    def _final_summary(self, state: GraphState) -> Dict[str, Any]:
        return {
            "total_iterations": state.iteration,
            "best_metric": state.best_metric,
            "history": [h.__dict__ for h in state.history],
            "final_team": [
                {
                    "title": agent.title,
                    "expertise": agent.expertise,
                    "specialization_depth": agent.specialization_depth,
                }
                for agent in state.roster
            ],
        }
