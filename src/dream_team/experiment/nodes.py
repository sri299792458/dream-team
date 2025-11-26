"""
Node implementations for LangGraph experiment orchestration.

This module contains the refactored logic from ExperimentOrchestrator,
organized as clean node functions that operate on ExperimentState.

Each node:
- Accepts ExperimentState
- Uses existing domain logic (agents, executor, etc.)
- Updates state fields appropriately
- Returns updated ExperimentState
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import pandas as pd

from .state import ExperimentState, AgentConfig, IterationSummary
from ..agent import Agent
from ..executor import CodeExecutor, extract_code_from_text
from ..meetings import TeamMeeting, IndividualMeeting
from ..evolution import EvolutionEngine
from ..knowledge_state import KnowledgeGraph, extract_concepts_from_text
from ..team import Team
from ..utils import save_json
from ..research import get_research_assistant


# ============================================================================
# Context Management
# ============================================================================

class ExecutionContext:
    """
    Manages non-serializable objects needed during execution.

    Holds references to executor, agents, research API, etc.
    This avoids storing these in ExperimentState.
    """

    def __init__(
        self,
        data_context: Dict[str, Any],
        results_dir: Path,
        research_api: Optional[Any] = None
    ):
        self.data_context = data_context
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Create executor
        # Add artifacts_dir for agents to save objects
        artifacts_dir = self.results_dir / 'artifacts'
        artifacts_dir.mkdir(exist_ok=True)
        data_context['artifacts_dir'] = artifacts_dir

        self.executor = CodeExecutor(data_context=data_context)

        # Research API
        if research_api is None:
            research = get_research_assistant()
            self.research_api = research.ss_api if hasattr(research, 'ss_api') else None
        else:
            self.research_api = research_api

        # Agent instances (will be created from state configs)
        self.team_lead: Optional[Agent] = None
        self.team_members: List[Agent] = []
        self.coding_agent: Optional[Agent] = None
        self.all_agents: List[Agent] = []

        # Evolution engine
        self.evolution_engine = EvolutionEngine()

        # Mathematical framework objects
        self.problem_graph: Optional[KnowledgeGraph] = None
        self.team: Optional[Team] = None

    def create_agents_from_state(self, state: ExperimentState):
        """Create Agent instances from state configuration"""
        # Create team lead
        lead_cfg = state.team.team_lead
        self.team_lead = Agent(
            title=lead_cfg.title,
            expertise=lead_cfg.expertise,
            goal=lead_cfg.goal,
            role=lead_cfg.role,
            model=lead_cfg.model,
            specialization_depth=lead_cfg.specialization_depth
        )

        # Create team members
        self.team_members = [
            Agent(
                title=m.title,
                expertise=m.expertise,
                goal=m.goal,
                role=m.role,
                model=m.model,
                specialization_depth=m.specialization_depth
            )
            for m in state.team.team_members
        ]

        # Create coding agent
        coding_cfg = state.team.coding_agent
        self.coding_agent = Agent(
            title=coding_cfg.title,
            expertise=coding_cfg.expertise,
            goal=coding_cfg.goal,
            role=coding_cfg.role,
            model=coding_cfg.model,
            specialization_depth=coding_cfg.specialization_depth
        )

        # Update all agents list
        self.all_agents = [self.team_lead] + self.team_members

    def update_state_from_agents(self, state: ExperimentState):
        """Update state configuration from current agent instances"""
        # Update team lead config
        state.team.team_lead = AgentConfig(
            title=self.team_lead.title,
            expertise=self.team_lead.expertise,
            goal=self.team_lead.goal,
            role=self.team_lead.role,
            model=self.team_lead.model,
            specialization_depth=self.team_lead.specialization_depth
        )

        # Update team members config
        state.team.team_members = [
            AgentConfig(
                title=agent.title,
                expertise=agent.expertise,
                goal=agent.goal,
                role=agent.role,
                model=agent.model,
                specialization_depth=agent.specialization_depth
            )
            for agent in self.team_members
        ]

        # Update coding agent config
        state.team.coding_agent = AgentConfig(
            title=self.coding_agent.title,
            expertise=self.coding_agent.expertise,
            goal=self.coding_agent.goal,
            role=self.coding_agent.role,
            model=self.coding_agent.model,
            specialization_depth=self.coding_agent.specialization_depth
        )


# ============================================================================
# Node Functions (using ExecutionContext)
# ============================================================================

def create_bootstrap_node(ctx: ExecutionContext):
    """Create bootstrap node with access to execution context"""

    def bootstrap_node(state: ExperimentState) -> ExperimentState:
        """
        Bootstrap phase: PI explores problem and recruits team.

        Migrated from ExperimentOrchestrator._bootstrap_exploration
        """
        print("\n" + "="*60)
        print("NODE: Bootstrap")
        print("="*60)

        if state.bootstrap_completed:
            print("   ✓ Bootstrap already completed")
            state.phase = "init_math"
            return state

        # Create initial agents (PI and coding agent only)
        ctx.create_agents_from_state(state)

        print(f"\n{ctx.team_lead.title} is exploring the problem alone...\n")

        # PI decides what exploration is needed
        exploration_task = f"""
You've received a new research problem. Before assembling a team, you need to understand what you're dealing with.

## Problem:
{state.config.problem_statement}

## Available Data:
{list(ctx.data_context.keys())}

## Your Task:
Decide what initial exploration will help you understand:
1. What the data looks like (schemas, sizes, distributions)
2. What the challenge involves
3. What expertise you'll need on your team

In 2-3 sentences, describe what exploration code should be written.
"""

        meeting = IndividualMeeting(save_dir=str(ctx.results_dir / 'meetings'))
        exploration_plan = meeting.run(
            agent=ctx.team_lead,
            task=exploration_task,
            num_iterations=1
        )

        print(f"\n{ctx.team_lead.title}'s exploration plan:\n{exploration_plan}\n")

        # Coding agent implements exploration
        print(f"💻 {ctx.coding_agent.title} implementing exploration...\n")

        # Build available data context string
        data_keys = list(ctx.data_context.keys())

        code_task = f"""
The PI wants to do initial exploration. Write Python code to implement this:

## PI's Request:
{exploration_plan}

## Problem Statement (for reference):
{state.config.problem_statement}

## Available in execution context:
- Pre-imported libraries: pandas (pd), numpy (np), torch, pathlib.Path
- Variables: {data_keys}
  (You can use any of these variables directly in your code)

## Requirements:
- Inspect dataframes: print(df.info()), df.head(), df.describe(), df.columns
- ONLY print what you observe - no summaries, interpretations, or conclusions
- Use variables from "Available in execution context" above
- Suppress warnings if needed

Output ONLY the Python code, wrapped in ```python code blocks.
"""

        code_meeting = IndividualMeeting(save_dir=str(ctx.results_dir / 'meetings'))
        code_output = code_meeting.run(
            agent=ctx.coding_agent,
            task=code_task,
            num_iterations=1,
            use_react_coding=True
        )

        code = extract_code_from_text(code_output)

        # Save exploration code
        code_file = ctx.results_dir / 'code' / 'iteration_00.py'
        code_file.parent.mkdir(exist_ok=True)
        code_file.write_text(code)

        # Execute exploration
        print("⚙️  Executing exploration...\n")
        results = ctx.executor.execute(code=code, description="Bootstrap exploration")

        if results['success']:
            print("✅ Exploration successful!\n")
            print("Output:")
            print("-" * 60)
            output_preview = results['output'][:1000] if len(results['output']) > 1000 else results['output']
            print(output_preview)
            print("-" * 60)

            # Extract column schemas
            print("\n📋 Extracting column schemas...")
            for key in data_keys:
                df = ctx.executor.get_variable(key)
                if df is not None and hasattr(df, 'columns'):
                    cols = list(df.columns)
                    state.column_schemas[key] = cols
                    print(f"   {key}: {len(cols)} columns")

        # PI recruits team
        print(f"\n{ctx.team_lead.title} recruiting team...\n")

        recruitment_task = f"""
Based on the problem and exploration results, decide what expertise you need on your team.

## Problem:
{state.config.problem_statement}

## Exploration Results:
{results['output'][:2000] if results['success'] else "Exploration failed"}

## Your Task:
List 1-3 team members you want to recruit. For each, provide:
- Title (e.g., "ML Strategist", "Domain Expert")
- Expertise (what they should know)
- Role (what they'll contribute)

Be specific about the skills needed.

Format your response as a list.
"""

        recruitment_meeting = IndividualMeeting(
            save_dir=str(ctx.results_dir / 'meetings'),
            research_api=ctx.research_api
        )
        recruitment_plan = recruitment_meeting.run(
            agent=ctx.team_lead,
            task=recruitment_task,
            num_iterations=1,
            use_react=True
        )

        print(f"Recruitment plan:\n{recruitment_plan}\n")

        # Parse and recruit agents
        recruited = _parse_and_recruit_agents(ctx, recruitment_plan, results_dir=ctx.results_dir)

        # Add to context
        ctx.team_members.extend(recruited)
        ctx.all_agents = [ctx.team_lead] + ctx.team_members

        # Update state from agents
        ctx.update_state_from_agents(state)

        print(f"\n✅ Team assembled! {len(recruited)} member(s) recruited:")
        for agent in recruited:
            print(f"   - {agent.title}")

        # Save bootstrap summary
        bootstrap_summary = IterationSummary(
            iteration=0,
            phase='bootstrap',
            approach=exploration_plan,
            code=code,
            results={
                'success': results['success'],
                'output': results['output'],
                'error': results.get('error'),
                'traceback': results.get('traceback'),
            },
            metrics={},
            agents_snapshot=[ctx.team_lead.title, ctx.coding_agent.title],
            recruitment_plan=recruitment_plan,
            recruited_agents=[
                {
                    'title': a.title,
                    'expertise': a.expertise,
                    'role': a.role,
                    'goal': a.goal
                }
                for a in recruited
            ]
        )

        state.history.append(bootstrap_summary)
        save_json(
            bootstrap_summary.model_dump(),
            ctx.results_dir / 'iteration_00_bootstrap.json'
        )

        state.bootstrap_completed = True
        state.phase = "init_math"

        return state

    return bootstrap_node


def _parse_and_recruit_agents(ctx: ExecutionContext, recruitment_plan: str, results_dir: Path) -> List[Agent]:
    """Helper to parse recruitment plan and create agents"""

    parse_task = f"""
Parse this recruitment plan and extract agent specifications.

## Recruitment Plan:
{recruitment_plan}

## Your Task:
For each team member mentioned, extract:
- Title
- Expertise
- Role

Output in this exact format (one agent per block):

AGENT 1:
Title: [exact title from plan]
Expertise: [expertise description]
Role: [role description]

AGENT 2:
Title: [exact title from plan]
Expertise: [expertise description]
Role: [role description]

Only output the agent specifications, nothing else.
"""

    meeting = IndividualMeeting(save_dir=str(results_dir / 'meetings'))
    parsed_output = meeting.run(
        agent=ctx.team_lead,
        task=parse_task,
        num_iterations=1
    )

    # Parse structured output
    agents = []
    current_agent = {}

    for line in parsed_output.split('\n'):
        line = line.strip()

        if line.startswith('Title:'):
            current_agent['title'] = line.replace('Title:', '').strip()
        elif line.startswith('Expertise:'):
            current_agent['expertise'] = line.replace('Expertise:', '').strip()
        elif line.startswith('Role:'):
            current_agent['role'] = line.replace('Role:', '').strip()

            # Create agent when we have all fields
            if all(k in current_agent for k in ['title', 'expertise', 'role']):
                agent = Agent(
                    title=current_agent['title'],
                    expertise=current_agent['expertise'],
                    goal="contribute specialized expertise to optimize the target metric",
                    role=current_agent['role']
                )
                agents.append(agent)
                current_agent = {}

    # Fallback if parsing failed
    if not agents:
        print("   ⚠️  Could not parse recruitment, creating default ML Strategist")
        agents = [Agent(
            title="ML Strategist",
            expertise="machine learning, feature engineering, model selection",
            goal="design effective predictive approaches",
            role="propose modeling strategies"
        )]

    return agents


# TODO: Continue implementing other nodes (plan, code, execute, evaluate, evolve)
# This would be continued in the actual implementation...
