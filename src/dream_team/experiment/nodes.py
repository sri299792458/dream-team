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
        # Build data info
        data_list = [k for k in ctx.data_context.keys() if k != 'artifacts_dir']

        exploration_task = f"""
You've received a new research problem. Before assembling a team, you need to understand what you're dealing with.

## Problem:
{state.config.problem_statement}

## Available Data (ALREADY LOADED in memory):
{', '.join(data_list)}

These dataframes are ALREADY LOADED. You can use them directly (e.g., `batches_train.head()`).
DO NOT re-load data from files or create dummy/synthetic data.

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
                    # Convert all column names to strings for Pydantic validation
                    cols = [str(col) for col in df.columns]
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


def create_init_math_framework_node(ctx: ExecutionContext):
    """Create init math framework node with access to execution context"""

    def init_math_framework_node(state: ExperimentState) -> ExperimentState:
        """
        Initialize mathematical framework for the experiment.

        Migrated from ExperimentOrchestrator._initialize_mathematical_framework
        """
        print("\n🧮 Initializing mathematical framework...")

        # Extract problem as knowledge graph
        problem_text = f"{state.config.problem_statement} Target metric: {state.config.target_metric}"
        concepts = extract_concepts_from_text(problem_text, use_llm=False)

        # Create problem graph
        ctx.problem_graph = KnowledgeGraph()

        # Add domain-specific concepts
        important_concepts = {
            'regression', 'classification', 'prediction', 'forecasting',
            'optimization', 'machine_learning', 'deep_learning',
            'gradient_boosting', 'neural_network', 'feature_engineering',
            state.config.target_metric.lower().replace('_', ' ')
        }

        all_concepts = concepts | important_concepts

        for concept in all_concepts:
            if concept.lower() in state.config.problem_statement.lower():
                importance = 2.0
            elif concept == state.config.target_metric.lower().replace('_', ' '):
                importance = 3.0
            else:
                importance = 1.0

            ctx.problem_graph.add_concept(concept, importance=importance)

        print(f"   ✓ Problem graph: {len(ctx.problem_graph.concepts)} concepts")

        # Store concepts in state
        state.mathematical_state.problem_concepts = list(ctx.problem_graph.concepts)

        # Create team object
        ctx.team = Team(ctx.all_agents)
        print(f"   ✓ Team initialized: {len(ctx.team.agents)} agents")

        # Compute initial diversity
        diversity = ctx.team.compute_diversity()
        state.mathematical_state.team_diversity = diversity
        print(f"   ✓ Team diversity: {diversity:.3f}")

        state.phase = "plan"
        return state

    return init_math_framework_node


def _build_comprehensive_history(state: ExperimentState, max_iterations: int = 5) -> str:
    """
    Build comprehensive history context showing patterns across iterations.

    Includes:
    - What approaches were tried
    - What errors occurred
    - What worked vs what didn't
    - Patterns to avoid
    """
    if not state.history:
        return ""

    sections = []

    # Section 1: Iteration-by-iteration summary (last N iterations)
    recent_history = state.history[-max_iterations:]
    if recent_history:
        history_lines = ["## Iteration History (most recent first):"]
        for hist in reversed(recent_history):
            status = "✅ SUCCESS" if hist.results.get('success') else "❌ FAILED"
            metrics_str = str(hist.metrics) if hist.metrics else "no metrics"

            # Show approach (first 200 chars)
            approach_preview = (hist.approach[:200] + "...") if hist.approach and len(hist.approach) > 200 else (hist.approach or "N/A")

            history_lines.append(f"\n### Iteration {hist.iteration}: {status}")
            history_lines.append(f"Approach: {approach_preview}")
            history_lines.append(f"Metrics: {metrics_str}")

            # Show error if failed
            if not hist.results.get('success') and hist.results.get('error'):
                error_preview = hist.results['error'][:150]
                history_lines.append(f"Error: {error_preview}")

        sections.append("\n".join(history_lines))

    # Section 2: Error patterns (what went wrong repeatedly)
    errors = []
    for hist in state.history:
        if not hist.results.get('success') and hist.results.get('error'):
            errors.append({
                'iteration': hist.iteration,
                'error': hist.results['error'],
                'approach': hist.approach[:100] if hist.approach else ''
            })

    if errors:
        error_lines = ["\n## Error Patterns (DO NOT REPEAT THESE MISTAKES):"]
        for err in errors[-3:]:  # Last 3 errors
            error_lines.append(f"- Iteration {err['iteration']}: {err['error'][:100]}")
        sections.append("\n".join(error_lines))

    # Section 3: What has worked (successful patterns)
    successes = []
    for hist in state.history:
        if hist.results.get('success') and hist.metrics:
            successes.append({
                'iteration': hist.iteration,
                'metrics': hist.metrics,
                'approach': hist.approach[:150] if hist.approach else ''
            })

    if successes:
        success_lines = ["\n## Successful Approaches (build on these):"]
        for succ in successes[-3:]:  # Last 3 successes
            success_lines.append(f"- Iteration {succ['iteration']}: {succ['metrics']}")
            success_lines.append(f"  Approach: {succ['approach']}")
        sections.append("\n".join(success_lines))

    # Section 4: Best metric tracking
    if state.best_metric is not None:
        sections.append(f"\n## Best {state.config.target_metric} so far: {state.best_metric:.4f} (iteration {state.best_iteration})")

    return "\n".join(sections)


def create_plan_node(ctx: ExecutionContext):
    """Create plan node with access to execution context"""

    def plan_node(state: ExperimentState) -> ExperimentState:
        """
        Team planning meeting to decide approach.

        Migrated from ExperimentOrchestrator._team_planning_meeting
        """
        print("\n👥 Team planning meeting...\n")

        # Build comprehensive history context
        history_context = _build_comprehensive_history(state, max_iterations=5)

        # Get last iteration output for detailed review
        last_output_preview = ""
        if state.history:
            last = state.history[-1]
            if last.results and last.results.get('output'):
                output = last.results['output']
                if last.iteration == 0:  # Bootstrap
                    preview_len = min(3000, len(output))
                    last_output_preview = f"\n\n## Bootstrap Output (first {preview_len} chars):\n```\n{output[:preview_len]}\n```"
                else:
                    preview_len = min(10000, len(output))
                    last_output_preview = f"\n\n## Last Iteration Output (last {preview_len} chars):\n```\n...{output[-preview_len:]}\n```"

        # Column schemas
        columns_summary = ""
        if state.column_schemas:
            columns_summary = "\n## AVAILABLE COLUMNS (ONLY use these exact column names):\n"
            for df_name, cols in state.column_schemas.items():
                columns_summary += f"\n{df_name}: {cols}\n"

        agenda = f"""
## Problem:
{state.config.problem_statement}

## Available Dataframes:
{list(ctx.data_context.keys())}
{columns_summary}

{history_context}

{last_output_preview}

## CRITICAL INSTRUCTIONS:
- Review the Error Patterns section above - DO NOT repeat those mistakes
- Build on Successful Approaches - iterate on what worked
- If the same error occurs twice, try a FUNDAMENTALLY DIFFERENT approach

## Roles:
- **Team Members**: Analyze history, identify patterns, propose DIFFERENT approaches if stuck
- **Lead**: Synthesize into decisive action plan that avoids past mistakes

## Task:
Team members: What patterns do you see? What should we try DIFFERENTLY?
Lead: Make a decision that breaks out of any failure patterns.
"""

        meeting = TeamMeeting(
            save_dir=str(ctx.results_dir / 'meetings'),
            research_api=ctx.research_api
        )
        summary = meeting.run(
            team_lead=ctx.team_lead,
            team_members=ctx.team_members,
            agenda=agenda,
            num_rounds=1
        )

        # Save meeting transcript
        meeting.save(f'iteration_{state.iteration + 1:02d}_team_meeting.json')

        # Extract summary text
        summary_text = summary.get('summary', '') if isinstance(summary, dict) else str(summary)

        print("\n📝 TEAM SYNTHESIS:")
        preview = summary_text[:200] + "..." if len(summary_text) > 200 else summary_text
        print(f"   {preview}\n")

        state.current_approach = summary_text
        state.phase = "code"
        return state

    return plan_node


def create_code_node(ctx: ExecutionContext):
    """Create code node with access to execution context"""

    def code_node(state: ExperimentState) -> ExperimentState:
        """
        Coding agent implements the planned approach.

        Migrated from ExperimentOrchestrator._implement_approach
        """
        print(f"💻 {ctx.coding_agent.title} implementing approach...\n")

        # Build previous output context
        previous_output_context = ""
        if state.history:
            last = state.history[-1]
            if last.results and last.results.get('output'):
                output = last.results['output']
                if last.iteration == 0:
                    preview_len = min(3000, len(output))
                    previous_output_context = f"\n## Bootstrap Exploration Output (first {preview_len} chars):\n```\n{output[:preview_len]}\n```\n"
                else:
                    preview_len = min(15000, len(output))
                    if len(output) > 15000:
                        previous_output_context = f"\n## Previous Iteration Output (last {preview_len} chars):\n```\n...{output[-preview_len:]}\n```\n"
                    else:
                        previous_output_context = f"\n## Previous Iteration Output:\n```\n{output}\n```\n"

        # Column schemas
        schema_info = ""
        if state.column_schemas:
            schema_info = "\n## DataFrame Schemas (use EXACT column names):\n"
            for df_name, cols in state.column_schemas.items():
                schema_info += f"- {df_name}: {cols}\n"

        # Build explicit data context info
        data_info = "\n## Data Available (ALREADY LOADED in memory - use directly):\n"
        for df_name in ctx.data_context.keys():
            if df_name == 'artifacts_dir':
                continue
            df = ctx.data_context.get(df_name)
            if hasattr(df, '__len__') and hasattr(df, 'columns'):
                data_info += f"- {df_name}: DataFrame with {len(df)} rows (use as `{df_name}`, already in memory)\n"
            else:
                data_info += f"- {df_name}: {type(df).__name__}\n"
        data_info += "\n**CRITICAL**: These dataframes are ALREADY LOADED. Use them directly (e.g., `batches_train.head()`).\n"
        data_info += "**DO NOT** re-load data from files or create synthetic/dummy data.\n"

        task = f"""
Implement the team's plan.

## Team's Plan:
{state.current_approach}
{data_info}
{schema_info}
{previous_output_context}
## Pre-imported libraries:
- pandas as pd
- numpy as np
- torch

## Requirements:
- Write complete, executable code that uses the REAL data listed above
- Use EXACT column names from DataFrame Schemas above
- **CRITICAL**: After computing your metric, assign it to a variable named EXACTLY `{state.config.target_metric}` (lowercase)
  Example: `{state.config.target_metric} = computed_metric_value`  # NOT {state.config.target_metric.upper()}, NOT mean_absolute_error, etc.
- Print important outputs: metrics, feature importance, model summaries
- Save trained models if training took long
- Suppress verbose output: warnings.filterwarnings('ignore'), use verbose=0 or verbose=-1

Output ONLY Python code in ```python blocks.
"""

        meeting = IndividualMeeting(save_dir=str(ctx.results_dir / 'meetings'))
        code_output = meeting.run(
            agent=ctx.coding_agent,
            task=task,
            num_iterations=1,
            use_react_coding=True
        )

        # Save coding meeting
        meeting.save(f'iteration_{state.iteration + 1:02d}_coding.json')

        # Extract code
        code = extract_code_from_text(code_output)

        # Save code
        code_file = ctx.results_dir / 'code' / f'iteration_{state.iteration + 1:02d}.py'
        code_file.parent.mkdir(exist_ok=True)
        code_file.write_text(code)

        print(f"   Generated {len(code.split(chr(10)))} lines")
        print(f"   Saved to: {code_file}\n")

        state.current_code = code
        state.phase = "execute"
        return state

    return code_node


def _refresh_column_schemas(ctx: ExecutionContext, state: ExperimentState):
    """
    Refresh column schemas from current executor state.

    This captures any new DataFrames created during execution
    (e.g., engineered features, transformed data).
    """
    import pandas as pd

    for key, value in ctx.executor.data_context.items():
        if isinstance(value, pd.DataFrame):
            # Convert all column names to strings for Pydantic validation
            cols = [str(col) for col in value.columns]

            # Check if this is new or changed
            if key not in state.column_schemas:
                print(f"   📋 New DataFrame detected: {key} ({len(cols)} columns)")
                state.column_schemas[key] = cols
            elif state.column_schemas[key] != cols:
                old_count = len(state.column_schemas[key])
                print(f"   📋 DataFrame updated: {key} ({old_count} → {len(cols)} columns)")
                state.column_schemas[key] = cols


def create_execute_node(ctx: ExecutionContext):
    """Create execute node with access to execution context"""

    def execute_node(state: ExperimentState) -> ExperimentState:
        """
        Execute the generated code with retry.

        Migrated from ExperimentOrchestrator._execute_with_retry
        """
        print("⚙️  Executing implementation...\n")

        max_retries = 2
        current_code = state.current_code
        attempt = 0
        fix_attempts = []  # Track what fixes were attempted

        while attempt <= max_retries:
            if attempt > 0:
                print(f"   🔄 Retry attempt {attempt}/{max_retries}\n")

            # Execute code
            result = ctx.executor.execute(
                code=current_code,
                description=f"Iteration {state.iteration + 1} implementation"
            )

            # If successful, save and return
            if result['success']:
                if attempt > 0:
                    print(f"   ✅ Fixed after {attempt} attempt(s)!\n")

                # Refresh column schemas from executor state
                _refresh_column_schemas(ctx, state)

                state.current_results = {
                    'success': result['success'],
                    'output': result['output'],
                    'error': result.get('error'),
                    'traceback': result.get('traceback'),
                    'code': current_code,
                    'description': state.current_approach or ''
                }
                state.phase = "evaluate"
                return state

            # Check for missing package
            if 'missing_package' in result:
                package = result['missing_package']
                print(f"   📦 Missing package: {package}")
                if ctx.executor._install_package(package):
                    print(f"   🔄 Retrying after installing {package}...\n")
                    continue

            # If failed and we have retries left, ask agent to fix
            if attempt < max_retries:
                print(f"   ❌ Error: {result['error']}")
                print(f"   🔧 Asking agent to fix...\n")

                # Pass fix history to the fixer
                current_code, fix_description = _fix_code_error_with_history(
                    ctx=ctx,
                    state=state,
                    failed_code=current_code,
                    error=result['error'],
                    traceback=result.get('traceback', ''),
                    previous_attempts=fix_attempts
                )

                # Record this attempt
                fix_attempts.append({
                    'attempt': attempt + 1,
                    'error': result['error'][:200],
                    'fix_description': fix_description
                })

                # Save retry code
                code_file = ctx.results_dir / 'code' / f'iteration_{state.iteration + 1:02d}_retry_{attempt+1}.py'
                code_file.parent.mkdir(exist_ok=True)
                code_file.write_text(current_code)

            attempt += 1

        # Max retries exhausted
        print(f"   ⚠️  Max retries ({max_retries}) exhausted. Moving on with failure.\n")
        state.current_results = {
            'success': False,
            'output': result.get('output', ''),
            'error': result.get('error'),
            'traceback': result.get('traceback'),
            'code': current_code,
            'description': state.current_approach or ''
        }
        state.phase = "evaluate"
        return state

    return execute_node


def _fix_code_error_with_history(
    ctx: ExecutionContext,
    state: ExperimentState,
    failed_code: str,
    error: str,
    traceback: str,
    previous_attempts: list
) -> tuple[str, str]:
    """
    Ask coding agent to fix failed code, providing history of previous fix attempts.

    Returns: (fixed_code, description_of_fix)
    """
    # Build previous attempts context
    attempts_context = ""
    if previous_attempts:
        attempts_context = "\n## PREVIOUS FIX ATTEMPTS (DO NOT REPEAT THESE):\n"
        for att in previous_attempts:
            attempts_context += f"- Attempt {att['attempt']}: Tried to fix '{att['error'][:100]}'\n"
            attempts_context += f"  What was tried: {att['fix_description']}\n"
        attempts_context += "\n**You MUST try something DIFFERENT from the above attempts.**\n"

    # Previous output context
    previous_output_context = ""
    if state.history:
        last = state.history[-1]
        if last.results and last.results.get('output'):
            output = last.results['output']
            preview_len = min(15000, len(output))
            if len(output) > 15000:
                previous_output_context = f"\n## Previous Iteration Output (last {preview_len} chars):\n```\n...{output[-preview_len:]}\n```\n"
            else:
                previous_output_context = f"\n## Previous Iteration Output:\n```\n{output}\n```\n"

    # Column schemas
    schema_str = ""
    if state.column_schemas:
        for df_name, cols in state.column_schemas.items():
            schema_str += f"{df_name}: {cols}\n"

    task = f"""
Your code failed with an error. Fix it.

## Original Approach
{state.current_approach}

## Problem Statement (for reference):
{state.config.problem_statement}

## Your Code That Failed
```python
{failed_code}
```

## Error
{error}

## Traceback
{traceback}

{attempts_context}

## Available in execution context:
- Pre-imported libraries: pandas, numpy, torch, pathlib
- Variables: {list(ctx.data_context.keys())}

## DataFrame Schemas (use EXACT column names):
{schema_str}
{previous_output_context}
## Task
The error shows EXACTLY what's wrong. Read the traceback line number.

**IMPORTANT**: {f"You already tried {len(previous_attempts)} fix(es) that didn't work. Try something FUNDAMENTALLY DIFFERENT." if previous_attempts else ""}

**For NameError `'X' is not defined`:**
1. Look at the line number
2. Find where you used variable `X` without defining it
3. Either define `X = ...` BEFORE that line, or remove the usage

**For KeyError (column doesn't exist):**
- Check the DataFrame Schemas above for the EXACT column name

**CRITICAL REMINDER**: Your final metric MUST be assigned to a variable named exactly `{state.config.target_metric}` (lowercase).

**DO NOT output the same code again. Actually fix the specific line that failed.**

Before outputting code, briefly describe what you're changing (1 sentence).
Then output ONLY the FIXED Python code in ```python blocks.
"""

    meeting = IndividualMeeting(save_dir=str(ctx.results_dir / 'meetings'))
    code_output = meeting.run(
        agent=ctx.coding_agent,
        task=task,
        num_iterations=1
    )

    # Extract fix description (first line before code block)
    lines = code_output.split('\n')
    fix_description = ""
    for line in lines:
        if line.strip() and not line.strip().startswith('```'):
            fix_description = line.strip()
            break

    return extract_code_from_text(code_output), fix_description


def _update_agent_knowledge(ctx: ExecutionContext, state: ExperimentState):
    """
    Extract learnings from iteration and update agent knowledge bases.

    This ensures agents accumulate knowledge across iterations.
    """
    success = state.current_results.get('success', False) if state.current_results else False

    if success and state.current_metrics:
        # Record successful pattern
        pattern = f"Iteration {state.iteration + 1}: {state.current_approach[:150] if state.current_approach else 'N/A'} -> {state.current_metrics}"

        for agent in ctx.all_agents:
            agent.knowledge_base.successful_patterns.append(pattern)

            # Also add technique if we can identify it
            if state.current_approach:
                approach_lower = state.current_approach.lower()
                techniques = []
                if 'gradient boosting' in approach_lower or 'xgboost' in approach_lower or 'lightgbm' in approach_lower:
                    techniques.append('gradient_boosting')
                if 'neural' in approach_lower or 'deep learning' in approach_lower:
                    techniques.append('neural_networks')
                if 'feature engineering' in approach_lower:
                    techniques.append('feature_engineering')
                if 'cross-validation' in approach_lower or 'cv' in approach_lower:
                    techniques.append('cross_validation')

                for tech in techniques:
                    agent.knowledge_base.add_technique(tech)

    elif not success and state.current_results:
        # Record error insight
        error = state.current_results.get('error', 'Unknown error')
        insight = f"Iteration {state.iteration + 1}: {error[:200]}"

        for agent in ctx.all_agents:
            if insight not in agent.knowledge_base.error_insights:
                agent.knowledge_base.error_insights.append(insight)

        # Extract specific error types
        if 'KeyError' in error:
            fact = f"KeyError encountered - always verify column names exist before using"
            for agent in ctx.all_agents:
                agent.knowledge_base.add_fact(fact, source=f"Iteration {state.iteration + 1} error")

        if 'NameError' in error:
            fact = f"NameError encountered - always define variables before using them"
            for agent in ctx.all_agents:
                agent.knowledge_base.add_fact(fact, source=f"Iteration {state.iteration + 1} error")


def create_evaluate_node(ctx: ExecutionContext):
    """Create evaluate node with access to execution context"""

    def evaluate_node(state: ExperimentState) -> ExperimentState:
        """
        Evaluate execution results and extract metrics.

        Migrated from ExperimentOrchestrator._extract_metrics
        """
        print(f"\n📊 Evaluating iteration {state.iteration + 1}...\n")

        # Extract metrics from execution results
        import numpy as np

        raw_metrics = {}
        if state.current_results and state.current_results.get('success'):
            # Get metrics from executor variables
            variables = ctx.executor.data_context

            # Look for target metric
            target_metric = state.config.target_metric
            metric_names = ['mae', 'rmse', 'f1', 'accuracy', 'score', 'cv_scores', 'error']

            for key, value in variables.items():
                # Check if this matches our target metric
                if target_metric.lower() in key.lower():
                    try:
                        if isinstance(value, (int, float, np.integer, np.floating)):
                            raw_metrics[target_metric] = float(value)
                        elif isinstance(value, (list, np.ndarray)):
                            raw_metrics[target_metric] = float(np.mean(value))
                    except:
                        pass

                # Also collect other metrics
                for metric_name in metric_names:
                    if metric_name in key.lower():
                        try:
                            if isinstance(value, (int, float, np.integer, np.floating)):
                                raw_metrics[key] = float(value)
                            elif isinstance(value, (list, np.ndarray)):
                                raw_metrics[key] = float(np.mean(value))
                        except:
                            pass

        state.current_metrics = raw_metrics

        # Update best metric
        is_new_best = state.update_best_metric()
        if is_new_best:
            print(f"   ✨ New best {state.config.target_metric}: {state.best_metric:.4f}")

        # Print iteration summary
        print(f"\n{'='*60}")
        print(f"ITERATION {state.iteration + 1} SUMMARY")
        print(f"{'='*60}")
        print(f"Status: {'✅ Success' if state.current_results.get('success') else '❌ Failed'}")
        if state.current_metrics:
            for k, v in state.current_metrics.items():
                print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        else:
            print("No metrics extracted")
        if state.best_metric is not None:
            print(f"Best {state.config.target_metric} so far: {state.best_metric:.4f}")
        print(f"{'='*60}\n")

        # Extract and store learnings in agent knowledge bases
        _update_agent_knowledge(ctx, state)

        # Save iteration summary
        summary = state.get_iteration_summary()
        summary.iteration = state.iteration + 1  # Set to next iteration number
        state.history.append(summary)

        save_json(
            summary.model_dump(),
            ctx.results_dir / f'iteration_{state.iteration + 1:02d}.json'
        )

        # Increment iteration counter
        state.iteration += 1

        state.phase = "check_continue"
        return state

    return evaluate_node


def create_check_evolution_node(ctx: ExecutionContext):
    """Create check evolution node with access to execution context"""

    def check_evolution_node(state: ExperimentState) -> ExperimentState:
        """
        Check if team evolution is needed.

        Migrated from ExperimentOrchestrator._check_mathematical_evolution
        """
        print("\n📊 Checking for evolution signals...\n")

        if len(state.history) < 3:
            print("   ℹ️  Not enough history for evolution check")
            state.evolution.triggered = False
            state.phase = "plan"
            return state

        # Build metric history
        metric_history = []
        for h in state.history:
            if state.config.target_metric in h.metrics:
                metric_history.append(h.metrics[state.config.target_metric])
            else:
                metric_history.append(float('inf') if state.config.minimize_metric else float('-inf'))

        # Update agent dynamics
        if ctx.team and ctx.problem_graph:
            # Compute learning quality
            if len(metric_history) >= 2:
                recent_improvement = metric_history[-2] - metric_history[-1] if state.config.minimize_metric else metric_history[-1] - metric_history[-2]

                if recent_improvement > 0:
                    base_quality = 0.8
                elif abs(recent_improvement) < 0.01:
                    base_quality = 0.5
                else:
                    base_quality = 0.3

                learning_quality = {
                    concept: base_quality
                    for concept in ctx.problem_graph.concepts
                }

                # Update all agents' dynamics
                ctx.team.update_all_dynamics(
                    problem=ctx.problem_graph,
                    learning_quality=learning_quality,
                    dt=0.1
                )

            # Get team state
            team_state = ctx.team.diagnose_state(metric_history, minimize=state.config.minimize_metric)
            diversity = ctx.team.compute_diversity()

            print(f"   Team state: {team_state}")
            print(f"   Team diversity: {diversity:.3f}")

            # Update mathematical state
            state.mathematical_state.team_diversity = diversity
            state.mathematical_state.iteration_count = ctx.team.iteration

            # Check for evolution signals
            evolution_signals = []
            for agent in ctx.all_agents:
                should_evolve, evo_type = agent.should_evolve(ctx.problem_graph, ctx.team)
                if should_evolve:
                    evolution_signals.append((agent, evo_type))
                    gini = agent.δ.gini_coefficient()
                    effectiveness = agent.contribution_effectiveness()
                    print(f"   🔔 {agent.title}: {evo_type} (gini={gini:.2f}, eff={effectiveness:.2f})")

            # Trigger evolution if needed
            if evolution_signals or team_state in ["REFRAMING", "EXPLORATION"]:
                print(f"\n🔔 Evolution triggered")
                state.evolution.triggered = True
                state.evolution.trigger_names = [name for _, name in evolution_signals]
                if team_state in ["REFRAMING", "EXPLORATION"]:
                    state.evolution.trigger_names.append(f"TEAM_{team_state}")
                state.phase = "evolve"
            else:
                print("   ✓ No evolution needed")
                state.evolution.triggered = False
                state.phase = "plan"

        else:
            # Fallback to simple plateau detection
            should_evolve = state.should_evolve()
            if should_evolve:
                print("   🔔 Evolution triggered (plateau detected)")
                state.evolution.triggered = True
                state.evolution.trigger_names = ["PLATEAU"]
                state.phase = "evolve"
            else:
                print("   ✓ No evolution needed")
                state.evolution.triggered = False
                state.phase = "plan"

        return state

    return check_evolution_node


def create_evolve_node(ctx: ExecutionContext):
    """Create evolve node with access to execution context"""

    def evolve_node(state: ExperimentState) -> ExperimentState:
        """
        Evolve team composition based on performance.

        Migrated from ExperimentOrchestrator._evolve_team
        """
        print("\n🧬 Evolving team composition...\n")

        # Research papers
        print("📚 Researching latest approaches...\n")
        papers_summary = ""
        try:
            from ..llm import get_llm
            llm = get_llm()

            query_prompt = f"""
Extract 2-3 key academic search terms from this problem for searching research papers.

Problem: {state.config.problem_statement[:300]}

Output ONLY the search query (2-5 words, academic terminology, no quotes).
Examples: "shelf life prediction", "gradient boosting regression"
"""
            search_query = llm.generate(query_prompt, temperature=0.3).strip().strip('"\'')
            print(f"   Search query: '{search_query}'")

            if ctx.research_api:
                from ..research import get_research_assistant
                research = get_research_assistant()
                papers = research.research_topic(
                    query=search_query,
                    context=f"Current performance: {state.current_metrics}",
                    num_papers=2
                )

                if papers:
                    papers_summary = "\n## Research Findings:\n"
                    for paper in papers:
                        papers_summary += f"- {paper.title}: {paper.abstract[:120]}...\n"
        except Exception as e:
            print(f"   Note: Research skipped: {e}\n")

        # PI analyzes team
        current_team_info = "\n".join([
            f"- {agent.title}: {agent.expertise[:100]}"
            for agent in ctx.team_members
        ])

        recent_history = ""
        if len(state.history) >= 3:
            recent_history = "\n## Recent Progress:\n"
            for hist in state.history[-3:]:
                metrics_str = str(hist.metrics) if hist.metrics else "FAILED"
                recent_history += f"Iteration {hist.iteration}: {metrics_str}\n"

        situation_desc = "Progress has stalled."
        if not state.current_metrics:
            situation_desc = "The most recent iteration FAILED to produce metrics."

        evolution_task = f"""
Analyze the current situation and decide whether team evolution is needed.

## Situation:
{situation_desc}

## Current Team:
{current_team_info if current_team_info else "Only you (PI)"}

## Current Performance:
{state.current_metrics if state.current_metrics else "No metrics from last iteration"}

{recent_history}
{papers_summary}

## Your Options:
1. NO CHANGE: Current team is fine
2. ADD a new specialist
3. REMOVE an agent
4. DEEPEN an existing agent
5. MULTIPLE changes

## Your Task:
Is this a TEAM COMPOSITION issue or an IMPLEMENTATION issue?

Specify your decision:
- NO CHANGE: [Reason]
OR
- ADD: [Title] with expertise in [expertise] to [role]
- REMOVE: [Title] because [reason]
- DEEPEN: [Title] into [New Title] with expertise in [expertise]
"""

        meeting = IndividualMeeting(save_dir=str(ctx.results_dir / 'meetings'))
        evolution_plan = meeting.run(
            agent=ctx.team_lead,
            task=evolution_task,
            num_iterations=1
        )

        print(f"\n{ctx.team_lead.title}'s evolution plan:\n{evolution_plan}\n")

        # Execute evolution plan
        _execute_evolution_plan(ctx, state, evolution_plan, papers if 'papers' in locals() else [])

        # Update state from agents
        ctx.update_state_from_agents(state)

        state.evolution.triggered = False
        state.phase = "plan"
        return state

    return evolve_node


def _execute_evolution_plan(ctx: ExecutionContext, state: ExperimentState, plan: str, papers: List):
    """Helper to execute PI's evolution plan"""

    changes_made = []

    for line in plan.split('\n'):
        line = line.strip()

        # NO CHANGE
        if line.upper().startswith('NO CHANGE:'):
            reason = line.split(':', 1)[1].strip() if ':' in line else "Team composition is adequate"
            print(f"\n✋ No team evolution needed: {reason}")
            state.evolution.decision = "NO_CHANGE"
            state.evolution.reason = reason
            return

        # ADD agent
        elif line.upper().startswith('ADD:'):
            new_agent = Agent(
                title="Domain Specialist",
                expertise="specialized domain knowledge",
                goal="break through performance plateau",
                role="apply domain-specific insights"
            )
            ctx.team_members.append(new_agent)
            ctx.all_agents = [ctx.team_lead] + ctx.team_members
            if ctx.team:
                ctx.team.agents = ctx.all_agents
            changes_made.append(f"✅ Added {new_agent.title}")
            state.evolution.decision = "ADD_AGENT"

        # REMOVE agent
        elif line.upper().startswith('REMOVE:'):
            if ctx.team_members:
                removed = ctx.team_members.pop(0)
                ctx.all_agents = [ctx.team_lead] + ctx.team_members
                if ctx.team:
                    ctx.team.agents = ctx.all_agents
                changes_made.append(f"✅ Removed {removed.title}")
                state.evolution.decision = "REMOVE_AGENT"

        # DEEPEN agent
        elif line.upper().startswith('DEEPEN:'):
            if ctx.team_members:
                agent = ctx.team_members[0]
                old_title = agent.title

                context = {
                    'problem_description': plan,
                    'deepening': True
                }
                ctx.evolution_engine.evolve_agent(
                    agent=agent,
                    context=context,
                    papers=papers,
                    trigger_reason="Team evolution - specialization needed"
                )

                changes_made.append(f"✅ Deepened {old_title} → {agent.title}")
                state.evolution.decision = "DEEPEN_AGENT"

    if changes_made:
        print("\n🔄 Team Evolution Complete:")
        for change in changes_made:
            print(f"   {change}")

        print(f"\n👥 New team composition:")
        print(f"   - {ctx.team_lead.title} (Lead)")
        for agent in ctx.team_members:
            print(f"   - {agent.title}")
        print()

        # Save evolution record
        evolution_record = {
            'iteration': state.iteration,
            'evolution_plan': plan,
            'changes': changes_made,
            'new_team': [{'title': a.title, 'expertise': a.expertise} for a in ctx.all_agents]
        }
        save_json(evolution_record, ctx.results_dir / f'evolution_iter_{state.iteration}.json')
