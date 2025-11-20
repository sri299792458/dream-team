"""
LangGraph-based orchestrator for Dream Team.

Implements the Dream Team workflow as a state graph with proper
state management, tool use, and conditional routing.
"""

from typing import Dict, Any, List, Literal
from pathlib import Path
import json

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from .langgraph_state import (
    DreamTeamState,
    serialize_agent,
    deserialize_agent,
    IterationResult
)
from .langgraph_tools import (
    search_papers,
    execute_code,
    get_tools_for_agent,
    set_executor_context
)
from .agent import Agent
from .llm import get_llm
from .executor import extract_code_from_text
from .utils import save_json


# ============================================================================
# GRAPH NODES
# ============================================================================

def bootstrap_node(state: DreamTeamState) -> DreamTeamState:
    """
    Bootstrap phase: PI explores problem and recruits team.

    This node:
    1. Has PI explore the data with coding agent
    2. Extracts column schemas from exploration
    3. Has PI recruit team members based on findings
    4. Updates state with team and column schemas
    """
    print("\n" + "="*60)
    print("BOOTSTRAP: PI Initial Exploration")
    print("="*60)

    # Deserialize agents
    team_lead = deserialize_agent(state["team_lead"])
    coding_agent = deserialize_agent(state["coding_agent"])

    # Set up executor with data context
    set_executor_context(state["data_context"])

    llm = get_llm()

    # Step 1: PI decides what exploration is needed
    print(f"\n{team_lead.title} is exploring the problem...\n")

    exploration_task = f"""
You've received a new research problem. Before assembling a team, understand what you're dealing with.

## Problem:
{state['problem_statement']}

## Available Data:
{list(state['data_context'].keys())}

## Your Task:
Decide what initial exploration will help you understand:
1. What the data looks like (schemas, sizes, distributions)
2. What the challenge involves
3. What expertise you'll need on your team

In 2-3 sentences, describe what exploration code should be written.
"""

    exploration_plan = llm.generate(
        exploration_task,
        system_instruction=team_lead.prompt,
        temperature=0.7
    )

    print(f"\n{team_lead.title}'s plan:\n{exploration_plan}\n")

    # Step 2: Coding agent implements exploration
    print(f"💻 {coding_agent.title} implementing exploration...\n")

    code_task = f"""
Implement exploration code based on this plan:

{exploration_plan}

## Available dataframes:
{list(state['data_context'].keys())}

## Requirements:
- Print DataFrame shapes, columns, dtypes, summary statistics
- Check for missing values
- Show sample rows
- Output everything clearly so the PI can understand the data

Output ONLY Python code in ```python blocks.
"""

    code_output = llm.generate(
        code_task,
        system_instruction=coding_agent.prompt,
        temperature=0.3
    )

    code = extract_code_from_text(code_output)

    # Step 3: Execute exploration
    print("⚙️ Executing exploration...\n")

    from .executor import get_executor
    executor = get_executor()
    result = executor.execute(code=code, description="Bootstrap exploration")

    if not result['success']:
        print(f"❌ Exploration failed: {result['error']}\n")
        # Try simplified exploration
        fallback_code = """
import pandas as pd

for name, df in [(k, v) for k, v in globals().items() if isinstance(v, pd.DataFrame)]:
    print(f"\\n{'='*60}")
    print(f"DataFrame: {name}")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"\\nColumns: {list(df.columns)}")
    print(f"\\nDtypes:\\n{df.dtypes}")
    print(f"\\nSample:\\n{df.head()}")
"""
        result = executor.execute(code=fallback_code, description="Fallback exploration")

    output = result.get('output', '')
    print(f"Exploration output preview: {output[:500]}...\n")

    # Step 4: Extract column schemas
    column_schemas = _extract_column_schemas(output, state['data_context'])

    # Step 5: PI recruits team based on findings
    print(f"\n{team_lead.title} recruiting team members...\n")

    recruitment_task = f"""
Based on the data exploration, recruit 2-3 team members with expertise needed for this problem.

## Problem:
{state['problem_statement']}

## Exploration findings:
{output[:2000]}

## Your task:
Recruit 2-3 specialists. For each, provide:

Title: [Job title]
Expertise: [Comma-separated expertise areas]
Role: [What they'll contribute]

Keep it concise.
"""

    recruitment_output = llm.generate(
        recruitment_task,
        system_instruction=team_lead.prompt,
        temperature=0.7
    )

    # Parse recruitment output to create agents
    team_members = _parse_recruitment(recruitment_output)

    print(f"\n✅ Recruited {len(team_members)} team members:")
    for member in team_members:
        print(f"   - {member.title}")
    print()

    # Step 6: Save bootstrap results
    bootstrap_summary = {
        "iteration": 0,
        "approach": "Bootstrap exploration",
        "results": {
            "success": result['success'],
            "output": output,
            "code": code,
            "description": "Initial data exploration"
        },
        "metrics": {},
        "agents_snapshot": [team_lead.title] + [m.title for m in team_members]
    }

    # Update state
    return {
        **state,
        "bootstrap_completed": True,
        "column_schemas": column_schemas,
        "team_members": [serialize_agent(m) for m in team_members],
        "experiment_history": [bootstrap_summary],
        "iteration": 1
    }


def team_planning_node(state: DreamTeamState) -> DreamTeamState:
    """
    Team planning meeting: discuss approach.

    Uses multi-agent pattern where:
    1. Team lead opens meeting
    2. Each team member proposes approach (can use paper search)
    3. Team lead synthesizes into action plan
    """
    print(f"\n{'='*60}")
    print(f"ITERATION {state['iteration']} - TEAM PLANNING")
    print(f"{'='*60}\n")

    # Deserialize agents
    team_lead = deserialize_agent(state["team_lead"])
    team_members = [deserialize_agent(m) for m in state["team_members"]]

    llm = get_llm()

    # Build context from previous iterations
    history_context = _build_history_context(state)
    columns_context = _build_columns_context(state)

    agenda = f"""
**BE CONCISE.**

## Problem:
{state['problem_statement']}

## Available Dataframes:
{list(state['data_context'].keys())}
{columns_context}
{history_context}

## Roles:
- **Team Members**: Propose what to do next based on expertise
- **Lead**: Synthesize proposals into clear action plan

## Task:
Team members: Review previous results and propose next steps (2-3 sentences).
Lead: Synthesize into decisive action plan.
"""

    # Team lead opens meeting
    opening_prompt = f"""You are leading a team meeting.

Agenda: {agenda}

Open the meeting by:
1. Framing the problem
2. Asking key questions
3. Setting expectations

IMPORTANT: This is ONLY the opening. Do NOT generate proposals.
Keep it concise (2-3 paragraphs).
"""

    opening = llm.generate(opening_prompt, system_instruction=team_lead.prompt, temperature=0.7)
    print(f"💬 {team_lead.title}:\n{opening}\n")

    # Each team member proposes
    proposals = []
    for member in team_members:
        proposal_prompt = f"""You are participating in a team meeting.

Agenda: {agenda}

Lead's opening: {opening}

Provide your proposal as {member.title}. Draw on your expertise.
Keep it concise (2-3 sentences). What should we try next?
"""

        # Could integrate ReAct with paper search here
        # For now, simple generation
        proposal = llm.generate(proposal_prompt, system_instruction=member.prompt, temperature=0.7)
        proposals.append(proposal)
        print(f"💬 {member.title}:\n{proposal}\n")

    # Team lead synthesizes
    synthesis_prompt = f"""Synthesize the team's proposals into a clear action plan.

Agenda: {agenda}

Proposals:
{chr(10).join([f'- {p}' for p in proposals])}

Provide:
1. What we'll implement
2. Why this approach
3. Key requirements

Be decisive and concise (2-3 paragraphs).
"""

    synthesis = llm.generate(synthesis_prompt, system_instruction=team_lead.prompt, temperature=0.6)
    print(f"💬 {team_lead.title} (synthesis):\n{synthesis}\n")

    # Update state with approach
    return {
        **state,
        "current_approach": synthesis
    }


def code_generation_node(state: DreamTeamState) -> DreamTeamState:
    """
    Code generation: coding agent implements the approach.

    Uses ReAct pattern for iterative reasoning before coding.
    """
    print(f"\n💻 CODE GENERATION\n")

    # Deserialize coding agent
    coding_agent = deserialize_agent(state["coding_agent"])

    llm = get_llm()

    # Build context
    approach = state["current_approach"]
    columns_context = _build_columns_context(state)
    prev_output_context = _build_previous_output_context(state)

    # ReAct-style reasoning before coding
    print("   🧠 Planning implementation...\n")

    thoughts = []
    for step_num in range(3):
        if step_num == 0:
            prompt = f"""Plan how to implement this approach.

Approach: {approach}

Think about:
- Overall architecture/structure
- Main steps needed
- Libraries/methods to use

Output only: Thought: [Your thinking]
"""
        elif step_num == 1:
            prev_thoughts = "\n".join([f"Step {i+1}: {t}" for i, t in enumerate(thoughts)])
            prompt = f"""Refine implementation plan.

Approach: {approach}

Previous thinking:
{prev_thoughts}

Think about:
- Implementation details
- Edge cases
- Data flow

Output only: Thought: [Your thinking]
"""
        else:
            prev_thoughts = "\n".join([f"Step {i+1}: {t}" for i, t in enumerate(thoughts)])
            prompt = f"""Finalize implementation plan.

Approach: {approach}

Previous thinking:
{prev_thoughts}

Think about:
- Missing pieces
- Correctness checks
- Optimizations

Output only: Thought: [Your thinking]
"""

        thought = llm.generate(prompt, system_instruction=coding_agent.prompt, temperature=0.5)
        thought = thought.replace("Thought:", "").strip()
        thoughts.append(thought)
        print(f"   Step {step_num + 1}: {thought[:100]}...\n")

    # Generate code
    print("   ✍️ Writing code...\n")

    code_task = f"""
Implement the team's plan.

## Team's Plan:
{approach}

## Reasoning:
{chr(10).join([f'{i+1}. {t}' for i, t in enumerate(thoughts)])}

## Available dataframes:
{list(state['data_context'].keys())}
{columns_context}
{prev_output_context}

## Requirements:
- Use GPU for training
- Write complete, executable code
- Import needed libraries
- Use EXACT column names from schemas
- Compute MAE and store in variable
- Print important outputs
- Save models (joblib.dump, torch.save)
- Suppress verbose output

Output ONLY Python code in ```python blocks.
"""

    code_output = llm.generate(code_task, system_instruction=coding_agent.prompt, temperature=0.3)
    code = extract_code_from_text(code_output)

    # Save code
    code_dir = Path(state["code_dir"])
    code_dir.mkdir(parents=True, exist_ok=True)
    code_file = code_dir / f"iteration_{state['iteration']:02d}.py"
    code_file.write_text(code)

    print(f"   Generated {len(code.split(chr(10)))} lines")
    print(f"   Saved to: {code_file}\n")

    # Update state
    return {
        **state,
        "current_code": code
    }


def execution_node(state: DreamTeamState) -> DreamTeamState:
    """
    Execute code and evaluate results.

    Handles:
    - Code execution with error recovery (up to 2 retries)
    - Metric extraction
    - Best metric tracking
    """
    print(f"\n⚙️ EXECUTION & EVALUATION\n")

    code = state["current_code"]
    llm = get_llm()

    # Deserialize coding agent for error recovery
    coding_agent = deserialize_agent(state["coding_agent"])

    from .executor import get_executor
    executor = get_executor()

    # Execute with retry
    max_retries = 2
    attempt = 0
    result = None

    while attempt <= max_retries:
        if attempt > 0:
            print(f"   🔄 Retry {attempt}/{max_retries}\n")

        result = executor.execute(code=code, description=f"Iteration {state['iteration']}")

        if result['success']:
            if attempt > 0:
                print(f"   ✅ Fixed after {attempt} attempt(s)!\n")
            break

        # If failed and retries left, try to fix
        if attempt < max_retries:
            print(f"   ❌ Execution failed: {result['error']}\n")
            print("   Asking coding agent to fix...\n")

            # Get error context
            error_context = result.get('output', '')[-2000:] if result.get('output') else ''

            fix_task = f"""
The code failed with an error. Fix it.

## Original task:
{state['current_approach']}

## Your code:
```python
{code}
```

## Error:
{result['error']}

## Traceback:
{result.get('traceback', '')}

## Recent output:
{error_context}

Output the FIXED code in ```python blocks.
"""

            fix_output = llm.generate(fix_task, system_instruction=coding_agent.prompt, temperature=0.3)
            code = extract_code_from_text(fix_output)

        attempt += 1

    # Extract metrics
    metrics = _extract_metrics(result, state['target_metric'])

    # Update best metric
    best_metric = state.get("best_metric")
    best_iteration = state.get("best_iteration")

    if metrics and state['target_metric'] in metrics:
        current_value = metrics[state['target_metric']]

        if best_metric is None:
            best_metric = current_value
            best_iteration = state['iteration']
        else:
            if state['minimize_metric']:
                if current_value < best_metric:
                    best_metric = current_value
                    best_iteration = state['iteration']
            else:
                if current_value > best_metric:
                    best_metric = current_value
                    best_iteration = state['iteration']

    # Print summary
    print(f"\n{'='*60}")
    print(f"ITERATION {state['iteration']} SUMMARY")
    print(f"{'='*60}")
    print(f"Status: {'✅ Success' if result['success'] else '❌ Failed'}")
    if metrics:
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    else:
        print("No metrics extracted")
    if best_metric is not None:
        print(f"Best {state['target_metric']} so far: {best_metric:.4f}")
    print(f"{'='*60}\n")

    # Save iteration results
    iteration_result: IterationResult = {
        "iteration": state['iteration'],
        "approach": state['current_approach'],
        "results": {
            "success": result['success'],
            "output": result['output'],
            "error": result.get('error'),
            "traceback": result.get('traceback'),
            "code": code,
            "description": f"Iteration {state['iteration']}"
        },
        "metrics": metrics,
        "agents_snapshot": [state['team_lead']['title']] + [m['title'] for m in state['team_members']]
    }

    results_dir = Path(state["results_dir"])
    save_json(iteration_result, results_dir / f"iteration_{state['iteration']:02d}.json")

    safe_result = _make_msgpack_safe(result)
    safe_iteration_result = _make_msgpack_safe(iteration_result)

    # Update state
    updates = {
        **state,
        "current_results": safe_result,
        "current_metrics": _make_msgpack_safe(metrics),
        "best_metric": best_metric,
        "best_iteration": best_iteration,
        "experiment_history": state["experiment_history"] + [safe_iteration_result],
        "error_count": state["error_count"] + (0 if result['success'] else 1)
    }

    return updates


def check_completion_node(state: DreamTeamState) -> DreamTeamState:
    """
    Check if experiment should continue.

    Sets flags for:
    - goal_achieved: target score reached
    - should_evolve: team evolution needed
    """
    # Check goal achievement
    goal_achieved = False

    if state.get('target_score') is not None and state.get('current_metrics'):
        metric_value = state['current_metrics'].get(state['target_metric'])

        if metric_value is not None:
            if state['minimize_metric']:
                goal_achieved = metric_value <= state['target_score']
            else:
                goal_achieved = metric_value >= state['target_score']

    # Check if should evolve (simple heuristic for now)
    should_evolve = False

    # Evolve if:
    # - Multiple failures (error_count >= 2)
    # - Or performance plateau (last 2 iterations no improvement)
    if state['error_count'] >= 2:
        should_evolve = True
        print("🧬 Evolution triggered: multiple failures\n")
    elif len(state['experiment_history']) >= 3:
        # Check plateau
        recent_metrics = [
            h['metrics'].get(state['target_metric'])
            for h in state['experiment_history'][-3:]
            if h['metrics'].get(state['target_metric']) is not None
        ]

        if len(recent_metrics) >= 2:
            # No improvement if values are very similar
            if abs(max(recent_metrics) - min(recent_metrics)) < 0.001:
                should_evolve = True
                print("🧬 Evolution triggered: performance plateau\n")

    return {
        **state,
        "goal_achieved": goal_achieved,
        "should_evolve": should_evolve
    }


def evolution_node(state: DreamTeamState) -> DreamTeamState:
    """
    Evolve team members based on current needs.

    For now, simplified evolution that deepens expertise.
    """
    print(f"\n{'='*60}")
    print("TEAM EVOLUTION")
    print(f"{'='*60}\n")

    llm = get_llm()

    # Evolve one team member (the least specialized)
    team_members = [deserialize_agent(m) for m in state["team_members"]]

    if not team_members:
        return {**state, "should_evolve": False, "error_count": 0}

    # Choose member with lowest specialization depth
    target_member = min(team_members, key=lambda m: m.specialization_depth)

    print(f"Evolving: {target_member.title}\n")

    evolution_prompt = f"""
An agent needs to evolve to address current challenges.

Current agent:
- Title: {target_member.title}
- Expertise: {target_member.expertise}
- Specialization depth: {target_member.specialization_depth}

Recent challenges:
{state.get('current_results', {}).get('error', 'Performance plateau')}

Propose evolution:
- New Title: [More specialized title]
- New Expertise: [Deeper, more specific expertise]
- New Role: [Updated role]

Keep it concise.
"""

    evolution_output = llm.generate(evolution_prompt, temperature=0.7)

    # Parse (simple parsing)
    new_title = target_member.title + " (Evolved)"
    new_expertise = target_member.expertise + ", advanced techniques"
    new_role = target_member.role

    # Apply evolution
    target_member.evolve(
        new_title=new_title,
        new_expertise=new_expertise,
        new_role=new_role,
        trigger_reason="Performance issues / plateau"
    )

    # Update state
    updated_members = [serialize_agent(m) for m in team_members]

    return {
        **state,
        "team_members": updated_members,
        "should_evolve": False,
        "error_count": 0  # Reset error count
    }


def increment_iteration_node(state: DreamTeamState) -> DreamTeamState:
    """Increment iteration counter"""
    return {
        **state,
        "iteration": state["iteration"] + 1
    }


# ============================================================================
# CONDITIONAL ROUTING
# ============================================================================

def should_continue(state: DreamTeamState) -> Literal["continue", "evolve", "end"]:
    """
    Decide next step after checking completion.

    Returns:
    - "end" if goal achieved or max iterations reached
    - "evolve" if evolution needed
    - "continue" otherwise
    """
    if state.get("goal_achieved", False):
        print("🎯 Target achieved!\n")
        return "end"

    if state["iteration"] >= state["max_iterations"]:
        print("⏱️ Max iterations reached\n")
        return "end"

    if state.get("should_evolve", False):
        return "evolve"

    return "continue"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _extract_column_schemas(output: str, data_context: Dict) -> Dict[str, List[str]]:
    """Extract column names from exploration output"""
    import re

    schemas = {}

    # Look for DataFrame column patterns
    for line in output.split('\n'):
        if 'Columns:' in line or 'columns:' in line:
            # Extract list
            match = re.search(r'\[(.*?)\]', line)
            if match:
                cols_str = match.group(1)
                cols = [c.strip().strip("'\"") for c in cols_str.split(',')]
                # Try to match to a dataframe name
                for df_name in data_context.keys():
                    if df_name in line:
                        schemas[df_name] = cols
                        break

    return schemas


def _parse_recruitment(recruitment_text: str) -> List[Agent]:
    """Parse recruitment output into Agent objects"""
    agents = []
    current_agent = {}

    for line in recruitment_text.split('\n'):
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
                    goal="contribute specialized expertise",
                    role=current_agent['role']
                )
                agents.append(agent)
                current_agent = {}

    # Fallback
    if not agents:
        agents = [Agent(
            title="ML Strategist",
            expertise="machine learning, feature engineering, predictive modeling",
            goal="design effective approaches",
            role="propose modeling strategies"
        )]

    return agents


def _build_history_context(state: DreamTeamState) -> str:
    """Build context from experiment history"""
    if not state.get("experiment_history"):
        return ""

    history = state["experiment_history"]

    if len(history) == 1:
        # Only bootstrap
        output = history[0]['results'].get('output', '')
        preview = output[:3000] if len(output) > 3000 else output
        return f"\n## Bootstrap Exploration Output:\n```\n{preview}\n```\n"

    # Multiple iterations
    context = "\n## Iteration History:\n"

    # Last 3 iterations
    recent = [h for h in history if h.get('iteration', 0) > 0][-3:]

    for h in recent:
        iter_num = h['iteration']
        metrics = h.get('metrics', {})
        approach = h.get('approach', '')
        approach_preview = approach[:150] + "..." if len(approach) > 150 else approach
        context += f"- Iteration {iter_num}: {metrics}\n  Approach: {approach_preview}\n"

    if state.get('best_metric') is not None:
        context += f"\n**Best {state['target_metric']} so far**: {state['best_metric']:.4f}\n"

    # Add previous iteration output
    last = history[-1]
    output = last['results'].get('output', '')
    if output:
        preview = output[-5000:] if len(output) > 5000 else output
        context += f"\n## Previous Iteration Output:\n```\n{preview}\n```\n"

    return context


def _build_columns_context(state: DreamTeamState) -> str:
    """Build column schemas context"""
    if not state.get("column_schemas"):
        return ""

    context = "\n## AVAILABLE COLUMNS (use exact names):\n"
    for df_name, cols in state["column_schemas"].items():
        context += f"\n{df_name}: {cols}\n"

    return context


def _build_previous_output_context(state: DreamTeamState) -> str:
    """Build previous output context for coding agent"""
    if not state.get("experiment_history"):
        return ""

    last = state["experiment_history"][-1]
    output = last['results'].get('output', '')

    if not output:
        return ""

    # For bootstrap, show first 3000 chars
    if last.get('iteration', 0) == 0:
        preview = output[:3000] if len(output) > 3000 else output
        return f"\n## Bootstrap Output:\n```\n{preview}\n```\n"

    # For iterations, show last 5000 chars
    preview = output[-5000:] if len(output) > 5000 else output
    return f"\n## Previous Output:\n```\n{preview}\n```\n"


def _extract_metrics(result: Dict, target_metric: str) -> Dict[str, Any]:
    """Extract metrics from execution result"""
    metrics = {}

    if not result.get('success'):
        return metrics

    output = result.get('output', '')

    # Look for metric values in output
    import re

    # Pattern: metric_name = value or metric_name: value
    patterns = [
        r'(\w+)\s*=\s*([\d.]+)',
        r'(\w+)\s*:\s*([\d.]+)',
        r'(\w+)\s+is\s+([\d.]+)'
    ]

    for pattern in patterns:
        matches = re.findall(pattern, output.lower())
        for metric_name, value in matches:
            try:
                metrics[metric_name] = float(value)
            except:
                pass

    return metrics


def _make_msgpack_safe(value: Any) -> Any:
    """Recursively coerce values into msgpack-friendly forms."""
    import numpy as np
    import pandas as pd

    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, pd.DataFrame):
        return {
            "_type": "DataFrame",
            "shape": [int(value.shape[0]), int(value.shape[1])],
            "columns": value.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in value.dtypes.items()},
        }
    if isinstance(value, pd.Series):
        return {
            "_type": "Series",
            "length": int(len(value)),
            "dtype": str(value.dtype),
            "name": value.name,
        }

    if isinstance(value, dict):
        return {k: _make_msgpack_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        coerced = [_make_msgpack_safe(v) for v in value]
        return coerced if isinstance(value, list) else tuple(coerced)

    try:
        return repr(value)
    except Exception:
        return "<unserializable>"


# ============================================================================
# GRAPH BUILDER
# ============================================================================

def create_dream_team_graph(checkpointer=None):
    """
    Create the Dream Team LangGraph.

    Returns:
        Compiled graph ready for execution
    """
    # Create graph
    graph = StateGraph(DreamTeamState)

    # Add nodes
    graph.add_node("bootstrap", bootstrap_node)
    graph.add_node("team_planning", team_planning_node)
    graph.add_node("code_generation", code_generation_node)
    graph.add_node("execution", execution_node)
    graph.add_node("check_completion", check_completion_node)
    graph.add_node("evolution", evolution_node)
    graph.add_node("increment_iteration", increment_iteration_node)

    # Entry point
    graph.set_entry_point("bootstrap")

    # Bootstrap -> team_planning
    graph.add_edge("bootstrap", "team_planning")

    # Main iteration cycle
    graph.add_edge("team_planning", "code_generation")
    graph.add_edge("code_generation", "execution")
    graph.add_edge("execution", "check_completion")

    # Conditional routing from check_completion
    graph.add_conditional_edges(
        "check_completion",
        should_continue,
        {
            "continue": "increment_iteration",
            "evolve": "evolution",
            "end": END
        }
    )

    # After evolution, continue
    graph.add_edge("evolution", "increment_iteration")

    # After incrementing, loop back to planning
    graph.add_edge("increment_iteration", "team_planning")

    # Compile
    if checkpointer is None:
        checkpointer = MemorySaver()

    return graph.compile(checkpointer=checkpointer)
