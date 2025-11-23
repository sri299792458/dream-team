"""
Enhanced LangGraph orchestrator for Dream Team (V2).

Improvements over V1:
- Uses proper ReAct agents with create_react_agent()
- Smart context management with semantic summarization
- Multi-agent team meeting subgraph
- Mathematical state (K, θ, δ) integrated into prompts
- Streaming support
- Better error diagnostics
"""

import atexit
from typing import Dict, Any, List, Literal, Optional
from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage

from .langgraph_state import (
    DreamTeamState,
    serialize_agent,
    deserialize_agent,
    IterationResult
)
from .langgraph_agents import (
    invoke_coding_agent,
    invoke_research_agent,
    invoke_planning_agent
)
from .langgraph_context import (
    build_adaptive_context,
    format_context_for_planning,
    format_context_for_coding
)
from .langgraph_team_meeting import run_team_meeting
from .agent import Agent
from .llm import get_llm
from .executor import extract_code_from_text
from .utils import save_json
from .langgraph_tools import set_executor_context
from .serialization import make_msgpack_safe


# ============================================================================
# ENHANCED GRAPH NODES
# ============================================================================

def bootstrap_node_v2(state: DreamTeamState) -> DreamTeamState:
    """
    Enhanced bootstrap with ReAct agents.
    """
    print("\n" + "="*60)
    print("BOOTSTRAP: PI Initial Exploration (Enhanced)")
    print("="*60)

    team_lead = state["team_lead"]
    coding_agent = state["coding_agent"]

    set_executor_context(state["data_context"])

    if not state["data_context"]:
        print("⚠️  No data_context provided; skipping bootstrap execution and using placeholder summary.\n")
        exploration_plan = "No data available; skipping exploration until data_context is populated."
        output = "No dataframes available; provide data_context to run exploration."
    else:
        # Step 1: PI plans exploration using ReAct
        print(f"\n{team_lead['title']} planning exploration...\n")

        exploration_task = f"""You've received a new research problem. Plan initial data exploration.

Problem:
{state['problem_statement']}

Available Data:
{list(state['data_context'].keys())}

Decide what exploration code should be written to understand:
1. Data schemas, sizes, distributions
2. The challenge
3. What expertise is needed for the team

Output 2-3 sentences describing the exploration plan.
"""

        exploration_result = invoke_planning_agent(team_lead, exploration_task)
        exploration_plan = exploration_result["response"]

        print(f"\nExploration plan: {exploration_plan}\n")

        # Step 2: Coding agent implements using ReAct
        print(f"💻 {coding_agent['title']} implementing...\n")

        # Get dataframe info for the prompt
        df_info_lines = []
        for df_name, df in state['data_context'].items():
            df_info_lines.append(f"- {df_name}: {df.shape[0]} rows")

        code_task = f"""Write Python code for this exploration:

{exploration_plan}

Available in-memory pandas DataFrames (already loaded for you; do NOT read from disk):
{chr(10).join(df_info_lines)}

Requirements:
- Use the provided DataFrames exactly as named above; do NOT call pd.read_csv or assume file paths.
- For EACH dataframe, print: 'Columns: [exact_column_list]' using list(df.columns)
- Print DataFrame shapes, dtypes, and basic statistics
- Check for missing values
- Show sample rows
        - **Never** index a column unless you've confirmed it exists (e.g., `'<some_column>' in df.columns`). If a column is missing, skip that analysis gracefully.
        - DO NOT assume column names - discover them from the actual dataframes

Output ONLY Python code in ```python blocks.
"""

        code_result = invoke_coding_agent(
            coding_agent,
            code_task,
            max_iterations=3
        )

        code = extract_code_from_text(code_result["response"])

        # Step 3: Execute
        print("⚙️ Executing exploration...\n")

        from .executor import get_executor
        executor = get_executor()
        result = executor.execute(code=code, description="Bootstrap exploration")

        if not result['success']:
            print(f"❌ Failed: {result['error']}\n")
            # Fallback code
            code = """
import pandas as pd
for name, df in [(k, v) for k, v in globals().items() if isinstance(v, pd.DataFrame)]:
    print(f"DataFrame: {name}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Dtypes:\\n{df.dtypes}")
    print(f"Sample:\\n{df.head()}\\n")
"""
            result = executor.execute(code=code, description="Fallback exploration")

        output = result.get('output', '')

    # Step 4: Extract column schemas
    column_schemas = _extract_column_schemas(output, state['data_context'])

    # Step 5: Recruit team using ReAct
    print(f"\n{team_lead['title']} recruiting team...\n")

    recruitment_task = f"""Based on data exploration, recruit 2-3 specialists for this problem.

Problem:
{state['problem_statement']}

Exploration findings:
{output[:2000]}

For each team member, provide:
Title: [Job title]
Expertise: [Expertise areas]
Role: [What they'll contribute]

Be concise.
"""

    recruitment_result = invoke_planning_agent(team_lead, recruitment_task)
    team_members = _parse_recruitment(recruitment_result["response"])

    print(f"\n✅ Recruited {len(team_members)} members:")
    for m in team_members:
        print(f"   - {m.title}")
    print()

    # Save bootstrap results
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
        "agents_snapshot": [team_lead['title']] + [m.title for m in team_members]
    }

    safe_bootstrap_summary = _make_msgpack_safe(bootstrap_summary)

    updated_state: DreamTeamState = {
        **state,
        "bootstrap_completed": True,
        "column_schemas": column_schemas,
        "team_members": [serialize_agent(m) for m in team_members],
        "experiment_history": [safe_bootstrap_summary],
        "iteration": 1,
    }

    return _checkpoint_safe_state(updated_state)


def team_planning_node_v2(state: DreamTeamState) -> DreamTeamState:
    """
    Enhanced team planning using multi-agent meeting subgraph.
    """
    print(f"\n{'='*60}")
    print(f"ITERATION {state['iteration']} - TEAM PLANNING (Enhanced)")
    print(f"{'='*60}\n")

    # Build smart context
    context = build_adaptive_context(state)
    context_text = format_context_for_planning(context)

    agenda = f"""**BE CONCISE.**

{context_text}

## Task:
Team members: Propose next steps based on your expertise (2-3 sentences).
Lead: Synthesize into decisive action plan.
"""

    # Run team meeting (uses subgraph)
    meeting_result = run_team_meeting(
        team_lead=state["team_lead"],
        team_members=state["team_members"],
        agenda=agenda
    )

    synthesis = meeting_result["synthesis"]
    meeting_messages = _serialize_messages(meeting_result.get("messages", []))
    meeting_papers = meeting_result.get("papers_found", [])

    # Update agent KBs with papers found
    # (In full implementation, would update state["team_members"])

    updated_state = {
        **state,
        "current_approach": synthesis,
        "planning_context": context_text,
        "meeting_agenda": agenda,
        "meeting_messages": meeting_messages,
        "meeting_papers": meeting_papers
    }

    return _checkpoint_safe_state(updated_state)


def code_generation_node_v2(state: DreamTeamState) -> DreamTeamState:
    """
    Enhanced code generation with ReAct coding agent.
    """
    print(f"\n💻 CODE GENERATION (Enhanced)\n")

    coding_agent = state["coding_agent"]
    approach = state["current_approach"]

    # Build smart context
    context = build_adaptive_context(state)
    context_text = format_context_for_coding(context, approach)

    code_task = f"""{context_text}

## Requirements:
- Use GPU for training
- Write complete, executable code
- Import needed libraries
- Use EXACT column names from schemas
- Compute {state['target_metric'].upper()} and store in variable '{state['target_metric']}'
- Print important outputs
- Save models (joblib.dump, torch.save)
- Suppress verbose output

Output ONLY Python code in ```python blocks.
"""

    print("   Using ReAct agent for implementation...\n")

    # Invoke with ReAct
    result = invoke_coding_agent(
        coding_agent,
        code_task,
        max_iterations=5
    )

    code = extract_code_from_text(result["response"])

    # Save code
    code_dir = Path(state["code_dir"])
    code_dir.mkdir(parents=True, exist_ok=True)
    code_file = code_dir / f"iteration_{state['iteration']:02d}.py"
    code_file.write_text(code)

    print(f"   Generated {len(code.split(chr(10)))} lines")
    print(f"   Saved to: {code_file}\n")

    updated_state: DreamTeamState = {
        **state,
        "current_code": code,
        "coding_context": context_text
    }

    return _checkpoint_safe_state(updated_state)


def execution_node_v2(state: DreamTeamState) -> DreamTeamState:
    """
    Enhanced execution with better error diagnostics.
    """
    print(f"\n⚙️ EXECUTION & EVALUATION (Enhanced)\n")

    code = state["current_code"]
    coding_agent = state["coding_agent"]

    from .executor import get_executor
    executor = get_executor()

    # Execute with retry and diagnostics
    max_retries = 2
    attempt = 0
    result = None
    error_history = []

    while attempt <= max_retries:
        if attempt > 0:
            print(f"   🔄 Retry {attempt}/{max_retries}\n")

        result = executor.execute(code=code, description=f"Iteration {state['iteration']}")

        if result['success']:
            if attempt > 0:
                print(f"   ✅ Fixed after {attempt} attempt(s)!\n")
            break

        # Track error
        error_history.append({
            "attempt": attempt,
            "error": result['error'],
            "code_snippet": code[:200]
        })

        # If failed and retries left, use ReAct agent to fix
        if attempt < max_retries:
            print(f"   ❌ Execution failed: {result['error']}\n")
            print("   Using ReAct agent to diagnose and fix...\n")

            # Build diagnostic context
            error_context = result.get('output', '')[-2000:] if result.get('output') else ''

            fix_task = f"""The code failed. Diagnose the issue and provide a fix.

Original task:
{state['current_approach']}

Your code:
```python
{code}
```

Error:
{result['error']}

Traceback:
{result.get('traceback', '')}

Recent output:
{error_context}

Diagnose the problem and output the FIXED code in ```python blocks.
"""

            # Use ReAct to fix
            fix_result = invoke_coding_agent(
                coding_agent,
                fix_task,
                max_iterations=5
            )

            code = extract_code_from_text(fix_result["response"])

            # Save fixed code
            code_dir = Path(state["code_dir"])
            code_file = code_dir / f"iteration_{state['iteration']:02d}_attempt_{attempt+1}.py"
            code_file.write_text(code)

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
    if error_history:
        print(f"Errors encountered: {len(error_history)}")
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
            "description": f"Iteration {state['iteration']}",
            "error_history": error_history
        },
        "metrics": metrics,
        "context": {
            "planning": state.get("planning_context", ""),
            "coding": state.get("coding_context", "")
        },
        "meeting": {
            "agenda": state.get("meeting_agenda", ""),
            "messages": state.get("meeting_messages", []),
            "papers_found": state.get("meeting_papers", [])
        },
        "agents_snapshot": [state['team_lead']['title']] + [m['title'] for m in state['team_members']]
    }

    results_dir = Path(state["results_dir"])
    save_json(iteration_result, results_dir / f"iteration_{state['iteration']:02d}.json")

    safe_result = _make_msgpack_safe(result)
    safe_iteration_result = _make_msgpack_safe(iteration_result)

    updated_state: DreamTeamState = {
        **state,
        "current_results": safe_result,
        "current_metrics": _make_msgpack_safe(metrics),
        "best_metric": _make_msgpack_safe(best_metric),
        "best_iteration": best_iteration,
        "experiment_history": state["experiment_history"] + [safe_iteration_result],
        "error_count": state["error_count"] + (0 if result['success'] else 1)
    }

    return _checkpoint_safe_state(updated_state)


# ============================================================================
# RE-USE OTHER NODES FROM V1
# ============================================================================

# check_completion_node, evolution_node, increment_iteration_node
# are the same as V1

def check_completion_node(state: DreamTeamState) -> DreamTeamState:
    """Check if experiment should continue (same as V1)"""
    goal_achieved = False

    if state.get('target_score') is not None and state.get('current_metrics'):
        metric_value = state['current_metrics'].get(state['target_metric'])

        if metric_value is not None:
            if state['minimize_metric']:
                goal_achieved = metric_value <= state['target_score']
            else:
                goal_achieved = metric_value >= state['target_score']

    should_evolve = False

    if state['error_count'] >= 2:
        should_evolve = True
        print("🧬 Evolution triggered: multiple failures\n")
    elif len(state['experiment_history']) >= 3:
        recent_metrics = [
            h['metrics'].get(state['target_metric'])
            for h in state['experiment_history'][-3:]
            if h['metrics'].get(state['target_metric']) is not None
        ]

        if len(recent_metrics) >= 2:
            if abs(max(recent_metrics) - min(recent_metrics)) < 0.001:
                should_evolve = True
                print("🧬 Evolution triggered: performance plateau\n")

    updated_state: DreamTeamState = {
        **state,
        "goal_achieved": goal_achieved,
        "should_evolve": should_evolve
    }

    return _checkpoint_safe_state(updated_state)


def evolution_node(state: DreamTeamState) -> DreamTeamState:
    """Evolve team using LLM-guided evolution"""
    print(f"\n{'='*60}")
    print("TEAM EVOLUTION")
    print(f"{'='*60}\n")

    llm = get_llm()
    team_members = [deserialize_agent(m) for m in state["team_members"]]

    if not team_members:
        return _checkpoint_safe_state({**state, "should_evolve": False, "error_count": 0})

    target_member = min(team_members, key=lambda m: m.specialization_depth)

    print(f"Evolving: {target_member.title}\n")

    evolution_prompt = f"""Evolve this agent to address current challenges.

Current:
- Title: {target_member.title}
- Expertise: {target_member.expertise}
- Depth: {target_member.specialization_depth}

Challenges:
{state.get('current_results', {}).get('error', 'Performance plateau')}

Propose evolution in this exact format:
New Title: [More specialized title]
New Expertise: [Deeper, more specific expertise areas]
New Role: [Updated role description]
"""

    evolution_output = llm.generate(evolution_prompt, temperature=0.7)

    # Parse evolution output
    new_title, new_expertise, new_role = _parse_evolution_output(
        evolution_output,
        target_member
    )

    print(f"Evolution:\n  {target_member.title} → {new_title}\n")

    target_member.evolve(
        new_title=new_title,
        new_expertise=new_expertise,
        new_role=new_role,
        trigger_reason="Performance issues / plateau"
    )

    updated_members = [serialize_agent(m) for m in team_members]

    updated_state: DreamTeamState = {
        **state,
        "team_members": updated_members,
        "should_evolve": False,
        "error_count": 0
    }

    return _checkpoint_safe_state(updated_state)


def increment_iteration_node(state: DreamTeamState) -> DreamTeamState:
    """Increment iteration (same as V1)"""
    updated_state: DreamTeamState = {**state, "iteration": state["iteration"] + 1}

    return _checkpoint_safe_state(updated_state)


def should_continue(state: DreamTeamState) -> Literal["continue", "evolve", "end"]:
    """Routing logic (same as V1)"""
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

    for line in output.split('\n'):
        if 'Columns:' in line or 'columns:' in line:
            match = re.search(r'\[(.*?)\]', line)
            if match:
                cols_str = match.group(1)
                cols = [c.strip().strip("'\"") for c in cols_str.split(',')]
                for df_name in data_context.keys():
                    if df_name in line:
                        schemas[df_name] = cols
                        break

    return schemas


def _serialize_messages(messages: List[Any]) -> List[Dict[str, str]]:
    """Convert meeting or agent messages into serializable dicts."""
    serialized = []

    for msg in messages:
        speaker = getattr(msg, "name", "Unknown")
        content = getattr(msg, "content", str(msg))

        serialized.append({
            "speaker": speaker,
            "content": content
        })

    return serialized


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

            if all(k in current_agent for k in ['title', 'expertise', 'role']):
                agent = Agent(
                    title=current_agent['title'],
                    expertise=current_agent['expertise'],
                    goal="contribute specialized expertise",
                    role=current_agent['role']
                )
                agents.append(agent)
                current_agent = {}

    if not agents:
        agents = [Agent(
            title="ML Strategist",
            expertise="machine learning, feature engineering, predictive modeling",
            goal="design effective approaches",
            role="propose modeling strategies"
        )]

    return agents


def _extract_metrics(result: Dict, target_metric: str) -> Dict[str, Any]:
    """Extract metrics from execution result"""
    metrics = {}

    if not result.get('success'):
        return metrics

    output = result.get('output', '')

    import re

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


def _summarize_data_context(data_context: Dict[str, Any]) -> Dict[str, Any]:
    """Downsample data_context to checkpoint-friendly metadata."""
    summary: Dict[str, Any] = {}

    try:
        import pandas as pd
    except Exception:
        pd = None

    for name, obj in (data_context or {}).items():
        if pd is not None and isinstance(obj, pd.DataFrame):
            summary[name] = {
                "_type": "DataFrameMeta",
                "shape": [int(obj.shape[0]), int(obj.shape[1])],
                "columns": [str(c) for c in obj.columns],
                "dtypes": {str(col): str(dtype) for col, dtype in obj.dtypes.items()},
            }
        else:
            # Apply msgpack safety to all non-DataFrame values
            summary[name] = make_msgpack_safe(obj)

    return summary


def _checkpoint_safe_state(state: DreamTeamState) -> DreamTeamState:
    """Prepare state for checkpointing without hardcoding dataset assumptions."""
    sanitized = dict(state)
    sanitized["data_context"] = _summarize_data_context(state.get("data_context", {}))
    return make_msgpack_safe(sanitized)


# Use centralized make_msgpack_safe from serialization module
# Keeping alias for backwards compatibility with any internal uses
_make_msgpack_safe = make_msgpack_safe


def _parse_evolution_output(evolution_output: str, fallback_agent: Any) -> tuple:
    """
    Parse evolution output from LLM.

    Args:
        evolution_output: LLM output with evolution proposal
        fallback_agent: Agent to use for fallback values

    Returns:
        (new_title, new_expertise, new_role)
    """
    import re

    # Try to extract fields
    title_match = re.search(r'New Title:\s*(.+)', evolution_output, re.IGNORECASE)
    expertise_match = re.search(r'New Expertise:\s*(.+)', evolution_output, re.IGNORECASE)
    role_match = re.search(r'New Role:\s*(.+)', evolution_output, re.IGNORECASE)

    # Extract or fallback
    if title_match:
        new_title = title_match.group(1).strip()
    else:
        new_title = fallback_agent.title + " (Evolved)"

    if expertise_match:
        new_expertise = expertise_match.group(1).strip()
    else:
        new_expertise = fallback_agent.expertise + ", advanced techniques"

    if role_match:
        new_role = role_match.group(1).strip()
    else:
        new_role = fallback_agent.role

    return new_title, new_expertise, new_role


# ============================================================================
# GRAPH BUILDER V2
# ============================================================================

def create_dream_team_graph_v2(checkpoint_path: Optional[Path] = None):
    """
    Create enhanced Dream Team LangGraph.

    Improvements:
    - ReAct agents
    - Smart context management
    - Multi-agent team meetings
    - Mathematical state integration
    - Better error handling
    - Persistent checkpoints with SqliteSaver

    Args:
        checkpoint_path: Path to checkpoint database (None uses MemorySaver for testing)

    Returns:
        Compiled graph
    """
    graph = StateGraph(DreamTeamState)

    # Add nodes (V2 enhanced versions)
    graph.add_node("bootstrap", bootstrap_node_v2)
    graph.add_node("team_planning", team_planning_node_v2)
    graph.add_node("code_generation", code_generation_node_v2)
    graph.add_node("execution", execution_node_v2)
    graph.add_node("check_completion", check_completion_node)
    graph.add_node("evolution", evolution_node)
    graph.add_node("increment_iteration", increment_iteration_node)

    # Entry point
    graph.set_entry_point("bootstrap")

    # Edges (same flow as V1)
    graph.add_edge("bootstrap", "team_planning")
    graph.add_edge("team_planning", "code_generation")
    graph.add_edge("code_generation", "execution")
    graph.add_edge("execution", "check_completion")

    graph.add_conditional_edges(
        "check_completion",
        should_continue,
        {
            "continue": "increment_iteration",
            "evolve": "evolution",
            "end": END
        }
    )

    graph.add_edge("evolution", "increment_iteration")
    graph.add_edge("increment_iteration", "team_planning")

    # Compile with appropriate checkpointer
    if checkpoint_path is None:
        # Use MemorySaver for testing/debugging
        checkpointer = MemorySaver()
        print("⚠️  Using MemorySaver - checkpoints will be lost on exit")
    else:
        # Use SqliteSaver for persistent checkpoints
        saver_candidate = SqliteSaver.from_conn_string(str(checkpoint_path))
        if hasattr(saver_candidate, "__enter__") and hasattr(saver_candidate, "__exit__"):
            saver_context = saver_candidate
            checkpointer = saver_context.__enter__()
            atexit.register(saver_context.__exit__, None, None, None)
        else:
            checkpointer = saver_candidate
        print(f"✅ Using SqliteSaver - checkpoints saved to {checkpoint_path}")

    return graph.compile(checkpointer=checkpointer)
