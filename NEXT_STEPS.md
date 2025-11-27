# Dream Team Refactor - Next Steps & Implementation Guide

## ✅ Completed

### Documentation & Analysis
- [x] **ARCHITECTURE.md** - Complete system overview
- [x] **REFACTOR_ISSUES.md** - Detailed issue tracking with priorities
- [x] Created clean `langgraph-clean` branch
- [x] Removed deprecated code (~3,000 lines)
- [x] Fixed critical data context bug (agents hallucinating data)
- [x] Fixed timeout enforcement
- [x] Fixed Pydantic validation errors
- [x] Fixed duplicate routing function

## 🎯 Implementation Roadmap

### Phase 1: Module Reorganization (4 hours)

**Goal:** Break 1439-line `nodes.py` into focused modules

#### Step 1.1: Create Context Module (30 min)
```bash
# Create src/dream_team/experiment/context.py
# Move ExecutionContext class from nodes.py
# Update imports
```

**Files:**
- `context.py` - ExecutionContext class (150 lines)

**Changes:**
```python
# src/dream_team/experiment/context.py
from typing import Dict, Any, Optional, List
from pathlib import Path

from ..agent import Agent
from ..executor import CodeExecutor
from ..evolution import EvolutionEngine
from ..knowledge_state import KnowledgeGraph
from ..team import Team
from ..research import get_research_assistant

class ExecutionContext:
    """Non-serializable runtime context"""
    # ... move entire class here
```

####Step 1.2: Create Routing Module (30 min)
```bash
# Create src/dream_team/experiment/routing.py
# Move all routing functions from graph_app.py
```

**Files:**
- `routing.py` - All routing logic (50 lines)

**Functions to move:**
- `route_after_check_evolution()`
- `route_after_check_continue()`
- `route_after_bootstrap()`

#### Step 1.3: Split Nodes into Separate Files (2.5 hours)

**Directory structure:**
```
src/dream_team/experiment/graph/
├── __init__.py          # Public exports
├── nodes/
│   ├── __init__.py     # Export all node factories
│   ├── bootstrap.py    # create_bootstrap_node (300 lines)
│   ├── init_math.py    # create_init_math_framework_node (120 lines)
│   ├── plan.py         # create_plan_node (80 lines)
│   ├── code.py         # create_code_node (120 lines)
│   ├── execute.py      # create_execute_node (250 lines)
│   ├── evaluate.py     # create_evaluate_node (100 lines)
│   ├── evolution.py    # create_check_evolution_node, create_evolve_node (200 lines)
│   └── utils.py        # Shared utilities (_refresh_column_schemas, etc.)
└── builder.py          # create_experiment_graph, run_graph_experiment
```

**Template for each node file:**
```python
"""
Bootstrap node implementation.

Responsibilities:
- PI explores problem alone
- Extracts DataFrame schemas
- Recruits team members
"""

from typing import Any
from pathlib import Path

from ..state import ExperimentState
from ..context import ExecutionContext
from ...agent import Agent
from ...meetings import IndividualMeeting
from ...executor import extract_code_from_text

def create_bootstrap_node(ctx: ExecutionContext):
    """Factory function for bootstrap node"""
    def bootstrap_node(state: ExperimentState) -> ExperimentState:
        """Bootstrap phase: PI explores and recruits team"""
        # ... implementation
        return state

    return bootstrap_node
```

#### Step 1.4: Update Imports (30 min)

**Files to update:**
- `graph_app.py` → import from new modules
- `__init__.py` → export from new structure
- All test files

**Verification:**
```bash
python -m pytest tests/
python experiments/agentds_food/run_autonomous_experiment.py --dry-run
```

---

### Phase 2: LangGraph Checkpointer (2 hours)

**Goal:** Add state persistence and resumability

#### Step 2.1: Add Checkpointer Support (1 hour)

**Changes to `graph_app.py`:**
```python
from langgraph.checkpoint.memory import InMemorySaver

def create_experiment_graph(ctx: ExecutionContext, checkpointer=None) -> StateGraph:
    """Create graph with optional checkpointer"""
    graph = StateGraph(ExperimentState)
    # ... add nodes and edges

    # Compile with checkpointer
    if checkpointer is None:
        checkpointer = InMemorySaver()

    return graph.compile(checkpointer=checkpointer)

def run_graph_experiment(
    state: ExperimentState,
    data_context: Dict[str, Any],
    thread_id: str = "default",
    checkpointer = None,
    ...
) -> ExperimentState:
    """Run with checkpointing"""
    ctx = ExecutionContext(data_context, ...)
    graph = create_experiment_graph(ctx, checkpointer)

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    final_state = graph.invoke(state, config=config)
    return final_state
```

#### Step 2.2: Add Resume Support (30 min)

**Add to `state.py`:**
```python
class ExperimentConfig(BaseModel):
    """Experiment configuration"""
    problem_statement: str
    target_metric: str
    # ... existing fields

    # Checkpointing
    thread_id: str = "default"
    checkpoint_interval: int = 1  # Save every N iterations
```

**Update entrypoint:**
```python
# experiments/agentds_food/run_autonomous_experiment.py

# Resume from checkpoint
thread_id = f"shelf_life_{int(time.time())}"

final_state = run_graph_experiment(
    state,
    data_context,
    thread_id=thread_id,
    enable_tracing=True
)

# Later, to resume:
# final_state = run_graph_experiment(
#     state,
#     data_context,
#     thread_id="shelf_life_1234567890",  # Same thread ID
#     enable_tracing=True
# )
```

#### Step 2.3: Test Checkpointing (30 min)

**Create test:**
```python
# tests/test_checkpointing.py

def test_checkpoint_resume():
    """Test that experiments can be resumed from checkpoint"""
    checkpointer = InMemorySaver()
    thread_id = "test_thread"

    # Run 2 iterations
    config1 = ExperimentConfig(..., max_iterations=2)
    state1 = create_initial_state(config1, ...)
    final1 = run_graph_experiment(
        state1,
        data_context,
        thread_id=thread_id,
        checkpointer=checkpointer
    )

    assert final1.iteration == 2

    # Resume and run 3 more
    config2 = ExperimentConfig(..., max_iterations=5)
    state2 = create_initial_state(config2, ...)
    final2 = run_graph_experiment(
        state2,
        data_context,
        thread_id=thread_id,  # Same thread!
        checkpointer=checkpointer
    )

    assert final2.iteration == 5
    assert final2.history[0] == final1.history[0]  # Same bootstrap
```

---

### Phase 3: ToolNode Migration (6 hours)

**Goal:** Replace custom tool execution with LangGraph primitives

#### Step 3.1: Convert Research API to LangChain Tool (2 hours)

**Create `tools/research.py`:**
```python
from langchain_core.tools import tool
from typing import List, Dict, Any

@tool
def search_papers(query: str, num_papers: int = 5) -> List[Dict[str, Any]]:
    """Search for research papers using Semantic Scholar.

    Args:
        query: Search query (e.g., "shelf life prediction food storage")
        num_papers: Number of papers to return (default: 5)

    Returns:
        List of papers with title, year, abstract, url, citation_count
    """
    from dream_team.research import get_research_assistant

    research = get_research_assistant()
    results = research.ss_api.search_papers(query, limit=num_papers)

    return [
        {
            "title": r.title,
            "year": r.year,
            "abstract": r.abstract,
            "url": r.url,
            "citation_count": r.citation_count,
            "tldr": r.tldr
        }
        for r in results
    ]
```

#### Step 3.2: Convert Executor to LangChain Tool (2 hours)

**Create `tools/executor.py`:**
```python
from langchain_core.tools import tool
from typing import Dict, Any

@tool
def execute_python(code: str, description: str = "") -> Dict[str, Any]:
    """Execute Python code in a sandboxed environment.

    Args:
        code: Python code to execute
        description: Description of what the code does

    Returns:
        Dict with success, output, error, variables, metrics
    """
    from dream_team.executor import CodeExecutor

    # Get executor from context (via dependency injection)
    executor = get_current_executor()
    result = executor.execute(code, description)

    return result
```

#### Step 3.3: Replace IndividualMeeting with ReAct Pattern (2 hours)

**Old (custom):**
```python
# meetings.py - IndividualMeeting with custom ReAct loop
meeting = IndividualMeeting(...)
output = meeting.run(agent, task, num_iterations=3, use_react=True)
```

**New (LangGraph standard):**
```python
# graph/nodes/agent_react.py
from langgraph.prebuilt import ToolNode, create_react_agent
from langchain_core.messages import HumanMessage

def create_agent_react_node(ctx: ExecutionContext, tools: List):
    """Create a ReAct agent node with tools"""

    def agent_react_node(state: ExperimentState) -> ExperimentState:
        # Create agent with bound tools
        agent = create_react_agent(
            ctx.llm,  # Gemini LLM
            tools=tools,
            state_modifier="You are a {agent_title}. {expertise}"
        )

        # Run agent
        result = agent.invoke({
            "messages": [HumanMessage(content=state.current_task)]
        })

        state.agent_output = result["messages"][-1].content
        return state

    return agent_react_node
```

**Routing with ToolNode:**
```python
from langgraph.prebuilt import ToolNode

# In graph builder
tools = [search_papers, execute_python]
tool_node = ToolNode(tools)

graph.add_node("agent", create_agent_node(ctx))
graph.add_node("tools", tool_node)

def should_continue(state):
    """Route based on tool calls"""
    last_message = state.agent_output
    if last_message.tool_calls:
        return "tools"
    return "complete"

graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "complete": "evaluate"
    }
)

graph.add_edge("tools", "agent")  # Loop back
```

---

### Phase 4: Human-in-the-Loop (3 hours)

**Goal:** Add interrupt() for critical decisions

#### Step 4.1: Add Evolution Approval (1.5 hours)

**Update `evolution.py` node:**
```python
from langgraph.types import interrupt

def create_check_evolution_node(ctx: ExecutionContext):
    """Check evolution with HIL approval"""

    def check_evolution_node(state: ExperimentState) -> ExperimentState:
        # ... check triggers ...

        if state.evolution.triggered:
            # Interrupt for human approval
            decision = interrupt({
                "type": "evolution_decision",
                "reason": state.evolution.reason,
                "trigger_names": state.evolution.trigger_names,
                "recommended_action": state.evolution.decision,
                "current_team": [a.title for a in ctx.all_agents],
                "prompt": f"Evolution triggered: {state.evolution.reason}\nApprove {state.evolution.decision}?"
            })

            # decision comes from Command(resume=...)
            if decision.get("approved", False):
                # Proceed with evolution
                state.phase = "evolve"
            else:
                # Skip evolution
                state.evolution.triggered = False
                state.phase = "plan"

        return state

    return check_evolution_node
```

**Resume with approval:**
```python
from langgraph.types import Command

# When interrupted, resume with:
graph.invoke(
    None,  # No new state
    config=config,
    command=Command(resume={"approved": True})
)
```

#### Step 4.2: Add Code Review (Optional, 1.5 hours)

**Create HIL node for code review:**
```python
def create_code_review_node(ctx: ExecutionContext):
    """Optional human review of generated code"""

    def code_review_node(state: ExperimentState) -> ExperimentState:
        if state.config.require_code_approval:
            feedback = interrupt({
                "type": "code_review",
                "code": state.current_code,
                "approach": state.current_approach,
                "iteration": state.iteration,
                "prompt": "Review this code before execution. Approve or request changes."
            })

            if not feedback.get("approved", True):
                # Send back to coding with feedback
                state.review_feedback = feedback.get("comments", "")
                state.phase = "code"
                return state

        state.phase = "execute"
        return state

    return code_review_node
```

---

### Phase 5: Testing (6 hours)

#### Step 5.1: Unit Tests for Nodes (3 hours)

**Template:**
```python
# tests/graph/nodes/test_bootstrap.py

import pytest
from unittest.mock import Mock, patch
from dream_team.experiment.graph.nodes.bootstrap import create_bootstrap_node
from dream_team.experiment.state import ExperimentState, ExperimentConfig, TeamConfig
from dream_team.experiment.context import ExecutionContext

def test_bootstrap_node_explores_problem():
    """Test that bootstrap node runs exploration"""
    # Setup
    ctx = Mock(spec=ExecutionContext)
    ctx.team_lead = Mock()
    ctx.coding_agent = Mock()
    ctx.executor = Mock()

    state = ExperimentState(
        config=ExperimentConfig(...),
        team=TeamConfig(...),
        bootstrap_completed=False
    )

    # Run node
    bootstrap_node = create_bootstrap_node(ctx)
    result = bootstrap_node(state)

    # Assert
    assert result.bootstrap_completed == True
    assert len(result.column_schemas) > 0
    assert len(result.team.team_members) > 0
    ctx.executor.execute.assert_called_once()

def test_bootstrap_node_skips_if_completed():
    """Test that bootstrap is skipped if already done"""
    # ... similar pattern
```

#### Step 5.2: Integration Tests (2 hours)

**Template:**
```python
# tests/test_graph_integration.py

def test_full_experiment_flow(mock_llm, mock_data):
    """Test complete experiment flow end-to-end"""
    state = create_initial_state(...)

    with patch('dream_team.llm.get_llm', return_value=mock_llm):
        final_state = run_graph_experiment(
            state,
            mock_data,
            max_iterations=2
        )

    # Assert flow
    assert final_state.iteration == 2
    assert final_state.bootstrap_completed
    assert len(final_state.history) == 2
    assert final_state.best_metric is not None
```

#### Step 5.3: Regression Tests (1 hour)

**For each fixed bug, add test:**
```python
def test_data_context_explicit_in_prompts():
    """Regression: Agents must know data is pre-loaded"""
    # ... test that prompts contain "ALREADY LOADED"

def test_timeout_enforced():
    """Regression: Code execution must timeout"""
    # ... test that infinite loop code times out

def test_column_names_converted_to_strings():
    """Regression: Integer column names must be converted"""
    # ... test schema refresh with numeric columns
```

---

### Phase 6: Cleanup & Documentation (2 hours)

#### Step 6.1: Remove Dead Code (30 min)

**Files to check:**
- [ ] Remove unused imports
- [ ] Remove commented blocks
- [ ] Remove old phase name references
- [ ] Consolidate duplicate schema extraction logic

#### Step 6.2: Error Handling (1 hour)

**Pattern to apply everywhere:**
```python
import logging

logger = logging.getLogger(__name__)

try:
    # ... operation
except SpecificError as e:
    logger.error(f"Failed to X: {e}", exc_info=True)
    # Re-raise or handle gracefully
    raise
```

**Replace all `except: pass` with proper logging**

#### Step 6.3: Update Documentation (30 min)

**Update README.md:**
```markdown
## Quick Start

### With Checkpointing
```python
from dream_team import create_initial_state, run_graph_experiment

state = create_initial_state(config, team, results_dir)

final_state = run_graph_experiment(
    state,
    data_context,
    thread_id="my_experiment_1",  # For resuming
    enable_tracing=True
)
```

### Resume Experiment
```python
# Same thread_id resumes from checkpoint
final_state = run_graph_experiment(
    state,
    data_context,
    thread_id="my_experiment_1",  # Same ID!
    enable_tracing=True
)
```

### With Human-in-the-Loop
```python
# Evolution requires approval
result = run_graph_experiment(
    state,
    data_context,
    require_evolution_approval=True
)

# When interrupted, approve/reject via Command
from langgraph.types import Command

graph.invoke(
    None,
    config=config,
    command=Command(resume={"approved": True})
)
```
```

---

## Quick Wins (Can Do Now)

### 1. Fix Error Swallowing (30 min)
```bash
# Find all except: pass
grep -rn "except.*pass" src/dream_team/

# Replace with proper logging
except Exception as e:
    logger.error(f"Error in {context}: {e}", exc_info=True)
    raise
```

### 2. Add Type Hints (1 hour)
```python
# Before
def some_function(x, y):
    return x + y

# After
def some_function(x: float, y: float) -> float:
    """Add two numbers"""
    return x + y
```

### 3. Add Docstrings (1 hour)
```python
def create_bootstrap_node(ctx: ExecutionContext):
    """
    Create the bootstrap node for experiment initialization.

    The bootstrap node:
    1. Has PI explore the problem alone
    2. Extracts DataFrame schemas from data
    3. PI recruits 1-3 team members based on findings

    Args:
        ctx: Execution context with executor, research API, etc.

    Returns:
        Node function that accepts and returns ExperimentState

    Example:
        >>> ctx = ExecutionContext(data_context, results_dir)
        >>> bootstrap = create_bootstrap_node(ctx)
        >>> state = bootstrap(initial_state)
        >>> assert state.bootstrap_completed == True
    """
    # ... implementation
```

---

## Validation Checklist

After each phase, verify:

- [ ] All tests pass: `pytest tests/`
- [ ] Code compiles: `python -m py_compile src/**/*.py`
- [ ] Type hints valid: `mypy src/dream_team/`
- [ ] Example runs: `python experiments/agentds_food/run_autonomous_experiment.py`
- [ ] No regressions in behavior
- [ ] LangSmith traces look correct
- [ ] Documentation updated

---

## Progress Tracking

Use this table to track progress:

| Phase | Task | Status | Hours | Notes |
|-------|------|--------|-------|-------|
| P0 | Fix duplicate routing | ✅ Done | 0.1 | Committed |
| P0 | Fix error swallowing | 🔲 Todo | 0.5 | |
| P1 | Add checkpointer | 🔲 Todo | 2.0 | |
| P1 | ToolNode migration | 🔲 Todo | 6.0 | |
| P1 | HIL patterns | 🔲 Todo | 3.0 | |
| P2 | Split nodes.py | 🔲 Todo | 4.0 | |
| P2 | Refactor ExecutionContext | 🔲 Todo | 2.0 | |
| P3 | Unit tests | 🔲 Todo | 6.0 | |
| P3 | Integration tests | 🔲 Todo | 4.0 | |
| P4 | Cleanup | 🔲 Todo | 2.0 | |

**Total Completed:** 0.1 hours
**Total Remaining:** ~29.5 hours
