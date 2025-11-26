# Migration Guide: LangGraph Refactor

This guide helps you transition from the old `ExperimentOrchestrator` to the new LangGraph-based experiment orchestration.

## Summary of Changes

The Dream Team framework has been refactored to use **LangGraph** for experiment orchestration while preserving all domain logic, prompts, and mathematical framework.

### What Changed

- **Orchestration**: Procedural loop → Explicit graph with nodes and edges
- **State Management**: Scattered state → Single `ExperimentState` model
- **Observability**: Limited → Full LangSmith tracing support
- **API**: More explicit and composable

### What Stayed the Same

✅ All agent prompts and behaviors
✅ Mathematical framework (θ, δ, K)
✅ Evolution engine logic
✅ Code execution and metrics
✅ Team dynamics and recruitment

## Quick Start

### Old Way (Deprecated)

```python
from dream_team.orchestrator import ExperimentOrchestrator

# Create orchestrator
orchestrator = ExperimentOrchestrator(
    team_lead=pi_agent,
    coding_agent=coder,
    # ... many parameters
)

# Run
orchestrator.run(data_context)
```

### New Way

```python
from dream_team.experiment import (
    create_initial_state,
    AgentConfig,
    run_graph_experiment
)

# 1. Create agent configs
team_lead = AgentConfig(
    title="Principal Investigator",
    expertise="machine learning, experimental design",
    goal="optimize metrics",
    role="lead research"
)

coding_agent = AgentConfig(
    title="Research Engineer",
    expertise="Python, pandas, scikit-learn",
    goal="implement plans",
    role="write code"
)

# 2. Create initial state
state = create_initial_state(
    team_lead=team_lead,
    coding_agent=coding_agent,
    problem_statement="Your problem here...",
    target_metric="mae",
    minimize_metric=True,
    max_iterations=10
)

# 3. Prepare data
data_context = {
    'train_df': train_df,
    'test_df': test_df
}

# 4. Run graph
final_state = run_graph_experiment(
    state,
    data_context,
    enable_tracing=True  # Optional: LangSmith tracing
)

# 5. Access results
print(f"Best metric: {final_state.best_metric}")
print(f"Iterations: {final_state.iteration}")
```

## API Changes

### Agent Configuration

**Before:**
```python
from dream_team.agents import Agent

agent = Agent(
    title="Data Scientist",
    expertise="statistics",
    goal="analyze data",
    role="statistical analysis"
)
```

**After:**
```python
from dream_team.experiment import AgentConfig

agent_config = AgentConfig(
    title="Data Scientist",
    expertise="statistics",
    goal="analyze data",
    role="statistical analysis"
)
```

> **Note**: `AgentConfig` is a Pydantic model that's serializable. The actual `Agent` instances are created internally by the graph.

### State Access

**Before:**
```python
# State scattered across orchestrator attributes
orchestrator.iteration
orchestrator.best_metric
orchestrator.history
```

**After:**
```python
# Centralized in ExperimentState
final_state.iteration
final_state.best_metric
final_state.history
```

### Results

**Before:**
```python
# Orchestrator modifies itself
orchestrator.run(data_context)
print(orchestrator.best_metric)
```

**After:**
```python
# Functional: returns new state
final_state = run_graph_experiment(state, data_context)
print(final_state.best_metric)
```

## New Features

### 1. LangSmith Tracing

Full observability into experiment execution:

```python
# Set environment variable
export LANGSMITH_API_KEY=your_key

# Enable tracing
final_state = run_graph_experiment(
    state,
    data_context,
    enable_tracing=True,
    langsmith_project="my-experiments"
)
```

View traces at https://smith.langchain.com/

**What you can see:**
- Each node execution (bootstrap, plan, code, execute, etc.)
- Iteration progress
- Metrics evolution
- Team composition changes
- Evolution decisions
- Timing for each step

### 2. Explicit State Model

All state is now explicit and type-safe:

```python
from dream_team.experiment import ExperimentState

# Access with autocomplete and type hints
state.iteration  # int
state.phase  # Literal["init", "bootstrap", "plan", ...]
state.team  # TeamConfig
state.history  # List[IterationSummary]
```

### 3. Graph Visualization

You can visualize the experiment flow:

```python
from dream_team.experiment.graph_app import create_experiment_graph
from dream_team.experiment.nodes import ExecutionContext

ctx = ExecutionContext(data_context, results_dir, research_api=None)
graph = create_experiment_graph(ctx)

# Visualize (requires graphviz)
graph.get_graph().draw_mermaid_png()
```

### 4. Checkpointing (Coming Soon)

LangGraph enables state checkpointing:

```python
# Save state at any point
checkpoint = final_state.dict()

# Resume later
state = ExperimentState(**checkpoint)
final_state = run_graph_experiment(state, data_context)
```

## Graph Structure

The new architecture uses an explicit graph:

```
START
  ↓
bootstrap_node (if not completed)
  ↓
init_math_framework_node
  ↓
┌──→ plan_node
│     ↓
│   code_node
│     ↓
│   execute_node
│     ↓
│   evaluate_node
│     ↓
│   check_continue_node → [STOP if goal achieved or max iterations]
│     ↓
│   check_evolution_node
│     ↓
│   [evolve_node if needed]
│     ↓
└───(next iteration)
```

Each node:
- Receives `ExperimentState`
- Performs a specific task
- Returns updated `ExperimentState`

## Migration Checklist

- [ ] Update imports from `orchestrator` to `experiment`
- [ ] Convert `Agent` instances to `AgentConfig` models
- [ ] Use `create_initial_state()` to initialize state
- [ ] Replace `orchestrator.run()` with `run_graph_experiment()`
- [ ] Update result access to use `final_state.*`
- [ ] (Optional) Add LangSmith tracing with `enable_tracing=True`
- [ ] (Optional) Set `LANGSMITH_API_KEY` environment variable

## Example: Full Migration

### Before

```python
from dream_team.orchestrator import ExperimentOrchestrator
from dream_team.agents import Agent

# Create agents
pi = Agent(
    title="PI",
    expertise="ML",
    goal="optimize",
    role="lead"
)

coder = Agent(
    title="Engineer",
    expertise="Python",
    goal="code",
    role="implement"
)

# Create orchestrator
orchestrator = ExperimentOrchestrator(
    team_lead=pi,
    coding_agent=coder,
    problem_statement="Predict target",
    target_metric="mae",
    minimize_metric=True,
    max_iterations=10,
    results_dir="results"
)

# Run
orchestrator.run(data_context)

# Get results
print(orchestrator.best_metric)
```

### After

```python
from dream_team.experiment import (
    create_initial_state,
    AgentConfig,
    run_graph_experiment
)

# Create agent configs
pi_config = AgentConfig(
    title="PI",
    expertise="ML",
    goal="optimize",
    role="lead"
)

coder_config = AgentConfig(
    title="Engineer",
    expertise="Python",
    goal="code",
    role="implement"
)

# Create initial state
state = create_initial_state(
    team_lead=pi_config,
    coding_agent=coder_config,
    problem_statement="Predict target",
    target_metric="mae",
    minimize_metric=True,
    max_iterations=10,
    results_dir="results"
)

# Run graph
final_state = run_graph_experiment(
    state,
    data_context,
    enable_tracing=True  # NEW: Optional tracing
)

# Get results
print(final_state.best_metric)
```

## Troubleshooting

### Import Errors

**Error:** `ImportError: cannot import name 'ExperimentOrchestrator'`

**Solution:** Update imports:
```python
# Old
from dream_team.orchestrator import ExperimentOrchestrator

# New
from dream_team.experiment import run_graph_experiment
```

### Agent Type Errors

**Error:** `Agent is not serializable`

**Solution:** Use `AgentConfig` instead:
```python
# Old
agent = Agent(title="...", expertise="...", ...)

# New
agent_config = AgentConfig(title="...", expertise="...", ...)
```

### Tracing Not Working

**Problem:** Tracing doesn't appear in LangSmith

**Solutions:**
1. Set environment variable: `export LANGSMITH_API_KEY=your_key`
2. Ensure `enable_tracing=True` in `run_graph_experiment()`
3. Check API key is valid at https://smith.langchain.com/
4. Install langsmith: `pip install langsmith>=0.1.0`

### State Access Errors

**Error:** `AttributeError: 'ExperimentOrchestrator' object has no attribute 'current_metrics'`

**Solution:** Access via state object:
```python
# Old
orchestrator.current_metrics

# New
final_state.current_metrics
```

## Benefits of the New Architecture

### 1. **Observability**
- See exactly what's happening at each step
- Debug issues more easily
- Track performance over time

### 2. **Type Safety**
- Pydantic models with validation
- Autocomplete in IDEs
- Catch errors early

### 3. **Composability**
- Easy to extend with new nodes
- Custom routing logic
- Reuse components

### 4. **Testability**
- Test nodes in isolation
- Mock execution context
- Integration tests with graph

### 5. **State Management**
- Single source of truth
- Serializable state
- Easy to checkpoint and resume

## Questions?

- See `LANGGRAPH_REFACTOR_GUIDE.md` for technical details
- See `REFACTOR_STATUS.md` for current status
- Check examples in `scripts/run_with_tracing.py`
- Review tests in `tests/test_nodes.py` and `tests/test_graph_integration.py`

---

**Last Updated:** 2025-11-26
**Version:** 1.0
