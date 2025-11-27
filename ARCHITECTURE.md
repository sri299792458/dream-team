# Dream Team Framework - Architecture Summary

## Overview

The Dream Team framework is a LangGraph-based autonomous research system where AI agents collaborate to solve ML/data science problems. The system features dynamic team composition, research paper integration, and emergent evolution based on mathematical attention mechanisms.

## Entry Points

### Main Entry Point
- **`experiments/agentds_food/run_autonomous_experiment.py`**
  - Loads data (food production shelf life prediction)
  - Creates initial state with PI and coding agent
  - Calls `run_graph_experiment(state, data_context)`
  - Returns final optimized state

### Graph Construction & Execution
- **`src/dream_team/experiment/graph_app.py`**
  - `create_experiment_graph(ctx)` - builds StateGraph
  - `run_graph_experiment(state, data_context, ...)` - executes with tracing
  - Creates ExecutionContext with non-serializable objects

## Graph Structure

```
START → bootstrap → init_math → ┌→ plan → code → execute → evaluate → check_continue ─┐
                                 │                                                      │
                                 │                        ┌─────────────────────────────┘
                                 │                        ↓
                                 │                  check_evolution
                                 │                        ↓
                                 │                   [evolve?] ────→ evolve
                                 │                        │              │
                                 └────────────────────────┴──────────────┘
                                                          │
                                                          ↓
                                                      complete → END
```

### Node Responsibilities

1. **bootstrap** - PI explores problem, recruits team
2. **init_math** - Initialize knowledge graph, team diversity metrics
3. **plan** - Team meeting to decide WHAT to implement
4. **code** - Coding agent implements HOW (generates Python code)
5. **execute** - Run code with timeout, capture output/metrics
6. **evaluate** - Extract metrics, update best scores
7. **check_continue** - Check goal/iteration limits
8. **check_evolution** - Detect if team needs to evolve
9. **evolve** - Add/remove/deepen agents based on triggers
10. **complete** - Final summary

## State Management

### Primary State Model
- **`ExperimentState`** (Pydantic BaseModel in `state.py`)
  - `iteration`: int - current iteration number
  - `phase`: Literal["init", "bootstrap", "plan", ...] - current phase
  - `team`: TeamConfig - agent configurations
  - `config`: ExperimentConfig - problem statement, target metric, etc.
  - `history`: List[IterationSummary] - full execution history
  - `current_approach/code/results/metrics`: current iteration data
  - `best_metric/best_iteration`: tracking best performance
  - `bootstrap_completed`: bool
  - `column_schemas`: Dict[str, List[str]] - DataFrame schemas
  - `mathematical_state`: Mathematical framework (K, δ, θ)
  - `evolution`: Evolution triggers and decisions
  - `goal_achieved/should_stop`: termination flags

### Execution Context
- **`ExecutionContext`** (non-serializable, in `nodes.py`)
  - `data_context`: Dict with DataFrames, artifacts_dir
  - `executor`: CodeExecutor instance
  - `research_api`: SemanticScholarAPI
  - `team_lead/team_members/coding_agent`: Agent instances
  - `evolution_engine`: EvolutionEngine instance
  - `problem_graph`: KnowledgeGraph
  - `team`: Team (mathematical framework)

**Key Pattern:** State is Pydantic-serializable, Context holds runtime objects

## Core Components

### Agents (`agent.py`)
- **Agent** - Has title, expertise, goal, role, knowledge_base, attention mechanisms
- **KnowledgeBase** - Stores papers, concepts
- **Paper** - Research paper metadata

### Meetings (`meetings.py`)
- **TeamMeeting** - Multi-agent discussion with rounds
- **IndividualMeeting** - Single agent ReAct loops
- Uses research API during ReAct for paper search

### Code Execution (`executor.py`)
- **CodeExecutor** - Sandboxed Python execution
  - Single namespace for globals/locals (fixes data context access)
  - Signal-based timeout (SIGALRM on Unix)
  - Auto-package installation
  - Output truncation (30K chars max)

### Evolution (`evolution.py`)
- **EvolutionEngine** - Decides team changes
- **Triggers**: PerformancePlateau, ErrorPattern, KnowledgeGap
- **Decisions**: ADD_AGENT, REMOVE_AGENT, DEEPEN_AGENT, REPLAN, STOP

### Mathematical Framework (`knowledge_state.py`, `team.py`)
- **KnowledgeGraph** (K) - Problem concepts
- **AttentionDistribution** (θ) - Agent focus areas
- **DepthMap** (δ) - Specialization depth
- **Team** - Computes diversity, coverage, imbalance

## Tools Integration

### Current State (Issues Identified)
❌ **No LangGraph ToolNode used** - Custom tool execution in meetings.py
❌ **No interrupt() for HIL** - Evolution decisions are automatic
❌ **No checkpointer** - State management is manual via invoke()
❌ **No built-in tool registration** - Tools are ad-hoc function calls

### Available Tools
- **Research API** - Semantic Scholar paper search
- **Code Executor** - Python code execution
- (Future: file I/O, web search, etc.)

## Tracing & Observability

### LangSmith Integration (`tracing.py`)
- `configure_langsmith(project)` - Set up env vars
- `trace_experiment(name, metadata)` - Context manager for tracing
- Metadata includes problem statement, target metric, iteration count

### Current Observability
✅ Full state transitions visible in LangSmith
✅ Node execution traces
❌ No explicit tool call traces (tools hidden in meetings)
❌ No checkpointing/resumability
❌ No time-travel debugging

## Data Flow

### Bootstrap (Iteration 0)
```
1. PI receives problem + list of DataFrames (names only)
2. PI decides exploration plan
3. Coding agent writes exploration code
4. Executor runs code, captures output + schemas
5. PI reviews results
6. PI recruits 1-3 team members
7. Mathematical framework initialized
```

### Main Iterations (1-N)
```
1. Team meeting: discuss approach (WHAT to do)
2. Coding agent: implement approach (HOW to code)
3. Execute code: capture metrics + output
4. Evaluate: extract target metric, update best
5. Check continue: goal achieved? max iterations?
6. Check evolution: performance plateau? errors?
7. (If needed) Evolve: modify team
8. Loop back to plan
```

## Issues Identified

### Design Smells
1. **God file** - nodes.py is 1439 lines, all nodes in one file
2. **Implicit data context** - DataFrames not clearly communicated to agents
3. **Custom tool execution** - Not using LangGraph ToolNode
4. **No checkpointer** - Can't resume experiments
5. **No HIL patterns** - Evolution decisions are automatic
6. **Mixed concerns** - ExecutionContext does too much (agents + tools + math)
7. **Fuzzy prompts** - Agents don't know data is pre-loaded
8. **Error swallowing** - try/except blocks with pass
9. **Duplicate routing** - route_after_check_evolution defined twice in graph_app.py (lines 63, 118)

### Correctness Issues (Fixed)
✅ Research papers now update mathematical framework (K, δ, θ)
✅ Data context is explicit in prompts
✅ Signal-based timeout enforced
✅ Pydantic validation errors fixed (column names to strings)

### Missing Features
- No session management / multi-user support
- No tool call tracing
- No human-in-the-loop for critical decisions
- No unit tests for nodes
- No integration tests for graph flows

## Next Steps (Refactor Plan)

### Phase 1: Module Reorganization
- Split nodes.py into focused files (plan_node.py, code_node.py, etc.)
- Create graph/routing.py for all routing logic
- Create graph/builder.py for graph construction only

### Phase 2: Tool Integration
- Convert research API to LangChain Tool
- Convert executor to LangChain Tool
- Use ToolNode for tool execution
- Add tool call tracing

### Phase 3: State & Checkpointing
- Add InMemorySaver checkpointer
- Make all operations resumable
- Add thread_id to config

### Phase 4: HIL Patterns
- Add interrupt() for evolution decisions
- Add interrupt() for code approval (optional)
- Implement Command(resume=...) handling

### Phase 5: Testing
- Unit tests for each node
- Integration tests for common flows
- Regression tests for fixed bugs

### Phase 6: Cleanup
- Remove duplicate code
- Fix error handling
- Add comprehensive docstrings
- Update README with new patterns
