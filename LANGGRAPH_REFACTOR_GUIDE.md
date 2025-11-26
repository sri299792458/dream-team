# LangGraph Refactor Guide

This document tracks the refactoring of the Dream Team framework from procedural orchestration to LangGraph-based state management.

## Refactor Overview

**Goal**: Introduce a clean LangGraph-based orchestration layer with proper state management and observability, while preserving all domain logic, prompts, and mathematical framework intact.

**Key Principles**:
- ✅ Keep domain logic and math intact
- ✅ Don't change prompts, agent roles, or mathematical formulas
- ✅ Refactor is about orchestration, state, and observability
- ✅ Changes should be incremental and runnable at each phase

## Progress Tracker

### ✅ Phase 1: Map Current System and Freeze Behavior (COMPLETE)

**Completed Tasks**:
- ✅ Documented current flow in `src/dream_team/orchestrator.py`
- ✅ Created `scripts/smoke_run.py` for baseline behavior testing
- ✅ Identified key components to preserve:
  - EvolutionEngine: trigger detection, evolution proposals
  - CodeExecutor: safe execution with retry and auto-install
  - Agent roles and prompts (domain logic)
  - Team dynamics: diversity, collective knowledge, contribution tracking
  - Knowledge graph: K, θ, δ, evolution signals
  - Metrics tracking and optimization

**Files Created**:
- `scripts/smoke_run.py` - Smoke test for current behavior

**Files Modified**:
- `src/dream_team/orchestrator.py` - Added flow documentation

---

### ✅ Phase 2: Introduce Central ExperimentState Model (COMPLETE)

**Completed Tasks**:
- ✅ Created Pydantic models for all experiment state
- ✅ Defined `ExperimentState` as single source of truth
- ✅ All state is serializable for checkpointing
- ✅ Added LangGraph and dependencies to pyproject.toml

**Files Created**:
- `src/dream_team/experiment/state.py` - Core state models
  - `ExperimentState`: Main state container
  - `TeamConfig`: Team composition
  - `AgentConfig`: Individual agent config
  - `IterationSummary`: Iteration results
  - `MathematicalState`: Mathematical framework state
  - `EvolutionState`: Evolution decisions
  - `ExperimentConfig`: Experiment configuration
- `src/dream_team/experiment/__init__.py` - Package exports

**Key Models**:
```python
ExperimentState:
  - iteration: int
  - phase: Literal["init", "bootstrap", "plan", "code", "execute", "evaluate", "evolve", "complete"]
  - team: TeamConfig
  - config: ExperimentConfig
  - history: List[IterationSummary]
  - current_approach, current_code, current_results, current_metrics
  - best_metric, best_iteration
  - bootstrap_completed, column_schemas
  - mathematical_state, evolution
  - goal_achieved, should_stop
```

**Files Modified**:
- `pyproject.toml` - Added langgraph, langsmith, pydantic dependencies

---

### ✅ Phase 3: Set up LangGraph Skeleton (COMPLETE)

**Completed Tasks**:
- ✅ Created graph structure with nodes for each phase
- ✅ Defined edges and conditional routing
- ✅ Created placeholder nodes that compile and run
- ✅ Verified graph structure with test script

**Files Created**:
- `src/dream_team/experiment/graph_app.py` - Graph definition and runner
  - Nodes: bootstrap, init_math, plan, code, execute, evaluate, check_continue, check_evolution, evolve, complete
  - Routing: Conditional edges based on state
  - `create_experiment_graph()`: Graph factory
  - `run_graph_experiment()`: Main runner
- `scripts/test_graph_skeleton.py` - Graph structure test

**Graph Flow**:
```
START → bootstrap → init_math → ┌→ plan → code → execute → evaluate
                                 │   ↑                         ↓
                                 │   │                  check_continue
                                 │   │                         ↓
                                 │   │                  check_evolution
                                 │   │                   ↙         ↘
                                 │   └─────────────────(evolve)  (continue)
                                 │                       ↓
                                 └───────────────────────┘
                                                         ↓
                                                     complete → END
```

---

### 🚧 Phase 4: Move Internal Logic into Graph Nodes (IN PROGRESS)

**Current Status**: Bootstrap node migration started

**Completed**:
- ✅ Created `ExecutionContext` class for non-serializable objects
- ✅ Started migrating `bootstrap_node` logic from `_bootstrap_exploration`
- ⚠️  Partial implementation in `src/dream_team/experiment/nodes.py`

**Remaining Work**:

#### 4.1. Complete Bootstrap Node Migration
- [x] PI exploration planning
- [x] Coding agent implementation
- [x] Code execution
- [x] Column schema extraction
- [x] Team recruitment
- [ ] Test bootstrap node in isolation

#### 4.2. Migrate Init Math Framework Node
- [ ] Extract `_initialize_mathematical_framework` logic
- [ ] Create problem graph from problem statement
- [ ] Initialize Team object for dynamics
- [ ] Initialize agent θ, δ, K

**Source**: `orchestrator.py:1408-1424`

#### 4.3. Migrate Plan Node
- [ ] Extract `_team_planning_meeting` logic
- [ ] Run TeamMeeting with all agents
- [ ] Build context (history, previous output, column schemas)
- [ ] Synthesize action plan

**Source**: `orchestrator.py:500-646`

#### 4.4. Migrate Code Node
- [ ] Extract `_implement_approach` logic
- [ ] Coding agent translates plan to Python
- [ ] Include previous output context
- [ ] Save generated code

**Source**: `orchestrator.py:697-801`

#### 4.5. Migrate Execute Node
- [ ] Extract `_execute_with_retry` logic
- [ ] Execute code via executor
- [ ] Auto-retry on errors (max 2 retries)
- [ ] Auto-install missing packages
- [ ] Ask agent to fix errors

**Source**: `orchestrator.py:818-881`, `883-974`

#### 4.6. Migrate Evaluate Node
- [ ] Extract `_extract_metrics` logic
- [ ] Parse metrics from execution results
- [ ] Update best_metric tracking
- [ ] Save iteration summary

**Source**: `orchestrator.py:986-1018`

#### 4.7. Migrate Check Evolution Node
- [ ] Extract `_check_mathematical_evolution` logic
- [ ] Update agent dynamics (θ, δ)
- [ ] Compute team state
- [ ] Check evolution signals
- [ ] Set evolution.triggered flag

**Source**: `orchestrator.py:1461-1510`, `1511-1544`

#### 4.8. Update Graph App to Use Real Nodes
- [ ] Wire ExecutionContext into graph runner
- [ ] Replace placeholder nodes with real implementations
- [ ] Ensure state flows correctly through nodes
- [ ] Test end-to-end with real data

**Files to Create/Modify**:
- `src/dream_team/experiment/nodes.py` (in progress)
- `src/dream_team/experiment/graph_app.py` (update to use real nodes)

---

### ⏳ Phase 5: Integrate Evolution Engine as a Node (PENDING)

**Goal**: Move all evolution/adaptation logic into evolve_node

**Tasks**:
- [ ] Extract `_evolve_team` logic to evolve_node
- [ ] Use existing EvolutionEngine and triggers
- [ ] Research papers via research API
- [ ] PI analyzes team composition
- [ ] Execute evolution plan (NO_CHANGE, ADD, REMOVE, DEEPEN)
- [ ] Update team configuration in state
- [ ] Add conditional routing based on evolution.decision

**Source**: `orchestrator.py:1069-1263`

**Evolution Decisions**:
- NO_CHANGE: Continue with current team → plan_node
- ADD_AGENT: Add new specialist → plan_node
- REMOVE_AGENT: Remove agent → plan_node
- DEEPEN_AGENT: Specialize existing agent → plan_node
- STOP: Evolution determined experiment should stop → complete_node

**Files to Modify**:
- `src/dream_team/experiment/nodes.py` - Add evolve_node implementation
- `src/dream_team/experiment/graph_app.py` - Update routing

---

### ⏳ Phase 6: Integrate LangSmith Tracing (PENDING)

**Goal**: Make entire experiment observable at phase/node level

**Tasks**:
- [ ] Configure LangSmith client with environment variables
- [ ] Wrap graph execution in LangSmith tracing
- [ ] Add custom events/metadata for each node:
  - iteration number
  - phase name
  - key metrics (accuracy, loss, etc.)
  - evolution decisions
  - team composition changes
- [ ] Log state before/after each node
- [ ] Verify traces in LangSmith UI

**Environment Variables**:
```bash
export LANGSMITH_API_KEY=your_key_here
export LANGSMITH_PROJECT=dream-team-experiments
```

**Files to Modify**:
- `src/dream_team/experiment/graph_app.py` - Add tracing
- `scripts/run_with_tracing.py` - Example script with tracing enabled

**Tracing Points**:
- Node entry/exit
- Team meeting start/end
- Code execution start/end
- Evolution decisions
- Metric updates

---

### ⏳ Phase 7: Cleanup, Tests, and Documentation (PENDING)

**Goal**: Remove dead code, add tests, document new architecture

**Tasks**:

#### 7.1. Cleanup
- [ ] Identify unused methods in ExperimentOrchestrator
- [ ] Mark old orchestrator as deprecated
- [ ] Move to `legacy/` module with deprecation warnings
- [ ] Remove redundant utility functions

#### 7.2. Testing
- [ ] Unit tests for each node function
  - Test with minimal ExperimentState
  - Verify state fields updated correctly
  - Check no unexpected mutations
- [ ] Integration test: run full graph for 2-3 iterations
  - Assert metrics populated
  - Assert history recorded
  - Assert evolution routing works
- [ ] Test state serialization/deserialization
- [ ] Test resume functionality

**Test Files to Create**:
- `tests/test_state.py` - ExperimentState model tests
- `tests/test_nodes.py` - Individual node tests
- `tests/test_graph.py` - Full graph integration tests
- `tests/test_serialization.py` - State persistence tests

#### 7.3. Documentation
- [ ] Update README with LangGraph architecture
- [ ] Document ExperimentState model and fields
- [ ] Document graph nodes and responsibilities
- [ ] Document evolution decision routing
- [ ] Add inline docstrings to all nodes
- [ ] Create migration guide for existing users

**Documentation Files**:
- `README.md` - Update with new architecture
- `docs/ARCHITECTURE.md` - Detailed architecture documentation
- `docs/MIGRATION.md` - Migration guide from old to new API
- `docs/STATE_REFERENCE.md` - ExperimentState field reference

---

## Development Workflow

### Running Tests

```bash
# Test graph skeleton (placeholder nodes)
python scripts/test_graph_skeleton.py

# Smoke test with real data (after Phase 4)
python scripts/smoke_run.py

# Full test suite (after Phase 7)
pytest tests/
```

### Debugging State Flow

```bash
# Print state at each node
LANGGRAPH_DEBUG=1 python scripts/test_graph_skeleton.py

# Save state snapshots
LANGGRAPH_SAVE_SNAPSHOTS=1 python scripts/test_graph_skeleton.py
```

### Comparing Before/After

```bash
# Run old orchestrator
python experiments/agentds_food/run_autonomous_experiment.py

# Run new graph (after Phase 4)
python scripts/run_graph_experiment.py

# Compare outputs
diff -r results_old/ results_new/
```

---

## File Structure

```
dream-team/
├── src/dream_team/
│   ├── orchestrator.py          # [LEGACY] Original orchestration
│   ├── agent.py                 # [PRESERVE] Agent class, no changes
│   ├── team.py                  # [PRESERVE] Team dynamics, no changes
│   ├── evolution.py             # [PRESERVE] Evolution engine, no changes
│   ├── executor.py              # [PRESERVE] Code execution, no changes
│   ├── knowledge_state.py       # [PRESERVE] Mathematical framework, no changes
│   ├── meetings.py              # [PRESERVE] Meeting coordination, no changes
│   └── experiment/              # [NEW] LangGraph orchestration
│       ├── __init__.py
│       ├── state.py             # ✅ State models
│       ├── graph_app.py         # ✅ Graph definition
│       └── nodes.py             # 🚧 Node implementations
├── scripts/
│   ├── smoke_run.py             # ✅ Baseline test
│   └── test_graph_skeleton.py  # ✅ Graph structure test
├── tests/                        # ⏳ Test suite
├── docs/                         # ⏳ Documentation
└── LANGGRAPH_REFACTOR_GUIDE.md  # This file
```

---

## Key Design Decisions

### 1. State Management
- **Decision**: Single `ExperimentState` Pydantic model
- **Rationale**: Explicit, serializable, type-safe state
- **Trade-off**: More boilerplate, but much clearer what's happening

### 2. ExecutionContext
- **Decision**: Separate context for non-serializable objects
- **Rationale**: State must be serializable for checkpointing
- **Trade-off**: Extra layer, but enables proper state management

### 3. Agent Instances
- **Decision**: Create Agent instances from AgentConfig in ExecutionContext
- **Rationale**: Agents have non-serializable state (knowledge graphs, dynamics)
- **Trade-off**: Need to sync between instances and configs

### 4. Node Granularity
- **Decision**: One node per major phase (bootstrap, plan, code, execute, evaluate, evolve)
- **Rationale**: Matches conceptual phases, good for observability
- **Trade-off**: Could be finer-grained, but this feels right

### 5. Backwards Compatibility
- **Decision**: Keep old ExperimentOrchestrator as deprecated legacy code
- **Rationale**: Don't break existing experiments immediately
- **Trade-off**: More code to maintain during transition

---

## Common Pitfalls to Avoid

### ❌ Don't Store Non-Serializable Objects in State
```python
# BAD
state.executor = CodeExecutor()  # Not serializable!

# GOOD
ctx.executor = CodeExecutor()  # In ExecutionContext
```

### ❌ Don't Change Prompts or Domain Logic
```python
# BAD - changing prompts
agent_prompt = "You are an AI assistant..."  # Too generic!

# GOOD - preserve existing prompts
agent_prompt = agent.prompt  # Uses existing prompt logic
```

### ❌ Don't Forget to Update State from Agents
```python
# After modifying agents in context
ctx.update_state_from_agents(state)  # Sync back to state
```

### ❌ Don't Skip State Updates
```python
# BAD
def my_node(state):
    do_work()
    return state  # Forgot to update state.phase!

# GOOD
def my_node(state):
    do_work()
    state.phase = "next_phase"
    state.current_results = results
    return state
```

---

## Testing Checklist

Before merging each phase:

- [ ] Graph compiles without errors
- [ ] State serializes/deserializes correctly
- [ ] Node updates appropriate state fields
- [ ] No changes to prompts or math formulas
- [ ] Existing smoke test still passes
- [ ] New functionality has unit tests
- [ ] Documentation updated

---

## Questions & Decisions Log

### Q: Should we preserve exact prompt text?
**A**: Yes. Prompts are domain logic. Only refactor orchestration.

### Q: What about mathematical framework (θ, δ, K)?
**A**: Preserve completely. Maybe serialize to MathematicalState for checkpointing.

### Q: How to handle data_context (large DataFrames)?
**A**: Store metadata in state, actual data in ExecutionContext. Don't serialize large data.

### Q: Resume functionality?
**A**: After Phase 4. Load state from JSON, recreate ExecutionContext, continue graph.

### Q: Multiple iterations in one graph run?
**A**: Yes. Graph loops plan → code → execute → evaluate → check → plan until done.

---

## Next Steps

**Immediate**: Complete Phase 4 node migrations

**Priority Order**:
1. Complete bootstrap_node implementation and test
2. Migrate init_math_framework_node
3. Migrate plan_node (most critical for flow)
4. Migrate code_node
5. Migrate execute_node
6. Migrate evaluate_node
7. Test full iteration cycle
8. Migrate check_evolution_node
9. Complete Phase 5 (evolve_node)
10. Add LangSmith tracing (Phase 6)
11. Tests and docs (Phase 7)

**Estimated Remaining Effort**:
- Phase 4: 6-8 hours
- Phase 5: 2-3 hours
- Phase 6: 1-2 hours
- Phase 7: 4-6 hours
- **Total**: 13-19 hours

---

## Success Criteria

The refactor is complete when:

1. ✅ All phases 1-7 are complete
2. ✅ Smoke test passes with identical results to old orchestrator
3. ✅ Full test suite passes
4. ✅ LangSmith traces show clear execution flow
5. ✅ Documentation is complete
6. ✅ Old orchestrator is deprecated (not removed)
7. ✅ No prompts or math formulas were changed

---

## Contact & Questions

For questions about this refactor:
- Check this guide first
- Review the existing orchestrator.py for reference
- Test incrementally with smoke_run.py
- Document any new decisions in this file

---

*Last Updated: 2025-11-26*
*Current Phase: 4 (In Progress)*
