# Dream Team Refactor - Issues & Action Plan

## Bugs / Correctness Issues

### ✅ FIXED
- [x] **Data context hallucination** - Agents creating fake data instead of using pre-loaded DataFrames
  - Fixed by making prompts explicit: "ALREADY LOADED in memory"
- [x] **Research papers don't update math framework** - Papers added to KB but K, δ, θ not updated
  - Fixed by using `add_paper_to_knowledge()` instead of `knowledge_base.add_paper()`
- [x] **Timeout not enforced** - Infinite loops freeze experiment
  - Fixed with signal.SIGALRM timeout handler
- [x] **Pydantic validation errors** - Integer column names fail string validation
  - Fixed by converting column names to strings

### 🔴 OPEN
- [ ] **Duplicate routing function** - `route_after_check_evolution` defined twice in graph_app.py (lines 63, 118)
- [ ] **Error swallowing** - Multiple `except: pass` blocks hide failures
- [ ] **Fuzzy metric extraction** - Substring matching can grab wrong variables
  - Partially fixed with explicit prompts, but still uses heuristics
- [ ] **No validation of agent responses** - Assumes LLM always returns well-formed outputs
- [ ] **Bootstrap can fail silently** - If recruitment parsing fails, team_members stay empty

## Design Smells / Clutter

### God Files
- [ ] **nodes.py (1439 lines)** - All 8 node implementations in one file
  - Should be split into separate files per node
- [ ] **executor.py (200+ lines)** - Execution logic mixed with parsing
  - Should separate execution, timeout handling, package installation

### Anti-Patterns
- [ ] **ExecutionContext does too much** - Manages agents, executor, research API, evolution, math framework
  - Should separate concerns: AgentManager, ToolRegistry, MathFramework
- [ ] **Manual state machines** - Node functions have nested if/else for phase transitions
  - Should use LangGraph conditional edges
- [ ] **Ad-hoc tool calling** - Tools called directly in node functions
  - Should use ToolNode and bind_tools()
- [ ] **No separation of routing logic** - Routing mixed with graph_app.py
  - Should have dedicated routing.py
- [ ] **Nested closures** - All nodes are factory functions returning closures
  - Makes testing harder, should use classes or partial()

### Duplicated Logic
- [ ] **Schema extraction** - Done in 2 places (bootstrap, _refresh_column_schemas)
- [ ] **Agent creation** - Logic split between ExecutionContext and create_initial_state
- [ ] **Prompt building** - Similar patterns repeated across nodes (data_info, schema_info)
- [ ] **Meeting save logic** - Repeated in multiple nodes

### Dead Code
- [ ] **Unused imports** - Several modules import but don't use all imports
- [ ] **Commented code** - Some commented blocks remain in nodes.py
- [ ] **Old phase names** - References to "init_math" vs "math_init"

## Missing Tests / Guardrails

### No Unit Tests
- [ ] No tests for individual nodes
- [ ] No tests for routing functions
- [ ] No tests for ExecutionContext
- [ ] No tests for state transitions
- [ ] No tests for error handling paths

### No Integration Tests
- [ ] No end-to-end graph execution tests
- [ ] No tests for bootstrap→plan→code→execute flow
- [ ] No tests for evolution triggers
- [ ] No tests with mocked LLM responses

### Missing Validation
- [ ] No schema validation for DataFrame column names
- [ ] No validation that required DataFrames exist
- [ ] No validation of extracted metrics
- [ ] No validation of agent recruitment output
- [ ] No validation of code syntax before execution

## LangGraph Anti-Patterns

### Not Using Built-in Features
- [ ] **No checkpointer** - Can't resume experiments, no replay
- [ ] **No interrupt()** - No human-in-the-loop for decisions
- [ ] **No ToolNode** - Custom tool execution instead of LangGraph primitive
- [ ] **No bind_tools()** - Tools not registered with LLM
- [ ] **No tool call tracing** - Can't see tool invocations in LangSmith
- [ ] **No thread_id** - No session management

### Custom Implementations of Built-in Features
- [ ] **Manual state management** - invoke() with no checkpointer
- [ ] **Custom tool executor** - IndividualMeeting instead of ToolNode
- [ ] **Ad-hoc session persistence** - Saving JSON summaries instead of using checkpointer
- [ ] **Nested conditionals** - Should use conditional_edges more

### Missing Best Practices
- [ ] **No subgraphs** - Complex nodes should be subgraphs
- [ ] **No CommandGraph** - For multi-agent collaboration patterns
- [ ] **No streaming** - Could stream outputs for long-running nodes
- [ ] **No parallelization** - Some nodes could run in parallel

## Action Plan (Prioritized)

### P0: Critical Bugs
1. Fix duplicate routing function (5 min)
2. Fix error swallowing - add proper logging (30 min)
3. Add validation for bootstrap recruitment (1 hour)

### P1: LangGraph Migration
4. Add checkpointer support (2 hours)
   - InMemorySaver for dev
   - Add thread_id to config
   - Update run_graph_experiment
5. Migrate to ToolNode (4 hours)
   - Convert research API to LangChain Tool
   - Convert executor to LangChain Tool
   - Replace IndividualMeeting with standard ReAct pattern
6. Add interrupt() for evolution (2 hours)
   - HIL node for evolution approval
   - Command(resume=...) handling

### P2: Code Organization
7. Split nodes.py into separate files (3 hours)
   - graph/nodes/bootstrap.py
   - graph/nodes/plan.py
   - graph/nodes/code.py
   - graph/nodes/execute.py
   - graph/nodes/evaluate.py
   - graph/nodes/evolution.py
8. Create routing.py (1 hour)
   - Move all routing functions
   - Add tests for routing logic
9. Refactor ExecutionContext (2 hours)
   - Create AgentManager
   - Create ToolRegistry
   - Separate concerns

### P3: Testing
10. Add unit tests for nodes (6 hours)
    - Test each node with mock state
    - Test routing functions
    - Test error paths
11. Add integration tests (4 hours)
    - Test full graph flows
    - Mock LLM responses
    - Test checkpointing

### P4: Cleanup
12. Remove dead code (1 hour)
13. Fix all error handling (2 hours)
14. Add comprehensive docstrings (2 hours)
15. Update README (1 hour)

## Total Estimated Time: ~30 hours

## Success Criteria

### Code Quality
- [ ] No files > 500 lines
- [ ] All nodes have unit tests
- [ ] 100% of error paths logged
- [ ] No duplicate code
- [ ] Type hints everywhere

### LangGraph Idioms
- [ ] Using checkpointer
- [ ] Using ToolNode for tools
- [ ] Using interrupt() for HIL
- [ ] Using conditional_edges for routing
- [ ] Using subgraphs where appropriate

### Observability
- [ ] Full tool call tracing in LangSmith
- [ ] Clear error messages
- [ ] Resumable experiments
- [ ] Time-travel debugging support

### Testing
- [ ] >80% code coverage
- [ ] All critical paths tested
- [ ] Regression tests for fixed bugs
- [ ] Integration tests for main flows
