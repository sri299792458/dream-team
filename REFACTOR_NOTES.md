# Refactor Session Notes

## Architecture Snapshot
- **Entrypoints**: `scripts/run_with_tracing.py` (demo with LangSmith) and `experiments/agentds_food/run_autonomous_experiment.py` (full autonomous run) both call `dream_team.experiment.run_graph_experiment`.
- **Graph construction**: `src/dream_team/experiment/graph/builder.py` builds the LangGraph with nodes for bootstrap → init_math → plan → code → execute → evaluate → check_continue → check_evolution → evolve.
- **Node behaviors**: Implemented in `src/dream_team/experiment/graph/nodes.py` using `ExecutionContext` for executor, research API, and agent instances; meetings drive LLM output for planning/coding/recruiting.
- **State model**: Unified Pydantic state in `src/dream_team/experiment/graph/state.py` (`ExperimentState`, `ExperimentConfig`, `TeamConfig`, etc.) tracks iterations, history, metrics, evolution flags, and paths.
- **Tools & external deps**: Code execution via `CodeExecutor` (safe execution over provided `data_context`), research through `SemanticScholarAPI` (via `get_research_assistant`), and knowledge/evolution helpers in `knowledge_state.py`, `team.py`, and `evolution.py`.
- **Tracing**: `src/dream_team/experiment/tracing.py` configures optional LangSmith tracing used inside `run_graph_experiment`.

## Issues Observed
- **Bugs / correctness issues**:
  - No checkpointer used in `run_graph_experiment`, so runs cannot resume mid-graph.
  - Retry/fix loop in `create_execute_node` swallows specific exception types and may mask executor crashes.
- **Design smells / clutter**:
  - `nodes.py` remains very large and mixes helper utilities with node factories; branching logic is partially embedded inside node bodies instead of dedicated routers.
  - Tool execution is still handled manually via `IndividualMeeting` rather than LangGraph `ToolNode`/ReAct primitives.
  - State mutations happen in-place within nodes instead of returning partial updates.
- **Missing tests / guardrails**:
  - No coverage for routing functions (e.g., bootstrap vs. init math) or evolution decision logic.
  - No tests around failure paths in code execution retries or data schema refresh.
  - Lacking unit tests for `ExecutionContext` setup and state synchronization.
