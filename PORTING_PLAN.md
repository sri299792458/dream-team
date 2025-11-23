# LangGraph + LangSmith Porting Plan (Refined)

This plan refines the earlier remediation outline to ensure full coverage and actionable steps for porting the Dream Team framework to LangGraph with LangSmith tracing. It is organized by workstream with concrete file touchpoints, state contracts, and validation criteria. The goal is a balanced roster that can fluidly combine domain experts and a deepening generalist as the problem demands, while keeping prompts framework-generic (no food-specific phrasing) and preserving the PI-led workflow (PI seeds exploration + research + initial recruits → team ReAct meeting with tool access → execution with full historical context → evolution/knowledge-graph update after each iteration).

## 1) Orchestration on LangGraph with tracing
- **State schema**: Define a typed state (e.g., `GraphState` in `src/dream_team/orchestrator.py`) that carries problem spec, dataset refs, agent roster, context, iteration counters, and artifacts. Persist via checkpointing and serialization hooks.
- **Graph topology**: Map current loops into LangGraph nodes: planning (`TeamMeeting` synthesis), implementation (`CodeExecutor`), evaluation, evolution decision, evolution action, and termination. Edges handle retries, failure detours, and max-iteration exits.
- **PI-led kickoff**: Model the PI as the initial node that (a) requests exploratory analysis from the coding agent, (b) runs research/semantic search to recruit initial specialists, and (c) seeds the first roster/context before handing off to the planning meeting.
- **Tracing**: Wrap graph execution with LangSmith callbacks (project/tag metadata) and propagate trace IDs into node logs. Each node should emit child spans for sub-steps (planning, tool calls, executor runs).
- **Entrypoints**: Add a `run_experiment` helper that builds/runs the graph and a backward-compatible adapter so the existing `ExperimentOrchestrator.run` delegates to LangGraph while preserving CLI/SDK ergonomics.

## 2) Context/state propagation
- **Context contract**: Introduce a typed `Context` dataclass (problem summary, data schema, prior metrics, artifact manifests, knowledge snippets). Pass it explicitly through graph state instead of ad-hoc dicts.
- **Prompt assembly**: Centralize prompt builders (shared with `Agent.prompt`, `TeamMeeting`, executor prompts) so persona/system text and context slices stay consistent and lean, explicitly avoiding benchmark-specific phrasing (e.g., food production) to keep the framework general.
- **Versioning & persistence**: Add context serialization/version fields to support checkpoint/resume and ensure meeting/executor nodes read/write the same contract.

## 3) Evolution lifecycle and triggers
- **Signals**: Standardize `EvolutionEngine` triggers using explicit metrics (performance deltas, error taxonomy, coverage, agent contribution effectiveness). Store thresholds/config in a single module.
- **Graph nodes**: Add a decision node that emits evolution actions (specialize/diversify/broaden) and a mutation node that applies them, updates `evolution_history`, persists snapshots via `serialization.py`, and refreshes the agent roster.
- **Knowledge graph update**: After each iteration, ingest execution outcomes, errors, and research findings into the knowledge graph so future agents (generalist or specialists) inherit grounded context and we avoid forgetting useful prior attempts.
- **Research integration**: When specialization triggers, pull supporting knowledge via `research.py`, updating `KnowledgeBase` and concept graphs; log rationale/results to LangSmith spans.
- **Mathematical state fidelity**: Preserve and surface the existing formalism from `knowledge_state.py` and `langgraph_agents.py` in LangGraph nodes and prompts: (a) knowledge graph K=(V,E,W) with overlap/weighted-overlap against problem concepts, (b) attention distribution θ over concepts with entropy/KL metrics, and (c) depth map δ with Gini/max-depth specialization scores. Ensure the decision node uses these metrics (e.g., evolving the lowest-max-depth agent) and that prompts expose specialization level, top concepts, and known papers to keep agents self-aware and grounded.

## 4) ReAct planner/executor harmonization
- **Reusable planner**: Extract the ReAct synthesis logic from `TeamMeeting` into a shared planner module with step templates, stop conditions, and temperature policy; reuse for coding prompts to standardize reasoning. Ensure each participant can call the search/tooling stack for grounding claims and has access to the full execution history (what worked/failed) to curb hallucinations.
- **Guardrails**: Enforce output schemas (action plan + code blocks), add validation/fallbacks when LLM output is malformed, and surface structured errors to retry logic.
- **Tracing**: Instrument each reasoning step/tool call as LangSmith child spans for debuggability and benchmarking.

## 5) Artifact and dataset handling
- **Artifact contract**: Define canonical artifact paths/metadata shared by executor and evaluators; ensure `CodeExecutor` reads/writes within this contract and records stderr/stdout with statuses.
- **Safety**: Add sandboxing/timeouts for executor runs, capturing rich error objects that inform retries and evolution triggers.
- **Reproducibility**: Attach artifact metadata and dataset hashes to LangSmith spans for repeatability.

## 6) Testing and configuration
- **Config**: Centralize settings (models, temperatures, retry counts, tracing toggles, timeouts) in a typed config module read from env/CLI, replacing scattered literals.
- **Tests**: Add unit/integration tests covering graph happy path, retries, evolution path, and context serialization. Use lightweight fixtures and mocks for LLM and executor.
- **Docs**: Update README and add examples showing graph execution with tracing; include migration notes for existing `experiments/` scripts to use the graph runner.

## 7) Execution/migration checklist
- Implement state/dataclass definitions and graph wiring before prompt refactors to avoid churn.
- Gate LangSmith tracing with config to support offline runs.
- Incrementally migrate existing CLI/experiment entrypoints to the new graph adapter and delete legacy loops once parity is validated.
- Keep PRs scoped by workstream (graph scaffold → context contract → evolution → ReAct → artifacts → tests/docs) to simplify review and rollout.
