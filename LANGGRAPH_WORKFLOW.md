# LangGraph End-to-End Workflow

Use this walkthrough to run the LangGraph implementation (recommended V2) from a clean checkout through result inspection and resume.

## 1) Prerequisites
- Install dependencies: `pip install -e .`
- Export your Gemini key: `export GEMINI_API_KEY=your_key_here`
- (Optional) Enable LangSmith tracing: `export LANGSMITH_TRACING=true` and `export LANGSMITH_API_KEY=your_key_here`
- Ensure the AgentDS Food dataset is present under `experiments/agentds_food/data/FoodProduction` (already included in this repo snapshot).

## 2) Pick an entry point
- **Standard V2 run (recommended):** `python experiments/agentds_food/run_langgraph_experiment_v2.py`
- **V2 with checkpoints + resume:** `python experiments/agentds_food/run_langgraph_experiment_v2_resume.py`
- **V1 baseline (for comparison):** `python experiments/agentds_food/run_langgraph_experiment.py`

Both V2 scripts stream progress to the console (Bootstrap → Team Planning → Code Generation → Execution → Evolution) and save outputs to `experiments/agentds_food/results/langgraph_v2_shelf_life/`.

## 3) Understand the graph
The enhanced graph is built in `src/dream_team/langgraph_orchestrator_v2.py`:
- **Nodes:** `bootstrap` → `team_planning` → `code_generation` → `execution` → `check_completion` → (`increment_iteration` | `evolution` | END)
- **Context:** `langgraph_context.py` builds an adaptive context window so column schemas and recent iterations are always available.
- **Agents:** `langgraph_agents.py` provides ReAct planners/researchers/coders with mathematical state (K, θ, δ) injected into prompts.
- **Meetings:** `langgraph_team_meeting.py` runs the multi-agent planning subgraph.
- **Checkpointing:** In-memory by default; `run_langgraph_experiment_v2_resume.py` supplies a SQLite checkpoint path for resumability.

## 4) Run the workflow (V2)
```bash
cd experiments/agentds_food
python run_langgraph_experiment_v2.py
```
What you’ll see:
1. Data loading shapes for all CSVs.
2. Bootstrap exploration plan + executed code that discovers schemas.
3. Automatic specialist recruitment after bootstrap.
4. Iterative cycles of team planning, code generation, execution, and optional evolution (up to 5 iterations by default).
5. Streaming progress markers such as `📍 Completed: Team Planning (Iteration 2)` and metrics after execution.

Artifacts saved under `results/langgraph_v2_shelf_life/` include execution logs, histories, code, and (when enabled) SQLite checkpoints.

## 5) Resume from a checkpoint
```bash
cd experiments/agentds_food
python run_langgraph_experiment_v2_resume.py
```
- The script detects existing checkpoints in `results/langgraph_v2_shelf_life/checkpoints/` and prompts to resume or start fresh.
- If resuming, provide the same thread ID (`food_shelf_life_v2` by default) and data location; state is restored automatically before streaming continues.

## 6) Compare versions
Run V1 and V2 back-to-back, then diff the summaries:
```bash
python experiments/agentds_food/run_langgraph_experiment.py
python experiments/agentds_food/run_langgraph_experiment_v2.py

diff results/langgraph_shelf_life/final_summary.json \
     results/langgraph_v2_shelf_life/final_summary.json
```
This highlights how smart context, ReAct agents, and multi-agent planning change the outputs.

## 7) Troubleshooting tips
- If LangSmith tracing is enabled without an API key, the scripts warn and continue without tracing.
- Use the console stream to locate failing nodes (e.g., code execution errors) and inspect the corresponding artifacts in `results/langgraph_v2_shelf_life/`.
- To restart cleanly, delete the checkpoint directory: `rm -rf experiments/agentds_food/results/langgraph_v2_shelf_life/checkpoints`.
