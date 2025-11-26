# Dream Team Framework

**Dream Team** is a dynamic multi-agent framework with evolving personas. Agents start with general expertise and progressively specialize based on:

- Performance bottlenecks and error patterns
- Research paper integration (Semantic Scholar)
- Knowledge gaps discovered during collaboration
- Domain-specific challenges encountered

## Key Features

### 🧬 Evolving Agents
Agents don't have fixed personas—they evolve their expertise as research progresses:
- Start as generalists (e.g., "Data Scientist")
- Evolve into specialists (e.g., "Perishable Goods Physicist" with thermal abuse modeling expertise)
- Track complete evolution history and knowledge accumulation

### 📚 Research Integration
Agents autonomously research topics using Semantic Scholar:
- LLM analyzes paper relevance and extracts actionable insights
- Key findings integrated into agent knowledge bases
- Techniques from papers applied to solve challenges

### 🔄 Evolution Triggers
Automatic detection of when agents need to evolve:
- **Performance Plateau**: Metrics stop improving
- **Error Patterns**: Specific failure modes concentrated
- **Knowledge Gaps**: Agents express uncertainty in meetings

### 🎯 Benchmark-Driven Development
Fast iteration optimized for well-defined problems:
- Clear evaluation metrics (MAE, F1, RMSE)
- Leaderboard-driven hill climbing
- Observable progress tracking

### 🤖 Autonomous Experimentation
Agents operate with full autonomy:
- **Code Generation**: Agents write their own Python code
- **Execution**: Safe code execution environment
- **Self-Iteration**: Analyze results and adjust approach
- **Zero Handholding**: Give agents problem + data, they do the rest

## Quick Start

### Autonomous Experiment (Recommended)

Run a complete autonomous experiment with LangGraph orchestration:

```bash
# Quick example with tracing
python scripts/run_with_tracing.py

# Or run your own experiment
cd experiments/agentds_food
python run_autonomous_experiment.py
```

The graph autonomously:
1. **Bootstrap**: PI explores problem and recruits specialized team
2. **Plan**: Team collaborates on approach
3. **Code**: Generate executable Python code
4. **Execute**: Run code in safe environment
5. **Evaluate**: Extract metrics and analyze results
6. **Evolve**: Adapt team composition when stuck
7. **Iterate**: Repeat until goal achieved

### API Usage (LangGraph)

```python
from dream_team.experiment import (
    create_initial_state,
    AgentConfig,
    run_graph_experiment
)

# 1. Configure agents
team_lead = AgentConfig(
    title="Principal Investigator",
    expertise="machine learning, experimental design, research methodology",
    goal="optimize the target metric through systematic experimentation",
    role="explore problem, recruit team, coordinate research"
)

coding_agent = AgentConfig(
    title="Research Engineer",
    expertise="Python, pandas, scikit-learn, numpy, data analysis",
    goal="implement research plans accurately",
    role="translate strategies into executable code"
)

# 2. Create initial state
state = create_initial_state(
    team_lead=team_lead,
    coding_agent=coding_agent,
    problem_statement="""
Predict the target variable using available features.
Evaluation Metric: MAE (Mean Absolute Error) - lower is better
Goal: Build a predictive model to minimize MAE.
""",
    target_metric="mae",
    minimize_metric=True,
    max_iterations=10,
    results_dir="results/experiment"
)

# 3. Prepare data
data_context = {
    'train_df': train_df,  # Your training data
    'test_df': test_df     # Your test data
}

# 4. Run graph with optional LangSmith tracing
final_state = run_graph_experiment(
    state,
    data_context,
    enable_tracing=True,  # Optional: Full observability in LangSmith
    langsmith_project="my-experiments"
)

# 5. Access results
print(f"Best {final_state.config.target_metric}: {final_state.best_metric}")
print(f"Iterations: {final_state.iteration}")
print(f"Team size: {len(final_state.team.team_members) + 1}")
```

### Component Usage (Advanced)

For direct access to individual components:

```python
from dream_team import Agent, TeamMeeting, EvolutionEngine, get_research_assistant

# Create agents
pi = Agent(
    title="Principal Investigator",
    expertise="data science, ML, research strategy",
    goal="solve the prediction challenge",
    role="lead team and make decisions"
)

# Run team meeting
meeting = TeamMeeting(save_dir="results/meetings")
summary = meeting.run(
    team_lead=pi,
    team_members=[data_scientist],
    agenda="Predict shelf life using temperature data",
    num_rounds=2
)

# Research papers
research = get_research_assistant()
papers = research.research_topic(
    query="shelf life prediction food storage",
    num_papers=5
)

# Evolve agent
evolution = EvolutionEngine()
evolution.evolve_agent(
    agent=data_scientist,
    context={"problem": "Shelf life prediction"},
    papers=papers,
    trigger_reason="Need domain expertise"
)
```

## Architecture

### LangGraph Orchestration (New)

```
src/dream_team/
├── experiment/              # LangGraph-based orchestration (NEW)
│   ├── state.py            # ExperimentState, AgentConfig (Pydantic models)
│   ├── graph_app.py        # Graph construction and runner
│   ├── nodes.py            # All node implementations
│   ├── tracing.py          # LangSmith integration
│   └── __init__.py         # Public API
├── agent.py                 # Agent, KnowledgeBase, Paper classes
├── llm.py                   # Gemini API wrapper
├── research.py              # Semantic Scholar integration
├── evolution.py             # Evolution engine and triggers
├── meetings.py              # Team and individual meetings
├── executor.py              # Code execution environment
├── orchestrator.py          # (Deprecated - use experiment/ instead)
└── utils.py                 # Helper functions

scripts/
├── run_with_tracing.py      # Example with LangSmith tracing
├── smoke_run.py             # Baseline test
└── test_graph_skeleton.py   # Graph structure test

tests/
├── test_nodes.py            # Unit tests for nodes
└── test_graph_integration.py # Integration tests

experiments/agentds_food/
├── data/                    # Benchmark data
├── results/                 # Evolution history, meetings, experiments
└── run_autonomous_experiment.py
```

**Graph Flow:**
```
START → bootstrap → init_math → plan → code → execute → evaluate
                                  ↑                        ↓
                                  └── [evolve?] ← check ←─┘
```

Each node accepts and returns `ExperimentState`, enabling full observability and checkpointing.

### Migration Note

> **Migrating from old API?** See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for step-by-step instructions.

## New Features

### 🔍 LangSmith Tracing

Full observability into experiment execution:

```bash
export LANGSMITH_API_KEY=your_key
python scripts/run_with_tracing.py
```

View traces at https://smith.langchain.com/ to see:
- Each node execution (bootstrap, plan, code, execute, etc.)
- Iteration progress and metrics evolution
- Team composition changes
- Evolution decisions and triggers
- Timing for each step

### 📊 Explicit State Management

All experiment state is now explicit and type-safe:

```python
final_state.iteration        # Current iteration number
final_state.best_metric      # Best metric achieved
final_state.team             # Current team composition
final_state.history          # Complete iteration history
final_state.evolution        # Evolution state and decisions
```

The `ExperimentState` Pydantic model provides:
- Type safety with validation
- Serializable state for checkpointing
- Autocomplete support in IDEs
- Single source of truth

## Installation

```bash
# Clone the repository
git clone https://github.com/sri299792458/dream-team.git
cd dream-team

# Install dependencies
pip install -e .

# Optional: Install extras for experiments
pip install -e ".[dream-team]"
```

## Setup

```bash
# Set Gemini API key (free tier available)
export GEMINI_API_KEY=your_key_here
```

## Current Application: AgentDS Food Production Benchmark

Dream Team is being applied to the [AgentDS Food Production domain](https://agentds.org/domains/food):

1. **Shelf Life Prediction** - Predict remaining days (MAE metric)
2. **Quality Control** - Pass/fail classification (Macro-F1)
3. **Weekly Demand Forecasting** - Units sold prediction (RMSE)

See `experiments/agentds_food/` for the evolving research prototype.

## Philosophy

> **A multi-agent system that can evolve its agents learns to solve problems at a meta-level—not just solving Task X, but learning *how to become the type of team that solves tasks like X*.**

The framework enables:
- **Emergent expertise**: Agents discover what specialization is needed
- **Research-driven evolution**: Paper insights drive persona synthesis
- **Observable learning**: Complete history of evolution, decisions, and knowledge growth

## License

MIT License - see LICENSE file for details.

## Citation

If you use Dream Team in your research, please cite:

```bibtex
@software{dream_team_2024,
  title={Dream Team: A Dynamic Multi-Agent Framework with Evolving Personas},
  author={Sri},
  year={2024},
  url={https://github.com/sri299792458/dream-team}
}
```
