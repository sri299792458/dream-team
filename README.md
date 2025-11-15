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

### Autonomous Mode (Recommended)

```bash
cd experiments/agentds_food
python run_autonomous_experiment.py
```

Agents autonomously:
1. Plan approach based on problem
2. Write and execute code
3. Analyze results
4. Evolve when stuck
5. Iterate until goal achieved

### API Usage

```python
from dream_team import Agent, TeamMeeting, EvolutionEngine, get_research_assistant

# Create initial team
pi = Agent(
    title="Principal Investigator",
    expertise="data science, ML, research strategy",
    goal="solve the prediction challenge",
    role="lead team and make decisions"
)

data_scientist = Agent(
    title="Data Scientist",
    expertise="EDA, feature engineering, statistical modeling",
    goal="understand data and create features",
    role="analyze data and propose models"
)

# Run meeting
meeting = TeamMeeting(save_dir="results/meetings")
summary = meeting.run(
    team_lead=pi,
    team_members=[data_scientist],
    agenda="Predict shelf life for food batches using temperature and humidity data",
    num_rounds=2
)

# Research relevant papers
research = get_research_assistant()
papers = research.research_topic(
    query="shelf life prediction food storage temperature",
    context="Predicting remaining shelf life based on storage conditions",
    num_papers=5
)

# Evolve agent with new knowledge
evolution = EvolutionEngine()
evolution.evolve_agent(
    agent=data_scientist,
    context={"problem_description": "Shelf life prediction", "error_analysis": {...}},
    papers=papers,
    trigger_reason="Need domain expertise in food science"
)

# data_scientist is now specialized with paper insights integrated!
```

## Architecture

```
src/dream_team/
├── agent.py           # Agent, KnowledgeBase, Paper classes
├── llm.py             # Gemini API wrapper
├── research.py        # Semantic Scholar integration
├── evolution.py       # Evolution engine and triggers
├── meetings.py        # Team and individual meetings
├── executor.py        # Code execution environment
├── orchestrator.py    # Autonomous experiment orchestration
└── utils.py           # Helper functions

experiments/agentds_food/
├── data/                        # Benchmark data
├── results/                     # Evolution history, meetings, experiments
└── run_autonomous_experiment.py # Fully autonomous experiment
```

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
