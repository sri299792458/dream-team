# AgentDS Food Production Benchmark

This directory contains experiments for the AgentDS Food Production domain benchmark.

## Challenges

1. **Shelf Life Prediction** (MAE) - Predict remaining shelf life days
2. **Quality Control Pass/Fail** (Macro-F1) - Binary classification for quality
3. **Weekly Demand Forecasting** (RMSE) - Predict units sold next week

## Directory Structure

```
agentds_food/
├── data/               # Raw benchmark data
├── processed/          # Processed/cached data
├── results/            # Experiment outputs
│   ├── shelf_life/
│   ├── quality_control/
│   └── demand_forecast/
├── notebooks/          # Jupyter notebooks for experiments
└── task_configs/       # YAML configurations per challenge
```

## Getting Started

1. Download AgentDS Food domain data and place in `data/`
2. Set environment variable: `export GEMINI_API_KEY=your_key_here`
3. Run notebooks in order

## LangGraph workflow

Prefer the LangGraph implementations for fully automated runs. Key scripts:
- `run_langgraph_experiment_v2.py`: Enhanced V2 workflow with streaming progress.
- `run_langgraph_experiment_v2_resume.py`: Same as V2 but with SQLite checkpoints and resume support.

See `../../LANGGRAPH_WORKFLOW.md` for an end-to-end walkthrough (prerequisites, commands, checkpoints, and troubleshooting).

## Evolution Tracking

Each challenge tracks:
- Agent evolution history (persona changes)
- Meeting transcripts (decision process)
- Knowledge base growth (papers, techniques, insights)
- Experiment results (metrics over time)
