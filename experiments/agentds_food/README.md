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

## Evolution Tracking

Each challenge tracks:
- Agent evolution history (persona changes)
- Meeting transcripts (decision process)
- Knowledge base growth (papers, techniques, insights)
- Experiment results (metrics over time)
