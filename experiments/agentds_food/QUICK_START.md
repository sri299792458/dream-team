# Quick Start Guide: Shelf Life Prediction Experiment

This guide will help you run your first Dream Team experiment on the AgentDS Food Production benchmark.

## Prerequisites

1. **Install dependencies** (if not already done):
   ```bash
   cd ~/dream-team
   pip install -e ".[dream-team]"
   ```

2. **Set up Gemini API key**:
   ```bash
   export GEMINI_API_KEY=your_key_here
   ```

3. **Data placement**: Place the FoodProduction data in the correct location:
   ```
   experiments/agentds_food/data/FoodProduction/
   ├── batches_train.csv
   ├── batches_test.csv
   ├── demand_train.csv
   ├── demand_test.csv
   ├── lots_train.csv
   ├── lots_test.csv
   ├── products.csv
   ├── sites.csv
   ├── regions.csv
   └── market_memos.csv
   ```

## Running the Experiment

### Option 1: Jupyter Notebook (Recommended)

```bash
cd experiments/agentds_food/notebooks
jupyter notebook 01_shelf_life_prediction.ipynb
```

Then execute cells in order to:
1. Load and explore the data
2. Create initial team of agents
3. Run team meetings to discuss approach
4. Research academic papers on shelf life prediction
5. Evolve agents with domain expertise
6. Engineer features based on agent insights
7. Train and evaluate models
8. Generate test predictions

### Option 2: Python Script

Create a script `run_experiment.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent.parent / 'src'))

import pandas as pd
from dream_team import Agent, TeamMeeting, EvolutionEngine, get_research_assistant

# Load data
DATA_DIR = Path('experiments/agentds_food/data/FoodProduction')
batches_train = pd.read_csv(DATA_DIR / 'batches_train.csv')

# Create team
pi = Agent(
    title="Principal Investigator",
    expertise="data science, ML, research strategy",
    goal="solve shelf life prediction challenge",
    role="lead team and make decisions"
)

data_scientist = Agent(
    title="Data Scientist",
    expertise="EDA, feature engineering, statistical modeling",
    goal="understand data and create features",
    role="analyze data and propose models"
)

# Run meeting
meeting = TeamMeeting(save_dir="experiments/agentds_food/results/shelf_life/meetings")
summary = meeting.run(
    team_lead=pi,
    team_members=[data_scientist],
    agenda="Analyze shelf life prediction challenge and develop modeling strategy",
    num_rounds=2
)

print(summary)
```

Run with:
```bash
python run_experiment.py
```

## What the Framework Does

### 1. **Initial Team Creation**
- Creates agents with general expertise
- Each agent has a role, expertise, and goal

### 2. **Team Meetings**
- Agents discuss the problem
- Lead coordinates discussion rounds
- Generates meeting transcripts and summaries

### 3. **Research Phase**
- Agents search academic papers via Semantic Scholar
- LLM analyzes paper relevance
- Extracts key findings and techniques

### 4. **Agent Evolution**
- Agents evolve from generalists to specialists
- Knowledge bases grow with papers and insights
- Evolution history tracked

### 5. **Iterative Improvement**
- Features engineered based on agent insights
- Models trained and evaluated
- Performance triggers further evolution

## Expected Outputs

After running the notebook, you'll have:

```
results/shelf_life/
├── agents/                          # Agent snapshots
│   ├── pi_initial.json
│   ├── data_scientist_initial.json
│   ├── data_scientist_evolved_v1.json
│   └── ml_engineer_initial.json
├── meetings/                        # Meeting transcripts
│   ├── team_meeting_*.json
│   └── individual_meeting_*.json
├── predictions_baseline.csv         # Test predictions
└── baseline_results.json            # Performance metrics
```

## Experiment Results

The notebook will show:
- **Cross-validation MAE**: Model performance estimate
- **Feature importance**: Which features matter most
- **Agent evolution**: How agents specialized
- **Knowledge base**: Papers and techniques learned
- **Test predictions**: Ready for submission

## Next Steps

1. **Improve Performance**:
   - Try different models (GradientBoosting, XGBoost)
   - Hyperparameter tuning
   - More sophisticated feature engineering

2. **Trigger Evolution**:
   - Run multiple iterations to build history
   - Performance plateau will trigger agent evolution
   - Research more papers when stuck

3. **Error Analysis**:
   - Analyze prediction errors
   - Identify patterns in failures
   - Use insights to evolve specialists

4. **Try Other Challenges**:
   - Quality Control (Challenge 2)
   - Demand Forecasting (Challenge 3)

## Troubleshooting

### Issue: Data not found
**Solution**: Ensure FoodProduction folder is in `experiments/agentds_food/data/`

### Issue: GEMINI_API_KEY not set
**Solution**:
```bash
export GEMINI_API_KEY=your_key_here
```
Get a free key at: https://aistudio.google.com

### Issue: Module not found
**Solution**:
```bash
cd ~/dream-team
pip install -e ".[dream-team]"
```

### Issue: Permission errors
**Solution**:
```bash
chmod -R 755 experiments/agentds_food/
```

## Understanding the Output

### Meeting Transcripts
- Located in `results/shelf_life/meetings/`
- Shows agent discussions and decisions
- Tracks reasoning process

### Agent Evolution
- Each evolution creates a new snapshot
- Compare `initial.json` vs `evolved_v1.json`
- See knowledge base growth

### Performance Metrics
- `baseline_results.json` contains:
  - CV MAE scores
  - Feature count
  - Model metadata
  - Timestamp

## Philosophy

The Dream Team framework learns **how to become the type of team that solves tasks like this**:
- Agents discover needed specializations
- Research drives persona evolution
- Complete history provides observability

Happy experimenting! 🚀
