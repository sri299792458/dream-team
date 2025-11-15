# Quick Start Guide: Autonomous Shelf Life Prediction

Run your first fully autonomous Dream Team experiment on the AgentDS Food Production benchmark.

## What is Autonomous Mode?

Agents receive **problem statement + data** and autonomously:
- ✅ Plan their own approach
- ✅ Write their own code
- ✅ Execute and analyze results
- ✅ Evolve when stuck
- ✅ Iterate until goal achieved

**Zero handholding - full agent autonomy!**

---

## Prerequisites

1. **Install dependencies**:
   ```bash
   cd ~/dream-team
   pip install -e ".[dream-team]"
   ```

2. **Set up Gemini API key**:
   ```bash
   export GEMINI_API_KEY=your_key_here
   ```

   Get a free key at: https://aistudio.google.com

3. **Place your data**:
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

---

## Running the Autonomous Experiment

```bash
cd experiments/agentds_food
python run_autonomous_experiment.py
```

### What Happens

**Iteration 1:**
1. Agents hold team meeting → discuss approach
2. Data Scientist writes Python code (feature engineering, model, evaluation)
3. Code executes → outputs metrics
4. Results recorded

**Iteration 2:**
1. Agents review iteration 1 results
2. Adjust strategy based on performance
3. Write improved code
4. Execute and evaluate

**Iteration 3+:**
1. If performance plateaus → evolution trigger fires
2. Agents research academic papers (via Semantic Scholar)
3. Agents evolve with domain expertise (e.g., food science)
4. New specialized approach with evolved knowledge
5. Continue iterating

**Continues until:**
- Target metric achieved, OR
- Maximum iterations reached

---

## Output Files

All results saved to: `results/autonomous_shelf_life/`

```
results/autonomous_shelf_life/
├── iteration_01.json           # Complete iteration 1 record
├── iteration_02.json           # Iteration 2
├── iteration_03.json           # etc.
├── code/
│   ├── iteration_01.py         # Agent-generated code (iteration 1)
│   ├── iteration_02.py         # Evolved approach (iteration 2)
│   └── iteration_03.py         # Post-evolution code
├── meetings/
│   ├── team_meeting_*.json     # Team planning discussions
│   └── individual_meeting_*.json  # Code generation sessions
├── agents/
│   ├── data_scientist_iter_3.json  # Agent snapshot after evolution
│   └── ...
└── final_summary.json          # Complete experiment summary
```

### Key Files Explained

**`iteration_*.json`** - Complete record of each iteration:
- Approach decided in meeting
- Code generated
- Execution results
- Metrics achieved
- Agent states

**`code/iteration_*.py`** - Actual Python code written by agents:
- Feature engineering functions
- Model training code
- Cross-validation logic
- All agent-generated!

**`meetings/*.json`** - Full meeting transcripts:
- See agent discussions
- Evolution of strategy
- Decision-making process

**`agents/*.json`** - Agent evolution snapshots:
- Compare initial vs evolved personas
- Knowledge base growth
- Papers researched

---

## Example Output

```
======================================================================
🚀 AUTONOMOUS DREAM TEAM EXPERIMENT
======================================================================

Problem: Predict remaining shelf life in days for food production batches
Target Metric: mae (minimize)
Max Iterations: 5

============================================================
ITERATION 1/5
============================================================

👥 Team planning meeting...

📋 TEAM MEETING
   Lead: Principal Investigator
   Members: ['Data Scientist']
   Rounds: 2

💬 Principal Investigator:
Let's analyze this shelf life prediction challenge...

💬 Data Scientist:
Based on the data, I propose feature engineering focused on...

💻 Implementing approach...

👤 INDIVIDUAL MEETING
   Agent: Data Scientist
   Iterations: 1

💬 Data Scientist (initial):
```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Feature engineering
train_features = batches_train.merge(products, on='sku_id')
...
```

⚙️ Executing implementation...

   ✅ Success
   Output: Cross-validation MAE: 3.245

✨ New best mae: 3.2450

============================================================
ITERATION 2/5
============================================================

... (agents iterate, evolve, and improve)
```

---

## What the Framework Does

### 1. Agent Planning
Agents meet to discuss the problem and plan their approach based on:
- Problem statement
- Available data
- Previous iteration results (if any)

### 2. Code Generation
Agent writes complete Python code:
- Data loading and merging
- Feature engineering
- Model selection and training
- Cross-validation
- Metric calculation

### 3. Execution
Code runs in safe environment with access to:
- All data (train/test, reference tables)
- Common libraries (pandas, numpy, scikit-learn)
- Previous iteration variables

### 4. Evolution
When performance plateaus:
- Research papers via Semantic Scholar
- Extract relevant techniques
- Evolve agent with domain expertise
- New approach based on research

### 5. Iteration
Continues until:
- Target achieved
- Max iterations reached
- Agents decide they've optimized sufficiently

---

## Troubleshooting

### Issue: `GEMINI_API_KEY not set`
```bash
export GEMINI_API_KEY=your_key_here
```
Get free key at: https://aistudio.google.com

### Issue: Data directory not found
Ensure FoodProduction folder is at:
```
~/dream-team/experiments/agentds_food/data/FoodProduction/
```

### Issue: Import errors
```bash
cd ~/dream-team
pip install -e ".[dream-team]"
```

### Issue: Code execution fails
- Check generated code in `results/autonomous_shelf_life/code/`
- Review execution error in iteration JSON file
- Agents will see errors and adjust in next iteration

### Issue: API rate limits
- Gemini has generous free tier
- If hit limits, add delays or reduce max_iterations

---

## Understanding Agent Behavior

### Meeting Transcripts
Read `meetings/team_meeting_*.json` to see:
- How agents analyze the problem
- What strategies they propose
- How they incorporate previous results
- Evolution of thinking over iterations

### Generated Code
Inspect `code/iteration_*.py` to see:
- What features agents engineered
- Model choices and hyperparameters
- How code evolves iteration to iteration
- Impact of agent evolution on code quality

### Agent Evolution
Compare agent snapshots:
- `agents/data_scientist_initial.json` (if manually saved)
- `agents/data_scientist_iter_3.json` (post-evolution)

See:
- Title change (e.g., "Data Scientist" → "Food Science ML Specialist")
- Expertise deepening
- Knowledge base growth (papers, techniques, facts)

---

## Next Steps

### Improve Performance
1. **Increase iterations**: Change `max_iterations=5` to higher value
2. **Add agents**: Include more team members with different expertise
3. **Tune evolution triggers**: Adjust plateau detection sensitivity

### Try Other Challenges
```bash
# Quality Control (Challenge 2)
# Modify problem_statement and data_context in run_autonomous_experiment.py

# Demand Forecasting (Challenge 3)
# Use demand_train.csv and demand_test.csv
```

### Experiment with Prompts
Edit `run_autonomous_experiment.py`:
- Modify `problem_statement` to guide agents differently
- Change agent initial expertise
- Adjust max_iterations, target_score

### Extend the Framework
- Add new evolution triggers (error patterns, knowledge gaps)
- Implement agent specialization (multiple agents evolving differently)
- Add more sophisticated code execution (timeouts, sandboxing)

---

## Philosophy

> **The framework learns to become the type of team that solves tasks like this.**

Unlike traditional ML pipelines:
- ❌ No hardcoded features
- ❌ No predetermined models
- ❌ No fixed approach

Instead:
- ✅ Agents discover what works
- ✅ Evolve based on results
- ✅ Research when stuck
- ✅ Emergent expertise

**This is meta-learning: learning how to learn to solve problems.**

---

## Support

- **Issues**: https://github.com/sri299792458/dream-team/issues
- **Docs**: Main README.md in repo root
- **Framework code**: `src/dream_team/`

Happy experimenting! 🚀
