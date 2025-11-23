# Resume & Checkpoint Guide

Complete guide to using persistent checkpoints and resuming experiments with Dream Team LangGraph V2.

## 🎯 Why Resume Matters

Long-running experiments can fail for many reasons:
- **Network issues** (API rate limits)
- **Code errors** (bugs in generated code)
- **Resource limits** (out of memory)
- **Manual interruption** (Ctrl+C to stop)
- **Power/system issues**

With **SqliteSaver checkpoints**, you can resume from **exactly where you left off** without losing progress.

## 🚀 Quick Start

### Run with Resume Support

```bash
cd experiments/agentds_food
python run_langgraph_experiment_v2_resume.py
```

That's it! The script will:
1. Check for existing checkpoints
2. Show you the current state
3. Ask if you want to resume
4. Continue from where it stopped

### What You'll See

**First run (no checkpoint):**
```
DREAM TEAM V2 - WITH RESUME SUPPORT
...
✅ Using SqliteSaver - checkpoints saved to .../checkpoints.db
🚀 Starting execution...
   Thread ID: food_shelf_life_v2
   Mode: Fresh start
   ...
```

**After interruption (checkpoint exists):**
```
DREAM TEAM V2 - WITH RESUME SUPPORT
...
🔄 EXISTING CHECKPOINT FOUND
================================================================================
RESUMABLE CHECKPOINT: food_shelf_life_v2
================================================================================

Current State:
  Iteration: 3
  Bootstrap completed: True
  Best mae: 42.3456

Team:
  Lead: Principal Investigator
  Members: ['ML Strategist', 'Time Series Expert']

Progress: 3 iterations completed

To resume: Run with the same thread_id
================================================================================

Resume from checkpoint? (y/n): y
✅ Resuming from checkpoint
...
```

## 📖 How It Works

### Checkpoints are Automatic

Every time a node completes, LangGraph saves a checkpoint:

```
Bootstrap → Checkpoint 1
Team Planning → Checkpoint 2
Code Generation → Checkpoint 3
Execution → Checkpoint 4
...
```

**If your experiment crashes or you interrupt it, the last successful checkpoint is preserved.**

### Resume Process

When you run with resume:

1. **Detect checkpoint** - Check if `checkpoints.db` exists for your thread_id
2. **Load state** - Restore exact state (iteration, team, metrics, history)
3. **Continue execution** - Pick up from next node after last checkpoint
4. **Preserve history** - All previous iterations remain intact

### Thread IDs

Each experiment has a `thread_id` - think of it as a unique experiment name:

```python
thread_id = "food_shelf_life_v2"
```

**Same thread_id = Resume from that experiment**
**Different thread_id = Start fresh experiment**

You can run multiple experiments in parallel with different thread_ids!

## 🔧 Checkpoint Management

### View Checkpoints

```bash
python scripts/manage_checkpoints.py inspect \
    experiments/agentds_food/results/langgraph_v2_shelf_life \
    food_shelf_life_v2
```

Output:
```
RESUMABLE CHECKPOINT: food_shelf_life_v2
================================================================================

Current State:
  Iteration: 3
  Bootstrap completed: True
  Best mae: 42.3456

Team:
  Lead: Principal Investigator
  Members: ['ML Strategist']

Progress: 3 iterations completed
```

### List All Checkpoints

```bash
python scripts/manage_checkpoints.py list \
    experiments/agentds_food/results/langgraph_v2_shelf_life \
    food_shelf_life_v2
```

### Export Checkpoint to JSON

```bash
python scripts/manage_checkpoints.py export \
    experiments/agentds_food/results/langgraph_v2_shelf_life \
    food_shelf_life_v2 \
    backup.json
```

Useful for:
- Backup before making changes
- Inspecting state structure
- Sharing state with collaborators

### Delete Checkpoints

```bash
# With confirmation
python scripts/manage_checkpoints.py delete \
    experiments/agentds_food/results/langgraph_v2_shelf_life \
    food_shelf_life_v2

# Force without confirmation
python scripts/manage_checkpoints.py delete -f \
    experiments/agentds_food/results/langgraph_v2_shelf_life \
    food_shelf_life_v2
```

### Clean Old Checkpoints

Keep only N latest checkpoints per thread (saves disk space):

```bash
python scripts/manage_checkpoints.py clean \
    experiments/agentds_food/results/langgraph_v2_shelf_life \
    --keep 10
```

## 🎯 Common Scenarios

### Scenario 1: Experiment Crashed

```bash
# Run experiment
python run_langgraph_experiment_v2_resume.py

# ... crashes at iteration 4 ...

# Just run again
python run_langgraph_experiment_v2_resume.py

# Prompt appears:
Resume from checkpoint? (y/n): y
# ✅ Continues from iteration 4
```

### Scenario 2: Stop to Make Code Changes

```bash
# Running experiment...
# Press Ctrl+C

⚠️  Interrupted by user
✅ Experiment state is checkpointed!
   Checkpoint: .../checkpoints.db
   Thread ID: food_shelf_life_v2

# Make your code changes in src/dream_team/...

# Resume
python run_langgraph_experiment_v2_resume.py
Resume from checkpoint? (y/n): y
# ✅ Continues with new code
```

### Scenario 3: Start Fresh (Ignore Checkpoint)

```bash
python run_langgraph_experiment_v2_resume.py

Resume from checkpoint? (y/n): n
Starting fresh experiment (old checkpoint will be overwritten)
# ✅ Starts from scratch
```

### Scenario 4: Multiple Experiments in Parallel

```python
# Edit the script to use different thread_ids:

# Experiment 1: Standard approach
thread_id = "food_shelf_life_standard"

# Experiment 2: Deep learning approach
thread_id = "food_shelf_life_deep"

# Experiment 3: Ensemble approach
thread_id = "food_shelf_life_ensemble"
```

Each will have independent checkpoints!

## 💾 Checkpoint Storage

### Location

```
experiments/agentds_food/results/langgraph_v2_shelf_life/
└── checkpoints/
    └── checkpoints.db  ← SQLite database with all checkpoints
```

### Size

- ~10-50KB per checkpoint
- Grows with team size and history
- Clean old checkpoints regularly if disk space is limited

### Persistence

**Checkpoints persist across:**
- ✅ Process restarts
- ✅ System reboots
- ✅ Code changes
- ✅ Days/weeks/months

**Checkpoints DO NOT survive:**
- ❌ Deleting `checkpoints.db`
- ❌ Deleting results directory
- ❌ Running `manage_checkpoints.py delete`

## 🐛 Debugging with Checkpoints

### Inspect State at Failure

```bash
# Experiment failed at iteration 5

# Inspect what state it was in:
python scripts/manage_checkpoints.py inspect \
    experiments/agentds_food/results/langgraph_v2_shelf_life \
    food_shelf_life_v2

# Export for detailed inspection:
python scripts/manage_checkpoints.py export \
    experiments/agentds_food/results/langgraph_v2_shelf_life \
    food_shelf_life_v2 \
    failed_state.json

# Open failed_state.json in editor to see exact state
```

### Compare States Across Runs

```bash
# Export checkpoint after iteration 3
python scripts/manage_checkpoints.py export ... iter3.json

# Make changes and run to iteration 3 again

# Export new checkpoint
python scripts/manage_checkpoints.py export ... iter3_v2.json

# Compare
diff iter3.json iter3_v2.json
```

## ⚙️ Advanced Usage

### Custom Thread IDs

```python
# In your script:
import datetime

# Use timestamp for unique experiments
thread_id = f"food_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Use configuration in thread_id
thread_id = f"food_max_iter_{max_iterations}_target_{target_score}"

# Use git branch/commit
import subprocess
branch = subprocess.check_output(['git', 'branch', '--show-current']).decode().strip()
thread_id = f"food_{branch}"
```

### Programmatic Resume Check

```python
from dream_team.checkpoint_manager import check_resume_available

results_dir = Path("results/langgraph_v2_shelf_life")
thread_id = "food_shelf_life_v2"

if check_resume_available(results_dir, thread_id):
    print("Can resume from checkpoint")
    # Load checkpoint manager and inspect state
    manager = get_checkpoint_manager(results_dir)
    state = manager.get_checkpoint_state(thread_id)
    print(f"Last iteration: {state.get('iteration')}")
else:
    print("No checkpoint available")
```

### Conditional Resume

```python
# Auto-resume if less than 1 hour old
import time
from pathlib import Path

checkpoint_db = results_dir / "checkpoints" / "checkpoints.db"

if checkpoint_db.exists():
    age_seconds = time.time() - checkpoint_db.stat().st_mtime
    age_hours = age_seconds / 3600

    if age_hours < 1:
        print(f"Recent checkpoint ({age_hours:.1f}h old) - auto-resuming")
        resume = True
    else:
        print(f"Old checkpoint ({age_hours:.1f}h old) - starting fresh")
        resume = False
```

## ❓ FAQ

**Q: What if I change the code between runs?**
A: Checkpoints store state, not code. New code will be used when resuming. Just make sure changes are compatible with the state.

**Q: Can I resume on a different machine?**
A: Yes! Just copy the entire `results/` directory (including `checkpoints/`) to the new machine.

**Q: What happens if checkpoint is corrupted?**
A: SQLite is robust, but if corrupted, the script will fall back to starting fresh. Your iteration JSON files remain intact.

**Q: Does resume work with LangSmith?**
A: Yes! LangSmith will show the full trace including resumed execution.

**Q: Can I resume from a specific iteration, not the latest?**
A: Currently only latest checkpoint is supported. For specific iterations, export that checkpoint's state and create a new script to initialize from it.

**Q: How do I know if resume worked?**
A: Check the iteration number - it should continue from where it stopped, not restart at 1.

## 📊 Comparison: Original vs V2 Resume

| Feature | Original | LangGraph V2 |
|---------|----------|--------------|
| Checkpointing | Manual JSON files | Automatic SQLite |
| Resume | Manual detection | Automatic detection |
| State preservation | Partial (metrics only) | Complete (full state) |
| Resume granularity | Per iteration | Per node |
| Works across restarts | ✅ | ✅ |
| Interactive prompt | ❌ | ✅ |
| Checkpoint management | Manual | CLI tools |

## 🎉 Summary

Resume functionality makes Dream Team experiments **production-ready**:

✅ **Never lose progress** - Checkpoints after every node
✅ **Easy to use** - Just run the script again
✅ **Flexible** - Resume, skip, or start fresh
✅ **Robust** - SQLite persistence
✅ **Debuggable** - Inspect state at any point

**Use `run_langgraph_experiment_v2_resume.py` for all long-running experiments!**
