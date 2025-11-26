#!/usr/bin/env python3
"""
Example: Running Dream Team with LangSmith Tracing

This script demonstrates how to run experiments with full observability
using LangSmith tracing.

Setup:
    1. Get a LangSmith API key from https://smith.langchain.com/
    2. Set environment variable:
       export LANGSMITH_API_KEY=your_key_here
    3. Run this script:
       python scripts/run_with_tracing.py

Features:
    - Full observability of all graph nodes
    - Track iteration progress
    - Monitor evolution decisions
    - View team composition changes
    - Trace metrics over time

View traces at: https://smith.langchain.com/
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dream_team.experiment import (
    create_initial_state,
    AgentConfig,
    run_graph_experiment,
    configure_langsmith
)


def main():
    print("="*70)
    print("DREAM TEAM with LANGSMITH TRACING")
    print("="*70)
    print()

    # Check for API key
    if not os.getenv('LANGSMITH_API_KEY'):
        print("❌ LANGSMITH_API_KEY not set!")
        print()
        print("To enable tracing:")
        print("1. Get API key from: https://smith.langchain.com/")
        print("2. Export the key:")
        print("   export LANGSMITH_API_KEY=your_key_here")
        print()
        print("Running WITHOUT tracing (experiment will still work)...\n")
        enable_tracing = False
    else:
        enable_tracing = True

    # Create agent configs
    pi_config = AgentConfig(
        title="Principal Investigator",
        expertise="machine learning, experimental design, research methodology",
        goal="optimize the target metric through systematic experimentation",
        role="explore problem, recruit team, coordinate research"
    )

    coding_config = AgentConfig(
        title="Research Engineer",
        expertise="Python, pandas, scikit-learn, numpy, data analysis",
        goal="implement research plans accurately",
        role="translate strategies into executable code"
    )

    # Create test data
    print("📊 Creating synthetic test dataset...")
    np.random.seed(42)

    train_df = pd.DataFrame({
        'id': range(100),
        'feature_1': np.random.randn(100),
        'feature_2': np.random.randn(100),
        'feature_3': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.randn(100) * 10 + 50
    })

    test_df = pd.DataFrame({
        'id': range(100, 120),
        'feature_1': np.random.randn(20),
        'feature_2': np.random.randn(20),
        'feature_3': np.random.choice(['A', 'B', 'C'], 20)
    })

    print(f"   Train: {len(train_df)} rows")
    print(f"   Test: {len(test_df)} rows\n")

    # Create initial state
    state = create_initial_state(
        team_lead=pi_config,
        coding_agent=coding_config,
        problem_statement="""
Predict the target variable using available features.

Target Variable: target (continuous)
Evaluation Metric: MAE (Mean Absolute Error) - lower is better

Data:
- train_df: Training data with target
- test_df: Test data (predict target)

Goal: Build a predictive model to minimize MAE.
""",
        target_metric="mae",
        minimize_metric=True,
        max_iterations=2,  # Just 2 iterations for demo
        results_dir="scripts/traced_experiment_results"
    )

    # Prepare data context
    data_context = {
        'train_df': train_df,
        'test_df': test_df
    }

    # Run with tracing
    print("🚀 Running experiment with tracing...\n")

    if enable_tracing:
        print("📊 LangSmith tracing enabled!")
        print("   You'll be able to see:")
        print("   - Each node execution (bootstrap, plan, code, execute, etc.)")
        print("   - Iteration progress")
        print("   - Metrics evolution")
        print("   - Team composition changes")
        print("   - Evolution decisions")
        print()
        print("   View traces at: https://smith.langchain.com/")
        print()

    try:
        final_state = run_graph_experiment(
            state,
            data_context,
            enable_tracing=enable_tracing,
            langsmith_project="dream-team-demo"
        )

        # Print summary
        print("\n" + "="*70)
        print("EXPERIMENT SUMMARY")
        print("="*70)
        print(f"\nIterations completed: {final_state.iteration}")
        print(f"Best MAE achieved: {final_state.best_metric:.4f}")
        print(f"\nFinal Team:")
        print(f"  - {final_state.team.team_lead.title} (Lead)")
        for member in final_state.team.team_members:
            print(f"  - {member.title}")

        print(f"\nResults saved to: {final_state.results_dir}")

        if enable_tracing:
            print("\n" + "="*70)
            print("TRACING INFO")
            print("="*70)
            print(f"\n✅ Traces available at: https://smith.langchain.com/")
            print(f"   Project: dream-team-demo")
            print(f"   Experiment: {state.config.target_metric}_optimization")
            print()
            print("In LangSmith, you can:")
            print("  - View the full graph execution")
            print("  - See timing for each node")
            print("  - Inspect state at each step")
            print("  - Compare across experiments")
            print("  - Debug issues")

        print("\n✅ Experiment complete!\n")
        return 0

    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
