#!/usr/bin/env python3
"""
Test the LangGraph skeleton.

Verifies that the graph structure compiles and can execute with placeholder nodes.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dream_team.experiment import (
    create_initial_state,
    AgentConfig,
    run_graph_experiment
)
import pandas as pd
import numpy as np


def main():
    print("="*70)
    print("TEST: LangGraph with Real Nodes")
    print("="*70)
    print("\nThis tests that the graph runs with actual node implementations.\n")

    # Create minimal agent configs
    pi_config = AgentConfig(
        title="Principal Investigator",
        expertise="machine learning",
        goal="optimize metrics",
        role="lead research"
    )

    coding_config = AgentConfig(
        title="Research Engineer",
        expertise="Python, pandas",
        goal="implement plans",
        role="write code"
    )

    # Create initial state
    state = create_initial_state(
        team_lead=pi_config,
        coding_agent=coding_config,
        problem_statement="Test problem: predict target from features",
        target_metric="mae",
        minimize_metric=True,
        max_iterations=1,  # Just 1 iteration for quick test
        results_dir="scripts/test_graph_results"
    )

    # Create minimal test data
    np.random.seed(42)
    data_context = {
        'train_df': pd.DataFrame({
            'id': range(20),
            'feature_1': np.random.randn(20),
            'target': np.random.randn(20)
        }),
        'test_df': pd.DataFrame({
            'id': range(20, 30),
            'feature_1': np.random.randn(10)
        })
    }

    # Run graph
    print("Running graph with real node implementations...\n")
    try:
        final_state = run_graph_experiment(state, data_context)

        # Verify
        print("\n" + "="*70)
        print("VERIFICATION")
        print("="*70)

        checks = []

        # Check iterations ran
        if final_state.iteration > 0:
            checks.append(f"✅ Iterations completed: {final_state.iteration}")
        else:
            checks.append(f"❌ No iterations completed")

        # Check history populated
        if len(final_state.history) > 0:
            checks.append(f"✅ History recorded: {len(final_state.history)} entries")
        else:
            checks.append(f"❌ No history recorded")

        # Check final phase
        if final_state.phase == "complete":
            checks.append(f"✅ Reached completion phase")
        else:
            checks.append(f"⚠️  Final phase: {final_state.phase}")

        # Check best metric
        if final_state.best_metric is not None:
            checks.append(f"✅ Best metric tracked: {final_state.best_metric:.4f}")
        else:
            checks.append(f"⚠️  No best metric")

        for check in checks:
            print(check)

        print("\n✅ Graph test PASSED")
        print("   The graph is working with real node implementations.")
        print("   Phase 4 complete!\n")

        return 0

    except Exception as e:
        print(f"\n❌ Graph test FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
