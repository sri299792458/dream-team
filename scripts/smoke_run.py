#!/usr/bin/env python3
"""
Smoke Test for Dream Team Framework

Runs a minimal experiment to verify the current orchestration works.
This establishes a baseline before the LangGraph refactor.

Usage:
    export GEMINI_API_KEY=your_key_here
    export SEMANTIC_SCHOLAR_API_KEY=your_key_here
    python scripts/smoke_run.py

Expected Output:
    - Bootstrap phase completes
    - 1-2 iterations complete
    - Metrics are computed
    - Results saved to scripts/smoke_results/

This script should work identically before and after the refactor.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dream_team import Agent, ExperimentOrchestrator


def create_minimal_dataset():
    """Create a tiny synthetic dataset for quick testing"""
    # Minimal regression problem: predict age from features
    np.random.seed(42)

    train_df = pd.DataFrame({
        'id': range(50),
        'feature_1': np.random.randn(50),
        'feature_2': np.random.randn(50),
        'target': np.random.randn(50) * 10 + 50
    })

    test_df = pd.DataFrame({
        'id': range(50, 70),
        'feature_1': np.random.randn(20),
        'feature_2': np.random.randn(20),
    })

    return train_df, test_df


def main():
    print("="*70)
    print("SMOKE TEST - Dream Team Framework")
    print("="*70)
    print("\nThis runs a minimal experiment to verify current behavior.")
    print("Results should be identical before and after LangGraph refactor.\n")

    # Check environment
    if not os.getenv('GEMINI_API_KEY'):
        print("⚠️  Warning: GEMINI_API_KEY not set")
        print("   The smoke test may fail without API access")

    # Create minimal data
    print("📊 Creating minimal synthetic dataset...")
    train_df, test_df = create_minimal_dataset()
    print(f"   Train: {len(train_df)} rows")
    print(f"   Test: {len(test_df)} rows\n")

    # Create agents
    print("👥 Creating minimal team...")
    pi = Agent(
        title="Principal Investigator",
        expertise="machine learning, experimental design",
        goal="optimize the target metric",
        role="coordinate research and recruit team"
    )

    coding_agent = Agent(
        title="Research Engineer",
        expertise="Python, pandas, scikit-learn",
        goal="implement research plans",
        role="translate strategies into code"
    )

    print(f"   ✅ {pi.title}")
    print(f"   ✅ {coding_agent.title}\n")

    # Problem statement
    problem_statement = """
Simple regression problem: predict 'target' from features.

Target Variable: target (continuous)
Evaluation Metric: MAE (Mean Absolute Error) - lower is better

Data:
- train_df: Training data with target
- test_df: Test data (predict target)

Goal: Build a simple predictive model to minimize MAE.
"""

    # Data context
    data_context = {
        'train_df': train_df,
        'test_df': test_df,
    }

    # Create orchestrator
    results_dir = Path(__file__).parent / 'smoke_results'
    orchestrator = ExperimentOrchestrator(
        team_lead=pi,
        team_members=[],  # PI will recruit during bootstrap
        coding_agent=coding_agent,
        results_dir=results_dir
    )

    # Run minimal experiment
    print("🚀 Running smoke test experiment...\n")
    print("Expected phases:")
    print("  1. Bootstrap: PI explores and recruits")
    print("  2. Iteration 1: Team plans, codes, executes")
    print("  3. Save results\n")

    try:
        final_results = orchestrator.run(
            problem_statement=problem_statement,
            data_context=data_context,
            target_metric='mae',
            minimize_metric=True,
            max_iterations=2,  # Just 2 iterations for smoke test
            target_score=None,
            resume=False  # Always start fresh for smoke test
        )

        # Verify results
        print("\n" + "="*70)
        print("SMOKE TEST VERIFICATION")
        print("="*70)

        checks_passed = 0
        checks_total = 5

        # Check 1: Iterations completed
        if final_results['total_iterations'] >= 1:
            print("✅ Check 1: At least 1 iteration completed")
            checks_passed += 1
        else:
            print("❌ Check 1: No iterations completed")

        # Check 2: Bootstrap file exists
        bootstrap_file = results_dir / 'iteration_00_bootstrap.json'
        if bootstrap_file.exists():
            print("✅ Check 2: Bootstrap file created")
            checks_passed += 1
        else:
            print("❌ Check 2: Bootstrap file missing")

        # Check 3: Code generated
        code_dir = results_dir / 'code'
        if code_dir.exists() and list(code_dir.glob('*.py')):
            print("✅ Check 3: Code files generated")
            checks_passed += 1
        else:
            print("❌ Check 3: No code files")

        # Check 4: Team assembled
        if len(final_results['final_team']) > 1:
            print(f"✅ Check 4: Team assembled ({len(final_results['final_team'])} members)")
            checks_passed += 1
        else:
            print("❌ Check 4: Team not assembled")

        # Check 5: Metrics computed
        if final_results.get('best_metric') is not None:
            print(f"✅ Check 5: Metrics computed (best MAE: {final_results['best_metric']:.4f})")
            checks_passed += 1
        else:
            print("❌ Check 5: No metrics computed")

        print(f"\n{'='*70}")
        print(f"SMOKE TEST RESULT: {checks_passed}/{checks_total} checks passed")
        print(f"{'='*70}\n")

        if checks_passed == checks_total:
            print("✅ SMOKE TEST PASSED")
            print(f"\nResults saved to: {results_dir}")
            return 0
        else:
            print("⚠️  SMOKE TEST INCOMPLETE")
            print(f"\nResults saved to: {results_dir}")
            return 1

    except Exception as e:
        print(f"\n❌ SMOKE TEST FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
