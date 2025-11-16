#!/usr/bin/env python3
"""
Dream Team Framework: AUTONOMOUS Shelf Life Prediction Experiment

This script demonstrates the fully autonomous Dream Team framework with bootstrap:
- PI starts alone, explores problem and data
- PI recruits team based on what they learned
- Strategy team discusses WHAT to implement
- Research Engineer translates discussions into executable code
- Autonomous iteration and evolution until goal achieved

Bootstrap Phase (Iteration 0):
1. PI receives problem statement + data
2. PI decides what initial exploration is needed
3. Research Engineer implements exploration code
4. PI reviews exploration results
5. PI recruits team members based on expertise needed

Main Iterations:
6. Full team discusses what to implement (not how to code)
7. Research Engineer translates discussion into code
8. Execute and analyze results
9. Evolve when stuck
10. Iterate until optimized

This mirrors real research: PI doesn't assemble a team blindly,
they first understand the problem, then recruit the right expertise.

Usage:
    export GEMINI_API_KEY=your_key_here
    export SEMANTIC_SCHOLAR_API_KEY=your_key_here
    python run_autonomous_experiment.py

Requirements:
    - GEMINI_API_KEY environment variable set
    - SEMANTIC_SCHOLAR_API_KEY environment variable set
    - FoodProduction data in data/FoodProduction/
"""

import os
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from dream_team import (
    Agent,
    ExperimentOrchestrator
)


def main():
    """Run fully autonomous experiment"""

    print("="*70)
    print("DREAM TEAM AUTONOMOUS EXPERIMENT - SHELF LIFE PREDICTION")
    print("="*70)
    print()

    # Check setup
    if not os.getenv('GEMINI_API_KEY'):
        print("❌ GEMINI_API_KEY not set!")
        print("   Get a free key at: https://aistudio.google.com")
        print("   Then run: export GEMINI_API_KEY=your_key_here")
        sys.exit(1)

    if not os.getenv('SEMANTIC_SCHOLAR_API_KEY'):
        print("❌ SEMANTIC_SCHOLAR_API_KEY not set!")
        print("   Get a free key at: https://www.semanticscholar.org/product/api")
        print("   Then run: export SEMANTIC_SCHOLAR_API_KEY=your_key_here")
        sys.exit(1)

    data_dir = Path(__file__).parent / 'data' / 'FoodProduction'
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("   Please place FoodProduction data in experiments/agentds_food/data/")
        sys.exit(1)

    print("✅ Setup verified\n")

    # Load data
    print("📊 Loading data...")
    batches_train = pd.read_csv(data_dir / 'batches_train.csv')
    batches_test = pd.read_csv(data_dir / 'batches_test.csv')
    products = pd.read_csv(data_dir / 'products.csv')
    sites = pd.read_csv(data_dir / 'sites.csv')
    regions = pd.read_csv(data_dir / 'regions.csv')

    print(f"  Training batches: {len(batches_train):,}")
    print(f"  Test batches: {len(batches_test):,}\n")

    # Create initial setup (PI starts alone)
    print("👥 Initial setup...\n")

    # Principal Investigator starts alone
    pi = Agent(
        title="Principal Investigator",
        expertise="machine learning strategy, experimental design, research methodology, research literature review",
        goal="optimize the target metric through systematic experimentation and research-driven decisions",
        role="explore the problem, research relevant literature, recruit expert team, coordinate research direction"
    )

    # Coding agent (translates all discussions into code)
    coding_agent = Agent(
        title="Research Engineer",
        expertise="Python, pandas, scikit-learn, numpy, data analysis, implementation, translating research into code",
        goal="implement research plans accurately and efficiently",
        role="translate team's research-backed strategies into executable, production-quality code"
    )

    print(f"  Starting with:")
    print(f"    ✅ {pi.title} (will explore and recruit team)")
    print(f"    ✅ {coding_agent.title} (implementation)")
    print(f"\n  (Team members will be recruited during bootstrap)\n")

    # Prepare problem statement
    problem_statement = """
Predict the remaining shelf life in days for food production batches.

Target Variable: shelf_life_remaining_days (continuous)
Evaluation Metric: Mean Absolute Error (MAE) - lower is better

Data Available:
- batches_train: Training data with target variable
- batches_test: Test data (predict target)
- products: Product reference data
- sites: Production site reference data
- regions: Regional reference data

Your Goal:
Build the best possible predictive solution. Explore the data, decide your approach,
and iteratively improve your predictions to minimize MAE.
"""

    # Set up data context (what agents can access)
    data_context = {
        'batches_train': batches_train,
        'batches_test': batches_test,
        'products': products,
        'sites': sites,
        'regions': regions,
    }

    # Create orchestrator (team_members empty - PI will recruit during bootstrap)
    results_dir = Path(__file__).parent / 'results' / 'autonomous_shelf_life'

    orchestrator = ExperimentOrchestrator(
        team_lead=pi,
        team_members=[],  # Empty - PI will recruit after exploring
        coding_agent=coding_agent,
        results_dir=results_dir
    )

    # Run autonomous experiment
    print("🚀 Starting autonomous experiment...\n")
    print("The workflow:")
    print("  Bootstrap:")
    print("    1. PI explores problem and data alone")
    print("    2. PI recruits team based on what they learned")
    print("  Main iterations:")
    print("    3. Strategy team discusses what to implement")
    print("    4. Research Engineer translates discussion into code")
    print("    5. Execute and analyze results")
    print("    6. Evolve when stuck")
    print("    7. Iterate until goal achieved\n")

    final_results = orchestrator.run(
        problem_statement=problem_statement,
        data_context=data_context,
        target_metric='mae',
        minimize_metric=True,
        max_iterations=5,
        target_score=None  # No specific target, just minimize
    )

    # Print summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"\nIterations completed: {final_results['total_iterations']}")
    print(f"Best MAE achieved: {final_results['best_metric']:.4f}")

    print(f"\nFinal Team:")
    for agent_info in final_results['final_team']:
        print(f"  - {agent_info['title']}")
        print(f"    Expertise: {agent_info['expertise'][:80]}...")
        print(f"    Specialization: {agent_info['specialization_depth']}")

    print(f"\nAll results saved to: {results_dir}")
    print("\nFiles created:")
    print(f"  - Iteration logs: {results_dir}/iteration_*.json")
    print(f"  - Generated code: {results_dir}/code/")
    print(f"  - Meeting transcripts: {results_dir}/meetings/")
    print(f"  - Agent snapshots: {results_dir}/agents/")
    print(f"  - Final summary: {results_dir}/final_summary.json")

    print("\n" + "="*70)
    print("✅ AUTONOMOUS EXPERIMENT COMPLETE")
    print("="*70)
    print()


if __name__ == '__main__':
    main()
