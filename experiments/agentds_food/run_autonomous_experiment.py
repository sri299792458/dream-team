#!/usr/bin/env python3
"""
Dream Team Framework: AUTONOMOUS Shelf Life Prediction Experiment

This script demonstrates the fully autonomous Dream Team framework where agents:
1. Receive problem statement + data
2. Plan their own approach
3. Write their own code
4. Execute and analyze results
5. Evolve when they plateau
6. Iterate until goal achieved

Usage:
    python run_autonomous_experiment.py

Requirements:
    - GEMINI_API_KEY environment variable set
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

    # Create initial team
    print("👥 Creating initial team...\n")

    pi = Agent(
        title="Principal Investigator",
        expertise="data science, machine learning, experimental design, research strategy",
        goal="achieve the lowest possible MAE on shelf life prediction",
        role="lead the team, coordinate strategy, make high-level decisions"
    )

    data_scientist = Agent(
        title="Data Scientist",
        expertise="Python, pandas, scikit-learn, feature engineering, EDA, statistical modeling",
        goal="design and implement effective predictive models",
        role="write code for data analysis, feature engineering, and model training"
    )

    print(f"  ✅ {pi.title}")
    print(f"  ✅ {data_scientist.title}\n")

    # Prepare problem statement
    problem_statement = """
Predict the remaining shelf life in days for food production batches.

Challenge: Shelf Life Prediction for Food Production
Metric: Mean Absolute Error (MAE) - lower is better
Goal: Minimize prediction error for remaining shelf life days

Data Available:
- batches_train: Training data with features and shelf_life_remaining_days target
- batches_test: Test data (features only, need to predict target)
- products: Product catalog with category, storage_class, base_shelf_life_days
- sites: Production site information with region_id, line_type
- regions: Regional data with seasonality_amp

Key Features:
- dwell_hours: Hours in storage
- mean_temp_F: Average temperature (Fahrenheit)
- mean_rh_pct: Average relative humidity percentage
- door_opens_count: Number of door openings (temperature abuse indicator)
- sku_id: Product SKU (join with products)
- site_id: Production site (join with sites)

Task:
Design and implement a solution to predict shelf_life_remaining_days for test batches.
Focus on:
1. Feature engineering based on storage physics
2. Model selection and training
3. Cross-validation for robust evaluation
4. Iterative improvement based on results
"""

    # Set up data context (what agents can access)
    data_context = {
        'batches_train': batches_train,
        'batches_test': batches_test,
        'products': products,
        'sites': sites,
        'regions': regions,
    }

    # Create orchestrator
    results_dir = Path(__file__).parent / 'results' / 'autonomous_shelf_life'

    orchestrator = ExperimentOrchestrator(
        team_lead=pi,
        team_members=[data_scientist],
        results_dir=results_dir
    )

    # Run autonomous experiment
    print("🚀 Starting autonomous experiment...\n")
    print("The agents will now:")
    print("  1. Plan their approach")
    print("  2. Write code")
    print("  3. Execute and analyze results")
    print("  4. Evolve when stuck")
    print("  5. Iterate until goal achieved\n")

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
