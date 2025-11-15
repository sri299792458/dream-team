#!/usr/bin/env python3
"""
Dream Team Framework: Shelf Life Prediction Experiment

This script demonstrates the Dream Team framework on the AgentDS Food Production
benchmark's shelf life prediction challenge.

Usage:
    python run_shelf_life_experiment.py

Requirements:
    - GEMINI_API_KEY environment variable set
    - FoodProduction data in data/FoodProduction/
    - dream-team package installed
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from dream_team import (
    Agent,
    TeamMeeting,
    IndividualMeeting,
    EvolutionEngine,
    get_research_assistant,
    save_json
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


def check_setup():
    """Verify environment is properly configured."""
    print("🔍 Checking setup...\n")

    # Check API key
    if not os.getenv('GEMINI_API_KEY'):
        print("❌ GEMINI_API_KEY not set!")
        print("   Get a free key at: https://aistudio.google.com")
        print("   Then run: export GEMINI_API_KEY=your_key_here")
        sys.exit(1)

    # Check data directory
    data_dir = Path(__file__).parent / 'data' / 'FoodProduction'
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("   Please place FoodProduction data in experiments/agentds_food/data/")
        sys.exit(1)

    # Check required files
    required_files = ['batches_train.csv', 'batches_test.csv', 'products.csv',
                     'sites.csv', 'regions.csv']
    missing_files = [f for f in required_files if not (data_dir / f).exists()]
    if missing_files:
        print(f"❌ Missing data files: {missing_files}")
        sys.exit(1)

    print("✅ Setup verified!\n")
    return data_dir


def load_data(data_dir):
    """Load all required data."""
    print("📊 Loading data...\n")

    batches_train = pd.read_csv(data_dir / 'batches_train.csv')
    batches_test = pd.read_csv(data_dir / 'batches_test.csv')
    products = pd.read_csv(data_dir / 'products.csv')
    sites = pd.read_csv(data_dir / 'sites.csv')
    regions = pd.read_csv(data_dir / 'regions.csv')

    print(f"  Training batches: {len(batches_train):,}")
    print(f"  Test batches: {len(batches_test):,}")
    print(f"  Products: {len(products):,}")
    print(f"  Sites: {len(sites):,}")
    print(f"  Regions: {len(regions):,}\n")

    return batches_train, batches_test, products, sites, regions


def create_initial_team(results_dir):
    """Create initial team of agents."""
    print("👥 Creating initial team...\n")

    pi = Agent(
        title="Principal Investigator",
        expertise="data science, machine learning, research strategy, experimental design",
        goal="solve the shelf life prediction challenge with high accuracy (low MAE)",
        role="lead the team, make strategic decisions, coordinate research efforts"
    )

    data_scientist = Agent(
        title="Data Scientist",
        expertise="exploratory data analysis, feature engineering, statistical modeling, Python",
        goal="understand the data deeply and create informative features",
        role="analyze data patterns, engineer features, propose modeling approaches"
    )

    ml_engineer = Agent(
        title="ML Engineer",
        expertise="scikit-learn, model training, hyperparameter tuning, cross-validation",
        goal="build and optimize prediction models",
        role="implement models, tune hyperparameters, evaluate performance"
    )

    # Save initial agents
    agents_dir = results_dir / 'agents'
    agents_dir.mkdir(parents=True, exist_ok=True)

    pi.save(agents_dir / 'pi_initial.json')
    data_scientist.save(agents_dir / 'data_scientist_initial.json')
    ml_engineer.save(agents_dir / 'ml_engineer_initial.json')

    print(f"  ✅ {pi.title}: {pi.role}")
    print(f"  ✅ {data_scientist.title}: {data_scientist.role}")
    print(f"  ✅ {ml_engineer.title}: {ml_engineer.role}\n")

    return pi, data_scientist, ml_engineer


def run_initial_meeting(pi, data_scientist, ml_engineer, batches_train, results_dir):
    """Run initial team meeting to discuss the problem."""
    print("🏁 Running initial team meeting...\n")

    problem_context = f"""
Challenge: Shelf Life Prediction for Food Production Batches

Data Overview:
- Training samples: {len(batches_train):,}
- Target: shelf_life_remaining_days (continuous)
- Metric: MAE (Mean Absolute Error)

Available Features:
- dwell_hours, mean_temp_F, mean_rh_pct, door_opens_count
- sku_id (joinable with products), site_id (joinable with sites)

Target Statistics:
{batches_train['shelf_life_remaining_days'].describe()}

Key Challenges:
1. Food perishability - limited shelf life
2. Storage conditions impact (temperature, humidity)
3. Door opening patterns (temperature abuse)
4. Different product categories and storage classes
"""

    agenda = f"""
Analyze the shelf life prediction challenge and develop an initial modeling strategy.

Discussion Points:
1. What are the key factors affecting shelf life?
2. What features should we engineer?
3. What modeling approaches should we try?

Context:
{problem_context}
"""

    meeting = TeamMeeting(save_dir=str(results_dir / 'meetings'))
    summary = meeting.run(
        team_lead=pi,
        team_members=[data_scientist, ml_engineer],
        agenda=agenda,
        num_rounds=2
    )

    print("✅ Meeting completed!\n")
    return summary


def research_papers(data_scientist, results_dir):
    """Research relevant papers on shelf life prediction."""
    print("🔬 Researching shelf life prediction literature...\n")

    research = get_research_assistant()

    papers = research.research_topic(
        query="shelf life prediction food storage temperature humidity",
        context="""Predicting remaining shelf life for food products based on storage conditions.
        Key factors: temperature, humidity, door openings, product category.
        Goal: Minimize prediction error (MAE) for days remaining.""",
        num_papers=3  # Fewer papers for speed
    )

    print(f"✅ Found {len(papers)} relevant papers\n")

    for i, paper in enumerate(papers, 1):
        print(f"{i}. {paper.title}")
        print(f"   Relevance: {paper.relevance_score}/10\n")

    return papers


def evolve_agent(data_scientist, papers, results_dir):
    """Evolve data scientist with research insights."""
    print("🧬 Evolving Data Scientist with research insights...\n")

    evolution_engine = EvolutionEngine()

    evolution_context = {
        "problem_description": "Shelf life prediction for food products",
        "current_challenges": [
            "Understanding temperature abuse impact on shelf life",
            "Modeling humidity effects on perishable goods",
            "Incorporating product-specific deterioration rates"
        ],
        "performance_metrics": "Need to minimize MAE on shelf life predictions"
    }

    evolved = evolution_engine.evolve_agent(
        agent=data_scientist,
        context=evolution_context,
        papers=papers,
        trigger_reason="Need domain expertise in food science and shelf life modeling"
    )

    if evolved:
        print("✅ Agent evolved successfully!")
        print(f"\nNew Title: {data_scientist.title}")
        print(f"New Expertise: {data_scientist.expertise[:100]}...")
        print(f"Knowledge Base: {len(data_scientist.knowledge_base.papers)} papers\n")

        data_scientist.save(results_dir / 'agents' / 'data_scientist_evolved_v1.json')

    return data_scientist


def create_features(batches_df, products_df, sites_df, regions_df):
    """Create features for shelf life prediction."""
    df = batches_df.copy()

    # Join reference data
    df = df.merge(products_df, on='sku_id', how='left')
    df = df.merge(sites_df, on='site_id', how='left')
    df = df.merge(regions_df, on='region_id', how='left')

    # Basic features
    features = ['dwell_hours', 'mean_temp_F', 'mean_rh_pct',
                'door_opens_count', 'base_shelf_life_days']

    # Engineered features
    df['temp_abuse'] = np.maximum(0, df['mean_temp_F'] - 40)
    df['thermal_load'] = df['temp_abuse'] * df['dwell_hours']
    df['humidity_stress'] = np.abs(df['mean_rh_pct'] - 85)
    df['door_opens_per_hour'] = df['door_opens_count'] / (df['dwell_hours'] + 1)
    df['shelf_life_consumption_rate'] = df['dwell_hours'] / (df['base_shelf_life_days'] * 24 + 1)
    df['temp_humidity_interaction'] = df['mean_temp_F'] * df['mean_rh_pct']

    features.extend(['temp_abuse', 'thermal_load', 'humidity_stress',
                    'door_opens_per_hour', 'shelf_life_consumption_rate',
                    'temp_humidity_interaction', 'seasonality_amp'])

    # Categorical features
    df = pd.get_dummies(df, columns=['category', 'storage_class', 'line_type'],
                       prefix=['cat', 'storage', 'line'])
    cat_features = [col for col in df.columns if col.startswith(('cat_', 'storage_', 'line_'))]
    features.extend(cat_features)

    return df[features].fillna(0)


def train_model(batches_train, products, sites, regions, results_dir):
    """Train baseline model."""
    print("🏋️ Training baseline Random Forest model...\n")

    X_train = create_features(batches_train, products, sites, regions)
    y_train = batches_train['shelf_life_remaining_days']

    print(f"Features: {X_train.shape[1]}, Samples: {X_train.shape[0]:,}")

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    # Cross-validation
    cv_scores = -cross_val_score(model, X_train, y_train, cv=5,
                                 scoring='neg_mean_absolute_error')

    print(f"\n✅ Cross-validation MAE: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Train on full data
    model.fit(X_train, y_train)

    # Save results
    baseline_results = {
        'model_type': 'RandomForest',
        'cv_mae_mean': float(cv_scores.mean()),
        'cv_mae_std': float(cv_scores.std()),
        'cv_scores': cv_scores.tolist(),
        'n_features': X_train.shape[1],
        'timestamp': datetime.now().isoformat()
    }

    save_json(baseline_results, results_dir / 'baseline_results.json')

    return model, X_train.columns


def generate_predictions(model, feature_cols, batches_test, products, sites, regions, results_dir):
    """Generate test predictions."""
    print("\n📝 Generating test predictions...\n")

    X_test = create_features(batches_test, products, sites, regions)

    # Ensure same columns
    missing_cols = set(feature_cols) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0
    X_test = X_test[feature_cols]

    predictions = model.predict(X_test)

    submission = pd.DataFrame({
        'batch_id': batches_test['batch_id'],
        'shelf_life_remaining_days': predictions
    })

    submission.to_csv(results_dir / 'predictions_baseline.csv', index=False)

    print(f"✅ Generated {len(predictions):,} predictions")
    print(f"💾 Saved to: {results_dir / 'predictions_baseline.csv'}\n")


def main():
    """Run the complete experiment."""
    print("="*60)
    print("DREAM TEAM FRAMEWORK - SHELF LIFE PREDICTION")
    print("="*60)
    print()

    # Setup
    data_dir = check_setup()
    results_dir = Path(__file__).parent / 'results' / 'shelf_life'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    batches_train, batches_test, products, sites, regions = load_data(data_dir)

    # Create team
    pi, data_scientist, ml_engineer = create_initial_team(results_dir)

    # Run meeting
    meeting_summary = run_initial_meeting(pi, data_scientist, ml_engineer,
                                         batches_train, results_dir)

    # Research papers
    papers = research_papers(data_scientist, results_dir)

    # Evolve agent
    data_scientist = evolve_agent(data_scientist, papers, results_dir)

    # Train model
    model, feature_cols = train_model(batches_train, products, sites, regions, results_dir)

    # Generate predictions
    generate_predictions(model, feature_cols, batches_test, products, sites,
                        regions, results_dir)

    print("="*60)
    print("✅ EXPERIMENT COMPLETE!")
    print("="*60)
    print(f"\n📁 Results saved to: {results_dir}")
    print("\nNext steps:")
    print("  1. Review results in results/shelf_life/")
    print("  2. Examine meeting transcripts for agent insights")
    print("  3. Check evolved agent knowledge base")
    print("  4. Iterate to improve performance\n")


if __name__ == '__main__':
    main()
