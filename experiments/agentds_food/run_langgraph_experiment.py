"""
LangGraph-based Dream Team experiment runner.

This script demonstrates the LangGraph implementation which provides:
- Proper state management with TypedDict schemas
- Automatic checkpointing and resumability
- Better context handling (no manual truncation)
- Cleaner workflow orchestration
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import pandas as pd
from dream_team.agent import Agent
from dream_team.langgraph_state import serialize_agent, DreamTeamState
from dream_team.langgraph_orchestrator import create_dream_team_graph
from dream_team.executor import CodeExecutor


def main():
    """Run LangGraph-based autonomous experiment"""

    print("=" * 80)
    print("DREAM TEAM - LANGGRAPH IMPLEMENTATION")
    print("=" * 80)
    print()

    # ============================================================================
    # CONFIGURATION
    # ============================================================================

    # Problem statement
    problem_statement = """
Predict the shelf life (in days) of food products based on their properties.

Dataset: FoodProduction with columns describing food characteristics
Target: shelf_life (days until product expires)
Metric: Mean Absolute Error (MAE) - lower is better

Challenge: Need to handle different food types, storage conditions, and
compositional factors that affect shelf life.
"""

    # Paths
    data_dir = Path(__file__).parent / 'data' / 'FoodProduction'
    results_dir = Path(__file__).parent / 'results' / 'langgraph_shelf_life'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("📂 Loading data...")
    batches_train = pd.read_csv(data_dir / 'batches_train.csv')
    batches_test = pd.read_csv(data_dir / 'batches_test.csv')
    products = pd.read_csv(data_dir / 'products.csv')
    sites = pd.read_csv(data_dir / 'sites.csv')
    regions = pd.read_csv(data_dir / 'regions.csv')

    print(f"   batches_train: {batches_train.shape}")
    print(f"   batches_test: {batches_test.shape}")
    print(f"   products: {products.shape}")
    print(f"   sites: {sites.shape}")
    print(f"   regions: {regions.shape}\n")

    # ============================================================================
    # INITIALIZE STATE
    # ============================================================================

    print("🤖 Initializing agents...")

    # Create initial agents
    team_lead = Agent(
        title="Principal Investigator",
        expertise="research methodology, experimental design, data science strategy",
        goal="lead the team to optimize shelf life predictions through rigorous experimentation",
        role="coordinate team, synthesize insights, make strategic decisions"
    )

    coding_agent = Agent(
        title="Research Engineer",
        expertise="Python, pandas, scikit-learn, PyTorch, data pipelines",
        goal="implement robust, efficient code for data analysis and modeling",
        role="translate research ideas into executable code, debug issues"
    )

    print(f"   Team Lead: {team_lead.title}")
    print(f"   Coding Agent: {coding_agent.title}")
    print(f"   (Team members will be recruited during bootstrap)\n")

    # Create initial state
    initial_state: DreamTeamState = {
        # Problem definition
        "problem_statement": problem_statement,
        "target_metric": "mae",
        "minimize_metric": True,
        "target_score": 30.0,  # Target MAE of 30 days or better

        # Data context
        "data_context": {
            "batches_train": batches_train,
            "batches_test": batches_test,
            "products": products,
            "sites": sites,
            "regions": regions,
        },

        # Team (serialized)
        "team_lead": serialize_agent(team_lead),
        "team_members": [],  # Will be populated during bootstrap
        "coding_agent": serialize_agent(coding_agent),

        # Iteration state
        "iteration": 0,
        "max_iterations": 5,
        "bootstrap_completed": False,

        # History
        "experiment_history": [],

        # Control flags
        "goal_achieved": False,
        "should_evolve": False,
        "error_count": 0,

        # Directories
        "results_dir": str(results_dir),
        "meetings_dir": str(results_dir / "meetings"),
        "code_dir": str(results_dir / "code"),
    }

    # ============================================================================
    # RUN GRAPH
    # ============================================================================

    print("🚀 Starting LangGraph execution...")
    print(f"   Max iterations: {initial_state['max_iterations']}")
    print(f"   Target: {initial_state['target_metric']} <= {initial_state.get('target_score', 'N/A')}")
    print(f"   Results: {results_dir}\n")

    # Create graph
    graph = create_dream_team_graph()

    # Run graph
    try:
        # Configuration for running
        config = {
            "configurable": {
                "thread_id": "food_shelf_life_experiment"
            }
        }

        # Execute graph
        final_state = None
        for state_update in graph.stream(initial_state, config):
            # Stream returns dict of {node_name: updated_state}
            for node_name, updated_state in state_update.items():
                final_state = updated_state
                # State is automatically checkpointed at each node

        # ========================================================================
        # FINAL SUMMARY
        # ========================================================================

        print("\n" + "=" * 80)
        print("✅ EXPERIMENT COMPLETE")
        print("=" * 80)

        if final_state:
            print(f"\nTotal Iterations: {final_state.get('iteration', 0)}")
            print(f"Best {final_state['target_metric']}: {final_state.get('best_metric', 'N/A')}")
            print(f"Goal Achieved: {final_state.get('goal_achieved', False)}")
            print(f"\nResults saved to: {results_dir}")

            # Save final summary
            summary = {
                "problem": problem_statement,
                "total_iterations": final_state.get('iteration', 0),
                "best_metric": final_state.get('best_metric'),
                "best_iteration": final_state.get('best_iteration'),
                "goal_achieved": final_state.get('goal_achieved', False),
                "team_composition": {
                    "lead": final_state['team_lead']['title'],
                    "members": [m['title'] for m in final_state.get('team_members', [])],
                    "coding": final_state['coding_agent']['title']
                }
            }

            import json
            with open(results_dir / 'final_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)

        print()

    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        print("State is checkpointed - you can resume later")

    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

        print("\nState is checkpointed at last successful node")


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Error: GEMINI_API_KEY environment variable not set")
        print("Please set it: export GEMINI_API_KEY='your-key-here'")
        sys.exit(1)

    main()
