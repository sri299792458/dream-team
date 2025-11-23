"""
LangGraph V2 Dream Team experiment runner - Enhanced version.

New features:
- ReAct agents with proper tool use
- Smart context management (no truncation)
- Multi-agent team meeting subgraph
- Mathematical state (K, θ, δ) in prompts
- Streaming support for real-time progress
- Better error diagnostics
- LangSmith tracing (optional)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import pandas as pd
from dream_team.agent import Agent
from dream_team.langgraph_state import serialize_agent, DreamTeamState
from dream_team.langgraph_orchestrator_v2 import create_dream_team_graph_v2
from dream_team.executor import CodeExecutor
from dream_team.serialization import make_msgpack_safe
from dream_team.langgraph_tools import set_executor_context


def setup_langsmith_tracing():
    """
    Optional: Setup LangSmith tracing for observability.

    To use:
    1. export LANGSMITH_API_KEY='your-key'
    2. export LANGSMITH_TRACING=true
    """
    if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
        if not os.getenv("LANGSMITH_API_KEY"):
            print("⚠️ LANGSMITH_TRACING=true but LANGSMITH_API_KEY not set")
            print("   Tracing disabled. Set LANGSMITH_API_KEY to enable.")
            return False

        # Set project name
        os.environ["LANGCHAIN_PROJECT"] = "dream-team-v2"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"

        print("✅ LangSmith tracing enabled")
        print(f"   Project: {os.environ['LANGCHAIN_PROJECT']}")
        print(f"   View at: https://smith.langchain.com/\n")
        return True

    return False


def stream_graph_progress(graph, initial_state, config):
    """
    Stream graph execution with real-time progress updates.

    Args:
        graph: Compiled LangGraph
        initial_state: Initial state
        config: Graph config

    Returns:
        Final state
    """
    print("🔄 Streaming graph execution...\n")

    final_state = None

    # Stream node updates
    for event in graph.stream(initial_state, config, stream_mode="updates"):
        for node_name, updated_state in event.items():
            # Show progress
            if node_name == "bootstrap":
                print(f"📍 Completed: Bootstrap")
            elif node_name == "team_planning":
                iter_num = updated_state.get('iteration', '?')
                print(f"📍 Completed: Team Planning (Iteration {iter_num})")
            elif node_name == "code_generation":
                print(f"📍 Completed: Code Generation")
            elif node_name == "execution":
                metrics = updated_state.get('current_metrics', {})
                if metrics:
                    print(f"📍 Completed: Execution → {metrics}")
                else:
                    print(f"📍 Completed: Execution")
            elif node_name == "evolution":
                print(f"📍 Completed: Evolution")

            final_state = updated_state

    return final_state


def main():
    """Run enhanced LangGraph experiment"""

    print("=" * 80)
    print("DREAM TEAM V2 - ENHANCED LANGGRAPH IMPLEMENTATION")
    print("=" * 80)
    print("\nEnhancements:")
    print("✓ ReAct agents with create_react_agent()")
    print("✓ Smart context management (semantic summarization)")
    print("✓ Multi-agent team meeting subgraph")
    print("✓ Mathematical state (K, θ, δ) integrated in prompts")
    print("✓ Streaming support")
    print("✓ Better error diagnostics")
    print()

    # Setup tracing
    tracing_enabled = setup_langsmith_tracing()

    # ============================================================================
    # CONFIGURATION
    # ============================================================================

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
    results_dir = Path(__file__).parent / 'results' / 'langgraph_v2_shelf_life'
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

    print("🤖 Initializing agents with mathematical state...")

    team_lead = Agent(
        title="Principal Investigator",
        expertise="research methodology, experimental design, data science strategy, statistical analysis",
        goal="lead the team to optimize shelf life predictions through rigorous experimentation",
        role="coordinate team, synthesize insights, make strategic decisions"
    )

    coding_agent = Agent(
        title="Research Engineer",
        expertise="Python, pandas, scikit-learn, PyTorch, data pipelines, debugging",
        goal="implement robust, efficient code for data analysis and modeling",
        role="translate research ideas into executable code, debug issues, optimize implementations"
    )

    print(f"   Team Lead: {team_lead.title}")
    print(f"   Coding Agent: {coding_agent.title}")
    print(f"   Mathematical state: K={len(team_lead.K.concepts)} concepts, "
          f"θ gini={team_lead.δ.gini_coefficient():.2f}")
    print(f"   (Team members will be recruited during bootstrap)\n")

    # Create initial state
    initial_state: DreamTeamState = {
        "problem_statement": problem_statement,
        "target_metric": "mae",
        "minimize_metric": True,
        "target_score": 30.0,

        "data_context": {
            "batches_train": batches_train,
            "batches_test": batches_test,
            "products": products,
            "sites": sites,
            "regions": regions,
        },

        "team_lead": serialize_agent(team_lead),
        "team_members": [],
        "coding_agent": serialize_agent(coding_agent),

        "iteration": 0,
        "max_iterations": 5,
        "bootstrap_completed": False,

        "experiment_history": [],

        "goal_achieved": False,
        "should_evolve": False,
        "error_count": 0,

        "results_dir": str(results_dir),
        "meetings_dir": str(results_dir / "meetings"),
        "code_dir": str(results_dir / "code"),
    }

    # ============================================================================
    # RUN GRAPH WITH STREAMING
    # ============================================================================

    # Set executor context with actual DataFrames BEFORE running the graph
    # This allows the bootstrap node to access the data through set_executor_context
    set_executor_context(initial_state["data_context"])

    # Make initial state msgpack-safe for checkpointing (converts DataFrames to metadata)
    # The actual DataFrames are accessible through the executor context
    initial_state = make_msgpack_safe(initial_state)

    print("🚀 Starting enhanced LangGraph execution...")
    print(f"   Max iterations: {initial_state['max_iterations']}")
    print(f"   Target: {initial_state['target_metric']} <= {initial_state.get('target_score', 'N/A')}")
    print(f"   Results: {results_dir}")
    print(f"   Streaming: enabled\n")

    # Create graph
    graph = create_dream_team_graph_v2()

    # Run with streaming
    try:
        config = {
            "configurable": {
                "thread_id": "food_shelf_life_v2"
            }
        }

        # Option 1: Stream with progress updates
        use_streaming = True

        if use_streaming:
            final_state = stream_graph_progress(graph, initial_state, config)
        else:
            # Option 2: Regular invoke
            final_state = graph.invoke(initial_state, config)

        # ========================================================================
        # FINAL SUMMARY
        # ========================================================================

        print("\n" + "=" * 80)
        print("✅ EXPERIMENT COMPLETE")
        print("=" * 80)

        if final_state:
            print(f"\nTotal Iterations: {final_state.get('iteration', 0)}")
            print(f"Best {final_state['target_metric']}: {final_state.get('best_metric', 'N/A')}")
            if final_state.get('best_metric'):
                print(f"Best Iteration: {final_state.get('best_iteration', 'N/A')}")
            print(f"Goal Achieved: {final_state.get('goal_achieved', False)}")

            # Team composition
            print(f"\nFinal Team:")
            print(f"  Lead: {final_state['team_lead']['title']}")
            print(f"  Members: {[m['title'] for m in final_state.get('team_members', [])]}")
            print(f"  Coding: {final_state['coding_agent']['title']}")

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
                },
                "enhancements_used": [
                    "ReAct agents with create_react_agent()",
                    "Smart context management",
                    "Multi-agent team meetings",
                    "Mathematical state integration",
                    "Streaming execution",
                    "Enhanced error diagnostics"
                ]
            }

            import json
            with open(results_dir / 'final_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)

            if tracing_enabled:
                print("\n📊 View execution trace at: https://smith.langchain.com/")

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
