"""
LangGraph V2 Dream Team experiment runner with RESUME support.

New features over basic V2:
- Persistent checkpoints with SqliteSaver
- Automatic resume detection
- Interactive resume prompts
- Checkpoint management utilities
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import pandas as pd
from dream_team.agent import Agent
from dream_team.langgraph_state import serialize_agent, DreamTeamState
from dream_team.langgraph_orchestrator_v2 import create_dream_team_graph_v2
from dream_team.checkpoint_manager import CheckpointManager
from dream_team.executor import CodeExecutor


def setup_langsmith_tracing():
    """Setup LangSmith tracing (optional)"""
    if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
        if not os.getenv("LANGSMITH_API_KEY"):
            print("⚠️  LANGSMITH_TRACING=true but LANGSMITH_API_KEY not set")
            print("   Tracing disabled. Set LANGSMITH_API_KEY to enable.")
            return False

        os.environ["LANGCHAIN_PROJECT"] = "dream-team-v2"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"

        print("✅ LangSmith tracing enabled")
        print(f"   Project: {os.environ['LANGCHAIN_PROJECT']}")
        print(f"   View at: https://smith.langchain.com/\n")
        return True

    return False


def stream_graph_progress(graph, initial_state, config):
    """Stream graph execution with real-time progress"""
    print("🔄 Streaming graph execution...\n")

    final_state = None

    for event in graph.stream(initial_state, config, stream_mode="updates"):
        for node_name, updated_state in event.items():
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
    """Run enhanced LangGraph experiment with resume support"""

    print("=" * 80)
    print("DREAM TEAM V2 - WITH RESUME SUPPORT")
    print("=" * 80)
    print("\nEnhancements:")
    print("✓ ReAct agents with create_react_agent()")
    print("✓ Smart context management (semantic summarization)")
    print("✓ Multi-agent team meeting subgraph")
    print("✓ Mathematical state (K, θ, δ) integrated in prompts")
    print("✓ Streaming support")
    print("✓ Better error diagnostics")
    print("✓ Persistent checkpoints with SqliteSaver")
    print("✓ Automatic resume from last checkpoint")
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

    # Thread ID for this experiment
    thread_id = "food_shelf_life_v2"

    # Checkpoint setup
    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_path = checkpoint_dir / "checkpoints.db"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================================
    # RESUME DETECTION
    # ============================================================================

    checkpoint_manager = CheckpointManager(checkpoint_dir)

    resume = False
    if checkpoint_manager.has_checkpoints(thread_id):
        print("=" * 80)
        print("🔄 EXISTING CHECKPOINT FOUND")
        print("=" * 80)

        checkpoint_manager.print_resume_info(thread_id)

        # Ask user if they want to resume
        print()
        response = input("Resume from checkpoint? (y/n): ").strip().lower()

        if response == 'y' or response == 'yes':
            resume = True
            print("✅ Resuming from checkpoint\n")
        else:
            print("Starting fresh experiment (old checkpoint will be overwritten)\n")
            # Delete old checkpoints
            checkpoint_manager.delete_checkpoints(thread_id)

    # ============================================================================
    # LOAD DATA (always needed for execution context)
    # ============================================================================

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
    # INITIALIZE OR RESUME
    # ============================================================================

    if resume:
        # Resume mode - state will be loaded from checkpoint
        print("📦 Resuming from checkpoint...")
        print("   (Initial state will be overridden by checkpoint state)\n")

        # We still need to provide initial state for graph.stream()
        # but it will be ignored since we're resuming
        initial_state = None

    else:
        # Fresh start - initialize state
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
    # CREATE GRAPH WITH PERSISTENT CHECKPOINTS
    # ============================================================================

    print("🚀 Creating graph with persistent checkpoints...")
    graph = create_dream_team_graph_v2(checkpoint_path=checkpoint_path)
    print()

    # ============================================================================
    # RUN GRAPH WITH STREAMING
    # ============================================================================

    print("🚀 Starting execution...")
    print(f"   Thread ID: {thread_id}")
    print(f"   Mode: {'Resume' if resume else 'Fresh start'}")
    print(f"   Results: {results_dir}")
    print(f"   Streaming: enabled\n")

    try:
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        # Stream execution
        final_state = stream_graph_progress(graph, initial_state, config)

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
            print(f"Checkpoints saved to: {checkpoint_path}")

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
                    "Enhanced error diagnostics",
                    "Persistent checkpoints with SqliteSaver"
                ]
            }

            import json
            with open(results_dir / 'final_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)

            if tracing_enabled:
                print("\n📊 View execution trace at: https://smith.langchain.com/")

        print()

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        print("✅ Experiment state is checkpointed!")
        print(f"   Checkpoint: {checkpoint_path}")
        print(f"   Thread ID: {thread_id}")
        print("\nTo resume:")
        print(f"   python {Path(__file__).name}")
        print("   (Will automatically detect and prompt to resume)")

    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

        print("\n✅ Experiment state is checkpointed at last successful node!")
        print(f"   Checkpoint: {checkpoint_path}")
        print(f"   Thread ID: {thread_id}")
        print("\nTo resume:")
        print(f"   python {Path(__file__).name}")


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Error: GEMINI_API_KEY environment variable not set")
        print("Please set it: export GEMINI_API_KEY='your-key-here'")
        sys.exit(1)

    main()
