"""
LangGraph-based experiment orchestration.

This module defines the graph structure for experiment orchestration,
replacing the procedural loop in ExperimentOrchestrator with explicit nodes and edges.

## Graph Structure:

    START
      ↓
    bootstrap_node (if not completed)
      ↓
    init_math_framework_node
      ↓
    ┌──→ plan_node
    │     ↓
    │   code_node
    │     ↓
    │   execute_node
    │     ↓
    │   evaluate_node
    │     ↓
    │   check_continue_node → [STOP if goal achieved or max iterations]
    │     ↓
    │   check_evolution_node
    │     ↓
    │   [evolve_node if needed]
    │     ↓
    └───(next iteration)

Each node accepts and returns ExperimentState.
"""

from typing import Literal, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from pathlib import Path
import os

from .state import ExperimentState
from .nodes import (
    ExecutionContext,
    create_bootstrap_node,
    create_init_math_framework_node,
    create_plan_node,
    create_code_node,
    create_execute_node,
    create_evaluate_node,
    create_check_evolution_node,
    create_evolve_node
)
from .tracing import configure_langsmith, trace_experiment, create_experiment_metadata


# Node functions are now created via factory functions in nodes.py
# This avoids hardcoding them and allows dependency injection via ExecutionContext


# ============================================================================
# Routing Logic
# ============================================================================

def route_after_check_evolution(state: ExperimentState) -> Literal["evolve", "plan"]:
    """Route to evolution or next iteration based on evolution check"""
    return "evolve" if state.evolution.triggered else "plan"


def create_check_continue_node(ctx: ExecutionContext):
    """Create check continue node"""
    def check_continue_node(state: ExperimentState) -> ExperimentState:
        """Check if experiment should continue or stop"""
        print("\n" + "="*60)
        print(f"NODE: Check Continue")
        print("="*60)

        # Check goal achieved
        if state.check_goal_achieved():
            print(f"   🎯 Goal achieved! {state.config.target_metric}: {state.current_metrics[state.config.target_metric]:.4f}")
            state.goal_achieved = True
            state.should_stop = True
            state.phase = "complete"
            return state

        # Check max iterations
        if state.iteration >= state.config.max_iterations:
            print(f"   ⏱️  Max iterations ({state.config.max_iterations}) reached")
            state.should_stop = True
            state.phase = "complete"
            return state

        print(f"   ➡️  Continue to iteration {state.iteration + 1}")
        state.phase = "check_evolution"
        return state

    return check_continue_node


def create_complete_node(ctx: ExecutionContext):
    """Create complete node"""
    def complete_node(state: ExperimentState) -> ExperimentState:
        """Final node: experiment complete"""
        print("\n" + "="*60)
        print("NODE: Complete")
        print("="*60)

        print(f"\n✅ Experiment complete!")
        print(f"   Total iterations: {state.iteration}")
        print(f"   Best {state.config.target_metric}: {state.best_metric}")

        return state

    return complete_node


# ============================================================================
# Routing Logic
# ============================================================================

def route_after_check_evolution(state: ExperimentState) -> Literal["evolve", "plan"]:
    """Route to evolution or next iteration based on evolution check"""
    return "evolve" if state.evolution.triggered else "plan"


def route_after_check_continue(state: ExperimentState) -> Literal["complete", "check_evolution"]:
    """Route to completion or evolution check based on continue check"""
    return "complete" if state.should_stop else "check_evolution"


def route_after_bootstrap(state: ExperimentState) -> Literal["init_math", "plan"]:
    """Route from bootstrap to math init or directly to planning"""
    return "init_math" if not state.mathematical_state.iteration_count else "plan"


# ============================================================================
# Graph Construction
# ============================================================================

def create_experiment_graph(ctx: ExecutionContext) -> StateGraph:
    """
    Create the LangGraph state graph for experiment orchestration.

    Args:
        ctx: ExecutionContext with executor, agents, etc.

    Returns:
        Compiled StateGraph ready to execute
    """
    # Create graph
    graph = StateGraph(ExperimentState)

    # Create nodes with access to execution context
    graph.add_node("bootstrap", create_bootstrap_node(ctx))
    graph.add_node("init_math", create_init_math_framework_node(ctx))
    graph.add_node("plan", create_plan_node(ctx))
    graph.add_node("code", create_code_node(ctx))
    graph.add_node("execute", create_execute_node(ctx))
    graph.add_node("evaluate", create_evaluate_node(ctx))
    graph.add_node("check_continue", create_check_continue_node(ctx))
    graph.add_node("check_evolution", create_check_evolution_node(ctx))
    graph.add_node("evolve", create_evolve_node(ctx))
    graph.add_node("complete", create_complete_node(ctx))

    # Set entry point
    graph.set_entry_point("bootstrap")

    # Define edges
    graph.add_conditional_edges(
        "bootstrap",
        route_after_bootstrap,
        {
            "init_math": "init_math",
            "plan": "plan"
        }
    )

    graph.add_edge("init_math", "plan")
    graph.add_edge("plan", "code")
    graph.add_edge("code", "execute")
    graph.add_edge("execute", "evaluate")
    graph.add_edge("evaluate", "check_continue")

    graph.add_conditional_edges(
        "check_continue",
        route_after_check_continue,
        {
            "complete": "complete",
            "check_evolution": "check_evolution"
        }
    )

    graph.add_conditional_edges(
        "check_evolution",
        route_after_check_evolution,
        {
            "evolve": "evolve",
            "plan": "plan"
        }
    )

    graph.add_edge("evolve", "plan")
    graph.add_edge("complete", END)

    # Compile graph
    return graph.compile()


# ============================================================================
# Runner Function
# ============================================================================

def run_graph_experiment(
    state: ExperimentState,
    data_context: Dict[str, Any],
    research_api: Optional[Any] = None,
    enable_tracing: bool = True,
    langsmith_project: Optional[str] = None
) -> ExperimentState:
    """
    Run the experiment using the LangGraph.

    Args:
        state: Initial experiment state
        data_context: Data context with DataFrames and other objects
        research_api: Optional research API for paper search
        enable_tracing: Whether to enable LangSmith tracing (default: True)
        langsmith_project: Optional LangSmith project name (default: "dream-team-experiments")

    Returns:
        Final experiment state

    Environment Variables (for tracing):
        LANGSMITH_API_KEY: LangSmith API key (required for tracing)
        LANGSMITH_PROJECT: Project name (optional, overrides langsmith_project arg)
    """
    print("="*70)
    print("🚀 LANGGRAPH EXPERIMENT ORCHESTRATION")
    print("="*70)
    print(f"\nProblem: {state.config.problem_statement[:100]}...")
    print(f"Target Metric: {state.config.target_metric} ({'minimize' if state.config.minimize_metric else 'maximize'})")
    print(f"Max Iterations: {state.config.max_iterations}")
    print()

    # Configure LangSmith tracing if enabled
    if enable_tracing:
        project = langsmith_project or os.getenv('LANGSMITH_PROJECT') or "dream-team-experiments"
        configure_langsmith(project=project)

    # Create execution context with all non-serializable objects
    ctx = ExecutionContext(
        data_context=data_context,
        results_dir=Path(state.results_dir),
        research_api=research_api
    )

    # Create agents from state configuration
    ctx.create_agents_from_state(state)

    # Create and run graph
    graph = create_experiment_graph(ctx)

    # Run with tracing context
    experiment_name = f"{state.config.target_metric}_optimization"
    metadata = create_experiment_metadata(state)

    if enable_tracing and os.getenv('LANGSMITH_API_KEY'):
        with trace_experiment(experiment_name, metadata=metadata):
            final_state = graph.invoke(state)
    else:
        final_state = graph.invoke(state)

    print("\n" + "="*70)
    print("✅ EXPERIMENT COMPLETE")
    print("="*70)

    return final_state
