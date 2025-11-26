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

from .state import ExperimentState
from ..agent import Agent
from ..executor import CodeExecutor
from ..team import Team
from ..evolution import EvolutionEngine
from ..knowledge_state import KnowledgeGraph


# ============================================================================
# Node Functions
# ============================================================================

def bootstrap_node(state: ExperimentState) -> ExperimentState:
    """
    Bootstrap phase: PI explores problem and recruits team.

    For now, this is a placeholder that will call the existing _bootstrap_exploration logic.
    In Phase 4, we'll move the actual logic here.
    """
    print("\n" + "="*60)
    print("NODE: Bootstrap")
    print("="*60)

    if state.bootstrap_completed:
        print("   ✓ Bootstrap already completed, skipping")
        state.phase = "plan"
        return state

    # TODO Phase 4: Move actual bootstrap logic here
    # For now, just mark as completed and move to next phase
    print("   ⚠️  Bootstrap logic not yet migrated to node")
    state.bootstrap_completed = True
    state.phase = "plan"

    return state


def init_math_framework_node(state: ExperimentState) -> ExperimentState:
    """
    Initialize mathematical framework for the experiment.

    Creates problem graph and team object for dynamics.
    """
    print("\n" + "="*60)
    print("NODE: Initialize Mathematical Framework")
    print("="*60)

    # TODO Phase 4: Move _initialize_mathematical_framework logic here
    print("   ⚠️  Math framework initialization not yet migrated")

    state.phase = "plan"
    return state


def plan_node(state: ExperimentState) -> ExperimentState:
    """
    Team planning meeting to decide approach.

    Runs team meeting, synthesizes proposals into action plan.
    """
    print("\n" + "="*60)
    print(f"NODE: Plan (Iteration {state.iteration + 1})")
    print("="*60)

    # TODO Phase 4: Move _team_planning_meeting logic here
    print("   ⚠️  Planning logic not yet migrated")

    # Placeholder
    state.current_approach = f"Placeholder approach for iteration {state.iteration + 1}"
    state.phase = "code"

    return state


def code_node(state: ExperimentState) -> ExperimentState:
    """
    Coding agent implements the planned approach.

    Translates team's plan into executable Python code.
    """
    print("\n" + "="*60)
    print(f"NODE: Code (Iteration {state.iteration + 1})")
    print("="*60)

    # TODO Phase 4: Move _implement_approach logic here
    print("   ⚠️  Coding logic not yet migrated")

    # Placeholder
    state.current_code = "# Placeholder code\nprint('Hello from iteration {}')".format(state.iteration + 1)
    state.phase = "execute"

    return state


def execute_node(state: ExperimentState) -> ExperimentState:
    """
    Execute the generated code.

    Runs code with retry on errors, auto-installs packages.
    """
    print("\n" + "="*60)
    print(f"NODE: Execute (Iteration {state.iteration + 1})")
    print("="*60)

    # TODO Phase 4: Move _execute_with_retry logic here
    print("   ⚠️  Execution logic not yet migrated")

    # Placeholder
    state.current_results = {
        'success': True,
        'output': f'Execution output for iteration {state.iteration + 1}',
        'code': state.current_code,
        'description': state.current_approach
    }
    state.phase = "evaluate"

    return state


def evaluate_node(state: ExperimentState) -> ExperimentState:
    """
    Evaluate execution results and extract metrics.

    Updates metrics, checks for new best.
    """
    print("\n" + "="*60)
    print(f"NODE: Evaluate (Iteration {state.iteration + 1})")
    print("="*60)

    # TODO Phase 4: Move _extract_metrics logic here
    print("   ⚠️  Evaluation logic not yet migrated")

    # Placeholder: simulate some metrics
    import random
    state.current_metrics = {
        state.config.target_metric: random.uniform(0.5, 2.0)
    }

    # Update best metric
    is_new_best = state.update_best_metric()
    if is_new_best:
        print(f"   ✨ New best {state.config.target_metric}: {state.best_metric:.4f}")

    # Save iteration summary to history
    summary = state.get_iteration_summary()
    state.history.append(summary)

    # Increment iteration counter
    state.iteration += 1

    state.phase = "check_continue"

    return state


def check_continue_node(state: ExperimentState) -> ExperimentState:
    """
    Check if experiment should continue or stop.

    Checks:
    - Goal achieved?
    - Max iterations reached?
    """
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


def check_evolution_node(state: ExperimentState) -> ExperimentState:
    """
    Check if team evolution is needed.

    Uses mathematical signals and triggers to decide.
    """
    print("\n" + "="*60)
    print(f"NODE: Check Evolution")
    print("="*60)

    # TODO Phase 4: Move _check_mathematical_evolution logic here
    should_evolve = state.should_evolve()

    if should_evolve:
        print("   🔔 Evolution needed")
        state.evolution.triggered = True
        state.phase = "evolve"
    else:
        print("   ✓ No evolution needed, continuing")
        state.evolution.triggered = False
        state.phase = "plan"  # Back to planning for next iteration

    return state


def evolve_node(state: ExperimentState) -> ExperimentState:
    """
    Evolve team composition.

    Recruits new agents, deepens expertise, or restructures team.
    """
    print("\n" + "="*60)
    print(f"NODE: Evolve")
    print("="*60)

    # TODO Phase 5: Move _evolve_team logic here
    print("   ⚠️  Evolution logic not yet migrated")

    # Reset evolution state
    state.evolution.triggered = False
    state.evolution.decision = "NO_CHANGE"  # Placeholder

    state.phase = "plan"  # Back to planning

    return state


def complete_node(state: ExperimentState) -> ExperimentState:
    """
    Final node: experiment complete.

    Generates final summary and saves results.
    """
    print("\n" + "="*60)
    print("NODE: Complete")
    print("="*60)

    print(f"\n✅ Experiment complete!")
    print(f"   Total iterations: {state.iteration}")
    print(f"   Best {state.config.target_metric}: {state.best_metric}")

    return state


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
    if state.mathematical_state.iteration_count == 0:
        return "init_math"
    return "plan"


# ============================================================================
# Graph Construction
# ============================================================================

def create_experiment_graph() -> StateGraph:
    """
    Create the LangGraph state graph for experiment orchestration.

    Returns:
        Compiled StateGraph ready to execute
    """
    # Create graph
    graph = StateGraph(ExperimentState)

    # Add nodes
    graph.add_node("bootstrap", bootstrap_node)
    graph.add_node("init_math", init_math_framework_node)
    graph.add_node("plan", plan_node)
    graph.add_node("code", code_node)
    graph.add_node("execute", execute_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("check_continue", check_continue_node)
    graph.add_node("check_evolution", check_evolution_node)
    graph.add_node("evolve", evolve_node)
    graph.add_node("complete", complete_node)

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
    executor: CodeExecutor,
    research_api: Optional[Any] = None
) -> ExperimentState:
    """
    Run the experiment using the LangGraph.

    Args:
        state: Initial experiment state
        executor: Code executor for running generated code
        research_api: Optional research API for paper search

    Returns:
        Final experiment state
    """
    print("="*70)
    print("🚀 LANGGRAPH EXPERIMENT ORCHESTRATION")
    print("="*70)
    print(f"\nProblem: {state.config.problem_statement[:100]}...")
    print(f"Target Metric: {state.config.target_metric} ({'minimize' if state.config.minimize_metric else 'maximize'})")
    print(f"Max Iterations: {state.config.max_iterations}")
    print()

    # Create and run graph
    graph = create_experiment_graph()

    # Note: For now, we run the full graph once
    # The graph internally loops via plan → code → execute → evaluate → check → plan
    # until max iterations or goal achieved

    # TODO Phase 4: Integrate actual executor and research API into nodes

    final_state = graph.invoke(state)

    print("\n" + "="*70)
    print("✅ EXPERIMENT COMPLETE")
    print("="*70)

    return final_state
