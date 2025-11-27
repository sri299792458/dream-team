"""
LangSmith tracing integration for Dream Team experiments.

Provides observability into experiment orchestration at the node/phase level.
"""

import os
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def configure_langsmith(
    project: str = "dream-team-experiments",
    api_key: Optional[str] = None
) -> bool:
    """
    Configure LangSmith tracing for the experiment.

    Args:
        project: LangSmith project name
        api_key: Optional API key (defaults to LANGSMITH_API_KEY env var)

    Returns:
        True if configured successfully, False otherwise
    """
    # Check if API key is available
    api_key = api_key or os.getenv('LANGSMITH_API_KEY')

    if not api_key:
        print("⚠️  LangSmith API key not found. Tracing disabled.")
        print("   Set LANGSMITH_API_KEY environment variable to enable tracing.")
        return False

    # Set environment variables for LangSmith
    os.environ['LANGSMITH_API_KEY'] = api_key
    os.environ['LANGSMITH_PROJECT'] = project
    os.environ['LANGSMITH_TRACING'] = 'true'

    print(f"✅ LangSmith tracing enabled")
    print(f"   Project: {project}")
    print(f"   View traces at: https://smith.langchain.com/")

    return True


@contextmanager
def trace_experiment(
    experiment_name: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Context manager for tracing an entire experiment.

    Usage:
        with trace_experiment("shelf_life_prediction", metadata={"version": "1.0"}):
            final_state = run_graph_experiment(state, data_context)

    Args:
        experiment_name: Name of the experiment
        metadata: Optional metadata to attach to the trace
    """
    try:
        from langsmith import Client
        from langsmith.run_helpers import traceable

        client = Client()

        # Start experiment trace
        print(f"\n📊 Starting traced experiment: {experiment_name}")
        if metadata:
            print(f"   Metadata: {metadata}")

        # The actual tracing happens automatically via LangGraph's integration
        # We just need to ensure the environment is configured
        yield

        print(f"\n✅ Experiment trace complete")
        print(f"   View at: https://smith.langchain.com/")

    except ImportError:
        print("⚠️  langsmith not installed, tracing disabled")
        yield
    except Exception as e:
        print(f"⚠️  Tracing error: {e}")
        yield


def add_node_metadata(
    state,
    node_name: str,
    metadata: Dict[str, Any]
) -> None:
    """
    Add metadata for a specific node execution.

    This is called within nodes to add custom metadata to traces.

    Args:
        state: Current experiment state
        node_name: Name of the node
        metadata: Metadata to attach
    """
    try:
        from langsmith import Client

        # In LangGraph, metadata is automatically captured
        # We can log it for visibility
        if os.getenv('LANGSMITH_TRACING') == 'true':
            print(f"   [TRACE] {node_name}: {metadata}")
    except ImportError as e:
        logger.debug(f"LangSmith not available for node metadata: {e}")
    except Exception as e:
        logger.warning(f"Failed to create node metadata for {node_name}: {e}")


def create_experiment_metadata(state) -> Dict[str, Any]:
    """
    Create metadata dictionary from experiment state.

    Args:
        state: ExperimentState

    Returns:
        Dictionary of metadata for tracing
    """
    return {
        'iteration': state.iteration,
        'phase': state.phase,
        'target_metric': state.config.target_metric,
        'minimize_metric': state.config.minimize_metric,
        'max_iterations': state.config.max_iterations,
        'best_metric': state.best_metric,
        'team_size': len(state.team.team_members) + 1,  # +1 for lead
        'bootstrap_completed': state.bootstrap_completed,
        'goal_achieved': state.goal_achieved
    }


def create_node_metadata(
    state,
    node_name: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Create metadata for a specific node execution.

    Args:
        state: ExperimentState
        node_name: Name of the node
        **kwargs: Additional metadata specific to this node

    Returns:
        Dictionary of metadata
    """
    base_metadata = {
        'node': node_name,
        'iteration': state.iteration,
        'phase': state.phase,
    }

    # Add node-specific metadata
    if node_name == 'bootstrap':
        base_metadata.update({
            'bootstrap_completed': state.bootstrap_completed,
            'team_recruited': len(state.team.team_members)
        })

    elif node_name == 'plan':
        base_metadata.update({
            'team_size': len(state.team.team_members) + 1,
            'has_history': len(state.history) > 0
        })

    elif node_name == 'execute':
        base_metadata.update({
            'code_length': len(state.current_code) if state.current_code else 0
        })

    elif node_name == 'evaluate':
        base_metadata.update({
            'metrics': state.current_metrics,
            'best_metric': state.best_metric,
            'is_new_best': kwargs.get('is_new_best', False)
        })

    elif node_name == 'evolve':
        base_metadata.update({
            'evolution_triggered': state.evolution.triggered,
            'evolution_decision': state.evolution.decision,
            'trigger_names': state.evolution.trigger_names
        })

    # Add any extra kwargs
    base_metadata.update(kwargs)

    return base_metadata


# Example usage in a node:
"""
def my_node(state: ExperimentState) -> ExperimentState:
    # Do work...

    # Add tracing metadata
    metadata = create_node_metadata(
        state,
        'my_node',
        custom_field='custom_value'
    )
    add_node_metadata(state, 'my_node', metadata)

    return state
"""
