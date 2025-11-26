#!/usr/bin/env python3
"""
Unit tests for LangGraph nodes.

Tests individual node behavior in isolation.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

from dream_team.experiment.state import (
    ExperimentState,
    ExperimentConfig,
    TeamConfig,
    AgentConfig,
    MathematicalState,
    EvolutionState
)
from dream_team.experiment.nodes import (
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


@pytest.fixture
def test_data():
    """Create test data for experiments"""
    np.random.seed(42)
    train_df = pd.DataFrame({
        'id': range(50),
        'feature_1': np.random.randn(50),
        'feature_2': np.random.randn(50),
        'target': np.random.randn(50) * 10 + 50
    })

    test_df = pd.DataFrame({
        'id': range(50, 60),
        'feature_1': np.random.randn(10),
        'feature_2': np.random.randn(10)
    })

    return {'train_df': train_df, 'test_df': test_df}


@pytest.fixture
def agent_configs():
    """Create test agent configurations"""
    team_lead = AgentConfig(
        title="Principal Investigator",
        expertise="machine learning, experimental design",
        goal="optimize metrics",
        role="lead research"
    )

    coding_agent = AgentConfig(
        title="Research Engineer",
        expertise="Python, pandas, scikit-learn",
        goal="implement plans",
        role="write code"
    )

    return team_lead, coding_agent


@pytest.fixture
def initial_state(agent_configs, tmp_path):
    """Create initial experiment state"""
    team_lead, coding_agent = agent_configs

    config = ExperimentConfig(
        problem_statement="Predict target using features. Minimize MAE.",
        target_metric="mae",
        minimize_metric=True,
        max_iterations=3,
        goal_target=None
    )

    team = TeamConfig(
        team_lead=team_lead,
        team_members=[],
        coding_agent=coding_agent
    )

    return ExperimentState(
        iteration=0,
        phase="bootstrap",
        team=team,
        config=config,
        history=[],
        current_approach=None,
        current_code=None,
        current_results=None,
        current_metrics={},
        best_metric=None,
        best_iteration=None,
        bootstrap_completed=False,
        column_schemas={},
        mathematical_state=MathematicalState(),
        evolution=EvolutionState(),
        goal_achieved=False,
        should_stop=False,
        results_dir=str(tmp_path / "test_results")
    )


@pytest.fixture
def execution_context(test_data, tmp_path):
    """Create execution context for nodes"""
    return ExecutionContext(
        data_context=test_data,
        results_dir=tmp_path / "test_results",
        research_api=None
    )


# ============================================================================
# Bootstrap Node Tests
# ============================================================================

def test_bootstrap_node_completes(initial_state, execution_context):
    """Test that bootstrap node completes successfully"""
    bootstrap_node = create_bootstrap_node(execution_context)

    result_state = bootstrap_node(initial_state)

    # Check bootstrap completed
    assert result_state.bootstrap_completed is True

    # Check team recruited
    assert len(result_state.team.team_members) > 0

    # Check column schemas extracted
    assert len(result_state.column_schemas) > 0

    # Check phase updated
    assert result_state.phase == "init_math"


def test_bootstrap_node_skips_if_completed(initial_state, execution_context):
    """Test that bootstrap node skips if already completed"""
    initial_state.bootstrap_completed = True
    initial_state.team.team_members = [
        AgentConfig(
            title="Data Scientist",
            expertise="statistics",
            goal="analyze data",
            role="statistical analysis"
        )
    ]

    bootstrap_node = create_bootstrap_node(execution_context)
    result_state = bootstrap_node(initial_state)

    # Should skip and go straight to planning
    assert result_state.phase == "plan"


# ============================================================================
# Init Math Framework Node Tests
# ============================================================================

def test_init_math_framework_creates_structures(initial_state, execution_context):
    """Test that init math node creates problem graph and team dynamics"""
    # Complete bootstrap first
    initial_state.bootstrap_completed = True
    initial_state.team.team_members = [
        AgentConfig(
            title="Data Scientist",
            expertise="statistics, machine learning",
            goal="build models",
            role="modeling"
        )
    ]

    init_node = create_init_math_framework_node(execution_context)
    execution_context.create_agents_from_state(initial_state)

    result_state = init_node(initial_state)

    # Check mathematical state initialized
    assert result_state.mathematical_state.iteration_count == 1

    # Check phase updated
    assert result_state.phase == "plan"


# ============================================================================
# Plan Node Tests
# ============================================================================

def test_plan_node_generates_approach(initial_state, execution_context):
    """Test that plan node generates an approach"""
    # Setup state
    initial_state.bootstrap_completed = True
    initial_state.team.team_members = [
        AgentConfig(
            title="Data Scientist",
            expertise="machine learning",
            goal="build models",
            role="modeling"
        )
    ]
    initial_state.iteration = 1
    execution_context.create_agents_from_state(initial_state)

    plan_node = create_plan_node(execution_context)
    result_state = plan_node(initial_state)

    # Check approach generated
    assert result_state.current_approach is not None
    assert len(result_state.current_approach) > 0

    # Check phase updated
    assert result_state.phase == "code"


# ============================================================================
# Code Node Tests
# ============================================================================

def test_code_node_generates_code(initial_state, execution_context):
    """Test that code node generates executable code"""
    # Setup state with approach
    initial_state.bootstrap_completed = True
    initial_state.current_approach = "Use linear regression to predict target from features."
    initial_state.iteration = 1
    initial_state.team.team_members = [
        AgentConfig(
            title="Data Scientist",
            expertise="machine learning",
            goal="build models",
            role="modeling"
        )
    ]
    execution_context.create_agents_from_state(initial_state)

    code_node = create_code_node(execution_context)
    result_state = code_node(initial_state)

    # Check code generated
    assert result_state.current_code is not None
    assert len(result_state.current_code) > 0
    assert "train_df" in result_state.current_code or "test_df" in result_state.current_code

    # Check phase updated
    assert result_state.phase == "execute"


# ============================================================================
# Execute Node Tests
# ============================================================================

def test_execute_node_runs_code(initial_state, execution_context):
    """Test that execute node runs code and captures results"""
    # Setup state with simple code
    initial_state.current_code = """
import pandas as pd
import numpy as np

# Simple baseline
predictions = pd.DataFrame({
    'id': test_df['id'],
    'target': [50.0] * len(test_df)
})

# Save submission
predictions.to_csv('submission.csv', index=False)
"""
    initial_state.iteration = 1

    execute_node = create_execute_node(execution_context)
    result_state = execute_node(initial_state)

    # Check results captured
    assert result_state.current_results is not None

    # Check phase updated
    assert result_state.phase == "evaluate"


# ============================================================================
# Evaluate Node Tests
# ============================================================================

def test_evaluate_node_extracts_metrics(initial_state, execution_context):
    """Test that evaluate node extracts metrics from results"""
    # Setup state with execution results
    initial_state.current_results = {
        'stdout': 'MAE: 5.234',
        'stderr': '',
        'submission_file': 'submission.csv'
    }
    initial_state.iteration = 1

    evaluate_node = create_evaluate_node(execution_context)
    result_state = evaluate_node(initial_state)

    # Check metrics extracted
    assert 'mae' in result_state.current_metrics

    # Check best metric updated
    assert result_state.best_metric is not None

    # Check history updated
    assert len(result_state.history) == 1

    # Check phase updated
    assert result_state.phase == "check_continue"


def test_evaluate_node_updates_best_metric(initial_state, execution_context):
    """Test that evaluate node updates best metric when improved"""
    # Setup state with previous best
    initial_state.best_metric = 10.0
    initial_state.best_iteration = 1
    initial_state.current_results = {
        'stdout': 'MAE: 5.0',
        'stderr': ''
    }
    initial_state.iteration = 2

    evaluate_node = create_evaluate_node(execution_context)
    result_state = evaluate_node(initial_state)

    # Check best metric updated (lower is better for MAE)
    assert result_state.best_metric == 5.0
    assert result_state.best_iteration == 2


# ============================================================================
# Check Evolution Node Tests
# ============================================================================

def test_check_evolution_node_no_trigger(initial_state, execution_context):
    """Test that check evolution node doesn't trigger evolution too early"""
    initial_state.iteration = 1
    initial_state.history = []

    check_node = create_check_evolution_node(execution_context)
    result_state = check_node(initial_state)

    # Should not trigger evolution on first iteration
    assert result_state.evolution.triggered is False


def test_check_evolution_node_detects_stagnation(initial_state, execution_context):
    """Test that check evolution node detects stagnation"""
    # Setup state with stagnation pattern
    initial_state.iteration = 5
    initial_state.history = [
        type('obj', (), {
            'iteration': 1,
            'metrics': {'mae': 10.0},
            'approach': 'approach 1'
        })(),
        type('obj', (), {
            'iteration': 2,
            'metrics': {'mae': 10.0},
            'approach': 'approach 2'
        })(),
        type('obj', (), {
            'iteration': 3,
            'metrics': {'mae': 10.0},
            'approach': 'approach 3'
        })(),
        type('obj', (), {
            'iteration': 4,
            'metrics': {'mae': 10.0},
            'approach': 'approach 4'
        })(),
    ]

    check_node = create_check_evolution_node(execution_context)
    execution_context.create_agents_from_state(initial_state)

    result_state = check_node(initial_state)

    # Should trigger evolution due to stagnation
    assert result_state.evolution.triggered is True


# ============================================================================
# Integration: Multi-Node Flow
# ============================================================================

def test_node_sequence_flow(initial_state, execution_context):
    """Test that nodes flow together correctly"""
    # 1. Bootstrap
    bootstrap = create_bootstrap_node(execution_context)
    state = bootstrap(initial_state)
    assert state.bootstrap_completed is True

    # 2. Init math framework
    init_math = create_init_math_framework_node(execution_context)
    execution_context.create_agents_from_state(state)
    state = init_math(state)
    assert state.mathematical_state.iteration_count == 1

    # 3. Plan
    plan = create_plan_node(execution_context)
    state.iteration = 1
    state = plan(state)
    assert state.current_approach is not None

    # 4. Code
    code = create_code_node(execution_context)
    state = code(state)
    assert state.current_code is not None

    # Verify state consistency throughout
    assert state.iteration == 1
    assert state.bootstrap_completed is True


# ============================================================================
# Error Handling Tests
# ============================================================================

def test_execute_node_handles_code_errors(initial_state, execution_context):
    """Test that execute node handles code errors gracefully"""
    # Setup state with bad code
    initial_state.current_code = """
# This will raise an error
raise ValueError("Test error")
"""
    initial_state.iteration = 1

    execute_node = create_execute_node(execution_context)

    # Should not raise, but capture error in results
    result_state = execute_node(initial_state)

    # Check error captured in results
    assert result_state.current_results is not None
    # The node should have retried and eventually returned results


# ============================================================================
# State Immutability Tests
# ============================================================================

def test_nodes_return_new_state(initial_state, execution_context):
    """Test that nodes return modified state (Pydantic handles immutability)"""
    initial_state.bootstrap_completed = True
    initial_state.team.team_members = [
        AgentConfig(
            title="Data Scientist",
            expertise="ML",
            goal="optimize",
            role="modeling"
        )
    ]
    execution_context.create_agents_from_state(initial_state)

    plan_node = create_plan_node(execution_context)
    initial_state.iteration = 1

    result_state = plan_node(initial_state)

    # Result should have updated approach
    assert result_state.current_approach is not None

    # Original state should be unchanged (Pydantic model)
    # Note: Pydantic models are mutable, but nodes should treat state functionally


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
