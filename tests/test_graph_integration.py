#!/usr/bin/env python3
"""
Integration tests for the LangGraph experiment orchestration.

Tests full graph execution end-to-end.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from dream_team.experiment import (
    create_initial_state,
    AgentConfig,
    run_graph_experiment
)


@pytest.fixture
def test_data():
    """Create test dataset"""
    np.random.seed(42)

    train_df = pd.DataFrame({
        'id': range(100),
        'feature_1': np.random.randn(100),
        'feature_2': np.random.randn(100),
        'feature_3': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.randn(100) * 10 + 50
    })

    test_df = pd.DataFrame({
        'id': range(100, 120),
        'feature_1': np.random.randn(20),
        'feature_2': np.random.randn(20),
        'feature_3': np.random.choice(['A', 'B', 'C'], 20)
    })

    return {
        'train_df': train_df,
        'test_df': test_df
    }


@pytest.fixture
def agent_configs():
    """Create agent configurations"""
    team_lead = AgentConfig(
        title="Principal Investigator",
        expertise="machine learning, experimental design, research methodology",
        goal="optimize the target metric through systematic experimentation",
        role="explore problem, recruit team, coordinate research"
    )

    coding_agent = AgentConfig(
        title="Research Engineer",
        expertise="Python, pandas, scikit-learn, numpy, data analysis",
        goal="implement research plans accurately",
        role="translate strategies into executable code"
    )

    return team_lead, coding_agent


# ============================================================================
# Basic Integration Tests
# ============================================================================

def test_graph_runs_single_iteration(test_data, agent_configs, tmp_path):
    """Test that graph can complete a single iteration"""
    team_lead, coding_agent = agent_configs

    state = create_initial_state(
        team_lead=team_lead,
        coding_agent=coding_agent,
        problem_statement="""
Predict the target variable using available features.

Target Variable: target (continuous)
Evaluation Metric: MAE (Mean Absolute Error) - lower is better

Data:
- train_df: Training data with target
- test_df: Test data (predict target)

Goal: Build a predictive model to minimize MAE.
""",
        target_metric="mae",
        minimize_metric=True,
        max_iterations=1,
        results_dir=str(tmp_path / "results")
    )

    final_state = run_graph_experiment(
        state,
        test_data,
        enable_tracing=False  # Disable for tests
    )

    # Verify iteration completed
    assert final_state.iteration >= 1
    assert final_state.bootstrap_completed is True

    # Verify metrics tracked
    assert final_state.best_metric is not None
    assert 'mae' in final_state.current_metrics

    # Verify history recorded
    assert len(final_state.history) >= 1

    # Verify completion
    assert final_state.phase == "complete"
    assert final_state.should_stop is True


def test_graph_runs_multiple_iterations(test_data, agent_configs, tmp_path):
    """Test that graph can complete multiple iterations"""
    team_lead, coding_agent = agent_configs

    state = create_initial_state(
        team_lead=team_lead,
        coding_agent=coding_agent,
        problem_statement="""
Predict the target variable using available features.
Evaluation Metric: MAE - lower is better
""",
        target_metric="mae",
        minimize_metric=True,
        max_iterations=3,
        results_dir=str(tmp_path / "results")
    )

    final_state = run_graph_experiment(
        state,
        test_data,
        enable_tracing=False
    )

    # Verify multiple iterations completed
    assert final_state.iteration >= 1  # May stop early if goal achieved

    # Verify history has multiple entries or stopped early
    assert len(final_state.history) >= 1

    # Verify completion state
    assert final_state.phase == "complete"
    assert final_state.should_stop is True


def test_graph_stops_at_max_iterations(test_data, agent_configs, tmp_path):
    """Test that graph respects max_iterations limit"""
    team_lead, coding_agent = agent_configs

    state = create_initial_state(
        team_lead=team_lead,
        coding_agent=coding_agent,
        problem_statement="Predict target. Minimize MAE.",
        target_metric="mae",
        minimize_metric=True,
        max_iterations=2,
        results_dir=str(tmp_path / "results")
    )

    final_state = run_graph_experiment(
        state,
        test_data,
        enable_tracing=False
    )

    # Should not exceed max iterations
    assert final_state.iteration <= 2

    # Should be in complete phase
    assert final_state.phase == "complete"


# ============================================================================
# Bootstrap and Team Recruitment Tests
# ============================================================================

def test_graph_recruits_team_during_bootstrap(test_data, agent_configs, tmp_path):
    """Test that graph recruits team members during bootstrap"""
    team_lead, coding_agent = agent_configs

    state = create_initial_state(
        team_lead=team_lead,
        coding_agent=coding_agent,
        problem_statement="Complex ML problem requiring team collaboration.",
        target_metric="mae",
        minimize_metric=True,
        max_iterations=1,
        results_dir=str(tmp_path / "results")
    )

    final_state = run_graph_experiment(
        state,
        test_data,
        enable_tracing=False
    )

    # Verify team recruited
    assert len(final_state.team.team_members) > 0

    # Verify bootstrap completed
    assert final_state.bootstrap_completed is True


def test_graph_initializes_mathematical_framework(test_data, agent_configs, tmp_path):
    """Test that graph initializes mathematical framework"""
    team_lead, coding_agent = agent_configs

    state = create_initial_state(
        team_lead=team_lead,
        coding_agent=coding_agent,
        problem_statement="Predict target. Minimize MAE.",
        target_metric="mae",
        minimize_metric=True,
        max_iterations=1,
        results_dir=str(tmp_path / "results")
    )

    final_state = run_graph_experiment(
        state,
        test_data,
        enable_tracing=False
    )

    # Verify mathematical state initialized
    assert final_state.mathematical_state.iteration_count > 0


# ============================================================================
# Planning and Execution Tests
# ============================================================================

def test_graph_generates_and_executes_code(test_data, agent_configs, tmp_path):
    """Test that graph generates and executes code"""
    team_lead, coding_agent = agent_configs

    state = create_initial_state(
        team_lead=team_lead,
        coding_agent=coding_agent,
        problem_statement="Predict target. Minimize MAE.",
        target_metric="mae",
        minimize_metric=True,
        max_iterations=1,
        results_dir=str(tmp_path / "results")
    )

    final_state = run_graph_experiment(
        state,
        test_data,
        enable_tracing=False
    )

    # Verify code was generated
    assert final_state.current_code is not None
    assert len(final_state.current_code) > 0

    # Verify code was executed
    assert final_state.current_results is not None


def test_graph_tracks_best_metric(test_data, agent_configs, tmp_path):
    """Test that graph tracks best metric across iterations"""
    team_lead, coding_agent = agent_configs

    state = create_initial_state(
        team_lead=team_lead,
        coding_agent=coding_agent,
        problem_statement="Predict target. Minimize MAE.",
        target_metric="mae",
        minimize_metric=True,
        max_iterations=2,
        results_dir=str(tmp_path / "results")
    )

    final_state = run_graph_experiment(
        state,
        test_data,
        enable_tracing=False
    )

    # Verify best metric tracked
    assert final_state.best_metric is not None

    # Verify best iteration tracked
    assert final_state.best_iteration is not None
    assert final_state.best_iteration >= 1


# ============================================================================
# Evolution Tests
# ============================================================================

def test_graph_checks_evolution_signals(test_data, agent_configs, tmp_path):
    """Test that graph checks for evolution signals"""
    team_lead, coding_agent = agent_configs

    state = create_initial_state(
        team_lead=team_lead,
        coding_agent=coding_agent,
        problem_statement="Predict target. Minimize MAE.",
        target_metric="mae",
        minimize_metric=True,
        max_iterations=2,
        results_dir=str(tmp_path / "results")
    )

    final_state = run_graph_experiment(
        state,
        test_data,
        enable_tracing=False
    )

    # Evolution state should exist
    assert final_state.evolution is not None

    # Check if evolution was checked (may or may not trigger)
    # The evolution.triggered flag depends on signals


# ============================================================================
# Goal Achievement Tests
# ============================================================================

def test_graph_stops_when_goal_achieved(test_data, agent_configs, tmp_path):
    """Test that graph stops early when goal is achieved"""
    team_lead, coding_agent = agent_configs

    state = create_initial_state(
        team_lead=team_lead,
        coding_agent=coding_agent,
        problem_statement="Predict target. Minimize MAE.",
        target_metric="mae",
        minimize_metric=True,
        max_iterations=10,
        target_score=100.0,  # Easy goal to achieve
        results_dir=str(tmp_path / "results")
    )

    final_state = run_graph_experiment(
        state,
        test_data,
        enable_tracing=False
    )

    # Should achieve goal and stop early
    if final_state.goal_achieved:
        assert final_state.iteration < 10
        assert final_state.phase == "complete"


# ============================================================================
# History and State Tests
# ============================================================================

def test_graph_maintains_history(test_data, agent_configs, tmp_path):
    """Test that graph maintains iteration history"""
    team_lead, coding_agent = agent_configs

    state = create_initial_state(
        team_lead=team_lead,
        coding_agent=coding_agent,
        problem_statement="Predict target. Minimize MAE.",
        target_metric="mae",
        minimize_metric=True,
        max_iterations=2,
        results_dir=str(tmp_path / "results")
    )

    final_state = run_graph_experiment(
        state,
        test_data,
        enable_tracing=False
    )

    # Verify history exists
    assert len(final_state.history) > 0

    # Verify history entries have required fields
    for entry in final_state.history:
        assert hasattr(entry, 'iteration')
        assert hasattr(entry, 'metrics')
        assert hasattr(entry, 'approach')


def test_graph_updates_column_schemas(test_data, agent_configs, tmp_path):
    """Test that graph extracts and maintains column schemas"""
    team_lead, coding_agent = agent_configs

    state = create_initial_state(
        team_lead=team_lead,
        coding_agent=coding_agent,
        problem_statement="Predict target. Minimize MAE.",
        target_metric="mae",
        minimize_metric=True,
        max_iterations=1,
        results_dir=str(tmp_path / "results")
    )

    final_state = run_graph_experiment(
        state,
        test_data,
        enable_tracing=False
    )

    # Verify column schemas extracted
    assert 'train_df' in final_state.column_schemas
    assert 'test_df' in final_state.column_schemas


# ============================================================================
# Results Directory Tests
# ============================================================================

def test_graph_creates_results_directory(test_data, agent_configs, tmp_path):
    """Test that graph creates results directory"""
    team_lead, coding_agent = agent_configs

    results_dir = tmp_path / "test_results"

    state = create_initial_state(
        team_lead=team_lead,
        coding_agent=coding_agent,
        problem_statement="Predict target. Minimize MAE.",
        target_metric="mae",
        minimize_metric=True,
        max_iterations=1,
        results_dir=str(results_dir)
    )

    final_state = run_graph_experiment(
        state,
        test_data,
        enable_tracing=False
    )

    # Verify results directory exists
    assert results_dir.exists()


# ============================================================================
# Tracing Tests
# ============================================================================

def test_graph_runs_without_tracing(test_data, agent_configs, tmp_path):
    """Test that graph works without tracing enabled"""
    team_lead, coding_agent = agent_configs

    state = create_initial_state(
        team_lead=team_lead,
        coding_agent=coding_agent,
        problem_statement="Predict target. Minimize MAE.",
        target_metric="mae",
        minimize_metric=True,
        max_iterations=1,
        results_dir=str(tmp_path / "results")
    )

    # Should work fine without tracing
    final_state = run_graph_experiment(
        state,
        test_data,
        enable_tracing=False
    )

    assert final_state.phase == "complete"


# ============================================================================
# State Consistency Tests
# ============================================================================

def test_graph_maintains_state_consistency(test_data, agent_configs, tmp_path):
    """Test that graph maintains state consistency throughout execution"""
    team_lead, coding_agent = agent_configs

    state = create_initial_state(
        team_lead=team_lead,
        coding_agent=coding_agent,
        problem_statement="Predict target. Minimize MAE.",
        target_metric="mae",
        minimize_metric=True,
        max_iterations=2,
        results_dir=str(tmp_path / "results")
    )

    final_state = run_graph_experiment(
        state,
        test_data,
        enable_tracing=False
    )

    # Verify state consistency
    assert final_state.config.problem_statement == state.config.problem_statement
    assert final_state.config.target_metric == state.config.target_metric
    assert final_state.config.minimize_metric == state.config.minimize_metric

    # Verify team consistency
    assert final_state.team.team_lead.title == team_lead.title
    assert final_state.team.coding_agent.title == coding_agent.title


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
