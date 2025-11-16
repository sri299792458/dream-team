"""
Team-level dynamics for Dream Team framework.

Implements collective state and team-level emergence.
"""

from typing import List, Dict, Tuple, TYPE_CHECKING
import numpy as np
from .knowledge_state import KnowledgeGraph

if TYPE_CHECKING:
    from .agent import Agent


class Team:
    """Team with collective mathematical state"""

    def __init__(self, agents: List['Agent']):
        self.agents = agents
        self.iteration = 0

    def compute_diversity(self) -> float:
        """
        D(T,t) = 1/n(n-1) ∑_{i≠j} KL(θ_i || θ_j)

        Measures how different agents' attention distributions are
        High diversity = agents focused on different things
        Low diversity = agents all focused on same things
        """
        if len(self.agents) <= 1:
            return 1.0  # Maximum diversity (trivial case)

        n = len(self.agents)
        total_divergence = 0.0
        count = 0

        for i in range(n):
            for j in range(i + 1, n):  # Avoid double counting
                kl = self.agents[i].θ.kl_divergence(self.agents[j].θ)
                total_divergence += kl
                count += 1

        return total_divergence / count if count > 0 else 0.0

    def compute_collective_knowledge(self) -> KnowledgeGraph:
        """
        K_T(t) = ⋃_i K_i(t)

        Union of all agent knowledge graphs
        """
        collective = KnowledgeGraph()

        for agent in self.agents:
            # Add all concepts
            for concept in agent.K.concepts:
                if concept not in collective.concepts:
                    embedding = agent.K.embeddings.get(concept)
                    collective.add_concept(concept, embedding)

            # Add all edges (take maximum weight)
            for edge, weight in agent.K.edges.items():
                if edge in collective.edges:
                    collective.edges[edge] = max(collective.edges[edge], weight)
                else:
                    collective.edges[edge] = weight

        return collective

    def compute_problem_difficulty(self, problem: KnowledgeGraph) -> float:
        """
        Δ(T,P,t) = 1 - coverage

        How much of the problem is NOT covered by team
        """
        collective = self.compute_collective_knowledge()
        coverage = collective.compute_overlap(problem)

        return 1.0 - coverage

    def diagnose_state(self, performance_history: List[float], minimize: bool = True) -> str:
        """
        Diagnose what kind of help team needs

        Returns: "EXPLOITATION", "EXPLORATION", or "REFRAMING"
        """
        if len(performance_history) < 3:
            return "EXPLOITATION"  # Not enough data, keep exploring

        # Check trajectory (last 5 iterations)
        recent = performance_history[-min(5, len(performance_history)):]

        # Compute improvement
        if minimize:
            improvement = recent[0] - recent[-1]  # Positive if improving (lower is better)
        else:
            improvement = recent[-1] - recent[0]  # Positive if improving (higher is better)

        # Compute variance (stability)
        variance = np.var(recent)

        # Get diversity
        diversity = self.compute_diversity()

        # Decision logic
        if improvement > 0.01:
            # Making progress
            return "EXPLOITATION"

        elif abs(improvement) < 0.01 and variance < 0.001:
            # Plateaued (no improvement, stable)
            if diversity < 0.3:
                return "REFRAMING"  # Low diversity, need outside perspective
            else:
                return "EXPLORATION"  # Have diversity, just need to try more

        else:
            # Unclear / unstable
            return "EXPLORATION"

    def recommend_evolution(self, problem: KnowledgeGraph) -> List[Tuple['Agent', str]]:
        """
        Recommend which agents should evolve and how

        Returns: [(agent, evolution_type), ...]
        """
        recommendations = []

        for agent in self.agents:
            should_evolve, evo_type = agent.should_evolve(problem, self)
            if should_evolve:
                recommendations.append((agent, evo_type))

        return recommendations

    def get_uncovered_concepts(self, problem: KnowledgeGraph) -> List[str]:
        """
        Find concepts in problem not well covered by team

        Returns concepts sorted by importance
        """
        collective = self.compute_collective_knowledge()

        uncovered = []
        for concept in problem.concepts:
            if concept not in collective.concepts:
                importance = problem.concept_importance.get(concept, 1.0)
                uncovered.append((concept, importance))

        # Sort by importance
        uncovered.sort(key=lambda x: x[1], reverse=True)

        return [c for c, _ in uncovered]

    def get_highest_pressure_concepts(self, agent: 'Agent', problem: KnowledgeGraph, k: int = 3) -> List[Tuple[str, float]]:
        """
        Get concepts with highest specialization pressure for agent

        Returns: [(concept, pressure), ...]
        """
        pressures = []

        for concept in problem.concepts:
            pressure = agent.compute_specialization_pressure(concept, problem, self)
            pressures.append((concept, pressure))

        # Sort by pressure
        pressures.sort(key=lambda x: x[1], reverse=True)

        return pressures[:k]

    def update_all_dynamics(self, problem: KnowledgeGraph, learning_quality: Dict[str, float], dt: float = 0.1):
        """
        Update all agents' mathematical state

        Simulates time step in the dynamical system
        """
        time = self.iteration * dt

        for agent in self.agents:
            # Update attention based on pressures
            agent.update_attention(problem, self, dt, time)

            # Update depth based on attention
            agent.update_depth(learning_quality, dt)

        self.iteration += 1

    def get_state_summary(self) -> Dict:
        """Get summary of team state for logging"""
        diversity = self.compute_diversity()

        agent_states = []
        for agent in self.agents:
            agent_states.append({
                'title': agent.title,
                'gini': agent.δ.gini_coefficient(),
                'max_depth': agent.δ.max_depth(),
                'mean_depth': agent.δ.mean_depth(),
                'effectiveness': agent.contribution_effectiveness(),
                'top_concepts': agent.δ.top_concepts(3)
            })

        return {
            'diversity': diversity,
            'iteration': self.iteration,
            'agents': agent_states
        }
