"""
LangGraph state schema for Dream Team.

Defines the state that flows through the graph, with proper serialization
for mathematical framework components (K, θ, δ).
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from typing_extensions import NotRequired
import operator


class SerializedKnowledgeGraph(TypedDict):
    """Serializable knowledge graph"""
    concepts: List[str]
    edges: List[tuple[str, str, float]]  # (concept1, concept2, weight)
    embeddings: Dict[str, List[float]]  # Concept -> embedding as list
    concept_importance: Dict[str, float]


class SerializedAttentionDistribution(TypedDict):
    """Serializable attention distribution"""
    distribution: Dict[str, float]


class SerializedDepthMap(TypedDict):
    """Serializable depth map"""
    depths: Dict[str, float]


class SerializedDynamicsState(TypedDict):
    """Serializable dynamics state"""
    attention_history: List[SerializedAttentionDistribution]
    depth_history: List[SerializedDepthMap]
    contribution_scores: List[float]
    timestamps: List[float]


class SerializedAgent(TypedDict):
    """Serializable agent state"""
    title: str
    expertise: str
    goal: str
    role: str
    model: str
    specialization_depth: int
    meetings_participated: int

    # Knowledge base
    papers: List[Dict[str, Any]]
    domain_facts: List[str]
    techniques: List[str]
    successful_patterns: List[str]
    error_insights: List[str]

    # Mathematical state (serialized)
    K: SerializedKnowledgeGraph
    θ: SerializedAttentionDistribution
    δ: SerializedDepthMap
    dynamics: SerializedDynamicsState

    # Evolution history
    evolution_history: List[Dict[str, Any]]


class IterationResult(TypedDict):
    """Results from a single iteration"""
    iteration: int
    approach: str
    results: Dict[str, Any]
    metrics: Dict[str, Any]
    context: NotRequired[Dict[str, str]]
    meeting: NotRequired[Dict[str, Any]]
    agents_snapshot: List[str]


class DreamTeamState(TypedDict):
    """
    Main state for LangGraph Dream Team workflow.

    This state is passed between nodes and persisted at checkpoints.
    """
    # Problem definition
    problem_statement: str
    target_metric: str
    minimize_metric: bool
    target_score: NotRequired[Optional[float]]

    # Data context
    data_context: Dict[str, Any]  # DataFrames and available variables
    column_schemas: NotRequired[Dict[str, List[str]]]  # Extracted column names

    # Team composition
    team_lead: SerializedAgent
    team_members: Annotated[List[SerializedAgent], operator.add]  # Allows adding members
    coding_agent: SerializedAgent

    # Iteration state
    iteration: int
    max_iterations: int
    bootstrap_completed: bool

    # Experiment history
    experiment_history: Annotated[List[IterationResult], operator.add]

    # Current iteration working state
    current_approach: NotRequired[str]
    current_code: NotRequired[str]
    current_results: NotRequired[Dict[str, Any]]
    current_metrics: NotRequired[Dict[str, Any]]
    planning_context: NotRequired[str]
    coding_context: NotRequired[str]
    meeting_agenda: NotRequired[str]
    meeting_messages: NotRequired[List[Dict[str, Any]]]
    meeting_papers: NotRequired[List[Dict[str, Any]]]

    # Best tracking
    best_metric: NotRequired[Optional[float]]
    best_iteration: NotRequired[Optional[int]]

    # Control flow flags
    goal_achieved: bool
    should_evolve: bool
    error_count: int

    # Execution context (persistent variables between iterations)
    executor_vars: NotRequired[Dict[str, Any]]

    # Directories
    results_dir: str
    meetings_dir: str
    code_dir: str


# Serialization helpers

def serialize_knowledge_graph(K) -> SerializedKnowledgeGraph:
    """Convert KnowledgeGraph to serializable dict"""
    from .knowledge_state import KnowledgeGraph

    edges_list = [(c1, c2, w) for (c1, c2), w in K.edges.items()]
    embeddings_dict = {c: emb.tolist() for c, emb in K.embeddings.items()}

    return {
        "concepts": list(K.concepts),
        "edges": edges_list,
        "embeddings": embeddings_dict,
        "concept_importance": K.concept_importance.copy()
    }


def deserialize_knowledge_graph(data: SerializedKnowledgeGraph):
    """Convert serialized dict back to KnowledgeGraph"""
    from .knowledge_state import KnowledgeGraph
    import numpy as np

    K = KnowledgeGraph()
    K.concepts = set(data["concepts"])
    K.edges = {(c1, c2): w for c1, c2, w in data["edges"]}
    K.embeddings = {c: np.array(emb) for c, emb in data["embeddings"].items()}
    K.concept_importance = data["concept_importance"].copy()

    return K


def serialize_attention_distribution(θ) -> SerializedAttentionDistribution:
    """Convert AttentionDistribution to serializable dict"""
    return {"distribution": θ.distribution.copy()}


def deserialize_attention_distribution(data: SerializedAttentionDistribution):
    """Convert serialized dict back to AttentionDistribution"""
    from .knowledge_state import AttentionDistribution

    θ = AttentionDistribution()
    θ.distribution = data["distribution"].copy()
    return θ


def serialize_depth_map(δ) -> SerializedDepthMap:
    """Convert DepthMap to serializable dict"""
    return {"depths": δ.depths.copy()}


def deserialize_depth_map(data: SerializedDepthMap):
    """Convert serialized dict back to DepthMap"""
    from .knowledge_state import DepthMap

    δ = DepthMap()
    δ.depths = data["depths"].copy()
    return δ


def serialize_dynamics_state(dynamics) -> SerializedDynamicsState:
    """Convert DynamicsState to serializable dict"""
    return {
        "attention_history": [serialize_attention_distribution(a) for a in dynamics.attention_history],
        "depth_history": [serialize_depth_map(d) for d in dynamics.depth_history],
        "contribution_scores": dynamics.contribution_scores.copy(),
        "timestamps": dynamics.timestamps.copy()
    }


def deserialize_dynamics_state(data: SerializedDynamicsState):
    """Convert serialized dict back to DynamicsState"""
    from .knowledge_state import DynamicsState

    dynamics = DynamicsState()
    dynamics.attention_history = [deserialize_attention_distribution(a) for a in data["attention_history"]]
    dynamics.depth_history = [deserialize_depth_map(d) for d in data["depth_history"]]
    dynamics.contribution_scores = data["contribution_scores"].copy()
    dynamics.timestamps = data["timestamps"].copy()

    return dynamics


def serialize_agent(agent) -> SerializedAgent:
    """Convert Agent to serializable dict"""
    from .agent import Agent

    return {
        "title": agent.title,
        "expertise": agent.expertise,
        "goal": agent.goal,
        "role": agent.role,
        "model": agent.model,
        "specialization_depth": agent.specialization_depth,
        "meetings_participated": agent.meetings_participated,
        "papers": [p.to_dict() for p in agent.knowledge_base.papers],
        "domain_facts": agent.knowledge_base.domain_facts.copy(),
        "techniques": agent.knowledge_base.techniques_mastered.copy(),
        "successful_patterns": agent.knowledge_base.successful_patterns.copy(),
        "error_insights": agent.knowledge_base.error_insights.copy(),
        "K": serialize_knowledge_graph(agent.K),
        "θ": serialize_attention_distribution(agent.θ),
        "δ": serialize_depth_map(agent.δ),
        "dynamics": serialize_dynamics_state(agent.dynamics),
        "evolution_history": [snap.__dict__ for snap in agent.evolution_history]
    }


def deserialize_agent(data: SerializedAgent):
    """Convert serialized dict back to Agent"""
    from .agent import Agent, Paper, KnowledgeBase, AgentSnapshot

    agent = Agent(
        title=data["title"],
        expertise=data["expertise"],
        goal=data["goal"],
        role=data["role"],
        model=data["model"]
    )

    agent.specialization_depth = data["specialization_depth"]
    agent.meetings_participated = data["meetings_participated"]

    # Restore knowledge base
    agent.knowledge_base.papers = [Paper(**p) for p in data["papers"]]
    agent.knowledge_base.domain_facts = data["domain_facts"].copy()
    agent.knowledge_base.techniques_mastered = data["techniques"].copy()
    agent.knowledge_base.successful_patterns = data["successful_patterns"].copy()
    agent.knowledge_base.error_insights = data["error_insights"].copy()

    # Restore mathematical state
    agent.K = deserialize_knowledge_graph(data["K"])
    agent.θ = deserialize_attention_distribution(data["θ"])
    agent.δ = deserialize_depth_map(data["δ"])
    agent.dynamics = deserialize_dynamics_state(data["dynamics"])

    # Restore evolution history for continuity across checkpoints
    agent.evolution_history = [
        AgentSnapshot(**snapshot_dict)
        for snapshot_dict in data.get("evolution_history", [])
    ]

    return agent
