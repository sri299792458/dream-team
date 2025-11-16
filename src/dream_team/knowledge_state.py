"""
Mathematical state representations for Dream Team agents.

Implements the core mathematical framework:
- Knowledge graphs: K_i = (V, E, W)
- Attention distributions: θ_i : V → [0,1]
- Depth maps: δ_i : V → [0,1]
"""

from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple
import numpy as np
from collections import defaultdict


@dataclass
class KnowledgeGraph:
    """
    Mathematical knowledge graph K_i = (V, E, W)

    V: Set of concept nodes
    E: Set of edges (relations)
    W: Edge weights
    """

    concepts: Set[str] = field(default_factory=set)  # V_i
    edges: Dict[Tuple[str, str], float] = field(default_factory=dict)  # E_i with weights
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)  # φ(v)
    concept_importance: Dict[str, float] = field(default_factory=dict)  # w(v)

    def add_concept(self, concept: str, embedding: np.ndarray = None, importance: float = 1.0):
        """Add concept to knowledge graph"""
        self.concepts.add(concept)
        self.concept_importance[concept] = importance

        if embedding is not None:
            self.embeddings[concept] = embedding
        elif concept not in self.embeddings:
            # Initialize random embedding if not provided
            self.embeddings[concept] = np.random.randn(128) / np.sqrt(128)  # d=128, normalized

    def add_edge(self, concept1: str, concept2: str, weight: float = 1.0):
        """Add relation between concepts"""
        if concept1 in self.concepts and concept2 in self.concepts:
            self.edges[(concept1, concept2)] = weight
            self.edges[(concept2, concept1)] = weight  # Undirected

    def compute_overlap(self, other: 'KnowledgeGraph') -> float:
        """
        Compute Ω_i(P,t) = |V_i ∩ V_P| / |V_P|

        Simple overlap coefficient
        """
        if len(other.concepts) == 0:
            return 0.0

        intersection = self.concepts & other.concepts
        return len(intersection) / len(other.concepts)

    def compute_weighted_overlap(self, other: 'KnowledgeGraph', depth_map: 'DepthMap') -> float:
        """
        Weighted coverage: Ω_i(P,t) = ∑_{v∈V_P} δ_i(v)·w_P(v) / ∑w_P(v)

        Takes into account both depth of knowledge and importance of concepts
        """
        if len(other.concepts) == 0:
            return 0.0

        total_weight = sum(other.concept_importance.get(v, 1.0) for v in other.concepts)
        if total_weight == 0:
            return 0.0

        covered_weight = sum(
            depth_map[v] * other.concept_importance.get(v, 1.0)
            for v in other.concepts
        )

        return covered_weight / total_weight

    def compute_structural_similarity(self, other: 'KnowledgeGraph') -> float:
        """
        Measure structural similarity between graphs

        Uses embeddings to find similar concepts even with different names
        """
        if not self.embeddings or not other.embeddings:
            return self.compute_overlap(other)

        total_sim = 0.0
        count = 0

        for v1 in self.concepts:
            if v1 not in self.embeddings:
                continue

            max_sim = 0.0
            for v2 in other.concepts:
                if v2 not in other.embeddings:
                    continue

                # Cosine similarity
                sim = np.dot(self.embeddings[v1], other.embeddings[v2])
                sim /= (np.linalg.norm(self.embeddings[v1]) * np.linalg.norm(other.embeddings[v2]) + 1e-10)
                max_sim = max(max_sim, sim)

            total_sim += max_sim
            count += 1

        return total_sim / count if count > 0 else 0.0


@dataclass
class AttentionDistribution:
    """
    θ_i(t) : V → [0,1] with ∑θ(v) = 1

    Represents how agent allocates cognitive resources across concepts
    """

    distribution: Dict[str, float] = field(default_factory=dict)

    def __getitem__(self, concept: str) -> float:
        return self.distribution.get(concept, 0.0)

    def __setitem__(self, concept: str, value: float):
        self.distribution[concept] = max(0.0, value)

    def normalize(self):
        """Ensure ∑θ(v) = 1"""
        total = sum(self.distribution.values())
        if total > 0:
            for concept in self.distribution:
                self.distribution[concept] /= total
        elif self.distribution:
            # Uniform if all zeros
            uniform = 1.0 / len(self.distribution)
            for concept in self.distribution:
                self.distribution[concept] = uniform

    def entropy(self) -> float:
        """
        Shannon entropy H(θ) = -∑θ(v)log(θ(v))

        Measures how focused vs distributed attention is
        """
        return -sum(
            p * np.log(p + 1e-10)
            for p in self.distribution.values()
            if p > 0
        )

    def kl_divergence(self, other: 'AttentionDistribution') -> float:
        """
        KL(self || other) = ∑θ_i(v)log(θ_i(v)/θ_j(v))

        Measures how different two attention distributions are
        """
        concepts = set(self.distribution.keys()) | set(other.distribution.keys())
        return sum(
            self[v] * np.log((self[v] + 1e-10) / (other[v] + 1e-10))
            for v in concepts
            if self[v] > 0
        )

    def top_concepts(self, k: int = 3) -> List[Tuple[str, float]]:
        """Get top k concepts by attention"""
        sorted_concepts = sorted(
            self.distribution.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_concepts[:k]


@dataclass
class DepthMap:
    """
    δ_i : V → [0,1]

    Represents depth of understanding for each concept
    0 = heard of it
    1 = world expert
    """

    depths: Dict[str, float] = field(default_factory=dict)

    def __getitem__(self, concept: str) -> float:
        return self.depths.get(concept, 0.0)

    def __setitem__(self, concept: str, depth: float):
        self.depths[concept] = np.clip(depth, 0.0, 1.0)

    def mean_depth(self) -> float:
        """Average depth across all concepts"""
        if not self.depths:
            return 0.0
        return np.mean(list(self.depths.values()))

    def max_depth(self) -> float:
        """Maximum depth in any concept (specialization indicator)"""
        if not self.depths:
            return 0.0
        return max(self.depths.values())

    def gini_coefficient(self) -> float:
        """
        Gini coefficient: measure of specialization

        0 = perfectly uniform (generalist)
        1 = all depth in one concept (specialist)
        """
        if not self.depths:
            return 0.0

        depths = sorted(self.depths.values())
        n = len(depths)

        if n == 0 or sum(depths) == 0:
            return 0.0

        index = np.arange(1, n + 1)
        return (2 * np.sum(index * depths)) / (n * np.sum(depths)) - (n + 1) / n

    def top_concepts(self, k: int = 3) -> List[Tuple[str, float]]:
        """Get top k concepts by depth"""
        sorted_concepts = sorted(
            self.depths.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_concepts[:k]


class DynamicsState:
    """
    Tracks agent dynamics over time

    Used for computing gradients and pressures
    """

    def __init__(self):
        self.attention_history: List[AttentionDistribution] = []
        self.depth_history: List[DepthMap] = []
        self.contribution_scores: List[float] = []
        self.timestamps: List[float] = []

    def record(self, attention: AttentionDistribution, depth: DepthMap, score: float, time: float):
        """Record state at timestamp"""
        # Deep copy to avoid mutation
        import copy
        self.attention_history.append(copy.deepcopy(attention))
        self.depth_history.append(copy.deepcopy(depth))
        self.contribution_scores.append(score)
        self.timestamps.append(time)

    def compute_attention_velocity(self) -> Dict[str, float]:
        """
        Compute dθ/dt for each concept

        Returns rate of change in attention allocation
        """
        if len(self.attention_history) < 2:
            return {}

        recent = self.attention_history[-1]
        previous = self.attention_history[-2]
        dt = self.timestamps[-1] - self.timestamps[-2]

        if dt == 0:
            return {}

        velocity = {}
        concepts = set(recent.distribution.keys()) | set(previous.distribution.keys())

        for concept in concepts:
            dθ = recent[concept] - previous[concept]
            velocity[concept] = dθ / dt

        return velocity

    def compute_depth_velocity(self) -> Dict[str, float]:
        """
        Compute dδ/dt for each concept

        Returns rate of learning
        """
        if len(self.depth_history) < 2:
            return {}

        recent = self.depth_history[-1]
        previous = self.depth_history[-2]
        dt = self.timestamps[-1] - self.timestamps[-2]

        if dt == 0:
            return {}

        velocity = {}
        concepts = set(recent.depths.keys()) | set(previous.depths.keys())

        for concept in concepts:
            dδ = recent[concept] - previous[concept]
            velocity[concept] = dδ / dt

        return velocity

    def estimate_learning_rate(self) -> float:
        """
        Estimate α (learning rate parameter)

        Uses recent history to fit dδ/dt = α·θ·quality - β·δ·(1-θ)
        """
        if len(self.depth_history) < 3:
            return 0.1  # Default

        # Simple estimation: average rate of depth increase
        depth_increases = []

        for i in range(1, len(self.depth_history)):
            prev_depth = self.depth_history[i-1].mean_depth()
            curr_depth = self.depth_history[i].mean_depth()
            dt = self.timestamps[i] - self.timestamps[i-1]

            if dt > 0:
                depth_increases.append((curr_depth - prev_depth) / dt)

        if depth_increases:
            return np.mean([d for d in depth_increases if d > 0])

        return 0.1


def extract_concepts_from_text(text: str, use_llm: bool = False) -> Set[str]:
    """
    Extract concepts from text

    Args:
        text: Text to extract from
        use_llm: Whether to use LLM for extraction (more accurate but slower)

    Returns:
        Set of concept strings
    """
    if use_llm:
        # Use LLM for better extraction
        from .llm import get_llm
        llm = get_llm()

        prompt = f"""Extract key technical concepts from this text.

Text: {text[:500]}

Output a JSON array of concepts (single words or short phrases):
["concept1", "concept2", ...]

Focus on technical terms, methodologies, and domain areas.
"""

        try:
            result = llm.generate_json(prompt, temperature=0.3)
            if isinstance(result, list):
                return set(c.lower() for c in result)
            elif isinstance(result, dict) and 'concepts' in result:
                return set(c.lower() for c in result['concepts'])
        except:
            pass  # Fall back to heuristic

    # Heuristic extraction
    concepts = set()

    # Simple: extract meaningful words
    words = text.lower().split()

    # Common technical suffixes/patterns
    tech_patterns = ['learning', 'model', 'algorithm', 'method', 'theory',
                     'analysis', 'optimization', 'network', 'system']

    for word in words:
        # Remove punctuation
        word = ''.join(c for c in word if c.isalnum() or c == '_')

        # Keep if it's long enough or matches pattern
        if len(word) > 6 or any(pattern in word for pattern in tech_patterns):
            concepts.add(word)

    # Look for multi-word phrases (simple bigrams)
    for i in range(len(words) - 1):
        w1 = ''.join(c for c in words[i] if c.isalnum())
        w2 = ''.join(c for c in words[i+1] if c.isalnum())

        if len(w1) > 4 and len(w2) > 4:
            concepts.add(f"{w1}_{w2}")

    return concepts
