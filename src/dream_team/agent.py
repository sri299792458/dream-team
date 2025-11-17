"""
Agent module for Dream Team framework.

Implements evolving agents with growing knowledge bases and mathematical state.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, TYPE_CHECKING
from datetime import datetime
import json
import numpy as np
from .knowledge_state import (
    KnowledgeGraph, AttentionDistribution, DepthMap,
    DynamicsState, extract_concepts_from_text
)

if TYPE_CHECKING:
    from .team import Team


@dataclass
class Paper:
    """Research paper representation"""
    title: str
    authors: List[str]
    year: int
    abstract: str
    key_findings: List[str] = field(default_factory=list)
    techniques: List[str] = field(default_factory=list)
    relevance_score: float = 0.0
    semantic_scholar_id: Optional[str] = None
    citation_count: int = 0
    applied: bool = False
    impact_notes: Optional[str] = None

    def to_dict(self):
        return {
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "key_findings": self.key_findings,
            "techniques": self.techniques,
            "relevance": self.relevance_score,
            "citation_count": self.citation_count,
            "applied": self.applied,
            "impact": self.impact_notes
        }


@dataclass
class KnowledgeBase:
    """Agent's accumulated knowledge"""
    domain_facts: List[str] = field(default_factory=list)
    papers: List[Paper] = field(default_factory=list)
    techniques_mastered: List[str] = field(default_factory=list)
    error_insights: List[str] = field(default_factory=list)
    successful_patterns: List[str] = field(default_factory=list)

    def add_paper(self, paper: Paper):
        """Add paper, avoiding duplicates"""
        if not any(p.title == paper.title for p in self.papers):
            self.papers.append(paper)

    def add_fact(self, fact: str, source: Optional[str] = None):
        """Add domain fact with optional citation"""
        fact_with_source = f"{fact} [Source: {source}]" if source else fact
        if fact_with_source not in self.domain_facts:
            self.domain_facts.append(fact_with_source)

    def add_technique(self, technique: str):
        if technique not in self.techniques_mastered:
            self.techniques_mastered.append(technique)

    def to_dict(self):
        return {
            "domain_facts": self.domain_facts,
            "papers": [p.to_dict() for p in self.papers],
            "techniques": self.techniques_mastered,
            "error_insights": self.error_insights,
            "successful_patterns": self.successful_patterns
        }

    def to_prompt_context(self, max_papers: int = 5) -> str:
        """Convert KB to string for LLM context"""
        parts = []

        if self.domain_facts:
            parts.append("## Domain Knowledge:")
            parts.extend([f"- {fact}" for fact in self.domain_facts[:10]])

        if self.papers:
            parts.append("\n## Research Papers:")
            for paper in self.papers[:max_papers]:
                parts.append(f"- {paper.title} ({paper.year})")
                if paper.key_findings:
                    parts.extend([f"  * {finding}" for finding in paper.key_findings])
                elif paper.abstract:
                    # Show abstract excerpt if no key_findings available
                    abstract_excerpt = paper.abstract[:200] + "..." if len(paper.abstract) > 200 else paper.abstract
                    parts.append(f"  Abstract: {abstract_excerpt}")

        if self.techniques_mastered:
            parts.append("\n## Techniques Mastered:")
            parts.extend([f"- {tech}" for tech in self.techniques_mastered])

        if self.successful_patterns:
            parts.append("\n## What Has Worked:")
            parts.extend([f"- {pattern}" for pattern in self.successful_patterns[-5:]])

        if self.error_insights:
            parts.append("\n## Known Issues:")
            parts.extend([f"- {insight}" for insight in self.error_insights[-5:]])

        return "\n".join(parts) if parts else "No knowledge accumulated yet."


@dataclass
class AgentSnapshot:
    """Snapshot of agent state at a point in time"""
    timestamp: str
    title: str
    expertise: str
    role: str
    specialization_depth: int
    knowledge_base_summary: Dict
    trigger_reason: Optional[str] = None
    # Mathematical state
    gini_coefficient: float = 0.0
    max_depth: float = 0.0
    top_concepts: List[Tuple[str, float]] = field(default_factory=list)


class Agent:
    """Evolving agent with mathematical state"""

    def __init__(
        self,
        title: str,
        expertise: str,
        goal: str,
        role: str,
        model: str = "gemini-2.0-flash-exp",
        specialization_depth: int = 0
    ):
        # Traditional attributes
        self.title = title
        self.expertise = expertise
        self.goal = goal
        self.role = role
        self.model = model
        self.specialization_depth = specialization_depth

        self.knowledge_base = KnowledgeBase()
        self.evolution_history: List[AgentSnapshot] = []

        # Track contributions
        self.meetings_participated = 0
        self.experiments_proposed = 0
        self.successful_contributions = 0

        # Mathematical state (NEW)
        self.K = KnowledgeGraph()  # K_i(t)
        self.θ = AttentionDistribution()  # θ_i(t)
        self.δ = DepthMap()  # δ_i(v,t)
        self.dynamics = DynamicsState()  # Track history

        # Contribution tracking for information theory
        self.contribution_history = {
            'proposed': [],
            'adopted': [],
            'information_gains': []
        }

        # Dynamics parameters
        self.α = 0.1  # Learning rate
        self.β = 0.05  # Forgetting rate
        self.λ_spec = 1.0  # Specialization weight
        self.λ_div = 0.5  # Diversification weight

        # Initialize from expertise
        self._initialize_from_expertise(expertise)

    def _initialize_from_expertise(self, expertise: str):
        """Extract initial concepts from expertise string"""
        concepts = extract_concepts_from_text(expertise, use_llm=False)

        if not concepts:
            # Fallback: use title
            concepts = {self.title.lower().replace(" ", "_")}

        # Add to knowledge graph with shallow depth
        for concept in concepts:
            self.K.add_concept(concept)
            self.δ[concept] = 0.3  # Initial shallow knowledge
            self.θ[concept] = 1.0

        self.θ.normalize()

    @property
    def prompt(self) -> str:
        """Generate system prompt incorporating knowledge base"""
        # Get mathematical state summary
        gini = self.δ.gini_coefficient()
        max_depth = self.δ.max_depth()
        top_concepts = self.δ.top_concepts(3)

        math_state = ""
        if top_concepts:
            math_state = f"\n## Current Focus:\n"
            math_state += f"Specialization level: {gini:.2f} (0=generalist, 1=specialist)\n"
            math_state += f"Deep expertise ({max_depth:.2f}) in:\n"
            for concept, depth in top_concepts:
                math_state += f"- {concept} (depth: {depth:.2f})\n"

        # Role-specific closing to activate relevant expertise
        if "Principal Investigator" in self.title or "Lead" in self.title:
            closing = """You are part of a research team solving data science challenges.

As the lead, ensure your approach embodies rigorous experimental methodology. Consider what constitutes sound scientific practice: how to validate hypotheses, establish baselines, make incremental progress, and learn systematically from results. Apply your full expertise to guide the team toward methodologically sound decisions."""
        else:
            closing = """You are part of a research team solving data science challenges.

When proposing approaches, consider both domain expertise and experimental rigor. Think about validation, incremental progress, and learning from previous work. Provide insightful, actionable contributions grounded in sound methodology."""

        base_prompt = f"""You are {self.title}.

Expertise: {self.expertise}

Goal: {self.goal}

Role: {self.role}

{math_state}

{self.knowledge_base.to_prompt_context()}

{closing}"""

        return base_prompt

    def snapshot(self, trigger_reason: Optional[str] = None) -> AgentSnapshot:
        """Take snapshot of current state"""
        return AgentSnapshot(
            timestamp=datetime.now().isoformat(),
            title=self.title,
            expertise=self.expertise,
            role=self.role,
            specialization_depth=self.specialization_depth,
            knowledge_base_summary=self.knowledge_base.to_dict(),
            trigger_reason=trigger_reason,
            gini_coefficient=self.δ.gini_coefficient(),
            max_depth=self.δ.max_depth(),
            top_concepts=self.δ.top_concepts(3)
        )

    def evolve(self, new_title: str, new_expertise: str, new_role: str, trigger_reason: str):
        """Evolve agent to new persona"""
        snapshot = self.snapshot(trigger_reason)
        self.evolution_history.append(snapshot)

        self.title = new_title
        self.expertise = new_expertise
        self.role = new_role
        self.specialization_depth += 1

        print(f"🧬 EVOLUTION: {snapshot.title} → {new_title}")
        print(f"   Reason: {trigger_reason}")
        print(f"   Depth: {self.specialization_depth}")
        print(f"   Gini: {snapshot.gini_coefficient:.2f} → (updating...)")

    def save(self, filepath: str):
        """Save agent state to JSON"""
        state = {
            "title": self.title,
            "expertise": self.expertise,
            "goal": self.goal,
            "role": self.role,
            "model": self.model,
            "specialization_depth": self.specialization_depth,
            "knowledge_base": self.knowledge_base.to_dict(),
            "evolution_history": [
                {
                    "timestamp": s.timestamp,
                    "title": s.title,
                    "expertise": s.expertise,
                    "role": s.role,
                    "depth": s.specialization_depth,
                    "trigger": s.trigger_reason,
                    "gini": s.gini_coefficient,
                    "max_depth": s.max_depth
                }
                for s in self.evolution_history
            ],
            "stats": {
                "meetings_participated": self.meetings_participated,
                "experiments_proposed": self.experiments_proposed,
                "successful_contributions": self.successful_contributions
            },
            "mathematical_state": {
                "concepts": list(self.K.concepts),
                "depths": self.δ.depths,
                "attention": self.θ.distribution,
                "gini": self.δ.gini_coefficient(),
                "max_depth": self.δ.max_depth()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'Agent':
        """Load agent from JSON"""
        with open(filepath, 'r') as f:
            state = json.load(f)

        agent = cls(
            title=state["title"],
            expertise=state["expertise"],
            goal=state["goal"],
            role=state["role"],
            model=state.get("model", "gemini-2.0-flash-exp"),
            specialization_depth=state.get("specialization_depth", 0)
        )

        # Restore knowledge base
        kb_data = state.get("knowledge_base", {})
        agent.knowledge_base.domain_facts = kb_data.get("domain_facts", [])
        agent.knowledge_base.techniques_mastered = kb_data.get("techniques", [])
        agent.knowledge_base.error_insights = kb_data.get("error_insights", [])
        agent.knowledge_base.successful_patterns = kb_data.get("successful_patterns", [])

        for p_data in kb_data.get("papers", []):
            paper = Paper(
                title=p_data["title"],
                authors=p_data.get("authors", []),
                year=p_data.get("year", 2024),
                abstract="",
                key_findings=p_data.get("key_findings", []),
                techniques=p_data.get("techniques", []),
                relevance_score=p_data.get("relevance", 0.0),
                applied=p_data.get("applied", False),
                impact_notes=p_data.get("impact")
            )
            agent.knowledge_base.papers.append(paper)

        # Restore mathematical state
        math_state = state.get("mathematical_state", {})
        if math_state:
            for concept in math_state.get("concepts", []):
                agent.K.add_concept(concept)

            depths = math_state.get("depths", {})
            for concept, depth in depths.items():
                agent.δ[concept] = depth

            attention = math_state.get("attention", {})
            for concept, att in attention.items():
                agent.θ[concept] = att
            agent.θ.normalize()

        # Restore stats
        stats = state.get("stats", {})
        agent.meetings_participated = stats.get("meetings_participated", 0)
        agent.experiments_proposed = stats.get("experiments_proposed", 0)
        agent.successful_contributions = stats.get("successful_contributions", 0)

        return agent

    # Information-theoretic methods

    def compute_coverage(self, problem_graph: KnowledgeGraph) -> float:
        """Ω_i(P,t) = simple overlap"""
        return self.K.compute_overlap(problem_graph)

    def compute_weighted_coverage(self, problem_graph: KnowledgeGraph) -> float:
        """Weighted coverage using depth: Ω_i(P,t) = ∑δ_i(v)·w_P(v) / ∑w_P(v)"""
        return self.K.compute_weighted_overlap(problem_graph, self.δ)

    def estimate_information_gain(self, contribution: str, team_response: str = "") -> float:
        """
        Estimate I(c_i) = H(before) - H(after)

        Simplified: based on adoption in team response
        """
        if not contribution.strip():
            return 0.0

        # Simple heuristic: check if ideas appear in team response
        contribution_words = set(contribution.lower().split())
        response_words = set(team_response.lower().split())

        overlap = len(contribution_words & response_words)
        total = len(contribution_words)

        if total == 0:
            return 0.0

        # Information gain proportional to adoption
        return min(1.0, overlap / total)

    def contribution_effectiveness(self) -> float:
        """What fraction of contributions were valuable?"""
        if len(self.contribution_history['information_gains']) == 0:
            return 0.5  # Neutral prior

        return np.mean(self.contribution_history['information_gains'])

    # Pressure functions (core of emergence)

    def compute_specialization_pressure(
        self,
        concept: str,
        problem: KnowledgeGraph,
        team: 'Team'
    ) -> float:
        """
        P_spec(i,v,t) = ∂I/∂δ_i(v) · Ω_i(P,t)

        How much would deepening in concept v increase my contribution?
        """
        if concept not in problem.concepts:
            return 0.0  # Not relevant

        current_depth = self.δ[concept]

        # Marginal return to depth (initially increasing, then diminishing)
        if current_depth < 0.4:
            marginal_return = 1.5 * (1.0 - current_depth)  # Increasing returns early
        elif current_depth < 0.7:
            marginal_return = 1.0 * (1.0 - current_depth)  # Linear
        else:
            marginal_return = 0.5 * (1.0 - current_depth)  # Diminishing returns

        # Weight by problem importance
        importance = problem.concept_importance.get(concept, 1.0)

        # Weight by current relevance
        relevance = self.compute_coverage(problem)

        return marginal_return * importance * (1.0 + relevance)

    def compute_diversification_pressure(
        self,
        problem: KnowledgeGraph,
        team: 'Team'
    ) -> float:
        """
        P_div(i,t) = -D(T,t) · (1 - coverage)

        High when team diversity is low AND I'm not covering problem
        """
        diversity = team.compute_diversity()

        # How much of problem am I covering?
        coverage = self.compute_weighted_coverage(problem)
        uncovered = 1.0 - coverage

        # Low diversity → high pressure to diversify
        # High uncovered → high pressure to explore
        return (1.0 - diversity) * uncovered

    # Dynamics updates

    def update_attention(
        self,
        problem: KnowledgeGraph,
        team: 'Team',
        dt: float = 0.1,
        time: float = 0.0
    ):
        """
        dθ/dt = λ_spec·P_spec + λ_div·P_div·U

        Update attention distribution based on pressures
        """
        new_θ = AttentionDistribution()

        # Diversification pressure (scalar)
        P_div = self.compute_diversification_pressure(problem, team)

        # All concepts (existing + problem)
        all_concepts = self.K.concepts | problem.concepts

        for concept in all_concepts:
            # Ensure concept in graph
            if concept not in self.K.concepts:
                self.K.add_concept(concept)

            # Specialization pressure for this concept
            P_spec = self.compute_specialization_pressure(concept, problem, team)

            # Uniform exploration
            U = 1.0 / max(len(all_concepts), 1)

            # Gradient
            dθ = self.λ_spec * P_spec + self.λ_div * P_div * U

            # Update with momentum (smooth changes)
            current = self.θ[concept]
            new_θ[concept] = current + dθ * dt

        # Normalize
        new_θ.normalize()
        self.θ = new_θ

        # Record state
        score = self.contribution_effectiveness()
        self.dynamics.record(self.θ, self.δ, score, time)

    def update_depth(
        self,
        learning_quality: Dict[str, float],
        dt: float = 0.1
    ):
        """
        dδ/dt = α·θ·quality - β·δ·(1-θ)

        Depth increases where attention is focused, decreases elsewhere
        """
        for concept in list(self.K.concepts):
            quality = learning_quality.get(concept, 0.5)

            # Learning (where attention is focused)
            learning = self.α * self.θ[concept] * quality

            # Forgetting (where attention isn't focused)
            forgetting = self.β * self.δ[concept] * (1.0 - self.θ[concept])

            # Update
            dδ = learning - forgetting
            new_depth = self.δ[concept] + dδ * dt
            self.δ[concept] = new_depth

            # Remove concepts that have been completely forgotten
            if self.δ[concept] < 0.01:
                self.K.concepts.discard(concept)

    # Evolution decision

    def should_evolve(self, problem: KnowledgeGraph, team: 'Team') -> Tuple[bool, str]:
        """
        Decide if evolution needed based on mathematical state

        Returns: (should_evolve, evolution_type)
        """
        gini = self.δ.gini_coefficient()
        effectiveness = self.contribution_effectiveness()

        # Compute pressures
        P_div = self.compute_diversification_pressure(problem, team)

        max_P_spec = 0.0
        for concept in problem.concepts:
            P_spec = self.compute_specialization_pressure(concept, problem, team)
            max_P_spec = max(max_P_spec, P_spec)

        # Decision logic
        if effectiveness < 0.3 and P_div > 0.5:
            return True, "DIVERSIFY"

        elif effectiveness > 0.6 and max_P_spec > 1.0 and gini < 0.6:
            return True, "SPECIALIZE"

        elif gini > 0.8 and P_div > 0.3:
            return True, "BROADEN"

        return False, "NONE"

    # Knowledge integration

    def add_paper_to_knowledge(self, paper: Paper):
        """Integrate paper into both traditional KB and mathematical graph"""
        # Traditional KB
        self.knowledge_base.add_paper(paper)

        # Extract concepts for mathematical graph
        concepts = extract_concepts_from_text(
            f"{paper.title} {' '.join(paper.key_findings)} {' '.join(paper.techniques)}",
            use_llm=False
        )

        for concept in concepts:
            if concept not in self.K.concepts:
                self.K.add_concept(concept)
                self.δ[concept] = 0.2  # Papers give shallow knowledge initially
            else:
                # Deepen existing knowledge
                self.δ[concept] = min(1.0, self.δ[concept] + 0.15)

            # Create edges between co-occurring concepts
            for other_concept in concepts:
                if concept != other_concept:
                    self.K.add_edge(concept, other_concept, weight=0.5)

    def record_contribution(self, contribution: str, adopted: bool, info_gain: float):
        """Record contribution for tracking"""
        self.contribution_history['proposed'].append(contribution)
        if adopted:
            self.contribution_history['adopted'].append(contribution)
        self.contribution_history['information_gains'].append(info_gain)
