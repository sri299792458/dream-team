"""
Agent module for Dream Team framework.

Implements evolving agents with growing knowledge bases.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import json


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


class Agent:
    """Evolving agent with growing knowledge base"""

    def __init__(
        self,
        title: str,
        expertise: str,
        goal: str,
        role: str,
        model: str = "gemini-2.0-flash-exp",
        specialization_depth: int = 0
    ):
        self.title = title
        self.expertise = expertise
        self.goal = goal
        self.role = role
        self.model = model
        self.specialization_depth = specialization_depth

        self.knowledge_base = KnowledgeBase()
        self.evolution_history: List[AgentSnapshot] = []

        # Track agent's contributions
        self.meetings_participated = 0
        self.experiments_proposed = 0
        self.successful_contributions = 0

    @property
    def prompt(self) -> str:
        """Generate system prompt incorporating knowledge base"""
        base_prompt = f"""You are {self.title}.

Expertise: {self.expertise}

Goal: {self.goal}

Role: {self.role}

{self.knowledge_base.to_prompt_context()}

You are part of a research team solving data science challenges. Draw on your expertise and knowledge base to provide insightful, actionable contributions."""

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
            trigger_reason=trigger_reason
        )

    def evolve(self, new_title: str, new_expertise: str, new_role: str, trigger_reason: str):
        """Evolve agent to new persona"""
        # Save current state
        snapshot = self.snapshot(trigger_reason)
        self.evolution_history.append(snapshot)

        # Evolve
        self.title = new_title
        self.expertise = new_expertise
        self.role = new_role
        self.specialization_depth += 1

        print(f"🧬 EVOLUTION: {snapshot.title} → {new_title}")
        print(f"   Reason: {trigger_reason}")
        print(f"   Depth: {self.specialization_depth}")

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
                    "trigger": s.trigger_reason
                }
                for s in self.evolution_history
            ],
            "stats": {
                "meetings_participated": self.meetings_participated,
                "experiments_proposed": self.experiments_proposed,
                "successful_contributions": self.successful_contributions
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

        # Restore papers
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

        # Restore stats
        stats = state.get("stats", {})
        agent.meetings_participated = stats.get("meetings_participated", 0)
        agent.experiments_proposed = stats.get("experiments_proposed", 0)
        agent.successful_contributions = stats.get("successful_contributions", 0)

        return agent
