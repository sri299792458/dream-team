"""
Evolution module for Dream Team framework.

Implements agent evolution triggers and evolution engine.
"""

from typing import List, Dict, Optional, Tuple
from .agent import Agent, Paper
from .llm import get_llm


class EvolutionTrigger:
    """Base class for evolution triggers"""

    def should_trigger(self, context: Dict) -> Tuple[bool, str]:
        """
        Check if evolution should trigger
        Returns: (should_trigger, reason)
        """
        raise NotImplementedError


class PerformancePlateauTrigger(EvolutionTrigger):
    """Trigger when performance plateaus"""

    def __init__(self, patience: int = 3, min_improvement: float = 0.01):
        self.patience = patience
        self.min_improvement = min_improvement

    def should_trigger(self, context: Dict) -> Tuple[bool, str]:
        metric_history = context.get("metric_history", [])

        if len(metric_history) < self.patience + 1:
            return False, ""

        recent = metric_history[-self.patience:]
        best_recent = min(recent) if context.get("minimize_metric") else max(recent)
        previous_best = min(metric_history[:-self.patience]) if context.get("minimize_metric") else max(metric_history[:-self.patience])

        improvement = abs(best_recent - previous_best) / (abs(previous_best) + 1e-10)

        if improvement < self.min_improvement:
            return True, f"Performance plateaued (improvement: {improvement:.4f} < {self.min_improvement})"

        return False, ""


class ErrorPatternTrigger(EvolutionTrigger):
    """Trigger when specific error patterns detected"""

    def __init__(self, error_threshold: float = 0.3):
        self.error_threshold = error_threshold

    def should_trigger(self, context: Dict) -> Tuple[bool, str]:
        error_analysis = context.get("error_analysis", {})

        # Check if any error category exceeds threshold
        for category, proportion in error_analysis.items():
            if proportion > self.error_threshold:
                return True, f"High error rate in {category}: {proportion:.2%}"

        return False, ""


class KnowledgeGapTrigger(EvolutionTrigger):
    """Trigger when agents express uncertainty"""

    def should_trigger(self, context: Dict) -> Tuple[bool, str]:
        meeting_transcript = context.get("last_meeting_transcript", "")

        # Simple heuristic: look for uncertainty markers
        uncertainty_markers = [
            "don't know", "not sure", "uncertain", "unclear",
            "need expertise", "require knowledge", "lack understanding"
        ]

        for marker in uncertainty_markers:
            if marker.lower() in meeting_transcript.lower():
                return True, f"Knowledge gap detected: agents expressed uncertainty"

        return False, ""


class EvolutionEngine:
    """Orchestrates agent evolution"""

    def __init__(self, llm=None, triggers: List[EvolutionTrigger] = None):
        self.llm = llm or get_llm()
        self.triggers = triggers or [
            PerformancePlateauTrigger(),
            ErrorPatternTrigger(),
            KnowledgeGapTrigger()
        ]

    def check_triggers(self, context: Dict) -> List[Tuple[str, str]]:
        """
        Check all triggers
        Returns: List of (trigger_name, reason)
        """
        triggered = []
        for trigger in self.triggers:
            should_trigger, reason = trigger.should_trigger(context)
            if should_trigger:
                triggered.append((trigger.__class__.__name__, reason))

        return triggered

    def propose_evolution(
        self,
        agent: Agent,
        context: Dict,
        papers: List[Paper] = None
    ) -> Dict:
        """
        Use LLM to propose agent evolution

        Returns: {
            "new_title": str,
            "new_expertise": str,
            "new_role": str,
            "reasoning": str
        }
        """

        papers_context = ""
        if papers:
            papers_context = "\n## Relevant Research:\n"
            for paper in papers[:3]:
                papers_context += f"- {paper.title} ({paper.year})\n"
                for finding in paper.key_findings[:2]:
                    papers_context += f"  * {finding}\n"

        error_context = ""
        if "error_analysis" in context:
            error_context = "\n## Error Analysis:\n"
            for category, proportion in context["error_analysis"].items():
                error_context += f"- {category}: {proportion:.1%}\n"

        performance_context = ""
        if "metric_history" in context:
            metrics = context["metric_history"][-5:]
            performance_context = f"\n## Recent Performance:\n{metrics}\n"

        prompt = f"""You are a meta-agent responsible for evolving team members to solve challenges more effectively.

## Current Agent:
Title: {agent.title}
Expertise: {agent.expertise}
Role: {agent.role}
Specialization Depth: {agent.specialization_depth}

## Agent's Current Knowledge:
{agent.knowledge_base.to_prompt_context(max_papers=3)}

## Problem Context:
{context.get('problem_description', 'N/A')}

{performance_context}
{error_context}
{papers_context}

## Evolution Request:
The agent needs to evolve to address current challenges. Propose an evolved persona that:
1. Retains core identity but deepens specialization
2. Incorporates insights from research papers (if provided)
3. Addresses specific performance gaps or error patterns
4. Provides concrete, actionable expertise

Respond in JSON format:
{{
    "new_title": "specific, descriptive title",
    "new_expertise": "detailed expertise areas with specifics",
    "new_role": "concrete responsibilities and contributions",
    "reasoning": "why this evolution helps",
    "knowledge_to_add": {{
        "domain_facts": ["fact 1", "fact 2", ...],
        "techniques": ["technique 1", "technique 2", ...]
    }}
}}
"""

        evolution_proposal = self.llm.generate_json(prompt, temperature=0.7)
        return evolution_proposal

    def evolve_agent(
        self,
        agent: Agent,
        context: Dict,
        papers: List[Paper] = None,
        trigger_reason: str = "Manual evolution"
    ) -> Agent:
        """
        Evolve an agent based on context and papers
        """

        print(f"\n🧬 Evolving {agent.title}...")
        print(f"   Reason: {trigger_reason}")

        # Get evolution proposal from LLM
        proposal = self.propose_evolution(agent, context, papers)

        print(f"   Proposed: {proposal['new_title']}")
        print(f"   Reasoning: {proposal['reasoning']}")

        # Apply evolution
        agent.evolve(
            new_title=proposal["new_title"],
            new_expertise=proposal["new_expertise"],
            new_role=proposal["new_role"],
            trigger_reason=trigger_reason
        )

        # Add papers to knowledge base
        if papers:
            for paper in papers:
                agent.knowledge_base.add_paper(paper)

        # Add new knowledge from proposal
        if "knowledge_to_add" in proposal:
            for fact in proposal["knowledge_to_add"].get("domain_facts", []):
                agent.knowledge_base.add_fact(fact, source="Evolution synthesis")

            for technique in proposal["knowledge_to_add"].get("techniques", []):
                agent.knowledge_base.add_technique(technique)

        print(f"   ✅ Evolution complete. Depth: {agent.specialization_depth}\n")

        return agent

    def create_specialist_agent(
        self,
        need_description: str,
        context: Dict,
        papers: List[Paper] = None,
        base_model: str = "gemini-2.5-flash"
    ) -> Agent:
        """
        Create a brand new specialist agent from scratch
        """

        print(f"\n🌟 Creating new specialist agent...")
        print(f"   Need: {need_description}")

        papers_context = ""
        if papers:
            papers_context = "\n## Relevant Research:\n"
            for paper in papers[:3]:
                papers_context += f"- {paper.title} ({paper.year}): {paper.abstract[:200]}...\n"

        prompt = f"""You are designing a new expert agent to join a research team.

## Team Need:
{need_description}

## Problem Context:
{context.get('problem_description', 'N/A')}

{papers_context}

Design an expert agent with specific, actionable expertise to fill this need.

Respond in JSON format:
{{
    "title": "specific job title",
    "expertise": "detailed areas of expertise",
    "goal": "what this agent aims to achieve",
    "role": "specific responsibilities",
    "initial_knowledge": {{
        "domain_facts": ["fact 1", "fact 2", ...],
        "techniques": ["technique 1", "technique 2", ...]
    }}
}}
"""

        agent_spec = self.llm.generate_json(prompt, temperature=0.8)

        # Create agent
        agent = Agent(
            title=agent_spec["title"],
            expertise=agent_spec["expertise"],
            goal=agent_spec["goal"],
            role=agent_spec["role"],
            model=base_model,
            specialization_depth=0
        )

        # Add initial knowledge
        if "initial_knowledge" in agent_spec:
            for fact in agent_spec["initial_knowledge"].get("domain_facts", []):
                agent.knowledge_base.add_fact(fact, source="Initial creation")

            for technique in agent_spec["initial_knowledge"].get("techniques", []):
                agent.knowledge_base.add_technique(technique)

        # Add papers
        if papers:
            for paper in papers:
                agent.knowledge_base.add_paper(paper)

        print(f"   ✅ Created: {agent.title}\n")

        return agent
