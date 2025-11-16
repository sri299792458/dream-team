"""
Meeting module for Dream Team framework.

Implements team and individual meeting orchestration.
"""

from typing import List, Dict, Optional
from .agent import Agent
from .llm import get_llm
from datetime import datetime
import json
import os


class Meeting:
    """Base class for meetings"""

    def __init__(self, save_dir: Optional[str] = None, research_api=None):
        self.save_dir = save_dir
        self.transcript = []
        self.llm = get_llm()
        self.research_api = research_api  # Optional research API for on-demand paper search

    def add_message(self, agent_name: str, message: str):
        """Add message to transcript"""
        self.transcript.append({
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "message": message
        })

    def save(self, filename: str):
        """Save meeting transcript"""
        if not self.save_dir:
            return

        os.makedirs(self.save_dir, exist_ok=True)

        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.transcript, f, indent=2)

        print(f"💾 Meeting saved: {filepath}")


class TeamMeeting(Meeting):
    """Multi-agent team discussion"""

    def run(
        self,
        team_lead: Agent,
        team_members: List[Agent],
        agenda: str,
        num_rounds: int = 2,
        temperature: float = 0.7
    ) -> Dict:
        """
        Run a team meeting

        Returns: {
            "summary": str,
            "decisions": List[str],
            "action_items": List[str]
        }
        """

        print(f"\n📋 TEAM MEETING")
        print(f"   Lead: {team_lead.title}")
        print(f"   Members: {[m.title for m in team_members]}")
        print(f"   Rounds: {num_rounds}\n")

        # Opening: Team lead sets context
        opening_prompt = f"""You are leading a team meeting.

Agenda: {agenda}

As the team lead, open the meeting by:
1. Framing the problem
2. Asking key questions for the team to address
3. Setting expectations for the discussion

Keep it concise (2-3 paragraphs).
"""

        opening = self.llm.generate(
            opening_prompt,
            system_instruction=team_lead.prompt,
            temperature=temperature
        )

        self.add_message(team_lead.title, opening)
        print(f"💬 {team_lead.title}:")
        print(f"{opening}\n")

        # Discussion rounds
        for round_num in range(num_rounds):
            print(f"--- Round {round_num + 1}/{num_rounds} ---\n")

            # Each member contributes
            for member in team_members:
                # Build context from transcript
                context = self._build_context()

                member_prompt = f"""You are participating in a team meeting.

Agenda: {agenda}

Discussion so far:
{context}

Provide your input as {member.title}. Draw on your expertise and knowledge base.
Be specific and actionable. If relevant, cite papers or techniques you know.

Keep your response focused (1-2 paragraphs).
"""

                response = self.llm.generate(
                    member_prompt,
                    system_instruction=member.prompt,
                    temperature=temperature
                )

                self.add_message(member.title, response)
                member.meetings_participated += 1

                print(f"💬 {member.title}:")
                print(f"{response}\n")

                # On-demand paper search: search for papers related to what agent just proposed
                if self.research_api and len(response) > 50:  # Only if meaningful contribution
                    self._search_papers_for_proposal(member, response)

            # Team lead synthesizes
            synthesis_prompt = f"""You are synthesizing the team discussion.

Agenda: {agenda}

Discussion so far:
{self._build_context()}

As team lead, synthesize the key points and {'provide final recommendations' if round_num == num_rounds - 1 else 'guide the next round of discussion'}.
"""

            synthesis = self.llm.generate(
                synthesis_prompt,
                system_instruction=team_lead.prompt,
                temperature=temperature * 0.8  # Slightly more focused
            )

            self.add_message(team_lead.title, synthesis)
            print(f"💬 {team_lead.title} (synthesis):")
            print(f"{synthesis}\n")

        # Final summary
        summary_prompt = f"""Based on this meeting transcript, create a structured summary:

{self._build_context()}

Provide in JSON format:
{{
    "summary": "brief overview of discussion",
    "key_insights": ["insight 1", "insight 2", ...],
    "decisions": ["decision 1", "decision 2", ...],
    "action_items": ["action 1", "action 2", ...]
}}
"""

        summary = self.llm.generate_json(summary_prompt, temperature=0.3)

        return summary

    def _build_context(self, max_messages: int = 10) -> str:
        """Build context string from recent transcript"""
        recent = self.transcript[-max_messages:]
        return "\n\n".join([
            f"{msg['agent']}: {msg['message']}"
            for msg in recent
        ])

    def _search_papers_for_proposal(self, agent, proposal: str):
        """Search for papers related to agent's proposal and add to their knowledge base"""
        try:
            # Extract search query from proposal
            query_prompt = f"""Extract a concise search query (2-4 words) for finding relevant research papers.

Agent's proposal:
{proposal[:500]}

Generate a search query that captures the main technique/approach they're proposing.
Output ONLY the search query (2-4 words).
"""
            search_query = self.llm.generate(query_prompt, temperature=0.3).strip().strip('"\'')

            # Limit search query length
            if len(search_query) > 50:
                search_query = search_query[:50]

            print(f"   🔍 Searching papers for '{search_query}'...")

            # Search for recent, relevant papers
            raw_results = self.research_api.search(
                query=search_query,
                limit=5,
                year_range=(2018, 2025)  # Recent papers only
            )

            if raw_results:
                # Get existing papers to avoid duplicates
                existing_titles = [p.title for p in agent.knowledge_base.papers]

                # Add top 2 new papers
                added = 0
                for result in raw_results[:2]:
                    if result.title not in existing_titles:
                        paper = result.to_paper()
                        agent.knowledge_base.add_paper(paper)
                        print(f"      ✓ {paper.title[:60]}... ({paper.year})")
                        added += 1

                if added == 0:
                    print(f"      (all papers already in knowledge base)")
            else:
                print(f"      (no papers found)")

        except Exception as e:
            # Don't break meeting if search fails
            print(f"      ⚠️  Paper search failed: {e}")


class IndividualMeeting(Meeting):
    """One-on-one meeting with critic"""

    def run(
        self,
        agent: Agent,
        task: str,
        critic_agent: Optional[Agent] = None,
        num_iterations: int = 2,
        temperature: float = 0.7
    ) -> str:
        """
        Run individual meeting with iterative refinement

        Returns: Final output
        """

        print(f"\n👤 INDIVIDUAL MEETING")
        print(f"   Agent: {agent.title}")
        print(f"   Iterations: {num_iterations}\n")

        # Initial work
        work_prompt = f"""Task: {task}

Complete this task drawing on your expertise and knowledge base.
Be specific, detailed, and actionable.
"""

        output = self.llm.generate(
            work_prompt,
            system_instruction=agent.prompt,
            temperature=temperature
        )

        self.add_message(agent.title, output)
        agent.meetings_participated += 1

        print(f"💬 {agent.title} (initial):")
        print(f"{output[:200]}...\n")

        # Iterative refinement with critic
        if critic_agent:
            for iteration in range(num_iterations):
                print(f"--- Iteration {iteration + 1}/{num_iterations} ---\n")

                # Critic provides feedback
                critique_prompt = f"""You are reviewing work from {agent.title}.

Task: {task}

Their output:
{output}

Provide constructive criticism:
1. What's done well?
2. What's missing or could be improved?
3. Specific suggestions for refinement

Be brief but specific.
"""

                critique = self.llm.generate(
                    critique_prompt,
                    system_instruction=critic_agent.prompt,
                    temperature=temperature * 0.8
                )

                self.add_message(critic_agent.title, critique)
                print(f"💬 {critic_agent.title}:")
                print(f"{critique}\n")

                # Agent revises
                revision_prompt = f"""Task: {task}

Your previous output:
{output}

Feedback from {critic_agent.title}:
{critique}

Revise your work based on this feedback. Improve and extend as needed.
"""

                output = self.llm.generate(
                    revision_prompt,
                    system_instruction=agent.prompt,
                    temperature=temperature
                )

                self.add_message(agent.title, output)
                print(f"💬 {agent.title} (revised):")
                print(f"{output[:200]}...\n")

        return output
