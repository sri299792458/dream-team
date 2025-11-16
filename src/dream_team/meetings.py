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

                # ReAct loop: Reasoning + Acting iteratively
                if self.research_api:
                    response = self._react_proposal(member, agenda, context, temperature)
                else:
                    # Fallback: simple proposal without ReAct
                    response = self.llm.generate(
                        f"""You are participating in a team meeting.

Agenda: {agenda}

Discussion so far:
{context}

Provide your input as {member.title}. Draw on your expertise.
Keep it concise (1-2 paragraphs).
""",
                        system_instruction=member.prompt,
                        temperature=temperature
                    )

                self.add_message(member.title, response)
                member.meetings_participated += 1

                print(f"💬 {member.title}:")
                print(f"{response}\n")

            # Team lead synthesizes
            is_final_round = (round_num == num_rounds - 1)

            if is_final_round:
                # Final synthesis: make decisions
                synthesis_prompt = f"""You are synthesizing the team discussion to make FINAL DECISIONS.

Agenda: {agenda}

Discussion so far:
{self._build_context()}

As team lead, synthesize the team's proposals into a FINAL DECISION and action plan.

IMPORTANT:
- Make FINAL DECISIONS, do NOT ask clarifying questions
- Synthesize what the team proposed into a clear action plan
- Be decisive and specific about what to implement

Keep it focused (2-3 paragraphs).
"""
            else:
                # Intermediate synthesis: guide discussion
                synthesis_prompt = f"""You are synthesizing the team discussion.

Agenda: {agenda}

Discussion so far:
{self._build_context()}

As team lead, synthesize the key points and guide the next round of discussion.
Highlight areas of agreement and any gaps that need more exploration.

Keep it concise (1-2 paragraphs).
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

    def _react_proposal(self, agent, agenda: str, context: str, temperature: float, max_steps: int = 2) -> str:
        """
        ReAct loop: agent reasons and acts iteratively before final proposal.

        Pattern:
        1. Thought: Based on my expertise, I propose X because...
        2. Action: Search papers to ground/support this idea
        3. Observation: Papers found to support this
        4. (Repeat 1-3 to build up grounded proposal)
        5. Final Answer: Proposal with citations as supporting evidence

        IMPORTANT: Papers are for GROUNDING the agent's expert thinking,
        not for RESTRICTING what the agent can propose.
        """
        print(f"   🧠 {agent.title} using ReAct reasoning...")

        react_history = []

        for step in range(max_steps):
            # Thought: Agent proposes something from their expertise
            thought_prompt = f"""You are {agent.title} preparing for a team meeting.

Agenda: {agenda}

Discussion so far:
{context}

{"Previous reasoning:" if react_history else ""}
{self._format_react_history(react_history)}

Based on YOUR EXPERTISE and knowledge of machine learning, what do you think would be a good approach for this problem?
Think about what you would propose, then identify what you'd want to search for to find supporting research.

Output format:
Thought: [What I'm proposing based on my expertise and why]
Action: Search papers on "[2-4 word search query]" to find supporting evidence

Be concise. Only output Thought and Action.
"""

            thought_action = self.llm.generate(
                thought_prompt,
                system_instruction=agent.prompt,
                temperature=temperature * 0.8
            )

            # Parse thought and action
            thought = ""
            search_query = ""

            for line in thought_action.split('\n'):
                if line.startswith('Thought:'):
                    thought = line.replace('Thought:', '').strip()
                elif line.startswith('Action:'):
                    action_text = line.replace('Action:', '').strip()
                    # Extract query from "Search papers on 'X'" or similar
                    if '"' in action_text:
                        search_query = action_text.split('"')[1]
                    elif "'" in action_text:
                        search_query = action_text.split("'")[1]
                    else:
                        # Fallback: use last few words
                        words = action_text.split()
                        search_query = ' '.join(words[-4:]) if len(words) > 4 else action_text

            if not search_query:
                break  # Stop if can't parse

            print(f"      Step {step+1} Thought: {thought[:80]}...")
            print(f"      Step {step+1} Action: Search '{search_query}'")

            # Action: Search papers
            observation = self._search_and_observe(agent, search_query)

            print(f"      Step {step+1} Observation: {observation[:100]}...")

            react_history.append({
                'thought': thought,
                'action': f"Search papers on '{search_query}'",
                'observation': observation
            })

        # Final Answer: Generate proposal with citations
        final_prompt = f"""You are {agent.title} in a team meeting.

Agenda: {agenda}

Discussion so far:
{context}

Your ReAct reasoning process:
{self._format_react_history(react_history)}

Provide your final proposal based on YOUR EXPERTISE and the reasoning you've done.
Use the papers you found as SUPPORTING EVIDENCE to ground your recommendations.

You are NOT limited to only what's in the papers - propose what YOU think is best based on your ML expertise.
The papers are there to cite as evidence, not to restrict your thinking.

**Cite relevant papers from your knowledge base to support your recommendations.**
Format citations as: (Author et al., Year)

Keep it focused (1-2 paragraphs).
"""

        final_proposal = self.llm.generate(
            final_prompt,
            system_instruction=agent.prompt,
            temperature=temperature * 0.9
        )

        return final_proposal

    def _format_react_history(self, history: list) -> str:
        """Format ReAct history for prompts"""
        if not history:
            return ""

        formatted = []
        for i, step in enumerate(history, 1):
            formatted.append(f"Step {i}:")
            formatted.append(f"  Thought: {step['thought']}")
            formatted.append(f"  Action: {step['action']}")
            formatted.append(f"  Observation: {step['observation']}")

        return '\n'.join(formatted)

    def _search_and_observe(self, agent, search_query: str) -> str:
        """Search papers and return observation summary with key insights"""
        try:
            # Limit query length
            if len(search_query) > 50:
                search_query = search_query[:50]

            # Search
            raw_results = self.research_api.search(
                query=search_query,
                limit=5,
                year_range=(2018, 2025)
            )

            if not raw_results:
                return "No relevant papers found."

            # Get existing papers
            existing_titles = [p.title for p in agent.knowledge_base.papers]

            # Add new papers with analysis
            papers_found = []
            for result in raw_results[:2]:
                if result.title not in existing_titles:
                    paper = result.to_paper()

                    # Analyze paper to extract key insights
                    # Fast analysis for iteration speed
                    analysis_prompt = f"""Extract 2-3 key actionable insights from this paper abstract.

Title: {paper.title}
Abstract: {paper.abstract}

Output ONLY a JSON array of 2-3 brief insights:
["insight 1", "insight 2", "insight 3"]

Focus on methods, findings, or techniques that could be applied."""

                    try:
                        insights = self.llm.generate_json(analysis_prompt, temperature=0.3)
                        if isinstance(insights, list):
                            paper.key_findings = insights[:3]
                    except Exception:
                        # Fallback: use first sentence of abstract
                        paper.key_findings = [paper.abstract.split('.')[0] + '.'] if paper.abstract else []

                    agent.knowledge_base.add_paper(paper)
                    papers_found.append(paper)
                    print(f"         ✓ {paper.title[:60]}... ({paper.year})")

            if not papers_found:
                return "Papers already in knowledge base."

            # Build observation summary with insights
            observation = f"Found {len(papers_found)} relevant papers:\n"
            observations = []
            for paper in papers_found:
                obs = f"- {paper.title[:60]}... ({', '.join(paper.authors[:2])} et al., {paper.year})"
                if paper.key_findings:
                    obs += f"\n  Key insights: {'; '.join(paper.key_findings[:2])}"
                observations.append(obs)

            observation += '\n'.join(observations)
            return observation

        except Exception as e:
            return f"Search failed: {e}"

    def _search_papers_to_verify(self, agent, draft_proposal: str):
        """Search for papers to verify/support agent's draft proposal"""
        try:
            # Extract search query from draft proposal
            query_prompt = f"""Extract a concise search query (2-4 words) for finding papers to verify this proposal.

Draft proposal:
{draft_proposal[:500]}

Generate a search query that captures the main technique/approach being proposed.
Output ONLY the search query (2-4 words).
"""
            search_query = self.llm.generate(query_prompt, temperature=0.3).strip().strip('"\'')

            # Limit search query length
            if len(search_query) > 50:
                search_query = search_query[:50]

            print(f"   🔍 {agent.title} verifying with papers: '{search_query}'...")

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
                    print(f"      (papers already in knowledge base)")
            else:
                print(f"      (no papers found)")

        except Exception as e:
            # Don't break meeting if search fails
            print(f"      ⚠️  Search failed: {e}")


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
