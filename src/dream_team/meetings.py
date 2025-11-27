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

    def _react_synthesis_task(self, team_lead, agenda: str, context: str, temperature: float, is_final: bool, max_steps: int = 3) -> str:
        """
        ReAct loop for PI synthesis: iterative reasoning to synthesize team proposals.

        Pattern:
        1. Thought: Analyze what each team member proposed
        2. Thought: Identify conflicts, dependencies, priorities
        3. Thought: Formulate clear action plan for coding agent
        4. Final Answer: Write decisive synthesis

        This is internal reasoning only - no external search.
        Used for team lead to think through synthesis step-by-step.
        """
        print(f"   🧠 {team_lead.title} using ReAct reasoning for synthesis...")

        reasoning_steps = []

        for step in range(max_steps):
            # Build context from previous thoughts
            previous_thoughts = ""
            if reasoning_steps:
                previous_thoughts = "\n\nPrevious reasoning:\n" + "\n".join([
                    f"Step {i+1}: {thought}" for i, thought in enumerate(reasoning_steps)
                ])

            # Iterative thinking prompts
            if step == 0:
                thinking_prompt = f"""You are synthesizing team proposals.

Agenda: {agenda}

Discussion so far:
{context}

Think step-by-step about WHAT WAS PROPOSED:
- What did each team member propose?
- What are the key ideas?
- Are there common themes?

Output only:
Thought: [Your analysis of team proposals]
"""
            elif step == 1:
                thinking_prompt = f"""You are refining your synthesis.

Agenda: {agenda}

Discussion so far:
{context}
{previous_thoughts}

Think step-by-step about PRIORITIES AND DEPENDENCIES:
- Are there any conflicts between proposals?
- What needs to be done first?
- What's most important for the goal?

Output only:
Thought: [Your thinking about priorities and dependencies]
"""
            else:
                thinking_prompt = f"""You are finalizing your synthesis.

Agenda: {agenda}

Discussion so far:
{context}
{previous_thoughts}

Think step-by-step about the ACTION PLAN:
- What specific steps should the coding agent take?
- In what order?
- Any important details to emphasize?

Output only:
Thought: [Your final thoughts on the action plan]
"""

            thought = self.llm.generate(
                thinking_prompt,
                system_instruction=team_lead.prompt,
                temperature=temperature * 0.7
            ).strip()

            # Remove "Thought:" prefix if present
            if thought.startswith('Thought:'):
                thought = thought.replace('Thought:', '').strip()

            print(f"      Step {step+1}: {thought[:100]}...")
            reasoning_steps.append(thought)

        # Final: Write synthesis based on all reasoning
        if is_final:
            final_prompt = f"""You are providing FINAL SYNTHESIS and DECISIONS.

Agenda: {agenda}

Discussion so far:
{context}

Your reasoning process:
{chr(10).join([f"Step {i+1}: {thought}" for i, thought in enumerate(reasoning_steps)])}

Now write your FINAL SYNTHESIS based on your reasoning.

IMPORTANT:
- Make FINAL DECISIONS, do NOT ask clarifying questions
- Synthesize what the team proposed into a clear action plan
- Be decisive and specific about what to implement
- Structure it clearly for the coding agent to understand

Keep it focused (2-3 paragraphs).
"""
        else:
            final_prompt = f"""You are providing intermediate synthesis.

Agenda: {agenda}

Discussion so far:
{context}

Your reasoning process:
{chr(10).join([f"Step {i+1}: {thought}" for i, thought in enumerate(reasoning_steps)])}

Now write your synthesis based on your reasoning.

As team lead, synthesize the key points and guide the next round of discussion.
Highlight areas of agreement and any gaps that need more exploration.

Keep it concise (1-2 paragraphs).
"""

        synthesis = self.llm.generate(
            final_prompt,
            system_instruction=team_lead.prompt,
            temperature=temperature * 0.8
        )

        return synthesis

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

IMPORTANT: This is ONLY the opening. Do NOT generate team member proposals or synthesis.
Your team members will speak next after you finish opening.

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

            # Team lead synthesizes using ReAct
            is_final_round = (round_num == num_rounds - 1)
            context = self._build_context()

            synthesis = self._react_synthesis_task(
                team_lead=team_lead,
                agenda=agenda,
                context=context,
                temperature=temperature * 0.8,  # Slightly more focused
                is_final=is_final_round
            )

            self.add_message(team_lead.title, synthesis)
            print(f"💬 {team_lead.title} (synthesis):")
            print(f"{synthesis}\n")

        # Use the team lead's final synthesis as the summary
        # This is the actual action plan, not a generic overview
        # The last synthesis in the transcript is from the final round
        final_synthesis = synthesis  # This is the team lead's synthesis from the last round

        # Generate structured metadata from transcript
        metadata_prompt = f"""Based on this meeting transcript, extract structured metadata:

{self._build_context()}

Provide in JSON format:
{{
    "key_insights": ["insight 1", "insight 2", ...],
    "decisions": ["decision 1", "decision 2", ...],
    "action_items": ["action 1", "action 2", ...]
}}
"""

        metadata = self.llm.generate_json(metadata_prompt, temperature=0.3)

        # Return team lead's synthesis as the main summary
        return {
            "summary": final_synthesis,  # The detailed action plan
            "key_insights": metadata.get("key_insights", []),
            "decisions": metadata.get("decisions", []),
            "action_items": metadata.get("action_items", [])
        }

    def _build_context(self, max_messages: int = 10) -> str:
        """Build context string from recent transcript"""
        recent = self.transcript[-max_messages:]
        return "\n\n".join([
            f"{msg['agent']}: {msg['message']}"
            for msg in recent
        ])

    def _react_proposal(self, agent, agenda: str, context: str, temperature: float, max_steps: int = 3) -> str:
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
            if step == 0:
                thought_prompt = f"""You are {agent.title} preparing for a team meeting.

Agenda: {agenda}

Discussion so far:
{context}

Based on YOUR EXPERTISE and knowledge of machine learning, what do you think would be a good approach for this problem?
Think about what you would propose, then identify what you'd want to search for to find supporting research.

Output format:
Thought: [What I'm proposing based on my expertise and why]
Action: Search papers on "[2-4 word search query]" to find supporting evidence

Be concise. Only output Thought and Action.
"""
            else:
                # IMPORTANT: Reference previous observation to refine thinking
                thought_prompt = f"""You are {agent.title} refining your proposal for a team meeting.

Agenda: {agenda}

Discussion so far:
{context}

Your previous reasoning:
{self._format_react_history(react_history)}

Based on what you learned from the papers, refine your thinking:
- Did the papers support your idea? Contradict it? Suggest modifications?
- What additional evidence would strengthen your proposal?
- Should you search for something different?

Output format:
Thought: [Your refined thinking based on what you learned]
Action: Search papers on "[2-4 word search query]" OR Action: Finalize proposal

If you have enough evidence, use "Action: Finalize proposal" to stop searching.

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

            # Check for early termination when agent has enough evidence
            if "finalize" in search_query.lower() or "stop" in search_query.lower():
                print(f"      Step {step+1}: Agent ready to finalize")
                break

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

            # Search with wider year range to include seminal papers
            raw_results = self.research_api.search(
                query=search_query,
                limit=5,
                year_range=(2015, 2025)
            )

            if not raw_results:
                return f"No papers found for '{search_query}'. Consider: (1) broader terms, (2) different phrasing, (3) related concepts."

            # Get existing papers
            existing_titles = [p.title for p in agent.knowledge_base.papers]

            # Add new papers with analysis
            papers_found = []
            for result in raw_results[:5]:
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

                    # Add paper to knowledge base AND update mathematical state (K, δ, θ)
                    agent.add_paper_to_knowledge(paper)
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
            return f"Search failed: {e}. Consider proceeding with your expertise or trying a different query."

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
                        # Add paper to knowledge base AND update mathematical state (K, δ, θ)
                        agent.add_paper_to_knowledge(paper)
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

    def _react_coding_task(self, agent, task: str, temperature: float, max_steps: int = 3) -> str:
        """
        ReAct loop for coding tasks: iterative reasoning to plan implementation.

        Pattern:
        1. Thought: Think about the approach/architecture
        2. Thought: Refine the approach and consider edge cases
        3. Thought: Finalize implementation details
        4. Final Answer: Write the complete code

        This is internal reasoning only - no external search.
        Used for coding agent to think through implementation step-by-step.
        """
        print(f"   🧠 {agent.title} using ReAct reasoning...")

        reasoning_steps = []

        for step in range(max_steps):
            # Build context from previous thoughts
            previous_thoughts = ""
            if reasoning_steps:
                previous_thoughts = "\n\nPrevious reasoning:\n" + "\n".join([
                    f"Step {i+1}: {thought}" for i, thought in enumerate(reasoning_steps)
                ])

            # Iterative thinking prompts
            if step == 0:
                thinking_prompt = f"""You are planning how to implement a coding task.

Task: {task}

Think step-by-step about the APPROACH:
- What's the overall architecture/structure?
- What are the main steps?
- What libraries/methods will you use?

Output only:
Thought: [Your thinking about the overall approach]
"""
            elif step == 1:
                thinking_prompt = f"""You are refining your implementation plan.

Task: {task}
{previous_thoughts}

Think step-by-step about IMPLEMENTATION DETAILS:
- What edge cases need handling?
- What's the data flow?
- What features/transformations are needed?

Output only:
Thought: [Your thinking about implementation details]
"""
            else:
                thinking_prompt = f"""You are finalizing your implementation plan.

Task: {task}
{previous_thoughts}

Think step-by-step about FINAL DETAILS:
- Are there any missing pieces?
- How will you ensure correctness?
- Any optimizations needed?

Output only:
Thought: [Your final thoughts before coding]
"""

            thought = self.llm.generate(
                thinking_prompt,
                system_instruction=agent.prompt,
                temperature=temperature * 0.7
            ).strip()

            # Remove "Thought:" prefix if present
            if thought.startswith('Thought:'):
                thought = thought.replace('Thought:', '').strip()

            print(f"      Step {step+1}: {thought[:100]}...")
            reasoning_steps.append(thought)

        # Final: Write code based on all reasoning
        final_prompt = f"""You are implementing a coding task.

Task: {task}

Your reasoning process:
{chr(10).join([f"Step {i+1}: {thought}" for i, thought in enumerate(reasoning_steps)])}

Now write the COMPLETE, EXECUTABLE code based on your reasoning.
Be thorough and handle all the details you thought through.

Output ONLY the code in ```python blocks.
"""

        final_output = self.llm.generate(
            final_prompt,
            system_instruction=agent.prompt,
            temperature=temperature * 0.9
        )

        return final_output

    def _react_individual_task(self, agent, task: str, temperature: float, max_steps: int = 3) -> str:
        """
        ReAct loop for individual task: agent reasons and searches papers before final output.

        Pattern:
        1. Thought: Think about the task and what approach to take
        2. Action: Search papers to inform the decision
        3. Observation: Papers found
        4. (Repeat to build grounded thinking)
        5. Final Answer: Complete the task with citations

        Used for bootstrap recruitment and other individual tasks.
        """
        print(f"   🧠 {agent.title} using ReAct reasoning...")

        react_history = []

        for step in range(max_steps):
            # Thought: Think about task and what to search
            thought_prompt = f"""You are working on a task.

Task: {task}

{"Previous reasoning:" if react_history else ""}
{self._format_react_history(react_history)}

Based on YOUR EXPERTISE, think about how to approach this task.
What do you need to know? What should you search for to inform your decision?

Output format:
Thought: [Your thinking about how to approach this task]
Action: Search papers on "[2-4 word search query]" to inform this decision

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

        # Final Answer: Complete task with citations
        final_prompt = f"""You are completing a task.

Task: {task}

Your ReAct reasoning process:
{self._format_react_history(react_history)}

Complete the task based on YOUR EXPERTISE and the research you've done.
Use papers you found as SUPPORTING EVIDENCE to inform your decisions.

**Cite relevant papers to support your recommendations.**
Format citations as: (Author et al., Year)

Be specific, detailed, and actionable.
"""

        final_output = self.llm.generate(
            final_prompt,
            system_instruction=agent.prompt,
            temperature=temperature * 0.9
        )

        return final_output

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

            # Search with wider year range to include seminal papers
            raw_results = self.research_api.search(
                query=search_query,
                limit=5,
                year_range=(2015, 2025)
            )

            if not raw_results:
                return f"No papers found for '{search_query}'. Consider: (1) broader terms, (2) different phrasing, (3) related concepts."

            # Get existing papers
            existing_titles = [p.title for p in agent.knowledge_base.papers]

            # Add new papers with analysis
            papers_found = []
            for result in raw_results[:5]:
                if result.title not in existing_titles:
                    paper = result.to_paper()

                    # Analyze paper to extract key insights
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

                    # Add paper to knowledge base AND update mathematical state (K, δ, θ)
                    agent.add_paper_to_knowledge(paper)
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
            return f"Search failed: {e}. Consider proceeding with your expertise or trying a different query."

    def run(
        self,
        agent: Agent,
        task: str,
        critic_agent: Optional[Agent] = None,
        num_iterations: int = 2,
        temperature: float = 0.7,
        use_react: bool = False,
        use_react_coding: bool = False
    ) -> str:
        """
        Run individual meeting with iterative refinement

        Args:
            use_react: If True and research_api available, use ReAct to search papers
            use_react_coding: If True, use ReAct for coding (internal reasoning, no search)

        Returns: Final output
        """

        print(f"\n👤 INDIVIDUAL MEETING")
        print(f"   Agent: {agent.title}")
        print(f"   Iterations: {num_iterations}")
        if use_react and self.research_api:
            print(f"   Using ReAct: Yes (with paper search)")
        elif use_react_coding:
            print(f"   Using ReAct: Yes (internal reasoning)")
        print()

        # Initial work - choose ReAct type
        if use_react_coding:
            # Coding agent: iterative reasoning without external search
            output = self._react_coding_task(agent, task, temperature)
        elif use_react and self.research_api:
            # Domain expert: ReAct with paper search
            output = self._react_individual_task(agent, task, temperature)
        else:
            # Simple generation without ReAct
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
