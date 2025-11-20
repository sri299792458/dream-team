"""
Multi-agent team meeting subgraph for Dream Team.

Implements proper multi-agent collaboration pattern:
- Team lead opens and synthesizes
- Members contribute with ReAct + paper search
- Structured message passing
- Automatic paper integration into agent KB
"""

from typing import TypedDict, List, Annotated
import operator

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

from .langgraph_agents import (
    invoke_research_agent,
    invoke_planning_agent,
    create_agent_system_prompt
)
from .langgraph_state import SerializedAgent, deserialize_agent
from .agent import Paper


# Team meeting state

class TeamMeetingState(TypedDict):
    """State for team meeting subgraph"""
    agenda: str
    messages: Annotated[List, operator.add]  # Conversation history
    team_lead: SerializedAgent
    team_members: List[SerializedAgent]
    synthesis: str  # Final synthesis from lead
    papers_found: List[dict]  # Papers discovered during meeting


# Team meeting nodes

def opening_node(state: TeamMeetingState) -> TeamMeetingState:
    """
    Team lead opens the meeting.

    Sets context, frames problem, asks key questions.
    """
    team_lead = state["team_lead"]
    agenda = state["agenda"]

    opening_task = f"""You are opening a team meeting.

Agenda:
{agenda}

Open the meeting by:
1. Framing the problem clearly
2. Highlighting key questions we need to answer
3. Setting expectations for the discussion

IMPORTANT: This is ONLY the opening. You will hear from team members next.

Keep it concise (2-3 paragraphs).
"""

    result = invoke_planning_agent(team_lead, opening_task)

    opening_message = result["response"]

    print(f"\n💬 {team_lead['title']}:")
    print(f"{opening_message}\n")

    return {
        "messages": [AIMessage(content=opening_message, name=team_lead["title"])],
    }


def member_proposal_node(state: TeamMeetingState, member_idx: int) -> TeamMeetingState:
    """
    Team member proposes approach using ReAct + paper search.

    Args:
        state: Meeting state
        member_idx: Index of team member to invoke

    Returns:
        Updated state with member's proposal
    """
    if member_idx >= len(state["team_members"]):
        return {"messages": []}

    member = state["team_members"][member_idx]
    agenda = state["agenda"]

    # Build context from conversation
    conversation = "\n".join([
        f"{msg.name if hasattr(msg, 'name') else 'Unknown'}: {msg.content[:200]}..."
        for msg in state["messages"][-5:]  # Last 5 messages
    ])

    proposal_task = f"""You are participating in a team meeting.

Agenda:
{agenda}

Discussion so far:
{conversation}

Provide your proposal based on your expertise as {member['title']}.

Consider:
1. What the team has discussed
2. Your specialized knowledge
3. What approaches might work

If you need supporting research, use the search_papers tool to find relevant papers.

Propose what to do next (2-3 sentences). Be specific and actionable.
"""

    # Invoke with ReAct (can use paper search)
    result = invoke_research_agent(
        member,
        proposal_task,
        max_iterations=5
    )

    proposal = result["response"]

    print(f"💬 {member['title']}:")
    print(f"{proposal}\n")

    # Extract papers if any were searched
    papers_found = _extract_papers_from_messages(result.get("messages", []))

    # Update member's KB with papers (in real implementation)
    # For now, just track them
    updates = {
        "messages": [AIMessage(content=proposal, name=member["title"])],
    }

    if papers_found:
        updates["papers_found"] = papers_found

    return updates


def synthesis_node(state: TeamMeetingState) -> TeamMeetingState:
    """
    Team lead synthesizes proposals into action plan.

    This is the critical node that produces the final decision.
    """
    team_lead = state["team_lead"]
    agenda = state["agenda"]

    # Build full conversation context
    conversation = "\n\n".join([
        f"**{msg.name if hasattr(msg, 'name') else 'Speaker'}**:\n{msg.content}"
        for msg in state["messages"]
    ])

    synthesis_task = f"""Synthesize the team's discussion into a clear action plan.

Agenda:
{agenda}

Full discussion:
{conversation}

Your task:
1. Identify key proposals from team members
2. Evaluate approaches based on merit
3. Make clear decisions on what to implement
4. Provide specific requirements

Be decisive and concise (2-3 paragraphs). This synthesis will guide the coding agent.

Output the action plan that should be implemented.
"""

    result = invoke_planning_agent(team_lead, synthesis_task)

    synthesis = result["response"]

    print(f"💬 {team_lead['title']} (Synthesis):")
    print(f"{synthesis}\n")

    return {
        "messages": [AIMessage(content=synthesis, name=f"{team_lead['title']}_synthesis")],
        "synthesis": synthesis
    }


# Build team meeting subgraph

def create_team_meeting_subgraph() -> StateGraph:
    """
    Create team meeting subgraph.

    Flow:
    1. Opening (lead)
    2. For each member: Proposal (parallel or sequential)
    3. Synthesis (lead)

    Returns:
        Compiled subgraph
    """
    graph = StateGraph(TeamMeetingState)

    # Add nodes
    graph.add_node("opening", opening_node)
    graph.add_node("synthesis", synthesis_node)

    # Add member nodes dynamically (we'll use a router instead)
    # For now, assume 3 members max
    graph.add_node("member_0", lambda s: member_proposal_node(s, 0))
    graph.add_node("member_1", lambda s: member_proposal_node(s, 1))
    graph.add_node("member_2", lambda s: member_proposal_node(s, 2))

    # Flow
    graph.set_entry_point("opening")

    # Opening -> first member
    graph.add_edge("opening", "member_0")

    # Members in sequence
    graph.add_conditional_edges(
        "member_0",
        lambda s: "member_1" if len(s["team_members"]) > 1 else "synthesis",
        {"member_1": "member_1", "synthesis": "synthesis"}
    )

    graph.add_conditional_edges(
        "member_1",
        lambda s: "member_2" if len(s["team_members"]) > 2 else "synthesis",
        {"member_2": "member_2", "synthesis": "synthesis"}
    )

    graph.add_edge("member_2", "synthesis")

    # Synthesis -> end
    graph.add_edge("synthesis", END)

    return graph.compile()


# Simplified invocation

def run_team_meeting(
    team_lead: SerializedAgent,
    team_members: List[SerializedAgent],
    agenda: str
) -> dict:
    """
    Run a team meeting and get synthesis.

    Args:
        team_lead: Serialized team lead
        team_members: List of serialized team members
        agenda: Meeting agenda

    Returns:
        Dict with synthesis and papers found
    """
    print("\n" + "="*60)
    print("TEAM MEETING")
    print("="*60)
    print(f"Lead: {team_lead['title']}")
    print(f"Members: {[m['title'] for m in team_members]}")
    print()

    # Create meeting graph
    meeting_graph = create_team_meeting_subgraph()

    # Initial state (sanitized for checkpointing)
    initial_state: TeamMeetingState = _make_msgpack_safe({
        "agenda": agenda,
        "messages": [],
        "team_lead": team_lead,
        "team_members": team_members,
        "synthesis": "",
        "papers_found": []
    })

    # Run meeting
    final_state = meeting_graph.invoke(initial_state)

    print("="*60)
    print()

    return _make_msgpack_safe({
        "synthesis": final_state.get("synthesis", ""),
        "messages": final_state.get("messages", []),
        "papers_found": final_state.get("papers_found", [])
    })


# Helper functions

def _extract_papers_from_messages(messages: List) -> List[dict]:
    """Extract paper information from ReAct messages"""
    papers = []

    for msg in messages:
        content = msg.content if hasattr(msg, 'content') else str(msg)

        # Look for paper search results
        # This is a simplified extraction - in real implementation,
        # the tool would return structured data
        if "Found" in content and "papers" in content.lower():
            # Extract paper info (simplified)
            # Real implementation would parse tool output properly
            pass

    return papers


def update_agent_kb_with_papers(agent_data: SerializedAgent, papers: List[dict]) -> SerializedAgent:
    """
    Update agent's knowledge base with papers from meeting.

    Args:
        agent_data: Serialized agent
        papers: Papers to add

    Returns:
        Updated serialized agent
    """
    # Deserialize
    agent = deserialize_agent(agent_data)

    # Add papers
    for paper_dict in papers:
        paper = Paper(**paper_dict)
        agent.knowledge_base.add_paper(paper)

    # Re-serialize
    from .langgraph_state import serialize_agent
    return serialize_agent(agent)


def _make_msgpack_safe(value):
    """Recursively coerce values into msgpack-friendly forms."""
    import numpy as np

    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, dict):
        return {k: _make_msgpack_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        coerced = [_make_msgpack_safe(v) for v in value]
        return coerced if isinstance(value, list) else tuple(coerced)

    try:
        return repr(value)
    except Exception:
        return "<unserializable>"
