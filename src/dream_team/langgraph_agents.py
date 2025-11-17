"""
Enhanced agent creation with ReAct patterns for Dream Team.

Provides factory functions to create agents with:
- Proper ReAct loops using create_react_agent()
- Dynamic system prompts that include mathematical state
- Tool integration for research and coding
"""

from typing import List, Optional, Dict, Any
from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph

from .langgraph_tools import search_papers, execute_code, get_tools_for_agent
from .langgraph_state import SerializedAgent, deserialize_agent


def create_agent_system_prompt(agent_data: SerializedAgent) -> str:
    """
    Create dynamic system prompt that includes mathematical state.

    This makes K, θ, δ actually useful by showing the agent:
    - What concepts they know (K)
    - What they're focused on (θ)
    - How deep their expertise is (δ)
    """
    # Extract mathematical state
    K = agent_data["K"]
    θ = agent_data["θ"]
    δ = agent_data["δ"]

    # Compute key metrics
    gini = _compute_gini(δ["depths"])
    max_depth = max(δ["depths"].values()) if δ["depths"] else 0.0
    top_concepts = sorted(δ["depths"].items(), key=lambda x: x[1], reverse=True)[:3]

    # Build mathematical state context
    math_state = ""
    if top_concepts:
        math_state = f"\n## Your Current Focus:\n"
        math_state += f"Specialization level: {gini:.2f} (0=generalist, 1=specialist)\n"
        math_state += f"Deep expertise ({max_depth:.2f}) in:\n"
        for concept, depth in top_concepts:
            math_state += f"- {concept} (depth: {depth:.2f})\n"

    # Build knowledge base context
    kb_context = ""
    if agent_data["papers"]:
        kb_context += f"\n## Research Papers You Know ({len(agent_data['papers'])}):\n"
        for paper in agent_data["papers"][:5]:  # Top 5
            kb_context += f"- {paper['title']} ({paper['year']})\n"
            if paper.get('key_findings'):
                kb_context += f"  Key: {paper['key_findings'][0]}\n"

    if agent_data["techniques"]:
        kb_context += f"\n## Techniques You've Mastered:\n"
        for tech in agent_data["techniques"][:5]:
            kb_context += f"- {tech}\n"

    if agent_data["domain_facts"]:
        kb_context += f"\n## Domain Knowledge:\n"
        for fact in agent_data["domain_facts"][:3]:
            kb_context += f"- {fact}\n"

    # Role-specific closing
    if "Principal Investigator" in agent_data["title"] or "Lead" in agent_data["title"]:
        closing = """
You are part of a research team solving data science challenges.

As the lead, ensure your approach embodies rigorous experimental methodology.
Consider validation, baselines, incremental progress, and systematic learning.
Apply your full expertise to guide the team toward methodologically sound decisions.

When you need information, use the search_papers tool to find relevant research.
"""
    elif "Engineer" in agent_data["title"] or "Coding" in agent_data["title"]:
        closing = """
You are part of a research team solving data science challenges.

As the implementation expert, focus on writing robust, efficient, and correct code.
Consider edge cases, error handling, and code quality.
Use the execute_code tool to test your implementations.

When stuck, break problems down step-by-step and verify each piece works.
"""
    else:
        closing = """
You are part of a research team solving data science challenges.

When proposing approaches, consider both domain expertise and experimental rigor.
Think about validation, incremental progress, and learning from previous work.
Provide insightful, actionable contributions grounded in sound methodology.

Use the search_papers tool to ground your proposals in current research.
"""

    # Assemble full prompt
    system_prompt = f"""You are {agent_data['title']}.

Expertise: {agent_data['expertise']}

Goal: {agent_data['goal']}

Role: {agent_data['role']}
{math_state}
{kb_context}

{closing}"""

    return system_prompt


def create_research_agent(agent_data: SerializedAgent, llm: Optional[ChatGoogleGenerativeAI] = None):
    """
    Create a ReAct agent for research tasks (team lead or team members).

    This agent can:
    - Search for papers
    - Reason step-by-step
    - Ground proposals in research

    Args:
        agent_data: Serialized agent state
        llm: Optional LLM instance (will create if not provided)

    Returns:
        ReAct agent that can be invoked with state
    """
    if llm is None:
        from .llm import get_llm as get_gemini_llm
        llm_backend = get_gemini_llm()
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.7
        )

    # Get system prompt with mathematical state
    system_prompt = create_agent_system_prompt(agent_data)

    # Create ReAct agent with paper search tool
    tools = [search_papers]

    agent = create_react_agent(
        llm,
        tools,
        state_modifier=system_prompt  # Injects system prompt into every call
    )

    return agent


def create_coding_agent(agent_data: SerializedAgent, llm: Optional[ChatGoogleGenerativeAI] = None):
    """
    Create a ReAct agent for coding tasks.

    This agent can:
    - Execute code
    - Reason through implementation
    - Debug errors

    Args:
        agent_data: Serialized agent state
        llm: Optional LLM instance

    Returns:
        ReAct agent for coding
    """
    if llm is None:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.3  # Lower temp for coding
        )

    # Get system prompt
    system_prompt = create_agent_system_prompt(agent_data)

    # Create ReAct agent with code execution tool
    tools = [execute_code]

    agent = create_react_agent(
        llm,
        tools,
        state_modifier=system_prompt
    )

    return agent


def create_planning_agent(agent_data: SerializedAgent, llm: Optional[ChatGoogleGenerativeAI] = None):
    """
    Create agent for planning tasks (no tools, just reasoning).

    Used for synthesis, decision-making, high-level planning.

    Args:
        agent_data: Serialized agent state
        llm: Optional LLM instance

    Returns:
        Agent for planning tasks
    """
    if llm is None:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.6
        )

    # Get system prompt
    system_prompt = create_agent_system_prompt(agent_data)

    # Simple agent without tools
    # Just wraps LLM with system prompt
    def planning_agent_fn(messages: List):
        """Planning agent function"""
        full_messages = [SystemMessage(content=system_prompt)] + messages
        response = llm.invoke(full_messages)
        return {"messages": [response]}

    return planning_agent_fn


# Helper functions

def _compute_gini(depths: Dict[str, float]) -> float:
    """Compute Gini coefficient for specialization"""
    if not depths:
        return 0.0

    import numpy as np
    depths_list = sorted(depths.values())
    n = len(depths_list)

    if n == 0 or sum(depths_list) == 0:
        return 0.0

    index = np.arange(1, n + 1)
    return (2 * np.sum(index * depths_list)) / (n * np.sum(depths_list)) - (n + 1) / n


# Agent invocation helpers

def invoke_research_agent(
    agent_data: SerializedAgent,
    task: str,
    context: str = "",
    max_iterations: int = 5
) -> Dict[str, Any]:
    """
    Invoke research agent on a task.

    Args:
        agent_data: Serialized agent state
        task: Task description
        context: Additional context
        max_iterations: Max ReAct iterations

    Returns:
        Dict with response and metadata
    """
    agent = create_research_agent(agent_data)

    # Build input
    full_task = f"{context}\n\n{task}" if context else task

    # Invoke with ReAct loop
    result = agent.invoke(
        {"messages": [HumanMessage(content=full_task)]},
        {"recursion_limit": max_iterations}
    )

    # Extract final response
    final_message = result["messages"][-1]

    return {
        "response": final_message.content,
        "messages": result["messages"],
        "agent": agent_data["title"]
    }


def invoke_coding_agent(
    agent_data: SerializedAgent,
    task: str,
    context: str = "",
    max_iterations: int = 5
) -> Dict[str, Any]:
    """
    Invoke coding agent on a task.

    Args:
        agent_data: Serialized agent state
        task: Coding task
        context: Additional context
        max_iterations: Max ReAct iterations

    Returns:
        Dict with code and metadata
    """
    agent = create_coding_agent(agent_data)

    # Build input
    full_task = f"{context}\n\n{task}" if context else task

    # Invoke with ReAct loop
    result = agent.invoke(
        {"messages": [HumanMessage(content=full_task)]},
        {"recursion_limit": max_iterations}
    )

    # Extract response
    final_message = result["messages"][-1]

    return {
        "response": final_message.content,
        "messages": result["messages"],
        "agent": agent_data["title"]
    }


def invoke_planning_agent(
    agent_data: SerializedAgent,
    task: str,
    context: str = ""
) -> Dict[str, Any]:
    """
    Invoke planning agent on a task.

    Args:
        agent_data: Serialized agent state
        task: Planning task
        context: Additional context

    Returns:
        Dict with plan and metadata
    """
    agent_fn = create_planning_agent(agent_data)

    # Build input
    full_task = f"{context}\n\n{task}" if context else task

    # Invoke
    result = agent_fn([HumanMessage(content=full_task)])

    # Extract response
    final_message = result["messages"][-1]

    return {
        "response": final_message.content,
        "agent": agent_data["title"]
    }
