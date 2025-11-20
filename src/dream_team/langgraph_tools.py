"""
LangGraph tools for Dream Team agents.

Provides standardized tool interfaces for:
- Paper search (Semantic Scholar)
- Code execution
"""

from typing import Optional, List, Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from .executor import get_executor, set_executor_context


# Tool input schemas

class PaperSearchInput(BaseModel):
    """Input for paper search tool"""
    query: str = Field(description="Search query for finding relevant papers (2-5 words)")
    limit: int = Field(default=5, description="Maximum number of papers to return")


class CodeExecutionInput(BaseModel):
    """Input for code execution tool"""
    code: str = Field(description="Python code to execute")
    description: str = Field(default="", description="Brief description of what the code does")


# Paper search tool

@tool(args_schema=PaperSearchInput)
def search_papers(query: str, limit: int = 5) -> str:
    """
    Search for academic papers using Semantic Scholar.

    Returns a formatted string with paper titles, authors, years, and key insights.
    Papers are automatically added to the agent's knowledge base.

    Args:
        query: Search query for relevant papers (e.g., "gradient boosting", "time series forecasting")
        limit: Maximum number of papers to return (default: 5)

    Returns:
        Formatted string with paper information
    """
    try:
        from .research import get_research_assistant

        # Get research assistant (singleton)
        research = get_research_assistant()

        # Search for papers
        raw_results = research.ss_api.search(
            query=query,
            limit=limit,
            year_range=(2018, 2025)
        )

        if not raw_results:
            return f"No papers found for query: {query}"

        # Format results
        output = f"Found {len(raw_results)} papers for '{query}':\n\n"

        for i, result in enumerate(raw_results, 1):
            paper = result.to_paper()
            output += f"{i}. {paper.title}\n"
            output += f"   Authors: {', '.join(paper.authors[:3])}"
            if len(paper.authors) > 3:
                output += " et al."
            output += f"\n   Year: {paper.year}\n"

            if paper.key_findings:
                output += f"   Key insights: {'; '.join(paper.key_findings[:2])}\n"

            output += "\n"

        # Note: In actual use, papers would be added to agent KB through state updates
        return output.strip()

    except Exception as e:
        return f"Paper search failed: {str(e)}"


# Code execution tool

@tool(args_schema=CodeExecutionInput)
def execute_code(code: str, description: str = "") -> str:
    """
    Execute Python code in a controlled environment.

    The code has access to pre-imported libraries (pandas, numpy, torch, sklearn)
    and variables from previous executions (persistent namespace).

    Args:
        code: Python code to execute
        description: Brief description of what the code does

    Returns:
        Execution output (stdout, results, or error messages)
    """
    try:
        from .executor import get_executor

        # Get global executor (maintains persistent namespace)
        executor = get_executor()

        # Execute code
        result = executor.execute(code=code, description=description)

        if result['success']:
            output = result['output']
            # Truncate very long outputs
            if len(output) > 5000:
                output = output[:2500] + f"\n\n... [{len(output) - 5000} chars omitted] ...\n\n" + output[-2500:]
            return output
        else:
            error_msg = f"Execution failed:\n{result['error']}\n\n"
            if result.get('traceback'):
                error_msg += f"Traceback:\n{result['traceback']}"
            return error_msg

    except Exception as e:
        return f"Code execution error: {str(e)}"


# Tool lists for different agent types

RESEARCH_AGENT_TOOLS = [search_papers]
CODING_AGENT_TOOLS = [execute_code]
ALL_TOOLS = [search_papers, execute_code]


# Helper: Get tools for agent type

def get_tools_for_agent(agent_type: str) -> List:
    """
    Get appropriate tools for agent type.

    Args:
        agent_type: One of "team_lead", "team_member", "coding_agent"

    Returns:
        List of tools for that agent type
    """
    if agent_type == "coding_agent":
        return CODING_AGENT_TOOLS
    elif agent_type in ["team_lead", "team_member"]:
        return RESEARCH_AGENT_TOOLS
    else:
        return []


_research_assistant = None


def get_research_assistant():
    """Get or create singleton research assistant"""
    global _research_assistant

    if _research_assistant is None:
        from .research import ResearchAssistant, SemanticScholarAPI
        import os

        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        ss_api = SemanticScholarAPI(api_key=api_key)

        from .llm import get_llm
        llm = get_llm()

        _research_assistant = ResearchAssistant(ss_api=ss_api, llm=llm)

    return _research_assistant


