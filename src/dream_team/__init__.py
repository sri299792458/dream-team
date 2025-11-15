"""
Dream Team: A dynamic multi-agent framework with evolving personas.

This framework enables AI agents to:
- Evolve their expertise based on problem needs
- Integrate research from Semantic Scholar
- Collaborate through structured meetings
- Build knowledge bases that grow over time
"""

from .agent import Agent, Paper, KnowledgeBase
from .llm import GeminiLLM, get_llm
from .research import SemanticScholarAPI, ResearchAssistant, get_research_assistant
from .evolution import (
    EvolutionEngine,
    EvolutionTrigger,
    PerformancePlateauTrigger,
    ErrorPatternTrigger,
    KnowledgeGapTrigger
)
from .meetings import TeamMeeting, IndividualMeeting
from .executor import CodeExecutor, extract_code_from_text
from .orchestrator import ExperimentOrchestrator
from .utils import save_json, load_json, load_summaries

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "Paper",
    "KnowledgeBase",
    "GeminiLLM",
    "get_llm",
    "SemanticScholarAPI",
    "ResearchAssistant",
    "get_research_assistant",
    "EvolutionEngine",
    "EvolutionTrigger",
    "PerformancePlateauTrigger",
    "ErrorPatternTrigger",
    "KnowledgeGapTrigger",
    "TeamMeeting",
    "IndividualMeeting",
    "CodeExecutor",
    "extract_code_from_text",
    "ExperimentOrchestrator",
    "save_json",
    "load_json",
    "load_summaries",
]
