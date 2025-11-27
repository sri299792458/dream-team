"""
Dream Team: A dynamic multi-agent framework with evolving personas.

This framework enables AI agents to:
- Evolve their expertise based on problem needs
- Integrate research from Semantic Scholar
- Collaborate through structured meetings
- Build knowledge bases that grow over time
- Use mathematical state for emergent evolution
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
from .utils import save_json, load_json, load_summaries
from .serialization import RobustJSONEncoder, robust_dump, robust_dumps
from .knowledge_state import (
    KnowledgeGraph,
    AttentionDistribution,
    DepthMap,
    DynamicsState,
    extract_concepts_from_text
)
from .team import Team

# LangGraph API (recommended)
from .experiment import (
    create_initial_state,
    run_graph_experiment,
    AgentConfig,
    ExperimentState,
    IterationSummary
)

__version__ = "0.1.0"

__all__ = [
    # Core components
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
    "save_json",
    "load_json",
    "load_summaries",
    "RobustJSONEncoder",
    "robust_dump",
    "robust_dumps",
    # Mathematical framework
    "KnowledgeGraph",
    "AttentionDistribution",
    "DepthMap",
    "DynamicsState",
    "extract_concepts_from_text",
    "Team",
    # LangGraph API (recommended)
    "create_initial_state",
    "run_graph_experiment",
    "AgentConfig",
    "ExperimentState",
    "IterationSummary",
]
