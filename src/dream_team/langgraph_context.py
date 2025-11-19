"""
Advanced context management for Dream Team LangGraph.

Addresses the core issue: manual truncation loses critical information.

This module provides:
- Semantic summarization of old iterations
- Smart context windowing
- Column schema preservation
- Error context extraction
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .langgraph_state import DreamTeamState, IterationResult
from .llm import get_llm


@dataclass
class ContextWindow:
    """
    Structured context window with different sections.

    Instead of manual truncation, we intelligently select what to include.
    """
    problem: str
    column_schemas: str
    recent_iterations: str  # Last N iterations in full
    summarized_history: str  # Older iterations summarized
    best_so_far: str
    errors_and_learnings: str


def build_smart_context(
    state: DreamTeamState,
    keep_recent: int = 2,
    summarize_older: bool = True
) -> ContextWindow:
    """
    Build smart context window that doesn't lose information.

    Strategy:
    - Keep column schemas (critical, never truncate)
    - Keep last N iterations in full
    - Summarize older iterations (what worked, what failed, why)
    - Highlight best iteration
    - Extract key errors and learnings

    Args:
        state: Current graph state
        keep_recent: How many recent iterations to keep in full
        summarize_older: Whether to summarize older iterations

    Returns:
        Structured context window
    """
    # Problem statement
    problem = state["problem_statement"]

    # Column schemas (NEVER truncate - this prevents hallucination)
    column_schemas = _build_column_schemas_context(state)

    # Split history into recent and old
    history = state.get("experiment_history", [])

    # Filter out bootstrap (iteration 0)
    iterations = [h for h in history if h.get("iteration", 0) > 0]

    if len(iterations) <= keep_recent:
        # All iterations are recent
        recent_iterations = _format_iterations_full(iterations)
        summarized_history = ""
    else:
        # Split
        recent = iterations[-keep_recent:]
        older = iterations[:-keep_recent]

        recent_iterations = _format_iterations_full(recent)

        if summarize_older:
            summarized_history = _summarize_iterations(older, state)
        else:
            summarized_history = _format_iterations_full(older)

    # Best so far
    best_so_far = ""
    if state.get("best_metric") is not None:
        best_iter = state.get("best_iteration", "?")
        best_val = state["best_metric"]
        metric_name = state["target_metric"]
        best_so_far = f"Best {metric_name} so far: {best_val:.4f} (Iteration {best_iter})"

    # Errors and learnings
    errors_and_learnings = _extract_errors_and_learnings(iterations)

    return ContextWindow(
        problem=problem,
        column_schemas=column_schemas,
        recent_iterations=recent_iterations,
        summarized_history=summarized_history,
        best_so_far=best_so_far,
        errors_and_learnings=errors_and_learnings
    )


def format_context_for_planning(context: ContextWindow) -> str:
    """
    Format context window for team planning meeting.

    Returns:
        Formatted string suitable for LLM prompt
    """
    sections = []

    sections.append(f"## Problem:\n{context.problem}")

    if context.column_schemas:
        sections.append(context.column_schemas)

    if context.best_so_far:
        sections.append(f"\n## Progress:\n{context.best_so_far}")

    if context.summarized_history:
        sections.append(f"\n## Previous Work (Summarized):\n{context.summarized_history}")

    if context.recent_iterations:
        sections.append(f"\n## Recent Iterations (Full Details):\n{context.recent_iterations}")

    if context.errors_and_learnings:
        sections.append(f"\n## Key Learnings:\n{context.errors_and_learnings}")

    return "\n\n".join(sections)


def format_context_for_coding(
    context: ContextWindow,
    approach: str,
    include_recent_output: bool = True
) -> str:
    """
    Format context for coding agent.

    Coding agent needs:
    - Column schemas (exact names)
    - Recent output (to build on)
    - Approach to implement

    Returns:
        Formatted string for coding task
    """
    sections = []

    sections.append(f"## Task:\nImplement this approach:\n{approach}")

    if context.column_schemas:
        sections.append(context.column_schemas)

    if include_recent_output:
        # Get most recent iteration output
        recent_output = _extract_recent_output(context.recent_iterations)
        if recent_output:
            sections.append(f"\n## Previous Output (for context):\n```\n{recent_output}\n```")

    if context.errors_and_learnings:
        sections.append(f"\n## Avoid These Issues:\n{context.errors_and_learnings}")

    return "\n\n".join(sections)


# Helper functions

def _build_column_schemas_context(state: DreamTeamState) -> str:
    """Build column schemas context (critical for preventing hallucination)"""
    schemas = state.get("column_schemas", {})

    if not schemas:
        return ""

    context = "## AVAILABLE COLUMNS (use EXACT names, do NOT make up columns):\n"

    for df_name, cols in schemas.items():
        # Format as comma-separated list without brackets
        if isinstance(cols, list):
            cols_formatted = ', '.join(cols)
        else:
            cols_formatted = str(cols)
        context += f"\n**{df_name}**: {cols_formatted}\n"

    context += "\nIMPORTANT: Only use columns listed above. Do not assume other columns exist.\n"

    return context


def _format_iterations_full(iterations: List[IterationResult]) -> str:
    """Format iterations with full details"""
    if not iterations:
        return ""

    lines = []

    for it in iterations:
        iter_num = it["iteration"]
        approach = it["approach"]
        metrics = it.get("metrics", {})
        success = it["results"].get("success", False)

        lines.append(f"### Iteration {iter_num}:")
        lines.append(f"**Approach**: {approach[:200]}..." if len(approach) > 200 else f"**Approach**: {approach}")
        lines.append(f"**Result**: {'✅ Success' if success else '❌ Failed'}")

        if metrics:
            lines.append(f"**Metrics**: {metrics}")

        if not success:
            error = it["results"].get("error", "Unknown error")
            lines.append(f"**Error**: {error[:150]}...")

        # Add output preview
        output = it["results"].get("output", "")
        if output:
            # Smart extraction: get metrics and key results
            preview = _extract_key_results(output)
            lines.append(f"**Key Results**: {preview}")

        lines.append("")

    return "\n".join(lines)


def _summarize_iterations(iterations: List[IterationResult], state: DreamTeamState) -> str:
    """
    Use LLM to summarize older iterations.

    This is the key innovation: instead of truncating, we compress semantically.
    """
    if not iterations:
        return ""

    # Build summary prompt
    iteration_summaries = []
    for it in iterations:
        iter_num = it["iteration"]
        approach = it["approach"]
        metrics = it.get("metrics", {})
        success = it["results"].get("success", False)

        summary = f"Iteration {iter_num}: {approach[:100]}..."
        summary += f" → {'Success' if success else 'Failed'}"
        if metrics:
            summary += f" {metrics}"

        iteration_summaries.append(summary)

    llm = get_llm()

    summary_prompt = f"""Summarize these past iterations from a research experiment.

Iterations:
{chr(10).join(iteration_summaries)}

Provide a concise summary (2-3 sentences) covering:
1. What approaches were tried
2. What worked and what failed
3. Key insights learned

Focus on actionable insights for future work.
"""

    try:
        summary = llm.generate(summary_prompt, temperature=0.3)
        return summary.strip()
    except Exception as e:
        # Fallback: just list them
        return "\n".join(iteration_summaries)


def _extract_errors_and_learnings(iterations: List[IterationResult]) -> str:
    """Extract common errors and learnings from iterations"""
    errors = []
    successes = []

    for it in iterations:
        if not it["results"].get("success"):
            error = it["results"].get("error", "Unknown")
            if error and error not in [e for e, _ in errors]:
                errors.append((error, it["iteration"]))

        else:
            approach = it["approach"]
            metrics = it.get("metrics", {})
            if metrics:
                successes.append((approach[:100], metrics, it["iteration"]))

    lines = []

    if errors:
        lines.append("**Common Errors**:")
        for error, iter_num in errors[-3:]:  # Last 3 errors
            error_short = error[:100] + "..." if len(error) > 100 else error
            lines.append(f"- Iteration {iter_num}: {error_short}")

    if successes:
        lines.append("\n**What Worked**:")
        for approach, metrics, iter_num in successes[-2:]:  # Last 2 successes
            lines.append(f"- Iteration {iter_num}: {approach} → {metrics}")

    return "\n".join(lines) if lines else ""


def _extract_key_results(output: str, max_length: int = 500) -> str:
    """
    Extract key results from execution output.

    Looks for:
    - Metric values
    - Important printed results
    - Model summaries
    """
    lines = output.split('\n')

    # Keywords that indicate important results
    important_keywords = [
        'mae', 'mse', 'rmse', 'r2', 'accuracy', 'f1',
        'test', 'validation', 'score',
        'feature importance', 'best parameters',
        'final', 'result'
    ]

    key_lines = []

    for line in lines:
        line_lower = line.lower()
        if any(kw in line_lower for kw in important_keywords):
            key_lines.append(line.strip())

        # Stop if we have enough
        if len('\n'.join(key_lines)) > max_length:
            break

    if key_lines:
        result = '\n'.join(key_lines)
        if len(result) > max_length:
            result = result[:max_length] + "..."
        return result

    # Fallback: last N chars
    if len(output) > max_length:
        return "..." + output[-max_length:]
    return output


def _extract_recent_output(recent_iterations_text: str, max_length: int = 3000) -> str:
    """Extract recent output from formatted iterations"""
    # Look for "Key Results:" section
    if "Key Results:" in recent_iterations_text:
        parts = recent_iterations_text.split("Key Results:")
        if len(parts) > 1:
            # Get the last occurrence
            last_result = parts[-1].split("###")[0].strip()
            return last_result[:max_length]

    # Fallback
    return ""


# Advanced context strategies

def get_context_strategy(state: DreamTeamState) -> str:
    """
    Determine optimal context strategy based on experiment state.

    Strategies:
    - "bootstrap": Include full bootstrap output (column discovery)
    - "exploration": Keep more history (early iterations)
    - "exploitation": Focus on recent + best (later iterations)
    - "recovery": Include error patterns (after failures)

    Returns:
        Strategy name
    """
    iteration = state.get("iteration", 0)
    error_count = state.get("error_count", 0)

    if iteration <= 1:
        return "bootstrap"
    elif error_count >= 2:
        return "recovery"
    elif iteration <= 3:
        return "exploration"
    else:
        return "exploitation"


def build_adaptive_context(state: DreamTeamState) -> ContextWindow:
    """
    Build context using adaptive strategy.

    Adjusts context based on experiment phase.
    """
    strategy = get_context_strategy(state)

    if strategy == "bootstrap":
        # Keep full bootstrap output
        return build_smart_context(state, keep_recent=1, summarize_older=False)

    elif strategy == "recovery":
        # Focus on errors
        context = build_smart_context(state, keep_recent=3, summarize_older=True)
        # Enhance error section
        return context

    elif strategy == "exploration":
        # Keep more history
        return build_smart_context(state, keep_recent=3, summarize_older=True)

    else:  # exploitation
        # Focus on recent + best
        return build_smart_context(state, keep_recent=2, summarize_older=True)
