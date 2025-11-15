"""
Experiment orchestration for autonomous Dream Team operation.

Coordinates agents, code execution, and iterative improvement.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np

from .agent import Agent
from .executor import CodeExecutor, extract_code_from_text
from .meetings import TeamMeeting, IndividualMeeting
from .evolution import EvolutionEngine, EvolutionTrigger
from .research import get_research_assistant
from .utils import save_json, load_json


class ExperimentOrchestrator:
    """Orchestrates autonomous experimentation with evolving agents"""

    def __init__(
        self,
        team_lead: Agent,
        team_members: List[Agent],
        coding_agent: Agent,
        results_dir: Path,
        evolution_engine: Optional[EvolutionEngine] = None
    ):
        """
        Initialize orchestrator.

        Args:
            team_lead: Lead agent who coordinates
            team_members: Other agents on the team (strategists, domain experts)
            coding_agent: Dedicated agent who implements code based on team discussions
            results_dir: Directory to save results
            evolution_engine: Engine for agent evolution
        """
        self.team_lead = team_lead
        self.team_members = team_members
        self.coding_agent = coding_agent
        self.all_agents = [team_lead] + team_members
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.evolution_engine = evolution_engine or EvolutionEngine()
        self.executor = None  # Created when run() is called
        self.research = get_research_assistant()

        self.iteration = 0
        self.experiment_history = []
        self.best_metric = None

    def run(
        self,
        problem_statement: str,
        data_context: Dict[str, Any],
        target_metric: str,
        minimize_metric: bool = True,
        max_iterations: int = 5,
        target_score: Optional[float] = None,
        resume: bool = True
    ) -> Dict[str, Any]:
        """
        Run autonomous experimentation.

        Args:
            problem_statement: Description of the challenge
            data_context: Dictionary with data (e.g., {'train_df': df, 'test_df': df})
            target_metric: Name of metric to optimize (e.g., 'mae', 'f1')
            minimize_metric: Whether lower is better
            max_iterations: Maximum iterations before stopping
            target_score: Optional target score to achieve
            resume: Whether to resume from previous checkpoint (default: True)

        Returns:
            Final experiment summary
        """
        print("="*60)
        print("🚀 AUTONOMOUS DREAM TEAM EXPERIMENT")
        print("="*60)
        print(f"\nProblem: {problem_statement[:100]}...")
        print(f"Target Metric: {target_metric} ({'minimize' if minimize_metric else 'maximize'})")
        print(f"Max Iterations: {max_iterations}")
        if target_score:
            print(f"Target Score: {target_score}")
        print()

        # Initialize executor with data
        self.executor = CodeExecutor(data_context=data_context)

        # Check for resume
        start_iteration = 1
        if resume:
            resumed = self._try_resume(problem_statement, target_metric, minimize_metric)
            if resumed:
                start_iteration = self.iteration + 1
                print(f"\n✅ Resumed from iteration {self.iteration}")
                print(f"   Best {target_metric} so far: {self.best_metric}")
                print(f"   Starting iteration {start_iteration}\n")

        # Main iteration loop
        for self.iteration in range(start_iteration, max_iterations + 1):
            print(f"\n{'='*60}")
            print(f"ITERATION {self.iteration}/{max_iterations}")
            print(f"{'='*60}\n")

            # Step 1: Team meeting to discuss approach
            approach = self._team_planning_meeting(problem_statement)

            # Step 2: Agent implements the approach (writes code)
            implementation = self._implement_approach(approach)

            # Step 3: Execute code and get results (with automatic error recovery)
            results = self._execute_with_retry(implementation, approach, max_retries=2)

            # Step 4: Evaluate performance
            metrics = self._extract_metrics(results, target_metric)

            # Step 5: Record iteration
            # Extract only serializable parts of results
            serializable_results = {
                'success': results['success'],
                'output': results['output'],
                'error': results.get('error'),
                'traceback': results.get('traceback'),
                'code': results['code'],
                'description': results['description']
            }

            iteration_summary = {
                'iteration': self.iteration,
                'approach': approach,
                'results': serializable_results,
                'metrics': metrics,
                'agents_snapshot': [a.title for a in self.all_agents]
            }
            self.experiment_history.append(iteration_summary)

            # Save iteration results
            save_json(
                iteration_summary,
                self.results_dir / f'iteration_{self.iteration:02d}.json'
            )

            # Step 6: Check if target achieved
            if self._check_goal_achieved(metrics, target_metric, target_score, minimize_metric):
                print(f"\n🎯 Target achieved! {target_metric}: {metrics.get(target_metric)}")
                break

            # Step 7: Check if evolution needed
            should_evolve = self._check_evolution_triggers(metrics, target_metric, minimize_metric)

            if should_evolve:
                self._evolve_team(problem_statement, metrics)

            # Step 8: Update best metric
            self._update_best_metric(metrics, target_metric, minimize_metric)

        # Final summary
        final_summary = self._generate_final_summary()
        save_json(final_summary, self.results_dir / 'final_summary.json')

        print(f"\n{'='*60}")
        print("✅ EXPERIMENT COMPLETE")
        print(f"{'='*60}")
        print(f"\nTotal Iterations: {self.iteration}")
        print(f"Best {target_metric}: {self.best_metric}")
        print(f"Results saved to: {self.results_dir}\n")

        return final_summary

    def _team_planning_meeting(self, problem_statement: str) -> str:
        """Run team meeting to plan approach"""
        print("👥 Team planning meeting...\n")

        # Get current context
        history_context = ""
        if self.experiment_history:
            last = self.experiment_history[-1]

            # Build history context with output from previous iteration
            output_preview = ""
            if last['results'].get('output'):
                # Show last 1000 chars of output (most recent results)
                output = last['results']['output']
                if len(output) > 1000:
                    output_preview = f"\n\nOutput (last 1000 chars):\n```\n...{output[-1000:]}\n```"
                else:
                    output_preview = f"\n\nOutput:\n```\n{output}\n```"

            history_context = f"\n## Previous Iteration:\nApproach: {last['approach'][:200]}...\nMetrics: {last['metrics']}{output_preview}\n"

        agenda = f"""
**BE CONCISE.** Decide what to implement this iteration.

## Problem:
{problem_statement}

## Available Data:
{list(self.executor.data_context.keys())}

{history_context}

## Your Task:
In 2-3 sentences, describe what needs to be implemented this iteration.
Focus on WHAT to do, not HOW to code it.

A coding agent will receive your discussion and implement it.

Keep your response SHORT and ACTION-ORIENTED.
"""

        meeting = TeamMeeting(save_dir=str(self.results_dir / 'meetings'))
        summary = meeting.run(
            team_lead=self.team_lead,
            team_members=self.team_members,
            agenda=agenda,
            num_rounds=1  # Reduced from 2 to 1 for speed
        )

        return summary

    def _implement_approach(self, approach: str) -> str:
        """Have coding agent write code to implement the approach"""
        print(f"💻 {self.coding_agent.title} implementing approach...\n")

        task = f"""
The team has discussed what to implement. Write Python code to implement their plan.

## Team's Discussion:
{approach}

## Available in execution context:
- Libraries: pandas (pd), numpy (np), pathlib.Path
- Variables: {list(self.executor.data_context.keys())}
  (You can use any of these variables directly in your code)

## Requirements:
- Write complete, executable Python code that implements what the team discussed
- Include print statements for key results
- Store metrics in variables (e.g., mae, cv_scores, f1_score)
- Variables you create will persist to the next iteration

Output ONLY the Python code, wrapped in ```python code blocks.
"""

        meeting = IndividualMeeting(save_dir=str(self.results_dir / 'meetings'))
        code_output = meeting.run(
            agent=self.coding_agent,
            task=task,
            num_iterations=1
        )

        # Extract code from output
        code = extract_code_from_text(code_output)

        # Save generated code
        code_file = self.results_dir / 'code' / f'iteration_{self.iteration:02d}.py'
        code_file.parent.mkdir(exist_ok=True)
        code_file.write_text(code)

        print(f"   Code saved to: {code_file}\n")

        return code

    def _execute_implementation(self, code: str) -> Dict[str, Any]:
        """Execute the generated code"""
        print("⚙️ Executing implementation...\n")

        result = self.executor.execute(
            code=code,
            description=f"Iteration {self.iteration} implementation"
        )

        if not result['success']:
            print(f"   ❌ Execution failed: {result['error']}\n")
            print(f"   Traceback:\n{result['traceback']}\n")

        return result

    def _execute_with_retry(self, code: str, approach: str, max_retries: int = 2) -> Dict[str, Any]:
        """
        Execute code with automatic error recovery.

        If execution fails, give the error to the agent and ask for a fix.
        Retry up to max_retries times.

        Args:
            code: Initial code to execute
            approach: The approach description (for context)
            max_retries: Maximum number of retry attempts

        Returns:
            Execution results (final attempt)
        """
        current_code = code
        attempt = 0

        while attempt <= max_retries:
            if attempt > 0:
                print(f"   🔄 Retry attempt {attempt}/{max_retries}\n")

            # Execute code
            result = self._execute_implementation(current_code)

            # If successful, return
            if result['success']:
                if attempt > 0:
                    print(f"   ✅ Fixed after {attempt} attempt(s)!\n")
                return result

            # If failed and we have retries left, ask agent to fix
            if attempt < max_retries:
                print(f"   🔧 Asking agent to fix the error...\n")
                current_code = self._fix_code_error(
                    failed_code=current_code,
                    error=result['error'],
                    traceback=result.get('traceback', ''),
                    approach=approach
                )

                # Save the fixed code attempt
                code_file = self.results_dir / 'code' / f'iteration_{self.iteration:02d}_retry_{attempt+1}.py'
                code_file.parent.mkdir(exist_ok=True)
                code_file.write_text(current_code)
                print(f"   Fixed code saved to: {code_file}\n")

            attempt += 1

        # Max retries exhausted, return last failed result
        print(f"   ⚠️ Max retries ({max_retries}) exhausted. Moving on with failure.\n")
        return result

    def _fix_code_error(self, failed_code: str, error: str, traceback: str, approach: str) -> str:
        """
        Ask coding agent to fix code that failed execution.

        Args:
            failed_code: The code that failed
            error: Error message
            traceback: Full traceback
            approach: Original approach description

        Returns:
            Fixed code
        """
        print(f"   🔧 {self.coding_agent.title} fixing error...\n")

        task = f"""
Your code failed with an error. Fix it.

## Original Approach
{approach}

## Your Code That Failed
```python
{failed_code}
```

## Error
{error}

## Traceback
{traceback}

## Task
Analyze the error and fix the code. Common issues:
- Missing imports
- Incorrect variable names
- Data type mismatches
- Index errors
- Division by zero

Output ONLY the FIXED Python code, wrapped in ```python code blocks.
"""

        meeting = IndividualMeeting(save_dir=str(self.results_dir / 'meetings'))
        code_output = meeting.run(
            agent=self.coding_agent,
            task=task,
            num_iterations=1
        )

        # Extract fixed code
        fixed_code = extract_code_from_text(code_output)

        return fixed_code

    def _extract_metrics(self, results: Dict[str, Any], target_metric: str) -> Dict[str, float]:
        """Extract metrics from execution results, ensuring JSON-serializable values only"""
        raw_metrics = results.get('metrics', {})

        # Filter to only keep JSON-serializable numeric values
        metrics = {}
        for key, value in raw_metrics.items():
            try:
                # Only keep simple numeric types
                if isinstance(value, (int, float, np.integer, np.floating)):
                    metrics[key] = float(value)
                elif isinstance(value, (list, np.ndarray)):
                    # For arrays, take the mean
                    metrics[key] = float(np.mean(value))
            except (TypeError, ValueError, AttributeError):
                # Skip non-numeric or non-serializable values
                pass

        # Try to find target metric in variables if not in metrics
        if target_metric not in metrics:
            for key, value in results.get('variables', {}).items():
                if target_metric in key.lower():
                    try:
                        # Handle arrays (take mean)
                        if hasattr(value, '__iter__') and not isinstance(value, str):
                            metrics[target_metric] = float(np.mean(value))
                        else:
                            metrics[target_metric] = float(value)
                        break
                    except (TypeError, ValueError, AttributeError):
                        pass

        return metrics

    def _check_goal_achieved(
        self,
        metrics: Dict[str, float],
        target_metric: str,
        target_score: Optional[float],
        minimize: bool
    ) -> bool:
        """Check if target score achieved"""
        if not target_score or target_metric not in metrics:
            return False

        current = metrics[target_metric]

        if minimize:
            return current <= target_score
        else:
            return current >= target_score

    def _check_evolution_triggers(
        self,
        metrics: Dict[str, float],
        target_metric: str,
        minimize: bool
    ) -> bool:
        """Check if agents should evolve"""
        if len(self.experiment_history) < 3:
            return False  # Need history to detect plateau

        # Build context for triggers
        metric_history = [
            h['metrics'].get(target_metric, float('inf') if minimize else float('-inf'))
            for h in self.experiment_history
        ]

        context = {
            'metric_history': metric_history,
            'minimize_metric': minimize
        }

        triggers = self.evolution_engine.check_triggers(context)

        if triggers:
            print(f"\n🔔 Evolution triggers detected:")
            for trigger_name, reason in triggers:
                print(f"   - {trigger_name}: {reason}")
            return True

        return False

    def _evolve_team(self, problem_statement: str, current_metrics: Dict[str, float]):
        """Evolve team members based on current challenges"""
        print("\n🧬 Evolving team...\n")

        # Research relevant papers
        print("📚 Researching papers...\n")
        papers = self.research.research_topic(
            query=problem_statement[:200],
            context=f"Current performance: {current_metrics}",
            num_papers=3
        )

        # Evolve first team member (or could evolve all)
        if self.team_members:
            agent = self.team_members[0]

            context = {
                'problem_description': problem_statement,
                'performance_metrics': current_metrics,
                'iteration': self.iteration
            }

            self.evolution_engine.evolve_agent(
                agent=agent,
                context=context,
                papers=papers,
                trigger_reason=f"Performance plateau at iteration {self.iteration}"
            )

            # Save evolved agent
            agent.save(
                self.results_dir / 'agents' / f'{agent.title.lower().replace(" ", "_")}_iter_{self.iteration}.json'
            )

    def _update_best_metric(
        self,
        metrics: Dict[str, float],
        target_metric: str,
        minimize: bool
    ):
        """Update best metric seen so far"""
        if target_metric not in metrics:
            return

        current = metrics[target_metric]

        if self.best_metric is None:
            self.best_metric = current
        elif minimize and current < self.best_metric:
            self.best_metric = current
            print(f"\n✨ New best {target_metric}: {current:.4f}")
        elif not minimize and current > self.best_metric:
            self.best_metric = current
            print(f"\n✨ New best {target_metric}: {current:.4f}")

    def _try_resume(self, problem_statement: str, target_metric: str, minimize: bool) -> bool:
        """
        Try to resume from previous experiment.

        Returns: True if resumed, False if starting fresh
        """
        # Check if there are any iteration files
        iteration_files = sorted(self.results_dir.glob('iteration_*.json'))

        if not iteration_files:
            print("📝 No previous experiment found. Starting fresh.\n")
            return False

        print(f"🔄 Found previous experiment with {len(iteration_files)} iterations")
        print("   Resuming from checkpoint...\n")

        # Load all iteration summaries
        for iter_file in iteration_files:
            iteration_data = load_json(iter_file)
            self.experiment_history.append(iteration_data)

            # Update iteration counter
            self.iteration = iteration_data['iteration']

            # Update best metric
            if 'metrics' in iteration_data and target_metric in iteration_data['metrics']:
                metric_value = iteration_data['metrics'][target_metric]
                if self.best_metric is None:
                    self.best_metric = metric_value
                elif minimize and metric_value < self.best_metric:
                    self.best_metric = metric_value
                elif not minimize and metric_value > self.best_metric:
                    self.best_metric = metric_value

        # Re-execute all code to rebuild executor state
        print("   Rebuilding execution context...")
        code_files = sorted(self.results_dir.glob('code/iteration_*.py'))

        for code_file in code_files:
            iter_num = int(code_file.stem.split('_')[-1])
            code = code_file.read_text()

            print(f"   Re-executing iteration {iter_num}...")
            result = self.executor.execute(
                code=code,
                description=f"Resume: Iteration {iter_num}"
            )

            if not result['success']:
                print(f"   ⚠️  Warning: Iteration {iter_num} failed on re-execution")
                print(f"   Error: {result['error']}")
                # Continue anyway - maybe environment changed

        # Load agent states
        agent_files = sorted(self.results_dir.glob('agents/*.json'))
        if agent_files:
            # Load most recent agent states
            latest_agents = {}
            for agent_file in agent_files:
                # Parse filename to get agent name and iteration
                parts = agent_file.stem.rsplit('_iter_', 1)
                if len(parts) == 2:
                    agent_name = parts[0]
                    iter_num = int(parts[1])

                    if agent_name not in latest_agents or iter_num > latest_agents[agent_name][1]:
                        latest_agents[agent_name] = (agent_file, iter_num)

            # Load the latest version of each agent
            for agent_name, (agent_file, iter_num) in latest_agents.items():
                # Find matching agent in team
                for agent in self.all_agents:
                    if agent.title.lower().replace(" ", "_") == agent_name:
                        agent.load(agent_file)
                        print(f"   Loaded {agent.title} (iteration {iter_num})")
                        break

        print(f"\n   ✅ Successfully resumed from iteration {self.iteration}")

        return True

    def _generate_final_summary(self) -> Dict[str, Any]:
        """Generate final experiment summary"""
        return {
            'total_iterations': self.iteration,
            'best_metric': self.best_metric,
            'final_team': [
                {
                    'title': a.title,
                    'expertise': a.expertise,
                    'specialization_depth': a.specialization_depth
                }
                for a in self.all_agents
            ],
            'iteration_history': self.experiment_history,
            'execution_summary': self.executor.summary()
        }
