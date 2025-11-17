"""
Experiment orchestration for autonomous Dream Team operation.

Coordinates agents, code execution, and iterative improvement.
Uses mathematical framework for emergent evolution.
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
from .knowledge_state import KnowledgeGraph, extract_concepts_from_text
from .team import Team


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
        self.team_members = team_members  # Can be empty initially - PI recruits after bootstrap
        self.coding_agent = coding_agent
        self.all_agents = [team_lead] + team_members
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.evolution_engine = evolution_engine or EvolutionEngine()
        self.executor = None  # Created when run() is called
        self.research = get_research_assistant()

        # Get LLM for query generation and other tasks
        from .llm import get_llm
        self.llm = get_llm()

        self.iteration = 0
        self.experiment_history = []
        self.best_metric = None
        self.bootstrap_completed = len(team_members) > 0  # Skip bootstrap if team already exists
        self.column_schemas = {}  # Will be populated during bootstrap

        # Mathematical framework for emergent evolution
        self.problem_graph = None  # KnowledgeGraph extracted from problem
        self.team = None  # Team object for collective dynamics

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
        # Add artifacts_dir so agents can save important objects
        artifacts_dir = self.results_dir / 'artifacts'
        artifacts_dir.mkdir(exist_ok=True)
        if data_context is None:
            data_context = {}
        data_context['artifacts_dir'] = artifacts_dir

        self.executor = CodeExecutor(data_context=data_context)

        # Store problem statement for use in prompts
        self.problem_statement = problem_statement

        # Check for resume
        start_iteration = 1
        if resume:
            resumed = self._try_resume(problem_statement, target_metric, minimize_metric)
            if resumed:
                start_iteration = self.iteration + 1
                print(f"\n✅ Resumed from iteration {self.iteration}")
                print(f"   Best {target_metric} so far: {self.best_metric}")
                print(f"   Starting iteration {start_iteration}\n")
                # If resumed, bootstrap already completed
                self.bootstrap_completed = True

        # Bootstrap phase: PI explores problem and recruits team
        if not self.bootstrap_completed:
            self._bootstrap_exploration(problem_statement)
            self.bootstrap_completed = True
            print("\n" + "="*60)

        # Initialize mathematical framework
        self._initialize_mathematical_framework(problem_statement, target_metric)
        print("Bootstrap complete. Starting team iterations...\n")

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

            # Step 5: Evaluate performance
            metrics = self._extract_metrics(results, target_metric)

            # Step 6: Record iteration
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

            # Update best metric BEFORE printing summary
            self._update_best_metric(metrics, target_metric, minimize_metric)

            # Iteration summary
            print(f"\n{'='*60}")
            print(f"ITERATION {self.iteration} SUMMARY")
            print(f"{'='*60}")
            print(f"Status: {'✅ Success' if results['success'] else '❌ Failed'}")
            if metrics:
                for k, v in metrics.items():
                    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
            else:
                print("No metrics extracted")
            if self.best_metric is not None:
                print(f"Best {target_metric} so far: {self.best_metric:.4f}")
            print(f"{'='*60}\n")

            # Step 6: Check if target achieved
            if self._check_goal_achieved(metrics, target_metric, target_score, minimize_metric):
                print(f"\n🎯 Target achieved! {target_metric}: {metrics.get(target_metric)}")
                break

            # Step 7: Update dynamics and check if evolution needed
            should_evolve = self._check_mathematical_evolution(metrics, target_metric, minimize_metric)

            if should_evolve:
                self._evolve_team(problem_statement, metrics)

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

    def _bootstrap_exploration(self, problem_statement: str):
        """
        Bootstrap phase: PI explores problem and recruits team.

        The PI (team lead) starts alone, explores the data with coding agent,
        sees what the problem is about, then decides what expertise is needed
        and recruits team members.
        """
        print("\n" + "="*60)
        print("BOOTSTRAP: PI Initial Exploration")
        print("="*60)
        print(f"\n{self.team_lead.title} is exploring the problem alone...\n")

        # PI decides what initial exploration is needed
        exploration_task = f"""
You've received a new research problem. Before assembling a team, you need to understand what you're dealing with.

## Problem:
{problem_statement}

## Available Data:
{list(self.executor.data_context.keys())}

## Your Task:
Decide what initial exploration will help you understand:
1. What the data looks like (schemas, sizes, distributions)
2. What the challenge involves
3. What expertise you'll need on your team

In 2-3 sentences, describe what exploration code should be written.
"""

        meeting = IndividualMeeting(save_dir=str(self.results_dir / 'meetings'))
        exploration_plan = meeting.run(
            agent=self.team_lead,
            task=exploration_task,
            num_iterations=1
        )

        print(f"\n{self.team_lead.title}'s exploration plan:\n{exploration_plan}\n")

        # Coding agent implements exploration
        print(f"💻 {self.coding_agent.title} implementing exploration...\n")

        code_task = f"""
The PI wants to do initial exploration. Write Python code to implement this:

## PI's Request:
{exploration_plan}

## Problem Statement (for reference):
{problem_statement}

## Available in execution context:
- Pre-imported libraries: pandas (pd), numpy (np), torch, pathlib.Path
- Variables: {list(self.executor.data_context.keys())}
  (You can use any of these variables directly in your code)

## Requirements:
- Inspect dataframes: print(df.info()), df.head(), df.describe(), df.columns
- ONLY print what you observe - no summaries, interpretations, or conclusions
- Use variables from "Available in execution context" above
- Suppress warnings if needed

Output ONLY the Python code, wrapped in ```python code blocks.
"""

        code_meeting = IndividualMeeting(
            save_dir=str(self.results_dir / 'meetings')
        )
        code_output = code_meeting.run(
            agent=self.coding_agent,
            task=code_task,
            num_iterations=1,
            use_react_coding=True  # Coding agent uses ReAct for iterative reasoning
        )


        code = extract_code_from_text(code_output)

        # Save exploration code
        code_file = self.results_dir / 'code' / 'iteration_00.py'
        code_file.parent.mkdir(exist_ok=True)
        code_file.write_text(code)

        # Execute exploration with retry on failure
        print("⚙️  Executing exploration...\n")
        results = self._execute_with_retry(
            code=code,
            approach=exploration_plan,
            max_retries=2
        )

        if results['success']:
            print("✅ Exploration successful!\n")
            print("Output:")
            print("-" * 60)
            print(results['output'])
            print("-" * 60)

            # Extract column schemas from explored dataframes
            print("\n📋 Extracting column schemas...")
            column_schemas = {}
            for df_name in ['batches_train', 'batches_test', 'products', 'sites', 'regions']:
                df = self.executor.get_variable(df_name)
                if df is not None and hasattr(df, 'columns'):
                    column_schemas[df_name] = list(df.columns)
                    print(f"   {df_name}: {len(df.columns)} columns - {list(df.columns)[:10]}...")

            # Store schemas for use in team meetings
            self.column_schemas = column_schemas
        else:
            print("❌ Exploration failed after retries:")
            print(results['error'])
            # Continue anyway - PI can recruit based on problem statement

        # PI reviews results and recruits team using ReAct
        print(f"\n{self.team_lead.title} reviewing exploration results and recruiting team...\n")

        recruitment_task = f"""
Based on the problem and exploration results, decide what expertise you need on your team.

## Problem:
{problem_statement}

## Exploration Results:
{results['output'][:2000] if results['success'] else "Exploration failed, but you have the problem statement."}

## Your Task:
List 1-3 team members you want to recruit. For each, provide:
- Title (e.g., "ML Strategist", "Domain Expert", "Data Analyst")
- Expertise (what they should know)
- Role (what they'll contribute)

Be specific about the skills needed based on what you learned from exploration and research papers.

Format your response as a simple list, one team member per line.
"""

        # Use ReAct so PI can search papers while thinking about recruitment
        recruitment_meeting = IndividualMeeting(
            save_dir=str(self.results_dir / 'meetings'),
            research_api=self.research.ss_api if hasattr(self, 'research') else None
        )
        recruitment_plan = recruitment_meeting.run(
            agent=self.team_lead,
            task=recruitment_task,
            num_iterations=1,
            use_react=True  # PI uses ReAct to search papers during recruitment
        )

        print(f"Recruitment plan:\n{recruitment_plan}\n")

        # Parse and create team members from PI's plan
        # For now, create ML Strategist as default (user can extend this)
        # In future, could use LLM to parse and create custom agents
        recruited_agents = self._parse_and_recruit(recruitment_plan)

        self.team_members.extend(recruited_agents)
        self.all_agents = [self.team_lead] + self.team_members

        print(f"\n✅ Team assembled! {len(recruited_agents)} member(s) recruited:")
        for agent in recruited_agents:
            print(f"   - {agent.title}")

        # Save bootstrap results
        # Use structure compatible with regular iterations so team can see bootstrap output
        bootstrap_summary = {
            'iteration': 0,
            'phase': 'bootstrap',
            'approach': exploration_plan,  # What was planned
            'results': {  # Match iteration structure so meeting code works
                'success': results['success'],
                'output': results['output'] if results['success'] else results.get('error', ''),
                'error': results.get('error'),
                'traceback': results.get('traceback'),
                'code': code,
                'description': 'Bootstrap exploration'
            },
            'metrics': {},  # No metrics in bootstrap, but include empty dict for consistency
            'agents_snapshot': [self.team_lead.title, self.coding_agent.title],
            'recruitment_plan': recruitment_plan,
            'recruited_agents': [
                {
                    'title': a.title,
                    'expertise': a.expertise,
                    'role': a.role,
                    'goal': a.goal
                }
                for a in recruited_agents
            ]
        }

        # Add to experiment history so iteration 1 can see bootstrap output!
        self.experiment_history.append(bootstrap_summary)

        save_json(bootstrap_summary, self.results_dir / 'iteration_00_bootstrap.json')

    def _parse_and_recruit(self, recruitment_plan: str) -> List[Agent]:
        """
        Parse PI's recruitment plan and create agents.

        Uses LLM to extract agent specifications from PI's plan.
        """
        # Use LLM to parse the recruitment plan and extract agent definitions
        parse_task = f"""
Parse this recruitment plan and extract agent specifications.

## Recruitment Plan:
{recruitment_plan}

## Your Task:
For each team member mentioned, extract:
- Title
- Expertise
- Role

Output in this exact format (one agent per block):

AGENT 1:
Title: [exact title from plan]
Expertise: [expertise description from plan]
Role: [role description from plan]

AGENT 2:
Title: [exact title from plan]
Expertise: [expertise description from plan]
Role: [role description from plan]

Only output the agent specifications, nothing else.
"""

        meeting = IndividualMeeting(save_dir=str(self.results_dir / 'meetings'))
        parsed_output = meeting.run(
            agent=self.team_lead,
            task=parse_task,
            num_iterations=1
        )

        # Parse the structured output and create Agent objects
        agents = []
        current_agent = {}

        for line in parsed_output.split('\n'):
            line = line.strip()

            if line.startswith('Title:'):
                current_agent['title'] = line.replace('Title:', '').strip()
            elif line.startswith('Expertise:'):
                current_agent['expertise'] = line.replace('Expertise:', '').strip()
            elif line.startswith('Role:'):
                current_agent['role'] = line.replace('Role:', '').strip()

                # When we have all three fields, create agent
                if 'title' in current_agent and 'expertise' in current_agent and 'role' in current_agent:
                    agent = Agent(
                        title=current_agent['title'],
                        expertise=current_agent['expertise'],
                        goal=f"contribute specialized expertise to optimize the target metric",
                        role=current_agent['role']
                    )
                    agents.append(agent)
                    current_agent = {}  # Reset for next agent

        # Fallback: if parsing failed, create a generic ML specialist
        if not agents:
            print("   ⚠️  Could not parse recruitment plan, creating default ML Strategist")
            agents = [Agent(
                title="ML Strategist",
                expertise="machine learning, feature engineering, model selection, predictive modeling",
                goal="design effective predictive approaches",
                role="propose modeling strategies and analytical approaches"
            )]

        return agents

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
                output = last['results']['output']
                # For bootstrap (iteration 0), show FIRST 3000 chars to include column info
                # For other iterations, show LAST 15000 chars (enough for team to review)
                if last.get('iteration', 0) == 0:
                    if len(output) > 3000:
                        output_preview = f"\n\nBootstrap Exploration Output (first 3000 chars):\n```\n{output[:3000]}...\n```"
                    else:
                        output_preview = f"\n\nBootstrap Exploration Output:\n```\n{output}\n```"
                else:
                    if len(output) > 15000:
                        output_preview = f"\n\nPrevious Iteration Output (last 15000 chars):\n```\n...{output[-15000:]}\n```"
                    else:
                        output_preview = f"\n\nPrevious Iteration Output:\n```\n{output}\n```"

            # Extract approach preview to avoid slicing syntax issues in f-string
            approach = last['approach']
            approach_preview = approach[:200] + "..." if len(approach) > 200 else approach
            history_context = f"\n## Previous Iteration Results:\nApproach tried: {approach_preview}\nMetrics achieved: {last['metrics']}\n(Note: These are PREVIOUS iteration metrics, not current){output_preview}\n"

        # Research context removed - agents now use ReAct loop during meetings
        # They search papers iteratively as they reason about proposals
        research_context = ""

        # Use column schemas extracted during bootstrap
        columns_summary = ""
        if hasattr(self, 'column_schemas') and self.column_schemas:
            columns_summary = "\n## AVAILABLE COLUMNS (ONLY use these exact column names):\n"
            for df_name, cols in self.column_schemas.items():
                columns_summary += f"\n{df_name}: {cols}\n"

        agenda = f"""
**BE CONCISE.**

## Problem:
{problem_statement}

## Available Dataframes:
{list(self.executor.data_context.keys())}
{columns_summary}
{history_context}
{research_context}

## Roles:
- **Team Members**: Review previous results, then propose what to do next (will use ReAct to search papers and ground proposals)
- **Lead**: Synthesize team's analysis and proposals into clear decisions

## Task:
Team members:
1. First, review the previous iteration - what worked? what failed? what did you learn from the output?
2. Then propose what to implement next based on your expertise and learnings (2-3 sentences). Use ONLY the columns listed above.

Lead: Synthesize the team's analysis and proposals into a decisive action plan.
"""

        # Log agenda summary (not full text - too verbose)
        print("\n📋 TEAM MEETING CONTEXT:")
        print(f"   Dataframes: {list(self.executor.data_context.keys())}")
        if history_context:
            print(f"   Previous metrics: {last['metrics']}")
            # Show what information is available from previous iteration
            if not last['metrics'] or len(last['metrics']) == 0:
                print(f"   ℹ️  No metrics yet - team will see exploration output from iteration {last.get('iteration', 0)}")
                if last['results'].get('output'):
                    output_len = len(last['results']['output'])
                    print(f"   ℹ️  Output available: {output_len} chars (data exploration, column info, statistics)")
        print()

        meeting = TeamMeeting(
            save_dir=str(self.results_dir / 'meetings'),
            research_api=self.research.ss_api if hasattr(self, 'research') else None
        )
        summary = meeting.run(
            team_lead=self.team_lead,
            team_members=self.team_members,
            agenda=agenda,
            num_rounds=1  # Reduced from 2 to 1 for speed
        )

        # Save meeting transcript
        meeting.save(f'iteration_{self.iteration:02d}_team_meeting.json')

        return summary

    def _validate_approach(self, approach: str) -> str:
        """Validate team's proposal against actual data before implementation"""
        print("🔍 Validating team's proposal against actual data...\n")

        # Get exploration output to see what columns actually exist
        exploration_output = ""
        if self.experiment_history and self.experiment_history[0].get('iteration', -1) == 0:
            exploration_output = self.experiment_history[0]['results'].get('output', '')

        validation_task = f"""
You are a critic reviewing a team's proposal. Your job is to validate it against the actual data.

## Team's Proposal:
{approach}

## Exploration Output (what columns ACTUALLY exist):
{exploration_output[:4000]}

## Your Task:
1. Check if the proposal mentions any columns that DON'T exist in the exploration output
2. If columns are hallucinated, identify what actual columns could be used instead
3. Output a CORRECTED version of the proposal using ONLY columns that actually exist

**If the proposal is valid:** Output "VALIDATED: " followed by the original proposal.
**If columns are hallucinated:** Output "CORRECTED: " followed by the corrected proposal, explaining what you changed.

Be concise. Focus only on column name issues.
"""

        validation_meeting = IndividualMeeting(save_dir=str(self.results_dir / 'meetings'))
        validated_approach = validation_meeting.run(
            agent=self.team_lead,  # Use team lead as critic
            task=validation_task,
            num_iterations=1
        )

        # Extract the validated/corrected approach
        if "CORRECTED:" in validated_approach:
            print("⚠️  Critic found issues and corrected the proposal\n")
            corrected = validated_approach.split("CORRECTED:", 1)[1].strip()
            return corrected
        elif "VALIDATED:" in validated_approach:
            print("✅ Critic validated the proposal\n")
            return approach
        else:
            # Fallback: use the validation output as-is
            print("⚠️  Using critic's output\n")
            return validated_approach

    def _implement_approach(self, approach: str) -> str:
        """Have coding agent write code to implement the approach"""
        print(f"💻 {self.coding_agent.title} implementing approach...\n")

        # Include previous iteration output for context (especially exploration results)
        previous_output_context = ""
        if self.experiment_history:
            last = self.experiment_history[-1]
            if last['results'].get('output'):
                output = last['results']['output']

                # For bootstrap, show first 3000 chars (includes column schemas)
                # For iterations, show last 15000 chars (enough context)
                if last.get('iteration', 0) == 0:
                    if len(output) > 3000:
                        previous_output_context = f"\n## Bootstrap Exploration Output (first 3000 chars):\n```\n{output[:3000]}...\n```\n"
                    else:
                        previous_output_context = f"\n## Bootstrap Exploration Output:\n```\n{output}\n```\n"
                else:
                    if len(output) > 15000:
                        previous_output_context = f"\n## Previous Iteration Output (last 15000 chars):\n```\n...{output[-15000:]}\n```\n"
                    else:
                        previous_output_context = f"\n## Previous Iteration Output:\n```\n{output}\n```\n"

        # Build column schema info for coding agent
        schema_info = ""
        if hasattr(self, 'column_schemas') and self.column_schemas:
            schema_info = "\n## DataFrame Schemas (use EXACT column names):\n"
            for df_name, cols in self.column_schemas.items():
                schema_info += f"{df_name}: {cols}\n"

        task = f"""
Implement the team's plan.

## Team's Plan:
{approach}

## Available dataframes:
{list(self.executor.data_context.keys())}
{schema_info}
{previous_output_context}
## Available in execution context:
- Pre-imported libraries: pandas (pd), numpy (np), torch

## Requirements:
- Use GPU when training models
- Write complete, executable code
- Import what you need, define variables
- Use the EXACT column names from DataFrame Schemas above
- If training/evaluating a model, compute MAE and store it in a variable (e.g., mae = ...)
- Print important outputs: metrics, feature importance, model summaries
- Save trained models (e.g., joblib.dump, torch.save) so they can be reused if training took long
- Suppress verbose output: `warnings.filterwarnings('ignore')`, use `verbose=0` or `verbose=-1` in models

Output ONLY Python code in ```python blocks.
"""

        meeting = IndividualMeeting(
            save_dir=str(self.results_dir / 'meetings')
        )
        code_output = meeting.run(
            agent=self.coding_agent,
            task=task,
            num_iterations=1,
            use_react_coding=True  # Coding agent uses ReAct for iterative reasoning
        )

        # Save coding meeting transcript
        meeting.save(f'iteration_{self.iteration:02d}_coding.json')

        # Extract code from output
        code = extract_code_from_text(code_output)

        # Save generated code
        code_file = self.results_dir / 'code' / f'iteration_{self.iteration:02d}.py'
        code_file.parent.mkdir(exist_ok=True)
        code_file.write_text(code)

        # Log code preview
        code_lines = code.split('\n')
        print(f"   Generated {len(code_lines)} lines of code")
        print(f"   Saved to: {code_file}")

        # Show first few imports to see what libraries are being used
        imports = [line for line in code_lines[:20] if line.strip().startswith(('import ', 'from '))]
        if imports:
            print(f"   Libraries: {', '.join([imp.split()[1].split('.')[0] for imp in imports[:5]])}")
        print()

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

            # Check if failure was due to missing package - install and retry automatically
            # This doesn't count against retry limit - it's just installing a dependency
            if 'missing_package' in result:
                package = result['missing_package']
                print(f"   📦 Missing package detected: {package}")
                if self.executor._install_package(package):
                    print(f"   🔄 Retrying after installing {package}...\n")
                    continue  # Retry with same code after installation (doesn't increment attempt)
                else:
                    print(f"   ⚠️ Failed to install {package}, asking agent to use alternative...\n")

            # If failed and we have retries left, ask agent to fix
            if attempt < max_retries:
                print(f"   ❌ Error: {result['error']}")
                print(f"   🔧 Asking agent to fix...\n")
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

        # Include previous iteration output for context
        previous_output_context = ""
        if self.experiment_history:
            last = self.experiment_history[-1]
            if last['results'].get('output'):
                output = last['results']['output']
                if len(output) > 15000:
                    previous_output_context = f"\n## Previous Iteration Output (last 15000 chars):\n```\n...{output[-15000:]}\n```\n"
                else:
                    previous_output_context = f"\n## Previous Iteration Output:\n```\n{output}\n```\n"

        task = f"""
Your code failed with an error. Fix it.

## Original Approach
{approach}

## Problem Statement (for reference):
{self.problem_statement}

## Your Code That Failed
```python
{failed_code}
```

## Error
{error}

## Traceback
{traceback}

## Available in execution context:
- Pre-imported libraries: pandas, numpy, torch, pathlib
- Variables: {list(self.executor.data_context.keys())}
  Note: Missing packages are auto-installed, so if you see ModuleNotFoundError, just wait - it will retry automatically

## DataFrame Schemas (use EXACT column names):
{self._format_column_schemas()}
{previous_output_context}
## Task
The error shows EXACTLY what's wrong. Read the traceback line number.

**For NameError `'X' is not defined`:**
1. Look at the line number in traceback
2. Find where you used variable `X` without defining it first
3. Either: define `X = ...` BEFORE that line, or remove the usage

**For KeyError (column doesn't exist):**
- Check the DataFrame Schemas above for the EXACT column name
- Use only columns that exist in the schemas

**DO NOT output the same code again. Actually fix the specific line that failed.**

Output ONLY the FIXED Python code in ```python blocks.
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

    def _format_column_schemas(self) -> str:
        """Format column schemas for display in prompts"""
        if not hasattr(self, 'column_schemas') or not self.column_schemas:
            return "No schema information available."

        result = ""
        for df_name, cols in self.column_schemas.items():
            result += f"{df_name}: {cols}\n"
        return result

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
        """
        Organically evolve team composition based on what the problem demands.

        PI analyzes current situation and decides:
        - Add new specialists?
        - Remove agents no longer contributing?
        - Deepen expertise of existing agents?
        """
        print("\n🧬 Evolving team composition...\n")

        # Research relevant papers to inform evolution
        print("📚 Researching latest approaches...\n")
        papers_summary = ""
        try:
            # Use LLM to extract academic search terms
            query_extraction_prompt = f"""
Extract 2-3 key academic search terms from this problem for searching research papers.

Problem: {problem_statement[:300]}

Output ONLY the search query (2-5 words, academic terminology, no quotes).
Examples: "shelf life prediction", "gradient boosting regression", "deep learning forecasting"
"""
            search_query = self.llm.generate(query_extraction_prompt, temperature=0.3).strip()
            search_query = search_query.strip('"\'')
            print(f"   Search query: '{search_query}'")

            papers = self.research.research_topic(
                query=search_query,
                context=f"Current performance: {current_metrics}. Looking for new approaches.",
                num_papers=2  # Reduced to avoid rate limiting
            )

            if papers:
                papers_summary = "\n## Research Findings:\n"
                for paper in papers:
                    papers_summary += f"- {paper.title}: {paper.abstract[:120]}...\n"
        except Exception as e:
            print(f"   Note: Research search skipped (API rate limit or error): {e}\n")

        # PI analyzes team composition and decides what changes are needed
        current_team_info = "\n".join([
            f"- {agent.title}: {agent.expertise[:100]}..."
            for agent in self.team_members
        ])

        recent_history = ""
        if len(self.experiment_history) >= 3:
            recent_history = "\n## Recent Progress:\n"
            for hist in self.experiment_history[-3:]:
                metrics_str = hist.get('metrics', {})
                if not metrics_str:
                    metrics_str = "FAILED (no metrics produced)"
                recent_history += f"Iteration {hist['iteration']}: {metrics_str}\n"

        # Determine situation: failure vs plateau
        if not current_metrics or len(current_metrics) == 0:
            situation_desc = "The most recent iteration FAILED to produce any metrics (likely code execution error, wrong column names, or implementation issue)."
        else:
            situation_desc = "Progress has stalled."

        evolution_task = f"""
Analyze the current situation and decide whether team evolution is needed.

## Situation:
{situation_desc}

## Current Team:
{current_team_info if current_team_info else "Only you (PI)"}

## Current Performance:
{current_metrics if current_metrics else "No metrics from last iteration"}

{recent_history}

{papers_summary}

## Your Options:
1. NO CHANGE: Current team is fine, the issue is elsewhere (e.g., implementation bug, not lack of expertise)
2. ADD a new specialist (e.g., "Add Time Series Expert with expertise in...")
3. REMOVE an agent (e.g., "Remove ML Strategist - insights already incorporated")
4. DEEPEN an existing agent (e.g., "Deepen ML Strategist into Deep Learning Specialist with expertise in...")
5. MULTIPLE changes (e.g., "Add X, Remove Y")

## Your Task:
Analyze whether this is a TEAM COMPOSITION issue or an IMPLEMENTATION issue.
If the last iteration failed completely, is that because we lack expertise, or is it a bug that needs fixing?

Specify your decision:
- NO CHANGE: [Reason why current team is adequate]
OR
- ADD: [Title] with expertise in [expertise] to [role]
- REMOVE: [Title] because [reason]
- DEEPEN: [Title] into [New Title] with expertise in [new expertise]

Be strategic - only evolve the team if lack of expertise is the actual problem.
"""

        meeting = IndividualMeeting(save_dir=str(self.results_dir / 'meetings'))
        evolution_plan = meeting.run(
            agent=self.team_lead,
            task=evolution_task,
            num_iterations=1
        )

        print(f"\n{self.team_lead.title}'s evolution plan:\n{evolution_plan}\n")

        # Execute the evolution plan
        self._execute_evolution_plan(evolution_plan, papers)

    def _execute_evolution_plan(self, plan: str, papers: List[Dict]):
        """
        Parse and execute PI's evolution plan.

        Handles NO CHANGE, ADD, REMOVE, DEEPEN commands.
        """
        changes_made = []

        for line in plan.split('\n'):
            line = line.strip()

            # NO CHANGE - PI decided not to evolve team
            if line.upper().startswith('NO CHANGE:'):
                reason = line.split(':', 1)[1].strip() if ':' in line else "Team composition is adequate"
                print(f"\n✋ No team evolution needed: {reason}")
                return  # Exit early - no changes

            # ADD new agent
            elif line.upper().startswith('ADD:'):
                # Parse: "ADD: Time Series Expert with expertise in ... to ..."
                # For now, create generic specialist
                # Future: parse and create custom agent
                new_agent = Agent(
                    title="Domain Specialist",
                    expertise="specialized domain knowledge based on current challenge requirements",
                    goal="provide specialized expertise to break through performance plateau",
                    role="apply domain-specific insights and advanced techniques"
                )
                self.team_members.append(new_agent)
                changes_made.append(f"✅ Added {new_agent.title}")

            # REMOVE agent
            elif line.upper().startswith('REMOVE:'):
                # Parse: "REMOVE: ML Strategist because ..."
                # Extract agent title
                if self.team_members and 'ML Strategist' in line:
                    # Simple heuristic - remove first team member
                    removed = self.team_members.pop(0)
                    changes_made.append(f"✅ Removed {removed.title}")

            # DEEPEN agent
            elif line.upper().startswith('DEEPEN:'):
                # Parse: "DEEPEN: ML Strategist into Time Series Specialist..."
                # Deepen first team member
                if self.team_members:
                    agent = self.team_members[0]
                    old_title = agent.title

                    # Use evolution engine to deepen expertise
                    context = {
                        'problem_description': plan,
                        'deepening': True
                    }
                    self.evolution_engine.evolve_agent(
                        agent=agent,
                        context=context,
                        papers=papers,
                        trigger_reason="Team composition evolution - specialization needed"
                    )

                    changes_made.append(f"✅ Deepened {old_title} → {agent.title}")

        # Update all_agents list
        self.all_agents = [self.team_lead] + self.team_members

        print("\n🔄 Team Evolution Complete:")
        for change in changes_made:
            print(f"   {change}")

        print(f"\n👥 New team composition:")
        print(f"   - {self.team_lead.title} (Lead)")
        for agent in self.team_members:
            print(f"   - {agent.title}")
        print()

        # Save evolution record
        evolution_record = {
            'iteration': self.iteration,
            'evolution_plan': plan,
            'changes': changes_made,
            'new_team': [{'title': a.title, 'expertise': a.expertise} for a in self.all_agents]
        }
        save_json(evolution_record, self.results_dir / f'evolution_iter_{self.iteration}.json')

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

        # Reconstruct team composition from bootstrap
        # Check if iteration 0 (bootstrap) has recruited_agents
        bootstrap_data = next((item for item in self.experiment_history if item.get('iteration') == 0), None)
        if bootstrap_data and 'recruited_agents' in bootstrap_data:
            print("   Reconstructing team from bootstrap...")
            from .agent import Agent
            recruited_agents = bootstrap_data['recruited_agents']

            # Recreate Agent objects
            for agent_data in recruited_agents:
                agent = Agent(
                    title=agent_data['title'],
                    expertise=agent_data['expertise'],
                    role=agent_data.get('role', ''),
                    goal=agent_data.get('goal', '')
                )
                self.team_members.append(agent)
                print(f"   Restored: {agent.title}")

            # Update all_agents
            self.all_agents = [self.team_lead] + self.team_members
            print(f"   ✅ Restored {len(recruited_agents)} team member(s)\n")

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

    # ========== Mathematical Evolution Methods ==========

    def _initialize_mathematical_framework(self, problem_statement: str, target_metric: str):
        """Initialize problem graph and team for mathematical evolution"""
        print("\n🧮 Initializing mathematical framework...")

        # Extract problem as knowledge graph
        self.problem_graph = self._extract_problem_graph(problem_statement, target_metric)
        print(f"   ✓ Problem graph: {len(self.problem_graph.concepts)} concepts")

        # Create team object
        self.team = Team(self.all_agents)
        print(f"   ✓ Team initialized: {len(self.team.agents)} agents")

        # Show initial team state
        diversity = self.team.compute_diversity()
        print(f"   ✓ Team diversity: {diversity:.3f}")
        print()

    def _extract_problem_graph(self, problem_statement: str, target_metric: str) -> KnowledgeGraph:
        """Extract problem as knowledge graph"""
        # Combine problem statement with target metric for better concept extraction
        problem_text = f"{problem_statement} Target metric: {target_metric}"

        # Extract concepts
        concepts = extract_concepts_from_text(problem_text, use_llm=False)

        # Create knowledge graph
        graph = KnowledgeGraph()

        # Add domain-specific important concepts
        important_concepts = {
            'regression', 'classification', 'prediction', 'forecasting',
            'optimization', 'machine_learning', 'deep_learning',
            'gradient_boosting', 'neural_network', 'feature_engineering',
            target_metric.lower().replace('_', ' ')
        }

        # Combine extracted + important
        all_concepts = concepts | important_concepts

        # Add concepts with importance
        for concept in all_concepts:
            # Higher importance for concepts in problem statement
            if concept.lower() in problem_statement.lower():
                importance = 2.0
            elif concept == target_metric.lower().replace('_', ' '):
                importance = 3.0  # Metric is very important
            else:
                importance = 1.0

            graph.add_concept(concept, importance=importance)

        return graph

    def _check_mathematical_evolution(
        self,
        metrics: Dict[str, float],
        target_metric: str,
        minimize: bool
    ) -> bool:
        """Check if evolution needed using mathematical framework"""
        if len(self.experiment_history) < 3:
            return False  # Need history

        # Build metric history
        metric_history = [
            h['metrics'].get(target_metric, float('inf') if minimize else float('-inf'))
            for h in self.experiment_history
        ]

        # Update agent dynamics based on iteration results
        self._update_agent_dynamics(metric_history, minimize)

        # Get team state diagnosis
        state = self.team.diagnose_state(metric_history, minimize=minimize)
        print(f"\n📊 Team state: {state}")

        # Get diversity
        diversity = self.team.compute_diversity()
        print(f"   Team diversity: {diversity:.3f}")

        # Check each agent for evolution signals
        evolution_signals = []
        for agent in self.all_agents:
            should_evolve, evo_type = agent.should_evolve(self.problem_graph, self.team)
            if should_evolve:
                evolution_signals.append((agent, evo_type))
                gini = agent.δ.gini_coefficient()
                effectiveness = agent.contribution_effectiveness()
                print(f"   🔔 {agent.title}: {evo_type} (gini={gini:.2f}, eff={effectiveness:.2f})")

        # Evolution triggered if any agent needs it OR team diagnosed need
        if evolution_signals:
            print(f"\n🔔 Mathematical evolution triggered:")
            for agent, evo_type in evolution_signals:
                print(f"   - {agent.title}: {evo_type}")
            return True

        if state in ["REFRAMING", "EXPLORATION"]:
            print(f"   - Team needs {state}")
            return True

        return False

    def _update_agent_dynamics(self, metric_history: List[float], minimize: bool):
        """Update agent mathematical state based on iteration results"""
        if len(metric_history) < 2:
            return

        # Compute learning quality for each concept
        # Simple heuristic: if metrics improved, quality is high
        recent_improvement = metric_history[-2] - metric_history[-1] if minimize else metric_history[-1] - metric_history[-2]

        # Quality proportional to improvement
        if recent_improvement > 0:
            base_quality = 0.8  # Good iteration
        elif abs(recent_improvement) < 0.01:
            base_quality = 0.5  # Plateau
        else:
            base_quality = 0.3  # Regression

        # All concepts get similar quality (could be more sophisticated)
        learning_quality = {
            concept: base_quality
            for concept in self.problem_graph.concepts
        }

        # Update all agents' dynamics
        self.team.update_all_dynamics(
            problem=self.problem_graph,
            learning_quality=learning_quality,
            dt=0.1
        )

        # Log team state
        team_state = self.team.get_state_summary()
        print(f"\n   📈 Mathematical state updated (diversity: {team_state['diversity']:.3f})")
