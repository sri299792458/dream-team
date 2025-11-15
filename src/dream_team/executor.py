"""
Code execution module for Dream Team framework.

Provides safe execution environment for agent-generated code.
"""

import sys
import io
import traceback
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path


class CodeExecutor:
    """Executes Python code in a controlled environment"""

    def __init__(self, data_context: Dict[str, Any] = None):
        """
        Initialize executor with data context.

        Args:
            data_context: Dictionary of data/variables available to executed code
        """
        self.data_context = data_context or {}
        self.execution_history = []

    def execute(
        self,
        code: str,
        description: str = "",
        timeout: int = 300
    ) -> Dict[str, Any]:
        """
        Execute Python code and return results.

        Args:
            code: Python code to execute
            description: Description of what this code does
            timeout: Timeout in seconds (not enforced yet, for future use)

        Returns:
            {
                'success': bool,
                'output': stdout output,
                'error': error message if failed,
                'variables': dict of new variables created,
                'metrics': extracted metrics if any
            }
        """
        print(f"\n⚙️ Executing: {description or 'Code block'}")

        # Capture stdout
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        # Prepare execution environment
        exec_globals = {
            'pd': pd,
            'np': np,
            'Path': Path,
            '__builtins__': __builtins__,
        }

        # Add data context
        exec_globals.update(self.data_context)

        exec_locals = {}

        result = {
            'success': False,
            'output': '',
            'error': None,
            'variables': {},
            'metrics': {},
            'code': code,
            'description': description
        }

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            # Execute code
            exec(code, exec_globals, exec_locals)

            result['success'] = True
            result['output'] = stdout_capture.getvalue()

            # Extract new variables (skip private/builtin)
            result['variables'] = {
                k: v for k, v in exec_locals.items()
                if not k.startswith('_')
            }

            # Try to extract metrics (look for common metric variables)
            metric_names = ['mae', 'rmse', 'f1', 'accuracy', 'score', 'cv_scores', 'error']
            result['metrics'] = {
                k: v for k, v in exec_locals.items()
                if any(metric in k.lower() for metric in metric_names)
            }

            print(f"   ✅ Success")
            if result['output']:
                print(f"   Output: {result['output'][:200]}...")

        except Exception as e:
            result['success'] = False
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            result['output'] = stdout_capture.getvalue()

            print(f"   ❌ Error: {e}")

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        # Update data context with new variables
        if result['success']:
            self.data_context.update(exec_locals)

        # Store in history
        self.execution_history.append(result)

        return result

    def execute_with_retry(
        self,
        code: str,
        description: str = "",
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Execute code with automatic retry on failure.

        Returns the result of first successful execution or last failure.
        """
        for attempt in range(max_retries + 1):
            result = self.execute(code, description)
            if result['success']:
                return result

            if attempt < max_retries:
                print(f"   🔄 Retry {attempt + 1}/{max_retries}")

        return result

    def get_variable(self, name: str) -> Any:
        """Get a variable from the execution context"""
        return self.data_context.get(name)

    def set_variable(self, name: str, value: Any):
        """Set a variable in the execution context"""
        self.data_context[name] = value

    def get_metrics_history(self) -> list:
        """Extract all metrics from execution history"""
        metrics = []
        for execution in self.execution_history:
            if execution['metrics']:
                metrics.append({
                    'description': execution['description'],
                    'metrics': execution['metrics']
                })
        return metrics

    def clear_history(self):
        """Clear execution history"""
        self.execution_history = []

    def summary(self) -> str:
        """Get execution summary"""
        total = len(self.execution_history)
        successful = sum(1 for e in self.execution_history if e['success'])
        failed = total - successful

        return f"Executions: {total} total, {successful} successful, {failed} failed"


def extract_code_from_text(text: str) -> str:
    """
    Extract Python code from markdown code blocks or plain text.

    Args:
        text: Text that may contain code blocks

    Returns:
        Extracted Python code
    """
    # Try to find markdown code blocks
    if "```python" in text or "```" in text:
        parts = text.split("```")
        for i, part in enumerate(parts):
            if part.startswith("python\n") or part.startswith("python "):
                return part[6:].strip()
            elif i % 2 == 1:  # Odd indices are code blocks
                return part.strip()

    # If no code blocks, return as-is
    return text.strip()
