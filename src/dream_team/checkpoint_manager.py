"""
Checkpoint management for Dream Team experiments.

Provides:
- Persistent checkpoints using SqliteSaver
- Resume from last checkpoint
- List and inspect checkpoints
- Clean up old checkpoints
"""

import atexit
from pathlib import Path
from typing import Optional, List, Dict, Any
import sqlite3
from datetime import datetime

from langgraph.checkpoint.sqlite import SqliteSaver


class CheckpointManager:
    """
    Manager for LangGraph checkpoints.

    Handles checkpoint creation, listing, and resumption.
    """

    def __init__(self, checkpoint_dir: Path):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint database
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.checkpoint_dir / "checkpoints.db"

    def get_checkpointer(self) -> SqliteSaver:
        """
        Get SqliteSaver for this experiment.

        Returns:
            SqliteSaver instance
        """
        saver_candidate = SqliteSaver.from_conn_string(str(self.db_path))

        # langgraph>=0.2 wraps SqliteSaver.from_conn_string in a contextmanager
        # to ensure connections close cleanly. Materialize the saver so callers
        # receive the expected object with get_next_version/put capabilities.
        if hasattr(saver_candidate, "__enter__") and hasattr(saver_candidate, "__exit__"):
            saver_context = saver_candidate
            saver = saver_context.__enter__()
            atexit.register(saver_context.__exit__, None, None, None)
            return saver

        return saver_candidate

    def has_checkpoints(self, thread_id: str) -> bool:
        """
        Check if thread has any checkpoints.

        Args:
            thread_id: Thread ID to check

        Returns:
            True if checkpoints exist
        """
        if not self.db_path.exists():
            return False

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                SELECT COUNT(*) FROM checkpoints
                WHERE thread_id = ?
            """, (thread_id,))

            count = cursor.fetchone()[0]
            conn.close()

            return count > 0

        except Exception as e:
            print(f"Error checking checkpoints: {e}")
            return False

    def get_latest_checkpoint_id(self, thread_id: str) -> Optional[str]:
        """
        Get ID of latest checkpoint for thread.

        Args:
            thread_id: Thread ID

        Returns:
            Checkpoint ID or None
        """
        if not self.db_path.exists():
            return None

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                SELECT checkpoint_id FROM checkpoints
                WHERE thread_id = ?
                ORDER BY checkpoint_id DESC
                LIMIT 1
            """, (thread_id,))

            row = cursor.fetchone()
            conn.close()

            return row[0] if row else None

        except Exception as e:
            print(f"Error getting latest checkpoint: {e}")
            return None

    def list_checkpoints(self, thread_id: str) -> List[Dict[str, Any]]:
        """
        List all checkpoints for thread.

        Args:
            thread_id: Thread ID

        Returns:
            List of checkpoint metadata
        """
        if not self.db_path.exists():
            return []

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                SELECT checkpoint_id, checkpoint_ns, parent_checkpoint_id
                FROM checkpoints
                WHERE thread_id = ?
                ORDER BY checkpoint_id DESC
            """, (thread_id,))

            checkpoints = []
            for row in cursor.fetchall():
                checkpoints.append({
                    "checkpoint_id": row[0],
                    "namespace": row[1],
                    "parent_id": row[2]
                })

            conn.close()
            return checkpoints

        except Exception as e:
            print(f"Error listing checkpoints: {e}")
            return []

    def get_checkpoint_state(self, thread_id: str, checkpoint_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get state at specific checkpoint.

        Args:
            thread_id: Thread ID
            checkpoint_id: Checkpoint ID (None for latest)

        Returns:
            State dictionary or None
        """
        checkpointer = self.get_checkpointer()

        config = {"configurable": {"thread_id": thread_id}}

        if checkpoint_id:
            config["configurable"]["checkpoint_id"] = checkpoint_id

        try:
            checkpoint_tuple = checkpointer.get_tuple(config)

            if checkpoint_tuple:
                # checkpoint_tuple is (config, checkpoint, metadata, parent_config)
                checkpoint = checkpoint_tuple.checkpoint
                return checkpoint.get("channel_values", {})

        except Exception as e:
            print(f"Error getting checkpoint state: {e}")

        return None

    def delete_checkpoints(self, thread_id: str):
        """
        Delete all checkpoints for thread.

        Args:
            thread_id: Thread ID
        """
        if not self.db_path.exists():
            return

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                DELETE FROM checkpoints
                WHERE thread_id = ?
            """, (thread_id,))

            conn.commit()
            conn.close()

            print(f"Deleted checkpoints for thread: {thread_id}")

        except Exception as e:
            print(f"Error deleting checkpoints: {e}")

    def clean_old_checkpoints(self, keep_latest: int = 10):
        """
        Clean up old checkpoints, keeping only N latest per thread.

        Args:
            keep_latest: Number of latest checkpoints to keep per thread
        """
        if not self.db_path.exists():
            return

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Get all threads
            cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
            threads = [row[0] for row in cursor.fetchall()]

            total_deleted = 0

            for thread_id in threads:
                # Get checkpoints for this thread
                cursor.execute("""
                    SELECT checkpoint_id FROM checkpoints
                    WHERE thread_id = ?
                    ORDER BY checkpoint_id DESC
                """, (thread_id,))

                checkpoints = [row[0] for row in cursor.fetchall()]

                # Delete old ones
                if len(checkpoints) > keep_latest:
                    to_delete = checkpoints[keep_latest:]

                    for checkpoint_id in to_delete:
                        cursor.execute("""
                            DELETE FROM checkpoints
                            WHERE thread_id = ? AND checkpoint_id = ?
                        """, (thread_id, checkpoint_id))
                        total_deleted += 1

            conn.commit()
            conn.close()

            if total_deleted > 0:
                print(f"Cleaned up {total_deleted} old checkpoints")

        except Exception as e:
            print(f"Error cleaning checkpoints: {e}")

    def export_checkpoint(self, thread_id: str, output_file: Path, checkpoint_id: Optional[str] = None):
        """
        Export checkpoint state to JSON for backup/inspection.

        Args:
            thread_id: Thread ID
            output_file: Output file path
            checkpoint_id: Checkpoint ID (None for latest)
        """
        state = self.get_checkpoint_state(thread_id, checkpoint_id)

        if not state:
            print("Checkpoint not found")
            return

        import json

        export_data = {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id or "latest",
            "exported_at": datetime.now().isoformat(),
            "state": state
        }

        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        print(f"Checkpoint exported to: {output_file}")

    def print_resume_info(self, thread_id: str):
        """
        Print information about resumable state.

        Args:
            thread_id: Thread ID
        """
        if not self.has_checkpoints(thread_id):
            print(f"No checkpoints found for thread: {thread_id}")
            return

        state = self.get_checkpoint_state(thread_id)

        if not state:
            print("Could not load checkpoint state")
            return

        print("=" * 80)
        print(f"RESUMABLE CHECKPOINT: {thread_id}")
        print("=" * 80)

        # Extract key information
        iteration = state.get('iteration', 'unknown')
        bootstrap = state.get('bootstrap_completed', False)
        best_metric = state.get('best_metric')
        target_metric = state.get('target_metric', 'N/A')

        print(f"\nCurrent State:")
        print(f"  Iteration: {iteration}")
        print(f"  Bootstrap completed: {bootstrap}")

        if best_metric is not None:
            print(f"  Best {target_metric}: {best_metric:.4f}")

        # Team info
        team_lead = state.get('team_lead', {})
        team_members = state.get('team_members', [])

        print(f"\nTeam:")
        print(f"  Lead: {team_lead.get('title', 'N/A')}")
        print(f"  Members: {[m.get('title', 'Unknown') for m in team_members]}")

        # History
        history = state.get('experiment_history', [])
        print(f"\nProgress: {len(history)} iterations completed")

        print(f"\nTo resume: Run with the same thread_id")
        print("=" * 80)


# Utility functions

def get_checkpoint_manager(results_dir: Path) -> CheckpointManager:
    """
    Get checkpoint manager for results directory.

    Args:
        results_dir: Results directory

    Returns:
        CheckpointManager instance
    """
    checkpoint_dir = results_dir / "checkpoints"
    return CheckpointManager(checkpoint_dir)


def check_resume_available(results_dir: Path, thread_id: str) -> bool:
    """
    Check if resume is available for experiment.

    Args:
        results_dir: Results directory
        thread_id: Thread ID

    Returns:
        True if can resume
    """
    manager = get_checkpoint_manager(results_dir)
    return manager.has_checkpoints(thread_id)
