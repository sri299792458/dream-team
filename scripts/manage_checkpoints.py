#!/usr/bin/env python3
"""
Checkpoint management CLI for Dream Team experiments.

Usage:
    python manage_checkpoints.py list <results_dir> <thread_id>
    python manage_checkpoints.py inspect <results_dir> <thread_id>
    python manage_checkpoints.py export <results_dir> <thread_id> <output_file>
    python manage_checkpoints.py delete <results_dir> <thread_id>
    python manage_checkpoints.py clean <results_dir> [--keep N]
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dream_team.checkpoint_manager import CheckpointManager


def cmd_list(args):
    """List checkpoints for thread"""
    manager = CheckpointManager(Path(args.results_dir) / "checkpoints")

    checkpoints = manager.list_checkpoints(args.thread_id)

    if not checkpoints:
        print(f"No checkpoints found for thread: {args.thread_id}")
        return

    print(f"Checkpoints for thread '{args.thread_id}':")
    print()

    for i, cp in enumerate(checkpoints, 1):
        print(f"{i}. ID: {cp['checkpoint_id']}")
        print(f"   Namespace: {cp['namespace']}")
        print(f"   Parent: {cp['parent_id']}")
        print()


def cmd_inspect(args):
    """Inspect checkpoint state"""
    manager = CheckpointManager(Path(args.results_dir) / "checkpoints")

    manager.print_resume_info(args.thread_id)


def cmd_export(args):
    """Export checkpoint to JSON"""
    manager = CheckpointManager(Path(args.results_dir) / "checkpoints")

    output_file = Path(args.output_file)
    manager.export_checkpoint(args.thread_id, output_file)


def cmd_delete(args):
    """Delete checkpoints for thread"""
    manager = CheckpointManager(Path(args.results_dir) / "checkpoints")

    if not args.force:
        response = input(f"Delete all checkpoints for '{args.thread_id}'? (y/n): ").strip().lower()
        if response != 'y' and response != 'yes':
            print("Cancelled")
            return

    manager.delete_checkpoints(args.thread_id)


def cmd_clean(args):
    """Clean old checkpoints"""
    manager = CheckpointManager(Path(args.results_dir) / "checkpoints")

    manager.clean_old_checkpoints(keep_latest=args.keep)


def main():
    parser = argparse.ArgumentParser(description="Manage Dream Team checkpoints")

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # list command
    list_parser = subparsers.add_parser('list', help='List checkpoints for thread')
    list_parser.add_argument('results_dir', help='Results directory')
    list_parser.add_argument('thread_id', help='Thread ID')

    # inspect command
    inspect_parser = subparsers.add_parser('inspect', help='Inspect checkpoint state')
    inspect_parser.add_argument('results_dir', help='Results directory')
    inspect_parser.add_argument('thread_id', help='Thread ID')

    # export command
    export_parser = subparsers.add_parser('export', help='Export checkpoint to JSON')
    export_parser.add_argument('results_dir', help='Results directory')
    export_parser.add_argument('thread_id', help='Thread ID')
    export_parser.add_argument('output_file', help='Output file path')

    # delete command
    delete_parser = subparsers.add_parser('delete', help='Delete checkpoints for thread')
    delete_parser.add_argument('results_dir', help='Results directory')
    delete_parser.add_argument('thread_id', help='Thread ID')
    delete_parser.add_argument('-f', '--force', action='store_true', help='Force without confirmation')

    # clean command
    clean_parser = subparsers.add_parser('clean', help='Clean old checkpoints')
    clean_parser.add_argument('results_dir', help='Results directory')
    clean_parser.add_argument('--keep', type=int, default=10, help='Number of checkpoints to keep per thread (default: 10)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Execute command
    if args.command == 'list':
        cmd_list(args)
    elif args.command == 'inspect':
        cmd_inspect(args)
    elif args.command == 'export':
        cmd_export(args)
    elif args.command == 'delete':
        cmd_delete(args)
    elif args.command == 'clean':
        cmd_clean(args)


if __name__ == "__main__":
    main()
