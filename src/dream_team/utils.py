"""
Utility functions for Dream Team framework.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from .serialization import RobustJSONEncoder


def save_json(data: Any, filepath: str):
    """Save data to JSON file using robust encoder that handles any Python object"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, cls=RobustJSONEncoder, indent=2)


def load_json(filepath: str) -> Any:
    """Load data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_summaries(discussion_paths: List[Path]) -> List[Dict]:
    """Load summaries from discussion files"""
    summaries = []
    for path in discussion_paths:
        if path.exists():
            summaries.append(load_json(str(path)))
    return summaries
