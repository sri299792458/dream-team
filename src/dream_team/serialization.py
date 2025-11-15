"""
Robust JSON serialization utilities for Dream Team framework.

Handles serialization of arbitrary Python objects that agents might create.
"""

import json
import numpy as np
import pandas as pd
from typing import Any
from pathlib import Path


class RobustJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles ANY Python object.

    Never fails - converts unsupported types to safe representations.
    """

    def default(self, obj):
        """Convert obj to JSON-serializable format"""

        # NumPy types
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)

        # Pandas types
        elif isinstance(obj, pd.DataFrame):
            return {
                '_type': 'DataFrame',
                'shape': obj.shape,
                'columns': obj.columns.tolist(),
                'head': obj.head(3).to_dict('records') if len(obj) > 0 else []
            }
        elif isinstance(obj, pd.Series):
            return {
                '_type': 'Series',
                'length': len(obj),
                'dtype': str(obj.dtype),
                'head': obj.head(3).tolist() if len(obj) > 0 else []
            }
        elif isinstance(obj, pd.Index):
            return {
                '_type': 'Index',
                'values': obj.tolist()[:10]  # First 10 values
            }

        # Path objects
        elif isinstance(obj, Path):
            return str(obj)

        # Sets
        elif isinstance(obj, set):
            return list(obj)

        # Complex numbers
        elif isinstance(obj, complex):
            return {'real': obj.real, 'imag': obj.imag}

        # Bytes
        elif isinstance(obj, bytes):
            try:
                return obj.decode('utf-8')
            except:
                return f"<bytes: {len(obj)} bytes>"

        # Callables (functions, methods, classes)
        elif callable(obj):
            return {
                '_type': 'callable',
                'name': getattr(obj, '__name__', 'unknown'),
                'module': getattr(obj, '__module__', 'unknown'),
                'class': obj.__class__.__name__
            }

        # Objects with __dict__ (custom classes)
        elif hasattr(obj, '__dict__'):
            return {
                '_type': 'object',
                'class': obj.__class__.__name__,
                'module': obj.__class__.__module__,
                'repr': repr(obj)[:200]  # Truncated representation
            }

        # Fallback: string representation
        else:
            try:
                return {
                    '_type': 'unknown',
                    'class': obj.__class__.__name__,
                    'repr': repr(obj)[:200]
                }
            except:
                return '<unserializable object>'


def robust_dump(data: Any, filepath: Path, **kwargs):
    """
    Save data to JSON file using RobustJSONEncoder.

    Args:
        data: Any Python object
        filepath: Path to save JSON file
        **kwargs: Additional arguments passed to json.dump
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(data, f, cls=RobustJSONEncoder, indent=2, **kwargs)


def robust_dumps(data: Any, **kwargs) -> str:
    """
    Convert data to JSON string using RobustJSONEncoder.

    Args:
        data: Any Python object
        **kwargs: Additional arguments passed to json.dumps

    Returns:
        JSON string
    """
    return json.dumps(data, cls=RobustJSONEncoder, indent=2, **kwargs)


def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize an object for JSON serialization.

    This is useful when you want to pre-process data before JSON encoding.

    Args:
        obj: Any Python object

    Returns:
        JSON-serializable version of the object
    """
    # Use the encoder to convert, then parse back
    # This ensures consistency with the encoder
    json_str = robust_dumps(obj)
    return json.loads(json_str)
