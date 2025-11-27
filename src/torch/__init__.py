"""Minimal torch stub for offline environments.

This is sufficient for imports used in the codebase; it does not implement
actual tensor computations.
"""
from __future__ import annotations

class Tensor(list):
    pass


__all__ = ["Tensor"]
