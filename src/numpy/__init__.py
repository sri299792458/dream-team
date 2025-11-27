"""Lightweight numpy compatibility stub for offline testing.

This module implements a minimal subset of NumPy used by the project tests.
It is **not** a drop-in replacement for real NumPy and only supports the
functions and dtypes referenced in the codebase.
"""
from __future__ import annotations

import random as _random
import math
import builtins
from typing import Sequence

__all__ = [
    "random",
    "ndarray",
    "array",
    "float32",
    "float64",
    "int64",
    "bool_",
    "integer",
    "floating",
]

# ---------------------------------------------------------------------------
# DType markers
# ---------------------------------------------------------------------------
integer = int
floating = float
bool_ = bool
float32 = float
float64 = float
int64 = int


class ndarray(list):
    """A tiny list-backed stand-in that provides ``tolist`` like NumPy arrays."""

    def tolist(self) -> list:
        return list(self)

    def _apply_scalar(self, other: float, op):
        return ndarray(op(float(x), float(other)) for x in self)

    def _apply_sequence(self, other, op):
        other_list = list(other)
        return ndarray(op(float(a), float(b)) for a, b in zip(self, other_list))

    def __mul__(self, other):
        if isinstance(other, (list, ndarray)):
            return self._apply_sequence(other, lambda a, b: a * b)
        return self._apply_scalar(other, lambda a, b: a * b)

    def __rmul__(self, other: float):
        return self.__mul__(other)

    def __add__(self, other: float):
        return self._apply_scalar(other, lambda a, b: a + b)

    def __radd__(self, other: float):
        return self.__add__(other)


def sqrt(value: float) -> float:
    return math.sqrt(value)


def clip(value, min_value, max_value):
    if isinstance(value, ndarray):
        return ndarray(min(max(float(x), min_value), max_value) for x in value)
    if isinstance(value, (int, float)):
        return min(max(float(value), min_value), max_value)
    try:
        return ndarray(min(max(float(x), min_value), max_value) for x in value)
    except TypeError:
        return value


def arange(start, stop=None, step=1):
    if stop is None:
        start, stop = 0, start
    values = []
    current = float(start)
    while current < float(stop):
        values.append(current)
        current += float(step)
    return ndarray(values)


def log(value):
    return math.log(value)


def sum(values):
    if isinstance(values, ndarray):
        return float(builtins.sum(float(v) for v in values))
    return float(builtins.sum(values))


def mean(values):
    values_list = list(values)
    if not values_list:
        return 0.0
    return float(sum(values_list) / len(values_list))


def var(values):
    values_list = list(values)
    if not values_list:
        return 0.0
    mu = mean(values_list)
    return float(sum((float(v) - mu) ** 2 for v in values_list) / len(values_list))


# ---------------------------------------------------------------------------
# Array helpers
# ---------------------------------------------------------------------------

def array(values: Sequence) -> ndarray:
    return ndarray(values)


# ---------------------------------------------------------------------------
# Random helpers
# ---------------------------------------------------------------------------
class _RandomModule:
    def seed(self, seed: int) -> None:
        _random.seed(seed)

    def randn(self, n: int) -> ndarray:
        # Use Gaussian noise similar to numpy.random.randn
        return ndarray(_random.gauss(0, 1) for _ in range(n))

    def choice(self, seq: Sequence, size: int) -> ndarray:
        """Select ``size`` random elements with replacement from ``seq``."""
        choices = (_random.choice(seq) for _ in range(size))
        return ndarray(choices)


random = _RandomModule()
