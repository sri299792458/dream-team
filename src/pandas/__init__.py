"""Lightweight pandas compatibility stub for offline testing.

Only implements the minimal surface used by the project code and tests. For
full functionality, install real pandas in environments with network access.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import csv

__all__ = ["DataFrame", "Series", "Index"]


class Index(list):
    def tolist(self) -> list:
        return list(self)


class Series(list):
    def __init__(self, data: Iterable[Any]):
        super().__init__(data)
        self.dtype = type(next(iter(self), None)).__name__ if self else "object"

    def head(self, n: int = 5) -> "Series":
        return Series(self[:n])


class DataFrame:
    def __init__(self, data: Optional[Dict[str, Iterable[Any]]] = None):
        data = data or {}
        self._data: Dict[str, List[Any]] = {k: list(v) for k, v in data.items()}
        # Normalize column lengths
        max_len = max((len(v) for v in self._data.values()), default=0)
        for k, v in self._data.items():
            if len(v) < max_len:
                v.extend([None] * (max_len - len(v)))
        self.columns = Index(list(self._data.keys()))

    # Basic metadata helpers
    def __len__(self) -> int:
        return len(next(iter(self._data.values()), []))

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self), len(self.columns))

    def head(self, n: int = 5) -> "DataFrame":
        return DataFrame({k: v[:n] for k, v in self._data.items()})

    def to_dict(self, orient: str = "records") -> Any:
        if orient != "records":
            raise ValueError("Stub DataFrame only supports orient='records'")
        records = []
        for i in range(len(self)):
            records.append({k: self._data[k][i] for k in self.columns})
        return records

    def to_csv(self, path: str | Path, index: bool = False) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.columns))
            writer.writeheader()
            for row in self.to_dict("records"):
                writer.writerow(row)

    # Pretty-print stubs
    def info(self) -> str:
        return f"StubDataFrame(rows={len(self)}, cols={len(self.columns)})"

    def describe(self) -> "DataFrame":
        return self.head(1)


# Convenience constructor mirrors pandas API

def DataFrame_from_dict(data: Dict[str, Iterable[Any]]) -> DataFrame:
    return DataFrame(data)


# Expose constructors at module level
DataFrame = DataFrame
Series = Series
Index = Index
