"""Clock adapters. Satisfy the Clock port."""

from __future__ import annotations


class SystemClock:
    """Wall clock, epoch seconds."""

    def now(self) -> float:
        pass


class ManualClock:
    """A clock advanced by hand, so staleness is testable."""

    def __init__(self, start: float = 0.0) -> None:
        pass

    def now(self) -> float:
        pass

    def advance(self, seconds: float) -> float:
        """Move time forward and return the new value."""
        pass
