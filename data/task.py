"""Task configuration for the ALBench-S2F application layer.

``TaskConfig`` is defined in ``albench.task`` (the standalone AL engine);
this module re-exports it so that application code in ``data/`` can import it
from the canonical location without creating a circular dependency.
"""

from albench.task import TaskConfig  # noqa: F401

__all__ = ["TaskConfig"]
