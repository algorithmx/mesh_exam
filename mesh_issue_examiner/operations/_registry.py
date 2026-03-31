"""
Global registry singleton — separated from base.py to avoid circular imports.
"""

_registry = None


def get_registry():
    """Get the global operation registry (lazy initialization)."""
    global _registry
    if _registry is None:
        from .operation_registry import OperationRegistry
        _registry = OperationRegistry()
    return _registry
