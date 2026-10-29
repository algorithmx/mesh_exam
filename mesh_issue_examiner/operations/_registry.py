"""
Global registry singleton — separated from base.py to avoid circular imports.

This module exists solely to break the circular import chain:
    base.py → _registry.py → operation_registry.py → base.py

By deferring the import of OperationRegistry into the function body,
base.py can be fully loaded before operation_registry.py tries to
import from it.

The primary singleton is the module-level ``registry`` in
``operation_registry.py``.  This module provides a lazy accessor
used by ``base.register_operation()``.
"""

_registry = None


def _clear_cache():
    """Reset the cached singleton so the next get_registry() re-resolves.

    Used by ``operation_registry.reset_registry()`` so the lazy accessor
    picks up a freshly created registry after a reset.
    """
    global _registry
    _registry = None


def get_registry():
    """Get the global operation registry singleton.

    Returns the *same* instance as ``operation_registry.registry``
    rather than constructing a new one, so that operations registered
    via ``@register_operation`` appear in the registry used by
    the MCP server and workflows.
    """
    global _registry
    if _registry is None:
        from .operation_registry import registry as _reg
        _registry = _reg
    return _registry
