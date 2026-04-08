"""
Base classes for mesh inspection operations.
"""

from .base import (
    BaseOperation,
    QualityOperation,
    OperationMetadata,
    OperationError,
    register_operation,
)

# Import operation subpackages to trigger self-registration.
# Each concrete operation module calls registry.register() at import time.
from . import topological    # noqa: F401
from . import quality        # noqa: F401
from . import volumetric     # noqa: F401

__all__ = [
    "BaseOperation",
    "QualityOperation",
    "OperationMetadata",
    "OperationError",
    "register_operation",
]
