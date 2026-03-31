"""
Base classes for mesh inspection operations.
"""

from .base import (
    BaseOperation,
    QualityOperation,
    OperationMetadata,
    register_operation,
)

__all__ = [
    "BaseOperation",
    "QualityOperation",
    "OperationMetadata",
    "register_operation",
]
