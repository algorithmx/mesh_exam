"""
Mesh Issue Examiner MCP Framework

A comprehensive framework for automated mesh issue identification and quality assessment.
Designed for MCP (Model Context Protocol) server integration.
"""

__version__ = "0.1.0"

from .core.models import (
    MeshInput,
    MeshIssue,
    QualityMetrics,
    InspectionResult,
    IssueCategory,
    IssueSeverity,
    MeshDimensionality,
    OperationConfig,
)

# Import the operations package — this triggers operations/__init__.py
# which imports subpackages (topological, quality, volumetric) to
# self-register all built-in operations into the registry.
from . import operations  # noqa: F401

from .operations.operation_registry import registry
from .operations.base import register_operation

__all__ = [
    "MeshInput",
    "MeshIssue",
    "QualityMetrics",
    "InspectionResult",
    "IssueCategory",
    "IssueSeverity",
    "MeshDimensionality",
    "OperationConfig",
    "registry",
    "register_operation",
]
