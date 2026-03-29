"""
Base classes for mesh inspection operations.

This module defines the abstract base classes that all mesh inspection
operations must implement, along with metadata structures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Type, Callable
import time

from core.models import (
    MeshInput,
    MeshIssue,
    QualityMetrics,
    IssueCategory,
    IssueSeverity,
    OperationConfig,
)


@dataclass
class OperationMetadata:
    """Metadata describing an operation."""
    operation_id: str
    display_name: str
    description: str
    category: IssueCategory
    supported_libraries: List[str]
    recommended_library: str
    requires_adjacency: bool = False
    complexity: str = "O(n)"  # O(n), O(n log n), O(n^2), etc.
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'operation_id': self.operation_id,
            'display_name': self.display_name,
            'description': self.description,
            'category': self.category.value,
            'supported_libraries': self.supported_libraries,
            'recommended_library': self.recommended_library,
            'requires_adjacency': self.requires_adjacency,
            'complexity': self.complexity,
            'tags': self.tags
        }


class BaseOperation(ABC):
    """
    Abstract base class for all mesh inspection operations.

    Subclasses must implement the 'execute' method and define
    operation metadata through the 'metadata' property.

    Example:
        class MyOperation(BaseOperation):
            @property
            def metadata(self) -> OperationMetadata:
                return OperationMetadata(
                    operation_id='custom.my_operation',
                    display_name='My Custom Operation',
                    description='Does something useful',
                    category=IssueCategory.GEOMETRIC,
                    supported_libraries=['trimesh', 'pyvista'],
                    recommended_library='pyvista'
                )

            def execute(self, mesh: MeshInput) -> List[MeshIssue]:
                # Implementation here
                return []
    """

    def __init__(self, config: Optional[OperationConfig] = None):
        self.config = config or OperationConfig()
        self._execution_count = 0
        self._total_time = 0.0

    @property
    @abstractmethod
    def metadata(self) -> OperationMetadata:
        """Return operation metadata. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        """
        Execute the operation on the input mesh.

        Args:
            mesh: Input mesh in canonical format

        Returns:
            List of detected issues
        """
        pass

    def run(self, mesh: MeshInput) -> List[MeshIssue]:
        """
        Execute operation with timing and error handling.

        This method wraps execute() with timing and error handling,
        providing consistent behavior across all operations.
        """
        start_time = time.perf_counter()
        self._execution_count += 1

        try:
            issues = self.execute(mesh)
            self._total_time += time.perf_counter() - start_time
            return issues
        except Exception as e:
            self._total_time += time.perf_counter() - start_time
            # Wrap exception in operation context
            raise OperationError(
                f"Operation '{self.metadata.operation_id}' failed: {str(e)}"
            ) from e

    @property
    def statistics(self) -> Dict[str, Any]:
        """Return execution statistics."""
        return {
            'operation_id': self.metadata.operation_id,
            'execution_count': self._execution_count,
            'total_time': self._total_time,
            'avg_time': self._total_time / self._execution_count if self._execution_count > 0 else 0
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.metadata.operation_id}>"


class QualityOperation(ABC):
    """
    Abstract base for operations that compute quality metrics.

    These operations return QualityMetrics instead of or in addition
    to MeshIssue objects.
    """

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """Return the name of this quality metric."""
        pass

    @abstractmethod
    def compute(self, mesh: MeshInput) -> QualityMetrics:
        """
        Compute quality metrics for the mesh.

        Args:
            mesh: Input mesh

        Returns:
            QualityMetrics object with computed values
        """
        pass

    @abstractmethod
    def get_threshold(self) -> float:
        """Return the quality threshold for this metric."""
        pass

    def get_recommended_severity(self, violation_rate: float) -> IssueSeverity:
        """
        Determine severity based on violation rate.

        Args:
            violation_rate: Fraction of elements below threshold (0-1)

        Returns:
            Appropriate IssueSeverity
        """
        if violation_rate > 0.5:
            return IssueSeverity.CRITICAL
        elif violation_rate > 0.2:
            return IssueSeverity.ERROR
        elif violation_rate > 0.05:
            return IssueSeverity.WARNING
        else:
            return IssueSeverity.INFO


class OperationError(Exception):
    """Exception raised when an operation fails."""
    pass


# Global registry instance (lazy initialization)
_registry = None


def get_registry() -> 'OperationRegistry':
    """Get the global operation registry."""
    global _registry
    if _registry is None:
        from operations.operation_registry import OperationRegistry
        _registry = OperationRegistry()
    return _registry


def register_operation(operation: BaseOperation) -> BaseOperation:
    """
    Register an operation with the global registry.

    Can be used as a decorator:
        @register_operation
        class MyOperation(BaseOperation):
            ...

    Or called directly:
        register_operation(MyOperation())
    """
    get_registry().register(operation)
    return operation


def _register_decorator(cls: Type[BaseOperation]) -> Type[BaseOperation]:
    """
    Decorator to register an operation class.

    Usage:
        @register_operation
        class MyOperation(BaseOperation):
            ...

    Note: The class will be instantiated before registration.
    """
    registry = get_registry()
    registry.register(cls())
    return cls


# Alias for backward compatibility
register_operation_decorator = _register_decorator
