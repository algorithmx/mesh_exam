"""
Central registry for mesh inspection operations.

Provides operation lookup, composition, and execution tracking.
"""

from typing import Dict, List, Type, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict

from core.models import (
    MeshInput,
    InspectionResult,
    IssueCategory,
    IssueSeverity,
    OperationConfig,
)
from operations.base import BaseOperation, OperationMetadata


@dataclass
class OperationCategory:
    """Container for operations grouped by category."""
    name: str
    operations: Dict[str, BaseOperation] = field(default_factory=dict)

    def register(self, operation: BaseOperation):
        """Register an operation in this category."""
        self.operations[operation.metadata.operation_id] = operation

    def get(self, operation_id: str) -> Optional[BaseOperation]:
        """Get operation by ID."""
        return self.operations.get(operation_id)

    def list_operations(self) -> List[OperationMetadata]:
        """List all operations in this category."""
        return [op.metadata for op in self.operations.values()]

    def __len__(self) -> int:
        return len(self.operations)

    def __iter__(self):
        return iter(self.operations.values())


class OperationRegistry:
    """
    Central registry for all mesh inspection operations.

    Provides operation lookup, composition, and execution tracking.
    Operations are automatically organized by category.

    Usage:
        registry = OperationRegistry()
        registry.register(MyOperation())

        # Get operation
        op = registry.get('custom.my_operation')

        # List all
        all_ops = registry.list_all()

        # Create workflow
        workflow = registry.create_workflow(['op1', 'op2', 'op3'])
        result = workflow(mesh)
    """

    def __init__(self):
        self._categories: Dict[IssueCategory, OperationCategory] = {}
        self._operations: Dict[str, BaseOperation] = {}
        self._composites: Dict[str, Callable] = {}
        self._initialize_categories()

    def _initialize_categories(self):
        """Initialize operation categories."""
        for category in IssueCategory:
            self._categories[category] = OperationCategory(
                name=category.value,
                operations={}
            )

    def register(self, operation: BaseOperation) -> 'OperationRegistry':
        """
        Register an operation.

        The operation's metadata determines its category automatically.

        Args:
            operation: Operation instance to register

        Returns:
            self for method chaining
        """
        op_id = operation.metadata.operation_id
        category = operation.metadata.category

        # Validate operation has unique ID
        if op_id in self._operations:
            raise ValueError(f"Operation with ID '{op_id}' already registered")

        # Register in main dictionary
        self._operations[op_id] = operation

        # Register in category
        self._categories[category].register(operation)

        return self

    def unregister(self, operation_id: str) -> bool:
        """
        Unregister an operation by ID.

        Args:
            operation_id: ID of operation to remove

        Returns:
            True if operation was removed, False if not found
        """
        if operation_id in self._operations:
            operation = self._operations.pop(operation_id)
            category = operation.metadata.category
            del self._categories[category].operations[operation_id]
            return True
        return False

    def get(self, operation_id: str) -> Optional[BaseOperation]:
        """Get operation by ID."""
        return self._operations.get(operation_id)

    def get_by_category(self, category: IssueCategory) -> OperationCategory:
        """Get all operations in a category."""
        return self._categories.get(category)

    def list_all(self) -> List[OperationMetadata]:
        """List metadata for all registered operations."""
        return [op.metadata for op in self._operations.values()]

    def list_by_category(self, category: IssueCategory) -> List[OperationMetadata]:
        """List operations in a specific category."""
        return self._categories[category].list_operations()

    def list_by_library(self, library: str) -> List[OperationMetadata]:
        """List operations supported by a specific library."""
        return [
            op.metadata for op in self._operations.values()
            if library in op.metadata.supported_libraries
        ]

    def list_by_tag(self, tag: str) -> List[OperationMetadata]:
        """List operations with a specific tag."""
        return [
            op.metadata for op in self._operations.values()
            if tag in op.metadata.tags
        ]

    def create_workflow(
        self,
        operation_ids: List[str],
        combine_results: bool = True,
        config: Optional[OperationConfig] = None
    ) -> Callable[[MeshInput], InspectionResult]:
        """
        Create a composed workflow from multiple operations.

        Args:
            operation_ids: List of operation IDs to compose
            combine_results: Whether to merge results into single report
            config: Optional configuration for all operations

        Returns:
            Callable that executes the composed workflow

        Raises:
            ValueError: If any operation ID is not found
        """
        operations = []
        for oid in operation_ids:
            op = self.get(oid)
            if op is None:
                raise ValueError(f"Unknown operation: {oid}")
            if config is not None:
                op.config = config
            operations.append(op)

        def workflow(mesh: MeshInput) -> InspectionResult:
            """Execute composed workflow on mesh."""
            result = InspectionResult(
                mesh_id=mesh.mesh_id,
                file_path=getattr(mesh, 'file_path', None)
            )

            import time
            start_time = time.perf_counter()

            for operation in operations:
                issues = operation.run(mesh)
                result.issues.extend(issues)

            result.processing_time = time.perf_counter() - start_time

            if combine_results:
                result.compute_summary()

            return result

        return workflow

    def create_category_workflow(
        self,
        category: IssueCategory,
        config: Optional[OperationConfig] = None
    ) -> Callable[[MeshInput], InspectionResult]:
        """
        Create a workflow for all operations in a category.

        Args:
            category: Category to include
            config: Optional configuration

        Returns:
            Workflow callable
        """
        ops_in_category = list(self._categories[category].operations.keys())
        return self.create_workflow(ops_in_category, config=config)

    def create_full_workflow(
        self,
        config: Optional[OperationConfig] = None
    ) -> Callable[[MeshInput], InspectionResult]:
        """
        Create a workflow for all registered operations.

        Args:
            config: Optional configuration

        Returns:
            Workflow callable
        """
        all_ops = list(self._operations.keys())
        return self.create_workflow(all_ops, config=config)

    def get_recommended_library(self, operation_id: str) -> Optional[str]:
        """Get the recommended library for an operation."""
        operation = self.get(operation_id)
        if operation:
            return operation.metadata.recommended_library
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregated statistics for all operations."""
        total_executions = sum(op._execution_count for op in self._operations.values())
        total_time = sum(op._total_time for op in self._operations.values())

        category_stats = {}
        for category, cat_ops in self._categories.items():
            category_stats[category.value] = {
                'operation_count': len(cat_ops),
                'executions': sum(op._execution_count for op in cat_ops.operations.values())
            }

        return {
            'total_operations': len(self._operations),
            'total_executions': total_executions,
            'total_time': total_time,
            'by_category': category_stats
        }

    def __len__(self) -> int:
        return len(self._operations)

    def __contains__(self, operation_id: str) -> bool:
        return operation_id in self._operations

    def __iter__(self):
        return iter(self._operations.values())


# Module-level registry instance
registry = OperationRegistry()
