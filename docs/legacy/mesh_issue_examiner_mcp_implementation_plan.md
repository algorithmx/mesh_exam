# Mesh Issue Examiner MCP Implementation Plan

## 1. Executive Summary

This document presents a strategic implementation plan for building an MCP (Model Context Protocol) server dedicated to automated mesh issue identification and quality assessment. The implementation leverages three carefully selected Python libraries—**PyVista**, **TRimesh**, and **PyMeshLab**—to provide comprehensive coverage of elementary mesh inspection operations. This framework delivers composable, atomic operations that can be orchestrated into complex validation workflows for industrial simulation, 3D printing preparation, and scientific visualization pipelines.

## 2. Library Selection Rationale

### 2.1 Selection Criteria

The library selection process evaluated candidates against the following criteria essential for a robust MCP mesh examiner:

- **Functional Coverage**: Ability to perform topological, geometric, and quality-based mesh inspections
- **Integration Capability**: Clean Python API compatible with MCP server architecture
- **Format Support**: Broad file format compatibility for industrial workflows
- **Performance Characteristics**: Suitable for server-side batch processing
- **License Compatibility**: Permissive licensing for commercial deployment
- **Maintenance Status**: Active development with community support

### 2.2 Selected Libraries

| Library | Primary Role | Strengths | License | Best For |
|---------|--------------|-----------|---------|----------|
| **PyVista** | Quality Analysis Engine | VTK-backed cell quality metrics, boundary analysis, visualization | MIT | Element shape quality, skewness, aspect ratio computation |
| **TRimesh** | Topology Operations Core | Pure Python, NumPy integration, easy extension, watertight surface focus | MIT | Non-manifold detection, connectivity analysis, duplicate geometry |
| **PyMeshLab** | Advanced Filtering & Repair | 30+ mesh filters, C++ performance, batch processing paradigm | GPL | Mesh repair, complex filtering workflows, format conversion |

### 2.3 Capability Matrix

| Capability Area | PyVista | TRimesh | PyMeshLab |
|-----------------|---------|---------|-----------|
| Non-manifold edge detection | Limited | Excellent | Excellent |
| Element quality metrics | Excellent | Good | Moderate |
| Self-intersection detection | Moderate | Excellent | Excellent |
| Normal/orientation analysis | Good | Excellent | Moderate |
| Volume/area computation | Excellent | Excellent | Moderate |
| Mesh repair operations | Limited | Moderate | Excellent |
| Format support breadth | Good | Excellent | Excellent |
| Performance (large meshes) | Excellent | Good | Excellent |
| MCP integration ease | Excellent | Excellent | Moderate |

## 3. System Architecture

### 3.1 Architecture Overview

The Mesh Issue Examiner MCP follows a layered architecture pattern that separates concerns while maintaining flexibility for operation composition:

```
┌─────────────────────────────────────────────────────────────────┐
│                     MCP Server Layer                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ Tool Router │  │  Workflow   │  │   Result Aggregator      │ │
│  │             │  │  Composer   │  │                          │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                    Operation Registry                            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │    Atomic Operations (Topological | Geometric | Quality)     ││
│  └─────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────┤
│                    Library Abstraction Layer                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   PyVista   │  │   TRimesh   │  │       PyMeshLab          │ │
│  │   Adapter   │  │   Adapter   │  │        Adapter           │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                      Data Models                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  MeshInput  │  │ IssueReport │  │    QualityMetrics        │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Responsibilities

**MCP Server Layer**: Handles protocol-level communication, tool registration, and request routing. Provides the interface through which MCP clients invoke mesh inspection operations.

**Operation Registry**: Central catalog of all available atomic operations, indexed by category and capability. Supports dynamic operation discovery and composition.

**Library Abstraction Layer**: Encapsulates library-specific implementations behind common interfaces, enabling transparent operation routing to the most suitable backend.

**Data Models**: Defines canonical data structures for mesh inputs, inspection results, and quality metrics, ensuring consistent data flow across operations.

## 4. Code Framework

### 4.1 Project Structure

```
mesh_issue_examiner/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── models.py                 # Data models and schemas
│   ├── operation_registry.py     # Operation catalog and routing
│   └── exceptions.py             # Custom exception definitions
├── operations/
│   ├── __init__.py
│   ├── base.py                   # Abstract base for operations
│   ├── topological/
│   │   ├── __init__.py
│   │   ├── nonmanifold_edges.py
│   │   ├── nonmanifold_vertices.py
│   │   ├── free_edges.py
│   │   ├── disconnected_components.py
│   │   └── duplicate_geometry.py
│   ├── geometric/
│   │   ├── __init__.py
│   │   ├── degenerate_triangles.py
│   │   ├── degenerate_tets.py
│   │   ├── self_intersections.py
│   │   └── intersections.py
│   └── quality/
│       ├── __init__.py
│       ├── aspect_ratio.py
│       ├── skewness.py
│       ├── warping.py
│       └── cell_quality.py
├── adapters/
│   ├── __init__.py
│   ├── base_adapter.py           # Abstract adapter interface
│   ├── trimesh_adapter.py
│   ├── pyvista_adapter.py
│   └── pymeshlab_adapter.py
├── mcp/
│   ├── __init__.py
│   ├── server.py                # MCP server implementation
│   ├── tools.py                 # MCP tool definitions
│   └── handlers.py              # Request handlers
└── utils/
    ├── __init__.py
    ├── mesh_loader.py
    └── metrics.py
```

### 4.2 Core Data Models

```python
# core/models.py
"""
Canonical data models for mesh issue examination.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple
import numpy as np


class IssueSeverity(Enum):
    """Severity classification for mesh issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class IssueCategory(Enum):
    """Categories of mesh issues."""
    TOPOLOGICAL = "topological"
    GEOMETRIC = "geometric"
    QUALITY = "quality"
    ORIENTATION = "orientation"


@dataclass
class Vertex:
    """Represents a 3D vertex."""
    index: int
    coordinates: Tuple[float, float, float]
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Face:
    """Represents a mesh face (triangle or polygon)."""
    indices: List[int]
    normal: Optional[Tuple[float, float, float]] = None
    area: Optional[float] = None


@dataclass
class Edge:
    """Represents a mesh edge."""
    vertices: Tuple[int, int]
    face_count: int
    incident_faces: List[int] = field(default_factory=list)


@dataclass
class IssueLocation:
    """Location information for a mesh issue."""
    element_type: str  # 'vertex', 'edge', 'face', 'cell'
    element_index: int
    coordinates: Optional[Tuple[float, float, float]] = None
    adjacent_elements: Dict[str, List[int]] = field(default_factory=dict)


@dataclass
class MeshIssue:
    """Represents a detected mesh issue."""
    operation_id: str
    category: IssueCategory
    severity: IssueSeverity
    message: str
    locations: List[IssueLocation] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert issue to dictionary representation."""
        return {
            'operation_id': self.operation_id,
            'category': self.category.value,
            'severity': self.severity.value,
            'message': self.message,
            'location_count': len(self.locations),
            'metrics': self.metrics,
            'metadata': self.metadata
        }


@dataclass
class QualityMetrics:
    """Container for mesh quality metrics."""
    operation_id: str
    metric_name: str
    values: np.ndarray
    statistics: Dict[str, float] = field(default_factory=dict)
    threshold: Optional[float] = None
    violation_count: int = 0

    def __post_init__(self):
        """Compute statistics from values array."""
        if len(self.values) > 0:
            self.statistics = {
                'min': float(np.min(self.values)),
                'max': float(np.max(self.values)),
                'mean': float(np.mean(self.values)),
                'median': float(np.median(self.values)),
                'std': float(np.std(self.values)),
                'p5': float(np.percentile(self.values, 5)),
                'p95': float(np.percentile(self.values, 95))
            }
            if self.threshold is not None:
                self.violation_count = int(np.sum(self.values < self.threshold))


@dataclass
class InspectionResult:
    """Complete result of mesh inspection."""
    mesh_id: str
    file_path: Optional[str]
    issues: List[MeshIssue] = field(default_factory=list)
    quality_metrics: List[QualityMetrics] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0

    def add_issue(self, issue: MeshIssue):
        """Add an issue to the result."""
        self.issues.append(issue)

    def add_metrics(self, metrics: QualityMetrics):
        """Add quality metrics to the result."""
        self.quality_metrics.append(metrics)

    def compute_summary(self):
        """Compute summary statistics."""
        severity_counts = {s.value: 0 for s in IssueSeverity}
        category_counts = {c.value: 0 for c in IssueCategory}

        for issue in self.issues:
            severity_counts[issue.severity.value] += 1
            category_counts[issue.category.value] += 1

        self.summary = {
            'total_issues': len(self.issues),
            'severity_breakdown': severity_counts,
            'category_breakdown': category_counts,
            'quality_metric_count': len(self.quality_metrics),
            'has_critical': severity_counts[IssueSeverity.CRITICAL.value] > 0,
            'has_errors': severity_counts[IssueSeverity.ERROR.value] > 0,
            'is_watertight': all(
                i.category != IssueCategory.TOPOLOGICAL for i in self.issues
            )
        }


@dataclass
class OperationConfig:
    """Configuration for mesh operations."""
    tolerance: float = 1e-6
    quality_threshold: float = 0.1
    enable_caching: bool = True
    max_iterations: int = 1000
    parallel_enabled: bool = True
    library_preference: Optional[str] = None  # 'trimesh', 'pyvista', 'pymeshlab'


class MeshInput:
    """
    Canonical mesh representation for MCP operations.
    Wraps library-specific mesh objects for transparent interchange.
    """

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        mesh_id: str = "",
        source_format: str = ""
    ):
        self.mesh_id = mesh_id or f"mesh_{id(self)}"
        self.vertices = vertices
        self.faces = faces
        self.source_format = source_format
        self._native_objects: Dict[str, Any] = {}
        self._adjacency_cache: Dict[str, Any] = {}

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    @property
    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (min_coords, max_coords) bounding box."""
        return np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)

    def get_native(self, library: str) -> Any:
        """Get library-specific mesh object, constructing if needed."""
        if library not in self._native_objects:
            self._construct_native(library)
        return self._native_objects[library]

    def _construct_native(self, library: str):
        """Construct library-specific mesh object."""
        if library == 'trimesh':
            import trimesh
            self._native_objects['trimesh'] = trimesh.Trimesh(
                vertices=self.vertices,
                faces=self.faces
            )
        elif library == 'pyvista':
            import pyvista
            self._native_objects['pyvista'] = pyvista.PolyData(
                self.vertices,
                np.column_stack([
                    np.full(len(self.faces), 3),
                    self.faces
                ])
            )
        elif library == 'pymeshlab':
            import pymeshlab
            ms = pymeshlab.MeshSet()
            mesh = pymeshlab.Mesh(
                vertex_matrix=self.vertices.tolist(),
                face_matrix=self.faces.tolist()
            )
            ms.add_mesh(mesh)
            self._native_objects['pymeshlab'] = ms
        else:
            raise ValueError(f"Unsupported library: {library}")

    def invalidate_cache(self):
        """Clear cached data when mesh is modified."""
        self._adjacency_cache.clear()
```

### 4.3 Operation Registry and Base Classes

```python
# operations/base.py
"""
Base classes for mesh inspection operations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Type
import time

from core.models import (
    MeshInput, MeshIssue, QualityMetrics,
    IssueCategory, IssueSeverity, OperationConfig
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


class BaseOperation(ABC):
    """
    Abstract base class for all mesh inspection operations.

    Subclasses must implement the 'execute' method and define
    operation metadata through the 'metadata' property.
    """

    def __init__(self, config: Optional[OperationConfig] = None):
        self.config = config or OperationConfig()
        self._execution_count = 0
        self._total_time = 0.0

    @property
    @abstractmethod
    def metadata(self) -> OperationMetadata:
        """Return operation metadata."""
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
        """Execute operation with timing and error handling."""
        start_time = time.perf_counter()
        self._execution_count += 1

        try:
            issues = self.execute(mesh)
            self._total_time += time.perf_counter() - start_time
            return issues
        except Exception as e:
            self._total_time += time.perf_counter() - start_time
            raise

    @property
    def statistics(self) -> Dict[str, Any]:
        """Return execution statistics."""
        return {
            'execution_count': self._execution_count,
            'total_time': self._total_time,
            'avg_time': self._total_time / self._execution_count if self._execution_count > 0 else 0
        }


class QualityOperation(ABC):
    """
    Abstract base for operations that compute quality metrics.
    """

    @abstractmethod
    def compute(self, mesh: MeshInput) -> QualityMetrics:
        """Compute quality metrics for the mesh."""
        pass

    @abstractmethod
    def get_threshold(self) -> float:
        """Return the quality threshold for this metric."""
        pass
```

```python
# operations/operation_registry.py
"""
Central registry for mesh inspection operations.
"""

from typing import Dict, List, Type, Optional, Callable
from dataclasses import dataclass, field

from operations.base import BaseOperation, OperationMetadata, QualityOperation
from core.models import IssueCategory


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


class OperationRegistry:
    """
    Central registry for all mesh inspection operations.

    Provides operation lookup, composition, and execution tracking.
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

        Returns self for method chaining.
        """
        op_id = operation.metadata.operation_id
        category = operation.metadata.category

        self._operations[op_id] = operation
        self._categories[category].register(operation)

        return self

    def get(self, operation_id: str) -> Optional[BaseOperation]:
        """Get operation by ID."""
        return self._operations.get(operation_id)

    def get_by_category(self, category: IssueCategory) -> OperationCategory:
        """Get all operations in a category."""
        return self._categories.get(category)

    def list_all(self) -> List[OperationMetadata]:
        """List metadata for all registered operations."""
        return [op.metadata for op in self._operations.values()]

    def list_by_library(self, library: str) -> List[OperationMetadata]:
        """List operations supported by a specific library."""
        return [
            op.metadata for op in self._operations.values()
            if library in op.metadata.supported_libraries
        ]

    def create_workflow(
        self,
        operation_ids: List[str],
        combine_results: bool = True
    ) -> Callable:
        """
        Create a composed workflow from multiple operations.

        Args:
            operation_ids: List of operation IDs to compose
            combine_results: Whether to merge results into single report

        Returns:
            Callable that executes the composed workflow
        """
        operations = [self.get(oid) for oid in operation_ids]
        if any(op is None for op in operations):
            missing = [oid for oid, op in zip(operation_ids, operations) if op is None]
            raise ValueError(f"Unknown operations: {missing}")

        def workflow(mesh):
            from core.models import InspectionResult
            result = InspectionResult(
                mesh_id=mesh.mesh_id,
                file_path=getattr(mesh, 'file_path', None)
            )

            for operation in operations:
                issues = operation.run(mesh)
                result.issues.extend(issues)

            if combine_results:
                result.compute_summary()

            return result

        return workflow

    def get_recommended_library(self, operation_id: str) -> Optional[str]:
        """Get the recommended library for an operation."""
        operation = self.get(operation_id)
        if operation:
            return operation.metadata.recommended_library
        return None


# Global registry instance
registry = OperationRegistry()


def register_operation(operation_class: Type[BaseOperation]):
    """
    Decorator to register an operation class.

    Usage:
        @register_operation
        class MyOperation(BaseOperation):
            ...
    """
    def wrapper(cls):
        registry.register(cls())
        return cls
    return wrapper
```

### 4.4 Library Adapters

```python
# adapters/base_adapter.py
"""
Abstract base adapter for library-specific implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional
import numpy as np

from core.models import MeshInput


class MeshAdapter(ABC):
    """
    Abstract base for library-specific mesh adapters.

    Adapters provide a unified interface to library-specific
    mesh representations and operations.
    """

    @property
    @abstractmethod
    def library_name(self) -> str:
        """Return the name of the wrapped library."""
        pass

    @abstractmethod
    def load_mesh(self, file_path: str, **kwargs) -> Any:
        """Load mesh from file."""
        pass

    @abstractmethod
    def create_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> Any:
        """Create mesh from arrays."""
        pass

    @abstractmethod
    def get_vertices(self, mesh: Any) -> np.ndarray:
        """Extract vertex array from mesh."""
        pass

    @abstractmethod
    def get_faces(self, mesh: Any) -> np.ndarray:
        """Extract face array from mesh."""
        pass

    @abstractmethod
    def is_watertight(self, mesh: Any) -> bool:
        """Check if mesh is watertight (closed manifold)."""
        pass

    @abstractmethod
    def is_manifold(self, mesh: Any) -> Tuple[bool, List[Any]]:
        """Check if mesh is manifold, return issues if any."""
        pass

    @abstractmethod
    def get_boundary_edges(self, mesh: Any) -> np.ndarray:
        """Get edges that are on the mesh boundary."""
        pass

    @abstractmethod
    def get_nonmanifold_edges(self, mesh: Any) -> List[Tuple[int, int, int]]:
        """Get non-manifold edges as (v1, v2, face_count) tuples."""
        pass

    @abstractmethod
    def get_duplicate_vertices(self, mesh: Any, tolerance: float) -> List[int]:
        """Get indices of duplicate vertices."""
        pass

    @abstractmethod
    def compute_face_areas(self, mesh: Any) -> np.ndarray:
        """Compute area of each face."""
        pass

    @abstractmethod
    def compute_face_normals(self, mesh: Any) -> np.ndarray:
        """Compute normal vector for each face."""
        pass

    @abstractmethod
    def detect_self_intersections(self, mesh: Any) -> List[Tuple[int, int]]:
        """Detect self-intersecting face pairs."""
        pass

    @abstractmethod
    def compute_cell_quality(
        self,
        mesh: Any,
        quality_type: str
    ) -> np.ndarray:
        """
        Compute cell quality metric.

        Args:
            mesh: Input mesh
            quality_type: One of 'aspect_ratio', 'skewness', 'scaled_jacobian',
                         'max_angle', 'min_angle', 'condition', 'Jacobian'
        """
        pass

    def to_canonical(self, mesh: Any) -> MeshInput:
        """Convert library mesh to canonical format."""
        return MeshInput(
            vertices=self.get_vertices(mesh),
            faces=self.get_faces(mesh),
            source_format=self.library_name
        )

    def from_canonical(self, mesh_input: MeshInput) -> Any:
        """Create library mesh from canonical format."""
        return self.create_mesh(mesh_input.vertices, mesh_input.faces)
```

```python
# adapters/trimesh_adapter.py
"""
TRimesh library adapter implementation.
"""

from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import trimesh
from trimesh import Trimesh

from adapters.base_adapter import MeshAdapter
from core.models import MeshInput


class TrimeshAdapter(MeshAdapter):
    """Adapter for the TRimesh library."""

    @property
    def library_name(self) -> str:
        return 'trimesh'

    def load_mesh(self, file_path: str, **kwargs) -> Trimesh:
        """Load mesh from file using TRimesh."""
        return trimesh.load_mesh(file_path, **kwargs)

    def create_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> Trimesh:
        """Create TRimesh from arrays."""
        return Trimesh(vertices=vertices, faces=faces)

    def get_vertices(self, mesh: Trimesh) -> np.ndarray:
        """Extract vertex array."""
        return np.array(mesh.vertices)

    def get_faces(self, mesh: Trimesh) -> np.ndarray:
        """Extract face array."""
        return np.array(mesh.faces)

    def is_watertight(self, mesh: Trimesh) -> bool:
        """Check if mesh is watertight."""
        return mesh.is_watertight

    def is_manifold(self, mesh: Trimesh) -> Tuple[bool, List[Any]]:
        """Check if mesh is manifold."""
        issues = []

        # Check for degenerate faces
        degenerate = mesh.degenerate_faces
        if len(degenerate) > 0:
            issues.append({'type': 'degenerate_faces', 'indices': degenerate})

        # Check for duplicate vertices
        unique_edges = mesh.edges_unique
        edges_sorted = np.sort(unique_edges, axis=1)
        # Non-manifold detection logic
        # ...

        is_manifold = len(issues) == 0
        return is_manifold, issues

    def get_boundary_edges(self, mesh: Trimesh) -> np.ndarray:
        """Get boundary (free) edges."""
        return mesh.edges_boundary

    def get_nonmanifold_edges(self, mesh: Trimesh) -> List[Tuple[int, int, int]]:
        """Detect non-manifold edges."""
        edges_face_map = mesh.edges_face_adjacency_graph()
        nonmanifold = []

        for edge, faces in edges_face_map.items():
            if len(faces) != 2:
                nonmanifold.append((edge[0], edge[1], len(faces)))

        return nonmanifold

    def get_duplicate_vertices(
        self,
        mesh: Trimesh,
        tolerance: float
    ) -> List[int]:
        """Find duplicate vertices within tolerance."""
        # Use TRimesh's merge vertices functionality
        original_count = len(mesh.vertices)
        merged = mesh.merge_vertices(tolerance)
        duplicate_count = original_count - len(merged.vertices)

        # Return indices of vertices that would be duplicates
        # This requires tracking which vertices were merged
        return list(range(duplicate_count))

    def compute_face_areas(self, mesh: Trimesh) -> np.ndarray:
        """Compute area of each face."""
        return mesh.area_faces

    def compute_face_normals(self, mesh: Trimesh) -> np.ndarray:
        """Compute face normals."""
        return mesh.face_normals

    def detect_self_intersections(self, mesh: Trimesh) -> List[Tuple[int, int]]:
        """Detect self-intersections."""
        if mesh.is_watertight:
            return []  # Watertight meshes can't self-intersect

        intersecting = mesh.split().is_watertight
        # For detailed intersection detection, use mesh.check_self_intersection()
        return []

    def compute_cell_quality(
        self,
        mesh: Trimesh,
        quality_type: str
    ) -> np.ndarray:
        """Compute cell quality metrics."""
        # TRimesh has limited built-in quality metrics
        # For comprehensive metrics, use PyVista adapter
        quality_functions = {
            'area': lambda: mesh.area_faces,
            'aspect_ratio': self._compute_aspect_ratio_trimesh,
        }

        if quality_type in quality_functions:
            return quality_functions[quality_type]()
        else:
            raise ValueError(f"Unsupported quality type: {quality_type}")

    def _compute_aspect_ratio_trimesh(self, mesh: Trimesh) -> np.ndarray:
        """Compute aspect ratio using TRimesh primitives."""
        aspect_ratios = []
        for face in mesh.faces:
            vertices = mesh.vertices[face]
            # Compute circumradius and inradius ratio
            # Simplified implementation
            edges = [
                np.linalg.norm(vertices[1] - vertices[0]),
                np.linalg.norm(vertices[2] - vertices[1]),
                np.linalg.norm(vertices[0] - vertices[2])
            ]
            area = mesh.area_faces[list(mesh.faces).index(face)]
            circumradius = max(edges) / 2
            if area > 0:
                inradius = 2 * area / sum(edges)
                aspect_ratios.append(circumradius / inradius if inradius > 0 else 0)
            else:
                aspect_ratios.append(0)
        return np.array(aspect_ratios)
```

```python
# adapters/pyvista_adapter.py
"""
PyVista library adapter implementation.
"""

from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pyvista as pv
from pyvista import PolyData

from adapters.base_adapter import MeshAdapter
from core.models import MeshInput


class PyVistaAdapter(MeshAdapter):
    """Adapter for the PyVista library (VTK wrapper)."""

    @property
    def library_name(self) -> str:
        return 'pyvista'

    def load_mesh(self, file_path: str, **kwargs) -> PolyData:
        """Load mesh from file using PyVista."""
        return pv.read(file_path)

    def create_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> PolyData:
        """Create PolyData from arrays."""
        n_faces = len(faces)
        n_vertices_per_face = 3  # Assuming triangles

        # Create face array with vertex count prefix
        face_arr = np.column_stack([
            np.full(n_faces, n_vertices_per_face, dtype=np.int32),
            faces
        ]).ravel()

        return pv.PolyData(vertices, face_arr)

    def get_vertices(self, mesh: PolyData) -> np.ndarray:
        """Extract vertex array."""
        return np.array(mesh.points)

    def get_faces(self, mesh: PolyData) -> np.ndarray:
        """Extract face array (removing vertex count prefix)."""
        faces = mesh.faces
        if len(faces) == 0:
            return np.array([])

        # Faces are stored as [n0, v0, v1, v2, n1, v0, v1, v2, ...]
        n_faces = mesh.n_cells
        face_indices = []

        idx = 0
        for _ in range(n_faces):
            n_verts = faces[idx]
            idx += 1
            face_indices.append(faces[idx:idx + n_verts])
            idx += n_verts

        return np.array(face_indices)

    def is_watertight(self, mesh: PolyData) -> bool:
        """Check if mesh is watertight."""
        return mesh.is_all_triangles() and mesh.n_open_edges == 0

    def is_manifold(self, mesh: PolyData) -> Tuple[bool, List[Any]]:
        """Check if mesh is manifold."""
        issues = []

        # Use VTK's manifold check
        if not mesh.is_manifold:
            issues.append({'type': 'non_manifold', 'message': 'Mesh contains non-manifold geometry'})

        # Check for degenerate cells
        quality = mesh.compute_cell_quality(qualitymeasure='volume')
        volumes = quality.cell_data.get('Volume', [])
        degenerate = np.where(volumes == 0)[0]
        if len(degenerate) > 0:
            issues.append({'type': 'degenerate_cells', 'indices': degenerate.tolist()})

        is_manifold = len(issues) == 0
        return is_manifold, issues

    def get_boundary_edges(self, mesh: PolyData) -> np.ndarray:
        """Get boundary edges."""
        mesh = mesh.extract_all_edges()
        boundary = mesh.extract_feature_edges(boundary_edges=True, feature_edges=False)
        return boundary.lines

    def get_nonmanifold_edges(self, mesh: PolyData) -> List[Tuple[int, int, int]]:
        """Detect non-manifold edges."""
        # Build edge-to-face adjacency
        edges = {}
        faces = self.get_faces(mesh)

        for face_idx, face in enumerate(faces):
            for i in range(3):
                v1, v2 = face[i], face[(i + 1) % 3]
                edge = tuple(sorted([v1, v2]))
                if edge not in edges:
                    edges[edge] = []
                edges[edge].append(face_idx)

        nonmanifold = [
            (edge[0], edge[1], len(faces))
            for edge, faces in edges.items()
            if len(faces) != 2
        ]
        return nonmanifold

    def get_duplicate_vertices(
        self,
        mesh: PolyData,
        tolerance: float
    ) -> List[int]:
        """Find duplicate vertices."""
        # Use VTK's duplicate point removal
        cleaned = mesh.clean(tolerance=tolerance, merge_tol=tolerance)
        original_count = mesh.n_points
        cleaned_count = cleaned.n_points

        # Return indices that were removed
        return list(range(cleaned_count, original_count))

    def compute_face_areas(self, mesh: PolyData) -> np.ndarray:
        """Compute area of each face."""
        quality = mesh.compute_cell_quality(qualitymeasure='area')
        return quality.cell_data.get('Area', np.array([]))

    def compute_face_normals(self, mesh: PolyData) -> np.ndarray:
        """Compute face normals."""
        mesh = mesh.compute_normals(cell_oriented=True)
        return mesh.cell_data.get('Normals', np.array([]))

    def detect_self_intersections(self, mesh: PolyData) -> List[Tuple[int, int]]:
        """Detect self-intersections using VTK."""
        # Use VTK's self-intersection detection
        intersections = []
        # Implementation depends on VTK version
        return intersections

    def compute_cell_quality(
        self,
        mesh: PolyData,
        quality_type: str
    ) -> np.ndarray:
        """
        Compute cell quality metrics using VTK's cell quality filter.

        Supported quality types:
        - 'area': Cell area
        - 'aspect_ratio': Aspect ratio
        - 'skewness': Maximum angle skewness
        - 'scaled_jacobian': Scaled Jacobian determinant
        - 'max_angle': Maximum angle
        - 'min_angle': Minimum angle
        - 'condition': Condition number
        - 'Jacobian': Jacobian determinant
        - 'volume': Cell volume
        - 'aspect_gamma': Aspect gamma
        - 'med_aspect_frobenius': Medial aspect Frobenius
        - 'normalized_inradius': Normalized inradius
        """
        quality_map = {
            'area': 'area',
            'aspect_ratio': 'aspect_ratio',
            'skewness': 'skew',
            'max_angle': 'max_angle',
            'min_angle': 'min_angle',
            'condition': 'condition',
            'Jacobian': 'jacobian',
            'volume': 'volume',
            'scaled_jacobian': 'scaled_jacobian',
        }

        vtk_quality = quality_map.get(quality_type, quality_type)

        try:
            quality_mesh = mesh.compute_cell_quality(qualitymeasure=vtk_quality)
            return quality_mesh.cell_data.get(f'CellQuality{vtk_quality.capitalize()}', np.array([]))
        except Exception:
            return np.array([])
```

```python
# adapters/pymeshlab_adapter.py
"""
PyMeshLab library adapter implementation.
"""

from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pymeshlab as ml

from adapters.base_adapter import MeshAdapter
from core.models import MeshInput


class PyMeshLabAdapter(MeshAdapter):
    """Adapter for the PyMeshLab library."""

    def __init__(self):
        self._meshset: Optional[ml.MeshSet] = None
        self._current_mesh_index: int = 0

    @property
    def library_name(self) -> str:
        return 'pymeshlab'

    def load_mesh(self, file_path: str, **kwargs) -> ml.MeshSet:
        """Load mesh from file using PyMeshLab."""
        ms = ml.MeshSet()
        ms.load_new_mesh(file_path)
        self._meshset = ms
        self._current_mesh_index = 0
        return ms

    def create_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> ml.MeshSet:
        """Create mesh from arrays."""
        ms = ml.MeshSet()
        mesh = ml.Mesh(
            vertex_matrix=vertices.tolist(),
            face_matrix=faces.tolist()
        )
        ms.add_mesh(mesh)
        self._meshset = ms
        self._current_mesh_index = 0
        return ms

    def get_vertices(self, meshset: ml.MeshSet) -> np.ndarray:
        """Extract vertex array."""
        mesh = meshset.current_mesh()
        return np.array(mesh.vertex_matrix())

    def get_faces(self, meshset: ml.MeshSet) -> np.ndarray:
        """Extract face array."""
        mesh = meshset.current_mesh()
        return np.array(mesh.face_matrix())

    def is_watertight(self, meshset: ml.MeshSet) -> bool:
        """Check if mesh is watertight."""
        mesh = meshset.current_mesh()
        return mesh.is_watertight()

    def is_manifold(self, meshset: ml.MeshSet) -> Tuple[bool, List[Any]]:
        """Check if mesh is manifold using PyMeshLab filters."""
        issues = []

        # Select non-manifold edges
        meshset.select_non_manifold_edges()
        if meshset.selected_vertex_number() > 0:
            issues.append({'type': 'non_manifold_edges', 'count': meshset.selected_vertex_number()})

        meshset.mesh_unselect_all()

        # Select degenerate faces
        meshset.select_degenerate_faces()
        if meshset.selected_face_number() > 0:
            issues.append({'type': 'degenerate_faces', 'count': meshset.selected_face_number()})

        meshset.mesh_unselect_all()

        is_manifold = len(issues) == 0
        return is_manifold, issues

    def get_boundary_edges(self, meshset: ml.MeshSet) -> np.ndarray:
        """Get boundary (non-manifold in PyMeshLab terms) edges."""
        meshset.select_non_manifold_edges()
        # Extract selected edge indices
        selected = meshset.selected_vertices()
        meshset.mesh_unselect_all()
        return np.array(selected)

    def get_nonmanifold_edges(self, meshset: ml.MeshSet) -> List[Tuple[int, int, int]]:
        """Detect non-manifold edges."""
        meshset.select_non_manifold_edges()
        nonmanifold_verts = meshset.selected_vertices()
        meshset.mesh_unselect_all()

        # Convert vertices to edge representation
        # This requires additional processing
        return [(v, -1, -1) for v in nonmanifold_verts]

    def get_duplicate_vertices(
        self,
        meshset: ml.MeshSet,
        tolerance: float
    ) -> List[int]:
        """Find duplicate vertices."""
        original_count = meshset.current_mesh().vertex_number()
        meshset.merge_close_vertices(threshold=tolerance)
        cleaned_count = meshset.current_mesh().vertex_number()

        return list(range(cleaned_count, original_count))

    def compute_face_areas(self, meshset: ml.MeshSet) -> np.ndarray:
        """Compute area of each face."""
        meshset.compute_selection_of_faces_by_curvature_per_face()
        mesh = meshset.current_mesh()
        # Extract curvature data as proxy for area
        return np.array([])  # Placeholder

    def compute_face_normals(self, meshset: ml.MeshSet) -> np.ndarray:
        """Compute face normals."""
        meshset.compute_normal_for_faces(flip=False, unseam=False)
        mesh = meshset.current_mesh()
        return np.array(meshface_normal_list())

    def detect_self_intersections(self, meshset: ml.MeshSet) -> List[Tuple[int, int]]:
        """Detect self-intersections using PyMeshLab filters."""
        meshset.select_self_intersecting_faces()
        intersecting = meshset.selected_face_indices()
        meshset.mesh_unselect_all()
        return [(i, -1) for i in intersecting]

    def compute_cell_quality(
        self,
        meshset: ml.MeshSet,
        quality_type: str
    ) -> np.ndarray:
        """Compute cell quality using PyMeshLab."""
        # PyMeshLab has various quality filters
        quality_filters = {
            'area': lambda: self._compute_face_area_pml(meshset),
        }

        if quality_type in quality_filters:
            return quality_filters[quality_type]()
        else:
            return np.array([])

    def _compute_face_area_pml(self, meshset: ml.MeshSet) -> np.ndarray:
        """Compute face areas using PyMeshLab."""
        meshset.compute_geometric_measures()
        mesh = meshset.current_mesh()
        # Access face area data
        return np.array(mesh.face_area_list())
```

### 4.5 Concrete Operation Implementations

```python
# operations/topological/nonmanifold_edges.py
"""
Non-manifold edge detection operation.
"""

from typing import List, Optional
import numpy as np

from core.models import (
    MeshInput, MeshIssue, IssueLocation,
    IssueCategory, IssueSeverity, OperationConfig
)
from operations.base import BaseOperation, OperationMetadata, register_operation
from adapters.trimesh_adapter import TrimeshAdapter
from adapters.pyvista_adapter import PyVistaAdapter
from adapters.pymeshlab_adapter import PyMeshLabAdapter


class NonManifoldEdgeDetection(BaseOperation):
    """
    Detects edges shared by more than two faces.

    Violates the 2-manifold requirement that each edge must be
    incident to exactly two faces.
    """

    @property
    def metadata(self) -> OperationMetadata:
        return OperationMetadata(
            operation_id='topology.detect_nonmanifold_edges',
            display_name='Non-Manifold Edge Detection',
            description='Identifies edges shared by more than two faces, '
                       'violating 2-manifold property.',
            category=IssueCategory.TOPOLOGICAL,
            supported_libraries=['trimesh', 'pyvista', 'pymeshlab'],
            recommended_library='pyvista',
            requires_adjacency=True,
            complexity='O(n)'
        )

    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        """Execute non-manifold edge detection."""
        # Use PyVista adapter for best results
        adapter = PyVistaAdapter()
        native_mesh = adapter.from_canonical(mesh)

        nonmanifold_edges = adapter.get_nonmanifold_edges(native_mesh)

        if not nonmanifold_edges:
            return []

        locations = []
        for v1, v2, face_count in nonmanifold_edges:
            coords = mesh.vertices[v1] if v1 < len(mesh.vertices) else None
            locations.append(IssueLocation(
                element_type='edge',
                element_index=-1,  # Edge index not tracked
                coordinates=tuple(coords) if coords is not None else None,
                adjacent_elements={
                    'vertices': [v1, v2],
                    'face_count': face_count
                }
            ))

        # Determine severity based on extent
        severity = IssueSeverity.ERROR
        if len(nonmanifold_edges) > len(mesh.faces) * 0.01:
            severity = IssueSeverity.CRITICAL

        issue = MeshIssue(
            operation_id=self.metadata.operation_id,
            category=IssueCategory.TOPOLOGICAL,
            severity=severity,
            message=f'Detected {len(nonmanifold_edges)} non-manifold edges. '
                   f'These edges are shared by {len(set(nonmanifold_edges))} unique edges.',
            locations=locations,
            metrics={
                'count': len(nonmanifold_edges),
                'percentage': len(nonmanifold_edges) / max(len(mesh.faces), 1) * 100
            }
        )

        return [issue]


# Register the operation
registry.register(NonManifoldEdgeDetection())
```

```python
# operations/quality/cell_quality.py
"""
Cell quality computation operation.
"""

from typing import List, Dict, Optional
import numpy as np

from core.models import (
    MeshInput, MeshIssue, QualityMetrics,
    IssueCategory, IssueSeverity, OperationConfig
)
from operations.base import BaseOperation, OperationMetadata, register_operation
from adapters.pyvista_adapter import PyVistaAdapter


class CellQualityComputation(BaseOperation):
    """
    Computes various cell quality metrics using PyVista/VTK.

    Supports: aspect_ratio, skewness, scaled_jacobian, max_angle,
    min_angle, condition, Jacobian, volume.
    """

    def __init__(
        self,
        config: Optional[OperationConfig] = None,
        quality_types: Optional[List[str]] = None
    ):
        super().__init__(config)
        self.quality_types = quality_types or ['aspect_ratio', 'skewness', 'min_angle']

    @property
    def metadata(self) -> OperationMetadata:
        return OperationMetadata(
            operation_id='quality.compute_cell_quality',
            display_name='Cell Quality Computation',
            description='Computes element shape quality metrics including '
                       'aspect ratio, skewness, and angle measures.',
            category=IssueCategory.QUALITY,
            supported_libraries=['pyvista'],
            recommended_library='pyvista',
            requires_adjacency=False,
            complexity='O(n)'
        )

    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        """Execute cell quality computation."""
        adapter = PyVistaAdapter()
        native_mesh = adapter.from_canonical(mesh)

        issues = []

        for quality_type in self.quality_types:
            try:
                values = adapter.compute_cell_quality(native_mesh, quality_type)

                if len(values) == 0:
                    continue

                # Compute thresholds based on quality type
                threshold = self._get_threshold(quality_type)

                metrics = QualityMetrics(
                    operation_id=f'{self.metadata.operation_id}.{quality_type}',
                    metric_name=quality_type,
                    values=values,
                    threshold=threshold
                )

                # Count violations
                violations = np.sum(values < threshold)

                if violations > 0:
                    severity = self._determine_severity(violations, len(values))

                    issue = MeshIssue(
                        operation_id=f'{self.metadata.operation_id}.{quality_type}',
                        category=IssueCategory.QUALITY,
                        severity=severity,
                        message=f'{quality_type}: {violations}/{len(values)} '
                               f'cells ({violations/len(values)*100:.1f}%) '
                               f'below threshold {threshold}',
                        metrics={
                            'violation_count': int(violations),
                            'total_cells': len(values),
                            'violation_rate': float(violations / len(values)),
                            'mean': float(np.mean(values)),
                            'min': float(np.min(values)),
                            'max': float(np.max(values)),
                            'threshold': threshold
                        }
                    )
                    issues.append(issue)

            except Exception as e:
                # Log error but continue with other metrics
                continue

        return issues

    def _get_threshold(self, quality_type: str) -> float:
        """Get quality threshold for metric type."""
        thresholds = {
            'aspect_ratio': 0.3,
            'skewness': 0.5,
            'min_angle': 20.0,
            'max_angle': 140.0,
            'scaled_jacobian': 0.1,
            'condition': 10.0,
            'Jacobian': 0.0,
            'volume': 1e-10
        }
        return thresholds.get(quality_type, 0.1)

    def _determine_severity(self, violations: int, total: int) -> IssueSeverity:
        """Determine issue severity based on violation rate."""
        rate = violations / total
        if rate > 0.5:
            return IssueSeverity.CRITICAL
        elif rate > 0.2:
            return IssueSeverity.ERROR
        elif rate > 0.05:
            return IssueSeverity.WARNING
        else:
            return IssueSeverity.INFO


# Register the operation
registry.register(CellQualityComputation())
```

### 4.6 MCP Server Implementation

```python
# mcp/server.py
"""
MCP server implementation for mesh issue examination.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path

from core.models import MeshInput, InspectionResult, OperationConfig
from operations.operation_registry import registry
from adapters.trimesh_adapter import TrimeshAdapter
from utils.mesh_loader import load_mesh_from_file


@dataclass
class MCPMessage:
    """Represents an MCP protocol message."""
    method: str
    params: Dict[str, Any]
    id: Optional[str] = None


@dataclass
class MCPTool:
    """Represents an MCP tool definition."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable


class MeshIssueExaminerServer:
    """
    MCP server for mesh issue examination.

    Provides tools for:
    - Loading meshes from files
    - Running individual or composed inspection operations
    - Retrieving inspection results
    """

    def __init__(self, config: Optional[OperationConfig] = None):
        self.config = config or OperationConfig()
        self._mesh_cache: Dict[str, MeshInput] = {}
        self._result_cache: Dict[str, InspectionResult] = {}
        self._tools: Dict[str, MCPTool] = {}
        self._initialize_tools()

    def _initialize_tools(self):
        """Initialize MCP tools from registered operations."""
        for operation in registry.list_all():
            tool = MCPTool(
                name=f'mesh_{operation.operation_id}',
                description=operation.description,
                input_schema=self._generate_input_schema(operation),
                handler=self._create_handler(operation.operation_id)
            )
            self._tools[tool.name] = tool

        # Add utility tools
        self._tools['mesh_load'] = MCPTool(
            name='mesh_load',
            description='Load a mesh file and return mesh ID for subsequent operations',
            input_schema={
                'type': 'object',
                'properties': {
                    'file_path': {'type': 'string', 'description': 'Path to mesh file'},
                    'mesh_id': {'type': 'string', 'description': 'Optional custom mesh ID'}
                },
                'required': ['file_path']
            },
            handler=self._handle_load
        )

        self._tools['mesh_inspect_all'] = MCPTool(
            name='mesh_inspect_all',
            description='Run all mesh inspection operations',
            input_schema={
                'type': 'object',
                'properties': {
                    'mesh_id': {'type': 'string', 'description': 'Mesh ID from load operation'}
                },
                'required': ['mesh_id']
            },
            handler=self._handle_inspect_all
        )

    def _generate_input_schema(self, operation) -> Dict[str, Any]:
        """Generate JSON schema for operation input."""
        return {
            'type': 'object',
            'properties': {
                'mesh_id': {'type': 'string', 'description': 'Mesh ID from load operation'},
                'config': {
                    'type': 'object',
                    'properties': {
                        'tolerance': {'type': 'number'},
                        'quality_threshold': {'type': 'number'}
                    }
                }
            },
            'required': ['mesh_id']
        }

    def _create_handler(self, operation_id: str) -> Callable:
        """Create handler for specific operation."""
        async def handler(params: Dict[str, Any]) -> Dict[str, Any]:
            mesh_id = params.get('mesh_id')
            mesh = self._mesh_cache.get(mesh_id)

            if mesh is None:
                return {'error': f'Mesh not found: {mesh_id}'}

            operation = registry.get(operation_id)
            if operation is None:
                return {'error': f'Operation not found: {operation_id}'}

            issues = operation.run(mesh)

            return {
                'operation_id': operation_id,
                'mesh_id': mesh_id,
                'issue_count': len(issues),
                'issues': [issue.to_dict() for issue in issues]
            }

        return handler

    async def _handle_load(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle mesh load request."""
        file_path = params['file_path']
        mesh_id = params.get('mesh_id', f'mesh_{len(self._mesh_cache)}')

        try:
            mesh_input = load_mesh_from_file(file_path, mesh_id=mesh_id)
            mesh_input.file_path = file_path
            self._mesh_cache[mesh_id] = mesh_input

            return {
                'mesh_id': mesh_id,
                'n_vertices': mesh_input.n_vertices,
                'n_faces': mesh_input.n_faces,
                'bounding_box': mesh_input.bounding_box
            }
        except Exception as e:
            return {'error': str(e)}

    async def _handle_inspect_all(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle full inspection request."""
        mesh_id = params['mesh_id']
        mesh = self._mesh_cache.get(mesh_id)

        if mesh is None:
            return {'error': f'Mesh not found: {mesh_id}'}

        # Create workflow for all operations
        all_operations = [op.metadata.operation_id for op in registry.list_all()]
        workflow = registry.create_workflow(all_operations)

        result = workflow(mesh)
        self._result_cache[mesh_id] = result

        return asdict(result)

    async def handle_request(self, message: MCPMessage) -> Dict[str, Any]:
        """Handle incoming MCP request."""
        method = message.method

        if method == 'tools/list':
            return {
                'tools': [
                    {
                        'name': t.name,
                        'description': t.description,
                        'input_schema': t.input_schema
                    }
                    for t in self._tools.values()
                ]
            }

        elif method == 'tools/call':
            tool_name = message.params.get('name')
            tool_params = message.params.get('params', {})

            tool = self._tools.get(tool_name)
            if tool is None:
                return {'error': f'Unknown tool: {tool_name}'}

            return await tool.handler(tool_params)

        elif method == 'mesh/cache/clear':
            self._mesh_cache.clear()
            self._result_cache.clear()
            return {'status': 'cleared'}

        else:
            return {'error': f'Unknown method: {method}'}


async def main():
    """Main entry point for MCP server."""
    server = MeshIssueExaminerServer()

    # Example usage
    print("Mesh Issue Examiner MCP Server")
    print(f"Available tools: {list(server._tools.keys())}")

    # Example: Load mesh and inspect
    # result = await server.handle_request(MCPMessage(
    #     method='tools/call',
    #     params={'name': 'mesh_load', 'params': {'file_path': 'example.stl'}}
    # ))


if __name__ == '__main__':
    asyncio.run(main())
```

### 4.7 Utility Functions

```python
# utils/mesh_loader.py
"""
Mesh loading utilities with format support.
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np

from core.models import MeshInput


def load_mesh_from_file(
    file_path: Union[str, Path],
    mesh_id: Optional[str] = None,
    library: str = 'trimesh'
) -> MeshInput:
    """
    Load mesh from file into canonical format.

    Args:
        file_path: Path to mesh file
        mesh_id: Optional mesh identifier
        library: Library to use for loading ('trimesh', 'pyvista', 'pymeshlab')

    Returns:
        MeshInput in canonical format

    Supported formats:
        - STL, OBJ, PLY, OFF: Universal support
        - GLTF, GLB: trimesh, pyvista
        - VTK, VTP: pyvista
        - 3MF: trimesh
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {file_path}")

    suffix = file_path.suffix.lower()

    # Map suffixes to library support
    format_support = {
        '.stl': ['trimesh', 'pyvista', 'pymeshlab'],
        '.obj': ['trimesh', 'pyvista', 'pymeshlab'],
        '.ply': ['trimesh', 'pyvista', 'pymeshlab'],
        '.off': ['trimesh', 'pymeshlab'],
        '.gltf': ['trimesh', 'pyvista'],
        '.glb': ['trimesh', 'pyvista'],
        '.vtp': ['pyvista'],
        '.vtk': ['pyvista'],
        '.3mf': ['trimesh'],
    }

    if suffix not in format_support:
        raise ValueError(f"Unsupported format: {suffix}")

    if library not in format_support[suffix]:
        # Fallback to first available
        library = format_support[suffix][0]

    # Load based on library
    if library == 'trimesh':
        from adapters.trimesh_adapter import TrimeshAdapter
        adapter = TrimeshAdapter()
        native_mesh = adapter.load_mesh(str(file_path))
        vertices = adapter.get_vertices(native_mesh)
        faces = adapter.get_faces(native_mesh)

    elif library == 'pyvista':
        from adapters.pyvista_adapter import PyVistaAdapter
        adapter = PyVistaAdapter()
        native_mesh = adapter.load_mesh(str(file_path))
        vertices = adapter.get_vertices(native_mesh)
        faces = adapter.get_faces(native_mesh)

    elif library == 'pymeshlab':
        from adapters.pymeshlab_adapter import PyMeshLabAdapter
        adapter = PyMeshLabAdapter()
        meshset = adapter.load_mesh(str(file_path))
        vertices = adapter.get_vertices(meshset)
        faces = adapter.get_faces(meshset)

    mesh_id = mesh_id or file_path.stem

    return MeshInput(
        vertices=vertices,
        faces=faces,
        mesh_id=mesh_id,
        source_format=suffix[1:]  # Remove leading dot
    )


def create_mesh_from_arrays(
    vertices: np.ndarray,
    faces: np.ndarray,
    mesh_id: str = ""
) -> MeshInput:
    """
    Create MeshInput from vertex and face arrays.

    Args:
        vertices: Nx3 array of vertex coordinates
        faces: Mx3 array of face indices
        mesh_id: Optional mesh identifier

    Returns:
        MeshInput in canonical format
    """
    return MeshInput(
        vertices=vertices,
        faces=faces,
        mesh_id=mesh_id or f"mesh_{id(vertices)}"
    )
```

## 5. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

| Task | Description | Deliverable |
|------|-------------|-------------|
| Project setup | Initialize repository structure, dependencies | Project scaffold |
| Data models | Implement core models (MeshInput, Issue, Metrics) | models.py |
| Adapter base | Create adapter interface and base implementation | base_adapter.py |
| Registry foundation | Implement operation registry | operation_registry.py |

### Phase 2: Core Operations (Week 3-4)

| Task | Description | Deliverable |
|------|-------------|-------------|
| TRimesh adapter | Full adapter implementation | trimesh_adapter.py |
| PyVista adapter | Full adapter implementation | pyvista_adapter.py |
| Topological ops | Non-manifold, free edge, duplicate detection | topological/*.py |
| Quality ops | Cell quality metrics using PyVista | quality/*.py |

### Phase 3: MCP Integration (Week 5-6)

| Task | Description | Deliverable |
|------|-------------|-------------|
| MCP server | Server implementation with tool registration | server.py |
| Request handlers | Handle MCP protocol messages | handlers.py |
| Result aggregation | Consolidate results from operations | result aggregation |

### Phase 4: Enhancement (Week 7-8)

| Task | Description | Deliverable |
|------|-------------|-------------|
| PyMeshLab adapter | Advanced filtering capabilities | pymeshlab_adapter.py |
| Composite workflows | Operation composition utilities | workflow utilities |
| Performance optimization | Caching, parallel execution | optimization |

## 6. Testing Strategy

### Unit Tests

```python
# tests/test_operations.py
"""
Unit tests for mesh inspection operations.
"""

import pytest
import numpy as np
from core.models import MeshInput, create_mesh_from_arrays
from operations.topological.nonmanifold_edges import NonManifoldEdgeDetection


class TestNonManifoldEdgeDetection:
    """Tests for non-manifold edge detection."""

    def test_watertight_mesh_no_issues(self):
        """Watertight mesh should have no non-manifold edges."""
        # Create simple tetrahedron
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        faces = np.array([
            [0, 1, 2],
            [0, 3, 1],
            [1, 3, 2],
            [2, 3, 0]
        ])

        mesh = create_mesh_from_arrays(vertices, faces)
        operation = NonManifoldEdgeDetection()
        issues = operation.execute(mesh)

        assert len(issues) == 0

    def test_non_manifold_mesh_detected(self):
        """Non-manifold mesh should be detected."""
        # Create mesh with non-manifold edge
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1]
        ])
        faces = np.array([
            [0, 1, 2],
            [0, 3, 1],
            [1, 3, 4],
            [1, 4, 2],
            [2, 4, 0]
        ])

        mesh = create_mesh_from_arrays(vertices, faces)
        operation = NonManifoldEdgeDetection()
        issues = operation.execute(mesh)

        assert len(issues) > 0
        assert issues[0].category.value == 'topological'
```

## 7. Configuration and Deployment

### requirements.txt

```
# Core dependencies
numpy>=1.21.0
trimesh>=4.0.0
pyvista>=0.40.0
vtk>=9.2.0

# Optional
pymeshlab>=2022.2

# MCP server
pydantic>=2.0.0

# Testing
pytest>=7.0.0
pytest-asyncio>=0.21.0
```

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Start MCP server
python -m mcp.server
```

## 8. Conclusion

This implementation plan provides a comprehensive framework for building a mesh issue examiner MCP server. The three selected libraries—PyVista, TRimesh, and PyMeshLab—provide complementary capabilities that cover the full spectrum of mesh inspection operations while maintaining flexibility for future extension.

The framework architecture ensures:
- **Composability**: Atomic operations can be combined into complex workflows
- **Extensibility**: New operations and library adapters can be added without modifying existing code
- **Interoperability**: Library-specific implementations are abstracted behind common interfaces
- **Performance**: The most suitable library is automatically selected for each operation

This foundation enables rapid development of MCP-powered mesh inspection capabilities suitable for integration into larger computational workflows, CAD systems, and automated quality assurance pipelines.
