"""
Canonical data models for mesh issue examination.

This module defines the core data structures used throughout the framework
for representing meshes, issues, and quality metrics in a canonical format.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple, Union
from pathlib import Path
import numpy as np


class IssueSeverity(Enum):
    """Severity classification for mesh issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    def __lt__(self, other):
        """Compare severities for sorting."""
        order = [self.INFO, self.WARNING, self.ERROR, self.CRITICAL]
        return order.index(self) < order.index(other)


class IssueCategory(Enum):
    """Categories of mesh issues."""
    TOPOLOGICAL = "topological"
    GEOMETRIC = "geometric"
    QUALITY = "quality"
    ORIENTATION = "orientation"


class MeshDimensionality(Enum):
    """Classification of mesh spatial dimensionality."""
    PURE_2D = "pure_2d"
    SURFACE_2D = "surface_2d"
    VOLUMETRIC_3D = "volumetric_3d"
    MIXED_2D_3D = "mixed_2d_3d"


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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'element_type': self.element_type,
            'element_index': self.element_index,
            'coordinates': self.coordinates,
            'adjacent_elements': self.adjacent_elements
        }


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
            'locations': [loc.to_dict() for loc in self.locations[:10]],  # Limit for serialization
            'metrics': self.metrics,
            'metadata': self.metadata
        }

    @property
    def is_critical(self) -> bool:
        """Check if issue is critical."""
        return self.severity == IssueSeverity.CRITICAL


@dataclass
class QualityMetrics:
    """Container for mesh quality metrics."""
    operation_id: str
    metric_name: str
    values: Optional[Union[np.ndarray, List[float]]]
    statistics: Dict[str, float] = field(default_factory=dict)
    threshold: Optional[float] = None
    violation_count: int = 0

    def __post_init__(self):
        """Compute statistics from values array."""
        if self.values is None:
            self.values = np.array([])
        elif isinstance(self.values, list):
            self.values = np.array(self.values)

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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'operation_id': self.operation_id,
            'metric_name': self.metric_name,
            'statistics': self.statistics,
            'threshold': self.threshold,
            'violation_count': self.violation_count,
            'total_count': len(self.values)
        }


@dataclass
class InspectionResult:
    """Complete result of mesh inspection."""
    mesh_id: str
    file_path: Optional[str]
    issues: List[MeshIssue] = field(default_factory=list)
    quality_metrics: List[QualityMetrics] = field(default_factory=list)
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
            'has_warnings': severity_counts[IssueSeverity.WARNING.value] > 0,
            'critical_count': severity_counts[IssueSeverity.CRITICAL.value],
            'error_count': severity_counts[IssueSeverity.ERROR.value],
            'warning_count': severity_counts[IssueSeverity.WARNING.value],
            'info_count': severity_counts[IssueSeverity.INFO.value]
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'mesh_id': self.mesh_id,
            'file_path': self.file_path,
            'issue_count': len(self.issues),
            'issues': [issue.to_dict() for issue in self.issues],
            'quality_metrics': [m.to_dict() for m in self.quality_metrics],
            'summary': self.summary,
            'processing_time': self.processing_time
        }

    @property
    def passed(self) -> bool:
        """Check if inspection passed (no critical or error issues)."""
        return not self.summary.get('has_critical', False) and \
               not self.summary.get('has_errors', False)


@dataclass
class OperationConfig:
    """Configuration for mesh operations."""
    tolerance: float = 1e-6
    quality_threshold: float = 0.1
    enable_caching: bool = True
    max_iterations: int = 1000
    parallel_enabled: bool = True
    library_preference: Optional[str] = None  # 'trimesh', 'pyvista', 'pymeshlab'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'tolerance': self.tolerance,
            'quality_threshold': self.quality_threshold,
            'enable_caching': self.enable_caching,
            'max_iterations': self.max_iterations,
            'parallel_enabled': self.parallel_enabled,
            'library_preference': self.library_preference
        }


class MeshInput:
    """
    Canonical mesh representation for MCP operations.

    Wraps library-specific mesh objects for transparent interchange
    between different mesh processing libraries. Supports both surface
    meshes (faces) and volumetric meshes (volume_cells).
    """

    def __init__(
        self,
        vertices: Union[np.ndarray, List[List[float]]],
        faces: Union[np.ndarray, List[List[int]]],
        mesh_id: str = "",
        source_format: str = "",
        file_path: Optional[str] = None,
        volume_cells: Optional[Dict[str, Union[np.ndarray, List[List[int]]]]] = None
    ):
        self.mesh_id = mesh_id or f"mesh_{id(self)}"
        self.file_path = file_path

        # Convert to numpy arrays
        self._vertices = np.array(vertices) if not isinstance(vertices, np.ndarray) else vertices
        self._faces = np.array(faces) if not isinstance(faces, np.ndarray) else faces

        self.source_format = source_format
        self._native_objects: Dict[str, Any] = {}
        self._adjacency_cache: Dict[str, Any] = {}

        # Volume cells: dict mapping element type name -> NxK connectivity array
        self._volume_cells: Dict[str, np.ndarray] = {}
        if volume_cells is not None:
            for etype, arr in volume_cells.items():
                converted = np.array(arr) if not isinstance(arr, np.ndarray) else arr
                self._volume_cells[etype] = converted

    @property
    def vertices(self) -> np.ndarray:
        """Get vertex array (copy to prevent cache invalidation bypass)."""
        return self._vertices.copy()

    @vertices.setter
    def vertices(self, value: np.ndarray):
        """Set vertex array and invalidate cache."""
        self._vertices = np.array(value) if not isinstance(value, np.ndarray) else value
        self.invalidate_cache()

    @property
    def faces(self) -> np.ndarray:
        """Get face array (copy to prevent cache invalidation bypass)."""
        return self._faces.copy()

    @faces.setter
    def faces(self, value: np.ndarray):
        """Set face array and invalidate cache."""
        self._faces = np.array(value) if not isinstance(value, np.ndarray) else value
        self.invalidate_cache()

    @property
    def n_vertices(self) -> int:
        """Number of vertices."""
        return len(self._vertices)

    @property
    def n_faces(self) -> int:
        """Number of faces."""
        return len(self._faces)

    @property
    def volume_cells(self) -> Dict[str, np.ndarray]:
        """Get volume cells dict (element_type -> connectivity array).

        Returns a deep copy to prevent cache invalidation bypass. Modifying
        the returned dict or its arrays will NOT affect the mesh. To update
        volume cells, use the setter: ``mesh.volume_cells = new_dict``.
        """
        return {k: v.copy() for k, v in self._volume_cells.items()}

    @volume_cells.setter
    def volume_cells(self, value: Optional[Dict[str, Union[np.ndarray, List[List[int]]]]]):
        """Set volume cells and invalidate cache."""
        if value is None:
            self._volume_cells = {}
        else:
            self._volume_cells = {
                k: np.array(v) if not isinstance(v, np.ndarray) else v
                for k, v in value.items()
            }
        self.invalidate_cache()

    @property
    def n_volume_cells(self) -> int:
        """Total number of volume cells across all element types."""
        return sum(len(arr) for arr in self._volume_cells.values())

    @property
    def has_volume_cells(self) -> bool:
        """True if mesh contains volumetric cells."""
        return len(self._volume_cells) > 0

    @property
    def cell_types(self) -> List[str]:
        """List of element type names present in the mesh."""
        return list(self._volume_cells.keys())

    @property
    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (min_coords, max_coords) bounding box."""
        return np.min(self._vertices, axis=0), np.max(self._vertices, axis=0)

    @property
    def center(self) -> np.ndarray:
        """Get mesh center (centroid of bounding box)."""
        min_b, max_b = self.bounding_box
        return (min_b + max_b) / 2

    @property
    def extent(self) -> np.ndarray:
        """Get mesh extent (size of bounding box)."""
        min_b, max_b = self.bounding_box
        return max_b - min_b

    def get_native(self, library: str) -> Any:
        """Get library-specific mesh object, constructing if needed.
        
        .. note::
            This method is not thread-safe. If multiple threads call
            ``get_native()`` simultaneously for the same library,
            ``_construct_native()`` may be called multiple times.
            For thread-safe usage, external synchronization is required.
        """
        caching = getattr(self, '_caching_enabled', True)
        if caching:
            if library not in self._native_objects:
                self._construct_native(library)
            return self._native_objects[library]
        # Bypass cache: construct fresh object without storing it
        from ..adapters.factory import get_adapter
        adapter = get_adapter(library)
        return adapter.from_canonical(self)

    def _construct_native(self, library: str):
        """Construct library-specific mesh object via the adapter factory."""
        from ..adapters.factory import get_adapter
        adapter = get_adapter(library)
        self._native_objects[library] = adapter.from_canonical(self)

    def invalidate_cache(self):
        """Clear cached data when mesh is modified."""
        self._native_objects.clear()
        self._adjacency_cache.clear()

    def copy(self) -> 'MeshInput':
        """Create a deep copy of this mesh."""
        vc_copy = {k: v.copy() for k, v in self._volume_cells.items()} if self._volume_cells else None
        return MeshInput(
            vertices=self._vertices.copy(),
            faces=self._faces.copy(),
            mesh_id=f"{self.mesh_id}_copy",
            source_format=self.source_format,
            file_path=self.file_path,
            volume_cells=vc_copy
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            'mesh_id': self.mesh_id,
            'source_format': self.source_format,
            'file_path': self.file_path,
            'n_vertices': self.n_vertices,
            'n_faces': self.n_faces,
            'bounding_box': {
                'min': self.bounding_box[0].tolist(),
                'max': self.bounding_box[1].tolist()
            }
        }
        if self.has_volume_cells:
            result['n_volume_cells'] = self.n_volume_cells
            result['cell_types'] = self.cell_types
        return result
