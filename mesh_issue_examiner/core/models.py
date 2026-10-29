"""
Canonical data models for mesh issue examination.

This module defines the core data structures used throughout the framework
for representing meshes, issues, and quality metrics in a canonical format.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple, Union
from pathlib import Path
import os
import threading
import weakref

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
    # Provenance tags, e.g. {'linearized': True} when quadratic cells
    # were evaluated through their corner-only linear equivalents.
    metadata: Dict[str, Any] = field(default_factory=dict)

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
                # Filter out undefined values (-1 is VTK's default for undefined quality)
                valid_values = self.values[self.values != -1.0]
                if len(valid_values) > 0:
                    # Metrics where higher values are worse (violation if value > threshold)
                    HIGHER_IS_WORSE = {'max_angle', 'condition', 'aspect_ratio'}
                    if self.metric_name in HIGHER_IS_WORSE:
                        self.violation_count = int(np.sum(valid_values > self.threshold))
                    else:
                        self.violation_count = int(np.sum(valid_values < self.threshold))
                else:
                    self.violation_count = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'operation_id': self.operation_id,
            'metric_name': self.metric_name,
            'statistics': self.statistics,
            'threshold': self.threshold,
            'violation_count': self.violation_count,
            'total_count': len(self.values),
            'metadata': self.metadata,
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


@dataclass(frozen=True)
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


# ---------------------------------------------------------------------------
# Process-wide native-object cache tracker
# ---------------------------------------------------------------------------
# Each MeshInput may memoize up to MeshInput._MAX_CACHE_SIZE native
# (library-specific) mesh objects; across many long-lived meshes those
# per-instance caches add up unboundedly. This module-level LRU tracker
# bounds the AGGREGATE number of cached native objects (audit item 2.4).
#
# Cleanup is weakref-based: the tracker maps ``id(mesh)`` to
# ``weakref.ref(mesh, callback)``. When a tracked mesh is garbage-collected
# the callback fires and removes the dead entry under
# ``_global_mesh_lru_lock``, so the tracker never keeps a MeshInput alive
# and never leaks entries for meshes that are gone.
_global_mesh_lru: 'OrderedDict[int, weakref.ref]' = OrderedDict()
_global_mesh_lru_lock = threading.Lock()

# Sentinel distinguishing "not cached yet" from a cached None value.
_CACHED_NATIVE_MISS = object()


def _default_global_cache_limit() -> int:
    """Read the aggregate native-cache limit from the environment.

    Called once at module import to initialize
    :attr:`MeshInput.GLOBAL_CACHE_LIMIT`.

    Returns:
        Value of ``MESH_EXAMINER_GLOBAL_CACHE_LIMIT`` clamped to a minimum
        of 1, or 64 when unset or not a valid integer.
    """
    raw = os.environ.get('MESH_EXAMINER_GLOBAL_CACHE_LIMIT', '64')
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 64


def _touch_global_lru(mesh_key: int):
    """Mark a tracked mesh as most-recently-used in the global LRU.

    Args:
        mesh_key: ``id()`` of the mesh being accessed.
    """
    with _global_mesh_lru_lock:
        if mesh_key in _global_mesh_lru:
            _global_mesh_lru.move_to_end(mesh_key)


def _enforce_global_native_cache_limit(exclude_key: int):
    """Evict cached native objects until the aggregate total is in bounds.

    While the sum of cached native objects across all live tracked meshes
    exceeds :attr:`MeshInput.GLOBAL_CACHE_LIMIT`, the least-recently-used
    live mesh (other than ``exclude_key``) has its cache invalidated via
    :meth:`MeshInput.invalidate_cache`. Meshes with no cached objects are
    skipped as victims (invalidating them cannot reduce the total). The
    loop also breaks without touching ``exclude_key`` when it is the only
    eligible entry left, avoiding a livelock against the caller that is
    currently populating it.

    Must be called WITHOUT the calling mesh's instance lock held, because
    eviction acquires other meshes' instance locks.

    Args:
        exclude_key: ``id()`` of the mesh currently constructing a native
            object; it is never chosen as the eviction victim.
    """
    with _global_mesh_lru_lock:
        while True:
            total = 0
            victim = None
            for key, ref in _global_mesh_lru.items():
                mesh = ref()
                if mesh is None:
                    continue  # dead entry; weakref callback will remove it
                count = len(mesh._native_objects)
                total += count
                if victim is None and count > 0 and key != exclude_key:
                    victim = mesh  # first (LRU-oldest) eligible mesh
            if total <= MeshInput.GLOBAL_CACHE_LIMIT or victim is None:
                return
            victim.invalidate_cache()


class MeshInput:
    """
    Canonical mesh representation for MCP operations.

    Wraps library-specific mesh objects for transparent interchange
    between different mesh processing libraries. Supports both surface
    meshes (faces) and volumetric meshes (volume_cells).

    Example -- surface mesh::

        mesh = MeshInput(
            vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]]),
            faces=np.array([[0,1,2],[0,1,3],[1,2,3],[0,2,3]]),
            mesh_id="tet_surface"
        )

    Example -- volumetric mesh with tetrahedral cells::

        mesh = MeshInput(
            vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]]),
            faces=np.array([[0,1,2],[0,1,3],[1,2,3],[0,2,3]]),
            mesh_id="one_tet",
            volume_cells={"tet4": np.array([[0,1,2,3]])}
        )

    The ``volume_cells`` dict maps element type names (e.g. ``"tet4"``,
    ``"hex8"``, ``"prism6"``, ``"pyramid5"``) to NxK connectivity arrays
    where N is the number of cells and K is the number of nodes per cell.
    """

    _MAX_CACHE_SIZE = 8

    # Aggregate bound on cached native objects across ALL live tracked
    # meshes (see ``_global_mesh_lru`` above). Read once at module import
    # from MESH_EXAMINER_GLOBAL_CACHE_LIMIT (minimum 1).
    GLOBAL_CACHE_LIMIT = _default_global_cache_limit()

    def __init__(
        self,
        vertices: Union[np.ndarray, List[List[float]]],
        faces: Union[np.ndarray, List[List[int]]],
        mesh_id: str = "",
        source_format: str = "",
        file_path: Optional[str] = None,
        volume_cells: Optional[Dict[str, Union[np.ndarray, List[List[int]]]]] = None,
        cell_materials: Optional[Union[np.ndarray, List[int]]] = None,
        material_names: Optional[Dict[int, str]] = None,
    ):
        self.mesh_id = mesh_id or f"mesh_{id(self)}"
        self.file_path = file_path

        # Convert to numpy arrays
        self._vertices = np.array(vertices) if not isinstance(vertices, np.ndarray) else vertices
        self._faces = np.array(faces) if not isinstance(faces, np.ndarray) else faces

        self.source_format = source_format
        self._native_objects: Dict[str, Any] = {}
        self._adjacency_cache: Dict[str, Any] = {}
        self._lock = threading.Lock()

        # Volume cells: dict mapping element type name -> NxK connectivity array
        self._volume_cells: Dict[str, np.ndarray] = {}
        if volume_cells is not None:
            for etype, arr in volume_cells.items():
                converted = np.array(arr) if not isinstance(arr, np.ndarray) else arr
                self._volume_cells[etype] = converted

        # Material support for multi-material meshes
        self._cell_materials: Optional[np.ndarray] = None
        if cell_materials is not None:
            self._cell_materials = np.array(cell_materials) if not isinstance(cell_materials, np.ndarray) else cell_materials
        self._material_names: Optional[Dict[int, str]] = material_names

        # Track this mesh in the process-wide native-object LRU. The
        # weakref callback removes our entry as soon as this mesh is
        # garbage-collected, so the tracker only ever references live
        # meshes and never prolongs their lifetime.
        mesh_key = id(self)

        def _drop_from_lru(_dead_ref, key=mesh_key):
            with _global_mesh_lru_lock:
                _global_mesh_lru.pop(key, None)

        with _global_mesh_lru_lock:
            _global_mesh_lru[mesh_key] = weakref.ref(self, _drop_from_lru)

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
    def cell_materials(self) -> Optional[np.ndarray]:
        """Get material ID for each cell. Shape: (n_cells,)."""
        return self._cell_materials.copy() if self._cell_materials is not None else None

    @cell_materials.setter
    def cell_materials(self, value: Optional[Union[np.ndarray, List[int]]]):
        """Set material assignments and invalidate cache."""
        if value is None:
            self._cell_materials = None
        else:
            self._cell_materials = np.array(value) if not isinstance(value, np.ndarray) else value
        self.invalidate_cache()

    @property
    def material_names(self) -> Optional[Dict[int, str]]:
        """Get material ID to name mapping."""
        return self._material_names

    @material_names.setter
    def material_names(self, value: Optional[Dict[int, str]]):
        """Set material name mapping."""
        self._material_names = value

    def has_multiple_materials(self) -> bool:
        """Check if mesh has multiple materials assigned."""
        if self._cell_materials is None:
            return False
        return len(np.unique(self._cell_materials)) > 1

    def get_cells_by_material(self, material_id: int) -> np.ndarray:
        """Get indices of cells with the given material ID."""
        if self._cell_materials is None:
            return np.array([], dtype=int)
        return np.where(self._cell_materials == material_id)[0]

    def get_material_for_cell(self, cell_idx: int) -> Optional[int]:
        """Get material ID for a specific cell."""
        if self._cell_materials is None:
            return None
        return int(self._cell_materials[cell_idx])

    def validate_materials(self) -> List[str]:
        """Validate material assignments. Returns list of warning messages."""
        warnings = []
        if self._cell_materials is None:
            return warnings

        total_cells = self.n_volume_cells
        if total_cells > 0 and len(self._cell_materials) != total_cells:
            warnings.append(
                f"cell_materials length ({len(self._cell_materials)}) "
                f"doesn't match total cells ({total_cells})"
            )

        if self._material_names:
            used_ids = set(np.unique(self._cell_materials).tolist())
            defined_ids = set(self._material_names.keys())
            orphaned = used_ids - defined_ids
            if orphaned:
                warnings.append(f"Undefined material IDs: {orphaned}")
            unused = defined_ids - used_ids
            if unused:
                warnings.append(f"Unused material definitions: {unused}")

        return warnings

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

        Uses double-checked locking for thread safety: the outer check
        avoids acquiring the lock on the hot path (already-cached), while
        the inner check prevents redundant construction when multiple
        threads pass the outer check simultaneously. The object is bound
        to a local before the lock is released so a concurrent
        invalidate_cache() cannot race the final lookup.

        Every call refreshes this mesh's position in the process-wide
        native-object LRU tracker; when a fresh construction pushes the
        aggregate over :attr:`GLOBAL_CACHE_LIMIT`, the least-recently-used
        other live mesh has its cache evicted (see
        ``_enforce_global_native_cache_limit``).
        """
        caching = getattr(self, '_caching_enabled', True)
        if not caching:
            # Bypass cache: construct fresh object without storing it
            from ..adapters.factory import get_adapter
            adapter = get_adapter(library)
            return adapter.from_canonical(self)

        obj = self._native_objects.get(library, _CACHED_NATIVE_MISS)
        if obj is not _CACHED_NATIVE_MISS:
            _touch_global_lru(id(self))
            return obj
        with self._lock:
            # Double-checked: another thread may have constructed the
            # object while this one waited for the lock.
            obj = self._native_objects.get(library, _CACHED_NATIVE_MISS)
            if obj is _CACHED_NATIVE_MISS:
                self._construct_native(library)
                obj = self._native_objects.get(library, _CACHED_NATIVE_MISS)
        # Enforce the aggregate bound outside self._lock: eviction may
        # invalidate other meshes' caches (acquiring their locks).
        _enforce_global_native_cache_limit(id(self))
        _touch_global_lru(id(self))
        return obj

    def _construct_native(self, library: str):
        """Construct library-specific mesh object via the adapter factory."""
        from ..adapters.factory import get_adapter
        adapter = get_adapter(library)
        self._native_objects[library] = adapter.from_canonical(self)
        # Evict oldest entries if cache exceeds limit
        while len(self._native_objects) > self._MAX_CACHE_SIZE:
            oldest_key = next(iter(self._native_objects))
            del self._native_objects[oldest_key]

    def get_edge_face_adjacency(self) -> Dict[Tuple[int, int], List[int]]:
        """Return the edge-to-incident-faces adjacency map, memoized.

        Computed once via ``utils.topology.build_edge_face_map`` and
        memoized in ``self._adjacency_cache['edge_face']`` using the same
        double-checked-locking pattern as :meth:`get_native`. Any
        cache-invalidating mutation (e.g. the ``faces`` setter or
        :meth:`invalidate_cache`) drops the memoized entry so the next
        call recomputes it from the new connectivity.

        The ``build_edge_face_map`` import is deferred into the method
        body to avoid import cycles between ``core.models`` and
        ``utils.topology`` consumers.

        Returns:
            Dict mapping canonical (sorted) ``(v1, v2)`` edge tuples to
            the list of face indices incident to that edge.
        """
        obj = self._adjacency_cache.get('edge_face')
        if obj is not None:
            return obj
        with self._lock:
            obj = self._adjacency_cache.get('edge_face')
            if obj is None:
                from ..utils.topology import build_edge_face_map
                obj = build_edge_face_map(self._faces)
                self._adjacency_cache['edge_face'] = obj
        return obj

    def invalidate_cache(self):
        """Clear cached data when mesh is modified."""
        with self._lock:
            self._native_objects.clear()
            self._adjacency_cache.clear()

    def copy(self) -> 'MeshInput':
        """Create a deep copy of this mesh."""
        vc_copy = {k: v.copy() for k, v in self._volume_cells.items()} if self._volume_cells else None
        cm_copy = self._cell_materials.copy() if self._cell_materials is not None else None
        mn_copy = dict(self._material_names) if self._material_names else None
        return MeshInput(
            vertices=self._vertices.copy(),
            faces=self._faces.copy(),
            mesh_id=f"{self.mesh_id}_copy",
            source_format=self.source_format,
            file_path=self.file_path,
            volume_cells=vc_copy,
            cell_materials=cm_copy,
            material_names=mn_copy,
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
        if self._cell_materials is not None:
            unique_materials = np.unique(self._cell_materials).tolist()
            result['materials'] = {
                'count': len(unique_materials),
                'ids': unique_materials,
                'names': self._material_names or {},
            }
        return result
