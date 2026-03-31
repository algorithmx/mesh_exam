"""
Library adapters for mesh processing.

This package provides adapters for various mesh processing libraries,
providing a unified interface for both 2D surface meshes and 3D volumetric
meshes across different backends.

**Quick Start:**
    >>> from mesh_issue_examiner.adapters import get_adapter
    >>> adapter = get_adapter('pyvista')
    >>> mesh = adapter.load_mesh('model.stl')
    >>> areas = adapter.compute_face_areas(mesh)

**Capability-based Selection:**
    >>> from mesh_issue_examiner.adapters import (
    ...     find_adapter_for_task,
    ...     AdapterCapability
    ... )
    >>> name = find_adapter_for_task(
    ...     AdapterCapability.VOLUMETRIC_MESH | AdapterCapability.QUALITY_METRICS
    ... )
    >>> adapter = get_adapter(name)

**Listing Adapters:**
    >>> from mesh_issue_examiner.adapters import (
    ...     list_supported_libraries,
    ...     get_available_libraries,
    ...     get_capability_matrix
    ... )
    >>> all_adapters = list_supported_libraries()
    >>> available = get_available_libraries()

**Available Adapters:**
- ``pyvista``: VTK-based, full volumetric support
- ``trimesh``: Pure Python, surface meshes only
- ``pymeshlab``: MeshLab bindings, surface focus with remeshing
- ``meshio``: Universal I/O, 30+ formats
- ``vedo``: Visualization and analysis
"""

# Base adapter and capabilities
from .base_adapter import MeshAdapter
from .capabilities import (
    AdapterCapability,
    require_capability,
    require_any_capability,
    AdapterRegistry,
)
from .exceptions import (
    AdapterError,
    UnsupportedOperationError,
    MeshConversionError,
    QualityMetricError,
    FileIOError,
    NoAdapterAvailableError,
    AdapterNotAvailableError,
)

# Factory functions
from .factory import (
    get_adapter,
    find_adapter_for_task,
    list_supported_libraries,
    get_available_libraries,
    get_capability_matrix,
    get_adapter_capabilities,
    has_capability,
    clear_adapter_cache,
    SUPPORTED_LIBRARIES,
)

# Adapter implementations
from .trimesh_adapter import TrimeshAdapter
from .pyvista_adapter import PyVistaAdapter
from .pymeshlab_adapter import PyMeshLabAdapter
from .meshio_adapter import MeshioAdapter
from .vedo_adapter import VedoAdapter

__all__ = [
    # Base classes and capabilities
    "MeshAdapter",
    "AdapterCapability",
    "require_capability",
    "require_any_capability",
    "AdapterRegistry",
    # Exceptions
    "AdapterError",
    "UnsupportedOperationError",
    "MeshConversionError",
    "QualityMetricError",
    "FileIOError",
    "NoAdapterAvailableError",
    "AdapterNotAvailableError",
    # Factory functions
    "get_adapter",
    "find_adapter_for_task",
    "list_supported_libraries",
    "get_available_libraries",
    "get_capability_matrix",
    "get_adapter_capabilities",
    "has_capability",
    "clear_adapter_cache",
    "SUPPORTED_LIBRARIES",
    # Adapter implementations
    "TrimeshAdapter",
    "PyVistaAdapter",
    "PyMeshLabAdapter",
    "MeshioAdapter",
    "VedoAdapter",
]

# Auto-register all adapters with the registry
# This ensures adapters are available via AdapterRegistry
AdapterRegistry.register('pyvista', PyVistaAdapter)
AdapterRegistry.register('trimesh', TrimeshAdapter)
AdapterRegistry.register('pymeshlab', PyMeshLabAdapter)
AdapterRegistry.register('meshio', MeshioAdapter)
AdapterRegistry.register('vedo', VedoAdapter)
