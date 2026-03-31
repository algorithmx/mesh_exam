"""
Adapter factory module.

This module provides convenient factory functions for creating and managing
mesh adapter instances. It serves as a simplified interface to the underlying
AdapterRegistry system.

**Quick Start - Legacy API:**
    >>> from mesh_issue_examiner.adapters import get_adapter
    >>> adapter = get_adapter('pyvista')
    >>> mesh = adapter.load_mesh('model.vtu')

**Capability-based Discovery:**
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
    >>> capabilities = get_capability_matrix()
"""

from typing import Dict, List, Optional

from .base_adapter import MeshAdapter
from .capabilities import AdapterCapability, AdapterRegistry
from .exceptions import NoAdapterAvailableError

# Supported adapter libraries - kept for reference and validation
SUPPORTED_LIBRARIES = ['trimesh', 'pyvista', 'pymeshlab', 'meshio', 'vedo']


def get_adapter(library: str) -> MeshAdapter:
    """Get or create an adapter instance for the specified library.

    This is the main factory function for retrieving adapter instances.
    Adapters are cached, so subsequent calls with the same library name
    return the same instance.

    Supported libraries:
        - ``'trimesh'``: Pure Python surface mesh processing
        - ``'pyvista'``: VTK-based volumetric and surface mesh processing
        - ``'pymeshlab'``: MeshLab bindings for surface mesh filtering
        - ``'meshio'``: Universal mesh I/O (30+ formats)
        - ``'vedo'``: Scientific visualization and analysis

    Args:
        library: Name of the library adapter to retrieve

    Returns:
        MeshAdapter instance for the specified library

    Raises:
        ValueError: If the library name is not registered
        AdapterNotAvailableError: If the adapter library is not installed

    Example:
        >>> adapter = get_adapter('pyvista')
        >>> print(adapter.capabilities)
        >>> mesh = adapter.load_mesh('model.vtu')
    """
    return AdapterRegistry.get_adapter(library)


def find_adapter_for_task(
    requirements: AdapterCapability,
    prefer: Optional[str] = None,
    require_available: bool = True
) -> str:
    """Find the best adapter for given capability requirements.

    This function searches registered adapters and returns the name of
    an adapter that meets all specified requirements.

    Args:
        requirements: Required capability flags (combine with ``|``)
        prefer: Preferred adapter name (used if it meets requirements)
        require_available: If True, only consider adapters whose
            libraries are installed

    Returns:
        Name of the best matching adapter

    Raises:
        NoAdapterAvailableError: If no adapter meets the requirements

    Example:
        >>> # Find adapter for volumetric quality checks
        >>> name = find_adapter_for_task(
        ...     AdapterCapability.VOLUMETRIC_MESH | AdapterCapability.QUALITY_METRICS
        ... )
        >>> adapter = get_adapter(name)

        >>> # Prefer meshio but fall back to others
        >>> name = find_adapter_for_task(
        ...     AdapterCapability.FILE_IO,
        ...     prefer='meshio'
        ... )
    """
    return AdapterRegistry.find_adapter_for_task(
        requirements=requirements,
        prefer=prefer,
        require_available=require_available
    )


def list_supported_libraries() -> List[str]:
    """Return list of all registered adapter names.

    Returns:
        List of adapter names that can be passed to :func:`get_adapter`

    Example:
        >>> libraries = list_supported_libraries()
        >>> print(libraries)
        ['trimesh', 'pyvista', 'pymeshlab', 'meshio', 'vedo']
    """
    return sorted(AdapterRegistry.list_adapters())


def get_available_libraries() -> List[str]:
    """Return list of adapters that are installed and ready to use.

    This checks each adapter's availability by attempting to import
    the underlying library. This may be slow on first call.

    Returns:
        List of available adapter names

    Example:
        >>> available = get_available_libraries()
        >>> if 'pyvista' in available:
        ...     adapter = get_adapter('pyvista')
    """
    return sorted(AdapterRegistry.list_available_adapters())


def get_capability_matrix() -> Dict[str, AdapterCapability]:
    """Get capabilities of all registered adapters.

    Returns:
        Dictionary mapping adapter names to their capability flags

    Example:
        >>> capabilities = get_capability_matrix()
        >>> for name, caps in capabilities.items():
        ...     print(f"{name}: {caps}")
    """
    return AdapterRegistry.get_capability_matrix()


def clear_adapter_cache() -> None:
    """Clear the adapter instance cache.

    This forces new adapter instances to be created on subsequent
    :func:`get_adapter` calls. Useful for testing or when adapter
    state needs to be reset.

    Example:
        >>> clear_adapter_cache()
        >>> # Next call will create fresh adapter instance
        >>> adapter = get_adapter('pyvista')
    """
    AdapterRegistry.clear_cache()


def get_adapter_capabilities(library: str) -> AdapterCapability:
    """Get capabilities for a specific adapter.

    Args:
        library: Adapter name

    Returns:
        Capability flags for the adapter

    Raises:
        ValueError: If adapter is not registered

    Example:
        >>> caps = get_adapter_capabilities('pyvista')
        >>> if caps & AdapterCapability.VOLUMETRIC_MESH:
        ...     print("Supports volumetric meshes")
    """
    return AdapterRegistry.get_capabilities(library)


def has_capability(library: str, capability: AdapterCapability) -> bool:
    """Check if an adapter has a specific capability.

    Args:
        library: Adapter name
        capability: Capability to check

    Returns:
        True if the adapter has the capability

    Example:
        >>> if has_capability('trimesh', AdapterCapability.VOLUMETRIC_MESH):
        ...     print("Trimesh supports volumetric meshes")
        ... else:
        ...     print("Trimesh does not support volumetric meshes")
    """
    try:
        caps = AdapterRegistry.get_capabilities(library)
        return bool(caps & capability)
    except ValueError:
        return False


# Convenience re-exports for backward compatibility
__all__ = [
    # Main factory functions
    'get_adapter',
    'find_adapter_for_task',
    # Listing functions
    'list_supported_libraries',
    'get_available_libraries',
    'get_capability_matrix',
    'get_adapter_capabilities',
    'has_capability',
    # Utility functions
    'clear_adapter_cache',
    # Constants
    'SUPPORTED_LIBRARIES',
]
