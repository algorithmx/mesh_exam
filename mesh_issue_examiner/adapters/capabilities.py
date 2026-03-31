"""
Capability-based adapter management system.

This module provides:
- AdapterCapability enum for declaring adapter features
- Capability decorators for runtime checking
- AdapterRegistry for centralized adapter management
"""

from enum import Flag, auto
from functools import wraps
from typing import Dict, Type, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .base_adapter import MeshAdapter


class AdapterCapability(Flag):
    """Capability flags for mesh adapters.
    
    Adapters declare their supported capabilities by returning
    appropriate flags from their `capabilities` property. This
    allows runtime discovery of adapter functionality.
    
    Example:
        >>> adapter.capabilities
        AdapterCapability.SURFACE_MESH | AdapterCapability.FILE_IO
        >>> 
        >>> # Check if adapter supports volumetric meshes
        >>> if adapter.capabilities & AdapterCapability.VOLUMETRIC_MESH:
        ...     print("Supports volumetric meshes")
    """
    
    # Mesh type support
    SURFACE_MESH = auto()
    """Supports triangular surface meshes (2D)."""
    
    VOLUMETRIC_MESH = auto()
    """Supports volumetric meshes: tetrahedra, hexahedra, prisms, pyramids (3D)."""
    
    # I/O capabilities
    FILE_IO = auto()
    """Supports loading and saving mesh files."""
    
    # Quality computation
    QUALITY_METRICS = auto()
    """Supports computing cell quality metrics (aspect ratio, skewness, etc.)."""
    
    # Topology checking
    TOPOLOGY_CHECKS = auto()
    """Supports topology checks: manifold, watertight, boundary edges."""
    
    # Intersection detection
    SELF_INTERSECTION = auto()
    """Supports self-intersection detection."""
    
    # Mesh modification
    REMESHING = auto()
    """Supports remeshing operations: smoothing, decimation, subdivision."""
    
    # Common capability combinations
    SURFACE_FULL = SURFACE_MESH | FILE_IO | QUALITY_METRICS | TOPOLOGY_CHECKS
    """Full surface mesh support."""
    
    VOLUMETRIC_FULL = VOLUMETRIC_MESH | FILE_IO | QUALITY_METRICS | TOPOLOGY_CHECKS
    """Full volumetric mesh support."""
    
    ALL = SURFACE_FULL | VOLUMETRIC_FULL | SELF_INTERSECTION | REMESHING
    """All capabilities (rare in practice)."""


def require_capability(cap: AdapterCapability):
    """Decorator that checks adapter capability before method execution.
    
    This decorator should be used on adapter methods that require specific
capabilities. If the adapter does not have the required capability,
    UnsupportedOperationError is raised with an informative message.
    
    Args:
        cap: The required capability flag(s)
        
    Example:
        >>> class MyAdapter(MeshAdapter):
        ...     @require_capability(AdapterCapability.VOLUMETRIC_MESH)
        ...     def from_canonical_volumetric(self, mesh_input):
        ...         # This method will only be called if adapter has VOLUMETRIC_MESH
        ...         pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Import here to avoid circular imports
            from .exceptions import UnsupportedOperationError
            
            adapter_caps = getattr(self, 'capabilities', AdapterCapability(0))
            
            if not (adapter_caps & cap):
                raise UnsupportedOperationError(
                    f"{self.library_name} adapter does not support {cap.name}. "
                    f"Available capabilities: {adapter_caps}")
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def require_any_capability(*caps: AdapterCapability):
    """Decorator that requires at least one of the specified capabilities.
    
    Args:
        *caps: Capability flags, at least one must be present
        
    Example:
        >>> @require_any_capability(
        ...     AdapterCapability.SURFACE_MESH,
        ...     AdapterCapability.VOLUMETRIC_MESH
        ... )
        ... def process_mesh(self, mesh):
        ...     pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            from .exceptions import UnsupportedOperationError
            
            adapter_caps = getattr(self, 'capabilities', AdapterCapability(0))
            
            if not any(adapter_caps & cap for cap in caps):
                cap_names = " or ".join(cap.name for cap in caps)
                raise UnsupportedOperationError(
                    f"{self.library_name} adapter requires one of: {cap_names}. "
                    f"Available capabilities: {adapter_caps}")
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


class AdapterRegistry:
    """Central registry for adapter classes and their capabilities.
    
    This registry provides:
    - Adapter registration with automatic capability detection
    - Capability-based adapter discovery
    - Best adapter selection for given requirements
    
    Example:
        >>> # Register an adapter
        >>> AdapterRegistry.register('pyvista', PyVistaAdapter)
        >>> 
        >>> # Find best adapter for volumetric quality checks
        >>> name = AdapterRegistry.find_adapter_for_task(
        ...     requirements=AdapterCapability.VOLUMETRIC_MESH | AdapterCapability.QUALITY_METRICS,
        ...     prefer='pyvista'
        ... )
        >>> adapter = AdapterRegistry.get_adapter(name)
    """
    
    _adapters: Dict[str, Type['MeshAdapter']] = {}
    _capabilities: Dict[str, AdapterCapability] = {}
    _instances: Dict[str, 'MeshAdapter'] = {}
    
    @classmethod
    def register(cls, name: str, adapter_class: Type['MeshAdapter']):
        """Register an adapter class.
        
        Args:
            name: Unique identifier for this adapter
            adapter_class: The adapter class to register
            
        Example:
            >>> AdapterRegistry.register('pyvista', PyVistaAdapter)
            >>> AdapterRegistry.register('trimesh', TrimeshAdapter)
        """
        cls._adapters[name] = adapter_class
        
        # Try to get capabilities from a temporary instance
        try:
            temp_instance = adapter_class()
            cls._capabilities[name] = temp_instance.capabilities
        except Exception:
            # If instantiation fails (e.g., missing dependencies),
            # store empty capabilities
            cls._capabilities[name] = AdapterCapability(0)
    
    @classmethod
    def get_adapter(cls, name: str) -> 'MeshAdapter':
        """Get or create an adapter instance.
        
        This method implements singleton-like behavior - the same
        adapter instance is returned for subsequent calls with the
        same name.
        
        Args:
            name: Registered adapter name
            
        Returns:
            Adapter instance
            
        Raises:
            ValueError: If adapter name is not registered
            AdapterNotAvailableError: If adapter library is not installed
        """
        from .exceptions import AdapterNotAvailableError
        
        if name not in cls._instances:
            if name not in cls._adapters:
                raise ValueError(
                    f"Unsupported library: {name}. "
                    f"Supported libraries: {', '.join(cls._adapters.keys())}"
                )
            
            adapter_class = cls._adapters[name]
            instance = adapter_class()
            
            # Check if library is available
            if not instance.is_available:
                raise AdapterNotAvailableError(
                    f"Adapter '{name}' is registered but the underlying "
                    f"library is not installed. Please install {name} to use "
                    f"this adapter."
                )
            
            cls._instances[name] = instance
        
        return cls._instances[name]
    
    @classmethod
    def get_capabilities(cls, name: str) -> AdapterCapability:
        """Get capabilities for a registered adapter.
        
        Args:
            name: Registered adapter name
            
        Returns:
            Capability flags for the adapter
            
        Raises:
            ValueError: If adapter name is not registered
        """
        if name not in cls._capabilities:
            raise ValueError(f"Unknown adapter: '{name}'")
        return cls._capabilities[name]
    
    @classmethod
    def list_adapters(cls) -> Set[str]:
        """Return set of all registered adapter names."""
        return set(cls._adapters.keys())
    
    @classmethod
    def list_available_adapters(cls) -> Set[str]:
        """Return set of adapters whose libraries are installed."""
        available = set()
        for name in cls._adapters:
            try:
                adapter = cls.get_adapter(name)
                if adapter.is_available:
                    available.add(name)
            except Exception:
                pass
        return available
    
    @classmethod
    def find_adapter_for_task(
        cls,
        requirements: AdapterCapability,
        prefer: Optional[str] = None,
        require_available: bool = True
    ) -> str:
        """Find the best adapter for given capability requirements.
        
        This method searches registered adapters and returns the name
        of an adapter that meets all the specified requirements.
        
        Args:
            requirements: Required capability flags (combined with |)
            prefer: Preferred adapter name (used if it meets requirements)
            require_available: If True, only consider adapters whose
                libraries are installed
                
        Returns:
            Name of the best matching adapter
            
        Raises:
            NoAdapterAvailableError: If no adapter meets requirements
            
        Example:
            >>> # Find adapter for volumetric quality checks
            >>> name = AdapterRegistry.find_adapter_for_task(
            ...     AdapterCapability.VOLUMETRIC_MESH | AdapterCapability.QUALITY_METRICS
            ... )
            >>> 
            >>> # Prefer meshio but fall back to others
            >>> name = AdapterRegistry.find_adapter_for_task(
            ...     AdapterCapability.FILE_IO,
            ...     prefer='meshio'
            ... )
        """
        from .exceptions import NoAdapterAvailableError
        
        # Check preferred adapter first
        if prefer and prefer in cls._adapters:
            caps = cls._capabilities.get(prefer, AdapterCapability(0))
            if caps & requirements == requirements:
                if not require_available:
                    return prefer
                try:
                    adapter = cls.get_adapter(prefer)
                    if adapter.is_available:
                        return prefer
                except Exception:
                    pass
        
        # Search all adapters
        candidates = []
        for name, caps in cls._capabilities.items():
            if caps & requirements == requirements:
                candidates.append((name, caps))
        
        if not candidates:
            raise NoAdapterAvailableError(
                f"No adapter available for requirements: {requirements}. "
                f"Registered adapters: {list(cls._adapters.keys())}"
            )
        
        # Filter by availability if required
        if require_available:
            available_candidates = []
            for name, caps in candidates:
                try:
                    adapter = cls.get_adapter(name)
                    if adapter.is_available:
                        available_candidates.append((name, caps))
                except Exception:
                    pass
            candidates = available_candidates
        
        if not candidates:
            raise NoAdapterAvailableError(
                f"No available adapter for requirements: {requirements}. "
                f"Required libraries may not be installed."
            )
        
        # Select best candidate (most specific capabilities)
        # Prefer adapters with fewer extra capabilities beyond requirements
        def specificity_score(item):
            name, caps = item
            extra_caps = caps & ~requirements
            # Fewer extra caps = more specific = better
            return bin(extra_caps.value).count('1')
        
        candidates.sort(key=specificity_score)
        return candidates[0][0]
    
    @classmethod
    def get_capability_matrix(cls) -> Dict[str, AdapterCapability]:
        """Get capabilities of all registered adapters.
        
        Returns:
            Dictionary mapping adapter names to their capabilities
        """
        return cls._capabilities.copy()
    
    @classmethod
    def clear_cache(cls):
        """Clear the adapter instance cache.
        
        This is useful for testing or when adapter configuration
        needs to be reset.
        """
        cls._instances.clear()
    
    @classmethod
    def unregister(cls, name: str):
        """Unregister an adapter.
        
        Args:
            name: Adapter name to unregister
            
        Note:
            This also clears the instance cache for that adapter.
        """
        cls._adapters.pop(name, None)
        cls._capabilities.pop(name, None)
        cls._instances.pop(name, None)
