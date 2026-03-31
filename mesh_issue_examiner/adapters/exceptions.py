"""
Standardized exceptions for mesh adapters.

This module defines a hierarchy of exceptions that adapters can raise
to provide clear, actionable error messages to users.
"""


class AdapterError(Exception):
    """Base exception for all adapter-related errors.
    
    All adapter exceptions should inherit from this class to allow
catch-all error handling by users.
    """
    pass


class UnsupportedOperationError(AdapterError):
    """Raised when an adapter does not support a requested operation.
    
    This error indicates that the adapter is being asked to perform
    an operation outside its declared capabilities. Users should check
    the adapter's capabilities before calling operations.
    
    Example:
        >>> adapter = get_adapter('trimesh')
        >>> if not (adapter.capabilities & AdapterCapability.VOLUMETRIC_MESH):
        ...     print("Volumetric operations not supported")
    """
    pass


class MeshConversionError(AdapterError):
    """Raised when mesh format conversion fails.
    
    This can occur when:
    - Converting from canonical to native format fails
    - Converting from native to canonical format fails
    - Element types are not supported by target format
    - Mesh data is corrupted or invalid
    """
    pass


class QualityMetricError(AdapterError):
    """Raised when quality metric computation fails.
    
    This can occur when:
    - The requested quality metric is not supported
    - The mesh element type doesn't support the metric
    - The computation encounters numerical errors
    - Required libraries are not available
    """
    pass


class FileIOError(AdapterError):
    """Raised when mesh file I/O operations fail.
    
    This can occur when:
    - File does not exist or is not readable
    - File format is not supported
    - File is corrupted or malformed
    - Write permissions are insufficient
    """
    pass


class NoAdapterAvailableError(AdapterError):
    """Raised when no adapter can be found for given requirements.
    
    This is raised by AdapterFactory when no registered adapter
    meets the requested capability requirements.
    """
    pass


class AdapterNotAvailableError(AdapterError):
    """Raised when requested adapter library is not installed.
    
    This is distinct from UnsupportedOperationError - the adapter
    class exists but the underlying library (e.g., pyvista, trimesh)
    is not installed in the environment.
    """
    pass
