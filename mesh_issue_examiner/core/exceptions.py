"""
Custom exceptions for mesh issue examination.
"""


class MeshExaminerError(Exception):
    """Base exception for mesh examiner errors."""
    pass


class MeshLoadError(MeshExaminerError):
    """Raised when mesh cannot be loaded from file."""
    pass


class UnsupportedFormatError(MeshExaminerError):
    """Raised when file format is not supported."""
    pass


class OperationError(MeshExaminerError):
    """Raised when an operation fails."""
    pass


class InvalidMeshError(MeshExaminerError):
    """Raised when mesh data is invalid or corrupted."""
    pass


class LibraryNotAvailableError(MeshExaminerError):
    """Raised when required library is not installed."""
    pass


class VisualizationError(MeshExaminerError):
    """Raised when rendering or visualization fails."""
    pass
