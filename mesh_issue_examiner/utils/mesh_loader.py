"""
Mesh loading utilities with format support.
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any
import numpy as np

from core.models import MeshInput
from core.exceptions import MeshLoadError, UnsupportedFormatError


# Format to library support mapping
FORMAT_SUPPORT = {
    '.stl': ['trimesh', 'pyvista', 'pymeshlab'],
    '.obj': ['trimesh', 'pyvista', 'pymeshlab'],
    '.ply': ['trimesh', 'pyvista', 'pymeshlab'],
    '.off': ['trimesh', 'pymeshlab'],
    '.gltf': ['trimesh', 'pyvista'],
    '.glb': ['trimesh', 'pyvista'],
    '.vtp': ['pyvista'],
    '.vtk': ['pyvista'],
    '.3mf': ['trimesh'],
    '.nas': ['trimesh', 'pyvista', 'pymeshlab'],
    '.bdf': ['trimesh', 'pyvista', 'pymeshlab'],
}


def get_supported_formats() -> Dict[str, list]:
    """Return dictionary of supported formats and their libraries."""
    return FORMAT_SUPPORT.copy()


def is_format_supported(suffix: str) -> bool:
    """Check if file format is supported."""
    return suffix.lower() in FORMAT_SUPPORT


def load_mesh_from_file(
    file_path: Union[str, Path],
    mesh_id: Optional[str] = None,
    library: Optional[str] = None,
    **kwargs
) -> MeshInput:
    """
    Load mesh from file into canonical format.

    Args:
        file_path: Path to mesh file
        mesh_id: Optional mesh identifier
        library: Preferred library ('trimesh', 'pyvista', 'pymeshlab')
                If None, first available library is used
        **kwargs: Additional library-specific arguments

    Returns:
        MeshInput in canonical format

    Raises:
        FileNotFoundError: If file does not exist
        UnsupportedFormatError: If file format is not supported
        MeshLoadError: If mesh cannot be loaded
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix not in FORMAT_SUPPORT:
        supported = ', '.join(FORMAT_SUPPORT.keys())
        raise UnsupportedFormatError(
            f"Unsupported format: '{suffix}'. Supported formats: {supported}"
        )

    # NASTRAN formats: parse with custom parser, then build MeshInput directly
    if suffix in ('.nas', '.bdf'):
        from utils.nas_parser import parse_nas
        try:
            vertices, faces = parse_nas(str(file_path))
        except Exception as e:
            raise MeshLoadError(
                f"Failed to parse NASTRAN file {file_path}: {e}"
            ) from e

        mesh_id = mesh_id or file_path.stem
        return MeshInput(
            vertices=vertices,
            faces=faces,
            mesh_id=mesh_id,
            source_format='nas',
            file_path=str(file_path)
        )

    # Determine library to use
    available_libraries = FORMAT_SUPPORT[suffix]

    if library is None:
        # Use first available library
        selected_library = None
        for lib in available_libraries:
            if _is_library_available(lib):
                selected_library = lib
                break

        if selected_library is None:
            raise MeshLoadError(
                f"None of the supported libraries for {suffix} are installed. "
                f"Install at least one of: {', '.join(available_libraries)}"
            )
    else:
        if library not in available_libraries:
            raise UnsupportedFormatError(
                f"Library '{library}' does not support {suffix} format. "
                f"Supported libraries: {', '.join(available_libraries)}"
            )
        if not _is_library_available(library):
            raise MeshLoadError(
                f"Library '{library}' is not installed. "
                f"Install with: pip install {library}"
            )
        selected_library = library

    # Load using selected library
    try:
        adapter = _get_adapter(selected_library)
        native_mesh = adapter.load_mesh(str(file_path), **kwargs)
        vertices = adapter.get_vertices(native_mesh)
        faces = adapter.get_faces(native_mesh)

        mesh_id = mesh_id or file_path.stem

        return MeshInput(
            vertices=vertices,
            faces=faces,
            mesh_id=mesh_id,
            source_format=suffix[1:],  # Remove leading dot
            file_path=str(file_path)
        )

    except ImportError as e:
        raise MeshLoadError(
            f"Failed to import {selected_library}: {str(e)}"
        ) from e
    except Exception as e:
        raise MeshLoadError(
            f"Failed to load mesh from {file_path}: {str(e)}"
        ) from e


def create_mesh_from_arrays(
    vertices: Union[np.ndarray, list],
    faces: Union[np.ndarray, list],
    mesh_id: str = "",
    library: str = 'trimesh'
) -> MeshInput:
    """
    Create MeshInput from vertex and face arrays.

    Args:
        vertices: Nx3 array of vertex coordinates
        faces: Mx3 array of face indices
        mesh_id: Optional mesh identifier
        library: Library to use for native mesh creation

    Returns:
        MeshInput in canonical format
    """
    # Convert to numpy arrays
    vertices = np.array(vertices) if not isinstance(vertices, np.ndarray) else vertices
    faces = np.array(faces) if not isinstance(faces, np.ndarray) else faces

    mesh_id = mesh_id or f"mesh_{id(vertices)}"

    return MeshInput(
        vertices=vertices,
        faces=faces,
        mesh_id=mesh_id,
        source_format='arrays'
    )


def save_mesh_to_file(
    mesh: MeshInput,
    file_path: Union[str, Path],
    library: Optional[str] = None,
    **kwargs
) -> None:
    """
    Save mesh to file.

    Args:
        mesh: MeshInput in canonical format
        file_path: Output file path
        library: Library to use (default: auto-detect from extension)
        **kwargs: Additional library-specific arguments
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if library is None:
        if suffix not in FORMAT_SUPPORT:
            library = 'trimesh'  # Fallback
        else:
            library = FORMAT_SUPPORT[suffix][0]

    adapter = _get_adapter(library)
    native_mesh = adapter.from_canonical(mesh)
    native_mesh.save(str(file_path), **kwargs)


def _is_library_available(library: str) -> bool:
    """Check if a library is installed."""
    try:
        _get_adapter(library)
        return True
    except ImportError:
        return False


def _get_adapter(library: str):
    """Get adapter instance for library."""
    if library == 'trimesh':
        from adapters.trimesh_adapter import TrimeshAdapter
        return TrimeshAdapter()
    elif library == 'pyvista':
        from adapters.pyvista_adapter import PyVistaAdapter
        return PyVistaAdapter()
    elif library == 'pymeshlab':
        from adapters.pymeshlab_adapter import PyMeshLabAdapter
        return PyMeshLabAdapter()
    else:
        raise ValueError(f"Unknown library: {library}")
