"""
Mesh loading utilities with format support.
"""

import os
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import numpy as np

from ..core.models import MeshInput
from ..core.exceptions import MeshLoadError, UnsupportedFormatError


def _validate_mesh_arrays(vertices: np.ndarray, faces: np.ndarray) -> None:
    """Validate vertex and face arrays after loading.

    Raises:
        MeshLoadError: If arrays are empty, malformed, or contain invalid indices.
    """
    if len(vertices) == 0:
        raise MeshLoadError("Loaded mesh has no vertices")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise MeshLoadError(
            f"Vertices must be Nx3, got shape {vertices.shape}"
        )
    if len(faces) == 0:
        raise MeshLoadError("Loaded mesh has no faces")
    if faces.ndim != 2 or faces.shape[1] < 3:
        raise MeshLoadError(
            f"Faces must be NxK with K>=3, got shape {faces.shape}"
        )
    max_idx = int(faces.max())
    if max_idx >= len(vertices):
        raise MeshLoadError(
            f"Face index {max_idx} exceeds vertex count {len(vertices)}"
        )


# Format to library support mapping
# Libraries are listed in order of preference (first = preferred)
FORMAT_SUPPORT = {
    # Common surface formats
    '.stl': ['trimesh', 'pyvista', 'pymeshlab', 'meshio'],
    '.obj': ['trimesh', 'pyvista', 'pymeshlab', 'meshio'],
    '.ply': ['trimesh', 'pyvista', 'pymeshlab', 'meshio'],
    '.off': ['trimesh', 'pymeshlab', 'meshio'],
    '.gltf': ['trimesh', 'pyvista'],
    '.glb': ['trimesh', 'pyvista'],
    
    # VTK formats (volumetric support)
    '.vtp': ['pyvista', 'meshio'],
    '.vtk': ['pyvista', 'meshio', 'vedo'],
    '.vtu': ['pyvista', 'meshio', 'vedo'],
    
    # 3MF
    '.3mf': ['trimesh', 'meshio'],
    
    # NASTRAN formats
    '.nas': ['trimesh', 'pyvista', 'pymeshlab', 'meshio'],
    '.bdf': ['trimesh', 'pyvista', 'pymeshlab', 'meshio'],
    '.fem': ['meshio'],  # NASTRAN alternative extension
    
    # Gmsh formats
    '.msh': ['meshio', 'pyvista'],  # Gmsh native format
    
    # Abaqus
    '.inp': ['meshio'],
    
    # ANSYS
    '.ans': ['meshio'],
    
    # Exodus (parallel HDF5 format)
    '.e': ['meshio'],
    '.exo': ['meshio'],
    
    # XDMF (FEM standard)
    '.xdmf': ['meshio'],
    '.xmf': ['meshio'],
    
    # MED/Salome
    '.med': ['meshio'],
    
    # CGNS (CFD format)
    '.cgns': ['meshio'],
    
    # FLAC3D
    '.f3grid': ['meshio'],
    
    # TetGen
    '.node': ['meshio'],
    '.ele': ['meshio'],
    
    # Medit
    '.mesh': ['meshio'],
    '.meshb': ['meshio'],
    
    # SU2
    '.su2': ['meshio'],
    
    # UGRID
    '.ugrid': ['meshio'],
    
    # Tecplot
    '.dat': ['meshio'],
    
    # PERMAS
    '.post': ['meshio'],
    '.dato': ['meshio'],
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
    extract_volume_cells: bool = True,
    allowed_dirs: Optional[List[Union[str, Path]]] = None,
    **kwargs
) -> MeshInput:
    """
    Load mesh from file into canonical format.

    Args:
        file_path: Path to mesh file
        mesh_id: Optional mesh identifier
        library: Preferred library ('trimesh', 'pyvista', 'pymeshlab', 'meshio', 'vedo')
                If None, first available library is used
        extract_volume_cells: If True, extract volumetric cells for 3D meshes
                (requires meshio for best support)
        allowed_dirs: Optional list of directories that mesh files may be loaded from.
                If provided, file_path must resolve to a location under one of these dirs.
        **kwargs: Additional library-specific arguments

    Returns:
        MeshInput in canonical format (with volume_cells populated for 3D meshes)

    Raises:
        FileNotFoundError: If file does not exist
        UnsupportedFormatError: If file format is not supported
        MeshLoadError: If mesh cannot be loaded
        ValueError: If file_path is outside allowed directories
    """
    file_path = Path(file_path).resolve()

    if allowed_dirs is not None:
        resolved_allowed = [Path(d).resolve() for d in allowed_dirs]
        if not any(str(file_path).startswith(str(d) + os.sep) or str(file_path) == str(d) for d in resolved_allowed):
            raise ValueError(
                f"File '{file_path}' is outside allowed directories: "
                f"{[str(d) for d in resolved_allowed]}"
            )

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
        from .nas_parser import parse_nas
        try:
            vertices, faces, volume_cells = parse_nas(str(file_path))
        except Exception as e:
            raise MeshLoadError(
                f"Failed to parse NASTRAN file {file_path}: {e}"
            ) from e

        _validate_mesh_arrays(vertices, faces)
        mesh_id = mesh_id or file_path.stem
        return MeshInput(
            vertices=vertices,
            faces=faces,
            mesh_id=mesh_id,
            source_format='nas',
            file_path=str(file_path),
            volume_cells=volume_cells if volume_cells else None,
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
        from ..adapters.factory import get_adapter
        adapter = get_adapter(selected_library)

        native_mesh = adapter.load_mesh(str(file_path), **kwargs)

        if extract_volume_cells:
            # Delegate to the adapter's to_canonical() which handles
            # both surface and volumetric meshes appropriately.
            mesh_input = adapter.to_canonical(
                native_mesh,
                mesh_id=mesh_id or file_path.stem
            )
            mesh_input.file_path = str(file_path)
            mesh_input.source_format = suffix[1:]
            return mesh_input

        # Surface-only fast path (skip volume cells)
        vertices = adapter.get_vertices(native_mesh)
        faces = adapter.get_faces(native_mesh)

        _validate_mesh_arrays(vertices, faces)
        mesh_id = mesh_id or file_path.stem

        return MeshInput(
            vertices=vertices,
            faces=faces,
            mesh_id=mesh_id,
            source_format=suffix[1:],
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

    _validate_mesh_arrays(vertices, faces)
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
                Use 'meshio' for best volumetric mesh support
        **kwargs: Additional library-specific arguments
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if library is None:
        if suffix not in FORMAT_SUPPORT:
            # For unsupported formats, try meshio first (most capable)
            if _is_library_available('meshio'):
                library = 'meshio'
            else:
                library = 'pyvista'  # Fallback
        else:
            # Prefer meshio for formats that support volumetric data
            libs = FORMAT_SUPPORT[suffix]
            if 'meshio' in libs and _is_library_available('meshio'):
                library = 'meshio'
            else:
                library = libs[0]

    from ..adapters.factory import get_adapter
    adapter = get_adapter(library)

    # Special handling for meshio adapter's save method
    if library == 'meshio':
        native_mesh = adapter.from_canonical(mesh)
        adapter.save_mesh(native_mesh, str(file_path), **kwargs)
    else:
        native_mesh = adapter.from_canonical(mesh)
        native_mesh.save(str(file_path), **kwargs)


def _is_library_available(library: str) -> bool:
    """Check if a library is installed."""
    try:
        from ..adapters.factory import get_adapter
        get_adapter(library)
        return True
    except ImportError:
        return False
