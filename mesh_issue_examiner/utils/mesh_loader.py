"""
Mesh loading utilities with format support.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import numpy as np

from ..core.models import MeshInput
from ..core.exceptions import MeshLoadError, UnsupportedFormatError

logger = logging.getLogger(__name__)


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
    material_array_name: Optional[str] = None,
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
        material_array_name: Optional name of cell data array containing material IDs.
                When provided, the loader will attempt to extract material assignments
                from the mesh file's cell data. Supported for formats like VTK, Exodus,
                Gmsh, etc. that store per-cell data.
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
        mesh_input = MeshInput(
            vertices=vertices,
            faces=faces,
            mesh_id=mesh_id,
            source_format='nas',
            file_path=str(file_path),
            volume_cells=volume_cells if volume_cells else None,
        )

        # Extract materials from NASTRAN if requested.  NASTRAN stores
        # material assignments in property cards (PSOLID PID->MID), not in
        # named cell-data arrays, so material_array_name is not forwarded
        # as a card key here; the default PSOLID mapping is used.
        if material_array_name:
            from .material_reader import extract_materials_from_nas
            cell_mats, mat_names = extract_materials_from_nas(str(file_path))
            if cell_mats is not None:
                mesh_input.cell_materials = cell_mats
                if mat_names:
                    mesh_input.material_names = mat_names

        return mesh_input

    # Gmsh .msh: try the native 2.2 ASCII parser first (physical groups
    # become cell_materials/material_names); fall back to meshio for the
    # variants the native parser declines (binary, 4.x).
    if suffix == '.msh':
        try:
            from .gmsh_parser import parse_msh
            vertices, faces, volume_cells, cell_mats, mat_names = parse_msh(
                str(file_path))
        except ValueError as e:
            logger.info(
                "Native Gmsh parser declined %s (%s); falling back to "
                "meshio", file_path, e)
        else:
            _validate_mesh_arrays(vertices, faces)
            mesh_input = MeshInput(
                vertices=vertices,
                faces=faces,
                mesh_id=mesh_id or file_path.stem,
                source_format='msh',
                file_path=str(file_path),
                volume_cells=volume_cells if volume_cells else None,
            )
            # Physical-group tags are the native material mechanism for
            # .msh files; attach them when any material array is requested.
            if material_array_name and cell_mats is not None:
                mesh_input.cell_materials = cell_mats
                if mat_names:
                    mesh_input.material_names = mat_names
            return mesh_input

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

            # Extract material assignments if requested
            if material_array_name:
                _extract_material_data(mesh_input, native_mesh, material_array_name)

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
    except SystemExit as e:
        # Some libraries (meshio's read helper) sys.exit() on unreadable
        # files instead of raising; translate into the loader's contract.
        raise MeshLoadError(
            f"Failed to load mesh from {file_path}: library exited "
            f"with status {e.code}"
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

    # NASTRAN write-back: emit a free-field bulk-data deck natively so
    # the quadratic CHEXA/CPENTA forms keep their midside nodes.
    if suffix in ('.nas', '.bdf'):
        from .nas_parser import write_nas
        write_nas(
            mesh.vertices,
            mesh.faces,
            mesh.volume_cells if mesh.has_volume_cells else {},
            str(file_path),
        )
        return

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


def _extract_material_data(
    mesh_input: MeshInput,
    native_mesh: Any,
    array_name: str,
) -> None:
    """Extract material assignments from a native mesh object's cell data.

    Tries PyVista cell_data first, then falls back to meshio-style
    cell_data dicts.  For PyVista grids the array is in *grid cell order*;
    it is then reordered into ``volume_cells`` block order (per element
    type, first-occurrence) so that ``cell_materials[i]`` corresponds to
    the cell at global index ``i`` used by the multi-material operations.

    Args:
        mesh_input: The MeshInput to attach materials to.
        native_mesh: The library-specific mesh object.
        array_name: Name of the cell data array containing material IDs.
    """
    cell_materials = None

    # Try PyVista-style cell_data access
    if hasattr(native_mesh, 'cell_data'):
        cd = native_mesh.cell_data
        if isinstance(cd, dict) and array_name in cd:
            cell_materials = np.asarray(cd[array_name]).flatten()
        elif hasattr(cd, 'get') and cd.get(array_name) is not None:
            cell_materials = np.asarray(cd.get(array_name)).flatten()

    # Reorder from grid cell order into volume_cells block order.  The
    # canonical volume_cells dict groups cells by element type (in order
    # of first occurrence in the grid), so a grid-order array only aligns
    # when cells happen to be grouped by type.
    if (
        cell_materials is not None
        and mesh_input.has_volume_cells
        and hasattr(native_mesh, 'celltypes')
    ):
        cell_materials = _reorder_materials_to_block_order(
            cell_materials, native_mesh, mesh_input
        )

    # Try meshio-style cell_data (dict of dicts)
    if cell_materials is None and hasattr(native_mesh, 'cell_data_dict'):
        cdd = native_mesh.cell_data_dict
        if array_name in cdd:
            # meshio returns per-block arrays; concatenate
            arrays = list(cdd[array_name].values())
            if arrays:
                cell_materials = np.concatenate([np.asarray(a).flatten() for a in arrays])

    if cell_materials is not None and len(cell_materials) > 0:
        mesh_input.cell_materials = cell_materials


def _reorder_materials_to_block_order(
    grid_order_materials: np.ndarray,
    native_mesh: Any,
    mesh_input: MeshInput,
) -> np.ndarray:
    """Map a grid-order per-cell array onto the volume_cells enumeration.

    Mirrors the enumeration used by the PyVista adapter's
    ``get_volume_cells``: iterate grid cells in order, keep cells whose VTK
    type has a canonical name, and group them per element type in order of
    first occurrence.  Returns the input unchanged when the mapping cannot
    be established safely (mismatched counts or unknown types).
    """
    from .cell_types import get_vtk_type_to_name

    celltypes = np.asarray(native_mesh.celltypes)
    if len(celltypes) != len(grid_order_materials):
        logger.debug(
            "celltypes length %d != materials length %d; keeping grid order",
            len(celltypes), len(grid_order_materials),
        )
        return grid_order_materials

    vtk_to_name = get_vtk_type_to_name()
    positions_by_type: Dict[str, List[int]] = {}
    for i, vtk_type in enumerate(celltypes):
        name = vtk_to_name.get(int(vtk_type))
        if name is None:
            continue  # not a recognized volumetric cell
        positions_by_type.setdefault(name, []).append(i)

    volume_cells = mesh_input.volume_cells
    reordered: List[int] = []
    for etype, connectivity in volume_cells.items():
        positions = positions_by_type.get(etype)
        if positions is None or len(positions) != len(connectivity):
            # Enumeration mismatch — grid order is the best we can offer
            logger.debug(
                "cell type %r count mismatch (%s grid vs %d canonical); "
                "keeping grid order",
                etype, len(positions) if positions else 0, len(connectivity),
            )
            return grid_order_materials
        reordered.extend(int(grid_order_materials[p]) for p in positions)

    return np.asarray(reordered)
