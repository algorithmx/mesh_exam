"""
PyVista library adapter implementation.

PyVista is a Python wrapper for VTK (Visualization Toolkit), providing
a streamlined interface for mesh processing, analysis, and visualization.
This adapter provides full support for both surface (PolyData) and
volumetric (UnstructuredGrid) meshes.
"""

import logging
from typing import Any, Dict, List, Tuple, Optional, Union

import numpy as np

from .base_adapter import MeshAdapter
from .exceptions import QualityMetricError, FileIOError, MeshConversionError

logger = logging.getLogger(__name__)


class PyVistaAdapter(MeshAdapter):
    """
    Adapter for the PyVista library (VTK wrapper).

    PyVista provides a streamlined interface to VTK for mesh processing,
    analysis, and visualization. This adapter supports both surface meshes
    (triangular/polygons as PolyData) and volumetric meshes (tet/hex/prism
    as UnstructuredGrid).

    **Capabilities:**
    - **Surface meshes** (PolyData): Full support for triangular and polygonal faces
    - **Volumetric meshes** (UnstructuredGrid): Full support for tetrahedra,
      hexahedra, prisms, and pyramids
    - **File I/O**: All VTK formats (.vtk, .vtu, .vtp, .stl, .ply, etc.)
    - **Quality metrics**: All VTK cell quality measures (see below)
    - **Topology checks**: Manifold detection, watertight checking
    - **Self-intersection**: Detection via VTK filters

    **Supported Quality Metrics:**
    - ``area``: Face/cell area
    - ``aspect_ratio``: Longest/shortest edge ratio
    - ``scaled_jacobian``: Scaled Jacobian determinant
    - ``max_angle``: Maximum interior angle
    - ``min_angle``: Minimum interior angle
    - ``condition``: Condition number
    - ``jacobian``: Jacobian determinant
    - ``volume``: Cell volume (volumetric only)
    - ``distortion``: Distortion metric
    - ``shear``: Shear metric
    - ``shape``: Shape metric
    - ``radius_ratio``: Radius ratio
    - ``warpage``: Face warpage (quadrilateral faces)
    - ``stretch``: Stretch metric

    **Usage Examples:**

    Surface mesh processing::

        >>> adapter = PyVistaAdapter()
        >>> mesh = adapter.load_mesh('surface.stl')
        >>> areas = adapter.compute_face_areas(mesh)
        >>> print(f"Total surface area: {areas.sum()}")

    Volumetric mesh processing::

        >>> mesh = adapter.load_mesh('volume.vtu')
        >>> volumes = adapter.compute_cell_quality(mesh, 'volume')
        >>> print(f"Total volume: {volumes.sum()}")

    Creating from canonical format::

        >>> from mesh_issue_examiner.core.models import MeshInput
        >>> mesh_input = MeshInput(vertices=verts, faces=faces)
        >>> pv_mesh = adapter.from_canonical(mesh_input)

    **Notes:**
    - PyVista is the most capable adapter in this framework
    - For volumetric meshes, use PyVista for full functionality
    - Other adapters may delegate volumetric operations to PyVista
    """

    @property
    def capabilities(self) -> 'AdapterCapability':
        """Return capabilities supported by PyVista.
        
        Returns:
            AdapterCapability flags indicating all supported features
        """
        from .capabilities import AdapterCapability
        return (
            AdapterCapability.SURFACE_MESH |
            AdapterCapability.VOLUMETRIC_MESH |
            AdapterCapability.FILE_IO |
            AdapterCapability.QUALITY_METRICS |
            AdapterCapability.TOPOLOGY_CHECKS |
            AdapterCapability.SELF_INTERSECTION
        )

    @property
    def library_name(self) -> str:
        """Return library name."""
        return 'pyvista'

    def _check_import(self):
        """Check that pyvista is importable."""
        import pyvista
        return pyvista

    def load_mesh(self, file_path: str, **kwargs) -> 'pyvista.PolyData':
        """Load mesh from file using PyVista.
        
        Supports all VTK-based formats including:
        - .vtk, .vtu, .vtp (VTK formats)
        - .stl (STereoLithography)
        - .ply (Polygon File Format)
        - .obj (Wavefront)
        - .ply (Stanford Polygon)
        
        Args:
            file_path: Path to mesh file
            **kwargs: Additional pyvista.read arguments
            
        Returns:
            PyVista mesh object (PolyData or UnstructuredGrid)
            
        Raises:
            FileIOError: If file cannot be loaded
        """
        import pyvista as pv
        from .exceptions import FileIOError
        
        try:
            return pv.read(file_path, **kwargs)
        except Exception as e:
            raise FileIOError(
                f"Failed to load mesh from {file_path} using pyvista: {e}"
            ) from e

    def create_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> 'pyvista.PolyData':
        """Create PolyData from vertex and face arrays.
        
        Args:
            vertices: Nx3 array of vertex coordinates
            faces: Mx3 array of face indices (triangles)
            
        Returns:
            pyvista.PolyData mesh object
        """
        import pyvista as pv

        n_faces = len(faces)

        # Create face array with vertex count prefix
        # VTK format: [n_verts, v0, v1, v2, n_verts, v0, v1, v2, ...]
        face_arr = np.column_stack([
            np.full(n_faces, 3, dtype=np.int32),
            faces
        ]).ravel()

        return pv.PolyData(vertices, face_arr)

    def from_canonical(self, mesh_input: 'MeshInput') -> Any:
        """Create PyVista mesh from canonical format.

        Supports both surface (PolyData) and volumetric (UnstructuredGrid) meshes.
        
        For volumetric meshes, converts volume_cells to VTK UnstructuredGrid
        format using the cell type mapping.
        
        Args:
            mesh_input: MeshInput in canonical format
            
        Returns:
            PolyData for surface meshes, UnstructuredGrid for volumetric
            
        Raises:
            MeshConversionError: If conversion fails (e.g., unknown element type)
        """
        import pyvista as pv
        from ..utils.cell_types import get_cell_type_map

        if mesh_input.has_volume_cells:
            try:
                cell_type_map = get_cell_type_map()
                all_cells = []
                all_celltypes = []
                
                for etype, connectivity in mesh_input.volume_cells.items():
                    vtk_type = cell_type_map.get(etype)
                    if vtk_type is None:
                        raise MeshConversionError(
                            f"Unknown element type '{etype}'. "
                            f"Supported: {list(cell_type_map.keys())}"
                        )
                    
                    for row in connectivity:
                        all_cells.append(len(row))
                        all_cells.extend(row.tolist())
                        all_celltypes.append(vtk_type)
                
                cells_arr = np.array(all_cells, dtype=np.int64)
                celltypes_arr = np.array(all_celltypes, dtype=np.uint8)
                
                return pv.UnstructuredGrid(
                    cells_arr, celltypes_arr, mesh_input.vertices
                )
            except MeshConversionError:
                raise
            except Exception as e:
                raise MeshConversionError(
                    f"Failed to create UnstructuredGrid from volumetric mesh: {e}"
                ) from e

        # Surface mesh
        n_faces = len(mesh_input.faces)
        if n_faces == 0:
            return pv.PolyData(mesh_input.vertices)
        
        face_arr = np.column_stack([
            np.full(n_faces, 3, dtype=np.int32),
            mesh_input.faces
        ]).ravel()
        
        return pv.PolyData(mesh_input.vertices, face_arr)

    def get_vertices(self, mesh: 'pyvista.PolyData') -> np.ndarray:
        """Extract vertex array.
        
        Args:
            mesh: PyVista mesh object
            
        Returns:
            Nx3 array of vertex coordinates
        """
        return np.array(mesh.points)

    # Maximum reasonable vertices per face to prevent issues with malformed data
    _MAX_VERTS_PER_FACE = 1000

    def get_faces(self, mesh: 'pyvista.PolyData') -> Union[np.ndarray, List[List[int]]]:
        """Extract face array (removing vertex count prefix).
        
        Handles mixed face types (triangles, quads, polygons) by returning
        a list of lists when face sizes are inconsistent.
        
        Args:
            mesh: PyVista mesh object
            
        Returns:
            - Mx3 array for triangular meshes
            - Mx4 array for quad meshes  
            - List of lists for mixed face types
            - Empty array if mesh has no faces
            
        Note:
            Returns inconsistent types based on face uniformity. Callers should
            check return type with ``isinstance(result, list)`` to handle both cases.
            For consistent array output, use ``get_faces_as_array()`` instead.
        """
        faces = mesh.faces
        if faces is None or len(faces) == 0:
            return np.array([])

        # Parse faces from PyVista's face array format
        face_list = []
        idx = 0
        while idx < len(faces):
            n_verts = int(faces[idx])
            
            # Validate vertex count with specific error messages for debugging
            if n_verts < 0:
                logger.warning(f"Negative vertex count at index {idx}: {n_verts}")
                break
            elif n_verts == 0:
                logger.warning(f"Zero vertex count at index {idx}")
                break
            elif n_verts > self._MAX_VERTS_PER_FACE:
                logger.warning(
                    f"Unreasonably large vertex count at index {idx}: {n_verts} "
                    f"(max allowed: {self._MAX_VERTS_PER_FACE})"
                )
                break
            elif idx + 1 + n_verts > len(faces):
                logger.warning(
                    f"Truncated face data at index {idx}: need {n_verts} vertices "
                    f"but only {len(faces) - idx - 1} available"
                )
                break
            
            idx += 1
            face_list.append(list(faces[idx:idx + n_verts]))
            idx += n_verts

        if not face_list:
            return np.array([])

        # Check if all faces have the same number of vertices
        face_sizes = set(len(f) for f in face_list)
        
        if len(face_sizes) == 1:
            # Uniform face size - return as 2D array
            try:
                return np.array(face_list)
            except ValueError:
                # Fallback to list if numpy conversion fails
                return face_list
        else:
            # Mixed face sizes - return as list of lists
            logger.debug(f"Mixed face sizes detected: {face_sizes}")
            return face_list

    def is_watertight(self, mesh: 'pyvista.PolyData') -> bool:
        """Check if mesh is watertight.
        
        A mesh is watertight if it has no boundary (open) edges.
        
        Args:
            mesh: PyVista mesh object
            
        Returns:
            True if mesh has no open edges
        """
        # A mesh is watertight if it has no open edges
        return mesh.n_open_edges == 0

    def is_manifold(self, mesh: 'pyvista.PolyData') -> Tuple[bool, List[Dict[str, Any]]]:
        """Check if mesh is manifold using VTK filters.
        
        Args:
            mesh: PyVista mesh object
            
        Returns:
            Tuple of (is_manifold, list of issues)
        """
        issues = []

        # Check if mesh is manifold
        if not mesh.is_manifold:
            issues.append({
                'type': 'non_manifold',
                'message': 'Mesh contains non-manifold geometry'
            })

        # Check for degenerate cells (zero-area faces)
        try:
            quality = mesh.compute_cell_quality(quality_measure='volume')
            volumes = quality.cell_data.get('CellQualityVolume', [])
            degenerate_indices = np.where(volumes == 0)[0]
            if len(degenerate_indices) > 0:
                issues.append({
                    'type': 'degenerate_cells',
                    'count': len(degenerate_indices),
                    'indices': degenerate_indices.tolist()
                })
        except Exception as e:
            logger.debug(f"Could not compute cell quality for degenerate check: {e}")

        return len(issues) == 0, issues

    def get_boundary_edges(self, mesh: 'pyvista.PolyData') -> np.ndarray:
        """Get boundary edges.
        
        Boundary edges are edges that belong to only one face.
        
        Args:
            mesh: PyVista mesh object
            
        Returns:
            Array of boundary edge vertex pairs
        """
        try:
            # Extract feature edges (boundary edges)
            boundary = mesh.extract_feature_edges(
                boundary_edges=True, 
                feature_edges=False,
                non_manifold_edges=False
            )
            if boundary.n_cells > 0:
                return boundary.lines
        except Exception as e:
            logger.debug(f"Could not extract boundary edges: {e}")
        return np.array([])

    def get_nonmanifold_edges(self, mesh: 'pyvista.PolyData') -> List[Tuple[int, int, int]]:
        """Detect non-manifold edges.
        
        Non-manifold edges are edges shared by more than 2 faces.
        
        Args:
            mesh: PyVista mesh object
            
        Returns:
            List of (vertex1, vertex2, incident_face_count) tuples
        """
        faces = self.get_faces(mesh)
        
        # Handle list return type for mixed faces
        if isinstance(faces, list):
            n_faces = len(faces)
        else:
            n_faces = len(faces)
            if n_faces == 0:
                return []

        # Build edge-to-face adjacency
        edges = {}
        for face_idx in range(n_faces):
            if isinstance(faces, list):
                face = faces[face_idx]
            else:
                face = faces[face_idx]
            
            n_verts = len(face)
            for i in range(n_verts):
                v1, v2 = int(face[i]), int(face[(i + 1) % n_verts])
                edge = tuple(sorted([v1, v2]))
                if edge not in edges:
                    edges[edge] = []
                edges[edge].append(face_idx)

        # Find non-manifold edges
        nonmanifold = [
            (edge[0], edge[1], len(face_list))
            for edge, face_list in edges.items()
            if len(face_list) != 2
        ]

        return nonmanifold

    def get_duplicate_vertices(
        self,
        mesh: 'pyvista.PolyData',
        tolerance: float
    ) -> List[int]:
        """Find duplicate vertices.
        
        Uses VTK's duplicate point removal filter.
        
        Args:
            mesh: PyVista mesh object
            tolerance: Distance threshold for considering vertices duplicate
            
        Returns:
            List of duplicate vertex indices
        """
        try:
            # Use VTK's duplicate point removal
            cleaned = mesh.clean(tolerance=tolerance, merge_tol=tolerance)
            original_count = mesh.n_points
            cleaned_count = cleaned.n_points

            # Return indices of removed vertices
            return list(range(cleaned_count, original_count))
        except Exception as e:
            logger.debug(f"Could not find duplicate vertices: {e}")
            return []

    def compute_face_areas(self, mesh: 'pyvista.PolyData') -> np.ndarray:
        """Compute area of each face.
        
        Uses VTK's cell quality filter if available, falls back to
        manual computation.
        
        Args:
            mesh: PyVista mesh object
            
        Returns:
            Array of face areas
        """
        try:
            quality = mesh.compute_cell_quality(quality_measure='area')
            return quality.cell_data.get('CellQualityArea', np.array([]))
        except Exception as e:
            logger.debug(f"VTK area computation failed, using manual: {e}")
            # Fallback to manual computation
            return self._compute_areas_manual(mesh)

    def _compute_areas_manual(self, mesh: 'pyvista.PolyData') -> np.ndarray:
        """Manually compute face areas.
        
        Args:
            mesh: PyVista mesh object
            
        Returns:
            Array of face areas
        """
        faces = self.get_faces(mesh)
        vertices = self.get_vertices(mesh)
        
        # Handle list return type
        if isinstance(faces, list):
            n_faces = len(faces)
        else:
            n_faces = len(faces)
        
        if n_faces == 0:
            return np.array([])

        areas = np.zeros(n_faces)
        for i in range(n_faces):
            if isinstance(faces, list):
                face = faces[i]
            else:
                face = faces[i]
            
            v = vertices[face]
            if len(v) >= 3:
                edge1 = v[1] - v[0]
                edge2 = v[2] - v[0]
                cross = np.cross(edge1, edge2)
                areas[i] = np.linalg.norm(cross) / 2

        return areas

    def compute_face_normals(self, mesh: 'pyvista.PolyData') -> np.ndarray:
        """Compute face normals.
        
        Uses VTK's normal computation if available, falls back to
        manual computation.
        
        Args:
            mesh: PyVista mesh object
            
        Returns:
            Nx3 array of face normal vectors
        """
        try:
            mesh_copy = mesh.copy()
            mesh_copy.compute_normals(cell_oriented=True, inplace=True)
            normals = mesh_copy.cell_data.get('Normals', np.array([]))
            if len(normals) == 0:
                return self._compute_normals_manual(mesh)
            return normals
        except Exception as e:
            logger.debug(f"VTK normal computation failed, using manual: {e}")
            return self._compute_normals_manual(mesh)

    def _compute_normals_manual(self, mesh: 'pyvista.PolyData') -> np.ndarray:
        """Manually compute face normals."""
        faces = self.get_faces(mesh)
        vertices = self.get_vertices(mesh)
        
        if isinstance(faces, list):
            n_faces = len(faces)
        else:
            n_faces = len(faces)
        
        if n_faces == 0:
            return np.array([])

        normals = np.zeros((n_faces, 3))
        for i in range(n_faces):
            if isinstance(faces, list):
                face = faces[i]
            else:
                face = faces[i]
            
            v = vertices[face]
            if len(v) >= 3:
                edge1 = v[1] - v[0]
                edge2 = v[2] - v[0]
                cross = np.cross(edge1, edge2)
                norm = np.linalg.norm(cross)
                if norm > 0:
                    normals[i] = cross / norm

        return normals

    def detect_self_intersections(self, mesh: 'pyvista.PolyData') -> List[Tuple[int, int]]:
        """Detect self-intersections using VTK.
        
        Args:
            mesh: PyVista mesh object
            
        Returns:
            List of (face1_index, face2_index) tuples
        """
        try:
            # Try to use mesh intersection check
            if hasattr(mesh, 'check_intersection'):
                intersections = mesh.check_intersection()
                return intersections
        except Exception as e:
            logger.debug(f"Could not detect self-intersections: {e}")
        return []

    def compute_cell_quality(
        self,
        mesh: 'pyvista.PolyData',
        quality_type: str
    ) -> np.ndarray:
        """Compute cell quality metrics using VTK's cell quality filter.

        Works on both PolyData (surface) and UnstructuredGrid (volumetric) meshes.

        Supported quality types:
        - ``area``: Cell area (surface) or face area (volumetric)
        - ``aspect_ratio``: Aspect ratio
        - ``skewness``: Maximum angle skewness
        - ``scaled_jacobian``: Scaled Jacobian determinant
        - ``max_angle``: Maximum angle
        - ``min_angle``: Minimum angle
        - ``condition``: Condition number
        - ``jacobian``: Jacobian determinant
        - ``volume``: Cell volume (volumetric only)
        - ``distortion``: Distortion metric
        - ``shear``: Shear metric
        - ``shape``: Shape metric
        - ``radius_ratio``: Radius ratio
        - ``warpage``: Warpage (hex faces)
        - ``stretch``: Stretch metric

        Args:
            mesh: PyVista mesh object
            quality_type: Name of quality metric to compute
            
        Returns:
            Array of quality values for each cell
            
        Raises:
            QualityMetricError: If quality computation fails
        """
        import pyvista as pv
        
        # Map quality type names to VTK/pyvista names
        quality_map = {
            'area': 'area',
            'aspect_ratio': 'aspect_ratio',
            'skewness': 'skew',
            'max_angle': 'max_angle',
            'min_angle': 'min_angle',
            'condition': 'condition',
            'Jacobian': 'jacobian',
            'volume': 'volume',
            'scaled_jacobian': 'scaled_jacobian',
            'distortion': 'distortion',
            'shear': 'shear',
            'shape': 'shape',
            'radius_ratio': 'radius_ratio',
            'warpage': 'warpage',
            'stretch': 'stretch',
        }

        vtk_quality = quality_map.get(quality_type, quality_type)

        try:
            # Use the modern cell_quality API
            quality_mesh = mesh.cell_quality(quality_measure=vtk_quality)
            values = quality_mesh.cell_data.get(vtk_quality, np.array([]))
            if len(values) > 0:
                return values
            return np.array([])
            
        except Exception as primary_error:
            logger.debug(f"Primary quality computation failed: {primary_error}")
            
            try:
                # Fallback to deprecated compute_cell_quality for older pyvista
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    quality_mesh = mesh.compute_cell_quality(quality_measure=vtk_quality)
                    for key in ['CellQuality', vtk_quality]:
                        values = quality_mesh.cell_data.get(key, np.array([]))
                        if len(values) > 0:
                            return values
                return np.array([])
                
            except Exception as fallback_error:
                # Last resort: manual computation for basic metrics
                if quality_type == 'area':
                    return self._compute_areas_manual(mesh)
                elif quality_type == 'volume':
                    # Try to compute volume manually
                    try:
                        return np.array([mesh.volume])
                    except Exception:
                        return np.array([])
                
                raise QualityMetricError(
                    f"Failed to compute quality metric '{quality_type}': {primary_error}"
                ) from primary_error
