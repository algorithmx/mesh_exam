"""
PyMeshLab library adapter implementation.
"""

import logging
import warnings
from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING

import numpy as np

from .base_adapter import MeshAdapter
from .exceptions import QualityMetricError, FileIOError, MeshConversionError

if TYPE_CHECKING:
    import pymeshlab
    from ..core.models import MeshInput

logger = logging.getLogger(__name__)


class PyMeshLabAdapter(MeshAdapter['pymeshlab.MeshSet']):
    """
    Adapter for the PyMeshLab library.

    PyMeshLab provides Python bindings for MeshLab's mesh processing
    capabilities, including many filters for mesh repair and analysis.
    
    **Capabilities:**
    - Surface meshes: Full support with advanced filtering
    - Volumetric meshes: Surface boundary extraction only
    - File I/O: Many formats via MeshLab
    - Quality metrics: Basic surface metrics
    - Remeshing: Smoothing, decimation, subdivision
    
    **Limitations:**
    - No native volumetric cell support (tet4, hex8, etc.)
    - Volumetric meshes are processed as surface boundary only
    - Thread-safety: Not thread-safe (instance state)

    **Volumetric Mesh Handling:**
    When given a volumetric mesh (with volume_cells), this adapter extracts
    the surface boundary using PyVista and processes that surface.
    
    **Thread Safety:**
    This adapter stores internal state (_current_meshset) and is NOT
    thread-safe. Each thread should use its own adapter instance.
    
    Example:
        >>> adapter = PyMeshLabAdapter()
        >>> 
        >>> # Surface mesh
        >>> mesh = adapter.load_mesh('model.obj')
        >>> adapter.smooth(mesh, n_iterations=10)
        >>> 
        >>> # Volumetric mesh - processes surface boundary
        >>> surface = adapter.from_canonical(volumetric_mesh_input)
    """

    def __init__(self):
        self._current_meshset = None

    @property
    def capabilities(self) -> 'AdapterCapability':
        """Return capabilities supported by PyMeshLab."""
        from .capabilities import AdapterCapability
        return (
            AdapterCapability.SURFACE_MESH |
            AdapterCapability.FILE_IO |
            AdapterCapability.QUALITY_METRICS |
            AdapterCapability.TOPOLOGY_CHECKS |
            AdapterCapability.REMESHING |
            AdapterCapability.SELF_INTERSECTION
        )

    @property
    def library_name(self) -> str:
        return 'pymeshlab'

    def _check_import(self):
        """Check that pymeshlab is importable."""
        import pymeshlab
        return pymeshlab

    def _is_volumetric_file(self, file_path: str) -> bool:
        """Check if file likely contains volumetric data based on extension.
        
        Args:
            file_path: Path to mesh file
            
        Returns:
            True if file extension indicates volumetric format
        """
        volumetric_extensions = {'.vtu', '.vtk', '.cgns', '.exo', '.e', '.msh'}
        ext = file_path.lower()
        return any(ext.endswith(ve) for ve in volumetric_extensions)

    def load_mesh(self, file_path: str, **kwargs) -> 'pymeshlab.MeshSet':
        """Load mesh from file using PyMeshLab.
        
        Args:
            file_path: Path to mesh file
            **kwargs: Additional pymeshlab arguments
            
        Returns:
            pymeshlab.MeshSet object
            
        Raises:
            FileIOError: If file cannot be loaded
            
        Note:
            For volumetric files (.vtu, .vtk, etc.), a warning is issued
            that only the surface boundary will be processed.
        """
        import pymeshlab as ml
        from .exceptions import FileIOError
        
        try:
            # Check if file likely contains volumetric data
            if self._is_volumetric_file(file_path):
                warnings.warn(
                    f"File '{file_path}' appears to contain volumetric data. "
                    f"PyMeshLab will process the surface boundary only. "
                    f"For full volumetric support, use the PyVista adapter.",
                    UserWarning,
                    stacklevel=2
                )
                logger.info(
                    "Loading volumetric file '%s' - surface boundary only",
                    file_path
                )
            
            ms = ml.MeshSet()
            ms.load_new_mesh(file_path)
            self._current_meshset = ms
            return ms
            
        except Exception as e:
            raise FileIOError(
                f"PyMeshLab failed to load mesh from {file_path}: {e}"
            ) from e

    def create_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> 'pymeshlab.MeshSet':
        """Create mesh from arrays.
        
        Args:
            vertices: Nx3 array of vertex coordinates
            faces: Mx3 array of face indices
            
        Returns:
            pymeshlab.MeshSet object
        """
        import pymeshlab as ml

        mesh = ml.Mesh(
            vertex_matrix=vertices.tolist(),
            face_matrix=faces.tolist()
        )
        ms = ml.MeshSet()
        ms.add_mesh(mesh)
        self._current_meshset = ms
        return ms

    def get_vertices(self, meshset: 'pymeshlab.MeshSet') -> np.ndarray:
        """Extract vertex array."""
        mesh = meshset.current_mesh()
        return np.array(mesh.vertex_matrix())

    def get_faces(self, meshset: 'pymeshlab.MeshSet') -> np.ndarray:
        """Extract face array."""
        mesh = meshset.current_mesh()
        return np.array(mesh.face_matrix())

    def is_watertight(self, meshset: 'pymeshlab.MeshSet') -> bool:
        """Check if mesh is watertight."""
        mesh = meshset.current_mesh()
        return mesh.is_watertight()

    def is_manifold(self, meshset: 'pymeshlab.MeshSet') -> Tuple[bool, List[Dict[str, Any]]]:
        """Check if mesh is manifold using PyMeshLab filters."""
        issues = []

        # Select non-manifold edges
        meshset.select_non_manifold_edges()
        if meshset.selected_vertex_number() > 0:
            issues.append({
                'type': 'non_manifold_edges',
                'count': meshset.selected_vertex_number()
            })
        meshset.mesh_unselect_all()

        # Select degenerate faces
        meshset.select_degenerate_faces()
        if meshset.selected_face_number() > 0:
            issues.append({
                'type': 'degenerate_faces',
                'count': meshset.selected_face_number()
            })
        meshset.mesh_unselect_all()

        # Select duplicate vertices
        meshset.select_duplicate_vertices()
        if meshset.selected_vertex_number() > 0:
            issues.append({
                'type': 'duplicate_vertices',
                'count': meshset.selected_vertex_number()
            })
        meshset.mesh_unselect_all()

        return len(issues) == 0, issues

    def get_boundary_edges(self, meshset: 'pymeshlab.MeshSet') -> np.ndarray:
        """Get boundary edges.
        
        Boundary edges are edges that belong to only one face.
        
        Args:
            meshset: PyMeshLab MeshSet
            
        Returns:
            Array of boundary edge vertex pairs, shape (N, 2)
        """
        mesh = meshset.current_mesh()
        faces = mesh.face_matrix()  # (F, 3) array for triangular meshes
        
        if faces is None or len(faces) == 0:
            return np.array([]).reshape(0, 2)
        
        # Build edge-to-face count map
        edge_count = {}
        for face in faces:
            # Three edges per triangle
            edges = [
                (int(min(face[0], face[1])), int(max(face[0], face[1]))),
                (int(min(face[1], face[2])), int(max(face[1], face[2]))),
                (int(min(face[2], face[0])), int(max(face[2], face[0]))),
            ]
            for edge in edges:
                edge_count[edge] = edge_count.get(edge, 0) + 1
        
        # Boundary edges have count == 1
        boundary_edges = [list(edge) for edge, count in edge_count.items() if count == 1]
        
        if not boundary_edges:
            return np.array([]).reshape(0, 2)
        
        return np.array(boundary_edges)

    def get_nonmanifold_edges(self, meshset: 'pymeshlab.MeshSet') -> List[Tuple[int, int, int]]:
        """Detect non-manifold edges."""
        meshset.select_non_manifold_edges()
        nonmanifold_verts = meshset.selected_vertices()
        meshset.mesh_unselect_all()

        # Convert vertex selection to edge representation
        # This is a simplified representation
        return [(int(v), -1, -1) for v in nonmanifold_verts]

    def get_duplicate_vertices(
        self,
        meshset: 'pymeshlab.MeshSet',
        tolerance: float
    ) -> List[int]:
        """Find duplicate vertices.
        
        Args:
            meshset: PyMeshLab MeshSet
            tolerance: Distance threshold (note: PyMeshLab uses different units)
            
        Returns:
            List of duplicate vertex indices (approximate)
        """
        original_count = meshset.current_mesh().vertex_number()

        # Use PyMeshLab's merge close vertices
        # Note: PyMeshLab uses percentage of bounding box diagonal
        # We convert absolute tolerance to approximate percentage
        mesh = meshset.current_mesh()
        bbox = mesh.bounding_box()
        diagonal = np.linalg.norm(bbox.dim())
        
        if diagonal > 0:
            threshold_pct = (tolerance / diagonal) * 100
            meshset.merge_close_vertices(threshold=threshold_pct)
        
        cleaned_count = meshset.current_mesh().vertex_number()

        # Return indices of vertices that were merged (approximate)
        return list(range(cleaned_count, original_count))

    def compute_face_areas(self, meshset: 'pymeshlab.MeshSet') -> np.ndarray:
        """Compute area of each face."""
        # PyMeshLab can compute per-face geometric measures
        meshset.compute_geometric_measures()
        mesh = meshset.current_mesh()

        # Try to get per-face area data
        if hasattr(mesh, 'face_area_list'):
            return np.array(mesh.face_area_list())

        # If not available, compute manually
        vertices = self.get_vertices(meshset)
        faces = self.get_faces(meshset)
        n_faces = len(faces)

        areas = np.zeros(n_faces)
        for i, face in enumerate(faces):
            v = vertices[face]
            edge1 = v[1] - v[0]
            edge2 = v[2] - v[0]
            cross = np.cross(edge1, edge2)
            areas[i] = np.linalg.norm(cross) / 2

        return areas

    def compute_face_normals(self, meshset: 'pymeshlab.MeshSet') -> np.ndarray:
        """Compute face normals."""
        meshset.compute_normal_for_faces(flip=False, unseam=False)
        mesh = meshset.current_mesh()

        if hasattr(mesh, 'face_normal_list'):
            return np.array(mesh.face_normal_list())

        return np.array([])

    def detect_self_intersections(self, meshset: 'pymeshlab.MeshSet') -> List[Tuple[int, int]]:
        """Detect self-intersections using PyMeshLab filters."""
        meshset.select_self_intersecting_faces()
        intersecting = meshset.selected_face_indices()
        meshset.mesh_unselect_all()

        return [(int(i), -1) for i in intersecting]

    def compute_cell_quality(
        self,
        meshset: 'pymeshlab.MeshSet',
        quality_type: str
    ) -> np.ndarray:
        """Compute cell quality using PyMeshLab.
        
        Supported quality types:
        - 'area': Face area
        - 'aspect_ratio': Ratio of longest to shortest edge
        
        Args:
            meshset: PyMeshLab MeshSet
            quality_type: Type of quality metric
            
        Returns:
            Array of quality values
            
        Raises:
            QualityMetricError: If quality type is not supported
        """
        try:
            if quality_type == 'area':
                return self.compute_face_areas(meshset)

            # For other metrics, fallback to manual computation
            vertices = self.get_vertices(meshset)
            faces = self.get_faces(meshset)
            n_faces = len(faces)

            if quality_type == 'aspect_ratio':
                aspect_ratios = []
                for face in faces:
                    v = vertices[face]
                    edges = [
                        np.linalg.norm(v[1] - v[0]),
                        np.linalg.norm(v[2] - v[1]),
                        np.linalg.norm(v[0] - v[2])
                    ]
                    # Simple aspect ratio approximation
                    aspect_ratios.append(max(edges) / min(edges) if min(edges) > 0 else 0)
                return np.array(aspect_ratios)
            
            raise QualityMetricError(
                f"PyMeshLab does not support quality type: {quality_type}. "
                f"Supported: area, aspect_ratio"
            )
            
        except QualityMetricError:
            raise
        except Exception as e:
            raise QualityMetricError(
                f"Failed to compute {quality_type}: {e}"
            ) from e

    def from_canonical(self, mesh_input: 'MeshInput') -> 'pymeshlab.MeshSet':
        """Create PyMeshLab MeshSet from canonical format.
        
        For surface meshes (with faces), creates a MeshSet directly.
        For volumetric meshes (with volume_cells), extracts the surface
        boundary using PyVista and creates a MeshSet from that surface.
        
        Args:
            mesh_input: MeshInput in canonical format
            
        Returns:
            pymeshlab.MeshSet object (surface mesh)
            
        Note:
            Volumetric meshes are converted to their surface boundary.
            This is because PyMeshLab does not natively support volumetric
            cells. Use PyVista adapter for full volumetric support.
        """
        if mesh_input.has_volume_cells:
            logger.info(
                "Volumetric mesh detected. Extracting surface boundary "
                "for PyMeshLab (surface-focused library)."
            )
            return self._extract_surface_boundary(mesh_input)
        
        return self.create_mesh(mesh_input.vertices, mesh_input.faces)

    def _extract_surface_boundary(self, mesh_input: 'MeshInput') -> 'pymeshlab.MeshSet':
        """Extract surface boundary from volumetric mesh.
        
        Uses PyVista to extract the surface boundary of a volumetric mesh,
        then creates a MeshSet from that surface.
        
        Args:
            mesh_input: MeshInput with volume_cells
            
        Returns:
            pymeshlab.MeshSet of the surface boundary
            
        Raises:
            MeshConversionError: If surface extraction fails
        """
        import pymeshlab as ml
        
        try:
            # Use PyVista to get volumetric representation
            grid = mesh_input.get_native('pyvista')
            
            # Extract surface
            surface = grid.extract_surface(algorithm='dataset_surface')
            
            # Get vertices and faces
            vertices = surface.points
            faces_raw = surface.faces
            
            # Convert VTK face format to simple face array
            face_list = []
            if len(faces_raw) > 0:
                idx = 0
                while idx < len(faces_raw):
                    n_verts = int(faces_raw[idx])
                    if n_verts <= 0 or idx + 1 + n_verts > len(faces_raw):
                        break
                    idx += 1
                    face_list.append(faces_raw[idx:idx + n_verts].tolist())
                    idx += n_verts
            
            if not face_list:
                raise ValueError("No faces extracted from volumetric mesh")
            
            # Create MeshSet from surface
            mesh = ml.Mesh(
                vertex_matrix=vertices.tolist(),
                face_matrix=face_list
            )
            ms = ml.MeshSet()
            ms.add_mesh(mesh)
            self._current_meshset = ms
            return ms
            
        except Exception as e:
            raise MeshConversionError(
                f"Failed to extract surface boundary from volumetric mesh: {e}. "
                f"Ensure pyvista is installed for volumetric mesh support."
            ) from e

    # ======================================================================
    # PyMeshLab-specific remeshing methods (not part of base interface)
    # ======================================================================

    def smooth(
        self,
        meshset: 'pymeshlab.MeshSet',
        n_iterations: int = 3,
        method: str = 'laplacian'
    ) -> 'pymeshlab.MeshSet':
        """Smooth mesh using Laplacian smoothing.
        
        Args:
            meshset: MeshSet to smooth
            n_iterations: Number of smoothing iterations
            method: Smoothing method ('laplacian' or 'taubin')
            
        Returns:
            Smoothed MeshSet (modifies in place)
        """
        if method == 'laplacian':
            meshset.laplacian_smooth(stepsmoothnum=n_iterations)
        elif method == 'taubin':
            meshset.taubin_smooth(stepsmoothnum=n_iterations)
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
        return meshset

    def decimate(
        self,
        meshset: 'pymeshlab.MeshSet',
        target_reduction: float = 0.5
    ) -> 'pymeshlab.MeshSet':
        """Simplify mesh by reducing face count.
        
        Args:
            meshset: MeshSet to decimate
            target_reduction: Target reduction factor (0-1)
            
        Returns:
            Decimated MeshSet (modifies in place)
        """
        meshset.simplification_quadric_edge_collapse_decimation(
            targetperc=target_reduction
        )
        return meshset

    def repair(self, meshset: 'pymeshlab.MeshSet') -> 'pymeshlab.MeshSet':
        """Apply various mesh repair filters.
        
        This applies a standard repair sequence:
        1. Remove duplicate vertices
        2. Remove unreferenced vertices
        3. Remove duplicate faces
        4. Remove zero-area faces
        
        Args:
            meshset: MeshSet to repair
            
        Returns:
            Repaired MeshSet (modifies in place)
        """
        meshset.merge_close_vertices()
        meshset.remove_unreferenced_vertices()
        meshset.remove_duplicate_faces()
        meshset.remove_zero_area_faces()
        return meshset
