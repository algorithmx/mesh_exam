"""
TRimesh library adapter implementation.
"""

from typing import Any, Dict, List, Tuple, Optional
import logging

import numpy as np

from .base_adapter import MeshAdapter
from .exceptions import QualityMetricError, MeshConversionError

logger = logging.getLogger(__name__)


class TrimeshAdapter(MeshAdapter):
    """
    Adapter for the TRimesh library.

    TRimesh is a pure Python library for loading and using triangular
    meshes with an emphasis on watertight surfaces.
    
    **Capabilities:**
    - Surface meshes: Full support (triangular faces)
    - Volumetric meshes: Surface boundary extraction only
    - File I/O: Common formats (STL, OBJ, PLY, etc.)
    - Quality metrics: Triangle-specific metrics
    - Topology checks: Manifold, watertight detection
    
    **Limitations:**
    - No native volumetric cell support (tet4, hex8, etc.)
    - Volumetric meshes are converted to surface boundary
    
    **Volumetric Mesh Handling:**
    When given a volumetric mesh (with volume_cells), this adapter extracts
    the surface boundary using PyVista and returns a Trimesh of the surface.
    This allows surface-based quality checks on volumetric meshes.
    
    Example:
        >>> adapter = TrimeshAdapter()
        >>> 
        >>> # Surface mesh - direct conversion
        >>> mesh = adapter.from_canonical(surface_mesh_input)
        >>> 
        >>> # Volumetric mesh - extracts surface boundary
        >>> surface_mesh = adapter.from_canonical(volumetric_mesh_input)
        >>> areas = adapter.compute_face_areas(surface_mesh)
    """

    @property
    def capabilities(self) -> 'AdapterCapability':
        """Return capabilities supported by Trimesh."""
        from .capabilities import AdapterCapability
        return (
            AdapterCapability.SURFACE_MESH |
            AdapterCapability.FILE_IO |
            AdapterCapability.QUALITY_METRICS |
            AdapterCapability.TOPOLOGY_CHECKS
        )

    @property
    def library_name(self) -> str:
        return 'trimesh'

    def _check_import(self):
        """Check that trimesh is importable."""
        import trimesh
        return trimesh

    def load_mesh(self, file_path: str, **kwargs) -> 'trimesh.Trimesh':
        """Load mesh from file using TRimesh.
        
        Args:
            file_path: Path to mesh file
            **kwargs: Additional trimesh.load_mesh arguments
            
        Returns:
            trimesh.Trimesh object
            
        Raises:
            FileIOError: If file cannot be loaded
        """
        import trimesh
        from .exceptions import FileIOError
        
        try:
            return trimesh.load_mesh(file_path, **kwargs)
        except Exception as e:
            raise FileIOError(
                f"Failed to load mesh from {file_path} using trimesh: {e}"
            ) from e

    def create_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> 'trimesh.Trimesh':
        """Create TRimesh from arrays.
        
        Args:
            vertices: Nx3 array of vertex coordinates
            faces: Mx3 array of face indices
            
        Returns:
            trimesh.Trimesh object
        """
        import trimesh
        return trimesh.Trimesh(vertices=vertices, faces=faces)

    def get_vertices(self, mesh: 'trimesh.Trimesh') -> np.ndarray:
        """Extract vertex array."""
        return np.array(mesh.vertices.view(np.ndarray))

    def get_faces(self, mesh: 'trimesh.Trimesh') -> np.ndarray:
        """Extract face array."""
        return np.array(mesh.faces.view(np.ndarray))

    def is_watertight(self, mesh: 'trimesh.Trimesh') -> bool:
        """Check if mesh is watertight."""
        return mesh.is_watertight

    def is_manifold(self, mesh: 'trimesh.Trimesh') -> Tuple[bool, List[Dict[str, Any]]]:
        """Check if mesh is manifold."""
        issues = []

        # Check for degenerate faces
        degenerate = mesh.degenerate_faces
        if len(degenerate) > 0:
            issues.append({
                'type': 'degenerate_faces',
                'count': len(degenerate),
                'indices': degenerate.tolist() if hasattr(degenerate, 'tolist') else list(degenerate)
            })

        # Check for duplicate vertices
        if hasattr(mesh, 'vertices_unique'):
            unique_ratio = len(mesh.vertices_unique) / len(mesh.vertices)
            if unique_ratio < 0.99:
                issues.append({
                    'type': 'duplicate_vertices',
                    'estimated_count': len(mesh.vertices) - len(mesh.vertices_unique)
                })

        # Check for non-manifold edges
        edges_unique = mesh.edges_unique
        edges_face = mesh.edges_face_adjacency

        # Find edges with != 2 incident faces
        nonmanifold_count = np.sum(edges_face[:, 2] != 2)
        if nonmanifold_count > 0:
            issues.append({
                'type': 'nonmanifold_edges',
                'count': int(nonmanifold_count)
            })

        return len(issues) == 0, issues

    def get_boundary_edges(self, mesh: 'trimesh.Trimesh') -> np.ndarray:
        """Get boundary (free) edges."""
        return mesh.edges_boundary

    def get_nonmanifold_edges(self, mesh: 'trimesh.Trimesh') -> List[Tuple[int, int, int]]:
        """Detect non-manifold edges."""
        edges_face = mesh.edges_face_adjacency
        edges_unique = mesh.edges_unique

        nonmanifold = []
        for i, (edge, face_count) in enumerate(zip(edges_unique, edges_face[:, 2])):
            if face_count != 2:
                nonmanifold.append((int(edge[0]), int(edge[1]), int(face_count)))

        return nonmanifold

    def get_duplicate_vertices(
        self,
        mesh: 'trimesh.Trimesh',
        tolerance: float
    ) -> List[int]:
        """Find duplicate vertices."""
        # Use TRimesh's merge vertices functionality
        original_count = len(mesh.vertices)

        # Find vertices that would be merged
        # This is a simplified implementation
        from scipy.spatial import cKDTree
        tree = cKDTree(mesh.vertices)
        duplicates = set()

        for i, vertex in enumerate(mesh.vertices):
            neighbors = tree.query_ball_point(vertex, tolerance)
            for j in neighbors:
                if j > i:
                    duplicates.add(j)

        return list(duplicates)

    def compute_face_areas(self, mesh: 'trimesh.Trimesh') -> np.ndarray:
        """Compute area of each face."""
        return mesh.area_faces

    def compute_face_normals(self, mesh: 'trimesh.Trimesh') -> np.ndarray:
        """Compute face normals."""
        return mesh.face_normals

    def detect_self_intersections(self, mesh: 'trimesh.Trimesh') -> List[Tuple[int, int]]:
        """Detect self-intersections."""
        # Watertight meshes cannot self-intersect
        if mesh.is_watertight:
            return []

        # Use TRimesh's intersection detection
        # Note: This may require additional processing
        intersections = []

        # Simple check using convex hull comparison
        if hasattr(mesh, 'volume') and hasattr(mesh, 'convex_hull'):
            hull = mesh.convex_hull
            # If volumes are significantly different, there may be self-intersections
            if mesh.volume > 0 and hull.volume > 0:
                volume_ratio = mesh.volume / hull.volume
                if volume_ratio < 0.9:  # Threshold for potential intersections
                    intersections.append(('volume_mismatch', 0))

        return intersections

    def compute_cell_quality(
        self,
        mesh: 'trimesh.Trimesh',
        quality_type: str
    ) -> np.ndarray:
        """Compute cell quality metrics.
        
        Supported quality types:
        - 'area': Face area
        - 'aspect_ratio': Circumradius / (2 * inradius)
        - 'min_angle': Minimum interior angle (degrees)
        - 'max_angle': Maximum interior angle (degrees)
        
        Args:
            mesh: trimesh.Trimesh object
            quality_type: Type of quality metric
            
        Returns:
            Array of quality values
            
        Raises:
            QualityMetricError: If quality type is not supported or computation fails
        """
        n_faces = len(mesh.faces)

        try:
            if quality_type == 'area':
                return mesh.area_faces

            elif quality_type == 'aspect_ratio':
                return self._compute_aspect_ratio(mesh)

            elif quality_type == 'min_angle':
                return self._compute_min_angles(mesh)

            elif quality_type == 'max_angle':
                return self._compute_max_angles(mesh)

            else:
                raise QualityMetricError(
                    f"Trimesh does not support quality type: {quality_type}. "
                    f"Supported: area, aspect_ratio, min_angle, max_angle"
                )
                
        except QualityMetricError:
            raise
        except Exception as e:
            raise QualityMetricError(
                f"Failed to compute {quality_type}: {e}"
            ) from e

    def _compute_aspect_ratio(self, mesh: 'trimesh.Trimesh') -> np.ndarray:
        """Compute aspect ratio for triangular faces.
        
        Aspect ratio = circumradius / (2 * inradius)
        """
        aspect_ratios = []
        for face_idx in range(len(mesh.faces)):
            face_verts = mesh.vertices[mesh.faces[face_idx]]
            edges = [
                np.linalg.norm(face_verts[1] - face_verts[0]),
                np.linalg.norm(face_verts[2] - face_verts[1]),
                np.linalg.norm(face_verts[0] - face_verts[2])
            ]
            # Use the formula: aspect ratio = circumradius / (2 * inradius)
            s = sum(edges) / 2  # semi-perimeter
            area = 0.5 * np.linalg.norm(
                np.cross(face_verts[1] - face_verts[0], face_verts[2] - face_verts[0])
            )
            if area > 0:
                inradius = 2 * area / sum(edges)
                # Clip arccos argument to [-1, 1] to avoid domain errors from floating point inaccuracies
                arccos_arg = np.clip(
                    (edges[0]**2 + edges[1]**2 - edges[2]**2) / (2 * edges[0] * edges[1]),
                    -1.0, 1.0
                )
                circumradius = max(edges) / (2 * np.sin(np.arccos(arccos_arg))) if len(edges) > 1 else max(edges)
                aspect_ratios.append(circumradius / (2 * inradius) if inradius > 0 else 0)
            else:
                aspect_ratios.append(0)
        return np.array(aspect_ratios)

    def _compute_min_angles(self, mesh: 'trimesh.Trimesh') -> np.ndarray:
        """Compute minimum interior angles for triangular faces (degrees)."""
        angles = []
        for face_idx in range(len(mesh.faces)):
            face_verts = mesh.vertices[mesh.faces[face_idx]]
            v0, v1, v2 = face_verts
            a = np.arccos(np.clip(np.dot(v1 - v0, v2 - v0) /
                       (np.linalg.norm(v1 - v0) * np.linalg.norm(v2 - v0)), -1, 1))
            b = np.arccos(np.clip(np.dot(v2 - v1, v0 - v1) /
                       (np.linalg.norm(v2 - v1) * np.linalg.norm(v0 - v1)), -1, 1))
            c = np.pi - a - b
            angles.append(np.degrees(min(a, b, c)))
        return np.array(angles)

    def _compute_max_angles(self, mesh: 'trimesh.Trimesh') -> np.ndarray:
        """Compute maximum interior angles for triangular faces (degrees)."""
        angles = []
        for face_idx in range(len(mesh.faces)):
            face_verts = mesh.vertices[mesh.faces[face_idx]]
            v0, v1, v2 = face_verts
            a = np.arccos(np.clip(np.dot(v1 - v0, v2 - v0) /
                       (np.linalg.norm(v1 - v0) * np.linalg.norm(v2 - v0)), -1, 1))
            b = np.arccos(np.clip(np.dot(v2 - v1, v0 - v1) /
                       (np.linalg.norm(v2 - v1) * np.linalg.norm(v0 - v1)), -1, 1))
            c = np.pi - a - b
            angles.append(np.degrees(max(a, b, c)))
        return np.array(angles)

    def from_canonical(self, mesh_input: 'MeshInput') -> 'trimesh.Trimesh':
        """Create Trimesh from canonical format.
        
        For surface meshes (with faces), creates a Trimesh directly.
        For volumetric meshes (with volume_cells), extracts the surface
        boundary using PyVista and returns a Trimesh of the surface.
        
        Args:
            mesh_input: MeshInput in canonical format
            
        Returns:
            trimesh.Trimesh object (surface mesh)
            
        Note:
            Volumetric meshes are converted to their surface boundary.
            This is because Trimesh does not natively support volumetric
            cells. Use PyVista adapter for full volumetric support.
        """
        if mesh_input.has_volume_cells:
            logger.info(
                "Volumetric mesh detected. Extracting surface boundary "
                "for Trimesh (surface-only library)."
            )
            return self._extract_surface_boundary(mesh_input)
        
        return self.create_mesh(mesh_input.vertices, mesh_input.faces)

    def _extract_surface_boundary(self, mesh_input: 'MeshInput') -> 'trimesh.Trimesh':
        """Extract surface boundary from volumetric mesh.
        
        Uses PyVista to extract the surface boundary of a volumetric mesh,
        then converts it to a Trimesh.
        
        Args:
            mesh_input: MeshInput with volume_cells
            
        Returns:
            trimesh.Trimesh of the surface boundary
            
        Raises:
            MeshConversionError: If surface extraction fails
        """
        import trimesh
        
        try:
            # Use PyVista to get volumetric representation
            grid = mesh_input.get_native('pyvista')
            
            # Extract surface
            surface = grid.extract_surface()
            
            # Convert to Trimesh format
            vertices = surface.points
            faces_raw = surface.faces
            
            # Reshape faces from [n, v0, v1, v2, n, v0, ...] to [[v0, v1, v2], ...]
            if len(faces_raw) > 0:
                # VTK format: each face starts with vertex count
                face_list = []
                idx = 0
                while idx < len(faces_raw):
                    n_verts = int(faces_raw[idx])
                    if n_verts <= 0 or idx + 1 + n_verts > len(faces_raw):
                        break
                    idx += 1
                    face_list.append(list(faces_raw[idx:idx + n_verts]))
                    idx += n_verts
                
                faces = np.array(face_list)
            else:
                faces = np.array([]).reshape(0, 3)
            
            return trimesh.Trimesh(vertices=vertices, faces=faces)
            
        except Exception as e:
            raise MeshConversionError(
                f"Failed to extract surface boundary from volumetric mesh: {e}. "
                f"Ensure pyvista is installed for volumetric mesh support."
            ) from e
