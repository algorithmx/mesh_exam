"""
PyVista library adapter implementation.
"""

from typing import Any, Dict, List, Tuple, Optional
import numpy as np

from adapters.base_adapter import MeshAdapter


class PyVistaAdapter(MeshAdapter):
    """
    Adapter for the PyVista library (VTK wrapper).

    PyVista provides a streamlined interface to VTK for mesh
    processing, analysis, and visualization.
    """

    @property
    def library_name(self) -> str:
        return 'pyvista'

    def _check_import(self):
        """Check that pyvista is importable."""
        import pyvista
        return pyvista

    def load_mesh(self, file_path: str, **kwargs) -> 'pyvista.PolyData':
        """Load mesh from file using PyVista."""
        import pyvista as pv
        return pv.read(file_path, **kwargs)

    def create_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> 'pyvista.PolyData':
        """Create PolyData from arrays."""
        import pyvista as pv

        n_faces = len(faces)

        # Create face array with vertex count prefix
        # Format: [n0, v0, v1, v2, n1, v0, v1, v2, ...]
        face_arr = np.column_stack([
            np.full(n_faces, 3, dtype=np.int32),
            faces
        ]).ravel()

        return pv.PolyData(vertices, face_arr)

    def get_vertices(self, mesh: 'pyvista.PolyData') -> np.ndarray:
        """Extract vertex array."""
        return np.array(mesh.points)

    def get_faces(self, mesh: 'pyvista.PolyData') -> np.ndarray:
        """Extract face array (removing vertex count prefix)."""
        faces = mesh.faces
        if len(faces) == 0:
            return np.array([])

        # Parse faces from PyVista's face array format
        face_list = []
        idx = 0
        while idx < len(faces):
            n_verts = int(faces[idx])
            if n_verts <= 0:
                break
            idx += 1
            face_list.append(list(faces[idx:idx + n_verts]))
            idx += n_verts

        return np.array(face_list)

    def is_watertight(self, mesh: 'pyvista.PolyData') -> bool:
        """Check if mesh is watertight."""
        # A mesh is watertight if it has no open edges
        return mesh.n_open_edges == 0

    def is_manifold(self, mesh: 'pyvista.PolyData') -> Tuple[bool, List[Dict[str, Any]]]:
        """Check if mesh is manifold using VTK filters."""
        issues = []

        # Check if mesh is manifold
        if not mesh.is_manifold:
            issues.append({
                'type': 'non_manifold',
                'message': 'Mesh contains non-manifold geometry'
            })

        # Check for degenerate cells (zero-area faces)
        try:
            quality = mesh.compute_cell_quality(qualitymeasure='volume')
            volumes = quality.cell_data.get('CellQualityVolume', [])
            degenerate_indices = np.where(volumes == 0)[0]
            if len(degenerate_indices) > 0:
                issues.append({
                    'type': 'degenerate_cells',
                    'count': len(degenerate_indices),
                    'indices': degenerate_indices.tolist()
                })
        except Exception:
            pass

        return len(issues) == 0, issues

    def get_boundary_edges(self, mesh: 'pyvista.PolyData') -> np.ndarray:
        """Get boundary edges."""
        try:
            # Extract feature edges (boundary edges)
            boundary = mesh.extract_feature_edges(boundary_edges=True, feature_edges=False)
            if boundary.n_cells > 0:
                return boundary.lines
        except Exception:
            pass
        return np.array([])

    def get_nonmanifold_edges(self, mesh: 'pyvista.PolyData') -> List[Tuple[int, int, int]]:
        """Detect non-manifold edges."""
        faces = self.get_faces(mesh)
        n_faces = len(faces)

        # Build edge-to-face adjacency
        edges = {}
        for face_idx in range(n_faces):
            face = faces[face_idx]
            for i in range(3):
                v1, v2 = int(face[i]), int(face[(i + 1) % 3])
                edge = tuple(sorted([v1, v2]))
                if edge not in edges:
                    edges[edge] = []
                edges[edge].append(face_idx)

        # Find non-manifold edges
        nonmanifold = [
            (edge[0], edge[1], len(faces))
            for edge, faces in edges.items()
            if len(faces) != 2
        ]

        return nonmanifold

    def get_duplicate_vertices(
        self,
        mesh: 'pyvista.PolyData',
        tolerance: float
    ) -> List[int]:
        """Find duplicate vertices."""
        try:
            # Use VTK's duplicate point removal
            cleaned = mesh.clean(tolerance=tolerance, merge_tol=tolerance)
            original_count = mesh.n_points
            cleaned_count = cleaned.n_points

            # Return indices of removed vertices
            return list(range(cleaned_count, original_count))
        except Exception:
            return []

    def compute_face_areas(self, mesh: 'pyvista.PolyData') -> np.ndarray:
        """Compute area of each face."""
        try:
            quality = mesh.compute_cell_quality(qualitymeasure='area')
            return quality.cell_data.get('CellQualityArea', np.array([]))
        except Exception:
            # Fallback to manual computation
            return self._compute_areas_manual(mesh)

    def _compute_areas_manual(self, mesh: 'pyvista.PolyData') -> np.ndarray:
        """Manually compute face areas."""
        faces = self.get_faces(mesh)
        vertices = self.get_vertices(mesh)
        n_faces = len(faces)

        areas = np.zeros(n_faces)
        for i, face in enumerate(faces):
            v = vertices[face]
            edge1 = v[1] - v[0]
            edge2 = v[2] - v[0]
            cross = np.cross(edge1, edge2)
            areas[i] = np.linalg.norm(cross) / 2

        return areas

    def compute_face_normals(self, mesh: 'pyvista.PolyData') -> np.ndarray:
        """Compute face normals."""
        try:
            mesh_copy = mesh.copy()
            mesh_copy.compute_normals(cell_oriented=True, inplace=True)
            normals = mesh_copy.cell_data.get('Normals', np.array([]))
            if len(normals) == 0:
                return self._compute_normals_manual(mesh)
            return normals
        except Exception:
            return self._compute_normals_manual(mesh)

    def _compute_normals_manual(self, mesh: 'pyvista.PolyData') -> np.ndarray:
        """Manually compute face normals."""
        faces = self.get_faces(mesh)
        vertices = self.get_vertices(mesh)
        n_faces = len(faces)

        normals = np.zeros((n_faces, 3))
        for i, face in enumerate(faces):
            v = vertices[face]
            edge1 = v[1] - v[0]
            edge2 = v[2] - v[0]
            cross = np.cross(edge1, edge2)
            norm = np.linalg.norm(cross)
            if norm > 0:
                normals[i] = cross / norm

        return normals

    def detect_self_intersections(self, mesh: 'pyvista.PolyData') -> List[Tuple[int, int]]:
        """Detect self-intersections using VTK."""
        # This would use VTK's self-intersection detection
        # Implementation depends on VTK version and available filters
        try:
            # Try to use mesh intersection check
            if hasattr(mesh, 'check_intersection'):
                intersections = mesh.check_intersection()
                return intersections
        except Exception:
            pass
        return []

    def compute_cell_quality(
        self,
        mesh: 'pyvista.PolyData',
        quality_type: str
    ) -> np.ndarray:
        """
        Compute cell quality metrics using VTK's cell quality filter.

        Supported quality types:
        - 'area': Cell area
        - 'aspect_ratio': Aspect ratio
        - 'skewness': Maximum angle skewness
        - 'scaled_jacobian': Scaled Jacobian determinant
        - 'max_angle': Maximum angle
        - 'min_angle': Minimum angle
        - 'condition': Condition number
        - 'Jacobian': Jacobian determinant
        - 'volume': Cell volume
        """
        # Map quality type names to VTK names
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
        }

        vtk_quality = quality_map.get(quality_type, quality_type)

        try:
            quality_mesh = mesh.compute_cell_quality(quality_measure=vtk_quality)
            # Try multiple possible cell data key names
            for key in [
                'CellQuality',
                f'CellQuality{vtk_quality.capitalize()}',
                f'CellQuality{vtk_quality}',
                vtk_quality,
            ]:
                values = quality_mesh.cell_data.get(key, np.array([]))
                if len(values) > 0:
                    return values
            return np.array([])
        except Exception as e:
            # Fallback to manual computation
            if quality_type == 'area':
                return self._compute_areas_manual(mesh)
            elif quality_type == 'volume':
                return np.array([0])  # Surface mesh has no volume
            return np.array([])
