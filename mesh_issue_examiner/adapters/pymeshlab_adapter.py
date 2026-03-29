"""
PyMeshLab library adapter implementation.
"""

from typing import Any, Dict, List, Tuple, Optional
import numpy as np

from adapters.base_adapter import MeshAdapter


class PyMeshLabAdapter(MeshAdapter):
    """
    Adapter for the PyMeshLab library.

    PyMeshLab provides Python bindings for MeshLab's mesh processing
    capabilities, including many filters for mesh repair and analysis.
    """

    def __init__(self):
        self._current_meshset = None

    @property
    def library_name(self) -> str:
        return 'pymeshlab'

    def _check_import(self):
        """Check that pymeshlab is importable."""
        import pymeshlab
        return pymeshlab

    def load_mesh(self, file_path: str, **kwargs) -> 'pymeshlab.MeshSet':
        """Load mesh from file using PyMeshLab."""
        import pymeshlab as ml
        ms = ml.MeshSet()
        ms.load_new_mesh(file_path)
        self._current_meshset = ms
        return ms

    def create_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> 'pymeshlab.MeshSet':
        """Create mesh from arrays."""
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
        """Get boundary (non-manifold) edges."""
        meshset.select_non_manifold_edges()
        selected = meshset.selected_vertices()
        meshset.mesh_unselect_all()
        return np.array(selected)

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
        """Find duplicate vertices."""
        original_count = meshset.current_mesh().vertex_number()

        # Use PyMeshLab's merge close vertices
        meshset.merge_close_vertices(threshold=tolerance)

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
        """Compute cell quality using PyMeshLab."""
        # PyMeshLab has various quality computation capabilities
        # This is a simplified implementation

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

        return np.array([])
