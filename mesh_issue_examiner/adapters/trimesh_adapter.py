"""
TRimesh library adapter implementation.
"""

from typing import Any, Dict, List, Tuple, Optional
import numpy as np

from adapters.base_adapter import MeshAdapter


class TrimeshAdapter(MeshAdapter):
    """
    Adapter for the TRimesh library.

    TRimesh is a pure Python library for loading and using triangular
    meshes with an emphasis on watertight surfaces.
    """

    @property
    def library_name(self) -> str:
        return 'trimesh'

    def _check_import(self):
        """Check that trimesh is importable."""
        import trimesh
        return trimesh

    def load_mesh(self, file_path: str, **kwargs) -> 'trimesh.Trimesh':
        """Load mesh from file using TRimesh."""
        import trimesh
        return trimesh.load_mesh(file_path, **kwargs)

    def create_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> 'trimesh.Trimesh':
        """Create TRimesh from arrays."""
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
        """Compute cell quality metrics."""
        n_faces = len(mesh.faces)

        if quality_type == 'area':
            return mesh.area_faces

        elif quality_type == 'aspect_ratio':
            # Compute aspect ratio using edge lengths
            aspect_ratios = []
            for face_idx in range(n_faces):
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
                    circumradius = max(edges) / (2 * np.sin(np.arccos(
                        (edges[0]**2 + edges[1]**2 - edges[2]**2) / (2 * edges[0] * edges[1])
                    ))) if len(edges) > 1 else max(edges)
                    aspect_ratios.append(circumradius / (2 * inradius) if inradius > 0 else 0)
                else:
                    aspect_ratios.append(0)
            return np.array(aspect_ratios)

        elif quality_type == 'min_angle':
            angles = []
            for face_idx in range(n_faces):
                face_verts = mesh.vertices[mesh.faces[face_idx]]
                # Compute angles at each vertex
                v0, v1, v2 = face_verts
                a = np.arccos(np.clip(np.dot(v1 - v0, v2 - v0) /
                           (np.linalg.norm(v1 - v0) * np.linalg.norm(v2 - v0)), -1, 1))
                b = np.arccos(np.clip(np.dot(v2 - v1, v0 - v1) /
                           (np.linalg.norm(v2 - v1) * np.linalg.norm(v0 - v1)), -1, 1))
                c = np.pi - a - b
                angles.append(np.degrees(min(a, b, c)))
            return np.array(angles)

        elif quality_type == 'max_angle':
            angles = []
            for face_idx in range(n_faces):
                face_verts = mesh.vertices[mesh.faces[face_idx]]
                v0, v1, v2 = face_verts
                a = np.arccos(np.clip(np.dot(v1 - v0, v2 - v0) /
                           (np.linalg.norm(v1 - v0) * np.linalg.norm(v2 - v0)), -1, 1))
                b = np.arccos(np.clip(np.dot(v2 - v1, v0 - v1) /
                           (np.linalg.norm(v2 - v1) * np.linalg.norm(v0 - v1)), -1, 1))
                c = np.pi - a - b
                angles.append(np.degrees(max(a, b, c)))
            return np.array(angles)

        else:
            raise ValueError(f"Unsupported quality type: {quality_type}")
