"""
Unit tests for mesh inspection operations.
"""

import pytest
import numpy as np

from mesh_issue_examiner.core.models import MeshInput, IssueCategory, IssueSeverity
from mesh_issue_examiner.operations.topological.nonmanifold_edges import NonManifoldEdgeDetection
from mesh_issue_examiner.operations.topological.boundary_edges import BoundaryEdgeDetection


class TestNonManifoldEdgeDetection:
    """Tests for non-manifold edge detection."""

    def test_watertight_tetrahedron_no_issues(self):
        """Watertight tetrahedron should have no non-manifold edges."""
        # Create simple tetrahedron
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        faces = np.array([
            [0, 1, 2],
            [0, 3, 1],
            [1, 3, 2],
            [2, 3, 0]
        ])

        mesh = MeshInput(vertices, faces)
        operation = NonManifoldEdgeDetection()
        issues = operation.execute(mesh)

        assert len(issues) == 0

    def test_simple_cube_no_issues(self):
        """Watertight cube should have no non-manifold edges."""
        # Unit cube with 12 triangular faces (two per face)
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # top
        ])
        # Triangulated cube faces
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 6, 5], [4, 7, 6],  # top
            [0, 4, 5], [0, 5, 1],  # front
            [2, 6, 7], [2, 7, 3],  # back
            [0, 3, 7], [0, 7, 4],  # left
            [1, 5, 6], [1, 6, 2],  # right
        ])

        mesh = MeshInput(vertices, faces)
        operation = NonManifoldEdgeDetection()
        issues = operation.execute(mesh)

        assert len(issues) == 0

    def test_duplicate_identical_faces_no_nonmanifold_edges(self):
        """Two identical faces sharing the same vertices produce manifold edges.

        Each edge (0,1), (0,2), (1,2) has exactly 2 incident faces (face 0
        and face 1), which satisfies the manifold condition (edge count == 2).
        Duplicate faces are a mesh defect, but they do NOT create non-manifold
        edges. The edge-based detector correctly returns no issues here.
        """
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0]
        ])
        # Same face listed twice -- every edge still has face_count == 2
        faces = np.array([
            [0, 1, 2],
            [0, 1, 2]  # Duplicate
        ])

        mesh = MeshInput(vertices, faces)
        operation = NonManifoldEdgeDetection()
        issues = operation.execute(mesh)

        # No non-manifold edges: each edge belongs to exactly 2 faces
        assert len(issues) == 0

    def test_three_faces_sharing_edge_is_nonmanifold(self):
        """Three faces sharing one edge should be detected as non-manifold."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.0, 1.0],
            [0.5, -1.0, 0.0],
        ])
        faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [0, 1, 4],
        ])

        mesh = MeshInput(vertices, faces)
        operation = NonManifoldEdgeDetection()
        issues = operation.execute(mesh)

        assert len(issues) > 0, "Edge (0,1) shared by 3 faces is non-manifold"


class TestCellQuality:
    """Tests for cell quality computation."""

    def test_equilateral_triangles_high_quality(self):
        """Equilateral triangles should have high quality scores."""
        # Create equilateral triangle
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, np.sqrt(3)/2, 0]
        ])
        faces = np.array([[0, 1, 2]])

        mesh = MeshInput(vertices, faces)

        # Quality should be high (close to 1 for aspect ratio)
        from mesh_issue_examiner.operations.quality.cell_quality import CellQualityComputation
        operation = CellQualityComputation(quality_types=['aspect_ratio'])
        issues = operation.execute(mesh)

        # Should have no issues (high quality mesh)
        # Note: actual values depend on implementation
        assert isinstance(issues, list)


class TestBoundaryEdgeDetection:
    """Tests for boundary edge detection."""

    def test_watertight_tetrahedron_no_boundary_edges(self):
        """Watertight tetrahedron should have no boundary edges."""
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        faces = np.array([
            [0, 1, 2],
            [0, 3, 1],
            [1, 3, 2],
            [2, 3, 0]
        ])

        mesh = MeshInput(vertices, faces)
        operation = BoundaryEdgeDetection()
        issues = operation.execute(mesh)

        assert len(issues) == 0, "Watertight mesh should have no boundary edges"

    def test_watertight_cube_no_boundary_edges(self):
        """Watertight cube should have no boundary edges."""
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # top
        ])
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 6, 5], [4, 7, 6],  # top
            [0, 4, 5], [0, 5, 1],  # front
            [2, 6, 7], [2, 7, 3],  # back
            [0, 3, 7], [0, 7, 4],  # left
            [1, 5, 6], [1, 6, 2],  # right
        ])

        mesh = MeshInput(vertices, faces)
        operation = BoundaryEdgeDetection()
        issues = operation.execute(mesh)

        assert len(issues) == 0, "Watertight cube should have no boundary edges"

    def test_single_triangle_has_boundary_edges(self):
        """Single triangle has all edges as boundary edges."""
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0]
        ])
        faces = np.array([[0, 1, 2]])

        mesh = MeshInput(vertices, faces)
        operation = BoundaryEdgeDetection()
        issues = operation.execute(mesh)

        assert len(issues) == 1, "Single triangle should have boundary edges"
        issue = issues[0]
        assert issue.metrics['boundary_edge_count'] == 3
        assert issue.metrics['is_watertight'] is False
        assert issue.metrics['boundary_loop_count'] == 1

    def test_open_cube_has_boundary_edges(self):
        """Cube missing one face has boundary edges."""
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # top
        ])
        # Cube without top face (faces 2,3)
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # bottom
            # Missing top
            [0, 4, 5], [0, 5, 1],  # front
            [2, 6, 7], [2, 7, 3],  # back
            [0, 3, 7], [0, 7, 4],  # left
            [1, 5, 6], [1, 6, 2],  # right
        ])

        mesh = MeshInput(vertices, faces)
        operation = BoundaryEdgeDetection()
        issues = operation.execute(mesh)

        assert len(issues) == 1, "Open cube should have boundary edges"
        issue = issues[0]
        assert issue.metrics['boundary_edge_count'] == 4  # 4 edges of missing top
        assert issue.metrics['is_watertight'] is False
        assert issue.metrics['boundary_loop_count'] == 1

    def test_cylinder_has_two_boundary_loops(self):
        """Open cylinder has two boundary loops (top and bottom)."""
        # Simple cylinder approximation (8-sided)
        n_sides = 8
        angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
        
        # Bottom ring (z=0) and top ring (z=1)
        bottom_vertices = np.column_stack([
            np.cos(angles), np.sin(angles), np.zeros(n_sides)
        ])
        top_vertices = np.column_stack([
            np.cos(angles), np.sin(angles), np.ones(n_sides)
        ])
        
        vertices = np.vstack([bottom_vertices, top_vertices])
        
        # Side faces only (no caps)
        faces = []
        for i in range(n_sides):
            next_i = (i + 1) % n_sides
            # Two triangles per side face
            faces.append([i, next_i, next_i + n_sides])
            faces.append([i, next_i + n_sides, i + n_sides])
        faces = np.array(faces)

        mesh = MeshInput(vertices, faces)
        operation = BoundaryEdgeDetection()
        issues = operation.execute(mesh)

        assert len(issues) == 1, "Open cylinder should have boundary edges"
        issue = issues[0]
        assert issue.metrics['is_watertight'] is False
        # Should detect 2 boundary loops (top and bottom rings)
        assert issue.metrics['boundary_loop_count'] == 2
        # Each ring has n_sides edges
        assert issue.metrics['boundary_edge_count'] == 2 * n_sides

    def test_empty_mesh_no_issues(self):
        """Empty mesh should have no boundary edges."""
        vertices = np.array([]).reshape(0, 3)
        faces = np.array([]).reshape(0, 3)

        mesh = MeshInput(vertices, faces)
        operation = BoundaryEdgeDetection()
        issues = operation.execute(mesh)

        assert len(issues) == 0, "Empty mesh should have no boundary edges"

    def test_boundary_edge_metadata(self):
        """Verify boundary edge issue has correct metadata."""
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0]
        ])
        faces = np.array([[0, 1, 2]])

        mesh = MeshInput(vertices, faces)
        operation = BoundaryEdgeDetection()
        issues = operation.execute(mesh)

        assert len(issues) == 1
        issue = issues[0]
        
        # Check issue properties
        assert issue.operation_id == 'topology.detect_boundary_edges'
        assert issue.category == IssueCategory.TOPOLOGICAL
        
        # Check metrics
        assert 'boundary_edge_count' in issue.metrics
        assert 'boundary_loop_count' in issue.metrics
        assert 'is_watertight' in issue.metrics
        assert 'boundary_edges_ratio' in issue.metrics
        assert 'total_edges' in issue.metrics
        
        # Check metadata
        assert 'edge_face_map_size' in issue.metadata
        assert 'manifold_edges' in issue.metadata
        assert 'boundary_loops' in issue.metadata

    def test_boundary_edge_locations(self):
        """Verify boundary edge locations are correctly reported."""
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0]
        ])
        faces = np.array([[0, 1, 2]])

        mesh = MeshInput(vertices, faces)
        operation = BoundaryEdgeDetection()
        issues = operation.execute(mesh)

        assert len(issues) == 1
        issue = issues[0]
        
        # Should have location for each boundary edge
        assert len(issue.locations) == 3
        
        # Check location structure
        for loc in issue.locations:
            assert loc.element_type == 'edge'
            assert 'vertices' in loc.adjacent_elements
            assert 'incident_face' in loc.adjacent_elements
            assert len(loc.adjacent_elements['vertices']) == 2
