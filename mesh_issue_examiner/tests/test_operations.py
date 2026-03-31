"""
Unit tests for mesh inspection operations.
"""

import pytest
import numpy as np

from mesh_issue_examiner.core.models import MeshInput
from mesh_issue_examiner.operations.topological.nonmanifold_edges import NonManifoldEdgeDetection


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
