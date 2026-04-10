"""
Unit tests for SurfaceOrientationCheck operation.

Tests the surface orientation consistency check that detects faces with
inconsistent winding order (flipped normals).

References:
-----------
- Borodin, P., Zachmann, G., & Klein, R. (2004).
  "Consistent Normal Orientation for Polygonal Meshes"
  Computer Graphics International (CGI), pp. 18-25. IEEE.
  https://user.informatik.uni-bremen.de/zach/papers/orientation_print.pdf

- StackOverflow - "how to check if all the faces face outward"
  https://stackoverflow.com/questions/77927993/how-to-check-if-all-the-faces-face-outward
"""

import numpy as np
import pytest

from mesh_issue_examiner.core.models import MeshInput, IssueCategory, IssueSeverity
from mesh_issue_examiner.operations.topological.surface_orientation import SurfaceOrientationCheck


class TestSurfaceOrientationCheck:
    """Test cases for surface orientation consistency check."""

    def test_consistent_mesh_no_issues(self):
        """
        Test that a consistently oriented mesh produces no issues.
        
        Two triangles sharing an edge with consistent winding:
        - Face 0: (0, 1, 2) - edge (0,1) traversed as 0→1
        - Face 1: (1, 3, 2) - shared edge is (1,2), traversed as 1→2
        
        Edge (1,2) is shared: Face 0 has it as 1→2? No, Face 0 has (0,1), (1,2), (2,0)
        So Face 0 edge (1,2) is 1→2, Face 1 edge (1,2) is... (1,3,2) has edges (1,3), (3,2), (2,1)
        So shared edge (1,2): Face 0 has 1→2, Face 1 has 2→1. Opposite! Consistent.
        """
        # Two triangles forming a quad, consistently oriented
        vertices = np.array([
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.0, 1.0, 0.0],  # 2
            [1.0, 1.0, 0.0],  # 3
        ])
        faces = np.array([
            [0, 1, 2],  # Bottom-left triangle
            [1, 3, 2],  # Top-right triangle (shares edge 1-2)
        ])
        
        mesh = MeshInput(vertices=vertices, faces=faces, mesh_id="consistent_quad")
        operation = SurfaceOrientationCheck()
        issues = operation.execute(mesh)
        
        assert len(issues) == 0, "Consistently oriented mesh should have no issues"

    def test_inconsistent_orientation_detected(self):
        """
        Test detection of inconsistent face orientation.
        
        Two triangles sharing an edge with INCONSISTENT winding:
        - Face 0: (0, 1, 2) - edge (1,2) traversed as 1→2
        - Face 1: (1, 2, 3) - shared edge (1,2) traversed as 1→2 (SAME!)
        
        Per StackOverflow algorithm: "If the other face has them in the order `a,b`, 
        flip its winding" - we detect this as inconsistent.
        """
        vertices = np.array([
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
        ])
        faces = np.array([
            [0, 1, 2],  # Triangle with normal +Z
            [1, 2, 3],  # Triangle sharing edge (1,2), same winding = flipped!
        ])
        
        mesh = MeshInput(vertices=vertices, faces=faces, mesh_id="inconsistent_quad")
        operation = SurfaceOrientationCheck()
        issues = operation.execute(mesh)
        
        assert len(issues) == 1, "Should detect orientation inconsistency"
        issue = issues[0]
        assert issue.category == IssueCategory.ORIENTATION
        assert issue.metrics['inconsistent_face_count'] == 2
        assert issue.metrics['inconsistent_edge_count'] == 1

    def test_single_face_no_issues(self):
        """Test that a single face mesh produces no issues (no shared edges to check)."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ])
        faces = np.array([[0, 1, 2]])
        
        mesh = MeshInput(vertices=vertices, faces=faces, mesh_id="single_triangle")
        operation = SurfaceOrientationCheck()
        issues = operation.execute(mesh)
        
        assert len(issues) == 0, "Single face has no shared edges to validate"

    def test_boundary_edges_skipped(self):
        """
        Test that boundary edges (1 incident face) are skipped.
        
        Per Borodin et al.: "As boundary edges we define all edges which are 
        incident to only one polygon."
        
        Open mesh should not produce orientation issues from boundary edges.
        """
        # Single triangle - all edges are boundary edges
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ])
        faces = np.array([[0, 1, 2]])
        
        mesh = MeshInput(vertices=vertices, faces=faces, mesh_id="open_mesh")
        operation = SurfaceOrientationCheck()
        issues = operation.execute(mesh)
        
        assert len(issues) == 0, "Boundary edges should not trigger orientation issues"

    def test_non_manifold_edges_skipped(self):
        """
        Test that non-manifold edges (>2 incident faces) are skipped.
        
        Per Borodin et al.: "As nonmanifold edges we define all edges which are 
        incident to more than two polygons."
        """
        # Create a non-manifold edge (edge shared by 3 faces)
        vertices = np.array([
            [0.0, 0.0, 0.0],  # 0 - shared vertex
            [1.0, 0.0, 0.0],  # 1 - shared vertex (forms shared edge 0-1)
            [0.0, 1.0, 0.0],  # 2
            [1.0, 1.0, 0.0],  # 3
            [0.5, 0.5, 1.0],  # 4 - top vertex for "fin"
        ])
        # Three faces sharing edge (0,1)
        faces = np.array([
            [0, 1, 2],  # Face 0
            [0, 1, 3],  # Face 1
            [0, 1, 4],  # Face 2 - shares edge (0,1) with both above
        ])
        
        mesh = MeshInput(vertices=vertices, faces=faces, mesh_id="nonmanifold")
        operation = SurfaceOrientationCheck()
        issues = operation.execute(mesh)
        
        # Non-manifold edges should be skipped, so no orientation issues detected
        # (The faces may still be consistently oriented, just can't verify via shared edge)
        assert len(issues) == 0, "Non-manifold edges should be skipped"

    def test_disconnected_components(self):
        """
        Test handling of disconnected components.
        
        Per Borodin et al. Section 3.2: "we divide it into a set of manifold 
        surface patches, which either are not connected with each other"
        
        Each component should be analyzed independently.
        """
        # Two separate triangles (disconnected components)
        vertices = np.array([
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.5, 1.0, 0.0],  # 2
            [5.0, 0.0, 0.0],  # 3 - far away
            [6.0, 0.0, 0.0],  # 4
            [5.5, 1.0, 0.0],  # 5
        ])
        faces = np.array([
            [0, 1, 2],  # Component 0
            [3, 4, 5],  # Component 1 (disconnected)
        ])
        
        mesh = MeshInput(vertices=vertices, faces=faces, mesh_id="disconnected")
        operation = SurfaceOrientationCheck()
        issues = operation.execute(mesh)
        
        # No shared edges between components, so no inconsistency detected
        assert len(issues) == 0, "Disconnected components have no shared edges"
        
        # But verify components are correctly counted
        edge_map = operation._build_edge_face_map(mesh)
        adjacency = operation._build_face_adjacency(edge_map)
        components = operation._find_connected_components(len(faces), adjacency)
        assert len(components) == 2, "Should detect 2 disconnected components"

    def test_cube_consistent(self):
        """Test that a closed cube mesh has consistent orientation."""
        # Unit cube vertices
        vertices = np.array([
            [0, 0, 0],  # 0
            [1, 0, 0],  # 1
            [1, 1, 0],  # 2
            [0, 1, 0],  # 3
            [0, 0, 1],  # 4
            [1, 0, 1],  # 5
            [1, 1, 1],  # 6
            [0, 1, 1],  # 7
        ])
        # Cube faces (triangulated quads, all CCW when viewed from outside)
        faces = np.array([
            # Bottom (facing down, so CW from above = CCW from below)
            [0, 2, 1], [0, 3, 2],
            # Top (facing up, CCW from above)
            [4, 5, 6], [4, 6, 7],
            # Front (facing -Y, CCW from front)
            [0, 1, 5], [0, 5, 4],
            # Back (facing +Y, CCW from back)
            [2, 3, 7], [2, 7, 6],
            # Left (facing -X, CCW from left)
            [0, 4, 7], [0, 7, 3],
            # Right (facing +X, CCW from right)
            [1, 2, 6], [1, 6, 5],
        ])
        
        mesh = MeshInput(vertices=vertices, faces=faces, mesh_id="cube")
        operation = SurfaceOrientationCheck()
        issues = operation.execute(mesh)
        
        # Cube should be consistently oriented
        assert len(issues) == 0, f"Cube should be consistently oriented, got: {issues}"

    def test_flipped_face_in_cube(self):
        """
        Test detection of a single flipped face in a cube.
        
        One face with reversed winding should be detected.
        """
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ])
        faces = np.array([
            [0, 2, 1], [0, 3, 2],  # Bottom
            [4, 5, 6], [4, 6, 7],  # Top
            [0, 1, 5], [0, 5, 4],  # Front
            [2, 3, 7], [2, 7, 6],  # Back
            [0, 4, 7], [0, 7, 3],  # Left
            [1, 6, 2], [1, 5, 6],  # Right - FLIPPED: [1, 2, 6] -> [1, 6, 2]
        ])
        
        mesh = MeshInput(vertices=vertices, faces=faces, mesh_id="cube_with_flip")
        operation = SurfaceOrientationCheck()
        issues = operation.execute(mesh)
        
        assert len(issues) == 1, "Should detect the flipped face"
        issue = issues[0]
        assert issue.metrics['inconsistent_face_count'] >= 2  # Flipped face + adjacent faces

    def test_empty_mesh(self):
        """Test handling of empty mesh."""
        vertices = np.array([]).reshape(0, 3)
        faces = np.array([]).reshape(0, 3)
        
        mesh = MeshInput(vertices=vertices, faces=faces, mesh_id="empty")
        operation = SurfaceOrientationCheck()
        issues = operation.execute(mesh)
        
        assert len(issues) == 0, "Empty mesh should have no issues"

    def test_metrics_structure(self):
        """Test that issue metrics have expected structure."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        faces = np.array([
            [0, 1, 2],
            [1, 2, 3],  # Inconsistent with face 0
        ])
        
        mesh = MeshInput(vertices=vertices, faces=faces, mesh_id="test_metrics")
        operation = SurfaceOrientationCheck()
        issues = operation.execute(mesh)
        
        assert len(issues) == 1
        issue = issues[0]
        
        # Check required metrics
        assert 'inconsistent_face_count' in issue.metrics
        assert 'total_faces' in issue.metrics
        assert 'violation_rate' in issue.metrics
        assert 'inconsistent_edge_count' in issue.metrics
        assert 'total_component_count' in issue.metrics
        
        # Check values
        assert issue.metrics['total_faces'] == 2
        assert issue.metrics['violation_rate'] == 1.0  # Both faces inconsistent

    def test_operation_metadata(self):
        """Test operation metadata is correctly defined."""
        operation = SurfaceOrientationCheck()
        meta = operation.metadata
        
        assert meta.operation_id == 'topology.detect_inconsistent_orientation'
        assert meta.category == IssueCategory.ORIENTATION
        assert 'Borodin' in meta.description or 'StackOverflow' in meta.description
        assert 'orientation' in meta.tags


class TestSurfaceOrientationEdgeCases:
    """Edge case tests for surface orientation check."""

    def test_polygonal_faces(self):
        """Test handling of non-triangular faces."""
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0.5, 1.5, 0], [0, 1, 0]
        ])
        # Two faces: one quad and one triangle sharing an edge
        # Use object dtype for mixed-size faces
        faces = np.array([
            [0, 1, 2, 3],  # Quad 1
            [0, 3, 4],     # Triangle sharing edge (0,3)
        ], dtype=object)
        
        mesh = MeshInput(vertices=vertices, faces=faces, mesh_id="polygonal")
        operation = SurfaceOrientationCheck()
        
        # Should handle without error
        issues = operation.execute(mesh)
        # Result depends on winding, but should not crash

    def test_degenerate_face_handling(self):
        """Test handling of degenerate faces (zero area)."""
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0],
            [0.5, 0, 0],  # On edge between 0 and 1
        ])
        # Degenerate triangle (0, 3, 1 are collinear)
        faces = np.array([
            [0, 1, 2],  # Normal triangle
            [0, 3, 1],  # Degenerate (collinear points)
        ])
        
        mesh = MeshInput(vertices=vertices, faces=faces, mesh_id="degenerate")
        operation = SurfaceOrientationCheck()
        
        # Should handle without error
        issues = operation.execute(mesh)

    def test_large_mesh_performance(self):
        """Test that operation handles reasonably large meshes."""
        # Create a larger grid mesh
        n = 50  # 50x50 grid = 2500 quads = 5000 triangles
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        xx, yy = np.meshgrid(x, y)
        vertices = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(n * n)])
        
        # Create triangulated quads
        faces = []
        for i in range(n - 1):
            for j in range(n - 1):
                v0 = i * n + j
                v1 = v0 + 1
                v2 = (i + 1) * n + j
                v3 = v2 + 1
                # Two triangles per quad, consistently oriented
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])
        faces = np.array(faces)
        
        mesh = MeshInput(vertices=vertices, faces=faces, mesh_id="large_grid")
        operation = SurfaceOrientationCheck()
        issues = operation.execute(mesh)
        
        assert len(issues) == 0, "Large consistent mesh should have no issues"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
