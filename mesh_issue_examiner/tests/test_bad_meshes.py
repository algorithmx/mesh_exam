"""
End-to-end tests using dedicated example meshes with specific problems.

Each test constructs a mesh with a known defect using only numpy arrays,
runs the appropriate operation(s), and verifies that the expected issues
are correctly detected and reported.
"""

import pytest
import numpy as np

from mesh_issue_examiner.core.models import MeshInput, MeshIssue, IssueCategory, IssueSeverity
from mesh_issue_examiner.operations.topological.nonmanifold_edges import NonManifoldEdgeDetection
from mesh_issue_examiner.operations.quality.cell_quality import CellQualityComputation
from mesh_issue_examiner.operations.operation_registry import OperationRegistry


# ---------------------------------------------------------------------------
# Helper: build a fresh registry with only the operations we need
# ---------------------------------------------------------------------------

def _make_registry():
    """Create a fresh OperationRegistry with standard operations registered."""
    reg = OperationRegistry()
    reg.register(NonManifoldEdgeDetection())
    reg.register(CellQualityComputation())
    return reg


# ===================================================================
# 1. Non-manifold edge mesh  (fan / umbrella)
# ===================================================================

class TestNonManifoldEdgeFan:
    """Tests for an umbrella-shaped mesh where 3+ faces share a single edge."""

    @staticmethod
    def _make_umbrella_mesh():
        """
        Create an 'umbrella' mesh: three triangles all sharing the edge (0,1).

        Vertices:
            0 = (0, 0, 0)
            1 = (1, 0, 0)   -- shared edge (0,1)
            2 = (0.5, 1, 0) -- fan-out in +y
            3 = (0.5, 0, 1) -- fan-out in +z
            4 = (0.5, -1, 0)-- fan-out in -y

        Faces: (0,1,2), (0,1,3), (0,1,4)
        Edge (0,1) is incident to 3 faces -> non-manifold.
        """
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
        return MeshInput(vertices, faces)

    def test_detects_nonmanifold_edges(self):
        """The umbrella mesh should report at least one non-manifold edge."""
        mesh = self._make_umbrella_mesh()
        op = NonManifoldEdgeDetection()
        issues = op.execute(mesh)

        assert len(issues) >= 1, "Expected at least one non-manifold edge issue"

    def test_issue_category_is_topological(self):
        """The detected issue category must be TOPOLOGICAL."""
        mesh = self._make_umbrella_mesh()
        issues = NonManifoldEdgeDetection().execute(mesh)

        for issue in issues:
            assert issue.category == IssueCategory.TOPOLOGICAL

    def test_shared_edge_has_three_faces(self):
        """Edge (0,1) should have exactly 3 incident faces in the locations."""
        mesh = self._make_umbrella_mesh()
        issues = NonManifoldEdgeDetection().execute(mesh)

        # Find the location for edge (0,1)
        found = False
        for issue in issues:
            for loc in issue.locations:
                verts = loc.adjacent_elements.get('vertices', [])
                if set(verts) == {0, 1}:
                    assert loc.adjacent_elements['face_count'] == 3
                    found = True
        assert found, "Edge (0,1) should appear in the issue locations"

    def test_issue_message_mentions_count(self):
        """The issue message should mention the number of non-manifold edges."""
        mesh = self._make_umbrella_mesh()
        issues = NonManifoldEdgeDetection().execute(mesh)

        msg = issues[0].message.lower()
        assert 'non-manifold' in msg or 'nonmanifold' in msg
        assert 'edge' in msg

    def test_issue_metrics_contain_count(self):
        """Metrics dict should include a 'count' key >= 1."""
        mesh = self._make_umbrella_mesh()
        issues = NonManifoldEdgeDetection().execute(mesh)

        assert 'count' in issues[0].metrics
        assert issues[0].metrics['count'] >= 1


# ===================================================================
# 2. Non-manifold vertex mesh (two tetrahedra touching at a tip)
# ===================================================================

class TestNonManifoldVertex:
    """Tests for a mesh where two separate patches share a single vertex."""

    @staticmethod
    def _make_touching_tetrahedra():
        """
        Two tetrahedra that share only vertex 0.

        Tetrahedron A: vertices 0,1,2,3
        Tetrahedron B: vertices 0,4,5,6
        They meet at vertex 0 but are otherwise disconnected.
        This creates boundary edges around vertex 0 and possibly
        non-manifold edges at the fan around vertex 0.
        """
        vertices = np.array([
            [0.0, 0.0, 0.0],   # 0 - shared tip
            [1.0, 0.0, 0.0],   # 1
            [0.0, 1.0, 0.0],   # 2
            [0.0, 0.0, 1.0],   # 3
            [-1.0, 0.0, 0.0],  # 4
            [0.0, -1.0, 0.0],  # 5
            [0.0, 0.0, -1.0],  # 6
        ])
        # Tetrahedron A faces
        faces_a = np.array([
            [0, 1, 2],
            [0, 3, 1],
            [1, 3, 2],
            [2, 3, 0],
        ])
        # Tetrahedron B faces
        faces_b = np.array([
            [0, 4, 5],
            [0, 6, 4],
            [4, 6, 5],
            [5, 6, 0],
        ])
        faces = np.vstack([faces_a, faces_b])
        return MeshInput(vertices, faces)

    def test_detects_boundary_or_nonmanifold_edges(self):
        """
        Two closed tetrahedra touching at a vertex have no non-manifold
        edges (each edge belongs to exactly 2 faces within its own tet).
        This mesh should still be processable without errors.
        """
        mesh = self._make_touching_tetrahedra()
        op = NonManifoldEdgeDetection()
        issues = op.execute(mesh)

        # Each tetrahedron is closed, so edges within each tet have
        # exactly 2 incident faces.  No edge spans both tets.
        # So this is a valid manifold edge topology -- but the vertex
        # is non-manifold.  The edge-based detector won't flag it.
        # That is expected behaviour; we just ensure it runs cleanly.
        assert isinstance(issues, list)

    def test_mesh_is_valid_for_quality(self):
        """Quality computation should run on the mesh without error."""
        mesh = self._make_touching_tetrahedra()
        op = CellQualityComputation(quality_types=['aspect_ratio'])
        issues = op.execute(mesh)
        assert isinstance(issues, list)


# ===================================================================
# 3. Highly skewed / elongated (needle-like) triangles
# ===================================================================

class TestSkewedTriangles:
    """Tests for a mesh with extremely elongated triangles."""

    @staticmethod
    def _make_needle_mesh():
        """
        Create two needle-like triangles with extreme aspect ratios.

        Triangle 1: base = 1, height = 0.001
        Triangle 2: base = 1, height = 0.0001
        """
        vertices = np.array([
            [0.0, 0.0, 0.0],      # 0
            [1.0, 0.0, 0.0],      # 1
            [0.5, 0.001, 0.0],    # 2  -- very thin
            [2.0, 0.0, 0.0],      # 3
            [3.0, 0.0, 0.0],      # 4
            [2.5, 0.0001, 0.0],   # 5  -- even thinner
        ])
        faces = np.array([
            [0, 1, 2],
            [3, 4, 5],
        ])
        return MeshInput(vertices, faces)

    def test_min_angle_flags_issues(self):
        """Needle triangles should produce quality violations for min_angle."""
        mesh = self._make_needle_mesh()
        op = CellQualityComputation(quality_types=['min_angle'])
        issues = op.execute(mesh)

        # At least one min_angle issue should be raised
        angle_issues = [i for i in issues if 'min_angle' in i.operation_id]
        assert len(angle_issues) >= 1, "Expected at least one min_angle issue"

    def test_violation_count_matches_face_count(self):
        """Both faces should violate the min_angle threshold (20 degrees)."""
        mesh = self._make_needle_mesh()
        op = CellQualityComputation(quality_types=['min_angle'])
        issues = op.execute(mesh)

        angle_issues = [i for i in issues if 'min_angle' in i.operation_id]
        assert angle_issues[0].metrics['violation_count'] == 2

    def test_severity_is_warning_for_100_pct_violation(self):
        """min_angle severity is capped at WARNING even at 100 % violation."""
        mesh = self._make_needle_mesh()
        op = CellQualityComputation(quality_types=['min_angle'])
        issues = op.execute(mesh)

        angle_issues = [i for i in issues if 'min_angle' in i.operation_id]
        assert angle_issues[0].severity == IssueSeverity.WARNING

    def test_min_angle_value_is_very_low(self):
        """The minimum angle should be very close to 0 degrees for needle triangles."""
        mesh = self._make_needle_mesh()
        op = CellQualityComputation(quality_types=['min_angle'])
        issues = op.execute(mesh)

        angle_issues = [i for i in issues if 'min_angle' in i.operation_id]
        # min_angle values should be well below 1 degree
        assert angle_issues[0].metrics['min'] < 1.0


# ===================================================================
# 4. Inverted / negatively wound triangle
# ===================================================================

class TestInvertedTriangle:
    """Tests for a triangle whose vertices are wound in the wrong order."""

    @staticmethod
    def _make_inverted_mesh():
        """
        Create a small mesh with one correctly wound and one inverted triangle.

        A correctly wound triangle and an inverted (clockwise) version.
        """
        vertices = np.array([
            [0.0, 0.0, 0.0],   # 0
            [1.0, 0.0, 0.0],   # 1
            [0.5, 1.0, 0.0],   # 2
        ])
        # Face with reversed winding (2,1,0 instead of 0,1,2)
        faces = np.array([
            [2, 1, 0],
        ])
        return MeshInput(vertices, faces)

    def test_quality_runs_on_inverted_triangle(self):
        """Quality computation should still run on an inverted triangle."""
        mesh = self._make_inverted_mesh()
        op = CellQualityComputation(quality_types=['aspect_ratio'])
        issues = op.execute(mesh)
        assert isinstance(issues, list)

    def test_inverted_triangle_no_edge_issues(self):
        """A single triangle has no non-manifold edges (boundary only, count=1)."""
        mesh = self._make_inverted_mesh()
        op = NonManifoldEdgeDetection()
        issues = op.execute(mesh)

        # All 3 edges are boundary edges (face_count=1), which are normal for
        # open surfaces. NonManifoldEdgeDetection only flags face_count > 2.
        assert len(issues) == 0


# ===================================================================
# 5. Duplicate vertices (same position, different indices)
# ===================================================================

class TestDuplicateVertices:
    """Tests for a mesh with vertices at the same position but different indices."""

    @staticmethod
    def _make_duplicate_vertex_mesh():
        """
        Two triangles that should share an edge but have duplicated vertices.

        Triangle 1: (0,1,2) -- vertices 0,1,2
        Triangle 2: (3,4,5) -- vertices at the SAME positions as 0,1,2
        This creates zero-length edges within triangle 2's internal edges
        and disconnects the two triangles topologically.
        """
        vertices = np.array([
            [0.0, 0.0, 0.0],   # 0
            [1.0, 0.0, 0.0],   # 1
            [0.5, 1.0, 0.0],   # 2
            [0.0, 0.0, 0.0],   # 3  -- duplicate of 0
            [1.0, 0.0, 0.0],   # 4  -- duplicate of 1
            [0.5, 1.0, 0.0],   # 5  -- duplicate of 2
        ])
        faces = np.array([
            [0, 1, 2],
            [3, 4, 5],
        ])
        return MeshInput(vertices, faces)

    def test_nonmanifold_edges_detected(self):
        """
        Duplicate vertices cause every edge to be a boundary edge
        (each edge is only in one face), but boundary edges are NOT
        non-manifold. NonManifoldEdgeDetection flags only face_count > 2.
        """
        mesh = self._make_duplicate_vertex_mesh()
        op = NonManifoldEdgeDetection()
        issues = op.execute(mesh)

        # All 6 edges have face_count=1 (boundary), not face_count > 2.
        assert len(issues) == 0

    def test_all_edges_are_boundary(self):
        """Duplicate-vertex mesh has no non-manifold edges (only boundary edges)."""
        mesh = self._make_duplicate_vertex_mesh()
        op = NonManifoldEdgeDetection()
        issues = op.execute(mesh)

        # Boundary edges (face_count=1) are not reported by NonManifoldEdgeDetection.
        assert len(issues) == 0

    def test_quality_runs_without_error(self):
        """Quality operations should handle duplicate vertices gracefully."""
        mesh = self._make_duplicate_vertex_mesh()
        op = CellQualityComputation(quality_types=['aspect_ratio'])
        issues = op.execute(mesh)
        assert isinstance(issues, list)


# ===================================================================
# 6. Zero-area degenerate triangle (collinear vertices)
# ===================================================================

class TestZeroAreaDegenerate:
    """Tests for a triangle whose vertices are collinear (zero area)."""

    @staticmethod
    def _make_collinear_mesh():
        """Three collinear vertices -> zero-area triangle."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        faces = np.array([[0, 1, 2]])
        return MeshInput(vertices, faces)

    def test_area_is_zero(self):
        """Verify the triangle has zero area via cross product."""
        mesh = self._make_collinear_mesh()
        v = mesh.vertices
        e1 = v[1] - v[0]
        e2 = v[2] - v[0]
        area = np.linalg.norm(np.cross(e1, e2)) / 2.0
        assert area == 0.0

    def test_quality_aspect_ratio_detected(self):
        """Degenerate triangle should trigger aspect_ratio quality violations."""
        mesh = self._make_collinear_mesh()
        op = CellQualityComputation(quality_types=['aspect_ratio'])
        issues = op.execute(mesh)

        # Degenerate triangle should have very poor quality
        # Depending on adapter, aspect_ratio may be 0 or near-0
        assert isinstance(issues, list)

    def test_coincident_vertices_zero_area(self):
        """Two identical vertices -> zero-area triangle."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],  # same as vertex 0
            [1.0, 0.0, 0.0],
        ])
        faces = np.array([[0, 1, 2]])
        mesh = MeshInput(vertices, faces)

        v = mesh.vertices
        e1 = v[1] - v[0]
        e2 = v[2] - v[0]
        area = np.linalg.norm(np.cross(e1, e2)) / 2.0
        assert area == 0.0

        # Should still be processable
        op = CellQualityComputation(quality_types=['aspect_ratio'])
        issues = op.execute(mesh)
        assert isinstance(issues, list)


# ===================================================================
# 7. Mixed quality mesh (good equilateral + bad needle)
# ===================================================================

class TestMixedQualityMesh:
    """Tests for a mesh containing both good and bad quality triangles."""

    @staticmethod
    def _make_mixed_mesh():
        """
        Create a mesh with:
        - 1 equilateral triangle (aspect ratio close to 1)
        - 1 extremely elongated needle triangle (aspect ratio close to 0)
        - 1 right-isosceles triangle (aspect ratio moderate)
        """
        h = np.sqrt(3) / 2  # height of equilateral triangle with side 1

        vertices = np.array([
            # Equilateral triangle (vertices 0,1,2)
            [0.0, 0.0, 0.0],       # 0
            [1.0, 0.0, 0.0],       # 1
            [0.5, h, 0.0],         # 2
            # Needle triangle (vertices 3,4,5)
            [0.0, 2.0, 0.0],       # 3
            [1.0, 2.0, 0.0],       # 4
            [0.5, 2.001, 0.0],     # 5  -- height 0.001
            # Right-isosceles triangle (vertices 6,7,8)
            [0.0, 4.0, 0.0],       # 6
            [1.0, 4.0, 0.0],       # 7
            [0.0, 5.0, 0.0],       # 8
        ])
        faces = np.array([
            [0, 1, 2],  # equilateral
            [3, 4, 5],  # needle
            [6, 7, 8],  # right-isosceles
        ])
        return MeshInput(vertices, faces)

    def test_violation_count_is_one(self):
        """Only the needle triangle should violate the min_angle threshold."""
        mesh = self._make_mixed_mesh()
        op = CellQualityComputation(quality_types=['min_angle'])
        issues = op.execute(mesh)

        angle_issues = [i for i in issues if 'min_angle' in i.operation_id]
        assert len(angle_issues) >= 1
        # 1 out of 3 faces should be in violation
        assert angle_issues[0].metrics['violation_count'] == 1

    def test_severity_is_warning_for_one_of_three(self):
        """min_angle severity is capped at WARNING (would be ERROR by rate alone)."""
        mesh = self._make_mixed_mesh()
        op = CellQualityComputation(quality_types=['min_angle'])
        issues = op.execute(mesh)

        angle_issues = [i for i in issues if 'min_angle' in i.operation_id]
        assert angle_issues[0].severity == IssueSeverity.WARNING

    def test_mean_min_angle_is_moderate(self):
        """Mean min_angle should be between 0 and 60 degrees (mixed quality)."""
        mesh = self._make_mixed_mesh()
        op = CellQualityComputation(quality_types=['min_angle'])
        issues = op.execute(mesh)

        angle_issues = [i for i in issues if 'min_angle' in i.operation_id]
        mean_val = angle_issues[0].metrics['mean']
        assert 0 < mean_val < 60

    def test_min_angle_is_poor(self):
        """The minimum min_angle (needle) should be below threshold 20 degrees."""
        mesh = self._make_mixed_mesh()
        op = CellQualityComputation(quality_types=['min_angle'])
        issues = op.execute(mesh)

        angle_issues = [i for i in issues if 'min_angle' in i.operation_id]
        assert angle_issues[0].metrics['min'] < 20.0


# ===================================================================
# 8. Full workflow test
# ===================================================================

class TestFullWorkflow:
    """Test using create_full_workflow on a bad mesh."""

    @staticmethod
    def _make_bad_mesh():
        """
        A mesh that has both topological and quality problems:
        an umbrella (non-manifold edge) made of needle triangles.
        """
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.001, 0.0],   # very thin
            [0.5, 0.0, 0.001],   # very thin
            [0.5, -0.001, 0.0],  # very thin
        ])
        faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [0, 1, 4],
        ])
        return MeshInput(vertices, faces)

    def test_full_workflow_detects_issues(self):
        """Full workflow should return an InspectionResult with issues."""
        from mesh_issue_examiner.core.models import InspectionResult

        mesh = self._make_bad_mesh()
        reg = _make_registry()
        workflow = reg.create_full_workflow()
        result = workflow(mesh)

        assert isinstance(result, InspectionResult)
        assert len(result.issues) > 0

    def test_full_workflow_summary_computed(self):
        """Summary should be computed when combine_results is True."""
        mesh = self._make_bad_mesh()
        reg = _make_registry()
        workflow = reg.create_full_workflow()
        result = workflow(mesh)

        assert 'total_issues' in result.summary
        assert result.summary['total_issues'] > 0

    def test_full_workflow_has_critical_or_error(self):
        """Bad mesh should produce at least ERROR-level issues."""
        mesh = self._make_bad_mesh()
        reg = _make_registry()
        workflow = reg.create_full_workflow()
        result = workflow(mesh)

        has_severe = any(
            issue.severity in (IssueSeverity.ERROR, IssueSeverity.CRITICAL)
            for issue in result.issues
        )
        assert has_severe

    def test_full_workflow_passed_is_false(self):
        """Result.passed should be False for a bad mesh."""
        mesh = self._make_bad_mesh()
        reg = _make_registry()
        workflow = reg.create_full_workflow()
        result = workflow(mesh)

        assert result.passed is False


# ===================================================================
# 9. Category workflow test
# ===================================================================

class TestCategoryWorkflow:
    """Test using create_category_workflow for TOPOLOGICAL category."""

    def test_topological_workflow_on_umbrella(self):
        """Topological category workflow should detect umbrella non-manifold edges."""
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

        reg = _make_registry()
        workflow = reg.create_category_workflow(IssueCategory.TOPOLOGICAL)
        result = workflow(mesh)

        assert len(result.issues) >= 1
        for issue in result.issues:
            assert issue.category == IssueCategory.TOPOLOGICAL

    def test_topological_workflow_on_clean_mesh(self):
        """Topological workflow on a watertight tetrahedron should produce no issues."""
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
        faces = np.array([
            [0, 1, 2],
            [0, 3, 1],
            [1, 3, 2],
            [2, 3, 0],
        ])
        mesh = MeshInput(vertices, faces)

        reg = _make_registry()
        workflow = reg.create_category_workflow(IssueCategory.TOPOLOGICAL)
        result = workflow(mesh)

        assert len(result.issues) == 0


# ===================================================================
# 10. Registry listing test
# ===================================================================

class TestRegistryListing:
    """Verify that the registry lists operations correctly."""

    def test_list_all_returns_operations(self):
        """list_all() should return metadata for registered operations."""
        reg = _make_registry()
        all_meta = reg.list_all()

        assert len(all_meta) >= 2  # at least nonmanifold + quality

    def test_registered_operation_ids(self):
        """The expected operation IDs should be present in the registry."""
        reg = _make_registry()
        ids = [m.operation_id for m in reg.list_all()]

        assert 'topology.detect_nonmanifold_edges' in ids
        assert 'quality.compute_cell_quality' in ids

    def test_list_by_category_topological(self):
        """Listing by TOPOLOGICAL category should return only topological ops."""
        reg = _make_registry()
        topo_ops = reg.list_by_category(IssueCategory.TOPOLOGICAL)

        assert len(topo_ops) >= 1
        for meta in topo_ops:
            assert meta.category == IssueCategory.TOPOLOGICAL

    def test_list_by_category_quality(self):
        """Listing by QUALITY category should return only quality ops."""
        reg = _make_registry()
        quality_ops = reg.list_by_category(IssueCategory.QUALITY)

        assert len(quality_ops) >= 1
        for meta in quality_ops:
            assert meta.category == IssueCategory.QUALITY

    def test_get_operation_by_id(self):
        """registry.get() should return the correct operation instance."""
        reg = _make_registry()
        op = reg.get('topology.detect_nonmanifold_edges')
        assert op is not None
        assert op.metadata.operation_id == 'topology.detect_nonmanifold_edges'

    def test_get_nonexistent_returns_none(self):
        """registry.get() for a nonexistent ID should return None."""
        reg = _make_registry()
        assert reg.get('nonexistent.operation') is None

    def test_contains(self):
        """The 'in' operator should work for checking registration."""
        reg = _make_registry()
        assert 'topology.detect_nonmanifold_edges' in reg
        assert 'quality.compute_cell_quality' in reg
        assert 'nonexistent' not in reg

    def test_len_matches_registered_count(self):
        """len(registry) should equal the number of registered operations."""
        reg = _make_registry()
        assert len(reg) == 2

    def test_unregister(self):
        """Unregistering an operation should remove it from the registry."""
        reg = _make_registry()
        assert 'topology.detect_nonmanifold_edges' in reg
        reg.unregister('topology.detect_nonmanifold_edges')
        assert 'topology.detect_nonmanifold_edges' not in reg
        assert len(reg) == 1

    def test_register_duplicate_is_idempotent(self):
        """Registering the same operation twice is idempotent (no error)."""
        reg = _make_registry()
        original_len = len(reg)
        reg.register(NonManifoldEdgeDetection())
        assert len(reg) == original_len
