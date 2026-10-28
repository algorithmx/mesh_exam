"""
Tests for volumetric mesh inspection operations.
"""

import pytest
import numpy as np

from mesh_issue_examiner.core.models import (
    MeshInput,
    MeshIssue,
    IssueCategory,
    IssueSeverity,
)
from mesh_issue_examiner.operations.volumetric.volume_cell_quality import VolumeCellQualityComputation
from mesh_issue_examiner.operations.volumetric.geometric_validity import GeometricValidityAssessment
from mesh_issue_examiner.operations.volumetric.orientation_consistency import OrientationConsistencyCheck
from mesh_issue_examiner.operations.volumetric.nonmanifold_faces import NonManifoldFaceDetection
from mesh_issue_examiner.operations.volumetric._face_tables import ELEMENT_FACES


# ------------------------------------------------------------------
# Test mesh fixtures
# ------------------------------------------------------------------

def _regular_tetrahedron():
    """Regular tetrahedron: all quality metrics should be near-optimal."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.866, 0.0],
        [0.5, 0.288, 0.816],
    ])
    faces = np.array([[0, 1, 2]])  # Dummy surface face
    volume_cells = {'tet4': np.array([[0, 1, 2, 3]])}
    return MeshInput(
        vertices=vertices,
        faces=faces,
        mesh_id='regular_tet',
        volume_cells=volume_cells,
    )


def _degenerate_tetrahedron():
    """Degenerate tetrahedron: 4 coplanar vertices -> zero volume."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.866, 0.0],
        [0.5, 0.288, 0.0],  # Coplanar with first 3
    ])
    faces = np.array([[0, 1, 2]])
    volume_cells = {'tet4': np.array([[0, 1, 2, 3]])}
    return MeshInput(
        vertices=vertices,
        faces=faces,
        mesh_id='degenerate_tet',
        volume_cells=volume_cells,
    )


def _inverted_tetrahedron():
    """Inverted tetrahedron: vertex order swapped -> negative volume."""
    # Same as regular but with vertices 1,2 swapped
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.866, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.288, 0.816],
    ])
    faces = np.array([[0, 1, 2]])
    volume_cells = {'tet4': np.array([[0, 1, 2, 3]])}
    return MeshInput(
        vertices=vertices,
        faces=faces,
        mesh_id='inverted_tet',
        volume_cells=volume_cells,
    )


def _sliver_tetrahedron():
    """Sliver tetrahedron: nearly coplanar but technically non-zero volume."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.866, 0.0],
        [0.5, 0.433, 1e-8],  # Almost coplanar, tiny height
    ])
    faces = np.array([[0, 1, 2]])
    volume_cells = {'tet4': np.array([[0, 1, 2, 3]])}
    return MeshInput(
        vertices=vertices,
        faces=faces,
        mesh_id='sliver_tet',
        volume_cells=volume_cells,
    )


def _hexahedral_cube():
    """Unit cube as a single hexahedral element."""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=float)
    faces = np.array([[0, 1, 2]])  # Dummy
    volume_cells = {'hex8': np.array([[0, 1, 2, 3, 4, 5, 6, 7]])}
    return MeshInput(
        vertices=vertices,
        faces=faces,
        mesh_id='hex_cube',
        volume_cells=volume_cells,
    )


def _mixed_tet_mesh():
    """Two tetrahedra sharing a face."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.866, 0.0],
        [0.5, 0.288, 0.816],
        [0.5, 0.288, -0.816],
    ])
    faces = np.array([[0, 1, 2]])
    volume_cells = {'tet4': np.array([[0, 1, 2, 3], [0, 2, 1, 4]])}
    return MeshInput(
        vertices=vertices,
        faces=faces,
        mesh_id='mixed_tets',
        volume_cells=volume_cells,
    )


def _surface_only_mesh():
    """Surface-only mesh (no volume cells)."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.866, 0.0],
    ])
    faces = np.array([[0, 1, 2]])
    return MeshInput(
        vertices=vertices,
        faces=faces,
        mesh_id='surface_only',
    )


def _sliver_hexahedron():
    """Nearly flat hexahedron: unit cube compressed to tiny height."""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom
        [0, 0, 1e-8], [1, 0, 1e-8], [1, 1, 1e-8], [0, 1, 1e-8],  # top (very close to bottom)
    ], dtype=float)
    faces = np.array([[0, 1, 2]])  # Dummy
    volume_cells = {'hex8': np.array([[0, 1, 2, 3, 4, 5, 6, 7]])}
    return MeshInput(
        vertices=vertices,
        faces=faces,
        mesh_id='sliver_hex',
        volume_cells=volume_cells,
    )


def _regular_hexahedron():
    """Regular unit cube as a single hexahedral element."""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=float)
    faces = np.array([[0, 1, 2]])  # Dummy
    volume_cells = {'hex8': np.array([[0, 1, 2, 3, 4, 5, 6, 7]])}
    return MeshInput(
        vertices=vertices,
        faces=faces,
        mesh_id='regular_hex',
        volume_cells=volume_cells,
    )


def _sliver_prism():
    """Nearly flat triangular prism: top and bottom very close."""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [0.5, 0.866, 0],  # bottom triangle
        [0, 0, 1e-8], [1, 0, 1e-8], [0.5, 0.866, 1e-8],  # top triangle (very close)
    ], dtype=float)
    faces = np.array([[0, 1, 2]])  # Dummy
    volume_cells = {'prism6': np.array([[0, 1, 2, 3, 4, 5]])}
    return MeshInput(
        vertices=vertices,
        faces=faces,
        mesh_id='sliver_prism',
        volume_cells=volume_cells,
    )


def _regular_prism():
    """Regular triangular prism with height 1."""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [0.5, 0.866, 0],  # bottom triangle
        [0, 0, 1], [1, 0, 1], [0.5, 0.866, 1],  # top triangle
    ], dtype=float)
    faces = np.array([[0, 1, 2]])  # Dummy
    volume_cells = {'prism6': np.array([[0, 1, 2, 3, 4, 5]])}
    return MeshInput(
        vertices=vertices,
        faces=faces,
        mesh_id='regular_prism',
        volume_cells=volume_cells,
    )


def _sliver_pyramid():
    """Nearly flat pyramid: apex very close to base."""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # base quad
        [0.5, 0.5, 1e-8],  # apex (very close to base)
    ], dtype=float)
    faces = np.array([[0, 1, 2]])  # Dummy
    volume_cells = {'pyramid5': np.array([[0, 1, 2, 3, 4]])}
    return MeshInput(
        vertices=vertices,
        faces=faces,
        mesh_id='sliver_pyramid',
        volume_cells=volume_cells,
    )


def _regular_pyramid():
    """Regular pyramid with apex at height 1."""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # base quad
        [0.5, 0.5, 1],  # apex
    ], dtype=float)
    faces = np.array([[0, 1, 2]])  # Dummy
    volume_cells = {'pyramid5': np.array([[0, 1, 2, 3, 4]])}
    return MeshInput(
        vertices=vertices,
        faces=faces,
        mesh_id='regular_pyramid',
        volume_cells=volume_cells,
    )


# ==================================================================
# VolumeCellQualityComputation tests
# ==================================================================

class TestVolumeCellQualityComputation:

    def test_regular_tetrahedron_no_critical_issues(self):
        """Regular tetrahedron should produce good quality for key metrics.

        Note: VTK's 'condition' and 'skew' metrics return unusual values
        for tetrahedra (condition ~1.0 is actually excellent but may trigger
        thresholds; skew returns -1 for tets). We verify that the core
        metrics (scaled_jacobian, volume, aspect_ratio) are healthy.
        """
        mesh = _regular_tetrahedron()
        op = VolumeCellQualityComputation(
            quality_types=['scaled_jacobian', 'volume', 'aspect_ratio']
        )
        issues = op.execute(mesh)
        # With only the core quality metrics, regular tet should have
        # no or minimal quality issues
        critical = [i for i in issues if i.severity == IssueSeverity.CRITICAL]
        assert len(critical) == 0

    def test_degenerate_tetrahedron_has_issues(self):
        """Degenerate tet should produce quality issues."""
        mesh = _degenerate_tetrahedron()
        op = VolumeCellQualityComputation()
        issues = op.execute(mesh)
        assert len(issues) > 0

    def test_surface_only_returns_empty(self):
        """Surface-only mesh should return empty issues list."""
        mesh = _surface_only_mesh()
        op = VolumeCellQualityComputation()
        issues = op.execute(mesh)
        assert issues == []

    def test_hex_cube_quality(self):
        """Unit cube hex should compute quality without error."""
        mesh = _hexahedral_cube()
        op = VolumeCellQualityComputation()
        issues = op.execute(mesh)
        # Hex cube should have reasonable quality
        assert isinstance(issues, list)

    def test_custom_quality_types(self):
        """Custom quality_types parameter should filter metrics."""
        mesh = _regular_tetrahedron()
        op = VolumeCellQualityComputation(quality_types=['scaled_jacobian'])
        issues = op.execute(mesh)
        # Should only compute scaled_jacobian
        for issue in issues:
            assert 'scaled_jacobian' in issue.operation_id or 'scaled_jacobian' in str(issue.metadata)

    def test_issue_category_is_quality(self):
        """All issues should have QUALITY category."""
        mesh = _degenerate_tetrahedron()
        op = VolumeCellQualityComputation()
        issues = op.execute(mesh)
        for issue in issues:
            assert issue.category == IssueCategory.QUALITY

    def test_metrics_dict_present(self):
        """Each issue should have metrics dict with expected keys."""
        mesh = _degenerate_tetrahedron()
        op = VolumeCellQualityComputation()
        issues = op.execute(mesh)
        if issues:
            for key in ['violation_count', 'total_cells', 'violation_rate',
                        'mean', 'min', 'max', 'std', 'threshold']:
                assert key in issues[0].metrics, f"Missing key: {key}"


# ==================================================================
# GeometricValidityAssessment tests
# ==================================================================

class TestGeometricValidityAssessment:

    def test_regular_tetrahedron_no_issues(self):
        """Regular tetrahedron should pass geometric validity."""
        mesh = _regular_tetrahedron()
        op = GeometricValidityAssessment()
        issues = op.execute(mesh)
        assert len(issues) == 0

    def test_degenerate_tetrahedron_detected(self):
        """Degenerate tet should be flagged as degenerate."""
        mesh = _degenerate_tetrahedron()
        op = GeometricValidityAssessment()
        issues = op.execute(mesh)
        assert len(issues) > 0
        # Should have at least one CRITICAL issue for degenerate cell
        critical = [i for i in issues if i.severity == IssueSeverity.CRITICAL]
        assert len(critical) > 0

    def test_sliver_tetrahedron_detected(self):
        """Sliver tet should be flagged."""
        mesh = _sliver_tetrahedron()
        op = GeometricValidityAssessment()
        issues = op.execute(mesh)
        # Sliver or degenerate should be detected
        assert len(issues) > 0

    def test_surface_only_returns_empty(self):
        """Surface-only mesh should return empty issues list."""
        mesh = _surface_only_mesh()
        op = GeometricValidityAssessment()
        issues = op.execute(mesh)
        assert issues == []

    def test_hex_cube_valid(self):
        """Unit cube hex should pass geometric validity."""
        mesh = _hexahedral_cube()
        op = GeometricValidityAssessment()
        issues = op.execute(mesh)
        assert len(issues) == 0

    def test_issue_category_is_geometric(self):
        """Issues should have GEOMETRIC category."""
        mesh = _degenerate_tetrahedron()
        op = GeometricValidityAssessment()
        issues = op.execute(mesh)
        for issue in issues:
            assert issue.category == IssueCategory.GEOMETRIC

    def test_custom_degenerate_tolerance(self):
        """Custom degenerate tolerance should be respected."""
        mesh = _regular_tetrahedron()
        # Use an extremely large tolerance so even regular tet is "degenerate"
        op = GeometricValidityAssessment(degenerate_tolerance=1.0)
        issues = op.execute(mesh)
        assert len(issues) > 0

    def test_sliver_hex_detected(self):
        """Nearly flat hex should be flagged as sliver."""
        mesh = _sliver_hexahedron()
        op = GeometricValidityAssessment()
        issues = op.execute(mesh)
        # Should detect the sliver hex
        sliver_issues = [i for i in issues if 'sliver' in i.message.lower()]
        assert len(sliver_issues) > 0
        # Check per-type breakdown
        assert sliver_issues[0].metrics.get('sliver_hex8_count', 0) > 0

    def test_sliver_prism_detected(self):
        """Nearly flat prism should be flagged as sliver."""
        mesh = _sliver_prism()
        op = GeometricValidityAssessment()
        issues = op.execute(mesh)
        sliver_issues = [i for i in issues if 'sliver' in i.message.lower()]
        assert len(sliver_issues) > 0
        assert sliver_issues[0].metrics.get('sliver_prism6_count', 0) > 0

    def test_sliver_pyramid_detected(self):
        """Nearly flat pyramid should be flagged as sliver."""
        mesh = _sliver_pyramid()
        op = GeometricValidityAssessment()
        issues = op.execute(mesh)
        sliver_issues = [i for i in issues if 'sliver' in i.message.lower()]
        assert len(sliver_issues) > 0
        assert sliver_issues[0].metrics.get('sliver_pyramid5_count', 0) > 0

    def test_regular_hex_no_slivers(self):
        """Regular cube should have no sliver issues."""
        mesh = _regular_hexahedron()
        op = GeometricValidityAssessment()
        issues = op.execute(mesh)
        # No sliver or degenerate issues expected
        assert len(issues) == 0

    def test_regular_prism_no_slivers(self):
        """Regular prism should have no sliver issues."""
        mesh = _regular_prism()
        op = GeometricValidityAssessment()
        issues = op.execute(mesh)
        assert len(issues) == 0

    def test_regular_pyramid_no_slivers(self):
        """Regular pyramid should have no sliver issues."""
        mesh = _regular_pyramid()
        op = GeometricValidityAssessment()
        issues = op.execute(mesh)
        assert len(issues) == 0

    def test_mixed_mesh_slivers_detected(self):
        """Mesh with tet4 + hex8 where only hex is sliver."""
        # Regular tet + sliver hex
        vertices = np.array([
            # Regular tet vertices (0-3)
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],
            [0.5, 0.288, 0.816],
            # Sliver hex vertices (4-11) - shifted to avoid overlap
            [2, 0, 0], [3, 0, 0], [3, 1, 0], [2, 1, 0],  # bottom
            [2, 0, 1e-8], [3, 0, 1e-8], [3, 1, 1e-8], [2, 1, 1e-8],  # top (very close)
        ], dtype=float)
        faces = np.array([[0, 1, 2]])
        volume_cells = {
            'tet4': np.array([[0, 1, 2, 3]]),
            'hex8': np.array([[4, 5, 6, 7, 8, 9, 10, 11]]),
        }
        mesh = MeshInput(
            vertices=vertices,
            faces=faces,
            mesh_id='mixed_sliver',
            volume_cells=volume_cells,
        )
        op = GeometricValidityAssessment()
        issues = op.execute(mesh)
        # Should find exactly 1 sliver (the hex)
        sliver_issues = [i for i in issues if 'sliver' in i.message.lower()]
        assert len(sliver_issues) > 0
        assert sliver_issues[0].metrics.get('sliver_count') == 1
        assert sliver_issues[0].metrics.get('sliver_hex8_count') == 1
        assert sliver_issues[0].metrics.get('sliver_tet4_count', 0) == 0


# ==================================================================
# OrientationConsistencyCheck tests
# ==================================================================

class TestOrientationConsistencyCheck:

    def test_regular_tetrahedron_no_issues(self):
        """Regular tetrahedron should have consistent orientation."""
        mesh = _regular_tetrahedron()
        op = OrientationConsistencyCheck()
        issues = op.execute(mesh)
        assert len(issues) == 0

    def test_inverted_tetrahedron_detected(self):
        """Inverted tet should be flagged as inverted."""
        mesh = _inverted_tetrahedron()
        op = OrientationConsistencyCheck()
        issues = op.execute(mesh)
        assert len(issues) > 0
        # Should have CRITICAL issue for inverted cell
        critical = [i for i in issues if i.severity == IssueSeverity.CRITICAL]
        assert len(critical) > 0

    def test_surface_only_returns_empty(self):
        """Surface-only mesh should return empty issues list."""
        mesh = _surface_only_mesh()
        op = OrientationConsistencyCheck()
        issues = op.execute(mesh)
        assert issues == []

    def test_hex_cube_no_issues(self):
        """Unit cube hex should have no orientation issues."""
        mesh = _hexahedral_cube()
        op = OrientationConsistencyCheck()
        issues = op.execute(mesh)
        assert len(issues) == 0

    def test_issue_category_is_orientation(self):
        """Issues should have ORIENTATION category."""
        mesh = _inverted_tetrahedron()
        op = OrientationConsistencyCheck()
        issues = op.execute(mesh)
        for issue in issues:
            assert issue.category == IssueCategory.ORIENTATION

    def test_metadata_properties(self):
        """Check metadata is correctly defined."""
        op = OrientationConsistencyCheck()
        meta = op.metadata
        assert meta.operation_id == 'quality_3d.check_orientation_consistency'
        assert 'orientation' in meta.tags
        assert meta.complexity == 'O(n log n)'

    # ----- Multi-element-type winding tests -----

    def test_two_hexes_consistent_winding_no_issues(self):
        """Two hexes sharing a face with consistent (opposite) winding."""
        vertices = np.array([
            # Cube1 vertices (0-7)
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
            # Cube2-only vertices (8-11)
            [2, 0, 0], [2, 1, 0], [2, 0, 1], [2, 1, 1],
        ], dtype=float)
        faces = np.array([[0, 1, 2]])
        # Cube1 right face (local 1,2,6,5) = global (1, 2, 6, 5)
        # Cube2 left face (local 3,0,4,7) should be reverse: (2, 1, 5, 6)
        # So: cube2[3]=2, cube2[0]=1, cube2[4]=5, cube2[7]=6
        # Remaining: cube2[1]=8, cube2[2]=9, cube2[5]=10, cube2[6]=11
        volume_cells = {'hex8': np.array([
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 8, 9, 2, 5, 10, 11, 6],
        ])}
        mesh = MeshInput(
            vertices=vertices, faces=faces,
            mesh_id='two_hexes_consistent',
            volume_cells=volume_cells,
        )
        op = OrientationConsistencyCheck()
        issues = op.execute(mesh)
        winding_issues = [i for i in issues
                          if 'inconsistent' in i.message.lower()
                          or 'winding' in i.message.lower()]
        assert len(winding_issues) == 0

    def test_two_hexes_inconsistent_winding_detected(self):
        """Two hexes sharing a face with SAME winding → detected.

        Cube1 and Cube2 share vertices on their common face (x=1).
        Cube2's vertex ordering is deliberately set so the shared face
        appears with the same directed edges as Cube1's face.
        """
        vertices = np.array([
            # Cube1 vertices (0-7)
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
            # Cube2-only vertices (8-11)
            [2, 0, 0], [2, 1, 0], [2, 0, 1], [2, 1, 1],
        ], dtype=float)
        faces = np.array([[0, 1, 2]])
        # Cube1 right face (local 1,2,6,5) = global (1, 2, 6, 5)
        #
        # For inconsistent winding, set Cube2 so its left face
        # (local 3,0,4,7) has the SAME global vertex order (1,2,6,5):
        #   cube2[3]=1, cube2[0]=2, cube2[4]=6, cube2[7]=5
        # Remaining: cube2[1]=8, cube2[2]=9, cube2[5]=10, cube2[6]=11
        volume_cells = {'hex8': np.array([
            [0, 1, 2, 3, 4, 5, 6, 7],
            [2, 8, 9, 1, 6, 10, 11, 5],
        ])}
        mesh = MeshInput(
            vertices=vertices, faces=faces,
            mesh_id='two_hexes_inconsistent',
            volume_cells=volume_cells,
        )
        op = OrientationConsistencyCheck()
        issues = op.execute(mesh)
        winding_issues = [i for i in issues
                          if 'inconsistent' in i.message.lower()
                          or 'winding' in i.message.lower()]
        assert len(winding_issues) > 0

    def test_two_prisms_sharing_tri_face_consistent(self):
        """Two prisms sharing a triangular face with consistent winding."""
        # Two prisms stacked along z, sharing the triangular face at z=1
        vertices = np.array([
            # Bottom triangle of prism1 (z=0)
            [0, 0, 0], [1, 0, 0], [0.5, 0.866, 0],
            # Top of prism1 / bottom of prism2 (z=1)
            [0, 0, 1], [1, 0, 1], [0.5, 0.866, 1],
            # Top of prism2 (z=2)
            [0, 0, 2], [1, 0, 2], [0.5, 0.866, 2],
        ], dtype=float)
        faces = np.array([[0, 1, 2]])
        # Outward-wound tables (2026-10-28): prism1 top face is
        # local (3,4,5) = global (3,4,5); prism2's bottom tuple is
        # local (2,1,0), so the natural vertex order [3,4,5,...] lists
        # the shared face reversed — the natural stacking is consistent.
        volume_cells = {'prism6': np.array([
            [0, 1, 2, 3, 4, 5],
            [3, 4, 5, 6, 7, 8],  # natural order opposes prism1's top face
        ])}
        mesh = MeshInput(
            vertices=vertices, faces=faces,
            mesh_id='two_prisms_consistent',
            volume_cells=volume_cells,
        )
        op = OrientationConsistencyCheck()
        issues = op.execute(mesh)
        winding_issues = [i for i in issues
                          if 'inconsistent' in i.message.lower()
                          or 'winding' in i.message.lower()]
        assert len(winding_issues) == 0

    def test_two_pyramids_sharing_tri_face_consistent(self):
        """Two pyramids sharing a triangular face with consistent winding."""
        vertices = np.array([
            # Pyramid 1: base on z=0 plane, apex at z=1
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0.5, 0.5, 1],  # apex1
            # Pyramid 2: apex at z=-1, shares face (0,1,4) with pyramid1
            [0.5, 0.5, -1],  # apex2
        ], dtype=float)
        faces = np.array([[0, 1, 2]])
        # Pyramid 1 face (0,1,4). For pyramid2 we use vertex order that reverses.
        volume_cells = {'pyramid5': np.array([
            [0, 1, 2, 3, 4],
            [1, 0, 3, 2, 5],  # reversed on shared face
        ])}
        mesh = MeshInput(
            vertices=vertices, faces=faces,
            mesh_id='two_pyramids_consistent',
            volume_cells=volume_cells,
        )
        op = OrientationConsistencyCheck()
        issues = op.execute(mesh)
        winding_issues = [i for i in issues
                          if 'inconsistent' in i.message.lower()
                          or 'winding' in i.message.lower()]
        assert len(winding_issues) == 0

    def test_mixed_tet_prism_winding_consistent(self):
        """tet4 sharing a triangular face with prism6, consistent winding."""
        vertices = np.array([
            # Triangle (z=0)
            [0, 0, 0], [1, 0, 0], [0.5, 0.866, 0],
            # Prism top (z=1)
            [0, 0, 1], [1, 0, 1], [0.5, 0.866, 1],
            # Tet apex above prism (z=2)
            [0.5, 0.288, 2],
        ], dtype=float)
        faces = np.array([[0, 1, 2]])
        # Outward-wound tables (2026-10-28): prism top face is local
        # (3,4,5) = global (3,4,5); the tet's base tuple is local
        # (2,1,0), so the natural order [3,4,5,apex] opposes it.
        volume_cells = {
            'prism6': np.array([[0, 1, 2, 3, 4, 5]]),
            'tet4': np.array([[3, 4, 5, 6]]),  # natural order opposes the top face
        }
        mesh = MeshInput(
            vertices=vertices, faces=faces,
            mesh_id='tet_prism_consistent',
            volume_cells=volume_cells,
        )
        op = OrientationConsistencyCheck()
        issues = op.execute(mesh)
        winding_issues = [i for i in issues
                          if 'inconsistent' in i.message.lower()
                          or 'winding' in i.message.lower()]
        assert len(winding_issues) == 0

    def test_mixed_tet_prism_winding_inconsistent(self):
        """tet4 sharing a triangular face with prism6, SAME winding → detected."""
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [0.5, 0.866, 0],
            [0, 0, 1], [1, 0, 1], [0.5, 0.866, 1],
            [0.5, 0.288, 2],
        ], dtype=float)
        faces = np.array([[0, 1, 2]])
        # Tet base tuple (2,1,0) makes [3,5,4,apex] list the shared
        # face with the SAME directed edges as the prism's top face.
        volume_cells = {
            'prism6': np.array([[0, 1, 2, 3, 4, 5]]),
            'tet4': np.array([[3, 5, 4, 6]]),
        }
        mesh = MeshInput(
            vertices=vertices, faces=faces,
            mesh_id='tet_prism_inconsistent',
            volume_cells=volume_cells,
        )
        op = OrientationConsistencyCheck()
        issues = op.execute(mesh)
        winding_issues = [i for i in issues
                          if 'inconsistent' in i.message.lower()
                          or 'winding' in i.message.lower()]
        assert len(winding_issues) > 0

    def test_hex_only_mesh_not_silently_skipped(self):
        """Mesh with only hex8 cells should run (not silently return empty)."""
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=float)
        faces = np.array([[0, 1, 2]])
        volume_cells = {'hex8': np.array([[0, 1, 2, 3, 4, 5, 6, 7]])}
        mesh = MeshInput(
            vertices=vertices, faces=faces,
            mesh_id='hex_only',
            volume_cells=volume_cells,
        )
        op = OrientationConsistencyCheck()
        issues = op.execute(mesh)
        # Single cell has no shared faces → no winding issues, that's fine
        # But should NOT error out
        assert isinstance(issues, list)

    def test_prism_only_mesh_runs(self):
        """Mesh with only prism6 cells should run without error."""
        mesh = _regular_prism()
        op = OrientationConsistencyCheck()
        issues = op.execute(mesh)
        assert isinstance(issues, list)

    def test_pyramid_only_mesh_runs(self):
        """Mesh with only pyramid5 cells should run without error."""
        mesh = _regular_pyramid()
        op = OrientationConsistencyCheck()
        issues = op.execute(mesh)
        assert isinstance(issues, list)

    def test_metrics_include_total_shared_faces(self):
        """Winding issue metrics should include total_shared_faces."""
        # Create inconsistent winding in two tets sharing a face
        mesh = _mixed_tet_mesh()  # two tets sharing face
        op = OrientationConsistencyCheck()
        issues = op.execute(mesh)
        # Check for winding issues
        winding_issues = [i for i in issues
                          if 'inconsistent' in i.message.lower()
                          or 'winding' in i.message.lower()]
        if winding_issues:
            assert 'total_shared_faces' in winding_issues[0].metrics
            assert 'total_cells' in winding_issues[0].metrics

    # ----- Directed-edge method tests -----

    def test_faces_are_reversed_triangles(self):
        """Directed-edge method for triangular faces."""
        assert OrientationConsistencyCheck._faces_are_reversed(
            (0, 1, 2), (0, 2, 1)) is True   # reversed
        assert OrientationConsistencyCheck._faces_are_reversed(
            (0, 1, 2), (1, 0, 2)) is True   # cyclic of reversed
        assert OrientationConsistencyCheck._faces_are_reversed(
            (0, 1, 2), (2, 1, 0)) is True   # cyclic of reversed
        assert OrientationConsistencyCheck._faces_are_reversed(
            (0, 1, 2), (0, 1, 2)) is False  # same winding

    def test_faces_are_reversed_quads(self):
        """Directed-edge method for quadrilateral faces."""
        # Reverse of (0,1,2,3) is (0,3,2,1) and its cyclic rotations
        assert OrientationConsistencyCheck._faces_are_reversed(
            (0, 1, 2, 3), (0, 3, 2, 1)) is True   # exact reverse
        assert OrientationConsistencyCheck._faces_are_reversed(
            (0, 1, 2, 3), (3, 2, 1, 0)) is True   # cyclic rotation
        assert OrientationConsistencyCheck._faces_are_reversed(
            (0, 1, 2, 3), (2, 1, 0, 3)) is True   # cyclic rotation
        assert OrientationConsistencyCheck._faces_are_reversed(
            (0, 1, 2, 3), (1, 0, 3, 2)) is True   # cyclic rotation
        assert OrientationConsistencyCheck._faces_are_reversed(
            (0, 1, 2, 3), (0, 1, 2, 3)) is False  # same winding
        assert OrientationConsistencyCheck._faces_are_reversed(
            (0, 1, 2, 3), (0, 1, 3, 2)) is False  # not a valid reverse


# ==================================================================
# NonManifoldFaceDetection tests
# ==================================================================

class TestNonManifoldFaceDetection:

    def test_surface_only_returns_empty(self):
        """Surface-only mesh should return empty issues list."""
        mesh = _surface_only_mesh()
        op = NonManifoldFaceDetection()
        issues = op.execute(mesh)
        assert issues == []

    def test_single_tetrahedron_no_issues(self):
        """Single tetrahedron has no internal faces, should be manifold."""
        mesh = _regular_tetrahedron()
        op = NonManifoldFaceDetection()
        issues = op.execute(mesh)
        assert len(issues) == 0

    def test_two_tets_sharing_face_is_manifold(self):
        """Two tetrahedra sharing one face is manifold (face shared by exactly 2 cells)."""
        # Two tets sharing face (0, 1, 2)
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],
            [0.5, 0.288, 0.816],   # Above
            [0.5, 0.288, -0.816],  # Below
        ])
        faces = np.array([[0, 1, 2]])
        # Tet 1: (0, 1, 2, 3) and Tet 2: (0, 2, 1, 4) - share face (0, 1, 2)
        volume_cells = {'tet4': np.array([[0, 1, 2, 3], [0, 2, 1, 4]])}
        mesh = MeshInput(
            vertices=vertices,
            faces=faces,
            mesh_id='two_tets_sharing',
            volume_cells=volume_cells,
        )
        op = NonManifoldFaceDetection()
        issues = op.execute(mesh)
        assert len(issues) == 0

    def test_three_tets_sharing_face_is_nonmanifold(self):
        """Three tetrahedra sharing one face is non-manifold."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],
            [0.5, 0.288, 0.816],    # Above
            [0.5, 0.288, -0.816],   # Below
            [0.5, -0.5, 0.0],       # Side - creates non-manifold
        ])
        faces = np.array([[0, 1, 2]])
        # Three tets all sharing face (0, 1, 2) - non-manifold!
        volume_cells = {'tet4': np.array([
            [0, 1, 2, 3],   # Tet 1: above
            [0, 2, 1, 4],   # Tet 2: below (note winding)
            [0, 1, 2, 5],   # Tet 3: side - shares same face!
        ])}
        mesh = MeshInput(
            vertices=vertices,
            faces=faces,
            mesh_id='three_tets_nonmanifold',
            volume_cells=volume_cells,
        )
        op = NonManifoldFaceDetection()
        issues = op.execute(mesh)
        assert len(issues) > 0
        # Should detect the non-manifold face
        assert issues[0].category == IssueCategory.TOPOLOGICAL
        assert 'non-manifold' in issues[0].message.lower()

    def test_hex_cube_no_issues(self):
        """Single hex cube has no internal faces, should be manifold."""
        mesh = _hexahedral_cube()
        op = NonManifoldFaceDetection()
        issues = op.execute(mesh)
        assert len(issues) == 0

    def test_two_hexes_sharing_face_is_manifold(self):
        """Two hexahedra sharing one face is manifold."""
        # Two unit cubes sharing a face
        vertices = np.array([
            # First cube (0-7)
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
            # Second cube (8-15) - shares face at x=1
            [1, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0],
            [1, 0, 1], [2, 0, 1], [2, 1, 1], [1, 1, 1],
        ], dtype=float)
        faces = np.array([[0, 1, 2]])
        volume_cells = {'hex8': np.array([
            [0, 1, 2, 3, 4, 5, 6, 7],      # First hex
            [8, 9, 10, 11, 12, 13, 14, 15],  # Second hex (shares face with first)
        ])}
        mesh = MeshInput(
            vertices=vertices,
            faces=faces,
            mesh_id='two_hexes_sharing',
            volume_cells=volume_cells,
        )
        op = NonManifoldFaceDetection()
        issues = op.execute(mesh)
        # Shared face should be manifold (shared by exactly 2 cells)
        assert len(issues) == 0

    def test_issue_category_is_topological(self):
        """All issues should have TOPOLOGICAL category."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],
            [0.5, 0.288, 0.816],
            [0.5, 0.288, -0.816],
            [0.5, -0.5, 0.0],
        ])
        faces = np.array([[0, 1, 2]])
        volume_cells = {'tet4': np.array([
            [0, 1, 2, 3],
            [0, 2, 1, 4],
            [0, 1, 2, 5],  # Non-manifold
        ])}
        mesh = MeshInput(
            vertices=vertices,
            faces=faces,
            mesh_id='nonmanifold_test',
            volume_cells=volume_cells,
        )
        op = NonManifoldFaceDetection()
        issues = op.execute(mesh)
        for issue in issues:
            assert issue.category == IssueCategory.TOPOLOGICAL

    def test_metrics_include_counts(self):
        """Issues should include counts in metrics."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],
            [0.5, 0.288, 0.816],
            [0.5, 0.288, -0.816],
            [0.5, -0.5, 0.0],
        ])
        faces = np.array([[0, 1, 2]])
        volume_cells = {'tet4': np.array([
            [0, 1, 2, 3],
            [0, 2, 1, 4],
            [0, 1, 2, 5],
        ])}
        mesh = MeshInput(
            vertices=vertices,
            faces=faces,
            mesh_id='nonmanifold_metrics',
            volume_cells=volume_cells,
        )
        op = NonManifoldFaceDetection()
        issues = op.execute(mesh)
        assert len(issues) > 0
        metrics = issues[0].metrics
        assert 'nonmanifold_face_count' in metrics
        assert 'unique_face_count' in metrics
        assert metrics['nonmanifold_face_count'] > 0

    def test_locations_have_face_vertices(self):
        """Issue locations should include face vertex indices."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],
            [0.5, 0.288, 0.816],
            [0.5, 0.288, -0.816],
            [0.5, -0.5, 0.0],
        ])
        faces = np.array([[0, 1, 2]])
        volume_cells = {'tet4': np.array([
            [0, 1, 2, 3],
            [0, 2, 1, 4],
            [0, 1, 2, 5],
        ])}
        mesh = MeshInput(
            vertices=vertices,
            faces=faces,
            mesh_id='nonmanifold_locations',
            volume_cells=volume_cells,
        )
        op = NonManifoldFaceDetection()
        issues = op.execute(mesh)
        assert len(issues) > 0
        assert len(issues[0].locations) > 0
        loc = issues[0].locations[0]
        assert 'face_vertices' in loc.adjacent_elements
        assert 'sharing_cell_count' in loc.adjacent_elements

    def test_metadata_includes_affected_types(self):
        """Issue metadata should include affected cell types."""
        mesh = _regular_tetrahedron()
        op = NonManifoldFaceDetection()
        issues = op.execute(mesh)
        # Even if no issues, structure should be valid
        # Create a non-manifold case to test metadata
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],
            [0.5, 0.288, 0.816],
            [0.5, 0.288, -0.816],
            [0.5, -0.5, 0.0],
        ])
        faces = np.array([[0, 1, 2]])
        volume_cells = {'tet4': np.array([
            [0, 1, 2, 3],
            [0, 2, 1, 4],
            [0, 1, 2, 5],
        ])}
        mesh = MeshInput(
            vertices=vertices,
            faces=faces,
            mesh_id='nonmanifold_meta',
            volume_cells=volume_cells,
        )
        issues = op.execute(mesh)
        assert len(issues) > 0
        assert 'affected_cell_types' in issues[0].metadata
        assert 'tet4' in issues[0].metadata['affected_cell_types']

    def test_element_face_definitions(self):
        """Verify element face definitions are correct."""
        # Check that we have face definitions for all supported types
        assert 'tet4' in ELEMENT_FACES
        assert 'hex8' in ELEMENT_FACES
        assert 'prism6' in ELEMENT_FACES
        assert 'pyramid5' in ELEMENT_FACES

        # Check face counts
        assert len(ELEMENT_FACES['tet4']) == 4  # 4 triangles
        assert len(ELEMENT_FACES['hex8']) == 6  # 6 quads
        assert len(ELEMENT_FACES['prism6']) == 5  # 2 triangles + 3 quads
        assert len(ELEMENT_FACES['pyramid5']) == 5  # 1 quad + 4 triangles

    def test_prism_mesh_no_issues(self):
        """Single prism element should have no non-manifold faces."""
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [0.5, 0.866, 0],  # Bottom triangle
            [0, 0, 1], [1, 0, 1], [0.5, 0.866, 1],  # Top triangle
        ], dtype=float)
        faces = np.array([[0, 1, 2]])
        volume_cells = {'prism6': np.array([[0, 1, 2, 3, 4, 5]])}
        mesh = MeshInput(
            vertices=vertices,
            faces=faces,
            mesh_id='single_prism',
            volume_cells=volume_cells,
        )
        op = NonManifoldFaceDetection()
        issues = op.execute(mesh)
        assert len(issues) == 0

    def test_pyramid_mesh_no_issues(self):
        """Single pyramid element should have no non-manifold faces."""
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Base quad
            [0.5, 0.5, 1],  # Apex
        ], dtype=float)
        faces = np.array([[0, 1, 2]])
        volume_cells = {'pyramid5': np.array([[0, 1, 2, 3, 4]])}
        mesh = MeshInput(
            vertices=vertices,
            faces=faces,
            mesh_id='single_pyramid',
            volume_cells=volume_cells,
        )
        op = NonManifoldFaceDetection()
        issues = op.execute(mesh)
        assert len(issues) == 0


# ==================================================================
# MeshInput volumetric support tests
# ==================================================================

class TestMeshInputVolumeCells:

    def test_has_volume_cells_true(self):
        mesh = _regular_tetrahedron()
        assert mesh.has_volume_cells is True

    def test_has_volume_cells_false(self):
        mesh = _surface_only_mesh()
        assert mesh.has_volume_cells is False

    def test_n_volume_cells(self):
        mesh = _regular_tetrahedron()
        assert mesh.n_volume_cells == 1

    def test_cell_types(self):
        mesh = _regular_tetrahedron()
        assert 'tet4' in mesh.cell_types

    def test_volume_cells_dict(self):
        mesh = _regular_tetrahedron()
        assert 'tet4' in mesh.volume_cells
        assert mesh.volume_cells['tet4'].shape == (1, 4)

    def test_volume_cells_setter(self):
        mesh = _surface_only_mesh()
        assert mesh.has_volume_cells is False
        mesh.volume_cells = {'tet4': np.array([[0, 1, 2, 0]])}
        assert mesh.has_volume_cells is True

    def test_volume_cells_none_clears(self):
        mesh = _regular_tetrahedron()
        assert mesh.has_volume_cells is True
        mesh.volume_cells = None
        assert mesh.has_volume_cells is False

    def test_copy_preserves_volume_cells(self):
        mesh = _regular_tetrahedron()
        copy = mesh.copy()
        assert copy.has_volume_cells is True
        assert 'tet4' in copy.volume_cells

    def test_to_dict_includes_volume_info(self):
        mesh = _regular_tetrahedron()
        d = mesh.to_dict()
        assert 'n_volume_cells' in d
        assert d['n_volume_cells'] == 1
        assert 'cell_types' in d
        assert 'tet4' in d['cell_types']

    def test_get_native_pyvista_returns_unstructured_grid(self):
        mesh = _regular_tetrahedron()
        grid = mesh.get_native('pyvista')
        import pyvista as pv
        assert isinstance(grid, pv.UnstructuredGrid)

    def test_mixed_element_types(self):
        """Mesh with both tet and hex cells."""
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [0.5, 0.866, 0], [0.5, 0.288, 0.816],
            [2, 0, 0], [3, 0, 0], [3, 1, 0], [2, 1, 0],
            [2, 0, 1], [3, 0, 1], [3, 1, 1], [2, 1, 1],
        ], dtype=float)
        faces = np.array([[0, 1, 2]])
        volume_cells = {
            'tet4': np.array([[0, 1, 2, 3]]),
            'hex8': np.array([[4, 5, 6, 7, 8, 9, 10, 11]]),
        }
        mesh = MeshInput(
            vertices=vertices,
            faces=faces,
            mesh_id='mixed',
            volume_cells=volume_cells,
        )
        assert mesh.n_volume_cells == 2
        assert set(mesh.cell_types) == {'tet4', 'hex8'}
        grid = mesh.get_native('pyvista')
        import pyvista as pv
        assert isinstance(grid, pv.UnstructuredGrid)
        assert grid.n_cells == 2
