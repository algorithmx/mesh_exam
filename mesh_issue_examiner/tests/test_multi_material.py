"""
Tests for multi-material interface checks (Stage 3).

Tests cover:
    1. MeshInput material support (fields, properties, validation)
    2. MaterialInterfaceDetection operation
    3. InterfaceQualityCheck operation
    4. PerMaterialStatistics operation
    5. Integration: all three operations on a multi-material mesh
"""

import numpy as np
import pytest

from ..core.models import (
    MeshInput,
    MeshIssue,
    IssueSeverity,
    IssueCategory,
)
from ..operations.volumetric.multi_material import (
    MaterialInterfaceDetection,
    InterfaceQualityCheck,
    PerMaterialStatistics,
    _compute_tet_volume,
    _compute_scaled_jacobian_tet,
    _build_face_to_cell_map,
    _material_label,
)
from ..operations.volumetric._face_tables import ELEMENT_FACES


# ======================================================================
# Test fixtures: mesh builders
# ======================================================================

def _make_single_tet():
    """Create a single tetrahedron (regular-ish)."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.866, 0.0],
        [0.5, 0.289, 0.816],
    ])
    faces = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]])
    volume_cells = {'tet4': np.array([[0, 1, 2, 3]])}
    return vertices, faces, volume_cells


def _make_two_material_mesh():
    """Create a mesh with two materials sharing an interface.

    Two tetrahedra sharing a triangular face:
        Tet A (material 1): vertices 0,1,2,3
        Tet B (material 2): vertices 0,1,2,4

    The shared face is (0,1,2).
    """
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.866, 0.0],
        [0.5, 0.289, 0.816],  # tet A apex (above)
        [0.5, 0.289, -0.816], # tet B apex (below)
    ])
    faces = np.array([
        [0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3],
        [0, 1, 4], [1, 2, 4], [0, 2, 4],
    ])
    volume_cells = {
        'tet4': np.array([
            [0, 1, 2, 3],  # material 1
            [2, 1, 0, 4],  # material 2 (reversed for proper orientation)
        ])
    }
    cell_materials = np.array([1, 2])
    material_names = {1: 'Steel', 2: 'Aluminum'}
    return vertices, faces, volume_cells, cell_materials, material_names


def _make_three_material_mesh():
    """Create a mesh with three materials.

    Four tetrahedra sharing edges, with materials 1, 2, 3.
    """
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.866, 0.0],
        [0.5, 0.289, 0.816],
        [0.5, 0.289, -0.816],
        [-0.5, 0.289, 0.0],
    ])
    faces = np.array([
        [0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3],
        [0, 1, 4], [1, 2, 4], [0, 2, 4],
        [0, 2, 5], [2, 0, 5], [0, 5, 2],
    ])
    volume_cells = {
        'tet4': np.array([
            [0, 1, 2, 3],  # material 1
            [2, 1, 0, 4],  # material 2 (reversed for orientation)
            [0, 2, 5, 3],  # material 1
            [2, 0, 5, 4],  # material 3 (reversed for orientation)
        ])
    }
    cell_materials = np.array([1, 2, 1, 3])
    material_names = {1: 'Steel', 2: 'Aluminum', 3: 'Copper'}
    return vertices, faces, volume_cells, cell_materials, material_names


def _make_degenerate_interface_mesh():
    """Create a mesh with a degenerate (flat) tet at the interface."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.866, 0.0],
        [0.5, 0.289, 0.816],
        [0.5, 0.289, 0.816 + 1e-15],  # nearly coincident with vertex 3
    ])
    faces = np.array([
        [0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3],
        [0, 1, 4], [1, 2, 4], [0, 2, 4],
    ])
    volume_cells = {
        'tet4': np.array([
            [0, 1, 2, 3],  # material 1 - good
            [2, 1, 0, 4],  # material 2 - degenerate (flat)
        ])
    }
    cell_materials = np.array([1, 2])
    return vertices, faces, volume_cells, cell_materials


# ======================================================================
# Test 1: MeshInput material support
# ======================================================================

class TestMeshInputMaterialSupport:
    """Tests for MeshInput material fields and methods."""

    def test_no_materials_by_default(self):
        """MeshInput with no materials should return None/empty."""
        v, f, vc = _make_single_tet()
        mesh = MeshInput(vertices=v, faces=f, volume_cells=vc)
        assert mesh.cell_materials is None
        assert mesh.material_names is None
        assert mesh.has_multiple_materials() is False
        assert len(mesh.get_cells_by_material(1)) == 0
        assert mesh.get_material_for_cell(0) is None

    def test_set_cell_materials(self):
        """Setting cell_materials should work and return copies."""
        v, f, vc = _make_single_tet()
        mesh = MeshInput(
            vertices=v, faces=f, volume_cells=vc,
            cell_materials=np.array([1]),
        )
        assert mesh.cell_materials is not None
        np.testing.assert_array_equal(mesh.cell_materials, [1])
        assert mesh.has_multiple_materials() is False

    def test_multi_material_detection(self):
        """has_multiple_materials should detect 2+ unique IDs."""
        v, f, vc, cm, mn = _make_two_material_mesh()
        mesh = MeshInput(
            vertices=v, faces=f, volume_cells=vc,
            cell_materials=cm, material_names=mn,
        )
        assert mesh.has_multiple_materials() is True

    def test_get_cells_by_material(self):
        """get_cells_by_material should return correct indices."""
        v, f, vc, cm, mn = _make_three_material_mesh()
        mesh = MeshInput(
            vertices=v, faces=f, volume_cells=vc,
            cell_materials=cm, material_names=mn,
        )
        steel_cells = mesh.get_cells_by_material(1)
        np.testing.assert_array_equal(steel_cells, [0, 2])
        alum_cells = mesh.get_cells_by_material(2)
        np.testing.assert_array_equal(alum_cells, [1])
        copper_cells = mesh.get_cells_by_material(3)
        np.testing.assert_array_equal(copper_cells, [3])

    def test_get_material_for_cell(self):
        """get_material_for_cell should return correct material ID."""
        v, f, vc, cm, _ = _make_two_material_mesh()
        mesh = MeshInput(
            vertices=v, faces=f, volume_cells=vc,
            cell_materials=cm,
        )
        assert mesh.get_material_for_cell(0) == 1
        assert mesh.get_material_for_cell(1) == 2

    def test_validate_materials_ok(self):
        """Valid material assignments should produce no warnings."""
        v, f, vc, cm, mn = _make_two_material_mesh()
        mesh = MeshInput(
            vertices=v, faces=f, volume_cells=vc,
            cell_materials=cm, material_names=mn,
        )
        warnings = mesh.validate_materials()
        assert len(warnings) == 0

    def test_validate_materials_length_mismatch(self):
        """cell_materials with wrong length should warn."""
        v, f, vc = _make_single_tet()
        mesh = MeshInput(
            vertices=v, faces=f, volume_cells=vc,
            cell_materials=np.array([1, 2]),  # 2 materials for 1 cell
        )
        warnings = mesh.validate_materials()
        assert len(warnings) == 1
        assert "doesn't match" in warnings[0]

    def test_validate_materials_orphaned_ids(self):
        """Material IDs not in material_names should warn."""
        v, f, vc = _make_single_tet()
        mesh = MeshInput(
            vertices=v, faces=f, volume_cells=vc,
            cell_materials=np.array([1]),
            material_names={99: 'Unknown'},  # 1 not in names
        )
        warnings = mesh.validate_materials()
        assert any("Undefined material IDs" in w for w in warnings)

    def test_material_names_property(self):
        """material_names getter/setter should work."""
        v, f, vc = _make_single_tet()
        mesh = MeshInput(vertices=v, faces=f, volume_cells=vc)
        assert mesh.material_names is None
        mesh.material_names = {1: 'Steel'}
        assert mesh.material_names == {1: 'Steel'}

    def test_copy_preserves_materials(self):
        """Deep copy should include material data."""
        v, f, vc, cm, mn = _make_two_material_mesh()
        mesh = MeshInput(
            vertices=v, faces=f, volume_cells=vc,
            cell_materials=cm, material_names=mn,
        )
        mesh_copy = mesh.copy()
        np.testing.assert_array_equal(mesh_copy.cell_materials, cm)
        assert mesh_copy.material_names == mn

    def test_to_dict_includes_materials(self):
        """to_dict should include material info when present."""
        v, f, vc, cm, mn = _make_two_material_mesh()
        mesh = MeshInput(
            vertices=v, faces=f, volume_cells=vc,
            cell_materials=cm, material_names=mn,
        )
        d = mesh.to_dict()
        assert 'materials' in d
        assert d['materials']['count'] == 2
        assert d['materials']['names'] == mn


# ======================================================================
# Test 2: Volume computation helpers
# ======================================================================

class TestVolumeComputation:
    """Tests for cell volume computation used by multi-material operations."""

    def test_tet_volume(self):
        """Regular tetrahedron volume should be positive and correct."""
        v, _, _ = _make_single_tet()
        conn = np.array([0, 1, 2, 3])
        vol = _compute_tet_volume(v, conn)
        assert vol > 0
        # For our tet, volume ≈ 0.1178
        assert abs(vol - 0.1178) < 0.01

    def test_degenerate_tet_volume_near_zero(self):
        """Degenerate (flat) tet should have near-zero volume."""
        v = np.array([
            [0, 0, 0], [1, 0, 0], [0.5, 0.866, 0],
            [0.5, 0.289, 1e-15],  # nearly coplanar
        ])
        conn = np.array([0, 1, 2, 3])
        vol = _compute_tet_volume(v, conn)
        assert vol < 1e-12


# ======================================================================
# Test 3: Scaled Jacobian computation
# ======================================================================

class TestScaledJacobian:
    """Tests for scaled Jacobian computation."""

    def test_regular_tet_positive_jacobian(self):
        """Regular tet should have positive scaled Jacobian."""
        v, _, _ = _make_single_tet()
        conn = np.array([0, 1, 2, 3])
        sj = _compute_scaled_jacobian_tet(v, conn)
        assert sj > 0
        # For a regular tet, scaled Jacobian is close to 1.0
        assert sj > 0.5

    def test_degenerate_tet_low_jacobian(self):
        """Degenerate tet should have very low scaled Jacobian."""
        v = np.array([
            [0, 0, 0], [1, 0, 0], [0.5, 0.866, 0],
            [0.5, 0.289, 1e-15],
        ])
        conn = np.array([0, 1, 2, 3])
        sj = _compute_scaled_jacobian_tet(v, conn)
        assert abs(sj) < 0.1  # Nearly zero for flat tet


# ======================================================================
# Test 4: Face-to-cell map
# ======================================================================

class TestFaceToCellMap:
    """Tests for the shared face-to-cell mapping utility."""

    def test_two_tets_shared_face(self):
        """Two tets sharing a face should produce one shared face key."""
        v, f, vc, cm, mn = _make_two_material_mesh()
        mesh = MeshInput(
            vertices=v, faces=f, volume_cells=vc,
            cell_materials=cm,
        )
        face_map = _build_face_to_cell_map(mesh)

        # Find the shared face (0,1,2) sorted
        shared_key = tuple(sorted([0, 1, 2]))
        assert shared_key in face_map
        assert len(face_map[shared_key]) == 2

    def test_boundary_faces_single_entry(self):
        """Boundary faces should appear exactly once in the map."""
        v, f, vc, cm, mn = _make_two_material_mesh()
        mesh = MeshInput(
            vertices=v, faces=f, volume_cells=vc,
            cell_materials=cm,
        )
        face_map = _build_face_to_cell_map(mesh)

        boundary_faces = [k for k, cells in face_map.items() if len(cells) == 1]
        shared_faces = [k for k, cells in face_map.items() if len(cells) == 2]

        # Two tets with 4 faces each = 8 total, minus 1 shared = 7 unique
        # But shared face appears twice, so 7 entries total
        # Actually: each tet has 4 faces. Shared face (0,1,2) appears in both.
        # Boundary faces: 3 per tet (non-shared) = 6 boundary + 1 shared = 7 entries
        assert len(shared_faces) >= 1
        assert len(boundary_faces) >= 6


# ======================================================================
# Test 5: MaterialInterfaceDetection
# ======================================================================

class TestMaterialInterfaceDetection:
    """Tests for the MaterialInterfaceDetection operation."""

    def test_no_volume_cells_returns_empty(self):
        """Surface mesh should return empty results."""
        v = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.866, 0]])
        f = np.array([[0, 1, 2]])
        mesh = MeshInput(vertices=v, faces=f)
        op = MaterialInterfaceDetection()
        issues = op.execute(mesh)
        assert len(issues) == 0

    def test_no_materials_returns_empty(self):
        """Mesh without material assignments should return empty."""
        v, f, vc = _make_single_tet()
        mesh = MeshInput(vertices=v, faces=f, volume_cells=vc)
        op = MaterialInterfaceDetection()
        issues = op.execute(mesh)
        assert len(issues) == 0

    def test_single_material_returns_empty(self):
        """Mesh with only one material should return empty."""
        v, f, vc = _make_single_tet()
        mesh = MeshInput(
            vertices=v, faces=f, volume_cells=vc,
            cell_materials=np.array([1]),
        )
        op = MaterialInterfaceDetection()
        issues = op.execute(mesh)
        assert len(issues) == 0

    def test_two_materials_detects_interface(self):
        """Two materials sharing a face should be detected."""
        v, f, vc, cm, mn = _make_two_material_mesh()
        mesh = MeshInput(
            vertices=v, faces=f, volume_cells=vc,
            cell_materials=cm, material_names=mn,
        )
        op = MaterialInterfaceDetection()
        issues = op.execute(mesh)

        assert len(issues) == 1
        issue = issues[0]
        assert issue.severity == IssueSeverity.INFO
        assert issue.metrics['interface_face_count'] >= 1
        assert issue.metrics['interface_cell_count'] == 2
        assert issue.metrics['total_interface_area'] > 0
        # Check material pair
        mat_ids = {issue.metrics['material_1'], issue.metrics['material_2']}
        assert mat_ids == {1, 2}

    def test_three_materials_multiple_interfaces(self):
        """Three materials should produce multiple interface issues."""
        v, f, vc, cm, mn = _make_three_material_mesh()
        mesh = MeshInput(
            vertices=v, faces=f, volume_cells=vc,
            cell_materials=cm, material_names=mn,
        )
        op = MaterialInterfaceDetection()
        issues = op.execute(mesh)

        # Should detect interfaces between: (1,2), (1,3), (2,3)
        # depending on which cells share faces
        assert len(issues) >= 1
        for issue in issues:
            assert issue.metrics['interface_face_count'] >= 1

    def test_interface_issue_has_locations(self):
        """Interface issues should include cell locations."""
        v, f, vc, cm, mn = _make_two_material_mesh()
        mesh = MeshInput(
            vertices=v, faces=f, volume_cells=vc,
            cell_materials=cm, material_names=mn,
        )
        op = MaterialInterfaceDetection()
        issues = op.execute(mesh)

        assert len(issues) > 0
        assert len(issues[0].locations) > 0
        assert issues[0].locations[0].element_type == 'cell'


# ======================================================================
# Test 6: InterfaceQualityCheck
# ======================================================================

class TestInterfaceQualityCheck:
    """Tests for the InterfaceQualityCheck operation."""

    def test_no_materials_returns_empty(self):
        """Mesh without materials should return empty."""
        v, f, vc = _make_single_tet()
        mesh = MeshInput(vertices=v, faces=f, volume_cells=vc)
        op = InterfaceQualityCheck()
        issues = op.execute(mesh)
        assert len(issues) == 0

    def test_good_interface_no_issues(self):
        """Good quality cells at interface should produce no quality issues."""
        v, f, vc, cm, mn = _make_two_material_mesh()
        mesh = MeshInput(
            vertices=v, faces=f, volume_cells=vc,
            cell_materials=cm, material_names=mn,
        )
        op = InterfaceQualityCheck(min_scaled_jacobian=0.1)
        issues = op.execute(mesh)
        # Both tets are well-shaped; only INFO or empty
        jacobian_issues = [
            i for i in issues
            if i.metadata.get('check') == 'scaled_jacobian'
            and i.severity != IssueSeverity.INFO
        ]
        assert len(jacobian_issues) == 0

    def test_degenerate_interface_reports_quality(self):
        """Degenerate tet at interface should report quality issue."""
        v, f, vc, cm = _make_degenerate_interface_mesh()
        mesh = MeshInput(
            vertices=v, faces=f, volume_cells=vc,
            cell_materials=cm,
        )
        op = InterfaceQualityCheck(min_scaled_jacobian=0.5)
        issues = op.execute(mesh)
        # Should find quality problems at the interface
        jacobian_issues = [
            i for i in issues
            if i.metadata.get('check') == 'scaled_jacobian'
        ]
        assert len(jacobian_issues) >= 1

    def test_quality_check_has_locations(self):
        """Quality issues should reference specific cells."""
        v, f, vc, cm = _make_degenerate_interface_mesh()
        mesh = MeshInput(
            vertices=v, faces=f, volume_cells=vc,
            cell_materials=cm,
        )
        op = InterfaceQualityCheck(min_scaled_jacobian=0.5)
        issues = op.execute(mesh)
        for issue in issues:
            if issue.severity != IssueSeverity.INFO:
                assert len(issue.locations) > 0


# ======================================================================
# Test 7: PerMaterialStatistics
# ======================================================================

class TestPerMaterialStatistics:
    """Tests for the PerMaterialStatistics operation."""

    def test_no_materials_returns_empty(self):
        """Mesh without materials should return empty."""
        v, f, vc = _make_single_tet()
        mesh = MeshInput(vertices=v, faces=f, volume_cells=vc)
        op = PerMaterialStatistics()
        issues = op.execute(mesh)
        assert len(issues) == 0

    def test_two_materials_two_reports(self):
        """Two materials should produce two INFO issues."""
        v, f, vc, cm, mn = _make_two_material_mesh()
        mesh = MeshInput(
            vertices=v, faces=f, volume_cells=vc,
            cell_materials=cm, material_names=mn,
        )
        op = PerMaterialStatistics()
        issues = op.execute(mesh)

        assert len(issues) == 2
        for issue in issues:
            assert issue.severity == IssueSeverity.INFO
            assert 'cell_count' in issue.metrics
            assert issue.metrics['cell_count'] == 1.0

    def test_statistics_include_volume(self):
        """Per-material stats should include total volume."""
        v, f, vc, cm, mn = _make_two_material_mesh()
        mesh = MeshInput(
            vertices=v, faces=f, volume_cells=vc,
            cell_materials=cm, material_names=mn,
        )
        op = PerMaterialStatistics()
        issues = op.execute(mesh)

        for issue in issues:
            assert 'total_volume' in issue.metrics
            assert issue.metrics['total_volume'] > 0

    def test_statistics_include_quality(self):
        """Per-material stats should include quality metrics for tet cells."""
        v, f, vc, cm, mn = _make_two_material_mesh()
        mesh = MeshInput(
            vertices=v, faces=f, volume_cells=vc,
            cell_materials=cm, material_names=mn,
        )
        op = PerMaterialStatistics()
        issues = op.execute(mesh)

        for issue in issues:
            assert 'avg_quality' in issue.metrics
            assert 'min_quality' in issue.metrics
            assert 'max_quality' in issue.metrics
            # Both tets are well-shaped
            assert issue.metrics['avg_quality'] > 0.3

    def test_three_materials_three_reports(self):
        """Three materials should produce three INFO issues."""
        v, f, vc, cm, mn = _make_three_material_mesh()
        mesh = MeshInput(
            vertices=v, faces=f, volume_cells=vc,
            cell_materials=cm, material_names=mn,
        )
        op = PerMaterialStatistics()
        issues = op.execute(mesh)

        assert len(issues) == 3
        material_ids = {issue.metadata['material_id'] for issue in issues}
        assert material_ids == {1, 2, 3}

    def test_material_name_in_message(self):
        """Issue message should include material name."""
        v, f, vc, cm, mn = _make_two_material_mesh()
        mesh = MeshInput(
            vertices=v, faces=f, volume_cells=vc,
            cell_materials=cm, material_names=mn,
        )
        op = PerMaterialStatistics()
        issues = op.execute(mesh)

        messages = ' '.join(i.message for i in issues)
        assert 'Steel' in messages
        assert 'Aluminum' in messages


# ======================================================================
# Test 8: Integration — all operations together
# ======================================================================

class TestMultiMaterialIntegration:
    """Integration tests running all multi-material operations."""

    def test_full_pipeline(self):
        """Run all three operations on a multi-material mesh."""
        v, f, vc, cm, mn = _make_two_material_mesh()
        mesh = MeshInput(
            vertices=v, faces=f, volume_cells=vc,
            cell_materials=cm, material_names=mn,
        )

        # Validate materials
        warnings = mesh.validate_materials()
        assert len(warnings) == 0

        # Run all operations
        all_issues = []
        for OpClass in [MaterialInterfaceDetection, InterfaceQualityCheck, PerMaterialStatistics]:
            op = OpClass()
            issues = op.execute(mesh)
            all_issues.extend(issues)

        # Should have at least:
        # 1 interface detection + 2 per-material stats
        assert len(all_issues) >= 3

        # Check categories
        quality_issues = [i for i in all_issues if i.category == IssueCategory.QUALITY]
        assert len(quality_issues) == len(all_issues)

    def test_operations_registered_in_registry(self):
        """All three operations should be findable in the global registry."""
        from ..operations.operation_registry import registry

        assert 'quality_3d.detect_material_interfaces' in registry
        assert 'quality_3d.check_interface_quality' in registry
        assert 'quality_3d.per_material_statistics' in registry

    def test_metadata_properties(self):
        """Each operation should have valid metadata."""
        for OpClass in [MaterialInterfaceDetection, InterfaceQualityCheck, PerMaterialStatistics]:
            op = OpClass()
            meta = op.metadata
            assert meta.operation_id.startswith('quality_3d.')
            assert meta.category == IssueCategory.QUALITY
            assert len(meta.supported_libraries) > 0
            assert len(meta.tags) > 0


# ======================================================================
# Test 9: Material label helper
# ======================================================================

class TestMaterialLabel:
    """Tests for the _material_label helper."""

    def test_with_name(self):
        """Should return named material."""
        v, f, vc, cm, mn = _make_two_material_mesh()
        mesh = MeshInput(
            vertices=v, faces=f, volume_cells=vc,
            cell_materials=cm, material_names=mn,
        )
        assert _material_label(mesh, 1) == 'Steel'
        assert _material_label(mesh, 2) == 'Aluminum'

    def test_without_name(self):
        """Should return generic label when no name mapping."""
        v, f, vc = _make_single_tet()
        mesh = MeshInput(vertices=v, faces=f, volume_cells=vc)
        assert _material_label(mesh, 1) == 'Material_1'

    def test_missing_name(self):
        """Should return generic label when ID not in mapping."""
        v, f, vc = _make_single_tet()
        mesh = MeshInput(
            vertices=v, faces=f, volume_cells=vc,
            material_names={1: 'Steel'},
        )
        assert _material_label(mesh, 99) == 'Material_99'
