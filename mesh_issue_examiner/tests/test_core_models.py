"""
Comprehensive unit tests for core/models.py module.

Tests all data models: IssueSeverity, IssueCategory, Vertex, Face, Edge,
IssueLocation, MeshIssue, QualityMetrics, InspectionResult, OperationConfig,
and MeshInput.
"""

import pytest
import numpy as np

from core.models import (
    IssueSeverity,
    IssueCategory,
    Vertex,
    Face,
    Edge,
    IssueLocation,
    MeshIssue,
    QualityMetrics,
    InspectionResult,
    OperationConfig,
    MeshInput,
)


# ---------------------------------------------------------------------------
# IssueSeverity
# ---------------------------------------------------------------------------

class TestIssueSeverity:
    """Tests for the IssueSeverity enum."""

    def test_values(self):
        """All four severity levels have the expected string values."""
        assert IssueSeverity.INFO.value == "info"
        assert IssueSeverity.WARNING.value == "warning"
        assert IssueSeverity.ERROR.value == "error"
        assert IssueSeverity.CRITICAL.value == "critical"

    def test_member_count(self):
        """There are exactly four severity levels."""
        assert len(IssueSeverity) == 4

    def test_ordering_info_less_than_warning(self):
        assert IssueSeverity.INFO < IssueSeverity.WARNING

    def test_ordering_warning_less_than_error(self):
        assert IssueSeverity.WARNING < IssueSeverity.ERROR

    def test_ordering_error_less_than_critical(self):
        assert IssueSeverity.ERROR < IssueSeverity.CRITICAL

    def test_ordering_transitive(self):
        """INFO < CRITICAL via transitivity."""
        assert IssueSeverity.INFO < IssueSeverity.CRITICAL

    def test_ordering_equal_not_less(self):
        """A severity is not less than itself."""
        assert not (IssueSeverity.WARNING < IssueSeverity.WARNING)

    def test_ordering_critical_not_less_than_info(self):
        """CRITICAL is not less than INFO."""
        assert not (IssueSeverity.CRITICAL < IssueSeverity.INFO)

    def test_sorting(self):
        """Sorting a list of severities yields ascending order."""
        severities = [IssueSeverity.CRITICAL, IssueSeverity.INFO,
                      IssueSeverity.ERROR, IssueSeverity.WARNING]
        assert sorted(severities) == [IssueSeverity.INFO, IssueSeverity.WARNING,
                                      IssueSeverity.ERROR, IssueSeverity.CRITICAL]


# ---------------------------------------------------------------------------
# IssueCategory
# ---------------------------------------------------------------------------

class TestIssueCategory:
    """Tests for the IssueCategory enum."""

    def test_values(self):
        """All four category values exist and are accessible."""
        assert IssueCategory.TOPOLOGICAL.value == "topological"
        assert IssueCategory.GEOMETRIC.value == "geometric"
        assert IssueCategory.QUALITY.value == "quality"
        assert IssueCategory.ORIENTATION.value == "orientation"

    def test_member_count(self):
        """There are exactly four categories."""
        assert len(IssueCategory) == 4


# ---------------------------------------------------------------------------
# Vertex, Face, Edge dataclasses
# ---------------------------------------------------------------------------

class TestVertex:
    """Tests for the Vertex dataclass."""

    def test_construction_basic(self):
        v = Vertex(index=0, coordinates=(1.0, 2.0, 3.0))
        assert v.index == 0
        assert v.coordinates == (1.0, 2.0, 3.0)
        assert v.properties == {}

    def test_construction_with_properties(self):
        v = Vertex(index=5, coordinates=(0.0, 0.0, 0.0),
                   properties={"color": "red", "group": 1})
        assert v.properties == {"color": "red", "group": 1}

    def test_default_properties_is_empty_dict(self):
        v1 = Vertex(index=0, coordinates=(0.0, 0.0, 0.0))
        v2 = Vertex(index=1, coordinates=(1.0, 1.0, 1.0))
        # Each instance gets its own dict (no shared mutable default)
        v1.properties["x"] = 99
        assert "x" not in v2.properties


class TestFace:
    """Tests for the Face dataclass."""

    def test_construction_basic(self):
        f = Face(indices=[0, 1, 2])
        assert f.indices == [0, 1, 2]
        assert f.normal is None
        assert f.area is None

    def test_construction_with_normal_and_area(self):
        f = Face(indices=[0, 1, 2], normal=(0.0, 0.0, 1.0), area=0.5)
        assert f.normal == (0.0, 0.0, 1.0)
        assert f.area == 0.5


class TestEdge:
    """Tests for the Edge dataclass."""

    def test_construction_basic(self):
        e = Edge(vertices=(0, 1), face_count=2)
        assert e.vertices == (0, 1)
        assert e.face_count == 2
        assert e.incident_faces == []

    def test_construction_with_incident_faces(self):
        e = Edge(vertices=(2, 3), face_count=1, incident_faces=[5])
        assert e.incident_faces == [5]

    def test_default_incident_faces_is_empty_list(self):
        e1 = Edge(vertices=(0, 1), face_count=1)
        e2 = Edge(vertices=(2, 3), face_count=1)
        e1.incident_faces.append(99)
        assert e2.incident_faces == []


# ---------------------------------------------------------------------------
# IssueLocation
# ---------------------------------------------------------------------------

class TestIssueLocation:
    """Tests for the IssueLocation dataclass."""

    def test_construction_basic(self):
        loc = IssueLocation(element_type="vertex", element_index=3)
        assert loc.element_type == "vertex"
        assert loc.element_index == 3
        assert loc.coordinates is None
        assert loc.adjacent_elements == {}

    def test_construction_full(self):
        loc = IssueLocation(
            element_type="face",
            element_index=10,
            coordinates=(1.0, 2.0, 3.0),
            adjacent_elements={"edges": [4, 5, 6]}
        )
        assert loc.coordinates == (1.0, 2.0, 3.0)
        assert loc.adjacent_elements == {"edges": [4, 5, 6]}

    def test_to_dict(self):
        loc = IssueLocation(
            element_type="edge",
            element_index=7,
            coordinates=(0.5, 0.5, 0.0),
            adjacent_elements={"faces": [0, 1]}
        )
        d = loc.to_dict()
        assert d == {
            "element_type": "edge",
            "element_index": 7,
            "coordinates": (0.5, 0.5, 0.0),
            "adjacent_elements": {"faces": [0, 1]}
        }

    def test_to_dict_defaults(self):
        loc = IssueLocation(element_type="vertex", element_index=0)
        d = loc.to_dict()
        assert d["coordinates"] is None
        assert d["adjacent_elements"] == {}


# ---------------------------------------------------------------------------
# MeshIssue
# ---------------------------------------------------------------------------

class TestMeshIssue:
    """Tests for the MeshIssue dataclass."""

    def _make_issue(self, severity=IssueSeverity.WARNING, n_locations=0):
        locations = [
            IssueLocation(element_type="vertex", element_index=i)
            for i in range(n_locations)
        ]
        return MeshIssue(
            operation_id="op_1",
            category=IssueCategory.GEOMETRIC,
            severity=severity,
            message="test issue",
            locations=locations,
        )

    def test_construction_defaults(self):
        issue = MeshIssue(
            operation_id="op_1",
            category=IssueCategory.TOPOLOGICAL,
            severity=IssueSeverity.INFO,
            message="info msg",
        )
        assert issue.locations == []
        assert issue.metrics == {}
        assert issue.metadata == {}

    def test_is_critical_true(self):
        issue = self._make_issue(severity=IssueSeverity.CRITICAL)
        assert issue.is_critical is True

    def test_is_critical_false_for_error(self):
        issue = self._make_issue(severity=IssueSeverity.ERROR)
        assert issue.is_critical is False

    def test_is_critical_false_for_warning(self):
        issue = self._make_issue(severity=IssueSeverity.WARNING)
        assert issue.is_critical is False

    def test_is_critical_false_for_info(self):
        issue = self._make_issue(severity=IssueSeverity.INFO)
        assert issue.is_critical is False

    def test_to_dict_normal(self):
        issue = self._make_issue(n_locations=3)
        d = issue.to_dict()
        assert d["operation_id"] == "op_1"
        assert d["category"] == "geometric"
        assert d["severity"] == "warning"
        assert d["message"] == "test issue"
        assert d["location_count"] == 3
        assert len(d["locations"]) == 3
        assert d["metrics"] == {}
        assert d["metadata"] == {}

    def test_to_dict_truncates_locations_to_ten(self):
        """When there are more than 10 locations, to_dict truncates to 10."""
        issue = self._make_issue(n_locations=15)
        assert len(issue.locations) == 15
        d = issue.to_dict()
        # location_count reflects the true total
        assert d["location_count"] == 15
        # but only 10 are serialized
        assert len(d["locations"]) == 10
        # verify they are the first 10
        for i, loc_dict in enumerate(d["locations"]):
            assert loc_dict["element_index"] == i

    def test_to_dict_with_exactly_ten_locations(self):
        """Exactly 10 locations should not be truncated."""
        issue = self._make_issue(n_locations=10)
        d = issue.to_dict()
        assert d["location_count"] == 10
        assert len(d["locations"]) == 10

    def test_to_dict_includes_metrics_and_metadata(self):
        issue = MeshIssue(
            operation_id="op_2",
            category=IssueCategory.QUALITY,
            severity=IssueSeverity.ERROR,
            message="bad quality",
            metrics={"aspect_ratio": 12.5},
            metadata={"source": "unit_test"},
        )
        d = issue.to_dict()
        assert d["metrics"] == {"aspect_ratio": 12.5}
        assert d["metadata"] == {"source": "unit_test"}


# ---------------------------------------------------------------------------
# QualityMetrics
# ---------------------------------------------------------------------------

class TestQualityMetrics:
    """Tests for the QualityMetrics dataclass."""

    def test_with_valid_values_and_threshold(self):
        """Statistics are computed and violations counted."""
        vals = np.array([0.5, 0.8, 0.3, 0.95])
        qm = QualityMetrics(
            operation_id="op_q1",
            metric_name="aspect_ratio",
            values=vals,
            threshold=0.4,
        )
        assert qm.statistics["min"] == pytest.approx(0.3)
        assert qm.statistics["max"] == pytest.approx(0.95)
        assert qm.statistics["mean"] == pytest.approx(np.mean([0.5, 0.8, 0.3, 0.95]))
        assert qm.statistics["median"] == pytest.approx(np.median([0.5, 0.8, 0.3, 0.95]))
        assert qm.statistics["std"] == pytest.approx(np.std([0.5, 0.8, 0.3, 0.95]))
        # 0.3 < 0.4 -> 1 violation
        assert qm.violation_count == 1

    def test_list_input_converted_to_ndarray(self):
        """Passing a plain list should be converted to ndarray internally."""
        qm = QualityMetrics(
            operation_id="op_q2",
            metric_name="skewness",
            values=[1.0, 2.0, 3.0],
        )
        assert isinstance(qm.values, np.ndarray)

    def test_empty_values_no_statistics(self):
        """An empty values array should not produce statistics."""
        qm = QualityMetrics(
            operation_id="op_q3",
            metric_name="empty_metric",
            values=np.array([]),
        )
        assert qm.statistics == {}
        assert qm.violation_count == 0

    def test_threshold_none_no_violation_counting(self):
        """When threshold is None, violation_count stays at default 0."""
        qm = QualityMetrics(
            operation_id="op_q4",
            metric_name="area",
            values=np.array([0.1, 0.2, 0.3]),
        )
        assert qm.threshold is None
        assert qm.violation_count == 0

    def test_specific_statistics_values(self):
        """Verify exact statistics for [0.5, 0.8, 0.3, 0.95]."""
        vals = [0.5, 0.8, 0.3, 0.95]
        qm = QualityMetrics(
            operation_id="op_q5",
            metric_name="test",
            values=vals,
        )
        assert qm.statistics["min"] == pytest.approx(0.3)
        assert qm.statistics["max"] == pytest.approx(0.95)
        assert qm.statistics["mean"] == pytest.approx(0.6375)
        assert qm.statistics["median"] == pytest.approx(0.65)
        assert qm.statistics["std"] == pytest.approx(np.std(vals))
        assert qm.statistics["p5"] == pytest.approx(np.percentile(vals, 5))
        assert qm.statistics["p95"] == pytest.approx(np.percentile(vals, 95))

    def test_to_dict_no_raw_values(self):
        """to_dict must NOT include raw values, but must include total_count."""
        qm = QualityMetrics(
            operation_id="op_q6",
            metric_name="diag",
            values=np.array([1.0, 2.0, 3.0]),
        )
        d = qm.to_dict()
        assert "values" not in d
        assert d["total_count"] == 3
        assert d["operation_id"] == "op_q6"
        assert d["metric_name"] == "diag"
        assert "statistics" in d
        assert d["threshold"] is None
        assert d["violation_count"] == 0

    def test_single_value_array(self):
        """Edge case: a single-element array should still compute statistics."""
        qm = QualityMetrics(
            operation_id="op_q7",
            metric_name="singleton",
            values=np.array([4.2]),
        )
        assert qm.statistics["min"] == pytest.approx(4.2)
        assert qm.statistics["max"] == pytest.approx(4.2)
        assert qm.statistics["mean"] == pytest.approx(4.2)
        assert qm.statistics["median"] == pytest.approx(4.2)
        assert qm.statistics["std"] == pytest.approx(0.0)

    def test_violation_count_all_below_threshold(self):
        """All values below threshold -> violation_count == len(values)."""
        qm = QualityMetrics(
            operation_id="op_q8",
            metric_name="area",
            values=np.array([0.01, 0.02, 0.03]),
            threshold=0.5,
        )
        assert qm.violation_count == 3

    def test_violation_count_none_below_threshold(self):
        """No values below threshold -> violation_count == 0."""
        qm = QualityMetrics(
            operation_id="op_q9",
            metric_name="area",
            values=np.array([0.6, 0.7, 0.8]),
            threshold=0.5,
        )
        assert qm.violation_count == 0


# ---------------------------------------------------------------------------
# InspectionResult
# ---------------------------------------------------------------------------

class TestInspectionResult:
    """Tests for the InspectionResult dataclass."""

    def _make_issue(self, severity, category=IssueCategory.GEOMETRIC):
        return MeshIssue(
            operation_id="op_test",
            category=category,
            severity=severity,
            message="test",
        )

    def test_empty_result_passed_is_true(self):
        """Empty result after compute_summary -> passed is True."""
        result = InspectionResult(mesh_id="mesh_1", file_path=None)
        result.compute_summary()
        assert result.passed is True

    def test_one_warning_passed_is_true(self):
        """A single WARNING issue should still pass."""
        result = InspectionResult(mesh_id="mesh_2", file_path=None)
        result.add_issue(self._make_issue(IssueSeverity.WARNING))
        result.compute_summary()
        assert result.passed is True

    def test_one_error_passed_is_false(self):
        """A single ERROR issue should cause failure."""
        result = InspectionResult(mesh_id="mesh_3", file_path=None)
        result.add_issue(self._make_issue(IssueSeverity.ERROR))
        result.compute_summary()
        assert result.passed is False

    def test_one_critical_passed_is_false(self):
        """A single CRITICAL issue should cause failure."""
        result = InspectionResult(mesh_id="mesh_4", file_path=None)
        result.add_issue(self._make_issue(IssueSeverity.CRITICAL))
        result.compute_summary()
        assert result.passed is False

    def test_passed_true_before_compute_summary_by_design(self):
        """
        Before compute_summary() is called, summary is {} and passed returns True
        even if issues have been added. This is an intentional design choice:
        the caller must call compute_summary() to populate the summary dict.
        """
        result = InspectionResult(mesh_id="mesh_5", file_path=None)
        result.add_issue(self._make_issue(IssueSeverity.CRITICAL))
        # summary is still empty
        assert result.summary == {}
        # passed returns True because has_critical / has_errors are not in summary
        assert result.passed is True

    def test_add_metrics(self):
        result = InspectionResult(mesh_id="mesh_6", file_path=None)
        qm = QualityMetrics(
            operation_id="op_m",
            metric_name="area",
            values=np.array([1.0, 2.0]),
        )
        result.add_metrics(qm)
        assert len(result.quality_metrics) == 1
        assert result.quality_metrics[0].metric_name == "area"

    def test_compute_summary_counts(self):
        """Verify severity and category breakdowns in summary."""
        result = InspectionResult(mesh_id="mesh_7", file_path=None)
        result.add_issue(self._make_issue(IssueSeverity.WARNING, IssueCategory.GEOMETRIC))
        result.add_issue(self._make_issue(IssueSeverity.WARNING, IssueCategory.TOPOLOGICAL))
        result.add_issue(self._make_issue(IssueSeverity.CRITICAL, IssueCategory.QUALITY))
        result.add_issue(self._make_issue(IssueSeverity.INFO, IssueCategory.ORIENTATION))
        result.compute_summary()

        s = result.summary
        assert s["total_issues"] == 4
        assert s["warning_count"] == 2
        assert s["critical_count"] == 1
        assert s["info_count"] == 1
        assert s["error_count"] == 0
        assert s["has_critical"] is True
        assert s["has_errors"] is False
        assert s["has_warnings"] is True

        assert s["severity_breakdown"]["warning"] == 2
        assert s["severity_breakdown"]["critical"] == 1
        assert s["severity_breakdown"]["info"] == 1
        assert s["severity_breakdown"]["error"] == 0

        assert s["category_breakdown"]["geometric"] == 1
        assert s["category_breakdown"]["topological"] == 1
        assert s["category_breakdown"]["quality"] == 1
        assert s["category_breakdown"]["orientation"] == 1

    def test_to_dict(self):
        result = InspectionResult(mesh_id="mesh_8", file_path="/tmp/test.ply")
        result.add_issue(self._make_issue(IssueSeverity.INFO))
        result.compute_summary()
        d = result.to_dict()
        assert d["mesh_id"] == "mesh_8"
        assert d["file_path"] == "/tmp/test.ply"
        assert d["issue_count"] == 1
        assert len(d["issues"]) == 1
        assert d["summary"]["total_issues"] == 1
        assert d["processing_time"] == 0.0


# ---------------------------------------------------------------------------
# OperationConfig
# ---------------------------------------------------------------------------

class TestOperationConfig:
    """Tests for the OperationConfig dataclass."""

    def test_default_values(self):
        cfg = OperationConfig()
        assert cfg.tolerance == 1e-6
        assert cfg.quality_threshold == 0.1
        assert cfg.enable_caching is True
        assert cfg.max_iterations == 1000
        assert cfg.parallel_enabled is True
        assert cfg.library_preference is None

    def test_custom_values(self):
        cfg = OperationConfig(
            tolerance=1e-10,
            quality_threshold=0.5,
            enable_caching=False,
            max_iterations=500,
            parallel_enabled=False,
            library_preference="trimesh",
        )
        assert cfg.tolerance == 1e-10
        assert cfg.quality_threshold == 0.5
        assert cfg.enable_caching is False
        assert cfg.max_iterations == 500
        assert cfg.parallel_enabled is False
        assert cfg.library_preference == "trimesh"

    def test_to_dict(self):
        cfg = OperationConfig()
        d = cfg.to_dict()
        assert d == {
            "tolerance": 1e-6,
            "quality_threshold": 0.1,
            "enable_caching": True,
            "max_iterations": 1000,
            "parallel_enabled": True,
            "library_preference": None,
        }


# ---------------------------------------------------------------------------
# MeshInput
# ---------------------------------------------------------------------------

class TestMeshInput:
    """Tests for the MeshInput class."""

    # Unit cube: 8 corners
    UNIT_CUBE_VERTS = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ], dtype=float)

    # 12 triangles forming a cube
    UNIT_CUBE_FACES = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 1, 5], [0, 5, 4],  # front
        [2, 3, 7], [2, 7, 6],  # back
        [0, 3, 7], [0, 7, 4],  # left
        [1, 2, 6], [1, 6, 5],  # right
    ])

    def test_construction_with_numpy_arrays(self):
        mesh = MeshInput(self.UNIT_CUBE_VERTS, self.UNIT_CUBE_FACES,
                         mesh_id="cube")
        assert mesh.mesh_id == "cube"
        np.testing.assert_array_equal(mesh.vertices, self.UNIT_CUBE_VERTS)
        np.testing.assert_array_equal(mesh.faces, self.UNIT_CUBE_FACES)

    def test_construction_with_plain_lists(self):
        verts = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        faces = [[0, 1, 2]]
        mesh = MeshInput(verts, faces)
        assert isinstance(mesh.vertices, np.ndarray)
        assert isinstance(mesh.faces, np.ndarray)

    def test_default_mesh_id_generation(self):
        """When mesh_id is empty, a default is generated using id(self)."""
        mesh = MeshInput([[0, 0, 0]], [[0]], mesh_id="")
        assert mesh.mesh_id.startswith("mesh_")
        # The id suffix should be numeric
        suffix = mesh.mesh_id.split("_", 1)[1]
        assert suffix.isdigit()

    def test_n_vertices_and_n_faces(self):
        mesh = MeshInput(self.UNIT_CUBE_VERTS, self.UNIT_CUBE_FACES,
                         mesh_id="cube")
        assert mesh.n_vertices == 8
        assert mesh.n_faces == 12

    def test_bounding_box_unit_cube(self):
        mesh = MeshInput(self.UNIT_CUBE_VERTS, self.UNIT_CUBE_FACES,
                         mesh_id="cube")
        bb_min, bb_max = mesh.bounding_box
        np.testing.assert_array_equal(bb_min, [0, 0, 0])
        np.testing.assert_array_equal(bb_max, [1, 1, 1])

    def test_center_unit_cube(self):
        mesh = MeshInput(self.UNIT_CUBE_VERTS, self.UNIT_CUBE_FACES,
                         mesh_id="cube")
        np.testing.assert_array_almost_equal(mesh.center, [0.5, 0.5, 0.5])

    def test_extent_unit_cube(self):
        mesh = MeshInput(self.UNIT_CUBE_VERTS, self.UNIT_CUBE_FACES,
                         mesh_id="cube")
        np.testing.assert_array_almost_equal(mesh.extent, [1.0, 1.0, 1.0])

    def test_bounding_box_shifted_cube(self):
        """Vertices from (2,3,4) to (5,6,7) -> bbox min=(2,3,4), max=(5,6,7)."""
        verts = np.array([
            [2, 3, 4], [5, 3, 4], [5, 6, 4], [2, 6, 4],
            [2, 3, 7], [5, 3, 7], [5, 6, 7], [2, 6, 7],
        ], dtype=float)
        mesh = MeshInput(verts, [[0, 1, 2]])
        bb_min, bb_max = mesh.bounding_box
        np.testing.assert_array_equal(bb_min, [2, 3, 4])
        np.testing.assert_array_equal(bb_max, [5, 6, 7])
        np.testing.assert_array_almost_equal(mesh.center, [3.5, 4.5, 5.5])
        np.testing.assert_array_almost_equal(mesh.extent, [3.0, 3.0, 3.0])

    def test_setter_invalidates_cache(self):
        """Setting vertices should clear the native objects cache."""
        mesh = MeshInput(self.UNIT_CUBE_VERTS, self.UNIT_CUBE_FACES,
                         mesh_id="cube")
        # Manually put something in the cache to verify it gets cleared
        mesh._native_objects["dummy"] = "cached_value"
        assert len(mesh._native_objects) == 1

        mesh.vertices = np.zeros_like(self.UNIT_CUBE_VERTS)
        assert len(mesh._native_objects) == 0
        assert len(mesh._adjacency_cache) == 0

    def test_faces_setter_invalidates_cache(self):
        """Setting faces should clear the native objects cache."""
        mesh = MeshInput(self.UNIT_CUBE_VERTS, self.UNIT_CUBE_FACES,
                         mesh_id="cube")
        mesh._native_objects["dummy"] = "cached"
        mesh.faces = self.UNIT_CUBE_FACES.copy()
        assert len(mesh._native_objects) == 0

    def test_get_native_unknown_library_raises_valueerror(self):
        mesh = MeshInput(self.UNIT_CUBE_VERTS, self.UNIT_CUBE_FACES,
                         mesh_id="cube")
        with pytest.raises(ValueError, match="Unsupported library"):
            mesh.get_native("nonexistent_lib")

    def test_get_native_trimesh_import_error(self):
        """If trimesh is not installed, get_native('trimesh') should raise ImportError."""
        mesh = MeshInput(self.UNIT_CUBE_VERTS, self.UNIT_CUBE_FACES,
                         mesh_id="cube")
        try:
            import trimesh  # noqa: F401
            pytest.skip("trimesh is installed; cannot test ImportError path")
        except ImportError:
            with pytest.raises(ImportError, match="[Tt][Rr]imesh"):
                mesh.get_native("trimesh")

    def test_get_native_trimesh_happy_path(self):
        """If trimesh is installed, get_native should return a trimesh.Trimesh."""
        mesh = MeshInput(self.UNIT_CUBE_VERTS, self.UNIT_CUBE_FACES,
                         mesh_id="cube")
        try:
            import trimesh  # noqa: F401
        except ImportError:
            pytest.skip("trimesh not installed")
        result = mesh.get_native("trimesh")
        import trimesh as tm
        assert isinstance(result, tm.Trimesh)

    def test_copy_creates_independent_copy(self):
        mesh = MeshInput(self.UNIT_CUBE_VERTS, self.UNIT_CUBE_FACES,
                         mesh_id="original")
        mesh_copy = mesh.copy()
        assert mesh_copy.mesh_id == "original_copy"
        np.testing.assert_array_equal(mesh_copy.vertices, mesh.vertices)
        np.testing.assert_array_equal(mesh_copy.faces, mesh.faces)

        # Modify the copy; original should be unaffected
        mesh_copy.vertices[0, 0] = 999.0
        assert mesh.vertices[0, 0] != 999.0

    def test_copy_preserves_source_format_and_file_path(self):
        mesh = MeshInput(self.UNIT_CUBE_VERTS, self.UNIT_CUBE_FACES,
                         mesh_id="m1", source_format="stl",
                         file_path="/tmp/m.stl")
        mesh_copy = mesh.copy()
        assert mesh_copy.source_format == "stl"
        assert mesh_copy.file_path == "/tmp/m.stl"

    def test_to_dict_includes_expected_keys(self):
        mesh = MeshInput(self.UNIT_CUBE_VERTS, self.UNIT_CUBE_FACES,
                         mesh_id="cube", source_format="obj",
                         file_path="/tmp/cube.obj")
        d = mesh.to_dict()
        assert d["mesh_id"] == "cube"
        assert d["source_format"] == "obj"
        assert d["file_path"] == "/tmp/cube.obj"
        assert d["n_vertices"] == 8
        assert d["n_faces"] == 12
        assert "bounding_box" in d
        assert d["bounding_box"]["min"] == [0, 0, 0]
        assert d["bounding_box"]["max"] == [1, 1, 1]

    def test_minimal_mesh_one_triangle(self):
        """Edge case: a single-triangle mesh should still compute bounding_box."""
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2]])
        mesh = MeshInput(verts, faces, mesh_id="tri")
        assert mesh.n_vertices == 3
        assert mesh.n_faces == 1
        bb_min, bb_max = mesh.bounding_box
        np.testing.assert_array_equal(bb_min, [0, 0, 0])
        np.testing.assert_array_equal(bb_max, [1, 1, 0])
        np.testing.assert_array_almost_equal(mesh.center, [0.5, 0.5, 0.0])
        np.testing.assert_array_almost_equal(mesh.extent, [1.0, 1.0, 0.0])

    def test_faces_setter_with_list_input(self):
        """Setting faces via a plain list should convert to ndarray."""
        mesh = MeshInput(self.UNIT_CUBE_VERTS, self.UNIT_CUBE_FACES,
                         mesh_id="cube")
        new_faces = [[0, 1, 2]]
        mesh.faces = new_faces
        assert isinstance(mesh.faces, np.ndarray)
        assert mesh.n_faces == 1

    def test_vertices_setter_with_list_input(self):
        """Setting vertices via a plain list should convert to ndarray."""
        mesh = MeshInput(self.UNIT_CUBE_VERTS, self.UNIT_CUBE_FACES,
                         mesh_id="cube")
        new_verts = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        mesh.vertices = new_verts
        assert isinstance(mesh.vertices, np.ndarray)
        assert mesh.n_vertices == 3

    def test_invalidate_cache(self):
        mesh = MeshInput(self.UNIT_CUBE_VERTS, self.UNIT_CUBE_FACES,
                         mesh_id="cube")
        mesh._native_objects["x"] = "y"
        mesh._adjacency_cache["a"] = "b"
        mesh.invalidate_cache()
        assert len(mesh._native_objects) == 0
        assert len(mesh._adjacency_cache) == 0

    def test_get_native_caches_result(self):
        """Calling get_native twice should return the same cached object."""
        mesh = MeshInput(self.UNIT_CUBE_VERTS, self.UNIT_CUBE_FACES,
                         mesh_id="cube")
        # Use a library that is definitely not installed to test ValueError
        # For caching, we can verify via the _native_objects dict directly
        # by using a known path. Since we don't know which libs are installed,
        # we test caching by inserting a fake value.
        mesh._native_objects["fake"] = "cached_obj"
        assert mesh.get_native("fake") == "cached_obj"
