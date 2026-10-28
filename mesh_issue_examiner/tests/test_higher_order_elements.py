"""
Higher-order (quadratic) element support tests.

Covers:
- Registration of the complete quadratic types (hex27, wedge18, pyramid14)
  in utils/cell_types.py.
- Corner-first face-table resolution in operations/volumetric/_face_tables.py.
- Volumetric connectivity and orientation operations on quadratic meshes
  (two quadratic tets sharing a face).
- Graceful degradation for pyramid14, which has no VTK equivalent: the
  full workflow must not crash and native-dependent checks are skipped.
"""

import numpy as np
import pytest

from mesh_issue_examiner.core.models import MeshInput
from mesh_issue_examiner.operations.operation_registry import registry
from mesh_issue_examiner.utils.cell_types import (
    ELEMENT_NODE_COUNT,
    detect_element_type,
    get_cell_type_map,
    validate_volume_cells,
)
from mesh_issue_examiner.operations.volumetric._face_tables import (
    ELEMENT_CORNER_COUNT,
    LINEAR_BASE_TYPE,
    build_face_to_cell_map,
    get_face_definitions,
)
from mesh_issue_examiner.operations.volumetric.boundary_closure import (
    BoundaryClosureCheck,
)
from mesh_issue_examiner.operations.volumetric.duplicate_cells import (
    DuplicateCellDetection,
)
from mesh_issue_examiner.operations.volumetric.nonmanifold_faces import (
    NonManifoldFaceDetection,
)
from mesh_issue_examiner.operations.volumetric.orphaned_cells import (
    OrphanedCellDetection,
)
from mesh_issue_examiner.operations.volumetric.orientation_consistency import (
    OrientationConsistencyCheck,
)
from mesh_issue_examiner.operations.volumetric.volume_cell_quality import (
    VolumeCellQualityComputation,
)


# ==================================================================
# Mesh builders
# ==================================================================

def _two_tet10_mesh(duplicate=False, inconsistent=False):
    """Two quadratic tets sharing the base face (mirrors the e2e fixture).

    Corners a,b,c form the shared base; d/e are the apices.  The second
    tet lists the shared face reversed, so winding is consistent unless
    ``inconsistent=True``.  Midside nodes follow VTK QUADRATIC_TETRA
    ordering: edges (0,1),(1,2),(2,0),(0,3),(1,3),(2,3).
    """
    a, b, c = [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    d, e = [0.0, 0.0, 1.0], [0.25, 0.25, -1.0]

    def mid(p, q):
        return [(pi + qi) / 2 for pi, qi in zip(p, q)]

    vertices = np.array([
        a, b, c, d, e,
        mid(a, b), mid(b, c), mid(c, a),
        mid(a, d), mid(b, d), mid(c, d),
        mid(b, e), mid(a, e), mid(c, e),
    ], float)

    tet1 = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10]
    if inconsistent:
        # Same winding on the shared face (b,c,a cycle of tet1's a,b,c)
        tet2 = [1, 2, 0, 4, 6, 7, 5, 11, 13, 12]
    else:
        tet2 = [1, 0, 2, 4, 5, 7, 6, 11, 12, 13]

    cells = [tet1, tet2]
    if duplicate:
        cells.append(list(tet1))
    return MeshInput(
        vertices=vertices,
        faces=np.zeros((0, 3), dtype=int),
        mesh_id="two_tet10",
        volume_cells={"tet10": np.array(cells)},
    )


def _pyramid14_mesh():
    """A single 14-node pyramid (5 corners + 8 edge mids + base centre)."""
    corners = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1],
    ], float)

    def mid(p, q):
        return (p + q) / 2

    edge_mids = [
        mid(corners[0], corners[1]), mid(corners[1], corners[2]),
        mid(corners[2], corners[3]), mid(corners[3], corners[0]),
        mid(corners[0], corners[4]), mid(corners[1], corners[4]),
        mid(corners[2], corners[4]), mid(corners[3], corners[4]),
    ]
    base_centre = corners[:4].mean(axis=0)
    vertices = np.vstack([corners, edge_mids, [base_centre]])
    cells = np.array([list(range(14))])
    return MeshInput(
        vertices=vertices,
        faces=np.zeros((0, 3), dtype=int),
        mesh_id="pyramid14",
        volume_cells={"pyramid14": cells},
    )


# ==================================================================
# Registration
# ==================================================================

class TestCompleteQuadraticRegistration:
    def test_node_counts_registered(self):
        assert ELEMENT_NODE_COUNT["hex27"] == 27
        assert ELEMENT_NODE_COUNT["wedge18"] == 18
        assert ELEMENT_NODE_COUNT["pyramid14"] == 14

    def test_vtk_map_includes_convertible_types(self):
        import pyvista as pv

        vtk_map = get_cell_type_map()
        assert vtk_map["hex27"] == pv.CellType.TRIQUADRATIC_HEXAHEDRON
        assert vtk_map["wedge18"] == pv.CellType.BIQUADRATIC_QUADRATIC_WEDGE

    def test_vtk_map_has_no_pyramid14(self):
        # VTK's tri-quadratic pyramid has 19 nodes; there is no 14-node
        # VTK cell, so pyramid14 is connectivity-level only.
        assert "pyramid14" not in get_cell_type_map()

    def test_detect_element_type_from_column_count(self):
        assert detect_element_type(np.zeros((1, 27), int)) == "hex27"
        assert detect_element_type(np.zeros((1, 18), int)) == "wedge18"
        assert detect_element_type(np.zeros((1, 14), int)) == "pyramid14"

    def test_validate_volume_cells_accepts_new_types(self):
        validate_volume_cells({
            "hex27": np.zeros((1, 27), int),
            "wedge18": np.zeros((1, 18), int),
            "pyramid14": np.zeros((1, 14), int),
        })


# ==================================================================
# Corner-first face tables
# ==================================================================

class TestCornerFaceTables:
    def test_corner_counts_cover_all_registered_types(self):
        for etype in ELEMENT_NODE_COUNT:
            assert etype in ELEMENT_CORNER_COUNT, etype

    def test_quadratic_types_resolve_to_linear_base(self):
        assert LINEAR_BASE_TYPE["tet10"] == "tet4"
        assert LINEAR_BASE_TYPE["hex27"] == "hex8"
        assert LINEAR_BASE_TYPE["wedge18"] == "prism6"
        assert LINEAR_BASE_TYPE["pyramid14"] == "pyramid5"

    def test_get_face_definitions_resolves_quadratic_types(self):
        for etype in ("tet10", "hex20", "hex27", "prism15", "wedge18",
                      "pyramid13", "pyramid14"):
            defs = get_face_definitions(etype)
            assert defs is not None, etype
        assert get_face_definitions("not_a_type") is None

    def test_tet10_shared_face_has_two_incident_cells(self):
        mesh = _two_tet10_mesh()
        face_map = build_face_to_cell_map(mesh)
        shared = tuple(sorted((0, 1, 2)))   # base face corners a,b,c
        assert shared in face_map
        assert len(face_map[shared]) == 2
        # Global cell indices follow the mesh's cell numbering
        assert {entry[0] for entry in face_map[shared]} == {0, 1}

    def test_unknown_type_advances_global_cell_index(self):
        mesh = MeshInput(
            vertices=np.zeros((5, 3)),
            faces=np.zeros((0, 3), dtype=int),
            mesh_id="mixed_unknown",
            volume_cells={
                "weird9": np.array([[0, 1, 2, 3, 0, 1, 2, 3, 4],
                                    [0, 1, 2, 3, 0, 1, 2, 3, 4]]),
                "tet4": np.array([[0, 1, 2, 4]]),
            },
        )
        face_map = build_face_to_cell_map(mesh)
        # The tet4 cell is globally the third cell (index 2): the two
        # unknown-type cells must still advance the numbering.
        assert all(entry[0] == 2 for entries in face_map.values()
                   for entry in entries)
        assert len(face_map) == 4   # one tet4 -> four faces


# ==================================================================
# Connectivity operations on quadratic meshes
# ==================================================================

class TestOpsOnQuadraticMeshes:
    def test_nonmanifold_faces_silent_on_clean_tet10(self):
        issues = NonManifoldFaceDetection().execute(_two_tet10_mesh())
        assert issues == []

    def test_boundary_closure_closed_tet10_bipyramid(self):
        issues = BoundaryClosureCheck().execute(_two_tet10_mesh())
        assert issues == []

    def test_duplicate_cells_detects_tet10_duplicate(self):
        issues = DuplicateCellDetection().execute(
            _two_tet10_mesh(duplicate=True))
        assert len(issues) == 1
        assert issues[0].metrics["duplicate_cell_count"] == 1
        assert issues[0].metrics["total_cell_count"] == 3

    def test_orphaned_cells_flags_out_of_range_tet10(self):
        mesh = _two_tet10_mesh()
        broken = mesh.volume_cells["tet10"].copy()
        broken[0][9] = 99   # midside node out of range
        mesh = MeshInput(
            vertices=mesh.vertices,
            faces=np.zeros((0, 3), dtype=int),
            mesh_id="orphaned_tet10",
            volume_cells={"tet10": broken},
        )
        issues = OrphanedCellDetection().execute(mesh)
        assert len(issues) == 1
        assert issues[0].metrics["orphaned_cell_count"] == 1

    def test_orientation_consistent_tet10_no_winding_issue(self):
        issues = OrientationConsistencyCheck().execute(_two_tet10_mesh())
        winding = [i for i in issues if "winding" in i.message.lower()]
        inverted = [i for i in issues if "inverted" in i.message.lower()]
        # vtkCellQuality cannot evaluate quadratic tets: its -1 sentinel
        # must be filtered, not reported as an inverted cell.
        assert winding == []
        assert inverted == []

    def test_orientation_inconsistent_tet10_detected(self):
        issues = OrientationConsistencyCheck().execute(
            _two_tet10_mesh(inconsistent=True))
        winding = [i for i in issues if "winding" in i.message.lower()]
        assert len(winding) == 1
        assert winding[0].metrics["inconsistent_winding_cells"] == 2


# ==================================================================
# pyramid14: no VTK equivalent, must degrade gracefully
# ==================================================================

class TestPyramid14Degradation:
    def test_full_workflow_does_not_crash(self):
        mesh = _pyramid14_mesh()
        workflow = registry.create_full_workflow()
        result = workflow(mesh)   # must not raise
        # Connectivity-level checks still ran on the corner nodes.
        assert isinstance(result.issues, list)

    def test_connectivity_ops_run_on_pyramid14(self):
        mesh = _pyramid14_mesh()
        assert NonManifoldFaceDetection().execute(mesh) == []
        assert BoundaryClosureCheck().execute(mesh) == []
        assert OrphanedCellDetection().execute(mesh) == []
        assert DuplicateCellDetection().execute(mesh) == []

    def test_orientation_winding_check_skips_native_for_pyramid14(self):
        issues = OrientationConsistencyCheck().execute(_pyramid14_mesh())
        assert isinstance(issues, list)
        assert issues == []   # single cell: no shared faces, no inversion data


# ==================================================================
# Corner-linearised quality metrics (Phase 4: vtkCellQuality returns
# its -1 sentinel on quadratic cells, so quality_3d recomputes every
# metric on the corner-only linear grid and tags the provenance)
# ==================================================================

class TestLinearizedQuality:
    def _metrics_by_name(self, mesh):
        op = VolumeCellQualityComputation()
        issues = op.execute(mesh)
        by_name = {qm.metric_name: qm for qm in op.last_quality_metrics}
        return op, issues, by_name

    def test_tet10_metrics_real_and_tagged_linearized(self):
        """No -1 sentinels survive; every metric carries the tag."""
        op, issues, by_name = self._metrics_by_name(_two_tet10_mesh())
        assert issues == []   # good corner tets: no threshold violations
        assert set(by_name) >= {"scaled_jacobian", "volume",
                                "aspect_ratio"}
        for name, qm in by_name.items():
            assert not np.any(qm.values == -1.0), name
            assert qm.metadata.get("linearized") is True, name
            assert qm.metadata["linearized_types"] == ["tet10"]
            assert len(qm.values) == 2   # one value per original cell

    def test_tet10_matches_corner_equivalent_tet4(self):
        """Linearised values equal the corner tet4 mesh exactly."""
        quad = _two_tet10_mesh()
        corner = MeshInput(
            vertices=quad.vertices,
            faces=np.zeros((0, 3), dtype=int),
            mesh_id="two_tet10_corners",
            volume_cells={"tet4": np.array([[0, 1, 2, 3], [1, 0, 2, 4]])},
        )
        _, _, quad_m = self._metrics_by_name(quad)
        op4, issues4, lin_m = self._metrics_by_name(corner)
        assert issues4 == []
        for name, qm_quad in quad_m.items():
            assert name in lin_m, name
            np.testing.assert_allclose(
                qm_quad.values, lin_m[name].values, rtol=1e-12, err_msg=name)
        # linear mesh must not be tagged
        assert all("linearized" not in qm.metadata for qm in lin_m.values())

    def test_pyramid14_receives_linearized_metrics(self):
        """pyramid14 (no VTK cell) gets metrics via its pyramid5 corners."""
        op, issues, by_name = self._metrics_by_name(_pyramid14_mesh())
        assert "scaled_jacobian" in by_name
        qm = by_name["scaled_jacobian"]
        assert qm.metadata.get("linearized") is True
        assert qm.metadata["linearized_types"] == ["pyramid14"]
        assert not np.any(qm.values == -1.0)
        assert len(qm.values) == 1

    def test_mixed_linear_quadratic_preserves_cell_count(self):
        """hex8 block + tet10 block: one value per cell, in order."""
        quad = _two_tet10_mesh()
        hex_v = np.array([
            [5, 0, 0], [6, 0, 0], [6, 1, 0], [5, 1, 0],
            [5, 0, 1], [6, 0, 1], [6, 1, 1], [5, 1, 1],
        ], float)
        mesh = MeshInput(
            vertices=np.vstack([quad.vertices, hex_v]),
            faces=np.zeros((0, 3), dtype=int),
            mesh_id="mixed_hex8_tet10",
            volume_cells={
                "hex8": np.array([[14, 15, 16, 17, 18, 19, 20, 21]]),
                "tet10": quad.volume_cells["tet10"],
            },
        )
        _, issues, by_name = self._metrics_by_name(mesh)
        vol = by_name["volume"]
        assert len(vol.values) == 3   # 1 hex + 2 tets
        assert vol.metadata.get("linearized") is True
        assert vol.metadata["linearized_types"] == ["tet10"]
        # unit cube hex first, then the two tets (1/6 each)
        np.testing.assert_allclose(vol.values, [1.0, 1 / 6, 1 / 6])

    def test_linear_mesh_not_tagged(self):
        mesh = MeshInput(
            vertices=np.array([
                [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            ], float),
            faces=np.zeros((0, 3), dtype=int),
            mesh_id="single_tet4",
            volume_cells={"tet4": np.array([[0, 1, 2, 3]])},
        )
        _, issues, by_name = self._metrics_by_name(mesh)
        assert "scaled_jacobian" in by_name
        assert all("linearized" not in qm.metadata
                   for qm in by_name.values())
