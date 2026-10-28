"""
Tests for the midside_deviation quality metric (quadratic curvature).

midside_deviation measures, per quadratic cell, how far its mid-edge
nodes sit from the midpoints of their corner edges, normalised by the
cell's longest corner edge (dimensionless).  Straight cells score 0.0
and stay silent; curved cells fire.  Values are computed directly from
the quadratic connectivity (never through vtkCellQuality, which returns
-1 for every quadratic cell) and cover quadratic cells only.
"""

import numpy as np
import pytest

from mesh_issue_examiner.core.models import IssueSeverity, MeshInput
from mesh_issue_examiner.operations.volumetric.volume_cell_quality import (
    VolumeCellQualityComputation,
)

MID = 'midside_deviation'


def _op() -> VolumeCellQualityComputation:
    return VolumeCellQualityComputation(quality_types=[MID])


class TestStraightCellsSilent:
    """Straight quadratic cells score 0.0 and raise no issue."""

    def test_straight_tet10_silent(self):
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [0.5, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0],
            [0, 0, 0.5], [0.5, 0, 0.5], [0, 0.5, 0.5],
        ])
        mesh = MeshInput(
            vertices=verts,
            faces=np.zeros((0, 3), dtype=int),
            mesh_id='straight_tet10',
            volume_cells={'tet10': np.arange(10).reshape(1, 10)},
        )
        op = _op()
        issues = op.execute(mesh)
        assert issues == []
        (qm,) = op.last_quality_metrics
        assert qm.metric_name == MID
        assert qm.violation_count == 0
        assert qm.values == pytest.approx([0.0], abs=1e-12)

    def test_straight_hex20_silent(self):
        corners = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], float)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
                 (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        mids = np.array([(corners[a] + corners[b]) / 2 for a, b in edges])
        verts = np.vstack([corners, mids])
        conn = np.arange(20, dtype=np.int64).reshape(1, 20)
        mesh = MeshInput(
            vertices=verts, faces=np.zeros((0, 3), dtype=int),
            mesh_id='straight_hex20', volume_cells={'hex20': conn},
        )
        op = _op()
        assert op.execute(mesh) == []
        (qm,) = op.last_quality_metrics
        assert qm.values == pytest.approx([0.0], abs=1e-12)

    def test_complete_forms_ignore_face_and_body_centres(self):
        """hex27 body nodes, wedge18 face centres, pyramid14 base centre
        are NOT edge nodes — garbage values there must be ignored."""
        rng = np.random.default_rng(42)

        def build(n_extra, base):
            junk = rng.uniform(50, 100, size=(n_extra, 3))
            verts = np.vstack([base, junk])
            n = len(verts)
            return verts, np.arange(n, dtype=np.int64).reshape(1, n)

        # hex27: 8 corners + 12 edge midsides (straight) + 7 junk centres
        corners = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], float)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
                 (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        hex_base = np.vstack(
            [corners, np.array([(corners[a] + corners[b]) / 2
                                for a, b in edges])])
        # prism15: 6 corners + 9 straight midsides (+3 junk face centres)
        # slots 6-8 bottom (0,1)(1,2)(2,0); 9-11 top (3,4)(4,5)(5,3);
        # 12-14 verticals (0,3)(1,4)(2,5)
        prism_base = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1],
            [0.5, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0],
            [0.5, 0, 1], [0.5, 0.5, 1], [0, 0.5, 1],
            [0, 0, 0.5], [1, 0, 0.5], [0, 1, 0.5],
        ], float)
        # pyramid14: 5 corners + 8 straight midsides (+1 junk base centre)
        pyr_base = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1],
            [0.5, 0, 0], [1, 0.5, 0], [0.5, 1, 0], [0, 0.5, 0],
            [0.25, 0.25, 0.5], [0.75, 0.25, 0.5],
            [0.75, 0.75, 0.5], [0.25, 0.75, 0.5],
        ], float)

        verts_h, conn_h = build(7, hex_base)  # 8 corners + 12 midsides + 7 centres
        verts_p, conn_p = build(1, pyr_base)  # 5 corners + 8 midsides + 1 centre
        n_prism = 6 + 9
        verts_w = np.vstack([
            prism_base[:n_prism],
            rng.uniform(50, 100, size=(3, 3)),  # wedge18 face centres 15-17
        ])
        conn_w = np.arange(18, dtype=np.int64).reshape(1, 18)

        all_verts = np.vstack([verts_h, verts_p, verts_w])
        offset_h = 0
        offset_p = len(verts_h)
        offset_w = offset_p + len(verts_p)
        cells = {
            'hex27': conn_h + offset_h,
            'pyramid14': conn_p + offset_p,
            'wedge18': conn_w + offset_w,
        }
        mesh = MeshInput(
            vertices=all_verts, faces=np.zeros((0, 3), dtype=int),
            mesh_id='complete_forms', volume_cells=cells,
        )
        op = _op()
        assert op.execute(mesh) == []
        (qm,) = op.last_quality_metrics
        assert qm.values == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
        assert qm.metadata['quadratic_only'] is True
        assert qm.metadata['quadratic_types'] == ['hex27', 'pyramid14', 'wedge18']


class TestCurvedCellsFire:
    """Pulled midside nodes produce deviations above the threshold."""

    def test_curved_tet10_fires_with_global_location(self):
        # Slot 9 = edge (2,3); move it off the midpoint by 0.5.
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [0.5, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0],
            [0, 0, 0.5], [0.5, 0, 0.5],
            [0, 0.5, 1.0],  # midpoint is (0, 0.5, 0.5) -> dev 0.5
        ])
        mesh = MeshInput(
            vertices=verts, faces=np.zeros((0, 3), dtype=int),
            mesh_id='curved_tet10',
            volume_cells={'tet10': np.arange(10).reshape(1, 10)},
        )
        op = _op()
        issues = op.execute(mesh)
        assert len(issues) == 1
        issue = issues[0]
        assert issue.severity == IssueSeverity.CRITICAL  # rate 1.0
        assert issue.metrics['violation_count'] == 1
        assert issue.metrics['max'] == pytest.approx(0.5 / np.sqrt(2))
        assert issue.locations[0].element_index == 0
        assert 'linearized' not in issue.metadata

    def test_mixed_linear_quadratic_reports_quadratic_only(self):
        """Values cover quadratic cells only; locations use global ids."""
        tet4 = np.array([[0, 1, 2, 3]], dtype=np.int64)
        tet10 = np.arange(10, dtype=np.int64).reshape(1, 10)
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [0.5, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0],
            [0, 0, 0.5], [0.5, 0, 0.5],
            [0, 0.5, 1.0],  # curved
        ])
        mesh = MeshInput(
            vertices=verts, faces=np.zeros((0, 3), dtype=int),
            mesh_id='mixed',
            volume_cells={'tet4': tet4, 'tet10': tet10},
        )
        op = _op()
        issues = op.execute(mesh)
        assert len(issues) == 1
        # The tet4 (global cell 0) is linear -> excluded; the curved
        # tet10 sits at global cell 1.
        assert issues[0].locations[0].element_index == 1
        (qm,) = op.last_quality_metrics
        assert len(qm.values) == 1

    def test_curved_hex20_edge_slot_fires(self):
        corners = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], float)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
                 (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        mids = np.array([(corners[a] + corners[b]) / 2 for a, b in edges])
        mids[7] = [0.0, 0.5, 1.3]  # slot 15 = edge (7,4); midpoint (0,.5,1)
        verts = np.vstack([corners, mids])
        mesh = MeshInput(
            vertices=verts, faces=np.zeros((0, 3), dtype=int),
            mesh_id='curved_hex20',
            volume_cells={'hex20': np.arange(20).reshape(1, 20)},
        )
        issues = _op().execute(mesh)
        assert len(issues) == 1
        assert issues[0].metrics['max'] == pytest.approx(0.3)

    def test_scale_invariance(self):
        """The metric is relative: scaling the mesh preserves values."""
        base = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [0.5, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0],
            [0, 0, 0.5], [0.5, 0, 0.5],
            [0, 0.5, 1.0],
        ])

        def run(scale):
            mesh = MeshInput(
                vertices=base * scale,
                faces=np.zeros((0, 3), dtype=int),
                mesh_id=f'scaled_{scale}',
                volume_cells={'tet10': np.arange(10).reshape(1, 10)},
            )
            op = _op()
            op.execute(mesh)
            return op.last_quality_metrics[0]

        qm1, qm10 = run(1.0), run(10.0)
        assert qm1.values == pytest.approx(qm10.values)


class TestApplicability:
    """The metric is quadratic-only by construction."""

    def test_linear_mesh_returns_no_metric(self):
        mesh = MeshInput(
            vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            faces=np.zeros((0, 3), dtype=int),
            mesh_id='linear_only',
            volume_cells={'tet4': np.array([[0, 1, 2, 3]])},
        )
        op = _op()
        assert op.execute(mesh) == []
        assert op.last_quality_metrics == []

    def test_prism15_straight_silent(self):
        corners = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1],
        ], float)
        edges = [(0, 1), (1, 2), (2, 0),
                 (3, 4), (4, 5), (5, 3),
                 (0, 3), (1, 4), (2, 5)]
        mids = np.array([(corners[a] + corners[b]) / 2 for a, b in edges])
        mesh = MeshInput(
            vertices=np.vstack([corners, mids]),
            faces=np.zeros((0, 3), dtype=int),
            mesh_id='straight_prism15',
            volume_cells={'prism15': np.arange(15).reshape(1, 15)},
        )
        op = _op()
        assert op.execute(mesh) == []
        (qm,) = op.last_quality_metrics
        assert qm.values == pytest.approx([0.0], abs=1e-12)
