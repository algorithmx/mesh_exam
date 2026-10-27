"""
Quality metrics surfacing: operations -> workflow -> MCP endpoint.

Regression tests for the gap documented in docs/legacy/feature_gaps.md
("Quality metrics not returned for surface meshes"): mesh_get_quality_metrics
returned metric_count=0 for every real mesh because no code path populated
InspectionResult.quality_metrics. Quality operations now surface
QualityMetrics distributions via BaseOperation.last_quality_metrics,
collected by the registry workflow (and thus by the MCP server).
"""

import asyncio

import numpy as np

from mesh_issue_examiner.core.models import MeshInput
from mesh_issue_examiner.mcp.server import MeshIssueExaminerServer, MCPRequest
from mesh_issue_examiner.operations.operation_registry import registry


# ---------------------------------------------------------------------------
# Shared mesh builders
# ---------------------------------------------------------------------------

def _cube_surface_mesh(mesh_id: str = "cube_surface") -> MeshInput:
    """Unit-cube surface: 8 vertices, 12 triangles, clean quality.

    All angles are 45/45/90, so min_angle (45) > 15, max_angle (90) < 120,
    and aspect_ratio stays well below 10 -- zero expected violations.
    """
    vertices = np.array(
        [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 2, 1], [0, 3, 2],      # bottom (z=0)
            [4, 5, 6], [4, 6, 7],      # top (z=1)
            [0, 1, 5], [0, 5, 4],      # front (y=0)
            [2, 3, 7], [2, 7, 6],      # back (y=1)
            [1, 2, 6], [1, 6, 5],      # right (x=1)
            [0, 4, 7], [0, 7, 3],      # left (x=0)
        ]
    )
    return MeshInput(vertices=vertices, faces=faces, mesh_id=mesh_id)


def _single_tet_mesh(mesh_id: str = "one_tet") -> MeshInput:
    """A single regular tetrahedron with a tet4 volume cell."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3) / 2, 0.0],
            [0.5, np.sqrt(3) / 6, np.sqrt(2.0 / 3.0)],
        ]
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])
    return MeshInput(
        vertices=vertices,
        faces=faces,
        mesh_id=mesh_id,
        volume_cells={"tet4": np.array([[0, 1, 2, 3]])},
    )


STAT_KEYS = ("min", "max", "mean", "median", "std", "p5", "p95")


# ---------------------------------------------------------------------------
# Workflow level
# ---------------------------------------------------------------------------

class TestWorkflowSurfacesMetrics:
    """Registry workflows collect QualityMetrics from quality operations."""

    def test_surface_quality_metrics_in_result(self):
        mesh = _cube_surface_mesh()
        workflow = registry.create_workflow(["quality.compute_cell_quality"])
        result = workflow(mesh)

        names = {m.metric_name for m in result.quality_metrics}
        assert {"aspect_ratio", "min_angle", "max_angle"} <= names
        for m in result.quality_metrics:
            for key in STAT_KEYS:
                assert key in m.statistics, f"{m.metric_name}: missing {key}"
            assert m.statistics["max"] >= m.statistics["min"]
        # Values cover every face of the mesh (12 triangles)
        for m in result.quality_metrics:
            assert len(m.values) == 12
        # Summary counts the surfaced metrics
        assert result.summary["quality_metric_count"] == len(result.quality_metrics)

    def test_clean_mesh_yields_zero_violation_metrics(self):
        """Zero-violation metrics must still be surfaced (the point of the
        metrics endpoint), and the clean cube raises no quality issues."""
        mesh = _cube_surface_mesh()
        workflow = registry.create_workflow(["quality.compute_cell_quality"])
        result = workflow(mesh)

        assert len(result.quality_metrics) >= 3
        for m in result.quality_metrics:
            assert m.violation_count == 0, m.metric_name
        quality_issues = [
            i for i in result.issues
            if i.operation_id.startswith("quality.compute_cell_quality")
        ]
        assert quality_issues == []

    def test_volume_quality_metrics_in_result(self):
        mesh = _single_tet_mesh()
        workflow = registry.create_workflow(
            ["quality_3d.compute_volume_cell_quality"]
        )
        result = workflow(mesh)

        names = {m.metric_name for m in result.quality_metrics}
        # Default quality types all applicable to tet4
        assert {
            "scaled_jacobian", "volume", "aspect_ratio", "condition", "shape",
        } <= names
        for m in result.quality_metrics:
            assert len(m.values) == 1
            assert m.threshold is not None

    def test_metrics_reset_between_runs(self):
        """The side-channel resets per run: no accumulation across workflows."""
        mesh = _cube_surface_mesh()
        workflow = registry.create_workflow(["quality.compute_cell_quality"])
        r1 = workflow(mesh)
        r2 = workflow(mesh)
        assert len(r1.quality_metrics) == len(r2.quality_metrics) == 3

    def test_non_quality_operation_leaves_metrics_empty(self):
        """Topological operations do not produce quality metrics."""
        mesh = _cube_surface_mesh()
        workflow = registry.create_workflow(
            ["topology.detect_boundary_edges"]
        )
        result = workflow(mesh)
        assert result.quality_metrics == []


# ---------------------------------------------------------------------------
# MCP server level
# ---------------------------------------------------------------------------

class TestMCPQualityMetricsEndpoint:
    """mesh_load-style flow: inspect then mesh_get_quality_metrics."""

    def _inspect(self, server, mesh_id, operations):
        resp = asyncio.run(server.handle_request(MCPRequest(
            method="mesh_inspect",
            params={"mesh_id": mesh_id, "operations": operations},
        )))
        assert resp.error is None, resp.error
        return resp.result

    def test_get_quality_metrics_returns_live_data(self):
        server = MeshIssueExaminerServer(auto_register_operations=True)
        server._mesh_cache["mcp_cube"] = _cube_surface_mesh("mcp_cube")

        inspect = self._inspect(server, "mcp_cube", ["quality.compute_cell_quality"])
        assert inspect["summary"]["quality_metric_count"] >= 3

        resp = asyncio.run(server.handle_request(MCPRequest(
            method="mesh_get_quality_metrics",
            params={"mesh_id": "mcp_cube"},
        )))
        assert resp.error is None, resp.error
        assert resp.result["metric_count"] >= 3
        names = {m["metric_name"] for m in resp.result["metrics"]}
        assert {"aspect_ratio", "min_angle", "max_angle"} <= names
        for m in resp.result["metrics"]:
            for key in STAT_KEYS:
                assert key in m["statistics"]
            assert m["total_count"] == 12

    def test_metric_name_filter(self):
        server = MeshIssueExaminerServer(auto_register_operations=True)
        server._mesh_cache["mcp_cube2"] = _cube_surface_mesh("mcp_cube2")
        self._inspect(server, "mcp_cube2", ["quality.compute_cell_quality"])

        resp = asyncio.run(server.handle_request(MCPRequest(
            method="mesh_get_quality_metrics",
            params={"mesh_id": "mcp_cube2", "metric_name": "aspect_ratio"},
        )))
        assert resp.error is None, resp.error
        assert resp.result["metric_count"] == 1
        assert resp.result["metrics"][0]["metric_name"] == "aspect_ratio"

    def test_volumetric_metrics_via_mcp(self):
        server = MeshIssueExaminerServer(auto_register_operations=True)
        server._mesh_cache["mcp_tet"] = _single_tet_mesh("mcp_tet")
        self._inspect(server, "mcp_tet", ["quality_3d.compute_volume_cell_quality"])

        resp = asyncio.run(server.handle_request(MCPRequest(
            method="mesh_get_quality_metrics",
            params={"mesh_id": "mcp_tet"},
        )))
        assert resp.error is None, resp.error
        assert resp.result["metric_count"] >= 1
        names = {m["metric_name"] for m in resp.result["metrics"]}
        assert "scaled_jacobian" in names
