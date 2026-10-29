"""
Tests for the controlled mesh visualization module and the mesh_render
MCP tool.

All rendering tests are skipped when the environment cannot create an
offscreen GL context (headless CI without OpenGL), keeping the suite
portable.
"""

import asyncio
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyvista")

from mesh_issue_examiner.core.exceptions import VisualizationError
from mesh_issue_examiner.core.models import (
    InspectionResult,
    IssueCategory,
    IssueLocation,
    IssueSeverity,
    MeshInput,
    MeshIssue,
    QualityMetrics,
)
from mesh_issue_examiner.mcp.server import MCPRequest, MeshIssueExaminerServer
from mesh_issue_examiner.visualization import RenderResult, render_mesh
from mesh_issue_examiner.visualization import renderer as renderer_module

if not renderer_module._offscreen_available():
    pytest.skip(
        "Offscreen rendering unavailable in this environment",
        allow_module_level=True,
    )

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _valid_png(path) -> bool:
    data = Path(path).read_bytes()
    return data[:8] == PNG_MAGIC and len(data) > 1000


def _surface_mesh() -> MeshInput:
    """Closed tetrahedron surface (4 triangular faces)."""
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 0.5, 1.0]], dtype=float
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])
    return MeshInput(vertices=vertices, faces=faces, mesh_id="surface")


def _volume_mesh(materials=None) -> MeshInput:
    """Two disjoint tet4 cells separated along x."""
    vertices = np.array(
        [
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [10, 0, 0], [11, 0, 0], [10, 1, 0], [10, 0, 1],
        ],
        dtype=float,
    )
    volume_cells = {"tet4": np.array([[0, 1, 2, 3], [4, 5, 6, 7]])}
    return MeshInput(
        vertices=vertices,
        faces=np.array([]),
        mesh_id="volume",
        volume_cells=volume_cells,
        cell_materials=materials,
    )


def _point_cloud() -> MeshInput:
    return MeshInput(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float),
        faces=np.array([]),
        mesh_id="cloud",
    )


def _issue(severity, locations):
    return MeshIssue(
        operation_id="test.op",
        category=IssueCategory.TOPOLOGICAL,
        severity=severity,
        message="synthetic issue",
        locations=locations,
    )


# ---------------------------------------------------------------------------
# Core rendering
# ---------------------------------------------------------------------------


class TestRenderMeshBasics:

    def test_surface_render_produces_valid_png(self, tmp_path):
        rr = render_mesh(_surface_mesh(), tmp_path / "surface.png")
        assert isinstance(rr, RenderResult)
        assert _valid_png(rr.path)
        assert (rr.width, rr.height) == (1200, 800)
        assert rr.n_highlighted == 0
        assert rr.scalars_used is None

    @pytest.mark.parametrize("view", ["iso", "front", "top", "right", "back"])
    def test_named_views_produce_valid_pngs(self, tmp_path, view):
        rr = render_mesh(_surface_mesh(), tmp_path / f"{view}.png", view=view)
        assert _valid_png(rr.path)

    def test_camera_fine_control(self, tmp_path):
        rr = render_mesh(
            _surface_mesh(),
            tmp_path / "angled.png",
            azimuth=20.0,
            elevation=-10.0,
            zoom=1.5,
        )
        assert _valid_png(rr.path)

    def test_top_view_handles_degenerate_up_vector(self, tmp_path):
        # elevation 90 deg means the default up vector is parallel to the
        # view direction; the renderer must fall back to (0, 1, 0).
        rr = render_mesh(_surface_mesh(), tmp_path / "top.png", view="top")
        assert _valid_png(rr.path)

    def test_custom_window_size(self, tmp_path):
        rr = render_mesh(
            _surface_mesh(), tmp_path / "small.png", width=320, height=240
        )
        assert (rr.width, rr.height) == (320, 240)
        assert _valid_png(rr.path)

    @pytest.mark.parametrize("representation", ["wireframe", "points"])
    def test_representations(self, tmp_path, representation):
        rr = render_mesh(
            _surface_mesh(),
            tmp_path / f"{representation}.png",
            representation=representation,
        )
        assert _valid_png(rr.path)


class TestRenderMeshTypes:

    def test_volume_mesh_render(self, tmp_path):
        rr = render_mesh(_volume_mesh(), tmp_path / "volume.png")
        assert _valid_png(rr.path)

    def test_volume_wireframe_render(self, tmp_path):
        rr = render_mesh(
            _volume_mesh(),
            tmp_path / "volume_wf.png",
            representation="wireframe",
        )
        assert _valid_png(rr.path)

    def test_point_cloud_render(self, tmp_path):
        rr = render_mesh(_point_cloud(), tmp_path / "cloud.png")
        assert _valid_png(rr.path)


class TestScalarColoring:

    def test_material_scalars(self, tmp_path):
        mesh = _volume_mesh(materials=np.array([10, 20]))
        rr = render_mesh(mesh, tmp_path / "mat.png", scalars="material")
        assert rr.scalars_used == "material"
        assert _valid_png(rr.path)

    def test_material_scalars_without_material_data_raises(self, tmp_path):
        mesh = _volume_mesh()  # no cell_materials
        with pytest.raises(ValueError, match="material_array_name"):
            render_mesh(mesh, tmp_path / "mat.png", scalars="material")

    def test_metric_scalars_on_surface(self, tmp_path):
        mesh = _surface_mesh()  # 4 faces
        result = InspectionResult(mesh_id="surface", file_path=None)
        result.add_metrics(
            QualityMetrics(
                operation_id="test.op",
                metric_name="aspect_ratio",
                values=[1.0, 1.5, 2.0, 1.2],
                threshold=5.0,
            )
        )
        rr = render_mesh(
            mesh, tmp_path / "metric.png", scalars="aspect_ratio", result=result
        )
        assert rr.scalars_used == "aspect_ratio"
        assert _valid_png(rr.path)

    def test_volume_metric_scalars_render_grid(self, tmp_path):
        mesh = _volume_mesh()  # 2 cells
        result = InspectionResult(mesh_id="volume", file_path=None)
        result.add_metrics(
            QualityMetrics(
                operation_id="test.op",
                metric_name="volume",
                values=[0.15, -1.0],  # -1 sentinel must become NaN
                threshold=1e-12,
            )
        )
        rr = render_mesh(
            mesh, tmp_path / "vol_metric.png", scalars="volume", result=result
        )
        assert _valid_png(rr.path)

    def test_unknown_metric_raises_with_available_list(self, tmp_path):
        mesh = _surface_mesh()
        result = InspectionResult(mesh_id="surface", file_path=None)
        result.add_metrics(
            QualityMetrics(
                operation_id="test.op",
                metric_name="aspect_ratio",
                values=[1.0, 1.0, 1.0, 1.0],
            )
        )
        with pytest.raises(ValueError, match="aspect_ratio"):
            render_mesh(
                mesh, tmp_path / "x.png", scalars="nope", result=result
            )

    def test_metric_without_result_raises(self, tmp_path):
        with pytest.raises(ValueError, match="mesh_inspect"):
            render_mesh(
                _surface_mesh(), tmp_path / "x.png", scalars="aspect_ratio"
            )

    def test_scalar_length_mismatch_raises(self, tmp_path):
        mesh = _surface_mesh()  # 4 faces
        result = InspectionResult(mesh_id="surface", file_path=None)
        result.add_metrics(
            QualityMetrics(
                operation_id="test.op",
                metric_name="aspect_ratio",
                values=[1.0, 2.0],  # wrong length
            )
        )
        with pytest.raises(ValueError, match="does not match"):
            render_mesh(
                mesh, tmp_path / "x.png", scalars="aspect_ratio", result=result
            )

    def test_point_cloud_scalars_rejected(self, tmp_path):
        result = InspectionResult(mesh_id="cloud", file_path=None)
        result.add_metrics(
            QualityMetrics(
                operation_id="test.op",
                metric_name="aspect_ratio",
                values=[1.0, 1.0, 1.0, 1.0],
            )
        )
        with pytest.raises(ValueError, match="point cloud"):
            render_mesh(
                _point_cloud(),
                tmp_path / "x.png",
                scalars="aspect_ratio",
                result=result,
            )


class TestIssueHighlighting:

    def test_markers_from_coordinates(self, tmp_path):
        issues = [
            _issue(
                IssueSeverity.CRITICAL,
                [IssueLocation("vertex", 0, coordinates=(0.0, 0.0, 0.0))],
            ),
            _issue(
                IssueSeverity.ERROR,
                [IssueLocation("vertex", 1, coordinates=(1.0, 0.0, 0.0))],
            ),
        ]
        rr = render_mesh(
            _surface_mesh(), tmp_path / "marked.png", issues=issues
        )
        assert rr.n_highlighted == 2
        assert _valid_png(rr.path)

    def test_marker_resolved_from_vertex_index(self, tmp_path):
        issues = [
            _issue(
                IssueSeverity.ERROR,
                [IssueLocation("vertex", 2)],  # no coordinates
            )
        ]
        rr = render_mesh(
            _surface_mesh(), tmp_path / "marked.png", issues=issues
        )
        assert rr.n_highlighted == 1

    def test_marker_resolved_from_cell_centroid(self, tmp_path):
        issues = [
            _issue(
                IssueSeverity.CRITICAL,
                [IssueLocation("cell", 1)],  # second tet, no coordinates
            )
        ]
        rr = render_mesh(_volume_mesh(), tmp_path / "marked.png", issues=issues)
        assert rr.n_highlighted == 1

    def test_marker_resolved_from_face_centroid(self, tmp_path):
        issues = [
            _issue(IssueSeverity.WARNING, [IssueLocation("face", 0)])
        ]
        rr = render_mesh(
            _surface_mesh(),
            tmp_path / "marked.png",
            issues=issues,
            highlight_severities=("warning",),
        )
        assert rr.n_highlighted == 1

    def test_severity_filter_excludes_other_severities(self, tmp_path):
        issues = [
            _issue(
                IssueSeverity.WARNING,
                [IssueLocation("vertex", 0, coordinates=(0.0, 0.0, 0.0))],
            )
        ]
        # Default highlight severities are critical + error -> no markers
        rr = render_mesh(
            _surface_mesh(), tmp_path / "marked.png", issues=issues
        )
        assert rr.n_highlighted == 0

    def test_out_of_range_location_is_skipped(self, tmp_path):
        issues = [
            _issue(IssueSeverity.ERROR, [IssueLocation("vertex", 999)])
        ]
        rr = render_mesh(
            _surface_mesh(), tmp_path / "marked.png", issues=issues
        )
        assert rr.n_highlighted == 0
        assert _valid_png(rr.path)


class TestRenderErrors:

    def test_invalid_view_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown view"):
            render_mesh(_surface_mesh(), tmp_path / "x.png", view="diagonal")

    def test_invalid_representation_raises(self, tmp_path):
        with pytest.raises(ValueError, match="representation"):
            render_mesh(
                _surface_mesh(), tmp_path / "x.png", representation="solid"
            )

    def test_invalid_opacity_raises(self, tmp_path):
        with pytest.raises(ValueError, match="opacity"):
            render_mesh(_surface_mesh(), tmp_path / "x.png", opacity=1.5)

    def test_invalid_window_size_raises(self, tmp_path):
        with pytest.raises(ValueError, match="width/height"):
            render_mesh(_surface_mesh(), tmp_path / "x.png", width=10)

    def test_invalid_zoom_raises(self, tmp_path):
        with pytest.raises(ValueError, match="zoom"):
            render_mesh(_surface_mesh(), tmp_path / "x.png", zoom=-1.0)

    def test_missing_output_directory_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Output directory"):
            render_mesh(
                _surface_mesh(), tmp_path / "nope" / "x.png"
            )

    def test_offscreen_unavailable_raises_visualization_error(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(renderer_module, "_offscreen_available", lambda: False)
        with pytest.raises(VisualizationError, match="Offscreen"):
            render_mesh(_surface_mesh(), tmp_path / "x.png")

    def test_suffixless_path_gets_png_extension(self, tmp_path):
        rr = render_mesh(_surface_mesh(), tmp_path / "noext")
        assert Path(rr.path).suffix == ".png"
        assert _valid_png(rr.path)


# ---------------------------------------------------------------------------
# MCP server integration
# ---------------------------------------------------------------------------


def _make_server():
    server = MeshIssueExaminerServer(auto_register_operations=False)
    server._register_builtin_tools()
    return server


class TestMeshRenderTool:

    def test_tool_registered(self):
        server = _make_server()
        names = {t["name"] for t in server.get_tools()}
        assert "mesh_render" in names

    def test_render_via_handle_request(self, tmp_path):
        server = _make_server()
        server._mesh_cache["viz"] = _surface_mesh()
        out = tmp_path / "via_server.png"

        req = MCPRequest(
            method="mesh_render",
            params={"mesh_id": "viz", "output_path": str(out), "view": "front"},
        )
        resp = asyncio.run(server.handle_request(req))
        assert resp.error is None
        assert _valid_png(resp.result["path"])
        assert resp.result["width"] == 1200
        assert resp.result["bytes"] > 0
        assert resp.result["scalars_used"] is None

    def test_unknown_mesh_id_returns_business_error(self, tmp_path):
        server = _make_server()
        req = MCPRequest(
            method="mesh_render",
            params={"mesh_id": "ghost", "output_path": str(tmp_path / "g.png")},
        )
        resp = asyncio.run(server.handle_request(req))
        assert resp.error is not None
        assert resp.error["code"] == "business_error"
        assert "ghost" in resp.error["message"]

    def test_highlight_without_inspection_returns_error(self, tmp_path):
        server = _make_server()
        server._mesh_cache["viz"] = _surface_mesh()
        req = MCPRequest(
            method="mesh_render",
            params={
                "mesh_id": "viz",
                "output_path": str(tmp_path / "h.png"),
                "highlight_issues": True,
            },
        )
        resp = asyncio.run(server.handle_request(req))
        assert resp.error is not None
        assert "mesh_inspect" in resp.error["message"]

    def test_default_output_path_writes_temp_png(self):
        server = _make_server()
        server._mesh_cache["viz_default_test"] = _surface_mesh()
        result = server._handle_render({"mesh_id": "viz_default_test"})
        assert result["path"].endswith("viz_default_test_render.png")
        assert _valid_png(result["path"])
        Path(result["path"]).unlink(missing_ok=True)

    def test_render_with_issues_and_severity_filter(self, tmp_path):
        server = _make_server()
        server._mesh_cache["viz"] = _surface_mesh()

        result_obj = InspectionResult(mesh_id="viz", file_path=None)
        result_obj.add_issue(
            _issue(
                IssueSeverity.CRITICAL,
                [IssueLocation("vertex", 0, coordinates=(0.0, 0.0, 0.0))],
            )
        )
        result_obj.add_issue(
            _issue(
                IssueSeverity.WARNING,
                [IssueLocation("vertex", 1, coordinates=(1.0, 0.0, 0.0))],
            )
        )
        server._result_cache["viz"] = result_obj

        out = tmp_path / "highlighted.png"
        result = server._handle_render(
            {
                "mesh_id": "viz",
                "output_path": str(out),
                "highlight_issues": True,
                "severity": "warning",
            }
        )
        assert result["n_highlighted"] == 1
        assert _valid_png(result["path"])

    def test_render_with_metric_scalars_via_server(self, tmp_path):
        server = _make_server()
        server._mesh_cache["viz"] = _surface_mesh()

        result_obj = InspectionResult(mesh_id="viz", file_path=None)
        result_obj.add_metrics(
            QualityMetrics(
                operation_id="test.op",
                metric_name="aspect_ratio",
                values=[1.0, 1.4, 1.8, 1.1],
                threshold=5.0,
            )
        )
        server._result_cache["viz"] = result_obj

        out = tmp_path / "metric.png"
        result = server._handle_render(
            {
                "mesh_id": "viz",
                "output_path": str(out),
                "scalars": "aspect_ratio",
            }
        )
        assert result["scalars_used"] == "aspect_ratio"
        assert _valid_png(result["path"])

    def test_material_scalars_via_server(self, tmp_path):
        server = _make_server()
        server._mesh_cache["vol"] = _volume_mesh(materials=np.array([10, 20]))
        out = tmp_path / "mat.png"
        result = server._handle_render(
            {"mesh_id": "vol", "output_path": str(out), "scalars": "material"}
        )
        assert result["scalars_used"] == "material"
        assert _valid_png(result["path"])
