"""
Unit tests for the mesh_save MCP tool (business + FastMCP layers).

mesh_save exposes utils.mesh_loader.save_mesh_to_file through MCP with
an extension-driven format selection and an optional server-side
allowed_save_dirs guard.  Registration on the business layer is opt-in
(enable_save_tool) so the default tool surface stays unchanged; the
FastMCP layer (create_mcp_server) enables it by default.
"""

import asyncio
from pathlib import Path

import pytest

from mesh_issue_examiner.mcp.server import (
    MCPRequest,
    MeshIssueExaminerServer,
    create_mcp_server,
)
from mesh_issue_examiner.utils.mesh_loader import load_mesh_from_file

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "e2e" / "sample_meshes"
NAS_FIXTURE = SAMPLE_DIR / "nas_mixed_volume" / "nas_mixed_volume.nas"

DEFAULT_TOOLS = {
    "mesh_load",
    "mesh_inspect",
    "mesh_get_issues",
    "mesh_get_quality_metrics",
    "mesh_render",
    "mesh_clear_cache",
}


def _nas_server(**kwargs) -> MeshIssueExaminerServer:
    """Server with the nas_mixed_volume fixture loaded and cached."""
    server = MeshIssueExaminerServer(enable_save_tool=True, **kwargs)
    mesh = load_mesh_from_file(str(NAS_FIXTURE))
    server._mesh_cache[mesh.mesh_id] = mesh
    return server


class TestToolRegistration:
    """mesh_save registration on both layers."""

    def test_default_server_tool_surface_unchanged(self):
        """Compatibility guard: default registration stays at six tools."""
        server = MeshIssueExaminerServer()
        names = {t["name"] for t in server.get_tools()}
        assert names == DEFAULT_TOOLS

    def test_enable_save_tool_adds_mesh_save(self):
        server = MeshIssueExaminerServer(enable_save_tool=True)
        tools = {t["name"]: t for t in server.get_tools()}
        assert "mesh_save" in tools
        schema = tools["mesh_save"]["input_schema"]
        assert schema["required"] == ["mesh_id", "output_path"]
        assert schema["properties"]["output_path"]["minLength"] == 1

    def test_dispatch_available_even_when_not_advertised(self):
        """handle_request routes mesh_save regardless of registration."""
        server = MeshIssueExaminerServer()  # enable_save_tool=False
        resp = asyncio.run(server.handle_request(MCPRequest(
            method="mesh_save",
            params={"mesh_id": "nope", "output_path": "x.nas"},
        )))
        # Dispatch happened (business ValueError, not unknown_method).
        assert resp.error is not None
        assert resp.error["code"] == "business_error"
        assert "nope" in resp.error["message"]

    def test_fastmcp_layer_exposes_mesh_save(self):
        mcp = create_mcp_server()
        tools = asyncio.run(mcp.list_tools())
        names = {t.name for t in tools}
        assert "mesh_save" in names
        assert "mesh_render" in names  # user render delta coexists


class TestHandleSave:
    """Business-layer _handle_save behavior via handle_request."""

    def test_save_nas_round_trip(self, tmp_path):
        """Saving the cached deck as .nas reproduces the parsed cells."""
        server = _nas_server()
        out = tmp_path / "saved.nas"
        resp = asyncio.run(server.handle_request(MCPRequest(
            method="mesh_save",
            params={"mesh_id": "nas_mixed_volume", "output_path": str(out)},
        )))
        assert resp.error is None, resp.error
        result = resp.result
        assert result["format"] == "nas"
        assert result["bytes"] > 0
        assert result["n_vertices"] == 30
        assert result["n_volume_cells"] == 3

        reloaded = load_mesh_from_file(str(out))
        assert reloaded.has_volume_cells
        assert set(reloaded.cell_types) == {"tet4", "hex20", "prism6"}
        assert reloaded.n_volume_cells == 3

    def test_save_vtu_round_trip(self, tmp_path):
        """A non-NASTRAN extension goes through the meshio writer."""
        server = _nas_server()
        out = tmp_path / "saved.vtu"
        resp = asyncio.run(server.handle_request(MCPRequest(
            method="mesh_save",
            params={"mesh_id": "nas_mixed_volume", "output_path": str(out)},
        )))
        assert resp.error is None, resp.error
        assert resp.result["format"] == "vtu"

        reloaded = load_mesh_from_file(str(out))
        assert reloaded.has_volume_cells
        assert "hex20" in reloaded.cell_types

    def test_unknown_mesh_id_returns_business_error(self, tmp_path):
        server = _nas_server()
        resp = asyncio.run(server.handle_request(MCPRequest(
            method="mesh_save",
            params={"mesh_id": "ghost", "output_path": str(tmp_path / "g.nas")},
        )))
        assert resp.error is not None
        assert resp.error["type"] == "ValueError"
        assert "ghost" in resp.error["message"]

    def test_missing_extension_rejected(self, tmp_path):
        server = _nas_server()
        resp = asyncio.run(server.handle_request(MCPRequest(
            method="mesh_save",
            params={
                "mesh_id": "nas_mixed_volume",
                "output_path": str(tmp_path / "no_extension"),
            },
        )))
        assert resp.error is not None
        assert resp.error["type"] == "ValueError"
        assert "extension" in resp.error["message"]

    def test_missing_parent_directory_rejected(self, tmp_path):
        server = _nas_server()
        resp = asyncio.run(server.handle_request(MCPRequest(
            method="mesh_save",
            params={
                "mesh_id": "nas_mixed_volume",
                "output_path": str(tmp_path / "no_such_dir" / "out.nas"),
            },
        )))
        assert resp.error is not None
        assert "Output directory" in resp.error["message"]


class TestAllowedSaveDirs:
    """Server-side output path restriction."""

    def test_output_outside_allowed_dirs_rejected(self, tmp_path):
        allowed = tmp_path / "exports"
        allowed.mkdir()
        server = _nas_server(allowed_save_dirs=[allowed])
        outside = tmp_path / "elsewhere.nas"  # sibling, not under exports/
        resp = asyncio.run(server.handle_request(MCPRequest(
            method="mesh_save",
            params={
                "mesh_id": "nas_mixed_volume",
                "output_path": str(outside),
            },
        )))
        assert resp.error is not None
        assert resp.error["type"] == "ValueError"
        assert "outside allowed save directories" in resp.error["message"]
        assert not outside.exists()

    def test_output_inside_allowed_dirs_accepted(self, tmp_path):
        allowed = tmp_path / "exports"
        allowed.mkdir()
        server = _nas_server(allowed_save_dirs=[str(allowed)])
        inside = allowed / "deck.nas"
        resp = asyncio.run(server.handle_request(MCPRequest(
            method="mesh_save",
            params={
                "mesh_id": "nas_mixed_volume",
                "output_path": str(inside),
            },
        )))
        assert resp.error is None, resp.error
        assert inside.exists()

    def test_no_restriction_when_unset(self, tmp_path):
        server = _nas_server()  # allowed_save_dirs=None
        out = tmp_path / "free.nas"
        resp = asyncio.run(server.handle_request(MCPRequest(
            method="mesh_save",
            params={
                "mesh_id": "nas_mixed_volume",
                "output_path": str(out),
            },
        )))
        assert resp.error is None, resp.error
        assert out.exists()
