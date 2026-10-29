"""
Unit tests for mesh_load server-side restrictions (audit 5.1/5.2).

Covers the ``allowed_load_dirs`` containment guard, the
``max_file_size_bytes`` limit, and their environment-variable defaults
(``MESH_EXAMINER_ALLOWED_DIRS`` / ``MESH_EXAMINER_MAX_FILE_BYTES``) on
the business-logic layer.  The FastMCP tools route through the same
``_handle_load`` handler, so one enforcement point covers both layers.
"""

import asyncio
import os
import shutil
from pathlib import Path

from mesh_issue_examiner.mcp.server import MCPRequest, MeshIssueExaminerServer

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "e2e" / "sample_meshes"
NAS_FIXTURE = SAMPLE_DIR / "nas_mixed_volume" / "nas_mixed_volume.nas"


def _make_server(**kwargs) -> MeshIssueExaminerServer:
    """Server without operation registration (load dispatch needs none)."""
    return MeshIssueExaminerServer(auto_register_operations=False, **kwargs)


def _load(server: MeshIssueExaminerServer, file_path) -> object:
    """Run a mesh_load request through handle_request."""
    return asyncio.run(server.handle_request(MCPRequest(
        method="mesh_load",
        params={"file_path": str(file_path)},
    )))


class TestAllowedLoadDirs:
    """Server-side input path restriction."""

    def test_load_outside_allowed_dirs_rejected(self, tmp_path):
        allowed = tmp_path / "meshes"
        allowed.mkdir()
        outside = tmp_path / "secret.nas"
        outside.write_bytes(b"BEGIN BULK\nENDATA\n")

        server = _make_server(allowed_load_dirs=[allowed])
        resp = _load(server, outside)

        assert resp.error is not None
        assert resp.error["type"] == "ValueError"
        assert "outside allowed load directories" in resp.error["message"]
        assert "meshes" in resp.error["message"]
        assert server._mesh_cache == {}

    def test_load_inside_allowed_dirs_succeeds(self, tmp_path):
        """Real NASTRAN fixture loaded from within the allowed dir."""
        allowed = tmp_path / "meshes"
        allowed.mkdir()
        deck = allowed / "deck.nas"
        shutil.copyfile(NAS_FIXTURE, deck)

        server = _make_server(allowed_load_dirs=[str(allowed)])
        resp = _load(server, deck)

        assert resp.error is None, resp.error
        assert resp.result["mesh_id"] == "deck"
        assert resp.result["format"] == "nas"
        assert resp.result["n_vertices"] > 0
        assert "deck" in server._mesh_cache

    def test_allowed_dirs_resolved_at_construction(self, tmp_path):
        server = _make_server(allowed_load_dirs=[str(tmp_path)])
        assert server.allowed_load_dirs == [Path(tmp_path).resolve()]

    def test_allowed_dirs_from_env_semicolon_separated(self, tmp_path, monkeypatch):
        """MESH_EXAMINER_ALLOWED_DIRS accepts comma/semicolon lists."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        monkeypatch.setenv(
            "MESH_EXAMINER_ALLOWED_DIRS", f"{dir_a};{dir_b}"
        )

        server = _make_server()
        assert server.allowed_load_dirs == [
            dir_a.resolve(), dir_b.resolve()
        ]

        # End-to-end: a file next to (not under) either dir is rejected.
        outside = tmp_path / "outside.nas"
        outside.write_bytes(b"BEGIN BULK\nENDATA\n")
        resp = _load(server, outside)
        assert resp.error is not None
        assert "outside allowed load directories" in resp.error["message"]

    def test_param_overrides_env(self, tmp_path, monkeypatch):
        env_dir = tmp_path / "env_only"
        env_dir.mkdir()
        allowed = tmp_path / "meshes"
        allowed.mkdir()
        monkeypatch.setenv("MESH_EXAMINER_ALLOWED_DIRS", str(env_dir))

        deck = allowed / "deck.nas"
        shutil.copyfile(NAS_FIXTURE, deck)

        server = _make_server(allowed_load_dirs=[allowed])
        assert server.allowed_load_dirs == [allowed.resolve()]

        resp = _load(server, deck)
        assert resp.error is None, resp.error


class TestMaxFileSize:
    """Server-side input size limit."""

    def test_load_exceeding_max_size_rejected(self, tmp_path):
        big = tmp_path / "big.nas"
        big.write_bytes(b"x" * 200)

        server = _make_server(max_file_size_bytes=100)
        resp = _load(server, big)

        assert resp.error is not None
        assert resp.error["type"] == "ValueError"
        assert "exceeding the configured limit of 100 bytes" in resp.error["message"]
        assert server._mesh_cache == {}

    def test_file_at_exact_limit_and_allowed_dir_succeeds(self, tmp_path):
        """A file exactly at the limit is accepted (limit is inclusive)."""
        allowed = tmp_path / "meshes"
        allowed.mkdir()
        deck = allowed / "deck.nas"
        shutil.copyfile(NAS_FIXTURE, deck)

        server = _make_server(
            allowed_load_dirs=[allowed],
            max_file_size_bytes=os.path.getsize(deck),
        )
        resp = _load(server, deck)

        assert resp.error is None, resp.error
        assert resp.result["mesh_id"] == "deck"

    def test_max_size_from_env(self, tmp_path, monkeypatch):
        big = tmp_path / "big.nas"
        big.write_bytes(b"x" * 200)
        monkeypatch.setenv("MESH_EXAMINER_MAX_FILE_BYTES", "100")

        server = _make_server()
        assert server.max_file_size_bytes == 100

        resp = _load(server, big)
        assert resp.error is not None
        assert "exceeding the configured limit of 100 bytes" in resp.error["message"]


class TestDefaults:
    """Unset configuration means unrestricted, preserving old behavior."""

    def test_unset_env_and_params_mean_unrestricted(self, monkeypatch):
        monkeypatch.delenv("MESH_EXAMINER_ALLOWED_DIRS", raising=False)
        monkeypatch.delenv("MESH_EXAMINER_MAX_FILE_BYTES", raising=False)

        server = _make_server()
        assert server.allowed_load_dirs is None
        assert server.max_file_size_bytes is None

    def test_invalid_env_size_ignored(self, monkeypatch):
        monkeypatch.setenv("MESH_EXAMINER_MAX_FILE_BYTES", "not-a-number")

        server = _make_server()
        assert server.max_file_size_bytes is None
