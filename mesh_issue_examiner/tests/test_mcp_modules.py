"""
Unit tests for mcp/server.py, mcp/handlers.py, and mcp/tools.py.

Tests data models, tool definitions, request routing, and basic server
functionality.
"""

import asyncio

import numpy as np
import pytest

from mesh_issue_examiner.mcp.server import MCPRequest, MCPResponse, MCPTool, MeshIssueExaminerServer
from mesh_issue_examiner.mcp.handlers import RequestHandler
from mesh_issue_examiner.mcp.tools import (
    ToolDefinition,
    SCHEMA_COMPONENTS,
    create_input_schema,
    create_array_schema,
    get_tool_definitions,
)


def _make_server():
    """Create a server with builtin tools but skip operation registration.

    The global ``registry`` singleton raises ValueError if the same operation
    is registered twice.  Using ``auto_register_operations=False`` avoids that
    while still letting us manually register the 5 builtin MCP tools.
    """
    server = MeshIssueExaminerServer(auto_register_operations=False)
    server._register_builtin_tools()
    return server


# ---------------------------------------------------------------------------
# Part 1: mcp/server.py data classes
# ---------------------------------------------------------------------------


class TestMCPRequest:
    """Tests for MCPRequest dataclass."""

    def test_construction_with_all_fields(self):
        req = MCPRequest(
            method="mesh_load",
            params={"file_path": "test.stl"},
            request_id="req-1",
        )
        assert req.method == "mesh_load"
        assert req.params == {"file_path": "test.stl"}
        assert req.request_id == "req-1"

    def test_default_request_id_is_none(self):
        req = MCPRequest(method="mesh_load", params={})
        assert req.request_id is None

    def test_params_is_dict(self):
        req = MCPRequest(method="list_operations", params={"category": "topological"})
        assert isinstance(req.params, dict)
        assert req.params["category"] == "topological"


class TestMCPResponse:
    """Tests for MCPResponse dataclass."""

    def test_construction_defaults(self):
        resp = MCPResponse()
        assert resp.result is None
        assert resp.error is None
        assert resp.request_id is None

    def test_construction_with_result(self):
        resp = MCPResponse(result={"mesh_id": "abc"}, request_id="r1")
        assert resp.result == {"mesh_id": "abc"}
        assert resp.error is None
        assert resp.request_id == "r1"

    def test_construction_with_error(self):
        err = {"code": "not_found", "message": "Mesh not found"}
        resp = MCPResponse(error=err, request_id="r2")
        assert resp.result is None
        assert resp.error == err

    def test_to_dict_with_result_only(self):
        resp = MCPResponse(result={"status": "ok"})
        d = resp.to_dict()
        assert "result" in d
        assert d["result"] == {"status": "ok"}
        assert "error" not in d

    def test_to_dict_with_error_only(self):
        err = {"code": "fail", "message": "bad"}
        resp = MCPResponse(error=err)
        d = resp.to_dict()
        assert "error" in d
        assert d["error"] == err
        assert "result" not in d

    def test_to_dict_error_takes_precedence_over_result(self):
        """When both result and error are set, error wins in to_dict."""
        resp = MCPResponse(result={"data": 1}, error={"code": "err"})
        d = resp.to_dict()
        assert "error" in d
        assert "result" not in d

    def test_to_dict_with_defaults(self):
        resp = MCPResponse()
        d = resp.to_dict()
        assert d == {"result": None}


class TestMCPTool:
    """Tests for MCPTool dataclass."""

    def test_construction(self):
        tool = MCPTool(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {}},
        )
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.input_schema == {"type": "object", "properties": {}}

    def test_to_dict(self):
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        tool = MCPTool(name="my_tool", description="desc", input_schema=schema)
        d = tool.to_dict()
        assert d == {
            "name": "my_tool",
            "description": "desc",
            "input_schema": schema,
        }


# ---------------------------------------------------------------------------
# Part 1 continued: MeshIssueExaminerServer
# ---------------------------------------------------------------------------


class TestMeshIssueExaminerServer:
    """Tests for MeshIssueExaminerServer.

    Uses ``auto_register_operations=False`` to avoid collisions with the
    global ``registry`` singleton, then manually registers builtin tools.
    """

    def test_construction_creates_empty_caches(self):
        server = _make_server()
        assert isinstance(server._mesh_cache, dict)
        assert len(server._mesh_cache) == 0
        assert isinstance(server._result_cache, dict)
        assert len(server._result_cache) == 0

    def test_get_tools_returns_five_definitions(self):
        server = _make_server()
        tools = server.get_tools()
        assert len(tools) == 5
        names = {t["name"] for t in tools}
        assert names == {
            "mesh_load",
            "mesh_inspect",
            "mesh_get_result",
            "mesh_clear_cache",
            "list_operations",
        }

    def test_handle_request_unknown_method_returns_error(self):
        server = _make_server()
        req = MCPRequest(method="nonexistent_method", params={})
        resp = asyncio.run(server.handle_request(req))
        assert resp.error is not None
        assert resp.error["code"] == "unknown_method"
        assert "nonexistent_method" in resp.error["message"]

    def test_list_operations_without_filter(self):
        """list_operations with no category or library returns all registered ops."""
        server = _make_server()
        req = MCPRequest(method="list_operations", params={})
        resp = asyncio.run(server.handle_request(req))
        assert resp.error is None
        assert isinstance(resp.result, list)

    def test_list_operations_with_category_filter(self):
        server = _make_server()
        req = MCPRequest(
            method="list_operations", params={"category": "topological"}
        )
        resp = asyncio.run(server.handle_request(req))
        assert resp.error is None
        assert isinstance(resp.result, list)

    def test_clear_cache(self):
        """Clearing cache without a mesh_id empties both caches."""
        server = _make_server()
        # Manually populate caches to verify clearing
        server._mesh_cache["dummy"] = "something"
        server._result_cache["dummy"] = "something"
        req = MCPRequest(method="mesh_clear_cache", params={})
        resp = asyncio.run(server.handle_request(req))
        assert resp.error is None
        assert resp.result["status"] == "cleared"
        assert len(server._mesh_cache) == 0
        assert len(server._result_cache) == 0

    def test_get_result_nonexistent_mesh_returns_error(self):
        """handle_request catches ValueError from _handle_get_result and wraps it."""
        server = _make_server()
        req = MCPRequest(
            method="mesh_get_result", params={"mesh_id": "does_not_exist"}
        )
        resp = asyncio.run(server.handle_request(req))
        # The server catches all exceptions, so ValueError is wrapped into an error response
        assert resp.error is not None
        assert resp.error["type"] == "ValueError"
        assert "does_not_exist" in resp.error["message"]

    def test_inspect_nonexistent_mesh_returns_error(self):
        """handle_request catches ValueError from _handle_inspect and wraps it."""
        server = _make_server()
        req = MCPRequest(
            method="mesh_inspect", params={"mesh_id": "no_such_mesh"}
        )
        resp = asyncio.run(server.handle_request(req))
        # ValueError is caught by the generic except handler
        assert resp.error is not None
        assert resp.error["type"] == "ValueError"
        assert "no_such_mesh" in resp.error["message"]

    def test_server_init_with_auto_register_does_not_raise(self):
        """MeshIssueExaminerServer(auto_register_operations=True) must not raise.

        Regression guard for Bug #5: before the fix, the constructor called
        _register_operations which tried to register NonManifoldEdgeDetection
        a second time, causing ValueError (not caught by except ImportError).
        """
        server = MeshIssueExaminerServer(auto_register_operations=True)
        assert isinstance(server._mesh_cache, dict)

    def test_server_tools_registered_after_auto_init(self):
        """After init with auto_register, the 5 builtin MCP tools are available."""
        server = MeshIssueExaminerServer(auto_register_operations=True)
        tools = server.get_tools()
        assert len(tools) == 5
        names = {t["name"] for t in tools}
        assert names == {
            "mesh_load",
            "mesh_inspect",
            "mesh_get_result",
            "mesh_clear_cache",
            "list_operations",
        }

    def test_server_init_with_custom_config(self):
        """Server should accept and store a custom OperationConfig."""
        from mesh_issue_examiner.core.models import OperationConfig
        cfg = OperationConfig(tolerance=1e-10, parallel_enabled=False)
        server = MeshIssueExaminerServer(
            config=cfg,
            auto_register_operations=False,
        )
        server._register_builtin_tools()
        assert server.config is cfg
        assert server.config.tolerance == 1e-10

    def test_server_init_without_auto_register_has_no_tools(self):
        """With auto_register_operations=False, no tools or ops are registered.

        This verifies the constructor's behavior when the flag is False.
        """
        server = MeshIssueExaminerServer(auto_register_operations=False)
        assert len(server._tools) == 0

    def test_server_repr(self):
        server = _make_server()
        r = repr(server)
        assert "MeshIssueExaminerServer" in r
        assert "5 tools" in r


# ---------------------------------------------------------------------------
# Part 2: mcp/handlers.py
# ---------------------------------------------------------------------------


class TestRequestHandler:
    """Tests for RequestHandler."""

    def test_register_and_retrieve(self):
        handler = RequestHandler()

        async def dummy(params):
            return params

        handler.register("test_method", dummy)
        retrieved = handler.get_handler("test_method")
        assert retrieved is dummy

    def test_has_handler_true_for_registered(self):
        handler = RequestHandler()

        async def dummy(params):
            return params

        handler.register("my_op", dummy)
        assert handler.has_handler("my_op") is True

    def test_has_handler_false_for_unknown(self):
        handler = RequestHandler()
        assert handler.has_handler("nonexistent") is False

    def test_get_handler_raises_key_error_for_unknown(self):
        handler = RequestHandler()
        with pytest.raises(KeyError):
            handler.get_handler("no_such_method")


# ---------------------------------------------------------------------------
# Part 3: mcp/tools.py
# ---------------------------------------------------------------------------


class TestToolDefinition:
    """Tests for ToolDefinition dataclass."""

    def test_construction_without_output_schema(self):
        td = ToolDefinition(
            name="tool_a",
            description="Tool A",
            input_schema={"type": "object"},
        )
        assert td.name == "tool_a"
        assert td.output_schema is None

    def test_to_dict_without_output_schema(self):
        td = ToolDefinition(
            name="tool_a",
            description="Tool A",
            input_schema={"type": "object"},
        )
        d = td.to_dict()
        assert "output_schema" not in d
        assert d == {
            "name": "tool_a",
            "description": "Tool A",
            "input_schema": {"type": "object"},
        }

    def test_construction_with_output_schema(self):
        td = ToolDefinition(
            name="tool_b",
            description="Tool B",
            input_schema={"type": "object"},
            output_schema={"type": "object", "properties": {"x": {"type": "int"}}},
        )
        assert td.output_schema is not None

    def test_to_dict_with_output_schema(self):
        out = {"type": "object", "properties": {"status": {"type": "string"}}}
        td = ToolDefinition(
            name="tool_b",
            description="Tool B",
            input_schema={"type": "object"},
            output_schema=out,
        )
        d = td.to_dict()
        assert "output_schema" in d
        assert d["output_schema"] == out


class TestCreateInputSchema:
    """Tests for create_input_schema helper."""

    def test_basic_construction(self):
        schema = create_input_schema(
            required=["mesh_id"],
            properties={"mesh_id": {"type": "string"}},
            description="Schema for loading",
        )
        assert schema["type"] == "object"
        assert schema["required"] == ["mesh_id"]
        assert schema["properties"] == {"mesh_id": {"type": "string"}}
        assert schema["description"] == "Schema for loading"

    def test_default_description_is_empty(self):
        schema = create_input_schema(
            required=["id"], properties={"id": {"type": "string"}}
        )
        assert schema["description"] == ""


class TestCreateArraySchema:
    """Tests for create_array_schema helper."""

    def test_basic_construction(self):
        schema = create_array_schema(item_type="string")
        assert schema["type"] == "array"
        assert schema["items"] == {"type": "string"}
        assert "min_items" not in schema
        assert "max_items" not in schema

    def test_with_min_max_items(self):
        schema = create_array_schema(item_type="integer", min_items=1, max_items=10)
        assert schema["min_items"] == 1
        assert schema["max_items"] == 10

    def test_min_items_only(self):
        schema = create_array_schema(item_type="number", min_items=2)
        assert schema["min_items"] == 2
        assert "max_items" not in schema


class TestSchemaComponents:
    """Tests for SCHEMA_COMPONENTS dict."""

    def test_expected_keys_exist(self):
        expected = {"mesh_id", "file_path", "operation_id", "severity", "category"}
        assert expected == set(SCHEMA_COMPONENTS.keys())

    def test_mesh_id_schema(self):
        sc = SCHEMA_COMPONENTS["mesh_id"]
        assert sc["type"] == "string"

    def test_file_path_schema(self):
        sc = SCHEMA_COMPONENTS["file_path"]
        assert sc["type"] == "string"

    def test_operation_id_schema(self):
        sc = SCHEMA_COMPONENTS["operation_id"]
        assert sc["type"] == "string"

    def test_severity_schema(self):
        sc = SCHEMA_COMPONENTS["severity"]
        assert sc["type"] == "string"
        assert "enum" in sc

    def test_category_schema(self):
        sc = SCHEMA_COMPONENTS["category"]
        assert sc["type"] == "string"
        assert "enum" in sc


class TestGetToolDefinitions:
    """Tests for get_tool_definitions function."""

    def test_returns_non_empty_list(self):
        from unittest.mock import patch

        with patch("mesh_issue_examiner.mcp.server.MeshIssueExaminerServer", side_effect=_make_server):
            defs = get_tool_definitions()
        assert isinstance(defs, list)
        assert len(defs) > 0

    def test_all_items_are_tool_definitions(self):
        from unittest.mock import patch

        with patch("mesh_issue_examiner.mcp.server.MeshIssueExaminerServer", side_effect=_make_server):
            defs = get_tool_definitions()
        for td in defs:
            assert isinstance(td, ToolDefinition)
