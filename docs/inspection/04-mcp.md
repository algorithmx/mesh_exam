# MCP layer — deep inspection

Files read: `mesh_issue_examiner/mcp/__init__.py`, `mcp/server.py` (1315 lines), `mcp/handlers.py`, `mcp/tools.py`, `mesh_issue_examiner/setup.py`, `mesh_issue_examiner/tests/test_mcp_modules.py`, `tests/test_mcp_save_tool.py`, `.mcp.json`. Note: tests live in `mesh_issue_examiner/tests/`, not `tests/`.

## Findings

### Module split
- `server.py` holds **all** functionality in one file: protocol dataclasses `MCPRequest`/`MCPResponse`/`MCPTool` (server.py:45,65,79), business-logic class `MeshIssueExaminerServer` (server.py:100), FastMCP factory `create_mcp_server` (server.py:868), and `main()` (server.py:1307).
- `handlers.py` exports only `RequestHandler`, a generic async handler registry with `register`/`get_handler`/`has_handler` (handlers.py:27,37,52). `_setup_default_handlers` is a no-op `pass` (handlers.py:23-25). **It is never used by `MeshIssueExaminerServer`** — routing is a hardcoded if/elif chain in `handle_request` (server.py:503-529); only tests import it (test_mcp_modules.py:14). Pure scaffolding.
- `tools.py` exports `ToolDefinition`, `get_tool_definitions`, `SCHEMA_COMPONENTS`, `create_input_schema`, `create_array_schema` (tools.py:9,29,51,77,101). `get_tool_definitions()` instantiates a throwaway `MeshIssueExaminerServer()` and copies `server._tools` (tools.py:36-47) — so it **never lists `mesh_save`** (backend default `enable_save_tool=False`). `SCHEMA_COMPONENTS`/`create_input_schema`/`create_array_schema` are not referenced by `server.py` (schemas are written inline in `_register_builtin_tools`).
- `__init__.py` re-exports `MeshIssueExaminerServer`, `create_mcp_server`, `get_tool_definitions` (mcp/__init__.py:5-12).

### Backend class `MeshIssueExaminerServer`
- Ctor: `config=None, auto_register_operations=True, enable_save_tool=False, allowed_save_dirs=None` (server.py:127-133). State: `config` (OperationConfig), `_mesh_cache: Dict[str, MeshInput]`, `_result_cache: Dict[str, InspectionResult]`, `_tools: Dict[str, MCPTool]`, `allowed_save_dirs` (each entry `Path(d).resolve()`, or `None` = unrestricted) (server.py:150-158). `__repr__` = `<MeshIssueExaminerServer: N tools, M cached meshes>` (server.py:859-860).
- All tool methods are private `_handle_*` taking a plain params dict; `params['key']` raises `KeyError` (→ `internal_error`) for missing required keys:
  - `_handle_load` (async, server.py:572): `file_path` (required), `mesh_id`, `library`, `material_array_name`. Returns `mesh_id, n_vertices, n_faces, bounding_box{min,max}, format` (server.py:594-603) plus `materials{count, ids, names, n_cells_assigned}` when `cell_materials` extracted (server.py:606-614).
  - `_handle_inspect` (async, server.py:618): `mesh_id` (req), `categories`, `operations` (XOR enforced). Operation IDs validated against `registry.list_all()` before workflow creation (server.py:636-643); categories expanded via `IssueCategory(c)` + `registry.list_by_category` (server.py:646-653); default = `create_full_workflow` (server.py:656). Times with `time.perf_counter` and stores `result.processing_time` (server.py:658-660). Returns `mesh_id, passed, processing_time, summary` (server.py:666-671).
  - `_handle_get_issues` (server.py:673): `mesh_id` (req), `severity`, `category`, `limit`, `offset`. Returns `total_count, returned_count, offset, issues[]` (server.py:701-706).
  - `_handle_get_quality_metrics` (server.py:708): `mesh_id` (req), `metric_name` (exact-match filter; unknown name → `metric_count: 0`, no error; test_mcp_modules.py:703-709). Returns `mesh_id, metric_count, metrics[]` (server.py:725-729).
  - `_handle_render` (server.py:731): `mesh_id` (req), `output_path`, `view`, `azimuth`, `elevation`, `zoom`, `width`, `height`, `color`, `show_edges`, `scalars`, `cmap`, `scalar_bar`, `highlight_issues`, `severity`. Output default `<tempdir>/<mesh_id>_render.png`, `.png` appended if no suffix (server.py:756-762). Returns `mesh_id, path, width, height, bytes, n_highlighted, scalars_used` (server.py:788-796). `render_mesh` lazy-imported so pyvista/VTK stay off the startup path (server.py:766-767).
  - `_handle_save` (server.py:818): `mesh_id`, `output_path` (both required), `library`. Returns `mesh_id, path, format (suffix[1:].lower()), bytes, n_vertices, n_volume_cells` (server.py:848-857).
  - `_handle_clear_cache` (server.py:798): optional `mesh_id`; falsy value (`""`) is treated as "clear all" via `if mesh_id:` (server.py:802). Calls `mesh.invalidate_cache()` on evicted meshes when present (server.py:804-805,810-811). Returns `{status: 'cleared', cleared_ids}` (server.py:816).

### FastMCP wrapper
- `create_mcp_server(config=None, auto_register_operations=True)` (server.py:868-871): lazy-imports `mcp.server.fastmcp.FastMCP` (server.py:891), builds a `MeshIssueExaminerServer(enable_save_tool=True)` backend (server.py:893-897), wraps it as `FastMCP("mesh-issue-examiner")` (server.py:899). **It accepts no `allowed_save_dirs` parameter**, so the stdio-launched server can never have a save allowlist.
- Seven `@mcp.tool()` functions mirror backend methods 1:1; schemas come from Python type hints + Google-style docstrings, **not** from the backend's hand-written `input_schema` dicts. Sync vs async split: `mesh_load`/`mesh_inspect`/`mesh_save` are `async` (server.py:903,959,1187); `mesh_get_issues`/`mesh_get_quality_metrics`/`mesh_render`/`mesh_clear_cache` are plain sync functions (server.py:1012,1060,1097,1233).
- Each wrapper builds a params dict (omitting `None`s) and calls `backend._handle_*(params)` **directly, bypassing `handle_request`** — so backend exceptions propagate raw to FastMCP instead of becoming `{code: business_error}` envelopes. All tools return `json.dumps(result, indent=2)` strings (e.g. server.py:955,1056,1183).

### Where save registration differs (pinpoint)
1. Backend advertisement is opt-in: `mesh_save` is added to `self._tools` only under `if self._enable_save_tool:` (server.py:424-425), default `False` (server.py:131) — done so `tools/list` stays 6 tools (rationale in docstring server.py:140-145; guarded by test_mcp_save_tool.py:47-59).
2. The FastMCP layer constructs its backend with `enable_save_tool=True` (server.py:896) **and** registers an independent `@mcp.tool()` `mesh_save` wrapper (server.py:1187-1229); `mcp.list_tools()` therefore always includes it (test_mcp_save_tool.py:73-78).
3. Dispatch is decoupled from advertisement: `handle_request` routes `mesh_save` unconditionally (server.py:521-522), so a default server that never advertises the tool still executes it (test_mcp_save_tool.py:61-71).
4. `get_tool_definitions()` in tools.py constructs with defaults, so mesh_save never appears there (tools.py:36-47).

### Resources
- `mesh://operations` → `json.dumps([op.to_dict() for op in registry.list_all()], indent=2)` (server.py:1258-1267).
- `mesh://{mesh_id}/info` → `mesh.to_dict()` from `_mesh_cache`; on miss returns a JSON body `{"error": "Mesh not found: ...", "hint": "Load a mesh first using mesh_load."}` instead of raising (server.py:1269-1282).
- `mesh://{mesh_id}/summary` → `mesh_id, passed, processing_time, summary` from `_result_cache`; on miss returns `{"error": "No inspection result for mesh: ...", "hint": "Run mesh_inspect first."}` (server.py:1284-1302). Resources read backend dicts directly (no handler reuse).

### Error paths (via `handle_request`, server.py:490-570)
- Unknown method → `{'code': 'unknown_method', 'message': f"Unknown method: {method}"}` (server.py:529).
- Unknown mesh ID → `ValueError` `"Mesh not found: {mesh_id}"` from inspect/render/save (server.py:631,736,823).
- Missing inspection result → `ValueError` `"No inspection result for mesh: {id}. Run mesh_inspect first."` from get_issues/get_quality_metrics/render (when `highlight_issues` or metric `scalars` requested, server.py:742-748) (server.py:678-681,713-716,745-748).
- Unknown operation IDs → `ValueError` `"Unknown operation(s): {ids}. Discover IDs via the mesh://operations resource."` (server.py:639-642).
- Both `categories` and `operations` → `ValueError` `"Provide 'categories' or 'operations', not both."` (server.py:625-627).
- Invalid pagination is **not** an error: `limit = max(1, min(limit, 200))`, `offset = max(0, offset)` — clamped silently (server.py:697-698). Invalid severity/category strings raise via enum constructors `IssueSeverity()/IssueCategory()` (server.py:687,692) → business_error.
- `mesh_load` on missing file → `FileNotFoundError "Mesh file not found: {resolved}"` (server.py:578-579).
- Three-tier mapping: `ValueError`/`FileNotFoundError` → `business_error` (server.py:537-547); `MeshExaminerError` subclasses (e.g. `UnsupportedFormatError`) → `business_error` with `type` set (server.py:548-558; test_mcp_modules.py:570-587); any other exception → `internal_error` with generic `"Internal server error"` and full traceback logged server-side (server.py:559-570; tests test_mcp_modules.py:534-559).
- Quirk: success/error selection uses `error if 'error' in locals() else None` (server.py:533).

### mesh_save specifics
- Format from extension: `.nas/.bdf` → native NASTRAN writer (quadratic CHEXA/CPENTA midside nodes retained); other extensions via meshio/pyvista; `library` enum adds `meshio` vs load's 3 libraries (server.py:427-462,846).
- Failure modes in order: unknown mesh_id (server.py:822-823) → no extension `"output_path must include a file extension..."` (server.py:826-830) → missing parent `"Output directory does not exist"` (server.py:831-832) → allowlist rejection (server.py:834-844).
- Allowlist enforcement: exact string equality or `str(out_path).startswith(str(d) + os.sep)` against each pre-resolved dir (server.py:834-844) → `ValueError "Output '{path}' is outside allowed save directories: [...]"`. `None` = unrestricted (test_mcp_save_tool.py:190-201). Round-trip: NAS fixture (30 verts, 3 volume cells, tet4/hex20/prism6) survives save+reload; `.vtu` keeps hex20 (test_mcp_save_tool.py:84-118).

### Launch
- Console script: `mesh-examiner=mesh_issue_examiner.mcp.server:main` (setup.py:61-65); `main()` = `create_mcp_server().run()` (server.py:1307-1314), stdio by default, `transport="streamable-http"` optional (server.py:886-889). No `__main__.py` in the mcp package.
- `.mcp.json` (repo root) launches the same server via `python -c "from ...server import create_mcp_server; create_mcp_server().run()"`, type stdio (.mcp.json:3-9) — bypassing the console script entirely.
- Package `mesh-issue-examiner` 0.1.0; install_requires `numpy>=1.21.0`, `mcp>=1.0.0`; extras trimesh/pyvista/pymeshlab/io/dev/all (setup.py:12-59). `find_packages(exclude=["tests","tests.*"])` does **not** exclude the in-package `mesh_issue_examiner.tests` (setup.py:21).

### Concurrency / state lifetime
- No locking anywhere: `_mesh_cache`/`_result_cache` are plain dicts mutated with no `threading`/`asyncio` primitives (imports server.py:16-35). Sync FastMCP tools (get_issues, render, clear_cache) share these dicts with async tools; interleaved calls can observe partial state (e.g. get_issues during a concurrent mesh_inspect's cache overwrite at server.py:663).
- Lifetime hooks: only `invalidate_cache()` on meshes at clear (server.py:804-805,810-811); no TTL, no size cap, no eviction — memory grows until `mesh_clear_cache` or process exit.
- Every `create_mcp_server()` and every `get_tool_definitions()` call builds a fresh backend instance (server.py:893, tools.py:38); state is strictly per-instance.

## Numbers & limits
- Issues pagination: `limit` default 50, clamped `[1, 200]`; `offset` default 0, clamped `>= 0` (server.py:697-698; schema bounds server.py:279-291).
- Render: width/height default 1200/800, schema min 64 / max 8192; `zoom` `exclusiveMinimum: 0`; default color `lightblue`, `show_edges=True`, `cmap="viridis"`, `scalar_bar=True`; default highlight severities `('critical', 'error')` (server.py:354-417,754,776-782).
- `minLength: 1` on `file_path`/`mesh_id` (load, inspect, get_issues, metrics) and `output_path` (save); `operations` array `minItems: 1` (server.py:180-185,246,450).
- Enums: severity `info/warning/error/critical`; category `topological/geometric/quality/orientation`; load `library` trimesh/pyvista/pymeshlab; save `library` adds `meshio`; `view` has 7 values iso/front/back/left/right/top/bottom (server.py:189,231,271,354,454).
- All tool/resource payloads serialized with `json.dumps(..., indent=2)`.
- setup.py: `python_requires=">=3.8"` (setup.py:36) — but `create_mcp_server`'s tool functions use PEP 604 annotations evaluated at def time, e.g. `mesh_id: str | None = None` (server.py:906), which raises `TypeError` on Python < 3.10; no `from __future__ import annotations` is present.
- Version 0.1.0, author "MiniMax Agent" (setup.py:14-15).

## Candidate additions for the HTML note
- `handlers.py` (`RequestHandler`) and the schema helpers in `tools.py` (`SCHEMA_COMPONENTS`, `create_input_schema`, `create_array_schema`) are dead code relative to the server — routing is a hardcoded if/elif chain, schemas are inline, and `get_tool_definitions()` builds an ephemeral backend that can never advertise `mesh_save`.
- Save-registration asymmetry pinned to four code points: opt-in backend advertisement (server.py:424), FastMCP forcing `enable_save_tool=True` + its own decorator (server.py:896,1187), always-on dispatch in `handle_request` (server.py:521), default-off in `get_tool_definitions` (tools.py:38).
- FastMCP tool wrappers bypass `handle_request` entirely, calling `_handle_*` directly — the three-tier error envelope (`business_error`/`internal_error`) exists only on the internal dispatch path; over stdio, clients see raw exceptions instead.
- FastMCP tools are JSON-string-returning; sync/async split (load/inspect/save async; issues/metrics/render/clear sync) with no locks over shared caches.
- `create_mcp_server` cannot configure `allowed_save_dirs`, so the deployed stdio server always has unrestricted `mesh_save` output paths.
- Resources degrade to error+hint JSON bodies (HTTP-200-style) rather than exceptions; the same miss conditions raise `ValueError` in tools.
- Launch paths: `mesh-examiner` console script → `main()` → `.run()` (stdio default), plus a parallel `.mcp.json` launcher using `python -c` with a hardcoded conda path.
- `python_requires>=3.8` conflicts with PEP 604 `str | None` annotations in `create_mcp_server` (3.10+ required in practice).

## Corrections
none
