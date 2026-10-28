"""
MCP server implementation for mesh issue examination.

This module provides the main server class that handles MCP protocol
communications and routes requests to appropriate operations.

Two layers are provided:

1. :class:`MeshIssueExaminerServer` – internal business-logic layer that
   manages mesh loading, inspection workflows, and result caching.
2. :func:`create_mcp_server` / :func:`main` – thin FastMCP wrapper that
   exposes the business logic via the standard MCP protocol (JSON-RPC 2.0
   over stdio / streamable-http).
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

import numpy as np

from ..core.exceptions import MeshExaminerError
from ..core.models import (
    MeshInput,
    InspectionResult,
    OperationConfig,
    IssueCategory,
    IssueSeverity,
)
from ..operations.operation_registry import registry
from ..utils.mesh_loader import load_mesh_from_file, save_mesh_to_file

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal data classes (kept for backward compatibility & direct testing)
# ---------------------------------------------------------------------------


@dataclass
class MCPRequest:
    """Represents an MCP protocol request.

    The ``params`` field is intentionally typed as ``Dict[str, Any]`` to
    accommodate varying parameter schemas for different MCP methods.
    For type-safe access, validate parameters against the tool schemas
    returned by ``list_operations`` before processing.

    Example:
        >>> request = MCPRequest(
        ...     method='mesh_load',
        ...     params={'file_path': '/path/to/mesh.stl'}
        ... )
    """
    method: str
    params: Dict[str, Any]
    request_id: Optional[str] = None


@dataclass
class MCPResponse:
    """Represents an MCP protocol response."""
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        if self.error:
            return {'error': self.error}
        return {'result': self.result}


@dataclass
class MCPTool:
    """Represents an MCP tool definition."""
    name: str
    description: str
    input_schema: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'input_schema': self.input_schema
        }


# ---------------------------------------------------------------------------
# Business-logic server (transport-agnostic)
# ---------------------------------------------------------------------------


class MeshIssueExaminerServer:
    """
    MCP server for mesh issue examination.

    Provides tools for:
    - Loading meshes from files
    - Running inspection operations (by category, by ID, or all)
    - Retrieving filtered/paginated issues
    - Retrieving quality metrics
    - Cache management

    Usage:
        server = MeshIssueExaminerServer()

        # Load a mesh
        load_result = await server.handle_request(MCPRequest(
            method='mesh_load',
            params={'file_path': 'example.stl'}
        ))

        # Inspect the mesh (runs all operations by default)
        inspect_result = await server.handle_request(MCPRequest(
            method='mesh_inspect',
            params={'mesh_id': 'example'}
        ))
    """

    def __init__(
        self,
        config: Optional[OperationConfig] = None,
        auto_register_operations: bool = True,
        enable_save_tool: bool = False,
        allowed_save_dirs: Optional[List[Union[str, Path]]] = None,
    ):
        """
        Initialize the MCP server.

        Args:
            config: Optional configuration for operations
            auto_register_operations: Whether to auto-register built-in operations
            enable_save_tool: Register the ``mesh_save`` tool in the default
                tool surface. Off by default so callers that pin the tool
                list (``tools/list`` assertions) see an unchanged surface;
                the FastMCP layer (:func:`create_mcp_server`) enables it.
                The ``mesh_save`` *dispatch* in :meth:`handle_request` is
                always available regardless of this flag.
            allowed_save_dirs: Optional list of directories ``mesh_save``
                may write into. When provided, an output_path that does
                not resolve under one of these directories is rejected.
        """
        self.config = config or OperationConfig()
        self._mesh_cache: Dict[str, MeshInput] = {}
        self._result_cache: Dict[str, InspectionResult] = {}
        self._tools: Dict[str, MCPTool] = {}
        self._enable_save_tool = enable_save_tool
        self.allowed_save_dirs = (
            [Path(d).resolve() for d in allowed_save_dirs]
            if allowed_save_dirs is not None else None
        )

        if auto_register_operations:
            self._register_builtin_tools()

    def _register_builtin_tools(self):
        """Register built-in MCP tools."""
        # Mesh load tool
        self._tools['mesh_load'] = MCPTool(
            name='mesh_load',
            description=(
                'Load a mesh file for inspection. Reads a mesh from disk '
                '(STL, OBJ, PLY, VTK, NAS, and 30+ other formats) and '
                'stores it in the server cache. Returns mesh_id, vertex/face '
                'counts, bounding box, and format.'
            ),
            input_schema={
                'type': 'object',
                'properties': {
                    'file_path': {
                        'type': 'string',
                        'description': 'Absolute path to the mesh file on the server filesystem.',
                        'minLength': 1
                    },
                    'mesh_id': {
                        'type': 'string',
                        'description': 'Custom identifier. Defaults to filename stem.',
                        'minLength': 1
                    },
                    'library': {
                        'type': 'string',
                        'enum': ['trimesh', 'pyvista', 'pymeshlab'],
                        'description': 'Preferred loader. Auto-detected when omitted.'
                    },
                    'material_array_name': {
                        'type': 'string',
                        'description': (
                            'Name of the per-cell data array holding material '
                            'IDs (e.g. "material", "gmsh:physical", '
                            '"region_id"). When provided, material '
                            'assignments are extracted so multi-material '
                            'operations (quality_3d.detect_material_interfaces, '
                            'check_interface_quality, per_material_statistics) '
                            'can run.'
                        )
                    }
                },
                'required': ['file_path']
            }
        )

        # Mesh inspect tool
        self._tools['mesh_inspect'] = MCPTool(
            name='mesh_inspect',
            description=(
                'Execute inspection operations on a previously loaded mesh. '
                'Provide ONE of categories or operations to run a subset; '
                'omit both to run every registered operation. Returns summary '
                'only — use mesh_get_issues and mesh_get_quality_metrics to '
                'drill down.'
            ),
            input_schema={
                'type': 'object',
                'properties': {
                    'mesh_id': {
                        'type': 'string',
                        'description': 'Identifier returned by mesh_load.',
                        'minLength': 1
                    },
                    'categories': {
                        'type': 'array',
                        'items': {
                            'type': 'string',
                            'enum': ['topological', 'geometric', 'quality', 'orientation']
                        },
                        'description': (
                            'Run all operations in these categories. '
                            'Mutually exclusive with operations.'
                        )
                    },
                    'operations': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': (
                            'Run these specific operation IDs. '
                            'Discover IDs via the mesh://operations resource. '
                            'Mutually exclusive with categories.'
                        ),
                        'minItems': 1
                    }
                },
                'required': ['mesh_id']
            }
        )

        # Get issues (replaces mesh_get_result)
        self._tools['mesh_get_issues'] = MCPTool(
            name='mesh_get_issues',
            description=(
                'Retrieve detected issues from a completed inspection. '
                'Call after mesh_inspect. Supports filtering by severity '
                'and category, and pagination to control response size.'
            ),
            input_schema={
                'type': 'object',
                'properties': {
                    'mesh_id': {
                        'type': 'string',
                        'description': 'Identifier from mesh_load.',
                        'minLength': 1
                    },
                    'severity': {
                        'type': 'string',
                        'enum': ['info', 'warning', 'error', 'critical'],
                        'description': 'Filter by severity. Returns all when omitted.'
                    },
                    'category': {
                        'type': 'string',
                        'enum': ['topological', 'geometric', 'quality', 'orientation'],
                        'description': 'Filter by category. Returns all when omitted.'
                    },
                    'limit': {
                        'type': 'integer',
                        'description': 'Max issues to return (1-200, default 50).',
                        'default': 50,
                        'minimum': 1,
                        'maximum': 200
                    },
                    'offset': {
                        'type': 'integer',
                        'description': 'Skip this many issues for pagination.',
                        'default': 0,
                        'minimum': 0
                    }
                },
                'required': ['mesh_id']
            }
        )

        # Get quality metrics
        self._tools['mesh_get_quality_metrics'] = MCPTool(
            name='mesh_get_quality_metrics',
            description=(
                'Retrieve quality metrics from a completed inspection. '
                'Call after mesh_inspect. Returns per-metric statistics '
                '(min, max, mean, median, std, p5, p95) and violation counts.'
            ),
            input_schema={
                'type': 'object',
                'properties': {
                    'mesh_id': {
                        'type': 'string',
                        'description': 'Identifier from mesh_load.',
                        'minLength': 1
                    },
                    'metric_name': {
                        'type': 'string',
                        'description': (
                            'Specific metric to retrieve (e.g. "aspect_ratio", '
                            '"scaled_jacobian"). Returns all metrics when omitted.'
                        )
                    }
                },
                'required': ['mesh_id']
            }
        )

        # Save mesh to file (opt-in: keeps the default tool surface stable)
        if self._enable_save_tool:
            self._tools['mesh_save'] = MCPTool(
                name='mesh_save',
                description=(
                    'Save a previously loaded mesh to a file. The output '
                    'format is driven by the file extension: .nas/.bdf use '
                    'the native NASTRAN writer (quadratic CHEXA/CPENTA '
                    'midside nodes retained); other supported extensions '
                    'are written via meshio/pyvista. The parent directory '
                    'must exist.'
                ),
                input_schema={
                    'type': 'object',
                    'properties': {
                        'mesh_id': {
                            'type': 'string',
                            'description': 'Identifier returned by mesh_load.',
                            'minLength': 1
                        },
                        'output_path': {
                            'type': 'string',
                            'description': (
                                'Absolute path for the output file; the '
                                'extension selects the format (e.g. .nas, '
                                '.vtu, .msh, .stl).'
                            ),
                            'minLength': 1
                        },
                        'library': {
                            'type': 'string',
                            'enum': ['trimesh', 'pyvista', 'pymeshlab', 'meshio'],
                            'description': (
                                'Writer override for non-NASTRAN formats. '
                                'Auto-detected when omitted.'
                            )
                        }
                    },
                    'required': ['mesh_id', 'output_path']
                }
            )

        # Clear cache
        self._tools['mesh_clear_cache'] = MCPTool(
            name='mesh_clear_cache',
            description=(
                'Free cached mesh data and inspection results. '
                'Call to release memory or force a fresh reload.'
            ),
            input_schema={
                'type': 'object',
                'properties': {
                    'mesh_id': {
                        'type': 'string',
                        'description': (
                            'Specific mesh to clear. If omitted, clears all '
                            'cached meshes and results.'
                        )
                    }
                }
            }
        )

    def get_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools."""
        return [tool.to_dict() for tool in self._tools.values()]

    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """
        Handle an incoming MCP request.

        Args:
            request: MCPRequest object

        Returns:
            MCPResponse object
        """
        try:
            method = request.method

            if method == 'tools/list':
                result = self.get_tools()

            elif method == 'mesh_load':
                result = await self._handle_load(request.params)

            elif method == 'mesh_inspect':
                result = await self._handle_inspect(request.params)

            elif method == 'mesh_get_issues':
                result = self._handle_get_issues(request.params)

            elif method == 'mesh_get_quality_metrics':
                result = self._handle_get_quality_metrics(request.params)

            elif method == 'mesh_save':
                result = self._handle_save(request.params)

            elif method == 'mesh_clear_cache':
                result = self._handle_clear_cache(request.params)

            else:
                result = None
                error = {'code': 'unknown_method', 'message': f"Unknown method: {method}"}

            return MCPResponse(
                result=result,
                error=error if 'error' in locals() else None,
                request_id=request.request_id
            )

        except (ValueError, FileNotFoundError) as e:
            # Business-logic errors — safe to expose to caller
            return MCPResponse(
                result=None,
                error={
                    'code': 'business_error',
                    'message': str(e),
                    'type': type(e).__name__
                },
                request_id=request.request_id
            )
        except MeshExaminerError as e:
            # Domain errors (UnsupportedFormatError, MeshLoadError, OperationError, etc.)
            return MCPResponse(
                result=None,
                error={
                    'code': 'business_error',
                    'message': str(e),
                    'type': type(e).__name__
                },
                request_id=request.request_id
            )
        except Exception as e:
            # System errors — do NOT expose internals
            logger.error("Unhandled error in %s: %s", request.method, e, exc_info=True)
            return MCPResponse(
                result=None,
                error={
                    'code': 'internal_error',
                    'message': 'Internal server error',
                    'type': 'internal_error'
                },
                request_id=request.request_id
            )

    async def _handle_load(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle mesh load request."""
        file_path = params['file_path']

        # Resolve and validate path to prevent traversal attacks
        resolved = Path(file_path).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Mesh file not found: {resolved}")

        mesh_id = params.get('mesh_id')
        library = params.get('library')
        material_array_name = params.get('material_array_name')

        mesh_input = load_mesh_from_file(
            resolved,
            mesh_id=mesh_id,
            library=library,
            material_array_name=material_array_name
        )

        self._mesh_cache[mesh_input.mesh_id] = mesh_input

        result = {
            'mesh_id': mesh_input.mesh_id,
            'n_vertices': mesh_input.n_vertices,
            'n_faces': mesh_input.n_faces,
            'bounding_box': {
                'min': mesh_input.bounding_box[0].tolist(),
                'max': mesh_input.bounding_box[1].tolist()
            },
            'format': mesh_input.source_format
        }

        # Materials summary when extraction was requested and succeeded
        cell_materials = mesh_input.cell_materials
        if cell_materials is not None:
            unique_ids = np.unique(cell_materials).tolist()
            result['materials'] = {
                'count': len(unique_ids),
                'ids': unique_ids,
                'names': mesh_input.material_names or {},
                'n_cells_assigned': int(len(cell_materials))
            }

        return result

    async def _handle_inspect(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle mesh inspection request."""
        mesh_id = params['mesh_id']
        categories = params.get('categories')
        operations = params.get('operations')

        if categories is not None and operations is not None:
            raise ValueError(
                "Provide 'categories' or 'operations', not both."
            )

        mesh = self._mesh_cache.get(mesh_id)
        if mesh is None:
            raise ValueError(f"Mesh not found: {mesh_id}")

        # Determine which operations to run
        if operations is not None:
            # Validate operation IDs against registry
            available = {op.operation_id for op in registry.list_all()}
            invalid = [o for o in operations if o not in available]
            if invalid:
                raise ValueError(
                    f"Unknown operation(s): {', '.join(invalid)}. "
                    f"Discover IDs via the mesh://operations resource."
                )
            workflow = registry.create_workflow(operations, config=self.config)
        elif categories is not None:
            # Build workflow from categories
            op_ids = []
            for c in categories:
                cat = IssueCategory(c)
                op_ids.extend(
                    op.operation_id
                    for op in registry.list_by_category(cat)
                )
            workflow = registry.create_workflow(op_ids, config=self.config)
        else:
            # Default: run all operations
            workflow = registry.create_full_workflow(config=self.config)

        start_time = time.perf_counter()
        result = workflow(mesh)
        result.processing_time = time.perf_counter() - start_time

        # Cache result
        self._result_cache[mesh_id] = result

        # Return summary only
        return {
            "mesh_id": mesh_id,
            "passed": result.passed,
            "processing_time": result.processing_time,
            "summary": result.summary,
        }

    def _handle_get_issues(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve detected issues with filtering and pagination."""
        mesh_id = params['mesh_id']
        result = self._result_cache.get(mesh_id)
        if result is None:
            raise ValueError(
                f"No inspection result for mesh: {mesh_id}. "
                f"Run mesh_inspect first."
            )

        issues = result.issues

        # Filter by severity
        if 'severity' in params:
            sev = IssueSeverity(params['severity'])
            issues = [i for i in issues if i.severity == sev]

        # Filter by category
        if 'category' in params:
            cat = IssueCategory(params['category'])
            issues = [i for i in issues if i.category == cat]

        # Pagination
        total = len(issues)
        limit = max(1, min(params.get('limit', 50), 200))
        offset = max(0, params.get('offset', 0))
        page = issues[offset : offset + limit]

        return {
            'total_count': total,
            'returned_count': len(page),
            'offset': offset,
            'issues': [i.to_dict() for i in page],
        }

    def _handle_get_quality_metrics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve quality metrics from a completed inspection."""
        mesh_id = params['mesh_id']
        result = self._result_cache.get(mesh_id)
        if result is None:
            raise ValueError(
                f"No inspection result for mesh: {mesh_id}. "
                f"Run mesh_inspect first."
            )

        metrics = result.quality_metrics

        # Optional filter by metric name
        if 'metric_name' in params:
            metric_name = params['metric_name']
            metrics = [m for m in metrics if m.metric_name == metric_name]

        return {
            'mesh_id': mesh_id,
            'metric_count': len(metrics),
            'metrics': [m.to_dict() for m in metrics],
        }

    def _handle_clear_cache(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle clear cache request."""
        mesh_id = params.get('mesh_id')

        if mesh_id:
            mesh = self._mesh_cache.pop(mesh_id, None)
            if mesh is not None and hasattr(mesh, 'invalidate_cache'):
                mesh.invalidate_cache()
            self._result_cache.pop(mesh_id, None)
            cleared = [mesh_id]
        else:
            for mesh in self._mesh_cache.values():
                if hasattr(mesh, 'invalidate_cache'):
                    mesh.invalidate_cache()
            cleared = list(self._mesh_cache.keys())
            self._mesh_cache.clear()
            self._result_cache.clear()

        return {'status': 'cleared', 'cleared_ids': cleared}

    def _handle_save(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle mesh save request (extension-driven format selection)."""
        mesh_id = params['mesh_id']
        mesh = self._mesh_cache.get(mesh_id)
        if mesh is None:
            raise ValueError(f"Mesh not found: {mesh_id}")

        out_path = Path(params['output_path']).resolve()
        if out_path.suffix == '':
            raise ValueError(
                f"output_path must include a file extension so the output "
                f"format can be selected: {out_path}"
            )
        if not out_path.parent.exists():
            raise ValueError(f"Output directory does not exist: {out_path.parent}")

        if self.allowed_save_dirs is not None:
            inside = any(
                str(out_path) == str(d)
                or str(out_path).startswith(str(d) + os.sep)
                for d in self.allowed_save_dirs
            )
            if not inside:
                raise ValueError(
                    f"Output '{out_path}' is outside allowed save "
                    f"directories: {[str(d) for d in self.allowed_save_dirs]}"
                )

        save_mesh_to_file(mesh, out_path, library=params.get('library'))

        return {
            'mesh_id': mesh_id,
            'path': str(out_path),
            'format': out_path.suffix[1:].lower(),
            'bytes': os.path.getsize(out_path),
            'n_vertices': mesh.n_vertices,
            'n_volume_cells': (
                mesh.n_volume_cells if mesh.has_volume_cells else 0
            ),
        }

    def __repr__(self) -> str:
        return f"<MeshIssueExaminerServer: {len(self._tools)} tools, {len(self._mesh_cache)} cached meshes>"


# ---------------------------------------------------------------------------
# FastMCP protocol layer (standard MCP SDK)
# ---------------------------------------------------------------------------


def create_mcp_server(
    config: Optional[OperationConfig] = None,
    auto_register_operations: bool = True,
) -> "FastMCP":
    """Create a standards-compliant MCP server using the official SDK.

    Returns a :class:`mcp.server.fastmcp.FastMCP` instance with all mesh
    examination tools registered via the canonical ``@mcp.tool()`` decorator
    pattern.  Use ``mcp.run()`` to start serving over stdio (default) or
    streamable-http.

    Args:
        config: Optional operation configuration forwarded to the
            business-logic layer.
        auto_register_operations: Whether to auto-register built-in
            inspection operations.

    Example::

        mcp = create_mcp_server()
        mcp.run()                       # stdio transport (default)
        mcp.run(transport="streamable-http")  # HTTP transport
    """
    from mcp.server.fastmcp import FastMCP

    backend = MeshIssueExaminerServer(
        config=config,
        auto_register_operations=auto_register_operations,
        enable_save_tool=True,
    )

    mcp = FastMCP("mesh-issue-examiner")

    # -- mesh_load --------------------------------------------------------

    @mcp.tool()
    async def mesh_load(
        file_path: str,
        mesh_id: str | None = None,
        library: str | None = None,
        material_array_name: str | None = None,
    ) -> str:
        """Load a mesh file for subsequent inspection operations.

        Use this tool when you need to import a mesh from a local file
        (STL, OBJ, PLY, VTK, NAS, and other formats supported by meshio).

        Args:
            file_path: Absolute or relative path to the mesh file.
            mesh_id: Optional custom identifier.  If omitted the server
                generates one from the filename.
            library: Preferred loading library.  One of "trimesh",
                "pyvista", "pymeshlab".  If omitted the server picks
                automatically.
            material_array_name: Optional name of the per-cell data array
                holding material IDs (e.g. "material", "gmsh:physical",
                "region_id").  When provided, material assignments are
                extracted so the multi-material operations
                (quality_3d.detect_material_interfaces,
                check_interface_quality, per_material_statistics) can run.
                The load response then includes a materials summary.

        Returns:
            JSON string with mesh_id, vertex/face counts, bounding box,
            and source format (plus a materials summary when
            material_array_name extracted data).

        Example:
            Load "model.stl" and then inspect it::

                result = mesh_load(file_path="model.stl")
                mesh_inspect(mesh_id=<mesh_id from result>)

            Load a two-material VTU with material IDs in the "material"
            cell-data array::

                result = mesh_load(file_path="model.vtu",
                                   material_array_name="material")
        """
        params: Dict[str, Any] = {"file_path": file_path}
        if mesh_id is not None:
            params["mesh_id"] = mesh_id
        if library is not None:
            params["library"] = library
        if material_array_name is not None:
            params["material_array_name"] = material_array_name
        result = await backend._handle_load(params)
        return json.dumps(result, indent=2)

    # -- mesh_inspect -----------------------------------------------------

    @mcp.tool()
    async def mesh_inspect(
        mesh_id: str,
        categories: list[str] | None = None,
        operations: list[str] | None = None,
    ) -> str:
        """Execute inspection operations on a previously loaded mesh.

        Provide ONE of ``categories`` or ``operations`` to run a subset;
        omit both to run every registered operation.

        Args:
            mesh_id: Identifier returned by mesh_load.
            categories: Run all operations in these categories.
                Each value must be one of: "topological", "geometric",
                "quality", "orientation".
            operations: Run these specific operation IDs.
                Discover IDs via the mesh://operations resource.

        Returns:
            JSON summary with severity counts, pass/fail flag, category
            breakdown, and processing time.  DOES NOT include individual
            issues or quality metrics -- use mesh_get_issues and
            mesh_get_quality_metrics to drill down.

        Errors:
            ValueError -- mesh_id not found in cache.
            ValueError -- unknown operation ID.
            ValueError -- both categories and operations provided
                (mutual exclusion).

        Example::

            # Run everything
            mesh_inspect(mesh_id="bracket")

            # Run just topological checks
            mesh_inspect(mesh_id="bracket", categories=["topological"])

            # Run one specific operation
            mesh_inspect(mesh_id="bracket",
                         operations=["topology.detect_nonmanifold_edges"])
        """
        params: Dict[str, Any] = {"mesh_id": mesh_id}
        if categories is not None:
            params["categories"] = categories
        if operations is not None:
            params["operations"] = operations
        result = await backend._handle_inspect(params)
        return json.dumps(result, indent=2)

    # -- mesh_get_issues --------------------------------------------------

    @mcp.tool()
    def mesh_get_issues(
        mesh_id: str,
        severity: str | None = None,
        category: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> str:
        """Retrieve detected issues from a completed inspection.

        Call after mesh_inspect.  Supports filtering and pagination to
        control response size.

        Args:
            mesh_id: Identifier from mesh_load.
            severity: Filter -- "critical" | "error" | "warning" | "info".
                Returns all severities when omitted.
            category: Filter -- "topological" | "geometric" | "quality"
                | "orientation".  Returns all categories when omitted.
            limit: Max issues to return (1-200, default 50).
            offset: Skip this many issues for pagination.

        Returns:
            JSON with total_count (matching filter), returned_count,
            offset, and the issue list.

        Errors:
            ValueError -- mesh_id not found or no inspection run yet.

        Example::

            # First page of critical issues
            mesh_get_issues(mesh_id="bracket", severity="critical")

            # Paginated geometric issues
            mesh_get_issues(mesh_id="bracket", category="geometric",
                            limit=20, offset=0)
        """
        params: Dict[str, Any] = {"mesh_id": mesh_id, "limit": limit, "offset": offset}
        if severity is not None:
            params["severity"] = severity
        if category is not None:
            params["category"] = category
        result = backend._handle_get_issues(params)
        return json.dumps(result, indent=2)

    # -- mesh_get_quality_metrics -----------------------------------------

    @mcp.tool()
    def mesh_get_quality_metrics(
        mesh_id: str,
        metric_name: str | None = None,
    ) -> str:
        """Retrieve quality metrics from a completed inspection.

        Call after mesh_inspect.  Returns per-metric statistics (min,
        max, mean, median, std, p5, p95) and violation counts.

        Args:
            mesh_id: Identifier from mesh_load.
            metric_name: Specific metric to retrieve (e.g. "aspect_ratio",
                "scaled_jacobian").  Returns all metrics when omitted.

        Returns:
            JSON with metric_count and array of metric objects.  Each
            contains metric_name, statistics, threshold, violation_count,
            and total_count.

        Errors:
            ValueError -- mesh_id not found or no inspection run yet.

        Example::

            mesh_get_quality_metrics(mesh_id="bracket")
            mesh_get_quality_metrics(mesh_id="bracket",
                                     metric_name="aspect_ratio")
        """
        params: Dict[str, Any] = {"mesh_id": mesh_id}
        if metric_name is not None:
            params["metric_name"] = metric_name
        result = backend._handle_get_quality_metrics(params)
        return json.dumps(result, indent=2)

    # -- mesh_save ---------------------------------------------------------

    @mcp.tool()
    async def mesh_save(
        mesh_id: str,
        output_path: str,
        library: str | None = None,
    ) -> str:
        """Save a previously loaded mesh to a file.

        The output format is driven by the output file's extension.
        NASTRAN decks (.nas/.bdf) are written by the native free-field
        writer (quadratic CHEXA/CPENTA midside nodes retained); other
        formats (.vtu/.vtk/.msh/.stl/...) are written via meshio/pyvista.

        Args:
            mesh_id: Identifier returned by mesh_load.
            output_path: Absolute path for the output file. The parent
                directory must exist; the extension selects the format.
            library: Writer override for non-NASTRAN formats -- "trimesh"
                | "pyvista" | "pymeshlab" | "meshio". Auto-detected from
                the extension when omitted.

        Returns:
            JSON with mesh_id, path, format, bytes, n_vertices, and
            n_volume_cells.

        Errors:
            ValueError -- mesh_id not found, missing file extension,
                missing output directory, or output_path outside the
                server's allowed save directories (when configured).

        Example::

            mesh_save(mesh_id="bracket", output_path="out/bracket.nas")
            mesh_save(mesh_id="bracket", output_path="out/bracket.vtu")
        """
        params: Dict[str, Any] = {
            "mesh_id": mesh_id,
            "output_path": output_path,
        }
        if library is not None:
            params["library"] = library
        result = await backend._handle_save(params)
        return json.dumps(result, indent=2)

    # -- mesh_clear_cache -------------------------------------------------

    @mcp.tool()
    def mesh_clear_cache(mesh_id: str | None = None) -> str:
        """Free cached mesh data and inspection results.

        Call to release memory or force a fresh reload.

        Args:
            mesh_id: Specific mesh to clear.  If omitted, clears all
                cached meshes and results.

        Returns:
            JSON string with status and list of cleared IDs.

        Example:
            Clear everything:  mesh_clear_cache()
            Clear one mesh:    mesh_clear_cache(mesh_id="my_mesh")
        """
        params: Dict[str, Any] = {}
        if mesh_id is not None:
            params["mesh_id"] = mesh_id
        result = backend._handle_clear_cache(params)
        return json.dumps(result, indent=2)

    # -- Resources --------------------------------------------------------

    @mcp.resource("mesh://operations")
    def operations_catalog() -> str:
        """Catalog of all available inspection operations.

        Returns JSON array of operation descriptors including operation_id,
        display_name, description, category, supported_libraries,
        recommended_library, complexity, and tags.
        """
        operations = registry.list_all()
        return json.dumps([op.to_dict() for op in operations], indent=2)

    @mcp.resource("mesh://{mesh_id}/info")
    def mesh_info(mesh_id: str) -> str:
        """Mesh metadata: vertex/face counts, bounding box, format,
        volume cell types.

        Available after mesh_load.
        """
        mesh = backend._mesh_cache.get(mesh_id)
        if mesh is None:
            return json.dumps({
                "error": f"Mesh not found: {mesh_id}",
                "hint": "Load a mesh first using mesh_load.",
            })
        return json.dumps(mesh.to_dict(), indent=2)

    @mcp.resource("mesh://{mesh_id}/summary")
    def inspection_summary(mesh_id: str) -> str:
        """Inspection summary: severity counts, pass/fail flag,
        category breakdown, processing time.

        Available after mesh_inspect.
        """
        result = backend._result_cache.get(mesh_id)
        if result is None:
            return json.dumps({
                "error": f"No inspection result for mesh: {mesh_id}",
                "hint": "Run mesh_inspect first.",
            })
        return json.dumps({
            "mesh_id": mesh_id,
            "passed": result.passed,
            "processing_time": result.processing_time,
            "summary": result.summary,
        }, indent=2)

    return mcp


def main():
    """Entry point for the mesh-examiner console command.

    Starts a standards-compliant MCP server over stdio transport using
    the official MCP Python SDK (FastMCP).
    """
    mcp = create_mcp_server()
    mcp.run()
