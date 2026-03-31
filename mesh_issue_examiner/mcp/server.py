"""
MCP server implementation for mesh issue examination.

This module provides the main server class that handles MCP protocol
communications and routes requests to appropriate operations.
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..core.models import (
    MeshInput,
    InspectionResult,
    OperationConfig,
    IssueCategory,
)
from ..operations.operation_registry import registry
from ..utils.mesh_loader import load_mesh_from_file

# Ensure built-in operations are imported so they self-register at module level
from ..operations.topological import nonmanifold_edges as _  # noqa: F401
from ..operations.quality import cell_quality as _  # noqa: F401
from ..operations.volumetric import volume_cell_quality as _  # noqa: F401
from ..operations.volumetric import geometric_validity as _  # noqa: F401
from ..operations.volumetric import orientation_consistency as _  # noqa: F401
from ..operations.volumetric import nonmanifold_faces as _  # noqa: F401


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


class MeshIssueExaminerServer:
    """
    MCP server for mesh issue examination.

    Provides tools for:
    - Loading meshes from files
    - Running individual or composed inspection operations
    - Retrieving inspection results
    - Batch inspection workflows

    Usage:
        server = MeshIssueExaminerServer()

        # Load a mesh
        load_result = await server.handle_request(MCPRequest(
            method='mesh_load',
            params={'file_path': 'example.stl'}
        ))

        # Inspect the mesh
        inspect_result = await server.handle_request(MCPRequest(
            method='mesh_inspect',
            params={'mesh_id': 'example', 'operations': ['all']}
        ))
    """

    def __init__(
        self,
        config: Optional[OperationConfig] = None,
        auto_register_operations: bool = True
    ):
        """
        Initialize the MCP server.

        Args:
            config: Optional configuration for operations
            auto_register_operations: Whether to auto-register built-in operations
        """
        self.config = config or OperationConfig()
        self._mesh_cache: Dict[str, MeshInput] = {}
        self._result_cache: Dict[str, InspectionResult] = {}
        self._tools: Dict[str, MCPTool] = {}

        if auto_register_operations:
            self._register_builtin_tools()

    def _register_builtin_tools(self):
        """Register built-in MCP tools."""
        # Mesh load tool
        self._tools['mesh_load'] = MCPTool(
            name='mesh_load',
            description='Load a mesh file and return mesh ID for subsequent operations',
            input_schema={
                'type': 'object',
                'properties': {
                    'file_path': {
                        'type': 'string',
                        'description': 'Path to mesh file (STL, OBJ, PLY, etc.)'
                    },
                    'mesh_id': {
                        'type': 'string',
                        'description': 'Optional custom mesh ID'
                    },
                    'library': {
                        'type': 'string',
                        'enum': ['trimesh', 'pyvista', 'pymeshlab'],
                        'description': 'Preferred loading library'
                    }
                },
                'required': ['file_path']
            }
        )

        # Mesh inspect tool
        self._tools['mesh_inspect'] = MCPTool(
            name='mesh_inspect',
            description='Run mesh inspection operations',
            input_schema={
                'type': 'object',
                'properties': {
                    'mesh_id': {
                        'type': 'string',
                        'description': 'Mesh ID from load operation'
                    },
                    'operations': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': 'List of operation IDs, or ["all"] for all operations'
                    },
                    'categories': {
                        'type': 'array',
                        'items': {
                            'type': 'string',
                            'enum': ['topological', 'geometric', 'quality', 'orientation']
                        },
                        'description': 'Filter by category'
                    }
                },
                'required': ['mesh_id']
            }
        )

        # Get cached result
        self._tools['mesh_get_result'] = MCPTool(
            name='mesh_get_result',
            description='Get cached inspection result',
            input_schema={
                'type': 'object',
                'properties': {
                    'mesh_id': {
                        'type': 'string',
                        'description': 'Mesh ID'
                    },
                    'include_details': {
                        'type': 'boolean',
                        'description': 'Include detailed issue information'
                    }
                },
                'required': ['mesh_id']
            }
        )

        # Clear cache
        self._tools['mesh_clear_cache'] = MCPTool(
            name='mesh_clear_cache',
            description='Clear mesh and result cache',
            input_schema={
                'type': 'object',
                'properties': {
                    'mesh_id': {
                        'type': 'string',
                        'description': 'Optional specific mesh ID to clear'
                    }
                }
            }
        )

        # List operations
        self._tools['list_operations'] = MCPTool(
            name='list_operations',
            description='List all available inspection operations',
            input_schema={
                'type': 'object',
                'properties': {
                    'category': {
                        'type': 'string',
                        'enum': ['topological', 'geometric', 'quality', 'orientation'],
                        'description': 'Filter by category'
                    },
                    'library': {
                        'type': 'string',
                        'description': 'Filter by supported library'
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

            elif method == 'mesh_get_result':
                result = self._handle_get_result(request.params)

            elif method == 'mesh_clear_cache':
                result = self._handle_clear_cache(request.params)

            elif method == 'list_operations':
                result = self._handle_list_operations(request.params)

            else:
                result = None
                error = {'code': 'unknown_method', 'message': f"Unknown method: {method}"}

            return MCPResponse(
                result=result,
                error=error if 'error' in locals() else None,
                request_id=request.request_id
            )

        except Exception as e:
            return MCPResponse(
                result=None,
                error={
                    'code': 'internal_error',
                    'message': str(e),
                    'type': type(e).__name__
                },
                request_id=request.request_id
            )

    async def _handle_load(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle mesh load request."""
        file_path = params['file_path']
        mesh_id = params.get('mesh_id')
        library = params.get('library')

        mesh_input = load_mesh_from_file(
            file_path,
            mesh_id=mesh_id,
            library=library
        )

        self._mesh_cache[mesh_input.mesh_id] = mesh_input

        return {
            'mesh_id': mesh_input.mesh_id,
            'n_vertices': mesh_input.n_vertices,
            'n_faces': mesh_input.n_faces,
            'bounding_box': {
                'min': mesh_input.bounding_box[0].tolist(),
                'max': mesh_input.bounding_box[1].tolist()
            },
            'format': mesh_input.source_format
        }

    async def _handle_inspect(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle mesh inspection request."""
        mesh_id = params['mesh_id']
        operations = params.get('operations', ['all'])
        categories = params.get('categories')

        mesh = self._mesh_cache.get(mesh_id)
        if mesh is None:
            raise ValueError(f"Mesh not found: {mesh_id}")

        # Determine which operations to run
        if operations == ['all'] or operations == 'all':
            op_ids = None  # Will use all registered
        else:
            op_ids = operations

        # Create and execute workflow
        start_time = time.perf_counter()

        if op_ids is None:
            # Run all operations
            workflow = registry.create_full_workflow(config=self.config)
        else:
            workflow = registry.create_workflow(op_ids, config=self.config)

        result = workflow(mesh)
        result.processing_time = time.perf_counter() - start_time

        # Cache result
        self._result_cache[mesh_id] = result

        return result.to_dict()

    def _handle_get_result(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get cached result request."""
        mesh_id = params['mesh_id']
        include_details = params.get('include_details', False)

        result = self._result_cache.get(mesh_id)
        if result is None:
            raise ValueError(f"Result not found for mesh: {mesh_id}")

        return result.to_dict()

    def _handle_clear_cache(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle clear cache request."""
        mesh_id = params.get('mesh_id')

        if mesh_id:
            self._mesh_cache.pop(mesh_id, None)
            self._result_cache.pop(mesh_id, None)
            cleared = [mesh_id]
        else:
            self._mesh_cache.clear()
            self._result_cache.clear()
            cleared = list(self._mesh_cache.keys()) if hasattr(self, '_mesh_cache') else []

        return {'status': 'cleared', 'cleared_ids': cleared}

    def _handle_list_operations(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle list operations request."""
        category = params.get('category')
        library = params.get('library')

        if category:
            cat_enum = IssueCategory(category)
            operations = registry.list_by_category(cat_enum)
        elif library:
            operations = registry.list_by_library(library)
        else:
            operations = registry.list_all()

        return [op.to_dict() for op in operations]

    def __repr__(self) -> str:
        return f"<MeshIssueExaminerServer: {len(self._tools)} tools, {len(self._mesh_cache)} cached meshes>"


def main():
    """Entry point for the mesh-examiner console command."""
    server = MeshIssueExaminerServer()
    print(f"Mesh Issue Examiner Server initialized: {server}")
    return server
