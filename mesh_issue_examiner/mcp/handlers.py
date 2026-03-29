"""
Request handlers for MCP server.
"""

from typing import Dict, Any, Callable, Awaitable
import asyncio

from core.models import MeshInput, InspectionResult


# Type alias for handler functions
Handler = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]


class RequestHandler:
    """
    Request handler registry for MCP server.

    Maps request methods to handler functions.
    """

    def __init__(self):
        self._handlers: Dict[str, Handler] = {}
        self._setup_default_handlers()

    def _setup_default_handlers(self):
        """Set up default request handlers."""
        pass

    def register(self, method: str, handler: Handler):
        """
        Register a request handler.

        Args:
            method: HTTP/WebSocket method name
            handler: Async function to handle requests
        """
        self._handlers[method] = handler

    def get_handler(self, method: str) -> Handler:
        """
        Get handler for method.

        Args:
            method: Method name

        Returns:
            Handler function

        Raises:
            KeyError: If no handler registered for method
        """
        return self._handlers[method]

    def has_handler(self, method: str) -> bool:
        """Check if handler exists for method."""
        return method in self._handlers


class AsyncRequestHandler(RequestHandler):
    """
    Async request handler with mesh context.

    Maintains mesh cache and result cache across requests.
    """

    def __init__(self, config=None):
        super().__init__()
        self._mesh_cache: Dict[str, MeshInput] = {}
        self._result_cache: Dict[str, InspectionResult] = {}
        self._config = config

    def _setup_default_handlers(self):
        """Set up default async handlers."""
        self.register('mesh_load', self._handle_mesh_load)
        self.register('mesh_inspect', self._handle_mesh_inspect)
        self.register('mesh_get_result', self._handle_mesh_get_result)
        self.register('mesh_clear_cache', self._handle_mesh_clear_cache)

    async def _handle_mesh_load(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle mesh load request."""
        from utils.mesh_loader import load_mesh_from_file

        file_path = params['file_path']
        mesh_id = params.get('mesh_id')

        mesh = load_mesh_from_file(file_path, mesh_id=mesh_id)
        self._mesh_cache[mesh.mesh_id] = mesh

        return {
            'mesh_id': mesh.mesh_id,
            'n_vertices': mesh.n_vertices,
            'n_faces': mesh.n_faces,
            'status': 'loaded'
        }

    async def _handle_mesh_inspect(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle mesh inspection request."""
        from operations.operation_registry import registry

        mesh_id = params['mesh_id']
        operations = params.get('operations', ['all'])

        mesh = self._mesh_cache.get(mesh_id)
        if mesh is None:
            return {'error': f'Mesh not found: {mesh_id}'}

        if operations == ['all'] or operations == 'all':
            workflow = registry.create_full_workflow()
        else:
            workflow = registry.create_workflow(operations)

        result = workflow(mesh)
        self._result_cache[mesh_id] = result

        return result.to_dict()

    async def _handle_mesh_get_result(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get cached result request."""
        mesh_id = params['mesh_id']
        result = self._result_cache.get(mesh_id)

        if result is None:
            return {'error': f'Result not found: {mesh_id}'}

        return result.to_dict()

    async def _handle_mesh_clear_cache(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle clear cache request."""
        mesh_id = params.get('mesh_id')

        if mesh_id:
            self._mesh_cache.pop(mesh_id, None)
            self._result_cache.pop(mesh_id, None)
        else:
            self._mesh_cache.clear()
            self._result_cache.clear()

        return {'status': 'cleared'}
