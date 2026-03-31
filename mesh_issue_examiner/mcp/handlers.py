"""
Request handlers for MCP server.
"""

from typing import Dict, Any, Callable, Awaitable


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
