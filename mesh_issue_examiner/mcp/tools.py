"""
MCP tool definitions and helpers.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ToolDefinition:
    """Definition of an MCP tool."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP tool format."""
        result = {
            'name': self.name,
            'description': self.description,
            'input_schema': self.input_schema
        }
        if self.output_schema:
            result['output_schema'] = self.output_schema
        return result


def get_tool_definitions() -> List[ToolDefinition]:
    """
    Get all MCP tool definitions.

    Returns:
        List of ToolDefinition objects
    """
    from mcp.server import MeshIssueExaminerServer

    server = MeshIssueExaminerServer()

    return [
        ToolDefinition(
            name=tool.name,
            description=tool.description,
            input_schema=tool.input_schema
        )
        for tool in server._tools.values()
    ]


# Common schema components
SCHEMA_COMPONENTS = {
    'mesh_id': {
        'type': 'string',
        'description': 'Mesh identifier from load operation'
    },
    'file_path': {
        'type': 'string',
        'description': 'Path to mesh file'
    },
    'operation_id': {
        'type': 'string',
        'description': 'Operation identifier (e.g., "topology.detect_nonmanifold_edges")'
    },
    'severity': {
        'type': 'string',
        'enum': ['info', 'warning', 'error', 'critical'],
        'description': 'Issue severity level'
    },
    'category': {
        'type': 'string',
        'enum': ['topological', 'geometric', 'quality', 'orientation'],
        'description': 'Issue category'
    }
}


def create_input_schema(
    required: List[str],
    properties: Dict[str, Any],
    description: str = ""
) -> Dict[str, Any]:
    """
    Create a JSON schema for tool input.

    Args:
        required: List of required property names
        properties: Property definitions
        description: Schema description

    Returns:
        JSON schema dictionary
    """
    return {
        'type': 'object',
        'properties': properties,
        'required': required,
        'description': description
    }


def create_array_schema(
    item_type: str,
    min_items: Optional[int] = None,
    max_items: Optional[int] = None
) -> Dict[str, Any]:
    """
    Create a JSON array schema.

    Args:
        item_type: Type of array items
        min_items: Minimum number of items
        max_items: Maximum number of items

    Returns:
        Array schema dictionary
    """
    schema = {
        'type': 'array',
        'items': {'type': item_type}
    }
    if min_items is not None:
        schema['min_items'] = min_items
    if max_items is not None:
        schema['max_items'] = max_items
    return schema
