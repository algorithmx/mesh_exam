"""
Isolated vertex detection operation.

Detects vertices that are not referenced by any face in the mesh. These
vertices waste memory and may indicate incomplete mesh construction or
data corruption.

Algorithm — Set-difference:
    Collect the set of vertex indices used in the face array. Any vertex
    index present in the vertex array but absent from the face array is
    isolated.

    Complexity: O(V + F * k) where k = vertices per face.
"""

from typing import List

import numpy as np

from ...core.models import (
    MeshInput,
    MeshIssue,
    IssueLocation,
    IssueCategory,
    IssueSeverity,
)
from ..base import BaseOperation, OperationMetadata
from ..operation_registry import registry


class IsolatedVertexDetection(BaseOperation):
    """
    Detects vertices not referenced by any face.

    Isolated vertices consume memory without contributing to the mesh
    geometry and may indicate incomplete mesh construction.
    """

    @property
    def metadata(self) -> OperationMetadata:
        return OperationMetadata(
            operation_id='topology.detect_isolated_vertices',
            display_name='Isolated Vertex Detection',
            description='Identifies vertices not referenced by any face, '
                       'indicating unused geometry data.',
            category=IssueCategory.TOPOLOGICAL,
            supported_libraries=['trimesh', 'pyvista', 'pymeshlab'],
            recommended_library='pyvista',
            requires_adjacency=False,
            complexity='O(V + F)',
            tags=['topology', 'vertices', 'isolated', 'unreferenced']
        )

    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        """Execute isolated vertex detection."""
        faces = mesh.faces
        n_verts = len(mesh.vertices)

        if n_verts == 0:
            return []

        if n_faces := len(faces) == 0:
            # All vertices are isolated when there are no faces
            isolated = set(range(n_verts))
        else:
            # Set-difference: all vertices minus used vertices
            used = set(faces.ravel().tolist())
            isolated = set(range(n_verts)) - used

        if not isolated:
            return []

        # Build locations for each isolated vertex
        vertices = mesh.vertices
        locations = []
        for vi in isolated:
            coords = tuple(vertices[vi]) if vi < len(vertices) else None
            locations.append(IssueLocation(
                element_type='vertex',
                element_index=vi,
                coordinates=coords,
            ))

        violation_rate = len(isolated) / n_verts

        # Severity based on extent
        if violation_rate >= 0.05:
            severity = IssueSeverity.WARNING
        else:
            severity = IssueSeverity.INFO

        issue = MeshIssue(
            operation_id=self.metadata.operation_id,
            category=IssueCategory.TOPOLOGICAL,
            severity=severity,
            message=f'Detected {len(isolated)} isolated vertices '
                   f'({violation_rate * 100:.2f}% of {n_verts} total). '
                   f'These vertices are not referenced by any face.',
            locations=locations,
            metrics={
                'isolated_vertex_count': len(isolated),
                'violation_rate': violation_rate,
                'total_vertices': n_verts,
            },
            metadata={
                'isolated_indices': sorted(isolated),
            }
        )

        return [issue]


# Register operation
registry.register(IsolatedVertexDetection())
