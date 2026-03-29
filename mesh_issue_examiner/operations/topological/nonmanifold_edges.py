"""
Non-manifold edge detection operation.

Detects edges shared by more than two faces, which violates
the 2-manifold property required for valid solid models.
"""

from typing import List, Optional
import numpy as np

from core.models import (
    MeshInput,
    MeshIssue,
    IssueLocation,
    IssueCategory,
    IssueSeverity,
    OperationConfig,
)
from operations.base import BaseOperation, OperationMetadata
from operations.operation_registry import registry


class NonManifoldEdgeDetection(BaseOperation):
    """
    Detects edges shared by more than two faces.

    A valid 2-manifold mesh requires that every edge is incident
    to exactly two faces. Edges shared by more than two faces
    create topological ambiguity and cause issues in downstream
    processing like boolean operations and volume computation.
    """

    @property
    def metadata(self) -> OperationMetadata:
        return OperationMetadata(
            operation_id='topology.detect_nonmanifold_edges',
            display_name='Non-Manifold Edge Detection',
            description='Identifies edges shared by more than two faces, '
                       'violating the 2-manifold property required for solid models.',
            category=IssueCategory.TOPOLOGICAL,
            supported_libraries=['trimesh', 'pyvista', 'pymeshlab'],
            recommended_library='pyvista',
            requires_adjacency=True,
            complexity='O(n)',
            tags=['topology', 'manifold', 'connectivity', 'basic']
        )

    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        """Execute non-manifold edge detection."""
        # Build edge-to-face adjacency
        edge_faces = self._build_edge_face_map(mesh)

        # Find edges with != 2 incident faces
        nonmanifold_edges = []
        for edge, faces in edge_faces.items():
            if len(faces) != 2:
                nonmanifold_edges.append((edge, len(faces), faces))

        if not nonmanifold_edges:
            return []

        # Create issue locations
        locations = []
        for (v1, v2), face_count, incident_faces in nonmanifold_edges:
            if v1 < len(mesh.vertices):
                coords = tuple(mesh.vertices[v1])
            else:
                coords = None

            locations.append(IssueLocation(
                element_type='edge',
                element_index=-1,
                coordinates=coords,
                adjacent_elements={
                    'vertices': [int(v1), int(v2)],
                    'face_count': face_count,
                    'incident_faces': [int(f) for f in incident_faces]
                }
            ))

        # Determine severity based on extent
        severity = IssueSeverity.ERROR
        violation_rate = len(nonmanifold_edges) / max(len(mesh.faces), 1)
        if violation_rate > 0.1:
            severity = IssueSeverity.CRITICAL
        elif violation_rate > 0.01:
            severity = IssueSeverity.ERROR
        elif violation_rate > 0.001:
            severity = IssueSeverity.WARNING

        issue = MeshIssue(
            operation_id=self.metadata.operation_id,
            category=IssueCategory.TOPOLOGICAL,
            severity=severity,
            message=f'Detected {len(nonmanifold_edges)} non-manifold edges '
                   f'({violation_rate * 100:.2f}% of faces). '
                   f'These edges are shared by more than two faces.',
            locations=locations,
            metrics={
                'count': len(nonmanifold_edges),
                'violation_rate': violation_rate,
                'total_faces': len(mesh.faces)
            },
            metadata={
                'edge_face_map_size': len(edge_faces),
                'manifold_edges': len(edge_faces) - len(nonmanifold_edges)
            }
        )

        return [issue]

    def _build_edge_face_map(self, mesh: MeshInput) -> dict:
        """Build mapping from edges to incident faces."""
        edge_faces = {}

        for face_idx, face in enumerate(mesh.faces):
            n = len(face)
            for i in range(n):
                v1, v2 = int(face[i]), int(face[(i + 1) % n])
                edge = (min(v1, v2), max(v1, v2))

                if edge not in edge_faces:
                    edge_faces[edge] = []
                edge_faces[edge].append(face_idx)

        return edge_faces


# Register operation
registry.register(NonManifoldEdgeDetection())
