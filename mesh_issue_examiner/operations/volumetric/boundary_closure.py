"""
Boundary closure (watertightness) check for volumetric (3D) meshes.

Extracts the boundary faces of a volume mesh (faces incident to exactly
one cell) and verifies that this boundary surface is itself closed: every
edge of the boundary complex is shared by exactly two boundary faces.

This complements ``topology_3d.detect_nonmanifold_faces`` (faces shared by
more than two cells). The classic failure caught here is a non-conforming
interface -- e.g. two hex blocks touching with mismatched face sizes
(a T-junction) -- where the blocks "leak" into each other without sharing
topological faces, leaving free edges on the extracted boundary.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from ...core.models import (
    MeshInput,
    MeshIssue,
    IssueLocation,
    IssueCategory,
    IssueSeverity,
    OperationConfig,
)
from ..base import BaseOperation, OperationMetadata
from ...utils.metrics import determine_severity
from ..operation_registry import registry
from ._face_tables import build_face_to_cell_map


class BoundaryClosureCheck(BaseOperation):
    """
    Checks whether a volumetric mesh bounds a closed (watertight) solid.

    Algorithm:
    1. Build the face-to-cells adjacency map (shared with non-manifold
       face detection).
    2. Boundary faces = faces incident to exactly one cell.
    3. Count edge incidences across boundary faces (undirected edges).
       A closed 2-manifold boundary has every edge shared by exactly two
       boundary faces; edges shared by one (free edges) or more than two
       (non-manifold boundary edges) indicate an open or defective
       boundary -- the volume does not enclose a well-defined solid.

    Returns no issue when the boundary is closed. A single element, a
    filled grid, or a grid with an interior cavity all have closed
    boundaries and pass this check.
    """

    # Maximum number of IssueLocation objects per issue (avoids huge reports)
    _MAX_LOCATIONS = 100

    def __init__(self, config: Optional[OperationConfig] = None):
        super().__init__(config)

    @property
    def metadata(self) -> OperationMetadata:
        return OperationMetadata(
            operation_id='topology_3d.check_boundary_closure',
            display_name='Boundary Closure Check (3D)',
            description='Verifies that the boundary of a volumetric mesh is '
                       'closed (watertight): every boundary edge is shared by '
                       'exactly two boundary faces. Catches non-conforming '
                       'block interfaces and leaking volumes.',
            category=IssueCategory.TOPOLOGICAL,
            supported_libraries=['pyvista'],
            recommended_library='pyvista',
            requires_adjacency=True,
            complexity='O(n)',
            tags=['topology', 'watertight', 'boundary', 'closure',
                  'volumetric', '3d']
        )

    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        """Execute boundary closure check on a volumetric mesh.

        Args:
            mesh: Input mesh with volumetric cells.

        Returns:
            List with a single MeshIssue when the boundary has free or
            non-manifold edges; empty list when closed (or no volume cells).
        """
        if not mesh.has_volume_cells:
            return []

        face_to_cells = build_face_to_cell_map(mesh)
        if not face_to_cells:
            return []

        boundary_faces = [
            face_verts
            for entries in face_to_cells.values()
            if len(entries) == 1
            for _, face_verts in entries
        ]

        # Count undirected edge incidences across boundary faces.
        edge_count: Dict[Tuple[int, int], int] = {}
        for face_verts in boundary_faces:
            n = len(face_verts)
            for i in range(n):
                a, b = face_verts[i], face_verts[(i + 1) % n]
                edge = (a, b) if a < b else (b, a)
                edge_count[edge] = edge_count.get(edge, 0) + 1

        free_edges = [e for e, c in edge_count.items() if c == 1]
        nonmanifold_edges = [e for e, c in edge_count.items() if c > 2]

        if not free_edges and not nonmanifold_edges:
            return []  # Watertight: closed 2-manifold boundary.

        total_boundary_edges = len(edge_count)
        defective = len(free_edges) + len(nonmanifold_edges)
        rate = defective / max(total_boundary_edges, 1)
        severity = determine_severity(rate)
        if severity == IssueSeverity.INFO:
            # Any opening in the boundary invalidates watertightness.
            severity = IssueSeverity.WARNING

        locations = []
        vertices = mesh.vertices
        for edge in (free_edges + nonmanifold_edges)[:self._MAX_LOCATIONS]:
            centroid = None
            if all(0 <= v < mesh.n_vertices for v in edge):
                coords = vertices[list(edge)]
                centroid = tuple(float(x) for x in np.mean(coords, axis=0))
            locations.append(IssueLocation(
                element_type='edge',
                element_index=-1,
                coordinates=centroid,
                adjacent_elements={
                    'edge_vertices': list(edge),
                    'incidence_count': edge_count[edge],
                }
            ))

        message = (
            f'Volume boundary is not closed: {len(free_edges)} free edge(s) '
            f'and {len(nonmanifold_edges)} non-manifold boundary edge(s) '
            f'({rate * 100:.2f}% of {total_boundary_edges} boundary edges). '
            f'The mesh does not bound a watertight solid.'
        )

        issue = MeshIssue(
            operation_id=self.metadata.operation_id,
            category=IssueCategory.TOPOLOGICAL,
            severity=severity,
            message=message,
            locations=locations,
            metrics={
                'is_watertight': False,
                'boundary_face_count': len(boundary_faces),
                'boundary_edge_count': total_boundary_edges,
                'free_edge_count': len(free_edges),
                'nonmanifold_boundary_edge_count': len(nonmanifold_edges),
                'violation_rate': rate,
            },
            metadata={
                'total_unique_faces': len(face_to_cells),
            },
        )
        return [issue]


# Register operation
registry.register(BoundaryClosureCheck())
