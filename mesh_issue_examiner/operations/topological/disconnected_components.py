"""
Disconnected component detection operation.

Detects separate connected components in a mesh by analyzing vertex
connectivity through shared edges. For a mesh expected to be a single
continuous surface, multiple disconnected components may indicate a
modeling error or misassembly.

Algorithm — Union-Find (Disjoint Set Union):
    For each face, union all vertex pairs that form edges. After processing
    all faces, count distinct root sets. Components with more than one
    member indicate disconnected regions.

    Complexity: O(V + F * k) where k = vertices per face (near-linear
    due to path compression).

Definition:
    "A mesh is connected if every pair of vertices can be joined by a
    path of edges." — standard mesh connectivity definition.

Repair reference:
    Many mesh repair tools (e.g., MeshLab, Blender) can separate or merge
    disconnected components as a post-processing step.
"""

from typing import List, Dict, Tuple

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
from ...utils.metrics import determine_severity


class DisconnectedComponentDetection(BaseOperation):
    """
    Detects disconnected components in a mesh using Union-Find.

    Two vertices are in the same component if connected by a path of edges.
    A mesh with more than one component is considered disconnected.
    """

    @property
    def metadata(self) -> OperationMetadata:
        return OperationMetadata(
            operation_id='topology.detect_disconnected_components',
            display_name='Disconnected Component Detection',
            description='Identifies separate connected components in the mesh '
                       'by analyzing vertex connectivity through shared edges.',
            category=IssueCategory.TOPOLOGICAL,
            supported_libraries=['trimesh', 'pyvista', 'pymeshlab'],
            recommended_library='pyvista',
            requires_adjacency=True,
            complexity='O(V + F)',
            tags=['topology', 'connectivity', 'components', 'disconnected']
        )

    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        """Execute disconnected component detection."""
        faces = mesh.faces
        n_faces = len(faces)
        n_verts = len(mesh.vertices)

        if n_faces == 0 or n_verts == 0:
            return []

        # Union-Find with path compression and union by rank
        parent = np.arange(n_verts, dtype=np.intp)
        rank = np.zeros(n_verts, dtype=np.intp)

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression (halving)
                x = parent[x]
            return x

        def union(a: int, b: int):
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1

        # Union all edge-connected vertices
        if faces.ndim == 2:
            # Vectorized path for 2D face arrays
            k = faces.shape[1]
            for i in range(k):
                v1 = faces[:, i]
                v2 = faces[:, (i + 1) % k]
                for a, b in zip(v1, v2):
                    union(int(a), int(b))
        else:
            # General polygon path
            for face in faces:
                n = len(face)
                for i in range(n):
                    union(int(face[i]), int(face[(i + 1) % n]))

        # Volume meshes: each cell is internally connected, so union its
        # vertices.  The extracted surface omits interior nodes (they
        # would otherwise register as singleton components), and cell
        # adjacency glues neighbouring surface patches together.
        if mesh.has_volume_cells:
            for conn in mesh.volume_cells.values():
                for row in np.asarray(conn):
                    root = int(row[0])
                    for v in row[1:]:
                        union(root, int(v))

        # Count components and group vertices
        roots = np.array([find(v) for v in range(n_verts)])
        unique_roots, counts = np.unique(roots, return_counts=True)
        n_components = len(unique_roots)

        if n_components <= 1:
            return []

        # Group faces by component
        # Map each vertex to its component index
        root_to_comp = {int(r): i for i, r in enumerate(unique_roots)}
        vert_comp = np.array([root_to_comp[find(v)] for v in range(n_verts)])

        # Count faces per component
        comp_face_counts = np.zeros(n_components, dtype=int)
        for face in faces:
            comp_idx = vert_comp[int(face[0])]
            comp_face_counts[comp_idx] += 1

        comp_vert_counts = counts
        largest_idx = int(np.argmax(comp_vert_counts))

        # Build locations — one per non-largest component
        locations = []
        for ci in range(n_components):
            if ci == largest_idx:
                continue
            # Pick first vertex in this component for coordinates
            comp_verts = np.where(vert_comp == ci)[0]
            sample_vi = int(comp_verts[0])
            coords = tuple(mesh.vertices[sample_vi])
            locations.append(IssueLocation(
                element_type='component',
                element_index=ci,
                coordinates=coords,
                adjacent_elements={
                    'vertex_count': int(comp_vert_counts[ci]),
                    'face_count': int(comp_face_counts[ci]),
                }
            ))

        severity = IssueSeverity.WARNING  # May be intentional

        issue = MeshIssue(
            operation_id=self.metadata.operation_id,
            category=IssueCategory.TOPOLOGICAL,
            severity=severity,
            message=f'Detected {n_components} disconnected components '
                   f'(largest has {int(comp_vert_counts[largest_idx])} vertices '
                   f'and {int(comp_face_counts[largest_idx])} faces). '
                   f'Mesh is not a single connected surface.',
            locations=locations,
            metrics={
                'component_count': n_components,
                'largest_component_vertices': int(comp_vert_counts[largest_idx]),
                'largest_component_faces': int(comp_face_counts[largest_idx]),
                'is_connected': False,
                'total_vertices': n_verts,
                'total_faces': n_faces,
            },
            metadata={
                'component_vertex_counts': [int(c) for c in comp_vert_counts],
                'component_face_counts': [int(c) for c in comp_face_counts],
            }
        )

        return [issue]


# Register operation
registry.register(DisconnectedComponentDetection())
