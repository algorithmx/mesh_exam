"""
Boundary edge detection operation.

Detects edges that are incident to exactly one face, indicating
open surfaces or holes in the mesh.
"""

from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict

from ...core.models import (
    MeshInput,
    MeshIssue,
    IssueLocation,
    IssueCategory,
    IssueSeverity,
    OperationConfig,
)
from ...utils.topology import build_edge_face_map
from ..base import BaseOperation, OperationMetadata
from ..operation_registry import registry
from ...utils.metrics import determine_severity


class BoundaryEdgeDetection(BaseOperation):
    """
    Detects boundary edges in a mesh.

    A boundary edge is an edge incident to exactly one face. Closed/watertight
    meshes have zero boundary edges. Open meshes, or meshes with holes, have
    boundary edges forming continuous loops.

    This detection is useful for:
    - Determining if a mesh is watertight (suitable for 3D printing)
    - Identifying holes that need repair
    - Validating mesh closure against ground truth datasets
    """

    @property
    def metadata(self) -> OperationMetadata:
        return OperationMetadata(
            operation_id='topology.detect_boundary_edges',
            display_name='Boundary Edge Detection',
            description='Identifies edges incident to exactly one face, '
                       'indicating open surfaces or holes in the mesh.',
            category=IssueCategory.TOPOLOGICAL,
            supported_libraries=['trimesh', 'pyvista', 'pymeshlab'],
            recommended_library='pyvista',
            requires_adjacency=True,
            complexity='O(n)',
            tags=['topology', 'boundary', 'open', 'watertight', 'holes']
        )

    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        """Execute boundary edge detection."""
        # Build edge-to-face adjacency
        edge_faces = build_edge_face_map(mesh.faces)

        if not edge_faces:
            return []

        # Find boundary edges (exactly 1 incident face)
        boundary_edges = []
        for edge, faces in edge_faces.items():
            if len(faces) == 1:
                boundary_edges.append((edge, faces[0]))

        if not boundary_edges:
            # Mesh is watertight
            return []

        # Group boundary edges into continuous loops
        boundary_loops = self._extract_boundary_loops(boundary_edges)

        # Create issue locations for each boundary edge
        locations = []
        for (v1, v2), face_idx in boundary_edges:
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
                    'incident_face': int(face_idx)
                }
            ))

        # Determine severity based on boundary extent
        total_edges = len(edge_faces)
        boundary_ratio = len(boundary_edges) / max(total_edges, 1)

        # Severity classification:
        # - INFO: Few boundary edges (<5%) - may be intentional
        # - WARNING: Moderate boundary edges (5-20%) - possible design intent
        # - ERROR: Significant boundary edges (>20%) - likely open mesh issues
        if boundary_ratio > 0.20:
            severity = IssueSeverity.ERROR
        elif boundary_ratio > 0.05:
            severity = IssueSeverity.WARNING
        else:
            severity = IssueSeverity.INFO

        # Build informative message
        if len(boundary_loops) == 1:
            loop_msg = "1 boundary loop"
        else:
            loop_msg = f"{len(boundary_loops)} boundary loops"

        issue = MeshIssue(
            operation_id=self.metadata.operation_id,
            category=IssueCategory.TOPOLOGICAL,
            severity=severity,
            message=f'Detected {len(boundary_edges)} boundary edges '
                   f'({boundary_ratio * 100:.2f}% of edges) forming {loop_msg}. '
                   f'Mesh is not watertight.',
            locations=locations,
            metrics={
                'boundary_edge_count': len(boundary_edges),
                'boundary_loop_count': len(boundary_loops),
                'is_watertight': False,
                'boundary_edges_ratio': boundary_ratio,
                'total_edges': total_edges
            },
            metadata={
                'edge_face_map_size': len(edge_faces),
                'manifold_edges': total_edges - len(boundary_edges),
                'boundary_loops': [len(loop) for loop in boundary_loops]
            }
        )

        return [issue]

    def _extract_boundary_loops(
        self,
        boundary_edges: List[Tuple[Tuple[int, int], int]]
    ) -> List[List[int]]:
        """
        Group boundary edges into continuous loops.

        Each boundary edge connects two vertices. By traversing connected
        edges, we identify closed boundary loops (cycles).

        Args:
            boundary_edges: List of ((v1, v2), face_idx) tuples

        Returns:
            List of vertex sequences forming boundary loops
        """
        if not boundary_edges:
            return []

        # Build adjacency: vertex -> [(neighbor_vertex, edge), ...]
        adjacency = defaultdict(list)
        edge_set = set()

        for (v1, v2), _ in boundary_edges:
            adjacency[v1].append(v2)
            adjacency[v2].append(v1)
            edge_set.add((min(v1, v2), max(v1, v2)))

        loops = []
        visited_edges = set()

        for start_edge in edge_set:
            if start_edge in visited_edges:
                continue

            # Start a new loop
            start_vertex = start_edge[0]
            current_vertex = start_edge[1]
            loop = [start_vertex, current_vertex]
            visited_edges.add(start_edge)

            # Follow the loop until we return to start
            while current_vertex != start_vertex:
                neighbors = adjacency[current_vertex]
                next_vertex = None

                for neighbor in neighbors:
                    edge = (min(current_vertex, neighbor), max(current_vertex, neighbor))
                    if edge not in visited_edges:
                        next_vertex = neighbor
                        visited_edges.add(edge)
                        break

                if next_vertex is None:
                    # Dead end - incomplete loop (shouldn't happen in valid mesh)
                    break

                loop.append(next_vertex)
                current_vertex = next_vertex

            # Close the loop by removing duplicate start/end if present
            if len(loop) > 1 and loop[0] == loop[-1]:
                loop = loop[:-1]

            if len(loop) >= 3:  # Valid loop needs at least 3 vertices
                loops.append(loop)

        return loops


# Register operation
registry.register(BoundaryEdgeDetection())
