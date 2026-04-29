"""
Non-manifold vertex detection operation.

Detects vertices whose incident faces form more than one disconnected group
(one-ring connected components), which violates the 2-manifold property
required for valid solid models.

Algorithm — one-ring connected component analysis:
    For each vertex v, collect all faces incident to v. Build a graph where
    two incident faces are connected if they share an edge containing v.
    If this graph has more than one connected component, v is non-manifold.

    Ref: Jiao & He, "AHF: Array-based Half-Facet Data Structure for
    Mixed-Dimensional and Non-manifold Meshes", 2014:
    "v is a non-manifold vertex if the [one-ring face adjacency] graph has
    more than one connected component."
    https://www.ams.sunysb.edu/~jiao/papers/AHF_EwC.pdf

Definition:
    "A mesh is vertex-manifold if faces with a common vertex form an open
    or closed fan." — MATLAB isVertexManifold
    https://www.mathworks.com/help/lidar/ref/surfacemesh.isvertexmanifold.html

Repair reference:
    CGAL resolves non-manifold vertices by duplicating the vertex per
    connected component (see PMPPolygonSoupOrientationVisitor).
    https://doc.cgal.org/latest/Polygon_mesh_processing/classPMPPolygonSoupOrientationVisitor.html

Motivation:
    22.4% of real-world meshes in the Thingi10K dataset have non-manifold
    vertices (vertex_manifold: 77.6% True).
"""

from collections import defaultdict, deque
from typing import List, Tuple, Dict, Set

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
from ..operation_registry import registry
from ...utils.metrics import determine_severity


class NonManifoldVertexDetection(BaseOperation):
    """
    Detects non-manifold vertices using one-ring connected component analysis.

    A vertex is non-manifold if the faces incident to it form more than one
    disconnected group when connected by shared edges through that vertex.

    For triangular meshes this simplifies: two incident faces are connected
    iff they share another vertex (besides the one being checked), because
    each triangle edge through v has exactly one other vertex. This
    simplification does not hold for general polygons.
    """

    @property
    def metadata(self) -> OperationMetadata:
        return OperationMetadata(
            operation_id='topology.detect_nonmanifold_vertices',
            display_name='Non-Manifold Vertex Detection',
            description='Identifies vertices whose incident faces form '
                       'disconnected groups, violating the 2-manifold property.',
            category=IssueCategory.TOPOLOGICAL,
            supported_libraries=['trimesh', 'pyvista', 'pymeshlab'],
            recommended_library='pyvista',
            requires_adjacency=True,
            complexity='O(V + F)',
            tags=['topology', 'manifold', 'connectivity', 'vertices']
        )

    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        """Execute non-manifold vertex detection."""
        faces = mesh.faces
        n_faces = len(faces)
        if n_faces == 0:
            return []

        n_verts_per_face = faces.shape[1] if faces.ndim == 2 else None

        if n_verts_per_face == 3:
            nm_vertices = self._detect_tri(mesh)
        else:
            nm_vertices = self._detect_general(mesh)

        if not nm_vertices:
            return []

        # Build issue locations
        locations = []
        vertices = mesh.vertices
        for vi, n_components, component_faces in nm_vertices:
            coords = tuple(vertices[vi]) if vi < len(vertices) else None
            locations.append(IssueLocation(
                element_type='vertex',
                element_index=int(vi),
                coordinates=coords,
                adjacent_elements={
                    'connected_components': int(n_components),
                    'incident_face_count': sum(len(c) for c in component_faces),
                    'component_faces': [list(map(int, c)) for c in component_faces],
                }
            ))

        violation_rate = len(nm_vertices) / max(len(vertices), 1)
        severity = determine_severity(violation_rate)

        issue = MeshIssue(
            operation_id=self.metadata.operation_id,
            category=IssueCategory.TOPOLOGICAL,
            severity=severity,
            message=f'Detected {len(nm_vertices)} non-manifold vertices '
                   f'({violation_rate * 100:.2f}% of vertices). '
                   f'These vertices have incident faces that split into '
                   f'disconnected groups.',
            locations=locations,
            metrics={
                'count': len(nm_vertices),
                'violation_rate': violation_rate,
                'total_vertices': len(vertices),
            },
            metadata={
                'total_faces': n_faces,
            }
        )

        return [issue]

    def _detect_tri(
        self, mesh: MeshInput
    ) -> List[Tuple[int, int, List[List[int]]]]:
        """Vectorized path for triangular meshes.

        For triangles, two faces incident to vertex v are connected iff they
        share another vertex, because each triangle edge through v has exactly
        one other vertex. This simplification is specific to triangles and
        does not hold for general polygons.

        Returns list of (vertex_index, n_components, component_faces) tuples.
        """
        faces = mesh.faces
        n_faces = len(faces)
        n_verts = len(mesh.vertices)

        # Build vertex -> list of incident face indices
        # Each face contributes 3 vertex-face pairs.
        # ravel() gives vertices in face order: [f0[0],f0[1],f0[2], f1[0],...]
        # so face_indices must use repeat (not tile) to stay aligned.
        all_verts = faces.ravel()  # (3*F,)
        face_indices = np.repeat(np.arange(n_faces), 3)  # (3*F,)

        # Sort by vertex id for grouping
        sort_order = np.argsort(all_verts, kind='mergesort')
        sorted_verts = all_verts[sort_order]
        sorted_faces = face_indices[sort_order]

        # Find boundaries between different vertex ids
        boundaries = np.searchsorted(sorted_verts, np.arange(n_verts), side='left')
        boundaries_end = np.searchsorted(sorted_verts, np.arange(n_verts), side='right')

        nm_vertices = []
        for vi in range(n_verts):
            start, end = boundaries[vi], boundaries_end[vi]
            n_incident = end - start
            if n_incident <= 1:
                # 0 or 1 incident faces — trivially manifold (or isolated)
                continue

            incident_face_indices = sorted_faces[start:end]

            # For each incident face, get the two "other" vertices
            incident_faces = faces[incident_face_indices]  # (n_incident, 3)
            # Mask out the central vertex to get the two neighbors
            other_verts = np.where(
                incident_faces != vi,
                incident_faces,
                -1  # placeholder
            )

            # Build face-face adjacency: two faces are connected if they share
            # an edge through vertex vi (i.e., they have 2 vertices in common:
            # vi and one other vertex).
            n_inc = len(incident_face_indices)

            # Build adjacency: faces share an edge iff they have 2 vertices in common
            adj: Dict[int, Set[int]] = defaultdict(set)
            for i in range(n_inc):
                for j in range(i + 1, n_inc):
                    # Get vertices of each face (excluding vi)
                    verts_i = set(incident_faces[i]) - {vi}
                    verts_j = set(incident_faces[j]) - {vi}
                    # Faces share an edge if they share exactly one other vertex
                    # (plus vi = 2 vertices total in common)
                    if len(verts_i & verts_j) == 1:
                        adj[i].add(j)
                        adj[j].add(i)

            # BFS to count connected components
            visited = [False] * n_inc
            components: List[List[int]] = []

            for i in range(n_inc):
                if visited[i]:
                    continue
                component: List[int] = []
                queue = deque([i])
                visited[i] = True
                while queue:
                    node = queue.popleft()
                    component.append(node)
                    for neighbor in adj.get(node, set()):
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)
                components.append(component)

            if len(components) > 1:
                # Map local indices back to global face indices
                global_components = []
                for comp in components:
                    global_components.append(
                        sorted_faces[start:end][comp].tolist()
                    )
                nm_vertices.append((vi, len(components), global_components))

        return nm_vertices

    def _detect_general(
        self, mesh: MeshInput
    ) -> List[Tuple[int, int, List[List[int]]]]:
        """Iterative path for polygonal meshes.

        For general polygons, two faces incident to vertex v are connected
        iff they share an edge (v, w) for some neighbor w. This is more
        precise than the triangle shortcut because polygon edges through v
        are defined by consecutive vertex pairs in the face, not just any
        shared vertex.

        Returns list of (vertex_index, n_components, component_faces) tuples.
        """
        faces = mesh.faces
        n_faces = len(faces)

        # Build vertex -> list of incident face indices
        vert_faces: Dict[int, List[int]] = defaultdict(list)
        for fi in range(n_faces):
            for v in faces[fi]:
                vert_faces[int(v)].append(fi)

        nm_vertices = []
        for vi, incident_fis in vert_faces.items():
            if len(incident_fis) <= 1:
                continue

            # For each incident face, extract the edges through vi
            # An edge through vi is (vi, prev) or (vi, next) where
            # prev/next are the adjacent vertices in the face polygon.
            face_edges: Dict[int, Set[Tuple[int, ...]]] = {}
            for fi in incident_fis:
                face_arr = faces[fi]
                n = len(face_arr)
                edges: Set[Tuple[int, ...]] = set()
                for k in range(n):
                    if int(face_arr[k]) == vi:
                        prev_v = int(face_arr[(k - 1) % n])
                        next_v = int(face_arr[(k + 1) % n])
                        edges.add((min(vi, prev_v), max(vi, prev_v)))
                        edges.add((min(vi, next_v), max(vi, next_v)))
                face_edges[fi] = edges

            # Two faces are connected if they share any edge through vi
            adj: Dict[int, Set[int]] = defaultdict(set)
            for i_idx in range(len(incident_fis)):
                for j_idx in range(i_idx + 1, len(incident_fis)):
                    fi, fj = incident_fis[i_idx], incident_fis[j_idx]
                    if face_edges[fi] & face_edges[fj]:
                        adj[fi].add(fj)
                        adj[fj].add(fi)

            # BFS to count connected components
            visited: Set[int] = set()
            components: List[List[int]] = []
            for fi in incident_fis:
                if fi in visited:
                    continue
                component: List[int] = []
                queue = deque([fi])
                visited.add(fi)
                while queue:
                    node = queue.popleft()
                    component.append(node)
                    for neighbor in adj.get(node, set()):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                components.append(component)

            if len(components) > 1:
                nm_vertices.append((vi, len(components), components))

        return nm_vertices


# Register operation
registry.register(NonManifoldVertexDetection())
