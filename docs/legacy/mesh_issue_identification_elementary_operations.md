# Elementary Operations for Mesh Issue Identification and Scoring

## Executive Summary

This document provides a comprehensive catalog of elementary operations required for automated mesh issue identification without visual inspection. Each operation is designed to be composable into complex workflows for the MCP (Model Context Protocol) server, enabling systematic mesh quality assessment across topological, geometric, shape, and orientation defect categories.

The elementary operations catalogued herein derive from the analysis of common mesh quality issues documented in the technical reference "Common Mesh Quality Issues in 3D Surface and Volume Meshes." These operations serve as atomic units of computation that can be orchestrated by an MCP server to construct sophisticated mesh validation pipelines suitable for industrial simulation preparation, 3D printing workflows, scientific visualization, and finite element analysis.

---

## 1. Topological Integrity Operations

Topological integrity operations examine the combinatorial structure of mesh connectivity independent of geometric coordinates. These operations detect violations of the 2-manifold property essential for solid modeling, Boolean operations, and volume definition.

### 1.1 Non-Manifold Edge Detection

**Operation Identifier:** `topology.detect_nonmanifold_edges`

**Description:** Identifies edges shared by more than two faces, violating the fundamental 2-manifold requirement that each edge must be incident to exactly two faces. Non-manifold edges create topological ambiguity where interior/exterior determination becomes impossible.

**Basic Requirements:**
- Input mesh must have explicit or reconstructable edge-face adjacency information
- Vertex coordinate tolerance for geometric matching when explicit connectivity is unavailable
- Memory allocation for edge-to-face hash map construction

**Elementary Operation Implementation:**

```python
def detect_nonmanifold_edges(vertices, faces, tolerance=1e-6):
    """
    Detect edges shared by more than two faces.

    Parameters:
        vertices: numpy array of shape (n_vertices, 3)
        faces: numpy array of shape (n_faces, 3) or (n_faces, n_vertices_per_face)
        tolerance: float for vertex coordinate equality comparison

    Returns:
        nonmanifold_edges: list of edge tuples with incident face count > 2
        edge_face_map: dict mapping edge to list of incident face indices
    """
    import numpy as np

    # Build edge-to-face adjacency map
    edge_face_map = {}

    for face_idx, face in enumerate(faces):
        n = len(face)
        for i in range(n):
            # Create undirected edge as sorted vertex pair
            v1, v2 = face[i], face[(i + 1) % n]
            edge = tuple(sorted([v1, v2]))

            if edge not in edge_face_map:
                edge_face_map[edge] = []
            edge_face_map[edge].append(face_idx)

    # Identify non-manifold edges
    nonmanifold_edges = []
    for edge, faces_incident in edge_face_map.items():
        if len(faces_incident) != 2:
            nonmanifold_edges.append({
                'edge': edge,
                'face_count': len(faces_incident),
                'incident_faces': faces_incident
            })

    return nonmanifold_edges, edge_face_map
```

**Library Implementations:**

| Library | Function | Performance | License | Notes |
|---------|----------|-------------|---------|-------|
| **MeshLib** | `findNonManifoldEdges()` | 10× faster than alternatives | Commercial (free for education) | GPU acceleration available; pybind11 bindings |
| **PyPrimeMesh** | `SurfaceSearch.get_surface_diagnostic_summary(compute_multi_edges=True)` | Server-optimized, parallel | Proprietary | Requires Ansys Prime Server license |
| **TRimesh** | Manual via `edges_unique` + `edges_face` | O(F) | MIT | Pure Python with NumPy; requires manual implementation |
| **PyMeshLab** | `select_non_manifold_edges` filter | C++ optimized | GPL | Batch filter paradigm; 30+ formats |
| **PyMeshFix** | `MeshFix.find_defects()` | Fast | Apache 2.0 | Wrapper for libCGAL; specifically detects non-manifold edges |
| **scipy.spatial** | Custom with `cKDTree` + adjacency | O(F log F) | BSD | Requires manual edge construction; flexible tolerance |
| **blenderbpy** | `bmesh.ops.triangle_fill` + validation | Moderate | GPL | Available in Blender environment; mesh inspection mode |

**Advanced Implementation with scipy:**

```python
import numpy as np
from scipy.spatial import cKDTree

def detect_nonmanifold_edges_scipy(vertices, faces, tolerance=1e-6):
    """Non-manifold detection using scipy spatial structures."""
    # Build vertex proximity tree for tolerance-based matching
    tree = cKDTree(vertices)

    # For each edge, find vertices within tolerance
    # Then build edge-face adjacency with tolerance
    edge_face_map = {}
    # ... implementation with tree.query_pairs() for tolerance matching
    return nonmanifold_edges
```

**Scoring Metrics:**
- Non-manifold edge count normalized by total edge count
- Ratio of non-manifold edge instances to unique non-manifold edges
- Classification by incident face count (3, 4, or more faces)

---

### 1.2 Non-Manifold Vertex Detection

**Operation Identifier:** `topology.detect_nonmanifold_vertices`

**Description:** Identifies vertices whose adjacent faces fail to form a single, continuous, simply-connected fan. The mathematical condition requires that the link of the vertex (boundary of a small spherical neighborhood intersected with the surface) be homeomorphic to a circle.

**Basic Requirements:**
- 1-ring neighborhood connectivity for each vertex
- Face adjacency graph construction within vertex neighborhood
- Union-find data structure for connected component analysis

**Elementary Operation Implementation:**

```python
def detect_nonmanifold_vertices(vertices, faces):
    """
    Detect vertices with disconnected face adjacency fans.

    Parameters:
        vertices: numpy array of shape (n_vertices, 3)
        faces: numpy array of shape (n_faces, 3)

    Returns:
        nonmanifold_vertices: list of dicts with vertex index and violation type
        vertex_links: dict mapping vertex to its link structure
    """
    import numpy as np

    # Build vertex-to-faces adjacency
    vertex_faces = {}
    for face_idx, face in enumerate(faces):
        for v in face:
            if v not in vertex_faces:
                vertex_faces[v] = []
            vertex_faces[v].append(face_idx)

    nonmanifold_vertices = []
    vertex_links = {}

    for v_idx, incident_faces in vertex_faces.items():
        # Build face adjacency graph in 1-ring neighborhood
        face_neighbors = {}
        for f_idx in incident_faces:
            face_neighbors[f_idx] = set()

        # Find face-face adjacencies through edges
        for i, f1 in enumerate(incident_faces):
            for j, f2 in enumerate(incident_faces):
                if i >= j:
                    continue
                # Check if faces share an edge
                f1_verts = set(faces[f1])
                f2_verts = set(faces[f2])
                shared = f1_verts & f2_verts
                if len(shared) == 2:  # Shared edge
                    face_neighbors[f1].add(f2)
                    face_neighbors[f2].add(f1)

        # Find connected components in face adjacency
        visited = set()
        components = []

        def bfs(start):
            component = []
            queue = [start]
            while queue:
                f = queue.pop(0)
                if f in visited:
                    continue
                visited.add(f)
                component.append(f)
                queue.extend(face_neighbors[f])
            return component

        for f_idx in incident_faces:
            if f_idx not in visited:
                comp = bfs(f_idx)
                if comp:
                    components.append(comp)

        # Non-manifold if multiple disconnected components
        if len(components) > 1:
            nonmanifold_vertices.append({
                'vertex_index': v_idx,
                'num_components': len(components),
                'components': components,
                'violation_type': 'disconnected_face_fan'
            })

        vertex_links[v_idx] = {
            'num_faces': len(incident_faces),
            'num_components': len(components),
            'cycles': len(components)
        }

    return nonmanifold_vertices, vertex_links
```

**Library Implementations:**

| Library | Function | Complexity | License | Notes |
|---------|----------|------------|---------|-------|
| **MeshLib** | `findNonManifoldVertices()` | O(V α(V)) | Commercial | Union-find with exact arithmetic; automatic repair capability |
| **PyPrimeMesh** | Embedded in surface diagnostic summary | Implicit | Proprietary | Less explicit API exposure than edge detection |
| **PyMeshLab** | `select_non_manifold_vertices` filter | O(V k) | GPL | Vertex classification by incident face count |
| **PyMeshFix** | `MeshFix.find_defects()` | O(V + F) | Apache 2.0 | Comprehensive defect detection including vertices |
| **scipy.ndimage** | `label()` for connected components | O(F) | BSD | Can be adapted for vertex neighborhood analysis |
| **compas** | `compas.topology` functions | Varies | MIT | Framework for computational analysis and mesh processing |
| **blenderbpy** | `bmesh` vertex validation | Moderate | GPL | Mesh analysis mode in Blender |

**Implementation with scipy.ndimage for Vertex Neighborhood:**

```python
import numpy as np
from scipy import ndimage

def detect_nonmanifold_vertices_scipy(vertices, faces):
    """Alternative using scipy.ndimage connected components on vertex adjacency."""
    n_vertices = len(vertices)
    n_faces = len(faces)

    # Build vertex-face incidence as sparse binary matrix
    # Then analyze each vertex's face neighborhood
    # Use label() on face neighborhood graphs

    return nonmanifold_vertices
```

**Scoring Metrics:**
- Non-manifold vertex count
- Number of disconnected face components per vertex
- Severity classification based on component count

---

### 1.3 Mixed-Dimensionality Detection

**Operation Identifier:** `topology.detect_mixed_dimensionality`

**Description:** Identifies improper connections between elements of different intrinsic dimensions—2D surfaces connected by 1D edges only, 3D volumes touching at 0D points or 1D edges rather than shared 2D faces.

**Basic Requirements:**
- Element dimensionality labeling
- Cross-dimensional adjacency detection
- Classification of valid versus invalid connection patterns

**Elementary Operation Implementation:**

```python
def detect_mixed_dimensionality(mesh_entities):
    """
    Detect elements of different dimensions improperly joined.

    Parameters:
        mesh_entities: dict with keys 'vertices', 'edges', 'faces', 'volumes'
                      each containing numpy arrays of entity definitions

    Returns:
        violations: list of dicts describing dimensional connection violations
        entity_dimensions: dict mapping entity indices to dimensional labels
    """
    violations = []
    entity_dimensions = {}

    # Label vertices as 0-dimensional
    if 'vertices' in mesh_entities:
        for v in mesh_entities['vertices']:
            entity_dimensions[('vertex', v)] = 0

    # Label edges as 1-dimensional
    if 'edges' in mesh_entities:
        for e in mesh_entities['edges']:
            entity_dimensions[('edge', tuple(e))] = 1

    # Label faces as 2-dimensional
    if 'faces' in mesh_entities:
        for f in mesh_entities['faces']:
            entity_dimensions[('face', f)] = 2

    # Label volumes as 3-dimensional
    if 'volumes' in mesh_entities:
        for vol in mesh_entities['volumes']:
            entity_dimensions[('volume', vol)] = 3

    # Detect invalid cross-dimensional connections
    # For volumes: must connect through shared faces, not edges or vertices
    # For surfaces: must connect through shared edges, not vertices only

    return violations, entity_dimensions
```

**Library Implementations:**

| Library | Capability | License | Notes |
|---------|------------|---------|-------|
| **PyPrimeMesh** | Volume-surface intersection diagnostics | Proprietary | Ray-casting with spatial indexing |
| **VTK/Custom** | `vtkCellType` analysis + `vtkCellLinks` | BSD | Substantial development effort required |
| **Gmsh** | `model.mesh.intersect()` | GPL | Dimensional classification for mesh generation |
| **MeshLab/CGAL** | `check_dimensional_alignment` | LGPL/GPL | Explicit dimensional consistency checking |
| **PyMeshLab** | `check_meshes_dimensionality` filter | GPL | Batch processing with dimensionality validation |
| **compas** | `compas.geometry` dimensional entities | MIT | Native support for multi-dimensional meshes |

**Advanced Gmsh Python Implementation:**

```python
import gmsh

def detect_mixed_dimensionality_gmsh(model):
    """Use Gmsh API for dimensional consistency checking."""
    # Get all entities and their dimensions
    dim_tags = model.getEntities()
    dim_entities = {}
    for dim, tag in dim_tags:
        if dim not in dim_entities:
            dim_entities[dim] = []
        dim_entities[dim].append(tag)

    # Check physical groups for dimensional consistency
    physical_groups = model.getPhysicalGroups()
    # Validate that physical groups contain entities of single dimension

    return violations
```

**Scoring Metrics:**
- Number of dimensional violation clusters
- Severity based on dimension difference (e.g., 3D-0D more severe than 3D-2D)
- Ratio of invalid connections to total connections

---

### 1.4 Free Edge (Hole) Detection

**Operation Identifier:** `topology.detect_free_edges`

**Description:** Identifies edges belonging to exactly one face, indicating missing surface patches that prevent watertight closure. Also termed naked edges or boundary edges.

**Basic Requirements:**
- Edge-to-face adjacency map
- Single-face incidence classification
- Distinction from intended open boundaries

**Elementary Operation Implementation:**

```python
def detect_free_edges(faces):
    """
    Identify edges belonging to exactly one face (boundary/naked edges).

    Parameters:
        faces: numpy array of shape (n_faces, n_vertices_per_face)

    Returns:
        free_edges: list of edge tuples that are boundary edges
        boundary_loops: list of connected edge cycles forming holes
    """
    import numpy as np
    from collections import defaultdict

    edge_faces = defaultdict(list)

    for face_idx, face in enumerate(faces):
        n = len(face)
        for i in range(n):
            v1, v2 = face[i], face[(i + 1) % n]
            edge = tuple(sorted([v1, v2]))
            edge_faces[edge].append(face_idx)

    free_edges = [edge for edge, faces_list in edge_faces.items()
                  if len(faces_list) == 1]

    # Group into boundary loops
    boundary_loops = []
    visited_edges = set(free_edges)
    remaining = set(free_edges)

    while remaining:
        edge = next(iter(remaining))
        remaining.discard(edge)
        loop = [edge]

        current = edge
        while True:
            # Find adjacent edge in loop
            found_next = False
            for next_edge in remaining:
                if current[1] == next_edge[0] or current[1] == next_edge[1]:
                    loop.append(next_edge)
                    remaining.discard(next_edge)
                    current = next_edge
                    found_next = True
                    break

            if not found_next:
                break

        boundary_loops.append({
            'edges': loop,
            'num_edges': len(loop),
            'is_simple_loop': loop[0][0] == loop[-1][1]
        })

    return free_edges, b
