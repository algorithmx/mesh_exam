"""
Topological utilities for mesh analysis.

Provides functions for:
- Vertex valence computation
- Edge-face adjacency building
- Vertex-face mapping
"""

from typing import Dict, List, Tuple, Set
import numpy as np


def compute_vertex_valence(faces: np.ndarray, n_vertices: int) -> np.ndarray:
    """
    Compute valence (number of incident faces) for each vertex.
    
    Valence is the number of edges/faces incident to a vertex.
    Abnormal valence indicates mesh irregularities.
    
    For triangle meshes:
    - Interior vertices: ideal = 6, problematic < 4 or > 8
    - Boundary vertices: ideal = 4, problematic < 2 or > 6
    
    For quad meshes:
    - Interior vertices: ideal = 4, problematic < 3 or > 6
    - Boundary vertices: ideal = 2, problematic < 2 or > 4
    
    Algorithm: Count face occurrences for each vertex
    
    Time: O(F * k) where k = vertices per face
    Space: O(V)
    
    Parameters
    ----------
    faces : (n_faces, k) array or 1D object array for mixed types
    n_vertices : int
        Total number of vertices in mesh
        
    Returns
    -------
    valence : (n_vertices,) array of int
        Valence for each vertex
    """
    valence = np.zeros(n_vertices, dtype=int)
    
    # Handle both uniform and ragged faces
    if faces.ndim == 2:
        # Uniform faces (all triangles or all quads)
        for i in range(faces.shape[1]):
            np.add.at(valence, faces[:, i], 1)
    else:
        # Ragged faces (mixed types)
        for face in faces:
            for vertex in face:
                valence[int(vertex)] += 1
    
    return valence


def build_vertex_faces_map(faces: np.ndarray, n_vertices: int) -> List[List[int]]:
    """
    Build mapping from vertices to adjacent faces.
    
    Parameters
    ----------
    faces : (n_faces, k) array or 1D object array
    n_vertices : int
        Total number of vertices
        
    Returns
    -------
    vertex_faces : list of lists
        vertex_faces[v] = list of face indices containing vertex v
    """
    vertex_faces: List[List[int]] = [[] for _ in range(n_vertices)]
    
    for f_idx, face in enumerate(faces):
        for v_idx in face:
            vertex_faces[int(v_idx)].append(f_idx)
    
    return vertex_faces


def build_edge_face_map(faces: np.ndarray) -> Dict[Tuple[int, int], List[int]]:
    """
    Build mapping from edges to adjacent faces.

    An edge is defined by two vertex indices (sorted for canonical ordering).

    Uniform triangular meshes (2D array with 3 vertices per face and a
    numeric dtype) take a fully vectorized fast path; ragged, object-dtype,
    or non-triangular face arrays fall back to the reference Python loop
    (:func:`_build_edge_face_map_loop`).

    Parameters
    ----------
    faces : (n_faces, k) array or 1D object array
        Face connectivity array

    Returns
    -------
    edge_faces : dict
        Mapping from (v1, v2) edge tuple to list of face indices.
        Face lists are kept in ascending face order.
    """
    if len(faces) == 0:
        return {}

    # Vectorized fast path for uniform triangle meshes
    if faces.ndim == 2 and faces.shape[1] == 3 and faces.dtype.kind in 'iuf':
        return _build_edge_face_map_triangles(faces)

    # Fallback for ragged/object-dtype or non-triangular faces
    return _build_edge_face_map_loop(faces)


def _build_edge_face_map_triangles(faces: np.ndarray) -> Dict[Tuple[int, int], List[int]]:
    """
    Vectorized edge-face adjacency for uniform triangle meshes.

    Each undirected edge is packed into a single int64 key
    ``min * n_vertices + max`` so edge occurrences can be grouped with
    ``np.unique(return_inverse=True)`` plus a stable argsort of the inverse
    array, instead of Python-level dict updates.

    Parameters
    ----------
    faces : (n_faces, 3) array with numeric dtype
        Triangle connectivity array

    Returns
    -------
    edge_faces : dict
        Mapping from (v_min, v_max) edge tuple to the ascending list of
        incident face indices
    """
    n_faces = faces.shape[0]
    n_vertices = int(faces.max()) + 1

    tri = faces.astype(np.int64, copy=False)

    # Per-face edge arrays for the three triangle edges:
    # (v0, v1), (v1, v2), (v0, v2)
    e_min = np.vstack([
        np.minimum(tri[:, 0], tri[:, 1]),
        np.minimum(tri[:, 1], tri[:, 2]),
        np.minimum(tri[:, 0], tri[:, 2]),
    ])
    e_max = np.vstack([
        np.maximum(tri[:, 0], tri[:, 1]),
        np.maximum(tri[:, 1], tri[:, 2]),
        np.maximum(tri[:, 0], tri[:, 2]),
    ])

    # Pack each edge into one int64 key. Flatten in face-major order so the
    # stable sort below keeps every edge's face list in ascending face order,
    # exactly like the sequential-loop implementation.
    keys = (e_min * n_vertices + e_max).ravel(order='F')
    face_indices = np.repeat(np.arange(n_faces, dtype=np.int64), 3)

    unique_keys, inverse = np.unique(keys, return_inverse=True)

    # Stable argsort groups edge occurrences by edge key while preserving the
    # face-major occurrence order within each group. Groups are then cut at
    # the boundaries where the grouped inverse values change (equivalent to
    # np.split, but slicing one flat Python list is much cheaper than
    # materializing a small ndarray per group).
    order = np.argsort(inverse, kind='stable')
    grouped_faces = face_indices[order]
    boundaries = np.flatnonzero(np.diff(inverse[order])) + 1

    counts = np.diff(np.concatenate(([0], boundaries, [3 * n_faces])))
    flat_faces = grouped_faces.tolist()
    edge_mins = (unique_keys // n_vertices).tolist()
    edge_maxs = (unique_keys % n_vertices).tolist()

    edge_faces: Dict[Tuple[int, int], List[int]] = {}
    position = 0
    for v_min, v_max, count in zip(edge_mins, edge_maxs, counts.tolist()):
        edge_faces[(v_min, v_max)] = flat_faces[position:position + count]
        position += count

    return edge_faces


def _build_edge_face_map_loop(faces: np.ndarray) -> Dict[Tuple[int, int], List[int]]:
    """
    Reference loop implementation of :func:`build_edge_face_map`.

    Kept as the correctness/performance baseline for the equivalence tests
    and benchmarks, and as the fallback for ragged, object-dtype, or
    non-triangular face arrays.

    Parameters
    ----------
    faces : (n_faces, k) array or 1D object array
        Face connectivity array

    Returns
    -------
    edge_faces : dict
        Mapping from (v1, v2) edge tuple to list of face indices
    """
    edge_faces: Dict[Tuple[int, int], List[int]] = {}

    for f_idx, face in enumerate(faces):
        n_verts = len(face)
        for i in range(n_verts):
            v1 = int(face[i])
            v2 = int(face[(i + 1) % n_verts])

            # Canonical edge ordering (smaller index first)
            edge = tuple(sorted([v1, v2]))

            if edge not in edge_faces:
                edge_faces[edge] = []
            edge_faces[edge].append(f_idx)

    return edge_faces


def extract_unique_edges(faces: np.ndarray) -> np.ndarray:
    """
    Extract unique edges from faces.
    
    Parameters
    ----------
    faces : (n_faces, k) array or 1D object array
        
    Returns
    -------
    edges : (n_edges, 2) array
        Unique edges as vertex pairs
    """
    edge_set: Set[Tuple[int, int]] = set()
    
    for face in faces:
        n_verts = len(face)
        for i in range(n_verts):
            v1 = int(face[i])
            v2 = int(face[(i + 1) % n_verts])
            edge = tuple(sorted([v1, v2]))
            edge_set.add(edge)
    
    return np.array(sorted(edge_set), dtype=int)


def compute_edge_lengths(vertices: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Compute length of each edge.
    
    Parameters
    ----------
    vertices : (n_vertices, 3) array
    edges : (n_edges, 2) array
        Edge vertex indices
        
    Returns
    -------
    lengths : (n_edges,) array
        Length of each edge
    """
    v1 = vertices[edges[:, 0]]
    v2 = vertices[edges[:, 1]]
    return np.linalg.norm(v2 - v1, axis=1)


def compute_face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute area of each face.
    
    Parameters
    ----------
    vertices : (n_vertices, 3) array
    faces : (n_faces, k) array or 1D object array
        
    Returns
    -------
    areas : (n_faces,) array
        Area of each face
    """
    areas = np.zeros(len(faces))
    
    for i, face in enumerate(faces):
        if len(face) >= 3:
            verts = vertices[face.astype(int)]
            # Use cross product of first two edges
            if len(face) == 3:
                # Triangle
                e1 = verts[1] - verts[0]
                e2 = verts[2] - verts[0]
                areas[i] = 0.5 * np.linalg.norm(np.cross(e1, e2))
            else:
                # Polygon - triangulate from first vertex
                area = 0.0
                for j in range(1, len(face) - 1):
                    e1 = verts[j] - verts[0]
                    e2 = verts[j + 1] - verts[0]
                    area += 0.5 * np.linalg.norm(np.cross(e1, e2))
                areas[i] = area
    
    return areas
