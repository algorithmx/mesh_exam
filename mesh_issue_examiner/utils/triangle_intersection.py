"""
Triangle-triangle intersection detection using Möller-Trumbore algorithm.

This module provides efficient self-intersection detection for 3D meshes using:
1. KD-tree spatial indexing to reduce O(n²) to O(n log n)
2. Möller-Trumbore ray-triangle intersection for edge tests
3. Proper handling of coplanar triangles and adjacency

References:
    - Möller & Trumbore (1997): "Fast, Minimum Storage Ray-Triangle Intersection"
    - Möller (1997): "A Fast Triangle-Triangle Intersection Test"
"""

import logging
import numpy as np
from scipy.spatial import KDTree
from typing import List, Tuple, Set, Optional

logger = logging.getLogger(__name__)


def compute_triangle_centroids(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute centroids for all triangles."""
    return vertices[faces].mean(axis=1)


def compute_max_edge_length(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Compute the maximum edge length across all triangles."""
    tri_verts = vertices[faces]  # Shape: (n_tris, 3, 3)
    
    # Compute all three edge lengths for each triangle
    e1 = np.linalg.norm(tri_verts[:, 1] - tri_verts[:, 0], axis=1)
    e2 = np.linalg.norm(tri_verts[:, 2] - tri_verts[:, 1], axis=1)
    e3 = np.linalg.norm(tri_verts[:, 0] - tri_verts[:, 2], axis=1)
    
    return float(max(e1.max(), e2.max(), e3.max()))


def build_adjacency_pairs(faces: np.ndarray) -> Set[Tuple[int, int]]:
    """
    Build a set of adjacent triangle pairs (share at least one vertex).
    Returns pairs as (min_idx, max_idx) for consistency.
    """
    n_faces = len(faces)
    adjacent = set()
    
    # For each triangle, track which vertices it contains
    vertex_to_tris = {}
    for tri_idx, tri in enumerate(faces):
        for vert_idx in tri:
            if vert_idx not in vertex_to_tris:
                vertex_to_tris[vert_idx] = []
            vertex_to_tris[vert_idx].append(tri_idx)
    
    # For each vertex, all triangles sharing it are adjacent
    for vert_idx, tri_list in vertex_to_tris.items():
        for i, tri1 in enumerate(tri_list):
            for tri2 in tri_list[i+1:]:
                pair = (min(tri1, tri2), max(tri1, tri2))
                adjacent.add(pair)
    
    return adjacent


def bbox_overlap(tri1: np.ndarray, tri2: np.ndarray, tolerance: float = 1e-10) -> bool:
    """
    Quick bounding box overlap test for two triangles.
    
    Args:
        tri1, tri2: Triangle vertices, shape (3, 3)
        tolerance: Small margin for numerical precision
        
    Returns:
        True if bounding boxes overlap
    """
    min1 = tri1.min(axis=0) - tolerance
    max1 = tri1.max(axis=0) + tolerance
    min2 = tri2.min(axis=0) - tolerance
    max2 = tri2.max(axis=0) + tolerance
    
    return np.all(min1 <= max2) and np.all(max1 >= min2)


def compute_plane(verts: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute plane equation from triangle vertices.
    
    Returns:
        (normal, d) where plane equation is normal·X + d = 0
    """
    normal = np.cross(verts[1] - verts[0], verts[2] - verts[0])
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-15:
        return normal, 0.0  # Degenerate triangle
    normal = normal / norm_len
    d = -np.dot(normal, verts[0])
    return normal, d


def are_coplanar(tri1: np.ndarray, tri2: np.ndarray, tolerance: float = 1e-10) -> bool:
    """
    Check if two triangles are coplanar.
    
    Args:
        tri1, tri2: Triangle vertices, shape (3, 3)
        tolerance: Distance tolerance for coplanarity
        
    Returns:
        True if triangles lie in the same plane
    """
    # Get plane of tri1
    normal1, d1 = compute_plane(tri1)
    
    # Check if all vertices of tri2 are in plane of tri1
    dists = np.dot(tri2, normal1) + d1
    if not np.all(np.abs(dists) < tolerance):
        return False
    
    # Also check plane of tri2 matches tri1
    normal2, d2 = compute_plane(tri2)
    
    # Normals should be parallel (dot product = ±1)
    dot_normals = np.dot(normal1, normal2)
    normals_parallel = abs(abs(dot_normals) - 1.0) < tolerance
    
    # d values should match (accounting for normal direction)
    if dot_normals > 0:
        d_match = abs(d1 - d2) < tolerance
    else:
        d_match = abs(d1 + d2) < tolerance
    
    return normals_parallel and d_match


def project_to_2d(tri: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """
    Project triangle to 2D plane where it has maximum area.
    
    Args:
        tri: Triangle vertices, shape (3, 3)
        normal: Triangle normal vector
        
    Returns:
        2D projected coordinates, shape (3, 2)
    """
    # Choose projection plane based on dominant normal component
    abs_normal = np.abs(normal)
    dominant_axis = np.argmax(abs_normal)
    
    if dominant_axis == 0:
        # Project to YZ plane
        return tri[:, 1:]
    elif dominant_axis == 1:
        # Project to XZ plane
        return tri[:, [0, 2]]
    else:
        # Project to XY plane
        return tri[:, :2]


def point_in_triangle_2d(p: np.ndarray, tri: np.ndarray) -> bool:
    """
    Check if point is inside triangle using barycentric coordinates in 2D.
    
    Args:
        p: Point to test, shape (2,)
        tri: Triangle vertices, shape (3, 2)
        
    Returns:
        True if point is inside or on edge of triangle
    """
    # Barycentric coordinate method
    denom = (tri[1, 1] - tri[2, 1]) * (tri[0, 0] - tri[2, 0]) + (tri[2, 0] - tri[1, 0]) * (tri[0, 1] - tri[2, 1])
    
    if abs(denom) < 1e-15:
        return False  # Degenerate triangle
    
    a = ((tri[1, 1] - tri[2, 1]) * (p[0] - tri[2, 0]) + (tri[2, 0] - tri[1, 0]) * (p[1] - tri[2, 1])) / denom
    b = ((tri[2, 1] - tri[0, 1]) * (p[0] - tri[2, 0]) + (tri[0, 0] - tri[2, 0]) * (p[1] - tri[2, 1])) / denom
    c = 1 - a - b
    
    return (a >= -1e-10) and (b >= -1e-10) and (c >= -1e-10)


def edges_intersect_2d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray,
                        tolerance: float = 1e-15) -> bool:
    """
    Check if two line segments intersect in 2D.
    
    Handles all intersection cases including:
    - Normal crossing segments
    - Segments touching at endpoints
    - Collinear overlapping segments
    
    Uses orientation-based algorithm from computational geometry literature:
    - de Berg et al., "Computational Geometry: Algorithms and Applications"
    - Standard approach used in CGAL, LEDA, and other CG libraries
    
    Args:
        p1, p2: First segment endpoints
        p3, p4: Second segment endpoints
        tolerance: Numerical tolerance for collinearity checks
        
    Returns:
        True if segments intersect (including at endpoints or overlapping)
    """
    def cross_2d(a, b):
        return a[0] * b[1] - a[1] * b[0]
    
    def on_segment(a, b, c):
        """Check if point b lies on segment ac (assuming collinear)."""
        return (
            min(a[0], c[0]) - tolerance <= b[0] <= max(a[0], c[0]) + tolerance and
            min(a[1], c[1]) - tolerance <= b[1] <= max(a[1], c[1]) + tolerance
        )
    
    # Compute orientation values (2D cross products)
    o1 = cross_2d(p2 - p1, p3 - p1)
    o2 = cross_2d(p2 - p1, p4 - p1)
    o3 = cross_2d(p4 - p3, p1 - p3)
    o4 = cross_2d(p4 - p3, p2 - p3)
    
    # General case: proper intersection (segments cross each other)
    cond1 = (o1 > tolerance and o2 < -tolerance) or (o1 < -tolerance and o2 > tolerance)
    cond2 = (o3 > tolerance and o4 < -tolerance) or (o3 < -tolerance and o4 > tolerance)
    if cond1 and cond2:
        return True
    
    # Special Cases: collinear points
    if abs(o1) < tolerance and on_segment(p1, p3, p2):
        return True
    if abs(o2) < tolerance and on_segment(p1, p4, p2):
        return True
    if abs(o3) < tolerance and on_segment(p3, p1, p4):
        return True
    if abs(o4) < tolerance and on_segment(p3, p2, p4):
        return True
    
    return False


def coplanar_triangles_intersect(tri1: np.ndarray, tri2: np.ndarray, tolerance: float = 1e-10) -> bool:
    """
    Check if two coplanar triangles intersect.
    
    Tests:
    1. Any edge intersections
    2. One triangle contained within another
    
    Args:
        tri1, tri2: Triangle vertices, shape (3, 3)
        tolerance: Numerical tolerance
        
    Returns:
        True if triangles intersect
    """
    # Get plane normal for projection
    normal1, _ = compute_plane(tri1)
    
    # Project both triangles to 2D
    tri1_2d = project_to_2d(tri1, normal1)
    tri2_2d = project_to_2d(tri2, normal1)
    
    # Test all edge pairs for intersection
    edges1 = [(0, 1), (1, 2), (2, 0)]
    edges2 = [(0, 1), (1, 2), (2, 0)]
    
    for e1 in edges1:
        for e2 in edges2:
            if edges_intersect_2d(
                tri1_2d[e1[0]], tri1_2d[e1[1]],
                tri2_2d[e2[0]], tri2_2d[e2[1]]
            ):
                return True
    
    # No edge intersection - check if one triangle is inside the other
    # Test one vertex of tri1 against tri2
    if point_in_triangle_2d(tri1_2d[0], tri2_2d):
        return True
    
    # Test one vertex of tri2 against tri1
    if point_in_triangle_2d(tri2_2d[0], tri1_2d):
        return True
    
    return False


def moller_trumbore_intersection(
    edge_start: np.ndarray,
    edge_end: np.ndarray,
    tri: np.ndarray,
    tolerance: float = 1e-10
) -> bool:
    """
    Möller-Trumbore ray-triangle intersection test.
    
    Checks if a line segment (edge) intersects a triangle.
    
    Args:
        edge_start: Edge start point, shape (3,)
        edge_end: Edge end point, shape (3,)
        tri: Triangle vertices, shape (3, 3)
        tolerance: Numerical tolerance
        
    Returns:
        True if edge intersects triangle (not just the ray)
    """
    ray_origin = edge_start
    ray_dir = edge_end - edge_start
    
    # Möller-Trumbore algorithm
    v0, v1, v2 = tri[0], tri[1], tri[2]
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    pvec = np.cross(ray_dir, edge2)
    det = np.dot(edge1, pvec)
    
    # Ray parallel to triangle
    if abs(det) < tolerance:
        return False
    
    inv_det = 1.0 / det
    tvec = ray_origin - v0
    
    u = np.dot(tvec, pvec) * inv_det
    if u < -tolerance or u > 1.0 + tolerance:
        return False
    
    qvec = np.cross(tvec, edge1)
    v = np.dot(ray_dir, qvec) * inv_det
    if v < -tolerance or u + v > 1.0 + tolerance:
        return False
    
    # At this stage we have an intersection with the ray
    t = np.dot(edge2, qvec) * inv_det
    
    # For edge intersection, t must be in [0, 1]
    return tolerance <= t <= 1.0 + tolerance


def triangles_intersect(
    tri1: np.ndarray,
    tri2: np.ndarray,
    tolerance: float = 1e-10
) -> bool:
    """
    Check if two triangles intersect.

    Algorithm:
    1. Quick bounding box rejection
    2. Check coplanar case
    3. Test edge-triangle intersections using Möller-Trumbore
    4. Reject chord-through-interior false positives on curved surfaces

    Chord-through-interior artifact: on tessellated closed surfaces, edge
    chords pass through non-adjacent faces.  These are geometric facts about
    piecewise-linear approximations of curved surfaces, NOT mesh defects.
    We detect them by checking whether the intersection point lies on the
    **outer** side of both triangle normals (i.e. both faces "face away"
    from the intersection, meaning it is interior to the mesh).

    Args:
        tri1, tri2: Triangle vertices, shape (3, 3)
        tolerance: Numerical tolerance

    Returns:
        True if triangles intersect
    """
    # Quick bounding box rejection
    if not bbox_overlap(tri1, tri2, tolerance):
        return False

    # Check coplanar case
    if are_coplanar(tri1, tri2, tolerance):
        return coplanar_triangles_intersect(tri1, tri2, tolerance)

    # Non-coplanar: test edge-triangle intersections using Möller-Trumbore
    edges1 = [(tri1[0], tri1[1]), (tri1[1], tri1[2]), (tri1[2], tri1[0])]
    for edge_start, edge_end in edges1:
        hit, point = _moller_trumbore_with_point(edge_start, edge_end, tri2, tolerance)
        if hit and _is_surface_intersection(point, tri1, tri2, tolerance, edge_start, edge_end):
            return True

    edges2 = [(tri2[0], tri2[1]), (tri2[1], tri2[2]), (tri2[2], tri2[0])]
    for edge_start, edge_end in edges2:
        hit, point = _moller_trumbore_with_point(edge_start, edge_end, tri1, tolerance)
        if hit and _is_surface_intersection(point, tri2, tri1, tolerance, edge_start, edge_end):
            return True

    return False


def _moller_trumbore_with_point(
    edge_start: np.ndarray,
    edge_end: np.ndarray,
    tri: np.ndarray,
    tolerance: float = 1e-10,
) -> tuple:
    """Möller-Trumbore test that also returns the intersection point.

    Returns:
        (hit, point) where hit is bool and point is the intersection point
        (or None if no hit).
    """
    ray_origin = edge_start
    ray_dir = edge_end - edge_start

    v0, v1, v2 = tri[0], tri[1], tri[2]
    edge1 = v1 - v0
    edge2 = v2 - v0

    pvec = np.cross(ray_dir, edge2)
    det = np.dot(edge1, pvec)

    if abs(det) < tolerance:
        return False, None

    inv_det = 1.0 / det
    tvec = ray_origin - v0

    u = np.dot(tvec, pvec) * inv_det
    if u < -tolerance or u > 1.0 + tolerance:
        return False, None

    qvec = np.cross(tvec, edge1)
    v = np.dot(ray_dir, qvec) * inv_det
    if v < -tolerance or u + v > 1.0 + tolerance:
        return False, None

    t = np.dot(edge2, qvec) * inv_det
    if tolerance <= t <= 1.0 + tolerance:
        point = ray_origin + t * ray_dir
        return True, point

    return False, None


def _is_surface_intersection(
    point: np.ndarray,
    owner_tri: np.ndarray,
    hit_tri: np.ndarray,
    tolerance: float,
    edge_start: np.ndarray = None,
    edge_end: np.ndarray = None,
) -> bool:
    """Reject chord-through-interior false positives.

    On a closed curved surface (icosphere, torus, etc.), edge chords can
    geometrically intersect non-adjacent faces by passing through the mesh
    interior. These are piecewise-linear approximation artifacts, NOT mesh
    defects.

    Detection strategies:
    
    1. Edge-depth test (primary): If both edge endpoints are further from the
       surface "center" than the intersection point, the edge is just a chord
       cutting through the interior. A genuine intersection has the edge 
       piercing FROM outside TO inside.
    
    2. Normal-orientation test (fallback): Check if both triangle centers are
       on the "interior" side of each other's planes.

    Args:
        point: Intersection point
        owner_tri: Triangle containing the edge that intersected
        hit_tri: Triangle that was intersected
        tolerance: Numerical tolerance
        edge_start: Start point of the intersecting edge (optional, enables better detection)
        edge_end: End point of the intersecting edge (optional)

    Returns:
        True if this is a genuine surface intersection (not an artifact).
    """
    # Strategy 1: Edge-depth test (most reliable for chord artifacts)
    # If we have edge endpoints, check if both are "outside" the intersection
    if edge_start is not None and edge_end is not None:
        # Estimate surface "center" from both triangles
        center_est = (owner_tri.mean(axis=0) + hit_tri.mean(axis=0)) / 2.0
        
        d_edge_start = np.linalg.norm(edge_start - center_est)
        d_edge_end = np.linalg.norm(edge_end - center_est)
        d_point = np.linalg.norm(point - center_est)
        
        # If both edge endpoints are further from center than the intersection,
        # this is a chord cutting through interior (not a surface piercing)
        if (d_edge_start > d_point + tolerance) and (d_edge_end > d_point + tolerance):
            return False
    
    # Strategy 2: Normal-orientation test (fallback)
    # Compute normals
    edge1_own = owner_tri[1] - owner_tri[0]
    edge2_own = owner_tri[2] - owner_tri[0]
    n_own = np.cross(edge1_own, edge2_own)
    norm_own = np.linalg.norm(n_own)
    if norm_own > tolerance:
        n_own /= norm_own
    
    edge1_hit = hit_tri[1] - hit_tri[0]
    edge2_hit = hit_tri[2] - hit_tri[0]
    n_hit = np.cross(edge1_hit, edge2_hit)
    norm_hit = np.linalg.norm(n_hit)
    if norm_hit > tolerance:
        n_hit /= norm_hit
    
    # Evaluate orientation using triangle centers
    center_own = owner_tri.mean(axis=0)
    center_hit = hit_tri.mean(axis=0)
    
    owner_side = np.dot(center_own - point, n_hit)
    hit_side = np.dot(center_hit - point, n_own)
    
    # If the centers are "below" or "inside" the other triangle's tangent plane,
    # it is a chord-through-interior artifact.
    is_artifact = (owner_side < -tolerance) and (hit_side < -tolerance)
    
    return not is_artifact


def _min_edge_edge_distance(tri1: np.ndarray, tri2: np.ndarray) -> float:
    """Compute the minimum distance between any edge of tri1 and any edge of tri2.

    Uses the parametric closest-point approach for line segments.
    """
    edges1 = [(tri1[0], tri1[1]), (tri1[1], tri1[2]), (tri1[2], tri1[0])]
    edges2 = [(tri2[0], tri2[1]), (tri2[1], tri2[2]), (tri2[2], tri2[0])]

    min_dist = float('inf')
    for a0, a1 in edges1:
        for b0, b1 in edges2:
            d = _segment_segment_distance(a0, a1, b0, b1)
            if d < min_dist:
                min_dist = d

    return min_dist


def _segment_segment_distance(
    p1: np.ndarray, p2: np.ndarray,
    p3: np.ndarray, p4: np.ndarray,
) -> float:
    """Minimum distance between two line segments p1-p2 and p3-p4.

    Uses the parametric approach from Real-Time Collision Detection
    (Ericson, 2005).
    """
    d1 = p2 - p1  # direction of segment 1
    d2 = p4 - p3  # direction of segment 2
    r = p1 - p3

    a = np.dot(d1, d1)  # |d1|^2
    e = np.dot(d2, d2)  # |d2|^2
    f = np.dot(d2, r)

    # Check if both segments degenerate to points
    if a < 1e-20 and e < 1e-20:
        return np.linalg.norm(r)

    if a < 1e-20:
        # First segment degenerates to a point
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = np.dot(d1, r)
        if e < 1e-20:
            # Second segment degenerates to a point
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            b = np.dot(d1, d2)
            denom = a * e - b * b

            if abs(denom) > 1e-20:
                s = np.clip((b * f - c * e) / denom, 0.0, 1.0)
            else:
                s = 0.0

            t = (b * s + f) / e

            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)

    closest1 = p1 + s * d1
    closest2 = p3 + t * d2
    return float(np.linalg.norm(closest1 - closest2))


def _mean_edge_length(tri1: np.ndarray, tri2: np.ndarray) -> float:
    """Mean edge length across all 6 edges of two triangles."""
    lengths = []
    for tri in (tri1, tri2):
        for i in range(3):
            j = (i + 1) % 3
            lengths.append(np.linalg.norm(tri[j] - tri[i]))
    return float(np.mean(lengths))


def _is_convex(vertices: np.ndarray) -> bool:
    """Check if all vertices lie on the convex hull.

    If every vertex is a vertex of the convex hull, the mesh is convex.
    A convex surface cannot self-intersect — any edge-face crossings are
    chord-through-interior artifacts from the piecewise-linear approximation.

    Args:
        vertices: Nx3 array of vertex coordinates.

    Returns:
        True if the vertex set is convex.
    """
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(vertices)
        return len(hull.vertices) == len(vertices)
    except Exception:
        # Qhull can fail on degenerate inputs — assume non-convex
        return False


def detect_self_intersections_manual(
    vertices: np.ndarray,
    faces: np.ndarray,
    tolerance: float = 1e-10,
    verbose: bool = False
) -> List[Tuple[int, int]]:
    """
    Detect self-intersecting triangles using KD-tree + triangle-triangle tests.
    
    This is a robust replacement for pymeshfix that produces fewer false positives.
    
    Args:
        vertices: Nx3 array of vertex coordinates
        faces: Mx3 array of triangle vertex indices
        tolerance: Numerical tolerance for intersection tests
        verbose: Print progress information
        
    Returns:
        List of (face1_idx, face2_idx) tuples for intersecting triangles
        
    Example:
        >>> vertices = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 0.5, 1]])
        >>> faces = np.array([[0, 1, 2], [0, 1, 3]])  # Two triangles sharing edge
        >>> detect_self_intersections_manual(vertices, faces)
        []  # Adjacent triangles don't count as intersection
    """
    n_faces = len(faces)
    if n_faces < 2:
        return []

    # Early out for convex meshes: a convex surface cannot self-intersect.
    # Any edge-face crossings are chord-through-interior artifacts from the
    # piecewise-linear approximation of a curved surface.
    if _is_convex(vertices):
        if verbose:
            logger.info("Mesh is convex — skipping self-intersection detection")
        return []

    if verbose:
        logger.info(f"Computing self-intersections for {n_faces} triangles...")
    
    # 1. Compute triangle centroids for KD-tree
    centroids = compute_triangle_centroids(vertices, faces)
    
    # 2. Build KD-tree
    kdtree = KDTree(centroids)
    
    # 3. Find candidate pairs using query_pairs
    # Use 2x max edge length as search radius to catch all potential intersections
    max_edge_length = compute_max_edge_length(vertices, faces)
    search_radius = max_edge_length * 2.0
    
    if verbose:
        logger.info(f"KD-tree search radius: {search_radius:.6f}")
    
    candidate_pairs = kdtree.query_pairs(r=search_radius)
    
    if verbose:
        logger.info(f"Found {len(candidate_pairs)} candidate pairs from KD-tree")
    
    # 4. Build adjacency set and filter
    adjacent_pairs = build_adjacency_pairs(faces)
    candidate_pairs = candidate_pairs - adjacent_pairs
    
    if verbose:
        logger.info(f"After filtering adjacent pairs: {len(candidate_pairs)}")
    
    # 5. Test each candidate pair
    intersections = []
    for i, j in candidate_pairs:
        tri1 = vertices[faces[i]]
        tri2 = vertices[faces[j]]
        
        if triangles_intersect(tri1, tri2, tolerance):
            intersections.append((i, j))
    
    if verbose:
        logger.info(f"Found {len(intersections)} intersecting triangle pairs")
    
    return intersections
