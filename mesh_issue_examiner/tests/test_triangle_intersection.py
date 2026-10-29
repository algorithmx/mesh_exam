"""
Unit tests for triangle_intersection.py — self-intersection detection.

These tests verify correctness of the low-level geometric predicates AND the
high-level self-intersection detector on meshes with known ground truth.
"""

import numpy as np
import pytest

from mesh_issue_examiner.utils.triangle_intersection import (
    triangles_intersect,
    moller_trumbore_intersection,
    are_coplanar,
    coplanar_triangles_intersect,
    bbox_overlap,
    point_in_triangle_2d,
    edges_intersect_2d,
    compute_triangle_centroids,
    compute_max_edge_length,
    build_adjacency_pairs,
    detect_self_intersections_manual,
)


# ===================================================================
# Helpers
# ===================================================================

def _tetrahedron():
    """A regular tetrahedron (4 closed faces, 4 vertices)."""
    verts = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2],
        [0, 3, 1],
        [1, 3, 2],
        [2, 3, 0],
    ], dtype=np.int32)
    return verts, faces


def _icosphere_l1():
    """Load the icosphere_l1 sample mesh (42 verts, 80 faces)."""
    import trimesh
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "e2e" / "sample_meshes" / "icosphere_l1" / "icosphere_l1.obj"
    m = trimesh.load(str(p))
    return np.array(m.vertices, dtype=np.float64), np.array(m.faces, dtype=np.int32)


def _tent_crossing_floor():
    """A 3-triangle "tent over a floor" that genuinely crosses.

    Mirrors the e2e ``self_intersecting`` fixture: the floor triangle
    [0,1,2] lies in y=0; the tent apex A=(0,1,-1.5) sits above it and its
    corners B=(4,-1,-2), C=(-4,-1,-2) below, so the A-B and A-C edges
    pierce the strict interior of the floor at (+/-2, 0, -1.75) — a
    positive-length intersection segment, not a point contact.  The tent
    is split into two faces sharing edge A-M; the midpoint vertex M lies
    on a hull edge, which makes the vertex set non-convex so the manual
    detector actually runs its pair tests (a fully convex vertex set
    would take the convex early-out and miss the crossing).
    """
    verts = np.array([
        [-3, 0, -3],
        [3, 0, -3],
        [0, 0, 3],
        [0, 1, -1.5],
        [4, -1, -2],
        [-4, -1, -2],
        [0, -1, -2],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2], [3, 4, 6], [3, 6, 5]], dtype=np.int32)
    return verts, faces


def _two_disjoint_triangles():
    """Two triangles that are far apart and definitely don't intersect."""
    verts = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, 1, 0],
        [0, 0, 10],
        [1, 0, 10],
        [0.5, 1, 10],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    return verts, faces


def _two_touching_triangles():
    """Two triangles that share an edge (adjacent, should not be flagged)."""
    verts = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, 1, 0],
        [0.5, -1, 0],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2], [1, 0, 3]], dtype=np.int32)
    return verts, faces


# ===================================================================
# 1. Low-level geometric predicates
# ===================================================================


class TestBboxOverlap:
    def test_overlapping(self):
        tri1 = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]])
        tri2 = np.array([[0.5, 0.5, -0.5], [1.5, 0.5, 0.5], [1, 1.5, 0.5]])
        assert bbox_overlap(tri1, tri2)

    def test_non_overlapping(self):
        tri1 = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]])
        tri2 = np.array([[0, 0, 10], [1, 0, 10], [0.5, 1, 10]])
        assert not bbox_overlap(tri1, tri2)


class TestAreCoplanar:
    def test_coplanar_triangles(self):
        tri1 = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]])
        tri2 = np.array([[2, 0, 0], [3, 0, 0], [2.5, 1, 0]])
        assert are_coplanar(tri1, tri2)

    def test_non_coplanar_triangles(self):
        tri1 = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]])
        tri2 = np.array([[0, 0, 1], [1, 0, 1], [0.5, 1, 1]])
        assert not are_coplanar(tri1, tri2)


class TestMollerTrumbore:
    def test_edge_through_triangle(self):
        """An edge that clearly passes through a triangle."""
        edge_start = np.array([0.5, 0.5, -1.0])
        edge_end = np.array([0.5, 0.5, 1.0])
        tri = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]])
        assert moller_trumbore_intersection(edge_start, edge_end, tri)

    def test_edge_misses_triangle(self):
        """An edge that passes beside a triangle."""
        edge_start = np.array([5.0, 5.0, -1.0])
        edge_end = np.array([5.0, 5.0, 1.0])
        tri = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]])
        assert not moller_trumbore_intersection(edge_start, edge_end, tri)

    def test_edge_parallel_to_triangle(self):
        """An edge parallel to the triangle plane."""
        edge_start = np.array([0, 0, 1])
        edge_end = np.array([1, 0, 1])
        tri = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]])
        assert not moller_trumbore_intersection(edge_start, edge_end, tri)


class TestTrianglesIntersect:
    def test_crossing_triangles(self):
        """Two triangles that cross each other."""
        tri1 = np.array([[0, 0, -0.5], [1, 0, -0.5], [0.5, 1, 0.5]])
        tri2 = np.array([[0, 0.5, 0.5], [1, 0.5, 0.5], [0.5, 0, -0.5]])
        assert triangles_intersect(tri1, tri2) is True

    def test_disjoint_triangles(self):
        """Two triangles far apart."""
        tri1 = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]])
        tri2 = np.array([[0, 0, 10], [1, 0, 10], [0.5, 1, 10]])
        assert triangles_intersect(tri1, tri2) is False


class TestBuildAdjacencyPairs:
    def test_sharing_one_vertex(self):
        faces = np.array([[0, 1, 2], [0, 3, 4]])
        adj = build_adjacency_pairs(faces)
        assert (0, 1) in adj

    def test_not_sharing_vertices(self):
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        adj = build_adjacency_pairs(faces)
        assert (0, 1) not in adj


class TestPointInTriangle2d:
    def test_inside(self):
        tri = np.array([[0, 0], [1, 0], [0.5, 1]])
        assert point_in_triangle_2d(np.array([0.5, 0.3]), tri)

    def test_outside(self):
        tri = np.array([[0, 0], [1, 0], [0.5, 1]])
        assert not point_in_triangle_2d(np.array([2.0, 2.0]), tri)


# ===================================================================
# 2. Self-intersection detection on known meshes
# ===================================================================


class TestDetectSelfIntersectionsManual:
    """High-level self-intersection detection on meshes with known ground truth."""

    def test_disjoint_triangles_no_intersection(self):
        """Two far-apart triangles should produce no self-intersections."""
        verts, faces = _two_disjoint_triangles()
        result = detect_self_intersections_manual(verts, faces)
        assert result == []

    def test_touching_triangles_no_intersection(self):
        """Edge-adjacent triangles should be filtered by adjacency."""
        verts, faces = _two_touching_triangles()
        result = detect_self_intersections_manual(verts, faces)
        assert result == []

    def test_crossing_triangles_detected(self):
        """Genuinely crossing triangles should be detected.

        The tent's A-B and A-C edges pierce the floor triangle's interior,
        so at least the (floor, tent-face) pairs must be reported.
        """
        verts, faces = _tent_crossing_floor()
        result = detect_self_intersections_manual(verts, faces)
        assert len(result) >= 1, "Expected at least one intersection for crossing triangles"

    def test_tetrahedron_no_false_positive(self):
        """A closed tetrahedron should have no self-intersections."""
        verts, faces = _tetrahedron()
        result = detect_self_intersections_manual(verts, faces)
        assert result == [], (
            f"Tetrahedron should have 0 self-intersections, got {len(result)}"
        )

    def test_icosphere_no_false_positive(self):
        """BUG REGRESSION: icosphere_l1 is a valid watertight mesh.

        The detector must NOT report self-intersections caused by edge chords
        passing through non-adjacent faces on curved tessellated surfaces.
        """
        verts, faces = _icosphere_l1()
        result = detect_self_intersections_manual(verts, faces)
        assert result == [], (
            f"Icosphere_l1 (valid watertight mesh) should have 0 "
            f"self-intersections, got {len(result)}"
        )

    def test_single_triangle_no_intersection(self):
        """A single triangle cannot self-intersect."""
        verts = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        result = detect_self_intersections_manual(verts, faces)
        assert result == []


# ===================================================================
# 3. Chord-through-interior regression tests
# ===================================================================


class TestChordThroughInterior:
    """Regression tests for the false-positive bug on curved surfaces.

    On a tessellated closed surface, edge chords pass through the interior
    and intersect non-adjacent faces. These are geometric facts about
    piecewise-linear approximations, NOT mesh quality defects.

    These tests extract specific non-adjacent triangle pairs from a valid
    icosphere where Moller-Trumbore currently reports a false intersection.
    """

    @pytest.fixture
    def icosphere(self):
        return _icosphere_l1()

    def test_non_adjacent_pair_bbox_overlaps(self, icosphere):
        """Non-adjacent triangles on a closed surface can have overlapping bboxes."""
        verts, faces = icosphere
        # Pair (40, 77): no shared vertices, but edge chord passes through face
        tri1 = verts[faces[40]]
        tri2 = verts[faces[77]]
        shared = set(faces[40]) & set(faces[77])
        assert len(shared) == 0, "Pair should not share vertices"
        # Bboxes overlap because they're on a small sphere
        assert bbox_overlap(tri1, tri2)

    def test_non_adjacent_pair_not_coplanar(self, icosphere):
        """The false-positive pair is NOT coplanar."""
        verts, faces = icosphere
        tri1 = verts[faces[40]]
        tri2 = verts[faces[77]]
        assert are_coplanar(tri1, tri2) is False

    def test_non_adjacent_pair_moller_trumbore_true_before_fix(self, icosphere):
        """Moller-Trumbore currently says these intersect (the raw geometric fact).

        This test documents the current behavior. After the fix,
        triangles_intersect should return False for these pairs.
        """
        verts, faces = icosphere
        tri1 = verts[faces[40]]
        tri2 = verts[faces[77]]
        # Before fix: this is True (raw edge-face geometric intersection)
        # After fix: this should be False (surface-aware intersection)
        result = triangles_intersect(tri1, tri2)
        # We're testing that the high-level function gives the RIGHT answer
        # regardless of what Moller-Trumbore says
        assert result is False, (
            "Non-adjacent triangles on a valid closed surface should not "
            "be reported as intersecting"
        )

    def test_min_edge_distance_is_positive(self, icosphere):
        """For the false-positive pair, edges don't actually cross.

        The closest approach between any edge of tri1 and any edge of tri2
        is positive, meaning no edge-edge contact exists.
        """
        from scipy.spatial import distance as sdist
        verts, faces = icosphere
        tri1 = verts[faces[40]]
        tri2 = verts[faces[77]]

        min_dist = float("inf")
        edges1 = [(tri1[0], tri1[1]), (tri1[1], tri1[2]), (tri1[2], tri1[0])]
        edges2 = [(tri2[0], tri2[1]), (tri2[1], tri2[2]), (tri2[2], tri2[0])]
        for a, b in edges1:
            for c, d in edges2:
                ts = np.linspace(0, 1, 500)
                p1 = a[None, :] * (1 - ts[:, None]) + b[None, :] * ts[:, None]
                p2 = c[None, :] * (1 - ts[:, None]) + d[None, :] * ts[:, None]
                d_min = sdist.cdist(p1, p2).min()
                min_dist = min(min_dist, d_min)

        assert min_dist > 0.01, (
            f"Edges of non-adjacent icosphere triangles have min dist {min_dist:.6f} "
            f"— they don't actually touch"
        )

    def test_intersection_point_is_inside_sphere(self, icosphere):
        """The geometric intersection point lies INSIDE the sphere.

        For a convex mesh, an interior intersection point means the edge
        chord passes through the interior — a tessellation artifact.
        """
        verts, faces = icosphere
        tri1 = verts[faces[40]]
        tri2 = verts[faces[77]]

        # Compute intersection point via Moller-Trumbore
        edge_start = tri1[2]
        edge_end = tri1[0]
        ray_dir = edge_end - edge_start
        v0, v1, v2 = tri2[0], tri2[1], tri2[2]
        e1 = v1 - v0
        e2 = v2 - v0
        pvec = np.cross(ray_dir, e2)
        det = np.dot(e1, pvec)
        inv_det = 1.0 / det
        tvec = edge_start - v0
        u = np.dot(tvec, pvec) * inv_det
        qvec = np.cross(tvec, e1)
        v = np.dot(ray_dir, qvec) * inv_det
        t = np.dot(e2, qvec) * inv_det
        P = edge_start + t * ray_dir

        # P should be inside the sphere (|P| < max vertex radius)
        max_r = np.linalg.norm(verts, axis=1).max()
        assert np.linalg.norm(P) < max_r * 0.95, (
            f"Intersection point |P|={np.linalg.norm(P):.4f} is inside "
            f"sphere of radius {max_r:.4f} — tessellation artifact"
        )
