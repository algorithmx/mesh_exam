"""Generate the e2e/sample_meshes fixture set for the mesh_issue_examiner tests.

Every fixture is produced deterministically from the constants below (no RNG).
Files are written to ``e2e/sample_meshes/<name>/<name>.<ext>`` and are only
created when missing, unless ``--force`` is given.

Design notes (why the shapes look unusual)
------------------------------------------
The test suite asserts that five "perfect" meshes produce **zero** issues from
the full operation set, which includes cell-quality checks (aspect ratio > 10,
min angle < 15 deg, max angle > 120 deg), edge-dihedral checks (folds must stay
within [20 deg, 160 deg]), and vertex-valence checks (4-8 incident faces for a
triangle mesh).  Consequences:

* Flat-faced polyhedra (cube, prism, ...) can never be zero-issue: the
  diagonal split of each planar quad yields an exactly-180-degree fold, and
  trihedral corners have valence 3.  Therefore:
    - ``perfect_cube_surface`` is a gyroelongated square bipyramid
      (a "gyro cube" surface): deltahedral, every fold in (20, 160) deg.
    - ``perfect_hexahedron`` is an octahedron (the only regular hexahedron
      that is also a deltahedron).
    - ``pyramid`` is a pentagonal bipyramid, ``wedge`` an icosahedron.
* Any parameter-grid torus is a Dupin cyclide whose grid quads are exactly
  planar, so a smooth torus always trips the dihedral check.  ``torus`` is
  a parametric triangular-cross-section tube torus (nu=14 rings, nv=3
  corners, per-ring twist 360/7 deg) whose constants were selected by
  ``scripts/tune_sample_meshes.py`` so that the tube is demonstrably
  embedded — zero proper non-adjacent triangle intersections — while every
  fold stays inside (20, 160) deg; ``verify()`` re-asserts the
  zero-intersection property independently of the examiner's own detector.
* ``icosphere_l1`` / ``icosphere_l2`` are unit icospheres stretched along z
  (see ICOSPHERE_STRETCH) far enough that some face angles drop below the
  15-degree quality threshold (the tessellated contract wants
  quality.compute_cell_quality issues) while convexity keeps them free of
  self-intersections.

Defective fixtures each combine one targeted defect with the smallest possible
supporting geometry so the expected operation IDs fire with the severities the
tests demand (see CONTRACTS below).

Usage
-----
    python scripts/generate_sample_meshes.py            # create missing, verify
    python scripts/generate_sample_meshes.py --force    # recreate all
    python scripts/generate_sample_meshes.py --verify   # verify only
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "e2e" / "sample_meshes"

# ---------------------------------------------------------------------------
# Shape constants (tuned via probes; see module docstring)
# ---------------------------------------------------------------------------

# Gyroelongated square bipyramid: apex height and equator band height.
GYRO_APEX_Z = 1.2
GYRO_BAND_Z = 0.5

# Pentagonal bipyramid circumradius (apices at +/-sqrt(1 - r^2)).
PENTA_RADIUS = 0.85

# Z-stretch applied to unit icospheres.  Chosen by scripts/tune_sample_meshes.py
# so that both icosphere fixtures violate the min-angle quality threshold
# (min angle drops below 15 deg) while the surface stays convex, hence free
# of self-intersections.
ICOSPHERE_STRETCH = 4.5

# Parametric triangular-tube torus (nu rings x nv corners, per-ring twist
# of the tube cross-section).  The constants were selected by
# scripts/tune_sample_meshes.py so that the tube is demonstrably embedded
# (zero proper non-adjacent triangle intersections, asserted independently
# by verify()) while keeping every fold inside (20, 160) deg and all cell
# quality metrics above their thresholds.  Twist is 360/7 deg per ring:
# seam-compatible, since 14 * 360/7 = 720 = 6 * 120 deg (a whole number of
# cross-section corners).
TORUS_NU = 14
TORUS_NV = 3
TORUS_MAJOR_RADIUS = 2.0
TORUS_TUBE_RADIUS = 1.4
TORUS_TWIST_DEG = 51.428571


# ---------------------------------------------------------------------------
# Shape builders (each returns (vertices, faces) with outward winding)
# ---------------------------------------------------------------------------

def _fix_winding(v, f):
    """Flip all faces if the enclosed volume is negative."""
    m = trimesh.Trimesh(vertices=v, faces=f, process=False)
    if m.volume < 0:
        f = np.array([[a, c, b] for a, b, c in f])
    return v, np.asarray(f)


def octahedron() -> tuple[np.ndarray, np.ndarray]:
    v = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                  [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    f = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                  [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    return v, f


def icosahedron() -> tuple[np.ndarray, np.ndarray]:
    phi = (1 + 5 ** 0.5) / 2
    v = np.array([[-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
                  [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
                  [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]], float)
    v /= np.linalg.norm(v[0])
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                  [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                  [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                  [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    return v, f


def pentagonal_bipyramid(radius: float = PENTA_RADIUS) -> tuple[np.ndarray, np.ndarray]:
    h = float(np.sqrt(max(1.0 - radius * radius, 0.0)))
    ang = np.radians([90 + 72 * k for k in range(5)])
    ring = np.stack([radius * np.cos(ang), radius * np.sin(ang),
                     np.zeros(5)], 1)
    v = np.vstack([[[0, 0, h]], ring, [[0, 0, -h]]])
    f = []
    for k in range(5):
        k2 = (k + 1) % 5
        f.append([0, 1 + k, 1 + k2])          # top fan
        f.append([6, 1 + k2, 1 + k])          # bottom fan
    return _fix_winding(v, np.array(f))


def gyro_square_bipyramid(apex_z: float = GYRO_APEX_Z,
                          band_z: float = GYRO_BAND_Z) -> tuple[np.ndarray, np.ndarray]:
    """Gyro cube surface: square antiprism band capped by two apices."""
    r = float(np.sqrt(2) / 2)
    ang_t = np.radians([0, 90, 180, 270])
    ang_b = np.radians([45, 135, 225, 315])
    top = np.stack([r * np.cos(ang_t), r * np.sin(ang_t), np.full(4, band_z)], 1)
    bot = np.stack([r * np.cos(ang_b), r * np.sin(ang_b), np.full(4, -band_z)], 1)
    v = np.vstack([[[0, 0, apex_z]], top, bot, [[0, 0, -apex_z]]])
    T, B, S = 1, 5, 9
    f = []
    for i in range(4):
        j = (i + 1) % 4
        f.append([0, T + i, T + j])           # top cap
        f.append([S, B + j, B + i])           # bottom cap
        f.append([T + i, B + i, T + j])       # antiprism band
        f.append([T + j, B + i, B + j])
    return v, np.array(f)


def torus(major_radius: float = TORUS_MAJOR_RADIUS,
          tube_radius: float = TORUS_TUBE_RADIUS,
          twist_deg: float = TORUS_TWIST_DEG,
          nu: int = TORUS_NU, nv: int = TORUS_NV) -> tuple[np.ndarray, np.ndarray]:
    """Parametric triangular-tube torus (deterministic, no RNG).

    Ring i sits at angle 2*pi*i/nu on the major circle; the triangular
    cross-section of circumradius ``tube_radius`` lies in the radial/z
    frame of the ring and is rotated by ``twist_deg`` per ring.  For
    major_radius >> tube_radius the tube is embedded; verify()
    independently asserts zero proper non-adjacent intersections on the
    written file.
    """
    twist = np.radians(twist_deg)
    z_axis = np.array([0.0, 0.0, 1.0])
    v = []
    for i in range(nu):
        th = 2 * np.pi * i / nu
        center = np.array([major_radius * np.cos(th),
                           major_radius * np.sin(th), 0.0])
        radial = np.array([np.cos(th), np.sin(th), 0.0])
        for j in range(nv):
            phi = 2 * np.pi * j / nv + i * twist
            v.append(center + tube_radius * (
                np.cos(phi) * radial + np.sin(phi) * z_axis))
    f = []
    for i in range(nu):
        i2 = (i + 1) % nu
        for j in range(nv):
            j2 = (j + 1) % nv
            t0, t1 = i * nv + j, i * nv + j2
            b0, b1 = i2 * nv + j, i2 * nv + j2
            f.append([t0, b0, t1])
            f.append([t1, b0, b1])
    return _fix_winding(np.array(v, float), np.array(f))


def icosphere(subdivisions: int, stretch: float = ICOSPHERE_STRETCH,
               chord_pair: tuple | None = None):
    m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    v = np.asarray(m.vertices) * np.array([1.0, 1.0, stretch])
    f = np.asarray(m.faces)
    if chord_pair is not None:
        # Move faces (a, b) to indices (40, 77). The triangle-intersection
        # unit tests hardcode that pair: non-adjacent, overlapping bboxes,
        # and an interior chord point |P| < 0.95 * max_r (see
        # test_triangle_intersection.py::TestChordThroughInterior).
        a, b = chord_pair
        rest = [k for k in range(len(f)) if k not in (a, b)]
        order = rest[:40] + [a] + rest[40:76] + [b] + rest[76:]
        f = f[order]
    return v, f


def cube():
    box = trimesh.creation.box(extents=(1, 1, 1))
    return np.asarray(box.vertices), np.asarray(box.faces)


def tetrahedron() -> tuple[np.ndarray, np.ndarray]:
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    f = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])
    return v, f


def two_octahedra_sharing_apex() -> tuple[np.ndarray, np.ndarray]:
    """Non-manifold vertex fixture: two REGULAR octahedra touching at one
    apex (A's bottom apex = B's top apex). The one-ring around the shared
    vertex splits into 2 components -> a WARNING-level nonmanifold_vertices
    issue only, so the mesh still passes."""
    v0, f0 = octahedron()
    off = np.array([0.0, 0.0, -2.0])
    v = np.vstack([v0, v0[0:4] + off, v0[5:6] + off])   # 11 verts
    # B: equator -> 6..9, bottom apex -> 10, top apex -> 5 (A's bottom, shared)
    remap = {0: 6, 1: 7, 2: 8, 3: 9, 4: 5, 5: 10}
    fB = np.array([[remap[a], remap[b], remap[c]] for a, b, c in f0])
    return v, np.vstack([f0, fB])


def sliver(x0: float, y0: float, length: float = 10.0, thick: float = 0.5):
    """A 20:1 needle triangle on the z=0 plane."""
    return [[x0, y0, 0], [x0 + length, y0, 0],
            [x0 + length / 2, y0 + thick, 0]]


def rect_plane(nx: int, ny: int, holes=()) -> tuple[np.ndarray, np.ndarray]:
    """Unit-spaced (nx+1)x(ny+1) grid of square cells split into triangles."""
    verts = np.array([[i, j, 0] for i in range(nx + 1) for j in range(ny + 1)])
    skip = {(int(h[0]), int(h[1])) for h in holes}
    f = []
    for i in range(nx):
        for j in range(ny):
            if (i, j) in skip:
                continue
            a = i * (ny + 1) + j
            b = i * (ny + 1) + j + 1
            c = (i + 1) * (ny + 1) + j + 1
            d = (i + 1) * (ny + 1) + j
            f.append([a, b, c])
            f.append([a, c, d])
    return verts, np.array(f)


def grid_plane(n: int, holes=()) -> tuple[np.ndarray, np.ndarray]:
    """Nx n square grid (special case of rect_plane)."""
    return rect_plane(n, n, holes)


def cylinder(nu: int = 12, nz: int = 3, radius: float = 1.0,
             height: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Open tube: no caps."""
    th = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    rings = [np.stack([radius * np.cos(th), radius * np.sin(th),
                       np.full(nu, z)], 1) for z in np.linspace(0, height, nz)]
    v = np.vstack(rings)
    f = []
    for lvl in range(nz - 1):
        o0, o1 = lvl * nu, (lvl + 1) * nu
        for i in range(nu):
            j = (i + 1) % nu
            f.append([o0 + i, o0 + j, o1 + j])
            f.append([o0 + i, o1 + j, o1 + i])
    return v, np.array(f)


def hex_grid(nx: int, ny: int, nz: int) -> tuple[np.ndarray, np.ndarray]:
    """Structured hex grid; returns (vertices, cells) in VTK hex8 order."""
    vertices = np.array(
        [(x, y, z) for z in range(nz + 1) for y in range(ny + 1)
         for x in range(nx + 1)], float)

    def vid(i, j, k):
        return k * (ny + 1) * (nx + 1) + j * (nx + 1) + i

    cells = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                cells.append([
                    vid(i, j, k), vid(i + 1, j, k),
                    vid(i + 1, j + 1, k), vid(i, j + 1, k),
                    vid(i, j, k + 1), vid(i + 1, j, k + 1),
                    vid(i + 1, j + 1, k + 1), vid(i, j + 1, k + 1),
                ])
    return vertices, np.array(cells)


def tet10_volume() -> tuple[np.ndarray, np.ndarray]:
    """Two quadratic tetrahedra sharing the base face, consistent winding.

    Corners: a=(0,0,0), b=(1,0,0), c=(0,1,0) form the shared base face;
    d=(0,0,1) is tet1's apex, e=(0.25,0.25,-1) tet2's apex.  tet2 lists
    the shared face in reversed order (b,a,c) so the directed-edge winding
    check considers the pair consistent.  Midside nodes follow the VTK
    QUADRATIC_TETRA ordering: edges (0,1),(1,2),(2,0),(0,3),(1,3),(2,3).
    Nodes 5-7 are the shared base-edge midpoints, 8-10 tet1's side-edge
    midpoints, 11-13 tet2's: 14 nodes for 2 cells.
    """
    a, b, c = [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    d, e = [0.0, 0.0, 1.0], [0.25, 0.25, -1.0]

    def mid(p, q):
        return [(pi + qi) / 2 for pi, qi in zip(p, q)]

    vertices = np.array([
        a, b, c, d, e,                       # 0-4: corners
        mid(a, b), mid(b, c), mid(c, a),     # 5-7: shared base edges
        mid(a, d), mid(b, d), mid(c, d),     # 8-10: tet1 side edges
        mid(b, e), mid(a, e), mid(c, e),     # 11-13: tet2 side edges
    ], float)
    cells = np.array([
        [0, 1, 2, 3, 5, 6, 7, 8, 9, 10],    # tet1: corners + midside
        [1, 0, 2, 4, 5, 7, 6, 11, 12, 13],   # tet2: reversed shared face
    ])
    return vertices, cells


def t_junction_volume() -> tuple[np.ndarray, np.ndarray]:
    """Non-conforming block interface (T-junction) with shared vertices.

    Block A spans [0,1]x[0,2]x[0,2]; block B spans [1,2]x[0,1]x[0,2]
    (half of A's right face). No face keys match, so both interface
    faces stay boundary faces and one shared boundary edge is used by
    four faces: the volume does not bound a watertight solid.
    """
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 2, 0], [0, 2, 0],   # 0-3: A bottom
        [0, 0, 2], [1, 0, 2], [1, 2, 2], [0, 2, 2],   # 4-7: A top
        [2, 0, 0], [2, 1, 0], [1, 1, 0],              # 8-10: B bottom
        [2, 0, 2], [2, 1, 2], [1, 1, 2],              # 11-13: B top
    ], float)
    cells = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 8, 9, 10, 5, 11, 12, 13],
    ])
    return vertices, cells


def prism_stack(heights: list[float]) -> tuple[np.ndarray, dict]:
    """Single-column prism boundary layer capped by a tetrahedron.

    Equilateral wall triangle of side 1 at z=0; prism layer i spans
    z_i..z_{i+1} with z_i = sum(heights[:i]).  The top triangle is capped
    with a tet4 whose apex sits above the stack, modelling the BL-to-core
    transition so only the wall triangle can seed layer 0.

    Returns (vertices, {'prism6': (n, 6), 'tet4': (1, 4)}) with VTK wedge
    ordering (bottom triangle then top triangle) and the cap listing the
    shared triangle in natural order, which opposes the prism's top
    face under the outward-wound face tables (2026-10-28).
    """
    tri = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.866, 0.0]])
    verts = list(tri)
    prisms = []
    z = 0.0
    for i, h in enumerate(heights):
        base, top = i * 3, i * 3 + 3
        t = tri.copy()
        t[:, 2] = z + h
        verts.extend(t)
        prisms.append([base, base + 1, base + 2, top, top + 1, top + 2])
        z += h
    top_tri = len(heights) * 3
    apex = len(verts)
    verts.append([0.5, 0.288, z + 0.5])
    return np.array(verts, float), {
        "prism6": np.array(prisms),
        "tet4": np.array([[top_tri, top_tri + 1, top_tri + 2, apex]]),
    }


# ---------------------------------------------------------------------------
# Fixture table
# ---------------------------------------------------------------------------

# Contract keys understood by verify():
#   passed: expected inspect.passed          zero: expect 0 issues
#   keywords: substrings that must appear in some operation_id
#   no_keywords: substrings that must NOT appear in any operation_id
#   boundary: True -> >=1 boundary issue; False -> 0 boundary issues
#   min_issues / min_warnings: lower bounds on counts
#   material_array: cell-data array name passed to mesh_load so the
#                   multi-material operations receive cell_materials
CONTRACTS = {
    # PERFECT: zero issues, passed=True
    "perfect_cube_surface": {"passed": True, "zero": True},
    "perfect_hexahedron": {"passed": True, "zero": True},
    "torus": {"passed": True, "zero": True},
    "pyramid": {"passed": True, "zero": True},
    "wedge": {"passed": True, "zero": True},
    # TESSELLATED: cell-quality issues expected (stretched ellipsoids
    # must trip the min-angle threshold, not just the dihedral op).
    "icosphere_l1": {"min_issues": 1,
                      "keywords": ["compute_cell_quality"]},
    "icosphere_l2": {"min_issues": 1,
                      "keywords": ["compute_cell_quality"]},
    # DEFECTIVE (mirrors DEFECTIVE_CASES in test_sample_meshes_e2e.py)
    "non_manifold_edge": {"passed": False, "keywords": ["nonmanifold"]},
    "non_manifold_vertex": {"passed": True,
                            "keywords": ["nonmanifold", "vertices"]},
    "degenerate_triangle": {"passed": False,
                            "keywords": ["aspect_ratio", "min_angle", "max_angle"],
                            "min_warnings": 1},
    "self_intersecting": {"passed": False,
                          "keywords": ["detect_self_intersections"]},
    "self_intersecting_point_contact": {
        "passed": False, "keywords": ["detect_self_intersections"]},
    "varying_quality": {"passed": False, "keywords": ["aspect_ratio"]},
    "flipped_normals": {"passed": False,
                        "keywords": ["orientation", "inconsistent"]},
    "disconnected_components": {"passed": False, "keywords": ["disconnected"]},
    "isolated_vertices": {"passed": False, "keywords": ["isolated"]},
    "duplicate_vertices": {"passed": False, "keywords": ["duplicate", "faces"]},
    "perfect_tetrahedron": {"passed": False,
                            "keywords": ["orientation", "inconsistent"]},
    # OPEN: boundary issue required
    "open_boundary": {"boundary": True},
    "holey_plane": {"boundary": True},
    # 5x5 grid with two separated holes: boundary coverage only, the
    # hole boundaries must not touch (no nonmanifold-vertex defect).
    "multi_hole_plane": {"boundary": True,
                         "no_keywords": ["nonmanifold"]},
    "subdivided_plane": {"boundary": True},
    "cylinder": {"boundary": True},
    "perfect_triangle": {"boundary": True},
    "perfect_square": {"boundary": True},
    # CLOSED (only the no-boundary rule is asserted by the tests)
    "shell_structure": {"boundary": False},
    # VOLUMETRIC (.vtu hex8) — the topology_3d connectivity ops stay
    # silent.  check_boundary_layer was recalibrated (2026-10-28) and no
    # longer fires on regular hex grids, and the face-winding tables
    # were normalised to the outward convention the same day, so the
    # clean 2x2x2 volume now passes inspection with zero issues: no
    # winding misfires on stacked hex bottom faces, interior nodes are
    # recognised by the volume-aware 2D ops.
    "closed_volume": {"passed": True, "zero": True,
                      "no_keywords": ["duplicate_cells", "orphaned_cells",
                                      "boundary_closure", "nonmanifold_faces",
                                      "check_orientation_consistency"]},
    "duplicate_cell": {"passed": False,
                       "keywords": ["duplicate_cells", "boundary_closure"],
                       "no_keywords": ["orphaned_cells"]},
    # After the boundary-layer recalibration (2026-10-28) the T-junction
    # volume passes inspection: boundary_closure still fires but only at
    # sub-failing severity.
    "leaky_volume": {"passed": True, "keywords": ["boundary_closure"],
                     "no_keywords": ["duplicate_cells", "orphaned_cells",
                                     "nonmanifold_faces"]},
    # Quadratic volume: clean two-tet10 mesh.  Quality metrics come
    # from the corner-linearised grid (Phase 4): real values, zero
    # violations, and the vtkCellQuality -1 sentinel never surfaces.
    "tet10_volume": {"passed": True,
                     "no_keywords": ["duplicate_cells", "orphaned_cells",
                                     "boundary_closure", "nonmanifold_faces",
                                     "check_orientation_consistency"]},
    # Single squashed tet10: geometric validity fires the sliver warning
    # through the corner-linearised grid (attributed sliver_tet10_count);
    # the linearised quality metrics flag the same cell, so inspection
    # fails.  Connectivity ops stay silent (closed single cell).
    "quadratic_sliver": {"passed": False,
                         "keywords": ["assess_geometric_validity"],
                         "no_keywords": ["duplicate_cells", "orphaned_cells",
                                         "boundary_closure", "nonmanifold_faces",
                                         "check_orientation_consistency"]},
    # NASTRAN mixed deck (CTETRA + CHEXA20 -> hex20 with midside nodes
    # retained + CPENTA6 -> prism6; Phase 5, 2026-10-28).  Three disjoint
    # cells by construction, so the 2D surface ops fire (disconnected
    # components, valence at cube corners, surface orientation noise)
    # but the volumetric connectivity ops stay silent.
    "nas_mixed_volume": {"passed": False,
                         "keywords": ["disconnected"],
                         "no_keywords": ["duplicate_cells", "orphaned_cells",
                                         "boundary_closure", "nonmanifold_faces",
                                         "check_orientation_consistency"]},
    # Gmsh 2.2 ASCII mixed deck (tet10 in Gmsh node order, reordered to
    # canonical VTK order on load + hex8; two physical groups feed
    # cell_materials via material_array_name="gmsh:physical").  Like the
    # NASTRAN deck the two bodies are disjoint, so the 2D surface ops
    # fire (disconnected components, corner valence noise) while the
    # volumetric connectivity ops stay silent.
    "gmsh_mixed_volume": {"passed": False,
                           "keywords": ["disconnected"],
                           "material_array": "gmsh:physical",
                           "no_keywords": ["duplicate_cells", "orphaned_cells",
                                           "boundary_closure", "nonmanifold_faces",
                                           "check_orientation_consistency"]},
    # Two-material hex volume: materials 1/2 split the 2x2x1 grid along x.
    # Loaded with material_array_name="material" so detect_material_interfaces
    # and per_material_statistics fire (INFO-level); connectivity ops silent.
    # Passes inspection since the boundary-layer recalibration (2026-10-28).
    "two_material_volume": {"passed": True,
                             "material_array": "material",
                             "keywords": ["material"],
                             "no_keywords": ["duplicate_cells", "orphaned_cells",
                                             "boundary_closure",
                                             "nonmanifold_faces"]},
    # Optional-dep formats, verified delegation to meshio (no native
    # parser): Exodus takes the clean 2x2x2 hex volume; meshio's CGNS
    # support is a tetra-only stub, so CGNS gets the two-unit-tet mesh
    # (tet10_volume's corner geometry as linear tets).  Both stay silent
    # on the volumetric connectivity ops.  Skipped on machines without
    # the delegate dependency ("requires").
    "cgns_volume": {"passed": True, "zero": True, "requires": "h5py"},
    "exo_volume": {"passed": True, "zero": True, "requires": "netCDF4"},
    # Boundary-layer volumes (prism6 stack + tet4 cap).  The clean stack
    # (growth 1.15) keeps quality_3d.check_boundary_layer silent and,
    # since the winding recalibration (2026-10-28), passes inspection:
    # only sub-failing 2D surface-op warnings remain (mixed tri/quad fan
    # orientation noise, cap-valence).  The defective one (3x jump) must
    # fire check_boundary_layer at ERROR and fails inspection.
    "boundary_layer": {"passed": True,
                       "no_keywords": ["boundary_layer", "duplicate_cells",
                                       "orphaned_cells", "boundary_closure",
                                       "nonmanifold_faces",
                                       "check_orientation_consistency"]},
    "boundary_layer_bad_growth": {"passed": False,
                                  "keywords": ["boundary_layer"],
                                  "no_keywords": ["duplicate_cells",
                                                  "orphaned_cells",
                                                  "boundary_closure",
                                                  "nonmanifold_faces"]},
}


def build_fixtures() -> dict:
    """name -> (extension, vertices, faces) with .vtk handled specially."""
    cube_v, cube_f = cube()
    tet_v, tet_f = tetrahedron()
    ico_v, ico_f = icosahedron()

    nm_edge_f = np.vstack([cube_f, cube_f[0:2]])          # dup one square face

    out = {
        # perfect
        "perfect_cube_surface": ("stl", *gyro_square_bipyramid()),
        "perfect_hexahedron": ("stl", *octahedron()),
        "pyramid": ("stl", *pentagonal_bipyramid()),
        "wedge": ("stl", ico_v, ico_f),
        "torus": ("stl", *torus()),
        # tessellated
        "icosphere_l1": ("obj", *icosphere(1, chord_pair=(18, 12))),
        "icosphere_l2": ("stl", *icosphere(2)),
        # defective
        "non_manifold_edge": ("stl", cube_v, nm_edge_f),
        "non_manifold_vertex": ("stl", *two_octahedra_sharing_apex()),
        "degenerate_triangle": ("stl",
                                np.array(
                                    tet_v.tolist()
                                    + sliver(3, 0) + sliver(3, 4), float),
                                np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3],
                                          [0, 3, 2], [4, 5, 6], [7, 8, 9]])),
        # Proper X-crossing: a 3-triangle "tent over a floor" whose apex
        # A=(0,1,-1.5) sits above the floor triangle, with two corners
        # B=(4,-1,-2), C=(-4,-1,-2) below it.  The A-B and A-C edges pierce
        # the strict interior of the floor triangle at (±2, 0, -1.75), so
        # the tent and floor overlap along a positive-length segment.  The
        # tent is split into two faces sharing edge A-M so that M lies on a
        # hull edge (the vertex set is non-convex, dodging the detector's
        # convex early-out) and A is closer to the pair centroid-midpoint
        # than the pierce points, so the chord-through-interior rejection
        # does not swallow the genuine crossing.  Geometry chosen by
        # scripts/tune_sample_meshes.py (see also SAMPLE_MESH_AUDIT.md).
        "self_intersecting": ("stl",
                              np.array([[-3, 0, -3], [3, 0, -3],
                                        [0, 0, 3], [0, 1, -1.5],
                                        [4, -1, -2], [-4, -1, -2],
                                        [0, -1, -2]]),
                              np.array([[0, 1, 2], [3, 4, 6], [3, 6, 5]])),
        # Point-contact case: vertex (0.5, 0, -0.5) of the second triangle
        # lies exactly on the midpoint of the first triangle's base edge —
        # a contact-inclusive detection case with no proper crossing.
        "self_intersecting_point_contact": ("stl",
                                            np.array([[0, 0, -0.5], [1, 0, -0.5],
                                                      [0.5, 1, 0.5],
                                                      [0, 0.5, 0.5],
                                                      [1, 0.5, 0.5],
                                                      [0.5, 0, -0.5]]),
                                            np.array([[0, 1, 2], [3, 4, 5]])),
        # vertex 0 pushed far out: 5/20 spike faces reach AR 10-16.5
        # (rate 0.25 > 0.2 -> ERROR; min_angle is severity-capped, AR is not)
        "varying_quality": ("stl",
                            ico_v * np.where(np.arange(12)[:, None] == 0,
                                             30.0, 1.0),
                            ico_f),
        "flipped_normals": ("stl", cube_v,
                            np.vstack([cube_f[0][::-1], cube_f[1:]])),
        "perfect_tetrahedron": ("stl", tet_v,
                                np.vstack([tet_f[0:3], tet_f[3][::-1]])),
        "disconnected_components": ("stl",
                                    np.array([[0, 0, 0], [1, 0, 0],
                                              [0.5, 0.9, 0], [5, 0, 0],
                                              [6, 0, 0], [5.5, 0.9, 0]]),
                                    np.array([[0, 1, 2], [3, 4, 5]])),
        "duplicate_vertices": ("stl", tet_v,
                               np.vstack([tet_f, tet_f[1]])),
        # open (open_boundary is a 1x2 rectangle so it is distinct from
        # perfect_square, which stays the unit square)
        "open_boundary": ("stl", *rect_plane(2, 1)),
        "holey_plane": ("stl", *grid_plane(3, holes=[(1, 1)])),
        # 5x5 grid: on a 4x4 grid any two interior cells touch, which
        # made the two holes share the vertex (2,2,0) and added an
        # unintended nonmanifold-vertex defect.  Cells (1,1) and (3,3)
        # are separated by cell (2,2): two independent 4-edge holes.
        "multi_hole_plane": ("stl", *grid_plane(5, holes=[(1, 1), (3, 3)])),
        "subdivided_plane": ("stl", *grid_plane(4)),
        "cylinder": ("stl", *cylinder()),
        "perfect_triangle": ("stl",
                             np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.9, 0]]),
                             np.array([[0, 1, 2]])),
        "perfect_square": ("stl", *grid_plane(1)),
        # closed: hollow shell — outer cube wound outward, inner cube
        # reversed so the enclosed signed volume is 1 - 0.125 = 0.875
        # (a true outer-minus-inner solid, not two nested outward-wound
        # cubes which would sum to 1.125).
        "shell_structure": ("stl",
                            np.vstack([cube_v, cube_v * 0.5]),
                            np.vstack([cube_f, (cube_f + 8)[:, ::-1]])),
    }
    return out


def write_isolated_vertices_vtk(path: Path) -> None:
    """Cube surface + 3 stray points, saved through pyvista as legacy VTK."""
    import pyvista as pv

    cube_v, cube_f = cube()
    pts = np.vstack([cube_v, [[3, 0, 0], [0, 3, 0], [0, 0, 3]]])
    faces_vtk = np.hstack([np.full((len(cube_f), 1), 3), cube_f]).ravel()
    poly = pv.PolyData(pts, faces_vtk)
    path.parent.mkdir(parents=True, exist_ok=True)
    poly.save(str(path))


def write_volume_vtu(path: Path, vertices, cells,
                     element_type: str = "hex8",
                     cell_data: dict | None = None) -> None:
    """Save a volume mesh as VTK unstructured grid (.vtu).

    ``cell_data`` optionally maps array names to one value per cell
    (e.g. material IDs), stored as VTK cell data.
    """
    import pyvista as pv

    sys.path.insert(0, str(REPO_ROOT))
    from mesh_issue_examiner.utils.cell_types import (
        ELEMENT_NODE_COUNT, get_cell_type_map,
    )

    n_nodes = ELEMENT_NODE_COUNT[element_type]
    cell_type = get_cell_type_map()[element_type]
    cells_vtk = np.hstack([np.full((len(cells), 1), n_nodes), cells]).ravel()
    grid = pv.UnstructuredGrid(
        cells_vtk,
        np.full(len(cells), cell_type),
        np.asarray(vertices, float),
    )
    if cell_data:
        for name, values in cell_data.items():
            grid.cell_data[name] = np.asarray(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(path))


def write_mixed_volume_vtu(path: Path, vertices,
                            cells_by_type: dict) -> None:
    """Save a mixed-element volume mesh as VTK unstructured grid (.vtu).

    ``cells_by_type`` maps element type names to (n, nodes) arrays; cells
    are written grouped by type in dict order (prisms first, then caps).
    """
    import pyvista as pv

    sys.path.insert(0, str(REPO_ROOT))
    from mesh_issue_examiner.utils.cell_types import (
        ELEMENT_NODE_COUNT, get_cell_type_map,
    )

    type_map = get_cell_type_map()
    parts, types = [], []
    for etype, cells in cells_by_type.items():
        n_nodes = ELEMENT_NODE_COUNT[etype]
        cells = np.asarray(cells)
        assert cells.shape[1] == n_nodes, (
            f"{etype}: expected {n_nodes} nodes per cell"
        )
        parts.append(
            np.hstack([np.full((len(cells), 1), n_nodes), cells]).ravel()
        )
        types.append(np.full(len(cells), type_map[etype]))
    grid = pv.UnstructuredGrid(
        np.concatenate(parts),
        np.concatenate(types),
        np.asarray(vertices, float),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(path))


def quadratic_sliver() -> tuple[np.ndarray, np.ndarray]:
    """Single squashed tet10: corner volume/max_edge^3 ~ 4e-8.

    Long-needle corners a=(0,0,0), b=(10,0,0), c=(5,0.005,0),
    d=(5,0.0025,0.005); V = 4.17e-5 with max edge 10 -> ratio 4.2e-8,
    far below the 1e-6 sliver threshold and far above the 1e-12
    degenerate tolerance.  Midside nodes sit at straight edge midpoints
    (VTK QUADRATIC_TETRA edge order (0,1),(1,2),(2,0),(0,3),(1,3),(2,3))
    so curvature is zero and the corner geometry alone makes the cell
    a sliver — geometric validity must catch it through the corner-
    linearised grid.
    """
    corners = np.array([
        [0, 0, 0], [10, 0, 0], [5, 0.005, 0], [5, 0.0025, 0.005],
    ], float)
    mids = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    vertices = np.vstack(
        [corners, [(corners[i] + corners[j]) / 2 for i, j in mids]])
    cells = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    return vertices, cells


def build_volume_fixtures() -> dict:
    """name -> (vertices, cells, element type) for .vtu fixtures."""
    gv, gc = hex_grid(2, 2, 2)
    dv, dc = hex_grid(2, 1, 1)
    return {
        "closed_volume": (gv, gc, "hex8"),
        "duplicate_cell": (dv, np.vstack([dc, dc[0]]), "hex8"),
        "leaky_volume": (*t_junction_volume(), "hex8"),
        "tet10_volume": (*tet10_volume(), "tet10"),
        "quadratic_sliver": (*quadratic_sliver(), "tet10"),
    }


def build_material_fixtures() -> dict:
    """name -> (vertices, cells, element type, cell_data) for .vtu fixtures
    carrying per-cell data arrays (materials)."""
    v, c = hex_grid(2, 2, 1)
    # Cells are enumerated k-major, then j, then i: the x-half of each
    # cell determines its material -> [1, 2, 1, 2].  The two interface
    # faces at x=1 are conforming (shared vertices), so the mesh is a
    # clean watertight 2x2x1 block.
    materials = [1 + (i % 2) for i in range(len(c))]
    return {
        "two_material_volume": (v, c, "hex8", {"material": materials}),
    }


# Boundary-layer fixtures: single prism column on a unit equilateral wall
# triangle, capped with a tet.  Heights chosen so centroid-metric growth
# ratios (layer spacing = (h_i + h_{i+1}) / 2) stay under the 1.2 threshold
# for the clean fixture and blow past it (3x jump) for the defective one.
BL_LAYER_HEIGHTS = [0.2, 0.23, 0.2645, 0.3042, 0.3498]      # r = 1.15
BL_BAD_HEIGHTS = [0.2, 0.23, 0.69, 0.7935, 0.9125]          # 3x jump


def build_boundary_layer_fixtures() -> dict:
    """name -> (vertices, cells_by_type) for mixed-element .vtu fixtures."""
    return {
        "boundary_layer": prism_stack(BL_LAYER_HEIGHTS),
        "boundary_layer_bad_growth": prism_stack(BL_BAD_HEIGHTS),
    }


NAS_MIXED_DECK = """$ NASTRAN mixed volumetric fixture: CTETRA + CHEXA(20-node form,
$ midside nodes retained -> hex20) + CPENTA(6-node -> prism6).
BEGIN BULK
GRID,1,0,0.0,0.0,0.0
GRID,2,0,1.0,0.0,0.0
GRID,3,0,0.0,1.0,0.0
GRID,4,0,0.0,0.0,1.0
CTETRA,101,1,1,2,3,4
$ unit hex at x=5..6 (G1-G4 bottom, G5-G8 top)
GRID,5,0,5.0,0.0,0.0
GRID,6,0,6.0,0.0,0.0
GRID,7,0,6.0,1.0,0.0
GRID,8,0,5.0,1.0,0.0
GRID,9,0,5.0,0.0,1.0
GRID,10,0,6.0,0.0,1.0
GRID,11,0,6.0,1.0,1.0
GRID,12,0,5.0,1.0,1.0
$ hex midside nodes (exercised by the 20-node form, dropped for hex8)
GRID,13,0,5.5,0.0,0.0
GRID,14,0,6.0,0.5,0.0
GRID,15,0,5.5,1.0,0.0
GRID,16,0,5.0,0.5,0.0
GRID,17,0,5.5,0.0,1.0
GRID,18,0,6.0,0.5,1.0
GRID,19,0,5.5,1.0,1.0
GRID,20,0,5.0,0.5,1.0
GRID,21,0,5.0,0.0,0.5
GRID,22,0,6.0,0.0,0.5
GRID,23,0,6.0,1.0,0.5
GRID,24,0,5.0,1.0,0.5
CHEXA,102,1,5,6,7,8,9,10,+H1
+H1,11,12,13,14,15,16,17,18,+H2
+H2,19,20,21,22,23,24
$ unit wedge at x=10..11 (G1-G3 bottom, G4-G6 top)
GRID,25,0,10.0,0.0,0.0
GRID,26,0,11.0,0.0,0.0
GRID,27,0,10.5,1.0,0.0
GRID,28,0,10.0,0.0,1.0
GRID,29,0,11.0,0.0,1.0
GRID,30,0,10.5,1.0,1.0
CPENTA,103,1,25,26,27,28,29,+P1
+P1,30
ENDDATA
"""


def write_nas_mixed_volume(path: Path) -> None:
    """Write the mixed NASTRAN deck (see NAS_MIXED_DECK)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(NAS_MIXED_DECK, encoding="ascii")


# Gmsh 2.2 ASCII mixed volumetric fixture.  The tet10 keeps Gmsh's node
# ordering (midsides at straight edge midpoints; the last two midside
# entries differ from VTK's order, exercising the permutation), the
# hex8 is VTK-ordered.  Physical groups 1 (fluid) and 2 (solid) become
# cell_materials on load.
GMSH_MIXED_DECK = """$ Gmsh 2.2 ASCII mixed volumetric fixture:
$ tet10 (Gmsh node order) in group 1, hex8 in group 2.
$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
2
3 1 "fluid"
3 2 "solid"
$EndPhysicalNames
$Nodes
18
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 0.0 1.0 0.0
4 0.0 0.0 1.0
5 0.5 0.0 0.0
6 0.5 0.5 0.0
7 0.0 0.5 0.0
8 0.0 0.0 0.5
9 0.0 0.5 0.5
10 0.5 0.0 0.5
11 5.0 0.0 0.0
12 6.0 0.0 0.0
13 6.0 1.0 0.0
14 5.0 1.0 0.0
15 5.0 0.0 1.0
16 6.0 0.0 1.0
17 6.0 1.0 1.0
18 5.0 1.0 1.0
$EndNodes
$Elements
2
1 11 2 1 1 1 2 3 4 5 6 7 8 9 10
2 5 2 2 2 11 12 13 14 15 16 17 18
$EndElements
"""


def write_gmsh_mixed_volume(path: Path) -> None:
    """Write the mixed Gmsh deck (see GMSH_MIXED_DECK)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(GMSH_MIXED_DECK, encoding="ascii")


def write_meshio_volume(path: Path, ext: str) -> None:
    """Write an optional-dep volume fixture via meshio (CGNS / Exodus).

    Requires the delegate dependency meshio uses for the format (h5py
    for .cgns, netCDF4 for .exo); callers catch ImportError when it is
    absent so the fixture is simply not written.  meshio's CGNS
    reader/writer pair only handles tetra cells, so CGNS gets the
    two-unit-tet mesh while Exodus gets the clean 2x2x2 hex grid.
    """
    import meshio

    if ext == "cgns":
        pts = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.25, 0.25, -1],
        ], float)
        blocks = [("tetra", np.array([[0, 1, 2, 3], [1, 0, 2, 4]]))]
    else:
        pts, hexes = hex_grid(2, 2, 2)
        pts = np.asarray(pts, float)
        blocks = [("hexahedron", np.asarray(hexes))]

    mesh = meshio.Mesh(points=pts, cells=blocks)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.write(str(path))


# ---------------------------------------------------------------------------
# Independent geometric checks (do NOT use the examiner's detector)
# ---------------------------------------------------------------------------

def _load_surface(path: Path):
    """Load an on-disk surface fixture with exactly merged coordinates."""
    m = trimesh.load(str(path))
    m.merge_vertices()
    return np.asarray(m.vertices, float), np.asarray(m.faces)


def _edge_pierce_interior(seg_a, seg_b, tri) -> bool:
    """True when segment a-b crosses the strict interior of tri
    (endpoints strictly on opposite sides; hit point strictly inside)."""
    n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    da, db = np.dot(seg_a - tri[0], n), np.dot(seg_b - tri[0], n)
    if da * db >= -1e-15:
        return False
    t = da / (da - db)
    p = seg_a + t * (seg_b - seg_a)
    e1, e2 = tri[1] - tri[0], tri[2] - tri[0]
    ap = p - tri[0]
    d00, d01, d11 = (np.dot(e1, e1), np.dot(e1, e2), np.dot(e2, e2))
    d20, d21 = np.dot(ap, e1), np.dot(ap, e2)
    den = d00 * d11 - d01 * d01
    u = (d11 * d20 - d01 * d21) / den
    v = (d00 * d21 - d01 * d20) / den
    eps = 1e-9
    return eps < u and eps < v and u + v < 1 - eps


def proper_intersection_pairs(path: Path) -> list:
    """Independent proper-crossing check on an on-disk surface file.

    Reports non-adjacent face pairs (i, j) where an edge of one triangle
    passes through the strict interior of the other.  Self-contained so
    the fixture verification does not depend on the examiner's detector
    (which is the code under test).  Adjacent faces (shared vertex) are
    ordinary contact and excluded.
    """
    V, F = _load_surface(path)
    out = []
    for i in range(len(F)):
        for j in range(i + 1, len(F)):
            if set(F[i]) & set(F[j]):
                continue                       # adjacent: ordinary contact
            T1, T2 = V[F[i]], V[F[j]]
            lo = np.maximum(T1.min(0), T2.min(0))
            hi = np.minimum(T1.max(0), T2.max(0))
            if np.any(hi < lo - 1e-12):
                continue                       # disjoint bounding boxes
            hit = False
            for T, U in ((T1, T2), (T2, T1)):
                for a, b in ((T[0], T[1]), (T[1], T[2]), (T[2], T[0])):
                    if _edge_pierce_interior(a, b, U):
                        hit = True
                        break
                if hit:
                    break
            if hit:
                out.append((int(i), int(j)))
    return out


def generate(force: bool = False) -> list:
    """Write all fixtures; returns list of written paths."""
    fixtures = build_fixtures()
    written = []
    for name, (ext, v, f) in fixtures.items():
        d = SAMPLE_DIR / name
        p = d / f"{name}.{ext}"
        if p.exists() and not force:
            continue
        d.mkdir(parents=True, exist_ok=True)
        mesh = trimesh.Trimesh(vertices=np.asarray(v, float),
                               faces=np.asarray(f), process=False)
        mesh.export(str(p))
        written.append(p)
    vtk_path = SAMPLE_DIR / "isolated_vertices" / "isolated_vertices_surface.vtk"
    if vtk_path.exists() and not force:
        pass
    else:
        write_isolated_vertices_vtk(vtk_path)
        written.append(vtk_path)
    for name, (v, c, etype) in build_volume_fixtures().items():
        p = SAMPLE_DIR / name / f"{name}.vtu"
        if p.exists() and not force:
            continue
        write_volume_vtu(p, v, c, etype)
        written.append(p)
    for name, (v, c, etype, cdata) in build_material_fixtures().items():
        p = SAMPLE_DIR / name / f"{name}.vtu"
        if p.exists() and not force:
            continue
        write_volume_vtu(p, v, c, etype, cell_data=cdata)
        written.append(p)
    for name, (v, cells_by_type) in build_boundary_layer_fixtures().items():
        p = SAMPLE_DIR / name / f"{name}.vtu"
        if p.exists() and not force:
            continue
        write_mixed_volume_vtu(p, v, cells_by_type)
        written.append(p)
    nas_path = SAMPLE_DIR / "nas_mixed_volume" / "nas_mixed_volume.nas"
    if not (nas_path.exists() and not force):
        write_nas_mixed_volume(nas_path)
        written.append(nas_path)
    gmsh_path = SAMPLE_DIR / "gmsh_mixed_volume" / "gmsh_mixed_volume.msh"
    if not (gmsh_path.exists() and not force):
        write_gmsh_mixed_volume(gmsh_path)
        written.append(gmsh_path)
    # Optional-dep formats: written only when the delegate dependency
    # (h5py for CGNS, netCDF4 for Exodus) is importable.
    for name, ext in (("cgns_volume", "cgns"), ("exo_volume", "exo")):
        p = SAMPLE_DIR / name / f"{name}.{ext}"
        if p.exists() and not force:
            continue
        try:
            write_meshio_volume(p, ext)
        except ImportError:
            continue
        written.append(p)
    return written


# ---------------------------------------------------------------------------
# Verification: run the real pipeline on every fixture
# ---------------------------------------------------------------------------

async def verify() -> int:
    sys.path.insert(0, str(REPO_ROOT))
    from mesh_issue_examiner.mcp.server import MeshIssueExaminerServer, MCPRequest

    server = MeshIssueExaminerServer(auto_register_operations=True)
    failures = 0

    async def inspect(path: Path, material_array: str | None = None):
        load_params: dict = {"file_path": str(path)}
        if material_array:
            load_params["material_array_name"] = material_array
        r = await server.handle_request(MCPRequest(method="mesh_load",
                                                   params=load_params))
        assert r.error is None, f"load failed for {path.name}: {r.error}"
        mid = r.result["mesh_id"]
        ri = await server.handle_request(MCPRequest(method="mesh_inspect",
                                                    params={"mesh_id": mid}))
        assert ri.error is None, f"inspect failed for {path.name}: {ri.error}"
        r2 = await server.handle_request(MCPRequest(method="mesh_get_issues",
                                                    params={"mesh_id": mid}))
        assert r2.error is None, f"get_issues failed for {path.name}: {r2.error}"
        return ri.result, r2.result

    for name, contract in CONTRACTS.items():
        requires = contract.get("requires")
        if requires and importlib.util.find_spec(requires) is None:
            print(f"skip {name}: requires {requires}")
            continue
        d = SAMPLE_DIR / name
        path = None
        for ext in ("stl", "obj", "vtu", "nas", "msh", "cgns", "exo"):
            cand = d / f"{name}.{ext}"
            if cand.exists():
                path = cand
                break
        if name == "isolated_vertices":
            path = d / "isolated_vertices_surface.vtk"
        if path is None or not path.exists():
            print(f"FAIL {name}: fixture missing")
            failures += 1
            continue
        try:
            insp, iss = await inspect(path, contract.get("material_array"))
        except AssertionError as exc:
            print(f"FAIL {name}: {exc}")
            failures += 1
            continue
        op_ids = {i["operation_id"] for i in iss["issues"]}
        problems = []
        if "passed" in contract and insp["passed"] != contract["passed"]:
            problems.append(f"passed={insp['passed']} want {contract['passed']}")
        if contract.get("zero") and iss["total_count"] != 0:
            problems.append(f"issues={sorted(op_ids)}")
        # Independent gate: a zero-issue surface fixture must also be free
        # of proper non-adjacent triangle intersections, checked without
        # the examiner's own detector (which missed the 2026-10 torus
        # self-intersections recorded in SAMPLE_MESH_AUDIT.md).
        if contract.get("zero") and path.suffix in (".stl", ".obj"):
            crossings = proper_intersection_pairs(path)
            if crossings:
                problems.append(
                    f"{len(crossings)} proper non-adjacent intersection(s), "
                    f"first at faces {crossings[0]}")
        if not contract.get("zero") and iss["total_count"] == 0:
            problems.append("expected issues but got none")
        for kw in contract.get("keywords", []):
            if not any(kw in oid for oid in op_ids):
                problems.append(f"missing op id containing {kw!r}")
        for kw in contract.get("no_keywords", []):
            if any(kw in oid for oid in op_ids):
                problems.append(f"unexpected op id containing {kw!r}")
        if "boundary" in contract:
            nb = sum(1 for oid in op_ids if "boundary" in oid)
            if contract["boundary"] and nb < 1:
                problems.append("expected boundary issue")
            if not contract["boundary"] and nb > 0:
                problems.append("unexpected boundary issue")
        if iss["total_count"] < contract.get("min_issues", 0):
            problems.append("too few issues")
        n_warn = sum(1 for i in iss["issues"] if i["severity"] == "warning")
        if n_warn < contract.get("min_warnings", 0):
            problems.append(f"only {n_warn} warnings")
        if problems:
            print(f"FAIL {name}: " + "; ".join(problems))
            failures += 1
        else:
            sev = insp["summary"]
            print(f"ok   {name}: passed={insp['passed']} "
                  f"issues={sev['total_issues']} ops={sorted(op_ids) or '-'}")
    print(f"\n{'ALL FIXTURES OK' if failures == 0 else f'{failures} FAILURES'}")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true",
                    help="recreate fixtures even if they already exist")
    ap.add_argument("--verify", action="store_true",
                    help="only run pipeline verification (no writing)")
    args = ap.parse_args()
    if not args.verify:
        written = generate(force=args.force)
        state = "wrote" if written else "up-to-date"
        print(f"{state}: {len(written)} file(s), output root {SAMPLE_DIR}")
    return asyncio.run(verify())


if __name__ == "__main__":
    raise SystemExit(main())
