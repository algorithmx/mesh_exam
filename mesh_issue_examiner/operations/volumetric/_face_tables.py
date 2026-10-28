"""
Canonical face vertex definitions for volumetric element types.

Each entry maps an element type name to a list of face tuples.  The tuples
contain **local** vertex indices (0-based) into the element's connectivity
array.  The ordering follows the right-hand-rule convention: for each face,
the cross product of successive edge vectors gives the **outward-pointing**
normal.

Quadratic element types (tet10, hex20, prism15, pyramid13 and the complete
variants hex27, wedge18, pyramid14) do not need their own tables: VTK
node ordering places the **corner nodes first**, followed by edge/face/body
nodes.  The linear face tables therefore apply unchanged to quadratic
cells when restricted to the corner nodes — the face tuples index into the
first ``ELEMENT_CORNER_COUNT[etype]`` entries of the connectivity row, and
because corners come first the raw local indices can be used directly.

Shared by ``nonmanifold_faces.py``, ``orientation_consistency.py``, and
``boundary_closure.py`` so that face topology definitions live in exactly
one place.
"""

from typing import Dict, List, Optional, Tuple

# Recalibrated 2026-10-28: every entry now follows the outward
# right-hand rule under VTK node ordering (verified per-face via signed
# volumes).  The previous table mixed conventions — the hex8 bottom face,
# prism6 bottom cap and pyramid5 base were inward-wound while the other
# faces were outward — so vertically stacked cells listed shared faces
# with identical directed edges and check_orientation_consistency
# reported winding errors on perfectly regular grids.  The tet4 table
# was uniformly inverted (harmless for the pairwise check, since every
# face flipped together) and is normalised here as well.  Face ORDER is
# unchanged: face 0 remains the bottom/base face (see
# boundary_layer_quality.BASE_FACE_INDICES).
ELEMENT_FACES: Dict[str, List[Tuple[int, ...]]] = {
    'tet4': [
        (2, 1, 0),  # base triangle
        (3, 2, 0),  # side triangle
        (1, 3, 0),  # side triangle
        (2, 3, 1),  # side triangle
    ],
    'hex8': [
        (3, 2, 1, 0),  # bottom face
        (4, 5, 6, 7),  # top face
        (0, 1, 5, 4),  # front face
        (1, 2, 6, 5),  # right face
        (2, 3, 7, 6),  # back face
        (3, 0, 4, 7),  # left face
    ],
    'prism6': [
        (2, 1, 0),     # bottom triangle
        (3, 4, 5),     # top triangle
        (0, 1, 4, 3),  # side quad
        (1, 2, 5, 4),  # side quad
        (2, 0, 3, 5),  # side quad
    ],
    'pyramid5': [
        (3, 2, 1, 0),  # base quad
        (0, 1, 4),     # side triangle
        (1, 2, 4),     # side triangle
        (2, 3, 4),     # side triangle
        (3, 0, 4),     # side triangle
    ],
}

# Number of corner (linear) nodes per element type.  For linear elements
# this equals the total node count; for quadratic elements VTK ordering
# places these nodes first in the connectivity row.
ELEMENT_CORNER_COUNT: Dict[str, int] = {
    # Linear elements
    'tet4': 4,
    'hex8': 8,
    'prism6': 6,
    'pyramid5': 5,
    # Quadratic elements (corner nodes first)
    'tet10': 4,
    'hex20': 8,
    'prism15': 6,
    'pyramid13': 5,
    'hex27': 8,
    'wedge18': 6,
    'pyramid14': 5,
}

# Quadratic element type -> linear element type whose face table applies
# to its corner nodes.
LINEAR_BASE_TYPE: Dict[str, str] = {
    'tet10': 'tet4',
    'hex20': 'hex8',
    'hex27': 'hex8',
    'prism15': 'prism6',
    'wedge18': 'prism6',
    'pyramid13': 'pyramid5',
    'pyramid14': 'pyramid5',
}


def get_face_definitions(element_type: str) -> Optional[List[Tuple[int, ...]]]:
    """Return the linear corner face definitions for an element type.

    For quadratic types the base linear table is returned; because VTK
    node ordering places corner nodes first, its local indices address the
    corner nodes of the quadratic connectivity row directly.

    Args:
        element_type: Element type name (e.g. 'tet4', 'tet10', 'hex27').

    Returns:
        List of face tuples, or None if the type has no face table.
    """
    base = LINEAR_BASE_TYPE.get(element_type, element_type)
    return ELEMENT_FACES.get(base)


def build_face_to_cell_map(mesh):
    """Build a canonical face-to-cell adjacency map for a volumetric mesh.

    Iterates all volume cells in ``mesh.volume_cells`` insertion order,
    assigning a global cell index to each cell (the same numbering used by
    duplicate/orphaned-cell detection), and maps every face (keyed by its
    sorted corner-vertex tuple) to the list of ``(global_cell_index,
    face_vertices)`` entries incident to it.

    Quadratic element types contribute their corner faces via
    :func:`get_face_definitions`.  Element types without a face table are
    excluded from the map but still advance the global cell index so that
    reported indices stay aligned with the mesh's global cell numbering.

    Args:
        mesh: ``MeshInput`` with ``volume_cells``.

    Returns:
        Dict mapping sorted face tuples to lists of
        ``(global_cell_index, face_vertices_tuple)``.
    """
    face_to_cells = {}
    global_cell_idx = 0

    for element_type, connectivity in mesh.volume_cells.items():
        face_definitions = get_face_definitions(element_type)
        if face_definitions is None:
            # Skip unknown element types, but keep the global cell
            # numbering aligned with other operations.
            global_cell_idx += len(connectivity)
            continue

        n_cells = len(connectivity)

        for local_cell_idx in range(n_cells):
            cell_vertices = connectivity[local_cell_idx]

            for face_def in face_definitions:
                face_verts = tuple(int(cell_vertices[i]) for i in face_def)
                face_key = tuple(sorted(face_verts))

                face_to_cells.setdefault(face_key, []).append(
                    (global_cell_idx, face_verts)
                )

            global_cell_idx += 1

    return face_to_cells
