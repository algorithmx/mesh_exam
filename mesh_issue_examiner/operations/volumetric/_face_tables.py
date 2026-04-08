"""
Canonical face vertex definitions for volumetric element types.

Each entry maps an element type name to a list of face tuples.  The tuples
contain **local** vertex indices (0-based) into the element's connectivity
array.  The ordering follows the right-hand-rule convention: for each face,
the cross product of successive edge vectors gives the **outward-pointing**
normal.

Shared by ``nonmanifold_faces.py`` and ``orientation_consistency.py`` so
that face topology definitions live in exactly one place.
"""

from typing import Dict, List, Tuple

ELEMENT_FACES: Dict[str, List[Tuple[int, ...]]] = {
    'tet4': [
        (0, 1, 2),  # base triangle
        (0, 2, 3),  # side triangle
        (0, 3, 1),  # side triangle
        (1, 3, 2),  # side triangle
    ],
    'hex8': [
        (0, 1, 2, 3),  # bottom face
        (4, 5, 6, 7),  # top face
        (0, 1, 5, 4),  # front face
        (1, 2, 6, 5),  # right face
        (2, 3, 7, 6),  # back face
        (3, 0, 4, 7),  # left face
    ],
    'prism6': [
        (0, 1, 2),     # bottom triangle
        (3, 4, 5),     # top triangle
        (0, 1, 4, 3),  # side quad
        (1, 2, 5, 4),  # side quad
        (2, 0, 3, 5),  # side quad
    ],
    'pyramid5': [
        (0, 1, 2, 3),  # base quad
        (0, 1, 4),     # side triangle
        (1, 2, 4),     # side triangle
        (2, 3, 4),     # side triangle
        (3, 0, 4),     # side triangle
    ],
}
