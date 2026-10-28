"""
Gmsh .msh 2.2 ASCII parser.

Parses the ``$MeshFormat`` / ``$PhysicalNames`` / ``$Nodes`` / ``$Elements``
sections of legacy (version 2.x) ASCII Gmsh meshes into the canonical
surface + volumetric arrays, retaining the higher-order element forms:

* linear: tri3 / quad4 / tet4 / hex8 / prism6 / pyramid5
* quadratic: tet10 / hex20 / hex27 / prism15 / wedge18 / pyramid13 / pyramid14

Gmsh numbers nodes and elements from 1 with arbitrary tags; like the
NASTRAN parser, all IDs are remapped to a contiguous 0-based indexing
over the used nodes.

**Node ordering.**  Gmsh cells are mostly ordered like VTK with a few
exceptions (tet10, hex20, hex27, wedge15, pyramid13); the permutations
below mirror the ones meshio applies when reading .msh files, so the
canonical connectivity produced here matches ``meshio``'s for the same
file (see ``meshio/gmsh/common.py::_gmsh_to_meshio_order``).

**Materials.**  The first tag of each element is its physical-group id
(the Gmsh convention).  Volume-element tags are returned as
``cell_materials`` in ``volume_cells`` enumeration order (per element
type, first occurrence), with names from ``$PhysicalNames``.

The parser deliberately raises ``ValueError`` for files it does not
handle (binary, version 4.x, missing sections) so ``mesh_loader`` can
fall back to meshio.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .nas_parser import (
    _HEX_QUADS,
    _PRISM_QUADS,
    _PRISM_TRIS,
    _extract_boundary_faces,
)

# Gmsh element type code -> (canonical name, total nodes, corner nodes,
# kind).  Codes per the MSH 2 format specification; verified against
# meshio's gmsh reader table.
GMSH_ELEMENT_TYPES: Dict[int, Tuple[str, int, int, str]] = {
    2: ('tri3', 3, 3, 'surface'),
    3: ('quad4', 4, 4, 'surface'),
    4: ('tet4', 4, 4, 'volume'),
    5: ('hex8', 8, 8, 'volume'),
    6: ('prism6', 6, 6, 'volume'),
    7: ('pyramid5', 5, 5, 'volume'),
    9: ('tri6', 6, 3, 'surface'),
    10: ('quad9', 9, 4, 'surface'),
    11: ('tet10', 10, 4, 'volume'),
    12: ('hex27', 27, 8, 'volume'),
    13: ('wedge18', 18, 6, 'volume'),
    14: ('pyramid14', 14, 5, 'volume'),
    16: ('quad8', 8, 4, 'surface'),
    17: ('hex20', 20, 8, 'volume'),
    18: ('prism15', 15, 6, 'volume'),
    19: ('pyramid13', 13, 5, 'volume'),
}

# Gmsh -> canonical (VTK) node permutations for the types whose orderings
# differ.  Corner nodes come first in both conventions for every type
# here, so the permutations are the identity on the corner prefix.
GMSH_TO_CANONICAL_ORDER: Dict[str, List[int]] = {
    'tet10': [0, 1, 2, 3, 4, 5, 6, 7, 9, 8],
    'hex20': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13,
        9, 16, 18, 19, 17, 10, 12, 14, 15,
    ],
    'hex27': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13,
        9, 16, 18, 19, 17, 10, 12, 14, 15,
        22, 23, 21, 24, 20, 25, 26,
    ],
    'prism15': [0, 1, 2, 3, 4, 5, 6, 9, 7, 12, 14, 13, 8, 10, 11],
    'pyramid13': [0, 1, 2, 3, 4, 5, 8, 10, 6, 7, 9, 11, 12],
}

# Corner-face decompositions.  tet/hex/prism tables are shared with the
# NASTRAN parser; the pyramid table is local (base quad + 4 side tris).
_TET_TRIS = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
_PYRAMID_QUAD = (0, 1, 2, 3)
_PYRAMID_TRIS = [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]


def parse_msh(
    file_path: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray],
           Optional[np.ndarray], Optional[Dict[int, str]]]:
    """Parse a Gmsh .msh 2.2 ASCII file.

    Args:
        file_path: Path to .msh file.

    Returns:
        Tuple of (vertices_Nx3, faces_Mx3, volume_cells, cell_materials,
        material_names).  ``volume_cells`` maps canonical element type
        names to (n, nodes) connectivity arrays (insertion order).
        ``cell_materials`` (or ``material_names``) is None when the file
        carries no physical-group tags (or names); when present it is
        aligned with the ``volume_cells`` enumeration order.

    Raises:
        ValueError: If the file is not a version 2.x ASCII mesh the
            parser understands (binary or 4.x files are declined so the
            loader can fall back to meshio), or is malformed.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    nodes: Dict[int, Tuple[float, float, float]] = {}
    physical_names: Dict[int, str] = {}
    surface_faces: List[Tuple[int, int, int]] = []
    volume_faces: List[Tuple[int, int, int]] = []
    volume_cells_raw: Dict[str, List[List[int]]] = {}
    materials_raw: Dict[str, List[int]] = {}
    saw_elements = False
    all_tagged = True

    i = 0
    n_lines = len(lines)
    while i < n_lines:
        stripped = lines[i].strip()
        i += 1

        if not stripped.startswith('$'):
            continue

        section = stripped[1:].lower()

        if section == 'meshformat':
            if i >= n_lines:
                raise ValueError('Truncated $MeshFormat section')
            parts = lines[i].split()
            version, file_type = parts[0], int(parts[1])
            if not version.startswith('2.'):
                raise ValueError(
                    f"Unsupported Gmsh version {version} (native parser "
                    f"handles 2.x only)")
            if file_type != 0:
                raise ValueError('Binary Gmsh files are not supported')
            i += 1
        elif section == 'physicalnames':
            if i >= n_lines:
                raise ValueError('Truncated $PhysicalNames section')
            count = int(lines[i].split()[0])
            i += 1
            for _ in range(count):
                if i >= n_lines:
                    raise ValueError('Truncated $PhysicalNames entries')
                entry = lines[i].strip()
                i += 1
                if '"' in entry:
                    left, name = entry.split('"', 1)
                    name = name.rsplit('"', 1)[0]
                    fields = left.split()
                else:
                    fields = entry.split()
                    name = fields[-1] if len(fields) > 2 else ''
                if len(fields) >= 2:
                    try:
                        physical_names[int(fields[1])] = name
                    except ValueError:
                        pass
        elif section == 'nodes':
            if i >= n_lines:
                raise ValueError('Truncated $Nodes section')
            count = int(lines[i].split()[0])
            i += 1
            for _ in range(count):
                if i >= n_lines:
                    raise ValueError('Truncated $Nodes entries')
                parts = lines[i].split()
                i += 1
                try:
                    nid = int(parts[0])
                    nodes[nid] = (float(parts[1]), float(parts[2]),
                                  float(parts[3]))
                except (IndexError, ValueError):
                    raise ValueError(
                        f'Malformed node entry: {lines[i - 1]!r}')
        elif section == 'elements':
            saw_elements = True
            if i >= n_lines:
                raise ValueError('Truncated $Elements section')
            count = int(lines[i].split()[0])
            i += 1
            for _ in range(count):
                if i >= n_lines:
                    raise ValueError('Truncated $Elements entries')
                parts = lines[i].split()
                i += 1
                try:
                    _parse_element_line(
                        parts, nodes, surface_faces, volume_faces,
                        volume_cells_raw, materials_raw)
                except _UntaggedElements:
                    all_tagged = False
                except ValueError:
                    raise ValueError(
                        f'Malformed element entry: {lines[i - 1]!r}')
        # Unknown sections ($Periodic, $NodeData, ...) are skipped via
        # the "$"-header loop.

    if not nodes:
        raise ValueError(f'No $Nodes section found in {file_path}')
    if not saw_elements or (not surface_faces and not volume_faces):
        raise ValueError(f'No usable elements found in {file_path}')

    # Boundary faces of the volume elements complement explicit surfaces.
    if volume_faces:
        surface_faces.extend(_extract_boundary_faces(volume_faces))

    # Collect used node IDs and remap to contiguous 0-based indexing.
    used_nodes = set()
    for face in surface_faces:
        used_nodes.update(face)
    for cells in volume_cells_raw.values():
        for cell in cells:
            used_nodes.update(cell)

    id_map = {old_id: new_idx
              for new_idx, old_id in enumerate(sorted(used_nodes))}

    vertices = np.array([nodes[nid] for nid in sorted(used_nodes)],
                        dtype=np.float64)
    faces = np.array(
        [tuple(id_map[v] for v in face) for face in surface_faces],
        dtype=np.int64)

    volume_cells: Dict[str, np.ndarray] = {
        etype: np.array([[id_map[n] for n in cell] for cell in cells],
                        dtype=np.int64)
        for etype, cells in volume_cells_raw.items()
    }

    cell_materials = None
    if materials_raw and all_tagged:
        # volume_cells dict preserves insertion order (first occurrence
        # of each element type); concatenate materials accordingly.
        cell_materials = np.array(
            [tag for etype in volume_cells
             for tag in materials_raw.get(etype, [])],
            dtype=np.int64)

    material_names = physical_names or None

    return vertices, faces, volume_cells, cell_materials, material_names


class _UntaggedElements(Exception):
    """Signal: element(s) without physical tags -> no material array."""


def _parse_element_line(
    parts: List[str],
    nodes: Dict[int, Tuple[float, float, float]],
    surface_faces: List[Tuple[int, int, int]],
    volume_faces: List[Tuple[int, int, int]],
    volume_cells_raw: Dict[str, List[List[int]]],
    materials_raw: Dict[str, List[int]],
) -> None:
    """Parse one $Elements line: elm-number elm-type num-tags tags nodes."""
    elem_type = int(parts[1])
    num_tags = int(parts[2])
    spec = GMSH_ELEMENT_TYPES.get(elem_type)
    if spec is None:
        return  # unsupported type (lines, points, high-order variants)

    name, total, corners, kind = spec
    node_ids = [int(t) for t in parts[3 + num_tags:3 + num_tags + total]]
    if len(node_ids) != total:
        raise ValueError('wrong node count')
    for nid in node_ids:
        if nid not in nodes:
            raise ValueError(f'unknown node id {nid}')

    # Gmsh -> canonical (VTK) node ordering.
    permutation = GMSH_TO_CANONICAL_ORDER.get(name)
    if permutation is not None:
        node_ids = [node_ids[p] for p in permutation]

    if kind == 'surface':
        conn = node_ids[:corners]
        if name in ('tri3', 'tri6'):
            surface_faces.append(tuple(conn))
        else:  # quad4 / quad8 / quad9 -> two triangles
            surface_faces.append((conn[0], conn[1], conn[2]))
            surface_faces.append((conn[0], conn[2], conn[3]))
        return

    volume_cells_raw.setdefault(name, []).append(node_ids)
    volume_faces.extend(_corner_faces(node_ids, corners))

    if num_tags >= 1:
        materials_raw.setdefault(name, []).append(int(parts[3]))
    else:
        raise _UntaggedElements()


def _corner_faces(node_ids: List[int], corners: int) -> List[Tuple[int, int, int]]:
    """Corner-based triangular decomposition of a volume cell boundary.

    Midside nodes never change face identity, so only the corner prefix
    is used — the same convention as the NASTRAN parser.
    """
    c = node_ids[:corners]
    if corners == 4:      # tet
        return [(c[a], c[b], c[d]) for a, b, d in _TET_TRIS]
    if corners == 8:      # hex
        tris = []
        for a, b, d, e in _HEX_QUADS:
            tris.append((c[a], c[b], c[d]))
            tris.append((c[a], c[d], c[e]))
        return tris
    if corners == 6:      # prism / wedge
        tris = [(c[a], c[b], c[d]) for a, b, d in _PRISM_TRIS]
        for a, b, d, e in _PRISM_QUADS:
            tris.append((c[a], c[b], c[d]))
            tris.append((c[a], c[d], c[e]))
        return tris
    if corners == 5:      # pyramid
        a, b, d, e = _PYRAMID_QUAD
        return ([(c[a], c[b], c[d]), (c[a], c[d], c[e])]
                + [(c[x], c[y], c[4]) for x, y, _ in _PYRAMID_TRIS])
    raise ValueError(f'Unsupported corner count {corners}')
