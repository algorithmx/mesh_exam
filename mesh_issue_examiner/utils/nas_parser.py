"""
NASTRAN BDF/NAS file parser.

Parses GRID, CTRIA3, CQUAD4, CTETRA, CHEXA, and CPENTA cards from NASTRAN
bulk Data files and converts them to triangular surface mesh arrays
(Vertices Nx3, faces Mx3).

Supports:
- Free-field format (comma-separated)
- Short-field format (8-character fixed columns)
- Continuation cards for multi-card entries (CHEXA, CPENTA)

Volume elements are preserved in ``volume_cells`` with their higher-order
forms retained (2026-10-28): CTETRA -> 'tet4', CHEXA -> 'hex8' for the
8-node form or 'hex20' for the 20-node form, CPENTA -> 'prism6' for the
6-node form or 'prism15' for the 15-node form.  NASTRAN's midside node
ordering (bottom ring, top ring, verticals) matches the VTK quadratic
ordering, so the full node list maps directly onto hex20/prism15.
Corner reduction applies only to forms with no registered element type.
"""

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def parse_nas(file_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Parse a NASTRAN BDF/NAS file into surface mesh arrays and volumetric cells.

    Volume elements (CTETRA, CHEXA, CPENTA) are decomposed into corner-based
    triangular faces for the boundary surface (midside nodes do not change
    face identity), while ``volume_cells`` retains the full connectivity:
    'tet4'/'hex8'/'prism6' for linear forms and 'hex20'/'prism15' when the
    20/15-node forms supply midside nodes.

    Args:
        file_path: Path to .nas or .bdf file.

    Returns:
        Tuple of (vertices_Nx3, faces_Mx3, volume_cells) where
        volume_cells maps element type names to NxK connectivity arrays
        (e.g. {'tet4': np.array(...)}). Empty dict if no volumetric elements.

    Raises:
        ValueError: If no GRID or no element cards are found.
    """
    nodes: Dict[int, Tuple[float, float, float]] = {}
    surface_faces: List[Tuple[int, int, int]] = []
    volume_faces: List[Tuple[int, int, int]] = []
    volume_cells_raw: Dict[str, List[List[int]]] = {}

    in_bulk = False
    with open(file_path, 'r') as f:
        lines = f.readlines()

    i = 0
    n_lines = len(lines)
    while i < n_lines:
        raw = lines[i].rstrip('\n\r')
        i += 1

        # Skip until BEGIN BULK
        if not in_bulk:
            if raw.strip().upper().startswith('BEGIN BULK'):
                in_bulk = True
            continue

        # End of bulk data
        stripped = raw.strip().upper()
        if stripped.startswith('ENDATA') or stripped.startswith('ENDDATA'):
            break

        # Skip comments and blank lines
        if not raw or raw.startswith('$'):
            continue

        # Determine card keyword (first 8 chars or first comma-delimited field)
        if ',' in raw:
            fields = _parse_free_field(raw)
        else:
            fields = _parse_short_field(raw)

        if not fields:
            continue

        keyword = fields[0].upper()

        if keyword == 'GRID':
            _parse_grid(fields, nodes)

        elif keyword == 'CTRIA3':
            face = _parse_ctria3(fields)
            if face:
                surface_faces.append(face)

        elif keyword == 'CQUAD4':
            tris = _parse_cquad4(fields)
            if tris:
                surface_faces.extend(tris)

        elif keyword == 'CTETRA':
            result = _parse_ctetra_volumetric(fields)
            if result is not None:
                tris, cell_nodes = result
                volume_faces.extend(tris)
                volume_cells_raw.setdefault('tet4', []).append(cell_nodes)

        elif keyword in ('CHEXA', 'CPENTA'):
            # Multi-card entries: gather continuation cards (first field
            # starts with '+') before parsing the node list.
            if keyword == 'CHEXA':
                n_corners, quad_total, quad_type = 8, 20, 'hex20'
            else:
                n_corners, quad_total, quad_type = 6, 15, 'prism15'
            node_fields = list(fields[3:])
            while i < n_lines:
                nxt_raw = lines[i].rstrip('\n\r')
                if not nxt_raw or nxt_raw.startswith('$'):
                    break
                nxt_fields = (_parse_free_field(nxt_raw) if ',' in nxt_raw
                              else _parse_short_field(nxt_raw))
                if not nxt_fields or not nxt_fields[0].upper().startswith('+'):
                    break
                i += 1
                node_fields.extend(nxt_fields[1:])
            result = _parse_volume_cell(node_fields, n_corners,
                                        quad_total, quad_type)
            if result is not None:
                tris, cell_nodes, etype = result
                volume_faces.extend(tris)
                volume_cells_raw.setdefault(etype, []).append(cell_nodes)

    if not nodes:
        raise ValueError(f"No GRID cards found in {file_path}")

    if not surface_faces and not volume_faces:
        raise ValueError(f"No element cards found in {file_path}")

    # For volume elements, extract boundary faces only
    if volume_faces:
        boundary_faces = _extract_boundary_faces(volume_faces)
        surface_faces.extend(boundary_faces)

    # Collect all used node IDs (from faces and volume cells)
    used_nodes = set()
    for face in surface_faces:
        used_nodes.update(face)
    for cells in volume_cells_raw.values():
        for cell in cells:
            used_nodes.update(cell)

    id_map = {old_id: new_idx for new_idx, old_id in enumerate(sorted(used_nodes))}

    vertices = np.array(
        [nodes[nid] for nid in sorted(used_nodes)],
        dtype=np.float64
    )

    faces = np.array(
        [tuple(id_map[v] for v in face) for face in surface_faces],
        dtype=np.int64
    )

    # Remap volume cell node IDs
    volume_cells: Dict[str, np.ndarray] = {}
    for etype, cells in volume_cells_raw.items():
        volume_cells[etype] = np.array(
            [[id_map[n] for n in cell] for cell in cells],
            dtype=np.int64
        )

    return vertices, faces, volume_cells


# ---------------------------------------------------------------------------
# Field parsers
# ---------------------------------------------------------------------------

def _parse_free_field(line: str) -> List[str]:
    """Parse a free-field (comma-separated) line."""
    return [f.strip() for f in line.split(',')]


def _parse_short_field(line: str) -> List[str]:
    """Parse a short-field (8-char columns) line.

    First field is 8 chars (keyword), subsequent fields are each 8 chars.
    """
    fields = []
    for i in range(0, len(line), 8):
        field = line[i:i + 8].strip()
        fields.append(field)
    return [f for f in fields if f]


# ---------------------------------------------------------------------------
# Card parsers
# ---------------------------------------------------------------------------

def _parse_grid(fields: List[str], nodes: Dict[int, Tuple[float, float, float]]):
    """Parse GRID card: GRID,ID,CP,X1,X2,X3  (free-field)
    or  GRID_____ID______CP______X1______X2______X3 (short-field)

    Field 2 (CP, coordinate system) is skipped.
    """
    try:
        nid = int(fields[1])
        # fields[2] is CP (coordinate system), skip
        x = float(fields[3])
        y = float(fields[4])
        z = float(fields[5])
        nodes[nid] = (x, y, z)
    except (IndexError, ValueError):
        pass


def _parse_ctria3(fields: List[str]) -> Optional[Tuple[int, int, int]]:
    """Parse CTRIA3 card: CTRIA3,EID,PID,G1,G2,G3"""
    try:
        return (int(fields[3]), int(fields[4]), int(fields[5]))
    except (IndexError, ValueError):
        return None


def _parse_cquad4(fields: List[str]) -> List[Tuple[int, int, int]] | None:
    """Parse CQUAD4 card: CQUAD4,EID,PID,G1,G2,G3,G4

    Split quad into 2 triangles.
    """
    try:
        g1, g2, g3, g4 = int(fields[3]), int(fields[4]), int(fields[5]), int(fields[6])
        return [(g1, g2, g3), (g1, g3, g4)]
    except (IndexError, ValueError):
        return None


def _parse_ctetra(fields: List[str]) -> List[Tuple[int, int, int]] | None:
    """Parse CTETRA card: CTETRA,EID,PID,G1,G2,G3,G4

    Decompose tet into 4 triangular faces.
    """
    try:
        g1, g2, g3, g4 = int(fields[3]), int(fields[4]), int(fields[5]), int(fields[6])
        return [
            (g1, g2, g3),
            (g1, g2, g4),
            (g1, g3, g4),
            (g2, g3, g4),
        ]
    except (IndexError, ValueError):
        return None


def _parse_ctetra_volumetric(fields: List[str]) -> tuple | None:
    """Parse CTETRA card returning both surface decomposition and raw connectivity.

    Returns:
        Tuple of (triangular_faces, [g1, g2, g3, g4]) or None on parse error.
    """
    try:
        g1, g2, g3, g4 = int(fields[3]), int(fields[4]), int(fields[5]), int(fields[6])
        faces = [
            (g1, g2, g3),
            (g1, g2, g4),
            (g1, g3, g4),
            (g2, g3, g4),
        ]
        return faces, [g1, g2, g3, g4]
    except (IndexError, ValueError):
        return None


# Corner node order matches the VTK hex8 convention (G5-G8 above G1-G4).
_HEX_QUADS = [
    (0, 1, 2, 3), (4, 5, 6, 7),
    (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
]

# Corner node order matches the VTK wedge convention (G4-G6 above G1-G3).
_PRISM_TRIS = [(0, 1, 2), (3, 4, 5)]
_PRISM_QUADS = [(0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)]


def _parse_volume_cell(node_fields: List[str], n_corners: int,
                       quad_total: int, quad_type: str) -> tuple | None:
    """Parse CHEXA/CPENTA node fields into connectivity + corner faces.

    ``node_fields`` is the concatenation of the parent card's node fields
    (from field 4 on, i.e. after keyword/EID/PID) and any continuation-card
    node fields.  Continuation markers (fields starting with '+') are
    ignored.

    Higher-order forms are retained: when at least ``quad_total`` node IDs
    are present the full node list becomes a ``quad_type`` cell (hex20 /
    prism15); NASTRAN's midside ordering matches the VTK quadratic
    ordering, so the connectivity needs no reordering.  Corner-only forms
    (and forms with no registered higher-order type) keep the linear
    hex8/prism6 connectivity.

    The boundary-face decomposition is always corner-based: midside nodes
    do not change face identity.

    Returns:
        Tuple of (triangular_faces, node_list, element_type) or None if
        fewer than ``n_corners`` valid node IDs are present.
    """
    nodes: List[int] = []
    for f in node_fields:
        if f.startswith('+'):
            continue
        try:
            nodes.append(int(f))
        except ValueError:
            continue
    if len(nodes) < n_corners:
        return None

    if len(nodes) >= quad_total:
        etype = quad_type
        cell_nodes = nodes[:quad_total]
    else:
        etype = 'hex8' if quad_total == 20 else 'prism6'
        cell_nodes = nodes[:n_corners]
    corners = nodes[:n_corners]

    tris: List[Tuple[int, int, int]] = []
    if n_corners == 8:
        for a, b, c, d in _HEX_QUADS:
            tris.append((corners[a], corners[b], corners[c]))
            tris.append((corners[a], corners[c], corners[d]))
    else:  # wedge
        for a, b, c in _PRISM_TRIS:
            tris.append((corners[a], corners[b], corners[c]))
        for a, b, c, d in _PRISM_QUADS:
            tris.append((corners[a], corners[b], corners[c]))
            tris.append((corners[a], corners[c], corners[d]))
    return tris, cell_nodes, etype


# ---------------------------------------------------------------------------
# Boundary extraction
# ---------------------------------------------------------------------------

def _extract_boundary_faces(
    faces: List[Tuple[int, int, int]],
) -> List[Tuple[int, int, int]]:
    """Remove internal (shared) faces, keeping only the boundary surface.

    A face is internal if it appears exactly twice (shared between two volume
    elements). Boundary faces appear exactly once.
    """
    # Count occurrences using sorted-tuple keys (orientation-independent)
    counter: Counter = Counter()
    face_lookup: Dict[tuple, Tuple[int, int, int]] = {}

    for face in faces:
        key = tuple(sorted(face))
        counter[key] += 1
        # Keep the original (oriented) face for the first occurrence
        if key not in face_lookup:
            face_lookup[key] = face

    return [face_lookup[key] for key, count in counter.items() if count == 1]
