"""
NASTRAN BDF/NAS file parser.

Parses GRID, CTRIA3, CQUAD4, and CTETRA cards from NASTRAN Bulk Data files
and converts them to triangular surface mesh arrays (vertices Nx3, faces Mx3).

Supports:
- Free-field format (comma-separated)
- Short-field format (8-character fixed columns)
"""

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def parse_nas(file_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Parse a NASTRAN BDF/NAS file into surface mesh arrays and volumetric cells.

    Volume elements (CTETRA) are decomposed into triangular faces for the
    boundary surface, and their raw node connectivity is preserved in
    ``volume_cells`` for volumetric analysis.

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
        for line in f:
            raw = line.rstrip('\n\r')

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
