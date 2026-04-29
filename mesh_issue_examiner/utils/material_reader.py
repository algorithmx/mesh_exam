"""
Material data extraction from mesh files.

Provides functions to read material/region assignments from various
mesh file formats (NASTRAN, Gmsh, Exodus, etc.).
"""

import logging
from typing import Optional, Tuple, Dict

import numpy as np

logger = logging.getLogger(__name__)


def extract_materials_from_nas(
    file_path: str,
    property_key: str = 'PSOLID',
) -> Tuple[Optional[np.ndarray], Optional[Dict[int, str]]]:
    """Extract material assignments from a NASTRAN file.

    Reads PSOLID (or other property) entries to map elements to material IDs.

    Args:
        file_path: Path to NASTRAN (.nas/.bdf) file.
        property_key: Property card to read (default: 'PSOLID').

    Returns:
        Tuple of (cell_materials array, material_names dict) or (None, None).
    """
    try:
        from .nas_parser import parse_nas
        # Re-parse to get property info — the basic parser returns
        # (vertices, faces, volume_cells). For material data we need
        # to read the raw file and extract property assignments.
    except ImportError:
        return None, None

    material_map: Dict[int, int] = {}  # property_id -> material_id
    element_properties: Dict[int, int] = {}  # element_id -> property_id
    prop_to_mat: Dict[int, int] = {}  # property_id -> material_id
    material_names: Dict[int, str] = {}

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception:
        return None, None

    # Parse PSOLID entries: PSOLID  PID  MID
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('PSOLID'):
            parts = _split_nas_line(stripped)
            if len(parts) >= 3:
                try:
                    pid = int(parts[1])
                    mid = int(parts[2])
                    prop_to_mat[pid] = mid
                    material_names[mid] = f'Material_{mid}'
                except (ValueError, IndexError):
                    continue

    # Parse CTETRA entries: CTETRA  EID  PID  G1  G2  G3  G4
    # Parse CHEXA entries similarly
    for line in lines:
        stripped = line.strip()
        for card in ('CTETRA', 'CHEXA', 'CPENTA', 'CPYRAM'):
            if stripped.startswith(card):
                parts = _split_nas_line(stripped)
                if len(parts) >= 3:
                    try:
                        eid = int(parts[1])
                        pid = int(parts[2])
                        element_properties[eid] = pid
                    except (ValueError, IndexError):
                        continue
                break

    if not element_properties or not prop_to_mat:
        return None, None

    # Build cell_materials array in element ID order
    sorted_eids = sorted(element_properties.keys())
    cell_materials = np.array([
        prop_to_mat.get(element_properties[eid], 0) for eid in sorted_eids
    ])

    return cell_materials, material_names if len(cell_materials) > 0 else (None, None)


def extract_materials_from_gmsh(
    file_path: str,
) -> Tuple[Optional[np.ndarray], Optional[Dict[int, str]]]:
    """Extract material assignments from a Gmsh .msh file.

    Gmsh stores physical groups as cell data. Physical groups often
    represent materials or regions.

    Args:
        file_path: Path to Gmsh .msh file.

    Returns:
        Tuple of (cell_materials array, material_names dict) or (None, None).
    """
    try:
        import meshio
    except ImportError:
        logger.debug("meshio not available for Gmsh material reading")
        return None, None

    try:
        mesh = meshio.read(file_path)
    except Exception as e:
        logger.debug(f"Failed to read Gmsh file: {e}")
        return None, None

    # Gmsh stores physical groups in cell_data
    cell_materials = None
    if hasattr(mesh, 'cell_data') and mesh.cell_data:
        if 'gmsh:physical' in mesh.cell_data:
            arrays = mesh.cell_data['gmsh:physical']
            if arrays:
                cell_materials = np.concatenate(
                    [np.asarray(a).flatten() for a in arrays]
                )

    # Try to read physical names from $PhysicalNames section
    material_names = _read_gmsh_physical_names(file_path)

    if cell_materials is not None and len(cell_materials) > 0:
        return cell_materials, material_names

    return None, None


def _read_gmsh_physical_names(file_path: str) -> Dict[int, str]:
    """Read $PhysicalNames section from a Gmsh .msh file."""
    names: Dict[int, str] = {}
    try:
        with open(file_path, 'r') as f:
            in_section = False
            for line in f:
                stripped = line.strip()
                if stripped == '$PhysicalNames':
                    in_section = True
                    continue
                if stripped == '$EndPhysicalNames':
                    break
                if in_section:
                    parts = stripped.split()
                    if len(parts) >= 3:
                        try:
                            dim = int(parts[0])
                            tag = int(parts[1])
                            name = ' '.join(parts[2:]).strip('"')
                            names[tag] = name
                        except (ValueError, IndexError):
                            continue
    except Exception:
        pass
    return names


def _split_nas_line(line: str):
    """Split a NASTRAN free-format or fixed-field line into fields."""
    if ',' in line:
        return [p.strip() for p in line.split(',')]
    # Fixed-field: 8-char fields
    parts = []
    i = 0
    while i < len(line):
        field = line[i:i+8].strip()
        if field:
            parts.append(field)
        i += 8
    return parts
