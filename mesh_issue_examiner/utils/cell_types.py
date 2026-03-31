"""
Element type constants and helpers for volumetric mesh cells.

Maps between human-readable element type names (tet4, hex8, etc.),
node counts, and VTK/pyvista CellType enum values.
"""

from typing import Dict, List

import numpy as np


# Element type name -> number of nodes
ELEMENT_NODE_COUNT: Dict[str, int] = {
    'tet4': 4,
    'hex8': 8,
    'prism6': 6,
    'pyramid5': 5,
}

# Reverse map: node count -> default element type name
NODE_COUNT_TO_ELEMENT_TYPE: Dict[int, str] = {
    v: k for k, v in ELEMENT_NODE_COUNT.items()
}

# Element type name -> pyvista CellType integer value.
# Imported lazily to avoid requiring pyvista at import time.
_CELL_TYPE_MAP = None


def get_cell_type_map() -> Dict[str, int]:
    """Return element type name -> VTK CellType enum value mapping.
    
    Raises:
        ImportError: If pyvista is not installed. PyVista is required for
            volumetric mesh support.
    """
    global _CELL_TYPE_MAP
    if _CELL_TYPE_MAP is None:
        try:
            import pyvista as pv
            _CELL_TYPE_MAP = {
                'tet4': pv.CellType.TETRA,
                'hex8': pv.CellType.HEXAHEDRON,
                'prism6': pv.CellType.WEDGE,
                'pyramid5': pv.CellType.PYRAMID,
            }
        except ImportError as e:
            raise ImportError(
                "pyvista is required for volumetric mesh support. "
                "Install with: pip install pyvista"
            ) from e
    return _CELL_TYPE_MAP


# Reverse: VTK CellType int -> element type name
_VTK_TYPE_TO_NAME = None


def get_vtk_type_to_name() -> Dict[int, str]:
    """Return VTK CellType int -> element type name mapping."""
    global _VTK_TYPE_TO_NAME
    if _VTK_TYPE_TO_NAME is None:
        _VTK_TYPE_TO_NAME = {v: k for k, v in get_cell_type_map().items()}
    return _VTK_TYPE_TO_NAME


def detect_element_type(connectivity_array: np.ndarray) -> str:
    """Detect element type from connectivity array column count.

    Args:
        connectivity_array: NxK array of node indices.

    Returns:
        Element type name (e.g. 'tet4', 'hex8').

    Raises:
        ValueError: If column count doesn't match a known element type.
    """
    if connectivity_array.ndim != 2:
        raise ValueError(
            f"Expected 2D array, got shape {connectivity_array.shape}"
        )
    ncols = connectivity_array.shape[1]
    etype = NODE_COUNT_TO_ELEMENT_TYPE.get(ncols)
    if etype is None:
        raise ValueError(
            f"Cannot detect element type from column count {ncols}. "
            f"Known: {NODE_COUNT_TO_ELEMENT_TYPE}"
        )
    return etype


def validate_volume_cells(volume_cells: Dict[str, np.ndarray]) -> None:
    """Validate a volume_cells dict has correct shapes.

    Raises:
        ValueError: If any array has wrong column count for its type.
    """
    for etype, arr in volume_cells.items():
        if etype not in ELEMENT_NODE_COUNT:
            raise ValueError(
                f"Unknown element type '{etype}'. "
                f"Known: {list(ELEMENT_NODE_COUNT.keys())}"
            )
        expected = ELEMENT_NODE_COUNT[etype]
        if arr.ndim != 2 or arr.shape[1] != expected:
            raise ValueError(
                f"Element type '{etype}' expects {expected} columns, "
                f"got shape {arr.shape}"
            )
