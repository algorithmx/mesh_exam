"""
Utility functions for mesh loading and processing.
"""

from .mesh_loader import load_mesh_from_file, create_mesh_from_arrays
from .metrics import (
    compute_all_quality_metrics,
    get_default_thresholds,
    get_volumetric_thresholds,
    get_threshold,
)
from .dimensionality import detect_dimensionality, detect_mesh_dimensionality
from .cell_types import (
    ELEMENT_NODE_COUNT,
    NODE_COUNT_TO_ELEMENT_TYPE,
    get_cell_type_map,
    get_vtk_type_to_name,
    detect_element_type,
    validate_volume_cells,
)

__all__ = [
    "load_mesh_from_file",
    "create_mesh_from_arrays",
    "compute_all_quality_metrics",
    "get_default_thresholds",
    "get_volumetric_thresholds",
    "get_threshold",
    "detect_dimensionality",
    "detect_mesh_dimensionality",
    "ELEMENT_NODE_COUNT",
    "NODE_COUNT_TO_ELEMENT_TYPE",
    "get_cell_type_map",
    "get_vtk_type_to_name",
    "detect_element_type",
    "validate_volume_cells",
]
