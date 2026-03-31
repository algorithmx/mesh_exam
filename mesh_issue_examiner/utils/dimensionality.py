"""
Mesh dimensionality detection utility.

Classifies meshes into PURE_2D, SURFACE_2D, VOLUMETRIC_3D, or MIXED_2D_3D
based on vertex geometry and element topology.
"""

from typing import Optional

import numpy as np

from ..core.models import MeshDimensionality

_COPLANARITY_THRESHOLD: float = 1e-10


def detect_dimensionality(
    vertices: np.ndarray,
    surface_cells: Optional[np.ndarray] = None,
    volume_cells: Optional[np.ndarray] = None,
    *,
    coplanarity_threshold: float = _COPLANARITY_THRESHOLD,
) -> MeshDimensionality:
    """Detect the dimensionality of a mesh from its geometry and topology.

    Args:
        vertices: Nx3 array of vertex coordinates.
        surface_cells: Optional MxK array of surface element connectivity
            (triangles, quads). Each row is one element; K is the node count.
        volume_cells: Optional PxK array of volume element connectivity
            (tetra, hex, pyramid, wedge, etc.).
        coplanarity_threshold: Relative threshold for SVD-based planarity
            detection. Smaller values are stricter.

    Returns:
        MeshDimensionality enum value.

    Raises:
        ValueError: If vertices array is empty or has wrong shape.
    """
    _validate_vertices(vertices)

    has_surface = surface_cells is not None and len(surface_cells) > 0
    has_volume = volume_cells is not None and len(volume_cells) > 0

    if has_volume and has_surface:
        return MeshDimensionality.MIXED_2D_3D

    if has_volume:
        return MeshDimensionality.VOLUMETRIC_3D

    if _check_coplanarity(vertices, coplanarity_threshold):
        return MeshDimensionality.PURE_2D

    return MeshDimensionality.SURFACE_2D


def detect_mesh_dimensionality(
    mesh,
    *,
    coplanarity_threshold: float = _COPLANARITY_THRESHOLD,
) -> MeshDimensionality:
    """Convenience wrapper accepting a MeshInput object.

    Detects dimensionality considering both surface faces and volumetric cells.

    Args:
        mesh: MeshInput object.
        coplanarity_threshold: Threshold for planarity detection.

    Returns:
        MeshDimensionality enum value.
    """
    # For volumetric meshes, use the first volume cell array as volume_cells param
    vol_cells = None
    if mesh.has_volume_cells:
        # Use the first available volume cell array for dimensionality detection
        vol_cells = next(iter(mesh.volume_cells.values()))

    return detect_dimensionality(
        vertices=mesh.vertices,
        surface_cells=mesh.faces if mesh.n_faces > 0 else None,
        volume_cells=vol_cells,
        coplanarity_threshold=coplanarity_threshold,
    )


def _validate_vertices(vertices: np.ndarray) -> None:
    """Validate vertex array shape and contents."""
    if not isinstance(vertices, np.ndarray):
        raise ValueError("vertices must be a numpy array")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            f"vertices must be Nx3 array, got shape {vertices.shape}"
        )
    if len(vertices) == 0:
        raise ValueError("vertices array must not be empty")


def _check_coplanarity(
    vertices: np.ndarray,
    threshold: float,
) -> bool:
    """Determine if all vertices lie on a single plane using SVD.

    Algorithm:
        1. Subtract the centroid.
        2. Compute SVD singular values of the centered Nx3 matrix.
        3. If singular_values[2] / singular_values[0] < threshold,
           the points are effectively coplanar.

    For fewer than 3 vertices, trivially returns True.
    """
    n = len(vertices)
    if n < 3:
        return True

    extent = np.ptp(vertices, axis=0)
    if np.min(extent) < threshold:
        return True

    centered = vertices - np.mean(vertices, axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)

    ratio = singular_values[2] / singular_values[0]
    return bool(ratio < threshold)
