"""
Controlled, offscreen-first mesh rendering built on PyVista.

This module provides deterministic, programmatic visualization of
:class:`~mesh_issue_examiner.core.models.MeshInput` objects:

- Offscreen rendering to PNG files (no blocking interactive windows)
- Named camera views (iso/front/back/left/right/top/bottom) plus azimuth /
  elevation / zoom fine-grained control
- Appearance controls (window size, background, base color, edges,
  opacity, representation)
- Scalar coloring by material ID or by a quality metric from an
  :class:`~mesh_issue_examiner.core.models.InspectionResult`
- Severity-colored markers overlaid at detected issue locations

Rendering is serialized behind a module-level lock because VTK render
windows are not thread-safe.
"""

import logging
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.exceptions import VisualizationError
from ..core.models import InspectionResult, MeshInput, MeshIssue

logger = logging.getLogger(__name__)

# VTK render windows are not thread-safe; serialize all rendering work.
_RENDER_LOCK = threading.Lock()

#: Named camera views expressed as (azimuth, elevation) degrees.
VIEW_ANGLES: Dict[str, Tuple[float, float]] = {
    "iso": (45.0, 30.0),
    "front": (0.0, 0.0),
    "back": (180.0, 0.0),
    "right": (90.0, 0.0),
    "left": (-90.0, 0.0),
    "top": (0.0, 90.0),
    "bottom": (0.0, -90.0),
}

#: Marker colors for each issue severity.
SEVERITY_COLORS: Dict[str, str] = {
    "critical": "red",
    "error": "orange",
    "warning": "yellow",
    "info": "gray",
}

#: Upper bound on drawn issue markers to keep renders fast on huge meshes.
_MAX_MARKERS = 5000

_offscreen_status: Optional[bool] = None


@dataclass
class RenderResult:
    """Outcome of a successful render."""

    path: str
    width: int
    height: int
    n_highlighted: int = 0
    scalars_used: Optional[str] = None


def _offscreen_available() -> bool:
    """Return True if VTK can create an offscreen GL context (cached)."""
    global _offscreen_status
    if _offscreen_status is None:
        try:
            import pyvista as pv

            probe = pv.Plotter(off_screen=True, window_size=(32, 32))
            try:
                probe.add_mesh(pv.Plane(), color="white")
                probe.screenshot(None)
            finally:
                probe.close()
            _offscreen_status = True
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning("Offscreen rendering unavailable: %s", exc)
            _offscreen_status = False
    return _offscreen_status


def _place_camera(plotter, mesh: MeshInput, view: str,
                  azimuth: Optional[float], elevation: Optional[float],
                  zoom: Optional[float]) -> None:
    """Position the camera deterministically from view angles."""
    base_az, base_el = VIEW_ANGLES[view]
    az = base_az + (azimuth or 0.0)
    el = max(-89.9, min(89.9, base_el + (elevation or 0.0)))

    az_rad, el_rad = math.radians(az), math.radians(el)
    direction = np.array([
        math.cos(el_rad) * math.cos(az_rad),
        math.cos(el_rad) * math.sin(az_rad),
        math.sin(el_rad),
    ])

    min_b, max_b = mesh.bounding_box
    diagonal = float(np.linalg.norm(max_b - min_b))
    if diagonal <= 0.0:
        diagonal = 1.0

    focal = np.asarray(mesh.center, dtype=float)
    position = focal + direction * (2.5 * diagonal)

    # Keep the up vector perpendicular to the view direction.
    up = (0.0, 0.0, 1.0) if abs(direction[2]) < 0.99 else (0.0, 1.0, 0.0)
    plotter.camera_position = [tuple(position), tuple(focal), up]

    if zoom is not None:
        if zoom <= 0.0:
            raise ValueError(f"zoom must be positive, got {zoom}")
        plotter.camera.zoom(zoom)


def _cell_centroid(mesh: MeshInput, global_index: int) -> Optional[np.ndarray]:
    """Centroid of the volume cell at canonical global index."""
    offset = 0
    for connectivity in mesh.volume_cells.values():
        n = len(connectivity)
        if offset <= global_index < offset + n:
            row = np.asarray(connectivity[global_index - offset])
            return np.asarray(mesh.vertices[row], dtype=float).mean(axis=0)
        offset += n
    return None


def _face_centroid(mesh: MeshInput, face_index: int) -> Optional[np.ndarray]:
    """Centroid of the surface face at the given index."""
    if 0 <= face_index < mesh.n_faces:
        face = mesh.faces[face_index]
        return np.asarray(mesh.vertices[np.asarray(face)], dtype=float).mean(axis=0)
    return None


def _collect_markers(
    mesh: MeshInput,
    issues: Optional[Sequence[MeshIssue]],
    severities: Sequence[str],
) -> Dict[str, np.ndarray]:
    """Return ``{severity: (N, 3) positions}`` for issue highlight markers."""
    by_severity: Dict[str, List[Tuple[float, float, float]]] = {}

    for issue in issues or []:
        severity = issue.severity.value
        if severity not in severities:
            continue
        for loc in issue.locations:
            coord: Optional[np.ndarray] = None
            if loc.coordinates is not None:
                coord = np.asarray(loc.coordinates, dtype=float)
            else:
                index = loc.element_index
                if loc.element_type == "vertex":
                    if 0 <= index < mesh.n_vertices:
                        coord = np.asarray(mesh.vertices[index], dtype=float)
                elif loc.element_type == "face":
                    coord = _face_centroid(mesh, index)
                elif loc.element_type == "cell":
                    coord = _cell_centroid(mesh, index)

            if coord is None or not np.all(np.isfinite(coord)):
                continue
            by_severity.setdefault(severity, []).append(
                (float(coord[0]), float(coord[1]), float(coord[2]))
            )

    return {sev: np.array(pts) for sev, pts in by_severity.items() if pts}


def _resolve_scalars(
    mesh: MeshInput,
    scalars: Optional[str],
    result: Optional[InspectionResult],
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Resolve a scalar source name into (values, display_name)."""
    if scalars is None:
        return None, None

    if scalars == "material":
        if mesh.cell_materials is None:
            raise ValueError(
                "Mesh has no material data; load with material_array_name "
                "to populate cell_materials."
            )
        return np.asarray(mesh.cell_materials, dtype=float), "material"

    if result is None:
        raise ValueError(
            f"Scalar source '{scalars}' requires an inspection result; "
            "run mesh_inspect first."
        )
    for qm in result.quality_metrics:
        if qm.metric_name == scalars:
            values = np.array(qm.values, dtype=float, copy=True)
            # VTK uses -1 as the undefined-quality sentinel; render as NaN.
            values[values == -1.0] = np.nan
            return values, scalars

    available = sorted({qm.metric_name for qm in result.quality_metrics})
    raise ValueError(
        f"Unknown scalar source '{scalars}'. Available metrics: {available}"
    )


def render_mesh(
    mesh: MeshInput,
    output_path,
    *,
    view: str = "iso",
    azimuth: Optional[float] = None,
    elevation: Optional[float] = None,
    zoom: Optional[float] = None,
    width: int = 1200,
    height: int = 800,
    background: str = "white",
    color: str = "lightblue",
    show_edges: bool = True,
    opacity: float = 1.0,
    representation: str = "surface",
    scalars: Optional[str] = None,
    cmap: str = "viridis",
    scalar_bar: bool = True,
    issues: Optional[Sequence[MeshIssue]] = None,
    highlight_severities: Sequence[str] = ("critical", "error"),
    result: Optional[InspectionResult] = None,
    off_screen: bool = True,
) -> RenderResult:
    """Render a mesh to a PNG file under full programmatic control.

    Args:
        mesh: Canonical mesh to render (surface, volumetric, or point cloud).
        output_path: Destination PNG path; parent directory must exist.
        view: Named camera view: iso | front | back | left | right | top | bottom.
        azimuth: Extra azimuth rotation in degrees (added to the view angle).
        elevation: Extra elevation rotation in degrees (added to the view angle).
        zoom: Camera zoom factor; 1.0 keeps default framing, >1 zooms in.
        width: Image width in pixels (64-8192).
        height: Image height in pixels (64-8192).
        background: Background color name or hex string.
        color: Base mesh color (ignored when ``scalars`` is provided).
        show_edges: Draw mesh edges on top of the surface.
        opacity: Mesh opacity in (0, 1].
        representation: "surface" | "wireframe" | "points".
        scalars: "material" or a metric name from ``result`` for cell coloring.
        cmap: Colormap name for scalar coloring.
        scalar_bar: Show a scalar bar when scalars are rendered.
        issues: Issues whose locations are marked in the image.
        highlight_severities: Severities eligible for highlight markers.
        result: InspectionResult used to resolve metric scalars.
        off_screen: Render offscreen (default). ``False`` opens a blocking
            interactive window — for local debugging only, never via MCP.

    Returns:
        RenderResult with the written path, image dimensions, marker count,
        and the scalar source actually used.

    Raises:
        ValueError: Invalid parameters or unresolvable scalar source.
        VisualizationError: Offscreen rendering unavailable or render failed.
    """
    import pyvista as pv

    view = str(view).lower()
    if view not in VIEW_ANGLES:
        raise ValueError(f"Unknown view '{view}'. Supported: {sorted(VIEW_ANGLES)}")
    if representation not in ("surface", "wireframe", "points"):
        raise ValueError(
            f"Unknown representation '{representation}'. "
            "Supported: surface, wireframe, points"
        )
    if not (0.0 < float(opacity) <= 1.0):
        raise ValueError(f"opacity must be in (0, 1], got {opacity}")
    width, height = int(width), int(height)
    if not (64 <= width <= 8192) or not (64 <= height <= 8192):
        raise ValueError(f"width/height must be in [64, 8192], got {width}x{height}")

    out = Path(output_path)
    if not out.parent.exists():
        raise ValueError(f"Output directory does not exist: {out.parent}")
    if out.suffix == "":
        out = out.with_suffix(".png")

    if off_screen and not _offscreen_available():
        raise VisualizationError(
            "Offscreen rendering is not available in this environment "
            "(no GL context could be created). Ensure OpenGL support is "
            "available, or pass off_screen=False for an interactive window."
        )

    values, scalars_name = _resolve_scalars(mesh, scalars, result)

    is_volumetric = mesh.has_volume_cells
    is_point_cloud = (not is_volumetric) and mesh.n_faces == 0

    if values is not None and is_point_cloud:
        raise ValueError("Scalar coloring is not supported for point clouds.")
    if values is not None:
        expected = mesh.n_volume_cells if is_volumetric else mesh.n_faces
        if len(values) != expected:
            kind = "volume cells" if is_volumetric else "faces"
            raise ValueError(
                f"Scalar array length {len(values)} does not match "
                f"{kind} count {expected}."
            )

    with _RENDER_LOCK:
        plotter = pv.Plotter(off_screen=off_screen, window_size=(width, height))
        try:
            plotter.set_background(background)

            native = mesh.get_native("pyvista")

            if is_point_cloud or representation == "points":
                plotter.add_points(
                    native.points,
                    color=color,
                    point_size=6,
                    render_points_as_spheres=True,
                    opacity=opacity,
                )
            else:
                actor_mesh = native
                if is_volumetric:
                    if representation == "surface" and values is None:
                        # Explicit algorithm keeps behavior stable and avoids
                        # PyVista's future-default deprecation warning.
                        actor_mesh = native.extract_surface(
                            algorithm="dataset_surface"
                        )
                        if actor_mesh.n_cells == 0:
                            actor_mesh = native
                if values is not None:
                    # Attach scalars to a copy so the cached native object
                    # (MeshInput._native_objects) is never mutated.
                    actor_mesh = actor_mesh.copy()
                    actor_mesh.cell_data[scalars_name] = values

                plotter.add_mesh(
                    actor_mesh,
                    color=None if values is not None else color,
                    show_edges=show_edges,
                    opacity=opacity,
                    style="wireframe" if representation == "wireframe" else None,
                    scalars=scalars_name if values is not None else None,
                    cmap=cmap,
                    show_scalar_bar=(bool(scalar_bar) if values is not None else False),
                    scalar_bar_args=(
                        {"title": scalars_name} if values is not None else None
                    ),
                )

            # Issue highlight markers
            markers = _collect_markers(mesh, issues, tuple(highlight_severities))
            n_markers = 0
            legend_entries: List[Tuple[str, str]] = []
            for severity, points in markers.items():
                if n_markers + len(points) > _MAX_MARKERS:
                    points = points[: _MAX_MARKERS - n_markers]
                    logger.warning(
                        "Issue markers capped at %d for this render", _MAX_MARKERS
                    )
                if len(points) == 0:
                    continue
                n_markers += len(points)
                marker_color = SEVERITY_COLORS.get(severity, "magenta")
                plotter.add_points(
                    points,
                    color=marker_color,
                    point_size=14,
                    render_points_as_spheres=True,
                    reset_camera=False,
                )
                legend_entries.append((severity, marker_color))
            if legend_entries:
                plotter.add_legend(legend_entries)

            _place_camera(plotter, mesh, view, azimuth, elevation, zoom)
            img = plotter.screenshot(str(out))
        finally:
            plotter.close()

    if img is None or getattr(img, "ndim", 0) < 3 or not Path(out).exists():
        raise VisualizationError(f"Renderer produced no image at {out}")

    return RenderResult(
        path=str(out),
        width=int(img.shape[1]),
        height=int(img.shape[0]),
        n_highlighted=n_markers,
        scalars_used=scalars_name,
    )
