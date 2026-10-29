"""
Controlled mesh visualization.

Programmatic, offscreen-first rendering of canonical meshes to PNG images
with camera, appearance, scalar-coloring, and issue-highlight control.
"""

from .renderer import (
    RenderResult,
    SEVERITY_COLORS,
    VIEW_ANGLES,
    VisualizationError,
    render_mesh,
)

__all__ = [
    "RenderResult",
    "SEVERITY_COLORS",
    "VIEW_ANGLES",
    "VisualizationError",
    "render_mesh",
]
