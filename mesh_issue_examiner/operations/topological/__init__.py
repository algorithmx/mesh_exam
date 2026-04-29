"""
Topological mesh inspection operations.
"""

from .nonmanifold_edges import NonManifoldEdgeDetection
from .nonmanifold_vertices import NonManifoldVertexDetection
from .self_intersection import SelfIntersectionDetection
from .duplicate_vertices import DuplicateVertexDetection
from .boundary_edges import BoundaryEdgeDetection
from .surface_orientation import SurfaceOrientationCheck
from .disconnected_components import DisconnectedComponentDetection
from .isolated_vertices import IsolatedVertexDetection
from .duplicate_faces import DuplicateFaceDetection

# Granular reporting operations (Gap #13)
from .vertex_valence import VertexValenceDetection
from .edge_dihedral import EdgeDihedralAngleDetection
from .edge_length_ratio import EdgeLengthRatioDetection

__all__ = [
    "NonManifoldEdgeDetection",
    "NonManifoldVertexDetection",
    "SelfIntersectionDetection",
    "DuplicateVertexDetection",
    "BoundaryEdgeDetection",
    "SurfaceOrientationCheck",
    "DisconnectedComponentDetection",
    "IsolatedVertexDetection",
    "DuplicateFaceDetection",
    # Granular reporting operations
    "VertexValenceDetection",
    "EdgeDihedralAngleDetection",
    "EdgeLengthRatioDetection",
]
