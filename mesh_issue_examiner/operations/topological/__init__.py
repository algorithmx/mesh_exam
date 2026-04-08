"""
Topological mesh inspection operations.
"""

from .nonmanifold_edges import NonManifoldEdgeDetection
from .self_intersection import SelfIntersectionDetection
from .duplicate_vertices import DuplicateVertexDetection

__all__ = ["NonManifoldEdgeDetection", "SelfIntersectionDetection", "DuplicateVertexDetection"]
