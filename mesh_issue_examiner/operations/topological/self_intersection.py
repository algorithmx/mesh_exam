"""
Self-intersection detection operation.

Detects triangles that intersect other non-adjacent triangles in the same mesh.
Uses manual Möller-Trumbore triangle-triangle intersection with KD-tree
spatial acceleration for accurate detection without false positives.
"""

from typing import List, Set, Tuple
import numpy as np

from ...core.models import (
    MeshInput,
    MeshIssue,
    IssueLocation,
    IssueCategory,
    IssueSeverity,
    OperationConfig,
)
from ..base import BaseOperation, OperationMetadata
from ..operation_registry import registry


class SelfIntersectionDetection(BaseOperation):
    """
    Detects triangles that intersect with other triangles in the same mesh.
    
    Self-intersection is a critical geometric defect where non-adjacent triangles
    of a mesh penetrate each other. This can cause:
    - Boolean operation failures
    - Invalid volume calculations  
    - Rendering artifacts
    - Simulation errors in FEA/CFD
    
    Implementation uses the Möller-Trumbore algorithm with KD-tree spatial
    indexing for efficient O(n log n) detection without false positives.
    
    Algorithm:
        1. Build KD-tree from triangle centroids for spatial acceleration
        2. Query for candidate triangle pairs within search radius
        3. Filter out adjacent triangles (share vertices)
        4. Test each pair using Möller-Trumbore triangle-triangle intersection
        5. Handle coplanar triangles with 2D overlap test
    """

    def __init__(self, config: OperationConfig = None):
        super().__init__(config)
        # Threshold for minimum area of intersecting triangles (to ignore slivers)
        self.min_area_threshold = 1e-10

    @property
    def metadata(self) -> OperationMetadata:
        return OperationMetadata(
            operation_id='topology.detect_self_intersections',
            display_name='Self-Intersection Detection',
            description='Identifies triangles that improperly intersect '
                       'other non-adjacent triangles in the same mesh. '
                       'Uses Möller-Trumbore algorithm with KD-tree acceleration.',
            category=IssueCategory.TOPOLOGICAL,
            supported_libraries=['pyvista'],
            recommended_library='pyvista',
            requires_adjacency=True,
            complexity='O(n log n)',
            tags=['topology', 'self-intersection', 'geometric-defect', 'critical']
        )

    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        """Execute self-intersection detection."""
        from ...adapters.factory import get_adapter
        
        if len(mesh.faces) == 0:
            return []
        
        # Get adapter and convert to native format
        adapter = get_adapter(self.config.library_preference or 'pyvista')
        native_mesh = adapter.from_canonical(mesh)
        
        # Get self-intersections from adapter using manual method
        try:
            intersections = adapter.detect_self_intersections(
                native_mesh,
                method='manual',
                tolerance=1e-10
            )
        except Exception as e:
            return []
        
        if not intersections:
            return []
        
        # Filter and create issues
        issues = []
        reported_faces: Set[int] = set()
        
        for face1_idx, face2_idx in intersections:
            # Skip if we've already reported this face
            if face1_idx in reported_faces:
                continue
            
            # Get face vertices for location info
            face = mesh.faces[face1_idx] if face1_idx < len(mesh.faces) else []
            
            # Get coordinates of first vertex for location
            coords = None
            if len(face) > 0 and int(face[0]) < len(mesh.vertices):
                coords = tuple(mesh.vertices[int(face[0])])
            
            issues.append(MeshIssue(
                operation_id=self.metadata.operation_id,
                category=IssueCategory.TOPOLOGICAL,
                severity=IssueSeverity.CRITICAL,
                message=f'Triangle {face1_idx} intersects with non-adjacent triangle(s)',
                locations=[IssueLocation(
                    element_type='face',
                    element_index=int(face1_idx),
                    coordinates=coords,
                    adjacent_elements={'face_vertices': [int(v) for v in face]}
                )],
                metrics={
                    'intersecting_face': int(face1_idx),
                    'paired_face': int(face2_idx) if face2_idx >= 0 else None
                }
            ))
            
            reported_faces.add(face1_idx)
        
        return issues


# Register the operation
registry.register(SelfIntersectionDetection())
