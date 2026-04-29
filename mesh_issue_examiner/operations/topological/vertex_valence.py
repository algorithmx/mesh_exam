"""
Vertex valence detection operation.

Identifies vertices with abnormal valence (number of incident edges/faces).
Abnormal valence indicates mesh irregularities and singularities.
"""

from typing import List, Optional, Dict, Any
import numpy as np
import logging

from ...core.models import (
    MeshInput,
    MeshIssue,
    IssueLocation,
    IssueCategory,
    IssueSeverity,
    OperationConfig,
)
from ...utils.topology import compute_vertex_valence
from ..base import BaseOperation, OperationMetadata
from ..operation_registry import registry

logger = logging.getLogger(__name__)


class VertexValenceDetection(BaseOperation):
    """
    Detects vertices with abnormal valence.
    
    Valence is the number of edges/faces incident to a vertex.
    Abnormal valence indicates mesh irregularities.
    
    For triangle meshes:
    - Interior vertices: ideal = 6, problematic < 4 or > 8
    - Boundary vertices: ideal = 4, problematic < 2 or > 6
    
    For quad meshes:
    - Interior vertices: ideal = 4, problematic < 3 or > 6
    - Boundary vertices: ideal = 2, problematic < 2 or > 4
    
    References:
    - "Topological Mesh Operators" (IMPA) - Valence definitions
    - "5-6-7 Meshes" - Vertex valence in remeshing
    - Loop Subdivision - Regular vertex valence
    
    Parameters
    ----------
    min_valence_triangle : int
        Minimum acceptable valence for triangle meshes (default: 4)
    max_valence_triangle : int
        Maximum acceptable valence for triangle meshes (default: 8)
    min_valence_quad : int
        Minimum acceptable valence for quad meshes (default: 3)
    max_valence_quad : int
        Maximum acceptable valence for quad meshes (default: 6)
    """
    
    # Maximum number of IssueLocation objects per issue
    _MAX_LOCATIONS = 100

    def __init__(
        self,
        config: Optional[OperationConfig] = None,
        min_valence_triangle: int = 4,
        max_valence_triangle: int = 8,
        min_valence_quad: int = 3,
        max_valence_quad: int = 6
    ):
        super().__init__(config)
        self.min_valence_triangle = min_valence_triangle
        self.max_valence_triangle = max_valence_triangle
        self.min_valence_quad = min_valence_quad
        self.max_valence_quad = max_valence_quad

    @property
    def metadata(self) -> OperationMetadata:
        return OperationMetadata(
            operation_id='topology.detect_vertex_valence',
            display_name='Vertex Valence Detection',
            description='Identifies vertices with abnormal valence (connected edge count). '
                       'For triangle meshes: ideal=6, acceptable=4-8. '
                       'For quad meshes: ideal=4, acceptable=3-6.',
            category=IssueCategory.TOPOLOGICAL,
            supported_libraries=['pyvista', 'trimesh'],
            recommended_library='pyvista',
            requires_adjacency=True,
            complexity='O(F)',
            tags=['topology', 'vertex', 'valence', 'irregular', 'singularity']
        )

    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        """Execute vertex valence detection.
        
        Returns:
            List of MeshIssue objects for vertices with abnormal valence.
            Empty list if no irregular vertices found or mesh has no faces.
        """
        if len(mesh.faces) == 0:
            return []
        
        # Compute valence for all vertices
        valence = compute_vertex_valence(mesh.faces, len(mesh.vertices))
        
        # Determine mesh type (triangle vs quad)
        is_triangle_mesh = self._is_triangle_mesh(mesh.faces)
        
        # Set thresholds based on mesh type
        if is_triangle_mesh:
            min_val = self.min_valence_triangle
            max_val = self.max_valence_triangle
            mesh_type = "triangle"
        else:
            min_val = self.min_valence_quad
            max_val = self.max_valence_quad
            mesh_type = "quad"
        
        # Find irregular vertices
        irregular_mask = (valence < min_val) | (valence > max_val)
        irregular_indices = np.where(irregular_mask)[0]
        
        if len(irregular_indices) == 0:
            return []
        
        # Create locations for irregular vertices
        locations = []
        for v_idx in irregular_indices[:self._MAX_LOCATIONS]:
            locations.append(IssueLocation(
                element_type='vertex',
                element_index=int(v_idx),
                coordinates=tuple(float(x) for x in mesh.vertices[v_idx]),
                adjacent_elements={'valence': int(valence[v_idx])}
            ))
        
        # Determine severity based on percentage of irregular vertices
        irregular_rate = len(irregular_indices) / len(mesh.vertices)
        severity = self._determine_severity(irregular_rate)
        
        # Create issue
        issue = MeshIssue(
            operation_id=self.metadata.operation_id,
            category=IssueCategory.TOPOLOGICAL,
            severity=severity,
            message=f"Found {len(irregular_indices)} vertices with irregular valence "
                   f"in {mesh_type} mesh (threshold: {min_val}-{max_val})",
            locations=locations,
            metrics={
                'irregular_count': int(len(irregular_indices)),
                'total_vertices': int(len(mesh.vertices)),
                'irregular_rate': float(irregular_rate),
                'mesh_type': mesh_type,
                'min_threshold': min_val,
                'max_threshold': max_val,
                'valence_distribution': self._compute_distribution(valence),
            },
            metadata={
                'irregular_indices': [int(x) for x in irregular_indices[:50]],
                'valences': [int(valence[i]) for i in irregular_indices[:50]],
            }
        )
        
        return [issue]
    
    def _is_triangle_mesh(self, faces: np.ndarray) -> bool:
        """Determine if mesh is primarily triangle-based.
        
        Args:
            faces: Face connectivity array
            
        Returns:
            True if average vertices per face < 3.5 (mostly triangles)
        """
        if faces.ndim == 2:
            # Uniform faces
            return faces.shape[1] == 3
        else:
            # Mixed faces
            avg_verts = np.mean([len(f) for f in faces])
            return avg_verts < 3.5
    
    def _determine_severity(self, irregular_rate: float) -> IssueSeverity:
        """Determine severity based on percentage of irregular vertices."""
        if irregular_rate > 0.2:
            return IssueSeverity.ERROR
        elif irregular_rate > 0.05:
            return IssueSeverity.WARNING
        else:
            return IssueSeverity.INFO
    
    def _compute_distribution(self, valence: np.ndarray) -> Dict[str, Any]:
        """Compute distribution statistics for valence values."""
        unique, counts = np.unique(valence, return_counts=True)
        return {
            'valence_values': [int(v) for v in unique],
            'counts': [int(c) for c in counts],
            'mean': float(np.mean(valence)),
            'min': int(np.min(valence)),
            'max': int(np.max(valence)),
        }


# Register operation
registry.register(VertexValenceDetection())
