"""
Edge dihedral angle detection operation.

Detects edges with extreme dihedral angles between adjacent faces.
Useful for identifying sharp features, creases, and mesh irregularities.
"""

from typing import List, Optional
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
from ...utils.topology import build_edge_face_map
from ..base import BaseOperation, OperationMetadata
from ..operation_registry import registry

logger = logging.getLogger(__name__)


class EdgeDihedralAngleDetection(BaseOperation):
    """
    Detects edges with extreme dihedral angles.
    
    Dihedral angle is the angle between two adjacent faces.
    - Smooth surface: dihedral ~ 0-20°
    - Sharp edge: dihedral ~ 90°
    - Crease/edge: dihedral > 90°
    - Inverted faces: dihedral approaching 180°
    
    For CFD meshes:
    - Avoid dihedral angles < 20° or > 150°
    - Smooth transitions preferred
    
    For FEA meshes:
    - Sharp edges may be intentional (features)
    - Very small angles indicate mesh quality issues
    
    References:
    - "Enforcing Mesh Quality Constraints" - Dihedral angle computation
    - Labelle Thesis - Dihedral angles critical for tet quality
    
    Parameters
    ----------
    min_angle : float
        Minimum acceptable dihedral angle in degrees (default: 20)
    max_angle : float
        Maximum acceptable dihedral angle in degrees (default: 160)
    """
    
    # Maximum number of IssueLocation objects per issue
    _MAX_LOCATIONS = 100

    def __init__(
        self,
        config: Optional[OperationConfig] = None,
        min_angle: float = 20.0,
        max_angle: float = 160.0
    ):
        super().__init__(config)
        self.min_angle = min_angle
        self.max_angle = max_angle

    @property
    def metadata(self) -> OperationMetadata:
        return OperationMetadata(
            operation_id='topology.detect_edge_dihedral_angle',
            display_name='Edge Dihedral Angle Detection',
            description='Identifies edges with extreme dihedral angles between faces. '
                       'Smooth surfaces: ~0-20°, Sharp edges: ~90°. '
                       'Default threshold: 20°-160°.',
            category=IssueCategory.GEOMETRIC,
            supported_libraries=['pyvista', 'trimesh'],
            recommended_library='pyvista',
            requires_adjacency=True,
            complexity='O(E)',
            tags=['topology', 'edge', 'dihedral', 'angle', 'smoothness', 'sharp']
        )

    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        """Execute edge dihedral angle detection.
        
        Returns:
            List of MeshIssue objects for edges with extreme dihedral angles.
            Empty list if no extreme angles found or mesh has insufficient faces.
        """
        if len(mesh.faces) < 2:
            return []
        
        # Compute face normals using adapter
        try:
            from ...adapters.factory import get_adapter
            adapter = get_adapter('pyvista')
            native_mesh = adapter.from_canonical(mesh)
            face_normals = adapter.compute_face_normals(native_mesh)
        except Exception as e:
            logger.warning(f"Could not compute face normals: {e}")
            return []
        
        # Build edge-face map
        edge_faces = build_edge_face_map(mesh.faces)
        
        # Find edges with extreme dihedral angles
        extreme_edges = []
        
        for edge, adjacent_faces in edge_faces.items():
            # Only check internal edges (shared by exactly 2 faces)
            if len(adjacent_faces) != 2:
                continue
            
            f1, f2 = adjacent_faces[0], adjacent_faces[1]
            n1 = face_normals[f1]
            n2 = face_normals[f2]
            
            # Compute angle between face normals
            cos_angle = np.dot(n1, n2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            
            # Dihedral angle is supplementary to normal angle
            dihedral = 180.0 - angle
            
            # Check if extreme
            if dihedral < self.min_angle or dihedral > self.max_angle:
                extreme_edges.append({
                    'edge': edge,
                    'dihedral': dihedral,
                    'faces': adjacent_faces,
                    'v1': edge[0],
                    'v2': edge[1],
                })
        
        if not extreme_edges:
            return []
        
        # Create locations for extreme edges
        locations = []
        for edge_info in extreme_edges[:self._MAX_LOCATIONS]:
            v1, v2 = edge_info['v1'], edge_info['v2']
            midpoint = (mesh.vertices[v1] + mesh.vertices[v2]) / 2.0
            
            locations.append(IssueLocation(
                element_type='edge',
                element_index=-1,  # Edges don't have global index
                coordinates=tuple(float(x) for x in midpoint),
                adjacent_elements={
                    'vertices': [int(v1), int(v2)],
                    'dihedral_angle': float(edge_info['dihedral']),
                    'face_indices': [int(f) for f in edge_info['faces']],
                }
            ))
        
        # Determine severity
        extreme_rate = len(extreme_edges) / len(edge_faces)
        severity = self._determine_severity(extreme_rate)
        
        # Compute statistics
        dihedrals = [e['dihedral'] for e in extreme_edges]
        
        issue = MeshIssue(
            operation_id=self.metadata.operation_id,
            category=IssueCategory.GEOMETRIC,
            severity=severity,
            message=f"Found {len(extreme_edges)} edges with extreme dihedral angles "
                   f"(threshold: {self.min_angle}°-{self.max_angle}°)",
            locations=locations,
            metrics={
                'extreme_edge_count': int(len(extreme_edges)),
                'total_internal_edges': int(len([e for e in edge_faces.values() if len(e) == 2])),
                'extreme_rate': float(extreme_rate),
                'min_dihedral': float(np.min(dihedrals)),
                'max_dihedral': float(np.max(dihedrals)),
                'mean_dihedral': float(np.mean(dihedrals)),
                'threshold_min': self.min_angle,
                'threshold_max': self.max_angle,
            },
            metadata={
                'extreme_edges': [
                    {
                        'vertices': list(e['edge']),
                        'dihedral': float(e['dihedral']),
                    }
                    for e in extreme_edges[:50]
                ],
            }
        )
        
        return [issue]
    
    def _determine_severity(self, extreme_rate: float) -> IssueSeverity:
        """Determine severity based on percentage of extreme edges."""
        if extreme_rate > 0.2:
            return IssueSeverity.ERROR
        elif extreme_rate > 0.05:
            return IssueSeverity.WARNING
        else:
            return IssueSeverity.INFO


# Register operation
registry.register(EdgeDihedralAngleDetection())
