"""
Non-manifold face detection operation for volumetric (3D) meshes.

Detects faces shared by more than two volume cells, which violates
the manifold property required for valid solid models.
"""

from typing import List, Optional, Dict, Tuple, Set
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
from ...utils.metrics import determine_severity
from ...utils.cell_types import ELEMENT_NODE_COUNT
from ..operation_registry import registry
from ._face_tables import ELEMENT_FACES, build_face_to_cell_map


class NonManifoldFaceDetection(BaseOperation):
    """
    Detects faces shared by more than two volume cells.

    Dual of NonManifoldEdgeDetection: where the 2D operation checks edges
    shared by >2 faces, this 3D operation checks faces shared by >2 cells.

    A valid manifold volume mesh requires that every internal face is shared
    by exactly two cells. Faces shared by more than two cells create
    topological ambiguity and cause issues in FEA solvers, CFD, and
    boolean operations.

    Boundary faces (on the exterior of the mesh) should be shared by exactly
    one cell. This operation focuses on internal faces shared by >2 cells,
    which indicate topological errors.
    """

    def __init__(self, config: Optional[OperationConfig] = None):
        super().__init__(config)

    @property
    def metadata(self) -> OperationMetadata:
        return OperationMetadata(
            operation_id='topology_3d.detect_nonmanifold_faces',
            display_name='Non-Manifold Face Detection (3D)',
            description='Identifies faces shared by more than two volume cells, '
                       'violating the manifold property required for valid solid models.',
            category=IssueCategory.TOPOLOGICAL,
            supported_libraries=['pyvista'],
            recommended_library='pyvista',
            requires_adjacency=True,
            complexity='O(n)',
            tags=['topology', 'manifold', 'connectivity', 'volumetric', '3d']
        )

    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        """Execute non-manifold face detection on a volumetric mesh.

        Builds a face-to-cells adjacency map and identifies any face that is
        shared by more than two cells, which indicates a non-manifold topology.

        Args:
            mesh: Input mesh with volumetric cells.

        Returns:
            List of MeshIssue objects describing non-manifold faces found.
            Empty list if no issues detected or mesh has no volume cells.
        """
        # Must have volumetric cells to analyze
        if not mesh.has_volume_cells:
            return []

        # Build face-to-cells adjacency map
        face_to_cells = self._build_face_to_cell_map(mesh)

        if not face_to_cells:
            return []

        # Find faces shared by more than 2 cells (non-manifold)
        nonmanifold_faces = []
        for face_key, cell_entries in face_to_cells.items():
            if len(cell_entries) > 2:
                # This face is shared by more than 2 cells - non-manifold
                cell_indices = [entry[0] for entry in cell_entries]
                face_vertices = cell_entries[0][1]  # First occurrence has the face vertices
                nonmanifold_faces.append((face_key, face_vertices, cell_indices))

        if not nonmanifold_faces:
            return []

        # Create issue locations for each non-manifold face
        locations = self._create_locations(nonmanifold_faces, mesh)

        # Calculate statistics
        total_faces = len(face_to_cells)
        nm_count = len(nonmanifold_faces)
        violation_rate = nm_count / max(total_faces, 1)
        severity = self._determine_severity(violation_rate)

        # Get affected cell types for metadata
        affected_cell_types = self._get_affected_cell_types(nonmanifold_faces, mesh)

        issue = MeshIssue(
            operation_id=self.metadata.operation_id,
            category=IssueCategory.TOPOLOGICAL,
            severity=severity,
            message=f'Detected {nm_count} non-manifold face(s) shared by more '
                   f'than 2 cells ({violation_rate * 100:.2f}% of unique faces). '
                   f'Each internal face should be shared by exactly 2 cells.',
            locations=locations,
            metrics={
                'nonmanifold_face_count': nm_count,
                'unique_face_count': total_faces,
                'violation_rate': violation_rate,
                'manifold_faces': total_faces - nm_count,
            },
            metadata={
                'affected_cell_types': list(affected_cell_types),
                'max_sharing_count': max(len(cells) for _, _, cells in nonmanifold_faces),
            }
        )

        return [issue]

    def _build_face_to_cell_map(
        self,
        mesh: MeshInput
    ) -> Dict[Tuple[int, ...], List[Tuple[int, Tuple[int, ...]]]]:
        """Build mapping from sorted face tuples to list of (cell_idx, face_vertices).

        For each cell, extracts all faces and creates a canonical key by sorting
        vertex indices. This allows faces with different windings but the same
        geometry to be matched.

        Args:
            mesh: Input mesh with volumetric cells.

        Returns:
            Dictionary mapping sorted face tuples to list of (cell_index, face_tuple).
            Faces shared by multiple cells will have multiple entries in the list.
        """
        return build_face_to_cell_map(mesh)

    def _create_locations(
        self,
        nonmanifold_faces: List[Tuple[Tuple[int, ...], Tuple[int, ...], List[int]]],
        mesh: MeshInput
    ) -> List[IssueLocation]:
        """Create IssueLocation objects for non-manifold faces.

        Args:
            nonmanifold_faces: List of (face_key, face_vertices, cell_indices) tuples.
            mesh: Input mesh for vertex coordinate lookup.

        Returns:
            List of IssueLocation objects, capped at _MAX_LOCATIONS.
        """
        locations = []
        vertices = mesh.vertices

        for face_key, face_verts, cell_indices in nonmanifold_faces[:self._MAX_LOCATIONS]:
            # Get centroid of face for coordinates
            try:
                face_coords = vertices[list(face_verts)]
                centroid = tuple(float(x) for x in np.mean(face_coords, axis=0))
            except (IndexError, ValueError):
                centroid = None

            locations.append(IssueLocation(
                element_type='face',
                element_index=-1,  # Face doesn't have a global index
                coordinates=centroid,
                adjacent_elements={
                    'face_vertices': list(face_verts),
                    'sharing_cell_count': len(cell_indices),
                    'sharing_cell_indices': cell_indices[:20],  # Cap for metadata
                }
            ))

        return locations

    def _determine_severity(self, violation_rate: float) -> IssueSeverity:
        """Determine issue severity based on violation rate.

        Non-manifold faces are always serious topology errors, but the severity
        scales with the extent of the problem. Uses centralized thresholds from
        utils.metrics for consistency.

        Args:
            violation_rate: Fraction of unique faces that are non-manifold.

        Returns:
            Appropriate IssueSeverity level.
        """
        from ...utils.metrics import determine_severity
        return determine_severity(violation_rate)

    # Maximum number of IssueLocation objects per issue (avoids huge reports)
    # This prevents memory issues when processing large meshes with many defects
    _MAX_LOCATIONS = 100

    def _get_affected_cell_types(
        self,
        nonmanifold_faces: List[Tuple[Tuple[int, ...], Tuple[int, ...], List[int]]],
        mesh: MeshInput
    ) -> Set[str]:
        """Determine which element types are affected by non-manifold faces.

        Args:
            nonmanifold_faces: List of non-manifold face entries.
            mesh: Input mesh.

        Returns:
            Set of element type names that have cells involved in non-manifold faces.
        """
        affected_cells: Set[int] = set()
        for _, _, cell_indices in nonmanifold_faces:
            affected_cells.update(cell_indices)

        # Map global cell indices back to element types
        affected_types: Set[str] = set()
        global_idx = 0
        for element_type, connectivity in mesh.volume_cells.items():
            n_cells = len(connectivity)
            for i in range(n_cells):
                if global_idx in affected_cells:
                    affected_types.add(element_type)
                global_idx += 1

        return affected_types



# Register operation
registry.register(NonManifoldFaceDetection())

# Note on severity thresholds:
# This operation uses stricter thresholds than other operations because
# non-manifold faces are fundamental topology errors that always indicate
# invalid geometry. The thresholds (10%/5%/1%) are intentionally set lower
# than the standard (50%/20%/5%) used in quality metrics.
