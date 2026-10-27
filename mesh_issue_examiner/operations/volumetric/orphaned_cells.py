"""
Orphaned cell detection operation for volumetric (3D) meshes.

Detects cells that reference vertex indices outside the vertex array.
Such references are invalid topology data and can crash native mesh
libraries (PyVista/VTK/Trimesh) that dereference them directly.
"""

from typing import List, Optional

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


class OrphanedCellDetection(BaseOperation):
    """
    Detects volume cells referencing invalid vertex indices.

    A cell is orphaned/invalid when any of its vertex indices is negative
    or greater than or equal to the number of vertices in the mesh. Any
    occurrence is reported as CRITICAL: the connectivity data is corrupt
    and native-library access risks a hard crash (out-of-bounds read).
    """

    # Maximum number of IssueLocation objects per issue (avoids huge reports)
    _MAX_LOCATIONS = 100

    def __init__(self, config: Optional[OperationConfig] = None):
        super().__init__(config)

    @property
    def metadata(self) -> OperationMetadata:
        return OperationMetadata(
            operation_id='topology_3d.detect_orphaned_cells',
            display_name='Orphaned Cell Detection (3D)',
            description='Identifies volume cells referencing vertex indices '
                       'outside the vertex array (invalid topology data that '
                       'can crash native mesh libraries).',
            category=IssueCategory.TOPOLOGICAL,
            supported_libraries=['pyvista'],
            recommended_library='pyvista',
            requires_adjacency=False,
            complexity='O(n)',
            tags=['topology', 'validation', 'connectivity', 'volumetric', '3d']
        )

    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        """Execute orphaned cell detection on a volumetric mesh.

        Args:
            mesh: Input mesh with volumetric cells.

        Returns:
            List with a single MeshIssue if orphaned cells were found,
            otherwise an empty list.
        """
        if not mesh.has_volume_cells:
            return []

        n_vertices = mesh.n_vertices
        orphaned = []  # (global_cell_idx, element_type, invalid_indices)

        global_idx = 0
        for element_type, connectivity in mesh.volume_cells.items():
            conn = np.asarray(connectivity)
            for local_idx in range(len(conn)):
                row = conn[local_idx]
                invalid = [int(v) for v in row if v < 0 or v >= n_vertices]
                if invalid:
                    orphaned.append((global_idx, element_type, invalid))
                global_idx += 1

        if not orphaned:
            return []

        total_cells = global_idx
        orphaned_count = len(orphaned)

        locations = [
            IssueLocation(
                element_type='cell',
                element_index=gidx,
                adjacent_elements={
                    'element_type': etype,
                    'invalid_vertex_indices': invalid,
                },
            )
            for gidx, etype, invalid in orphaned[:self._MAX_LOCATIONS]
        ]

        issue = MeshIssue(
            operation_id=self.metadata.operation_id,
            category=IssueCategory.TOPOLOGICAL,
            severity=IssueSeverity.CRITICAL,
            message=f'{orphaned_count}/{total_cells} volume cell(s) reference '
                    f'vertex indices outside the vertex array '
                    f'(n_vertices={n_vertices}). This is invalid topology data '
                    f'and can crash native mesh libraries.',
            locations=locations,
            metrics={
                'orphaned_cell_count': orphaned_count,
                'total_cell_count': total_cells,
                'violation_rate': orphaned_count / max(total_cells, 1),
                'n_vertices': n_vertices,
            },
            metadata={
                'max_invalid_index': max(
                    max(invalid) for _, _, invalid in orphaned
                ),
            },
        )
        return [issue]


# Register operation
registry.register(OrphanedCellDetection())
