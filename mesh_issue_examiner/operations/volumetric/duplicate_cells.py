"""
Duplicate cell detection operation for volumetric (3D) meshes.

Detects volume cells with identical vertex sets (regardless of ordering),
which waste memory and confuse downstream solvers.
"""

from typing import Dict, List, Optional, Tuple

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
from ..operation_registry import registry


class DuplicateCellDetection(BaseOperation):
    """
    Detects volume cells whose vertex sets are identical to another cell's.

    Uses a canonical sorted-vertex-tuple key per cell (mirroring the 2D
    duplicate-face detection): any cell whose key was already seen is a
    duplicate, regardless of winding order. Keys of elements with different
    node counts never collide, so a single map covers mixed-element meshes.
    """

    # Maximum number of IssueLocation objects per issue (avoids huge reports)
    _MAX_LOCATIONS = 100

    def __init__(self, config: Optional[OperationConfig] = None):
        super().__init__(config)

    @property
    def metadata(self) -> OperationMetadata:
        return OperationMetadata(
            operation_id='topology_3d.detect_duplicate_cells',
            display_name='Duplicate Cell Detection (3D)',
            description='Identifies volume cells with identical vertex sets, '
                       'which waste memory and confuse downstream solvers.',
            category=IssueCategory.TOPOLOGICAL,
            supported_libraries=['pyvista'],
            recommended_library='pyvista',
            requires_adjacency=False,
            complexity='O(n log n)',
            tags=['topology', 'duplicate', 'connectivity', 'volumetric', '3d']
        )

    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        """Execute duplicate cell detection on a volumetric mesh.

        Args:
            mesh: Input mesh with volumetric cells.

        Returns:
            List with a single MeshIssue if duplicates were found,
            otherwise an empty list.
        """
        if not mesh.has_volume_cells:
            return []

        seen: Dict[Tuple[int, ...], int] = {}   # canonical key -> first cell idx
        type_spans: List[Tuple[str, int, int]] = []  # (element_type, start, count)
        duplicates: List[Tuple[int, Tuple[int, ...], int]] = []

        global_idx = 0
        for element_type, connectivity in mesh.volume_cells.items():
            type_spans.append((element_type, global_idx, len(connectivity)))
            for row in np.asarray(connectivity):
                key = tuple(sorted(int(v) for v in row))
                if key in seen:
                    duplicates.append((global_idx, key, seen[key]))
                else:
                    seen[key] = global_idx
                global_idx += 1

        if not duplicates:
            return []

        total_cells = global_idx
        unique_cells = len(seen)
        dup_count = len(duplicates)
        violation_rate = dup_count / max(total_cells, 1)
        severity = determine_severity(violation_rate)

        def _type_of(idx: int) -> str:
            for etype, start, count in type_spans:
                if start <= idx < start + count:
                    return etype
            return 'unknown'

        n_vertices = mesh.n_vertices
        locations = []
        for gidx, key, first_idx in duplicates[:self._MAX_LOCATIONS]:
            valid = [int(v) for v in key if 0 <= int(v) < n_vertices]
            centroid = None
            if valid:
                coords = mesh.vertices[valid]
                centroid = tuple(float(x) for x in np.mean(coords, axis=0))
            locations.append(IssueLocation(
                element_type='cell',
                element_index=gidx,
                coordinates=centroid,
                adjacent_elements={
                    'cell_vertices': list(key),
                    'element_type': _type_of(gidx),
                    'duplicate_of_cell': first_idx,
                }
            ))

        issue = MeshIssue(
            operation_id=self.metadata.operation_id,
            category=IssueCategory.TOPOLOGICAL,
            severity=severity,
            message=f'Detected {dup_count} duplicate volume cell(s) '
                    f'({violation_rate * 100:.2f}% of {total_cells} cells; '
                    f'{unique_cells} unique vertex sets). Duplicate cells waste '
                    f'memory and can confuse solvers.',
            locations=locations,
            metrics={
                'duplicate_cell_count': dup_count,
                'total_cell_count': total_cells,
                'unique_cell_count': unique_cells,
                'violation_rate': violation_rate,
            },
            metadata={
                'duplicate_group_count': len({key for _, key, _ in duplicates}),
            },
        )
        return [issue]


# Register operation
registry.register(DuplicateCellDetection())
