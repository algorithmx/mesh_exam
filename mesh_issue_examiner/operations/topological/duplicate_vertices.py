"""
Duplicate vertex detection operation.

Identifies vertices that have identical or nearly identical coordinates,
which can cause topological issues and waste memory.
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
from ...utils.metrics import determine_severity


class DuplicateVertexDetection(BaseOperation):
    """
    Detects duplicate vertices in the mesh.

    Duplicate vertices are vertices that have the same or nearly identical
    coordinates. They can occur during mesh generation, file import, or
    merging operations. While not always a critical issue, they can:
    - Waste memory
    - Cause rendering artifacts
    - Affect topological analysis
    - Impact simulation accuracy

    The tolerance parameter controls how close vertices must be to be
    considered duplicates. A tolerance of 0.0 requires exact match.

    .. important::
        **LOADER BEHAVIOR NOTE:** Many mesh loaders automatically deduplicate
        vertices on load, which can affect this detection:

        - **trimesh** (default for STL, OBJ): Automatically merges duplicate
          vertices. Duplicate vertex detection will always return 0 when using
          the default trimesh loader.
        
        - **PyVista**: Preserves original vertex data. Duplicate vertex detection
          works correctly when meshes are loaded via PyVista or created manually.
        
        - **meshio**: May deduplicate depending on format and options.

        To reliably detect duplicate vertices from files, use VTK format or
        ensure the loader preserves exact vertex data (e.g., use PyVista adapter
        directly via ``load_mesh_from_file(..., library='pyvista')``).
    """

    def __init__(
        self,
        config: Optional[OperationConfig] = None,
        tolerance: float = 1e-6
    ):
        """
        Initialize duplicate vertex detection.

        Args:
            config: Operation configuration
            tolerance: Distance threshold for considering vertices duplicate.
                      Vertices closer than this distance are considered duplicates.
                      Default is 1e-6 (suitable for most unit-scale meshes).
        """
        super().__init__(config)
        self.tolerance = tolerance

    @property
    def metadata(self) -> OperationMetadata:
        return OperationMetadata(
            operation_id='topology.detect_duplicate_vertices',
            display_name='Duplicate Vertex Detection',
            description='Identifies vertices with identical or nearly identical '
                       'coordinates that may indicate mesh cleanup is needed.',
            category=IssueCategory.TOPOLOGICAL,
            supported_libraries=['trimesh', 'pyvista', 'pymeshlab', 'meshio', 'vedo'],
            recommended_library='pyvista',
            requires_adjacency=False,
            complexity='O(n log n)',
            tags=['topology', 'cleanup', 'vertices', 'duplicate', 'weld']
        )

    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        """Execute duplicate vertex detection."""
        from ...adapters.factory import get_adapter

        # Get adapter for duplicate vertex detection
        try:
            adapter = get_adapter(self.metadata.recommended_library)
        except ImportError:
            # Try other supported libraries
            for lib in self.metadata.supported_libraries:
                try:
                    adapter = get_adapter(lib)
                    break
                except ImportError:
                    continue
            else:
                return []

        # Convert to native mesh
        native_mesh = adapter.from_canonical(mesh)

        # Find duplicate vertices
        duplicate_indices = adapter.get_duplicate_vertices(native_mesh, self.tolerance)

        if not duplicate_indices:
            return []

        # Create issue locations for each duplicate vertex
        locations = []
        for idx in duplicate_indices:
            if idx < len(mesh.vertices):
                coords = tuple(mesh.vertices[idx])
                locations.append(IssueLocation(
                    element_type='vertex',
                    element_index=idx,
                    coordinates=coords,
                    adjacent_elements={}
                ))

        # Determine severity based on percentage of duplicates
        duplicate_rate = len(duplicate_indices) / max(len(mesh.vertices), 1)
        severity = determine_severity(duplicate_rate)

        # Adjust severity - duplicate vertices are usually not critical
        if severity == IssueSeverity.CRITICAL and duplicate_rate < 0.5:
            severity = IssueSeverity.ERROR

        issue = MeshIssue(
            operation_id=self.metadata.operation_id,
            category=IssueCategory.TOPOLOGICAL,
            severity=severity,
            message=f'Detected {len(duplicate_indices)} duplicate vertices '
                   f'({duplicate_rate * 100:.2f}% of total). '
                   f'Vertices within tolerance {self.tolerance} are considered duplicates.',
            locations=locations,
            metrics={
                'count': len(duplicate_indices),
                'duplicate_rate': duplicate_rate,
                'total_vertices': len(mesh.vertices),
                'tolerance': self.tolerance
            },
            metadata={
                'unique_vertices': len(mesh.vertices) - len(duplicate_indices)
            }
        )

        return [issue]


# Register operation
registry.register(DuplicateVertexDetection())
