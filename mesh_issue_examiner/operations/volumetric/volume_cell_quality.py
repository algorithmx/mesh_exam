"""
Volume cell quality computation operation for volumetric (3D) meshes.

Computes quality metrics for 3D elements (tetrahedra, hexahedra,
prisms, pyramids) such as Jacobian ratio, scaled Jacobian,
volume-length ratio, and condition number.
"""

from typing import List, Dict, Optional

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


class VolumeCellQualityComputation(BaseOperation):
    """
    Computes quality metrics for volumetric (3D) cells.

    Dual of CellQualityComputation: where the 2D operation computes
    triangle/quad quality, this computes tet/hex/prism/pyramid quality.

    Supported metrics:
        - scaled_jacobian: scaled Jacobian determinant
        - volume: cell volume (detects zero/negative volumes)
        - aspect_ratio: ratio of longest to shortest edge
        - condition: condition number of the Jacobian matrix
        - skewness: deviation from ideal cell shape
        - skew: same as skewness (alias)
        - orthogonality: face normal orthogonality (hex/prism)
    """

    def __init__(
        self,
        config: Optional[OperationConfig] = None,
        quality_types: Optional[List[str]] = None
    ):
        super().__init__(config)
        self.quality_types = quality_types or [
            'scaled_jacobian', 'volume', 'aspect_ratio',
            'condition', 'shape',
        ]

        # Default thresholds for 3D elements
        self.thresholds = {
            'scaled_jacobian': 0.1,
            'volume': 1e-12,
            'aspect_ratio': 0.1,
            'condition': 100.0,
            'shape': 0.3,
            'skewness': 0.8,
            'skew': 0.8,
            'orthogonality': 0.15,
        }

        # Metrics where higher values are worse (violation if value > threshold)
        self._higher_is_worse = {'condition'}

    @property
    def metadata(self) -> OperationMetadata:
        return OperationMetadata(
            operation_id='quality_3d.compute_volume_cell_quality',
            display_name='Volume Cell Quality Computation (3D)',
            description='Computes quality metrics for volumetric cells '
                       '(tet/hex/prism/pyramid) including scaled Jacobian, '
                       'volume, and aspect ratio.',
            category=IssueCategory.QUALITY,
            supported_libraries=['pyvista'],
            recommended_library='pyvista',
            requires_adjacency=False,
            complexity='O(n)',
            tags=['quality', 'shape', 'volumetric', '3d', 'jacobian', 'volume']
        )

    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        """Execute volume cell quality computation.

        Returns an empty list if the mesh has no volume cells.
        Otherwise converts to a PyVista UnstructuredGrid and computes
        each configured quality metric, reporting cells that fall below
        the corresponding threshold.
        """
        if not mesh.has_volume_cells:
            return []

        from ...adapters.factory import get_adapter
        adapter = get_adapter('pyvista')
        grid = mesh.get_native('pyvista')

        issues: List[MeshIssue] = []

        for quality_type in self.quality_types:
            try:
                values = adapter.compute_cell_quality(grid, quality_type)
            except Exception:
                # Skip metrics that this VTK version cannot compute
                continue

            if len(values) == 0:
                continue

            threshold = self.thresholds.get(quality_type)
            if threshold is None:
                continue

            # Filter out VTK sentinel value -1 (undefined metric)
            valid_mask = values != -1.0
            valid_values = values[valid_mask]
            if len(valid_values) == 0:
                continue

            # Determine violations based on metric direction
            if quality_type in self._higher_is_worse:
                violation_mask = valid_values > threshold
            else:
                violation_mask = valid_values < threshold
            violations = int(np.sum(violation_mask))
            total = len(valid_values)

            if violations == 0:
                continue

            violation_rate = violations / total
            severity = self._determine_severity(violation_rate)

            # Collect locations of violating cells, capped at 100
            violation_indices = np.where(violation_mask)[0]
            locations = [
                IssueLocation(element_type='cell', element_index=int(i))
                for i in violation_indices[:100]
            ]

            issue = MeshIssue(
                operation_id=f'{self.metadata.operation_id}.{quality_type}',
                category=IssueCategory.QUALITY,
                severity=severity,
                message=(
                    f'{quality_type}: {violations}/{total} cells '
                    f'({violation_rate * 100:.1f}%) below threshold {threshold}'
                ),
                metrics={
                    'violation_count': violations,
                    'total_cells': total,
                    'violation_rate': violation_rate,
                    'mean': float(np.mean(valid_values)),
                    'min': float(np.min(valid_values)),
                    'max': float(np.max(valid_values)),
                    'std': float(np.std(valid_values)),
                    'threshold': threshold,
                },
                metadata={
                    'quality_type': quality_type,
                    'p5': float(np.percentile(valid_values, 5)),
                    'p95': float(np.percentile(valid_values, 95)),
                },
                locations=locations,
            )
            issues.append(issue)

        return issues

    def _determine_severity(self, violation_rate: float) -> IssueSeverity:
        """Determine severity based on violation rate."""
        if violation_rate > 0.5:
            return IssueSeverity.CRITICAL
        elif violation_rate > 0.2:
            return IssueSeverity.ERROR
        elif violation_rate > 0.05:
            return IssueSeverity.WARNING
        else:
            return IssueSeverity.INFO


# Register operation
registry.register(VolumeCellQualityComputation())
