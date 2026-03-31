"""
Cell quality computation operation.

Computes various element shape quality metrics using PyVista/VTK.
"""

from typing import List, Dict, Optional, Any
import numpy as np

from ...core.models import (
    MeshInput,
    MeshIssue,
    QualityMetrics,
    IssueCategory,
    IssueSeverity,
    OperationConfig,
)
from ..base import BaseOperation, OperationMetadata
from ..operation_registry import registry


class CellQualityComputation(BaseOperation):
    """
    Computes various cell quality metrics.

    Supports: aspect_ratio, skewness, min_angle, max_angle,
    scaled_jacobian, condition, Jacobian, volume.
    """

    def __init__(
        self,
        config: Optional[OperationConfig] = None,
        quality_types: Optional[List[str]] = None
    ):
        super().__init__(config)
        self.quality_types = quality_types or ['aspect_ratio', 'skewness', 'min_angle']

        # Default thresholds
        self.thresholds = {
            'aspect_ratio': 0.3,
            'skewness': 0.5,
            'min_angle': 20.0,
            'max_angle': 140.0,
            'scaled_jacobian': 0.1,
            'condition': 10.0,
            'Jacobian': 0.0,
            'volume': 1e-10,
        }

    @property
    def metadata(self) -> OperationMetadata:
        return OperationMetadata(
            operation_id='quality.compute_cell_quality',
            display_name='Cell Quality Computation',
            description='Computes element shape quality metrics including '
                       'aspect ratio, skewness, and angle measures.',
            category=IssueCategory.QUALITY,
            supported_libraries=['pyvista', 'trimesh'],
            recommended_library='pyvista',
            requires_adjacency=False,
            complexity='O(n)',
            tags=['quality', 'shape', 'aspect_ratio', 'skewness', 'angles']
        )

    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        """Execute cell quality computation."""
        try:
            from ...adapters.factory import get_adapter
            adapter = get_adapter('pyvista')
            native_mesh = adapter.from_canonical(mesh)
        except ImportError:
            # Fallback to TRimesh
            from ...adapters.factory import get_adapter
            adapter = get_adapter('trimesh')
            native_mesh = adapter.from_canonical(mesh)

        issues = []

        for quality_type in self.quality_types:
            try:
                values = adapter.compute_cell_quality(native_mesh, quality_type)

                if len(values) == 0:
                    continue

                threshold = self.thresholds.get(quality_type)
                violations = int(np.sum(values < threshold)) if threshold is not None else 0
                total = len(values)
                violation_rate = violations / total if total > 0 else 0

                # Determine severity
                severity = self._determine_severity(violation_rate)

                if violations > 0:
                    issue = MeshIssue(
                        operation_id=f'{self.metadata.operation_id}.{quality_type}',
                        category=IssueCategory.QUALITY,
                        severity=severity,
                        message=f'{quality_type}: {violations}/{total} cells '
                               f'({violation_rate * 100:.1f}%) below threshold {threshold}',
                        metrics={
                            'violation_count': violations,
                            'total_cells': total,
                            'violation_rate': violation_rate,
                            'mean': float(np.mean(values)),
                            'min': float(np.min(values)),
                            'max': float(np.max(values)),
                            'std': float(np.std(values)),
                            'threshold': threshold
                        },
                        metadata={
                            'quality_type': quality_type,
                            'p5': float(np.percentile(values, 5)),
                            'p95': float(np.percentile(values, 95))
                        }
                    )
                    issues.append(issue)

            except Exception as e:
                # Skip metrics that fail
                continue

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
registry.register(CellQualityComputation())
