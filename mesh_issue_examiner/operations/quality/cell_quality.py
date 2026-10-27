"""
Cell quality computation operation.

Computes various element shape quality metrics using PyVista/VTK.
"""

from typing import List, Dict, Optional, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)

from ...core.models import (
    MeshInput,
    MeshIssue,
    QualityMetrics,
    IssueCategory,
    IssueSeverity,
    OperationConfig,
)
from ...utils.metrics import compute_percentile_stats
from ..base import BaseOperation, OperationMetadata
from ..operation_registry import registry
from ...adapters.exceptions import MeshConversionError


class CellQualityComputation(BaseOperation):
    """
    Computes various cell quality metrics.

    Supports: aspect_ratio, min_angle, max_angle, scaled_jacobian,
    condition, shape, area, distortion, radius_ratio.
    
    Note: 'skewness' is not valid for triangular meshes (VTK returns -1).
    Use 'scaled_jacobian' or 'shape' instead for triangle quality assessment.
    """

    def __init__(
        self,
        config: Optional[OperationConfig] = None,
        quality_types: Optional[List[str]] = None
    ):
        super().__init__(config)
        # Default quality types for surface meshes (triangles/polygons)
        # Note: 'skewness' is NOT valid for triangles (returns -1 from VTK)
        # Valid metrics for triangles: aspect_ratio, min_angle, max_angle,
        # scaled_jacobian, shape, area, condition, distortion, radius_ratio
        self.quality_types = quality_types or ['aspect_ratio', 'min_angle', 'max_angle']

        # Default thresholds for surface mesh quality metrics
        # Note: aspect_ratio and max_angle are in HIGHER_IS_WORSE (checked > threshold)
        # min_angle and scaled_jacobian are checked < threshold
        self.thresholds = {
            'aspect_ratio': 10.0,      # aspect_ratio > 10 is poor quality
            'min_angle': 15.0,         # degrees, angles below this are poor
            'max_angle': 120.0,        # degrees, angles above this are poor
            'scaled_jacobian': 0.5,    # scaled_jacobian < 0.5 is poor
            'condition': 100.0,        # condition number > 100 is poor
            'shape': 0.25,             # shape metric (1=ideal, <0.25 poor)
            'area': 1e-10,             # near-zero area is degenerate
            'volume': 1e-10,           # near-zero volume is degenerate
        }

    @property
    def metadata(self) -> OperationMetadata:
        return OperationMetadata(
            operation_id='quality.compute_cell_quality',
            display_name='Cell Quality Computation',
            description='Computes element shape quality metrics including '
                       'aspect ratio, min/max angles, and scaled Jacobian.',
            category=IssueCategory.QUALITY,
            supported_libraries=['pyvista', 'trimesh'],
            recommended_library='pyvista',
            requires_adjacency=False,
            complexity='O(n)',
            tags=['quality', 'shape', 'aspect_ratio', 'skewness', 'angles']
        )

    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        """Execute cell quality computation."""
        from ...adapters.factory import get_adapter
        
        # Respect config.library_preference if set, otherwise use recommended library
        preferred_library = self.config.library_preference
        if preferred_library is not None and preferred_library in self.metadata.supported_libraries:
            # Use user-specified preferred library
            adapter = get_adapter(preferred_library)
        else:
            # Use recommended library (pyvista) with fallback to trimesh
            try:
                adapter = get_adapter(self.metadata.recommended_library)
            except ImportError:
                # Fallback: try supported libraries in order
                for lib in self.metadata.supported_libraries:
                    try:
                        adapter = get_adapter(lib)
                        break
                    except ImportError:
                        continue
                else:
                    raise ImportError(
                        f"None of the supported libraries are available: {self.metadata.supported_libraries}"
                    )
        
        # Some element types (e.g. 'pyramid14') have no VTK equivalent;
        # skip gracefully rather than failing the whole inspection.
        try:
            native_mesh = adapter.from_canonical(mesh)
        except MeshConversionError as e:
            logger.warning(f"Skipping cell quality computation: {e}")
            self.last_quality_metrics = []
            return []

        self.last_quality_metrics = []
        issues = []

        # Metrics where higher values are worse (violation if value > threshold)
        HIGHER_IS_WORSE = {'max_angle', 'condition', 'aspect_ratio'}
        
        for quality_type in self.quality_types:
            try:
                values = adapter.compute_cell_quality(native_mesh, quality_type)

                if len(values) == 0:
                    continue

                threshold = self.thresholds.get(quality_type)
                # Filter out undefined values (-1 is VTK's default for undefined quality)
                valid_values = values[values != -1.0] if len(values) > 0 else values
                
                # Determine violations based on metric type
                if quality_type in HIGHER_IS_WORSE:
                    # For these metrics, higher values are worse (e.g., max_angle > 140)
                    violations = int(np.sum(valid_values > threshold)) if threshold is not None else 0
                    comparison = 'above'
                else:
                    # For these metrics, lower values are worse (e.g., min_angle < 20)
                    violations = int(np.sum(valid_values < threshold)) if threshold is not None else 0
                    comparison = 'below'
                
                total = len(valid_values)
                violation_rate = violations / total if total > 0 else 0

                # Always surface the metric distribution (even with zero
                # violations) so mesh_get_quality_metrics returns live data.
                qm = QualityMetrics(
                    operation_id=f'{self.metadata.operation_id}.{quality_type}',
                    metric_name=quality_type,
                    values=valid_values,
                    threshold=threshold,
                )
                # Authoritative count consistent with issue reporting
                # (handles metric direction quirks uniformly).
                qm.violation_count = violations
                self.last_quality_metrics.append(qm)

                # Determine severity
                severity = self._determine_severity(violation_rate)

                # For 2D min_angle, cap severity at WARNING (never ERROR/CRITICAL)
                # and include detailed percentile statistics
                is_min_angle = quality_type == 'min_angle'

                if is_min_angle and severity in (IssueSeverity.ERROR, IssueSeverity.CRITICAL):
                    severity = IssueSeverity.WARNING

                # Build percentile stats for min_angle (or reuse p5/p95 for others)
                if is_min_angle:
                    pct_stats = compute_percentile_stats(
                        valid_values,
                        percentiles=[1, 5, 10, 20, 50],
                        higher_is_worse=False,
                    )
                else:
                    pct_stats = {
                        'p5': float(np.percentile(values, 5)),
                        'p95': float(np.percentile(values, 95)),
                    }

                if violations > 0:
                    if is_min_angle:
                        # Show percentile breakdown instead of simple violation count
                        pct_lines = ', '.join(
                            f'{k}={v:.2f}' for k, v in pct_stats.items()
                        )
                        message = (
                            f'{quality_type}: {violations}/{total} cells '
                            f'({violation_rate * 100:.1f}%) {comparison} threshold {threshold}. '
                            f'Worst-angle percentiles: {pct_lines}'
                        )
                    else:
                        message = (
                            f'{quality_type}: {violations}/{total} cells '
                            f'({violation_rate * 100:.1f}%) {comparison} threshold {threshold}'
                        )

                    issue = MeshIssue(
                        operation_id=f'{self.metadata.operation_id}.{quality_type}',
                        category=IssueCategory.QUALITY,
                        severity=severity,
                        message=message,
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
                            **pct_stats,
                        }
                    )
                    issues.append(issue)

            except Exception as e:
                logger.warning("Skipping quality metric %r: %s", quality_type, e)
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
