"""
Volume cell quality computation operation for volumetric (3D) meshes.

Computes quality metrics for 3D elements (tetrahedra, hexahedra,
prisms, pyramids) such as Jacobian ratio, scaled Jacobian,
volume-length ratio, and condition number.

Extended Quality Metrics (added 2026-04):
    - warpage: deviation of quadrilateral faces from planarity
    - equiangle_skew: angular deviation from ideal (CFD standard)
    - stretch: ratio of min edge to max diagonal
    - taper: ratio of opposite face areas (hex)
    - radius_ratio: circumradius / inradius (tets)
    - collapse_ratio: minimum altitude / max edge
    - min_dihedral_angle: minimum angle between adjacent faces
    - max_dihedral_angle: maximum angle between adjacent faces
"""

import logging
from typing import List, Dict, Optional, Tuple

import numpy as np

from ...core.models import (
    MeshInput,
    MeshIssue,
    IssueLocation,
    IssueCategory,
    IssueSeverity,
    OperationConfig,
    QualityMetrics,
)
from ..base import BaseOperation, OperationMetadata
from ..operation_registry import registry
from ...adapters.exceptions import MeshConversionError

logger = logging.getLogger(__name__)


# Extended quality types with new metrics
EXTENDED_QUALITY_TYPES = [
    # Existing metrics
    'scaled_jacobian', 'volume', 'aspect_ratio', 'condition', 'shape',
    'skewness', 'skew', 'orthogonality',
    # New metrics - Phase 1
    'warpage', 'equiangle_skew', 'stretch',
    # New metrics - Phase 2
    'taper', 'radius_ratio', 'collapse_ratio',
    # New metrics - Phase 3 (computed, not via VTK)
    # 'min_dihedral_angle', 'max_dihedral_angle',
]

# Quadratic element types whose VTK equivalents support quality
# computation (corner-first node ordering; pyramid14 has no VTK cell).
_QUAD_VTK = ['tet10', 'hex20', 'hex27', 'prism15', 'wedge18', 'pyramid13']

# Element-specific metric applicability
# Maps metric name to list of element types that support it
METRIC_ELEMENT_SUPPORT: Dict[str, List[str]] = {
    # All element types (linear + quadratic)
    'scaled_jacobian': ['tet4', 'hex8', 'prism6', 'pyramid5'] + _QUAD_VTK,
    'volume': ['tet4', 'hex8', 'prism6', 'pyramid5'] + _QUAD_VTK,
    'aspect_ratio': ['tet4', 'hex8', 'prism6', 'pyramid5'] + _QUAD_VTK,
    'condition': ['tet4', 'hex8', 'prism6', 'pyramid5'] + _QUAD_VTK,
    'shape': ['tet4', 'hex8', 'prism6', 'pyramid5'] + _QUAD_VTK,
    'skewness': ['tet4', 'hex8', 'prism6', 'pyramid5'] + _QUAD_VTK,
    'skew': ['tet4', 'hex8', 'prism6', 'pyramid5'] + _QUAD_VTK,
    'stretch': ['tet4', 'hex8', 'prism6', 'pyramid5'] + _QUAD_VTK,
    'collapse_ratio': ['tet4', 'hex8', 'prism6', 'pyramid5'] + _QUAD_VTK,
    'equiangle_skew': ['tet4', 'hex8', 'prism6', 'pyramid5'] + _QUAD_VTK,
    'radius_ratio': ['tet4', 'tet10'],  # Primary for tets
    # Elements with quad faces
    'warpage': ['hex8', 'prism6', 'pyramid5',
                'hex20', 'hex27', 'prism15', 'wedge18', 'pyramid13'],
    'taper': ['hex8', 'hex20', 'hex27'],
    'orthogonality': ['hex8', 'prism6',
                      'hex20', 'hex27', 'prism15', 'wedge18'],
}

# Industry-standard thresholds for quality metrics
# Based on research from ANSYS, SimScale, FEA Academy
DEFAULT_THRESHOLDS = {
    # Existing metrics
    'scaled_jacobian': 0.1,
    'volume': 1e-12,
    'aspect_ratio': 0.1,
    'condition': 100.0,
    'shape': 0.3,
    'skewness': 0.8,
    'skew': 0.8,
    'orthogonality': 0.15,
    # New metrics - Phase 1
    'warpage': 0.5,           # < 0.5 good, > 0.7 poor (ANSYS)
    'equiangle_skew': 0.25,   # CFD: avg < 0.25, max < 0.9
    'stretch': 0.25,          # > 0.25 acceptable
    # New metrics - Phase 2
    'taper': 0.5,             # < 0.5 good
    'radius_ratio': 10.0,     # < 10 good, > 10 poor (tets)
    'collapse_ratio': 0.1,    # Context dependent
}


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
        
    Extended metrics (new):
        - warpage: deviation of quadrilateral faces from planarity (hex/prism/pyramid)
        - equiangle_skew: angular deviation from ideal (CFD standard)
        - stretch: ratio of minimum edge to maximum diagonal
        - taper: ratio of opposite face areas (hexahedra)
        - radius_ratio: ratio of circumradius to inradius (tetrahedra)
        - collapse_ratio: minimum altitude divided by maximum edge

    Element-specific applicability:
        - warpage: hex8, prism6, pyramid5 (elements with quad faces)
        - taper: hex8 (elements with opposite faces)
        - radius_ratio: tet4 (primary), extensible to others
        - Other metrics: all element types
    """

    def __init__(
        self,
        config: Optional[OperationConfig] = None,
        quality_types: Optional[List[str]] = None,
        skip_unsupported_metrics: bool = True,
    ):
        super().__init__(config)
        self.quality_types = quality_types or [
            'scaled_jacobian', 'volume', 'aspect_ratio',
            'condition', 'shape',
        ]
        self.skip_unsupported_metrics = skip_unsupported_metrics

        # Default thresholds for 3D elements
        self.thresholds = DEFAULT_THRESHOLDS.copy()

    # Metrics where higher values are worse (violation if value > threshold)
    _higher_is_worse = {
        'condition', 'warpage', 'equiangle_skew', 'taper', 'radius_ratio'
    }

    @property
    def metadata(self) -> OperationMetadata:
        return OperationMetadata(
            operation_id='quality_3d.compute_volume_cell_quality',
            display_name='Volume Cell Quality Computation (3D)',
            description='Computes quality metrics for volumetric cells '
                       '(tet/hex/prism/pyramid) including scaled Jacobian, '
                       'volume, aspect ratio, warpage, equiangle skew, and more.',
            category=IssueCategory.QUALITY,
            supported_libraries=['pyvista'],
            recommended_library='pyvista',
            requires_adjacency=False,
            complexity='O(n)',
            tags=['quality', 'shape', 'volumetric', '3d', 'jacobian', 'volume',
                  'warpage', 'skew', 'taper', 'stretch']
        )

    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        """Execute volume cell quality computation.

        Returns an empty list if the mesh has no volume cells.
        Otherwise converts to a PyVista UnstructuredGrid and computes
        each configured quality metric, reporting cells that fall below
        the corresponding threshold.
        
        Metrics that are not applicable to the mesh's element types
        are skipped with a warning (if skip_unsupported_metrics=True).

        Every computed metric is also surfaced as a QualityMetrics
        distribution via ``self.last_quality_metrics`` (including
        zero-violation metrics), so the mesh_get_quality_metrics
        endpoint returns live data for volumetric meshes.
        """
        self.last_quality_metrics = []
        if not mesh.has_volume_cells:
            return []

        from ...adapters.factory import get_adapter
        adapter = get_adapter('pyvista')
        try:
            grid = mesh.get_native('pyvista')
        except MeshConversionError as exc:
            # Mesh contains element types with no VTK equivalent (e.g.
            # pyramid14); volume quality metrics need native computation.
            logger.warning(
                'Skipping volume cell quality metrics (no VTK equivalent '
                'for some element types): %s', exc,
            )
            return []

        issues: List[MeshIssue] = []
        
        # Get mesh element types
        mesh_element_types = list(mesh.volume_cells.keys())

        for quality_type in self.quality_types:
            # Check if metric is applicable to this mesh
            if self.skip_unsupported_metrics:
                if not self._is_metric_applicable(quality_type, mesh_element_types):
                    logger.debug(
                        f"Skipping metric '{quality_type}' - not applicable to "
                        f"mesh element types: {mesh_element_types}"
                    )
                    continue
            
            # Compute metric with error handling
            values = self._compute_metric_safe(adapter, grid, quality_type)
            
            if values is None:
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
                    f'({violation_rate * 100:.1f}%) violate threshold {threshold}'
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

    def _is_metric_applicable(
        self, 
        metric: str, 
        mesh_element_types: List[str]
    ) -> bool:
        """Check if a quality metric is applicable to mesh element types.
        
        Args:
            metric: Quality metric name
            mesh_element_types: List of element types in the mesh
            
        Returns:
            True if metric applies to any mesh element type
        """
        supported_elements = METRIC_ELEMENT_SUPPORT.get(metric, [])
        
        # Metric applies if any mesh element type is in supported list
        return any(elem in supported_elements for elem in mesh_element_types)

    def _compute_metric_safe(
        self, 
        adapter, 
        grid, 
        quality_type: str
    ) -> Optional[np.ndarray]:
        """Compute a quality metric with graceful error handling.
        
        Args:
            adapter: Mesh adapter instance
            grid: PyVista grid object
            quality_type: Quality metric name
            
        Returns:
            Array of quality values, or None if computation failed
        """
        try:
            values = adapter.compute_cell_quality(grid, quality_type)
            
            if len(values) == 0:
                return None
                
            # VTK returns -1 for undefined metrics
            if np.all(values == -1.0):
                logger.debug(
                    f"Metric '{quality_type}' returned all -1 (unsupported for this mesh)"
                )
                return None
                
            return values
            
        except Exception as e:
            logger.debug(f"Failed to compute metric '{quality_type}': {e}")
            return None

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
