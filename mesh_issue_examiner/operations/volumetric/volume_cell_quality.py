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
from ._face_tables import (
    ELEMENT_CORNER_COUNT, LINEAR_BASE_TYPE, QUADRATIC_EDGE_MIDSIDES,
    linearized_volume_cells,
)

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

# Quadratic element types.  vtkCellQuality cannot evaluate them
# directly (it returns its -1 sentinel), so quality is computed on the
# corner-only linear equivalents (see _linearized_volume_cells);
# pyramid14 has no VTK cell at all and reaches the same path through
# its pyramid5 corners.
_QUAD_VTK = ['tet10', 'hex20', 'hex27', 'prism15', 'wedge18',
             'pyramid13', 'pyramid14']


# Backwards-compatible alias: the helper lived in this module before it
# moved to _face_tables.py so geometric validity could share it.
_linearized_volume_cells = linearized_volume_cells

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
                'hex20', 'hex27', 'prism15', 'wedge18',
                'pyramid13', 'pyramid14'],
    'taper': ['hex8', 'hex20', 'hex27'],
    'orthogonality': ['hex8', 'prism6',
                      'hex20', 'hex27', 'prism15', 'wedge18'],
    # Pure-quadratic diagnostic (not computable on linear cells)
    'midside_deviation': list(_QUAD_VTK),
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
    # Dimensionless: max midside-node offset from its corner-edge
    # midpoint, normalised by the cell's longest corner edge.
    'midside_deviation': 0.1,  # < 0.1 straight-ish; > 0.1 curved
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
        - midside_deviation: relative midside-node curvature of quadratic
          cells (diagnostic for how far a quadratic cell is from its
          corner-linearised stand-in; pure numpy, quadratic cells only)

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
        # Global cell indices of the last midside_deviation computation
        # (issue locations map through it); empty when not applicable.
        self._midside_cell_indices = np.empty(0, dtype=np.int64)

        # Default thresholds for 3D elements
        self.thresholds = DEFAULT_THRESHOLDS.copy()

    # Metrics where higher values are worse (violation if value > threshold)
    _higher_is_worse = {
        'condition', 'warpage', 'equiangle_skew', 'taper', 'radius_ratio',
        'midside_deviation',
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
                  'warpage', 'skew', 'taper', 'stretch', 'midside', 'curvature']
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

        linearized_cells = linearized_volume_cells(mesh)
        grid = None
        try:
            grid = mesh.get_native('pyvista')
        except MeshConversionError as exc:
            if linearized_cells is None:
                # Mesh contains element types with no VTK equivalent and
                # no linear corner fallback; quality needs native support.
                logger.warning(
                    'Skipping volume cell quality metrics (no VTK '
                    'equivalent for some element types): %s', exc,
                )
                return []

        # vtkCellQuality cannot evaluate quadratic cells (VTK returns
        # its -1 sentinel for every quadratic cell), so whenever the
        # mesh contains quadratic types the metrics are computed on a
        # corner-only linearised grid instead.  This also unblocks
        # pyramid14 (no VTK cell), whose pyramid5 corners carry the
        # metric.  The linearised blocks preserve the original cell
        # ordering, so values attribute to the original cells directly.
        quality_grid = grid
        linearized = False
        if linearized_cells is not None:
            try:
                lin_mesh = MeshInput(
                    vertices=mesh.vertices,
                    faces=mesh.faces,
                    mesh_id=f'{mesh.mesh_id}_corner_linearized',
                    volume_cells=linearized_cells,
                )
                quality_grid = lin_mesh.get_native('pyvista')
                linearized = True
            except MeshConversionError as exc:
                if grid is None:
                    logger.warning(
                        'Skipping volume cell quality metrics (native and '
                        'linearised conversion both failed): %s', exc,
                    )
                    return []
                quality_grid = grid

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

            # midside_deviation reads the quadratic connectivity directly
            # (pure numpy); it must NOT run through vtkCellQuality, which
            # returns its -1 sentinel for every quadratic cell.
            if quality_type == 'midside_deviation':
                values = self._compute_midside_deviation(mesh)
            else:
                # Compute metric with error handling
                values = self._compute_metric_safe(adapter, quality_grid,
                                                   quality_type)
            
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
            if quality_type == 'midside_deviation':
                # Not a linearised value: this metric *measures* the
                # quadratic geometry the linearisation discards.
                qm.metadata['quadratic_only'] = True
                qm.metadata['quadratic_types'] = sorted(
                    t for t in mesh.volume_cells
                    if t in QUADRATIC_EDGE_MIDSIDES)
            elif linearized:
                # Mark corner-linearised provenance: midside curvature
                # is ignored, values approximate the quadratic cells.
                qm.metadata['linearized'] = True
                qm.metadata['linearized_types'] = sorted(
                    t for t in mesh.volume_cells if t in LINEAR_BASE_TYPE)
            self.last_quality_metrics.append(qm)

            if violations == 0:
                continue

            violation_rate = violations / total
            severity = self._determine_severity(violation_rate)

            # Collect locations of violating cells, capped at 100.
            # midside_deviation values are per-quadratic-cell; map their
            # positions back to the mesh's global cell indices.
            violation_positions = np.where(violation_mask)[0]
            if quality_type == 'midside_deviation':
                violation_indices = self._midside_cell_indices[violation_positions]
            else:
                violation_indices = violation_positions
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
                    **({'linearized': True}
                       if linearized and quality_type != 'midside_deviation'
                       else {}),
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

    def _compute_midside_deviation(
        self, mesh: MeshInput
    ) -> Optional[np.ndarray]:
        """Relative midside-node curvature of quadratic cells (pure numpy).

        For every quadratic cell, the deviation of each mid-edge node from
        the midpoint of its corner-edge is measured and normalised by the
        cell's longest corner-edge length (dimensionless, scale
        invariant).  The per-cell value is the maximum over its edges:
        0.0 for perfectly straight cells, growing as the midside nodes
        bow away from the straight edge — exactly the geometric effect the
        corner-linearised metrics discard, so large values flag cells
        where those linearised metrics are least trustworthy.

        Face-centre and body-centre nodes (hex27 slots 20-26, wedge18
        slots 15-17, pyramid14 slot 13) are excluded via
        ``QUADRATIC_EDGE_MIDSIDES``.

        Args:
            mesh: MeshInput with volume_cells.

        Returns:
            Array with one value per *quadratic* cell (global cell order);
            ``self._midside_cell_indices`` holds their global cell indices.
            None when the mesh has no supported quadratic types.
        """
        self._midside_cell_indices = np.empty(0, dtype=np.int64)
        quad = {
            t: conn for t, conn in mesh.volume_cells.items()
            if t in QUADRATIC_EDGE_MIDSIDES
        }
        if not quad:
            return None

        try:
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            values: List[np.ndarray] = []
            global_indices: List[np.ndarray] = []
            gidx = 0
            for etype, conn in mesh.volume_cells.items():
                conn = np.asarray(conn, dtype=np.int64)
                edges = QUADRATIC_EDGE_MIDSIDES.get(etype)
                if edges is None:
                    gidx += len(conn)
                    continue

                a = conn[:, [e[0] for e in edges]]
                b = conn[:, [e[1] for e in edges]]
                m = conn[:, [e[2] for e in edges]]

                pa = vertices[a]
                pb = vertices[b]
                pm = vertices[m]

                mid_dev = np.linalg.norm(pm - 0.5 * (pa + pb), axis=2)
                edge_len = np.linalg.norm(pa - pb, axis=2)
                scale = edge_len.max(axis=1)
                # Degenerate cells (all zero-length edges): keep 0.0.
                scale = np.where(scale > 0.0, scale, 1.0)

                values.append(mid_dev.max(axis=1) / scale)
                global_indices.append(gidx + np.arange(len(conn)))
                gidx += len(conn)

            self._midside_cell_indices = np.concatenate(global_indices)
            return np.concatenate(values)
        except Exception as e:
            logger.debug("Failed to compute midside_deviation: %s", e)
            return None

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
