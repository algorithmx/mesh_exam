"""
Quality metric utilities and threshold definitions.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from ..core.models import MeshInput, QualityMetrics, IssueSeverity


# Default quality thresholds (surface meshes - triangles/polygons)
# Note: 'skewness' is NOT valid for triangular meshes (VTK returns -1)
# Use 'scaled_jacobian' or 'shape' instead for triangle quality assessment
#
# Threshold interpretation:
# - Metrics where lower is worse (violation if value < threshold):
#   min_angle, scaled_jacobian, shape, area
# - Metrics where higher is worse (violation if value > threshold):
#   max_angle, aspect_ratio, condition
#
# Based on VTK's acceptable ranges for TRIANGLE cells:
# - aspect_ratio: acceptable (1.0, 1.3), normal (1.0, inf)
# - scaled_jacobian: acceptable (0.5, 1.15), normal (-1.15, 1.15)
# - min_angle: ideal=60 for triangle, 90 for quad
# - max_angle: ideal=60 for triangle, 90 for quad
DEFAULT_THRESHOLDS = {
    'aspect_ratio': 10.0,      # aspect_ratio > 10 is poor quality (1=ideal)
    'min_angle': 15.0,         # degrees, angles below this are poor
    'max_angle': 120.0,        # degrees, angles above this are poor
    'scaled_jacobian': 0.5,    # scaled_jacobian < 0.5 is poor (1=ideal)
    'condition': 100.0,        # condition number > 100 is poor
    'shape': 0.25,             # shape metric (1=ideal, <0.25 poor)
    'area': 1e-10,             # near-zero area is degenerate
    'distortion': 0.5,         # distortion metric (1=ideal, <0.5 poor)
    'radius_ratio': 2.0,       # radius ratio (1=ideal, >2 poor)
}

# Volumetric (3D) element quality thresholds
VOLUMETRIC_THRESHOLDS = {
    'scaled_jacobian': 0.1,    # <=0 invalid, <0.1 poor
    'volume': 1e-12,           # near-zero = degenerate
    'aspect_ratio': 0.1,      # VTK convention (1=ideal, 0=degenerate)
    'condition': 100.0,       # Condition number (lower = better)
    'skewness': 0.8,          # Equiangular skewness (>0.8 = poor)
    'orthogonality': 0.15,    # Face normal orthogonality
    'shape': 0.3,             # Shape metric (1=ideal)
    'distortion': 0.5,        # Distortion metric (1=ideal)
    'radius_ratio': 3.0,      # Radius ratio (1=ideal)
    'collapse_ratio': 0.1,    # Collapse ratio
    'degenerate_volume': 1e-12,  # Threshold for degenerate cell detection
    'sliver_ratio': 1e-6,     # Volume / max_edge^3 ratio for sliver detection
}


def get_default_thresholds() -> Dict[str, float]:
    """Get default thresholds for all quality metrics."""
    return DEFAULT_THRESHOLDS.copy()


def get_volumetric_thresholds() -> Dict[str, float]:
    """Get default thresholds for volumetric (3D) quality metrics."""
    return VOLUMETRIC_THRESHOLDS.copy()


def get_threshold(metric_name: str) -> float:
    """Get default threshold for a specific metric."""
    return DEFAULT_THRESHOLDS.get(
        metric_name,
        VOLUMETRIC_THRESHOLDS.get(metric_name, 0.1)
    )


def compute_all_quality_metrics(
    mesh: MeshInput,
    adapter: Any,
    metrics: Optional[List[str]] = None,
    thresholds: Optional[Dict[str, float]] = None
) -> List[QualityMetrics]:
    """
    Compute multiple quality metrics for a mesh.

    Args:
        mesh: Input mesh
        adapter: Library adapter for computation
        metrics: List of metric names to compute
        thresholds: Optional threshold overrides

    Returns:
        List of QualityMetrics objects
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    if metrics is None:
        # Default metrics for surface meshes (valid for triangles)
        metrics = ['aspect_ratio', 'min_angle', 'max_angle', 'scaled_jacobian']

    native_mesh = adapter.from_canonical(mesh)
    results = []

    for metric in metrics:
        try:
            values = adapter.compute_cell_quality(native_mesh, metric)

            if len(values) == 0:
                continue

            threshold = thresholds.get(metric)

            quality_metrics = QualityMetrics(
                operation_id=f'quality.{metric}',
                metric_name=metric,
                values=values,
                threshold=threshold
            )

            results.append(quality_metrics)

        except Exception:
            # Skip metrics that fail
            continue

    return results


def compute_violation_stats(
    metrics_list: List[QualityMetrics]
) -> Dict[str, Any]:
    """
    Compute violation statistics from quality metrics.

    Args:
        metrics_list: List of QualityMetrics

    Returns:
        Dictionary with violation statistics
    """
    total_elements = sum(len(m.values) for m in metrics_list)
    total_violations = sum(m.violation_count for m in metrics_list)

    by_metric = {}
    for metrics in metrics_list:
        by_metric[metrics.metric_name] = {
            'violation_count': metrics.violation_count,
            'total_count': len(metrics.values),
            'violation_rate': metrics.violation_count / len(metrics.values) if len(metrics.values) > 0 else 0,
            'mean': metrics.statistics.get('mean', 0),
            'threshold': metrics.threshold
        }

    return {
        'total_violations': total_violations,
        'total_elements': total_elements,
        'overall_violation_rate': total_violations / total_elements if total_elements > 0 else 0,
        'by_metric': by_metric
    }


def compute_percentile_stats(
    values: np.ndarray,
    percentiles: Optional[List[int]] = None,
    higher_is_worse: bool = False,
) -> Dict[str, float]:
    """Compute tail-percentile statistics for a quality metric.

    For *lower-is-worse* metrics (``higher_is_worse=False``, e.g. min_angle,
    scaled_jacobian) the "worst *p*%" values sit in the **lower** tail so the
    function returns the standard *p*-th percentiles.

    For *higher-is-worse* metrics (``higher_is_worse=True``, e.g. max_angle,
    aspect_ratio) the worst values sit in the **upper** tail so the function
    returns the *(100 - p)*-th percentiles (e.g. for p=5 → 95th percentile,
    meaning 5 % of cells exceed that value).

    Args:
        values: 1-D array of metric values (undefined/-1 values should be
            filtered out *before* calling this helper).
        percentiles: List of integer percentile levels to report.  Defaults to
            ``[1, 5, 10, 20, 50]`` (i.e. "worst 1 % … worst 50 %").
        higher_is_worse: ``True`` when large values indicate poor quality.

    Returns:
        Dict mapping ``'p{n}'`` → float value.  For example, with the
        default percentiles on a min_angle array the result would be::

            {'p1': 5.2, 'p5': 10.1, 'p10': 13.5, 'p20': 18.0, 'p50': 28.3}
    """
    if percentiles is None:
        percentiles = [1, 5, 10, 20, 50]

    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {f'p{p}': float('nan') for p in percentiles}

    result: Dict[str, float] = {}
    for p in percentiles:
        ptile = (100 - p) if higher_is_worse else p
        result[f'p{p}'] = float(np.percentile(arr, ptile))
    return result


def determine_severity(violation_rate: float) -> IssueSeverity:
    """
    Determine issue severity based on violation rate.

    Args:
        violation_rate: Fraction of elements below threshold (0-1)

    Returns:
        Appropriate IssueSeverity
    """
    if violation_rate > 0.5:
        return IssueSeverity.CRITICAL
    elif violation_rate > 0.2:
        return IssueSeverity.ERROR
    elif violation_rate > 0.05:
        return IssueSeverity.WARNING
    else:
        return IssueSeverity.INFO
