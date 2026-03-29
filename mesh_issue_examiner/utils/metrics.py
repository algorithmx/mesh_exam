"""
Quality metric utilities and threshold definitions.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from core.models import MeshInput, QualityMetrics, IssueSeverity


# Default quality thresholds
DEFAULT_THRESHOLDS = {
    'aspect_ratio': 0.3,
    'skewness': 0.5,
    'min_angle': 20.0,  # degrees
    'max_angle': 140.0,  # degrees
    'scaled_jacobian': 0.1,
    'condition': 10.0,
    'Jacobian': 0.0,
    'volume': 1e-10,
    'area': 1e-10,
}


def get_default_thresholds() -> Dict[str, float]:
    """Get default thresholds for all quality metrics."""
    return DEFAULT_THRESHOLDS.copy()


def get_threshold(metric_name: str) -> float:
    """Get default threshold for a specific metric."""
    return DEFAULT_THRESHOLDS.get(metric_name, 0.1)


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
        metrics = ['aspect_ratio', 'skewness', 'min_angle', 'max_angle']

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
    total_elements = sum(len(m.total_count) if hasattr(m, 'total_count') else len(m.values) for m in metrics_list)
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
