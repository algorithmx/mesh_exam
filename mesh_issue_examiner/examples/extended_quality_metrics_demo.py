"""
Demonstration of extended quality metrics (Gap #8 implementation).

This script demonstrates the new quality metrics added to the 
VolumeCellQualityComputation operation:
    - warpage
    - equiangle_skew
    - stretch
    - taper
    - radius_ratio
    - collapse_ratio
"""

import numpy as np
from mesh_issue_examiner.operations.volumetric.volume_cell_quality import (
    VolumeCellQualityComputation,
    METRIC_ELEMENT_SUPPORT,
    DEFAULT_THRESHOLDS,
)
from mesh_issue_examiner.core.models import MeshInput


def create_perfect_hex_cube():
    """Create a perfect hexahedral cube."""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=float)
    
    hex_cells = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    
    return MeshInput(
        vertices=vertices,
        faces=np.array([]),
        volume_cells={'hex8': hex_cells}
    )


def create_regular_tet():
    """Create a regular tetrahedron."""
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, np.sqrt(3)/2, 0],
        [0.5, np.sqrt(3)/6, np.sqrt(2/3)]
    ], dtype=float)
    
    tet_cells = np.array([[0, 1, 2, 3]])
    
    return MeshInput(
        vertices=vertices,
        faces=np.array([]),
        volume_cells={'tet4': tet_cells}
    )


def print_metric_info():
    """Print information about the extended metrics."""
    print("=" * 70)
    print("EXTENDED QUALITY METRICS INFORMATION")
    print("=" * 70)
    
    print("\n📊 New Metrics Added:")
    print("-" * 50)
    new_metrics = ['warpage', 'equiangle_skew', 'stretch', 'taper', 'radius_ratio', 'collapse_ratio']
    
    for metric in new_metrics:
        supported = METRIC_ELEMENT_SUPPORT.get(metric, [])
        threshold = DEFAULT_THRESHOLDS.get(metric, 'N/A')
        print(f"\n  • {metric}:")
        print(f"    - Supported elements: {', '.join(supported)}")
        print(f"    - Threshold: {threshold}")
    
    print("\n" + "=" * 70)


def demo_hex_quality():
    """Demonstrate quality metrics on a hex mesh."""
    print("\n🏗️  HEXAHEDRAL MESH QUALITY DEMO")
    print("-" * 50)
    
    mesh = create_perfect_hex_cube()
    
    # Run all applicable metrics for hex mesh
    operation = VolumeCellQualityComputation(quality_types=[
        'scaled_jacobian',
        'volume',
        'aspect_ratio',
        'condition',
        'shape',
        'warpage',           # NEW
        'equiangle_skew',    # NEW
        'stretch',           # NEW
        'taper',             # NEW
    ])
    
    issues = operation.execute(mesh)
    
    print(f"Mesh: 1 hex8 element (perfect cube)")
    print(f"Quality types checked: {len(operation.quality_types)}")
    print(f"Issues found: {len(issues)}")
    
    if issues:
        print("\n  Issues detected:")
        for issue in issues:
            print(f"    - {issue.message}")
    else:
        print("\n  ✅ No quality issues - perfect cube!")


def demo_tet_quality():
    """Demonstrate quality metrics on a tet mesh."""
    print("\n🔺 TETRAHEDRAL MESH QUALITY DEMO")
    print("-" * 50)
    
    mesh = create_regular_tet()
    
    # Run all applicable metrics for tet mesh
    operation = VolumeCellQualityComputation(quality_types=[
        'scaled_jacobian',
        'volume',
        'aspect_ratio',
        'condition',
        'shape',
        'equiangle_skew',    # NEW
        'stretch',           # NEW
        'radius_ratio',      # NEW
        'collapse_ratio',    # NEW
    ])
    
    issues = operation.execute(mesh)
    
    print(f"Mesh: 1 tet4 element (regular tetrahedron)")
    print(f"Quality types checked: {len(operation.quality_types)}")
    print(f"Issues found: {len(issues)}")
    
    if issues:
        print("\n  Issues detected:")
        for issue in issues:
            print(f"    - {issue.message}")
    else:
        print("\n  ✅ No quality issues - regular tetrahedron!")


def demo_element_type_filtering():
    """Demonstrate element-type filtering for metrics."""
    print("\n🚫 ELEMENT-TYPE FILTERING DEMO")
    print("-" * 50)
    
    mesh = create_regular_tet()
    
    # Try to run warpage on tet mesh (should be skipped)
    operation = VolumeCellQualityComputation(
        quality_types=['warpage', 'taper'],  # These don't apply to tets
        skip_unsupported_metrics=True
    )
    
    issues = operation.execute(mesh)
    
    print(f"Mesh: 1 tet4 element")
    print(f"Requested metrics: warpage, taper (hex-only metrics)")
    print(f"Issues found: {len(issues)}")
    print("\n  ✅ Metrics automatically skipped for incompatible element types!")


def demo_thresholds():
    """Demonstrate industry-standard thresholds."""
    print("\n📏 INDUSTRY-STANDARD THRESHOLDS")
    print("-" * 50)
    
    thresholds = {
        'warpage': (0.5, 'Warping factor (ANSYS)'),
        'equiangle_skew': (0.25, 'CFD average threshold (SimScale)'),
        'stretch': (0.25, 'Edge uniformity'),
        'taper': (0.5, 'Hex distortion'),
        'radius_ratio': (10.0, 'Tet quality (VTK default)'),
    }
    
    print("\n  Metric Thresholds:")
    for metric, (value, description) in thresholds.items():
        print(f"    • {metric}: {value}")
        print(f"      ({description})")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("EXTENDED QUALITY METRICS DEMONSTRATION")
    print("Gap #8 Implementation - Complete")
    print("=" * 70)
    
    print_metric_info()
    demo_hex_quality()
    demo_tet_quality()
    demo_element_type_filtering()
    demo_thresholds()
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nFor more information, see:")
    print("  - docs/research/research-gap-08-quality-metrics.md")
    print("  - docs/IMPLEMENTATION_STAGE_1_QUALITY_METRICS.md")
    print()


if __name__ == '__main__':
    main()
