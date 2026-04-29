"""
Demonstration of granular reporting operations (Gap #13 implementation).

This script demonstrates the new vertex-level and edge-level diagnostics:
    - VertexValenceDetection
    - EdgeDihedralAngleDetection
    - EdgeLengthRatioDetection
"""

import numpy as np
from mesh_issue_examiner.operations.topological import (
    VertexValenceDetection,
    EdgeDihedralAngleDetection,
    EdgeLengthRatioDetection,
)
from mesh_issue_examiner.utils.topology import (
    compute_vertex_valence,
    build_edge_face_map,
    extract_unique_edges,
)
from mesh_issue_examiner.core.models import MeshInput


def create_regular_grid():
    """Create a regular triangle mesh grid."""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [2, 0, 0],
        [0, 1, 0], [1, 1, 0], [2, 1, 0],
        [0, 2, 0], [1, 2, 0], [2, 2, 0],
    ])
    # Split quads into triangles
    faces = np.array([
        [0, 1, 3], [1, 4, 3],
        [1, 2, 4], [2, 5, 4],
        [3, 4, 6], [4, 7, 6],
        [4, 5, 7], [5, 8, 7],
    ])
    return MeshInput(vertices=vertices, faces=faces)


def create_high_valence_mesh():
    """Create a mesh with a high-valence vertex."""
    # Central vertex (0) connected to many surrounding vertices
    vertices = np.array([
        [0.5, 0.5, 0],  # Center - high valence
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom plane
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # Top plane
    ])
    # Many triangles sharing vertex 0
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1],  # Bottom
        [0, 1, 5], [0, 2, 6], [0, 3, 7], [0, 4, 8],  # Vertical
        [0, 5, 6], [0, 6, 7], [0, 7, 8], [0, 8, 5],  # Top
    ])
    return MeshInput(vertices=vertices, faces=faces)


def create_sharp_edge_mesh():
    """Create a mesh with a sharp edge."""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [0.5, 1, 0],   # Triangle 1 (z=0)
        [0, 0, 0], [1, 0, 0], [0.5, 0, 1],   # Triangle 2 (perpendicular)
    ])
    faces = np.array([
        [0, 1, 2],
        [3, 4, 5],
    ])
    return MeshInput(vertices=vertices, faces=faces)


def create_non_uniform_mesh():
    """Create a mesh with non-uniform edge lengths."""
    vertices = np.array([
        [0, 0, 0],
        [0.1, 0, 0],   # Short edge
        [0.2, 0, 0],   # Short edge
        [10, 0, 0],    # Very long edge
        [0, 0.1, 0],   # Short edge
    ])
    faces = np.array([
        [0, 1, 4],
        [1, 2, 4],
        [2, 3, 4],  # This face has the long edge
    ])
    return MeshInput(vertices=vertices, faces=faces)


def demo_vertex_valence():
    """Demonstrate vertex valence detection."""
    print("\n📊 VERTEX VALENCE DETECTION DEMO")
    print("-" * 50)
    
    # Regular grid
    mesh = create_regular_grid()
    op = VertexValenceDetection()
    issues = op.execute(mesh)
    
    print("Regular Grid:")
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Faces: {len(mesh.faces)}")
    print(f"  Issues found: {len(issues)}")
    
    # Compute and show valence distribution
    valence = compute_vertex_valence(mesh.faces, len(mesh.vertices))
    unique, counts = np.unique(valence, return_counts=True)
    print(f"  Valence distribution:")
    for v, c in zip(unique, counts):
        print(f"    Valence {v}: {c} vertices")
    
    # High valence mesh
    mesh = create_high_valence_mesh()
    op = VertexValenceDetection(max_valence_triangle=8)
    issues = op.execute(mesh)
    
    print("\nHigh-Valence Mesh:")
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Faces: {len(mesh.faces)}")
    print(f"  Issues found: {len(issues)}")
    
    if issues:
        print(f"  Irregular vertices: {issues[0].metrics['irregular_count']}")
        valence = compute_vertex_valence(mesh.faces, len(mesh.vertices))
        print(f"  Vertex 0 valence: {valence[0]} (center vertex)")


def demo_edge_dihedral():
    """Demonstrate edge dihedral angle detection."""
    print("\n📐 EDGE DIHEDRAL ANGLE DETECTION DEMO")
    print("-" * 50)
    
    # Smooth surface
    mesh = create_regular_grid()
    op = EdgeDihedralAngleDetection(min_angle=10, max_angle=170)
    issues = op.execute(mesh)
    
    print("Smooth Surface (Regular Grid):")
    print(f"  Edges checked: {len(build_edge_face_map(mesh.faces))}")
    print(f"  Issues found: {len(issues)}")
    
    # Sharp edge
    mesh = create_sharp_edge_mesh()
    op = EdgeDihedralAngleDetection(max_angle=80)
    issues = op.execute(mesh)
    
    print("\nSharp Edge Mesh:")
    print(f"  Edges checked: {len(build_edge_face_map(mesh.faces))}")
    print(f"  Issues found: {len(issues)}")
    
    if issues:
        print(f"  Extreme edges: {issues[0].metrics['extreme_edge_count']}")
        print(f"  Dihedral range: {issues[0].metrics['min_dihedral']:.1f}° - "
              f"{issues[0].metrics['max_dihedral']:.1f}°")


def demo_edge_length_ratio():
    """Demonstrate edge length ratio detection."""
    print("\n📏 EDGE LENGTH RATIO DETECTION DEMO")
    print("-" * 50)
    
    # Uniform mesh
    mesh = create_regular_grid()
    op = EdgeLengthRatioDetection(max_ratio=5.0)
    issues = op.execute(mesh)
    
    print("Uniform Mesh (Regular Grid):")
    edges = extract_unique_edges(mesh.faces)
    print(f"  Total edges: {len(edges)}")
    print(f"  Issues found: {len(issues)}")
    
    # Non-uniform mesh
    mesh = create_non_uniform_mesh()
    op = EdgeLengthRatioDetection(max_ratio=5.0)
    issues = op.execute(mesh)
    
    print("\nNon-Uniform Mesh:")
    edges = extract_unique_edges(mesh.faces)
    print(f"  Total edges: {len(edges)}")
    print(f"  Issues found: {len(issues)}")
    
    if issues:
        print(f"  High ratio edges: {issues[0].metrics['high_ratio_count']}")
        print(f"  Max ratio: {issues[0].metrics['max_ratio']:.1f}x")


def print_usage_examples():
    """Print usage examples."""
    print("\n💡 USAGE EXAMPLES")
    print("-" * 50)
    
    examples = [
        ("Vertex Valence", "op = VertexValenceDetection(max_valence_triangle=8)"),
        ("Edge Dihedral", "op = EdgeDihedralAngleDetection(min_angle=20, max_angle=160)"),
        ("Edge Length Ratio", "op = EdgeLengthRatioDetection(max_ratio=10.0)"),
    ]
    
    for name, code in examples:
        print(f"\n{name}:")
        print(f"  {code}")
        print(f"  issues = op.execute(mesh)")
    
    print("\n  IssueLocation structure:")
    print("    - element_type: 'vertex' | 'edge'")
    print("    - element_index: vertex/edge index")
    print("    - coordinates: (x, y, z) position")
    print("    - adjacent_elements: metric-specific data")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("GRANULAR REPORTING OPERATIONS DEMONSTRATION")
    print("Gap #13 Implementation - Complete")
    print("=" * 70)
    
    demo_vertex_valence()
    demo_edge_dihedral()
    demo_edge_length_ratio()
    print_usage_examples()
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nNew operations available:")
    print("  • VertexValenceDetection")
    print("  • EdgeDihedralAngleDetection")
    print("  • EdgeLengthRatioDetection")
    print("\nFor more information, see:")
    print("  - docs/research/research-gap-13-reporting-granularity.md")
    print("  - docs/IMPLEMENTATION_STAGE_5_REPORTING_GRANULARITY.md")
    print()


if __name__ == '__main__':
    main()
