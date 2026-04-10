# Open Mesh Self-Intersection False Positives Inspection Report

## Executive Summary
This report analyzes the root cause of the "massive self-intersection false positives" observed on open meshes (e.g., `file_id=131372, 1017 faces`), tracking it down to the chord-through-interior artifact suppression logic in the `_is_surface_intersection` function, and its negative interaction with boundary edges on curved surfaces.

## Architecture Overview
The self-intersection detection in `mesh_issue_examiner/utils/triangle_intersection.py` uses the Möller-Trumbore method with KD-Tree acceleration. To prevent curved piece-wise linear surfaces from falsely reporting intersections when their chord-like edges dip "beneath" the surface and intersect neighboring non-adjacent faces, the system implements a suppression mechanism in `_is_surface_intersection`.

## Implementation Analysis

### The `_is_surface_intersection` Distance Heuristic
When `triangles_intersect` finds a mathematical intersection, it validates it using `_is_surface_intersection` ([utils/triangle_intersection.py](mesh_issue_examiner/utils/triangle_intersection.py#L427)). The function is meant to discard "chord-through-interior" artifacts. 

The docstring states that these artifacts should be detected by checking "whether the intersection point lies on the **outer** side of both triangle normals". However, the actual code does **not** use normals at all. Instead, it relies entirely on a distance threshold heuristic:

```python
def _is_surface_intersection(point, owner_tri, hit_tri, tolerance):
    min_edge_dist = _min_edge_edge_distance(owner_tri, hit_tri)
    mean_edge = _mean_edge_length(owner_tri, hit_tri)
    threshold = 0.05 * mean_edge
    return min_edge_dist <= threshold
```

This logic assumes that in a *genuine* surface intersection, the intersecting edges actually cross each other, meaning `min_edge_dist` is near zero (`<= 0.05 * mean_edge`). Conversely, if an edge dips under a surface and crosses in the *middle* of another face, the distance from the crossing point to the edges of the punctured face is large, so `min_edge_dist > threshold`, and the intersection is suppressed.

## Why Open Meshes Produce False Positives
The assumption that chord artifacts always have `min_edge_dist > 0.05 * mean_edge` breaks down around the boundary edges of open, curved meshes:

1. **Boundary Edge Curling:** On a curved open surface, boundary edges act as geometric chords between boundary vertices. Because it is a boundary edge, it can freely dip "under/over" the adjacent faces following the curvature, structurally intersecting the face of a nearby triangle.
2. **Tessellation Density:** For dense meshes, the `hit_tri` (the face intersected by the boundary edge) is topologically very close (often sharing a vertex or separated by only one intermediate face). As a result, *another* edge of `owner_tri` is spatially extremely close to an edge of `hit_tri`, even though the exact crossing point is in the middle of the face.
3. **Threshold Failure:** This spatial proximity between non-crossing edges forces `min_edge_dist` below the `0.05 * mean_edge` threshold. The code incorrectly assumes the intersection is genuine ("genuine self-intersections have edges that cross or nearly touch"), preventing the chord artifact from being suppressed.
4. **Convex Hull Bypass:** Closed meshes like cylinders or spheres can often bypass this flaw via `_is_convex(vertices)` early-out logic, but open meshes (which are rarely convex hulls of themselves) fall fully victim to this distance heuristic flaw.

## Findings
The root cause is that the `_is_surface_intersection` heuristic is insufficient and misaligned with its own docstring. Relying strictly on `_min_edge_edge_distance` causes the detector to misinterpret natural edge-chords on curved boundaries of open meshes as genuine self-intersections. 

### Recommendations
1. Refactor `_is_surface_intersection` to actually implement the normal-based check described in its docstring (i.e. verifying whether the midpoint/intersection falls on the "interior" or "exterior" side of the facing plane).
2. Explicitly track and identify boundary edges before intersection checks to apply a more tolerant artifact filter for boundary edges specifically.