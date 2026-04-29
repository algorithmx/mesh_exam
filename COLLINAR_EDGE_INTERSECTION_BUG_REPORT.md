# Triangle Intersection: Missing Collinear Edge Detection - Bug Report

## Executive Summary

**Issue:** The `edges_intersect_2d` function in `mesh_issue_examiner/utils/triangle_intersection.py` fails to detect intersections between collinear overlapping line segments.

**Status:** ✅ **FIXED**

**Severity:** Medium-High - Can cause missed self-intersections in coplanar triangle pairs

---

## 1. Bug Verification

### 1.1 Original Code (Buggy)

```python
def edges_intersect_2d(p1, p2, p3, p4):
    def cross_2d(a, b):
        return a[0] * b[1] - a[1] * b[0]
    
    r = p2 - p1
    s = p4 - p3
    
    denom = cross_2d(r, s)
    
    if abs(denom) < 1e-15:
        # Lines are parallel
        return False  # BUG: Doesn't check for collinear overlap!
    
    diff = p3 - p1
    t = cross_2d(diff, s) / denom
    u = cross_2d(diff, r) / denom
    
    return (0 <= t <= 1) and (0 <= u <= 1)
```

### 1.2 Test Results - Before Fix

| Test Case | Expected | Actual | Status |
|-----------|----------|--------|--------|
| Collinear overlapping segments | True | **False** | ❌ FAIL |
| Collinear touching at endpoint | True | **False** | ❌ FAIL |
| Collinear disjoint segments | False | False | ✅ PASS |
| One segment inside other | True | **False** | ❌ FAIL |
| Parallel non-collinear | False | False | ✅ PASS |
| Normal crossing | True | True | ✅ PASS |

### 1.3 Root Cause

When two line segments are parallel (`denom ≈ 0`), the code immediately returns `False` without checking if the segments are:
1. **Collinear** (lying on the same infinite line)
2. **Overlapping** in their 1D projections

---

## 2. Research Findings

### 2.1 Standard Algorithm

The correct approach uses **orientation tests** (also known as the "crossing number" or "orientation predicate" method):

**References:**
1. **de Berg et al.** - "Computational Geometry: Algorithms and Applications" (Chapter 2)
2. **CGAL** - Computational Geometry Algorithms Library (Segment_2 intersection)
3. **e-maxx** - "Finding intersection of two segments" (competitive programming algorithms)
4. **Bentley-Ottmann** sweep-line algorithm foundations

### 2.2 Algorithm Description

Two segments p1-p2 and p3-p4 intersect if and only if:

1. **General case:** The orientations satisfy:
   - `orient(p1, p2, p3)` and `orient(p1, p2, p4)` have opposite signs AND
   - `orient(p3, p4, p1)` and `orient(p3, p4, p2)` have opposite signs

2. **Special cases** (collinear): Any endpoint lies on the other segment:
   - p3 lies on segment p1-p2, OR
   - p4 lies on segment p1-p2, OR
   - p1 lies on segment p3-p4, OR
   - p2 lies on segment p3-p4

Where `orient(a, b, c) = cross(b-a, c-a)` determines if point c is to the left, right, or on line ab.

### 2.3 Key Insight

The orientation test elegantly handles ALL cases:
- Normal crossings
- Endpoint touching
- Collinear overlaps
- Complete containment

---

## 3. Impact Analysis

### 3.1 Affected Functionality

| Component | Impact | Description |
|-----------|--------|-------------|
| `edges_intersect_2d` | Direct | Fails for collinear overlaps |
| `coplanar_triangles_intersect` | Indirect | May miss coplanar triangle intersections |
| `detect_self_intersections_manual` | Indirect | May miss self-intersections in planar regions |

### 3.2 Real-World Mesh Defects That Could Be Missed

1. **Overlapping Coplanar Faces:** Two triangles in the same plane with overlapping edges
   - Common in: CAD exports with duplicate geometry, 2D mesh extrusions
   
2. **Zero-Thickness Penetrations:** Edges sliding along coplanar surfaces
   - Common in: Architectural models, PCB designs

3. **T-Junctions on Coplanar Surfaces:** Edge terminating on another coplanar edge
   - Common in: Subdivided surfaces, tessellated NURBS

4. **Degenerate Overlaps:** Multiple triangles occupying the same planar region
   - Common in: Boolean operation failures, corrupted exports

### 3.3 Impact Assessment

- **Frequency:** Low-Medium (requires coplanar triangle pairs)
- **Severity:** High (missed intersections are real defects)
- **Workaround:** None (algorithmic bug)

---

## 4. Proposed Fix

### 4.1 Fixed Implementation

```python
def edges_intersect_2d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray,
                        tolerance: float = 1e-15) -> bool:
    """
    Check if two line segments intersect in 2D.
    
    Handles all intersection cases including:
    - Normal crossing segments
    - Segments touching at endpoints
    - Collinear overlapping segments
    
    Uses orientation-based algorithm from computational geometry literature:
    - de Berg et al., "Computational Geometry: Algorithms and Applications"
    - Standard approach used in CGAL, LEDA, and other CG libraries
    
    Args:
        p1, p2: First segment endpoints
        p3, p4: Second segment endpoints
        tolerance: Numerical tolerance for collinearity checks
        
    Returns:
        True if segments intersect (including at endpoints or overlapping)
    """
    def cross_2d(a, b):
        return a[0] * b[1] - a[1] * b[0]
    
    def on_segment(a, b, c):
        """Check if point b lies on segment ac (assuming collinear)."""
        return (
            min(a[0], c[0]) - tolerance <= b[0] <= max(a[0], c[0]) + tolerance and
            min(a[1], c[1]) - tolerance <= b[1] <= max(a[1], c[1]) + tolerance
        )
    
    # Compute orientation values (2D cross products)
    o1 = cross_2d(p2 - p1, p3 - p1)
    o2 = cross_2d(p2 - p1, p4 - p1)
    o3 = cross_2d(p4 - p3, p1 - p3)
    o4 = cross_2d(p4 - p3, p2 - p3)
    
    # General case: proper intersection (segments cross each other)
    cond1 = (o1 > tolerance and o2 < -tolerance) or (o1 < -tolerance and o2 > tolerance)
    cond2 = (o3 > tolerance and o4 < -tolerance) or (o3 < -tolerance and o4 > tolerance)
    if cond1 and cond2:
        return True
    
    # Special Cases: collinear points
    if abs(o1) < tolerance and on_segment(p1, p3, p2):
        return True
    if abs(o2) < tolerance and on_segment(p1, p4, p2):
        return True
    if abs(o3) < tolerance and on_segment(p3, p1, p4):
        return True
    if abs(o4) < tolerance and on_segment(p3, p2, p4):
        return True
    
    return False
```

### 4.2 Key Improvements

1. **Orientation-based approach:** Uses cross products for robust geometric predicates
2. **Handles all collinear cases:** Overlaps, endpoint touches, and containment
3. **Configurable tolerance:** Added `tolerance` parameter for numerical stability
4. **Well-documented:** References standard CG literature

### 4.3 Numerical Stability

The implementation maintains numerical stability by:
- Using tolerance-based comparisons instead of exact equality
- Checking bounding boxes for on-segment tests
- Following established patterns from CGAL and other robust libraries

---

## 5. Verification

### 5.1 Test Results - After Fix

| Test Case | Expected | Actual | Status |
|-----------|----------|--------|--------|
| Collinear overlapping segments | True | True | ✅ PASS |
| Collinear touching at endpoint | True | True | ✅ PASS |
| Collinear disjoint segments | False | False | ✅ PASS |
| One segment inside other | True | True | ✅ PASS |
| Parallel non-collinear | False | False | ✅ PASS |
| Normal crossing | True | True | ✅ PASS |
| Coplanar triangles - collinear edges | True | True | ✅ PASS |

### 5.2 Regression Testing

All 24 existing unit tests pass:
- `TestBboxOverlap`: 2/2 passed
- `TestAreCoplanar`: 2/2 passed
- `TestMollerTrumbore`: 3/3 passed
- `TestTrianglesIntersect`: 2/2 passed
- `TestBuildAdjacencyPairs`: 2/2 passed
- `TestPointInTriangle2d`: 2/2 passed
- `TestDetectSelfIntersectionsManual`: 6/6 passed
- `TestChordThroughInterior`: 5/5 passed

---

## 6. Conclusion

The collinear edge intersection bug has been successfully fixed. The new implementation:

1. ✅ Correctly detects collinear overlapping segments
2. ✅ Passes all existing regression tests
3. ✅ Uses industry-standard computational geometry algorithms
4. ✅ Maintains numerical stability with configurable tolerance
5. ✅ Is well-documented with academic references

**Recommendation:** The fix should be merged into the main codebase.

---

## 7. References

1. de Berg, M., Cheong, O., van Kreveld, M., & Overmars, M. (2008). *Computational Geometry: Algorithms and Applications*. Springer.

2. CGAL, *Computational Geometry Algorithms Library*, https://www.cgal.org/

3. Möller, T. (1997). "A Fast Triangle-Triangle Intersection Test", *Journal of Graphics Tools*, 2(2), 25-30.

4. Ericson, C. (2005). *Real-Time Collision Detection*. Morgan Kaufmann.

5. e-maxx, "Finding intersection of two segments", https://cp-algorithms.com/geometry/intersecting_segments.html
