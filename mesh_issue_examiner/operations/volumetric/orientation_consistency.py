"""
Orientation and consistency check operation for volumetric (3D) meshes.

Detects inverted elements (negative volume) and inconsistent face winding
in volumetric mesh cells, which cause solver failures and incorrect results.

Supports all linear element types: tet4, hex8, prism6, pyramid5, and
mixed-element meshes.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from ...core.models import (
    MeshInput,
    MeshIssue,
    IssueCategory,
    IssueSeverity,
    IssueLocation,
    OperationConfig,
)
from ..base import BaseOperation, OperationMetadata
from ..operation_registry import registry
from ._face_tables import ELEMENT_FACES


class OrientationConsistencyCheck(BaseOperation):
    """Detects orientation and consistency defects in volumetric mesh cells.

    Two checks are performed:

    1. **Inverted elements** — cells whose signed volume is negative.  For a
       valid mesh every cell must have positive volume.  Inverted cells cause
       negative Jacobians and solver failures.

    2. **Inconsistent winding** — for each internal face shared by two cells
       the face vertices should appear in opposite order in the two cells.
       If both cells list the face with the same vertex order the winding is
       inconsistent, leading to assembly errors in FEA/CFD.

    Supports tet4, hex8, prism6, pyramid5, and mixed-element meshes.
    """

    # Maximum number of IssueLocation objects per issue (avoids huge reports).
    _MAX_LOCATIONS = 100

    def __init__(self, config: Optional[OperationConfig] = None):
        super().__init__(config)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> OperationMetadata:
        return OperationMetadata(
            operation_id='quality_3d.check_orientation_consistency',
            display_name='Orientation and Consistency Check (3D)',
            description='Detects inverted elements and inconsistent face '
                        'winding in volumetric meshes.',
            category=IssueCategory.ORIENTATION,
            supported_libraries=['pyvista'],
            recommended_library='pyvista',
            requires_adjacency=True,
            complexity='O(n log n)',
            tags=['orientation', 'volumetric', '3d', 'inverted',
                  'winding', 'consistency', 'tet4', 'hex8',
                  'prism6', 'pyramid5'],
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        """Run orientation and consistency checks on *mesh*.

        Args:
            mesh: A :class:`MeshInput` with volumetric cells.

        Returns:
            List of :class:`MeshIssue` instances (may be empty).
        """
        issues: List[MeshIssue] = []

        # 1. Must have volumetric cells
        if not mesh.has_volume_cells:
            return issues

        # 2. Obtain pyvista UnstructuredGrid
        grid = mesh.get_native('pyvista')

        # 3. Inverted-element detection (negative signed volume)
        issues.extend(self._check_inverted_cells(mesh, grid))

        # 4. Inconsistent face-winding detection
        issues.extend(self._check_winding_consistency(mesh))

        return issues

    # ------------------------------------------------------------------
    # Inverted element detection
    # ------------------------------------------------------------------

    def _check_inverted_cells(
        self,
        mesh: MeshInput,
        grid,
    ) -> List[MeshIssue]:
        """Detect cells with negative signed volume.

        Uses pyvista ``cell_quality(quality_measure='volume')`` to obtain
        signed cell volumes.  Cells whose volume is less than
        ``-config.tolerance`` are flagged as CRITICAL.
        """
        volumes = grid.cell_data['Volume'] if 'Volume' in grid.cell_data else None
        if volumes is None:
            try:
                volumes = grid.cell_quality(
                    quality_measure='volume'
                ).cell_data['volume']
            except Exception:
                volumes = self._compute_signed_volumes(mesh)

        if volumes is None or len(volumes) == 0:
            return []

        tol = self.config.tolerance
        inverted_mask = volumes < -tol
        inverted_indices = np.where(inverted_mask)[0]

        if len(inverted_indices) == 0:
            return []

        locations = [
            IssueLocation(
                element_type='cell',
                element_index=int(idx),
            )
            for idx in inverted_indices[:self._MAX_LOCATIONS]
        ]

        issue = MeshIssue(
            operation_id=self.metadata.operation_id,
            category=IssueCategory.ORIENTATION,
            severity=IssueSeverity.CRITICAL,
            message=(
                f'Found {len(inverted_indices)} inverted cell(s) with '
                f'negative volume (< {-tol}).'
            ),
            locations=locations,
            metrics={
                'inverted_count': float(len(inverted_indices)),
                'min_volume': float(np.min(volumes)),
                'total_cells': float(len(volumes)),
            },
        )
        return [issue]

    # ------------------------------------------------------------------
    # Face-winding consistency
    # ------------------------------------------------------------------

    def _check_winding_consistency(self, mesh: MeshInput) -> List[MeshIssue]:
        """Check that shared faces have opposite winding in adjacent cells.

        Works for all element types (tet4, hex8, prism6, pyramid5) and
        mixed-element meshes.  Uses the directed-edge method to compare
        winding: for each shared face, every directed edge (u->v) from one
        cell must appear as (v->u) in the adjacent cell.
        """
        face_to_cells = self._build_face_to_cell_map(mesh)
        if not face_to_cells:
            return []

        inconsistent_cells: set = set()
        total_shared_faces = 0

        for _key, entries in face_to_cells.items():
            # Only check internal faces (shared by exactly 2 cells)
            if len(entries) != 2:
                continue
            total_shared_faces += 1
            (cell_a, face_a), (cell_b, face_b) = entries[0], entries[1]
            if not self._faces_are_reversed(face_a, face_b):
                inconsistent_cells.add(cell_a)
                inconsistent_cells.add(cell_b)

        if not inconsistent_cells:
            return []

        # Compute total cell count across all element types
        total_cells = sum(
            len(conn) for conn in mesh.volume_cells.values()
        )

        sorted_cells = sorted(inconsistent_cells)
        locations = [
            IssueLocation(
                element_type='cell',
                element_index=int(idx),
            )
            for idx in sorted_cells[:self._MAX_LOCATIONS]
        ]

        issue = MeshIssue(
            operation_id=self.metadata.operation_id,
            category=IssueCategory.ORIENTATION,
            severity=IssueSeverity.ERROR,
            message=(
                f'Found {len(sorted_cells)} cell(s) with inconsistent '
                f'face winding across shared internal faces.'
            ),
            locations=locations,
            metrics={
                'inconsistent_winding_cells': float(len(sorted_cells)),
                'total_shared_faces': float(total_shared_faces),
                'total_cells': float(total_cells),
            },
        )
        return [issue]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_face_to_cell_map(
        mesh: MeshInput,
    ) -> Dict[Tuple[int, ...], List[Tuple[int, Tuple[int, ...]]]]:
        """Build mapping from sorted face tuples to (cell_idx, oriented_face).

        Iterates over all element types in ``mesh.volume_cells``.  For each
        cell, extracts faces using the :data:`ELEMENT_FACES` table and
        creates a canonical key by sorting vertex indices.  The original
        (oriented) face tuple is stored alongside the global cell index so
        that winding can be compared later.

        Args:
            mesh: Input mesh with volumetric cells.

        Returns:
            Dictionary mapping sorted face tuples to lists of
            ``(global_cell_index, oriented_face_tuple)`` pairs.
        """
        face_to_cells: Dict[
            Tuple[int, ...],
            List[Tuple[int, Tuple[int, ...]]],
        ] = {}

        global_cell_idx = 0

        for element_type, connectivity in mesh.volume_cells.items():
            if element_type not in ELEMENT_FACES:
                # Skip unknown element types but advance global index
                global_cell_idx += len(connectivity)
                continue

            face_defs = ELEMENT_FACES[element_type]
            n_cells = len(connectivity)

            for local_idx in range(n_cells):
                cell_verts = connectivity[local_idx]

                for face_def in face_defs:
                    face = tuple(int(cell_verts[i]) for i in face_def)
                    key = tuple(sorted(face))
                    face_to_cells.setdefault(key, []).append(
                        (global_cell_idx, face)
                    )

                global_cell_idx += 1

        return face_to_cells

    @staticmethod
    def _faces_are_reversed(
        face_a: Tuple[int, ...],
        face_b: Tuple[int, ...],
    ) -> bool:
        """Return True if *face_b* has opposite winding to *face_a*.

        Uses the **directed-edge method**: for opposite winding, every
        directed edge ``(u, v)`` in ``face_a`` must appear as ``(v, u)``
        in ``face_b``.  This works uniformly for triangular and
        quadrilateral faces.
        """
        n = len(face_a)
        # Collect directed edges from face_b into a set for O(1) lookup
        edges_b = {(face_b[i], face_b[(i + 1) % n]) for i in range(n)}
        # Every edge in face_a must have its reverse in face_b
        return all(
            (face_a[(i + 1) % n], face_a[i]) in edges_b
            for i in range(n)
        )

    @staticmethod
    def _compute_signed_volumes(mesh: MeshInput) -> Optional[np.ndarray]:
        """Compute signed volumes for all volumetric cells.

        For tet4 cells, uses the determinant formula directly.
        For hex8, prism6, pyramid5, decomposes each cell into tetrahedra
        and sums the signed volumes.

        Returns:
            1D array of signed volumes (one per cell, in global order),
            or None if there are no volumetric cells.
        """
        if not mesh.has_volume_cells:
            return None

        verts = mesh.vertices  # (N, 3)
        all_volumes: List[float] = []

        for element_type, connectivity in mesh.volume_cells.items():
            n_cells = len(connectivity)
            if n_cells == 0:
                continue

            if element_type == 'tet4':
                vol = _signed_volume_tet4(verts, connectivity)
                all_volumes.extend(vol.tolist())

            elif element_type == 'hex8':
                for i in range(n_cells):
                    cell = connectivity[i]
                    all_volumes.append(
                        _signed_volume_decomposed(verts, cell, _HEX8_TETS)
                    )

            elif element_type == 'prism6':
                for i in range(n_cells):
                    cell = connectivity[i]
                    all_volumes.append(
                        _signed_volume_decomposed(verts, cell, _PRISM6_TETS)
                    )

            elif element_type == 'pyramid5':
                for i in range(n_cells):
                    cell = connectivity[i]
                    all_volumes.append(
                        _signed_volume_decomposed(verts, cell, _PYRAMID5_TETS)
                    )

            else:
                # Unknown type — skip (will be 0 volume)
                all_volumes.extend([0.0] * n_cells)

        if not all_volumes:
            return None

        return np.array(all_volumes)


# Tetrahedral decompositions for signed-volume computation.
# Each sub-list contains 4 local vertex indices forming a tet.
# The decomposition covers the full cell without overlap.

# hex8 → 5 tets (using vertex 0 as hub)
_HEX8_TETS = [
    (0, 1, 3, 4),
    (1, 2, 3, 6),
    (1, 4, 5, 6),
    (3, 4, 6, 7),
    (1, 3, 4, 6),
]

# prism6 → 3 tets
_PRISM6_TETS = [
    (0, 1, 2, 3),
    (1, 2, 3, 4),
    (2, 3, 4, 5),
]

# pyramid5 → 2 tets (using the quad diagonal)
_PYRAMID5_TETS = [
    (0, 1, 3, 4),
    (1, 2, 3, 4),
]


def _signed_volume_tet4(
    verts: np.ndarray,
    connectivity: np.ndarray,
) -> np.ndarray:
    """Vectorised signed volume for tet4 cells.

    For a tet with vertices v0, v1, v2, v3:
        V = det([v1-v0, v2-v0, v3-v0]) / 6
    """
    v0 = verts[connectivity[:, 0]]
    v1 = verts[connectivity[:, 1]]
    v2 = verts[connectivity[:, 2]]
    v3 = verts[connectivity[:, 3]]

    e1 = v1 - v0
    e2 = v2 - v0
    e3 = v3 - v0

    vol = (
        e1[:, 0] * (e2[:, 1] * e3[:, 2] - e2[:, 2] * e3[:, 1])
        - e1[:, 1] * (e2[:, 0] * e3[:, 2] - e2[:, 2] * e3[:, 0])
        + e1[:, 2] * (e2[:, 0] * e3[:, 1] - e2[:, 1] * e3[:, 0])
    ) / 6.0

    return vol


def _signed_volume_decomposed(
    verts: np.ndarray,
    cell: np.ndarray,
    tet_defs: list,
) -> float:
    """Signed volume for a cell decomposed into tetrahedra.

    Args:
        verts: (N, 3) vertex array.
        cell: 1D array of global vertex IDs for the cell.
        tet_defs: List of 4-tuples of local vertex indices.

    Returns:
        Sum of signed volumes of the sub-tetrahedra.
    """
    total = 0.0
    for tet_def in tet_defs:
        v0 = verts[cell[tet_def[0]]]
        v1 = verts[cell[tet_def[1]]]
        v2 = verts[cell[tet_def[2]]]
        v3 = verts[cell[tet_def[3]]]

        e1 = v1 - v0
        e2 = v2 - v0
        e3 = v3 - v0

        vol = (
            e1[0] * (e2[1] * e3[2] - e2[2] * e3[1])
            - e1[1] * (e2[0] * e3[2] - e2[2] * e3[0])
            + e1[2] * (e2[0] * e3[1] - e2[1] * e3[0])
        ) / 6.0
        total += vol

    return total


# Register operation
registry.register(OrientationConsistencyCheck())
