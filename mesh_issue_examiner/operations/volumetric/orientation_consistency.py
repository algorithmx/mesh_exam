"""
Orientation and consistency check operation for volumetric (3D) meshes.

Detects inverted elements (negative volume) and inconsistent face winding
in volumetric mesh cells, which cause solver failures and incorrect results.
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
    """

    # Maximum number of IssueLocation objects per issue (avoids huge reports).
    # This cap prevents excessive memory usage and report size when processing
    # large meshes with widespread orientation issues.
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
                  'winding', 'consistency'],
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
                # Fall back to manual signed-volume computation for tet4 cells
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

        Only tet4 cells are analysed because they have a well-defined set of
        four triangular faces.  For each internal face shared by exactly two
        cells the oriented face tuples must be reverses of each other.  If
        they are identical the winding is inconsistent.
        """
        tet_cells = mesh.volume_cells.get('tet4', np.empty((0, 4), dtype=int))
        if tet_cells.ndim != 2 or tet_cells.shape[0] == 0:
            return []

        face_to_cells = self._build_face_to_cell_map(tet_cells)

        inconsistent_cells: set = set()
        for _key, entries in face_to_cells.items():
            # Only check internal faces (shared by exactly 2 cells)
            if len(entries) != 2:
                continue
            (cell_a, face_a), (cell_b, face_b) = entries[0], entries[1]
            if not self._faces_are_reversed(face_a, face_b):
                inconsistent_cells.add(cell_a)
                inconsistent_cells.add(cell_b)

        if not inconsistent_cells:
            return []

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
                'total_tet_cells': float(tet_cells.shape[0]),
            },
        )
        return [issue]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_face_to_cell_map(
        tet_cells: np.ndarray,
    ) -> Dict[Tuple[int, ...], List[Tuple[int, Tuple[int, int, int]]]]:
        """Build mapping from sorted face tuples to (cell_idx, oriented_face).

        For each tet cell ``(a, b, c, d)`` the four triangular faces are:
        ``(a, b, c)``, ``(a, c, d)``, ``(a, d, b)``, ``(b, d, c)``.

        The dict key is the *sorted* face tuple so that differently-oriented
        but geometrically identical faces map to the same entry.
        """
        face_to_cells: Dict[
            Tuple[int, ...],
            List[Tuple[int, Tuple[int, int, int]]],
        ] = {}

        for cell_idx in range(tet_cells.shape[0]):
            a, b, c, d = tet_cells[cell_idx]
            faces = [(a, b, c), (a, c, d), (a, d, b), (b, d, c)]
            for face in faces:
                key = tuple(sorted(face))
                if key not in face_to_cells:
                    face_to_cells[key] = []
                face_to_cells[key].append((cell_idx, face))

        return face_to_cells

    @staticmethod
    def _faces_are_reversed(
        face_a: Tuple[int, int, int],
        face_b: Tuple[int, int, int],
    ) -> bool:
        """Return True if *face_b* is the reverse winding of *face_a*.

        Two triangular faces are reverses when ``face_b`` equals
        ``(face_a[0], face_a[2], face_a[1])`` or either cyclic
        rotation of that reversed ordering.
        """
        # The reverse of (x, y, z) is (x, z, y).
        # The three cyclic rotations of the reverse are:
        rev = (face_a[0], face_a[2], face_a[1])
        cyclic = [
            rev,
            (rev[1], rev[2], rev[0]),
            (rev[2], rev[0], rev[1]),
        ]
        return face_b in cyclic

    @staticmethod
    def _compute_signed_volumes(mesh: MeshInput) -> Optional[np.ndarray]:
        """Compute signed volumes for tet4 cells using the determinant formula.

        For a tet with vertices v0, v1, v2, v3:
            V = det([v1-v0, v2-v0, v3-v0]) / 6

        Returns None if there are no tet4 cells.
        """
        tet_cells = mesh.volume_cells.get('tet4')
        if tet_cells is None or len(tet_cells) == 0:
            return None

        verts = mesh.vertices  # (N, 3)
        v0 = verts[tet_cells[:, 0]]  # (M, 3)
        v1 = verts[tet_cells[:, 1]]
        v2 = verts[tet_cells[:, 2]]
        v3 = verts[tet_cells[:, 3]]

        # Columns of the 3x3 Jacobian for each cell
        e1 = v1 - v0
        e2 = v2 - v0
        e3 = v3 - v0

        # Scalar triple product = det([e1|e2|e3])
        vol = (
            e1[:, 0] * (e2[:, 1] * e3[:, 2] - e2[:, 2] * e3[:, 1])
            - e1[:, 1] * (e2[:, 0] * e3[:, 2] - e2[:, 2] * e3[:, 0])
            + e1[:, 2] * (e2[:, 0] * e3[:, 1] - e2[:, 1] * e3[:, 0])
        ) / 6.0

        return vol


# Register operation
registry.register(OrientationConsistencyCheck())
