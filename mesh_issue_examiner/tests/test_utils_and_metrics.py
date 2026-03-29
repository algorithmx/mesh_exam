"""
Unit tests for utils/metrics.py and utils/mesh_loader.py.

Uses ONLY numpy (no trimesh/pyvista/pymeshlab).
"""

import pytest
import numpy as np

from utils.metrics import (
    DEFAULT_THRESHOLDS,
    get_default_thresholds,
    get_threshold,
    determine_severity,
    compute_all_quality_metrics,
    compute_violation_stats,
)
from utils.mesh_loader import (
    FORMAT_SUPPORT,
    get_supported_formats,
    is_format_supported,
    create_mesh_from_arrays,
    _get_adapter,
    _is_library_available,
)
from core.models import MeshInput, QualityMetrics, IssueSeverity


# ---------------------------------------------------------------------------
# Part 1: utils/metrics.py
# ---------------------------------------------------------------------------


class TestDefaultThresholds:
    """Tests for DEFAULT_THRESHOLDS constant."""

    EXPECTED_KEYS_AND_VALUES = {
        'aspect_ratio': 0.3,
        'skewness': 0.5,
        'min_angle': 20.0,
        'max_angle': 140.0,
        'scaled_jacobian': 0.1,
        'condition': 10.0,
        'Jacobian': 0.0,
        'volume': 1e-10,
        'area': 1e-10,
    }

    def test_all_expected_keys_exist(self):
        """Every documented key must be present in DEFAULT_THRESHOLDS."""
        for key in self.EXPECTED_KEYS_AND_VALUES:
            assert key in DEFAULT_THRESHOLDS, f"Missing key: {key}"

    def test_all_expected_values_match(self):
        """Every documented key must have the correct value."""
        for key, expected_val in self.EXPECTED_KEYS_AND_VALUES.items():
            assert DEFAULT_THRESHOLDS[key] == expected_val, (
                f"Key {key!r}: expected {expected_val}, got {DEFAULT_THRESHOLDS[key]}"
            )

    def test_no_extra_keys(self):
        """DEFAULT_THRESHOLDS should contain exactly the documented keys."""
        assert set(DEFAULT_THRESHOLDS.keys()) == set(self.EXPECTED_KEYS_AND_VALUES.keys())


class TestGetDefaultThresholds:
    """Tests for get_default_thresholds()."""

    def test_returns_copy_not_reference(self):
        """Modifying the returned dict must not affect the module constant."""
        original = get_default_thresholds()
        original['aspect_ratio'] = 999.0
        fresh = get_default_thresholds()
        assert fresh['aspect_ratio'] != 999.0

    def test_contents_match_default_thresholds(self):
        """Returned dict values must equal DEFAULT_THRESHOLDS values."""
        result = get_default_thresholds()
        for key, val in DEFAULT_THRESHOLDS.items():
            assert result[key] == val


class TestGetThreshold:
    """Tests for get_threshold(metric_name)."""

    def test_known_metric_returns_correct_value(self):
        """Known metric names should return their defined threshold."""
        assert get_threshold('aspect_ratio') == 0.3
        assert get_threshold('skewness') == 0.5
        assert get_threshold('volume') == 1e-10

    def test_unknown_metric_returns_fallback(self):
        """Unknown metric names should return the fallback value 0.1."""
        assert get_threshold('nonexistent_metric') == 0.1
        assert get_threshold('') == 0.1


class TestDetermineSeverity:
    """Tests for determine_severity(violation_rate)."""

    def test_zero_is_info(self):
        assert determine_severity(0.0) == IssueSeverity.INFO

    def test_exact_0_05_is_info(self):
        """0.05 is not > 0.05, so it stays INFO."""
        assert determine_severity(0.05) == IssueSeverity.INFO

    def test_0_051_is_warning(self):
        assert determine_severity(0.051) == IssueSeverity.WARNING

    def test_exact_0_2_is_warning(self):
        """0.2 is not > 0.2, so it stays WARNING."""
        assert determine_severity(0.2) == IssueSeverity.WARNING

    def test_0_201_is_error(self):
        assert determine_severity(0.201) == IssueSeverity.ERROR

    def test_exact_0_5_is_error(self):
        """0.5 is not > 0.5, so it stays ERROR."""
        assert determine_severity(0.5) == IssueSeverity.ERROR

    def test_0_501_is_critical(self):
        assert determine_severity(0.501) == IssueSeverity.CRITICAL

    def test_1_0_is_critical(self):
        assert determine_severity(1.0) == IssueSeverity.CRITICAL


class TestComputeAllQualityMetrics:
    """Tests for compute_all_quality_metrics(mesh, adapter, metrics, thresholds)."""

    @staticmethod
    def _make_mock_adapter(compute_map=None):
        """
        Create a mock adapter object.

        Parameters
        ----------
        compute_map : dict or None
            Maps metric_name -> numpy array of values.
            If None, returns a uniform array of length 4 for any metric.
        """
        class MockAdapter:
            def __init__(self, cmap):
                self._compute_map = cmap

            def from_canonical(self, mesh):
                return mesh

            def compute_cell_quality(self, native_mesh, metric):
                if self._compute_map is not None and metric in self._compute_map:
                    val = self._compute_map[metric]
                    if isinstance(val, Exception):
                        raise val
                    return np.asarray(val, dtype=float)
                # default: 4 cells with value 0.5
                return np.full(4, 0.5)

        return MockAdapter(compute_map)

    @staticmethod
    def _make_mesh():
        """Create a minimal valid MeshInput for testing."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)
        faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ], dtype=int)
        return MeshInput(vertices=vertices, faces=faces, mesh_id="test_mesh")

    # -- Default metrics tests --

    def test_default_metrics_are_computed(self):
        """When metrics=None, the four default metrics are computed."""
        adapter = self._make_mock_adapter()
        mesh = self._make_mesh()
        results = compute_all_quality_metrics(mesh, adapter)

        metric_names = [r.metric_name for r in results]
        expected_defaults = ['aspect_ratio', 'skewness', 'min_angle', 'max_angle']
        assert set(metric_names) == set(expected_defaults)

    def test_creates_quality_metrics_objects(self):
        """Each result must be a QualityMetrics instance."""
        adapter = self._make_mock_adapter()
        mesh = self._make_mesh()
        results = compute_all_quality_metrics(mesh, adapter)
        for r in results:
            assert isinstance(r, QualityMetrics)

    def test_provided_metrics_override_defaults(self):
        """Passing explicit metrics list overrides defaults."""
        adapter = self._make_mock_adapter()
        mesh = self._make_mesh()
        results = compute_all_quality_metrics(mesh, adapter, metrics=['aspect_ratio'])
        assert len(results) == 1
        assert results[0].metric_name == 'aspect_ratio'

    # -- Threshold tests --

    def test_uses_provided_thresholds(self):
        """Explicitly passed thresholds should be used."""
        custom_thresholds = {'aspect_ratio': 0.9}
        adapter = self._make_mock_adapter()
        mesh = self._make_mesh()
        results = compute_all_quality_metrics(
            mesh, adapter,
            metrics=['aspect_ratio'],
            thresholds=custom_thresholds,
        )
        assert len(results) == 1
        assert results[0].threshold == 0.9

    def test_uses_default_thresholds_when_none(self):
        """When thresholds=None, DEFAULT_THRESHOLDS should be used."""
        adapter = self._make_mock_adapter()
        mesh = self._make_mesh()
        results = compute_all_quality_metrics(
            mesh, adapter,
            metrics=['aspect_ratio'],
            thresholds=None,
        )
        assert len(results) == 1
        assert results[0].threshold == DEFAULT_THRESHOLDS['aspect_ratio']

    # -- Exception handling tests --

    def test_silently_catches_exceptions_per_metric(self):
        """Metrics that raise should be silently skipped."""
        adapter = self._make_mock_adapter(compute_map={
            'aspect_ratio': np.array([0.5, 0.6, 0.7, 0.8]),
            'skewness': RuntimeError("compute error"),
            'min_angle': np.array([30.0, 40.0, 50.0, 60.0]),
        })
        mesh = self._make_mesh()
        results = compute_all_quality_metrics(
            mesh, adapter,
            metrics=['aspect_ratio', 'skewness', 'min_angle'],
        )
        metric_names = [r.metric_name for r in results]
        assert 'skewness' not in metric_names
        assert 'aspect_ratio' in metric_names
        assert 'min_angle' in metric_names

    def test_empty_values_skipped(self):
        """Metrics returning empty arrays should not produce results."""
        adapter = self._make_mock_adapter(compute_map={
            'aspect_ratio': np.array([]),
        })
        mesh = self._make_mesh()
        results = compute_all_quality_metrics(
            mesh, adapter,
            metrics=['aspect_ratio'],
        )
        assert len(results) == 0


class TestComputeViolationStats:
    """Tests for compute_violation_stats(metrics_list)."""

    def test_empty_list_returns_zero_totals(self):
        """Calling with an empty list should return zeroed-out stats."""
        # NOTE: There is a known bug on line 99 of utils/metrics.py.
        # The generator expression:
        #   sum(len(m.total_count) for m in metrics_list if hasattr(m, 'total_count') else [len(m.values)])
        # is syntactically invalid.  QualityMetrics has no `total_count` attribute.
        # We test the observed behaviour with an empty list, which should work
        # because the generator is never iterated.
        result = compute_violation_stats([])
        assert result['total_violations'] == 0
        assert result['total_elements'] == 0
        assert result['overall_violation_rate'] == 0
        assert result['by_metric'] == {}

    def test_nonempty_list_with_quality_metrics(self):
        """
        NOTE: Line 99 of utils/metrics.py originally had a SyntaxError in the
        generator expression.  The intended logic was a ternary selecting between
        len(m.total_count) and len(m.values).  After the minimal syntax fix,
        QualityMetrics has no `total_count` attribute, so hasattr returns False
        and the function correctly falls back to len(m.values).
        """
        qm = QualityMetrics(
            operation_id='quality.aspect_ratio',
            metric_name='aspect_ratio',
            values=np.array([0.1, 0.2, 0.9]),
            threshold=0.3,
        )
        result = compute_violation_stats([qm])
        # violation_count: values < 0.3 -> 0.1, 0.2  => 2 violations
        assert result['total_violations'] == 2
        assert result['total_elements'] == 3  # len(values)
        assert result['by_metric']['aspect_ratio']['violation_count'] == 2
        assert result['by_metric']['aspect_ratio']['total_count'] == 3


# ---------------------------------------------------------------------------
# Part 2: utils/mesh_loader.py
# ---------------------------------------------------------------------------


class TestFormatSupport:
    """Tests for FORMAT_SUPPORT constant."""

    EXPECTED_EXTENSIONS = {
        '.stl', '.obj', '.ply', '.off', '.gltf', '.glb', '.vtp', '.vtk', '.3mf',
    }

    def test_all_expected_extensions_exist(self):
        """All documented format extensions must be present."""
        for ext in self.EXPECTED_EXTENSIONS:
            assert ext in FORMAT_SUPPORT, f"Missing extension: {ext}"

    def test_no_extra_extensions(self):
        """FORMAT_SUPPORT should contain exactly the documented extensions."""
        assert set(FORMAT_SUPPORT.keys()) == self.EXPECTED_EXTENSIONS

    def test_stl_supports_trimesh_and_pyvista(self):
        """.stl must list at least trimesh and pyvista."""
        libs = FORMAT_SUPPORT['.stl']
        assert 'trimesh' in libs
        assert 'pyvista' in libs

    def test_vtp_supports_pyvista_only(self):
        """.vtp must list only pyvista."""
        assert FORMAT_SUPPORT['.vtp'] == ['pyvista']


class TestGetSupportedFormats:
    """Tests for get_supported_formats()."""

    def test_returns_copy_not_reference(self):
        """Modifying the returned dict must not affect FORMAT_SUPPORT."""
        fmts = get_supported_formats()
        fmts['.new_ext'] = ['fake_lib']
        assert '.new_ext' not in get_supported_formats()

    def test_contents_match_format_support(self):
        """Returned dict must equal FORMAT_SUPPORT."""
        result = get_supported_formats()
        assert result == FORMAT_SUPPORT


class TestIsFormatSupported:
    """Tests for is_format_supported(suffix)."""

    def test_uppercase_stl(self):
        """Case-insensitive: '.STL' should return True."""
        assert is_format_supported('.STL') is True

    def test_lowercase_stl(self):
        assert is_format_supported('.stl') is True

    def test_unsupported_format(self):
        assert is_format_supported('.xyz') is False

    def test_mixed_case(self):
        assert is_format_supported('.ObJ') is True


class TestCreateMeshFromArrays:
    """Tests for create_mesh_from_arrays(vertices, faces, mesh_id, library)."""

    def test_basic_creation_with_numpy_arrays(self):
        """Should create a valid MeshInput from numpy arrays."""
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2]], dtype=int)
        mesh = create_mesh_from_arrays(verts, faces)
        assert isinstance(mesh, MeshInput)
        np.testing.assert_array_equal(mesh.vertices, verts)
        np.testing.assert_array_equal(mesh.faces, faces)

    def test_mesh_id_set_correctly(self):
        """mesh_id parameter should be stored on the result."""
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2]], dtype=int)
        mesh = create_mesh_from_arrays(verts, faces, mesh_id="my_mesh")
        assert mesh.mesh_id == "my_mesh"

    def test_source_format_is_arrays(self):
        """source_format must always be 'arrays'."""
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2]], dtype=int)
        mesh = create_mesh_from_arrays(verts, faces)
        assert mesh.source_format == 'arrays'

    def test_list_inputs_are_converted(self):
        """Plain Python lists should be accepted and converted to numpy arrays."""
        verts = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        faces = [[0, 1, 2]]
        mesh = create_mesh_from_arrays(verts, faces)
        assert isinstance(mesh.vertices, np.ndarray)
        assert isinstance(mesh.faces, np.ndarray)
        assert mesh.vertices.shape == (3, 3)
        assert mesh.faces.shape == (1, 3)

    def test_library_parameter_accepted_but_unused(self):
        """The library kwarg is a placeholder; it should not raise."""
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2]], dtype=int)
        # Should not raise even with an arbitrary library name
        mesh = create_mesh_from_arrays(verts, faces, library='nonexistent')
        assert isinstance(mesh, MeshInput)


class TestGetAdapter:
    """Tests for _get_adapter(library)."""

    def test_unknown_library_raises_value_error(self):
        """An unrecognized library name must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown library"):
            _get_adapter('not_a_real_lib')


class TestIsLibraryAvailable:
    """Tests for _is_library_available(library)."""

    def test_known_library_returns_bool(self):
        """
        For a known library name, the result is either True or False
        depending on whether it is installed.
        """
        result = _is_library_available('trimesh')
        assert isinstance(result, bool)

    def test_unknown_library_raises_value_error(self):
        """
        _get_adapter raises ValueError (not ImportError) for unknown libraries,
        so _is_library_available does NOT catch it and ValueError propagates.
        This is the observed behaviour.
        """
        with pytest.raises(ValueError, match="Unknown library"):
            _is_library_available('totally_fake_library')
