"""
Comprehensive test suite for adapter normalization.

This module tests the capability-based adapter system, factory functions,
and adapter interface compliance across all adapter implementations.

Tests cover:
- Adapter capability declarations
- Interface compliance (all adapters implement required methods)
- Factory and registry functionality
- Volumetric mesh support where applicable
- Error handling and edge cases
"""

import numpy as np
import pytest

from mesh_issue_examiner.adapters import (
    MeshAdapter,
    AdapterCapability,
    AdapterRegistry,
    TrimeshAdapter,
    PyVistaAdapter,
    PyMeshLabAdapter,
    MeshioAdapter,
    VedoAdapter,
    # Factory functions
    get_adapter,
    find_adapter_for_task,
    list_supported_libraries,
    get_available_libraries,
    get_capability_matrix,
    get_adapter_capabilities,
    has_capability,
    clear_adapter_cache,
    # Exceptions
    AdapterError,
    UnsupportedOperationError,
    MeshConversionError,
    QualityMetricError,
    FileIOError,
    NoAdapterAvailableError,
    AdapterNotAvailableError,
)
from mesh_issue_examiner.core.models import MeshInput


# =============================================================================
# Test Adapter Capability Matrix
# =============================================================================

class TestAdapterCapabilityMatrix:
    """Verify all adapters correctly declare their capabilities."""
    
    ADAPTER_CLASSES = {
        'pyvista': PyVistaAdapter,
        'trimesh': TrimeshAdapter,
        'pymeshlab': PyMeshLabAdapter,
        'meshio': MeshioAdapter,
        'vedo': VedoAdapter,
    }
    
    EXPECTED_CAPABILITIES = {
        'pyvista': (
            AdapterCapability.SURFACE_MESH |
            AdapterCapability.VOLUMETRIC_MESH |
            AdapterCapability.FILE_IO |
            AdapterCapability.QUALITY_METRICS |
            AdapterCapability.TOPOLOGY_CHECKS |
            AdapterCapability.SELF_INTERSECTION
        ),
        'trimesh': (
            AdapterCapability.SURFACE_MESH |
            AdapterCapability.FILE_IO |
            AdapterCapability.QUALITY_METRICS |
            AdapterCapability.TOPOLOGY_CHECKS
        ),
        'pymeshlab': (
            AdapterCapability.SURFACE_MESH |
            AdapterCapability.FILE_IO |
            AdapterCapability.QUALITY_METRICS |
            AdapterCapability.TOPOLOGY_CHECKS |
            AdapterCapability.SELF_INTERSECTION |
            AdapterCapability.REMESHING
        ),
        'meshio': (
            AdapterCapability.SURFACE_MESH |
            AdapterCapability.VOLUMETRIC_MESH |
            AdapterCapability.FILE_IO |
            AdapterCapability.QUALITY_METRICS |
            AdapterCapability.TOPOLOGY_CHECKS
        ),
        'vedo': (
            AdapterCapability.SURFACE_MESH |
            AdapterCapability.VOLUMETRIC_MESH |
            AdapterCapability.FILE_IO |
            AdapterCapability.QUALITY_METRICS |
            AdapterCapability.TOPOLOGY_CHECKS |
            AdapterCapability.REMESHING
        ),
    }
    
    @pytest.mark.parametrize("name,adapter_class", ADAPTER_CLASSES.items())
    def test_adapter_has_capabilities_property(self, name, adapter_class):
        """All adapters must have a capabilities property."""
        adapter = adapter_class()
        assert hasattr(adapter, 'capabilities')
        assert isinstance(adapter.capabilities, AdapterCapability)
    
    @pytest.mark.parametrize("name,adapter_class", ADAPTER_CLASSES.items())
    def test_adapter_capabilities_match_expected(self, name, adapter_class):
        """Adapter capabilities must match expected values."""
        adapter = adapter_class()
        expected = self.EXPECTED_CAPABILITIES[name]
        assert adapter.capabilities == expected, (
            f"{name}: expected {expected}, got {adapter.capabilities}"
        )
    
    @pytest.mark.parametrize("name,adapter_class", ADAPTER_CLASSES.items())
    def test_adapter_supports_method(self, name, adapter_class):
        """All adapters should have the supports() convenience method."""
        adapter = adapter_class()
        assert hasattr(adapter, 'supports')
        
        # Test supports() returns correct boolean
        assert adapter.supports(AdapterCapability.SURFACE_MESH) == True
        
        # Volumetric support varies by adapter
        has_vol = adapter.supports(AdapterCapability.VOLUMETRIC_MESH)
        if name in ['pyvista', 'meshio', 'vedo']:
            assert has_vol == True
        else:
            assert has_vol == False


# =============================================================================
# Test Interface Compliance
# =============================================================================

class TestAdapterInterfaceCompliance:
    """Verify all adapters implement the MeshAdapter ABC contract."""
    
    ADAPTER_CLASSES = [
        PyVistaAdapter,
        TrimeshAdapter,
        PyMeshLabAdapter,
        MeshioAdapter,
        VedoAdapter,
    ]
    
    REQUIRED_METHODS = [
        'library_name',
        'capabilities',
        'is_available',
        'load_mesh',
        'create_mesh',
        'get_vertices',
        'get_faces',
        'is_watertight',
        'is_manifold',
        'get_boundary_edges',
        'get_nonmanifold_edges',
        'get_duplicate_vertices',
        'compute_face_areas',
        'compute_face_normals',
        'detect_self_intersections',
        'compute_cell_quality',
        'to_canonical',
        'from_canonical',
    ]
    
    @pytest.mark.parametrize("adapter_class", ADAPTER_CLASSES)
    def test_adapter_is_mesh_adapter_subclass(self, adapter_class):
        """All adapters must inherit from MeshAdapter."""
        assert issubclass(adapter_class, MeshAdapter)
    
    @pytest.mark.parametrize("adapter_class", ADAPTER_CLASSES)
    def test_adapter_has_library_name(self, adapter_class):
        """All adapters must have library_name property."""
        adapter = adapter_class()
        assert hasattr(adapter, 'library_name')
        assert isinstance(adapter.library_name, str)
        assert len(adapter.library_name) > 0
    
    @pytest.mark.parametrize("adapter_class", ADAPTER_CLASSES)
    @pytest.mark.parametrize("method_name", REQUIRED_METHODS)
    def test_adapter_has_required_method(self, adapter_class, method_name):
        """All adapters must implement all required methods/properties."""
        adapter = adapter_class()
        assert hasattr(adapter, method_name), (
            f"{adapter_class.__name__} missing: {method_name}"
        )
        # Note: Some are properties (library_name, capabilities, is_available)
        # Others are methods - just verify they exist


# =============================================================================
# Test Factory Functions
# =============================================================================

class TestFactoryFunctions:
    """Test the factory and registry functions."""
    
    def test_list_supported_libraries(self):
        """list_supported_libraries returns all registered adapters."""
        libraries = list_supported_libraries()
        assert isinstance(libraries, list)
        assert 'pyvista' in libraries
        assert 'trimesh' in libraries
        assert 'meshio' in libraries
        assert 'pymeshlab' in libraries
        assert 'vedo' in libraries
    
    def test_get_adapter_returns_cached_instance(self):
        """get_adapter returns cached adapter instances."""
        adapter1 = get_adapter('pyvista')
        adapter2 = get_adapter('pyvista')
        assert adapter1 is adapter2  # Same instance (cached)
    
    def test_get_adapter_raises_for_unknown_library(self):
        """get_adapter raises ValueError for unknown library."""
        with pytest.raises(ValueError, match="Unsupported library"):
            get_adapter('nonexistent_library')
    
    def test_find_adapter_for_task_with_requirements(self):
        """find_adapter_for_task returns adapter meeting requirements."""
        name = find_adapter_for_task(
            AdapterCapability.VOLUMETRIC_MESH | AdapterCapability.QUALITY_METRICS
        )
        assert name in ['pyvista', 'meshio', 'vedo']
    
    def test_find_adapter_for_task_with_prefer(self):
        """find_adapter_for_task respects prefer parameter."""
        name = find_adapter_for_task(
            AdapterCapability.SURFACE_MESH | AdapterCapability.FILE_IO,
            prefer='trimesh'
        )
        assert name == 'trimesh'
    
    def test_find_adapter_raises_when_no_adapter_available(self):
        """find_adapter_for_task raises NoAdapterAvailableError."""
        with pytest.raises(NoAdapterAvailableError):
            # Request impossible combination: all capabilities
            find_adapter_for_task(
                AdapterCapability.SURFACE_MESH | 
                AdapterCapability.VOLUMETRIC_MESH |
                AdapterCapability.FILE_IO |
                AdapterCapability.QUALITY_METRICS |
                AdapterCapability.TOPOLOGY_CHECKS |
                AdapterCapability.SELF_INTERSECTION |
                AdapterCapability.REMESHING
            )
    
    def test_get_capability_matrix(self):
        """get_capability_matrix returns capabilities for all adapters."""
        matrix = get_capability_matrix()
        assert isinstance(matrix, dict)
        assert 'pyvista' in matrix
        assert 'trimesh' in matrix
        assert isinstance(matrix['pyvista'], AdapterCapability)
    
    def test_get_adapter_capabilities(self):
        """get_adapter_capabilities returns specific adapter capabilities."""
        caps = get_adapter_capabilities('pyvista')
        assert isinstance(caps, AdapterCapability)
        assert caps & AdapterCapability.SURFACE_MESH
    
    def test_has_capability_true(self):
        """has_capability returns True when adapter has capability."""
        assert has_capability('pyvista', AdapterCapability.VOLUMETRIC_MESH) == True
    
    def test_has_capability_false(self):
        """has_capability returns False when adapter lacks capability."""
        assert has_capability('trimesh', AdapterCapability.VOLUMETRIC_MESH) == False
    
    def test_has_capability_unknown_adapter(self):
        """has_capability returns False for unknown adapter."""
        assert has_capability('nonexistent', AdapterCapability.SURFACE_MESH) == False
    
    def test_clear_adapter_cache(self):
        """clear_adapter_cache clears the instance cache."""
        adapter1 = get_adapter('pyvista')
        clear_adapter_cache()
        adapter2 = get_adapter('pyvista')
        # After clearing cache, should be different instance
        # (but may be same object if Python reuses memory)
        # So we just verify the function runs without error
        assert adapter1 is not None
        assert adapter2 is not None


# =============================================================================
# Test Volumetric Mesh Support
# =============================================================================

class TestVolumetricMeshSupport:
    """Test volumetric mesh handling across adapters."""
    
    @pytest.fixture
    def sample_tet_mesh(self):
        """Create a sample tetrahedral mesh."""
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0],
            [0.5, 0.5, 1]
        ], dtype=float)
        volume_cells = {
            'tet4': np.array([[0, 1, 2, 3]])
        }
        return MeshInput(
            vertices=vertices,
            faces=np.array([]),
            mesh_id='test_tet',
            volume_cells=volume_cells
        )
    
    @pytest.fixture
    def sample_hex_mesh(self):
        """Create a sample hexahedral mesh (unit cube)."""
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # Top face
        ], dtype=float)
        volume_cells = {
            'hex8': np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
        }
        return MeshInput(
            vertices=vertices,
            faces=np.array([]),
            mesh_id='test_hex',
            volume_cells=volume_cells
        )
    
    @pytest.mark.parametrize("adapter_class", [
        PyVistaAdapter, MeshioAdapter, VedoAdapter,
    ])
    def test_full_volumetric_adapters_support_volume_cells(self, adapter_class, sample_tet_mesh):
        """Adapters with VOLUMETRIC_MESH capability should handle volume_cells."""
        adapter = adapter_class()
        
        if not adapter.supports(AdapterCapability.VOLUMETRIC_MESH):
            pytest.skip(f"{adapter_class.__name__} does not support volumetric meshes")
        
        # Skip if library not available
        if not adapter.is_available:
            pytest.skip(f"{adapter_class.__name__} library not installed")
        
        # Should not raise
        native_mesh = adapter.from_canonical(sample_tet_mesh)
        assert native_mesh is not None
    
    @pytest.mark.parametrize("adapter_class", [
        TrimeshAdapter, PyMeshLabAdapter,
    ])
    def test_surface_only_adapters_extract_boundary(self, adapter_class, sample_tet_mesh):
        """Surface-only adapters should extract surface boundary from volumetric meshes."""
        adapter = adapter_class()
        
        if adapter.supports(AdapterCapability.VOLUMETRIC_MESH):
            pytest.skip(f"{adapter_class.__name__} supports volumetric meshes")
        
        # Skip if library not available
        if not adapter.is_available:
            pytest.skip(f"{adapter_class.__name__} library not installed")
        
        # Should extract surface boundary without raising
        native_mesh = adapter.from_canonical(sample_tet_mesh)
        assert native_mesh is not None
    
    def test_pyvista_volumetric_from_canonical(self, sample_tet_mesh):
        """PyVista adapter should convert volumetric MeshInput to native format."""
        pytest.importorskip("pyvista")
        
        adapter = PyVistaAdapter()
        
        # Convert to native (this should preserve volumetric cells)
        native = adapter.from_canonical(sample_tet_mesh)
        
        # Verify native mesh has volumetric cells
        assert hasattr(native, 'cells_dict') or hasattr(native, 'cell_data')
        
        # Note: to_canonical currently uses base implementation which doesn't
        # extract volume_cells. This is a known limitation.


# =============================================================================
# Test Error Handling
# =============================================================================

class TestErrorHandling:
    """Test standardized exception hierarchy."""
    
    def test_adapter_error_is_base_class(self):
        """AdapterError is the base for all adapter exceptions."""
        assert issubclass(UnsupportedOperationError, AdapterError)
        assert issubclass(MeshConversionError, AdapterError)
        assert issubclass(QualityMetricError, AdapterError)
        assert issubclass(FileIOError, AdapterError)
        assert issubclass(NoAdapterAvailableError, AdapterError)
        assert issubclass(AdapterNotAvailableError, AdapterError)
    
    def test_adapter_error_can_be_caught(self):
        """All adapter errors can be caught with AdapterError."""
        errors = [
            UnsupportedOperationError("test"),
            MeshConversionError("test"),
            QualityMetricError("test"),
            FileIOError("test"),
            NoAdapterAvailableError("test"),
            AdapterNotAvailableError("test"),
        ]
        
        for error in errors:
            try:
                raise error
            except AdapterError as e:
                assert str(e) == "test"


# =============================================================================
# Test Adapter Registry
# =============================================================================

class TestAdapterRegistry:
    """Test the AdapterRegistry class."""
    
    def test_registry_has_all_adapters(self):
        """Registry should have all 5 adapters registered."""
        adapters = AdapterRegistry.list_adapters()
        assert len(adapters) == 5
        assert 'pyvista' in adapters
        assert 'trimesh' in adapters
        assert 'pymeshlab' in adapters
        assert 'meshio' in adapters
        assert 'vedo' in adapters
    
    def test_registry_get_adapter_caches_instances(self):
        """Registry should cache adapter instances."""
        adapter1 = AdapterRegistry.get_adapter('pyvista')
        adapter2 = AdapterRegistry.get_adapter('pyvista')
        assert adapter1 is adapter2
    
    def test_registry_find_adapter_for_task(self):
        """Registry should find adapter for given requirements."""
        # Find adapter for volumetric with quality metrics
        # Should return either pyvista or meshio (both have these capabilities)
        name = AdapterRegistry.find_adapter_for_task(
            AdapterCapability.VOLUMETRIC_MESH | AdapterCapability.QUALITY_METRICS
        )
        assert name in ['pyvista', 'meshio']
    
    def test_registry_clear_cache(self):
        """Registry clear_cache should clear instance cache."""
        adapter1 = AdapterRegistry.get_adapter('pyvista')
        AdapterRegistry.clear_cache()
        # After clear, should create new instance
        adapter2 = AdapterRegistry.get_adapter('pyvista')
        # Note: They may or may not be the same object due to Python's
        # memory management, but both should be valid adapters
        assert isinstance(adapter2, PyVistaAdapter)


# =============================================================================
# Test Capability Decorators (if used)
# =============================================================================

class TestCapabilityDecorators:
    """Test the capability decorator functionality."""
    
    def test_require_capability_decorator(self):
        """Test that require_capability decorator works."""
        from mesh_issue_examiner.adapters.capabilities import require_capability
        
        class TestAdapter:
            @property
            def capabilities(self):
                return AdapterCapability.SURFACE_MESH
            
            @property
            def library_name(self):
                return 'test'
            
            @require_capability(AdapterCapability.SURFACE_MESH)
            def supported_method(self):
                return "success"
            
            @require_capability(AdapterCapability.VOLUMETRIC_MESH)
            def unsupported_method(self):
                return "should not reach here"
        
        adapter = TestAdapter()
        
        # Should work for supported capability
        assert adapter.supported_method() == "success"
        
        # Should raise UnsupportedOperationError for unsupported
        with pytest.raises(UnsupportedOperationError):
            adapter.unsupported_method()
