"""
Comprehensive test suite for quantum intelligence engine modularization
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

# Import the modularized components
from quantum_intelligence.core.engine import QuantumLearningIntelligenceEngine
from quantum_intelligence.core.enums import QuantumLearningMode, QuantumState, IntelligenceLevel
from quantum_intelligence.core.data_structures import QuantumLearningContext, QuantumResponse
from quantum_intelligence.core.exceptions import QuantumEngineError, ModelLoadError
from quantum_intelligence.config.settings import QuantumEngineConfig
from quantum_intelligence.config.dependencies import DependencyContainer
from quantum_intelligence.utils.caching import MemoryCache
from quantum_intelligence.utils.monitoring import MetricsService, HealthCheckService


class TestModularizationBackwardCompatibility:
    """Test backward compatibility with original quantum engine"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        config = Mock(spec=QuantumEngineConfig)
        config.groq_api_key = "test_key"
        config.enable_neural_networks = True
        config.enable_metrics = True
        config.enable_health_checks = True
        config.primary_model = "test_model"
        config.cache_backend = "memory"
        config.max_cache_size = 100
        config.cache_ttl = 3600
        return config
    
    @pytest.fixture
    def mock_cache_service(self):
        """Mock cache service for testing"""
        return MemoryCache(max_size=100, default_ttl=3600)
    
    @pytest.fixture
    def mock_metrics_service(self):
        """Mock metrics service for testing"""
        return MetricsService(enabled=True)
    
    @pytest.fixture
    def mock_health_service(self):
        """Mock health service for testing"""
        return HealthCheckService(enabled=True)
    
    @pytest.fixture
    async def quantum_engine(self, mock_config, mock_cache_service, mock_metrics_service, mock_health_service):
        """Create quantum engine instance for testing"""
        with patch('quantum_intelligence.core.engine.AsyncGroq') as mock_groq:
            mock_groq.return_value = AsyncMock()
            
            engine = QuantumLearningIntelligenceEngine(
                config=mock_config,
                cache_service=mock_cache_service,
                metrics_service=mock_metrics_service,
                health_service=mock_health_service
            )
            
            yield engine
            await engine.close()
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, quantum_engine):
        """Test that engine initializes correctly"""
        assert quantum_engine is not None
        assert hasattr(quantum_engine, 'config')
        assert hasattr(quantum_engine, 'cache')
        assert hasattr(quantum_engine, 'metrics')
        assert hasattr(quantum_engine, 'health')
    
    @pytest.mark.asyncio
    async def test_get_quantum_response_backward_compatibility(self, quantum_engine):
        """Test that get_quantum_response maintains backward compatibility"""
        
        # Mock AI provider response
        with patch.object(quantum_engine, '_generate_fallback_response') as mock_fallback:
            mock_response = QuantumResponse(
                content="Test response",
                quantum_mode=QuantumLearningMode.ADAPTIVE_QUANTUM,
                quantum_state=QuantumState.DISCOVERY,
                intelligence_level=IntelligenceLevel.ENHANCED,
                processing_time=0.5,
                confidence=0.8
            )
            mock_fallback.return_value = mock_response
            
            # Test the main method
            response = await quantum_engine.get_quantum_response(
                user_message="Test message",
                user_id="test_user",
                session_id="test_session"
            )
            
            assert isinstance(response, QuantumResponse)
            assert response.content == "Test response"
            assert response.quantum_mode == QuantumLearningMode.ADAPTIVE_QUANTUM
    
    @pytest.mark.asyncio
    async def test_learning_mode_determination(self, quantum_engine):
        """Test learning mode determination logic"""
        
        # Test different message types
        test_cases = [
            ("Why does this work?", QuantumLearningMode.SOCRATIC_DISCOVERY),
            ("How to debug this error?", QuantumLearningMode.DEBUG_MASTERY),
            ("Give me a challenge", QuantumLearningMode.CHALLENGE_MODE),
            ("General question", QuantumLearningMode.ADAPTIVE_QUANTUM)
        ]
        
        for message, expected_mode in test_cases:
            context = QuantumLearningContext(
                user_id="test_user",
                session_id="test_session",
                message=message
            )
            
            mode = await quantum_engine._determine_learning_mode(message, context)
            assert mode == expected_mode
    
    @pytest.mark.asyncio
    async def test_fallback_response_generation(self, quantum_engine):
        """Test fallback response when AI providers fail"""
        
        # Mock all providers to fail
        quantum_engine._ai_providers = {}
        
        response = await quantum_engine._create_fallback_response(
            "Test message",
            "test_user",
            "test_session"
        )
        
        assert isinstance(response, QuantumResponse)
        assert "I'm here to help you learn" in response.content
        assert response.metadata.get("fallback") is True
    
    @pytest.mark.asyncio
    async def test_health_checks(self, quantum_engine):
        """Test health check functionality"""
        
        # Test AI provider health check
        ai_health = quantum_engine._check_ai_providers()
        assert isinstance(ai_health, bool)
        
        # Test cache health check
        cache_health = await quantum_engine._check_cache_service()
        assert isinstance(cache_health, bool)
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, quantum_engine):
        """Test metrics collection during operation"""
        
        initial_metrics = quantum_engine.metrics.get_metrics()
        
        # Mock a response generation
        with patch.object(quantum_engine, '_generate_fallback_response') as mock_fallback:
            mock_response = QuantumResponse(
                content="Test response",
                quantum_mode=QuantumLearningMode.ADAPTIVE_QUANTUM,
                quantum_state=QuantumState.DISCOVERY,
                intelligence_level=IntelligenceLevel.ENHANCED,
                processing_time=0.5,
                confidence=0.8
            )
            mock_fallback.return_value = mock_response
            
            await quantum_engine.get_quantum_response(
                user_message="Test message",
                user_id="test_user",
                session_id="test_session"
            )
        
        # Check that metrics were updated
        final_metrics = quantum_engine.metrics.get_metrics()
        assert "quantum_responses.generated" in final_metrics["counters"]
    
    def test_data_structure_serialization(self):
        """Test data structure serialization/deserialization"""
        
        # Test QuantumLearningContext
        context = QuantumLearningContext(
            user_id="test_user",
            session_id="test_session",
            message="test message"
        )
        
        context_dict = context.to_dict()
        restored_context = QuantumLearningContext.from_dict(context_dict)
        
        assert restored_context.user_id == context.user_id
        assert restored_context.session_id == context.session_id
        assert restored_context.message == context.message
        
        # Test QuantumResponse
        response = QuantumResponse(
            content="Test response",
            quantum_mode=QuantumLearningMode.ADAPTIVE_QUANTUM,
            quantum_state=QuantumState.DISCOVERY,
            intelligence_level=IntelligenceLevel.ENHANCED,
            processing_time=0.5,
            confidence=0.8
        )
        
        response_dict = response.to_dict()
        restored_response = QuantumResponse.from_dict(response_dict)
        
        assert restored_response.content == response.content
        assert restored_response.quantum_mode == response.quantum_mode
        assert restored_response.confidence == response.confidence
    
    def test_enum_values(self):
        """Test that all enum values are accessible"""
        
        # Test QuantumLearningMode
        assert QuantumLearningMode.ADAPTIVE_QUANTUM.value == "adaptive_quantum"
        assert QuantumLearningMode.SOCRATIC_DISCOVERY.value == "socratic_discovery"
        
        # Test QuantumState
        assert QuantumState.DISCOVERY.value == "discovery"
        assert QuantumState.MASTERY.value == "mastery"
        
        # Test IntelligenceLevel
        assert IntelligenceLevel.BASIC == 1
        assert IntelligenceLevel.QUANTUM == 5


class TestDependencyInjection:
    """Test dependency injection system"""
    
    def test_container_registration(self):
        """Test service registration in container"""
        container = DependencyContainer()
        
        # Test singleton registration
        test_service = Mock()
        container.register_singleton("test_service", test_service)
        
        retrieved_service = container.get("test_service")
        assert retrieved_service is test_service
    
    def test_container_factory(self):
        """Test factory registration in container"""
        container = DependencyContainer()
        
        # Test factory registration
        def test_factory():
            return Mock()
        
        container.register_factory("test_factory", test_factory)
        
        service1 = container.get("test_factory")
        service2 = container.get("test_factory")
        
        # Should be different instances from factory
        assert service1 is not service2
    
    def test_container_error_handling(self):
        """Test container error handling"""
        container = DependencyContainer()
        
        with pytest.raises(Exception):  # Should raise ConfigurationError
            container.get("nonexistent_service")


class TestCachingSystem:
    """Test caching system functionality"""
    
    @pytest.mark.asyncio
    async def test_memory_cache_operations(self):
        """Test memory cache basic operations"""
        cache = MemoryCache(max_size=10, default_ttl=60)
        
        # Test set and get
        await cache.set("test_key", "test_value")
        value = await cache.get("test_key")
        assert value == "test_value"
        
        # Test exists
        exists = await cache.exists("test_key")
        assert exists is True
        
        # Test delete
        deleted = await cache.delete("test_key")
        assert deleted is True
        
        # Test get after delete
        value = await cache.get("test_key")
        assert value is None
        
        await cache.close()
    
    @pytest.mark.asyncio
    async def test_cache_ttl(self):
        """Test cache TTL functionality"""
        cache = MemoryCache(max_size=10, default_ttl=1)  # 1 second TTL
        
        await cache.set("test_key", "test_value", ttl=1)
        
        # Should exist immediately
        value = await cache.get("test_key")
        assert value == "test_value"
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        
        # Should be expired
        value = await cache.get("test_key")
        assert value is None
        
        await cache.close()
    
    @pytest.mark.asyncio
    async def test_cache_get_or_compute(self):
        """Test cache get_or_compute functionality"""
        cache = MemoryCache(max_size=10, default_ttl=60)
        
        call_count = 0
        
        def compute_func():
            nonlocal call_count
            call_count += 1
            return f"computed_value_{call_count}"
        
        # First call should compute
        value1 = await cache.get_or_compute("test_key", compute_func)
        assert value1 == "computed_value_1"
        assert call_count == 1
        
        # Second call should use cache
        value2 = await cache.get_or_compute("test_key", compute_func)
        assert value2 == "computed_value_1"  # Same value from cache
        assert call_count == 1  # Function not called again
        
        await cache.close()


class TestMonitoringSystem:
    """Test monitoring and metrics system"""
    
    def test_metrics_service_operations(self):
        """Test metrics service basic operations"""
        metrics = MetricsService(enabled=True)
        
        # Test counter
        metrics.increment_counter("test_counter", 5.0)
        metrics_data = metrics.get_metrics()
        assert metrics_data["counters"]["test_counter"] == 5.0
        
        # Test gauge
        metrics.set_gauge("test_gauge", 10.0)
        metrics_data = metrics.get_metrics()
        assert metrics_data["gauges"]["test_gauge"] == 10.0
        
        # Test histogram
        metrics.record_histogram("test_histogram", 1.5)
        metrics.record_histogram("test_histogram", 2.5)
        metrics_data = metrics.get_metrics()
        
        histogram_stats = metrics_data["histograms"]["test_histogram"]
        assert histogram_stats["count"] == 2
        assert histogram_stats["avg"] == 2.0
    
    @pytest.mark.asyncio
    async def test_health_check_service(self):
        """Test health check service"""
        health = HealthCheckService(enabled=True, check_interval=1)
        
        # Register a test check
        def test_check():
            return True
        
        health.register_check("test_check", test_check)
        
        # Run the check
        result = await health.run_check("test_check")
        assert result is True
        
        # Get health status
        status = await health.get_health_status()
        assert status["status"] in ["healthy", "degraded", "unhealthy"]
        assert "test_check" in status["checks"]
        
        await health.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
