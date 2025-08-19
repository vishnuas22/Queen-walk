#!/usr/bin/env python3
"""
Simple test script to validate the quantum intelligence engine migration
"""

import sys
import os
import asyncio
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

def test_imports():
    """Test that all core modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        # Test core enums
        from quantum_intelligence.core.enums import QuantumLearningMode, QuantumState, IntelligenceLevel
        print("✅ Core enums imported successfully")
        
        # Test data structures
        from quantum_intelligence.core.data_structures import QuantumLearningContext, QuantumResponse
        print("✅ Core data structures imported successfully")
        
        # Test neural networks
        from quantum_intelligence.neural_networks.quantum_processor import QuantumResponseProcessor
        from quantum_intelligence.neural_networks.difficulty_network import AdaptiveDifficultyNetwork
        print("✅ Neural networks imported successfully")
        
        # Test learning modes
        from quantum_intelligence.learning_modes.adaptive_quantum import AdaptiveQuantumMode
        from quantum_intelligence.learning_modes.socratic_discovery import SocraticDiscoveryMode
        print("✅ Learning modes imported successfully")
        
        # Test configuration
        from quantum_intelligence.config.settings import QuantumEngineConfig
        print("✅ Configuration imported successfully")
        
        # Test utilities
        from quantum_intelligence.utils.caching import MemoryCache
        from quantum_intelligence.utils.monitoring import MetricsService, HealthCheckService
        print("✅ Utilities imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_enum_values():
    """Test that enum values are correct"""
    print("\n🧪 Testing enum values...")
    
    try:
        from quantum_intelligence.core.enums import QuantumLearningMode, QuantumState, IntelligenceLevel
        
        # Test QuantumLearningMode
        assert QuantumLearningMode.ADAPTIVE_QUANTUM.value == "adaptive_quantum"
        assert QuantumLearningMode.SOCRATIC_DISCOVERY.value == "socratic_discovery"
        print("✅ QuantumLearningMode values correct")
        
        # Test QuantumState
        assert QuantumState.DISCOVERY.value == "discovery"
        assert QuantumState.MASTERY.value == "mastery"
        print("✅ QuantumState values correct")
        
        # Test IntelligenceLevel
        assert IntelligenceLevel.BASIC == 1
        assert IntelligenceLevel.QUANTUM == 5
        print("✅ IntelligenceLevel values correct")
        
        return True
        
    except Exception as e:
        print(f"❌ Enum test failed: {e}")
        return False


def test_data_structures():
    """Test data structure creation and serialization"""
    print("\n🧪 Testing data structures...")
    
    try:
        from quantum_intelligence.core.data_structures import QuantumLearningContext, QuantumResponse
        from quantum_intelligence.core.enums import QuantumLearningMode, QuantumState, IntelligenceLevel
        from datetime import datetime, timezone
        
        # Test QuantumResponse creation
        response = QuantumResponse(
            content="Test response",
            quantum_mode=QuantumLearningMode.ADAPTIVE_QUANTUM,
            quantum_state=QuantumState.DISCOVERY,
            intelligence_level=IntelligenceLevel.ENHANCED,
            personalization_score=0.8,
            engagement_prediction=0.7,
            learning_velocity_boost=0.6,
            concept_connections=["concept1", "concept2"],
            knowledge_gaps_identified=["gap1"],
            next_optimal_concepts=["next1"],
            metacognitive_insights=["insight1"],
            emotional_resonance_score=0.75,
            adaptive_recommendations=[],
            streaming_metadata={},
            quantum_analytics={},
            suggested_actions=[],
            next_steps=""
        )
        
        # Test serialization
        response_dict = response.to_dict()
        assert response_dict["content"] == "Test response"
        assert response_dict["quantum_mode"] == "adaptive_quantum"
        print("✅ QuantumResponse creation and serialization works")
        
        # Test deserialization
        restored_response = QuantumResponse.from_dict(response_dict)
        assert restored_response.content == response.content
        assert restored_response.quantum_mode == response.quantum_mode
        print("✅ QuantumResponse deserialization works")
        
        return True
        
    except Exception as e:
        print(f"❌ Data structure test failed: {e}")
        return False


async def test_learning_modes():
    """Test learning mode functionality"""
    print("\n🧪 Testing learning modes...")
    
    try:
        from quantum_intelligence.learning_modes.adaptive_quantum import AdaptiveQuantumMode
        from quantum_intelligence.learning_modes.socratic_discovery import SocraticDiscoveryMode
        from quantum_intelligence.core.data_structures import QuantumLearningContext
        from quantum_intelligence.core.enums import QuantumLearningMode, QuantumState, IntelligenceLevel
        
        # Create test context
        context = QuantumLearningContext(
            user_id="test_user",
            session_id="test_session",
            current_quantum_state=QuantumState.DISCOVERY,
            learning_dna=None,  # Will be handled by the mode
            mood_adaptation=None,  # Will be handled by the mode
            active_mode=QuantumLearningMode.ADAPTIVE_QUANTUM,
            intelligence_level=IntelligenceLevel.ENHANCED,
            knowledge_graph_state={},
            analytics_insights={},
            gamification_state={},
            metacognitive_progress={},
            temporal_context={},
            performance_metrics={},
            adaptive_parameters=None
        )
        
        # Test adaptive quantum mode
        adaptive_mode = AdaptiveQuantumMode()
        analysis = await adaptive_mode.analyze_user_input("What is machine learning?", context)
        assert isinstance(analysis, dict)
        assert "concepts" in analysis
        print("✅ AdaptiveQuantumMode analysis works")
        
        # Test socratic discovery mode
        socratic_mode = SocraticDiscoveryMode()
        analysis = await socratic_mode.analyze_user_input("Why does this work?", context)
        assert isinstance(analysis, dict)
        assert "thinking_level" in analysis
        print("✅ SocraticDiscoveryMode analysis works")
        
        return True
        
    except Exception as e:
        print(f"❌ Learning mode test failed: {e}")
        return False


async def test_caching():
    """Test caching functionality"""
    print("\n🧪 Testing caching...")
    
    try:
        from quantum_intelligence.utils.caching import MemoryCache
        
        # Create cache
        cache = MemoryCache(max_size=10, default_ttl=60)
        
        # Test set and get
        await cache.set("test_key", "test_value")
        value = await cache.get("test_key")
        assert value == "test_value"
        print("✅ Cache set/get works")
        
        # Test exists
        exists = await cache.exists("test_key")
        assert exists is True
        print("✅ Cache exists works")
        
        # Test delete
        deleted = await cache.delete("test_key")
        assert deleted is True
        
        # Test get after delete
        value = await cache.get("test_key")
        assert value is None
        print("✅ Cache delete works")
        
        await cache.close()
        return True
        
    except Exception as e:
        print(f"❌ Caching test failed: {e}")
        return False


def test_monitoring():
    """Test monitoring functionality"""
    print("\n🧪 Testing monitoring...")
    
    try:
        from quantum_intelligence.utils.monitoring import MetricsService, HealthCheckService
        
        # Test metrics service
        metrics = MetricsService(enabled=True, prometheus_enabled=False)
        
        # Test counter
        metrics.increment_counter("test_counter", 5.0)
        metrics_data = metrics.get_metrics()
        assert metrics_data["counters"]["test_counter"] == 5.0
        print("✅ Metrics counter works")
        
        # Test gauge
        metrics.set_gauge("test_gauge", 10.0)
        metrics_data = metrics.get_metrics()
        assert metrics_data["gauges"]["test_gauge"] == 10.0
        print("✅ Metrics gauge works")
        
        # Test health service
        health = HealthCheckService(enabled=True, check_interval=60)  # Long interval for testing
        
        def test_check():
            return True
        
        health.register_check("test_check", test_check)
        print("✅ Health check registration works")
        
        return True
        
    except Exception as e:
        print(f"❌ Monitoring test failed: {e}")
        return False


async def main():
    """Run all tests"""
    print("🚀 Starting Quantum Intelligence Engine Migration Tests\n")
    
    tests = [
        ("Import Tests", test_imports),
        ("Enum Tests", test_enum_values),
        ("Data Structure Tests", test_data_structures),
        ("Learning Mode Tests", test_learning_modes),
        ("Caching Tests", test_caching),
        ("Monitoring Tests", test_monitoring),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running {test_name}")
        print('='*50)
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
                
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print('='*50)
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Migration is successful!")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
