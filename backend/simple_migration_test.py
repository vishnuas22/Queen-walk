#!/usr/bin/env python3
"""
Simple migration test without external dependencies
"""

import sys
import os
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

def test_basic_imports():
    """Test basic imports without external dependencies"""
    print("🧪 Testing basic imports...")
    
    try:
        # Test core enums
        from quantum_intelligence.core.enums import QuantumLearningMode, QuantumState, IntelligenceLevel
        print("✅ Core enums imported successfully")
        
        # Test enum values
        assert QuantumLearningMode.ADAPTIVE_QUANTUM.value == "adaptive_quantum"
        assert QuantumState.DISCOVERY.value == "discovery"
        assert IntelligenceLevel.BASIC == 1
        print("✅ Enum values are correct")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_structures():
    """Test data structures without external dependencies"""
    print("\n🧪 Testing data structures...")
    
    try:
        from quantum_intelligence.core.data_structures import QuantumResponse
        from quantum_intelligence.core.enums import QuantumLearningMode, QuantumState, IntelligenceLevel
        
        # Create a simple response
        response = QuantumResponse(
            content="Test response",
            quantum_mode=QuantumLearningMode.ADAPTIVE_QUANTUM,
            quantum_state=QuantumState.DISCOVERY,
            intelligence_level=IntelligenceLevel.ENHANCED,
            personalization_score=0.8,
            engagement_prediction=0.7,
            learning_velocity_boost=0.6,
            concept_connections=["concept1"],
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
        
        return True
        
    except Exception as e:
        print(f"❌ Data structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_neural_networks():
    """Test neural network imports (without torch)"""
    print("\n🧪 Testing neural network structure...")

    try:
        # Test that the files exist
        nn_dir = Path("quantum_intelligence/neural_networks")
        assert (nn_dir / "quantum_processor.py").exists()
        assert (nn_dir / "difficulty_network.py").exists()
        print("✅ Neural network files exist")

        # Test that the package can be imported (even if torch is missing)
        import quantum_intelligence.neural_networks
        print("✅ Neural networks package can be imported")

        # Test that classes exist (even if they raise ImportError when instantiated)
        from quantum_intelligence.neural_networks import QuantumResponseProcessor, AdaptiveDifficultyNetwork
        print("✅ Neural network classes can be imported")

        # Note: We don't test instantiation since torch might not be available
        print("ℹ️  Note: Neural network functionality requires PyTorch")

        return True

    except Exception as e:
        print(f"❌ Neural network test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_learning_modes_structure():
    """Test learning modes structure"""
    print("\n🧪 Testing learning modes structure...")
    
    try:
        # Test that the files exist
        lm_dir = Path("quantum_intelligence/learning_modes")
        assert (lm_dir / "base_mode.py").exists()
        assert (lm_dir / "adaptive_quantum.py").exists()
        assert (lm_dir / "socratic_discovery.py").exists()
        print("✅ Learning mode files exist")
        
        return True
        
    except Exception as e:
        print(f"❌ Learning mode structure test failed: {e}")
        return False


def test_directory_structure():
    """Test that the modular directory structure exists"""
    print("\n🧪 Testing directory structure...")
    
    try:
        base_dir = Path("quantum_intelligence")
        
        # Test core directories
        required_dirs = [
            "core",
            "neural_networks", 
            "learning_modes",
            "services",
            "config",
            "utils"
        ]
        
        for dir_name in required_dirs:
            dir_path = base_dir / dir_name
            assert dir_path.exists(), f"Directory {dir_path} does not exist"
            assert (dir_path / "__init__.py").exists(), f"__init__.py missing in {dir_path}"
        
        print("✅ All required directories exist with __init__.py files")
        
        # Test core files
        core_files = [
            "core/enums.py",
            "core/data_structures.py", 
            "core/exceptions.py",
            "core/engine.py"
        ]
        
        for file_path in core_files:
            full_path = base_dir / file_path
            assert full_path.exists(), f"Core file {full_path} does not exist"
        
        print("✅ All core files exist")
        
        return True
        
    except Exception as e:
        print(f"❌ Directory structure test failed: {e}")
        return False


def test_configuration_structure():
    """Test configuration structure"""
    print("\n🧪 Testing configuration structure...")
    
    try:
        config_dir = Path("quantum_intelligence/config")
        
        # Test config files exist
        config_files = [
            "settings.py",
            "dependencies.py"
        ]
        
        for file_name in config_files:
            file_path = config_dir / file_name
            assert file_path.exists(), f"Config file {file_path} does not exist"
        
        print("✅ Configuration files exist")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration structure test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Starting Simple Migration Validation Tests\n")
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Basic Imports", test_basic_imports),
        ("Data Structures", test_data_structures),
        ("Neural Networks Structure", test_neural_networks),
        ("Learning Modes Structure", test_learning_modes_structure),
        ("Configuration Structure", test_configuration_structure),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running {test_name}")
        print('='*50)
        
        try:
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
        print("🎉 ALL TESTS PASSED! Migration structure is correct!")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
