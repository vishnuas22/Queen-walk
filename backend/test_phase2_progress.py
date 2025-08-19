#!/usr/bin/env python3
"""
Test Phase 2 Progress - Service Extraction Validation

This script validates that Phase 2 service extraction is proceeding correctly.
"""

import sys
import asyncio
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

def test_personalization_services():
    """Test personalization services extraction"""
    print("🧪 Testing personalization services...")
    
    try:
        # Test personalization engine
        from quantum_intelligence.services.personalization.engine import PersonalizationEngine
        from quantum_intelligence.services.personalization.learning_dna import LearningDNAManager
        from quantum_intelligence.services.personalization.adaptive_parameters import AdaptiveParametersEngine
        from quantum_intelligence.services.personalization.mood_adaptation import MoodAdaptationEngine
        
        print("✅ All personalization service classes can be imported")
        
        # Test instantiation
        personalization_engine = PersonalizationEngine()
        learning_dna_manager = LearningDNAManager()
        adaptive_params_engine = AdaptiveParametersEngine()
        mood_adaptation_engine = MoodAdaptationEngine()
        
        print("✅ All personalization services can be instantiated")
        
        # Test that they have the expected methods
        assert hasattr(personalization_engine, 'analyze_learning_dna')
        assert hasattr(learning_dna_manager, 'get_learning_dna')
        assert hasattr(adaptive_params_engine, 'calculate_adaptive_parameters')
        assert hasattr(mood_adaptation_engine, 'analyze_mood_from_interaction')
        
        print("✅ All personalization services have expected methods")
        
        return True
        
    except Exception as e:
        print(f"❌ Personalization services test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_personalization_functionality():
    """Test personalization functionality"""
    print("\n🧪 Testing personalization functionality...")
    
    try:
        from quantum_intelligence.services.personalization.engine import PersonalizationEngine
        from quantum_intelligence.core.data_structures import LearningDNA
        
        # Test PersonalizationEngine
        engine = PersonalizationEngine()
        
        # Test learning DNA analysis
        learning_dna = await engine.analyze_learning_dna("test_user")
        assert isinstance(learning_dna, LearningDNA)
        assert learning_dna.user_id == "test_user"
        
        print("✅ Learning DNA analysis works")
        
        # Test mood analysis
        mood_adaptation = await engine.analyze_mood_and_adapt(
            "test_user", 
            [{"content": "I'm excited to learn!", "timestamp": "2024-01-01T12:00:00"}],
            {}
        )
        
        assert hasattr(mood_adaptation, 'current_mood')
        assert hasattr(mood_adaptation, 'energy_level')
        
        print("✅ Mood analysis and adaptation works")
        
        # Test personalization score calculation
        score = await engine.calculate_personalization_score(
            "test_user",
            "This is a test content about machine learning",
            {"topic_difficulty": 0.6}
        )
        
        assert 0.0 <= score <= 1.0
        
        print("✅ Personalization score calculation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Personalization functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_mood_adaptation_functionality():
    """Test mood adaptation functionality"""
    print("\n🧪 Testing mood adaptation functionality...")
    
    try:
        from quantum_intelligence.services.personalization.mood_adaptation import MoodAdaptationEngine
        from quantum_intelligence.core.data_structures import MoodBasedAdaptation
        
        engine = MoodAdaptationEngine()
        
        # Test mood analysis from text
        mood_analysis = await engine.analyze_mood_from_interaction(
            "test_user",
            "I'm feeling frustrated with this difficult problem",
            {"response_time": 12.0, "recent_success_rate": 0.3}
        )
        
        assert mood_analysis["detected_mood"] == "frustrated"
        assert "energy_level" in mood_analysis
        assert "stress_level" in mood_analysis
        
        print("✅ Mood detection from text works")
        
        # Test mood adaptation creation
        adaptation = await engine.create_mood_adaptation(
            "test_user",
            mood_analysis,
            {"session_length_minutes": 45}
        )
        
        assert isinstance(adaptation, MoodBasedAdaptation)
        assert adaptation.current_mood == "frustrated"
        assert adaptation.difficulty_adjustment < 0  # Should reduce difficulty for frustrated mood
        
        print("✅ Mood adaptation creation works")
        
        # Test mood progression tracking
        progression = await engine.track_mood_progression("test_user", 30)
        assert "progression" in progression
        
        print("✅ Mood progression tracking works")
        
        return True
        
    except Exception as e:
        print(f"❌ Mood adaptation functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_service_integration():
    """Test service integration through main services module"""
    print("\n🧪 Testing service integration...")
    
    try:
        # Test importing from main services module
        from quantum_intelligence.services import (
            PersonalizationEngine,
            LearningDNAManager,
            AdaptiveParametersEngine,
            MoodAdaptationEngine
        )
        
        print("✅ Services can be imported from main services module")
        
        # Test importing from main quantum_intelligence module
        from quantum_intelligence import (
            QuantumLearningMode,
            QuantumState,
            IntelligenceLevel,
            QuantumResponse
        )
        
        print("✅ Core components still work with service extraction")
        
        return True
        
    except Exception as e:
        print(f"❌ Service integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_directory_structure():
    """Test that the directory structure is correct"""
    print("\n🧪 Testing directory structure...")
    
    try:
        # Check personalization service structure
        personalization_dir = Path("quantum_intelligence/services/personalization")
        required_files = [
            "__init__.py",
            "engine.py",
            "learning_dna.py",
            "adaptive_parameters.py",
            "mood_adaptation.py"
        ]
        
        for file_name in required_files:
            file_path = personalization_dir / file_name
            assert file_path.exists(), f"Missing file: {file_path}"
        
        print("✅ Personalization service directory structure is correct")
        
        # Check analytics service structure
        analytics_dir = Path("quantum_intelligence/services/analytics")
        assert analytics_dir.exists(), "Analytics directory missing"
        assert (analytics_dir / "__init__.py").exists(), "Analytics __init__.py missing"
        
        print("✅ Analytics service directory structure is correct")
        
        return True
        
    except Exception as e:
        print(f"❌ Directory structure test failed: {e}")
        return False

async def main():
    """Run all Phase 2 progress tests"""
    print("🚀 PHASE 2 SERVICE EXTRACTION - PROGRESS VALIDATION")
    print("=" * 60)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Personalization Services", test_personalization_services),
        ("Personalization Functionality", test_personalization_functionality),
        ("Mood Adaptation Functionality", test_mood_adaptation_functionality),
        ("Service Integration", test_service_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 {test_name}")
        print('='*60)
        
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
    
    print(f"\n{'='*60}")
    print(f"🏆 PHASE 2 PROGRESS RESULTS: {passed}/{total} tests passed")
    print('='*60)
    
    if passed == total:
        print("🎉 PHASE 2 PERSONALIZATION SERVICES EXTRACTION SUCCESSFUL!")
        print("""
✅ ACHIEVEMENTS:
• Personalization Engine extracted and functional
• Learning DNA Manager implemented with pattern analysis
• Adaptive Parameters Engine with real-time adaptation
• Mood Adaptation Engine with emotional intelligence
• All services properly integrated and tested

🚀 NEXT STEPS:
• Continue with Analytics Services extraction
• Extract remaining major phases
• Implement integration tests
• Validate backward compatibility
""")
        return True
    else:
        print("⚠️  Some Phase 2 tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
