#!/usr/bin/env python3
"""
Test Phase 4 Predictive Intelligence - Service Extraction Validation

This script validates that Phase 4 predictive intelligence service extraction is working correctly.
"""

import sys
import asyncio
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

def test_predictive_services_import():
    """Test predictive services can be imported"""
    print("🧪 Testing predictive services import...")
    
    try:
        # Test predictive services
        from quantum_intelligence.services.predictive.outcomes import (
            LearningOutcomePredictionEngine,
            LearningOutcomeMetrics,
            LearningOutcomePredictionNetwork
        )
        from quantum_intelligence.services.predictive.forecasting import (
            PerformanceForecastingEngine,
            PerformanceForecast,
            PerformanceForecastingModel
        )
        from quantum_intelligence.services.predictive.behavioral import (
            BehavioralAnalysisEngine,
            CareerPathMetrics,
            SkillGapAnalysis
        )
        
        print("✅ Predictive service classes can be imported")
        
        # Test instantiation
        outcome_engine = LearningOutcomePredictionEngine()
        forecasting_engine = PerformanceForecastingEngine()
        behavioral_engine = BehavioralAnalysisEngine()
        
        print("✅ Predictive services can be instantiated")
        
        # Test that they have the expected methods
        assert hasattr(outcome_engine, 'initialize_models')
        assert hasattr(outcome_engine, 'predict_learning_outcomes')
        assert hasattr(forecasting_engine, 'forecast_performance')
        assert hasattr(behavioral_engine, 'analyze_learning_behavior')
        assert hasattr(behavioral_engine, 'predict_career_path')
        
        print("✅ Predictive services have expected methods")
        
        return True
        
    except Exception as e:
        print(f"❌ Predictive services import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_learning_outcome_prediction():
    """Test learning outcome prediction functionality"""
    print("\n🧪 Testing learning outcome prediction...")
    
    try:
        from quantum_intelligence.services.predictive.outcomes import LearningOutcomePredictionEngine
        
        engine = LearningOutcomePredictionEngine()
        
        # Test initialization
        init_result = await engine.initialize_models()
        
        assert init_result['status'] == 'success'
        assert 'models_initialized' in init_result
        assert 'total_parameters' in init_result
        
        print("✅ Learning outcome prediction engine initialization works")
        
        # Test learning outcome prediction
        learning_data = {
            'engagement_score': 0.8,
            'completion_rate': 0.9,
            'interaction_frequency': 0.7,
            'comprehension_score': 0.85,
            'problem_solving_score': 0.8,
            'learning_velocity': 0.75
        }
        
        context = {
            'difficulty_level': 0.6,
            'content_type_score': 0.7
        }
        
        prediction_result = await engine.predict_learning_outcomes(
            user_id="test_user",
            learning_data=learning_data,
            context=context
        )
        
        assert prediction_result['status'] == 'success'
        assert 'outcome_metrics' in prediction_result
        assert 'insights' in prediction_result
        assert prediction_result['user_id'] == "test_user"
        
        print("✅ Learning outcome prediction works")
        
        # Test retention probability analysis
        content_data = {
            'content_type': 'video',
            'difficulty': 0.7,
            'interaction_data': {'views': 5, 'notes': 3}
        }
        
        retention_result = await engine.analyze_retention_probability(
            user_id="test_user",
            content_data=content_data
        )
        
        assert retention_result['status'] == 'success'
        assert 'retention_analysis' in retention_result
        assert 'recommendations' in retention_result
        
        print("✅ Retention probability analysis works")
        
        # Test mastery timeline prediction
        skill_data = {
            'current_level': 0.3,
            'target_level': 0.9,
            'learning_rate': 0.05
        }
        
        timeline_result = await engine.predict_mastery_timeline(
            user_id="test_user",
            skill_data=skill_data
        )
        
        assert timeline_result['status'] == 'success'
        assert 'mastery_timeline' in timeline_result
        assert 'insights' in timeline_result
        
        print("✅ Mastery timeline prediction works")
        
        return True
        
    except Exception as e:
        print(f"❌ Learning outcome prediction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_performance_forecasting():
    """Test performance forecasting functionality"""
    print("\n🧪 Testing performance forecasting...")
    
    try:
        from quantum_intelligence.services.predictive.forecasting import PerformanceForecastingEngine
        
        engine = PerformanceForecastingEngine()
        
        # Test initialization
        init_result = await engine.initialize_models()
        
        assert init_result['status'] == 'success'
        assert 'models_initialized' in init_result
        assert 'total_parameters' in init_result
        
        print("✅ Performance forecasting engine initialization works")
        
        # Test performance forecasting
        historical_data = [
            {'performance_score': 0.7, 'engagement_level': 0.8, 'completion_rate': 0.9},
            {'performance_score': 0.75, 'engagement_level': 0.82, 'completion_rate': 0.88},
            {'performance_score': 0.8, 'engagement_level': 0.85, 'completion_rate': 0.92},
        ] * 10  # Simulate 30 data points
        
        forecast_result = await engine.forecast_performance(
            user_id="test_user",
            historical_data=historical_data,
            forecast_horizon="medium"
        )
        
        assert forecast_result['status'] == 'success'
        assert 'performance_forecast' in forecast_result
        assert 'insights' in forecast_result
        assert forecast_result['user_id'] == "test_user"
        assert forecast_result['forecast_horizon'] == "medium"
        
        print("✅ Performance forecasting works")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance forecasting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_behavioral_analysis():
    """Test behavioral analysis functionality"""
    print("\n🧪 Testing behavioral analysis...")
    
    try:
        from quantum_intelligence.services.predictive.behavioral import BehavioralAnalysisEngine
        
        engine = BehavioralAnalysisEngine()
        
        # Test learning behavior analysis
        behavioral_data = {
            'sessions_per_week': 5,
            'avg_session_duration': 45,
            'engagement_scores': [0.8, 0.85, 0.9, 0.82, 0.88],
            'content_preferences': {'video': 0.7, 'text': 0.5, 'interactive': 0.8},
            'learning_times': ['09:00', '19:30', '10:15', '20:00', '09:45']
        }
        
        behavior_result = await engine.analyze_learning_behavior(
            user_id="test_user",
            behavioral_data=behavioral_data,
            analysis_period=90
        )
        
        assert behavior_result['status'] == 'success'
        assert 'behavioral_patterns' in behavior_result
        assert 'learning_preferences' in behavior_result
        assert 'behavioral_trends' in behavior_result
        assert 'insights' in behavior_result
        
        print("✅ Learning behavior analysis works")
        
        # Test career path prediction
        profile_data = {
            'technical_skills': {'programming': 0.8, 'system_design': 0.6},
            'soft_skills': {'communication': 0.7, 'leadership': 0.5},
            'domain_expertise': {'web_development': 0.8, 'data_analysis': 0.6},
            'experience_years': 3
        }
        
        career_goals = {
            'target_role': 'Senior Software Engineer',
            'industry_preference': 'technology',
            'timeline_years': 2
        }
        
        career_result = await engine.predict_career_path(
            user_id="test_user",
            profile_data=profile_data,
            career_goals=career_goals
        )
        
        assert career_result['status'] == 'success'
        assert 'career_metrics' in career_result
        assert 'recommended_paths' in career_result
        assert 'recommendations' in career_result
        
        print("✅ Career path prediction works")
        
        # Test skill gap analysis
        current_skills = {
            'programming': 0.7,
            'system_design': 0.5,
            'leadership': 0.4,
            'communication': 0.6
        }
        
        gap_result = await engine.analyze_skill_gaps(
            user_id="test_user",
            current_skills=current_skills,
            target_role="Technical Lead",
            industry="technology"
        )
        
        assert gap_result['status'] == 'success'
        assert 'skill_gap_analysis' in gap_result
        assert 'learning_recommendations' in gap_result
        
        print("✅ Skill gap analysis works")
        
        return True
        
    except Exception as e:
        print(f"❌ Behavioral analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_predictive_integration():
    """Test predictive services integration"""
    print("\n🧪 Testing predictive services integration...")
    
    try:
        # Test importing from main services module
        from quantum_intelligence.services.predictive import (
            LearningOutcomePredictionEngine,
            PerformanceForecastingEngine,
            BehavioralAnalysisEngine,
            LearningOutcomeMetrics,
            PerformanceForecast,
            CareerPathMetrics
        )
        
        print("✅ Predictive services can be imported from main predictive module")
        
        # Test that core components still work
        from quantum_intelligence import (
            QuantumLearningMode,
            QuantumState,
            IntelligenceLevel
        )
        
        print("✅ Core components still work with predictive extraction")
        
        return True
        
    except Exception as e:
        print(f"❌ Predictive integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_combined_predictive_workflow():
    """Test combined predictive workflow"""
    print("\n🧪 Testing combined predictive workflow...")
    
    try:
        from quantum_intelligence.services.predictive.outcomes import LearningOutcomePredictionEngine
        from quantum_intelligence.services.predictive.forecasting import PerformanceForecastingEngine
        from quantum_intelligence.services.predictive.behavioral import BehavioralAnalysisEngine
        
        # Initialize engines
        outcome_engine = LearningOutcomePredictionEngine()
        forecasting_engine = PerformanceForecastingEngine()
        behavioral_engine = BehavioralAnalysisEngine()
        
        # Initialize all components
        outcome_init = await outcome_engine.initialize_models()
        forecasting_init = await forecasting_engine.initialize_models()
        
        assert outcome_init['status'] == 'success'
        assert forecasting_init['status'] == 'success'
        
        # Test workflow: behavioral analysis -> outcome prediction -> performance forecasting
        
        # 1. Analyze learning behavior
        behavioral_data = {
            'sessions_per_week': 4,
            'avg_session_duration': 50,
            'engagement_scores': [0.8, 0.85, 0.9]
        }
        
        behavior_analysis = await behavioral_engine.analyze_learning_behavior(
            user_id="test_user",
            behavioral_data=behavioral_data
        )
        
        # 2. Predict learning outcomes
        learning_data = {
            'engagement_score': 0.85,
            'completion_rate': 0.9,
            'learning_velocity': 0.8
        }
        
        outcome_prediction = await outcome_engine.predict_learning_outcomes(
            user_id="test_user",
            learning_data=learning_data
        )
        
        # 3. Forecast performance
        historical_data = [
            {'performance_score': 0.8, 'engagement_level': 0.85}
        ] * 30
        
        performance_forecast = await forecasting_engine.forecast_performance(
            user_id="test_user",
            historical_data=historical_data
        )
        
        # Verify the workflow produced meaningful results
        assert behavior_analysis['status'] == 'success'
        assert outcome_prediction['status'] == 'success'
        assert performance_forecast['status'] == 'success'
        
        print("✅ Combined predictive workflow works")
        
        return True
        
    except Exception as e:
        print(f"❌ Combined predictive workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all Phase 4 predictive intelligence tests"""
    print("🔮 PHASE 4 PREDICTIVE INTELLIGENCE SERVICE EXTRACTION - VALIDATION")
    print("=" * 80)
    
    tests = [
        ("Predictive Services Import", test_predictive_services_import),
        ("Learning Outcome Prediction", test_learning_outcome_prediction),
        ("Performance Forecasting", test_performance_forecasting),
        ("Behavioral Analysis", test_behavioral_analysis),
        ("Predictive Integration", test_predictive_integration),
        ("Combined Predictive Workflow", test_combined_predictive_workflow),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*80}")
        print(f"🧪 {test_name}")
        print('='*80)
        
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
    
    print(f"\n{'='*80}")
    print(f"🏆 PHASE 4 PREDICTIVE RESULTS: {passed}/{total} tests passed")
    print('='*80)
    
    if passed == total:
        print("🎉 PHASE 4 PREDICTIVE INTELLIGENCE SERVICE EXTRACTION SUCCESSFUL!")
        print("""
✅ ACHIEVEMENTS:
• Learning Outcome Prediction Engine with 95% accuracy target
• Performance Forecasting Models with time-series analysis
• Behavioral Analysis Engine with career path optimization
• Comprehensive skill gap analysis and learning recommendations
• Full integration with existing quantum intelligence system

🚀 NEXT STEPS:
• Continue with remaining service extractions
• Add production ML dependencies for full functionality
• Implement real-time prediction capabilities
• Validate production readiness with comprehensive testing
""")
        return True
    else:
        print("⚠️  Some Phase 4 predictive tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
