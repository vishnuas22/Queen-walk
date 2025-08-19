"""
Comprehensive Test Suite for Advanced Predictive Learning Analytics Engine

Tests all predictive analytics components including predictive modeling,
learning analytics, intervention systems, outcome forecasting, and
orchestration for maximum coverage and reliability.

🧪 TEST COVERAGE:
- Predictive modeling with quantum-enhanced transformers
- Learning analytics dashboard generation and insights
- Intervention trigger detection and strategy generation
- Outcome forecasting and resource requirement prediction
- Analytics orchestration and cross-component integration

Author: MasterX AI Team - Predictive Analytics Division
Version: 1.0 - Phase 10 Advanced Predictive Learning Analytics Engine
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
import sys
import os

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import predictive analytics components
from quantum_intelligence.services.predictive_analytics import (
    PredictiveAnalyticsEngine,
    AnalyticsOrchestrator,
    PredictiveModelingEngine,
    LearningAnalyticsEngine,
    InterventionSystemsEngine,
    OutcomeForecastingEngine,
    
    # Data structures
    PredictionRequest,
    AnalyticsRequest,
    LearningGoal,
    
    # Enums
    PredictionType,
    PredictionHorizon,
    AnalyticsView,
    TimeRange,
    ForecastHorizon,
    
    # Convenience functions
    create_predictive_analytics_engine
)

# Import personalization components for testing
from quantum_intelligence.services.personalization import (
    LearningDNA, BehaviorEvent, BehaviorType, LearningStyle,
    CognitivePattern, PersonalityTrait, LearningPace, MotivationStyle
)

class TestPredictiveAnalyticsEngine:
    """Test suite for the main PredictiveAnalyticsEngine"""
    
    @pytest.fixture
    async def predictive_analytics_engine(self):
        """Create predictive analytics engine for testing"""
        return await create_predictive_analytics_engine()
    
    @pytest.fixture
    def sample_learning_dna(self):
        """Sample learning DNA for testing"""
        return LearningDNA(
            user_id='test_user_001',
            learning_style=LearningStyle.VISUAL,
            cognitive_patterns=[CognitivePattern.ANALYTICAL, CognitivePattern.SEQUENTIAL],
            personality_traits={
                PersonalityTrait.OPENNESS: 0.8,
                PersonalityTrait.CONSCIENTIOUSNESS: 0.7,
                PersonalityTrait.EXTRAVERSION: 0.6
            },
            preferred_pace=LearningPace.MODERATE,
            motivation_style=MotivationStyle.ACHIEVEMENT,
            optimal_difficulty_level=0.7,
            processing_speed=0.8,
            social_learning_preference=0.4,
            feedback_sensitivity=0.6,
            metacognitive_awareness=0.7,
            creativity_index=0.8,
            focus_duration_minutes=45,
            challenge_tolerance=0.7,
            confidence_score=0.75,
            profile_completeness=0.85
        )
    
    @pytest.fixture
    def sample_performance_data(self):
        """Sample performance data for testing"""
        return [
            {
                'session_id': 'session_001',
                'subject': 'mathematics',
                'difficulty_level': 0.6,
                'accuracy': 0.8,
                'completion_rate': 0.9,
                'duration': 30,
                'timestamp': datetime.now() - timedelta(days=1)
            },
            {
                'session_id': 'session_002',
                'subject': 'physics',
                'difficulty_level': 0.7,
                'accuracy': 0.75,
                'completion_rate': 0.85,
                'duration': 35,
                'timestamp': datetime.now() - timedelta(hours=12)
            },
            {
                'session_id': 'session_003',
                'subject': 'mathematics',
                'difficulty_level': 0.65,
                'accuracy': 0.85,
                'completion_rate': 0.95,
                'duration': 28,
                'timestamp': datetime.now() - timedelta(hours=6)
            }
        ]
    
    @pytest.fixture
    def sample_behavioral_history(self):
        """Sample behavioral history for testing"""
        return [
            BehaviorEvent(
                user_id='test_user_001',
                event_type=BehaviorType.ENGAGEMENT,
                engagement_level=0.8,
                duration=1800,  # 30 minutes
                success_indicator=True,
                emotional_state='positive',
                timestamp=datetime.now() - timedelta(hours=2)
            ),
            BehaviorEvent(
                user_id='test_user_001',
                event_type=BehaviorType.INTERACTION,
                engagement_level=0.9,
                duration=1200,  # 20 minutes
                success_indicator=True,
                emotional_state='focused',
                timestamp=datetime.now() - timedelta(hours=1)
            )
        ]
    
    @pytest.fixture
    def sample_learning_goals(self):
        """Sample learning goals for testing"""
        return [
            LearningGoal(
                goal_id='goal_001',
                user_id='test_user_001',
                goal_name='Master Linear Algebra',
                description='Achieve proficiency in linear algebra concepts',
                target_skills=['matrix_operations', 'vector_spaces', 'eigenvalues'],
                success_criteria={'accuracy': 0.85, 'completion_rate': 0.9},
                target_completion_date=datetime.now() + timedelta(days=30),
                priority_level='high'
            )
        ]
    
    @pytest.mark.asyncio
    async def test_predict_learning_outcomes(
        self, predictive_analytics_engine, sample_learning_dna, 
        sample_performance_data, sample_behavioral_history
    ):
        """Test learning outcome prediction"""
        
        # Predict learning outcomes
        prediction_result = await predictive_analytics_engine.predict_learning_outcomes(
            user_id='test_user_001',
            learning_dna=sample_learning_dna,
            recent_performance=sample_performance_data,
            behavioral_history=sample_behavioral_history,
            prediction_horizon='medium_term'
        )
        
        # Verify prediction result
        assert prediction_result['user_id'] == 'test_user_001'
        assert prediction_result['prediction_type'] == 'learning_outcome'
        assert prediction_result['prediction_horizon'] == 'medium_term'
        assert 'predicted_outcome' in prediction_result
        assert 'confidence_score' in prediction_result
        assert 'risk_level' in prediction_result
        assert 'trajectory_points' in prediction_result
        assert 'recommendations' in prediction_result
        
        # Verify prediction quality
        assert 0 <= prediction_result['confidence_score'] <= 1
        assert prediction_result['risk_level'] in ['low', 'moderate', 'high', 'critical']
        
        print(f"✅ Learning outcome prediction successful")
        print(f"   Confidence Score: {prediction_result['confidence_score']:.3f}")
        print(f"   Risk Level: {prediction_result['risk_level']}")
        print(f"   Trajectory Points: {len(prediction_result['trajectory_points'])}")
    
    @pytest.mark.asyncio
    async def test_generate_learning_analytics(
        self, predictive_analytics_engine, sample_learning_dna,
        sample_performance_data, sample_behavioral_history
    ):
        """Test learning analytics dashboard generation"""
        
        # Generate learning analytics
        analytics_result = await predictive_analytics_engine.generate_learning_analytics(
            user_id='test_user_001',
            learning_dna=sample_learning_dna,
            recent_performance=sample_performance_data,
            behavioral_history=sample_behavioral_history,
            analytics_view='overview',
            time_range='last_month'
        )
        
        # Verify analytics result
        assert analytics_result['user_id'] == 'test_user_001'
        assert 'dashboard_id' in analytics_result
        assert analytics_result['analytics_view'] == 'overview'
        assert analytics_result['time_range'] == 'last_month'
        assert 'performance_metrics' in analytics_result
        assert 'engagement_analytics' in analytics_result
        assert 'key_insights' in analytics_result
        assert 'recommendations' in analytics_result
        assert 'charts' in analytics_result
        
        # Verify analytics quality
        assert 0 <= analytics_result['analytics_confidence'] <= 1
        assert len(analytics_result['key_insights']) > 0
        assert len(analytics_result['recommendations']) > 0
        
        print(f"✅ Learning analytics generation successful")
        print(f"   Analytics Confidence: {analytics_result['analytics_confidence']:.3f}")
        print(f"   Key Insights: {len(analytics_result['key_insights'])}")
        print(f"   Charts Generated: {len(analytics_result['charts'])}")
    
    @pytest.mark.asyncio
    async def test_detect_intervention_needs(
        self, predictive_analytics_engine, sample_learning_dna,
        sample_performance_data, sample_behavioral_history
    ):
        """Test intervention needs detection"""
        
        # Detect intervention needs
        intervention_result = await predictive_analytics_engine.detect_intervention_needs(
            user_id='test_user_001',
            learning_dna=sample_learning_dna,
            recent_performance=sample_performance_data,
            behavioral_history=sample_behavioral_history
        )
        
        # Verify intervention result
        assert intervention_result['user_id'] == 'test_user_001'
        assert 'triggers_detected' in intervention_result
        assert 'intervention_triggers' in intervention_result
        assert 'intervention_strategies' in intervention_result
        assert 'detection_timestamp' in intervention_result
        
        # Verify intervention data structure
        triggers_detected = intervention_result['triggers_detected']
        assert triggers_detected >= 0
        
        if triggers_detected > 0:
            # Verify trigger structure
            first_trigger = intervention_result['intervention_triggers'][0]
            assert 'trigger_id' in first_trigger
            assert 'condition' in first_trigger
            assert 'severity_score' in first_trigger
            assert 'detected_issues' in first_trigger
            assert 'confidence_score' in first_trigger
            
            # Verify strategy structure
            first_strategy = intervention_result['intervention_strategies'][0]
            assert 'strategy_id' in first_strategy
            assert 'intervention_type' in first_strategy
            assert 'urgency_level' in first_strategy
            assert 'strategy_name' in first_strategy
            assert 'predicted_success_probability' in first_strategy
        
        print(f"✅ Intervention needs detection successful")
        print(f"   Triggers Detected: {triggers_detected}")
        print(f"   Strategies Generated: {len(intervention_result['intervention_strategies'])}")
    
    @pytest.mark.asyncio
    async def test_forecast_learning_outcomes(
        self, predictive_analytics_engine, sample_learning_dna,
        sample_performance_data, sample_behavioral_history, sample_learning_goals
    ):
        """Test learning outcome forecasting"""
        
        # Forecast learning outcomes
        forecast_result = await predictive_analytics_engine.forecast_learning_outcomes(
            user_id='test_user_001',
            learning_dna=sample_learning_dna,
            recent_performance=sample_performance_data,
            behavioral_history=sample_behavioral_history,
            learning_goals=sample_learning_goals,
            forecast_horizon='next_month'
        )
        
        # Verify forecast result
        assert forecast_result['user_id'] == 'test_user_001'
        assert 'forecast_id' in forecast_result
        assert forecast_result['forecast_horizon'] == 'next_month'
        assert 'predicted_outcome' in forecast_result
        assert 'achievement_probability' in forecast_result
        assert 'confidence_level' in forecast_result
        assert 'estimated_completion_date' in forecast_result
        assert 'required_study_hours' in forecast_result
        assert 'milestone_timeline' in forecast_result
        assert 'optimization_recommendations' in forecast_result
        
        # Verify forecast quality
        assert 0 <= forecast_result['achievement_probability'] <= 1
        assert forecast_result['confidence_level'] in ['low', 'moderate', 'high', 'very_high']
        assert forecast_result['required_study_hours'] > 0
        
        print(f"✅ Learning outcome forecasting successful")
        print(f"   Achievement Probability: {forecast_result['achievement_probability']:.3f}")
        print(f"   Confidence Level: {forecast_result['confidence_level']}")
        print(f"   Required Study Hours: {forecast_result['required_study_hours']}")
    
    @pytest.mark.asyncio
    async def test_execute_comprehensive_analytics(
        self, predictive_analytics_engine, sample_learning_dna,
        sample_performance_data, sample_behavioral_history, sample_learning_goals
    ):
        """Test comprehensive analytics execution"""
        
        # Execute comprehensive analytics
        comprehensive_result = await predictive_analytics_engine.execute_comprehensive_analytics(
            user_id='test_user_001',
            learning_dna=sample_learning_dna,
            recent_performance=sample_performance_data,
            behavioral_history=sample_behavioral_history,
            learning_goals=sample_learning_goals
        )
        
        # Verify comprehensive result
        assert comprehensive_result['user_id'] == 'test_user_001'
        assert 'workflow_id' in comprehensive_result
        assert 'predictive_insights' in comprehensive_result
        assert 'learning_analytics_summary' in comprehensive_result
        assert 'intervention_recommendations' in comprehensive_result
        assert 'outcome_forecasts' in comprehensive_result
        assert 'comprehensive_insights' in comprehensive_result
        assert 'priority_actions' in comprehensive_result
        assert 'risk_assessment' in comprehensive_result
        assert 'overall_effectiveness_score' in comprehensive_result
        assert 'confidence_score' in comprehensive_result
        assert 'immediate_actions' in comprehensive_result
        assert 'short_term_strategies' in comprehensive_result
        assert 'long_term_planning' in comprehensive_result
        
        # Verify comprehensive analytics quality
        assert 0 <= comprehensive_result['overall_effectiveness_score'] <= 1
        assert 0 <= comprehensive_result['confidence_score'] <= 1
        assert len(comprehensive_result['comprehensive_insights']) > 0
        assert len(comprehensive_result['priority_actions']) > 0
        
        print(f"✅ Comprehensive analytics execution successful")
        print(f"   Overall Effectiveness: {comprehensive_result['overall_effectiveness_score']:.3f}")
        print(f"   Confidence Score: {comprehensive_result['confidence_score']:.3f}")
        print(f"   Comprehensive Insights: {len(comprehensive_result['comprehensive_insights'])}")
        print(f"   Priority Actions: {len(comprehensive_result['priority_actions'])}")
        print(f"   Processing Time: {comprehensive_result['processing_time_ms']:.1f}ms")
    
    @pytest.mark.asyncio
    async def test_engine_status(self, predictive_analytics_engine):
        """Test engine status reporting"""
        
        # Get engine status
        status = predictive_analytics_engine.get_engine_status()
        
        # Verify status structure
        assert 'engine_version' in status
        assert 'initialization_time' in status
        assert 'uptime_seconds' in status
        assert 'metrics' in status
        assert 'components_status' in status
        assert 'orchestration_status' in status
        
        # Verify components are active
        components = status['components_status']
        assert components['predictive_modeling'] == 'active'
        assert components['learning_analytics'] == 'active'
        assert components['intervention_systems'] == 'active'
        assert components['outcome_forecasting'] == 'active'
        assert components['analytics_orchestrator'] == 'active'
        
        # Verify metrics structure
        metrics = status['metrics']
        assert 'total_predictions_made' in metrics
        assert 'total_analytics_generated' in metrics
        assert 'total_interventions_triggered' in metrics
        assert 'total_forecasts_created' in metrics
        
        print(f"✅ Engine status reporting successful")
        print(f"   Engine Version: {status['engine_version']}")
        print(f"   Uptime: {status['uptime_seconds']:.1f} seconds")
        print(f"   All Components Active: {all(s == 'active' for s in components.values())}")


class TestIntegration:
    """Integration tests for the complete predictive analytics system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_predictive_analytics(self):
        """Test complete end-to-end predictive analytics workflow"""
        
        # Initialize engine
        engine = await create_predictive_analytics_engine()
        
        # Create comprehensive test data
        user_id = 'integration_test_user'
        
        learning_dna = LearningDNA(
            user_id=user_id,
            learning_style=LearningStyle.MULTIMODAL,
            cognitive_patterns=[CognitivePattern.ANALYTICAL, CognitivePattern.ACTIVE],
            personality_traits={
                PersonalityTrait.OPENNESS: 0.9,
                PersonalityTrait.CONSCIENTIOUSNESS: 0.8
            },
            preferred_pace=LearningPace.FAST,
            motivation_style=MotivationStyle.MASTERY,
            optimal_difficulty_level=0.8,
            processing_speed=0.9,
            confidence_score=0.8,
            profile_completeness=0.9
        )
        
        performance_data = [
            {
                'subject': 'computer_science',
                'difficulty_level': 0.8,
                'accuracy': 0.9,
                'completion_rate': 0.95,
                'duration': 40,
                'timestamp': datetime.now() - timedelta(hours=i)
            } for i in range(1, 6)
        ]
        
        behavioral_history = [
            BehaviorEvent(
                user_id=user_id,
                event_type=BehaviorType.ENGAGEMENT,
                engagement_level=0.9,
                duration=2400,
                success_indicator=True,
                emotional_state='motivated',
                timestamp=datetime.now() - timedelta(hours=i)
            ) for i in range(1, 4)
        ]
        
        learning_goals = [
            LearningGoal(
                goal_id='advanced_cs_goal',
                user_id=user_id,
                goal_name='Advanced Computer Science Mastery',
                description='Master advanced CS concepts',
                target_skills=['algorithms', 'data_structures', 'system_design'],
                success_criteria={'accuracy': 0.9, 'completion_rate': 0.95},
                target_completion_date=datetime.now() + timedelta(days=60),
                priority_level='high'
            )
        ]
        
        # Execute comprehensive analytics
        result = await engine.execute_comprehensive_analytics(
            user_id, learning_dna, performance_data, behavioral_history, learning_goals
        )
        
        # Verify comprehensive workflow
        assert result['user_id'] == user_id
        assert 'workflow_id' in result
        assert result['overall_effectiveness_score'] > 0
        assert result['confidence_score'] > 0
        
        # Verify all components executed
        assert 'predictive_insights' in result
        assert 'learning_analytics_summary' in result
        assert 'intervention_recommendations' in result
        assert 'outcome_forecasts' in result
        
        # Verify insights and recommendations
        assert len(result['comprehensive_insights']) > 0
        assert len(result['priority_actions']) > 0
        assert len(result['immediate_actions']) > 0
        
        # Get engine status
        status = engine.get_engine_status()
        assert status['metrics']['total_predictions_made'] > 0
        assert status['metrics']['total_analytics_generated'] > 0
        assert status['metrics']['total_forecasts_created'] > 0
        
        print(f"✅ End-to-end predictive analytics workflow completed successfully")
        print(f"   Workflow ID: {result['workflow_id']}")
        print(f"   Overall Effectiveness: {result['overall_effectiveness_score']:.3f}")
        print(f"   Confidence Score: {result['confidence_score']:.3f}")
        print(f"   Processing Time: {result['processing_time_ms']:.1f}ms")
        print(f"   Components Executed: All active")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
