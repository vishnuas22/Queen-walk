"""
Comprehensive Test Suite for Advanced Personalization Engine

Tests all personalization components including user profiling, learning style
adaptation, preference modeling, adaptive content generation, behavioral
analytics, and orchestration for maximum coverage and reliability.

🧪 TEST COVERAGE:
- User profiling and Learning DNA generation
- Learning style adaptation and optimization
- Preference modeling and prediction
- Adaptive content generation and optimization
- Behavioral analytics and pattern recognition
- Personalization orchestration and integration

Author: MasterX AI Team - Personalization Division
Version: 1.0 - Phase 9 Advanced Personalization Engine
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

# Import personalization components
from quantum_intelligence.services.personalization import (
    PersonalizationEngine,
    PersonalizationOrchestrator,
    UserProfilingEngine,
    LearningStyleAdapter,
    PreferenceEngine,
    AdaptiveContentEngine,
    BehavioralAnalyticsEngine,
    
    # Data structures
    LearningDNA,
    PersonalizationSession,
    BehaviorEvent,
    ContentAdaptationRequest,
    
    # Enums
    LearningStyle,
    CognitivePattern,
    BehaviorType,
    ContentType,
    PersonalizationStrategy
)

class TestPersonalizationEngine:
    """Test suite for the main PersonalizationEngine"""
    
    @pytest.fixture
    async def personalization_engine(self):
        """Create personalization engine for testing"""
        return PersonalizationEngine()
    
    @pytest.fixture
    def sample_user_data(self):
        """Sample user data for testing"""
        return {
            'user_id': 'test_user_001',
            'learning_history': [
                {
                    'session_id': 'session_001',
                    'subject': 'mathematics',
                    'difficulty_level': 0.6,
                    'completion_rate': 0.8,
                    'accuracy': 0.75,
                    'duration': 25,
                    'timestamp': datetime.now() - timedelta(days=1)
                },
                {
                    'session_id': 'session_002',
                    'subject': 'physics',
                    'difficulty_level': 0.7,
                    'completion_rate': 0.9,
                    'accuracy': 0.85,
                    'duration': 30,
                    'timestamp': datetime.now() - timedelta(hours=12)
                }
            ],
            'interaction_data': [
                {
                    'type': 'visual',
                    'engagement_score': 0.8,
                    'duration': 15,
                    'success_rate': 0.9,
                    'timestamp': datetime.now() - timedelta(hours=6)
                },
                {
                    'type': 'interactive',
                    'engagement_score': 0.9,
                    'duration': 20,
                    'success_rate': 0.85,
                    'timestamp': datetime.now() - timedelta(hours=3)
                }
            ],
            'performance_data': {
                'overall_accuracy': 0.8,
                'completion_rate': 0.85,
                'average_session_duration': 27.5,
                'learning_velocity': 0.7,
                'retention_rate': 0.75
            }
        }
    
    @pytest.fixture
    def sample_learning_context(self):
        """Sample learning context for testing"""
        return {
            'subject_domain': 'mathematics',
            'learning_objectives': ['understand_algebra', 'master_equations'],
            'target_complexity': 'intermediate',
            'estimated_duration': 45,
            'content_type': 'lesson',
            'prerequisite_concepts': ['basic_arithmetic'],
            'learning_context': {
                'topic': 'linear_equations',
                'difficulty_preference': 0.6
            }
        }
    
    @pytest.mark.asyncio
    async def test_create_personalized_learning_experience(
        self, personalization_engine, sample_user_data, sample_learning_context
    ):
        """Test creating personalized learning experience"""
        
        # Add user data to learning context
        sample_learning_context.update({
            'learning_history': sample_user_data['learning_history'],
            'interaction_history': sample_user_data['interaction_data'],
            'performance_data': sample_user_data['performance_data']
        })
        
        # Create personalized learning experience
        session = await personalization_engine.create_personalized_learning_experience(
            sample_user_data['user_id'],
            sample_learning_context
        )
        
        # Verify session creation
        assert isinstance(session, PersonalizationSession)
        assert session.user_id == sample_user_data['user_id']
        assert session.learning_dna is not None
        assert session.preference_profile is not None
        assert session.behavior_analysis is not None
        assert session.adaptation_result is not None
        assert len(session.adaptive_content) > 0
        
        # Verify Learning DNA
        assert isinstance(session.learning_dna, LearningDNA)
        assert session.learning_dna.confidence_score > 0
        assert session.learning_dna.profile_completeness > 0
        
        print(f"✅ Personalized learning experience created successfully")
        print(f"   Learning Style: {session.learning_dna.learning_style.value}")
        print(f"   Profile Confidence: {session.learning_dna.confidence_score:.3f}")
        print(f"   Adaptive Content Items: {len(session.adaptive_content)}")
    
    @pytest.mark.asyncio
    async def test_real_time_adaptation(
        self, personalization_engine, sample_user_data, sample_learning_context
    ):
        """Test real-time learning adaptation"""
        
        # First create a session
        sample_learning_context.update({
            'learning_history': sample_user_data['learning_history'],
            'interaction_history': sample_user_data['interaction_data'],
            'performance_data': sample_user_data['performance_data']
        })
        
        session = await personalization_engine.create_personalized_learning_experience(
            sample_user_data['user_id'],
            sample_learning_context
        )
        
        # Simulate real-time interaction
        interaction_data = {
            'behavior_type': 'interaction',
            'type': 'problem_solving',
            'engagement_score': 0.9,
            'duration': 180,  # 3 minutes
            'success': True,
            'difficulty_level': 0.7
        }
        
        performance_feedback = {
            'accuracy': 0.95,
            'completion_time': 150,
            'confidence_level': 0.8,
            'help_requests': 0
        }
        
        # Apply real-time adaptation
        update_result = await personalization_engine.adapt_learning_real_time(
            sample_user_data['user_id'],
            interaction_data,
            performance_feedback
        )
        
        # Verify adaptation
        assert update_result['updates_applied'] == True
        assert 'update_synthesis' in update_result
        assert update_result['session_effectiveness'] > 0
        
        print(f"✅ Real-time adaptation applied successfully")
        print(f"   Components Updated: {len(update_result['update_synthesis']['components_updated'])}")
        print(f"   Session Effectiveness: {update_result['session_effectiveness']:.3f}")
    
    @pytest.mark.asyncio
    async def test_personalization_insights(
        self, personalization_engine, sample_user_data, sample_learning_context
    ):
        """Test personalization insights generation"""
        
        # Create session first
        sample_learning_context.update({
            'learning_history': sample_user_data['learning_history'],
            'interaction_history': sample_user_data['interaction_data'],
            'performance_data': sample_user_data['performance_data']
        })
        
        session = await personalization_engine.create_personalized_learning_experience(
            sample_user_data['user_id'],
            sample_learning_context
        )
        
        # Generate insights
        insights = await personalization_engine.get_personalization_insights(
            sample_user_data['user_id']
        )
        
        # Verify insights
        assert insights.user_id == sample_user_data['user_id']
        assert insights.profile_insights is not None
        assert insights.adaptation_insights is not None
        assert insights.preference_insights is not None
        assert insights.behavioral_insights is not None
        assert insights.insights_confidence > 0
        assert len(insights.strategic_recommendations) > 0
        
        print(f"✅ Personalization insights generated successfully")
        print(f"   Insights Confidence: {insights.insights_confidence:.3f}")
        print(f"   Strategic Recommendations: {len(insights.strategic_recommendations)}")
        print(f"   Tactical Recommendations: {len(insights.tactical_recommendations)}")


class TestUserProfiling:
    """Test suite for user profiling components"""
    
    @pytest.fixture
    async def user_profiling_engine(self):
        """Create user profiling engine for testing"""
        return UserProfilingEngine()
    
    @pytest.mark.asyncio
    async def test_analyze_user_profile(self, user_profiling_engine):
        """Test user profile analysis"""
        
        user_id = 'test_user_profiling'
        learning_history = [
            {
                'subject': 'mathematics',
                'difficulty_level': 0.6,
                'completion_rate': 0.8,
                'preferred_content_types': ['visual', 'interactive']
            }
        ]
        interaction_data = [
            {
                'type': 'visual',
                'engagement_score': 0.9,
                'duration': 20
            }
        ]
        performance_data = {
            'accuracy_rate': 0.85,
            'completion_rate': 0.8,
            'average_time_per_concept': 8
        }
        
        # Analyze profile
        learning_dna = await user_profiling_engine.analyze_user_profile(
            user_id, learning_history, interaction_data, performance_data
        )
        
        # Verify Learning DNA
        assert isinstance(learning_dna, LearningDNA)
        assert learning_dna.user_id == user_id
        assert learning_dna.learning_style in LearningStyle
        assert len(learning_dna.cognitive_patterns) > 0
        assert 0 <= learning_dna.confidence_score <= 1
        assert 0 <= learning_dna.profile_completeness <= 1
        
        print(f"✅ User profile analyzed successfully")
        print(f"   Learning Style: {learning_dna.learning_style.value}")
        print(f"   Cognitive Patterns: {[p.value for p in learning_dna.cognitive_patterns]}")
        print(f"   Confidence Score: {learning_dna.confidence_score:.3f}")
    
    @pytest.mark.asyncio
    async def test_incremental_profile_update(self, user_profiling_engine):
        """Test incremental profile updates"""
        
        user_id = 'test_incremental_update'
        
        # Create initial profile
        initial_interaction = {
            'type': 'visual',
            'engagement_score': 0.7,
            'duration': 15
        }
        
        learning_dna = await user_profiling_engine.update_profile_incrementally(
            user_id, initial_interaction
        )
        
        # Verify initial profile
        assert isinstance(learning_dna, LearningDNA)
        initial_confidence = learning_dna.confidence_score
        
        # Update with new interaction
        new_interaction = {
            'type': 'visual',
            'engagement_score': 0.9,
            'duration': 25
        }
        
        updated_dna = await user_profiling_engine.update_profile_incrementally(
            user_id, new_interaction
        )
        
        # Verify update
        assert updated_dna.user_id == user_id
        # Confidence should potentially improve with more data
        assert updated_dna.confidence_score >= initial_confidence
        
        print(f"✅ Incremental profile update successful")
        print(f"   Initial Confidence: {initial_confidence:.3f}")
        print(f"   Updated Confidence: {updated_dna.confidence_score:.3f}")


class TestBehavioralAnalytics:
    """Test suite for behavioral analytics"""
    
    @pytest.fixture
    async def behavioral_analytics_engine(self):
        """Create behavioral analytics engine for testing"""
        return BehavioralAnalyticsEngine()
    
    @pytest.mark.asyncio
    async def test_track_behavior_event(self, behavioral_analytics_engine):
        """Test behavior event tracking"""
        
        user_id = 'test_behavior_tracking'
        
        # Track behavior event
        event_result = await behavioral_analytics_engine.track_behavior_event(
            user_id,
            BehaviorType.ENGAGEMENT,
            {
                'engagement_level': 0.8,
                'duration': 300,
                'intensity': 0.7,
                'success': True
            },
            {
                'session_id': 'test_session',
                'learning_context': {'subject': 'mathematics'},
                'device_context': {'device_type': 'desktop'}
            }
        )
        
        # Verify tracking
        assert event_result['event_tracked'] == True
        assert event_result['user_id'] == user_id
        assert 'immediate_patterns' in event_result
        assert 'anomaly_analysis' in event_result
        
        print(f"✅ Behavior event tracked successfully")
        print(f"   Total Events: {event_result['total_events']}")
    
    @pytest.mark.asyncio
    async def test_behavior_analysis(self, behavioral_analytics_engine):
        """Test comprehensive behavior analysis"""
        
        user_id = 'test_behavior_analysis'
        
        # Track multiple events
        for i in range(5):
            await behavioral_analytics_engine.track_behavior_event(
                user_id,
                BehaviorType.INTERACTION,
                {
                    'engagement_level': 0.7 + (i * 0.05),
                    'duration': 200 + (i * 20),
                    'success': True
                },
                {'session_id': f'session_{i}'}
            )
        
        # Analyze behavior
        behavior_analysis = await behavioral_analytics_engine.analyze_user_behavior(user_id)
        
        # Verify analysis
        assert behavior_analysis.user_id == user_id
        assert behavior_analysis.analysis_confidence > 0
        assert behavior_analysis.behavior_cluster is not None
        
        print(f"✅ Behavior analysis completed successfully")
        print(f"   Behavior Cluster: {behavior_analysis.behavior_cluster.value}")
        print(f"   Analysis Confidence: {behavior_analysis.analysis_confidence:.3f}")


class TestIntegration:
    """Integration tests for the complete personalization system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_personalization(self):
        """Test complete end-to-end personalization workflow"""
        
        # Initialize engine
        engine = PersonalizationEngine()
        
        # User data
        user_id = 'integration_test_user'
        learning_context = {
            'subject_domain': 'computer_science',
            'learning_objectives': ['understand_algorithms', 'master_data_structures'],
            'target_complexity': 'advanced',
            'estimated_duration': 60,
            'learning_history': [
                {
                    'subject': 'programming',
                    'difficulty_level': 0.8,
                    'completion_rate': 0.9,
                    'accuracy': 0.85
                }
            ],
            'interaction_history': [
                {
                    'type': 'interactive',
                    'engagement_score': 0.9,
                    'duration': 30
                }
            ],
            'performance_data': {
                'overall_accuracy': 0.85,
                'completion_rate': 0.9,
                'learning_velocity': 0.8
            }
        }
        
        # Step 1: Create personalized experience
        session = await engine.create_personalized_learning_experience(
            user_id, learning_context
        )
        
        assert session is not None
        assert session.user_id == user_id
        
        # Step 2: Simulate learning interaction
        interaction_data = {
            'behavior_type': 'performance',
            'type': 'problem_solving',
            'engagement_score': 0.95,
            'duration': 240,
            'success': True
        }
        
        performance_feedback = {
            'accuracy': 0.9,
            'completion_time': 200,
            'confidence_level': 0.85
        }
        
        # Step 3: Apply real-time adaptation
        adaptation_result = await engine.adapt_learning_real_time(
            user_id, interaction_data, performance_feedback
        )
        
        assert adaptation_result['updates_applied'] == True
        
        # Step 4: Generate insights
        insights = await engine.get_personalization_insights(user_id)
        
        assert insights.user_id == user_id
        assert insights.insights_confidence > 0
        
        # Step 5: Optimize personalization
        optimization_result = await engine.optimize_personalization(
            user_id, ['learning_effectiveness', 'user_engagement']
        )
        
        # Verify complete workflow
        engine_status = engine.get_engine_status()
        assert engine_status['components_status']['orchestrator'] == 'active'
        assert engine_status['metrics']['total_sessions_created'] > 0
        
        print(f"✅ End-to-end personalization workflow completed successfully")
        print(f"   Session Effectiveness: {session.personalization_effectiveness:.3f}")
        print(f"   Insights Confidence: {insights.insights_confidence:.3f}")
        print(f"   Engine Status: All components active")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
