"""
Comprehensive Test Suite for Streaming AI Services

Tests all components of the streaming AI system including:
- Live Tutoring Analysis Engine
- Real-Time Difficulty Adjustment
- Instant Feedback Generation
- Live Collaboration Intelligence
- Stream Quality Optimization
- Bandwidth-Adaptive Content
- Performance Monitoring
- WebSocket Handlers
- Orchestration System
"""

import pytest
import asyncio
import time
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

# Add the backend directory to the Python path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import streaming AI components
from quantum_intelligence.services.streaming_ai import (
    StreamingAIOrchestrator,
    LiveTutoringAnalysisEngine,
    RealTimeDifficultyAdjustment,
    InstantFeedbackEngine,
    LiveCollaborationIntelligence,
    StreamQualityOptimizer,
    BandwidthAdaptiveContent,
    RealTimePerformanceMonitor,
    StreamingWebSocketHandler,
    TutoringSessionHandler,
    FeedbackHandler,
    CollaborationHandler,
    
    # Data structures
    StreamQuality,
    TutoringMode,
    FeedbackType,
    CollaborationType,
    BandwidthCategory,
    NetworkCondition,
    StreamingMetrics,
    LiveTutoringSession,
    InstantFeedback,
    
    # Enums
    StreamingMode,
    PerformanceZone,
    EmotionalTone,
    GroupDynamicsState,
    MetricType,
    AlertType,
    
    # Utility functions
    create_websocket_message,
    create_streaming_event
)


class TestStreamingAIOrchestrator:
    """Test the main streaming AI orchestrator"""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance for testing"""
        return StreamingAIOrchestrator()
    
    @pytest.mark.asyncio
    async def test_start_streaming_session(self, orchestrator):
        """Test starting a streaming session"""
        session_config = {
            'mode': 'enhanced',
            'subject': 'mathematics',
            'learning_objectives': ['algebra', 'problem_solving']
        }
        participants = ['user1', 'user2', 'user3']
        
        result = await orchestrator.start_streaming_session(session_config, participants)
        
        assert result['status'] == 'success'
        assert 'session_id' in result
        assert result['participants'] == len(participants)
        assert result['streaming_mode'] == 'enhanced'
        assert 'websocket_endpoint' in result
    
    @pytest.mark.asyncio
    async def test_session_status_tracking(self, orchestrator):
        """Test session status tracking"""
        # Start session
        session_config = {'mode': 'basic'}
        participants = ['user1']
        
        result = await orchestrator.start_streaming_session(session_config, participants)
        session_id = result['session_id']
        
        # Check status
        status = await orchestrator.get_session_status(session_id)
        
        assert status['status'] == 'success'
        assert status['session_id'] == session_id
        assert status['participants'] == 1
        assert 'current_metrics' in status
        assert 'uptime_seconds' in status
    
    @pytest.mark.asyncio
    async def test_end_streaming_session(self, orchestrator):
        """Test ending a streaming session"""
        # Start session
        session_config = {'mode': 'basic'}
        participants = ['user1']
        
        start_result = await orchestrator.start_streaming_session(session_config, participants)
        session_id = start_result['session_id']
        
        # End session
        end_result = await orchestrator.end_streaming_session(session_id)
        
        assert end_result['status'] == 'success'
        assert end_result['session_ended'] == True
        assert 'total_duration_seconds' in end_result


class TestLiveTutoringAnalysisEngine:
    """Test the live tutoring analysis engine"""
    
    @pytest.fixture
    def tutoring_engine(self):
        """Create tutoring engine instance for testing"""
        return LiveTutoringAnalysisEngine()
    
    @pytest.mark.asyncio
    async def test_create_tutoring_session(self, tutoring_engine):
        """Test creating a live tutoring session"""
        session_id = "test_session_001"
        participants = ["student1", "student2", "tutor1"]
        mode = TutoringMode.SMALL_GROUP
        subject = "mathematics"
        objectives = ["solve_equations", "understand_concepts"]
        
        session = await tutoring_engine.create_live_tutoring_session(
            session_id, participants, mode, subject, objectives
        )
        
        assert isinstance(session, LiveTutoringSession)
        assert session.session_id == session_id
        assert session.participants == participants
        assert session.mode == mode
        assert session.subject == subject
        assert len(session.learning_objectives) == len(objectives)
    
    @pytest.mark.asyncio
    async def test_session_dynamics_analysis(self, tutoring_engine):
        """Test session dynamics analysis"""
        # Create session first
        session_id = "test_session_002"
        participants = ["student1", "tutor1"]
        
        await tutoring_engine.create_live_tutoring_session(
            session_id, participants, TutoringMode.ONE_ON_ONE, "math", ["algebra"]
        )
        
        # Analyze dynamics
        real_time_data = {
            'participant_actions': {
                'student1': {'engagement': 0.8, 'performance': 0.7},
                'tutor1': {'teaching_quality': 0.9, 'responsiveness': 0.8}
            },
            'session_metrics': {
                'duration_minutes': 15,
                'interaction_count': 25
            }
        }
        
        analysis = await tutoring_engine.analyze_session_dynamics(session_id, real_time_data)
        
        assert analysis['session_id'] == session_id
        assert 'session_health_score' in analysis
        assert 'participant_analytics' in analysis
        assert 'optimization_recommendations' in analysis


class TestRealTimeDifficultyAdjustment:
    """Test the real-time difficulty adjustment system"""
    
    @pytest.fixture
    def difficulty_engine(self):
        """Create difficulty adjustment engine for testing"""
        return RealTimeDifficultyAdjustment()
    
    @pytest.mark.asyncio
    async def test_initialize_user_model(self, difficulty_engine):
        """Test initializing user difficulty model"""
        user_id = "test_user_001"
        learning_profile = {
            'estimated_ability': 0.6,
            'learning_velocity': 0.7,
            'challenge_tolerance': 0.8
        }
        subject_domain = "mathematics"
        
        model = await difficulty_engine.initialize_user_model(
            user_id, learning_profile, subject_domain
        )
        
        assert model.user_id == user_id
        assert model.current_difficulty == learning_profile['estimated_ability']
        assert model.learning_velocity == learning_profile['learning_velocity']
        assert model.challenge_tolerance == learning_profile['challenge_tolerance']
    
    @pytest.mark.asyncio
    async def test_performance_analysis(self, difficulty_engine):
        """Test performance analysis and adjustment recommendation"""
        user_id = "test_user_002"
        
        # Initialize model first
        await difficulty_engine.initialize_user_model(
            user_id, {'estimated_ability': 0.5}, "math"
        )
        
        # Create performance event
        event = {
            'user_id': user_id,
            'session_id': 'session_001',
            'event_data': {
                'accuracy': 0.3,  # Low performance
                'response_time': 120,
                'engagement_level': 0.6,
                'cognitive_load': 0.9  # High cognitive load
            }
        }
        
        analysis = await difficulty_engine.analyze_performance_change(event)
        
        assert analysis['user_id'] == user_id
        assert 'current_performance' in analysis
        assert 'adjustment_recommendation' in analysis
        assert 'processing_latency_ms' in analysis
        
        # Should recommend difficulty reduction due to low performance
        recommendation = analysis['adjustment_recommendation']
        if recommendation['adjustment_needed']:
            assert recommendation['target_difficulty'] < 0.5


class TestInstantFeedbackEngine:
    """Test the instant feedback generation engine"""
    
    @pytest.fixture
    def feedback_engine(self):
        """Create feedback engine for testing"""
        return InstantFeedbackEngine()
    
    @pytest.mark.asyncio
    async def test_generate_feedback(self, feedback_engine):
        """Test generating instant feedback"""
        user_action = {
            'action_type': 'problem_solving',
            'success_level': 0.8,
            'completed': True,
            'confidence_level': 0.7
        }
        context = {
            'user_id': 'test_user_001',
            'session_id': 'session_001',
            'subject': 'mathematics',
            'difficulty_level': 0.6
        }
        
        feedback = await feedback_engine.generate_feedback(
            user_action, context, 'encouragement'
        )
        
        assert isinstance(feedback, InstantFeedback)
        assert feedback.user_id == context['user_id']
        assert feedback.session_id == context['session_id']
        assert feedback.feedback_type == FeedbackType.ENCOURAGEMENT
        assert len(feedback.content) > 0
        assert feedback.confidence_score > 0
    
    @pytest.mark.asyncio
    async def test_real_time_feedback_generation(self, feedback_engine):
        """Test real-time feedback generation from event"""
        event = {
            'user_id': 'test_user_002',
            'session_id': 'session_002',
            'event_data': {
                'action_type': 'question_asked',
                'success_level': 0.5,
                'help_requested': True,
                'emotional_state': 'confused'
            }
        }
        
        result = await feedback_engine.generate_real_time_feedback(event)
        
        assert result['status'] == 'success'
        assert 'feedback' in result
        assert 'processing_latency_ms' in result
        assert result['processing_latency_ms'] < 200  # Should be fast


class TestLiveCollaborationIntelligence:
    """Test the live collaboration intelligence system"""
    
    @pytest.fixture
    def collaboration_engine(self):
        """Create collaboration engine for testing"""
        return LiveCollaborationIntelligence()
    
    @pytest.mark.asyncio
    async def test_create_collaborative_session(self, collaboration_engine):
        """Test creating a collaborative session"""
        session_id = "collab_session_001"
        collaboration_type = CollaborationType.GROUP_DISCUSSION
        participants = ["student1", "student2", "student3", "moderator1"]
        objectives = ["discuss_concepts", "peer_learning"]
        subject = "science"
        
        result = await collaboration_engine.create_collaborative_session(
            session_id, collaboration_type, participants, objectives, subject
        )
        
        assert result['status'] == 'success'
        assert result['session_config']['session_id'] == session_id
        assert result['session_config']['collaboration_type'] == collaboration_type.value
        assert result['groups_created'] > 0
        assert 'predicted_effectiveness' in result
    
    @pytest.mark.asyncio
    async def test_process_collaboration_event(self, collaboration_engine):
        """Test processing collaboration events"""
        # Create session first
        session_id = "collab_session_002"
        await collaboration_engine.create_collaborative_session(
            session_id, CollaborationType.PEER_TUTORING, 
            ["student1", "student2"], ["peer_help"], "math"
        )
        
        # Process collaboration event
        event = {
            'session_id': session_id,
            'participant_id': 'student1',
            'event_type': 'knowledge_sharing',
            'content': {
                'shared_concept': 'quadratic_equations',
                'explanation_quality': 0.8
            }
        }
        
        result = await collaboration_engine.process_collaboration_event(event)
        
        assert result['status'] == 'success'
        assert result['event_processed'] == True
        assert 'collaboration_impact' in result
        assert 'adaptive_recommendations' in result


class TestStreamQualityOptimizer:
    """Test the stream quality optimizer"""
    
    @pytest.fixture
    def stream_optimizer(self):
        """Create stream optimizer for testing"""
        return StreamQualityOptimizer()
    
    @pytest.mark.asyncio
    async def test_optimize_stream_quality(self, stream_optimizer):
        """Test stream quality optimization"""
        user_id = "test_user_001"
        session_id = "session_001"
        
        network_conditions = NetworkCondition(
            bandwidth_kbps=300.0,
            latency_ms=80.0,
            packet_loss_rate=0.01,
            connection_stability=0.9,
            device_capabilities={'video': True, 'audio': True},
            optimal_quality=StreamQuality.MEDIUM,
            adaptive_recommendations=[]
        )
        
        content_context = {
            'content_type': 'rich_media',
            'interaction_level': 'high',
            'learning_criticality': 0.8
        }
        
        result = await stream_optimizer.optimize_stream_quality(
            user_id, session_id, network_conditions, content_context
        )
        
        assert result['status'] == 'success'
        assert 'optimization_result' in result
        optimization = result['optimization_result']
        assert 'recommended_quality' in optimization
        assert 'network_analysis' in optimization
        assert 'optimization_confidence' in optimization


class TestBandwidthAdaptiveContent:
    """Test the bandwidth-adaptive content system"""
    
    @pytest.fixture
    def adaptive_content(self):
        """Create adaptive content system for testing"""
        return BandwidthAdaptiveContent()
    
    @pytest.mark.asyncio
    async def test_adapt_content_for_bandwidth(self, adaptive_content):
        """Test content adaptation for bandwidth"""
        user_id = "test_user_001"
        session_id = "session_001"
        
        content_request = {
            'content_id': 'lesson_001',
            'content_type': 'mixed_media',
            'estimated_size_kb': 2000,
            'interactivity_level': 'high'
        }
        
        current_bandwidth = 100.0  # Low bandwidth
        
        network_conditions = NetworkCondition(
            bandwidth_kbps=current_bandwidth,
            latency_ms=120.0,
            packet_loss_rate=0.02,
            connection_stability=0.7,
            device_capabilities={'video': True, 'audio': True},
            optimal_quality=StreamQuality.LOW,
            adaptive_recommendations=[]
        )
        
        result = await adaptive_content.adapt_content_for_bandwidth(
            user_id, session_id, content_request, current_bandwidth, network_conditions
        )
        
        assert result['status'] == 'success'
        assert 'adaptation_result' in result
        adaptation = result['adaptation_result']
        assert adaptation['bandwidth_category'] == BandwidthCategory.LOW.value
        assert 'quality_impact' in adaptation
        assert 'bandwidth_savings' in adaptation


class TestRealTimePerformanceMonitor:
    """Test the real-time performance monitor"""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create performance monitor for testing"""
        return RealTimePerformanceMonitor("test_session_001")
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, performance_monitor):
        """Test starting and stopping monitoring"""
        # Start monitoring
        await performance_monitor.start_monitoring()
        assert performance_monitor.monitoring_active == True
        
        # Stop monitoring
        await performance_monitor.stop_monitoring()
        assert performance_monitor.monitoring_active == False
    
    @pytest.mark.asyncio
    async def test_update_metrics(self, performance_monitor):
        """Test updating performance metrics"""
        metrics = StreamingMetrics(
            latency_ms=85.0,
            throughput_kbps=450.0,
            packet_loss_rate=0.005,
            jitter_ms=15.0,
            cpu_usage=0.6,
            memory_usage=0.7,
            network_quality_score=0.9,
            user_engagement_score=0.8,
            content_delivery_success_rate=0.95,
            adaptive_adjustments_count=3
        )
        
        await performance_monitor.update_metrics(metrics)
        
        current_metrics = performance_monitor.get_current_metrics()
        assert current_metrics.latency_ms == 85.0
        assert current_metrics.throughput_kbps == 450.0
        assert current_metrics.network_quality_score == 0.9
    
    @pytest.mark.asyncio
    async def test_performance_report(self, performance_monitor):
        """Test generating performance report"""
        # Update some metrics first
        metrics = StreamingMetrics(
            latency_ms=100.0,
            throughput_kbps=300.0,
            packet_loss_rate=0.01,
            jitter_ms=20.0,
            cpu_usage=0.5,
            memory_usage=0.6,
            network_quality_score=0.8,
            user_engagement_score=0.7,
            content_delivery_success_rate=0.9,
            adaptive_adjustments_count=2
        )
        await performance_monitor.update_metrics(metrics)
        
        report = await performance_monitor.get_performance_report()
        
        assert 'session_id' in report
        assert 'current_metrics' in report
        assert 'metric_trends' in report
        assert 'optimization_recommendations' in report


class TestWebSocketHandlers:
    """Test WebSocket handlers"""
    
    @pytest.fixture
    def websocket_handler(self):
        """Create WebSocket handler for testing"""
        return StreamingWebSocketHandler()
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self, websocket_handler):
        """Test WebSocket connection handling"""
        user_id = "test_user_001"
        session_id = "session_001"
        connection_info = {
            'user_agent': 'test_browser',
            'ip_address': '127.0.0.1'
        }
        
        # Test connection
        result = await websocket_handler.connect(user_id, session_id, connection_info)
        assert result == True
        
        # Test disconnection
        connection_id = f"{user_id}_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # Find actual connection ID
        for conn_id in websocket_handler.active_connections:
            if user_id in conn_id and session_id in conn_id:
                connection_id = conn_id
                break
        
        disconnect_result = await websocket_handler.disconnect(connection_id)
        assert disconnect_result == True


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_create_websocket_message(self):
        """Test WebSocket message creation"""
        message = create_websocket_message(
            "user_action", "user1", "session1", {"action": "click"}
        )
        
        assert message.message_type == "user_action"
        assert message.sender_id == "user1"
        assert message.session_id == "session1"
        assert message.payload == {"action": "click"}
        assert message.timestamp is not None
    
    def test_create_streaming_event(self):
        """Test streaming event creation"""
        from quantum_intelligence.services.streaming_ai.data_structures import StreamingEventType
        
        event = create_streaming_event(
            StreamingEventType.USER_ACTION, "user1", "session1", 
            {"action": "submit_answer"}, priority=8
        )
        
        assert event.event_type == StreamingEventType.USER_ACTION
        assert event.user_id == "user1"
        assert event.session_id == "session1"
        assert event.event_data == {"action": "submit_answer"}
        assert event.priority == 8


# Integration tests
class TestStreamingAIIntegration:
    """Integration tests for the complete streaming AI system"""
    
    @pytest.mark.asyncio
    async def test_complete_tutoring_workflow(self):
        """Test complete tutoring workflow from start to finish"""
        # Initialize orchestrator
        orchestrator = StreamingAIOrchestrator()
        
        # Start streaming session
        session_config = {
            'mode': 'enhanced',
            'subject': 'mathematics',
            'learning_objectives': ['algebra', 'problem_solving']
        }
        participants = ['student1', 'tutor1']
        
        session_result = await orchestrator.start_streaming_session(session_config, participants)
        assert session_result['status'] == 'success'
        session_id = session_result['session_id']
        
        # Simulate user action that triggers feedback
        user_event = {
            'session_id': session_id,
            'user_id': 'student1',
            'event_type': 'user_action',
            'event_data': {
                'action_type': 'problem_attempt',
                'success_level': 0.6,
                'time_taken': 90,
                'help_requested': False
            }
        }
        
        # Add event to orchestrator
        await orchestrator.add_streaming_event(user_event)
        
        # Brief wait for processing
        await asyncio.sleep(0.1)
        
        # Check session status
        status = await orchestrator.get_session_status(session_id)
        assert status['status'] == 'success'
        assert status['participants'] == len(participants)
        
        # End session
        end_result = await orchestrator.end_streaming_session(session_id)
        assert end_result['status'] == 'success'
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self):
        """Test performance monitoring integration"""
        # Create monitor
        monitor = RealTimePerformanceMonitor("integration_test_session")
        
        # Start monitoring
        await monitor.start_monitoring()
        
        # Update metrics multiple times
        for i in range(5):
            metrics = StreamingMetrics(
                latency_ms=50.0 + i * 10,
                throughput_kbps=400.0 - i * 20,
                packet_loss_rate=0.001 * i,
                jitter_ms=10.0 + i * 2,
                cpu_usage=0.4 + i * 0.1,
                memory_usage=0.5 + i * 0.05,
                network_quality_score=0.9 - i * 0.05,
                user_engagement_score=0.8,
                content_delivery_success_rate=0.95,
                adaptive_adjustments_count=i
            )
            await monitor.update_metrics(metrics)
            await asyncio.sleep(0.01)  # Small delay
        
        # Generate report
        report = await monitor.get_performance_report()
        assert 'metric_trends' in report
        assert 'optimization_recommendations' in report
        
        # Stop monitoring
        await monitor.stop_monitoring()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
