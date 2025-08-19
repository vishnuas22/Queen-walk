"""
Real-Time Streaming AI Services

Advanced real-time streaming AI system for educational platforms, extracted from the quantum intelligence engine.
This comprehensive suite provides sophisticated real-time processing capabilities including live tutoring analysis,
dynamic difficulty adjustment, instant feedback generation, collaboration intelligence, stream optimization,
and bandwidth-adaptive content delivery.

This package contains:
- Live Tutoring Analysis: Real-time student behavior monitoring and session optimization
- Dynamic Difficulty Adjustment: Adaptive content complexity based on real-time performance
- Instant Feedback Generation: Contextual, personalized feedback with sub-100ms latency
- Collaboration Intelligence: Multi-participant learning session orchestration
- Stream Quality Optimization: Network-aware content delivery optimization
- Bandwidth-Adaptive Content: Dynamic content adaptation for varying network conditions

Example usage:
    from quantum_intelligence.services.streaming_ai import (
        StreamingAIOrchestrator, LiveTutoringAnalysisEngine
    )
    
    # Initialize streaming AI system
    orchestrator = StreamingAIOrchestrator()
    
    # Start live tutoring session
    session = await orchestrator.start_live_tutoring_session(
        participants, subject, learning_objectives
    )
"""

# Version information
__version__ = "1.0.0"
__author__ = "Quantum Intelligence Research Team"

# Import core data structures and enums
from .data_structures import (
    StreamQuality,
    CollaborationRole,
    FeedbackType,
    TutoringMode,
    StreamingMetrics,
    LiveTutoringSession,
    InstantFeedback,
    CollaborationEvent,
    NetworkCondition,
    AlertSeverity,
    PerformanceAlert,
    RealTimeAnalytics,
    StreamingEventType,
    StreamingEvent,
    WebSocketMessage,
    DifficultyAdjustment,
    DifficultyAdjustmentReason,
    ContentAdaptation,
    SessionState,
    create_websocket_message,
    create_streaming_event
)

# Import main orchestration classes
from .orchestrator import (
    StreamingAIOrchestrator,
    StreamingSession,
    RealTimeEvent,
    StreamingMode,
    SessionStatus
)

# Import core streaming AI engines
from .live_tutoring import LiveTutoringAnalysisEngine, ParticipantRole, SessionHealthStatus
from .difficulty_adjustment import RealTimeDifficultyAdjustment, PerformanceZone, AdjustmentUrgency
from .instant_feedback import InstantFeedbackEngine, FeedbackUrgency, EmotionalTone
from .collaboration_intelligence import LiveCollaborationIntelligence, CollaborationType, GroupDynamicsState
from .stream_optimization import StreamQualityOptimizer, ContentType, OptimizationStrategy
from .adaptive_content import BandwidthAdaptiveContent, BandwidthCategory, AdaptationStrategy

# Import WebSocket and real-time communication
from .websocket_handlers import (
    StreamingWebSocketHandler,
    TutoringSessionHandler,
    FeedbackHandler,
    CollaborationHandler,
    WebSocketMessageType
)

# Import performance monitoring
from .performance_monitoring import (
    RealTimePerformanceMonitor,
    MetricType,
    AlertType
)

# Define what gets imported with "from streaming_ai import *"
__all__ = [
    # Core data structures
    "StreamQuality",
    "CollaborationRole",
    "FeedbackType",
    "TutoringMode",
    "StreamingMetrics",
    "LiveTutoringSession",
    "InstantFeedback",
    "CollaborationEvent",
    "NetworkCondition",
    "AlertSeverity",
    "PerformanceAlert",
    "RealTimeAnalytics",
    "StreamingEventType",
    "StreamingEvent",
    "WebSocketMessage",
    "DifficultyAdjustment",
    "DifficultyAdjustmentReason",
    "ContentAdaptation",
    "SessionState",

    # Main orchestration
    "StreamingAIOrchestrator",
    "StreamingSession",
    "RealTimeEvent",
    "StreamingMode",
    "SessionStatus",

    # Core engines
    "LiveTutoringAnalysisEngine",
    "RealTimeDifficultyAdjustment",
    "InstantFeedbackEngine",
    "LiveCollaborationIntelligence",
    "StreamQualityOptimizer",
    "BandwidthAdaptiveContent",

    # Engine-specific enums
    "ParticipantRole",
    "SessionHealthStatus",
    "PerformanceZone",
    "AdjustmentUrgency",
    "FeedbackUrgency",
    "EmotionalTone",
    "CollaborationType",
    "GroupDynamicsState",
    "ContentType",
    "OptimizationStrategy",
    "BandwidthCategory",
    "AdaptationStrategy",

    # WebSocket handlers
    "StreamingWebSocketHandler",
    "TutoringSessionHandler",
    "FeedbackHandler",
    "CollaborationHandler",
    "WebSocketMessageType",

    # Performance monitoring
    "RealTimePerformanceMonitor",
    "MetricType",
    "AlertType",

    # Utility functions
    "create_websocket_message",
    "create_streaming_event",
]


# Convenience functions for quick setup
def create_streaming_orchestrator(config=None):
    """
    Create streaming AI orchestrator with default configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        StreamingAIOrchestrator: Configured streaming AI orchestrator
    """
    return StreamingAIOrchestrator(config=config)


def create_live_tutoring_engine(cache_service=None):
    """
    Create live tutoring analysis engine with default configuration.
    
    Args:
        cache_service: Optional cache service for performance optimization
        
    Returns:
        LiveTutoringAnalysisEngine: Configured live tutoring engine
    """
    return LiveTutoringAnalysisEngine(cache_service)


def create_instant_feedback_engine(cache_service=None):
    """
    Create instant feedback engine with default configuration.
    
    Args:
        cache_service: Optional cache service for performance optimization
        
    Returns:
        InstantFeedbackEngine: Configured instant feedback engine
    """
    return InstantFeedbackEngine(cache_service)


async def start_streaming_session(session_config, participants):
    """
    Start a streaming AI session with minimal setup.
    
    Args:
        session_config: Session configuration
        participants: List of participant IDs
        
    Returns:
        Dict: Streaming session result
    """
    orchestrator = create_streaming_orchestrator()
    return await orchestrator.start_streaming_session(session_config, participants)


async def generate_instant_feedback(user_action, context, feedback_type='suggestion'):
    """
    Generate instant feedback with default settings.
    
    Args:
        user_action: User action data
        context: Learning context
        feedback_type: Type of feedback to generate
        
    Returns:
        InstantFeedback: Generated feedback
    """
    engine = create_instant_feedback_engine()
    return await engine.generate_feedback(user_action, context, feedback_type)


# Add convenience functions to exports
__all__.extend([
    "create_streaming_orchestrator",
    "create_live_tutoring_engine", 
    "create_instant_feedback_engine",
    "start_streaming_session",
    "generate_instant_feedback"
])


# Package metadata
__package_info__ = {
    'name': 'quantum_intelligence.services.streaming_ai',
    'version': __version__,
    'description': 'Real-time streaming AI services for educational platforms',
    'author': __author__,
    'components': [
        'live_tutoring',
        'difficulty_adjustment',
        'instant_feedback',
        'collaboration_intelligence',
        'stream_optimization',
        'adaptive_content'
    ],
    'capabilities': [
        'Real-time student behavior monitoring',
        'Dynamic difficulty adjustment (< 200ms)',
        'Instant feedback generation (< 100ms)',
        'Multi-participant collaboration orchestration',
        'Adaptive stream quality optimization',
        'Bandwidth-aware content delivery',
        'WebSocket real-time communication',
        'Low-latency performance optimization'
    ]
}


# Logging configuration
import logging

# Create logger for the streaming AI package
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add console handler if none exists
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

logger.info(f"Quantum Intelligence Streaming AI Services v{__version__} initialized")
logger.info(f"Available components: {', '.join(__package_info__['components'])}")


# Health check function
def health_check():
    """
    Perform a basic health check of the streaming AI package.
    
    Returns:
        Dict: Health check results
    """
    try:
        # Test basic imports and initialization
        orchestrator = create_streaming_orchestrator()
        tutoring_engine = create_live_tutoring_engine()
        feedback_engine = create_instant_feedback_engine()
        
        return {
            'status': 'healthy',
            'version': __version__,
            'components_loaded': len(__package_info__['components']),
            'orchestrator_initialized': orchestrator is not None,
            'tutoring_engine_initialized': tutoring_engine is not None,
            'feedback_engine_initialized': feedback_engine is not None,
            'capabilities': __package_info__['capabilities']
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'version': __version__
        }


# Export health check
__all__.append('health_check')
