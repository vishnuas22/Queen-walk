"""
Real-Time Streaming AI Data Structures

Core data structures and enums for the streaming AI system, extracted from the quantum intelligence engine.
Provides comprehensive data models for live tutoring, instant feedback, collaboration, and streaming optimization.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime


class StreamQuality(Enum):
    """Stream quality levels for adaptive content delivery"""
    ULTRA_LOW = "ultra_low"      # 64 kbps, text only
    LOW = "low"                  # 128 kbps, basic content
    MEDIUM = "medium"            # 256 kbps, standard content
    HIGH = "high"                # 512 kbps, rich content
    ULTRA_HIGH = "ultra_high"    # 1+ Mbps, premium content
    ADAPTIVE = "adaptive"        # Dynamic based on conditions


class CollaborationRole(Enum):
    """Roles in collaborative learning sessions"""
    LEARNER = "learner"
    PEER_TUTOR = "peer_tutor"
    MODERATOR = "moderator"
    EXPERT_ASSISTANT = "expert_assistant"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"
    OBSERVER = "observer"


class FeedbackType(Enum):
    """Types of instant feedback"""
    CORRECTNESS = "correctness"
    SUGGESTION = "suggestion"
    ENCOURAGEMENT = "encouragement"
    CLARIFICATION = "clarification"
    CHALLENGE = "challenge"
    REDIRECT = "redirect"
    METACOGNITIVE = "metacognitive"


class TutoringMode(Enum):
    """Live tutoring session modes"""
    ONE_ON_ONE = "one_on_one"
    SMALL_GROUP = "small_group"
    PEER_TO_PEER = "peer_to_peer"
    AI_FACILITATED = "ai_facilitated"
    HYBRID = "hybrid"


class StreamingEventType(Enum):
    """Types of real-time streaming events"""
    USER_ACTION = "user_action"
    FEEDBACK_GENERATED = "feedback_generated"
    DIFFICULTY_ADJUSTED = "difficulty_adjusted"
    COLLABORATION_EVENT = "collaboration_event"
    STREAM_QUALITY_CHANGED = "stream_quality_changed"
    SESSION_STATE_CHANGED = "session_state_changed"
    PERFORMANCE_ALERT = "performance_alert"
    NETWORK_CONDITION_CHANGED = "network_condition_changed"


class DifficultyAdjustmentReason(Enum):
    """Reasons for difficulty adjustments"""
    PERFORMANCE_TOO_LOW = "performance_too_low"
    PERFORMANCE_TOO_HIGH = "performance_too_high"
    ENGAGEMENT_DROPPING = "engagement_dropping"
    COGNITIVE_OVERLOAD = "cognitive_overload"
    MASTERY_ACHIEVED = "mastery_achieved"
    TIME_PRESSURE = "time_pressure"
    PEER_COMPARISON = "peer_comparison"


@dataclass
class StreamingMetrics:
    """Real-time streaming performance metrics"""
    latency_ms: float
    throughput_kbps: float
    packet_loss_rate: float
    jitter_ms: float
    cpu_usage: float
    memory_usage: float
    network_quality_score: float
    user_engagement_score: float
    content_delivery_success_rate: float
    adaptive_adjustments_count: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LiveTutoringSession:
    """Live tutoring session data structure"""
    session_id: str
    participants: List[str]
    mode: TutoringMode
    subject: str
    current_topic: str
    start_time: datetime
    estimated_duration: int  # minutes
    learning_objectives: List[str]
    difficulty_level: float
    collaboration_metrics: Dict[str, Any]
    real_time_analytics: Dict[str, Any]
    adaptive_adjustments: List[Dict[str, Any]]
    stream_quality: StreamQuality
    bandwidth_allocation: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class InstantFeedback:
    """Instant feedback data structure"""
    feedback_id: str
    user_id: str
    session_id: str
    feedback_type: FeedbackType
    content: str
    confidence_score: float
    relevance_score: float
    timing_appropriateness: float
    learning_impact_prediction: float
    suggested_actions: List[str]
    emotional_tone: str
    personalization_factors: Dict[str, Any]
    delivery_timestamp: datetime
    response_required: bool = False
    expiry_timestamp: Optional[datetime] = None


@dataclass
class CollaborationEvent:
    """Real-time collaboration event"""
    event_id: str
    session_id: str
    participant_id: str
    event_type: str
    content: Dict[str, Any]
    collaboration_impact: float
    peer_learning_opportunity: bool
    knowledge_sharing_quality: float
    social_learning_metrics: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class NetworkCondition:
    """Current network conditions for adaptive streaming"""
    bandwidth_kbps: float
    latency_ms: float
    packet_loss_rate: float
    connection_stability: float
    device_capabilities: Dict[str, Any]
    optimal_quality: StreamQuality
    adaptive_recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DifficultyAdjustment:
    """Difficulty adjustment event data structure"""
    adjustment_id: str
    user_id: str
    session_id: str
    previous_difficulty: float
    new_difficulty: float
    adjustment_magnitude: float
    reason: DifficultyAdjustmentReason
    confidence_score: float
    expected_impact: Dict[str, float]
    adjustment_timestamp: datetime
    performance_context: Dict[str, Any]
    learning_context: Dict[str, Any]


@dataclass
class RealTimeAnalytics:
    """Real-time analytics data structure"""
    analytics_id: str
    user_id: str
    session_id: str
    engagement_score: float
    performance_score: float
    attention_score: float
    collaboration_score: float
    learning_velocity: float
    cognitive_load: float
    emotional_state: str
    prediction_confidence: float
    analytics_timestamp: datetime
    trend_indicators: Dict[str, float]
    intervention_recommendations: List[str]


@dataclass
class StreamingEvent:
    """Generic streaming event data structure"""
    event_id: str
    event_type: StreamingEventType
    user_id: str
    session_id: str
    event_data: Dict[str, Any]
    priority: int  # 1-10, 10 being highest priority
    requires_immediate_action: bool
    event_timestamp: datetime
    processing_deadline: Optional[datetime] = None
    related_events: List[str] = field(default_factory=list)


@dataclass
class WebSocketMessage:
    """WebSocket message data structure"""
    message_id: str
    message_type: str
    sender_id: str
    recipient_id: Optional[str]  # None for broadcast
    session_id: str
    payload: Dict[str, Any]
    priority: int
    requires_acknowledgment: bool
    timestamp: datetime
    expiry_timestamp: Optional[datetime] = None


@dataclass
class PerformanceAlert:
    """Performance alert data structure"""
    alert_id: str
    alert_type: str
    severity: AlertSeverity
    user_id: str = ""
    session_id: str = ""
    metric_type: str = ""
    current_value: float = 0.0
    threshold_value: float = 0.0
    message: str = ""
    first_detected: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    active: bool = True
    related_metrics: Dict[str, Any] = field(default_factory=dict)
    # Legacy fields for compatibility
    metric_name: str = ""
    alert_message: str = ""
    suggested_actions: List[str] = field(default_factory=list)
    alert_timestamp: datetime = field(default_factory=datetime.now)
    auto_resolve: bool = False
    resolution_timestamp: Optional[datetime] = None


@dataclass
class ContentAdaptation:
    """Content adaptation data structure"""
    adaptation_id: str
    user_id: str
    session_id: str
    original_content_id: str
    adapted_content_id: str
    adaptation_type: str
    adaptation_reason: str
    quality_reduction: float  # 0-1, 0 being no reduction
    bandwidth_saved: float  # kbps
    learning_impact_score: float
    adaptation_timestamp: datetime = field(default_factory=datetime.now)
    user_satisfaction_prediction: float = 0.8


@dataclass
class SessionState:
    """Current session state data structure"""
    session_id: str
    state: str  # 'initializing', 'active', 'paused', 'ending', 'ended'
    participants: List[str]
    active_participants: List[str]
    current_activity: str
    session_metrics: StreamingMetrics
    network_conditions: Dict[str, NetworkCondition]
    recent_events: List[StreamingEvent]
    performance_alerts: List[PerformanceAlert]
    last_updated: datetime


# Utility functions for data structure operations
def create_streaming_event(event_type: StreamingEventType, 
                         user_id: str, 
                         session_id: str, 
                         event_data: Dict[str, Any],
                         priority: int = 5) -> StreamingEvent:
    """Create a streaming event with default values"""
    return StreamingEvent(
        event_id=f"{event_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
        event_type=event_type,
        user_id=user_id,
        session_id=session_id,
        event_data=event_data,
        priority=priority,
        requires_immediate_action=priority >= 8,
        event_timestamp=datetime.now()
    )


def create_websocket_message(message_type: str,
                           sender_id: str,
                           session_id: str,
                           payload: Dict[str, Any],
                           recipient_id: Optional[str] = None,
                           priority: int = 5) -> WebSocketMessage:
    """Create a WebSocket message with default values"""
    return WebSocketMessage(
        message_id=f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
        message_type=message_type,
        sender_id=sender_id,
        recipient_id=recipient_id,
        session_id=session_id,
        payload=payload,
        priority=priority,
        requires_acknowledgment=priority >= 7,
        timestamp=datetime.now()
    )


def create_performance_alert(alert_type: str,
                           severity: str,
                           user_id: str,
                           session_id: str,
                           metric_name: str,
                           current_value: float,
                           threshold_value: float,
                           alert_message: str) -> PerformanceAlert:
    """Create a performance alert with default values"""
    return PerformanceAlert(
        alert_id=f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
        alert_type=alert_type,
        severity=severity,
        user_id=user_id,
        session_id=session_id,
        metric_name=metric_name,
        current_value=current_value,
        threshold_value=threshold_value,
        alert_message=alert_message,
        suggested_actions=[],
        alert_timestamp=datetime.now()
    )
