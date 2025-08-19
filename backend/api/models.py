"""
API Models for MasterX Quantum Intelligence Platform

Comprehensive Pydantic models for request/response structures that integrate
with all quantum intelligence services and provide type safety and validation.

📋 MODEL CATEGORIES:
- Authentication and user management models
- Chat and conversation models
- Learning and progress tracking models
- Personalization and user profiling models
- Analytics and prediction models
- Content generation and assessment models
- Real-time streaming and WebSocket models

Author: MasterX AI Team - API Integration Division
Version: 1.0 - Phase 12 Enhanced Backend APIs Integration
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Literal
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid

# ============================================================================
# BASE MODELS
# ============================================================================

class BaseResponse(BaseModel):
    """Base response model"""
    success: bool = True
    message: str = "Operation completed successfully"
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class ErrorResponse(BaseResponse):
    """Error response model"""
    success: bool = False
    error_code: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None

# ============================================================================
# AUTHENTICATION MODELS
# ============================================================================

class UserRole(str, Enum):
    """User role enumeration"""
    STUDENT = "student"
    TEACHER = "teacher"
    ADMIN = "admin"
    GUEST = "guest"

class LoginRequest(BaseModel):
    """Login request model"""
    email: str = Field(..., description="User email address")
    password: str = Field(..., description="User password")
    remember_me: bool = Field(default=False, description="Remember login session")

class LoginResponse(BaseResponse):
    """Login response model"""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    user_info: Dict[str, Any] = Field(..., description="User information")

class UserProfile(BaseModel):
    """User profile model"""
    user_id: str = Field(..., description="Unique user identifier")
    email: str = Field(..., description="User email address")
    name: str = Field(..., description="User full name")
    role: UserRole = Field(..., description="User role")
    created_at: datetime = Field(..., description="Account creation timestamp")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")

# ============================================================================
# CHAT MODELS
# ============================================================================

class ChatMessageType(str, Enum):
    """Chat message type enumeration"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"
    CODE = "code"

class ChatMessage(BaseModel):
    """Chat message model"""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = Field(..., description="Chat session identifier")
    user_id: str = Field(..., description="User identifier")
    message_type: ChatMessageType = Field(default=ChatMessageType.TEXT)
    content: str = Field(..., description="Message content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: datetime = Field(default_factory=datetime.now)

class ChatRequest(BaseModel):
    """Chat request model"""
    session_id: Optional[str] = Field(None, description="Existing session ID or None for new session")
    message: str = Field(..., description="User message")
    message_type: ChatMessageType = Field(default=ChatMessageType.TEXT)
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    stream: bool = Field(default=False, description="Enable streaming response")
    task_type: Optional[str] = Field(None, description="Task type for intelligent model selection")
    provider: Optional[str] = Field(None, description="Preferred LLM provider")

class ChatResponse(BaseResponse):
    """Chat response model"""
    session_id: str = Field(..., description="Chat session identifier")
    message_id: str = Field(..., description="Response message identifier")
    response: str = Field(..., description="AI response")
    suggestions: List[str] = Field(default_factory=list, description="Follow-up suggestions")
    learning_insights: Optional[Dict[str, Any]] = Field(None, description="Learning insights")
    personalization_data: Optional[Dict[str, Any]] = Field(None, description="Personalization updates")

# ============================================================================
# LEARNING MODELS
# ============================================================================

class LearningGoalStatus(str, Enum):
    """Learning goal status enumeration"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PAUSED = "paused"

class LearningGoal(BaseModel):
    """Learning goal model"""
    goal_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = Field(..., description="User identifier")
    title: str = Field(..., description="Goal title")
    description: str = Field(..., description="Goal description")
    subject: str = Field(..., description="Subject area")
    target_skills: List[str] = Field(..., description="Target skills to develop")
    difficulty_level: float = Field(..., ge=0.0, le=1.0, description="Difficulty level (0-1)")
    estimated_duration_hours: int = Field(..., description="Estimated completion time")
    status: LearningGoalStatus = Field(default=LearningGoalStatus.NOT_STARTED)
    progress_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    created_at: datetime = Field(default_factory=datetime.now)
    target_completion_date: Optional[datetime] = Field(None)

class LearningSession(BaseModel):
    """Learning session model"""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = Field(..., description="User identifier")
    goal_id: Optional[str] = Field(None, description="Associated learning goal")
    subject: str = Field(..., description="Subject area")
    duration_minutes: int = Field(..., description="Session duration")
    activities: List[Dict[str, Any]] = Field(..., description="Learning activities")
    performance_metrics: Dict[str, float] = Field(..., description="Performance metrics")
    engagement_score: float = Field(..., ge=0.0, le=1.0, description="Engagement score")
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = Field(None)

class ProgressRequest(BaseModel):
    """Learning progress request model"""
    user_id: str = Field(..., description="User identifier")
    time_range: Literal["day", "week", "month", "quarter", "year"] = Field(default="week")
    subject_filter: Optional[str] = Field(None, description="Filter by subject")
    goal_filter: Optional[str] = Field(None, description="Filter by goal")

class ProgressResponse(BaseResponse):
    """Learning progress response model"""
    user_id: str = Field(..., description="User identifier")
    overall_progress: float = Field(..., ge=0.0, le=100.0, description="Overall progress percentage")
    goals_progress: List[Dict[str, Any]] = Field(..., description="Progress by goals")
    subject_progress: Dict[str, float] = Field(..., description="Progress by subjects")
    recent_sessions: List[LearningSession] = Field(..., description="Recent learning sessions")
    achievements: List[Dict[str, Any]] = Field(..., description="Recent achievements")
    recommendations: List[str] = Field(..., description="Learning recommendations")

# ============================================================================
# PERSONALIZATION MODELS
# ============================================================================

class LearningStyle(str, Enum):
    """Learning style enumeration"""
    VISUAL = "visual"
    AUDITORY = "auditory"
    KINESTHETIC = "kinesthetic"
    READING_WRITING = "reading_writing"

class PersonalizationRequest(BaseModel):
    """Personalization request model"""
    user_id: str = Field(..., description="User identifier")
    learning_preferences: Optional[Dict[str, Any]] = Field(None)
    performance_data: Optional[List[Dict[str, Any]]] = Field(None)
    behavioral_data: Optional[List[Dict[str, Any]]] = Field(None)

class LearningDNAProfile(BaseModel):
    """Learning DNA profile model"""
    user_id: str = Field(..., description="User identifier")
    learning_style: LearningStyle = Field(..., description="Primary learning style")
    cognitive_patterns: List[str] = Field(..., description="Cognitive patterns")
    personality_traits: Dict[str, float] = Field(..., description="Personality trait scores")
    preferred_pace: str = Field(..., description="Preferred learning pace")
    motivation_style: str = Field(..., description="Motivation style")
    optimal_difficulty_level: float = Field(..., ge=0.0, le=1.0)
    processing_speed: float = Field(..., ge=0.0, le=1.0)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    profile_completeness: float = Field(..., ge=0.0, le=1.0)
    last_updated: datetime = Field(default_factory=datetime.now)

class PersonalizationResponse(BaseResponse):
    """Personalization response model"""
    user_id: str = Field(..., description="User identifier")
    learning_dna: LearningDNAProfile = Field(..., description="Learning DNA profile")
    personalized_content: List[Dict[str, Any]] = Field(..., description="Personalized content recommendations")
    adaptive_strategies: List[str] = Field(..., description="Adaptive learning strategies")
    optimization_suggestions: List[str] = Field(..., description="Learning optimization suggestions")

# ============================================================================
# ANALYTICS MODELS
# ============================================================================

class PredictionType(str, Enum):
    """Prediction type enumeration"""
    LEARNING_OUTCOME = "learning_outcome"
    PERFORMANCE_TRAJECTORY = "performance_trajectory"
    SKILL_MASTERY = "skill_mastery"
    ENGAGEMENT_FORECAST = "engagement_forecast"
    RISK_ASSESSMENT = "risk_assessment"

class AnalyticsRequest(BaseModel):
    """Analytics request model"""
    user_id: str = Field(..., description="User identifier")
    prediction_type: PredictionType = Field(..., description="Type of prediction")
    time_horizon: Literal["immediate", "short_term", "medium_term", "long_term"] = Field(default="medium_term")
    include_interventions: bool = Field(default=True, description="Include intervention recommendations")

class PredictionResult(BaseModel):
    """Prediction result model"""
    prediction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prediction_type: PredictionType = Field(..., description="Type of prediction")
    predicted_outcome: Dict[str, Any] = Field(..., description="Predicted outcome data")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    risk_level: str = Field(..., description="Risk level assessment")
    contributing_factors: List[str] = Field(..., description="Key contributing factors")
    recommendations: List[str] = Field(..., description="Actionable recommendations")
    created_at: datetime = Field(default_factory=datetime.now)

class AnalyticsResponse(BaseResponse):
    """Analytics response model"""
    user_id: str = Field(..., description="User identifier")
    predictions: List[PredictionResult] = Field(..., description="Prediction results")
    learning_analytics: Dict[str, Any] = Field(..., description="Learning analytics dashboard")
    intervention_recommendations: List[Dict[str, Any]] = Field(..., description="Intervention recommendations")
    performance_insights: Dict[str, Any] = Field(..., description="Performance insights")

# ============================================================================
# CONTENT MODELS
# ============================================================================

class ContentType(str, Enum):
    """Content type enumeration"""
    LESSON = "lesson"
    EXERCISE = "exercise"
    QUIZ = "quiz"
    PROJECT = "project"
    RESOURCE = "resource"

class ContentRequest(BaseModel):
    """Content generation request model"""
    user_id: str = Field(..., description="User identifier")
    subject: str = Field(..., description="Subject area")
    topic: str = Field(..., description="Specific topic")
    content_type: ContentType = Field(..., description="Type of content to generate")
    difficulty_level: float = Field(..., ge=0.0, le=1.0, description="Difficulty level")
    duration_minutes: int = Field(..., description="Target duration")
    learning_objectives: List[str] = Field(..., description="Learning objectives")
    personalization_context: Optional[Dict[str, Any]] = Field(None)

class GeneratedContent(BaseModel):
    """Generated content model"""
    content_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content_type: ContentType = Field(..., description="Type of content")
    title: str = Field(..., description="Content title")
    description: str = Field(..., description="Content description")
    content_data: Dict[str, Any] = Field(..., description="Actual content data")
    metadata: Dict[str, Any] = Field(..., description="Content metadata")
    difficulty_level: float = Field(..., ge=0.0, le=1.0)
    estimated_duration_minutes: int = Field(..., description="Estimated completion time")
    learning_objectives: List[str] = Field(..., description="Learning objectives")
    created_at: datetime = Field(default_factory=datetime.now)

class ContentResponse(BaseResponse):
    """Content generation response model"""
    user_id: str = Field(..., description="User identifier")
    generated_content: GeneratedContent = Field(..., description="Generated content")
    personalization_notes: List[str] = Field(..., description="Personalization notes")
    usage_recommendations: List[str] = Field(..., description="Usage recommendations")

# ============================================================================
# STREAMING MODELS
# ============================================================================

class StreamingEventType(str, Enum):
    """Streaming event type enumeration"""
    CHAT_MESSAGE = "chat_message"
    LEARNING_UPDATE = "learning_update"
    PROGRESS_UPDATE = "progress_update"
    NOTIFICATION = "notification"
    SYSTEM_EVENT = "system_event"

class StreamingEvent(BaseModel):
    """Streaming event model"""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: StreamingEventType = Field(..., description="Type of streaming event")
    user_id: str = Field(..., description="Target user identifier")
    data: Dict[str, Any] = Field(..., description="Event data")
    timestamp: datetime = Field(default_factory=datetime.now)

class WebSocketMessage(BaseModel):
    """WebSocket message model"""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message_type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(..., description="Message data")
    timestamp: datetime = Field(default_factory=datetime.now)

# ============================================================================
# ASSESSMENT MODELS
# ============================================================================

class AssessmentType(str, Enum):
    """Assessment type enumeration"""
    QUIZ = "quiz"
    TEST = "test"
    PROJECT = "project"
    PRACTICAL = "practical"
    ORAL = "oral"

class AssessmentRequest(BaseModel):
    """Assessment request model"""
    user_id: str = Field(..., description="User identifier")
    subject: str = Field(..., description="Subject area")
    topics: List[str] = Field(..., description="Topics to assess")
    assessment_type: AssessmentType = Field(..., description="Type of assessment")
    difficulty_level: float = Field(..., ge=0.0, le=1.0)
    duration_minutes: int = Field(..., description="Assessment duration")
    question_count: int = Field(..., description="Number of questions")

class AssessmentResult(BaseModel):
    """Assessment result model"""
    assessment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = Field(..., description="User identifier")
    score: float = Field(..., ge=0.0, le=100.0, description="Assessment score")
    max_score: float = Field(..., description="Maximum possible score")
    percentage: float = Field(..., ge=0.0, le=100.0, description="Score percentage")
    time_taken_minutes: int = Field(..., description="Time taken to complete")
    question_results: List[Dict[str, Any]] = Field(..., description="Individual question results")
    strengths: List[str] = Field(..., description="Identified strengths")
    weaknesses: List[str] = Field(..., description="Areas for improvement")
    recommendations: List[str] = Field(..., description="Learning recommendations")
    completed_at: datetime = Field(default_factory=datetime.now)

class AssessmentResponse(BaseResponse):
    """Assessment response model"""
    assessment: Dict[str, Any] = Field(..., description="Generated assessment")
    result: Optional[AssessmentResult] = Field(None, description="Assessment result if completed")
    next_steps: List[str] = Field(..., description="Recommended next steps")
