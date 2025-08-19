"""
Core data structures for Quantum Intelligence Engine
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List
from datetime import datetime, timezone

from .enums import QuantumLearningMode, QuantumState, IntelligenceLevel


@dataclass
class QuantumLearningContext:
    """Comprehensive context for quantum learning"""
    user_id: str
    session_id: str
    current_quantum_state: QuantumState
    learning_dna: 'LearningDNA'
    mood_adaptation: 'MoodBasedAdaptation'
    active_mode: QuantumLearningMode
    intelligence_level: IntelligenceLevel
    knowledge_graph_state: Dict[str, Any]
    analytics_insights: Dict[str, Any]
    gamification_state: Dict[str, Any]
    metacognitive_progress: Dict[str, Any]
    temporal_context: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    adaptive_parameters: 'AdaptiveContentParameters'
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "current_quantum_state": self.current_quantum_state.value,
            "learning_dna": self.learning_dna.to_dict(),
            "mood_adaptation": self.mood_adaptation.to_dict(),
            "active_mode": self.active_mode.value,
            "intelligence_level": self.intelligence_level.value,
            "knowledge_graph_state": self.knowledge_graph_state,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuantumLearningContext':
        """Create from dictionary"""
        timestamp = datetime.fromisoformat(data.get("timestamp", datetime.now(timezone.utc).isoformat()))
        return cls(
            user_id=data["user_id"],
            session_id=data["session_id"],
            message=data["message"],
            learning_dna=data.get("learning_dna", {}),
            context=data.get("context", {}),
            timestamp=timestamp,
            difficulty_preference=data.get("difficulty_preference", 0.5),
            learning_velocity=data.get("learning_velocity", 0.6),
            attention_span_minutes=data.get("attention_span_minutes", 30),
            curiosity_index=data.get("curiosity_index", 0.7),
            complexity_level=data.get("complexity_level", 0.5),
            engagement_level=data.get("engagement_level", 0.7),
            current_mood=data.get("current_mood", "neutral")
        )


@dataclass
class QuantumResponse:
    """Revolutionary AI response with quantum intelligence"""
    content: str
    quantum_mode: QuantumLearningMode
    quantum_state: QuantumState
    intelligence_level: IntelligenceLevel
    personalization_score: float
    engagement_prediction: float
    learning_velocity_boost: float
    concept_connections: List[str]
    knowledge_gaps_identified: List[str]
    next_optimal_concepts: List[str]
    metacognitive_insights: List[str]
    emotional_resonance_score: float
    adaptive_recommendations: List[Dict[str, Any]]
    streaming_metadata: Dict[str, Any]
    quantum_analytics: Dict[str, Any]
    suggested_actions: List[str]
    next_steps: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "content": self.content,
            "quantum_mode": self.quantum_mode.value,
            "quantum_state": self.quantum_state.value,
            "intelligence_level": self.intelligence_level.value,
            "personalization_score": self.personalization_score,
            "engagement_prediction": self.engagement_prediction,
            "learning_velocity_boost": self.learning_velocity_boost,
            "concept_connections": self.concept_connections,
            "knowledge_gaps_identified": self.knowledge_gaps_identified,
            "next_optimal_concepts": self.next_optimal_concepts,
            "metacognitive_insights": self.metacognitive_insights,
            "emotional_resonance_score": self.emotional_resonance_score,
            "adaptive_recommendations": self.adaptive_recommendations,
            "streaming_metadata": self.streaming_metadata,
            "quantum_analytics": self.quantum_analytics,
            "suggested_actions": self.suggested_actions,
            "next_steps": self.next_steps,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuantumResponse':
        """Create from dictionary"""
        timestamp = datetime.fromisoformat(data.get("timestamp", datetime.now(timezone.utc).isoformat()))
        return cls(
            content=data["content"],
            quantum_mode=QuantumLearningMode(data["quantum_mode"]),
            quantum_state=QuantumState(data["quantum_state"]),
            intelligence_level=IntelligenceLevel(data["intelligence_level"]),
            personalization_score=data.get("personalization_score", 0.0),
            engagement_prediction=data.get("engagement_prediction", 0.0),
            learning_velocity_boost=data.get("learning_velocity_boost", 0.0),
            concept_connections=data.get("concept_connections", []),
            knowledge_gaps_identified=data.get("knowledge_gaps_identified", []),
            next_optimal_concepts=data.get("next_optimal_concepts", []),
            metacognitive_insights=data.get("metacognitive_insights", []),
            emotional_resonance_score=data.get("emotional_resonance_score", 0.0),
            adaptive_recommendations=data.get("adaptive_recommendations", []),
            streaming_metadata=data.get("streaming_metadata", {}),
            quantum_analytics=data.get("quantum_analytics", {}),
            suggested_actions=data.get("suggested_actions", []),
            next_steps=data.get("next_steps", ""),
            timestamp=timestamp
        )


@dataclass
class LearningDNA:
    """User's learning DNA profile"""
    user_id: str
    learning_velocity: float = 0.6
    difficulty_preference: float = 0.5
    curiosity_index: float = 0.7
    metacognitive_awareness: float = 0.5
    concept_retention_rate: float = 0.7
    attention_span_minutes: int = 30
    preferred_modalities: List[str] = field(default_factory=lambda: ["text", "visual"])
    learning_style: str = "balanced"
    motivation_factors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "user_id": self.user_id,
            "learning_velocity": self.learning_velocity,
            "difficulty_preference": self.difficulty_preference,
            "curiosity_index": self.curiosity_index,
            "metacognitive_awareness": self.metacognitive_awareness,
            "concept_retention_rate": self.concept_retention_rate,
            "attention_span_minutes": self.attention_span_minutes,
            "preferred_modalities": self.preferred_modalities,
            "learning_style": self.learning_style,
            "motivation_factors": self.motivation_factors
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningDNA':
        """Create from dictionary"""
        return cls(
            user_id=data["user_id"],
            learning_velocity=data.get("learning_velocity", 0.6),
            difficulty_preference=data.get("difficulty_preference", 0.5),
            curiosity_index=data.get("curiosity_index", 0.7),
            metacognitive_awareness=data.get("metacognitive_awareness", 0.5),
            concept_retention_rate=data.get("concept_retention_rate", 0.7),
            attention_span_minutes=data.get("attention_span_minutes", 30),
            preferred_modalities=data.get("preferred_modalities", ["text", "visual"]),
            learning_style=data.get("learning_style", "balanced"),
            motivation_factors=data.get("motivation_factors", [])
        )


@dataclass
class AdaptiveContentParameters:
    """Parameters for adaptive content generation"""
    complexity_level: float = 0.5
    engagement_level: float = 0.7
    interactivity_level: float = 0.6
    explanation_depth: float = 0.5
    example_density: float = 0.4
    challenge_level: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "complexity_level": self.complexity_level,
            "engagement_level": self.engagement_level,
            "interactivity_level": self.interactivity_level,
            "explanation_depth": self.explanation_depth,
            "example_density": self.example_density,
            "challenge_level": self.challenge_level
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdaptiveContentParameters':
        """Create from dictionary"""
        return cls(
            complexity_level=data.get("complexity_level", 0.5),
            engagement_level=data.get("engagement_level", 0.7),
            interactivity_level=data.get("interactivity_level", 0.6),
            explanation_depth=data.get("explanation_depth", 0.5),
            example_density=data.get("example_density", 0.4),
            challenge_level=data.get("challenge_level", 0.5)
        )


@dataclass
class MoodBasedAdaptation:
    """Mood-based learning adaptations"""
    current_mood: str = "neutral"
    energy_level: float = 0.7
    stress_level: float = 0.3
    motivation_level: float = 0.7
    focus_capacity: float = 0.8
    
    # Adaptation parameters
    content_pacing: float = 1.0
    difficulty_adjustment: float = 0.0
    encouragement_level: float = 0.5
    break_frequency: int = 30  # minutes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "current_mood": self.current_mood,
            "energy_level": self.energy_level,
            "stress_level": self.stress_level,
            "motivation_level": self.motivation_level,
            "focus_capacity": self.focus_capacity,
            "content_pacing": self.content_pacing,
            "difficulty_adjustment": self.difficulty_adjustment,
            "encouragement_level": self.encouragement_level,
            "break_frequency": self.break_frequency
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MoodBasedAdaptation':
        """Create from dictionary"""
        return cls(
            current_mood=data.get("current_mood", "neutral"),
            energy_level=data.get("energy_level", 0.7),
            stress_level=data.get("stress_level", 0.3),
            motivation_level=data.get("motivation_level", 0.7),
            focus_capacity=data.get("focus_capacity", 0.8),
            content_pacing=data.get("content_pacing", 1.0),
            difficulty_adjustment=data.get("difficulty_adjustment", 0.0),
            encouragement_level=data.get("encouragement_level", 0.5),
            break_frequency=data.get("break_frequency", 30)
        )
