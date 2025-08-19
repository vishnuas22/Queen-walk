"""
Mood Adaptation Engine

Extracted from quantum_intelligence_engine.py - handles mood-based learning adaptations
for emotional intelligence and wellbeing-aware learning experiences.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import re
from collections import defaultdict, deque

# Try to import structlog, fall back to standard logging
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    logger = logging.getLogger(__name__)

from ...core.data_structures import MoodBasedAdaptation
from ...core.enums import EmotionalState, LearningPace
from ...core.exceptions import QuantumEngineError
from ...utils.caching import CacheService


class MoodAdaptationEngine:
    """
    🎭 MOOD ADAPTATION ENGINE
    
    Handles mood-based learning adaptations for emotional intelligence.
    Extracted from the original quantum engine's mood adaptation logic.
    """
    
    def __init__(self, cache_service: Optional[CacheService] = None):
        self.cache = cache_service
        
        # Mood tracking
        self.mood_history = defaultdict(deque)
        self.mood_patterns = {}
        self.adaptation_cache = {}
        
        # Mood detection patterns
        self.mood_keywords = {
            "frustrated": ["frustrated", "annoying", "difficult", "hard", "stuck", "confused", "don't understand"],
            "excited": ["excited", "awesome", "great", "love", "amazing", "fantastic", "wonderful"],
            "tired": ["tired", "exhausted", "sleepy", "break", "rest", "fatigue"],
            "anxious": ["worried", "nervous", "anxious", "stressed", "pressure", "overwhelmed"],
            "confident": ["confident", "ready", "understand", "got it", "clear", "easy"],
            "curious": ["why", "how", "what if", "interesting", "curious", "wonder"],
            "bored": ["boring", "dull", "repetitive", "same", "tedious"]
        }
        
        # Adaptation strategies
        self.adaptation_strategies = {
            "frustrated": {
                "difficulty_adjustment": -0.2,
                "encouragement_level": 0.8,
                "break_frequency": 20,
                "content_pacing": 0.7,
                "explanation_depth": 0.8
            },
            "excited": {
                "difficulty_adjustment": 0.1,
                "encouragement_level": 0.3,
                "break_frequency": 40,
                "content_pacing": 1.2,
                "explanation_depth": 0.6
            },
            "tired": {
                "difficulty_adjustment": -0.1,
                "encouragement_level": 0.6,
                "break_frequency": 15,
                "content_pacing": 0.8,
                "explanation_depth": 0.5
            },
            "anxious": {
                "difficulty_adjustment": -0.15,
                "encouragement_level": 0.9,
                "break_frequency": 25,
                "content_pacing": 0.8,
                "explanation_depth": 0.7
            },
            "confident": {
                "difficulty_adjustment": 0.05,
                "encouragement_level": 0.4,
                "break_frequency": 35,
                "content_pacing": 1.1,
                "explanation_depth": 0.5
            },
            "curious": {
                "difficulty_adjustment": 0.0,
                "encouragement_level": 0.5,
                "break_frequency": 30,
                "content_pacing": 1.0,
                "explanation_depth": 0.8
            },
            "bored": {
                "difficulty_adjustment": 0.1,
                "encouragement_level": 0.7,
                "break_frequency": 20,
                "content_pacing": 1.3,
                "explanation_depth": 0.4
            },
            "neutral": {
                "difficulty_adjustment": 0.0,
                "encouragement_level": 0.5,
                "break_frequency": 30,
                "content_pacing": 1.0,
                "explanation_depth": 0.6
            }
        }
        
        logger.info("Mood Adaptation Engine initialized")
    
    async def analyze_mood_from_interaction(
        self, 
        user_id: str, 
        user_message: str,
        interaction_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Analyze user mood from interaction data
        
        Extracted from original mood analysis logic
        """
        try:
            # Detect mood from text
            detected_mood = self._detect_mood_from_text(user_message)
            
            # Analyze contextual mood indicators
            contextual_mood = self._analyze_contextual_mood_indicators(interaction_context or {})
            
            # Combine text and contextual analysis
            combined_mood_analysis = self._combine_mood_analyses(detected_mood, contextual_mood)
            
            # Update mood history
            self._update_mood_history(user_id, combined_mood_analysis)
            
            # Analyze mood patterns
            mood_patterns = self._analyze_mood_patterns(user_id)
            
            return {
                "detected_mood": combined_mood_analysis["primary_mood"],
                "mood_confidence": combined_mood_analysis["confidence"],
                "mood_indicators": combined_mood_analysis["indicators"],
                "energy_level": combined_mood_analysis["energy_level"],
                "stress_level": combined_mood_analysis["stress_level"],
                "motivation_level": combined_mood_analysis["motivation_level"],
                "focus_capacity": combined_mood_analysis["focus_capacity"],
                "mood_patterns": mood_patterns,
                "temporal_context": self._get_temporal_mood_context(user_id)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing mood for user {user_id}: {e}")
            return self._get_default_mood_analysis()
    
    async def create_mood_adaptation(
        self, 
        user_id: str, 
        mood_analysis: Dict[str, Any],
        learning_context: Dict[str, Any] = None
    ) -> MoodBasedAdaptation:
        """
        Create mood-based adaptation strategy
        
        Extracted from original mood adaptation creation logic
        """
        try:
            # Check cache first
            cache_key = f"mood_adaptation:{user_id}:{mood_analysis['detected_mood']}"
            if self.cache:
                cached_adaptation = await self.cache.get(cache_key)
                if cached_adaptation:
                    return MoodBasedAdaptation.from_dict(cached_adaptation)
            
            # Get base adaptation strategy
            primary_mood = mood_analysis["detected_mood"]
            base_strategy = self.adaptation_strategies.get(primary_mood, self.adaptation_strategies["neutral"])
            
            # Create mood-based adaptation
            adaptation = MoodBasedAdaptation(
                current_mood=primary_mood,
                energy_level=mood_analysis["energy_level"],
                stress_level=mood_analysis["stress_level"],
                motivation_level=mood_analysis["motivation_level"],
                focus_capacity=mood_analysis["focus_capacity"],
                content_pacing=base_strategy["content_pacing"],
                difficulty_adjustment=base_strategy["difficulty_adjustment"],
                encouragement_level=base_strategy["encouragement_level"],
                break_frequency=base_strategy["break_frequency"]
            )
            
            # Apply contextual adjustments
            if learning_context:
                adaptation = self._apply_contextual_adjustments(adaptation, learning_context)
            
            # Apply temporal adjustments
            adaptation = self._apply_temporal_adjustments(adaptation, user_id, mood_analysis)
            
            # Cache the adaptation
            if self.cache:
                await self.cache.set(cache_key, adaptation.to_dict(), ttl=1800)  # 30 minutes
            
            return adaptation
            
        except Exception as e:
            logger.error(f"Error creating mood adaptation for user {user_id}: {e}")
            return self._get_default_mood_adaptation()
    
    async def get_mood_based_recommendations(
        self, 
        user_id: str, 
        current_adaptation: MoodBasedAdaptation,
        session_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get mood-based learning recommendations
        
        Extracted from original mood recommendation logic
        """
        try:
            recommendations = {
                "content_adjustments": {},
                "interaction_suggestions": [],
                "wellbeing_tips": [],
                "session_modifications": {},
                "encouragement_messages": []
            }
            
            # Generate content adjustments
            recommendations["content_adjustments"] = self._generate_content_adjustments(
                current_adaptation, 
                session_data
            )
            
            # Generate interaction suggestions
            recommendations["interaction_suggestions"] = self._generate_interaction_suggestions(
                current_adaptation
            )
            
            # Generate wellbeing tips
            recommendations["wellbeing_tips"] = self._generate_wellbeing_tips(
                current_adaptation
            )
            
            # Generate session modifications
            recommendations["session_modifications"] = self._generate_session_modifications(
                current_adaptation, 
                session_data
            )
            
            # Generate encouragement messages
            recommendations["encouragement_messages"] = self._generate_encouragement_messages(
                current_adaptation
            )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting mood recommendations for user {user_id}: {e}")
            return self._get_default_recommendations()
    
    async def track_mood_progression(
        self, 
        user_id: str, 
        session_duration: int
    ) -> Dict[str, Any]:
        """
        Track mood progression throughout learning session
        
        Extracted from original mood tracking logic
        """
        try:
            mood_history = list(self.mood_history[user_id])
            
            if len(mood_history) < 2:
                return {"progression": "insufficient_data", "trend": "unknown"}
            
            # Analyze mood progression
            recent_moods = mood_history[-5:]  # Last 5 mood detections
            
            # Calculate mood stability
            mood_stability = self._calculate_mood_stability(recent_moods)
            
            # Detect mood trends
            mood_trend = self._detect_mood_trend(recent_moods)
            
            # Analyze session impact
            session_impact = self._analyze_session_mood_impact(recent_moods, session_duration)
            
            # Generate progression insights
            progression_insights = self._generate_progression_insights(
                mood_stability, 
                mood_trend, 
                session_impact
            )
            
            return {
                "progression": "tracked",
                "mood_stability": mood_stability,
                "mood_trend": mood_trend,
                "session_impact": session_impact,
                "insights": progression_insights,
                "recommendations": self._generate_progression_recommendations(
                    mood_trend, 
                    session_impact
                )
            }
            
        except Exception as e:
            logger.error(f"Error tracking mood progression for user {user_id}: {e}")
            return {"progression": "error", "trend": "unknown"}
    
    # Private helper methods
    
    def _detect_mood_from_text(self, text: str) -> Dict[str, Any]:
        """Detect mood from text using keyword analysis"""
        text_lower = text.lower()
        mood_scores = defaultdict(float)
        
        # Score each mood based on keyword matches
        for mood, keywords in self.mood_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    mood_scores[mood] += 1.0
                    
                    # Boost score for exact matches
                    if f" {keyword} " in f" {text_lower} ":
                        mood_scores[mood] += 0.5
        
        # Analyze text sentiment indicators
        sentiment_indicators = self._analyze_text_sentiment(text_lower)
        
        # Combine keyword scores with sentiment
        for mood, sentiment_score in sentiment_indicators.items():
            mood_scores[mood] += sentiment_score
        
        # Determine primary mood
        if mood_scores:
            primary_mood = max(mood_scores, key=mood_scores.get)
            confidence = mood_scores[primary_mood] / sum(mood_scores.values())
        else:
            primary_mood = "neutral"
            confidence = 0.5
        
        # Calculate derived metrics
        energy_level = self._calculate_energy_level_from_mood(primary_mood, text_lower)
        stress_level = self._calculate_stress_level_from_mood(primary_mood, text_lower)
        motivation_level = self._calculate_motivation_level_from_mood(primary_mood, text_lower)
        focus_capacity = self._calculate_focus_capacity_from_mood(primary_mood, stress_level)
        
        return {
            "primary_mood": primary_mood,
            "confidence": confidence,
            "mood_scores": dict(mood_scores),
            "indicators": list(mood_scores.keys()),
            "energy_level": energy_level,
            "stress_level": stress_level,
            "motivation_level": motivation_level,
            "focus_capacity": focus_capacity
        }
    
    def _analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze text sentiment for mood detection"""
        sentiment_scores = defaultdict(float)
        
        # Positive sentiment indicators
        positive_words = ["good", "great", "excellent", "perfect", "amazing", "wonderful", "fantastic"]
        positive_count = sum(1 for word in positive_words if word in text)
        
        # Negative sentiment indicators
        negative_words = ["bad", "terrible", "awful", "horrible", "hate", "worst", "stupid"]
        negative_count = sum(1 for word in negative_words if word in text)
        
        # Question patterns (curiosity indicators)
        question_count = text.count("?")
        
        # Exclamation patterns (excitement/frustration indicators)
        exclamation_count = text.count("!")
        
        # Apply sentiment scoring
        if positive_count > negative_count:
            sentiment_scores["excited"] += 0.3
            sentiment_scores["confident"] += 0.2
        elif negative_count > positive_count:
            sentiment_scores["frustrated"] += 0.3
            sentiment_scores["anxious"] += 0.2
        
        if question_count > 0:
            sentiment_scores["curious"] += 0.2 * question_count
        
        if exclamation_count > 0:
            if positive_count > 0:
                sentiment_scores["excited"] += 0.2 * exclamation_count
            else:
                sentiment_scores["frustrated"] += 0.2 * exclamation_count
        
        return sentiment_scores
    
    def _analyze_contextual_mood_indicators(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze contextual mood indicators"""
        contextual_mood = {
            "primary_mood": "neutral",
            "confidence": 0.3,
            "indicators": [],
            "energy_level": 0.7,
            "stress_level": 0.3,
            "motivation_level": 0.7,
            "focus_capacity": 0.8
        }
        
        # Analyze response time
        response_time = context.get("response_time", 5.0)
        if response_time > 15.0:
            contextual_mood["primary_mood"] = "tired"
            contextual_mood["energy_level"] = 0.4
            contextual_mood["confidence"] = 0.6
        elif response_time < 2.0:
            contextual_mood["primary_mood"] = "excited"
            contextual_mood["energy_level"] = 0.9
            contextual_mood["confidence"] = 0.5
        
        # Analyze session length
        session_length = context.get("session_length_minutes", 30)
        if session_length > 60:
            contextual_mood["stress_level"] += 0.2
            contextual_mood["focus_capacity"] -= 0.2
        
        # Analyze time of day
        hour = context.get("hour_of_day", 12)
        if hour < 8 or hour > 22:
            contextual_mood["energy_level"] -= 0.2
            contextual_mood["focus_capacity"] -= 0.1
        
        # Analyze recent performance
        recent_success = context.get("recent_success_rate", 0.7)
        if recent_success < 0.5:
            contextual_mood["primary_mood"] = "frustrated"
            contextual_mood["stress_level"] += 0.3
            contextual_mood["confidence"] = 0.7
        elif recent_success > 0.8:
            contextual_mood["primary_mood"] = "confident"
            contextual_mood["motivation_level"] += 0.2
            contextual_mood["confidence"] = 0.6
        
        return contextual_mood
    
    def _combine_mood_analyses(
        self, 
        text_mood: Dict[str, Any], 
        contextual_mood: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine text and contextual mood analyses"""
        # Weight text analysis higher than contextual
        text_weight = 0.7
        context_weight = 0.3
        
        # Combine confidence scores
        combined_confidence = (
            text_mood["confidence"] * text_weight + 
            contextual_mood["confidence"] * context_weight
        )
        
        # Choose primary mood based on confidence
        if text_mood["confidence"] > contextual_mood["confidence"]:
            primary_mood = text_mood["primary_mood"]
        else:
            primary_mood = contextual_mood["primary_mood"]
        
        # Combine metrics
        combined_analysis = {
            "primary_mood": primary_mood,
            "confidence": combined_confidence,
            "indicators": list(set(text_mood["indicators"] + contextual_mood["indicators"])),
            "energy_level": (
                text_mood["energy_level"] * text_weight + 
                contextual_mood["energy_level"] * context_weight
            ),
            "stress_level": (
                text_mood["stress_level"] * text_weight + 
                contextual_mood["stress_level"] * context_weight
            ),
            "motivation_level": (
                text_mood["motivation_level"] * text_weight + 
                contextual_mood["motivation_level"] * context_weight
            ),
            "focus_capacity": (
                text_mood["focus_capacity"] * text_weight + 
                contextual_mood["focus_capacity"] * context_weight
            )
        }
        
        # Ensure values are within bounds
        for key in ["energy_level", "stress_level", "motivation_level", "focus_capacity"]:
            combined_analysis[key] = max(0.0, min(1.0, combined_analysis[key]))
        
        return combined_analysis
    
    def _update_mood_history(self, user_id: str, mood_analysis: Dict[str, Any]):
        """Update mood history for user"""
        mood_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "mood": mood_analysis["primary_mood"],
            "confidence": mood_analysis["confidence"],
            "energy_level": mood_analysis["energy_level"],
            "stress_level": mood_analysis["stress_level"],
            "motivation_level": mood_analysis["motivation_level"],
            "focus_capacity": mood_analysis["focus_capacity"]
        }
        
        self.mood_history[user_id].append(mood_entry)
        
        # Keep only recent history (last 50 entries)
        if len(self.mood_history[user_id]) > 50:
            self.mood_history[user_id].popleft()
    
    def _analyze_mood_patterns(self, user_id: str) -> Dict[str, Any]:
        """Analyze mood patterns for user"""
        history = list(self.mood_history[user_id])
        
        if len(history) < 3:
            return {"pattern": "insufficient_data", "stability": 0.5}
        
        # Analyze mood frequency
        mood_frequency = defaultdict(int)
        for entry in history:
            mood_frequency[entry["mood"]] += 1
        
        # Calculate mood stability
        total_entries = len(history)
        dominant_mood = max(mood_frequency, key=mood_frequency.get)
        stability = mood_frequency[dominant_mood] / total_entries
        
        # Analyze mood transitions
        transitions = []
        for i in range(1, len(history)):
            if history[i]["mood"] != history[i-1]["mood"]:
                transitions.append((history[i-1]["mood"], history[i]["mood"]))
        
        return {
            "pattern": "analyzed",
            "dominant_mood": dominant_mood,
            "stability": stability,
            "mood_frequency": dict(mood_frequency),
            "transition_count": len(transitions),
            "common_transitions": self._get_common_transitions(transitions)
        }
    
    def _get_common_transitions(self, transitions: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Get most common mood transitions"""
        transition_counts = defaultdict(int)
        for transition in transitions:
            transition_counts[transition] += 1
        
        # Return top 3 most common transitions
        sorted_transitions = sorted(
            transition_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [transition for transition, count in sorted_transitions[:3]]
    
    def _get_temporal_mood_context(self, user_id: str) -> Dict[str, Any]:
        """Get temporal context for mood analysis"""
        history = list(self.mood_history[user_id])
        
        if not history:
            return {"context": "no_history"}
        
        # Analyze recent mood trend
        recent_moods = [entry["mood"] for entry in history[-5:]]
        
        # Check for mood improvement/deterioration
        if len(recent_moods) >= 3:
            positive_moods = ["excited", "confident", "curious"]
            negative_moods = ["frustrated", "tired", "anxious"]
            
            recent_positive = sum(1 for mood in recent_moods[-3:] if mood in positive_moods)
            recent_negative = sum(1 for mood in recent_moods[-3:] if mood in negative_moods)
            
            if recent_positive > recent_negative:
                trend = "improving"
            elif recent_negative > recent_positive:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "unknown"
        
        return {
            "context": "analyzed",
            "recent_trend": trend,
            "session_mood_count": len(recent_moods),
            "last_mood": recent_moods[-1] if recent_moods else "unknown"
        }
    
    # Mood metric calculation methods
    
    def _calculate_energy_level_from_mood(self, mood: str, text: str) -> float:
        """Calculate energy level from mood and text"""
        energy_mapping = {
            "excited": 0.9,
            "confident": 0.8,
            "curious": 0.7,
            "neutral": 0.7,
            "frustrated": 0.5,
            "anxious": 0.4,
            "tired": 0.3,
            "bored": 0.4
        }
        
        base_energy = energy_mapping.get(mood, 0.7)
        
        # Adjust based on text indicators
        if any(word in text for word in ["energy", "energetic", "active"]):
            base_energy += 0.1
        elif any(word in text for word in ["tired", "exhausted", "sleepy"]):
            base_energy -= 0.2
        
        return max(0.0, min(1.0, base_energy))
    
    def _calculate_stress_level_from_mood(self, mood: str, text: str) -> float:
        """Calculate stress level from mood and text"""
        stress_mapping = {
            "excited": 0.2,
            "confident": 0.1,
            "curious": 0.2,
            "neutral": 0.3,
            "frustrated": 0.7,
            "anxious": 0.8,
            "tired": 0.5,
            "bored": 0.3
        }
        
        base_stress = stress_mapping.get(mood, 0.3)
        
        # Adjust based on text indicators
        if any(word in text for word in ["stress", "pressure", "overwhelmed"]):
            base_stress += 0.2
        elif any(word in text for word in ["calm", "relaxed", "peaceful"]):
            base_stress -= 0.2
        
        return max(0.0, min(1.0, base_stress))
    
    def _calculate_motivation_level_from_mood(self, mood: str, text: str) -> float:
        """Calculate motivation level from mood and text"""
        motivation_mapping = {
            "excited": 0.9,
            "confident": 0.8,
            "curious": 0.8,
            "neutral": 0.6,
            "frustrated": 0.4,
            "anxious": 0.5,
            "tired": 0.3,
            "bored": 0.2
        }
        
        base_motivation = motivation_mapping.get(mood, 0.6)
        
        # Adjust based on text indicators
        if any(word in text for word in ["goal", "achieve", "want to", "motivated"]):
            base_motivation += 0.2
        elif any(word in text for word in ["give up", "quit", "don't want"]):
            base_motivation -= 0.3
        
        return max(0.0, min(1.0, base_motivation))
    
    def _calculate_focus_capacity_from_mood(self, mood: str, stress_level: float) -> float:
        """Calculate focus capacity from mood and stress level"""
        focus_mapping = {
            "excited": 0.7,
            "confident": 0.8,
            "curious": 0.9,
            "neutral": 0.7,
            "frustrated": 0.4,
            "anxious": 0.3,
            "tired": 0.4,
            "bored": 0.3
        }
        
        base_focus = focus_mapping.get(mood, 0.7)
        
        # Adjust based on stress level (high stress reduces focus)
        stress_adjustment = -0.5 * stress_level
        
        return max(0.0, min(1.0, base_focus + stress_adjustment))
    
    # Adaptation methods
    
    def _apply_contextual_adjustments(
        self, 
        adaptation: MoodBasedAdaptation, 
        context: Dict[str, Any]
    ) -> MoodBasedAdaptation:
        """Apply contextual adjustments to mood adaptation"""
        adjusted_adaptation = MoodBasedAdaptation(
            current_mood=adaptation.current_mood,
            energy_level=adaptation.energy_level,
            stress_level=adaptation.stress_level,
            motivation_level=adaptation.motivation_level,
            focus_capacity=adaptation.focus_capacity,
            content_pacing=adaptation.content_pacing,
            difficulty_adjustment=adaptation.difficulty_adjustment,
            encouragement_level=adaptation.encouragement_level,
            break_frequency=adaptation.break_frequency
        )
        
        # Adjust based on session length
        session_length = context.get("planned_session_length", 30)
        if session_length > 60:
            adjusted_adaptation.break_frequency = max(15, adjusted_adaptation.break_frequency - 10)
            adjusted_adaptation.content_pacing *= 0.9
        
        # Adjust based on topic difficulty
        topic_difficulty = context.get("topic_difficulty", 0.5)
        if topic_difficulty > 0.7:
            adjusted_adaptation.difficulty_adjustment -= 0.1
            adjusted_adaptation.encouragement_level += 0.1
        
        return adjusted_adaptation
    
    def _apply_temporal_adjustments(
        self, 
        adaptation: MoodBasedAdaptation, 
        user_id: str, 
        mood_analysis: Dict[str, Any]
    ) -> MoodBasedAdaptation:
        """Apply temporal adjustments based on mood patterns"""
        mood_patterns = mood_analysis.get("mood_patterns", {})
        
        # If user has unstable mood patterns, be more conservative
        if mood_patterns.get("stability", 0.5) < 0.3:
            adaptation.difficulty_adjustment *= 0.8
            adaptation.encouragement_level += 0.1
            adaptation.break_frequency = max(20, adaptation.break_frequency - 5)
        
        return adaptation
    
    # Default fallback methods
    
    def _get_default_mood_analysis(self) -> Dict[str, Any]:
        """Get default mood analysis for fallback"""
        return {
            "detected_mood": "neutral",
            "mood_confidence": 0.5,
            "mood_indicators": [],
            "energy_level": 0.7,
            "stress_level": 0.3,
            "motivation_level": 0.7,
            "focus_capacity": 0.8,
            "mood_patterns": {"pattern": "default", "stability": 0.5},
            "temporal_context": {"context": "default"}
        }
    
    def _get_default_mood_adaptation(self) -> MoodBasedAdaptation:
        """Get default mood adaptation for fallback"""
        return MoodBasedAdaptation(
            current_mood="neutral",
            energy_level=0.7,
            stress_level=0.3,
            motivation_level=0.7,
            focus_capacity=0.8,
            content_pacing=1.0,
            difficulty_adjustment=0.0,
            encouragement_level=0.5,
            break_frequency=30
        )
    
    def _get_default_recommendations(self) -> Dict[str, Any]:
        """Get default recommendations for fallback"""
        return {
            "content_adjustments": {},
            "interaction_suggestions": ["Continue with current approach"],
            "wellbeing_tips": ["Take breaks when needed"],
            "session_modifications": {},
            "encouragement_messages": ["You're doing great! Keep learning."]
        }
    
    # Recommendation generation methods
    
    def _generate_content_adjustments(
        self, 
        adaptation: MoodBasedAdaptation, 
        session_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate content adjustments based on mood"""
        adjustments = {}
        
        if adaptation.current_mood == "frustrated":
            adjustments.update({
                "reduce_complexity": True,
                "add_examples": True,
                "increase_explanation_depth": True,
                "add_encouragement": True
            })
        elif adaptation.current_mood == "excited":
            adjustments.update({
                "increase_challenge": True,
                "add_advanced_topics": True,
                "reduce_basic_explanations": True
            })
        elif adaptation.current_mood == "tired":
            adjustments.update({
                "simplify_content": True,
                "add_visual_elements": True,
                "reduce_text_density": True,
                "suggest_break": True
            })
        elif adaptation.current_mood == "bored":
            adjustments.update({
                "increase_interactivity": True,
                "add_variety": True,
                "increase_pace": True,
                "add_challenges": True
            })
        
        return adjustments
    
    def _generate_interaction_suggestions(self, adaptation: MoodBasedAdaptation) -> List[str]:
        """Generate interaction suggestions based on mood"""
        suggestions = []
        
        mood_suggestions = {
            "frustrated": [
                "Take a step back and review fundamentals",
                "Break down complex problems into smaller parts",
                "Ask for clarification when needed",
                "Remember that struggle is part of learning"
            ],
            "excited": [
                "Channel your enthusiasm into deeper exploration",
                "Try tackling more challenging problems",
                "Share your excitement and insights",
                "Set ambitious but achievable goals"
            ],
            "tired": [
                "Consider taking a short break",
                "Focus on review rather than new material",
                "Use visual aids to maintain engagement",
                "Keep sessions shorter today"
            ],
            "anxious": [
                "Take deep breaths and go at your own pace",
                "Focus on one concept at a time",
                "Remember that it's okay to make mistakes",
                "Celebrate small victories"
            ],
            "confident": [
                "Challenge yourself with advanced topics",
                "Help others who might be struggling",
                "Explore related concepts independently",
                "Set new learning goals"
            ],
            "curious": [
                "Follow your questions wherever they lead",
                "Explore connections between concepts",
                "Ask 'what if' questions",
                "Dive deeper into topics that interest you"
            ],
            "bored": [
                "Try a different learning approach",
                "Look for real-world applications",
                "Challenge yourself with harder problems",
                "Take a break and come back refreshed"
            ]
        }
        
        return mood_suggestions.get(adaptation.current_mood, ["Continue with your current approach"])
    
    def _generate_wellbeing_tips(self, adaptation: MoodBasedAdaptation) -> List[str]:
        """Generate wellbeing tips based on mood"""
        tips = []
        
        if adaptation.stress_level > 0.6:
            tips.extend([
                "Practice deep breathing exercises",
                "Take regular breaks to reduce stress",
                "Remember that learning is a process, not a race"
            ])
        
        if adaptation.energy_level < 0.4:
            tips.extend([
                "Consider taking a longer break",
                "Stay hydrated and have a healthy snack",
                "Get some fresh air or light exercise"
            ])
        
        if adaptation.motivation_level < 0.5:
            tips.extend([
                "Remind yourself of your learning goals",
                "Celebrate small achievements",
                "Connect with others who share your interests"
            ])
        
        if adaptation.focus_capacity < 0.5:
            tips.extend([
                "Minimize distractions in your environment",
                "Try shorter, more focused learning sessions",
                "Use techniques like the Pomodoro method"
            ])
        
        return tips if tips else ["Keep up the great work!"]
    
    def _generate_session_modifications(
        self, 
        adaptation: MoodBasedAdaptation, 
        session_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate session modifications based on mood"""
        modifications = {}
        
        # Adjust session length
        if adaptation.focus_capacity < 0.5 or adaptation.energy_level < 0.4:
            modifications["suggested_session_length"] = min(20, session_data.get("planned_length", 30))
        elif adaptation.energy_level > 0.8 and adaptation.motivation_level > 0.8:
            modifications["suggested_session_length"] = min(60, session_data.get("planned_length", 30) * 1.5)
        
        # Adjust break frequency
        modifications["break_frequency"] = adaptation.break_frequency
        
        # Adjust content pacing
        modifications["content_pacing"] = adaptation.content_pacing
        
        # Suggest specific activities
        if adaptation.current_mood == "frustrated":
            modifications["suggested_activities"] = ["review", "practice_basics", "get_help"]
        elif adaptation.current_mood == "excited":
            modifications["suggested_activities"] = ["explore_advanced", "tackle_challenges", "create_projects"]
        elif adaptation.current_mood == "tired":
            modifications["suggested_activities"] = ["light_review", "visual_learning", "short_exercises"]
        
        return modifications
    
    def _generate_encouragement_messages(self, adaptation: MoodBasedAdaptation) -> List[str]:
        """Generate encouragement messages based on mood"""
        messages = []
        
        mood_messages = {
            "frustrated": [
                "Every expert was once a beginner. You're making progress!",
                "Challenges are opportunities to grow stronger.",
                "It's okay to struggle - that's how learning happens.",
                "Take it one step at a time. You've got this!"
            ],
            "excited": [
                "Your enthusiasm is contagious! Keep that energy flowing.",
                "Amazing work! Your excitement for learning shows.",
                "You're on fire! Channel that energy into your next challenge.",
                "Your positive attitude is a superpower in learning!"
            ],
            "tired": [
                "Rest is part of the learning process. Be kind to yourself.",
                "Even small steps forward are still progress.",
                "Your brain is working hard. A break might help you recharge.",
                "Learning takes energy. Make sure to take care of yourself."
            ],
            "anxious": [
                "You're braver than you believe and stronger than you seem.",
                "Anxiety is normal when learning new things. You're not alone.",
                "Take deep breaths. You have everything you need to succeed.",
                "Progress, not perfection, is the goal."
            ],
            "confident": [
                "Your confidence is well-earned! Keep building on your success.",
                "You're demonstrating real mastery. Well done!",
                "Your self-assurance is inspiring. Keep reaching higher!",
                "Confidence and competence go hand in hand. You have both!"
            ],
            "curious": [
                "Your curiosity is your greatest learning asset!",
                "Questions are the engine of learning. Keep asking!",
                "Your inquisitive mind will take you far.",
                "Curiosity is the spark that ignites understanding."
            ],
            "bored": [
                "Sometimes a change of pace can reignite interest.",
                "Boredom might mean you're ready for a new challenge.",
                "Every topic has hidden depths waiting to be discovered.",
                "Your mind is seeking stimulation. Let's find it together!"
            ]
        }
        
        mood_specific = mood_messages.get(adaptation.current_mood, [])
        
        # Add general encouragement based on adaptation levels
        if adaptation.encouragement_level > 0.7:
            mood_specific.extend([
                "You're doing fantastic! Keep up the excellent work!",
                "Your dedication to learning is truly admirable.",
                "Every moment you spend learning is an investment in yourself."
            ])
        
        return mood_specific[:3]  # Return top 3 messages
    
    # Mood progression analysis methods
    
    def _calculate_mood_stability(self, recent_moods: List[Dict[str, Any]]) -> float:
        """Calculate mood stability from recent mood history"""
        if len(recent_moods) < 2:
            return 0.5
        
        mood_changes = 0
        for i in range(1, len(recent_moods)):
            if recent_moods[i]["mood"] != recent_moods[i-1]["mood"]:
                mood_changes += 1
        
        # Stability is inverse of change rate
        stability = 1.0 - (mood_changes / (len(recent_moods) - 1))
        return max(0.0, min(1.0, stability))
    
    def _detect_mood_trend(self, recent_moods: List[Dict[str, Any]]) -> str:
        """Detect mood trend from recent history"""
        if len(recent_moods) < 3:
            return "insufficient_data"
        
        positive_moods = ["excited", "confident", "curious"]
        negative_moods = ["frustrated", "tired", "anxious", "bored"]
        
        # Analyze first half vs second half
        mid_point = len(recent_moods) // 2
        first_half = recent_moods[:mid_point]
        second_half = recent_moods[mid_point:]
        
        first_positive = sum(1 for mood in first_half if mood["mood"] in positive_moods)
        second_positive = sum(1 for mood in second_half if mood["mood"] in positive_moods)
        
        first_negative = sum(1 for mood in first_half if mood["mood"] in negative_moods)
        second_negative = sum(1 for mood in second_half if mood["mood"] in negative_moods)
        
        # Calculate trend
        positive_trend = second_positive - first_positive
        negative_trend = second_negative - first_negative
        
        if positive_trend > 0 and negative_trend <= 0:
            return "improving"
        elif negative_trend > 0 and positive_trend <= 0:
            return "declining"
        else:
            return "stable"
    
    def _analyze_session_mood_impact(
        self, 
        recent_moods: List[Dict[str, Any]], 
        session_duration: int
    ) -> Dict[str, Any]:
        """Analyze how the session has impacted mood"""
        if len(recent_moods) < 2:
            return {"impact": "insufficient_data"}
        
        session_start_mood = recent_moods[0]["mood"]
        session_end_mood = recent_moods[-1]["mood"]
        
        positive_moods = ["excited", "confident", "curious"]
        negative_moods = ["frustrated", "tired", "anxious", "bored"]
        
        # Determine impact
        if session_start_mood in negative_moods and session_end_mood in positive_moods:
            impact = "very_positive"
        elif session_start_mood in positive_moods and session_end_mood in negative_moods:
            impact = "negative"
        elif session_start_mood == session_end_mood:
            impact = "stable"
        elif session_end_mood in positive_moods:
            impact = "positive"
        else:
            impact = "mixed"
        
        # Calculate mood score change
        mood_scores = {
            "excited": 0.9, "confident": 0.8, "curious": 0.7, "neutral": 0.5,
            "bored": 0.4, "tired": 0.3, "anxious": 0.2, "frustrated": 0.1
        }
        
        start_score = mood_scores.get(session_start_mood, 0.5)
        end_score = mood_scores.get(session_end_mood, 0.5)
        score_change = end_score - start_score
        
        return {
            "impact": impact,
            "start_mood": session_start_mood,
            "end_mood": session_end_mood,
            "score_change": score_change,
            "session_duration": session_duration
        }
    
    def _generate_progression_insights(
        self, 
        mood_stability: float, 
        mood_trend: str, 
        session_impact: Dict[str, Any]
    ) -> List[str]:
        """Generate insights about mood progression"""
        insights = []
        
        # Stability insights
        if mood_stability > 0.8:
            insights.append("Your mood has been very stable throughout the session.")
        elif mood_stability < 0.3:
            insights.append("Your mood has been quite variable - this is normal when learning challenging material.")
        
        # Trend insights
        if mood_trend == "improving":
            insights.append("Your mood has been improving as the session progresses - great sign!")
        elif mood_trend == "declining":
            insights.append("Your mood seems to be declining - consider taking a break or adjusting the difficulty.")
        
        # Session impact insights
        impact = session_impact.get("impact", "unknown")
        if impact == "very_positive":
            insights.append("This session has had a very positive impact on your mood!")
        elif impact == "positive":
            insights.append("This session has been beneficial for your mood.")
        elif impact == "negative":
            insights.append("This session seems to have been challenging for your mood - let's adjust our approach.")
        
        return insights if insights else ["Your mood patterns are being tracked for personalized learning."]
    
    def _generate_progression_recommendations(
        self, 
        mood_trend: str, 
        session_impact: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on mood progression"""
        recommendations = []
        
        if mood_trend == "declining":
            recommendations.extend([
                "Consider taking a break to reset your mood",
                "Try switching to a different type of learning activity",
                "Reduce the difficulty level temporarily"
            ])
        elif mood_trend == "improving":
            recommendations.extend([
                "You're on a positive trajectory - keep going!",
                "Consider gradually increasing the challenge level",
                "This approach seems to be working well for you"
            ])
        
        impact = session_impact.get("impact", "unknown")
        if impact == "negative":
            recommendations.extend([
                "End the session on a positive note with a review of what you've learned",
                "Plan a shorter session next time",
                "Consider what factors might have contributed to the mood change"
            ])
        elif impact in ["positive", "very_positive"]:
            recommendations.extend([
                "Great session! Consider what made it successful",
                "You might be ready for more challenging material",
                "This session length and approach seem optimal for you"
            ])
        
        return recommendations if recommendations else ["Continue monitoring your mood patterns for optimal learning."]
