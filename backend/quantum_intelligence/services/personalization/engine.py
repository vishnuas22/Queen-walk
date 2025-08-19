"""
Personalization Engine

Extracted from quantum_intelligence_engine.py - handles all personalization logic
including learning DNA analysis, mood adaptation, and adaptive content parameters.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
from collections import defaultdict

# Try to import structlog, fall back to standard logging
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    logger = logging.getLogger(__name__)

from ...core.data_structures import LearningDNA, AdaptiveContentParameters, MoodBasedAdaptation
from ...core.enums import LearningStyle, EmotionalState, LearningPace
from ...core.exceptions import QuantumEngineError
from ...utils.caching import CacheService


class PersonalizationEngine:
    """
    🎯 PERSONALIZATION ENGINE
    
    Handles all personalization logic extracted from the original quantum engine.
    Provides learning DNA analysis, mood adaptation, and adaptive content parameters.
    """
    
    def __init__(self, cache_service: Optional[CacheService] = None):
        self.cache = cache_service
        
        # Personalization caches
        self.user_profiles = {}
        self.learning_dna_cache = {}
        self.mood_adaptation_cache = {}
        self.adaptive_parameters_cache = {}
        
        # Analysis engines
        self.learning_dna_manager = None  # Will be injected
        self.mood_adaptation_engine = None  # Will be injected
        self.adaptive_parameters_engine = None  # Will be injected
        
        logger.info("Personalization Engine initialized")
    
    async def analyze_learning_dna(self, user_id: str) -> LearningDNA:
        """
        Analyze user's learning DNA for personalization
        
        Extracted from original _calculate_personalization_score and related methods
        """
        try:
            # Check cache first
            if self.cache:
                cached_dna = await self.cache.get(f"learning_dna:{user_id}")
                if cached_dna:
                    return LearningDNA.from_dict(cached_dna)
            
            # Check in-memory cache
            if user_id in self.learning_dna_cache:
                return self.learning_dna_cache[user_id]
            
            # Analyze learning DNA (simplified version of original logic)
            learning_dna = await self._analyze_user_learning_patterns(user_id)
            
            # Cache the result
            self.learning_dna_cache[user_id] = learning_dna
            if self.cache:
                await self.cache.set(f"learning_dna:{user_id}", learning_dna.to_dict(), ttl=3600)
            
            return learning_dna
            
        except Exception as e:
            logger.error(f"Error analyzing learning DNA for user {user_id}: {e}")
            # Return default learning DNA
            return self._get_default_learning_dna(user_id)
    
    async def analyze_mood_and_adapt(
        self, 
        user_id: str, 
        recent_messages: List[Dict[str, Any]], 
        context: Dict[str, Any]
    ) -> MoodBasedAdaptation:
        """
        Analyze user mood and provide adaptations
        
        Extracted from original mood adaptation logic
        """
        try:
            # Check cache first
            cache_key = f"mood_adaptation:{user_id}:{hash(str(recent_messages))}"
            if self.cache:
                cached_adaptation = await self.cache.get(cache_key)
                if cached_adaptation:
                    return MoodBasedAdaptation.from_dict(cached_adaptation)
            
            # Analyze mood from recent messages
            mood_analysis = await self._analyze_mood_from_messages(recent_messages, context)
            
            # Create mood-based adaptation
            adaptation = MoodBasedAdaptation(
                current_mood=mood_analysis.get("detected_mood", "neutral"),
                energy_level=mood_analysis.get("energy_level", 0.7),
                stress_level=mood_analysis.get("stress_level", 0.3),
                motivation_level=mood_analysis.get("motivation_level", 0.7),
                focus_capacity=mood_analysis.get("focus_capacity", 0.8),
                content_pacing=mood_analysis.get("content_pacing", 1.0),
                difficulty_adjustment=mood_analysis.get("difficulty_adjustment", 0.0),
                encouragement_level=mood_analysis.get("encouragement_level", 0.5),
                break_frequency=mood_analysis.get("break_frequency", 30)
            )
            
            # Cache the result
            if self.cache:
                await self.cache.set(cache_key, adaptation.to_dict(), ttl=1800)  # 30 minutes
            
            return adaptation
            
        except Exception as e:
            logger.error(f"Error analyzing mood for user {user_id}: {e}")
            # Return default mood adaptation
            return self._get_default_mood_adaptation()
    
    async def get_adaptive_content_parameters(
        self, 
        user_id: str, 
        context: Dict[str, Any]
    ) -> AdaptiveContentParameters:
        """
        Get adaptive content parameters for personalized content generation
        
        Extracted from original adaptive content logic
        """
        try:
            # Check cache first
            cache_key = f"adaptive_params:{user_id}:{hash(str(context))}"
            if self.cache:
                cached_params = await self.cache.get(cache_key)
                if cached_params:
                    return AdaptiveContentParameters.from_dict(cached_params)
            
            # Get learning DNA for personalization
            learning_dna = await self.analyze_learning_dna(user_id)
            
            # Calculate adaptive parameters based on learning DNA and context
            parameters = AdaptiveContentParameters(
                complexity_level=self._calculate_complexity_level(learning_dna, context),
                engagement_level=self._calculate_engagement_level(learning_dna, context),
                interactivity_level=self._calculate_interactivity_level(learning_dna, context),
                explanation_depth=self._calculate_explanation_depth(learning_dna, context),
                example_density=self._calculate_example_density(learning_dna, context),
                challenge_level=self._calculate_challenge_level(learning_dna, context)
            )
            
            # Cache the result
            if self.cache:
                await self.cache.set(cache_key, parameters.to_dict(), ttl=1800)  # 30 minutes
            
            return parameters
            
        except Exception as e:
            logger.error(f"Error getting adaptive parameters for user {user_id}: {e}")
            # Return default parameters
            return AdaptiveContentParameters()
    
    async def calculate_personalization_score(
        self, 
        user_id: str, 
        content: str, 
        context: Dict[str, Any]
    ) -> float:
        """
        Calculate personalization score for content
        
        Extracted from original _calculate_personalization_score method
        """
        try:
            learning_dna = await self.analyze_learning_dna(user_id)
            mood_adaptation = await self.analyze_mood_and_adapt(user_id, [], context)
            
            # Calculate personalization score based on multiple factors
            style_match = self._calculate_learning_style_match(learning_dna, content)
            mood_match = self._calculate_mood_match(mood_adaptation, content)
            difficulty_match = self._calculate_difficulty_match(learning_dna, context)
            pace_match = self._calculate_pace_match(learning_dna, context)
            
            # Weighted average
            personalization_score = (
                style_match * 0.3 +
                mood_match * 0.25 +
                difficulty_match * 0.25 +
                pace_match * 0.2
            )
            
            return min(1.0, max(0.0, personalization_score))
            
        except Exception as e:
            logger.error(f"Error calculating personalization score: {e}")
            return 0.5  # Default score
    
    # Private helper methods
    
    async def _analyze_user_learning_patterns(self, user_id: str) -> LearningDNA:
        """Analyze user learning patterns to create learning DNA"""
        # This would integrate with database and analytics in production
        # For now, return a default learning DNA with some variation
        
        return LearningDNA(
            user_id=user_id,
            learning_velocity=0.6 + (hash(user_id) % 40) / 100,  # 0.6-1.0
            difficulty_preference=0.5 + (hash(user_id) % 50) / 100,  # 0.5-1.0
            curiosity_index=0.7 + (hash(user_id) % 30) / 100,  # 0.7-1.0
            metacognitive_awareness=0.5 + (hash(user_id) % 50) / 100,
            concept_retention_rate=0.7 + (hash(user_id) % 30) / 100,
            attention_span_minutes=20 + (hash(user_id) % 40),  # 20-60 minutes
            preferred_modalities=["text", "visual"],  # Could be personalized
            learning_style="balanced",
            motivation_factors=["achievement", "curiosity"]
        )
    
    def _get_default_learning_dna(self, user_id: str) -> LearningDNA:
        """Get default learning DNA for fallback"""
        return LearningDNA(
            user_id=user_id,
            learning_velocity=0.6,
            difficulty_preference=0.5,
            curiosity_index=0.7,
            metacognitive_awareness=0.5,
            concept_retention_rate=0.7,
            attention_span_minutes=30,
            preferred_modalities=["text", "visual"],
            learning_style="balanced",
            motivation_factors=["achievement"]
        )
    
    async def _analyze_mood_from_messages(
        self, 
        messages: List[Dict[str, Any]], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze mood from recent messages"""
        # Simplified mood analysis - in production this would use NLP
        
        if not messages:
            return {
                "detected_mood": "neutral",
                "energy_level": 0.7,
                "stress_level": 0.3,
                "motivation_level": 0.7,
                "focus_capacity": 0.8,
                "content_pacing": 1.0,
                "difficulty_adjustment": 0.0,
                "encouragement_level": 0.5,
                "break_frequency": 30
            }
        
        # Analyze message content for mood indicators
        recent_content = " ".join([msg.get("content", "") for msg in messages[-3:]])
        content_lower = recent_content.lower()
        
        # Simple keyword-based mood detection
        if any(word in content_lower for word in ["frustrated", "confused", "difficult", "hard"]):
            detected_mood = "frustrated"
            energy_level = 0.4
            stress_level = 0.7
            encouragement_level = 0.8
            difficulty_adjustment = -0.2
        elif any(word in content_lower for word in ["excited", "great", "awesome", "love"]):
            detected_mood = "excited"
            energy_level = 0.9
            stress_level = 0.2
            encouragement_level = 0.3
            difficulty_adjustment = 0.1
        elif any(word in content_lower for word in ["tired", "exhausted", "break"]):
            detected_mood = "tired"
            energy_level = 0.3
            stress_level = 0.5
            encouragement_level = 0.7
            break_frequency = 20
        else:
            detected_mood = "neutral"
            energy_level = 0.7
            stress_level = 0.3
            encouragement_level = 0.5
            difficulty_adjustment = 0.0
        
        return {
            "detected_mood": detected_mood,
            "energy_level": energy_level,
            "stress_level": stress_level,
            "motivation_level": 0.7,
            "focus_capacity": max(0.3, 1.0 - stress_level),
            "content_pacing": energy_level,
            "difficulty_adjustment": difficulty_adjustment,
            "encouragement_level": encouragement_level,
            "break_frequency": locals().get("break_frequency", 30)
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
    
    # Adaptive parameter calculation methods
    
    def _calculate_complexity_level(self, learning_dna: LearningDNA, context: Dict[str, Any]) -> float:
        """Calculate optimal complexity level"""
        base_complexity = learning_dna.difficulty_preference
        
        # Adjust based on context
        if context.get("topic_difficulty", 0.5) > 0.7:
            base_complexity *= 0.8  # Reduce complexity for difficult topics
        
        return min(1.0, max(0.0, base_complexity))
    
    def _calculate_engagement_level(self, learning_dna: LearningDNA, context: Dict[str, Any]) -> float:
        """Calculate optimal engagement level"""
        base_engagement = learning_dna.curiosity_index
        
        # Adjust based on learning velocity
        if learning_dna.learning_velocity > 0.8:
            base_engagement += 0.1  # Higher engagement for fast learners
        
        return min(1.0, max(0.0, base_engagement))
    
    def _calculate_interactivity_level(self, learning_dna: LearningDNA, context: Dict[str, Any]) -> float:
        """Calculate optimal interactivity level"""
        # Base on learning style and attention span
        base_interactivity = 0.6
        
        if learning_dna.attention_span_minutes < 25:
            base_interactivity += 0.2  # More interactivity for shorter attention spans
        
        return min(1.0, max(0.0, base_interactivity))
    
    def _calculate_explanation_depth(self, learning_dna: LearningDNA, context: Dict[str, Any]) -> float:
        """Calculate optimal explanation depth"""
        return learning_dna.metacognitive_awareness
    
    def _calculate_example_density(self, learning_dna: LearningDNA, context: Dict[str, Any]) -> float:
        """Calculate optimal example density"""
        # More examples for visual learners and lower difficulty preference
        base_density = 0.5
        
        if "visual" in learning_dna.preferred_modalities:
            base_density += 0.2
        
        if learning_dna.difficulty_preference < 0.5:
            base_density += 0.1
        
        return min(1.0, max(0.0, base_density))
    
    def _calculate_challenge_level(self, learning_dna: LearningDNA, context: Dict[str, Any]) -> float:
        """Calculate optimal challenge level"""
        return learning_dna.difficulty_preference
    
    # Personalization score calculation methods
    
    def _calculate_learning_style_match(self, learning_dna: LearningDNA, content: str) -> float:
        """Calculate how well content matches learning style"""
        # Simplified analysis - in production would use NLP
        content_lower = content.lower()
        
        score = 0.5  # Base score
        
        # Check for visual elements
        if "visual" in learning_dna.preferred_modalities:
            if any(word in content_lower for word in ["diagram", "chart", "visual", "image", "graph"]):
                score += 0.2
        
        # Check for text elements
        if "text" in learning_dna.preferred_modalities:
            if len(content) > 100:  # Substantial text content
                score += 0.1
        
        return min(1.0, score)
    
    def _calculate_mood_match(self, mood_adaptation: MoodBasedAdaptation, content: str) -> float:
        """Calculate how well content matches current mood"""
        content_lower = content.lower()
        
        score = 0.5  # Base score
        
        # Adjust based on mood
        if mood_adaptation.current_mood == "frustrated":
            if any(word in content_lower for word in ["step", "simple", "easy", "break down"]):
                score += 0.3
        elif mood_adaptation.current_mood == "excited":
            if any(word in content_lower for word in ["challenge", "advanced", "explore"]):
                score += 0.3
        elif mood_adaptation.current_mood == "tired":
            if any(word in content_lower for word in ["summary", "key points", "brief"]):
                score += 0.3
        
        return min(1.0, score)
    
    def _calculate_difficulty_match(self, learning_dna: LearningDNA, context: Dict[str, Any]) -> float:
        """Calculate difficulty match"""
        content_difficulty = context.get("estimated_difficulty", 0.5)
        preferred_difficulty = learning_dna.difficulty_preference
        
        # Calculate how close the difficulties are
        difficulty_diff = abs(content_difficulty - preferred_difficulty)
        
        # Convert to match score (closer = higher score)
        return 1.0 - difficulty_diff
    
    def _calculate_pace_match(self, learning_dna: LearningDNA, context: Dict[str, Any]) -> float:
        """Calculate pace match"""
        content_pace = context.get("estimated_pace", 0.6)
        preferred_pace = learning_dna.learning_velocity
        
        # Calculate how close the paces are
        pace_diff = abs(content_pace - preferred_pace)
        
        # Convert to match score
        return 1.0 - pace_diff
