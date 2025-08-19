"""
Learning DNA Manager

Extracted from quantum_intelligence_engine.py - manages user learning DNA profiles
and provides advanced learning pattern analysis.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from collections import defaultdict, deque
import json

# Try to import structlog, fall back to standard logging
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    logger = logging.getLogger(__name__)

from ...core.data_structures import LearningDNA
from ...core.enums import LearningStyle, LearningPace, MotivationType
from ...core.exceptions import QuantumEngineError
from ...utils.caching import CacheService


class LearningDNAManager:
    """
    🧬 LEARNING DNA MANAGER
    
    Manages user learning DNA profiles with advanced pattern analysis.
    Extracted from the original quantum engine's personalization logic.
    """
    
    def __init__(self, cache_service: Optional[CacheService] = None):
        self.cache = cache_service
        
        # Learning DNA storage
        self.dna_profiles = {}
        self.learning_history = defaultdict(deque)
        self.pattern_cache = {}
        
        # Analysis parameters
        self.history_window = 50  # Number of recent interactions to analyze
        self.adaptation_rate = 0.1  # How quickly DNA adapts to new patterns
        
        logger.info("Learning DNA Manager initialized")
    
    async def get_learning_dna(self, user_id: str) -> LearningDNA:
        """
        Get user's learning DNA profile
        
        Extracted from original learning DNA retrieval logic
        """
        try:
            # Check cache first
            if self.cache:
                cached_dna = await self.cache.get(f"learning_dna:{user_id}")
                if cached_dna:
                    return LearningDNA.from_dict(cached_dna)
            
            # Check in-memory storage
            if user_id in self.dna_profiles:
                return self.dna_profiles[user_id]
            
            # Create new learning DNA profile
            learning_dna = await self._create_initial_learning_dna(user_id)
            
            # Store in cache and memory
            self.dna_profiles[user_id] = learning_dna
            if self.cache:
                await self.cache.set(f"learning_dna:{user_id}", learning_dna.to_dict(), ttl=7200)
            
            return learning_dna
            
        except Exception as e:
            logger.error(f"Error getting learning DNA for user {user_id}: {e}")
            return self._get_default_learning_dna(user_id)
    
    async def update_learning_dna(
        self, 
        user_id: str, 
        interaction_data: Dict[str, Any]
    ) -> LearningDNA:
        """
        Update learning DNA based on new interaction data
        
        Extracted from original DNA adaptation logic
        """
        try:
            # Get current DNA
            current_dna = await self.get_learning_dna(user_id)
            
            # Add interaction to history
            self.learning_history[user_id].append({
                **interaction_data,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Keep only recent history
            if len(self.learning_history[user_id]) > self.history_window:
                self.learning_history[user_id].popleft()
            
            # Analyze patterns and update DNA
            updated_dna = await self._analyze_and_update_dna(
                current_dna, 
                list(self.learning_history[user_id])
            )
            
            # Store updated DNA
            self.dna_profiles[user_id] = updated_dna
            if self.cache:
                await self.cache.set(f"learning_dna:{user_id}", updated_dna.to_dict(), ttl=7200)
            
            logger.info(f"Updated learning DNA for user {user_id}")
            return updated_dna
            
        except Exception as e:
            logger.error(f"Error updating learning DNA for user {user_id}: {e}")
            return await self.get_learning_dna(user_id)
    
    async def analyze_learning_patterns(
        self, 
        user_id: str
    ) -> Dict[str, Any]:
        """
        Analyze detailed learning patterns for a user
        
        Extracted from original pattern analysis logic
        """
        try:
            learning_dna = await self.get_learning_dna(user_id)
            history = list(self.learning_history[user_id])
            
            if not history:
                return self._get_default_pattern_analysis()
            
            # Analyze various learning patterns
            patterns = {
                "velocity_trends": self._analyze_velocity_trends(history),
                "difficulty_preferences": self._analyze_difficulty_preferences(history),
                "engagement_patterns": self._analyze_engagement_patterns(history),
                "optimal_session_length": self._analyze_session_length_patterns(history),
                "learning_style_indicators": self._analyze_learning_style_indicators(history),
                "motivation_factors": self._analyze_motivation_factors(history),
                "retention_patterns": self._analyze_retention_patterns(history),
                "struggle_indicators": self._analyze_struggle_indicators(history),
                "breakthrough_patterns": self._analyze_breakthrough_patterns(history),
                "metacognitive_development": self._analyze_metacognitive_development(history)
            }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing learning patterns for user {user_id}: {e}")
            return self._get_default_pattern_analysis()
    
    async def predict_learning_outcomes(
        self, 
        user_id: str, 
        proposed_content: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Predict learning outcomes for proposed content
        
        Extracted from original outcome prediction logic
        """
        try:
            learning_dna = await self.get_learning_dna(user_id)
            patterns = await self.analyze_learning_patterns(user_id)
            
            # Predict various outcomes
            predictions = {
                "engagement_probability": self._predict_engagement(learning_dna, proposed_content, patterns),
                "comprehension_probability": self._predict_comprehension(learning_dna, proposed_content, patterns),
                "retention_probability": self._predict_retention(learning_dna, proposed_content, patterns),
                "completion_probability": self._predict_completion(learning_dna, proposed_content, patterns),
                "satisfaction_score": self._predict_satisfaction(learning_dna, proposed_content, patterns),
                "optimal_difficulty": self._predict_optimal_difficulty(learning_dna, patterns),
                "recommended_session_length": self._predict_optimal_session_length(learning_dna, patterns)
            }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting learning outcomes for user {user_id}: {e}")
            return self._get_default_predictions()
    
    async def get_personalization_insights(self, user_id: str) -> Dict[str, Any]:
        """
        Get comprehensive personalization insights
        
        Extracted from original personalization insights logic
        """
        try:
            learning_dna = await self.get_learning_dna(user_id)
            patterns = await self.analyze_learning_patterns(user_id)
            
            insights = {
                "learning_profile": {
                    "primary_learning_style": learning_dna.learning_style,
                    "learning_velocity": learning_dna.learning_velocity,
                    "difficulty_preference": learning_dna.difficulty_preference,
                    "curiosity_index": learning_dna.curiosity_index,
                    "attention_span": learning_dna.attention_span_minutes
                },
                "strengths": self._identify_learning_strengths(learning_dna, patterns),
                "growth_areas": self._identify_growth_areas(learning_dna, patterns),
                "optimal_conditions": self._identify_optimal_conditions(learning_dna, patterns),
                "recommendations": self._generate_learning_recommendations(learning_dna, patterns),
                "adaptation_suggestions": self._generate_adaptation_suggestions(learning_dna, patterns)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting personalization insights for user {user_id}: {e}")
            return self._get_default_insights()
    
    # Private helper methods
    
    async def _create_initial_learning_dna(self, user_id: str) -> LearningDNA:
        """Create initial learning DNA profile for new user"""
        # In production, this might use onboarding data or assessments
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
    
    async def _analyze_and_update_dna(
        self, 
        current_dna: LearningDNA, 
        history: List[Dict[str, Any]]
    ) -> LearningDNA:
        """Analyze interaction history and update DNA"""
        if not history:
            return current_dna
        
        # Calculate new metrics based on recent interactions
        recent_interactions = history[-10:]  # Last 10 interactions
        
        # Update learning velocity
        avg_response_time = sum(
            interaction.get("response_time", 5.0) 
            for interaction in recent_interactions
        ) / len(recent_interactions)
        
        # Faster response times indicate higher velocity
        velocity_adjustment = max(-0.2, min(0.2, (5.0 - avg_response_time) * 0.05))
        new_velocity = current_dna.learning_velocity + (velocity_adjustment * self.adaptation_rate)
        new_velocity = max(0.1, min(1.0, new_velocity))
        
        # Update difficulty preference based on success rates
        success_rate = sum(
            1 for interaction in recent_interactions 
            if interaction.get("success", False)
        ) / len(recent_interactions)
        
        if success_rate > 0.8:  # High success rate, can handle more difficulty
            difficulty_adjustment = 0.1
        elif success_rate < 0.5:  # Low success rate, reduce difficulty
            difficulty_adjustment = -0.1
        else:
            difficulty_adjustment = 0.0
        
        new_difficulty_pref = current_dna.difficulty_preference + (difficulty_adjustment * self.adaptation_rate)
        new_difficulty_pref = max(0.1, min(1.0, new_difficulty_pref))
        
        # Update curiosity index based on question asking behavior
        question_rate = sum(
            1 for interaction in recent_interactions 
            if "?" in interaction.get("user_message", "")
        ) / len(recent_interactions)
        
        curiosity_adjustment = (question_rate - 0.3) * 0.2  # Baseline of 30% questions
        new_curiosity = current_dna.curiosity_index + (curiosity_adjustment * self.adaptation_rate)
        new_curiosity = max(0.1, min(1.0, new_curiosity))
        
        # Create updated DNA
        updated_dna = LearningDNA(
            user_id=current_dna.user_id,
            learning_velocity=new_velocity,
            difficulty_preference=new_difficulty_pref,
            curiosity_index=new_curiosity,
            metacognitive_awareness=current_dna.metacognitive_awareness,  # Updated separately
            concept_retention_rate=current_dna.concept_retention_rate,  # Updated separately
            attention_span_minutes=current_dna.attention_span_minutes,  # Updated separately
            preferred_modalities=current_dna.preferred_modalities,
            learning_style=current_dna.learning_style,
            motivation_factors=current_dna.motivation_factors
        )
        
        return updated_dna
    
    # Pattern analysis methods
    
    def _analyze_velocity_trends(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze learning velocity trends"""
        if len(history) < 5:
            return {"trend": "insufficient_data", "confidence": 0.0}
        
        recent_times = [h.get("response_time", 5.0) for h in history[-10:]]
        older_times = [h.get("response_time", 5.0) for h in history[-20:-10]] if len(history) >= 20 else recent_times
        
        recent_avg = sum(recent_times) / len(recent_times)
        older_avg = sum(older_times) / len(older_times)
        
        if recent_avg < older_avg * 0.8:
            trend = "accelerating"
        elif recent_avg > older_avg * 1.2:
            trend = "decelerating"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "recent_avg_time": recent_avg,
            "older_avg_time": older_avg,
            "confidence": min(1.0, len(history) / 20)
        }
    
    def _analyze_difficulty_preferences(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze difficulty preference patterns"""
        difficulty_success = defaultdict(list)
        
        for interaction in history:
            difficulty = interaction.get("content_difficulty", 0.5)
            success = interaction.get("success", False)
            difficulty_success[round(difficulty, 1)].append(success)
        
        optimal_difficulty = 0.5
        best_success_rate = 0.0
        
        for difficulty, successes in difficulty_success.items():
            success_rate = sum(successes) / len(successes)
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                optimal_difficulty = difficulty
        
        return {
            "optimal_difficulty": optimal_difficulty,
            "best_success_rate": best_success_rate,
            "difficulty_distribution": dict(difficulty_success)
        }
    
    def _analyze_engagement_patterns(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze engagement patterns"""
        engagement_scores = [h.get("engagement_score", 0.5) for h in history]
        
        if not engagement_scores:
            return {"average_engagement": 0.5, "trend": "unknown"}
        
        avg_engagement = sum(engagement_scores) / len(engagement_scores)
        
        # Analyze trend
        if len(engagement_scores) >= 10:
            recent_avg = sum(engagement_scores[-5:]) / 5
            older_avg = sum(engagement_scores[-10:-5]) / 5
            
            if recent_avg > older_avg * 1.1:
                trend = "increasing"
            elif recent_avg < older_avg * 0.9:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "average_engagement": avg_engagement,
            "trend": trend,
            "peak_engagement": max(engagement_scores),
            "low_engagement": min(engagement_scores)
        }
    
    def _analyze_session_length_patterns(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze optimal session length patterns"""
        session_data = defaultdict(list)
        
        for interaction in history:
            session_length = interaction.get("session_length_minutes", 30)
            engagement = interaction.get("engagement_score", 0.5)
            session_data[round(session_length / 10) * 10].append(engagement)  # Group by 10-minute intervals
        
        optimal_length = 30
        best_engagement = 0.0
        
        for length, engagements in session_data.items():
            avg_engagement = sum(engagements) / len(engagements)
            if avg_engagement > best_engagement:
                best_engagement = avg_engagement
                optimal_length = length
        
        return {
            "optimal_session_length": optimal_length,
            "best_engagement_at_length": best_engagement,
            "session_engagement_data": dict(session_data)
        }
    
    def _analyze_learning_style_indicators(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze learning style indicators"""
        style_indicators = {
            "visual": 0,
            "auditory": 0,
            "kinesthetic": 0,
            "reading_writing": 0
        }
        
        for interaction in history:
            content_type = interaction.get("content_type", "text")
            success = interaction.get("success", False)
            
            if success:
                if content_type in ["diagram", "chart", "image"]:
                    style_indicators["visual"] += 1
                elif content_type in ["audio", "speech"]:
                    style_indicators["auditory"] += 1
                elif content_type in ["interactive", "hands_on"]:
                    style_indicators["kinesthetic"] += 1
                else:
                    style_indicators["reading_writing"] += 1
        
        total_successes = sum(style_indicators.values())
        if total_successes > 0:
            style_preferences = {
                style: count / total_successes 
                for style, count in style_indicators.items()
            }
        else:
            style_preferences = {style: 0.25 for style in style_indicators}
        
        return {
            "style_preferences": style_preferences,
            "dominant_style": max(style_preferences, key=style_preferences.get),
            "style_diversity": len([s for s in style_preferences.values() if s > 0.1])
        }
    
    def _analyze_motivation_factors(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze motivation factors"""
        # This would be more sophisticated in production
        motivation_indicators = {
            "achievement": 0,
            "curiosity": 0,
            "social": 0,
            "mastery": 0
        }
        
        for interaction in history:
            user_message = interaction.get("user_message", "").lower()
            
            if any(word in user_message for word in ["goal", "achieve", "complete", "finish"]):
                motivation_indicators["achievement"] += 1
            if any(word in user_message for word in ["why", "how", "what if", "curious"]):
                motivation_indicators["curiosity"] += 1
            if any(word in user_message for word in ["share", "others", "team", "group"]):
                motivation_indicators["social"] += 1
            if any(word in user_message for word in ["master", "expert", "deep", "advanced"]):
                motivation_indicators["mastery"] += 1
        
        total_indicators = sum(motivation_indicators.values())
        if total_indicators > 0:
            motivation_profile = {
                factor: count / total_indicators 
                for factor, count in motivation_indicators.items()
            }
        else:
            motivation_profile = {factor: 0.25 for factor in motivation_indicators}
        
        return {
            "motivation_profile": motivation_profile,
            "primary_motivator": max(motivation_profile, key=motivation_profile.get),
            "motivation_diversity": len([m for m in motivation_profile.values() if m > 0.1])
        }
    
    def _analyze_retention_patterns(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze retention patterns"""
        # Simplified retention analysis
        retention_data = []
        
        for i, interaction in enumerate(history):
            if i > 0:
                # Check if concepts from previous interactions are referenced
                current_content = interaction.get("user_message", "").lower()
                prev_concepts = history[i-1].get("concepts_covered", [])
                
                retention_score = sum(
                    1 for concept in prev_concepts 
                    if concept.lower() in current_content
                ) / max(len(prev_concepts), 1)
                
                retention_data.append(retention_score)
        
        if retention_data:
            avg_retention = sum(retention_data) / len(retention_data)
        else:
            avg_retention = 0.7  # Default assumption
        
        return {
            "average_retention": avg_retention,
            "retention_trend": "stable",  # Would be calculated from trend analysis
            "retention_data_points": len(retention_data)
        }
    
    def _analyze_struggle_indicators(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze struggle indicators"""
        struggle_indicators = []
        
        for interaction in history:
            user_message = interaction.get("user_message", "").lower()
            response_time = interaction.get("response_time", 5.0)
            success = interaction.get("success", True)
            
            struggle_score = 0
            
            # Check for struggle keywords
            if any(word in user_message for word in ["confused", "don't understand", "difficult", "hard"]):
                struggle_score += 0.3
            
            # Check for long response times
            if response_time > 10.0:
                struggle_score += 0.2
            
            # Check for lack of success
            if not success:
                struggle_score += 0.3
            
            struggle_indicators.append(struggle_score)
        
        if struggle_indicators:
            avg_struggle = sum(struggle_indicators) / len(struggle_indicators)
            recent_struggle = sum(struggle_indicators[-5:]) / min(5, len(struggle_indicators))
        else:
            avg_struggle = 0.2
            recent_struggle = 0.2
        
        return {
            "average_struggle_level": avg_struggle,
            "recent_struggle_level": recent_struggle,
            "struggle_trend": "increasing" if recent_struggle > avg_struggle * 1.2 else "stable"
        }
    
    def _analyze_breakthrough_patterns(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze breakthrough patterns"""
        breakthroughs = []
        
        for i, interaction in enumerate(history):
            if i >= 2:  # Need some history to detect breakthroughs
                current_success = interaction.get("success", False)
                prev_successes = [h.get("success", False) for h in history[max(0, i-3):i]]
                
                # Breakthrough: success after struggles
                if current_success and sum(prev_successes) <= 1:
                    breakthroughs.append({
                        "index": i,
                        "context": interaction.get("topic", "unknown"),
                        "difficulty": interaction.get("content_difficulty", 0.5)
                    })
        
        return {
            "breakthrough_count": len(breakthroughs),
            "breakthrough_contexts": [b["context"] for b in breakthroughs],
            "average_breakthrough_difficulty": sum(b["difficulty"] for b in breakthroughs) / max(len(breakthroughs), 1)
        }
    
    def _analyze_metacognitive_development(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze metacognitive development"""
        metacognitive_indicators = []
        
        for interaction in history:
            user_message = interaction.get("user_message", "").lower()
            
            metacognitive_score = 0
            
            # Check for self-reflection indicators
            if any(phrase in user_message for phrase in ["i think", "i believe", "i understand", "i'm confused"]):
                metacognitive_score += 0.2
            
            # Check for strategy mentions
            if any(word in user_message for word in ["strategy", "approach", "method", "way"]):
                metacognitive_score += 0.2
            
            # Check for learning awareness
            if any(phrase in user_message for phrase in ["i learned", "i realize", "i see", "makes sense"]):
                metacognitive_score += 0.3
            
            metacognitive_indicators.append(metacognitive_score)
        
        if metacognitive_indicators:
            avg_metacognitive = sum(metacognitive_indicators) / len(metacognitive_indicators)
            recent_metacognitive = sum(metacognitive_indicators[-5:]) / min(5, len(metacognitive_indicators))
        else:
            avg_metacognitive = 0.3
            recent_metacognitive = 0.3
        
        return {
            "average_metacognitive_awareness": avg_metacognitive,
            "recent_metacognitive_awareness": recent_metacognitive,
            "metacognitive_development": "improving" if recent_metacognitive > avg_metacognitive * 1.1 else "stable"
        }
    
    # Prediction methods
    
    def _predict_engagement(
        self, 
        learning_dna: LearningDNA, 
        content: Dict[str, Any], 
        patterns: Dict[str, Any]
    ) -> float:
        """Predict engagement probability"""
        base_engagement = learning_dna.curiosity_index
        
        # Adjust based on content difficulty match
        content_difficulty = content.get("difficulty", 0.5)
        difficulty_match = 1.0 - abs(content_difficulty - learning_dna.difficulty_preference)
        
        # Adjust based on content type preferences
        content_type = content.get("type", "text")
        type_match = 0.8 if content_type in learning_dna.preferred_modalities else 0.6
        
        engagement_prediction = (base_engagement * 0.4 + difficulty_match * 0.3 + type_match * 0.3)
        
        return min(1.0, max(0.0, engagement_prediction))
    
    def _predict_comprehension(
        self, 
        learning_dna: LearningDNA, 
        content: Dict[str, Any], 
        patterns: Dict[str, Any]
    ) -> float:
        """Predict comprehension probability"""
        base_comprehension = learning_dna.metacognitive_awareness
        
        # Adjust based on content complexity vs. learning velocity
        content_complexity = content.get("complexity", 0.5)
        velocity_match = learning_dna.learning_velocity / max(content_complexity, 0.1)
        velocity_match = min(1.0, velocity_match)
        
        comprehension_prediction = (base_comprehension * 0.6 + velocity_match * 0.4)
        
        return min(1.0, max(0.0, comprehension_prediction))
    
    def _predict_retention(
        self, 
        learning_dna: LearningDNA, 
        content: Dict[str, Any], 
        patterns: Dict[str, Any]
    ) -> float:
        """Predict retention probability"""
        base_retention = learning_dna.concept_retention_rate
        
        # Adjust based on historical retention patterns
        historical_retention = patterns.get("retention_patterns", {}).get("average_retention", 0.7)
        
        retention_prediction = (base_retention * 0.7 + historical_retention * 0.3)
        
        return min(1.0, max(0.0, retention_prediction))
    
    def _predict_completion(
        self, 
        learning_dna: LearningDNA, 
        content: Dict[str, Any], 
        patterns: Dict[str, Any]
    ) -> float:
        """Predict completion probability"""
        # Based on attention span vs. content length
        content_length = content.get("estimated_duration_minutes", 30)
        attention_match = learning_dna.attention_span_minutes / max(content_length, 1)
        attention_match = min(1.0, attention_match)
        
        # Adjust based on engagement prediction
        engagement_pred = self._predict_engagement(learning_dna, content, patterns)
        
        completion_prediction = (attention_match * 0.6 + engagement_pred * 0.4)
        
        return min(1.0, max(0.0, completion_prediction))
    
    def _predict_satisfaction(
        self, 
        learning_dna: LearningDNA, 
        content: Dict[str, Any], 
        patterns: Dict[str, Any]
    ) -> float:
        """Predict satisfaction score"""
        # Combine multiple prediction factors
        engagement_pred = self._predict_engagement(learning_dna, content, patterns)
        comprehension_pred = self._predict_comprehension(learning_dna, content, patterns)
        completion_pred = self._predict_completion(learning_dna, content, patterns)
        
        satisfaction_prediction = (
            engagement_pred * 0.4 + 
            comprehension_pred * 0.4 + 
            completion_pred * 0.2
        )
        
        return min(1.0, max(0.0, satisfaction_prediction))
    
    def _predict_optimal_difficulty(self, learning_dna: LearningDNA, patterns: Dict[str, Any]) -> float:
        """Predict optimal difficulty level"""
        base_difficulty = learning_dna.difficulty_preference
        
        # Adjust based on recent success patterns
        difficulty_patterns = patterns.get("difficulty_preferences", {})
        optimal_from_history = difficulty_patterns.get("optimal_difficulty", base_difficulty)
        
        # Weighted average
        optimal_difficulty = (base_difficulty * 0.6 + optimal_from_history * 0.4)
        
        return min(1.0, max(0.1, optimal_difficulty))
    
    def _predict_optimal_session_length(self, learning_dna: LearningDNA, patterns: Dict[str, Any]) -> int:
        """Predict optimal session length"""
        base_length = learning_dna.attention_span_minutes
        
        # Adjust based on session length patterns
        session_patterns = patterns.get("optimal_session_length", {})
        optimal_from_history = session_patterns.get("optimal_session_length", base_length)
        
        # Weighted average
        optimal_length = int(base_length * 0.7 + optimal_from_history * 0.3)
        
        return max(10, min(120, optimal_length))  # Between 10 and 120 minutes
    
    # Default fallback methods
    
    def _get_default_pattern_analysis(self) -> Dict[str, Any]:
        """Get default pattern analysis for fallback"""
        return {
            "velocity_trends": {"trend": "stable", "confidence": 0.0},
            "difficulty_preferences": {"optimal_difficulty": 0.5, "best_success_rate": 0.7},
            "engagement_patterns": {"average_engagement": 0.7, "trend": "stable"},
            "optimal_session_length": {"optimal_session_length": 30, "best_engagement_at_length": 0.7},
            "learning_style_indicators": {"dominant_style": "balanced", "style_diversity": 4},
            "motivation_factors": {"primary_motivator": "achievement", "motivation_diversity": 2},
            "retention_patterns": {"average_retention": 0.7, "retention_trend": "stable"},
            "struggle_indicators": {"average_struggle_level": 0.2, "recent_struggle_level": 0.2},
            "breakthrough_patterns": {"breakthrough_count": 0, "breakthrough_contexts": []},
            "metacognitive_development": {"average_metacognitive_awareness": 0.5, "metacognitive_development": "stable"}
        }
    
    def _get_default_predictions(self) -> Dict[str, float]:
        """Get default predictions for fallback"""
        return {
            "engagement_probability": 0.7,
            "comprehension_probability": 0.7,
            "retention_probability": 0.7,
            "completion_probability": 0.8,
            "satisfaction_score": 0.7,
            "optimal_difficulty": 0.5,
            "recommended_session_length": 30
        }
    
    def _get_default_insights(self) -> Dict[str, Any]:
        """Get default insights for fallback"""
        return {
            "learning_profile": {
                "primary_learning_style": "balanced",
                "learning_velocity": 0.6,
                "difficulty_preference": 0.5,
                "curiosity_index": 0.7,
                "attention_span": 30
            },
            "strengths": ["Balanced learning approach", "Good curiosity level"],
            "growth_areas": ["Metacognitive awareness", "Retention strategies"],
            "optimal_conditions": ["Moderate difficulty", "30-minute sessions", "Mixed content types"],
            "recommendations": ["Continue with current approach", "Gradually increase difficulty"],
            "adaptation_suggestions": ["Monitor engagement patterns", "Adjust based on performance"]
        }
    
    # Insight generation methods
    
    def _identify_learning_strengths(self, learning_dna: LearningDNA, patterns: Dict[str, Any]) -> List[str]:
        """Identify learning strengths"""
        strengths = []
        
        if learning_dna.learning_velocity > 0.7:
            strengths.append("Fast learning pace")
        
        if learning_dna.curiosity_index > 0.8:
            strengths.append("High curiosity and exploration drive")
        
        if learning_dna.metacognitive_awareness > 0.7:
            strengths.append("Strong self-awareness of learning process")
        
        if learning_dna.concept_retention_rate > 0.8:
            strengths.append("Excellent concept retention")
        
        engagement_patterns = patterns.get("engagement_patterns", {})
        if engagement_patterns.get("average_engagement", 0.5) > 0.8:
            strengths.append("Consistently high engagement")
        
        return strengths if strengths else ["Balanced learning approach"]
    
    def _identify_growth_areas(self, learning_dna: LearningDNA, patterns: Dict[str, Any]) -> List[str]:
        """Identify areas for growth"""
        growth_areas = []
        
        if learning_dna.metacognitive_awareness < 0.5:
            growth_areas.append("Developing metacognitive awareness")
        
        if learning_dna.concept_retention_rate < 0.6:
            growth_areas.append("Improving retention strategies")
        
        struggle_indicators = patterns.get("struggle_indicators", {})
        if struggle_indicators.get("recent_struggle_level", 0.2) > 0.4:
            growth_areas.append("Managing learning challenges")
        
        engagement_patterns = patterns.get("engagement_patterns", {})
        if engagement_patterns.get("trend") == "decreasing":
            growth_areas.append("Maintaining engagement over time")
        
        return growth_areas if growth_areas else ["Continue current development"]
    
    def _identify_optimal_conditions(self, learning_dna: LearningDNA, patterns: Dict[str, Any]) -> List[str]:
        """Identify optimal learning conditions"""
        conditions = []
        
        # Difficulty level
        optimal_difficulty = patterns.get("difficulty_preferences", {}).get("optimal_difficulty", 0.5)
        if optimal_difficulty < 0.4:
            conditions.append("Lower difficulty content")
        elif optimal_difficulty > 0.7:
            conditions.append("Higher difficulty content")
        else:
            conditions.append("Moderate difficulty content")
        
        # Session length
        optimal_length = patterns.get("optimal_session_length", {}).get("optimal_session_length", 30)
        conditions.append(f"{optimal_length}-minute learning sessions")
        
        # Learning style
        style_indicators = patterns.get("learning_style_indicators", {})
        dominant_style = style_indicators.get("dominant_style", "balanced")
        if dominant_style != "balanced":
            conditions.append(f"{dominant_style.title()} learning materials")
        else:
            conditions.append("Mixed content types")
        
        return conditions
    
    def _generate_learning_recommendations(self, learning_dna: LearningDNA, patterns: Dict[str, Any]) -> List[str]:
        """Generate learning recommendations"""
        recommendations = []
        
        # Based on velocity trends
        velocity_trends = patterns.get("velocity_trends", {})
        if velocity_trends.get("trend") == "accelerating":
            recommendations.append("Consider increasing content complexity")
        elif velocity_trends.get("trend") == "decelerating":
            recommendations.append("Take breaks and review fundamentals")
        
        # Based on engagement patterns
        engagement_patterns = patterns.get("engagement_patterns", {})
        if engagement_patterns.get("trend") == "decreasing":
            recommendations.append("Try different content formats to boost engagement")
        
        # Based on struggle indicators
        struggle_indicators = patterns.get("struggle_indicators", {})
        if struggle_indicators.get("recent_struggle_level", 0.2) > 0.4:
            recommendations.append("Focus on foundational concepts before advancing")
        
        return recommendations if recommendations else ["Continue with current learning approach"]
    
    def _generate_adaptation_suggestions(self, learning_dna: LearningDNA, patterns: Dict[str, Any]) -> List[str]:
        """Generate adaptation suggestions"""
        suggestions = []
        
        # Metacognitive development
        metacognitive = patterns.get("metacognitive_development", {})
        if metacognitive.get("metacognitive_development") == "improving":
            suggestions.append("Encourage more self-reflection activities")
        
        # Motivation factors
        motivation = patterns.get("motivation_factors", {})
        primary_motivator = motivation.get("primary_motivator", "achievement")
        
        if primary_motivator == "achievement":
            suggestions.append("Set clear goals and milestones")
        elif primary_motivator == "curiosity":
            suggestions.append("Provide exploratory learning opportunities")
        elif primary_motivator == "social":
            suggestions.append("Include collaborative learning elements")
        elif primary_motivator == "mastery":
            suggestions.append("Offer deep-dive advanced content")
        
        return suggestions if suggestions else ["Monitor learning patterns for optimization opportunities"]
