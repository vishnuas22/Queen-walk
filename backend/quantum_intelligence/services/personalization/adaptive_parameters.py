"""
Adaptive Parameters Engine

Extracted from quantum_intelligence_engine.py - manages adaptive content parameters
for personalized learning experiences.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import math

# Try to import structlog, fall back to standard logging
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    logger = logging.getLogger(__name__)

from ...core.data_structures import AdaptiveContentParameters, LearningDNA
from ...core.enums import DifficultyLevel, ContentType, LearningPace
from ...core.exceptions import QuantumEngineError
from ...utils.caching import CacheService


class AdaptiveParametersEngine:
    """
    ⚙️ ADAPTIVE PARAMETERS ENGINE
    
    Manages adaptive content parameters for personalized learning experiences.
    Extracted from the original quantum engine's adaptive content logic.
    """
    
    def __init__(self, cache_service: Optional[CacheService] = None):
        self.cache = cache_service
        
        # Parameter caches
        self.parameter_cache = {}
        self.adaptation_history = {}
        
        # Adaptation settings
        self.adaptation_rate = 0.15  # How quickly parameters adapt
        self.stability_threshold = 0.05  # Minimum change to trigger adaptation
        
        logger.info("Adaptive Parameters Engine initialized")
    
    async def calculate_adaptive_parameters(
        self, 
        user_id: str, 
        learning_dna: LearningDNA,
        context: Dict[str, Any],
        recent_performance: List[Dict[str, Any]] = None
    ) -> AdaptiveContentParameters:
        """
        Calculate adaptive content parameters based on user profile and context
        
        Extracted from original adaptive parameter calculation logic
        """
        try:
            # Check cache first
            cache_key = f"adaptive_params:{user_id}:{hash(str(context))}"
            if self.cache:
                cached_params = await self.cache.get(cache_key)
                if cached_params:
                    return AdaptiveContentParameters.from_dict(cached_params)
            
            # Calculate base parameters from learning DNA
            base_params = self._calculate_base_parameters(learning_dna)
            
            # Adapt based on context
            context_adapted_params = self._adapt_for_context(base_params, context)
            
            # Adapt based on recent performance
            if recent_performance:
                performance_adapted_params = self._adapt_for_performance(
                    context_adapted_params, 
                    recent_performance
                )
            else:
                performance_adapted_params = context_adapted_params
            
            # Apply temporal adaptations
            final_params = self._apply_temporal_adaptations(
                performance_adapted_params, 
                user_id, 
                context
            )
            
            # Cache the result
            if self.cache:
                await self.cache.set(cache_key, final_params.to_dict(), ttl=1800)
            
            # Store in adaptation history
            self._store_adaptation_history(user_id, final_params, context)
            
            return final_params
            
        except Exception as e:
            logger.error(f"Error calculating adaptive parameters for user {user_id}: {e}")
            return self._get_default_parameters()
    
    async def adapt_parameters_real_time(
        self, 
        user_id: str, 
        current_params: AdaptiveContentParameters,
        feedback: Dict[str, Any]
    ) -> AdaptiveContentParameters:
        """
        Adapt parameters in real-time based on user feedback
        
        Extracted from original real-time adaptation logic
        """
        try:
            # Analyze feedback signals
            adaptation_signals = self._analyze_feedback_signals(feedback)
            
            # Calculate parameter adjustments
            adjustments = self._calculate_parameter_adjustments(
                current_params, 
                adaptation_signals
            )
            
            # Apply adjustments with bounds checking
            adapted_params = self._apply_parameter_adjustments(
                current_params, 
                adjustments
            )
            
            # Validate adapted parameters
            validated_params = self._validate_parameters(adapted_params)
            
            logger.info(f"Real-time parameter adaptation for user {user_id}")
            return validated_params
            
        except Exception as e:
            logger.error(f"Error in real-time parameter adaptation for user {user_id}: {e}")
            return current_params
    
    async def optimize_parameters_for_goal(
        self, 
        user_id: str, 
        learning_dna: LearningDNA,
        learning_goal: str,
        target_metrics: Dict[str, float]
    ) -> AdaptiveContentParameters:
        """
        Optimize parameters for specific learning goals
        
        Extracted from original goal-based optimization logic
        """
        try:
            # Define optimization targets based on goal
            optimization_targets = self._define_optimization_targets(
                learning_goal, 
                target_metrics
            )
            
            # Get base parameters
            base_params = self._calculate_base_parameters(learning_dna)
            
            # Optimize for each target metric
            optimized_params = base_params
            
            for metric, target_value in optimization_targets.items():
                optimized_params = self._optimize_for_metric(
                    optimized_params, 
                    metric, 
                    target_value,
                    learning_dna
                )
            
            # Validate optimized parameters
            validated_params = self._validate_parameters(optimized_params)
            
            logger.info(f"Optimized parameters for goal '{learning_goal}' for user {user_id}")
            return validated_params
            
        except Exception as e:
            logger.error(f"Error optimizing parameters for user {user_id}: {e}")
            return self._calculate_base_parameters(learning_dna)
    
    async def get_parameter_recommendations(
        self, 
        user_id: str, 
        current_params: AdaptiveContentParameters,
        performance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get recommendations for parameter adjustments
        
        Extracted from original parameter recommendation logic
        """
        try:
            recommendations = {
                "adjustments": {},
                "reasoning": {},
                "confidence": {},
                "expected_impact": {}
            }
            
            # Analyze current performance
            performance_analysis = self._analyze_performance_data(performance_data)
            
            # Generate recommendations for each parameter
            param_recommendations = {
                "complexity_level": self._recommend_complexity_adjustment(
                    current_params.complexity_level, 
                    performance_analysis
                ),
                "engagement_level": self._recommend_engagement_adjustment(
                    current_params.engagement_level, 
                    performance_analysis
                ),
                "interactivity_level": self._recommend_interactivity_adjustment(
                    current_params.interactivity_level, 
                    performance_analysis
                ),
                "explanation_depth": self._recommend_explanation_depth_adjustment(
                    current_params.explanation_depth, 
                    performance_analysis
                ),
                "example_density": self._recommend_example_density_adjustment(
                    current_params.example_density, 
                    performance_analysis
                ),
                "challenge_level": self._recommend_challenge_level_adjustment(
                    current_params.challenge_level, 
                    performance_analysis
                )
            }
            
            # Compile recommendations
            for param, rec in param_recommendations.items():
                if rec["adjustment"] != 0:
                    recommendations["adjustments"][param] = rec["adjustment"]
                    recommendations["reasoning"][param] = rec["reasoning"]
                    recommendations["confidence"][param] = rec["confidence"]
                    recommendations["expected_impact"][param] = rec["expected_impact"]
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting parameter recommendations for user {user_id}: {e}")
            return {"adjustments": {}, "reasoning": {}, "confidence": {}, "expected_impact": {}}
    
    # Private helper methods
    
    def _calculate_base_parameters(self, learning_dna: LearningDNA) -> AdaptiveContentParameters:
        """Calculate base parameters from learning DNA"""
        return AdaptiveContentParameters(
            complexity_level=learning_dna.difficulty_preference,
            engagement_level=learning_dna.curiosity_index,
            interactivity_level=self._calculate_interactivity_from_dna(learning_dna),
            explanation_depth=learning_dna.metacognitive_awareness,
            example_density=self._calculate_example_density_from_dna(learning_dna),
            challenge_level=learning_dna.difficulty_preference
        )
    
    def _calculate_interactivity_from_dna(self, learning_dna: LearningDNA) -> float:
        """Calculate interactivity level from learning DNA"""
        # Higher interactivity for shorter attention spans and kinesthetic learners
        base_interactivity = 0.6
        
        # Adjust for attention span
        if learning_dna.attention_span_minutes < 25:
            base_interactivity += 0.2
        elif learning_dna.attention_span_minutes > 45:
            base_interactivity -= 0.1
        
        # Adjust for learning modalities
        if "kinesthetic" in learning_dna.preferred_modalities:
            base_interactivity += 0.2
        
        return min(1.0, max(0.0, base_interactivity))
    
    def _calculate_example_density_from_dna(self, learning_dna: LearningDNA) -> float:
        """Calculate example density from learning DNA"""
        # More examples for visual learners and lower difficulty preference
        base_density = 0.5
        
        if "visual" in learning_dna.preferred_modalities:
            base_density += 0.2
        
        if learning_dna.difficulty_preference < 0.5:
            base_density += 0.1
        
        if learning_dna.learning_velocity < 0.5:
            base_density += 0.1  # More examples for slower learners
        
        return min(1.0, max(0.0, base_density))
    
    def _adapt_for_context(
        self, 
        base_params: AdaptiveContentParameters, 
        context: Dict[str, Any]
    ) -> AdaptiveContentParameters:
        """Adapt parameters based on context"""
        adapted_params = AdaptiveContentParameters(
            complexity_level=base_params.complexity_level,
            engagement_level=base_params.engagement_level,
            interactivity_level=base_params.interactivity_level,
            explanation_depth=base_params.explanation_depth,
            example_density=base_params.example_density,
            challenge_level=base_params.challenge_level
        )
        
        # Adapt based on topic difficulty
        topic_difficulty = context.get("topic_difficulty", 0.5)
        if topic_difficulty > 0.7:
            adapted_params.complexity_level *= 0.8  # Reduce complexity for difficult topics
            adapted_params.explanation_depth += 0.1  # More explanation for difficult topics
            adapted_params.example_density += 0.1  # More examples for difficult topics
        
        # Adapt based on session context
        session_length = context.get("session_length_minutes", 30)
        if session_length < 20:
            adapted_params.interactivity_level += 0.1  # More interactivity for short sessions
        
        # Adapt based on time of day
        hour = context.get("hour_of_day", 12)
        if hour < 9 or hour > 20:  # Early morning or late evening
            adapted_params.complexity_level *= 0.9  # Slightly reduce complexity
            adapted_params.engagement_level += 0.05  # Slightly increase engagement
        
        # Adapt based on device type
        device_type = context.get("device_type", "desktop")
        if device_type == "mobile":
            adapted_params.interactivity_level += 0.1  # More interactivity on mobile
            adapted_params.explanation_depth -= 0.05  # Shorter explanations on mobile
        
        return self._validate_parameters(adapted_params)
    
    def _adapt_for_performance(
        self, 
        base_params: AdaptiveContentParameters, 
        performance_data: List[Dict[str, Any]]
    ) -> AdaptiveContentParameters:
        """Adapt parameters based on recent performance"""
        if not performance_data:
            return base_params
        
        # Analyze recent performance
        recent_success_rate = sum(
            1 for p in performance_data[-5:] if p.get("success", False)
        ) / min(5, len(performance_data))
        
        recent_engagement = sum(
            p.get("engagement_score", 0.5) for p in performance_data[-5:]
        ) / min(5, len(performance_data))
        
        recent_completion_rate = sum(
            1 for p in performance_data[-5:] if p.get("completed", False)
        ) / min(5, len(performance_data))
        
        # Calculate adaptations
        adapted_params = AdaptiveContentParameters(
            complexity_level=base_params.complexity_level,
            engagement_level=base_params.engagement_level,
            interactivity_level=base_params.interactivity_level,
            explanation_depth=base_params.explanation_depth,
            example_density=base_params.example_density,
            challenge_level=base_params.challenge_level
        )
        
        # Adapt based on success rate
        if recent_success_rate < 0.5:
            adapted_params.complexity_level *= 0.9  # Reduce complexity
            adapted_params.explanation_depth += 0.1  # More explanation
            adapted_params.example_density += 0.1  # More examples
        elif recent_success_rate > 0.8:
            adapted_params.complexity_level *= 1.1  # Increase complexity
            adapted_params.challenge_level += 0.05  # More challenge
        
        # Adapt based on engagement
        if recent_engagement < 0.6:
            adapted_params.engagement_level += 0.1  # Increase engagement
            adapted_params.interactivity_level += 0.1  # More interactivity
        
        # Adapt based on completion rate
        if recent_completion_rate < 0.7:
            adapted_params.interactivity_level += 0.05  # More interactivity
            adapted_params.explanation_depth -= 0.05  # Shorter explanations
        
        return self._validate_parameters(adapted_params)
    
    def _apply_temporal_adaptations(
        self, 
        base_params: AdaptiveContentParameters, 
        user_id: str, 
        context: Dict[str, Any]
    ) -> AdaptiveContentParameters:
        """Apply temporal adaptations based on time patterns"""
        # Get historical adaptation data
        history = self.adaptation_history.get(user_id, [])
        
        if len(history) < 3:
            return base_params  # Not enough history for temporal adaptation
        
        # Analyze temporal patterns
        current_hour = context.get("hour_of_day", 12)
        current_day = context.get("day_of_week", 1)  # 1=Monday
        
        # Find similar time periods in history
        similar_periods = [
            h for h in history 
            if abs(h.get("hour_of_day", 12) - current_hour) <= 2
        ]
        
        if not similar_periods:
            return base_params
        
        # Calculate average performance for similar periods
        avg_performance = sum(
            p.get("performance_score", 0.7) for p in similar_periods
        ) / len(similar_periods)
        
        # Adapt based on historical performance at this time
        adapted_params = AdaptiveContentParameters(
            complexity_level=base_params.complexity_level,
            engagement_level=base_params.engagement_level,
            interactivity_level=base_params.interactivity_level,
            explanation_depth=base_params.explanation_depth,
            example_density=base_params.example_density,
            challenge_level=base_params.challenge_level
        )
        
        if avg_performance < 0.6:
            # Poor performance at this time - make content easier
            adapted_params.complexity_level *= 0.95
            adapted_params.engagement_level += 0.05
        elif avg_performance > 0.8:
            # Good performance at this time - can handle more challenge
            adapted_params.complexity_level *= 1.05
            adapted_params.challenge_level += 0.05
        
        return self._validate_parameters(adapted_params)
    
    def _store_adaptation_history(
        self, 
        user_id: str, 
        params: AdaptiveContentParameters, 
        context: Dict[str, Any]
    ):
        """Store adaptation history for temporal analysis"""
        if user_id not in self.adaptation_history:
            self.adaptation_history[user_id] = []
        
        history_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "parameters": params.to_dict(),
            "context": context,
            "hour_of_day": context.get("hour_of_day", 12),
            "day_of_week": context.get("day_of_week", 1)
        }
        
        self.adaptation_history[user_id].append(history_entry)
        
        # Keep only recent history (last 100 entries)
        if len(self.adaptation_history[user_id]) > 100:
            self.adaptation_history[user_id] = self.adaptation_history[user_id][-100:]
    
    def _analyze_feedback_signals(self, feedback: Dict[str, Any]) -> Dict[str, float]:
        """Analyze feedback signals for real-time adaptation"""
        signals = {
            "complexity_signal": 0.0,
            "engagement_signal": 0.0,
            "interactivity_signal": 0.0,
            "explanation_signal": 0.0,
            "example_signal": 0.0,
            "challenge_signal": 0.0
        }
        
        # Analyze explicit feedback
        if "too_difficult" in feedback:
            signals["complexity_signal"] = -0.2
            signals["explanation_signal"] = 0.1
            signals["example_signal"] = 0.1
        
        if "too_easy" in feedback:
            signals["complexity_signal"] = 0.2
            signals["challenge_signal"] = 0.1
        
        if "boring" in feedback:
            signals["engagement_signal"] = 0.2
            signals["interactivity_signal"] = 0.1
        
        if "confusing" in feedback:
            signals["explanation_signal"] = 0.2
            signals["example_signal"] = 0.1
        
        # Analyze implicit feedback
        response_time = feedback.get("response_time", 5.0)
        if response_time > 10.0:
            signals["complexity_signal"] = -0.1  # Taking too long, reduce complexity
        elif response_time < 2.0:
            signals["complexity_signal"] = 0.1  # Too fast, increase complexity
        
        engagement_score = feedback.get("engagement_score", 0.5)
        if engagement_score < 0.5:
            signals["engagement_signal"] = 0.1
            signals["interactivity_signal"] = 0.1
        
        return signals
    
    def _calculate_parameter_adjustments(
        self, 
        current_params: AdaptiveContentParameters, 
        signals: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate parameter adjustments based on signals"""
        adjustments = {}
        
        # Apply adaptation rate to signals
        for param, signal in signals.items():
            param_name = param.replace("_signal", "")
            if abs(signal) > self.stability_threshold:
                adjustments[param_name] = signal * self.adaptation_rate
        
        return adjustments
    
    def _apply_parameter_adjustments(
        self, 
        current_params: AdaptiveContentParameters, 
        adjustments: Dict[str, float]
    ) -> AdaptiveContentParameters:
        """Apply parameter adjustments with bounds checking"""
        adjusted_params = AdaptiveContentParameters(
            complexity_level=current_params.complexity_level + adjustments.get("complexity", 0),
            engagement_level=current_params.engagement_level + adjustments.get("engagement", 0),
            interactivity_level=current_params.interactivity_level + adjustments.get("interactivity", 0),
            explanation_depth=current_params.explanation_depth + adjustments.get("explanation", 0),
            example_density=current_params.example_density + adjustments.get("example", 0),
            challenge_level=current_params.challenge_level + adjustments.get("challenge", 0)
        )
        
        return self._validate_parameters(adjusted_params)
    
    def _validate_parameters(self, params: AdaptiveContentParameters) -> AdaptiveContentParameters:
        """Validate and bound parameters to valid ranges"""
        return AdaptiveContentParameters(
            complexity_level=max(0.0, min(1.0, params.complexity_level)),
            engagement_level=max(0.0, min(1.0, params.engagement_level)),
            interactivity_level=max(0.0, min(1.0, params.interactivity_level)),
            explanation_depth=max(0.0, min(1.0, params.explanation_depth)),
            example_density=max(0.0, min(1.0, params.example_density)),
            challenge_level=max(0.0, min(1.0, params.challenge_level))
        )
    
    def _get_default_parameters(self) -> AdaptiveContentParameters:
        """Get default parameters for fallback"""
        return AdaptiveContentParameters(
            complexity_level=0.5,
            engagement_level=0.7,
            interactivity_level=0.6,
            explanation_depth=0.5,
            example_density=0.4,
            challenge_level=0.5
        )
    
    # Goal optimization methods
    
    def _define_optimization_targets(
        self, 
        learning_goal: str, 
        target_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Define optimization targets based on learning goal"""
        goal_targets = {
            "comprehension": {"explanation_depth": 0.8, "example_density": 0.7},
            "engagement": {"engagement_level": 0.9, "interactivity_level": 0.8},
            "challenge": {"challenge_level": 0.8, "complexity_level": 0.7},
            "retention": {"explanation_depth": 0.7, "example_density": 0.6},
            "speed": {"complexity_level": 0.6, "explanation_depth": 0.4}
        }
        
        # Merge goal-based targets with custom targets
        targets = goal_targets.get(learning_goal.lower(), {})
        targets.update(target_metrics)
        
        return targets
    
    def _optimize_for_metric(
        self, 
        params: AdaptiveContentParameters, 
        metric: str, 
        target_value: float,
        learning_dna: LearningDNA
    ) -> AdaptiveContentParameters:
        """Optimize parameters for a specific metric"""
        optimized_params = AdaptiveContentParameters(
            complexity_level=params.complexity_level,
            engagement_level=params.engagement_level,
            interactivity_level=params.interactivity_level,
            explanation_depth=params.explanation_depth,
            example_density=params.example_density,
            challenge_level=params.challenge_level
        )
        
        # Apply metric-specific optimizations
        if metric == "complexity_level":
            optimized_params.complexity_level = target_value
        elif metric == "engagement_level":
            optimized_params.engagement_level = target_value
        elif metric == "interactivity_level":
            optimized_params.interactivity_level = target_value
        elif metric == "explanation_depth":
            optimized_params.explanation_depth = target_value
        elif metric == "example_density":
            optimized_params.example_density = target_value
        elif metric == "challenge_level":
            optimized_params.challenge_level = target_value
        
        return self._validate_parameters(optimized_params)
    
    # Performance analysis methods
    
    def _analyze_performance_data(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance data for recommendations"""
        return {
            "success_rate": performance_data.get("success_rate", 0.7),
            "engagement_score": performance_data.get("engagement_score", 0.7),
            "completion_rate": performance_data.get("completion_rate", 0.8),
            "response_time": performance_data.get("avg_response_time", 5.0),
            "struggle_indicators": performance_data.get("struggle_indicators", 0.2),
            "satisfaction_score": performance_data.get("satisfaction_score", 0.7)
        }
    
    # Recommendation methods
    
    def _recommend_complexity_adjustment(
        self, 
        current_level: float, 
        performance: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recommend complexity level adjustment"""
        success_rate = performance.get("success_rate", 0.7)
        struggle_indicators = performance.get("struggle_indicators", 0.2)
        
        if success_rate < 0.5 or struggle_indicators > 0.4:
            adjustment = -0.1
            reasoning = "Reduce complexity due to low success rate or high struggle indicators"
            confidence = 0.8
            expected_impact = "Improved comprehension and reduced frustration"
        elif success_rate > 0.85 and struggle_indicators < 0.1:
            adjustment = 0.1
            reasoning = "Increase complexity due to high success rate and low struggle"
            confidence = 0.7
            expected_impact = "Maintained engagement with appropriate challenge"
        else:
            adjustment = 0.0
            reasoning = "Current complexity level appears appropriate"
            confidence = 0.6
            expected_impact = "Stable performance"
        
        return {
            "adjustment": adjustment,
            "reasoning": reasoning,
            "confidence": confidence,
            "expected_impact": expected_impact
        }
    
    def _recommend_engagement_adjustment(
        self, 
        current_level: float, 
        performance: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recommend engagement level adjustment"""
        engagement_score = performance.get("engagement_score", 0.7)
        completion_rate = performance.get("completion_rate", 0.8)
        
        if engagement_score < 0.6 or completion_rate < 0.7:
            adjustment = 0.1
            reasoning = "Increase engagement due to low engagement score or completion rate"
            confidence = 0.8
            expected_impact = "Higher motivation and session completion"
        elif engagement_score > 0.9:
            adjustment = 0.0
            reasoning = "Engagement level is already very high"
            confidence = 0.9
            expected_impact = "Maintain current high engagement"
        else:
            adjustment = 0.0
            reasoning = "Current engagement level appears appropriate"
            confidence = 0.6
            expected_impact = "Stable engagement"
        
        return {
            "adjustment": adjustment,
            "reasoning": reasoning,
            "confidence": confidence,
            "expected_impact": expected_impact
        }
    
    def _recommend_interactivity_adjustment(
        self, 
        current_level: float, 
        performance: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recommend interactivity level adjustment"""
        engagement_score = performance.get("engagement_score", 0.7)
        response_time = performance.get("response_time", 5.0)
        
        if engagement_score < 0.6 and response_time > 8.0:
            adjustment = 0.1
            reasoning = "Increase interactivity to boost engagement and reduce response time"
            confidence = 0.7
            expected_impact = "More engaging and responsive learning experience"
        elif response_time < 3.0 and engagement_score > 0.8:
            adjustment = -0.05
            reasoning = "Slightly reduce interactivity as user is highly engaged and responsive"
            confidence = 0.5
            expected_impact = "Streamlined experience without overwhelming interactions"
        else:
            adjustment = 0.0
            reasoning = "Current interactivity level appears appropriate"
            confidence = 0.6
            expected_impact = "Stable interaction patterns"
        
        return {
            "adjustment": adjustment,
            "reasoning": reasoning,
            "confidence": confidence,
            "expected_impact": expected_impact
        }
    
    def _recommend_explanation_depth_adjustment(
        self, 
        current_level: float, 
        performance: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recommend explanation depth adjustment"""
        success_rate = performance.get("success_rate", 0.7)
        struggle_indicators = performance.get("struggle_indicators", 0.2)
        
        if success_rate < 0.6 or struggle_indicators > 0.3:
            adjustment = 0.1
            reasoning = "Increase explanation depth due to comprehension difficulties"
            confidence = 0.8
            expected_impact = "Better understanding and reduced confusion"
        elif success_rate > 0.9 and struggle_indicators < 0.1:
            adjustment = -0.05
            reasoning = "Reduce explanation depth as user demonstrates strong comprehension"
            confidence = 0.6
            expected_impact = "More concise content without sacrificing understanding"
        else:
            adjustment = 0.0
            reasoning = "Current explanation depth appears appropriate"
            confidence = 0.6
            expected_impact = "Stable comprehension levels"
        
        return {
            "adjustment": adjustment,
            "reasoning": reasoning,
            "confidence": confidence,
            "expected_impact": expected_impact
        }
    
    def _recommend_example_density_adjustment(
        self, 
        current_level: float, 
        performance: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recommend example density adjustment"""
        success_rate = performance.get("success_rate", 0.7)
        satisfaction_score = performance.get("satisfaction_score", 0.7)
        
        if success_rate < 0.6:
            adjustment = 0.1
            reasoning = "Increase example density to improve understanding"
            confidence = 0.7
            expected_impact = "Better concept comprehension through concrete examples"
        elif success_rate > 0.9 and satisfaction_score > 0.8:
            adjustment = -0.05
            reasoning = "Reduce example density as user demonstrates strong understanding"
            confidence = 0.5
            expected_impact = "More efficient learning without redundant examples"
        else:
            adjustment = 0.0
            reasoning = "Current example density appears appropriate"
            confidence = 0.6
            expected_impact = "Stable learning support"
        
        return {
            "adjustment": adjustment,
            "reasoning": reasoning,
            "confidence": confidence,
            "expected_impact": expected_impact
        }
    
    def _recommend_challenge_level_adjustment(
        self, 
        current_level: float, 
        performance: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recommend challenge level adjustment"""
        success_rate = performance.get("success_rate", 0.7)
        engagement_score = performance.get("engagement_score", 0.7)
        
        if success_rate > 0.85 and engagement_score < 0.7:
            adjustment = 0.1
            reasoning = "Increase challenge level to boost engagement while maintaining success"
            confidence = 0.7
            expected_impact = "Higher engagement through appropriate challenge"
        elif success_rate < 0.5:
            adjustment = -0.1
            reasoning = "Reduce challenge level due to low success rate"
            confidence = 0.8
            expected_impact = "Improved success rate and confidence"
        else:
            adjustment = 0.0
            reasoning = "Current challenge level appears appropriate"
            confidence = 0.6
            expected_impact = "Balanced challenge and success"
        
        return {
            "adjustment": adjustment,
            "reasoning": reasoning,
            "confidence": confidence,
            "expected_impact": expected_impact
        }
