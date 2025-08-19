"""
Performance Prediction Engine

Extracted from quantum_intelligence_engine.py - advanced performance prediction
and learning outcome forecasting using machine learning techniques.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import statistics
import math

# Try to import structlog, fall back to standard logging
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    logger = logging.getLogger(__name__)

from ...core.data_structures import LearningDNA
from ...core.enums import LearningStyle, DifficultyLevel
from ...core.exceptions import QuantumEngineError
from ...utils.caching import CacheService


class PerformancePredictionEngine:
    """
    🔮 PERFORMANCE PREDICTION ENGINE
    
    Advanced performance prediction and learning outcome forecasting.
    Extracted from the original quantum engine's predictive analytics logic.
    """
    
    def __init__(self, cache_service: Optional[CacheService] = None):
        self.cache = cache_service
        
        # Prediction models and data
        self.user_models = {}
        self.prediction_history = {}
        self.performance_baselines = {}
        
        # Prediction parameters
        self.prediction_window = 7  # Days to predict ahead
        self.model_update_threshold = 10  # Minimum interactions to update model
        self.confidence_decay_rate = 0.1  # How confidence decreases over time
        
        logger.info("Performance Prediction Engine initialized")
    
    async def predict_learning_performance(
        self, 
        user_id: str, 
        learning_dna: LearningDNA,
        upcoming_content: List[Dict[str, Any]],
        prediction_horizon: int = 7
    ) -> Dict[str, Any]:
        """
        Predict learning performance for upcoming content
        
        Extracted from original performance prediction logic
        """
        try:
            # Get or create user prediction model
            user_model = await self._get_user_prediction_model(user_id, learning_dna)
            
            # Predict performance for each content item
            content_predictions = []
            for i, content in enumerate(upcoming_content):
                prediction = await self._predict_content_performance(
                    user_id, 
                    content, 
                    user_model,
                    i
                )
                content_predictions.append(prediction)
            
            # Calculate aggregate predictions
            aggregate_predictions = self._calculate_aggregate_predictions(content_predictions)
            
            # Generate performance insights
            insights = await self._generate_performance_insights(
                user_id, 
                content_predictions, 
                aggregate_predictions
            )
            
            # Calculate prediction confidence
            confidence = self._calculate_prediction_confidence(
                user_model, 
                len(upcoming_content),
                prediction_horizon
            )
            
            return {
                "user_id": user_id,
                "prediction_horizon_days": prediction_horizon,
                "content_count": len(upcoming_content),
                "content_predictions": content_predictions,
                "aggregate_predictions": aggregate_predictions,
                "insights": insights,
                "confidence": confidence,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error predicting performance for user {user_id}: {e}")
            return self._get_default_performance_prediction()
    
    async def predict_engagement_levels(
        self, 
        user_id: str, 
        session_plan: Dict[str, Any],
        historical_engagement: List[float]
    ) -> Dict[str, Any]:
        """
        Predict engagement levels for planned learning session
        
        Extracted from original engagement prediction logic
        """
        try:
            # Analyze historical engagement patterns
            engagement_patterns = self._analyze_engagement_patterns(historical_engagement)
            
            # Predict engagement based on session characteristics
            session_engagement = await self._predict_session_engagement(
                user_id, 
                session_plan, 
                engagement_patterns
            )
            
            # Identify engagement risk factors
            risk_factors = self._identify_engagement_risks(session_plan, engagement_patterns)
            
            # Generate engagement optimization suggestions
            optimizations = await self._generate_engagement_optimizations(
                session_plan, 
                session_engagement, 
                risk_factors
            )
            
            return {
                "user_id": user_id,
                "session_plan": session_plan,
                "predicted_engagement": session_engagement,
                "engagement_patterns": engagement_patterns,
                "risk_factors": risk_factors,
                "optimizations": optimizations,
                "confidence": engagement_patterns.get("confidence", 0.6)
            }
            
        except Exception as e:
            logger.error(f"Error predicting engagement for user {user_id}: {e}")
            return self._get_default_engagement_prediction()
    
    async def predict_completion_probability(
        self, 
        user_id: str, 
        learning_path: List[Dict[str, Any]],
        user_constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predict probability of completing learning path
        
        Extracted from original completion prediction logic
        """
        try:
            # Analyze path characteristics
            path_analysis = self._analyze_learning_path(learning_path)
            
            # Get user completion history
            completion_history = await self._get_completion_history(user_id)
            
            # Predict completion for each milestone
            milestone_predictions = []
            cumulative_probability = 1.0
            
            for i, milestone in enumerate(learning_path):
                milestone_prob = await self._predict_milestone_completion(
                    user_id, 
                    milestone, 
                    completion_history,
                    cumulative_probability,
                    user_constraints
                )
                milestone_predictions.append(milestone_prob)
                cumulative_probability *= milestone_prob["completion_probability"]
            
            # Calculate overall completion metrics
            overall_completion = self._calculate_overall_completion_metrics(
                milestone_predictions, 
                path_analysis
            )
            
            # Identify completion barriers
            barriers = self._identify_completion_barriers(
                milestone_predictions, 
                user_constraints
            )
            
            # Generate completion strategies
            strategies = await self._generate_completion_strategies(
                barriers, 
                overall_completion
            )
            
            return {
                "user_id": user_id,
                "learning_path_length": len(learning_path),
                "milestone_predictions": milestone_predictions,
                "overall_completion": overall_completion,
                "completion_barriers": barriers,
                "completion_strategies": strategies,
                "path_analysis": path_analysis
            }
            
        except Exception as e:
            logger.error(f"Error predicting completion for user {user_id}: {e}")
            return self._get_default_completion_prediction()
    
    async def predict_learning_velocity(
        self, 
        user_id: str, 
        content_difficulty: float,
        learning_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predict learning velocity for given content and context
        
        Extracted from original velocity prediction logic
        """
        try:
            # Get user velocity model
            velocity_model = await self._get_velocity_model(user_id)
            
            # Predict base velocity
            base_velocity = self._predict_base_velocity(
                velocity_model, 
                content_difficulty
            )
            
            # Apply contextual adjustments
            adjusted_velocity = self._apply_velocity_adjustments(
                base_velocity, 
                learning_context
            )
            
            # Predict velocity confidence intervals
            confidence_intervals = self._calculate_velocity_confidence_intervals(
                adjusted_velocity, 
                velocity_model
            )
            
            # Generate velocity insights
            velocity_insights = self._generate_velocity_insights(
                adjusted_velocity, 
                content_difficulty, 
                learning_context
            )
            
            return {
                "user_id": user_id,
                "content_difficulty": content_difficulty,
                "predicted_velocity": adjusted_velocity,
                "confidence_intervals": confidence_intervals,
                "velocity_insights": velocity_insights,
                "context_factors": learning_context,
                "model_confidence": velocity_model.get("confidence", 0.6)
            }
            
        except Exception as e:
            logger.error(f"Error predicting velocity for user {user_id}: {e}")
            return self._get_default_velocity_prediction()
    
    async def update_prediction_models(
        self, 
        user_id: str, 
        actual_performance: Dict[str, Any],
        predicted_performance: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update prediction models based on actual vs predicted performance
        
        Extracted from original model update logic
        """
        try:
            # Calculate prediction accuracy
            accuracy_metrics = self._calculate_prediction_accuracy(
                actual_performance, 
                predicted_performance
            )
            
            # Update user prediction model
            updated_model = await self._update_user_model(
                user_id, 
                actual_performance, 
                accuracy_metrics
            )
            
            # Store prediction history
            self._store_prediction_history(
                user_id, 
                predicted_performance, 
                actual_performance, 
                accuracy_metrics
            )
            
            # Generate model improvement insights
            improvement_insights = self._generate_model_insights(
                accuracy_metrics, 
                updated_model
            )
            
            return {
                "user_id": user_id,
                "accuracy_metrics": accuracy_metrics,
                "model_updated": True,
                "improvement_insights": improvement_insights,
                "model_confidence": updated_model.get("confidence", 0.6),
                "update_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating prediction models for user {user_id}: {e}")
            return {"model_updated": False, "error": str(e)}
    
    # Private prediction methods
    
    async def _get_user_prediction_model(self, user_id: str, learning_dna: LearningDNA) -> Dict[str, Any]:
        """Get or create user prediction model"""
        if user_id not in self.user_models:
            # Create new model based on learning DNA
            self.user_models[user_id] = {
                "base_performance": learning_dna.concept_retention_rate,
                "learning_velocity": learning_dna.learning_velocity,
                "difficulty_tolerance": learning_dna.difficulty_preference,
                "engagement_baseline": learning_dna.curiosity_index,
                "consistency_factor": 0.7,  # Default
                "confidence": 0.5,  # Low confidence for new model
                "last_updated": datetime.utcnow().isoformat(),
                "interaction_count": 0
            }
        
        return self.user_models[user_id]
    
    async def _predict_content_performance(
        self, 
        user_id: str, 
        content: Dict[str, Any], 
        user_model: Dict[str, Any],
        content_index: int
    ) -> Dict[str, Any]:
        """Predict performance for specific content"""
        # Extract content characteristics
        difficulty = content.get("difficulty", 0.5)
        estimated_duration = content.get("estimated_duration", 30)
        content_type = content.get("type", "text")
        
        # Base prediction from user model
        base_performance = user_model["base_performance"]
        
        # Adjust for difficulty
        difficulty_factor = self._calculate_difficulty_factor(
            difficulty, 
            user_model["difficulty_tolerance"]
        )
        
        # Adjust for content type
        type_factor = self._calculate_content_type_factor(content_type, user_model)
        
        # Adjust for fatigue (later content in sequence)
        fatigue_factor = max(0.7, 1.0 - (content_index * 0.05))
        
        # Calculate final prediction
        predicted_performance = base_performance * difficulty_factor * type_factor * fatigue_factor
        
        # Predict other metrics
        predicted_engagement = self._predict_content_engagement(content, user_model)
        predicted_completion_time = self._predict_completion_time(
            estimated_duration, 
            user_model["learning_velocity"]
        )
        
        return {
            "content_index": content_index,
            "content_id": content.get("id", f"content_{content_index}"),
            "predicted_performance": min(1.0, max(0.0, predicted_performance)),
            "predicted_engagement": predicted_engagement,
            "predicted_completion_time": predicted_completion_time,
            "difficulty": difficulty,
            "confidence": user_model["confidence"] * 0.9 ** content_index  # Decreasing confidence
        }
    
    def _calculate_difficulty_factor(self, content_difficulty: float, user_tolerance: float) -> float:
        """Calculate how difficulty affects performance prediction"""
        difficulty_gap = abs(content_difficulty - user_tolerance)
        
        if difficulty_gap < 0.1:
            return 1.0  # Perfect match
        elif difficulty_gap < 0.3:
            return 0.9  # Good match
        elif difficulty_gap < 0.5:
            return 0.7  # Moderate mismatch
        else:
            return 0.5  # Poor match
    
    def _calculate_content_type_factor(self, content_type: str, user_model: Dict[str, Any]) -> float:
        """Calculate how content type affects performance"""
        # Simplified content type preferences
        type_preferences = {
            "text": 0.8,
            "visual": 0.9,
            "interactive": 0.85,
            "video": 0.75,
            "audio": 0.7
        }
        
        return type_preferences.get(content_type, 0.8)
    
    def _predict_content_engagement(self, content: Dict[str, Any], user_model: Dict[str, Any]) -> float:
        """Predict engagement for specific content"""
        base_engagement = user_model["engagement_baseline"]
        
        # Adjust based on content characteristics
        content_type = content.get("type", "text")
        if content_type in ["interactive", "visual"]:
            base_engagement += 0.1
        
        # Adjust based on difficulty match
        difficulty = content.get("difficulty", 0.5)
        difficulty_factor = self._calculate_difficulty_factor(
            difficulty, 
            user_model["difficulty_tolerance"]
        )
        
        return min(1.0, max(0.0, base_engagement * difficulty_factor))
    
    def _predict_completion_time(self, estimated_duration: int, learning_velocity: float) -> int:
        """Predict actual completion time"""
        # Adjust estimated duration based on learning velocity
        velocity_factor = 1.0 / max(0.1, learning_velocity)
        predicted_time = estimated_duration * velocity_factor
        
        return max(5, int(predicted_time))  # Minimum 5 minutes
    
    def _calculate_aggregate_predictions(self, content_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate predictions from individual content predictions"""
        if not content_predictions:
            return {"overall_performance": 0.5, "overall_engagement": 0.5}
        
        performances = [pred["predicted_performance"] for pred in content_predictions]
        engagements = [pred["predicted_engagement"] for pred in content_predictions]
        completion_times = [pred["predicted_completion_time"] for pred in content_predictions]
        confidences = [pred["confidence"] for pred in content_predictions]
        
        return {
            "overall_performance": statistics.mean(performances),
            "overall_engagement": statistics.mean(engagements),
            "total_completion_time": sum(completion_times),
            "performance_consistency": 1.0 - statistics.stdev(performances) if len(performances) > 1 else 1.0,
            "engagement_consistency": 1.0 - statistics.stdev(engagements) if len(engagements) > 1 else 1.0,
            "average_confidence": statistics.mean(confidences),
            "success_probability": sum(1 for p in performances if p > 0.7) / len(performances)
        }
    
    async def _generate_performance_insights(
        self, 
        user_id: str, 
        content_predictions: List[Dict[str, Any]], 
        aggregate_predictions: Dict[str, Any]
    ) -> List[str]:
        """Generate insights from performance predictions"""
        insights = []
        
        overall_performance = aggregate_predictions["overall_performance"]
        success_probability = aggregate_predictions["success_probability"]
        
        if overall_performance > 0.8:
            insights.append("Excellent performance predicted across all content")
        elif overall_performance > 0.6:
            insights.append("Good performance expected with some challenges")
        else:
            insights.append("Performance may be challenging - consider adjustments")
        
        if success_probability < 0.5:
            insights.append("High risk of struggle - recommend reducing difficulty")
        
        # Identify specific challenging content
        challenging_content = [
            pred for pred in content_predictions 
            if pred["predicted_performance"] < 0.5
        ]
        
        if challenging_content:
            insights.append(f"{len(challenging_content)} content items may be particularly challenging")
        
        return insights
    
    def _calculate_prediction_confidence(
        self, 
        user_model: Dict[str, Any], 
        content_count: int,
        prediction_horizon: int
    ) -> float:
        """Calculate overall confidence in predictions"""
        base_confidence = user_model.get("confidence", 0.5)
        
        # Reduce confidence for more content items
        content_factor = max(0.5, 1.0 - (content_count * 0.05))
        
        # Reduce confidence for longer prediction horizons
        horizon_factor = max(0.3, 1.0 - (prediction_horizon * 0.1))
        
        return base_confidence * content_factor * horizon_factor
    
    # Engagement prediction methods
    
    def _analyze_engagement_patterns(self, historical_engagement: List[float]) -> Dict[str, Any]:
        """Analyze historical engagement patterns"""
        if not historical_engagement:
            return {"confidence": 0.3, "baseline": 0.7, "trend": "unknown"}
        
        baseline = statistics.mean(historical_engagement)
        
        # Calculate trend
        if len(historical_engagement) >= 5:
            recent = historical_engagement[-3:]
            older = historical_engagement[-6:-3] if len(historical_engagement) >= 6 else historical_engagement[:-3]
            
            if older:
                recent_avg = statistics.mean(recent)
                older_avg = statistics.mean(older)
                
                if recent_avg > older_avg * 1.1:
                    trend = "increasing"
                elif recent_avg < older_avg * 0.9:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        # Calculate consistency
        consistency = 1.0 - statistics.stdev(historical_engagement) if len(historical_engagement) > 1 else 0.5
        
        return {
            "baseline": baseline,
            "trend": trend,
            "consistency": consistency,
            "confidence": min(1.0, len(historical_engagement) / 10),
            "data_points": len(historical_engagement)
        }
    
    async def _predict_session_engagement(
        self, 
        user_id: str, 
        session_plan: Dict[str, Any], 
        engagement_patterns: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Predict engagement for a learning session"""
        baseline_engagement = engagement_patterns["baseline"]
        
        # Adjust based on session characteristics
        session_length = session_plan.get("duration_minutes", 30)
        content_variety = len(set(session_plan.get("content_types", ["text"])))
        difficulty_level = session_plan.get("average_difficulty", 0.5)
        
        # Length adjustment
        if session_length > 60:
            length_factor = 0.8  # Longer sessions may reduce engagement
        elif session_length < 20:
            length_factor = 0.9  # Very short sessions may feel rushed
        else:
            length_factor = 1.0
        
        # Variety adjustment
        variety_factor = min(1.2, 1.0 + (content_variety - 1) * 0.1)
        
        # Difficulty adjustment
        if difficulty_level > 0.8:
            difficulty_factor = 0.8  # Very high difficulty may reduce engagement
        elif difficulty_level < 0.3:
            difficulty_factor = 0.9  # Very low difficulty may be boring
        else:
            difficulty_factor = 1.0
        
        predicted_engagement = baseline_engagement * length_factor * variety_factor * difficulty_factor
        
        return {
            "predicted_engagement": min(1.0, max(0.0, predicted_engagement)),
            "baseline_engagement": baseline_engagement,
            "length_factor": length_factor,
            "variety_factor": variety_factor,
            "difficulty_factor": difficulty_factor,
            "confidence": engagement_patterns["confidence"]
        }
    
    def _identify_engagement_risks(
        self, 
        session_plan: Dict[str, Any], 
        engagement_patterns: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify potential engagement risks"""
        risks = []
        
        # Check session length
        session_length = session_plan.get("duration_minutes", 30)
        if session_length > 90:
            risks.append({
                "type": "session_too_long",
                "severity": "high",
                "description": "Session length may lead to fatigue and reduced engagement"
            })
        
        # Check content variety
        content_types = session_plan.get("content_types", ["text"])
        if len(set(content_types)) == 1 and len(content_types) > 3:
            risks.append({
                "type": "low_content_variety",
                "severity": "medium",
                "description": "Lack of content variety may reduce engagement"
            })
        
        # Check historical engagement trend
        if engagement_patterns.get("trend") == "decreasing":
            risks.append({
                "type": "declining_engagement_trend",
                "severity": "high",
                "description": "Historical engagement shows declining trend"
            })
        
        return risks
    
    async def _generate_engagement_optimizations(
        self, 
        session_plan: Dict[str, Any], 
        session_engagement: Dict[str, Any], 
        risk_factors: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate engagement optimization suggestions"""
        optimizations = []
        
        predicted_engagement = session_engagement["predicted_engagement"]
        
        if predicted_engagement < 0.6:
            optimizations.append({
                "type": "increase_interactivity",
                "priority": "high",
                "description": "Add more interactive elements to boost engagement",
                "expected_impact": 0.2
            })
        
        # Address specific risk factors
        for risk in risk_factors:
            if risk["type"] == "session_too_long":
                optimizations.append({
                    "type": "reduce_session_length",
                    "priority": "high",
                    "description": "Break session into shorter segments with breaks",
                    "expected_impact": 0.15
                })
            elif risk["type"] == "low_content_variety":
                optimizations.append({
                    "type": "increase_variety",
                    "priority": "medium",
                    "description": "Add different content types to maintain interest",
                    "expected_impact": 0.1
                })
        
        return optimizations
    
    # Default fallback methods
    
    def _get_default_performance_prediction(self) -> Dict[str, Any]:
        """Get default performance prediction for fallback"""
        return {
            "user_id": "unknown",
            "prediction_horizon_days": 7,
            "content_count": 0,
            "content_predictions": [],
            "aggregate_predictions": {
                "overall_performance": 0.7,
                "overall_engagement": 0.7,
                "success_probability": 0.7
            },
            "insights": ["Insufficient data for detailed predictions"],
            "confidence": 0.3,
            "generated_at": datetime.utcnow().isoformat()
        }
    
    def _get_default_engagement_prediction(self) -> Dict[str, Any]:
        """Get default engagement prediction for fallback"""
        return {
            "user_id": "unknown",
            "predicted_engagement": {"predicted_engagement": 0.7},
            "engagement_patterns": {"baseline": 0.7, "confidence": 0.3},
            "risk_factors": [],
            "optimizations": [],
            "confidence": 0.3
        }
    
    def _get_default_completion_prediction(self) -> Dict[str, Any]:
        """Get default completion prediction for fallback"""
        return {
            "user_id": "unknown",
            "learning_path_length": 0,
            "milestone_predictions": [],
            "overall_completion": {"completion_probability": 0.7},
            "completion_barriers": [],
            "completion_strategies": [],
            "path_analysis": {}
        }
    
    def _get_default_velocity_prediction(self) -> Dict[str, Any]:
        """Get default velocity prediction for fallback"""
        return {
            "user_id": "unknown",
            "content_difficulty": 0.5,
            "predicted_velocity": 0.6,
            "confidence_intervals": {"lower": 0.4, "upper": 0.8},
            "velocity_insights": ["Insufficient data for velocity prediction"],
            "context_factors": {},
            "model_confidence": 0.3
        }

    # Additional missing methods for velocity prediction

    async def _get_velocity_model(self, user_id: str) -> Dict[str, Any]:
        """Get velocity model for user"""
        # Simplified velocity model
        return {
            "base_velocity": 0.6,
            "difficulty_sensitivity": 0.3,
            "context_factors": {},
            "confidence": 0.5,
            "last_updated": datetime.utcnow().isoformat()
        }

    def _predict_base_velocity(self, velocity_model: Dict[str, Any], content_difficulty: float) -> float:
        """Predict base learning velocity"""
        base_velocity = velocity_model.get("base_velocity", 0.6)
        difficulty_sensitivity = velocity_model.get("difficulty_sensitivity", 0.3)

        # Adjust velocity based on difficulty
        difficulty_adjustment = (0.5 - content_difficulty) * difficulty_sensitivity
        adjusted_velocity = base_velocity + difficulty_adjustment

        return max(0.1, min(1.0, adjusted_velocity))

    def _apply_velocity_adjustments(self, base_velocity: float, context: Dict[str, Any]) -> float:
        """Apply contextual adjustments to velocity"""
        adjusted_velocity = base_velocity

        # Time of day adjustment
        hour = context.get("hour_of_day", 12)
        if hour < 9 or hour > 20:
            adjusted_velocity *= 0.9  # Slower at non-optimal times

        # Session length adjustment
        session_length = context.get("session_length", 30)
        if session_length > 60:
            adjusted_velocity *= 0.8  # Slower in long sessions

        return max(0.1, min(1.0, adjusted_velocity))

    def _calculate_velocity_confidence_intervals(
        self,
        predicted_velocity: float,
        velocity_model: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate confidence intervals for velocity prediction"""
        model_confidence = velocity_model.get("confidence", 0.5)

        # Calculate interval width based on confidence
        interval_width = (1.0 - model_confidence) * 0.4

        lower_bound = max(0.1, predicted_velocity - interval_width)
        upper_bound = min(1.0, predicted_velocity + interval_width)

        return {
            "lower": lower_bound,
            "upper": upper_bound,
            "width": upper_bound - lower_bound
        }

    def _generate_velocity_insights(
        self,
        predicted_velocity: float,
        content_difficulty: float,
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate insights about predicted velocity"""
        insights = []

        if predicted_velocity > 0.8:
            insights.append("High learning velocity predicted - user likely to progress quickly")
        elif predicted_velocity < 0.4:
            insights.append("Low learning velocity predicted - may need additional support")
        else:
            insights.append("Moderate learning velocity predicted")

        # Context-specific insights
        if content_difficulty > 0.7 and predicted_velocity > 0.6:
            insights.append("Good velocity despite high content difficulty")

        return insights

    # Completion prediction methods

    def _analyze_learning_path(self, learning_path: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze characteristics of learning path"""
        if not learning_path:
            return {"total_steps": 0, "average_difficulty": 0.5}

        difficulties = [step.get("difficulty", 0.5) for step in learning_path]
        durations = [step.get("estimated_duration", 30) for step in learning_path]

        return {
            "total_steps": len(learning_path),
            "average_difficulty": statistics.mean(difficulties),
            "total_estimated_duration": sum(durations),
            "difficulty_progression": self._analyze_difficulty_progression(difficulties),
            "complexity_score": statistics.mean(difficulties) * len(learning_path)
        }

    def _analyze_difficulty_progression(self, difficulties: List[float]) -> str:
        """Analyze how difficulty progresses through the path"""
        if len(difficulties) < 2:
            return "insufficient_data"

        # Simple trend analysis
        first_half = difficulties[:len(difficulties)//2]
        second_half = difficulties[len(difficulties)//2:]

        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)

        if second_avg > first_avg + 0.1:
            return "increasing"
        elif second_avg < first_avg - 0.1:
            return "decreasing"
        else:
            return "stable"

    async def _get_completion_history(self, user_id: str) -> Dict[str, Any]:
        """Get user's completion history"""
        # Simplified completion history
        return {
            "average_completion_rate": 0.75,
            "typical_session_length": 35,
            "completion_patterns": ["consistent"],
            "historical_data_points": 10
        }

    async def _predict_milestone_completion(
        self,
        user_id: str,
        milestone: Dict[str, Any],
        completion_history: Dict[str, Any],
        cumulative_probability: float,
        user_constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Predict completion probability for a milestone"""
        base_completion_rate = completion_history.get("average_completion_rate", 0.75)
        milestone_difficulty = milestone.get("difficulty", 0.5)

        # Adjust for difficulty
        difficulty_factor = max(0.3, 1.0 - (milestone_difficulty - 0.5))

        # Adjust for cumulative fatigue
        fatigue_factor = max(0.5, cumulative_probability)

        # Adjust for constraints
        constraint_factor = self._calculate_constraint_factor(user_constraints)

        completion_probability = base_completion_rate * difficulty_factor * fatigue_factor * constraint_factor

        return {
            "milestone_id": milestone.get("id", "unknown"),
            "completion_probability": min(1.0, max(0.1, completion_probability)),
            "difficulty": milestone_difficulty,
            "estimated_duration": milestone.get("estimated_duration", 30),
            "confidence": 0.6
        }

    def _calculate_constraint_factor(self, constraints: Dict[str, Any]) -> float:
        """Calculate how user constraints affect completion probability"""
        factor = 1.0

        # Time constraints
        available_time = constraints.get("available_time_hours", 10)
        if available_time < 5:
            factor *= 0.7

        # Motivation level
        motivation = constraints.get("motivation_level", 0.7)
        factor *= motivation

        # External support
        support_level = constraints.get("support_level", 0.5)
        factor *= (0.8 + support_level * 0.2)

        return max(0.2, min(1.0, factor))

    def _calculate_overall_completion_metrics(
        self,
        milestone_predictions: List[Dict[str, Any]],
        path_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate overall completion metrics"""
        if not milestone_predictions:
            return {"completion_probability": 0.5}

        # Calculate cumulative completion probability
        cumulative_prob = 1.0
        for prediction in milestone_predictions:
            cumulative_prob *= prediction["completion_probability"]

        # Calculate average completion probability
        avg_completion_prob = statistics.mean([p["completion_probability"] for p in milestone_predictions])

        # Calculate total estimated time
        total_time = sum(p["estimated_duration"] for p in milestone_predictions)

        return {
            "completion_probability": cumulative_prob,
            "average_milestone_probability": avg_completion_prob,
            "total_estimated_duration": total_time,
            "path_complexity": path_analysis.get("complexity_score", 0.5),
            "overall_confidence": statistics.mean([p["confidence"] for p in milestone_predictions])
        }

    def _identify_completion_barriers(
        self,
        milestone_predictions: List[Dict[str, Any]],
        user_constraints: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify potential completion barriers"""
        barriers = []

        # Identify low-probability milestones
        for prediction in milestone_predictions:
            if prediction["completion_probability"] < 0.5:
                barriers.append({
                    "type": "difficult_milestone",
                    "milestone_id": prediction["milestone_id"],
                    "probability": prediction["completion_probability"],
                    "severity": "high" if prediction["completion_probability"] < 0.3 else "medium"
                })

        # Check time constraints
        total_time = sum(p["estimated_duration"] for p in milestone_predictions)
        available_time = user_constraints.get("available_time_hours", 10) * 60  # Convert to minutes

        if total_time > available_time:
            barriers.append({
                "type": "time_constraint",
                "required_time": total_time,
                "available_time": available_time,
                "severity": "high"
            })

        return barriers

    async def _generate_completion_strategies(
        self,
        barriers: List[Dict[str, Any]],
        overall_completion: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate strategies to improve completion probability"""
        strategies = []

        for barrier in barriers:
            if barrier["type"] == "difficult_milestone":
                strategies.append({
                    "strategy": "milestone_support",
                    "description": "Provide additional support for difficult milestones",
                    "actions": [
                        "Add prerequisite review",
                        "Provide additional examples",
                        "Offer alternative learning paths"
                    ],
                    "target_barrier": barrier["milestone_id"]
                })
            elif barrier["type"] == "time_constraint":
                strategies.append({
                    "strategy": "time_optimization",
                    "description": "Optimize learning path for available time",
                    "actions": [
                        "Prioritize essential milestones",
                        "Suggest shorter learning sessions",
                        "Provide time management guidance"
                    ],
                    "target_barrier": "time_management"
                })

        return strategies

    # Model update methods

    def _calculate_prediction_accuracy(
        self,
        actual_performance: Dict[str, Any],
        predicted_performance: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate accuracy of predictions"""
        # Extract actual and predicted values
        actual_success = actual_performance.get("success_rate", 0.5)
        predicted_success = predicted_performance.get("overall_performance", 0.5)

        actual_engagement = actual_performance.get("engagement_score", 0.5)
        predicted_engagement = predicted_performance.get("overall_engagement", 0.5)

        # Calculate accuracy metrics
        success_accuracy = 1.0 - abs(actual_success - predicted_success)
        engagement_accuracy = 1.0 - abs(actual_engagement - predicted_engagement)

        overall_accuracy = (success_accuracy + engagement_accuracy) / 2

        return {
            "success_accuracy": success_accuracy,
            "engagement_accuracy": engagement_accuracy,
            "overall_accuracy": overall_accuracy,
            "prediction_error": 1.0 - overall_accuracy
        }

    async def _update_user_model(
        self,
        user_id: str,
        actual_performance: Dict[str, Any],
        accuracy_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update user prediction model based on actual performance"""
        current_model = self.user_models.get(user_id, {})

        # Update model confidence based on accuracy
        current_confidence = current_model.get("confidence", 0.5)
        accuracy = accuracy_metrics["overall_accuracy"]

        # Adjust confidence (moving average)
        new_confidence = (current_confidence * 0.8) + (accuracy * 0.2)

        # Update interaction count
        interaction_count = current_model.get("interaction_count", 0) + 1

        # Update model
        updated_model = {
            **current_model,
            "confidence": new_confidence,
            "interaction_count": interaction_count,
            "last_updated": datetime.utcnow().isoformat(),
            "last_accuracy": accuracy
        }

        self.user_models[user_id] = updated_model

        return updated_model

    def _store_prediction_history(
        self,
        user_id: str,
        predicted_performance: Dict[str, Any],
        actual_performance: Dict[str, Any],
        accuracy_metrics: Dict[str, Any]
    ):
        """Store prediction history for analysis"""
        if user_id not in self.prediction_history:
            self.prediction_history[user_id] = []

        history_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "predicted": predicted_performance,
            "actual": actual_performance,
            "accuracy": accuracy_metrics
        }

        self.prediction_history[user_id].append(history_entry)

        # Keep only recent history
        if len(self.prediction_history[user_id]) > 50:
            self.prediction_history[user_id] = self.prediction_history[user_id][-50:]

    def _generate_model_insights(
        self,
        accuracy_metrics: Dict[str, Any],
        updated_model: Dict[str, Any]
    ) -> List[str]:
        """Generate insights about model performance"""
        insights = []

        overall_accuracy = accuracy_metrics["overall_accuracy"]

        if overall_accuracy > 0.8:
            insights.append("Prediction model showing high accuracy")
        elif overall_accuracy < 0.5:
            insights.append("Prediction model needs improvement")

        model_confidence = updated_model.get("confidence", 0.5)
        if model_confidence > 0.8:
            insights.append("High confidence in prediction model")
        elif model_confidence < 0.4:
            insights.append("Low confidence in prediction model - more data needed")

        return insights
