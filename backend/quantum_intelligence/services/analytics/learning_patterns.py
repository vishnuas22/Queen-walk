"""
Learning Pattern Analysis Engine

Extracted from quantum_intelligence_engine.py - advanced learning pattern analysis
and predictive analytics for personalized learning optimization.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from collections import defaultdict, deque
import statistics
import math

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


class LearningPatternAnalysisEngine:
    """
    📊 LEARNING PATTERN ANALYSIS ENGINE
    
    Advanced learning pattern analysis and predictive analytics.
    Extracted from the original quantum engine's analytics logic.
    """
    
    def __init__(self, cache_service: Optional[CacheService] = None):
        self.cache = cache_service
        
        # Pattern analysis storage
        self.user_patterns = defaultdict(dict)
        self.session_data = defaultdict(deque)
        self.performance_metrics = defaultdict(dict)
        
        # Analysis parameters
        self.pattern_window = 100  # Number of interactions to analyze
        self.trend_sensitivity = 0.1  # Sensitivity for trend detection
        self.confidence_threshold = 0.7  # Minimum confidence for predictions
        
        logger.info("Learning Pattern Analysis Engine initialized")
    
    async def analyze_learning_patterns(
        self, 
        user_id: str, 
        interaction_history: List[Dict[str, Any]],
        time_window_days: int = 30
    ) -> Dict[str, Any]:
        """
        Comprehensive learning pattern analysis
        
        Extracted from original pattern analysis logic
        """
        try:
            # Filter recent interactions
            cutoff_date = datetime.utcnow() - timedelta(days=time_window_days)
            recent_interactions = [
                interaction for interaction in interaction_history
                if datetime.fromisoformat(interaction.get("timestamp", "2024-01-01T00:00:00")) > cutoff_date
            ]
            
            if not recent_interactions:
                return self._get_default_pattern_analysis()
            
            # Analyze multiple pattern dimensions
            patterns = {
                "temporal_patterns": await self._analyze_temporal_patterns(user_id, recent_interactions),
                "performance_patterns": await self._analyze_performance_patterns(user_id, recent_interactions),
                "engagement_patterns": await self._analyze_engagement_patterns(user_id, recent_interactions),
                "difficulty_progression": await self._analyze_difficulty_progression(user_id, recent_interactions),
                "learning_velocity_trends": await self._analyze_velocity_trends(user_id, recent_interactions),
                "concept_mastery_patterns": await self._analyze_concept_mastery(user_id, recent_interactions),
                "session_optimization": await self._analyze_session_patterns(user_id, recent_interactions),
                "struggle_recovery_patterns": await self._analyze_struggle_recovery(user_id, recent_interactions),
                "breakthrough_indicators": await self._analyze_breakthrough_patterns(user_id, recent_interactions),
                "metacognitive_development": await self._analyze_metacognitive_growth(user_id, recent_interactions)
            }
            
            # Generate pattern insights
            insights = await self._generate_pattern_insights(patterns)
            
            # Calculate pattern confidence
            confidence = self._calculate_pattern_confidence(patterns, len(recent_interactions))
            
            # Store patterns for future reference
            self.user_patterns[user_id] = {
                "patterns": patterns,
                "insights": insights,
                "confidence": confidence,
                "last_updated": datetime.utcnow().isoformat(),
                "data_points": len(recent_interactions)
            }
            
            return {
                "user_id": user_id,
                "analysis_period": f"{time_window_days} days",
                "data_points": len(recent_interactions),
                "patterns": patterns,
                "insights": insights,
                "confidence": confidence,
                "recommendations": await self._generate_pattern_recommendations(patterns, insights)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing learning patterns for user {user_id}: {e}")
            return self._get_default_pattern_analysis()
    
    async def predict_learning_outcomes(
        self, 
        user_id: str, 
        proposed_learning_path: List[Dict[str, Any]],
        prediction_horizon_days: int = 7
    ) -> Dict[str, Any]:
        """
        Predict learning outcomes based on patterns
        
        Extracted from original outcome prediction logic
        """
        try:
            # Get user patterns
            user_patterns = self.user_patterns.get(user_id, {})
            if not user_patterns:
                return self._get_default_predictions()
            
            patterns = user_patterns.get("patterns", {})
            
            # Predict outcomes for each step in the learning path
            path_predictions = []
            cumulative_confidence = 1.0
            
            for i, learning_step in enumerate(proposed_learning_path):
                step_prediction = await self._predict_step_outcome(
                    user_id, 
                    learning_step, 
                    patterns,
                    i,
                    cumulative_confidence
                )
                path_predictions.append(step_prediction)
                cumulative_confidence *= step_prediction.get("confidence", 0.7)
            
            # Calculate overall path predictions
            overall_predictions = self._calculate_overall_path_predictions(path_predictions)
            
            # Generate optimization suggestions
            optimizations = await self._generate_path_optimizations(
                proposed_learning_path, 
                path_predictions, 
                patterns
            )
            
            return {
                "user_id": user_id,
                "prediction_horizon": f"{prediction_horizon_days} days",
                "learning_path_length": len(proposed_learning_path),
                "step_predictions": path_predictions,
                "overall_predictions": overall_predictions,
                "optimizations": optimizations,
                "confidence": overall_predictions.get("overall_confidence", 0.5)
            }
            
        except Exception as e:
            logger.error(f"Error predicting learning outcomes for user {user_id}: {e}")
            return self._get_default_predictions()
    
    async def identify_learning_bottlenecks(
        self, 
        user_id: str, 
        performance_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Identify learning bottlenecks and optimization opportunities
        
        Extracted from original bottleneck analysis logic
        """
        try:
            bottlenecks = {
                "cognitive_bottlenecks": [],
                "temporal_bottlenecks": [],
                "motivational_bottlenecks": [],
                "content_bottlenecks": [],
                "environmental_bottlenecks": []
            }
            
            # Analyze cognitive bottlenecks
            cognitive_issues = await self._identify_cognitive_bottlenecks(performance_data)
            bottlenecks["cognitive_bottlenecks"] = cognitive_issues
            
            # Analyze temporal bottlenecks
            temporal_issues = await self._identify_temporal_bottlenecks(performance_data)
            bottlenecks["temporal_bottlenecks"] = temporal_issues
            
            # Analyze motivational bottlenecks
            motivational_issues = await self._identify_motivational_bottlenecks(performance_data)
            bottlenecks["motivational_bottlenecks"] = motivational_issues
            
            # Analyze content bottlenecks
            content_issues = await self._identify_content_bottlenecks(performance_data)
            bottlenecks["content_bottlenecks"] = content_issues
            
            # Analyze environmental bottlenecks
            environmental_issues = await self._identify_environmental_bottlenecks(performance_data)
            bottlenecks["environmental_bottlenecks"] = environmental_issues
            
            # Prioritize bottlenecks by impact
            prioritized_bottlenecks = self._prioritize_bottlenecks(bottlenecks)
            
            # Generate resolution strategies
            resolution_strategies = await self._generate_bottleneck_resolutions(prioritized_bottlenecks)
            
            return {
                "user_id": user_id,
                "bottlenecks": bottlenecks,
                "prioritized_bottlenecks": prioritized_bottlenecks,
                "resolution_strategies": resolution_strategies,
                "impact_assessment": self._assess_bottleneck_impact(bottlenecks)
            }
            
        except Exception as e:
            logger.error(f"Error identifying bottlenecks for user {user_id}: {e}")
            return {"bottlenecks": {}, "prioritized_bottlenecks": [], "resolution_strategies": []}
    
    async def generate_learning_insights(
        self, 
        user_id: str, 
        analysis_depth: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive learning insights
        
        Extracted from original insight generation logic
        """
        try:
            user_patterns = self.user_patterns.get(user_id, {})
            if not user_patterns:
                return self._get_default_insights()
            
            patterns = user_patterns.get("patterns", {})
            confidence = user_patterns.get("confidence", 0.5)
            
            insights = {
                "learning_strengths": await self._identify_learning_strengths(patterns),
                "improvement_opportunities": await self._identify_improvement_opportunities(patterns),
                "optimal_learning_conditions": await self._identify_optimal_conditions(patterns),
                "personalization_recommendations": await self._generate_personalization_recommendations(patterns),
                "learning_trajectory": await self._analyze_learning_trajectory(patterns),
                "mastery_predictions": await self._predict_mastery_timeline(patterns),
                "adaptive_strategies": await self._recommend_adaptive_strategies(patterns)
            }
            
            # Add depth-specific insights
            if analysis_depth == "comprehensive":
                insights.update({
                    "advanced_analytics": await self._generate_advanced_analytics(patterns),
                    "comparative_analysis": await self._generate_comparative_insights(user_id, patterns),
                    "predictive_modeling": await self._generate_predictive_insights(patterns)
                })
            
            return {
                "user_id": user_id,
                "analysis_depth": analysis_depth,
                "insights": insights,
                "confidence": confidence,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating insights for user {user_id}: {e}")
            return self._get_default_insights()
    
    # Prediction and insight generation methods (continued in next methods)

    async def _predict_step_outcome(
        self,
        user_id: str,
        learning_step: Dict[str, Any],
        patterns: Dict[str, Any],
        step_index: int,
        cumulative_confidence: float
    ) -> Dict[str, Any]:
        """Predict outcome for a single learning step"""
        # Simplified prediction logic
        base_success_prob = 0.7

        # Adjust based on difficulty
        difficulty = learning_step.get("difficulty", 0.5)
        performance_patterns = patterns.get("performance_patterns", {})
        optimal_difficulty = performance_patterns.get("optimal_difficulty", 0.5)

        difficulty_adjustment = 1.0 - abs(difficulty - optimal_difficulty)

        # Adjust based on engagement patterns
        engagement_patterns = patterns.get("engagement_patterns", {})
        avg_engagement = engagement_patterns.get("average_engagement", 0.7)

        success_probability = base_success_prob * difficulty_adjustment * avg_engagement

        return {
            "step_index": step_index,
            "success_probability": min(1.0, max(0.0, success_probability)),
            "confidence": cumulative_confidence * 0.9,
            "difficulty": difficulty,
            "estimated_duration": learning_step.get("estimated_duration", 30)
        }

    def _calculate_overall_path_predictions(self, step_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall predictions for the learning path"""
        if not step_predictions:
            return {"overall_confidence": 0.5}

        success_probs = [pred["success_probability"] for pred in step_predictions]
        confidences = [pred["confidence"] for pred in step_predictions]

        return {
            "overall_success_probability": statistics.mean(success_probs),
            "completion_probability": min(success_probs) if success_probs else 0.5,
            "average_confidence": statistics.mean(confidences),
            "overall_confidence": min(confidences) if confidences else 0.5,
            "total_estimated_duration": sum(pred["estimated_duration"] for pred in step_predictions)
        }

    async def _generate_path_optimizations(
        self,
        learning_path: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        patterns: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate optimization suggestions for the learning path"""
        optimizations = []

        for i, (step, prediction) in enumerate(zip(learning_path, predictions)):
            if prediction["success_probability"] < 0.6:
                optimizations.append({
                    "step_index": i,
                    "issue": "low_success_probability",
                    "suggestion": "reduce_difficulty",
                    "current_difficulty": step.get("difficulty", 0.5),
                    "recommended_difficulty": max(0.1, step.get("difficulty", 0.5) - 0.2)
                })

        return optimizations

    # Bottleneck identification methods

    async def _identify_cognitive_bottlenecks(self, performance_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify cognitive bottlenecks"""
        bottlenecks = []

        # Analyze response times
        response_times = [data.get("response_time", 5.0) for data in performance_data]
        if response_times:
            avg_response_time = statistics.mean(response_times)
            if avg_response_time > 10.0:
                bottlenecks.append({
                    "type": "slow_processing",
                    "severity": "high" if avg_response_time > 15.0 else "medium",
                    "description": "User takes longer than average to process information",
                    "metric": avg_response_time
                })

        return bottlenecks

    async def _identify_temporal_bottlenecks(self, performance_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify temporal bottlenecks"""
        bottlenecks = []

        # Analyze session timing
        session_lengths = [data.get("session_length", 30) for data in performance_data]
        if session_lengths:
            avg_session_length = statistics.mean(session_lengths)
            if avg_session_length < 15:
                bottlenecks.append({
                    "type": "short_sessions",
                    "severity": "medium",
                    "description": "Sessions are too short for effective learning",
                    "metric": avg_session_length
                })

        return bottlenecks

    async def _identify_motivational_bottlenecks(self, performance_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify motivational bottlenecks"""
        bottlenecks = []

        # Analyze engagement trends
        engagement_scores = [data.get("engagement_score", 0.5) for data in performance_data]
        if engagement_scores:
            avg_engagement = statistics.mean(engagement_scores)
            if avg_engagement < 0.5:
                bottlenecks.append({
                    "type": "low_engagement",
                    "severity": "high" if avg_engagement < 0.3 else "medium",
                    "description": "User engagement is below optimal levels",
                    "metric": avg_engagement
                })

        return bottlenecks

    async def _identify_content_bottlenecks(self, performance_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify content-related bottlenecks"""
        bottlenecks = []

        # Analyze difficulty vs performance
        difficulties = [data.get("difficulty", 0.5) for data in performance_data]
        successes = [data.get("success", False) for data in performance_data]

        if difficulties and successes:
            high_difficulty_indices = [i for i, d in enumerate(difficulties) if d > 0.7]
            if high_difficulty_indices:
                high_diff_success_rate = sum(successes[i] for i in high_difficulty_indices) / len(high_difficulty_indices)
                if high_diff_success_rate < 0.3:
                    bottlenecks.append({
                        "type": "content_too_difficult",
                        "severity": "high",
                        "description": "Content difficulty exceeds user capability",
                        "metric": high_diff_success_rate
                    })

        return bottlenecks

    async def _identify_environmental_bottlenecks(self, performance_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify environmental bottlenecks"""
        bottlenecks = []

        # This would analyze environmental factors like device type, time of day, etc.
        # For now, return empty list as we don't have enough environmental data

        return bottlenecks

    def _prioritize_bottlenecks(self, bottlenecks: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Prioritize bottlenecks by severity and impact"""
        all_bottlenecks = []

        for category, category_bottlenecks in bottlenecks.items():
            for bottleneck in category_bottlenecks:
                bottleneck["category"] = category
                all_bottlenecks.append(bottleneck)

        # Sort by severity (high > medium > low)
        severity_order = {"high": 3, "medium": 2, "low": 1}
        all_bottlenecks.sort(key=lambda x: severity_order.get(x.get("severity", "low"), 1), reverse=True)

        return all_bottlenecks

    async def _generate_bottleneck_resolutions(self, prioritized_bottlenecks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate resolution strategies for bottlenecks"""
        resolutions = []

        for bottleneck in prioritized_bottlenecks:
            bottleneck_type = bottleneck.get("type", "unknown")

            if bottleneck_type == "slow_processing":
                resolutions.append({
                    "bottleneck_type": bottleneck_type,
                    "strategy": "reduce_cognitive_load",
                    "actions": ["Break content into smaller chunks", "Add more examples", "Increase explanation depth"]
                })
            elif bottleneck_type == "low_engagement":
                resolutions.append({
                    "bottleneck_type": bottleneck_type,
                    "strategy": "increase_engagement",
                    "actions": ["Add interactive elements", "Vary content types", "Provide immediate feedback"]
                })
            elif bottleneck_type == "content_too_difficult":
                resolutions.append({
                    "bottleneck_type": bottleneck_type,
                    "strategy": "adjust_difficulty",
                    "actions": ["Reduce content complexity", "Add prerequisite review", "Provide scaffolding"]
                })

        return resolutions

    def _assess_bottleneck_impact(self, bottlenecks: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Assess the overall impact of identified bottlenecks"""
        total_bottlenecks = sum(len(category_bottlenecks) for category_bottlenecks in bottlenecks.values())
        high_severity_count = sum(
            1 for category_bottlenecks in bottlenecks.values()
            for bottleneck in category_bottlenecks
            if bottleneck.get("severity") == "high"
        )

        impact_level = "high" if high_severity_count > 2 else "medium" if total_bottlenecks > 3 else "low"

        return {
            "total_bottlenecks": total_bottlenecks,
            "high_severity_count": high_severity_count,
            "impact_level": impact_level,
            "resolution_priority": "immediate" if impact_level == "high" else "moderate" if impact_level == "medium" else "low"
        }

    # Insight generation methods (simplified versions)

    async def _identify_learning_strengths(self, patterns: Dict[str, Any]) -> List[str]:
        """Identify learning strengths from patterns"""
        strengths = []

        performance_patterns = patterns.get("performance_patterns", {})
        if performance_patterns.get("overall_performance", 0.5) > 0.7:
            strengths.append("Strong overall performance")

        engagement_patterns = patterns.get("engagement_patterns", {})
        if engagement_patterns.get("average_engagement", 0.5) > 0.8:
            strengths.append("High engagement levels")

        return strengths if strengths else ["Consistent learning approach"]

    async def _identify_improvement_opportunities(self, patterns: Dict[str, Any]) -> List[str]:
        """Identify improvement opportunities from patterns"""
        opportunities = []

        performance_patterns = patterns.get("performance_patterns", {})
        if performance_patterns.get("overall_performance", 0.5) < 0.6:
            opportunities.append("Focus on improving success rate")

        return opportunities if opportunities else ["Continue current development"]

    async def _identify_optimal_conditions(self, patterns: Dict[str, Any]) -> List[str]:
        """Identify optimal learning conditions"""
        conditions = []

        temporal_patterns = patterns.get("temporal_patterns", {})
        optimal_hours = temporal_patterns.get("optimal_hours", [])
        if optimal_hours:
            best_hour = optimal_hours[0].get("hour", 12)
            conditions.append(f"Learn around {best_hour}:00 for best performance")

        return conditions if conditions else ["Maintain consistent learning schedule"]

    async def _generate_personalization_recommendations(self, patterns: Dict[str, Any]) -> List[str]:
        """Generate personalization recommendations"""
        recommendations = []

        difficulty_patterns = patterns.get("difficulty_progression", {})
        if difficulty_patterns.get("ready_for_increase", False):
            recommendations.append("Ready for increased difficulty level")

        return recommendations if recommendations else ["Continue with current personalization"]

    async def _analyze_learning_trajectory(self, patterns: Dict[str, Any]) -> str:
        """Analyze overall learning trajectory"""
        performance_patterns = patterns.get("performance_patterns", {})
        trend = performance_patterns.get("performance_trend", "stable")

        if trend == "increasing":
            return "positive_trajectory"
        elif trend == "decreasing":
            return "concerning_trajectory"
        else:
            return "stable_trajectory"

    async def _predict_mastery_timeline(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Predict mastery timeline"""
        concept_mastery = patterns.get("concept_mastery_patterns", {})
        mastery_rate = concept_mastery.get("overall_mastery_rate", 0.5)

        if mastery_rate > 0.8:
            timeline = "1-2 weeks"
        elif mastery_rate > 0.6:
            timeline = "2-4 weeks"
        else:
            timeline = "4+ weeks"

        return {"timeline": timeline, "confidence": 0.6}

    async def _recommend_adaptive_strategies(self, patterns: Dict[str, Any]) -> List[str]:
        """Recommend adaptive strategies"""
        strategies = []

        velocity_patterns = patterns.get("learning_velocity_trends", {})
        if velocity_patterns.get("learning_acceleration", 0) > 0.1:
            strategies.append("Increase content complexity gradually")

        return strategies if strategies else ["Monitor progress and adapt as needed"]

    async def _generate_advanced_analytics(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Generate advanced analytics"""
        return {
            "pattern_complexity": "moderate",
            "prediction_accuracy": 0.75,
            "optimization_potential": 0.6
        }

    async def _generate_comparative_insights(self, user_id: str, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparative insights"""
        return {
            "peer_comparison": "above_average",
            "improvement_rate": "steady",
            "relative_strengths": ["engagement", "consistency"]
        }

    async def _generate_predictive_insights(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Generate predictive insights"""
        return {
            "success_probability_next_week": 0.75,
            "engagement_forecast": "stable",
            "recommended_adjustments": ["maintain_current_approach"]
        }

    async def _generate_pattern_insights(self, patterns: Dict[str, Any]) -> List[str]:
        """Generate insights from patterns"""
        insights = []

        # Analyze each pattern type for insights
        for pattern_type, pattern_data in patterns.items():
            if isinstance(pattern_data, dict) and pattern_data.get("pattern") == "analyzed":
                if pattern_type == "performance_patterns":
                    trend = pattern_data.get("performance_trend", "stable")
                    if trend == "increasing":
                        insights.append("Performance is improving over time")
                    elif trend == "decreasing":
                        insights.append("Performance shows declining trend - intervention needed")

                elif pattern_type == "engagement_patterns":
                    avg_engagement = pattern_data.get("average_engagement", 0.5)
                    if avg_engagement > 0.8:
                        insights.append("Excellent engagement levels maintained")
                    elif avg_engagement < 0.5:
                        insights.append("Engagement levels need improvement")

        return insights if insights else ["Learning patterns are being established"]

    async def _generate_pattern_recommendations(self, patterns: Dict[str, Any], insights: List[str]) -> List[str]:
        """Generate recommendations based on patterns and insights"""
        recommendations = []

        # Generate recommendations based on insights
        for insight in insights:
            if "declining trend" in insight:
                recommendations.append("Consider reducing difficulty or taking a break")
            elif "improving" in insight:
                recommendations.append("Consider gradually increasing challenge level")
            elif "engagement" in insight and "need" in insight:
                recommendations.append("Try different content types or interactive elements")

        return recommendations if recommendations else ["Continue monitoring learning patterns"]

    # Private analysis methods

    async def _analyze_temporal_patterns(self, user_id: str, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal learning patterns"""
        if not interactions:
            return {"pattern": "insufficient_data"}
        
        # Extract timestamps and performance
        temporal_data = []
        for interaction in interactions:
            timestamp = datetime.fromisoformat(interaction.get("timestamp", "2024-01-01T00:00:00"))
            performance = interaction.get("success", False)
            engagement = interaction.get("engagement_score", 0.5)
            
            temporal_data.append({
                "hour": timestamp.hour,
                "day_of_week": timestamp.weekday(),
                "performance": 1.0 if performance else 0.0,
                "engagement": engagement
            })
        
        # Analyze by hour of day
        hourly_performance = defaultdict(list)
        hourly_engagement = defaultdict(list)
        
        for data in temporal_data:
            hourly_performance[data["hour"]].append(data["performance"])
            hourly_engagement[data["hour"]].append(data["engagement"])
        
        # Find optimal hours
        optimal_hours = []
        for hour, performances in hourly_performance.items():
            if len(performances) >= 3:  # Minimum data points
                avg_performance = statistics.mean(performances)
                avg_engagement = statistics.mean(hourly_engagement[hour])
                
                if avg_performance > 0.7 and avg_engagement > 0.7:
                    optimal_hours.append({
                        "hour": hour,
                        "performance": avg_performance,
                        "engagement": avg_engagement
                    })
        
        # Analyze by day of week
        daily_performance = defaultdict(list)
        for data in temporal_data:
            daily_performance[data["day_of_week"]].append(data["performance"])
        
        optimal_days = []
        for day, performances in daily_performance.items():
            if len(performances) >= 2:
                avg_performance = statistics.mean(performances)
                if avg_performance > 0.7:
                    optimal_days.append({
                        "day": day,
                        "performance": avg_performance
                    })
        
        return {
            "pattern": "analyzed",
            "optimal_hours": sorted(optimal_hours, key=lambda x: x["performance"], reverse=True)[:3],
            "optimal_days": sorted(optimal_days, key=lambda x: x["performance"], reverse=True)[:3],
            "total_sessions": len(interactions),
            "analysis_confidence": min(1.0, len(interactions) / 20)
        }
    
    async def _analyze_performance_patterns(self, user_id: str, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance patterns and trends"""
        if len(interactions) < 5:
            return {"pattern": "insufficient_data"}
        
        # Extract performance metrics
        performances = []
        response_times = []
        difficulties = []
        
        for interaction in interactions:
            performances.append(1.0 if interaction.get("success", False) else 0.0)
            response_times.append(interaction.get("response_time", 5.0))
            difficulties.append(interaction.get("difficulty", 0.5))
        
        # Calculate trends
        performance_trend = self._calculate_trend(performances)
        response_time_trend = self._calculate_trend(response_times)
        
        # Analyze performance by difficulty
        difficulty_performance = defaultdict(list)
        for i, difficulty in enumerate(difficulties):
            difficulty_bin = round(difficulty, 1)
            difficulty_performance[difficulty_bin].append(performances[i])
        
        optimal_difficulty = 0.5
        best_performance = 0.0
        
        for difficulty, perfs in difficulty_performance.items():
            if len(perfs) >= 3:
                avg_perf = statistics.mean(perfs)
                if avg_perf > best_performance:
                    best_performance = avg_perf
                    optimal_difficulty = difficulty
        
        return {
            "pattern": "analyzed",
            "overall_performance": statistics.mean(performances),
            "performance_trend": performance_trend,
            "response_time_trend": response_time_trend,
            "optimal_difficulty": optimal_difficulty,
            "best_performance_rate": best_performance,
            "consistency": 1.0 - statistics.stdev(performances) if len(performances) > 1 else 0.5,
            "improvement_rate": self._calculate_improvement_rate(performances)
        }
    
    async def _analyze_engagement_patterns(self, user_id: str, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze engagement patterns"""
        if not interactions:
            return {"pattern": "insufficient_data"}
        
        engagement_scores = [interaction.get("engagement_score", 0.5) for interaction in interactions]
        session_lengths = [interaction.get("session_length", 30) for interaction in interactions]
        
        # Analyze engagement trends
        engagement_trend = self._calculate_trend(engagement_scores)
        
        # Analyze engagement by session length
        length_engagement = defaultdict(list)
        for i, length in enumerate(session_lengths):
            length_bin = (length // 10) * 10  # Group by 10-minute intervals
            length_engagement[length_bin].append(engagement_scores[i])
        
        optimal_length = 30
        best_engagement = 0.0
        
        for length, engagements in length_engagement.items():
            if len(engagements) >= 2:
                avg_engagement = statistics.mean(engagements)
                if avg_engagement > best_engagement:
                    best_engagement = avg_engagement
                    optimal_length = length
        
        return {
            "pattern": "analyzed",
            "average_engagement": statistics.mean(engagement_scores),
            "engagement_trend": engagement_trend,
            "optimal_session_length": optimal_length,
            "best_engagement_score": best_engagement,
            "engagement_consistency": 1.0 - statistics.stdev(engagement_scores) if len(engagement_scores) > 1 else 0.5,
            "peak_engagement": max(engagement_scores),
            "low_engagement_threshold": statistics.mean(engagement_scores) - statistics.stdev(engagement_scores) if len(engagement_scores) > 1 else 0.3
        }
    
    async def _analyze_difficulty_progression(self, user_id: str, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze difficulty progression patterns"""
        if len(interactions) < 10:
            return {"pattern": "insufficient_data"}
        
        difficulties = [interaction.get("difficulty", 0.5) for interaction in interactions]
        successes = [interaction.get("success", False) for interaction in interactions]
        
        # Analyze difficulty progression
        difficulty_trend = self._calculate_trend(difficulties)
        
        # Calculate success rate by difficulty level
        difficulty_success = defaultdict(list)
        for i, difficulty in enumerate(difficulties):
            difficulty_bin = round(difficulty, 1)
            difficulty_success[difficulty_bin].append(successes[i])
        
        # Find optimal difficulty progression
        progression_analysis = []
        sorted_difficulties = sorted(difficulty_success.keys())
        
        for difficulty in sorted_difficulties:
            success_rate = sum(difficulty_success[difficulty]) / len(difficulty_success[difficulty])
            progression_analysis.append({
                "difficulty": difficulty,
                "success_rate": success_rate,
                "attempts": len(difficulty_success[difficulty])
            })
        
        # Identify readiness for next level
        current_difficulty = difficulties[-5:] if len(difficulties) >= 5 else difficulties
        current_avg_difficulty = statistics.mean(current_difficulty)
        recent_success_rate = sum(successes[-5:]) / len(successes[-5:]) if len(successes) >= 5 else 0.5
        
        ready_for_increase = recent_success_rate > 0.8 and len(successes[-5:]) >= 3
        
        return {
            "pattern": "analyzed",
            "difficulty_trend": difficulty_trend,
            "current_difficulty": current_avg_difficulty,
            "recent_success_rate": recent_success_rate,
            "ready_for_increase": ready_for_increase,
            "progression_analysis": progression_analysis,
            "recommended_next_difficulty": min(1.0, current_avg_difficulty + 0.1) if ready_for_increase else current_avg_difficulty
        }
    
    async def _analyze_velocity_trends(self, user_id: str, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze learning velocity trends"""
        if len(interactions) < 5:
            return {"pattern": "insufficient_data"}
        
        response_times = [interaction.get("response_time", 5.0) for interaction in interactions]
        completion_times = [interaction.get("completion_time", 300) for interaction in interactions]
        
        # Calculate velocity metrics
        response_time_trend = self._calculate_trend(response_times)
        completion_time_trend = self._calculate_trend(completion_times)
        
        # Calculate learning acceleration
        recent_response_times = response_times[-5:] if len(response_times) >= 5 else response_times
        older_response_times = response_times[-10:-5] if len(response_times) >= 10 else response_times[:-5] if len(response_times) > 5 else []
        
        if older_response_times:
            recent_avg = statistics.mean(recent_response_times)
            older_avg = statistics.mean(older_response_times)
            acceleration = (older_avg - recent_avg) / older_avg if older_avg > 0 else 0
        else:
            acceleration = 0
        
        return {
            "pattern": "analyzed",
            "response_time_trend": response_time_trend,
            "completion_time_trend": completion_time_trend,
            "learning_acceleration": acceleration,
            "current_velocity": 1.0 / statistics.mean(recent_response_times) if recent_response_times else 0.2,
            "velocity_consistency": 1.0 - (statistics.stdev(response_times) / statistics.mean(response_times)) if len(response_times) > 1 and statistics.mean(response_times) > 0 else 0.5
        }
    
    async def _analyze_concept_mastery(self, user_id: str, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze concept mastery patterns"""
        if not interactions:
            return {"pattern": "insufficient_data"}
        
        concept_performance = defaultdict(list)
        
        for interaction in interactions:
            concepts = interaction.get("concepts", [])
            success = interaction.get("success", False)
            
            for concept in concepts:
                concept_performance[concept].append(1.0 if success else 0.0)
        
        # Analyze mastery for each concept
        mastery_analysis = {}
        for concept, performances in concept_performance.items():
            if len(performances) >= 3:
                mastery_score = statistics.mean(performances)
                consistency = 1.0 - statistics.stdev(performances)
                
                mastery_analysis[concept] = {
                    "mastery_score": mastery_score,
                    "consistency": consistency,
                    "attempts": len(performances),
                    "mastery_level": self._determine_mastery_level(mastery_score, consistency)
                }
        
        # Identify mastered and struggling concepts
        mastered_concepts = [
            concept for concept, data in mastery_analysis.items()
            if data["mastery_score"] > 0.8 and data["consistency"] > 0.7
        ]
        
        struggling_concepts = [
            concept for concept, data in mastery_analysis.items()
            if data["mastery_score"] < 0.5 or data["consistency"] < 0.3
        ]
        
        return {
            "pattern": "analyzed",
            "mastery_analysis": mastery_analysis,
            "mastered_concepts": mastered_concepts,
            "struggling_concepts": struggling_concepts,
            "overall_mastery_rate": statistics.mean([data["mastery_score"] for data in mastery_analysis.values()]) if mastery_analysis else 0.5,
            "concepts_analyzed": len(mastery_analysis)
        }
    
    async def _analyze_session_patterns(self, user_id: str, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze session optimization patterns"""
        if not interactions:
            return {"pattern": "insufficient_data"}
        
        session_data = defaultdict(list)
        
        for interaction in interactions:
            session_length = interaction.get("session_length", 30)
            engagement = interaction.get("engagement_score", 0.5)
            performance = 1.0 if interaction.get("success", False) else 0.0
            
            session_data[session_length].append({
                "engagement": engagement,
                "performance": performance
            })
        
        # Find optimal session characteristics
        optimal_sessions = []
        for length, data in session_data.items():
            if len(data) >= 2:
                avg_engagement = statistics.mean([d["engagement"] for d in data])
                avg_performance = statistics.mean([d["performance"] for d in data])
                
                optimal_sessions.append({
                    "length": length,
                    "engagement": avg_engagement,
                    "performance": avg_performance,
                    "combined_score": (avg_engagement + avg_performance) / 2
                })
        
        # Sort by combined score
        optimal_sessions.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return {
            "pattern": "analyzed",
            "optimal_sessions": optimal_sessions[:3],
            "recommended_session_length": optimal_sessions[0]["length"] if optimal_sessions else 30,
            "session_variety": len(set(interaction.get("session_length", 30) for interaction in interactions)),
            "total_sessions_analyzed": len(interactions)
        }
    
    async def _analyze_struggle_recovery(self, user_id: str, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze struggle and recovery patterns"""
        if len(interactions) < 10:
            return {"pattern": "insufficient_data"}
        
        struggle_episodes = []
        current_struggle = None
        
        for i, interaction in enumerate(interactions):
            success = interaction.get("success", False)
            difficulty = interaction.get("difficulty", 0.5)
            response_time = interaction.get("response_time", 5.0)
            
            # Detect struggle (failure + high difficulty or long response time)
            is_struggling = not success and (difficulty > 0.6 or response_time > 10.0)
            
            if is_struggling:
                if current_struggle is None:
                    current_struggle = {
                        "start_index": i,
                        "struggles": [interaction],
                        "difficulty_sum": difficulty
                    }
                else:
                    current_struggle["struggles"].append(interaction)
                    current_struggle["difficulty_sum"] += difficulty
            else:
                if current_struggle is not None:
                    # End of struggle episode
                    current_struggle["end_index"] = i - 1
                    current_struggle["duration"] = len(current_struggle["struggles"])
                    current_struggle["recovery_interaction"] = interaction
                    struggle_episodes.append(current_struggle)
                    current_struggle = None
        
        # Analyze recovery patterns
        recovery_analysis = {
            "total_episodes": len(struggle_episodes),
            "average_duration": statistics.mean([ep["duration"] for ep in struggle_episodes]) if struggle_episodes else 0,
            "recovery_success_rate": sum(1 for ep in struggle_episodes if ep.get("recovery_interaction", {}).get("success", False)) / max(len(struggle_episodes), 1),
            "common_struggle_difficulties": []
        }
        
        if struggle_episodes:
            difficulties = [ep["difficulty_sum"] / ep["duration"] for ep in struggle_episodes]
            recovery_analysis["average_struggle_difficulty"] = statistics.mean(difficulties)
        
        return {
            "pattern": "analyzed",
            "struggle_episodes": len(struggle_episodes),
            "recovery_analysis": recovery_analysis,
            "resilience_score": 1.0 - (len(struggle_episodes) / len(interactions)) if interactions else 0.5
        }
    
    async def _analyze_breakthrough_patterns(self, user_id: str, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze breakthrough learning patterns"""
        if len(interactions) < 15:
            return {"pattern": "insufficient_data"}
        
        breakthroughs = []
        
        for i in range(5, len(interactions)):
            current_success = interactions[i].get("success", False)
            recent_failures = sum(1 for j in range(max(0, i-5), i) if not interactions[j].get("success", True))
            
            # Breakthrough: success after multiple failures
            if current_success and recent_failures >= 3:
                breakthrough = {
                    "index": i,
                    "prior_failures": recent_failures,
                    "difficulty": interactions[i].get("difficulty", 0.5),
                    "concept": interactions[i].get("concepts", ["unknown"])[0] if interactions[i].get("concepts") else "unknown",
                    "response_time": interactions[i].get("response_time", 5.0)
                }
                breakthroughs.append(breakthrough)
        
        # Analyze breakthrough characteristics
        if breakthroughs:
            avg_difficulty = statistics.mean([b["difficulty"] for b in breakthroughs])
            avg_prior_failures = statistics.mean([b["prior_failures"] for b in breakthroughs])
            breakthrough_concepts = [b["concept"] for b in breakthroughs]
        else:
            avg_difficulty = 0.5
            avg_prior_failures = 0
            breakthrough_concepts = []
        
        return {
            "pattern": "analyzed",
            "breakthrough_count": len(breakthroughs),
            "breakthrough_rate": len(breakthroughs) / len(interactions) if interactions else 0,
            "average_difficulty": avg_difficulty,
            "average_prior_failures": avg_prior_failures,
            "breakthrough_concepts": breakthrough_concepts,
            "persistence_indicator": avg_prior_failures if breakthroughs else 0
        }
    
    async def _analyze_metacognitive_growth(self, user_id: str, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze metacognitive development patterns"""
        if not interactions:
            return {"pattern": "insufficient_data"}
        
        metacognitive_indicators = []
        
        for interaction in interactions:
            user_message = interaction.get("user_message", "").lower()
            
            # Count metacognitive indicators
            indicators = 0
            
            # Self-awareness indicators
            if any(phrase in user_message for phrase in ["i think", "i believe", "i understand", "i realize"]):
                indicators += 1
            
            # Strategy awareness
            if any(word in user_message for word in ["strategy", "approach", "method", "way"]):
                indicators += 1
            
            # Learning awareness
            if any(phrase in user_message for phrase in ["i learned", "now i see", "makes sense", "i get it"]):
                indicators += 1
            
            # Reflection indicators
            if any(word in user_message for word in ["because", "since", "therefore", "so"]):
                indicators += 1
            
            metacognitive_indicators.append(indicators)
        
        # Analyze growth trend
        if len(metacognitive_indicators) >= 10:
            first_half = metacognitive_indicators[:len(metacognitive_indicators)//2]
            second_half = metacognitive_indicators[len(metacognitive_indicators)//2:]
            
            first_avg = statistics.mean(first_half)
            second_avg = statistics.mean(second_half)
            
            growth_rate = (second_avg - first_avg) / max(first_avg, 0.1)
        else:
            growth_rate = 0
        
        return {
            "pattern": "analyzed",
            "average_metacognitive_score": statistics.mean(metacognitive_indicators) if metacognitive_indicators else 0,
            "growth_rate": growth_rate,
            "development_trend": "improving" if growth_rate > 0.1 else "stable" if growth_rate > -0.1 else "declining",
            "total_interactions_analyzed": len(interactions)
        }
    
    # Helper methods
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values"""
        if len(values) < 3:
            return "insufficient_data"
        
        # Simple linear trend calculation
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum) if (n * x2_sum - x_sum * x_sum) != 0 else 0
        
        if slope > self.trend_sensitivity:
            return "increasing"
        elif slope < -self.trend_sensitivity:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_improvement_rate(self, performances: List[float]) -> float:
        """Calculate improvement rate from performance data"""
        if len(performances) < 5:
            return 0.0
        
        first_quarter = performances[:len(performances)//4] if len(performances) >= 4 else performances[:1]
        last_quarter = performances[-len(performances)//4:] if len(performances) >= 4 else performances[-1:]
        
        first_avg = statistics.mean(first_quarter)
        last_avg = statistics.mean(last_quarter)
        
        return (last_avg - first_avg) / max(first_avg, 0.1)
    
    def _determine_mastery_level(self, mastery_score: float, consistency: float) -> str:
        """Determine mastery level from score and consistency"""
        if mastery_score > 0.9 and consistency > 0.8:
            return "expert"
        elif mastery_score > 0.8 and consistency > 0.7:
            return "proficient"
        elif mastery_score > 0.6 and consistency > 0.5:
            return "developing"
        elif mastery_score > 0.4:
            return "novice"
        else:
            return "struggling"
    
    def _calculate_pattern_confidence(self, patterns: Dict[str, Any], data_points: int) -> float:
        """Calculate overall confidence in pattern analysis"""
        # Base confidence on amount of data
        data_confidence = min(1.0, data_points / 50)
        
        # Adjust based on pattern consistency
        pattern_scores = []
        for pattern_type, pattern_data in patterns.items():
            if isinstance(pattern_data, dict) and "pattern" in pattern_data:
                if pattern_data["pattern"] == "analyzed":
                    pattern_scores.append(1.0)
                elif pattern_data["pattern"] == "insufficient_data":
                    pattern_scores.append(0.0)
                else:
                    pattern_scores.append(0.5)
        
        pattern_confidence = statistics.mean(pattern_scores) if pattern_scores else 0.5
        
        return (data_confidence + pattern_confidence) / 2
    
    # Default fallback methods
    
    def _get_default_pattern_analysis(self) -> Dict[str, Any]:
        """Get default pattern analysis for fallback"""
        return {
            "user_id": "unknown",
            "analysis_period": "0 days",
            "data_points": 0,
            "patterns": {
                "temporal_patterns": {"pattern": "insufficient_data"},
                "performance_patterns": {"pattern": "insufficient_data"},
                "engagement_patterns": {"pattern": "insufficient_data"},
                "difficulty_progression": {"pattern": "insufficient_data"},
                "learning_velocity_trends": {"pattern": "insufficient_data"},
                "concept_mastery_patterns": {"pattern": "insufficient_data"},
                "session_optimization": {"pattern": "insufficient_data"},
                "struggle_recovery_patterns": {"pattern": "insufficient_data"},
                "breakthrough_indicators": {"pattern": "insufficient_data"},
                "metacognitive_development": {"pattern": "insufficient_data"}
            },
            "insights": [],
            "confidence": 0.0,
            "recommendations": []
        }
    
    def _get_default_predictions(self) -> Dict[str, Any]:
        """Get default predictions for fallback"""
        return {
            "user_id": "unknown",
            "prediction_horizon": "7 days",
            "learning_path_length": 0,
            "step_predictions": [],
            "overall_predictions": {
                "success_probability": 0.7,
                "completion_probability": 0.8,
                "engagement_prediction": 0.7,
                "overall_confidence": 0.5
            },
            "optimizations": [],
            "confidence": 0.5
        }
    
    def _get_default_insights(self) -> Dict[str, Any]:
        """Get default insights for fallback"""
        return {
            "user_id": "unknown",
            "analysis_depth": "basic",
            "insights": {
                "learning_strengths": ["Consistent engagement"],
                "improvement_opportunities": ["More practice needed"],
                "optimal_learning_conditions": ["Regular sessions"],
                "personalization_recommendations": ["Continue current approach"],
                "learning_trajectory": "stable",
                "mastery_predictions": {"timeline": "unknown"},
                "adaptive_strategies": ["Monitor progress"]
            },
            "confidence": 0.5,
            "generated_at": datetime.utcnow().isoformat()
        }
