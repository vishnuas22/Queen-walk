"""
Cognitive Load Assessment Engine

Extracted from quantum_intelligence_engine.py - cognitive load measurement
and optimization for enhanced learning experiences.
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

from ...core.exceptions import QuantumEngineError
from ...utils.caching import CacheService


class CognitiveLoadAssessmentEngine:
    """
    🧠 COGNITIVE LOAD ASSESSMENT ENGINE
    
    Cognitive load measurement and optimization for enhanced learning.
    Extracted from the original quantum engine's cognitive load logic.
    """
    
    def __init__(self, cache_service: Optional[CacheService] = None):
        self.cache = cache_service
        
        # Cognitive load tracking
        self.load_measurements = {}
        self.load_patterns = {}
        self.optimization_history = {}
        
        # Load assessment parameters
        self.load_threshold_high = 0.8
        self.load_threshold_low = 0.3
        self.measurement_window = 10  # Number of interactions to analyze
        
        logger.info("Cognitive Load Assessment Engine initialized")
    
    async def assess_cognitive_load(
        self, 
        user_id: str, 
        interaction_data: Dict[str, Any],
        content_characteristics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess cognitive load from interaction data
        
        Extracted from original cognitive load assessment logic
        """
        try:
            # Calculate intrinsic cognitive load
            intrinsic_load = self._calculate_intrinsic_load(content_characteristics)
            
            # Calculate extraneous cognitive load
            extraneous_load = self._calculate_extraneous_load(
                interaction_data, 
                content_characteristics
            )
            
            # Calculate germane cognitive load
            germane_load = self._calculate_germane_load(interaction_data)
            
            # Calculate total cognitive load
            total_load = self._calculate_total_load(
                intrinsic_load, 
                extraneous_load, 
                germane_load
            )
            
            # Assess load level
            load_assessment = self._assess_load_level(total_load)
            
            # Generate load insights
            insights = self._generate_load_insights(
                intrinsic_load, 
                extraneous_load, 
                germane_load, 
                total_load
            )
            
            # Store measurement
            self._store_load_measurement(user_id, {
                "intrinsic_load": intrinsic_load,
                "extraneous_load": extraneous_load,
                "germane_load": germane_load,
                "total_load": total_load,
                "assessment": load_assessment,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return {
                "user_id": user_id,
                "cognitive_load_breakdown": {
                    "intrinsic_load": intrinsic_load,
                    "extraneous_load": extraneous_load,
                    "germane_load": germane_load,
                    "total_load": total_load
                },
                "load_assessment": load_assessment,
                "insights": insights,
                "measurement_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error assessing cognitive load for user {user_id}: {e}")
            return self._get_default_load_assessment()
    
    async def optimize_cognitive_load(
        self, 
        user_id: str, 
        current_load: Dict[str, Any],
        learning_objectives: List[str]
    ) -> Dict[str, Any]:
        """
        Generate cognitive load optimization recommendations
        
        Extracted from original load optimization logic
        """
        try:
            # Analyze current load distribution
            load_analysis = self._analyze_load_distribution(current_load)
            
            # Identify optimization opportunities
            optimization_opportunities = self._identify_optimization_opportunities(
                current_load, 
                load_analysis
            )
            
            # Generate specific recommendations
            recommendations = self._generate_load_recommendations(
                optimization_opportunities, 
                learning_objectives
            )
            
            # Predict optimization impact
            impact_prediction = self._predict_optimization_impact(
                current_load, 
                recommendations
            )
            
            return {
                "user_id": user_id,
                "current_load_analysis": load_analysis,
                "optimization_opportunities": optimization_opportunities,
                "recommendations": recommendations,
                "predicted_impact": impact_prediction,
                "optimization_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error optimizing cognitive load for user {user_id}: {e}")
            return self._get_default_optimization()
    
    async def monitor_load_patterns(
        self, 
        user_id: str, 
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Monitor cognitive load patterns over time
        
        Extracted from original load monitoring logic
        """
        try:
            # Get recent load measurements
            recent_measurements = self._get_recent_measurements(user_id, time_window_hours)
            
            if not recent_measurements:
                return {"pattern": "insufficient_data"}
            
            # Analyze load trends
            load_trends = self._analyze_load_trends(recent_measurements)
            
            # Identify load patterns
            load_patterns = self._identify_load_patterns(recent_measurements)
            
            # Detect load anomalies
            anomalies = self._detect_load_anomalies(recent_measurements)
            
            # Generate pattern insights
            pattern_insights = self._generate_pattern_insights(
                load_trends, 
                load_patterns, 
                anomalies
            )
            
            return {
                "user_id": user_id,
                "monitoring_window_hours": time_window_hours,
                "measurements_count": len(recent_measurements),
                "load_trends": load_trends,
                "load_patterns": load_patterns,
                "anomalies": anomalies,
                "pattern_insights": pattern_insights,
                "monitoring_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error monitoring load patterns for user {user_id}: {e}")
            return {"pattern": "error", "error": str(e)}
    
    # Private assessment methods
    
    def _calculate_intrinsic_load(self, content_characteristics: Dict[str, Any]) -> float:
        """Calculate intrinsic cognitive load based on content complexity"""
        # Extract content complexity factors
        difficulty = content_characteristics.get("difficulty", 0.5)
        concept_count = content_characteristics.get("concept_count", 1)
        abstraction_level = content_characteristics.get("abstraction_level", 0.5)
        prerequisite_complexity = content_characteristics.get("prerequisite_complexity", 0.3)
        
        # Calculate intrinsic load components
        difficulty_load = difficulty * 0.4
        concept_load = min(0.3, concept_count * 0.05)  # Cap at 0.3
        abstraction_load = abstraction_level * 0.2
        prerequisite_load = prerequisite_complexity * 0.1
        
        intrinsic_load = difficulty_load + concept_load + abstraction_load + prerequisite_load
        
        return min(1.0, max(0.0, intrinsic_load))
    
    def _calculate_extraneous_load(
        self, 
        interaction_data: Dict[str, Any], 
        content_characteristics: Dict[str, Any]
    ) -> float:
        """Calculate extraneous cognitive load from interface and presentation"""
        # Extract interaction complexity factors
        response_time = interaction_data.get("response_time", 5.0)
        interface_complexity = content_characteristics.get("interface_complexity", 0.3)
        distraction_level = interaction_data.get("distraction_indicators", 0.2)
        navigation_difficulty = content_characteristics.get("navigation_difficulty", 0.2)
        
        # Calculate extraneous load components
        time_pressure_load = min(0.3, max(0.0, (response_time - 5.0) / 10.0))  # Normalize response time
        interface_load = interface_complexity * 0.25
        distraction_load = distraction_level * 0.25
        navigation_load = navigation_difficulty * 0.2
        
        extraneous_load = time_pressure_load + interface_load + distraction_load + navigation_load
        
        return min(1.0, max(0.0, extraneous_load))
    
    def _calculate_germane_load(self, interaction_data: Dict[str, Any]) -> float:
        """Calculate germane cognitive load from learning processing"""
        # Extract learning processing indicators
        engagement_score = interaction_data.get("engagement_score", 0.5)
        metacognitive_activity = interaction_data.get("metacognitive_indicators", 0.3)
        schema_construction = interaction_data.get("schema_construction_indicators", 0.4)
        transfer_attempts = interaction_data.get("transfer_attempts", 0.2)
        
        # Calculate germane load (positive cognitive effort)
        engagement_load = engagement_score * 0.3
        metacognitive_load = metacognitive_activity * 0.3
        schema_load = schema_construction * 0.25
        transfer_load = transfer_attempts * 0.15
        
        germane_load = engagement_load + metacognitive_load + schema_load + transfer_load
        
        return min(1.0, max(0.0, germane_load))
    
    def _calculate_total_load(
        self, 
        intrinsic_load: float, 
        extraneous_load: float, 
        germane_load: float
    ) -> float:
        """Calculate total cognitive load"""
        # Total load is weighted sum (intrinsic and extraneous are costs, germane is beneficial)
        total_load = (intrinsic_load * 0.5) + (extraneous_load * 0.4) + (germane_load * 0.1)
        
        return min(1.0, max(0.0, total_load))
    
    def _assess_load_level(self, total_load: float) -> Dict[str, Any]:
        """Assess cognitive load level and provide interpretation"""
        if total_load > self.load_threshold_high:
            level = "high"
            interpretation = "Cognitive overload risk - consider reducing complexity"
            urgency = "high"
        elif total_load < self.load_threshold_low:
            level = "low"
            interpretation = "Underutilized cognitive capacity - can increase challenge"
            urgency = "low"
        else:
            level = "optimal"
            interpretation = "Cognitive load within optimal range"
            urgency = "none"
        
        return {
            "level": level,
            "score": total_load,
            "interpretation": interpretation,
            "urgency": urgency,
            "threshold_high": self.load_threshold_high,
            "threshold_low": self.load_threshold_low
        }
    
    def _generate_load_insights(
        self, 
        intrinsic_load: float, 
        extraneous_load: float, 
        germane_load: float, 
        total_load: float
    ) -> List[str]:
        """Generate insights about cognitive load distribution"""
        insights = []
        
        # Analyze load distribution
        if intrinsic_load > 0.7:
            insights.append("High intrinsic load - content complexity may be challenging")
        
        if extraneous_load > 0.5:
            insights.append("High extraneous load - interface or presentation issues detected")
        
        if germane_load < 0.3:
            insights.append("Low germane load - limited deep learning processing detected")
        elif germane_load > 0.7:
            insights.append("High germane load - excellent deep learning engagement")
        
        # Overall load insights
        if total_load > 0.8:
            insights.append("Total cognitive load is very high - immediate intervention recommended")
        elif total_load < 0.3:
            insights.append("Total cognitive load is low - opportunity to increase challenge")
        
        return insights if insights else ["Cognitive load within normal parameters"]
    
    def _store_load_measurement(self, user_id: str, measurement: Dict[str, Any]):
        """Store cognitive load measurement for user"""
        if user_id not in self.load_measurements:
            self.load_measurements[user_id] = []
        
        self.load_measurements[user_id].append(measurement)
        
        # Keep only recent measurements
        if len(self.load_measurements[user_id]) > 100:
            self.load_measurements[user_id] = self.load_measurements[user_id][-100:]
    
    # Optimization methods
    
    def _analyze_load_distribution(self, current_load: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current cognitive load distribution"""
        breakdown = current_load.get("cognitive_load_breakdown", {})
        
        intrinsic = breakdown.get("intrinsic_load", 0.5)
        extraneous = breakdown.get("extraneous_load", 0.3)
        germane = breakdown.get("germane_load", 0.4)
        total = breakdown.get("total_load", 0.5)
        
        # Analyze proportions
        if total > 0:
            intrinsic_proportion = intrinsic / total
            extraneous_proportion = extraneous / total
            germane_proportion = germane / total
        else:
            intrinsic_proportion = extraneous_proportion = germane_proportion = 0.33
        
        return {
            "load_breakdown": breakdown,
            "proportions": {
                "intrinsic_proportion": intrinsic_proportion,
                "extraneous_proportion": extraneous_proportion,
                "germane_proportion": germane_proportion
            },
            "dominant_load_type": max(
                [("intrinsic", intrinsic), ("extraneous", extraneous), ("germane", germane)],
                key=lambda x: x[1]
            )[0]
        }
    
    def _identify_optimization_opportunities(
        self, 
        current_load: Dict[str, Any], 
        load_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify cognitive load optimization opportunities"""
        opportunities = []
        
        breakdown = load_analysis["load_breakdown"]
        proportions = load_analysis["proportions"]
        
        # High extraneous load opportunity
        if breakdown.get("extraneous_load", 0) > 0.5:
            opportunities.append({
                "type": "reduce_extraneous_load",
                "priority": "high",
                "current_value": breakdown["extraneous_load"],
                "target_value": 0.3,
                "potential_improvement": breakdown["extraneous_load"] - 0.3
            })
        
        # Low germane load opportunity
        if breakdown.get("germane_load", 0) < 0.4:
            opportunities.append({
                "type": "increase_germane_load",
                "priority": "medium",
                "current_value": breakdown["germane_load"],
                "target_value": 0.6,
                "potential_improvement": 0.6 - breakdown["germane_load"]
            })
        
        # High intrinsic load opportunity
        if breakdown.get("intrinsic_load", 0) > 0.8:
            opportunities.append({
                "type": "reduce_intrinsic_load",
                "priority": "high",
                "current_value": breakdown["intrinsic_load"],
                "target_value": 0.6,
                "potential_improvement": breakdown["intrinsic_load"] - 0.6
            })
        
        return opportunities
    
    def _generate_load_recommendations(
        self, 
        opportunities: List[Dict[str, Any]], 
        learning_objectives: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate specific cognitive load recommendations"""
        recommendations = []
        
        for opportunity in opportunities:
            opp_type = opportunity["type"]
            
            if opp_type == "reduce_extraneous_load":
                recommendations.append({
                    "category": "interface_optimization",
                    "action": "Simplify interface design and reduce distractions",
                    "specific_steps": [
                        "Remove unnecessary visual elements",
                        "Improve navigation clarity",
                        "Reduce cognitive overhead in interactions"
                    ],
                    "expected_impact": opportunity["potential_improvement"],
                    "implementation_effort": "medium"
                })
            
            elif opp_type == "increase_germane_load":
                recommendations.append({
                    "category": "learning_enhancement",
                    "action": "Increase meaningful cognitive processing",
                    "specific_steps": [
                        "Add reflection prompts",
                        "Include schema-building activities",
                        "Encourage metacognitive thinking"
                    ],
                    "expected_impact": opportunity["potential_improvement"],
                    "implementation_effort": "low"
                })
            
            elif opp_type == "reduce_intrinsic_load":
                recommendations.append({
                    "category": "content_optimization",
                    "action": "Reduce content complexity temporarily",
                    "specific_steps": [
                        "Break content into smaller chunks",
                        "Provide more scaffolding",
                        "Reduce concept density"
                    ],
                    "expected_impact": opportunity["potential_improvement"],
                    "implementation_effort": "high"
                })
        
        return recommendations
    
    def _predict_optimization_impact(
        self, 
        current_load: Dict[str, Any], 
        recommendations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Predict the impact of optimization recommendations"""
        current_total = current_load.get("cognitive_load_breakdown", {}).get("total_load", 0.5)
        
        # Calculate potential improvement
        total_potential_improvement = sum(
            rec.get("expected_impact", 0) for rec in recommendations
        )
        
        predicted_new_load = max(0.1, current_total - total_potential_improvement)
        
        return {
            "current_total_load": current_total,
            "predicted_new_load": predicted_new_load,
            "total_improvement": total_potential_improvement,
            "improvement_percentage": (total_potential_improvement / max(current_total, 0.1)) * 100,
            "confidence": 0.7  # Moderate confidence in predictions
        }
    
    # Monitoring methods
    
    def _get_recent_measurements(self, user_id: str, hours: int) -> List[Dict[str, Any]]:
        """Get recent cognitive load measurements for user"""
        if user_id not in self.load_measurements:
            return []
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_measurements = []
        
        for measurement in self.load_measurements[user_id]:
            measurement_time = datetime.fromisoformat(measurement["timestamp"])
            if measurement_time > cutoff_time:
                recent_measurements.append(measurement)
        
        return recent_measurements
    
    def _analyze_load_trends(self, measurements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze cognitive load trends over time"""
        if len(measurements) < 3:
            return {"trend": "insufficient_data"}
        
        # Extract load values over time
        total_loads = [m["total_load"] for m in measurements]
        intrinsic_loads = [m["intrinsic_load"] for m in measurements]
        extraneous_loads = [m["extraneous_load"] for m in measurements]
        germane_loads = [m["germane_load"] for m in measurements]
        
        # Calculate trends (simplified linear trend)
        def calculate_trend(values):
            if len(values) < 2:
                return "stable"
            
            first_half = values[:len(values)//2]
            second_half = values[len(values)//2:]
            
            first_avg = statistics.mean(first_half)
            second_avg = statistics.mean(second_half)
            
            change = second_avg - first_avg
            
            if change > 0.1:
                return "increasing"
            elif change < -0.1:
                return "decreasing"
            else:
                return "stable"
        
        return {
            "total_load_trend": calculate_trend(total_loads),
            "intrinsic_load_trend": calculate_trend(intrinsic_loads),
            "extraneous_load_trend": calculate_trend(extraneous_loads),
            "germane_load_trend": calculate_trend(germane_loads),
            "average_total_load": statistics.mean(total_loads),
            "load_variability": statistics.stdev(total_loads) if len(total_loads) > 1 else 0
        }
    
    def _identify_load_patterns(self, measurements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify patterns in cognitive load measurements"""
        if not measurements:
            return {"patterns": "no_data"}
        
        # Analyze load level distribution
        high_load_count = sum(1 for m in measurements if m["total_load"] > self.load_threshold_high)
        low_load_count = sum(1 for m in measurements if m["total_load"] < self.load_threshold_low)
        optimal_load_count = len(measurements) - high_load_count - low_load_count
        
        # Identify dominant pattern
        if high_load_count > len(measurements) * 0.6:
            dominant_pattern = "frequently_overloaded"
        elif low_load_count > len(measurements) * 0.6:
            dominant_pattern = "frequently_underloaded"
        elif optimal_load_count > len(measurements) * 0.6:
            dominant_pattern = "generally_optimal"
        else:
            dominant_pattern = "variable_load"
        
        return {
            "dominant_pattern": dominant_pattern,
            "high_load_frequency": high_load_count / len(measurements),
            "low_load_frequency": low_load_count / len(measurements),
            "optimal_load_frequency": optimal_load_count / len(measurements),
            "total_measurements": len(measurements)
        }
    
    def _detect_load_anomalies(self, measurements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect anomalies in cognitive load measurements"""
        if len(measurements) < 5:
            return []
        
        total_loads = [m["total_load"] for m in measurements]
        mean_load = statistics.mean(total_loads)
        std_load = statistics.stdev(total_loads) if len(total_loads) > 1 else 0
        
        anomalies = []
        
        for i, measurement in enumerate(measurements):
            load = measurement["total_load"]
            
            # Detect outliers (more than 2 standard deviations from mean)
            if std_load > 0 and abs(load - mean_load) > 2 * std_load:
                anomalies.append({
                    "index": i,
                    "timestamp": measurement["timestamp"],
                    "load_value": load,
                    "deviation": abs(load - mean_load),
                    "type": "outlier",
                    "severity": "high" if abs(load - mean_load) > 3 * std_load else "medium"
                })
        
        return anomalies
    
    def _generate_pattern_insights(
        self, 
        trends: Dict[str, Any], 
        patterns: Dict[str, Any], 
        anomalies: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate insights from cognitive load patterns"""
        insights = []
        
        # Trend insights
        total_trend = trends.get("total_load_trend", "stable")
        if total_trend == "increasing":
            insights.append("Cognitive load is increasing over time - monitor for overload")
        elif total_trend == "decreasing":
            insights.append("Cognitive load is decreasing - may indicate adaptation or underchallenge")
        
        # Pattern insights
        dominant_pattern = patterns.get("dominant_pattern", "unknown")
        if dominant_pattern == "frequently_overloaded":
            insights.append("User frequently experiences cognitive overload - reduce complexity")
        elif dominant_pattern == "frequently_underloaded":
            insights.append("User cognitive capacity is underutilized - increase challenge")
        elif dominant_pattern == "generally_optimal":
            insights.append("Cognitive load is generally well-managed")
        
        # Anomaly insights
        if len(anomalies) > 0:
            high_severity_anomalies = [a for a in anomalies if a.get("severity") == "high"]
            if high_severity_anomalies:
                insights.append(f"Detected {len(high_severity_anomalies)} high-severity load anomalies")
        
        return insights if insights else ["Cognitive load patterns within normal range"]
    
    # Default fallback methods
    
    def _get_default_load_assessment(self) -> Dict[str, Any]:
        """Get default cognitive load assessment for fallback"""
        return {
            "user_id": "unknown",
            "cognitive_load_breakdown": {
                "intrinsic_load": 0.5,
                "extraneous_load": 0.3,
                "germane_load": 0.4,
                "total_load": 0.5
            },
            "load_assessment": {
                "level": "optimal",
                "score": 0.5,
                "interpretation": "Default assessment - insufficient data",
                "urgency": "none"
            },
            "insights": ["Insufficient data for cognitive load assessment"],
            "measurement_timestamp": datetime.utcnow().isoformat()
        }
    
    def _get_default_optimization(self) -> Dict[str, Any]:
        """Get default optimization recommendations for fallback"""
        return {
            "user_id": "unknown",
            "current_load_analysis": {
                "dominant_load_type": "unknown"
            },
            "optimization_opportunities": [],
            "recommendations": [{
                "category": "general",
                "action": "Monitor cognitive load patterns",
                "specific_steps": ["Collect more interaction data"],
                "expected_impact": 0.0,
                "implementation_effort": "low"
            }],
            "predicted_impact": {
                "current_total_load": 0.5,
                "predicted_new_load": 0.5,
                "total_improvement": 0.0,
                "confidence": 0.3
            },
            "optimization_timestamp": datetime.utcnow().isoformat()
        }
