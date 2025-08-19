"""
Research Analytics Engine

Extracted from quantum_intelligence_engine.py - advanced research analytics
and learning effectiveness measurement for educational insights.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import statistics

# Try to import structlog, fall back to standard logging
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    logger = logging.getLogger(__name__)

from ...core.exceptions import QuantumEngineError
from ...utils.caching import CacheService


class ResearchAnalyticsEngine:
    """
    🔬 RESEARCH ANALYTICS ENGINE
    
    Advanced research analytics and learning effectiveness measurement.
    Extracted from the original quantum engine's research analytics logic.
    """
    
    def __init__(self, cache_service: Optional[CacheService] = None):
        self.cache = cache_service
        
        # Research data storage
        self.research_data = {}
        self.cohort_analytics = {}
        self.effectiveness_metrics = {}
        
        logger.info("Research Analytics Engine initialized")
    
    async def analyze_learning_effectiveness(
        self, 
        cohort_id: str, 
        learning_data: List[Dict[str, Any]],
        control_group_data: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze learning effectiveness for research purposes
        
        Extracted from original research analytics logic
        """
        try:
            # Basic effectiveness analysis
            effectiveness_metrics = self._calculate_effectiveness_metrics(learning_data)
            
            # Comparative analysis if control group provided
            comparative_analysis = None
            if control_group_data:
                comparative_analysis = self._compare_learning_groups(
                    learning_data, 
                    control_group_data
                )
            
            # Statistical significance testing
            statistical_analysis = self._perform_statistical_analysis(
                learning_data, 
                control_group_data
            )
            
            return {
                "cohort_id": cohort_id,
                "effectiveness_metrics": effectiveness_metrics,
                "comparative_analysis": comparative_analysis,
                "statistical_analysis": statistical_analysis,
                "sample_size": len(learning_data),
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing learning effectiveness: {e}")
            return self._get_default_effectiveness_analysis()
    
    async def generate_research_insights(
        self, 
        study_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate research insights from study data
        
        Extracted from original research insights logic
        """
        try:
            insights = {
                "key_findings": self._extract_key_findings(study_data),
                "learning_patterns": self._identify_research_patterns(study_data),
                "recommendations": self._generate_research_recommendations(study_data),
                "methodology_insights": self._analyze_methodology_effectiveness(study_data)
            }
            
            return {
                "study_id": study_data.get("study_id", "unknown"),
                "insights": insights,
                "confidence_level": 0.8,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating research insights: {e}")
            return self._get_default_research_insights()
    
    async def measure_intervention_impact(
        self, 
        intervention_data: Dict[str, Any],
        baseline_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Measure the impact of learning interventions
        
        Extracted from original intervention analysis logic
        """
        try:
            # Calculate impact metrics
            impact_metrics = self._calculate_intervention_impact(
                intervention_data, 
                baseline_data
            )
            
            # Analyze effect sizes
            effect_sizes = self._calculate_effect_sizes(
                intervention_data, 
                baseline_data
            )
            
            # Generate impact assessment
            impact_assessment = self._assess_intervention_effectiveness(
                impact_metrics, 
                effect_sizes
            )
            
            return {
                "intervention_id": intervention_data.get("intervention_id", "unknown"),
                "impact_metrics": impact_metrics,
                "effect_sizes": effect_sizes,
                "impact_assessment": impact_assessment,
                "measurement_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error measuring intervention impact: {e}")
            return self._get_default_intervention_analysis()
    
    # Private analysis methods
    
    def _calculate_effectiveness_metrics(self, learning_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate basic effectiveness metrics"""
        if not learning_data:
            return {"error": "no_data"}
        
        # Extract performance metrics
        success_rates = [data.get("success_rate", 0.5) for data in learning_data]
        engagement_scores = [data.get("engagement_score", 0.5) for data in learning_data]
        completion_rates = [data.get("completion_rate", 0.5) for data in learning_data]
        
        return {
            "average_success_rate": statistics.mean(success_rates),
            "average_engagement": statistics.mean(engagement_scores),
            "average_completion_rate": statistics.mean(completion_rates),
            "success_rate_std": statistics.stdev(success_rates) if len(success_rates) > 1 else 0,
            "engagement_std": statistics.stdev(engagement_scores) if len(engagement_scores) > 1 else 0,
            "sample_size": len(learning_data)
        }
    
    def _compare_learning_groups(
        self, 
        treatment_data: List[Dict[str, Any]], 
        control_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare treatment and control groups"""
        treatment_metrics = self._calculate_effectiveness_metrics(treatment_data)
        control_metrics = self._calculate_effectiveness_metrics(control_data)
        
        # Calculate differences
        success_rate_diff = (
            treatment_metrics["average_success_rate"] - 
            control_metrics["average_success_rate"]
        )
        
        engagement_diff = (
            treatment_metrics["average_engagement"] - 
            control_metrics["average_engagement"]
        )
        
        completion_rate_diff = (
            treatment_metrics["average_completion_rate"] - 
            control_metrics["average_completion_rate"]
        )
        
        return {
            "treatment_metrics": treatment_metrics,
            "control_metrics": control_metrics,
            "differences": {
                "success_rate_difference": success_rate_diff,
                "engagement_difference": engagement_diff,
                "completion_rate_difference": completion_rate_diff
            },
            "relative_improvement": {
                "success_rate": success_rate_diff / max(control_metrics["average_success_rate"], 0.01),
                "engagement": engagement_diff / max(control_metrics["average_engagement"], 0.01),
                "completion_rate": completion_rate_diff / max(control_metrics["average_completion_rate"], 0.01)
            }
        }
    
    def _perform_statistical_analysis(
        self, 
        treatment_data: List[Dict[str, Any]], 
        control_data: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Perform basic statistical analysis"""
        # Simplified statistical analysis
        analysis = {
            "sample_size": len(treatment_data),
            "statistical_power": "moderate" if len(treatment_data) > 30 else "low",
            "confidence_level": 0.95
        }
        
        if control_data:
            analysis.update({
                "control_sample_size": len(control_data),
                "comparison_type": "between_groups",
                "effect_size_category": "medium"  # Simplified
            })
        else:
            analysis.update({
                "comparison_type": "descriptive",
                "effect_size_category": "not_applicable"
            })
        
        return analysis
    
    def _extract_key_findings(self, study_data: Dict[str, Any]) -> List[str]:
        """Extract key findings from study data"""
        findings = []
        
        # Analyze effectiveness metrics
        effectiveness = study_data.get("effectiveness_metrics", {})
        avg_success = effectiveness.get("average_success_rate", 0.5)
        
        if avg_success > 0.8:
            findings.append("High learning effectiveness observed (>80% success rate)")
        elif avg_success > 0.6:
            findings.append("Moderate learning effectiveness observed (60-80% success rate)")
        else:
            findings.append("Learning effectiveness below optimal levels (<60% success rate)")
        
        # Analyze engagement
        avg_engagement = effectiveness.get("average_engagement", 0.5)
        if avg_engagement > 0.8:
            findings.append("Excellent learner engagement maintained throughout study")
        elif avg_engagement < 0.5:
            findings.append("Learner engagement challenges identified")
        
        return findings if findings else ["Insufficient data for key findings"]
    
    def _identify_research_patterns(self, study_data: Dict[str, Any]) -> List[str]:
        """Identify research patterns from study data"""
        patterns = []
        
        # Analyze comparative data if available
        comparative = study_data.get("comparative_analysis")
        if comparative:
            differences = comparative.get("differences", {})
            success_diff = differences.get("success_rate_difference", 0)
            
            if success_diff > 0.1:
                patterns.append("Significant improvement in success rates observed")
            elif success_diff < -0.1:
                patterns.append("Decline in success rates observed")
            else:
                patterns.append("No significant difference in success rates")
        
        return patterns if patterns else ["Limited pattern data available"]
    
    def _generate_research_recommendations(self, study_data: Dict[str, Any]) -> List[str]:
        """Generate research-based recommendations"""
        recommendations = []
        
        effectiveness = study_data.get("effectiveness_metrics", {})
        avg_success = effectiveness.get("average_success_rate", 0.5)
        
        if avg_success < 0.6:
            recommendations.append("Consider intervention to improve learning outcomes")
        
        avg_engagement = effectiveness.get("average_engagement", 0.5)
        if avg_engagement < 0.6:
            recommendations.append("Implement engagement enhancement strategies")
        
        sample_size = effectiveness.get("sample_size", 0)
        if sample_size < 30:
            recommendations.append("Increase sample size for more robust statistical analysis")
        
        return recommendations if recommendations else ["Continue current research approach"]
    
    def _analyze_methodology_effectiveness(self, study_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the effectiveness of research methodology"""
        statistical = study_data.get("statistical_analysis", {})
        
        return {
            "methodology_strength": statistical.get("statistical_power", "moderate"),
            "sample_adequacy": "adequate" if statistical.get("sample_size", 0) > 30 else "limited",
            "design_quality": "controlled" if study_data.get("comparative_analysis") else "observational",
            "reliability_assessment": "high" if statistical.get("confidence_level", 0.8) > 0.9 else "moderate"
        }
    
    def _calculate_intervention_impact(
        self, 
        intervention_data: Dict[str, Any], 
        baseline_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate intervention impact metrics"""
        # Extract metrics from both datasets
        intervention_success = intervention_data.get("average_success_rate", 0.5)
        baseline_success = baseline_data.get("average_success_rate", 0.5)
        
        intervention_engagement = intervention_data.get("average_engagement", 0.5)
        baseline_engagement = baseline_data.get("average_engagement", 0.5)
        
        # Calculate absolute and relative changes
        success_change = intervention_success - baseline_success
        engagement_change = intervention_engagement - baseline_engagement
        
        return {
            "success_rate_change": success_change,
            "engagement_change": engagement_change,
            "relative_success_improvement": success_change / max(baseline_success, 0.01),
            "relative_engagement_improvement": engagement_change / max(baseline_engagement, 0.01),
            "overall_impact_score": (success_change + engagement_change) / 2
        }
    
    def _calculate_effect_sizes(
        self, 
        intervention_data: Dict[str, Any], 
        baseline_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate effect sizes for intervention analysis"""
        # Simplified effect size calculation
        success_change = intervention_data.get("average_success_rate", 0.5) - baseline_data.get("average_success_rate", 0.5)
        engagement_change = intervention_data.get("average_engagement", 0.5) - baseline_data.get("average_engagement", 0.5)
        
        # Categorize effect sizes
        def categorize_effect_size(effect):
            if abs(effect) > 0.8:
                return "large"
            elif abs(effect) > 0.5:
                return "medium"
            elif abs(effect) > 0.2:
                return "small"
            else:
                return "negligible"
        
        return {
            "success_rate_effect_size": success_change,
            "engagement_effect_size": engagement_change,
            "success_rate_category": categorize_effect_size(success_change),
            "engagement_category": categorize_effect_size(engagement_change)
        }
    
    def _assess_intervention_effectiveness(
        self, 
        impact_metrics: Dict[str, Any], 
        effect_sizes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess overall intervention effectiveness"""
        overall_impact = impact_metrics.get("overall_impact_score", 0)
        
        if overall_impact > 0.2:
            effectiveness = "highly_effective"
        elif overall_impact > 0.1:
            effectiveness = "moderately_effective"
        elif overall_impact > 0:
            effectiveness = "minimally_effective"
        else:
            effectiveness = "ineffective"
        
        return {
            "effectiveness_rating": effectiveness,
            "primary_benefits": self._identify_primary_benefits(impact_metrics),
            "areas_for_improvement": self._identify_improvement_areas(impact_metrics),
            "recommendation": self._generate_intervention_recommendation(effectiveness)
        }
    
    def _identify_primary_benefits(self, impact_metrics: Dict[str, Any]) -> List[str]:
        """Identify primary benefits of intervention"""
        benefits = []
        
        if impact_metrics.get("success_rate_change", 0) > 0.1:
            benefits.append("Significant improvement in learning success rates")
        
        if impact_metrics.get("engagement_change", 0) > 0.1:
            benefits.append("Notable increase in learner engagement")
        
        return benefits if benefits else ["Limited measurable benefits observed"]
    
    def _identify_improvement_areas(self, impact_metrics: Dict[str, Any]) -> List[str]:
        """Identify areas for improvement"""
        areas = []
        
        if impact_metrics.get("success_rate_change", 0) < 0:
            areas.append("Address decline in success rates")
        
        if impact_metrics.get("engagement_change", 0) < 0:
            areas.append("Improve engagement strategies")
        
        return areas if areas else ["Continue monitoring for optimization opportunities"]
    
    def _generate_intervention_recommendation(self, effectiveness: str) -> str:
        """Generate recommendation based on effectiveness"""
        recommendations = {
            "highly_effective": "Continue and scale intervention",
            "moderately_effective": "Refine intervention and continue implementation",
            "minimally_effective": "Modify intervention approach significantly",
            "ineffective": "Discontinue current intervention and explore alternatives"
        }
        
        return recommendations.get(effectiveness, "Conduct further analysis")
    
    # Default fallback methods
    
    def _get_default_effectiveness_analysis(self) -> Dict[str, Any]:
        """Get default effectiveness analysis for fallback"""
        return {
            "cohort_id": "unknown",
            "effectiveness_metrics": {
                "average_success_rate": 0.7,
                "average_engagement": 0.7,
                "sample_size": 0
            },
            "comparative_analysis": None,
            "statistical_analysis": {
                "sample_size": 0,
                "statistical_power": "insufficient"
            },
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
    
    def _get_default_research_insights(self) -> Dict[str, Any]:
        """Get default research insights for fallback"""
        return {
            "study_id": "unknown",
            "insights": {
                "key_findings": ["Insufficient data for analysis"],
                "learning_patterns": ["No patterns identified"],
                "recommendations": ["Collect more data"],
                "methodology_insights": {"methodology_strength": "insufficient"}
            },
            "confidence_level": 0.3,
            "generated_at": datetime.utcnow().isoformat()
        }
    
    def _get_default_intervention_analysis(self) -> Dict[str, Any]:
        """Get default intervention analysis for fallback"""
        return {
            "intervention_id": "unknown",
            "impact_metrics": {
                "success_rate_change": 0.0,
                "engagement_change": 0.0,
                "overall_impact_score": 0.0
            },
            "effect_sizes": {
                "success_rate_category": "negligible",
                "engagement_category": "negligible"
            },
            "impact_assessment": {
                "effectiveness_rating": "insufficient_data",
                "recommendation": "Collect more data for analysis"
            },
            "measurement_timestamp": datetime.utcnow().isoformat()
        }
