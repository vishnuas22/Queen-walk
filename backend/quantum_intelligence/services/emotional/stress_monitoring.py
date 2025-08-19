"""
Stress Monitoring Services

Extracted from quantum_intelligence_engine.py (lines 8204-10287) - advanced stress
monitoring and burnout prevention for optimal learning conditions.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging

# Try to import structlog, fall back to standard logging
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    logger = logging.getLogger(__name__)

from ...core.exceptions import QuantumEngineError
from ...utils.caching import CacheService


@dataclass
class StressLevelData:
    """Comprehensive stress level analysis"""
    overall_stress_level: float = 0.0
    stress_category: str = "low"  # low, moderate, high, critical
    stress_indicators: List[str] = field(default_factory=list)
    physiological_stress: float = 0.0
    cognitive_stress: float = 0.0
    emotional_stress: float = 0.0
    behavioral_stress: float = 0.0
    stress_sources: List[str] = field(default_factory=list)
    coping_mechanisms: List[str] = field(default_factory=list)
    recovery_recommendations: List[str] = field(default_factory=list)
    burnout_risk: float = 0.0


class StressMonitoringSystem:
    """
    📊 STRESS MONITORING SYSTEM
    
    Advanced stress monitoring and analysis for learning optimization.
    Extracted from the original quantum engine's emotional AI logic.
    """
    
    def __init__(self, cache_service: Optional[CacheService] = None):
        self.cache = cache_service
        
        # Monitoring configuration
        self.config = {
            'monitoring_interval': 300,  # 5 minutes
            'stress_threshold_moderate': 0.4,
            'stress_threshold_high': 0.7,
            'stress_threshold_critical': 0.9,
            'burnout_threshold': 0.8
        }
        
        # Stress tracking
        self.stress_history = []
        self.monitoring_active = False
        
        logger.info("Stress Monitoring System initialized")
    
    async def monitor_stress_levels(self,
                                  user_id: str,
                                  monitoring_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Monitor and analyze stress levels for a user
        
        Args:
            user_id: User identifier
            monitoring_data: Data for stress analysis
            
        Returns:
            Dict with comprehensive stress analysis
        """
        try:
            # Analyze current stress level
            stress_analysis = await self._analyze_current_stress(monitoring_data)
            
            # Check stress trends
            stress_trends = await self._analyze_stress_trends(user_id)
            
            # Assess burnout risk
            burnout_assessment = await self._assess_burnout_risk(user_id, stress_analysis)
            
            # Generate recommendations
            recommendations = await self._generate_stress_recommendations(stress_analysis, burnout_assessment)
            
            # Create comprehensive result
            result = {
                'user_id': user_id,
                'stress_analysis': stress_analysis.__dict__,
                'stress_trends': stress_trends,
                'burnout_assessment': burnout_assessment,
                'recommendations': recommendations,
                'monitoring_timestamp': datetime.utcnow().isoformat()
            }
            
            # Store stress history
            self.stress_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'user_id': user_id,
                'stress_level': stress_analysis.overall_stress_level,
                'stress_category': stress_analysis.stress_category
            })
            
            return {
                'status': 'success',
                **result
            }
            
        except Exception as e:
            logger.error(f"Error monitoring stress levels for user {user_id}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _analyze_current_stress(self, monitoring_data: Dict[str, Any]) -> StressLevelData:
        """Analyze current stress level from monitoring data"""
        # Extract stress indicators from different sources
        physiological_stress = self._analyze_physiological_stress(monitoring_data)
        cognitive_stress = self._analyze_cognitive_stress(monitoring_data)
        emotional_stress = self._analyze_emotional_stress(monitoring_data)
        behavioral_stress = self._analyze_behavioral_stress(monitoring_data)
        
        # Calculate overall stress level
        overall_stress = (
            physiological_stress * 0.3 +
            cognitive_stress * 0.25 +
            emotional_stress * 0.25 +
            behavioral_stress * 0.2
        )
        
        # Determine stress category
        if overall_stress >= self.config['stress_threshold_critical']:
            stress_category = "critical"
        elif overall_stress >= self.config['stress_threshold_high']:
            stress_category = "high"
        elif overall_stress >= self.config['stress_threshold_moderate']:
            stress_category = "moderate"
        else:
            stress_category = "low"
        
        # Identify stress indicators
        stress_indicators = []
        if physiological_stress > 0.6:
            stress_indicators.append('elevated_heart_rate')
        if cognitive_stress > 0.6:
            stress_indicators.append('cognitive_overload')
        if emotional_stress > 0.6:
            stress_indicators.append('emotional_distress')
        if behavioral_stress > 0.6:
            stress_indicators.append('behavioral_changes')
        
        # Identify stress sources
        stress_sources = self._identify_stress_sources(monitoring_data)
        
        # Suggest coping mechanisms
        coping_mechanisms = self._suggest_coping_mechanisms(stress_category, stress_indicators)
        
        # Generate recovery recommendations
        recovery_recommendations = self._generate_recovery_recommendations(overall_stress, stress_indicators)
        
        return StressLevelData(
            overall_stress_level=overall_stress,
            stress_category=stress_category,
            stress_indicators=stress_indicators,
            physiological_stress=physiological_stress,
            cognitive_stress=cognitive_stress,
            emotional_stress=emotional_stress,
            behavioral_stress=behavioral_stress,
            stress_sources=stress_sources,
            coping_mechanisms=coping_mechanisms,
            recovery_recommendations=recovery_recommendations,
            burnout_risk=min(1.0, overall_stress * 1.2)  # Burnout risk slightly higher than stress
        )
    
    def _analyze_physiological_stress(self, data: Dict[str, Any]) -> float:
        """Analyze physiological stress indicators"""
        heart_rate = data.get('heart_rate', 70)
        blood_pressure = data.get('blood_pressure', {'systolic': 120, 'diastolic': 80})
        skin_conductance = data.get('skin_conductance', 0.5)
        
        # Normalize heart rate stress (60-100 normal, >100 stressed)
        hr_stress = max(0, min(1, (heart_rate - 60) / 40))
        
        # Normalize blood pressure stress
        systolic = blood_pressure.get('systolic', 120)
        bp_stress = max(0, min(1, (systolic - 120) / 40))
        
        # Skin conductance stress (higher = more stressed)
        sc_stress = min(1, skin_conductance)
        
        return (hr_stress + bp_stress + sc_stress) / 3
    
    def _analyze_cognitive_stress(self, data: Dict[str, Any]) -> float:
        """Analyze cognitive stress indicators"""
        task_performance = data.get('task_performance', 0.8)
        response_time = data.get('response_time', 1.0)
        error_rate = data.get('error_rate', 0.1)
        attention_span = data.get('attention_span', 30)
        
        # Performance decline indicates stress
        performance_stress = max(0, 1 - task_performance)
        
        # Slower response times indicate stress
        response_stress = max(0, min(1, (response_time - 1.0) / 2.0))
        
        # Higher error rates indicate stress
        error_stress = min(1, error_rate * 10)
        
        # Shorter attention span indicates stress
        attention_stress = max(0, min(1, (30 - attention_span) / 30))
        
        return (performance_stress + response_stress + error_stress + attention_stress) / 4
    
    def _analyze_emotional_stress(self, data: Dict[str, Any]) -> float:
        """Analyze emotional stress indicators"""
        emotion_data = data.get('emotion_analysis', {})
        
        primary_emotion = emotion_data.get('primary_emotion', 'neutral')
        valence = emotion_data.get('valence_level', 0.5)
        arousal = emotion_data.get('arousal_level', 0.5)
        stress_indicators = emotion_data.get('stress_indicators', [])
        
        # Negative emotions indicate stress
        if primary_emotion in ['anger', 'fear', 'sadness', 'frustration']:
            emotion_stress = 0.8
        elif primary_emotion in ['stress', 'anxiety']:
            emotion_stress = 0.9
        else:
            emotion_stress = 0.2
        
        # Low valence indicates stress
        valence_stress = max(0, 1 - valence * 2)
        
        # High arousal can indicate stress
        arousal_stress = max(0, (arousal - 0.7) / 0.3) if arousal > 0.7 else 0
        
        # Stress indicators from emotion analysis
        indicator_stress = min(1, len(stress_indicators) * 0.3)
        
        return (emotion_stress + valence_stress + arousal_stress + indicator_stress) / 4
    
    def _analyze_behavioral_stress(self, data: Dict[str, Any]) -> float:
        """Analyze behavioral stress indicators"""
        learning_patterns = data.get('learning_patterns', {})
        
        session_frequency = learning_patterns.get('session_frequency', 1.0)
        session_duration = learning_patterns.get('session_duration', 30)
        completion_rate = learning_patterns.get('completion_rate', 0.8)
        break_frequency = learning_patterns.get('break_frequency', 0.2)
        
        # Irregular session patterns indicate stress
        frequency_stress = abs(session_frequency - 1.0)
        
        # Very short or very long sessions indicate stress
        if session_duration < 15 or session_duration > 90:
            duration_stress = 0.6
        else:
            duration_stress = 0.2
        
        # Low completion rates indicate stress
        completion_stress = max(0, 1 - completion_rate)
        
        # Too few or too many breaks indicate stress
        if break_frequency < 0.1 or break_frequency > 0.5:
            break_stress = 0.5
        else:
            break_stress = 0.1
        
        return (frequency_stress + duration_stress + completion_stress + break_stress) / 4
    
    def _identify_stress_sources(self, data: Dict[str, Any]) -> List[str]:
        """Identify potential sources of stress"""
        sources = []
        
        # Check workload
        if data.get('task_difficulty', 0.5) > 0.8:
            sources.append('high_task_difficulty')
        
        if data.get('time_pressure', 0.5) > 0.7:
            sources.append('time_pressure')
        
        # Check environment
        if data.get('noise_level', 0.3) > 0.7:
            sources.append('noisy_environment')
        
        if data.get('interruptions', 0) > 3:
            sources.append('frequent_interruptions')
        
        # Check personal factors
        if data.get('sleep_quality', 0.8) < 0.5:
            sources.append('poor_sleep')
        
        if data.get('social_stress', 0.3) > 0.6:
            sources.append('social_pressure')
        
        return sources
    
    def _suggest_coping_mechanisms(self, stress_category: str, indicators: List[str]) -> List[str]:
        """Suggest appropriate coping mechanisms"""
        mechanisms = []
        
        if stress_category in ['high', 'critical']:
            mechanisms.extend([
                'deep_breathing_exercises',
                'progressive_muscle_relaxation',
                'mindfulness_meditation'
            ])
        
        if 'cognitive_overload' in indicators:
            mechanisms.extend([
                'break_tasks_into_smaller_chunks',
                'use_pomodoro_technique',
                'prioritize_tasks'
            ])
        
        if 'emotional_distress' in indicators:
            mechanisms.extend([
                'emotional_regulation_techniques',
                'positive_self_talk',
                'seek_social_support'
            ])
        
        if 'elevated_heart_rate' in indicators:
            mechanisms.extend([
                'physical_exercise',
                'yoga_or_stretching',
                'controlled_breathing'
            ])
        
        return mechanisms
    
    def _generate_recovery_recommendations(self, stress_level: float, indicators: List[str]) -> List[str]:
        """Generate recovery recommendations"""
        recommendations = []
        
        if stress_level > 0.8:
            recommendations.append('Take immediate break - 15-30 minutes')
            recommendations.append('Consider ending learning session for today')
        elif stress_level > 0.6:
            recommendations.append('Take 10-15 minute break')
            recommendations.append('Switch to easier content')
        elif stress_level > 0.4:
            recommendations.append('Take 5-10 minute break')
            recommendations.append('Practice stress reduction technique')
        
        if len(indicators) > 2:
            recommendations.append('Monitor stress levels closely')
            recommendations.append('Consider professional stress management support')
        
        return recommendations
    
    async def _analyze_stress_trends(self, user_id: str) -> Dict[str, Any]:
        """Analyze stress trends for a user"""
        # Get recent stress history for user
        recent_stress = [
            entry for entry in self.stress_history[-100:]  # Last 100 entries
            if entry['user_id'] == user_id
        ]
        
        if not recent_stress:
            return {
                'trend_direction': 'stable',
                'average_stress': 0.3,
                'stress_volatility': 0.2,
                'peak_stress_times': [],
                'stress_pattern': 'normal'
            }
        
        stress_levels = [entry['stress_level'] for entry in recent_stress]
        
        # Calculate trend
        if len(stress_levels) > 1:
            if stress_levels[-1] > stress_levels[0]:
                trend_direction = 'increasing'
            elif stress_levels[-1] < stress_levels[0]:
                trend_direction = 'decreasing'
            else:
                trend_direction = 'stable'
        else:
            trend_direction = 'stable'
        
        # Calculate averages and volatility
        avg_stress = sum(stress_levels) / len(stress_levels)
        
        if len(stress_levels) > 1:
            stress_variance = sum((s - avg_stress) ** 2 for s in stress_levels) / len(stress_levels)
            stress_volatility = min(1.0, stress_variance ** 0.5)
        else:
            stress_volatility = 0.0
        
        # Identify peak stress times
        peak_stress_times = [
            entry['timestamp'] for entry in recent_stress
            if entry['stress_level'] > 0.7
        ]
        
        # Determine stress pattern
        if avg_stress > 0.7:
            stress_pattern = 'chronically_high'
        elif stress_volatility > 0.6:
            stress_pattern = 'highly_variable'
        elif len(peak_stress_times) > len(stress_levels) * 0.3:
            stress_pattern = 'frequent_spikes'
        else:
            stress_pattern = 'normal'
        
        return {
            'trend_direction': trend_direction,
            'average_stress': avg_stress,
            'stress_volatility': stress_volatility,
            'peak_stress_times': peak_stress_times,
            'stress_pattern': stress_pattern
        }
    
    async def _assess_burnout_risk(self, user_id: str, stress_analysis: StressLevelData) -> Dict[str, Any]:
        """Assess burnout risk for a user"""
        # Get stress trends
        trends = await self._analyze_stress_trends(user_id)
        
        # Calculate burnout risk factors
        risk_factors = []
        risk_score = 0
        
        # High sustained stress
        if trends['average_stress'] > 0.7:
            risk_factors.append('sustained_high_stress')
            risk_score += 0.3
        
        # Increasing stress trend
        if trends['trend_direction'] == 'increasing':
            risk_factors.append('increasing_stress_trend')
            risk_score += 0.2
        
        # High stress volatility
        if trends['stress_volatility'] > 0.6:
            risk_factors.append('high_stress_volatility')
            risk_score += 0.1
        
        # Current high stress
        if stress_analysis.overall_stress_level > 0.8:
            risk_factors.append('current_high_stress')
            risk_score += 0.2
        
        # Multiple stress indicators
        if len(stress_analysis.stress_indicators) > 3:
            risk_factors.append('multiple_stress_indicators')
            risk_score += 0.1
        
        # Chronic stress pattern
        if trends['stress_pattern'] == 'chronically_high':
            risk_factors.append('chronic_stress_pattern')
            risk_score += 0.2
        
        # Determine burnout risk level
        if risk_score > 0.8:
            risk_level = 'critical'
        elif risk_score > 0.6:
            risk_level = 'high'
        elif risk_score > 0.4:
            risk_level = 'moderate'
        else:
            risk_level = 'low'
        
        return {
            'burnout_risk_score': risk_score,
            'burnout_risk_level': risk_level,
            'risk_factors': risk_factors,
            'intervention_urgency': 'immediate' if risk_level == 'critical' else 'planned'
        }
    
    async def _generate_stress_recommendations(self,
                                             stress_analysis: StressLevelData,
                                             burnout_assessment: Dict[str, Any]) -> List[str]:
        """Generate comprehensive stress management recommendations"""
        recommendations = []
        
        # Add stress-specific recommendations
        recommendations.extend(stress_analysis.recovery_recommendations)
        
        # Add coping mechanism recommendations
        recommendations.extend([f"Try: {mechanism}" for mechanism in stress_analysis.coping_mechanisms[:3]])
        
        # Add burnout prevention recommendations
        if burnout_assessment['burnout_risk_level'] in ['high', 'critical']:
            recommendations.extend([
                'Consider reducing learning workload',
                'Schedule regular rest periods',
                'Seek professional stress management support',
                'Implement work-life balance strategies'
            ])
        
        # Add source-specific recommendations
        for source in stress_analysis.stress_sources:
            if source == 'high_task_difficulty':
                recommendations.append('Break complex tasks into smaller, manageable steps')
            elif source == 'time_pressure':
                recommendations.append('Review and adjust learning schedule')
            elif source == 'poor_sleep':
                recommendations.append('Prioritize sleep hygiene and adequate rest')
        
        return list(set(recommendations))  # Remove duplicates


class BurnoutPreventionEngine:
    """
    🛡️ BURNOUT PREVENTION ENGINE
    
    Proactive burnout prevention and intervention system.
    """
    
    def __init__(self, stress_monitor: StressMonitoringSystem):
        self.stress_monitor = stress_monitor
        
        # Prevention configuration
        self.config = {
            'prevention_threshold': 0.6,
            'intervention_threshold': 0.8,
            'monitoring_frequency': 1800,  # 30 minutes
            'recovery_time_minimum': 3600  # 1 hour
        }
        
        logger.info("Burnout Prevention Engine initialized")
    
    async def assess_burnout_prevention_needs(self,
                                            user_id: str,
                                            current_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess burnout prevention needs for a user
        
        Args:
            user_id: User identifier
            current_data: Current monitoring data
            
        Returns:
            Dict with burnout prevention assessment and recommendations
        """
        try:
            # Get current stress analysis
            stress_result = await self.stress_monitor.monitor_stress_levels(user_id, current_data)
            
            if stress_result['status'] != 'success':
                return stress_result
            
            stress_analysis = stress_result['stress_analysis']
            burnout_assessment = stress_result['burnout_assessment']
            
            # Determine prevention actions
            prevention_actions = self._determine_prevention_actions(stress_analysis, burnout_assessment)
            
            # Generate intervention plan
            intervention_plan = self._generate_intervention_plan(burnout_assessment)
            
            return {
                'status': 'success',
                'user_id': user_id,
                'burnout_risk': burnout_assessment['burnout_risk_score'],
                'prevention_needed': burnout_assessment['burnout_risk_score'] > self.config['prevention_threshold'],
                'intervention_needed': burnout_assessment['burnout_risk_score'] > self.config['intervention_threshold'],
                'prevention_actions': prevention_actions,
                'intervention_plan': intervention_plan,
                'assessment_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error assessing burnout prevention needs for user {user_id}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _determine_prevention_actions(self,
                                    stress_analysis: Dict[str, Any],
                                    burnout_assessment: Dict[str, Any]) -> List[str]:
        """Determine appropriate prevention actions"""
        actions = []
        
        risk_level = burnout_assessment['burnout_risk_level']
        
        if risk_level == 'moderate':
            actions.extend([
                'Schedule regular breaks',
                'Monitor stress levels more frequently',
                'Implement stress reduction techniques'
            ])
        elif risk_level == 'high':
            actions.extend([
                'Reduce learning intensity',
                'Increase break frequency',
                'Focus on stress management',
                'Consider shorter learning sessions'
            ])
        elif risk_level == 'critical':
            actions.extend([
                'Immediate stress intervention required',
                'Suspend intensive learning activities',
                'Implement comprehensive recovery plan',
                'Seek professional support'
            ])
        
        return actions
    
    def _generate_intervention_plan(self, burnout_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive intervention plan"""
        risk_level = burnout_assessment['burnout_risk_level']
        
        if risk_level == 'low':
            return {
                'intervention_type': 'preventive',
                'duration': '1-2 days',
                'activities': ['maintain_current_practices', 'monitor_stress_levels'],
                'follow_up': 'weekly'
            }
        elif risk_level == 'moderate':
            return {
                'intervention_type': 'early_intervention',
                'duration': '3-5 days',
                'activities': ['stress_reduction_techniques', 'workload_adjustment', 'regular_breaks'],
                'follow_up': 'every_2_days'
            }
        elif risk_level == 'high':
            return {
                'intervention_type': 'active_intervention',
                'duration': '1-2 weeks',
                'activities': ['significant_workload_reduction', 'stress_management_program', 'recovery_activities'],
                'follow_up': 'daily'
            }
        else:  # critical
            return {
                'intervention_type': 'crisis_intervention',
                'duration': '2-4 weeks',
                'activities': ['complete_rest', 'professional_support', 'comprehensive_recovery_program'],
                'follow_up': 'twice_daily'
            }
