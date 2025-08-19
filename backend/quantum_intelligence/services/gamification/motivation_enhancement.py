"""
Motivation Enhancement Services

Extracted from quantum_intelligence_engine.py (lines 12526-15023) - advanced motivation
enhancement, personalized motivation systems, learning motivation analysis, and
motivational intervention systems for optimal learning engagement.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import math
import random

# Try to import structlog, fall back to standard logging
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    logger = logging.getLogger(__name__)

from ...core.exceptions import QuantumEngineError
from ...utils.caching import CacheService


class MotivationType(Enum):
    """Types of motivation"""
    INTRINSIC = "intrinsic"
    EXTRINSIC = "extrinsic"
    ACHIEVEMENT = "achievement"
    SOCIAL = "social"
    MASTERY = "mastery"
    PURPOSE = "purpose"
    AUTONOMY = "autonomy"
    COMPETENCE = "competence"


class MotivationState(Enum):
    """Current motivation states"""
    HIGHLY_MOTIVATED = "highly_motivated"
    MOTIVATED = "motivated"
    NEUTRAL = "neutral"
    DEMOTIVATED = "demotivated"
    HIGHLY_DEMOTIVATED = "highly_demotivated"


class InterventionType(Enum):
    """Types of motivational interventions"""
    GOAL_SETTING = "goal_setting"
    PROGRESS_HIGHLIGHTING = "progress_highlighting"
    SOCIAL_SUPPORT = "social_support"
    AUTONOMY_ENHANCEMENT = "autonomy_enhancement"
    COMPETENCE_BUILDING = "competence_building"
    PURPOSE_CONNECTION = "purpose_connection"
    REWARD_OPTIMIZATION = "reward_optimization"
    CHALLENGE_ADJUSTMENT = "challenge_adjustment"


@dataclass
class MotivationProfile:
    """User's motivation profile"""
    user_id: str = ""
    primary_motivation_type: MotivationType = MotivationType.INTRINSIC
    secondary_motivations: List[MotivationType] = field(default_factory=list)
    current_motivation_state: MotivationState = MotivationState.NEUTRAL
    motivation_score: float = 0.5  # 0.0-1.0
    motivation_factors: Dict[str, float] = field(default_factory=dict)
    motivation_triggers: List[str] = field(default_factory=list)
    motivation_barriers: List[str] = field(default_factory=list)
    optimal_motivation_strategies: List[str] = field(default_factory=list)
    motivation_history: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: str = ""


@dataclass
class MotivationalIntervention:
    """Motivational intervention plan"""
    intervention_id: str = ""
    intervention_type: InterventionType = InterventionType.GOAL_SETTING
    target_motivation_increase: float = 0.2
    intervention_strategies: List[Dict[str, Any]] = field(default_factory=list)
    implementation_timeline: Dict[str, Any] = field(default_factory=dict)
    success_metrics: List[str] = field(default_factory=list)
    personalization_factors: List[str] = field(default_factory=list)
    expected_duration_days: int = 7
    created_at: str = ""
    is_active: bool = True


@dataclass
class MotivationInsight:
    """Motivation analysis insight"""
    insight_id: str = ""
    insight_type: str = ""
    insight_message: str = ""
    confidence_score: float = 0.0
    actionable_recommendations: List[str] = field(default_factory=list)
    supporting_evidence: List[str] = field(default_factory=list)
    priority_level: str = "medium"
    created_at: str = ""


class LearningMotivationAnalyzer:
    """
    🧠 LEARNING MOTIVATION ANALYZER
    
    Advanced analyzer for learning motivation patterns and optimization.
    Extracted from the original quantum engine's gamification logic.
    """
    
    def __init__(self, cache_service: Optional[CacheService] = None):
        self.cache = cache_service
        
        # Analysis configuration
        self.config = {
            'motivation_factors': [
                'goal_clarity', 'progress_visibility', 'autonomy_level', 'competence_feeling',
                'social_connection', 'purpose_alignment', 'challenge_level', 'reward_satisfaction'
            ],
            'motivation_thresholds': {
                'high_motivation': 0.8,
                'moderate_motivation': 0.6,
                'low_motivation': 0.4,
                'critical_motivation': 0.2
            },
            'analysis_window_days': 14,
            'motivation_decay_rate': 0.05  # Daily decay without positive reinforcement
        }
        
        # Analysis tracking
        self.motivation_profiles = {}
        self.analysis_history = []
        
        logger.info("Learning Motivation Analyzer initialized")
    
    async def analyze_learning_motivation(self,
                                        user_id: str,
                                        learning_data: Dict[str, Any],
                                        behavioral_data: Dict[str, Any],
                                        context_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze user's learning motivation comprehensively
        
        Args:
            user_id: User identifier
            learning_data: Learning progress and performance data
            behavioral_data: User behavior patterns and engagement data
            context_data: Environmental and contextual factors
            
        Returns:
            Dict with comprehensive motivation analysis
        """
        try:
            # Analyze motivation factors
            motivation_factors = await self._analyze_motivation_factors(learning_data, behavioral_data, context_data)
            
            # Determine motivation types
            motivation_types = await self._identify_motivation_types(behavioral_data, learning_data)
            
            # Calculate current motivation state
            motivation_state = await self._calculate_motivation_state(motivation_factors, behavioral_data)
            
            # Identify motivation triggers and barriers
            triggers_barriers = await self._identify_triggers_and_barriers(behavioral_data, learning_data)
            
            # Generate motivation insights
            motivation_insights = await self._generate_motivation_insights(
                motivation_factors, motivation_state, triggers_barriers
            )
            
            # Create or update motivation profile
            motivation_profile = MotivationProfile(
                user_id=user_id,
                primary_motivation_type=motivation_types['primary'],
                secondary_motivations=motivation_types['secondary'],
                current_motivation_state=motivation_state['state'],
                motivation_score=motivation_state['score'],
                motivation_factors=motivation_factors,
                motivation_triggers=triggers_barriers['triggers'],
                motivation_barriers=triggers_barriers['barriers'],
                optimal_motivation_strategies=await self._identify_optimal_strategies(motivation_types, motivation_factors),
                motivation_history=self._update_motivation_history(user_id, motivation_state['score']),
                last_updated=datetime.utcnow().isoformat()
            )
            
            # Store profile
            self.motivation_profiles[user_id] = motivation_profile
            
            # Track analysis
            self.analysis_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'user_id': user_id,
                'motivation_score': motivation_state['score'],
                'motivation_state': motivation_state['state'].value,
                'primary_motivation': motivation_types['primary'].value
            })
            
            return {
                'status': 'success',
                'motivation_profile': motivation_profile.__dict__,
                'motivation_insights': [insight.__dict__ for insight in motivation_insights],
                'analysis_summary': {
                    'motivation_level': motivation_state['state'].value,
                    'key_factors': list(motivation_factors.keys())[:3],
                    'intervention_needed': motivation_state['score'] < self.config['motivation_thresholds']['moderate_motivation']
                },
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing learning motivation for user {user_id}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _analyze_motivation_factors(self,
                                        learning_data: Dict[str, Any],
                                        behavioral_data: Dict[str, Any],
                                        context_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze individual motivation factors"""
        factors = {}
        
        # Goal clarity
        learning_goals = learning_data.get('learning_goals', [])
        goal_progress = learning_data.get('goal_progress', {})
        if learning_goals:
            goal_clarity = sum(1 for goal in learning_goals if goal in goal_progress) / len(learning_goals)
        else:
            goal_clarity = 0.3  # Low if no goals set
        factors['goal_clarity'] = goal_clarity
        
        # Progress visibility
        recent_achievements = learning_data.get('recent_achievements', [])
        progress_tracking = behavioral_data.get('progress_tracking_engagement', 0.5)
        factors['progress_visibility'] = min(1.0, (len(recent_achievements) / 10) * 0.5 + progress_tracking * 0.5)
        
        # Autonomy level
        choice_frequency = behavioral_data.get('choice_making_frequency', 0.5)
        self_directed_learning = behavioral_data.get('self_directed_learning_ratio', 0.5)
        factors['autonomy_level'] = (choice_frequency + self_directed_learning) / 2
        
        # Competence feeling
        recent_performance = learning_data.get('recent_performance_score', 0.7)
        skill_confidence = learning_data.get('skill_confidence_average', 0.6)
        factors['competence_feeling'] = (recent_performance + skill_confidence) / 2
        
        # Social connection
        peer_interactions = behavioral_data.get('peer_interaction_frequency', 0.3)
        social_support = context_data.get('social_support_level', 0.4)
        factors['social_connection'] = (peer_interactions + social_support) / 2
        
        # Purpose alignment
        purpose_clarity = context_data.get('purpose_clarity_score', 0.5)
        goal_meaning = learning_data.get('goal_meaningfulness_score', 0.6)
        factors['purpose_alignment'] = (purpose_clarity + goal_meaning) / 2
        
        # Challenge level
        difficulty_appropriateness = learning_data.get('difficulty_appropriateness', 0.7)
        challenge_preference = behavioral_data.get('challenge_preference', 0.5)
        factors['challenge_level'] = (difficulty_appropriateness + challenge_preference) / 2
        
        # Reward satisfaction
        reward_effectiveness = behavioral_data.get('reward_effectiveness_score', 0.6)
        reward_frequency_satisfaction = behavioral_data.get('reward_frequency_satisfaction', 0.5)
        factors['reward_satisfaction'] = (reward_effectiveness + reward_frequency_satisfaction) / 2
        
        return factors
    
    async def _identify_motivation_types(self,
                                       behavioral_data: Dict[str, Any],
                                       learning_data: Dict[str, Any]) -> Dict[str, Any]:
        """Identify user's motivation types"""
        type_scores = {
            MotivationType.INTRINSIC: 0.0,
            MotivationType.EXTRINSIC: 0.0,
            MotivationType.ACHIEVEMENT: 0.0,
            MotivationType.SOCIAL: 0.0,
            MotivationType.MASTERY: 0.0,
            MotivationType.PURPOSE: 0.0,
            MotivationType.AUTONOMY: 0.0,
            MotivationType.COMPETENCE: 0.0
        }
        
        # Intrinsic motivation indicators
        curiosity_driven_learning = behavioral_data.get('curiosity_driven_learning', 0.5)
        self_directed_exploration = behavioral_data.get('self_directed_exploration', 0.4)
        type_scores[MotivationType.INTRINSIC] = (curiosity_driven_learning + self_directed_exploration) / 2
        
        # Extrinsic motivation indicators
        reward_responsiveness = behavioral_data.get('reward_responsiveness', 0.6)
        external_recognition_seeking = behavioral_data.get('external_recognition_seeking', 0.4)
        type_scores[MotivationType.EXTRINSIC] = (reward_responsiveness + external_recognition_seeking) / 2
        
        # Achievement motivation indicators
        goal_completion_drive = learning_data.get('goal_completion_rate', 0.7)
        competition_engagement = behavioral_data.get('competition_engagement', 0.3)
        type_scores[MotivationType.ACHIEVEMENT] = (goal_completion_drive + competition_engagement) / 2
        
        # Social motivation indicators
        peer_learning_preference = behavioral_data.get('peer_learning_preference', 0.4)
        social_sharing_frequency = behavioral_data.get('social_sharing_frequency', 0.3)
        type_scores[MotivationType.SOCIAL] = (peer_learning_preference + social_sharing_frequency) / 2
        
        # Mastery motivation indicators
        skill_depth_focus = learning_data.get('skill_depth_vs_breadth', 0.5)
        practice_persistence = behavioral_data.get('practice_persistence', 0.6)
        type_scores[MotivationType.MASTERY] = (skill_depth_focus + practice_persistence) / 2
        
        # Purpose motivation indicators
        meaning_seeking = behavioral_data.get('meaning_seeking_behavior', 0.5)
        value_alignment = learning_data.get('value_alignment_score', 0.6)
        type_scores[MotivationType.PURPOSE] = (meaning_seeking + value_alignment) / 2
        
        # Autonomy motivation indicators
        choice_preference = behavioral_data.get('choice_preference', 0.6)
        self_regulation = behavioral_data.get('self_regulation_score', 0.5)
        type_scores[MotivationType.AUTONOMY] = (choice_preference + self_regulation) / 2
        
        # Competence motivation indicators
        skill_building_focus = learning_data.get('skill_building_focus', 0.7)
        feedback_seeking = behavioral_data.get('feedback_seeking_frequency', 0.5)
        type_scores[MotivationType.COMPETENCE] = (skill_building_focus + feedback_seeking) / 2
        
        # Determine primary and secondary motivations
        sorted_types = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)
        primary_motivation = sorted_types[0][0]
        secondary_motivations = [mtype for mtype, score in sorted_types[1:4] if score > 0.4]
        
        return {
            'primary': primary_motivation,
            'secondary': secondary_motivations,
            'scores': {mtype.value: score for mtype, score in type_scores.items()}
        }
    
    async def _calculate_motivation_state(self,
                                        motivation_factors: Dict[str, float],
                                        behavioral_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate current motivation state"""
        # Weight factors by importance
        factor_weights = {
            'goal_clarity': 0.15,
            'progress_visibility': 0.15,
            'autonomy_level': 0.12,
            'competence_feeling': 0.18,
            'social_connection': 0.10,
            'purpose_alignment': 0.12,
            'challenge_level': 0.10,
            'reward_satisfaction': 0.08
        }
        
        # Calculate weighted motivation score
        motivation_score = sum(
            motivation_factors.get(factor, 0.5) * weight
            for factor, weight in factor_weights.items()
        )
        
        # Adjust based on recent engagement trends
        engagement_trend = behavioral_data.get('engagement_trend', 'stable')
        if engagement_trend == 'increasing':
            motivation_score *= 1.1
        elif engagement_trend == 'decreasing':
            motivation_score *= 0.9
        
        # Determine motivation state
        if motivation_score >= self.config['motivation_thresholds']['high_motivation']:
            state = MotivationState.HIGHLY_MOTIVATED
        elif motivation_score >= self.config['motivation_thresholds']['moderate_motivation']:
            state = MotivationState.MOTIVATED
        elif motivation_score >= self.config['motivation_thresholds']['low_motivation']:
            state = MotivationState.NEUTRAL
        elif motivation_score >= self.config['motivation_thresholds']['critical_motivation']:
            state = MotivationState.DEMOTIVATED
        else:
            state = MotivationState.HIGHLY_DEMOTIVATED
        
        return {
            'score': motivation_score,
            'state': state,
            'contributing_factors': sorted(motivation_factors.items(), key=lambda x: x[1], reverse=True)[:3]
        }
    
    async def _identify_triggers_and_barriers(self,
                                            behavioral_data: Dict[str, Any],
                                            learning_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Identify motivation triggers and barriers"""
        triggers = []
        barriers = []
        
        # Analyze positive patterns (triggers)
        if behavioral_data.get('achievement_response_positive', False):
            triggers.append('achievement_recognition')
        
        if behavioral_data.get('social_interaction_boost', False):
            triggers.append('peer_interaction')
        
        if behavioral_data.get('progress_visualization_engagement', 0.5) > 0.7:
            triggers.append('progress_visibility')
        
        if behavioral_data.get('choice_autonomy_boost', False):
            triggers.append('autonomy_enhancement')
        
        if learning_data.get('skill_mastery_satisfaction', 0.5) > 0.7:
            triggers.append('mastery_achievement')
        
        # Analyze negative patterns (barriers)
        if behavioral_data.get('frustration_indicators', []):
            barriers.append('difficulty_mismatch')
        
        if behavioral_data.get('isolation_indicators', False):
            barriers.append('social_isolation')
        
        if learning_data.get('goal_confusion_indicators', False):
            barriers.append('unclear_goals')
        
        if behavioral_data.get('reward_saturation_indicators', False):
            barriers.append('reward_fatigue')
        
        if behavioral_data.get('autonomy_restriction_frustration', False):
            barriers.append('limited_autonomy')
        
        return {'triggers': triggers, 'barriers': barriers}
    
    async def _generate_motivation_insights(self,
                                          motivation_factors: Dict[str, float],
                                          motivation_state: Dict[str, Any],
                                          triggers_barriers: Dict[str, List[str]]) -> List[MotivationInsight]:
        """Generate actionable motivation insights"""
        insights = []
        
        # Low motivation factors insights
        low_factors = [(factor, score) for factor, score in motivation_factors.items() if score < 0.4]
        
        for factor, score in low_factors:
            insight = MotivationInsight(
                insight_id=f"insight_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{factor}",
                insight_type="low_factor",
                insight_message=f"Low {factor.replace('_', ' ')} (score: {score:.2f}) is impacting motivation",
                confidence_score=0.8,
                actionable_recommendations=self._get_factor_recommendations(factor),
                supporting_evidence=[f"Factor score: {score:.2f}", f"Below threshold of 0.4"],
                priority_level="high" if score < 0.3 else "medium",
                created_at=datetime.utcnow().isoformat()
            )
            insights.append(insight)
        
        # Motivation state insights
        current_state = motivation_state['state']
        if current_state in [MotivationState.DEMOTIVATED, MotivationState.HIGHLY_DEMOTIVATED]:
            insight = MotivationInsight(
                insight_id=f"insight_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_state",
                insight_type="motivation_state",
                insight_message=f"Current motivation state is {current_state.value} - intervention recommended",
                confidence_score=0.9,
                actionable_recommendations=self._get_state_recommendations(current_state),
                supporting_evidence=[f"Motivation score: {motivation_state['score']:.2f}"],
                priority_level="high",
                created_at=datetime.utcnow().isoformat()
            )
            insights.append(insight)
        
        # Barrier insights
        for barrier in triggers_barriers['barriers']:
            insight = MotivationInsight(
                insight_id=f"insight_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{barrier}",
                insight_type="barrier",
                insight_message=f"Motivation barrier detected: {barrier.replace('_', ' ')}",
                confidence_score=0.7,
                actionable_recommendations=self._get_barrier_recommendations(barrier),
                supporting_evidence=[f"Barrier identified: {barrier}"],
                priority_level="medium",
                created_at=datetime.utcnow().isoformat()
            )
            insights.append(insight)
        
        # Trigger optimization insights
        if triggers_barriers['triggers']:
            insight = MotivationInsight(
                insight_id=f"insight_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_triggers",
                insight_type="trigger_optimization",
                insight_message=f"Leverage identified motivation triggers: {', '.join(triggers_barriers['triggers'][:3])}",
                confidence_score=0.8,
                actionable_recommendations=self._get_trigger_recommendations(triggers_barriers['triggers']),
                supporting_evidence=[f"Active triggers: {len(triggers_barriers['triggers'])}"],
                priority_level="medium",
                created_at=datetime.utcnow().isoformat()
            )
            insights.append(insight)
        
        return insights
    
    def _get_factor_recommendations(self, factor: str) -> List[str]:
        """Get recommendations for improving specific motivation factors"""
        recommendations = {
            'goal_clarity': [
                "Set specific, measurable learning goals",
                "Break down large goals into smaller milestones",
                "Regularly review and adjust goals"
            ],
            'progress_visibility': [
                "Use progress tracking tools and visualizations",
                "Celebrate small wins and milestones",
                "Create progress journals or logs"
            ],
            'autonomy_level': [
                "Provide more learning path choices",
                "Enable self-paced learning options",
                "Increase decision-making opportunities"
            ],
            'competence_feeling': [
                "Focus on skill-building activities",
                "Provide constructive feedback",
                "Adjust difficulty to optimal challenge level"
            ],
            'social_connection': [
                "Join learning communities or study groups",
                "Engage in peer learning activities",
                "Seek mentorship or coaching"
            ],
            'purpose_alignment': [
                "Connect learning to personal values and goals",
                "Explore real-world applications",
                "Reflect on learning purpose and meaning"
            ],
            'challenge_level': [
                "Adjust difficulty to match skill level",
                "Provide appropriate scaffolding",
                "Offer varied challenge types"
            ],
            'reward_satisfaction': [
                "Diversify reward types and timing",
                "Align rewards with personal preferences",
                "Focus on intrinsic reward recognition"
            ]
        }
        
        return recommendations.get(factor, ["Seek personalized guidance for improvement"])
    
    def _get_state_recommendations(self, state: MotivationState) -> List[str]:
        """Get recommendations based on motivation state"""
        if state == MotivationState.HIGHLY_DEMOTIVATED:
            return [
                "Take a break and reassess learning goals",
                "Seek support from mentors or peers",
                "Start with very small, achievable tasks",
                "Consider changing learning approach or environment"
            ]
        elif state == MotivationState.DEMOTIVATED:
            return [
                "Focus on quick wins and easy achievements",
                "Reconnect with learning purpose and goals",
                "Adjust challenge level to build confidence",
                "Increase social support and encouragement"
            ]
        else:
            return ["Continue current approach with minor optimizations"]
    
    def _get_barrier_recommendations(self, barrier: str) -> List[str]:
        """Get recommendations for overcoming specific barriers"""
        recommendations = {
            'difficulty_mismatch': [
                "Adjust learning difficulty to match current skill level",
                "Provide additional scaffolding and support",
                "Break complex tasks into smaller steps"
            ],
            'social_isolation': [
                "Join learning communities or study groups",
                "Engage in collaborative learning activities",
                "Seek peer support and interaction"
            ],
            'unclear_goals': [
                "Clarify and refine learning objectives",
                "Set specific, measurable goals",
                "Create detailed learning plans"
            ],
            'reward_fatigue': [
                "Diversify reward types and timing",
                "Focus more on intrinsic motivation",
                "Reduce frequency of external rewards"
            ],
            'limited_autonomy': [
                "Increase choice and control in learning",
                "Provide multiple learning path options",
                "Enable self-directed learning opportunities"
            ]
        }
        
        return recommendations.get(barrier, ["Address barrier through personalized intervention"])
    
    def _get_trigger_recommendations(self, triggers: List[str]) -> List[str]:
        """Get recommendations for leveraging motivation triggers"""
        recommendations = []
        
        if 'achievement_recognition' in triggers:
            recommendations.append("Implement regular achievement celebrations and recognition")
        
        if 'peer_interaction' in triggers:
            recommendations.append("Increase collaborative learning and social activities")
        
        if 'progress_visibility' in triggers:
            recommendations.append("Enhance progress tracking and visualization tools")
        
        if 'autonomy_enhancement' in triggers:
            recommendations.append("Provide more choice and control in learning activities")
        
        if 'mastery_achievement' in triggers:
            recommendations.append("Focus on skill mastery and expertise development")
        
        return recommendations if recommendations else ["Leverage identified triggers for motivation enhancement"]
    
    async def _identify_optimal_strategies(self,
                                         motivation_types: Dict[str, Any],
                                         motivation_factors: Dict[str, float]) -> List[str]:
        """Identify optimal motivation strategies for user"""
        strategies = []
        
        primary_type = motivation_types['primary']
        
        # Type-specific strategies
        if primary_type == MotivationType.INTRINSIC:
            strategies.extend(['curiosity_driven_learning', 'self_directed_exploration', 'mastery_focus'])
        elif primary_type == MotivationType.EXTRINSIC:
            strategies.extend(['reward_optimization', 'recognition_systems', 'external_validation'])
        elif primary_type == MotivationType.ACHIEVEMENT:
            strategies.extend(['goal_setting', 'progress_tracking', 'competition_elements'])
        elif primary_type == MotivationType.SOCIAL:
            strategies.extend(['peer_learning', 'collaborative_projects', 'social_recognition'])
        elif primary_type == MotivationType.MASTERY:
            strategies.extend(['skill_building', 'expertise_development', 'deliberate_practice'])
        elif primary_type == MotivationType.PURPOSE:
            strategies.extend(['meaning_connection', 'value_alignment', 'impact_visualization'])
        elif primary_type == MotivationType.AUTONOMY:
            strategies.extend(['choice_provision', 'self_regulation', 'personalized_paths'])
        elif primary_type == MotivationType.COMPETENCE:
            strategies.extend(['skill_building', 'feedback_systems', 'confidence_building'])
        
        # Factor-based strategies
        low_factors = [factor for factor, score in motivation_factors.items() if score < 0.5]
        for factor in low_factors:
            if factor == 'goal_clarity':
                strategies.append('goal_setting_support')
            elif factor == 'social_connection':
                strategies.append('social_engagement_enhancement')
            elif factor == 'competence_feeling':
                strategies.append('confidence_building_activities')
        
        return list(set(strategies))  # Remove duplicates
    
    def _update_motivation_history(self, user_id: str, current_score: float) -> List[Dict[str, Any]]:
        """Update motivation history for user"""
        if user_id in self.motivation_profiles:
            history = self.motivation_profiles[user_id].motivation_history.copy()
        else:
            history = []
        
        history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'motivation_score': current_score,
            'date': datetime.utcnow().date().isoformat()
        })
        
        # Keep only recent history (last 30 entries)
        return history[-30:]


class PersonalizedMotivationSystem:
    """
    🎯 PERSONALIZED MOTIVATION SYSTEM
    
    Advanced system for creating personalized motivation enhancement plans.
    """
    
    def __init__(self, motivation_analyzer: LearningMotivationAnalyzer):
        self.motivation_analyzer = motivation_analyzer
        
        # System configuration
        self.config = {
            'personalization_factors': [
                'motivation_type', 'learning_style', 'personality', 'goals',
                'preferences', 'context', 'history', 'barriers'
            ],
            'adaptation_learning_rate': 0.1,
            'effectiveness_tracking': True,
            'intervention_cooldown_days': 3
        }
        
        # System tracking
        self.personalized_plans = {}
        self.effectiveness_history = []
        
        logger.info("Personalized Motivation System initialized")
    
    async def create_personalized_motivation_plan(self,
                                                user_id: str,
                                                user_profile: Dict[str, Any],
                                                learning_context: Dict[str, Any],
                                                motivation_goals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create personalized motivation enhancement plan
        
        Args:
            user_id: User identifier
            user_profile: User's complete profile
            learning_context: Current learning context
            motivation_goals: Specific motivation goals and targets
            
        Returns:
            Dict with personalized motivation plan
        """
        try:
            # Get current motivation analysis
            if user_id in self.motivation_analyzer.motivation_profiles:
                motivation_profile = self.motivation_analyzer.motivation_profiles[user_id]
            else:
                # Perform motivation analysis first
                analysis_result = await self.motivation_analyzer.analyze_learning_motivation(
                    user_id, learning_context, user_profile, {}
                )
                if analysis_result['status'] == 'success':
                    motivation_profile = MotivationProfile(**analysis_result['motivation_profile'])
                else:
                    return analysis_result
            
            # Identify personalization factors
            personalization_factors = await self._identify_personalization_factors(
                user_profile, motivation_profile, learning_context
            )
            
            # Create motivation strategies
            motivation_strategies = await self._create_motivation_strategies(
                motivation_profile, personalization_factors, motivation_goals
            )
            
            # Design implementation plan
            implementation_plan = await self._design_implementation_plan(
                motivation_strategies, user_profile, learning_context
            )
            
            # Create monitoring system
            monitoring_system = await self._create_monitoring_system(
                motivation_profile, motivation_goals
            )
            
            # Generate personalized recommendations
            personalized_recommendations = await self._generate_personalized_recommendations(
                motivation_profile, personalization_factors, motivation_strategies
            )
            
            # Create complete plan
            motivation_plan = {
                'plan_id': f"plan_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{user_id}",
                'user_id': user_id,
                'motivation_profile': motivation_profile.__dict__,
                'personalization_factors': personalization_factors,
                'motivation_strategies': motivation_strategies,
                'implementation_plan': implementation_plan,
                'monitoring_system': monitoring_system,
                'personalized_recommendations': personalized_recommendations,
                'created_at': datetime.utcnow().isoformat(),
                'is_active': True
            }
            
            # Store plan
            self.personalized_plans[user_id] = motivation_plan
            
            return {
                'status': 'success',
                'motivation_plan': motivation_plan,
                'plan_summary': {
                    'primary_motivation': motivation_profile.primary_motivation_type.value,
                    'strategies_count': len(motivation_strategies),
                    'implementation_phases': len(implementation_plan.get('phases', [])),
                    'personalization_level': len(personalization_factors)
                },
                'creation_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating personalized motivation plan for user {user_id}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _identify_personalization_factors(self,
                                              user_profile: Dict[str, Any],
                                              motivation_profile: MotivationProfile,
                                              learning_context: Dict[str, Any]) -> Dict[str, Any]:
        """Identify factors for personalization"""
        factors = {}
        
        # Motivation type factors
        factors['primary_motivation'] = motivation_profile.primary_motivation_type.value
        factors['secondary_motivations'] = [m.value for m in motivation_profile.secondary_motivations]
        
        # Learning style factors
        learning_style = user_profile.get('learning_style', {})
        factors['learning_style'] = learning_style
        
        # Personality factors
        personality = user_profile.get('personality', {})
        factors['personality_traits'] = personality
        
        # Goal factors
        learning_goals = user_profile.get('learning_goals', [])
        factors['learning_goals'] = learning_goals
        
        # Preference factors
        preferences = user_profile.get('preferences', {})
        factors['user_preferences'] = preferences
        
        # Context factors
        factors['learning_environment'] = learning_context.get('environment', 'online')
        factors['time_availability'] = learning_context.get('time_availability', 'flexible')
        factors['social_context'] = learning_context.get('social_context', 'individual')
        
        # Historical factors
        factors['motivation_history'] = motivation_profile.motivation_history[-5:]  # Recent history
        factors['motivation_barriers'] = motivation_profile.motivation_barriers
        factors['motivation_triggers'] = motivation_profile.motivation_triggers
        
        return factors
    
    async def _create_motivation_strategies(self,
                                          motivation_profile: MotivationProfile,
                                          personalization_factors: Dict[str, Any],
                                          motivation_goals: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create personalized motivation strategies"""
        strategies = []
        
        # Primary motivation type strategies
        primary_type = motivation_profile.primary_motivation_type
        
        if primary_type == MotivationType.INTRINSIC:
            strategies.append({
                'strategy_type': 'intrinsic_enhancement',
                'name': 'Curiosity-Driven Learning',
                'description': 'Enhance natural curiosity and interest in learning',
                'tactics': ['exploration_rewards', 'discovery_challenges', 'interest_alignment'],
                'priority': 'high'
            })
        
        elif primary_type == MotivationType.ACHIEVEMENT:
            strategies.append({
                'strategy_type': 'achievement_optimization',
                'name': 'Goal Achievement System',
                'description': 'Optimize goal setting and achievement recognition',
                'tactics': ['smart_goals', 'milestone_celebrations', 'progress_tracking'],
                'priority': 'high'
            })
        
        elif primary_type == MotivationType.SOCIAL:
            strategies.append({
                'strategy_type': 'social_engagement',
                'name': 'Social Learning Enhancement',
                'description': 'Leverage social connections for motivation',
                'tactics': ['peer_learning', 'social_recognition', 'collaborative_goals'],
                'priority': 'high'
            })
        
        # Address low motivation factors
        low_factors = [
            factor for factor, score in motivation_profile.motivation_factors.items()
            if score < 0.5
        ]
        
        for factor in low_factors:
            if factor == 'competence_feeling':
                strategies.append({
                    'strategy_type': 'competence_building',
                    'name': 'Confidence Building Program',
                    'description': 'Build confidence through skill development',
                    'tactics': ['skill_scaffolding', 'success_experiences', 'feedback_optimization'],
                    'priority': 'medium'
                })
            
            elif factor == 'autonomy_level':
                strategies.append({
                    'strategy_type': 'autonomy_enhancement',
                    'name': 'Choice and Control Enhancement',
                    'description': 'Increase learner autonomy and control',
                    'tactics': ['choice_provision', 'self_pacing', 'path_customization'],
                    'priority': 'medium'
                })
        
        # Address motivation barriers
        for barrier in motivation_profile.motivation_barriers:
            if barrier == 'social_isolation':
                strategies.append({
                    'strategy_type': 'social_connection',
                    'name': 'Social Connection Building',
                    'description': 'Build social connections and support',
                    'tactics': ['community_engagement', 'peer_matching', 'mentorship'],
                    'priority': 'medium'
                })
        
        # Leverage motivation triggers
        for trigger in motivation_profile.motivation_triggers:
            if trigger == 'progress_visibility':
                strategies.append({
                    'strategy_type': 'progress_enhancement',
                    'name': 'Progress Visualization System',
                    'description': 'Enhance progress visibility and tracking',
                    'tactics': ['visual_dashboards', 'milestone_mapping', 'achievement_galleries'],
                    'priority': 'low'
                })
        
        return strategies
    
    async def _design_implementation_plan(self,
                                        motivation_strategies: List[Dict[str, Any]],
                                        user_profile: Dict[str, Any],
                                        learning_context: Dict[str, Any]) -> Dict[str, Any]:
        """Design implementation plan for motivation strategies"""
        # Sort strategies by priority
        high_priority = [s for s in motivation_strategies if s['priority'] == 'high']
        medium_priority = [s for s in motivation_strategies if s['priority'] == 'medium']
        low_priority = [s for s in motivation_strategies if s['priority'] == 'low']
        
        # Create implementation phases
        phases = []
        
        if high_priority:
            phases.append({
                'phase': 1,
                'name': 'Foundation Building',
                'strategies': high_priority,
                'duration_days': 7,
                'success_criteria': ['motivation_score_increase', 'engagement_improvement']
            })
        
        if medium_priority:
            phases.append({
                'phase': 2,
                'name': 'Enhancement Implementation',
                'strategies': medium_priority,
                'duration_days': 14,
                'success_criteria': ['sustained_motivation', 'barrier_reduction']
            })
        
        if low_priority:
            phases.append({
                'phase': 3,
                'name': 'Optimization and Refinement',
                'strategies': low_priority,
                'duration_days': 14,
                'success_criteria': ['motivation_optimization', 'long_term_sustainability']
            })
        
        return {
            'phases': phases,
            'total_duration_days': sum(phase['duration_days'] for phase in phases),
            'implementation_approach': 'gradual_rollout',
            'adaptation_checkpoints': ['week_1', 'week_2', 'week_4'],
            'success_metrics': ['motivation_score', 'engagement_level', 'goal_progress']
        }
    
    async def _create_monitoring_system(self,
                                      motivation_profile: MotivationProfile,
                                      motivation_goals: Dict[str, Any]) -> Dict[str, Any]:
        """Create monitoring system for motivation plan"""
        return {
            'monitoring_frequency': 'daily',
            'key_metrics': [
                'motivation_score',
                'engagement_level',
                'goal_progress',
                'barrier_presence',
                'trigger_activation'
            ],
            'alert_thresholds': {
                'motivation_drop': 0.2,
                'engagement_decline': 0.3,
                'barrier_emergence': 'immediate'
            },
            'feedback_collection': {
                'user_surveys': 'weekly',
                'behavioral_tracking': 'continuous',
                'goal_assessment': 'bi_weekly'
            },
            'adaptation_triggers': [
                'motivation_score_drop',
                'strategy_ineffectiveness',
                'user_feedback',
                'goal_changes'
            ]
        }
    
    async def _generate_personalized_recommendations(self,
                                                   motivation_profile: MotivationProfile,
                                                   personalization_factors: Dict[str, Any],
                                                   motivation_strategies: List[Dict[str, Any]]) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        # Type-specific recommendations
        primary_type = motivation_profile.primary_motivation_type
        
        if primary_type == MotivationType.INTRINSIC:
            recommendations.extend([
                "Follow your curiosity and explore topics that genuinely interest you",
                "Set learning goals that align with your personal interests",
                "Celebrate the joy of discovery and understanding"
            ])
        
        elif primary_type == MotivationType.ACHIEVEMENT:
            recommendations.extend([
                "Set clear, challenging but achievable goals",
                "Track your progress regularly and celebrate milestones",
                "Compare your progress with past performance rather than others"
            ])
        
        elif primary_type == MotivationType.SOCIAL:
            recommendations.extend([
                "Join learning communities and study groups",
                "Share your learning journey with friends and family",
                "Seek opportunities to teach or mentor others"
            ])
        
        # Factor-specific recommendations
        low_factors = [
            factor for factor, score in motivation_profile.motivation_factors.items()
            if score < 0.5
        ]
        
        for factor in low_factors[:2]:  # Top 2 low factors
            if factor == 'goal_clarity':
                recommendations.append("Spend time clarifying and refining your learning goals")
            elif factor == 'competence_feeling':
                recommendations.append("Focus on building confidence through small, achievable wins")
            elif factor == 'social_connection':
                recommendations.append("Actively seek out learning partners and communities")
        
        # Personalization-based recommendations
        learning_style = personalization_factors.get('learning_style', {})
        if learning_style.get('visual', 0) > 0.7:
            recommendations.append("Use visual learning tools and progress tracking dashboards")
        
        if learning_style.get('kinesthetic', 0) > 0.7:
            recommendations.append("Incorporate hands-on activities and practical applications")
        
        return recommendations[:8]  # Limit to top 8 recommendations
