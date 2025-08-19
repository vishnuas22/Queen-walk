"""
Engagement Mechanics Services

Extracted from quantum_intelligence_engine.py (lines 12526-15023) - advanced engagement
mechanics, challenge generation, progress visualization, and habit formation systems
for optimal user engagement and motivation.
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


class EngagementMechanicType(Enum):
    """Types of engagement mechanics"""
    PROGRESSIVE_CHALLENGES = "progressive_challenges"
    STREAK_MECHANICS = "streak_mechanics"
    SURPRISE_ELEMENTS = "surprise_elements"
    SOCIAL_CHALLENGES = "social_challenges"
    TIME_PRESSURE = "time_pressure"
    EXPLORATION_REWARDS = "exploration_rewards"
    MASTERY_PATHS = "mastery_paths"
    COLLABORATIVE_GOALS = "collaborative_goals"


class ChallengeType(Enum):
    """Types of dynamic challenges"""
    SKILL_CHALLENGE = "skill_challenge"
    SPEED_CHALLENGE = "speed_challenge"
    ACCURACY_CHALLENGE = "accuracy_challenge"
    CREATIVITY_CHALLENGE = "creativity_challenge"
    ENDURANCE_CHALLENGE = "endurance_challenge"
    COLLABORATION_CHALLENGE = "collaboration_challenge"
    EXPLORATION_CHALLENGE = "exploration_challenge"
    INNOVATION_CHALLENGE = "innovation_challenge"


class VisualizationType(Enum):
    """Types of progress visualization"""
    PROGRESS_BAR = "progress_bar"
    SKILL_TREE = "skill_tree"
    JOURNEY_MAP = "journey_map"
    ACHIEVEMENT_GALLERY = "achievement_gallery"
    STATISTICS_DASHBOARD = "statistics_dashboard"
    TIMELINE_VIEW = "timeline_view"
    COMPARISON_CHARTS = "comparison_charts"
    INTERACTIVE_GRAPH = "interactive_graph"


@dataclass
class DynamicChallenge:
    """Dynamic AI-generated challenge"""
    challenge_id: str = ""
    title: str = ""
    description: str = ""
    challenge_type: ChallengeType = ChallengeType.SKILL_CHALLENGE
    difficulty_level: float = 0.5  # 0.0-1.0
    estimated_duration_minutes: int = 30
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    reward_points: int = 100
    bonus_conditions: List[Dict[str, Any]] = field(default_factory=list)
    personalization_factors: List[str] = field(default_factory=list)
    is_time_limited: bool = False
    expires_at: Optional[str] = None
    created_at: str = ""
    engagement_hooks: List[str] = field(default_factory=list)


@dataclass
class ProgressVisualization:
    """Progress visualization configuration"""
    visualization_id: str = ""
    visualization_type: VisualizationType = VisualizationType.PROGRESS_BAR
    data_sources: List[str] = field(default_factory=list)
    visual_elements: Dict[str, Any] = field(default_factory=dict)
    interactive_features: List[str] = field(default_factory=list)
    personalization_settings: Dict[str, Any] = field(default_factory=dict)
    update_frequency: str = "real_time"
    animation_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HabitFormationPlan:
    """Habit formation plan for learning behaviors"""
    plan_id: str = ""
    target_habit: str = ""
    current_streak: int = 0
    target_streak: int = 21  # 21 days for habit formation
    habit_triggers: List[str] = field(default_factory=list)
    reward_schedule: Dict[str, Any] = field(default_factory=dict)
    progress_milestones: List[Dict[str, Any]] = field(default_factory=list)
    habit_strength: float = 0.0  # 0.0-1.0
    formation_stage: str = "initiation"  # initiation, development, maintenance
    created_at: str = ""
    last_activity: Optional[str] = None


class EngagementMechanicsEngine:
    """
    🎮 ENGAGEMENT MECHANICS ENGINE
    
    Advanced engine for creating and managing engagement mechanics.
    Extracted from the original quantum engine's gamification logic.
    """
    
    def __init__(self, cache_service: Optional[CacheService] = None):
        self.cache = cache_service
        
        # Engine configuration
        self.config = {
            'engagement_strategies': {
                'high_achiever': [EngagementMechanicType.PROGRESSIVE_CHALLENGES, EngagementMechanicType.MASTERY_PATHS],
                'social_learner': [EngagementMechanicType.SOCIAL_CHALLENGES, EngagementMechanicType.COLLABORATIVE_GOALS],
                'explorer': [EngagementMechanicType.EXPLORATION_REWARDS, EngagementMechanicType.SURPRISE_ELEMENTS],
                'consistent_learner': [EngagementMechanicType.STREAK_MECHANICS, EngagementMechanicType.TIME_PRESSURE]
            },
            'engagement_thresholds': {
                'low_engagement': 0.3,
                'medium_engagement': 0.6,
                'high_engagement': 0.8
            },
            'mechanic_effectiveness_tracking': True,
            'adaptive_difficulty': True
        }
        
        # Engine tracking
        self.active_mechanics = {}  # user_id -> list of active mechanics
        self.engagement_history = []
        self.mechanic_effectiveness = {}
        
        logger.info("Engagement Mechanics Engine initialized")
    
    async def optimize_engagement_mechanics(self,
                                          user_id: str,
                                          user_profile: Dict[str, Any],
                                          engagement_data: Dict[str, Any],
                                          learning_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize engagement mechanics for user
        
        Args:
            user_id: User identifier
            user_profile: User's learning profile and preferences
            engagement_data: Current engagement metrics and history
            learning_context: Current learning context and goals
            
        Returns:
            Dict with optimized engagement mechanics
        """
        try:
            # Analyze current engagement state
            engagement_analysis = await self._analyze_engagement_state(engagement_data)
            
            # Identify user engagement profile
            engagement_profile = await self._identify_engagement_profile(user_profile, engagement_data)
            
            # Select optimal mechanics
            optimal_mechanics = await self._select_optimal_mechanics(
                engagement_profile, engagement_analysis, learning_context
            )
            
            # Configure mechanics for user
            configured_mechanics = await self._configure_mechanics(
                optimal_mechanics, user_profile, learning_context
            )
            
            # Create implementation plan
            implementation_plan = await self._create_implementation_plan(configured_mechanics)
            
            # Update active mechanics
            self.active_mechanics[user_id] = configured_mechanics
            
            # Track optimization
            self.engagement_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'user_id': user_id,
                'engagement_score': engagement_analysis.get('current_engagement', 0.0),
                'mechanics_count': len(configured_mechanics),
                'profile_type': engagement_profile.get('primary_type', 'unknown')
            })
            
            return {
                'status': 'success',
                'engagement_analysis': engagement_analysis,
                'engagement_profile': engagement_profile,
                'optimal_mechanics': [m.__dict__ for m in configured_mechanics],
                'implementation_plan': implementation_plan,
                'optimization_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error optimizing engagement mechanics for user {user_id}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _analyze_engagement_state(self, engagement_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current engagement state"""
        current_engagement = engagement_data.get('current_engagement_score', 0.5)
        engagement_trend = engagement_data.get('engagement_trend', 'stable')
        recent_activities = engagement_data.get('recent_activities', [])
        
        # Calculate engagement metrics
        activity_frequency = len(recent_activities) / max(1, engagement_data.get('tracking_days', 7))
        
        # Analyze engagement patterns
        engagement_patterns = {}
        if recent_activities:
            # Time-based patterns
            hourly_activity = {}
            for activity in recent_activities:
                hour = datetime.fromisoformat(activity.get('timestamp', datetime.utcnow().isoformat())).hour
                hourly_activity[hour] = hourly_activity.get(hour, 0) + 1
            
            peak_hours = sorted(hourly_activity.items(), key=lambda x: x[1], reverse=True)[:3]
            engagement_patterns['peak_hours'] = [hour for hour, _ in peak_hours]
            
            # Session length patterns
            session_lengths = [activity.get('duration_minutes', 30) for activity in recent_activities]
            engagement_patterns['avg_session_length'] = sum(session_lengths) / len(session_lengths)
            engagement_patterns['preferred_session_length'] = 'short' if engagement_patterns['avg_session_length'] < 20 else 'long'
        
        # Determine engagement level
        if current_engagement >= self.config['engagement_thresholds']['high_engagement']:
            engagement_level = 'high'
        elif current_engagement >= self.config['engagement_thresholds']['medium_engagement']:
            engagement_level = 'medium'
        else:
            engagement_level = 'low'
        
        return {
            'current_engagement': current_engagement,
            'engagement_level': engagement_level,
            'engagement_trend': engagement_trend,
            'activity_frequency': activity_frequency,
            'engagement_patterns': engagement_patterns,
            'needs_intervention': engagement_level == 'low' or engagement_trend == 'declining'
        }
    
    async def _identify_engagement_profile(self,
                                         user_profile: Dict[str, Any],
                                         engagement_data: Dict[str, Any]) -> Dict[str, Any]:
        """Identify user's engagement profile type"""
        # Analyze user characteristics
        personality = user_profile.get('personality', {})
        learning_style = user_profile.get('learning_style', {})
        preferences = user_profile.get('preferences', {})
        
        # Score different profile types
        profile_scores = {
            'high_achiever': 0.0,
            'social_learner': 0.0,
            'explorer': 0.0,
            'consistent_learner': 0.0
        }
        
        # High achiever indicators
        if personality.get('conscientiousness', 0.5) > 0.7:
            profile_scores['high_achiever'] += 0.3
        if preferences.get('challenge_preference', 0.5) > 0.7:
            profile_scores['high_achiever'] += 0.3
        if engagement_data.get('achievement_focus', 0.5) > 0.6:
            profile_scores['high_achiever'] += 0.4
        
        # Social learner indicators
        if personality.get('extraversion', 0.5) > 0.6:
            profile_scores['social_learner'] += 0.3
        if preferences.get('social_learning_preference', 0.5) > 0.6:
            profile_scores['social_learner'] += 0.4
        if engagement_data.get('social_activity_level', 0.3) > 0.5:
            profile_scores['social_learner'] += 0.3
        
        # Explorer indicators
        if personality.get('openness', 0.5) > 0.7:
            profile_scores['explorer'] += 0.3
        if preferences.get('variety_preference', 0.5) > 0.6:
            profile_scores['explorer'] += 0.4
        if engagement_data.get('exploration_behavior', 0.4) > 0.6:
            profile_scores['explorer'] += 0.3
        
        # Consistent learner indicators
        if engagement_data.get('consistency_score', 0.5) > 0.7:
            profile_scores['consistent_learner'] += 0.4
        if preferences.get('routine_preference', 0.5) > 0.6:
            profile_scores['consistent_learner'] += 0.3
        if engagement_data.get('streak_behavior', 0.3) > 0.5:
            profile_scores['consistent_learner'] += 0.3
        
        # Determine primary profile
        primary_type = max(profile_scores.items(), key=lambda x: x[1])
        
        # Identify secondary profiles
        secondary_types = sorted(profile_scores.items(), key=lambda x: x[1], reverse=True)[1:3]
        secondary_types = [ptype for ptype, score in secondary_types if score > 0.3]
        
        return {
            'primary_type': primary_type[0],
            'primary_score': primary_type[1],
            'secondary_types': secondary_types,
            'profile_scores': profile_scores,
            'profile_confidence': primary_type[1]
        }
    
    async def _select_optimal_mechanics(self,
                                      engagement_profile: Dict[str, Any],
                                      engagement_analysis: Dict[str, Any],
                                      learning_context: Dict[str, Any]) -> List[EngagementMechanicType]:
        """Select optimal engagement mechanics"""
        primary_type = engagement_profile['primary_type']
        engagement_level = engagement_analysis['engagement_level']
        needs_intervention = engagement_analysis['needs_intervention']
        
        # Get base mechanics for profile type
        base_mechanics = self.config['engagement_strategies'].get(primary_type, [])
        
        # Add intervention mechanics if needed
        if needs_intervention:
            if engagement_level == 'low':
                # Add high-impact mechanics
                base_mechanics.extend([
                    EngagementMechanicType.SURPRISE_ELEMENTS,
                    EngagementMechanicType.PROGRESSIVE_CHALLENGES
                ])
            
            if engagement_analysis['engagement_trend'] == 'declining':
                # Add re-engagement mechanics
                base_mechanics.append(EngagementMechanicType.EXPLORATION_REWARDS)
        
        # Add secondary profile mechanics
        for secondary_type in engagement_profile.get('secondary_types', []):
            secondary_mechanics = self.config['engagement_strategies'].get(secondary_type, [])
            base_mechanics.extend(secondary_mechanics[:1])  # Add top mechanic from secondary profile
        
        # Remove duplicates and limit count
        unique_mechanics = list(set(base_mechanics))
        
        # Prioritize based on effectiveness
        if hasattr(self, 'mechanic_effectiveness') and self.mechanic_effectiveness:
            unique_mechanics.sort(key=lambda m: self.mechanic_effectiveness.get(m.value, 0.5), reverse=True)
        
        # Limit to 3-5 mechanics to avoid overwhelming
        return unique_mechanics[:5]
    
    async def _configure_mechanics(self,
                                 mechanics: List[EngagementMechanicType],
                                 user_profile: Dict[str, Any],
                                 learning_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Configure mechanics with user-specific parameters"""
        configured_mechanics = []
        
        for mechanic in mechanics:
            config = await self._configure_single_mechanic(mechanic, user_profile, learning_context)
            configured_mechanics.append(config)
        
        return configured_mechanics
    
    async def _configure_single_mechanic(self,
                                       mechanic: EngagementMechanicType,
                                       user_profile: Dict[str, Any],
                                       learning_context: Dict[str, Any]) -> Dict[str, Any]:
        """Configure a single engagement mechanic"""
        base_config = {
            'mechanic_type': mechanic.value,
            'enabled': True,
            'priority': 'medium',
            'personalization_level': 'high'
        }
        
        if mechanic == EngagementMechanicType.PROGRESSIVE_CHALLENGES:
            base_config.update({
                'difficulty_progression_rate': 0.1,
                'challenge_frequency': 'daily',
                'success_threshold': 0.7,
                'adaptive_difficulty': True
            })
        
        elif mechanic == EngagementMechanicType.STREAK_MECHANICS:
            base_config.update({
                'streak_targets': [3, 7, 14, 30],
                'streak_rewards': [50, 150, 400, 1000],
                'streak_recovery_grace_period_hours': 24,
                'streak_visualization': 'calendar'
            })
        
        elif mechanic == EngagementMechanicType.SURPRISE_ELEMENTS:
            base_config.update({
                'surprise_frequency': 0.15,  # 15% chance per session
                'surprise_types': ['bonus_points', 'hidden_achievement', 'special_content'],
                'surprise_timing': 'random',
                'surprise_magnitude': 'medium'
            })
        
        elif mechanic == EngagementMechanicType.SOCIAL_CHALLENGES:
            base_config.update({
                'challenge_types': ['peer_comparison', 'group_goals', 'mentoring'],
                'social_visibility': user_profile.get('social_sharing_preference', 'friends'),
                'collaboration_preference': user_profile.get('collaboration_preference', 'small_groups')
            })
        
        elif mechanic == EngagementMechanicType.TIME_PRESSURE:
            base_config.update({
                'time_pressure_level': 'moderate',
                'countdown_visibility': True,
                'time_bonus_multiplier': 1.5,
                'pressure_adaptation': True
            })
        
        elif mechanic == EngagementMechanicType.EXPLORATION_REWARDS:
            base_config.update({
                'exploration_bonus_rate': 0.2,
                'discovery_rewards': ['new_content', 'bonus_points', 'achievements'],
                'exploration_tracking': 'comprehensive',
                'curiosity_incentives': True
            })
        
        elif mechanic == EngagementMechanicType.MASTERY_PATHS:
            base_config.update({
                'path_visualization': 'skill_tree',
                'mastery_thresholds': [0.6, 0.8, 0.95],
                'path_branching': True,
                'mastery_rewards': 'progressive'
            })
        
        elif mechanic == EngagementMechanicType.COLLABORATIVE_GOALS:
            base_config.update({
                'group_size_preference': user_profile.get('group_size_preference', 'small'),
                'collaboration_style': user_profile.get('collaboration_style', 'cooperative'),
                'shared_progress_visibility': True,
                'collective_rewards': True
            })
        
        return base_config
    
    async def _create_implementation_plan(self, configured_mechanics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create implementation plan for mechanics"""
        # Prioritize mechanics
        high_priority = [m for m in configured_mechanics if m.get('priority') == 'high']
        medium_priority = [m for m in configured_mechanics if m.get('priority') == 'medium']
        low_priority = [m for m in configured_mechanics if m.get('priority') == 'low']
        
        # Create phased rollout
        implementation_phases = []
        
        if high_priority:
            implementation_phases.append({
                'phase': 1,
                'mechanics': high_priority,
                'timeline': 'immediate',
                'success_metrics': ['engagement_increase', 'mechanic_adoption']
            })
        
        if medium_priority:
            implementation_phases.append({
                'phase': 2,
                'mechanics': medium_priority,
                'timeline': '1_week',
                'success_metrics': ['sustained_engagement', 'mechanic_effectiveness']
            })
        
        if low_priority:
            implementation_phases.append({
                'phase': 3,
                'mechanics': low_priority,
                'timeline': '2_weeks',
                'success_metrics': ['long_term_engagement', 'mechanic_synergy']
            })
        
        return {
            'implementation_phases': implementation_phases,
            'total_mechanics': len(configured_mechanics),
            'estimated_setup_time': len(configured_mechanics) * 2,  # 2 minutes per mechanic
            'monitoring_plan': {
                'metrics_to_track': ['engagement_score', 'mechanic_usage', 'user_satisfaction'],
                'monitoring_frequency': 'daily',
                'adjustment_triggers': ['engagement_drop', 'mechanic_fatigue', 'user_feedback']
            }
        }


class ChallengeGenerationSystem:
    """
    🏁 CHALLENGE GENERATION SYSTEM
    
    Advanced system for generating personalized, dynamic challenges.
    """
    
    def __init__(self):
        # Generation configuration
        self.config = {
            'challenge_templates': {
                ChallengeType.SKILL_CHALLENGE: [
                    "Master {skill} in {timeframe}",
                    "Achieve {accuracy}% accuracy in {domain}",
                    "Complete {count} {skill} exercises"
                ],
                ChallengeType.SPEED_CHALLENGE: [
                    "Complete {task} in under {time_limit} minutes",
                    "Solve {count} problems in {timeframe}",
                    "Beat your best time in {activity}"
                ],
                ChallengeType.CREATIVITY_CHALLENGE: [
                    "Create an innovative solution for {problem}",
                    "Design {count} unique approaches to {challenge}",
                    "Combine {skill1} and {skill2} creatively"
                ],
                ChallengeType.COLLABORATION_CHALLENGE: [
                    "Work with {count} peers to solve {problem}",
                    "Mentor {count} learners in {skill}",
                    "Lead a team project in {domain}"
                ]
            },
            'difficulty_factors': {
                'beginner': {'time_multiplier': 2.0, 'accuracy_threshold': 0.6},
                'intermediate': {'time_multiplier': 1.5, 'accuracy_threshold': 0.75},
                'advanced': {'time_multiplier': 1.0, 'accuracy_threshold': 0.9},
                'expert': {'time_multiplier': 0.8, 'accuracy_threshold': 0.95}
            }
        }
        
        # Generation tracking
        self.generated_challenges = {}
        self.challenge_history = []
        
        logger.info("Challenge Generation System initialized")
    
    async def generate_personalized_challenge(self,
                                            user_profile: Dict[str, Any],
                                            learning_context: Dict[str, Any],
                                            challenge_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate personalized challenge for user
        
        Args:
            user_profile: User's profile and skill levels
            learning_context: Current learning context
            challenge_preferences: User's challenge preferences
            
        Returns:
            Dict with generated challenge
        """
        try:
            # Determine challenge type
            challenge_type = await self._determine_challenge_type(user_profile, challenge_preferences)
            
            # Calculate difficulty level
            difficulty_level = await self._calculate_difficulty_level(user_profile, learning_context)
            
            # Generate challenge content
            challenge_content = await self._generate_challenge_content(
                challenge_type, difficulty_level, user_profile, learning_context
            )
            
            # Create success criteria
            success_criteria = await self._create_success_criteria(challenge_type, difficulty_level)
            
            # Calculate rewards
            rewards = await self._calculate_challenge_rewards(challenge_type, difficulty_level)
            
            # Add engagement hooks
            engagement_hooks = await self._create_engagement_hooks(challenge_type, user_profile)
            
            # Create dynamic challenge
            challenge_id = f"challenge_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{user_profile.get('user_id', 'unknown')}"
            
            dynamic_challenge = DynamicChallenge(
                challenge_id=challenge_id,
                title=challenge_content['title'],
                description=challenge_content['description'],
                challenge_type=challenge_type,
                difficulty_level=difficulty_level,
                estimated_duration_minutes=challenge_content['duration'],
                success_criteria=success_criteria,
                reward_points=rewards['points'],
                bonus_conditions=rewards['bonus_conditions'],
                personalization_factors=challenge_content['personalization_factors'],
                is_time_limited=challenge_content.get('is_time_limited', False),
                expires_at=challenge_content.get('expires_at'),
                created_at=datetime.utcnow().isoformat(),
                engagement_hooks=engagement_hooks
            )
            
            # Store challenge
            self.generated_challenges[challenge_id] = dynamic_challenge
            
            # Track generation
            self.challenge_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'user_id': user_profile.get('user_id'),
                'challenge_type': challenge_type.value,
                'difficulty_level': difficulty_level,
                'estimated_duration': challenge_content['duration']
            })
            
            return {
                'status': 'success',
                'dynamic_challenge': dynamic_challenge.__dict__,
                'generation_rationale': {
                    'type_selection': f"Selected {challenge_type.value} based on user preferences",
                    'difficulty_calculation': f"Difficulty {difficulty_level:.2f} based on skill level",
                    'personalization_applied': len(challenge_content['personalization_factors'])
                },
                'generation_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating personalized challenge: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _determine_challenge_type(self,
                                      user_profile: Dict[str, Any],
                                      challenge_preferences: Dict[str, Any]) -> ChallengeType:
        """Determine optimal challenge type for user"""
        # Check explicit preferences
        preferred_types = challenge_preferences.get('preferred_types', [])
        if preferred_types:
            try:
                return ChallengeType(random.choice(preferred_types))
            except ValueError:
                pass
        
        # Determine based on user characteristics
        personality = user_profile.get('personality', {})
        learning_style = user_profile.get('learning_style', {})
        
        type_scores = {
            ChallengeType.SKILL_CHALLENGE: 0.5,  # Base score
            ChallengeType.SPEED_CHALLENGE: 0.3,
            ChallengeType.ACCURACY_CHALLENGE: 0.4,
            ChallengeType.CREATIVITY_CHALLENGE: 0.3,
            ChallengeType.ENDURANCE_CHALLENGE: 0.2,
            ChallengeType.COLLABORATION_CHALLENGE: 0.3,
            ChallengeType.EXPLORATION_CHALLENGE: 0.4,
            ChallengeType.INNOVATION_CHALLENGE: 0.2
        }
        
        # Adjust based on personality
        if personality.get('openness', 0.5) > 0.7:
            type_scores[ChallengeType.CREATIVITY_CHALLENGE] += 0.3
            type_scores[ChallengeType.INNOVATION_CHALLENGE] += 0.3
        
        if personality.get('conscientiousness', 0.5) > 0.7:
            type_scores[ChallengeType.ACCURACY_CHALLENGE] += 0.3
            type_scores[ChallengeType.ENDURANCE_CHALLENGE] += 0.2
        
        if personality.get('extraversion', 0.5) > 0.6:
            type_scores[ChallengeType.COLLABORATION_CHALLENGE] += 0.4
        
        # Adjust based on learning style
        if learning_style.get('kinesthetic', 0.5) > 0.6:
            type_scores[ChallengeType.SPEED_CHALLENGE] += 0.2
        
        if learning_style.get('visual', 0.5) > 0.6:
            type_scores[ChallengeType.EXPLORATION_CHALLENGE] += 0.2
        
        # Select highest scoring type
        best_type = max(type_scores.items(), key=lambda x: x[1])
        return best_type[0]
    
    async def _calculate_difficulty_level(self,
                                        user_profile: Dict[str, Any],
                                        learning_context: Dict[str, Any]) -> float:
        """Calculate appropriate difficulty level"""
        # Base difficulty on user's skill level
        skills = user_profile.get('skills', {})
        avg_skill_level = sum(skills.values()) / len(skills) if skills else 0.5
        
        # Adjust based on recent performance
        recent_performance = learning_context.get('recent_performance', 0.7)
        
        # Adjust based on challenge preference
        challenge_preference = user_profile.get('challenge_preference', 0.5)
        
        # Calculate difficulty
        difficulty = (avg_skill_level * 0.4 + recent_performance * 0.3 + challenge_preference * 0.3)
        
        # Add some randomness for variety
        difficulty += random.uniform(-0.1, 0.1)
        
        return max(0.1, min(1.0, difficulty))
    
    async def _generate_challenge_content(self,
                                        challenge_type: ChallengeType,
                                        difficulty_level: float,
                                        user_profile: Dict[str, Any],
                                        learning_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate challenge content"""
        templates = self.config['challenge_templates'].get(challenge_type, ["Complete a challenge"])
        template = random.choice(templates)
        
        # Get user's current focus areas
        current_skills = list(user_profile.get('skills', {}).keys())
        learning_goals = user_profile.get('learning_goals', [])
        
        # Select relevant skill/domain
        if learning_goals:
            focus_area = random.choice(learning_goals)
        elif current_skills:
            focus_area = random.choice(current_skills)
        else:
            focus_area = "your learning area"
        
        # Generate content based on difficulty
        if challenge_type == ChallengeType.SKILL_CHALLENGE:
            title = template.format(
                skill=focus_area,
                timeframe=f"{int(7 + difficulty_level * 14)} days",
                accuracy=int(60 + difficulty_level * 30),
                domain=focus_area,
                count=int(5 + difficulty_level * 15)
            )
            description = f"Focus on improving your {focus_area} skills through targeted practice and assessment."
            duration = int(30 + difficulty_level * 60)
        
        elif challenge_type == ChallengeType.SPEED_CHALLENGE:
            title = template.format(
                task=f"{focus_area} exercises",
                time_limit=int(30 - difficulty_level * 15),
                count=int(3 + difficulty_level * 7),
                timeframe=f"{int(60 - difficulty_level * 30)} minutes",
                activity=focus_area
            )
            description = f"Test your speed and efficiency in {focus_area} with time-based challenges."
            duration = int(15 + difficulty_level * 30)
        
        elif challenge_type == ChallengeType.CREATIVITY_CHALLENGE:
            title = template.format(
                problem=f"{focus_area} challenge",
                count=int(2 + difficulty_level * 3),
                challenge=focus_area,
                skill1=focus_area,
                skill2=random.choice(current_skills) if len(current_skills) > 1 else "problem-solving"
            )
            description = f"Unleash your creativity to solve {focus_area} problems in innovative ways."
            duration = int(45 + difficulty_level * 75)
        
        else:
            title = template.format(
                skill=focus_area,
                count=int(2 + difficulty_level * 3),
                problem=f"{focus_area} challenge",
                domain=focus_area
            )
            description = f"Take on this {challenge_type.value.replace('_', ' ')} to advance your skills."
            duration = int(30 + difficulty_level * 60)
        
        # Determine if time-limited
        is_time_limited = random.random() < 0.3  # 30% chance
        expires_at = None
        if is_time_limited:
            expires_at = (datetime.utcnow() + timedelta(days=int(3 + difficulty_level * 4))).isoformat()
        
        return {
            'title': title,
            'description': description,
            'duration': duration,
            'is_time_limited': is_time_limited,
            'expires_at': expires_at,
            'personalization_factors': ['skill_focus', 'difficulty_level', 'duration_preference']
        }
    
    async def _create_success_criteria(self,
                                     challenge_type: ChallengeType,
                                     difficulty_level: float) -> Dict[str, Any]:
        """Create success criteria for challenge"""
        base_criteria = {
            'completion_required': True,
            'minimum_score': 0.6 + difficulty_level * 0.3
        }
        
        if challenge_type == ChallengeType.SKILL_CHALLENGE:
            base_criteria.update({
                'skill_improvement_required': 0.1 + difficulty_level * 0.2,
                'practice_sessions_minimum': int(3 + difficulty_level * 5)
            })
        
        elif challenge_type == ChallengeType.SPEED_CHALLENGE:
            base_criteria.update({
                'time_limit_adherence': True,
                'speed_improvement_target': 0.15 + difficulty_level * 0.25
            })
        
        elif challenge_type == ChallengeType.ACCURACY_CHALLENGE:
            base_criteria.update({
                'accuracy_threshold': 0.7 + difficulty_level * 0.25,
                'error_rate_maximum': 0.3 - difficulty_level * 0.2
            })
        
        elif challenge_type == ChallengeType.CREATIVITY_CHALLENGE:
            base_criteria.update({
                'originality_score_minimum': 0.6 + difficulty_level * 0.3,
                'solution_diversity_required': int(2 + difficulty_level * 2)
            })
        
        return base_criteria
    
    async def _calculate_challenge_rewards(self,
                                         challenge_type: ChallengeType,
                                         difficulty_level: float) -> Dict[str, Any]:
        """Calculate rewards for challenge"""
        # Base points by type
        base_points = {
            ChallengeType.SKILL_CHALLENGE: 150,
            ChallengeType.SPEED_CHALLENGE: 100,
            ChallengeType.ACCURACY_CHALLENGE: 120,
            ChallengeType.CREATIVITY_CHALLENGE: 200,
            ChallengeType.ENDURANCE_CHALLENGE: 180,
            ChallengeType.COLLABORATION_CHALLENGE: 160,
            ChallengeType.EXPLORATION_CHALLENGE: 140,
            ChallengeType.INNOVATION_CHALLENGE: 250
        }
        
        base_reward = base_points.get(challenge_type, 150)
        difficulty_multiplier = 1.0 + difficulty_level
        
        points = int(base_reward * difficulty_multiplier)
        
        # Create bonus conditions
        bonus_conditions = []
        
        if difficulty_level > 0.7:
            bonus_conditions.append({
                'condition': 'perfect_completion',
                'description': 'Complete with 100% accuracy',
                'bonus_points': int(points * 0.5)
            })
        
        if challenge_type in [ChallengeType.SPEED_CHALLENGE, ChallengeType.ENDURANCE_CHALLENGE]:
            bonus_conditions.append({
                'condition': 'time_bonus',
                'description': 'Complete ahead of schedule',
                'bonus_points': int(points * 0.3)
            })
        
        bonus_conditions.append({
            'condition': 'first_attempt_success',
            'description': 'Succeed on first attempt',
            'bonus_points': int(points * 0.2)
        })
        
        return {
            'points': points,
            'bonus_conditions': bonus_conditions,
            'base_reward': base_reward,
            'difficulty_multiplier': difficulty_multiplier
        }
    
    async def _create_engagement_hooks(self,
                                     challenge_type: ChallengeType,
                                     user_profile: Dict[str, Any]) -> List[str]:
        """Create engagement hooks for challenge"""
        hooks = []
        
        # Type-specific hooks
        if challenge_type == ChallengeType.SPEED_CHALLENGE:
            hooks.extend(['countdown_timer', 'speed_visualization', 'personal_best_tracking'])
        
        elif challenge_type == ChallengeType.CREATIVITY_CHALLENGE:
            hooks.extend(['inspiration_gallery', 'creative_prompts', 'idea_sharing'])
        
        elif challenge_type == ChallengeType.COLLABORATION_CHALLENGE:
            hooks.extend(['team_progress', 'peer_communication', 'shared_achievements'])
        
        # User preference hooks
        personality = user_profile.get('personality', {})
        
        if personality.get('extraversion', 0.5) > 0.6:
            hooks.append('social_sharing_prompts')
        
        if personality.get('openness', 0.5) > 0.7:
            hooks.append('exploration_bonuses')
        
        if personality.get('conscientiousness', 0.5) > 0.7:
            hooks.append('progress_tracking')
        
        # General engagement hooks
        hooks.extend(['progress_celebration', 'milestone_notifications', 'achievement_preview'])
        
        return list(set(hooks))  # Remove duplicates
