"""
Achievement Engine Services

Extracted from quantum_intelligence_engine.py (lines 12526-15023) - dynamic achievement
generation, achievement tracking, badge design, and mastery progress tracking for
comprehensive gamification systems.
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


class AchievementType(Enum):
    """Types of dynamic achievements"""
    LEARNING_MILESTONE = "learning_milestone"
    STREAK_ACHIEVEMENT = "streak_achievement"
    MASTERY_BADGE = "mastery_badge"
    SOCIAL_RECOGNITION = "social_recognition"
    INNOVATION_AWARD = "innovation_award"
    PERSISTENCE_TROPHY = "persistence_trophy"
    COLLABORATION_HONOR = "collaboration_honor"
    BREAKTHROUGH_MEDAL = "breakthrough_medal"


class AchievementRarity(Enum):
    """Achievement rarity levels"""
    COMMON = "common"
    UNCOMMON = "uncommon"
    RARE = "rare"
    EPIC = "epic"
    LEGENDARY = "legendary"


class BadgeDesignStyle(Enum):
    """Badge design styles"""
    MINIMALIST = "minimalist"
    DETAILED = "detailed"
    ARTISTIC = "artistic"
    TECHNICAL = "technical"
    PLAYFUL = "playful"


@dataclass
class DynamicAchievement:
    """Dynamic AI-generated achievement"""
    achievement_id: str = ""
    title: str = ""
    description: str = ""
    category: AchievementType = AchievementType.LEARNING_MILESTONE
    difficulty_tier: int = 1  # 1-10
    points_reward: int = 100
    badge_design: Dict[str, Any] = field(default_factory=dict)
    unlock_criteria: Dict[str, Any] = field(default_factory=dict)
    personalization_factors: List[str] = field(default_factory=list)
    rarity_score: float = 0.5  # 0.0-1.0
    social_sharing_bonus: int = 0
    created_at: str = ""
    expires_at: Optional[str] = None
    is_limited_time: bool = False
    progress_tracking: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AchievementProgress:
    """User's progress toward an achievement"""
    user_id: str = ""
    achievement_id: str = ""
    current_progress: float = 0.0  # 0.0-1.0
    progress_milestones: List[Dict[str, Any]] = field(default_factory=list)
    started_at: str = ""
    last_updated: str = ""
    estimated_completion: Optional[str] = None
    is_completed: bool = False
    completed_at: Optional[str] = None


@dataclass
class BadgeDesign:
    """Badge visual design specification"""
    badge_id: str = ""
    design_style: BadgeDesignStyle = BadgeDesignStyle.MINIMALIST
    color_scheme: Dict[str, str] = field(default_factory=dict)
    icon_elements: List[str] = field(default_factory=list)
    shape: str = "circular"
    size: str = "medium"
    animation_effects: List[str] = field(default_factory=list)
    rarity_indicators: Dict[str, Any] = field(default_factory=dict)
    personalization_elements: Dict[str, Any] = field(default_factory=dict)


class DynamicAchievementGenerator:
    """
    🏆 DYNAMIC ACHIEVEMENT GENERATOR
    
    Advanced generator for creating personalized, dynamic achievements.
    Extracted from the original quantum engine's gamification logic.
    """
    
    def __init__(self, cache_service: Optional[CacheService] = None):
        self.cache = cache_service
        
        # Generation configuration
        self.config = {
            'achievement_templates': {
                AchievementType.LEARNING_MILESTONE: [
                    "Complete {count} lessons in {subject}",
                    "Master {skill_level} level {skill_name}",
                    "Achieve {percentage}% accuracy in {domain}"
                ],
                AchievementType.STREAK_ACHIEVEMENT: [
                    "Maintain a {days}-day learning streak",
                    "Complete daily goals for {weeks} weeks",
                    "Study consistently for {months} months"
                ],
                AchievementType.MASTERY_BADGE: [
                    "Demonstrate mastery in {subject}",
                    "Excel in {skill_category} skills",
                    "Become an expert in {domain}"
                ],
                AchievementType.SOCIAL_RECOGNITION: [
                    "Help {count} fellow learners",
                    "Receive {likes} likes on contributions",
                    "Mentor {students} students successfully"
                ]
            },
            'difficulty_scaling': {
                1: {'multiplier': 1.0, 'rarity': 0.1},
                2: {'multiplier': 1.2, 'rarity': 0.2},
                3: {'multiplier': 1.5, 'rarity': 0.3},
                4: {'multiplier': 2.0, 'rarity': 0.4},
                5: {'multiplier': 2.5, 'rarity': 0.5},
                6: {'multiplier': 3.0, 'rarity': 0.6},
                7: {'multiplier': 4.0, 'rarity': 0.7},
                8: {'multiplier': 5.0, 'rarity': 0.8},
                9: {'multiplier': 7.0, 'rarity': 0.9},
                10: {'multiplier': 10.0, 'rarity': 0.95}
            },
            'personalization_factors': [
                'learning_style', 'skill_level', 'interests', 'goals',
                'progress_rate', 'social_preference', 'challenge_preference'
            ]
        }
        
        # Generation tracking
        self.generated_achievements = {}
        self.generation_history = []
        
        logger.info("Dynamic Achievement Generator initialized")
    
    async def generate_personalized_achievement(self,
                                              user_profile: Dict[str, Any],
                                              learning_context: Dict[str, Any],
                                              achievement_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate personalized achievement for user
        
        Args:
            user_profile: User's learning profile and preferences
            learning_context: Current learning context and progress
            achievement_preferences: User's achievement preferences
            
        Returns:
            Dict with generated personalized achievement
        """
        try:
            # Analyze user context for achievement generation
            context_analysis = await self._analyze_achievement_context(user_profile, learning_context)
            
            # Determine optimal achievement type
            achievement_type = await self._determine_achievement_type(context_analysis, achievement_preferences)
            
            # Calculate difficulty tier
            difficulty_tier = await self._calculate_difficulty_tier(user_profile, learning_context)
            
            # Generate achievement content
            achievement_content = await self._generate_achievement_content(
                achievement_type, difficulty_tier, context_analysis
            )
            
            # Create unlock criteria
            unlock_criteria = await self._create_unlock_criteria(
                achievement_type, difficulty_tier, context_analysis
            )
            
            # Calculate rewards
            rewards = await self._calculate_achievement_rewards(achievement_type, difficulty_tier)
            
            # Determine personalization factors
            personalization_factors = await self._identify_personalization_factors(
                user_profile, achievement_type
            )
            
            # Calculate rarity score
            rarity_score = self.config['difficulty_scaling'][difficulty_tier]['rarity']
            
            # Create dynamic achievement
            achievement_id = f"ach_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{user_profile.get('user_id', 'unknown')}"
            
            dynamic_achievement = DynamicAchievement(
                achievement_id=achievement_id,
                title=achievement_content['title'],
                description=achievement_content['description'],
                category=achievement_type,
                difficulty_tier=difficulty_tier,
                points_reward=rewards['points'],
                unlock_criteria=unlock_criteria,
                personalization_factors=personalization_factors,
                rarity_score=rarity_score,
                social_sharing_bonus=rewards['social_bonus'],
                created_at=datetime.utcnow().isoformat(),
                expires_at=achievement_content.get('expires_at'),
                is_limited_time=achievement_content.get('is_limited_time', False),
                progress_tracking={'current_progress': 0.0, 'milestones': []}
            )
            
            # Store generated achievement
            self.generated_achievements[achievement_id] = dynamic_achievement
            
            # Track generation
            self.generation_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'user_id': user_profile.get('user_id'),
                'achievement_type': achievement_type.value,
                'difficulty_tier': difficulty_tier,
                'rarity_score': rarity_score
            })
            
            return {
                'status': 'success',
                'dynamic_achievement': dynamic_achievement.__dict__,
                'generation_analysis': context_analysis,
                'generation_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating personalized achievement: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _analyze_achievement_context(self,
                                         user_profile: Dict[str, Any],
                                         learning_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze context for achievement generation"""
        # Analyze current learning progress
        current_skills = user_profile.get('skills', {})
        learning_goals = user_profile.get('learning_goals', [])
        recent_activities = learning_context.get('recent_activities', [])
        
        # Identify achievement opportunities
        opportunities = []
        
        # Skill-based opportunities
        for skill, level in current_skills.items():
            if level > 0.7:  # Near mastery
                opportunities.append({
                    'type': 'mastery_opportunity',
                    'skill': skill,
                    'current_level': level,
                    'potential': 'high'
                })
            elif level > 0.4:  # Intermediate
                opportunities.append({
                    'type': 'improvement_opportunity',
                    'skill': skill,
                    'current_level': level,
                    'potential': 'medium'
                })
        
        # Goal-based opportunities
        for goal in learning_goals:
            goal_progress = learning_context.get('goal_progress', {}).get(goal, 0.0)
            if goal_progress > 0.5:
                opportunities.append({
                    'type': 'goal_completion_opportunity',
                    'goal': goal,
                    'progress': goal_progress,
                    'potential': 'high'
                })
        
        # Activity-based opportunities
        activity_patterns = self._analyze_activity_patterns(recent_activities)
        if activity_patterns.get('consistency_score', 0.0) > 0.7:
            opportunities.append({
                'type': 'consistency_opportunity',
                'pattern': 'high_consistency',
                'potential': 'medium'
            })
        
        return {
            'opportunities': opportunities,
            'activity_patterns': activity_patterns,
            'skill_distribution': current_skills,
            'goal_alignment': learning_goals,
            'context_strength': len(opportunities) / max(1, len(current_skills) + len(learning_goals))
        }
    
    def _analyze_activity_patterns(self, recent_activities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze user's recent activity patterns"""
        if not recent_activities:
            return {'consistency_score': 0.0, 'activity_frequency': 0.0}
        
        # Calculate consistency
        activity_dates = [
            datetime.fromisoformat(activity.get('timestamp', datetime.utcnow().isoformat())).date()
            for activity in recent_activities
        ]
        
        unique_dates = set(activity_dates)
        date_range = (max(activity_dates) - min(activity_dates)).days + 1 if activity_dates else 1
        consistency_score = len(unique_dates) / date_range
        
        # Calculate frequency
        activity_frequency = len(recent_activities) / max(1, date_range)
        
        return {
            'consistency_score': consistency_score,
            'activity_frequency': activity_frequency,
            'total_activities': len(recent_activities),
            'active_days': len(unique_dates),
            'date_range': date_range
        }
    
    async def _determine_achievement_type(self,
                                        context_analysis: Dict[str, Any],
                                        achievement_preferences: Dict[str, Any]) -> AchievementType:
        """Determine optimal achievement type based on context"""
        opportunities = context_analysis.get('opportunities', [])
        
        # Score different achievement types based on opportunities
        type_scores = {
            AchievementType.LEARNING_MILESTONE: 0.0,
            AchievementType.STREAK_ACHIEVEMENT: 0.0,
            AchievementType.MASTERY_BADGE: 0.0,
            AchievementType.SOCIAL_RECOGNITION: 0.0,
            AchievementType.INNOVATION_AWARD: 0.0,
            AchievementType.PERSISTENCE_TROPHY: 0.0,
            AchievementType.COLLABORATION_HONOR: 0.0,
            AchievementType.BREAKTHROUGH_MEDAL: 0.0
        }
        
        for opportunity in opportunities:
            if opportunity['type'] == 'mastery_opportunity':
                type_scores[AchievementType.MASTERY_BADGE] += opportunity.get('potential_score', 1.0)
            elif opportunity['type'] == 'goal_completion_opportunity':
                type_scores[AchievementType.LEARNING_MILESTONE] += opportunity.get('potential_score', 1.0)
            elif opportunity['type'] == 'consistency_opportunity':
                type_scores[AchievementType.STREAK_ACHIEVEMENT] += opportunity.get('potential_score', 1.0)
            elif opportunity['type'] == 'social_opportunity':
                type_scores[AchievementType.SOCIAL_RECOGNITION] += opportunity.get('potential_score', 1.0)
        
        # Apply user preferences
        preferred_types = achievement_preferences.get('preferred_types', [])
        for pref_type in preferred_types:
            try:
                achievement_type = AchievementType(pref_type)
                type_scores[achievement_type] *= 1.5  # Boost preferred types
            except ValueError:
                pass
        
        # Select highest scoring type
        best_type = max(type_scores.items(), key=lambda x: x[1])
        return best_type[0] if best_type[1] > 0 else AchievementType.LEARNING_MILESTONE
    
    async def _calculate_difficulty_tier(self,
                                       user_profile: Dict[str, Any],
                                       learning_context: Dict[str, Any]) -> int:
        """Calculate appropriate difficulty tier for user"""
        # Base difficulty on user's skill level
        skills = user_profile.get('skills', {})
        avg_skill_level = sum(skills.values()) / len(skills) if skills else 0.5
        
        # Adjust based on recent performance
        recent_performance = learning_context.get('recent_performance_score', 0.7)
        
        # Adjust based on challenge preference
        challenge_preference = user_profile.get('challenge_preference', 0.5)
        
        # Calculate base difficulty
        base_difficulty = (avg_skill_level * 0.4 + recent_performance * 0.3 + challenge_preference * 0.3)
        
        # Map to difficulty tier (1-10)
        difficulty_tier = max(1, min(10, int(base_difficulty * 10) + 1))
        
        # Adjust for user experience
        user_level = user_profile.get('user_level', 1)
        if user_level > 10:
            difficulty_tier = min(10, difficulty_tier + 1)
        elif user_level < 5:
            difficulty_tier = max(1, difficulty_tier - 1)
        
        return difficulty_tier
    
    async def _generate_achievement_content(self,
                                          achievement_type: AchievementType,
                                          difficulty_tier: int,
                                          context_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate achievement title and description"""
        templates = self.config['achievement_templates'].get(achievement_type, [])
        if not templates:
            templates = ["Complete a challenging task"]
        
        # Select template
        template = random.choice(templates)
        
        # Generate content based on context
        opportunities = context_analysis.get('opportunities', [])
        relevant_opportunity = None
        
        for opp in opportunities:
            if (achievement_type == AchievementType.MASTERY_BADGE and opp['type'] == 'mastery_opportunity') or \
               (achievement_type == AchievementType.LEARNING_MILESTONE and opp['type'] == 'goal_completion_opportunity'):
                relevant_opportunity = opp
                break
        
        # Fill template with context-specific values
        if relevant_opportunity:
            if achievement_type == AchievementType.MASTERY_BADGE:
                title = f"Master {relevant_opportunity.get('skill', 'Advanced Skills')}"
                description = f"Demonstrate mastery in {relevant_opportunity.get('skill', 'your chosen field')} by achieving expert-level performance"
            elif achievement_type == AchievementType.LEARNING_MILESTONE:
                title = f"Complete {relevant_opportunity.get('goal', 'Learning Goal')}"
                description = f"Successfully complete your learning goal: {relevant_opportunity.get('goal', 'advanced learning objective')}"
            else:
                title = template.format(
                    count=difficulty_tier * 5,
                    subject="your current focus area",
                    days=difficulty_tier * 3,
                    percentage=70 + difficulty_tier * 3
                )
                description = f"Challenge yourself with this {difficulty_tier}-tier achievement"
        else:
            # Generic content based on difficulty
            title = template.format(
                count=difficulty_tier * 5,
                subject="your learning area",
                skill_level="intermediate" if difficulty_tier < 5 else "advanced",
                skill_name="core skills",
                days=difficulty_tier * 3,
                weeks=difficulty_tier,
                months=max(1, difficulty_tier // 3),
                percentage=70 + difficulty_tier * 3,
                likes=difficulty_tier * 10,
                students=difficulty_tier * 2
            )
            description = f"A tier-{difficulty_tier} achievement that will challenge and reward your progress"
        
        # Determine if limited time
        is_limited_time = random.random() < 0.2  # 20% chance of limited time
        expires_at = None
        
        if is_limited_time:
            expires_at = (datetime.utcnow() + timedelta(days=7 + difficulty_tier)).isoformat()
        
        return {
            'title': title,
            'description': description,
            'is_limited_time': is_limited_time,
            'expires_at': expires_at
        }
    
    async def _create_unlock_criteria(self,
                                    achievement_type: AchievementType,
                                    difficulty_tier: int,
                                    context_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create unlock criteria for achievement"""
        base_criteria = {
            'type': achievement_type.value,
            'difficulty_tier': difficulty_tier,
            'requirements': []
        }
        
        # Type-specific criteria
        if achievement_type == AchievementType.LEARNING_MILESTONE:
            base_criteria['requirements'] = [
                {'type': 'lessons_completed', 'target': difficulty_tier * 5},
                {'type': 'skill_level_achieved', 'target': 0.6 + difficulty_tier * 0.04}
            ]
        
        elif achievement_type == AchievementType.STREAK_ACHIEVEMENT:
            base_criteria['requirements'] = [
                {'type': 'consecutive_days', 'target': difficulty_tier * 3},
                {'type': 'daily_goal_completion', 'target': 0.8}
            ]
        
        elif achievement_type == AchievementType.MASTERY_BADGE:
            base_criteria['requirements'] = [
                {'type': 'skill_mastery_level', 'target': 0.8 + difficulty_tier * 0.02},
                {'type': 'assessment_score', 'target': 0.85 + difficulty_tier * 0.015}
            ]
        
        elif achievement_type == AchievementType.SOCIAL_RECOGNITION:
            base_criteria['requirements'] = [
                {'type': 'peer_interactions', 'target': difficulty_tier * 10},
                {'type': 'helpful_contributions', 'target': difficulty_tier * 3}
            ]
        
        else:
            # Generic criteria
            base_criteria['requirements'] = [
                {'type': 'general_progress', 'target': 0.7 + difficulty_tier * 0.03},
                {'type': 'time_investment_hours', 'target': difficulty_tier * 5}
            ]
        
        return base_criteria
    
    async def _calculate_achievement_rewards(self,
                                           achievement_type: AchievementType,
                                           difficulty_tier: int) -> Dict[str, Any]:
        """Calculate rewards for achievement"""
        # Base points by type
        base_points = {
            AchievementType.LEARNING_MILESTONE: 100,
            AchievementType.STREAK_ACHIEVEMENT: 80,
            AchievementType.MASTERY_BADGE: 150,
            AchievementType.SOCIAL_RECOGNITION: 120,
            AchievementType.INNOVATION_AWARD: 200,
            AchievementType.PERSISTENCE_TROPHY: 180,
            AchievementType.COLLABORATION_HONOR: 140,
            AchievementType.BREAKTHROUGH_MEDAL: 250
        }
        
        base_reward = base_points.get(achievement_type, 100)
        difficulty_multiplier = self.config['difficulty_scaling'][difficulty_tier]['multiplier']
        
        points_reward = int(base_reward * difficulty_multiplier)
        
        # Social sharing bonus
        social_bonus = int(points_reward * 0.2) if difficulty_tier >= 5 else 0
        
        return {
            'points': points_reward,
            'social_bonus': social_bonus,
            'base_reward': base_reward,
            'difficulty_multiplier': difficulty_multiplier
        }
    
    async def _identify_personalization_factors(self,
                                              user_profile: Dict[str, Any],
                                              achievement_type: AchievementType) -> List[str]:
        """Identify personalization factors for achievement"""
        factors = []
        
        # Always include basic factors
        factors.extend(['user_level', 'achievement_type'])
        
        # Add factors based on user profile
        if user_profile.get('learning_style'):
            factors.append('learning_style')
        
        if user_profile.get('interests'):
            factors.append('interests')
        
        if user_profile.get('goals'):
            factors.append('goals')
        
        # Add type-specific factors
        if achievement_type in [AchievementType.SOCIAL_RECOGNITION, AchievementType.COLLABORATION_HONOR]:
            factors.append('social_preference')
        
        if achievement_type in [AchievementType.MASTERY_BADGE, AchievementType.BREAKTHROUGH_MEDAL]:
            factors.append('challenge_preference')
        
        return factors


class AchievementTrackingSystem:
    """
    📊 ACHIEVEMENT TRACKING SYSTEM
    
    Advanced system for tracking user progress toward achievements.
    """
    
    def __init__(self, achievement_generator: DynamicAchievementGenerator):
        self.achievement_generator = achievement_generator
        
        # Tracking configuration
        self.config = {
            'progress_update_frequency': 'real_time',
            'milestone_notification_threshold': 0.25,  # Notify every 25% progress
            'completion_verification_required': True,
            'progress_decay_enabled': False
        }
        
        # Tracking data
        self.user_progress = {}  # user_id -> {achievement_id -> AchievementProgress}
        self.tracking_history = []
        
        logger.info("Achievement Tracking System initialized")
    
    async def track_achievement_progress(self,
                                       user_id: str,
                                       achievement_id: str,
                                       activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Track user's progress toward a specific achievement
        
        Args:
            user_id: User identifier
            achievement_id: Achievement identifier
            activity_data: Recent activity data for progress calculation
            
        Returns:
            Dict with updated progress information
        """
        try:
            # Get achievement details
            if achievement_id not in self.achievement_generator.generated_achievements:
                return {'status': 'error', 'error': 'Achievement not found'}
            
            achievement = self.achievement_generator.generated_achievements[achievement_id]
            
            # Get or create progress tracking
            if user_id not in self.user_progress:
                self.user_progress[user_id] = {}
            
            if achievement_id not in self.user_progress[user_id]:
                self.user_progress[user_id][achievement_id] = AchievementProgress(
                    user_id=user_id,
                    achievement_id=achievement_id,
                    started_at=datetime.utcnow().isoformat()
                )
            
            progress = self.user_progress[user_id][achievement_id]
            
            # Calculate current progress
            new_progress = await self._calculate_progress(achievement, activity_data)
            
            # Update progress
            old_progress = progress.current_progress
            progress.current_progress = new_progress
            progress.last_updated = datetime.utcnow().isoformat()
            
            # Check for milestones
            milestones_reached = await self._check_milestones(progress, old_progress, new_progress)
            
            # Check for completion
            completion_result = await self._check_completion(achievement, progress)
            
            # Update estimated completion
            if not progress.is_completed and new_progress > 0:
                progress.estimated_completion = await self._estimate_completion(progress, activity_data)
            
            # Track progress update
            self.tracking_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'user_id': user_id,
                'achievement_id': achievement_id,
                'old_progress': old_progress,
                'new_progress': new_progress,
                'milestones_reached': len(milestones_reached)
            })
            
            return {
                'status': 'success',
                'progress': progress.__dict__,
                'milestones_reached': milestones_reached,
                'completion_result': completion_result,
                'progress_delta': new_progress - old_progress,
                'tracking_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error tracking achievement progress for user {user_id}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _calculate_progress(self,
                                achievement: DynamicAchievement,
                                activity_data: Dict[str, Any]) -> float:
        """Calculate progress toward achievement based on activity data"""
        unlock_criteria = achievement.unlock_criteria
        requirements = unlock_criteria.get('requirements', [])
        
        if not requirements:
            return 0.0
        
        total_progress = 0.0
        
        for requirement in requirements:
            req_type = requirement.get('type')
            target = requirement.get('target', 1.0)
            current = activity_data.get(req_type, 0.0)
            
            # Calculate progress for this requirement
            req_progress = min(1.0, current / target) if target > 0 else 0.0
            total_progress += req_progress
        
        # Average progress across all requirements
        return total_progress / len(requirements)
    
    async def _check_milestones(self,
                              progress: AchievementProgress,
                              old_progress: float,
                              new_progress: float) -> List[Dict[str, Any]]:
        """Check for milestone achievements"""
        milestones_reached = []
        
        milestone_thresholds = [0.25, 0.5, 0.75, 0.9]
        
        for threshold in milestone_thresholds:
            if old_progress < threshold <= new_progress:
                milestone = {
                    'threshold': threshold,
                    'percentage': int(threshold * 100),
                    'reached_at': datetime.utcnow().isoformat(),
                    'message': f"{int(threshold * 100)}% progress toward achievement!"
                }
                
                milestones_reached.append(milestone)
                progress.progress_milestones.append(milestone)
        
        return milestones_reached
    
    async def _check_completion(self,
                              achievement: DynamicAchievement,
                              progress: AchievementProgress) -> Dict[str, Any]:
        """Check if achievement is completed"""
        if progress.is_completed:
            return {'already_completed': True}
        
        if progress.current_progress >= 1.0:
            # Mark as completed
            progress.is_completed = True
            progress.completed_at = datetime.utcnow().isoformat()
            
            return {
                'newly_completed': True,
                'completion_timestamp': progress.completed_at,
                'achievement_title': achievement.title,
                'points_earned': achievement.points_reward,
                'social_sharing_bonus': achievement.social_sharing_bonus
            }
        
        return {'completed': False, 'progress_remaining': 1.0 - progress.current_progress}
    
    async def _estimate_completion(self,
                                 progress: AchievementProgress,
                                 activity_data: Dict[str, Any]) -> str:
        """Estimate completion time based on current progress rate"""
        if progress.current_progress <= 0:
            return "Unable to estimate"
        
        # Calculate progress rate
        started_at = datetime.fromisoformat(progress.started_at)
        time_elapsed = (datetime.utcnow() - started_at).total_seconds() / 3600  # hours
        
        if time_elapsed <= 0:
            return "Unable to estimate"
        
        progress_rate = progress.current_progress / time_elapsed  # progress per hour
        
        if progress_rate <= 0:
            return "Unable to estimate"
        
        # Estimate remaining time
        remaining_progress = 1.0 - progress.current_progress
        estimated_hours = remaining_progress / progress_rate
        
        # Convert to human-readable format
        if estimated_hours < 1:
            return "Less than 1 hour"
        elif estimated_hours < 24:
            return f"About {int(estimated_hours)} hours"
        else:
            estimated_days = estimated_hours / 24
            return f"About {int(estimated_days)} days"


class BadgeDesignEngine:
    """
    🎨 BADGE DESIGN ENGINE
    
    Advanced engine for creating personalized badge designs.
    """
    
    def __init__(self):
        # Design configuration
        self.config = {
            'color_schemes': {
                AchievementRarity.COMMON: {'primary': '#CD7F32', 'secondary': '#D2691E', 'accent': '#228B22'},
                AchievementRarity.UNCOMMON: {'primary': '#C0C0C0', 'secondary': '#A9A9A9', 'accent': '#4169E1'},
                AchievementRarity.RARE: {'primary': '#FFD700', 'secondary': '#FFA500', 'accent': '#FF6347'},
                AchievementRarity.EPIC: {'primary': '#9932CC', 'secondary': '#8A2BE2', 'accent': '#FF1493'},
                AchievementRarity.LEGENDARY: {'primary': '#FF4500', 'secondary': '#FF6347', 'accent': '#FFD700'}
            },
            'design_elements': {
                'shapes': ['circular', 'hexagonal', 'shield', 'star', 'diamond'],
                'icons': ['trophy', 'star', 'crown', 'medal', 'ribbon', 'gem', 'flame', 'lightning'],
                'effects': ['glow', 'pulse', 'sparkle', 'gradient', 'shadow', 'border']
            }
        }
        
        logger.info("Badge Design Engine initialized")
    
    async def design_achievement_badge(self,
                                     achievement: DynamicAchievement,
                                     user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Design personalized badge for achievement
        
        Args:
            achievement: Achievement to design badge for
            user_preferences: User's design preferences
            
        Returns:
            Dict with badge design specification
        """
        try:
            # Determine rarity level
            rarity = self._determine_rarity(achievement.rarity_score)
            
            # Select design style
            design_style = self._select_design_style(user_preferences, achievement.category)
            
            # Create color scheme
            color_scheme = self._create_color_scheme(rarity, user_preferences)
            
            # Select visual elements
            visual_elements = self._select_visual_elements(achievement, design_style)
            
            # Create badge design
            badge_design = BadgeDesign(
                badge_id=f"badge_{achievement.achievement_id}",
                design_style=design_style,
                color_scheme=color_scheme,
                icon_elements=visual_elements['icons'],
                shape=visual_elements['shape'],
                size=visual_elements['size'],
                animation_effects=visual_elements['effects'],
                rarity_indicators=self._create_rarity_indicators(rarity),
                personalization_elements=self._create_personalization_elements(user_preferences)
            )
            
            return {
                'status': 'success',
                'badge_design': badge_design.__dict__,
                'design_rationale': {
                    'rarity_level': rarity.value,
                    'style_choice': design_style.value,
                    'personalization_applied': len(badge_design.personalization_elements)
                },
                'design_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error designing achievement badge: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _determine_rarity(self, rarity_score: float) -> AchievementRarity:
        """Determine rarity level from score"""
        if rarity_score >= 0.95:
            return AchievementRarity.LEGENDARY
        elif rarity_score >= 0.8:
            return AchievementRarity.EPIC
        elif rarity_score >= 0.6:
            return AchievementRarity.RARE
        elif rarity_score >= 0.3:
            return AchievementRarity.UNCOMMON
        else:
            return AchievementRarity.COMMON
    
    def _select_design_style(self,
                           user_preferences: Dict[str, Any],
                           achievement_category: AchievementType) -> BadgeDesignStyle:
        """Select appropriate design style"""
        # Check user preferences first
        preferred_style = user_preferences.get('badge_style')
        if preferred_style:
            try:
                return BadgeDesignStyle(preferred_style)
            except ValueError:
                pass
        
        # Default based on achievement category
        category_styles = {
            AchievementType.LEARNING_MILESTONE: BadgeDesignStyle.MINIMALIST,
            AchievementType.MASTERY_BADGE: BadgeDesignStyle.DETAILED,
            AchievementType.SOCIAL_RECOGNITION: BadgeDesignStyle.PLAYFUL,
            AchievementType.INNOVATION_AWARD: BadgeDesignStyle.ARTISTIC,
            AchievementType.BREAKTHROUGH_MEDAL: BadgeDesignStyle.TECHNICAL
        }
        
        return category_styles.get(achievement_category, BadgeDesignStyle.MINIMALIST)
    
    def _create_color_scheme(self,
                           rarity: AchievementRarity,
                           user_preferences: Dict[str, Any]) -> Dict[str, str]:
        """Create color scheme for badge"""
        base_scheme = self.config['color_schemes'][rarity].copy()
        
        # Apply user color preferences if available
        preferred_colors = user_preferences.get('preferred_colors', {})
        if preferred_colors:
            # Blend user preferences with rarity colors
            for key, color in preferred_colors.items():
                if key in base_scheme:
                    base_scheme[key] = color
        
        return base_scheme
    
    def _select_visual_elements(self,
                              achievement: DynamicAchievement,
                              design_style: BadgeDesignStyle) -> Dict[str, Any]:
        """Select visual elements for badge"""
        # Select shape based on achievement type
        type_shapes = {
            AchievementType.LEARNING_MILESTONE: 'circular',
            AchievementType.STREAK_ACHIEVEMENT: 'star',
            AchievementType.MASTERY_BADGE: 'shield',
            AchievementType.SOCIAL_RECOGNITION: 'hexagonal',
            AchievementType.INNOVATION_AWARD: 'diamond'
        }
        
        shape = type_shapes.get(achievement.category, 'circular')
        
        # Select icons
        type_icons = {
            AchievementType.LEARNING_MILESTONE: ['trophy', 'star'],
            AchievementType.STREAK_ACHIEVEMENT: ['flame', 'lightning'],
            AchievementType.MASTERY_BADGE: ['crown', 'gem'],
            AchievementType.SOCIAL_RECOGNITION: ['ribbon', 'star'],
            AchievementType.INNOVATION_AWARD: ['lightning', 'gem']
        }
        
        icons = type_icons.get(achievement.category, ['trophy'])
        
        # Select size based on difficulty
        if achievement.difficulty_tier >= 8:
            size = 'large'
        elif achievement.difficulty_tier >= 5:
            size = 'medium'
        else:
            size = 'small'
        
        # Select effects based on style and rarity
        effects = ['glow'] if achievement.rarity_score > 0.7 else []
        
        if design_style == BadgeDesignStyle.ARTISTIC:
            effects.extend(['gradient', 'sparkle'])
        elif design_style == BadgeDesignStyle.DETAILED:
            effects.extend(['shadow', 'border'])
        
        return {
            'shape': shape,
            'icons': icons,
            'size': size,
            'effects': effects
        }
    
    def _create_rarity_indicators(self, rarity: AchievementRarity) -> Dict[str, Any]:
        """Create rarity indicators for badge"""
        return {
            'rarity_level': rarity.value,
            'rarity_border': rarity != AchievementRarity.COMMON,
            'special_effects': rarity in [AchievementRarity.EPIC, AchievementRarity.LEGENDARY],
            'glow_intensity': {
                AchievementRarity.COMMON: 0.0,
                AchievementRarity.UNCOMMON: 0.2,
                AchievementRarity.RARE: 0.4,
                AchievementRarity.EPIC: 0.7,
                AchievementRarity.LEGENDARY: 1.0
            }[rarity]
        }
    
    def _create_personalization_elements(self, user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Create personalization elements for badge"""
        elements = {}
        
        # Add user's initials if available
        if user_preferences.get('show_initials'):
            elements['user_initials'] = user_preferences.get('initials', '')
        
        # Add favorite symbols
        if user_preferences.get('favorite_symbols'):
            elements['custom_symbols'] = user_preferences['favorite_symbols']
        
        # Add theme preferences
        if user_preferences.get('theme'):
            elements['theme'] = user_preferences['theme']
        
        return elements


class MasteryProgressTracker:
    """
    📈 MASTERY PROGRESS TRACKER
    
    Advanced tracker for monitoring mastery progress across skills and domains.
    """
    
    def __init__(self):
        # Tracking configuration
        self.config = {
            'mastery_threshold': 0.8,
            'expertise_threshold': 0.9,
            'mastery_decay_rate': 0.02,  # Monthly decay without practice
            'skill_categories': [
                'technical_skills', 'soft_skills', 'domain_knowledge',
                'creative_skills', 'analytical_skills', 'communication_skills'
            ]
        }
        
        # Tracking data
        self.mastery_profiles = {}  # user_id -> mastery profile
        self.mastery_history = []
        
        logger.info("Mastery Progress Tracker initialized")
    
    async def track_mastery_progress(self,
                                   user_id: str,
                                   skill_assessments: Dict[str, float],
                                   learning_activities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Track user's mastery progress across skills
        
        Args:
            user_id: User identifier
            skill_assessments: Current skill level assessments
            learning_activities: Recent learning activities
            
        Returns:
            Dict with mastery progress analysis
        """
        try:
            # Get or create mastery profile
            if user_id not in self.mastery_profiles:
                self.mastery_profiles[user_id] = {
                    'user_id': user_id,
                    'skill_masteries': {},
                    'mastery_achievements': [],
                    'expertise_areas': [],
                    'learning_trajectory': [],
                    'last_updated': datetime.utcnow().isoformat()
                }
            
            profile = self.mastery_profiles[user_id]
            
            # Update skill masteries
            mastery_updates = await self._update_skill_masteries(profile, skill_assessments)
            
            # Analyze mastery achievements
            new_masteries = await self._analyze_mastery_achievements(profile, mastery_updates)
            
            # Update expertise areas
            expertise_updates = await self._update_expertise_areas(profile, skill_assessments)
            
            # Track learning trajectory
            trajectory_update = await self._update_learning_trajectory(profile, learning_activities)
            
            # Generate mastery insights
            mastery_insights = await self._generate_mastery_insights(profile, mastery_updates)
            
            # Update profile
            profile['last_updated'] = datetime.utcnow().isoformat()
            
            # Track mastery history
            self.mastery_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'user_id': user_id,
                'new_masteries': len(new_masteries),
                'expertise_areas': len(profile['expertise_areas']),
                'overall_mastery_score': sum(skill_assessments.values()) / len(skill_assessments) if skill_assessments else 0.0
            })
            
            return {
                'status': 'success',
                'mastery_profile': profile,
                'mastery_updates': mastery_updates,
                'new_masteries': new_masteries,
                'expertise_updates': expertise_updates,
                'trajectory_update': trajectory_update,
                'mastery_insights': mastery_insights,
                'tracking_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error tracking mastery progress for user {user_id}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _update_skill_masteries(self,
                                    profile: Dict[str, Any],
                                    skill_assessments: Dict[str, float]) -> Dict[str, Any]:
        """Update skill mastery levels"""
        updates = {}
        
        for skill, current_level in skill_assessments.items():
            previous_level = profile['skill_masteries'].get(skill, 0.0)
            
            # Check for mastery achievement
            mastery_achieved = current_level >= self.config['mastery_threshold'] and previous_level < self.config['mastery_threshold']
            expertise_achieved = current_level >= self.config['expertise_threshold'] and previous_level < self.config['expertise_threshold']
            
            # Update mastery level
            profile['skill_masteries'][skill] = current_level
            
            updates[skill] = {
                'previous_level': previous_level,
                'current_level': current_level,
                'improvement': current_level - previous_level,
                'mastery_achieved': mastery_achieved,
                'expertise_achieved': expertise_achieved,
                'mastery_status': self._get_mastery_status(current_level)
            }
        
        return updates
    
    def _get_mastery_status(self, level: float) -> str:
        """Get mastery status for skill level"""
        if level >= self.config['expertise_threshold']:
            return 'expert'
        elif level >= self.config['mastery_threshold']:
            return 'mastery'
        elif level >= 0.6:
            return 'proficient'
        elif level >= 0.4:
            return 'intermediate'
        elif level >= 0.2:
            return 'beginner'
        else:
            return 'novice'
    
    async def _analyze_mastery_achievements(self,
                                          profile: Dict[str, Any],
                                          mastery_updates: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze new mastery achievements"""
        new_masteries = []
        
        for skill, update in mastery_updates.items():
            if update['mastery_achieved']:
                mastery_achievement = {
                    'skill': skill,
                    'achievement_type': 'mastery',
                    'level_achieved': update['current_level'],
                    'achieved_at': datetime.utcnow().isoformat(),
                    'improvement_from': update['previous_level']
                }
                
                new_masteries.append(mastery_achievement)
                profile['mastery_achievements'].append(mastery_achievement)
            
            elif update['expertise_achieved']:
                expertise_achievement = {
                    'skill': skill,
                    'achievement_type': 'expertise',
                    'level_achieved': update['current_level'],
                    'achieved_at': datetime.utcnow().isoformat(),
                    'improvement_from': update['previous_level']
                }
                
                new_masteries.append(expertise_achievement)
                profile['mastery_achievements'].append(expertise_achievement)
        
        return new_masteries
    
    async def _update_expertise_areas(self,
                                    profile: Dict[str, Any],
                                    skill_assessments: Dict[str, float]) -> Dict[str, Any]:
        """Update user's expertise areas"""
        current_expertise = set(profile.get('expertise_areas', []))
        new_expertise = set()
        
        for skill, level in skill_assessments.items():
            if level >= self.config['expertise_threshold']:
                new_expertise.add(skill)
        
        # Find newly achieved expertise
        newly_achieved = new_expertise - current_expertise
        
        # Update profile
        profile['expertise_areas'] = list(new_expertise)
        
        return {
            'current_expertise_areas': list(new_expertise),
            'newly_achieved_expertise': list(newly_achieved),
            'total_expertise_count': len(new_expertise)
        }
    
    async def _update_learning_trajectory(self,
                                        profile: Dict[str, Any],
                                        learning_activities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Update learning trajectory"""
        # Calculate learning velocity
        if learning_activities:
            recent_activities = learning_activities[-10:]  # Last 10 activities
            activity_count = len(recent_activities)
            
            # Calculate time span
            if len(recent_activities) > 1:
                start_time = datetime.fromisoformat(recent_activities[0].get('timestamp', datetime.utcnow().isoformat()))
                end_time = datetime.fromisoformat(recent_activities[-1].get('timestamp', datetime.utcnow().isoformat()))
                time_span_hours = (end_time - start_time).total_seconds() / 3600
                
                learning_velocity = activity_count / max(1, time_span_hours)
            else:
                learning_velocity = 0.0
        else:
            learning_velocity = 0.0
        
        # Update trajectory
        trajectory_point = {
            'timestamp': datetime.utcnow().isoformat(),
            'learning_velocity': learning_velocity,
            'activity_count': len(learning_activities),
            'mastery_count': len([skill for skill, level in profile['skill_masteries'].items() if level >= self.config['mastery_threshold']])
        }
        
        profile['learning_trajectory'].append(trajectory_point)
        
        # Keep only recent trajectory points
        if len(profile['learning_trajectory']) > 100:
            profile['learning_trajectory'] = profile['learning_trajectory'][-100:]
        
        return trajectory_point
    
    async def _generate_mastery_insights(self,
                                       profile: Dict[str, Any],
                                       mastery_updates: Dict[str, Any]) -> List[str]:
        """Generate insights about mastery progress"""
        insights = []
        
        # Overall mastery insights
        mastery_count = len([skill for skill, level in profile['skill_masteries'].items() if level >= self.config['mastery_threshold']])
        expertise_count = len(profile.get('expertise_areas', []))
        
        if mastery_count > 5:
            insights.append(f"Impressive! You've achieved mastery in {mastery_count} skills")
        elif mastery_count > 0:
            insights.append(f"Great progress! {mastery_count} skills mastered")
        
        if expertise_count > 0:
            insights.append(f"Expert level achieved in {expertise_count} areas")
        
        # Recent improvement insights
        significant_improvements = [
            skill for skill, update in mastery_updates.items()
            if update['improvement'] > 0.1
        ]
        
        if significant_improvements:
            insights.append(f"Significant improvement in: {', '.join(significant_improvements[:3])}")
        
        # Learning trajectory insights
        trajectory = profile.get('learning_trajectory', [])
        if len(trajectory) >= 2:
            recent_velocity = trajectory[-1]['learning_velocity']
            previous_velocity = trajectory[-2]['learning_velocity']
            
            if recent_velocity > previous_velocity * 1.2:
                insights.append("Learning velocity is accelerating!")
            elif recent_velocity < previous_velocity * 0.8:
                insights.append("Consider increasing learning activity frequency")
        
        return insights
