"""
Social Competition Services

Extracted from quantum_intelligence_engine.py (lines 12526-15023) - social competition
engines, leaderboard systems, team challenge management, and competitive analytics
for engaging social learning experiences.
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


class CompetitionType(Enum):
    """Types of social competitions"""
    INDIVIDUAL_LEADERBOARD = "individual_leaderboard"
    TEAM_CHALLENGE = "team_challenge"
    PEER_TOURNAMENT = "peer_tournament"
    COLLABORATIVE_GOAL = "collaborative_goal"
    SKILL_SHOWCASE = "skill_showcase"
    SPEED_COMPETITION = "speed_competition"
    CREATIVITY_CONTEST = "creativity_contest"
    KNOWLEDGE_BATTLE = "knowledge_battle"


class LeaderboardType(Enum):
    """Types of leaderboards"""
    GLOBAL_RANKING = "global_ranking"
    PEER_GROUP_RANKING = "peer_group_ranking"
    SKILL_BASED_RANKING = "skill_based_ranking"
    PROGRESS_RANKING = "progress_ranking"
    ACHIEVEMENT_RANKING = "achievement_ranking"
    CONTRIBUTION_RANKING = "contribution_ranking"
    STREAK_RANKING = "streak_ranking"
    IMPROVEMENT_RANKING = "improvement_ranking"


class TeamFormationStrategy(Enum):
    """Strategies for forming teams"""
    RANDOM_ASSIGNMENT = "random_assignment"
    SKILL_BALANCED = "skill_balanced"
    MIXED_ABILITY = "mixed_ability"
    FRIEND_GROUPS = "friend_groups"
    COMPLEMENTARY_SKILLS = "complementary_skills"
    SIMILAR_GOALS = "similar_goals"
    GEOGRAPHIC_PROXIMITY = "geographic_proximity"
    LEARNING_STYLE_MATCH = "learning_style_match"


@dataclass
class SocialCompetition:
    """Social competition configuration"""
    competition_id: str = ""
    title: str = ""
    description: str = ""
    competition_type: CompetitionType = CompetitionType.INDIVIDUAL_LEADERBOARD
    participants: List[str] = field(default_factory=list)
    teams: List[Dict[str, Any]] = field(default_factory=list)
    competition_rules: Dict[str, Any] = field(default_factory=dict)
    scoring_system: Dict[str, Any] = field(default_factory=dict)
    duration_days: int = 7
    start_date: str = ""
    end_date: str = ""
    prizes_rewards: Dict[str, Any] = field(default_factory=dict)
    leaderboard_config: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    created_at: str = ""


@dataclass
class Leaderboard:
    """Leaderboard configuration and data"""
    leaderboard_id: str = ""
    title: str = ""
    leaderboard_type: LeaderboardType = LeaderboardType.GLOBAL_RANKING
    ranking_criteria: Dict[str, Any] = field(default_factory=dict)
    participants: List[Dict[str, Any]] = field(default_factory=list)
    update_frequency: str = "real_time"
    display_settings: Dict[str, Any] = field(default_factory=dict)
    privacy_settings: Dict[str, Any] = field(default_factory=dict)
    time_period: str = "all_time"
    last_updated: str = ""
    created_at: str = ""


@dataclass
class TeamChallenge:
    """Team challenge configuration"""
    challenge_id: str = ""
    title: str = ""
    description: str = ""
    team_size: int = 4
    max_teams: int = 10
    challenge_objectives: List[str] = field(default_factory=list)
    team_formation_strategy: TeamFormationStrategy = TeamFormationStrategy.SKILL_BALANCED
    collaboration_requirements: Dict[str, Any] = field(default_factory=dict)
    team_scoring: Dict[str, Any] = field(default_factory=dict)
    individual_scoring: Dict[str, Any] = field(default_factory=dict)
    duration_days: int = 14
    start_date: str = ""
    end_date: str = ""
    formed_teams: List[Dict[str, Any]] = field(default_factory=list)
    is_active: bool = True
    created_at: str = ""


class SocialCompetitionEngine:
    """
    🏆 SOCIAL COMPETITION ENGINE
    
    Advanced engine for creating and managing social competitions.
    Extracted from the original quantum engine's gamification logic.
    """
    
    def __init__(self, cache_service: Optional[CacheService] = None):
        self.cache = cache_service
        
        # Engine configuration
        self.config = {
            'competition_templates': {
                CompetitionType.INDIVIDUAL_LEADERBOARD: {
                    'min_participants': 5,
                    'max_participants': 100,
                    'default_duration': 7
                },
                CompetitionType.TEAM_CHALLENGE: {
                    'min_participants': 8,
                    'max_participants': 40,
                    'default_duration': 14
                },
                CompetitionType.PEER_TOURNAMENT: {
                    'min_participants': 8,
                    'max_participants': 32,
                    'default_duration': 10
                }
            },
            'scoring_weights': {
                'completion_rate': 0.3,
                'accuracy_score': 0.25,
                'speed_bonus': 0.15,
                'collaboration_score': 0.2,
                'innovation_score': 0.1
            },
            'fair_play_monitoring': True,
            'adaptive_balancing': True
        }
        
        # Engine tracking
        self.active_competitions = {}
        self.competition_history = []
        self.participant_analytics = {}
        
        logger.info("Social Competition Engine initialized")
    
    async def create_social_competition(self,
                                      competition_config: Dict[str, Any],
                                      participant_pool: List[Dict[str, Any]],
                                      learning_objectives: List[str]) -> Dict[str, Any]:
        """
        Create social competition with optimal configuration
        
        Args:
            competition_config: Competition configuration and preferences
            participant_pool: Available participants for the competition
            learning_objectives: Learning objectives for the competition
            
        Returns:
            Dict with created social competition
        """
        try:
            # Determine competition type
            competition_type = await self._determine_competition_type(competition_config, participant_pool)
            
            # Validate participant pool
            validation_result = await self._validate_participant_pool(competition_type, participant_pool)
            if validation_result['status'] != 'success':
                return validation_result
            
            # Create competition structure
            competition_structure = await self._create_competition_structure(
                competition_type, participant_pool, learning_objectives, competition_config
            )
            
            # Design scoring system
            scoring_system = await self._design_scoring_system(competition_type, learning_objectives)
            
            # Create leaderboard configuration
            leaderboard_config = await self._create_leaderboard_configuration(competition_type, competition_config)
            
            # Set up prizes and rewards
            prizes_rewards = await self._setup_prizes_rewards(competition_type, len(participant_pool))
            
            # Generate competition content
            competition_content = await self._generate_competition_content(
                competition_type, learning_objectives, competition_config
            )
            
            # Create social competition
            competition_id = f"comp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            start_date = datetime.utcnow().isoformat()
            duration_days = competition_config.get('duration_days', self.config['competition_templates'][competition_type]['default_duration'])
            end_date = (datetime.utcnow() + timedelta(days=duration_days)).isoformat()
            
            social_competition = SocialCompetition(
                competition_id=competition_id,
                title=competition_content['title'],
                description=competition_content['description'],
                competition_type=competition_type,
                participants=[p['user_id'] for p in participant_pool],
                teams=competition_structure.get('teams', []),
                competition_rules=competition_structure['rules'],
                scoring_system=scoring_system,
                duration_days=duration_days,
                start_date=start_date,
                end_date=end_date,
                prizes_rewards=prizes_rewards,
                leaderboard_config=leaderboard_config,
                created_at=datetime.utcnow().isoformat()
            )
            
            # Store competition
            self.active_competitions[competition_id] = social_competition
            
            # Initialize participant tracking
            await self._initialize_participant_tracking(competition_id, participant_pool)
            
            return {
                'status': 'success',
                'social_competition': social_competition.__dict__,
                'competition_preview': {
                    'participant_count': len(participant_pool),
                    'team_count': len(competition_structure.get('teams', [])),
                    'duration': f"{duration_days} days",
                    'competition_type': competition_type.value
                },
                'creation_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating social competition: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _determine_competition_type(self,
                                        competition_config: Dict[str, Any],
                                        participant_pool: List[Dict[str, Any]]) -> CompetitionType:
        """Determine optimal competition type"""
        # Check explicit preference
        preferred_type = competition_config.get('competition_type')
        if preferred_type:
            try:
                return CompetitionType(preferred_type)
            except ValueError:
                pass
        
        # Determine based on participant characteristics
        participant_count = len(participant_pool)
        
        # Analyze participant preferences
        social_preferences = [p.get('social_preference', 0.5) for p in participant_pool]
        avg_social_preference = sum(social_preferences) / len(social_preferences)
        
        collaboration_preferences = [p.get('collaboration_preference', 0.5) for p in participant_pool]
        avg_collaboration_preference = sum(collaboration_preferences) / len(collaboration_preferences)
        
        # Score different competition types
        type_scores = {
            CompetitionType.INDIVIDUAL_LEADERBOARD: 0.5,
            CompetitionType.TEAM_CHALLENGE: 0.3,
            CompetitionType.PEER_TOURNAMENT: 0.2,
            CompetitionType.COLLABORATIVE_GOAL: 0.2,
            CompetitionType.SKILL_SHOWCASE: 0.3,
            CompetitionType.SPEED_COMPETITION: 0.2,
            CompetitionType.CREATIVITY_CONTEST: 0.2,
            CompetitionType.KNOWLEDGE_BATTLE: 0.3
        }
        
        # Adjust based on participant count
        if participant_count >= 20:
            type_scores[CompetitionType.INDIVIDUAL_LEADERBOARD] += 0.3
        elif participant_count >= 8:
            type_scores[CompetitionType.TEAM_CHALLENGE] += 0.4
            type_scores[CompetitionType.PEER_TOURNAMENT] += 0.3
        
        # Adjust based on social preferences
        if avg_social_preference > 0.7:
            type_scores[CompetitionType.TEAM_CHALLENGE] += 0.3
            type_scores[CompetitionType.COLLABORATIVE_GOAL] += 0.3
        
        if avg_collaboration_preference > 0.7:
            type_scores[CompetitionType.TEAM_CHALLENGE] += 0.4
            type_scores[CompetitionType.COLLABORATIVE_GOAL] += 0.4
        
        # Select highest scoring type
        best_type = max(type_scores.items(), key=lambda x: x[1])
        return best_type[0]
    
    async def _validate_participant_pool(self,
                                       competition_type: CompetitionType,
                                       participant_pool: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate participant pool for competition type"""
        template = self.config['competition_templates'][competition_type]
        participant_count = len(participant_pool)
        
        if participant_count < template['min_participants']:
            return {
                'status': 'error',
                'error': f'Insufficient participants. Need at least {template["min_participants"]}, got {participant_count}'
            }
        
        if participant_count > template['max_participants']:
            return {
                'status': 'error',
                'error': f'Too many participants. Maximum {template["max_participants"]}, got {participant_count}'
            }
        
        # Check for required participant data
        required_fields = ['user_id', 'skill_level']
        for participant in participant_pool:
            for field in required_fields:
                if field not in participant:
                    return {
                        'status': 'error',
                        'error': f'Participant missing required field: {field}'
                    }
        
        return {'status': 'success'}
    
    async def _create_competition_structure(self,
                                          competition_type: CompetitionType,
                                          participant_pool: List[Dict[str, Any]],
                                          learning_objectives: List[str],
                                          competition_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create competition structure"""
        structure = {
            'rules': await self._create_competition_rules(competition_type, learning_objectives),
            'phases': await self._create_competition_phases(competition_type, competition_config),
            'teams': []
        }
        
        # Create teams if needed
        if competition_type in [CompetitionType.TEAM_CHALLENGE, CompetitionType.COLLABORATIVE_GOAL]:
            team_size = competition_config.get('team_size', 4)
            formation_strategy = competition_config.get('team_formation_strategy', 'skill_balanced')
            
            teams = await self._form_teams(participant_pool, team_size, formation_strategy)
            structure['teams'] = teams
        
        return structure
    
    async def _create_competition_rules(self,
                                      competition_type: CompetitionType,
                                      learning_objectives: List[str]) -> Dict[str, Any]:
        """Create competition rules"""
        base_rules = {
            'fair_play_required': True,
            'collaboration_allowed': competition_type in [CompetitionType.TEAM_CHALLENGE, CompetitionType.COLLABORATIVE_GOAL],
            'time_limits': {},
            'scoring_transparency': True,
            'appeal_process': True
        }
        
        # Type-specific rules
        if competition_type == CompetitionType.SPEED_COMPETITION:
            base_rules['time_limits'] = {
                'per_task': 30,  # minutes
                'total_session': 120  # minutes
            }
            base_rules['speed_bonus_enabled'] = True
        
        elif competition_type == CompetitionType.CREATIVITY_CONTEST:
            base_rules['originality_required'] = True
            base_rules['peer_voting_enabled'] = True
            base_rules['submission_guidelines'] = {
                'max_submissions': 3,
                'revision_allowed': True
            }
        
        elif competition_type == CompetitionType.TEAM_CHALLENGE:
            base_rules['team_communication_required'] = True
            base_rules['individual_contribution_tracking'] = True
            base_rules['team_coordination_bonus'] = True
        
        # Objective-specific rules
        for objective in learning_objectives:
            if 'accuracy' in objective.lower():
                base_rules['accuracy_threshold'] = 0.8
            elif 'speed' in objective.lower():
                base_rules['time_bonus_multiplier'] = 1.5
        
        return base_rules
    
    async def _create_competition_phases(self,
                                       competition_type: CompetitionType,
                                       competition_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create competition phases"""
        duration_days = competition_config.get('duration_days', 7)
        
        if competition_type == CompetitionType.PEER_TOURNAMENT:
            # Tournament with elimination phases
            return [
                {
                    'phase': 'qualification',
                    'duration_days': duration_days // 3,
                    'description': 'Qualification round for tournament seeding'
                },
                {
                    'phase': 'elimination',
                    'duration_days': duration_days // 2,
                    'description': 'Elimination rounds'
                },
                {
                    'phase': 'finals',
                    'duration_days': duration_days // 6,
                    'description': 'Final championship round'
                }
            ]
        
        elif duration_days > 7:
            # Multi-phase competition
            return [
                {
                    'phase': 'warm_up',
                    'duration_days': 2,
                    'description': 'Warm-up and team formation'
                },
                {
                    'phase': 'main_competition',
                    'duration_days': duration_days - 4,
                    'description': 'Main competition period'
                },
                {
                    'phase': 'final_sprint',
                    'duration_days': 2,
                    'description': 'Final sprint and submissions'
                }
            ]
        
        else:
            # Single phase competition
            return [
                {
                    'phase': 'main_competition',
                    'duration_days': duration_days,
                    'description': 'Main competition period'
                }
            ]
    
    async def _form_teams(self,
                        participant_pool: List[Dict[str, Any]],
                        team_size: int,
                        formation_strategy: str) -> List[Dict[str, Any]]:
        """Form teams based on strategy"""
        teams = []
        participants = participant_pool.copy()
        team_count = len(participants) // team_size
        
        if formation_strategy == 'skill_balanced':
            # Sort by skill level and distribute evenly
            participants.sort(key=lambda p: p.get('skill_level', 0.5))
            
            for i in range(team_count):
                team = {
                    'team_id': f"team_{i+1:02d}",
                    'team_name': f"Team {i+1}",
                    'members': [],
                    'formation_strategy': formation_strategy
                }
                
                # Distribute participants to balance skill levels
                for j in range(team_size):
                    if participants:
                        # Take from different skill levels
                        index = (j * len(participants) // team_size) % len(participants)
                        team['members'].append(participants.pop(index))
                
                teams.append(team)
        
        elif formation_strategy == 'random_assignment':
            # Random team assignment
            random.shuffle(participants)
            
            for i in range(team_count):
                team = {
                    'team_id': f"team_{i+1:02d}",
                    'team_name': f"Team {i+1}",
                    'members': participants[i*team_size:(i+1)*team_size],
                    'formation_strategy': formation_strategy
                }
                teams.append(team)
        
        elif formation_strategy == 'complementary_skills':
            # Form teams with complementary skills
            skill_categories = ['technical', 'creative', 'analytical', 'communication']
            
            for i in range(team_count):
                team = {
                    'team_id': f"team_{i+1:02d}",
                    'team_name': f"Team {i+1}",
                    'members': [],
                    'formation_strategy': formation_strategy
                }
                
                # Try to get one person from each skill category
                for category in skill_categories:
                    suitable_participants = [
                        p for p in participants 
                        if p.get('primary_skill_category') == category
                    ]
                    
                    if suitable_participants and len(team['members']) < team_size:
                        selected = random.choice(suitable_participants)
                        team['members'].append(selected)
                        participants.remove(selected)
                
                # Fill remaining spots randomly
                while len(team['members']) < team_size and participants:
                    team['members'].append(participants.pop(0))
                
                teams.append(team)
        
        return teams
    
    async def _design_scoring_system(self,
                                   competition_type: CompetitionType,
                                   learning_objectives: List[str]) -> Dict[str, Any]:
        """Design scoring system for competition"""
        base_scoring = {
            'point_system': 'weighted_sum',
            'components': {},
            'bonuses': {},
            'penalties': {}
        }
        
        # Add scoring components based on competition type
        if competition_type == CompetitionType.SPEED_COMPETITION:
            base_scoring['components'] = {
                'completion_speed': 0.4,
                'accuracy': 0.4,
                'task_completion': 0.2
            }
            base_scoring['bonuses']['speed_bonus'] = {
                'threshold': 'top_25_percent',
                'multiplier': 1.3
            }
        
        elif competition_type == CompetitionType.CREATIVITY_CONTEST:
            base_scoring['components'] = {
                'originality': 0.3,
                'quality': 0.3,
                'peer_votes': 0.2,
                'expert_evaluation': 0.2
            }
            base_scoring['bonuses']['innovation_bonus'] = {
                'threshold': 'highly_original',
                'points': 500
            }
        
        elif competition_type == CompetitionType.TEAM_CHALLENGE:
            base_scoring['components'] = {
                'individual_contribution': 0.4,
                'team_performance': 0.4,
                'collaboration_quality': 0.2
            }
            base_scoring['bonuses']['team_synergy'] = {
                'threshold': 'high_collaboration',
                'multiplier': 1.2
            }
        
        else:
            # Default scoring for individual competitions
            base_scoring['components'] = self.config['scoring_weights'].copy()
        
        # Add objective-specific scoring
        for objective in learning_objectives:
            if 'accuracy' in objective.lower():
                if 'accuracy_score' in base_scoring['components']:
                    base_scoring['components']['accuracy_score'] *= 1.2
            elif 'collaboration' in objective.lower():
                if 'collaboration_score' in base_scoring['components']:
                    base_scoring['components']['collaboration_score'] *= 1.3
        
        # Normalize component weights
        total_weight = sum(base_scoring['components'].values())
        if total_weight > 0:
            for component in base_scoring['components']:
                base_scoring['components'][component] /= total_weight
        
        return base_scoring
    
    async def _create_leaderboard_configuration(self,
                                              competition_type: CompetitionType,
                                              competition_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create leaderboard configuration"""
        leaderboard_config = {
            'update_frequency': 'real_time',
            'display_top_n': 10,
            'show_progress': True,
            'show_rankings': True,
            'privacy_level': 'public',
            'historical_tracking': True
        }
        
        # Adjust based on competition type
        if competition_type == CompetitionType.TEAM_CHALLENGE:
            leaderboard_config['team_leaderboard'] = True
            leaderboard_config['individual_leaderboard'] = True
            leaderboard_config['display_team_members'] = True
        
        elif competition_type == CompetitionType.PEER_TOURNAMENT:
            leaderboard_config['bracket_display'] = True
            leaderboard_config['elimination_tracking'] = True
        
        # Apply user preferences
        if competition_config.get('privacy_preference') == 'private':
            leaderboard_config['privacy_level'] = 'participants_only'
        
        return leaderboard_config
    
    async def _setup_prizes_rewards(self,
                                  competition_type: CompetitionType,
                                  participant_count: int) -> Dict[str, Any]:
        """Set up prizes and rewards"""
        base_points = 1000
        
        prizes = {
            'point_rewards': {
                '1st_place': base_points,
                '2nd_place': int(base_points * 0.7),
                '3rd_place': int(base_points * 0.5)
            },
            'badges': {
                '1st_place': 'champion_badge',
                '2nd_place': 'runner_up_badge',
                '3rd_place': 'bronze_medal_badge'
            },
            'special_recognition': {}
        }
        
        # Add participation rewards
        if participant_count > 10:
            prizes['participation_reward'] = {
                'points': 100,
                'badge': 'participant_badge'
            }
        
        # Add type-specific rewards
        if competition_type == CompetitionType.CREATIVITY_CONTEST:
            prizes['special_recognition']['most_creative'] = {
                'points': 300,
                'badge': 'creativity_master'
            }
        
        elif competition_type == CompetitionType.TEAM_CHALLENGE:
            prizes['team_rewards'] = {
                'best_collaboration': {
                    'points_per_member': 200,
                    'team_badge': 'collaboration_champions'
                }
            }
        
        elif competition_type == CompetitionType.SPEED_COMPETITION:
            prizes['special_recognition']['speed_demon'] = {
                'points': 250,
                'badge': 'speed_master'
            }
        
        return prizes
    
    async def _generate_competition_content(self,
                                          competition_type: CompetitionType,
                                          learning_objectives: List[str],
                                          competition_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate competition content"""
        objectives_text = ', '.join(learning_objectives[:3]) if learning_objectives else "learning skills"
        
        # Type-specific content templates
        content_templates = {
            CompetitionType.INDIVIDUAL_LEADERBOARD: {
                'title': f"Master {objectives_text} - Individual Challenge",
                'description': f"Compete individually to master {objectives_text}. Climb the leaderboard and prove your expertise!"
            },
            CompetitionType.TEAM_CHALLENGE: {
                'title': f"Team Quest: {objectives_text}",
                'description': f"Join forces with your team to conquer {objectives_text}. Collaboration and strategy are key to victory!"
            },
            CompetitionType.SPEED_COMPETITION: {
                'title': f"Speed Master: {objectives_text}",
                'description': f"Race against time to master {objectives_text}. Speed and accuracy will determine the champion!"
            },
            CompetitionType.CREATIVITY_CONTEST: {
                'title': f"Creative Excellence in {objectives_text}",
                'description': f"Showcase your creativity while mastering {objectives_text}. Innovation and originality will be rewarded!"
            },
            CompetitionType.PEER_TOURNAMENT: {
                'title': f"Tournament of Champions: {objectives_text}",
                'description': f"Battle through elimination rounds to become the ultimate {objectives_text} champion!"
            }
        }
        
        template = content_templates.get(competition_type, content_templates[CompetitionType.INDIVIDUAL_LEADERBOARD])
        
        return {
            'title': template['title'],
            'description': template['description'],
            'theme': competition_config.get('theme', 'academic_excellence'),
            'motivational_message': f"Push your limits and achieve greatness in {objectives_text}!"
        }
    
    async def _initialize_participant_tracking(self,
                                             competition_id: str,
                                             participant_pool: List[Dict[str, Any]]):
        """Initialize tracking for competition participants"""
        for participant in participant_pool:
            user_id = participant['user_id']
            
            if user_id not in self.participant_analytics:
                self.participant_analytics[user_id] = {}
            
            self.participant_analytics[user_id][competition_id] = {
                'joined_at': datetime.utcnow().isoformat(),
                'initial_skill_level': participant.get('skill_level', 0.5),
                'participation_score': 0.0,
                'achievements_earned': [],
                'team_id': None,  # Will be set if in team competition
                'performance_history': []
            }


class LeaderboardSystem:
    """
    📊 LEADERBOARD SYSTEM
    
    Advanced system for managing dynamic leaderboards.
    """
    
    def __init__(self):
        # System configuration
        self.config = {
            'update_intervals': {
                'real_time': 0,  # Immediate
                'frequent': 300,  # 5 minutes
                'hourly': 3600,
                'daily': 86400
            },
            'ranking_algorithms': {
                'simple_score': 'sum_of_points',
                'weighted_score': 'weighted_components',
                'elo_rating': 'elo_system',
                'percentile_rank': 'percentile_based'
            }
        }
        
        # System tracking
        self.active_leaderboards = {}
        self.leaderboard_history = {}
        
        logger.info("Leaderboard System initialized")
    
    async def create_leaderboard(self,
                               leaderboard_config: Dict[str, Any],
                               participants: List[Dict[str, Any]],
                               ranking_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create dynamic leaderboard
        
        Args:
            leaderboard_config: Leaderboard configuration
            participants: List of participants
            ranking_criteria: Criteria for ranking participants
            
        Returns:
            Dict with created leaderboard
        """
        try:
            # Determine leaderboard type
            leaderboard_type = LeaderboardType(leaderboard_config.get('type', 'global_ranking'))
            
            # Create leaderboard
            leaderboard_id = f"lb_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            leaderboard = Leaderboard(
                leaderboard_id=leaderboard_id,
                title=leaderboard_config.get('title', 'Competition Leaderboard'),
                leaderboard_type=leaderboard_type,
                ranking_criteria=ranking_criteria,
                participants=await self._initialize_participant_rankings(participants),
                update_frequency=leaderboard_config.get('update_frequency', 'real_time'),
                display_settings=leaderboard_config.get('display_settings', {}),
                privacy_settings=leaderboard_config.get('privacy_settings', {}),
                time_period=leaderboard_config.get('time_period', 'all_time'),
                last_updated=datetime.utcnow().isoformat(),
                created_at=datetime.utcnow().isoformat()
            )
            
            # Store leaderboard
            self.active_leaderboards[leaderboard_id] = leaderboard
            
            return {
                'status': 'success',
                'leaderboard': leaderboard.__dict__,
                'creation_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating leaderboard: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def update_leaderboard(self,
                               leaderboard_id: str,
                               participant_scores: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Update leaderboard with new scores
        
        Args:
            leaderboard_id: Leaderboard identifier
            participant_scores: Updated scores for participants
            
        Returns:
            Dict with updated leaderboard
        """
        try:
            if leaderboard_id not in self.active_leaderboards:
                return {'status': 'error', 'error': 'Leaderboard not found'}
            
            leaderboard = self.active_leaderboards[leaderboard_id]
            
            # Update participant scores
            for participant in leaderboard.participants:
                user_id = participant['user_id']
                if user_id in participant_scores:
                    participant.update(participant_scores[user_id])
            
            # Recalculate rankings
            updated_rankings = await self._calculate_rankings(leaderboard)
            leaderboard.participants = updated_rankings
            leaderboard.last_updated = datetime.utcnow().isoformat()
            
            # Track ranking changes
            ranking_changes = await self._track_ranking_changes(leaderboard_id, updated_rankings)
            
            return {
                'status': 'success',
                'updated_leaderboard': leaderboard.__dict__,
                'ranking_changes': ranking_changes,
                'update_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating leaderboard: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _initialize_participant_rankings(self, participants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Initialize participant rankings"""
        rankings = []
        
        for i, participant in enumerate(participants):
            ranking_entry = {
                'user_id': participant['user_id'],
                'display_name': participant.get('display_name', f"Participant {i+1}"),
                'current_rank': i + 1,
                'previous_rank': i + 1,
                'total_score': 0.0,
                'score_components': {},
                'achievements': [],
                'last_activity': datetime.utcnow().isoformat()
            }
            rankings.append(ranking_entry)
        
        return rankings
    
    async def _calculate_rankings(self, leaderboard: Leaderboard) -> List[Dict[str, Any]]:
        """Calculate new rankings based on scores"""
        participants = leaderboard.participants.copy()
        
        # Calculate total scores based on ranking criteria
        for participant in participants:
            total_score = 0.0
            
            for component, weight in leaderboard.ranking_criteria.items():
                component_score = participant.get('score_components', {}).get(component, 0.0)
                total_score += component_score * weight
            
            participant['total_score'] = total_score
        
        # Sort by total score (descending)
        participants.sort(key=lambda p: p['total_score'], reverse=True)
        
        # Update ranks
        for i, participant in enumerate(participants):
            participant['previous_rank'] = participant['current_rank']
            participant['current_rank'] = i + 1
        
        return participants
    
    async def _track_ranking_changes(self,
                                   leaderboard_id: str,
                                   updated_rankings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Track changes in rankings"""
        changes = []
        
        for participant in updated_rankings:
            current_rank = participant['current_rank']
            previous_rank = participant['previous_rank']
            
            if current_rank != previous_rank:
                change = {
                    'user_id': participant['user_id'],
                    'previous_rank': previous_rank,
                    'current_rank': current_rank,
                    'rank_change': previous_rank - current_rank,  # Positive = moved up
                    'timestamp': datetime.utcnow().isoformat()
                }
                changes.append(change)
        
        # Store in history
        if leaderboard_id not in self.leaderboard_history:
            self.leaderboard_history[leaderboard_id] = []
        
        self.leaderboard_history[leaderboard_id].extend(changes)
        
        return changes


class CompetitiveAnalytics:
    """
    📈 COMPETITIVE ANALYTICS
    
    Advanced analytics for competition performance and insights.
    """
    
    def __init__(self):
        # Analytics configuration
        self.config = {
            'metrics_tracked': [
                'participation_rate', 'engagement_level', 'performance_improvement',
                'collaboration_quality', 'competition_satisfaction', 'retention_rate'
            ],
            'analysis_windows': ['daily', 'weekly', 'competition_duration', 'all_time'],
            'benchmark_calculations': True
        }
        
        # Analytics tracking
        self.competition_metrics = {}
        self.participant_insights = {}
        
        logger.info("Competitive Analytics initialized")
    
    async def analyze_competition_performance(self,
                                            competition_id: str,
                                            competition_data: Dict[str, Any],
                                            participant_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze competition performance and generate insights
        
        Args:
            competition_id: Competition identifier
            competition_data: Competition configuration and results
            participant_data: Participant performance data
            
        Returns:
            Dict with comprehensive competition analytics
        """
        try:
            # Calculate participation metrics
            participation_metrics = await self._calculate_participation_metrics(participant_data)
            
            # Analyze engagement patterns
            engagement_analysis = await self._analyze_engagement_patterns(participant_data)
            
            # Calculate performance improvements
            performance_analysis = await self._analyze_performance_improvements(participant_data)
            
            # Analyze competition effectiveness
            effectiveness_analysis = await self._analyze_competition_effectiveness(
                competition_data, participation_metrics, engagement_analysis
            )
            
            # Generate insights and recommendations
            insights_recommendations = await self._generate_insights_recommendations(
                participation_metrics, engagement_analysis, performance_analysis, effectiveness_analysis
            )
            
            # Store analytics
            self.competition_metrics[competition_id] = {
                'participation_metrics': participation_metrics,
                'engagement_analysis': engagement_analysis,
                'performance_analysis': performance_analysis,
                'effectiveness_analysis': effectiveness_analysis,
                'insights_recommendations': insights_recommendations,
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
            return {
                'status': 'success',
                'competition_analytics': self.competition_metrics[competition_id],
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing competition performance: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _calculate_participation_metrics(self, participant_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate participation metrics"""
        total_participants = len(participant_data)
        
        if total_participants == 0:
            return {'total_participants': 0, 'participation_rate': 0.0}
        
        # Calculate active participation
        active_participants = len([p for p in participant_data if p.get('total_score', 0) > 0])
        participation_rate = active_participants / total_participants
        
        # Calculate engagement levels
        high_engagement = len([p for p in participant_data if p.get('engagement_score', 0) > 0.8])
        medium_engagement = len([p for p in participant_data if 0.5 <= p.get('engagement_score', 0) <= 0.8])
        low_engagement = len([p for p in participant_data if p.get('engagement_score', 0) < 0.5])
        
        # Calculate completion rates
        completed_participants = len([p for p in participant_data if p.get('completion_status') == 'completed'])
        completion_rate = completed_participants / total_participants
        
        return {
            'total_participants': total_participants,
            'active_participants': active_participants,
            'participation_rate': participation_rate,
            'completion_rate': completion_rate,
            'engagement_distribution': {
                'high': high_engagement,
                'medium': medium_engagement,
                'low': low_engagement
            }
        }
    
    async def _analyze_engagement_patterns(self, participant_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze engagement patterns"""
        if not participant_data:
            return {'average_engagement': 0.0, 'engagement_trend': 'stable'}
        
        # Calculate average engagement
        engagement_scores = [p.get('engagement_score', 0.5) for p in participant_data]
        average_engagement = sum(engagement_scores) / len(engagement_scores)
        
        # Analyze engagement over time
        engagement_timeline = []
        for participant in participant_data:
            timeline = participant.get('engagement_timeline', [])
            engagement_timeline.extend(timeline)
        
        # Calculate engagement trend
        if len(engagement_timeline) >= 3:
            recent_avg = sum(engagement_timeline[-3:]) / 3
            earlier_avg = sum(engagement_timeline[:-3]) / max(1, len(engagement_timeline) - 3)
            
            if recent_avg > earlier_avg + 0.1:
                trend = 'increasing'
            elif recent_avg < earlier_avg - 0.1:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        return {
            'average_engagement': average_engagement,
            'engagement_trend': trend,
            'peak_engagement_periods': await self._identify_peak_periods(engagement_timeline),
            'engagement_consistency': self._calculate_consistency(engagement_scores)
        }
    
    def _calculate_consistency(self, scores: List[float]) -> float:
        """Calculate consistency of scores"""
        if len(scores) <= 1:
            return 1.0
        
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        consistency = 1.0 / (1.0 + variance)
        
        return consistency
    
    async def _identify_peak_periods(self, timeline: List[float]) -> List[str]:
        """Identify peak engagement periods"""
        # Simple peak detection - in production, this would be more sophisticated
        peaks = []
        
        if len(timeline) >= 3:
            for i in range(1, len(timeline) - 1):
                if timeline[i] > timeline[i-1] and timeline[i] > timeline[i+1] and timeline[i] > 0.8:
                    peaks.append(f"period_{i}")
        
        return peaks
    
    async def _analyze_performance_improvements(self, participant_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance improvements"""
        improvements = []
        
        for participant in participant_data:
            initial_score = participant.get('initial_score', 0.0)
            final_score = participant.get('final_score', 0.0)
            
            if initial_score > 0:
                improvement = (final_score - initial_score) / initial_score
                improvements.append(improvement)
        
        if not improvements:
            return {'average_improvement': 0.0, 'improvement_distribution': {}}
        
        average_improvement = sum(improvements) / len(improvements)
        
        # Categorize improvements
        significant_improvement = len([i for i in improvements if i > 0.2])
        moderate_improvement = len([i for i in improvements if 0.05 <= i <= 0.2])
        minimal_improvement = len([i for i in improvements if -0.05 <= i < 0.05])
        decline = len([i for i in improvements if i < -0.05])
        
        return {
            'average_improvement': average_improvement,
            'improvement_distribution': {
                'significant_improvement': significant_improvement,
                'moderate_improvement': moderate_improvement,
                'minimal_improvement': minimal_improvement,
                'decline': decline
            },
            'top_improvers': sorted(improvements, reverse=True)[:3]
        }
    
    async def _analyze_competition_effectiveness(self,
                                               competition_data: Dict[str, Any],
                                               participation_metrics: Dict[str, Any],
                                               engagement_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall competition effectiveness"""
        # Calculate effectiveness score
        effectiveness_components = {
            'participation_rate': participation_metrics.get('participation_rate', 0.0),
            'completion_rate': participation_metrics.get('completion_rate', 0.0),
            'average_engagement': engagement_analysis.get('average_engagement', 0.0),
            'engagement_consistency': engagement_analysis.get('engagement_consistency', 0.0)
        }
        
        effectiveness_score = sum(effectiveness_components.values()) / len(effectiveness_components)
        
        # Determine effectiveness level
        if effectiveness_score >= 0.8:
            effectiveness_level = 'highly_effective'
        elif effectiveness_score >= 0.6:
            effectiveness_level = 'moderately_effective'
        elif effectiveness_score >= 0.4:
            effectiveness_level = 'somewhat_effective'
        else:
            effectiveness_level = 'needs_improvement'
        
        return {
            'effectiveness_score': effectiveness_score,
            'effectiveness_level': effectiveness_level,
            'effectiveness_components': effectiveness_components,
            'success_factors': await self._identify_success_factors(competition_data, effectiveness_components),
            'improvement_areas': await self._identify_improvement_areas(effectiveness_components)
        }
    
    async def _identify_success_factors(self,
                                      competition_data: Dict[str, Any],
                                      effectiveness_components: Dict[str, Any]) -> List[str]:
        """Identify factors contributing to competition success"""
        success_factors = []
        
        if effectiveness_components['participation_rate'] > 0.8:
            success_factors.append('high_participation_rate')
        
        if effectiveness_components['engagement_consistency'] > 0.7:
            success_factors.append('consistent_engagement')
        
        competition_type = competition_data.get('competition_type', '')
        if 'team' in competition_type and effectiveness_components['average_engagement'] > 0.7:
            success_factors.append('effective_team_dynamics')
        
        duration = competition_data.get('duration_days', 7)
        if duration <= 7 and effectiveness_components['completion_rate'] > 0.8:
            success_factors.append('optimal_duration')
        
        return success_factors
    
    async def _identify_improvement_areas(self, effectiveness_components: Dict[str, Any]) -> List[str]:
        """Identify areas for improvement"""
        improvement_areas = []
        
        if effectiveness_components['participation_rate'] < 0.6:
            improvement_areas.append('increase_participation_incentives')
        
        if effectiveness_components['completion_rate'] < 0.5:
            improvement_areas.append('reduce_competition_difficulty')
        
        if effectiveness_components['average_engagement'] < 0.6:
            improvement_areas.append('enhance_engagement_mechanics')
        
        if effectiveness_components['engagement_consistency'] < 0.5:
            improvement_areas.append('improve_sustained_motivation')
        
        return improvement_areas
    
    async def _generate_insights_recommendations(self,
                                               participation_metrics: Dict[str, Any],
                                               engagement_analysis: Dict[str, Any],
                                               performance_analysis: Dict[str, Any],
                                               effectiveness_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights and recommendations"""
        insights = []
        recommendations = []
        
        # Participation insights
        participation_rate = participation_metrics.get('participation_rate', 0.0)
        if participation_rate > 0.8:
            insights.append("Excellent participation rate indicates strong initial engagement")
        elif participation_rate < 0.5:
            insights.append("Low participation rate suggests barriers to entry or insufficient motivation")
            recommendations.append("Review entry requirements and increase initial incentives")
        
        # Engagement insights
        avg_engagement = engagement_analysis.get('average_engagement', 0.0)
        engagement_trend = engagement_analysis.get('engagement_trend', 'stable')
        
        if avg_engagement > 0.8:
            insights.append("High engagement levels indicate effective competition design")
        elif avg_engagement < 0.5:
            insights.append("Low engagement suggests need for more compelling mechanics")
            recommendations.append("Implement more interactive and rewarding elements")
        
        if engagement_trend == 'decreasing':
            insights.append("Declining engagement indicates potential fatigue or difficulty issues")
            recommendations.append("Consider mid-competition adjustments or additional motivation")
        
        # Performance insights
        avg_improvement = performance_analysis.get('average_improvement', 0.0)
        if avg_improvement > 0.15:
            insights.append("Significant performance improvements demonstrate effective learning")
        elif avg_improvement < 0.05:
            insights.append("Limited performance improvement suggests need for better learning support")
            recommendations.append("Provide more guidance and scaffolding for participants")
        
        # Effectiveness insights
        effectiveness_level = effectiveness_analysis.get('effectiveness_level', 'needs_improvement')
        if effectiveness_level == 'highly_effective':
            insights.append("Competition achieved high effectiveness across all metrics")
            recommendations.append("Consider replicating this format for future competitions")
        elif effectiveness_level == 'needs_improvement':
            insights.append("Competition effectiveness below expectations")
            recommendations.extend(effectiveness_analysis.get('improvement_areas', []))
        
        return {
            'key_insights': insights,
            'actionable_recommendations': recommendations,
            'success_metrics': {
                'participation_success': participation_rate > 0.7,
                'engagement_success': avg_engagement > 0.7,
                'learning_success': avg_improvement > 0.1,
                'overall_success': effectiveness_analysis.get('effectiveness_score', 0.0) > 0.7
            }
        }
