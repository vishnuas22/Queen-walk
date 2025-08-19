"""
Gamified Learning Pathways Services

Extracted from quantum_intelligence_engine.py (lines 12526-15023) - gamified learning
pathways, adaptive difficulty engines, learning quest systems, and progression mechanics
for immersive learning experiences.
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


class PathwayType(Enum):
    """Types of gamified learning pathways"""
    LINEAR_PROGRESSION = "linear_progression"
    BRANCHING_ADVENTURE = "branching_adventure"
    OPEN_WORLD_EXPLORATION = "open_world_exploration"
    SKILL_TREE_MASTERY = "skill_tree_mastery"
    QUEST_BASED_JOURNEY = "quest_based_journey"
    COLLABORATIVE_EXPEDITION = "collaborative_expedition"
    COMPETITIVE_RACE = "competitive_race"
    MYSTERY_INVESTIGATION = "mystery_investigation"


class QuestType(Enum):
    """Types of learning quests"""
    MAIN_QUEST = "main_quest"
    SIDE_QUEST = "side_quest"
    DAILY_QUEST = "daily_quest"
    WEEKLY_CHALLENGE = "weekly_challenge"
    EPIC_QUEST = "epic_quest"
    EXPLORATION_QUEST = "exploration_quest"
    SOCIAL_QUEST = "social_quest"
    MASTERY_QUEST = "mastery_quest"


class ProgressionMechanic(Enum):
    """Types of progression mechanics"""
    EXPERIENCE_POINTS = "experience_points"
    SKILL_LEVELS = "skill_levels"
    UNLOCK_SYSTEM = "unlock_system"
    MASTERY_BADGES = "mastery_badges"
    ACHIEVEMENT_TIERS = "achievement_tiers"
    REPUTATION_SYSTEM = "reputation_system"
    PRESTIGE_LEVELS = "prestige_levels"
    COLLECTION_SYSTEM = "collection_system"


@dataclass
class GamifiedPathway:
    """Gamified learning pathway"""
    pathway_id: str = ""
    title: str = ""
    description: str = ""
    pathway_type: PathwayType = PathwayType.LINEAR_PROGRESSION
    learning_objectives: List[str] = field(default_factory=list)
    pathway_nodes: List[Dict[str, Any]] = field(default_factory=list)
    progression_mechanics: List[ProgressionMechanic] = field(default_factory=list)
    difficulty_curve: Dict[str, Any] = field(default_factory=dict)
    estimated_duration_hours: int = 20
    prerequisite_skills: List[str] = field(default_factory=list)
    reward_structure: Dict[str, Any] = field(default_factory=dict)
    narrative_theme: str = ""
    personalization_factors: List[str] = field(default_factory=list)
    created_at: str = ""
    is_adaptive: bool = True


@dataclass
class LearningQuest:
    """Learning quest within a pathway"""
    quest_id: str = ""
    title: str = ""
    description: str = ""
    quest_type: QuestType = QuestType.MAIN_QUEST
    learning_objectives: List[str] = field(default_factory=list)
    quest_steps: List[Dict[str, Any]] = field(default_factory=list)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    reward_points: int = 100
    estimated_duration_minutes: int = 60
    difficulty_level: float = 0.5
    prerequisite_quests: List[str] = field(default_factory=list)
    narrative_context: str = ""
    is_repeatable: bool = False
    expires_at: Optional[str] = None
    created_at: str = ""


@dataclass
class ProgressionState:
    """User's progression state in pathway"""
    user_id: str = ""
    pathway_id: str = ""
    current_node: str = ""
    completed_nodes: List[str] = field(default_factory=list)
    active_quests: List[str] = field(default_factory=list)
    completed_quests: List[str] = field(default_factory=list)
    experience_points: int = 0
    skill_levels: Dict[str, int] = field(default_factory=dict)
    unlocked_content: List[str] = field(default_factory=list)
    achievements_earned: List[str] = field(default_factory=list)
    progression_percentage: float = 0.0
    last_activity: str = ""
    pathway_started_at: str = ""


class GamifiedLearningPathways:
    """
    🗺️ GAMIFIED LEARNING PATHWAYS
    
    Advanced system for creating immersive, gamified learning pathways.
    Extracted from the original quantum engine's gamification logic.
    """
    
    def __init__(self, cache_service: Optional[CacheService] = None):
        self.cache = cache_service
        
        # Pathway configuration
        self.config = {
            'pathway_templates': {
                PathwayType.LINEAR_PROGRESSION: {
                    'structure': 'sequential',
                    'branching_factor': 1,
                    'exploration_freedom': 0.2
                },
                PathwayType.BRANCHING_ADVENTURE: {
                    'structure': 'tree',
                    'branching_factor': 3,
                    'exploration_freedom': 0.6
                },
                PathwayType.OPEN_WORLD_EXPLORATION: {
                    'structure': 'network',
                    'branching_factor': 5,
                    'exploration_freedom': 0.9
                },
                PathwayType.SKILL_TREE_MASTERY: {
                    'structure': 'hierarchical',
                    'branching_factor': 2,
                    'exploration_freedom': 0.4
                }
            },
            'narrative_themes': [
                'space_exploration', 'medieval_adventure', 'detective_mystery',
                'scientific_discovery', 'artistic_journey', 'entrepreneurial_quest',
                'historical_expedition', 'futuristic_mission'
            ],
            'difficulty_progression_rate': 0.15,
            'adaptive_adjustment_threshold': 0.3
        }
        
        # Pathway tracking
        self.created_pathways = {}
        self.user_progressions = {}
        self.pathway_analytics = {}
        
        logger.info("Gamified Learning Pathways initialized")
    
    async def create_gamified_pathway(self,
                                    learning_objectives: List[str],
                                    user_profile: Dict[str, Any],
                                    pathway_preferences: Dict[str, Any],
                                    content_library: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create personalized gamified learning pathway
        
        Args:
            learning_objectives: List of learning objectives to achieve
            user_profile: User's profile and preferences
            pathway_preferences: Specific pathway preferences
            content_library: Available learning content and resources
            
        Returns:
            Dict with created gamified pathway
        """
        try:
            # Determine optimal pathway type
            pathway_type = await self._determine_pathway_type(user_profile, pathway_preferences)
            
            # Select narrative theme
            narrative_theme = await self._select_narrative_theme(user_profile, pathway_preferences)
            
            # Design pathway structure
            pathway_structure = await self._design_pathway_structure(
                pathway_type, learning_objectives, content_library
            )
            
            # Create progression mechanics
            progression_mechanics = await self._create_progression_mechanics(
                pathway_type, user_profile, learning_objectives
            )
            
            # Design difficulty curve
            difficulty_curve = await self._design_difficulty_curve(
                pathway_structure, user_profile
            )
            
            # Create reward structure
            reward_structure = await self._create_reward_structure(
                progression_mechanics, learning_objectives
            )
            
            # Generate pathway content
            pathway_content = await self._generate_pathway_content(
                pathway_structure, narrative_theme, learning_objectives
            )
            
            # Create gamified pathway
            pathway_id = f"pathway_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{user_profile.get('user_id', 'unknown')}"
            
            gamified_pathway = GamifiedPathway(
                pathway_id=pathway_id,
                title=pathway_content['title'],
                description=pathway_content['description'],
                pathway_type=pathway_type,
                learning_objectives=learning_objectives,
                pathway_nodes=pathway_structure['nodes'],
                progression_mechanics=progression_mechanics,
                difficulty_curve=difficulty_curve,
                estimated_duration_hours=pathway_structure['estimated_duration'],
                prerequisite_skills=pathway_content['prerequisites'],
                reward_structure=reward_structure,
                narrative_theme=narrative_theme,
                personalization_factors=await self._identify_personalization_factors(user_profile),
                created_at=datetime.utcnow().isoformat(),
                is_adaptive=True
            )
            
            # Store pathway
            self.created_pathways[pathway_id] = gamified_pathway
            
            # Initialize user progression
            await self._initialize_user_progression(user_profile.get('user_id'), pathway_id)
            
            return {
                'status': 'success',
                'gamified_pathway': gamified_pathway.__dict__,
                'pathway_preview': {
                    'total_nodes': len(pathway_structure['nodes']),
                    'estimated_completion': f"{pathway_structure['estimated_duration']} hours",
                    'difficulty_range': f"{difficulty_curve['min_difficulty']:.1f} - {difficulty_curve['max_difficulty']:.1f}",
                    'progression_types': [m.value for m in progression_mechanics]
                },
                'creation_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating gamified pathway: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _determine_pathway_type(self,
                                    user_profile: Dict[str, Any],
                                    pathway_preferences: Dict[str, Any]) -> PathwayType:
        """Determine optimal pathway type for user"""
        # Check explicit preferences
        preferred_type = pathway_preferences.get('pathway_type')
        if preferred_type:
            try:
                return PathwayType(preferred_type)
            except ValueError:
                pass
        
        # Determine based on user characteristics
        personality = user_profile.get('personality', {})
        learning_style = user_profile.get('learning_style', {})
        preferences = user_profile.get('preferences', {})
        
        type_scores = {
            PathwayType.LINEAR_PROGRESSION: 0.3,
            PathwayType.BRANCHING_ADVENTURE: 0.4,
            PathwayType.OPEN_WORLD_EXPLORATION: 0.2,
            PathwayType.SKILL_TREE_MASTERY: 0.3,
            PathwayType.QUEST_BASED_JOURNEY: 0.4,
            PathwayType.COLLABORATIVE_EXPEDITION: 0.2,
            PathwayType.COMPETITIVE_RACE: 0.2,
            PathwayType.MYSTERY_INVESTIGATION: 0.3
        }
        
        # Adjust based on personality
        if personality.get('openness', 0.5) > 0.7:
            type_scores[PathwayType.OPEN_WORLD_EXPLORATION] += 0.3
            type_scores[PathwayType.BRANCHING_ADVENTURE] += 0.2
        
        if personality.get('conscientiousness', 0.5) > 0.7:
            type_scores[PathwayType.LINEAR_PROGRESSION] += 0.3
            type_scores[PathwayType.SKILL_TREE_MASTERY] += 0.2
        
        if personality.get('extraversion', 0.5) > 0.6:
            type_scores[PathwayType.COLLABORATIVE_EXPEDITION] += 0.4
            type_scores[PathwayType.COMPETITIVE_RACE] += 0.3
        
        # Adjust based on learning preferences
        if preferences.get('structure_preference', 0.5) > 0.7:
            type_scores[PathwayType.LINEAR_PROGRESSION] += 0.2
            type_scores[PathwayType.SKILL_TREE_MASTERY] += 0.2
        
        if preferences.get('exploration_preference', 0.5) > 0.7:
            type_scores[PathwayType.OPEN_WORLD_EXPLORATION] += 0.3
            type_scores[PathwayType.MYSTERY_INVESTIGATION] += 0.2
        
        if preferences.get('gamification_preference', 0.5) > 0.7:
            type_scores[PathwayType.QUEST_BASED_JOURNEY] += 0.3
        
        # Select highest scoring type
        best_type = max(type_scores.items(), key=lambda x: x[1])
        return best_type[0]
    
    async def _select_narrative_theme(self,
                                    user_profile: Dict[str, Any],
                                    pathway_preferences: Dict[str, Any]) -> str:
        """Select appropriate narrative theme"""
        # Check explicit preferences
        preferred_theme = pathway_preferences.get('narrative_theme')
        if preferred_theme and preferred_theme in self.config['narrative_themes']:
            return preferred_theme
        
        # Select based on interests and personality
        interests = user_profile.get('interests', [])
        personality = user_profile.get('personality', {})
        
        theme_scores = {theme: 0.0 for theme in self.config['narrative_themes']}
        
        # Interest-based scoring
        if 'science' in interests or 'technology' in interests:
            theme_scores['space_exploration'] += 0.3
            theme_scores['scientific_discovery'] += 0.4
            theme_scores['futuristic_mission'] += 0.3
        
        if 'history' in interests:
            theme_scores['medieval_adventure'] += 0.3
            theme_scores['historical_expedition'] += 0.4
        
        if 'art' in interests or 'creativity' in interests:
            theme_scores['artistic_journey'] += 0.4
        
        if 'business' in interests or 'entrepreneurship' in interests:
            theme_scores['entrepreneurial_quest'] += 0.4
        
        # Personality-based scoring
        if personality.get('openness', 0.5) > 0.7:
            theme_scores['space_exploration'] += 0.2
            theme_scores['artistic_journey'] += 0.2
        
        if personality.get('conscientiousness', 0.5) > 0.7:
            theme_scores['detective_mystery'] += 0.2
            theme_scores['scientific_discovery'] += 0.2
        
        # Select highest scoring theme
        best_theme = max(theme_scores.items(), key=lambda x: x[1])
        return best_theme[0] if best_theme[1] > 0.2 else random.choice(self.config['narrative_themes'])
    
    async def _design_pathway_structure(self,
                                      pathway_type: PathwayType,
                                      learning_objectives: List[str],
                                      content_library: Dict[str, Any]) -> Dict[str, Any]:
        """Design the structure of the pathway"""
        template = self.config['pathway_templates'][pathway_type]
        
        # Calculate number of nodes based on objectives
        base_nodes = len(learning_objectives) * 3  # 3 nodes per objective on average
        node_count = max(5, min(20, base_nodes))  # Between 5-20 nodes
        
        # Create pathway nodes
        nodes = []
        for i in range(node_count):
            node = {
                'node_id': f"node_{i+1:02d}",
                'title': f"Learning Stage {i+1}",
                'node_type': self._determine_node_type(i, node_count, pathway_type),
                'learning_content': self._assign_learning_content(i, learning_objectives, content_library),
                'connections': self._create_node_connections(i, node_count, template),
                'unlock_requirements': self._create_unlock_requirements(i, pathway_type),
                'rewards': self._create_node_rewards(i, node_count),
                'estimated_duration_minutes': random.randint(30, 90)
            }
            nodes.append(node)
        
        # Calculate total estimated duration
        total_duration = sum(node['estimated_duration_minutes'] for node in nodes) // 60
        
        return {
            'nodes': nodes,
            'structure_type': template['structure'],
            'branching_factor': template['branching_factor'],
            'estimated_duration': total_duration,
            'total_nodes': len(nodes)
        }
    
    def _determine_node_type(self, index: int, total_nodes: int, pathway_type: PathwayType) -> str:
        """Determine the type of a pathway node"""
        progress_ratio = index / (total_nodes - 1) if total_nodes > 1 else 0
        
        if index == 0:
            return 'introduction'
        elif index == total_nodes - 1:
            return 'capstone'
        elif progress_ratio < 0.3:
            return 'foundation'
        elif progress_ratio < 0.7:
            return 'development'
        else:
            return 'mastery'
    
    def _assign_learning_content(self,
                               index: int,
                               learning_objectives: List[str],
                               content_library: Dict[str, Any]) -> Dict[str, Any]:
        """Assign learning content to a node"""
        # Simple content assignment - in production, this would be more sophisticated
        objective_index = index % len(learning_objectives) if learning_objectives else 0
        current_objective = learning_objectives[objective_index] if learning_objectives else "General Learning"
        
        return {
            'primary_objective': current_objective,
            'content_types': ['video', 'reading', 'exercise', 'quiz'],
            'estimated_difficulty': 0.3 + (index / 20) * 0.7,  # Progressive difficulty
            'content_ids': [f"content_{index}_{i}" for i in range(3)]  # Mock content IDs
        }
    
    def _create_node_connections(self,
                               index: int,
                               total_nodes: int,
                               template: Dict[str, Any]) -> List[str]:
        """Create connections between nodes"""
        connections = []
        
        if template['structure'] == 'sequential':
            # Linear progression
            if index < total_nodes - 1:
                connections.append(f"node_{index+2:02d}")
        
        elif template['structure'] == 'tree':
            # Branching structure
            branching_factor = template['branching_factor']
            for i in range(1, branching_factor + 1):
                next_index = index + i
                if next_index < total_nodes:
                    connections.append(f"node_{next_index+1:02d}")
        
        elif template['structure'] == 'network':
            # More complex network structure
            for i in range(1, 4):  # Connect to next 3 nodes
                next_index = index + i
                if next_index < total_nodes:
                    connections.append(f"node_{next_index+1:02d}")
        
        return connections
    
    def _create_unlock_requirements(self, index: int, pathway_type: PathwayType) -> Dict[str, Any]:
        """Create unlock requirements for a node"""
        if index == 0:
            return {'type': 'none'}
        
        requirements = {
            'type': 'completion',
            'required_nodes': [f"node_{index:02d}"],
            'minimum_score': 0.7
        }
        
        # Add additional requirements for certain pathway types
        if pathway_type == PathwayType.SKILL_TREE_MASTERY and index > 3:
            requirements['skill_level_required'] = {
                'skill': 'foundation_skills',
                'level': index // 3
            }
        
        return requirements
    
    def _create_node_rewards(self, index: int, total_nodes: int) -> Dict[str, Any]:
        """Create rewards for completing a node"""
        base_points = 100 + (index * 20)  # Progressive point increase
        
        rewards = {
            'experience_points': base_points,
            'skill_points': index // 2 + 1,
            'unlocks': []
        }
        
        # Special rewards for milestone nodes
        if index % 5 == 4:  # Every 5th node
            rewards['special_badge'] = f"milestone_{index//5 + 1}"
            rewards['bonus_points'] = base_points // 2
        
        # Final node rewards
        if index == total_nodes - 1:
            rewards['completion_certificate'] = True
            rewards['mastery_badge'] = True
            rewards['bonus_points'] = base_points
        
        return rewards
    
    async def _create_progression_mechanics(self,
                                          pathway_type: PathwayType,
                                          user_profile: Dict[str, Any],
                                          learning_objectives: List[str]) -> List[ProgressionMechanic]:
        """Create progression mechanics for the pathway"""
        mechanics = [ProgressionMechanic.EXPERIENCE_POINTS]  # Always include XP
        
        # Add mechanics based on pathway type
        if pathway_type == PathwayType.SKILL_TREE_MASTERY:
            mechanics.extend([ProgressionMechanic.SKILL_LEVELS, ProgressionMechanic.UNLOCK_SYSTEM])
        
        elif pathway_type == PathwayType.QUEST_BASED_JOURNEY:
            mechanics.extend([ProgressionMechanic.ACHIEVEMENT_TIERS, ProgressionMechanic.COLLECTION_SYSTEM])
        
        elif pathway_type == PathwayType.COMPETITIVE_RACE:
            mechanics.extend([ProgressionMechanic.REPUTATION_SYSTEM, ProgressionMechanic.PRESTIGE_LEVELS])
        
        else:
            mechanics.extend([ProgressionMechanic.MASTERY_BADGES, ProgressionMechanic.UNLOCK_SYSTEM])
        
        # Add mechanics based on user preferences
        personality = user_profile.get('personality', {})
        
        if personality.get('conscientiousness', 0.5) > 0.7:
            if ProgressionMechanic.MASTERY_BADGES not in mechanics:
                mechanics.append(ProgressionMechanic.MASTERY_BADGES)
        
        if personality.get('extraversion', 0.5) > 0.6:
            if ProgressionMechanic.REPUTATION_SYSTEM not in mechanics:
                mechanics.append(ProgressionMechanic.REPUTATION_SYSTEM)
        
        return mechanics[:4]  # Limit to 4 mechanics to avoid complexity
    
    async def _design_difficulty_curve(self,
                                     pathway_structure: Dict[str, Any],
                                     user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Design the difficulty curve for the pathway"""
        nodes = pathway_structure['nodes']
        total_nodes = len(nodes)
        
        # Base difficulty progression
        progression_rate = self.config['difficulty_progression_rate']
        
        # Adjust based on user skill level
        user_skill_level = sum(user_profile.get('skills', {}).values()) / max(1, len(user_profile.get('skills', {})))
        base_difficulty = max(0.2, min(0.8, user_skill_level))
        
        # Create difficulty curve
        difficulties = []
        for i, node in enumerate(nodes):
            progress_ratio = i / (total_nodes - 1) if total_nodes > 1 else 0
            
            # Exponential difficulty increase with plateaus
            if progress_ratio < 0.3:
                # Gentle start
                difficulty = base_difficulty + (progress_ratio * 0.3)
            elif progress_ratio < 0.7:
                # Steady increase
                difficulty = base_difficulty + 0.1 + ((progress_ratio - 0.3) * 0.4)
            else:
                # Final challenge
                difficulty = base_difficulty + 0.3 + ((progress_ratio - 0.7) * 0.3)
            
            difficulty = max(0.1, min(1.0, difficulty))
            difficulties.append(difficulty)
            
            # Update node with difficulty
            node['difficulty'] = difficulty
        
        return {
            'difficulties': difficulties,
            'min_difficulty': min(difficulties),
            'max_difficulty': max(difficulties),
            'progression_type': 'exponential_with_plateaus',
            'adaptive_enabled': True
        }
    
    async def _create_reward_structure(self,
                                     progression_mechanics: List[ProgressionMechanic],
                                     learning_objectives: List[str]) -> Dict[str, Any]:
        """Create reward structure for the pathway"""
        reward_structure = {
            'point_system': {
                'base_points_per_node': 100,
                'bonus_multipliers': {
                    'perfect_score': 1.5,
                    'first_attempt': 1.2,
                    'speed_bonus': 1.3,
                    'help_others': 1.1
                }
            },
            'milestone_rewards': {},
            'completion_rewards': {}
        }
        
        # Add mechanic-specific rewards
        for mechanic in progression_mechanics:
            if mechanic == ProgressionMechanic.SKILL_LEVELS:
                reward_structure['skill_progression'] = {
                    'points_per_level': 50,
                    'max_level': 10,
                    'level_up_bonus': 200
                }
            
            elif mechanic == ProgressionMechanic.MASTERY_BADGES:
                reward_structure['badge_system'] = {
                    'badges_per_objective': len(learning_objectives),
                    'badge_points': 300,
                    'collection_bonus': 500
                }
            
            elif mechanic == ProgressionMechanic.ACHIEVEMENT_TIERS:
                reward_structure['achievement_tiers'] = {
                    'bronze': {'threshold': 0.6, 'points': 100},
                    'silver': {'threshold': 0.8, 'points': 250},
                    'gold': {'threshold': 0.95, 'points': 500}
                }
        
        # Milestone rewards (every 25% completion)
        for milestone in [25, 50, 75, 100]:
            reward_structure['milestone_rewards'][f"{milestone}%"] = {
                'points': milestone * 10,
                'special_unlock': f"milestone_{milestone}",
                'badge': f"progress_{milestone}"
            }
        
        # Completion rewards
        reward_structure['completion_rewards'] = {
            'certificate': True,
            'final_badge': True,
            'bonus_points': 1000,
            'pathway_mastery_unlock': True
        }
        
        return reward_structure
    
    async def _generate_pathway_content(self,
                                      pathway_structure: Dict[str, Any],
                                      narrative_theme: str,
                                      learning_objectives: List[str]) -> Dict[str, Any]:
        """Generate content for the pathway"""
        # Theme-based content generation
        theme_content = {
            'space_exploration': {
                'title_template': "Mission to Master {objectives}",
                'description_template': "Embark on an interstellar journey to master {objectives}. Navigate through cosmic challenges and unlock the secrets of the universe.",
                'node_prefix': "Sector"
            },
            'medieval_adventure': {
                'title_template': "Quest for {objectives} Mastery",
                'description_template': "Join a noble quest to master {objectives}. Face challenges worthy of legends and earn your place among the learned.",
                'node_prefix': "Chapter"
            },
            'detective_mystery': {
                'title_template': "The {objectives} Investigation",
                'description_template': "Solve the mystery of {objectives} through careful investigation and deductive reasoning. Each clue brings you closer to mastery.",
                'node_prefix': "Case"
            },
            'scientific_discovery': {
                'title_template': "Discovering {objectives}",
                'description_template': "Conduct groundbreaking research to understand {objectives}. Make discoveries that will advance human knowledge.",
                'node_prefix': "Experiment"
            }
        }
        
        theme_info = theme_content.get(narrative_theme, theme_content['scientific_discovery'])
        objectives_text = ', '.join(learning_objectives[:3]) if learning_objectives else "Advanced Skills"
        
        title = theme_info['title_template'].format(objectives=objectives_text)
        description = theme_info['description_template'].format(objectives=objectives_text)
        
        # Update node titles with theme
        for i, node in enumerate(pathway_structure['nodes']):
            node['title'] = f"{theme_info['node_prefix']} {i+1}: {node['title']}"
        
        # Determine prerequisites
        prerequisites = []
        if learning_objectives:
            # Simple prerequisite determination
            for objective in learning_objectives:
                if 'advanced' in objective.lower():
                    prerequisites.append('intermediate_skills')
                elif 'intermediate' in objective.lower():
                    prerequisites.append('basic_skills')
        
        return {
            'title': title,
            'description': description,
            'prerequisites': prerequisites,
            'narrative_elements': {
                'theme': narrative_theme,
                'story_arc': 'hero_journey',
                'character_progression': True
            }
        }
    
    async def _identify_personalization_factors(self, user_profile: Dict[str, Any]) -> List[str]:
        """Identify personalization factors for the pathway"""
        factors = ['user_skill_level', 'learning_objectives']
        
        # Add factors based on available profile data
        if user_profile.get('learning_style'):
            factors.append('learning_style')
        
        if user_profile.get('personality'):
            factors.append('personality_traits')
        
        if user_profile.get('interests'):
            factors.append('interests')
        
        if user_profile.get('preferences'):
            factors.append('user_preferences')
        
        if user_profile.get('goals'):
            factors.append('learning_goals')
        
        return factors
    
    async def _initialize_user_progression(self, user_id: str, pathway_id: str):
        """Initialize user's progression state for the pathway"""
        if not user_id:
            return
        
        progression_state = ProgressionState(
            user_id=user_id,
            pathway_id=pathway_id,
            current_node="node_01",
            pathway_started_at=datetime.utcnow().isoformat()
        )
        
        if user_id not in self.user_progressions:
            self.user_progressions[user_id] = {}
        
        self.user_progressions[user_id][pathway_id] = progression_state


class AdaptiveDifficultyEngine:
    """
    ⚖️ ADAPTIVE DIFFICULTY ENGINE
    
    Advanced engine for dynamically adjusting pathway difficulty.
    """
    
    def __init__(self):
        # Engine configuration
        self.config = {
            'adaptation_sensitivity': 0.2,
            'performance_window': 5,  # Number of recent activities to consider
            'difficulty_bounds': {'min': 0.1, 'max': 1.0},
            'adaptation_frequency': 'per_node',
            'performance_thresholds': {
                'too_easy': 0.9,
                'optimal': 0.7,
                'too_hard': 0.4
            }
        }
        
        # Adaptation tracking
        self.adaptation_history = {}
        
        logger.info("Adaptive Difficulty Engine initialized")
    
    async def adapt_pathway_difficulty(self,
                                     user_id: str,
                                     pathway_id: str,
                                     performance_data: Dict[str, Any],
                                     current_difficulty: float) -> Dict[str, Any]:
        """
        Adapt pathway difficulty based on user performance
        
        Args:
            user_id: User identifier
            pathway_id: Pathway identifier
            performance_data: Recent performance data
            current_difficulty: Current difficulty level
            
        Returns:
            Dict with adapted difficulty and rationale
        """
        try:
            # Analyze recent performance
            performance_analysis = await self._analyze_performance(performance_data)
            
            # Calculate difficulty adjustment
            difficulty_adjustment = await self._calculate_difficulty_adjustment(
                performance_analysis, current_difficulty
            )
            
            # Apply adaptation bounds
            new_difficulty = await self._apply_adaptation_bounds(
                current_difficulty, difficulty_adjustment
            )
            
            # Generate adaptation rationale
            adaptation_rationale = await self._generate_adaptation_rationale(
                performance_analysis, difficulty_adjustment, new_difficulty
            )
            
            # Track adaptation
            await self._track_adaptation(user_id, pathway_id, current_difficulty, new_difficulty, performance_analysis)
            
            return {
                'status': 'success',
                'original_difficulty': current_difficulty,
                'adapted_difficulty': new_difficulty,
                'difficulty_change': new_difficulty - current_difficulty,
                'performance_analysis': performance_analysis,
                'adaptation_rationale': adaptation_rationale,
                'adaptation_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error adapting pathway difficulty: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _analyze_performance(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user performance patterns"""
        recent_scores = performance_data.get('recent_scores', [])
        completion_times = performance_data.get('completion_times', [])
        attempt_counts = performance_data.get('attempt_counts', [])
        
        if not recent_scores:
            return {'average_score': 0.7, 'performance_trend': 'stable', 'confidence': 0.3}
        
        # Calculate performance metrics
        average_score = sum(recent_scores) / len(recent_scores)
        
        # Calculate trend
        if len(recent_scores) >= 3:
            recent_avg = sum(recent_scores[-3:]) / 3
            earlier_avg = sum(recent_scores[:-3]) / max(1, len(recent_scores) - 3)
            
            if recent_avg > earlier_avg + 0.1:
                trend = 'improving'
            elif recent_avg < earlier_avg - 0.1:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        # Calculate consistency
        if len(recent_scores) > 1:
            score_variance = sum((score - average_score) ** 2 for score in recent_scores) / len(recent_scores)
            consistency = 1.0 - min(1.0, score_variance)
        else:
            consistency = 0.5
        
        # Analyze completion efficiency
        if completion_times:
            avg_completion_time = sum(completion_times) / len(completion_times)
            efficiency = 1.0 / (1.0 + avg_completion_time / 60)  # Normalize by hour
        else:
            efficiency = 0.5
        
        # Analyze struggle indicators
        if attempt_counts:
            avg_attempts = sum(attempt_counts) / len(attempt_counts)
            struggle_indicator = min(1.0, (avg_attempts - 1) / 3)  # 0-1 scale
        else:
            struggle_indicator = 0.0
        
        return {
            'average_score': average_score,
            'performance_trend': trend,
            'consistency': consistency,
            'efficiency': efficiency,
            'struggle_indicator': struggle_indicator,
            'confidence': min(1.0, len(recent_scores) / self.config['performance_window'])
        }
    
    async def _calculate_difficulty_adjustment(self,
                                             performance_analysis: Dict[str, Any],
                                             current_difficulty: float) -> float:
        """Calculate how much to adjust difficulty"""
        average_score = performance_analysis['average_score']
        trend = performance_analysis['performance_trend']
        struggle_indicator = performance_analysis['struggle_indicator']
        confidence = performance_analysis['confidence']
        
        # Base adjustment on performance vs thresholds
        if average_score > self.config['performance_thresholds']['too_easy']:
            base_adjustment = 0.1  # Increase difficulty
        elif average_score < self.config['performance_thresholds']['too_hard']:
            base_adjustment = -0.15  # Decrease difficulty
        else:
            base_adjustment = 0.0  # No change needed
        
        # Adjust based on trend
        if trend == 'improving':
            base_adjustment += 0.05
        elif trend == 'declining':
            base_adjustment -= 0.05
        
        # Adjust based on struggle
        if struggle_indicator > 0.5:
            base_adjustment -= 0.1
        
        # Scale by confidence
        final_adjustment = base_adjustment * confidence * self.config['adaptation_sensitivity']
        
        return final_adjustment
    
    async def _apply_adaptation_bounds(self,
                                     current_difficulty: float,
                                     adjustment: float) -> float:
        """Apply bounds to difficulty adaptation"""
        new_difficulty = current_difficulty + adjustment
        
        # Apply absolute bounds
        new_difficulty = max(self.config['difficulty_bounds']['min'], new_difficulty)
        new_difficulty = min(self.config['difficulty_bounds']['max'], new_difficulty)
        
        # Limit maximum change per adaptation
        max_change = 0.2
        if abs(new_difficulty - current_difficulty) > max_change:
            if new_difficulty > current_difficulty:
                new_difficulty = current_difficulty + max_change
            else:
                new_difficulty = current_difficulty - max_change
        
        return new_difficulty
    
    async def _generate_adaptation_rationale(self,
                                           performance_analysis: Dict[str, Any],
                                           adjustment: float,
                                           new_difficulty: float) -> str:
        """Generate human-readable rationale for adaptation"""
        average_score = performance_analysis['average_score']
        trend = performance_analysis['performance_trend']
        
        if abs(adjustment) < 0.01:
            return "Difficulty maintained - performance is in optimal range"
        
        elif adjustment > 0:
            reasons = []
            if average_score > 0.9:
                reasons.append("high success rate indicates content may be too easy")
            if trend == 'improving':
                reasons.append("improving performance trend suggests readiness for more challenge")
            
            return f"Difficulty increased to {new_difficulty:.2f} because " + " and ".join(reasons)
        
        else:
            reasons = []
            if average_score < 0.4:
                reasons.append("low success rate indicates content may be too difficult")
            if trend == 'declining':
                reasons.append("declining performance suggests need for easier content")
            if performance_analysis.get('struggle_indicator', 0) > 0.5:
                reasons.append("multiple attempts indicate struggle with current difficulty")
            
            return f"Difficulty decreased to {new_difficulty:.2f} because " + " and ".join(reasons)
    
    async def _track_adaptation(self,
                              user_id: str,
                              pathway_id: str,
                              old_difficulty: float,
                              new_difficulty: float,
                              performance_analysis: Dict[str, Any]):
        """Track difficulty adaptations for analysis"""
        adaptation_key = f"{user_id}_{pathway_id}"
        
        if adaptation_key not in self.adaptation_history:
            self.adaptation_history[adaptation_key] = []
        
        self.adaptation_history[adaptation_key].append({
            'timestamp': datetime.utcnow().isoformat(),
            'old_difficulty': old_difficulty,
            'new_difficulty': new_difficulty,
            'performance_score': performance_analysis['average_score'],
            'performance_trend': performance_analysis['performance_trend'],
            'adaptation_magnitude': abs(new_difficulty - old_difficulty)
        })
        
        # Keep only recent history
        if len(self.adaptation_history[adaptation_key]) > 50:
            self.adaptation_history[adaptation_key] = self.adaptation_history[adaptation_key][-50:]
